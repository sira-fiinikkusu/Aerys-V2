"""Her durable skills, owner write gate, and deployment/audit CLI."""
from __future__ import annotations

import argparse
import atexit
from functools import lru_cache
import json
from pathlib import Path
import re
import threading
from types import SimpleNamespace

import yaml
from deepagents.backends import CompositeBackend, StoreBackend
from deepagents.backends.utils import perform_string_replacement
from deepagents.middleware.filesystem import FilesystemMiddleware, FilesystemPermission
from deepagents.middleware.skills import SkillsMiddleware
from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import ToolMessage

from aerys_v2.rooms import his_private_read, runtime_identity
from aerys_v2.config import db_url

NAMESPACE = ("aerys", "skills")
WRITE_TOOLS = frozenset({"write_file", "edit_file"})
REFUSAL = "Not saved: I only write my skills when it's you, in a room that's just us."
OUTSIDE = "Not saved: my skills live only under /skills/<name>/."
#: our own fence, on every write and every approved write: upstream's "/**" deny rule misses dot-files
#: ("/home/x/.bashrc" got through as a store key; Gemini review 11/14)
SKILL_FILE = re.compile(r"^/skills/[a-z0-9]+(?:-[a-z0-9]+)*/(?:[A-Za-z0-9][A-Za-z0-9_-]*/)*[A-Za-z0-9][A-Za-z0-9._-]*$")
FILES_PROMPT = """I may write and improve my own skills at /skills/<slug>/SKILL.md, with YAML
name: and description: front matter. A description is one plain line: never ids,
numbers about him or other personal values. Those belong in the body, read on demand."""
SKILLS_PROMPT = """My skills are reusable ways of doing things. When a task matches, I read the
full skill with read_file before using it.
{skills_locations}
{skills_list}{skills_load_warnings}"""


@lru_cache(maxsize=None)
def _postgres_store(database_url, timeout):
    from langgraph.store.postgres import PostgresStore
    from psycopg.rows import dict_row
    from psycopg_pool import ConnectionPool

    pool = ConnectionPool(
        conninfo=database_url, min_size=1, max_size=1,
        check=ConnectionPool.check_connection,
        kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row},
        open=False,
    )
    try:
        pool.open(wait=True, timeout=timeout)
    except Exception:
        pool.close()
        raise
    atexit.register(pool.close)
    return PostgresStore(pool)


def store_for(settings):
    """One long-lived, reconnecting connection per process; never migrate at boot."""
    if not settings.database_url:
        raise ValueError("Skills require a database.")
    return _postgres_store(db_url(settings.database_url), settings.checkpoint_pool_open_timeout_s)


def backend_for(store):
    return StoreBackend(namespace=lambda rt: NAMESPACE, store=store)


class FreshSkillsMiddleware(SkillsMiddleware):
    def before_agent(self, state, runtime, config):
        return super().before_agent({**state, "skills_metadata": None}, runtime, config)

    async def abefore_agent(self, state, runtime, config):
        return await super().abefore_agent({**state, "skills_metadata": None}, runtime, config)


#: one skill write at a time in this process: her parallel tool calls and the approved replay share it.
#: Two containers writing the same skill in the same instant would still race; that is not a case she has.
_WRITE_LOCK = threading.Lock()


class OwnerSkillWrites(AgentMiddleware):
    def __init__(self, owner, backend):
        self.owner = owner
        self.backend = backend

    def refusal(self, request):
        if request.tool_call['name'] not in WRITE_TOOLS:
            return None
        name, call_id = request.tool_call['name'], request.tool_call['id']
        identity, thread = runtime_identity(request.runtime)
        if his_private_read(identity, thread, self.owner) is not None:
            return ToolMessage(content=REFUSAL, name=name, tool_call_id=call_id, status='error')
        path = str((request.tool_call.get('args') or {}).get('file_path') or '')
        if path.startswith(BODY_SKILLS):
            return ToolMessage(content=HOME_SKILL_REFUSAL, name=name, tool_call_id=call_id, status='error')
        if not writable_skill_path(path):
            return ToolMessage(content=OUTSIDE, name=name, tool_call_id=call_id, status='error')
        if path.endswith('/SKILL.md'):
            args = request.tool_call.get('args') or {}
            try:
                content = args.get('content', '')
                if name == 'edit_file':
                    current = self.backend.download_files([path])[0]
                    if current.error:
                        raise ValueError('Skill file could not be read; nothing changed.')
                    edited = perform_string_replacement(
                        current.content.decode('utf-8'), args.get('old_string', ''),
                        args.get('new_string', ''), args.get('replace_all', False))
                    if isinstance(edited, str):
                        raise ValueError(edited)
                    content = edited[0]
                validate_skill(path.split('/')[-2], content)
            except (ValueError, UnicodeError) as exc:
                return ToolMessage(content=f'Not saved: {exc}', name=name,
                                   tool_call_id=call_id, status='error')
        return None

    def wrap_tool_call(self, request, handler):
        if request.tool_call['name'] not in WRITE_TOOLS:
            return handler(request)
        # Check and write as one step: two edits validated against the same snapshot could
        # otherwise combine into a file neither would pass (Codex 11/14 second pass).
        with _WRITE_LOCK:
            refused = self.refusal(request)
            return refused if refused is not None else handler(request)

    async def awrap_tool_call(self, request, handler):
        if request.tool_call['name'] not in WRITE_TOOLS:
            return await handler(request)
        import asyncio

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:                      # driven without a loop (tests): nothing else can run
            loop = None
        if loop is None:
            _WRITE_LOCK.acquire()
        else:                                     # never block the loop: another write may hold the lock
            await loop.run_in_executor(None, _WRITE_LOCK.acquire)
        try:
            refused = self.refusal(request)
            return refused if refused is not None else await handler(request)
        finally:
            _WRITE_LOCK.release()


#: D67 (Chris 11/18 12:22, "yes please"): her bodies' skills, as they reach the house through the door, on
#: a read-only shelf beside her own. 12:19 her brain said it had no MapTap skill (its list held one) while her
#: home body had written a fuller one at 11:29: one identity, two shelves.
BODY_SKILLS = "/home-skills/"
_BODY_NAMESPACE = ("aerys", "body-skills")
_BODY_SLUG = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
#: Every skill event in the door's feed, in feed order. A revision arrives as an insert of the new version
#: and THEN a tombstone of the old one under the same label (live 11/18), so liveness is per memory, not
#: per label: rows_from_feed works it out.
FEED_SKILLS_SQL = """SELECT rack_seq, memory_id, key_label, event_type, content, context, source, observed_at
    FROM portable_memory_feed WHERE key_label LIKE 'skill.%%' ORDER BY rack_seq"""
_READ_ONLY = "Not saved: skills from my bodies are written there, not here; this shelf only reads them."
#: 11/19 14:50 (Chris, the MapTap share rule): her brain asked him to approve an edit to a home skill, he
#: tapped Approve, and the edit was refused here. A home-body skill is written by her home body.
HOME_SKILL_REFUSAL = ("Not saved: that skill lives on my home machine, and this shelf only reads it. To change it, "
                      "start a home job that edits it there (the same file, the same change).")


def writable_skill_path(path: str) -> bool:
    """A path her own skill shelf can write (the fences OwnerSkillWrites applies): /skills/<slug>/..."""
    return bool(SKILL_FILE.match(path or '')) and '..' not in (path or '')
_FEED_DOWN = ("My bodies' skills couldn't be read just now (the door's feed didn't answer), so I can't tell "
              "which ones I have from there.")


def rows_from_feed(rows):
    """Feed rows (rack_seq, memory_id, key_label, event_type, content, context, source, observed_at), in
    feed order -> her live, well-formed skills: per memory its last event decides (a tombstone retires it);
    per label the newest live memory is the skill."""
    last = {}
    for row in rows:
        last[row[1]] = row
    newest = {}
    for seq, _memory, label, event, content, context, source, observed in last.values():
        slug = str(label or "").removeprefix("skill.")
        if (not str(label or "").startswith("skill.") or event == "tombstone" or not _BODY_SLUG.fullmatch(slug)
                or not str(content or "").strip()):
            continue
        if slug not in newest or seq > newest[slug][0]:
            newest[slug] = (seq, {"slug": slug, "content": str(content),
                                  "when": str(context or "").removeprefix("skill: ").strip(),
                                  "source": str(source or "portable"), "observed": observed})
    return [skill for _, skill in sorted(newest.values(), key=lambda pair: pair[1]["slug"])]


def body_skill_md(row):
    """Her body's skill as a SKILL.md: name is its slug (the directory), description one plain line."""
    source = "".join(ch for ch in row["source"] if ch.isalnum() or ch in "-_") or "portable"
    description = " ".join(f"From my {source} body. {row['when']}".split())[:200]
    date = row["observed"].strftime("%b %d, %Y") if hasattr(row["observed"], "strftime") else ""
    return (f"---\nname: {row['slug']}\ndescription: {json.dumps(description)}\n---\n\n"
            f"(Written by my {source} body{' on ' + date if date else ''}; it reached me through the door. "
            f"I read it here; my {source} body keeps it.)\n\n{row['content'].rstrip()}\n")


def feed_loader(settings, timeout=5):
    """Production: read the door's feed from her brain's own database."""
    def load():
        import psycopg

        with psycopg.connect(db_url(settings.database_url), connect_timeout=timeout) as conn:
            return rows_from_feed(conn.execute(FEED_SKILLS_SQL).fetchall())
    return load


class BodySkillsBackend(StoreBackend):
    """Her bodies' skills, read-only, from the door's feed: an in-memory store refilled at most every
    refresh_s (a feed hiccup keeps the last view). Everything that would change it says no."""

    def __init__(self, load, *, refresh_s=30.0, clock=None):
        from langgraph.store.memory import InMemoryStore
        import time

        super().__init__(namespace=lambda rt: _BODY_NAMESPACE, store=InMemoryStore())
        self._load, self._refresh_s, self._clock = load, refresh_s, clock or time.monotonic
        self._loaded_at = None
        self._loaded_ok = False          # until one load works, a failure is an error, not an empty shelf
        self._refresh_lock = threading.Lock()

    def _refresh(self):
        from langgraph.store.memory import InMemoryStore

        with self._refresh_lock:
            now = self._clock()
            if self._loaded_at is not None and now - self._loaded_at < self._refresh_s:
                return
            self._loaded_at = now
            try:
                rows = self._load()
            except Exception:
                import logging
                logging.getLogger(__name__).warning("body skills: the door's feed didn't answer; keeping the last view",
                                                    exc_info=True)
                return
            fresh = StoreBackend(namespace=lambda rt: _BODY_NAMESPACE, store=InMemoryStore())
            for row in rows:
                fresh.write(f"/{row['slug']}/SKILL.md", body_skill_md(row))
            self._store, self._loaded_ok = fresh._store, True

    def ls(self, path):
        self._refresh()
        if not self._loaded_ok:          # Codex 11/18: never let "couldn't read" look like "has none"
            from deepagents.backends.protocol import LsResult
            return LsResult(error=_FEED_DOWN)
        return super().ls(path)

    def read(self, file_path, offset=0, limit=2000):
        self._refresh()
        return super().read(file_path, offset=offset, limit=limit)

    def download_files(self, paths):
        self._refresh()
        return super().download_files(paths)

    async def als(self, path):
        import asyncio
        await asyncio.to_thread(self._refresh)
        if not self._loaded_ok:
            from deepagents.backends.protocol import LsResult
            return LsResult(error=_FEED_DOWN)
        return await super().als(path)

    async def aread(self, file_path, offset=0, limit=2000):
        import asyncio
        await asyncio.to_thread(self._refresh)
        return await super().aread(file_path, offset=offset, limit=limit)

    async def adownload_files(self, paths):
        import asyncio
        await asyncio.to_thread(self._refresh)
        return await super().adownload_files(paths)

    def write(self, file_path, content):
        from deepagents.backends.protocol import WriteResult
        return WriteResult(error=_READ_ONLY)

    async def awrite(self, file_path, content):
        return self.write(file_path, content)

    def edit(self, file_path, old_string, new_string, replace_all=False):
        from deepagents.backends.protocol import EditResult
        return EditResult(error=_READ_ONLY)

    async def aedit(self, file_path, old_string, new_string, replace_all=False):
        return self.edit(file_path, old_string, new_string, replace_all)

    def delete(self, file_path):
        from deepagents.backends.protocol import DeleteResult
        return DeleteResult(error=_READ_ONLY)

    async def adelete(self, file_path):
        return self.delete(file_path)

    async def aupload_files(self, files):
        return self.upload_files(files)

    def upload_files(self, files):
        from deepagents.backends.protocol import FileUploadResponse
        return [FileUploadResponse(path=path, error="permission_denied") for path, _ in files]


def skills_middleware(store, owner_person_id, *, body_skills=None):
    own = backend_for(store)
    # D67: her bodies' skills at /home-skills/, listed FIRST so her own shelf wins a name clash (later
    # sources override); writes stay her own shelf's alone (the permissions below and OwnerSkillWrites).
    backend = CompositeBackend(default=own, routes={BODY_SKILLS: body_skills}) if body_skills is not None else own
    sources = ([BODY_SKILLS] if body_skills is not None else []) + ["/skills/"]
    return [
        FilesystemMiddleware(
            backend=backend, tools=["ls", "read_file", "write_file", "edit_file"],
            tool_token_limit_before_evict=None, human_message_token_limit_before_evict=None,
            permissions=[
                FilesystemPermission(operations=["write"], paths=["/skills/**"], mode="allow"),
                FilesystemPermission(operations=["write"], paths=["/**"], mode="deny"),
            ], system_prompt=FILES_PROMPT,
        ),
        FreshSkillsMiddleware(backend=backend, sources=sources, system_prompt=SKILLS_PROMPT),
        OwnerSkillWrites(owner_person_id, own),
    ]


def approved_write(middleware, row):
    """Execute exactly the approved arguments through the same owner/path fences.

    Watcher execution is outside the graph, so supply ToolRuntime explicitly.
    The store is already bound to the backend; no graph state is needed.
    """
    from langchain.tools import ToolRuntime

    name = row.args['tool']
    config = {'configurable': {'identity': row.args.get('identity') or {}, 'thread_id': row.thread_id}}
    runtime = ToolRuntime(state={}, context=None, config=config, stream_writer=lambda _: None,
                          tool_call_id='approved-skill', store=None)
    request = SimpleNamespace(tool_call={'name': name, 'id': 'approved-skill', 'args': row.args['tool_args']},
                              runtime=runtime)
    tool = next(t for t in middleware[0].tools if t.name == name)
    return middleware[-1].wrap_tool_call(
        request, lambda _: tool.invoke({**row.args['tool_args'], 'runtime': runtime}, config=config))


def skill_path(slug):
    if not re.fullmatch(r'[a-z0-9]+(?:-[a-z0-9]+)*', slug) or len(slug) > 64:
        raise ValueError('Use a lowercase skill slug of at most 64 characters.')
    return f'/skills/{slug}/SKILL.md'


def validate_skill(slug, content):
    skill_path(slug)
    match = re.match(r'\A---\r?\n(.*?)\r?\n---(?:\r?\n|$)', content, re.S)
    if not match:
        raise ValueError('Skill needs YAML front matter.')
    try:
        metadata = yaml.safe_load(match[1])
    except yaml.YAMLError:
        raise ValueError('Invalid skill front matter.') from None
    if not isinstance(metadata, dict) or metadata.get('name') != slug:
        raise ValueError('Skill name must match its slug.')
    if set(metadata) - {'name', 'description'}:
        raise ValueError('Skill front matter may contain only name and description.')
    description = metadata.get('description')
    if not isinstance(description, str) or not description.strip() or len(description) > 1024:
        raise ValueError('Skill needs a plain description of at most 1024 characters.')
    # Descriptions are always in context: reject numeric identifiers, UUIDs,
    # entity ids, account handles, addresses and long opaque tokens.
    if re.search(r'[\r\n\d@]|\b\w+[.:/]\w+|\b\w{32,}\b', description):
        raise ValueError('Description must be one plain line without ids or personal values.')
    return metadata


def main(argv=None, *, store=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('setup')
    commands.add_parser('list')
    commands.add_parser('get').add_argument('slug')
    put = commands.add_parser('put')
    put.add_argument('slug')
    put.add_argument('file', type=Path)
    args = parser.parse_args(argv)
    # Validate before opening a database connection.
    if args.command in ('get', 'put'):
        path = skill_path(args.slug)
    if args.command == 'put':
        content = args.file.read_text()
        validate_skill(args.slug, content)
    if store is None:
        from aerys_v2.config import Settings
        store = store_for(Settings())
    backend = backend_for(store)
    if args.command == 'setup':
        store.setup()
    elif args.command == 'put':
        result = backend.write(path, content)
        if result.error:
            raise ValueError(result.error)
        print(f'Updated file {path}')
    elif args.command == 'get':
        result = backend.download_files([path])[0]
        if result.error:
            raise ValueError(result.error)
        print(result.content.decode('utf-8'), end='')
    else:
        offset = 0
        while items := store.search(NAMESPACE, limit=100, offset=offset):
            for item in items:
                if not re.fullmatch(r'/skills/[^/]+/SKILL.md', item.key):
                    continue
                response = backend.download_files([item.key])[0]
                slug = item.key.split('/')[2]
                try:
                    metadata = validate_skill(slug, response.content.decode('utf-8'))
                except (ValueError, AttributeError, UnicodeError):
                    metadata = {'name': slug, 'description': '(invalid skill; inspect with get)'}
                print(json.dumps(dict(slug=slug, name=metadata['name'], description=metadata['description'],
                                      created_at=str(item.created_at), updated_at=str(item.updated_at))))
            offset += len(items)


if __name__ == '__main__':
    main()
