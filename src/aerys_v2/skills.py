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
from deepagents.backends import StoreBackend
from deepagents.backends.utils import perform_string_replacement
from deepagents.middleware.filesystem import FilesystemMiddleware, FilesystemPermission
from deepagents.middleware.skills import SkillsMiddleware
from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import ToolMessage

from aerys_v2.rooms import his_private_read, runtime_identity

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
    return _postgres_store(settings.database_url, settings.checkpoint_pool_open_timeout_s)


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
        if not SKILL_FILE.match(path) or '..' in path:
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


def skills_middleware(store, owner_person_id):
    backend = backend_for(store)
    return [
        FilesystemMiddleware(
            backend=backend, tools=["ls", "read_file", "write_file", "edit_file"],
            tool_token_limit_before_evict=None, human_message_token_limit_before_evict=None,
            permissions=[
                FilesystemPermission(operations=["write"], paths=["/skills/**"], mode="allow"),
                FilesystemPermission(operations=["write"], paths=["/**"], mode="deny"),
            ], system_prompt=FILES_PROMPT,
        ),
        FreshSkillsMiddleware(backend=backend, sources=["/skills/"], system_prompt=SKILLS_PROMPT),
        OwnerSkillWrites(owner_person_id, backend),
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
