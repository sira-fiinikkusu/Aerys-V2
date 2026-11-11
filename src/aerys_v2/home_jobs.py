"""Her new loop starts long jobs on her home body (board #176, s3-02).

Her home body is aerys-portable's home/server.py on her own VM: an Agent Protocol server, which
is exactly what LangChain's AsyncSubAgentMiddleware (deepagents) talks to. So this wires the
official middleware into her create_agent loop (action_loop.py) with ONE async subagent, "home",
and puts the D10 gate (hands_gate.py) in front of it:

  * his typed words start any home job; his voice waits for his Approve button (D10), and the
    approved job is started by the watcher with the same official tool, once;
  * a change mid-job interrupts the running job on the same thread (D11, "adjust in flight");
  * "I'm done" comes back through the done-notice (home_notices.py, Stage 3), not by polling:
    notify_parent is off, as the middleware documents for apps that deliver results themselves.

Two things differ from a stock deep agent, both because of how her turns run:

  * Her action turns are ONE-SHOT (no checkpointer; the main thread keeps the conversation), so
    the middleware's `async_tasks` state would vanish when the turn ends and a later "how's the
    planner?" couldn't find the job. His jobs are kept in a small ledger (v2_home_tasks), keyed to
    him, not to a thread: a job started from Discord is found from the office speaker. The ledger
    is handed to the official tools through their own runtime state, and every update the tools
    return is written back.
  * Her turns run on `invoke` (sync). The official hooks only check task statuses on the async
    path, and as graph nodes they would cost super-steps against her recursion rail. They're left
    out (a subclass resets them to the base class's), so the compiled graph is still model + tools:
    statuses come from check/list, and the outcome from the done-notice.

Fail closed: nothing arms without HOME_JOBS, the new loop, her home address and token, and the
D10 gate armed on the approval button. The LangGraph SDK sends any LANGGRAPH/LANGSMITH/LANGCHAIN
API key in the environment as x-api-key, so it won't arm with one present (that key isn't hers to
give her home machine). The job's words are the private part: never logged here.
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import json
import logging
import os
import re
import threading
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Callable

from aerys_v2.hands_gate import (
    CANCEL,
    HOME_JOB,
    HOME_TOOLS,
    OTHER,
    START,
    TEXT_PLATFORMS,
    UPDATE,
    HandsGate,
    HandsWatcher,
    asker,
    hands_gate_for,
    home_job_middleware,
)
from aerys_v2.state import identity_from_config, is_voice_turn

log = logging.getLogger(__name__)

HOME_AGENT = "home"
#: The assistant id her home server runs (home/server.py takes any; 'hands' is its default).
HOME_GRAPH_ID = "hands"
#: Environment keys the LangGraph SDK would send to her home machine as x-api-key.
SDK_KEY_ENV = ("LANGGRAPH_API_KEY", "LANGSMITH_API_KEY", "LANGCHAIN_API_KEY")
#: What a turn loads: EVERY unfinished job (so none can become unreachable), and up to MAX_TASKS
#: finished ones from the last few days, newest first.
KEEP_FINISHED = timedelta(days=3)
MAX_TASKS = 25
_UNFINISHED = ("pending", "running", "waiting")
#: How long a home tool waits for another call on his jobs to finish (they're quick HTTP calls).
LOCK_WAIT_S = 30
#: Where he last asked about a job (discord | telegram | voice), kept on its record: its done-notice goes
#: back there as text when that was Discord or Telegram (the owner's D20 rule, for jobs since #183 11/10).
ASKED_ON = "asked_on"

CANT_READ = ("Not done: I couldn't get to his home jobs just now (they couldn't be read, or another "
             "request on them was still going), so nothing happened. Try again in a moment.")
NOT_RECORDED = (" (I couldn't write this job down, so I may not be able to check on it or change it "
                "later. He still hears when it's done.)")

HOME_DESCRIPTION = (
    "Your own home machine (a VM that's yours): you there, on Opus, with your helpers and minions, "
    "for long work that shouldn't hold up a conversation: building something, researching in depth, "
    "organizing files. A job gets 45 minutes and 150 steps; Chris hears when it's done. Deleting, "
    "money, messaging people and reaching outside the machine still stop for him there."
)

HOME_JOBS_OVERLAY = (
    "LONG JOBS AT HOME: for work that takes more than a few minutes (building, deep research, "
    "organizing files), start it on your home machine with start_async_task, subagent_type \"home\", "
    "and a full description of the job. It runs in the background; you don't wait for it, and you "
    "don't check on it right after starting. When it finishes, Chris is told (\"I'm done\"), so "
    "don't promise to report back yourself. If he changes his mind mid-job, send the change with "
    "update_async_task: the job adjusts in flight. check_async_task and list_async_tasks say how his "
    "jobs are going when he asks; cancel_async_task stops one. If a tool says it's waiting for "
    "Chris's tap, tell him it's waiting on his button and never say it started. To hand him a file "
    "your home job made, send_file_to_chris with its path: it lands in his DM. When he asks what you "
    "sent him, files_sent_to_chris is the record (direct sends aren't jobs, so the job list won't "
    "show them); answer from it and don't send the file again unless he asks you to."
)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _parse(stamp) -> datetime | None:
    try:
        return datetime.strptime(str(stamp), "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except (TypeError, ValueError):
        return None


def _recent(tasks: dict, now: datetime) -> dict:
    """The jobs a turn should know: every unfinished one, and the newest finished ones from the
    last few days (Codex review 11/09: a cap on unfinished jobs made the oldest unreachable)."""
    unfinished, finished = [], []
    for task in tasks.values():
        if not isinstance(task, dict) or not task.get("task_id"):
            continue
        when = _parse(task.get("last_updated_at")) or _parse(task.get("created_at"))
        if task.get("status") in _UNFINISHED:
            unfinished.append(task)
        elif when is not None and now - when <= KEEP_FINISHED:
            finished.append((when, task))
    finished.sort(key=lambda pair: pair[0], reverse=True)
    return {t["task_id"]: t for t in [*unfinished, *(t for _, t in finished[:MAX_TASKS])]}


def _plain_failure(text) -> str:
    """A failed home call, said without the SDK's text (it can carry the server's response body,
    and so the job's words): what failed, and the HTTP status when there is one."""
    text = str(text or "")
    code = re.search(r"\b([45]\d\d)\b", text)
    status = f" (HTTP {code.group(1)})" if code else ""
    if text.startswith("Failed to update"):
        return f"her home machine didn't take the change{status}"
    if text.startswith("Failed to cancel"):
        return f"her home machine didn't take the cancel{status}"
    if text.startswith("Failed to"):
        return f"her home machine didn't take the job{status}"
    return " ".join(text.split())[:160]


class _NoSdkText(logging.Filter):
    """The official tools log a failed launch or update with the SDK's exception text, and a
    failed earlier-run cancel with its traceback; keep the line, drop that text (Codex review
    11/09; the cancel line, Gemini 11/09)."""

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            said = record.getMessage()
        except Exception:
            return True
        if said.startswith(("Failed to launch async subagent", "Failed to update async subagent")):
            record.msg, record.args = "home jobs: %s", (_plain_failure(said),)
        elif said.startswith("Failed to cancel earlier run"):
            record.exc_info, record.exc_text = None, None
        return True


def _quiet_official_logs() -> None:
    official = logging.getLogger("deepagents.middleware.async_subagents")
    if not any(isinstance(f, _NoSdkText) for f in official.filters):
        official.addFilter(_NoSdkText())


# --- the ledger: his jobs, across turns and surfaces ----------------------------------------


class MemoryHomeTaskLedger:
    """In-process ledger (tests, dev boxes). Same semantics as the Postgres one: locked(owner)
    holds his jobs for one load, call, save at a time."""

    def __init__(self, clock: Callable[[], datetime] | None = None):
        self.rows: dict[str, tuple[str, dict]] = {}
        self.clock = clock or _now
        self._lock = threading.Lock()
        self._owners: dict[str, threading.Lock] = defaultdict(threading.Lock)

    @contextlib.contextmanager
    def locked(self, owner: str):
        with self._lock:
            mine = self._owners[owner]
        if not mine.acquire(timeout=LOCK_WAIT_S):
            raise TimeoutError("another call on his home jobs is still running")
        try:
            yield self
        finally:
            mine.release()

    def load(self, owner: str) -> dict:
        with self._lock:
            mine = {tid: dict(task) for tid, (who, task) in self.rows.items() if who == owner}
        return _recent(mine, self.clock())

    def save(self, owner: str, tasks: dict) -> None:
        with self._lock:
            for tid, task in (tasks or {}).items():
                who = self.rows.get(tid, (owner, None))[0]
                if who == owner and isinstance(task, dict):
                    self.rows[tid] = (owner, dict(task))

    def asked_on(self, thread_id: str) -> str | None:
        """Where he last asked about the job running on her home thread `thread_id`, or None."""
        with self._lock:
            for tid, (_, task) in self.rows.items():
                if thread_id in (tid, task.get("thread_id")):
                    return task.get(ASKED_ON)
        return None


class PostgresHomeTaskLedger:
    """v2_home_tasks (db/migrations/024): one row per job, the official task record as JSON.

    locked(owner) is one session holding his advisory lock: every transport and the watcher take
    it around load, call, save, so a slower call can never write an older run over a newer one
    (Codex review 11/09: a check that read r-1 saved it after a change had moved the job to r-2,
    and a cancel then stopped the wrong run)."""

    LOAD = ("SELECT task FROM v2_home_tasks WHERE owner = %s AND task->>'status' = ANY(%s) "
            "UNION ALL (SELECT task FROM v2_home_tasks WHERE owner = %s AND NOT (task->>'status' = ANY(%s)) "
            "AND updated_at > now() - %s ORDER BY updated_at DESC LIMIT %s)")
    SAVE = ("INSERT INTO v2_home_tasks (task_id, owner, task, updated_at) VALUES (%s, %s, %s, now()) "
            "ON CONFLICT (task_id) DO UPDATE SET task = EXCLUDED.task, updated_at = now() "
            "WHERE v2_home_tasks.owner = EXCLUDED.owner")

    def __init__(self, database_url: str):
        self.database_url = database_url

    def _connect(self, **kw):
        import psycopg

        return psycopg.connect(self.database_url, connect_timeout=5, options="-c statement_timeout=5000", **kw)

    def _load(self, conn, owner: str) -> dict:
        rows = conn.execute(self.LOAD, (owner, list(_UNFINISHED), owner, list(_UNFINISHED),
                                        KEEP_FINISHED, MAX_TASKS)).fetchall()
        tasks = {}
        for (task,) in rows:
            task = task if isinstance(task, dict) else json.loads(task)
            if task.get("task_id"):
                tasks[task["task_id"]] = task
        return _recent(tasks, _now())

    def _save(self, conn, owner: str, tasks: dict) -> None:
        for tid, task in (tasks or {}).items():
            if isinstance(task, dict):
                # Upsert, but never across owners: a job is his alone.
                conn.execute(self.SAVE, (str(tid), owner, json.dumps(task)))

    def load(self, owner: str) -> dict:
        with self._connect() as conn:
            return self._load(conn, owner)

    def asked_on(self, thread_id: str) -> str | None:
        with self._connect() as conn:
            row = conn.execute("SELECT task->>%s FROM v2_home_tasks WHERE task_id = %s OR task->>'thread_id' = %s "
                               "ORDER BY updated_at DESC LIMIT 1", (ASKED_ON, thread_id, thread_id)).fetchone()
        return row[0] if row else None

    def save(self, owner: str, tasks: dict) -> None:
        if tasks:
            with self._connect() as conn:
                self._save(conn, owner, tasks)

    @contextlib.contextmanager
    def locked(self, owner: str):
        conn = self._connect(autocommit=True)
        try:
            conn.execute(f"SET lock_timeout = '{LOCK_WAIT_S}s'")
            conn.execute(f"SET statement_timeout = '{LOCK_WAIT_S + 5}s'")
            conn.execute("SELECT pg_advisory_lock(hashtext(%s))", (f"home-jobs:{owner}",))
            conn.execute("SET statement_timeout = '5s'")
            yield _Session(self, conn)
        finally:
            conn.close()                 # the session ends, and his lock with it


class _Session:
    """The ledger, on the one connection that holds his lock."""

    def __init__(self, ledger: PostgresHomeTaskLedger, conn):
        self.ledger, self.conn = ledger, conn

    def load(self, owner: str) -> dict:
        return self.ledger._load(self.conn, owner)

    def save(self, owner: str, tasks: dict) -> None:
        self.ledger._save(self.conn, owner, tasks)


def asked_from(identity, thread_id) -> str | None:
    """The surface a turn came on, as a job's record keeps it: discord | telegram | voice; None for his
    other lines (G2, panel, Sticky, Kael's line), whose notices keep the desk's follow-me delivery."""
    ident = identity if isinstance(identity, dict) else {}
    if is_voice_turn(ident, thread_id):
        return "voice"
    platform = str(ident.get("platform") or "").strip().lower()
    return platform if platform in TEXT_PLATFORMS else None


def keep_asked_on(tasks: dict, before: dict, asked: str | None = None) -> dict:
    """The official tools rebuild a job's record on every call, and only with their own fields: what
    the record already said about where he asked is carried over; `asked` (his ask that moved the job:
    a start or a change) replaces it."""
    out = {}
    for tid, task in (tasks or {}).items():
        if not isinstance(task, dict):
            out[tid] = task
            continue
        where = asked or task.get(ASKED_ON) or (before.get(tid) or {}).get(ASKED_ON)
        out[tid] = {**task, ASKED_ON: where} if where else task
    return out


# --- the middleware ----------------------------------------------------------------------------


def home_subagent_middleware(url: str, token: str):
    """The official AsyncSubAgentMiddleware with her one home subagent, minus its hook nodes."""
    from deepagents.middleware.async_subagents import AsyncSubAgentMiddleware
    from langchain.agents.middleware import AgentMiddleware

    class HomeSubAgents(AsyncSubAgentMiddleware):
        # On the sync path these hooks do nothing (statuses are pulled only on the async path),
        # and as nodes they'd spend her recursion rail; reset to the base class, create_agent adds
        # no node for them. The tools and the task state are the official ones, untouched.
        before_agent = AgentMiddleware.before_agent
        abefore_agent = AgentMiddleware.abefore_agent
        before_model = AgentMiddleware.before_model
        abefore_model = AgentMiddleware.abefore_model
        after_model = AgentMiddleware.after_model
        aafter_model = AgentMiddleware.aafter_model
        wrap_model_call = AgentMiddleware.wrap_model_call
        awrap_model_call = AgentMiddleware.awrap_model_call

    _quiet_official_logs()
    return HomeSubAgents(async_subagents=[{
        "name": HOME_AGENT,
        "description": HOME_DESCRIPTION,
        "graph_id": HOME_GRAPH_ID,
        "url": url,
        "headers": {"Authorization": f"Bearer {token}"},
        "notify_parent": False,              # "I'm done" is the done-notice's job (Stage 3)
        "update_strategy": "interrupt",      # D11: a change adjusts the running job in flight
    }])


def home_ledger_middleware(ledger, owner_person_id: str):
    """Hands his jobs to the official home tools (through their runtime state) and writes back
    every update they return, all under his lock: one load, call, save at a time, so nothing older
    is ever written over something newer. What's on record wins over what the turn last saw. If
    his jobs can't be read, nothing runs (fail closed); if a job went but couldn't be written down,
    she's told so. Only for him: for anyone else his ledger isn't even read (the D10 gate, next in
    line, refuses them)."""
    from langchain.agents.middleware import AgentMiddleware
    from langchain_core.messages import ToolMessage
    from langgraph.types import Command

    def mine(request) -> bool:
        if (request.tool_call or {}).get("name") not in HOME_TOOLS:
            return False
        config = getattr(request.runtime, "config", None) or {}
        thread_id = (config.get("configurable") or {}).get("thread_id")
        return asker(identity_from_config(config), thread_id, owner_person_id) != OTHER

    def refused(request):
        call = request.tool_call or {}
        return ToolMessage(content=CANT_READ, tool_call_id=call.get("id") or "", name=call.get("name"))

    def with_jobs(request, tasks: dict):
        state = request.runtime.state if isinstance(request.runtime.state, dict) else {}
        merged = {**state, "async_tasks": {**(state.get("async_tasks") or {}), **tasks}}
        return request.override(state=merged, runtime=dataclasses.replace(request.runtime, state=merged))

    def kept(session, result, before: dict, asked: str | None):
        update = result.update if isinstance(result, Command) and isinstance(result.update, dict) else None
        tasks = (update or {}).get("async_tasks")
        if not tasks:
            return result
        try:
            session.save(owner_person_id, keep_asked_on(tasks, before, asked))
            return result
        except Exception:
            # The job itself went (or changed); only her record of it didn't: say so.
            log.warning("home jobs: couldn't record %d job update(s)", len(tasks), exc_info=True)
        messages = [m.model_copy(update={"content": f"{m.content}{NOT_RECORDED}"}) if isinstance(m, ToolMessage) else m
                    for m in update.get("messages") or []]
        return Command(update={**update, "messages": messages})

    def asked(request) -> str | None:
        if (request.tool_call or {}).get("name") not in (START, UPDATE):
            return None                  # a check, a list or a stop doesn't move where he asked
        config = getattr(request.runtime, "config", None) or {}
        return asked_from(identity_from_config(config), (config.get("configurable") or {}).get("thread_id"))

    def guarded(request, handler):
        with contextlib.ExitStack() as stack:
            try:
                session = stack.enter_context(ledger.locked(owner_person_id))
                tasks = session.load(owner_person_id)
            except Exception:
                log.warning("home jobs: couldn't read his jobs; nothing runs", exc_info=True)
                return refused(request)
            return kept(session, handler(with_jobs(request, tasks)), tasks, asked(request))

    class HomeTaskLedger(AgentMiddleware):
        def wrap_tool_call(self, request, handler):
            return guarded(request, handler) if mine(request) else handler(request)

        async def awrap_tool_call(self, request, handler):
            if not mine(request):
                return await handler(request)
            loop = asyncio.get_running_loop()
            # The lock and the ledger are blocking; the official tool's async path runs inside.
            return await asyncio.to_thread(
                guarded, request, lambda req: asyncio.run_coroutine_threadsafe(handler(req), loop).result())

    return HomeTaskLedger()


def home_job_executor(subagents, ledger, owner_person_id: str) -> Callable:
    """What carries out a home job he approved (HandsWatcher): the same official tool she'd have
    called, under his lock, with his jobs loaded and her thread as the parent, then the update
    written down. Its outcome is plain words, never the SDK's text."""
    from langchain.tools import ToolRuntime
    from langgraph.types import Command

    tools = {t.name: t for t in subagents.tools}

    def run(row) -> tuple[bool, str]:
        args = dict(row.args or {})
        name = args.get("tool")
        if name == START:
            call = {"description": str(args.get("description") or ""), "subagent_type": str(args.get("subagent_type") or "")}
        elif name == UPDATE:
            call = {"task_id": str(args.get("task_id") or ""), "message": str(args.get("message") or "")}
        elif name == CANCEL:
            # D17 Q3 (11/09): a stop from a voice she wasn't sure was his, once he tapped Approve.
            call = {"task_id": str(args.get("task_id") or "")}
        else:
            return False, "not a home job"
        with ledger.locked(owner_person_id) as session:
            try:
                state = {"messages": [], "async_tasks": session.load(owner_person_id)}
            except Exception:
                log.warning("home jobs: couldn't read his jobs; the approved %s didn't run", name, exc_info=True)
                return False, ("I couldn't read his home jobs, so nothing was stopped" if name == CANCEL
                               else "I couldn't read his home jobs, so nothing was started")
            runtime = ToolRuntime(state=state, context=None, config={"configurable": {"thread_id": row.thread_id}},
                                  stream_writer=lambda *_: None, tool_call_id=f"approved-{row.id}", store=None)
            result = tools[name].func(**call, runtime=runtime)
            update = result.update if isinstance(result, Command) and isinstance(result.update, dict) else None
            tasks = (update or {}).get("async_tasks")
            if not tasks:
                return False, _plain_failure(result)
            recorded = ""
            try:
                session.save(owner_person_id, keep_asked_on(tasks, state["async_tasks"]))
            except Exception:
                log.warning("home jobs: an approved job went, but recording it failed", exc_info=True)
                recorded = NOT_RECORDED
        (task,) = tasks.values()
        if name == START:
            return True, f"It's running on your home machine (task {task['task_id']}).{recorded}"
        if name == CANCEL:
            return True, f"The job was stopped (task {task['task_id']}).{recorded}"
        return True, f"The change went to the running job (task {task['task_id']}); it adjusts in flight.{recorded}"

    return run


# --- arming ------------------------------------------------------------------------------------


@dataclass
class HomeJobs:
    """What her loop mounts (middleware, in order), what she's told (overlay), and the gate's
    watcher (--serve starts it and gives it the note injector)."""

    middleware: list
    overlay: str
    gate: HandsGate
    watcher: HandsWatcher


def build_home_jobs(*, url: str, token: str, ledger, owner_person_id: str,
                    gate_factory: Callable[[dict], HandsGate | None], files=None) -> HomeJobs | None:
    """files: send_file_middleware (tools/send_file.py), mounted last when there's a bot to send with."""
    subagents = home_subagent_middleware(url, token)
    gate = gate_factory({HOME_JOB: home_job_executor(subagents, ledger, owner_person_id)})
    if gate is None:
        return None
    if HOME_AGENT in gate.read_only_agents:
        # "home" is her whole machine, writes and all: it can never be the read-only exception that
        # lets his voice start a job without his button (Codex review 11/09).
        log.warning("home jobs: HOME_READ_ONLY_AGENTS lists %r, which can write; ignored", HOME_AGENT)
        gate.read_only_agents = frozenset(a for a in gate.read_only_agents if a != HOME_AGENT)
    # Order matters: his jobs are loaded first, so the gate can name the job a change is for;
    # then the gate decides; then the official tools run.
    middleware = [home_ledger_middleware(ledger, owner_person_id), home_job_middleware(gate), subagents]
    if files is not None:
        middleware.append(files)
    return HomeJobs(middleware=middleware, overlay=HOME_JOBS_OVERLAY, gate=gate, watcher=HandsWatcher(gate))


def home_jobs_for(settings, *, environ=None) -> HomeJobs | None:
    """None unless armed (see the module doc); each refusal says why in the log."""
    if not getattr(settings, "home_jobs", False):
        return None
    environ = os.environ if environ is None else environ
    token = getattr(settings, "home_api_token", None)
    reasons = []
    if getattr(settings, "action_loop", "graph") != "agent":
        reasons.append("it needs her new loop (ACTION_LOOP=agent)")
    if not str(getattr(settings, "home_url", "") or "").strip() or token is None:
        reasons.append("HOME_URL and HOME_API_TOKEN are needed")
    if not getattr(settings, "database_url", None) or getattr(settings, "owner_person_id", None) is None:
        reasons.append("a database and an owner are needed")
    found = [k for k in SDK_KEY_ENV if environ.get(k)]
    if found:
        reasons.append(f"{', '.join(found)} is set, and the LangGraph SDK would send it to her home machine")
    if reasons:
        log.warning("home jobs: not armed: %s", "; ".join(reasons))
        return None
    url = str(settings.home_url).strip().rstrip("/")
    files = None
    from aerys_v2.tools.send_file import (PostgresFileSends, build_files_sent_tool, build_send_file_tool,
                                          deliver_for, send_file_middleware)

    deliver = deliver_for(settings)
    if deliver is not None:     # #183: he gets the files her home jobs make, in his DM
        owner = str(settings.owner_person_id)
        sends = PostgresFileSends(settings.database_url)    # #187: the record, and the resend guard
        files = send_file_middleware(build_send_file_tool(home_url=url, home_token=token.get_secret_value(),
                                                          owner_person_id=owner, deliver=deliver, ledger=sends),
                                     owner, sent_tool=build_files_sent_tool(ledger=sends, owner_person_id=owner))
    jobs = build_home_jobs(url=url, token=token.get_secret_value(),
                           ledger=PostgresHomeTaskLedger(settings.database_url),
                           owner_person_id=str(settings.owner_person_id),
                           gate_factory=lambda executors: hands_gate_for(settings, executors=executors),
                           files=files)
    if jobs is None:
        log.warning("home jobs: not armed: the D10 gate isn't (HANDS_GATES and the approval button are needed)")
        return None
    log.info("home jobs armed | %s, D10 gate on%s", HOME_AGENT, ", files to him" if files is not None else "")
    return jobs
