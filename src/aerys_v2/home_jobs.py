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
import dataclasses
import json
import logging
import os
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Callable

from aerys_v2.hands_gate import (
    HOME_JOB,
    HOME_TOOLS,
    OTHER,
    START,
    UPDATE,
    HandsGate,
    HandsWatcher,
    asker,
    hands_gate_for,
    home_job_middleware,
)
from aerys_v2.state import identity_from_config

log = logging.getLogger(__name__)

HOME_AGENT = "home"
#: The assistant id her home server runs (home/server.py takes any; 'hands' is its default).
HOME_GRAPH_ID = "hands"
#: Environment keys the LangGraph SDK would send to her home machine as x-api-key.
SDK_KEY_ENV = ("LANGGRAPH_API_KEY", "LANGSMITH_API_KEY", "LANGCHAIN_API_KEY")
#: What a turn loads: every unfinished job, and finished ones from the last few days (newest first).
KEEP_FINISHED = timedelta(days=3)
MAX_TASKS = 25
_UNFINISHED = ("pending", "running", "waiting")

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
    "Chris's tap, tell him it's waiting on his button and never say it started."
)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _parse(stamp) -> datetime | None:
    try:
        return datetime.strptime(str(stamp), "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except (TypeError, ValueError):
        return None


def _recent(tasks: dict, now: datetime) -> dict:
    """The jobs a turn should know: unfinished ones, and finished ones from the last few days."""
    keep = []
    for task in tasks.values():
        if not isinstance(task, dict) or not task.get("task_id"):
            continue
        when = _parse(task.get("last_updated_at")) or _parse(task.get("created_at"))
        if task.get("status") in _UNFINISHED or (when is not None and now - when <= KEEP_FINISHED):
            keep.append((when or now, task))
    keep.sort(key=lambda pair: pair[0], reverse=True)
    return {t["task_id"]: t for _, t in keep[:MAX_TASKS]}


# --- the ledger: his jobs, across turns and surfaces ----------------------------------------


class MemoryHomeTaskLedger:
    """In-process ledger (tests, dev boxes). Same semantics as the Postgres one."""

    def __init__(self, clock: Callable[[], datetime] | None = None):
        self.rows: dict[str, tuple[str, dict]] = {}
        self.clock = clock or _now
        self._lock = threading.Lock()

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


class PostgresHomeTaskLedger:
    """v2_home_tasks (db/migrations/024): one row per job, the official task record as JSON."""

    def __init__(self, database_url: str):
        self.database_url = database_url

    def _connect(self):
        import psycopg

        return psycopg.connect(self.database_url, connect_timeout=5, options="-c statement_timeout=5000")

    def load(self, owner: str) -> dict:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT task FROM v2_home_tasks WHERE owner = %s AND (task->>'status' = ANY(%s) "
                "OR updated_at > now() - %s) ORDER BY updated_at DESC LIMIT %s",
                (owner, list(_UNFINISHED), KEEP_FINISHED, MAX_TASKS)).fetchall()
        tasks = {}
        for (task,) in rows:
            task = task if isinstance(task, dict) else json.loads(task)
            if task.get("task_id"):
                tasks[task["task_id"]] = task
        return _recent(tasks, _now())

    def save(self, owner: str, tasks: dict) -> None:
        if not tasks:
            return
        with self._connect() as conn:
            for tid, task in tasks.items():
                if not isinstance(task, dict):
                    continue
                # Upsert, but never across owners: a job is his alone.
                conn.execute(
                    "INSERT INTO v2_home_tasks (task_id, owner, task, updated_at) VALUES (%s, %s, %s, now()) "
                    "ON CONFLICT (task_id) DO UPDATE SET task = EXCLUDED.task, updated_at = now() "
                    "WHERE v2_home_tasks.owner = EXCLUDED.owner",
                    (str(tid), owner, json.dumps(task)))


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
    every update they return. Only for him: for anyone else his ledger isn't even read (the D10
    gate, next in line, refuses them)."""
    from langchain.agents.middleware import AgentMiddleware
    from langgraph.types import Command

    def mine(request) -> bool:
        if (request.tool_call or {}).get("name") not in HOME_TOOLS:
            return False
        config = getattr(request.runtime, "config", None) or {}
        thread_id = (config.get("configurable") or {}).get("thread_id")
        return asker(identity_from_config(config), thread_id, owner_person_id) != OTHER

    def loaded(request):
        try:
            tasks = ledger.load(owner_person_id)
        except Exception:
            log.warning("home jobs: couldn't read his jobs; this call goes without them", exc_info=True)
            return request
        state = request.runtime.state if isinstance(request.runtime.state, dict) else {}
        merged = {**state, "async_tasks": {**tasks, **(state.get("async_tasks") or {})}}
        return request.override(state=merged, runtime=dataclasses.replace(request.runtime, state=merged))

    def kept(result) -> None:
        update = getattr(result, "update", None) if isinstance(result, Command) else None
        tasks = update.get("async_tasks") if isinstance(update, dict) else None
        if not tasks:
            return
        try:
            ledger.save(owner_person_id, tasks)
        except Exception:
            # The job itself went (or changed); only her record of it didn't. "I'm done" still
            # reaches him through the done-notice.
            log.warning("home jobs: couldn't record %d job update(s)", len(tasks), exc_info=True)

    class HomeTaskLedger(AgentMiddleware):
        def wrap_tool_call(self, request, handler):
            if not mine(request):
                return handler(request)
            result = handler(loaded(request))
            kept(result)
            return result

        async def awrap_tool_call(self, request, handler):
            if not mine(request):
                return await handler(request)
            result = await handler(await asyncio.to_thread(loaded, request))
            await asyncio.to_thread(kept, result)
            return result

    return HomeTaskLedger()


def home_job_executor(subagents, ledger, owner_person_id: str) -> Callable:
    """What carries out a home job he approved (HandsWatcher): the same official tool she'd have
    called, with his jobs loaded and her thread as the parent, then the update written down."""
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
        else:
            return False, f"not a home job tool: {name!r}"
        state = {"messages": [], "async_tasks": ledger.load(owner_person_id)}
        runtime = ToolRuntime(state=state, context=None, config={"configurable": {"thread_id": row.thread_id}},
                              stream_writer=lambda *_: None, tool_call_id=f"approved-{row.id}", store=None)
        result = tools[name].func(**call, runtime=runtime)
        update = result.update if isinstance(result, Command) and isinstance(result.update, dict) else None
        tasks = (update or {}).get("async_tasks")
        if not tasks:
            return False, str(result)[:300]
        try:
            ledger.save(owner_person_id, tasks)
        except Exception:
            log.warning("home jobs: an approved job went, but recording it failed", exc_info=True)
        (task,) = tasks.values()
        if name == START:
            return True, f"It's running on her home machine (task {task['task_id']})."
        return True, f"The change went to the running job (task {task['task_id']}); it adjusts in flight."

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
                    gate_factory: Callable[[dict], HandsGate | None]) -> HomeJobs | None:
    subagents = home_subagent_middleware(url, token)
    gate = gate_factory({HOME_JOB: home_job_executor(subagents, ledger, owner_person_id)})
    if gate is None:
        return None
    # Order matters: his jobs are loaded first, so the gate can name the job a change is for;
    # then the gate decides; then the official tools run.
    return HomeJobs(middleware=[home_ledger_middleware(ledger, owner_person_id), home_job_middleware(gate), subagents],
                    overlay=HOME_JOBS_OVERLAY, gate=gate, watcher=HandsWatcher(gate))


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
    jobs = build_home_jobs(url=str(settings.home_url).strip().rstrip("/"), token=token.get_secret_value(),
                           ledger=PostgresHomeTaskLedger(settings.database_url),
                           owner_person_id=str(settings.owner_person_id),
                           gate_factory=lambda executors: hands_gate_for(settings, executors=executors))
    if jobs is None:
        log.warning("home jobs: not armed: the D10 gate isn't (HANDS_GATES and the approval button are needed)")
        return None
    log.info("home jobs armed | %s, D10 gate on", HOME_AGENT)
    return jobs
