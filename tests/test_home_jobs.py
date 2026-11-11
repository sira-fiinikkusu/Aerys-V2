"""Her new loop starts long jobs on her home body (board #176, s3-02), through LangChain's
AsyncSubAgentMiddleware, behind the D10 gate (hands_gate.py).

What these pin, end to end through a real create_agent loop, the real LangGraph SDK client and a
stand-in for her home server (the Agent Protocol routes aerys-portable's home/server.py serves):
  * his typed ask starts the job; a later turn, on any of his surfaces, still finds it (her action
    turns are one-shot, so the jobs live in a ledger keyed to him, not in the turn's state);
  * by voice it waits for his button and runs once, after Approve (D10), on the same official tools;
  * a change mid-job interrupts the running job on the same thread (D11);
  * guests never see or touch his jobs; nothing about this arms unless everything it needs is there;
  * no graph nodes are added, so her recursion rail is unchanged;
  * the done-notice for a job she started lands at the desk, carrying where she asked from.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any

import httpx
import pytest

pytest.importorskip("langchain.agents")
pytest.importorskip("deepagents.middleware.async_subagents")
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool
from langchain_core.utils.function_calling import convert_to_openai_tool
from langgraph_sdk.client import SyncLangGraphClient
from pydantic import ConfigDict

import aerys_v2.factory as factory
from aerys_v2.action_loop import AgentToolModels, build_agent_action_graph
from aerys_v2.approvals import Approvals, MemoryApprovalStore
from aerys_v2.hands_gate import HOME_JOB, HandsGate, MemoryGatedActionStore
from aerys_v2.home_jobs import (
    HOME_AGENT,
    HOME_GRAPH_ID,
    HOME_JOBS_OVERLAY,
    MemoryHomeTaskLedger,
    build_home_jobs,
    home_jobs_for,
)
from aerys_v2.home_notices import HomeNotices, MemoryNoticeStore

NOW = datetime(2026, 11, 9, 15, 0, tzinfo=timezone.utc)
OWNER = "6e6bcbed-0000-4000-8000-000000000001"
DM_THREAD = f"person:{OWNER}"
VOICE_THREAD = "voice:office"
HOME_URL = "http://home.test:2024"
TOKEN = "home-api-token"

TEXT = {"user_id": OWNER, "display_name": "Chris", "platform": "discord", "channel_kind": "dm"}
STRONG = {"user_id": OWNER, "display_name": "Chris", "voice": True,
          "speaker": {"id": "chris", "confidence": 0.82, "demoted": False}}
GUEST = {"user_id": "someone-else", "display_name": "Sam", "platform": "discord", "channel_kind": "guild"}


@pytest.fixture(autouse=True)
def fixed_clock(monkeypatch):
    monkeypatch.setattr(factory, "_where_when_line", lambda thread, identity: "\nIt is noon.")


# --- her home server, as the SDK sees it -------------------------------------------------------


class HomeServer:
    """The Agent Protocol routes home/server.py serves, in memory, with its auth and its rules:
    a run needs a user message; a busy thread takes a follow-up only with 'interrupt' (D11)."""

    def __init__(self):
        self.threads: dict[str, dict] = {}
        self.runs: dict[str, dict] = {}
        self.calls: list[tuple[str, str, Any]] = []
        self.finish_as = "success"

    def handler(self, request: httpx.Request) -> httpx.Response:
        if request.headers.get("authorization") != f"Bearer {TOKEN}":
            return httpx.Response(401, json={"detail": "unauthorized"})
        if "x-api-key" in request.headers:
            return httpx.Response(400, json={"detail": "a key that isn't hers was sent"})
        body = json.loads(request.content or b"{}") if request.content else {}
        path = request.url.path.strip("/").split("/")
        self.calls.append((request.method, request.url.path, body))
        if request.method == "POST" and path == ["threads"]:
            tid = f"t-{len(self.threads) + 1}"
            self.threads[tid] = {"thread_id": tid, "metadata": body.get("metadata") or {}, "status": "idle",
                                 "values": {"messages": [{"type": "ai", "content": "Planner page built."}]},
                                 "created_at": "2026-11-09T15:00:00Z"}
            return httpx.Response(200, json=self.threads[tid])
        if request.method == "POST" and len(path) == 3 and path[0] == "threads" and path[2] == "runs":
            tid = path[1]
            if tid not in self.threads:
                return httpx.Response(404, json={"detail": "Thread not found"})
            msgs = (body.get("input") or {}).get("messages") or []
            if not any(m.get("role") in ("user", "human") and str(m.get("content", "")).strip() for m in msgs):
                return httpx.Response(400, json={"detail": "a run needs a user message"})
            active = [r for r in self.runs.values() if r["thread_id"] == tid and r["status"] in ("pending", "running")]
            strategy = body.get("multitask_strategy")
            if active and strategy != "interrupt":
                return httpx.Response(409, json={"detail": "a run is already active on this thread"})
            for r in active:
                r["status"] = "interrupted"
            rid = f"r-{len(self.runs) + 1}"
            self.runs[rid] = {"run_id": rid, "thread_id": tid, "assistant_id": body.get("assistant_id") or "hands",
                              "status": "running", "created_at": "2026-11-09T15:00:00Z", "error": None}
            return httpx.Response(200, json=self.runs[rid])
        if request.method == "GET" and len(path) == 4 and path[2] == "runs":
            run = self.runs.get(path[3])
            return httpx.Response(200, json=run) if run else httpx.Response(404, json={"detail": "Run not found"})
        if request.method == "GET" and len(path) == 3 and path[2] == "runs":
            mine = [r for r in self.runs.values() if r["thread_id"] == path[1]]
            return httpx.Response(200, json=list(reversed(mine))[:1])
        if request.method == "GET" and len(path) == 2 and path[0] == "threads":
            thread = self.threads.get(path[1])
            return httpx.Response(200, json=thread) if thread else httpx.Response(404, json={"detail": "nope"})
        if request.method == "POST" and len(path) == 5 and path[4] == "cancel":
            run = self.runs.get(path[3])
            if run:
                run["status"] = "interrupted"
            return httpx.Response(200, json={})
        return httpx.Response(404, json={"detail": f"no route {request.method} {request.url.path}"})

    def finish(self, status="success"):
        for r in self.runs.values():
            if r["status"] in ("pending", "running"):
                r["status"] = status

    def created(self, what):
        return [c for c in self.calls if c[0] == "POST" and (c[1] == "/threads" if what == "thread"
                                                             else c[1].endswith("/runs"))]


@pytest.fixture()
def home(monkeypatch):
    """Her home server behind the REAL LangGraph SDK client (only the socket is replaced)."""
    server = HomeServer()
    import deepagents.middleware.async_subagents as official

    def client_for(*, url=None, api_key=None, headers=None, timeout=None):
        assert url == HOME_URL
        return SyncLangGraphClient(httpx.Client(base_url=url, headers=dict(headers or {}),
                                                transport=httpx.MockTransport(lambda request: server.handler(request))))

    def async_client_for(*, url=None, api_key=None, headers=None, timeout=None):
        from langgraph_sdk.client import LangGraphClient

        assert url == HOME_URL
        return LangGraphClient(httpx.AsyncClient(base_url=url, headers=dict(headers or {}),
                                                 transport=httpx.MockTransport(lambda request: server.handler(request))))

    monkeypatch.setattr(official, "get_sync_client", client_for)
    monkeypatch.setattr(official, "get_client", async_client_for)
    return server


# --- her loop ----------------------------------------------------------------------------------


class Rec(BaseChatModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    script: Any
    log: Any
    bound: Any = ()

    @property
    def _llm_type(self) -> str:
        return "rec"

    def bind_tools(self, tools, *, tool_choice=None, **kwargs):
        return self.model_copy(update={"bound": tuple(convert_to_openai_tool(t)["function"]["name"] for t in tools)})

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.log.append({"tools": self.bound, "messages": list(messages)})
        if not self.script:
            raise AssertionError("model called more times than scripted")
        return ChatResult(generations=[ChatGeneration(message=self.script.pop(0))])


@tool
def echo(text: str) -> str:
    """Say it back."""
    return f"echo: {text}"


class Box:
    """Her home jobs wired as production wires them, over in-memory stores."""

    def __init__(self, members=None):
        self.approval_store = MemoryApprovalStore()
        self.pinged = []
        self.approvals = Approvals(self.approval_store, notify=lambda t, m: self.pinged.append((t, m)),
                                   clock=lambda: NOW, sleep=lambda s: None)
        self.ledger = MemoryHomeTaskLedger()
        self.actions = MemoryGatedActionStore()

        def gate_factory(executors):
            return HandsGate(self.approvals, self.actions, owner_person_id=OWNER, people=None,
                             executors=executors, read_only_agents=(), clock=lambda: NOW, members=members)

        self.jobs = build_home_jobs(url=HOME_URL, token=TOKEN, ledger=self.ledger, owner_person_id=OWNER,
                                    gate_factory=gate_factory, projects=sorted(set((members or {}).values())))
        self.notes = []
        self.jobs.watcher.note = lambda thread, text: self.notes.append((thread, text))

    def turn(self, script, identity, thread=DM_THREAD, text="do it"):
        log = []
        model = Rec(script=list(script), log=log)
        graph = build_agent_action_graph(AgentToolModels(model, (model, True)), "SOUL", [echo],
                                         extra_middleware=self.jobs.middleware)
        out = graph.invoke({"messages": [HumanMessage(content=text)]},
                           {"configurable": {"thread_id": thread, "identity": identity}, "recursion_limit": 12})
        return out, log

    def tap(self, approval_id, approve=True):
        return self.approval_store.decide(approval_id, approve, 1, NOW)


def call(name, args, cid="c1"):
    return AIMessage(content="", tool_calls=[{"id": cid, "name": name, "args": args}])


def tool_texts(out):
    return [m.content for m in out["messages"] if isinstance(m, ToolMessage)]


START = {"description": "Build the fridge planner page", "subagent_type": HOME_AGENT}


# --- the tests ---------------------------------------------------------------------------------


def test_his_typed_ask_starts_a_job_on_her_home_body_and_a_later_turn_finds_it(home):
    b = Box()
    out, log = b.turn([call("start_async_task", START), AIMessage(content="Started.")], TEXT)
    assert any("Launched async subagent" in t for t in tool_texts(out))
    assert "start_async_task" in log[0]["tools"] and "list_async_tasks" in log[0]["tools"]
    (thread_call,) = home.created("thread")
    (run_call,) = home.created("run")
    # what her home server needs: the task as a user message on assistant 'hands', and the
    # thread named as hers, with where she asked from (echoed back in its done-notice)
    assert thread_call[2]["metadata"] == {"subagent_name": HOME_AGENT, "parent_thread_id": DM_THREAD}
    assert run_call[2]["assistant_id"] == HOME_GRAPH_ID
    assert run_call[2]["input"]["messages"] == [{"role": "user", "content": START["description"]}]
    (task_id,) = b.ledger.load(OWNER)
    home.finish()
    # a NEW turn (her action turns are one-shot) still finds it, and reads how it went
    out, _ = b.turn([call("check_async_task", {"task_id": task_id}), AIMessage(content="It's done.")], TEXT,
                    text="how's the planner?")
    (checked,) = tool_texts(out)
    assert '"status": "success"' in checked and "Planner page built." in checked
    assert b.ledger.load(OWNER)[task_id]["status"] == "success"


def test_a_job_started_on_one_of_his_surfaces_is_found_from_another(home):
    b = Box()
    b.turn([call("start_async_task", START), AIMessage(content="Started.")], TEXT)
    out, _ = b.turn([call("list_async_tasks", {}), AIMessage(content="One job.")], STRONG, thread=VOICE_THREAD,
                    text="what's running at home?")
    (listed,) = tool_texts(out)
    assert "t-1" in listed and "No tracked" not in listed


def test_his_voice_waits_for_his_button_and_the_job_runs_once_after_approve(home):
    b = Box()
    out, _ = b.turn([call("start_async_task", START), AIMessage(content="Asked him.")], STRONG,
                    thread=VOICE_THREAD)
    (said,) = tool_texts(out)
    assert "Waiting for Chris" in said
    assert home.created("thread") == [] and home.created("run") == []     # nothing until he taps
    assert len(b.pinged) == 1
    (row,) = b.actions.waiting()
    b.tap(row.approval_id)
    assert b.jobs.watcher.tick()["done"] == 1
    (thread_call,) = home.created("thread")
    assert thread_call[2]["metadata"]["parent_thread_id"] == VOICE_THREAD
    assert len(home.created("run")) == 1
    assert list(b.ledger.load(OWNER)) == ["t-1"]
    assert b.notes and "has started" in b.notes[-1][1]
    assert b.jobs.watcher.tick() == {"done": 0, "failed": 0, "dropped": 0}   # never twice
    assert len(home.created("run")) == 1


def test_a_change_mid_job_interrupts_the_running_job_on_the_same_thread(home):
    b = Box()
    b.turn([call("start_async_task", START), AIMessage(content="Started.")], TEXT)
    (task_id,) = b.ledger.load(OWNER)
    out, _ = b.turn([call("update_async_task", {"task_id": task_id, "message": "Make the header blue."}),
                     AIMessage(content="Adjusted.")], TEXT, text="make it blue")
    assert any("Updated async subagent" in t for t in tool_texts(out))
    first, second = home.created("run")
    assert second[1] == f"/threads/{task_id}/runs" and second[2]["multitask_strategy"] == "interrupt"
    assert home.runs["r-1"]["status"] == "interrupted"
    assert b.ledger.load(OWNER)[task_id]["run_id"] == "r-2"


def test_a_change_by_voice_waits_for_his_button_then_adjusts_the_job(home):
    b = Box()
    b.turn([call("start_async_task", START), AIMessage(content="Started.")], TEXT)
    (task_id,) = b.ledger.load(OWNER)
    out, _ = b.turn([call("update_async_task", {"task_id": task_id, "message": "Make it blue."}),
                     AIMessage(content="Asked him.")], STRONG, thread=VOICE_THREAD)
    assert "Waiting for Chris" in tool_texts(out)[0]
    (row,) = b.actions.waiting()
    # his button names the job the change is for (his jobs are loaded before the gate decides)
    shown = b.approval_store.get(row.approval_id)
    assert "Home agent: home." in shown.detail and "(home)" in shown.summary
    b.tap(row.approval_id)
    assert b.jobs.watcher.tick()["done"] == 1
    assert home.created("run")[-1][2]["multitask_strategy"] == "interrupt"
    assert b.ledger.load(OWNER)[task_id]["run_id"] == "r-2"


def test_cancelling_his_job_stops_it_on_her_home_body(home):
    b = Box()
    b.turn([call("start_async_task", START), AIMessage(content="Started.")], TEXT)
    (task_id,) = b.ledger.load(OWNER)
    b.turn([call("cancel_async_task", {"task_id": task_id}), AIMessage(content="Stopped.")], TEXT, text="stop it")
    assert home.runs["r-1"]["status"] == "interrupted"
    assert b.ledger.load(OWNER)[task_id]["status"] == "cancelled"


UNSURE = {"user_id": OWNER, "display_name": "Chris", "voice": True,
          "speaker": {"id": "unknown", "confidence": 0.31, "demoted": False}}


def test_a_stop_from_a_voice_she_isnt_sure_of_waits_for_his_button_then_stops_the_job_once(home):
    # D17 Q3 (11/09): it could be someone else in the house, and a cancel can end a long job.
    b = Box()
    b.turn([call("start_async_task", START), AIMessage(content="Started.")], TEXT)
    (task_id,) = b.ledger.load(OWNER)
    out, _ = b.turn([call("cancel_async_task", {"task_id": task_id}), AIMessage(content="Asked him.")], UNSURE,
                    thread=VOICE_THREAD, text="stop that")
    assert "Waiting for Chris" in tool_texts(out)[0]
    assert home.runs["r-1"]["status"] == "running"                         # nothing stopped yet
    assert b.ledger.load(OWNER)[task_id]["status"] != "cancelled"
    (row,) = b.actions.waiting()
    b.tap(row.approval_id)
    assert b.jobs.watcher.tick()["done"] == 1
    assert home.runs["r-1"]["status"] == "interrupted"
    assert b.ledger.load(OWNER)[task_id]["status"] == "cancelled"
    assert b.notes and "stopped" in b.notes[-1][1] and task_id in b.notes[-1][1]
    cancels = [c for c in home.calls if c[0] == "POST" and c[1].endswith("/cancel")]
    assert len(cancels) == 1
    assert b.jobs.watcher.tick() == {"done": 0, "failed": 0, "dropped": 0}   # never twice
    assert len([c for c in home.calls if c[0] == "POST" and c[1].endswith("/cancel")]) == 1


def test_an_approved_stop_of_a_job_she_doesnt_know_fails_in_plain_words(home):
    b = Box()
    b.turn([call("cancel_async_task", {"task_id": "no-such-task"}), AIMessage(content="Asked him.")], UNSURE,
           thread=VOICE_THREAD, text="stop that")
    (row,) = b.actions.waiting()
    b.tap(row.approval_id)
    assert b.jobs.watcher.tick()["failed"] == 1
    assert not [c for c in home.calls if c[1].endswith("/cancel")]
    assert b.notes and "not a home job" not in b.notes[-1][1]


def test_guests_never_see_or_touch_his_jobs(home):
    b = Box()
    b.turn([call("start_async_task", START), AIMessage(content="Started.")], TEXT)
    seen = []
    real_load = b.ledger.load
    b.ledger.load = lambda owner: seen.append(owner) or real_load(owner)
    out, _ = b.turn([call("list_async_tasks", {}), AIMessage(content="No.")], GUEST, thread="discord:guild:1")
    (said,) = tool_texts(out)
    assert "only Chris" in said and "t-1" not in said
    assert seen == []                                  # his ledger isn't even read for a guest
    out, _ = b.turn([call("start_async_task", START), AIMessage(content="No.")], GUEST, thread="discord:guild:1")
    assert "only Chris" in tool_texts(out)[0] and len(home.created("run")) == 1


def test_her_loop_gains_no_graph_nodes_so_her_recursion_rail_is_unchanged(home):
    b = Box()
    model = Rec(script=[], log=[])
    plain = build_agent_action_graph(AgentToolModels(model, (model, True)), "SOUL", [echo])
    with_home = build_agent_action_graph(AgentToolModels(model, (model, True)), "SOUL", [echo],
                                         extra_middleware=b.jobs.middleware)
    assert set(with_home.get_graph().nodes) == set(plain.get_graph().nodes)


def test_the_home_tools_are_offered_only_with_the_middleware(home):
    b = Box()
    _, log = b.turn([AIMessage(content="hi")], TEXT, text="hello")
    assert {"start_async_task", "check_async_task", "update_async_task", "cancel_async_task",
            "list_async_tasks"} <= set(log[0]["tools"])
    model = Rec(script=[AIMessage(content="hi")], log=[])
    plain = build_agent_action_graph(AgentToolModels(model, (model, True)), "SOUL", [echo])
    plain.invoke({"messages": [HumanMessage(content="hello")]},
                 {"configurable": {"thread_id": DM_THREAD, "identity": TEXT}, "recursion_limit": 12})
    assert not any(n.endswith("async_task") or n == "list_async_tasks" for n in model.log[0]["tools"])
    assert "home" in HOME_JOBS_OVERLAY.lower() and "start_async_task" in HOME_JOBS_OVERLAY


def test_a_failed_launch_after_his_ok_is_told_honestly(home, monkeypatch):
    b = Box()
    b.turn([call("start_async_task", START), AIMessage(content="Asked him.")], STRONG, thread=VOICE_THREAD)
    (row,) = b.actions.waiting()
    b.tap(row.approval_id)
    home.handler = lambda request: httpx.Response(503, json={"detail": "down"})
    assert b.jobs.watcher.tick()["failed"] == 1
    assert b.ledger.load(OWNER) == {}
    assert "didn't go through" in b.notes[-1][1]


def test_the_done_notice_for_a_job_she_started_lands_at_the_desk_carrying_where_she_asked(home):
    b = Box()
    b.turn([call("start_async_task", START), AIMessage(content="Started.")], TEXT)
    home.finish()
    # her home server's done-notice: the run, its final status, and the thread's own metadata
    run = home.runs["r-1"]
    payload = {"thread_id": run["thread_id"], "run_id": run["run_id"], "status": run["status"],
               "metadata": home.threads[run["thread_id"]]["metadata"]}
    told = []
    desk = SimpleNamespace(store=SimpleNamespace(decided=lambda key, since: False),
                           reach_out=lambda event: told.append(event) or "sent")
    store = MemoryNoticeStore()
    notices = HomeNotices(desk, store, clock=lambda: NOW)
    assert notices.receive(payload) == "new"
    assert notices.tick()["handed"] == 1
    (event,) = told
    assert event.dedupe_key == "home-job:r-1:success" and event.owed is True and event.source == "home-job"
    (kept,) = store.rows.values()
    assert kept.metadata == {"subagent_name": HOME_AGENT, "parent_thread_id": DM_THREAD}



TELEGRAM = {"user_id": OWNER, "display_name": "Chris", "platform": "telegram", "channel_kind": "dm"}


def test_a_job_remembers_which_of_his_surfaces_asked_for_it(home):
    # #183 (11/10): the done-notice goes back where he asked (D20), so the job keeps where that was.
    # The official tools rebuild their task records on every call; what they don't carry is kept.
    b = Box()
    b.turn([call("start_async_task", START), AIMessage(content="Started.")], TEXT)
    ((task_id, task),) = b.ledger.load(OWNER).items()
    assert task["asked_on"] == "discord"
    b.turn([call("check_async_task", {"task_id": task_id}), AIMessage(content="Running.")], STRONG,
           thread=VOICE_THREAD, text="how's it going?")
    assert b.ledger.load(OWNER)[task_id]["asked_on"] == "discord"          # a check doesn't move it
    b.turn([call("update_async_task", {"task_id": task_id, "message": "make it blue"}),
            AIMessage(content="Changed.")], TELEGRAM, text="make it blue")
    assert b.ledger.load(OWNER)[task_id]["asked_on"] == "telegram"         # his latest ask does
    assert b.ledger.asked_on(task["thread_id"]) == "telegram"
    assert b.ledger.asked_on("no-such-thread") is None


def test_a_job_he_started_in_discord_tells_him_there_when_its_done(home):
    b = Box()
    b.turn([call("start_async_task", START), AIMessage(content="Started.")], TEXT)
    home.finish()
    run = home.runs["r-1"]
    payload = {"thread_id": run["thread_id"], "run_id": run["run_id"], "status": run["status"],
               "metadata": home.threads[run["thread_id"]]["metadata"]}
    told = []
    desk = SimpleNamespace(store=SimpleNamespace(decided=lambda key, since: False),
                           reach_out=lambda event: told.append(event) or "sent")
    notices = HomeNotices(desk, MemoryNoticeStore(), clock=lambda: NOW,
                          asked_on=lambda notice: b.ledger.asked_on(notice.thread_id))
    notices.receive(payload)
    notices.tick()
    (event,) = told
    assert event.text_only is True and event.text_to == "discord"


# --- arming: nothing unless everything is there ------------------------------------------------


def _settings(**over):
    from pydantic import SecretStr
    base = dict(home_jobs=True, action_loop="agent", home_url=HOME_URL, home_api_token=SecretStr(TOKEN),
                database_url="postgresql://x/aerys_v2", owner_person_id=OWNER, hands_gates=True, approvals=True,
                reach_notify_service="mobile_app_phone", storm_notify_service="",
                ha_token=SimpleNamespace(get_secret_value=lambda: "t"), ha_base_url="http://ha:8123",
                memories_database_url="postgresql://x/aerys", home_read_only_agents="")
    return SimpleNamespace(**{**base, **over})


def test_not_armed_unless_everything_it_needs_is_there():
    assert home_jobs_for(_settings(home_jobs=False), environ={}) is None
    assert home_jobs_for(_settings(action_loop="graph"), environ={}) is None
    assert home_jobs_for(_settings(home_url=""), environ={}) is None
    assert home_jobs_for(_settings(home_api_token=None), environ={}) is None
    assert home_jobs_for(_settings(hands_gates=False), environ={}) is None     # the D10 gate is required
    assert home_jobs_for(_settings(approvals=False), environ={}) is None
    # the LangGraph SDK would send any of these to her home machine as x-api-key
    assert home_jobs_for(_settings(), environ={"LANGSMITH_API_KEY": "ls-key"}) is None
    armed = home_jobs_for(_settings(), environ={})
    assert armed is not None and HOME_JOB in armed.gate.executors
    names = [type(m).__name__ for m in armed.middleware]
    assert len(names) == 3 and names[1] == "HomeJobGate"


def test_with_a_bot_to_send_with_his_files_ride_along_last():
    # #183: send_file_to_chris mounts with her home jobs when a Discord or Telegram bot can deliver.
    from pydantic import SecretStr
    armed = home_jobs_for(_settings(discord_bot_token=SecretStr("discord-bot-token")), environ={})
    names = [type(m).__name__ for m in armed.middleware]
    assert len(names) == 4 and names[-1] == "SendFile"
    # #187: the record of what she sent rides with it.
    assert [t.name for t in armed.middleware[-1].tools] == ["send_file_to_chris", "files_sent_to_chris"]


# --- wiring: the action stack mounts it only when armed ----------------------------------------


def test_the_action_stack_mounts_home_jobs_on_the_new_loop_only_when_armed(home, monkeypatch):
    import aerys_v2.action_loop as action_loop
    import aerys_v2.home_jobs as home_jobs
    from aerys_v2.config import Settings
    from aerys_v2.factory import action_stack_for

    seen = []
    real = action_loop.build_agent_action_graph
    monkeypatch.setattr(action_loop, "build_agent_action_graph",
                        lambda *a, **kw: seen.append(kw) or real(*a, **kw))
    b = Box()
    monkeypatch.setattr(home_jobs, "home_jobs_for", lambda settings: b.jobs)

    def settings(**over):
        return Settings(_env_file=None, anthropic_api_key="sk-test", ha_token="ha-token", action_loop="agent", **over)

    _, graph = action_stack_for(settings(home_jobs=True), soul="s")
    assert graph.home_jobs is b.jobs and seen[-1]["extra_middleware"] == b.jobs.middleware
    assert HOME_JOBS_OVERLAY in seen[-1]["overlay"]
    _, graph = action_stack_for(settings(), soul="s")        # HOME_JOBS off: nothing changes
    assert graph.home_jobs is None and "extra_middleware" not in seen[-1]
    assert HOME_JOBS_OVERLAY not in seen[-1]["overlay"]


# --- Codex review 1 (11/09) --------------------------------------------------------------------


def test_listing_home_as_read_only_never_lets_his_voice_start_it_directly(home):
    b = Box()
    b.jobs = build_home_jobs(url=HOME_URL, token=TOKEN, ledger=b.ledger, owner_person_id=OWNER,
                             gate_factory=lambda ex: HandsGate(b.approvals, b.actions, owner_person_id=OWNER,
                                                               people=None, executors=ex,
                                                               read_only_agents=(HOME_AGENT,), clock=lambda: NOW))
    out, _ = b.turn([call("start_async_task", START), AIMessage(content="Asked him.")], STRONG, thread=VOICE_THREAD)
    assert "Waiting for Chris" in tool_texts(out)[0] and home.created("run") == []


def test_a_check_and_a_change_at_the_same_moment_never_leave_a_stale_run_on_record(home):
    import threading

    b = Box()
    b.turn([call("start_async_task", START), AIMessage(content="Started.")], TEXT)
    (task_id,) = b.ledger.load(OWNER)
    real = home.handler
    hold, change_reached = threading.Event(), threading.Event()

    def handler(request):
        if request.method == "POST" and request.url.path == f"/threads/{task_id}/runs":
            change_reached.set()
        response = real(request)
        if request.method == "GET" and request.url.path.endswith("/runs/r-1"):
            hold.wait(5)                     # the check has read r-1 as running, and hasn't saved yet
        return response

    home.handler = handler
    check = threading.Thread(target=b.turn, args=([call("check_async_task", {"task_id": task_id}),
                                                   AIMessage(content="Running.")], TEXT), kwargs={"text": "how's it going?"})
    change = threading.Thread(target=b.turn, args=([call("update_async_task", {"task_id": task_id, "message": "Blue."}),
                                                    AIMessage(content="Adjusted.")], TEXT), kwargs={"text": "make it blue"})
    check.start()
    change.start()
    change_reached.wait(1.0)        # only gets here if the change didn't have to wait for the check
    hold.set()
    check.join(10)
    change.join(10)
    assert b.ledger.load(OWNER)[task_id]["run_id"] == "r-2"
    home.handler = real
    b.turn([call("cancel_async_task", {"task_id": task_id}), AIMessage(content="Stopped.")], TEXT, text="stop")
    assert home.runs["r-2"]["status"] == "interrupted"


def test_when_his_jobs_cant_be_read_nothing_starts(home):
    b = Box()

    def down(owner):
        raise RuntimeError("db down")

    b.ledger.load = down
    out, _ = b.turn([call("start_async_task", START), AIMessage(content="Couldn't.")], TEXT)
    assert "nothing happened" in tool_texts(out)[0] and home.created("run") == []


def test_a_job_that_started_but_couldnt_be_written_down_says_so(home):
    b = Box()

    def down(owner, tasks):
        raise RuntimeError("db down")

    b.ledger.save = down
    out, _ = b.turn([call("start_async_task", START), AIMessage(content="Started.")], TEXT)
    (said,) = tool_texts(out)
    assert "Launched" in said and "couldn't write" in said and len(home.created("run")) == 1


def test_every_unfinished_job_stays_reachable_however_many_there_are():
    from aerys_v2.home_jobs import MAX_TASKS

    ledger = MemoryHomeTaskLedger(clock=lambda: NOW)
    stamp = "2026-11-09T14:00:00Z"
    running = {f"run-{i}": {"task_id": f"run-{i}", "agent_name": "home", "thread_id": f"run-{i}", "run_id": "r",
                            "status": "running", "created_at": stamp, "last_checked_at": stamp,
                            "last_updated_at": stamp} for i in range(MAX_TASKS + 5)}
    done = {f"done-{i}": {**t, "task_id": f"done-{i}", "status": "success"} for i, t in enumerate(running.values())}
    ledger.save(OWNER, {**running, **done})
    got = ledger.load(OWNER)
    assert set(running) <= set(got)
    assert len([t for t in got.values() if t["status"] == "success"]) == MAX_TASKS


def test_the_sdks_error_text_never_reaches_the_logs_or_his_record(home, caplog):
    b = Box()
    marker = "PRIVATE-JOB-WORDS-7731"
    home.handler = lambda request: httpx.Response(400, json={"detail": f"echo: {marker}"})
    with caplog.at_level("DEBUG"):
        out, _ = b.turn([call("start_async_task", START), AIMessage(content="Couldn't.")], TEXT)
    assert marker not in caplog.text
    b2 = Box()
    b2.turn([call("start_async_task", START), AIMessage(content="Asked him.")], STRONG, thread=VOICE_THREAD)
    (row,) = b2.actions.waiting()
    b2.tap(row.approval_id)
    with caplog.at_level("DEBUG"):
        assert b2.jobs.watcher.tick()["failed"] == 1
    assert marker not in caplog.text
    (finished,) = b2.actions.rows.values() if hasattr(b2.actions, "rows") else [None]
    assert finished is None or marker not in (finished.outcome or "")
    assert marker not in b2.notes[-1][1]


def test_a_stop_that_couldnt_cancel_an_earlier_run_logs_the_line_without_the_sdks_text(caplog):
    # The official cancel logs a failed earlier-run cancel with the SDK's traceback (Gemini, 11/09):
    # keep the line, drop that text, as for a failed launch or update.
    import logging

    Box()                                                       # arming installs the filter
    official = logging.getLogger("deepagents.middleware.async_subagents")
    marker = "PRIVATE-SDK-TEXT-5512"
    with caplog.at_level("DEBUG"):
        try:
            raise RuntimeError(f"Client error '400' body: {marker}")
        except RuntimeError:
            official.warning("Failed to cancel earlier run %s of task %s", "r-1", "t-1", exc_info=True)
    assert "Failed to cancel earlier run r-1 of task t-1" in caplog.text
    assert marker not in caplog.text


def test_the_ledgers_async_hook_keeps_his_jobs_the_same_way():
    import asyncio

    from langchain.tools import ToolRuntime
    from langgraph.prebuilt.tool_node import ToolCallRequest
    from langgraph.types import Command

    from aerys_v2.home_jobs import home_ledger_middleware

    ledger = MemoryHomeTaskLedger(clock=lambda: NOW)
    stamp = "2026-11-09T15:00:00Z"
    ledger.save(OWNER, {"t-1": {"task_id": "t-1", "agent_name": "home", "thread_id": "t-1", "run_id": "r-1",
                                "status": "running", "created_at": stamp, "last_checked_at": stamp,
                                "last_updated_at": stamp}})
    mw = home_ledger_middleware(ledger, OWNER)

    def request(identity):
        rt = ToolRuntime(state={"messages": []}, context=None, stream_writer=lambda *_: None, store=None,
                         config={"configurable": {"thread_id": DM_THREAD, "identity": identity}}, tool_call_id="c1")
        return ToolCallRequest(tool_call={"id": "c1", "name": "check_async_task", "args": {"task_id": "t-1"}},
                               tool=None, state={"messages": []}, runtime=rt)

    async def handler(req):
        seen = req.runtime.state["async_tasks"]["t-1"]
        return Command(update={"messages": [ToolMessage("ok", tool_call_id="c1")],
                               "async_tasks": {"t-1": {**seen, "status": "success", "run_id": "r-2"}}})

    asyncio.run(mw.awrap_tool_call(request(TEXT), handler))
    assert ledger.load(OWNER)["t-1"]["run_id"] == "r-2"

    async def guest_handler(req):
        assert "async_tasks" not in req.runtime.state       # his jobs never reach a guest's call
        return "refused by the gate"

    assert asyncio.run(mw.awrap_tool_call(request(GUEST), guest_handler)) == "refused by the gate"


def test_what_is_on_record_wins_over_what_the_turn_last_saw():
    from langchain.tools import ToolRuntime
    from langgraph.prebuilt.tool_node import ToolCallRequest

    from aerys_v2.home_jobs import home_ledger_middleware

    stamp = "2026-11-09T15:00:00Z"
    rec = {"task_id": "t-1", "agent_name": "home", "thread_id": "t-1", "status": "running",
           "created_at": stamp, "last_checked_at": stamp, "last_updated_at": stamp}
    ledger = MemoryHomeTaskLedger(clock=lambda: NOW)
    ledger.save(OWNER, {"t-1": {**rec, "run_id": "r-2"}})            # another surface moved it on
    stale = {"messages": [], "async_tasks": {"t-1": {**rec, "run_id": "r-1"}}}
    rt = ToolRuntime(state=stale, context=None, stream_writer=lambda *_: None, store=None, tool_call_id="c1",
                     config={"configurable": {"thread_id": DM_THREAD, "identity": TEXT}})
    req = ToolCallRequest(tool_call={"id": "c1", "name": "cancel_async_task", "args": {"task_id": "t-1"}},
                          tool=None, state=stale, runtime=rt)
    seen = []
    home_ledger_middleware(ledger, OWNER).wrap_tool_call(
        req, lambda r: seen.append(r.runtime.state["async_tasks"]["t-1"]["run_id"]) or "ok")
    assert seen == ["r-2"]


def test_approved_member_executor_uses_member_ledger_and_strips_private_args():
    from langgraph.types import Command
    from aerys_v2.home_jobs import home_job_executor
    from aerys_v2.hands_gate import GatedAction

    member = "22222222-2222-2222-2222-222222222222"
    ledger = MemoryHomeTaskLedger(clock=lambda: NOW)
    ledger.save(OWNER, {"owner-task": {"task_id": "owner-task", "status": "running"}})
    calls = []

    def start(*, description, subagent_type, runtime):
        calls.append((description, subagent_type, runtime.state["async_tasks"]))
        return Command(update={"async_tasks": {"member-task": {"task_id": "member-task", "status": "running"}}})

    executor = home_job_executor(SimpleNamespace(tools=[SimpleNamespace(name="start_async_task", func=start)]),
                                 ledger, OWNER)
    row = GatedAction(1, 1, HOME_JOB, "test", 1, "person:member",
                      {"tool": "start_async_task", "description": "Build a page", "subagent_type": "home-member",
                       "_job_owner": member, "_start_candidates": {"2": {"person": member}}})
    assert executor(row)[0] is True
    assert calls == [("Build a page", "home-member", {})]
    assert set(ledger.load(OWNER)) == {"owner-task"}
    assert set(ledger.load(member)) == {"member-task"}


# --- per-person job lists (D19 slice 3) ----------------------------------------------------------------

MEMBER = "22222222-2222-2222-2222-222222222222"
MEMBER_TEXT = {"user_id": MEMBER, "platform": "discord", "channel_kind": "dm"}
SHARED_THREAD = "voice:room"


def _rec(tid, run="r-1", status="running"):
    stamp = "2026-11-09T15:00:00Z"
    return {"task_id": tid, "agent_name": "home", "thread_id": tid, "run_id": run, "status": status,
            "created_at": stamp, "last_checked_at": stamp, "last_updated_at": stamp}


def _ledger_request(identity, name="list_async_tasks", args=None, thread=SHARED_THREAD, state=None):
    from langchain.tools import ToolRuntime
    from langgraph.prebuilt.tool_node import ToolCallRequest

    state = state if state is not None else {"messages": []}
    rt = ToolRuntime(state=state, context=None, stream_writer=lambda *_: None, store=None, tool_call_id="c1",
                     config={"configurable": {"thread_id": thread, "identity": identity}})
    return ToolCallRequest(tool_call={"id": "c1", "name": name, "args": args or {}}, tool=None, state=state, runtime=rt)


def test_her_turn_sees_her_jobs_only_and_they_never_land_in_a_shared_threads_state():
    from langgraph.types import Command

    from aerys_v2.home_jobs import home_ledger_middleware

    ledger = MemoryHomeTaskLedger(clock=lambda: NOW)
    ledger.save(OWNER, {"his": _rec("his")})
    ledger.save(MEMBER, {"hers": _rec("hers")})
    room = {"messages": [], "async_tasks": {"his-in-room": _rec("his-in-room")}}   # a room his voice uses too
    mw = home_ledger_middleware(ledger, OWNER, {MEMBER: "home-member"})
    seen = {}

    def handler(req):
        seen.update(req.runtime.state["async_tasks"])
        return Command(update={"messages": [ToolMessage("1 tracked task(s)", tool_call_id="c1")],
                               "async_tasks": {"hers": {**seen["hers"], "status": "success", "run_id": "r-2"}}})

    result = mw.wrap_tool_call(_ledger_request(MEMBER_TEXT, state=room), handler)
    assert set(seen) == {"hers"} and seen["hers"]["owner"] == MEMBER       # hers, marked hers
    assert "async_tasks" not in result.update and result.update["messages"]  # the thread never holds hers
    assert ledger.load(MEMBER)["hers"]["run_id"] == "r-2"
    assert set(ledger.load(OWNER)) == {"his"} and ledger.load(OWNER)["his"]["run_id"] == "r-1"


def test_his_turns_are_unchanged_when_she_is_a_member():
    from langgraph.types import Command

    from aerys_v2.home_jobs import home_ledger_middleware

    ledger = MemoryHomeTaskLedger(clock=lambda: NOW)
    ledger.save(OWNER, {"his": _rec("his")})
    ledger.save(MEMBER, {"hers": _rec("hers")})
    room = {"messages": [], "async_tasks": {"from-the-room": _rec("from-the-room")}}
    seen = {}

    def handler(req):
        seen.update(req.runtime.state["async_tasks"])
        return Command(update={"messages": [ToolMessage("ok", tool_call_id="c1")],
                               "async_tasks": {"his": {**seen["his"], "run_id": "r-2"}}})

    for members in (None, {MEMBER: "home-member"}):
        seen.clear()
        result = home_ledger_middleware(ledger, OWNER, members).wrap_tool_call(
            _ledger_request(TEXT, thread=DM_THREAD, state=room), handler)
        assert set(seen) == {"his", "from-the-room"} and "owner" not in seen["his"]
        assert result.update["async_tasks"]["his"]["run_id"] == "r-2"
    assert set(ledger.load(MEMBER)) == {"hers"} and ledger.load(MEMBER)["hers"]["run_id"] == "r-1"


def test_her_async_turn_uses_her_ledger_and_a_guest_still_gets_none():
    import asyncio

    from langgraph.types import Command

    from aerys_v2.home_jobs import home_ledger_middleware

    ledger = MemoryHomeTaskLedger(clock=lambda: NOW)
    ledger.save(OWNER, {"his": _rec("his")})
    ledger.save(MEMBER, {"hers": _rec("hers")})
    mw = home_ledger_middleware(ledger, OWNER, {MEMBER: "home-member"})

    async def handler(req):
        assert set(req.runtime.state["async_tasks"]) == {"hers"}
        return Command(update={"messages": [ToolMessage("stopped", tool_call_id="c1")],
                               "async_tasks": {"hers": {**_rec("hers"), "status": "cancelled"}}})

    req = _ledger_request(MEMBER_TEXT, "cancel_async_task", {"task_id": "hers"}, thread=f"person:{MEMBER}")
    result = asyncio.run(mw.awrap_tool_call(req, handler))
    assert "async_tasks" not in result.update and ledger.load(MEMBER)["hers"]["status"] == "cancelled"

    async def guest(req):
        assert "async_tasks" not in req.runtime.state
        return "refused by the gate"

    assert asyncio.run(mw.awrap_tool_call(_ledger_request(GUEST), guest)) == "refused by the gate"


def test_her_typed_start_is_recorded_in_her_list_through_the_real_gate():
    from langgraph.types import Command

    from aerys_v2.home_jobs import home_ledger_middleware
    from aerys_v2.hands_gate import home_job_middleware

    ledger = MemoryHomeTaskLedger(clock=lambda: NOW)
    gate = HandsGate(Approvals(MemoryApprovalStore(), clock=lambda: NOW), MemoryGatedActionStore(),
                     owner_person_id=OWNER, members={MEMBER: "home-member"}, clock=lambda: NOW)
    outer, inner = home_ledger_middleware(ledger, OWNER, gate.members), home_job_middleware(gate)

    def tool(req):
        return Command(update={"messages": [ToolMessage("started", tool_call_id="c1")],
                               "async_tasks": {"new": _rec("new")}})

    start = {"description": "Make a calendar", "subagent_type": "home-member"}
    req = _ledger_request(MEMBER_TEXT, "start_async_task", start, thread=f"person:{MEMBER}")
    result = outer.wrap_tool_call(req, lambda r: inner.wrap_tool_call(r, tool))
    assert set(ledger.load(MEMBER)) == {"new"} and ledger.load(OWNER) == {}
    assert "async_tasks" not in result.update
    listed = {}

    def lister(req):
        listed.update(req.runtime.state["async_tasks"])
        return "1 tracked task(s)"

    assert outer.wrap_tool_call(_ledger_request(MEMBER_TEXT, thread=f"person:{MEMBER}"),
                                lambda r: inner.wrap_tool_call(r, lister)) == "1 tracked task(s)"
    assert set(listed) == {"new"}


def test_her_turns_keep_a_commands_routing_even_when_her_record_fails():
    from langgraph.types import Command

    from aerys_v2.home_jobs import home_ledger_middleware

    class Failing(MemoryHomeTaskLedger):
        def save(self, owner, tasks):
            raise ConnectionError("database away")

    for ledger in (MemoryHomeTaskLedger(clock=lambda: NOW), Failing(clock=lambda: NOW)):
        MemoryHomeTaskLedger.save(ledger, MEMBER, {"hers": _rec("hers")})
        mw = home_ledger_middleware(ledger, OWNER, {MEMBER: "home-member"})

        def handler(req):
            return Command(graph=Command.PARENT, goto="next", resume="value",
                           update={"messages": [ToolMessage("ok", tool_call_id="c1")], "async_tasks": {"hers": _rec("hers")}})

        result = mw.wrap_tool_call(_ledger_request(MEMBER_TEXT, thread=f"person:{MEMBER}"), handler)
        assert (result.graph, result.goto, result.resume) == (Command.PARENT, "next", "value")
        assert "async_tasks" not in result.update and result.update["messages"]



# --- her own project on the brain (D19 slice 4) ----------------------------------------------------------

def test_each_household_project_is_its_own_home_subagent():
    from aerys_v2.home_jobs import MEMBER_DESCRIPTION, home_subagent_middleware

    mw = home_subagent_middleware(HOME_URL, TOKEN, ["home-member", "home-member", "home", " ", "hands"])
    (start,) = [t for t in mw.tools if t.name == "start_async_task"]
    assert "home-member" in start.description and HOME_AGENT in start.description
    assert MEMBER_DESCRIPTION.format(project="home-member") in start.description
    alone = home_subagent_middleware(HOME_URL, TOKEN)
    (plain,) = [t for t in alone.tools if t.name == "start_async_task"]
    assert "home-member" not in plain.description


def test_her_typed_job_runs_as_her_project_and_is_hers_alone(home):
    b = Box(members={MEMBER: "home-member"})
    his = {"description": "Build the fridge planner page", "subagent_type": HOME_AGENT}
    b.turn([call("start_async_task", his), AIMessage(content="Started.")], TEXT)
    mine = {"description": "Make a calendar", "subagent_type": "home-member"}
    out, _ = b.turn([call("start_async_task", mine), AIMessage(content="Started.")], MEMBER_TEXT,
                    thread=f"person:{MEMBER}")
    assert any("Launched async subagent" in t for t in tool_texts(out))
    runs = home.created("run")
    assert [r[2]["assistant_id"] for r in runs] == [HOME_GRAPH_ID, "home-member"]   # her folder at home
    (her_task,) = b.ledger.load(MEMBER)
    assert her_task not in b.ledger.load(OWNER) and not (out.get("async_tasks") or {})
    out, _ = b.turn([call("list_async_tasks", {}), AIMessage(content="One job.")], MEMBER_TEXT,
                    thread=f"person:{MEMBER}", text="what's running?")
    (listed,) = tool_texts(out)
    assert her_task in listed and "1 tracked task" in listed


def test_his_jobs_never_go_into_her_project(home):
    b = Box(members={MEMBER: "home-member"})
    out, _ = b.turn([call("start_async_task", {"description": "Tidy her calendar", "subagent_type": "home-member"}),
                     AIMessage(content="Not started.")], TEXT)
    (said,) = tool_texts(out)
    assert said.startswith("Not started: that's a household member's own project")
    assert not home.created("run") and b.ledger.load(OWNER) == {}
