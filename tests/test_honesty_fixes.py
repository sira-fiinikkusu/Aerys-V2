"""Offline tests for the two honesty fixes (2026-07-12 production incidents).

FIX 1 — the action-honesty gate: an 'action' turn that produced ZERO executed tool
calls is bounced ONCE with a correction; if it STILL touches nothing its reply is
emitted but the row is marked 'no_tool_action'. Live bug: "turn off the office
lights" -> "Both office lights are off." with tool_calls=[] (lights stayed on).

FIX 2 — an honest reply when the turn is rate-limited: the oauth/session-limit
RuntimeError used to re-raise into silence (the "Are you sure?" glasses turn got NO
reply). It now emits a short, honest, in-voice line — with the reset time converted
to Eastern when the error carries one — through the normal emitted_reply path (so
every transport benefits).

Fakes all the way down, same idiom as test_action_orchestration / test_turns.
"""

import json
import threading
import time
from datetime import datetime

import httpx
import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage

from aerys_v2.factory import build_action_graph, build_graph
from aerys_v2.router import RouteDecision
from aerys_v2.service import (
    ACTION_NO_TOOL_CORRECTION,
    EASTERN,
    GATE_EMIT,
    GATE_MARK,
    GATE_RETRY,
    NO_TOOL_ACTION_MARKER,
    action_honesty_gate,
    ask,
    rate_limit_reply,
)
from aerys_v2.service import _parse_reset_eastern
from aerys_v2.tools.home_control import build_home_control_tool, canary_set

CHRIS = {"user_id": "person-1", "display_name": "Chris"}


def fake_model(*replies) -> GenericFakeChatModel:
    msgs = [r if isinstance(r, AIMessage) else AIMessage(content=r) for r in replies]
    return GenericFakeChatModel(messages=iter(msgs))


def action_router(_text: str) -> RouteDecision:
    return RouteDecision(route="action", ack="[warmly] On it")


def home_tool():
    client = httpx.Client(
        transport=httpx.MockTransport(lambda req: httpx.Response(200, json=[]))
    )
    return build_home_control_tool(
        base_url="http://ha.test:8123", token="t",
        canary_entities=canary_set("light.desk"), client=client,
    )


class Recorder:
    """Thread-safe capturing recorder — ask() fires it on a daemon thread."""

    def __init__(self):
        self.rows: list[dict] = []
        self._lock = threading.Lock()

    def __call__(self, row: dict) -> None:
        with self._lock:
            self.rows.append(row)

    def wait(self, n: int = 1, timeout: float = 3.0) -> list[dict]:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._lock:
                if len(self.rows) >= n:
                    return list(self.rows)
            time.sleep(0.01)
        raise AssertionError(f"recorded {len(self.rows)} rows, expected {n}")


# ── FIX 1: the pure gate decision table ─────────────────────────────────────────

def test_gate_chat_route_is_never_gated():
    # chat may legitimately answer with no tool — the gate only ever touches 'action'.
    assert action_honesty_gate("chat", [], already_retried=False) == GATE_EMIT
    assert action_honesty_gate("chat", [], already_retried=True) == GATE_EMIT


def test_gate_action_with_executed_tool_emits():
    calls = [{"name": "home_control", "ok": True, "error_class": None}]
    assert action_honesty_gate("action", calls, already_retried=False) == GATE_EMIT


def test_gate_action_failed_tool_still_counts_as_executed():
    # A tool that RAN and returned an honest error is not a hallucinated action —
    # emptiness is the only thing the gate reacts to.
    calls = [{"name": "home_control", "ok": False, "error_class": "ha_write_failed"}]
    assert action_honesty_gate("action", calls, already_retried=False) == GATE_EMIT


def test_gate_action_zero_tools_first_pass_retries():
    assert action_honesty_gate("action", [], already_retried=False) == GATE_RETRY


def test_gate_action_zero_tools_after_retry_marks():
    assert action_honesty_gate("action", [], already_retried=True) == GATE_MARK


# ── FIX 1: gate wiring through ask() ─────────────────────────────────────────────

class ZeroToolGraph:
    """An action graph that ALWAYS answers in plain text — no tool ever runs. The
    exact hallucinated-action shape: a claimed 'done' that touched nothing."""

    def __init__(self, final="Both office lights are off."):
        self.final = final
        self.calls: list[str] = []

    def invoke(self, inp: dict, config: dict) -> dict:
        self.calls.append(inp["messages"][-1].content)
        return {"messages": [AIMessage(content=self.final)]}


def test_nonvoice_zero_tool_action_bounces_once_then_marks():
    rec = Recorder()
    graph = build_graph(fake_model("unused"), soul="s")
    stub = ZeroToolGraph()
    out = ask(graph, "turn off the office lights", identity=CHRIS, thread_id="t-mark",
              router=action_router, action_graph=stub, record_turn=rec)
    assert out == "Both office lights are off."          # the reply is still emitted
    # bounced exactly once — the second invoke carried the verbatim correction
    assert stub.calls == ["turn off the office lights", ACTION_NO_TOOL_CORRECTION]
    row = rec.wait(1)[0]
    assert NO_TOOL_ACTION_MARKER in json.loads(row["degraded"])  # the pattern is audited
    assert json.loads(row["tool_calls"]) == []                   # nothing ever executed


def test_correction_text_keeps_the_bounce_internal():
    # The anti-fabrication rail must not teach a user-facing confession (owner
    # ask 2026-07-21): honesty about the slip stays internal plumbing.
    lowered = ACTION_NO_TOOL_CORRECTION.lower()
    assert "internal plumbing" in lowered
    assert "never mention" in lowered
    # gap #11: the bounce offers the search_web door before the refusal door
    assert "search_web" in lowered


def test_nonvoice_zero_tool_action_that_calls_a_tool_on_retry_is_clean():
    # The correction WORKS: pass 1 hallucinates "done"; pass 2 (after the corrective)
    # actually calls the tool. The turn emits the honest outcome and carries NO marker.
    rec = Recorder()
    tool = home_tool()
    model = fake_model(
        AIMessage(content="Both office lights are off."),   # pass 1: zero tools -> bounce
        AIMessage(content="", tool_calls=[{                 # pass 2: calls the tool
            "name": "home_control",
            "args": {"operation": "turn_off", "entity_id": "light.desk"},
            "id": "call-1", "type": "tool_call",
        }]),
        "Okay — the desk light is off now.",                # after the tool executes
    )
    action = build_action_graph(model, soul="s", tools=[tool])
    graph = build_graph(fake_model("unused"), soul="s")
    out = ask(graph, "turn off the office lights", identity=CHRIS, thread_id="t-fixed",
              router=action_router, action_graph=action, record_turn=rec)
    assert out == "Okay — the desk light is off now."
    row = rec.wait(1)[0]
    assert NO_TOOL_ACTION_MARKER not in json.loads(row["degraded"])
    assert json.loads(row["tool_calls"])[0]["name"] == "home_control"


def test_voice_zero_tool_action_marks_row_in_background():
    rec = Recorder()
    graph = build_graph(fake_model("speculative"), soul="s")
    stub = ZeroToolGraph(final="Both office lights are on now.")
    out = ask(graph, "turn on the office lights", identity=CHRIS, thread_id="voice:mark",
              router=action_router, action_graph=stub, record_turn=rec)
    assert out == "[warmly] On it"                        # caller hears the ack
    row = rec.wait(1)[0]                                  # fired from the background thread
    assert row["emitted_reply"] == "[warmly] On it"
    assert row["raw_reply"] == "Both office lights are on now."
    assert NO_TOOL_ACTION_MARKER in json.loads(row["degraded"])


# ── FIX 2: the pure rate-limit reply ────────────────────────────────────────────

# The verbatim error the oauth (Max-pool) backend raised on 2026-07-12.
SESSION_LIMIT_ERR = (
    "oauth backend error: subtype='success' "
    "result=\"You've hit your session limit · resets 7:10pm (UTC)\" "
    "num_turns=1 stop_reason='stop_sequence'"
)


def test_rate_limit_reply_parses_and_converts_reset_time():
    # 10am Eastern = 14:00 UTC; the 7:10pm-UTC reset is later today -> 3:10pm Eastern.
    now = datetime(2026, 7, 12, 10, 0, tzinfo=EASTERN)
    reply = rate_limit_reply(SESSION_LIMIT_ERR, now=now)
    assert reply is not None
    assert "rate-limited" in reply and "word budget" in reply
    assert "3:10pm" in reply                              # UTC -> Eastern conversion
    assert "[" not in reply and "]" not in reply         # no emotion tags (safe for text)


def test_rate_limit_reply_missing_time_is_generic():
    reply = rate_limit_reply("You've hit your session limit, try again later.")
    assert reply is not None
    assert "rate-limited" in reply
    assert "a little while" in reply                      # generic tail, no clock claimed


def test_rate_limit_reply_matches_rate_and_usage_limits_too():
    assert rate_limit_reply("429 rate limit exceeded") is not None
    assert rate_limit_reply("monthly usage limit reached") is not None


def test_rate_limit_reply_returns_none_for_other_failures():
    # A generic model failure is NOT converted — the caller keeps re-raising.
    assert rate_limit_reply("oauth backend error: subtype='error' result=None") is None
    assert rate_limit_reply("model 500 internal error") is None
    assert rate_limit_reply("") is None


def test_parse_reset_eastern_direct_and_none():
    now = datetime(2026, 7, 12, 10, 0, tzinfo=EASTERN)
    assert _parse_reset_eastern("resets 7:10pm (UTC)", now=now) == "3:10pm"
    # bare hour, no minutes, defaults to UTC when no zone label is present
    assert _parse_reset_eastern("resets 7pm", now=now) == "3:00pm"
    assert _parse_reset_eastern("no reset time here", now=now) is None


# ── FIX 2: honest reply wiring through ask() (all transports, one seam) ──────────

class RaisingGraph:
    """A chat/action graph whose invoke raises a chosen exception."""

    checkpointer = None

    def __init__(self, exc):
        self._exc = exc

    def invoke(self, _inp, _config):
        raise self._exc

    def get_state(self, _config):
        import types
        return types.SimpleNamespace(values={"messages": []})

    def update_state(self, *_a, **_k):
        pass


def test_rate_limited_chat_turn_emits_honest_reply_not_silence():
    rec = Recorder()
    out = ask(RaisingGraph(RuntimeError(SESSION_LIMIT_ERR)), "Are you sure?",
              identity=CHRIS, thread_id="cli", record_turn=rec)
    assert out.startswith("I'm rate-limited")            # a reply, never silence
    row = rec.wait(1)[0]
    assert row["emitted_reply"] == out                   # what the user heard is recorded
    assert row["raw_reply"] is None                      # the model produced nothing
    assert "turn_failed" in json.loads(row["degraded"])  # the failure stays visible


def test_rate_limited_action_turn_emits_honest_reply():
    rec = Recorder()
    graph = build_graph(fake_model("unused"), soul="s")
    out = ask(graph, "turn off the lights", identity=CHRIS, thread_id="discord:dm:1",
              router=action_router, action_graph=RaisingGraph(RuntimeError(SESSION_LIMIT_ERR)),
              record_turn=rec)
    assert out.startswith("I'm rate-limited")
    row = rec.wait(1)[0]
    assert row["classifier_intent"] == "action"
    assert row["emitted_reply"] == out
    assert "turn_failed" in json.loads(row["degraded"])


def test_non_limit_chat_failure_still_raises_unchanged():
    # A generic failure keeps the historical re-raise (only the rate-limit class is
    # converted). The row still records the failure before propagating.
    rec = Recorder()
    with pytest.raises(RuntimeError):
        ask(RaisingGraph(RuntimeError("model 500 internal error")), "hi",
            identity=CHRIS, thread_id="cli", record_turn=rec)
    row = rec.wait(1)[0]
    assert row["emitted_reply"] is None                  # nothing emitted — it raised
    assert "turn_failed" in json.loads(row["degraded"])
