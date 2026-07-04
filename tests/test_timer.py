"""Offline tests for the timer tool — fake HA (httpx.MockTransport), no live HA.

What these prove: natural-duration parsing (words → seconds), START/CANCEL drive
HA's NATIVE assist-timer intent (HassStartTimer/HassCancelTimer) at
/api/intent/handle carrying the ORIGINATING device_id (read from injected config,
never guessed), a successful CANCEL leads with WRITE_OK_PREFIX so service.py's
silent-success rule skips a spoken follow-up, while a successful START speaks the
duration back (the LED ring is VPE-only — a spoken confirm is the only fleet-wide
feedback), every failure path returns an HONEST string and NEVER
raises (a raise inside a ToolNode kills the whole action turn), and the no-device
text/DM path degrades gracefully — a configured fallback helper or an honest
"I can't set a device timer from here", never a silent pretend-success.
"""

import json

import httpx
import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage

from aerys_v2.factory import build_action_graph
from aerys_v2.tools.home_control import WRITE_OK_PREFIX
from aerys_v2.tools.timer import (
    build_timer_tool,
    describe_duration,
    parse_duration,
)

CHRIS = {"user_id": "person-1", "display_name": "Chris"}


# ---- fakes ---------------------------------------------------------------------

class FakeHA:
    """Records every request (method, path, json body); scripts HA's responses."""

    def __init__(self, *, intent_response: dict | None = None, fail: bool = False):
        self.requests: list[tuple[str, str, dict]] = []
        # default: HassStartTimer/Cancel success — action_done, no speech
        self.intent_response = intent_response or {
            "response_type": "action_done", "speech": {}, "data": {}
        }
        self.fail = fail

    def handler(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content) if request.content else {}
        self.requests.append((request.method, request.url.path, body))
        if self.fail:
            return httpx.Response(503, text="ha melted")
        if request.url.path == "/api/intent/handle":
            return httpx.Response(200, json=self.intent_response)
        if request.url.path.startswith("/api/services/timer/"):
            return httpx.Response(200, json=[])
        return httpx.Response(404)

    def client(self) -> httpx.Client:
        return httpx.Client(transport=httpx.MockTransport(self.handler))


def make_tool(ha: FakeHA, fallback_entity: str | None = None):
    return build_timer_tool(
        base_url="http://ha.test:8123",
        token="t0ken",
        client=ha.client(),
        fallback_entity=fallback_entity,
    )


def cfg(device_id: str | None = None) -> dict:
    """A RunnableConfig carrying the per-call identity (with/without device_id)."""
    identity = dict(CHRIS)
    if device_id is not None:
        identity["device_id"] = device_id
    return {"configurable": {"identity": identity}}


ERROR_RESP = {
    "response_type": "error",
    "speech": {"plain": {"speech": "Device does not support timers"}},
    "data": {"code": "failed_to_handle"},
}


# ---- parse_duration: natural language -> seconds --------------------------------

@pytest.mark.parametrize(
    "text,expected",
    [
        ("5 minutes", 300),
        ("5 min", 300),
        ("90 seconds", 90),
        ("90 secs", 90),
        ("1 hour 30 minutes", 5400),
        ("1h30m", 5400),
        ("1.5 hours", 5400),
        ("an hour", 3600),
        ("a minute", 60),
        ("half an hour", 1800),
        ("half a minute", 30),
        ("an hour and a half", 5400),
        ("quarter of an hour", 900),
        ("10", 600),          # bare number = minutes (the common shorthand)
        ("2 hours", 7200),
        ("45 s", 45),
        # DIGIT-form fractional tails must be summed, not silently truncated to
        # the leading whole unit (regression: '1 hour and a half' was 3600).
        ("1 hour and a half", 5400),
        ("2 hours and a half", 9000),
        ("1 hour and a quarter", 4500),
        ("5 minutes and a half", 330),
    ],
)
def test_parse_duration_recognizes_natural_forms(text, expected):
    assert parse_duration(text) == expected


@pytest.mark.parametrize(
    "text",
    ["human minutes", "roman hours", "german seconds"],
)
def test_parse_duration_ignores_article_glued_inside_a_word(text):
    # 'an'/'a' must be a standalone word to count as qty=1 — the 'an' buried in
    # 'human' must NOT glue to a following unit and fabricate a 60s timer.
    assert parse_duration(text) is None


@pytest.mark.parametrize("text", ["", "   ", "banana", "soon", "0 minutes", "0"])
def test_parse_duration_returns_none_when_no_duration(text):
    assert parse_duration(text) is None


def test_describe_duration_reads_naturally():
    assert describe_duration(300) == "5 minutes"
    assert describe_duration(60) == "1 minute"
    assert describe_duration(5400) == "1 hour 30 minutes"
    assert describe_duration(3600) == "1 hour"
    assert describe_duration(45) == "45 seconds"


# ---- START on the originating device (native intent, LED wheel) -----------------

def test_start_targets_native_intent_on_device_and_omits_zero_slots():
    ha = FakeHA()
    out = make_tool(ha).invoke({"action": "start", "duration": "5 minutes"}, cfg("dev-pe"))
    assert not out.startswith(WRITE_OK_PREFIX)          # B: START speaks (no silent-success)
    assert "5 minutes" in out                            # duration spoken back, catchable by ear
    [(method, path, body)] = ha.requests
    assert (method, path) == ("POST", "/api/intent/handle")
    # native assist timer, carrying the originating device_id, minutes-only slot
    assert body == {"name": "HassStartTimer", "data": {"minutes": 5}, "device_id": "dev-pe"}


def test_start_splits_compound_duration_into_slots():
    ha = FakeHA()
    make_tool(ha).invoke({"action": "start", "duration": "1 hour 30 minutes"}, cfg("dev-pe"))
    assert ha.requests[0][2]["data"] == {"hours": 1, "minutes": 30}


def test_start_passes_optional_name_slot():
    ha = FakeHA()
    out = make_tool(ha).invoke(
        {"action": "start", "duration": "10 minutes", "name": "pasta"}, cfg("dev-pe")
    )
    assert ha.requests[0][2]["data"] == {"minutes": 10, "name": "pasta"}
    assert "'pasta'" in out


def test_start_unparseable_duration_is_honest_and_never_calls_ha():
    ha = FakeHA()
    out = make_tool(ha).invoke({"action": "start", "duration": "sometime soon"}, cfg("dev-pe"))
    assert "couldn't tell how long" in out
    assert ha.requests == []  # nothing sent to HA on a parse failure


def test_start_device_not_timer_capable_relays_ha_error_not_done():
    # a device_id that isn't a registered timer device -> HA returns an error
    # IntentResponse; the tool relays it honestly and does NOT claim success.
    ha = FakeHA(intent_response=ERROR_RESP)
    out = make_tool(ha).invoke({"action": "start", "duration": "5 minutes"}, cfg("dev-desk"))
    assert not out.startswith(WRITE_OK_PREFIX)
    assert "Device does not support timers" in out
    assert "couldn't start" in out


def test_start_ha_unreachable_is_honest_string_not_exception():
    out = make_tool(FakeHA(fail=True)).invoke(
        {"action": "start", "duration": "5 minutes"}, cfg("dev-pe")
    )
    assert not out.startswith(WRITE_OK_PREFIX)
    assert "unreachable" in out


# ---- START with NO device (text/DM) — graceful degrade --------------------------

def test_start_no_device_no_fallback_is_honest_refusal():
    ha = FakeHA()
    out = make_tool(ha).invoke({"action": "start", "duration": "5 minutes"}, cfg(None))
    assert "can't tell which device" in out or "which device you're on" in out
    assert not out.startswith(WRITE_OK_PREFIX)  # NOT a silent success — must be spoken
    assert ha.requests == []  # never touched HA


def test_start_no_device_with_fallback_uses_generic_helper_and_is_honest():
    ha = FakeHA()
    out = make_tool(ha, fallback_entity="timer.aerys_fallback").invoke(
        {"action": "start", "duration": "5 minutes"}, cfg(None)
    )
    # best-effort generic timer helper via the timer.start SERVICE (no LED wheel)
    [(method, path, body)] = ha.requests
    assert (method, path) == ("POST", "/api/services/timer/start")
    assert body == {"entity_id": "timer.aerys_fallback", "duration": "00:05:00"}
    # honest: says it can't ring on a speaker / show a light, and is NOT a "Done:"
    assert not out.startswith(WRITE_OK_PREFIX)
    assert "background" in out and ("speaker" in out or "light" in out)


# ---- CANCEL ---------------------------------------------------------------------

def test_cancel_on_device_hits_native_cancel_intent():
    ha = FakeHA()
    out = make_tool(ha).invoke({"action": "cancel"}, cfg("dev-pe"))
    assert out.startswith(WRITE_OK_PREFIX)
    [(method, path, body)] = ha.requests
    assert (method, path) == ("POST", "/api/intent/handle")
    assert body == {"name": "HassCancelTimer", "data": {}, "device_id": "dev-pe"}


def test_cancel_with_name_passes_name_slot():
    ha = FakeHA()
    make_tool(ha).invoke({"action": "cancel", "name": "pasta"}, cfg("dev-pe"))
    assert ha.requests[0][2]["data"] == {"name": "pasta"}


def test_cancel_no_matching_timer_relays_ha_error():
    ha = FakeHA(intent_response={
        "response_type": "error",
        "speech": {"plain": {"speech": "There are no timers running"}},
        "data": {"code": "no_timer"},
    })
    out = make_tool(ha).invoke({"action": "cancel"}, cfg("dev-pe"))
    assert not out.startswith(WRITE_OK_PREFIX)
    assert "no timers running" in out


def test_cancel_no_device_no_fallback_is_honest():
    ha = FakeHA()
    out = make_tool(ha).invoke({"action": "cancel"}, cfg(None))
    assert "no device timer" in out or "which voice device" in out
    assert ha.requests == []


def test_cancel_no_device_with_fallback_cancels_generic_helper():
    ha = FakeHA()
    out = make_tool(ha, fallback_entity="timer.aerys_fallback").invoke(
        {"action": "cancel"}, cfg(None)
    )
    [(method, path, body)] = ha.requests
    assert (method, path) == ("POST", "/api/services/timer/cancel")
    assert body == {"entity_id": "timer.aerys_fallback"}
    assert "background timer" in out


# ---- robustness: unknown action + never-raise -----------------------------------

def test_unknown_action_is_honest_string():
    out = make_tool(FakeHA()).invoke({"action": "pause", "duration": "5 minutes"}, cfg("dev-pe"))
    assert "Unknown timer action" in out


def test_tool_never_raises_on_transport_error():
    def boom(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no route to ha")

    tool = build_timer_tool(
        base_url="http://ha.test:8123",
        token="t0ken",
        client=httpx.Client(transport=httpx.MockTransport(boom)),
    )
    # an exception inside a ToolNode kills the whole turn — so this MUST be a string
    out = tool.invoke({"action": "start", "duration": "5 minutes"}, cfg("dev-pe"))
    assert "unreachable" in out
    assert not out.startswith(WRITE_OK_PREFIX)


@pytest.mark.parametrize("bad_body", [[], None])
def test_start_non_dict_intent_body_is_honest_not_exception(bad_body):
    # HA returning a 200 whose body is a bare list or literal null must NOT raise
    # AttributeError out of the tool (resp.get on a list/None) — contract #1:
    # never raise inside a ToolNode. It degrades to an honest failure string.
    ha = FakeHA()
    ha.intent_response = bad_body  # set directly: '' / [] / None are falsy, bypass ctor default
    out = make_tool(ha).invoke({"action": "start", "duration": "5 minutes"}, cfg("dev-pe"))
    assert isinstance(out, str)
    assert not out.startswith(WRITE_OK_PREFIX)
    assert "couldn't start" in out


def test_missing_config_degrades_like_no_device():
    # no config at all (identity_from_config -> UNKNOWN_CALLER, no device_id):
    # the tool must still return a string, never raise a KeyError.
    out = make_tool(FakeHA()).invoke({"action": "start", "duration": "5 minutes"})
    assert isinstance(out, str) and "which device" in out


# ---- integration: device_id flows config -> ToolNode -> native intent -----------

def test_action_graph_wires_device_id_from_config_into_timer_intent():
    """The load-bearing wiring: the compiled action subgraph threads the per-call
    identity (with device_id) through ToolNode into the timer tool, which targets
    the originating satellite's native timer — the whole point of the feature."""
    ha = FakeHA()
    tool = make_tool(ha)
    model = GenericFakeChatModel(messages=iter([
        AIMessage(content="", tool_calls=[{
            "name": "timer",
            "args": {"action": "start", "duration": "10 minutes"},
            "id": "call-1",
            "type": "tool_call",
        }]),
        AIMessage(content="Ten-minute timer's going."),
    ]))
    graph = build_action_graph(model, soul="s", tools=[tool])
    identity = {**CHRIS, "device_id": "dev-office-pe"}
    result = graph.invoke(
        {"messages": [HumanMessage(content="set a 10 minute timer")]},
        {"configurable": {"thread_id": "voice:beta", "identity": identity},
         "recursion_limit": 10},
    )
    assert result["messages"][-1].content == "Ten-minute timer's going."
    # the tool actually fired the native intent on the ORIGINATING device
    [(method, path, body)] = ha.requests
    assert (method, path) == ("POST", "/api/intent/handle")
    assert body == {
        "name": "HassStartTimer", "data": {"minutes": 10}, "device_id": "dev-office-pe"
    }
    # and the ToolMessage speaks the duration back (B: no silent-success on START —
    # the LED ring is VPE-only, so the spoken confirm is the only fleet-wide feedback)
    tool_note = next(m.content for m in result["messages"] if getattr(m, "type", "") == "tool")
    assert not tool_note.startswith(WRITE_OK_PREFIX)
    assert "10 minutes" in tool_note
