"""Offline tests for the v2_turns writer — pure helpers + the ask() recording seam.

No DB, no network: the recorder is a plain capturing callable injected into ask()
(same seam philosophy as the checkpointer / speak_fn / router). What these prove:
one row per completed turn on EVERY path (chat, action, voice-chat, voice-action,
deep-cap downgrade, timeout), tool_calls is STRUCTURED (list of {name, ok,
error_class} — load-bearing for the capability loop, never a prose string),
degraded is a structured marker list, the row lands OFF the hot path (a daemon
thread the transport never waits on), and a failing recorder never touches the turn.
"""

import json
import logging
import threading
import time
import types

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from aerys_v2.factory import build_graph, turn_recorder_for
from aerys_v2.router import RouteDecision
from aerys_v2.service import TurnTimeout, ask
from aerys_v2.turns import (
    build_turn_row,
    classify_tool_result,
    degraded_markers,
    derive_channel,
    extract_tool_calls,
)

# a resolved person = a real UUID (lands in person_id); a cold caller is a handle.
OWNER = {"user_id": "6e6bcbed-03ef-4d17-95d2-89c467414335", "display_name": "Chris"}
COLD = {"user_id": "discord:60426", "display_name": "Nosy Guildmate"}


def fake_model(*replies) -> GenericFakeChatModel:
    msgs = [r if isinstance(r, AIMessage) else AIMessage(content=r) for r in replies]
    return GenericFakeChatModel(messages=iter(msgs))


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


def chat_router(_t: str) -> RouteDecision:
    return RouteDecision(route="chat", ack="")


def action_router(_t: str) -> RouteDecision:
    return RouteDecision(route="action", ack="[warmly] On it")


def deep_chat_router(_t: str) -> RouteDecision:
    return RouteDecision(route="chat", ack="", tier="deep")


class StubActionGraph:
    def __init__(self, final="the light is on now"):
        self.final = final

    def invoke(self, inp, config):
        return {"messages": [AIMessage(content=self.final)]}


class ToolTraceActionGraph:
    """Returns a real-shaped subgraph trace: AI(tool_calls) -> ToolMessages -> AI."""

    def __init__(self, notes, final="all set"):
        self.notes = notes  # list of (name, content, status)
        self.final = final

    def invoke(self, inp, config):
        tool_calls = [
            {"name": n, "args": {}, "id": f"c{i}", "type": "tool_call"}
            for i, (n, _c, _s) in enumerate(self.notes)
        ]
        msgs = [AIMessage(content="", tool_calls=tool_calls)]
        for i, (name, content, status) in enumerate(self.notes):
            msgs.append(
                ToolMessage(content=content, tool_call_id=f"c{i}", name=name, status=status)
            )
        msgs.append(AIMessage(content=self.final))
        return {"messages": [*inp["messages"], *msgs]}


# ── pure helpers ────────────────────────────────────────────────────────────

def test_derive_channel_covers_every_transport_prefix():
    assert derive_channel("voice:beta") == "voice"
    assert derive_channel("discord:dm:60426") == "discord_dm"
    assert derive_channel("discord:guild:123") == "guild"
    assert derive_channel("telegram:dm:9") == "telegram_dm"
    assert derive_channel("telegram:group:9") == "telegram_group"
    assert derive_channel("cli") == "cli"
    assert derive_channel("http:default") == "http"
    # NOT NULL column — unknown shapes still return SOMETHING, never crash
    assert derive_channel("weird:thing") == "weird"
    assert derive_channel("") == "unknown"


def test_classify_tool_result():
    assert classify_tool_result("Done: turn_on sent to light.desk", "success") is None
    # a policy refusal is NOT a failure — the tool did its job
    assert classify_tool_result("Refused: not on the allowlist.", "success") is None
    # ToolNode caught a raise = highest-trust failure signal
    assert classify_tool_result("boom", "error") == "exception"
    # never-raise tools' honest failure strings map to an error_class
    assert classify_tool_result("Home Assistant is unreachable right now (x).", "success") == "unreachable"
    assert classify_tool_result("web search failed: HTTP 500.", "success") == "search_failed"


def test_extract_tool_calls_is_structured_not_prose():
    graph = ToolTraceActionGraph(
        notes=[
            ("home_control", "Done: turn_on sent to light.desk", "success"),
            ("search_web", "web search failed: the search service returned HTTP 500.", "success"),
        ]
    )
    messages = graph.invoke({"messages": [HumanMessage(content="x")]}, {})["messages"]
    calls = extract_tool_calls(messages)
    assert calls == [
        {"name": "home_control", "ok": True, "error_class": None},
        {"name": "search_web", "ok": False, "error_class": "search_failed"},
    ]
    # every entry is a dict with exactly the fingerprint keys the loop mines
    for c in calls:
        assert set(c) == {"name", "ok", "error_class"}
        assert isinstance(c["ok"], bool)


def test_degraded_markers_from_ha_unreachable_and_extra():
    graph = ToolTraceActionGraph(
        notes=[("home_control", "Home Assistant is unreachable right now (timeout).", "success")]
    )
    messages = graph.invoke({"messages": [HumanMessage(content="x")]}, {})["messages"]
    markers = degraded_markers(messages, extra=["deep_cap_downgraded"])
    # caller-supplied marker leads; the mined ha marker follows; no dupes
    assert markers == ["deep_cap_downgraded", "ha_unreachable"]


def test_build_turn_row_person_vs_platform_and_json_shape():
    row = build_turn_row(
        thread_id="discord:guild:1", identity=OWNER, input_text="hi",
        latency_ms=12, classifier_intent="chat", tier="standard",
        raw_reply="hey", emitted_reply="hey", messages=[],
    )
    assert row["channel"] == "guild"
    assert row["person_id"] == OWNER["user_id"]      # UUID -> person_id
    assert row["platform_identity"] is None
    # JSONB columns arrive as JSON STRINGS ready for ::jsonb, not python lists
    assert row["tool_calls"] == "[]" and json.loads(row["tool_calls"]) == []
    assert row["degraded"] == "[]"

    cold = build_turn_row(
        thread_id="discord:guild:1", identity=COLD, input_text="hi",
        latency_ms=1, messages=[],
    )
    assert cold["person_id"] is None                 # non-UUID -> NOT person_id
    assert cold["platform_identity"] == "discord:60426"


# ── ask() recording, per completion path ────────────────────────────────────

def test_chat_only_path_records_one_clean_row():
    rec = Recorder()
    graph = build_graph(fake_model("hello there"), soul="s")
    out = ask(graph, "hi", identity=OWNER, thread_id="cli", record_turn=rec)
    assert out == "hello there"
    (row,) = rec.wait(1)
    assert row["channel"] == "cli"
    assert row["input_text"] == "hi"
    assert row["raw_reply"] == "hello there" == row["emitted_reply"]
    assert row["classifier_intent"] is None          # no router ran on this path
    assert row["tool_calls"] == "[]" and row["degraded"] == "[]"
    assert row["error"] is None and isinstance(row["latency_ms"], int)


def test_nonvoice_action_path_records_structured_tool_calls():
    rec = Recorder()
    graph = build_graph(fake_model("unused"), soul="s")
    action = ToolTraceActionGraph(
        notes=[("home_control", "Done: turn_on sent to light.desk", "success")],
        final="Desk light's on.",
    )
    out = ask(graph, "turn on the desk light", identity=OWNER, thread_id="discord:dm:1",
              router=action_router, action_graph=action, record_turn=rec)
    assert out == "Desk light's on."
    (row,) = rec.wait(1)
    assert row["channel"] == "discord_dm"
    assert row["classifier_intent"] == "action"
    assert row["raw_reply"] == "Desk light's on." == row["emitted_reply"]
    assert json.loads(row["tool_calls"]) == [
        {"name": "home_control", "ok": True, "error_class": None}
    ]


def test_deep_cap_downgrade_is_recorded():
    rec = Recorder()
    graph = build_graph(fake_model("deep-ish answer"), soul="s")
    ask(graph, "analyze this deeply", identity=OWNER, thread_id="discord:dm:1",
        router=deep_chat_router, action_graph=StubActionGraph(),
        deep_allowed=lambda: False, record_turn=rec)
    (row,) = rec.wait(1)
    assert row["tier"] == "standard"                 # served tier, downgraded
    assert row["tier_override_source"] == "deep_cap"
    assert json.loads(row["degraded"]) == ["deep_cap_downgraded"]


def test_voice_chat_path_records_standard_pinned_row():
    rec = Recorder()
    graph = build_graph(fake_model("spoken reply"), soul="s")
    out = ask(graph, "say something nice", identity=OWNER, thread_id="voice:beta",
              router=chat_router, action_graph=StubActionGraph(), record_turn=rec)
    assert out == "spoken reply"
    (row,) = rec.wait(1)
    assert row["channel"] == "voice"
    assert row["classifier_intent"] == "chat"
    assert row["tier"] == "standard"                 # ChannelPolicy pin


def test_person_keyed_voice_turn_still_audits_as_voice_channel():
    # Post person-keying, a voice turn rides the owner's 'person:{id}' thread — which
    # derive_channel would mislabel 'person'. The explicit voice flag keeps the audit
    # row's channel 'voice' (and the standard-tier pin still fires off that flag).
    rec = Recorder()
    graph = build_graph(fake_model("spoken reply"), soul="s")
    voice_owner = {**OWNER, "voice": True}
    out = ask(graph, "say something nice", identity=voice_owner,
              thread_id="person:6e6bcbed-03ef-4d17-95d2-89c467414335",
              router=chat_router, action_graph=StubActionGraph(), record_turn=rec)
    assert out == "spoken reply"
    (row,) = rec.wait(1)
    assert row["channel"] == "voice"                 # NOT 'person' — the flag wins
    assert row["tier"] == "standard"                 # ChannelPolicy pin off the flag


def test_voice_action_path_records_ack_as_emitted_and_final_as_raw():
    rec = Recorder()
    graph = build_graph(fake_model("speculative"), soul="s")
    action = ToolTraceActionGraph(
        notes=[("home_control", "Done: turn_off sent to light.desk", "success")],
        final="Light's off.",
    )
    out = ask(graph, "turn off the desk light", identity=OWNER, thread_id="voice:beta",
              router=action_router, action_graph=action, record_turn=rec)
    assert out == "[warmly] On it"                   # the caller heard the ack
    (row,) = rec.wait(1)                              # fired from the background thread
    assert row["emitted_reply"] == "[warmly] On it"  # what went to the channel first
    assert row["raw_reply"] == "Light's off."        # the action's real outcome
    assert row["classifier_intent"] == "action"
    assert json.loads(row["tool_calls"])[0]["name"] == "home_control"


def test_voice_action_failure_records_error_and_marker():
    rec = Recorder()

    class Exploding:
        def invoke(self, inp, config):
            raise RuntimeError("HA melted")

    graph = build_graph(fake_model("speculative"), soul="s")
    ask(graph, "toggle the desk light", identity=OWNER, thread_id="voice:x",
        router=action_router, action_graph=Exploding(), record_turn=rec)
    (row,) = rec.wait(1)
    assert row["error"] and "didn't complete" in row["error"]
    assert json.loads(row["degraded"]) == ["action_failed"]


def test_timeout_path_still_records_before_raising():
    rec = Recorder()
    from aerys_v2.service import Rails

    graph = build_graph(fake_model("late reply"), soul="s")
    with pytest.raises(TurnTimeout):
        ask(graph, "hi", identity=OWNER, thread_id="cli",
            rails=Rails(wall_clock_s=0.0), record_turn=rec)
    (row,) = rec.wait(1)
    assert row["raw_reply"] == "late reply"           # the reply existed
    assert row["error"] and "budget" in row["error"]
    assert "wall_clock_exceeded" in json.loads(row["degraded"])


# ── fail-open + off-the-hot-path ────────────────────────────────────────────

def test_failing_recorder_never_breaks_the_turn():
    def boom(_row):
        raise OSError("NAS down")

    graph = build_graph(fake_model("still works"), soul="s")
    # the turn returns normally even though every audit write raises
    assert ask(graph, "hi", identity=OWNER, thread_id="cli", record_turn=boom) == "still works"


def test_recorder_factory_is_fail_open_on_db_error(monkeypatch):
    import psycopg

    def explode(*_a, **_k):
        raise OSError("NAS unreachable")

    monkeypatch.setattr(psycopg, "connect", explode)
    fake_settings = types.SimpleNamespace(
        database_url="postgresql://sira:x@nas:5432/aerys_v2"
    )
    record = turn_recorder_for(fake_settings)  # type: ignore[arg-type]
    assert record is not None
    # a dead NAS logs and returns — it must NOT raise into the caller
    record({"thread_id": "cli", "channel": "cli"})


def test_recorder_factory_none_without_database_url():
    fake_settings = types.SimpleNamespace(database_url=None)
    assert turn_recorder_for(fake_settings) is None  # type: ignore[arg-type]


def test_record_off_hot_path_does_not_block_return():
    # A recorder that blocks for a long time must not delay the reply — the write
    # rides a daemon thread the transport never joins.
    started = threading.Event()

    def slow(_row):
        started.set()
        time.sleep(5.0)  # would blow the test budget if it were on the hot path

    graph = build_graph(fake_model("fast reply"), soul="s")
    t0 = time.monotonic()
    out = ask(graph, "hi", identity=OWNER, thread_id="cli", record_turn=slow)
    assert out == "fast reply"
    assert time.monotonic() - t0 < 2.0               # returned without waiting on the write
    assert started.wait(2.0)                          # the write DID start, in the background


def test_audit_thread_spawn_failure_is_swallowed(monkeypatch):
    # cross-review hotpath H: under thread exhaustion Thread.start() raises RuntimeError.
    # That spawn was the ONE audit-path line outside a try/except — it must be caught
    # inside _fire_turn_record so it can never unwind into the live turn. (Patching
    # Thread globally would also break langchain's invoke, so exercise the seam directly.)
    import aerys_v2.service as svc

    def boom_thread(*_a, **_k):
        raise RuntimeError("can't start new thread")

    monkeypatch.setattr(svc.threading, "Thread", boom_thread)
    rec = Recorder()
    config = {"configurable": {"thread_id": "cli", "identity": OWNER}}
    # the spawn raises internally; _fire_turn_record MUST swallow it and return
    svc._fire_turn_record(rec, config, "hi", 5, raw_reply="x", emitted_reply="x")
    assert rec.rows == []                             # thread never ran, but nothing raised


def test_audit_over_inflight_cap_drops_instead_of_piling_up(monkeypatch):
    # cross-review hotpath M: when the in-flight fuse is blown (NAS slow/down) the write
    # is DROPPED, not queued — bounding threads/connections so audit can't starve the
    # hot path's own DB. Swap in a pre-drained fuse so the cap is hit deterministically
    # (mutating the shared module semaphore would race the other tests' writer threads).
    import aerys_v2.service as svc

    drained = threading.BoundedSemaphore(1)
    assert drained.acquire(blocking=False)            # now the only permit is gone
    monkeypatch.setattr(svc, "_audit_inflight", drained)

    rec = Recorder()
    config = {"configurable": {"thread_id": "cli", "identity": OWNER}}
    svc._fire_turn_record(rec, config, "hi", 5, raw_reply="x", emitted_reply="x")
    assert rec.rows == []                             # dropped, no thread spawned


# ── error exits still audit exactly one row (cross-review correctness H) ──────

class _RaisingGraph:
    """Chat graph whose invoke raises — a model 500 / recursion-rail trip."""

    def __init__(self, exc):
        self._exc = exc

    def invoke(self, _inp, _config):
        raise self._exc


class _RaisingVoiceGraph(_RaisingGraph):
    """Voice needs the speculative-thread seam (get_state/update_state/checkpointer)
    around an invoke that still raises."""

    checkpointer = None

    def get_state(self, _config):
        return types.SimpleNamespace(values={"messages": []})

    def update_state(self, *_a, **_k):
        pass


def test_chat_raise_records_failure_row_before_reraising():
    rec = Recorder()
    with pytest.raises(RuntimeError):
        ask(_RaisingGraph(RuntimeError("model 500")), "hi",
            identity=OWNER, thread_id="cli", record_turn=rec)
    (row,) = rec.wait(1)
    assert row["error"] and "model 500" in row["error"]
    assert "turn_failed" in json.loads(row["degraded"])
    assert row["raw_reply"] is None                   # no reply ever existed


def test_nonvoice_action_raise_records_failure_row():
    rec = Recorder()
    graph = build_graph(fake_model("unused"), soul="s")
    with pytest.raises(RuntimeError):
        ask(graph, "do a thing", identity=OWNER, thread_id="discord:dm:1",
            router=action_router, action_graph=_RaisingGraph(RuntimeError("tool loop blew up")),
            record_turn=rec)
    (row,) = rec.wait(1)
    assert row["classifier_intent"] == "action"
    assert row["error"] and "blew up" in row["error"]
    assert "turn_failed" in json.loads(row["degraded"])


def test_voice_chat_raise_records_failure_row():
    rec = Recorder()
    with pytest.raises(RuntimeError):
        ask(_RaisingVoiceGraph(RuntimeError("voice model 500")), "say hi",
            identity=OWNER, thread_id="voice:beta",
            router=chat_router, action_graph=StubActionGraph(), record_turn=rec)
    (row,) = rec.wait(1)
    assert row["channel"] == "voice"
    assert row["tier"] == "standard"                  # pinned even on the error exit
    assert "turn_failed" in json.loads(row["degraded"])


# ── provenance: a success payload can't forge a failure (sharp-3 H) ──────────

def test_successful_multiline_search_echoing_failure_phrase_is_ok():
    # A web result snippet literally containing 'web search failed' must NOT be
    # recorded as a failed call — only the tool's OWN single-line failure string,
    # which leads the content, may. The echo rides on line 2+, so it can't match.
    content = (
        "Sources:\n"
        "1. Debugging n8n | https://x | a blog post titled 'web search failed' about 500s"
    )
    assert classify_tool_result(content, "success") is None
    graph = ToolTraceActionGraph(notes=[("search_web", content, "success")])
    messages = graph.invoke({"messages": [HumanMessage(content="x")]}, {})["messages"]
    assert extract_tool_calls(messages) == [
        {"name": "search_web", "ok": True, "error_class": None}
    ]


def test_successful_document_body_echoing_failure_phrase_is_ok():
    # read_document success prefixes 'Contents of <file>:' — attacker text in the
    # body can echo any sentinel and still not forge a failure.
    content = "Contents of report.pdf:\n\nThe vendor couldn't fetch the document, they wrote."
    assert classify_tool_result(content, "success") is None


# ── contract: every tool failure string still maps to an error_class (sharp-3 H) ──
# If a tool reword drifts out of the sentinel table, a REAL failure would silently
# record as ok:true — this test breaks CI on that drift instead.
_TOOL_FAILURE_STRINGS = [
    "The turn_on on light.desk FAILED — Home Assistant said: 500.",
    "Home Assistant has no entity named light.foo.",
    "Home Assistant is unreachable right now (timeout).",
    "The vision service is unreachable right now (ConnectError).",
    "The vision call came back malformed: KeyError('choices')",
    "Couldn't fetch the document (ConnectTimeout).",
    "Fetching the document failed with HTTP 500.",
    "Fetched report.pdf but couldn't extract its text (BadZipFile).",
    "The summarization service is unreachable right now (ConnectError).",
    "Summarizing the transcript failed with HTTP 503.",
    "The summarization call came back malformed: KeyError",
    "web search failed: the search service is unreachable right now (x).",
    "web search failed: the search service returned HTTP 500.",
    "web search failed: the search service returned a malformed response.",
]


@pytest.mark.parametrize("failure_string", _TOOL_FAILURE_STRINGS)
def test_every_tool_failure_string_maps_to_a_non_null_error_class(failure_string):
    assert classify_tool_result(failure_string, "success") is not None


def test_exception_status_refines_cause_beyond_bare_exception():
    # A genuine raise carries a machine-set message — fingerprint the cause distinctly
    # so a timeout and an auth error don't merge into one 'exception' (sharp-3 M).
    assert classify_tool_result("HTTPSConnectionPool: Read timed out.", "error") == "timeout"
    assert classify_tool_result("ConnectionRefusedError: Connection refused", "error") == "unreachable"
    assert classify_tool_result("401 Unauthorized", "error") == "auth_error"
    assert classify_tool_result("something wholly unexpected", "error") == "exception"


def test_ha_write_rejection_is_distinct_from_unreachable():
    # A reachable-but-refusing HA (4xx) is a permissions gap, not a connectivity one —
    # its degraded marker must NOT collapse into 'ha_unreachable' (sharp-3 L).
    note = "The turn_on on light.desk FAILED — Home Assistant said: 403 Forbidden."
    graph = ToolTraceActionGraph(notes=[("home_control", note, "success")])
    messages = graph.invoke({"messages": [HumanMessage(content="x")]}, {})["messages"]
    assert extract_tool_calls(messages)[0]["error_class"] == "ha_write_failed"
    assert degraded_markers(messages) == ["ha_write_failed"]


def test_unresolved_tool_name_logs_a_warning(caplog):
    # A ToolMessage with no .name and an id absent from the AIMessage map collapses to
    # 'unknown' — that blind fingerprint merge must be visible in logs (sharp-3 L).
    orphan = ToolMessage(content="web search failed: x", tool_call_id="not-in-any-map")
    with caplog.at_level(logging.WARNING):
        calls = extract_tool_calls([orphan])
    assert calls[0]["name"] == "unknown"
    assert "unresolved" in caplog.text.lower()
