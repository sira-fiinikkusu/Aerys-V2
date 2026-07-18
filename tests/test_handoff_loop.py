"""Offline tests for the #2 RETURN LOOP — chat→action escalation (owner design,
2026-07-18).

The router classifies from the CURRENT message only, so follow-up-shaped action
requests ("yes, go ahead", "what about tomorrow?") land on the chat path — where
the model, which sees full history, knows the turn needs hands. The chat prompt
has it open such a reply with HANDOFF_MARKER; ask() detects the token and re-runs
the turn on the action graph. These tests prove: text escalation returns the
action outcome and leaves history exactly as if the router had said action; voice
escalation speaks the model's own handoff line as the ack and lands the real
outcome in the background; unarmed surfaces refuse honestly instead of promising;
the marker never survives into emitted text or durable history; and the audit
rows pair up (chat_handoff ↔ escalated_from_chat).
"""

import threading
import time

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from aerys_v2.factory import build_graph
from aerys_v2.router import FALLBACK_ACK, HANDOFF_MARKER, RouteDecision
from aerys_v2.service import HANDOFF_UNARMED_REPLY, ask

CHRIS = {"user_id": "person-1", "display_name": "Chris"}


def fake_model(*replies) -> GenericFakeChatModel:
    msgs = [r if isinstance(r, AIMessage) else AIMessage(content=r) for r in replies]
    return GenericFakeChatModel(messages=iter(msgs))


def chat_router(_text: str) -> RouteDecision:
    return RouteDecision(route="chat", ack="")


class SeedCapturingActionGraph:
    """Stub subgraph that records every full seed it was invoked with."""

    def __init__(self, final: str = "light is off now"):
        self.final = final
        self.seeds: list[list] = []

    def invoke(self, inp: dict, config: dict) -> dict:
        self.seeds.append(list(inp["messages"]))
        return {"messages": [AIMessage(content=self.final)]}


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


def wait_for_messages(graph, thread_id: str, count: int, timeout_s: float = 3.0) -> list:
    deadline = time.monotonic() + timeout_s
    msgs: list = []
    while time.monotonic() < deadline:
        state = graph.get_state({"configurable": {"thread_id": thread_id}})
        msgs = state.values.get("messages", [])
        if len(msgs) >= count:
            return msgs
        time.sleep(0.02)
    raise AssertionError(f"thread never reached {count} messages: {msgs}")


# ---- text path -------------------------------------------------------------------

def test_text_chat_handoff_escalates_to_action():
    graph = build_graph(
        fake_model(f"{HANDOFF_MARKER} Let me actually flip that for you."), soul="s"
    )
    stub = SeedCapturingActionGraph("both display lights are off")
    out = ask(graph, "yes, go ahead", identity=CHRIS, thread_id="t1",
              router=chat_router, action_graph=stub)
    # the caller gets the ACTION outcome, not the handoff line
    assert out == "both display lights are off"
    assert stub.seeds  # the action graph really ran


def test_text_escalated_history_reads_like_a_router_action_route():
    graph = build_graph(fake_model(f"{HANDOFF_MARKER} On it."), soul="s")
    stub = SeedCapturingActionGraph("done — office is dark")
    ask(graph, "turn off the office", identity=CHRIS, thread_id="t1",
        router=chat_router, action_graph=stub)
    msgs = graph.get_state({"configurable": {"thread_id": "t1"}}).values["messages"]
    # EXACTLY human + outcome: the checkpointed handoff line was REPLACED by id,
    # not appended-around — same shape a router action verdict produces.
    assert [m.content for m in msgs] == ["turn off the office", "done — office is dark"]
    assert all(HANDOFF_MARKER not in str(m.content) for m in msgs)


def test_text_escalated_action_seed_ends_on_human_turn_without_marker():
    graph = build_graph(fake_model(f"{HANDOFF_MARKER} Handing off."), soul="s")
    stub = SeedCapturingActionGraph()
    ask(graph, "try it now", identity=CHRIS, thread_id="t1",
        router=chat_router, action_graph=stub)
    seed = stub.seeds[0]
    # the action model reasons from the request, not from a note about handing off
    assert getattr(seed[-1], "type", "") == "human"
    assert seed[-1].content == "try it now"
    assert sum(1 for m in seed if getattr(m, "type", "") == "human") == 1
    assert all(HANDOFF_MARKER not in str(m.content) for m in seed)


def test_text_handoff_audit_rows_pair_up():
    rec = Recorder()
    graph = build_graph(fake_model(f"{HANDOFF_MARKER} On it."), soul="s")
    ask(graph, "yes, go ahead", identity=CHRIS, thread_id="t1",
        router=chat_router, action_graph=SeedCapturingActionGraph(),
        record_turn=rec)
    rows = rec.wait(2)
    chat_row = next(r for r in rows if r.get("classifier_intent") == "chat")
    action_row = next(r for r in rows if r.get("classifier_intent") == "action")
    # the chat row is the receipt a misroute happened; raw keeps the marker
    assert "chat_handoff" in (chat_row.get("degraded") or [])
    assert HANDOFF_MARKER in (chat_row.get("raw_reply") or "")
    # the action row is the recovery, stamped as escalated
    assert "escalated_from_chat" in (action_row.get("degraded") or [])


def test_marker_in_action_final_is_stripped():
    graph = build_graph(fake_model(f"{HANDOFF_MARKER} Handing off."), soul="s")
    stub = SeedCapturingActionGraph(f"{HANDOFF_MARKER} echoed marker, light on")
    out = ask(graph, "lights on", identity=CHRIS, thread_id="t1",
              router=chat_router, action_graph=stub)
    # one-hop belt: an action final can never re-emit a live marker
    assert HANDOFF_MARKER not in out
    assert "light on" in out


def test_plain_chat_reply_unaffected():
    graph = build_graph(fake_model("just talking"), soul="s")
    stub = SeedCapturingActionGraph()
    out = ask(graph, "how are you?", identity=CHRIS, thread_id="t1",
              router=chat_router, action_graph=stub)
    assert out == "just talking"
    assert stub.seeds == []  # no escalation, action never touched


# ---- unarmed surfaces (no action graph to hand to) -------------------------------

def test_chat_only_handoff_refuses_honestly_and_patches_history():
    # no router/action_graph = the chat-only path (dev box, or guest stripped by
    # the allowlist gate with no media graph): a handoff has nowhere to go, so
    # the emitted reply must be an honest refusal — never a dangling promise.
    graph = build_graph(fake_model(f"{HANDOFF_MARKER} Let me grab that."), soul="s")
    out = ask(graph, "turn on the lights", identity=CHRIS, thread_id="t1")
    assert out == HANDOFF_UNARMED_REPLY
    msgs = graph.get_state({"configurable": {"thread_id": "t1"}}).values["messages"]
    # history matches what was actually said — the marker line was patched out
    assert msgs[-1].content == HANDOFF_UNARMED_REPLY
    assert all(HANDOFF_MARKER not in str(m.content) for m in msgs)


def test_guest_handoff_escalates_into_guest_graph_only():
    graph = build_graph(fake_model(f"{HANDOFF_MARKER} Looking at it."), soul="s")
    full = SeedCapturingActionGraph("FULL GRAPH RAN")
    guest = SeedCapturingActionGraph("described the image")
    out = ask(graph, "what's in this picture?", identity=CHRIS, thread_id="t1",
              router=chat_router, action_graph=full,
              guest_action_graph=guest,
              action_allowlist=frozenset({"someone-else"}))
    # the allowlist gate swapped the graphs BEFORE routing — escalation obeys it
    assert out == "described the image"
    assert full.seeds == []
    assert guest.seeds


# ---- voice path ------------------------------------------------------------------

def test_voice_handoff_speaks_own_line_then_action_lands():
    graph = build_graph(
        fake_model(f"{HANDOFF_MARKER} Let me actually check the forecast."), soul="s"
    )
    stub = SeedCapturingActionGraph("rain until 6pm, then clear")
    out = ask(graph, "what about tomorrow?", identity=CHRIS, thread_id="voice:h1",
              router=chat_router, action_graph=stub)
    # the model's OWN handoff line is the immediate spoken ack
    assert out == "Let me actually check the forecast."
    # ...and the real outcome lands in the REAL thread in the background
    msgs = wait_for_messages(graph, "voice:h1", 2)
    assert msgs[-1].content == "rain until 6pm, then clear"
    assert msgs[0].content == "what about tomorrow?"
    assert sum(1 for m in msgs if getattr(m, "type", "") == "human") == 1
    assert all(HANDOFF_MARKER not in str(m.content) for m in msgs)


def test_voice_handoff_marker_only_falls_back_to_stock_ack():
    graph = build_graph(fake_model(HANDOFF_MARKER), soul="s")
    stub = SeedCapturingActionGraph("done")
    out = ask(graph, "go ahead", identity=CHRIS, thread_id="voice:h1",
              router=chat_router, action_graph=stub)
    assert out == FALLBACK_ACK  # never speak a bare token
    wait_for_messages(graph, "voice:h1", 2)


def test_voice_handoff_delivers_spoken_followup():
    spoken: list[tuple[str, str]] = []
    graph = build_graph(fake_model(f"{HANDOFF_MARKER} Checking."), soul="s")
    stub = SeedCapturingActionGraph("72 and sunny")  # no tool notes -> must speak
    ask(graph, "what's it like out?", identity=CHRIS, thread_id="voice:h1",
        router=chat_router, action_graph=stub,
        speak_fn=lambda text, entity: spoken.append((text, entity)),
        satellite_for=lambda _device: "assist_satellite.office")
    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline and not spoken:
        time.sleep(0.02)
    assert spoken == [("72 and sunny", "assist_satellite.office")]


def test_voice_handoff_audit_row_is_escalated_action():
    rec = Recorder()
    graph = build_graph(fake_model(f"{HANDOFF_MARKER} On it."), soul="s")
    ask(graph, "yes, do it", identity=CHRIS, thread_id="voice:h1",
        router=chat_router, action_graph=SeedCapturingActionGraph(),
        record_turn=rec)
    rows = rec.wait(1)
    action_row = next(r for r in rows if r.get("classifier_intent") == "action")
    assert "escalated_from_chat" in (action_row.get("degraded") or [])
    # emitted = the handoff line the caller heard; raw = the action's real outcome
    assert action_row.get("emitted_reply") == "On it."
    assert action_row.get("raw_reply") == "light is off now"
