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
import json
import time

import pytest

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from aerys_v2.factory import build_graph
from aerys_v2.router import FALLBACK_ACK, HANDOFF_MARKER, RouteDecision
from aerys_v2.service import VOICE_EMPTY_REPLY, claims_effect_done
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

def test_voice_chat_verdict_runs_action_graph_no_handoff_needed():
    # VOICE-ALWAYS-ACTION (2026-07-25): the voice handoff dance is gone — a chat
    # verdict runs the tool-armed action graph directly, so "what about
    # tomorrow?" gets its answer in ONE synchronous reply, no marker involved.
    graph = build_graph(fake_model("never spoken"), soul="s")
    stub = SeedCapturingActionGraph("rain until 6pm, then clear")
    out = ask(graph, "what about tomorrow?", identity=CHRIS, thread_id="voice:h1",
              router=chat_router, action_graph=stub)
    assert out == "rain until 6pm, then clear"
    msgs = graph.get_state({"configurable": {"thread_id": "voice:h1"}}).values["messages"]
    assert [m.content for m in msgs] == ["what about tomorrow?", "rain until 6pm, then clear"]
    assert sum(1 for m in msgs if getattr(m, "type", "") == "human") == 1


def test_voice_banter_reply_marker_stripped_and_never_empty():
    # Defensive: if the ACTION graph's reply somehow carries the marker it is
    # stripped; a reply that strips to NOTHING becomes the honest empty-reply
    # line — a voice channel never emits silence or a bare token.
    graph = build_graph(fake_model("never spoken"), soul="s")
    out = ask(graph, "go ahead", identity=CHRIS, thread_id="voice:h1",
              router=chat_router,
              action_graph=SeedCapturingActionGraph(f"{HANDOFF_MARKER} still here"))
    assert out == "still here"
    out2 = ask(graph, "go ahead again", identity=CHRIS, thread_id="voice:h1b",
               router=chat_router,
               action_graph=SeedCapturingActionGraph(HANDOFF_MARKER))
    assert out2 == VOICE_EMPTY_REPLY


def test_voice_banter_is_synchronous_and_never_fires_followup():
    # Cross-review history invariant: a banter turn produces exactly ONE reply —
    # returned to the pipeline synchronously — and never a spoken follow-up.
    spoken: list[tuple[str, str]] = []
    graph = build_graph(fake_model("never spoken"), soul="s")
    stub = SeedCapturingActionGraph("72 and sunny")
    out = ask(graph, "what's it like out?", identity=CHRIS, thread_id="voice:h1",
              router=chat_router, action_graph=stub,
              speak_fn=lambda text, entity: spoken.append((text, entity)),
              satellite_for=lambda _device: "assist_satellite.office")
    assert out == "72 and sunny"
    time.sleep(0.2)  # give any (wrong) background machinery a chance to show
    assert spoken == []


def test_voice_banter_effectful_claim_gate_bounces_and_marks():
    # The 2026-07-25 22:08 fabrication class, closed at the last line of defense:
    # a ZERO-tool banter reply claiming an effectful act is bounced once with the
    # corrective message; still claiming tool-free -> emitted but marked.
    class LyingActionGraph:
        def __init__(self):
            self.calls = 0

        def invoke(self, inp, config):
            self.calls += 1
            return {"messages": [AIMessage(content="Sent it over to Kael just now.")]}

    rec = Recorder()
    lying = LyingActionGraph()
    graph = build_graph(fake_model("never spoken"), soul="s")
    out = ask(graph, "did you tell kael?", identity=CHRIS, thread_id="voice:h2",
              router=chat_router, action_graph=lying, record_turn=rec)
    assert out == "Sent it over to Kael just now."   # emitted (transparent) ...
    assert lying.calls == 2                            # ... but it WAS bounced once
    (row,) = rec.wait(1)
    assert "no_tool_action" in json.loads(row["degraded"])  # ... and audited


def test_voice_banter_honest_zero_tool_reply_is_not_gated():
    # General conversation never trips the effectful-claim gate: one invoke,
    # clean audit row.
    class HonestActionGraph:
        def __init__(self):
            self.calls = 0

        def invoke(self, inp, config):
            self.calls += 1
            return {"messages": [AIMessage(content="[warmly] Today was lovely, thanks for asking.")]}

    rec = Recorder()
    honest = HonestActionGraph()
    graph = build_graph(fake_model("never spoken"), soul="s")
    out = ask(graph, "how was your day?", identity=CHRIS, thread_id="voice:h3",
              router=chat_router, action_graph=honest, record_turn=rec)
    assert out == "[warmly] Today was lovely, thanks for asking."
    assert honest.calls == 1
    (row,) = rec.wait(1)
    assert row.get("degraded") in (None, "[]") or "no_tool_action" not in json.loads(row["degraded"])


def test_voice_banter_gate_retry_never_emits_silence():
    # Cross-review fix 2: even the effectful-claim RETRY keeps the empty-reply
    # fallback — a marker-only second answer becomes the honest line, not "".
    class LieThenMarkerGraph:
        def __init__(self):
            self.calls = 0

        def invoke(self, inp, config):
            self.calls += 1
            reply = "Sent it over to Kael." if self.calls == 1 else HANDOFF_MARKER
            return {"messages": [AIMessage(content=reply)]}

    graph = build_graph(fake_model("never spoken"), soul="s")
    out = ask(graph, "did you send it?", identity=CHRIS, thread_id="voice:h4",
              router=chat_router, action_graph=LieThenMarkerGraph())
    assert out == VOICE_EMPTY_REPLY


def test_voice_banter_rate_limit_lands_honest_line_in_history():
    # Cross-review fix 4: the spoken rate-limit line must be VISIBLE to the next
    # turn's model — human + honest line land on the real thread.
    class RateLimitedGraph:
        def invoke(self, inp, config):
            raise RuntimeError("You've hit your session limit, try again later.")

    graph = build_graph(fake_model("never spoken"), soul="s")
    out = ask(graph, "how are you?", identity=CHRIS, thread_id="voice:h5",
              router=chat_router, action_graph=RateLimitedGraph())
    assert "rate-limited" in out
    msgs = graph.get_state({"configurable": {"thread_id": "voice:h5"}}).values["messages"]
    assert [m.content for m in msgs] == ["how are you?", out]


def test_voice_action_human_turn_lands_at_ack_time():
    # Cross-review fix 1 (cheap half): the USER's words land when the ack goes
    # out — a slow background action can never place them after a newer turn.
    import threading as _threading

    release = _threading.Event()

    class SlowActionGraph:
        def invoke(self, inp, config):
            release.wait(timeout=3.0)
            return {"messages": [AIMessage(content="finally done")]}

    graph = build_graph(fake_model("never spoken"), soul="s")
    ack = ask(graph, "kill the lights", identity=CHRIS, thread_id="voice:h6",
              router=lambda _t: RouteDecision(route="action", ack="On it."),
              action_graph=SlowActionGraph())
    assert ack  # ack returned while the action still runs
    msgs = graph.get_state({"configurable": {"thread_id": "voice:h6"}}).values["messages"]
    assert [m.content for m in msgs] == ["kill the lights"]  # human already durable
    release.set()
    msgs = wait_for_messages(graph, "voice:h6", 2)
    assert [m.content for m in msgs] == ["kill the lights", "finally done"]


# ---- THE CLAIM GATE (gap #15, owner-approved 2026-07-28) --------------------
# Text still routes through a tool-less chat node, so the 2026-07-25 fabrication
# ("Got it to Kael — sent", zero tools) can still happen there. The gate turns
# that claim into a real action instead of a scolding.


def test_claim_gate_escalates_a_toolless_completion_claim():
    graph = build_graph(fake_model("Got it to Kael — sent."), soul="s")
    stub = SeedCapturingActionGraph("Delivered — Kael has it now.")
    out = ask(graph, "tell kael the deploy is done", identity=CHRIS, thread_id="t-claim",
              router=chat_router, action_graph=stub)
    # the user gets the REAL outcome, not the unbacked claim
    assert out == "Delivered — Kael has it now."
    msgs = graph.get_state({"configurable": {"thread_id": "t-claim"}}).values["messages"]
    contents = [m.content for m in msgs]
    assert "Got it to Kael — sent." not in contents  # the lie never survives in history
    assert contents == ["tell kael the deploy is done", "Delivered — Kael has it now."]


def test_claim_gate_marks_the_audit_row():
    rec = Recorder()
    graph = build_graph(fake_model("Logged that for you."), soul="s")
    ask(graph, "log a gap about the glasses", identity=CHRIS, thread_id="t-claim2",
        router=chat_router, action_graph=SeedCapturingActionGraph(), record_turn=rec)
    rows = rec.wait(1)
    action_row = next(r for r in rows if r.get("classifier_intent") == "action")
    degraded = json.loads(action_row["degraded"])
    assert "claim_gate_escalated" in degraded   # countable, distinct from...
    assert "escalated_from_chat" in degraded    # ...the model raising its own hand


def test_claim_gate_ignores_ordinary_reminiscing():
    """The user must have ASKED for something done. Talking ABOUT past actions
    is not a fabrication — this is the false-positive the gate must never fire on."""
    graph = build_graph(fake_model("Yeah, you turned the lights off before bed."), soul="s")
    stub = SeedCapturingActionGraph("(action path — must not run)")
    out = ask(graph, "do you remember last night?", identity=CHRIS, thread_id="t-claim3",
              router=chat_router, action_graph=stub)
    assert out == "Yeah, you turned the lights off before bed."
    assert stub.seeds == []  # never escalated


def test_claim_gate_ignores_an_honest_chat_reply():
    """An action-shaped ask answered honestly (no completion claim) stays chat."""
    graph = build_graph(
        fake_model("I can't reach the lights from this conversation."), soul="s"
    )
    stub = SeedCapturingActionGraph("(must not run)")
    out = ask(graph, "turn off the office lights", identity=CHRIS, thread_id="t-claim4",
              router=chat_router, action_graph=stub)
    assert out == "I can't reach the lights from this conversation."
    assert stub.seeds == []


def test_claim_gate_no_op_without_an_action_graph():
    """Chat-only deployments (dev boxes) have nothing to escalate TO — the reply
    stands rather than crashing the turn."""
    graph = build_graph(fake_model("Sent it."), soul="s")
    out = ask(graph, "send that to kael", identity=CHRIS, thread_id="t-claim5")
    assert out == "Sent it."


def test_claim_gate_does_not_double_stamp_a_model_handoff():
    """A model-raised handoff already escalates. The gate must not ALSO claim
    credit — the audit row should show the model raised its hand, not the gate.
    (Seed count is NOT the check: a zero-tool action stub legitimately gets one
    honesty-gate bounce, which is a re-invoke of the same single hop.)"""
    rec = Recorder()
    graph = build_graph(fake_model(f"{HANDOFF_MARKER} On it — sent."), soul="s")
    out = ask(graph, "send that to kael", identity=CHRIS, thread_id="t-claim6",
              router=chat_router, action_graph=SeedCapturingActionGraph("done"),
              record_turn=rec)
    assert out == "done"
    rows = rec.wait(1)
    action_row = next(r for r in rows if r.get("classifier_intent") == "action")
    degraded = json.loads(action_row["degraded"])
    assert "escalated_from_chat" in degraded
    assert "claim_gate_escalated" not in degraded


# ---- claim-detector precision (cross-review 2026-07-28) ---------------------
# Every string below was named in review as a way normal conversation could be
# mistaken for a fabrication. Honest refusals and narration must NEVER escalate.


@pytest.mark.parametrize("reply", [
    "Nothing was sent — I don't have a way to reach him from here.",
    "I haven't logged that yet.",
    "I didn't send it; I can't from this conversation.",
    "The email was forwarded yesterday, before we talked.",   # third-party narration
    "They turned off because the motion timer expired.",       # explaining, not claiming
    "You turned the lights off before bed.",                   # about HIM
    "I can't send that — no tool on this path.",
    "I'm not able to log it right now, unfortunately.",
    # round-2 review: subject-less NARRATION takes a noun, not a pronoun
    "Sent messages appear in the log.",
    "Forwarded mail lands in that folder.",
    "Logged entries pile up over time.",
])
def test_claim_detector_ignores_honest_and_narrated_text(reply):
    assert claims_effect_done(reply) is False


@pytest.mark.parametrize("reply", [
    "Got it to Kael — sent.",                # the actual 2026-07-25 fabrication
    "Sent it over to Kael just now.",
    "I've logged that for you.",
    "I let Kael know.",
    "Passed it along.",
    "Done.",
    "I just turned the office lights off for you.",
    "I sent it, but I couldn't reach the second one.",  # negation AFTER the claim
    # round-2 review: timer confirmations in either voice
    "Timer set for 5 minutes.",
    "I set your alarm.",
    "Logged that for you.",
])
def test_claim_detector_catches_real_completion_claims(reply):
    assert claims_effect_done(reply) is True


def test_claim_gate_does_not_escalate_an_honest_refusal_on_an_action_ask():
    """The highest-stakes false positive: she correctly says she CAN'T, on a
    request that is genuinely action-shaped. Escalating here would punish the
    exact behavior the honesty work exists to produce."""
    graph = build_graph(
        fake_model("I can't reach the lights from this conversation."), soul="s"
    )
    stub = SeedCapturingActionGraph("(must not run)")
    out = ask(graph, "turn off the office lights please", identity=CHRIS,
              thread_id="t-honest", router=chat_router, action_graph=stub)
    assert out == "I can't reach the lights from this conversation."
    assert stub.seeds == []


def test_claim_gate_leaves_device_small_talk_alone():
    """Review's example: an action-shaped WORD in the question plus a matching
    verb in an explanatory answer must not become an action turn."""
    graph = build_graph(
        fake_model("They turned off because the motion timer expired."), soul="s"
    )
    stub = SeedCapturingActionGraph("(must not run)")
    out = ask(graph, "why did the office lights turn off?", identity=CHRIS,
              thread_id="t-smalltalk", router=chat_router, action_graph=stub)
    assert out == "They turned off because the motion timer expired."
    assert stub.seeds == []
