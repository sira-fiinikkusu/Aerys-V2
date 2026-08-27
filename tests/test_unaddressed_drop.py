"""Offline tests for FALSE-WAKE GRACE (owner ask 2026-08-27, Alexa-style).

A wake word that misfires mid-human-conversation produces a captured fragment
that was never directed at Aerys. The router now judges "unaddressed" alongside
route/tier; on voice and lens turns with the feature armed, ask() drops the
turn — nothing spoken, nothing run, the fragment never enters thread history —
but ALWAYS writes a receipt row (dropped_unaddressed) so the judgment is
auditable and tunable.

Pinned here: the parse contract (flag read, absent = False), the STRUCTURAL
guard (a command-shaped message is never dropped, whatever the model said),
both drop paths (voice, lens-text) with their receipts, the kill-switch, the
typed-surface immunity (unaddressed verdict is inert off voice/lens), and that
a dropped fragment stays out of durable history.

v2 (same day): the first prompt's "fragment with no ask shape" clause dropped
three of the owner's real mid-conversation replies within three hours of
arming — the router sees one message at a time and cannot distinguish a
continuation from an overheard fragment. The rule is now explicit-evidence
only (direct second-person address to someone else, or a wave-off); the
mechanics tested here are unchanged, and the feature ships DISARMED
(VOICE_DROP_UNADDRESSED=false) until the owner blesses re-arm.
"""

import threading
import time

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from aerys_v2.factory import build_action_graph, build_graph
from aerys_v2.router import RouteDecision, parse_route_reply
from aerys_v2.service import DROPPED_UNADDRESSED_MARKER, ask

CHRIS = {"user_id": "person-1", "display_name": "Chris"}
# v2 rule (live failure 2026-08-27): only DIRECT second-person address to
# someone else, or an explicit wave-off, may drop. A story that merely MENTIONS
# another person is ADDRESSED — this constant is the droppable kind.
SIDE_CHATTER = "Megan, can you grab my towel real quick?"


def fake_model(*replies) -> GenericFakeChatModel:
    msgs = [r if isinstance(r, AIMessage) else AIMessage(content=r) for r in replies]
    return GenericFakeChatModel(messages=iter(msgs))


def unaddressed_router(_text: str) -> RouteDecision:
    return RouteDecision(route="chat", ack="", unaddressed=True)


def chat_router(_text: str) -> RouteDecision:
    return RouteDecision(route="chat", ack="")


class Recorder:
    def __init__(self):
        self.rows: list[dict] = []
        self._lock = threading.Lock()

    def __call__(self, row: dict) -> None:
        with self._lock:
            self.rows.append(row)

    def wait_for_rows(self, n: int, timeout: float = 2.0) -> list[dict]:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._lock:
                if len(self.rows) >= n:
                    return list(self.rows)
            time.sleep(0.01)
        with self._lock:
            return list(self.rows)


# ---- parse contract ----


def test_parser_reads_the_unaddressed_flag():
    d = parse_route_reply(
        '{"route": "chat", "ack": "", "tier": "fast", "unaddressed": true}',
        SIDE_CHATTER,
    )
    assert d.unaddressed is True


def test_flag_absent_or_false_means_addressed():
    assert (
        parse_route_reply('{"route": "chat", "ack": ""}', "hey aerys").unaddressed
        is False
    )
    assert (
        parse_route_reply(
            '{"route": "chat", "ack": "", "unaddressed": false}', "hey"
        ).unaddressed
        is False
    )


def test_structural_guard_command_shape_is_never_dropped():
    # The model said unaddressed, but the text carries a command shape — the
    # plausibly_asks_for_action union overrides. Eating a real "turn off the
    # lights" is the one failure this feature must not have.
    d = parse_route_reply(
        '{"route": "action", "ack": "on it", "unaddressed": true}',
        "turn off the office lights",
    )
    assert d.unaddressed is False


# ---- voice path ----


def voice_ask(router, *, drop: bool, recorder=None, model_reply="hi there"):
    graph = build_graph(fake_model(model_reply), "SOUL")
    action_graph = build_action_graph(fake_model(model_reply), "SOUL", tools=[])
    return ask(
        graph,
        SIDE_CHATTER,
        identity={**CHRIS, "voice": True},
        thread_id="person:person-1",
        router=router,
        action_graph=action_graph,
        record_turn=recorder,
        drop_unaddressed=drop,
    ), graph


def test_voice_unaddressed_turn_is_dropped_with_a_receipt():
    rec = Recorder()
    reply, graph = voice_ask(unaddressed_router, drop=True, recorder=rec)
    assert reply == ""
    rows = rec.wait_for_rows(1)
    assert len(rows) == 1
    assert rows[0]["classifier_intent"] == "unaddressed"
    assert DROPPED_UNADDRESSED_MARKER in (rows[0]["degraded"] or [])
    assert rows[0]["emitted_reply"] == ""


def test_dropped_fragment_never_enters_thread_history():
    _, graph = voice_ask(unaddressed_router, drop=True)
    state = graph.get_state({"configurable": {"thread_id": "person:person-1"}})
    assert not state.values.get("messages")


def test_kill_switch_disarms_the_drop():
    reply, _ = voice_ask(unaddressed_router, drop=False)
    assert reply != ""  # turn ran normally despite the verdict


def test_addressed_voice_turns_flow_unchanged():
    reply, _ = voice_ask(chat_router, drop=True)
    assert reply != ""


# ---- lens text path ----


def lens_ask(identity, *, drop: bool, recorder=None):
    graph = build_graph(fake_model("hi there"), "SOUL")
    action_graph = build_action_graph(fake_model("unused"), "SOUL", tools=[])
    return ask(
        graph,
        SIDE_CHATTER,
        identity=identity,
        thread_id="glasses:s1",
        router=unaddressed_router,
        action_graph=action_graph,
        record_turn=recorder,
        drop_unaddressed=drop,
    )


def test_lens_text_turn_is_dropped_with_a_receipt():
    rec = Recorder()
    reply = lens_ask({**CHRIS, "surface": "lens"}, drop=True, recorder=rec)
    assert reply == ""
    rows = rec.wait_for_rows(1)
    assert rows and rows[0]["classifier_intent"] == "unaddressed"
    assert DROPPED_UNADDRESSED_MARKER in (rows[0]["degraded"] or [])


def test_typed_surfaces_are_immune_even_when_the_router_says_unaddressed():
    # No voice flag, no lens surface: a typed message is intentional by
    # definition — the verdict is inert metadata and the turn runs normally.
    reply = lens_ask(dict(CHRIS), drop=True)
    assert reply != ""
