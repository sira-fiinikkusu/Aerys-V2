"""Offline tests for the route classifier — fake models, no network.

What these prove: strict JSON parsing, ack passthrough (generated, not templated),
and the locked failure direction — an unusable router answer falls back to the
keyword heuristic, which on device-shaped text fails TOWARD the audited action
path (never toward chat, where a hallucinated "done!" can't be caught).
"""

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from aerys_v2.router import (
    FALLBACK_ACK,
    RouteDecision,
    build_router,
    parse_route_reply,
    plausibly_commands_device,
)


def fake_model(*replies: str) -> GenericFakeChatModel:
    return GenericFakeChatModel(messages=iter([AIMessage(content=r) for r in replies]))


class ExplodingModel(GenericFakeChatModel):
    """Fake whose call always raises — the router-is-down scenario."""

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        raise RuntimeError("router down")


# ---- happy path: strict JSON in, decision out ---------------------------------

def test_chat_route_parsed():
    router = build_router(fake_model('{"route": "chat", "ack": ""}'), soul="s")
    assert router("how was your day?") == RouteDecision(route="chat", ack="")


def test_action_route_passes_generated_ack_through():
    # the ack is the router's own words — it must survive verbatim to the caller
    router = build_router(
        fake_model('{"route": "action", "ack": "[softly] Dousing the office light now"}'),
        soul="s",
    )
    decision = router("kill the office light")
    assert decision.route == "action"
    assert decision.ack == "[softly] Dousing the office light now"


def test_code_fenced_json_still_parses():
    # temp-0 models still occasionally wrap JSON; the slice-to-braces tolerates it
    raw = '```json\n{"route": "action", "ack": "Flipping it now"}\n```'
    assert parse_route_reply(raw, "toggle the fan").route == "action"


# ---- failure direction: unusable answer -> heuristic, biased toward action ----

def test_garbage_reply_with_devicey_text_fails_toward_action():
    decision = parse_route_reply("I think you want the lights on!", "turn on the lamp")
    assert decision.route == "action"
    assert decision.ack == FALLBACK_ACK  # the ONLY templated ack: degraded path only


def test_garbage_reply_with_chatty_text_falls_to_chat():
    assert parse_route_reply("not json at all", "tell me a story").route == "chat"


def test_invalid_route_value_rejected_strictly():
    # "maybe" is not on the contract — strict validation, then heuristic
    decision = parse_route_reply('{"route": "maybe", "ack": "hm"}', "toggle the desk lamp")
    assert decision.route == "action"


def test_action_route_with_empty_ack_gets_fallback_ack():
    # route survives (the model classified fine); only the ack degraded
    decision = parse_route_reply('{"route": "action", "ack": ""}', "lights off")
    assert decision == RouteDecision(route="action", ack=FALLBACK_ACK)


def test_router_model_exception_falls_to_heuristic():
    router = build_router(ExplodingModel(messages=iter([])), soul="s")
    assert router("turn off the bedroom light").route == "action"  # device-shaped
    assert router("what's for dinner?").route == "chat"            # not device-shaped


def test_device_heuristic_shapes():
    assert plausibly_commands_device("please TURN ON the porch light")
    assert plausibly_commands_device("toggle the fan")
    assert not plausibly_commands_device("how are you feeling today?")
