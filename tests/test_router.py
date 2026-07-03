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
    fallback_decision,
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


# ---- state questions route ACTION regardless of opinion phrasing --------------
# Regression for the 2026-07-02 live miss: "Do you think Jolteon has enough
# charge to get to Tampa and back?" routed chat (opinion-phrased), so the
# tool-less chat path answered "no charge level on file" while the action path
# had read 96% minutes earlier. Questions needing CURRENT device/sensor state
# must be action, however they're phrased.

def test_opinion_phrased_charge_question_heuristic_fails_toward_action():
    # even the degraded-path heuristic must catch state words, not just commands
    decision = fallback_decision(
        "do you think Jolteon has enough charge to get to Tampa and back?"
    )
    assert decision.route == "action"


def test_state_question_shapes_hit_heuristic():
    assert plausibly_commands_device("I wonder if the car's battery is topped off")
    assert plausibly_commands_device("would the house be too warm? check the temperature")
    assert plausibly_commands_device("is the front door locked?")


def test_genuine_opinion_chat_stays_chat():
    assert fallback_decision("do you think cats love us?").route == "chat"
    assert fallback_decision("I wonder if we'll ever visit Japan").route == "chat"


# ---- web-lookup questions route ACTION on the degraded path --------------------
# The search_web tool lives on the action path; a chat answer to "what's the
# weather this weekend?" is stale training data dressed as fact — same
# hallucinated-answer failure the media/state guards prevent.

def test_web_lookup_shapes_hit_heuristic():
    from aerys_v2.router import plausibly_wants_web_search

    assert plausibly_wants_web_search("search for the latest on the merger")
    assert plausibly_wants_web_search("what's the weather this weekend?")
    assert plausibly_wants_web_search("look up who won last night")
    assert plausibly_wants_web_search("what's the current price of bitcoin")
    # timeless opinion/knowledge stays OUT of the search heuristic
    assert not plausibly_wants_web_search("do you think cats love us?")


def test_web_lookup_fails_toward_action():
    assert fallback_decision("search for tornado warnings near me").route == "action"
    assert fallback_decision("what's the weather this weekend?").route == "action"
    assert fallback_decision("what's the latest news on the election").route == "action"


def test_router_prompt_teaches_web_lookup_is_action():
    # prompt-level regression guard: the search-routing rule must stay in the
    # instructions verbatim enough to keep firing, and the tool name must match
    # tools/web_search.py's @tool (the V1 name-mismatch bug).
    from aerys_v2.router import _ROUTER_INSTRUCTIONS

    text = _ROUTER_INSTRUCTIONS.lower()
    assert "live web lookup" in text
    assert "current events" in text
    assert "search for" in text
    assert "training cutoff" in text


def test_router_prompt_teaches_state_questions_are_action():
    # prompt-level regression guard: the routing rule that fixed the Jolteon
    # miss must stay in the instructions — opinion phrasing never makes a
    # live-state question "chat", and uncertainty fails toward action.
    from aerys_v2.router import _ROUTER_INSTRUCTIONS

    text = _ROUTER_INSTRUCTIONS.lower()
    assert "current state" in text
    assert "charge" in text
    assert "opinion or speculation wording is still" in text
    assert 'unsure whether live state is needed, choose "action"' in text


def test_action_overlay_permits_read_plus_reasoning():
    # the action graph must be allowed to COMBINE a sensor read with general
    # reasoning (EV range math) instead of stopping at the raw number
    from aerys_v2.factory import ACTION_OVERLAY

    lowered = ACTION_OVERLAY.lower()
    assert "read-only questions" in lowered
    assert "combine" in lowered


def test_garbled_state_reply_fails_toward_action():
    # unusable router JSON + state-shaped text -> heuristic sends it to action
    decision = parse_route_reply(
        "she probably has plenty!",
        "do you think jolteon has enough charge to reach tampa?",
    )
    assert decision.route == "action"
    assert decision.ack == FALLBACK_ACK
