"""D56 (11/15): her morning brief, evening heads-up and welcome home are her own turns. 10:10 that
morning the brief said a speaker was "still playing" from a 12:25 AM note; it had been paused at
5:52. The owner: "she needs to be able to DECIDE to look at HA or what have you at the time she plans
to do the daily call ... we need to stop treating her like a child." So she gets her tools and the
notes stamped with when they happened, and what to check is hers. Placeholder ids and text only."""

from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langchain_core.tools import tool

from aerys_v2.brief import BRIEF_PROMPT, TURN_TEACHING, Facts, composer_for, facts_text, plain_brief, stamped
from aerys_v2.factory import build_action_graph
from aerys_v2.wakeups import TurnResult, WakeGate, her_turn_for, wake_middleware, wake_tools
from aerys_v2.welcome import WELCOME_PROMPT, WelcomeFacts, welcome_facts_text
from test_morning_brief import House
from test_wakeups import OWNER, call, gate_box, settings

ET = timezone(timedelta(hours=-5))
NOTE = SimpleNamespace(headline="I couldn't stop the music from here, so it's still playing.",
                       created_at=datetime(2026, 11, 15, 0, 25, tzinfo=ET))


def her_graph(script, ran):
    """Her wake graph, wired the way factory.action_stack_for wires it (the fence included)."""

    @tool("home_control")
    def home_control(operation: str, entity_id: str) -> str:
        """House."""
        ran.append((operation, entity_id))
        if operation == "get_state":
            return f"{entity_id} is paused"
        return f"Done: {entity_id} turned {operation.removeprefix('turn_')}."

    @tool("remember")
    def remember(fact: str) -> str:
        """Keep."""
        ran.append(("remember", fact))
        return "Kept: it"

    graph = build_action_graph(GenericFakeChatModel(messages=iter(script)), "test",
                               wake_tools([home_control, remember]), outside_gate=WakeGate(gate_box().gate),
                               guard_owner=OWNER, extra_middleware=wake_middleware([]))
    return SimpleNamespace(wake_graph=graph)


def test_she_checks_what_she_wants_before_she_speaks_through_the_real_graph():
    ran, seen = [], []
    script = [call("home_control", {"operation": "get_state", "entity_id": "media_player.example_speaker"}, "g"),
              call("remember", {"fact": "the speaker was paused"}, "m"),
              AIMessage(content="Morning. The music ran past its two hours last night; it's off now.")]
    turn = her_turn_for(settings(), her_graph(script, ran))
    compose = composer_for(settings(), turn=lambda thread, seed: seen.append((thread, seed)) or turn(thread, seed))
    assert compose(Facts(weather="sunny", held=[NOTE])) == "Morning. The music ran past its two hours last night; it's off now."
    assert ran == [("get_state", "media_player.example_speaker")]       # her check ran; her memory stayed out
    thread, seed = seen[0]
    assert thread.startswith("brief:")
    assert "(12:25 AM) I couldn't stop the music from here" in seed     # the note says when it was true
    assert TURN_TEACHING.strip() in seed and "how things stood THEN" in seed
    assert compose.house == []


def test_a_house_change_she_makes_while_composing_reaches_his_text_copy():
    ran = []
    script = [call("home_control", {"operation": "turn_off", "entity_id": "light.example_lamp"}, "w"),
              AIMessage(content="Morning! I turned the lamp off for you.")]
    h = House()
    h.brief.compose = composer_for(settings(), turn=her_turn_for(settings(), her_graph(script, ran)))
    assert h.brief.tick() == "sent"
    assert h.world.spoken == [("Morning! I turned the lamp off for you.", "dev-office")]
    assert ran == [("turn_off", "light.example_lamp")]
    assert any("While I put that together, in the house: light.example_lamp turned off." in body
               for _, body, *_ in h.world.texts)


def test_a_turn_that_fails_or_overstays_gives_way_to_the_plain_version():
    def broken(thread, seed):
        raise RuntimeError("model down")

    h = House()
    h.brief.compose = composer_for(settings(), turn=broken)
    assert h.brief.tick() == "sent" and h.world.spoken[0][0] == plain_brief(h.gather(None, [], []))

    def slow(thread, seed):
        time.sleep(1.0)
        return TurnResult("too late", [], ["light.example_lamp turned off."])

    compose = composer_for(settings(), turn=slow, wall_s=0.05)
    t = time.monotonic()
    assert compose(Facts()) == "" and compose.house == []                 # he isn't kept waiting
    assert time.monotonic() - t < 0.9

    from aerys_v2.wakeups import open_house_record
    records = []

    def wrote_then_stalled(thread, seed):
        records.append(open_house_record(thread))                         # as wake_config opens it
        records[0]["did"].append("light.example_lamp turned off.")
        time.sleep(1.0)
        return TurnResult("too late", [], [])

    compose = composer_for(settings(), turn=wrote_then_stalled, wall_s=0.2)
    assert compose(Facts()) == ""
    assert compose.house == ["light.example_lamp turned off."]            # what it did before the clock ran out
    assert records[0]["closed"]                                            # and nothing more on its own


def test_notes_carry_the_time_they_happened_in_every_brief_like_moment():
    assert stamped(NOTE) == "(12:25 AM) I couldn't stop the music from here, so it's still playing."
    assert stamped(SimpleNamespace(headline="no time", created_at=None)) == "no time"
    assert "- (12:25 AM) I couldn't stop" in facts_text(Facts(held=[NOTE]))
    assert "how things stood THEN" in BRIEF_PROMPT                       # the tool-free fallback is taught too
    welcome = welcome_facts_text(WelcomeFacts(away=timedelta(hours=2), room="office", held=[NOTE],
                                              brief=Facts(quiet=[NOTE])))
    assert welcome.count("(12:25 AM)") == 2


def test_her_own_turn_needs_the_wake_graph_and_takes_his_blocklist_whatever_arms_first():
    from aerys_v2 import wakeups as wk

    assert her_turn_for(settings(), SimpleNamespace(wake_graph=None)) is None
    assert her_turn_for(settings(), None) is None
    try:
        assert her_turn_for(settings(autonomy_blocked="Sprinkler"), SimpleNamespace(wake_graph=object())) is not None
        assert wk.autonomy_blocked("switch.sprinkler_zone_1")
    finally:
        wk.set_autonomy_blocked("")


def test_the_evening_heads_up_and_a_welcome_home_report_house_changes_too():
    from test_evening_heads_up import Evening
    from test_welcome_home import House as Home

    def lamp(thread, seed):
        return TurnResult("Heads-up for tomorrow, and I switched the lamp off.", ["home_control"],
                          ["light.example_lamp turned off."])

    line = "While I put that together, in the house: light.example_lamp turned off."
    e = Evening()
    e.heads_up.compose = composer_for(settings(), turn=lamp, kind="evening")
    assert e.heads_up.tick() == "sent" and any(line in body for _, body, *_ in e.world.texts)
    h = Home()
    h.hello.compose = composer_for(settings(), prompt=WELCOME_PROMPT, text_of=welcome_facts_text, turn=lamp,
                                   kind="welcome")                  # as welcome_home_for wires it
    assert h.hello.tick() == "sent" and any(line in body for _, body, *_ in h.world.texts)
