"""D99 (Chris 11/25 12:46, "your pick works here"): in a group room, a household member's "looking back"
question goes to her tool side (search_memory / search_history, D95-A); guests get no search tools.

Rules tested here:
1. Only a household identity typing in a Discord guild or Telegram group is ever moved. DMs, voice, the lens,
   guests and cold handles never are, even if the check would say yes.
2. One Jev look decides: at REROUTE_AT or above the chat-routed turn runs on the action graph as a specialist
   turn; below it, or on any failure (timeout, error, bad score, no key), the turn stays on the chat side.
3. Turns the router already sent to the action side are never touched, and the check isn't asked.
"""

from __future__ import annotations

import pytest

from aerys_v2 import looking_back as lb
from aerys_v2.factory import build_graph
from aerys_v2.service import room_recall_eligible
from test_d79_text_one_loop import (ALLOW, GUEST, OWNER, WIFE, Action, FakeJev, dm, fake_model, guild,
                                    route, run, tool_then)

BACK = "remember when we talked about the router?"


def group(user_id=OWNER):
    return {"user_id": user_id, "display_name": "Chris", "privacy_context": "public",
            "platform": "telegram", "channel_kind": "group", "channel_id": "-100", "channel_name": "house"}


class Check:
    """A stand-in for LookingBackCheck: answers `p` and remembers what it was asked."""

    def __init__(self, p):
        self.p, self.asked = p, []

    def __call__(self, message):
        self.asked.append(message)
        return self.p


FOUND = "found it: you said the router lives in the closet"


def armed(p, *script):
    """An action graph whose look-back turn really searches (an action turn that touches no tool is bounced
    once by the action-honesty gate, so a real look back calls search_history)."""
    action = Action(*(script or (tool_then("search_history", FOUND, "11/20 #general: router lives in the closet",
                                           args={"query": "router"}),)))
    action.looking_back = Check(p)
    return action


# ---- who is eligible ------------------------------------------------------------------------

@pytest.mark.parametrize("identity,allow,owner,ok", [
    (guild(), ALLOW, OWNER, True),
    (guild(WIFE), ALLOW, OWNER, True),                      # household, by the allowlist
    (group(), ALLOW, OWNER, True),                          # a Telegram group
    (guild(GUEST), ALLOW, OWNER, False),                    # a guest in the room
    (dm(), ALLOW, OWNER, False),                            # a DM: D79's one loop serves those
    ({**guild(), "voice": True}, ALLOW, OWNER, False),
    ({**guild(), "surface": "lens"}, ALLOW, OWNER, False),
    ({**guild(), "platform": "kael"}, ALLOW, OWNER, False),
    ({"user_id": OWNER, "privacy_context": "public"}, ALLOW, OWNER, False),   # no surface
    (guild("discord:123"), None, "", False),                # a cold handle
    (guild(), None, OWNER, True),                           # dev box: the owner only
    (guild(WIFE), None, OWNER, False),
])
def test_only_household_in_a_group_room_is_eligible(identity, allow, owner, ok):
    assert room_recall_eligible(identity, allow, owner) is ok


# ---- the reroute ----------------------------------------------------------------------------

def test_a_household_look_back_in_the_guild_runs_on_the_tool_side():
    action = armed(.95)
    out = run(BACK, identity=guild(), action=action)
    assert out == FOUND
    assert action.looking_back.asked == [BACK]
    seed, conf = action.calls[0]
    assert conf["specialist"] is True and seed[-1].content == BACK


def test_a_telegram_group_look_back_moves_too():
    action = armed(.9)
    assert run(BACK, identity=group(WIFE), action=action).startswith("found it")


def test_below_the_line_the_turn_stays_on_the_chat_side():
    graph = build_graph(fake_model("from chat"), soul="s")
    action = armed(lb.REROUTE_AT - .01)
    assert run("what's the weather tomorrow?", identity=guild(), graph=graph, action=action) == "from chat"
    assert action.calls == [] and action.looking_back.asked == ["what's the weather tomorrow?"]


def test_no_answer_keeps_the_old_path():
    graph = build_graph(fake_model("from chat"), soul="s")
    action = armed(None)
    assert run(BACK, identity=guild(), graph=graph, action=action) == "from chat"
    assert action.calls == []


def test_no_check_armed_keeps_the_old_path():
    graph = build_graph(fake_model("from chat", "from chat"), soul="s")
    action = Action("should not run")                         # no .looking_back at all
    assert run(BACK, identity=guild(), graph=graph, action=action) == "from chat"
    action.looking_back = None
    assert run(BACK, identity=guild(), graph=graph, action=action) == "from chat"
    assert action.calls == []


def test_a_guest_in_the_room_never_reaches_search_even_if_their_graph_would_say_yes():
    graph = build_graph(fake_model("from chat"), soul="s")
    main, guest = armed(.99), armed(.99, "guest graph!")
    assert run(BACK, identity=guild(GUEST), graph=graph, action=main, guest=guest) == "from chat"
    assert main.calls == [] and guest.calls == []
    assert main.looking_back.asked == [] and guest.looking_back.asked == []


def test_a_dm_is_never_rerouted_by_this_check():
    graph = build_graph(fake_model("from chat"), soul="s")
    action = armed(.99)
    assert run(BACK, identity=dm(), graph=graph, action=action, flag=False) == "from chat"
    assert action.calls == [] and action.looking_back.asked == []


def test_an_action_route_in_the_room_is_untouched_and_the_check_is_not_asked():
    action = armed(.99, tool_then("home_control", "lights off", args={"operation": "turn_off"}))
    assert run("turn off the office", identity=guild(), action=action, router=route("action")) == "lights off"
    assert action.looking_back.asked == []


# ---- the check itself -----------------------------------------------------------------------

@pytest.mark.parametrize("p,moved", [(None, False), (0.0, False), (0.69, False), (0.7, True), (0.97, True)])
def test_the_line_is_at_point_seven(p, moved):
    assert lb.reroute(p) is moved


def test_the_check_sends_the_message_alone_with_one_question():
    jev = FakeJev([{lb.QUESTION_ID: .97}])
    assert lb.LookingBackCheck(jev)("  remember   when\nwe talked? ") == .97
    (req,) = jev.requests
    assert req["state"] == {"message": "remember when we talked?"}
    assert list(req["questions"]) == [lb.QUESTION_ID] and req["model"] == lb.MODEL


def test_a_long_message_is_cut_and_marked():
    state = lb.build_state("x" * (lb.MESSAGE_MAX + 50))
    assert len(state["message"]) == lb.MESSAGE_MAX + 1 and state["message"].endswith("…")


@pytest.mark.parametrize("answer", [RuntimeError("jev down"), {lb.QUESTION_ID: True}, {lb.QUESTION_ID: 1.5},
                                    {lb.QUESTION_ID: float("nan")}])
def test_errors_and_bad_scores_are_no_answer(answer):
    assert lb.LookingBackCheck(FakeJev([answer]))(BACK) is None


def test_a_slow_answer_is_never_used():
    check = lb.LookingBackCheck(FakeJev([{lb.QUESTION_ID: .99}], delay=.3), timeout_s=.05)
    assert check(BACK) is None


def test_an_empty_message_asks_nothing():
    jev = FakeJev([{lb.QUESTION_ID: .99}])
    assert lb.LookingBackCheck(jev)("   ") is None and jev.requests == []


def test_no_key_builds_no_check():
    class S:
        typesafe_api_key = None

    assert lb.looking_back_for(S()) is None


# ---- what a room search brings back (advisor 11/25) ------------------------------------------

def _rt(identity, sink):
    from types import SimpleNamespace
    return SimpleNamespace(config={"configurable": {"identity": identity, "room_sink": sink,
                                                    "thread_id": f"person:{identity.get('user_id')}"}})


def test_other_peoples_lines_from_a_room_search_count_as_outside_text():
    """D98 keeps her own past chats tap-free; a guest's line pulled into a tool-armed turn gets the same
    mark the room's recent activity gets, so it can inform the turn but never authorize a reach-out."""
    from types import SimpleNamespace

    from aerys_v2.outside_guard import sources_for
    from test_recall_search import ALEX, ROOM, guild as room_identity, history_tool, strict_conn, turn

    sink = {}
    rows = [turn("room talk about the attic fan", channel="guild", channel_id=ROOM, person=ALEX, name="Alex")]
    out = history_tool(strict_conn(rows)).func(runtime=_rt(room_identity(), sink), query="attic")
    assert "attic fan" in out and sink == {"outside": True}
    request = SimpleNamespace(runtime=SimpleNamespace(config={"configurable": {"room_sink": sink}}),
                              state={"messages": []})
    assert "room" in sources_for(request)


def test_the_callers_own_lines_leave_the_turn_unmarked():
    from test_recall_search import ROOM, dm as dm_identity, guild as room_identity, history_tool, strict_conn, turn

    sink = {}
    own = [turn("my attic fan is loud", channel="guild", channel_id=ROOM)]
    history_tool(strict_conn(own)).func(runtime=_rt(room_identity(), sink), query="attic")
    assert sink == {}
    history_tool(strict_conn([turn()])).func(runtime=_rt(dm_identity(), sink), query="attic")
    assert sink == {}                                  # D98: his own past chats need no tap
