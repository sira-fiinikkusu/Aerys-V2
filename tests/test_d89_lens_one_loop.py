"""D89 (the owner, 11/22 13:30, "Yeah"): his own glasses turns take the one loop and the honesty judge,
like his household DMs (D79). 11/22 18:26 on the glasses: two real web searches, then the no-tools chat
side said "I can't run web searches in this mode... those results were not from a search". The twins
disowning a real act. Owner only: a household member's or a guest's lens turn, and voice, stay as they were."""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from aerys_v2.service import text_one_loop_eligible
from aerys_v2.factory import build_graph
import aerys_v2.factory as factory
from test_d79_text_one_loop import ALLOW, GUEST, OWNER, WIFE, Action, fake_model, need, run


@pytest.fixture(autouse=True)
def owner(monkeypatch):
    """Production sets factory._OWNER_ID from settings.owner_person_id at startup (cli.py)."""
    monkeypatch.setattr(factory, "_OWNER_ID", OWNER)


def lens(user_id=OWNER, **extra):
    """What the g2-bridge's /ask builds: the owner's person id, surface 'lens', no platform."""
    return {"user_id": user_id, "display_name": "Chris (Glasses)", "privacy_context": "private",
            "surface": "lens", **extra}


@pytest.mark.parametrize("identity,allow,owner,ok", [
    (lens(), ALLOW, OWNER, True),                          # his glasses
    (lens(), None, OWNER, True),                           # dev box: still the owner
    (lens(WIFE), ALLOW, OWNER, False),                     # household, but not his lens: owner only
    (lens(GUEST), ALLOW, OWNER, False),
    (lens(voice=True), ALLOW, OWNER, False),               # voice has its own one loop
    (lens(), ALLOW, "", False),                            # no owner configured: nobody
    (lens(""), ALLOW, OWNER, False),
])
def test_only_the_owners_own_lens_turns_are_eligible(identity, allow, owner, ok):
    assert text_one_loop_eligible(identity, allow, owner) is ok


def test_a_chat_shaped_lens_turn_runs_on_the_one_loop():
    action = Action("It's the museum's neighbor; I searched it.")
    out = run("whose house is next to the museum", identity=lens(), action=action)
    assert out == "It's the museum's neighbor; I searched it." and len(action.calls) == 1
    assert action.calls[0][1].get("text_one_loop") is True


def test_with_the_switch_off_a_lens_turn_is_unchanged():
    graph = build_graph(fake_model("a chat reply"), soul="s")
    action = Action("main graph!")
    out = run("whose house is that", identity=lens(), graph=graph, action=action, flag=False)
    assert out == "a chat reply" and action.calls == []


def test_a_household_members_lens_turn_stays_off_the_one_loop():
    graph = build_graph(fake_model("hi there"), soul="s")
    action = Action("main graph!")
    out = run("hi", identity=lens(WIFE), graph=graph, action=action)
    assert out == "hi there" and action.calls == []


def test_the_lens_one_loop_prompt_carries_the_correction_and_the_lens_style_not_the_dm_banter():
    need()
    from aerys_v2.action_loop import AgentToolModels, build_agent_action_graph
    from aerys_v2.factory import LENS_SURFACE_OVERLAY, TEXT_BANTER_OVERLAY
    from test_action_loop_agent import TOOLS, Rec

    for specialist in (False, True):                       # the one loop, and a judged action pass
        log = []
        conv = Rec(label="conv", script=[AIMessage(content="ok")], log=log)
        std = Rec(label="std", script=[AIMessage(content="ok")], log=log)
        g = build_agent_action_graph(AgentToolModels(conv, (std, False)), "SOUL", TOOLS,
                                     context_fn=lambda *a: "PROFILE", conversational_context_fn=lambda *a: "FULL")
        g.invoke({"messages": [HumanMessage(content="what is it")]},
                 {"configurable": {"thread_id": "person:o", "identity": lens(), "specialist": specialist,
                                   "text_one_loop": not specialist, "one_loop_correction": "DONT DISOWN THE SEARCH"},
                  "recursion_limit": 10})
        system = log[0]["messages"][0][1]
        assert "DONT DISOWN THE SEARCH" in system and LENS_SURFACE_OVERLAY in system
        assert TEXT_BANTER_OVERLAY not in system
        assert [t for t, _c in log[0]["messages"]].count("human") == 1     # the correction is not a turn
