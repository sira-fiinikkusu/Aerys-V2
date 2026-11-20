"""D79 (Chris 11/20 13:55-16:32): one her, not twins. Household DMs run every typed turn
through the tool-armed action graph, the way voice has since 2026-07-25
(VOICE-ALWAYS-ACTION). The router's verdict only shapes the reply. Behind a flag that
starts off: flag off is today's chat/action split, unchanged.

The trigger: turn 3476 (11/20 10:55). His "you are good to update your skill" was routed
to the tool-less chat side, and she answered "I've started a home job" with nothing
running. Both word-list gates missed it.
"""

from __future__ import annotations

import threading
import time

import pytest

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

import aerys_v2.factory as factory
from aerys_v2.factory import build_graph
from aerys_v2.router import RouteDecision
from aerys_v2.service import (
    ONE_LOOP_MARKER,
    claims_effect_done,
    _claims_effect_without_doing_it,
    _promises_effect_without_doing_it,
    ask,
    text_one_loop_eligible,
    text_one_loop_flag,
)
from aerys_v2.turns import ACTION_NO_TOOL_CORRECTION

OWNER = "11111111-1111-1111-1111-111111111111"
WIFE = "22222222-2222-2222-2222-222222222222"
GUEST = "33333333-3333-3333-3333-333333333333"
ALLOW = frozenset({OWNER, WIFE})


def dm(user_id=OWNER, platform="discord", **extra):
    return {"user_id": user_id, "display_name": "Chris", "privacy_context": "private",
            "platform": platform, "channel_kind": "dm", "channel_id": "9", **extra}


def guild(user_id=OWNER):
    return {"user_id": user_id, "display_name": "Chris", "privacy_context": "public",
            "platform": "discord", "channel_kind": "guild", "channel_id": "7", "channel_name": "#general"}


def fake_model(*replies) -> GenericFakeChatModel:
    msgs = [r if isinstance(r, AIMessage) else AIMessage(content=r) for r in replies]
    return GenericFakeChatModel(messages=iter(msgs))


class NoCalls:
    """A chat model that must never be asked: the one loop never runs the chat graph."""

    def invoke(self, *_a, **_k):
        raise AssertionError("the chat graph ran on a one-loop turn")


def route(name="chat", tier="standard"):
    return lambda _text: RouteDecision(route=name, ack="", tier=tier)


class Action:
    """A stub action graph: records each invoke (seed + configurable), answers from a script.

    Each script item is either a final string or a list of messages to append (tool calls)."""

    def __init__(self, *script):
        self.script = list(script)
        self.calls: list[tuple[list, dict]] = []

    def invoke(self, inp, config):
        self.calls.append((list(inp["messages"]), dict(config["configurable"])))
        item = self.script.pop(0) if self.script else "ok"
        if isinstance(item, Exception):
            raise item
        if isinstance(item, list):
            return {"messages": [*inp["messages"], *item]}
        return {"messages": [*inp["messages"], AIMessage(content=item)]}


def tool_then(name, final, content="ok"):
    return [AIMessage(content="", tool_calls=[{"id": "c1", "name": name, "args": {}}]),
            ToolMessage(content=content, name=name, tool_call_id="c1"),
            AIMessage(content=final)]


class Recorder:
    def __init__(self):
        self.rows: list[dict] = []
        self._lock = threading.Lock()

    def __call__(self, row):
        with self._lock:
            self.rows.append(row)

    def wait(self, n=1, timeout=3.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._lock:
                if len(self.rows) >= n:
                    return list(self.rows)
            time.sleep(0.01)
        raise AssertionError(f"recorded {len(self.rows)} rows, expected {n}")


def run(text, *, identity=None, graph=None, action=None, guest=None, flag=True, router=None,
        record=None, allow=ALLOW, deep_allowed=None, thread="person:o"):
    graph = graph or build_graph(NoCalls(), soul="s")
    return ask(graph, text, identity=identity or dm(), thread_id=thread,
               router=router or route(), action_graph=action, guest_action_graph=guest,
               action_allowlist=allow, record_turn=record, text_one_loop=flag,
               deep_allowed=deep_allowed)


# ---- the flag and who is eligible -----------------------------------------------------------

def test_flag_off_is_the_old_chat_path():
    graph = build_graph(fake_model("hey you"), soul="s")
    action = Action("should not run")
    out = run("hi", graph=graph, action=action, flag=False)
    assert out == "hey you" and action.calls == []


def test_flag_on_a_household_dm_chat_turn_runs_the_action_graph_and_never_the_chat_graph():
    action = Action("hey you")
    out = run("hi", action=action)
    assert out == "hey you"
    seed, conf = action.calls[0]
    assert conf["specialist"] is False and conf["text_one_loop"] is True
    assert seed[-1].content == "hi"


def test_the_flag_can_be_a_callable_read_every_turn():
    reads = iter([False, True])
    graph = build_graph(fake_model("from chat"), soul="s")
    action = Action("from the one loop")
    flag = lambda: next(reads)  # noqa: E731
    assert run("one", graph=graph, action=action, flag=flag) == "from chat"
    assert run("two", graph=graph, action=action, flag=flag) == "from the one loop"


def test_the_flag_file_wins_over_the_setting(tmp_path):
    class S:
        text_one_loop = False
        text_one_loop_file = str(tmp_path / "flag")

    read = text_one_loop_flag(S())
    assert read() is False                                  # no file: the setting
    (tmp_path / "flag").write_text("on\n")
    assert read() is True
    (tmp_path / "flag").write_text("off")
    assert read() is False
    S.text_one_loop = True
    (tmp_path / "flag").write_text("")                      # empty: the setting again
    assert read() is True


@pytest.mark.parametrize("identity,allow,owner,ok", [
    (dm(), ALLOW, OWNER, True),
    (dm(WIFE), ALLOW, OWNER, True),                          # household, by the allowlist
    (dm(platform="telegram"), ALLOW, OWNER, True),
    (dm(GUEST), ALLOW, OWNER, False),                        # a DM from someone outside the house
    (guild(), ALLOW, OWNER, False),                          # a shared room, even his
    (dm(voice=True), ALLOW, OWNER, False),                   # voice has its own one loop
    (dm(surface="lens"), ALLOW, OWNER, False),
    ({"user_id": OWNER, "privacy_context": "private"}, ALLOW, OWNER, False),   # no surface: not a DM
    ({**dm(), "platform": "kael"}, ALLOW, OWNER, False),     # her Kael lane is not a DM
    (dm("discord:123"), None, "", False),                    # dev box, cold caller
    (dm(), None, OWNER, True),                               # dev box: the owner only
    (dm(WIFE), None, OWNER, False),
])
def test_eligibility_is_household_identity_on_a_dm_surface(identity, allow, owner, ok):
    assert text_one_loop_eligible(identity, allow, owner) is ok


def test_a_guest_dm_stays_on_the_guest_path():
    graph = build_graph(fake_model("hello there"), soul="s")
    action, guest = Action("main graph!"), Action("guest graph!")
    out = run("hi", identity=dm(GUEST), graph=graph, action=action, guest=guest)
    assert out == "hello there"
    assert action.calls == [] and guest.calls == []


def test_a_guild_turn_is_unchanged():
    graph = build_graph(fake_model("hi room"), soul="s")
    action = Action("main graph!")
    out = run("hi", identity=guild(), graph=graph, action=action)
    assert out == "hi room" and action.calls == []


def test_an_action_route_still_takes_the_specialist_path():
    action = Action(tool_then("home_control", "lights off"))
    out = run("turn off the office", action=action, router=route("action"))
    assert out == "lights off"
    assert action.calls[0][1]["specialist"] is True
    assert "text_one_loop" not in action.calls[0][1]


def test_the_1055_turn_lands_in_the_action_graph_with_its_tools():
    action = Action(tool_then("list_async_tasks", "Starting it now.", "No jobs running."),
                    tool_then("start_async_task", "Started the home job.", "started job j1"))
    out = run("you are good to update your skill", action=action)
    assert action.calls, "the action graph never ran"
    assert action.calls[0][1]["text_one_loop"] is True
    assert out in {"Starting it now.", "Started the home job."}


# ---- what the one loop carries over from chat ---------------------------------------------

def thread_with(graph, thread, *pairs):
    msgs = []
    for human, ai in pairs:
        msgs += [HumanMessage(content=human, additional_kwargs={"privacy": "private"}), AIMessage(content=ai)]
    graph.update_state({"configurable": {"thread_id": thread}}, {"messages": msgs}, as_node="chat")


def test_the_seed_keeps_her_replies_for_referents_and_the_200_message_window():
    graph = build_graph(NoCalls(), soul="s", history_window_messages=4)
    thread_with(graph, "person:o", ("the blue lamp in the den", "the blue one, got it"),
                ("and the kitchen?", "kitchen too"), ("what about it", "sure"))
    action = Action("ok")
    run("turn it on", graph=graph, action=action)
    seed = [m.content for m in action.calls[0][0]]
    assert seed[-1] == "turn it on"
    assert "sure" in seed                                    # her reply rides along (referents)
    assert "the blue lamp in the den" not in seed            # outside the graph's window


def test_the_default_window_is_two_hundred():
    assert build_graph(NoCalls(), soul="s").history_window_messages == 200


def test_her_old_emotion_tags_are_not_shown_on_a_typed_turn():
    graph = build_graph(NoCalls(), soul="s")
    thread_with(graph, "person:o", ("hey", "[warmly] Hey you."))
    action = Action("ok")
    run("what's up", graph=graph, action=action)
    shown = [m.content for m in action.calls[0][0] if isinstance(m, AIMessage)]
    assert shown == ["Hey you."]


def test_a_private_dm_sees_its_private_history_and_the_seed_is_room_gated():
    graph = build_graph(NoCalls(), soul="s")
    thread_with(graph, "person:o", ("my secret thing", "kept"))
    action = Action("ok")
    run("remind me what I told you", graph=graph, action=action)
    assert any(getattr(m, "content", "") == "my secret thing" for m in action.calls[0][0])


def test_one_durable_exchange_and_one_audit_row():
    graph = build_graph(NoCalls(), soul="s")
    action, rec = Action("hey you"), Recorder()
    run("hi", graph=graph, action=action, record=rec)
    msgs = graph.get_state({"configurable": {"thread_id": "person:o"}}).values["messages"]
    assert [(m.type, m.content) for m in msgs] == [("human", "hi"), ("ai", "hey you")]
    rows = rec.wait(1)
    time.sleep(0.05)
    assert len(rec.rows) == 1
    assert rows[0]["classifier_intent"] == "chat"
    assert ONE_LOOP_MARKER in (rows[0]["degraded"] or [])


def test_a_failure_records_a_row_and_raises():
    graph = build_graph(NoCalls(), soul="s")
    action, rec = Action(RuntimeError("boom")), Recorder()
    with pytest.raises(RuntimeError):
        run("hi", graph=graph, action=action, record=rec)
    rows = rec.wait(1)
    assert "turn_failed" in (rows[0]["degraded"] or [])


def test_a_rate_limit_is_said_honestly_and_lands_in_history():
    graph = build_graph(NoCalls(), soul="s")
    action, rec = Action(RuntimeError("You've hit your session limit · resets 7pm (UTC)")), Recorder()
    out = run("hi", graph=graph, action=action, record=rec)
    assert out
    msgs = graph.get_state({"configurable": {"thread_id": "person:o"}}).values["messages"]
    assert [m.type for m in msgs] == ["human", "ai"] and msgs[1].content == out


# ---- tier policy ----------------------------------------------------------------------------

def test_a_deep_verdict_is_served_standard_and_audited_without_spending_a_deep_credit():
    spent = []
    action, rec = Action("a long thoughtful answer"), Recorder()
    run("explain the history of the Roman calendar in depth", action=action, record=rec,
        router=route(tier="deep"), deep_allowed=lambda: spent.append(1) or True)
    assert spent == []
    row = rec.wait(1)[0]
    assert row["tier"] == "standard" and row["tier_override_source"] == "one_loop"
    assert action.calls[0][1]["tier"] == "standard"


# ---- the checks that stay, as postconditions on what she says -----------------------------

def test_a_claim_with_zero_tools_is_bounced_once():
    action = Action("I've started a home job to update the skill.", "I haven't started it yet.")
    out = run("you are good to update your skill", action=action)
    assert out == "I haven't started it yet."
    assert len(action.calls) == 2
    assert action.calls[1][0][-1].content == ACTION_NO_TOOL_CORRECTION


def test_an_unrelated_read_does_not_back_a_claimed_job():
    action = Action(tool_then("list_async_tasks", "I’ve started a home job for it.", "No jobs."),
                    "Nothing is running yet; want me to start it?")
    out = run("you are good to update your skill", action=action)
    assert len(action.calls) == 2 and out.startswith("Nothing is running")


def test_a_real_start_backs_the_claim():
    action = Action(tool_then("start_async_task", "I've started a home job for it.", "started j1"))
    out = run("you are good to update your skill", action=action)
    assert len(action.calls) == 1 and out == "I've started a home job for it."


def test_a_claim_that_survives_the_bounce_is_marked():
    rec = Recorder()
    action = Action("I've sent it to Kael.", "I've sent it to Kael.")
    run("tell kael hi", action=action, record=rec)
    assert "no_tool_action" in (rec.wait(1)[0]["degraded"] or [])


def test_an_audio_claim_is_bounced():
    from aerys_v2.service import AUDIO_CLAIM_CORRECTION
    action = Action("I heard the music playing on the speakers!", "I can't hear audio, but tell me the tune.")
    out = run("did you like my humming", action=action)
    assert action.calls[1][0][-1].content == AUDIO_CLAIM_CORRECTION
    assert out.startswith("I can't hear")


def test_a_record_claim_needs_a_write():
    from aerys_v2.service import RECORD_CLAIM_CORRECTION
    action = Action("Logged: 182 lbs this morning.", "I haven't saved that yet.")
    run("I weighed 182 this morning", action=action)
    assert action.calls[1][0][-1].content == RECORD_CLAIM_CORRECTION
    backed = Action(tool_then("remember", "Logged: 182 lbs this morning.", "saved"))
    run("I weighed 182 this morning", action=backed)
    assert len(backed.calls) == 1


def test_a_running_job_claim_needs_a_look():
    action = Action("The MapTap run is still going on my home machine.", "Let me look.")
    run("how's maptap going", action=action)
    assert len(action.calls) == 2
    looked = Action(tool_then("list_async_tasks", "The MapTap run is still going.", "maptap: running"))
    run("how's maptap going", action=looked)
    assert len(looked.calls) == 1


def test_plain_conversation_is_never_bounced():
    action = Action("Morning! I'm good, how did you sleep?")
    run("morning, you good?", action=action)
    assert len(action.calls) == 1


# ---- the outside guard still holds ---------------------------------------------------------

def test_the_turn_carries_the_outside_text_holder_and_the_same_gated_graph():
    action = Action("ok")
    run("hi", action=action)
    conf = action.calls[0][1]
    assert "outside_text" in conf and conf["identity"]["user_id"] == OWNER


# ---- the word lists, now postconditions, learn the 10:55 phrasing --------------------------

def test_the_detectors_know_the_1055_wording():
    assert claims_effect_done("I’ve started a home job to update the MapTap skill, Chris.")
    assert claims_effect_done("I've kicked off the run.")
    reply = "I’ve started a home job to update the MapTap skill, Chris."
    assert _claims_effect_without_doing_it("you are good to update your skill", reply)
    assert _claims_effect_without_doing_it("feel free to start it", reply)
    assert _claims_effect_without_doing_it("you have my go", reply)
    assert _promises_effect_without_doing_it("you're good to update your skill", "I'll start it now.")
    assert not _claims_effect_without_doing_it("remember when you started that job?", "Yes, I've started one before.")


# ---- the prompt: her conversational voice, full memory ------------------------------------

def test_the_one_loop_prompt_has_her_soul_the_text_banter_overlay_and_full_memory():
    pytest.importorskip("langchain.agents")
    from aerys_v2.action_loop import build_agent_action_graph
    from aerys_v2.factory import TEXT_BANTER_OVERLAY
    from test_action_loop_agent import Rec, TOOLS

    asked = []
    profile = lambda pid, q, priv: asked.append("profile") or "PROFILE"     # noqa: E731
    full = lambda pid, q, priv: asked.append("full") or "FULL MEMORY"       # noqa: E731

    def graph_with(*replies):
        log = []
        conv = Rec(label="conv", script=list(replies), log=log)
        std = Rec(label="std", script=list(replies), log=log)
        g = build_agent_action_graph(
            factory_models(conv, std), "SOUL", TOOLS, context_fn=profile, conversational_context_fn=full)
        return g, log

    def factory_models(conv, std):
        from aerys_v2.action_loop import AgentToolModels
        return AgentToolModels(conv, (std, False))

    g, log = graph_with(AIMessage(content="hey"))
    g.invoke({"messages": [HumanMessage(content="hi")]},
             {"configurable": {"thread_id": "person:o", "identity": dm(), "specialist": False,
                               "text_one_loop": True}, "recursion_limit": 10})
    system = log[0]["messages"][0][1]
    assert system.startswith("SOUL") and TEXT_BANTER_OVERLAY in system and "FULL MEMORY" in system
    assert asked[:1] == ["full"]

    asked.clear()
    g, log = graph_with(AIMessage(content="done"))
    g.invoke({"messages": [HumanMessage(content="lights off")]},
             {"configurable": {"thread_id": "person:o", "identity": dm(), "specialist": True},
              "recursion_limit": 10})
    system = log[0]["messages"][0][1]
    assert TEXT_BANTER_OVERLAY not in system and "PROFILE" in system and asked[:1] == ["profile"]

    asked.clear()
    g, log = graph_with(AIMessage(content="hey"))
    g.invoke({"messages": [HumanMessage(content="hi")]},
             {"configurable": {"thread_id": "person:o", "identity": {**dm(), "voice": True}},
              "recursion_limit": 10})
    assert asked[:1] == ["profile"]                         # voice banter is unchanged
