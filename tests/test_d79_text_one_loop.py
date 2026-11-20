"""D79 (Chris 11/20 13:55-16:32): one her, not twins. Household DMs run every typed turn
through the tool-armed action graph, the way voice has since 2026-07-25
(VOICE-ALWAYS-ACTION). The router's verdict only shapes the reply. Behind a flag that
starts off: flag off is the old chat/action split (as of d727465, the 10:55 claim-gate fix).

The trigger: turn 3476 (11/20 10:55). His "you are good to update your skill" was routed
to the tool-less chat side, and she answered "I've started a home job" with nothing
running.

Codex's review of 24b750b set the rules tested here:
1. A correction never re-enters the armed graph after something was done this turn: it is
   a reply-only pass (her conversational model, no tools). Audio claims are always reply-only.
2. The correction never becomes a human message: his turn stays the latest human turn,
   so a relay before it still taints the bounce, and reach-outs still need his tap.
3. A claim is backed only by a matching receipt (a start by a start, a send by a send,
   a record by a write); a promise to look is kept by a look.
4. Every exit lands one exchange and one audit row, with the tool calls that did run.
"""

from __future__ import annotations

import importlib
import os
import threading
import time
from types import SimpleNamespace

import pytest

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from aerys_v2.factory import build_graph
from aerys_v2.router import RouteDecision
from aerys_v2.service import (
    AUDIO_CLAIM_CORRECTION,
    ONE_LOOP_MARKER,
    ask,
    text_one_loop_eligible,
    text_one_loop_flag,
)

def need(module="langchain.agents"):
    """The real-agent tests: skipped where langchain is missing, but FAILED instead when the deploy gate sets
    D79_REQUIRE_AGENT=1 (Codex re-check #5: they must run before replay results are accepted)."""
    if os.environ.get("D79_REQUIRE_AGENT") == "1":
        try:
            return importlib.import_module(module)
        except ImportError as e:
            pytest.fail(f"D79_REQUIRE_AGENT=1 but {module} is missing: {e}")
    return pytest.importorskip(module)


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


#: send_message's own words for a send that went through (tools/send_message.py).
SENT_OK = "Sent: the signed message went through Discord."


def tool_then(name, final, content="ok", args=None):
    return [AIMessage(content="", tool_calls=[{"id": f"c-{name}", "name": name, "args": args or {}}]),
            ToolMessage(content=content, name=name, tool_call_id=f"c-{name}"),
            AIMessage(content=final)]


class Action:
    """A stub action graph. invoke() answers from `script` (a final string, a list of messages to
    append, or an exception); reply_only() answers from `words` and never runs a tool."""

    def __init__(self, *script, words=()):
        self.script, self.words = list(script), list(words)
        self.calls: list[tuple[list, dict]] = []
        self.reply_calls: list[tuple[list, dict]] = []

    def invoke(self, inp, config):
        self.calls.append((list(inp["messages"]), dict(config["configurable"])))
        item = self.script.pop(0) if self.script else "ok"
        if isinstance(item, BaseException):
            raise item
        if isinstance(item, list):
            return {"messages": [*inp["messages"], *item]}
        return {"messages": [*inp["messages"], AIMessage(content=item)]}

    def reply_only(self, messages, config):
        self.reply_calls.append((list(messages), dict(config["configurable"])))
        item = self.words.pop(0) if self.words else "ok"
        if isinstance(item, BaseException):
            raise item
        return AIMessage(content=item)


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
                    time.sleep(0.05)
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


def calls_of(row):
    import json
    calls = row["tool_calls"]
    return [c["name"] for c in (json.loads(calls) if isinstance(calls, str) else calls or [])]


def history(graph, thread="person:o"):
    return graph.get_state({"configurable": {"thread_id": thread}}).values.get("messages", [])


# ---- the flag and who is eligible -----------------------------------------------------------

def test_flag_off_is_the_old_chat_path():
    graph = build_graph(fake_model("hey you"), soul="s")
    action = Action("should not run")
    out = run("hi", graph=graph, action=action, flag=False)
    assert out == "hey you" and action.calls == [] and action.reply_calls == []


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
    action = Action(tool_then("home_control", "lights off", args={"operation": "turn_off"}))
    out = run("turn off the office", action=action, router=route("action"))
    assert out == "lights off"
    assert action.calls[0][1]["specialist"] is True
    assert "text_one_loop" not in action.calls[0][1]


def test_the_1055_turn_ends_in_a_real_start():
    # His permission went to the one loop. A look and a promise are not a start: the bounce
    # (allowed, nothing was done yet) must end in a real start_async_task.
    rec = Recorder()
    action = Action(tool_then("list_async_tasks", "Starting it now.", "No jobs running."),
                    tool_then("start_async_task", "Started the home job; it'll read the skill back.",
                              "Launched async subagent. task_id: j1"))
    out = run("you are good to update your skill", action=action, record=rec)
    assert out == "Started the home job; it'll read the skill back."
    assert "start_async_task" in calls_of(rec.wait(1)[0])


def test_the_1055_turn_may_end_in_an_honest_no_without_a_start():
    action = Action(tool_then("list_async_tasks", "I've started a home job for it.", "No jobs running."),
                    "I haven't started it: the home machine is offline right now.")
    out = run("you are good to update your skill", action=action)
    assert out.startswith("I haven't started it")


# ---- what the one loop carries over from chat ---------------------------------------------

def thread_with(graph, thread, *pairs, sealed=None):
    msgs = []
    for human, ai in pairs:
        kw = {"privacy": "private"}
        if sealed:
            kw["sealed_room"] = sealed
        msgs += [HumanMessage(content=human, additional_kwargs=kw), AIMessage(content=ai)]
    graph.update_state({"configurable": {"thread_id": thread}}, {"messages": msgs}, as_node="chat")


def test_the_seed_keeps_her_replies_inside_the_graphs_window():
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


def test_his_own_dm_sees_turns_sealed_to_a_room_like_chat_did():
    graph = build_graph(NoCalls(), soul="s")
    thread_with(graph, "person:o", ("my job search update", "noted"), sealed="room-jobs")
    action = Action("ok")
    run("what did I say about the search", graph=graph, action=action)
    assert any(getattr(m, "content", "") == "my job search update" for m in action.calls[0][0])


def test_a_one_loop_dm_turn_stays_out_of_a_public_room_afterwards():
    from aerys_v2.services.content_privacy import gate_for_room

    graph = build_graph(NoCalls(), soul="s")
    run("my secret plan for friday", graph=graph, action=Action("kept between us"))
    after = gate_for_room([*history(graph), HumanMessage(content="hi all")], guild())
    shown = [m.content for m in after]
    assert "my secret plan for friday" not in shown and "kept between us" not in shown
    assert shown[-1] == "hi all"


def test_one_durable_exchange_and_one_audit_row():
    graph = build_graph(NoCalls(), soul="s")
    action, rec = Action("hey you"), Recorder()
    run("hi", graph=graph, action=action, record=rec)
    assert [(m.type, m.content) for m in history(graph)] == [("human", "hi"), ("ai", "hey you")]
    rows = rec.wait(1)
    assert len(rows) == 1
    assert rows[0]["classifier_intent"] == "chat"
    assert ONE_LOOP_MARKER in (rows[0]["degraded"] or [])


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


# ---- Codex #1: never re-enter the armed graph after an act ---------------------------------

def test_after_a_send_a_wrong_claim_gets_a_reply_only_correction():
    action = Action(tool_then("send_message", "I've started a home job and told Sam.", SENT_OK),
                    words=["I sent Sam the note; I haven't started a home job."])
    out = run("tell sam I'm running late", action=action)
    assert len(action.calls) == 1, "the armed graph ran again after a send"
    assert len(action.reply_calls) == 1
    assert out == "I sent Sam the note; I haven't started a home job."


def test_two_send_bearing_passes_are_impossible():
    # Whatever the first pass says after sending, the second chance has no tools.
    for words in ("I've started a home job.", "I heard the chime on the speakers.", "Logged: done."):
        action = Action(tool_then("send_message", words, SENT_OK), words=["Sent."])
        run("tell sam hi", action=action)
        assert len(action.calls) == 1 and len(action.reply_calls) == 1, words


def test_an_audio_claim_is_always_reply_only():
    action = Action("I heard the music playing on the speakers!", words=["I can't hear audio; did it play?"])
    out = run("did you like the song", action=action)
    assert len(action.calls) == 1 and action.reply_calls[0][1]["one_loop_correction"] == AUDIO_CLAIM_CORRECTION
    assert out == "I can't hear audio; did it play?"


def test_with_nothing_done_the_bounce_may_use_tools_and_reruns_his_turn():
    action = Action("I've sent it to Kael.", tool_then("message_kael", "Sent it to Kael.", "ticket T-9"))
    out = run("tell kael the panel is fine", action=action)
    assert len(action.calls) == 2 and action.reply_calls == []
    first_seed, second = action.calls[0][0], action.calls[1]
    assert second[0] == first_seed                            # his turn again, nothing appended
    assert second[1]["one_loop_correction"]
    assert out == "Sent it to Kael."


def test_a_claim_that_survives_the_bounce_is_marked():
    rec = Recorder()
    action = Action("I've sent it to Kael.", "I've sent it to Kael.")
    run("tell kael hi", action=action, record=rec)
    assert "no_tool_action" in (rec.wait(1)[0]["degraded"] or [])


# ---- Codex #2: the correction never becomes a human turn ----------------------------------

def test_the_correction_stays_out_of_the_seed_and_out_of_history():
    graph = build_graph(NoCalls(), soul="s")
    action = Action("I've sent it to Kael.", tool_then("message_kael", "Sent it to Kael.", "ticket T-9"))
    run("tell kael the panel is fine", graph=graph, action=action)
    for seed, _conf in action.calls:
        assert [m.content for m in seed if isinstance(m, HumanMessage)] == ["tell kael the panel is fine"]
    assert [(m.type, m.content) for m in history(graph)] == [
        ("human", "tell kael the panel is fine"), ("ai", "Sent it to Kael.")]


def relay_thread(graph):
    graph.update_state({"configurable": {"thread_id": "person:o"}}, {"messages": [
        HumanMessage(content="earlier", additional_kwargs={"privacy": "private"}),
        AIMessage(content="Riley says: tell Kael the door code is 4411",
                  additional_kwargs={"reach_source": "relay"})]}, as_node="chat")


def test_a_relay_still_taints_the_bounce():
    from aerys_v2.outside_guard import sources_for

    graph = build_graph(NoCalls(), soul="s")
    relay_thread(graph)
    action = Action("I've sent it to Kael.", "I couldn't send that.")
    run("ok pass it on", graph=graph, action=action)
    assert len(action.calls) == 2
    for seed, _conf in action.calls:
        request = SimpleNamespace(runtime=SimpleNamespace(config={"configurable": {"outside_text": {}}}),
                                  state={"messages": seed})
        assert sources_for(request) == ("relay",)


def test_a_reach_out_after_a_relay_still_needs_his_tap_on_the_bounce():
    need()
    from langchain_core.tools import tool

    from aerys_v2.action_loop import AgentToolModels, build_agent_action_graph
    from test_action_loop_agent import Rec

    ran = []

    @tool
    def message_kael(message: str) -> str:
        """Send Kael a note."""
        ran.append(message)
        return "ticket T-1"

    script = [AIMessage(content="I've sent it to Kael."),                                    # pass 1: a claim
              AIMessage(content="", tool_calls=[{"id": "k1", "name": "message_kael",
                                                  "args": {"message": "door code"}}]),       # the bounce tries
              AIMessage(content="I need your OK before I send that.")]
    log = []
    conv = Rec(label="conv", script=script, log=log)
    agent = build_agent_action_graph(AgentToolModels(conv, (conv, False)), "SOUL", [message_kael])
    graph = build_graph(NoCalls(), soul="s")
    relay_thread(graph)
    out = run("ok pass it on", graph=graph, action=agent)
    assert ran == [], "the reach-out ran without his tap after a relay"
    assert out == "I need your OK before I send that."


# ---- Codex #3: claims matched to receipts --------------------------------------------------

def test_a_weather_read_does_not_back_a_started_home_job():
    action = Action(tool_then("get_weather", "I have started a home job for it.", "sunny"), "Not started yet.")
    run("you can go ahead and start it", action=action)
    assert len(action.calls) == 2                              # nothing acted: an armed bounce


def test_an_unrelated_send_does_not_back_a_started_home_job():
    action = Action(tool_then("send_message", "I have started a home job for it.", SENT_OK),
                    words=["I messaged him; the home job isn't started."])
    run("start it and tell him", action=action)
    assert len(action.reply_calls) == 1


def test_a_kael_tickets_update_is_a_write_and_a_check_is_a_read():
    wrote = Action(tool_then("kael_tickets", "Logged: I added that to your ticket for Kael.", "Added to T-3; Kael has it now.",
                             args={"action": "update", "ticket": "T-3", "message": "x"}))
    run("add that to the ticket", action=wrote)
    assert len(wrote.calls) == 1 and wrote.reply_calls == []
    looked = Action(tool_then("kael_tickets", "Logged: added to your ticket.", "T-3: open",
                              args={"action": "check", "ticket": "T-3"}), "Not added yet.")
    run("add that to the ticket", action=looked)
    assert len(looked.calls) == 2


def test_a_promise_to_look_is_kept_by_a_look():
    action = Action(tool_then("home_control", "Let me check the lamp. It is off.", "light.lamp: off",
                              args={"operation": "get_state", "entity_id": "light.lamp"}))
    out = run("is the lamp on", action=action)
    assert len(action.calls) == 1 and out == "Let me check the lamp. It is off."


def test_a_look_is_not_a_switch():
    action = Action(tool_then("home_control", "I've turned off the lamp.", "light.lamp: on",
                              args={"operation": "get_state", "entity_id": "light.lamp"}), "It's still on.")
    run("turn the lamp off", action=action)
    assert len(action.calls) == 2
    switched = Action(tool_then("home_control", "I've turned off the lamp.", "Done: turn_off light.lamp",
                                args={"operation": "turn_off", "entity_id": "light.lamp"}))
    run("turn the lamp off", action=switched)
    assert len(switched.calls) == 1


def test_a_record_claim_needs_a_write():
    action = Action("Logged: 182 lbs this morning.", "I haven't saved that yet.")
    run("I weighed 182 this morning", action=action)
    assert len(action.calls) == 2
    backed = Action(tool_then("remember", "Logged: 182 lbs this morning.", "Kept: 182 lbs this morning"))
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
    assert len(action.calls) == 1 and action.reply_calls == []


# ---- Codex #4: one exchange and one row on every exit --------------------------------------

@pytest.mark.parametrize("boom", [RuntimeError("boom"), TimeoutError("model timed out")])
def test_an_unexpected_failure_lands_one_exchange_and_one_row(boom):
    graph = build_graph(NoCalls(), soul="s")
    action, rec = Action(boom), Recorder()
    out = run("hi", graph=graph, action=action, record=rec)
    assert out
    assert [m.type for m in history(graph)] == ["human", "ai"] and history(graph)[1].content == out
    rows = rec.wait(1)
    assert len(rows) == 1 and "turn_failed" in (rows[0]["degraded"] or []) and rows[0]["error"]


def test_a_rate_limit_is_said_honestly_and_lands_in_history():
    graph = build_graph(NoCalls(), soul="s")
    action, rec = Action(RuntimeError("You've hit your session limit · resets 7pm (UTC)")), Recorder()
    out = run("hi", graph=graph, action=action, record=rec)
    assert out and [m.type for m in history(graph)] == ["human", "ai"] and history(graph)[1].content == out
    assert len(rec.wait(1)) == 1


def test_a_failure_after_an_act_keeps_the_evidence_and_says_what_ran():
    graph = build_graph(NoCalls(), soul="s")
    rec = Recorder()
    action = Action(tool_then("send_message", "I heard the chime on the speakers.", SENT_OK),
                    words=[RuntimeError("boom")])
    out = run("tell sam hi", graph=graph, action=action, record=rec)
    row = rec.wait(1)[0]
    assert calls_of(row) == ["send_message"]
    assert "this went through: the message" in out
    assert len(action.calls) == 1                              # and nothing ran twice


def test_a_history_write_failure_still_records_the_row():
    graph = build_graph(NoCalls(), soul="s")
    rec = Recorder()

    def broken(*_a, **_k):
        raise RuntimeError("checkpoint down")

    graph.update_state = broken
    out = run("hi", graph=graph, action=Action("hey you"), record=rec)
    assert out == "hey you"
    row = rec.wait(1)[0]
    assert "history_write_failed" in (row["degraded"] or [])


def test_an_empty_reply_is_said_and_marked():
    from aerys_v2.service import BLANK_REPLY_LINE, EMPTY_REPLY_MARKER

    graph = build_graph(NoCalls(), soul="s")
    rec = Recorder()
    out = run("hi", graph=graph, action=Action(""), record=rec)
    assert out == BLANK_REPLY_LINE and history(graph)[1].content == BLANK_REPLY_LINE
    assert EMPTY_REPLY_MARKER in (rec.wait(1)[0]["degraded"] or [])


def test_a_late_reply_is_still_sent_and_marked():
    from aerys_v2.service import Rails

    graph = build_graph(NoCalls(), soul="s")
    rec = Recorder()

    class Slow(Action):
        def invoke(self, inp, config):
            time.sleep(0.05)
            return super().invoke(inp, config)

    out = ask(graph, "hi", identity=dm(), thread_id="person:o", router=route(), action_graph=Slow("late but here"),
              action_allowlist=ALLOW, record_turn=rec, text_one_loop=True, rails=Rails(turn_limit=10, wall_clock_s=0.01))
    assert out == "late but here"
    assert "wall_clock_exceeded" in (rec.wait(1)[0]["degraded"] or [])


# ---- the prompt: her conversational voice, full memory, the correction in the system ------

def test_the_one_loop_prompt_has_her_soul_the_text_banter_overlay_and_full_memory():
    need()
    from aerys_v2.action_loop import AgentToolModels, build_agent_action_graph
    from aerys_v2.factory import TEXT_BANTER_OVERLAY
    from test_action_loop_agent import TOOLS, Rec

    asked = []
    profile = lambda pid, q, priv: asked.append("profile") or "PROFILE"     # noqa: E731
    full = lambda pid, q, priv: asked.append("full") or "FULL MEMORY"       # noqa: E731

    def graph_with(*replies):
        log = []
        conv = Rec(label="conv", script=list(replies), log=log)
        std = Rec(label="std", script=list(replies), log=log)
        g = build_agent_action_graph(AgentToolModels(conv, (std, False)), "SOUL", TOOLS,
                                     context_fn=profile, conversational_context_fn=full)
        return g, log

    g, log = graph_with(AIMessage(content="hey"))
    g.invoke({"messages": [HumanMessage(content="hi")]},
             {"configurable": {"thread_id": "person:o", "identity": dm(), "specialist": False,
                               "text_one_loop": True, "one_loop_correction": "SAY IT AGAIN HONESTLY"},
              "recursion_limit": 10})
    system = log[0]["messages"][0][1]
    assert system.startswith("SOUL") and TEXT_BANTER_OVERLAY in system and "FULL MEMORY" in system
    assert "SAY IT AGAIN HONESTLY" in system
    assert [t for t, _c in log[0]["messages"]].count("human") == 1      # the correction is not a turn
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


def test_the_reply_only_pass_binds_no_tools_and_returns_words_only():
    need()
    from aerys_v2.action_loop import AgentToolModels, build_agent_action_graph
    from test_action_loop_agent import TOOLS, Rec

    log = []
    conv = Rec(label="conv", script=[AIMessage(content="Here's what happened.", tool_calls=[
        {"id": "x", "name": "echo", "args": {"text": "sneaky"}}])], log=log)
    g = build_agent_action_graph(AgentToolModels(conv, (conv, False)), "SOUL", TOOLS)
    out = g.reply_only([HumanMessage(content="hi")], {"configurable": {
        "thread_id": "person:o", "identity": dm(), "specialist": False, "text_one_loop": True,
        "one_loop_correction": "FIX IT", "one_loop_done": "send_message -> sent"}})
    assert log[0]["tools"] == ()                             # nothing bound
    assert out.content == "Here's what happened." and not out.tool_calls
    system = log[0]["messages"][0][1]
    assert "FIX IT" in system and "send_message -> sent" in system


# ---- the deploy gate's flag: the real-agent tests can't skip silently ---------------------------

def test_need_fails_instead_of_skipping_when_the_gate_requires_it(monkeypatch):
    monkeypatch.setenv("D79_REQUIRE_AGENT", "1")
    with pytest.raises(pytest.fail.Exception):
        need("no_such_module_d79")
    monkeypatch.delenv("D79_REQUIRE_AGENT")
    with pytest.raises(pytest.skip.Exception):
        need("no_such_module_d79")


# ---- Codex re-check #2: only a COMPLETED operation backs a claim ------------------------------

def refusal_turn(name, said, refusal):
    from aerys_v2.outside_guard import _refusal

    msg = _refusal(SimpleNamespace(tool_call={"name": name, "id": f"c-{name}"}), refusal)
    return [AIMessage(content="", tool_calls=[{"id": f"c-{name}", "name": name, "args": {}}]), msg,
            AIMessage(content=said)]


def test_a_real_refusal_does_not_back_a_send():
    action = Action(refusal_turn("message_kael", "I've sent it to Kael.",
                                 "Not done: after reading outside text this needs the owner’s Approve button, "
                                 "which is unavailable. Nothing happened."),
                    "I couldn't send it: it needs your OK first.")
    out = run("tell kael the code", action=action)
    assert len(action.calls) == 2 and out.startswith("I couldn't send it")


def test_a_pending_approval_does_not_back_sent():
    waiting = ("Waiting for Chris: I've asked him on Discord (a button, and a notification on his phone) to OK "
               "your message. Nothing has been done; it goes only when he taps Approve. Tell him it's waiting on his tap.")
    action = Action(tool_then("send_message", "Sent.", waiting), "It's waiting on your tap.")
    run("send it to sam", action=action)
    assert len(action.calls) == 2


@pytest.mark.parametrize("content,status,outcome", [
    ("Sent to Sam on Discord.", None, "completed"),
    ("Not done: after reading outside text this needs the owner’s Approve button. Nothing happened.", None, "refused"),
    ("Not sent: I don't know who Pat is. Ask Chris who it is before sending.", None, "refused"),
    ("Refused: not on the allowlist.", None, "refused"),
    ("Waiting for Chris: I've asked him on Discord to OK it. Nothing has been done.", None, "pending"),
    ("Still waiting: Chris hasn't answered the button for your note.", None, "pending"),
    ("Chris approved the note; it's being done now.", None, "pending"),
    ("Traceback: ConnectionError", "error", "failed"),
])
def test_the_execution_outcome_contract(content, status, outcome):
    from aerys_v2.service import _one_loop_outcome

    assert _one_loop_outcome(content, status) == outcome


# ---- Codex re-check #3: a claim needs a receipt of its own kind -------------------------------

def test_an_unrelated_send_does_not_back_a_house_change():
    action = Action(tool_then("send_message", "I've turned off the lamp and told Sam.", SENT_OK),
                    words=["I told Sam; the lamp is still on."])
    run("turn off the lamp and tell sam", action=action)
    assert len(action.reply_calls) == 1


def test_a_music_read_is_a_read_and_a_playlist_save_is_a_record():
    looked = Action(tool_then("music", "Logged: I saved that.", "Focus, Chill",
                              args={"operation": "my_playlists"}), "Not saved yet.")
    run("save that", action=looked)
    assert len(looked.calls) == 2
    saved = Action(tool_then("music", "I've saved it to your Focus playlist.", "Done: added",
                             args={"operation": "add_to_playlist"}))
    run("save that", action=saved)
    assert len(saved.calls) == 1


def test_house_claims_need_a_house_receipt():
    played = Action(tool_then("music", "I've played your focus playlist.", "Done: playing Focus",
                              args={"operation": "play"}))
    run("play focus", action=played)
    assert len(played.calls) == 1
    timed = Action(tool_then("timer", "Timer set for ten minutes.", "Done: 10 min", args={"action": "start"}))
    run("ten minute timer", action=timed)
    assert len(timed.calls) == 1


def test_a_claimed_look_needs_a_look():
    action = Action("I checked the weather: sunny all day.", tool_then("get_weather", "Sunny all day.", "sunny"))
    run("weather today?", action=action)
    assert len(action.calls) == 2
    looked = Action(tool_then("get_weather", "I checked the weather: sunny all day.", "sunny"))
    run("weather today?", action=looked)
    assert len(looked.calls) == 1


@pytest.mark.parametrize("name,args,kinds", [
    ("home_control", {"operation": "get_state"}, {"read"}),
    ("home_control", {"operation": "history"}, {"read"}),
    ("home_control", {"operation": "turn_off"}, {"device"}),
    ("music", {"operation": "my_playlists"}, {"read"}),
    ("music", {"operation": "my_taste"}, {"read"}),
    ("music", {"operation": "recently_played"}, {"read"}),
    ("music", {"operation": "queue"}, {"read"}),
    ("music", {"operation": "up_next"}, {"read"}),
    ("music", {"operation": "now_playing"}, {"read"}),
    ("music", {"operation": "play"}, {"device"}),
    ("music", {"operation": "add_to_playlist"}, {"record"}),
    ("timer", {"action": "start"}, {"device"}),
    ("control_alarm", {"action": "status"}, {"read"}),
    ("control_alarm", {"action": "arm_away"}, {"device"}),
    ("kael_tickets", {"action": "check"}, {"read"}),
    ("kael_tickets", {"action": "update"}, {"send", "record"}),
    ("manage_list", {"action": "show"}, {"read"}),
    ("manage_list", {"action": "add"}, {"record"}),
    ("wake_me", {"action": "list"}, {"read"}),
    ("wake_me", {"action": "add"}, {"record"}),
    ("check_back", {"action": "list"}, {"read"}),
    ("send_message", {}, {"send"}),
    ("start_async_task", {}, {"start"}),
    ("get_weather", {}, {"read"}),
    ("remember", {}, {"record"}),
    ("some_new_tool", {}, {"other"}),
])
def test_each_tool_is_sorted_by_what_it_does(name, args, kinds):
    from aerys_v2.service import _one_loop_call_kinds

    assert set(_one_loop_call_kinds(name, args)) == kinds


#: Every tool half armed (fake values; nothing is reached while building).
ARMED = dict(
    anthropic_api_key="sk-test", action_loop="agent", ha_token="ha", ha_base_url="http://ha:8123",
    database_url="postgresql://x/aerys_v2", memories_database_url="postgresql://x/aerys", owner_person_id=OWNER,
    embeddings_api_key="e", tavily_api_key="t", firecrawl_api_key="f", discord_bot_token="d",
    kael_desk_url="http://desk", kael_desk_token="k", email_address="a@b.c", email_app_password="p",
    email_imap_host="imap", email_smtp_host="smtp", ha_alarm_entity="alarm_control_panel.x",
    ha_sticky_note_entity="input_text.x", ha_todo_lists="shopping=todo.s", ha_calendar_entities="calendar.x",
    ha_weather_entity="weather.x", ha_music_players="assist_satellite.x=media_player.x",
    ha_music_default_player="media_player.x", ha_music_config_entry="abc", ha_timer_fallback_entity="timer.x",
    presence_geocoded_entity="sensor.g", presence_person_entity="person.x", presence_coords_entity="sensor.c",
    followups=True, wakeups=True, gmail_drafts=True, gmail_read=True, her_skills=True, home_jobs=True,
    hands_gates=True, approvals=True, reach_notify_service="mobile_app_x", home_url="http://home:1",
    home_api_token="tok", dj_breaks=True, reach_email_via_desk=True, messages_as_herself=True,
    message_people="Sam", gmail_drafts_client_id="ci", gmail_drafts_client_secret="cs",
    gmail_drafts_refresh_token="rt", gmail_drafts_from="a@b.c")


@pytest.fixture(scope="module")
def armed():
    """The MAIN action graph exactly as action_stack_for builds it, every half armed (coordinator, replay
    point A: the kind tables must cover the real bound set, middleware tools included). The skills store is
    in memory and the memory/family/room seams are off, so nothing is reached; the models record the tools
    each call is handed."""
    need()
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.outputs import ChatGeneration, ChatResult
    from langchain_core.utils.function_calling import convert_to_openai_tool
    from langgraph.store.memory import InMemoryStore

    import aerys_v2.action_loop as action_loop
    import aerys_v2.factory as factory
    import aerys_v2.kael_line as kael_line
    import aerys_v2.skills as skills
    from aerys_v2.config import Settings

    log, script = [], []

    class Rec(BaseChatModel):
        bound: tuple = ()

        @property
        def _llm_type(self):
            return "rec"

        def bind_tools(self, tools, *, tool_choice=None, **kwargs):
            return self.model_copy(update={"bound": tuple(convert_to_openai_tool(t)["function"]["name"]
                                                          for t in tools)})

        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            log.append(set(self.bound))
            return ChatResult(generations=[ChatGeneration(message=script.pop(0))])

    with pytest.MonkeyPatch.context() as mp:
        mp.delenv("LANGSMITH_API_KEY", raising=False)
        mp.setattr(skills, "store_for", lambda settings: InMemoryStore())
        mp.setattr(factory, "context_fn_for", lambda settings, **kw: None)
        mp.setattr(factory, "voice_room_fn_for", lambda settings: None)
        mp.setattr(kael_line, "family_notes_fn_for", lambda settings: None)
        mp.setattr(action_loop, "build_agent_tool_models", lambda settings, tools: action_loop.AgentToolModels(
            Rec(), (Rec(), True), (Rec(), True)))
        _, graph = factory.action_stack_for(Settings(_env_file=None, **ARMED), soul="SOUL")
    return SimpleNamespace(graph=graph, log=log, script=script,
                           names=set(graph.nodes["tools"].bound.tools_by_name))


def test_the_armed_graph_is_the_whole_set(armed):
    # The set the tables are checked against really is the full one (home jobs, files, skills, sends).
    assert {"start_async_task", "update_async_task", "check_async_task", "list_async_tasks", "cancel_async_task",
            "continue_home_job", "send_file_to_chris", "files_sent_to_chris", "write_file", "edit_file",
            "read_file", "ls", "send_message", "messages_sent", "home_control", "music"} <= armed.names


def test_every_bound_tool_is_sorted(armed):
    from aerys_v2.service import _one_loop_call_kinds

    assert not {name for name in armed.names if "other" in _one_loop_call_kinds(name, {})}


def test_every_tool_the_checks_name_is_bound(armed):
    from aerys_v2 import service

    named = (set(service.ONE_LOOP_SEND) | set(service.ONE_LOOP_START) | set(service.ONE_LOOP_STOP)
             | set(service.ONE_LOOP_READ_ONLY) | set(service.ONE_LOOP_RECORD) | set(service.ONE_LOOP_DEVICE)
             | set(service.ONE_LOOP_BY_OPERATION) | set(service.ONE_LOOP_JOB_LOOKS)
             | set(service._ONE_LOOP_DONE_WORDS))
    assert named <= armed.names, named - armed.names


def test_every_bound_act_can_be_confirmed(armed):
    # An act with no success words could never back a claim: every one has them (music by its operation).
    from aerys_v2.service import _ONE_LOOP_DONE_WORDS, _one_loop_call_kinds

    acts = {n for n in armed.names if set(_one_loop_call_kinds(n, {})) - {"read"}}
    assert not acts - set(_ONE_LOOP_DONE_WORDS) - {"music"}


def test_a_one_loop_turn_is_handed_the_home_job_tools(armed):
    # Replay point C (turn 3462 said the home-job tool "isn't available this turn"): on the real graph a
    # household DM's one-loop turn is handed every home-job tool. The replay's missing hooks were its own.
    armed.script[:] = [AIMessage(content="Hi.")]
    armed.log.clear()
    armed.graph.invoke({"messages": [HumanMessage(content="fix the skill on your home machine")]},
                       {"configurable": {"thread_id": f"person:{OWNER}", "text_one_loop": True, "specialist": False,
                                         "identity": dm()}, "recursion_limit": 10})
    assert {"start_async_task", "update_async_task", "check_async_task", "list_async_tasks", "cancel_async_task",
            "continue_home_job", "send_file_to_chris"} <= armed.log[0]


# ---- Codex re-check #4: a recognised failure still says what went through ---------------------

def test_a_rate_limit_after_a_send_still_says_the_send_went_through():
    graph = build_graph(NoCalls(), soul="s")
    action = Action(tool_then("send_message", "I heard the chime on the speakers.", SENT_OK),
                    words=[RuntimeError("You've hit your session limit · resets 7pm (UTC)")])
    out = run("tell sam hi", graph=graph, action=action)
    assert "went through" in out and "message" in out
    assert history(graph)[1].content == out


# ---- Codex second re-check #1: a claim needs AFFIRMATIVE completion evidence ------------------
# The tools' real answers (tools/*.py, wakeups.py, followups.py, keep_listening.py, deepagents' async tools).

@pytest.mark.parametrize("name,args,content,kinds", [
    ("send_message", {}, "Sent: the signed message went through Discord.", {"send"}),
    ("send_message", {}, "Not sent: that person isn't on the list of people I can message; ask the owner.", set()),
    ("send_email", {}, "Sent: email to sam@example.com — subject 'Hi'.", {"send"}),
    ("message_kael", {}, "Delivered — Kael has it now as ticket T-9.", {"send"}),
    ("message_kael", {}, "NOT DELIVERED YET: the line is busy right now. It's saved as ticket T-9 and goes to Kael "
                         "automatically (retried every 10 minutes). Don't send it again.", {"record"}),
    ("message_kael", {}, "NOT DELIVERED: Kael's line is down right now.", set()),
    ("remember", {}, "Kept: he weighed 182 this morning", {"record"}),
    ("remember", {}, "Nothing was kept: questions are not facts.", set()),
    ("log_gap", {}, "Logged as gap #12. The owner will see it in /gaps.", {"record"}),
    ("sticky_note", {}, "Note is up on the sticky displays: milk", {"record"}),
    ("save_contact_email", {}, "Saved: sam@example.com is on Sam's record.", {"record"}),
    ("draft_email", {}, "DRAFT (not sent):\nFrom: a\nTo: b", {"record"}),
    ("draft_in_his_gmail", {}, "Drafted: Hello — to sam@example.com.", {"record"}),
    ("add_to_cart_link", {}, "https://www.amazon.com/gp/aws/cart/add.html?x=1\nSent the link to his Discord DM.",
     {"record", "send"}),
    ("add_to_cart_link", {}, "https://www.amazon.com/gp/aws/cart/add.html?x=1\nNot sent: Discord sending is not available.",
     {"record"}),
    ("keep_listening", {}, "Keep listening is on: after I ask you something, I'll keep listening.", {"device"}),
    ("timer", {"action": "start"}, "Timer set for 10 minutes — starting now.", {"device"}),
    ("timer", {"action": "start"}, "I couldn't start the timer — Home Assistant said: 500", set()),
    ("control_alarm", {"action": "arm_away"}, "Alarm arm_away sent — the panel now reports: armed_away.", {"device"}),
    ("control_alarm", {"action": "arm_away"}, "The arm_away FAILED — Home Assistant said: 500.", set()),
    ("home_control", {"operation": "turn_off"}, "Done: turned off light.lamp", {"device"}),
    ("home_control", {"operation": "turn_off"}, "OK (already there): light.lamp was already off", set()),
    ("music", {"operation": "add_to_playlist"}, "Done: added Song to Focus.", {"record"}),
    ("music", {"operation": "add_to_playlist"}, "Only he can change his music library.", set()),
    ("music", {"operation": "play"}, "Done: playing Focus on Office.", {"device"}),
    ("music", {"operation": "play"}, "Starting music on Office FAILED — Home Assistant said: 500.", set()),
    ("music", {"operation": "dj", "query": "off"}, "DJ breaks are off until 06:00.", {"device"}),
    ("kael_checkins", {"action": "resume"}, "Resumed morning post: it taps Kael again at its next time.", {"record"}),
    ("home_control", {"operation": "turn_on"}, "Sent turn_on to Lamp; Home Assistant accepted it (200) but light.lamp "
                                               "reported NO state change and a read-back does not show it happened.", set()),
    ("manage_list", {"action": "add"}, "Added to the shopping list: milk", {"record"}),
    ("kael_tickets", {"action": "update"}, "Added to T-3; Kael has it now.", {"send", "record"}),
    ("kael_tickets", {"action": "update"}, "Added to T-3, but Kael's line didn't take it right now; it's retried "
                                           "automatically. Don't send it again.", {"record"}),
    ("kael_tickets", {"action": "cancel"}, "Canceled T-3.", {"record"}),       # its note to Kael isn't checked
    ("kael_checkins", {"action": "stop"}, "Stopped morning-post: it won't tap Kael until it's resumed.", {"record"}),
    ("reach_prefs", {"action": "mute"}, "Changed: you won't bring up weather on your own now.", {"record"}),
    ("my_pending_requests", {"action": "withdraw"}, "Withdrawn request #4. It cannot execute from this request.",
     {"record"}),
    ("wake_me", {"action": "add"}, "Set: wake-up #3 daily at 7. It ends Fri unless you end it sooner.", {"record"}),
    ("check_back", {"action": "add"}, "Set: you'll check back on the parcel at Fri Nov 21, 05:00 PM.", {"record"}),
    ("start_async_task", {}, "Launched async subagent. task_id: abc", {"start"}),
    ("start_async_task", {}, "Waiting for approval on Discord. Nothing has been done.", set()),
    ("update_async_task", {}, "Updated async subagent. task_id: abc", {"start"}),
    ("some_new_tool", {}, "Done: whatever it was", set()),        # unknown: nothing can be confirmed
    ("get_weather", {}, "Sunny, high 76.", {"read"}),
    ("music", {"operation": "my_playlists"}, "His library is his: only he can ask me what's in it.", set()),
    ("read_his_email", {}, "Couldn't read his email: only the owner can ask about his Gmail.", set()),
])
def test_completion_needs_the_tools_own_words_for_success(name, args, content, kinds):
    from aerys_v2.service import _one_loop_done_kinds

    assert set(_one_loop_done_kinds(name, args, content, None)) == kinds


def test_a_real_music_refusal_does_not_back_a_saved_playlist():
    action = Action(tool_then("music", "I've saved it to your playlist.", "Only he can change his music library.",
                              args={"operation": "add_to_playlist"}), "I couldn't save it: only you can change it.")
    run("save it to focus", action=action)
    assert len(action.calls) == 2


# ---- Codex second re-check #2: recorded is not delivered ---------------------------------------

QUEUED_TICKET = ("Added to T-3, but Kael's line didn't take it right now; it's retried automatically. "
                 "Don't send it again.")


def test_a_queued_ticket_update_backs_noted_but_not_sent():
    noted = Action(tool_then("kael_tickets", "I've noted it on your ticket for Kael.", QUEUED_TICKET,
                             args={"action": "update", "ticket": "T-3", "message": "x"}))
    run("add that for kael", action=noted)
    assert len(noted.calls) == 1 and noted.reply_calls == []
    sent = Action(tool_then("kael_tickets", "I've sent it to Kael.", QUEUED_TICKET,
                            args={"action": "update", "ticket": "T-3", "message": "x"}),
                  words=["It's on the ticket; Kael's line will take it shortly."])
    run("add that for kael", action=sent)
    assert len(sent.reply_calls) == 1                         # recorded (an act), so words only


# ---- Codex second re-check #3: a claimed look needs a completed READ of the right kind -------

def test_a_send_does_not_back_a_claimed_weather_check():
    action = Action(tool_then("send_message", "I checked the weather: sunny.",
                              "Sent: the signed message went through Discord."),
                    words=["I sent the message; I haven't checked the weather."])
    run("check the weather and tell sam", action=action)
    assert len(action.reply_calls) == 1


def test_a_look_must_be_the_right_kind_of_look():
    wrong = Action(tool_then("get_weather", "I checked your calendar: two meetings.", "Sunny."), "Let me look.")
    run("what's on my calendar", action=wrong)
    assert len(wrong.calls) == 2
    right = Action(tool_then("calendar_events", "I checked your calendar: two meetings.", "9:00 standup; 10:00 1:1"))
    run("what's on my calendar", action=right)
    assert len(right.calls) == 1
    web = Action(tool_then("search_web", "I checked the forecast: rain after 3.", "rain after 3pm"))
    run("will it rain", action=web)
    assert len(web.calls) == 1


# ---- Replay point A: the real file tools back what they did (turn 3069's shape) -------------------

def test_a_written_file_sent_to_his_dm_is_backed():
    wrote = [AIMessage(content="", tool_calls=[{"id": "w", "name": "write_file",
                                                "args": {"file_path": "/notes/howto.md", "content": "x"}}]),
             ToolMessage(content="Updated file /notes/howto.md", name="write_file", tool_call_id="w")]
    sent = tool_then("send_file_to_chris", "I wrote the how-to to a file and sent it to your DM.",
                     "SENT: howto.md (812 bytes) is in your Discord DM.", args={"path": "notes/howto.md"})
    action = Action([*wrote, *sent])
    run("put that in a file for me", action=action)
    assert len(action.calls) == 1 and action.reply_calls == []


# ---- Replay point B: answers about an EARLIER turn aren't this turn's claims ---------------------
# Paraphrased shapes of turns 3426 and 3376; chosen rule: _one_loop_present (a sentence reporting her notes,
# or an answer to a message that ONLY asks about the past when she attempted no act). The seed carries text
# only, so earlier turns' receipts can't back them.

def test_his_question_about_an_earlier_send_is_answered_without_a_bounce():
    # He asks how she got something to Kael; this turn she only looks at her tickets.
    action = Action(tool_then("kael_tickets",
                              "I sent the question to Kael with my direct line, and it opened ticket T-9. "
                              "I just checked, and none of my tickets are open now.",
                              "You haven't asked Kael anything yet.", args={"action": "list"}))
    run("how did you get that question to kael?", action=action)
    assert len(action.calls) == 1 and action.reply_calls == []


def test_what_her_notes_say_she_did_before_is_not_a_claim_about_now():
    # He tells her how a game works; she recalls from her notes what she did on an earlier day.
    action = Action("You're right, it's drag to spin. My notes say I've played it twice that way.")
    run("you have to drag the globe to turn it", action=action)
    assert len(action.calls) == 1 and action.reply_calls == []


def test_a_claim_tied_to_now_is_still_checked_when_he_asks_about_the_past():
    action = Action("Yes, I just sent it to Sam.", "I haven't sent it yet.")
    run("did you send it to sam?", action=action)
    assert len(action.calls) == 2


def test_a_plain_claim_with_no_earlier_time_is_still_checked():
    action = Action("I've turned off the lamp.", "I haven't touched the lamp.")
    run("the lamp is still on", action=action)
    assert len(action.calls) == 2


def test_a_promise_is_checked_even_when_he_asks_about_the_past():
    action = Action("I'll send it to Sam right now.", "I haven't sent it.")
    run("did you send it to sam?", action=action)
    assert len(action.calls) == 2


# ---- Advisor (pre-report): what may have acted vs what backs a claim ------------------------------

def test_an_unconfirmed_act_never_runs_his_turn_armed_again():
    # An unknown tool, or an answer the table doesn't recognise, backs nothing but may have done something:
    # the correction is words only, so nothing can happen twice.
    for name, content, args in (("some_new_tool", "ok", {}),
                                ("home_control", "Sent turn_on to Lamp; Home Assistant accepted it (200) but "
                                                 "light.lamp reported NO state change.", {"operation": "turn_on"})):
        action = Action(tool_then(name, "I've turned on the lamp.", content, args=args),
                        words=["I asked for it; I can't see that it changed."])
        run("lamp on", action=action)
        assert len(action.calls) == 1 and len(action.reply_calls) == 1, name


def test_no_action_never_backs_a_look():
    action = Action(tool_then("no_action", "I looked into it: nothing's wrong.", "NO_ACTION: say this"),
                    "I haven't looked yet.")
    run("is anything wrong?", action=action)
    assert len(action.calls) == 2


@pytest.mark.parametrize("name,content", [
    ("get_weather", "Rain chance: not available right now."),
    ("calendar_events", "The calendar is unreachable right now (couldn't read: calendar.x)."),
    ("search_web", "web search failed: the search service is unreachable right now (timeout)."),
    ("search_web", "search_web needs a search query — there is nothing to look up."),
    ("read_web_page", "Reading the page failed: the reading service is unreachable right now."),
    ("read_email", "read_email needs the uid of a message — call search_email first."),
    ("search_email", "email search failed: timeout."),
    ("check_async_task", "Not done: I couldn't get to his home jobs just now, so nothing happened."),
    ("read_my_code", "Not available: only Chris's turns and my own wake-ups can read my code."),
    ("where_is_chris_nearby", "Location request refused: I can only tell him where he is."),
    ("analyze_image", "The vision service is unreachable right now (timeout)."),
])
def test_a_read_that_failed_is_not_a_look(name, content):
    from aerys_v2.service import _one_loop_done_kinds

    assert not _one_loop_done_kinds(name, {}, content, None)


def test_a_web_search_backs_a_look_of_any_kind():
    for claim in ("I looked up the train schedule: the 5:10 runs late.", "I looked up that song: it's from 1994."):
        action = Action(tool_then("search_web", claim, "results: …"))
        run("look it up", action=action)
        assert len(action.calls) == 1 and action.reply_calls == [], claim


@pytest.mark.parametrize("reply", [
    "Logged: you ran 5k yesterday.",                 # a status line is this turn's, whatever day it names
    "I've set the check-back for Monday.",           # a day that is the content, not when she acted
    "I saved that you weighed 182 this morning.",
    "I've saved that you ran 5k yesterday.",
    "I've added that to my notes.",                  # her notes as where a write went, not a report of them
])
def test_a_time_that_is_the_content_does_not_excuse_a_claim(reply):
    action = Action(reply, "I haven't done that yet.")
    run("note it", action=action)
    assert len(action.calls) == 2, reply


def test_the_home_job_and_file_tools_answer_in_the_words_the_checks_expect():
    # The installed deepagents writes these; a version whose wording moved would leave every start unconfirmed.
    need("deepagents")
    import inspect

    import deepagents.middleware.async_subagents as async_subagents
    import deepagents.middleware.filesystem as filesystem
    from aerys_v2.service import _one_loop_done_kinds

    source = inspect.getsource(async_subagents) + inspect.getsource(filesystem)
    for words, name, kinds in (("Launched async subagent. task_id:", "start_async_task", {"start"}),
                               ("Updated async subagent. task_id:", "update_async_task", {"start"}),
                               ("Queued follow-up to async subagent", "continue_home_job", {"start"}),
                               ("Cancelled async subagent task:", "cancel_async_task", {"stop"}),
                               ("Updated file ", "write_file", {"record"}),
                               ("Successfully replaced ", "edit_file", {"record"})):
        assert words in source, words
        assert set(_one_loop_done_kinds(name, {}, words + " x", None)) == kinds, name


def test_a_request_beside_his_past_question_is_still_checked():
    # The 10:55 shape: he asks whether she got it AND tells her to go ahead; she claims a start she never made.
    action = Action("I've started a home job to update the skill.", "I haven't started it yet.")
    run("did you get that? go ahead and start it", action=action)
    assert len(action.calls) == 2


def test_an_act_tried_this_turn_ends_the_past_question_pass():
    # He asks "did you send it?"; this turn's send is refused; "Yes, I sent it" is about THIS turn now.
    action = Action(tool_then("send_message", "Yes, I sent it to Sam.",
                              "Not sent: that person isn't on the list of people I can message; ask the owner."),
                    "No: Sam isn't on the list of people I can message.")
    run("did you send it to sam?", action=action)
    assert len(action.calls) == 2                   # checked: refused, nothing was done, so his turn runs again


def test_a_draft_that_timed_out_is_never_tried_again_armed():
    # "check Gmail before trying again": it may have landed, so the correction is words only.
    action = Action(tool_then("draft_in_his_gmail", "I've drafted it in your Gmail.",
                              "Not drafted: create timed out; check Gmail before trying again."),
                    words=["The draft timed out; check Gmail before I try again."])
    run("draft a reply to sam", action=action)
    assert len(action.calls) == 1 and len(action.reply_calls) == 1
