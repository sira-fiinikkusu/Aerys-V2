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
                    tool_then("start_async_task", "Started the home job; it'll read the skill back.", "started j1"))
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
    action = Action(tool_then("send_message", "I've started a home job and told Sam.", "sent to Sam"),
                    words=["I sent Sam the note; I haven't started a home job."])
    out = run("tell sam I'm running late", action=action)
    assert len(action.calls) == 1, "the armed graph ran again after a send"
    assert len(action.reply_calls) == 1
    assert out == "I sent Sam the note; I haven't started a home job."


def test_two_send_bearing_passes_are_impossible():
    # Whatever the first pass says after sending, the second chance has no tools.
    for words in ("I've started a home job.", "I heard the chime on the speakers.", "Logged: done."):
        action = Action(tool_then("send_message", words, "sent"), words=["Sent."])
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
    pytest.importorskip("langchain.agents")
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
    action = Action(tool_then("send_message", "I have started a home job for it.", "sent"),
                    words=["I messaged him; the home job isn't started."])
    run("start it and tell him", action=action)
    assert len(action.reply_calls) == 1


def test_a_kael_tickets_update_is_a_write_and_a_check_is_a_read():
    wrote = Action(tool_then("kael_tickets", "Logged: I added that to your ticket for Kael.", "updated T-3",
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
    action = Action(tool_then("send_message", "I heard the chime on the speakers.", "sent"),
                    words=[RuntimeError("boom")])
    out = run("tell sam hi", graph=graph, action=action, record=rec)
    row = rec.wait(1)[0]
    assert calls_of(row) == ["send_message"]
    assert "send_message" in out or "sent" in out.lower()
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
    pytest.importorskip("langchain.agents")
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
    pytest.importorskip("langchain.agents")
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
