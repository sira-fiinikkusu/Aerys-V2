"""11/20 10:55 (turn 3476): the old chat path let a made-up start through.

Chris typed "you are good to update your skill". The router sent it to the tool-less
chat side, and she answered "I’ve started a home job to update the MapTap skill" with
nothing running. The claim gate missed on both sides:
- his words are permission ("you are good to..."), not an order or a question;
- her "I’ve" used a curly apostrophe, and "started" was not one of the claim verbs.
With both fixed, the gate sends the turn to the side with tools, which really starts
it or says honestly that it can't.
"""

from __future__ import annotations

import threading
import time

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, ToolMessage

from aerys_v2.factory import build_graph
from aerys_v2.router import RouteDecision
from aerys_v2.service import (
    CLAIM_GATE_MARKER,
    _claims_effect_without_doing_it,
    _promises_effect_without_doing_it,
    ask,
    claims_effect_done,
)

CHRIS = {"user_id": "person-1", "display_name": "Chris", "privacy_context": "private",
         "platform": "discord", "channel_kind": "dm"}
SAID = "I’ve started a home job to update the MapTap skill, Chris. It’s not saved yet."


def fake_model(*replies):
    return GenericFakeChatModel(messages=iter([AIMessage(content=r) for r in replies]))


class Action:
    def __init__(self):
        self.calls = []

    def invoke(self, inp, config):
        self.calls.append(list(inp["messages"]))
        return {"messages": [*inp["messages"],
                             AIMessage(content="", tool_calls=[{"id": "c", "name": "start_async_task", "args": {}}]),
                             ToolMessage(content="started job j1", name="start_async_task", tool_call_id="c"),
                             AIMessage(content="Started the home job; it will read the skill back.")]}


class Recorder:
    def __init__(self):
        self.rows, self._lock = [], threading.Lock()

    def __call__(self, row):
        with self._lock:
            self.rows.append(row)

    def wait(self, n, timeout=3.0):
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            with self._lock:
                if len(self.rows) >= n:
                    return list(self.rows)
            time.sleep(0.01)
        raise AssertionError(f"{len(self.rows)} rows")


def test_the_1055_turn_escalates_to_the_side_with_tools():
    graph = build_graph(fake_model(SAID), soul="s")
    action, rec = Action(), Recorder()
    out = ask(graph, "you are good to update your skill", identity=CHRIS, thread_id="person:1",
              router=lambda _t: RouteDecision(route="chat", ack=""), action_graph=action, record_turn=rec)
    assert action.calls, "the claim gate let the made-up start through"
    assert out == "Started the home job; it will read the skill back."
    assert any(CLAIM_GATE_MARKER in (r.get("degraded") or []) for r in rec.wait(2))
    msgs = graph.get_state({"configurable": {"thread_id": "person:1"}}).values["messages"]
    assert [m.content for m in msgs] == ["you are good to update your skill", out]   # the claim never stays


def test_her_wording_counts_as_a_claim():
    assert claims_effect_done(SAID)
    assert claims_effect_done("I've kicked off the run.")
    assert claims_effect_done("I have launched today's MapTap.")
    assert not claims_effect_done("I haven't started a home job yet.")
    assert not claims_effect_done("I started thinking about it.")          # no started THING


def test_his_permission_counts_as_asking():
    for his in ("you are good to update your skill", "you're good to go ahead", "feel free to start it",
                "you have my go", "You’re fine to run it"):
        assert _claims_effect_without_doing_it(his, SAID), his
    assert _promises_effect_without_doing_it("you're good to update your skill", "I'll start it now.")


def test_talk_still_stays_talk():
    assert not _claims_effect_without_doing_it("remember when you started that job?", "Yes, I've started one before.")
    assert not _claims_effect_without_doing_it("how was your day", SAID)
