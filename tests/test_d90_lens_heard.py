"""D90 (the owner, 11/22 13:34): a LENIENT "was that for me?" on his glasses' follow-ups. "I just dont want it
to be over strict is all." Quiet only when Jev is very sure the words weren't said to her; a first turn, a DM,
a doubt, a timeout or an error all answer."""

from __future__ import annotations

import time
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage

import aerys_v2.factory as factory
from aerys_v2.addressed_check import QUESTION_ID, SKIP_BELOW, AddressedCheck, build_state, skip
from aerys_v2.service import LENS_HEARD_SKIP_MARKER, ask
from test_d79_text_one_loop import ALLOW, OWNER, Action, Recorder, dm, fake_model, route
from aerys_v2.factory import build_graph


@pytest.fixture(autouse=True)
def owner(monkeypatch):
    monkeypatch.setattr(factory, "_OWNER_ID", OWNER)


def lens(user_id=OWNER):
    return {"user_id": user_id, "display_name": "Chris (Glasses)", "privacy_context": "private", "surface": "lens"}


class Check:
    """A stand-in for AddressedCheck: answers p (or raises), and records what it was shown."""

    def __init__(self, p):
        self.p, self.seen = p, []

    def __call__(self, heard, she_said, he_said_before):
        self.seen.append((heard, she_said, he_said_before))
        if isinstance(self.p, BaseException):
            raise self.p
        return self.p


def go(text, *, p, identity=None, in_flight=True, drop=True, graph=None, thread="person:o", record=None):
    graph = graph or build_graph(fake_model("Sure, the museum closes at five."), soul="s")
    graph.update_state({"configurable": {"thread_id": thread}},
                       {"messages": [HumanMessage(content="Tell me about the museum."),
                                     AIMessage(content="It opened in 1925. Did you two go inside?")]}, as_node="chat")
    action = Action("Sure, the museum closes at five.")
    action.addressed_check = Check(p)
    registry = {thread: time.monotonic()} if in_flight else {}
    out = ask(graph, text, identity=identity or lens(), thread_id=thread, router=route(), action_graph=action,
              action_allowlist=ALLOW, record_turn=record, text_one_loop=True, drop_unaddressed=drop,
              activity_registry=registry)
    return out, action


def test_background_speech_mid_conversation_is_skipped_quietly():
    rec = Recorder()
    out, action = go("Babe, grab the tickets from the car.", p=0.05, record=rec)
    assert out == "" and action.calls == []
    heard, she_said, before = action.addressed_check.seen[0]
    assert heard.startswith("Babe") and she_said.startswith("It opened") and before == "Tell me about the museum."
    row = rec.wait(1)[0]
    assert LENS_HEARD_SKIP_MARKER in (row.get("degraded") or []) and row["classifier_intent"] == "unaddressed"


@pytest.mark.parametrize("p", [SKIP_BELOW, 0.4, 0.9, None, RuntimeError("jev down")])
def test_anything_short_of_very_sure_is_answered(p):
    out, action = go("We didn't go in.", p=p)
    assert out == "Sure, the museum closes at five." and len(action.calls) == 1


def test_a_first_turn_is_never_checked():
    out, action = go("What time does it close?", p=0.01, in_flight=False)
    assert out == "Sure, the museum closes at five." and action.addressed_check.seen == []


def test_only_the_lens_and_only_with_the_drop_on():
    out, action = go("hi", p=0.01, identity=dm())
    assert action.addressed_check.seen == [] and out
    out, action = go("hi", p=0.01, drop=False)
    assert action.addressed_check.seen == [] and out


# ---- the check itself -----------------------------------------------------------------------------------

class FakeTS:
    def __init__(self, value=None, delay=0.0, raises=None):
        self.value, self.delay, self.raises, self.asked = value, delay, raises, []

    def system_one(self, *, state, questions, model):
        self.asked.append((state, questions, model))
        time.sleep(self.delay)
        if self.raises:
            raise self.raises
        return SimpleNamespace(answers={QUESTION_ID: SimpleNamespace(noul=self.value)})


def test_skip_is_lenient_and_never_on_doubt():
    assert skip(0.05) and skip(SKIP_BELOW - 0.001)
    assert not skip(SKIP_BELOW) and not skip(0.5) and not skip(None)
    assert SKIP_BELOW <= 0.2                                 # "not over strict"


def test_the_check_returns_a_real_score_or_nothing():
    assert AddressedCheck(FakeTS(0.3))("x", "y", "z") == 0.3
    for bad in (float("nan"), 1.5, -0.1, True, "0.2", None):
        assert AddressedCheck(FakeTS(bad))("x", "y", "z") is None
    assert AddressedCheck(FakeTS(raises=RuntimeError("down")))("x", "y", "z") is None
    assert AddressedCheck(FakeTS(0.01, delay=0.3), timeout_s=0.1)("x", "y", "z") is None   # late: never used


def test_the_state_is_bounded_and_keeps_the_end_of_her_reply():
    st = build_state("h" * 1000, "start " + "x" * 2000 + " did you go inside?", "b" * 1000)
    assert len(st["she_said"]) <= 601 and st["she_said"].endswith("did you go inside?")
    assert len(st["heard"]) <= 401 and len(st["he_said_before"]) <= 401
    assert set(st) == {"heard", "she_said", "he_said_before"}


# ---- Codex review of a533f1e ----------------------------------------------------------------------------

from aerys_v2.router import RouteDecision                                     # noqa: E402
from aerys_v2.service import CANCEL_ACK, HEARD_BUDGET_S, _said_to_her         # noqa: E402
from test_d79_text_one_loop import GUEST, WIFE                                # noqa: E402


@pytest.mark.parametrize("who", [WIFE, GUEST])
def test_only_his_own_glasses_are_checked(who):
    out, action = go("Babe, grab the tickets.", p=0.01, identity=lens(who))
    assert action.addressed_check.seen == []                                   # never silenced by the check


def test_no_owner_configured_means_no_check(monkeypatch):
    monkeypatch.setattr(factory, "_OWNER_ID", None)
    out, action = go("Babe, grab the tickets.", p=0.01)
    assert action.addressed_check.seen == [] and out


def test_a_turn_still_pending_is_never_paired_with_an_older_reply():
    graph = build_graph(fake_model("Sure."), soul="s")
    graph.update_state({"configurable": {"thread_id": "person:o"}},
                       {"messages": [HumanMessage(content="Tell me about it."), AIMessage(content="It opened in 1925."),
                                     HumanMessage(content="and the house next door?")]}, as_node="chat")
    check = Check(0.01)
    assert _said_to_her(graph, check, "person:o", "Babe, grab the tickets.") is None and check.seen == []


def test_a_slow_history_read_answers_the_turn_inside_the_budget():
    class SlowGraph:
        def get_state(self, _config):
            time.sleep(HEARD_BUDGET_S + 1.0)

    started = time.monotonic()
    assert _said_to_her(SlowGraph(), Check(0.01), "person:o", "x") is None
    assert time.monotonic() - started < HEARD_BUDGET_S + 0.5


def test_saturation_answers_the_turn():
    from aerys_v2 import service

    taken = 0
    while service._HEARD_SLOTS.acquire(blocking=False):
        taken += 1
    try:
        graph = build_graph(fake_model("x"), soul="s")
        assert _said_to_her(graph, Check(0.01), "person:o", "x") is None
    finally:
        for _ in range(taken):
            service._HEARD_SLOTS.release()
    chk = AddressedCheck(FakeTS(0.01))
    held = 0
    while chk._inflight.acquire(blocking=False):
        held += 1
    assert chk("x", "y", "z") is None


def test_a_cancel_on_the_glasses_is_handled_before_the_check():
    graph = build_graph(fake_model("unused"), soul="s")
    graph.update_state({"configurable": {"thread_id": "person:o"}},
                       {"messages": [HumanMessage(content="Tell me about it."), AIMessage(content="It opened in 1925.")]},
                       as_node="chat")
    action = Action("unused")
    action.addressed_check = Check(0.01)
    out = ask(graph, "never mind", identity=lens(), thread_id="person:o",
              router=lambda _t: RouteDecision(route="chat", ack="", cancelled=True), action_graph=action,
              action_allowlist=ALLOW, text_one_loop=True, drop_unaddressed=True,
              activity_registry={"person:o": time.monotonic()})
    assert out == CANCEL_ACK and action.addressed_check.seen == []


def test_a_skipped_snippet_leaves_the_thread_untouched():
    graph = build_graph(fake_model("unused"), soul="s")
    before_len = None
    out, action = go("Babe, grab the tickets.", p=0.05, graph=graph)
    after = graph.get_state({"configurable": {"thread_id": "person:o"}}).values["messages"]
    assert out == "" and [m.content for m in after] == ["Tell me about the museum.", "It opened in 1925. Did you two go inside?"]
