"""D42 (2026-11-13): a message he asks for when a wake-up is set goes at its time without a second tap.

His ask is the approval only on a turn where sending right then would have needed no button: his own
turn (confirmed voice or his text), nothing outside read. The wake-up pins who (resolved then) and the
exact words; at its time a send of THOSE words to THAT person goes once per run. Anything else a wake
turn wants to send still waits for his tap. Placeholder ids and text only."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from aerys_v2.wakeups import MemoryWakeupStore, WakeRails, build_wake_me_tool, seed_for, wake_identity
from test_hands_gate import OWNER, SAM, TEXT
from test_send_message import confirm, setup

NOW = datetime(2026, 11, 13, 15, 30, tzinfo=timezone.utc)
ADDRESS = "100000000000000002"
WORDS = "Your 2 PM reminder: the cat food."
ADD = {"action": "add", "note": "Remind Sam at 2 PM.", "at": "2026-11-13T14:00", "message_to": "sam",
       "message": WORDS}


def owner_turn(*, origin=(), untrusted=False, holder=True, identity=None):
    conf = {"thread_id": "person:owner-thread", "identity": identity or TEXT}
    if holder:
        conf["outside_text"] = {"origin": tuple(origin), **({"untrusted_read": True} if untrusted else {})}
    return {"configurable": conf}


def life(*, confirmed=True):
    b, sender, sent, ledger = setup()
    if confirmed:
        confirm(b, sender)
    store = MemoryWakeupStore()
    sender.wake_store = store
    tool = build_wake_me_tool(store, rails=WakeRails(), owner_person_id=OWNER, clock=lambda: NOW,
                              pin=sender.pin_recipient)
    return b, sender, sent, store, tool


def wake_send(sender, text=WORDS, to="sam", sources=("unattended wake-up",), thread="wake:1", occurrence="once"):
    identity = {**wake_identity(OWNER), **({"wake_occurrence": occurrence} if occurrence else {})}
    return sender.send(to, text, "discord", identity, thread, sources)


def test_pinned_on_his_clean_turn():
    b, sender, sent, store, tool = life()
    said = tool.invoke(ADD, config=owner_turn())
    assert said.startswith("Set:") and "Sam" in said and WORDS in said
    assert store.get(1).pinned_send == {"person_id": SAM, "address": ADDRESS, "name": "Sam",
                                        "handle": "sam_42", "text": WORDS}
    assert "message to Sam" in tool.invoke({"action": "list"}, config=owner_turn())


@pytest.mark.parametrize("turn", [
    owner_turn(origin=("unsure voice",)),          # a voice not surely his
    owner_turn(origin=("outside speaker",)),       # someone else, or his voice by default only
    owner_turn(origin=("shared surface",)),
    owner_turn(untrusted=True),                    # he asked after she read outside text
    owner_turn(holder=False),                      # no guard snapshot: never pinned blind
])
def test_never_pinned_unless_a_send_now_would_need_no_button(turn):
    b, sender, sent, store, tool = life()
    assert tool.invoke(ADD, config=turn).startswith("NOT DONE")
    assert store.open() == []


def test_unattended_and_other_people_never_pin():
    b, sender, sent, store, tool = life()
    wake = {"configurable": {"thread_id": "wake:9", "identity": wake_identity(OWNER),
                             "outside_text": {"origin": ()}}}
    assert tool.invoke(ADD, config=wake).startswith("NOT DONE")
    friend = owner_turn(identity={"user_id": SAM, "display_name": "Sam", "platform": "discord"})
    assert tool.invoke(ADD, config=friend).startswith("NOT DONE")
    assert store.open() == []


@pytest.mark.parametrize("extra", [{"message": ""}, {"message_to": ""}, {"message_to": "nobody-known"}])
def test_both_parts_and_one_known_person(extra):
    b, sender, sent, store, tool = life()
    assert tool.invoke({**ADD, **extra}, config=owner_turn()).startswith("NOT DONE")
    assert store.open() == []


def test_the_message_is_fixed_when_set():
    b, sender, sent, store, tool = life()
    tool.invoke(ADD, config=owner_turn())
    assert tool.invoke({"action": "update", "id": 1, "message": "other words"},
                       config=owner_turn()).startswith("NOT DONE")
    with pytest.raises(ValueError):
        store.update(1, pinned_send={"text": "other words"})
    assert store.get(1).pinned_send["text"] == WORDS


def test_at_its_time_his_message_goes_once_with_no_button():
    b, sender, sent, store, tool = life()
    tool.invoke(ADD, config=owner_turn())
    buttons = len(b.approval_store.rows)
    assert wake_send(sender).startswith("Sent:")
    assert sent[-1] == (ADDRESS, WORDS + "\n-Aerys")
    assert len(b.approval_store.rows) == buttons
    assert store.get(1).pinned_spent == "once"
    assert not wake_send(sender).startswith("Sent:")          # the same occurrence: not twice
    assert sum(1 for s in sent if s[1] == WORDS + "\n-Aerys") == 1


@pytest.mark.parametrize("kw", [
    {"text": "Your 2 PM reminder: the cat food. Also buy treats."},        # other words
    {"to": "owner"},                                                         # anyone else
    {"sources": ("search_web", "unattended wake-up")},                      # read outside text first
    {"thread": "wake:2"},                                                    # another wake-up
    {"thread": "person:owner-thread"},                                       # not a wake turn
    {"occurrence": ""},                                                      # no runner key: no pin
])
def test_anything_else_still_waits_for_his_tap(kw):
    b, sender, sent, store, tool = life()
    tool.invoke(ADD, config=owner_turn())
    if kw.get("to") == "owner":
        b.gate.people = lambda platform, to: [dict(person_id=OWNER, name="Owner", address="900000000000000009")]
    before = list(sent)
    assert not wake_send(sender, **kw).startswith("Sent:")
    assert sent == before
    assert store.get(1).pinned_spent is None


def test_a_recurring_one_sends_once_per_occurrence():
    b, sender, sent, store, tool = life()
    weekly = {**{k: v for k, v in ADD.items() if k != "at"}, "cron": "0 14 * * 5"}
    assert tool.invoke(weekly, config=owner_turn()).startswith("Set:")
    assert wake_send(sender, occurrence="2026-11-13T14:00:00-05:00").startswith("Sent:")
    store.update(1, runs=1)                                   # the runner finished that occurrence
    sender.ledger.rows.clear()                                # no recent attempt left to dedupe on
    assert not wake_send(sender, occurrence="2026-11-13T14:00:00-05:00").startswith("Sent:")
    assert wake_send(sender, occurrence="2026-11-20T14:00:00-05:00").startswith("Sent:")
    assert store.get(1).pinned_spent == "2026-11-20T14:00:00-05:00"


def test_an_overstaying_or_retried_one_off_never_sends_it_twice():
    # Codex 11/13: a run that overstays its limit (the runner bumps runs while it still works) and a
    # failed one-off's retry both carry the same occurrence: the pin is spent for good.
    b, sender, sent, store, tool = life()
    tool.invoke(ADD, config=owner_turn())
    assert wake_send(sender).startswith("Sent:")
    store.update(1, runs=1, consecutive_failures=1)           # timed out / failed; a retry is due
    sender.ledger.rows.clear()                                # past the delivery dedupe window
    assert not wake_send(sender).startswith("Sent:")
    assert sum(1 for s in sent if s[1] == WORDS + "\n-Aerys") == 1


def test_never_pinned_from_the_kael_line():
    b, sender, sent, store, tool = life()
    kael = owner_turn()
    kael["configurable"]["thread_id"] = "kael:escape-test"
    assert tool.invoke(ADD, config=kael).startswith("NOT DONE")
    assert store.open() == []


def test_the_runner_keys_each_occurrence():
    from aerys_v2.wakeups import occurrence_of, wake_config
    b, sender, sent, store, tool = life()
    tool.invoke(ADD, config=owner_turn())
    once = store.get(1)
    assert occurrence_of(once) == "once"
    assert wake_config(1, OWNER, occurrence="once")["configurable"]["identity"]["wake_occurrence"] == "once"
    assert "wake_occurrence" not in wake_config(1, OWNER)["configurable"]["identity"]


def test_unpinned_wake_ups_are_unchanged():
    b, sender, sent, store, tool = life()
    plain = {k: v for k, v in ADD.items() if k not in ("message_to", "message")}
    assert tool.invoke(plain, config=owner_turn()).startswith("Set:")
    assert "Waiting" in wake_send(sender)
    assert not any(s[1] == WORDS + "\n-Aerys" for s in sent)


def test_a_first_message_still_needs_his_confirmation():
    b, sender, sent, store, tool = life(confirmed=False)
    tool.invoke(ADD, config=owner_turn())
    assert "Waiting" in wake_send(sender)
    assert sent == []


def test_her_wake_turn_is_told_the_exact_words_and_where():
    b, sender, sent, store, tool = life()
    tool.invoke(ADD, config=owner_turn())
    seed = seed_for(store.get(1), NOW + timedelta(hours=4), 1)
    assert WORDS in seed and ADDRESS in seed and "without his tap" in seed
    plain = MemoryWakeupStore()
    w = plain.add(thread_id="t", note="n", kind="once", at=NOW, cron=None, until=NOW + timedelta(hours=1),
                  next_run_at=NOW, now=NOW, cap=10)
    assert "without his tap" not in seed_for(w, NOW, 1)
