"""What waits for his tap (board #176, Stage 4). D10 (11/08): "Voice can start anything text can but
requires me to approve via discord button. Else read only jobs". D12 (11/09): the first message to
anyone new waits for his button, which shows exactly who and where; no match or several: she asks.
Fail closed: nothing is done without his Approve, and nothing he approved is done twice."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from aerys_v2.approvals import Approvals, MemoryApprovalStore
from aerys_v2.hands_gate import (
    HOME_JOB,
    MESSAGE,
    OTHER,
    OWNER_TEXT,
    OWNER_VOICE,
    VOICE_UNSURE,
    HandsGate,
    HandsWatcher,
    MemoryGatedActionStore,
    asker,
    hands_gate_for,
    home_job_middleware,
)
from aerys_v2.state import VOICE_OWNER_ASSUMED_KEY

import aerys_v2.hands_gate as hands_gate

OWNER_ASK = getattr(hands_gate, "OWNER_ASK", "owner_ask")

NOW = datetime(2026, 11, 9, 15, 0, tzinfo=timezone.utc)
OWNER = "6e6bcbed-0000-4000-8000-000000000001"
SAM = "ef6df13a-0000-4000-8000-000000000002"
OWNER_THREAD = f"person:{OWNER}"

TEXT = {"user_id": OWNER, "display_name": "Chris", "platform": "discord", "channel_kind": "dm"}
STRONG = {"user_id": OWNER, "display_name": "Chris", "voice": True,
          "speaker": {"id": "chris", "confidence": 0.82, "demoted": False}}
UNTAGGED = {"user_id": OWNER, "display_name": "Chris", "voice": True, VOICE_OWNER_ASSUMED_KEY: True}
UNSURE = {"user_id": OWNER, "display_name": "Chris", "voice": True, VOICE_OWNER_ASSUMED_KEY: True,
          "speaker": {"id": "unknown", "confidence": 0.3, "demoted": False}}
GUEST_VOICE = {"user_id": "guest-voice", "display_name": "Guest", "voice": True,
               "speaker": {"id": "unknown", "confidence": 0.1, "demoted": True}}
KAEL_ASK = {"user_id": OWNER, "display_name": "Kael", "privacy_context": "private"}
LENS_ASK = {"user_id": OWNER, "display_name": "Chris", "privacy_context": "private", "surface": "lens"}
PANEL_ASK = {"user_id": OWNER, "display_name": "Chris", "privacy_context": "private", "device_id": "panel-1"}
STRANGER_ASK = {"user_id": "someone-else", "display_name": "Someone", "privacy_context": "private"}
KAEL_THREAD = "kael:escape-test"
SAM_TEXT = {"user_id": SAM, "display_name": "Sam", "platform": "discord", "channel_kind": "guild"}

START_WRITE = {"description": "Build the fridge planner page", "subagent_type": "home"}
START_READ = {"description": "Read the planner notes and summarize them", "subagent_type": "home-readonly"}

PEOPLE = {
    ("discord", "sam_42"): [{"person_id": SAM, "name": "Sam", "address": "100000000000000002", "handle": "sam_42"}],
    ("discord", "sam"): [{"person_id": SAM, "name": "Sam", "address": "100000000000000002", "handle": "sam_42"}],
    ("discord", "chris_h"): [{"person_id": OWNER, "name": "Chris", "address": "100000000000000001", "handle": "chris_h"}],
    ("email", "sam@example.net"): [{"person_id": SAM, "name": "Sam", "address": "sam@example.net",
                                    "handle": "sam@example.net"}],
    ("discord", "chris"): [{"person_id": OWNER, "name": "Chris", "address": "1", "handle": "chris_h"},
                           {"person_id": "orphan-chris", "name": "Chris", "address": "2", "handle": "chris2"}],
}


class Box:
    """A gate over in-memory stores, with a clock and executors that record what they did."""

    def __init__(self, *, executors=None, read_only=("home-readonly",), people=None):
        self.clock = {"now": NOW}
        self.approval_store = MemoryApprovalStore()
        self.pinged = []
        self.approvals = Approvals(self.approval_store, notify=lambda t, m: self.pinged.append((t, m)),
                                   clock=lambda: self.clock["now"], sleep=lambda s: None)
        self.store = MemoryGatedActionStore()
        self.done = []
        default = {HOME_JOB: lambda row: self.done.append(("job", row.args)) or (True, "task_id: t-1"),
                   MESSAGE: lambda row: self.done.append(("msg", row.args)) or (True, "sent")}
        lookup = people if people is not None else (lambda platform, to: PEOPLE.get((platform, to.lower()), []))
        self.gate = HandsGate(self.approvals, self.store, owner_person_id=OWNER, people=lookup,
                              executors=default if executors is None else executors,
                              read_only_agents=read_only, clock=lambda: self.clock["now"])
        self.notes = []
        self.watcher = HandsWatcher(self.gate, note=lambda thread, text: self.notes.append((thread, text)))

    def tap(self, approve=True, approval_id=None):
        approval_id = approval_id or max(self.approval_store.rows)
        return self.approval_store.decide(approval_id, approve, 1, self.clock["now"])


# --- who is asking --------------------------------------------------------------------------


@pytest.mark.parametrize("identity,expected", [
    (TEXT, OWNER_TEXT),
    ({**TEXT, "platform": "telegram"}, OWNER_TEXT),
    (STRONG, OWNER_VOICE),
    (UNTAGGED, VOICE_UNSURE),
    (UNSURE, VOICE_UNSURE),
    ({**STRONG, "speaker": {"id": "chris", "confidence": 0.5, "demoted": False}, VOICE_OWNER_ASSUMED_KEY: True},
     VOICE_UNSURE),
    ({"user_id": OWNER, "voice": True, "speaker": {"id": "unknown", "confidence": 0.5}}, VOICE_UNSURE),
    ({"user_id": OWNER, "voice": True}, VOICE_UNSURE),          # the HA chat shim: no speaker verdict at all
    (GUEST_VOICE, OTHER),
    (KAEL_ASK, OWNER_ASK),                 # /ask carries his id: his own line, but not proof he typed it (D17 Q4)
    (LENS_ASK, OWNER_ASK),
    (PANEL_ASK, OWNER_ASK),
    (STRANGER_ASK, OTHER),
    (SAM_TEXT, OTHER),
    ({}, OTHER),
    (None, OTHER),
])
def test_who_is_asking(identity, expected):
    assert asker(identity, OWNER_THREAD, OWNER) == expected


def test_without_an_owner_nobody_is_him():
    assert asker(TEXT, OWNER_THREAD, None) == OTHER


# --- D10: what voice may start at home --------------------------------------------------------


def test_his_text_starts_any_job_without_a_button():
    b = Box()
    assert b.gate.home_job(TEXT, OWNER_THREAD, "start_async_task", START_WRITE).go
    assert not b.approval_store.rows and not b.store.rows


def test_his_voice_on_a_strong_match_starts_a_read_only_job_directly():
    b = Box()
    assert b.gate.home_job(STRONG, OWNER_THREAD, "start_async_task", START_READ).go
    assert not b.approval_store.rows


def test_his_voice_asking_for_a_job_that_can_write_waits_for_his_button():
    b = Box()
    v = b.gate.home_job(STRONG, OWNER_THREAD, "start_async_task", START_WRITE)
    assert not v.go and "Waiting for Chris" in v.text and "Nothing has been done" in v.text
    (a,) = b.approval_store.rows.values()
    assert a.kind == "home job" and "Build the fridge planner page" in a.summary and a.status == "pending"
    assert "strong match" in a.detail and "home" in a.detail
    (row,) = b.store.rows.values()
    assert row.action == HOME_JOB and row.status == "waiting" and row.approval_id == a.id
    assert len(b.pinged) == 1 and not b.done


def test_a_voice_it_wasnt_sure_of_waits_for_the_button_even_for_a_read_only_job():
    for identity in (UNTAGGED, UNSURE):
        b = Box()
        v = b.gate.home_job(identity, OWNER_THREAD, "start_async_task", START_READ)
        assert not v.go and len(b.approval_store.rows) == 1
        assert "wasn't sure" in next(iter(b.approval_store.rows.values())).detail


def test_read_only_is_the_agent_listed_never_her_word_for_it():
    b = Box(read_only=())
    v = b.gate.home_job(STRONG, OWNER_THREAD, "start_async_task", START_READ)
    assert not v.go and len(b.approval_store.rows) == 1


def test_guests_unknown_voices_and_lines_that_arent_his_are_refused_with_no_button():
    for identity in (GUEST_VOICE, STRANGER_ASK, SAM_TEXT, None):
        b = Box()
        for tool in ("start_async_task", "update_async_task", "check_async_task", "cancel_async_task",
                     "list_async_tasks"):
            v = b.gate.home_job(identity, OWNER_THREAD, tool, START_WRITE)
            assert not v.go and "only Chris" in v.text
        assert not b.approval_store.rows and not b.store.rows


def test_checking_and_listing_his_jobs_are_his_any_way_he_asks():
    b = Box()
    for identity in (TEXT, STRONG, UNTAGGED, UNSURE, KAEL_ASK, LENS_ASK):
        for tool in ("check_async_task", "list_async_tasks"):
            assert b.gate.home_job(identity, OWNER_THREAD, tool, {"task_id": "t-1"}).go
    assert not b.approval_store.rows


def test_his_typed_words_and_his_voice_on_a_strong_match_stop_a_job_directly():
    b = Box()
    for identity in (TEXT, STRONG):
        assert b.gate.home_job(identity, OWNER_THREAD, "cancel_async_task", {"task_id": "t-1"}).go
    assert not b.approval_store.rows


def test_a_voice_it_wasnt_sure_of_stopping_a_job_waits_for_his_button():
    """D17 Q3 (11/09): it could be someone else in the house, and a cancel can end a job 40 minutes in."""
    state = {"async_tasks": {"t-rw": {"agent_name": "home", "status": "running"}}}
    for identity in (UNTAGGED, UNSURE):
        b = Box()
        v = b.gate.home_job(identity, OWNER_THREAD, "cancel_async_task", {"task_id": "t-rw"}, state)
        assert not v.go and "Waiting for Chris" in v.text
        (a,) = b.approval_store.rows.values()
        assert a.kind == "home job" and a.summary.startswith("Stop a running home job (home)")
        assert "wasn't sure" in a.detail and "t-rw" in a.detail
        b.tap()
        assert b.watcher.tick()["done"] == 1
        assert b.done == [("job", {"tool": "cancel_async_task", "task_id": "t-rw",
                                   "what": "stopping the home job, asked by voice"})]


@pytest.mark.parametrize("identity,thread,source", [
    (KAEL_ASK, KAEL_THREAD, "Kael's line"),
    (LENS_ASK, OWNER_THREAD, "the G2 lens"),
    (PANEL_ASK, OWNER_THREAD, "the panel or Sticky"),
])
def test_his_own_lines_get_his_button_instead_of_a_no(identity, thread, source):
    """D17 Q4 (11/09): the G2 lens, the panel, Sticky and Kael's line carry his id but can't show
    he typed it: starting, changing or stopping a job waits for his tap; nothing is refused."""
    b = Box()
    v = b.gate.home_job(identity, thread, "start_async_task", START_READ)
    assert not v.go and "Waiting for Chris" in v.text            # even a read-only agent: not his voice
    (a,) = b.approval_store.rows.values()
    assert source in a.summary and source in a.detail
    state = {"async_tasks": {"t-rw": {"agent_name": "home"}}}
    assert not b.gate.home_job(identity, thread, "update_async_task", {"task_id": "t-rw", "message": "x"}, state).go
    assert not b.gate.home_job(identity, thread, "cancel_async_task", {"task_id": "t-rw"}, state).go
    assert len(b.approval_store.rows) == 3 and not b.done
    b.tap(approval_id=a.id)
    assert b.watcher.tick()["done"] == 1 and b.done[0][1]["tool"] == "start_async_task"


def test_changing_a_running_job_by_voice_follows_the_job_agent():
    b = Box()
    state = {"async_tasks": {"t-ro": {"agent_name": "home-readonly"}, "t-rw": {"agent_name": "home"}}}
    assert b.gate.home_job(STRONG, OWNER_THREAD, "update_async_task",
                           {"task_id": "t-ro", "message": "also check the calendar"}, state).go
    v = b.gate.home_job(STRONG, OWNER_THREAD, "update_async_task",
                        {"task_id": "t-rw", "message": "make it blue"}, state)
    assert not v.go
    (a,) = b.approval_store.rows.values()
    assert "Change a running home job (home)" in a.summary and "make it blue" in a.summary
    # a task she has no record of is never taken for read-only
    v = b.gate.home_job(STRONG, OWNER_THREAD, "update_async_task", {"task_id": "nope", "message": "x"}, state)
    assert not v.go and len(b.approval_store.rows) == 2


def test_asking_again_before_he_answers_is_the_same_button():
    b = Box()
    b.gate.home_job(STRONG, OWNER_THREAD, "start_async_task", START_WRITE)
    v = b.gate.home_job(STRONG, OWNER_THREAD, "start_async_task", START_WRITE)
    assert not v.go and "Still waiting" in v.text
    assert len(b.approval_store.rows) == 1 and len(b.store.rows) == 1 and len(b.pinged) == 1


def test_his_approve_starts_the_job_once_and_she_hears_it_started():
    b = Box()
    b.gate.home_job(STRONG, OWNER_THREAD, "start_async_task", START_WRITE)
    assert b.watcher.tick() == {"done": 0, "failed": 0, "dropped": 0} and not b.done
    b.tap()
    assert b.watcher.tick()["done"] == 1
    assert b.done == [("job", {"tool": "start_async_task", **START_WRITE,
                               "what": "the home job you asked for by voice"})]
    assert b.notes and b.notes[0][0] == OWNER_THREAD and "has started" in b.notes[0][1] and "t-1" in b.notes[0][1]
    assert b.watcher.tick()["done"] == 0 and len(b.done) == 1
    # her asking again after the tap never starts it a second time
    v = b.gate.home_job(STRONG, OWNER_THREAD, "start_async_task", START_WRITE)
    assert not v.go and "isn't done twice" in v.text and len(b.done) == 1
    assert len(b.approval_store.rows) == 1


def test_approved_but_not_yet_carried_out_is_never_started_by_her():
    b = Box()
    b.gate.home_job(STRONG, OWNER_THREAD, "start_async_task", START_WRITE)
    b.tap()
    v = b.gate.home_job(STRONG, OWNER_THREAD, "start_async_task", START_WRITE)
    assert not v.go and "being done now" in v.text and not b.done


def test_declined_or_unanswered_is_dropped_never_done_and_she_hears_why():
    b = Box()
    b.gate.home_job(STRONG, OWNER_THREAD, "start_async_task", START_WRITE)
    b.tap(approve=False)
    assert b.watcher.tick()["dropped"] == 1 and not b.done
    assert "declined" in b.notes[-1][1] and "wasn't done" in b.notes[-1][1]
    (row,) = b.store.rows.values()
    assert row.status == "dropped"
    b2 = Box()
    b2.gate.home_job(UNTAGGED, OWNER_THREAD, "start_async_task", START_WRITE)
    b2.clock["now"] = NOW + timedelta(minutes=31)
    assert b2.watcher.tick()["dropped"] == 1 and not b2.done
    assert "didn't answer in time" in b2.notes[-1][1]


def test_after_a_no_asking_again_is_a_new_button_not_the_old_answer():
    b = Box()
    b.gate.home_job(STRONG, OWNER_THREAD, "start_async_task", START_WRITE)
    b.tap(approve=False)
    b.watcher.tick()
    v = b.gate.home_job(STRONG, OWNER_THREAD, "start_async_task", START_WRITE)
    assert not v.go and "Waiting for Chris" in v.text
    assert len(b.approval_store.rows) == 2 and sorted(r.attempt for r in b.store.rows.values()) == [1, 2]


def test_a_job_that_fails_after_his_ok_is_told_honestly_and_never_retried():
    calls = []

    def boom(row):
        calls.append(row.id)
        raise ConnectionError("home server down")

    b = Box(executors={HOME_JOB: boom})
    b.gate.home_job(STRONG, OWNER_THREAD, "start_async_task", START_WRITE)
    b.tap()
    assert b.watcher.tick()["failed"] == 1
    assert "didn't go through" in b.notes[-1][1] and "ConnectionError" in b.notes[-1][1]
    assert "done" not in b.notes[-1][1].split("don't say it's done")[0]
    b.watcher.tick()
    assert len(calls) == 1


def test_two_watchers_never_both_carry_it_out():
    b = Box()
    b.gate.home_job(STRONG, OWNER_THREAD, "start_async_task", START_WRITE)
    b.tap()
    other = HandsWatcher(b.gate)
    (row,) = b.store.rows.values()
    assert b.store.claim(row.id)                 # the other watcher got there first
    assert b.watcher.tick()["done"] == 0 and not b.done
    assert other.tick()["done"] == 0


def test_an_answer_it_cant_read_is_never_an_approval():
    b = Box()
    b.gate.home_job(STRONG, OWNER_THREAD, "start_async_task", START_WRITE)
    b.tap()

    def down(_):
        raise ConnectionError("nas")

    b.approval_store.get = down
    assert b.watcher.tick() == {"done": 0, "failed": 0, "dropped": 0} and not b.done
    v = b.gate.home_job(STRONG, OWNER_THREAD, "start_async_task", START_WRITE)
    assert not v.go and "couldn't check" in v.text


def test_nothing_to_carry_it_out_means_refused_not_queued():
    b = Box(executors={})
    v = b.gate.home_job(STRONG, OWNER_THREAD, "start_async_task", START_WRITE)
    assert not v.go and "waits until he types it" in v.text
    assert not b.approval_store.rows and not b.store.rows


def test_a_button_that_cant_be_backed_is_withdrawn():
    b = Box()

    def broken(*a, **k):
        raise ConnectionError("nas")

    b.store.add = broken
    v = b.gate.home_job(STRONG, OWNER_THREAD, "start_async_task", START_WRITE)
    assert not v.go and "couldn't put this to Chris" in v.text
    (a,) = b.approval_store.rows.values()
    assert a.status == "withdrawn"


def test_a_request_it_cant_write_is_a_no():
    b = Box()

    def broken(*a, **k):
        raise ConnectionError("nas")

    b.approval_store.request = broken
    v = b.gate.home_job(STRONG, OWNER_THREAD, "start_async_task", START_WRITE)
    assert not v.go and not b.store.rows


# --- D12: her first message to anyone new -------------------------------------------------------


def test_messages_to_him_go():
    b = Box()
    assert b.gate.message(OWNER_THREAD, "discord", "chris_h", "hi").go
    assert not b.approval_store.rows


def test_a_first_message_waits_for_a_button_naming_exactly_who_and_where():
    b = Box()
    v = b.gate.message(OWNER_THREAD, "discord", "@sam_42", "Hey Sam, Chris asked me to send the list.")
    assert not v.go and "Waiting for Chris" in v.text and "Nothing has been done" in v.text
    (a,) = b.approval_store.rows.values()
    assert a.kind == "first message" and a.summary == "First message to Sam · Discord: sam_42"
    assert "Chris asked me to send the list" in a.detail
    assert not b.done


def test_a_name_is_resolved_to_the_person_and_the_tap_confirms_it():
    b = Box()
    v = b.gate.message(OWNER_THREAD, "discord", "Sam", "Hey Sam")
    assert not v.go
    (a,) = b.approval_store.rows.values()
    assert a.summary == "First message to Sam · Discord: sam_42"
    b.tap()
    assert b.watcher.tick()["done"] == 1
    kind, args = b.done[0]
    assert kind == "msg" and args["address"] == "100000000000000002" and args["text"] == "Hey Sam"
    assert "was sent" in b.notes[-1][1]
    # confirmed: the next message to Sam on Discord goes without a button
    assert b.gate.message(OWNER_THREAD, "discord", "sam_42", "one more thing").go
    assert len(b.approval_store.rows) == 1


def test_a_confirmed_contact_is_that_address_only():
    b = Box()
    b.gate.message(OWNER_THREAD, "discord", "sam_42", "hi")
    b.tap()
    b.watcher.tick()
    v = b.gate.message(OWNER_THREAD, "email", "sam@example.net", "hi by email")
    assert not v.go and len(b.approval_store.rows) == 2
    assert list(b.approval_store.rows.values())[-1].summary == "First message to Sam · email: sam@example.net"


def test_an_address_that_matches_nobody_means_ask_him_and_no_button():
    b = Box()
    v = b.gate.message(OWNER_THREAD, "discord", "stranger99", "hi")
    assert not v.go and "don't know who" in v.text and "Ask Chris" in v.text
    assert not b.approval_store.rows and not b.store.rows


def test_more_than_one_match_means_ask_him_which_and_no_button():
    b = Box()
    v = b.gate.message(OWNER_THREAD, "discord", "Chris", "hi")
    assert not v.go and "more than one person" in v.text and "chris_h" in v.text and "chris2" in v.text
    assert not b.approval_store.rows


def test_a_second_message_while_the_first_waits_is_not_a_second_button():
    b = Box()
    b.gate.message(OWNER_THREAD, "discord", "sam_42", "first")
    v = b.gate.message(OWNER_THREAD, "discord", "sam_42", "second")
    assert not v.go and "Still waiting" in v.text and "first message to Sam" in v.text
    assert len(b.approval_store.rows) == 1 and len(b.store.rows) == 1


def test_a_declined_first_message_is_never_sent_and_the_contact_stays_unconfirmed():
    b = Box()
    b.gate.message(OWNER_THREAD, "discord", "sam_42", "first")
    b.tap(approve=False)
    assert b.watcher.tick()["dropped"] == 1 and not b.done
    assert not b.gate.message(OWNER_THREAD, "discord", "sam_42", "again").go


def test_people_records_it_cant_read_mean_nothing_sent():
    def down(platform, to):
        raise ConnectionError("prod db")

    b = Box(people=down)
    v = b.gate.message(OWNER_THREAD, "discord", "sam_42", "hi")
    assert not v.go and "can't check who" in v.text and not b.approval_store.rows
    b2 = Box()
    b2.gate.people = None
    assert not b2.gate.message(OWNER_THREAD, "discord", "sam_42", "hi").go


def test_a_message_needs_a_platform_and_a_recipient():
    b = Box()
    assert not b.gate.message(OWNER_THREAD, "sms", "sam_42", "hi").go
    assert not b.gate.message(OWNER_THREAD, "discord", "  ", "hi").go
    assert not b.approval_store.rows


def test_confirmations_it_cant_read_mean_nothing_sent():
    b = Box()
    b.gate.message(OWNER_THREAD, "discord", "sam_42", "first")
    b.tap()
    b.watcher.tick()

    def down(_):
        raise ConnectionError("nas")

    b.approval_store.get = down
    assert not b.gate.message(OWNER_THREAD, "discord", "sam_42", "again").go


# --- the middleware: D10 at the moment a job is created -------------------------------------------


def _request(name, args, identity, state=None, thread=OWNER_THREAD):
    config = {"configurable": {"thread_id": thread, "identity": identity}}
    return SimpleNamespace(tool_call={"name": name, "args": args, "id": "call-1"},
                           runtime=SimpleNamespace(config=config), state=state or {})


def test_the_middleware_lets_his_text_through_and_stops_everything_else():
    b = Box()
    mw = home_job_middleware(b.gate)
    ran = []

    def handler(req):
        ran.append(req.tool_call["name"])
        return "launched"

    assert mw.wrap_tool_call(_request("start_async_task", START_WRITE, TEXT), handler) == "launched"
    out = mw.wrap_tool_call(_request("start_async_task", START_WRITE, STRONG), handler)
    assert out.tool_call_id == "call-1" and "Waiting for Chris" in out.content
    out = mw.wrap_tool_call(_request("start_async_task", START_WRITE, GUEST_VOICE), handler)
    assert "only Chris" in out.content
    assert ran == ["start_async_task"]
    # tools that aren't hers at home pass untouched, whoever asks
    assert mw.wrap_tool_call(_request("web_search", {"q": "x"}, GUEST_VOICE), handler) == "launched"


def test_the_middleware_refuses_when_the_gate_itself_breaks():
    b = Box()

    def broken(*a, **k):
        raise RuntimeError("bug")

    b.gate.home_job = broken
    mw = home_job_middleware(b.gate)
    out = mw.wrap_tool_call(_request("start_async_task", START_WRITE, TEXT), lambda req: "launched")
    assert out != "launched" and "nothing happened" in out.content


def test_the_middleware_async_path_gates_the_same():
    b = Box()
    mw = home_job_middleware(b.gate)

    async def handler(req):
        return "launched"

    assert asyncio.run(mw.awrap_tool_call(_request("start_async_task", START_WRITE, TEXT), handler)) == "launched"
    out = asyncio.run(mw.awrap_tool_call(_request("start_async_task", START_WRITE, UNTAGGED), handler))
    assert "Waiting for Chris" in out.content


# --- arming --------------------------------------------------------------------------------------


def test_not_armed_unless_switched_on_with_the_button_armed():
    base = dict(hands_gates=True, database_url="postgresql://x/aerys_v2", owner_person_id=OWNER,
                approvals=True, reach_notify_service="mobile_app_phone", storm_notify_service="",
                ha_token=SimpleNamespace(get_secret_value=lambda: "t"), ha_base_url="http://ha:8123",
                memories_database_url="postgresql://x/aerys", home_read_only_agents=" home-readonly , ")
    assert hands_gate_for(SimpleNamespace(**{**base, "hands_gates": False})) is None
    assert hands_gate_for(SimpleNamespace(**{**base, "approvals": False})) is None
    assert hands_gate_for(SimpleNamespace(**{**base, "reach_notify_service": ""})) is None
    assert hands_gate_for(SimpleNamespace(**{**base, "owner_person_id": None})) is None
    g = hands_gate_for(SimpleNamespace(**base))
    assert g is not None and g.read_only_agents == frozenset({"home-readonly"}) and g.people is not None
    assert g.executors == {}                    # nothing carries out an approved action until wired


# --- Codex review 1 (11/09) -------------------------------------------------------------------


def test_two_first_messages_racing_never_send_words_he_didnt_see():
    b = Box()
    real_add = b.store.add
    raced, started = [], []

    def add_after_the_other_turn(*a, **k):
        if not started:                                  # her other turn gets in between ask and record
            started.append(True)
            raced.append(b.gate.message(OWNER_THREAD, "discord", "sam_42", "UNSHOWN replacement"))
        return real_add(*a, **k)

    b.store.add = add_after_the_other_turn
    first = b.gate.message(OWNER_THREAD, "discord", "sam_42", "SHOWN original")
    b.store.add = real_add
    assert not first.go and not raced[0].go
    for a in b.approval_store.rows.values():
        if a.status == "pending":
            b.tap(approval_id=a.id)
    b.watcher.tick()
    assert len(b.done) == 1
    sent = b.done[0][1]["text"]
    (row,) = [r for r in b.store.rows.values() if r.status == "done"]
    assert sent in b.approval_store.rows[row.approval_id].detail      # what went is what he saw


def test_retrying_an_approved_first_message_never_sends_it_twice():
    b = Box()
    b.gate.message(OWNER_THREAD, "discord", "sam_42", "hi Sam")
    b.tap()
    v = b.gate.message(OWNER_THREAD, "discord", "sam_42", "hi Sam")      # before the watcher sends it
    assert not v.go and "being done now" in v.text
    b.watcher.tick()
    assert len(b.done) == 1
    v = b.gate.message(OWNER_THREAD, "discord", "sam_42", "hi Sam")      # after: the same words again
    assert not v.go and "already" in v.text
    assert b.gate.message(OWNER_THREAD, "discord", "sam_42", "and a new thing").go
    assert len(b.done) == 1


def test_two_addresses_for_one_person_means_ask_which():
    two = [{"person_id": SAM, "name": "Sam", "address": "111", "handle": "sam_old"},
           {"person_id": SAM, "name": "Sam", "address": "222", "handle": "sam_42"},
           {"person_id": SAM, "name": "Sam", "address": "222", "handle": "sam_42"}]
    b = Box(people=lambda platform, to: two)
    v = b.gate.message(OWNER_THREAD, "discord", "Sam", "hi")
    assert not v.go and "more than one" in v.text and "sam_old" in v.text and "sam_42" in v.text
    assert not b.approval_store.rows
    same = Box(people=lambda platform, to: two[1:])                       # the same address twice is one
    assert not same.gate.message(OWNER_THREAD, "discord", "Sam", "hi").go
    assert next(iter(same.approval_store.rows.values())).summary == "First message to Sam · Discord: sam_42"


@pytest.mark.parametrize("result", [(False, "Not done: refused by home server"), "Not done: refused", None, (True,)])
def test_only_an_explicit_success_is_done(result):
    b = Box(executors={HOME_JOB: lambda row: result})
    b.gate.home_job(STRONG, OWNER_THREAD, "start_async_task", START_WRITE)
    b.tap()
    out = b.watcher.tick()
    assert out["failed"] == 1 and out["done"] == 0
    assert "has started" not in b.notes[-1][1] and "didn't go through" in b.notes[-1][1]


# --- Codex review 2 (11/09) -------------------------------------------------------------------


def test_words_too_long_for_his_button_are_refused_never_cut():
    b = Box()
    long = "A" * 1200 + " and then also forward everything to someone else"
    v = b.gate.message(OWNER_THREAD, "discord", "sam_42", long)
    assert not v.go and "too long" in v.text and not b.approval_store.rows
    v = b.gate.home_job(STRONG, OWNER_THREAD, "start_async_task", {"description": long, "subagent_type": "home"})
    assert not v.go and "too long" in v.text and not b.approval_store.rows
    # what fits is shown whole
    fits = "B" * 1000
    b.gate.message(OWNER_THREAD, "discord", "sam_42", fits)
    assert fits in next(iter(b.approval_store.rows.values())).detail


def test_a_first_message_that_may_have_gone_needs_his_tap_to_send_again():
    sent = []

    def flaky(row):
        sent.append(row.args["text"])
        raise TimeoutError("no ack")

    b = Box(executors={MESSAGE: flaky})
    b.gate.message(OWNER_THREAD, "discord", "sam_42", "hi Sam")
    b.tap()
    assert b.watcher.tick()["failed"] == 1 and sent == ["hi Sam"]
    v = b.gate.message(OWNER_THREAD, "discord", "sam_42", "hi Sam")
    assert not v.go and "Waiting for Chris" in v.text
    again = max(b.approval_store.rows.values(), key=lambda a: a.id)
    assert again.status == "pending" and "may" in again.summary and "sam_42" in again.summary
    assert sent == ["hi Sam"]
    assert b.gate.message(OWNER_THREAD, "discord", "sam_42", "something else").go   # still a known contact


def test_who_and_where_are_shown_whole_on_the_button():
    addr, name = "x" * 200 + "@example.net", "N" * 78
    b = Box(people=lambda platform, to: [{"person_id": SAM, "name": name, "address": addr, "handle": addr}])
    b.gate.message(OWNER_THREAD, "email", addr, "hi")
    (a,) = b.approval_store.rows.values()
    assert addr in a.detail and name in a.detail and "hi" in a.detail


def test_the_resend_warning_is_in_what_he_reads_whole():
    def flaky(row):
        raise TimeoutError("no ack")

    b = Box(executors={MESSAGE: flaky})
    b.gate.message(OWNER_THREAD, "discord", "sam_42", "hi Sam")
    b.tap()
    b.watcher.tick()
    b.gate.message(OWNER_THREAD, "discord", "sam_42", "hi Sam")
    again = max(b.approval_store.rows.values(), key=lambda a: a.id)
    assert "may have gone through" in again.detail and "sam_42" in again.detail


# --- D17 Q5 (11/09): after his no, she leaves it for the day ----------------------------------
# "ok, keep it adaptive. She can resurface it later if it makes sense to ask again": a no holds
# for the rest of that day (Eastern); from the next day she may ask again. Him raising it himself
# (his typed words, or his voice on a strong match) is never held.


def _declined(b, identity, thread=OWNER_THREAD, args=START_WRITE):
    b.gate.home_job(identity, thread, "start_async_task", args)
    b.tap(approve=False)
    b.watcher.tick()


def test_after_his_no_she_leaves_the_same_job_for_the_rest_of_the_day():
    b = Box()
    _declined(b, UNTAGGED)
    b.clock["now"] = NOW + timedelta(hours=5)                    # 3 PM Eastern, same day
    v = b.gate.home_job(UNTAGGED, OWNER_THREAD, "start_async_task", START_WRITE)
    assert not v.go and "said no" in v.text and "10:00 AM" in v.text and "today" in v.text
    assert len(b.approval_store.rows) == 1 and len(b.pinged) == 1 and not b.done
    # a different job is a different request: it gets its own button
    assert "Waiting for Chris" in b.gate.home_job(UNTAGGED, OWNER_THREAD, "start_async_task",
                                                  {"description": "Something else", "subagent_type": "home"}).text
    # the next day she may bring it back
    b.clock["now"] = NOW + timedelta(days=1)
    v = b.gate.home_job(UNTAGGED, OWNER_THREAD, "start_async_task", START_WRITE)
    assert not v.go and "Waiting for Chris" in v.text and len(b.approval_store.rows) == 3


def test_the_day_is_his_eastern_day_not_utc():
    b = Box()
    b.clock["now"] = datetime(2026, 11, 10, 3, 30, tzinfo=timezone.utc)      # 10:30 PM Eastern, 11/9
    _declined(b, KAEL_ASK, KAEL_THREAD)
    b.clock["now"] = datetime(2026, 11, 10, 4, 30, tzinfo=timezone.utc)      # 11:30 PM Eastern: same day
    v = b.gate.home_job(KAEL_ASK, KAEL_THREAD, "start_async_task", START_WRITE)
    assert not v.go and "10:30 PM" in v.text and len(b.approval_store.rows) == 1
    b.clock["now"] = datetime(2026, 11, 10, 5, 30, tzinfo=timezone.utc)      # 12:30 AM Eastern, 11/10
    assert "Waiting for Chris" in b.gate.home_job(KAEL_ASK, KAEL_THREAD, "start_async_task", START_WRITE).text


def test_him_raising_it_again_himself_is_never_held():
    b = Box()
    _declined(b, UNTAGGED)
    assert b.gate.home_job(TEXT, OWNER_THREAD, "start_async_task", START_WRITE).go      # his typed words
    v = b.gate.home_job(STRONG, OWNER_THREAD, "start_async_task", START_WRITE)            # his voice, strong match
    assert not v.go and "Waiting for Chris" in v.text and len(b.approval_store.rows) == 2


def test_a_request_that_ran_out_or_was_withdrawn_isnt_a_no():
    b = Box()
    b.gate.home_job(UNTAGGED, OWNER_THREAD, "start_async_task", START_WRITE)
    b.clock["now"] = NOW + timedelta(minutes=31)                 # expired unanswered
    b.watcher.tick()
    v = b.gate.home_job(UNTAGGED, OWNER_THREAD, "start_async_task", START_WRITE)
    assert not v.go and "Waiting for Chris" in v.text and len(b.approval_store.rows) == 2


def test_a_declined_first_message_isnt_asked_again_today_but_new_words_are():
    b = Box()
    b.gate.message(OWNER_THREAD, "discord", "sam_42", "hi Sam")
    b.tap(approve=False)
    b.watcher.tick()
    v = b.gate.message(OWNER_THREAD, "discord", "sam_42", "hi Sam")
    assert not v.go and "said no" in v.text and "today" in v.text and len(b.approval_store.rows) == 1
    v = b.gate.message(OWNER_THREAD, "discord", "sam_42", "a different message")
    assert not v.go and "Waiting for Chris" in v.text and len(b.approval_store.rows) == 2


def test_him_asking_for_the_same_message_again_gets_a_new_button():
    b = Box()
    b.gate.message(OWNER_THREAD, "discord", "sam_42", "hi Sam")
    b.tap(approve=False)
    b.watcher.tick()
    # her own asking (no identity, a voice it wasn't sure of, one of his /ask lines) stays held
    for identity in (None, UNSURE, KAEL_ASK):
        assert "said no" in b.gate.message(OWNER_THREAD, "discord", "sam_42", "hi Sam", identity=identity).text
    v = b.gate.message(OWNER_THREAD, "discord", "sam_42", "hi Sam", identity=TEXT)
    assert not v.go and "Waiting for Chris" in v.text and len(b.approval_store.rows) == 2
