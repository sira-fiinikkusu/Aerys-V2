"""D15 meeting rule (the owner, 11/08): "doesnt speak over voice while a meeting is active
and speaks at the next window with a 10 minute buffer". Offline: a fake HA calendar API."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from langchain_core.messages import AIMessage

from aerys_v2.kael_line import (
    kael_note_for,
    meeting_hold_fn_for,
    owner_reach_for,
    ticket_relay_for,
)
from aerys_v2.tickets import Ticket

OWNER = "6e6bcbed-0000-0000-0000-000000000000"
NOW = datetime(2026, 11, 9, 15, 0, tzinfo=timezone.utc)          # 10:00 Eastern


class _Secret:
    def __init__(self, v):
        self.v = v

    def get_secret_value(self):
        return self.v


def settings(**kw):
    base = dict(owner_meeting_calendars="calendar.work", owner_meeting_buffer_min=10, ha_token=_Secret("ha"),
                ha_base_url="http://ha.test:8123", owner_person_id=OWNER, telegram_bot_token=None,
                discord_bot_token=_Secret("dc"), owner_quiet_start_hour=21, owner_quiet_end_hour=9,
                database_url=None, memories_database_url=None,
                soul_file_path=__import__("pathlib").Path("/nonexistent/soul.md"))
    base.update(kw)
    return SimpleNamespace(**base)


def timed(start, end):
    return {"summary": "Standup", "start": {"dateTime": start.isoformat()}, "end": {"dateTime": end.isoformat()}}


class Calendar:
    def __init__(self, events=None, fail=False):
        self.events, self.fail, self.calls = events or [], fail, []

    def __call__(self, url, headers):
        self.calls.append(url)
        if self.fail:
            raise ConnectionError("ha down")
        return SimpleNamespace(raise_for_status=lambda: None, json=lambda: self.events)


def hold_with(cal, now=NOW, **kw):
    clock = {"now": now}
    fn = meeting_hold_fn_for(settings(**kw), get=cal, clock=lambda: clock["now"])
    return fn, clock


def test_no_meeting_calendar_or_no_ha_means_no_hold():
    assert meeting_hold_fn_for(settings(owner_meeting_calendars="")) is None
    assert meeting_hold_fn_for(settings(ha_token=None)) is None


def test_a_meeting_on_now_holds_until_ten_minutes_after_it_ends():
    fn, _ = hold_with(Calendar([timed(NOW - timedelta(minutes=20), NOW + timedelta(minutes=30))]))
    assert fn() == NOW + timedelta(minutes=40)


def test_the_buffer_after_a_meeting_still_holds_and_then_it_lets_go():
    ended = Calendar([timed(NOW - timedelta(minutes=60), NOW - timedelta(minutes=5))])
    fn, _ = hold_with(ended)
    assert fn() == NOW + timedelta(minutes=5)
    long_ago = Calendar([timed(NOW - timedelta(minutes=60), NOW - timedelta(minutes=11))])
    fn, _ = hold_with(long_ago)
    assert fn() is None


def test_back_to_back_meetings_chain_and_a_later_one_does_not_hold_yet():
    cal = Calendar([timed(NOW - timedelta(minutes=30), NOW + timedelta(minutes=5)),
                    timed(NOW + timedelta(minutes=5), NOW + timedelta(minutes=35)),   # not started yet
                    timed(NOW - timedelta(minutes=10), NOW + timedelta(minutes=15))])
    fn, _ = hold_with(cal)
    assert fn() == NOW + timedelta(minutes=25)


def test_an_all_day_entry_is_never_a_meeting():
    cal = Calendar([{"summary": "Out of office", "start": {"date": "2026-11-09"}, "end": {"date": "2026-11-10"}}])
    fn, _ = hold_with(cal)
    assert fn() is None


def test_an_unreadable_calendar_holds_nothing_and_says_so(caplog):
    import logging

    fn, _ = hold_with(Calendar(fail=True))
    with caplog.at_level(logging.WARNING, logger="aerys_v2.kael_line"):
        assert fn() is None
    assert any("not holding" in r.getMessage() for r in caplog.records)


def test_the_calendar_is_read_at_most_once_a_minute():
    cal = Calendar([timed(NOW - timedelta(minutes=5), NOW + timedelta(minutes=5))])
    fn, clock = hold_with(cal)
    fn(); fn()
    clock["now"] = NOW + timedelta(seconds=59)
    fn()
    assert len(cal.calls) == 1
    clock["now"] = NOW + timedelta(seconds=61)
    fn()
    assert len(cal.calls) == 2


def test_offsetless_event_times_are_eastern():
    start = datetime(2026, 11, 9, 9, 30)          # 9:30 Eastern, no offset
    end = datetime(2026, 11, 9, 10, 30)
    fn, _ = hold_with(Calendar([{"start": {"dateTime": start.isoformat()}, "end": {"dateTime": end.isoformat()}}]))
    assert fn() == datetime(2026, 11, 9, 15, 40, tzinfo=timezone.utc)


# --- where it bites ----------------------------------------------------------------------------

def reach_in(surface, meeting: bool, sent):
    return owner_reach_for(settings(), send=lambda *a: sent.append(a), lookup_surface=lambda s: (surface, "office"),
                           lookup_chat_id=lambda s, p: f"{p}-id", clock=lambda: NOW,
                           hold_fn=lambda: (NOW + timedelta(minutes=30)) if meeting else None)


def test_in_a_meeting_speech_waits_and_text_goes_without_a_buzz():
    sent = []
    voice = reach_in("device", True, sent)
    assert voice.held() is True                    # a ticket answer waits for the window
    assert voice("hi") and sent[-1] == ("discord", "discord-id", "hi", True)   # never out loud
    text = reach_in("discord", True, sent)
    assert text.held() is False                    # text doesn't speak over anything
    assert text("hi") and sent[-1] == ("discord", "discord-id", "hi", True)
    free = reach_in("device", False, sent)
    assert free.held() is False and free("hi") and sent[-1] == ("device", "office", "hi", False)


def test_a_ticket_answer_waits_for_the_window_without_composing_anything():
    class CountingModel:
        calls = 0

        def invoke(self, messages):
            CountingModel.calls += 1
            return AIMessage(content="Kael says yes.")

    sent, meeting = [], {"on": True}
    reach = owner_reach_for(settings(), send=lambda *a: sent.append(a), lookup_surface=lambda s: ("device", "office"),
                            lookup_chat_id=lambda s, p: "id", clock=lambda: NOW,
                            hold_fn=lambda: NOW + timedelta(minutes=5) if meeting["on"] else None)

    class Graph:
        def update_state(self, *a, **k):
            pass

    relay = ticket_relay_for(Graph(), settings(), model=CountingModel(), reach=reach)
    t = Ticket(id=1, thread_id="person:x", ask="is the backup green?", for_owner=True, status="completed",
               answer="yes", created_at=NOW)
    assert relay(t) is False and sent == [] and CountingModel.calls == 0
    meeting["on"] = False
    assert relay(t) is True and sent[0][0] == "device" and CountingModel.calls == 1


def test_the_kael_note_turn_in_a_meeting_goes_as_quiet_text_not_out_loud():
    spoke = []

    class Graph:
        def update_state(self, *a, **k):
            pass

    note = kael_note_for(Graph(), settings(), proactive_model=SimpleNamespace(invoke=lambda m: AIMessage(content="News!")),
                         proactive_send=lambda *a: spoke.append(a), proactive_lookup_chat_id=lambda s: "id",
                         proactive_lookup_surface=lambda s: ("device", "office"), proactive_sync=True,
                         proactive_meeting_hold=lambda: NOW + timedelta(minutes=5))
    note("person:" + OWNER, "family news", True)
    assert spoke and spoke[0][0] == "discord"


# --- Codex review 11/08 -------------------------------------------------------------------------

def test_a_meeting_that_starts_between_two_reads_holds_from_its_first_second():
    cal = Calendar([timed(NOW + timedelta(seconds=10), NOW + timedelta(minutes=30))])
    fn, clock = hold_with(cal)
    assert fn() is None                            # 09:59:50-ish: not started yet
    clock["now"] = NOW + timedelta(seconds=20)     # same cache minute, meeting now on
    assert fn() == NOW + timedelta(minutes=40)
    assert len(cal.calls) == 1


def test_the_kael_note_turn_rechecks_at_delivery_when_a_meeting_starts_while_she_composes():
    spoke, meeting = [], {"on": False}

    class Model:
        def invoke(self, messages):
            meeting["on"] = True                   # the meeting starts while she's composing
            return AIMessage(content="News!")

    class Graph:
        def update_state(self, *a, **k):
            pass

    note = kael_note_for(Graph(), settings(), proactive_model=Model(), proactive_send=lambda *a: spoke.append(a),
                         proactive_lookup_chat_id=lambda s: "id", proactive_lookup_surface=lambda s: ("device", "office"),
                         proactive_sync=True,
                         proactive_meeting_hold=lambda: NOW + timedelta(minutes=5) if meeting["on"] else None)
    note("person:" + OWNER, "family news", True)
    assert spoke and spoke[0][0] == "discord"


def test_an_answer_in_its_last_hour_stops_waiting_and_goes_as_quiet_text():
    sent = []
    reach = owner_reach_for(settings(), send=lambda *a: sent.append(a), lookup_surface=lambda s: ("device", "office"),
                            lookup_chat_id=lambda s, p: "id", clock=lambda: NOW,
                            hold_fn=lambda: NOW + timedelta(minutes=30))

    class Graph:
        def update_state(self, *a, **k):
            pass

    relay = ticket_relay_for(Graph(), settings(), model=SimpleNamespace(invoke=lambda m: AIMessage(content="Kael: yes.")),
                             reach=reach)
    old = Ticket(id=2, thread_id="person:x", ask="?", for_owner=True, status="completed", answer="yes",
                 created_at=NOW, answered_at=datetime.now(timezone.utc) - timedelta(hours=23, minutes=30))
    assert relay(old) is True
    assert sent[0][0] == "discord" and sent[0][3] is True      # quiet text, never out loud
