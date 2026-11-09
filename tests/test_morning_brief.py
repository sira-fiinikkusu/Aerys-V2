"""Her morning brief (board #176, Stage 2). The owner, 2026-11-08: Kael's brief stays on his
phone and watch; hers is spoken "in the office when she first detects me in there after
waking ... this way im more awake and it doesnt impact robin"."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from aerys_v2.brief import (
    Facts,
    MorningBrief,
    Reading,
    facts_text,
    gatherer_for,
    held_detail,
    morning_brief_for,
    plain_brief,
    reader_for,
)
from aerys_v2.reach import MemoryReachStore, Prefs, ReachDesk, ReachEvent, Where, decide
from aerys_v2.tools.reach_prefs import build_reach_prefs_tool
from test_reach_desk import World

ET = timezone(timedelta(hours=-5))                       # EST on these dates
NIGHT = datetime(2026, 11, 9, 23, 0, tzinfo=ET)           # Monday 11 PM
WOKE = datetime(2026, 11, 10, 8, 5, tzinfo=ET)            # Tuesday: his watch came on
IN_OFFICE = datetime(2026, 11, 10, 8, 20, tzinfo=ET)      # ...and he walked into the office
OFFICE = Where("room", "dev-office", "office")
KITCHEN = Where("room", "dev-kitchen", "kitchen")


def quiet(t):
    hour = t.astimezone(ET).hour
    return hour >= 21 or hour < 9


class House:
    """What the brief reads, what it composes, and the desk it hands to."""

    def __init__(self, now=IN_OFFICE, *, store=None):
        self.world = World(OFFICE)
        self.clock = {"now": now}
        self.meeting = None
        self.reading = Reading(room="office", confirmed=True, woke_at=WOKE.astimezone(timezone.utc))
        self.composed = []
        self.words = "Morning! Sunny, high of 82. You've got two work meetings, the first at ten."
        self.desk = ReachDesk(store or MemoryReachStore(), self.world.deliverer(), budget_default=3,
                              quiet=quiet, morning_hour=13,           # the brief is armed
                              meeting_hold=lambda: self.meeting,
                              replied_since=lambda when: self.world.replied,
                              said=lambda row, body: self.world.said.append(body),
                              clock=lambda: self.clock["now"])
        self.brief = MorningBrief(self.desk, read=lambda: self.reading, gather=self.gather,
                                  compose=self.compose, replied_since=lambda when: self.world.replied,
                                  clock=lambda: self.clock["now"])

    def gather(self, now, held, quiet_rows):
        return Facts(weather="sunny, high 82, low 66", events=["10:00 AM to 10:30 AM: a work meeting"],
                     held=held, quiet=quiet_rows)

    def compose(self, facts):
        self.composed.append(facts)
        if isinstance(self.words, Exception):
            raise self.words
        return self.words

    def at(self, when):
        self.clock["now"] = when
        return self


# --- when it goes -----------------------------------------------------------------------

def test_spoken_in_the_office_once_he_is_up_and_there_even_before_quiet_hours_end():
    h = House()
    assert h.brief.tick() == "sent"
    assert h.world.spoken == [(h.words, "dev-office")]
    assert h.world.texts == []                       # nothing held overnight: nothing to text
    row = h.desk.store.rows[1]
    assert row.brief and row.room == "office" and row.dedupe_key == "brief:2026-11-10"
    assert h.desk.store.sent_since(datetime(2026, 11, 10, 0, 0, tzinfo=ET)) == 0   # outside the budget
    assert h.world.said == [h.words]                 # her thread remembers she said it


def test_once_a_day_even_across_a_restart():
    h = House()
    assert h.brief.tick() == "sent"
    assert h.at(IN_OFFICE + timedelta(minutes=30)).brief.tick() == "done"
    again = MorningBrief(h.desk, read=lambda: h.reading, gather=h.gather, compose=h.compose,
                         clock=lambda: h.clock["now"])          # a fresh process, same ledger
    assert again.tick() == "done" and len(h.composed) == 1 and len(h.world.spoken) == 1


def test_not_in_the_bedroom_and_not_on_his_phone_alone():
    h = House()
    h.reading = Reading(room="bedroom", confirmed=True, woke_at=h.reading.woke_at)
    assert h.brief.tick() == "not in the office"
    h.reading = Reading(room="office", confirmed=False, woke_at=h.reading.woke_at)   # phone left charging there
    assert h.brief.tick() == "not in the office"
    assert h.world.spoken == [] and h.composed == []


def test_before_eight_it_needs_proof_he_is_up():
    early = datetime(2026, 11, 10, 7, 15, tzinfo=ET)
    h = House(early)
    h.reading = Reading(room="office", confirmed=True, woke_at=None)
    assert h.brief.tick() == "not up yet"
    h.reading.woke_at = datetime(2026, 11, 9, 8, 5, tzinfo=ET).astimezone(timezone.utc)   # yesterday's wake
    assert h.brief.tick() == "not up yet"
    h.world.replied = True                                # he talked to her this morning
    assert h.brief.tick() == "sent"


def test_from_eight_the_office_alone_will_do():
    h = House(datetime(2026, 11, 10, 8, 40, tzinfo=ET))
    h.reading = Reading(room="office", confirmed=True, woke_at=None)
    assert h.brief.tick() == "sent"


def test_only_inside_its_window():
    for when in (datetime(2026, 11, 10, 4, 59, tzinfo=ET), datetime(2026, 11, 10, 13, 0, tzinfo=ET)):
        h = House(when)
        assert h.brief.tick() == "outside the window" and h.composed == []


def test_a_meeting_holds_it_and_no_words_are_spent_while_held():
    h = House()
    h.meeting = IN_OFFICE + timedelta(minutes=40)
    assert h.brief.tick() == "held: meeting" and h.composed == []
    h.meeting = None
    assert h.brief.tick() == "sent"


def test_he_can_mute_it_in_words():
    h = House()
    tool = build_reach_prefs_tool(h.desk.store, clock=lambda: h.clock["now"])
    assert "won't" in tool.invoke({"action": "mute", "topic": "brief", "his_words": "skip the morning brief"})
    assert h.brief.tick() == "muted" and h.composed == []


def test_if_her_words_fail_the_plain_version_goes():
    h = House()
    h.words = RuntimeError("model down")
    assert h.brief.tick() == "sent"
    assert h.world.spoken[0][0].startswith("Good morning.")
    h2 = House()
    h2.words = "   "
    assert h2.brief.tick() == "sent" and h2.world.spoken[0][0].startswith("Good morning.")


# --- what the night held ------------------------------------------------------------------

def test_what_the_night_held_waits_for_the_brief_and_goes_with_it():
    h = House(NIGHT)
    assert h.desk.reach_out(ReachEvent(source="kael_note", topic="kael", headline="Kael finished the backup check.",
                                       detail="All green.")) == "queued"
    held = h.desk.store.rows[1]
    assert held.not_before == datetime(2026, 11, 10, 13, 0, tzinfo=ET)    # not 9: the brief carries it
    h.at(datetime(2026, 11, 10, 9, 5, tzinfo=ET))
    h.reading = Reading(room="bedroom", confirmed=True, woke_at=WOKE.astimezone(timezone.utc))
    assert h.desk.tick()["sent"] == 0 and h.brief.tick() == "not in the office"
    h.reading = Reading(room="office", confirmed=True, woke_at=WOKE.astimezone(timezone.utc))
    assert h.brief.tick() == "sent"
    assert [r.headline for r in h.composed[0].held] == ["Kael finished the backup check."]
    assert h.world.texts and "Kael finished the backup check." in h.world.texts[0][1] and "All green." in h.world.texts[0][1]
    assert h.desk.store.rows[1].status == "folded"
    h.at(datetime(2026, 11, 10, 13, 1, tzinfo=ET))
    assert h.desk.tick()["sent"] == 0 and len(h.world.spoken) == 1       # never twice


def test_with_no_brief_by_the_cutoff_the_held_item_goes_on_its_own():
    h = House(NIGHT)
    h.desk.reach_out(ReachEvent(source="kael_note", topic="kael", headline="Kael finished the backup check."))
    h.at(datetime(2026, 11, 10, 13, 1, tzinfo=ET))
    assert h.brief.tick() == "outside the window"
    assert h.desk.tick()["sent"] == 1 and h.world.spoken[0][0] == "Kael finished the backup check."


def test_pinned_to_the_office_it_waits_there_and_never_follows_him():
    h = House()
    h.world.busy = True                                   # the office satellite is mid-conversation
    assert h.brief.tick() == "queued" and h.world.spoken == []
    h.world.busy, h.world.where = False, KITCHEN          # ...and then he went to the kitchen
    h.at(IN_OFFICE + timedelta(minutes=2))
    h.desk.tick()
    assert h.world.spoken == [] and "waits for him in the office" in h.desk.store.rows[1].reason
    h.world.where = OFFICE
    h.at(IN_OFFICE + timedelta(minutes=5))
    assert h.desk.tick()["sent"] == 1 and h.world.spoken[0][1] == "dev-office"


def test_a_brief_that_never_reaches_the_office_expires_and_the_held_items_go_on_their_own():
    h = House(NIGHT)
    h.desk.reach_out(ReachEvent(source="kael_note", topic="kael", headline="Kael finished the backup check."))
    h.at(IN_OFFICE)
    h.world.busy = True
    assert h.brief.tick() == "queued"
    h.world.busy, h.world.where = False, KITCHEN
    h.at(datetime(2026, 11, 10, 13, 1, tzinfo=ET))
    h.desk.tick()
    statuses = {r.source: r.status for r in h.desk.store.rows.values()}
    assert statuses == {"kael_note": "sent", "brief": "dropped"}
    assert h.world.spoken == [("Kael finished the backup check.", "dev-kitchen")]


def test_no_in_case_you_missed_it_copy_for_a_brief():
    h = House()
    h.brief.tick()
    h.at(IN_OFFICE + timedelta(minutes=7))
    assert h.desk.tick()["copied"] == 0 and h.world.texts == []


def test_what_she_already_sent_quietly_overnight_is_mentioned_not_repeated():
    h = House(NIGHT)
    assert h.desk.reach_out(ReachEvent(source="ticket", topic="kael", headline="Kael says the NAS is fine.",
                                       owed=True)) == "sent"
    h.at(IN_OFFICE)
    assert h.brief.tick() == "sent"
    assert [r.headline for r in h.composed[0].quiet] == ["Kael says the NAS is fine."]
    assert h.desk.store.rows[1].status == "sent"         # not folded: it already went


# --- the desk's rules for a brief -----------------------------------------------------------

def test_decide_lets_a_brief_through_the_night_and_past_the_budget_but_not_a_meeting():
    brief = ReachEvent(source="brief", topic="brief", headline="Morning.", brief=True)
    kw = dict(prefs=Prefs(budget=0), sent_today=5, duplicate=False, morning_hour=9)
    assert decide(brief, IN_OFFICE, quiet=True, meeting_until=None, **kw).action == "send"
    assert decide(brief, IN_OFFICE, quiet=False, meeting_until=IN_OFFICE + timedelta(minutes=30), **kw).action == "queue"


# --- the words ------------------------------------------------------------------------------

def test_the_plain_version_and_the_facts_she_reads():
    facts = Facts(weather="rain, high 75", events=["9:30 AM: dentist", "2:00 PM to 3:00 PM: a work meeting"])
    assert plain_brief(facts) == "Good morning. Today: rain, high 75. First up, dentist at 9:30 AM, and 1 more after that."
    work = Facts(events=["10:00 AM to 10:30 AM: a work meeting"])
    assert plain_brief(work) == "Good morning. First up, a work meeting at 10:00 AM."
    assert plain_brief(Facts(events=["all day: Robin's birthday"])) == "Good morning. First up, all day: Robin's birthday."
    assert plain_brief(Facts()) == "Good morning. Nothing on the calendar for the rest of today."
    text = facts_text(facts)
    assert "Weather today: rain, high 75" in text and "- 9:30 AM: dentist" in text and "overnight: nothing" in text
    row = SimpleNamespace(headline="Kael finished.", detail="All green.")
    assert held_detail([row]) == "Held for you overnight:\n- Kael finished.\n  All green."
    assert held_detail([]) == ""


# --- production legs --------------------------------------------------------------------------

def settings(**kw):
    base = dict(ha_base_url="http://ha:8123", ha_token=SimpleNamespace(get_secret_value=lambda: "t"),
                presence_settled_entity="sensor.settled", brief_wake_entities="automation.brief,binary_sensor.watch",
                owner_meeting_calendars="calendar.work", brief_calendars="calendar.home,calendar.work",
                brief_weather_entity="weather.home", morning_brief=True, brief_room="office",
                reach_room_devices="office=dev-office,kitchen=dev-kitchen", soul_file_path="")
    base.update(kw)
    return SimpleNamespace(**base)


class Resp:
    def __init__(self, data):
        self.data, self.text = data, json.dumps(data)

    def raise_for_status(self):
        pass

    def json(self):
        return self.data


def test_the_house_reading():
    sent = {}

    def post(url, headers, body):
        sent["url"], sent["template"] = url, body["template"]
        return Resp({"room": "Office", "confidence": "confirmed",
                     "wakes": ["2026-11-10T13:05:00+00:00", "2026-11-10 13:07:12.5+00:00", "garbage"]})

    r = reader_for(settings(), post=post)()
    assert sent["url"] == "http://ha:8123/api/template"
    assert "state_attr('automation.brief', 'last_triggered')" in sent["template"]
    assert "is_state('binary_sensor.watch', 'on')" in sent["template"]
    assert r.room == "office" and r.confirmed and r.woke_at == datetime(2026, 11, 10, 13, 7, 12, 500000, tzinfo=timezone.utc)


def test_the_day_from_ha_work_meetings_as_times_only():
    def get(url, headers):
        if "calendar.work" in url:
            return Resp([{"summary": "Quarterly planning", "start": {"dateTime": "2026-11-10T10:00:00-05:00"},
                          "end": {"dateTime": "2026-11-10T10:30:00-05:00"}},
                         {"summary": "Out of office", "start": {"date": "2026-11-10"}, "end": {"date": "2026-11-11"}}])
        return Resp([{"summary": "Dentist", "start": {"dateTime": "2026-11-10T09:30:00-05:00"},
                      "end": {"dateTime": "2026-11-10T10:00:00-05:00"}},
                     {"summary": "Robin's birthday", "start": {"date": "2026-11-10"}, "end": {"date": "2026-11-11"}}])

    def post(url, headers, body):
        assert url.endswith("/api/services/weather/get_forecasts?return_response")
        return Resp({"service_response": {"weather.home": {"forecast": [
            {"condition": "partlycloudy", "temperature": 82.4, "templow": 66.0, "precipitation_probability": 20}]}}})

    facts = gatherer_for(settings(), get=get, post=post)(IN_OFFICE, [], [])
    assert facts.weather == "partly cloudy, high 82, low 66, 20% chance of rain"
    assert facts.events == ["all day: Robin's birthday", "9:30 AM to 10:00 AM: Dentist",
                            "10:00 AM to 10:30 AM: a work meeting"]
    assert not any("Quarterly" in e or "Out of office" in e for e in facts.events)


def test_a_failed_source_is_left_out_not_fatal():
    def get(url, headers):
        raise ConnectionError("calendar down")

    def post(url, headers, body):
        raise ConnectionError("weather down")

    facts = gatherer_for(settings(), get=get, post=post)(IN_OFFICE, [], [])
    assert facts.weather == "" and facts.events == []


@pytest.mark.parametrize("kw", [dict(morning_brief=False), dict(reach_room_devices="kitchen=dev-kitchen"),
                                dict(ha_token=None)])
def test_off_unless_armed_with_a_speaker_in_its_room(kw):
    assert morning_brief_for(settings(**kw), desk=object()) is None
    assert morning_brief_for(settings(), desk=None) is None


def test_armed():
    desk = SimpleNamespace(replied_since=lambda when: False)
    brief = morning_brief_for(settings(), desk)
    assert isinstance(brief, MorningBrief) and brief.room == "office" and brief.replied_since is desk.replied_since


# --- Codex review 11/08: each of these failed on the first cut ---------------------------------

def test_if_the_text_part_fails_the_carried_items_are_retried_as_text_not_lost():
    h = House(NIGHT)
    h.desk.reach_out(ReachEvent(source="kael_note", topic="kael", headline="Kael finished the backup check.",
                                detail="All green."))
    h.at(IN_OFFICE)
    real_text, fails = h.desk.deliverer.text, {"n": 1}

    def flaky_text(surface, body, silent):
        if fails["n"]:
            fails["n"] -= 1
            raise ConnectionError("discord down")
        real_text(surface, body, silent)

    h.desk.deliverer.text = flaky_text
    assert h.brief.tick() == "sent" and len(h.world.spoken) == 1 and h.world.texts == []
    h.at(IN_OFFICE + timedelta(minutes=6))
    assert h.desk.tick()["copied"] == 1
    (surface, body, _), = h.world.texts
    assert body.startswith("Held for you overnight:") and "All green." in body   # the detail only
    assert h.desk.store.rows[2].dm_at is not None and h.desk.store.rows[1].status == "folded"


def test_a_brief_composed_across_the_cutoff_is_never_spoken():
    h = House(datetime(2026, 11, 10, 12, 59, 30, tzinfo=ET))
    h.desk.store.add(ReachEvent(source="kael_note", topic="kael", headline="Held."),
                     NIGHT, status="queued", reason="night", not_before=datetime(2026, 11, 10, 13, 0, tzinfo=ET))

    def slow(facts):
        h.clock["now"] = datetime(2026, 11, 10, 13, 0, 20, tzinfo=ET)   # her words took 50 s
        return "Morning!"

    h.brief.compose = slow
    assert h.brief.tick() == "dropped" and h.world.spoken == []
    assert h.desk.tick()["sent"] == 1 and h.world.spoken == [("Held.", "dev-office")]   # it goes on its own


def test_the_moment_of_speech_checks_the_cutoff_too():
    h = House(datetime(2026, 11, 10, 12, 59, 50, tzinfo=ET))
    locate = h.desk.deliverer.locate

    def slow_locate():
        h.clock["now"] = datetime(2026, 11, 10, 13, 0, 5, tzinfo=ET)    # crossed while finding him
        return locate()

    h.desk.deliverer.locate = slow_locate
    assert h.brief.tick() == "dropped" and h.world.spoken == []


def test_what_it_carries_cant_also_go_on_its_own_while_she_composes():
    h = House(datetime(2026, 11, 10, 9, 30, tzinfo=ET))
    h.meeting = datetime(2026, 11, 10, 9, 40, tzinfo=ET)
    assert h.desk.reach_out(ReachEvent(source="kael_note", topic="kael", headline="Kael finished.")) == "queued"
    h.meeting = None

    def compose(facts):
        h.clock["now"] = datetime(2026, 11, 10, 9, 45, tzinfo=ET)   # the meeting's window opened meanwhile
        assert h.desk.tick()["sent"] == 0                             # reserved: the watcher leaves it
        return "Morning!"

    h.brief.compose = compose
    assert h.brief.tick() == "sent"
    assert h.world.spoken == [("Morning!", "dev-office")] and h.desk.store.rows[1].status == "folded"


def test_a_failed_fold_write_is_retried_and_the_item_never_goes_twice():
    class FlakySettle(MemoryReachStore):
        fail = 1

        def settle(self, reach_id, now, *, folds=(), **fields):
            if folds and self.fail:
                self.fail -= 1
                raise ConnectionError("synthetic commit failure")
            return super().settle(reach_id, now, folds=folds, **fields)

    h = House(NIGHT, store=FlakySettle())
    h.desk.reach_out(ReachEvent(source="kael_note", topic="kael", headline="Kael finished."))
    h.at(IN_OFFICE)
    assert h.brief.tick() == "sent" and h.desk.store.rows[1].status == "queued"   # the write failed
    h.at(IN_OFFICE + timedelta(minutes=1))
    h.desk.tick()
    assert h.desk.store.rows[1].status == "folded" and h.desk.store.rows[2].status == "sent"
    h.at(datetime(2026, 11, 10, 13, 1, tzinfo=ET))
    assert h.desk.tick()["sent"] == 0 and len(h.world.spoken) == 1


def test_it_doesnt_carry_what_expired_or_what_he_muted_since():
    h = House(NIGHT)
    h.desk.reach_out(ReachEvent(source="kael_note", topic="kael", headline="Stale.",
                                expires_at=datetime(2026, 11, 10, 7, 0, tzinfo=ET)))
    h.desk.reach_out(ReachEvent(source="weather", topic="weather", headline="Weather note."))
    h.desk.reach_out(ReachEvent(source="kael_note", topic="kael", headline="Fresh."))
    h.desk.store.set_prefs(Prefs(budget=3, muted=["weather"]), NIGHT, "mute weather", "")
    h.at(IN_OFFICE)
    assert h.brief.tick() == "sent"
    assert [r.headline for r in h.composed[0].held] == ["Fresh."]
    h.at(datetime(2026, 11, 10, 13, 1, tzinfo=ET))
    h.desk.tick()
    statuses = {r.headline: r.status for r in h.desk.store.rows.values()}
    assert statuses["Stale."] == "dropped" and statuses["Weather note."] == "dropped" and statuses["Fresh."] == "folded"
    assert len(h.world.spoken) == 1


def test_what_a_brief_carried_never_goes_again_while_its_record_keeps_failing():
    class DownStore(MemoryReachStore):
        down = True

        def settle(self, reach_id, now, *, folds=(), **fields):
            if folds and self.down:
                raise ConnectionError("NAS down")
            return super().settle(reach_id, now, folds=folds, **fields)

    h = House(NIGHT, store=DownStore())
    h.desk.reach_out(ReachEvent(source="kael_note", topic="kael", headline="Kael finished."))
    h.at(IN_OFFICE)
    assert h.brief.tick() == "sent"
    h.at(datetime(2026, 11, 10, 13, 1, tzinfo=ET))           # through the cutoff, still failing
    assert h.desk.tick()["sent"] == 0 and len(h.world.spoken) == 1
    h.desk.store.down = False
    h.at(datetime(2026, 11, 10, 13, 2, tzinfo=ET))
    h.desk.tick()
    assert h.desk.store.rows[1].status == "folded" and h.desk.unrecorded == {} and len(h.world.spoken) == 1
