"""Her evening heads-up (board #176, Stage 2): a sentence or three about tomorrow, before quiet
hours, in a room with a speaker that isn't the bedroom, and only when tomorrow earns it. The
owner on the morning brief, 2026-11-08: "personal and yet not noisy"."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from aerys_v2.brief import (
    EveningHeadsUp,
    Facts,
    Reading,
    evening_facts_text,
    evening_heads_up_for,
    plain_evening,
    worth_saying,
)
from aerys_v2.reach import MemoryReachStore, Prefs, ReachDesk, Where
from test_morning_brief import quiet, settings
from test_reach_desk import World

ET = timezone(timedelta(hours=-5))
EVENING = datetime(2026, 11, 9, 19, 30, tzinfo=ET)        # Monday evening
TOMORROW_EARLY = datetime(2026, 11, 10, 8, 30, tzinfo=ET)
KITCHEN = Where("room", "dev-kitchen", "kitchen")
BEDROOM = Where("room", "dev-bedroom", "bedroom")


class Evening:
    def __init__(self, now=EVENING):
        self.world = World(KITCHEN)
        self.clock = {"now": now}
        self.meeting = None
        self.reading = Reading(room="kitchen", confirmed=True)
        self.facts = Facts(weather="rain, high 75", events=["8:30 AM to 9:00 AM: a work meeting"],
                           first_at=TOMORROW_EARLY, wet=True)
        self.asked = []
        self.words = "Heads-up: you've got an early one tomorrow at eight thirty, and it'll be rainy."
        self.desk = ReachDesk(MemoryReachStore(), self.world.deliverer(), budget_default=3, quiet=quiet,
                              morning_hour=13, meeting_hold=lambda: self.meeting,
                              replied_since=lambda when: False, clock=lambda: self.clock["now"])
        self.heads_up = EveningHeadsUp(self.desk, read=lambda: self.reading, gather=self.gather,
                                       compose=lambda facts: self.words, clock=lambda: self.clock["now"])

    def gather(self, now, held, quiet_rows, day=None):
        self.asked.append(day)
        return self.facts


def test_tomorrow_said_where_he_is_once_an_evening():
    e = Evening()
    assert e.heads_up.tick() == "sent"
    assert e.world.spoken == [(e.words, "dev-kitchen")]
    assert e.asked == [datetime(2026, 11, 10).date()]               # it asks about TOMORROW
    row = e.desk.store.rows[1]
    assert row.brief and row.room == "kitchen" and row.dedupe_key == "evening:2026-11-09"
    assert row.expires_at == datetime(2026, 11, 9, 20, 55, tzinfo=ET)
    assert e.desk.store.sent_since(datetime(2026, 11, 9, 0, 0, tzinfo=ET)) == 0     # outside the budget
    e.clock["now"] += timedelta(minutes=20)
    assert e.heads_up.tick() == "done" and len(e.world.spoken) == 1


def test_a_quiet_tomorrow_gets_nothing_at_all():
    e = Evening()
    e.facts = Facts(weather="sunny, high 80", events=["11:00 AM to 11:30 AM: a work meeting"],
                    first_at=datetime(2026, 11, 10, 11, 0, tzinfo=ET))
    assert e.heads_up.tick() == "nothing worth saying" and e.world.spoken == []
    assert e.heads_up.tick() == "done" and e.asked == [datetime(2026, 11, 10).date()]   # asked once


def test_never_the_bedroom_and_not_on_his_phone_alone():
    e = Evening()
    e.reading = Reading(room="bedroom", confirmed=True)
    assert e.heads_up.tick() == "not in a room for it"
    e.reading = Reading(room="kitchen", confirmed=False)
    assert e.heads_up.tick() == "not in a room for it"
    assert EveningHeadsUp(e.desk, read=None, gather=None, compose=None,
                          rooms=("office", "bedroom")).rooms == ("office",)
    assert e.world.spoken == []


def test_only_from_seven_until_five_before_quiet_hours():
    for when in (datetime(2026, 11, 9, 18, 59, tzinfo=ET), datetime(2026, 11, 9, 20, 55, tzinfo=ET)):
        e = Evening(when)
        assert e.heads_up.tick() == "outside the window" and e.asked == []


def test_a_meeting_holds_it():
    e = Evening()
    e.meeting = EVENING + timedelta(minutes=30)
    assert e.heads_up.tick() == "held: meeting" and e.asked == []


def test_he_can_mute_it():
    e = Evening()
    e.desk.store.set_prefs(Prefs(budget=3, muted=["evening"]), EVENING, "mute evening", "")
    assert e.heads_up.tick() == "muted" and e.world.spoken == []


def test_it_never_runs_into_quiet_hours():
    e = Evening(datetime(2026, 11, 9, 20, 54, 30, tzinfo=ET))

    def slow(facts):
        e.clock["now"] = datetime(2026, 11, 9, 20, 55, 10, tzinfo=ET)
        return "Heads-up."

    e.heads_up.compose = slow
    assert e.heads_up.tick() == "dropped" and e.world.spoken == []


def test_if_her_words_fail_the_plain_version_goes():
    e = Evening()
    e.heads_up.compose = lambda facts: (_ for _ in ()).throw(RuntimeError("model down"))
    assert e.heads_up.tick() == "sent"
    assert e.world.spoken[0][0] == "Heads-up for tomorrow: a work meeting at 8:30 AM. Weather: rain, high 75."


def test_what_earns_a_heads_up():
    assert worth_saying(Facts(first_at=TOMORROW_EARLY))
    assert worth_saying(Facts(personal=1))
    assert worth_saying(Facts(wet=True))
    assert not worth_saying(Facts(first_at=datetime(2026, 11, 10, 10, 0, tzinfo=ET)))
    assert not worth_saying(Facts())
    text = evening_facts_text(Facts(weather="rain", events=["9:30 AM: dentist"]))
    assert "Tomorrow's weather: rain" in text and "- 9:30 AM: dentist" in text
    assert plain_evening(Facts(events=["all day: Megan's birthday"])) == "Heads-up for tomorrow: all day: Megan's birthday."


def test_the_gatherer_reads_tomorrow_by_date():
    from test_morning_brief import Resp

    from aerys_v2.brief import gatherer_for

    seen = {}

    def get(url, headers):
        seen.setdefault("urls", []).append(url)
        if "calendar.work" in url:
            return Resp([{"summary": "Standup", "start": {"dateTime": "2026-11-10T08:30:00-05:00"},
                          "end": {"dateTime": "2026-11-10T09:00:00-05:00"}}])
        return Resp([{"summary": "Dentist", "start": {"dateTime": "2026-11-10T14:00:00-05:00"},
                      "end": {"dateTime": "2026-11-10T15:00:00-05:00"}}])

    def post(url, headers, body):
        return Resp({"service_response": {"weather.home": {"forecast": [
            {"datetime": "2026-11-09T17:00:00+00:00", "condition": "sunny", "temperature": 80},
            {"datetime": "2026-11-10T17:00:00+00:00", "condition": "rainy", "temperature": 75, "templow": 66}]}}})

    facts = gatherer_for(settings(), get=get, post=post)(EVENING, [], [], day=datetime(2026, 11, 10).date())
    assert facts.weather == "rain, high 75, low 66" and facts.wet
    assert facts.events == ["8:30 AM to 9:00 AM: a work meeting", "2:00 PM to 3:00 PM: Dentist"]
    assert facts.first_at == TOMORROW_EARLY and facts.personal == 1
    assert all("start=2026-11-10T00%3A00%3A00-05%3A00" in u for u in seen["urls"])     # all of tomorrow
    assert worth_saying(facts)


def test_armed_only_with_a_speaker_in_one_of_its_rooms():
    desk = object()
    on = settings(evening_heads_up=True, evening_rooms="office,kitchen", evening_start_hour=19,
                  owner_quiet_start_hour=21)
    heads_up = evening_heads_up_for(on, desk)
    assert isinstance(heads_up, EveningHeadsUp) and heads_up.rooms == ("office", "kitchen")
    assert evening_heads_up_for(settings(evening_heads_up=False), desk) is None
    assert evening_heads_up_for(settings(evening_heads_up=True, evening_rooms="bedroom"), desk) is None
    assert evening_heads_up_for(on, None) is None


# --- Codex review 11/08: each of these failed on the first cut -----------------------------------

def test_an_ha_outage_is_a_gap_not_a_quiet_tomorrow():
    e = Evening()
    e.facts = Facts(complete=False)                  # nothing answered
    assert e.heads_up.tick() == "incomplete reading"
    e.facts = Facts(weather="rain, high 75", first_at=TOMORROW_EARLY, wet=True)
    assert e.heads_up.tick() == "sent" and len(e.world.spoken) == 1


def test_a_quiet_evening_is_remembered_across_a_restart():
    e = Evening()
    e.facts = Facts(weather="sunny", events=["11:00 AM to 11:30 AM: a work meeting"],
                    first_at=datetime(2026, 11, 10, 11, 0, tzinfo=ET))
    assert e.heads_up.tick() == "nothing worth saying"
    again = EveningHeadsUp(e.desk, read=lambda: e.reading, gather=e.gather, compose=lambda f: "x",
                           clock=lambda: e.clock["now"])
    assert again.tick() == "done" and e.asked == [datetime(2026, 11, 10).date()]
    assert e.desk.store.sent_since(datetime(2026, 11, 9, 0, 0, tzinfo=ET)) == 0


def _gather_with(forecast, calendar_ok=True):
    from test_morning_brief import Resp

    from aerys_v2.brief import gatherer_for

    def get(url, headers):
        if not calendar_ok:
            raise ConnectionError("calendar down")
        return Resp([{"summary": "Dentist", "start": {"dateTime": "2026-11-10T09:30:00-05:00"},
                      "end": {"dateTime": "2026-11-10T10:00:00-05:00"}}])

    def post(url, headers, body):
        return Resp({"service_response": {"weather.home": {"forecast": forecast}}})

    return gatherer_for(settings(), get=get, post=post)


def test_a_malformed_forecast_costs_only_the_weather():
    gather = _gather_with([{"condition": "sunny", "temperature": "warm"}])
    facts = gather(datetime(2026, 11, 10, 8, 20, tzinfo=ET), [], [])
    assert facts.weather == "" and "9:30 AM to 10:00 AM: Dentist" in facts.events and not facts.complete


def test_a_dated_forecast_without_today_is_no_weather_for_today():
    gather = _gather_with([{"datetime": "2026-11-11T17:00:00+00:00", "condition": "rainy", "temperature": 70}])
    facts = gather(datetime(2026, 11, 10, 8, 20, tzinfo=ET), [], [])
    assert facts.weather == "" and not facts.wet and not facts.complete


def test_a_calendar_that_cant_be_read_marks_the_reading_incomplete():
    gather = _gather_with([{"datetime": "2026-11-10T17:00:00+00:00", "condition": "sunny", "temperature": 80}],
                          calendar_ok=False)
    facts = gather(EVENING, [], [], day=datetime(2026, 11, 10).date())
    assert facts.weather == "sunny, high 80" and facts.events == [] and not facts.complete


def test_a_malformed_calendar_answer_is_a_gap_too():
    from test_morning_brief import Resp

    from aerys_v2.brief import gatherer_for

    for bad in ({"error": "nope"}, [42]):
        def get(url, headers, bad=bad):
            return Resp(bad) if "calendar.home" in url else Resp([])

        def post(url, headers, body):
            return Resp({"service_response": {"weather.home": {"forecast": [
                {"datetime": "2026-11-10T17:00:00+00:00", "condition": "sunny", "temperature": 80}]}}})

        facts = gatherer_for(settings(), get=get, post=post)(EVENING, [], [], day=datetime(2026, 11, 10).date())
        assert not facts.complete and facts.weather == "sunny, high 80"
