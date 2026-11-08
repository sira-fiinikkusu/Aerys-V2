"""The reach-out desk (board #176, Stage 2): every message she starts herself goes through
one desk with its checks in code. The owner's D4/D6/D7 and the meeting rule, 2026-11-08."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage

from aerys_v2.kael_line import kael_note_for, owner_notice_for, parse_room_devices, reach_desk_for, ticket_relay_for
from aerys_v2.reach import (
    GIVE_UP_AFTER,
    MISSED_PREFIX,
    Decision,
    Deliverer,
    MemoryReachStore,
    Prefs,
    ReachDesk,
    ReachEvent,
    Where,
    decide,
    next_local,
)
from aerys_v2.tickets import Ticket
from aerys_v2.tools.reach_prefs import build_reach_prefs_tool

ET = timezone(timedelta(hours=-5))                      # EST on these dates
DAY = datetime(2026, 11, 9, 14, 0, tzinfo=ET)            # Monday 2 PM
NIGHT = datetime(2026, 11, 9, 23, 0, tzinfo=ET)          # Monday 11 PM


def ev(**kw):
    base = dict(source="kael_note", topic="kael", headline="Kael finished the backup check.")
    base.update(kw)
    return ReachEvent(**base)


def d(event, now=DAY, *, prefs=None, sent_today=0, duplicate=False, quiet=False, meeting_until=None):
    return decide(event, now, prefs=prefs or Prefs(), sent_today=sent_today, duplicate=duplicate,
                  quiet=quiet, meeting_until=meeting_until, morning_hour=9)


# --- the checks, in order ------------------------------------------------------------------

def test_a_repeat_is_dropped():
    assert d(ev(dedupe_key="k"), duplicate=True).action == "drop"


def test_safety_always_goes_even_at_night_in_a_meeting_over_budget_and_muted():
    out = d(ev(urgency="safety", topic="weather"), NIGHT, prefs=Prefs(budget=0, muted=["weather"]),
            sent_today=9, quiet=True, meeting_until=DAY)
    assert out.action == "send" and out.voice == "ok"


def test_a_muted_topic_is_dropped_but_an_answer_he_asked_for_never_is():
    assert d(ev(topic="Weather"), prefs=Prefs(muted=["weather"])).action == "drop"
    assert d(ev(topic="kael", owed=True), prefs=Prefs(muted=["kael"])).action == "send"


def test_at_night_routine_waits_for_the_morning_and_owed_or_timely_goes_as_quiet_text():
    routine = d(ev(), NIGHT, quiet=True)
    assert routine.action == "queue" and routine.not_before == datetime(2026, 11, 10, 9, 0, tzinfo=ET)
    for e in (ev(owed=True), ev(urgency="timely")):
        out = d(e, NIGHT, quiet=True)
        assert out.action == "send" and out.silent and out.voice == "reroute"


def test_in_a_meeting_routine_waits_timely_goes_quiet_and_owed_waits_only_if_spoken():
    end = DAY + timedelta(minutes=40)
    assert d(ev(), meeting_until=end).action == "queue" and d(ev(), meeting_until=end).not_before == end
    timely = d(ev(urgency="timely"), meeting_until=end)
    assert timely.action == "send" and timely.voice == "reroute" and timely.silent
    owed = d(ev(owed=True), meeting_until=end)
    assert owed.action == "send" and owed.voice == "wait" and owed.not_before == end and owed.silent


def test_over_budget_routine_is_dropped_timely_and_owed_still_go():
    prefs = Prefs(budget=3)
    assert d(ev(), prefs=prefs, sent_today=3).action == "drop"
    assert d(ev(urgency="timely"), prefs=prefs, sent_today=3).action == "send"
    assert d(ev(owed=True), prefs=prefs, sent_today=30).action == "send"
    assert d(ev(), prefs=prefs, sent_today=2).action == "send"


def test_the_next_morning_is_built_in_local_time():
    assert next_local(datetime(2026, 11, 9, 7, 0, tzinfo=ET), 9) == datetime(2026, 11, 9, 9, 0, tzinfo=ET)
    from zoneinfo import ZoneInfo

    ny = ZoneInfo("America/New_York")      # 3/8/2026 springs forward: still 9 AM on the wall
    assert next_local(datetime(2026, 3, 7, 23, 0, tzinfo=ny), 9).astimezone(ny).hour == 9
    assert next_local(datetime(2026, 11, 1, 1, 30, tzinfo=ny), 9).astimezone(ny).hour == 9   # falls back


# --- the desk, end to end ------------------------------------------------------------------

class World:
    """Where he is and what reached him."""

    def __init__(self, where=Where("room", "dev-office", "office")):
        self.where, self.spoken, self.texts, self.busy, self.fail, self.said = where, [], [], False, 0, []
        self.replied = False

    def deliverer(self):
        def speak(text, device):
            if self.fail:
                self.fail -= 1
                raise ConnectionError("satellite down")
            self.spoken.append((text, device))

        return Deliverer(locate=lambda: self.where, speak=speak,
                         text=lambda surface, body, silent: self.texts.append((surface, body, silent)),
                         busy=lambda device: self.busy)


def desk_for(world, *, now, quiet=False, meeting=None, store=None):
    clock = {"now": now}
    desk = ReachDesk(store or MemoryReachStore(), world.deliverer(), budget_default=3,
                     quiet=lambda t: quiet if not callable(quiet) else quiet(t), morning_hour=9,
                     meeting_hold=lambda: meeting["until"] if isinstance(meeting, dict) else meeting,
                     replied_since=lambda when: world.replied,
                     said=lambda row, body: world.said.append(body), clock=lambda: clock["now"])
    return desk, clock


def test_by_day_she_speaks_the_headline_where_he_is_and_texts_the_detail():
    world = World()
    desk, _ = desk_for(world, now=DAY)
    assert desk.reach_out(ev(headline="Backup's green.", detail="All three databases, 03:00 run.")) == "sent"
    assert world.spoken == [("Backup's green.", "dev-office")]
    assert world.texts == [("discord", "Backup's green.\n\nAll three databases, 03:00 run.", False)]
    row = desk.store.rows[1]
    assert row.status == "sent" and row.spoken_at and row.dm_at and row.surface == "room:office"
    assert world.said == ["Backup's green.\n\nAll three databases, 03:00 run."]


def test_at_night_an_owed_answer_never_speaks_into_a_room_it_goes_as_quiet_text():
    world = World()
    desk, _ = desk_for(world, now=NIGHT, quiet=True)
    assert desk.reach_out(ev(owed=True, headline="Kael says yes.")) == "sent"
    assert world.spoken == [] and world.texts == [("discord", "Kael says yes.", True)]


def test_a_routine_note_queued_at_night_goes_in_the_morning():
    world = World()
    desk, clock = desk_for(world, now=NIGHT, quiet=lambda t: t.astimezone(ET).hour >= 21 or t.astimezone(ET).hour < 9)
    assert desk.reach_out(ev()) == "queued"
    clock["now"] = datetime(2026, 11, 10, 8, 59, tzinfo=ET)
    assert desk.tick()["sent"] == 0
    clock["now"] = datetime(2026, 11, 10, 9, 1, tzinfo=ET)
    assert desk.tick()["sent"] == 1 and len(world.spoken) == 1


def test_an_owed_answer_waits_out_a_meeting_when_she_would_speak_it():
    world, meeting = World(), {"until": DAY + timedelta(minutes=30)}
    desk, clock = desk_for(world, now=DAY, meeting=meeting)
    assert desk.reach_out(ev(owed=True)) == "queued" and world.spoken == []
    meeting["until"] = None
    clock["now"] = DAY + timedelta(minutes=31)
    assert desk.tick()["sent"] == 1 and len(world.spoken) == 1


def test_an_owed_answer_in_a_meeting_goes_as_quiet_text_when_she_would_have_texted_anyway():
    world = World(where=Where("discord"))
    desk, _ = desk_for(world, now=DAY, meeting=DAY + timedelta(minutes=30))
    assert desk.reach_out(ev(owed=True)) == "sent" and world.texts[0][2] is True


def test_never_over_a_conversation_in_progress():
    world = World()
    world.busy = True
    desk, clock = desk_for(world, now=DAY)
    assert desk.reach_out(ev()) == "queued"
    world.busy = False
    clock["now"] = DAY + timedelta(minutes=1)
    assert desk.tick()["sent"] == 1


def test_a_failed_delivery_retries_and_an_owed_answer_is_never_given_up():
    world = World()
    world.fail = 100
    desk, clock = desk_for(world, now=DAY)
    desk.reach_out(ev(owed=True))
    desk.reach_out(ev(headline="routine note"))
    for _ in range(GIVE_UP_AFTER + 2):
        clock["now"] += timedelta(minutes=31)
        desk.tick()
    rows = desk.store.rows
    assert rows[1].status == "queued" and rows[1].attempts > GIVE_UP_AFTER   # owed: still trying
    assert rows[2].status == "failed"


def test_a_spoken_message_he_doesnt_answer_gets_a_text_copy_and_one_he_answers_doesnt():
    world = World()
    desk, clock = desk_for(world, now=DAY)
    desk.reach_out(ev(headline="One"))
    clock["now"] += timedelta(minutes=4)
    assert desk.tick()["copied"] == 0
    clock["now"] += timedelta(minutes=2)
    assert desk.tick()["copied"] == 1
    assert world.texts[-1][1] == f"{MISSED_PREFIX} One"
    assert desk.tick()["copied"] == 0                     # once
    world.replied = True
    desk.reach_out(ev(headline="Two"))
    clock["now"] += timedelta(minutes=6)
    assert desk.tick() == {"sent": 0, "queued": 0, "dropped": 0, "duplicate": 0, "copied": 0, "heard": 1}


def test_the_budget_counts_only_what_she_started_herself():
    world = World(where=Where("discord"))
    desk, _ = desk_for(world, now=DAY)
    for i in range(3):
        assert desk.reach_out(ev(headline=f"note {i}")) == "sent"
    assert desk.reach_out(ev(owed=True, headline="answer")) == "sent"
    assert desk.reach_out(ev(headline="one too many")) == "dropped"
    assert "budget" in desk.store.rows[5].reason


def test_the_same_thing_twice_goes_once():
    world = World(where=Where("discord"))
    desk, _ = desk_for(world, now=DAY)
    assert desk.reach_out(ev(dedupe_key="ticket:1:x", owed=True)) == "sent"
    assert desk.reach_out(ev(dedupe_key="ticket:1:x", owed=True)) == "duplicate"   # heard, not resent


def test_a_reach_out_needs_words_and_a_known_urgency():
    desk, _ = desk_for(World(), now=DAY)
    with pytest.raises(ValueError):
        desk.reach_out(ev(headline="  "))
    with pytest.raises(ValueError):
        desk.reach_out(ev(urgency="whenever"))


# --- his words steer it --------------------------------------------------------------------

def test_he_steers_it_in_words_bounded_and_recorded():
    store = MemoryReachStore()
    tool = build_reach_prefs_tool(store, default_budget=3, clock=lambda: DAY)
    assert "up to 2" in tool.invoke({"action": "less", "his_words": "chill out"})
    for _ in range(10):
        tool.invoke({"action": "less"})
    assert "Already at the least" in tool.invoke({"action": "less"})
    assert store.prefs(3).budget == 0
    assert "won't bring up weather" in tool.invoke({"action": "mute", "topic": "Weather", "his_words": "stop with the weather"})
    assert "already muted" in tool.invoke({"action": "mute", "topic": "weather"})
    assert tool.invoke({"action": "mute", "topic": "storms"}).startswith("NOT DONE")
    assert "Muted: weather" in tool.invoke({"action": "show"})
    assert "again" in tool.invoke({"action": "unmute", "topic": "weather"})
    assert store.changes[0][1:] == ("budget 3 -> 2", "chill out")
    assert tool.invoke({"action": "louder"}).startswith("NOT DONE")


def test_a_desk_change_takes_effect_on_the_very_next_reach_out():
    world = World(where=Where("discord"))
    store = MemoryReachStore(prefs=Prefs(budget=3, muted=["kael"]))
    desk, _ = desk_for(world, now=DAY, store=store)
    assert desk.reach_out(ev(topic="kael")) == "dropped"


# --- the sources route through it ----------------------------------------------------------

class _Secret:
    def __init__(self, v):
        self.v = v

    def get_secret_value(self):
        return self.v


def owner_settings(**kw):
    base = dict(owner_person_id="6e6bcbed-0000-0000-0000-000000000000", database_url=None,
                memories_database_url=None, telegram_bot_token=None, discord_bot_token=_Secret("dc"),
                soul_file_path=__import__("pathlib").Path("/nonexistent/soul.md"))
    base.update(kw)
    return SimpleNamespace(**base)


class RecordingDesk:
    def __init__(self, outcome="queued"):
        self.events, self.outcome = [], outcome

    def reach_out(self, event):
        self.events.append(event)
        return self.outcome


def test_a_ticket_answer_goes_to_the_desk_as_owed_and_is_composed_once():
    calls = []

    class Model:
        def invoke(self, messages):
            calls.append(1)
            return AIMessage(content="Kael says the backup was green.")

    desk = RecordingDesk("queued")
    relay = ticket_relay_for(None, owner_settings(), model=Model(), desk=desk)
    answered = datetime(2026, 11, 9, 19, 0, tzinfo=timezone.utc)
    t = Ticket(id=4, thread_id="person:x", ask="backup?", for_owner=True, status="completed", answer="green",
               answered_at=answered)
    assert relay(t) is True and len(calls) == 1
    e = desk.events[0]
    assert e.owed and e.source == "ticket" and e.headline == "Kael says the backup was green."
    assert e.dedupe_key == f"ticket:4:{answered.isoformat()}"
    assert ticket_relay_for(None, owner_settings(), model=Model(), desk=RecordingDesk("dropped"))(t) is False


def test_the_still_open_notice_goes_to_the_desk_as_owed():
    desk = RecordingDesk("sent")
    notice = owner_notice_for(None, owner_settings(), desk=desk)
    t = Ticket(id=5, thread_id="person:x", ask="is it raining?", for_owner=True,
               created_at=datetime(2026, 11, 8, 15, 0, tzinfo=timezone.utc),
               waiting_since=datetime(2026, 11, 8, 15, 0, tzinfo=timezone.utc))
    assert notice(t, True) is True
    assert desk.events[0].owed and desk.events[0].source == "ticket_notice" and "T-5" in desk.events[0].headline


def test_the_kael_note_turn_goes_through_the_desk():
    desk = RecordingDesk("sent")

    class Graph:
        def update_state(self, *a, **k):
            pass

    note = kael_note_for(Graph(), owner_settings(), proactive_model=SimpleNamespace(invoke=lambda m: AIMessage(content="News!")),
                         proactive_lookup_chat_id=lambda s: "id", proactive_lookup_surface=lambda s: ("discord", None),
                         proactive_sync=True, desk=desk)
    note("person:6e6bcbed-0000-0000-0000-000000000000", "family news", True)
    assert [(e.source, e.topic, e.headline, e.owed) for e in desk.events] == [("kael_note", "kael", "News!", False)]


def test_no_database_or_owner_means_no_desk_and_rooms_parse():
    assert reach_desk_for(owner_settings(), None) is None
    assert parse_room_devices(" Office=dev1 , bedroom=dev2,,=x,kitchen=") == {"office": "dev1", "bedroom": "dev2"}


# --- Codex review 11/08 -------------------------------------------------------------------

def test_a_slow_delivery_that_crosses_nine_pm_doesnt_let_the_next_one_speak():
    spoken = []
    clock = {"now": datetime(2026, 11, 9, 20, 59, 59, tzinfo=ET)}

    def speak(text, device):
        spoken.append(clock["now"])
        clock["now"] += timedelta(seconds=2)           # this delivery runs past 21:00

    store = MemoryReachStore()
    desk = ReachDesk(store, Deliverer(locate=lambda: Where("room", "dev"), speak=speak,
                                      text=lambda *a: None),
                     quiet=lambda t: t.astimezone(ET).hour >= 21, clock=lambda: clock["now"])
    for i in range(2):
        store.add(ev(headline=f"q{i}"), clock["now"], status="queued", reason="", not_before=None)
    desk.tick()
    assert len(spoken) == 1 and store.rows[2].status == "queued"    # the second waits for morning


def test_a_delivery_whose_record_fails_is_never_delivered_twice():
    class FlakyMark(MemoryReachStore):
        fail = True

        def mark(self, *args, **kw):
            if self.fail:
                self.fail = False
                raise ConnectionError("synthetic commit failure")
            return super().mark(*args, **kw)

    texts = []
    store = FlakyMark()
    desk = ReachDesk(store, Deliverer(locate=lambda: Where("discord"), speak=lambda *a: None,
                                      text=lambda *a: texts.append(a)), clock=lambda: DAY)
    assert desk.reach_out(ev()) == "sent"              # delivered; the record failed quietly
    desk.tick()
    assert len(texts) == 1 and store.rows[1].status == "sent" and desk.unrecorded == {}


def test_a_ticket_relay_retried_after_the_desk_has_it_counts_as_heard():
    world = World(where=Where("discord"))
    desk, _ = desk_for(world, now=DAY)
    e = ev(owed=True, dedupe_key="ticket:9:x")
    assert desk.reach_out(e) == "sent"
    assert desk.reach_out(e) == "duplicate"
    relay = ticket_relay_for(None, owner_settings(), model=SimpleNamespace(invoke=lambda m: AIMessage(content="hi")),
                             desk=RecordingDesk("duplicate"))
    t = Ticket(id=9, thread_id="person:x", ask="?", for_owner=True, status="completed", answer="y",
               answered_at=DAY)
    assert relay(t) is True


def test_the_night_beginning_during_a_slow_lookup_stops_the_speech():
    clock = {"now": datetime(2026, 11, 9, 20, 59, 59, tzinfo=ET)}
    spoken = []

    def locate():
        clock["now"] += timedelta(seconds=2)            # a slow lookup crosses 21:00
        return Where("room", "dev")

    store = MemoryReachStore()
    desk = ReachDesk(store, Deliverer(locate=locate, speak=lambda t, d: spoken.append(t), text=lambda *a: None),
                     quiet=lambda t: t.astimezone(ET).hour >= 21, clock=lambda: clock["now"])
    assert desk.reach_out(ev()) == "queued" and spoken == []
    assert store.rows[1].not_before.astimezone(ET).hour == 9            # waits for the morning


def test_a_failed_detail_text_never_repeats_the_speech_and_the_detail_still_arrives():
    spoken, texts = [], []
    fail = {"n": 1}

    def text(surface, body, silent):
        if fail["n"]:
            fail["n"] -= 1
            raise ConnectionError("discord down")
        texts.append(body)

    clock = {"now": DAY}
    store = MemoryReachStore()
    desk = ReachDesk(store, Deliverer(locate=lambda: Where("room", "dev"), speak=lambda t, d: spoken.append(t), text=text),
                     replied_since=lambda when: True, clock=lambda: clock["now"])
    assert desk.reach_out(ev(owed=True, headline="Head.", detail="The details.")) == "sent"
    clock["now"] += timedelta(minutes=6)
    desk.tick()
    desk.tick()
    assert spoken == ["Head."] and texts == ["The details."]


def test_a_follow_through_copy_whose_record_fails_is_sent_once():
    class FlakyCopyMark(MemoryReachStore):
        fail = True

        def mark(self, rid, now, **fields):
            if "dm_at" in fields and self.fail:
                self.fail = False
                raise ConnectionError("synthetic")
            return super().mark(rid, now, **fields)

    texts, clock = [], {"now": DAY}
    store = FlakyCopyMark()
    desk = ReachDesk(store, Deliverer(locate=lambda: Where("room", "dev"), speak=lambda *a: None,
                                      text=lambda s, b, q: texts.append(b)),
                     replied_since=lambda when: False, clock=lambda: clock["now"])
    desk.reach_out(ev(headline="Hi."))
    clock["now"] += timedelta(minutes=6)
    desk.tick()
    desk.tick()
    desk.tick()
    assert texts == [f"{MISSED_PREFIX} Hi."] and store.rows[1].dm_at is not None


def test_a_safety_warning_speaks_through_the_night_without_looping():
    spoken = []
    store = MemoryReachStore()
    desk = ReachDesk(store, Deliverer(locate=lambda: Where("room", "dev"), speak=lambda t, d: spoken.append(t),
                                      text=lambda *a: None),
                     quiet=lambda t: True, meeting_hold=lambda: DAY, clock=lambda: NIGHT)
    assert desk.reach_out(ev(urgency="safety", headline="Tornado warning for Rotonda.")) == "sent"
    assert spoken == ["Tornado warning for Rotonda."]


def test_a_detail_that_keeps_failing_is_still_tried_after_an_hour():
    texts, clock, down = [], {"now": DAY}, {"on": True}

    def text(surface, body, silent):
        if down["on"]:
            raise ConnectionError("discord down")
        texts.append(body)

    store = MemoryReachStore()
    desk = ReachDesk(store, Deliverer(locate=lambda: Where("room", "dev"), speak=lambda *a: None, text=text),
                     replied_since=lambda when: True, clock=lambda: clock["now"])
    desk.reach_out(ev(owed=True, headline="Head.", detail="Details."))
    clock["now"] += timedelta(hours=2)
    desk.tick()
    down["on"] = False
    clock["now"] += timedelta(hours=1)
    desk.tick()
    assert texts == ["Details."]


# --- the nap hold (the owner picked it 11/08: "2 works") -----------------------------------

def test_a_nap_holds_her_like_a_meeting_and_ends_when_he_talks_to_her():
    world = World()                                         # he's settled in the bedroom: a room with a speaker
    store = MemoryReachStore()
    tool = build_reach_prefs_tool(store, clock=lambda: DAY)
    assert "Holding" in tool.invoke({"action": "nap", "his_words": "gonna take a nap"})
    replied = {"v": False}
    clock = {"now": DAY + timedelta(minutes=5)}
    desk = ReachDesk(store, world.deliverer(), replied_since=lambda when: replied["v"] and when >= DAY,
                     clock=lambda: clock["now"])
    assert desk.reach_out(ev(headline="news")) == "queued" and world.spoken == []
    assert "nap" in store.rows[1].reason
    assert desk.reach_out(ev(urgency="timely", headline="the oven timer")) == "sent"
    assert world.texts[-1] == ("discord", "the oven timer", True)        # silent text, never spoken
    replied["v"] = True                                                  # he's up and talking to her
    clock["now"] += timedelta(minutes=3)
    desk.tick()
    assert world.spoken == [("news", "dev-office")]                      # well before the hour


def test_a_nap_ends_on_its_own_after_the_hour():
    world, store = World(), MemoryReachStore()
    build_reach_prefs_tool(store, clock=lambda: DAY).invoke({"action": "nap"})
    clock = {"now": DAY + timedelta(minutes=10)}
    desk = ReachDesk(store, world.deliverer(), replied_since=lambda when: False, clock=lambda: clock["now"])
    assert desk.reach_out(ev()) == "queued"
    clock["now"] = DAY + timedelta(minutes=61)
    assert desk.tick()["sent"] == 1


def test_the_nap_asks_for_turns_that_started_after_it_began():
    world, store = World(), MemoryReachStore()
    build_reach_prefs_tool(store, clock=lambda: DAY).invoke({"action": "nap"})
    asked = []
    desk = ReachDesk(store, world.deliverer(), replied_since=lambda when: asked.append(when) or False,
                     clock=lambda: DAY + timedelta(minutes=3))
    assert desk.reach_out(ev()) == "queued"
    assert asked and all(DAY < w <= DAY + timedelta(seconds=10) for w in asked)   # skew grace only


def test_a_safety_warning_still_speaks_during_a_nap():
    world, store = World(), MemoryReachStore()
    build_reach_prefs_tool(store, clock=lambda: DAY).invoke({"action": "nap"})
    desk = ReachDesk(store, world.deliverer(), replied_since=lambda when: False, clock=lambda: DAY + timedelta(minutes=5))
    assert desk.reach_out(ev(urgency="safety", headline="Tornado warning.")) == "sent"
    assert world.spoken == [("Tornado warning.", "dev-office")]


def test_awake_clears_it_and_the_length_is_bounded():
    store = MemoryReachStore()
    tool = build_reach_prefs_tool(store, clock=lambda: DAY)
    assert "180 minutes" in tool.invoke({"action": "nap", "minutes": 600})
    assert "off" in tool.invoke({"action": "awake", "his_words": "I'm up"})
    assert store.prefs(3).hold_until is None
    assert "no nap hold" in tool.invoke({"action": "awake"})
    tool.invoke({"action": "nap", "minutes": 30})
    tool.invoke({"action": "less"})                                      # other changes keep the hold
    assert store.prefs(3).hold_until == DAY + timedelta(minutes=30)
