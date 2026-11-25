"""D91 (Chris, 11/22 17:11): "if she gives up maybe have it go through discord?"

11/22: home 16:20, in the living room (no speaker of hers) until 17:02; the 20-minute settle window
dropped the hello at 16:40 and nothing reached him. Now the window is the full hour
(WELCOME_HOME_SETTLE_MINUTES=60, #251), and a hello she gives up on (he never settled in time, or
something held her voice until the hour was out) goes to him as a Discord message, written for text.
It carries nothing (the morning brief and anything held keep waiting for his room) and stays quiet when
he's been greeted another way: the brief just said it or is still due to, or he's talked with her since
he got home. Never at night, never muted. Decided from durable records, so a restart decides the same
way (Codex review of 1e42a9c: three BLOCKERs, each reproduced below)."""

from __future__ import annotations

from datetime import datetime, timedelta

from aerys_v2.followups import FollowUps, MemoryFollowupStore
from aerys_v2.reach import ReachEvent
from aerys_v2.welcome import (
    ASKED,
    BRIEF_GREETS,
    GIVE_UP_GRACE,
    HELLO_FOR,
    JUST_GREETED,
    LOOKUP,
    NO_LOOKUP,
    TALKED,
    TEXT_HELLO_FOR,
    TEXT_SURFACE,
    WELCOME_PROMPT,
    WELCOME_PROMPT_TOOL_FREE,
    WELCOME_TEXT_PROMPT,
    WELCOME_TEXT_PROMPT_TOOL_FREE,
    HomeReading,
    WelcomeFacts,
    WelcomeHome,
    welcome_facts_text,
    welcome_home_for,
)
from test_welcome_home import EARLY_HOME, EARLY_LEFT, ET, HOME, MORNING_HOME, House, settings, utc, vet

GAVE_UP = HOME + HELLO_FOR + timedelta(minutes=1)
MORNING_LEFT = datetime(2026, 11, 10, 7, 30, tzinfo=ET)


class TextHouse(House):
    """The live shape: the hour-long window, her separate Discord writer, and him in the living room."""

    def __init__(self, now=HOME + timedelta(minutes=5), *, settle=60, **kw):
        self.settle = settle
        self.texted = []
        self.text_words = "Welcome back! How was the day out?"
        super().__init__(now, **kw)
        self.reading = HomeReading(home="home", since=utc(kw.get("home", HOME)), room="living room", confirmed=True)

    def make(self):
        return WelcomeHome(self.desk, read=lambda: self.reading, away_since=self.away_since, compose=self.compose,
                           compose_text=self.compose_text, visits=lambda left, arrived: list(self.appointments),
                           rooms=("office", "kitchen"), brief=self.brief, followups=self.fstore,
                           settle_minutes=self.settle, clock=lambda: self.clock["now"])

    def compose_text(self, facts):
        self.texted.append(facts)
        if isinstance(self.text_words, Exception):
            raise self.text_words
        return self.text_words

    def hellos(self):
        return [r for r in self.desk.store.rows.values() if r.source == "welcome"]


# --- when it goes -----------------------------------------------------------------------------

def test_never_settling_the_hour_ends_with_a_discord_hello():
    h = TextHouse()
    assert h.hello.tick() == "waiting to settle"
    assert h.at(HOME + timedelta(minutes=59)).hello.tick() == "waiting to settle"
    assert h.at(GAVE_UP).hello.tick() == "by text: sent"
    assert h.world.spoken == [] and h.world.texts == [(TEXT_SURFACE, h.text_words, False)]
    (row,) = h.hellos()
    assert row.status == "sent" and row.text_only and row.text_to == "discord" and row.room is None and row.brief
    assert row.expires_at == utc(GAVE_UP) + TEXT_HELLO_FOR and row.folds == ()
    assert h.composed == []                                   # the spoken writer never ran
    (facts,) = h.texted
    assert facts.home_for == HELLO_FOR + timedelta(minutes=1) and facts.away == timedelta(minutes=90)
    assert facts.room == ""
    # Once: this process is done with it, and a restart inside the grace finds it already decided.
    assert h.hello.tick() == "done" and h.make().tick() == "done" and len(h.world.texts) == 1


def test_a_shorter_window_gives_up_at_its_end_the_same_way():
    h = TextHouse(settle=20)
    assert h.hello.tick() == "waiting to settle"
    assert h.at(HOME + timedelta(minutes=21)).hello.tick() == "by text: sent"
    assert h.world.spoken == [] and len(h.world.texts) == 1


def test_held_the_whole_hour_in_his_room_she_gives_up_and_it_goes_by_discord():
    # Codex 1: a restart used to forget he'd been in her room. Whatever held her, if she never said it
    # and nothing else greeted him, the hour's end is a give-up like any other.
    h = TextHouse()
    h.reading.room = "office"
    h.meeting = utc(HOME + HELLO_FOR + timedelta(seconds=30))
    assert h.hello.tick() == "held: meeting"
    h.reading.room = "living room"
    assert h.at(HOME + timedelta(minutes=30)).hello.tick() == "waiting to settle"
    again = h.make()                                          # a restart mid-window
    h.meeting = None
    assert h.at(GAVE_UP).hello.tick() == "by text: sent" and again.tick() == "done"
    assert h.world.spoken == [] and len(h.world.texts) == 1


def test_a_restart_across_the_hour_still_sends_it_once():
    # Codex 2: waiting at minute 5, the brain restarts, the next process first looks at minute 61.
    h = TextHouse()
    assert h.hello.tick() == "waiting to settle"
    fresh = h.make()
    h.at(GAVE_UP)
    assert fresh.tick() == "by text: sent"
    assert h.history_reads == 2                               # the trip came from HA's history again
    assert h.make().tick() == "done" and len(h.world.texts) == 1


def test_past_the_grace_a_restart_lets_it_go():
    h = TextHouse(HOME + HELLO_FOR + GIVE_UP_GRACE + timedelta(minutes=1))
    assert h.hello.tick() == "too late"
    assert h.hellos() == [] and h.world.texts == [] and h.texted == []


def test_a_short_trip_found_after_a_restart_gets_nothing():
    h = TextHouse(GAVE_UP, left=HOME - timedelta(minutes=10))
    assert h.hello.tick() == "a short trip" and h.world.texts == []


# --- when it stays quiet ------------------------------------------------------------------------

def test_he_talked_with_her_since_he_got_home_nothing_goes():
    h = TextHouse()
    assert h.hello.tick() == "waiting to settle"
    h.world.replied = True
    assert h.at(GAVE_UP).hello.tick() == TALKED
    (row,) = h.hellos()
    assert row.status == "dropped" and row.reason == TALKED and h.world.texts == [] and h.texted == []


def test_the_brief_just_greeted_him_in_his_room_no_discord():
    # Codex 3: never settled until minute 61, he walks into the office, the brief greets him there
    # first; the hello must not also text him.
    h = TextHouse(MORNING_HOME + timedelta(minutes=5), left=MORNING_LEFT, home=MORNING_HOME, brief=True)
    assert h.hello.tick() == "waiting to settle"
    h.at(MORNING_HOME + HELLO_FOR + timedelta(minutes=1))
    assert h.brief.tick() == "sent" and h.world.spoken == [("Morning brief.", "dev-office")]
    assert h.hello.tick() == JUST_GREETED
    assert h.world.texts == [] and h.texted == []


def test_the_brief_still_due_greets_him_instead_one_greeting():
    h = TextHouse(MORNING_HOME + timedelta(minutes=5), left=MORNING_LEFT, home=MORNING_HOME, brief=True)
    assert h.hello.tick() == "waiting to settle"
    assert h.at(MORNING_HOME + HELLO_FOR + timedelta(minutes=1)).hello.tick() == BRIEF_GREETS
    assert h.world.texts == [] and h.texted == []
    # He reaches the office later: the brief goes on its own, said there, once.
    h.reading.room = "office"
    assert h.at(MORNING_HOME + timedelta(minutes=70)).brief.tick() == "sent"
    assert h.world.spoken == [("Morning brief.", "dev-office")]
    assert not [r for r in h.desk.store.rows.values() if r.dedupe_key == "brief:2026-11-10" and r.status == "folded"]


def test_given_up_into_the_night_it_goes_as_quiet_text():
    # D96 (11/24) replaced "never at night": the text still goes after hours, quietly, never aloud.
    home = datetime(2026, 11, 10, 20, 10, tzinfo=ET)
    h = TextHouse(home + timedelta(minutes=5), left=home - timedelta(hours=3), home=home)
    assert h.hello.tick() == "waiting to settle"
    assert h.at(home + timedelta(minutes=65)).hello.tick() == "by text: sent"     # 21:15, past the hour
    (row,) = h.hellos()
    assert row.status == "sent" and row.text_only
    assert h.world.spoken == [] and len(h.world.texts) == 1 and h.world.texts[0][2] is True


def test_before_nine_never_settled_with_the_brief_due_no_discord_and_the_brief_goes_alone():
    h = TextHouse(EARLY_HOME + timedelta(minutes=1), left=EARLY_LEFT, home=EARLY_HOME, brief=True, settle=20)
    assert h.hello.tick() == "waiting to settle"
    assert h.at(EARLY_HOME + timedelta(minutes=21)).hello.tick() == BRIEF_GREETS
    assert h.world.texts == [] and h.texted == []
    assert h.at(EARLY_HOME + timedelta(minutes=25)).brief.tick() == "sent"
    assert h.world.spoken == [("Morning brief.", "dev-office")]


def test_muted_after_he_got_home_nothing_goes():
    from aerys_v2.tools.reach_prefs import build_reach_prefs_tool

    h = TextHouse()
    assert h.hello.tick() == "waiting to settle"
    tool = build_reach_prefs_tool(h.desk.store, clock=lambda: utc(HOME + timedelta(minutes=10)))
    assert tool.invoke({"action": "mute", "topic": "welcome", "his_words": "don't greet me"}).startswith("Changed")
    assert h.at(GAVE_UP).hello.tick() == "you muted welcome"
    (row,) = h.hellos()
    assert row.status == "dropped" and h.world.texts == [] and h.texted == []


def test_a_failed_check_waits_for_the_next_pass_never_guesses():
    h = TextHouse()
    assert h.hello.tick() == "waiting to settle"
    calls = {"n": 0}

    def flaky(when):
        calls["n"] += 1
        if calls["n"] == 1:
            raise ConnectionError("db blip")
        return False

    h.desk.replied_since = flaky
    assert h.at(GAVE_UP).hello.tick() == "waiting: couldn't check whether he's been greeted"
    assert h.hellos() == [] and h.world.texts == []
    assert h.at(GAVE_UP + timedelta(minutes=1)).hello.tick() == "by text: sent"


# --- what it carries and says -------------------------------------------------------------------

def test_what_came_in_while_he_was_out_keeps_waiting_for_his_room():
    h = TextHouse()
    came_in = h.desk.store.add(ReachEvent(source="kael_note", topic="kael", headline="Backup's green.", room="office"),
                               utc(HOME - timedelta(minutes=30)), status="queued", reason="waits for him in the office",
                               not_before=None)
    assert h.hello.tick() == "waiting to settle"
    assert h.at(GAVE_UP).hello.tick() == "by text: sent"
    assert h.texted[0].held == [] and came_in.status == "queued"
    (row,) = h.hellos()
    assert row.folds == () and row.detail == ""
    # The spoken hello would have folded it (the control: the item is one a hello carries).
    spoken = House(HOME + timedelta(minutes=5))
    item = spoken.desk.store.add(ReachEvent(source="kael_note", topic="kael", headline="Backup's green.", room="office"),
                                 utc(HOME - timedelta(minutes=30)), status="queued", reason="waits for him in the office",
                                 not_before=None)
    assert spoken.hello.tick() == "sent" and item.status == "folded"


def test_after_the_brief_went_this_morning_an_afternoon_discord_hello_never_touches_it():
    h = TextHouse(brief=True)
    h.desk.store.add(ReachEvent(source="brief", topic="brief", headline="Morning.", brief=True,
                                dedupe_key="brief:2026-11-10"), utc(datetime(2026, 11, 10, 9, 0, tzinfo=ET)),
                     status="sent", reason="", not_before=None)
    assert h.hello.tick() == "waiting to settle"
    assert h.at(GAVE_UP).hello.tick() == "by text: sent"
    assert h.texted[0].brief is None
    assert [r.status for r in h.desk.store.rows.values() if r.dedupe_key == "brief:2026-11-10"] == ["sent"]


def test_the_discord_hello_asks_about_the_appointment_once():
    fs = MemoryFollowupStore()
    h = TextHouse(followups=fs)
    h.appointments = [vet()]
    assert h.hello.tick() == "waiting to settle"
    assert h.at(GAVE_UP).hello.tick() == "by text: sent"
    assert h.texted[0].visits[0].ask and h.texted[0].visits[0].title == "the vet"
    f = fs.by_key(vet().key)
    assert f.status == "raised" and f.reason == ASKED
    runner = FollowUps(h.desk, fs, say_asked=lambda f: "x", say_appointment=lambda a: "How'd the vet go?",
                       appointments=lambda now: [vet()], clock=lambda: utc(GAVE_UP + timedelta(minutes=10)))
    assert runner.tick()["raised"] == 0


def test_her_text_words_fail_the_plain_one_goes_by_discord():
    h = TextHouse()
    h.text_words = RuntimeError("model down")
    assert h.hello.tick() == "waiting to settle"
    assert h.at(GAVE_UP).hello.tick() == "by text: sent"
    assert h.world.texts[0][:2] == (TEXT_SURFACE, "Welcome home. Get up to anything good while you were out?")


def test_a_short_meeting_at_give_up_holds_the_message_until_it_ends():
    h = TextHouse()
    assert h.hello.tick() == "waiting to settle"
    h.meeting = utc(GAVE_UP + timedelta(minutes=10))
    assert h.at(GAVE_UP).hello.tick() == "by text: queued"
    h.meeting = None
    h.at(GAVE_UP + timedelta(minutes=21)).desk.tick()
    (row,) = h.hellos()
    assert row.status == "sent" and h.world.texts == [(TEXT_SURFACE, h.text_words, False)] and h.world.spoken == []


def test_a_long_meeting_at_give_up_and_the_message_never_goes_past_its_half_hour():
    h = TextHouse()
    assert h.hello.tick() == "waiting to settle"
    end = GAVE_UP + timedelta(hours=2)
    h.meeting = utc(end)
    assert h.at(GAVE_UP).hello.tick() == "by text: queued"
    h.meeting = None
    h.at(end + timedelta(minutes=11)).desk.tick()
    (row,) = h.hellos()
    assert row.status == "dropped" and h.world.texts == [] and h.world.spoken == []


def test_the_text_prompt_and_facts_say_it_is_a_discord_message():
    for spoken, text in ((WELCOME_PROMPT, WELCOME_TEXT_PROMPT), (WELCOME_PROMPT_TOOL_FREE, WELCOME_TEXT_PROMPT_TOOL_FREE)):
        assert text != spoken and "Discord message" in text and "out loud" not in text
        assert text.endswith("Reply with ONLY the message.")
    assert LOOKUP in WELCOME_TEXT_PROMPT and NO_LOOKUP in WELCOME_TEXT_PROMPT_TOOL_FREE
    line = welcome_facts_text(WelcomeFacts(away=timedelta(minutes=90), home_for=timedelta(minutes=61)))
    assert "got home about an hour ago" in line and "Discord message" in line and "he's in the" not in line
    spoken_line = welcome_facts_text(WelcomeFacts(away=timedelta(minutes=90), room="office"))
    assert "he's in the office now" in spoken_line and "Discord" not in spoken_line


def test_the_factory_arms_a_separate_text_writer():
    hello = welcome_home_for(settings(), object())
    assert callable(hello.compose_text) and hello.compose_text is not hello.compose


def test_a_brief_on_another_reading_never_makes_the_hello_wait_past_its_grace():
    # The brief waits only while the hello waits to settle (greeting(): settle_for), never through the
    # give-up: past the hour the brief is free to go on its own.
    h = TextHouse(MORNING_HOME + timedelta(minutes=5), left=MORNING_LEFT, home=MORNING_HOME, brief=True)
    assert h.hello.tick() == "waiting to settle"
    assert h.hello.greeting(utc(MORNING_HOME + timedelta(minutes=59))) is True
    assert h.hello.greeting(utc(MORNING_HOME + HELLO_FOR + timedelta(minutes=1))) is False


def test_even_called_with_the_brief_due_a_discord_hello_never_takes_or_carries_it():
    # Defence in depth: _give_up stays quiet while the brief is due, but _say(by_text=True) itself never
    # touches the brief either (a brief that came due between the check and the send stays its own).
    h = TextHouse(MORNING_HOME + timedelta(minutes=5), left=MORNING_LEFT, home=MORNING_HOME, brief=True)
    waiting = h.desk.store.add(ReachEvent(source="brief", topic="brief", headline="Morning.", brief=True,
                                          dedupe_key="brief:2026-11-10", room="office"),
                               utc(MORNING_HOME - timedelta(minutes=20)), status="queued", reason="waits for him in the office",
                               not_before=None)
    key = f"welcome:{utc(MORNING_HOME).isoformat(timespec='seconds')}"
    now = utc(MORNING_HOME + HELLO_FOR + timedelta(minutes=1))
    h.at(now)
    assert h.hello._say(key, utc(MORNING_HOME), utc(MORNING_LEFT), "", now, by_text=True) == "sent"
    assert waiting.status == "queued" and h.texted[0].brief is None
    assert [r.status for r in h.desk.store.rows.values() if r.dedupe_key == "brief:2026-11-10"] == ["queued"]


def test_even_called_with_the_brief_due_and_not_yet_queued_a_discord_hello_never_carries_it():
    h = TextHouse(MORNING_HOME + timedelta(minutes=5), left=MORNING_LEFT, home=MORNING_HOME, brief=True)
    key = f"welcome:{utc(MORNING_HOME).isoformat(timespec='seconds')}"
    now = utc(MORNING_HOME + HELLO_FOR + timedelta(minutes=1))
    h.at(now)
    assert h.hello._brief_due(now, MORNING_HOME.date())               # the control: it is due
    assert h.hello._say(key, utc(MORNING_HOME), utc(MORNING_LEFT), "", now, by_text=True) == "sent"
    assert h.texted[0].brief is None
    assert not [r for r in h.desk.store.rows.values() if r.dedupe_key == "brief:2026-11-10"]   # no marker


def test_a_brief_queued_for_his_room_greets_him_instead_one_greeting():
    # The brief reached the desk (he passed through the office while her voice was held) and now waits
    # there for him: queued, not yet said. The hello stays quiet; the brief greets him when he's back.
    h = TextHouse(MORNING_HOME + timedelta(minutes=5), left=MORNING_LEFT, home=MORNING_HOME, brief=True)
    assert h.hello.tick() == "waiting to settle"
    h.desk.store.add(ReachEvent(source="brief", topic="brief", headline="Morning.", brief=True,
                                dedupe_key="brief:2026-11-10", room="office"),
                     utc(MORNING_HOME + timedelta(minutes=40)), status="queued", reason="waits for him in the office",
                     not_before=None)
    assert h.at(MORNING_HOME + HELLO_FOR + timedelta(minutes=1)).hello.tick() == BRIEF_GREETS
    assert h.world.texts == [] and h.texted == []


# --- Codex review of c288eee: the brief can't greet him between the checks and the send ----------

def test_a_brief_delivered_between_reads_still_counts_one_read_decides():
    # The desk delivers the waiting brief (queued -> sent) right after the hello's deciding look reads it
    # (the one under the lock). Two reads there would see "not said yet" then "not waiting any more" and
    # text him too.
    import threading

    h = TextHouse(MORNING_HOME + timedelta(minutes=5), left=MORNING_LEFT, home=MORNING_HOME, brief=True)
    assert h.hello.tick() == "waiting to settle"
    lock = threading.Lock()
    h.hello.lock = lock
    real = h.desk.store.rows_for
    state = {"flipped": False, "locked_reads": 0, "pending": None}

    def delivered_after_the_deciding_read(key):
        rows = real(key)
        if key == "brief:2026-11-10" and lock.locked():
            state["locked_reads"] += 1
            if not state["flipped"] and state["pending"] is not None:
                snapshot = [type(r)(**vars(r)) for r in rows]
                b = state["pending"]
                b.status, b.delivered_at = "sent", utc(MORNING_HOME + HELLO_FOR + timedelta(seconds=50))
                state["flipped"] = True
                return snapshot
        return rows

    h.desk.store.rows_for = delivered_after_the_deciding_read
    real_stops = h.hello._stops

    def stops_then_queue(left, arrived):
        # The brief goes to the desk while the hello names the stops (outside the lock).
        state["pending"] = h.desk.store.add(ReachEvent(source="brief", topic="brief", headline="Morning.", brief=True,
                                                       dedupe_key="brief:2026-11-10", room="office"),
                                            utc(MORNING_HOME + HELLO_FOR), status="queued",
                                            reason="waits for him in the office", not_before=None)
        return real_stops(left, arrived)

    h.hello._stops = stops_then_queue
    h.hello.brief.done = MORNING_HOME.date()      # not due on its own: only the queued row says it waits
    assert h.at(MORNING_HOME + HELLO_FOR + timedelta(minutes=1)).hello.tick() == BRIEF_GREETS
    assert state["flipped"] and state["locked_reads"] == 1 and h.world.texts == [] and h.texted == []


def test_the_brief_lock_is_held_from_the_deciding_check_to_the_send():
    import threading

    h = TextHouse(brief=True)
    assert h.hello.tick() == "waiting to settle"
    lock = threading.Lock()
    h.hello.lock = lock
    held = []
    real_rows, real_reach, real_stops = h.desk.store.rows_for, h.desk.reach_out, h.hello._stops
    h.desk.store.rows_for = lambda key: held.append(("rows", lock.locked())) or real_rows(key)
    h.desk.reach_out = lambda ev: held.append(("send", lock.locked())) or real_reach(ev)
    h.hello._stops = lambda left, arrived: held.append(("stops", lock.locked())) or real_stops(left, arrived)
    assert h.at(GAVE_UP).hello.tick() == "by text: sent"
    # A first look unlocked (no look-ups for a quiet give-up), the stops named unlocked (D92: it can take
    # seconds), then the deciding look and the send under the lock.
    assert held == [("rows", False), ("stops", False), ("rows", True), ("send", True)]
