"""D91 (Chris, 11/22 17:11): "if she gives up maybe have it go through discord?"

11/22: home 16:20, in the living room (no speaker of hers) until 17:02; the 20-minute settle window
dropped the hello at 16:40 and nothing reached him. Now the window is the full hour
(WELCOME_HOME_SETTLE_MINUTES=60, #251), and a hello she gives up on because he never reached a room
with her speaker goes to him as a Discord message, written for text. It carries nothing (the morning
brief and anything held keep waiting for his room), never goes at night or when he muted it, and a
trip where he DID reach her speaker but she never got to say it is dropped as before."""

from __future__ import annotations

from datetime import datetime, timedelta

from aerys_v2.followups import FollowUps, MemoryFollowupStore
from aerys_v2.reach import ReachEvent
from aerys_v2.welcome import (
    ASKED,
    HELLO_FOR,
    LOOKUP,
    NO_LOOKUP,
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


def test_never_reaching_her_speaker_the_hour_ends_with_a_discord_hello():
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
    # Once: this process is done with it, and a restart past the hour doesn't send it again.
    assert h.hello.tick() == "done" and h.make().tick() == "too late" and len(h.world.texts) == 1


def test_a_shorter_window_gives_up_at_its_end_the_same_way():
    h = TextHouse(settle=20)
    assert h.hello.tick() == "waiting to settle"
    assert h.at(HOME + timedelta(minutes=21)).hello.tick() == "by text: sent"
    assert h.world.spoken == [] and len(h.world.texts) == 1


def test_he_reached_her_speaker_but_she_was_held_then_he_left_dropped_not_texted():
    h = TextHouse()
    h.reading.room = "office"
    h.meeting = utc(HOME + timedelta(hours=3))                # her voice is held the whole time
    assert h.hello.tick() == "held: meeting"
    h.reading.room = "living room"
    assert h.at(HOME + timedelta(minutes=30)).hello.tick() == "waiting to settle"
    assert h.at(GAVE_UP).hello.tick() == "too late"
    (row,) = h.hellos()
    assert row.status == "dropped" and row.reason == "never got to say it"
    assert h.world.texts == [] and h.texted == []


def test_given_up_into_the_night_nothing_goes():
    home = datetime(2026, 11, 10, 20, 10, tzinfo=ET)
    h = TextHouse(home + timedelta(minutes=5), left=home - timedelta(hours=3), home=home)
    assert h.hello.tick() == "waiting to settle"
    assert h.at(home + timedelta(minutes=65)).hello.tick() == "quiet hours"       # 21:15, past the hour
    (row,) = h.hellos()
    assert row.status == "dropped" and row.reason == "quiet hours"
    assert h.world.texts == [] and h.texted == []


def test_before_nine_never_settled_with_the_brief_due_no_discord_and_the_brief_goes_alone():
    h = TextHouse(EARLY_HOME + timedelta(minutes=1), left=EARLY_LEFT, home=EARLY_HOME, brief=True, settle=20)
    assert h.hello.tick() == "waiting to settle"
    assert h.at(EARLY_HOME + timedelta(minutes=21)).hello.tick() == "quiet hours"
    assert h.world.texts == [] and h.texted == []
    assert h.at(EARLY_HOME + timedelta(minutes=25)).brief.tick() == "sent"
    assert h.world.spoken == [("Morning brief.", "dev-office")]


def test_muted_after_he_got_home_nothing_goes():
    from aerys_v2.tools.reach_prefs import build_reach_prefs_tool

    h = TextHouse()
    assert h.hello.tick() == "waiting to settle"
    tool = build_reach_prefs_tool(h.desk.store, clock=lambda: utc(HOME + timedelta(minutes=10)))
    assert tool.invoke({"action": "mute", "topic": "welcome", "his_words": "don't greet me"}).startswith("Changed")
    assert h.at(GAVE_UP).hello.tick() == "muted"
    (row,) = h.hellos()
    assert row.status == "dropped" and h.world.texts == [] and h.texted == []


def test_a_discord_hello_carries_neither_the_brief_nor_what_was_held():
    h = TextHouse(MORNING_HOME + timedelta(minutes=5), left=datetime(2026, 11, 10, 7, 30, tzinfo=ET),
                  home=MORNING_HOME, brief=True, settle=20)
    overnight = h.desk.store.add(ReachEvent(source="kael_note", topic="kael", headline="Backup's green."),
                                 utc(datetime(2026, 11, 9, 23, 0, tzinfo=ET)), status="queued", reason="night",
                                 not_before=utc(datetime(2026, 11, 10, 13, 0, tzinfo=ET)))
    assert h.hello.tick() == "waiting to settle"
    assert h.at(MORNING_HOME + timedelta(minutes=21)).hello.tick() == "by text: sent"
    (facts,) = h.texted
    assert facts.brief is None and facts.held == []
    assert overnight.status == "queued"                                         # still his room's to say
    assert not [r for r in h.desk.store.rows.values() if r.dedupe_key == "brief:2026-11-10"]   # no carrier
    (row,) = h.hellos()
    assert row.folds == () and row.detail == ""
    # He reaches the office later: the brief goes on its own, said there.
    h.reading.room = "office"
    assert h.at(MORNING_HOME + timedelta(minutes=30)).brief.tick() == "sent"
    assert h.world.spoken == [("Morning brief.", "dev-office")]


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
