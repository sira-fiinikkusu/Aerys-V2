"""D96 (Chris, 11/24 23:05): "im home late yes and I dont want her blowing the house up after hours in voice
but I would have appreciated a welcome home thing from her tonight."

11/24: he left at 13:44 and got home at 22:30. The house knew, but 22:30 is inside her quiet hours, and the
hello was dropped whole: no voice, and no Discord either (D91 had said "never at night"). Now, after hours,
the hello goes to him as a Discord text, written for text and never spoken. At night the desk sends it as
quiet text (no buzz). D91's other reasons to stay quiet still hold: a short trip, welcomes muted, he's
talked with her since he got home, or the morning brief greets him (before nine, D18 #3)."""

from __future__ import annotations

from datetime import datetime, timedelta

from aerys_v2.welcome import BRIEF_GREETS, TALKED, TEXT_SURFACE
from test_d91_welcome_by_text import TextHouse
from test_welcome_home import EARLY_HOME, EARLY_LEFT, ET, utc

LATE = datetime(2026, 11, 10, 22, 30, tzinfo=ET)           # tonight's shape: home at 10:30 PM
LATE_LEFT = datetime(2026, 11, 10, 13, 44, tzinfo=ET)


def late_house(**kw):
    h = TextHouse(LATE + timedelta(minutes=2), left=LATE_LEFT, home=LATE, **kw)
    h.reading.room, h.reading.confirmed = "living room", True
    return h


def test_home_after_hours_the_hello_goes_by_text_never_aloud():
    h = late_house()
    assert h.hello.tick() == "by text: sent"
    assert h.world.spoken == []                            # never aloud
    (surface, words, silent), = h.world.texts
    assert surface == TEXT_SURFACE and words == h.text_words
    assert silent is True                                  # at night it doesn't buzz him
    (row,) = h.hellos()
    assert row.status == "sent" and row.text_only and row.text_to == "discord" and row.room is None
    (facts,) = h.texted
    assert facts.away == LATE - LATE_LEFT and facts.held == []


def test_it_goes_right_away_without_waiting_to_settle_in_a_speaker_room():
    h = late_house()
    h.reading.room, h.reading.confirmed = None, False       # not settled anywhere yet
    assert h.hello.tick() == "by text: sent" and len(h.world.texts) == 1


def test_settled_in_her_office_after_hours_it_is_still_a_text_not_her_voice():
    h = late_house()
    h.reading.room = "office"                               # one of her speaker rooms
    assert h.hello.tick() == "by text: sent"
    assert h.world.spoken == [] and len(h.world.texts) == 1


def test_once_per_arrival_even_across_a_restart():
    h = late_house()
    assert h.hello.tick() == "by text: sent"
    assert h.at(LATE + timedelta(minutes=20)).hello.tick() == "done"
    assert h.make().tick() == "done" and len(h.world.texts) == 1


def test_talked_with_her_since_he_got_home_nothing_goes():
    h = late_house()
    h.world.replied = True
    assert h.hello.tick() == TALKED
    (row,) = h.hellos()
    assert row.status == "dropped" and row.reason == TALKED and h.world.texts == []


def test_muted_nothing_goes():
    from aerys_v2.tools.reach_prefs import build_reach_prefs_tool

    h = late_house()
    tool = build_reach_prefs_tool(h.desk.store, clock=lambda: utc(LATE))
    assert tool.invoke({"action": "mute", "topic": "welcome", "his_words": "don't greet me"}).startswith("Changed")
    assert h.hello.tick() == "muted" and h.world.texts == []
    (row,) = h.hellos()
    assert row.status == "dropped" and row.reason == "you muted welcome"


def test_a_short_trip_after_hours_stays_quiet():
    h = TextHouse(LATE + timedelta(minutes=2), left=LATE - timedelta(minutes=12), home=LATE)
    assert h.hello.tick() == "a short trip" and h.world.texts == [] and h.hellos() == []


def test_given_up_into_the_night_it_still_goes_by_text():
    """Home at 20:10, never reached a speaker room; by 21:15 it's after hours: the text goes (it used to be dropped)."""
    home = datetime(2026, 11, 10, 20, 10, tzinfo=ET)
    h = TextHouse(home + timedelta(minutes=5), left=home - timedelta(hours=3), home=home)
    assert h.hello.tick() == "waiting to settle"
    assert h.at(home + timedelta(minutes=65)).hello.tick() == "by text: sent"
    assert h.world.spoken == [] and len(h.world.texts) == 1 and h.world.texts[0][2] is True


def test_before_nine_with_the_brief_due_the_brief_greets_him_not_a_text():
    """D18 #3 stands: home inside quiet hours with the morning brief still due, the brief greets him."""
    h = TextHouse(EARLY_HOME + timedelta(minutes=1), left=EARLY_LEFT, home=EARLY_HOME, brief=True, settle=20)
    assert h.hello.tick() == "waiting to settle"
    assert h.at(EARLY_HOME + timedelta(minutes=21)).hello.tick() == BRIEF_GREETS
    assert h.world.texts == [] and h.texted == []
    assert h.at(EARLY_HOME + timedelta(minutes=25)).brief.tick() == "sent"
    assert h.world.spoken == [("Morning brief.", "dev-office")]
