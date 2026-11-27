"""11/26: he told her in the afternoon "when you see me out again its because i was at her parents", and her
welcome-home hello asked about a place off the map instead. The hello is a fresh turn fed only facts, so
what he told her himself that day is now one of the facts, and it beats any map guess."""

from __future__ import annotations

import logging
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from aerys_v2.welcome import (
    TOLD_CHANNELS,
    TOLD_CHARS,
    TOLD_LINES,
    TOLD_WINDOW,
    WELCOME_PROMPT,
    WELCOME_PROMPT_TOOL_FREE,
    WELCOME_TEXT_PROMPT,
    Stop,
    WelcomeFacts,
    WelcomeHome,
    told_for,
    told_rows,
    welcome_facts_text,
)

from test_welcome_home import HOME, LEFT, House, utc

OWNER = "6e6bcbed-0000-4000-8000-000000000001"
T0 = datetime(2026, 11, 10, 15, 14, tzinfo=timezone.utc)
PLAN = "round two of the dessert at her parents soon, so when you see me out again that's where I was"


def test_every_version_of_her_instructions_puts_his_words_before_the_map():
    for prompt in (WELCOME_PROMPT, WELCOME_PROMPT_TOOL_FREE, WELCOME_TEXT_PROMPT):
        assert "what he told you himself, read that FIRST" in prompt
        assert "never offer a place from the map instead" in prompt
        assert "don't repeat the rest" in prompt
        assert prompt.index("what he told you himself") < prompt.index("Never guess: if it isn't clear")


def test_the_facts_carry_his_words_with_their_time_ahead_of_the_map():
    facts = WelcomeFacts(away=timedelta(hours=4), room="office", told=[(utc(LEFT) - timedelta(hours=1), PLAN)],
                         stops=[Stop(place="12 Example St", start=utc(LEFT), end=utc(HOME))])
    text = welcome_facts_text(facts)
    assert f"- 12:00 PM: {PLAN}" in text
    assert text.index("What he told you himself today") < text.index("Where his phone stayed a while")
    assert "What he told you himself" not in welcome_facts_text(WelcomeFacts(away=timedelta(hours=4)))


def test_his_lines_come_oldest_first_on_one_line_clipped_and_capped():
    rows = [(T0 + timedelta(minutes=i), f"line {i}") for i in range(TOLD_LINES + 5)][::-1]
    out = told_rows(rows)
    assert len(out) == TOLD_LINES and out[0][1] == "line 5" and out[-1][1] == f"line {TOLD_LINES + 4}"
    long = told_rows([(T0, "word " * 200)])[0][1]
    assert len(long) == TOLD_CHARS and long.endswith("…")
    assert told_rows([(T0, "two\nlines\n\nhere"), (T0, "   "), (T0, None), ("not a time", "x")]) == [(T0, "two lines here")]
    assert told_rows(None) == []


class FakeConn:
    def __init__(self, rows):
        self.rows, self.seen = rows, []

    def execute(self, sql, params):
        self.seen.append((sql, params))
        return SimpleNamespace(fetchall=lambda: self.rows)


def test_it_reads_only_his_own_private_thread_in_the_hours_around_the_trip():
    conn = FakeConn([(T0, PLAN)])

    @contextmanager
    def connect():
        yield conn

    read = told_for(SimpleNamespace(database_url="postgresql://x", owner_person_id=OWNER), connect=connect)
    assert read(utc(LEFT), utc(HOME)) == [(T0, PLAN)]
    sql, p = conn.seen[0]
    assert p["owner"] == OWNER and p["thread"] == f"person:{OWNER}"
    assert set(p["channels"]) == set(TOLD_CHANNELS) and "discord" not in p["channels"]
    assert p["start"] == utc(LEFT) - TOLD_WINDOW and p["end"] == utc(HOME) and p["n"] == TOLD_LINES
    for must in ("person_id = %(owner)s", "thread_id = %(thread)s", "channel = ANY(%(channels)s)", "sealed_room IS NULL"):
        assert must in sql, must
    assert "raw_reply" not in sql and "emitted_reply" not in sql      # his words only, never her replies


def test_without_an_owner_or_a_database_it_is_not_armed():
    assert told_for(SimpleNamespace(database_url="postgresql://x", owner_person_id=None)) is None
    assert told_for(SimpleNamespace(database_url=None, owner_person_id=OWNER)) is None


class ToldHouse(House):
    def __init__(self, told, **kw):
        self._told = told
        super().__init__(**kw)

    def make(self):
        return WelcomeHome(self.desk, read=lambda: self.reading, away_since=self.away_since, compose=self.compose,
                           visits=lambda left, arrived: list(self.appointments), rooms=("office", "kitchen"),
                           brief=self.brief, followups=self.fstore, clock=lambda: self.clock["now"], told=self._told)


def test_the_hello_is_composed_with_what_he_told_her():
    asked = []

    def told(left, arrived):
        asked.append((left, arrived))
        return [(T0, PLAN)]

    h = ToldHouse(told)
    assert h.hello.tick() == "sent"
    assert asked == [(utc(LEFT), utc(HOME))]
    assert h.composed[0].told == [(T0, PLAN)]


def test_a_failed_read_costs_only_those_lines_and_none_of_his_words_are_logged(caplog):
    def broken(left, arrived):
        raise RuntimeError(PLAN)                    # even an error carrying his words must not reach the log

    with caplog.at_level(logging.INFO, logger="aerys_v2.welcome"):
        h = ToldHouse(broken)
        assert h.hello.tick() == "sent"
        assert h.composed[0].told == []
        ok = ToldHouse(lambda left, arrived: [(T0, PLAN)])
        assert ok.hello.tick() == "sent"
    assert "parents" not in caplog.text and "RuntimeError" in caplog.text


def test_the_plain_house_without_it_still_says_hello():
    h = House()
    assert h.hello.tick() == "sent" and h.composed[0].told == []
