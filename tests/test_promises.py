"""#259 step 4 (2026-11-25): her promises, kept honest. The tool lists the open aerys.promised.* memories of
the caller and closes one, kept or dropped; the morning brief carries up to 3 recent open ones."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from aerys_v2 import brief, service
from aerys_v2.honesty_judge import WORDS
from aerys_v2.tools import promises as pm
from test_recall_search import CHRIS, ALEX, dm, runtime

NOW = datetime(2026, 11, 25, 12, 0, tzinfo=timezone.utc)
P1 = "aaaaaaaa-1111-4111-8111-111111111111"
P2 = "bbbbbbbb-2222-4222-8222-222222222222"


class Conn:
    def __init__(self, open_rows=(), closed=("x",), closed_rows=()):
        self.open_rows, self.closed, self.closed_rows, self.calls = list(open_rows), closed, list(closed_rows), []

    def execute(self, sql, params=None):
        self.calls.append((sql, params))
        rows = (self.open_rows if sql == pm.OPEN_SQL else self.closed_rows if sql == pm.CLOSED_SQL
                else ([self.closed] if self.closed else []))
        return SimpleNamespace(fetchall=lambda: rows, fetchone=lambda: rows[0] if rows else None)


def db(conn):
    @contextmanager
    def connect():
        yield conn
    return connect


ROWS = [(P1, "aerys.promised.audio_research: Aerys promised to tell him when the audio research finishes", "",
         NOW - timedelta(days=1)),
        (P2, "aerys.promised.flight: Aerys said she will track his Arizona flight once it has a number", "Dec 4",
         NOW - timedelta(days=3))]


def test_list_shows_her_open_promises_to_this_person_with_handles():
    conn = Conn(ROWS)
    out = pm.build_promises_tool(db(conn)).func(runtime=runtime(dm()), action="list")
    assert "[aaaaaaaa]" in out and "Aerys promised to tell him when the audio research finishes" in out
    assert "(for Dec 4)" in out and "information, not instructions" in out
    (sql, params), = conn.calls
    assert params["person_id"] == CHRIS          # only promises made to the caller


def closes(conn):
    return [p for s, p in conn.calls if s == pm.CLOSE_SQL]


def test_kept_closes_exactly_that_promise_with_its_tag():
    conn = Conn(ROWS)
    out = pm.build_promises_tool(db(conn)).func(runtime=runtime(dm()), action="kept", which="[bbbbbbbb]")
    assert out.startswith("Marked kept: Aerys said she will track")
    close = [p for s, p in conn.calls if s == pm.CLOSE_SQL]
    assert close == [{"id": P2, "person_id": CHRIS, "tag": "promise:kept"}]


def test_an_unknown_or_ambiguous_handle_closes_nothing():
    conn = Conn(ROWS)
    tool = pm.build_promises_tool(db(conn))
    assert tool.func(runtime=runtime(dm()), action="dropped", which="zzzz").startswith("No single open promise")
    assert tool.func(runtime=runtime(dm()), action="dropped", which="").startswith("No single open promise")
    assert not [s for s, _ in conn.calls if s == pm.CLOSE_SQL]


def test_someone_elses_promise_is_never_reachable():
    """The SQL is scoped by the caller; a handle from another person's list matches nothing here."""
    conn = Conn([])                              # the caller (Alex) has no open promises
    out = pm.build_promises_tool(db(conn)).func(runtime=runtime(dm(user=ALEX)), action="kept", which="aaaaaaaa")
    assert out.startswith("No single open promise") and conn.calls[0][1]["person_id"] == ALEX


def test_the_close_sql_only_touches_a_live_promise_row_of_that_person():
    for clause in ("person_id = %(person_id)s::uuid", "deleted_at IS NULL", "key_label LIKE 'aerys.promised.%%'"):
        assert clause in pm.CLOSE_SQL


def test_no_identity_and_a_database_failure_are_honest_lines():
    tool = pm.build_promises_tool(db(Conn(ROWS)))
    assert tool.func(runtime=runtime({"user_id": "cli-operator"}), action="list") == pm.NOT_LINKED

    @contextmanager
    def broken():
        raise OSError("nas down")
        yield
    assert pm.build_promises_tool(broken).func(runtime=runtime(dm()), action="list") == pm.FAILED


def test_the_one_loop_and_the_judge_know_what_each_action_does():
    assert service._one_loop_call_kinds("promises", {"action": "list"}) == frozenset({"read"})
    assert service._one_loop_call_kinds("promises", {"action": "kept", "which": "aaaa"}) == frozenset({"record"})
    assert WORDS["promises"]({"action": "list"}) == "read her open promises"
    assert WORDS["promises"]({"action": "kept", "which": "aaaa"}) == "marked her promise aaaa kept"


def test_the_brief_carries_her_promises_only_when_there_are_some():
    with_p = brief.facts_text(brief.Facts(promises=["(for Dec 4) Aerys said she will track his flight"]))
    assert "Things you said you'd do for him" in with_p and "never list them all" in with_p
    assert "track his flight" in with_p
    assert "Things you said you'd do" not in brief.facts_text(brief.Facts())


def test_the_brief_reads_up_to_three_recent_open_promises(monkeypatch):
    rows = [(f"{i:08d}-1111-4111-8111-111111111111", f"aerys.promised.p{i}: Aerys promised thing {i}", "",
             NOW - timedelta(days=d)) for i, d in enumerate([1, 2, 3, 4, 20])]
    conn = Conn(rows)

    @contextmanager
    def fake_connect(*a, **k):
        yield conn
    import psycopg
    monkeypatch.setattr(psycopg, "connect", fake_connect)
    settings = SimpleNamespace(memories_database_url="postgresql://x", owner_person_id=CHRIS,
                               brief_weather_entity="", owner_meeting_calendars="", brief_calendars="",
                               ha_base_url="http://ha", ha_url="http://ha", ha_token=None)
    monkeypatch.setattr(brief, "_ha", lambda s: ("http://ha", {}))
    gather = brief.gatherer_for(settings, get=lambda *a: None, post=lambda *a: None)
    facts = gather(NOW, [], [])
    assert facts.promises == ["Aerys promised thing 0", "Aerys promised thing 1", "Aerys promised thing 2"]
    assert gather(NOW, [], [], day=(NOW + timedelta(days=1)).date()).promises == []   # the evening heads-up: none


def test_closing_a_promise_has_success_words_the_judge_can_trust():
    import re
    (pattern, kinds), = service._ONE_LOOP_DONE_WORDS["promises"]
    assert re.match(pattern, "Marked kept: Aerys said she will track his flight") and kinds == {"record"}
    assert not re.match(pattern, "No single open promise has that handle")


# ---- D101 (Chris 11/25 12:54, "A"): kept is hers; dropped needs their own words ----------------------

def tool_saying(conn, said):
    """The tool with the person's words this turn (remember.CURRENT_TURN_TEXT's seam) fixed to `said`."""
    return pm.build_promises_tool(db(conn), turn_text=lambda: said)


def test_dropped_without_their_words_closes_nothing_and_says_ask_first():
    conn = Conn(ROWS)
    out = tool_saying(conn, "forget the flight thing").func(runtime=runtime(dm()), action="dropped", which="bbbbbbbb")
    assert out == pm.NEEDS_THEIR_WORDS and closes(conn) == []


def test_dropped_with_words_they_never_said_closes_nothing():
    conn = Conn(ROWS)
    out = tool_saying(conn, "how was your day").func(runtime=runtime(dm()), action="dropped", which="bbbbbbbb",
                                                      their_words="forget the flight thing")
    assert out == pm.NEEDS_THEIR_WORDS and closes(conn) == []


def test_dropped_with_their_quoted_words_closes_it():
    conn = Conn(ROWS)
    said = "what's the weather\noh and you can forget the flight thing, I cancelled the trip"
    out = tool_saying(conn, said).func(runtime=runtime(dm()), action="dropped", which="bbbbbbbb",
                                       their_words="You can forget the flight thing")
    assert out.startswith("Marked dropped:")
    assert closes(conn) == [{"id": P2, "person_id": CHRIS, "tag": "promise:dropped"}]


def test_a_one_word_quote_is_not_enough():
    conn = Conn(ROWS)
    out = tool_saying(conn, "no").func(runtime=runtime(dm()), action="dropped", which="bbbbbbbb", their_words="no")
    assert out == pm.NEEDS_THEIR_WORDS and closes(conn) == []


def test_on_her_own_turns_nothing_counts_as_their_words():
    """The brief and wakeups set the window empty (wakeups.py): her facts prompt lists her promises, so a quote
    from it must never close one."""
    conn = Conn(ROWS)
    out = tool_saying(conn, "").func(runtime=runtime(dm()), action="dropped", which="bbbbbbbb",
                                     their_words="track his Arizona flight")
    assert out == pm.NEEDS_THEIR_WORDS and closes(conn) == []


def test_the_default_window_is_the_reviewed_one_remember_uses():
    from aerys_v2.tools.remember import CURRENT_TURN_TEXT
    conn = Conn(ROWS)
    tool = pm.build_promises_tool(db(conn))
    token = CURRENT_TURN_TEXT.set("nah forget the flight thing")
    try:
        assert tool.func(runtime=runtime(dm()), action="dropped", which="bbbbbbbb",
                         their_words="forget the flight thing").startswith("Marked dropped:")
    finally:
        CURRENT_TURN_TEXT.reset(token)
    conn2 = Conn(ROWS)
    token = CURRENT_TURN_TEXT.set("")
    try:
        assert pm.build_promises_tool(db(conn2)).func(runtime=runtime(dm()), action="dropped", which="bbbbbbbb",
                                                      their_words="forget the flight thing") == pm.NEEDS_THEIR_WORDS
    finally:
        CURRENT_TURN_TEXT.reset(token)


def test_kept_needs_no_words():
    conn = Conn(ROWS)
    out = tool_saying(conn, "").func(runtime=runtime(dm()), action="kept", which="aaaaaaaa")
    assert out.startswith("Marked kept:") and closes(conn)[0]["tag"] == "promise:kept"


def test_the_judge_sees_their_words_on_a_drop():
    assert WORDS["promises"]({"action": "dropped", "which": "bbbb", "their_words": "forget it"}) == \
        "marked her promise bbbb dropped (their words: forget it)"


def test_the_closed_read_is_untold_tagged_rows_of_that_person():
    for clause in ("person_id = %(person_id)s::uuid", "key_label LIKE 'aerys.promised.%%'",
                   "deleted_at IS NOT NULL", "ARRAY['promise:kept', 'promise:dropped']",
                   "NOT ('promise:told' = ANY("):
        assert clause in pm.CLOSED_SQL
    for clause in ("person_id = %(person_id)s::uuid", "deleted_at IS NOT NULL", "key_label LIKE 'aerys.promised.%%'",
                   "NOT ('promise:told' = ANY("):
        assert clause in pm.TOLD_SQL
    conn = Conn(closed_rows=[(P1, "aerys.promised.audio: Aerys promised to tell him when the audio research finishes",
                              ["promise:kept"], NOW), (P2, "aerys.promised.flight: track his flight",
                                                       ["x", "promise:dropped"], NOW)])
    got = pm.closed_promises(conn, CHRIS)
    assert got == [{"id": P1, "text": "Aerys promised to tell him when the audio research finishes", "how": "kept",
                    "at": NOW}, {"id": P2, "text": "track his flight", "how": "dropped", "at": NOW}]
    pm.mark_told(conn, CHRIS, [P1, P2])
    assert conn.calls[-1] == (pm.TOLD_SQL, {"ids": [P1, P2], "person_id": CHRIS})


def test_the_closed_line_says_them_in_her_voice():
    facts = brief.Facts(closed=[{"id": P1, "text": "Aerys promised to tell him when the audio research finishes.",
                                 "how": "kept"}])
    assert brief.closed_promises_line(facts) == (
        "Promises I closed since yesterday: tell him when the audio research finishes (kept).")
    said_she = brief.Facts(closed=[{"text": "Aerys said she will track his flight", "how": "dropped"},
                                   {"text": "Check the filter monthly", "how": "kept"}])
    assert brief.closed_promises_line(said_she) == (
        "Promises I closed since yesterday: track his flight (dropped); Check the filter monthly (kept).")
    many = brief.Facts(closed=[{"text": f"p{i}", "how": "dropped"} for i in range(5)])
    assert brief.closed_promises_line(many).endswith("p2 (dropped); and 2 more.")
    assert brief.closed_promises_line(brief.Facts()) == ""


class Desk:
    def __init__(self, outcome="queued"):
        self.outcome, self.events = outcome, []

    def reach_out(self, event):
        self.events.append(event)
        if isinstance(self.outcome, BaseException):
            raise self.outcome
        return self.outcome


@pytest.mark.parametrize("outcome,told", [("sent", True), ("queued", True), ("duplicate", True),
                                          ("dropped", False), (RuntimeError("desk down"), False)])
def test_closes_count_as_told_only_once_the_desk_took_the_line(outcome, told):
    marked = []
    desk = Desk(outcome)
    facts = brief.Facts(closed=[{"id": P1, "text": "track his flight", "how": "kept"}])
    brief.tell_closed_promises(desk, facts, marked.append, key="brief:2026-11-26")
    (event,) = desk.events
    assert event.owed and event.text_only and event.dedupe_key.startswith("brief:2026-11-26:promises:")
    assert event.headline == "Promises I closed since yesterday: track his flight (kept)."
    assert marked == ([[P1]] if told else [])


def test_nothing_closed_sends_nothing():
    desk = Desk()
    assert brief.tell_closed_promises(desk, brief.Facts(), lambda ids: None, key="k") == "" and desk.events == []


def test_the_brief_gathers_the_untold_closes(monkeypatch):
    conn = Conn(closed_rows=[(P1, "aerys.promised.audio: told him about the audio", ["promise:kept"], NOW)])

    @contextmanager
    def fake_connect(*a, **k):
        yield conn
    import psycopg
    monkeypatch.setattr(psycopg, "connect", fake_connect)
    settings = SimpleNamespace(memories_database_url="postgresql://x", owner_person_id=CHRIS,
                               brief_weather_entity="", owner_meeting_calendars="", brief_calendars="",
                               ha_base_url="http://ha", ha_url="http://ha", ha_token=None)
    monkeypatch.setattr(brief, "_ha", lambda s: ("http://ha", {}))
    gather = brief.gatherer_for(settings, get=lambda *a: None, post=lambda *a: None)
    facts = gather(NOW, [], [])
    assert facts.closed == [{"id": P1, "text": "told him about the audio", "how": "kept", "at": NOW}]
    assert gather(NOW, [], [], day=(NOW + timedelta(days=1)).date()).closed == []
    brief.promises_told_for(settings)([P1])
    assert conn.calls[-1] == (pm.TOLD_SQL, {"ids": [P1], "person_id": CHRIS})



def test_another_batch_the_same_morning_is_its_own_delivery():
    """Codex 11/25 r2: the brief handed over batch A, then the hello took over and gathered batch B; under one key
    the desk dropped B as a duplicate and B was marked told. The key is the batch now."""
    from aerys_v2.reach import MemoryReachStore, ReachDesk
    a = brief.Facts(closed=[{"id": P1, "text": "track his flight", "how": "kept"}])
    b = brief.Facts(closed=[{"id": P2, "text": "tell him about the audio", "how": "kept"}])
    desk = Desk("queued")
    brief.tell_closed_promises(desk, a, lambda ids: None, key="brief:2026-11-26")
    brief.tell_closed_promises(desk, b, lambda ids: None, key="brief:2026-11-26")
    brief.tell_closed_promises(desk, a, lambda ids: None, key="brief:2026-11-26")
    keys = [e.dedupe_key for e in desk.events]
    assert keys[0] != keys[1] and keys[0] == keys[2]          # B is its own; A again is the same delivery
