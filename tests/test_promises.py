"""#259 step 4 (2026-11-25): her promises, kept honest. The tool lists the open aerys.promised.* memories of
the caller and closes one, kept or dropped; the morning brief carries up to 3 recent open ones."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from aerys_v2 import brief, service
from aerys_v2.honesty_judge import WORDS
from aerys_v2.tools import promises as pm
from test_recall_search import CHRIS, ALEX, dm, runtime

NOW = datetime(2026, 11, 25, 12, 0, tzinfo=timezone.utc)
P1 = "aaaaaaaa-1111-4111-8111-111111111111"
P2 = "bbbbbbbb-2222-4222-8222-222222222222"


class Conn:
    def __init__(self, open_rows=(), closed=("x",)):
        self.open_rows, self.closed, self.calls = list(open_rows), closed, []

    def execute(self, sql, params=None):
        self.calls.append((sql, params))
        rows = self.open_rows if sql == pm.OPEN_SQL else ([self.closed] if self.closed else [])
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
