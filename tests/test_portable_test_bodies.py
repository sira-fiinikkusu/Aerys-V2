"""Test bodies never count as his words (board #176, 11/09).

Kael's gate body (ember) and the home body's test jobs record their turns at the door as the
owner's (`portable:<body>:<seq>` under his person id). On 11/09 her portable block held 63 ember
lines shown to her as "[ember] Chris: ...". Every reader that treats v2_turns as his speech or
her context skips PORTABLE_TEST_BODIES; the rows stay in v2_turns.
"""
from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

OWNER = "00000000-0000-4000-8000-000000000001"
DEFAULT = ["portable:ember:%", "portable:home:%"]


class Result:
    def __init__(self, rows=None, one=None):
        self.rows, self.one = rows or [], one

    def fetchall(self):
        return self.rows

    def fetchone(self):
        return self.one


class Conn:
    """Records every (sql, params); answers by a function of the SQL."""

    def __init__(self, answer=None):
        self.calls, self.answer, self.read_only = [], answer or (lambda sql, params: Result()), False

    def execute(self, sql, params=None):
        self.calls.append((sql, params))
        return self.answer(sql, params)

    def transaction(self):
        import contextlib
        return contextlib.nullcontext()

    def commit(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _connect_to(conn, monkeypatch):
    import psycopg
    monkeypatch.setattr(psycopg, "connect", lambda *a, **k: conn)


# --- the helper ------------------------------------------------------------------------


def test_the_default_is_ember_and_home():
    from aerys_v2.test_bodies import names, patterns
    assert names(None) == ["ember", "home"] and patterns(None) == DEFAULT


def test_a_list_is_cleaned_and_odd_names_are_never_interpolated():
    from aerys_v2.test_bodies import names, patterns
    assert names(" Ember , home,ember,,x'; drop table v2_turns;--") == ["ember", "home"]
    assert patterns("lab_body") == ["portable:lab\\_body:%"]      # '_' is a LIKE wildcard, escaped
    assert patterns("") == []                                         # an empty setting excludes nothing


def test_settings_carry_the_default():
    from aerys_v2.config import Settings
    assert Settings.model_fields["portable_test_bodies"].default == "ember,home"


# --- her portable block -----------------------------------------------------------------


def test_her_portable_block_skips_test_bodies(monkeypatch):
    from aerys_v2.factory import portable_context_fn_for
    rows = [("from the stick", "ok", "portable:stick:4", datetime(2026, 11, 9, tzinfo=timezone.utc))]
    conn = Conn(lambda sql, params: Result(rows))
    _connect_to(conn, monkeypatch)
    block = portable_context_fn_for(SimpleNamespace(database_url="postgresql://x", portable_context_limit=100))(OWNER)
    sql, params = conn.calls[-1]
    assert "LIKE ANY" in sql and params["test_bodies"] == DEFAULT
    assert "[stick]" in block


def test_a_custom_list_reaches_the_block(monkeypatch):
    from aerys_v2.factory import portable_context_fn_for
    conn = Conn()
    _connect_to(conn, monkeypatch)
    portable_context_fn_for(SimpleNamespace(database_url="postgresql://x", portable_context_limit=100,
                                            portable_test_bodies="ember"))(OWNER)
    assert conn.calls[-1][1]["test_bodies"] == ["portable:ember:%"]


# --- her memory -------------------------------------------------------------------------


def test_the_extractor_never_reads_a_test_bodys_turn():
    from aerys_v2.workers import extraction
    assert "LIKE ANY(%(test_bodies)s::text[])" in " ".join(extraction.V2_TURNS_SQL.split())
    source, staging = Conn(), Conn()
    extraction.run_extraction(source, staging, llm=None, embedder=None)
    turns = [p for sql, p in staging.calls if sql is extraction.V2_TURNS_SQL]
    assert turns and turns[0]["test_bodies"] == DEFAULT


def test_the_extractor_takes_the_setting():
    from aerys_v2.workers import extraction
    source, staging = Conn(), Conn()
    extraction.run_extraction(source, staging, llm=None, embedder=None, test_bodies=["portable:lab:%"])
    turns = [p for sql, p in staging.calls if sql is extraction.V2_TURNS_SQL]
    assert turns[0]["test_bodies"] == ["portable:lab:%"]


# --- the plan miner ---------------------------------------------------------------------


def test_the_plan_miner_skips_test_bodies(monkeypatch):
    from aerys_v2 import followups
    conn = Conn()
    _connect_to(conn, monkeypatch)
    followups.plan_miner_turns(SimpleNamespace(database_url="postgresql://x", owner_person_id=OWNER), 5)
    sql, params = conn.calls[-1]
    assert "LIKE ANY" in sql and params == (OWNER, 5, DEFAULT, followups.MINER_BATCH)


# --- the gap miner ----------------------------------------------------------------------


def test_the_gap_miner_skips_test_bodies():
    from aerys_v2.workers import capability_requests as cr

    def answer(sql, params):
        return Result(one=(10, 10)) if sql is cr.PARITY_SQL else Result()

    conn = Conn(answer)
    cr.run_gap_mining(conn, allowlist=[OWNER])
    mined = [p for sql, p in conn.calls if sql is cr.MINER_SQL]
    assert "LIKE ANY(%(test_bodies)s::text[])" in " ".join(cr.MINER_SQL.split())
    assert mined and mined[0]["test_bodies"] == DEFAULT


# --- "his last surface" and "did he reply" ----------------------------------------------


def test_his_last_surface_is_never_a_test_body(monkeypatch):
    from aerys_v2 import kael_line
    conn = Conn(lambda sql, params: Result(one=("discord", "c1", 5)))
    _connect_to(conn, monkeypatch)
    kael_line._owner_last_surface(SimpleNamespace(database_url="postgresql://x", owner_person_id=OWNER))
    sql, params = conn.calls[-1]
    assert "LIKE ANY" in sql and params == (OWNER, DEFAULT)


def test_a_test_bodys_turn_never_counts_as_his_reply(monkeypatch):
    from aerys_v2 import kael_line
    conn = Conn(lambda sql, params: Result(one=None))
    _connect_to(conn, monkeypatch)
    when = datetime(2026, 11, 9, 19, 0, tzinfo=timezone.utc)
    assert kael_line.owner_replied_since(SimpleNamespace(database_url="postgresql://x", owner_person_id=OWNER),
                                         when) is False
    sql, params = conn.calls[-1]
    assert "LIKE ANY" in sql and params == (OWNER, when, DEFAULT)


@pytest.mark.parametrize("sql_name", ["_OWNER_LAST_SURFACE_SQL", "_OWNER_REPLIED_SQL"])
def test_the_placeholders_line_up(sql_name):
    from aerys_v2 import kael_line
    sql = getattr(kael_line, sql_name)
    assert sql.count("%s") == (2 if sql_name == "_OWNER_LAST_SURFACE_SQL" else 3)
