"""Offline tests for the read-only DB services — no Postgres, no network.

A FakeConn (canned list-of-rows) stands in for psycopg, the same trick as pinning
an n8n Postgres node's output to test the Code nodes after it. What these prove:
SQL param shapes, that the n8n quirk-workarounds (sentinels, CTE wrappers, ::text
casts) did NOT survive the port, assembly/formatting rules, and the scoring math.
"""

import uuid
from datetime import datetime, timedelta, timezone

import pytest

from aerys_v2.services.identity import resolve_identity
from aerys_v2.services.memory import (
    CONTEXT_CAP,
    RECENCY_WINDOW_S,
    combined_score,
    embedding_to_pgvector,
    format_memory_context,
    retrieve_memories,
)
from aerys_v2.services.profile import format_profile_context, get_profile

NOW = datetime(2026, 7, 2, 12, 0, 0, tzinfo=timezone.utc)
PERSON = "6e6bcbed-03ef-4d17-95d2-89c467414335"


class FakeCursor:
    def __init__(self, rows):
        self.rows = rows

    def fetchone(self):
        return self.rows[0] if self.rows else None

    def fetchall(self):
        return self.rows


class FakeConn:
    """Duck-typed psycopg connection: records every execute, replays canned rows."""

    def __init__(self, results=None):
        self.results = list(results or [])
        self.calls = []  # [(sql, params), ...]

    def execute(self, sql, params=None):
        self.calls.append((sql, params))
        return FakeCursor(self.results.pop(0) if self.results else [])


# --- identity ---------------------------------------------------------------


def test_resolve_identity_found():
    pid = uuid.uuid4()
    conn = FakeConn([[(pid, "Chris")]])
    result = resolve_identity(conn, "discord", "12345")
    assert result == {"person_id": str(pid), "display_name": "Chris", "is_new": False}
    sql, params = conn.calls[0]
    assert params == ("discord", "12345")
    # The n8n quirk-workarounds must be dead: no sentinel, no UUID text cast.
    assert "UNION" not in sql
    assert "::text" not in sql
    assert "LIMIT" not in sql  # LIMIT 1 only existed to pick real row over sentinel


def test_resolve_identity_miss_returns_none_without_writing():
    conn = FakeConn([[]])
    assert resolve_identity(conn, "discord", "nobody") is None
    # READ-ONLY contract: exactly one statement ran, and it's a SELECT —
    # no pending_links DELETE, no create-on-miss INSERTs.
    assert len(conn.calls) == 1
    assert conn.calls[0][0].lstrip().upper().startswith("SELECT")


# --- profile ----------------------------------------------------------------

PROFILE_COLS = (
    "core_id",
    "key_label",
    "claim_text",
    "status",
    "locked",
    "confidence",
    "sensitivity",
    "visibility",
)


def claim_row(key_label, claim_text, locked=False, confidence=0.9):
    return (uuid.uuid4(), key_label, claim_text, "approved", locked, confidence, "P3", "all")


def test_get_profile_param_shape_and_quirks_killed():
    conn = FakeConn([[]])
    get_profile(conn, PERSON)  # privacy_context defaults to 'public'
    sql, params = conn.calls[0]
    assert params == {"pid": PERSON, "pctx": "public"}
    assert "WITH params" not in sql  # duplicate-$1 CTE wrapper killed
    assert "UNION" not in sql  # zero-row sentinel killed
    # The privacy semantics that must survive verbatim:
    assert "('P2', 'P3')" in sql
    assert "LIMIT 15" in sql


def test_profile_cold_start_on_zero_rows():
    conn = FakeConn([[]])
    assert get_profile(conn, PERSON) == {
        "profile": {"display_name": None, "lines": [], "cold_start": True}
    }


def test_profile_category_order_fixed_then_insertion():
    conn = FakeConn(
        [
            [
                claim_row("zeta.hobby", "Collects meteorites"),
                claim_row("emotional.mood", "Optimistic under deadline"),
                claim_row("basic.name", "Preferred name: Chris"),
                claim_row("interests.gaming", "Playing Subnautica with Joe"),
                claim_row("alpha.misc", "Alpha thing"),
            ]
        ]
    )
    profile = get_profile(conn, PERSON)["profile"]
    # basic → interests → emotional (fixed order), then zeta, alpha (arrival order).
    assert profile["lines"] == [
        "• Preferred name: Chris",
        "• Playing Subnautica with Joe",
        "• Optimistic under deadline",
        "• Collects meteorites",
        "• Alpha thing",
    ]
    assert profile["cold_start"] is False


def test_profile_display_name_takes_last_two_char_separator_segment():
    # ': ' is a TWO-char separator — lone colons inside the value must survive,
    # and the LAST segment wins (JS .split(': ').pop()).
    profile = format_profile_context(
        [dict(zip(PROFILE_COLS, claim_row("basic.name", "Known as: aka: Kael")))]
    )["profile"]
    assert profile["display_name"] == "Kael"


def test_profile_display_name_bare_name_key_and_no_separator():
    profile = format_profile_context(
        [dict(zip(PROFILE_COLS, claim_row("name", "Chris")))]
    )["profile"]
    # No ': ' at all → the whole text is the name (JS split returns [whole]).
    assert profile["display_name"] == "Chris"


def test_profile_lines_use_bullet_u2022():
    profile = format_profile_context(
        [dict(zip(PROFILE_COLS, claim_row("other.note", "Loves n8n")))]
    )["profile"]
    assert profile["lines"] == ["• Loves n8n"]
    assert profile["display_name"] is None  # no name claim present


# --- memory: retrieval ------------------------------------------------------


def memory_row(content, source="discord", days_ago=1.0, score=0.8):
    return (
        uuid.uuid4(),
        uuid.UUID(PERSON),
        content,
        source,
        "public",
        NOW - timedelta(days=days_ago),
        score,
    )


def test_retrieve_param_shape_public():
    conn = FakeConn([[]])
    retrieve_memories(conn, PERSON, query_embedding=[0.25, -1.0])
    sql, params = conn.calls[0]
    assert params == {
        "embedding": "[0.25,-1.0]",
        "person_id": PERSON,
        "levels": ["public"],  # public context NEVER sees private memories
    }
    assert "UNION" not in sql  # zero-row sentinel killed
    assert "m.embedding IS NOT NULL" in sql  # NULLS-FIRST latent bug fixed
    assert "LIMIT 20" in sql


def test_retrieve_private_context_sees_both_levels():
    conn = FakeConn([[]])
    retrieve_memories(conn, PERSON, query_embedding=[0.1], privacy_context="private")
    assert conn.calls[0][1]["levels"] == ["public", "private"]


def test_retrieve_embed_seam_called_with_text():
    seen = []

    def fake_embed(text):
        seen.append(text)
        return [0.5, 0.5]

    conn = FakeConn([[memory_row("likes: coffee")]])
    rows = retrieve_memories(conn, PERSON, query_text="coffee?", embed=fake_embed)
    assert seen == ["coffee?"]
    assert rows[0]["content"] == "likes: coffee"
    assert isinstance(rows[0]["combined_score"], float)


def test_retrieve_empty_text_still_embeds_as_empty_string():
    # n8n embedded '' for empty messages — preserved behavior.
    seen = []
    conn = FakeConn([[]])
    retrieve_memories(conn, PERSON, embed=lambda t: seen.append(t) or [0.1])
    assert seen == [""]


def test_retrieve_short_circuits_on_empty_embedding():
    # n8n fell through to '[]' which would ERROR in Postgres; we return [] and
    # never touch the connection.
    conn = FakeConn()
    assert retrieve_memories(conn, PERSON, embed=lambda t: []) == []
    assert conn.calls == []


def test_retrieve_requires_embedding_or_seam():
    with pytest.raises(ValueError):
        retrieve_memories(FakeConn(), PERSON)


def test_embedding_serialization_matches_n8n_join():
    assert embedding_to_pgvector([0.25, -1.0, 3]) == "[0.25,-1.0,3]"


# --- memory: scoring math ---------------------------------------------------


def test_score_is_similarity_scaled_by_recency_boost():
    # Fresh memory (age 0): recency 1.0 → score = sim * (0.7 + 0.3) = sim
    assert combined_score(1.0, 0) == pytest.approx(1.0)
    assert combined_score(0.5, 0) == pytest.approx(0.5)
    # Exactly 30 days old: recency decayed to 0 → sim * 0.7 floor.
    assert combined_score(0.5, RECENCY_WINDOW_S) == pytest.approx(0.35)
    # Half the window: recency 0.5 → 0.5 * (0.7 + 0.15)
    assert combined_score(0.5, RECENCY_WINDOW_S / 2) == pytest.approx(0.425)


def test_score_recency_clamps_both_ends():
    # Older than 30 days: recency clamped to 0 → the 0.7 floor, never lower.
    assert combined_score(1.0, RECENCY_WINDOW_S * 10) == pytest.approx(0.7)
    # Future timestamp (clock skew): recency clamped to 1, never above sim.
    assert combined_score(1.0, -999_999) == pytest.approx(1.0)


def test_score_relevance_gates_recency():
    # THE 2026-07-03 regression: "who am I married to?" must rank an aged
    # on-point memory (sim 0.44, >30d old) above fresh small talk (sim 0.25).
    # The old additive form gave the fresh row +0.3 flat and buried the answer.
    aged_relevant = combined_score(0.44, RECENCY_WINDOW_S * 3)
    fresh_noise = combined_score(0.25, 0)
    assert aged_relevant > fresh_noise
    # And recency still breaks ties between equally relevant memories.
    assert combined_score(0.5, 0) > combined_score(0.5, RECENCY_WINDOW_S)


# --- memory: context formatting ----------------------------------------------


def test_format_empty_rows_is_empty_string():
    assert format_memory_context([], now=NOW) == ""


def test_format_line_shape_with_source_and_age():
    rows = [dict(zip("id person_id content source_platform privacy_level created_at combined_score".split(), memory_row("likes: black coffee", days_ago=3)))]
    assert format_memory_context(rows, now=NOW) == "* black coffee [discord] (2026-06-29, 3d ago)"


def test_format_without_source_has_single_spaces():
    # Cosmetic fix vs n8n: no interior double-space when source is missing.
    rows = [{"content": "plain memory", "source_platform": None, "created_at": NOW - timedelta(days=2)}]
    assert format_memory_context(rows, now=NOW) == "* plain memory (2026-06-30, 2d ago)"


def test_format_strips_only_first_colon():
    # Values containing colons (URLs, times) must survive the key-prefix strip.
    rows = [{"content": "homepage: https://kael.dev:8080/x", "source_platform": None, "created_at": NOW}]
    assert format_memory_context(rows, now=NOW) == "* https://kael.dev:8080/x (2026-07-02, 0d ago)"


def test_format_dedup_by_key_prefix_first_wins():
    # Rows arrive score-sorted DESC, so 'first wins' keeps the best-scored claim.
    rows = [
        {"content": "likes: espresso", "source_platform": None, "created_at": NOW},
        {"content": "LIKES : drip coffee", "source_platform": None, "created_at": NOW},
        {"content": "no colon here", "source_platform": None, "created_at": NOW},
        {"content": "No Colon Here", "source_platform": None, "created_at": NOW},
    ]
    out = format_memory_context(rows, now=NOW)
    assert out == "* espresso (2026-07-02, 0d ago)\n* no colon here (2026-07-02, 0d ago)"


def test_format_caps_at_five_after_dedup():
    rows = [
        {"content": f"key{i}: value {i}", "source_platform": None, "created_at": NOW}
        for i in range(10)
    ]
    assert len(format_memory_context(rows, now=NOW).splitlines()) == CONTEXT_CAP


def test_format_skips_null_content_rows():
    # Belt-and-braces sentinel filter kept from the n8n Format node.
    rows = [
        {"content": None, "source_platform": None, "created_at": None},
        {"content": "real: memory", "source_platform": None, "created_at": NOW},
    ]
    assert format_memory_context(rows, now=NOW) == "* memory (2026-07-02, 0d ago)"


def test_format_age_rounds_half_up_like_js():
    # 2.5 days → JS Math.round gives 3 (half rounds UP, not banker's).
    rows = [{"content": "x", "source_platform": None, "created_at": NOW - timedelta(days=2.5)}]
    assert format_memory_context(rows, now=NOW) == "* x (2026-06-30, 3d ago)"
