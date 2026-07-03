"""Offline tests for the shadow extraction worker — no Postgres, no network.

FakeConn routes canned rows by SQL substring (pinning n8n node outputs, again).
What these prove: person-grouping happens BEFORE the LLM sees a transcript (the
pre-05.1 misattribution lesson), the watermark stores the RAW Postgres string
verbatim (the ms-vs-µs precision bug), prod gets SELECTs only while every write
lands in aerys_v2 staging, and an empty window is a true no-op.
"""

import json
from datetime import datetime, timedelta, timezone

from aerys_v2.workers.extraction import (
    EXTRACTION_SYSTEM_PROMPT,
    SOURCE_COLUMNS,
    batch_date,
    parse_observations,
    run_extraction,
)

T0 = datetime(2026, 7, 3, 12, 0, 0, tzinfo=timezone.utc)
CHRIS = "6e6bcbed-03ef-4d17-95d2-89c467414335"
MEGAN = "11111111-2222-3333-4444-555555555555"


def row(person_id, content, *, at=T0, raw=None, speaker="Unknown",
        platform="discord", privacy="private", thread="v1:n8n_chat_histories"):
    """A source row in SOURCE_COLUMNS order — both queries return this shape."""
    values = {
        "id": "1",
        "person_id": person_id,
        "content": content,
        "source_platform": platform,
        "privacy_level": privacy,
        "created_at": at,
        "created_at_raw": raw or at.strftime("%Y-%m-%d %H:%M:%S.%f+00"),
        "speaker_name": speaker,
        "source_thread": thread,
    }
    return tuple(values[c] for c in SOURCE_COLUMNS)


class FakeCursor:
    def __init__(self, rows):
        self.rows = rows

    def fetchone(self):
        return self.rows[0] if self.rows else None

    def fetchall(self):
        return self.rows


class FakeConn:
    """Duck-typed psycopg connection: routes by SQL substring, records every call."""

    def __init__(self, routes=()):
        self.routes = list(routes)  # [(sql_substring, rows), ...] first match wins
        self.calls = []  # [(sql, params), ...]

    def execute(self, sql, params=None):
        self.calls.append((sql, params))
        for needle, rows in self.routes:
            if needle in sql:
                return FakeCursor(rows)
        return FakeCursor([])


class FakeLlm:
    """Records every (system, user) call, replays canned reply texts."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def __call__(self, system, user):
        self.calls.append((system, user))
        return self.replies.pop(0) if self.replies else "[]"


def fake_embedder(text):
    return [0.1, 0.2, 0.3]


OBS = json.dumps([{"key_label": "basic.location", "value_text": "Lives in Rotonda West, Florida",
                   "context": None, "event_date": None, "privacy_level": "public",
                   "asserted_by": "self", "confidence": 0.9}])


# --- grouping ----------------------------------------------------------------


def test_groups_by_person_before_extraction():
    """Interleaved two-person chat -> two LLM calls, each seeing ONLY its person."""
    prod = FakeConn([("FROM n8n_chat_histories", [
        row(CHRIS, "I drive a Dodge Ram"),
        row(MEGAN, "I collect SNES games"),
        row(CHRIS, "and I live in Rotonda West"),
    ])])
    staging = FakeConn()
    llm = FakeLlm([OBS, OBS])
    run_extraction(prod, staging, llm, fake_embedder)

    assert len(llm.calls) == 2
    transcripts = [user for _, user in llm.calls]
    chris_t = next(t for t in transcripts if "Dodge Ram" in t)
    megan_t = next(t for t in transcripts if "SNES" in t)
    assert "SNES" not in chris_t and "Rotonda West" in chris_t  # no cross-person bleed
    assert "Dodge Ram" not in megan_t
    assert all(system == EXTRACTION_SYSTEM_PROMPT for system, _ in llm.calls)


# --- high-water mark ---------------------------------------------------------


def test_watermark_round_trips_raw_postgres_string():
    """The stored watermark is the DB's created_at::text VERBATIM — microseconds
    intact, no isoformat 'T', no JS-style ms truncation, no +1ms bump."""
    raw = "2026-07-03 01:02:03.123456+00"  # µs precision a JS Date would destroy
    prod = FakeConn([("FROM n8n_chat_histories",
                      [row(CHRIS, "hi", at=T0, raw=raw)])])
    staging = FakeConn()
    run_extraction(prod, staging, FakeLlm(["[]"]), fake_embedder)

    saves = [(s, p) for s, p in staging.calls if "v2_extraction_watermark" in s and "INSERT" in s]
    assert len(saves) == 1  # prod source only; v2_turns was empty -> untouched
    assert saves[0][1] == {"source": "prod_chat", "raw": raw}


def test_watermark_picks_newest_by_timestamp_not_string():
    """'.9' vs '.15' fractional seconds: string-max lies, datetime-max must win."""
    raw_old = "2026-07-03 12:00:00.9+00"     # string-"bigger", actually older
    raw_new = "2026-07-03 12:00:01.15+00"
    prod = FakeConn([("FROM n8n_chat_histories", [
        row(CHRIS, "a", at=T0.replace(microsecond=900000), raw=raw_old),
        row(CHRIS, "b", at=T0 + timedelta(seconds=1, microseconds=150000), raw=raw_new),
    ])])
    staging = FakeConn()
    run_extraction(prod, staging, FakeLlm(["[]"]), fake_embedder)
    save = next(p for s, p in staging.calls if "v2_extraction_watermark" in s and "INSERT" in s)
    assert save["raw"] == raw_new


# --- shadow-mode write boundary ----------------------------------------------


def test_prod_conn_gets_selects_only_and_writes_go_to_staging():
    prod = FakeConn([("FROM n8n_chat_histories", [row(CHRIS, "I live in Rotonda West")])])
    staging = FakeConn()
    run_extraction(prod, staging, FakeLlm([OBS]), fake_embedder)

    # READ-ONLY contract on prod: every statement is a SELECT/WITH.
    assert prod.calls, "prod source was queried"
    for sql, _ in prod.calls:
        assert sql.lstrip().upper().startswith(("SELECT", "WITH"))
    # Every INSERT went to the staging conn, and only at v2_* tables.
    inserts = [s for s, _ in staging.calls if s.lstrip().upper().startswith("INSERT")]
    assert inserts, "memory + watermark writes landed"
    assert all("v2_memories_staging" in s or "v2_extraction_watermark" in s for s in inserts)
    assert not any("INTO memories" in s for s, _ in staging.calls)  # never prod's table


def test_memory_created_at_is_message_time_and_provenance_flows():
    raw = "2026-07-01 08:30:00.000001+00"
    prod = FakeConn([("FROM n8n_chat_histories",
                      [row(CHRIS, "I live in Rotonda West", at=T0 - timedelta(days=2), raw=raw)])])
    staging = FakeConn()
    run_extraction(prod, staging, FakeLlm([OBS]), fake_embedder)

    params = next(p for s, p in staging.calls if "v2_memories_staging" in s)
    assert params["created_at"] == raw            # original message time, not now()
    assert params["source_thread"] == "v1:n8n_chat_histories"
    assert params["embedding"] == "[0.1,0.2,0.3]"  # pgvector text format
    assert params["content"].startswith("basic.location: Lives in Rotonda West")


# --- empty window ------------------------------------------------------------


def test_empty_window_is_a_no_op():
    prod, staging, llm = FakeConn(), FakeConn(), FakeLlm([])
    summary = run_extraction(prod, staging, llm, fake_embedder)
    assert llm.calls == []  # no LLM spend
    writes = [s for s, _ in staging.calls if s.lstrip().upper().startswith("INSERT")]
    assert writes == []  # no staging rows, watermark untouched
    assert summary["inserted_total"] == 0
    assert summary["sources"]["prod_chat"]["rows"] == 0
    assert summary["sources"]["v2_turns"]["rows"] == 0


def test_llm_returning_empty_array_stages_nothing():
    prod = FakeConn([("FROM n8n_chat_histories", [row(CHRIS, "brb grabbing coffee")])])
    staging = FakeConn()
    summary = run_extraction(prod, staging, FakeLlm(["[]"]), fake_embedder)
    assert not any("v2_memories_staging" in s for s, _ in staging.calls)
    # but the watermark STILL advances — "nothing worth remembering" is processed
    assert any("v2_extraction_watermark" in s and "INSERT" in s for s, _ in staging.calls)
    assert summary["inserted_total"] == 0


# --- v2_turns source ---------------------------------------------------------


def test_v2_turns_read_from_staging_conn_with_thread_provenance():
    staging = FakeConn([("FROM v2_turns", [
        row(CHRIS, "my Even G2 glasses arrive next week", platform="voice",
            privacy="private", thread="voice:beta", raw="2026-07-03 09:00:00.5+00"),
    ])])
    prod = FakeConn()
    run_extraction(prod, staging, FakeLlm([OBS]), fake_embedder)  # prod empty -> 1 call

    assert not any("v2_turns" in s for s, _ in prod.calls)  # her turns never hit prod
    params = next(p for s, p in staging.calls if "v2_memories_staging" in s)
    assert params["source_thread"] == "voice:beta"
    assert params["source_platform"] == "voice"
    wm = [p for s, p in staging.calls if "v2_extraction_watermark" in s and "INSERT" in s]
    assert wm == [{"source": "v2_turns", "raw": "2026-07-03 09:00:00.5+00"}]


# --- parsing helpers ---------------------------------------------------------


def test_parse_observations_strips_code_fences():
    fenced = "```json\n" + OBS + "\n```"
    assert parse_observations(fenced)[0]["key_label"] == "basic.location"


def test_parse_observations_tolerates_garbage():
    assert parse_observations("Sure! Here you go: oops no json") == []
    assert parse_observations('{"not": "an array"}') == []


def test_batch_date_renders_owner_timezone():
    # 2026-07-03 01:00 UTC is still July 2 in Florida (EDT, UTC-4)
    assert batch_date(datetime(2026, 7, 3, 1, 0, tzinfo=timezone.utc)) == "Jul 2"
