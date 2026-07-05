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
    LEASE_KIND,
    SOURCE_COLUMNS,
    LiveWriteRefused,
    _safe_watermark,
    _trim_tie_boundary,
    _value_text_from_content,
    acquire_write_mutex,
    batch_date,
    ensure_lease_holder,
    parse_observations,
    run_extraction,
    run_live_extraction,
    triage_memory,
    values_similar,
)

T0 = datetime(2026, 7, 3, 12, 0, 0, tzinfo=timezone.utc)
CHRIS = "6e6bcbed-03ef-4d17-95d2-89c467414335"
MEGAN = "11111111-2222-3333-4444-555555555555"


def row(person_id, content, *, id="1", at=T0, raw=None, speaker="Unknown",
        platform="discord", privacy="private", thread="v1:n8n_chat_histories"):
    """A source row in SOURCE_COLUMNS order — both queries return this shape.

    id defaults to "1" (fine when a test only ever has one row, or doesn't
    care about row identity) — pass DISTINCT ids for tests that need to track
    which specific rows landed in a parse-failed group (_safe_watermark keys
    off row ids)."""
    values = {
        "id": id,
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


# --- watermark safety: parse failures must never be skipped past -------------


def test_parse_failure_on_the_very_first_row_freezes_the_watermark():
    """A truncated LLM reply (parse_observations -> None, not []) for the
    EARLIEST group in the window: there is nothing safe to advance past, so
    the watermark must freeze at the pre-pass value, not jump to whatever
    else was in the batch — otherwise this message is gone forever (next
    pass's `created_at > watermark` would exclude it)."""
    existing_wm = "2026-07-03 11:00:00+00"
    raw_chris = "2026-07-03 12:00:00.000001+00"
    prod = FakeConn([("FROM n8n_chat_histories", [
        row(CHRIS, "I drive a Dodge Ram", id="a", at=T0, raw=raw_chris),
    ])])
    staging = FakeConn([("last_processed_at", [(existing_wm,)])])
    summary = run_extraction(prod, staging, FakeLlm(["not json at all -- truncated"]), fake_embedder)

    assert not any("v2_memories_staging" in s for s, _ in staging.calls)  # nothing staged
    saves = [(s, p) for s, p in staging.calls if "v2_extraction_watermark" in s and "INSERT" in s]
    assert len(saves) == 1
    assert saves[0][1] == {"source": "prod_chat", "raw": existing_wm}  # frozen, not advanced
    assert summary["sources"]["prod_chat"]["parse_failures"] == 1


def test_parse_failure_mid_batch_holds_watermark_but_stages_other_groups():
    """A truncated reply for ONE person's group must not sink the whole batch:
    a DIFFERENT person's group in the same window still parses and stages
    normally, and the watermark advances up to (but not past) the failure."""
    raw_a = "2026-07-03 12:00:00.000001+00"
    raw_b = "2026-07-03 12:00:02.000002+00"
    raw_c = "2026-07-03 12:00:05.000003+00"  # this person's group fails to parse
    PERSON_C = "99999999-8888-7777-6666-555555555555"
    prod = FakeConn([("FROM n8n_chat_histories", [
        row(CHRIS, "I drive a Dodge Ram", id="a", at=T0, raw=raw_a),
        row(MEGAN, "I collect SNES games", id="b", at=T0 + timedelta(seconds=2), raw=raw_b),
        row(PERSON_C, "mumble mumble", id="c", at=T0 + timedelta(seconds=5), raw=raw_c),
    ])])
    staging = FakeConn()
    llm = FakeLlm([OBS, OBS, "not json at all -- truncated"])
    summary = run_extraction(prod, staging, llm, fake_embedder)

    # both successful groups still staged — one failure doesn't block the rest
    inserts = [s for s, _ in staging.calls if "v2_memories_staging" in s]
    assert len(inserts) == 2
    save = next(p for s, p in staging.calls if "v2_extraction_watermark" in s and "INSERT" in s)
    assert save["raw"] == raw_b  # advances past both successes, stops before the failure
    assert summary["sources"]["prod_chat"]["parse_failures"] == 1


def test_parse_failure_never_reaches_staging_but_success_in_same_source_still_lands():
    """v2_turns source specifically: a garbled reply must not stage a blank/
    garbage row, and must not silently look identical to a "no memories" pass."""
    staging = FakeConn([("FROM v2_turns", [
        row(CHRIS, "static on the line", id="only", platform="voice",
            privacy="private", thread="voice:beta", raw="2026-07-03 09:00:00.5+00"),
    ])])
    prod = FakeConn()
    summary = run_extraction(prod, staging, FakeLlm(["<<garbled, not json>>"]), fake_embedder)

    assert not any("v2_memories_staging" in s for s, _ in staging.calls)
    assert summary["sources"]["v2_turns"]["parse_failures"] == 1
    assert summary["sources"]["v2_turns"]["observations"] == 0


# --- watermark safety: LIMIT-boundary ties must never be split ---------------


def test_limit_boundary_tie_is_trimmed_not_split():
    """batch_limit=2 -> the worker overfetches 3 (limit+1). All 3 come back
    (the overfetch is "full", so more rows might exist beyond it) and the
    LAST TWO share one microsecond-identical created_at — a real hazard,
    since now() is transaction-time. Splitting the tie (processing one,
    remembering a watermark that excludes its identical-timestamp sibling)
    would strand that sibling forever. The whole tied pair must be trimmed
    off THIS pass and left for the next one, together."""
    tie_raw = "2026-07-03 12:00:05.000000+00"
    tie_at = T0 + timedelta(seconds=5)
    prod = FakeConn([("FROM n8n_chat_histories", [
        row(CHRIS, "first", id="1", at=T0, raw="2026-07-03 12:00:00.000000+00"),
        row(MEGAN, "tied-a", id="2", at=tie_at, raw=tie_raw),
        row(CHRIS, "tied-b", id="3", at=tie_at, raw=tie_raw),
    ])])
    staging = FakeConn()
    summary = run_extraction(prod, staging, FakeLlm([OBS]), fake_embedder, batch_limit=2)

    # only the untied first row was processed this pass
    assert summary["sources"]["prod_chat"]["rows"] == 1
    inserts = [s for s, _ in staging.calls if "v2_memories_staging" in s]
    assert len(inserts) == 1
    save = next(p for s, p in staging.calls if "v2_extraction_watermark" in s and "INSERT" in s)
    assert save["raw"] == "2026-07-03 12:00:00.000000+00"  # NOT the tied timestamp
    assert save["raw"] != tie_raw

    # the overfetch actually asked for limit+1, proving the SQL round-trip:
    fetch_params = next(p for s, p in prod.calls if "FROM n8n_chat_histories" in s)
    assert fetch_params["limit"] == 3


def test_limit_boundary_no_tie_processes_the_full_overfetch_headroom():
    """If the batch does NOT fill the +1 overfetch, there's nothing beyond
    it and no tie risk — every row must be processed, none held back."""
    prod = FakeConn([("FROM n8n_chat_histories", [
        row(CHRIS, "only one row", id="1", at=T0, raw="2026-07-03 12:00:00.000000+00"),
    ])])
    staging = FakeConn()
    summary = run_extraction(prod, staging, FakeLlm([OBS]), fake_embedder, batch_limit=2)
    assert summary["sources"]["prod_chat"]["rows"] == 1
    assert summary["inserted_total"] == 1


def test_trim_tie_boundary_unit():
    rows = [{"id": "1", "created_at": 1}, {"id": "2", "created_at": 2}, {"id": "3", "created_at": 2}]
    assert _trim_tie_boundary(rows, limit=2) == [{"id": "1", "created_at": 1}]
    # under the overfetch (didn't fill limit+1): nothing trimmed
    assert _trim_tie_boundary(rows, limit=5) == rows
    # pathological: every row ties -> nothing safe to trim to, return as-is
    all_tied = [{"id": "1", "created_at": 1}, {"id": "2", "created_at": 1}]
    assert _trim_tie_boundary(all_tied, limit=1) == all_tied


def test_safe_watermark_unit():
    rows = [
        {"id": "a", "created_at": 1, "created_at_raw": "a-raw"},
        {"id": "b", "created_at": 2, "created_at_raw": "b-raw"},
        {"id": "c", "created_at": 3, "created_at_raw": "c-raw"},
    ]
    # no failures: newest wins, exactly as before
    assert _safe_watermark(rows, set(), after="after-raw") == "c-raw"
    # failure mid-list: freeze at the newest row strictly older than it
    assert _safe_watermark(rows, {"b"}, after="after-raw") == "a-raw"
    # failure on the very first row: nothing older -> the pre-pass value
    assert _safe_watermark(rows, {"a"}, after="after-raw") == "after-raw"


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


def test_parse_observations_distinguishes_failure_from_genuine_empty():
    # Genuine "nothing worth remembering": the documented `[]` reply.
    assert parse_observations("[]") == []
    # Unparseable (e.g. max_tokens=1200 truncation mid-object on a busy
    # transcript) and a non-array top-level value are BOTH a parse FAILURE —
    # None, never [] — so the caller can tell "no memories" apart from
    # "couldn't tell" and hold the watermark back for the latter.
    assert parse_observations("Sure! Here you go: oops no json") is None
    assert parse_observations('{"not": "an array"}') is None
    assert parse_observations('[{"key_label": "basic.location", "value_te') is None  # truncated mid-object
    assert parse_observations("null") is None
    assert parse_observations('"just a string"') is None


def test_batch_date_renders_owner_timezone():
    # 2026-07-03 01:00 UTC is still July 2 in Florida (EDT, UTC-4)
    assert batch_date(datetime(2026, 7, 3, 1, 0, tzinfo=timezone.utc)) == "Jul 2"


# --- live-mode triage (prod `memories`) ---------------------------------------


def lease_conn(holder=None, *, mutex_acquired=True, routes=()):
    """A FakeConn pre-wired with a v2_writer_lease answer AND an
    advisory-mutex answer. holder=None means "no row yet" (ensure_lease_holder
    seeds + defaults to 'n8n'). mutex_acquired controls the THIRD gate
    (acquire_write_mutex / pg_try_advisory_xact_lock) — set False to simulate
    a second 'brain' process already holding the write mutex."""
    all_routes = [
        ("FROM v2_writer_lease", [(holder,)] if holder else []),
        ("pg_try_advisory_xact_lock", [(mutex_acquired,)]),
    ]
    all_routes.extend(routes)
    return FakeConn(all_routes)


def test_values_similar_exact_and_fuzzy_and_different():
    assert values_similar("Lives in Rotonda West, Florida", "Lives in Rotonda West, Florida")
    assert values_similar("Lives in Rotonda West, Florida", "lives in rotonda west, florida")  # case
    assert values_similar("Interested in a Dodge Ram truck", "Interested in a Dodge Ram")  # restated
    assert not values_similar("Lives in Rotonda West, Florida", "Lives in Austin, Texas")  # genuinely new


def test_value_text_from_content_strips_label_context_and_date():
    content = "basic.location: Lives in Rotonda West, Florida -- came up casually (Jul 2)"
    assert _value_text_from_content(content) == "Lives in Rotonda West, Florida"
    # no context, no date
    assert _value_text_from_content("basic.location: Lives in Rotonda West, Florida") == \
        "Lives in Rotonda West, Florida"


def test_triage_inserts_when_no_existing_row():
    conn = FakeConn([("FROM memories", [])])  # dedup check finds nothing live
    action = triage_memory(
        conn, person_id=CHRIS, key_label="basic.location", value_text="Lives in Rotonda West",
        content="basic.location: Lives in Rotonda West", context=None, event_date=None,
        embedding="[0.1,0.2,0.3]", source_platform="discord", privacy_level="public",
        created_at="2026-07-03 12:00:00+00",
    )
    assert action == "insert"
    inserts = [s for s, _ in conn.calls if s.lstrip().upper().startswith("INSERT")]
    assert len(inserts) == 1
    assert not any(s.lstrip().upper().startswith("UPDATE") for s, _ in conn.calls)


def test_triage_updates_when_value_restated():
    conn = FakeConn([("FROM memories", [("existing-id", "basic.location: Lives in Rotonda West")])])
    action = triage_memory(
        conn, person_id=CHRIS, key_label="basic.location", value_text="Lives in Rotonda West, FL",
        content="basic.location: Lives in Rotonda West, FL", context=None, event_date=None,
        embedding="[0.4,0.5,0.6]", source_platform="discord", privacy_level="public",
        created_at="2026-07-03 12:00:00+00",
    )
    assert action == "update"
    updates = [(s, p) for s, p in conn.calls if s.lstrip().upper().startswith("UPDATE")]
    assert len(updates) == 1
    assert updates[0][1]["id"] == "existing-id"
    assert updates[0][1]["embedding"] == "[0.4,0.5,0.6]"  # fresh embedding rides the UPDATE
    assert not any(s.lstrip().upper().startswith("INSERT") for s, _ in conn.calls)


def test_triage_replaces_when_value_changed():
    conn = FakeConn([("FROM memories", [("old-id", "basic.location: Lives in Rotonda West")])])
    action = triage_memory(
        conn, person_id=CHRIS, key_label="basic.location", value_text="Lives in Austin, Texas",
        content="basic.location: Lives in Austin, Texas", context=None, event_date=None,
        embedding="[0.7,0.8,0.9]", source_platform="discord", privacy_level="public",
        created_at="2026-07-03 12:00:00+00",
    )
    assert action == "replace"
    # atomic soft-delete + insert: ONE statement, the CTE, referencing old_id and
    # inserting the new row — never a bare UPDATE/DELETE plus a separate INSERT.
    replace_calls = [(s, p) for s, p in conn.calls if "soft_del" in s]
    assert len(replace_calls) == 1
    assert replace_calls[0][1]["old_id"] == "old-id"
    assert replace_calls[0][1]["content"] == "basic.location: Lives in Austin, Texas"
    assert not any(
        s.lstrip().upper().startswith("UPDATE") and "soft_del" not in s for s, _ in conn.calls
    )


# --- triage_memory: refuse to corrupt, never touch the DB when unusable ------


def test_triage_skips_blank_value_text_never_reaches_replace():
    # Cross-review bug: a blank value_text scored 0.0 similarity against ANY
    # existing value, forcing REPLACE — soft-deleting the real memory and
    # inserting a blank "key_label: " row in its place.
    conn = FakeConn([("FROM memories", [("existing-id", "basic.location: Lives in Rotonda West")])])
    action = triage_memory(
        conn, person_id=CHRIS, key_label="basic.location", value_text="",
        content="basic.location: ", context=None, event_date=None,
        embedding="[0.1,0.2,0.3]", source_platform="discord", privacy_level="public",
        created_at="2026-07-03 12:00:00+00",
    )
    assert action == "skipped"
    assert conn.calls == []  # refused before even the dedup SELECT — zero writes, zero reads


def test_triage_skips_whitespace_only_value_text():
    conn = FakeConn([("FROM memories", [("existing-id", "basic.location: Lives in Rotonda West")])])
    action = triage_memory(
        conn, person_id=CHRIS, key_label="basic.location", value_text="   ",
        content="basic.location:    ", context=None, event_date=None,
        embedding="[0.1,0.2,0.3]", source_platform="discord", privacy_level="public",
        created_at="2026-07-03 12:00:00+00",
    )
    assert action == "skipped"
    assert conn.calls == []


def test_triage_skips_null_key_label_unusable_for_dedup():
    # Cross-review bug: `key_label = NULL` is never true in SQL, so the dedup
    # SELECT always misses -> every observation without a key_label fell to
    # INSERT -> unbounded duplicate rows.
    conn = FakeConn([("FROM memories", [])])
    action = triage_memory(
        conn, person_id=CHRIS, key_label=None, value_text="some fact with no label",
        content=": some fact with no label", context=None, event_date=None,
        embedding="[0.1,0.2,0.3]", source_platform="discord", privacy_level="public",
        created_at="2026-07-03 12:00:00+00",
    )
    assert action == "skipped"
    assert conn.calls == []


def test_triage_skips_blank_key_label():
    conn = FakeConn([("FROM memories", [])])
    action = triage_memory(
        conn, person_id=CHRIS, key_label="   ", value_text="some fact",
        content="   : some fact", context=None, event_date=None,
        embedding="[0.1,0.2,0.3]", source_platform="discord", privacy_level="public",
        created_at="2026-07-03 12:00:00+00",
    )
    assert action == "skipped"
    assert conn.calls == []


# --- triage_memory: privacy_level tightens on UPDATE, never loosens ---------


def test_triage_update_writes_privacy_level_param_for_sql_side_tightening():
    """The UPDATE now carries privacy_level through, with a CASE that only
    ever tightens (public -> private) — a re-statement flagged private must
    not leave the existing row silently public in retrieval. The CASE logic
    itself lives in SQL (no real DB here to execute it against), so what's
    provable offline is the SHAPE: the column is in the SET list and the
    incoming privacy_level rides the params, every single time."""
    conn = FakeConn([("FROM memories", [("existing-id", "basic.location: Lives in Rotonda West")])])
    action = triage_memory(
        conn, person_id=CHRIS, key_label="basic.location", value_text="Lives in Rotonda West, FL",
        content="basic.location: Lives in Rotonda West, FL", context=None, event_date=None,
        embedding="[0.4,0.5,0.6]", source_platform="discord", privacy_level="private",
        created_at="2026-07-03 12:00:00+00",
    )
    assert action == "update"
    sql, params = next((s, p) for s, p in conn.calls if s.lstrip().upper().startswith("UPDATE"))
    assert "privacy_level" in sql
    assert "CASE" in sql.upper()  # tightening logic, not a blind overwrite
    assert params["privacy_level"] == "private"


def test_triage_update_case_only_tightens_toward_private_in_sql_text():
    # The CASE must resolve to 'private' when the incoming value is 'private',
    # and otherwise fall back to the EXISTING row's privacy_level column
    # (never blindly assign the incoming 'public') — asserted on the SQL text
    # itself since there's no real Postgres here to execute the CASE against.
    conn = FakeConn([("FROM memories", [("existing-id", "basic.location: Lives in Rotonda West")])])
    triage_memory(
        conn, person_id=CHRIS, key_label="basic.location", value_text="Lives in Rotonda West, FL",
        content="basic.location: Lives in Rotonda West, FL", context=None, event_date=None,
        embedding="[0.4,0.5,0.6]", source_platform="discord", privacy_level="public",
        created_at="2026-07-03 12:00:00+00",
    )
    sql, params = next((s, p) for s, p in conn.calls if s.lstrip().upper().startswith("UPDATE"))
    assert "THEN 'private'" in sql
    assert "ELSE privacy_level" in sql  # falls back to the EXISTING column, not the incoming value
    assert params["privacy_level"] == "public"


# --- live-mode hard gates ------------------------------------------------------


def test_live_refuses_when_lease_held_by_n8n():
    prod, staging, prod_write = FakeConn(), lease_conn(holder="n8n"), FakeConn()
    try:
        run_live_extraction(prod, staging, prod_write, FakeLlm([]), fake_embedder)
        assert False, "expected LiveWriteRefused"
    except LiveWriteRefused as e:
        assert LEASE_KIND in str(e)
    assert prod.calls == []
    assert prod_write.calls == []


def test_live_refuses_and_seeds_lease_row_when_absent():
    """No memory_extraction row yet (migration 001 seeded only 4 kinds) —
    ensure_lease_holder seeds it as 'n8n' (the pre-flip default) and live mode
    refuses exactly as if n8n already held it explicitly."""
    staging = lease_conn(holder=None)
    try:
        run_live_extraction(FakeConn(), staging, FakeConn(), FakeLlm([]), fake_embedder)
        assert False, "expected LiveWriteRefused"
    except LiveWriteRefused as e:
        assert "n8n" in str(e)
    seed_calls = [
        (s, p) for s, p in staging.calls
        if "v2_writer_lease" in s and s.lstrip().upper().startswith("INSERT")
    ]
    assert seed_calls, "the lease row was seeded on first touch"
    assert seed_calls[0][1] == {"kind": LEASE_KIND}


def test_ensure_lease_holder_returns_brain_without_seeding_twice():
    staging = lease_conn(holder="brain")
    assert ensure_lease_holder(staging) == "brain"


def test_acquire_write_mutex_true_and_false():
    assert acquire_write_mutex(lease_conn(holder="brain", mutex_acquired=True)) is True
    assert acquire_write_mutex(lease_conn(holder="brain", mutex_acquired=False)) is False


def test_live_refuses_when_write_mutex_already_held():
    """The lease ROW says 'brain' is authorized (gate #2 passes) but a SECOND
    brain process (the loop container, say, racing a manual --once --live)
    already holds the actual write mutex — this pass must refuse rather than
    race it. This is the gate ensure_lease_holder alone can't provide."""
    prod, prod_write = FakeConn(), FakeConn()
    staging = lease_conn(holder="brain", mutex_acquired=False)
    try:
        run_live_extraction(prod, staging, prod_write, FakeLlm([]), fake_embedder)
        assert False, "expected LiveWriteRefused"
    except LiveWriteRefused as e:
        assert LEASE_KIND in str(e)
    # refused BEFORE any read/write, same posture as the other two gates.
    assert prod.calls == []
    assert prod_write.calls == []
    assert not any("v2_extraction_watermark" in s for s, _ in staging.calls)


def test_live_proceeds_and_triages_when_lease_held_by_brain_and_n8n_inactive():
    prod = FakeConn([("FROM n8n_chat_histories", [row(CHRIS, "I live in Rotonda West")])])
    staging = lease_conn(holder="brain")
    prod_write = FakeConn([("FROM memories", [])])  # nothing existing -> insert
    summary = run_live_extraction(
        prod, staging, prod_write, FakeLlm([OBS]), fake_embedder,
    )
    assert summary["inserted_total"] == 1
    assert summary["updated_total"] == 0
    assert summary["replaced_total"] == 0
    inserts = [s for s, _ in prod_write.calls if s.lstrip().upper().startswith("INSERT")]
    assert inserts, "the triaged write landed on prod_write_conn, not staging_conn"
    assert not any("v2_memories_staging" in s for s, _ in staging.calls)  # never shadow-staged
    # watermark still advances, same table/mechanism as shadow mode
    assert any("v2_extraction_watermark" in s and "INSERT" in s for s, _ in staging.calls)


def test_live_computes_fresh_embedding_per_write_not_cached():
    """Two observations from one LLM reply -> two DIFFERENT embedder calls,
    each embedding riding its own write — never one embedding reused."""
    calls = {"n": 0}

    def counting_embedder(text):
        calls["n"] += 1
        return [float(calls["n"])]

    two_obs = json.dumps([
        {"key_label": "basic.location", "value_text": "Lives in Rotonda West", "context": None,
         "event_date": None, "privacy_level": "public", "asserted_by": "self", "confidence": 0.9},
        {"key_label": "interest.game", "value_text": "Collects SNES games", "context": None,
         "event_date": None, "privacy_level": "public", "asserted_by": "self", "confidence": 0.9},
    ])
    prod = FakeConn([("FROM n8n_chat_histories", [row(CHRIS, "I live in Rotonda West and collect SNES")])])
    staging = lease_conn(holder="brain")
    prod_write = FakeConn([("FROM memories", [])])
    run_live_extraction(prod, staging, prod_write, FakeLlm([two_obs]), counting_embedder)

    assert calls["n"] == 2  # one fresh embed call per observation, not one for the batch
    inserts = [p for s, p in prod_write.calls if s.lstrip().upper().startswith("INSERT")]
    embeddings = {p["embedding"] for p in inserts}
    assert embeddings == {"[1.0]", "[2.0]"}  # two distinct fresh vectors, not a shared/cached one


def test_live_extraction_skips_unusable_observations_and_counts_them():
    """A group whose reply contains one observation with NO key_label and one
    with a BLANK value_text: both are unusable for dedup/triage and must land
    ZERO writes in prod `memories` — but they still show up in the summary's
    skipped tally, not silently vanish as if nothing was extracted at all."""
    bad_obs = json.dumps([
        {"key_label": None, "value_text": "no label at all", "context": None,
         "event_date": None, "privacy_level": "public", "asserted_by": "self", "confidence": 0.9},
        {"key_label": "basic.location", "value_text": "", "context": None,
         "event_date": None, "privacy_level": "public", "asserted_by": "self", "confidence": 0.9},
    ])
    prod = FakeConn([("FROM n8n_chat_histories", [row(CHRIS, "mumble mumble")])])
    staging = lease_conn(holder="brain")
    prod_write = FakeConn([("FROM memories", [])])
    summary = run_live_extraction(prod, staging, prod_write, FakeLlm([bad_obs]), fake_embedder)

    assert summary["inserted_total"] == 0
    assert summary["updated_total"] == 0
    assert summary["replaced_total"] == 0
    assert summary["skipped_total"] == 2
    assert summary["sources"]["prod_chat"]["skipped"] == 2
    assert not any(
        s.lstrip().upper().startswith(("INSERT", "UPDATE")) for s, _ in prod_write.calls
    )
