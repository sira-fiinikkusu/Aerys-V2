"""Offline tests for the capability-request miner (self-iteration, Phase A).

No Postgres, no network — same seam philosophy as test_extraction / test_turns.
The load-bearing test is the un-forgeable-provenance one: a reply whose TEXT is
full of failure words, but whose STRUCTURED fields (tool_calls/degraded) are clean,
must classify as a 'complaint' (stricter gate), NEVER an 'error'. That is the whole
security model — the model authors reply text, it cannot author a degraded marker or
a tool_calls failure.
"""

import pathlib
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone

import pytest

import aerys_v2.workers.capability_requests as cap
from aerys_v2.workers.capability_requests import (
    EXCERPT_LIMIT,
    GAPS_COLUMNS,
    MINER_COLUMNS,
    GapMiningRefused,
    GapSignal,
    _bounded_excerpt,
    _complaint_signals,
    _error_signals,
    classify_turn,
    format_gaps,
    read_gaps,
    run_gap_mining,
)

T0 = datetime(2026, 7, 3, 12, 0, 0, tzinfo=timezone.utc)
CHRIS = "6e6bcbed-03ef-4d17-95d2-89c467414335"
MEGAN = "11111111-2222-3333-4444-555555555555"


def _raw(at: datetime) -> str:
    return at.strftime("%Y-%m-%d %H:%M:%S.%f+00")


def turn(
    *, id="1", person_id=CHRIS, at=T0, raw=None, thread="discord:dm:1", channel="discord_dm",
    raw_reply="ok", emitted_reply=None, tool_calls=None, degraded=None, error=None,
) -> dict:
    """A turn dict as run_gap_mining sees it after dict(zip(MINER_COLUMNS, ...))."""
    return {
        "id": id,
        "person_id": person_id,
        "created_at": at,
        "created_at_raw": raw or _raw(at),
        "thread_id": thread,
        "channel": channel,
        "raw_reply": raw_reply,
        "emitted_reply": emitted_reply if emitted_reply is not None else raw_reply,
        "tool_calls": tool_calls if tool_calls is not None else [],
        "degraded": degraded if degraded is not None else [],
        "error": error,
    }


def turn_tuple(**kw) -> tuple:
    """The same turn as a MINER_COLUMNS-ordered tuple (what the SQL fetch returns)."""
    d = turn(**kw)
    return tuple(d[c] for c in MINER_COLUMNS)


class FakeCursor:
    def __init__(self, rows):
        self.rows = rows

    def fetchone(self):
        return self.rows[0] if self.rows else None

    def fetchall(self):
        return self.rows


class GapConn:
    """Stateful duck-typed connection: models the (fingerprint, turn_id) uniqueness
    and how_often=COUNT(*)-over-child semantics the real tables enforce, so dedup and
    recurrence are proven end-to-end offline (not just asserted on emitted SQL).

    Routes by SQL substring (most-specific first). `raise_on_fingerprint` simulates a
    DB error mid-record for one signal, to exercise the failed-row watermark hold.
    """

    def __init__(self, *, turns=(), parity=(10, 5), watermark=None, raise_on_fingerprint=None):
        self._turns = list(turns)
        self._parity = parity
        self._watermark = watermark
        self._raise_on = raise_on_fingerprint
        self.examples: set[tuple[str, str]] = set()   # {(fingerprint, turn_id_str)}
        self.parents: dict[str, dict] = {}
        self.saved_watermark = None
        self.miner_params = None
        self.calls = []

    @contextmanager
    def transaction(self):
        """Models a psycopg SAVEPOINT: on exception, roll this turn's writes back to
        the snapshot and re-raise, leaving the outer 'transaction' usable."""
        snap_examples = set(self.examples)
        snap_parents = {k: dict(v) for k, v in self.parents.items()}
        try:
            yield
        except Exception:
            self.examples = snap_examples
            self.parents = snap_parents
            raise

    def execute(self, sql, params=None):
        p = params or {}
        self.calls.append((sql, p))
        if "count(*) FILTER" in sql:                                  # PARITY_SQL
            return FakeCursor([self._parity])
        if "FROM v2_turns t" in sql:                                  # MINER_SQL
            self.miner_params = p
            return FakeCursor(list(self._turns))
        if "SELECT last_processed_at FROM v2_extraction_watermark" in sql:
            return FakeCursor([(self._watermark,)] if self._watermark else [])
        if "INSERT INTO v2_extraction_watermark" in sql:             # save_watermark
            self.saved_watermark = p["raw"]
            self._watermark = p["raw"]
            return FakeCursor([])
        if "INSERT INTO v2_capability_request_examples" in sql:       # EXAMPLE_INSERT
            if self._raise_on is not None and p["fingerprint"] == self._raise_on:
                raise RuntimeError("simulated DB failure mid-record")
            key = (p["fingerprint"], str(p["turn_id"]))
            if key in self.examples:
                return FakeCursor([])            # ON CONFLICT DO NOTHING -> no RETURNING row
            self.examples.add(key)
            return FakeCursor([(p["turn_id"],)])
        if "INSERT INTO v2_capability_requests" in sql:              # PARENT_UPSERT
            fp = p["fingerprint"]
            count = sum(1 for (f, _t) in self.examples if f == fp)
            if fp in self.parents:
                row = self.parents[fp]
                row["how_often"] = count                             # COUNT(*)-over-child
                row["last_seen_at"] = max(row["last_seen_at"], p["last_seen_at"])
                # status/origin_class/summary/required_tier UNTOUCHED (ON CONFLICT rule)
            else:
                self.parents[fp] = {
                    "fingerprint": fp,
                    "signal_kind": p["signal_kind"],
                    "origin_class": p["origin_class"],
                    "summary": p["summary"],
                    "origin_trust": p["origin_trust"],
                    "required_tier": p["required_tier"],
                    "how_often": count,
                    "first_seen_at": p["first_seen_at"],
                    "last_seen_at": p["last_seen_at"],
                    "status": "open",
                }
            return FakeCursor([])
        return FakeCursor([])


# ── classification: the two detectors read DISJOINT columns ──────────────────


def test_degraded_marker_classifies_error():
    sigs = classify_turn(turn(degraded=["ha_unreachable"]))
    assert len(sigs) == 1
    s = sigs[0]
    assert s.signal_kind == "degraded"
    assert s.origin_class == "error"
    assert s.fingerprint == "degraded::ha_unreachable"
    assert s.required_tier == "standard"
    assert "ha_unreachable" in s.summary


def test_tool_failure_classifies_error():
    sigs = classify_turn(
        turn(tool_calls=[{"name": "search_web", "ok": False, "error_class": "timeout"}])
    )
    assert len(sigs) == 1
    s = sigs[0]
    assert s.signal_kind == "tool_error"
    assert s.origin_class == "error"
    assert s.fingerprint == "tool_error::search_web::timeout"
    assert s.required_tier == "standard"
    assert s.summary == "tool 'search_web' failed (timeout)"


def test_successful_tool_is_not_a_signal():
    sigs = classify_turn(
        turn(tool_calls=[{"name": "search_web", "ok": True, "error_class": None}],
             raw_reply="Here's what I found.")
    )
    assert sigs == []


def test_reply_phrase_classifies_complaint():
    sigs = classify_turn(
        turn(raw_reply="I'd love to help, but I don't have a tool for booking flights.")
    )
    assert len(sigs) == 1
    s = sigs[0]
    assert s.signal_kind == "reply_phrase"
    assert s.origin_class == "complaint"
    assert s.fingerprint == "reply_phrase::i don't have a tool for"
    assert s.required_tier == "stringent"
    assert "booking flights" in s.summary


def test_clean_structured_reply_with_failure_words_is_complaint_not_error():
    """THE security property. A reply whose TEXT is full of failure words but whose
    STRUCTURED fields are clean must NEVER classify as 'error' — only the machine-set
    tool_calls/degraded can mint an 'error', and here they're empty. The fakeable
    phrase path can only ever produce a 'complaint' on the stricter gate.

    An attacker who injects "the tool failed and the connection timed out" into a
    reply cannot upgrade its trust level: they author the text, not the label."""
    t = turn(
        raw_reply=(
            "Honestly the web search failed me and the connection timed out — "
            "the tool errored out. I don't have a tool for that yet."
        ),
        tool_calls=[],   # structured fields CLEAN
        degraded=[],
    )
    sigs = classify_turn(t)

    # No error is minted from reply text, no matter what words it contains.
    assert all(s.origin_class != "error" for s in sigs)
    # The structural detector never even READS reply text — it sees nothing here.
    assert _error_signals(t) == []
    # The phrase is caught, but as a complaint forced onto the stringent gate.
    complaints = [s for s in sigs if s.origin_class == "complaint"]
    assert complaints
    assert all(s.required_tier == "stringent" for s in complaints)
    assert all(s.signal_kind == "reply_phrase" for s in complaints)


def test_error_detector_ignores_reply_text_entirely():
    """Belt-and-braces on the boundary: _error_signals reads only structured fields.
    Same failure-word reply, no structured signal -> zero error signals."""
    t = turn(raw_reply="everything failed, timeout, unreachable, degraded, error",
             tool_calls=[], degraded=[])
    assert _error_signals(t) == []


def test_complaint_detector_ignores_structured_fields_entirely():
    """The mirror: _complaint_signals reads only reply text, never tool_calls/degraded.
    A real tool failure with a benign reply yields NO complaint (it's an error's job)."""
    t = turn(raw_reply="All set!", emitted_reply="All set!",
             tool_calls=[{"name": "x", "ok": False, "error_class": "timeout"}],
             degraded=["ha_unreachable"])
    assert _complaint_signals(t) == []


def test_required_tier_is_derived_from_origin_class():
    assert GapSignal("degraded", "error", "fp", "s").required_tier == "standard"
    assert GapSignal("reply_phrase", "complaint", "fp", "s").required_tier == "stringent"


def test_fingerprint_is_kind_and_cause_specific():
    tmo = classify_turn(turn(tool_calls=[{"name": "ha", "ok": False, "error_class": "timeout"}]))
    aut = classify_turn(turn(tool_calls=[{"name": "ha", "ok": False, "error_class": "auth_error"}]))
    # Same tool, different cause -> different fingerprints (they need different fixes).
    assert tmo[0].fingerprint != aut[0].fingerprint
    # Same marker across turns -> one recurring fingerprint.
    a = classify_turn(turn(id="1", degraded=["ha_unreachable"]))
    b = classify_turn(turn(id="2", degraded=["ha_unreachable"]))
    assert a[0].fingerprint == b[0].fingerprint


def test_complaint_summary_is_bounded_and_sanitized():
    dirty = "line1\nline2\tI wish I could " + ("x" * 400)
    s = classify_turn(turn(raw_reply=dirty))[0]
    assert "\n" not in s.summary and "\t" not in s.summary
    assert len(s.summary) <= EXCERPT_LIMIT
    assert s.summary.endswith("…")


def test_bounded_excerpt_short_text_unchanged():
    assert _bounded_excerpt("just a short line") == "just a short line"


def test_multiple_distinct_phrases_yield_one_signal_each():
    sigs = classify_turn(
        turn(raw_reply="I don't have a tool for that, and honestly I wish I could.")
    )
    fps = {s.fingerprint for s in sigs}
    assert fps == {"reply_phrase::i don't have a tool for", "reply_phrase::i wish i could"}


def test_malformed_structured_fields_coerce_to_empty():
    # None / non-list / non-dict entries degrade to nothing rather than crashing.
    assert _error_signals(turn(tool_calls=None, degraded="garbage")) == []
    assert _error_signals(turn(tool_calls=["not-a-dict"], degraded=[""])) == []


def test_nonlist_structured_field_warns_but_stays_safe(caplog):
    """A structured column that decoded to a non-list (str/dict — a jsonb decoding
    regression) must still coerce to [] (never crash, never fabricate a gap) AND emit
    a WARNING, so the silent structural-error under-report becomes visible. None stays
    quiet (expected pre-writer / JSON null)."""
    import logging

    with caplog.at_level(logging.WARNING, logger=cap.__name__):
        # tool_calls is a bare string (decoded wrong); degraded=None coerces to [] in
        # the fixture — so exactly one column trips the warning.
        assert _error_signals(turn(tool_calls="not-json", degraded=None)) == []
    ours = [r for r in caplog.records if r.name == cap.__name__]
    assert any("decoding regression" in r.getMessage() for r in ours)

    caplog.clear()
    with caplog.at_level(logging.WARNING, logger=cap.__name__):
        assert _error_signals(turn(tool_calls=None, degraded=None)) == []
    assert [r for r in caplog.records if r.name == cap.__name__] == []   # None never warns


def test_error_then_complaint_ordering_in_one_turn():
    """A turn can carry both an un-forgeable error AND a complaint; both surface,
    error first, each with its own machine-set label — the error is never
    contaminated by the complaint's fakeable provenance."""
    sigs = classify_turn(
        turn(degraded=["ha_unreachable"],
             raw_reply="HA is unreachable and I don't have a way to reach it.")
    )
    assert [s.origin_class for s in sigs] == ["error", "complaint"]


# ── run_gap_mining: gates, dedup, watermark ──────────────────────────────────


def test_empty_allowlist_refuses():
    with pytest.raises(GapMiningRefused, match="allowlist"):
        run_gap_mining(GapConn(), allowlist=[])


def test_parity_gate_refuses_when_no_armed_rows():
    conn = GapConn(parity=(20, 0), turns=[turn_tuple(degraded=["ha_unreachable"])])
    with pytest.raises(GapMiningRefused, match="writer has not landed"):
        run_gap_mining(conn, allowlist={CHRIS})
    assert conn.examples == set()   # refused before any write


def test_parity_gate_refuses_on_empty_turns_table():
    with pytest.raises(GapMiningRefused, match="no recent rows"):
        run_gap_mining(GapConn(parity=(0, 0)), allowlist={CHRIS})


def test_autocommit_connection_refuses_before_any_write():
    """The per-turn SAVEPOINT isolation requires a single outer transaction; under
    autocommit one poison row would abort the batch. Refuse loudly, write nothing."""
    conn = GapConn(turns=[turn_tuple(degraded=["ha_unreachable"])])
    conn.autocommit = True
    with pytest.raises(GapMiningRefused, match="autocommit"):
        run_gap_mining(conn, allowlist={CHRIS})
    assert conn.examples == set()   # refused before touching anything


def test_owner_filter_scopes_the_fetch():
    conn = GapConn(turns=[])
    run_gap_mining(conn, allowlist={CHRIS, MEGAN})
    assert conn.miner_params["person_ids"] == sorted([CHRIS, MEGAN])


def test_empty_window_is_a_noop():
    conn = GapConn(turns=[])
    stats = run_gap_mining(conn, allowlist={CHRIS})
    assert stats["turns"] == 0
    assert conn.examples == set()
    assert conn.saved_watermark is None   # watermark untouched on an empty window


def test_single_error_turn_records_signal_and_advances_watermark():
    t = turn_tuple(id="7", degraded=["ha_unreachable"], at=T0)
    conn = GapConn(turns=[t])
    stats = run_gap_mining(conn, allowlist={CHRIS})
    assert stats["signals"] == 1 and stats["errors"] == 1
    assert ("degraded::ha_unreachable", "7") in conn.examples
    assert conn.parents["degraded::ha_unreachable"]["how_often"] == 1
    assert conn.saved_watermark == _raw(T0)


def test_recurrence_across_distinct_turns_increments_how_often():
    """Two distinct turns, same fingerprint -> how_often = COUNT over the child = 2."""
    conn = GapConn(turns=[
        turn_tuple(id="1", degraded=["ha_unreachable"], at=T0),
        turn_tuple(id="2", degraded=["ha_unreachable"], at=T0 + timedelta(minutes=1)),
    ])
    run_gap_mining(conn, allowlist={CHRIS})
    assert conn.parents["degraded::ha_unreachable"]["how_often"] == 2
    assert len(conn.examples) == 2


def test_same_turn_reprocessed_counts_once():
    """Crash-retry safety: re-mining the SAME turn (the fake returns it both passes)
    hits the (fingerprint, turn_id) PK -> one example, how_often stays 1."""
    conn = GapConn(turns=[turn_tuple(id="9", degraded=["ha_unreachable"])])
    run_gap_mining(conn, allowlist={CHRIS})
    run_gap_mining(conn, allowlist={CHRIS})
    assert len(conn.examples) == 1
    assert conn.parents["degraded::ha_unreachable"]["how_often"] == 1


def test_terminal_status_is_not_resurrected_but_still_counts():
    """A rejected gap keeps counting recurrences but never flips back to 'open',
    and its diagnosis is never clobbered — the ON CONFLICT touches only observation
    fields."""
    fp = "degraded::ha_unreachable"
    conn = GapConn(turns=[turn_tuple(id="2", degraded=["ha_unreachable"], at=T0 + timedelta(minutes=1))])
    # Seed a pre-existing terminal parent + its first example.
    conn.examples.add((fp, "1"))
    conn.parents[fp] = {
        "fingerprint": fp, "signal_kind": "degraded", "origin_class": "error",
        "summary": "degraded subsystem marker: ha_unreachable", "origin_trust": "owner",
        "required_tier": "standard", "how_often": 1,
        "first_seen_at": _raw(T0), "last_seen_at": _raw(T0), "status": "rejected",
    }
    run_gap_mining(conn, allowlist={CHRIS})
    assert conn.parents[fp]["status"] == "rejected"       # NOT resurrected
    assert conn.parents[fp]["how_often"] == 2             # but still counts


def test_tie_boundary_is_trimmed():
    """A full fetch whose cut line ties on created_at trims the whole tied run, so
    the watermark can't strand a tied sibling (inherited _trim_tie_boundary)."""
    tied = T0 + timedelta(minutes=5)
    conn = GapConn(turns=[
        turn_tuple(id="1", degraded=["a"], at=T0),
        turn_tuple(id="2", degraded=["b"], at=tied),
        turn_tuple(id="3", degraded=["c"], at=tied),   # ties with #2 at the boundary
    ])
    run_gap_mining(conn, allowlist={CHRIS}, batch_limit=2)
    # Only turn #1 survives the trim; the tied pair waits for the next pass.
    assert set(conn.examples) == {("degraded::a", "1")}
    assert conn.saved_watermark == _raw(T0)


def test_failed_row_holds_the_watermark_below_it():
    """A turn whose record throws is HELD: the watermark freezes below it so the next
    pass retries it, exactly like extraction's parse-failure hold."""
    conn = GapConn(
        turns=[
            turn_tuple(id="1", degraded=["ha_unreachable"], at=T0),
            turn_tuple(id="2", degraded=["BOOM"], at=T0 + timedelta(minutes=1)),
        ],
        raise_on_fingerprint="degraded::BOOM",
    )
    stats = run_gap_mining(conn, allowlist={CHRIS})
    assert stats["processing_failures"] == 1
    assert ("degraded::ha_unreachable", "1") in conn.examples   # good turn landed
    assert conn.saved_watermark == _raw(T0)                     # frozen below the bad turn


def test_miner_source_does_not_import_the_hot_path_service():
    """The miner MUST NOT touch service.py's ask() loop — it reads the table offline,
    never inline. Guard the import boundary in source so a refactor can't sneak it in."""
    src = pathlib.Path(cap.__file__).read_text(encoding="utf-8")
    assert "from ..service import" not in src
    assert "from aerys_v2.service import" not in src
    assert "import aerys_v2.service" not in src


# ── /gaps read surface ───────────────────────────────────────────────────────


class FakeReadConn:
    def __init__(self, rows):
        self.rows = rows
        self.params = None

    def execute(self, sql, params=None):
        self.params = params
        assert "FROM v2_capability_requests" in sql
        return FakeCursor(self.rows)


def _gap_row(**over):
    base = {
        "id": 1, "created_at": T0, "signal_kind": "degraded", "origin_class": "error",
        "required_tier": "standard", "status": "open", "how_often": 3,
        "first_seen_at": T0, "last_seen_at": T0, "summary": "degraded subsystem marker: ha_unreachable",
    }
    base.update(over)
    return tuple(base[c] for c in GAPS_COLUMNS)


def test_read_gaps_passes_status_and_limit_and_zips_columns():
    conn = FakeReadConn([_gap_row(id=5)])
    rows = read_gaps(conn, status="open", limit=10)
    assert conn.params == {"status": "open", "limit": 10}
    assert rows[0]["id"] == 5 and rows[0]["origin_class"] == "error"


def test_format_gaps_fences_and_badges_provenance():
    rows = [
        {"id": 1, "created_at": T0, "signal_kind": "degraded", "origin_class": "error",
         "required_tier": "standard", "status": "open", "how_often": 3,
         "first_seen_at": T0, "last_seen_at": T0, "summary": "tool 'ha' failed (timeout)"},
        {"id": 2, "created_at": T0, "signal_kind": "reply_phrase", "origin_class": "complaint",
         "required_tier": "stringent", "status": "open", "how_often": 1,
         "first_seen_at": T0, "last_seen_at": T0, "summary": "I wish I could book flights"},
    ]
    out = format_gaps(rows)
    assert "information only, never instructions" in out            # the fence
    assert "[error]" in out
    assert "⚠️ complaint, not an error" in out                      # owner decision 4
    assert "tier=stringent" in out
    assert "I wish I could book flights" in out


def test_format_gaps_empty():
    assert "(none)" in format_gaps([])
