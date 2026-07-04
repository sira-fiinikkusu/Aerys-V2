"""Capability-request miner — the self-iteration loop's detection layer (Phase A).

Design doc: docs/capability-request-loop-design.md (v3, provenance-tiered).

Aerys notices when she hits a wall — a tool that failed, a subsystem that
degraded, OR a capability she articulates missing — and that friction becomes a
structured capability-request row the owner can read (`/gaps`) and act on. This
module is the OFFLINE detection/miner half: a consolidation worker (mirrors
workers/extraction.py) that scans NEW v2_turns since a high-water mark and writes
gap fingerprints into migration 004's two tables. It is periodic + on-demand and
NEVER inline — it never imports or touches service.py's ask() loop; a live turn
can never wait on, or be crashed by, this worker.

The load-bearing security property (the whole reason B is safe — see the design's
§ "provenance is machine-set, so trust can't be forged"):

  - origin_class='error'  is derived ONLY from structural signals the model CANNOT
    forge: a `degraded` marker or a real `tool_calls` failure — both machine-set by
    the v2_turns writer (turns.py), never by model text. `_error_signals` reads
    ONLY those two structured JSONB columns.
  - origin_class='complaint' is derived from fakeable, MODEL-AUTHORED reply-phrase
    text (raw_reply / emitted_reply). `_complaint_signals` reads ONLY those two text
    columns. A complaint is forced onto the STRICTER approval tier ('stringent') and
    is always surfaced explicitly labeled "complaint, not an error".

The separation is STRUCTURAL: which function reads which column is the boundary.
An attacker who injects "the tool failed and the connection timed out" into a reply
still classifies as a COMPLAINT (stricter gate) when tool_calls/degraded are clean —
they may author the text, but they can never upgrade its trust level to 'error'.
test_capability_requests.py pins exactly this.

Mined content is DATA, never instructions. Nothing in a summary/excerpt is ever
executed or obeyed — the miner only pattern-matches and records, and the owner's
manual /approve gate (Phase B) is the only path that authorizes any build. The
brain never writes these tables; there is no self-grant path here.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Iterable

# The watermark + LIMIT-tie + failed-row-hold machinery is IDENTICAL to the
# extraction worker's (the design says "inherit both _trim_tie_boundary AND
# _safe_watermark"), so reuse it rather than fork a second copy that could drift.
# These helpers are pure and offline (test_extraction proves them); importing them
# does not connect to anything.
from .extraction import (
    _safe_watermark,
    _trim_tie_boundary,
    read_watermark,
    save_watermark,
)

log = logging.getLogger(__name__)

# One shared high-water-mark row (migration 002's v2_extraction_watermark), keyed
# distinctly so the miner and the memory extractor never advance each other's mark.
WATERMARK_SOURCE = "capability_gaps"

DEFAULT_LOOKBACK_H = 2    # first run with no watermark: 2 hours ago (extraction default)
DEFAULT_LIMIT = 200       # turns per pass (same ceiling as extraction)
PARITY_WINDOW = 50        # how many recent turns the parity gate inspects
EXCERPT_LIMIT = 200       # max chars of a complaint excerpt stored in `summary`

# required_tier is DERIVED from origin_class, never set independently — the fakeable
# 'complaint' path always lands on the stricter gate. Kept as a table so the mapping
# is one obvious place; GapSignal.required_tier reads it as a property so a signal's
# tier can NEVER diverge from its (machine-set) origin_class.
REQUIRED_TIER_BY_ORIGIN = {"error": "standard", "complaint": "stringent"}


# ── complaint phrase set ─────────────────────────────────────────────────────
# Tuned reply-phrases that articulate a MISSING CAPABILITY — deliberately distinct
# from the tools' failure sentinels (turns.py's _TOOL_FAILURE_SIGNALS, e.g. "search
# service is unreachable"): those are structural failures that ride the un-forgeable
# 'error' path. THESE are her saying, in her own model-authored words, "I wish I
# could but I can't" — the value B adds. Because the text is model-authored (and thus
# attacker-influenceable), every match is stamped 'complaint' and forced onto the
# stringent gate; the phrase is never trusted, only its machine-set label. Matched
# case-insensitively as a substring of the normalized reply. Order = the fingerprint's
# canonical head-phrase, so "I don't have a tool for X" and "...for Y" collapse to one
# recurring request, not two.
_COMPLAINT_PHRASES: tuple[str, ...] = (
    "i don't have a tool for",
    "i don't have a tool to",
    "i don't have a way to",
    "i don't have access to",
    "i don't have the ability to",
    "i'm not able to",
    "i am not able to",
    "i'm unable to",
    "i am unable to",
    "i can't do that",
    "i cannot do that",
    "i wish i could",
    "i'd love to but i can't",
    "i would love to but i can't",
    "there's no tool for",
    "there is no tool for",
    "i don't have that capability",
)


@dataclass(frozen=True)
class GapSignal:
    """One detected gap on one turn. `origin_class` is set by which detector
    constructed it (error vs complaint); `required_tier` is a DERIVED property of
    origin_class, so the fakeable complaint path can never present as the trusted
    standard tier. `fingerprint` and `summary` are plain data — passed as bound SQL
    params, never interpolated, never executed."""

    signal_kind: str      # 'degraded' | 'tool_error' | 'reply_phrase'
    origin_class: str     # 'error' | 'complaint' — MACHINE-SET by the detector
    fingerprint: str
    summary: str

    @property
    def required_tier(self) -> str:
        # Derived, not stored — a complaint (fakeable) ALWAYS demands 'stringent'.
        return REQUIRED_TIER_BY_ORIGIN[self.origin_class]


def _as_list(value: object) -> list:
    """A structured JSONB column read back as a list, or [] for anything else.

    psycopg decodes jsonb to Python objects, so tool_calls/degraded arrive as
    lists. A malformed/legacy value (None, a dict, a prose string that slipped in)
    degrades to [] — the SAFE direction: under-report a gap, never crash the pass
    or fabricate one. Same doctrine as turns.py's tool classification."""
    if isinstance(value, list):
        return value
    if value is not None:
        # A non-None value that ISN'T a list means the structured column did not
        # decode to a JSONB array (a decoding regression: a non-default jsonb loader,
        # a connection-config change, or a legacy prose value that slipped in). We
        # still fail safe to [] — never crash, never fabricate a gap — but WARN,
        # because otherwise the whole structural 'error' path silently zeroes out and
        # only complaints would ever fire, invisibly. None is expected (pre-writer /
        # JSON null) and stays quiet. (cross-review — make the silent under-report loud.)
        log.warning(
            "gap mining: expected a JSONB list for a structured turn column but got "
            "%s — coercing to [] (structural 'error' signals for this turn are "
            "skipped; possible jsonb decoding regression)",
            type(value).__name__,
        )
    return []


def _bounded_excerpt(text: object, *, limit: int = EXCERPT_LIMIT) -> str:
    """A complaint's stored summary: her reply, sanitized and length-capped.

    Strips newlines/control chars (so the excerpt can't smuggle framing into the
    later fenced digest) and truncates — the full reply lives in v2_turns, this is
    only the owner-facing teaser. NOT a fence itself: the fence + "complaint, not
    an error" label are applied at surfacing time (format_gaps), because the
    trust-boundary framing only makes sense at the point of display."""
    cleaned = "".join(
        c for c in str(text) if c.isprintable() and c not in "\r\n\t"
    ).strip()
    cleaned = " ".join(cleaned.split())  # collapse runs of spaces
    if len(cleaned) > limit:
        cleaned = cleaned[: limit - 1].rstrip() + "…"
    return cleaned


def _error_signals(turn: dict) -> list[GapSignal]:
    """Structural, un-forgeable gap signals — reads ONLY the machine-set JSONB
    columns (`degraded`, `tool_calls`), NEVER any reply text. Summaries are fixed
    templates: no model-authored text ever enters a high-trust 'error' row.

    This function's input surface is the whole security boundary. The model authors
    reply_reply/emitted_reply; it does NOT author a degraded marker or a tool_calls
    {ok:false} entry (turns.py builds those from LangChain's authoritative signals
    and a curated sentinel table, scanning only the first line of each tool result).
    So an 'error' here is a fact about infrastructure, not a claim in prose."""
    signals: list[GapSignal] = []

    # degraded markers — coarse per-subsystem health (e.g. 'ha_unreachable',
    # 'turn_failed'). Fingerprint keys on the marker alone: every turn that trips
    # the same marker is the same recurring gap.
    for marker in _as_list(turn.get("degraded")):
        if not isinstance(marker, str) or not marker.strip():
            continue
        m = marker.strip()
        signals.append(
            GapSignal(
                signal_kind="degraded",
                origin_class="error",
                fingerprint=f"degraded::{m}",
                summary=f"degraded subsystem marker: {m}",
            )
        )

    # real tool failures — {name, ok:false, error_class}. Fingerprint keys on
    # (tool, error_class) so a timeout and an auth_error on the same tool stay
    # distinct gaps (they need different fixes).
    for call in _as_list(turn.get("tool_calls")):
        if not isinstance(call, dict) or call.get("ok") is not False:
            continue
        name = str(call.get("name") or "unknown")
        error_class = str(call.get("error_class") or "exception")
        signals.append(
            GapSignal(
                signal_kind="tool_error",
                origin_class="error",
                fingerprint=f"tool_error::{name}::{error_class}",
                summary=f"tool {name!r} failed ({error_class})",
            )
        )

    return signals


def _complaint_signals(turn: dict) -> list[GapSignal]:
    """Fakeable, model-authored gap signals — reads ONLY the reply text
    (`raw_reply`, then `emitted_reply`), NEVER the structured columns. Every match
    is stamped 'complaint' (stringent tier); the phrase is never trusted, only the
    machine-set label. The summary carries a bounded, sanitized excerpt of the
    matching reply — that excerpt is the value B adds (what she wished for) and is
    accepted precisely because it rides the stricter gate.

    One signal per UNIQUE matched phrase (its canonical head-phrase is the
    fingerprint), so "I don't have a tool for X" recurs as one request across
    turns rather than fragmenting per object."""
    # Prefer the model's own words (raw_reply); on chat they're equal, on voice-
    # action raw_reply is the real outcome while emitted_reply was the spoken ack.
    candidates = [
        c for c in (turn.get("raw_reply"), turn.get("emitted_reply"))
        if isinstance(c, str) and c.strip()
    ]
    if not candidates:
        return []

    seen: dict[str, str] = {}  # phrase -> the excerpt of the reply it matched in
    for reply in candidates:
        low = reply.lower()
        for phrase in _COMPLAINT_PHRASES:
            if phrase in low and phrase not in seen:
                seen[phrase] = _bounded_excerpt(reply)

    return [
        GapSignal(
            signal_kind="reply_phrase",
            origin_class="complaint",
            fingerprint=f"reply_phrase::{phrase}",
            summary=excerpt,
        )
        for phrase, excerpt in seen.items()
    ]


def classify_turn(turn: dict) -> list[GapSignal]:
    """All gap signals a single turn carries — errors (structural) THEN complaints
    (reply-phrase). The two detectors read DISJOINT columns, which is what makes a
    reply full of failure words but with clean structured fields classify as a
    complaint (stricter gate), never an error."""
    return _error_signals(turn) + _complaint_signals(turn)


class GapMiningRefused(RuntimeError):
    """A hard gate tripped — refuse loudly, never mine silently. Same posture as
    extraction.LiveWriteRefused / config.BootConfigError: a surface aimed at a
    dangerous or unproven state must refuse before touching anything."""


# ── the columns the miner reads from v2_turns ────────────────────────────────
MINER_COLUMNS = (
    "id",
    "person_id",
    "created_at",
    "created_at_raw",
    "thread_id",
    "channel",
    "raw_reply",
    "emitted_reply",
    "tool_calls",
    "degraded",
    "error",
)

# Owner-scoped fetch (cross-review H2 — NEVER mine a stranger's turns into the
# owner's roadmap): person_id = ANY(allowlist). A NULL person_id (cold caller)
# never matches ANY(...), so unresolved/stranger turns are excluded outright. Raw
# created_at::text rides alongside for the verbatim-string watermark; ORDER BY
# created_at ASC, id ASC is the deterministic keyset order _trim_tie_boundary needs.
MINER_SQL = """\
SELECT
  t.id::text AS id,
  t.person_id,
  t.created_at,
  t.created_at::text AS created_at_raw,
  t.thread_id,
  t.channel,
  t.raw_reply,
  t.emitted_reply,
  t.tool_calls,
  t.degraded,
  t.error
FROM v2_turns t
WHERE t.person_id = ANY(%(person_ids)s::uuid[])
  AND t.created_at > %(after)s::timestamptz
ORDER BY t.created_at ASC, t.id ASC
LIMIT %(limit)s
"""

# The Phase-A parity gate (design § Prerequisite): the whole feature MINES the
# v2_turns writer's structured fields, so it must refuse to run until that writer
# has demonstrably landed. Inspect the most recent PARITY_WINDOW turns: `armed`
# counts rows the writer actually populated (raw_reply + tool_calls + degraded all
# non-null). Pre-writer, nothing wrote those — armed would be 0 (or the table
# empty), and mining would read a table of false-premise NULLs.
PARITY_SQL = """\
SELECT
  count(*) AS total,
  count(*) FILTER (
    WHERE raw_reply IS NOT NULL
      AND tool_calls IS NOT NULL
      AND degraded  IS NOT NULL
  ) AS armed
FROM (
  SELECT raw_reply, tool_calls, degraded
  FROM v2_turns
  ORDER BY id DESC
  LIMIT %(window)s
) recent
"""

# Atomic dedup child insert. The (fingerprint, turn_id) PK makes a turn count once
# per fingerprint EVER — a crash-retry re-run hits ON CONFLICT DO NOTHING and the
# RETURNING yields no row, so `inserted` is False (stats only; the parent upsert
# runs regardless and self-heals any orphan). turn_id/seen_at cast explicitly since
# the id rides as text (keyset-order/watermark convention).
EXAMPLE_INSERT_SQL = """\
INSERT INTO v2_capability_request_examples (fingerprint, turn_id, seen_at)
VALUES (%(fingerprint)s, %(turn_id)s::bigint, %(seen_at)s::timestamptz)
ON CONFLICT (fingerprint, turn_id) DO NOTHING
RETURNING turn_id
"""

# Parent upsert. how_often is COUNT(*) over the examples child (NEVER a blind +1) on
# BOTH insert and update, so recurrence is exactly "distinct turns that carried this
# fingerprint". ON CONFLICT touches ONLY observation fields (how_often, last_seen_at,
# updated_at): status/origin_class/summary/required_tier and the Kael/owner workflow
# columns are left untouched, so a terminal ('rejected'/'wont_fix'/'built') row keeps
# counting but never auto-resurrects to 'open', and a diagnosis is never clobbered.
PARENT_UPSERT_SQL = """\
INSERT INTO v2_capability_requests
  (fingerprint, signal_kind, origin_class, summary, origin_trust,
   required_tier, how_often, first_seen_at, last_seen_at)
VALUES
  (%(fingerprint)s, %(signal_kind)s, %(origin_class)s, %(summary)s, %(origin_trust)s,
   %(required_tier)s,
   (SELECT count(*) FROM v2_capability_request_examples WHERE fingerprint = %(fingerprint)s),
   %(first_seen_at)s::timestamptz, %(last_seen_at)s::timestamptz)
ON CONFLICT (fingerprint) DO UPDATE SET
   how_often    = (SELECT count(*) FROM v2_capability_request_examples
                   WHERE fingerprint = %(fingerprint)s),
   last_seen_at = GREATEST(v2_capability_requests.last_seen_at, EXCLUDED.last_seen_at),
   updated_at   = now()
"""


def assert_turns_parity(conn: Any, *, window: int = PARITY_WINDOW) -> None:
    """Refuse to mine until the v2_turns writer has demonstrably landed.

    Raises GapMiningRefused when the last `window` turns carry no writer-populated
    fields (writer not landed / false-premise NULLs) or the table is empty (nothing
    to prove the writer works). Passing means recent rows really carry the
    raw_reply/tool_calls/degraded this feature mines."""
    row = conn.execute(PARITY_SQL, {"window": window}).fetchone()
    total = (row[0] if row else 0) or 0
    armed = (row[1] if row else 0) or 0
    if not total:
        raise GapMiningRefused(
            "v2_turns has no recent rows — cannot confirm the turns writer has "
            "landed; refusing to mine (parity gate)."
        )
    if not armed:
        raise GapMiningRefused(
            f"none of the last {window} v2_turns rows carry writer-populated fields "
            "(raw_reply/tool_calls/degraded all NULL) — the v2_turns writer has not "
            "landed; refusing to mine (parity gate)."
        )


def _record_signal(conn: Any, signal: GapSignal, turn: dict) -> bool:
    """Persist one signal: insert the (fingerprint, turn) example, then upsert the
    parent with how_often recomputed from the child. Both statements are idempotent,
    so a crash-retry re-run is safe and self-healing. Returns True when THIS turn
    was a new example for the fingerprint (stats only). last_seen_at/first_seen_at
    are the turn's OWN created_at (the h.created_at lesson — when the gap actually
    happened, not when the cron ran); GREATEST keeps last_seen_at monotonic."""
    cur = conn.execute(
        EXAMPLE_INSERT_SQL,
        {
            "fingerprint": signal.fingerprint,
            "turn_id": turn["id"],
            "seen_at": turn["created_at_raw"],
        },
    )
    inserted = cur.fetchone() is not None
    conn.execute(
        PARENT_UPSERT_SQL,
        {
            "fingerprint": signal.fingerprint,
            "signal_kind": signal.signal_kind,
            "origin_class": signal.origin_class,
            "summary": signal.summary,
            "origin_trust": "owner",  # Phase A mines only owner/allowlisted turns
            "required_tier": signal.required_tier,  # DERIVED from origin_class
            "first_seen_at": turn["created_at_raw"],
            "last_seen_at": turn["created_at_raw"],
        },
    )
    return inserted


def run_gap_mining(
    conn: Any,
    *,
    allowlist: Iterable[str],
    lookback_hours: int = DEFAULT_LOOKBACK_H,
    batch_limit: int = DEFAULT_LIMIT,
    parity_window: int = PARITY_WINDOW,
) -> dict:
    """One consolidation pass over NEW owner/allowlisted turns.

    conn — the brain's OWN aerys_v2 database: reads v2_turns, writes the two
    capability tables + the watermark. NO prod aerys connection: unlike extraction,
    this feature lives entirely in aerys_v2. Never a service.py import — the miner
    is offline/periodic, never inline on a turn.

    Flow (mirrors run_extraction): refuse on an empty allowlist -> parity gate ->
    watermark -> owner-scoped fetch (limit+1) -> tie-trim -> per-turn classify+record
    -> advance the watermark (held below any turn whose processing threw, via
    _safe_watermark). Returns a summary dict for logs.

    Owner-scope is a hard gate (cross-review H2 + the design's None-defeatable
    caveat): an empty allowlist means no owner is configured, and mining then would
    either match everyone or no one — refuse rather than guess."""
    # The per-turn crash isolation below (each turn's writes roll back to a SAVEPOINT
    # via `with conn.transaction()`) is load-bearing and holds ONLY on a non-autocommit
    # connection: psycopg3 nests a SAVEPOINT when a transaction is already open, but
    # under autocommit each `with conn.transaction()` becomes a top-level transaction,
    # so a single poison row would abort the batch AND the watermark save and re-stall
    # the same row every pass. The invariant was validated with offline fakes only
    # (house rules forbid a live DB in tests), so make it explicit here: refuse loudly
    # rather than mine under a posture the isolation was never proven against.
    # (cross-review — the flagged "assert not conn.autocommit at pass start".)
    if getattr(conn, "autocommit", False):
        raise GapMiningRefused(
            "connection is in autocommit mode — the miner's per-turn SAVEPOINT "
            "isolation requires a single outer transaction; refusing to mine."
        )

    allow = sorted({str(p) for p in allowlist if str(p).strip()})
    if not allow:
        raise GapMiningRefused(
            "no owner/allowlist configured — refusing to mine v2_turns (would "
            "either scope to nobody or, misconfigured, to everybody)."
        )

    # PARITY GATE first — before any watermark read or turn fetch.
    assert_turns_parity(conn, window=parity_window)

    after = read_watermark(conn, WATERMARK_SOURCE, lookback_hours=lookback_hours)
    raw_rows = conn.execute(
        MINER_SQL, {"person_ids": allow, "after": after, "limit": batch_limit + 1}
    ).fetchall()
    rows = [dict(zip(MINER_COLUMNS, r)) for r in raw_rows]
    rows = _trim_tie_boundary(rows, batch_limit)

    stats: dict[str, Any] = {
        "turns": len(rows),
        "signals": 0,
        "new_examples": 0,
        "errors": 0,
        "complaints": 0,
        "processing_failures": 0,
        "watermark": None,
    }
    if not rows:
        return stats  # empty window: no writes, watermark untouched (extraction parity)

    failed_row_ids: set[str] = set()
    for turn in rows:
        try:
            # Each turn's writes are all-or-nothing via a SAVEPOINT (conn.transaction()
            # nests one inside the pass's outer transaction). A turn whose processing
            # throws — a malformed row (Python) OR a DB hiccup mid-record — rolls back
            # cleanly to the savepoint, so (a) it leaves no half-written signal, and
            # (b) the outer transaction stays usable, so the good turns already
            # committed and the watermark save below still work. Stats are merged only
            # AFTER the savepoint commits, so a rolled-back turn never inflates counts.
            recorded: list[tuple[GapSignal, bool]] = []
            with conn.transaction():
                for signal in classify_turn(turn):
                    inserted = _record_signal(conn, signal, turn)
                    recorded.append((signal, inserted))
        except Exception:
            # HELD: _safe_watermark freezes the mark strictly below this turn so the
            # next pass retries it. Never advance past an un-mined turn — same posture
            # as extraction's parse-failure hold.
            log.warning(
                "gap mining: turn %r failed to process — holding the watermark below it",
                turn.get("id"),
                exc_info=True,
            )
            stats["processing_failures"] += 1
            failed_row_ids.add(turn["id"])
            continue
        for signal, inserted in recorded:
            stats["signals"] += 1
            stats["new_examples"] += int(inserted)
            stats["errors"] += int(signal.origin_class == "error")
            stats["complaints"] += int(signal.origin_class == "complaint")

    watermark_raw = _safe_watermark(rows, failed_row_ids, after)
    save_watermark(conn, WATERMARK_SOURCE, watermark_raw)
    stats["watermark"] = watermark_raw
    return stats


# ── owner read path — /gaps (read-only; Phase A) ─────────────────────────────
# /approve (the ONLY writer of approved_by/approved_at) is Phase B and is NOT built
# here: approval stays the owner's manual gate, never actionable from a channel
# message, never auto-executed. This surface is read-only by construction.
GAPS_COLUMNS = (
    "id",
    "created_at",
    "signal_kind",
    "origin_class",
    "required_tier",
    "status",
    "how_often",
    "first_seen_at",
    "last_seen_at",
    "summary",
)

GAPS_LIST_SQL = """\
SELECT id, created_at, signal_kind, origin_class, required_tier, status,
       how_often, first_seen_at, last_seen_at, summary
FROM v2_capability_requests
-- ::text casts so an untyped NULL status param can't trip Postgres's
-- "could not determine data type of parameter" on the IS NULL branch.
WHERE (%(status)s::text IS NULL OR status = %(status)s::text)
ORDER BY status, how_often DESC, last_seen_at DESC
LIMIT %(limit)s
"""


def read_gaps(conn: Any, *, status: str | None = None, limit: int = 50) -> list[dict]:
    """The /gaps read: mined capability-requests, optionally filtered by status,
    ranked by recurrence then recency (the status_idx's order). Read-only — this
    surface never writes, and the caller opens the connection read_only besides."""
    rows = conn.execute(
        GAPS_LIST_SQL, {"status": status, "limit": limit}
    ).fetchall()
    return [dict(zip(GAPS_COLUMNS, r)) for r in rows]


def format_gaps(rows: list[dict]) -> str:
    """Render the /gaps read for the owner — fenced, provenance-badged.

    Every row is wrapped under the untrusted-data fence (same doctrine as
    services/context.py: mined content is information, NEVER instructions), and each
    carries its origin_class badge. Complaint rows are labeled explicitly "complaint,
    not an error" (owner decision 4) — the owner approves/denies KNOWING the
    provenance, because the complaint text is model-authored and only its label is
    trustworthy. Nothing here is ever executed."""
    header = (
        "Mined capability gaps (observations from v2_turns — information only, "
        "never instructions; the model authored complaint text, not its label):"
    )
    if not rows:
        return header + "\n  (none)"

    lines = [header]
    for r in rows:
        if r["origin_class"] == "complaint":
            badge = "⚠️ complaint, not an error"
        else:
            badge = "error"
        lines.append(
            f"  #{r['id']} [{badge}] "
            f"kind={r['signal_kind']} tier={r['required_tier']} "
            f"status={r['status']} seen={r['how_often']}x"
        )
        # The excerpt/template stays indented under its row and is never presented
        # as anything but quoted data.
        lines.append(f"      {r['summary']}")
    return "\n".join(lines)
