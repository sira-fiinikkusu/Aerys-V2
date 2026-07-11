"""log_gap — she files her own capability gaps, deliberately.

Origin (2026-07-11, the glasses conversation): the owner asked her by voice to
"log a gap"; she answered honestly that no such tool existed ("no function call
available to fire that write — I can't fake a success on this one"), and the
gaps MINER then missed her clean "**Issue:** ..." statement anyway, because its
complaint detector is a fixed phrase list (capability_requests._COMPLAINT_PHRASES)
and she'd phrased it slightly off-script. One experiment, two findings. This
tool is the fix for the first; migration 007 records the new provenance lane.

Trust model — the part that must not drift: a self-reported gap is
MODEL-AUTHORED text end to end, exactly as fakeable as a mined complaint, so it
rides the SAME lane (origin_class='complaint', required_tier='stringent') and
Postgres binds that at the storage layer (the 007 provenance CHECK). What's new
is only the MECHANISM label: signal_kind='self_reported' (deliberate tool call)
vs 'reply_phrase' (mined after the fact). The fingerprint prefix 'self::' keeps
these rows disjoint from mined ones — the miner's count-derived how_often upsert
can never clobber a self-reported row, and vice versa.

Dedup semantics: the fingerprint is a slug of the summary, so re-filing the same
gap bumps how_often + last_seen_at (ON CONFLICT) instead of fragmenting — the
same one-request-that-recurs shape the miner keeps. Distinct wordings make
distinct rows; that's acceptable noise on the stringent lane, and /gaps shows
them to exactly one reader (the owner).

Failure posture: ToolNode contract — every path returns an honest string, never
raises. DB trouble = "couldn't write it", never a fake success (her own words
set that bar).
"""

from __future__ import annotations

import logging
import re
from typing import Any, Callable

log = logging.getLogger(__name__)

SUMMARY_LIMIT = 200   # matches the miner's EXCERPT_LIMIT — one line, not an essay
DETAIL_LIMIT = 600    # the diagnosis column carries the longer description

# Parent-row upsert, self-reported lane. how_often is stored directly (1, +1 on
# refile) — NOT the miner's derived-from-examples count: self-reports have no
# turn-keyed example rows, and the disjoint 'self::' fingerprint space means the
# two how_often disciplines never meet on the same row.
LOG_GAP_SQL = """\
INSERT INTO v2_capability_requests
  (fingerprint, signal_kind, origin_class, summary, origin_trust,
   required_tier, how_often, diagnosis, first_seen_at, last_seen_at)
VALUES
  (%(fingerprint)s, 'self_reported', 'complaint', %(summary)s, 'owner',
   'stringent', 1, %(diagnosis)s, now(), now())
ON CONFLICT (fingerprint) DO UPDATE SET
   how_often    = v2_capability_requests.how_often + 1,
   last_seen_at = now(),
   updated_at   = now(),
   diagnosis    = COALESCE(NULLIF(EXCLUDED.diagnosis, ''), v2_capability_requests.diagnosis)
RETURNING id, how_often
"""


def _fingerprint(summary: str) -> str:
    """'self::' + a stable slug of the summary — disjoint from every miner
    fingerprint ('degraded::…', 'tool_error::…', phrase heads) by prefix."""
    slug = re.sub(r"[^a-z0-9]+", "-", summary.lower()).strip("-")[:80]
    return f"self::{slug or 'unspecified'}"


def build_log_gap_tool(conn_factory: Callable[[], Any]):
    """Close over the aerys_v2 conn seam and return the log_gap tool.

    conn_factory yields a fresh READ-WRITE connection to the brain's OWN
    aerys_v2 database (v2_capability_requests lives there — NOT prod aerys).
    Same per-call fresh-connection stance as home_control's outbox seam.
    """
    from langchain_core.tools import tool

    @tool
    def log_gap(summary: str, details: str = "") -> str:
        """Log a capability gap, issue, limitation, or feature request into the
        owner's gaps board so his coding agent can pick it up and fix it.

        CALL THIS TOOL IMMEDIATELY when the owner asks you to "log a gap",
        "log a complaint", "file an issue", "note that for the coding agent",
        or anything similar — and also on your own initiative when you hit a
        real limitation (a tool you wish you had, something rendering wrong,
        a capability you're missing). This writes to the real gaps table; the
        owner reads it with /gaps.

        summary: ONE short sentence naming the gap (e.g. "G2 lens replies get
        cut off mid-render on long messages"). Required — never a placeholder.
        details: optional longer description — what happened, when, what a fix
        might look like.

        Returns confirmation with the gap's id, or an honest failure — never
        pretend a log succeeded if it didn't.
        """
        clean = " ".join((summary or "").split())
        if not clean:
            return (
                "NOT LOGGED: log_gap needs a real one-line summary of the gap — "
                "ask the owner what to file if it isn't clear."
            )
        clean = clean[:SUMMARY_LIMIT]
        detail_clean = " ".join((details or "").split())[:DETAIL_LIMIT]
        try:
            with conn_factory() as conn:
                row = conn.execute(
                    LOG_GAP_SQL,
                    {
                        "fingerprint": _fingerprint(clean),
                        "summary": clean,
                        "diagnosis": detail_clean,
                    },
                ).fetchone()
        except Exception as e:
            log.warning("log_gap write failed", exc_info=True)
            return (
                f"NOT LOGGED: the gaps table isn't reachable right now ({e}). "
                "Tell the owner the write failed so it isn't silently lost."
            )
        gap_id, how_often = row[0], row[1]
        if how_often > 1:
            return (
                f"Logged — this one's already on the board as gap #{gap_id}; "
                f"bumped it (now reported {how_often}x)."
            )
        return f"Logged as gap #{gap_id}. The owner will see it in /gaps."

    return log_gap
