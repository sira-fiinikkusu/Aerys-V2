"""Phase 3 in SHADOW: Jev judges content privacy beside the metered judge, and only
the two VERDICTS are recorded — never the content.

Design doc (aerys-jev-DESIGN.md §Phase 3): the privacy judge becomes a Noul in the
reflex call so the retag runs synchronously and cheaply. Chris 2026-09-20 00:20:
"Phase 3 can be done after the 10 am build." This is the evidence step that must
come first: no behaviour changes, the metered judge's verdict still decides, and a
report compares the two before anyone flips anything.

2026-10-03 (Chris, option B on the held-out rescore): no longer evidence only. Jev's
category answer is a second "private" vote (reflex_privacy_vote_below, kill switch 0).

Why the sink stores no text: the whole point of the judge is that this content may
be private. v2_privacy_shadow holds verdicts, probabilities, latency and lengths.
"""
from __future__ import annotations

import logging
import threading
from typing import Callable

from aerys_v2.services.content_privacy import PRIVATE, PUBLIC, keyword_verdict

log = logging.getLogger(__name__)

INSERT = (
    "INSERT INTO v2_privacy_shadow (judge, keyword_hit, jev_p_public, jev_error, jev_latency_ms, sample_len, "
    "jev_category, jev_p_ordinary) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)"
)


def vote_says_private(p_ordinary, vote_below: float) -> bool:
    """Option B's rule, shared by the gate and the report so the two can never disagree:
    no real number (None, a bool, a string, NaN) is a private vote (fail closed), and so
    is p(ordinary) below the bar."""
    if isinstance(p_ordinary, bool) or not isinstance(p_ordinary, (int, float)):
        return True
    return not p_ordinary >= vote_below          # NaN compares False -> private


def db_sink_for(database_url: str | None) -> Callable[[dict], None] | None:
    """A fail-open writer to v2_privacy_shadow (None without a database)."""
    if not database_url:
        return None
    import psycopg

    def sink(row: dict) -> None:
        try:
            with psycopg.connect(database_url, connect_timeout=5) as conn:
                conn.execute(INSERT, (row["judge"], row["keyword_hit"], row.get("jev_p_public"),
                                      row.get("jev_error"), row.get("jev_latency_ms"), row["sample_len"],
                                      row.get("jev_category"), row.get("jev_p_ordinary")))
        except Exception:  # audit only — never touch the turn or the judge's verdict
            log.warning("privacy shadow row not written", exc_info=True)

    return sink


def shadow_privacy_fn(judge: Callable[[str], str], reflex, sink: Callable[[dict], None] | None,
                      vote_below: float | None = None) -> Callable[[str], str]:
    """Wrap the metered classifier: Jev asked beside it, both recorded.

    Without a vote bar (the shadow), and whenever the judge says private, the judge's
    answer is returned unchanged and FIRST, and Jev runs on a daemon thread (evidence
    only). Only a judge 'public' under a vote bar waits for Jev, by design.

    With VOTE_BELOW (Chris 2026-10-03, option B), Jev's category is a SECOND "private"
    vote. A judge 'private' (or anything not 'public') is returned at once, exactly as
    before. A judge 'public' waits for Jev (ONE call, which also feeds the audit row) and
    turns 'private' when p(ordinary) < VOTE_BELOW or when Jev has no category answer:
    fail closed, and the retag path is off the hot path, so the wait costs nothing she
    says. Jev can only ever make a verdict MORE private.
    """
    voting = vote_below is not None and vote_below > 0

    def ask_jev(text: str) -> dict:
        try:
            return reflex.judge_privacy(text)
        except Exception as exc:  # the client promises not to raise; belt and braces
            return {"error": f"{type(exc).__name__}"}

    def record(text: str, verdict: str, jev: dict) -> None:
        row = {
            "judge": verdict if verdict in (PUBLIC, PRIVATE) else PRIVATE,
            "keyword_hit": keyword_verdict(text) == PRIVATE,
            "jev_p_public": jev.get("p_public"),
            "jev_error": jev.get("error"),
            "jev_latency_ms": jev.get("latency_ms"),
            "sample_len": len(text or ""),
            "jev_category": jev.get("category"),     # a label from a fixed list, never content
            "jev_p_ordinary": jev.get("p_ordinary"),
        }
        if sink is not None:
            sink(row)

    def in_background(work: Callable[[], None]) -> None:
        try:
            threading.Thread(target=work, name="privacy-shadow", daemon=True).start()
        except RuntimeError:
            log.warning("privacy shadow thread could not start", exc_info=True)

    def classify(text: str) -> str:
        verdict = judge(text)
        if voting and verdict == PUBLIC:
            jev = ask_jev(text)
            in_background(lambda: record(text, verdict, jev))
            p = jev.get("p_ordinary")
            if vote_says_private(p, vote_below):
                log.info("privacy vote: judge public, jev %s (p_ordinary=%s) -> private",
                         jev.get("category") or jev.get("error") or jev.get("category_error"), p)
                return PRIVATE
            return verdict
        in_background(lambda: record(text, verdict, ask_jev(text)))
        return verdict

    return classify


def privacy_shadow_for(settings, judge, reflex):
    """Arm the shadow when reflex is on and the flag is set; otherwise the judge as-is.
    reflex_privacy_vote_below > 0 makes Jev's category a second 'private' vote."""
    if judge is None or reflex is None or not getattr(settings, "reflex_privacy_shadow", False):
        return judge
    if not hasattr(reflex, "judge_privacy"):
        return judge
    return shadow_privacy_fn(judge, reflex, db_sink_for(settings.database_url),
                             vote_below=getattr(settings, "reflex_privacy_vote_below", None))
