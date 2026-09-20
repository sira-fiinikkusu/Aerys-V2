"""Phase 3 in SHADOW: Jev judges content privacy beside the metered judge, and only
the two VERDICTS are recorded — never the content.

Design doc (aerys-jev-DESIGN.md §Phase 3): the privacy judge becomes a Noul in the
reflex call so the retag runs synchronously and cheaply. Chris 2026-09-20 00:20:
"Phase 3 can be done after the 10 am build." This is the evidence step that must
come first: no behaviour changes, the metered judge's verdict still decides, and a
report compares the two before anyone flips anything.

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
    "INSERT INTO v2_privacy_shadow (judge, keyword_hit, jev_p_public, jev_error, jev_latency_ms, sample_len) "
    "VALUES (%s, %s, %s, %s, %s, %s)"
)


def db_sink_for(database_url: str | None) -> Callable[[dict], None] | None:
    """A fail-open writer to v2_privacy_shadow (None without a database)."""
    if not database_url:
        return None
    import psycopg

    def sink(row: dict) -> None:
        try:
            with psycopg.connect(database_url, connect_timeout=5) as conn:
                conn.execute(INSERT, (row["judge"], row["keyword_hit"], row.get("jev_p_public"),
                                      row.get("jev_error"), row.get("jev_latency_ms"), row["sample_len"]))
        except Exception:  # audit only — never touch the turn or the judge's verdict
            log.warning("privacy shadow row not written", exc_info=True)

    return sink


def shadow_privacy_fn(judge: Callable[[str], str], reflex, sink: Callable[[dict], None] | None) -> Callable[[str], str]:
    """Wrap the metered classifier: same verdict out, Jev asked beside it, both recorded.

    The judge's answer is returned unchanged and FIRST; Jev runs on a daemon thread so
    a slow or dead Jev never delays the retag path. Jev's answer is evidence only.
    """
    def classify(text: str) -> str:
        verdict = judge(text)

        def shadow() -> None:
            try:
                jev = reflex.judge_privacy(text)
            except Exception as exc:  # the client promises not to raise; belt and braces
                jev = {"error": f"{type(exc).__name__}"}
            row = {
                "judge": verdict if verdict in (PUBLIC, PRIVATE) else PRIVATE,
                "keyword_hit": keyword_verdict(text) == PRIVATE,
                "jev_p_public": jev.get("p_public"),
                "jev_error": jev.get("error"),
                "jev_latency_ms": jev.get("latency_ms"),
                "sample_len": len(text or ""),
            }
            if sink is not None:
                sink(row)

        try:
            threading.Thread(target=shadow, name="privacy-shadow", daemon=True).start()
        except RuntimeError:
            log.warning("privacy shadow thread could not start", exc_info=True)
        return verdict

    return classify


def privacy_shadow_for(settings, judge, reflex):
    """Arm the shadow when reflex is on and the flag is set; otherwise the judge as-is."""
    if judge is None or reflex is None or not getattr(settings, "reflex_privacy_shadow", False):
        return judge
    if not hasattr(reflex, "judge_privacy"):
        return judge
    return shadow_privacy_fn(judge, reflex, db_sink_for(settings.database_url))
