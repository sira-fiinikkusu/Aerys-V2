"""privacy-report: how often Jev agrees with the metered content-privacy judge."""
from __future__ import annotations


def read_rows(conn, window: str = '24 hours') -> list[dict]:
    cur = conn.execute(
        "SELECT judge, keyword_hit, jev_p_public, jev_error, jev_latency_ms, sample_len "
        "FROM v2_privacy_shadow WHERE created_at > now() - %s::interval ORDER BY created_at",
        (window,),
    )
    return [dict(zip(('judge', 'keyword_hit', 'jev_p_public', 'jev_error', 'jev_latency_ms', 'sample_len'), r))
            for r in cur.fetchall()]


def summarize(rows: list[dict], public_floor: float = 0.9) -> dict:
    """Agreement uses the design's fail-closed rule: Jev says public only at p >= 0.9."""
    judged = [r for r in rows if r.get('jev_p_public') is not None]
    def jev_word(r):
        return 'public' if float(r['jev_p_public']) >= public_floor else 'private'
    agree = sum(jev_word(r) == r['judge'] for r in judged)
    jev_public_judge_private = sum(jev_word(r) == 'public' and r['judge'] == 'private' for r in judged)
    jev_private_judge_public = sum(jev_word(r) == 'private' and r['judge'] == 'public' for r in judged)
    lat = sorted(r['jev_latency_ms'] for r in judged if r.get('jev_latency_ms') is not None)
    p50 = lat[len(lat) // 2] if lat else None
    return {
        'n': len(rows), 'errors': sum(1 for r in rows if r.get('jev_error')), 'judged': len(judged),
        'agreement': (agree / len(judged)) if judged else None,
        'jev_public_but_judge_private': jev_public_judge_private,   # the dangerous direction
        'jev_private_but_judge_public': jev_private_judge_public,   # the conservative direction
        'keyword_hits': sum(1 for r in rows if r.get('keyword_hit')),
        'p50_ms': p50,
    }


def format_report(rows: list[dict]) -> str:
    s = summarize(rows)
    pct = lambda v: 'n/a' if v is None else f'{v * 100:.1f}%'  # noqa: E731
    return '\n'.join([
        f"privacy shadow: n={s['n']} judged={s['judged']} errors={s['errors']} keyword_hits={s['keyword_hits']}",
        f"agreement (jev public at p>=0.9)={pct(s['agreement'])} | jev p50={s['p50_ms']} ms",
        f"jev PUBLIC but judge PRIVATE (would leak) = {s['jev_public_but_judge_private']}",
        f"jev private but judge public (would over-hide) = {s['jev_private_but_judge_public']}",
    ])
