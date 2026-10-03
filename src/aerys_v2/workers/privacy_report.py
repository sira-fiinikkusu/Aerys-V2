"""privacy-report: how often Jev agrees with the metered content-privacy judge."""
from __future__ import annotations

from aerys_v2.privacy_shadow import vote_says_private


def read_rows(conn, window: str = '24 hours') -> list[dict]:
    cur = conn.execute(
        "SELECT judge, keyword_hit, jev_p_public, jev_error, jev_latency_ms, sample_len, jev_category, jev_p_ordinary "
        "FROM v2_privacy_shadow WHERE created_at > now() - %s::interval ORDER BY created_at",
        (window,),
    )
    return [dict(zip(('judge', 'keyword_hit', 'jev_p_public', 'jev_error', 'jev_latency_ms', 'sample_len',
                      'jev_category', 'jev_p_ordinary'), r))
            for r in cur.fetchall()]


def summarize(rows: list[dict], public_floor: float = 0.9, vote_below: float = 0.3) -> dict:
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
        **_category_summary(rows),
        # Option B (10/03): judge-public turns the category vote made private (p(ordinary)
        # below the bar, or no category answer: fail closed). Rows before 10/03 show what
        # it WOULD have done.
        'vote_made_private': sum(r['judge'] == 'public' and vote_says_private(r.get('jev_p_ordinary'), vote_below)
                                 for r in rows),
        'vote_below': vote_below,
    }


def _category_summary(rows: list[dict], ordinary_floor: float = 0.5) -> dict:
    """The 9/26 candidate: private when p(ordinary) < 0.5. Counts by category, and both
    disagreement directions against the metered judge. Rows before migration 013 have none."""
    scored = [r for r in rows if r.get('jev_p_ordinary') is not None]
    private = lambda r: float(r['jev_p_ordinary']) < ordinary_floor  # noqa: E731
    by_category: dict[str, int] = {}
    for r in scored:
        if private(r):
            by_category[r.get('jev_category') or '?'] = by_category.get(r.get('jev_category') or '?', 0) + 1
    return {
        'category_scored': len(scored),
        'category_public_but_judge_private': sum(not private(r) and r['judge'] == 'private' for r in scored),
        'category_private_but_judge_public': sum(private(r) and r['judge'] == 'public' for r in scored),
        'category_private_by_kind': dict(sorted(by_category.items())),
    }


def format_report(rows: list[dict], vote_below: float = 0.3) -> str:
    s = summarize(rows, vote_below=vote_below)
    pct = lambda v: 'n/a' if v is None else f'{v * 100:.1f}%'  # noqa: E731
    return '\n'.join([
        f"privacy shadow: n={s['n']} judged={s['judged']} errors={s['errors']} keyword_hits={s['keyword_hits']}",
        f"agreement (jev public at p>=0.9)={pct(s['agreement'])} | jev p50={s['p50_ms']} ms",
        f"jev PUBLIC but judge PRIVATE (would leak) = {s['jev_public_but_judge_private']}",
        f"jev private but judge public (would over-hide) = {s['jev_private_but_judge_public']}",
        f"category candidate (private at p(ordinary)<0.5): scored={s['category_scored']} "
        f"public-but-judge-private={s['category_public_but_judge_private']} "
        f"private-but-judge-public={s['category_private_but_judge_public']} by kind={s['category_private_by_kind']}",
        f"vote (judge public -> private at p(ordinary)<{s['vote_below']:g} or no answer) = {s['vote_made_private']}",
    ])
