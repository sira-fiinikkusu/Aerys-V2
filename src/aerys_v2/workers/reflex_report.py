"""Read shadow evidence without giving either classifier authority over a turn."""
from __future__ import annotations


def read_reflex_rows(conn, window: str = '24 hours') -> list[dict]:
    rows = conn.execute(
        "SELECT input_text, reflex FROM v2_turns "
        "WHERE reflex IS NOT NULL AND created_at >= now() - %s::interval "
        "ORDER BY created_at", (window,),
    ).fetchall()
    return [{'input_text': text, 'reflex': reflex} for text, reflex in rows]


def _percentile(values, fraction):
    if not values:
        return None
    values = sorted(values)
    index = (len(values) - 1) * fraction
    low = int(index)
    high = min(low + 1, len(values) - 1)
    return values[low] + (values[high] - values[low]) * (index - low)


def _agreement(pairs, field, *, confident=False):
    eligible = [(jev, router) for jev, router in pairs
                if router.get(field) is not None and jev.get(field) is not None
                and (not confident or jev.get('confidence', 0) >= .6)
                # The router grades tier for CHAT routes only (the action subgraph
                # runs its own fixed tier), so a tier comparison on an action row
                # measures nothing. First live report (2026-09-19) showed 46% for
                # exactly this reason.
                and (field != 'tier' or router.get('route') == 'chat')]
    if not eligible:
        return None
    # Noul is a probability; the router's boolean is compared at the midpoint.
    matches = sum((jev[field] >= .5 if field == 'unaddressed' else jev[field]) == router[field]
                  for jev, router in eligible)
    return matches / len(eligible)


def summarize(rows: list[dict]) -> dict:
    shadows = [row['reflex'] for row in rows]
    pairs = [(s['jev'], s.get('router') or {}) for s in shadows if 'error' not in s['jev']]
    latencies = [s['jev']['latency_ms'] for s in shadows if s['jev'].get('latency_ms') is not None]
    return {
        'n': len(rows),
        'errors': sum('error' in s['jev'] for s in shadows),
        'p50_ms': _percentile(latencies, .5),
        'p90_ms': _percentile(latencies, .9),
        'route_agreement': _agreement(pairs, 'route'),
        'confident_route_agreement': _agreement(pairs, 'route', confident=True),
        'tier_agreement': _agreement(pairs, 'tier'),
        'unaddressed_agreement': _agreement(pairs, 'unaddressed'),
    }


def format_report(rows: list[dict]) -> str:
    report = summarize(rows)

    def pct(value):
        return 'n/a' if value is None else f'{value:.1%}'

    def ms(value):
        return 'n/a' if value is None else f'{value:.1f} ms'

    lines = [
        f"n={report['n']} errors={report['errors']}",
        f"latency p50={ms(report['p50_ms'])} p90={ms(report['p90_ms'])}",
        f"route agreement={pct(report['route_agreement'])}",
        f"route agreement (confidence >= 0.6)={pct(report['confident_route_agreement'])}",
        f"tier agreement={pct(report['tier_agreement'])}",
        f"unaddressed agreement (noul >= 0.5)={pct(report['unaddressed_agreement'])}",
    ]
    for row in rows:
        shadow = row['reflex']
        jev, router = shadow['jev'], shadow.get('router') or {}
        if 'error' in jev:
            continue
        disagreements = [field for field in ('route', 'tier', 'unaddressed')
                         if router.get(field) is not None and jev.get(field) is not None
                         and (field != 'tier' or router.get('route') == 'chat')
                         and (jev[field] >= .5 if field == 'unaddressed' else jev[field]) != router[field]]
        if disagreements:
            lines.append(
                f"truth={router.get('route')} jev={jev.get('route')} "
                f"p={jev.get('p_action')} conf={jev.get('confidence')} :: "
                f"{(row.get('input_text') or '')[:70]} "
                f"[diff: {', '.join(disagreements)}]"
            )
    return '\n'.join(lines)
