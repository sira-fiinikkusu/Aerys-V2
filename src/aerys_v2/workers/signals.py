"""Invariants over the traffic that already happened.

Chris, 2026-09-14, after six straight days of a clean capability-gap board while he
personally found four real bugs: the miner catches what she NOTICES she cannot do, and
every bug that week was one she did not notice. She answered honestly and moved on,
which is exactly what we asked of her. His framing, borrowed from Arize's Signal
concept: evaluate the traffic that already happened rather than inject synthetic
traffic.

That is also why this plants nothing. A plant-and-ask probe has to WRITE — a synthetic
turn in his thread, junk in her memory, noise in the audit — so testing the real path
would mean polluting the real path. Every check here is a statement about rows that
already exist, and the worker holds a read-only connection to prove it.

Each signal is one real bug from 2026-09-13/14, turned into something that cannot come
back quietly. Reading the list is reading the week.

SKIP is load-bearing and is not a pass. A check with nothing to look at has not
succeeded; a run that reports green off zero rows is how a monitor turns into a comfort
blanket. Only FAIL speaks.
"""
from __future__ import annotations

from dataclasses import dataclass
import re

PASS = 'pass'
FAIL = 'fail'
SKIP = 'skip'

#: Mirrors tools.remember.FACT_LIMIT. A memory sitting at EXACTLY the limit is the
#: fingerprint of a fact that was cut rather than one that happened to fit.
FACT_LIMIT = 500

#: The public surfaces, same set the room block itself fences on.
PUBLIC = frozenset({'guild', 'telegram_group'})

#: Voice keeps its emotion tags — they are spoken. Every other surface renders them as
#: literal brackets to a reader.
TAGGED = re.compile(r'^\s*\[[a-z][a-z ]*\]', re.I)

#: What the quarantine is allowed to hold. Anything else is the detector drifting back
#: into holding her own engineering reports (2026-09-13: five for five false).
EXPECTED_HOLD_REASONS = ('judge:', 'non-owner transcript', 'future observation')


@dataclass(frozen=True)
class Signal:
    name: str
    status: str
    detail: str
    checked: int = 0

    @property
    def failed(self) -> bool:
        return self.status == FAIL


def _result(name, offenders, checked, *, ok_detail, bad_detail):
    if not checked:
        return Signal(name, SKIP, 'nothing to check in this window', 0)
    if offenders:
        return Signal(name, FAIL, bad_detail(offenders), checked)
    return Signal(name, PASS, ok_detail, checked)


# ── the six ────────────────────────────────────────────────────────────────

def room_on_public_turns(turns) -> Signal:
    """A public turn should carry the room it was answered in (board #17)."""
    public = [t for t in turns if t.get('channel') in PUBLIC]
    missing = [t for t in public if not (t.get('room_context') or '').strip()]
    return _result(
        'room_on_public_turns', missing, len(public),
        ok_detail='every public turn carried its room',
        bad_detail=lambda bad: f"{len(bad)} public turn(s) carried no room: "
                               + ', '.join(str(t.get('id')) for t in bad[:5]))


def owner_named_in_room(turns, *, aliases) -> Signal:
    """His own room lines should read as HIM.

    His turns carry his canonical name because they pass the identity resolver; the
    live room read is the one path that bypasses it. On 2026-09-14 that had her tell
    him "the word came from Sira, not from you" — Sira being his own display name.
    """
    aliases = [a for a in (aliases or []) if a]
    if not aliases:
        return Signal('owner_named_in_room', SKIP, 'no aliases configured to look for', 0)
    with_room = [t for t in turns if (t.get('room_context') or '').strip()]
    offenders = []
    for turn in with_room:
        for alias in aliases:
            if re.search(rf'(?:^|\n)\s*{re.escape(alias)}\s*:', turn['room_context']):
                offenders.append((turn.get('id'), alias))
                break
    return _result(
        'owner_named_in_room', offenders, len(with_room),
        ok_detail='his room lines carry his own name',
        bad_detail=lambda bad: 'his display name appears unresolved in the room on turn(s) '
                               + ', '.join(f'{i} (as {a})' for i, a in bad[:5]))


def memory_not_truncated(memories) -> Signal:
    """No memory should sit at exactly the limit — that is a cut, not a coincidence."""
    at_limit = [m for m in memories if m.get('length') == FACT_LIMIT]
    return _result(
        'memory_not_truncated', at_limit, len(memories),
        ok_detail=f'no memory sits at exactly {FACT_LIMIT} characters',
        bad_detail=lambda bad: f"{len(bad)} memory/memories cut at {FACT_LIMIT}: "
                               + ', '.join(str(m.get('id')) for m in bad[:5]))


def typed_replies_untagged(turns) -> Signal:
    """A reply on a typed surface should carry no emotion tags (board #14)."""
    typed = [t for t in turns if t.get('channel') not in ('voice',) and t.get('emitted_reply')]
    typed = [t for t in typed if t.get('channel') != 'voice']
    tagged = [t for t in typed if TAGGED.match(t['emitted_reply'])]
    return _result(
        'typed_replies_untagged', tagged, len(typed),
        ok_detail='no typed reply carried a spoken-surface tag',
        bad_detail=lambda bad: f"{len(bad)} typed repl(y/ies) carried an emotion tag: "
                               + ', '.join(str(t.get('id')) for t in bad[:5]))


def quarantine_not_noisy(held) -> Signal:
    """The held queue should not be holding her own work (board #15)."""
    unexpected = [h for h in held
                  if not any(str(h.get('reason') or '').startswith(p)
                             for p in EXPECTED_HOLD_REASONS)]
    return _result(
        'quarantine_not_noisy', unexpected, len(held),
        ok_detail='everything held is held for a reason we expect',
        bad_detail=lambda bad: f"{len(bad)} item(s) held on a detector rule: "
                               + ', '.join(f"{h.get('id')} ({h.get('reason')})" for h in bad[:3]))


def portable_reaches_house(*, newest_portable_id, visible_ids) -> Signal:
    """The newest portable turn should be readable by the house body (board #12)."""
    if newest_portable_id is None:
        return Signal('portable_reaches_house', SKIP, 'no portable turns to carry', 0)
    if newest_portable_id in set(visible_ids or ()):
        return Signal('portable_reaches_house', PASS,
                      'the house can see her newest portable turn', 1)
    return Signal('portable_reaches_house', FAIL,
                  f'the house cannot see portable turn {newest_portable_id}', 1)


# ── the run ────────────────────────────────────────────────────────────────

def should_speak(results) -> bool:
    """Only a failure is worth his attention. A monitor that talks when nothing is
    wrong is one he learns to ignore, and then it is worse than nothing."""
    return any(r.failed for r in results)


def format_report(results) -> str:
    order = {FAIL: 0, SKIP: 1, PASS: 2}
    lines = []
    for r in sorted(results, key=lambda r: (order.get(r.status, 3), r.name)):
        mark = {FAIL: 'FAIL', PASS: 'ok  ', SKIP: 'skip'}[r.status]
        lines.append(f'{mark} {r.name} ({r.checked} checked) — {r.detail}')
    return '\n'.join(lines)


# ── reading the traffic (SELECT only; the worker holds a read-only connection) ──

WINDOW_TURNS_SQL = """\
SELECT id, channel, display_name, emitted_reply, room_context
FROM v2_turns
WHERE created_at > now() - %(window)s::interval
"""

MEMORY_LENGTHS_SQL = """\
SELECT id::text, length(content) AS length
FROM memories
WHERE deleted_at IS NULL AND created_at > now() - %(window)s::interval
"""

HELD_SQL = """\
SELECT id::text, reason FROM portable_held WHERE decision IS NULL
"""

NEWEST_PORTABLE_SQL = """\
SELECT id FROM v2_turns
WHERE channel_id LIKE 'portable:%%' AND person_id = %(person_id)s
ORDER BY created_at DESC LIMIT 1
"""

#: The same read the house makes when it splices her portable turns into a turn of
#: his — so this checks the ACTUAL path rather than a restatement of it.
PORTABLE_VISIBLE_SQL = """\
SELECT id FROM v2_turns
WHERE person_id = %(person_id)s
  AND channel_id LIKE 'portable:%%'
  AND input_text IS NOT NULL AND input_text <> ''
ORDER BY created_at DESC
LIMIT %(limit)s
"""


def run_signals(*, turns_conn, memories_conn=None, person_id=None, aliases=(),
                window='24 hours', portable_limit=100):
    """Every signal, over one window. Returns the results; never writes anything.

    memories_conn is separate because the memories live in the prod database while
    the turns and the quarantine live in the brain's own — the same split every other
    worker here observes.
    """
    turns = [dict(zip(('id', 'channel', 'display_name', 'emitted_reply', 'room_context'), row))
             for row in turns_conn.execute(WINDOW_TURNS_SQL, {'window': window}).fetchall()]
    held = [dict(zip(('id', 'reason'), row))
            for row in turns_conn.execute(HELD_SQL).fetchall()]

    results = [
        room_on_public_turns(turns),
        owner_named_in_room(turns, aliases=aliases),
        typed_replies_untagged(turns),
        quarantine_not_noisy(held),
    ]

    if memories_conn is not None:
        memories = [dict(zip(('id', 'length'), row))
                    for row in memories_conn.execute(MEMORY_LENGTHS_SQL,
                                                     {'window': window}).fetchall()]
        results.append(memory_not_truncated(memories))

    if person_id:
        newest = turns_conn.execute(NEWEST_PORTABLE_SQL, {'person_id': person_id}).fetchone()
        visible = turns_conn.execute(PORTABLE_VISIBLE_SQL,
                                     {'person_id': person_id, 'limit': portable_limit}).fetchall()
        results.append(portable_reaches_house(
            newest_portable_id=newest[0] if newest else None,
            visible_ids=[row[0] for row in visible]))
    return results
