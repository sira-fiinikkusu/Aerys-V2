"""Invariants over real traffic — the checks that would have caught this week.

Chris, 2026-09-14, after six straight days of a clean gap board while he personally
found four real bugs: the miner catches what she NOTICES she cannot do, and every bug
that week was one she did not notice. She answered honestly and moved on, which is
what we asked of her. He named the shape he wanted, from Arize's Signal: evaluate the
traffic that already happened rather than inject synthetic traffic.

That is the right call and it is also why this does not plant facts. A plant-and-ask
probe has to WRITE — a synthetic turn in his thread, junk in her memory, noise in the
audit — so testing the real path means polluting the real path. Every check here is a
statement about rows that already exist.

Each signal is one bug from 2026-09-13/14, turned into something that cannot come back
quietly:

  room_on_public_turns   the room going missing from a public turn   (#17)
  owner_named_in_room    his own lines reading as a stranger          (Sira/Chris)
  memory_not_truncated   a fact stored cut in half                    (#16)
  typed_replies_untagged her imitating her own voice replies          (#14)
  quarantine_not_noisy   the detector holding her own work            (#15)
  portable_reaches_house the cross-body loop breaking                 (#12)

A signal returns PASS, FAIL or SKIP. SKIP is load-bearing: no public turns in the
window is not a passing room check, and a green run built on absent evidence is how a
monitor becomes a comfort blanket.
"""
import pytest

signals = pytest.importorskip('aerys_v2.workers.signals',
                              reason='the signals worker is not built yet')


def rows(*dicts):
    return list(dicts)


# ── the shape of a result ───────────────────────────────────────────────────

def test_a_signal_reports_pass_fail_or_skip_and_says_why():
    result = signals.Signal('x', signals.PASS, 'nothing wrong', checked=4)
    assert result.status in (signals.PASS, signals.FAIL, signals.SKIP)
    assert result.detail and result.checked == 4
    assert not result.failed


def test_absent_evidence_is_a_SKIP_not_a_PASS():
    """The most important line in the file. A check with nothing to look at has not
    passed; a run that reports green off zero rows is how a monitor stops meaning
    anything."""
    result = signals.room_on_public_turns([])
    assert result.status == signals.SKIP
    assert not result.failed


# ── the six ────────────────────────────────────────────────────────────────

def test_a_public_turn_missing_its_room_fails():
    good = {'id': 1, 'channel': 'guild', 'room_context': 'Stratus: hi', 'emitted_reply': 'ok'}
    bad = {'id': 2, 'channel': 'guild', 'room_context': None, 'emitted_reply': 'ok'}
    assert signals.room_on_public_turns(rows(good)).status == signals.PASS
    failure = signals.room_on_public_turns(rows(good, bad))
    assert failure.failed and '2' in failure.detail


def test_his_own_room_line_reading_as_a_stranger_fails():
    """The exact bug: his turns said Chris, his room lines said Sira, and she told him
    the word came from someone else."""
    good = {'id': 1, 'channel': 'guild', 'display_name': 'Chris',
            'room_context': 'Chris: pickle\nStratus: hi'}
    bad = {'id': 2, 'channel': 'guild', 'display_name': 'Chris',
           'room_context': 'Sira: pickle\nStratus: hi'}
    assert signals.owner_named_in_room(rows(good), aliases=['Sira']).status == signals.PASS
    failure = signals.owner_named_in_room(rows(good, bad), aliases=['Sira'])
    assert failure.failed and 'Sira' in failure.detail


def test_a_memory_at_exactly_the_limit_fails():
    limit = signals.FACT_LIMIT
    assert signals.memory_not_truncated([{'id': 'a', 'length': limit - 1}]).status == signals.PASS
    failure = signals.memory_not_truncated([{'id': 'b', 'length': limit}])
    assert failure.failed and 'b' in failure.detail


def test_a_tagged_typed_reply_fails_and_a_voice_one_does_not():
    typed = {'id': 1, 'channel': 'discord_dm', 'emitted_reply': '[warmly] Good test.'}
    voice = {'id': 2, 'channel': 'voice', 'emitted_reply': '[warmly] Good test.'}
    assert signals.typed_replies_untagged(rows(voice)).status == signals.SKIP
    assert signals.typed_replies_untagged(rows(typed)).failed


def test_the_quarantine_holding_her_own_work_fails():
    clean = [{'id': 'a', 'reason': 'judge:unclear', 'text': 'something odd'}]
    noisy = [{'id': 'b', 'reason': 'high-entropy credential-shaped blob',
              'text': 'built /home/aerys/work/weather_news/run_report.sh'}]
    assert signals.quarantine_not_noisy(clean).status == signals.PASS
    assert signals.quarantine_not_noisy(noisy).failed


def test_a_portable_turn_the_house_cannot_see_fails():
    assert signals.portable_reaches_house(newest_portable_id=7, visible_ids=[7, 6]).status == signals.PASS
    assert signals.portable_reaches_house(newest_portable_id=8, visible_ids=[7, 6]).failed
    assert signals.portable_reaches_house(newest_portable_id=None, visible_ids=[]).status == signals.SKIP


# ── the run ────────────────────────────────────────────────────────────────

def test_a_green_run_is_silent_and_a_red_one_is_not():
    """A monitor that speaks when nothing is wrong is one he learns to ignore."""
    green = [signals.Signal('a', signals.PASS, 'fine', 3),
             signals.Signal('b', signals.SKIP, 'nothing to check', 0)]
    red = green + [signals.Signal('c', signals.FAIL, 'turn 2 has no room', 4)]
    assert signals.should_speak(green) is False
    assert signals.should_speak(red) is True


def test_the_report_names_what_broke_and_what_was_checked():
    report = signals.format_report([
        signals.Signal('room_on_public_turns', signals.FAIL, 'turn 2 carried no room', 4),
        signals.Signal('memory_not_truncated', signals.PASS, 'none at the limit', 120),
    ])
    assert 'room_on_public_turns' in report and 'turn 2 carried no room' in report
    assert '4' in report and '120' in report


# ── the run, against a fake database ────────────────────────────────────────

class FakeConn:
    """Answers by which SQL it was handed. No network, no server, no writes."""

    def __init__(self, turns=(), held=(), memories=(), newest=None, visible=()):
        self.turns, self.held, self.memories = turns, held, memories
        self.newest, self.visible = newest, visible
        self.statements = []

    def execute(self, sql, params=None):
        self.statements.append(sql)
        if 'FROM v2_turns' in sql and 'room_context' in sql:
            rows = [(t['id'], t['channel'], t.get('display_name'),
                     t.get('emitted_reply'), t.get('room_context'),
                     t.get('created_at')) for t in self.turns]
        elif 'portable_held' in sql:
            rows = [(h['id'], h['reason']) for h in self.held]
        elif 'FROM memories' in sql:
            rows = [(m['id'], m['length']) for m in self.memories]
        elif 'ORDER BY created_at DESC LIMIT 1' in sql:
            rows = [(self.newest,)] if self.newest is not None else []
        else:
            rows = [(i,) for i in self.visible]
        self._rows = rows
        return self

    def fetchall(self):
        return self._rows

    def fetchone(self):
        return self._rows[0] if self._rows else None


def test_a_healthy_window_produces_no_failures_and_says_nothing():
    conn = FakeConn(
        turns=[{'id': 1, 'channel': 'guild', 'display_name': 'Chris',
                'emitted_reply': 'Looks quiet.', 'room_context': 'Chris: pickle'}],
        held=[{'id': 'h1', 'reason': 'judge:unclear'}],
        memories=[{'id': 'm1', 'length': 120}], newest=9, visible=[9, 8])
    results = signals.run_signals(turns_conn=conn, memories_conn=conn,
                                  person_id='p1', aliases=['Sira'])
    assert not signals.should_speak(results), signals.format_report(results)


def test_the_week_that_actually_happened_lights_up():
    """Every bug from 2026-09-13/14 in one window — each must be caught."""
    conn = FakeConn(
        turns=[{'id': 1, 'channel': 'guild', 'display_name': 'Chris',
                'emitted_reply': 'ok', 'room_context': None},
               {'id': 2, 'channel': 'guild', 'display_name': 'Chris',
                'emitted_reply': 'ok', 'room_context': 'Sira: pickle'},
               {'id': 3, 'channel': 'discord_dm', 'display_name': 'Chris',
                'emitted_reply': '[warmly] Good test.', 'room_context': None}],
        held=[{'id': 'h1', 'reason': 'high-entropy credential-shaped blob'}],
        memories=[{'id': 'm1', 'length': signals.FACT_LIMIT}], newest=9, visible=[8])
    results = signals.run_signals(turns_conn=conn, memories_conn=conn,
                                  person_id='p1', aliases=['Sira'])
    failed = {r.name for r in results if r.failed}
    assert failed == {'room_on_public_turns', 'owner_named_in_room',
                      'typed_replies_untagged', 'quarantine_not_noisy',
                      'memory_not_truncated', 'portable_reaches_house'}, failed
    assert signals.should_speak(results)


def test_it_only_ever_reads():
    conn = FakeConn(turns=[], held=[], memories=[])
    signals.run_signals(turns_conn=conn, memories_conn=conn, person_id='p1', aliases=['Sira'])
    assert conn.statements, 'it did run something'
    for sql in conn.statements:
        assert sql.lstrip().upper().startswith('SELECT'), sql


def test_a_failure_says_WHEN_the_newest_offender_was():
    """Without a date every morning reads the same whether the bug is live or was
    fixed yesterday. The first real run proved it: all three failures were rows from
    before that morning's fixes and nothing on screen said so."""
    from datetime import datetime

    when = datetime(2026, 9, 14, 10, 13)
    result = signals.room_on_public_turns([
        {'id': 7, 'channel': 'guild', 'room_context': None, 'created_at': when}])
    assert result.failed and '09-14 10:13' in result.detail, result.detail


def test_undated_rows_simply_omit_the_when():
    result = signals.room_on_public_turns([{'id': 7, 'channel': 'guild', 'room_context': None}])
    assert result.failed and 'newest' not in result.detail
