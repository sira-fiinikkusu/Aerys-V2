"""D49: she can withdraw a mistaken request, never grant it herself."""
import asyncio
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from aerys_v2.approvals import Approvals, MemoryApprovalStore, handle_click, poll_once
from aerys_v2.hands_gate import HandsGate, HandsWatcher, MemoryGatedActionStore

NOW = datetime(2026, 11, 14, tzinfo=timezone.utc)


@pytest.fixture
def desk():
    from aerys_v2.tools.pending_requests import build_pending_requests_tool
    store = MemoryApprovalStore()
    approvals = Approvals(store, clock=lambda: NOW)
    return store, approvals, build_pending_requests_tool(approvals)


def test_list_only_live_pending_and_card_fields(desk):
    store, approvals, tool = desk
    aid = approvals.request('message', 'Send to Sam', detail='private detail')
    multi = approvals.request('job', 'Start a job', approvers=['123', '456'])
    for status in ('approved', 'declined', 'expired', 'withdrawn'):
        other = approvals.request('job', 'hidden ' + status)
        store.rows[other].status = status
    stale = approvals.request('job', 'hidden elapsed')
    store.rows[stale].expires_at = NOW
    store.rows[aid].message_id = 'already-posted'
    result = tool.invoke({'action': 'list'})
    assert f'#{aid}' in result and f'#{multi}' in result
    assert 'message' in result and 'Send to Sam' in result
    assert NOW.isoformat() in result and 'owner' in result
    assert '123' in result and '456' in result
    assert 'hidden' not in result and 'private detail' not in result


def test_empty_list(desk):
    assert 'No pending' in desk[2].invoke({'action': 'list'})


@pytest.mark.parametrize('status', ['approved', 'declined', 'expired', 'withdrawn', 'elapsed'])
def test_refuse_terminal_or_elapsed(desk, status):
    store, approvals, tool = desk
    aid = approvals.request('job', 'Example job')
    if status == 'elapsed':
        store.rows[aid].expires_at = NOW
    else:
        store.rows[aid].status = status
    result = tool.invoke({'action': 'withdraw', 'approval_id': aid})
    assert 'Not withdrawn' in result
    assert ('expired' if status == 'elapsed' else status) in result
    assert store.rows[aid].status == ('pending' if status == 'elapsed' else status)


def test_unknown_and_missing_id(desk):
    assert 'unknown' in desk[2].invoke({'action': 'withdraw', 'approval_id': 999})
    assert 'approval_id' in desk[2].invoke({'action': 'withdraw'})


@pytest.mark.parametrize('action', ['approve', 'decline', 'cancel_all', 'bogus'])
def test_no_other_verbs(desk, action):
    store, approvals, tool = desk
    aid = approvals.request('job', 'Example job')
    assert 'Only list or withdraw' in tool.invoke({'action': action, 'approval_id': aid})
    assert store.rows[aid].status == 'pending'


@pytest.mark.parametrize('approvers', [None, ['123', '456']])
def test_withdraw_voids_cards_and_drops_hands_even_after_late_tap(desk, approvers, monkeypatch):
    store, approvals, tool = desk
    aid = approvals.request('message', 'Send to Sam', approvers=approvers)
    untouched = approvals.request('job', 'Keep waiting')
    rows, executed, notes, edits = MemoryGatedActionStore(), [], [], []
    row = rows.add(aid, 'message', 'one', 1, 'thread', {}, NOW)
    gate = HandsGate(approvals, rows, owner_person_id='owner', clock=lambda: NOW,
                     executors={'message': lambda r: executed.append(r) or (True, 'sent')})
    watcher = HandsWatcher(gate, note=lambda thread, text: notes.append(text))

    async def send(a):
        return 'channel', 'message-' + str(a.recipient)

    async def edit(a, text):
        edits.append((a.id, text))

    # Keep the real poller, but run its synchronous store calls inline for socket-restricted CI.
    async def inline(fn, *args, **kwargs):
        return fn(*args, **kwargs)
    monkeypatch.setattr(asyncio, 'to_thread', inline)
    asyncio.run(poll_once(store, send=send, edit=edit, now=NOW))
    assert 'Withdrawn' in tool.invoke({'action': 'withdraw', 'approval_id': aid})
    assert store.get(aid).status == 'withdrawn'
    assert store.get(untouched).status == 'pending'
    assert handle_click(store, aid, True, 123, 123, NOW)[1] is None
    assert not approvals.approved(aid)
    assert watcher.tick()['dropped'] == 1
    assert watcher.tick()['done'] == 0
    assert row.status == 'dropped' and row.outcome == 'withdrawn'
    assert executed == [] and "wasn't done" in notes[0]
    asyncio.run(poll_once(store, send=send, edit=edit, now=NOW))
    assert len(edits) == (len(approvers) if approvers else 1)
    assert all(i == aid and 'Withdrawn, no longer needed' in text for i, text in edits)


def test_tap_wins_between_read_and_withdraw(desk, monkeypatch):
    store, approvals, tool = desk
    aid = approvals.request('job', 'Example job')
    original = store.withdraw

    def raced(aid, now):
        handle_click(store, aid, True, 123, 123, now)
        return original(aid, now)

    monkeypatch.setattr(store, 'withdraw', raced)
    result = tool.invoke({'action': 'withdraw', 'approval_id': aid})
    assert 'Not withdrawn' in result and 'approved' in result
    assert store.get(aid).status == 'approved'


def test_expiry_wins_at_store_boundary(desk, monkeypatch):
    store, approvals, tool = desk
    aid = approvals.request('job', 'Example job')
    original = store.withdraw
    monkeypatch.setattr(store, 'withdraw', lambda aid, now: original(aid, now + timedelta(hours=1)))
    assert 'Not withdrawn' in tool.invoke({'action': 'withdraw', 'approval_id': aid})
    assert store.get(aid).status == 'pending'


def test_storage_failure_is_honest(desk, monkeypatch):
    store, approvals, tool = desk
    aid = approvals.request('job', 'Example job')
    def fail(*args):
        raise ConnectionError('offline')
    monkeypatch.setattr(store, 'pending', fail)
    assert 'could not read' in tool.invoke({'action': 'list'})
    monkeypatch.setattr(store, 'withdraw', fail)
    reply = tool.invoke({'action': 'withdraw', 'approval_id': aid})
    assert 'Not withdrawn' in reply and 'still waiting on his tap' in reply


def test_owner_only_wiring(monkeypatch, desk):
    from aerys_v2.config import Settings
    from aerys_v2.factory import action_tools_for
    monkeypatch.setattr('aerys_v2.approvals.approvals_for', lambda settings: desk[1])
    settings = Settings(_env_file=None, anthropic_api_key="placeholder")
    assert 'my_pending_requests' in {t.name for t in action_tools_for(settings)}
    assert 'my_pending_requests' not in {t.name for t in action_tools_for(settings, guest=True)}
    monkeypatch.setattr('aerys_v2.approvals.approvals_for', lambda settings: None)
    assert 'my_pending_requests' not in {t.name for t in action_tools_for(settings)}


def test_outside_text_does_not_gate_withdrawal(desk):
    from langchain_core.messages import ToolMessage
    from aerys_v2.outside_guard import outside_guard
    store, approvals, tool = desk
    aid = approvals.request('job', 'Example job')
    request = SimpleNamespace(
        tool_call={'name': tool.name, 'id': 'call', 'args': {'action': 'withdraw', 'approval_id': aid}},
        runtime=SimpleNamespace(config={'configurable': {}}),
        state={'messages': [ToolMessage(content='outside', name='read_web_page', tool_call_id='read')]})
    result = outside_guard(request, lambda r: tool.invoke(r.tool_call['args']))
    assert 'Withdrawn' in result and store.get(aid).status == 'withdrawn'


@pytest.fixture
def sql_store():
    """Execute production predicates in SQLite; not a PostgreSQL integration test."""
    import sqlite3
    from aerys_v2.approvals import PostgresApprovalStore
    conn = sqlite3.connect(':memory:')
    columns = PostgresApprovalStore.COLUMNS.split(', ')
    conn.execute('CREATE TABLE v2_approvals (' + ', '.join(columns) + ')')
    class SQLStore(PostgresApprovalStore):
        def _one(self, sql, args):
            return conn.execute(sql.replace('%s', '?'), args).fetchone()
        def _all(self, sql, args):
            return [self._row(r) for r in conn.execute(sql.replace('%s', '?'), args).fetchall()]
    for aid, status, expiry in [(1, 'pending', NOW + timedelta(hours=1)),
                                (2, 'pending', NOW), (3, 'approved', NOW + timedelta(hours=1)),
                                (4, 'declined', NOW + timedelta(hours=1)),
                                (5, 'expired', NOW + timedelta(hours=1)),
                                (6, 'withdrawn', NOW + timedelta(hours=1))]:
        conn.execute('INSERT INTO v2_approvals (id, status, expires_at) VALUES (?, ?, ?)',
                     (aid, status, expiry))
    yield SQLStore('unused')
    conn.close()


def test_sql_list_predicates(sql_store):
    assert [r.id for r in sql_store.pending(NOW)] == [1]


@pytest.mark.parametrize('aid', [2, 3, 4, 5, 6, 999])
def test_sql_withdraw_guards(sql_store, aid):
    before = sql_store.get(aid)
    assert not sql_store.withdraw(aid, NOW)
    assert sql_store.get(aid) == before


def test_sql_withdraw_late_tap_and_other_ids(sql_store):
    assert sql_store.withdraw(1, NOW)
    assert sql_store.get(1).status == 'withdrawn'
    assert sql_store.get(3).status == 'approved'
    assert sql_store.decide(1, True, '123', NOW) is None


def test_sql_tap_before_withdraw(sql_store):
    assert sql_store.decide(1, True, '123', NOW).status == 'approved'
    assert not sql_store.withdraw(1, NOW)
    assert sql_store.get(1).status == 'approved'


def test_withdraw_targets_selected_id_not_first_or_all(desk):
    store, approvals, tool = desk
    first = approvals.request('job', 'Keep this one')
    selected = approvals.request('message', 'Withdraw this one')
    assert 'unknown' in tool.invoke({'action': 'withdraw', 'approval_id': 999})
    assert store.get(first).status == 'pending'
    assert 'Withdrawn' in tool.invoke({'action': 'withdraw', 'approval_id': selected})
    assert store.get(first).status == 'pending'
    assert store.get(selected).status == 'withdrawn'


def test_simultaneous_tap_and_withdraw_have_one_winner(desk):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier
    store, approvals, tool = desk
    aid = approvals.request('job', 'Example job')
    ready = Barrier(2)
    def withdraw():
        ready.wait(timeout=2)
        return approvals.withdraw(aid)
    def tap():
        ready.wait(timeout=2)
        return handle_click(store, aid, True, 123, 123, NOW)[1] is not None
    with ThreadPoolExecutor(max_workers=2) as pool:
        withdrawal = pool.submit(withdraw)
        approval = pool.submit(tap)
        withdrew, approved = withdrawal.result(timeout=3), approval.result(timeout=3)
    assert withdrew != approved
    assert store.get(aid).status == ('withdrawn' if withdrew else 'approved')


def test_withdrawn_card_edit_removes_buttons(desk):
    from aerys_v2.approvals import edit_dm, outcome_text
    store, approvals, tool = desk
    aid = approvals.request('job', 'Example job')
    row = store.get(aid)
    row.channel_id, row.message_id = '123', '456'
    tool.invoke({'action': 'withdraw', 'approval_id': aid})
    edits = []
    async def edit(**kwargs):
        edits.append(kwargs)
    class Channel:
        def get_partial_message(self, message_id):
            assert message_id == 456
            return SimpleNamespace(edit=edit)
    class Client:
        def get_partial_messageable(self, channel_id, **kwargs):
            assert channel_id == 123
            return Channel()
    asyncio.run(edit_dm(Client(), row, outcome_text(row)))
    assert edits == [{'content': '**Withdrawn, no longer needed** (job)\nExample job', 'view': None}]
