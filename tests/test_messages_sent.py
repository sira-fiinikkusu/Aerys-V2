from datetime import datetime, timedelta, timezone

import pytest

from test_send_message import setup, call
from test_hands_gate import TEXT, OWNER_THREAD


def read(sender, limit=5, identity=TEXT, thread=OWNER_THREAD):
    return sender.messages_sent(limit, identity, thread)


@pytest.mark.parametrize('status,expected', [('ok', 'sent'), ('failed', "didn't go through"),
                                            ('pending', 'outcome unknown')])
def test_delivery_status(status, expected):
    b, sender, _, ledger = setup()
    sender.names = lambda ids: {pid: 'Sam' for pid in ids}
    ledger.reserve('person', 'discord', 'hash')
    ledger.rows[0].update(status=status, time=datetime(2026, 11, 12, 3, 41, tzinfo=timezone.utc))
    assert read(sender) == f'Sam · Discord · Wed 10:41 PM · {expected}'


@pytest.mark.parametrize('status,expected', [('pending', "waiting on Chris's tap"),
    ('declined', 'Chris declined it'), ('expired', 'the button expired'),
    ('approved', 'outcome unknown'), ('withdrawn', 'outcome unknown')])
def test_button_status_before_watcher(status, expected):
    b, sender, _, _ = setup()
    call(sender, 'private words')
    next(iter(b.approval_store.rows.values())).status = status
    out = read(sender)
    assert expected in out
    assert all(secret not in out for secret in ('private words', 'sam_42', '100000000000000002'))


def test_approved_delivery_merges_and_old_delivery_does_not():
    b, sender, _, ledger = setup()
    call(sender)
    row = next(iter(b.store.rows.values()))
    b.tap()
    b.watcher.tick()
    assert read(sender).count('\n') == 0
    assert read(sender).endswith(' · sent')
    ledger.rows[0]['time'] = row.created_at - timedelta(seconds=1)
    assert len(read(sender).splitlines()) == 2
    assert 'outcome unknown' in read(sender)


def test_order_limit_and_missing_name():
    b, sender, _, ledger = setup()
    for i in range(15):
        ledger.reserve(str(i), 'discord', str(i))
        ledger.rows[-1]['time'] = datetime(2026, 11, 12, i, tzinfo=timezone.utc)
    sender.names = lambda ids: {pid: f'Person {pid}' for pid in ids}
    assert len(read(sender).splitlines()) == 5
    lines = read(sender, 100).splitlines()
    assert len(lines) == 10
    assert lines[0].startswith('Person 14 ·') and lines[-1].startswith('Person 5 ·')
    sender.names = lambda ids: (_ for _ in ()).throw(OSError('private'))
    assert read(sender, 1).startswith('someone ·')


@pytest.mark.parametrize('identity,thread', [({}, OWNER_THREAD), ({'user_id': 'guest'}, OWNER_THREAD),
                                           (TEXT, 'kael:test')])
def test_owner_only_before_reads(identity, thread):
    _, sender, _, ledger = setup()
    ledger.recent = lambda *a: pytest.fail('unauthorized read')
    assert 'only the owner' in read(sender, identity=identity, thread=thread)


@pytest.mark.parametrize('source', ['ledger', 'gate', 'approval'])
def test_read_failure(source):
    b, sender, _, ledger = setup()
    call(sender)
    def fail(*args):
        raise OSError('private details')
    if source == 'ledger':
        ledger.recent = fail
    elif source == 'gate':
        b.store.recent_messages = fail
    else:
        b.approvals.status = lambda *a: 'unknown'
    assert read(sender) == "I couldn't read my send record just now"


def test_tool_runtime_and_schema_on_engine():
    from aerys_v2.tools.send_message import build_messages_sent_tool
    from aerys_v2.factory import build_action_graph
    from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
    _, sender, _, ledger = setup()
    ledger.reserve('person', 'discord', 'hash')
    t = build_messages_sent_tool(sender)
    assert set(t.tool_call_schema.model_json_schema()['properties']) == {'limit'}
    graph = build_action_graph(GenericFakeChatModel(messages=iter([
        AIMessage(content='', tool_calls=[{'name': 'messages_sent', 'args': {}, 'id': 'r'}]),
        AIMessage(content='done')])), 'test', [t])
    result = graph.invoke({'messages': [HumanMessage(content='did it go?')]},
                          {'configurable': {'identity': TEXT, 'thread_id': OWNER_THREAD}})
    assert any(isinstance(m, ToolMessage) and 'outcome unknown' in m.content for m in result['messages'])


@pytest.mark.parametrize('delivery', ['failed', 'pending'])
def test_approved_failure_and_uncertain_delivery_merge(delivery):
    b, sender, _, ledger = setup()
    sender.deliver = lambda *a: (_ for _ in ()).throw(OSError())
    call(sender)
    b.tap(); b.watcher.tick()
    ledger.rows[0]['status'] = delivery
    out = read(sender)
    assert len(out.splitlines()) == 1
    assert out.endswith("didn't go through" if delivery == 'failed' else 'outcome unknown')


@pytest.mark.parametrize('approve,expected', [(False, 'Chris declined it'), (None, 'the button expired')])
def test_dropped_button(approve, expected):
    b, sender, _, _ = setup()
    call(sender)
    if approve is False:
        b.tap(approve=False)
    else:
        b.clock['now'] += timedelta(days=1)
    b.watcher.tick()
    assert next(iter(b.store.rows.values())).status == 'dropped'
    assert expected in read(sender)


def test_waiting_is_separate_from_same_payload_delivery():
    b, sender, _, ledger = setup()
    call(sender)
    g = next(iter(b.store.rows.values()))
    import hashlib
    ledger.reserve(g.args['person_id'], 'discord', hashlib.sha256(g.args['text'].encode()).hexdigest())
    ledger.finish(1, True)
    out = read(sender)
    assert len(out.splitlines()) == 2
    assert out.splitlines()[0].endswith(' · sent')
    assert out.splitlines()[1].endswith("waiting on Chris's tap")


def test_postgres_readers_and_display_name_lookup(monkeypatch):
    from aerys_v2.tools.send_message import PostgresMessageSends, message_names_for
    from aerys_v2.hands_gate import PostgresGatedActionStore, MESSAGE
    from types import SimpleNamespace
    from uuid import UUID
    import psycopg
    now = datetime.now(timezone.utc)
    pid = UUID('00000000-0000-4000-8000-000000000002')
    calls = []
    class Conn:
        def __enter__(self):
            return self
        def __exit__(self, *a):
            pass
        def execute(self, sql, args):
            calls.append((sql, args))
            if 'v2_message_sends' in sql:
                rows = [(pid, 'discord', 'hash', now, 'ok')]
            elif 'v2_gated_actions' in sql:
                rows = [(1, 2, MESSAGE, 'key', 1, 'thread', {}, 'waiting', now, None, None)]
            else:
                rows = [(pid, 'Sam')]
            return SimpleNamespace(fetchall=lambda: rows)
    ledger = PostgresMessageSends('unused')
    gate = PostgresGatedActionStore('unused')
    monkeypatch.setattr(ledger, '_connect', Conn)
    monkeypatch.setattr(gate, '_connect', Conn)
    monkeypatch.setattr(psycopg, 'connect', lambda *a, **k: Conn())
    assert ledger.recent(5) == [dict(person=str(pid), platform='discord', hash='hash', time=now, status='ok')]
    assert gate.recent_messages(5)[0].action == MESSAGE
    assert message_names_for('unused')({str(pid)}) == {str(pid): 'Sam'}
    assert calls[0][1] == (5,) and 'ORDER BY attempted_at DESC, id DESC LIMIT %s' in calls[0][0]
    assert calls[1][1] == (MESSAGE, 5) and 'ORDER BY created_at DESC, id DESC LIMIT %s' in calls[1][0]
    assert calls[2][1] == ([str(pid)],)
    assert all('address' not in sql and 'handle' not in sql for sql, _ in calls)
