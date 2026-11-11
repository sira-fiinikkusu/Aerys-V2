from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from aerys_v2.tools.send_message import MessageSends, MessageSender, build_send_message_tool
from aerys_v2.hands_gate import MESSAGE
from test_hands_gate import Box, OWNER, SAM, TEXT, OWNER_THREAD


def setup():
    b = Box()
    sent, ledger = [], MessageSends()
    sender = MessageSender(b.gate, {SAM}, ledger, lambda address, text: sent.append((address, text)))
    b.gate.executors[MESSAGE] = sender.execute
    return b, sender, sent, ledger


def call(sender, text='hello', to='sam', sources=(), identity=None):
    return sender.send(to, text, 'discord', identity or TEXT, OWNER_THREAD, sources)


def confirm(b, sender):
    assert 'Waiting' in call(sender, 'first')
    b.tap(); b.watcher.tick()


def test_first_button_and_watcher_exact_once():
    b, sender, sent, ledger = setup()
    assert 'Waiting' in call(sender)
    row = next(iter(b.approval_store.rows.values()))
    for part in ('Sam', 'Discord', 'sam_42', '100000000000000002', 'hello\n-Aerys'):
        assert part in row.detail
    b.watcher.tick()
    assert not sent
    b.tap(); b.watcher.tick(); b.watcher.tick()
    assert sent == [('100000000000000002', 'hello\n-Aerys')]
    assert ledger.rows[0]['status'] == 'ok'
    assert not call(sender).startswith('Sent:')


def test_confirmed_signed_recorded_deduped():
    b, sender, sent, ledger = setup()
    confirm(b, sender)
    assert call(sender, 'next\n-Aerys').startswith('Sent:')
    assert call(sender, 'next').startswith('Not sent:')
    assert sent[-1][1] == 'next\n-Aerys'
    assert len(sent) == len(ledger.rows) == 2
    assert 'text' not in ledger.rows[-1]


def test_unlisted_ambiguity_and_missing():
    b, sender, sent, ledger = setup()
    sender.people = {'other'}
    assert "isn't on the list" in call(sender)
    assert not b.approval_store.rows
    b.gate.people = lambda *a: [dict(person_id=SAM, name='Test', address='1'),
                               dict(person_id='other', name='Test', address='2')]
    assert 'more than one' in call(sender)
    sender.people = {SAM}
    b.gate.people = lambda *a: [dict(person_id=SAM, name='Test', address='')]
    assert 'address' in call(sender)
    assert not b.approval_store.rows and not sent


def test_failure_recorded_and_not_retried():
    b, sender, sent, ledger = setup()
    confirm(b, sender)
    def fail(*a):
        raise RuntimeError('private server response')
    sender.deliver = fail
    out = call(sender, 'second')
    assert out.startswith('Not sent:') and 'private' not in out
    assert ledger.rows[-1]['status'] == 'failed'
    assert 'already' in call(sender, 'second')


@pytest.mark.parametrize('identity,thread', [({'user_id': 'guest'}, OWNER_THREAD), (TEXT, 'kael:test')])
def test_other_identity_refused(identity, thread):
    b, sender, sent, ledger = setup()
    assert sender.send('sam', 'hello', 'discord', identity, thread, ()).startswith('Not sent:')
    assert not sent and not b.approval_store.rows


@pytest.mark.parametrize('platform', ['email', 'telegram', 'other'])
def test_other_platforms(platform):
    b, sender, sent, ledger = setup()
    out = sender.send('sam', 'hello', platform, TEXT, OWNER_THREAD, ())
    assert out.startswith('Not sent:') and 'send_email' in out and 'Telegram' in out


def test_long_message_refused_before_button():
    b, sender, sent, ledger = setup()
    assert call(sender, 'x' * 1990).startswith('Not sent:')
    assert not b.approval_store.rows


@pytest.mark.parametrize('same_pass', [False, True])
def test_engine_taint(same_pass):
    from aerys_v2.factory import build_action_graph
    from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
    from langchain_core.tools import tool
    b, sender, sent, ledger = setup()
    confirm(b, sender)
    @tool
    def read_email() -> str:
        """Read outside words."""
        return 'outside'
    send = {'name': 'send_message', 'args': {'to': 'sam', 'text': 'next'}, 'id': 's'}
    read = {'name': 'read_email', 'args': {}, 'id': 'r'}
    script = [AIMessage(content='', tool_calls=[read, send])] if same_pass else [
        AIMessage(content='', tool_calls=[read]), AIMessage(content='', tool_calls=[send])]
    graph = build_action_graph(GenericFakeChatModel(messages=iter([*script, AIMessage(content='done')])),
                               'test', [read_email, build_send_message_tool(sender)], outside_gate=b.gate)
    result = graph.invoke({'messages': [HumanMessage(content='send')]}, {'configurable': {
        'identity': TEXT, 'thread_id': OWNER_THREAD, 'outside_text': {}}})
    assert len(sent) == (2 if same_pass else 1)
    if not same_pass:
        assert any(isinstance(m, ToolMessage) and 'Waiting' in m.content for m in result['messages'])
        row = list(b.approval_store.rows.values())[-1]
        assert 'read_email' in row.detail and 'next\n-Aerys' in row.detail
        b.tap(); b.watcher.tick()
        assert sent[-1][1] == 'next\n-Aerys'


def settings(**kw):
    from aerys_v2.config import Settings
    return Settings(_env_file=None, anthropic_api_key='placeholder', **kw)


def test_switch_default_and_owner_tools_unchanged(monkeypatch):
    import json
    from aerys_v2.factory import action_tools_for
    from langchain_core.utils.function_calling import convert_to_openai_tool
    from aerys_v2.config import Settings
    before = settings(ha_token='placeholder')
    assert before.messages_as_herself is False
    # Absent fields model the pre-feature settings; every schema byte must match explicit off.
    del before.messages_as_herself
    del before.message_people
    old = json.dumps([convert_to_openai_tool(t) for t in action_tools_for(before)], sort_keys=True)
    after = settings(ha_token='placeholder', messages_as_herself=False, message_people=SAM)
    assert old == json.dumps([convert_to_openai_tool(t) for t in action_tools_for(after)], sort_keys=True)


@pytest.mark.parametrize('missing', ['switch', 'token', 'people', 'gate', 'lookup', 'database'])
def test_missing_legs(missing, caplog):
    from aerys_v2.tools.send_message import message_sender_for
    b, sender, sent, ledger = setup()
    values = dict(messages_as_herself=True, discord_bot_token='placeholder', message_people=SAM,
                  database_url='postgresql://placeholder/db')
    field = {'switch': 'messages_as_herself', 'token': 'discord_bot_token', 'people': 'message_people',
             'database': 'database_url'}.get(missing)
    if field:
        values[field] = False if missing == 'switch' else '' if missing == 'people' else None
    if missing == 'lookup':
        b.gate.people = None
    with caplog.at_level('INFO'):
        assert message_sender_for(settings(**values), None if missing == 'gate' else b.gate) is None
    assert 'not armed' in caplog.text


def test_factory_tool_and_executor_wired_and_guest_absent(monkeypatch):
    import aerys_v2.hands_gate as module
    from aerys_v2.factory import action_tools_for
    b, sender, sent, ledger = setup()
    monkeypatch.setattr(module, 'approvals_for', lambda s: b.approvals)
    monkeypatch.setattr(module, 'people_lookup_for', lambda s: b.gate.people)
    s = settings(messages_as_herself=True, message_people=SAM, discord_bot_token='placeholder',
                 database_url='postgresql://placeholder/db', hands_gates=True, owner_person_id=OWNER)
    gate = module.hands_gate_for(s)
    assert MESSAGE in gate.executors
    assert gate.message_sender is not None
    assert 'send_message' in [t.name for t in action_tools_for(s)]
    assert 'send_message' not in [t.name for t in action_tools_for(s, guest=True)]
    s.hands_gates = False
    assert 'send_message' not in [t.name for t in action_tools_for(s)]


def test_reservation_failure_prevents_network_and_finish_failure_never_claims_sent():
    b, sender, sent, ledger = setup()
    confirm(b, sender)
    ledger.reserve = lambda *a: (_ for _ in ()).throw(RuntimeError())
    assert call(sender, 'new').startswith('Not sent:') and len(sent) == 1
    sender.ledger = MessageSends()
    sender.ledger.finish = lambda *a: (_ for _ in ()).throw(RuntimeError())
    assert call(sender, 'new').startswith('Not sent:') and len(sent) == 2
    assert call(sender, 'new').startswith('Not sent:') and len(sent) == 2


def test_parallel_direct_calls_send_once():
    from concurrent.futures import ThreadPoolExecutor
    b, sender, sent, ledger = setup()
    confirm(b, sender)
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _: call(sender, 'parallel'), range(8)))
    assert sum(r.startswith('Sent:') for r in results) == 1
    assert len(sent) == 2


def test_owner_after_outside_text_and_origin_taint():
    b, sender, sent, ledger = setup()
    b.gate.people = lambda *a: [dict(person_id=OWNER, name='Owner', address='placeholder-id')]
    assert 'Waiting' in call(sender, to='owner', sources=('room', 'shared surface'))
    assert not sent
    assert 'room, shared surface' in next(iter(b.approval_store.rows.values())).detail


def test_schema_hides_runtime():
    b, sender, sent, ledger = setup()
    assert set(build_send_message_tool(sender).tool_call_schema.model_json_schema()['properties']) == {
        'to', 'text', 'platform'}


def test_approved_failure_is_failed_once_and_never_noted_as_sent():
    b, sender, sent, ledger = setup()
    def fail(*args):
        sent.append('attempt')
        raise OSError('private response')
    sender.deliver = fail
    assert 'Waiting' in call(sender)
    b.tap()
    assert b.watcher.tick()['failed'] == 1
    b.watcher.tick()
    assert sent == ['attempt'] and ledger.rows[0]['status'] == 'failed'
    assert not any('was sent' in text for _, text in b.notes)
    assert 'private response' not in str(b.notes)


def test_lookup_snapshot_is_used_once():
    b, sender, sent, ledger = setup()
    confirm(b, sender)
    original = b.gate.people
    calls = []
    def lookup(*args):
        calls.append(args)
        assert len(calls) == 1
        return original(*args)
    b.gate.people = lookup
    assert call(sender, 'snapshot').startswith('Sent:')
    assert len(calls) == 1


def test_revoked_person_is_not_sent_by_watcher():
    b, sender, sent, ledger = setup()
    assert 'Waiting' in call(sender)
    sender.people.clear()
    b.tap()
    assert b.watcher.tick()['failed'] == 1
    assert not sent and not ledger.rows


def test_postgres_reservation_commits_before_delivery(monkeypatch):
    from aerys_v2.tools.send_message import PostgresMessageSends
    events = []
    class Result:
        def __init__(self, row):
            self.row = row
        def fetchone(self):
            return self.row
    class Conn:
        def __enter__(self):
            events.append('begin')
            return self
        def __exit__(self, *args):
            events.append('commit')
        def execute(self, sql, args):
            events.append(sql.split()[0])
            if sql.startswith('SELECT id'):
                return Result(None)
            return Result((7,))
    b, sender, sent, ledger = setup()
    sender.ledger = PostgresMessageSends('unused')
    monkeypatch.setattr(sender.ledger, '_connect', Conn)
    def deliver(*args):
        assert events == ['begin', 'SELECT', 'SELECT', 'INSERT', 'commit']
        events.append('delivery')
    sender.deliver = deliver
    assert sender._deliver(SAM, 'discord', 'placeholder-id', 'words\n-Aerys') == (True, 'sent')
    assert events[-4:] == ['delivery', 'begin', 'UPDATE', 'commit']


def test_postgres_commit_failure_prevents_delivery(monkeypatch):
    from aerys_v2.tools.send_message import PostgresMessageSends
    class Conn:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            raise OSError('commit failed')
        def execute(self, sql, args):
            return SimpleNamespace(fetchone=lambda: None if sql.startswith('SELECT id') else (7,))
    b, sender, sent, ledger = setup()
    sender.ledger = PostgresMessageSends('unused')
    monkeypatch.setattr(sender.ledger, '_connect', Conn)
    assert sender._deliver(SAM, 'discord', 'placeholder-id', 'words\n-Aerys')[0] is False
    assert not sent


def test_owner_direct_and_recent_window_with_pending_fail_closed():
    from datetime import timedelta
    b, sender, sent, ledger = setup()
    b.gate.people = lambda *a: [dict(person_id=OWNER, name='Owner', address='placeholder-id')]
    assert call(sender, to='owner').startswith('Sent:')
    assert not b.approval_store.rows
    ledger.rows[0]['time'] -= timedelta(minutes=11)
    assert call(sender, to='owner').startswith('Sent:')
    ledger.rows[-1]['status'] = 'pending'
    ledger.rows[-1]['time'] -= timedelta(days=1)
    assert call(sender, to='owner').startswith('Not sent:')
    assert len(sent) == 2


def test_unsure_voice_to_a_confirmed_contact_waits_for_his_tap():
    # D28 #5: an unsure voice (maybe a guest in the room) is outside text, so even a confirmed
    # contact waits for his tap; the same words from his typed line go straight through.
    from aerys_v2.outside_guard import origin_sources
    from test_hands_gate import UNSURE
    b, sender, sent, ledger = setup()
    confirm(b, sender)
    sent.clear()
    sources = origin_sources(UNSURE, OWNER_THREAD, OWNER)
    assert sources
    assert 'Waiting' in call(sender, 'see you at six', sources=tuple(sources), identity=UNSURE)
    assert not sent
    assert call(sender, 'see you at six').startswith('Sent:') and len(sent) == 1
