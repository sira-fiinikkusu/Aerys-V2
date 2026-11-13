from datetime import timedelta
from test_pass_to_chris import setup, CONTACT
from test_reach_desk import DAY, World, desk_for, ev
from aerys_v2.reach import Prefs


def test_relay_budget_and_spontaneous_budget():
    world = World()
    desk, clock = desk_for(world, now=DAY)
    desk.store._prefs = Prefs(budget=0)
    assert desk.reach_out(ev(source='relay', topic='messages')) == 'sent'
    assert desk.store.sent_since(DAY) == 0
    assert desk.reach_out(ev()) == 'dropped'


def test_unlimited_distinct_and_rolling_duplicate():
    relay, sent, desk = setup()
    for i in range(5):
        assert relay.send(str(i), CONTACT, '').startswith('Queued for Chris.')
    assert 'already queued' in relay.send('0', CONTACT, '')
    assert len(desk.store.rows) == 5
    desk.clock = lambda: DAY + timedelta(minutes=10, seconds=1)
    assert relay.send('0', CONTACT, '').startswith('Queued for Chris.')
    assert len(desk.store.rows) == 6


def test_status_scoped_and_honest():
    relay, sent, desk = setup()
    relay.send('waiting words', CONTACT, '')
    relay.send('private other words', dict(CONTACT, user_id='second'), '')
    assert 'waiting' in relay.status(CONTACT, '')
    assert 'private other words' not in relay.status(CONTACT, '')
    desk.tick()
    assert 'delivered (DM, Mon 2:00 PM)' in relay.status(CONTACT, '')
    row = next(iter(desk.store.rows.values()))
    desk.store.mark(row.id, DAY, status='dropped', dm_at=None, delivered_at=None, reason='you muted messages')
    assert 'not delivered (you muted messages)' in relay.status(CONTACT, '')


def test_relay_meeting_wait():
    world = World()
    meeting = {'until': DAY + timedelta(minutes=20)}
    desk, clock = desk_for(world, now=DAY, meeting=meeting)
    assert desk.reach_out(ev(source='relay', topic='messages')) == 'queued'
    assert not world.spoken
    clock['now'] = meeting['until']
    meeting['until'] = None
    desk.tick()
    assert len(world.spoken) == 1


def test_lens_is_personal_sticky_shared_unsure_unchanged():
    from aerys_v2.outside_guard import origin_sources
    from test_hands_gate import TEXT, OWNER, OWNER_THREAD
    for identity in [dict(TEXT, surface='lens'), dict(TEXT, platform='lens')]:
        assert 'shared surface' not in origin_sources(identity, OWNER_THREAD, OWNER)
    assert 'shared surface' in origin_sources(dict(TEXT, surface='sticky'), OWNER_THREAD, OWNER)
    assert 'shared surface' in origin_sources(dict(TEXT, device_id='shared'), OWNER_THREAD, OWNER, ['shared'])
    assert 'unsure voice' in origin_sources(dict(TEXT, surface='lens', speaker={'unsure': True}), OWNER_THREAD, OWNER)


def test_status_window_order_failures_and_caller_rules(monkeypatch):
    relay, sent, desk = setup()
    relay.send('old', CONTACT, '')
    desk.clock = lambda: DAY + timedelta(hours=25)
    relay.send('new', CONTACT, '')
    relay.send('newest', CONTACT, '')
    result = relay.status(CONTACT, '')
    assert 'old' not in result and result.index('newest') < result.index('"new"')
    for identity, thread in [(dict(CONTACT, channel_kind='guild'), ''), (CONTACT, 'kael:test'),
                             (dict(CONTACT, user_id='owner'), ''), ({}, '')]:
        assert 'only a listed contact' in relay.status(identity, thread)
    monkeypatch.setattr(desk.store, 'relays_since', lambda *a: 1 / 0)
    assert relay.status(CONTACT, '') == "I couldn't check just now."


def test_relay_status_tool_injected_runtime():
    from aerys_v2.tools.pass_to_chris import build_relay_status_tool
    from aerys_v2.factory import build_action_graph
    from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
    relay, sent, desk = setup()
    relay.send('hello', CONTACT, '')
    tool = build_relay_status_tool(relay)
    assert not tool.tool_call_schema.model_json_schema()['properties']
    model = GenericFakeChatModel(messages=iter([AIMessage(content='', tool_calls=[
        {'name': 'relay_status', 'args': {}, 'id': 's'}]), AIMessage(content='waiting')]))
    result = build_action_graph(model, 'test', [tool]).invoke(
        {'messages': [HumanMessage(content='Did he get it?')]},
        {'configurable': {'identity': CONTACT, 'thread_id': 'person:contact'}})
    assert any(isinstance(m, ToolMessage) and 'waiting (why:' in m.content for m in result['messages'])


def test_same_clipped_words_distinct_full_messages_and_people():
    relay, sent, desk = setup()
    for words, identity in [('x' * 500 + 'a', CONTACT), ('x' * 500 + 'b', CONTACT),
                            ('x' * 500 + 'a', dict(CONTACT, user_id='second'))]:
        assert relay.send(words, identity, '').startswith('Queued for Chris.')
    assert len(desk.store.rows) == 3


def test_status_meeting_quiet_nap_and_muted():
    relay, sent, desk = setup()
    relay.send('hello', CONTACT, '')
    desk.meeting_hold = lambda: DAY + timedelta(hours=1)
    desk.tick()
    assert 'waiting (why: meeting:' in relay.status(CONTACT, '')
    desk.meeting_hold = lambda: None
    row = next(iter(desk.store.rows.values()))
    row.not_before = None
    desk.quiet = lambda now: True
    desk.tick()
    assert 'waiting (why: quiet hours:' in relay.status(CONTACT, '')
    desk.quiet = lambda now: False
    row.not_before = None
    desk.store._prefs = Prefs(hold_since=DAY, hold_until=DAY + timedelta(hours=1))
    desk.tick()
    assert 'waiting (why: nap:' in relay.status(CONTACT, '')
    row.not_before = None
    desk.store._prefs = Prefs(muted=['messages'])
    desk.tick()
    assert 'not delivered (you muted messages)' in relay.status(CONTACT, '')
    assert not sent


def test_postgres_atomic_dedupe_and_scoped_queries(monkeypatch):
    from types import SimpleNamespace
    from aerys_v2.reach import PostgresReachStore
    from aerys_v2.tools.pass_to_chris import relay_prefix
    import pytest
    relay, sent, desk = setup()
    calls = []
    state = {'exists': False, 'fail_commit': False}
    class Conn:
        def __enter__(self):
            calls.append(('begin', ()))
            return self
        def __exit__(self, *args):
            calls.append(('commit', ()))
            if state['fail_commit']:
                raise OSError('commit failed')
        def execute(self, sql, args):
            calls.append((sql, args))
            return SimpleNamespace(fetchone=lambda: (1,) if state['exists'] else None, fetchall=lambda: [])
    store = PostgresReachStore('unused')
    monkeypatch.setattr(store, '_connect', Conn)
    event = ev(source='relay', topic='messages', dedupe_key=relay_prefix('contact') + 'abc')
    assert store.enqueue_relay(event, DAY)
    assert ['begin', 'SELECT', 'SELECT', 'INSERT', 'commit'] == [sql.split()[0] for sql, _ in calls]
    assert calls[2][1] == (event.dedupe_key, DAY - timedelta(minutes=10))
    state['exists'] = True
    calls.clear()
    assert not store.enqueue_relay(event, DAY)
    assert not any(sql.startswith('INSERT') for sql, _ in calls)
    state.update(exists=False, fail_commit=True)
    with pytest.raises(OSError):
        store.enqueue_relay(event, DAY)
    state['fail_commit'] = False
    calls.clear()
    assert store.relays_since(relay_prefix('contact'), DAY) == []
    assert calls[1][1] == (relay_prefix('contact') + '%', DAY)
    assert "source = 'relay'" in calls[1][0] and 'ORDER BY created_at DESC, id DESC' in calls[1][0]
    state['exists'] = True
    calls.clear()
    assert store.sent_since(DAY) == 1
    assert "source <> 'relay'" in calls[1][0]


def test_status_spoken_eastern_time_and_legacy_privacy():
    from datetime import datetime, timezone
    relay, sent, desk = setup()
    now = datetime(2026, 11, 12, 18, 45, tzinfo=timezone.utc)
    desk.clock = lambda: now
    relay.send('spoken words', CONTACT, '')
    row = next(iter(desk.store.rows.values()))
    desk.store.mark(row.id, now, status='sent', spoken_at=now, delivered_at=now)
    desk.store.add(ev(source='relay', headline='Unattributed legacy words'), now,
                   status='sent', reason='', not_before=None)
    result = relay.status(CONTACT, '')
    assert 'delivered (spoken, Thu 1:45 PM)' in result
    assert 'Unattributed legacy words' not in result
