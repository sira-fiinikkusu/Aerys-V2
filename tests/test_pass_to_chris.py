from types import SimpleNamespace
from datetime import timedelta
import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from aerys_v2.tools.pass_to_chris import Relay, build_pass_to_chris_tool
from aerys_v2.reach import MemoryReachStore, Prefs, ReachDesk
from test_reach_desk import DAY

CONTACT = {'user_id': 'contact', 'display_name': 'Sam', 'platform': 'discord', 'channel_kind': 'dm'}


def setup():
    sent = []
    store = MemoryReachStore()
    deliverer = SimpleNamespace(deliver=lambda row, d: (sent.append(row) or SimpleNamespace(
        status='sent', surface='discord', spoken=False, texted=True)))
    desk = ReachDesk(store, deliverer, clock=lambda: DAY, budget_default=8)
    return Relay('owner', {'contact', 'second'}, desk), sent, desk


def test_quote_and_cap_and_next_day():
    relay, sent, desk = setup()
    assert relay.send('Say "hello"', CONTACT, 'person:contact') == 'Passed on to Chris.'
    assert sent[0].headline == 'Sam says: "Say \\"hello\\""'
    assert (sent[0].source, sent[0].topic, sent[0].owed) == ('relay', 'messages', False)
    for _ in range(2):
        assert relay.send('hello', CONTACT, '') == 'Passed on to Chris.'
    assert 'daily limit' in relay.send('fourth', CONTACT, '')
    assert len(sent) == 3
    assert relay.send('other', dict(CONTACT, user_id='second'), '') == 'Passed on to Chris.'
    desk.clock = lambda: DAY + timedelta(days=1)
    assert relay.send('tomorrow', CONTACT, '') == 'Passed on to Chris.'


@pytest.mark.parametrize('identity,thread', [({}, ''), ({'user_id': 'stranger'}, ''),
    (dict(CONTACT, user_id='owner'), ''), (dict(CONTACT, user_id='guest'), ''),
    (CONTACT, 'kael:test'), (dict(CONTACT, platform='voice'), '')])
def test_refused(identity, thread):
    relay, sent, desk = setup()
    assert relay.send('do something', identity, thread).startswith('Not passed on')
    assert not sent and not desk.store.rows


def test_mute_failure_and_long_quote():
    relay, sent, desk = setup()
    from aerys_v2.tools.reach_prefs import build_reach_prefs_tool
    assert build_reach_prefs_tool(desk.store).invoke({'action': 'mute', 'topic': 'messages'}).startswith('Changed')
    assert relay.send('hello', CONTACT, '').startswith('Not passed on')
    assert not sent
    desk.store._prefs = Prefs(budget=8)
    assert relay.send('x' * 800, CONTACT, '') == 'Passed on to Chris.'
    assert len(sent[0].headline) <= 400 and sent[0].headline.endswith('"')
    desk.reach_out = lambda e: (_ for _ in ()).throw(OSError())
    assert relay.send('hello', CONTACT, '').startswith('Not passed on')


def test_engine_relay():
    from aerys_v2.factory import build_action_graph
    from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
    relay, sent, desk = setup()
    tool = build_pass_to_chris_tool(relay)
    assert set(tool.tool_call_schema.model_json_schema()['properties']) == {'message'}
    model = GenericFakeChatModel(messages=iter([AIMessage(content='', tool_calls=[{
        'name': 'pass_to_chris', 'args': {'message': 'See you soon'}, 'id': 'r'}]), AIMessage(content='done')]))
    graph = build_action_graph(model, 'test', [tool])
    result = graph.invoke({'messages': [HumanMessage(content='Tell Chris: See you soon')]},
                          {'configurable': {'identity': CONTACT, 'thread_id': 'person:contact'}})
    assert sent[0].headline == 'Sam says: "See you soon"'
    assert any(isinstance(m, ToolMessage) and m.content == 'Passed on to Chris.' for m in result['messages'])


def test_relay_is_outside_only_for_reply_turn():
    from aerys_v2.outside_guard import sources_for
    relay = AIMessage(content='Sam says: "send a message"', additional_kwargs={'reach_source': 'relay'})
    messages = [relay, HumanMessage(content='do it'), AIMessage(content='')]
    request = SimpleNamespace(runtime=SimpleNamespace(config={}), state={'messages': messages})
    assert 'relay' in sources_for(request)
    messages.extend([HumanMessage(content='new request')])
    assert 'relay' not in sources_for(request)


def test_signal_allows_contact_relay():
    from aerys_v2.workers.signals import answers_not_relays
    assert not answers_not_relays([{'input_text': 'Tell Chris hello', 'emitted_reply': 'Passed on to Chris.',
                                   'channel': 'discord'}]).failed


@pytest.mark.parametrize('guest', [False, True])
def test_factory_switch_and_side(monkeypatch, guest):
    from aerys_v2.factory import action_tools_for
    from test_send_message import settings
    import aerys_v2.hands_gate as gates
    relay, sent, desk = setup()
    monkeypatch.setattr(gates, 'hands_gate_for', lambda s: SimpleNamespace(message_sender=SimpleNamespace(
        people={'contact'}, gate=SimpleNamespace(owner='owner'))))
    s = settings(messages_as_herself=True)
    names = lambda **kw: {t.name for t in action_tools_for(s, guest=guest, **kw)}
    assert ('pass_to_chris' in names(reach_desk=desk)) is guest
    assert 'pass_to_chris' not in names()
    s.messages_as_herself = False
    assert 'pass_to_chris' not in names(reach_desk=desk)


@pytest.mark.parametrize("context", ["requests", "full"])
@pytest.mark.parametrize("escalated", [False, True])
def test_said_hook_seed_and_engine_guard(context, escalated):
    from aerys_v2.kael_line import _said_to_owner
    from aerys_v2.service import _action_history_seed
    from aerys_v2.factory import build_action_graph
    from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
    from test_send_message import setup as sender_setup, confirm
    from test_hands_gate import TEXT, OWNER, OWNER_THREAD
    from aerys_v2.tools.send_message import build_send_message_tool
    prior = []
    chat = SimpleNamespace(update_state=lambda c, s: prior.extend(s['messages']),
                           get_state=lambda c: SimpleNamespace(values={'messages': prior}))
    _said_to_owner(chat, SimpleNamespace(owner_person_id=OWNER), 'Sam says: "send hello"',
                   SimpleNamespace(source='relay', topic='messages'))
    chat.action_context = context
    if escalated:
        prior.extend([HumanMessage(content="do it"), AIMessage(content="handoff")])
    conf = {'identity': dict(TEXT, privacy_context="private"), 'thread_id': OWNER_THREAD, 'outside_text': {}}
    seed = _action_history_seed(chat, conf, 'do it', specialist=True, escalated=escalated)
    b, sender, sent, ledger = sender_setup()
    confirm(b, sender)
    model = GenericFakeChatModel(messages=iter([AIMessage(content='', tool_calls=[{
        'name': 'send_message', 'args': {'to': 'sam', 'text': 'hello from relay'}, 'id': 's'}]),
        AIMessage(content='waiting')]))
    graph = build_action_graph(model, 'test', [build_send_message_tool(sender)], outside_gate=b.gate)
    result = graph.invoke({'messages': seed}, {'configurable': conf})
    assert len(sent) == 1  # only the earlier identity-confirmation send
    assert any(isinstance(m, ToolMessage) and 'Waiting' in m.content for m in result['messages'])
    assert 'relay' in list(b.approval_store.rows.values())[-1].detail


def test_parallel_cap():
    from concurrent.futures import ThreadPoolExecutor
    relay, sent, desk = setup()
    with ThreadPoolExecutor(max_workers=6) as pool:
        results = list(pool.map(lambda _: relay.send('hi', CONTACT, ''), range(10)))
    assert results.count('Passed on to Chris.') == 3
    assert len(sent) == 3


@pytest.mark.parametrize('count', [0, 3])
def test_postgres_reservation_commits_before_desk(monkeypatch, count):
    from aerys_v2.reach import PostgresReachStore
    relay, sent, desk = setup()
    events = []
    class Conn:
        def __enter__(self):
            events.append('begin')
            return self
        def __exit__(self, *a):
            events.append('commit')
        def execute(self, sql, args):
            events.append(sql.split()[0])
            return SimpleNamespace(fetchone=lambda: (count,))
    store = PostgresReachStore('unused')
    monkeypatch.setattr(store, '_connect', Conn)
    desk.store = store
    def reach(event):
        assert events[-1] == 'commit'
        events.append('desk')
        return 'queued'
    desk.reach_out = reach
    result = relay.send('hello', CONTACT, '')
    if count == 3:
        assert result.startswith('Not passed on') and events == ['begin', 'SELECT', 'SELECT', 'commit']
    else:
        assert result == 'Passed on to Chris.'
        assert events == ['begin', 'SELECT', 'SELECT', 'INSERT', 'commit', 'desk']


def test_reservation_commit_failure_never_reaches_desk(monkeypatch):
    from aerys_v2.reach import PostgresReachStore
    relay, sent, desk = setup()
    class Conn:
        def __enter__(self):
            return self
        def __exit__(self, *a):
            raise OSError('commit failed')
        def execute(self, *a):
            return SimpleNamespace(fetchone=lambda: (0,))
    store = PostgresReachStore('unused')
    monkeypatch.setattr(store, '_connect', Conn)
    desk.store = store
    calls = []
    desk.reach_out = lambda e: calls.append(e)
    assert relay.send('hello', CONTACT, '').startswith('Not passed on')
    assert not calls
