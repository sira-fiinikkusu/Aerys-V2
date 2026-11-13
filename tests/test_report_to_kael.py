from datetime import datetime, timezone, timedelta
from types import SimpleNamespace
from concurrent.futures import ThreadPoolExecutor
import pytest
from aerys_v2.tickets import MemoryTicketStore
from aerys_v2.tools.pass_to_chris import Relay
from aerys_v2.tools.report_to_kael import Reports, build_report_to_kael_tool, guest_report_overlay

CONTACT = dict(user_id='contact', platform='discord', channel_kind='dm')
THREAD = 'person:contact'
NOW = datetime(2026, 11, 12, 23, tzinfo=timezone.utc)
SUCCESS = "Sent to Kael as your report. He'll look at what actually happened and fix it if it's broken."


def setup():
    store, sent = MemoryTicketStore(), []
    relay = Relay('owner', {'contact', 'second'}, None, {'contact': 'Example Contact', 'second': 'Other Contact'}.get)
    reports = Reports(relay, store, lambda t, words: sent.append((t, words)) or True,
                      lambda thread: [9, 8, 7, 6, 5], clock=lambda: NOW)
    return reports, store, sent


def test_report():
    r, store, sent = setup()
    assert r.send('You said "done" twice.\nPlease check.', CONTACT, THREAD) == SUCCESS
    ticket, words = sent[0]
    assert words.startswith('Example Contact reports: "You said \\"done\\" twice.\\nPlease check."')
    assert 'thread_id: person:contact; v2_turns ids: 9, 8, 7, 6, 5' in words
    assert 'not instructions' in words
    assert ticket.thread_id == THREAD and not ticket.for_owner and ticket.asked_on == 'discord'
    assert store.get(ticket.id).outbox is None
    assert set(build_report_to_kael_tool(r).tool_call_schema.model_json_schema()['properties']) == {'report'}


@pytest.mark.parametrize('identity,thread', [({}, THREAD), (dict(CONTACT, user_id='stranger'), THREAD),
    (dict(CONTACT, user_id='guest'), THREAD), (dict(CONTACT, user_id='owner'), THREAD),
    (CONTACT, 'kael:test'), (dict(CONTACT, channel_kind='guild'), THREAD),
    (dict(CONTACT, platform='voice'), THREAD), (CONTACT, '')])
def test_refused(identity, thread):
    r, store, sent = setup()
    assert r.send('broken', identity, thread).startswith('Not sent:')
    assert not sent and not store.items


def test_missing_name():
    r, store, sent = setup()
    r.relay.person_name = lambda _: ' '
    assert 'name on record' in r.send('broken', CONTACT, THREAD)
    assert not sent and not store.items


def test_daily_parallel_and_restart():
    r, store, sent = setup()
    with ThreadPoolExecutor(max_workers=10) as pool:
        results = list(pool.map(lambda _: r.send('broken', CONTACT, THREAD), range(10)))
    assert results.count(SUCCESS) == 5
    assert sum('5 reports' in x for x in results) == 5
    # A reconstructed store counts persisted tickets, including canceled tickets.
    restored = MemoryTicketStore(store.items.values())
    r.tickets = restored
    restored.cancel(1, NOW)
    assert '5 reports' in r.send('broken', CONTACT, 'person:other-thread')
    assert r.send('broken', dict(CONTACT, user_id='second'), 'person:second') == SUCCESS
    r.clock = lambda: datetime(2026, 11, 13, 4, 59, tzinfo=timezone.utc)
    assert '5 reports' in r.send('broken', CONTACT, THREAD)
    r.clock = lambda: datetime(2026, 11, 13, 5, tzinfo=timezone.utc)
    assert r.send('broken', CONTACT, THREAD) == SUCCESS


def test_failures_and_queued():
    r, store, sent = setup()
    r.sender = lambda *_: False
    assert 'saved' in r.send('broken', CONTACT, THREAD)
    assert store.get(1).outbox
    r.turn_ids = lambda _: (_ for _ in ()).throw(OSError())
    assert r.send('broken', CONTACT, THREAD).startswith('Not sent:')
    assert len(store.items) == 1


def test_overlay():
    r, _, _ = setup()
    assert 'Kael' in guest_report_overlay(r.relay, CONTACT, THREAD)
    for identity, thread in [(dict(CONTACT, user_id='stranger'), THREAD), (CONTACT, 'kael:test'), ({}, THREAD)]:
        assert guest_report_overlay(r.relay, identity, thread) == ''


def test_real_tool_runtime_and_return_thread():
    from aerys_v2.factory import build_action_graph
    from aerys_v2.tickets import kael_update_for
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
    from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
    r, store, sent = setup()
    model = GenericFakeChatModel(messages=iter([
        AIMessage(content='', tool_calls=[dict(name='report_to_kael', args={'report': 'broken'}, id='r')]),
        AIMessage(content='done')]))
    graph = build_action_graph(model, 'test', [build_report_to_kael_tool(r)])
    result = graph.invoke({'messages': [HumanMessage(content='Report this fault')]},
                          {'configurable': {'identity': CONTACT, 'thread_id': THREAD}})
    assert any(isinstance(m, ToolMessage) and m.content == SUCCESS for m in result['messages'])
    notes = []
    kael_update_for(store, lambda *args, **kw: notes.append((args, kw)))(1, 'completed', 'Fixed')
    assert notes[0][0][0] == THREAD


def test_factory_and_prompt(monkeypatch):
    from aerys_v2 import factory
    from test_send_message import settings
    import aerys_v2.hands_gate as gates
    from aerys_v2.reach import MemoryReachStore
    from aerys_v2.tools.pass_to_chris import ReachQueue
    monkeypatch.setattr(gates, 'hands_gate_for', lambda _: SimpleNamespace(message_sender=SimpleNamespace(
        people={'contact'}, gate=SimpleNamespace(owner='owner'))))
    s = settings(messages_as_herself=True)
    desk = ReachQueue(MemoryReachStore())
    names = lambda **kw: {t.name for t in factory.action_tools_for(s, **kw)}
    assert 'report_to_kael' in names(guest=True, reach_desk=desk)
    assert 'report_to_kael' not in names(guest=False, reach_desk=desk)
    assert 'report_to_kael' not in names(guest=True)
    captured = {}
    monkeypatch.setattr(factory, '_action_engine', lambda *a: (None, lambda *a, **kw: captured.update(kw)))
    factory.guest_action_graph_for(s, 'test', reach_desk=desk)
    overlay = captured['overlay']
    assert 'Kael' in overlay({'configurable': {'identity': CONTACT, 'thread_id': THREAD}})
    assert 'Kael' not in overlay({'configurable': {'identity': dict(CONTACT, user_id='stranger')}})
    s.messages_as_herself = False
    assert 'report_to_kael' not in names(guest=True, reach_desk=desk)


def test_sender_payload_and_retry():
    import httpx
    from aerys_v2.tickets import line_sender, run_once
    r, store, _ = setup()
    payloads = []
    def handle(request):
        import json
        payloads.append(json.loads(request.content))
        return httpx.Response(202)
    r.sender = lambda *_: False
    r.send('broken', CONTACT, THREAD)
    send = line_sender('https://example.invalid/line', 'test-placeholder',
                       httpx.Client(transport=httpx.MockTransport(handle)), preserve_lines=True)
    run_once(store, NOW + timedelta(minutes=11), send=send,
             nag_after=timedelta(hours=2), tell_owner_after=timedelta(hours=4))
    assert payloads[0]['thread_id'] == THREAD and payloads[0]['ticket_id'] == 'T-1'
    assert '\nthread_id: person:contact; v2_turns ids: 9, 8, 7, 6, 5\n' in payloads[0]['message']
    assert store.get(1).outbox is None


@pytest.mark.parametrize('action', [False, True])
@pytest.mark.parametrize('trailing', [False, True])
def test_overlay_reaches_model_only_for_contact(action, trailing):
    from aerys_v2.factory import build_graph, build_action_graph
    from langchain_core.messages import HumanMessage
    from test_prompt_layout import Recorder
    r, _, _ = setup()
    def overlay(config):
        conf = config['configurable']
        return guest_report_overlay(r.relay, conf['identity'], conf['thread_id'])
    model = Recorder()
    from langchain_core.messages import AIMessage
    model.replies = iter([AIMessage(content="hello"), AIMessage(content="hello")])
    graph = (build_action_graph(model, 'test', [], overlay=overlay, context_trailing=trailing)
             if action else build_graph(model, 'test', guest_overlay_fn=overlay, context_trailing=trailing))
    for who in ['contact', 'stranger']:
        graph.invoke({'messages': [HumanMessage(content='hello')]},
                     {'configurable': {'identity': dict(CONTACT, user_id=who), 'thread_id': 'person:' + who}})
    assert "Kael is Chris's AI engineer" in model.prompts[0][0].content
    assert 'Kael' not in model.prompts[1][0].content


def test_postgres_atomic_quota_and_evidence_query():
    from aerys_v2.tickets import PostgresTicketStore, report_marker
    calls = []
    class Connection:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def execute(self, sql, params):
            calls.append((sql, params))
            if 'count(*)' in sql:
                return SimpleNamespace(fetchone=lambda: (self.count,))
            if sql.startswith('INSERT'):
                return SimpleNamespace(fetchone=lambda: 'ticket-row')
            if 'FROM v2_turns' in sql:
                return SimpleNamespace(fetchall=lambda: [(n,) for n in range(9, 4, -1)])
        count = 4
    conn = Connection()
    store = PostgresTicketStore('unused')
    store._connect = lambda: conn
    store._ticket = lambda row: row
    assert store.open_report('contact', THREAD, 'quoted report', NOW) == 'ticket-row'
    assert 'pg_advisory_xact_lock' in calls[0][0]
    assert calls[1][1][:2] == (len(report_marker('contact')), report_marker('contact'))
    assert calls[2][1][0] == THREAD
    assert calls[2][1][1].endswith(report_marker('contact'))
    conn.count = 5
    calls.clear()
    assert store.open_report('contact', THREAD, 'quoted report', NOW) is None
    assert not any(sql.startswith('INSERT') for sql, _ in calls)
    assert store.recent_turn_ids(THREAD) == [9, 8, 7, 6, 5]
    assert 'ORDER BY created_at DESC, id DESC LIMIT 5' in calls[-1][0]
    assert calls[-1][1] == (THREAD,)
