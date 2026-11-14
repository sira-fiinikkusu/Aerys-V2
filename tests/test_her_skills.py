from langgraph.store.memory import InMemoryStore


def test_only_four_tools():
    from aerys_v2.skills import skills_middleware
    middleware = skills_middleware(InMemoryStore(), "owner-placeholder")
    assert {t.name for m in middleware for t in getattr(m, "tools", ())} == {
        "ls", "read_file", "write_file", "edit_file"}

import json
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

from aerys_v2.skills import (
    NAMESPACE, REFUSAL, approved_write, backend_for, main, skills_middleware,
)
from aerys_v2.action_loop import AgentToolModels, build_agent_action_graph
from aerys_v2.outside_guard import OUTSIDE_ACTION, action_executor, outside_guard
from test_action_loop_agent import Rec

OWNER = 'owner-placeholder'
IDENTITY = {'user_id': OWNER, 'platform': 'discord', 'privacy_context': 'private'}
CONTENT = '---\nname: sample\ndescription: A sample workflow\n---\nBODY_ONLY_MARKER\n'
PATH = '/skills/sample/SKILL.md'


def call(name, **args):
    return AIMessage(content='', tool_calls=[{'name': name, 'args': args, 'id': 'call'}])


def native(script, store, *, tools=(), middleware=None, gate=None):
    log = []
    model = Rec(label='native', script=script, log=log)
    graph = build_agent_action_graph(
        AgentToolModels(model, (model, True)), 'SOUL', list(tools),
        extra_middleware=skills_middleware(store, OWNER) if middleware is None else middleware,
        checkpointer=InMemorySaver(), outside_gate=gate,
    )
    return graph, log


def config(identity=None, thread='person:owner-placeholder'):
    return {'configurable': {'thread_id': thread, 'identity': IDENTITY if identity is None else identity}}


def test_progressive_prompt_and_read():
    store = InMemoryStore()
    backend_for(store).write(PATH, CONTENT)
    graph, log = native([call('read_file', file_path=PATH), AIMessage(content='Done')], store)
    graph.invoke({'messages': [HumanMessage(content='Use the sample skill')]}, config())
    first = str(log[0]['messages'])
    assert 'sample' in first and 'A sample workflow' in first
    assert 'BODY_ONLY_MARKER' not in first
    assert 'BODY_ONLY_MARKER' in str(log[1]['messages'])
    assert set(log[0]['tools']) == {'ls', 'read_file', 'write_file', 'edit_file'}


def test_reload_same_thread_next_turn():
    store = InMemoryStore()
    graph, log = native([call('write_file', file_path=PATH, content=CONTENT),
                         AIMessage(content='Saved'), AIMessage(content='Ready')], store)
    cfg = config()
    graph.invoke({'messages': [HumanMessage(content='Save a skill')]}, cfg)
    assert 'A sample workflow' not in str(log[0]['messages'])
    graph.invoke({'messages': [HumanMessage(content='What skills do you have?')]}, cfg)
    systems = [content for kind, content in log[-1]['messages'] if kind == 'system']
    assert 'A sample workflow' in str(systems)
    assert 'BODY_ONLY_MARKER' not in str(systems)


@tool
def large_result() -> str:
    """Return a large fixture result."""
    return 'x' * 100_000


def test_no_eviction_of_large_results_or_human_messages():
    store = InMemoryStore()
    graph, _ = native([call('large_result'), AIMessage(content='Done')], store, tools=[large_result])
    graph.invoke({'messages': [HumanMessage(content='y' * 100_000)]}, config())
    assert store.search(NAMESPACE) == []


@pytest.mark.parametrize('name,args', [
    ('write_file', {'content': CONTENT}),
    ('edit_file', {'old_string': 'old', 'new_string': 'new'}),
])
@pytest.mark.parametrize('path', ['/outside/SKILL.md', '/skills-other/SKILL.md', '/skills/../outside/SKILL.md'])
def test_path_fence(name, args, path):
    store = InMemoryStore()
    # Existing file ensures edit would succeed if the permission check vanished.
    backend_for(store).write('/outside/SKILL.md', 'old')
    backend_for(store).write('/skills-other/SKILL.md', 'old')
    before = [(i.key, i.value) for i in store.search(NAMESPACE)]
    result = approved_write(skills_middleware(store, OWNER), SimpleNamespace(
        args={'tool': name, 'tool_args': {'file_path': path, **args}, 'identity': IDENTITY},
        thread_id='person:owner-placeholder'))
    assert result.status == 'error'
    assert [(i.key, i.value) for i in store.search(NAMESPACE)] == before


@pytest.mark.parametrize('identity,thread,allowed', [
    (IDENTITY, 'person:owner-placeholder', True),
    ({**IDENTITY, 'voice': True, 'speaker': {'id': 'speaker-placeholder'}}, 'voice:placeholder', True),
    ({**IDENTITY, 'platform': 'ask', 'surface': 'lens'}, 'glasses:placeholder', True),
    ({**IDENTITY, 'platform': 'wake'}, 'wake:placeholder', True),
    ({**IDENTITY, 'channel_kind': 'guild'}, 'discord:guild:placeholder', False),
    ({**IDENTITY, 'channel_kind': 'group'}, 'telegram:group:placeholder', False),
    ({**IDENTITY, 'sealed_room': True}, 'room:placeholder', False),
    ({**IDENTITY, 'voice': True, 'speaker': {'id': 'speaker-placeholder', 'unsure': True}}, 'voice:placeholder', False),
    ({**IDENTITY, 'user_id': 'other-placeholder'}, 'person:other-placeholder', False),
    (IDENTITY, 'kael:placeholder', False),
])
@pytest.mark.parametrize('name', ['write_file', 'edit_file'])
def test_owner_gate(identity, thread, allowed, name):
    store = InMemoryStore()
    backend_for(store).write(PATH, 'old')
    args = {'file_path': PATH, **({'content': CONTENT} if name == 'write_file' else
                                {'old_string': 'old', 'new_string': CONTENT})}
    graph, log = native([call(name, **args), AIMessage(content='Done')], store)
    result = graph.invoke({'messages': [HumanMessage(content='Save')]}, config(identity, thread))
    receipts = [m.content for m in result['messages'] if isinstance(m, ToolMessage)]
    assert (REFUSAL not in receipts) == allowed
    saved = backend_for(store).download_files([PATH])[0].content.decode()
    assert saved == (CONTENT if allowed else 'old')


class Gate:
    def __init__(self):
        self.requests = []

    def _ask(self, *args, **kwargs):
        self.requests.append((args, kwargs))
        return SimpleNamespace(go=False, text='Waiting for approval', infra=False)


@pytest.mark.parametrize('reader', ['read_web_page', 'check_async_task'])
@pytest.mark.parametrize('name', ['write_file', 'edit_file'])
def test_d28_holds_writes(reader, name):
    gate = Gate()
    request = SimpleNamespace(
        runtime=SimpleNamespace(config=config()),
        state={'messages': [HumanMessage(content='Save'), ToolMessage(content='outside', name=reader, tool_call_id='read')]},
        tool_call={'name': name, 'id': 'write', 'args': {'file_path': PATH, 'content': CONTENT}},
    )
    result = outside_guard(request, lambda _: pytest.fail('write escaped approval'), gate=gate)
    assert result.content == 'Waiting for approval'
    assert gate.requests[0][0][0] == OUTSIDE_ACTION


def test_approved_executor_writes_edits_and_keeps_fences():
    store = InMemoryStore()
    executor = action_executor([], skill_middleware=skills_middleware(store, OWNER))
    row = SimpleNamespace(thread_id='person:owner-placeholder', args={
        'identity': IDENTITY, 'tool': 'write_file', 'tool_args': {'file_path': PATH, 'content': CONTENT}})
    assert executor(row)[0]
    row.args.update(tool='edit_file', tool_args={'file_path': PATH, 'old_string': 'BODY_ONLY_MARKER', 'new_string': 'REVISED'})
    assert executor(row)[0]
    assert b'REVISED' in backend_for(store).download_files([PATH])[0].content
    row.args.update(identity={**IDENTITY, 'sealed_room': True})
    assert executor(row) == (False, REFUSAL)
    row.args.update(identity=IDENTITY, tool='write_file', tool_args={'file_path': '/elsewhere', 'content': CONTENT})
    assert executor(row)[0] is False
    assert store.get(NAMESPACE, '/elsewhere') is None


def test_admin_roundtrip(tmp_path, capsys):
    store = InMemoryStore()
    source = tmp_path / 'sample.md'
    source.write_text(CONTENT)
    main(['put', 'sample', str(source)], store=store)
    capsys.readouterr()
    main(['get', 'sample'], store=store)
    assert capsys.readouterr().out == CONTENT
    main(['list'], store=store)
    row = json.loads(capsys.readouterr().out)
    assert row['slug'] == row['name'] == 'sample'
    assert row['description'] == 'A sample workflow'
    assert row['created_at'] and row['updated_at']
    source.write_text(CONTENT.replace('BODY_ONLY_MARKER', 'REVISED'))
    main(['put', 'sample', str(source)], store=store)
    assert b'REVISED' in backend_for(store).download_files([PATH])[0].content


@pytest.mark.parametrize('description', ['Account 123456789012345678', 'Use sensor.placeholder',
                                        'Use owner@example.invalid', 'Line one\nline two'])
def test_admin_rejects_id_descriptions(tmp_path, description):
    import yaml
    source = tmp_path / 'sample.md'
    source.write_text('---\n' + yaml.safe_dump({'name': 'sample', 'description': description}) + '---\nbody')
    store = InMemoryStore()
    with pytest.raises(ValueError):
        main(['put', 'sample', str(source)], store=store)
    assert store.search(NAMESPACE) == []


def test_factory_owner_guest_and_off_prompts(monkeypatch):
    import aerys_v2.factory as factory
    import aerys_v2.skills as skills
    import aerys_v2.hands_gate as hands
    from aerys_v2.config import Settings

    store = InMemoryStore()
    backend_for(store).write(PATH, CONTENT)
    store_calls, logs = [], []
    gate = Gate()
    gate.executors = {}
    monkeypatch.setattr(skills, 'store_for', lambda s: store_calls.append(s) or store)
    monkeypatch.setattr(hands, 'hands_gate_for', lambda s: gate)
    monkeypatch.setattr(factory, 'action_tools_for', lambda *a, **kw: [large_result])
    monkeypatch.setattr(factory, '_where_when_line', lambda *a: '\nA fixed time.')
    monkeypatch.setattr(factory, 'action_overlay_for', lambda *a, **kw: 'OVERLAY')
    import aerys_v2.kael_line as kael_line
    monkeypatch.setattr(kael_line, 'family_notes_fn_for', lambda s: None)

    def engine(settings, tools):
        log = []
        model = Rec(label='native', script=[AIMessage(content='Ready')], log=log)
        logs.append(log)
        return AgentToolModels(model, (model, True)), factory.build_action_graph

    monkeypatch.setattr(factory, '_action_engine', engine)
    def settings(**kwargs):
        return Settings(_env_file=None, anthropic_api_key='placeholder', owner_person_id=OWNER, **kwargs)

    baseline = None
    for armed, database in [(False, None), (False, 'postgresql://placeholder'), (True, None)]:
        _, graph = factory.action_stack_for(settings(her_skills=armed, database_url=database), 'SOUL')
        graph.invoke({'messages': [HumanMessage(content='Hello')]}, config())
        assert logs[-1][0]['tools'] == ('large_result',)
        messages = logs[-1][0]['messages']
        baseline = messages if baseline is None else baseline
        assert messages == baseline
    assert store_calls == []
    armed = settings(her_skills=True, database_url='postgresql://placeholder')
    _, graph = factory.action_stack_for(armed, 'SOUL')
    graph.invoke({'messages': [HumanMessage(content='Hello')]}, config())
    assert len(store_calls) == 1
    assert 'A sample workflow' in str(logs[-1][0]['messages'])
    assert set(logs[-1][0]['tools']) == {'large_result', 'ls', 'read_file', 'write_file', 'edit_file'}
    assert OUTSIDE_ACTION in gate.executors
    guest = factory.guest_action_graph_for(armed, 'SOUL')
    guest.invoke({'messages': [HumanMessage(content='Hello')]}, config())
    assert logs[-1][0]['tools'] == ('large_result',)
    assert 'My skills' not in str(logs[-1][0]['messages'])
    assert 'A sample workflow' not in str(logs[-1][0]['messages'])
    assert len(store_calls) == 1


def test_store_lifecycle_and_explicit_setup(monkeypatch):
    from unittest.mock import MagicMock
    import aerys_v2.skills as skills
    import psycopg_pool
    import langgraph.store.postgres
    from psycopg.rows import dict_row

    pool_cls = MagicMock()
    store_cls = MagicMock()
    monkeypatch.setattr(psycopg_pool, 'ConnectionPool', pool_cls)
    monkeypatch.setattr(langgraph.store.postgres, 'PostgresStore', store_cls)
    monkeypatch.setattr(skills.atexit, 'register', MagicMock())
    settings = SimpleNamespace(database_url='postgresql://lifecycle-placeholder', checkpoint_pool_open_timeout_s=3)
    skills._postgres_store.cache_clear()
    try:
        assert skills.store_for(settings) is skills.store_for(settings)
        pool_cls.assert_called_once_with(
            conninfo=settings.database_url, min_size=1, max_size=1,
            check=pool_cls.check_connection,
            kwargs={'autocommit': True, 'prepare_threshold': 0, 'row_factory': dict_row}, open=False)
        pool_cls.return_value.open.assert_called_once_with(wait=True, timeout=3)
        store_cls.return_value.setup.assert_not_called()
        main(['setup'], store=store_cls.return_value)
        store_cls.return_value.setup.assert_called_once_with()
    finally:
        skills._postgres_store.cache_clear()


def test_async_reload_and_owner_gate(monkeypatch):
    from unittest.mock import AsyncMock
    from deepagents.middleware.skills import SkillsMiddleware
    from aerys_v2.skills import OwnerSkillWrites

    # Drive immediate coroutines without opening an event loop/socketpair in this sandbox.
    def finish(coroutine):
        with pytest.raises(StopIteration) as done:
            coroutine.send(None)
        return done.value.value

    parent = AsyncMock(return_value={'skills_metadata': []})
    monkeypatch.setattr(SkillsMiddleware, 'abefore_agent', parent)
    layer = skills_middleware(InMemoryStore(), OWNER)[1]
    finish(layer.abefore_agent({'skills_metadata': [{'name': 'stale'}]}, None, {}))
    assert parent.call_args.args[0]['skills_metadata'] is None
    request = SimpleNamespace(runtime=SimpleNamespace(config=config({**IDENTITY, 'sealed_room': True})),
                              tool_call={'name': 'write_file', 'id': 'write'})
    handler = AsyncMock()
    assert finish(OwnerSkillWrites(OWNER).awrap_tool_call(request, handler)).content == REFUSAL
    handler.assert_not_called()


@tool
def read_web_page() -> str:
    """Read outside fixture text."""
    return 'outside fixture'


def test_native_d28_write_waits_then_approval_executes():
    store = InMemoryStore()
    gate = Gate()
    layers = skills_middleware(store, OWNER)
    graph, _ = native([call('read_web_page'), call('write_file', file_path=PATH, content=CONTENT),
                       AIMessage(content='Waiting')], store, tools=[read_web_page], middleware=layers, gate=gate)
    graph.invoke({'messages': [HumanMessage(content='Read then save')]}, config())
    assert store.search(NAMESPACE) == []
    args, _ = gate.requests[0]
    assert args[0] == OUTSIDE_ACTION
    row = SimpleNamespace(thread_id=args[2], args=args[3])
    assert action_executor([], skill_middleware=layers)(row)[0]
    assert backend_for(store).download_files([PATH])[0].content.decode() == CONTENT
