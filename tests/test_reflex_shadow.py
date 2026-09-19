"""Opposite Jev answers must remain evidence, never instructions to the service."""
import contextvars
import json
import threading
import time
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage

from aerys_v2.router import RouteDecision
from aerys_v2.service import ask


class Graph:
    def __init__(self, reply='a calm reply', error=None):
        self.reply, self.error = reply, error
        self.configs = []

    def invoke(self, inp, config):
        self.configs.append(config)
        if self.error:
            raise self.error
        return {'messages': [AIMessage(content=self.reply)]}

    def get_state(self, config):
        return SimpleNamespace(values={'messages': []})

    def update_state(self, *args, **kwargs):
        pass


class Recorder:
    def __init__(self):
        self.rows = []
        self.done = threading.Event()

    def __call__(self, row):
        self.rows.append(row)
        self.done.set()

    def row(self):
        assert self.done.wait(2), 'audit row never arrived'
        return self.rows[0]


@pytest.mark.parametrize('voice,route', [(False, 'chat'), (False, 'action'), (True, 'chat'), (True, 'action')])
def test_shadow_preserves_router_and_records_both(voice, route):
    rec, graph, action = Recorder(), Graph('chat answer'), Graph('action answer')
    began = threading.Event()
    marker = contextvars.ContextVar('reflex_test')
    marker.set('traced')
    seen = []

    def reflex(text, context):
        seen.append((text, context, marker.get()))
        began.set()
        return {'route': 'action' if route == 'chat' else 'chat', 'tier': 'fast', 'unaddressed': 1.0}

    def router(text):
        assert began.wait(.5), 'reflex must start before the router'
        return RouteDecision(route=route, ack='router ack', tier='deep', unaddressed=False)

    result = ask(graph, 'hello', identity={'voice': voice, 'platform': 'discord', 'channel_kind': 'dm'},
                 thread_id='person:test', router=router, action_graph=action, reflex=reflex,
                 record_turn=rec)
    assert result == ('router ack' if voice and route == 'action' else
                      'action answer' if voice or route == 'action' else 'chat answer')
    shadow = json.loads(rec.row()['reflex'])
    assert shadow['mode'] == 'shadow'
    assert shadow['jev']['route'] != route
    assert shadow['router'] == {'route': route, 'tier': 'deep', 'unaddressed': False}
    assert seen == [('hello', {'surface': 'voice' if voice else 'discord_dm'}, 'traced')]
    if not voice and route == 'chat':
        assert graph.configs[0]['configurable']['tier'] == 'deep'


def test_slow_shadow_does_not_hold_fast_reply():
    rec = Recorder()
    release = threading.Event()

    def reflex(text, context):
        release.wait(1)
        return {'route': 'action'}

    reflex.timeout_s = .25

    def router(text):
        time.sleep(.03)
        return RouteDecision(route='chat', ack='')

    try:
        start = time.monotonic()
        assert ask(Graph(), 'hello', identity={}, thread_id='cli', router=router,
                   action_graph=Graph(), reflex=reflex, record_turn=rec) == 'a calm reply'
        assert time.monotonic() - start < .20
        assert json.loads(rec.row()['reflex'])['jev']['error'] == 'timeout'
    finally:
        release.set()


def test_disabled_reflex_is_null():
    rec = Recorder()
    ask(Graph(), 'hello', identity={}, thread_id='cli', record_turn=rec, reflex=None)
    assert rec.row()['reflex'] is None


def test_chat_only_is_observed_without_inventing_router_verdict():
    rec = Recorder()
    ask(Graph(), 'hello', identity={}, thread_id='cli', record_turn=rec,
        reflex=lambda text, context: {'route': 'action'})
    assert json.loads(rec.row()['reflex'])['router'] == {'route': None, 'tier': None, 'unaddressed': None}


def test_graph_failure_still_records_shadow():
    rec = Recorder()
    with pytest.raises(ValueError, match='broken'):
        ask(Graph(error=ValueError('broken')), 'hello', identity={}, thread_id='cli',
            router=lambda _: RouteDecision(route='chat', ack=''), action_graph=Graph(),
            record_turn=rec, reflex=lambda text, context: {'route': 'action'})
    assert json.loads(rec.row()['reflex'])['jev']['route'] == 'action'


@pytest.mark.parametrize('voice', [False, True])
def test_router_unaddressed_drop_remains_authoritative(voice):
    rec, graph, action = Recorder(), Graph(), Graph()
    assert ask(graph, 'overheard', identity={'voice': voice, 'surface': 'lens'},
               thread_id='cli', router=lambda _: RouteDecision(route='chat', ack='', unaddressed=True),
               action_graph=action, reflex=lambda *_: {'route': 'action', 'unaddressed': 0.0},
               record_turn=rec, drop_unaddressed=True, activity_registry={}) == ''
    assert not graph.configs and not action.configs
    shadow = json.loads(rec.row()['reflex'])
    assert shadow['router']['unaddressed'] is True
    assert shadow['jev']['unaddressed'] == 0.0


def test_router_deep_vote_survives_served_tier_downgrade():
    rec = Recorder()
    ask(Graph(), 'hello', identity={}, thread_id='cli',
        router=lambda _: RouteDecision(route='chat', ack='', tier='deep'),
        action_graph=Graph(), deep_allowed=lambda: False,
        record_turn=rec, reflex=lambda *_: {'route': 'action', 'tier': 'fast'})
    row = rec.row()
    assert row['tier'] == 'standard'
    assert json.loads(row['reflex'])['router']['tier'] == 'deep'


def test_late_answer_is_timeout_even_after_slow_router_finishes():
    rec = Recorder()

    def reflex(*_):
        time.sleep(.06)
        return {'route': 'action'}

    reflex.timeout_s = .03

    def router(_):
        time.sleep(.12)
        return RouteDecision(route='chat', ack='')

    ask(Graph(), 'hello', identity={}, thread_id='cli', router=router,
        action_graph=Graph(), record_turn=rec, reflex=reflex)
    assert json.loads(rec.row()['reflex'])['jev'] == {'error': 'timeout'}


def test_reflex_exception_never_breaks_reply():
    rec = Recorder()

    def broken(*_):
        raise ValueError('bad observer')

    assert ask(Graph(), 'hello', identity={}, thread_id='cli', reflex=broken,
               record_turn=rec) == 'a calm reply'
    assert json.loads(rec.row()['reflex'])['jev']['error'] == 'ValueError: bad observer'


def test_one_shot_cli_arms_shadow_and_waits_for_receipt(monkeypatch, capsys):
    import sys
    from contextlib import nullcontext
    import aerys_v2.cli as cli
    import aerys_v2.factory as factory
    from aerys_v2.config import Settings

    rec = Recorder()
    reflex = lambda *_: {'route': 'action'}
    settings = Settings(_env_file=None, anthropic_api_key='test')
    monkeypatch.setattr(cli, 'Settings', lambda: settings)
    monkeypatch.setattr(cli, 'reflex_for', lambda s: reflex)
    monkeypatch.setattr(factory, 'turn_recorder_for', lambda s: rec)
    monkeypatch.setattr(factory, 'checkpointer_for', lambda s: nullcontext(None))
    monkeypatch.setattr(factory, 'load_soul', lambda p: 'soul')
    monkeypatch.setattr(factory, 'build_model', lambda s: None)
    monkeypatch.setattr(factory, 'build_graph', lambda *a, **k: Graph())
    monkeypatch.setattr(sys, 'argv', ['aerys-v2', '--ask', 'hello'])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 0
    assert capsys.readouterr().out.strip() == 'a calm reply'
    assert rec.done.is_set()
    assert json.loads(rec.rows[0]['reflex'])['jev']['route'] == 'action'
