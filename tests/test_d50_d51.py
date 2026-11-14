"""Owner step budget and voice recipient clarification regressions."""
from types import SimpleNamespace
import pytest
from langchain_core.messages import AIMessage
from langgraph.errors import GraphRecursionError
from aerys_v2 import factory
from aerys_v2.service import Rails, ask, _honest_reply_for_failure
from aerys_v2.router import RouteDecision
from test_send_message import setup, call
from test_hands_gate import TEXT


class Graph:
    def __init__(self):
        self.configs = []

    def get_state(self, config):
        return SimpleNamespace(values={"messages": []})

    def update_state(self, config, value, **kwargs):
        pass

    def invoke(self, value, config):
        self.configs.append(config)
        return {'messages': [AIMessage(content='All right.')]}


@pytest.mark.parametrize('voice', [False, True])
@pytest.mark.parametrize('who,limit', [('owner', 60), ('guest', 10), ('allowed-guest', 10)])
def test_identity_step_budget(monkeypatch, voice, who, limit):
    monkeypatch.setattr(factory, '_OWNER_ID', 'owner')
    chat, action, guest = Graph(), Graph(), Graph()
    ask(chat, 'hello', identity={'user_id': who, 'voice': voice}, thread_id='person:test',
        router=lambda *a, **kw: RouteDecision(route='chat', ack=''), action_graph=action,
        guest_action_graph=guest, action_allowlist=frozenset({'owner', 'allowed-guest'}))
    configs = chat.configs + action.configs + guest.configs
    assert configs
    assert all(c['recursion_limit'] == limit for c in configs)


@pytest.mark.parametrize('owner', [None, 'someone-else'])
def test_unrecognized_owner_stays_at_ten(monkeypatch, owner):
    monkeypatch.setattr(factory, '_OWNER_ID', owner)
    graph = Graph()
    ask(graph, 'hello', identity={'user_id': 'caller'}, thread_id='text')
    assert graph.configs[0]['recursion_limit'] == 10


def test_explicit_rails_preserved_for_owner_and_capped_for_guest(monkeypatch):
    monkeypatch.setattr(factory, '_OWNER_ID', 'owner')
    for who, expected in [('owner', 42), ('guest', 10)]:
        graph = Graph()
        ask(graph, 'hello', identity={'user_id': who}, thread_id='text', rails=Rails(turn_limit=42))
        assert graph.configs[0]['recursion_limit'] == expected
    assert Rails().wall_clock_s == 90.0


def test_recursion_reply_is_general_and_makes_no_delivery_claim():
    line = _honest_reply_for_failure(GraphRecursionError('limit'))
    assert 'ran out of steps before finishing' in line
    assert 'Ask me again' in line and 'narrow' in line
    assert 'device' not in line and 'nothing' not in line and 'sent' not in line


@pytest.mark.parametrize('voice', [False, True])
def test_unknown_name_asks_without_sending_or_approval(voice):
    b, sender, sent, ledger = setup()
    b.gate.people = lambda *a: []
    result = call(sender, to='Unknown Person', identity={**TEXT, 'voice': voice})
    if voice:
        assert result == "NOT DONE: I heard a name that doesn't match anyone I know; ask him who he means."
    else:
        assert result == "Not sent: I don't know who Discord Unknown Person is. Ask Chris who it is before sending anything."
    assert not sent and not ledger.rows and not b.approval_store.rows


def test_legacy_voice_unknown_name():
    b, sender, sent, ledger = setup()
    b.gate.people = lambda *a: []
    assert sender.send('Unknown Person', 'hello', 'discord', TEXT, 'voice:test', ()).startswith('NOT DONE: I heard a name')
    assert not sent and not ledger.rows and not b.approval_store.rows


@pytest.mark.parametrize('sources', [(), ('search_web',)])
def test_voice_known_name_preserves_confirmation_and_outside_gate(sources):
    b, sender, sent, ledger = setup()
    assert 'Waiting' in call(sender, identity={**TEXT, 'voice': True})
    assert not sent
    b.tap(); b.watcher.tick()
    result = call(sender, 'next', identity={**TEXT, 'voice': True}, sources=sources)
    assert ('Waiting' if sources else 'Sent:') in result
    assert len(sent) == (1 if sources else 2)


@pytest.mark.parametrize('overlay', ['VOICE_ACK_OVERLAY', 'VOICE_BANTER_OVERLAY'])
def test_voice_overlay_unknown_recipient_exception(overlay):
    text = getattr(factory, overlay)
    assert 'name matches no one' in text
    assert 'ask who he meant' in text
    assert 'Never substitute another person' in text
    if overlay == 'VOICE_ACK_OVERLAY':
        assert 'except for an unknown recipient' in text
