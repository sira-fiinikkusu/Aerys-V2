from pathlib import Path

from langchain_core.messages import AIMessage

from aerys_v2.factory import build_graph
from aerys_v2.router import build_router, fallback_decision, parse_route_reply, RouteDecision
from aerys_v2.service import ask, action_honesty_gate, GATE_RETRY
from test_action_orchestration import fake_model, StubActionGraph


class Capture:
    def invoke(self, messages):
        self.prompt = messages[0].content
        return AIMessage(content='{"route":"lookup"}')


def test_default_prompt_snapshot():
    model = Capture()
    build_router(model, 'Test soul.')('hello')
    expected = Path(__file__).with_name('fixtures').joinpath('router_default_prompt.txt').read_text()
    assert model.prompt == expected
    build_router(model, 'Test soul.', extra_routes={})('hello')
    assert model.prompt == expected


def test_registered_route_and_degraded_predicate():
    model = Capture()
    route = build_router(model, 's', extra_routes={'lookup': 'Retrieve a catalog entry.'})
    assert route('catalog entry').route == 'lookup'
    assert '- "lookup": Retrieve a catalog entry.' in model.prompt
    assert '"chat" or "action" or "lookup"' in model.prompt
    assert fallback_decision('catalog', extra_predicates={'lookup': lambda t: t == 'catalog'}).route == 'lookup'


def test_unknown_route_uses_action_biased_fallback():
    assert parse_route_reply('{"route":"unknown"}', 'toggle the lamp').route == 'action'


def test_named_specialist_dispatch_and_honesty_gate():
    specialist = StubActionGraph('No result.')
    chat = build_graph(fake_model('chat'), soul='s')
    assert ask(chat, 'catalog entry', identity={'user_id': 'test'}, thread_id='test',
               router=lambda _: RouteDecision('lookup', ''),
               specialists={'lookup': specialist}) == 'No result.'
    assert len(specialist.calls) == 2
    assert action_honesty_gate('lookup', [], already_retried=False,
                               specialists={'lookup': specialist}) == GATE_RETRY
