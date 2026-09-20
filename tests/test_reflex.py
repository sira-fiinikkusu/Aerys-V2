"""The shadow observer has one message of authority and a hard time budget."""
import time
from types import SimpleNamespace

import pytest
from pydantic import SecretStr

from aerys_v2.config import Settings
from aerys_v2.reflex import ReflexClient, reflex_for


class FakeClient:
    def __init__(self, score=1.2, delay=0, error=None):
        self.score, self.delay, self.error = score, delay, error
        self.seen = None

    def system_one(self, **kwargs):
        self.seen = kwargs
        time.sleep(self.delay)
        if self.error:
            raise self.error
        return SimpleNamespace(
            answers={
                'route': SimpleNamespace(choice='action', probabilities={'action': .8, 'chat': .2}, confidence=.7),
                'tier': SimpleNamespace(score=self.score, confidence=.8),
                'unaddressed': SimpleNamespace(noul=.1),
                'cancelled': SimpleNamespace(noul=.05),
            }, model='jev-1.13.0', usage=SimpleNamespace(input_tokens=123))


def test_result_shape_and_state_boundary():
    fake = FakeClient()
    result = ReflexClient(client=fake)('x' * 2200, {
        'surface': 'guild', 'thread': 'never send', 'memories': ['never send'],
        'room_context': 'never send', 'portable': 'never send'})
    assert fake.seen['state'] == {'message': 'x' * 2000, 'surface': 'guild'}
    assert set(fake.seen['questions']) == {'route', 'tier', 'unaddressed', 'cancelled'}
    assert 'assistant_previous_reply' not in fake.seen['state']
    assert result == {
        'route': 'action', 'p_action': .8, 'confidence': .7, 'tier': 'standard',
        'tier_score': 1.2, 'unaddressed': .1, 'cancelled': .05, 'latency_ms': result['latency_ms'],
        'model': 'jev-1.13.0', 'input_tokens': 123}
    assert isinstance(result['latency_ms'], int)


@pytest.mark.parametrize('score,tier', [(.4, 'fast'), (1.2, 'standard'), (1.6, 'deep'), (-2, 'fast'), (9, 'deep')])
def test_tier_rounding_and_clamping(score, tier):
    assert ReflexClient(client=FakeClient(score))('hi', {})['tier'] == tier


def test_timeout_is_bounded_and_never_raises():
    started = time.monotonic()
    result = ReflexClient(client=FakeClient(delay=.3), timeout_s=.03)('hi', {})
    assert result['error'] == 'timeout'
    assert isinstance(result['latency_ms'], int)
    assert time.monotonic() - started < .2


def test_exception_is_an_error_dict():
    result = ReflexClient(client=FakeClient(error=ValueError('x' * 150)))('hi', {})
    assert result['error'] == 'ValueError: ' + 'x' * 120


@pytest.mark.parametrize('mode,key', [('off', SecretStr('test')), ('shadow', None), ('live', None)])
def test_unarmed(mode, key):
    settings = Settings(_env_file=None, anthropic_api_key='test', reflex_mode=mode, typesafe_api_key=key)
    assert reflex_for(settings) is None


def test_settings_defaults_and_invalid_mode():
    settings = Settings(_env_file=None, anthropic_api_key='test')
    assert (settings.reflex_mode, settings.reflex_model, settings.reflex_timeout_s) == ('off', 'jev-1.13.0', .6)
    with pytest.raises(ValueError):
        Settings(_env_file=None, anthropic_api_key='test', reflex_mode='typo')


def test_factory_pins_model_and_timeout(monkeypatch, caplog):
    import sys
    import aerys_v2.reflex as module

    constructed = []

    def client(**kwargs):
        constructed.append(kwargs)
        return FakeClient()

    monkeypatch.setitem(sys.modules, 'typesafe_sdk', SimpleNamespace(
        TypeSafeClient=client, RetryPolicy=lambda **kw: kw))
    monkeypatch.setattr(module, '_LIVE_WARNED', False)
    settings = Settings(_env_file=None, anthropic_api_key='test',
                        typesafe_api_key='fake-key', reflex_mode='live', reflex_timeout_s=.12)
    first = reflex_for(settings)
    second = reflex_for(settings)
    assert first('hello', {})['route'] == 'action'
    assert first.timeout_s == second.timeout_s == .12
    assert constructed[0] == dict(api_key='fake-key', model='jev-1.13.0',
                                  timeout=.12, retry={'max_retries': 0})
    # live is a real mode since Phase 2; the factory no longer warns about it.
    assert not any('reserved' in r.message for r in caplog.records)


def test_sdk_question_contract():
    from aerys_v2.reflex import QUESTIONS

    assert QUESTIONS['route'] == {
        'type': 'choice',
        'instructions': 'Which path must handle this message from the owner to his home assistant Aerys?',
        'criteria': {
            'action': "It needs to TOUCH or READ something outside the conversation: control a device, read live device/sensor state (charge, temperature, on/off, locked), look at an attachment/image/PDF/video link, search the web or current events/weather/prices/shopping availability, read or send email, keep a note or reminder for later ('remember that…', 'remind me'), play or control music, set a timer, or message Kael.",
            'chat': "Pure conversation: opinions, feelings, general knowledge that cannot have changed, reminiscing ('remember when…'), greetings, acknowledgments, jokes, planning talk that touches nothing.",
        },
    }
    assert QUESTIONS['tier'] == {
        'type': 'score', 'instructions': 'How much thinking does the reply deserve?',
        'criteria': ['fast: greeting, one-word acknowledgment, small talk, trivial',
                     'standard: an ordinary question or request',
                     'deep: multi-step reasoning, analysis, something emotionally weighty or ambiguous'],
    }
    assert QUESTIONS['unaddressed'] == {
        'type': 'noul',
        'instructions': 'The message is clearly NOT directed at the assistant at all (overheard speech, a wake word landing mid-sentence, talk to another person). Short acknowledgments and follow-ups ARE addressed.',
    }


@pytest.mark.parametrize('timeout', [0, -1, float('inf'), float('nan')])
def test_invalid_timeout_rejected(timeout):
    with pytest.raises(ValueError):
        Settings(_env_file=None, anthropic_api_key='test', reflex_timeout_s=timeout)


def test_option_b_context_adds_the_ctx_question_and_score():
    class Ctx(FakeClient):
        def system_one(self, **kwargs):
            r = super().system_one(**kwargs)
            r.answers['unaddressed_ctx'] = SimpleNamespace(noul=.91)
            return r
    ctx = Ctx()
    result = ReflexClient(client=ctx)('is the question I asked', {'surface': 'voice', 'last_reply': 'What was the question?'})
    assert ctx.seen['state']['assistant_previous_reply'] == 'What was the question?'
    assert 'unaddressed_ctx' in ctx.seen['questions'] and result['unaddressed_ctx'] == .91
