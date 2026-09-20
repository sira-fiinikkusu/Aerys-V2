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
    result = ReflexClient(client=ctx)('is the question I asked', {'surface': 'voice', 'last_reply': 'What was the question?',
                                                                  'unaddressed_source': 'ctx'})
    assert ctx.seen['state']['assistant_previous_reply'] == 'What was the question?'
    assert 'unaddressed_ctx' in ctx.seen['questions'] and result['unaddressed_ctx'] == .91


def test_round6_addressee_choice_on_a_trimmed_state():
    # Chris 2026-09-20 13:19 ("6 of 16 is not acceptable"): the default voice gate is a
    # Choice over WHO the capture is for, on a trimmed state — the 8-exchange history is
    # NOT sent (jev-1.13 jaggedness #5), time arrives as words, and the sum of the
    # background classes is the score.
    class Addr(FakeClient):
        def system_one(self, **kwargs):
            r = super().system_one(**kwargs)
            r.answers['addressee'] = SimpleNamespace(
                choice='another_person', confidence=.8,
                probabilities={'request_to_assistant': .02, 'reply_to_assistant': .03, 'correction_to_assistant': .01,
                               'another_person': .8, 'media_speech': .04, 'work_meeting_talk': .05, 'fragment_nobody': .05})
            return r
    fake = Addr()
    result = ReflexClient(client=fake)('so I told Megan we would probably leave around nine', {
        'surface': 'voice', 'last_reply': 'Turning on the sunroom lights.', 'seconds_since_assistant_spoke': 400.0,
        'device': 'office satellite', 'local_time': 'Sunday 13:00',
        'previous_capture': {'text': 'hey how are you', 'seconds_ago': 400.0, 'outcome': 'answered normally'},
        'background_captures_5m': 2,
        'recent_exchanges': [{'user': 'never send', 'assistant': 'never send'}] * 8})
    st = fake.seen['state']
    assert 'recent_exchanges' not in st and 'never send' not in str(st)
    assert st['assistant_previous_line'] == 'Turning on the sunroom lights.'
    assert st['assistant_last_spoke'] == 'a few minutes ago'
    assert st['previous_capture'] == {'how_long_ago': 'a few minutes ago', 'text': 'hey how are you', 'outcome': 'answered normally'}
    assert st['background_captures_in_last_5_minutes'] == 'several'
    assert 'unaddressed_ctx' not in fake.seen['questions'] and fake.seen['questions']['addressee']['type'] == 'choice'
    assert result['addressee'] == {'choice': 'another_person', 'confidence': .8, 'background': pytest.approx(.94)}


def test_round6_time_buckets_and_bypass_rules():
    from aerys_v2.reflex import time_bucket, unaddressed_bypass

    assert [time_bucket(v) for v in (None, 3, 15, 16, 120, 121, 600, 601)] == [
        'never', 'just now (within 15 seconds)', 'just now (within 15 seconds)', 'a moment ago (within 2 minutes)',
        'a moment ago (within 2 minutes)', 'a few minutes ago', 'a few minutes ago', 'not recently (over 10 minutes ago)']
    # a reply within 20 s of her question, or a few words within 30 s of any line of hers, is never background
    assert unaddressed_bypass('the office lights', {'last_reply': 'Which lights?', 'seconds_since_assistant_spoke': 12}) == 'question'
    assert unaddressed_bypass('In all of the office.', {'last_reply': 'Turning off the office light now.', 'seconds_since_assistant_spoke': 17}) == 'short_followup'
    assert unaddressed_bypass('and then he just left it in the driveway all weekend', {'last_reply': 'Turning it off.', 'seconds_since_assistant_spoke': 5}) is None
    # her question + a STORY 5 s later is not a reply to it (11 words; round-6 row at 0.99)
    assert unaddressed_bypass('and then he just left it in the driveway all weekend', {'last_reply': 'How about you?', 'seconds_since_assistant_spoke': 5}) is None
    assert unaddressed_bypass('the office lights', {'last_reply': 'Which lights?', 'seconds_since_assistant_spoke': 35}) is None
    assert unaddressed_bypass('yes', {'last_reply': 'Which lights?'}) is None
