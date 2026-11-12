"""D34: local speaker interruption and model-facing wording, offline."""
from unittest.mock import Mock

import httpx
import pytest

from aerys_v2.speaker_redirect import SpeakerRedirect
from aerys_v2.transports.http_api import AskRequest, build_app


@pytest.mark.parametrize('text,playing,voice,device,stops', [
    ('Aerys, stop!', True, True, 'office', True),
    ('stop harris', True, True, 'office', True),
    ('stop', False, True, 'office', False),
    ('stop the music', True, True, 'office', False),
    ('stop the timer', True, True, 'office', False),
    ('stop please', True, True, 'office', False),
    ('stop', True, False, 'office', False),
    ('stop', True, True, 'other', False),
])
def test_stop_shortcut(text, playing, voice, device, stops):
    ask = Mock(return_value='normal reply')
    speaker = Mock()
    speaker.handles.side_effect = lambda d: d == 'office'
    speaker.playing.return_value = playing
    app = build_app(ask, 'placeholder', speaker_redirect=speaker)
    route = next(r.endpoint for r in app.routes if r.path == '/ask')
    result = route(AskRequest(text=text, voice=voice, device_id=device))
    assert result.reply == ('' if stops else 'normal reply')
    assert result.keep_listening is False
    if stops:
        speaker.stop.assert_called_once_with(device)
        ask.assert_not_called()
        speaker.say.assert_not_called()
    else:
        ask.assert_called_once()
        speaker.stop.assert_not_called()


@pytest.mark.parametrize('method,path,key,timeout', [
    ('playing', '/health', 'playing', 1.5), ('stop', '/stop', 'stopped', 3.0),
])
@pytest.mark.parametrize('outcome', ['true', 'false', 'error', 'timeout', 'invalid'])
def test_redirect_control(method, path, key, timeout, outcome):
    calls = []
    def handle(req):
        calls.append(req)
        assert req.url.path == path
        assert req.method == ('GET' if method == 'playing' else 'POST')
        assert req.headers['authorization'] == 'Bearer placeholder'
        assert req.extensions['timeout']['read'] == timeout
        if outcome == 'timeout':
            raise httpx.ReadTimeout('offline')
        if outcome == 'invalid':
            return httpx.Response(200, text='invalid json')
        return httpx.Response(503 if outcome == 'error' else 200,
                              json={key: outcome == 'true'})
    speaker = SpeakerRedirect(
        ha_base_url='http://ha.example', ha_token='placeholder', engine='tts.example',
        voice='example', language='en', targets={'office': 'http://speaker.example'},
        speaker_token='placeholder', client=httpx.Client(transport=httpx.MockTransport(handle)),
    )
    assert getattr(speaker, method)('office') is (outcome == 'true')
    assert getattr(speaker, method)('other') is False
    assert len(calls) == 1


def test_ask_first_descriptions():
    from aerys_v2.tools.message_kael import build_message_kael_tool
    from aerys_v2.tools.kael_tickets import build_kael_tickets_tool
    for tool in [build_message_kael_tool('http://desk.example', 'placeholder'),
                 build_kael_tickets_tool(Mock(), Mock())]:
        assert 'ask them first' in tool.description
        assert 'actual content' in tool.description


def test_voice_wording():
    import inspect
    from aerys_v2 import factory
    assert 'Keep spoken replies short' in factory.VOICE_BANTER_OVERLAY
    source = inspect.getsource(factory)
    assert source.count('Keep spoken replies short') == 2
    assert source.count('never read lists, headings or markdown aloud') == 2
    assert source.count('you may keep talking over voice') == 2


@pytest.mark.parametrize('phrase', [
    'stop', 'stop talking', 'stop it', 'okay stop', 'ok stop', 'hold on',
    'hold up', "that's enough", 'that’s enough', 'enough', 'be quiet',
    'quiet', 'shush', 'shh',
])
@pytest.mark.parametrize('name', ['aerys', 'harris', 'iris', 'ares', 'aries', 'eris'])
def test_all_stop_phrases_and_names(phrase, name):
    from aerys_v2.transports.http_api import _is_stop_phrase
    assert _is_stop_phrase(f'  {name.upper()}, {phrase}!  ')
    assert _is_stop_phrase(f'{phrase}, {name}.')
    assert not _is_stop_phrase(f'{phrase} the music')


def test_killed_clip_does_not_reopen_microphone():
    def handle(req):
        if req.url.path == '/api/tts_get_url':
            return httpx.Response(200, json={'url': 'http://ha.example/clip.mp3'})
        return httpx.Response(200, json={'played': False})
    speaker = SpeakerRedirect(
        ha_base_url='http://ha.example', ha_token='placeholder', engine='tts.example',
        voice='example', language='en', targets={'office': 'http://speaker.example'},
        speaker_token='placeholder', client=httpx.Client(transport=httpx.MockTransport(handle)),
        async_send=False,
    )
    reopen = Mock()
    speaker.say('office', 'More?', then=reopen)
    reopen.assert_not_called()
