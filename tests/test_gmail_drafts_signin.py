import importlib.util
import io
import json
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import pytest

spec = importlib.util.spec_from_file_location('gmail_drafts_signin', Path(__file__).parents[1] / 'scripts/gmail_drafts_signin.py')
signin = importlib.util.module_from_spec(spec)
spec.loader.exec_module(signin)


def test_consent_url():
    url = signin.consent_url('placeholder-client', 'http://127.0.0.1:12345/', 'state', 'challenge')
    parsed = urlsplit(url)
    assert parsed.scheme == 'https' and parsed.netloc == 'accounts.google.com'
    query = parse_qs(parsed.query)
    assert set(query['scope'][0].split()) == {'https://www.googleapis.com/auth/gmail.readonly', 'https://www.googleapis.com/auth/gmail.compose'}
    assert query['access_type'] == ['offline'] and query['prompt'] == ['consent']
    assert query['redirect_uri'] == ['http://127.0.0.1:12345/']
    assert query['state'] == ['state'] and query['response_type'] == ['code']
    assert query['code_challenge_method'] == ['S256']
    assert query.get('include_granted_scopes', ['false']) == ['false']


def test_exchange():
    calls = []
    def open_fake(request, timeout):
        calls.append(request)
        assert timeout == 20
        return io.BytesIO(json.dumps({'refresh_token': 'placeholder-refresh', 'scope': signin.SCOPE}).encode())
    assert signin.exchange_code('placeholder-client', 'placeholder-secret', 'code',
                                'http://127.0.0.1:12345/', 'verifier', open_url=open_fake) == 'placeholder-refresh'
    request, = calls
    assert request.full_url == 'https://oauth2.googleapis.com/token'
    assert request.get_method() == 'POST'
    assert parse_qs(request.data.decode()) == {
        'grant_type': ['authorization_code'], 'client_id': ['placeholder-client'],
        'client_secret': ['placeholder-secret'], 'code': ['code'],
        'redirect_uri': ['http://127.0.0.1:12345/'], 'code_verifier': ['verifier']}


@pytest.mark.parametrize('payload', [{}, {'refresh_token': ''}, [], {'scope': 'other', 'refresh_token': 'placeholder'}])
def test_exchange_rejects_bad_response(payload):
    with pytest.raises(ValueError):
        signin.exchange_code('client', 'secret', 'code', 'http://127.0.0.1:12345/', 'verifier',
                             open_url=lambda *a, **kw: io.BytesIO(json.dumps(payload).encode()))


@pytest.mark.parametrize('query,accepted', [
    ('state=expected&code=placeholder-code', True),
    ('state=wrong&code=placeholder-code', False),
    ('code=placeholder-code', False),
    ('state=expected&state=wrong&code=placeholder-code', False),
    ('state=expected&code=one&code=two', False),
    ('state=expected&error=access_denied', True),
])
def test_callback_state_and_no_logging(query, accepted, capsys):
    result = {}
    handler_type = signin.callback_handler('expected', result)
    handler = object.__new__(handler_type)
    handler.path = '/?' + query
    handler.wfile = io.BytesIO()
    statuses = []
    handler.send_response = statuses.append
    handler.send_header = lambda *a: None
    handler.end_headers = lambda: None
    handler.do_GET()
    handler.log_message('private-code')
    assert bool(result) is accepted
    assert statuses == [200 if accepted else 400]
    assert 'placeholder-code' not in handler.wfile.getvalue().decode()
    assert capsys.readouterr() == ('', '')


@pytest.mark.parametrize('value', ['bad\nKEY=value', '${EXPAND}', "quote'", '', 'a b'])
def test_env_output_refuses_injection(value):
    with pytest.raises(ValueError):
        signin.env_lines('client', value, 'refresh')


def test_main_stdout_only_env(monkeypatch, capsys):
    prompts = iter(['placeholder-client', 'placeholder-secret'])
    monkeypatch.setattr(signin.getpass, 'getpass', lambda *a, **kw: next(prompts))
    observed = {}
    original_handler = signin.callback_handler
    def handler(state, result):
        observed['result'] = result
        return original_handler(state, result)
    monkeypatch.setattr(signin, 'callback_handler', handler)
    class Server:
        server_port = 12345
        def __init__(self, address, handler):
            assert address == ('127.0.0.1', 0)
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def handle_request(self):
            observed['result'].update(code='placeholder-code')
    monkeypatch.setattr(signin, 'HTTPServer', Server)
    monkeypatch.setattr(signin.webbrowser, 'open', lambda url: observed.update(url=url) or True)
    def exchange(client, secret, code, redirect, verifier):
        query = parse_qs(urlsplit(observed['url']).query)
        challenge = signin.base64.urlsafe_b64encode(signin.hashlib.sha256(verifier.encode()).digest()).rstrip(b'=').decode()
        assert query['code_challenge'] == [challenge]
        assert redirect == 'http://127.0.0.1:12345/'
        return 'placeholder-refresh'
    monkeypatch.setattr(signin, 'exchange_code', exchange)
    assert signin.main([]) == 0
    assert capsys.readouterr() == (
        'GMAIL_DRAFTS_CLIENT_ID=placeholder-client\n'
        'GMAIL_DRAFTS_CLIENT_SECRET=placeholder-secret\n'
        'GMAIL_DRAFTS_REFRESH_TOKEN=placeholder-refresh\n', '')


def test_main_failure_no_credentials(monkeypatch, capsys):
    def fail(*args, **kw):
        raise ValueError('private-response')
    monkeypatch.setattr(signin.getpass, 'getpass', fail)
    assert signin.main([]) == 1
    captured = capsys.readouterr()
    assert captured.out == '' and 'private-response' not in captured.err


def test_help(capsys):
    with pytest.raises(SystemExit) as exc:
        signin.main(['--help'])
    assert exc.value.code == 0
    text = ' '.join(capsys.readouterr().out.split())
    assert 'scope can create drafts and send' in text
    assert 'only creates drafts' in text

@pytest.mark.parametrize('scope', [None, '', 'https://www.googleapis.com/auth/gmail.compose', 'https://www.googleapis.com/auth/gmail.readonly', 'https://mail.google.com/', 'https://www.googleapis.com/auth/gmail.readonly https://www.googleapis.com/auth/gmail.compose extra'])
def test_exact_scope_set(scope):
    with pytest.raises(ValueError):
        signin.exchange_code('client', 'secret', 'code', 'redirect', 'verifier',
            open_url=lambda *a, **kw: io.BytesIO(json.dumps({'refresh_token': 'placeholder', 'scope': scope}).encode()))


def test_scope_order():
    assert signin.exchange_code('client', 'secret', 'code', 'redirect', 'verifier',
        open_url=lambda *a, **kw: io.BytesIO(json.dumps({'refresh_token': 'placeholder',
            'scope': ' '.join(reversed(signin.SCOPE.split()))}).encode())) == 'placeholder'
