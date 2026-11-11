import base64
from contextlib import nullcontext
from types import SimpleNamespace

import httpx
import pytest
from pydantic import SecretStr

from aerys_v2.tools import gmail_drafts as gmail
from test_gmail_drafts import OWNER, VALUES, settings


def part(text, mime='text/plain', **kw):
    return {'mimeType': mime, 'body': {'data': base64.urlsafe_b64encode(text.encode()).decode(),
                                     'size': len(text)}, **kw}


def message(payload=None):
    payload = payload or part('Mailbox words')
    payload['headers'] = [{'name': name, 'value': value} for name, value in
                          [('From', 'sender@example.com'), ('To', 'owner@example.com'),
                           ('Cc', 'copy@example.com'), ('Subject', 'A subject'), ('Date', 'Today')]]
    return {'id': 'abc123', 'snippet': 'A snippet', 'payload': payload}


def setup(payload=None, failure=None):
    requests = []
    def respond(req):
        requests.append(req)
        url = str(req.url.copy_with(query=None))
        assert (req.method, url) in {
            ('POST', gmail.TOKEN_URL), ('POST', gmail.DRAFT_URL),
            ('GET', gmail.MESSAGES_URL), ('GET', gmail.MESSAGES_URL + '/abc123')}
        if failure:
            result = failure(req)
            if result is not None:
                return result
        if url == gmail.TOKEN_URL:
            return httpx.Response(200, json={'access_token': 'placeholder', 'expires_in': 3600})
        assert req.headers['Authorization'] == 'Bearer placeholder'
        if url == gmail.DRAFT_URL:
            return httpx.Response(200, json={'id': 'draft'})
        if url == gmail.MESSAGES_URL:
            return httpx.Response(200, json={'messages': [{'id': 'abc123'}]})
        assert dict(req.url.params) == {'format': 'full'}
        return httpx.Response(200, json=message(payload))
    client = gmail.GmailDrafts(client_id='placeholder', client_secret=SecretStr('placeholder'),
                              refresh_token=SecretStr('placeholder'), from_address='owner@example.com',
                              transport=httpx.MockTransport(respond))
    return client, requests


def test_allowed_calls_search_read_draft_and_token_cache(monkeypatch):
    now = [100.0]
    monkeypatch.setattr(gmail.time, 'monotonic', lambda: now[0])
    client, requests = setup()
    result = client.search('from:sender@example.com', 99)
    assert result.startswith(gmail.DATA_HEADER)
    for text in ('id: abc123', 'From: sender@example.com', 'Subject: A subject', 'Date: Today', 'A snippet'):
        assert text in result
    assert dict(requests[1].url.params) == {'q': 'from:sender@example.com', 'maxResults': '10'}
    assert 'Mailbox words' in client.read('abc123')
    assert client.create('recipient@example.com', 'Reply', 'Words').startswith('Drafted:')
    assert sum(str(r.url) == gmail.TOKEN_URL for r in requests) == 1
    now[0] += 3541
    assert 'Mailbox words' in client.read('abc123')
    assert sum(str(r.url) == gmail.TOKEN_URL for r in requests) == 2


@pytest.mark.parametrize('ident', ['', '../drafts', 'abc/def', 'abc?format=raw', 'abc#x', '%61', 'abc\\def', 'é', 'abc\n', 'x'*129])
def test_invalid_ids_never_call(ident):
    client, requests = setup()
    assert client.read(ident).startswith("Couldn't read")
    assert requests == []


def test_bad_search_id_never_fetches():
    def failure(req):
        if str(req.url.copy_with(query=None)) == gmail.MESSAGES_URL:
            return httpx.Response(200, json={'messages': [{'id': '../drafts'}]})
    client, requests = setup(failure=failure)
    assert client.search('test').startswith("Couldn't read")
    assert len(requests) == 2


@pytest.mark.parametrize('maximum,expected', [(0, 1), (-10, 1), (5, 5), (100, 10)])
def test_search_cap(maximum, expected):
    client, requests = setup()
    client.search('test', maximum)
    assert requests[1].url.params['maxResults'] == str(expected)


def test_plain_preferred_clipped_and_attachment_not_fetched():
    payload = {'mimeType': 'multipart/mixed', 'parts': [
        {'mimeType': 'multipart/alternative', 'parts': [part('<b>HTML fallback</b>', 'text/html'),
                                                      part('x' * (gmail.READ_BODY_LIMIT + 10))]},
        part('ATTACHMENT SECRET', filename='notes.txt'),
        {'filename': 'image.png', 'body': {'attachmentId': 'do-not-fetch', 'size': 1234}}]}
    client, requests = setup(payload)
    result = client.read('abc123')
    assert '(clipped)' in result and 'x' * gmail.READ_BODY_LIMIT in result
    assert 'HTML fallback' not in result and 'ATTACHMENT SECRET' not in result
    assert 'notes.txt (17 bytes)' in result and 'image.png (1234 bytes)' in result
    assert 'Cc: copy@example.com' in result and len(requests) == 2


def test_html_reduced():
    client, _ = setup(part('<style>hidden</style><p>Hello &amp; welcome</p><script>hidden</script><div>Next</div>', 'text/html'))
    result = client.read('abc123')
    assert 'Hello & welcome' in result and '\nNext' in result
    assert '<p>' not in result and 'hidden' not in result


@pytest.mark.parametrize('operation,endpoint', [('search', 'token'), ('search', 'list'), ('search', 'get'),
                                                  ('read', 'token'), ('read', 'get')])
@pytest.mark.parametrize('error', [302, 403, 500, 'timeout', 'malformed', 'badtype'])
def test_errors_honest(operation, endpoint, error):
    def failure(req):
        target = {'token': gmail.TOKEN_URL, 'list': gmail.MESSAGES_URL, 'get': gmail.MESSAGES_URL + '/abc123'}[endpoint]
        if str(req.url.copy_with(query=None)) != target:
            return
        if error == 'timeout':
            raise httpx.ReadTimeout('private response')
        if error == 'malformed':
            return httpx.Response(200, text='private response')
        if error == 'badtype':
            return httpx.Response(200, json=[])
        return httpx.Response(error, text='private response', headers={'Location': 'https://example.com'})
    client, _ = setup(failure=failure)
    result = client.search('test') if operation == 'search' else client.read('abc123')
    assert result.startswith("Couldn't read") and 'private response' not in result


@pytest.mark.parametrize('identity,thread', [({}, 'owner:test'), ({'user_id': 'guest'}, 'owner:test'),
    ({'user_id': 'other'}, 'owner:test'), ({'user_id': OWNER}, 'kael:test')])
def test_owner_only(identity, thread):
    client, requests = setup()
    search, read = gmail.build_gmail_read_tools(client, owner_person_id=OWNER)
    runtime = SimpleNamespace(config={'configurable': {'identity': identity, 'thread_id': thread}})
    assert search.func(query='test', runtime=runtime).startswith("Couldn't read")
    assert read.func(message_id='abc123', runtime=runtime).startswith("Couldn't read")
    assert requests == []


@pytest.mark.parametrize('drafts,reads', [(False, False), (True, False), (False, True), (True, True)])
def test_independent_switches(drafts, reads, caplog):
    from aerys_v2.factory import action_tools_for
    baseline = {t.name for t in action_tools_for(settings())}
    configured = settings(**{**VALUES, 'gmail_drafts': drafts, 'gmail_read': reads})
    with caplog.at_level('INFO'):
        names = {t.name for t in action_tools_for(configured)}
    expected = ({'draft_in_his_gmail'} if drafts else set()) | ({'search_his_email', 'read_his_email'} if reads else set())
    assert names == baseline | expected | ({'no_action'} if expected else set())
    assert not expected.intersection(t.name for t in action_tools_for(configured, guest=True))
    assert caplog.text.count('gmail drafts ') == 1 and caplog.text.count('gmail read ') == 1
    assert not settings().gmail_read


@pytest.mark.parametrize('key', [k for k in VALUES if k != 'gmail_drafts'])
def test_reads_need_credentials(key, caplog):
    configured = settings(**{**VALUES, 'gmail_read': True, key: None})
    with caplog.at_level('INFO'):
        assert gmail.gmail_reads_for(configured) == []
    assert key.upper() in caplog.text and 'placeholder' not in caplog.text


def test_worker_never_builds_owner_tools(monkeypatch):
    import psycopg
    import aerys_v2.workers.__main__ as worker
    import aerys_v2.workers.email_watch as watch
    def forbidden(*args, **kwargs):
        pytest.fail('worker built owner Gmail tools')
    for name in ('GmailDrafts', 'gmail_tools_for', 'gmail_reads_for', 'gmail_drafts_for', 'build_gmail_read_tools'):
        monkeypatch.setattr(gmail, name, forbidden)
    observed = []
    monkeypatch.setattr(watch, 'imap_login_factory', lambda *args: observed.append(args) or 'her mailbox')
    monkeypatch.setattr(psycopg, 'connect', lambda *a, **kw: nullcontext(SimpleNamespace(execute=lambda *a: None)))
    monkeypatch.setattr(watch, 'run_once', lambda conn, factory, notify: {'mailbox': factory})
    configured = settings(**{**VALUES, 'gmail_read': True, 'email_address': 'hers@example.com',
                             'email_app_password': 'placeholder', 'database_url': 'postgresql://placeholder/aerys_v2'})
    assert worker._email_watch_once(configured, lambda *a: None) == {'mailbox': 'her mailbox'}
    assert observed == [(configured.email_imap_host, 'hers@example.com', 'placeholder')]


@pytest.mark.parametrize('reader', ['search_his_email', 'read_his_email'])
def test_actual_engine_read_then_draft_and_send(reader):
    from aerys_v2.factory import build_action_graph
    from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
    from langchain_core.tools import tool
    from aerys_v2.outside_guard import READERS, REACH_OUT
    client, requests = setup()
    reads = gmail.build_gmail_read_tools(client, owner_person_id=OWNER)
    draft = gmail.build_gmail_draft_tool(client, owner_person_id=OWNER)
    assert reader in READERS and draft.name not in REACH_OUT
    @tool
    def send_email(confirmed: bool = False) -> str:
        """Test send."""
        pytest.fail('sent after reading')
    calls = [(reader, {'query': 'test'} if reader.startswith('search') else {'message_id': 'abc123'}),
             (draft.name, {'to': 'recipient@example.com', 'subject': 'Reply', 'body': 'Words'}),
             ('send_email', {'confirmed': True})]
    model = GenericFakeChatModel(messages=iter([
        *[AIMessage(content='', tool_calls=[{'name': name, 'args': args, 'id': str(i)}])
          for i, (name, args) in enumerate(calls)], AIMessage(content='done')]))
    graph = build_action_graph(model, 'test', [*reads, draft, send_email])
    result = graph.invoke({'messages': [HumanMessage(content='Read my mail and draft a reply')]},
                          {'configurable': {'identity': {'user_id': OWNER, 'platform': 'discord'}, 'thread_id': 'owner:test', 'outside_text': {}}})
    receipts = [m.content for m in result['messages'] if isinstance(m, ToolMessage)]
    assert any(r.startswith(gmail.DATA_HEADER) for r in receipts)
    assert any(r.startswith('Drafted:') for r in receipts)
    assert any('Not sent' in r and 'Approve' in r for r in receipts)


@pytest.mark.parametrize('reader', ['search_his_email', 'read_his_email'])
def test_actual_engine_message_waits_for_tap(reader):
    from aerys_v2.factory import build_action_graph
    from aerys_v2.tools.send_message import MessageSender, MessageSends, build_send_message_tool
    from aerys_v2.hands_gate import MESSAGE
    from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
    from test_hands_gate import Box, OWNER as GATE_OWNER, TEXT, OWNER_THREAD
    b = Box()
    delivered = []
    sender = MessageSender(b.gate, [], MessageSends(), lambda *args: delivered.append(args))
    b.gate.executors[MESSAGE] = sender.execute
    client, _ = setup()
    reads = gmail.build_gmail_read_tools(client, owner_person_id=GATE_OWNER)
    model = GenericFakeChatModel(messages=iter([
        AIMessage(content='', tool_calls=[{'name': reader, 'args': {'query': 'test'} if reader.startswith('search')
                                          else {'message_id': 'abc123'}, 'id': 'read'}]),
        AIMessage(content='', tool_calls=[{'name': 'send_message', 'args': {'to': 'chris_h', 'text': 'Mailbox summary'}, 'id': 'send'}]),
        AIMessage(content='done')]))
    graph = build_action_graph(model, 'test', [*reads, build_send_message_tool(sender)], outside_gate=b.gate,
                               guard_owner=GATE_OWNER)
    result = graph.invoke({'messages': [HumanMessage(content='Read my mail and message me')]},
                          {'configurable': {'identity': TEXT, 'thread_id': OWNER_THREAD, 'outside_text': {}}})
    assert not delivered
    assert any(isinstance(m, ToolMessage) and 'Waiting' in m.content for m in result['messages'])
    b.watcher.tick()
    assert not delivered
    b.tap()
    b.watcher.tick()
    assert len(delivered) == 1


def test_search_limits_detail_fetches_even_if_server_overreturns():
    def failure(req):
        if str(req.url.copy_with(query=None)) == gmail.MESSAGES_URL:
            return httpx.Response(200, json={'messages': [{'id': 'abc123'}] * 20})
    client, requests = setup(failure=failure)
    assert client.search('test', 100).count('id: abc123') == 10
    assert len(requests) == 12


def test_exchange_missing_scope_rejected():
    import io
    import json
    from test_gmail_drafts_signin import signin
    with pytest.raises(ValueError):
        signin.exchange_code('client', 'secret', 'code', 'redirect', 'verifier',
            open_url=lambda *a, **kw: io.BytesIO(json.dumps({'refresh_token': 'placeholder'}).encode()))


@pytest.mark.parametrize('identity,thread,ok', [
    ({'user_id': OWNER, 'platform': 'discord'}, 'person:owner', True),                    # his typed line
    ({'user_id': OWNER, 'voice': True, 'speaker': {'id': 'owner-voice'}}, 'voice:office', True),   # his voice, sure
    ({'user_id': OWNER, 'voice': True, 'speaker': {'id': 'owner-voice', 'unsure': True}}, 'voice:office', False),
    ({'user_id': OWNER, 'voice': True, 'voice_owner_assumed': True}, 'voice:office', False),    # assumed, not heard
    ({'user_id': OWNER}, 'owner:panel', False),                                            # a shared surface's /ask
])
def test_his_mail_is_read_only_to_him(identity, thread, ok):
    client, requests = setup()
    search, read = gmail.build_gmail_read_tools(client, owner_person_id=OWNER)
    runtime = SimpleNamespace(config={'configurable': {'identity': identity, 'thread_id': thread}})
    said = read.func(message_id='abc123', runtime=runtime)
    if ok:
        assert requests and not said.startswith("Couldn't read")
    else:
        assert said == gmail.UNSURE_READER and requests == []
