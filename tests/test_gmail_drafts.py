import base64
import json
from email import policy
from email.parser import BytesParser
from types import SimpleNamespace
from urllib.parse import parse_qs

import httpx
import pytest
from pydantic import SecretStr

from aerys_v2.tools.gmail_drafts import GmailDrafts, build_gmail_draft_tool, gmail_drafts_for

TOKEN = 'https://oauth2.googleapis.com/token'
DRAFT = 'https://gmail.googleapis.com/gmail/v1/users/me/drafts'
OWNER = '00000000-0000-0000-0000-000000000001'
VALUES = dict(gmail_drafts=True, gmail_drafts_client_id='placeholder-client',
              gmail_drafts_client_secret='placeholder-secret', gmail_drafts_refresh_token='placeholder-refresh',
              gmail_drafts_from='owner@example.com', owner_person_id=OWNER)


def settings(**kw):
    from aerys_v2.config import Settings
    return Settings(_env_file=None, anthropic_api_key='placeholder', **kw)


def setup(handler=None):
    requests = []
    def respond(request):
        requests.append(request)
        assert request.method == 'POST'
        assert str(request.url) in (TOKEN, DRAFT), 'unexpected endpoint'
        assert all(v == 20 for v in request.extensions['timeout'].values())
        if handler:
            return handler(request)
        return httpx.Response(200, json={'access_token': 'placeholder-access'} if str(request.url) == TOKEN
                              else {'id': 'placeholder-draft'})
    client = GmailDrafts(client_id='placeholder-client', client_secret=SecretStr('placeholder-secret'),
                         refresh_token=SecretStr('placeholder-refresh'), from_address='owner@example.com',
                         transport=httpx.MockTransport(respond))
    return build_gmail_draft_tool(client, owner_person_id=OWNER), requests


def call(tool, identity=None, thread='owner:test', **kw):
    args = dict(to='recipient@example.com', subject='A subject', body='Hello café.\nSecond line.')
    args.update(kw)
    runtime = SimpleNamespace(config={'configurable': {
        'identity': {'user_id': OWNER} if identity is None else identity, 'thread_id': thread}})
    return tool.func(runtime=runtime, **args)


def test_create_only_and_message():
    tool, requests = setup()
    out = call(tool, cc='copy@example.com')
    assert out == 'Drafted: A subject — to recipient@example.com; cc copy@example.com.'
    assert [str(r.url) for r in requests] == [TOKEN, DRAFT]
    assert parse_qs(requests[0].content.decode()) == {
        'grant_type': ['refresh_token'], 'client_id': ['placeholder-client'],
        'client_secret': ['placeholder-secret'], 'refresh_token': ['placeholder-refresh']}
    assert requests[1].headers['Authorization'] == 'Bearer placeholder-access'
    payload = json.loads(requests[1].content)
    assert set(payload) == {'message'} and set(payload['message']) == {'raw'}
    msg = BytesParser(policy=policy.default).parsebytes(base64.urlsafe_b64decode(payload['message']['raw']))
    assert msg['From'] == 'owner@example.com'
    assert msg['To'] == 'recipient@example.com'
    assert msg['Cc'] == 'copy@example.com'
    assert msg['Subject'] == 'A subject'
    assert msg.get_content().replace('\r\n', '\n') == 'Hello café.\nSecond line.\n'
    assert '-Aerys' not in msg.get_content()
    assert set(tool.tool_call_schema.model_json_schema()['properties']) == {'to', 'subject', 'body', 'cc'}


@pytest.mark.parametrize('endpoint', [TOKEN, DRAFT])
@pytest.mark.parametrize('failure', [401, 403, 500, 503, 302, 'timeout', 'missing', 'malformed', 'badtype'])
def test_errors(endpoint, failure, caplog):
    def handler(request):
        if str(request.url) == endpoint:
            if failure == 'timeout':
                raise httpx.ReadTimeout('private-token-response')
            if failure == 'missing':
                return httpx.Response(200, json={})
            if failure == 'malformed':
                return httpx.Response(200, text='private-token-response')
            if failure == 'badtype':
                return httpx.Response(200, json=[])
            return httpx.Response(failure, text='private-token-response', headers={'Location': 'https://example.com'})
        return httpx.Response(200, json={'access_token': 'placeholder-access'})
    tool, requests = setup(handler)
    out = call(tool)
    assert out.startswith('Not drafted:')
    assert 'Drafted:' not in out and 'sent' not in out.lower()
    assert 'private-token-response' not in out + caplog.text
    assert len(requests) == (1 if endpoint == TOKEN else 2)


@pytest.mark.parametrize('field,value', [(f, v) for f in ('to', 'cc', 'subject') for v in ('x\ry', 'x\ny')]
                         + [(f, v) for f in ('to', 'cc') for v in ('bad', 'a@', 'a b@example.com', 'a@example.com,')]
                         + [('to', ''), ('body', ''), ('body', '  '), ('body', 'x' * 100001)])
def test_invalid_input(field, value):
    tool, requests = setup()
    assert call(tool, **{field: value}).startswith('Not drafted:')
    assert requests == []


@pytest.mark.parametrize('identity,thread', [({}, 'owner:test'), ({'user_id': 'guest'}, 'owner:test'),
    ({'user_id': 'other'}, 'owner:test'), ({'user_id': OWNER}, 'kael:test')])
def test_identity(identity, thread):
    tool, requests = setup()
    assert call(tool, identity=identity, thread=thread).startswith('Not drafted:')
    assert not requests


@pytest.mark.parametrize('missing', list(VALUES))
@pytest.mark.parametrize('empty', [None, ''])
def test_arming_missing(missing, empty, caplog):
    values = dict(VALUES)
    values[missing] = False if missing == 'gmail_drafts' else empty
    if missing == 'owner_person_id':
        values[missing] = None
    with caplog.at_level('INFO'):
        assert gmail_drafts_for(settings(**values)) is None
    assert missing.upper() in caplog.text
    assert 'placeholder-secret' not in caplog.text and 'placeholder-refresh' not in caplog.text


def test_arming_and_factory(caplog):
    from aerys_v2.factory import action_tools_for
    from langchain_core.utils.function_calling import convert_to_openai_tool
    before = settings()
    assert before.gmail_drafts is False
    del before.gmail_drafts
    old = [convert_to_openai_tool(t) for t in action_tools_for(before)]
    assert old == [convert_to_openai_tool(t) for t in action_tools_for(settings())]
    configured = settings(**VALUES)
    assert isinstance(configured.gmail_drafts_client_secret, SecretStr)
    assert isinstance(configured.gmail_drafts_refresh_token, SecretStr)
    with caplog.at_level('INFO'):
        assert gmail_drafts_for(configured) is not None
    assert 'gmail drafts armed' in caplog.text
    assert 'draft_in_his_gmail' in [t.name for t in action_tools_for(configured)]
    assert 'draft_in_his_gmail' not in [t.name for t in action_tools_for(configured, guest=True)]


def test_guard_and_executor():
    from aerys_v2.factory import build_action_graph
    from aerys_v2.outside_guard import REACH_OUT, action_executor
    from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
    from langchain_core.tools import tool
    draft, requests = setup()
    assert draft.name not in REACH_OUT
    execute = action_executor([draft])
    assert execute(SimpleNamespace(args={'tool': draft.name}))[0] is False
    assert not requests
    @tool
    def read_email() -> str:
        """Read outside text."""
        return 'outside words'
    model = GenericFakeChatModel(messages=iter([
        AIMessage(content='', tool_calls=[{'name': 'read_email', 'args': {}, 'id': 'read'}]),
        AIMessage(content='', tool_calls=[{'name': draft.name, 'args': {
            'to': 'recipient@example.com', 'subject': 'Reply', 'body': 'Hello'}, 'id': 'draft'}]),
        AIMessage(content='done')]))
    graph = build_action_graph(model, 'test', [read_email, draft])
    result = graph.invoke({'messages': [HumanMessage(content='Write a reply for me')]},
                          {'configurable': {'identity': {'user_id': OWNER}, 'thread_id': 'owner:test',
                                            'outside_text': {}}})
    assert len(requests) == 2
    assert any(isinstance(m, ToolMessage) and str(m.content).startswith('Drafted:') for m in result['messages'])


@pytest.mark.parametrize('address', ['bad', 'one@example.com,two@example.com', 'owner@example.com\r\nBcc: other@example.com'])
def test_invalid_from_not_armed(address, caplog):
    with caplog.at_level('INFO'):
        assert gmail_drafts_for(settings(**{**VALUES, 'gmail_drafts_from': address})) is None
    assert 'GMAIL_DRAFTS_FROM is invalid' in caplog.text
    assert address not in caplog.text


@pytest.mark.parametrize('endpoint', [TOKEN, DRAFT])
@pytest.mark.parametrize('value', ['', ' ', 123, False])
def test_invalid_success_marker(endpoint, value):
    def handler(request):
        key = 'access_token' if str(request.url) == TOKEN else 'id'
        return httpx.Response(200, json={key: value if str(request.url) == endpoint else 'placeholder'})
    tool, requests = setup(handler)
    assert call(tool).startswith('Not drafted:')
    assert len(requests) == (1 if endpoint == TOKEN else 2)
