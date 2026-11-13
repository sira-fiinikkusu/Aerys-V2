from types import SimpleNamespace

import pytest

from aerys_v2.tools.cart_link import build_cart_link, build_cart_link_tool

ASIN = 'B000TEST01'
AMAZON = f'https://www.amazon.com/gp/aws/cart/add.html?ASIN.1={ASIN}&Quantity.1=1'
WALMART = 'https://www.walmart.com/sc/cart/addToCart?items=12345_1'


@pytest.mark.parametrize('product', [ASIN, ASIN.lower(), '123456789X'] + [
    f'https://{host}/{path}/{ASIN}?ignored=yes#fragment'
    for host in ('amazon.com', 'www.amazon.com', 'smile.amazon.com')
    for path in ('dp', 'slug/dp', 'gp/product', 'gp/aw/d')])
def test_amazon(product):
    expected = '123456789X' if product == '123456789X' else ASIN
    assert build_cart_link('amazon', [{'product': product}]) == AMAZON.replace(ASIN, expected)


@pytest.mark.parametrize('product', ['12345'] + [
    f'https://{host}/ip/{path}12345?q=anything#fragment'
    for host in ('walmart.com', 'www.walmart.com') for path in ('', 'slug/')])
def test_walmart(product):
    assert build_cart_link('walmart', [{'product': product}]) == WALMART


@pytest.mark.parametrize('store,product,expected', [('amazon', ASIN, AMAZON + '&ASIN.2=B000TEST01&Quantity.2=20'),
    ('walmart', '12345', WALMART + ',12345_20')])
def test_multi(store, product, expected):
    assert build_cart_link(store, [{'product': product}, {'product': product, 'quantity': 20}]) == expected


@pytest.mark.parametrize('store,product', [('amazon', p) for p in (
    'https://evil.test/dp/B000TEST01', 'https://amazon.com.evil.test/dp/B000TEST01',
    'https://amazon.com/s?k=test', 'https://amazon.com/category/test', 'bad',
    'https://walmart.com/ip/12345', 'https://amazon.com/dp/B000TEST011',
    'https://user@amazon.com/dp/B000TEST01', 'ftp://amazon.com/dp/B000TEST01', 'https://[broken')]
    + [('walmart', p) for p in ('1234', '1234567890123456', '１２３４５', 'https://walmart.com/search?q=x',
    'https://amazon.com/dp/B000TEST01', 'https://walmart.com/ip/slug/12345/extra')])
def test_refused_item(store, product):
    result = build_cart_link(store, [{'product': ASIN if store == 'amazon' else '12345'}, {'product': product}])
    assert result.startswith('Refused: item 2:')


@pytest.mark.parametrize('host', ['a.co', 'amzn.to', 'walmrt.us'])
def test_short(host):
    assert 'open it and give me the full product page' in build_cart_link('amazon', [{'product': f'https://{host}/abc'}])


@pytest.mark.parametrize('quantity', [0, 21, -1, True, 1.5, '2', None])
def test_quantity(quantity):
    assert 'item 1: quantity' in build_cart_link('amazon', [{'product': ASIN, 'quantity': quantity}])


@pytest.mark.parametrize('items', [[], [{}] * 11, None, {}, [None], [{}]])
def test_bad_items(items):
    assert build_cart_link('amazon', items).startswith('Refused:')


def test_store():
    assert build_cart_link('other', [{'product': ASIN}]).startswith('Refused:')


@pytest.mark.parametrize('store,product,expected', [
    ('amazon', 'https://amazon.com/https:%2F%2Fevil.test&ASIN.2=BAD/dp/B000TEST01?ASIN.2=BAD&Quantity.1=99#https://evil.test', AMAZON),
    ('walmart', 'https://walmart.com/ip/https:%2F%2Fevil.test&ASIN.2=BAD/12345?items=99999_20#https://evil.test', WALMART)])
def test_injection(store, product, expected):
    assert build_cart_link(store, [{'product': product}]) == expected


def settings(**kw):
    from aerys_v2.config import Settings
    kw.setdefault('tavily_api_key', 'placeholder')
    return Settings(_env_file=None, anthropic_api_key='placeholder', owner_person_id='1', **kw)


def invoke(tool, identity, flag=False, product=ASIN, thread='person:1'):
    runtime = SimpleNamespace(config={'configurable': {'identity': identity, 'thread_id': thread}})
    return tool.func('amazon', [{'product': product}], runtime, flag)


@pytest.mark.parametrize('identity,flag,count', [
    ({'user_id': '1', 'voice': True, 'speaker': {'id': 'owner'}}, False, 1),
    ({'user_id': '1', 'surface': 'lens'}, False, 1),
    ({'user_id': '1', 'platform': 'discord'}, False, 0),
    ({'user_id': '1', 'platform': 'discord'}, True, 1),
    ({'user_id': '2', 'voice': True, 'speaker': {'id': 'member'}}, True, 0),
    ({'user_id': 'guest'}, True, 0),
    ({'user_id': '1', 'voice': True}, True, 0),
    ({'user_id': '1', 'voice': True, 'speaker': {'id': 'owner', 'unsure': True}}, True, 0),
    ({'user_id': '1', 'voice': True, 'speaker': {'id': 'owner', 'demoted': True}}, True, 0),
])
def test_delivery(identity, flag, count):
    sent = []
    tool = build_cart_link_tool(settings(discord_bot_token='placeholder', approvals_owner_discord_id='123'),
                                send=lambda *args: sent.append(args))
    result = invoke(tool, identity, flag, product='https://amazon.com/evil/dp/B000TEST01?ASIN.2=BAD#evil')
    assert len(sent) == count
    assert AMAZON in result
    if count:
        assert sent[0][1:] == ('123', f'Add to your Amazon cart (1 item):\n{AMAZON}\n-Aerys')
        assert 'Sent the link to his Discord DM.' in result
    elif flag:
        assert 'not sent' in result.lower()


@pytest.mark.parametrize('mode', ['raises', 'no_token', 'no_sender', 'no_owner_id'])
def test_delivery_unavailable(mode):
    calls = []
    def send(*args):
        calls.append(args)
        raise RuntimeError('private error')
    opts = {'discord_bot_token': 'placeholder', 'approvals_owner_discord_id': '123'}
    if mode == 'no_token':
        opts.pop('discord_bot_token')
    if mode == 'no_owner_id':
        opts['approvals_owner_discord_id'] = 'invalid'
    tool = build_cart_link_tool(settings(**opts), send=None if mode == 'no_sender' else send)
    result = invoke(tool, {'user_id': '1', 'surface': 'lens'})
    assert 'not sent' in result.lower() and AMAZON in result
    assert len(calls) == (1 if mode == 'raises' else 0)
    assert 'private error' not in result


def test_factory_and_prompt():
    from aerys_v2.factory import action_tools_for, action_overlay_for
    s = settings()
    assert 'add_to_cart_link' in {t.name for t in action_tools_for(s)}
    assert 'add_to_cart_link' not in {t.name for t in action_tools_for(s, guest=True)}
    assert 'add_to_cart_link' in action_overlay_for(s)
    assert 'add_to_cart_link' not in action_overlay_for(s, guest=True)


def test_toolnode_runtime_and_schema():
    from langchain_core.messages import AIMessage
    from langgraph.graph import END, START, MessagesState, StateGraph
    from langgraph.prebuilt import ToolNode
    sent = []
    tool = build_cart_link_tool(settings(discord_bot_token='placeholder', approvals_owner_discord_id='123'),
                                send=lambda *args: sent.append(args))
    assert set(tool.tool_call_schema.model_json_schema()['properties']) == {'store', 'items', 'send_to_his_discord', 'kind'}
    builder = StateGraph(MessagesState)
    builder.add_node('tools', ToolNode([tool]))
    builder.add_edge(START, 'tools')
    builder.add_edge('tools', END)
    node = builder.compile()
    def run(args):
        return node.invoke({'messages': [AIMessage(content='', tool_calls=[
            {'name': 'add_to_cart_link', 'args': args, 'id': 'cart-1', 'type': 'tool_call'}])]},
            config={'configurable': {'identity': {'user_id': '1', 'surface': 'lens'},
                                     'thread_id': 'person:1'}})['messages'][-1].content
    assert tool.tool_call_schema.model_json_schema()['properties']['store'].get('enum') == ['amazon', 'walmart']
    result = run({'store': 'Walmart', 'items': [{'product': '12345'}, {'product': '67890', 'quantity': 2}]})
    link = WALMART + ',67890_2'
    assert link in result
    assert sent[0][1:] == ('123', f'Add to your Walmart cart (2 items):\n{link}\n-Aerys')
    for bad in ({'product': ASIN, 'quantity': True}, {'product': ASIN, 'quantity': '2'}, {'product': 123}):
        result = run({'store': 'amazon', 'items': [{'product': ASIN}, bad]})
        assert result.startswith('Refused: item 2:')
    assert len(sent) == 1


def test_destination_is_resolved_each_send(monkeypatch):
    import aerys_v2.tools.cart_link as module
    destinations = iter([123, 456])
    monkeypatch.setattr(module, 'owner_id_for', lambda s: lambda: next(destinations))
    sent = []
    tool = build_cart_link_tool(settings(discord_bot_token='placeholder'), send=lambda *args: sent.append(args))
    for _ in range(2):
        invoke(tool, {'user_id': '1', 'surface': 'lens'})
    assert [args[1] for args in sent] == ['123', '456']


def test_factory_sender_wiring(monkeypatch):
    import aerys_v2.kael_line as line
    from aerys_v2.factory import action_tools_for
    sent = []
    monkeypatch.setattr(line, '_discord_dm_send', lambda *args: sent.append(args))
    for enabled in (False, True):
        s = settings(discord_bot_token='placeholder' if enabled else None, approvals_owner_discord_id='123')
        tool = next(t for t in action_tools_for(s) if t.name == 'add_to_cart_link')
        result = invoke(tool, {'user_id': '1', 'surface': 'lens'})
        assert AMAZON in result
        assert ('Sent the link' in result) is enabled
    assert len(sent) == 1


def test_kael_line_never_sends():
    sent = []
    tool = build_cart_link_tool(settings(discord_bot_token='placeholder', approvals_owner_discord_id='123'),
                                send=lambda *args: sent.append(args))
    result = invoke(tool, {'user_id': '1', 'surface': 'lens'}, True, thread='kael:aerys-line')
    assert AMAZON in result and 'not sent' in result.lower()
    assert sent == []


@pytest.mark.parametrize('text', ['add paper towels to my amazon cart', 'put dish soap in my cart',
                                  'stick two of those in the cart at walmart'])
def test_router_fallback_sends_cart_asks_to_action(text):
    from aerys_v2.router import fallback_decision
    assert fallback_decision(text).route == 'action'


def test_not_armed_without_web_tools():
    from aerys_v2.factory import action_tools_for, action_overlay_for
    s = settings(tavily_api_key=None)
    assert 'add_to_cart_link' not in {t.name for t in action_tools_for(s)}
    assert 'add_to_cart_link' not in action_overlay_for(s)


@pytest.mark.parametrize('text', ['I loved the cartoon yesterday', 'I left my cartridge upstairs',
                                  'The Amazon rainforest is beautiful', 'put the golf cart in the garage'])
def test_cart_predicate_ignores_lookalikes(text):
    from aerys_v2.router import plausibly_fills_a_cart, parse_route_reply
    assert not plausibly_fills_a_cart(text)
    assert parse_route_reply('{"route":"chat","unaddressed":true}', text).unaddressed


@pytest.mark.parametrize('product', [
    'https://amazon.com/gp/product/B000TEST01/../../s',
    'https://amazon.com/dp/B000TEST01/%2e%2e/%2e%2e/s',
    'https://amazon.com/a/b/dp/B000TEST01',
    'https://amazon.com/dp/B000TEST01/ref/extra',
    'B000TEß01',
])
def test_amazon_non_product_paths_refused(product):
    assert build_cart_link('amazon', [{'product': product}]).startswith('Refused: item 1:')


def test_amazon_ref_suffix_still_accepted():
    assert build_cart_link('amazon', [{'product': 'https://www.amazon.com/Some-Thing/dp/B000TEST01/ref=sr_1_1?k=x'}]) == AMAZON


def test_walmart_dot_segments_refused():
    assert build_cart_link('walmart', [{'product': 'https://walmart.com/ip/../12345'}]).startswith('Refused: item 1:')


def test_product_page_links_are_canonical():
    assert build_cart_link('amazon', [{'product': 'https://www.amazon.com/x/dp/B000TEST01/ref=a?k=https://evil.example'},
                                      {'product': 'b000test02', 'quantity': 3}], 'product_page') == (
        'https://www.amazon.com/dp/B000TEST01\nhttps://www.amazon.com/dp/B000TEST02')
    assert build_cart_link('walmart', [{'product': 'https://www.walmart.com/ip/Slug/12345?x=1#y'}],
                           'product_page') == 'https://www.walmart.com/ip/12345'
    assert build_cart_link('amazon', [{'product': ASIN}], 'checkout').startswith('Refused:')
    assert build_cart_link('amazon', [{'product': 'https://evil.example/dp/B000TEST01'}], 'product_page').startswith('Refused: item 1:')


def test_product_page_dm_on_voice():
    sent = []
    tool = build_cart_link_tool(settings(discord_bot_token='placeholder', approvals_owner_discord_id='123'),
                                send=lambda *args: sent.append(args))
    runtime = SimpleNamespace(config={'configurable': {
        'identity': {'user_id': '1', 'voice': True, 'speaker': {'id': 'owner'}}, 'thread_id': 'person:1'}})
    result = tool.func('amazon', [{'product': ASIN}], runtime, False, 'product_page')
    assert sent[0][1:] == ('123', f'For your Amazon cart, tap Add to Cart on it:\nhttps://www.amazon.com/dp/{ASIN}\n-Aerys')
    assert 'Sent the link to his Discord DM.' in result and 'add.html' not in result


def test_store_name_is_case_insensitive():
    assert build_cart_link(' Walmart ', [{'product': '12345'}]) == WALMART
    assert build_cart_link('AMAZON', [{'product': ASIN}]) == AMAZON
