"""Phase A privacy regression coverage; all location fixtures are synthetic."""
import json

import httpx
import pytest

from aerys_v2.config import Settings
from aerys_v2 import factory
from aerys_v2.workers.extraction import EXTRACTION_SYSTEM_PROMPT, parse_observations

OWNER = "synthetic-owner"
CITY = {"locality": "Springfield", "administrative_area": "Illinois", "country": "USA"}
ATTRS = dict(CITY, thoroughfare="Example Street", sub_locality="Example Quarter",
             sub_thoroughfare="FORBIDDEN_HOUSE", name="FORBIDDEN_NAME",
             location="FORBIDDEN_LOCATION", postal_code="FORBIDDEN_POSTAL",
             premises="FORBIDDEN_PREMISES", phone="FORBIDDEN_PHONE", url="FORBIDDEN_URL",
             iso_country_code="FORBIDDEN_ISO", sub_administrative_area="FORBIDDEN_COUNTY")
PRIVATE = {"user_id": OWNER, "privacy_context": "private"}


@pytest.fixture(autouse=True)
def isolated_location_context(monkeypatch):
    from aerys_v2.services import location_guard as guard
    token = guard.LOCATION_TURN.set(set())
    monkeypatch.setattr(guard, 'RECENT_LOCATION_TURNS', {})
    monkeypatch.setattr(factory, '_OWNER_ID', None)
    yield
    guard.LOCATION_TURN.reset(token)


def settings(**kw):
    return Settings(_env_file=None, anthropic_api_key="synthetic", **kw)


def test_city_one_call_filtered_and_home_hidden(monkeypatch):
    calls = []
    state = "not_home"
    def handler(request):
        calls.append(json.loads(request.content)["template"])
        return httpx.Response(200, json={"rooms": [], "person": state, "room": "", "city": ATTRS, "geocoded_state": "synthetic"})
    real = httpx.Client
    monkeypatch.setattr(httpx, "Client", lambda **kw: real(transport=httpx.MockTransport(handler), **kw))
    fn = factory.presence_fn_for(settings(ha_token="synthetic", presence_context=True,
        owner_person_id=OWNER, presence_geocoded_entity="sensor.synthetic_geocoded"))
    try:
        snapshot = fn()
        assert snapshot.city == CITY
        assert snapshot.whereabouts == ("not_home", "")
        text = factory.presence_block(PRIVATE, lambda: snapshot)
        assert "in Springfield, Illinois right now" in text
        assert "never save" in text and "weather" in text and "local time" in text
        assert "FORBIDDEN" not in text
        assert len(calls) == 1
        assert "state_attr('sensor.synthetic_geocoded', 'locality')" in calls[0]
        assert "!= 'home'" in calls[0]
        assert "sub_thoroughfare" not in calls[0]
        assert "Springfield" not in factory.presence_block(dict(PRIVATE, voice=True), lambda: snapshot)
        for identity in [dict(PRIVATE, privacy_context="public"), dict(PRIVATE, user_id="guest"), {"user_id": OWNER}]:
            assert factory.presence_block(identity, lambda: pytest.fail("must not read HA")) == ""
        state = "home"
        assert fn().city == {}
    finally:
        factory.set_owner_id(None)


def make_tool(handler, owner=OWNER):
    from aerys_v2.tools.whereabouts import build_whereabouts_tool
    return build_whereabouts_tool(base_url="http://ha.test", token="synthetic",
        person_entity="person.synthetic", geocoded_entity="sensor.synthetic_geocoded",
        owner_person_id=owner, client=httpx.Client(transport=httpx.MockTransport(handler)))


@pytest.mark.parametrize("identity", [dict(PRIVATE, privacy_context="public"), dict(PRIVATE, user_id="guest"),
    {"user_id": OWNER}, dict(PRIVATE, privacy_context="unknown"), {}])
def test_nearby_gate_no_ha(identity):
    tool = make_tool(lambda request: pytest.fail("unauthorized HA call"))
    assert "refused" in tool.invoke({}, config={"configurable": {"identity": identity}}).lower()
    assert "config" not in tool.tool_call_schema.model_json_schema().get("properties", {})


@pytest.mark.parametrize("state", ["home", "not_home"])
def test_nearby_allowlist_rounding_and_home(state):
    def handler(request):
        return httpx.Response(200, json=({"state": state, "attributes": {"latitude": 39.781234, "longitude": -89.650432}}
            if request.url.path.endswith("person.synthetic") else {"state": "FORBIDDEN_STATE", "attributes": ATTRS}))
    result = make_tool(handler).invoke({}, config={"configurable": {"identity": PRIVATE}})
    assert "Springfield" in result and "FORBIDDEN" not in result
    assert "never store" in result.lower()
    if state == "home":
        assert "home" in result and "39.781" not in result and "Example" not in result
    else:
        assert "Example Street" in result and "Example Quarter" in result and "Illinois" in result
        payload = json.JSONDecoder().raw_decode(result)[0]
        assert payload["latitude"] == "39.781" and payload["longitude"] == "-89.650"
        assert "39.781234" not in result and "-89.650432" not in result


@pytest.mark.parametrize("body,status", [(None, 500), ({"state": "unknown", "attributes": ATTRS}, 200),
    ({"state": "not_home", "attributes": {"latitude": "NaN", "longitude": 0}}, 200)])
def test_nearby_fails_closed(body, status):
    result = make_tool(lambda r: httpx.Response(status, json=body)).invoke({}, config={"configurable": {"identity": PRIVATE}})
    assert "unavailable" in result.lower() and "FORBIDDEN" not in result


@pytest.mark.parametrize("token,sensor,guest,expected", [(None,"sensor.synthetic",False,False),
    ("synthetic","",False,False), ("synthetic","sensor.synthetic",True,False), ("synthetic","sensor.synthetic",False,True)])
def test_registration(token, sensor, guest, expected):
    tools = factory.action_tools_for(settings(ha_token=token, presence_geocoded_entity=sensor), guest=guest)
    assert ("where_is_chris_nearby" in [t.name for t in tools]) == expected


def test_extraction_location_rule():
    for phrase in ["current location", "whereabouts", "coordinates", "nearby tool", "end date", "Nov 1-6, 2026"]:
        assert phrase in EXTRACTION_SYSTEM_PROMPT


@pytest.mark.parametrize("field", ["value_text", "context", "source_text"])
def test_extraction_drops_coordinates(field):
    obs = {"key_label": "event.trip", "value_text": "Conference in Springfield, Nov 1-6, 2026", field: "39.781, -89.650"}
    assert parse_observations(json.dumps([obs])) == []


def test_extraction_keeps_dated_trip():
    obs = {"key_label": "event.trip", "value_text": "Conference in Springfield, Nov 1-6, 2026"}
    assert len(parse_observations(json.dumps([obs]))) == 1


def test_nearby_without_owner_refuses():
    tool = make_tool(lambda r: pytest.fail("unconfigured owner must not read HA"), owner=None)
    assert "refused" in tool.invoke({}, config={"configurable": {"identity": PRIVATE}})


@pytest.mark.parametrize("sensor", [{"state": "unavailable", "attributes": ATTRS},
    {"state": "unknown", "attributes": ATTRS}, {"state": "synthetic", "attributes": []}])
def test_nearby_bad_geocoder_fails_closed(sensor):
    def handler(request):
        return httpx.Response(200, json=({"state": "not_home", "attributes": {"latitude": 39.781234, "longitude": -89.650432}}
            if request.url.path.endswith("person.synthetic") else sensor))
    assert "unavailable" in make_tool(handler).invoke({}, config={"configurable": {"identity": PRIVATE}})


def test_nearby_transport_exception_is_not_disclosed_or_logged(caplog):
    def handler(request):
        raise httpx.ConnectError("FORBIDDEN raw location", request=request)
    assert "unavailable" in make_tool(handler).invoke({}, config={"configurable": {"identity": PRIVATE}})
    assert "FORBIDDEN" not in caplog.text



@pytest.mark.parametrize("text", ["Prefers Python 3.11, 3.12 for new work", "Rated the two hotels 4.5, 4.8"])
def test_extraction_keeps_ordinary_decimal_lists(text):
    """Review fix (Kael, 11/02): the coordinate filter must not eat version lists or ratings."""
    from aerys_v2.workers.extraction import parse_observations
    obs = {"key_label": "preference.example", "value_text": text}
    kept = parse_observations(json.dumps([obs]))
    assert kept and kept[0]["value_text"] == text


def test_away_city_points_weather_at_the_web_not_the_house():
    from aerys_v2.services.presence import format_whereabouts
    text = " ".join(format_whereabouts("not_home", None, {"locality": "Springfield", "administrative_area": "Illinois"}))
    assert "Springfield, Illinois" in text and "HOUSE forecast" in text


@pytest.mark.parametrize('text', [
    '39.78, -89.65', '39.780 -89.00', '39.78\n-89.65',
    'lat: 39 lon: -89', 'latitude=39 longitude=-89',
    '{"latitude": 39, "longitude": -89}',
    'Chris is currently in Springfield', 'Chris is currently at Example Hall',
    'Chris is in Springfield right now', 'Chris is right now in Springfield',
    'Chris is currently staying at Example Hotel', 'Chris is near Example Street',
])
def test_shared_guard_blocks(text):
    from aerys_v2.services.location_guard import location_shaped
    assert location_shaped(text)
    for field in ('value_text', 'context', 'source_text', 'key_label'):
        obs = {'key_label': 'fact.example', 'value_text': 'Likes tea', field: text}
        assert parse_observations(json.dumps([obs])) == []


@pytest.mark.parametrize('text', [
    'Chris is at a conference in Springfield, Nov 1-6, 2026',
    'Chris is in Springfield until Nov 6, 2026',
    '3.11, 3.12', '4.5, 4.8', 'Benchmarks scored 123.456, 234.567',
])
def test_shared_guard_allows(text):
    from aerys_v2.services.location_guard import location_shaped
    assert not location_shaped(text)


@pytest.mark.parametrize('state', [None, [], {}, '', '   '])
@pytest.mark.parametrize('entity', ['person.synthetic', 'sensor.synthetic_geocoded'])
def test_malformed_states(state, entity):
    def handler(request):
        body = {'state': 'not_home', 'attributes': dict(ATTRS, latitude=39.78, longitude=-89.65)}
        if request.url.path.endswith(entity):
            body['state'] = state
        return httpx.Response(200, json=body)
    assert 'unavailable' in make_tool(handler).invoke({}, config={'configurable': {'identity': PRIVATE}})


def test_presence_exception_log_has_no_traceback(monkeypatch, caplog):
    def fail(*a, **kw):
        raise RuntimeError('FORBIDDEN location')
    monkeypatch.setattr(httpx, 'Client', fail)
    caplog.set_level('DEBUG')
    fn = factory.presence_fn_for(settings(ha_token='synthetic', presence_context=True))
    assert not fn()
    assert 'FORBIDDEN' not in caplog.text
    assert all(r.exc_info is None for r in caplog.records)


@pytest.mark.parametrize('marked', [True, False])
def test_location_marker_blocks_public_verdict(marked, monkeypatch):
    from types import SimpleNamespace
    from aerys_v2 import service
    from aerys_v2.services.location_guard import LOCATION_TURN
    from aerys_v2.services.content_privacy import CONTENT_PRIVACY_KEY
    marker = set()
    token = LOCATION_TURN.set(marker)
    class InlineThread:
        def __init__(self, target, **kw): self.target = target
        def start(self): self.target()
    monkeypatch.setattr(service.threading, 'Thread', InlineThread)
    msg = service._human_turn('Hello', 'private', 'synthetic-id')
    class Graph:
        def get_state(self, config): return SimpleNamespace(values={'messages': [msg]})
        def update_state(self, config, values, **kw):
            nonlocal msg
            msg = values['messages'][0]
    try:
        if marked:
            factory.set_owner_id(OWNER)
            block = factory.presence_block(PRIVATE, lambda: factory.PresenceSnapshot([], ('not_home', ''), CITY))
            assert 'Springfield' in block
            assert marker
        service._reclassify_content_privacy(Graph(), {'configurable': {'identity': PRIVATE if marked else {}}}, 'synthetic-id', 'Hello', 'Hi', lambda _: 'public')
        assert msg.additional_kwargs[CONTENT_PRIVACY_KEY] == ('private' if marked else 'public')
    finally:
        LOCATION_TURN.reset(token)
        factory.set_owner_id(None)


def test_remember_location_never_calls_writer():
    from aerys_v2.tools.remember import build_remember_tool
    tool = build_remember_tool(lambda _: pytest.fail('must not write'))
    result = tool.invoke({'fact': 'Chris is currently in Springfield'}, config={'configurable': {'identity': dict(PRIVATE, user_id='00000000-0000-0000-0000-000000000001')}})
    assert 'Nothing was kept' in result


def test_nearby_marker_survives_graph_tool_threads():
    from langgraph.graph import StateGraph, MessagesState, START, END
    from langgraph.prebuilt import ToolNode
    from langchain_core.messages import AIMessage
    from aerys_v2.services.location_guard import LOCATION_TURN
    from aerys_v2.turns import extract_tool_calls
    tool = make_tool(lambda r: httpx.Response(200, json={
        'state': 'not_home', 'attributes': dict(ATTRS, latitude=39.78, longitude=-89.65)}))
    builder = StateGraph(MessagesState)
    builder.add_node('tools', ToolNode([tool]))
    builder.add_edge(START, 'tools')
    builder.add_edge('tools', END)
    marker = set()
    token = LOCATION_TURN.set(marker)
    try:
        result = builder.compile().invoke({'messages': [AIMessage(content='', tool_calls=[
            {'name': tool.name, 'args': {}, 'id': 'synthetic-call'}])]},
            config={'configurable': {'identity': PRIVATE}})
        assert 'location' in marker
        assert extract_tool_calls(result['messages'])[0]['name'] == 'where_is_chris_nearby'
    finally:
        LOCATION_TURN.reset(token)


def test_direct_memory_writers_refuse_before_embedding(monkeypatch):
    from unittest.mock import Mock
    from aerys_v2.transports.discord_commands import tell, correct
    from aerys_v2.services.a2a_memory import a2a_memory_writer_for
    import aerys_v2.services.context as context
    conn = Mock()
    embed = Mock(side_effect=AssertionError('must not embed'))
    fact = 'Chris is currently in Springfield'
    assert 'Nothing was kept' in tell(conn, OWNER, fact, embed, 'private')
    assert 'Nothing was kept' in correct(conn, OWNER, 'old fact', fact, embed)
    conn.execute.assert_not_called()
    cfg = settings(memories_database_url='postgresql://synthetic/synthetic',
                   owner_person_id=OWNER, embeddings_api_key='synthetic')
    a2a_memory_writer_for(cfg, embed=embed, connect=conn, synchronous=True)(
        'kael:synthetic', fact, 'This is a synthetic reply for testing')
    conn.assert_not_called()
    monkeypatch.setattr(context, 'embedder_from_settings', lambda _: embed)
    assert 'Nothing was kept' in factory.remember_writer_for(cfg)({'fact': fact})
    embed.assert_not_called()


def test_triage_direct_call_guard():
    from unittest.mock import Mock
    from aerys_v2.workers.extraction import triage_memory
    conn = Mock()
    result = triage_memory(conn, person_id=OWNER, key_label='fact.example',
        value_text='likes tea', content='likes tea', context='39.78\n-89.65',
        event_date=None, embedding='[]', source_platform='synthetic',
        privacy_level='private', created_at='2026-11-02')
    assert result == 'skipped'
    conn.execute.assert_not_called()


def test_extraction_query_excludes_nearby_provenance():
    from aerys_v2.workers.extraction import V2_TURNS_SQL
    # SQL contract only; PostgreSQL execution is verified by the deployment runner.
    assert '''AND NOT COALESCE(t.tool_calls, '[]'::jsonb) @> '[{"name":"where_is_chris_nearby"}]'::jsonb''' in V2_TURNS_SQL


def test_later_turn_city_repeat_stays_private(monkeypatch):
    import threading
    from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
    from langchain_core.messages import AIMessage
    from aerys_v2 import service
    from aerys_v2.services.content_privacy import content_privacy_of
    # Join only service's retag threads, preserving real graph worker execution.
    threads = []
    real_thread = threading.Thread
    def track_thread(*a, **kw):
        t = real_thread(*a, **kw)
        threads.append(t)
        return t
    monkeypatch.setattr(service.threading, 'Thread', track_thread)
    snapshot = factory.PresenceSnapshot([], ('not_home', ''), CITY)
    factory.set_owner_id(OWNER)
    graph = factory.build_graph(GenericFakeChatModel(messages=iter([
        AIMessage(content='Springfield'), AIMessage(content='You are in Springfield')])),
        soul='Synthetic assistant', presence_fn=lambda: snapshot)
    try:
        for text in ('first synthetic greeting', 'second synthetic greeting'):
            service.ask(graph, text, identity=PRIVATE, thread_id='person:synthetic',
                        content_privacy_classifier=lambda _: 'public')
            for t in threads:
                t.join(timeout=2)
            snapshot = factory.PresenceSnapshot([])
        humans = [m for m in graph.get_state({'configurable': {'thread_id': 'person:synthetic'}}).values['messages'] if m.type == 'human']
        assert [content_privacy_of(m) for m in humans] == ['private', 'private']
    finally:
        factory.set_owner_id(None)


@pytest.mark.parametrize("text,held", [
    ("Currently in Springfield for work", True),       # extracted memories drop the name
    ("Right now at the Springfield Hilton", True),
    ("The rewrite is currently in progress", False),   # lowercase: not a place
    ("Scored 98.75, 87.50 on the two runs", False),    # positive two-decimal pair: not coordinates
    ("Prefers Python 3.11, 3.12", False),
])
def test_guard_review_cases(text, held):
    """Kael's review of round 2 (11/02): nameless whereabouts, and no hard-coded version exemption."""
    from aerys_v2.services.location_guard import location_shaped
    assert location_shaped(text) is held


@pytest.mark.parametrize("text,held", [
    ("Chris's current city is Springfield", True),
    ("Current location: Springfield, Illinois", True),
    ("Current address: Example Street", True),
    ("Current hotel: Springfield Hilton", True),
    ("staying at Springfield Hilton", True),
    ("currently staying at Springfield Hilton", True),
    ("coordinates: 39, -89", True), ("coords: 39.7, -89.6", True),
    ("His phone says he is away from home, in Springfield", True),
    ("Currently in Python development", False),
    ("currently in charge of the migration", False),
    ("Currently in Springfield for work", True),
    ("Right now at the Springfield Hilton", True),
    ("39.78, -89.65", True), ("39.780, 12.50", True),
    ("Temperature readings were -5.25, 12.50", False),
    ("Scored 98.75, 87.50", False),
])
def test_round3_shapes(text, held):
    from aerys_v2.services.location_guard import location_shaped
    assert location_shaped(text) is held


@pytest.mark.parametrize('state', ['unavailable', 'unknown', '', None, 5])
def test_ambient_invalid_geocoder_state(monkeypatch, state):
    calls = []
    def handler(request):
        calls.append(json.loads(request.content)['template'])
        return httpx.Response(200, json={'rooms': [], 'person': 'not_home', 'room': '',
            'city': ATTRS, 'geocoded_state': state})
    real = httpx.Client
    monkeypatch.setattr(httpx, 'Client', lambda **kw: real(transport=httpx.MockTransport(handler), **kw))
    fn = factory.presence_fn_for(settings(ha_token='synthetic', presence_context=True,
        presence_geocoded_entity='sensor.synthetic_geocoded'))
    assert fn().city == {}
    assert len(calls) == 1
    assert "states('sensor.synthetic_geocoded')" in calls[0]


@pytest.mark.parametrize('fact,blocked', [
    ('Enjoys coffee in springfield', True), ('Likes Example Street', True),
    ('Visits Illinois', True), ('Likes Example Quarter', True),
    ('Likes Springfieldian art', False),
    ('Conference in Springfield, Nov 1-6, 2026', False),
    ('Trip to Springfield until Nov 6, 2026', False),
    ('Springfield on Nov 1, 2026', True),
])
def test_round3_provenance(fact, blocked):
    from aerys_v2.services.location_guard import LOCATION_TURN, mark_location_turn, location_memory_blocked
    token = LOCATION_TURN.set(set())
    try:
        mark_location_turn(values=CITY.values())
        mark_location_turn(values=['Example Street', 'Example Quarter'])
        assert location_memory_blocked(fact) is blocked
    finally:
        LOCATION_TURN.reset(token)


def test_round3_memory_entrypoints(monkeypatch):
    from unittest.mock import Mock
    from aerys_v2.services.location_guard import LOCATION_TURN, mark_location_turn, LOCATION_REFUSAL
    from aerys_v2.tools.remember import build_remember_tool
    from aerys_v2.transports.discord_commands import tell, correct
    import aerys_v2.services.context as context
    token = LOCATION_TURN.set(set())
    conn = Mock()
    embed = Mock(side_effect=AssertionError('must not embed'))
    try:
        mark_location_turn(values=['Springfield'])
        fact = 'Enjoys coffee in Springfield'
        assert build_remember_tool(embed).invoke({'fact': fact}) == LOCATION_REFUSAL
        assert tell(conn, OWNER, fact, embed, 'private') == LOCATION_REFUSAL
        assert correct(conn, OWNER, 'old fact', fact, embed) == LOCATION_REFUSAL
        monkeypatch.setattr(context, 'embedder_from_settings', lambda _: embed)
        writer = factory.remember_writer_for(settings(memories_database_url='postgresql://synthetic/synthetic', embeddings_api_key='synthetic'))
        assert writer({'fact': fact}) == LOCATION_REFUSAL
        embed.assert_not_called()
        conn.execute.assert_not_called()
    finally:
        LOCATION_TURN.reset(token)


@pytest.mark.parametrize('who,state,recent,expected', [
    (OWNER, 'home', False, 'public'), (OWNER, 'not_home', False, 'private'),
    (OWNER, None, False, 'private'), (OWNER, 'error', False, 'private'),
    (OWNER, 'home', True, 'private'), ('guest', None, True, 'public'),
])
def test_round3_away_lock(monkeypatch, who, state, recent, expected):
    from types import SimpleNamespace
    from aerys_v2 import service
    from aerys_v2.services import location_guard as guard
    from aerys_v2.services.content_privacy import content_privacy_of
    factory.set_owner_id(OWNER)
    monkeypatch.setattr(guard, 'RECENT_LOCATION_TURNS', {})
    token = guard.LOCATION_TURN.set(set())
    class InlineThread:
        def __init__(self, target, **kw): self.target = target
        def start(self): self.target()
    monkeypatch.setattr(service.threading, 'Thread', InlineThread)
    msg = service._human_turn('Where was that?', 'private', 'test-id')
    calls = []
    def presence():
        calls.append(1)
        if state == 'error': raise RuntimeError('synthetic')
        return factory.PresenceSnapshot([], (state, ''))
    class Graph:
        presence_fn = staticmethod(presence) if state else None
        def get_state(self, config): return SimpleNamespace(values={'messages': [msg]})
        def update_state(self, config, values, **kw):
            nonlocal msg
            msg = values['messages'][0]
    try:
        if recent:
            guard.mark_location_turn(values=['Springfield'], person_id=OWNER)
            guard.LOCATION_TURN.set(set())
        service._reclassify_content_privacy(Graph(), {'configurable': {'identity': {'user_id': who}}},
            'test-id', 'Where was that?', 'Springfield', lambda _: 'public')
        assert content_privacy_of(msg) == expected
        assert len(calls) <= 1
    finally:
        guard.LOCATION_TURN.reset(token)
        factory.set_owner_id(None)


def test_location_lock_expires_after_24_hours(monkeypatch):
    from aerys_v2.services import location_guard as guard
    now = 100.0
    monkeypatch.setattr(guard.time, 'monotonic', lambda: now)
    guard.mark_location_turn(values=['Springfield'], person_id=OWNER)
    assert guard.RECENT_LOCATION_TURNS == {OWNER: 100.0}
    now += 24 * 60 * 60 - 1
    assert guard.recent_location_turn(OWNER)
    now += 1
    assert not guard.recent_location_turn(OWNER)


def test_nearby_marker_records_places_only():
    from aerys_v2.services import location_guard as guard
    tool = make_tool(lambda r: httpx.Response(200, json={
        'state': 'not_home', 'attributes': dict(ATTRS, latitude=39.78, longitude=-89.65)}))
    tool.invoke({}, config={'configurable': {'identity': PRIVATE}})
    assert guard.LOCATION_TURN.get() == {'location', 'place:Springfield', 'place:Illinois',
        'place:Example Street', 'place:Example Quarter'}
    assert OWNER in guard.RECENT_LOCATION_TURNS


def test_owner_known_when_presence_disabled():
    assert factory.presence_fn_for(settings(owner_person_id=OWNER, presence_context=False)) is None
    assert factory._OWNER_ID == OWNER


def test_remember_propagates_writer_location_refusal():
    from aerys_v2.services.location_guard import LOCATION_REFUSAL
    from aerys_v2.tools.remember import build_remember_tool
    tool = build_remember_tool(lambda _: LOCATION_REFUSAL)
    assert tool.invoke({'fact': 'Likes tea'}, config={'configurable': {'identity': {
        'user_id': '00000000-0000-0000-0000-000000000001'}}}) == LOCATION_REFUSAL


def test_nearby_reads_coordinates_from_the_person_when_presence_is_a_settled_sensor():
    """Live 2026-11-02: PRESENCE_PERSON_ENTITY is a settled home/away sensor with no GPS,
    so every lookup came back unavailable. Coordinates come from a separate entity."""
    from aerys_v2.tools.whereabouts import build_whereabouts_tool
    states = {
        "sensor.synthetic_settled": {"state": "not_home", "attributes": {"friendly_name": "settled"}},
        "person.synthetic": {"state": "not_home", "attributes": {"latitude": 39.7817, "longitude": -89.6501}},
        "sensor.synthetic_geocoded": {"state": "1 Main St", "attributes": {
            "locality": "Springfield", "administrative_area": "Illinois", "thoroughfare": "Main St"}},
    }
    def handler(request):
        return httpx.Response(200, json=states[request.url.path.rsplit("/", 1)[-1]])
    tool = build_whereabouts_tool(base_url="http://ha.test", token="synthetic",
        person_entity="sensor.synthetic_settled", geocoded_entity="sensor.synthetic_geocoded",
        coords_entity="person.synthetic", owner_person_id=OWNER,
        client=httpx.Client(transport=httpx.MockTransport(handler)))
    out = tool.invoke({}, config={"configurable": {"identity": PRIVATE}})
    assert '"latitude": "39.782"' in out and '"longitude": "-89.650"' in out and "Springfield" in out


def test_geocoder_null_strings_are_dropped():
    """Live 2026-11-02: the companion app reports an empty sub_locality as the string "null"."""
    from aerys_v2.services.presence import location_fields
    assert location_fields({"locality": "Springfield", "sub_locality": "null"}, nearby=True) == {"locality": "Springfield"}
