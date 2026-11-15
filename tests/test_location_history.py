"""D57 live location reads; all identities and places are placeholders."""
import json
from datetime import timedelta

import httpx
import pytest

from test_ha_visibility import HIS_DM, NOW, OWNER, rt
from aerys_v2.services.location_guard import LOCATION_TURN, RECENT_LOCATION_TURNS, location_memory_blocked
from aerys_v2.tools.home_control import build_home_control_tool
from aerys_v2.welcome import WELCOME_PROMPT

ENTITY = "sensor.example_phone_geocoded_location"
ADDRESS = "1 Example St"
ATTRS = {"latitude": 40.123456, "longitude": -75.123456, "gps_accuracy": 5, "address": ADDRESS}

@pytest.fixture(autouse=True)
def marker():
    token = LOCATION_TURN.set(set())
    old = RECENT_LOCATION_TURNS.copy()
    yield
    LOCATION_TURN.reset(token)
    RECENT_LOCATION_TURNS.clear()
    RECENT_LOCATION_TURNS.update(old)


def make_tool(calls, rows=None, *, prefix=False):
    def handle(request):
        calls.append(request)
        if "/history/" in request.url.path:
            return httpx.Response(200, json=[rows if rows is not None else [
                {"state": ADDRESS, "last_changed": "2026-01-10T12:00:00Z", "attributes": ATTRS},
                {"state": "2 Example Street", "last_changed": "2026-01-10T12:30:00Z", "attributes": ATTRS}]])
        return httpx.Response(200, json={"state": ADDRESS, "attributes": ATTRS})
    return build_home_control_tool(base_url="http://ha", token="example", canary_entities=frozenset(),
                                  owner_person_id=OWNER, clock=lambda: NOW,
                                  his_places=frozenset() if prefix else frozenset({ENTITY, "person.owner", "device_tracker.example_phone", "sensor.owner_location_coords", "sensor.owner_coords"}),
                                  private_entities=("sensor.example_", "person.owner", "device_tracker.example_", "sensor.owner_") if prefix else (),
                                  client=httpx.Client(transport=httpx.MockTransport(handle)))


@pytest.mark.parametrize("identity,thread", [
    (HIS_DM, "person:owner"),
    ({**HIS_DM, "platform": "wake", "unattended": True}, "wake:example"),
    ({**HIS_DM, "voice": True, "speaker": {"id": "owner-voice"}}, "voice:example"),
    ({**HIS_DM, "surface": "lens"}, "glasses:example"),
])
@pytest.mark.parametrize("entity", [ENTITY, "person.owner", "device_tracker.example_phone", "sensor.owner_location_coords"])
@pytest.mark.parametrize("prefix", [False, True])
def test_owner_live_and_history(identity, thread, entity, prefix):
    calls = []
    tool = make_tool(calls, prefix=prefix)
    state = json.loads(tool.func(operation="get_state", entity_id=entity, runtime=rt(identity, thread)))
    assert state["state"] == ADDRESS
    assert state["attributes"] == ATTRS
    trail = json.loads(tool.func(operation="history", entity_id=entity, runtime=rt(identity, thread)))
    assert trail["changes"] == [["7:00 AM", ADDRESS, "40.1235, -75.1235"],
                                ["7:30 AM", "2 Example Street", "40.1235, -75.1235"]]
    url = next(r.url for r in calls if "/history/" in r.url.path)
    assert not any(k in url.params for k in ("minimal_response", "no_attributes", "significant_changes_only"))
    assert "location" in LOCATION_TURN.get()
    assert OWNER in RECENT_LOCATION_TURNS
    assert location_memory_blocked(f"Owner was at {ADDRESS} today")


@pytest.mark.parametrize("identity,thread,reason", [
    ({**HIS_DM, "user_id": "guest"}, "person:guest", "only tell him"),
    ({**HIS_DM, "voice": True, "voice_owner_assumed": True}, "voice:example", "not sure"),
    ({**HIS_DM, "privacy_context": "public", "channel_kind": "guild"}, "guild:example", "others"),
    ({}, "", "only tell him"),
])
@pytest.mark.parametrize("op", ["get_state", "history"])
@pytest.mark.parametrize("entity", [ENTITY, "person.owner", "device_tracker.example_phone", "sensor.owner_location_coords"])
@pytest.mark.parametrize("prefix", [False, True])
def test_refuses_before_ha(identity, thread, reason, op, entity, prefix):
    calls = []
    out = make_tool(calls, prefix=prefix).func(operation=op, entity_id=entity, runtime=rt(identity, thread))
    assert reason in out
    assert calls == []


def test_thinning_preserves_stops_and_uses_gps_update_times():
    rows = [{"state": "home" if n < 250 else "away", "last_changed": NOW.isoformat(),
             "last_updated": (NOW + timedelta(minutes=n)).isoformat(),
             "attributes": {"latitude": 40 + n / 10000, "longitude": -75}} for n in range(500)]
    out = json.loads(make_tool([], rows).func(operation="history", entity_id=ENTITY, runtime=rt(HIS_DM)))
    assert len(out["changes"]) == 200
    assert out["changes"][0] == ["8:00 AM", "home", "40.0000, -75.0000"]
    assert ["12:10 PM", "away", "40.0250, -75.0000"] in out["changes"]
    assert out["changes"][-1][-1] == "40.0499, -75.0000"


def test_too_many_stops_requests_shorter_span():
    rows = [{"state": f"{n} Example Street", "last_changed": NOW.isoformat()} for n in range(201)]
    out = make_tool([], rows).func(operation="history", entity_id=ENTITY, runtime=rt(HIS_DM))
    assert "shorter" in out


def test_non_location_reads_unchanged():
    from test_ha_visibility import tool, read
    assert read(tool(), "sensor.office_temperature")["state"] == "72"
    out = json.loads(make_tool([], [{"state": "on", "last_changed": NOW.isoformat()}]).func(
        operation="history", entity_id="light.example", runtime=rt(HIS_DM)))
    assert out["changes"] == [["8:00 AM", "on"]]


def test_welcome_teaches_live_history():
    assert "may check where he's been" in WELCOME_PROMPT
    assert "phone's location history in Home Assistant" in WELCOME_PROMPT
    assert "never guess where he was" not in WELCOME_PROMPT


@pytest.mark.parametrize("op", ["get_state", "history"])
def test_mixed_target_refusal_is_atomic(op):
    calls = []
    out = make_tool(calls).func(operation=op, entity_id=f"light.example,{ENTITY}",
                               runtime=rt({**HIS_DM, "user_id": "guest"}))
    assert "only tell him" in out and not calls


def test_location_answer_stays_bounded_without_silently_losing_stops():
    from aerys_v2.tools.home_control import READ_MAX
    rows = [{"state": f"{n} " + "Example Street " * 100, "last_changed": NOW.isoformat()}
            for n in range(20)]
    out = make_tool([], rows).func(operation="history", entity_id=ENTITY, runtime=rt(HIS_DM))
    assert len(out) <= READ_MAX and "shorter" in out


def test_history_marks_addresses_in_attributes_before_thinning():
    rows = [{"state": "away", "last_changed": NOW.isoformat(), "attributes": ATTRS}]
    make_tool([], rows).func(operation="history", entity_id="person.owner", runtime=rt(HIS_DM))
    assert location_memory_blocked(f"Owner was at {ADDRESS} today")


@pytest.mark.parametrize("row", [None, {}, {"state": "away", "last_updated": "bad"}])
def test_malformed_location_history_returns_honest_error(row):
    out = make_tool([], [row]).func(operation="history", entity_id=ENTITY, runtime=rt(HIS_DM))
    assert "unreadable history" in out


@pytest.mark.parametrize("identity", [HIS_DM, {**HIS_DM, "user_id": "guest"}, {}])
@pytest.mark.parametrize("entity", ["person.other", "device_tracker.other", "zone.home"])
def test_other_whereabouts_keep_old_behavior(identity, entity):
    calls = []
    tool = make_tool(calls)
    state = json.loads(tool.func(operation="get_state", entity_id=entity, runtime=rt(identity)))
    assert state["state"] == ADDRESS and "attributes" not in state
    calls.clear()
    assert tool.func(operation="history", entity_id=entity, runtime=rt(identity)) == (
        f"No history for {entity}: where someone's been is never read back as a trail.")
    assert not calls


@pytest.mark.parametrize("identity", [HIS_DM, {**HIS_DM, "user_id": "guest"}])
def test_other_precise_places_keep_old_refusal(identity):
    calls = []
    for op in ("get_state", "history"):
        out = make_tool(calls).func(operation=op, entity_id="sensor.other_location", runtime=rt(identity))
        assert "where_is_chris_nearby" in out
    assert not calls


@pytest.mark.parametrize("state", ["home", "not_home", "away", "unknown", "unavailable", "5", "12345", True])
def test_memory_mark_only_place_values(state):
    rows = [{"state": state, "last_changed": NOW.isoformat(), "attributes": {
        **ATTRS, "friendly_name": "Example phone", "enabled": True, "zone": "ab", "postal": "1234"}}]
    make_tool([], rows).func(operation="history", entity_id="person.owner", runtime=rt(HIS_DM))
    assert location_memory_blocked("Chris was at 1 Example St")
    assert not location_memory_blocked("I'm home tonight at 5")
    assert not location_memory_blocked("Example phone is True and away or unknown or unavailable or not_home with ab and 12345")
    assert "place:1234" in LOCATION_TURN.get()
    assert "place:40.1235, -75.1235" in LOCATION_TURN.get()


@pytest.mark.parametrize("op", ["get_state", "history"])
def test_explicit_coords_entity_does_not_require_location_name(op):
    calls = []
    tool = make_tool(calls)
    out = json.loads(tool.func(operation=op, entity_id="sensor.owner_coords", runtime=rt(HIS_DM)))
    assert out["attributes"] == ATTRS if op == "get_state" else out["changes"]
    calls.clear()
    out = tool.func(operation=op, entity_id="sensor.owner_coords", runtime=rt({**HIS_DM, "user_id": "guest"}))
    assert "only tell him" in out and not calls


@pytest.mark.parametrize("identity,thread,reason", [
    ({**HIS_DM, "voice": True, "voice_owner_assumed": True}, "voice:example", "not sure"),
    ({**HIS_DM, "sealed_room": True}, "person:owner", "others"),
    (HIS_DM, "person:owner", None),
])
def test_nearby_uses_private_gate(identity, thread, reason):
    from aerys_v2.tools.whereabouts import build_whereabouts_tool
    calls = []
    def handle(request):
        calls.append(request)
        return httpx.Response(200, json={"state": "home", "attributes": {"locality": "Example City"}})
    tool = build_whereabouts_tool(base_url="http://ha", token="example", person_entity="person.owner",
        geocoded_entity=ENTITY, owner_person_id=OWNER,
        client=httpx.Client(transport=httpx.MockTransport(handle)))
    out = tool.func(config={"configurable": {
        "identity": identity, "thread_id": thread}})
    if reason:
        assert reason in out
        assert not calls
    else:
        assert "Chris is home" in out and calls


@pytest.mark.parametrize("identity", [HIS_DM, {**HIS_DM, "user_id": "guest"}])
@pytest.mark.parametrize("prefix", [False, True])
def test_search_location_privacy(identity, prefix):
    from aerys_v2.tools.home_control import build_search_entities_tool
    states = [{"entity_id": entity, "state": value, "attributes": {"friendly_name": "Location query"}}
              for entity, value in [(ENTITY, ADDRESS), ("sensor.other_geocoded_location", "12 Oak Street"),
                                    ("sensor.office_temperature", "72")]]
    tool = build_search_entities_tool(base_url="http://ha", token="example", owner_person_id=OWNER,
        his_places=frozenset() if prefix else frozenset({ENTITY}),
        private_entities=("sensor.example_",) if prefix else (),
        client=httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200, json=states))))
    out = tool.func(query="location", runtime=rt(identity))
    assert (ADDRESS if identity == HIS_DM else "(his location: not here)") in out
    if identity != HIS_DM:
        assert ADDRESS not in out
    assert "(a place: not shown)" in out and "12 Oak Street" not in out
    assert "72" in out


def test_history_memory_fence():
    make_tool([]).func(operation="history", entity_id=ENTITY, runtime=rt(HIS_DM))
    assert location_memory_blocked("Chris visited Example St today")
    assert location_memory_blocked("Chris visited 1 Example St until Nov 15, 2026")
    assert not location_memory_blocked("Chris went to the county fair today")
    assert not location_memory_blocked("Trip to Tampa until Nov 20, 2026")
    assert location_memory_blocked("Trip to 40.1234, -75.1234 until Nov 20, 2026")


def test_extraction_address_without_turn_marker():
    from aerys_v2.workers.extraction import parse_observations
    token = LOCATION_TURN.set(None)
    try:
        observations = [{"key_label": "activity", "value_text": text} for text in
                        ("Chris was at 12 Oak Street", "Chris went to the fair")]
        assert parse_observations(json.dumps(observations)) == [observations[1]]
    finally:
        LOCATION_TURN.reset(token)


def test_malformed_history_does_not_echo_raw_field():
    out = make_tool([], [{"state": ADDRESS, "last_updated": "raw-private-field"}]).func(
        operation="history", entity_id=ENTITY, runtime=rt(HIS_DM))
    assert out == f"Home Assistant returned an unreadable history for {ENTITY}."


@pytest.mark.parametrize("suffix", "St Street Ave Avenue Rd Road Blvd Dr Drive Ln Lane Way Ct Court Pl Place Hwy Highway Pkwy Cir Ter Trl".split())
def test_address_shape_suffixes(suffix):
    # Capitalized, as an address is written (HA's geocoder, her own words); the short forms count where
    # an address ends them. Lowercase "a 2 hour drive" / "3 hours on the road" stay memories.
    from aerys_v2.services.location_guard import address_shaped
    assert address_shaped(f"Visited 12 Example {suffix}.")
    assert address_shaped(f"Visited 12 Example {suffix}, then home")
    assert not address_shaped(f"Visited 12 example things on the {suffix.lower()} today") or suffix.lower() in ()
    assert address_shaped(f"12 North Old Example Hill {suffix.upper()}")
    assert not address_shaped("Chris went to the county fair")


def test_dated_marked_city_stays_allowed():
    from aerys_v2.services.location_guard import mark_location_turn
    mark_location_turn(["Tampa"])
    assert not location_memory_blocked("Trip to Tampa until Nov 20, 2026")
    assert location_memory_blocked("Chris visited Tampa today")


def test_address_shape_blocks_addresses_but_not_titles_saints_or_the_way_home():
    # Kael 11/15: the extractor drops street addresses, never a memory like a doctor's appointment.
    from aerys_v2.services.location_guard import address_shaped
    for text in ("1 Example St", "123 Main St, Springfield", "12 Oak Street in Tampa", "4200 Gulf Coast Blvd",
                 "10 Park Pl.", "88 Pine Way", "Chris visited 1 Example St today", "1 Example St Apt 4",
                 "Chris visited 1 Example St. He enjoyed the visit."):
        assert address_shaped(text), text
    for text in ("Chris has 2 appointments with Dr Smith next week", "Chris has 2 appointments with Dr. Smith next week",
                 "3 days in St Pete", "3 days in St. Pete", "Chris was 2 hours on the way home", "a 2 hour drive to Tampa",
                 "went to the county fair", "Sam likes 2 sugars"):
        assert not address_shaped(text), text


def test_the_street_without_its_number_is_never_excused_by_a_date():
    # Codex 11/15 round 3: "Chris visited Example St until Nov 15, 2026" passed after a read of "1 Example St".
    from aerys_v2.services.location_guard import LOCATION_TURN, location_memory_blocked, mark_location_turn
    token = LOCATION_TURN.set(set())
    try:
        mark_location_turn(values=["1 Example St"], person_id=None)
        for text in ("Chris visited Example St until Nov 15, 2026", "Chris visited 1 Example St until Nov 15, 2026",
                     "Chris visited Example St today"):
            assert location_memory_blocked(text), text
        assert not location_memory_blocked("Trip to Tampa until Nov 20, 2026")
        assert not location_memory_blocked("Chris went to the fair")
    finally:
        LOCATION_TURN.reset(token)
