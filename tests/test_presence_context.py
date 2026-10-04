"""Her gap #101, passive half: which rooms show occupancy, as ambient owner-only context.

The active half (speaking because someone arrived) is deliberately NOT built — each
event would be a new metered turn and a behaviour change (Chris, 2026-09-21).
"""
import json

import httpx
import pytest

from aerys_v2.config import Settings
from aerys_v2.factory import presence_block, presence_fn_for, set_owner_id
from aerys_v2.services.presence import DEFAULT_ROOMS, format_presence, parse_rooms

OWNER = "6e6bcbed-0000-0000-0000-000000000000"


@pytest.fixture(autouse=True)
def _owner():
    set_owner_id(OWNER)
    yield
    set_owner_id(None)


def test_without_a_configured_owner_the_block_stays_shut():
    set_owner_id(None)
    assert presence_block({"user_id": OWNER, "privacy_context": "private"}, lambda: ["office"]) == ""


def settings(**kw):
    return Settings(_env_file=None, anthropic_api_key="x", ha_token="ha-tok", **kw)


# --- the wording is careful on purpose ---------------------------------------

def test_the_block_reports_occupancy_and_refuses_to_name_a_person():
    text = format_presence(["office", "living room"], spoken_from="office")
    assert text.startswith("\n\n[House presence]")
    assert "Rooms showing occupancy right now: office, living room." in text
    assert "He is speaking from the office." in text
    # the limit is stated IN the block: several rooms read occupied at once, and an
    # occupancy sensor cannot tell Chris from Megan from a pet.
    assert "do not say WHO" in text and "never assert where someone is" in text
    assert "Chris is in" not in text


def test_nothing_to_say_is_silence_not_a_claim():
    assert format_presence([], None) == ""
    assert "No room is showing occupancy" in format_presence([], spoken_from="office")


def test_room_spec_parses_and_survives_typos():
    assert parse_rooms("") == DEFAULT_ROOMS and parse_rooms("   ") == DEFAULT_ROOMS
    assert parse_rooms(" Office = binary_sensor.a , broken , =x, den=binary_sensor.b ") == {
        "office": "binary_sensor.a", "den": "binary_sensor.b"}
    assert parse_rooms("all,junk,entries") == DEFAULT_ROOMS   # never an empty map


# --- the gate ------------------------------------------------------------------

def test_presence_is_off_until_armed():
    assert presence_fn_for(settings()) is None                      # flag off
    assert presence_fn_for(Settings(_env_file=None, anthropic_api_key="x",
                                    presence_context=True)) is None  # no HA token


def test_the_owner_gate_lives_in_the_block_not_the_call_site():
    """Presence disclosure is an allowlisted surface: an ambient block must not hand a
    guild member the answer the presence TOOL refuses them."""
    fn = lambda: ["office"]  # noqa: E731
    assert presence_block({"user_id": OWNER, "privacy_context": "public"}, fn) == ""
    assert presence_block({"user_id": "guild-member", "privacy_context": "private"}, fn) == ""
    assert presence_block({"privacy_context": "private"}, fn) == ""
    assert presence_block({"user_id": OWNER, "privacy_context": "private"}, fn).startswith("\n\n[House presence]")
    assert presence_block({"user_id": OWNER, "privacy_context": "private"}, None) == ""


def test_a_dark_house_costs_the_block_never_the_turn():
    def boom():
        raise OSError("HA is down")

    assert presence_block({"user_id": OWNER, "privacy_context": "private"}, boom) == ""


# --- the read ------------------------------------------------------------------

def test_the_whole_house_is_read_in_ONE_request_not_one_per_room():
    """A per-entity loop is N round trips and N timeouts — a slow-but-alive HA would
    add seconds to every turn. One template call answers all of them."""
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        assert request.headers["Authorization"] == "Bearer ha-tok"
        assert request.url.path == "/api/template" and request.method == "POST"
        tpl = json.loads(request.content)["template"]
        assert "binary_sensor.office_occupancy" in tpl and "binary_sensor.sunroom_occupancy" in tpl
        return httpx.Response(200, text='{"rooms": ["office", "living room"], "person": "home", "room": "Office"}')

    real = httpx.Client
    try:
        httpx.Client = lambda **kw: real(transport=httpx.MockTransport(handler), **kw)  # type: ignore[misc]
        fn = presence_fn_for(settings(presence_context=True))
        assert fn is not None and fn() == ["office", "living room"]
    finally:
        httpx.Client = real  # type: ignore[misc]
    assert len(calls) == 1, "one request for the whole house"


def test_an_unhappy_home_assistant_is_silence_not_a_guess():
    for resp in (httpx.Response(500, text="nope"), httpx.Response(200, text=""),
                 httpx.Response(200, text="office,living room"), httpx.Response(200, text='{"rooms": 3}')):
        real = httpx.Client
        try:
            httpx.Client = lambda **kw: real(transport=httpx.MockTransport(lambda r: resp), **kw)  # type: ignore[misc]
            fn = presence_fn_for(settings(presence_context=True))
            assert fn() == []
        finally:
            httpx.Client = real  # type: ignore[misc]


# --- his whereabouts (BLE soak, approved 2026-10-03) ---------------------------

from aerys_v2.services.presence import format_whereabouts  # noqa: E402


def test_home_and_away_are_stated_plainly():
    home = "Chris's own phone says he is home - that is him, so you can say it plainly."
    assert format_whereabouts("home", "") == [home]
    assert format_whereabouts("not_home", "Office") == ["Chris's own phone says he is away from home right now."]
    for unknown in ("", None, "unknown", "unavailable"):
        assert format_whereabouts(unknown, "Office") == [], "no person state is silence"


def test_the_office_is_likely_and_every_other_room_is_only_a_hint():
    office = format_whereabouts("home", "Office")
    assert office[1].startswith("His phone has settled in the office, so he is most likely there")
    (_, bedroom) = format_whereabouts("home", "Bedroom")
    assert "might be in the bedroom" in bedroom and "never act on it or state it as fact" in bedroom
    assert len(format_whereabouts("home", "away")) == 1, "phone not seen: home only"


def test_whereabouts_alone_make_a_block_without_an_empty_occupancy_line():
    text = format_presence([], None, whereabouts=("home", "Office"))
    assert text.startswith("\n\n[House presence]") and "says he is home" in text
    assert "No room is showing occupancy" not in text


def test_the_satellite_he_speaks_through_beats_the_lagging_phone_room():
    text = format_presence(["bedroom"], spoken_from="bedroom", whereabouts=("home", "Office"))
    assert "says he is home" in text and "He is speaking from the bedroom." in text
    assert "office" not in text.lower().replace("occupancy", "")


def test_the_one_request_also_carries_his_person_and_settled_room():
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        tpl = json.loads(request.content)["template"]
        seen.append(tpl)
        return httpx.Response(200, text='{"rooms": ["office"], "person": "home", "room": "Office"}')

    real = httpx.Client
    try:
        httpx.Client = lambda **kw: real(transport=httpx.MockTransport(handler), **kw)  # type: ignore[misc]
        fn = presence_fn_for(settings(presence_context=True))
        snapshot = fn()
    finally:
        httpx.Client = real  # type: ignore[misc]
    assert len(seen) == 1 and "states('person.chris')" in seen[0] and "states('sensor.chris_settled_room')" in seen[0]
    # Serialized by HA as JSON (rendered live 10/03: {"rooms":["office"],"person":"home","room":"Office"});
    # a plain-text join would let a room name shift his fields.
    assert seen[0].startswith("{{ {'rooms': [") and seen[0].endswith("} | to_json }}"), seen[0]
    assert "'person': states(" in seen[0] and "'room': states(" in seen[0]
    assert snapshot == ["office"] and snapshot.whereabouts == ("home", "Office")
    set_owner_id(OWNER)        # presence_fn_for re-reads the owner from settings (unset here)
    fn = lambda: snapshot  # noqa: E731
    block = presence_block({"user_id": OWNER, "privacy_context": "private"}, fn)
    assert "most likely there" in block and "Rooms showing occupancy right now: office." in block
    assert presence_block({"user_id": OWNER, "privacy_context": "public"}, fn) == "", "owner + private only"


def test_a_room_name_can_never_shift_his_fields():
    """Codex, 2026-10-03: with a ';;' text format, a room named 'den;;annex' parsed
    'annex' as his person state and told her he was away. JSON carries each field."""
    answer = json.dumps({"rooms": ["den;;annex", "a,b"], "person": "home", "room": "Office"})
    real = httpx.Client
    try:
        httpx.Client = lambda **kw: real(transport=httpx.MockTransport(  # type: ignore[misc]
            lambda r: httpx.Response(200, text=answer)), **kw)
        snapshot = presence_fn_for(settings(presence_context=True, presence_rooms="den;;annex=binary_sensor.x"))()
    finally:
        httpx.Client = real  # type: ignore[misc]
    assert snapshot == ["den;;annex", "a,b"] and snapshot.whereabouts == ("home", "Office")


@pytest.mark.parametrize("answer,rooms,whereabouts", [
    ({"rooms": [], "person": "home", "room": 7}, [], ("", "")),          # a bad room states nothing about him
    ({"rooms": [], "person": None, "room": "Office"}, [], ("", "")),     # a bad person state, likewise
    ({"rooms": ["office", 3, ""], "person": "home", "room": "Office"}, ["office"], ("home", "Office")),
    ({"rooms": "office", "person": "home", "room": "Office"}, [], None),  # not a list: silence, not letters
    ({"rooms": {"office": False}, "person": "home", "room": "Office"}, [], None),
    (["office"], [], None),
])
def test_a_malformed_answer_is_silence_field_by_field(answer, rooms, whereabouts):
    real = httpx.Client
    try:
        httpx.Client = lambda **kw: real(transport=httpx.MockTransport(  # type: ignore[misc]
            lambda r: httpx.Response(200, text=json.dumps(answer))), **kw)
        snapshot = presence_fn_for(settings(presence_context=True))()
    finally:
        httpx.Client = real  # type: ignore[misc]
    assert snapshot == rooms and getattr(snapshot, "whereabouts", None) == whereabouts
    if whereabouts == ("", ""):
        assert format_presence(snapshot, None, whereabouts=snapshot.whereabouts) == ""


def test_only_a_private_owner_turn_hears_where_he_is():
    snapshot_fn = lambda: __import__("aerys_v2.factory", fromlist=["PresenceSnapshot"]).PresenceSnapshot(  # noqa: E731
        [], whereabouts=("home", "Office"))
    for privacy in (None, "", "guild", "public"):
        identity = {"user_id": OWNER} if privacy is None else {"user_id": OWNER, "privacy_context": privacy}
        assert presence_block(identity, snapshot_fn) == "", privacy
    assert "most likely there" in presence_block({"user_id": OWNER, "privacy_context": "private"}, snapshot_fn)


def test_a_voice_turn_keeps_home_and_drops_the_lagging_phone_room():
    from aerys_v2.factory import PresenceSnapshot
    fn = lambda: PresenceSnapshot([], whereabouts=("home", "Office"))  # noqa: E731
    text = presence_block({"user_id": OWNER, "privacy_context": "private", "voice": True}, fn)
    assert "says he is home" in text and "office" not in text.lower()
