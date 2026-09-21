"""Her gap #101, passive half: which rooms show occupancy, as ambient owner-only context.

The active half (speaking because someone arrived) is deliberately NOT built — each
event would be a new metered turn and a behaviour change (Chris, 2026-09-21).
"""
import httpx
import pytest

from aerys_v2.config import Settings
from aerys_v2.factory import _OWNER_ID, presence_block, presence_fn_for
from aerys_v2.services.presence import DEFAULT_ROOMS, format_presence, parse_rooms

OWNER = "6e6bcbed-0000-0000-0000-000000000000"


@pytest.fixture(autouse=True)
def _owner():
    token = _OWNER_ID.set(OWNER)
    yield
    _OWNER_ID.reset(token)


def settings(**kw):
    return Settings(_env_file=None, anthropic_api_key="x", ha_token="ha-tok", **kw)


# --- the wording is careful on purpose ---------------------------------------

def test_the_block_reports_occupancy_and_refuses_to_name_a_person():
    text = format_presence(["office", "living room"], spoken_from="office")
    assert text.startswith("[House presence]")
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
    assert presence_block({"user_id": OWNER, "privacy_context": "private"}, fn).startswith("[House presence]")
    assert presence_block({"user_id": OWNER, "privacy_context": "private"}, None) == ""


def test_a_dark_house_costs_the_block_never_the_turn():
    def boom():
        raise OSError("HA is down")

    assert presence_block({"user_id": OWNER, "privacy_context": "private"}, boom) == ""


# --- the read ------------------------------------------------------------------

def test_only_rooms_reading_on_are_named_and_a_bad_status_is_not_occupancy():
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        entity = request.url.path.rsplit("/", 1)[-1]
        seen.append(entity)
        assert request.headers["Authorization"] == "Bearer ha-tok"
        if entity == "binary_sensor.office_occupancy":
            return httpx.Response(200, json={"state": "on"})
        if entity == "binary_sensor.sunroom_occupancy":
            return httpx.Response(500, json={})          # a broken entity is not "occupied"
        return httpx.Response(200, json={"state": "off"})

    import aerys_v2.factory as f

    real = httpx.Client
    try:
        httpx.Client = lambda **kw: real(transport=httpx.MockTransport(handler), **kw)  # type: ignore[misc]
        fn = presence_fn_for(settings(presence_context=True))
        assert fn is not None and fn() == ["office"]
    finally:
        httpx.Client = real  # type: ignore[misc]
    assert len(seen) == len(DEFAULT_ROOMS), "every configured room is read once"
    assert f is not None
