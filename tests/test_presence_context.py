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
        return httpx.Response(200, text="office,living room")

    real = httpx.Client
    try:
        httpx.Client = lambda **kw: real(transport=httpx.MockTransport(handler), **kw)  # type: ignore[misc]
        fn = presence_fn_for(settings(presence_context=True))
        assert fn is not None and fn() == ["office", "living room"]
    finally:
        httpx.Client = real  # type: ignore[misc]
    assert len(calls) == 1, "one request for the whole house"


def test_an_unhappy_home_assistant_is_silence_not_a_guess():
    for resp in (httpx.Response(500, text="nope"), httpx.Response(200, text="")):
        real = httpx.Client
        try:
            httpx.Client = lambda **kw: real(transport=httpx.MockTransport(lambda r: resp), **kw)  # type: ignore[misc]
            fn = presence_fn_for(settings(presence_context=True))
            assert fn() == []
        finally:
            httpx.Client = real  # type: ignore[misc]
