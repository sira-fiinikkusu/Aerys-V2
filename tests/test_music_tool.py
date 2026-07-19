"""Offline tests for the music tool (07-01 Play Music reborn, 2026-07-18).

httpx MockTransport stands in for HA: a path-dispatching handler records every
service call and scripts Music Assistant search responses. What these prove:
origin-device targeting (the owner spec — play on the device that heard you),
named-target fuzzy resolution against the player-map allowlist with honest
refusals outside it, search-then-play with category preference and media_type
override, WRITE_OK_PREFIX on writes (the voice silent-success contract — never
talk over the song), reads unprefixed, and honest strings (never raises) on
every failure shape.
"""

import json

import httpx

from aerys_v2.tools.home_control import WRITE_OK_PREFIX
from aerys_v2.tools.music import build_music_tool

OFFICE_DEV = "dev-office"
LIVING_DEV = "dev-living"
PLAYERS = {
    OFFICE_DEV: "media_player.office_satellite",
    LIVING_DEV: "media_player.living_room_speaker",
}

SEARCH_RESULTS = {
    "artists": [{"name": "Daft Punk", "uri": "spotify://artist/a1"}],
    "tracks": [
        {
            "name": "One More Time",
            "uri": "spotify://track/t1",
            "artists": [{"name": "Daft Punk"}],
        }
    ],
    "albums": [{"name": "Discovery", "uri": "spotify://album/al1"}],
    "playlists": [{"name": "Focus Beats", "uri": "spotify://playlist/p1"}],
}


class FakeHA:
    """Path-dispatching MockTransport handler that records calls."""

    def __init__(self, search_results=None, fail_play=False):
        self.calls: list[tuple[str, dict]] = []
        self.search_results = SEARCH_RESULTS if search_results is None else search_results
        self.fail_play = fail_play

    def __call__(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        body = json.loads(request.content) if request.content else {}
        self.calls.append((path, body))
        if path == "/api/services/music_assistant/search":
            return httpx.Response(200, json={"service_response": self.search_results})
        if path == "/api/services/music_assistant/play_media":
            if self.fail_play:
                return httpx.Response(500, text="boom")
            return httpx.Response(200, json=[])
        if path.startswith("/api/services/media_player/"):
            return httpx.Response(200, json=[])
        if path.startswith("/api/states/"):
            return httpx.Response(
                200,
                json={
                    "entity_id": path.rsplit("/", 1)[-1],
                    "state": "playing",
                    "attributes": {
                        "media_title": "One More Time",
                        "media_artist": "Daft Punk",
                        "volume_level": 0.35,
                    },
                },
            )
        return httpx.Response(404)


def make_tool(ha: FakeHA, *, default_player=None, players=PLAYERS):
    return build_music_tool(
        base_url="http://ha.test:8123",
        token="t",
        config_entry_id="entry-1",
        players=players,
        default_player=default_player,
        client=httpx.Client(transport=httpx.MockTransport(ha)),
    )


def cfg(device_id=None):
    identity = {"user_id": "person-1"}
    if device_id:
        identity["device_id"] = device_id
    return {"configurable": {"identity": identity}}


# ---- targeting -------------------------------------------------------------------

def test_play_targets_the_device_that_heard_the_request():
    ha = FakeHA()
    tool = make_tool(ha)
    out = tool.invoke({"operation": "play", "query": "one more time"}, config=cfg(OFFICE_DEV))
    assert out.startswith(WRITE_OK_PREFIX)
    assert "One More Time by Daft Punk" in out
    assert "office satellite" in out
    play = next(b for p, b in ha.calls if p.endswith("/play_media"))
    assert play["entity_id"] == "media_player.office_satellite"


def test_named_target_overrides_origin_device():
    ha = FakeHA()
    tool = make_tool(ha)
    out = tool.invoke(
        {"operation": "play", "query": "daft punk", "target": "living room"},
        config=cfg(OFFICE_DEV),
    )
    assert out.startswith(WRITE_OK_PREFIX)
    play = next(b for p, b in ha.calls if p.endswith("/play_media"))
    assert play["entity_id"] == "media_player.living_room_speaker"


def test_unknown_target_refuses_and_lists_speakers():
    ha = FakeHA()
    tool = make_tool(ha)
    out = tool.invoke(
        {"operation": "play", "query": "x", "target": "bathroom"}, config=cfg(OFFICE_DEV)
    )
    assert "don't have a speaker called 'bathroom'" in out
    assert "office satellite" in out and "living room speaker" in out
    assert ha.calls == []  # nothing was played


def test_no_device_no_default_asks_for_a_speaker():
    ha = FakeHA()
    tool = make_tool(ha)
    out = tool.invoke({"operation": "play", "query": "x"}, config=cfg())
    assert "no default speaker" in out
    assert ha.calls == []


def test_no_device_falls_back_to_default_player():
    ha = FakeHA()
    tool = make_tool(ha, default_player="media_player.office_satellite")
    out = tool.invoke({"operation": "play", "query": "one more time"}, config=cfg())
    assert out.startswith(WRITE_OK_PREFIX)
    play = next(b for p, b in ha.calls if p.endswith("/play_media"))
    assert play["entity_id"] == "media_player.office_satellite"


# ---- play resolution -------------------------------------------------------------

def test_play_prefers_tracks_and_reports_resolved_name():
    ha = FakeHA()
    tool = make_tool(ha)
    tool.invoke({"operation": "play", "query": "one more time"}, config=cfg(OFFICE_DEV))
    play = next(b for p, b in ha.calls if p.endswith("/play_media"))
    assert play["media_id"] == "spotify://track/t1"
    assert play["media_type"] == "track"


def test_media_type_override_picks_that_category():
    ha = FakeHA()
    tool = make_tool(ha)
    out = tool.invoke(
        {"operation": "play", "query": "focus", "media_type": "playlist"},
        config=cfg(OFFICE_DEV),
    )
    assert "Focus Beats" in out
    play = next(b for p, b in ha.calls if p.endswith("/play_media"))
    assert play["media_id"] == "spotify://playlist/p1"
    assert play["media_type"] == "playlist"


def test_no_results_is_honest():
    ha = FakeHA(search_results={})
    tool = make_tool(ha)
    out = tool.invoke(
        {"operation": "play", "query": "gibberishband"}, config=cfg(OFFICE_DEV)
    )
    assert "couldn't find anything" in out


def test_failed_play_media_is_honest_never_raises():
    ha = FakeHA(fail_play=True)
    tool = make_tool(ha)
    out = tool.invoke({"operation": "play", "query": "one more time"}, config=cfg(OFFICE_DEV))
    assert "FAILED" in out and "One More Time" in out


def test_empty_query_asks_what_to_play():
    ha = FakeHA()
    tool = make_tool(ha)
    out = tool.invoke({"operation": "play"}, config=cfg(OFFICE_DEV))
    assert "what to play" in out.lower()


# ---- transport + volume + read ---------------------------------------------------

def test_pause_hits_media_player_service_with_ok_prefix():
    ha = FakeHA()
    tool = make_tool(ha)
    out = tool.invoke({"operation": "pause"}, config=cfg(OFFICE_DEV))
    assert out.startswith(WRITE_OK_PREFIX)
    assert ha.calls[0][0] == "/api/services/media_player/media_pause"
    assert ha.calls[0][1]["entity_id"] == "media_player.office_satellite"


def test_next_and_resume_map_to_the_right_services():
    ha = FakeHA()
    tool = make_tool(ha)
    tool.invoke({"operation": "next"}, config=cfg(OFFICE_DEV))
    tool.invoke({"operation": "resume"}, config=cfg(OFFICE_DEV))
    assert ha.calls[0][0].endswith("/media_next_track")
    assert ha.calls[1][0].endswith("/media_play")


def test_volume_clamps_and_converts_to_level():
    ha = FakeHA()
    tool = make_tool(ha)
    out = tool.invoke({"operation": "volume", "query": "150"}, config=cfg(OFFICE_DEV))
    assert out.startswith(WRITE_OK_PREFIX) and "100%" in out
    assert ha.calls[0][1] == {
        "entity_id": "media_player.office_satellite",
        "volume_level": 1.0,
    }


def test_volume_garbage_is_honest():
    ha = FakeHA()
    tool = make_tool(ha)
    out = tool.invoke({"operation": "volume", "query": "loud"}, config=cfg(OFFICE_DEV))
    assert "0-100" in out
    assert ha.calls == []


def test_now_playing_reads_without_ok_prefix():
    ha = FakeHA()
    tool = make_tool(ha)
    out = tool.invoke({"operation": "now_playing"}, config=cfg(OFFICE_DEV))
    assert not out.startswith(WRITE_OK_PREFIX)
    assert "One More Time" in out and "Daft Punk" in out and "35%" in out


def test_unknown_operation_lists_valid_ones():
    ha = FakeHA()
    tool = make_tool(ha)
    out = tool.invoke({"operation": "shuffle"}, config=cfg(OFFICE_DEV))
    assert "Unknown operation" in out and "now_playing" in out


def test_ha_unreachable_is_honest_never_raises():
    def down(_request):
        raise httpx.ConnectError("refused")

    tool = build_music_tool(
        base_url="http://ha.test:8123",
        token="t",
        config_entry_id="entry-1",
        players=PLAYERS,
        client=httpx.Client(transport=httpx.MockTransport(down)),
    )
    out = tool.invoke({"operation": "play", "query": "x"}, config=cfg(OFFICE_DEV))
    assert "unreachable" in out
