"""Board #264, option A (11/26): she starts a song on his phone through Spotify while the phone is one of
Spotify's places to play (measured: about twelve minutes after Spotify was last used there), and says so
plainly when it isn't. Owner only. "Playing" is said only once Spotify itself reports it."""

import json

import httpx

from aerys_v2.tools.home_control import WRITE_OK_PREFIX
from aerys_v2.tools.music import build_music_tool

OWNER = "person-1"
PLAYER = "media_player.test_spotify"
PHONE = "Test Phone"
SEARCH = {
    "tracks": [{"name": "One More Time", "uri": "spotify--abc123://track/0DiWol3AO6WpXZgp0goxAV",
                "artists": [{"name": "Daft Punk"}]}],
    "artists": [{"name": "Daft Punk", "uri": "spotify--abc123://artist/4tZwfgrHOc3mvqYlEYSvVi"}],
    "albums": [{"name": "Discovery", "uri": "spotify--abc123://album/2noRn2Aes5aoNVsU6iWThc"}],
    "playlists": [{"name": "Focus Beats", "uri": "spotify--abc123://playlist/37i9dQZF1DWZeKCadgRdKQ"}],
}


class HA:
    def __init__(self, *, sources=(PHONE, "Kitchen"), starts=True, play_status=200, search=None, library=None,
                 player_status=200):
        self.sources, self.starts, self.play_status = list(sources), starts, play_status
        self.search, self.library, self.player_status = SEARCH if search is None else search, library or {}, player_status
        self.calls: list[tuple[str, dict]] = []
        self.playing = False

    def __call__(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        body = json.loads(request.content) if request.content else {}
        self.calls.append((path, body))
        if path == f"/api/states/{PLAYER}":
            attrs = {"source_list": self.sources, "source": PHONE if self.playing else None}
            return httpx.Response(self.player_status, json={"state": "playing" if self.playing else "idle", "attributes": attrs})
        if path == "/api/services/music_assistant/search":
            return httpx.Response(200, json={"service_response": self.search})
        if path == "/api/services/music_assistant/get_library":
            return httpx.Response(200, json={"service_response": {"items": self.library.get(body.get("media_type"), [])}})
        if path == "/api/services/media_player/select_source":
            return httpx.Response(200, json=[])
        if path == "/api/services/media_player/play_media":
            if self.play_status == 200 and self.starts:
                self.playing = True
            return httpx.Response(self.play_status, json=[])
        return httpx.Response(404, json={})

    def posts(self, service):
        return [b for p, b in self.calls if p == f"/api/services/{service}"]


def tool(ha, *, player=PLAYER, source=PHONE):
    return build_music_tool(
        base_url="http://ha.test", token="t", config_entry_id="entry-1",
        players={"dev-office": "media_player.office_satellite"}, default_player="media_player.office_satellite",
        client=httpx.Client(transport=httpx.MockTransport(ha)), owner_person_id=OWNER,
        phone_notify="mobile_app_test", phone_media_sensor="sensor.test_media",
        phone_spotify_player=player, phone_spotify_source=source, sleep=lambda s: None)


def him(**extra):
    return {"configurable": {"identity": {"user_id": OWNER, "platform": "discord", **extra}, "thread_id": "discord:dm:1"}}


def play(ha, query="one more time", config=None, media_type="", **kw):
    return tool(ha, **kw).invoke({"operation": "play", "target": "my phone", "query": query, "media_type": media_type},
                                 config=config or him())


def test_a_song_starts_on_his_phone_and_playing_is_said_once_spotify_says_so():
    ha = HA()
    out = play(ha)
    assert out.startswith(f"{WRITE_OK_PREFIX} playing One More Time by Daft Punk on his phone.")
    assert ha.posts("media_player/select_source") == [{"entity_id": PLAYER, "source": PHONE}]
    assert ha.posts("media_player/play_media") == [
        {"entity_id": PLAYER, "media_content_id": "spotify:track:0DiWol3AO6WpXZgp0goxAV", "media_content_type": "music"}]
    assert ha.posts("music_assistant/play_media") == []          # never through a house speaker


def test_an_artist_an_album_and_a_playlist_go_as_what_they_are():
    for kind, link in (("artist", "spotify:artist:4tZwfgrHOc3mvqYlEYSvVi"), ("album", "spotify:album:2noRn2Aes5aoNVsU6iWThc"),
                       ("playlist", "spotify:playlist:37i9dQZF1DWZeKCadgRdKQ")):
        ha = HA()
        out = play(ha, query="anything", media_type=kind)
        assert out.startswith(f"{WRITE_OK_PREFIX} playing"), (kind, out)
        assert ha.posts("media_player/play_media") == [
            {"entity_id": PLAYER, "media_content_id": link, "media_content_type": kind}]


def test_when_the_phone_is_not_one_of_spotifys_places_she_says_so_and_sends_nothing():
    ha = HA(sources=("Kitchen",))
    out = play(ha)
    assert "isn't reachable" in out and "open Spotify on his phone" in out and "Nothing was sent" in out
    assert not out.startswith(WRITE_OK_PREFIX)
    assert [p for p, _ in ha.calls if "/services/" in p] == []    # not even a search


def test_sent_but_not_started_is_never_called_playing():
    ha = HA(starts=False)
    out = play(ha)
    assert out.startswith(f"{WRITE_OK_PREFIX} sent One More Time by Daft Punk to his phone")
    assert "hasn't said it started" in out and "playing One More Time" not in out


def test_a_refused_play_is_said_plainly():
    out = play(HA(play_status=500))
    assert "FAILED" in out and not out.startswith(WRITE_OK_PREFIX)
    out = play(HA(player_status=500))
    assert "couldn't reach Spotify's player" in out and "nothing was sent" in out


def test_only_he_can_start_music_on_his_phone():
    someone = {"configurable": {"identity": {"user_id": "person-2", "platform": "discord"}, "thread_id": "discord:dm:2"}}
    for config in (someone, him(platform="wake"), him(unattended=True),
                   {"configurable": {"identity": {"user_id": OWNER, "platform": "discord"}, "thread_id": "kael:aerys-line"}}):
        ha = HA()
        out = play(ha, config=config)
        assert not out.startswith(WRITE_OK_PREFIX), out
        assert ha.calls == []


def test_his_own_library_song_is_found_by_its_exact_name_and_artist_never_a_near_one():
    library = {"track": [{"name": "One More Time", "uri": "library://track/52", "artists": [{"name": "Daft Punk"}]}]}
    ha = HA(library=library)
    assert play(ha).startswith(f"{WRITE_OK_PREFIX} playing")
    assert ha.posts("media_player/play_media")[0]["media_content_id"] == "spotify:track:0DiWol3AO6WpXZgp0goxAV"
    # the same title by someone else is not his song
    other = {"tracks": [{"name": "One More Time", "uri": "spotify--abc123://track/1111111111111111111111",
                         "artists": [{"name": "Somebody Else"}]}]}
    ha = HA(library=library, search=other)
    out = play(ha)
    assert ha.posts("media_player/play_media") == [] and "couldn't find" in out and "Nothing was sent" in out


def test_a_link_that_is_not_spotifys_is_never_sent():
    for uri in ("library://track/52", "spotify--abc123://track/../../x", "http://evil.test/track/0DiWol3AO6WpXZgp0goxAV",
                "spotify--abc123://podcast/2O2tukhyPYiNFJH3WVLwwM", ""):
        ha = HA(search={"tracks": [{"name": "one more time", "uri": uri, "artists": []}]})
        out = play(ha)
        assert ha.posts("media_player/play_media") == [], uri
        assert not out.startswith(WRITE_OK_PREFIX), (uri, out)


def test_without_the_spotify_settings_she_cannot_start_music_there():
    for kw in ({"player": ""}, {"source": ""}):
        ha = HA()
        out = play(ha, **kw)
        assert "can't start a song there" in out and ha.calls == []


def test_other_operations_on_the_phone_are_still_refused_before_anything_happens():
    for op in ("volume", "play_next", "add_to_queue", "shuffle", "like", "new_playlist", "queue"):
        ha = HA()
        out = tool(ha).invoke({"operation": op, "target": "phone", "query": "x"}, config=him())
        assert "Nothing else works on the phone" in out and ha.calls == [], (op, out)


def test_playing_in_a_room_is_unchanged():
    ha = HA()
    out = tool(ha).invoke({"operation": "play", "target": "office", "query": "one more time"}, config=him())
    assert "office" in out or "FAILED" in out
    assert ha.posts("media_player/select_source") == []           # the phone path was not taken
