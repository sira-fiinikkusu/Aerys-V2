"""D53: her Music Assistant MCP reach. The bridge runs through the REAL langchain.mcp + FastMCP stack against an
in-process server; the music tool runs against a recording fake. All songs, players and names are placeholders."""
import json
import threading
import warnings

import httpx
import pytest

from aerys_v2.tools.music import build_music_tool
from aerys_v2.tools.music_mcp import ALLOWED, MusicAssistantError, MusicAssistantMCP, ma_for
from test_dj_phase_a import HA, OWNER, PLAYER, PRIVATE, Private

warnings.simplefilter("ignore")      # LangChainBetaWarning from langchain.mcp


def fake_server(calls):
    from fastmcp import FastMCP

    server = FastMCP("ma-fake")

    @server.tool
    def media_add_to_favorites(uri: str) -> str:
        calls.append(("favorite", uri))
        return "ok"

    @server.tool
    def library_search_tracks(query: str, limit: int = 5) -> list[dict]:
        calls.append(("search", query))
        return [{"uri": "library://track/1", "name": "Example Song", "artists": ["Example Band"]}]

    @server.tool
    def players_set_volume(player_id: str, volume: int) -> str:        # offered by the server, never loaded
        calls.append(("volume", player_id))
        return "changed"

    @server.tool
    def queue_set_shuffle(queue_id: str, enabled: bool) -> str:
        raise ValueError("queue not found")
    return server


def bridge(calls):
    from fastmcp import Client
    server = fake_server(calls)
    return MusicAssistantMCP("http://ma.invalid/mcp/v1", "example-token", client_factory=lambda: Client(server))


def test_the_real_langchain_bridge_calls_an_allowed_tool_from_a_sync_caller():
    calls = []
    ma = bridge(calls)
    assert json.loads(ma.call("library_search_tracks", query="example", limit=2))[0]["uri"] == "library://track/1"
    assert ma.call("media_add_to_favorites", uri="library://track/1") == "ok"
    assert calls == [("search", "example"), ("favorite", "library://track/1")]


def test_tools_outside_the_allowlist_are_never_loaded_or_called():
    calls = []
    ma = bridge(calls)
    assert "players_set_volume" not in ALLOWED
    with pytest.raises(MusicAssistantError):
        ma.call("players_set_volume", player_id="x", volume=99)
    ma.call("media_add_to_favorites", uri="u")          # loads the tool list
    assert "players_set_volume" not in ma._tools and ("volume", "x") not in calls


def test_a_server_side_failure_is_an_honest_error_never_text_that_reads_like_success():
    ma = bridge([])
    with pytest.raises(MusicAssistantError, match="queue not found"):
        ma.call("queue_set_shuffle", queue_id="q", enabled=True)


def test_calls_from_many_threads_share_one_loop():
    calls = []
    ma = bridge(calls)
    errors = []

    def go(n):
        try:
            ma.call("media_add_to_favorites", uri=f"u{n}")
        except Exception as e:              # pragma: no cover - reported below
            errors.append(e)
    threads = [threading.Thread(target=go, args=(n,)) for n in range(8)]
    [t.start() for t in threads]
    [t.join(20) for t in threads]
    assert not errors and len([c for c in calls if c[0] == "favorite"]) == 8


def test_an_unreachable_server_is_one_plain_error_and_rediscovers_later():
    class Broken:
        async def __aenter__(self):
            raise ConnectionError("refused")

        async def __aexit__(self, *a):
            return False
    ma = MusicAssistantMCP("http://ma.invalid", "t", client_factory=Broken)
    with pytest.raises(MusicAssistantError, match="couldn't be reached"):
        ma.call("media_add_to_favorites", uri="u")
    assert ma._tools is None


def test_the_token_is_never_in_the_error_or_the_repr():
    ma = MusicAssistantMCP("http://ma.invalid", "secret-example-token")
    assert "secret-example-token" not in repr(ma) and "secret-example-token" not in str(vars(ma).get("url"))


def test_ma_for_needs_both_settings():
    from types import SimpleNamespace
    from pydantic import SecretStr
    assert ma_for(SimpleNamespace(ma_mcp_url="", ma_mcp_token=SecretStr("t"))) is None
    assert ma_for(SimpleNamespace(ma_mcp_url="http://x", ma_mcp_token=None)) is None
    assert isinstance(ma_for(SimpleNamespace(ma_mcp_url="http://x", ma_mcp_token=SecretStr("t"))), MusicAssistantMCP)


# ---- the music tool's D53 ops, against a recording Music Assistant ------------------------------------------
class MA:
    def __init__(self):
        self.calls = []
        self.search = [{"uri": "library://track/7", "name": "Example Song", "artists": ["Example Band"]}]
        self.queue = {"queue_id": "q1", "items": [{"name": "Next Example", "media_item": {
            "name": "Next Example", "artists": ["Example Band"]}}]}
        self.fail = ""

    def call(self, name, /, **args):
        self.calls.append((name, args))
        if name == self.fail:
            raise MusicAssistantError("Music Assistant couldn't be reached")
        return json.dumps({
            "library_search_tracks": self.search,
            "players_list_players": [{"player_id": "ma-office", "name": "Example Room"}],
            "queue_get_active_queue": self.queue,
            "metadata_recently_played": [{"uri": "library://track/3", "name": "Old Example", "artists": []},
                                         {"uri": "library://artist/4", "name": "Example Band", "artists": []}],
        }.get(name, "ok"))


def music(ha, ma):
    return Private(build_music_tool(base_url="http://ha.test", token="example", config_entry_id="example",
                                    players={"example": PLAYER}, default_player=PLAYER, owner_person_id=OWNER,
                                    client=httpx.Client(transport=httpx.MockTransport(ha)), ma=ma))


def ha_playing(title="Example Song", artist="Example Band", uri="spotify_connect://audio_source/example"):
    ha = HA()
    ha.state = {"state": "playing", "attributes": {"friendly_name": "Example Room", "media_title": title,
                                                    "media_artist": artist, "media_content_id": uri}}
    return ha


def test_like_the_song_playing_from_the_spotify_app_finds_it_exactly():
    ha, ma = ha_playing(), MA()
    out = music(ha, ma).invoke({"operation": "like"})
    assert out.startswith("Done: liked Example Song by Example Band")
    assert ("media_add_to_favorites", {"uri": "library://track/7"}) in ma.calls


def test_like_a_music_assistant_song_uses_its_own_uri_without_searching():
    ha, ma = ha_playing(uri="library://track/99"), MA()
    music(ha, ma).invoke({"operation": "like"})
    assert ma.calls == [("media_add_to_favorites", {"uri": "library://track/99"})]


@pytest.mark.parametrize("hits,word", [
    ([], "couldn't find exactly"),
    ([{"uri": "library://track/1", "name": "Example Songs", "artists": ["Example Band"]}], "couldn't find exactly"),
])
def test_like_never_guesses_a_near_song(hits, word):
    ha, ma = ha_playing(), MA()
    ma.search = hits
    out = music(ha, ma).invoke({"operation": "like", "query": "Example Song"})
    assert word in out and not any(c[0] == "media_add_to_favorites" for c in ma.calls)


def test_two_artists_same_title_asks_which():
    ha, ma = ha_playing(), MA()
    ma.search = [{"uri": "library://track/1", "name": "Example Song", "artists": ["Band One"]},
                 {"uri": "library://track/2", "name": "Example Song", "artists": ["Band Two"]}]
    out = music(ha, ma).invoke({"operation": "like", "query": "Example Song"})
    assert "which artist" in out and not any(c[0] == "media_add_to_favorites" for c in ma.calls)


def test_add_to_library_and_playlist():
    ha, ma = ha_playing(), MA()
    assert music(ha, ma).invoke({"operation": "add_to_library", "query": "Example Song"}).startswith("Done:")
    assert ("media_add_to_library", {"uri": "library://track/7"}) in ma.calls
    out = music(ha, ma).invoke({"operation": "add_to_playlist", "query": "Example Song", "playlist": "Example Mix"})
    assert out.startswith("Done: added Example Song by Example Band to Example Mix")
    assert ("playlists_add_track", {"playlist_id": "library://playlist/3", "track_uri": "library://track/7"}) in ma.calls
    assert "couldn't find a playlist" in music(ha, ma).invoke(
        {"operation": "add_to_playlist", "query": "Example Song", "playlist": "Nothing Like It"})


def test_new_playlist_and_the_playlist_cache_is_refreshed():
    ha, ma = ha_playing(), MA()
    tool = music(ha, ma)
    tool.invoke({"operation": "my_playlists"})                    # fills the playlist cache
    reads = [c for c in ha.calls if c[0] == "get_library"]
    assert tool.invoke({"operation": "new_playlist", "query": "Coding Example"}).startswith("Done: made the playlist")
    assert ("playlists_create_playlist", {"name": "Coding Example"}) in ma.calls
    tool.invoke({"operation": "my_playlists"})
    assert len([c for c in ha.calls if c[0] == "get_library"]) > len(reads)     # read again, so it shows up


def test_recently_played_and_queue_read_his_music():
    ha, ma = ha_playing(), MA()
    out = music(ha, ma).invoke({"operation": "recently_played"})
    assert "Old Example" in out and "Example Band" not in out.split("\n", 1)[1]      # tracks only
    out = music(ha, ma).invoke({"operation": "queue"})
    assert "Next Example by Example Band" in out
    assert ("queue_get_active_queue", {"player_id": "ma-office", "include_items": 10, "items_from_current": True}) in ma.calls


def test_shuffle_uses_music_assistants_queue_and_says_so_when_there_is_none():
    ha, ma = ha_playing(), MA()
    assert music(ha, ma).invoke({"operation": "shuffle", "query": "off"}).startswith("Done: shuffle set to off")
    assert ("queue_set_shuffle", {"queue_id": "q1", "enabled": False}) in ma.calls
    assert not any(c[0] == "shuffle_set" for c in ha.calls)          # never HA's refused shuffle_set
    ma.queue = None
    out = music(ha, ma).invoke({"operation": "repeat", "query": "all"})
    assert "Spotify app" in out and not any(c[0] == "queue_set_repeat" for c in ma.calls)


@pytest.mark.parametrize("identity,thread,word", [
    ({"user_id": OWNER, "platform": "discord", "privacy_context": "public", "channel_kind": "guild"}, "discord:1", "Not here"),
    ({"user_id": "guest-example", "platform": "discord", "privacy_context": "private"}, "person:g", "only he"),
])
@pytest.mark.parametrize("op", ["recently_played", "queue"])
def test_his_listening_stays_his(op, identity, thread, word):
    ha, ma = ha_playing(), MA()
    out = music(ha, ma).invoke({"operation": op}, config={"configurable": {"thread_id": thread, "identity": identity}})
    assert word in out.lower() or word in out
    assert ma.calls == []


@pytest.mark.parametrize("op,args", [("like", {}), ("add_to_library", {}), ("new_playlist", {"query": "X"}),
                                     ("add_to_playlist", {"query": "Example Song", "playlist": "Example Mix"})])
def test_writes_are_his_alone_but_a_shared_room_is_fine(op, args):
    ha, ma = ha_playing(), MA()
    guest = {"configurable": {"thread_id": "person:g", "identity": {
        "user_id": "guest-example", "platform": "discord", "privacy_context": "private"}}}
    assert "Only he can change" in music(ha, ma).invoke({"operation": op, **args}, config=guest)
    unsure = {"configurable": {"thread_id": "person:o", "identity": {
        "user_id": OWNER, "voice": True, "speaker": {"id": "unknown"}, "privacy_context": "private"}}}
    assert "not sure it's him" in music(ha, ma).invoke({"operation": op, **args}, config=unsure)
    assert ma.calls == []
    guild = {"configurable": {"thread_id": "discord:1", "identity": {
        "user_id": OWNER, "platform": "discord", "privacy_context": "public", "channel_kind": "guild"}}}
    assert music(ha, ma).invoke({"operation": op, **args}, config=guild).startswith("Done:")


def test_music_assistant_down_is_a_plain_line_and_nothing_else_breaks():
    ha, ma = ha_playing(), MA()
    ma.fail = "library_search_tracks"
    assert "didn't save" in music(ha, ma).invoke({"operation": "like"})
    ma.fail = "media_add_to_favorites"
    assert "didn't save" in music(ha, ma).invoke({"operation": "like", "query": "Example Song"})
    assert music(ha, None).invoke({"operation": "like"}).startswith("That needs Music Assistant")
    assert music(ha, None).invoke({"operation": "play", "query": "Example Mix"}).startswith("Done:")    # HA half unchanged


def test_a_tool_argument_called_name_reaches_the_server():
    from fastmcp import FastMCP, Client
    got = []
    server = FastMCP("ma-fake")

    @server.tool
    def playlists_create_playlist(name: str) -> str:
        got.append(name)
        return "made"
    ma = MusicAssistantMCP("http://ma.invalid", "t", client_factory=lambda: Client(server))
    assert ma.call("playlists_create_playlist", name="Example List") == "made" and got == ["Example List"]


def test_a_known_artist_must_match_never_another_artists_same_title():
    ha, ma = ha_playing(title="Example Song", artist="Band Playing"), MA()
    ma.search = [{"uri": "library://track/1", "name": "Example Song", "artists": ["Some Other Band"]}]
    out = music(ha, ma).invoke({"operation": "like"})
    assert "couldn't find exactly" in out and not any(c[0] == "media_add_to_favorites" for c in ma.calls)


def test_a_named_song_with_by_splits_title_and_artist():
    ha, ma = ha_playing(), MA()
    ma.search = [{"uri": "library://track/1", "name": "Example Song", "artists": ["Band One"]},
                 {"uri": "library://track/2", "name": "Example Song", "artists": ["Band Two"]}]
    assert music(ha, ma).invoke({"operation": "like", "query": "Example Song by Band Two"}).startswith("Done:")
    assert ("media_add_to_favorites", {"uri": "library://track/2"}) in ma.calls


def test_an_unmatched_speaker_says_so_instead_of_nothing_playing():
    ha, ma = ha_playing(), MA()
    ha.state["attributes"]["friendly_name"] = "A Room Music Assistant Doesn't Know"
    out = music(ha, ma).invoke({"operation": "shuffle", "query": "on"})
    assert "couldn't match" in out and "Spotify app" not in out



# ---- Codex 11/15 review fixes --------------------------------------------------------------------------------
def test_writes_check_who_asks_before_the_room():
    ha, ma = ha_playing(), MA()
    unknown_voice_public = {"configurable": {"thread_id": f"person:{OWNER}", "identity": {
        "user_id": OWNER, "voice": True, "speaker": {"id": "unknown"}, "privacy_context": "public"}}}
    out = music(ha, ma).invoke({"operation": "new_playlist", "query": "X"}, config=unknown_voice_public)
    assert "not sure it's him" in out and ma.calls == []


def test_a_write_matches_every_word_the_is_not_optional():
    ha, ma = ha_playing(), MA()
    ma.search = [{"uri": "library://track/1", "name": "The Example", "artists": ["Example Band"]}]
    out = music(ha, ma).invoke({"operation": "like", "query": "Example by Example Band"})
    assert "couldn't find exactly" in out and not any(c[0] == "media_add_to_favorites" for c in ma.calls)


def test_a_playlist_write_needs_one_exact_name():
    ha, ma = ha_playing(), MA()
    out = music(ha, ma).invoke({"operation": "add_to_playlist", "query": "Example Song", "playlist": "Example Mixes"})
    assert "couldn't find a playlist" in out and not any(c[0] == "playlists_add_track" for c in ma.calls)


def test_an_idle_players_leftover_title_is_never_liked():
    ha, ma = ha_playing(), MA()
    ha.state["state"] = "idle"
    out = music(ha, ma).invoke({"operation": "like"})
    assert "Nothing is playing" in out and ma.calls == []


def test_concurrent_first_calls_start_exactly_one_loop():
    calls = []
    ma = bridge(calls)
    before = {t.ident for t in threading.enumerate() if t.name == "ma-mcp"}
    threads = [threading.Thread(target=ma.call, args=("media_add_to_favorites",), kwargs={"uri": f"u{n}"})
               for n in range(8)]
    [t.start() for t in threads]
    [t.join(20) for t in threads]
    after = {t.ident for t in threading.enumerate() if t.name == "ma-mcp"} - before
    assert len(after) == 1 and len([c for c in calls if c[0] == "favorite"]) == 8
