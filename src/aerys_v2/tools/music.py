"""music — Spotify playback through Music Assistant, on the room that asked.

n8n mapping: this is workflow 07-01 "HA Action: Play Music (owner-gated)"
reborn as a native brain tool — the workflow died with the n8n retirement
(2026-07-11) and the owner asked for it back on LangGraph.

The owner spec (2026-07-18): "play on the device that heard me unless I
otherwise specify." The originating satellite's device_id rides the per-call
identity (state.Identity.device_id — the same seam the timer tool and the
voice follow-ups use), and a config map turns it into that room's Music
Assistant player entity. A caller naming a room instead ("play it in the
living room") fuzzy-matches against the same map. The map IS the write
allowlist: a target outside it is refused honestly, so this tool can never
drive a speaker the owner didn't explicitly wire up.

Playback is search-then-play on purpose: `music_assistant.play_media` accepts
a bare name, but resolving through `music_assistant.search` first means the
confirmation can say WHAT actually got queued ("Playing Discovery by Daft
Punk") instead of parroting the request — the same claims-follow-facts rule
as everywhere else. Honest-failure contract matches home_control: every
refusal/failure is a plain string back to the model, never a raise (an
exception inside a ToolNode kills the whole action turn).

Successful WRITES return WRITE_OK_PREFIX strings: music starting, pausing, or
skipping is its own audible feedback, so the voice path's silent-success rule
skips the spoken follow-up — she must never talk over the song she just
started. now_playing is a READ; its answer IS the follow-up, no prefix.
"""

import json
import logging

import httpx
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

from aerys_v2.state import identity_from_config
from aerys_v2.tools.home_control import WRITE_OK_PREFIX

log = logging.getLogger(__name__)

# Category preference when the caller didn't say what kind of thing to play:
# a named track beats an artist beats an album beats a playlist — "play one
# more time" should queue the song, not an artist page. Keys are the plural
# category names music_assistant.search responds with; values are the singular
# media_type play_media expects.
_CATEGORY_ORDER = (
    ("tracks", "track"),
    ("artists", "artist"),
    ("albums", "album"),
    ("playlists", "playlist"),
    ("radio", "radio"),
    ("audiobooks", "audiobook"),
    ("podcasts", "podcast"),
)
_TYPE_TO_CATEGORY = {singular: plural for plural, singular in _CATEGORY_ORDER}

# media_player service per operation (the plain HA half — pause/skip/etc. are
# ordinary media_player calls, only play routes through Music Assistant).
_TRANSPORT_OPS = {
    "pause": "media_pause",
    "resume": "media_play",
    "stop": "media_stop",
    "next": "media_next_track",
    "previous": "media_previous_track",
}

_SEARCH_LIMIT = 5


def _item_line(item: dict) -> tuple[str, str]:
    """(display name, uri) for a search result item — artist appended when known."""
    name = str(item.get("name") or "?")
    artists = item.get("artists") or []
    if artists:
        artist_names = ", ".join(str(a.get("name")) for a in artists if a.get("name"))
        if artist_names:
            name = f"{name} by {artist_names}"
    return name, str(item.get("uri") or "")


def build_music_tool(
    *,
    base_url: str,
    token: str,
    config_entry_id: str,
    players: dict[str, str],
    default_player: str | None = None,
    client: httpx.Client | None = None,
):
    """Close over the config and return the LangChain tool object.

    players: originating device_id -> that room's Music Assistant media_player
    entity (HA_MUSIC_PLAYERS, same csv format as HA_SATELLITE_MAP). ALSO the
    complete set of speakers this tool may ever drive. default_player catches
    callers with no device_id (text/DM/CLI) — None means those turns get an
    honest "name a speaker" instead of a guess.
    """
    base = base_url.rstrip("/")
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    http = client or httpx.Client(timeout=15.0)
    allowed = sorted(set(players.values()) | ({default_player} if default_player else set()))

    def _friendly(entity: str) -> str:
        # 'media_player.office_satellite' -> 'office satellite' — good enough for
        # speech, no extra HTTP round-trip on the hot path.
        return entity.split(".", 1)[-1].replace("_", " ")

    def _resolve_target(target: str, config: RunnableConfig) -> str:
        """Player entity for this turn, or an honest error string (callers
        distinguish by the 'media_player.' prefix)."""
        want = target.strip().lower()
        if want:
            # Named target: every word must appear in the entity id (slug match —
            # "living room" -> media_player.living_room_*). The map is the allowlist,
            # so an unknown name lists what exists instead of guessing.
            words = [w for w in want.replace(".", " ").split() if w]
            matches = [
                e for e in allowed
                if all(w in e.lower().replace("_", " ") for w in words)
            ]
            if len(matches) == 1:
                return matches[0]
            options = ", ".join(_friendly(e) for e in allowed) or "(none configured)"
            if not matches:
                return (
                    f"I don't have a speaker called '{target}'. "
                    f"Speakers I can play on: {options}."
                )
            return (
                f"'{target}' matches more than one speaker "
                f"({', '.join(_friendly(m) for m in matches)}) — which one?"
            )
        device_id = identity_from_config(config).get("device_id")
        if device_id and device_id in players:
            return players[device_id]
        if default_player:
            return default_player
        options = ", ".join(_friendly(e) for e in allowed) or "(none configured)"
        return (
            "I can't tell which room you're in from here, and no default speaker "
            f"is configured — name one of: {options}."
        )

    @tool
    def music(
        operation: str,
        query: str = "",
        media_type: str = "",
        target: str = "",
        config: RunnableConfig = None,
    ) -> str:
        """Play and control Spotify music on the room's speaker.

        CALL THIS TOOL whenever the user asks to play music, a song, an artist,
        an album, or a playlist ("play some daft punk", "put on my focus
        playlist", "play one more time"), or to pause/resume/skip/stop music or
        change the volume ("pause the music", "next song", "volume 40").

        operation: one of "play", "pause", "resume", "stop", "next", "previous",
        "volume", "now_playing".
        query: for play — what to play, in the user's words (artist, song,
        album, or playlist name). For volume — the level 0-100.
        media_type: optional, for play only — "track", "artist", "album",
        "playlist", "radio", "audiobook", or "podcast" when the user was
        explicit ("the album", "my playlist"); leave empty to let the tool pick.
        target: optional speaker override, ONLY when the user names a room or
        speaker ("in the living room", "on the office speaker"). Leave EMPTY
        otherwise — the tool automatically plays on the device that heard the
        request; NEVER ask which speaker.

        Never claim music is playing, paused, or changed unless this tool's
        reply said so.
        """
        op = operation.strip().lower().replace(" ", "_")

        player = _resolve_target(target, config)
        if not player.startswith("media_player."):
            return player  # honest resolution error, relayed as-is

        # ---- transport controls (plain media_player services) ----------------
        if op in _TRANSPORT_OPS:
            service = _TRANSPORT_OPS[op]
            try:
                r = http.post(
                    f"{base}/api/services/media_player/{service}",
                    headers=headers,
                    json={"entity_id": player},
                )
                r.raise_for_status()
            except httpx.HTTPError as e:
                return f"The {op} on {_friendly(player)} FAILED — Home Assistant said: {e}."
            return f"{WRITE_OK_PREFIX} {op} sent to {_friendly(player)}."

        if op == "volume":
            raw = query.strip().rstrip("%")
            try:
                level = int(float(raw))
            except ValueError:
                return (
                    "For volume, pass the level 0-100 in query — e.g. "
                    'operation="volume", query="40".'
                )
            level = max(0, min(100, level))
            try:
                r = http.post(
                    f"{base}/api/services/media_player/volume_set",
                    headers=headers,
                    json={"entity_id": player, "volume_level": level / 100},
                )
                r.raise_for_status()
            except httpx.HTTPError as e:
                return f"Setting volume on {_friendly(player)} FAILED — Home Assistant said: {e}."
            return f"{WRITE_OK_PREFIX} volume on {_friendly(player)} set to {level}%."

        if op == "now_playing":
            try:
                r = http.get(f"{base}/api/states/{player}", headers=headers)
                r.raise_for_status()
                data = r.json()
            except (httpx.HTTPError, ValueError) as e:
                return f"Couldn't read {_friendly(player)} — Home Assistant said: {e}."
            attrs = data.get("attributes") or {}
            state = data.get("state")
            if state != "playing":
                return f"Nothing is playing on {_friendly(player)} right now (state: {state})."
            title = attrs.get("media_title") or "something"
            artist = attrs.get("media_artist")
            vol = attrs.get("volume_level")
            note = f"Now playing on {_friendly(player)}: {title}"
            if artist:
                note += f" by {artist}"
            if vol is not None:
                note += f" (volume {int(vol * 100)}%)"
            return note + "."

        if op != "play":
            return (
                f"Unknown operation '{operation}'. Valid operations: play, pause, "
                "resume, stop, next, previous, volume, now_playing."
            )

        # ---- play: search Music Assistant, then queue the top hit ------------
        what = query.strip()
        if not what:
            return "Tell me what to play — an artist, song, album, or playlist name."
        try:
            r = http.post(
                f"{base}/api/services/music_assistant/search?return_response",
                headers=headers,
                json={
                    "config_entry_id": config_entry_id,
                    "name": what,
                    "limit": _SEARCH_LIMIT,
                },
            )
            r.raise_for_status()
            results = (r.json() or {}).get("service_response") or {}
        except (httpx.HTTPError, ValueError) as e:
            return f"Music search is unreachable right now ({e})."

        wanted = media_type.strip().lower()
        if wanted and wanted in _TYPE_TO_CATEGORY:
            order = ((_TYPE_TO_CATEGORY[wanted], wanted),)
        else:
            order = _CATEGORY_ORDER
        item, chosen_type = None, ""
        for category, singular in order:
            hits = results.get(category) or []
            if hits:
                item, chosen_type = hits[0], singular
                break
        if item is None:
            return f"I couldn't find anything matching '{what}' on Spotify."

        name, uri = _item_line(item)
        if not uri:
            return f"Music search returned '{name}' without a playable id — try rephrasing."
        try:
            r = http.post(
                f"{base}/api/services/music_assistant/play_media",
                headers=headers,
                json={"entity_id": player, "media_id": uri, "media_type": chosen_type},
            )
            r.raise_for_status()
        except httpx.HTTPError as e:
            return f"Starting '{name}' on {_friendly(player)} FAILED — Home Assistant said: {e}."
        return f"{WRITE_OK_PREFIX} playing {name} on {_friendly(player)}."

    return music
