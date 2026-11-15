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

Successful writes return WRITE_OK_PREFIX. Voice confirmations for music-only
successful turns are capped in service.py; reads and failures remain informative.
"""

from difflib import SequenceMatcher
import re
import time
from threading import Lock
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
_LIBRARY_TTL = 300
_LIBRARY_PAGE_SIZE = 100
_SAVED_PAGE_SIZE = 500
_SAVED_TTL = 1800
_TASTE_TOP = 15
_TASTE_PLAYLISTS = 60
_MATCH_THRESHOLD = 0.80
_MATCH_MARGIN = 0.10


def _name_key(value: str) -> str:
    return " ".join(re.findall(r"\w+", value.casefold()))


def _library_match(query: str, playlists: list[dict]) -> dict | None:
    """Prefer exact names; fuzzy matches need both confidence and separation."""
    want = _name_key(query)
    usable = [p for p in playlists if p.get("uri") and p.get("name")]
    exact = [p for p in usable if _name_key(p["name"]) == want]
    if len(exact) == 1:
        return exact[0]
    if len(exact) > 1:
        return None
    aliases = ("liked songs", "my liked songs", "my favorites", "my favourites",
               "favorites", "favourites", "favorite songs", "favourite songs")
    if any(SequenceMatcher(None, want, alias).ratio() >= _MATCH_THRESHOLD for alias in aliases):
        liked = [p for p in usable if _name_key(p["name"]).startswith("liked songs")]
        if len(liked) == 1:
            return liked[0]
        if len(liked) > 1:
            return None
        fallback = [p for p in usable if _name_key(p["name"]) == "all favorited tracks"]
        return fallback[0] if len(fallback) == 1 else None
    ranked = sorted(((SequenceMatcher(None, want, _name_key(p["name"])).ratio(), i)
                     for i, p in enumerate(usable)), reverse=True)
    if not ranked or ranked[0][0] < _MATCH_THRESHOLD:
        return None
    if len(ranked) > 1 and ranked[0][0] - ranked[1][0] < _MATCH_MARGIN:
        return None
    return usable[ranked[0][1]]


def _close(want: str, key: str) -> float:
    """SequenceMatcher ratio, skipped (0.0) when its cheap upper bounds can't reach THRESHOLD - MARGIN:
    such a name can neither win nor block the winner, so ~2,800 saved names stay fast on the Jetson."""
    floor = _MATCH_THRESHOLD - _MATCH_MARGIN
    sm = SequenceMatcher(None, want, key)
    if sm.real_quick_ratio() < floor or sm.quick_ratio() < floor:
        return 0.0
    return sm.ratio()


def _play_norm(t: object) -> str:
    """The name rule for "is this result what he said": the play path's promotion and the near-name
    yield (D52) must agree, or a yield could drop his saved song for a result play then skips."""
    return " ".join(re.findall(r"\w+", str(t or "").casefold().replace("the ", " ")))


def _saved_keys(item: dict) -> list[str]:
    """Names a saved track or artist answers to: its name, and name + artists."""
    name = str(item.get("name") or "")
    artists = " ".join(str(a.get("name")) for a in item.get("artists") or [] if a.get("name"))
    return [_name_key(name)] + ([_name_key(f"{name} {artists}")] if artists else [])


def _saved_match(query: str, tracks: list[dict], artists: list[dict]) -> tuple[dict, str, bool] | None:
    """(item, kind, exact) from his saved songs and artists, or None. 11/14: a saved song whose title
    speech-to-text misheard (one word swapped for a near-sound, twice) lost to a Spotify-wide search
    both times. Same exact-then-fuzzy rule as playlists. A recording is its title AND artists: two saved copies of one song (single +
    album) are one choice; two different artists' songs with one title are rivals, never a guess."""
    want = _name_key(query)
    pool = [(t, "track") for t in tracks if t.get("uri") and t.get("name")] + \
           [(a, "artist") for a in artists if a.get("uri") and a.get("name")]

    def recording(item: dict) -> str:
        return _saved_keys(item)[-1]

    exact = [c for c in pool if want in _saved_keys(c[0])]
    if exact:
        if len({(kind, recording(item)) for item, kind in exact}) == 1:
            return exact[0][0], exact[0][1], True
        songs = {recording(item) for item, kind in exact if kind == "track"}
        if len(songs) == 1:        # "X" is both an artist and one of his songs: the song he named
            item, kind = next(c for c in exact if c[1] == "track")
            return item, kind, True
        return None
    ranked = sorted(((max(_close(want, key) for key in _saved_keys(item)), n)
                     for n, (item, _) in enumerate(pool)), key=lambda r: (-r[0], r[1]))
    if not ranked or ranked[0][0] < _MATCH_THRESHOLD:
        return None
    best = pool[ranked[0][1]]
    rival = next((score for score, n in ranked[1:]
                  if (pool[n][1], recording(pool[n][0])) != (best[1], recording(best[0]))), 0.0)
    return (best[0], best[1], False) if ranked[0][0] - rival >= _MATCH_MARGIN else None


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
    owner_person_id: str | None = None,
):
    """Close over the config and return the LangChain tool object.

    players: originating device_id -> that room's Music Assistant media_player
    entity (HA_MUSIC_PLAYERS, same csv format as HA_SATELLITE_MAP). ALSO the
    complete set of speakers this tool may ever drive. default_player catches
    callers with no device_id (text/DM/CLI) — None means those turns get an
    honest "name a speaker" instead of a guess.
    owner_person_id: whose library this is. Reading it back (my_playlists, my_taste) follows D47's
    rule for his own things: him, in a room only he reads. Playing music needs no such check.
    """
    base = base_url.rstrip("/")
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    http = client or httpx.Client(timeout=15.0)
    allowed = sorted(set(players.values()) | ({default_player} if default_player else set()))

    library_cache: dict[str, tuple[float, list[dict]]] = {}
    library_lock = Lock()

    def _library(media_type: str, page_size: int, ttl: float = _LIBRARY_TTL) -> list[dict]:
        with library_lock:
            cached = library_cache.get(media_type)
            if cached is not None and time.monotonic() < cached[0]:
                return cached[1]
            items: list[dict] = []
            offset = 0
            while True:
                response = http.post(
                    f"{base}/api/services/music_assistant/get_library?return_response",
                    headers=headers,
                    json={"config_entry_id": config_entry_id, "media_type": media_type,
                          "limit": page_size, "offset": offset, "order_by": "name"},
                )
                response.raise_for_status()
                data = response.json()
                result = data.get("service_response") if isinstance(data, dict) else None
                page = result.get("items") if isinstance(result, dict) else None
                if not isinstance(page, list) or not all(isinstance(p, dict) for p in page):
                    raise ValueError("invalid library response")
                items.extend(page)
                if len(page) < page_size:
                    break
                offset += len(page)
                if offset >= 10000:
                    raise ValueError("library pagination limit reached")
            library_cache[media_type] = (time.monotonic() + ttl, items)
            return items

    def _playlists() -> list[dict]:
        return _library("playlist", _LIBRARY_PAGE_SIZE)

    def _saved(media_type: str) -> list[dict]:
        # ~2,200 saved songs live: five 0.2 s pages, so a longer cache than playlists.
        return _library(media_type, _SAVED_PAGE_SIZE, _SAVED_TTL)

    def _library_read_refusal(config: RunnableConfig) -> str | None:
        from aerys_v2.rooms import NOT_HIM, SHARED, his_private_read

        conf = (config or {}).get("configurable") or {}
        verdict = his_private_read(conf.get("identity") or {}, str(conf.get("thread_id") or ""),
                                   owner_person_id)
        if verdict is None:
            return None
        if verdict == SHARED:
            return ("Not here: his playlists and taste stay out of rooms others can read. "
                    "Ask me in a DM, or by voice.")
        if verdict == NOT_HIM:
            return "His library is his: only he can ask me what's in it."
        return "I'm not sure it's him asking, so his library stays private. Ask again, or in a DM."

    def _unshuffle(player: str) -> str:
        """A song he named plays first. 11/14: shuffle was left on, so her radio for "Call for Fire"
        (the song + similar ones) was shuffled and a radio pick played instead. Best effort: a failed
        read or write leaves the play as it was."""
        try:
            r = http.get(f"{base}/api/states/{player}", headers=headers)
            r.raise_for_status()
            if (r.json().get("attributes") or {}).get("shuffle") is not True:
                return ""
            r = http.post(f"{base}/api/services/media_player/shuffle_set", headers=headers,
                          json={"entity_id": player, "shuffle": False})
            r.raise_for_status()
        except (httpx.HTTPError, ValueError, AttributeError):
            return ""
        return " Shuffle was on; it's off now so this song plays first."

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
        """Play and control music on the speaker that heard the request.

        CALL THIS TOOL whenever the user asks to play music, a song, an artist,
        an album, or a playlist ("play some daft punk", "put on my focus
        playlist", "play one more time"), or to pause/resume/skip/stop music or
        change the volume ("pause the music", "next song", "volume 40").

        play: his own library first (playlists, then saved songs and artists,
            forgiving of misheard words), then Spotify search. Pass just the name;
            drop "from my list" / "from my liked songs".
        my_playlists: list library playlist names; optional query filters names.
        my_taste: read his library at a glance (top saved artists, playlists).
        play_next: resolve query like play and queue it next, without interrupting.
        add_to_queue: resolve query like play and append it to the queue.
        up_next: read the next item (HA exposes only one upcoming item).
        more_like_this: start similar-track radio from the track playing now.
        shuffle: query must be on or off.
        repeat: query must be off, one, or all.
        pause: pause playback.
        resume: resume playback.
        stop: stop playback.
        next: skip to the next track.
        previous: return to the previous track.
        volume: query is a level from 0 to 100.
        now_playing: read the current track and volume.

        Call for music requests. query is the user's name/artist/song/playlist.
        media_type optionally specifies track, artist, album, playlist, radio,
        audiobook or podcast for play/play_next/add_to_queue. Leave it empty
        unless explicit. target overrides the speaker ONLY if a room is named;
        otherwise leave empty and NEVER ask which speaker.
        "Play X then keep going with my liked songs": play X, then add_to_queue
        with query="my liked songs". Pick for a mood from my_playlists first.
        On the owner side only, use remember when the owner loves or hates a
        song, artist or kind of music, or skips the same thing twice. You may
        write your own DJ skill using the live skills tools as you learn.
        After music starts, name it in a few words of your own; vary the wording.
        At most one short sentence, no explaining, no offers, no questions;
        any question waits until the user asks. Never claim a change unless
        the tool reply confirms it.
        """
        op = operation.strip().lower().replace(" ", "_")

        if op in ("my_playlists", "my_taste"):
            refusal = _library_read_refusal(config)
            if refusal:
                return refusal

        if op == "my_playlists":
            try:
                items = _playlists()
            except (httpx.HTTPError, ValueError) as e:
                return f"Couldn't read your playlists ({e})."
            names = [str(p.get("name") or "?") for p in items
                     if _name_key(query) in _name_key(str(p.get("name") or ""))]
            return "Your playlists:\n" + "\n".join(names) if names else "No matching library playlists."

        if op == "my_taste":
            try:
                tracks, playlists = _saved("track"), _playlists()
            except (httpx.HTTPError, ValueError) as e:
                return f"Couldn't read his library ({e})."
            counts: dict[str, int] = {}
            for t in tracks:
                for a in t.get("artists") or []:
                    if a.get("name"):
                        counts[str(a["name"])] = counts.get(str(a["name"]), 0) + 1
            top = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:_TASTE_TOP]
            return (f"His library: {len(tracks)} saved songs, {len(playlists)} playlists.\n"
                    "Most-saved artists: " + (", ".join(f"{n} ({c})" for n, c in top) or "none") + ".\n"
                    "Playlists: " + (", ".join(str(p.get("name")) for p in playlists[:_TASTE_PLAYLISTS]) or "none")
                    + (" …" if len(playlists) > _TASTE_PLAYLISTS else "") + ".")

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

        if op in {"shuffle", "repeat"}:
            value = query.strip().lower()
            valid = {"on", "off"} if op == "shuffle" else {"off", "one", "all"}
            if value not in valid:
                return f"For {op}, query must be {'|'.join(sorted(valid))}."
            try:
                r = http.post(f"{base}/api/services/media_player/{op}_set", headers=headers,
                              json={"entity_id": player, op: value == "on" if op == "shuffle" else value})
                r.raise_for_status()
            except httpx.HTTPError as e:
                return f"Setting {op} FAILED ({e})."
            return f"{WRITE_OK_PREFIX} {op} set to {value} on {_friendly(player)}."

        if op == "up_next":
            try:
                r = http.post(f"{base}/api/services/music_assistant/get_queue?return_response",
                              headers=headers, json={"entity_id": player})
                r.raise_for_status()
                data = r.json()
                queue = data["service_response"][player]
                item = queue.get("next_item")
                if not item:
                    return f"No next item reported on {_friendly(player)}."
                name, _ = _item_line(item.get("media_item") or item)
            except (httpx.HTTPError, ValueError, KeyError, TypeError, AttributeError) as e:
                return f"Couldn't read the queue ({e})."
            return f"Up next: {name}. Home Assistant exposes only the next item, not the rest of the queue."

        if op == "more_like_this":
            try:
                r = http.get(f"{base}/api/states/{player}", headers=headers)
                r.raise_for_status()
                state = r.json()
                attrs = state.get("attributes") or {}
                uri = attrs.get("media_content_id") or attrs.get("uri")
                if state.get("state") != "playing" or not uri:
                    return "No current playing track with a usable URI; cannot start similar-track radio."
                r = http.post(f"{base}/api/services/music_assistant/play_media", headers=headers,
                              json={"entity_id": player, "media_id": uri, "media_type": "track", "radio_mode": True})
                r.raise_for_status()
            except (httpx.HTTPError, ValueError, TypeError, AttributeError) as e:
                return f"Starting similar-track radio FAILED ({e})."
            return f"{WRITE_OK_PREFIX} similar-track radio started on {_friendly(player)}. Keep the spoken confirmation to a few words."

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

        if op not in {"play", "play_next", "add_to_queue"}:
            return (
                f"Unknown operation '{operation}'. Valid operations: play, pause, "
                "resume, stop, next, previous, volume, now_playing, my_playlists, "
                "my_taste, play_next, add_to_queue, up_next, shuffle, repeat, more_like_this."
            )

        # ---- play: search Music Assistant, then queue the top hit ------------
        what = query.strip()
        if not what:
            return "Tell me what to play — an artist, song, album, or playlist name."
        wanted = media_type.strip().lower()
        results = saved = None
        try:
            # A library outage must not disable the existing global search.
            if wanted in {"", "playlist"}:
                hit = _library_match(what, _playlists())
                if hit:
                    results = {"playlists": [hit]}
            if results is None and wanted in {"", "track", "artist"}:
                saved = _saved_match(what, _saved("track") if wanted != "artist" else [],
                                     _saved("artist") if wanted != "track" else [])
        except (httpx.HTTPError, ValueError):
            results = saved = None
        if results is None and saved and saved[2]:
            results = {_TYPE_TO_CATEGORY[saved[1]]: [saved[0]]}
        if results is None:
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
                if not saved:
                    return f"Music search is unreachable right now ({e})."
                results = {}

        if wanted and wanted in _TYPE_TO_CATEGORY:
            order = ((_TYPE_TO_CATEGORY[wanted], wanted),)
        else:
            order = _CATEGORY_ORDER
        spotify_exact: list[tuple[dict, str]] = []
        if saved and not saved[2]:
            # A near name of his ("Imagined") yields to the exact name he said, if search has one it can
            # play ("Imagine": same categories, same name rule, any rank); otherwise his own saved song is
            # the better reading of a misheard ask.
            spotify_exact = [(hit, singular) for category, singular in order
                             for hit in (results.get(category) or [])
                             if isinstance(hit, dict) and _play_norm(hit.get("name")) == _play_norm(what)]
            if not spotify_exact:
                results = {_TYPE_TO_CATEGORY[saved[1]]: [saved[0]]}
        # Candidates in preference order. 2026-09-19 (bedroom, "play music from
        # Lilith Max"): the first TRACK hit won over the ARTIST the owner named,
        # radio mode asked Music Assistant for an "Endless Mix" the Spotify
        # provider could not fill, and the play failed with nothing to fall back
        # to. Two fixes: a result whose NAME is what the user said is promoted to
        # the front of the line whatever its category, and a failed play tries
        # the same item without radio mode, then the next candidate, before
        # giving up honestly.
        candidates: list[tuple[dict, str]] = []
        for category, singular in order:
            for hit in (results.get(category) or [])[:3]:
                candidates.append((hit, singular))
        exact = [c for c in candidates if _play_norm(c[0].get("name")) == _play_norm(what)]
        others = [c for c in candidates if c not in exact]
        # A yield (D52) puts the exact names first, then his near saved song, then the usual line,
        # so a failed exact play still falls through like any other ask.
        first = spotify_exact + ([(saved[0], saved[1])] if spotify_exact and saved else [])
        candidates = first + [c for c in exact + others if c not in first]
        if not candidates:
            return f"I couldn't find anything matching '{what}' on Spotify."
        tried: list[str] = []
        last_error = ""
        unshuffled: str | None = None
        for item, chosen_type in candidates[:4]:
            name, uri = _item_line(item)
            if not uri:
                continue
            attempts = [{"entity_id": player, "media_id": uri, "media_type": chosen_type}]
            if op != "play":
                attempts[0]["enqueue"] = "next" if op == "play_next" else "add"
            if chosen_type == "track" and op == "play" and unshuffled is None:
                unshuffled = _unshuffle(player)
            if chosen_type == "track" and op == "play":
                # Owner call (2026-07-18): asking for ONE song means "start music",
                # not "play 3:44 then silence" — radio mode keeps similar tracks
                # coming after the requested one. Albums/playlists/artists keep
                # their natural scope and end; "stop" works anytime. If the
                # provider cannot build the radio, the plain track still plays.
                attempts.insert(0, {**attempts[0], "radio_mode": True})
            for payload in attempts:
                try:
                    r = http.post(
                        f"{base}/api/services/music_assistant/play_media",
                        headers=headers,
                        json=payload,
                    )
                    r.raise_for_status()
                except httpx.HTTPError as e:
                    last_error = str(e)
                    tried.append(f"{name} ({chosen_type}{', radio' if payload.get('radio_mode') else ''})")
                    continue
                note = "" if op != "play" or payload.get("radio_mode") or chosen_type != "track" else " (just the song — similar-tracks radio wasn't available)"
                verb = "playing" if op == "play" else "queued next" if op == "play_next" else "queued"
                return (f"{WRITE_OK_PREFIX} {verb} {name} on {_friendly(player)}.{note}{unshuffled or ''} "
                        "Keep the spoken confirmation to a few words; no explanation, offers or questions.")
        return (
            f"Starting music on {_friendly(player)} FAILED — Home Assistant said: {last_error}. "
            f"Tried: {'; '.join(tried) or what}."
        )

    return music
