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

from aerys_v2.state import identity_from_config, is_voice_turn
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

# What his phone's media session accepts (the Home Assistant phone app's media command). It
# controls whatever app is already playing there; it cannot start a chosen song.
_PHONE_OPS = {"pause": "pause", "resume": "play", "stop": "stop", "next": "next", "previous": "previous"}
_PHONE_STATE_KEY = "playback_state_"
_PACKAGE = re.compile(r"[A-Za-z][A-Za-z0-9_]*(\.[A-Za-z0-9_]+)+")

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


def _sound_key(value: object) -> str:
    """A name with only its SPELLING variants folded (ie/y/ey/ee/ea -> i, ph -> f, ck -> k, doubled
    letters, a silent final e). Vowels stay, so different-sounding words stay apart: "Heart"/"Hurt",
    "Dreams"/"Drums", "Imagine"/"Imagined". 11/14: a saved artist, asked for by VOICE, went to a different
    Spotify artist spelled the common way ("Charlie"/"Charli"); speech-to-text can't hear spelling.
    Codex 11/15: an earlier consonants-only key was too loose."""
    words = []
    for word in re.findall(r"[a-z0-9]+", str(value or "").casefold()):
        if not word.isdigit():
            word = re.sub(r"(ie|ey|y)$", "i", word)
            word = word.replace("ph", "f").replace("ck", "k").replace("ee", "i").replace("ea", "i")
            word = re.sub(r"(.)\1+", r"\1", word)
            if len(word) > 3 and word.endswith("e"):
                word = word[:-1]
        words.append(word)
    return " ".join(words)


def _saved_keys(item: dict) -> list[str]:
    """Names a saved track or artist answers to: its name, and name + artists."""
    name = str(item.get("name") or "")
    artists = " ".join(str(a.get("name")) for a in item.get("artists") or [] if a.get("name"))
    return [_name_key(name)] + ([_name_key(f"{name} {artists}")] if artists else [])


def _saved_match(query: str, tracks: list[dict], artists: list[dict]) -> tuple[dict, str, str] | None:
    """(item, kind, how) from his saved songs and artists, or None; how is "exact", "sounds" (a near
    match spelled differently but sounding the same; only a VOICE ask may trust it) or "near". 11/14: a saved song whose title
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
            return exact[0][0], exact[0][1], "exact"
        songs = {recording(item) for item, kind in exact if kind == "track"}
        if len(songs) == 1:        # "X" is both an artist and one of his songs: the song he named
            item, kind = next(c for c in exact if c[1] == "track")
            return item, kind, "exact"
        return None
    ranked = sorted(((max(_close(want, key) for key in _saved_keys(item)), n)
                     for n, (item, _) in enumerate(pool)), key=lambda r: (-r[0], r[1]))
    if not ranked or ranked[0][0] < _MATCH_THRESHOLD:
        return None
    best = pool[ranked[0][1]]
    rival = next((score for score, n in ranked[1:]
                  if (pool[n][1], recording(pool[n][0])) != (best[1], recording(best[0]))), 0.0)
    if ranked[0][0] - rival < _MATCH_MARGIN:
        return None
    # A near match that sounds the same as what he said, only spelled differently.
    sounds_same = any(_sound_key(want) == _sound_key(key) for key in _saved_keys(best[0]))
    return best[0], best[1], "sounds" if sounds_same else "near"


def _artists_of(tracks: list[dict], artists: list[dict]) -> list[dict]:
    """The artists of his saved songs, as artist items, minus the ones already saved as artists. 11/16 (ticket
    T-21): "play Charli Morgan" (heard as "Charlie") found nothing of his, since she has 25+ saved songs but isn't a
    saved artist, and a Spotify search played a namesake. Her songs carry her library artist uri."""
    have = {_name_key(str(a.get("name") or "")) for a in artists}
    out, seen = [], set()
    for track in tracks:
        for a in track.get("artists") or []:
            if not isinstance(a, dict) or not a.get("uri") or not a.get("name"):
                continue
            key = _name_key(str(a["name"]))
            if key and key not in have and key not in seen:
                seen.add(key)
                out.append({"name": a["name"], "uri": a["uri"], "media_type": "artist"})
    return out


def _item_line(item: dict) -> tuple[str, str]:
    """(display name, uri) for a search result item — artist appended when known."""
    name = str(item.get("name") or "?")
    artists = item.get("artists") or []
    if artists:
        artist_names = ", ".join(str(a.get("name")) for a in artists if a.get("name"))
        if artist_names:
            name = f"{name} by {artist_names}"
    return name, str(item.get("uri") or "")


def is_phone_target(target: str) -> bool:
    """He named his phone as the place ("my phone", "on the phone"), not a room."""
    return "phone" in re.findall(r"[a-z]+", target.lower())


def phone_session(attributes: dict, op: str) -> tuple[str, str] | str:
    """(app package, its title) for the one media session [op] should act on, else a plain refusal.

    The phone's sensor carries one `playback_state_<package>` per app with a session. Pause, stop and
    skips act on the app that is playing; resume on the one that is paused. More than one candidate is
    asked about, never guessed."""
    states = {k[len(_PHONE_STATE_KEY):]: str(v) for k, v in (attributes or {}).items()
              if isinstance(k, str) and k.startswith(_PHONE_STATE_KEY) and _PACKAGE.fullmatch(k[len(_PHONE_STATE_KEY):])}
    playing = sorted(p for p, s in states.items() if s in ("Playing", "Buffering"))
    paused = sorted(p for p, s in states.items() if s == "Paused")
    if op == "resume":
        pick = paused if paused else []
        if not pick:
            return ("It's already playing on the phone." if playing
                    else "Nothing is paused on the phone that I can resume.")
    elif op in ("next", "previous"):
        pick = playing or paused
        if not pick:
            return "Nothing is playing on the phone right now."
    else:
        pick = playing
        if not pick:
            return "Nothing is playing on the phone right now."
    if len(pick) > 1:
        return f"More than one app has media on the phone ({', '.join(_app_name(p) for p in pick)}); which one?"
    return pick[0], str(attributes.get(f"title_{pick[0]}") or "")


def _app_name(package: str) -> str:
    """'com.spotify.music' -> 'spotify': enough to say which app, without a lookup table."""
    parts = [p for p in package.split(".") if p not in ("com", "org", "net", "android", "app", "apps", "music", "google")]
    return (parts[0] if parts else package).replace("_", " ")


def build_music_tool(
    *,
    base_url: str,
    token: str,
    config_entry_id: str,
    players: dict[str, str],
    default_player: str | None = None,
    client: httpx.Client | None = None,
    owner_person_id: str | None = None,
    dj_store=None,
    ma=None,
    phone_notify: str = "",
    phone_media_sensor: str = "",
):
    """Close over the config and return the LangChain tool object.

    players: originating device_id -> that room's Music Assistant media_player
    entity (HA_MUSIC_PLAYERS, same csv format as HA_SATELLITE_MAP). ALSO the
    complete set of speakers this tool may ever drive. default_player catches
    callers with no device_id (text/DM/CLI) — None means those turns get an
    honest "name a speaker" instead of a guess.
    phone_notify / phone_media_sensor: the Home Assistant notify service and media-session sensor
    of his phone. Both set = she may pause, resume, stop and skip what is playing there (target
    "phone"), for him only. Either empty = the phone is not a place she can reach.
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

    if dj_store is not None:
        dj_store.saved_tracks = lambda: _saved("track")

    def _mark_command(player):
        if dj_store is not None:
            try:
                dj_store.put("command:" + player, time.time())
            except Exception:
                log.warning("DJ command activity unavailable")

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

    def _his_alone(config: RunnableConfig) -> str | None:
        """None when it is him asking; else "not_him" or "unsure". WHO asks is checked first and on its
        own; a shared room is fine for a write (a like reveals nothing). Codex 11/15: checking the room
        first let his id with an unknown voice through wherever the room wasn't private."""
        from aerys_v2.hands_gate import OWNER_ASK, OWNER_TEXT, OWNER_VOICE, asker
        from aerys_v2.rooms import on_his_face

        conf = (config or {}).get("configurable") or {}
        ident = conf.get("identity") or {}
        thread = str(conf.get("thread_id") or "")
        if (not owner_person_id or str(ident.get("user_id") or "") != str(owner_person_id)
                or thread.startswith("kael:")):
            return "not_him"
        who = asker(ident, thread, owner_person_id)
        if who in (OWNER_TEXT, OWNER_VOICE) or (who == OWNER_ASK and (
                on_his_face(ident) or str(ident.get("platform") or "") == "wake")):
            return None
        return "unsure"

    def _his_write_refusal(config: RunnableConfig) -> str | None:
        """D53 writes (likes, library, playlists, shuffle) are his alone."""
        verdict = _his_alone(config)
        if verdict is None:
            return None
        if verdict == "not_him":
            return "Only he can change his music library."
        return "I'm not sure it's him asking, so I didn't change his library. Ask again."

    ma_players: dict[str, str] = {}

    def _ma_player_id(player: str) -> str | None:
        """Music Assistant's id for one of her allowed speakers (MA and HA share the display name)."""
        if player in ma_players:
            return ma_players[player]
        from aerys_v2.tools.music_mcp import parse

        r = http.get(f"{base}/api/states/{player}", headers=headers)
        r.raise_for_status()
        want = _name_key(str((r.json().get("attributes") or {}).get("friendly_name") or ""))
        found = [p for p in parse(ma.call("players_list_players")) or []
                 if isinstance(p, dict) and _name_key(str(p.get("name") or "")) == want and want]
        if len(found) != 1:
            return None
        ma_players[player] = str(found[0].get("player_id") or "")
        return ma_players[player] or None

    def _song_line(item: dict) -> str:
        artists = item.get("artists") or []
        names = [a if isinstance(a, str) else str((a or {}).get("name") or "") for a in artists]
        names = [n for n in names if n]
        return f"{item.get('name') or '?'}" + (f" by {', '.join(names)}" if names else "")

    def _track(query: str, player: str | None) -> tuple[str, str] | str:
        """(uri, "Title by Artist") for the song he named, else the one playing now; or a plain refusal.
        Exact only: a title (or title + artist) that matches one song. Never a near guess, never a like on
        the wrong song."""
        from aerys_v2.tools.music_mcp import parse

        title, artist = query.strip(), ""
        if " by " in title.lower():                          # "Example Song by Example Band"
            cut = title.lower().rindex(" by ")
            title, artist = title[:cut].strip(), title[cut + 4:].strip()
        if not title:
            if player is None:
                return "Tell me which song."
            r = http.get(f"{base}/api/states/{player}", headers=headers)
            r.raise_for_status()
            state = r.json()
            attrs = state.get("attributes") or {}
            title, artist = str(attrs.get("media_title") or ""), str(attrs.get("media_artist") or "")
            uri = str(attrs.get("media_content_id") or "")
            if not title or state.get("state") not in ("playing", "paused"):
                # Codex 11/15: an idle player keeps its last title; "like this" must not like a leftover.
                return f"Nothing is playing on {_friendly(player)}, so tell me which song."
            if "://" in uri and not uri.startswith("spotify_connect://") and state.get("state") == "playing":
                return uri, f"{title}" + (f" by {artist}" if artist else "")
        hits = [h for h in parse(ma.call("library_search_tracks", query=f"{title} {artist}".strip(), limit=10)) or []
                if isinstance(h, dict) and h.get("uri") and str(h.get("uri")).count("://") == 1
                and "/track/" in str(h.get("uri"))]
        # Writes match every word (Codex 11/15: "River" is not "The River"), unlike play's looser rule.
        want = _name_key(title)
        same_title = [h for h in hits if _name_key(str(h.get("name") or "")) == want]
        if artist:
            # Known artist: it must match. Never fall back to another artist's song with the same title.
            same_title = [h for h in same_title
                          if _name_key(artist) in {_name_key(str(a)) for a in h.get("artists") or []}]
        if not same_title:
            who = f" by {artist}" if artist else ""
            return f"I couldn't find exactly \"{title}\"{who}" + ("." if artist else "; tell me the artist too.")
        if len({tuple(sorted(_name_key(str(a)) for a in h.get("artists") or [])) for h in same_title}) > 1:
            return f"There's more than one \"{title}\"; which artist?"
        return str(same_title[0]["uri"]), _song_line(same_title[0])

    def _queue_brief(player: str, items: int) -> dict | None:
        from aerys_v2.tools.music_mcp import parse

        pid = _ma_player_id(player)
        if not pid:
            from aerys_v2.tools.music_mcp import MusicAssistantError
            raise MusicAssistantError(f"couldn't match {_friendly(player)} to a Music Assistant player")
        brief = parse(ma.call("queue_get_active_queue", player_id=pid, include_items=items, items_from_current=True))
        return brief if isinstance(brief, dict) else None

    def _friendly(entity: str) -> str:
        # 'media_player.office_satellite' -> 'office satellite' — good enough for
        # speech, no extra HTTP round-trip on the hot path.
        return entity.split(".", 1)[-1].replace("_", " ")

    def _resolve_target(target: str, config: RunnableConfig, quieting: bool = False) -> str:
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
        if quieting:
            # No room to go by (a wake-up, a typed message): stop the one allowed speaker that is
            # actually playing. 11/15: the stop was meant for the bedroom satellite, not the default.
            playing = []
            for entity in allowed:
                try:
                    r = http.get(f"{base}/api/states/{entity}", headers=headers)
                    if r.status_code == 200 and r.json().get("state") == "playing":
                        playing.append(entity)
                except (httpx.HTTPError, ValueError):
                    continue
            if len(playing) == 1:
                return playing[0]
            if len(playing) > 1:
                return ("Music is playing on more than one speaker "
                        f"({', '.join(_friendly(e) for e in playing)}); which one?")
        if default_player:
            return default_player
        options = ", ".join(_friendly(e) for e in allowed) or "(none configured)"
        return (
            "I can't tell which room you're in from here, and no default speaker "
            f"is configured — name one of: {options}."
        )

    def _phone(op: str, config: RunnableConfig) -> str:
        """Pause/resume/stop/skip on his phone, whatever app is playing. His alone; never a guess."""
        if not (phone_notify and phone_media_sensor):
            return "I can't reach his phone's media from here."
        if op not in _PHONE_OPS:
            return ("On the phone I can pause, resume, stop and skip what's already playing. "
                    "I can't start a song or change its volume there.")
        if identity_from_config(config).get("unattended"):
            return "Not done: controlling his phone needs him asking."     # never on her own wake-up
        verdict = _his_alone(config)
        if verdict == "not_him":
            return "Only he can control his phone."
        if verdict is not None:
            return "I'm not sure it's him asking, so I left his phone alone. Ask again."
        try:
            r = http.get(f"{base}/api/states/{phone_media_sensor}", headers=headers)
            r.raise_for_status()
            state = r.json()
        except (httpx.HTTPError, ValueError) as e:
            return f"I couldn't read the phone's media ({e}), so nothing was sent."
        if not isinstance(state, dict) or state.get("state") in (None, "unavailable", "unknown"):
            return "I can't see anything playing on the phone right now, so nothing was sent."
        found = phone_session(state.get("attributes") or {}, op)
        if isinstance(found, str):
            return found
        package, title = found
        try:
            r = http.post(
                f"{base}/api/services/notify/{phone_notify}",
                headers=headers,
                json={"message": "command_media",
                      "data": {"media_command": _PHONE_OPS[op], "media_package_name": package}},
            )
            r.raise_for_status()
        except httpx.HTTPError as e:
            return f"The {op} on the phone FAILED — Home Assistant said: {e}."
        about = f" ({_app_name(package)}" + (f": {title}" if title and op in ("pause", "resume", "stop") else "") + ")"
        # A push to the phone: Home Assistant accepted it, the phone hasn't confirmed anything.
        return f"{WRITE_OK_PREFIX} {op} sent to his phone{about}. It usually lands in a second or two."

    @tool
    def music(
        operation: str,
        query: str = "",
        media_type: str = "",
        target: str = "",
        playlist: str = "",
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
        dj: query on|off; owner only. Off lasts until 06:00 local.
        like: save a song to his favorites (query = the song; empty = the one playing).
        add_to_library: save a song to his library (same as like for which song).
        recently_played: what he's played lately (his private rooms only).
        queue: the songs coming up on this speaker (Music Assistant's queue).
        new_playlist: make a playlist; query is its name.
        add_to_playlist: add a song (query; empty = the one playing) to the playlist named in playlist.
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
        His phone: target="phone" with pause, resume, stop, next or previous controls
            whatever app is playing there (owner only). Nothing else works on the phone.

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

        if op == "dj":
            refusal = _library_read_refusal(config)
            if refusal:
                return refusal
            if identity_from_config(config).get("unattended"):
                return "Not done: DJ controls need the owner present."
            if query.strip().lower() not in {"on", "off"}:
                return "Use query on or off for DJ breaks."
            if dj_store is None:
                return "DJ controls are unavailable."
            from aerys_v2.dj import set_enabled
            try:
                return set_enabled(dj_store, query.strip().lower() == "on")
            except Exception:
                return "DJ control failed; the setting was not confirmed."

        if op in ("my_playlists", "my_taste", "recently_played", "queue"):
            refusal = _library_read_refusal(config)
            if refusal:
                return refusal

        if op in ("like", "add_to_library", "recently_played", "queue", "new_playlist", "add_to_playlist") and ma is None:
            return "That needs Music Assistant's connection, which isn't set up."

        if op in ("like", "add_to_library", "new_playlist", "add_to_playlist"):
            refusal = _his_write_refusal(config)
            if refusal:
                return refusal

        if op == "recently_played":
            from aerys_v2.tools.music_mcp import MusicAssistantError, parse
            try:
                items = parse(ma.call("metadata_recently_played", limit=10)) or []
            except MusicAssistantError as e:
                return f"Couldn't read what he's played lately ({e})."
            lines = [_song_line(i) for i in items if isinstance(i, dict) and "/track/" in str(i.get("uri") or "")]
            return "Recently played:\n" + "\n".join(lines) if lines else "Nothing played recently."

        if op == "new_playlist":
            from aerys_v2.tools.music_mcp import MusicAssistantError
            name = query.strip()
            if not name:
                return "Tell me what to call the playlist."
            try:
                ma.call("playlists_create_playlist", name=name)
            except MusicAssistantError as e:
                return f"Making the playlist FAILED ({e})."
            library_cache.pop("playlist", None)          # so add_to_playlist finds it right away
            return f"{WRITE_OK_PREFIX} made the playlist {name}."

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

        if is_phone_target(target):
            return _phone(op, config)

        player = _resolve_target(target, config, quieting=op in ("pause", "stop"))
        if not player.startswith("media_player."):
            return player  # honest resolution error, relayed as-is
        # D54 (Codex 11/15): a run he isn't in never changes a speaker on his autonomy blocklist, checked on
        # the speaker the target RESOLVED to (named, or picked because it was the one playing).
        from aerys_v2.wakeups import NOT_ON_ITS_OWN, autonomous, autonomy_blocked, house_write

        if (autonomous(identity_from_config(config)) and house_write("music", {"operation": op})
                and (autonomy_blocked(player) or autonomy_blocked(_friendly(player)))):
            return NOT_ON_ITS_OWN

        if op in ("like", "add_to_library", "add_to_playlist"):
            from aerys_v2.tools.music_mcp import MusicAssistantError
            try:
                found = _track(query, player)
                if isinstance(found, str):
                    return found
                uri, song = found
                if op == "add_to_playlist":
                    # A write needs one EXACT playlist name (Codex 11/15: "Road Trip 2026" must never land
                    # in "Road Trip 2025"); play's forgiving match is for listening, not for changing.
                    named = [p for p in _playlists() if p.get("uri") and playlist.strip()
                             and _name_key(str(p.get("name") or "")) == _name_key(playlist)]
                    target_list = named[0] if len(named) == 1 else None
                    if not target_list:
                        return f"I couldn't find a playlist called \"{playlist}\" exactly."
                    ma.call("playlists_add_track", playlist_id=str(target_list["uri"]), track_uri=uri)
                    return f"{WRITE_OK_PREFIX} added {song} to {target_list.get('name')}."
                ma.call("media_add_to_favorites" if op == "like" else "media_add_to_library", uri=uri)
            except (MusicAssistantError, httpx.HTTPError, ValueError) as e:
                return f"That didn't save ({e})."
            return f"{WRITE_OK_PREFIX} {'liked' if op == 'like' else 'added to his library:'} {song}."

        if op == "queue":
            from aerys_v2.tools.music_mcp import MusicAssistantError
            try:
                brief = _queue_brief(player, 10)
            except (MusicAssistantError, httpx.HTTPError, ValueError) as e:
                return f"Couldn't read the queue ({e})."
            items = (brief or {}).get("items") or []
            names = [_song_line(i.get("media_item") or i) if isinstance(i, dict) else str(i) for i in items]
            if not names:
                return (f"Nothing is queued on {_friendly(player)} through Music Assistant "
                        "(music from the Spotify app plays outside her queue).")
            return f"Queue on {_friendly(player)}:\n" + "\n".join(names)

        # ---- transport controls (plain media_player services) ----------------
        if op in _TRANSPORT_OPS:
            service = _TRANSPORT_OPS[op]
            if op in {"next", "previous", "resume"}:
                _mark_command(player)
            try:
                r = http.post(
                    f"{base}/api/services/media_player/{service}",
                    headers=headers,
                    json={"entity_id": player},
                )
                r.raise_for_status()
            except httpx.HTTPError as e:
                return f"The {op} on {_friendly(player)} FAILED — Home Assistant said: {e}."
            if op in {"next", "previous", "resume"}:
                _mark_command(player)
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
            if ma is not None:
                # 11/14: HA refuses media_player.shuffle_set on the office player; Music Assistant's own
                # queue control works. No queue (the Spotify app is playing) is said plainly.
                from aerys_v2.tools.music_mcp import MusicAssistantError
                try:
                    brief = _queue_brief(player, 0)
                    if not brief or not brief.get("queue_id"):
                        return (f"Nothing is playing through Music Assistant on {_friendly(player)}; "
                                f"if it's the Spotify app, {op} is in the app.")
                    if op == "shuffle":
                        ma.call("queue_set_shuffle", queue_id=str(brief["queue_id"]), enabled=value == "on")
                    else:
                        ma.call("queue_set_repeat", queue_id=str(brief["queue_id"]), repeat_mode=value)
                except (MusicAssistantError, httpx.HTTPError, ValueError) as e:
                    return f"Setting {op} FAILED ({e})."
                return f"{WRITE_OK_PREFIX} {op} set to {value} on {_friendly(player)}."
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
                "my_taste, play_next, add_to_queue, up_next, shuffle, repeat, more_like_this, dj, "
                "like, add_to_library, recently_played, queue, new_playlist, add_to_playlist."
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
                tracks = _saved("track")
                artists = _saved("artist") if wanted != "track" else []
                if wanted != "track":                     # an artist he only has songs of counts too (T-21)
                    artists = artists + _artists_of(tracks, artists)
                saved = _saved_match(what, tracks if wanted != "artist" else [], artists)
        except (httpx.HTTPError, ValueError):
            results = saved = None
        if saved:
            # Exact spelling is his; "sounds the same" is his only when he SAID it (voice can't spell).
            # A typed ask keeps its spelling: a near saved name still yields to an exact catalog one.
            conf = (config or {}).get("configurable") or {}
            voice = is_voice_turn(identity_from_config(config), conf.get("thread_id", ""))
            trusted = saved[2] == "exact" or (saved[2] == "sounds" and voice)
            saved = (saved[0], saved[1], trusted)
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
        for item, chosen_type in candidates[:4]:
            name, uri = _item_line(item)
            if not uri:
                continue
            attempts = [{"entity_id": player, "media_id": uri, "media_type": chosen_type}]
            if op != "play":
                attempts[0]["enqueue"] = "next" if op == "play_next" else "add"
            one_song = chosen_type == "track" and op == "play"
            if one_song:
                # 11/14: a radio queue (the song + similar ones) is shuffled when the player's shuffle is
                # on, and this player can't turn shuffle off (HA refuses shuffle_set), so a radio pick
                # played instead of the song he named. The song goes in ALONE first (replace: nothing
                # to shuffle), then its radio after it, below.
                attempts[0]["enqueue"] = "replace"
            for payload in attempts:
                if op == "play":
                    _mark_command(player)
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
                if op == "play":
                    _mark_command(player)
                note = ""
                if one_song:
                    # Owner call (2026-07-18): asking for ONE song means "start music", not "play 3:44
                    # then silence": similar tracks follow it. If the provider can't build the radio,
                    # the song still plays.
                    try:
                        r = http.post(f"{base}/api/services/music_assistant/play_media", headers=headers,
                                      json={**payload, "enqueue": "add", "radio_mode": True})
                        r.raise_for_status()
                    except httpx.HTTPError:
                        note = " (just the song — similar-tracks radio wasn't available)"
                verb = "playing" if op == "play" else "queued next" if op == "play_next" else "queued"
                return (f"{WRITE_OK_PREFIX} {verb} {name} on {_friendly(player)}.{note} "
                        "Keep the spoken confirmation to a few words; no explanation, offers or questions.")
        return (
            f"Starting music on {_friendly(player)} FAILED — Home Assistant said: {last_error}. "
            f"Tried: {'; '.join(tried) or what}."
        )

    return music
