"""home_control — Tool v1: the Brain's first native write capability.

n8n mapping: this replaces workflow 07-01 "HA Action: Play Music (owner-gated)"
as the pattern-setter — an HTTP Request node to Home Assistant's REST API, done
as a LangChain tool the action subgraph can call. The function is literally
named `home_control` because of the V1 toolWorkflow lesson: the tool name the
LLM sees MUST match what prompts call it, or the model hallucinates having
called it.

Three safety layers, outermost first:

1. CANARY ALLOWLIST — writes are refused (honest error string back to the
   model, so it can tell the caller the truth) for any entity not in
   ha_canary_entities. Reads are unrestricted: looking at a state can't break
   anything. This is the same crawl-walk-run gate as the V1 owner-gated action.
2. OUTBOX-INLINE — every write that reaches HA is recorded in v2_outbox
   (INSERT intent as 'executing' -> call HA -> UPDATE receipt/status), the
   write-ahead pattern from db/migrations/001. A crash mid-call leaves an
   'executing' row the sweeper can reconcile — never a silent mystery write.
3. HONEST FAILURE — HA unreachable / 4xx / refused all come back as plain
   error strings the model must relay. Never raise out of the tool: an
   exception inside a ToolNode kills the whole action turn (the V1
   failed-webhook-kills-execution outage mode, again).
"""

import json
import logging
import re
import time
from datetime import datetime, timedelta, timezone
import uuid
from urllib.parse import quote
from typing import Any, Callable

import httpx
from langchain_core.tools import tool
from langgraph.prebuilt import ToolRuntime

log = logging.getLogger(__name__)

WRITE_OPS = frozenset({"turn_on", "turn_off", "toggle", "set_brightness", "set_color"})
# HA color modes that mean "this light can take a color" (color_temp alone cannot).
COLOR_MODES = frozenset({"xy", "rgb", "hs", "rgbw", "rgbww"})
_HEX_COLOR = re.compile(r"^#?([0-9a-fA-F]{6})$")

# The ONLY string prefix a successful write returns — service.py's silent-success
# rule keys on it (a fast turn whose every tool note starts with this = the device
# visibly changed = skip the spoken follow-up). Change it here and nowhere else.
WRITE_OK_PREFIX = "Done:"
# A write HA accepted where a read-back shows the device was ALREADY in the
# requested state. Deliberately NOT WRITE_OK_PREFIX: nothing visibly changed, so
# the spoken follow-up should say so ("it was already off") instead of staying
# silent. (2026-09-04: replaces the old "NO state change — verify" note that
# cost the model a whole extra round-trip to read the state itself.)
NOOP_OK_PREFIX = "OK (already there):"
# Words that carry no target information in a room/device name.
_NAME_STOPWORDS = frozenset({"the", "all", "of", "in", "my", "every", "please"})
# Words that name a KIND of thing, not WHICH thing. A name made only of these
# ("the lights", "switches") would match every controllable entity — a
# house-wide write from an under-specified request. It becomes a question.
_GENERIC_TERMS = frozenset({"light", "lights", "lamp", "lamps", "switch", "switches", "plug", "plugs"})
# v1 scope: only domains where a misfire is an annoyance, not a hazard.
# (No locks, no covers, no climate — those arrive with confirmation semantics.)
WRITABLE_DOMAINS = frozenset({"light", "switch"})

# The connection seam: a zero-arg callable returning a DB connection usable as
# a context manager (psycopg.connect in prod, a fake in tests). None = no
# database_url = the outbox layer is simply absent (spike/dev boxes).
ConnFactory = Callable[[], Any]

# search_entities knobs: enough matches to disambiguate, few enough to not
# blow the tool-message budget; long states (weather blobs) get elided.
SEARCH_LIMIT = 30  # 9/04: 15 cut the garage off a "temperature" search (8 rooms x temp+humidity + Stickies)
STATE_TRUNCATE_AT = 60
# HA's "nothing home" states — noise in a discovery listing, filtered unless
# they're literally all we found (then honesty beats tidiness: show them).
DEAD_STATES = frozenset({"unavailable", "unknown"})

# --- search_entities: finding a thing by what it IS, not only what it is CALLED ---
# 2026-09-24, observed live: asked whether anything was open, she answered "everything
# is shut except Kitchen Window 2" while three sliding doors stood open. The matcher was
# pure substring, so "door"/"window" scored ZERO against `binary_sensor.kitchen_slider`
# — the sliders were structurally unfindable — and "doors"/"windows" scored zero against
# everything, because the plural is not a substring of the singular. Worse, the one thing
# she DID name is a retired sensor that reads permanently-open. Wrong in both directions.
#
# So a term may now also hit on the entity's device_class. The map is deliberately small
# and house-shaped; it is not a thesaurus.
CLASS_SYNONYMS: dict[str, frozenset[str]] = {
    "door": frozenset({"door", "garage_door", "opening"}),
    "slider": frozenset({"door", "opening"}),
    "sliding": frozenset({"door", "opening"}),
    "window": frozenset({"window", "opening"}),
    "opening": frozenset({"door", "window", "opening", "garage_door"}),
    "entry": frozenset({"door", "opening"}),
    # "is anything open?" names no entity at all — it names a STATE of a kind of thing.
    "open": frozenset({"door", "window", "opening", "garage_door"}),
    "opened": frozenset({"door", "window", "opening", "garage_door"}),
    "closed": frozenset({"door", "window", "opening", "garage_door"}),
    "shut": frozenset({"door", "window", "opening", "garage_door"}),
}

# Hardware that is down or removed reads PERMANENTLY OPEN. Home Assistant holds the one
# authoritative list (configuration.yaml), so this tool reads it rather than keeping a
# second copy that would drift. Absent group = no exclusions, behaviour unchanged.
# The class map above only reaches things Home Assistant has CLASSIFIED. The owner's three
# sliding doors carry device_class None — they are named "Kitchen Slider" and nothing
# else — so a class-only fix still could not find them. (I shipped exactly that half-fix
# first, because my test invented a slider WITH device_class="door" instead of using the
# shape the house actually has. Test against the real thing.) So a query word also
# expands to the OTHER WORDS a house calls that thing.
NAME_SYNONYMS: dict[str, frozenset[str]] = {
    "door": frozenset({"door", "slider", "sliding", "slide"}),
    "slider": frozenset({"slider", "sliding", "door"}),
    "sliding": frozenset({"slider", "sliding", "door"}),
    "window": frozenset({"window"}),
    # "open"/"shut" name no object, so they lean entirely on CLASS_SYNONYMS — except
    # for the sliders, which have no class at all and would otherwise stay invisible.
    "open": frozenset({"slider", "sliding"}),
    "opened": frozenset({"slider", "sliding"}),
    "closed": frozenset({"slider", "sliding"}),
    "shut": frozenset({"slider", "sliding"}),
}

# Asking "what is open?" matches ~100 entities in this house, and SEARCH_LIMIT cuts at
# 30 ranked by term-hits then entity_id — i.e. ALPHABETICALLY. Live check: she named the
# kitchen slider and missed the living-room and office ones purely because k sorts before
# l and o. A state word in the query is therefore a ranking signal: if you asked what is
# OPEN, the things that ARE open outrank the things that are not.
STATE_WORDS: dict[str, str] = {
    "open": "on", "opened": "on",
    "closed": "off", "shut": "off", "close": "off",
}

# Entities that carry an opening-ish device_class but are NOT openings.
# Chris's eMotion Air presence sensors expose their FACE BUTTON on an "opening"
# channel: each press flips the contact, either direction counts, and only the
# TRANSITION means anything — the state itself is meaningless. Home Assistant has no
# device class for "button", and twelve of his automations trigger on these, so the
# label stays correct-for-him in HA and the correction lives here (his call, 9/24:
# "i dont want to mess with them much since they are working pretty well").
#
# They are excluded from CLASS-based matching only. Asked for by name they still come
# back — this stops them answering "what is open?", it does not hide them.
# Words that ask about openings rather than name a thing. A deny-listed entity must not
# answer these BY NAME either — "open" is a substring of "…_opening", which is exactly how
# a face button would sneak back into "what is open?" through the name path.
OPENING_WORDS: frozenset[str] = frozenset(
    set(CLASS_SYNONYMS) | set(NAME_SYNONYMS) | set(STATE_WORDS)
)

NOT_OPENINGS_PREFIX_SUFFIX: tuple[tuple[str, str], ...] = (
    ("binary_sensor.emotion_air_", "_opening"),
)


def _is_not_really_an_opening(entity: str) -> bool:
    return any(entity.startswith(p) and entity.endswith(sfx)
               for p, sfx in NOT_OPENINGS_PREFIX_SUFFIX)


RETIRED_GROUP = "group.adt_retired_contacts"
RETIRED_NOTE = " (retired sensor — reads open because the hardware is gone, NOT a real open)"


def _term_variants(term: str) -> set[str]:
    """The term plus a naive singular, so "doors" reaches "door"."""
    out = {term}
    if term.endswith("es") and len(term) > 3:
        out.add(term[:-2])
    if term.endswith("s") and len(term) > 2:
        out.add(term[:-1])
    return out

# A transient transport failure (connection refused / DNS / timeout — a network
# blip or HA mid-restart) gets ONE quick retry after a short backoff before we
# fall back to the honest "unreachable" message. httpx.TransportError is the
# connect/timeout family ONLY — it excludes HTTPStatusError, so a real 404/500
# is never retried (that's a genuine answer, not a blip). Reads only: writes
# keep their single-shot + outbox path so a retry can never double-fire a toggle.
_READ_RETRY_BACKOFF_S = 0.6
# Read-back after a write HA reported no state change for: these lights (Tuya-class
# cloud devices) update their HA state a beat AFTER the service call returns —
# observed 9/04: the same lights answered "Done" for two bulbs and an empty
# changed-list for the other two in one command. So the read-back waits, and
# retries once, before calling a command dropped. Only the no-change path pays.
_VERIFY_DELAYS_S = (0.8, 1.5)


def _get_with_retry(http: httpx.Client, url: str, headers: dict) -> httpx.Response:
    """GET, retrying ONCE on a transient transport error (never on a status).

    On a momentary HA outage (today's SSD-reboot blip is the canonical case) the
    second attempt after a short backoff usually succeeds — so a brief restart
    no longer surfaces as an unreachable/degraded turn. If both attempts fail the
    final error propagates to the caller's existing graceful except-block.
    """
    try:
        return http.get(url, headers=headers)
    except httpx.TransportError:
        time.sleep(_READ_RETRY_BACKOFF_S)
        return http.get(url, headers=headers)  # final attempt; may raise → caught upstream


def _room_of(entity_id: str) -> str:
    """Best-effort place label from an entity id: the object-id words before the
    first generic term / trailing number. light.sunroom_light_1 -> "sunroom"."""
    obj = entity_id.split(".", 1)[-1]
    words = [w for w in obj.split("_") if w and not w.isdigit() and w not in _GENERIC_TERMS]
    return " ".join(w.rstrip("s") for w in words)


def room_aliases(spec: str) -> dict[str, tuple[str, ...]]:
    """Parse HA_ROOM_ALIASES: "office=switch.a,switch.b;office lights=switch.a" ->
    {"office": ("switch.a","switch.b"), "office lights": ("switch.a",)}. Names are
    lower-cased with spaces collapsed; a malformed part is skipped, never fatal."""
    out: dict[str, tuple[str, ...]] = {}
    for part in (spec or "").split(";"):
        if "=" not in part:
            continue
        name, ids = part.split("=", 1)
        key = " ".join(name.lower().split())
        ents = tuple(e.strip() for e in ids.split(",") if e.strip() and "." in e)
        if key and ents:
            out[key] = ents
    return out


def canary_set(csv: str) -> frozenset[str]:
    """Parse the HA_CANARY_ENTITIES csv into the allowlist set ('' -> empty)."""
    return frozenset(e.strip() for e in csv.split(",") if e.strip())


def resolve_targets(raw: str, op: str, canary_entities: frozenset[str],
                    aliases: dict[str, tuple[str, ...]] | None = None) -> tuple[list[str], str | None]:
    """WHAT the call is about: exact id, comma list of ids, or a room/device
    NAME matched against the controllable set. Returns (targets, problem).

    Why (2026-09-04 trace): "dim the sunroom lights by half" cost five model
    round-trips because the tool took ONE exact id per call — the model
    guessed ids, hit the allowlist refusal, then wrote four lights one by one.
    A name resolves here, against the allowlist, into ONE call."""
    raw = raw.strip()
    if "," in raw:
        ids = [e.strip() for e in raw.split(",") if e.strip()]
        bad = [e for e in ids if "." not in e]
        if bad:
            return [], (
                f"'{', '.join(bad)}' aren't entity ids — a list must be exact ids "
                "like light.a, light.b (or give ONE room/device name instead)."
            )
        return ids, None
    if "." in raw:
        return [raw], None
    # Owner-defined names first (HA_ROOM_ALIASES): "office" is whatever Chris said
    # it is, not whatever the word happens to match.
    key = " ".join(raw.lower().split())
    for prefix in ("the ", "all the ", "all of the ", "all "):
        if key.startswith(prefix):
            key = key[len(prefix):]
    if aliases and key in aliases:
        ids = list(aliases[key])
        log.info("home_control alias %r -> %s", raw, ids)
        return ids, None
    terms = [
        w for w in re.split(r"[\s_\-]+", raw.lower()) if w and w not in _NAME_STOPWORDS
    ]
    if not terms:
        return [], "home_control needs an entity id or a room/device name."
    if all(w in _GENERIC_TERMS for w in terms):
        rooms = sorted({_room_of(e) for e in canary_entities} - {""})
        return [], (
            f"Which ones? '{raw}' names a kind of device, not a place. Ask the user "
            f"ONE short question naming the choices: {', '.join(rooms) or 'none configured'}."
        )

    # WHOLE-TOKEN matching (Codex review 9/04): "office" must not reach
    # light.office_closet_1 by substring. Tokens of the object id; a term
    # matches a token exactly (plural-stripped), and "sun room" also matches
    # "sunroom" via the joined non-generic terms.
    joined = "".join(w for w in terms if w not in _GENERIC_TERMS)

    def hit(eid: str) -> bool:
        # Plural-insensitive on BOTH sides (Gemini review 9/05): "fan" must
        # reach switch.fans and "lights" must reach light.sunroom_light_1.
        toks = {tok.rstrip("s") for tok in re.split(r"[._\-\s]+", eid.lower())}
        ok = lambda w: w.rstrip("s") in toks  # noqa: E731
        return all(ok(w) for w in terms) or (
            bool(joined) and joined.rstrip("s") in toks
            and all(ok(w) for w in terms if w in _GENERIC_TERMS)
        )

    matches = sorted(e for e in canary_entities if hit(e))
    if matches:
        log.info("home_control resolved %r -> %s", raw, matches)
        return matches, None
    allowed = ", ".join(sorted(canary_entities)) or "(none configured)"
    if op == "get_state":
        return [], (
            f"No controllable entity matches '{raw}'. To read other devices "
            "(cars, sensors, phones) call search_entities with that name to get "
            f"the exact id, then get_state with it. Entities I can control: {allowed}."
        )
    return [], (
        f"Refused: nothing I may control matches '{raw}'. "
        f"The entities I may control are: {allowed}."
    )


def device_target_choices(canary_entities: frozenset[str], *, limit: int = 250,
                          aliases: dict[str, tuple[str, ...]] | None = None) -> dict[str, str]:
    """Choice criteria for a decision model: every ROOM and every DEVICE on the
    allowlist, keyed by a NAME resolve_targets() already understands.

    Phase 4 (2026-09-19): the decision model cannot generate an entity id, so it
    picks a name from this list and the tool resolves it exactly as it would a
    name the specialist typed. Rooms first ("sunroom" -> all four sunroom
    lights), then single devices ("office fan"). Keys are checked by tests to
    resolve to a non-empty target list.
    """
    out: dict[str, str] = {}
    # Owner-defined names come first and win over the derived room names.
    for name, ids in (aliases or {}).items():
        out[name] = f"the owner's name for: {', '.join(ids)}"
    rooms: dict[str, list[str]] = {}
    for e in sorted(canary_entities):
        r = _room_of(e)
        if r:
            rooms.setdefault(r, []).append(e)
    for room, ids in rooms.items():
        if room in out:
            continue
        targets, problem = resolve_targets(room, "turn_on", canary_entities, aliases)
        if targets and not problem:
            out[room] = f"everything in the {room}: {', '.join(targets)}"
    for e in sorted(canary_entities):
        obj = e.split(".", 1)[-1]
        name = " ".join(w for w in obj.split("_") if w)
        if name in out:
            continue
        targets, problem = resolve_targets(name, "turn_on", canary_entities, aliases)
        if targets == [e] and not problem:
            out[name] = e
        if len(out) >= limit:
            break
    return out


#: D47 (11/14): her read of a sensor is the whole reading, the way Kael reads it: units, class and the
#: attributes, not just the bare state ("172" with no "min" was all she got). Light/display noise and
#: oversized values stay out. D57: authorized whereabouts reads bypass this filtering; other readings
#: still drop attributes shaped like locations, so position data never rides along accidentally.
_DETAIL_SKIP = frozenset({"friendly_name", "icon", "entity_picture", "supported_features", "supported_color_modes",
                          "rgb_color", "hs_color", "xy_color", "brightness", "color_mode", "attribution",
                          "restored", "editable", "id", "options", "effect"})
DETAIL_MAX = 1500                 # characters of attributes per entity
FIELD_MAX = 64                    # unit, class, timestamps: short by nature, clipped anyway (Codex 11/14)
READ_MAX = 12000                  # one get_state/history answer, all targets together
_WHEREABOUTS_DOMAINS = ("device_tracker.", "person.", "zone.")
HISTORY_MAX_HOURS = 168
HISTORY_POINTS = 48
WHEREABOUTS_POINTS = 200
#: HA's own entity-id grammar. Anything else ("sensor.a/../sensor.b") is refused before any check or
#: URL: a path segment must never walk past the private-reading prefixes (Codex 11/14 blocker).
ENTITY_ID = re.compile(r"^[a-z0-9_]+\.[a-z0-9_]+$")
#: attribute keys that carry a position, whatever their value looks like (gps_latitude, address, …),
#: matched by whole word parts so health keys stay (sleep_latency, platelet_count; Codex 11/14)
_PLACE_WORDS = frozenset({"lat", "latitude", "lon", "lng", "long", "longitude", "gps", "coord", "coords",
                          "coordinates", "address", "street", "postal", "zip", "zipcode", "locality",
                          "thoroughfare", "geo", "geocoded", "location", "altitude", "accuracy", "course",
                          "heading", "place", "city", "country", "region", "zone", "position"})


#: long place words caught anywhere inside a key part ("gpslatitude", "homeaddress"): specific enough
#: that no health key holds one (Codex 11/14: GPSLatitude)
_PLACE_INSIDE = ("latitude", "longitude", "address", "location", "coordinate", "geocode", "altitude",
                 "thoroughfare", "locality", "postal")


def _place_key(key: str) -> bool:
    split = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", key)          # GPSLatitude -> GPS_Latitude
    split = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", split)             # homeAddress -> home_Address
    words = re.split(r"[^a-z0-9]+", split.lower())
    return any(w in _PLACE_WORDS or any(p in w for p in _PLACE_INSIDE) for w in words)


PRECISE_PLACE = ("That's his precise location. It's only read through where_is_chris_nearby, when he asks "
                 "for something near him.")

HIS_LOCATION_REFUSAL = ("Where he is or has been is his to ask: I can only tell him "
                 "(typed, or a voice I'm sure is his).")


def whereabouts(entity_id: str) -> bool:
    return entity_id.startswith(_WHEREABOUTS_DOMAINS) or "geocoded" in entity_id or "location" in entity_id


def precise_place(entity_id: str) -> bool:
    """A sensor whose state IS a street address or position (the phone's geocoded location)."""
    return not entity_id.startswith(_WHEREABOUTS_DOMAINS) and whereabouts(entity_id)


def _clip(value, limit: int = FIELD_MAX):
    return value[:limit] if isinstance(value, str) else value


def private_entity(entity_id: str, prefixes: tuple[str, ...]) -> bool:
    """One of his own phone/watch readings (HA_PRIVATE_ENTITIES prefixes, kept out of this public repo)."""
    return any(entity_id.startswith(p) for p in prefixes if p)


def reading_details(entity_id: str, data: dict) -> dict:
    from aerys_v2.services.location_guard import location_shaped

    attrs = data.get("attributes")
    attrs = attrs if isinstance(attrs, dict) else {}
    out: dict = {}
    for key in ("unit_of_measurement", "device_class", "state_class"):
        if attrs.get(key) is not None:
            out[key] = _clip(attrs[key] if isinstance(attrs[key], str) else str(attrs[key]))
    if data.get("last_updated"):
        out["last_updated"] = _clip(str(data["last_updated"]))
    if whereabouts(entity_id):
        return out
    extra: dict = {}
    size = 0
    for key, value in attrs.items():
        key = str(key)
        if key in _DETAIL_SKIP or key in out or key.endswith("_list") or _place_key(key):
            continue
        # the whole value is checked before any clipping, so a position can't hide past the cut
        if location_shaped(f"{key}: {value}") or location_shaped(str(value)):
            continue
        if isinstance(value, (list, tuple)):
            value = list(value)[:10]
        elif isinstance(value, dict) or not isinstance(value, (str, int, float, bool, type(None))):
            value = str(value)[:200]
        elif isinstance(value, str):
            value = value[:200]
        key = key[:FIELD_MAX]
        size += len(key) + len(str(value))
        if size > DETAIL_MAX:
            extra["more"] = "further attributes left out (size)"
            break
        extra[key] = value
    if extra:
        out["attributes"] = extra
    return out


class PositionHistory(Exception):
    """A history whose readings are positions: refused whole (where he's been is never a trail)."""


def history_points(rows: list, now: datetime) -> list[list[str]]:
    """HA history rows -> [[local time, state], ...]: dead and repeated states dropped, thinned evenly to
    HISTORY_POINTS (the first and last kept)."""
    from aerys_v2.hands_gate import EASTERN

    from aerys_v2.services.location_guard import location_shaped

    for row in rows:
        if not isinstance(row, dict):
            raise ValueError(f"a history row isn't an object: {str(row)[:40]!r}")
        if row.get("state") is None:
            raise ValueError("a history row has no state")
        if location_shaped(str(row["state"])):
            raise PositionHistory()
    points: list[list[str]] = []
    last = None
    for row in rows:
        state = _clip(str(row.get("state")), 120)
        if state in DEAD_STATES or state == last:
            continue
        at = datetime.fromisoformat(str(row.get("last_changed")).replace("Z", "+00:00")).astimezone(EASTERN)
        label = at.strftime("%I:%M %p").lstrip("0")
        if at.date() != now.astimezone(EASTERN).date():
            label = f"{at.strftime('%a')} {label}"
        points.append([label, state])
        last = state
    if len(points) > HISTORY_POINTS:
        step = (len(points) - 1) / (HISTORY_POINTS - 1)
        points = [points[round(i * step)] for i in range(HISTORY_POINTS)]
    return points


class TooManyLocationStops(ValueError):
    """The history cannot fit without dropping distinct stops."""


def whereabouts_points(rows: list, now: datetime) -> list[list[str]]:
    """Keep all state transitions; sample only redundant same-state GPS updates."""
    from aerys_v2.hands_gate import EASTERN

    points = []
    required = set()
    previous = None
    for row in rows:
        if not isinstance(row, dict) or row.get("state") is None:
            raise ValueError("a history row has no state")
        state = str(row["state"])
        attrs = row.get("attributes") or {}
        coords = []
        if attrs.get("latitude") is not None and attrs.get("longitude") is not None:
            coords = [f"{float(attrs['latitude']):.4f}, {float(attrs['longitude']):.4f}"]
        at = datetime.fromisoformat(str(row.get("last_updated") or row.get("last_changed"))
                                   .replace("Z", "+00:00")).astimezone(EASTERN)
        label = at.strftime("%I:%M %p").lstrip("0")
        if at.date() != now.astimezone(EASTERN).date():
            label = f"{at.strftime('%a')} {label}"
        if state != previous:
            required.add(len(points))
        elif not coords:
            continue
        points.append([label, state, *coords])
        previous = state
    if len(required) > WHEREABOUTS_POINTS:
        raise TooManyLocationStops()
    if len(points) > WHEREABOUTS_POINTS:
        optional = [i for i in range(len(points)) if i not in required]
        slots = WHEREABOUTS_POINTS - len(required)
        # Include the latest GPS point when space permits.
        chosen = {optional[round((i + 1) * len(optional) / slots) - 1] for i in range(slots)}
        points = [points[i] for i in sorted(required | chosen)]
    return points


def _mark_location_read(data, owner: str | None) -> None:
    """Mark only place values; generic states and sensor metadata are not places."""
    from aerys_v2.services.location_guard import mark_location_turn

    def place(value, *, state=False):
        if value is None or isinstance(value, bool):
            return []
        text = str(value).strip()
        if len(text) < 3 or (state and text.lower() in {
                "home", "not_home", "away", "unknown", "unavailable"}):
            return []
        numeric = re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", text)
        if numeric and (state or len(text) < 4):
            return []
        return [text]

    def attributes(value):
        if isinstance(value, dict):
            for key, child in value.items():
                if isinstance(child, (dict, list, tuple)):
                    yield from attributes(child)
                elif _place_key(str(key)):
                    yield from place(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                yield from attributes(child)

    places = []
    rows = data if isinstance(data, list) else [data]
    for row in rows:
        if not isinstance(row, dict):
            continue
        places.extend(place(row.get("state"), state=True))
        attrs = row.get("attributes") or {}
        places.extend(attributes(attrs))
        if isinstance(attrs, dict) and attrs.get("latitude") is not None and attrs.get("longitude") is not None:
            try:
                places.append(f"{float(attrs['latitude']):.4f}, {float(attrs['longitude']):.4f}")
            except (ValueError, TypeError):
                pass
    mark_location_turn(values=places, person_id=owner)


def build_home_control_tool(
    *,
    base_url: str,
    token: str,
    canary_entities: frozenset[str],
    client: httpx.Client | None = None,
    conn_factory: ConnFactory | None = None,
    aliases: dict[str, tuple[str, ...]] | None = None,
    private_entities: tuple[str, ...] = (),
    his_places: frozenset[str] = frozenset(),
    owner_person_id: str | None = None,
    clock: Callable[[], datetime] | None = None,
):
    """Close over the config and return the LangChain tool object.

    Everything injectable, same seam philosophy as the checkpointer: tests pass
    an httpx.Client on a MockTransport and a fake conn_factory; --serve passes
    the real things. The tool NEVER reads Settings — construction knows config,
    behavior doesn't.
    """
    base = base_url.rstrip("/")
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    http = client or httpx.Client(timeout=10.0)

    def _his_location(entity: str) -> bool:
        return entity in his_places or (private_entity(entity, private_entities) and whereabouts(entity))

    def _outbox_open(payload: dict) -> int | None:
        """INSERT the write-ahead intent row; returns its id (None = outbox off/failed).

        Lease check lives here because it must be decided BEFORE the intent row
        is written — the exception marker travels IN the payload, auditable.

        ── ONE-ARMED-WRITER EXCEPTION (deliberate, bounded, beta only) ─────────
        The v2_writer_lease rule (design doc 2026-07-02-turns-outbox-spine.md,
        cross-review #11) says a write capability REFUSES unless it holds the
        lease for its kind. During the voice beta, ha_write's lease still says
        'n8n' — but the callers reaching THIS tool are satellite-scoped voice
        threads that n8n never serves, so a double-fire is structurally
        impossible for this slice. Owner-ratified: execute anyway, but mark the
        payload {"lease_exception": "beta-canary"} so every such write is
        queryable (SELECT .. WHERE payload ? 'lease_exception') and the
        exception dies loudly when the lease flips to 'brain' and the marker
        stops appearing. This is the ONLY capability allowed to bend the rule.
        """
        if conn_factory is None:
            return None
        try:
            with conn_factory() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT holder FROM v2_writer_lease WHERE kind = 'ha_write'"
                    )
                    row = cur.fetchone()
                    holder = row[0] if row else None
                    if holder != "brain":
                        payload["lease_exception"] = "beta-canary"
                    cur.execute(
                        "INSERT INTO v2_outbox (kind, payload, idempotency_key, status) "
                        "VALUES ('ha_write', %s::jsonb, %s, 'executing') RETURNING id",
                        (json.dumps(payload), str(uuid.uuid4())),
                    )
                    return cur.fetchone()[0]
        except Exception:
            # Audit trouble must not cost the turn (the graceful contract) —
            # but an unaudited write is a real event, so it logs loudly.
            log.warning("outbox INSERT failed — HA write proceeds unaudited", exc_info=True)
            return None

    def _outbox_close(
        outbox_id: int | None, status: str, receipt: dict | None = None, error: str | None = None
    ) -> None:
        """UPDATE the intent row with what actually happened (receipt = evidence).

        Separate short connection from _outbox_open on purpose: holding a conn
        across the HA HTTP call buys nothing, and a crash between the two
        leaves exactly the 'executing' row the sweeper contract expects.
        """
        if outbox_id is None:
            return
        try:
            with conn_factory() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "UPDATE v2_outbox SET status = %s, receipt = %s::jsonb, "
                        "last_error = %s, attempts = attempts + 1, updated_at = now() "
                        "WHERE id = %s",
                        (
                            status,
                            json.dumps(receipt) if receipt is not None else None,
                            error,
                            outbox_id,
                        ),
                    )
        except Exception:
            log.warning("outbox UPDATE failed for row %s", outbox_id, exc_info=True)

    def _resolve_targets(raw: str, op: str) -> tuple[list[str], str | None]:
        return resolve_targets(raw, op, canary_entities, aliases)

    def _read_state(entity: str, full: bool = False) -> dict | str:
        """GET one entity's state: a dict, or an honest error STRING. full: the whole reading (her own
        get_state, D47); the write path's read-backs only need the state."""
        try:
            r = _get_with_retry(http, f"{base}/api/states/{entity}", headers)
            if r.status_code == 404:
                return f"Home Assistant has no entity named {entity}."
            r.raise_for_status()
            data = r.json()
            if not isinstance(data, dict):
                # Never raise inside a ToolNode (kills the turn): an odd body is an
                # honest string, and a read-back caller treats it as "unverified".
                return f"Home Assistant returned an unexpected reply for {entity}."
            attrs = data.get("attributes") or {}
            out = {
                "entity_id": entity,
                "state": data.get("state"),
                "friendly_name": attrs.get("friendly_name"),
            }
            if data.get("last_changed"):
                # When the state last CHANGED (HA's clock). The write path uses it to tell
                # "changed because of this command, a beat late" from "already there";
                # the model may read it too ("changed 3 minutes ago").
                out["last_changed"] = data["last_changed"]
            if str(data.get("state")) in DEAD_STATES:
                out["note"] = "not reporting to Home Assistant right now — do not retry; tell the user"
            # Lights report brightness 0-255; the model (and Chris) think in
            # percent — translate here so relative dimming math ("down by
            # half") never has to know about the 255 scale.
            if attrs.get("brightness") is not None:
                out["brightness_pct"] = round(attrs["brightness"] / 255 * 100)
            # Color (2026-09-19, her own gap report: "I can't control the color"):
            # say what the bulb CAN do and what it is showing, so "make it red"
            # can be refused honestly on a white-only bulb and answered on a
            # color one.
            if attrs.get("supported_color_modes") is not None:
                out["supported_color_modes"] = list(attrs["supported_color_modes"])
                out["color_capable"] = bool(COLOR_MODES & set(attrs["supported_color_modes"]))
            if attrs.get("rgb_color") is not None:
                out["rgb_color"] = list(attrs["rgb_color"])
            if full:
                out.update(reading_details(entity, data))
                if _his_location(entity):
                    _mark_location_read(data, owner_person_id)
                    out["state"] = data.get("state")
                    out["attributes"] = attrs
                    return out
                from aerys_v2.services.location_guard import location_shaped

                if location_shaped(str(out.get("state"))):
                    out["state"] = "(a position: read only through where_is_chris_nearby)"
                else:
                    out["state"] = _clip(out.get("state"), 255)
            return out
        except httpx.HTTPError as e:
            if _his_location(entity):
                return f"Home Assistant returned an unreadable history for {entity}."
            return f"Home Assistant is unreachable right now ({e})."
        except (ValueError, TypeError, AttributeError) as e:
            # Malformed JSON / odd attribute types: honest string, never a raise.
            if _his_location(entity):
                return f"Home Assistant returned an unreadable history for {entity}."
            return f"Home Assistant returned an unreadable reply for {entity} ({e})."

    def _private_refusal(targets: list[str], runtime) -> str | None:
        """D47: his own phone and watch readings only come to him: his DM or typed line, a voice she's
        sure is his, his glasses or his own wake-up, never a room others read."""
        location = any(_his_location(e) for e in targets)
        if not location and not any(private_entity(e, private_entities) for e in targets):
            return None
        from aerys_v2.rooms import NOT_HIM, SHARED, his_private_read, runtime_identity

        identity, thread = runtime_identity(runtime)
        # A second household voice joins this gate once it is enrolled (Stage 5).
        why = his_private_read(identity, thread, owner_person_id)
        if why is None:
            return None
        if location:
            if why == SHARED:
                return "Where he is or has been is private: others can read this room, so ask me in a DM or by voice."
            if why == NOT_HIM:
                return HIS_LOCATION_REFUSAL
            return "Where he is or has been is private: I'm not sure it's him, so type it to me or ask when I know your voice."
        if why == SHARED:
            return ("Couldn't read that here: it's one of his own phone or watch readings, and others can read "
                    "this room. Ask me in a DM, or by voice.")
        if why == NOT_HIM:
            return "Couldn't read that: it's one of his own phone or watch readings, for him only."
        return ("Couldn't read that: his phone and watch readings are for him, and I'm not sure it's him. "
                "Type it to me, or ask again when I know your voice.")

    def _history(entity: str, hours: int | None) -> dict | str:
        """How a reading moved over the last few hours: changes only, in his clock, thinned to a page."""
        location = _his_location(entity)
        if whereabouts(entity) and not location:
            return f"No history for {entity}: where someone's been is never read back as a trail."
        span = max(1, min(int(24 if hours is None else hours), HISTORY_MAX_HOURS))
        now = (clock or (lambda: datetime.now(timezone.utc)))()
        start = quote((now - timedelta(hours=span)).isoformat())
        # HA's history window ends one day after its start unless told otherwise: without end_time a
        # week-long ask returned only its oldest day (found 11/14, live on his sensors).
        end = quote(now.isoformat())
        flags = "" if location else "&minimal_response&no_attributes&significant_changes_only"
        try:
            r = _get_with_retry(
                http, f"{base}/api/history/period/{start}?filter_entity_id={quote(entity)}&end_time={end}"
                      f"{flags}", headers)
            r.raise_for_status()
            body = r.json()
            if not isinstance(body, list) or (body and not isinstance(body[0], list)):
                return f"Home Assistant returned an unreadable history for {entity}."
            rows = body[0] if body else []
            if location:
                _mark_location_read(rows, owner_person_id)
            current = _read_state(entity, full=True)
        except httpx.HTTPError as e:
            if _his_location(entity):
                return f"Home Assistant returned an unreadable history for {entity}."
            return f"Home Assistant is unreachable right now ({e})."
        except (ValueError, TypeError, IndexError) as e:
            if location:
                return f"Home Assistant returned an unreadable history for {entity}."
            return f"Home Assistant returned an unreadable history for {entity} ({e})."
        try:
            points = whereabouts_points(rows, now) if location else history_points(rows, now)
        except TooManyLocationStops:
            return "Too many state changes: ask for a shorter history span."
        except PositionHistory:
            return f"No history for {entity}: its readings are positions, and where he's been is never a trail."
        except (ValueError, TypeError, AttributeError) as e:
            if location:
                return f"Home Assistant returned an unreadable history for {entity}."
            return f"Home Assistant returned an unreadable history for {entity} ({e})."
        out = {"entity_id": entity, "hours": span, "changes": points}
        if isinstance(current, dict):
            for key in ("friendly_name", "unit_of_measurement", "device_class"):
                if current.get(key) is not None:
                    out[key] = current[key]
        if not points:
            out["note"] = "no recorded changes in that span"
        return out

    def _desired_phrase(op: str, pct: int | None, color: str | None = None) -> str:
        if op == "turn_off":
            return "off"
        if color is not None and pct is not None:
            return f"on, {color} at {pct}%"
        if color is not None:
            return f"on and {color}"
        if pct is not None:
            return f"on at {pct}%"
        return "on"

    def _verify_after_write(
        entities: list[str], op: str, pct: int | None, since: datetime | None = None
    ) -> tuple[list[str], list[str], list[str]]:
        """Read back entities HA reported no new state for: (already_there, not_applied, done_late).

        A verification that can FAIL: it reads the device, it does not trust the
        command. toggle has no knowable target state, so it can never verify.

        done_late (2026-09-20, the dressers): Zigbee/cloud devices report their new
        state a beat AFTER HA answers the command, so HA's changed-list omits them
        and the read-back finds them in the target state. That is NOT "already
        there" — if the entity's last_changed is at or after the command was sent,
        THIS command changed it. Aerys told Chris the dressers "were already off"
        after turning them off; this is the fix."""
        if op == "toggle":
            return [], list(entities), []

        def check(e: str) -> tuple[bool, bool]:
            """(in target state, changed at/after `since`)."""
            st = _read_state(e)
            if not isinstance(st, dict):
                return False, False
            want_on = op != "turn_off"
            ok = (st.get("state") == "on") == want_on
            if ok and want_on and pct is not None:
                ok = abs(int(st.get("brightness_pct") or 0) - pct) <= 2
            late = False
            if ok and since is not None and st.get("last_changed"):
                try:
                    changed_at = datetime.fromisoformat(str(st["last_changed"]).replace("Z", "+00:00"))
                    late = changed_at >= since
                except ValueError:
                    late = False
            return ok, late

        already: list[str] = []
        done_late: list[str] = []
        pending = list(entities)
        for delay in _VERIFY_DELAYS_S:
            if not pending:
                break
            time.sleep(delay)
            still = []
            for e in pending:
                ok, late = check(e)
                if ok and late:
                    done_late.append(e)
                elif ok:
                    already.append(e)
                else:
                    still.append(e)
            pending = still
        return already, pending, done_late

    @tool
    def home_control(
        operation: str, entity_id: str, brightness_pct: int | None = None,
        color: str | None = None, hours: int | None = None, runtime: ToolRuntime = None,
    ) -> str:
        """Control or inspect the smart home via Home Assistant.

        CALL THIS TOOL whenever the user asks to turn something on or off,
        toggle a device, dim/brighten a light, change a light's COLOR, or asks
        whether a light/switch is currently on or what color it is.

        operation: one of "get_state", "history", "turn_on", "turn_off", "toggle",
        "set_brightness", "set_color". get_state returns the whole reading: state,
        unit, device class, when it last updated and its attributes. history shows
        how a reading moved over the last `hours` (default 24, up to 168): its
        changes in his clock, for trends like how a night's sleep or a heart rate
        went. His own phone and watch readings answer only to him in his own rooms
        (a DM, his voice, his glasses); elsewhere the tool says so.
        entity_id: WHAT to control — any ONE of:
          - an exact entity id ("light.office_lamp", "switch.desk_fan");
          - a comma-separated list of exact ids (one call controls them all);
          - a ROOM or DEVICE NAME ("sunroom lights", "office", "sun room"):
            resolved against the controllable entities, and EVERY match is
            controlled in this ONE call. Use this for "the sunroom lights"
            style requests — do NOT call search_entities and do NOT loop
            over lights one by one.
          To READ something you can't control (a car, a sensor, a phone), call
          search_entities first to get the exact id, then get_state with it.
          Never invent an id.
        brightness_pct: 1-100, lights only. For an ABSOLUTE target ("half",
        "50%", "to 20 percent") call set_brightness with the number DIRECTLY —
        no get_state first. Only a RELATIVE ask ("a bit dimmer", "a little
        brighter", "down by half from where it is") needs get_state first: it
        returns the current brightness_pct; do the math, then set_brightness.
        For "off"/0%, use turn_off.
        color: a color NAME ("red", "warm white", "sky blue") or a hex code
        ("#ff0000"), lights only. "Turn the sunroom red" / "make it blue" =
        set_color with color="red"/"blue" — it turns the light on if needed.
        turn_on also accepts color (and brightness_pct) in the same call. A
        white-only bulb refuses honestly; get_state shows color_capable.

        For him, get_state/history on his phone's location entities returns where he is
        and where he's been: history is his stops through the day, with times.
        An address can be looked up with search_web to see what's there.
        These readings are live only, never save whereabouts to memory.

        get_state works on any entity. Writes only work on lights and switches
        on the beta allowlist — if the tool refuses, tell the user honestly and
        exactly why; NEVER claim a device changed state unless this tool said
        so. A reply starting "OK (already there)" means the device was already
        in the requested state — say that plainly.
        """
        op = operation.strip().lower()
        targets, problem = _resolve_targets(entity_id, "get_state" if op == "history" else op)
        if problem:
            return problem
        label = ", ".join(targets)

        # ---- reads: anything, except his own phone/watch readings outside his own rooms (D47) ----
        if op in ("get_state", "history"):
            bad = [e for e in targets if not ENTITY_ID.match(e)]
            if bad:
                return f"'{', '.join(b[:80] for b in bad)}' isn't an entity id (like sensor.office_temperature)."
            refusal = _private_refusal(targets, runtime)
            if refusal:
                return refusal
            if any(precise_place(e) and not _his_location(e) for e in targets):
                return PRECISE_PLACE
        if op in ("get_state", "history"):
            results = [_read_state(e, full=True) if op == "get_state" else _history(e, hours) for e in targets]
            text = (results[0] if isinstance(results[0], str) else json.dumps(results[0])) if len(results) == 1 \
                else json.dumps(results)
            if len(text) > READ_MAX:
                if any(_his_location(e) for e in targets):
                    return "Location reading is too large: ask for a shorter history span or fewer entities."
                return text[:READ_MAX] + " … (cut: ask for fewer things at once)"
            return text

        if op not in WRITE_OPS:
            return (
                f"Unknown operation '{operation}'. "
                "Valid operations: get_state, history, turn_on, turn_off, toggle, set_brightness, set_color."
            )

        # ---- writes: domain gate, then canary gate — over EVERY target -------
        domains = {e.split(".", 1)[0] for e in targets}
        bad = sorted(d for d in domains if d not in WRITABLE_DOMAINS)
        if bad:
            return (
                f"Refused: writes to '{bad[0]}' entities aren't enabled yet — "
                "only lights and switches can be controlled in this beta."
            )
        not_allowed = [e for e in targets if e not in canary_entities]
        if not_allowed:
            log.info("home_control refused %s: not on the beta write allowlist", not_allowed)
            # Honest refusal STRING back to the model — never an exception, and
            # never a lie. The model relays this so the caller learns the truth.
            allowed = ", ".join(sorted(canary_entities)) or "(none configured)"
            verb = "is" if len(not_allowed) == 1 else "are"
            return (
                f"Refused: {', '.join(not_allowed)} {verb} not on the beta write allowlist. "
                f"I can read state, but the only entities I may control are: {allowed}."
            )

        # ---- brightness: validate BEFORE any outbox row exists ---------------
        # set_brightness is sugar over HA's light/turn_on + brightness_pct —
        # there is no set_brightness service. Both ops share the same checks.
        wants_pct = op == "set_brightness" or (op == "turn_on" and brightness_pct is not None)
        if wants_pct:
            if op == "set_brightness" and brightness_pct is None:
                return (
                    "set_brightness needs brightness_pct (1-100). For a relative "
                    "change, call get_state first to read the current level."
                )
            if brightness_pct is not None and not 1 <= brightness_pct <= 100:
                return (
                    f"Refused: brightness_pct must be 1-100 (got {brightness_pct}). "
                    "For 0% / off, use turn_off instead."
                )
            if "light" not in domains:
                return (
                    f"Refused: brightness only applies to lights — {label} "
                    f"{'is a' if len(targets) == 1 else 'are'} {'/'.join(sorted(domains))}. "
                    "Use turn_on/turn_off for it."
                )
        # ---- color: validate BEFORE any outbox row exists --------------------
        # set_color is sugar over light/turn_on + color_name / rgb_color — there is
        # no set_color service. A color only makes sense on a light that has a
        # color mode; color_temp-only bulbs are refused with the reason.
        wants_color = op == "set_color" or (op == "turn_on" and color is not None)
        color_body: dict[str, Any] = {}
        color_label: str | None = None
        if wants_color:
            if op == "set_color" and (color is None or not str(color).strip()):
                return "set_color needs a color: a name like 'red' or a hex code like '#ff0000'."
            if "light" not in domains:
                return (
                    f"Refused: color only applies to lights — {label} "
                    f"{'is a' if len(targets) == 1 else 'are'} {'/'.join(sorted(domains))}."
                )
            raw = str(color).strip()
            m = _HEX_COLOR.match(raw)
            if m:
                h = m.group(1)
                color_body = {"rgb_color": [int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)]}
                color_label = f"#{h.lower()}"
            else:
                # HA accepts CSS color names; normalise "sky blue" -> "skyblue".
                name = re.sub(r"[\s_-]+", "", raw.lower())
                if not name.isalpha():
                    return (
                        f"Refused: '{raw}' isn't a color I can send — use a color name "
                        "('red', 'sky blue') or a hex code ('#ff0000')."
                    )
                color_body = {"color_name": name}
                color_label = raw.lower()
            # Capability check on the lights that will be written: a bulb that
            # reports its modes and has no color mode is refused honestly; a bulb
            # that reports nothing is tried (HA decides).
            no_color = []
            for e in targets:
                if not e.startswith("light."):
                    continue
                st = _read_state(e)
                if isinstance(st, dict) and st.get("color_capable") is False:
                    no_color.append(e)
            if no_color:
                return (
                    f"Refused: {', '.join(no_color)} "
                    f"{'does' if len(no_color) == 1 else 'do'} not support color — "
                    "white / color-temperature only. I can turn it on, off, or dim it."
                )
        # A name can span lights AND switches ("living room"). One HA call per
        # domain (Gemini review 9/04: a refusal here was a dead end — the model
        # can only address them by that same name). Brightness applies to the
        # lights; switches in the same name are left alone and SAID so.
        groups = {d: [e for e in targets if e.split(".", 1)[0] == d] for d in sorted(domains)}
        skipped: list[str] = []
        if wants_pct or wants_color:
            skipped = [e for d, es in groups.items() if d != "light" for e in es]
            groups = {"light": groups["light"]}
        did = op
        if color_label is not None:
            did = f"{op} {color_label}"
        if brightness_pct is not None:
            did = f"{did} {brightness_pct}%"
        done: list[str] = []
        already: list[str] = []
        dropped: list[str] = []
        for domain, group in groups.items():
            service = "turn_on" if (wants_pct or wants_color) else op
            entity_field: Any = group if len(group) > 1 else group[0]
            body: dict[str, Any] = {"entity_id": entity_field}
            if wants_pct:
                body["brightness_pct"] = brightness_pct
            if wants_color:
                body.update(color_body)
            # ---- the audited write: intent -> HA -> receipt ------------------
            payload = {"operation": op, "entity_id": entity_field, "domain": domain}
            if brightness_pct is not None:
                payload["brightness_pct"] = brightness_pct
            if color_label is not None:
                payload["color"] = color_label
            if skipped:
                # The resolved intent included these; they were left alone on
                # purpose (no brightness on a switch). Audit the whole intent.
                payload["skipped"] = skipped
            outbox_id = _outbox_open(payload)
            try:
                # HA REST: POST /api/services/<domain>/<service> — the same
                # endpoint the V1 HTTP Request node hit, minus the workflow
                # around it. ONE call per domain: HA takes an entity_id list.
                sent_at = datetime.now(timezone.utc)  # a read-back changed at/after this = OUR doing
                r = http.post(
                    f"{base}/api/services/{domain}/{service}",
                    headers=headers,
                    json=body,
                )
                r.raise_for_status()
            except httpx.HTTPError as e:
                _outbox_close(outbox_id, "failed", error=str(e))
                return f"The {op} on {', '.join(group)} FAILED — Home Assistant said: {e}."
            # Receipt with evidence, not a bare ok (hands-contract rule 4): HA
            # returns the list of states the call changed — that's the proof.
            try:
                changed = r.json()
            except ValueError:
                changed = None
            changed_ids = (
                {c.get("entity_id") for c in changed if isinstance(c, dict)}
                if isinstance(changed, list) else set()
            )
            unverified = [e for e in group if e not in changed_ids]
            done += [e for e in group if e in changed_ids]
            # HA accepted the command (200) but reported no new state for some
            # target. Two honest readings: it was ALREADY in that state, or the
            # device dropped the command (observed live 2026-08-10 with a Tuya
            # light: "on at 30%" while it stayed dark). Instead of handing the
            # model a "go verify" note (a whole extra round-trip), READ THE
            # DEVICE BACK here and say which it was.
            g_already, g_dropped, g_late = (
                _verify_after_write(unverified, op, brightness_pct, since=sent_at) if unverified else ([], [], [])
            )
            already += g_already
            dropped += g_dropped
            done += g_late  # changed a beat after HA answered — this command did it
            _outbox_close(
                outbox_id, "succeeded",
                receipt={
                    "status_code": r.status_code, "changed": changed,
                    "already_there": g_already, "not_applied": g_dropped,
                },
            )
        note = f" ({', '.join(skipped)} can't take a brightness — left as is.)" if skipped else ""
        if dropped:
            return (
                f"Sent {did} to {label}; Home Assistant accepted it (200) but "
                f"{', '.join(dropped)} reported NO state change and a read-back does not "
                "show the requested state — the device may not have applied it (cloud "
                "lights sometimes drop commands). Do not tell the user it definitely "
                f"happened.{note}"
            )
        if not done:
            verb = "was" if len(already) == 1 else "were"
            return (
                f"{NOOP_OK_PREFIX} {', '.join(already)} {verb} already "
                f"{_desired_phrase(op, brightness_pct)} — nothing to change.{note}"
            )
        extra = f" {', '.join(already)} already {_desired_phrase(op, brightness_pct)}." if already else ""
        return f"{WRITE_OK_PREFIX} {did} sent to {', '.join(done)} (HA responded 200).{extra}{note}"

    # D54 (Codex 11/15): the autonomy blocklist is checked on what a call RESOLVES to (a room or alias can
    # reach a blocked device), so the resolver rides on the tool for wakeups.wake_house_refusal.
    from aerys_v2.wakeups import RESOLVER_ATTR

    object.__setattr__(home_control, RESOLVER_ATTR, _resolve_targets)
    return home_control


def build_search_entities_tool(
    *,
    base_url: str,
    token: str,
    client: httpx.Client | None = None,
    his_places: frozenset[str] = frozenset(),
    private_entities: tuple[str, ...] = (),
    owner_person_id: str | None = None,
):
    """Close over the config and return the READ-ONLY entity discovery tool.

    Why this exists (observed live, 2026-07-03): home_control's get_state needs
    an EXACT entity id. Asked "what is jolteon's charge level?", the model
    guessed ids, got 404s, and had to ask the user — a discovery gap, not a
    reasoning gap. This tool is a fuzzy index over GET /api/states so the model
    can find the id itself. No canary allowlist on purpose: the allowlist gates
    WRITES; listing names and states can't break anything.
    """
    base = base_url.rstrip("/")
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    http = client or httpx.Client(timeout=10.0)

    @tool
    def search_entities(query: str, runtime: ToolRuntime = None) -> str:
        """Find Home Assistant entity ids by name. READ-ONLY — never changes anything.

        Call this BEFORE home_control's get_state when you lack the exact
        entity id of something you are READING (to CONTROL lights/switches,
        give home_control the room or device name directly — no search) —
        i.e. whenever the user names a device colloquially:
        a car name ("jolteon"), a sensor ("office temperature"), a person's phone,
        his watch or phone health readings ("heart rate", "steps", "sleep"),
        a nickname. NEVER guess an entity id — guesses 404; searching works.

        query: one or more words to match, e.g. "jolteon battery" or
        "office lamp". Matches entity ids and friendly names, case-insensitive,
        and ALSO matches what a thing IS: "door", "window", "open" and their
        plurals find anything Home Assistant classes as a door/window/opening,
        whatever it happens to be named (a sliding door called "Kitchen Slider"
        is found by "door"). A line ending in "(retired sensor ...)" is dead
        hardware stuck reading open — never report it as something being open.
        Returns up to 30 matches, one per line:
        "entity_id | friendly_name | state" (units included when known).
        Then call home_control get_state with the exact entity_id you picked,
        or answer directly from the state shown here.
        """
        terms = [t for t in query.lower().split() if t]
        if not terms:
            return "search_entities needs at least one word to search for."
        try:
            r = _get_with_retry(http, f"{base}/api/states", headers)
            r.raise_for_status()
            states = r.json()
        except (httpx.HTTPError, ValueError) as e:
            # Honest failure string, never a raise — same ToolNode contract
            # as home_control (an exception kills the whole action turn).
            return f"Home Assistant is unreachable right now ({e})."

        # The one authoritative retired list, read from the same payload — no second call.
        retired: frozenset[str] = frozenset()
        for item in states:
            if item.get("entity_id") == RETIRED_GROUP:
                members = (item.get("attributes") or {}).get("entity_id") or ()
                retired = frozenset(str(m) for m in members)
                break

        # Precompute, per term, the device_classes AND the alternative names it matches.
        term_classes: list[frozenset[str]] = []
        term_words: list[frozenset[str]] = []
        for t in terms:
            classes: set[str] = set()
            words: set[str] = set()
            for v in _term_variants(t):
                classes |= CLASS_SYNONYMS.get(v, frozenset())
                words |= NAME_SYNONYMS.get(v, frozenset())
                words.add(v)
            term_classes.append(frozenset(classes))
            term_words.append(frozenset(words))

        # A state word in the query ("open"/"shut") says which side to float to the top.
        wanted_states = {STATE_WORDS[v] for t in terms for v in _term_variants(t)
                         if v in STATE_WORDS}

        scored: list[tuple[int, str, str, dict]] = []
        for item in states:
            entity = item.get("entity_id") or ""
            attrs = item.get("attributes") or {}
            friendly = str(attrs.get("friendly_name") or "")
            haystack = f"{entity} {friendly}".lower()
            dev_class = str(attrs.get("device_class") or "").lower()
            hits = 0
            counts = not _is_not_really_an_opening(entity)
            for t, classes, words in zip(terms, term_classes, term_words):
                # A button on an opening channel answers to its own name, never to a
                # question about openings — by class OR by an unlucky substring.
                if not counts and _term_variants(t) <= OPENING_WORDS:
                    continue
                by_name = any(w in haystack for w in words)
                by_class = counts and bool(dev_class) and dev_class in classes
                if by_name or by_class:
                    hits += 1
            if hits:
                # Rank boost, never a filter: asking what is open must not hide a
                # closed door you named directly.
                if wanted_states and str(item.get("state")) in wanted_states:
                    hits += 1
                scored.append((hits, entity, friendly, item))
        if not scored:
            return f"No Home Assistant entities match '{query}'."

        # Rank: most query terms matched first, then entity_id for stability.
        scored.sort(key=lambda s: (-s[0], s[1]))
        live = [s for s in scored if str(s[3].get("state")) not in DEAD_STATES]
        if scored and not live:
            # 9/04: showing the dead matches sent a small model into a read/search
            # loop until the recursion rail. Say what it means, in the tool's voice.
            return (
                f"All {len(scored)} entities matching '{query}' are unavailable/unknown "
                "— that integration is NOT reporting to Home Assistant right now. Do "
                "not search or read again; tell the user it is not reporting."
            )
        picked = (live or scored)[:SEARCH_LIMIT]

        # D47: his own phone/watch readings show their names (she still needs the id) but not their
        # values outside his own rooms; get_state says why.
        from aerys_v2.rooms import his_private_read, runtime_identity

        identity, thread = runtime_identity(runtime)
        hide = his_private_read(identity, thread, owner_person_id) is not None
        lines = []
        for _, entity, friendly, item in picked:
            his_location = entity in his_places or (private_entity(entity, private_entities) and whereabouts(entity))
            if his_location:
                if hide:
                    lines.append(f"{entity} | {friendly or '(no name)'} | (his location: not here)")
                    continue
                _mark_location_read(item, owner_person_id)
            elif precise_place(entity):
                lines.append(f"{entity} | {friendly or '(no name)'} | (a place: not shown)")
                continue
            if hide and private_entity(entity, private_entities):
                lines.append(f"{entity} | {friendly or '(no name)'} | (his own reading: not here)")
                continue
            state = str(item.get("state"))
            if len(state) > STATE_TRUNCATE_AT:
                state = state[:STATE_TRUNCATE_AT] + "…"
            # Battery/temperature sensors are meaningless without the unit —
            # "78" vs "78 %" is exactly the EV6 charge-level use case.
            unit = (item.get("attributes") or {}).get("unit_of_measurement")
            if unit:
                state = f"{state} {unit}"
            if entity in retired:
                state = f"{state}{RETIRED_NOTE}"
            lines.append(f"{entity} | {friendly or '(no name)'} | {state}")
        return "\n".join(lines)

    return search_entities
