"""Which rooms of the house are showing occupancy right now — ambient, owner-only.

Her gap #101 (self-reported, 2026-09-20): "add arrival/departure presence events +
room-level presence so Aerys can respond in the context of the room Chris is actually
in". Chris approved the PASSIVE half only (2026-09-21): she may KNOW which rooms are
occupied when he asks her something. The active half — speaking because someone
arrived — is a new metered turn per event and a behaviour change, and is NOT built.

Two things the sensors cannot do, which the wording here is careful about:

1. They cannot name WHO. Several rooms read occupied at once (office and living room
   both on while this was written), and an occupancy sensor does not distinguish Chris
   from Megan from a passing cat. So this reports OCCUPANCY, never "Chris is in the
   sunroom" — claiming the latter would be the same error as treating a matching name
   as an identification.
2. They cannot be trusted as a safety signal. Everything here is fail-open: a dark or
   slow Home Assistant costs the block, never the turn.

ACCESS: presence disclosure is one of the gated surfaces (see ask()'s action_allowlist
— "actuating the house or disclosing presence is" sensitive). An ambient block would
quietly route around that gate: the presence TOOL stays allowlisted while the facts sit
in the prompt for anyone to ask about. So this block is injected on the OWNER's private
turns only; presence_block() enforces it, not the caller.
"""

from __future__ import annotations

import logging

log = logging.getLogger(__name__)

#: Rooms whose occupancy is worth naming, in the order she should read them.
#: Value = the entity to read; key = what she calls the room.
DEFAULT_ROOMS = {
    "office": "binary_sensor.office_occupancy",
    "living room": "binary_sensor.living_room_occupancy",
    "sunroom": "binary_sensor.sunroom_occupancy",
    "bedroom": "binary_sensor.emotion_air_bedroom",
    "guest room": "binary_sensor.emotion_air_guest_room",
    "kitchen": "binary_sensor.kitchen_motion_occupancy",
}


def parse_rooms(spec: str) -> dict[str, str]:
    """'office=binary_sensor.x,sunroom=binary_sensor.y' -> {room: entity}.

    Empty/blank spec falls back to DEFAULT_ROOMS. A malformed entry is skipped with a
    warning — a typo in config must not take the block (or the turn) down.
    """
    if not (spec or "").strip():
        return dict(DEFAULT_ROOMS)
    out: dict[str, str] = {}
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        room, sep, entity = item.partition("=")
        room, entity = room.strip().lower(), entity.strip()
        if not sep or not room or not entity:
            log.warning("presence rooms: skipping malformed entry %r", item)
            continue
        out[room] = entity
    return out or dict(DEFAULT_ROOMS)


def format_whereabouts(home: str | None, settled: str | None) -> list[str]:
    """Where CHRIS is, at the trust the week-long soak earned (Chris approved 2026-10-03).

    The BLE soak, 9/27-10/03: home/away never put him in a room while he was out (7
    nights), and his phone's settled room agreed with the office sensors 99.5% of the
    time; the bedroom read 50% (Megan, the cats, a still sleeper), and the settled room
    lags a move by about 6 minutes. So home/away and "most likely in the office" are
    stated; any other room is a soft guess she may mention lightly and never act on.
    """
    state = (home or "").strip()
    if not state or state in ("unknown", "unavailable"):
        return []
    if state != "home":
        return ["Chris's own phone says he is away from home right now."]
    # Attributed to HIS phone, out loud: the occupancy caveat below ("sensors do not say
    # WHO") made her doubt these lines too on the first live probe (2026-10-03).
    lines = ["Chris's own phone says he is home - that is him, so you can say it plainly."]
    room = (settled or "").strip()
    if room.lower() == "office":
        lines.append("His phone has settled in the office, so he is most likely there - say so if it comes up.")
    elif room and room.lower() not in ("unknown", "unavailable", "away", "not_home", "none"):
        lines.append(f"His phone suggests he might be in the {room.lower()} - a rough guess (other people "
                     "and pets confuse the room sensors, and it lags by minutes): mention it lightly if at "
                     "all, never act on it or state it as fact.")
    return lines


def format_presence(occupied: list[str], spoken_from: str | None = None,
                    whereabouts: tuple | None = None) -> str:
    """The block's text, or '' when there is nothing honest to say.

    Occupancy is phrased as occupancy, never as a person's location, and it says the
    limit out loud. WHEREABOUTS (person state, settled phone room) is the one place a
    location is stated, and only at the trust format_whereabouts allows.
    """
    # The satellite he is speaking through beats a phone room that lags by minutes.
    where = format_whereabouts(whereabouts[0], None if spoken_from else whereabouts[1]) if whereabouts else []
    if not occupied and not spoken_from and not where:
        return ""
    lines = list(where)
    if spoken_from:
        lines.append(f"He is speaking from the {spoken_from}.")
    if occupied:
        rooms = ", ".join(occupied)
        lines.append(f"Rooms showing occupancy right now: {rooms}.")
        lines.append("Occupancy sensors do not say WHO — that could be Megan or a pet. "
                     "Use this as ambient awareness; never assert where someone is from occupancy alone"
                     + (" (this caveat is about the room sensors, not his phone)." if where else "."))
    elif spoken_from or not where:
        lines.append("No room is showing occupancy right now.")
    # Leading-separated like room_block/portable_block — the prompt f-strings
    # concatenate blocks with no separator of their own.
    return "\n\n[House presence]\n" + "\n".join(lines)
