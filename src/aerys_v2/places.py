"""What's at the places his phone stayed a while (D92; the owner, 11/22 17:29: "Works for me").

D85 hands her each stop as the address his phone's own geocoder gives, which is only the nearest
street number, never the business (11/22: four stops; her own web search named one of them, and
her real welcome writer named none). He wanted her to notice where he ate and where he went, and ask
about those things.

So before she composes, each stop's coordinates are looked up with Google's Places API (New) nearby
search, twice:
  * the few named places right where his phone sat (ranked by distance, within NEAR_M), and
  * the best-known places close by (ranked by prominence, within AROUND_M), because a phone that
    checked in at a plaza's entrance sits nearer the bank on the corner than the supermarket he
    walked into (11/22: the corner bank at 2 m, the supermarket at 125 m).
What sits by a parking spot (an ATM, a bike rack, a bus stop) is left out. They reach her as data,
nearest first, each with what kind of place it is; she asks rather than claims.

The look-up is the one place his coordinates leave the house, and the owner chose it (the key sits
in the rack's environment only). Nothing here logs a coordinate or a name. A failed or slow look-up
costs only the names: the stop goes with its address, as before.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Callable

log = logging.getLogger(__name__)

URL = "https://places.googleapis.com/v1/places:searchNearby"
#: Both fields are on the Nearby Search Pro SKU (5,000 free a month as of 11/22; the owner's key also
#: has a daily request cap).
FIELDS = "places.displayName,places.primaryTypeDisplayName"
NEAR_M, NEAR_N = 80.0, 3
AROUND_M, AROUND_N = 200.0, 3
MAX_NAMES = 5
TIMEOUT_S = 4.0
NAME_MAX = 80
#: Not somewhere he went: what sits by the spot his phone checked in from.
NOT_A_DESTINATION = ["atm", "parking", "bus_stop", "transit_station", "transportation_service",
                     "electric_vehicle_charging_station", "public_bathroom"]


@dataclass(frozen=True)
class Place:
    name: str                         # data from the map, never an instruction (it rides inside DATA_ONLY)
    kind: str = ""                    # "Japanese restaurant", "Supermarket", ...


def _text(value) -> str:
    return " ".join(str(value if value is not None else "").split())[:NAME_MAX]


def coords(value) -> tuple[float, float] | None:
    """The geocoded sensor's `location` attribute ([lat, lon]) as a pair, or None when it isn't one."""
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    try:
        lat, lon = float(value[0]), float(value[1])
    except (TypeError, ValueError):
        return None
    if isinstance(value[0], bool) or isinstance(value[1], bool):
        return None
    if not (math.isfinite(lat) and math.isfinite(lon) and abs(lat) <= 90 and abs(lon) <= 180):
        return None
    return lat, lon


def places_near_for(settings, *, post=None) -> Callable[[float, float], list[Place]] | None:
    """near(lat, lon) -> up to MAX_NAMES named places, nearest first then the best-known; None when no
    key is configured (off). post(url, headers, body) is the test seam. One of the two searches failing
    costs only its half; both failing raises (the caller goes without names)."""
    key = getattr(settings, "google_places_api_key", None)
    secret = key.get_secret_value() if hasattr(key, "get_secret_value") else str(key or "")
    if not secret.strip():
        return None
    import httpx

    send = post or (lambda url, headers, body: httpx.post(url, headers=headers, json=body, timeout=TIMEOUT_S))
    headers = {"Content-Type": "application/json", "X-Goog-Api-Key": secret.strip(), "X-Goog-FieldMask": FIELDS}

    def search(lat: float, lon: float, radius: float, count: int, rank: str) -> list[Place]:
        body = {"maxResultCount": count, "rankPreference": rank, "languageCode": "en",
                "excludedPrimaryTypes": list(NOT_A_DESTINATION),
                "locationRestriction": {"circle": {"center": {"latitude": lat, "longitude": lon}, "radius": radius}}}
        r = send(URL, headers, body)
        r.raise_for_status()
        found = []
        for p in (r.json() or {}).get("places") or []:
            if not isinstance(p, dict):
                continue
            name = _text((p.get("displayName") or {}).get("text"))
            if name:
                found.append(Place(name, _text((p.get("primaryTypeDisplayName") or {}).get("text"))))
        return found

    def near(lat: float, lon: float) -> list[Place]:
        found, failed = [], None
        for radius, count, rank in ((NEAR_M, NEAR_N, "DISTANCE"), (AROUND_M, AROUND_N, "POPULARITY")):
            try:
                found += search(lat, lon, radius, count, rank)
            except Exception as e:          # the type only: the request carries his coordinates
                failed = e
                log.warning("places: a nearby search failed (%s)", type(e).__name__)
        if not found and failed is not None:
            raise failed
        seen, out = set(), []
        for p in found:
            if p.name.lower() not in seen:
                seen.add(p.name.lower())
                out.append(p)
        return out[:MAX_NAMES]

    return near
