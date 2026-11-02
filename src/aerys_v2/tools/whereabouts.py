"""Request-scoped nearby location, fenced before any HA read."""
import json
import math

import httpx
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

from aerys_v2.services.location_guard import mark_location_turn
from aerys_v2.state import identity_from_config
from aerys_v2.services.presence import location_fields

_GUIDANCE = (
    "For this request only: never store this location, never repeat coordinates "
    "back unless Chris asks, and never tell anyone else."
)


def build_whereabouts_tool(*, base_url: str, token: str, person_entity: str,
                           geocoded_entity: str, owner_person_id: str | None,
                           coords_entity: str = "",
                           client: httpx.Client | None = None):
    """Build the owner/private-only HA reader; identity is injected, never model input."""
    def read(http, entity):
        response = http.get(f"{base_url.rstrip('/')}/api/states/{entity}",
                            headers={"Authorization": f"Bearer {token}"})
        response.raise_for_status()
        return response.json()

    @tool
    def where_is_chris_nearby(config: RunnableConfig) -> str:
        """Call ONLY when Chris asks for something near him right now (food,
        a store, directions). The answer is for this request only: never store it,
        never repeat coordinates back unless he asks, never tell anyone else.
        """
        identity = identity_from_config(config)
        if (not owner_person_id or identity.get("user_id") != owner_person_id
                or identity.get("privacy_context") != "private"):
            return "Location request refused: only the owner on a private turn may use this tool."
        if not person_entity or not geocoded_entity:
            return "Current location is unavailable. Never guess."
        def locate(http):
            person = read(http, person_entity)
            state = person.get("state")
            if not isinstance(state, str) or not state.strip() or state in ("unknown", "unavailable"):
                raise ValueError("No person state")
            geocoded = read(http, geocoded_entity)
            geo_state = geocoded.get("state")
            if (not isinstance(geo_state, str) or not geo_state.strip()
                    or geo_state.strip().lower() in ("unknown", "unavailable")):
                raise ValueError("No geocoded state")
            attrs = geocoded.get("attributes")
            if not isinstance(attrs, dict):
                raise ValueError("No geocoded attributes")
            fields = location_fields(attrs, nearby=state != "home")
            if state == "home":
                mark_location_turn(values=[fields.get("locality", "")], person_id=identity.get("user_id"))
                return "Chris is home. " + json.dumps({"city": fields.get("locality", "")}) + " " + _GUIDANCE
            # Home/away comes from the presence entity, which in production is a settled
            # sensor with no GPS (live 2026-11-02: every lookup came back unavailable).
            # Coordinates come from the HA person, which carries the phone's tracker.
            source = read(http, coords_entity) if coords_entity and coords_entity != person_entity else person
            coords = source.get("attributes", {})
            if not isinstance(coords, dict):
                raise ValueError("No coordinate attributes")
            result = {label: fields.get(key, "") for label, key in (
                ("street", "thoroughfare"), ("neighborhood", "sub_locality"),
                ("city", "locality"), ("state", "administrative_area"))}
            for key, limit in (("latitude", 90), ("longitude", 180)):
                value = coords.get(key)
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise ValueError("No coordinates")
                if not math.isfinite(value) or abs(value) > limit:
                    raise ValueError("Invalid coordinates")
                result[key] = f"{value:.3f}"
            mark_location_turn(values=(result[k] for k in ("street", "neighborhood", "city", "state")),
                               person_id=identity.get("user_id"))
            return json.dumps(result) + " " + _GUIDANCE
        try:
            if client is not None:
                return locate(client)
            with httpx.Client(timeout=2.0) as http:
                return locate(http)
        except Exception:
            # Do not log response bodies or exception strings: they can contain
            # geocoded attributes and unrounded coordinates.
            return "Current location is unavailable. Never guess."

    return where_is_chris_nearby
