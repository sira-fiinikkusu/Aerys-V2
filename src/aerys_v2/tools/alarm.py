"""control_alarm — arm/disarm the house alarm, owner-gated, surface-aware.

Owner-commissioned (2026-08-07): "I do want to give you and Aerys the ability
to arm and disarm the house if I ask you to." The alarm panel lives in Home
Assistant (an alarm_control_panel entity backed by the monitoring service), so
this rides the same HA door as home_control — but it is NOT part of the canary
allowlist system, because the risk shape is different: a light misfire is an
annoyance, an alarm misfire is a security event. It gets its own gates.

Three gates, outermost first:

1. OWNER ONLY — the identity on the turn must be the owner's person_id.
   The action allowlist already keeps strangers out of the action graph
   entirely; this tightens further so that even an allowlisted household
   member can't drive the alarm until the owner says otherwise.
2. DISARM SURFACE RULE (the owner's own design, verbatim intent): arming is
   safe from anywhere — worst case the house gets MORE secure. DISARM is
   refused on open-air room voice (a satellite mic has no idea whose voice it
   heard; a shout through broken glass must not disarm the house). A turn is
   "open-air room voice" when the identity carries BOTH voice=True and an
   originating device_id — that is exactly the room-satellite path. The
   glasses and every authed text surface (Discord/Telegram/desk/plain HTTP)
   carry no room device_id, and only the owner wears the glasses.
3. OUTBOX RECEIPTS — every arm/disarm that reaches HA is written ahead to
   v2_outbox (kind 'alarm_control', executing -> succeeded/failed), same
   write-ahead pattern as home_control. An alarm action must never be a
   mystery write.

Failure posture: ToolNode contract — every path returns an honest string,
never raises. Refusals say exactly why, so she can relay the truth ("I can't
disarm from a room microphone — ask me from your phone").

Deliberately NO silent-success prefix: an alarm state change should be
CONFIRMED out loud, never silently absorbed.
"""

from __future__ import annotations

import json
import logging
import uuid
from typing import Any, Callable

import httpx
from langchain_core.runnables import RunnableConfig

log = logging.getLogger(__name__)

ConnFactory = Callable[[], Any]

_SERVICES = {
    "arm_home": "alarm_arm_home",
    "arm_away": "alarm_arm_away",
    "disarm": "alarm_disarm",
}

_REFUSE_VOICE_DISARM = (
    "REFUSED: I don't disarm the house from an open-air room microphone — a "
    "room mic can't verify who is speaking. Ask me from your phone (Discord/"
    "Telegram), the desk, or the glasses and I'll do it immediately."
)


def build_alarm_tool(
    *,
    base_url: str,
    token: str,
    entity_id: str,
    owner_person_id: str | None,
    client: httpx.Client | None = None,
    conn_factory: ConnFactory | None = None,
):
    """Close over config and return the control_alarm tool (seam philosophy:
    tests inject httpx.MockTransport + fake conn_factory; --serve passes real).
    """
    from langchain_core.tools import tool

    base = base_url.rstrip("/")
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    http = client or httpx.Client(timeout=10.0)

    def _receipt_open(payload: dict) -> int | None:
        if conn_factory is None:
            return None
        try:
            with conn_factory() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "INSERT INTO v2_outbox (kind, payload, idempotency_key, status) "
                        "VALUES ('alarm_control', %s::jsonb, %s, 'executing') RETURNING id",
                        (json.dumps(payload), str(uuid.uuid4())),
                    )
                    return cur.fetchone()[0]
        except Exception:
            log.warning("alarm outbox INSERT failed — action proceeds unaudited", exc_info=True)
            return None

    def _receipt_close(
        outbox_id: int | None, status: str, receipt: dict | None = None, error: str | None = None
    ) -> None:
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
            log.warning("alarm outbox UPDATE failed for row %s", outbox_id, exc_info=True)

    @tool
    def control_alarm(action: str, config: RunnableConfig) -> str:
        """Arm, disarm, or check the house security alarm (ADT panel).

        CALL THIS TOOL when the owner asks to arm the house, disarm the
        alarm, set the alarm for the night, or asks whether the house is
        armed. Only the owner may use it — relay refusals honestly.

        action: one of "status", "arm_home" (perimeter armed, people home),
        "arm_away" (full arm, nobody home), "disarm".

        Never claim the alarm changed state unless this tool confirmed it.
        """
        act = action.strip().lower()
        identity = ((config or {}).get("configurable") or {}).get("identity") or {}

        # Gate 1: owner only. No owner configured = the tool refuses everything
        # (a misconfigured box must fail toward "alarm untouchable").
        user_id = str(identity.get("user_id") or "")
        if owner_person_id is None or user_id != owner_person_id:
            return (
                "REFUSED: only the owner can operate the house alarm. "
                "This request was not made from the owner's identity."
            )

        if act == "status":
            try:
                r = http.get(f"{base}/api/states/{entity_id}", headers=headers)
                if r.status_code == 404:
                    return f"Home Assistant has no alarm entity named {entity_id}."
                r.raise_for_status()
                state = r.json().get("state")
                return f"The alarm panel is currently: {state}."
            except httpx.HTTPError as e:
                return f"Home Assistant is unreachable right now ({e})."

        service = _SERVICES.get(act)
        if service is None:
            return (
                f"Unknown action '{action}'. "
                "Valid actions: status, arm_home, arm_away, disarm."
            )

        # Gate 2: the disarm surface rule (see module docstring).
        origin_device = str(identity.get("device_id") or "")
        if act == "disarm" and identity.get("voice") and origin_device:
            return _REFUSE_VOICE_DISARM

        payload = {
            "action": act,
            "entity_id": entity_id,
            "requested_by": user_id,
            "voice": bool(identity.get("voice")),
            "device_id": origin_device or None,
        }
        outbox_id = _receipt_open(payload)
        try:
            r = http.post(
                f"{base}/api/services/alarm_control_panel/{service}",
                headers=headers,
                json={"entity_id": entity_id},
            )
            r.raise_for_status()
        except httpx.HTTPError as e:
            _receipt_close(outbox_id, "failed", error=str(e))
            return f"The {act} FAILED — Home Assistant said: {e}."
        try:
            changed = r.json()
        except ValueError:
            changed = None
        _receipt_close(
            outbox_id, "succeeded", receipt={"status_code": r.status_code, "changed": changed}
        )
        # Report the panel's own transition when HA returned it (arming has an
        # exit-delay window — "arming" is the honest immediate answer).
        new_state = None
        if isinstance(changed, list):
            for item in changed:
                if isinstance(item, dict) and item.get("entity_id") == entity_id:
                    new_state = item.get("state")
        if new_state:
            return f"Alarm {act} sent — the panel now reports: {new_state}."
        return f"Alarm {act} sent (HA responded {r.status_code})."

    return control_alarm
