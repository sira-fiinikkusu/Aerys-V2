"""control_alarm (owner-commissioned 2026-08-07) — offline, fakes all the way.

What these prove: the alarm is owner-only (everyone else refused, including
allowlisted household members), DISARM is refused on open-air room voice
(voice=True + device_id — the satellite path) while arming works from
anywhere, glasses/text surfaces (no room device_id) may disarm, every action
that reaches HA leaves a v2_outbox receipt, and failures come back as honest
strings — never raises.
"""

import json

import httpx

from aerys_v2.tools.alarm import build_alarm_tool

OWNER = "owner-uuid"
ENTITY = "alarm_control_panel.panel"


def transport(captured, *, fail=False, state_after="armed_home"):
    def handler(request: httpx.Request) -> httpx.Response:
        captured.append((request.method, request.url.path,
                         json.loads(request.content) if request.content else None))
        if fail:
            return httpx.Response(500, text="boom")
        if request.method == "GET":
            return httpx.Response(200, json={"entity_id": ENTITY, "state": "disarmed"})
        return httpx.Response(200, json=[{"entity_id": ENTITY, "state": state_after}])

    return httpx.MockTransport(handler)


class FakeCursor:
    def __init__(self, rows):
        self.rows = rows

    def execute(self, sql, params=None):
        self.rows.append((sql, params))

    def fetchone(self):
        return (42,)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class FakeConn:
    def __init__(self, rows):
        self.rows = rows

    def cursor(self):
        return FakeCursor(self.rows)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def make_tool(captured, *, rows=None, **tkw):
    return build_alarm_tool(
        base_url="http://ha.test",
        token="tok",
        entity_id=ENTITY,
        owner_person_id=OWNER,
        client=httpx.Client(transport=transport(captured, **tkw)),
        conn_factory=(lambda: FakeConn(rows)) if rows is not None else None,
    )


def cfg(identity):
    return {"configurable": {"identity": identity}}


OWNER_TEXT = {"user_id": OWNER}                                   # Discord/Telegram/desk
OWNER_GLASSES = {"user_id": OWNER, "voice": True}                 # glasses: voice, no room device
OWNER_SATELLITE = {"user_id": OWNER, "voice": True, "device_id": "office-sat"}


# ─────────────────────────────────────────────────────────────── gate 1: owner only


def test_non_owner_refused_even_for_status_and_arm():
    captured = []
    tool = make_tool(captured)
    for action in ("status", "arm_home", "disarm"):
        out = tool.invoke({"action": action}, config=cfg({"user_id": "megan-uuid"}))
        assert "REFUSED" in out and "owner" in out
    assert captured == []  # nothing ever reached HA


def test_no_owner_configured_fails_closed():
    captured = []
    tool = build_alarm_tool(
        base_url="http://ha.test", token="tok", entity_id=ENTITY,
        owner_person_id=None,
        client=httpx.Client(transport=transport(captured)),
    )
    out = tool.invoke({"action": "arm_home"}, config=cfg(OWNER_TEXT))
    assert "REFUSED" in out
    assert captured == []


# ─────────────────────────────────────────────── gate 2: the disarm surface rule


def test_disarm_refused_on_room_satellite_voice():
    captured = []
    tool = make_tool(captured)
    out = tool.invoke({"action": "disarm"}, config=cfg(OWNER_SATELLITE))
    assert "REFUSED" in out and "microphone" in out
    assert captured == []  # refusal happens before any HTTP


def test_arm_allowed_from_room_satellite_voice():
    captured = []
    tool = make_tool(captured)
    out = tool.invoke({"action": "arm_away"}, config=cfg(OWNER_SATELLITE))
    assert "REFUSED" not in out
    assert captured[0][1].endswith("/api/services/alarm_control_panel/alarm_arm_away")


def test_disarm_allowed_from_glasses_and_text():
    for identity in (OWNER_GLASSES, OWNER_TEXT):
        captured = []
        tool = make_tool(captured)
        out = tool.invoke({"action": "disarm"}, config=cfg(identity))
        assert "REFUSED" not in out
        assert captured[0][1].endswith("/alarm_disarm")


# ───────────────────────────────────────────────────────── behavior + receipts


def test_status_reports_panel_state_without_receipt():
    captured, rows = [], []
    tool = make_tool(captured, rows=rows)
    out = tool.invoke({"action": "status"}, config=cfg(OWNER_TEXT))
    assert "disarmed" in out
    assert rows == []  # reads leave no outbox rows


def test_arm_reports_transition_and_writes_receipt():
    captured, rows = [], []
    tool = make_tool(captured, rows=rows)
    out = tool.invoke({"action": "arm_home"}, config=cfg(OWNER_TEXT))
    assert "armed_home" in out
    inserts = [r for r in rows if "INSERT INTO v2_outbox" in r[0]]
    updates = [r for r in rows if "UPDATE v2_outbox" in r[0]]
    assert len(inserts) == 1 and "alarm_control" in inserts[0][0]
    payload = json.loads(inserts[0][1][0])
    assert payload["action"] == "arm_home" and payload["requested_by"] == OWNER
    assert len(updates) == 1 and updates[0][1][0] == "succeeded"


def test_ha_failure_is_honest_and_receipted_failed():
    captured, rows = [], []
    tool = make_tool(captured, rows=rows, fail=True)
    out = tool.invoke({"action": "arm_away"}, config=cfg(OWNER_TEXT))
    assert "FAILED" in out
    updates = [r for r in rows if "UPDATE v2_outbox" in r[0]]
    assert len(updates) == 1 and updates[0][1][0] == "failed"


def test_unknown_action_lists_valid_ones():
    captured = []
    tool = make_tool(captured)
    out = tool.invoke({"action": "self_destruct"}, config=cfg(OWNER_TEXT))
    assert "Unknown action" in out and "arm_home" in out
    assert captured == []


# ─────────────────────────────────────────────────────────── factory arming knob


def test_factory_arms_alarm_tool_only_when_entity_set():
    from aerys_v2.config import Settings
    from aerys_v2.factory import action_tools_for

    base = dict(
        _env_file=None,
        anthropic_api_key="sk-test",
        ha_token="tok",
        owner_person_id="7c9e6679-7425-40de-963d-3b1c0d2f8e11",
    )
    without = action_tools_for(Settings(**base))
    with_alarm = action_tools_for(Settings(**base, ha_alarm_entity=ENTITY))
    names_without = {t.name for t in without}
    names_with = {t.name for t in with_alarm}
    assert "control_alarm" not in names_without
    assert "control_alarm" in names_with
