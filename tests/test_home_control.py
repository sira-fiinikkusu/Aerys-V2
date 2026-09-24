"""Offline tests for the home_control tool — fake HA (httpx.MockTransport), fake DB.

What these prove: reads are unrestricted, writes obey the canary allowlist with
HONEST refusal strings (never exceptions — a raise inside ToolNode kills the
turn), and every write that reaches HA rides the outbox-inline lifecycle:
INSERT 'executing' -> call -> UPDATE receipt/status, with the lease-exception
marker when n8n still holds the ha_write lease.
"""

import json
import logging
import uuid

import httpx
import pytest

import aerys_v2.tools.home_control as _hc
_hc._VERIFY_DELAYS_S = (0.0, 0.0)
from aerys_v2.tools.home_control import (
    build_home_control_tool,
    build_search_entities_tool,
    canary_set,
)


# ---- fakes ---------------------------------------------------------------------

class FakeHA:
    """Records every request; scripted to behave like HA Green's REST API."""

    def __init__(self, fail_services: bool = False, states: list | None = None,
                 empty_changed: bool = False):
        self.requests: list[tuple[str, str]] = []  # (method, path)
        self.service_bodies: list[dict] = []  # JSON body of every service POST
        self.fail_services = fail_services
        self.empty_changed = empty_changed  # HA 200 but nothing changed (Tuya drop)
        self.states = states or []  # the GET /api/states listing (search tests)
        # entity -> extra top-level fields for GET /api/states/<entity> (e.g. last_changed)
        self.state_fields: dict[str, dict] = {}

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append((request.method, request.url.path))
        if request.url.path.startswith("/api/services/"):
            self.service_bodies.append(json.loads(request.content or b"{}"))
        if request.url.path == "/api/states":
            return httpx.Response(200, json=self.states)
        if request.url.path.startswith("/api/states/"):
            entity = request.url.path.rsplit("/", 1)[-1]
            if entity == "light.ghost":
                return httpx.Response(404)
            if entity == "light.dimmer":
                # a dimmable light mid-range: brightness is HA's 0-255 scale
                return httpx.Response(
                    200,
                    json={"state": "on", "attributes": {
                        "friendly_name": "Dimmer", "brightness": 128}},
                )
            return httpx.Response(
                200,
                json={"state": "off", "attributes": {"friendly_name": "Desk Lamp"},
                      **self.state_fields.get(entity, {})},
            )
        if self.fail_services:
            return httpx.Response(503, text="ha melted")
        # service calls return the list of changed states — the receipt evidence
        if self.empty_changed:
            return httpx.Response(200, json=[])
        return httpx.Response(200, json=[{"entity_id": "light.desk", "state": "on"}])

    def client(self) -> httpx.Client:
        return httpx.Client(transport=httpx.MockTransport(self.handler))


class FakeCursor:
    """Answers the exact two queries the outbox layer asks; records everything."""

    def __init__(self, store: "FakeDB"):
        self.store = store
        self._result = None

    def execute(self, sql: str, params=None) -> None:
        self.store.executed.append((sql, params))
        upper = sql.strip().upper()
        if upper.startswith("SELECT HOLDER"):
            self._result = (self.store.lease_holder,)
        elif "INSERT INTO V2_OUTBOX" in upper:
            self.store.next_id += 1
            self._result = (self.store.next_id,)
        else:
            self._result = None

    def fetchone(self):
        return self._result

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class FakeConn:
    def __init__(self, store: "FakeDB"):
        self.store = store

    def cursor(self) -> FakeCursor:
        return FakeCursor(self.store)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class FakeDB:
    """Shared state across the per-call connections (matches the prod pattern)."""

    def __init__(self, lease_holder: str = "brain"):
        self.executed: list[tuple[str, object]] = []
        self.lease_holder = lease_holder
        self.next_id = 100

    def factory(self):
        return FakeConn(self)

    # -- inspection helpers -------------------------------------------------
    def inserts(self):
        return [(s, p) for s, p in self.executed if "INSERT INTO v2_outbox" in s]

    def updates(self):
        return [(s, p) for s, p in self.executed if "UPDATE v2_outbox" in s]


def make_tool(ha: FakeHA, db: FakeDB | None = None, canary: str = "light.desk"):
    return build_home_control_tool(
        base_url="http://ha.test:8123",
        token="t0ken",
        canary_entities=canary_set(canary),
        client=ha.client(),
        conn_factory=db.factory if db is not None else None,
    )


# ---- reads: unrestricted -------------------------------------------------------

def test_get_state_reads_any_entity_no_allowlist():
    ha = FakeHA()
    tool = make_tool(ha, canary="")  # empty allowlist — reads must still work
    out = tool.invoke({"operation": "get_state", "entity_id": "light.desk"})
    assert json.loads(out) == {
        "entity_id": "light.desk", "state": "off", "friendly_name": "Desk Lamp",
    }
    assert ha.requests == [("GET", "/api/states/light.desk")]


def test_get_state_unknown_entity_is_honest():
    out = make_tool(FakeHA()).invoke({"operation": "get_state", "entity_id": "light.ghost"})
    assert "no entity named light.ghost" in out


# ---- writes: canary allowlist + domain gate ------------------------------------

def test_canary_write_succeeds():
    ha = FakeHA()
    out = make_tool(ha).invoke({"operation": "turn_on", "entity_id": "light.desk"})
    assert out.startswith("Done: turn_on sent to light.desk")
    assert ("POST", "/api/services/light/turn_on") in ha.requests


@pytest.mark.parametrize("entity_id,refused", [
    ("light.bedroom", ["light.bedroom"]),
    ("light.desk,light.bedroom,switch.guest", ["light.bedroom", "switch.guest"]),
])
def test_non_canary_write_refused_and_ha_never_called(caplog, entity_id, refused):
    ha = FakeHA()
    db = FakeDB()
    with caplog.at_level(logging.INFO, logger=_hc.__name__):
        out = make_tool(ha, db).invoke({"operation": "turn_off", "entity_id": entity_id})
    assert out.startswith("Refused:")
    assert "light.desk" in out          # the honest part: says what IS allowed
    assert ha.requests == []            # refusal happens before any HTTP
    assert db.executed == []
    assert [(r.levelno, r.getMessage()) for r in caplog.records if r.name == _hc.__name__] == [
        (logging.INFO, f"home_control refused {refused}: not on the beta write allowlist"),
    ]


def test_non_light_switch_domain_refused():
    ha = FakeHA()
    out = make_tool(ha, canary="lock.front_door").invoke(
        {"operation": "turn_on", "entity_id": "lock.front_door"}
    )
    assert "Refused" in out and ha.requests == []


def test_unknown_operation_is_honest_string_not_exception():
    out = make_tool(FakeHA()).invoke({"operation": "disco_mode", "entity_id": "light.desk"})
    assert "Unknown operation" in out


def test_ha_failure_reported_honestly():
    out = make_tool(FakeHA(fail_services=True)).invoke(
        {"operation": "toggle", "entity_id": "light.desk"}
    )
    assert "FAILED" in out


# ---- outbox-inline lifecycle ---------------------------------------------------

def test_write_records_outbox_insert_then_succeeded_update():
    db = FakeDB(lease_holder="brain")
    make_tool(FakeHA(), db).invoke({"operation": "turn_on", "entity_id": "light.desk"})

    [(insert_sql, insert_params)] = db.inserts()
    assert "'ha_write'" in insert_sql and "'executing'" in insert_sql
    payload = json.loads(insert_params[0])
    assert payload["operation"] == "turn_on" and payload["entity_id"] == "light.desk"
    uuid.UUID(insert_params[1])  # idempotency_key is a real uuid or this raises

    [(update_sql, update_params)] = db.updates()
    status, receipt_json, error, row_id = update_params
    assert status == "succeeded" and error is None and row_id == 101
    receipt = json.loads(receipt_json)
    assert receipt["status_code"] == 200
    assert receipt["changed"] == [{"entity_id": "light.desk", "state": "on"}]  # evidence, not bare ok


def test_ha_failure_marks_outbox_failed():
    db = FakeDB()
    make_tool(FakeHA(fail_services=True), db).invoke(
        {"operation": "turn_off", "entity_id": "light.desk"}
    )
    [(_, update_params)] = db.updates()
    status, receipt_json, error, _ = update_params
    assert status == "failed" and receipt_json is None and "503" in error


def test_refused_write_never_touches_outbox():
    # the outbox records intents that FIRE; a refusal is not an intent
    db = FakeDB()
    make_tool(FakeHA(), db).invoke({"operation": "turn_on", "entity_id": "light.bedroom"})
    assert db.executed == []


def test_lease_held_by_n8n_marks_beta_canary_exception():
    # the one-armed-writer exception: executes anyway, but the payload says so
    db = FakeDB(lease_holder="n8n")
    ha = FakeHA()
    make_tool(ha, db).invoke({"operation": "turn_on", "entity_id": "light.desk"})
    payload = json.loads(db.inserts()[0][1][0])
    assert payload["lease_exception"] == "beta-canary"
    assert ("POST", "/api/services/light/turn_on") in ha.requests  # still executed


def test_lease_held_by_brain_has_no_exception_marker():
    db = FakeDB(lease_holder="brain")
    make_tool(FakeHA(), db).invoke({"operation": "turn_on", "entity_id": "light.desk"})
    assert "lease_exception" not in json.loads(db.inserts()[0][1][0])


def test_no_conn_factory_means_no_outbox_but_write_works():
    ha = FakeHA()
    out = make_tool(ha, db=None).invoke({"operation": "turn_on", "entity_id": "light.desk"})
    assert out.startswith("Done:") and ("POST", "/api/services/light/turn_on") in ha.requests


# ---- search_entities: read-only discovery ---------------------------------------

def ha_state(entity_id, state="on", friendly=None, unit=None, device_class=None):
    attrs = {}
    if friendly is not None:
        attrs["friendly_name"] = friendly
    if unit is not None:
        attrs["unit_of_measurement"] = unit
    if device_class is not None:
        attrs["device_class"] = device_class
    return {"entity_id": entity_id, "state": state, "attributes": attrs}


def make_search(states: list):
    ha = FakeHA(states=states)
    tool = build_search_entities_tool(
        base_url="http://ha.test:8123", token="t0ken", client=ha.client()
    )
    return tool, ha


def test_search_matches_entity_id_and_friendly_name_case_insensitive():
    tool, ha = make_search([
        ha_state("sensor.ev6_battery_level", "78", friendly="Jolteon Battery", unit="%"),
        ha_state("light.office_lamp", "off", friendly="Office Lamp"),
        ha_state("switch.desk_fan", "on", friendly="Desk Fan"),
    ])
    out = tool.invoke({"query": "JOLTEON"})  # friendly-name hit, wrong case
    assert out == "sensor.ev6_battery_level | Jolteon Battery | 78 %"
    out = tool.invoke({"query": "office"})   # entity-id + friendly hit
    assert out == "light.office_lamp | Office Lamp | off"
    assert ("GET", "/api/states") in ha.requests
    assert all(m == "GET" for m, _ in ha.requests)  # READ-ONLY: no POST, ever


def test_search_ranking_more_matched_terms_first():
    tool, _ = make_search([
        ha_state("sensor.garage_temperature", "22", friendly="Garage Temperature"),
        ha_state("sensor.ev6_battery", "78", friendly="Jolteon Battery", unit="%"),
        ha_state("device_tracker.jolteon", "home", friendly="Jolteon Location"),
    ])
    lines = tool.invoke({"query": "jolteon battery"}).splitlines()
    # both terms matched beats one term matched
    assert lines[0].startswith("sensor.ev6_battery |")
    assert lines[1].startswith("device_tracker.jolteon |")
    assert len(lines) == 2  # garage never matched at all


def test_search_filters_unavailable_unless_nothing_else_matches():
    tool, _ = make_search([
        ha_state("sensor.car_a", "unavailable", friendly="Car A"),
        ha_state("sensor.car_b", "42", friendly="Car B", unit="%"),
    ])
    out = tool.invoke({"query": "car"})
    assert "sensor.car_b" in out and "sensor.car_a" not in out  # dead one filtered
    # ALL dead: not a listing but a verdict the model can relay (9/04: listing them
    # sent a small model into a read/search loop until the recursion rail)
    tool, _ = make_search([
        ha_state("sensor.jolteon_ev_battery_level", "unavailable", friendly="Jolteon EV Battery Level"),
        ha_state("sensor.jolteon_ev_range", "unavailable", friendly="Jolteon EV Range"),
    ])
    out = tool.invoke({"query": "jolteon"})
    assert out.startswith("All 2 entities matching 'jolteon' are unavailable")
    assert "Do not search or read again" in out


def test_get_state_on_unavailable_entity_says_not_reporting():
    class Dead(FakeHA):
        def handler(self, req):
            if req.url.path == "/api/states/sensor.jolteon_ev_battery_level":
                return httpx.Response(200, json={"state": "unavailable", "attributes": {"friendly_name": "Jolteon EV Battery Level"}})
            return super().handler(req)
    out = json.loads(make_tool(Dead()).invoke({"operation": "get_state", "entity_id": "sensor.jolteon_ev_battery_level"}))
    assert out["state"] == "unavailable" and "do not retry" in out["note"]


def test_search_caps_at_30_matches():
    # 9/04: 15 cut the garage off a "temperature" search (8 rooms x temp+humidity
    # + Stickies) and she named the wrong warmest room.
    tool, _ = make_search(
        [ha_state(f"light.room_{i:02d}", "on", friendly=f"Room {i:02d}") for i in range(45)]
    )
    lines = tool.invoke({"query": "room"}).splitlines()
    assert len(lines) == 30


def test_search_truncates_long_states():
    tool, _ = make_search([ha_state("sensor.weather_blob", "x" * 200, friendly="Weather")])
    out = tool.invoke({"query": "weather"})
    assert "x" * 60 + "…" in out and "x" * 61 not in out


def test_search_no_match_and_unreachable_are_honest_strings():
    tool, _ = make_search([ha_state("light.desk", "on", friendly="Desk Lamp")])
    assert "No Home Assistant entities match" in tool.invoke({"query": "flurble"})
    assert "at least one word" in tool.invoke({"query": "   "})

    def boom(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no route to ha")

    dead = build_search_entities_tool(
        base_url="http://ha.test:8123",
        token="t0ken",
        client=httpx.Client(transport=httpx.MockTransport(boom)),
    )
    assert "unreachable" in dead.invoke({"query": "desk"})


def test_search_retries_once_on_transient_transport_error_then_succeeds():
    """A momentary transport blip (HA mid-restart) is retried once and succeeds,
    so it never surfaces as an unreachable/degraded turn (build A, 2026-07-09)."""
    calls = {"n": 0}

    def flaky(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            raise httpx.ConnectError("blip")
        return httpx.Response(
            200,
            json=[
                {
                    "entity_id": "light.desk",
                    "state": "on",
                    "attributes": {"friendly_name": "Desk Lamp"},
                }
            ],
        )

    tool = build_search_entities_tool(
        base_url="http://ha.test:8123",
        token="t0ken",
        client=httpx.Client(transport=httpx.MockTransport(flaky)),
    )
    out = tool.invoke({"query": "desk"})
    assert calls["n"] == 2  # retried exactly once
    assert "light.desk" in out  # the second attempt's data came through
    assert "unreachable" not in out


def test_get_state_retries_once_on_transient_transport_error_then_succeeds():
    """home_control get_state gets the same single-retry resilience as search."""
    calls = {"n": 0}

    def flaky(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            raise httpx.ConnectError("blip")
        return httpx.Response(
            200, json={"state": "on", "attributes": {"friendly_name": "Desk Lamp"}}
        )

    tool = build_home_control_tool(
        base_url="http://ha.test:8123",
        token="t0ken",
        canary_entities=frozenset(),
        client=httpx.Client(transport=httpx.MockTransport(flaky)),
    )
    out = tool.invoke({"operation": "get_state", "entity_id": "light.desk"})
    assert calls["n"] == 2
    assert "unreachable" not in out
    assert "on" in out


# ---- brightness: set_brightness + turn_on at a level (gap: "down by 50%") ------

def test_set_brightness_rides_turn_on_with_pct():
    ha = FakeHA()
    out = make_tool(ha).invoke(
        {"operation": "set_brightness", "entity_id": "light.desk", "brightness_pct": 40})
    assert out.startswith("Done: set_brightness 40% sent to light.desk")
    # sugar over HA's light/turn_on — there is no set_brightness service
    assert ("POST", "/api/services/light/turn_on") in ha.requests
    assert ha.service_bodies == [{"entity_id": "light.desk", "brightness_pct": 40}]


def test_turn_on_accepts_optional_brightness():
    ha = FakeHA()
    out = make_tool(ha).invoke(
        {"operation": "turn_on", "entity_id": "light.desk", "brightness_pct": 25})
    assert out.startswith("Done: turn_on 25% sent to light.desk")
    assert ha.service_bodies == [{"entity_id": "light.desk", "brightness_pct": 25}]


def test_plain_turn_on_body_has_no_brightness_key():
    ha = FakeHA()
    make_tool(ha).invoke({"operation": "turn_on", "entity_id": "light.desk"})
    assert ha.service_bodies == [{"entity_id": "light.desk"}]


def test_set_brightness_without_pct_is_honest():
    ha = FakeHA()
    out = make_tool(ha).invoke(
        {"operation": "set_brightness", "entity_id": "light.desk"})
    assert "needs brightness_pct" in out
    assert ha.requests == []  # refused before any HTTP


def test_brightness_out_of_range_refused_with_turn_off_hint():
    ha = FakeHA()
    out = make_tool(ha).invoke(
        {"operation": "set_brightness", "entity_id": "light.desk", "brightness_pct": 0})
    assert "must be 1-100" in out and "turn_off" in out
    assert ha.requests == []


def test_brightness_on_a_switch_refused():
    ha = FakeHA()
    out = make_tool(ha, canary="switch.fan").invoke(
        {"operation": "set_brightness", "entity_id": "switch.fan", "brightness_pct": 50})
    assert "only applies to lights" in out
    assert ha.requests == []


def test_get_state_translates_brightness_to_pct():
    ha = FakeHA()
    out = make_tool(ha).invoke({"operation": "get_state", "entity_id": "light.dimmer"})
    data = json.loads(out)
    assert data["brightness_pct"] == 50  # 128/255 rounds to 50 — the relative-math input


def test_get_state_omits_brightness_when_absent():
    ha = FakeHA()
    out = make_tool(ha).invoke({"operation": "get_state", "entity_id": "light.desk"})
    assert "brightness_pct" not in json.loads(out)


def test_brightness_recorded_in_outbox_intent():
    ha, db = FakeHA(), FakeDB()
    make_tool(ha, db).invoke(
        {"operation": "set_brightness", "entity_id": "light.desk", "brightness_pct": 70})
    (sql, params), = db.inserts()
    payload = json.loads(params[0]) if isinstance(params[0], str) else params[0]
    assert payload["brightness_pct"] == 70 and payload["operation"] == "set_brightness"


def test_empty_changed_list_is_a_caveat_not_a_done():
    # HA 200 + changed [] = the Tuya silent-drop (she told Chris "on at 30%"
    # while the light stayed dark, 2026-08-10). Must NOT earn the Done: prefix
    # (which triggers silent-success). Since 2026-09-04 the tool reads the device
    # back ITSELF (FakeHA says light.desk is off) and reports the drop — no
    # "go verify" round-trip handed to the model.
    ha = FakeHA(empty_changed=True)
    out = make_tool(ha).invoke(
        {"operation": "set_brightness", "entity_id": "light.desk", "brightness_pct": 30})
    assert not out.startswith("Done:")
    assert "NO state change" in out and "not have applied" in out
    assert ("GET", "/api/states/light.desk") in ha.requests  # the read-back happened


# ---- color: set_color / turn_on with color (2026-09-19, her own gap report) ------

def test_set_color_by_name_rides_turn_on_with_color_name():
    ha = FakeHA()
    out = make_tool(ha).invoke({"operation": "set_color", "entity_id": "light.desk", "color": "red"})
    assert out.startswith("Done: set_color red sent to light.desk")
    assert ("POST", "/api/services/light/turn_on") in ha.requests
    assert ha.service_bodies == [{"entity_id": "light.desk", "color_name": "red"}]


def test_set_color_hex_becomes_rgb_and_names_normalise():
    ha = FakeHA()
    tool = make_tool(ha)
    tool.invoke({"operation": "set_color", "entity_id": "light.desk", "color": "#FF8000"})
    tool.invoke({"operation": "set_color", "entity_id": "light.desk", "color": "Sky Blue"})
    assert ha.service_bodies == [
        {"entity_id": "light.desk", "rgb_color": [255, 128, 0]},
        {"entity_id": "light.desk", "color_name": "skyblue"},
    ]


def test_turn_on_with_color_and_brightness_in_one_call():
    ha = FakeHA()
    out = make_tool(ha).invoke(
        {"operation": "turn_on", "entity_id": "light.desk", "brightness_pct": 30, "color": "blue"})
    assert out.startswith("Done: turn_on blue 30% sent to light.desk")
    assert ha.service_bodies == [{"entity_id": "light.desk", "brightness_pct": 30, "color_name": "blue"}]


def test_set_color_refuses_switches_garbage_and_missing_color():
    ha = FakeHA()
    tool = make_tool(ha, canary="light.desk,switch.fan")
    assert "only applies to lights" in tool.invoke(
        {"operation": "set_color", "entity_id": "switch.fan", "color": "red"})
    assert "isn't a color" in tool.invoke(
        {"operation": "set_color", "entity_id": "light.desk", "color": "rgb(1,2,3)"})
    assert "needs a color" in tool.invoke({"operation": "set_color", "entity_id": "light.desk"})
    assert ha.service_bodies == []  # every refusal happened before any HTTP


def test_set_color_refuses_a_white_only_bulb_honestly():
    class WhiteOnlyHA(FakeHA):
        def handler(self, request):
            if request.url.path == "/api/states/light.desk":
                return httpx.Response(200, json={"state": "on", "attributes": {
                    "friendly_name": "Desk Lamp", "supported_color_modes": ["color_temp"]}})
            return super().handler(request)
    ha = WhiteOnlyHA()
    out = make_tool(ha).invoke({"operation": "set_color", "entity_id": "light.desk", "color": "red"})
    assert out.startswith("Refused: light.desk does not support color")
    assert ha.service_bodies == []


def test_get_state_reports_color_capability_and_current_color():
    class ColorHA(FakeHA):
        def handler(self, request):
            if request.url.path == "/api/states/light.desk":
                return httpx.Response(200, json={"state": "on", "attributes": {
                    "friendly_name": "Desk Lamp", "supported_color_modes": ["color_temp", "xy"],
                    "rgb_color": [255, 0, 0], "brightness": 255}})
            return super().handler(request)
    out = json.loads(make_tool(ColorHA()).invoke({"operation": "get_state", "entity_id": "light.desk"}))
    assert out["color_capable"] is True and out["rgb_color"] == [255, 0, 0]


# ---- the dressers (2026-09-20 11:58, Aerys's own report via message_kael) ----------
# Zigbee plugs report their new state a beat AFTER HA answers the command, so HA's
# changed-list is empty and the read-back finds them off. She said "already off"
# after turning them off. last_changed decides whose doing the state is.

def _iso(dt):
    return dt.isoformat().replace("+00:00", "+00:00")


def test_state_that_changed_after_the_command_is_done_not_already_there():
    from datetime import datetime, timedelta, timezone
    ha = FakeHA(empty_changed=True)
    ha.state_fields["light.desk"] = {"last_changed": _iso(datetime.now(timezone.utc) + timedelta(seconds=5))}
    out = make_tool(ha).invoke({"operation": "turn_off", "entity_id": "light.desk"})
    assert out.startswith("Done:") and "light.desk" in out
    assert "already" not in out


def test_state_that_was_already_there_before_the_command_stays_already():
    from datetime import datetime, timedelta, timezone
    ha = FakeHA(empty_changed=True)
    ha.state_fields["light.desk"] = {"last_changed": _iso(datetime.now(timezone.utc) - timedelta(minutes=10))}
    out = make_tool(ha).invoke({"operation": "turn_off", "entity_id": "light.desk"})
    assert out.startswith("OK (already there):") and "already off" in out


def test_missing_or_bad_last_changed_falls_back_to_already_there():
    ha = FakeHA(empty_changed=True)                    # FakeHA default: no last_changed field
    out = make_tool(ha).invoke({"operation": "turn_off", "entity_id": "light.desk"})
    assert out.startswith("OK (already there):")
    ha.state_fields["light.desk"] = {"last_changed": "not-a-date"}
    out = make_tool(ha).invoke({"operation": "turn_off", "entity_id": "light.desk"})
    assert out.startswith("OK (already there):")


# ---- search_entities: found by what it IS, not only what it is called ----------
# Regression cover for 2026-09-24: asked what was open, she answered "everything is
# shut except Kitchen Window 2" while three sliding doors stood open and that window
# is dead hardware. Substring matching could not reach a door named "Slider", could
# not reach a singular from a plural, and happily quoted a retired sensor as open.

def _house_states():
    return [
        # ⚠️ The real sliders carry device_class None — only the NAME says what they
        # are. The first version of this fix passed because the test invented a
        # device_class the house does not have. Keep this shape.
        ha_state("binary_sensor.kitchen_slider", "on", friendly="Kitchen Slider"),
        ha_state("binary_sensor.office_window", "on", friendly="Office Window",
                 device_class="window"),
        ha_state("binary_sensor.front_door", "off", friendly="Front Door",
                 device_class="door"),
        ha_state("sensor.office_temperature", "78", friendly="Office Temperature",
                 unit="°F"),
    ]


def _retired_group(members):
    return {
        "entity_id": "group.adt_retired_contacts",
        "state": "on",
        "attributes": {"friendly_name": "ADT Retired Contacts",
                       "entity_id": list(members)},
    }


def _entities(out: str) -> set:
    return {line.split(" | ")[0] for line in out.splitlines()}


def test_search_finds_a_sliding_door_from_the_word_door():
    """The name holds neither "door" nor "window" — the device_class does."""
    tool, _ = make_search(_house_states())
    out = tool.invoke({"query": "door"})
    assert "binary_sensor.kitchen_slider" in _entities(out)


def test_search_plurals_reach_the_singular():
    tool, _ = make_search(_house_states())
    singular = _entities(tool.invoke({"query": "door window"}))
    plural = _entities(tool.invoke({"query": "doors windows"}))
    assert plural == singular
    assert "binary_sensor.kitchen_slider" in plural


def test_search_open_names_a_state_not_an_entity():
    """"is anything open?" names no entity; it names a kind of thing."""
    tool, _ = make_search(_house_states())
    found = _entities(tool.invoke({"query": "open"}))
    assert {"binary_sensor.kitchen_slider", "binary_sensor.office_window",
            "binary_sensor.front_door"} <= found
    assert "sensor.office_temperature" not in found


def test_retired_sensor_is_flagged_and_never_reads_as_a_plain_open():
    states = _house_states() + [_retired_group(["binary_sensor.office_window"])]
    tool, _ = make_search(states)
    out = tool.invoke({"query": "window"})
    line = [ln for ln in out.splitlines()
            if ln.startswith("binary_sensor.office_window")][0]
    assert "retired sensor" in line
    assert "hardware is gone" in line
    # the live one is untouched
    other = tool.invoke({"query": "door"})
    assert "retired sensor" not in other


def test_retired_group_absent_changes_nothing():
    with_group = make_search(_house_states() + [_retired_group([])])[0]
    without = make_search(_house_states())[0]
    assert without.invoke({"query": "window"}) == with_group.invoke({"query": "window"})
    assert "retired sensor" not in without.invoke({"query": "window"})


def test_retired_lookup_adds_no_second_request():
    states = _house_states() + [_retired_group(["binary_sensor.office_window"])]
    tool, ha = make_search(states)
    tool.invoke({"query": "window"})
    assert ha.requests == [("GET", "/api/states")]


def test_an_unclassified_slider_is_still_found_by_door():
    """device_class None, name says "Slider" — the house's real shape."""
    tool, _ = make_search(_house_states())
    for q in ("door", "doors", "open"):
        assert "binary_sensor.kitchen_slider" in _entities(tool.invoke({"query": q})), q


def test_a_word_still_matches_its_own_name():
    tool, _ = make_search(_house_states())
    assert "binary_sensor.kitchen_slider" in _entities(tool.invoke({"query": "slider"}))
    assert "sensor.office_temperature" in _entities(tool.invoke({"query": "temperature"}))


def test_asking_what_is_open_floats_the_open_things_above_the_shut_ones():
    """SEARCH_LIMIT cuts alphabetically; a state word says which side matters."""
    tool, _ = make_search(_house_states())
    out = tool.invoke({"query": "open"}).splitlines()
    order = [ln.split(" | ")[0] for ln in out]
    # front_door is closed and sorts FIRST alphabetically; it must not outrank
    # the two that are actually open.
    assert order.index("binary_sensor.kitchen_slider") < order.index("binary_sensor.front_door")
    assert order.index("binary_sensor.office_window") < order.index("binary_sensor.front_door")


def test_state_word_boosts_but_never_hides():
    tool, _ = make_search(_house_states())
    assert "binary_sensor.front_door" in _entities(tool.invoke({"query": "open"}))


def test_a_button_on_an_opening_channel_never_answers_what_is_open():
    """eMotion Air face buttons advertise as device_class opening; they are not openings."""
    states = _house_states() + [
        ha_state("binary_sensor.emotion_air_living_room_opening", "on",
                 friendly="eMotion Air Living Room Opening", device_class="opening"),
    ]
    tool, _ = make_search(states)
    for q in ("open", "door", "window"):
        assert "binary_sensor.emotion_air_living_room_opening" not in _entities(
            tool.invoke({"query": q})), q
    # but asked for BY NAME it is still findable — excluded from class matching, not hidden
    assert "binary_sensor.emotion_air_living_room_opening" in _entities(
        tool.invoke({"query": "emotion air living room"}))
