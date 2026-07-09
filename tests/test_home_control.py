"""Offline tests for the home_control tool — fake HA (httpx.MockTransport), fake DB.

What these prove: reads are unrestricted, writes obey the canary allowlist with
HONEST refusal strings (never exceptions — a raise inside ToolNode kills the
turn), and every write that reaches HA rides the outbox-inline lifecycle:
INSERT 'executing' -> call -> UPDATE receipt/status, with the lease-exception
marker when n8n still holds the ha_write lease.
"""

import json
import uuid

import httpx

from aerys_v2.tools.home_control import (
    build_home_control_tool,
    build_search_entities_tool,
    canary_set,
)


# ---- fakes ---------------------------------------------------------------------

class FakeHA:
    """Records every request; scripted to behave like HA Green's REST API."""

    def __init__(self, fail_services: bool = False, states: list | None = None):
        self.requests: list[tuple[str, str]] = []  # (method, path)
        self.fail_services = fail_services
        self.states = states or []  # the GET /api/states listing (search tests)

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append((request.method, request.url.path))
        if request.url.path == "/api/states":
            return httpx.Response(200, json=self.states)
        if request.url.path.startswith("/api/states/"):
            entity = request.url.path.rsplit("/", 1)[-1]
            if entity == "light.ghost":
                return httpx.Response(404)
            return httpx.Response(
                200,
                json={"state": "off", "attributes": {"friendly_name": "Desk Lamp"}},
            )
        if self.fail_services:
            return httpx.Response(503, text="ha melted")
        # service calls return the list of changed states — the receipt evidence
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


def test_non_canary_write_refused_and_ha_never_called():
    ha = FakeHA()
    out = make_tool(ha).invoke({"operation": "turn_off", "entity_id": "light.bedroom"})
    assert out.startswith("Refused:")
    assert "light.desk" in out          # the honest part: says what IS allowed
    assert ha.requests == []            # refusal happens before any HTTP


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

def ha_state(entity_id, state="on", friendly=None, unit=None):
    attrs = {}
    if friendly is not None:
        attrs["friendly_name"] = friendly
    if unit is not None:
        attrs["unit_of_measurement"] = unit
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
        ha_state("sensor.jolteon_battery", "unavailable", friendly="Jolteon Battery"),
        ha_state("sensor.jolteon_range", "180", friendly="Jolteon Range", unit="mi"),
        ha_state("sensor.lonely_ghost", "unknown", friendly="Lonely Ghost"),
    ])
    # a live match exists -> the unavailable sibling is filtered out
    out = tool.invoke({"query": "jolteon"})
    assert out == "sensor.jolteon_range | Jolteon Range | 180 mi"
    # ONLY dead matches -> show them anyway (honesty beats tidiness)
    out = tool.invoke({"query": "ghost"})
    assert out == "sensor.lonely_ghost | Lonely Ghost | unknown"


def test_search_caps_at_15_matches():
    tool, _ = make_search(
        [ha_state(f"light.room_{i:02d}", "on", friendly=f"Room {i:02d}") for i in range(30)]
    )
    lines = tool.invoke({"query": "room"}).splitlines()
    assert len(lines) == 15


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
