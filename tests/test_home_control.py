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

from aerys_v2.tools.home_control import build_home_control_tool, canary_set


# ---- fakes ---------------------------------------------------------------------

class FakeHA:
    """Records every request; scripted to behave like HA Green's REST API."""

    def __init__(self, fail_services: bool = False):
        self.requests: list[tuple[str, str]] = []  # (method, path)
        self.fail_services = fail_services

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append((request.method, request.url.path))
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
