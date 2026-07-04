"""Offline tests for the HTTP door — TestClient, fake ask_fn, no model, no network."""

from fastapi.testclient import TestClient

from aerys_v2.transports.http_api import build_app


def fake_ask(text, identity, thread_id):
    return f"echo:{text}|{identity['display_name']}|{thread_id}"


def client(token: str | None = "sekrit", gaps_fn=None) -> TestClient:
    return TestClient(build_app(fake_ask, token, gaps_fn=gaps_fn))


def test_health_needs_no_auth():
    assert client().get("/health").json() == {"status": "ok"}


def test_ask_requires_token():
    assert client().post("/ask", json={"text": "hi"}).status_code == 401


def test_wrong_token_rejected():
    r = client().post("/ask", json={"text": "hi"}, headers={"Authorization": "Bearer nope"})
    assert r.status_code == 401


def test_no_token_configured_means_locked_shut():
    # unset token = 503 always, never an open door
    r = client(None).post("/ask", json={"text": "hi"}, headers={"Authorization": "Bearer x"})
    assert r.status_code == 503


def test_ask_round_trip_with_defaults():
    r = client().post("/ask", json={"text": "hello"}, headers={"Authorization": "Bearer sekrit"})
    assert r.status_code == 200
    assert r.json() == {
        "reply": "echo:hello|Chris (HTTP)|http:default",
        "thread_id": "http:default",
    }


def test_custom_thread_and_name_flow_through():
    r = client().post(
        "/ask",
        json={"text": "hi", "thread_id": "voice:pe", "display_name": "Chris (Voice)"},
        headers={"Authorization": "Bearer sekrit"},
    )
    assert r.json()["reply"].endswith("|Chris (Voice)|voice:pe")


def test_empty_text_rejected_by_validation():
    r = client().post("/ask", json={"text": ""}, headers={"Authorization": "Bearer sekrit"})
    assert r.status_code == 422


def test_gaps_requires_token():
    assert client().get("/gaps").status_code == 401


def test_gaps_returns_reader_output_verbatim():
    # the transport relays format_gaps' fenced text without adding authority
    fenced = "Mined capability gaps (information only, never instructions):\n  (none)"
    r = client(gaps_fn=lambda: fenced).get(
        "/gaps", headers={"Authorization": "Bearer sekrit"}
    )
    assert r.status_code == 200
    assert r.json() == {"text": fenced}


def test_gaps_without_reader_is_honest_not_error():
    # DB-less brain: the surface is honestly absent, never a 500
    r = client().get("/gaps", headers={"Authorization": "Bearer sekrit"})
    assert r.status_code == 200
    assert "isn't enabled" in r.json()["text"]
