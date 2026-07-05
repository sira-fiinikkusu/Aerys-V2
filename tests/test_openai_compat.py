"""Offline tests for the /v1/chat/completions shim (HA Extended OpenAI Conversation)."""

from fastapi.testclient import TestClient

from aerys_v2.transports.http_api import build_app


def fake_ask(text, identity, thread_id):
    return f"echo:{text}|{thread_id}"


def client() -> TestClient:
    return TestClient(build_app(fake_ask, "sekrit"))


AUTH = {"Authorization": "Bearer sekrit"}


def test_requires_token():
    assert client().post("/v1/chat/completions", json={"messages": []}).status_code == 401


def test_takes_last_user_message_and_ignores_transcript():
    # HA resends its own history — we must use ONLY the newest user turn
    r = client().post("/v1/chat/completions", headers=AUTH, json={
        "model": "aerys",
        "messages": [
            {"role": "system", "content": "ha system prompt"},
            {"role": "user", "content": "old question"},
            {"role": "assistant", "content": "old answer"},
            {"role": "user", "content": "new question"},
        ],
    })
    body = r.json()
    assert body["choices"][0]["message"]["content"] == "echo:new question|voice:beta"
    assert body["choices"][0]["finish_reason"] == "stop"


def test_content_parts_form_flattened():
    r = client().post("/v1/chat/completions", headers=AUTH, json={
        "messages": [{"role": "user", "content": [{"type": "text", "text": "hi there"}]}],
    })
    assert "hi there" in r.json()["choices"][0]["message"]["content"]


def test_no_user_message_400():
    r = client().post("/v1/chat/completions", headers=AUTH, json={"messages": [
        {"role": "system", "content": "x"}]})
    assert r.status_code == 400


def test_models_list_for_ha_validation():
    r = client().get("/v1/models", headers=AUTH)
    assert r.json()["data"][0]["id"] == "aerys-v2"


def test_voice_compat_folds_into_owner_person_thread():
    # With an owner configured, the voice pipeline's turn rides the owner's continuous
    # 'person:{id}' thread (cross-surface continuity) — no longer the shared 'voice:beta'.
    c = TestClient(build_app(fake_ask, "sekrit", owner_person_id="owner-uuid"))
    r = c.post("/v1/chat/completions", headers=AUTH,
               json={"messages": [{"role": "user", "content": "hey"}]})
    assert r.json()["choices"][0]["message"]["content"] == "echo:hey|person:owner-uuid"


def test_voice_compat_falls_back_to_voice_thread_without_owner():
    # No owner (dev/CI): there's no owner thread to join, so keep the legacy shared
    # voice thread. Voice-ness still rides the explicit flag (see test_context.py).
    r = client().post("/v1/chat/completions", headers=AUTH,
                      json={"messages": [{"role": "user", "content": "hey"}]})
    assert r.json()["choices"][0]["message"]["content"] == "echo:hey|voice:beta"
