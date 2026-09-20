"""Offline tests for the HTTP door — TestClient, fake ask_fn, no model, no network."""

from fastapi.testclient import TestClient

from aerys_v2.transports.http_api import build_app


def fake_ask(text, identity, thread_id):
    return f"echo:{text}|{identity['display_name']}|{thread_id}"


def client(token: str | None = "sekrit", gaps_fn=None, health_probe=None) -> TestClient:
    return TestClient(build_app(fake_ask, token, gaps_fn=gaps_fn, health_probe=health_probe))


def test_health_needs_no_auth():
    assert client().get("/health").json() == {"status": "ok"}


# --- /health tells the truth about the store (the 7/30-31 outage) ------------
# The endpoint used to return a hardcoded 200 and reported healthy through a
# ~13.5 hour outage. These pin that it can now actually fail.


def test_health_ok_when_probe_passes():
    r = client(health_probe=lambda: None).get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_health_503_when_store_unreachable():
    def dead():
        raise RuntimeError("connection is closed")

    r = client(health_probe=dead).get("/health")
    assert r.status_code == 503
    assert r.json()["status"] == "degraded"


def test_health_503_body_never_leaks_the_dsn():
    """/health is UNAUTHENTICATED and tunnel-reachable; psycopg errors carry the
    password. The body must stay generic no matter what the probe raised."""
    secret = "postgresql://dbuser:hunter2@db.example:5432/aerys"

    def dead():
        raise RuntimeError(f"could not connect: {secret}")

    r = client(health_probe=dead).get("/health")
    assert r.status_code == 503
    assert "hunter2" not in r.text
    assert secret not in r.text
    assert r.json() == {"status": "degraded", "detail": "checkpoint store unreachable"}


def test_health_probe_failure_does_not_take_down_other_routes():
    """A dead store must not turn the whole app into 500s — the auth gate still
    answers on its own terms."""
    def dead():
        raise RuntimeError("nope")

    c = client(health_probe=dead)
    assert c.get("/health").status_code == 503
    assert c.post("/ask", json={"text": "hi"}).status_code == 401


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
    # a NON-voice caller's custom thread flows through verbatim. (A "voice:*" thread
    # is now treated as a voice turn and person-keyed — see the voice-prefix test —
    # so a plain flow-through must use a non-voice thread id.)
    r = client().post(
        "/ask",
        json={"text": "hi", "thread_id": "cli:pe", "display_name": "Chris"},
        headers={"Authorization": "Bearer sekrit"},
    )
    assert r.json()["reply"].endswith("|Chris|cli:pe")


def test_ask_legacy_voice_thread_person_keys_without_flag():
    # the HA aerys_conversation component posts thread_id="voice:beta" and NO voice
    # flag (predates the tie-in); the Brain must still person-key it onto the owner
    # thread rather than stranding it on voice:beta (the "2nd person_id" bug).
    c = TestClient(build_app(fake_ask, "sekrit", owner_person_id="owner-uuid"))
    r = c.post(
        "/ask",
        json={"text": "hi", "thread_id": "voice:beta", "display_name": "Chris (Voice)"},
        headers={"Authorization": "Bearer sekrit"},
    )
    assert r.json()["thread_id"] == "person:owner-uuid"


def test_ask_voice_flag_person_keys_into_owner_thread():
    # voice=True folds the /ask turn into the owner's continuous 'person:{id}' thread
    # (cross-surface continuity) — the caller's own thread_id is overridden.
    c = TestClient(build_app(fake_ask, "sekrit", owner_person_id="owner-uuid"))
    r = c.post(
        "/ask",
        json={"text": "hi", "thread_id": "whatever", "voice": True},
        headers={"Authorization": "Bearer sekrit"},
    )
    body = r.json()
    assert body["thread_id"] == "person:owner-uuid"        # response reports the used thread
    assert body["reply"].endswith("|person:owner-uuid")    # ask_fn saw the person thread


def test_ask_without_voice_flag_is_unchanged():
    # default voice=False: the caller's thread_id flows through verbatim (byte-for-byte
    # the old behavior), even with an owner configured.
    c = TestClient(build_app(fake_ask, "sekrit", owner_person_id="owner-uuid"))
    r = c.post(
        "/ask", json={"text": "hi", "thread_id": "http:default"},
        headers={"Authorization": "Bearer sekrit"},
    )
    assert r.json()["thread_id"] == "http:default"


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


# --- speaker ID (2026-09-20): who spoke decides whose turn it is -----------------

def _settings_with_speakers(**kw):
    from aerys_v2.config import Settings
    return Settings(_env_file=None, anthropic_api_key="x", owner_person_id="11111111-1111-1111-1111-111111111111",
                    voice_speaker_persons="chris=11111111-1111-1111-1111-111111111111,megan=22222222-2222-2222-2222-222222222222", **kw)


def _speaker_app(**kw):
    return TestClient(build_app(fake_ask, "sekrit", owner_person_id="11111111-1111-1111-1111-111111111111",
                                settings=_settings_with_speakers(**kw)))


def _ask(c, speaker):
    body = {"text": "hi", "thread_id": "voice:beta", "voice": True, "display_name": "Chris (Voice)"}
    if speaker is not None:
        body["speaker"] = speaker
    return c.post("/ask", json=body, headers={"Authorization": "Bearer sekrit"}).json()


def test_untagged_voice_turn_stays_the_owners():
    r = _ask(_speaker_app(), None)
    assert r["thread_id"] == "person:11111111-1111-1111-1111-111111111111" and "|Chris (Voice)|" in r["reply"]


def test_owner_tag_is_a_no_op_and_enrolled_family_gets_their_own_person_thread():
    r = _ask(_speaker_app(), {"id": "chris", "confidence": 0.8})
    assert r["thread_id"] == "person:11111111-1111-1111-1111-111111111111" and "|Chris (Voice)|" in r["reply"]
    r = _ask(_speaker_app(), {"id": "Megan", "confidence": 0.7})
    assert r["thread_id"] == "person:22222222-2222-2222-2222-222222222222" and "|Megan (Voice)|" in r["reply"]


def test_unknown_or_unlisted_or_low_confidence_speaker_is_a_guest_turn():
    for tag in ({"id": "unknown", "confidence": 0.3}, {"id": "dimitri", "confidence": 0.9}):
        r = _ask(_speaker_app(), tag)
        assert r["thread_id"] == "person:voice-guest" and "|Guest (Voice)|" in r["reply"], tag
    r = _ask(_speaker_app(voice_speaker_min_confidence=0.75), {"id": "megan", "confidence": 0.6})
    assert r["thread_id"] == "person:voice-guest"


def test_unknown_speaker_stays_the_owner_when_configured_so():
    r = _ask(_speaker_app(voice_unknown_speaker="owner"), {"id": "unknown", "confidence": 0.2})
    assert r["thread_id"] == "person:11111111-1111-1111-1111-111111111111"


def test_speaker_tag_never_moves_a_text_turn():
    c = _speaker_app()
    r = c.post("/ask", json={"text": "hi", "thread_id": "http:default", "speaker": {"id": "megan", "confidence": 0.9}},
               headers={"Authorization": "Bearer sekrit"}).json()
    assert r["thread_id"] == "http:default"


def test_speaker_person_map_parses_and_skips_junk():
    from aerys_v2.config import Settings, speaker_person_map
    s = Settings(_env_file=None, anthropic_api_key="x", voice_speaker_persons=" Chris = a , megan=b, broken, =c, d= ")
    assert speaker_person_map(s) == {"chris": "a", "megan": "b"}
