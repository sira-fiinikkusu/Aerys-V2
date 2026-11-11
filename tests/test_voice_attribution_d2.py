"""D2 (Chris 2026-11-08, board #176): a voice that wasn't clearly Chris's never becomes
his memory. Chat still answers an unsure or untagged turn as his (the 9/22 rule); the
turn is marked, the extractor skips it, and the remember tool won't keep its facts."""
from fastapi.testclient import TestClient

from aerys_v2.state import VOICE_OWNER_ASSUMED_KEY, VOICE_OWNER_ASSUMED_MARKER
from aerys_v2.transports.http_api import build_app

OWNER = "11111111-1111-1111-1111-111111111111"


def _app(**kw):
    from aerys_v2.config import Settings
    seen = []

    def ask(text, identity, thread_id):
        seen.append(dict(identity))
        return "ok"

    settings = Settings(_env_file=None, anthropic_api_key="x", owner_person_id=OWNER,
                        voice_speaker_persons=f"chris={OWNER},robin=22222222-2222-2222-2222-222222222222", **kw)
    return TestClient(build_app(ask, "sekrit", owner_person_id=OWNER, settings=settings)), seen


def _voice(c, speaker=None):
    body = {"text": "hi", "thread_id": "voice:beta", "voice": True, "display_name": "Chris (Voice)"}
    if speaker is not None:
        body["speaker"] = speaker
    return c.post("/ask", json=body, headers={"Authorization": "Bearer sekrit"})


def test_untagged_and_unsure_voice_turns_are_his_by_default_only():
    c, seen = _app()
    _voice(c)                                              # untagged (recognizer down, no prints)
    _voice(c, {"id": "unknown", "confidence": 0.31})       # the unsure band
    assert [s.get(VOICE_OWNER_ASSUMED_KEY) for s in seen] == [True, True]
    assert all(s["user_id"] == OWNER for s in seen), "chat still answers him (9/22 rule)"


def test_a_clear_match_robin_and_guests_are_not_marked():
    c, seen = _app()
    _voice(c, {"id": "chris", "confidence": 0.7})          # a named match to his print
    _voice(c, {"id": "robin", "confidence": 0.7})          # her own person, not his
    _voice(c, {"id": "unknown", "confidence": 0.1})        # a stranger: a guest turn
    assert [bool(s.get(VOICE_OWNER_ASSUMED_KEY)) for s in seen] == [False, False, False]


def test_a_named_owner_match_under_the_floor_is_not_a_clear_match():
    c, seen = _app(voice_speaker_min_confidence=0.75, voice_unknown_speaker="owner")
    _voice(c, {"id": "chris", "confidence": 0.6})
    assert seen[-1].get(VOICE_OWNER_ASSUMED_KEY) is True


def test_text_turns_are_never_marked():
    c, seen = _app()
    c.post("/ask", json={"text": "hi", "thread_id": "http:default"}, headers={"Authorization": "Bearer sekrit"})
    assert VOICE_OWNER_ASSUMED_KEY not in seen[-1]


def test_the_turn_row_carries_the_marker():
    import json
    from aerys_v2.turns import build_turn_row
    row = build_turn_row(thread_id=f"person:{OWNER}", identity={"user_id": OWNER, "voice": True,
                         VOICE_OWNER_ASSUMED_KEY: True}, input_text="I'm allergic to shellfish", latency_ms=1)
    assert VOICE_OWNER_ASSUMED_MARKER in json.loads(row["degraded"])
    clear = build_turn_row(thread_id=f"person:{OWNER}", identity={"user_id": OWNER, "voice": True},
                           input_text="hi", latency_ms=1)
    assert VOICE_OWNER_ASSUMED_MARKER not in json.loads(clear["degraded"])


def test_the_extractor_skips_marked_turns():
    from aerys_v2.workers.extraction import V2_TURNS_SQL
    assert "@> '[\"voice_owner_assumed\"]'::jsonb" in V2_TURNS_SQL
    assert "AND NOT COALESCE(t.degraded" in V2_TURNS_SQL


def test_remember_wont_keep_a_fact_from_an_unsure_voice():
    from aerys_v2.tools.remember import NOT_SURE_WHO, build_remember_tool
    wrote = []
    tool = build_remember_tool(lambda record: wrote.append(record) or "insert", key_labeler=lambda f: "preference.food")
    unsure = {"configurable": {"identity": {"user_id": OWNER, "voice": True, VOICE_OWNER_ASSUMED_KEY: True}}}
    assert tool.invoke({"fact": "Chris is allergic to shellfish"}, config=unsure) == NOT_SURE_WHO
    assert wrote == []
    sure = {"configurable": {"identity": {"user_id": OWNER, "voice": True}}}
    assert tool.invoke({"fact": "Chris is allergic to shellfish"}, config=sure).startswith("Kept")
    assert len(wrote) == 1


def test_member_below_floor_keeps_identity_but_gate_requires_both_buttons():
    from aerys_v2.config import Settings
    from aerys_v2.hands_gate import asker, VOICE_UNSURE, MEMBER_VOICE
    from aerys_v2.transports.http_api import AskRequest

    member = "22222222-2222-2222-2222-222222222222"
    seen = []
    settings = Settings(_env_file=None, anthropic_api_key="placeholder", owner_person_id=OWNER,
                        voice_speaker_persons=f"robin={member}", voice_speaker_min_confidence=0.75,
                        home_job_members=f"{member}=home-member")
    app = build_app(lambda text, identity, thread_id: seen.append((identity, thread_id)) or "ok",
                    None, owner_person_id=OWNER, settings=settings)
    endpoint = next(r.endpoint for r in app.routes if r.path == "/ask")
    for confidence, expected in ((0.6, VOICE_UNSURE), (0.8, MEMBER_VOICE)):
        endpoint(AskRequest(text="hello", voice=True, speaker={"id": "robin", "confidence": confidence}))
        identity, thread = seen[-1]
        assert identity["user_id"] == member
        assert asker(identity, thread, OWNER, {member: "home-member"}) == expected
