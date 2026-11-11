"""BENCH_MODE (board #176 s1-03, the voice bench): a copy of the --serve door that
answers HTTP and nothing else.

The voice flip is gated on a live bench of her voice path. Running it on a second
--serve container would start a second copy of every watcher (her morning brief, the
reach-out desk, storm alerts, ticket relays, home-job starts) and write the bench's
commands into the owner's conversation. These pin the two halves of the switch: voice
turns ride the bench thread, and no watcher starts before the bench door's exit.
"""
import re
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient

from aerys_v2.config import Settings
from aerys_v2.state import VOICE_OWNER_ASSUMED_KEY
from aerys_v2.transports.http_api import BENCH_VOICE_THREAD, build_app

CLI = Path(__file__).resolve().parents[1] / "src" / "aerys_v2" / "cli.py"


def bench_settings(on: bool):
    # Only the fields the door reads; build_app never needs the rest.
    return SimpleNamespace(bench_mode=on, voice_speaker_persons="", voice_unknown_speaker="owner",
                           voice_speaker_min_confidence=0.7, voice_guest_demote_below=0.5)


def door(on: bool):
    seen = []

    def ask(text, identity, thread):
        seen.append((identity, thread))
        return "ok"

    app = build_app(ask, "tok", owner_person_id="owner-uuid", settings=bench_settings(on))
    return TestClient(app), seen


def test_bench_mode_is_off_by_default():
    assert Settings.model_fields["bench_mode"].default is False


def test_bench_voice_turn_rides_the_bench_thread_not_his():
    c, seen = door(True)
    r = c.post("/ask", json={"text": "dim the sunroom lights to half", "voice": True},
               headers={"Authorization": "Bearer tok"})
    assert r.json()["thread_id"] == BENCH_VOICE_THREAD == "bench:voice"
    (identity, thread), = seen
    assert thread == "bench:voice"
    # Still the owner's house (the bench measures what HE gets), but an untagged voice
    # stays his by default only, so the memory extractor never keeps it.
    assert identity["user_id"] == "owner-uuid" and identity["voice"] is True
    assert identity[VOICE_OWNER_ASSUMED_KEY] is True


def test_bench_openai_voice_path_rides_the_bench_thread():
    c, seen = door(True)
    r = c.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "is the office light on"}]},
               headers={"Authorization": "Bearer tok"})
    assert r.status_code == 200
    (_, thread), = seen
    assert thread == "bench:voice"


def test_live_door_still_folds_voice_into_his_thread():
    c, seen = door(False)
    r = c.post("/ask", json={"text": "hi", "voice": True}, headers={"Authorization": "Bearer tok"})
    assert r.json()["thread_id"] == "person:owner-uuid"


def test_no_watcher_starts_before_the_bench_exit():
    # The serve block: everything from "--serve" to the "--discord" branch.
    src = CLI.read_text()
    serve = src[src.index('if "--serve" in sys.argv'):src.index('if "--discord" in sys.argv')]
    exit_at = serve.index("if settings.bench_mode:\n")
    bench_exit = serve[exit_at:exit_at + 600]
    assert "uvicorn.run(" in bench_exit and "sys.exit(0)" in bench_exit
    starts = [m.start() for m in re.finditer(r"\bstart_[a-z_]+\(|\.watcher\.start\(\)", serve)]
    assert starts, "the serve block starts no watcher at all? this test is reading the wrong text"
    early = [serve[s:s + 40] for s in starts if s < exit_at]
    assert early == [], f"a watcher starts before the bench exit: {early}"


def test_bench_copy_speaks_nothing_and_drives_no_face():
    src = CLI.read_text()
    for wiring in ("face_push = None if settings.bench_mode else face_pusher_for(settings)",
                   "speak_fn = None if settings.bench_mode else speak_fn_for(settings)",
                   "followup_router = None if settings.bench_mode else followup_router_for(settings)",
                   "display_push = None if settings.bench_mode else display_push_for(settings)",
                   "speaker_redirect=None if settings.bench_mode else speaker_redirect_for(settings)"):
        assert wiring in src, wiring
