"""The voice address: which pipeline run is the one being spoken now."""

from datetime import datetime, timedelta, timezone

from aerys_v2 import tts_link
from aerys_v2.tts_link import current_voice_link, pick_link

NOW = datetime(2026, 11, 28, 4, 22, 17, tzinfo=timezone.utc)


def run(seconds_ago: float, url="/api/tts_proxy/abc.flac", kind="run-start", ended_ago=None):
    stamp = (NOW - timedelta(seconds=seconds_ago)).isoformat()
    events = [{"type": "stt-start", "data": {}},
              {"type": kind, "data": {"tts_output": {"url": url, "mime_type": "audio/flac"}}}]
    if ended_ago is not None:
        events.append({"type": "run-end", "data": None,
                       "timestamp": (NOW - timedelta(seconds=ended_ago)).isoformat()})
    return stamp, events


def test_the_run_happening_now_gives_its_address():
    assert pick_link([run(15)], now=NOW) == "/api/tts_proxy/abc.flac"


def test_an_old_run_is_not_the_reply_being_spoken():
    assert pick_link([run(91)], now=NOW) is None
    assert pick_link([run(89)], now=NOW) == "/api/tts_proxy/abc.flac"


def test_the_newest_of_several_wins():
    older, newer = run(40, "/api/tts_proxy/old.flac"), run(5, "/api/tts_proxy/new.flac")
    assert pick_link([older, newer], now=NOW) == "/api/tts_proxy/new.flac"
    assert pick_link([newer, older], now=NOW) == "/api/tts_proxy/new.flac"


def test_only_a_home_assistant_voice_path_is_ever_returned():
    for bad in ("http://elsewhere/x.flac", "/api/other/x", "", None, 7):
        assert pick_link([run(5, bad)], now=NOW) is None
    assert pick_link([run(5, kind="tts-end")], now=NOW) is None
    assert pick_link([("not a time", [])], now=NOW) is None
    assert pick_link([], now=NOW) is None


def test_a_house_that_cannot_be_asked_gives_none_and_never_raises(monkeypatch):
    async def boom(*_a, **_k):
        raise ConnectionError("no house")

    monkeypatch.setattr(tts_link, "_ask", boom)
    assert current_voice_link("http://ha", "t") is None


def test_the_full_address_is_the_house_plus_the_path(monkeypatch):
    async def fake(base_url, token, timeout_s):
        return base_url.rstrip("/") + "/api/tts_proxy/abc.flac"

    monkeypatch.setattr(tts_link, "_ask", fake)
    assert current_voice_link("http://ha:8123/", "t") == "http://ha:8123/api/tts_proxy/abc.flac"


def test_a_reply_that_just_ended_its_run_still_counts():
    # the pipeline run ends within milliseconds of her answering; the lookup arrives a moment later
    assert pick_link([run(15, ended_ago=0.2)], now=NOW) == "/api/tts_proxy/abc.flac"
    assert pick_link([run(15, ended_ago=3.9)], now=NOW) == "/api/tts_proxy/abc.flac"


def test_an_acknowledgement_that_finished_is_never_handed_over_for_a_later_follow_up():
    # "On it." ran 30 s ago and ended then; the follow-up she speaks now has no run of its own
    assert pick_link([run(31, "/api/tts_proxy/ack.flac", ended_ago=30)], now=NOW) is None
    assert pick_link([run(12, "/api/tts_proxy/ack.flac", ended_ago=4.1)], now=NOW) is None


def test_a_newer_run_with_no_sound_does_not_hide_the_spoken_one():
    typed = ((NOW - timedelta(seconds=2)).isoformat(), [{"type": "run-start", "data": {}}])
    assert pick_link([typed, run(6, "/api/tts_proxy/spoken.flac")], now=NOW) == "/api/tts_proxy/spoken.flac"


class FakeHouse:
    """Stands in for the house's websocket: answers the four kinds of question _ask sends."""

    def __init__(self, runs, auth_ok=True):
        self.runs, self.auth_ok, self.inbox, self.asked = runs, auth_ok, ["{}"], []

    async def __aenter__(self): return self
    async def __aexit__(self, *a): return False

    async def recv(self):
        return self.inbox.pop(0)

    async def send(self, raw):
        import json
        m = json.loads(raw)
        self.asked.append(m.get("type"))
        if m["type"] == "auth":
            self.inbox.append(json.dumps({"type": "auth_ok" if self.auth_ok else "auth_invalid"}))
        elif m["type"] == "assist_pipeline/pipeline/list":
            self.inbox.append(json.dumps({"id": m["id"], "success": True, "result": {"pipelines": [{"id": p} for p in self.runs]}}))
        elif m["type"] == "assist_pipeline/pipeline_debug/list":
            self.inbox.append(json.dumps({"id": m["id"], "success": True, "result": {"pipeline_runs": [
                {"pipeline_run_id": rid, "timestamp": stamp} for rid, (stamp, _ev) in self.runs[m["pipeline_id"]].items()]}}))
        else:
            stamp, events = self.runs[m["pipeline_id"]][m["pipeline_run_id"]]
            self.inbox.append(json.dumps({"id": m["id"], "success": True, "result": {"events": events}}))


def _ask_with(monkeypatch, house):
    import asyncio
    import sys
    import types

    fake = types.SimpleNamespace(connect=lambda *a, **k: house)
    monkeypatch.setitem(sys.modules, "websockets", fake)
    return asyncio.run(tts_link._ask("http://ha:8123", "secret-token", 2.0))


def _recent(seconds_ago, url, ended_ago=None):
    now = datetime.now(timezone.utc)
    ev = [{"type": "run-start", "data": {"tts_output": {"url": url}}}]
    if ended_ago is not None:
        ev.append({"type": "run-end", "data": None, "timestamp": (now - timedelta(seconds=ended_ago)).isoformat()})
    return (now - timedelta(seconds=seconds_ago)).isoformat(), ev


def test_asking_the_house_finds_the_reply_being_spoken_across_pipelines(monkeypatch):
    house = FakeHouse({"voice": {"r1": _recent(40, "/api/tts_proxy/old.flac", ended_ago=39),
                                 "r2": _recent(5, "/api/tts_proxy/now.flac", ended_ago=0.1)},
                       "typed": {"r9": _recent(2, None)}})
    assert _ask_with(monkeypatch, house) == "http://ha:8123/api/tts_proxy/now.flac"
    assert house.asked[0] == "auth"


def test_asking_the_house_gives_nothing_when_only_a_finished_acknowledgement_is_there(monkeypatch):
    house = FakeHouse({"voice": {"r1": _recent(30, "/api/tts_proxy/ack.flac", ended_ago=29)}})
    assert _ask_with(monkeypatch, house) is None


def test_a_refused_token_gives_nothing_and_asks_nothing_more(monkeypatch):
    house = FakeHouse({"voice": {"r1": _recent(5, "/api/tts_proxy/now.flac")}}, auth_ok=False)
    assert _ask_with(monkeypatch, house) is None
    assert house.asked == ["auth"]


def test_no_websocket_library_gives_none_and_never_raises(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "websockets", None)      # makes `import websockets` fail
    assert current_voice_link("http://ha", "t") is None

