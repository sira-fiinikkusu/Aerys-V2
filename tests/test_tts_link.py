"""The voice address: which pipeline run is the one being spoken now."""

from datetime import datetime, timedelta, timezone

from aerys_v2 import tts_link
from aerys_v2.tts_link import current_voice_link, pick_link

NOW = datetime(2026, 11, 28, 4, 22, 17, tzinfo=timezone.utc)


def run(seconds_ago: float, url="/api/tts_proxy/abc.flac", kind="run-start"):
    stamp = (NOW - timedelta(seconds=seconds_ago)).isoformat()
    return stamp, [{"type": "stt-start", "data": {}},
                   {"type": kind, "data": {"tts_output": {"url": url, "mime_type": "audio/flac"}}}]


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
