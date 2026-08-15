"""Gap #48 — full-text ink push around upstream ESPHome's 500-byte VA cap.

voice_assistant.cpp resizes pipeline event text to 497 chars + '...' — the
brain can't change that, so a display-mapped device gets the FULL reply
re-sent through its uncapped ESPHome user action, delayed to land after the
capped write. These tests pin the closure's arming, targeting, and payload,
and that the voice banter path actually fires it with the device_id.
"""

from types import SimpleNamespace

import httpx

from aerys_v2.factory import build_graph, display_push_for
from aerys_v2.service import ask
from test_honesty_fixes import CHRIS, chat_router, fake_model


class _Secret:
    def __init__(self, value: str) -> None:
        self._value = value

    def get_secret_value(self) -> str:
        return self._value


def settings_with(token="t", displays="dev-1=esphome.sticky_show_followup"):
    return SimpleNamespace(
        ha_token=_Secret(token) if token else None,
        ha_base_url="http://ha.test:8123",
        ha_display_followups=displays,
    )


def test_unarmed_without_token_or_mappings():
    assert display_push_for(settings_with(token=None)) is None
    assert display_push_for(settings_with(displays="")) is None


def test_mapped_device_posts_full_text(monkeypatch):
    calls = []

    def fake_post(url, headers=None, json=None, timeout=None):
        calls.append((url, json))
        return httpx.Response(200, request=httpx.Request("POST", url))

    monkeypatch.setattr(httpx, "post", fake_post)
    push = display_push_for(settings_with(), delay_s=0)
    long_reply = "x" * 1900  # far past the firmware's 500-byte guillotine
    push(long_reply, "dev-1")
    # daemon thread with delay_s=0 — give it a beat
    import time

    for _ in range(50):
        if calls:
            break
        time.sleep(0.05)
    assert calls, "push never fired"
    url, payload = calls[0]
    assert url == "http://ha.test:8123/api/services/esphome/sticky_show_followup"
    assert payload == {"message": long_reply}  # FULL text, no cap


def test_unmapped_device_and_empty_text_noop(monkeypatch):
    calls = []
    monkeypatch.setattr(httpx, "post", lambda *a, **k: calls.append(a))
    push = display_push_for(settings_with(), delay_s=0)
    push("hello", "other-device")
    push("hello", None)
    push("", "dev-1")
    import time

    time.sleep(0.2)
    assert calls == []


def test_voice_banter_fires_display_push_with_device_id():
    pushes = []
    reply = "Here is a very long story that the pipeline would cap."
    graph = build_graph(fake_model(reply), soul="s")
    voice_chris = {**CHRIS, "voice": True, "device_id": "dev-1"}
    out = ask(
        graph,
        "tell me a story",
        identity=voice_chris,
        thread_id="t-display-push",
        router=chat_router,
        action_graph=build_graph(fake_model(reply), soul="s"),
        display_push=lambda text, device_id: pushes.append((text, device_id)),
    )
    assert out == reply
    assert pushes == [(reply, "dev-1")]
