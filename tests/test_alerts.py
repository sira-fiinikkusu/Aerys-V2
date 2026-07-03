"""AlertSink tests — offline, the webhook is faked at the urllib boundary."""

import io
import json

import aerys_v2.alerts as alerts_mod
from aerys_v2.alerts import AlertSink


class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _patch_urlopen(monkeypatch, body: dict | Exception):
    calls = []

    def fake_urlopen(req, timeout=None):
        if isinstance(body, Exception):
            raise body
        calls.append(json.loads(req.data.decode()))
        return FakeResponse(json.dumps(body).encode())

    monkeypatch.setattr(alerts_mod.urllib.request, "urlopen", fake_urlopen)
    return calls


def test_no_url_logs_only():
    assert AlertSink(None).alert("boom") is False  # degrades, never raises


def test_delivered_with_message_id(monkeypatch):
    calls = _patch_urlopen(monkeypatch, {"success": True, "message_id": "123"})
    assert AlertSink("http://x/webhook").alert("boom", source="backup") is True
    assert "backup" in calls[0]["text"] and "boom" in calls[0]["text"]


def test_null_message_id_is_failure(monkeypatch):
    # the webhook's "success:true, message_id:null" lie must not count as delivered
    _patch_urlopen(monkeypatch, {"success": True, "message_id": None})
    assert AlertSink("http://x/webhook").alert("boom") is False


def test_webhook_exception_swallowed(monkeypatch):
    _patch_urlopen(monkeypatch, OSError("network down"))
    assert AlertSink("http://x/webhook").alert("boom") is False  # sink never raises


def test_oversize_message_truncated(monkeypatch):
    calls = _patch_urlopen(monkeypatch, {"message_id": "1"})
    AlertSink("http://x/webhook").alert("x" * 5000)
    assert len(calls[0]["text"]) <= 1900
