"""ASK_ONCE: the HTTP door serves exactly one /ask request and exits.

Chris, 2026-10-07: the morning note (a scheduled turn, nobody waiting on it) rode the
voice container, which is metered on the API key by design. A one-shot run of the
SAME door in a container on the plan sends it there instead: same identity, same
tools, same thread, and none of the --serve watchers (which must stay single).
"""
import json

from aerys_v2.transports.http_api import build_app, serve_once


def test_serve_once_posts_one_ask_through_the_door(tmp_path, capsys):
    seen = []
    app = build_app(lambda text, identity, thread: seen.append((text, identity, thread)) or "note written",
                    "tok", "owner-uuid")
    body = tmp_path / "ask.json"
    body.write_text(json.dumps({"text": "Morning note pass", "thread_id": "aerys:living-note",
                                "display_name": "Morning bell"}))
    assert serve_once(app, str(body), "tok") == 0
    (text, identity, thread), = seen
    assert (text, thread) == ("Morning note pass", "aerys:living-note")
    assert identity["user_id"] == "owner-uuid" and identity["privacy_context"] == "private"
    assert "note written" in capsys.readouterr().out


def test_serve_once_fails_loudly_on_a_refused_request(tmp_path):
    app = build_app(lambda *a: "unused", "tok", "owner-uuid")
    body = tmp_path / "ask.json"
    body.write_text(json.dumps({"text": "x"}))
    assert serve_once(app, str(body), "wrong-token") == 1
