"""The Kael line's return direction (task #66) — offline, fakes all the way down.

What these prove: her message_kael tool carries the ORIGIN thread_id from the
runtime config (the return address), the /kael-note door injects a
clearly-attributed note into exactly that thread and 503s honestly when
unarmed, notes default family_visible FALSE (the owner's "tests do poison
memory" rule), and the family splice appears ONLY on the owner's threads —
guests structurally get nothing.
"""

import httpx
import pytest
from fastapi.testclient import TestClient

from aerys_v2.kael_line import NOTE_PREFIX, family_notes_fn_for, kael_note_for
from aerys_v2.tools.message_kael import build_message_kael_tool
from aerys_v2.transports.http_api import build_app


class FakeGraph:
    def __init__(self):
        self.updates = []

    def update_state(self, config, values):
        self.updates.append((config, values))


class FakeSettings:
    database_url = None
    owner_person_id = "owner-uuid"


# ─────────────────────────────────────────────── tool carries the return address


def transport_capture(captured):
    def handler(request: httpx.Request) -> httpx.Response:
        import json

        captured.append(json.loads(request.content))
        return httpx.Response(202)

    return httpx.MockTransport(handler)


def test_tool_sends_origin_thread_id_from_config():
    captured = []
    tool = build_message_kael_tool(
        "http://desk.test/aerys/message", "tok",
        client=httpx.Client(transport=transport_capture(captured)),
    )
    out = tool.invoke(
        {"message": "the office lights are acting up"},
        config={"configurable": {"thread_id": "person:chris"}},
    )
    assert "Delivered" in out
    assert captured[0]["thread_id"] == "person:chris"
    assert captured[0]["message"] == "the office lights are acting up"


def test_tool_omits_thread_id_when_config_has_none():
    captured = []
    tool = build_message_kael_tool(
        "http://desk.test/aerys/message", "tok",
        client=httpx.Client(transport=transport_capture(captured)),
    )
    out = tool.invoke({"message": "ping"}, config={"configurable": {}})
    assert "Delivered" in out
    assert "thread_id" not in captured[0]


# ─────────────────────────────────────────────────────── note injector semantics


def test_note_injects_attributed_message_into_the_named_thread():
    g = FakeGraph()
    note = kael_note_for(g, FakeSettings())
    msg_id = note("person:aerys-ping", "phone is SM-F966U, 44% charging")
    (config, values), = g.updates
    assert config["configurable"]["thread_id"] == "person:aerys-ping"
    msg = values["messages"][0]
    assert msg.content.startswith(NOTE_PREFIX)
    assert "SM-F966U" in msg.content
    assert msg.id == msg_id
    assert msg.additional_kwargs["content_privacy"] == "private"


# ──────────────────────────────────────────────────────────── the /kael-note door


def app_with(note_fn):
    return build_app(lambda *a, **k: "unused", "sekrit", kael_note_fn=note_fn)


def test_door_requires_token_and_calls_note_fn():
    calls = []
    app = app_with(lambda tid, text, fam: calls.append((tid, text, fam)) or "kn-1")
    c = TestClient(app)
    assert c.post("/kael-note", json={"thread_id": "t", "text": "x"}).status_code in (401, 403)
    r = c.post(
        "/kael-note",
        json={"thread_id": "person:chris", "text": "note body"},
        headers={"Authorization": "Bearer sekrit"},
    )
    assert r.status_code == 200
    assert r.json() == {"ok": True, "message_id": "kn-1"}
    # family_visible DEFAULTS FALSE — tests must not enter the family lore.
    assert calls == [("person:chris", "note body", False)]


def test_door_passes_family_visible_flag_and_503s_unarmed():
    calls = []
    app = app_with(lambda tid, text, fam: calls.append(fam) or "kn-2")
    c = TestClient(app)
    r = c.post(
        "/kael-note",
        json={"thread_id": "t", "text": "real talk", "family_visible": True},
        headers={"Authorization": "Bearer sekrit"},
    )
    assert r.status_code == 200 and calls == [True]

    unarmed = TestClient(app_with(None))
    r = unarmed.post(
        "/kael-note",
        json={"thread_id": "t", "text": "x"},
        headers={"Authorization": "Bearer sekrit"},
    )
    assert r.status_code == 503


def test_door_rejects_blank_fields():
    app = app_with(lambda *a: "kn-3")
    c = TestClient(app)
    r = c.post(
        "/kael-note",
        json={"thread_id": "  ", "text": "x"},
        headers={"Authorization": "Bearer sekrit"},
    )
    assert r.status_code == 422


# ─────────────────────────────────────────────────────────── family splice gating


def test_family_splice_is_none_without_db_or_owner():
    assert family_notes_fn_for(FakeSettings()) is None  # database_url None

    class NoOwner:
        database_url = "postgresql://x/aerys_v2"
        owner_person_id = None

    assert family_notes_fn_for(NoOwner()) is None


def test_family_splice_returns_empty_for_non_owner(monkeypatch):
    class DBSettings:
        database_url = "postgresql://unused/aerys_v2"
        owner_person_id = "owner-uuid"

    fn = family_notes_fn_for(DBSettings())
    # A guest identity never reaches the database at all.
    assert fn({"user_id": "guest-uuid"}) == ""
    assert fn({}) == ""


# ─────────────────────────────────── action graph carries the splice (owner nit 8/05)


def test_action_graph_splices_family_notes_for_owner():
    from langchain_core.messages import AIMessage, HumanMessage

    from aerys_v2.factory import build_action_graph

    seen = {}

    class CaptureModel:
        def invoke(self, messages):
            seen["system"] = messages[0].content
            return AIMessage(content="done")

    g = build_action_graph(
        CaptureModel(), "SOUL", tools=[],
        family_notes_fn=lambda identity: "\n\n[Family-visible notes]\n- note one",
    )
    g.invoke(
        {"messages": [HumanMessage(content="turn on the lights")]},
        {"configurable": {"thread_id": "t", "identity": {"user_id": "owner-uuid",
                                                          "privacy_context": "private"}}},
    )
    assert "[Family-visible notes]" in seen["system"]
    assert "note one" in seen["system"]


def test_action_graph_without_family_fn_is_unchanged():
    from langchain_core.messages import AIMessage, HumanMessage

    from aerys_v2.factory import build_action_graph

    seen = {}

    class CaptureModel:
        def invoke(self, messages):
            seen["system"] = messages[0].content
            return AIMessage(content="done")

    g = build_action_graph(CaptureModel(), "SOUL", tools=[])
    g.invoke(
        {"messages": [HumanMessage(content="hi")]},
        {"configurable": {"thread_id": "t"}},
    )
    assert "Family-visible" not in seen["system"]


# --- T-25 (11/18): a long text goes as several messages, never cut -------------------------

class _Resp:
    def __init__(self, body=b'{"id": "chan"}', status=200):
        self.body, self.status = body, status
    def read(self, *a):
        return self.body
    def __enter__(self):
        return self
    def __exit__(self, *a):
        return False


def _capture(monkeypatch):
    import io, json as _json, urllib.request
    sent = []
    def urlopen(req, timeout=None):
        sent.append((req.full_url, _json.loads(req.data.decode())))
        return io.BytesIO(b'{"id": "chan"}') if req.full_url.endswith('/users/@me/channels') else _Resp()
    monkeypatch.setattr(urllib.request, 'urlopen', urlopen)
    return sent


def test_a_long_discord_dm_goes_as_several_messages_with_nothing_cut(monkeypatch):
    from aerys_v2.kael_line import _discord_dm_send
    sent = _capture(monkeypatch)
    report = '\n\n'.join(f'Paragraph {i}: ' + 'word ' * 150 for i in range(6))      # ~4.6k chars
    _discord_dm_send('token', 'user', report)
    parts = [body['content'] for url, body in sent if url.endswith('/messages')]
    assert len(parts) >= 3 and all(len(p) <= 1990 for p in parts)
    assert ' '.join(' '.join(parts).split()) == ' '.join(report.split())          # every word arrived


def test_a_short_dm_is_still_one_message(monkeypatch):
    from aerys_v2.kael_line import _discord_dm_send
    sent = _capture(monkeypatch)
    _discord_dm_send('token', 'user', 'Solved in 3.', silent=True)
    parts = [body for url, body in sent if url.endswith('/messages')]
    assert parts == [{'content': 'Solved in 3.', 'flags': 4096}]


def test_a_long_telegram_text_goes_as_several(monkeypatch):
    from aerys_v2.kael_line import _telegram_send
    sent = _capture(monkeypatch)
    report = '\n\n'.join(f'Paragraph {i}: ' + 'word ' * 300 for i in range(6))      # ~9k chars
    _telegram_send('token', 'chat', report)
    parts = [body['text'] for url, body in sent]
    assert len(parts) >= 3 and all(len(p) <= 4000 for p in parts)


def test_a_rate_limited_part_is_retried_and_nothing_is_sent_twice():
    """Codex 11/18: a 429 on part 2, then a retry of the whole report, delivered A, A, B."""
    import urllib.error
    from aerys_v2.kael_line import _send_parts
    sent, failures, waited = [], {'B': 1}, []
    def post(part):
        if failures.get(part):
            failures[part] -= 1
            raise urllib.error.HTTPError('u', 429, 'slow down', {'Retry-After': '2'}, None)
        sent.append(part)
    _send_parts(['A', 'B', 'C'], post, sleep=waited.append)
    assert sent == ['A', 'B', 'C'] and waited == [2.0]


def test_a_later_part_that_never_goes_is_said_once_and_the_start_is_not_resent():
    import urllib.error
    from aerys_v2.kael_line import REST_DIDNT_SEND, _send_parts
    sent = []
    def post(part):
        if part == 'B':
            raise urllib.error.HTTPError('u', 500, 'down', {}, None)
        sent.append(part)
    _send_parts(['A', 'B', 'C'], post, sleep=lambda s: None)       # returns: no whole-report retry
    assert sent == ['A', REST_DIDNT_SEND]


def test_a_first_part_that_never_goes_raises_so_the_whole_report_can_be_retried():
    import pytest, urllib.error
    from aerys_v2.kael_line import _send_parts
    def post(part):
        raise urllib.error.HTTPError('u', 429, 'slow down', {'Retry-After': '1'}, None)
    with pytest.raises(urllib.error.HTTPError):
        _send_parts(['A', 'B'], post, sleep=lambda s: None)
