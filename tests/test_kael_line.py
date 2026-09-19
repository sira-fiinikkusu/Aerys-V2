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
