"""Per-invoke raw seam caching, through the action agent."""
from collections import Counter
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool

import aerys_v2.factory as factory
from aerys_v2.action_loop import build_agent_action_graph
from aerys_v2.service import _invoke_action_graph
from test_action_loop_agent import Rec


@pytest.mark.parametrize("memo", [False, True])
@pytest.mark.parametrize("privacy", ["public", "private"])
def test_raw_seams_once_per_invoke(memo, privacy, monkeypatch):
    from aerys_v2 import images
    counts = Counter()
    def seam(name, result):
        def read(*args):
            counts[name] += 1
            return result
        return read
    monkeypatch.setattr(factory, "_OWNER_ID", "sample")
    monkeypatch.setattr(factory, "_where_when_line", seam("clock", "noon"))
    monkeypatch.setattr(images, "fetch_image", seam("image", ("image/png", b"image")))
    # Preserve the normal default-fetch behavior for un-memoized callers too.
    monkeypatch.setattr(images.inline_current_images, "__kwdefaults__", {"fetch": images.fetch_image})
    @tool
    def echo() -> str:
        """Return a receipt."""
        return "ok"
    def script():
        return [AIMessage(content="", tool_calls=[{"id": "call", "name": "echo", "args": {}}]),
                AIMessage(content="Done")]
    model = Rec(label="model", script=script() + script(), log=[])
    graph = build_agent_action_graph(model, "s", [echo], context_fn=seam("profile", "profile"),
                  family_notes_fn=seam("family", "family"),
                  portable_context_fn=seam("portable", "portable"),
                  room_context_fn=seam("room", "room"),
                  voice_room_fn=seam("voice_room", "voice room"),
                  presence_fn=seam("presence", factory.PresenceSnapshot([], whereabouts=("home", ""))))
    config = {"configurable": {"identity": {"user_id": "sample", "voice": True,
              "privacy_context": privacy, "device_id": "satellite", "channel_id": "room"},
              "room_sink": {}}, "recursion_limit": 10}
    messages = [HumanMessage(content="check https://cdn.discordapp.com/attachments/a/b/photo.png")]
    for turn in range(1, 3):
        if memo:
            _invoke_action_graph(graph, messages, config)
        else:
            graph.invoke({"messages": messages}, config)
        expected = turn * (1 if memo else 2)
        for name in ("profile", "family", "voice_room", "image",
                     *(('portable', 'presence') if privacy == "private" else ('room',))):
            assert counts[name] == expected, (name, counts)
        assert counts["clock"] == turn * 2
        assert "action_context_memo" not in config["configurable"]
    if privacy == "public":
        assert config["configurable"]["room_sink"]["room"] == "room"


@pytest.mark.parametrize("failed", ["", None, [], RuntimeError("offline")])
def test_failed_seam_retried(failed):
    from aerys_v2.action_context import memoized
    calls = []
    def read(arg):
        calls.append(arg)
        if len(calls) == 1:
            if isinstance(failed, Exception):
                raise failed
            return failed
        return "ok"
    cached = memoized(read, {})
    try:
        cached("same")
    except RuntimeError:
        pass
    assert cached("same") == cached("same") == "ok"
    assert calls == ["same", "same"]


def test_exact_arguments_and_mutable_identity():
    from aerys_v2.action_context import memoized
    calls = []
    cached = memoized(lambda *args: calls.append(args) or "ok", {})
    identity = {"privacy_context": "private"}
    cached(identity, "first")
    cached(identity, "first")
    cached(identity, "second")
    identity["privacy_context"] = "public"
    cached(identity, "second")
    assert len(calls) == 3


@pytest.mark.parametrize("memo", [False, True])
@pytest.mark.parametrize("new_person", [False, True])
def test_people_named_by_tool_are_added_on_next_pass(memo, new_person, monkeypatch):
    from datetime import datetime, timezone
    import psycopg
    import aerys_v2.services.context as context
    from aerys_v2.services.people_memory import PEOPLE_SQL

    calls = Counter()
    class Conn:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def execute(self, sql, params=None):
            if sql == PEOPLE_SQL:
                calls["roster"] += 1
                rows = [("person-a", "Avery"), ("person-b", "Blake")]
            else:
                name = "Blake" if "Blake" in params["pattern"] else "Avery"
                calls[name] += 1
                rows = [(f"{name} likes puzzles", None, datetime(2026, 1, 1, tzinfo=timezone.utc))]
            return SimpleNamespace(fetchall=lambda: rows)
    def connect(*args):
        calls["connect"] += 1
        return Conn()
    monkeypatch.setattr(psycopg, "connect", connect)
    monkeypatch.setattr(context, "build_context", lambda *a, **k: "profile")
    seam = factory.context_fn_for(SimpleNamespace(memories_database_url="postgresql://example.invalid/db"),
                                  profile_only=True)
    @tool
    def lookup() -> str:
        """Look up the next person."""
        return "Ask Blake too." if new_person else "Avery is ready."
    model = Rec(label="model", script=[AIMessage(content="", tool_calls=[
        {"id": "call", "name": "lookup", "args": {}}]), AIMessage(content="Done")], log=[])
    graph = build_agent_action_graph(model, "s", [lookup], context_fn=seam)
    messages = [HumanMessage(content="Ask Avery")]
    config = {"configurable": {"identity": {"user_id": "sample", "privacy_context": "private"}}}
    if memo:
        _invoke_action_graph(graph, messages, config)
    else:
        graph.invoke({"messages": messages}, config)
    assert "About Avery" in model.log[0]["messages"][0][1]
    assert "About Blake" not in model.log[0]["messages"][0][1]
    assert ("About Blake" in model.log[1]["messages"][0][1]) is new_person
    assert calls["roster"] == (1 if memo else 2)
    assert calls["Avery"] == (2 if memo else 4)  # WHO + RECENT per section
    assert calls["Blake"] == (2 if new_person else 0)
    assert calls["connect"] == (2 + int(new_person) if memo else 4)  # one lazy people connection per pass with misses


def test_profile_and_image_failures_are_retried_in_graph(monkeypatch, caplog):
    from aerys_v2 import images
    calls = Counter()
    def profile(*args):
        calls["profile"] += 1
        return "" if calls["profile"] == 1 else "profile recovered"
    def image(url):
        calls["image"] += 1
        return None if calls["image"] == 1 else ("image/png", b"image")
    monkeypatch.setattr(images, "fetch_image", image)
    @tool
    def echo() -> str:
        """Return a receipt."""
        return "ok"
    model = Rec(label="model", script=[AIMessage(content="", tool_calls=[
        {"id": f"call{i}", "name": "echo", "args": {}}]) for i in range(2)] +
        [AIMessage(content="Done")], log=[])
    graph = build_agent_action_graph(model, "s", [echo], context_fn=profile)
    with caplog.at_level("DEBUG", logger="aerys_v2.factory"):
        _invoke_action_graph(graph, [HumanMessage(
            content="check https://cdn.discordapp.com/attachments/a/b/photo.png")], {})
    assert calls == {"profile": 2, "image": 2}
    assert "profile recovered" not in model.log[0]["messages"][0][1]
    assert "profile recovered" in model.log[1]["messages"][0][1]
    assert sum("action prompt built in" in r.message for r in caplog.records) == 3


def test_cached_reads_keep_privacy_fences_and_side_effects(monkeypatch):
    from aerys_v2.action_context import memoized
    import aerys_v2.services.location_guard as guard
    monkeypatch.setattr(factory, "_OWNER_ID", "sample")
    counts = Counter()
    def presence():
        counts["presence"] += 1
        return factory.PresenceSnapshot(["room"], city={"locality": "Example City"})
    def mark(**kwargs):
        counts["mark"] += 1
    monkeypatch.setattr(guard, "mark_location_turn", mark)
    cached = memoized(presence, {})
    identity = {"user_id": "sample", "privacy_context": "private"}
    for _ in range(2):
        assert factory.presence_block(identity, cached)
    assert counts == {"presence": 1, "mark": 2}
    identity["privacy_context"] = "public"
    assert factory.presence_block(identity, cached) == ""
    assert counts == {"presence": 1, "mark": 2}
    room = memoized(lambda *args: "room text", {})
    identity["channel_id"] = "room"
    sink = {}
    for _ in range(2):
        sink.clear()
        factory.room_block(identity, room, sink)
        assert sink["room"] == "room text"
        assert sink["outside"] is True       # #189: room text with no author provenance fails closed
    identity["privacy_context"] = "private"
    assert factory.room_block(identity, room, {}) == ""
