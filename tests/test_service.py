"""Offline tests for the graph + ask() seam — no API key, no network.

GenericFakeChatModel stands in for Claude (the same trick as pinning an n8n node's
output to test downstream wiring). What these prove: thread memory accumulates via the
checkpointer, threads are isolated, identity stays in config (never in state), and the
rails reject garbage input.
"""

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from aerys_v2.factory import FALLBACK_SOUL, build_graph, load_soul
from aerys_v2.service import ask
from aerys_v2.state import UNKNOWN_CALLER

CHRIS = {"user_id": "person-1", "display_name": "Chris"}


def fake_model(*replies: str) -> GenericFakeChatModel:
    return GenericFakeChatModel(messages=iter([AIMessage(content=r) for r in replies]))


def test_ask_returns_reply_text():
    graph = build_graph(fake_model("hello there"), soul="test soul")
    assert ask(graph, "hi", identity=CHRIS, thread_id="t1") == "hello there"


def test_thread_accumulates_history():
    graph = build_graph(fake_model("one", "two"), soul="test soul")
    ask(graph, "first", identity=CHRIS, thread_id="t1")
    ask(graph, "second", identity=CHRIS, thread_id="t1")
    state = graph.get_state({"configurable": {"thread_id": "t1"}})
    # 2 human + 2 ai — the checkpointer replayed turn 1 into turn 2
    assert len(state.values["messages"]) == 4


def test_threads_are_isolated():
    graph = build_graph(fake_model("a", "b"), soul="test soul")
    ask(graph, "hi", identity=CHRIS, thread_id="t1")
    ask(graph, "hi", identity=CHRIS, thread_id="t2")
    for tid in ("t1", "t2"):
        state = graph.get_state({"configurable": {"thread_id": tid}})
        assert len(state.values["messages"]) == 2  # each thread only its own turn


def test_identity_never_lands_in_state():
    graph = build_graph(fake_model("ok"), soul="test soul")
    ask(graph, "hi", identity=CHRIS, thread_id="t1")
    state = graph.get_state({"configurable": {"thread_id": "t1"}})
    assert set(state.values.keys()) == {"messages"}  # no identity key snuck in


def test_unknown_caller_default_flows():
    graph = build_graph(fake_model("ok"), soul="test soul")
    assert ask(graph, "hi", identity=UNKNOWN_CALLER, thread_id="t1") == "ok"


def test_empty_text_rejected():
    graph = build_graph(fake_model("never"), soul="test soul")
    with pytest.raises(ValueError):
        ask(graph, "   ", identity=CHRIS, thread_id="t1")


def test_load_soul_missing_file_falls_back(tmp_path):
    assert load_soul(tmp_path / "nope.md") == FALLBACK_SOUL


def test_load_soul_reads_file(tmp_path):
    p = tmp_path / "soul.md"
    p.write_text("I am the soul.")
    assert load_soul(p) == "I am the soul."
