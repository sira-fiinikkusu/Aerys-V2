"""Offline tests for the graph + ask() seam — no API key, no network.

GenericFakeChatModel stands in for Claude (the same trick as pinning an n8n node's
output to test downstream wiring). What these prove: thread memory accumulates via the
checkpointer, threads are isolated, identity stays in config (never in state), and the
rails reject garbage input.
"""

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from aerys_v2.factory import FALLBACK_SOUL, _channel_phrase, build_graph, load_soul
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


class RecordingModel(GenericFakeChatModel):
    """Fake that records the system prompt each turn (for prompt-shape tests)."""

    seen: list = []

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        type(self).seen.append(str(messages[0].content))
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)


def test_voice_threads_get_emotion_tag_instruction():
    RecordingModel.seen = []
    m = RecordingModel(messages=iter([AIMessage(content="a"), AIMessage(content="b")]))
    graph = build_graph(m, soul="s")
    ask(graph, "hi", identity=CHRIS, thread_id="voice:beta")
    ask(graph, "hi", identity=CHRIS, thread_id="t1")
    assert "[warmly]" in RecordingModel.seen[0]        # voice thread gets tags
    assert "[warmly]" not in RecordingModel.seen[1]    # text thread stays clean
    assert all("memory is durable" in s for s in RecordingModel.seen)  # capability line everywhere


# --- #2: "where + when" injection ------------------------------------------

def test_channel_phrase_surfaces():
    assert _channel_phrase("voice:beta") == "a live voice conversation"
    assert _channel_phrase("discord:dm:123") == "a private Discord DM"
    assert "shared Discord server" in _channel_phrase("discord:guild:555")
    # a supplied room name is used, and the channel id becomes a <#id> mention
    assert "#general" in _channel_phrase("discord:guild:555", "general")
    assert "<#555>" in _channel_phrase("discord:guild:555", "general")  # clickable link
    assert "reading" not in _channel_phrase("discord:guild:555", "general")  # no announce
    assert _channel_phrase("telegram:dm:9") == "a private Telegram chat"
    assert "'fam'" in _channel_phrase("telegram:group:9", "fam")
    assert _channel_phrase("weird:thing") == "a direct message"  # unknown degrades


def test_system_prompt_carries_clock_and_where():
    RecordingModel.seen = []
    m = RecordingModel(messages=iter([AIMessage(content="a"), AIMessage(content="b")]))
    graph = build_graph(m, soul="s")
    # public guild turn, with the resolver-supplied room label on identity
    ask(
        graph,
        "hi",
        identity={"user_id": "person-1", "display_name": "Chris", "channel_name": "general"},
        thread_id="discord:guild:555",
    )
    # private DM turn
    ask(graph, "hi", identity=CHRIS, thread_id="discord:dm:person-1")
    assert "Eastern" in RecordingModel.seen[0]                     # she has a clock now
    assert "#general" in RecordingModel.seen[0]                    # names the room
    assert "<#555>" in RecordingModel.seen[0]                      # clickable channel link
    assert "never cite URLs" in RecordingModel.seen[0]             # no plumbing narration
    assert "a private Discord DM" in RecordingModel.seen[1]        # DM says private
    assert "shared Discord server" not in RecordingModel.seen[1]   # DM never says shared
