"""The 2026-09-19 window bounds model input, never the durable thread."""
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from aerys_v2.config import Settings
from aerys_v2.factory import build_action_graph, build_graph
from aerys_v2.service import _action_history_seed


class Recorder:
    def __init__(self):
        self.prompts = []

    def invoke(self, messages):
        self.prompts.append(messages)
        return AIMessage(content="reply")


def history():
    return [HumanMessage(content="old", id="old"), AIMessage(content="old reply"),
            HumanMessage(content="recent", id="recent"), AIMessage(content="recent reply"),
            HumanMessage(content="current", id="current")]


# Over the limit the cut index is aligned to a limit/4 step, then any leading
# non-human message is dropped: 5 messages at limit 4 -> cut at 2 -> [2,3,4]; at
# limit 3 -> step 1, low-water 2 -> cut at 3 -> [3,4] -> leading reply goes -> [4].
@pytest.mark.parametrize("limit,expected", [(0, [0, 1, 2, 3, 4]), (1, [4]),
                                          (2, [4]), (3, [4]), (4, [2, 3, 4]), (5, [0, 1, 2, 3, 4]),
                                          (200, [0, 1, 2, 3, 4])])
def test_window_tail_starts_with_human_and_keeps_current(limit, expected):
    from aerys_v2.history import window_messages
    messages = history()
    before = [m.model_dump() for m in messages]
    kept = window_messages(messages, limit)
    assert kept == [messages[i] for i in expected]
    assert kept[-1] is messages[-1]
    assert [m.model_dump() for m in messages] == before


def test_window_boundary_moves_in_steps_not_every_turn():
    from aerys_v2.history import window_messages
    limit = 40  # cut back to 30 once over: the edge moves every 10 messages
    thread = []
    firsts = []
    for i in range(40):
        thread += [HumanMessage(content=f"h{i}"), AIMessage(content=f"a{i}")]
        firsts.append(window_messages(thread, limit)[0].content)
    # The leading edge changes far less often than the 40 turns it was asked on.
    assert len(set(firsts)) < 40 // 2
    # And it never exceeds the limit or starts with a reply.
    for i in range(40):
        kept = window_messages(thread[: 2 * (i + 1)], limit)
        assert len(kept) <= limit and kept[0].type == "human"


def test_window_drops_orphan_tool_and_assistant():
    from aerys_v2.history import window_messages
    messages = [HumanMessage(content="old"), AIMessage(content="call"),
                ToolMessage(content="result", tool_call_id="t"), HumanMessage(content="now")]
    assert window_messages(messages, 2) == messages[-1:]
    assert window_messages([], 2) == []
    assert window_messages([AIMessage(content="orphan")], 1) == []


def test_setting_default_and_environment(monkeypatch):
    monkeypatch.delenv("HISTORY_WINDOW_MESSAGES", raising=False)
    assert Settings(_env_file=None, anthropic_api_key="test").history_window_messages == 200
    monkeypatch.setenv("HISTORY_WINDOW_MESSAGES", "0")
    assert Settings(_env_file=None, anthropic_api_key="test").history_window_messages == 0


# 5 messages: limit 4 cuts at index 2 (step 1, low-water 3) -> 3 shown; limit 2
# cuts at 3 -> the leading reply goes -> only the current turn.
@pytest.mark.parametrize("limit,count", [(0, 5), (2, 1), (4, 3)])
def test_chat_windows_only_shown_history(limit, count):
    model = Recorder()
    graph = build_graph(model, "soul", history_window_messages=limit)
    config = {"configurable": {"thread_id": "window", "identity": {"privacy_context": "private"}}}
    messages = history()
    graph.invoke({"messages": messages}, config)
    shown = model.prompts[0][1:]
    assert len(shown) == count
    assert shown[0].type == "human"
    assert shown[-1].id == "current"
    assert graph.get_state(config).values["messages"][:-1] == messages


@pytest.mark.parametrize("specialist", [False, True])
@pytest.mark.parametrize("escalated", [False, True])
# limit 3 on the 4-message prior keeps ["recent", "recent reply"]; on the
# escalated 5-message prior the stepped cut lands on the reply, which is dropped,
# so only the current turn survives.
@pytest.mark.parametrize("limit", [0, 3])
def test_action_seed_windows_before_specialist_filter(specialist, escalated, limit):
    prior = history() if escalated else history()[:-1]
    if escalated:
        prior = [*prior, AIMessage(content="handoff")]
    graph = SimpleNamespace(history_window_messages=limit,
                            get_state=lambda _: SimpleNamespace(values={"messages": prior}))
    config = {"identity": {"privacy_context": "private"}}
    seed = _action_history_seed(graph, config, "current", specialist=specialist, escalated=escalated)
    model = Recorder()
    action = build_action_graph(model, "soul", [])
    action.invoke({"messages": seed}, {"configurable": config})
    shown = str([m.content for m in model.prompts[0][1:]])
    assert "current" in shown
    assert ("old" in shown) == (limit == 0)
    assert ("recent" in shown) == (limit == 0 or not escalated)
    assert "handoff" not in shown
    assert seed[0].type == "human"
    if escalated:
        assert seed[-1].id == "current"


def test_action_seed_inherits_chat_graph_window():
    model = Recorder()
    graph = build_graph(model, "soul", history_window_messages=3)
    config = {"thread_id": "shared-limit", "identity": {"privacy_context": "private"}}
    graph.update_state({"configurable": config}, {"messages": history()[:-1]}, as_node="chat")
    seed = _action_history_seed(graph, config, "current")
    action = build_action_graph(model, "soul", [])
    action.invoke({"messages": seed}, {"configurable": config})
    assert [m.content for m in seed] == ["recent", "recent reply", "current"]
    assert len(model.prompts[0]) == 4
    assert len(graph.get_state({"configurable": config}).values["messages"]) == 4


def test_chat_windows_after_privacy_gate():
    model = Recorder()
    graph = build_graph(model, "soul", history_window_messages=3)
    messages = [HumanMessage(content="public", additional_kwargs={"content_privacy": "public"}),
                AIMessage(content="public reply"),
                HumanMessage(content="private", additional_kwargs={"content_privacy": "private"}),
                AIMessage(content="private reply"), HumanMessage(content="current")]
    config = {"configurable": {"thread_id": "public", "identity": {"privacy_context": "public"}}}
    graph.invoke({"messages": messages}, config)
    assert [m.content for m in model.prompts[0][1:-1]] == ["public", "public reply"]
    assert len(graph.get_state(config).values["messages"]) == 6
