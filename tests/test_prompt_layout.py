"""Stable prefix, original checkpoint, and per-turn context at the live request."""
from copy import deepcopy

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool

from aerys_v2.factory import build_action_graph, build_graph


class Recorder:
    def __init__(self, replies=None):
        self.prompts = []
        self.replies = iter(replies or [AIMessage(content="reply")])

    def invoke(self, messages):
        self.prompts.append(messages)
        return next(self.replies)


@pytest.mark.parametrize("action", [False, True])
@pytest.mark.parametrize("content", ["original request", [
    {"type": "text", "text": "original request"},
    {"type": "image_url", "image_url": {"url": "https://example.test/image.png"}},
]])
def test_static_history_dynamic_layout_preserves_state(monkeypatch, action, content):
    monkeypatch.setattr("aerys_v2.factory._where_when_line", lambda *_: "\nCLOCK_SENTINEL")
    model = Recorder()
    kwargs = dict(context_fn=lambda *_: "KNOWLEDGE_SENTINEL")
    graph = (build_action_graph(model, "SOUL_SENTINEL", [], overlay="OVERLAY_SENTINEL", **kwargs)
             if action else build_graph(model, "SOUL_SENTINEL", **kwargs))
    current = HumanMessage(content=deepcopy(content), id="current", name="caller",
                           additional_kwargs={"content_privacy": "private", "extra": "kept"})
    before = current.model_dump()
    prior = [HumanMessage(content="earlier", id="earlier"), AIMessage(content="earlier reply")]
    config = {"configurable": {"thread_id": "layout", "identity": {"privacy_context": "private"}}}
    result = graph.invoke({"messages": [*prior, current]}, config)
    system, *shown = model.prompts[0]
    assert isinstance(system.content, str)
    assert "SOUL_SENTINEL" in system.content
    assert ("OVERLAY_SENTINEL" if action else "Your conversation memory is durable") in system.content
    assert "CLOCK_SENTINEL" not in system.content
    assert "KNOWLEDGE_SENTINEL" not in system.content
    assert shown[:-1] == prior
    augmented = shown[-1]
    assert isinstance(augmented, HumanMessage)
    assert isinstance(augmented.content, list)
    context = augmented.content[0]["text"]
    assert context.startswith("[Context for this turn]\n")
    assert context.index("The current caller") < context.index("KNOWLEDGE_SENTINEL") < context.index("CLOCK_SENTINEL")
    expected = [{"type": "text", "text": content}] if isinstance(content, str) else content
    assert augmented.content[1:] == expected
    assert augmented.id == current.id and augmented.name == current.name
    assert augmented.additional_kwargs == current.additional_kwargs
    assert augmented is not current
    assert current.model_dump() == before
    assert result["messages"][-2].model_dump() == before
    if not action:
        assert graph.get_state(config).values["messages"][-2].model_dump() == before


@pytest.mark.parametrize("action", [False, True])
def test_no_human_keeps_dynamic_context_in_system(monkeypatch, action):
    monkeypatch.setattr("aerys_v2.factory._where_when_line", lambda *_: "\nCLOCK_SENTINEL")
    model = Recorder()
    kwargs = dict(context_fn=lambda *_: "KNOWLEDGE_SENTINEL")
    graph = build_action_graph(model, "soul", [], **kwargs) if action else build_graph(model, "soul", **kwargs)
    graph.invoke({"messages": []}, {"configurable": {"thread_id": "empty"}})
    assert "CLOCK_SENTINEL" in model.prompts[0][0].content
    assert "KNOWLEDGE_SENTINEL" in model.prompts[0][0].content


def test_action_loop_augments_current_once_each_pass(monkeypatch):
    monkeypatch.setattr("aerys_v2.factory._where_when_line", lambda *_: "\nCLOCK_SENTINEL")
    @tool
    def read_status() -> str:
        """Read a fake status."""
        return "ready"
    model = Recorder([AIMessage(content="", tool_calls=[{"id": "t", "name": "read_status", "args": {}}]),
                      AIMessage(content="ready")])
    graph = build_action_graph(model, "soul", [read_status])
    current = HumanMessage(content="status?", id="request", additional_kwargs={"content_privacy": "private"})
    result = graph.invoke({"messages": [current]}, {"configurable": {"identity": {"privacy_context": "private"}}})
    assert len(model.prompts) == 2
    assert model.prompts[0][1].content == model.prompts[1][1].content
    assert len(model.prompts[1][1].content) == 2
    assert [m.type for m in model.prompts[1]] == ["system", "human", "ai", "tool"]
    assert result["messages"][0] == current
