"""The armed honesty retry stays inside the model node."""
import json

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from aerys_v2.action_loop import AgentToolModels, build_agent_action_graph
from aerys_v2.service import ACTION_NO_TOOL_CORRECTION, NO_TOOL_ACTION_MARKER, ask
from test_action_loop_agent import Rec, TOOLS, a_tool_then_done, cfg, agent
from test_honesty_fixes import Recorder, action_router, fake_model
from aerys_v2.factory import build_graph


def make(script):
    log = []
    model = Rec(label="model", script=script, log=log)
    return build_agent_action_graph(AgentToolModels(model, (model, True)), "s", TOOLS), log


@pytest.mark.parametrize("via_ask", [False, True])
@pytest.mark.parametrize("recovered", [False, True])
def test_armed_retry(via_ask, recovered, monkeypatch):
    graph, log = make([AIMessage(content="Done.")] +
                      (a_tool_then_done() if recovered else [AIMessage(content="Cannot do it.")]))
    if via_ask:
        from unittest.mock import Mock
        invoke = Mock(wraps=graph.invoke)
        monkeypatch.setattr(graph, "invoke", invoke)
        rec = Recorder()
        ask(build_graph(fake_model("unused"), soul="s"), "say hi",
            identity={"user_id": "sample", "display_name": "Caller"}, thread_id="honesty",
            router=action_router, action_graph=graph, record_turn=rec)
        assert invoke.call_count == 1
        row = rec.wait()[0]
        assert (NO_TOOL_ACTION_MARKER in json.loads(row["degraded"])) is (not recovered)
    else:
        result = graph.invoke({"messages": [HumanMessage(content="say hi")]},
                              cfg(specialist=True, honesty_gate=True))
        assert sum(m.content == ACTION_NO_TOOL_CORRECTION for m in result["messages"]) == 1
    assert len(log) == (3 if recovered else 2)
    assert log[1]["messages"][-1] == ("human", ACTION_NO_TOOL_CORRECTION)
    assert log[1]["choice"] is None


def test_unarmed_is_one_call():
    graph, log = make([AIMessage(content="Hello.")])
    graph.invoke({"messages": [HumanMessage(content="hi")]}, cfg())
    assert len(log) == 1


def test_seed_receipt_prevents_bounce():
    graph, log = make([AIMessage(content="Done.")])
    seed = [HumanMessage(content="say hi"), a_tool_then_done()[0],
            ToolMessage(content="echo: hi", name="echo", tool_call_id="c1")]
    graph.invoke({"messages": seed}, cfg(honesty_gate=True))
    assert len(log) == 1


def test_armed_nodes_and_runaway_limit():
    def forever():
        return [AIMessage(content="", tool_calls=[
            {"id": f"c{i}", "name": "echo", "args": {"text": "x"}}]) for i in range(20)]
    graph, log = agent(forever)
    assert set(graph.nodes) - {"__start__"} == {"model", "tools"}
    from langgraph.errors import GraphRecursionError
    with pytest.raises(GraphRecursionError):
        graph.invoke({"messages": [HumanMessage(content="loop")]}, cfg(honesty_gate=True))
    assert len(log) == 5
