"""Offline tests for the TOOLS-block orchestration in ask() + the action subgraph.

Fakes all the way down: GenericFakeChatModel scripts the tool-calling loop, a
stub action graph and plain-callable routers drive the ask() branching, and the
home_control tool runs against an httpx MockTransport. What these prove: the
subgraph's act ⇄ tools loop terminates into a final AIMessage, non-voice threads
route sequentially, voice threads parallel-start (chat verdict -> chat reply,
action verdict -> immediate ack + background result landing in the SAME thread),
and ask() without router/action_graph is byte-for-byte the old chat path.
"""

import time

import httpx
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage

from aerys_v2.factory import build_action_graph, build_graph
from aerys_v2.router import RouteDecision
from aerys_v2.service import ask
from aerys_v2.tools.home_control import build_home_control_tool, canary_set

CHRIS = {"user_id": "person-1", "display_name": "Chris"}


def fake_model(*replies) -> GenericFakeChatModel:
    msgs = [r if isinstance(r, AIMessage) else AIMessage(content=r) for r in replies]
    return GenericFakeChatModel(messages=iter(msgs))


def ha_client() -> httpx.Client:
    return httpx.Client(
        transport=httpx.MockTransport(lambda req: httpx.Response(200, json=[]))
    )


def home_tool():
    return build_home_control_tool(
        base_url="http://ha.test:8123",
        token="t",
        canary_entities=canary_set("light.desk"),
        client=ha_client(),
    )


class StubActionGraph:
    """Stands in for the compiled subgraph in ask() branching tests."""

    def __init__(self, final: str = "the light is on now"):
        self.final = final
        self.calls: list[str] = []

    def invoke(self, inp: dict, config: dict) -> dict:
        self.calls.append(inp["messages"][-1].content)
        return {"messages": [AIMessage(content=self.final)]}


def chat_router(_text: str) -> RouteDecision:
    return RouteDecision(route="chat", ack="")


def action_router(_text: str) -> RouteDecision:
    return RouteDecision(route="action", ack="[warmly] Getting that light for you")


def wait_for_messages(graph, thread_id: str, count: int, timeout_s: float = 3.0) -> list:
    """Poll the checkpointer until the background action lands (or fail loudly)."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        state = graph.get_state({"configurable": {"thread_id": thread_id}})
        msgs = state.values.get("messages", [])
        if len(msgs) >= count:
            return msgs
        time.sleep(0.02)
    raise AssertionError(f"thread never reached {count} messages: {msgs}")


# ---- the action subgraph itself -------------------------------------------------

def test_action_graph_runs_tool_loop_to_final_message():
    tool = home_tool()
    # turn 1: the model calls the tool; turn 2: it answers in plain text
    model = fake_model(
        AIMessage(
            content="",
            tool_calls=[{
                "name": "home_control",
                "args": {"operation": "turn_on", "entity_id": "light.desk"},
                "id": "call-1",
                "type": "tool_call",
            }],
        ),
        "Desk light's on.",
    )
    graph = build_action_graph(model, soul="s", tools=[tool])
    result = graph.invoke(
        {"messages": [HumanMessage(content="turn on the desk light")]},
        {"configurable": {"thread_id": "t", "identity": CHRIS}, "recursion_limit": 10},
    )
    assert result["messages"][-1].content == "Desk light's on."
    # the loop actually executed the tool — a ToolMessage is in the transcript
    assert any(getattr(m, "type", "") == "tool" for m in result["messages"])
    assert "Done: turn_on sent to light.desk" in next(
        m.content for m in result["messages"] if getattr(m, "type", "") == "tool"
    )


def test_action_graph_no_tool_call_ends_immediately():
    graph = build_action_graph(fake_model("nothing to do"), soul="s", tools=[home_tool()])
    result = graph.invoke(
        {"messages": [HumanMessage(content="hi")]},
        {"configurable": {"identity": CHRIS}, "recursion_limit": 10},
    )
    assert len(result["messages"]) == 2  # human + one AI reply, no loop


# ---- ask() branching: non-voice (sequential) ------------------------------------

def test_backward_compatible_without_router():
    # no router/action_graph kwargs = the pre-TOOLS path, untouched
    graph = build_graph(fake_model("plain chat"), soul="s")
    assert ask(graph, "hi", identity=CHRIS, thread_id="t1") == "plain chat"


def test_nonvoice_chat_route_uses_chat_graph():
    graph = build_graph(fake_model("chat reply"), soul="s")
    stub = StubActionGraph()
    out = ask(graph, "how are you?", identity=CHRIS, thread_id="t1",
              router=chat_router, action_graph=stub)
    assert out == "chat reply"
    assert stub.calls == []  # action path never touched


def test_nonvoice_action_route_returns_action_result_and_lands_in_thread():
    graph = build_graph(fake_model("never spoken"), soul="s")
    stub = StubActionGraph("light is on")
    out = ask(graph, "turn on the light", identity=CHRIS, thread_id="t1",
              router=action_router, action_graph=stub)
    assert out == "light is on"
    assert stub.calls == ["turn on the light"]
    # history is coherent: the human turn AND the outcome are in the thread
    msgs = graph.get_state({"configurable": {"thread_id": "t1"}}).values["messages"]
    assert [m.content for m in msgs] == ["turn on the light", "light is on"]


# ---- ask() branching: voice (parallel-start) ------------------------------------

def test_voice_chat_route_returns_chat_result():
    graph = build_graph(fake_model("spoken chat reply"), soul="s")
    stub = StubActionGraph()
    out = ask(graph, "tell me something nice", identity=CHRIS, thread_id="voice:beta",
              router=chat_router, action_graph=stub)
    assert out == "spoken chat reply"
    assert stub.calls == []


def test_voice_action_route_returns_ack_immediately_then_result_lands():
    graph = build_graph(fake_model("speculative chat"), soul="s")
    stub = StubActionGraph("done — desk light on")
    out = ask(graph, "turn on the desk light", identity=CHRIS, thread_id="voice:beta",
              router=action_router, action_graph=stub)
    # the caller hears the router's GENERATED ack, not the action result
    assert out == "[warmly] Getting that light for you"
    # ...and the real outcome arrives in the SAME thread shortly after
    msgs = wait_for_messages(graph, "voice:beta", 2)
    assert msgs[-1].content == "done — desk light on"
    assert msgs[0].content == "turn on the desk light"  # human turn present exactly once
    assert sum(1 for m in msgs if getattr(m, "type", "") == "human") == 1


def test_voice_action_failure_lands_honestly_in_thread():
    class ExplodingActionGraph:
        def invoke(self, inp, config):
            raise RuntimeError("HA melted")

    graph = build_graph(fake_model("speculative chat"), soul="s")
    out = ask(graph, "toggle the desk light", identity=CHRIS, thread_id="voice:x",
              router=action_router, action_graph=ExplodingActionGraph())
    assert out == "[warmly] Getting that light for you"  # ack still speaks
    msgs = wait_for_messages(graph, "voice:x", 2)
    assert "didn't complete" in msgs[-1].content  # honest failure, never silence
