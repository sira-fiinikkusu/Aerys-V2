"""Board #176, Stage 1: her action loop on create_agent, proven against the old graph.

The claim this slice rests on: same prompt, same model, same tools, same tool_choice,
same rails. These tests run identical turns through both engines and compare what the
model was handed, at the LangChain level (a recording fake model) and at the wire
(the Anthropic payload, cache marks included). The whole suite also runs against the
new loop with AERYS_TEST_ACTION_LOOP=agent (tests/conftest.py).
"""

from __future__ import annotations

import threading
from types import SimpleNamespace
from typing import Any

import anthropic
import httpx
import pytest

pytest.importorskip("langchain.agents")   # the create_agent loop needs langchain >= 1.4.9
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import ConfigDict

import aerys_v2.factory as factory
from aerys_v2.action_loop import (
    AgentToolModels,
    SurfaceSplitAgentModels,
    build_agent_action_graph,
)
from aerys_v2.factory import LOCAL_FALLBACK_FIRED, ToolModelPair

# The hand-rolled graph, even when the suite runs with AERYS_TEST_ACTION_LOOP=agent.
OLD_BUILD = getattr(factory, "_original_build_action_graph", factory.build_action_graph)


@pytest.fixture(autouse=True)
def fixed_clock(monkeypatch):
    # The where/when line carries the minute; both engines must see the same one.
    monkeypatch.setattr(factory, "_where_when_line", lambda thread, identity: "\nIt is noon.")


@tool
def echo(text: str) -> str:
    """Say it back."""
    return f"echo: {text}"


@tool
def home_control(entity_id: str, operation: str) -> str:
    """Control a house device."""
    return f"Done: {operation} {entity_id}"


TOOLS = [echo, home_control]


class Rec(BaseChatModel):
    """Records what each call was handed: model, bound tool names, tool_choice, messages."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    label: str
    script: Any        # Any, not list: pydantic copies a list field, and these must be shared
    log: Any
    bound: Any = ()
    choice: Any = None

    @property
    def _llm_type(self) -> str:
        return "rec"

    def bind_tools(self, tools, *, tool_choice=None, **kwargs):
        names = [convert_to_openai_tool(t)["function"]["name"] for t in tools]
        return self.model_copy(update={"bound": tuple(names), "choice": tool_choice})

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.log.append({"model": self.label, "tools": self.bound, "choice": self.choice,
                         "messages": [(m.type, m.content) for m in messages]})
        if not self.script:
            raise AssertionError("model called more times than scripted")
        return ChatResult(generations=[ChatGeneration(message=self.script.pop(0))])


def a_tool_then_done():
    return [AIMessage(content="", tool_calls=[{"id": "c1", "name": "echo", "args": {"text": "hi"}}]),
            AIMessage(content="Done.")]


def engines(script_fn):
    """Both engines over the same scripted models: (old_graph, old_log, new_graph, new_log)."""
    out = []
    for build in ("old", "new"):
        log, script = [], script_fn()
        conv, std, fast = (Rec(label=n, script=script, log=log) for n in ("conv", "std", "fast"))
        if build == "old":
            pair = ToolModelPair(conv.bind_tools(TOOLS), std.bind_tools(TOOLS),
                                 std.bind_tools(TOOLS, tool_choice="any"),
                                 fast.bind_tools(TOOLS), fast.bind_tools(TOOLS, tool_choice="any"))
            graph = OLD_BUILD(pair, "SOUL", TOOLS)
        else:
            graph = build_agent_action_graph(AgentToolModels(conv, (std, True), (fast, True)), "SOUL", TOOLS)
        out += [graph, log]
    return out


def cfg(**configurable):
    identity = configurable.pop("identity", {"user_id": "u1", "display_name": "Chris"})
    return {"configurable": {"thread_id": "person:u1", "identity": identity, **configurable},
            "recursion_limit": 10}


SURFACES = {
    "typed specialist": cfg(specialist=True),
    "fast tier": cfg(specialist=True, tier="fast"),
    "voice banter": cfg(identity={"user_id": "u1", "display_name": "Chris", "voice": True}),
    "spoken ack": cfg(specialist=True, spoken_ack="On it.",
                      identity={"user_id": "u1", "display_name": "Chris", "voice": True}),
    "lens": cfg(identity={"user_id": "u1", "display_name": "Chris", "voice": True, "surface": "lens"}),
    "typed chat": cfg(),
}


@pytest.mark.parametrize("surface", SURFACES)
def test_both_engines_hand_the_model_the_same_thing_on_every_pass(surface):
    old, old_log, new, new_log = engines(a_tool_then_done)
    turn = {"messages": [HumanMessage(content="say hi")]}
    old_out = old.invoke(turn, SURFACES[surface])
    new_out = new.invoke(turn, SURFACES[surface])
    assert len(old_log) == 2 and new_log == old_log
    assert [(m.type, m.content) for m in new_out["messages"]] == [(m.type, m.content) for m in old_out["messages"]]


def test_the_specialist_first_pass_is_forced_and_later_passes_are_free():
    _, _, new, log = engines(a_tool_then_done)
    new.invoke({"messages": [HumanMessage(content="say hi")]}, SURFACES["typed specialist"])
    assert [(c["model"], c["choice"]) for c in log] == [("std", "any"), ("std", None)]
    assert log[0]["tools"] == ("echo", "home_control")


def test_the_graph_is_model_and_tools_only_so_the_rail_means_what_it_meant():
    _, _, new, _ = engines(a_tool_then_done)
    assert set(new.nodes) - {"__start__"} == {"model", "tools"}


def test_a_runaway_loop_hits_the_same_wall_after_the_same_number_of_calls():
    def forever():
        return [AIMessage(content="", tool_calls=[{"id": f"c{i}", "name": "echo", "args": {"text": "x"}}])
                for i in range(20)]

    old, old_log, new, new_log = engines(forever)
    for graph in (old, new):
        with pytest.raises(Exception) as err:
            graph.invoke({"messages": [HumanMessage(content="loop")]}, SURFACES["typed specialist"])
        assert type(err.value).__name__ == "GraphRecursionError"
    assert len(new_log) == len(old_log) == 5


def test_the_service_attributes_ride_on_the_new_graph():
    from aerys_v2.factory import _action_engine

    s = SimpleNamespace(action_loop="agent", model_backend="api", oauth_tool_backend="api",
                        oauth_voice_backend="api", local_fallback_url=None, model="claude-sonnet-5-5",
                        tier_fast_model="claude-haiku-5-5", action_model="", action_force_tool=True,
                        anthropic_api_key="k", oauth_turn_timeout_s=60)
    models, build = _action_engine(s, TOOLS)
    assert build is build_agent_action_graph and isinstance(models, AgentToolModels)
    assert models.standard[1] is False      # Sonnet 5.5 refuses a forced tool on the API
    assert models.fast[1] is True           # Haiku 5.5 accepts it
    graph = build(models, "SOUL", TOOLS, presence_fn="p")
    assert graph.presence_fn == "p" and graph.action_loop == "agent"
    s.action_loop = "graph"
    assert _action_engine(s, TOOLS)[1] is factory.build_action_graph   # whatever the suite runs as "graph"


# --- voice on the API, text on the plan: chosen per call --------------------------------------

def test_the_surface_split_follows_reflex_surface_on_every_call_including_a_background_thread():
    from aerys_v2.reflex import REFLEX_SURFACE

    log, script = [], []
    text = AgentToolModels(Rec(label="text", script=script, log=log), (Rec(label="text-std", script=script, log=log), True))
    voice = AgentToolModels(Rec(label="voice", script=script, log=log), (Rec(label="voice-std", script=script, log=log), False))
    graph = build_agent_action_graph(SurfaceSplitAgentModels(text, voice), "SOUL", TOOLS)

    def run(surface, config, out):
        token = REFLEX_SURFACE.set(surface)
        try:
            script.extend([AIMessage(content="ok")])
            graph.invoke({"messages": [HumanMessage(content="hi")]}, config)
            out.append(log[-1]["model"])
        finally:
            REFLEX_SURFACE.reset(token)

    seen: list = []
    run("voice", SURFACES["voice banter"], seen)
    run("text", SURFACES["typed specialist"], seen)
    worker = threading.Thread(target=run, args=("voice", SURFACES["spoken ack"], seen))
    worker.start()
    worker.join()
    assert seen == ["voice", "text-std", "voice-std"]
    assert log[1]["choice"] == "any" and log[2]["choice"] is None   # forced on the plan, not on voice


# --- the lifeboat ----------------------------------------------------------------------------

class Boom(Rec):
    error: Any = None

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.log.append({"model": self.label, "tools": self.bound, "choice": self.choice,
                         "messages": [(m.type, m.content) for m in messages]})
        raise self.error


def lifeboat_graph(error):
    log = []
    boat = Rec(label="boat", script=[AIMessage(content="Lamp is on.")], log=log)
    primary = Boom(label="std", script=[], log=log, error=error)
    models = AgentToolModels(primary, (primary, True), lifeboat=boat, lifeboat_tools=[home_control])
    return build_agent_action_graph(models, "SOUL", TOOLS), log


def test_a_dead_connection_lands_on_the_lifeboat_with_the_house_tools_and_the_hardened_prompt():
    graph, log = lifeboat_graph(anthropic.APIConnectionError(request=httpx.Request("POST", "http://x")))
    token = LOCAL_FALLBACK_FIRED.set(False)
    try:
        # The receipt carries the marker out of the node's copied context, as in service.py.
        with factory.track_local_tool_fallback():
            out = graph.invoke({"messages": [HumanMessage(content="lamp on")]}, SURFACES["typed specialist"])
        assert LOCAL_FALLBACK_FIRED.get() is True
    finally:
        LOCAL_FALLBACK_FIRED.reset(token)
    assert out["messages"][-1].content == "Lamp is on."
    assert [c["model"] for c in log] == ["std", "boat"]
    assert log[1]["tools"] == ("home_control",) and log[1]["choice"] is None
    assert factory.LOCAL_ACTION_HARDENING.strip()[:40] in str(log[1]["messages"])


def test_a_client_error_is_not_a_lifeboat_case():
    resp = httpx.Response(400, request=httpx.Request("POST", "http://x"))
    graph, log = lifeboat_graph(anthropic.BadRequestError("bad", response=resp, body=None))
    with pytest.raises(anthropic.BadRequestError):
        graph.invoke({"messages": [HumanMessage(content="lamp on")]}, SURFACES["typed specialist"])
    assert [c["model"] for c in log] == ["std"]


# --- at the wire: the Anthropic payload, cache marks included --------------------------------

def _wire_turns(monkeypatch, build):
    from anthropic.types import Message, TextBlock, ToolUseBlock, Usage
    from langchain_anthropic import ChatAnthropic

    payloads, replies = [], [
        [ToolUseBlock(type="tool_use", id="toolu_1", name="echo", input={"text": "hi"})],
        [TextBlock(type="text", text="Done.")],
    ]

    def create(self, payload):          # the SDK call itself; everything above it is real
        payloads.append(payload)
        content = replies[len(payloads) - 1]
        message = Message(id="msg", type="message", role="assistant", model=payload["model"], content=content,
                          stop_reason="tool_use" if content[0].type == "tool_use" else "end_turn",
                          stop_sequence=None, usage=Usage(input_tokens=1, output_tokens=1))
        return SimpleNamespace(parse=lambda: message, headers={})

    monkeypatch.setattr(ChatAnthropic, "_create", create)
    from aerys_v2.anthropic_model import build_metered_model

    settings = SimpleNamespace(local_fallback_url=None)
    base = build_metered_model(settings, model="claude-haiku-5-5", api_key="test", max_tokens=1024,
                               timeout=60, max_retries=2, cache_prefix=True)
    graph = build(base)
    graph.invoke({"messages": [HumanMessage(content="say hi")]}, SURFACES["typed specialist"])
    return payloads


def test_the_wire_payloads_are_identical_old_and_new(monkeypatch):
    old = _wire_turns(monkeypatch, lambda base: OLD_BUILD(
        ToolModelPair(base.bind_tools(TOOLS), base.bind_tools(TOOLS),
                      base.bind_tools(TOOLS, tool_choice="any")), "SOUL", TOOLS))
    new = _wire_turns(monkeypatch, lambda base: build_agent_action_graph(
        AgentToolModels(base, (base, True)), "SOUL", TOOLS))
    assert len(old) == len(new) == 2
    assert new == old
    assert old[0]["tool_choice"]["type"] == "any" and "tool_choice" not in old[1]
    assert "cache_control" in str(old[0]["system"])           # the cached prefix survived the re-bind


# --- the plan model: re-binding per call keeps the warm process ------------------------------

def test_rebinding_the_plan_model_reuses_the_same_warm_client_key():
    from aerys_v2.oauth_model import ClaudeOAuthChatModel, _schema_key

    base = ClaudeOAuthChatModel(model="claude-sonnet-5-5")
    a, b = base.bind_tools(TOOLS), base.bind_tools(TOOLS, tool_choice=None)
    assert _schema_key(a.model, a.bound_tools) == _schema_key(b.model, b.bound_tools)


# --- Codex review 11/08: create_agent's repair of a malformed tool call ---------------------

def test_a_malformed_call_then_a_claim_is_still_zero_tools_on_both_engines():
    """Old graph: a malformed call ends the turn; the gate retries with its correction and a
    bare "Done." is caught (no tool ran). create_agent answers the malformed call with a
    synthetic error receipt first: it must not be shown to the model or counted as a tool."""
    from aerys_v2.service import ACTION_NO_TOOL_CORRECTION
    from aerys_v2.turns import extract_tool_calls

    def script():
        return [AIMessage(content="", invalid_tool_calls=[{"name": "home_control", "args": "{bad",
                                                           "id": "bad1", "error": "json"}]),
                AIMessage(content="Done, the lamp is on.")]

    old, old_log, new, new_log = engines(script)
    results = []
    for graph in (old, new):
        first = graph.invoke({"messages": [HumanMessage(content="lamp on")]}, SURFACES["typed specialist"])
        retry = [*first["messages"], HumanMessage(content=ACTION_NO_TOOL_CORRECTION)]
        results.append(graph.invoke({"messages": retry}, SURFACES["typed specialist"]))
    assert new_log == old_log and len(old_log) == 2
    for out in results:
        assert extract_tool_calls(out["messages"]) == []
        assert out["messages"][-1].content == "Done, the lamp is on."


def test_a_repair_receipt_is_not_a_tool_that_ran_but_a_real_failure_still_is():
    from langchain_core.messages import ToolMessage

    from aerys_v2.turns import extract_tool_calls

    msgs = [AIMessage(content="", tool_calls=[{"id": "ok1", "name": "echo", "args": {}}],
                      invalid_tool_calls=[{"name": "home_control", "args": "{", "id": "bad1", "error": "x"}]),
            ToolMessage(content="echo: hi", tool_call_id="ok1", name="echo"),
            ToolMessage(content="Tool call home_control with id bad1 could not be executed - arguments "
                                "were malformed or truncated.", tool_call_id="bad1", name="home_control",
                        status="error")]
    assert [c["name"] for c in extract_tool_calls(msgs)] == ["echo"]


def test_two_tools_in_one_pass_come_back_in_the_same_order_on_both_engines():
    """Real traces have parallel calls (calendar_events + manage_list in one pass).
    create_agent sends each call as its own task; the next model pass must still see
    the results in the order the old ToolNode gave them."""
    import time as _time

    @tool
    def slow(text: str) -> str:
        """Slow tool."""
        _time.sleep(0.05)
        return f"slow: {text}"

    tools = [echo, slow]

    def script():
        return [AIMessage(content="", tool_calls=[{"id": "a", "name": "slow", "args": {"text": "1"}},
                                                  {"id": "b", "name": "echo", "args": {"text": "2"}}]),
                AIMessage(content="Both done.")]

    logs = []
    for engine in ("old", "new"):
        log, s = [], script()
        conv, std = Rec(label="conv", script=s, log=log), Rec(label="std", script=s, log=log)
        if engine == "old":
            graph = OLD_BUILD(ToolModelPair(conv.bind_tools(tools), std.bind_tools(tools),
                                            std.bind_tools(tools, tool_choice="any")), "SOUL", tools)
        else:
            graph = build_agent_action_graph(AgentToolModels(conv, (std, True)), "SOUL", tools)
        out = graph.invoke({"messages": [HumanMessage(content="do both")]}, SURFACES["typed specialist"])
        logs.append((log, [(m.type, m.content) for m in out["messages"]]))
    (old_log, old_msgs), (new_log, new_msgs) = logs
    assert new_log == old_log and new_msgs == old_msgs
    assert [c for t, c in old_msgs if t == "tool"] == ["slow: 1", "echo: 2"]
