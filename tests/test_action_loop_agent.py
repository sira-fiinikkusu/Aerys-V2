"""Action-loop behavior: prompts, model choice, tools, rails, and wire payloads."""

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
from aerys_v2.factory import LOCAL_FALLBACK_FIRED

@pytest.fixture(autouse=True)
def fixed_clock(monkeypatch):
    # The where/when line carries the minute; tests use a stable one.
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


def agent(script_fn):
    log, script = [], script_fn()
    conv, std, fast = (Rec(label=n, script=script, log=log) for n in ("conv", "std", "fast"))
    return build_agent_action_graph(AgentToolModels(conv, (std, True), (fast, True)), "SOUL", TOOLS), log


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
def test_agent_preserves_surface_model_choice_and_tool_history(surface):
    graph, log = agent(a_tool_then_done)
    out = graph.invoke({"messages": [HumanMessage(content="say hi")]}, SURFACES[surface])
    specialist = SURFACES[surface]["configurable"].get("specialist", False)
    label = "fast" if surface == "fast tier" else "std" if specialist else "conv"
    assert [(p["model"], p["choice"]) for p in log] == [(label, "any" if specialist else None), (label, None)]
    assert all(p["tools"] == ("echo", "home_control") for p in log)
    assert [(m.type, m.content) for m in out["messages"]] == [
        ("human", "say hi"), ("ai", ""), ("tool", "echo: hi"), ("ai", "Done.")]
    assert log[1]["messages"][-3:] == [("human", "say hi"), ("ai", ""), ("tool", "echo: hi")]


def test_the_specialist_first_pass_is_forced_and_later_passes_are_free():
    new, log = agent(a_tool_then_done)
    new.invoke({"messages": [HumanMessage(content="say hi")]}, SURFACES["typed specialist"])
    assert [(c["model"], c["choice"]) for c in log] == [("std", "any"), ("std", None)]
    assert log[0]["tools"] == ("echo", "home_control")


def test_the_graph_is_model_and_tools_only_so_the_rail_means_what_it_meant():
    new, _ = agent(a_tool_then_done)
    assert set(new.nodes) - {"__start__"} == {"model", "tools"}


def test_a_runaway_loop_hits_the_rail_after_five_model_calls():
    def forever():
        return [AIMessage(content="", tool_calls=[{"id": f"c{i}", "name": "echo", "args": {"text": "x"}}])
                for i in range(20)]

    graph, log = agent(forever)
    from langgraph.errors import GraphRecursionError
    with pytest.raises(GraphRecursionError):
        graph.invoke({"messages": [HumanMessage(content="loop")]}, SURFACES["typed specialist"])
    assert len(log) == 5


def test_the_service_attributes_ride_on_the_new_graph():
    from aerys_v2.factory import _action_engine

    s = SimpleNamespace(action_loop="agent", model_backend="api", oauth_tool_backend="api",
                        oauth_voice_backend="api", local_fallback_url=None, model="claude-sonnet-5-5",
                        tier_fast_model="claude-haiku-5-5", action_model="", action_force_tool=True,
                        anthropic_api_key="k", oauth_turn_timeout_s=60)
    models, build = _action_engine(s, TOOLS)
    assert build is factory.build_action_graph and isinstance(models, AgentToolModels)
    assert models.standard[1] is False      # Sonnet 5.5 refuses a forced tool on the API
    assert models.fast[1] is True           # Haiku 5.5 accepts it
    graph = build(models, "SOUL", TOOLS, presence_fn="p")
    assert graph.presence_fn == "p" and graph.action_loop == "agent"


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


def test_agent_wire_payloads_preserve_tool_choice_and_cache(monkeypatch):
    payloads = _wire_turns(monkeypatch, lambda base: build_agent_action_graph(
        AgentToolModels(base, (base, True)), "SOUL", TOOLS))
    assert len(payloads) == 2
    assert payloads[0]["tool_choice"]["type"] == "any" and "tool_choice" not in payloads[1]
    assert "cache_control" in str(payloads[0]["system"])
    assert [t["name"] for t in payloads[0]["tools"]] == ["echo", "home_control"]
    assert payloads[1]["messages"][-1]["content"][0]["type"] == "tool_result"


# --- the plan model: re-binding per call keeps the warm process ------------------------------

def test_rebinding_the_plan_model_reuses_the_same_warm_client_key():
    from aerys_v2.oauth_model import ClaudeOAuthChatModel, _schema_key

    base = ClaudeOAuthChatModel(model="claude-sonnet-5-5")
    a, b = base.bind_tools(TOOLS), base.bind_tools(TOOLS, tool_choice=None)
    assert _schema_key(a.model, a.bound_tools) == _schema_key(b.model, b.bound_tools)


# --- Codex review 11/08: create_agent's repair of a malformed tool call ---------------------

def test_a_malformed_call_then_a_claim_is_still_zero_tools():
    """Synthetic repair receipts are hidden and never counted as executed tools."""
    from aerys_v2.service import ACTION_NO_TOOL_CORRECTION
    from aerys_v2.turns import extract_tool_calls

    def script():
        return [AIMessage(content="", invalid_tool_calls=[{"name": "home_control", "args": "{bad",
                                                           "id": "bad1", "error": "json"}]),
                AIMessage(content="Done, the lamp is on.")]

    graph, log = agent(script)
    first = graph.invoke({"messages": [HumanMessage(content="lamp on")]}, SURFACES["typed specialist"])
    retry = [*first["messages"], HumanMessage(content=ACTION_NO_TOOL_CORRECTION)]
    out = graph.invoke({"messages": retry}, SURFACES["typed specialist"])
    assert len(log) == 2
    assert all(kind != "tool" for kind, _ in log[1]["messages"])
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


def test_two_tools_in_one_pass_return_in_call_order():
    """Real traces have parallel calls (calendar_events + manage_list in one pass).
    create_agent sends each call as its own task; the next model pass must still see
    the results in call order."""
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

    log, replies = [], script()
    conv, std = (Rec(label=n, script=replies, log=log) for n in ("conv", "std"))
    graph = build_agent_action_graph(AgentToolModels(conv, (std, True)), "SOUL", tools)
    out = graph.invoke({"messages": [HumanMessage(content="do both")]}, SURFACES["typed specialist"])
    assert [m.content for m in out["messages"] if m.type == "tool"] == ["slow: 1", "echo: 2"]
    assert [content for kind, content in log[1]["messages"] if kind == "tool"] == ["slow: 1", "echo: 2"]


def test_action_loop_defaults_to_agent_and_accepts_live_environment(monkeypatch):
    from aerys_v2.config import Settings
    monkeypatch.delenv("ACTION_LOOP", raising=False)
    assert Settings(_env_file=None, anthropic_api_key="test").action_loop == "agent"
    monkeypatch.setenv("ACTION_LOOP", "agent")
    assert Settings(_env_file=None, anthropic_api_key="test").action_loop == "agent"


@pytest.mark.parametrize("value", ["graph", "other", ""])
def test_action_loop_rejects_retired_or_unknown_engines(monkeypatch, value):
    from aerys_v2.config import Settings
    from pydantic import ValidationError
    monkeypatch.setenv("ACTION_LOOP", value)
    with pytest.raises(ValidationError, match="action_loop"):
        Settings(_env_file=None, anthropic_api_key="test")
