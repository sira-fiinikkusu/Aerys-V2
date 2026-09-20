"""The subscription path WITH tools (2026-09-20, Chris: "shed my API costs"):
ClaudeOAuthChatModel binds LangChain tools, the CLI's PreToolUse hook defers every call,
tool_calls come back on the AIMessage, our graph runs them, results ride the next prompt.
Offline: the warm client is faked; nothing spawns a subprocess."""
import json

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

import aerys_v2.oauth_model as om
from aerys_v2.config import Settings
from aerys_v2.factory import SurfaceSplitToolModel, ToolModelPair, build_api_tool_model
from aerys_v2.oauth_model import OAuthBackendError, ClaudeOAuthChatModel, _FORCE_TOOL_LINE, _RESULTS_FINAL_LINE, _flatten
from aerys_v2.reflex import REFLEX_SURFACE


@tool
def light_state(entity_id: str) -> str:
    """Return the current on/off state of a light."""
    return "off"


class FakeWarm:
    """Stands in for _WarmClient: records prompts, returns canned (text, tool_calls, meta)."""
    made: list = []

    def __init__(self, model, schemas=None):
        self.model, self.schemas, self.prompts = model, list(schemas or []), []
        FakeWarm.made.append(self)
        self.reply = ("", [{"name": "light_state", "args": {"entity_id": "light.office_light_1"},
                            "id": "toolu_1", "type": "tool_call"}], {"stop_reason": "tool_deferred",
                                                                    "usage": {"input_tokens": 100, "output_tokens": 5},
                                                                    "rate_limit": {"type": "five_hour"}})

    def ask(self, prompt):
        self.prompts.append(prompt)
        return self.reply


def fresh(monkeypatch):
    FakeWarm.made = []
    monkeypatch.setattr(om, "_WarmClient", FakeWarm)
    om._CLIENTS.clear()


def test_flatten_carries_tool_calls_results_and_the_closing_lines():
    msgs = [SystemMessage(content="SOUL"), HumanMessage(content="is the office light on?"),
            AIMessage(content="", tool_calls=[{"name": "light_state", "args": {"entity_id": "light.office_light_1"}, "id": "t1"}]),
            ToolMessage(content='{"state": "off"}', tool_call_id="t1")]
    p = _flatten(msgs)
    assert p.startswith("[System instructions]\nSOUL")
    assert '<called tool light_state with {"entity_id": "light.office_light_1"}>' in p
    assert "[Tool result light_state]: {\"state\": \"off\"}" in p
    assert _RESULTS_FINAL_LINE in p and _FORCE_TOOL_LINE not in p
    assert p.endswith("\nAerys:")
    forced = _flatten(msgs[:2], force_tool=True)
    assert _FORCE_TOOL_LINE in forced and _RESULTS_FINAL_LINE not in forced


def test_bind_tools_copies_with_schemas_and_forced_first_pass(monkeypatch):
    fresh(monkeypatch)
    base = ClaudeOAuthChatModel(model="claude-sonnet-5")
    auto = base.bind_tools([light_state])
    forced = base.bind_tools([light_state], tool_choice="any")
    assert base.bound_tools == [] and auto.bound_tools[0]["name"] == "light_state"
    assert auto.force_tool is False and forced.force_tool is True
    assert "entity_id" in json.dumps(auto.bound_tools[0]["parameters"])
    # a deferred tool call becomes tool_calls on the AIMessage, with usage + rate-limit metadata
    out = forced.invoke([SystemMessage(content="S"), HumanMessage(content="is the office light on?")])
    assert out.tool_calls[0]["name"] == "light_state" and out.tool_calls[0]["id"] == "toolu_1"
    assert out.usage_metadata["input_tokens"] == 100
    assert out.response_metadata["rate_limit"]["type"] == "five_hour"
    assert _FORCE_TOOL_LINE in FakeWarm.made[0].prompts[0]
    # one warm client per (model, tool set): the auto and forced copies share it, chat does not
    auto.invoke([HumanMessage(content="hi")])
    base.invoke([HumanMessage(content="hi")])
    assert len(FakeWarm.made) == 2
    assert FakeWarm.made[0].schemas and FakeWarm.made[1].schemas == []


def test_second_pass_carries_the_result_and_answers(monkeypatch):
    fresh(monkeypatch)
    m = ClaudeOAuthChatModel(model="claude-sonnet-5").bind_tools([light_state])
    m.invoke([HumanMessage(content="x")])
    FakeWarm.made[0].reply = ("The office light is off.", [], {"stop_reason": "end_turn"})
    out = m.invoke([HumanMessage(content="is it on?"),
                    AIMessage(content="", tool_calls=[{"name": "light_state", "args": {"entity_id": "light.office_light_1"}, "id": "t1"}]),
                    ToolMessage(content="off", tool_call_id="t1")])
    assert out.content == "The office light is off." and out.tool_calls == []
    assert _RESULTS_FINAL_LINE in FakeWarm.made[0].prompts[-1]


def test_factory_puts_text_actions_on_the_plan_only_when_opted_in(monkeypatch):
    fresh(monkeypatch)
    api_only = Settings(_env_file=None, anthropic_api_key="t", model_backend="oauth")
    assert not isinstance(build_api_tool_model(api_only, [light_state]), SurfaceSplitToolModel)
    assert FakeWarm.made == []                                   # default: action path stays metered
    s = Settings(_env_file=None, anthropic_api_key="t", model_backend="oauth", oauth_tool_backend="oauth")
    model = build_api_tool_model(s, [light_state])
    assert isinstance(model, SurfaceSplitToolModel)
    tok = REFLEX_SURFACE.set("discord")
    try:
        out = model.invoke([HumanMessage(content="is the office light on?")], specialist=True)
    finally:
        REFLEX_SURFACE.reset(tok)
    assert out.tool_calls and out.tool_calls[0]["name"] == "light_state"
    assert _FORCE_TOOL_LINE in FakeWarm.made[-1].prompts[-1]    # forced first specialist pass
    both = Settings(_env_file=None, anthropic_api_key="t", model_backend="oauth", oauth_tool_backend="oauth",
                    oauth_voice_backend="oauth")
    assert isinstance(build_api_tool_model(both, [light_state]), ToolModelPair)


def test_double_failure_is_a_typed_backend_error(monkeypatch):
    fresh(monkeypatch)

    class Dead(FakeWarm):
        def ask(self, prompt):
            raise OAuthBackendError("CLIConnectionError: gone / retry CLIConnectionError: gone")

    monkeypatch.setattr(om, "_WarmClient", Dead)
    m = ClaudeOAuthChatModel(model="claude-sonnet-5")
    try:
        m.invoke([HumanMessage(content="hi")])
    except OAuthBackendError as e:
        assert "retry" in str(e)
    else:
        raise AssertionError("expected OAuthBackendError")
