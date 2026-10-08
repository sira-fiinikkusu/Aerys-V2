"""What each Claude model refuses, measured against the live API on 2026-10-07.

The 5.5 upgrade broke her voice path that night: the router (Haiku 5.5) sent
`temperature` ("deprecated for this model"), and the API action path forced a tool
call (`tool_choice` any), which Sonnet 5.5 refuses outright. Plain test calls had
passed; the exact settings her code sends had not been tried. These pin the rules.
"""
import pytest
from langchain_core.tools import tool

from aerys_v2.anthropic_model import accepts_temperature, build_metered_model, supports_forced_tool_choice
from aerys_v2.config import Settings


@pytest.mark.parametrize("model,ok", [
    ("claude-haiku-4-5", True), ("claude-haiku-4-5-20251001", True),
    ("claude-haiku-5-5", False), ("claude-sonnet-5", False), ("claude-sonnet-5-5", False),
    ("claude-opus-4-8", False), ("claude-opus-5", False), ("claude-opus-5-5", False),
    ("claude-fable-5-1", False),
])
def test_only_old_haiku_still_takes_a_temperature(model, ok):
    assert accepts_temperature(model) is ok


@pytest.mark.parametrize("model,ok", [
    ("claude-haiku-4-5", True), ("claude-haiku-5-5", True), ("claude-sonnet-5", True),
    ("claude-opus-4-8", True), ("claude-opus-5", True),
    ("claude-sonnet-5-5", False), ("claude-opus-5-5", False), ("claude-fable-5-1", False),
])
def test_forced_tool_choice_is_refused_by_the_newest_models(model, ok):
    assert supports_forced_tool_choice(model) is ok


def settings(**overrides):
    return Settings(_env_file=None, anthropic_api_key="sk-test", **overrides)


def test_the_metered_builder_drops_temperature_where_it_is_refused():
    assert build_metered_model(settings(), model="claude-haiku-5-5", api_key="sk-test",
                               temperature=0, max_tokens=8).temperature is None
    assert build_metered_model(settings(), model="claude-haiku-4-5", api_key="sk-test",
                               temperature=0, max_tokens=8).temperature == 0


@tool
def light(operation: str) -> str:
    """Control the test light."""
    return "ok"


def test_the_api_action_path_forces_a_tool_only_where_the_model_allows_it():
    from aerys_v2.factory import build_api_tool_model

    pair = build_api_tool_model(settings(model="claude-sonnet-5-5", tier_fast_model="claude-haiku-5-5"), [light])
    assert "tool_choice" not in pair._forced.kwargs, "Sonnet 5.5 refuses a forced tool call"
    assert pair._fast_forced.kwargs["tool_choice"] == {"type": "any"}, "Haiku 5.5 takes it"


def test_the_memory_labeler_sends_no_temperature_to_haiku_5_5(monkeypatch):
    import langchain_anthropic
    from aerys_v2.factory import memory_key_labeler_for

    seen = {}
    monkeypatch.setattr(langchain_anthropic, "ChatAnthropic", lambda **kw: seen.update(kw) or object())
    memory_key_labeler_for(settings(tier_fast_model="claude-haiku-5-5"))
    assert seen["model"] == "claude-haiku-5-5" and "temperature" not in seen


def test_the_router_sends_no_temperature_to_haiku_5_5(monkeypatch):
    import aerys_v2.anthropic_model as am
    from aerys_v2.router import router_for

    seen = {}
    real = am.build_metered_model

    def spy(settings_, **kwargs):
        model = real(settings_, **kwargs)
        seen["temperature"] = model.temperature
        return model

    monkeypatch.setattr(am, "build_metered_model", spy)
    router_for(settings(), "soul")
    assert "temperature" in seen and seen["temperature"] is None
