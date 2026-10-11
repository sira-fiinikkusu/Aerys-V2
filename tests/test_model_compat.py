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


@pytest.mark.parametrize("model,ok", [
    ("claude-opus-6", False), ("claude-sonnet-6", False), ("claude-haiku-6", False),
    ("claude-sonnet-5-5-20261001", False), ("claude-mythos-5", False), ("", False), (None, False),
    ("claude-haiku-4-5-20251001", True), ("claude-sonnet-5-20260601", True),
])
def test_a_model_nobody_measured_is_never_forced(model, ok):
    """An allow-list: the next release that refuses forcing must not break the API path
    the way Sonnet 5.5 did on 10/07. Whole ids or dated snapshots only."""
    assert supports_forced_tool_choice(model) is ok


def test_the_allow_list_keeps_every_verdict_the_rack_runs_on():
    """Every model id the deployed containers and tiers use (Settings defaults + the Jetson's
    MODEL pins on 2026-10-11) keeps the verdict the 10/07 deny-list gave it, so swapping the
    list changed nothing that runs today."""
    deny_list_verdict = {
        "claude-sonnet-5-5": False,   # brain/soak/telegram/extractor MODEL; tier_standard
        "claude-sonnet-5": True,      # gaps, email
        "claude-haiku-5-5": True,     # tier_fast; extraction_anthropic_model
        "claude-opus-5-5": False,     # tier_deep
        "claude-fable-5-1": False, "claude-mythos-5": False,
    }
    defaults = Settings(_env_file=None, anthropic_api_key="sk-test")
    for model in (defaults.model, defaults.tier_fast_model, defaults.tier_standard_model,
                  defaults.tier_deep_model, defaults.extraction_anthropic_model):
        assert model in deny_list_verdict, model
    for model, verdict in deny_list_verdict.items():
        assert supports_forced_tool_choice(model) is verdict, model


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
