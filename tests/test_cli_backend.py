"""MODEL_BACKEND=cli: chat tiers and the tool model on the Claude Max subscription
(langchain-claude-cli over the official Agent SDK; our graph runs the tools).
Chris 2026-09-20 00:52: "I would be elated if i can shed my API costs".
Offline: the CLI class is replaced by a fake; nothing spawns a subprocess."""
import threading
import time

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langchain_core.tools import tool

import aerys_v2.factory as factory
from aerys_v2.config import Settings
from aerys_v2.factory import (SurfaceSplitToolModel, ToolModelPair, build_api_tool_model, build_model,
                              tier_models_for)
from aerys_v2.reflex import REFLEX_SURFACE


class FakeCli(GenericFakeChatModel):
    """Records constructor args like ChatClaudeCli would receive them."""
    seen: list = []
    bound: list = []

    def __init__(self, **kw):
        FakeCli.seen.append(kw)
        super().__init__(messages=iter([AIMessage(content=f"cli:{kw['model']}")] * 50))

    @property
    def _llm_type(self) -> str:
        return "fake-cli"

    def bind_tools(self, tools, **kwargs):
        # ChatClaudeCli binds tools natively (deferred execution); the fake records the ask.
        FakeCli.bound.append([getattr(t, "name", str(t)) for t in tools] + [kwargs.get("tool_choice")])
        return self


@tool
def light_state(entity_id: str) -> str:
    """Return the state of a light."""
    return "off"


@pytest.fixture(autouse=True)
def fake_cli(monkeypatch):
    FakeCli.seen = []
    FakeCli.bound = []
    monkeypatch.setattr(factory, "CLI_MODEL_CLASS", FakeCli)
    yield


def settings(**kw):
    base = dict(anthropic_api_key="test", model_backend="cli", cli_prewarm=False)
    return Settings(_env_file=None, **{**base, **kw})


def test_api_backend_is_untouched():
    s = Settings(_env_file=None, anthropic_api_key="test")
    build_model(s); tier_models_for(s); build_api_tool_model(s, [light_state])
    assert FakeCli.seen == []


def test_build_model_and_all_three_tiers_ride_the_cli():
    s = settings()
    build_model(s)
    tiers = tier_models_for(s)
    names = [kw["model"] for kw in FakeCli.seen]
    assert names[0] == s.model
    assert set(names[1:]) == {s.tier_fast_model, s.tier_standard_model, s.tier_deep_model}
    assert all(kw["persistent"] is True and kw["max_retries"] == 0 for kw in FakeCli.seen)
    assert tiers["fast"].invoke("hi").content.startswith("cli:")


def test_persistent_is_a_setting():
    tier_models_for(settings(cli_persistent=False))
    assert all(kw["persistent"] is False for kw in FakeCli.seen)


def test_tool_model_binds_our_tools_on_the_cli_and_keeps_voice_metered():
    model = build_api_tool_model(settings(), [light_state])
    assert isinstance(model, SurfaceSplitToolModel)
    assert isinstance(model._text, ToolModelPair) and isinstance(model._voice, ToolModelPair)
    # text surface → cli pair; voice surface → the metered pair (no FakeCli inside it)
    token = REFLEX_SURFACE.set("discord")
    try:
        out = model.invoke([("human", "is the light on?")], specialist=True)
        assert out.content.startswith("cli:")
    finally:
        REFLEX_SURFACE.reset(token)
    cli_models = [kw["model"] for kw in FakeCli.seen]
    assert set(cli_models) >= {settings().model, settings().tier_fast_model}
    assert all(kw["max_tokens"] == 1024 for kw in FakeCli.seen)
    assert ["light_state", None] in FakeCli.bound          # our tool, bound on the cli model
    assert ["light_state", "any"] in FakeCli.bound         # the forced first specialist pass too


def test_voice_can_ride_the_cli_when_allowed():
    model = build_api_tool_model(settings(cli_voice_backend="cli"), [light_state])
    assert isinstance(model, ToolModelPair) and not isinstance(model, SurfaceSplitToolModel)


def test_tool_backend_switch_returns_the_action_path_to_the_api():
    model = build_api_tool_model(settings(cli_tool_backend="api"), [light_state])
    assert isinstance(model, ToolModelPair) and not isinstance(model, SurfaceSplitToolModel)
    assert FakeCli.seen == []


def test_prewarm_runs_one_turn_on_a_daemon_thread_and_never_raises():
    class Boom:
        def invoke(self, *a, **k):
            raise RuntimeError("cli missing")

    factory.prewarm_cli_model(Boom())                      # must not raise
    hit = threading.Event()

    class Ok:
        def invoke(self, msgs, **k):
            hit.set()

    factory.prewarm_cli_model(Ok())
    assert hit.wait(2)
    tier_models_for(settings(cli_prewarm=True))            # wired through the setting
    time.sleep(0.05)


def test_pins_are_in_pyproject_and_lock():
    py = open("pyproject.toml").read()
    lock = open("uv.lock").read()
    assert 'langchain-claude-cli==1.2.1' in py and 'claude-agent-sdk>=0.2.144,<0.3' in py
    assert 'name = "langchain-claude-cli"' in lock
