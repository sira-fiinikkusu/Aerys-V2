"""Offline tests for the OAuth backend — the SDK boundary is faked, nothing spawns."""

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from aerys_v2.config import Settings
from aerys_v2.factory import build_model
from aerys_v2.oauth_model import ClaudeOAuthChatModel, _flatten


def settings(backend: str) -> Settings:
    return Settings(anthropic_api_key="sk-test", model_backend=backend)  # type: ignore[arg-type]


def test_factory_picks_oauth_backend():
    m = build_model(settings("oauth"))
    assert isinstance(m, ClaudeOAuthChatModel)


def test_factory_default_stays_api():
    m = build_model(settings("api"))
    assert not isinstance(m, ClaudeOAuthChatModel)


def test_flatten_system_leads_and_speakers_labeled():
    prompt = _flatten(
        [
            SystemMessage(content="be aerys"),
            HumanMessage(content="hi"),
            AIMessage(content="hey"),
            HumanMessage(content="what number?"),
        ]
    )
    assert prompt.startswith("[System instructions]\nbe aerys\n\n")
    assert prompt.endswith("User: hi\nAerys: hey\nUser: what number?\nAerys:")


def test_generate_uses_result_message(monkeypatch):
    model = ClaudeOAuthChatModel()

    def fake_query(prompt):
        assert "User: ping" in prompt
        return "pong"

    monkeypatch.setattr(model, "_query", fake_query)
    out = model.invoke([SystemMessage(content="s"), HumanMessage(content="ping")])
    assert out.content == "pong"


def test_error_result_raises(monkeypatch):
    model = ClaudeOAuthChatModel()

    def fake_query(prompt):
        raise RuntimeError("oauth backend error: 'refused'")

    monkeypatch.setattr(model, "_query", fake_query)
    with pytest.raises(RuntimeError):
        model.invoke([HumanMessage(content="x")])


def test_connect_disables_all_builtin_tools(monkeypatch):
    """Regression guard for the 2026-07-03 voice bug: `allowed_tools=[]` is only
    auto-permission — the CLI still exposes every built-in tool unless `tools=[]`
    is set, and a tool attempt under max_turns=1 dies as error_max_turns."""
    import claude_agent_sdk

    captured = {}

    class FakeClient:
        def __init__(self, options=None):
            captured["options"] = options

        async def connect(self):
            pass

    monkeypatch.setattr(claude_agent_sdk, "ClaudeSDKClient", FakeClient)
    from aerys_v2.oauth_model import _WarmClient

    w = _WarmClient("claude-sonnet-5")
    client = w._run(w._connect())
    assert isinstance(client, FakeClient)
    opts = captured["options"]
    assert opts.tools == []          # no built-in tools EXIST for the chat backend
    assert opts.allowed_tools == []  # and none would be auto-permitted anyway
    assert opts.max_turns == 1
