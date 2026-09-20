"""MODEL_BACKEND=cli is BLOCKED (2026-09-20 Codex review of c91b7aa): langchain-claude-cli
resumes a CLI session by prefix or LangGraph thread_id and carries the previous turn's
context (privacy overlay, memories) into a turn whose context changed. The wiring stays;
every entry point refuses until upstream can run a fresh session per turn on a warm client.
The subscription path with tools is MODEL_BACKEND=oauth + OAUTH_TOOL_BACKEND=oauth."""
import pytest
from langchain_core.tools import tool

from aerys_v2.config import Settings
from aerys_v2.factory import CLI_BLOCKED, build_api_tool_model, build_model, tier_models_for


@tool
def light_state(entity_id: str) -> str:
    """Return the state of a light."""
    return "off"


def settings(**kw):
    return Settings(_env_file=None, anthropic_api_key="test", model_backend="cli", **kw)


@pytest.mark.parametrize("build", [
    lambda s: build_model(s),
    lambda s: tier_models_for(s),
    lambda s: build_api_tool_model(s, [light_state]),
])
def test_cli_backend_is_blocked_at_every_entry_point(build):
    with pytest.raises(RuntimeError, match="blocked"):
        build(settings())
    assert "fresh session" in CLI_BLOCKED or "context" in CLI_BLOCKED


def test_pins_stay_so_the_wiring_still_imports():
    py = open("pyproject.toml").read()
    assert "langchain-claude-cli==1.2.1" in py
