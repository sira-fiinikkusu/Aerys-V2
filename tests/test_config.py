import pytest
from pydantic import ValidationError
from aerys_v2.config import Settings


def test_settings_loads_with_key(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    settings = Settings(_env_file=None)
    assert settings.model == "claude-sonnet-5"
    assert settings.otlp_endpoint is None


def test_settings_requires_api_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(ValidationError):
        Settings(_env_file=None)
