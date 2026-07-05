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


def test_telegram_settings_default_off(monkeypatch):
    # Telegram transport is OFF by default — same arming pattern as discord_bot_token.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    s = Settings(_env_file=None)
    assert s.telegram_bot_token is None
    assert s.telegram_chat_ids == ""


def test_telegram_token_read_as_secret(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123:abc")
    monkeypatch.setenv("TELEGRAM_CHAT_IDS", "-1001,-1002")
    s = Settings(_env_file=None)
    assert s.telegram_bot_token is not None
    assert s.telegram_bot_token.get_secret_value() == "123:abc"
    assert s.telegram_chat_ids == "-1001,-1002"


# ---- empty API_TOKEN must fail closed (cross-review CRITICAL, 2026-07-04) -------


def test_blank_api_token_coerces_to_none(monkeypatch):
    # "API_TOKEN=" parses to SecretStr('') not None; the validator coerces it to
    # None so the --serve gate refuses to start rather than accept "Bearer " as auth.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setenv("API_TOKEN", "")
    assert Settings(_env_file=None).api_token is None


def test_whitespace_api_token_coerces_to_none(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setenv("API_TOKEN", "   ")
    assert Settings(_env_file=None).api_token is None


def test_real_api_token_is_kept(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setenv("API_TOKEN", "a-real-token-value")
    s = Settings(_env_file=None)
    assert s.api_token is not None and s.api_token.get_secret_value() == "a-real-token-value"


def test_malformed_owner_person_id_refuses_to_boot(monkeypatch):
    from aerys_v2.config import BootConfigError, run_boot_assertions

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setenv("OWNER_PERSON_ID", "not-a-uuid")
    s = Settings(_env_file=None)
    with pytest.raises(BootConfigError):
        run_boot_assertions(s, env_file=None)


def test_valid_owner_person_id_boots_clean(monkeypatch):
    from aerys_v2.config import run_boot_assertions

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setenv("OWNER_PERSON_ID", "6e6bcbed-03ef-4d17-95d2-89c467414335")
    run_boot_assertions(Settings(_env_file=None), env_file=None)  # no raise
