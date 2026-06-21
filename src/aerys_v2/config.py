from pathlib import Path
from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # tells pydantic WHERE to read from (like pointing dotenv at a file)
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # a field with NO default = REQUIRED. missing -> ValidationError at startup.
    # SecretStr masks the value in logs/reprs (prints ****, not the key)
    anthropic_api_key: SecretStr
    model: str = "claude-opus-4-8"
    soul_file_path: Path = Path("config/soul.md")
    otlp_endpoint: str | None = None
