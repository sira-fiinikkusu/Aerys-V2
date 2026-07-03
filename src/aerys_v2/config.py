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

    # None = Discord transport OFF (the spike only arms when a token is present).
    # One gateway client covers guild AND DMs — the katerlol two-adapter IPC race
    # (and its watchdog liturgy) structurally cannot exist here.
    discord_bot_token: SecretStr | None = None
    discord_guild_id: int | None = None          # only this guild is served (None = DMs only)
    discord_reply_channel_ids: str = ""          # csv of guild channel ids to listen in ("" = all)

    # HTTP API (--serve). None = the /ask door stays locked shut; callers (HA voice
    # pipeline, future satellites) present this as a Bearer token. LAN-only surface.
    api_token: SecretStr | None = None
    api_port: int = 8300

    # None = DB-backed services OFF. Tests and CI never need a live Postgres —
    # the services take an injected connection, and nothing connects unless this
    # is set. When set: postgresql://sira:***@192.168.1.231:5432/aerys — the same
    # NAS database the n8n workflows hit; V2 reads it directly, no webhook hop.
    database_url: str | None = None
