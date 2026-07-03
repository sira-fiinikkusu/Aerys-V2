from pathlib import Path
from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # tells pydantic WHERE to read from (like pointing dotenv at a file)
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # a field with NO default = REQUIRED. missing -> ValidationError at startup.
    # SecretStr masks the value in logs/reprs (prints ****, not the key)
    # "api" = metered ChatAnthropic (needs anthropic_api_key). "oauth" = the Claude
    # Agent SDK on the Max subscription — zero API tokens for daily conversation.
    # The api key stays REQUIRED either way: evals/CI/fallback run on it.
    model_backend: str = "api"
    anthropic_api_key: SecretStr
    model: str = "claude-sonnet-5"   # daily driver; env MODEL overrides; opus returns via tier routing
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

    # ---- MEMORY-RETRIEVAL block (long-term context) --------------------------
    # None = memory context OFF. Points at the PROD aerys database — the same
    # memories/core_claim tables the live n8n pipeline writes. V2 treats this
    # connection as READ-ONLY (retrieval only): while both brains coexist, the
    # n8n batch-extraction workflow stays the sole writer. Kept separate from
    # database_url on purpose — the checkpointer may live in its own DB, and
    # "durable threads" vs "prod memories" are different blast radii.
    memories_database_url: str | None = None

    # The owner's persons.id (UUID string). HTTP callers can't prove who they
    # are beyond the Bearer token, so when this is set, voice + /ask identities
    # resolve to the owner — voice-Chris retrieves HIS memories instead of an
    # anonymous "http-caller" bucket that matches nothing in the database.
    owner_person_id: str | None = None

    # ---- TOOLS block (Option C hybrid, owner-ratified) -----------------------
    # Chat turns stay on whatever model_backend says (oauth = free daily driver);
    # TOOL turns always run on the metered API backend — the SDK backend is
    # chat-only. None ha_token = the home_control tool (and the whole action
    # path: router + subgraph) simply doesn't exist — same arming pattern as
    # discord_bot_token. n8n mapping: this is the 07-01 "HA Action" workflow's
    # credential check, done at construction time instead of per-execution.
    ha_base_url: str = "http://192.168.1.155:8123"   # HA Green on the LAN
    ha_token: SecretStr | None = None
    # csv of entity_ids the Brain may WRITE to during beta (reads unrestricted).
    # e.g. "light.office_lamp,switch.desk_fan". Empty = every write refused —
    # the tool exists but is read-only, which is a valid canary stage zero.
    ha_canary_entities: str = ""

    # Embeddings seam — mirrors the n8n "Generate Embedding" HTTP Request node:
    # OpenAI-compatible /embeddings via OpenRouter (memory.EMBED_MODEL =
    # openai/text-embedding-3-small, 1536-dim). The model MUST match what the
    # write pipeline stored in memories.embedding, or cosine distance compares
    # apples to bananas. base_url is a setting so a different OpenAI-compat host
    # (or a local embedding server) is a .env change, not a code change.
    # None api_key = the memory half of context stays empty (profile still works).
    embeddings_api_key: SecretStr | None = None
    embeddings_base_url: str = "https://openrouter.ai/api/v1"
