import logging
import re
from pathlib import Path
from urllib.parse import urlsplit

from pydantic import SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

log = logging.getLogger(__name__)


class Settings(BaseSettings):
    # tells pydantic WHERE to read from (like pointing dotenv at a file)
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # a field with NO default = REQUIRED. missing -> ValidationError at startup.
    # SecretStr masks the value in logs/reprs (prints ****, not the key)
    # "api" = metered ChatAnthropic (needs anthropic_api_key). "oauth" = the Claude
    # Agent SDK on the Max subscription — zero API tokens for daily conversation.
    # "local" = an OpenAI-compatible server (Ollama/llama.cpp) at local_model_base_url;
    # every tier collapses onto it — local mode exists for staging twins and for
    # answering when the internet is gone, not for rationing spend.
    # The api key stays REQUIRED either way: evals/CI/fallback run on it.
    model_backend: str = "api"
    anthropic_api_key: SecretStr
    model: str = "claude-sonnet-5"   # daily driver; env MODEL overrides; opus returns via tier routing
    soul_file_path: Path = Path("config/soul.md")
    otlp_endpoint: str | None = None

    # Local model door (owner-gated by design: the alignment layer is the soul file,
    # not a vendor). base_url points at an OpenAI-compatible /v1; the placeholder
    # api key satisfies clients that require one.
    local_model_base_url: str = "http://127.0.0.1:11434/v1"
    local_model_name: str = "hermes3:8b"
    # CPU inference is slow by nature — a 7B writing a long reply can blow well
    # past the metered world's 60s assumption, and an aborted+retried generation
    # doubles the wait. The local door gets its own clock.
    local_model_timeout_s: float = 180.0
    # Set (e.g. to local_model_base_url) to arm metered->local failover: a chat
    # turn whose metered call dies falls back to the local model instead of dying,
    # logs a WARNING, and stamps 'local_model_fallback' into v2_turns.degraded.
    # None = off (the pre-8/03 behavior, byte-for-byte).
    local_fallback_url: str | None = None

    # None = Discord transport OFF (the spike only arms when a token is present).
    # One gateway client covers guild AND DMs — the katerlol two-adapter IPC race
    # (and its watchdog liturgy) structurally cannot exist here.
    discord_bot_token: SecretStr | None = None
    discord_guild_id: int | None = None          # only this guild is served (None = DMs only)
    discord_reply_channel_ids: str = ""          # csv of guild channel ids to listen in ("" = all)
    # The Aerys Admin role (guild role id) gating /admin-link + /admin-unlink.
    # None = admin slash commands always refuse (fail-closed, same posture as
    # every other optional credential). n8n mapping: 03-02's "Check Admin Role"
    # HTTP call — discord.py reads roles off the interaction, no API round-trip.
    discord_admin_role_id: int | None = None

    # None = Telegram transport OFF (same arming pattern as discord_bot_token — the
    # runner only wires up when a token is present). One aiogram long-polling
    # session covers DMs AND groups; n8n mapping is workflow 02-02 Telegram Adapter
    # (K1jR1tpKZTOiid8N), the webhook that fed the retiring Core Agent.
    telegram_bot_token: SecretStr | None = None
    telegram_chat_ids: str = ""                   # csv of group chat ids to serve ("" = DMs only; a group is fail-closed until its chat id is listed — mirrors discord_guild_id)

    # HTTP API (--serve). None = the /ask door stays locked shut; callers (HA voice
    # pipeline, future satellites) present this as a Bearer token. LAN-only surface.
    api_token: SecretStr | None = None
    api_port: int = 8300

    @field_validator("api_token", mode="after")
    @classmethod
    def _blank_api_token_is_none(cls, v: SecretStr | None) -> SecretStr | None:
        # An empty/whitespace API_TOKEN in .env parses to SecretStr('') — NOT None.
        # Coerce blank to None so the --serve gate + require_token refuse to start
        # (fail-closed); otherwise the Bearer credential becomes the literal
        # "Bearer " (empty token) — trivially guessable, and this repo is public.
        if v is not None and not v.get_secret_value().strip():
            return None
        return v

    # None = DB-backed services OFF. Tests and CI never need a live Postgres —
    # the services take an injected connection, and nothing connects unless this
    # is set. When set: postgresql://user:pass@localhost:5432/aerys_v2 — the
    # brain's OWN database on the NAS (checkpoints, outbox, model-usage cap).
    # MUST target `aerys_v2`, never prod `aerys` — run_boot_assertions() below
    # refuses to start otherwise (the two-database footgun, made unbootable).
    database_url: str | None = None

    # ---- MEMORY-RETRIEVAL block (long-term context) --------------------------
    # None = memory context OFF. Points at the PROD aerys database (memories/
    # persons/platform_identities/audit_log). Retrieval treats its connections
    # as READ-ONLY; since the 7/5 cutover the WRITE doors on this URL are the
    # extraction worker (behind the v2_writer_lease memory_extraction lease)
    # and the owner-facing slash commands (/aerys-tell, -forget, -correct —
    # explicit user edits, audit-logged). Kept separate from database_url on
    # purpose — the checkpointer may live in its own DB, and "durable threads"
    # vs "prod memories" are different blast radii.
    memories_database_url: str | None = None

    # Channel-recent room context (cross-surface continuity, multi-person half):
    # how many recent turns of a PUBLIC channel to splice into the system prompt so
    # she holds the shared room on top of the caller's person-keyed thread. Read from
    # v2_turns (database_url); None database_url = the feature is off. Only public
    # turns ever query it — DMs never do.
    room_context_limit: int = 50

    # The owner's persons.id (UUID string). HTTP callers can't prove who they
    # are beyond the Bearer token, so when this is set, voice + /ask identities
    # resolve to the owner — voice-Chris retrieves HIS memories instead of an
    # anonymous "http-caller" bucket that matches nothing in the database.
    owner_person_id: str | None = None

    # Additional person_ids (CSV) granted ACTION/house-control access beyond the
    # owner. The owner is ALWAYS included implicitly (factory.action_allowlist_for).
    # Extending access = add a person_id here, no code change — this is where
    # Megan's person_id lands once her identity is solutioned (identical house
    # access to Chris). Empty = owner only.
    house_control_person_ids: str = ""

    # ---- TOOLS block (Option C hybrid, owner-ratified) -----------------------
    # Chat turns stay on whatever model_backend says (oauth = free daily driver);
    # TOOL turns always run on the metered API backend — the SDK backend is
    # chat-only. None ha_token = the home_control tool (and the whole action
    # path: router + subgraph) simply doesn't exist — same arming pattern as
    # discord_bot_token. n8n mapping: this is the 07-01 "HA Action" workflow's
    # credential check, done at construction time instead of per-execution.
    ha_base_url: str = "http://homeassistant.local:8123"   # HA Green on the LAN
    ha_token: SecretStr | None = None
    # csv of entity_ids the Brain may WRITE to during beta (reads unrestricted).
    # e.g. "light.office_lamp,switch.desk_fan". Empty = every write refused —
    # the tool exists but is read-only, which is a valid canary stage zero.
    ha_canary_entities: str = ""
    # The house alarm panel entity (e.g. "alarm_control_panel.panel"). Empty =
    # the control_alarm tool doesn't exist — same arming pattern as ha_token.
    # Owner-commissioned 2026-08-07; the tool itself carries the owner-only and
    # disarm-surface gates (see tools/alarm.py), this knob only turns it on.
    ha_alarm_entity: str = ""
    # The input_text helper the sticky_note tool writes (the household e-ink
    # note slot, e.g. "input_text.sticky_note_from_aerys"). Empty = the tool
    # doesn't exist. Owner-commissioned 2026-08-07, the night Sticky One woke.
    ha_sticky_note_entity: str = ""
    # name=entity csv of the household to-do lists the manage_list tool may
    # touch (e.g. "shopping=todo.shopping_list,tasks=todo.tasks"). Empty = the
    # tool doesn't exist. A closed map on purpose (owner requirement
    # 2026-08-08: she must be CONSISTENT about lists) — the tool refuses names
    # outside it, so items can't scatter into invented lists.
    ha_todo_lists: str = ""
    # csv of calendar entities the calendar_events tool may read (gap #27,
    # owner-blessed 2026-08-07). Empty = the tool doesn't exist. An explicit
    # ALLOWLIST on purpose, never auto-discovery: the hub aggregates calendars
    # other integrations sync — including other people's shared calendars —
    # and reading those into replies would leak schedules the owner never
    # asked her to know. e.g. "calendar.personal,calendar.home".
    ha_calendar_entities: str = ""
    # Optional generic `timer.*` helper entity for the timer tool's NO-DEVICE
    # fallback (text/DM/CLI turns carry no originating satellite, so there is no
    # native LED-wheel timer to start). None = no fallback: the tool honestly says
    # it can't set a device timer from a text channel instead of pretending. When
    # set (e.g. "timer.aerys_fallback"), it starts that non-visual helper as a
    # best-effort background timer and is honest it won't ring on a speaker. Arms
    # nothing on its own — the timer tool itself is gated on ha_token like the
    # rest of the home half.
    ha_timer_fallback_entity: str | None = None

    # ---- MUSIC (Music Assistant + Spotify, the 07-01 Play Music rebirth) -----
    # The Music Assistant CONFIG ENTRY id in HA (Settings → Devices & Services →
    # Music Assistant) — music_assistant.search requires it. Empty = the music
    # tool doesn't exist this boot, same arming pattern as every optional half.
    ha_music_config_entry: str = ""
    # csv of "device_id=media_player_entity" pairs mapping an originating
    # satellite's device_id to the Music Assistant player in THAT room — the
    # owner spec: music plays on the device that heard the request. Same CSV
    # convention as ha_satellite_map. This map is ALSO the complete allowlist of
    # speakers the tool may drive; named targets resolve against it and nothing
    # else.
    ha_music_players: str = ""
    # Where music goes for turns with no originating device (text/DM/CLI).
    # None = those callers are asked to name a speaker instead of guessing.
    ha_music_default_player: str | None = None

    # The house's weather entity — get_weather reads current conditions and
    # forecasts from it (weather.get_forecasts). Local weather must come from
    # HERE, not web search (the Rotonda-Switzerland incident, 2026-07-19).
    ha_weather_entity: str = "weather.forecast_home"

    # ---- KAEL DESK LINE (gap #14 — her own ask, 2026-07-25) ------------------
    # Kael's channel-server endpoint for real-time pings (e.g.
    # "http://<host>:8399/aerys/message") + her lane's bearer token. Both set =
    # the message_kael tool arms; either None = tool absent, zero cost — same
    # arming pattern as every optional half. Address and token stay in the
    # environment: a LAN IP and a credential, and this repo is public.
    kael_desk_url: str | None = None
    kael_desk_token: SecretStr | None = None

    # ---- PANEL FACE (reTerminal desk avatar) ---------------------------------
    # The panel's state endpoint (e.g. "http://<panel-ip>:8300/state"). When set,
    # the brain mirrors each turn's phase onto her face: working while tools
    # grind, speaking while words play, mood-idle after. None = no panel, zero
    # cost — same arming pattern as every optional half. The address stays in
    # the environment: it's a LAN IP and this repo is public.
    panel_state_url: str | None = None
    # Her circadian rhythm (owner ask 2026-07-19): the --serve container runs a
    # presence watcher that darkens the panel when this occupancy entity goes
    # off AND the listed lights are out (the vacancy automation's fingerprint),
    # and wakes it with a little emote when occupancy returns. Empty = no
    # watcher — same arming pattern as every optional half.
    panel_presence_entity: str | None = None
    panel_presence_lights: str = ""

    # ---- STORM WATCH (owner ask 2026-08-17) -----------------------------------
    # Gulf-coast hurricane season, tiered so awareness beats the prep crowds:
    # NHC outlook bands days ahead, new named storms, and official NWS
    # watches/warnings for the exact home point. "lat,lon" — empty = no
    # watcher, zero cost (the standard arming pattern). The point stays in
    # the environment: it is a street address in disguise.
    storm_watch_latlon: str = ""
    # csv of EXACT NWS event names to alarm on; empty = the built-in set
    # (Hurricane/Tropical Storm/Storm Surge/Tornado families + a couple of
    # exact warnings — see storm_watch.py).
    storm_alert_events: str = ""
    # HA notify service for the phone leg (e.g. "mobile_app_<device>");
    # empty = Discord DM only.
    storm_notify_service: str = ""

    # ---- SPOKEN FOLLOW-UP (voice actions) ------------------------------------
    # Owner rule (2026-07-03): if a device action completes within this many
    # seconds of the ack going out, the spoken follow-up is SKIPPED — the light
    # changing IS the feedback. Slow actions and FAILURES are always spoken.
    # The follow-up lands in thread history either way (silent record).
    voice_followup_skip_s: float = 6.0
    # False-wake grace (owner ask 2026-08-27): when the router judges a
    # voice/lens capture was never directed at Aerys, drop it silently instead
    # of answering into someone else's conversation. Kill-switch: set
    # VOICE_DROP_UNADDRESSED=false to disarm instantly — the router keeps
    # judging (receipts continue), the drop just stops happening.
    voice_drop_unaddressed: bool = True
    # The assist satellite entity that speaks follow-ups (e.g.
    # "assist_satellite.home_assistant_voice_..._assist_satellite").
    # None = no spoken follow-ups (history-only) — same arming pattern as every
    # other optional transport.
    ha_announce_entity: str | None = None
    # csv of "device_id=entity_id" pairs mapping a ConversationInput.device_id to
    # the assist_satellite entity that should speak follow-ups FOR that device.
    # Empty or an unmapped device_id falls back to ha_announce_entity (today's
    # single-satellite behavior) — never a silent drop. Same CSV convention as
    # ha_canary_entities.
    ha_satellite_map: str = ""
    # csv of "device_id=domain.service" pairs for SPEAKERLESS voice devices
    # (e-ink displays): a follow-up for a mapped device is WRITTEN to its
    # screen via that HA service (an ESPHome user action, e.g.
    # "esphome.reterminal_sticky_show_followup") instead of spoken or fired
    # as a phone event. Checked after the satellite map, before the phone
    # fallback. Owner ask 2026-08-08: the Sticky rendered her ack but her
    # final reply only reached speaking surfaces.
    ha_display_followups: str = ""
    # csv of ConversationInput.device_ids that are SHARED HOUSEHOLD surfaces
    # (the Stickies): turns from them carry a speaker-may-not-be-the-owner
    # prompt block. The Bearer still authenticates the DEVICE as owner infra —
    # this only corrects the social assumption. Empty = feature off.
    # Entries are "device_id" or "device_id=Label" — a label lets her name
    # WHICH surface is speaking (Sticky One/Two/Three).
    ha_shared_surface_ids: str = ""

    # Embeddings seam — mirrors the n8n "Generate Embedding" HTTP Request node:
    # OpenAI-compatible /embeddings via OpenRouter (memory.EMBED_MODEL =
    # openai/text-embedding-3-small, 1536-dim). The model MUST match what the
    # write pipeline stored in memories.embedding, or cosine distance compares
    # apples to bananas. base_url is a setting so a different OpenAI-compat host
    # (or a local embedding server) is a .env change, not a code change.
    # None api_key = the memory half of context stays empty (profile still works).
    embeddings_api_key: SecretStr | None = None
    embeddings_base_url: str = "https://openrouter.ai/api/v1"

    # Web search seam — mirrors V1's `tavilyTool` community node (credentials
    # iZxeoPSLwObXXEGN / PRAECj0Em1imOqmW) that hung off the research sub-agent.
    # None = the search_web tool (and, if it is the only armed half, the whole
    # action path: router + subgraph) simply doesn't exist — same arming pattern
    # as ha_token and embeddings_api_key. When set, the action agent can look up
    # current events / news / weather / prices instead of guessing from stale
    # training knowledge.
    tavily_api_key: SecretStr | None = None

    # ---- EMAIL (her own mailbox — scope decided 2026-07-11) -------------------
    # Aerys's OWN Gmail account over IMAP/SMTP with an app password — the
    # rebuild of n8n's email trio (Email Sub-Agent kbKrKBVUgwU6n9gg, Gmail
    # Trigger 48toI7JVcl3MnL4n) minus the OAuth dance and minus the morning
    # brief (dropped by owner decision). None app password = the whole email
    # surface (watcher worker + email tools) doesn't exist — the standard
    # arming pattern. The owner's mail is a LATER add behind its own creds.
    email_address: str | None = None              # her Gmail address (also SMTP From)
    email_app_password: SecretStr | None = None   # Google app password (not the account password)
    email_imap_host: str = "imap.gmail.com"
    email_smtp_host: str = "smtp.gmail.com"
    email_poll_seconds: int = 180                 # watcher cadence
    # Where arrival pings land: the owner's Discord user id (DM via the bot's
    # REST API — the watcher is its own container and holds no gateway session).
    email_notify_discord_user_id: str | None = None

    # ---- EXTRACTION WORKER (shadow mode) --------------------------------------
    # The n8n batch extraction (IfqY4BrhBGeQrcTC) re-run as a V2 worker that reads
    # prod conversations READ-ONLY and writes ONLY to aerys_v2 staging tables
    # (migration 002) — output gets diffed against prod before any lease flip.
    # Reuses embeddings_api_key (it's an OpenRouter key) for the extraction LLM.
    extraction_model: str = "anthropic/claude-haiku-4.5"  # v1's extractor, via OpenRouter
    extraction_interval_minutes: int = 60   # loop-mode cadence (v1 cron: hourly)
    extraction_lookback_hours: int = 2      # first-run window when no watermark exists
    extraction_batch_limit: int = 200       # rows per source per pass (v1 LIMIT 200)

    # ---- CHECKPOINTER POOL ----------------------------------------------------
    # How long boot waits for the first checkpointer connection. Deliberately
    # fail-fast: a brain that cannot reach its own memory should not come up
    # quietly. Generous enough to ride out a NAS that is still finishing its
    # own boot, short enough that a genuinely dead DB surfaces as a crash loop
    # rather than a container hanging in "starting" forever.
    checkpoint_pool_open_timeout_s: float = 30.0

    # ---- TIER ROUTING (the V1 classify sandwich, folded into the router) ------
    # n8n mapping: V1's three tier sub-workflows (Sonnet/Opus/Gemini agents) and
    # the modelsConfig dict in Load Config. Tiers are named by ROLE, not vendor
    # (the haiku→gemini rename left a dead name in Parse Classification's
    # validation array — role names survive model swaps). tier applies to CHAT
    # routes on TEXT threads; voice stays pinned to standard (ChannelPolicy,
    # locked: the ~3.6s voice budget can't absorb opus latency, and fast-tier
    # identity wobbles are exactly what got Haiku demoted in V1).
    tier_fast_model: str = "claude-haiku-4-5"       # greetings, trivia — pennies
    tier_standard_model: str = "claude-sonnet-5"    # the daily driver (api backend
    #   only — on the oauth backend, standard IS `model` above: the subscription
    #   client is single-model, so this knob applies when chat bills the API key)
    tier_deep_model: str = "claude-opus-4-8"        # research/analysis — rationed
    # Deep turns per UTC day, enforced atomically in v2_model_usage (migration
    # 003) when database_url is set — V1's aerys_model_usage 10/day opus cap,
    # minus its check-then-increment race. Cap hit -> silently costs nothing:
    # the turn downgrades to standard and the downgrade is logged.
    deep_daily_cap: int = 10


# =============================================================================
# Boot assertions — the env-scare prevention.
#
# Two real incidents drive these checks:
#   1. The watchdog .env bug (2026-05-05): a relative AERYS_ENV_PATH resolved
#      against / under systemd — the service booted "fine" with missing config.
#   2. The two-database footgun: NAS Postgres hosts BOTH `aerys` (prod, n8n's,
#      sacred) and `aerys_v2` (this brain's own). DATABASE_URL pointed at prod
#      would checkpoint V2 threads INTO the production database; conversely,
#      MEMORIES_DATABASE_URL pointed at aerys_v2 reads memories from an empty
#      staging DB and silently retrieves nothing.
# n8n mapping: V1 had no equivalent — a misconfigured workflow just ran wrong
# until someone noticed. Refusing to boot is the upgrade.
# =============================================================================

# The one database this brain may write to. MEMORIES_DATABASE_URL must point at
# prod `aerys` (read-only retrieval) — the names must never swap.
V2_DATABASE_NAME = "aerys_v2"

# .env line shape (what python-dotenv itself accepts): optional `export`, a
# KEY, `=`. Comments and blanks never match.
_ENV_LINE = re.compile(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=")


class BootConfigError(RuntimeError):
    """Fatal misconfiguration found at startup — refuse to serve, say why."""


def database_name(url: str) -> str:
    """The database a postgres URL targets ('' when the URL has no path)."""
    return urlsplit(url).path.lstrip("/")


def duplicate_env_keys(env_file: Path) -> list[str]:
    """Keys assigned more than once in a dotenv file (last one silently wins).

    Deploy-side equivalent, for boxes where the running process can't see the
    file:  awk -F= '/^[A-Za-z_]/{print $1}' .env | sort | uniq -d
    """
    seen: dict[str, int] = {}
    try:
        lines = env_file.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    for line in lines:
        m = _ENV_LINE.match(line)
        if m:
            key = m.group(1)
            seen[key] = seen.get(key, 0) + 1
    return sorted(k for k, n in seen.items() if n > 1)


def run_boot_assertions(settings: Settings, env_file: Path | None = None) -> None:
    """Startup config sanity — called by --serve/--discord BEFORE anything binds.

    Raises BootConfigError on fatal misconfig (wrong database = refuse to
    start); logs loudly on suspicious-but-survivable config (backwards memory
    URL, duplicate .env keys). The direction is deliberate: a write surface
    aimed at the wrong database must never boot, while a read surface aimed
    somewhere odd degrades to empty context — visible, not destructive.
    """
    # 1. FATAL: the brain's own DB (checkpointer, outbox, model-usage cap)
    #    must be aerys_v2 — anything else risks writing into prod `aerys` or
    #    n8n's engine database.
    if settings.database_url is not None:
        name = database_name(settings.database_url)
        if name != V2_DATABASE_NAME:
            raise BootConfigError(
                f"DATABASE_URL targets database {name!r} — the V2 brain writes "
                f"(checkpoints, outbox, model-usage) belong in '{V2_DATABASE_NAME}'. "
                "If you meant the prod memories connection, that is "
                "MEMORIES_DATABASE_URL. Refusing to start."
            )

    # 2. LOUD WARNING: the memories connection pointed at aerys_v2 is the same
    #    mistake mirrored — retrieval reads an empty staging DB and every turn
    #    quietly knows nothing. Survivable (read-only), so warn, don't die.
    if settings.memories_database_url is not None:
        name = database_name(settings.memories_database_url)
        if name == V2_DATABASE_NAME:
            log.warning(
                "MEMORIES_DATABASE_URL targets '%s' — that is the brain's OWN "
                "database, not prod 'aerys'. Memory retrieval will find nothing. "
                "The two URLs look swapped.",
                V2_DATABASE_NAME,
            )

    # 3. LOUD WARNING: duplicate keys in the env file — dotenv keeps the LAST
    #    assignment, so an old line lower in the file silently overrides the
    #    one you just edited (the exact shape of the 2026-05-05 watchdog scare).
    if env_file is None:
        configured = Settings.model_config.get("env_file")
        env_file = Path(configured) if configured else None
    if env_file is not None and env_file.exists():
        dupes = duplicate_env_keys(env_file)
        if dupes:
            log.warning(
                "env file %s assigns these keys more than once (LAST one wins): "
                "%s — delete the stale lines.",
                env_file,
                ", ".join(dupes),
            )

    # 4. FATAL: owner_person_id, when set, must be a real UUID — it defines "who
    #    is the owner" for the HTTP passthrough AND seeds the action allowlist. A
    #    malformed value silently strips the owner of BOTH memories and house
    #    control (a non-UUID never matches a real person_id), so fail fast.
    import uuid

    if settings.owner_person_id is not None:
        try:
            uuid.UUID(settings.owner_person_id)
        except (ValueError, AttributeError, TypeError):
            raise BootConfigError(
                f"OWNER_PERSON_ID {settings.owner_person_id!r} is not a valid UUID "
                "— it defines the owner (HTTP identity + house-control allowlist). "
                "Refusing to start."
            )

    # 5. WARNING: a house_control_person_ids entry that isn't a UUID can never
    #    match a real person_id, so it grants nobody access — surface the typo.
    for pid in (
        p.strip() for p in settings.house_control_person_ids.split(",") if p.strip()
    ):
        try:
            uuid.UUID(pid)
        except (ValueError, AttributeError, TypeError):
            log.warning(
                "HOUSE_CONTROL_PERSON_IDS entry %r is not a UUID — it will never "
                "match a person and grants no access.",
                pid,
            )
