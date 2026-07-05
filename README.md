# Aerys

**A self-hosted AI companion with persistent memory, built on LangGraph.**

Aerys is a personal AI companion you talk to wherever you already are — Discord, Telegram, or by voice through Home Assistant satellites — and it stays *one continuous presence* across all of them. Every surface funnels into a single `ask()` seam in front of a checkpointed [LangGraph](https://langchain-ai.github.io/langgraph/) graph, so long-term memory, model tiering, a privacy gate, and a tool/action layer are built once and cover every channel at once.

`aerys-v2` is the ground-up rewrite of an earlier version that ran as a sprawl of workflow-engine automations. This is that reasoning-and-orchestration layer rebuilt as a self-hosted Python "brain": typed, tested, traceable, and degrade-safe by construction.

> Personal / portfolio project — actively evolving, built in public.

![Python](https://img.shields.io/badge/python-3.11%2B-blue) ![Built on](https://img.shields.io/badge/built%20on-LangGraph-informational) ![License](https://img.shields.io/badge/license-MIT-blue)

---

## What it is

One brain, many front doors. A message arrives from any transport, gets normalized to a transport-neutral shape, is resolved to *who is speaking* (the authorization boundary), and is handed to `ask()` — the single function every transport calls. `ask()` runs one conversational turn against a checkpointed graph and returns the reply. Because every channel goes through that one seam, safety rails, auditing, memory, tiering, and tracing are implemented in exactly one place instead of being re-patched into each adapter.

The design goal throughout: **capability grows without weakening the safety boundary.** New tools, new transports, and new memory sources arm independently, and anything that isn't configured is simply *off* — never half-wired.

---

## Architecture

```
   Discord ─┐
  Telegram ─┤   normalize + resolve identity
     Voice ─┤   (auth boundary: who is this,
  HTTP/curl─┘    what may we tell them?)
              │
              ▼
          ask()  ── the single seam every transport calls
              │
              ├──▶  router (one fast model call)
              │         • chat vs. action
              │         • tier: fast | standard | deep
              │         • a speakable ack for voice
              │
              ├──▶  chat graph  (START → chat → END, checkpointed)
              │         • long-term memory context injected per turn
              │         • short-term privacy gate over history
              │         • per-turn model chosen by tier
              │
              └──▶  action subgraph  (act ⇄ tools → END)
                        • home control (Home Assistant)
                        • media: vision / documents / YouTube
                        • web search
                        • countdown timers

   checkpointer (durability seam)        long-term memory (read-only)
   InMemory  ──or──  Postgres            an existing memories store
```

### One brain, many transports

All four transports — the Discord gateway (`discord.py`), the Telegram gateway (`aiogram`), the HTTP `/ask` door (`FastAPI`/`uvicorn`), and the voice pipeline (Home Assistant → the OpenAI-compatible shim) — do the same three things and nothing more: **normalize → `ask()` → reply.** No model logic lives in a transport.

Conversation threads are **person-keyed**: a given person's Discord DM, guild messages, Telegram chat, and voice turns all fold into one continuous thread (`person:{id}`), so Aerys remembers them as *one relationship* rather than four disconnected sessions. In a shared public channel, a separate room-context block splices in the last N turns of *that channel* (everyone) so she can hold the room on top of the caller's personal thread.

### Model tiering

The router grades how much thinking a turn deserves and picks a model per turn:

| Tier | For | Notes |
|------|-----|-------|
| `fast` | greetings, small talk, trivia | cheapest model |
| `standard` | everyday conversation, Q&A, code help, creative | the default |
| `deep` | genuine research / heavy analysis | rationed by an atomic daily cap |

Tiers are named by **role**, not by vendor, so swapping the underlying model never invalidates the routing contract. The tier is a *hint*: an unknown or missing tier normalizes to `standard`, so a misclassification costs pennies, never a wrong route. Voice turns are pinned to `standard` — the low-latency voice budget can't absorb a heavy model.

### The tool / action layer

When a turn needs to *touch or read* something outside the conversation, the router sends it to a LangGraph tool subgraph (model proposes tool calls → tools execute → loop until it answers in plain text). Every tool obeys the same hard contract: **never raise** (an exception would kill the whole turn), **return honest error strings** the model must relay, and **never claim success the tool didn't confirm.**

- **Home control** — reads and writes to Home Assistant. Writes are guarded by a *canary allowlist* (only explicitly listed entities can be actuated), recorded through a write-ahead outbox, and refuse honestly for anything off the list.
- **Entity search** — a fuzzy read-only index over Home Assistant entities, so the model finds the exact entity id instead of guessing.
- **Media** — vision (analyze an image), document reading (PDF / DOCX / TXT), and YouTube summarization (from captions, no video download).
- **Web search** — current-events / news / weather / price lookups via Tavily.
- **Timers** — starts and cancels native Home Assistant timers on the *originating* voice device, so the satellite's own countdown/ring UX works.

### Memory & privacy

- **Long-term memory** — each turn retrieves the caller's profile and relevant past memories (vector similarity with recency weighting) and injects them into the system prompt. Read-only against an existing memories store; retrieval is fenced so a database hiccup degrades to "no context," never a dead turn.
- **Short-term privacy gate** — every human turn is tagged `public` or `private`. In a public room, private-tagged turns *and their replies* are structurally stripped from what the model sees. The gate is **fail-closed**: anything not explicitly `public` is treated as private in public. A background classifier can later *relax* a DM turn to public only after judging the actual content is general — so general things said privately can carry into shared rooms while health, finances, secrets, and credentials never do.
- **Person-keyed continuity** — identity is resolved per call and rides alongside the turn (never checkpointed into shared state), which is what makes the cross-surface thread safe: a stranger, or any second user, resolves *cold* and can never inherit the owner's memories or identity.

### Auditing & observability

Every completed turn writes one audit row (off the hot path, fail-open) capturing who was resolved, which tier fired, tool activity, latency, and raw-vs-emitted output. Optional [Phoenix](https://github.com/Arize-ai/phoenix)/OpenTelemetry tracing ships every model call and graph step as spans — wired degrade-safe, so if the tracing backend is down the brain still serves.

---

## Features

- **Multi-transport, one brain** — Discord (guild + DMs from a single gateway), Telegram (DMs + groups), an authed HTTP `/ask` door, and voice via Home Assistant. Each transport arms only when its credentials are present.
- **Voice with sub-second acknowledgment** — on a voice device command, the router's spoken ack goes out immediately while the tool loop finishes in the background and appends the real outcome to the thread. A "silent-success" rule skips the spoken follow-up when the device change is its own feedback.
- **Cross-surface memory** — one continuous person-keyed conversation across DMs, groups, and voice, plus a public-channel room-context block.
- **Long-term memory retrieval** — profile + vector-scored memories injected per turn, with recency-weighted (multiplicative) scoring.
- **Short-term privacy gate** — fail-closed public/private redaction of conversation history, with an off-hot-path content classifier that relaxes only genuinely-general content.
- **Model tiering** — fast / standard / deep routing with an atomic per-day cap on the expensive tier.
- **Home control** — canary-allowlisted, outbox-audited writes to Home Assistant, plus unrestricted reads and fuzzy entity discovery.
- **Media understanding** — image analysis, PDF/DOCX/TXT extraction, and YouTube transcript summaries as honest read-only tools.
- **Live web search** — grounded current-events lookups via Tavily.
- **Native voice timers** — countdown timers on the originating satellite.
- **Two model backends** — metered API (`langchain-anthropic`) or a subscription-auth backend via the Claude Agent SDK for daily chat, with tool/router/judge calls always on the metered path.
- **Durable, checkpointed threads** — pluggable checkpointer (in-memory for tests/CI, Postgres for durability) so conversations survive restarts.
- **Turn auditing + Phoenix/OTel tracing** — one queryable row per turn and full span-level visibility, both degrade-safe.
- **Batch memory extraction (shadow + live)** — a background worker that reads conversations and extracts durable memories, runnable in a safe shadow mode (writes only to a staging table) before flipping to live.
- **Self-iteration miner** — an offline worker that mines audit rows for capability gaps Aerys hit, with machine-set provenance so a "real error" signal can't be forged by model text; surfaced to the owner via a read-only `/gaps` endpoint.
- **Boot-time config assertions** — refuses to start on dangerous misconfiguration (e.g., pointing the write database at the wrong target) with a plain-English reason instead of a stack trace.
- **A Home Assistant custom component** (`ha_custom_components/aerys_conversation/`) that lets HA's conversation pipeline talk to the brain and carries the originating device id for per-device voice follow-ups.

---

## Setup & running

### Prerequisites

- **Python 3.11**
- **[uv](https://docs.astral.sh/uv/)** for dependency management
- An **Anthropic API key** (required — evals, CI, and the tool/router path use it even when daily chat runs on the subscription backend)
- Everything else is optional and arms a feature only when configured

### Install

```bash
uv sync --dev          # install into .venv (or: make sync)
cp .env.example .env    # then fill in your keys — .env is gitignored, never commit it
```

### Configuration

All configuration is environment variables (loaded from `.env`), defined and documented in `src/aerys_v2/config.py`. The governing pattern: **`None`/unset = that feature is off.** Nothing half-arms.

**Core**

| Var | Purpose |
|-----|---------|
| `ANTHROPIC_API_KEY` | **Required.** Anthropic API key. |
| `MODEL` | Default chat model id. |
| `MODEL_BACKEND` | `api` (metered) or `oauth` (subscription via the Claude Agent SDK, chat-only). |
| `SOUL_FILE_PATH` | Path to the persona prompt file (falls back to a minimal persona if absent). |
| `TIER_FAST_MODEL` / `TIER_STANDARD_MODEL` / `TIER_DEEP_MODEL` | Per-tier model ids. |
| `DEEP_DAILY_CAP` | Max deep-tier turns per day (enforced when a database is configured). |
| `OTLP_ENDPOINT` | OpenTelemetry/Phoenix collector endpoint (unset = tracing off). |

**Transports** (each off unless set)

| Var | Purpose |
|-----|---------|
| `DISCORD_BOT_TOKEN`, `DISCORD_GUILD_ID`, `DISCORD_REPLY_CHANNEL_IDS` | Discord gateway. |
| `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_IDS` | Telegram gateway. |
| `API_TOKEN`, `API_PORT` | HTTP `/ask` door (Bearer-token auth; a blank token fails closed). |

**Databases**

| Var | Purpose |
|-----|---------|
| `DATABASE_URL` | The brain's *own* database — checkpoints, audit rows, action outbox, model-usage cap. Boot assertions refuse to start if it targets the wrong database. |
| `MEMORIES_DATABASE_URL` | An existing memories store, read-only, for long-term retrieval. |
| `ROOM_CONTEXT_LIMIT` | How many recent public-channel turns to splice in. |

**Identity & tools**

| Var | Purpose |
|-----|---------|
| `OWNER_PERSON_ID` | The owner's person id (UUID); makes authed HTTP/voice callers resolve to the owner. |
| `HOUSE_CONTROL_PERSON_IDS` | Additional person ids allowed to reach the sensitive (house-control) tools. |
| `HA_BASE_URL`, `HA_TOKEN` | Home Assistant base URL + long-lived token (arms home control + timers). |
| `HA_CANARY_ENTITIES` | Comma-separated entity ids the brain may *write* to (empty = read-only). |
| `HA_TIMER_FALLBACK_ENTITY` | Optional generic timer helper for no-device (text) turns. |
| `HA_ANNOUNCE_ENTITY`, `HA_SATELLITE_MAP`, `VOICE_FOLLOWUP_SKIP_S` | Spoken voice follow-up routing. |
| `EMBEDDINGS_API_KEY`, `EMBEDDINGS_BASE_URL` | OpenAI-compatible embeddings (arms the media tools *and* memory embeddings). |
| `TAVILY_API_KEY` | Arms the web-search tool. |

**Extraction worker**

`EXTRACTION_MODEL`, `EXTRACTION_INTERVAL_MINUTES`, `EXTRACTION_LOOKBACK_HOURS`, `EXTRACTION_BATCH_LIMIT`, and (for `--live` only) `N8N_BASE_URL` / `N8N_API_KEY`.

> Describe your own values in `.env`. Do not commit secrets, tokens, private URLs, or internal identifiers — `.env` is gitignored for exactly this reason.

### Database migrations

SQL migrations live in `db/migrations/` (`001`–`005`) and are split across two databases: the brain's own database (audit, outbox, model-usage cap, capability-request tables, room-context columns) and a pgvector-enabled staging database for shadow memory extraction. Apply them with your preferred Postgres client before running any database-backed feature. The brain runs fully without a database (in-memory checkpointer, no auditing) for local development and tests.

### Run modes

The `aerys-v2` entrypoint selects a mode by flag:

```bash
uv run aerys-v2 --health                 # config-load smoke check → prints "ok"
uv run aerys-v2 --ask "hello"            # one-shot turn (the simplest real call path)
uv run aerys-v2 --serve                  # the HTTP /ask door (voice + curl); needs API_TOKEN
uv run aerys-v2 --discord                # Discord gateway; needs DISCORD_BOT_TOKEN
uv run aerys-v2 --telegram               # Telegram gateway; needs TELEGRAM_BOT_TOKEN
uv run aerys-v2 --eval                   # run the LLM-judge eval harness against the graph
uv run aerys-v2 --replay                 # replay captured payloads through ask()
```

Background workers run as a separate process:

```bash
uv run python -m aerys_v2.workers extraction --once          # memory extraction (shadow)
uv run python -m aerys_v2.workers extraction --once --live   # extraction → live memories (gated)
uv run python -m aerys_v2.workers gaps-mine --once           # capability-gap miner
uv run python -m aerys_v2.workers gaps                       # read mined gaps
```

Without `--once`, the workers run on an interval (APScheduler).

### Tests & container

```bash
make test              # uv run pytest — the suite runs fully offline (no DB, no network, fake models)
make build             # native container image
make build-arm64       # cross-build for arm64 (e.g. a Jetson)
```

CI (GitHub Actions) installs with `uv` and runs the pytest suite on every push and pull request.

---

## Design notes

- **Why LangGraph.** The graph is the unit of orchestration: a checkpointed state machine where each node is a small pure-ish function of state. It gives durable per-thread history, a natural place to hang the tool loop, and a clean seam for injecting the checkpointer and the model. The graph shape doesn't change when the storage does.
- **The durability seam.** The checkpointer is *injected*, not hardcoded: `InMemorySaver` for tests and local runs, `PostgresSaver` when a database is configured. Same graph, swappable persistence — conversations survive restarts in production and cost nothing in CI.
- **`None` = feature off (degrade-safe by construction).** Every optional capability — each transport, memory, home control, media, web search, tracing, auditing, the deep-tier cap — is wired through a builder that returns `None` when its config is absent. The brain is backward-compatible with itself: turn a key off and that feature simply doesn't exist, rather than crashing or half-running. Runtime failures in optional subsystems (a down database, a dead tracing backend, a failing tool) log loudly and degrade to the safe path — they never take a conversational turn down.
- **Construction knows config; behavior doesn't.** Models, tools, and prompts are assembled once at startup into objects the rest of the app calls. Tools close over their configuration and an injectable HTTP/DB client, so they can be unit-tested with fakes and never read global settings at call time.
- **Safety rails at the seam, not in prompts.** Per-turn wall-clock and recursion limits are enforced in code, so a confused tool loop hits a hard stop instead of burning budget. The authorization identity is per-call only and never checkpointed, which structurally prevents one user's identity from leaking onto another.
- **Testing stance.** Pure functions (identity mapping, message normalization, privacy gating, output splitting, media detection) are unit-tested offline with fakes; the live I/O shells are exercised on real hardware. The suite runs with no database, no network, and no real model calls, which keeps CI fast and deterministic. There is also an LLM-judge eval harness and a replay harness for end-to-end confidence.

---

## Status & roadmap

Personal project, **actively evolving.** The core is in place and exercised: the `ask()` seam, the checkpointed chat graph, model tiering, the full tool/action layer (home control, media, web search, timers), long-term memory retrieval, the short-term privacy gate, cross-surface person-keyed continuity, turn auditing, Phoenix/OTel tracing, the shadow-mode extraction worker, and the capability-gap miner. All three text/voice transports plus the HTTP door are implemented, and there is a Home Assistant custom component for the voice path.

Being built and refined incrementally, in public, as a learning-forward exercise — expect rough edges, opinionated internals, and design docs (`docs/design/`) and a numbered learning series (`docs/learning/`) that narrate *why* each piece exists. It is a single-operator system; some defaults and integrations are shaped around one home setup and would need generalizing for other environments.

---

## License

Licensed under the **MIT License** — see [`LICENSE`](LICENSE).
