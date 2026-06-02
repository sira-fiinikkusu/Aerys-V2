# Aerys-V2

## What This Is

Aerys-V2 is an open-source, self-hosted personal AI agent built around a strict three-runtime
architecture: a Python/LangGraph **Brain** that orchestrates reasoning and tools, an n8n **Hands**
layer that owns every identity, memory, output, and credential-touching operation behind a typed
contract, and a future TypeScript **Voice** runtime for low-latency streaming. It is a ground-up
rebuild of an agent previously implemented entirely in n8n — moving orchestration into code while
keeping the deterministic, credential-bound work in a governed layer.

## Core Value

**The Brain orchestrates; the Hands govern.** Reasoning lives in swappable, inspectable code, while
every identity / memory / privacy / credential decision has exactly one canonical owner (Hands) — so
capability can grow without ever weakening the safety boundary.

## Requirements

### Validated

<!-- Shipped and confirmed valuable. -->

(None yet — greenfield)

### Active

<!-- Current scope. Building toward these. -->

- [ ] Orchestration skeleton: a local LangGraph agent loop with a swappable model, persona injection, and tracing from the first commit
- [ ] A typed Brain↔Hands capability contract with a local mock, so the boundary is testable before any real infrastructure exists
- [ ] Capabilities layered one per phase: read-only research tool → identity → memory read → memory write → output routing
- [ ] Cutover safety: shadow mode, then a single canary path in production, with rollback always available

### Out of Scope

<!-- Explicit boundaries. Includes reasoning to prevent re-adding. -->

- The n8n Hands workflow implementation itself — lives in a separate private infrastructure repo; this repo *defines and consumes* the contract, it does not implement Hands
- Real credentials, production database, and production persona content — never committed; injected at runtime via environment
- Aerys-Voice (TypeScript streaming runtime) — deferred until a latency benchmark justifies a separate runtime (post-v1)
- Self-evolution (GEPA-style optimization) and MCP integration — powerful, but both expand the trusted computing base; deferred to post-v1 once the boundary is hardened

## Context

- **Predecessor**: a fully working personal agent implemented across ~27 n8n workflows — Discord/Telegram/voice adapters, three-tier memory with pgvector, cross-platform identity resolution, a model-tiering router, and a regression eval suite. It works. The friction is that n8n is a *workflow engine* doing an *agent runtime's* job: every new agentic capability is friction-priced (tool-count ceilings, context-passing quirks, no real streaming, per-tier sub-workflow workarounds).
- **The thesis**: keep the parts n8n is genuinely good at (deterministic, credential-bound governance = "Hands") and move orchestration into a language ecosystem built for it (Python/LangGraph = "Brain"). The win is not "n8n is bad" — it's "orchestration belongs in code; governance belongs behind a contract."
- The architecture was developed and cross-reviewed across multiple model architectures before scoping; this roadmap is the build-order expression of it.

## Constraints

- **Tech stack**: Python + LangGraph for the Brain; n8n (existing) for Hands; TypeScript later for Voice. Cross-runtime calls over an explicit typed contract — HTTP/JSON to start, gRPC evaluated only if latency demands it.
- **Boundary**: NO shared business logic across runtimes. ONE canonical owner per concept. The Brain may *ask* Hands for a decision; it never caches or pre-resolves a Hands-owned policy (privacy, identity, memory).
- **Security / Privacy**: caller identity is *injected*, never a model-settable parameter (policy-by-architecture). Every state-changing Hands call carries an idempotency key. Streaming stays Brain-internal until Hands approves the response envelope.
- **Deployment**: runs on a Jetson Orin Nano (ARM). Reproducible, containerized deployment is a first-class concern, not an afterthought.
- **Public repo**: no secrets, no private infrastructure detail, ever. This is portfolio-visible by design.

## Key Decisions

<!-- Decisions that constrain future work. Add throughout project lifecycle. -->

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Build the Brain ourselves rather than adopt a framework wholesale | Every evaluated framework is "almost" — the agent's discriminators (persona, privacy semantics, cross-platform identity, tiered memory) aren't any framework's defaults; adoption means lock-in or a permanent fork tax | — Pending |
| Three-runtime split (Brain / Hands / Voice) with strict boundaries | Keep n8n's deterministic governance strengths; move orchestration to code; isolate latency-critical streaming to the runtime built for it | — Pending |
| Build-first, migrate-last phasing | Stand up the skeleton and layer capabilities incrementally (one per phase); bring shadow/canary cutover in once the Brain is feature-capable | — Pending (under cross-review) |
| Public Brain repo + private Hands infra | The architectural boundary doubles as a clean public/private seam — contract + mock are public; the credential-bound implementation stays private | — Pending |

## Open Questions

- **soul.md** (the agent's persona file): ship a public *template* and keep the real persona private, or publish it? (`.gitignore` toggle currently pending a decision)
- **Transport**: HTTP/JSON vs gRPC+Protobuf for Brain↔Hands — decide when a real latency number demands it, not before
- **Brain framework**: LangGraph as the primary orchestration layer vs a thinner FastAPI/anyio core — leaning LangGraph
- **Memory interface**: confirm the Brain only ever reaches memory through Hands endpoints (no direct Postgres access) at contract-design time

---
*Last updated: 2026-06-01 after initial scoping (DRAFT — phase breakdown pending cross-architecture review)*
