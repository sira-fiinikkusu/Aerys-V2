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

- [ ] Orchestration skeleton: a local LangGraph loop with a swappable model, persona injection, a durable Brain-owned checkpointer, a stream-shaped output-emit seam, and tracing from the first commit
- [ ] A typed, versioned Brain↔Hands contract (idempotency, rejection/compensation, trace propagation, golden fixtures) with a local mock — then proven against real n8n early on its semantics and concurrency, before orchestration is stacked on it
- [ ] Capabilities layered one per phase: research → identity → memory read → output → memory write (idempotent, compensating), with structural seams laid up front and an eval harness growing alongside
- [ ] Cutover safety: eval parity + rollback controls, then shadow mode, then a single canary path in production

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
- **Boundary**: NO shared business logic across runtimes. ONE canonical owner per concept. The Brain may *ask* Hands for a decision; it never caches or pre-resolves a Hands-owned policy (privacy, identity, memory). The "no direct database" rule scopes to **Hands-owned memory** (the pgvector store) — the Brain keeps its OWN checkpointer for conversation/orchestration state, which is not a boundary violation.
- **Structural seams**: invariants that are expensive to retrofit — the output-approval gate, injected caller-context, trace-context propagation, and failure/compensation — are laid at the skeleton/contract stage; later phases fill the policy behind them rather than inverting the architecture late.
- **Security / Privacy**: caller identity is *injected*, never a model-settable parameter (policy-by-architecture). Every state-changing Hands call carries an idempotency key. Streaming stays Brain-internal until Hands approves the response envelope.
- **Deployment**: runs on a Jetson Orin Nano (ARM). Reproducible, containerized deployment is a first-class concern, not an afterthought.
- **Public repo**: no secrets, no private infrastructure detail, ever. This is portfolio-visible by design.

## Key Decisions

<!-- Decisions that constrain future work. Add throughout project lifecycle. -->

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Build the Brain ourselves rather than adopt a framework wholesale | Every evaluated framework is "almost" — the agent's discriminators (persona, privacy semantics, cross-platform identity, tiered memory) aren't any framework's defaults; adoption means lock-in or a permanent fork tax | — Pending |
| Three-runtime split (Brain / Hands / Voice) with strict boundaries | Keep n8n's deterministic governance strengths; move orchestration to code; isolate latency-critical streaming to the runtime built for it | — Pending |
| Build-first, migrate-last phasing | Stand up the skeleton and layer capabilities incrementally (one per phase); bring shadow/canary cutover in once the Brain is feature-capable | ✓ Endorsed-with-changes (cross-review 2026-06-01) |
| Integrate real n8n early (Phase 3), not just a mock | Mocking the boundary away hides the exact n8n physics (cold starts, latency, partial failure) the rebuild exists to escape; prove the contract against real n8n before stacking orchestration | ✓ Adopted from cross-review |
| Lay structural seams up front, fill policy later | Output gate (stream-shaped), injected context, trace propagation, and compensation are invariants; retrofitting them inverts the architecture late and forces rewrites | ✓ Adopted from cross-review |
| Output before memory-write; durable compensation | Memory-write compensation must be conditioned on a real output status, with retryable (not synchronous) undo, to close the two-generals desync without trading it for the inverse | ✓ Adopted from 2nd review |
| Eval parity before shadow; harness grows from the first tool | Rebuilding a working agent demands continuous parity measurement — not a first comparison at shadow, eight phases in | ✓ Adopted from 2nd review |
| Concentrate production rigor at the state-changing & cutover phases | Full sagas/reconciliation/load-testing belong where stakes are real (memory-write, shadow, canary), not gold-plated across the skeleton — this is a solo build with the old agent still in production | ✓ Adopted from 2nd review |
| Public Brain repo + private Hands infra | The architectural boundary doubles as a clean public/private seam — contract + mock are public; the credential-bound implementation stays private | — Pending |

## Open Questions

- **soul.md** (the agent's persona file): ship a public *template* and keep the real persona private, or publish it? (`.gitignore` toggle currently pending a decision)
- **Transport**: HTTP/JSON vs gRPC+Protobuf for Brain↔Hands — decide when a real latency number demands it, not before
- **Brain framework**: LangGraph as the primary orchestration layer vs a thinner FastAPI/anyio core — leaning LangGraph
- **Memory interface**: confirm the Brain only ever reaches memory through Hands endpoints (no direct Postgres access) at contract-design time
- **Checkpointer durability** — now a Phase 2 deliverable (leaning local SQLite + mounted volume for restart-durability), with a declared Brain↔Hands reconciliation rule (leaning: Hands authoritative on conflict)

---
*Last updated: 2026-06-01 — roadmap v3, polished after a second cross-architecture review (build-first endorsed across three independent passes). See Key Decisions.*
