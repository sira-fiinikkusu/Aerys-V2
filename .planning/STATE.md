# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-06-01)

**Core value:** The Brain orchestrates; the Hands govern — capability grows without weakening the safety boundary.
**Current focus:** Phase 1 — Orchestration Skeleton (EXECUTING — 01-01 done, 01-02 next)

## Current Position

Phase: 1 of 12 (Orchestration Skeleton)
Plan: 1 of ~5 in current phase (01-01 COMPLETE)
Status: **01-01 (Scaffold + reproducible ARM build) COMPLETE** — hand-written by Chris, reviewed by Kael. uv project (Python 3.11 pinned), pinned deps + committed `uv.lock` (langgraph 1.2.6, langchain-anthropic, pydantic-settings, opentelemetry), multi-stage `linux/arm64` Dockerfile + non-root + HEALTHCHECK (native build verified), `Settings` (pydantic-settings) with fail-fast secrets gate, deployable entrypoint (`cli.py`: config-load → fail-fast → SIGTERM/SIGINT graceful shutdown via threading.Event → `--health` probe), Makefile, pytest skeleton (2 tests green), GitHub Actions CI **green**. Next: **01-02** (graph + state + swappable model). **S2 channel RESOLVED → Option A (`configurable` + single accessor)** — external validation complete 2026-06-22; A is proven-not-future-proof (`context`/`Runtime` is LangGraph's long-term direction, `config_schema` declaration deprecated/removal v2.0, but we don't declare one and `thread_id` rides `configurable` permanently). Single-accessor is the hedge for a bounded later swap.
Last activity: 2026-06-21 — built 01-01 hands-on (Chris hand-writes = learning tier; Kael reviews + SSH-verifies on Leviathan). Review caught real bugs Copilot introduced (`"none"` vs `None`; module-level-vs-`main()` scope; `@v8` non-existent moving tag). Final commit `7219cd1`, CI green + warning-free.

Progress: [██░░░░░░░░] ~20% (1 of ~5 plans)

## Performance Metrics

**Velocity:**
- Total plans completed: 1
- Average duration: ~1 session
- Total execution time: ~1 session (2026-06-21)

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 1 | 1 of ~5 | 1 session | ~1 session |

## Accumulated Context

### Decisions

Logged in PROJECT.md Key Decisions. Recent:

- Build the Brain ourselves; three-runtime split (Brain / Hands / Voice)
- **Build-first, integrate-early, migrate-last** — ✓ endorsed across three independent cross-reviews
- Integrate real n8n early (Phase 4) on semantics + concurrency, not just transport
- Structural seams up front (stream-shaped output gate, injected context, trace propagation)
- Output before memory-write; durable compensation; eval parity before shadow
- Production rigor (sagas/reconciliation/load-tests) concentrated at the state-changing & cutover phases
- Public Brain repo + private Hands infra
- **Framework: LangGraph — COMMITTED** (01-01 built on it; was "leaning")
- **01-01 build sequencing:** app code (Settings + entrypoint) BEFORE the Dockerfile, so the container wraps a real process with a real `--health` probe (vs a stub then rewrite)
- **Tracing backend = Arize Phoenix** (provisional; wires at 01-05) — OTLP exporter already in deps, so adopting it is config-not-code; runs self-hosted on the Jetson/NAS
- **Working mode:** Chris hand-writes all code (learning tier); Kael gives spec + building-blocks then reviews right/wrong + why over SSH. NOT gsd-executor.
- **S2 caller-context channel = `configurable` + single accessor — RESOLVED** (external validation 2026-06-22): proven, meets all 3 invariants, bounded-swappable to `context_schema` later via the single-accessor hedge. Amendment: A is proven *not future-proof* — `context` is the framework's long-term direction; single-accessor keeps a future move cheap.
- **Two-identity split surfaced (2026-06-22):** authorization (S2, *who an action runs for*) and attribution (*who said what*, a `Speaker (verified)` transcript label) are two separate jobs. Attribution was a footnote → elevated to a named Phase-6 deliverable (06-04 + success criterion #4).

### Pending Todos

- **Phase 6 attribution deliverable (06-04)** — implement the `Speaker (verified) / Message (untrusted, verbatim)` turn format + a shared-thread multi-speaker test (the attribution half of the two-identity split). Roadmap + 01-PLAN updated 2026-06-22; lands when Phase 6 is planned.

### Blockers/Concerns

- **None blocking.** 01-01 shipped green. S2 channel **RESOLVED → Option A** (`configurable` + single accessor; external validation complete 2026-06-22). Bounded swap to `context_schema` remains available pre-Phase-5 via the single-accessor hedge.
- Open questions in PROJECT.md: soul.md public/private (RESOLVED in plan — public `soul.example.md` + gitignored real; lands at 01-03); transport HTTP vs gRPC (Phase 3/4); checkpointer durability (Phase 2, leaning local SQLite + volume).

## Deferred Items

| Category | Item | Status | Deferred At |
|----------|------|--------|-------------|
| Voice | Aerys-Voice TS runtime | v2 (post-v1) | 2026-06-01 |
| Capability | GEPA self-evolution, MCP | v2 (post-v1) | 2026-06-01 |

## Session Continuity

Last session: 2026-06-21
Stopped at: **01-01 COMPLETE** — scaffold + containerized + tested + CI green (`7219cd1` on origin). All 7 sub-steps done.
Next: **01-02** — LangGraph chat graph + `ChatState` (messages-only, `add_messages`) + single `chat` node + injected `InMemorySaver` checkpointer + `build_model` factory + caller-supplied `session_id`. Built ONCE with persona (01-03) + stream entry (01-04) stubbed so they FILL not rewrite.
Resume file: .planning/phases/01-orchestration-skeleton/01-PLAN.md (Plan 01-02 section)
Working copy: Chris codes on **Leviathan** (`~/projects/aerys-v2`); Kael reviews over SSH. S2 external validation **COMPLETE 2026-06-22 → Option A confirmed** (no longer a parallel open item).
