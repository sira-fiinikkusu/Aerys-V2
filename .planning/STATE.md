# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-06-01)

**Core value:** The Brain orchestrates; the Hands govern — capability grows without weakening the safety boundary.
**Current focus:** Phase 1 — Orchestration Skeleton (ready to plan)

## Current Position

Phase: 1 of 12 (Orchestration Skeleton)
Plan: 0 of ~5 in current phase
Status: Phase 1 PLANNED. v2 PLAN complete (01-PLAN.md) — three-brain reviewed (Gemini + Codex), verified vs langgraph 1.2.6, reference-grounded; all 01-REVIEW.md blockers folded. Ready to EXECUTE (build 01-01 scaffold). S2 channel provisional `configurable` (single-accessor) — validate with work-AI in PARALLEL; non-blocking (bounded swap, cheapest pre-Phase-5).
Last activity: 2026-06-20 — drafted Phase 1 plan; Gemini + Codex reviews (Codex caught 3 real LangGraph-API blockers); verified vs langgraph 1.2.6 docs; cross-checked a reference implementation (confirmed most fixes, refuted Codex's configurable-blocker); gemini-cli auth fixed (api-key)

Progress: [░░░░░░░░░░] 0%

## Performance Metrics

**Velocity:**
- Total plans completed: 0
- Average duration: —
- Total execution time: 0 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| - | - | - | - |

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

### Pending Todos

None yet.

### Blockers/Concerns

- **None blocking.** Roadmap is cross-reviewed (two-brain convergence, build-first endorsed 3×) and polished to v3. Ready for `/gsd:plan-phase 1`.
- Open questions in PROJECT.md: soul.md public/private; transport HTTP vs gRPC; framework (leaning LangGraph); checkpointer durability (Phase 2 deliverable, leaning local SQLite + volume).

## Deferred Items

| Category | Item | Status | Deferred At |
|----------|------|--------|-------------|
| Voice | Aerys-Voice TS runtime | v2 (post-v1) | 2026-06-01 |
| Capability | GEPA self-evolution, MCP | v2 (post-v1) | 2026-06-01 |

## Session Continuity

Last session: 2026-06-20
Stopped at: Phase 1 v2 PLAN complete + reviewed + reference-grounded. Next: EXECUTE — build 01-01 (scaffold) hand-written by Chris, Kael reviews. In parallel: Chris validates the S2 `configurable` channel with work-AI (brief on NAS) — non-blocking.
Resume file: .planning/phases/01-orchestration-skeleton/01-PLAN.md (v2, ready to build)
