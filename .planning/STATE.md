# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-06-01)

**Core value:** The Brain orchestrates; the Hands govern — capability grows without weakening the safety boundary.
**Current focus:** Phase 1 — Orchestration Skeleton (ready to plan)

## Current Position

Phase: 1 of 12 (Orchestration Skeleton)
Plan: 0 of ~5 in current phase
Status: Roadmap v3 complete and polished (build-first endorsed across three independent cross-architecture passes). Ready to plan Phase 1.
Last activity: 2026-06-01 — second cross-review folded in; roadmap 10→12 phases (Phase 1 split, output reordered ahead of memory-write, eval-parity + cutover controls added, stream-seam / contract versioning / real-n8n semantics+concurrency / durable compensation)

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

Last session: 2026-06-01
Stopped at: Roadmap v3 written + second cross-review folded in (PROJECT / REQUIREMENTS / ROADMAP / STATE). Next: `/gsd:plan-phase 1`.
Resume file: None
