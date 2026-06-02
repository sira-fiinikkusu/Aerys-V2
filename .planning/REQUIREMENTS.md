# Requirements: Aerys-V2

**Defined:** 2026-06-01
**Core Value:** The Brain orchestrates; the Hands govern — capability grows without weakening the safety boundary.

## v1 Requirements

Requirements for the initial build. Each maps to a roadmap phase.

### Orchestration (Brain skeleton)

- [ ] **ORCH-01**: A LangGraph agent loop runs locally on the Jetson and returns a response to a text input
- [ ] **ORCH-02**: The model provider is swappable via config (no code change to change models)
- [ ] **ORCH-03**: The agent loads `soul.md` and injects the persona into the system prompt at runtime
- [ ] **ORCH-04**: Conversation state is maintained across turns within a session
- [ ] **ORCH-05**: Every run emits an OpenTelemetry trace from the first commit
- [ ] **ORCH-06**: All user-visible output routes through a single **stream-shaped** emit seam (async/chunked gate; the future "Hands approves the envelope" point) — never a synchronous pass-through
- [ ] **ORCH-07**: The container builds and boots reproducibly on ARM/Jetson; dependencies are pinned and a secrets-loading check passes

### State, Durability & Recovery

- [ ] **STATE-01**: The Brain-owned checkpointer has a decided durability story (e.g., local SQLite + mounted volume) that survives container restarts — distinct from Hands-owned memory
- [ ] **STATE-02**: A turn interrupted by a crash/kill resumes cleanly from the checkpointer
- [ ] **STATE-03**: The Brain↔Hands reconciliation rule is documented — Hands state is authoritative on conflict; the Brain reconciles to it on recovery (enforced at the memory phases)

### Hands Contract (boundary)

- [ ] **HANDS-01**: A typed, **versioned** capability contract defines identity, memory-read, memory-write, and output-approve/send operations
- [ ] **HANDS-02**: Each capability documents idempotency, auth, timeout, and privacy semantics ("Brain may ask, Hands decides")
- [ ] **HANDS-03**: A local mock Hands server lets the Brain exercise the contract (including a stream-capable approve-envelope) with no real infrastructure
- [ ] **HANDS-04**: Contract tests cover happy-path, duplicate-request, and rejection/compensation cases for every capability
- [ ] **HANDS-05**: The boundary doc records one canonical owner per concept; the Brain holds no Hands-owned credentials
- [ ] **HANDS-06**: Each capability defines explicit failure/rejection semantics ("Hands decides NO") and compensation behavior
- [ ] **HANDS-07**: The contract propagates W3C `traceparent` on every call and defines an error taxonomy + idempotency-key lifetime
- [ ] **HANDS-08**: Golden request/response fixtures exist and a compatibility test fails on an unversioned breaking change (public Brain / private Hands cannot silently drift)

### Transport Validation (real-n8n smoke)

- [ ] **TRAN-01**: The Brain completes a real round-trip against a bare n8n Hands workflow over the contract (not the mock)
- [ ] **TRAN-02**: Real-n8n semantics are exercised — a rejection, a timeout, a duplicate idempotency key, and a partial failure each behave per the contract
- [ ] **TRAN-03**: The bare workflow is hit with 10+ concurrent requests; async-to-webhook physics (queuing, thread-starvation) are characterized
- [ ] **TRAN-04**: A transport latency baseline is recorded; trace spans connect across the boundary (one trace, both sides)

### Research Tool (first capability)

- [ ] **TOOL-01**: The Brain runs a LangGraph tool loop (ToolNode + tools_condition)
- [ ] **TOOL-02**: A read-only research/summarization tool answers a question with no side effects
- [ ] **TOOL-03**: A slow tool can be cancelled/timed out without wedging the agent loop (heavy work runs off the main loop)
- [ ] **TOOL-04**: Tools receive caller context via injected state from the first tool (the injected-context seam), not as model-settable arguments
- [ ] **TOOL-05**: An eval harness exists and runs against the Brain; it grows by at least one case each subsequent phase

### Identity (policy-by-architecture)

- [ ] **IDENT-01**: The Brain resolves the current person by asking Hands
- [ ] **IDENT-02**: Caller identity arrives via the injected-context seam, never exposed as a model-settable parameter
- [ ] **IDENT-03**: A red-team test confirms the model cannot target an identity other than the caller, even under social-engineering prompts

### Memory — Read

- [ ] **MEMR-01**: The Brain retrieves relevant memory for the current person through a Hands call
- [ ] **MEMR-02**: The Brain has no direct database connection or credentials

### Output (privacy + streaming safety)

- [ ] **OUT-01**: User-visible output flows through the Hands Output Router, behind the Phase-1 stream-shaped emit seam
- [ ] **OUT-02**: PII scrubbing and conversation-privacy are enforced by Hands, not the Brain
- [ ] **OUT-03**: Streaming stays Brain-internal until Hands approves the response envelope

### Memory — Write (state-changing)

- [ ] **MEMW-01**: The Brain proposes a memory write; Hands performs it (subject to Hands-owned consolidation rules)
- [ ] **MEMW-02**: Every memory write carries an idempotency key; a duplicate-write test confirms exactly-once persistence
- [ ] **MEMW-03**: Writes move through an explicit state machine (proposed → staged → committed → compensated) tied to the Phase-8 output status; the Brain never writes to the database directly
- [ ] **MEMW-04**: A compensation that itself fails is durably retried (local queue/log) — no orphaned memory of an unseen conversation, and no silently-dropped write after a seen one
- [ ] **MEMW-05**: After a crash mid-write, recovery reconciles to Hands per the STATE-03 rule

### Eval Parity + Cutover Controls

- [ ] **EVAL-01**: The predecessor eval suite runs against Brain output
- [ ] **EVAL-02**: The Brain demonstrates parity (or better) on a representative case set
- [ ] **EVAL-03**: Rollback / kill-switch, proposal-only guarantees, and audit logging are designed and unit-proven (before any production traffic)

### Cutover — Shadow Mode

- [ ] **SHAD-01**: Production inputs are mirrored to the Brain; the Brain proposes but does not execute
- [ ] **SHAD-02**: Decisions, latency (vs the Phase 4 baseline), token cost, and eval scores are logged side-by-side with the production agent
- [ ] **SHAD-03**: Any Brain-side failure is diagnosable within 10 minutes from logs/traces alone (operator-diagnose test)
- [ ] **SHAD-04**: A distributed-system eval suite covers privacy red-team, duplicate sends, webhook replay, concurrent conversations, and stale identity

### Cutover — Canary

- [ ] **CAN-01**: One low-risk path (read-only research) runs Brain-driven in production
- [ ] **CAN-02**: The path is continuously evaluated against the regression suite
- [ ] **CAN-03**: The Phase-10 rollback switch returns the path to the production agent instantly
- [ ] **CAN-04**: The maintainer spends less time fighting infrastructure and more on agent behavior (the "maintainer metric")

## v2 Requirements

Deferred to a future release. Tracked, not in the current roadmap.

### Voice

- **VOICE-01**: A TypeScript streaming runtime serves sub-4s round-trip voice
- **VOICE-02**: Runtime choice is justified against the latency budget baselined in Phase 4

### Self-Evolution

- **EVOL-01**: A reflection/optimization loop improves prompts or persona against eval feedback

### Extensibility

- **MCP-01**: Scoped, sandboxed MCP servers extend the tool surface with no credential pass-through

### Earned Migration

- **MIGR-01**: Remaining production paths migrate to Brain one at a time, each with its own canary period

## Out of Scope

Explicitly excluded. Documented to prevent scope creep.

| Feature | Reason |
|---------|--------|
| n8n Hands workflow implementation | Lives in a separate private infra repo; this repo defines/consumes the contract only (Phase 4 uses a bare smoke workflow, not the production Hands) |
| Real credentials / production DB / production persona | Never committed; injected at runtime |
| Full rewrite of working n8n adapters (Discord/Telegram/voice) | They work; they stay as Hands components until a benchmark justifies otherwise |
| Aerys-Voice TS runtime (v1) | Deferred to v2 pending latency budget |
| GEPA self-evolution & MCP (v1) | Expand the trusted boundary; deferred until the boundary is hardened |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| ORCH-01..07 | Phase 1 | Pending |
| STATE-01..03 | Phase 2 | Pending |
| HANDS-01..08 | Phase 3 | Pending |
| TRAN-01..04 | Phase 4 | Pending |
| TOOL-01..05 | Phase 5 | Pending |
| IDENT-01..03 | Phase 6 | Pending |
| MEMR-01..02 | Phase 7 | Pending |
| OUT-01..03 | Phase 8 | Pending |
| MEMW-01..05 | Phase 9 | Pending |
| EVAL-01..03 | Phase 10 | Pending |
| SHAD-01..04 | Phase 11 | Pending |
| CAN-01..04 | Phase 12 | Pending |

**Coverage:**
- v1 requirements: 51 total
- Mapped to phases: 51
- Unmapped: 0 ✓

---
*Requirements defined: 2026-06-01*
*Last updated: 2026-06-01 — v3 after the second cross-architecture review (stream-seam, state/durability/recovery, contract versioning, real-n8n semantics+concurrency, output-before-memory-write reorder, durable compensation, eval-parity + cutover controls)*
