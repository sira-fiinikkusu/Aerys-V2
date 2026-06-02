# Requirements: Aerys-V2

**Defined:** 2026-06-01
**Core Value:** The Brain orchestrates; the Hands govern — capability grows without weakening the safety boundary.

## v1 Requirements

Requirements for the initial build. Each maps to a roadmap phase.

### Orchestration (Brain skeleton)

- [ ] **ORCH-01**: A LangGraph agent loop runs locally on the Jetson and returns a response to a text input
- [ ] **ORCH-02**: The model provider is swappable via config (no code change to change models)
- [ ] **ORCH-03**: The agent loads `soul.md` and injects the persona into the system prompt at runtime
- [ ] **ORCH-04**: Conversation state is maintained across turns via a Brain-owned checkpointer (distinct from Hands-owned memory); the store and its Jetson durability are a recorded decision
- [ ] **ORCH-05**: Every run emits an OpenTelemetry trace from the first commit
- [ ] **ORCH-06**: All user-visible output routes through a single emit seam — the structural gate that later becomes "Hands approves the envelope" (pass-through until Hands exists)
- [ ] **ORCH-07**: The "Brain has no direct database" rule is scoped explicitly to Hands-owned memory; the Brain's own checkpointer is not a violation

### Hands Contract (boundary)

- [ ] **HANDS-01**: A typed capability contract defines identity, memory-read, memory-write, and output-approve/send operations
- [ ] **HANDS-02**: Each capability documents idempotency, auth, timeout, and privacy semantics ("Brain may ask, Hands decides")
- [ ] **HANDS-03**: A local mock Hands server lets the Brain exercise the contract with no real infrastructure
- [ ] **HANDS-04**: Contract tests cover happy-path, duplicate-request, and rejection/compensation cases for every capability
- [ ] **HANDS-05**: The boundary doc records one canonical owner per concept; the Brain holds no Hands-owned credentials
- [ ] **HANDS-06**: Each capability defines explicit failure/rejection semantics ("Hands decides NO") and compensation behavior
- [ ] **HANDS-07**: The contract propagates W3C `traceparent` trace context on every Brain↔Hands call

### Transport Validation (real-n8n smoke)

- [ ] **TRAN-01**: The Brain completes a real round-trip against a bare n8n Hands workflow over the contract (not the mock)
- [ ] **TRAN-02**: Transport physics are measured — serialization overhead, timeout behavior, cold-start, p50/p95 latency
- [ ] **TRAN-03**: An end-to-end Jetson→Brain→n8n→Brain latency baseline is recorded; trace spans connect across the boundary (one trace, both sides)

### Research Tool (first capability)

- [ ] **TOOL-01**: The Brain runs a LangGraph tool loop (ToolNode + tools_condition)
- [ ] **TOOL-02**: A read-only research/summarization tool answers a question with no side effects
- [ ] **TOOL-03**: A slow tool can be cancelled/timed out without wedging the agent loop (heavy work runs off the main loop)
- [ ] **TOOL-04**: Tools receive caller context via injected state from the first tool (the injected-context seam), not as model-settable arguments

### Identity (policy-by-architecture)

- [ ] **IDENT-01**: The Brain resolves the current person by asking Hands
- [ ] **IDENT-02**: Caller identity arrives via the injected-context seam, never exposed as a model-settable parameter
- [ ] **IDENT-03**: A red-team test confirms the model cannot target an identity other than the caller, even under social-engineering prompts

### Memory — Read

- [ ] **MEMR-01**: The Brain retrieves relevant memory for the current person through a Hands call
- [ ] **MEMR-02**: The Brain has no direct database connection or credentials

### Memory — Write

- [ ] **MEMW-01**: The Brain proposes a memory write; Hands performs it (subject to Hands-owned consolidation rules)
- [ ] **MEMW-02**: Every memory write carries an idempotency key
- [ ] **MEMW-03**: A duplicate-write contract test confirms exactly-once persistence
- [ ] **MEMW-04**: A memory write is provisional/compensable until the corresponding output succeeds; an aborted output leaves no orphaned memory (two-generals fix)

### Output (privacy + streaming safety)

- [ ] **OUT-01**: User-visible output flows through the Hands Output Router, behind the Phase-1 emit seam
- [ ] **OUT-02**: PII scrubbing and conversation-privacy are enforced by Hands, not the Brain
- [ ] **OUT-03**: Streaming stays Brain-internal until Hands approves the response envelope

### Cutover — Shadow Mode

- [ ] **SHAD-01**: Production inputs are mirrored to the Brain; the Brain proposes but does not execute
- [ ] **SHAD-02**: Decisions, latency (vs the Phase 3 baseline), token cost, and eval scores are logged side-by-side with the production agent
- [ ] **SHAD-03**: Any Brain-side failure is diagnosable within 10 minutes from logs/traces alone (operator-diagnose test)
- [ ] **SHAD-04**: A distributed-system eval suite covers privacy red-team, duplicate sends, webhook replay, concurrent conversations, and stale identity

### Cutover — Canary

- [ ] **CAN-01**: One low-risk path (read-only research) runs Brain-driven in production
- [ ] **CAN-02**: The path is continuously evaluated against the regression suite
- [ ] **CAN-03**: A rollback switch returns the path to the production agent instantly
- [ ] **CAN-04**: The maintainer spends less time fighting infrastructure and more on agent behavior (the "maintainer metric")

## v2 Requirements

Deferred to a future release. Tracked, not in the current roadmap.

### Voice

- **VOICE-01**: A TypeScript streaming runtime serves sub-4s round-trip voice
- **VOICE-02**: Runtime choice is justified against the latency budget baselined in Phase 3

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
| n8n Hands workflow implementation | Lives in a separate private infra repo; this repo defines/consumes the contract only (Phase 3 uses a bare smoke workflow, not the production Hands) |
| Real credentials / production DB / production persona | Never committed; injected at runtime |
| Full rewrite of working n8n adapters (Discord/Telegram/voice) | They work; they stay as Hands components until a benchmark justifies otherwise |
| Aerys-Voice TS runtime (v1) | Deferred to v2 pending latency budget |
| GEPA self-evolution & MCP (v1) | Expand the trusted boundary; deferred until the boundary is hardened |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| ORCH-01..07 | Phase 1 | Pending |
| HANDS-01..07 | Phase 2 | Pending |
| TRAN-01..03 | Phase 3 | Pending |
| TOOL-01..04 | Phase 4 | Pending |
| IDENT-01..03 | Phase 5 | Pending |
| MEMR-01..02 | Phase 6 | Pending |
| MEMW-01..04 | Phase 7 | Pending |
| OUT-01..03 | Phase 8 | Pending |
| SHAD-01..04 | Phase 9 | Pending |
| CAN-01..04 | Phase 10 | Pending |

**Coverage:**
- v1 requirements: 36 total
- Mapped to phases: 36
- Unmapped: 0 ✓

---
*Requirements defined: 2026-06-01*
*Last updated: 2026-06-01 — v2 after cross-architecture review (seams, transport smoke, compensation, trace propagation, checkpointer scoping added)*
