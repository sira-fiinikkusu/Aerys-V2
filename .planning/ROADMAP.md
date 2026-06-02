# Roadmap: Aerys-V2

## Overview

A ground-up build of the **Brain** (Python/LangGraph) for a personal AI agent, growing one capability
per phase against a typed contract to the **Hands** (n8n) governance layer. The build is *build-first,
integrate-early, migrate-last*: stand up a streaming-capable orchestration skeleton, make the Brain's
own state durable and recoverable, define and **version** the Brain↔Hands contract, then **prove that
contract against real n8n early** — exercising the quirks (rejection, timeout, duplicate, partial
failure, concurrency) a mock would hide. Capabilities then layer one per phase — research, identity,
memory-read, output, memory-write — with the structural seams (streaming output gate, injected
caller-context, trace propagation, compensation) laid up front so policy fills in behind them with no
late rewrites. An evaluation harness grows from the first tool, and an explicit parity check gates
**cutover**: shadow mode, then a single canary path with rollback. Production-grade distributed rigor
(durable sagas, reconciliation, load testing) is concentrated at the state-changing and cutover phases
rather than gold-plating the skeleton.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): planned milestone work
- Decimal phases (2.1, 2.2): urgent insertions (marked INSERTED)

- [ ] **Phase 1: Orchestration Skeleton** - Streaming-capable LangGraph loop, persona, swappable model, the stream-shaped emit seam, tracing, reproducible ARM build
- [ ] **Phase 2: State, Durability & Recovery** - A Brain-owned checkpointer with a decided Jetson durability story, crash recovery, and the Brain↔Hands reconciliation rule
- [ ] **Phase 3: Hands Contract + Local Mock** - Typed, **versioned** boundary with idempotency, rejection/compensation, and trace-context semantics; golden fixtures; local mock
- [ ] **Phase 4: Real-n8n Transport, Semantics & Concurrency Smoke** - Prove the contract against real n8n: rejection, timeout, duplicate, partial failure, concurrency, latency baseline
- [ ] **Phase 5: Read-Only Research Tool** - The tool loop with the injected caller-context seam; a side-effect-free tool; the eval harness is stood up here and grows each phase
- [ ] **Phase 6: Identity via Hands** - Caller identity resolved by Hands and injected through the seam, never model-settable (policy-by-architecture)
- [ ] **Phase 7: Memory Read via Hands** - The Brain recalls per-person memory through Hands, with no direct database access
- [ ] **Phase 8: Output via Hands** - Fill the real PII / privacy / streaming policy behind the emit seam (before memory-write, so writes can be tied to output status)
- [ ] **Phase 9: Memory Write via Hands + Compensation** - Idempotent writes with a durable compensation state-machine tied to output status; reconciliation enforced
- [ ] **Phase 10: Eval Parity + Cutover Controls** - Wire the predecessor eval suite, prove parity, and design the rollback / kill-switch / proposal-only controls
- [ ] **Phase 11: Shadow Mode** - The Brain runs alongside production, proposes decisions, executes nothing — parity and operability proven before cutover
- [ ] **Phase 12: Single Canary Path** - One low-risk path goes Brain-driven in production, rollback proven

## Phase Details

### Phase 1: Orchestration Skeleton
**Goal**: A LangGraph Brain runs locally on the Jetson — loads `soul.md`, routes through a swappable model, holds conversation state, and routes ALL user-visible output through a single **stream-shaped emit seam** (an async/chunk-capable gate, not a synchronous pass-through — so the streaming safety model never has to be retrofitted). Tracing is wired from the first commit, and the build is reproducible on ARM.
**Depends on**: Nothing (first phase)
**Requirements**: ORCH-01, ORCH-02, ORCH-03, ORCH-04, ORCH-05, ORCH-06, ORCH-07
**Success Criteria** (what must be TRUE):
  1. A text message returns a persona-shaped reply on the Jetson
  2. The model provider is swappable via config with no code edit
  3. `soul.md` visibly shapes the response
  4. Conversation state persists across turns within a session
  5. All user-visible output flows through a single stream-capable emit seam (chunked/async, mock-approved for now)
  6. Every run emits an OpenTelemetry trace
  7. The container builds and boots reproducibly on ARM/Jetson; dependencies are pinned and a secrets-loading check passes
**Plans**: TBD

Plans:
- [ ] 01-01: Python scaffold (uv, layout, pinned deps, reproducible ARM/container build, secrets-loading test, cold-start boot)
- [ ] 01-02: LangGraph chat graph + conversation state + swappable model provider
- [ ] 01-03: `soul.md` persona injection + structured config
- [ ] 01-04: Stream-shaped emit seam (async/chunked gate all output routes through)
- [ ] 01-05: OpenTelemetry instrumentation + smoke test

### Phase 2: State, Durability & Recovery
**Goal**: Make the Brain's OWN orchestration state durable and recoverable BEFORE any state-changing capability exists. The Brain keeps its own LangGraph checkpointer (distinct from Hands-owned memory); this phase decides its store and Jetson durability, validates crash recovery, and declares the Brain↔Hands reconciliation rule (Hands state is authoritative on conflict; enforced once Hands state exists).
**Depends on**: Phase 1
**Requirements**: STATE-01, STATE-02, STATE-03
**Success Criteria** (what must be TRUE):
  1. The Brain-owned checkpointer has a decided durability story (e.g., local SQLite + mounted volume) that survives container restarts
  2. A turn interrupted by a crash/kill resumes cleanly from the checkpointer
  3. The Brain↔Hands reconciliation rule is documented (Hands authoritative; the Brain reconciles to it on recovery)
**Plans**: TBD

Plans:
- [ ] 02-01: Checkpointer store + Jetson durability decision and implementation
- [ ] 02-02: Crash-recovery validation (kill mid-turn → clean resume)
- [ ] 02-03: Reconciliation rule documented (enforced at the memory phases)

### Phase 3: Hands Contract + Local Mock
**Goal**: Define the typed, **versioned** Brain↔Hands capability boundary and a local mock. Per capability: idempotency, auth, timeout, privacy, AND failure/rejection + compensation semantics. The contract propagates W3C `traceparent` on every call, carries an explicit error taxonomy and idempotency-key lifetime, and ships golden request/response fixtures so the public Brain and private Hands cannot silently drift. The Phase-1 emit seam routes through a stream-capable approve-envelope capability (mock rubber-stamps for now).
**Depends on**: Phase 2
**Requirements**: HANDS-01, HANDS-02, HANDS-03, HANDS-04, HANDS-05, HANDS-06, HANDS-07, HANDS-08
**Success Criteria** (what must be TRUE):
  1. The contract defines identity, memory-read, memory-write, and output-approve/send with typed, **versioned** schemas
  2. Each capability documents idempotency, auth, timeout, privacy, rejection, and compensation semantics
  3. The contract propagates W3C `traceparent` on every call and defines an error taxonomy + idempotency-key lifetime
  4. Golden request/response fixtures exist; a compatibility test fails on unversioned breaking change
  5. A local mock Hands answers all calls (including a stream-capable approve-envelope) with no real infra
  6. Contract tests pass for happy-path, duplicate-request, and rejection/compensation paths
  7. The boundary doc names one canonical owner per concept; the Brain holds no Hands-owned credentials
**Plans**: TBD

Plans:
- [ ] 03-01: Contract definition — versioned schemas + idempotency/auth/timeout/privacy + rejection/compensation + error taxonomy (transport HTTP/JSON)
- [ ] 03-02: Trace-context propagation (`traceparent` on every call)
- [ ] 03-03: Golden fixtures + compatibility test; typed client; wire emit seam to stream-capable approve-envelope
- [ ] 03-04: Local mock Hands server (deterministic fixtures, including rejection cases)
- [ ] 03-05: Contract test suite (happy + duplicate + rejection/compensation) + boundary ownership doc

### Phase 4: Real-n8n Transport, Semantics & Concurrency Smoke
**Goal**: Prove the contract survives contact with real n8n BEFORE building capabilities on a clean mock — exercising the very behaviors a mock hides: rejection, timeout, duplicate idempotency keys, partial failure, and **concurrency**. Record the transport latency baseline.
**Depends on**: Phase 3
**Requirements**: TRAN-01, TRAN-02, TRAN-03, TRAN-04
**Success Criteria** (what must be TRUE):
  1. The Brain completes a real round-trip against a bare n8n Hands workflow over the contract (not the mock)
  2. Real-n8n semantics are exercised: a rejection, a timeout, a duplicate idempotency key, and a partial failure each behave per the contract
  3. The bare workflow is hit with 10+ concurrent requests; async-to-webhook physics (queuing, thread-starvation) are characterized
  4. A transport latency baseline is recorded and trace spans connect across the boundary (one trace, both sides)
**Plans**: TBD

Plans:
- [ ] 04-01: Bare n8n Hands workflow implementing one contract capability (with injectable rejection/timeout)
- [ ] 04-02: Semantics smoke — rejection, timeout, duplicate idempotency, partial failure against real n8n
- [ ] 04-03: Concurrency smoke (10+ concurrent) + transport latency baseline + cross-boundary trace continuity

### Phase 5: Read-Only Research Tool
**Goal**: The Brain gains its first real tool via the LangGraph tool loop. Tools receive caller context via **injected state from the very first tool** (the injected-context seam) so identity (Phase 6) slots in with no signature rewrite. Research is side-effect-free and identity-independent — the canary-safe first capability. The **evaluation harness is stood up here** and runs a growing subset every subsequent phase.
**Depends on**: Phase 1 (loop), Phase 3 (contract), Phase 4 (real transport validated)
**Requirements**: TOOL-01, TOOL-02, TOOL-03, TOOL-04, TOOL-05
**Success Criteria** (what must be TRUE):
  1. The Brain answers a research question through the LangGraph tool loop
  2. Tools receive caller context via injected state (not model-settable args) from the first tool
  3. A slow tool can be cancelled or timed out without wedging the agent loop
  4. An eval harness exists and runs against the Brain; it grows by at least one case each subsequent phase
**Plans**: TBD

Plans:
- [ ] 05-01: LangGraph tool loop (ToolNode + tools_condition) + injected caller-context seam
- [ ] 05-02: Read-only research/summarization tool (API key via env, never committed) + heavy-tool cancellation/timeout
- [ ] 05-03: Eval harness foundation (a small, growing case set run each phase)

### Phase 6: Identity via Hands
**Goal**: The Brain resolves *who it is talking to* by asking Hands; the resolved identity flows in through the injected-context seam — never a model-settable parameter. Policy-by-architecture: control is structural, not promptable.
**Depends on**: Phase 5 (injected-context seam), Phase 3 (contract)
**Requirements**: IDENT-01, IDENT-02, IDENT-03
**Success Criteria** (what must be TRUE):
  1. The Brain personalizes by the identity Hands resolves for the caller
  2. Caller identity arrives via the injected seam and is not a model-settable parameter
  3. A red-team test confirms the model cannot target another identity even under direct social-engineering prompts
**Plans**: TBD

Plans:
- [ ] 06-01: Identity-resolve capability wired through the contract
- [ ] 06-02: Feed resolved identity into the injected-context seam
- [ ] 06-03: Cross-identity red-team test

### Phase 7: Memory Read via Hands
**Goal**: The Brain retrieves relevant per-person memory through Hands. Hands owns the database and the vector store; the Brain only asks. No credentials cross the boundary.
**Depends on**: Phase 6 (identity — memory is per-person)
**Requirements**: MEMR-01, MEMR-02
**Success Criteria** (what must be TRUE):
  1. The Brain recalls a known fact about the current person via a Hands call
  2. The Brain has no database connection or credentials of its own
**Plans**: TBD

Plans:
- [ ] 07-01: Memory-retrieve capability (relevance/recency-scored results injected into context)
- [ ] 07-02: Verify Brain holds zero DB access; recall integration test

### Phase 8: Output via Hands
**Goal**: Fill the REAL output policy behind the stream-shaped emit seam — PII scrubbing and conversation-privacy (Hands-owned), and streaming safety (Hands approves the envelope before any user-visible release). Comes BEFORE memory-write so that writes can be conditioned on a real, observable output status. No rewrite: the stream-capable seam already exists; this phase populates its policy.
**Depends on**: Phase 6 (identity/privacy), Phase 5 (tool loop)
**Requirements**: OUT-01, OUT-02, OUT-03
**Success Criteria** (what must be TRUE):
  1. A response containing PII is scrubbed per policy by Hands, behind the existing emit seam
  2. Conversation-privacy is enforced by Hands as the canonical owner
  3. Streaming emits no user-visible tokens before Hands' approval; a privacy red-team passes
**Plans**: TBD

Plans:
- [ ] 08-01: Fill the Hands Output Router policy (PII scrub + conversation_privacy) behind the emit seam
- [ ] 08-02: Streaming-safety (Hands approves the envelope before user-visible release)
- [ ] 08-03: Privacy red-team suite against the output path

### Phase 9: Memory Write via Hands + Compensation
**Goal**: The Brain *proposes* memory writes; Hands performs them. First state-changing capability — idempotency keys mandatory, and a **durable compensation state-machine** (proposed → staged → committed → compensated) tied to the Phase-8 output status, with retry rather than a synchronous undo (a compensation call that itself fails is retried, not lost). The Phase-2 reconciliation rule is enforced here. This is where production-grade saga rigor belongs.
**Depends on**: Phase 8 (output status exists), Phase 7 (memory read)
**Requirements**: MEMW-01, MEMW-02, MEMW-03, MEMW-04, MEMW-05
**Success Criteria** (what must be TRUE):
  1. A fact stated in conversation is persisted via a Hands write
  2. Every write carries an idempotency key; a duplicate-write test confirms exactly-once persistence
  3. Writes move through an explicit state machine tied to output status; the Brain never writes to the database directly
  4. A compensation that fails is durably retried (local queue/log) — no orphaned memory of an unseen conversation, and no silently-dropped write after a seen one
  5. After a crash mid-write, recovery reconciles to Hands per the Phase-2 rule
**Plans**: TBD

Plans:
- [ ] 09-01: Memory-write capability with idempotency keys + the write state machine
- [ ] 09-02: Durable compensation (local queue/log, retryable) tied to output status
- [ ] 09-03: Duplicate-request + aborted-output + crash-recovery reconciliation tests

### Phase 10: Eval Parity + Cutover Controls
**Goal**: Before any production traffic, prove the Brain matches the predecessor and build the controls cutover depends on. Wire the existing eval suite to the Brain, demonstrate parity on a representative set, and design the rollback / kill-switch / proposal-only guarantees + audit logging that shadow and canary rely on.
**Depends on**: Phase 9 (Brain is feature-complete end-to-end)
**Requirements**: EVAL-01, EVAL-02, EVAL-03
**Success Criteria** (what must be TRUE):
  1. The predecessor eval suite runs against Brain output
  2. The Brain demonstrates parity (or better) on a representative case set
  3. Rollback / kill-switch, proposal-only guarantees, and audit logging are designed and unit-proven
**Plans**: TBD

Plans:
- [ ] 10-01: Wire the predecessor eval suite to the Brain
- [ ] 10-02: Parity run + gap report
- [ ] 10-03: Rollback / kill-switch / proposal-only controls + audit logging

### Phase 11: Shadow Mode
**Goal**: The Brain runs alongside the existing production agent — all inputs mirrored, the Brain proposes decisions and tool calls but executes nothing. Prove parity (decisions, latency vs the Phase 4 baseline, cost, eval scores) and operability before any real traffic moves.
**Depends on**: Phase 10
**Requirements**: SHAD-01, SHAD-02, SHAD-03, SHAD-04
**Success Criteria** (what must be TRUE):
  1. Production inputs are mirrored to the Brain; the Brain proposes but never executes
  2. Brain decisions match or improve on the production agent across a sustained window, with p95 latency within target of the Phase 4 baseline
  3. Any Brain-side failure is diagnosable within 10 minutes from logs/traces alone
  4. A distributed-system eval suite (privacy red-team, duplicate sends, webhook replay, concurrent conversations, stale identity) passes against Brain output
**Plans**: TBD

Plans:
- [ ] 11-01: Input mirroring + non-executing proposal mode
- [ ] 11-02: Side-by-side decision/latency/cost/eval logging (latency vs Phase 4 baseline)
- [ ] 11-03: Operator-diagnose test (≤10-min failure diagnosis from traces)
- [ ] 11-04: Distributed-system eval suite

### Phase 12: Single Canary Path
**Goal**: Move ONE low-risk path — read-only research/summarization — to Brain-driven production, for the maintainer first, then a small invited group. Rollback (built in Phase 10) is always one switch away.
**Depends on**: Phase 11
**Requirements**: CAN-01, CAN-02, CAN-03, CAN-04
**Success Criteria** (what must be TRUE):
  1. The research path runs Brain-driven in production
  2. It is continuously evaluated against the regression suite with no regressions
  3. The Phase-10 rollback switch instantly returns the path to the production agent
  4. The maintainer spends less time fighting infrastructure and more on agent behavior
**Plans**: TBD

Plans:
- [ ] 12-01: Canary routing for the research path (maintainer-first)
- [ ] 12-02: Continuous eval + rollback exercised
- [ ] 12-03: Canary soak + maintainer-metric review

## Progress

**Execution Order:**
Phases execute in numeric order: 1 → 2 → 3 → 4 → 5 → 6 → 7 → 8 → 9 → 10 → 11 → 12

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Orchestration Skeleton | 0/5 | Not started | - |
| 2. State, Durability & Recovery | 0/3 | Not started | - |
| 3. Hands Contract + Mock | 0/5 | Not started | - |
| 4. Real-n8n Transport/Semantics/Concurrency | 0/3 | Not started | - |
| 5. Read-Only Research Tool | 0/3 | Not started | - |
| 6. Identity via Hands | 0/3 | Not started | - |
| 7. Memory Read via Hands | 0/2 | Not started | - |
| 8. Output via Hands | 0/3 | Not started | - |
| 9. Memory Write + Compensation | 0/3 | Not started | - |
| 10. Eval Parity + Cutover Controls | 0/3 | Not started | - |
| 11. Shadow Mode | 0/4 | Not started | - |
| 12. Single Canary Path | 0/3 | Not started | - |

## Post-v1 (not in this roadmap)

Tracked in REQUIREMENTS.md v2 — surfaced here for shape only:
- **Aerys-Voice** (TypeScript streaming runtime, sub-4s) — gated on the latency budget (baselined in Phase 4)
- **Self-evolution** (GEPA-style optimization against eval feedback)
- **MCP integration** (scoped, sandboxed, no credential pass-through)
- **Earned migration** — remaining production paths move to Brain one at a time, each with its own canary

---
*Roadmap v3 — 2026-06-01. Polished after a second cross-architecture review (strong two-brain convergence; build-first endorsed across three independent passes). Phase 1 split; output reordered ahead of memory-write; eval-parity + cutover controls added before shadow; streaming emit seam, contract versioning, real-n8n semantics/concurrency, and durable compensation folded in. Production-grade saga/load rigor concentrated at the state-changing and cutover phases.*
