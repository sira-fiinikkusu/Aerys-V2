# Roadmap: Aerys-V2

## Overview

A ground-up build of the **Brain** (Python/LangGraph) for a personal AI agent, growing one capability
per phase against a typed contract to the **Hands** (n8n) governance layer. The build is *incremental*
and *build-first*: stand up a bare orchestration skeleton, define the Brain↔Hands boundary, **prove
that boundary against real n8n early** (before stacking orchestration on a mock), then layer real
capabilities — research, identity, memory, output — one at a time. Structural seams that are
expensive to retrofit (the output-approval gate, injected caller-context, trace propagation,
failure/compensation) are laid at the skeleton/contract stage; later phases fill the policy behind
them. Only once the Brain is feature-capable do the final phases bring **cutover safety**: shadow mode
alongside the existing production agent, then a single canary path in production with rollback.
Build-first, integrate-early, migrate-last.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): planned milestone work
- Decimal phases (2.1, 2.2): urgent insertions (marked INSERTED)

- [ ] **Phase 1: Orchestration Skeleton** - LangGraph Brain: persona, swappable model, conversation state via a Brain-owned checkpointer, the output-emit seam, tracing from commit 1
- [ ] **Phase 2: Hands Contract + Local Mock** - Typed boundary with idempotency, rejection/compensation, and trace-context semantics; exercised against a local mock
- [ ] **Phase 3: Real-n8n Transport Smoke + Latency Baseline** - Prove the contract survives contact with real n8n; measure the transport physics before building on it
- [ ] **Phase 4: Read-Only Research Tool** - The tool loop with the injected caller-context seam; a side-effect-free research tool and cancellation safety
- [ ] **Phase 5: Identity via Hands** - Caller identity resolved by Hands and injected through the seam, never model-settable (policy-by-architecture)
- [ ] **Phase 6: Memory Read via Hands** - The Brain recalls per-person memory through Hands, with no direct database access
- [ ] **Phase 7: Memory Write via Hands + Compensation** - Idempotent writes, provisional until output succeeds — no memory of a conversation the user never saw
- [ ] **Phase 8: Output via Hands** - Fill the real PII / privacy / streaming policy behind the emit seam built in Phase 1
- [ ] **Phase 9: Shadow Mode** - The Brain runs alongside production, proposes decisions, executes nothing — parity proven before cutover
- [ ] **Phase 10: Single Canary Path** - One low-risk path goes Brain-driven in production, rollback always available

## Phase Details

### Phase 1: Orchestration Skeleton
**Goal**: A LangGraph Brain runs locally on the Jetson — receives a text input, loads `soul.md` into the system prompt, routes through a model-swappable agent node, maintains conversation state via a **Brain-owned checkpointer** (its own orchestration state, distinct from Hands-owned memory), and routes ALL user-visible output through a single **emit seam** (a pass-through stub until Hands exists). Tracing is wired from the first commit.
**Depends on**: Nothing (first phase)
**Requirements**: ORCH-01, ORCH-02, ORCH-03, ORCH-04, ORCH-05, ORCH-06, ORCH-07
**Success Criteria** (what must be TRUE):
  1. A text message returns a persona-shaped reply on the Jetson
  2. The model provider can be changed via config with no code edit
  3. `soul.md` content visibly shapes the response
  4. Conversation state persists across turns via a Brain-owned checkpointer; the store and its Jetson durability are a recorded decision
  5. All user-visible output passes through a single emit seam (the future Hands-approval gate)
  6. Every run emits an OpenTelemetry trace
**Plans**: TBD

Plans:
- [ ] 01-01: Python project scaffold (uv, layout, reproducible/containerized Jetson run)
- [ ] 01-02: LangGraph chat graph + conversation state + Brain-owned checkpointer (decide store + Jetson durability) + swappable model provider
- [ ] 01-03: `soul.md` loading + persona injection; structured config loading
- [ ] 01-04: Output-emit seam — single gate all user-visible output routes through (pass-through until Hands)
- [ ] 01-05: OpenTelemetry instrumentation + local smoke test on Jetson

### Phase 2: Hands Contract + Local Mock
**Goal**: Define the typed Brain↔Hands capability boundary and a local mock. The contract carries — per capability — idempotency, auth, timeout, privacy, AND **failure/rejection + compensation semantics**, plus **W3C trace-context propagation** on every call. The emit seam from Phase 1 now routes through a Hands "approve-envelope" capability (the mock rubber-stamps for now).
**Depends on**: Phase 1
**Requirements**: HANDS-01, HANDS-02, HANDS-03, HANDS-04, HANDS-05, HANDS-06, HANDS-07
**Success Criteria** (what must be TRUE):
  1. The contract defines identity, memory-read, memory-write, and output-approve/send with typed schemas
  2. Each capability documents idempotency, auth, timeout, privacy, AND rejection/compensation semantics
  3. The contract propagates W3C `traceparent` on every call
  4. A local mock Hands answers all calls (including approve-envelope = rubber-stamp) with no real infra
  5. Contract tests pass for happy-path, duplicate-request, AND rejection/compensation paths
  6. The Phase-1 emit seam routes through the Hands approve-envelope capability
  7. The boundary doc names one canonical owner per concept; the Brain holds no Hands-owned credentials
**Plans**: TBD

Plans:
- [ ] 02-01: Contract definition — schemas + idempotency/auth/timeout/privacy + rejection/compensation semantics (transport HTTP/JSON, gRPC deferred)
- [ ] 02-02: Trace-context propagation across the contract (`traceparent` on every call)
- [ ] 02-03: Typed Hands client + wire the emit seam to the approve-envelope capability
- [ ] 02-04: Local mock Hands server (deterministic fixtures, including rejection cases)
- [ ] 02-05: Contract test suite (happy + duplicate + rejection/compensation) + boundary ownership doc

### Phase 3: Real-n8n Transport Smoke + Latency Baseline
**Goal**: Prove the contract survives contact with real n8n BEFORE building five phases of orchestration on a clean mock. One bare n8n workflow answers one contract call end-to-end; capture the transport physics (the very quirks — cold starts, webhook latency, serialization, partial failure — that the rebuild exists to escape).
**Depends on**: Phase 2
**Requirements**: TRAN-01, TRAN-02, TRAN-03
**Success Criteria** (what must be TRUE):
  1. The Brain completes a real round-trip against a bare n8n workflow over the contract (not the mock)
  2. Transport physics are measured: serialization overhead, timeout behavior, cold-start, p50/p95 latency
  3. An end-to-end Jetson→Brain→n8n→Brain latency baseline is recorded, and trace spans connect across the boundary (one trace, both sides)
**Plans**: TBD

Plans:
- [ ] 03-01: Bare "echo" n8n Hands workflow implementing one contract capability
- [ ] 03-02: Brain↔real-n8n round-trip; capture timeout / serialization / cold-start behavior
- [ ] 03-03: Record the latency baseline + verify cross-boundary trace continuity

### Phase 4: Read-Only Research Tool
**Goal**: The Brain gains its first real tool via the LangGraph tool loop. Tools receive caller context via **injected state from the very first tool** (the injected-context seam) — so identity (Phase 5) slots in with no signature rewrite. Research is side-effect-free and identity-independent: the canary-safe first capability and the future canary path.
**Depends on**: Phase 1 (loop), Phase 2 (contract), Phase 3 (real transport validated)
**Requirements**: TOOL-01, TOOL-02, TOOL-03, TOOL-04
**Success Criteria** (what must be TRUE):
  1. The Brain answers a research question through the LangGraph tool loop
  2. Tools receive caller context via injected state (not model-settable args) from the first tool
  3. A slow tool can be cancelled or timed out without wedging the agent loop
**Plans**: TBD

Plans:
- [ ] 04-01: LangGraph tool loop (ToolNode + tools_condition)
- [ ] 04-02: Injected caller-context seam (tools take context via injected state/config, never tool args)
- [ ] 04-03: Read-only research/summarization tool (API key via env, never committed)
- [ ] 04-04: Heavy-tool isolation — cancellation, timeout, off-main-loop execution

### Phase 5: Identity via Hands
**Goal**: The Brain resolves *who it is talking to* by asking Hands; the resolved identity flows in through the injected-context seam from Phase 4 — never a model-settable parameter. Policy-by-architecture: control is structural, not promptable.
**Depends on**: Phase 4 (injected-context seam), Phase 2 (contract)
**Requirements**: IDENT-01, IDENT-02, IDENT-03
**Success Criteria** (what must be TRUE):
  1. The Brain personalizes by the identity Hands resolves for the caller
  2. Caller identity arrives via the injected seam and is not a model-settable parameter
  3. A red-team test confirms the model cannot target another identity even under direct social-engineering prompts
**Plans**: TBD

Plans:
- [ ] 05-01: Identity-resolve capability wired through the contract
- [ ] 05-02: Feed resolved identity into the injected-context seam
- [ ] 05-03: Cross-identity red-team test

### Phase 6: Memory Read via Hands
**Goal**: The Brain retrieves relevant per-person memory through Hands. Hands owns the database and the vector store; the Brain only asks. No credentials cross the boundary.
**Depends on**: Phase 5 (identity — memory is per-person)
**Requirements**: MEMR-01, MEMR-02
**Success Criteria** (what must be TRUE):
  1. The Brain recalls a known fact about the current person via a Hands call
  2. The Brain has no database connection or credentials of its own
**Plans**: TBD

Plans:
- [ ] 06-01: Memory-retrieve capability (relevance/recency-scored results injected into context)
- [ ] 06-02: Verify Brain holds zero DB access; recall integration test

### Phase 7: Memory Write via Hands + Compensation
**Goal**: The Brain *proposes* memory writes; Hands performs them, subject to its own consolidation rules. First state-changing capability — idempotency keys mandatory, AND **compensation**: a memory write is provisional/compensable until the corresponding output succeeds, so an aborted output never leaves the Brain remembering a conversation the user never saw (the two-generals fix).
**Depends on**: Phase 6
**Requirements**: MEMW-01, MEMW-02, MEMW-03, MEMW-04
**Success Criteria** (what must be TRUE):
  1. A fact stated in conversation is persisted via a Hands write
  2. Every write carries an idempotency key
  3. A duplicate-write test confirms exactly-once persistence; the Brain never writes to the database directly
  4. A memory write is compensated/voided when the corresponding output send fails — no orphaned memory of an unseen conversation
**Plans**: TBD

Plans:
- [ ] 07-01: Memory-write capability with mandatory idempotency keys
- [ ] 07-02: Compensation/saga — provisional write tied to output success; compensate on output failure
- [ ] 07-03: Duplicate-request + aborted-output compensation tests

### Phase 8: Output via Hands
**Goal**: Fill the REAL output policy behind the emit seam built in Phase 1 — PII scrubbing and conversation-privacy (Hands-owned), and streaming safety (Hands approves the envelope before any user-visible release). No rewrite: the seam already exists; this phase populates its policy.
**Depends on**: Phase 5 (identity/privacy), Phase 7 (compensation pattern)
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

### Phase 9: Shadow Mode
**Goal**: The Brain runs alongside the existing production agent — all inputs mirrored, the Brain proposes decisions and tool calls but executes nothing. Prove parity (decisions, latency vs the Phase 3 baseline, cost, eval scores) and operability before any real traffic moves.
**Depends on**: Phase 8 (Brain is now feature-capable end-to-end)
**Requirements**: SHAD-01, SHAD-02, SHAD-03, SHAD-04
**Success Criteria** (what must be TRUE):
  1. Production inputs are mirrored to the Brain; the Brain proposes but never executes
  2. Brain decisions match or improve on the production agent across a sustained window, with p95 latency within target of the Phase 3 baseline
  3. Any Brain-side failure is diagnosable within 10 minutes from logs/traces alone
  4. A distributed-system eval suite (privacy red-team, duplicate sends, webhook replay, concurrent conversations, stale identity) passes against Brain output
**Plans**: TBD

Plans:
- [ ] 09-01: Input mirroring + non-executing proposal mode
- [ ] 09-02: Side-by-side decision/latency/cost/eval logging (latency vs Phase 3 baseline)
- [ ] 09-03: Operator-diagnose test (≤10-min failure diagnosis from traces)
- [ ] 09-04: Distributed-system eval suite

### Phase 10: Single Canary Path
**Goal**: Move ONE low-risk path — read-only research/summarization — to Brain-driven production, for the maintainer first, then a small invited group. Rollback is always one switch away.
**Depends on**: Phase 9
**Requirements**: CAN-01, CAN-02, CAN-03, CAN-04
**Success Criteria** (what must be TRUE):
  1. The research path runs Brain-driven in production
  2. It is continuously evaluated against the regression suite with no regressions
  3. A rollback switch instantly returns the path to the production agent
  4. The maintainer spends less time fighting infrastructure and more on agent behavior
**Plans**: TBD

Plans:
- [ ] 10-01: Canary routing for the research path (maintainer-first)
- [ ] 10-02: Continuous eval + rollback switch
- [ ] 10-03: Canary soak + maintainer-metric review

## Progress

**Execution Order:**
Phases execute in numeric order: 1 → 2 → 3 → 4 → 5 → 6 → 7 → 8 → 9 → 10

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Orchestration Skeleton | 0/5 | Not started | - |
| 2. Hands Contract + Mock | 0/5 | Not started | - |
| 3. Real-n8n Transport Smoke | 0/3 | Not started | - |
| 4. Read-Only Research Tool | 0/4 | Not started | - |
| 5. Identity via Hands | 0/3 | Not started | - |
| 6. Memory Read via Hands | 0/2 | Not started | - |
| 7. Memory Write + Compensation | 0/3 | Not started | - |
| 8. Output via Hands | 0/3 | Not started | - |
| 9. Shadow Mode | 0/4 | Not started | - |
| 10. Single Canary Path | 0/3 | Not started | - |

## Post-v1 (not in this roadmap)

Tracked in REQUIREMENTS.md v2 — surfaced here for shape only:
- **Aerys-Voice** (TypeScript streaming runtime, sub-4s) — gated on the latency budget (baselined in Phase 3)
- **Self-evolution** (GEPA-style optimization against eval feedback)
- **MCP integration** (scoped, sandboxed, no credential pass-through)
- **Earned migration** — remaining production paths move to Brain one at a time, each with its own canary

---
*Roadmap v2 — 2026-06-01. Revised after cross-architecture review. Build-first endorsed-with-changes: real-n8n integration pulled early (Phase 3), structural seams moved to skeleton/contract, compensation added at memory-write.*
