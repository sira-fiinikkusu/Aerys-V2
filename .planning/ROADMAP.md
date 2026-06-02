# Roadmap: Aerys-V2

## Overview

A ground-up build of the **Brain** (Python/LangGraph) for a personal AI agent, growing one capability
per phase against a typed contract to the **Hands** (n8n) governance layer. The build is deliberately
*incremental*: stand up a bare orchestration skeleton, define the Brain↔Hands boundary against a local
mock, then layer real capabilities — research, identity, memory, output — one at a time, each landing
the risk mitigation relevant to it. Only once the Brain is feature-capable do the final phases bring
**cutover safety**: shadow mode alongside the existing production agent, then a single canary path in
production with rollback. Build-first, migrate-last.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): planned milestone work
- Decimal phases (2.1, 2.2): urgent insertions (marked INSERTED)

- [ ] **Phase 1: Orchestration Skeleton** - A local LangGraph Brain that loads its persona, routes through a swappable model, holds state, and traces every run
- [ ] **Phase 2: The Hands Contract + Local Mock** - Typed Brain↔Hands capability boundary, exercised against a local stub before any real infra
- [ ] **Phase 3: First Tool — Read-Only Research** - The LangGraph tool loop, with a side-effect-free research tool and cancellation safety
- [ ] **Phase 4: Identity via Hands** - Caller identity resolved by Hands and *injected*, never model-settable (policy-by-architecture)
- [ ] **Phase 5: Memory Read via Hands** - The Brain recalls per-person memory through Hands, with no direct database access
- [ ] **Phase 6: Memory Write via Hands** - First state-changing capability — idempotent writes through Hands' consolidation rules
- [ ] **Phase 7: Output via Hands** - User-visible output through Hands' Output Router; PII scrub, privacy, and streaming safety
- [ ] **Phase 8: Shadow Mode** - The Brain runs alongside production, proposes decisions, executes nothing — parity proven before cutover
- [ ] **Phase 9: Single Canary Path** - One low-risk path goes Brain-driven in production, rollback always available

## Phase Details

### Phase 1: Orchestration Skeleton
**Goal**: A LangGraph Brain runs locally on the Jetson — receives a text input, loads `soul.md` into the system prompt, routes through a model-swappable agent node, maintains conversation state, and returns a persona-shaped reply. Tracing is wired from the first commit. No tools, no memory, no Hands yet.
**Depends on**: Nothing (first phase)
**Requirements**: ORCH-01, ORCH-02, ORCH-03, ORCH-04, ORCH-05
**Success Criteria** (what must be TRUE):
  1. A text message sent to the Brain returns a persona-shaped reply on the Jetson
  2. The model provider can be changed via config with no code edit
  3. `soul.md` content visibly shapes the response (persona injection works)
  4. Conversation state persists across turns within a session
  5. Every run emits an OpenTelemetry trace
**Plans**: TBD

Plans:
- [ ] 01-01: Python project scaffold (uv, dependency baseline, project layout, reproducible/containerized Jetson run)
- [ ] 01-02: LangGraph chat graph + conversation state + swappable model-provider abstraction
- [ ] 01-03: `soul.md` loading + persona injection; structured config loading
- [ ] 01-04: OpenTelemetry instrumentation + local smoke test on Jetson

### Phase 2: The Hands Contract + Local Mock
**Goal**: Define the typed Brain↔Hands capability boundary and a local mock so the Brain can "ask Hands" before any real infrastructure exists. This phase makes the boundary rule — "Brain may ask, Hands decides" — concrete and testable.
**Depends on**: Phase 1
**Requirements**: HANDS-01, HANDS-02, HANDS-03, HANDS-04, HANDS-05
**Success Criteria** (what must be TRUE):
  1. The contract defines identity, memory-read, memory-write, and output-send capabilities with typed request/response schemas
  2. Each capability documents idempotency, auth, timeout, and privacy semantics
  3. A local mock Hands server answers contract calls with no real infra
  4. Contract tests pass for happy-path and duplicate-request cases
  5. The boundary doc names one canonical owner per concept; the Brain holds no Hands-owned credentials
**Plans**: TBD

Plans:
- [ ] 02-01: Capability contract definition (schemas + semantics; transport = HTTP/JSON to start, gRPC deferred)
- [ ] 02-02: Generated/typed Hands client in the Brain
- [ ] 02-03: Local mock Hands server (deterministic fixtures)
- [ ] 02-04: Contract test suite (happy-path + idempotency/duplicate cases) + boundary ownership doc

### Phase 3: First Tool — Read-Only Research
**Goal**: The Brain gains its first real tool via the LangGraph tool loop — read-only research/summarization, chosen because it has no memory writes, no side effects, and no privacy-sensitive output. This is the safest possible first capability and the future canary path.
**Depends on**: Phase 1 (loop), Phase 2 (contract patterns)
**Requirements**: TOOL-01, TOOL-02, TOOL-03
**Success Criteria** (what must be TRUE):
  1. The Brain answers a research question by calling the tool through the LangGraph tool loop
  2. A slow tool call can be cancelled or timed out without wedging the agent loop
  3. Traces show tool-routing overhead near zero (orchestration is nearly free)
**Plans**: TBD

Plans:
- [ ] 03-01: LangGraph tool loop (ToolNode + tools_condition)
- [ ] 03-02: Read-only research/summarization tool (API key via env, never committed)
- [ ] 03-03: Heavy-tool isolation — cancellation, timeout, off-main-loop execution

### Phase 4: Identity via Hands
**Goal**: The Brain resolves *who it is talking to* by asking Hands, with the caller identity **injected** into tool calls — never a parameter the model can set. This is the architectural answer to the privacy/identity boundary (the project's #1 risk): control is structural, not promptable.
**Depends on**: Phase 2 (contract), Phase 3 (tool loop)
**Requirements**: IDENT-01, IDENT-02, IDENT-03
**Success Criteria** (what must be TRUE):
  1. The Brain personalizes by the identity Hands resolves for the caller
  2. Caller identity is injected into tool calls and is not exposed as a model-settable parameter
  3. A red-team test confirms the model cannot target another identity even under direct social-engineering prompts
**Plans**: TBD

Plans:
- [ ] 04-01: Identity-resolve capability wired through the contract
- [ ] 04-02: Injected-identity plumbing (caller identity via injected state/config, not tool args)
- [ ] 04-03: Cross-identity red-team test

### Phase 5: Memory Read via Hands
**Goal**: The Brain retrieves relevant per-person memory through Hands. Hands owns the database and the vector store; the Brain only asks. No credentials cross the boundary.
**Depends on**: Phase 2 (contract), Phase 4 (identity — memory is per-person)
**Requirements**: MEMR-01, MEMR-02
**Success Criteria** (what must be TRUE):
  1. The Brain recalls a known fact about the current person via a Hands call
  2. The Brain has no database connection or credentials of its own
**Plans**: TBD

Plans:
- [ ] 05-01: Memory-retrieve capability (relevance/recency-scored results injected into context)
- [ ] 05-02: Verify Brain holds zero DB access; recall integration test

### Phase 6: Memory Write via Hands
**Goal**: The Brain *proposes* memory writes; Hands performs them, subject to its own consolidation rules. This is the first state-changing capability, so idempotency keys are mandatory — a double-write must be impossible.
**Depends on**: Phase 5
**Requirements**: MEMW-01, MEMW-02, MEMW-03
**Success Criteria** (what must be TRUE):
  1. A fact stated in conversation is persisted via a Hands write
  2. Every write carries an idempotency key
  3. A duplicate-write test confirms exactly-once persistence; the Brain never writes to the database directly
**Plans**: TBD

Plans:
- [ ] 06-01: Memory-write capability with mandatory idempotency keys
- [ ] 06-02: Duplicate-request contract test (exactly-once persistence)

### Phase 7: Output via Hands
**Goal**: All user-visible output flows through Hands' Output Router, which owns PII scrubbing and conversation-privacy. Streaming stays Brain-internal until Hands approves the envelope — so tokens never leave the Brain ahead of a policy decision.
**Depends on**: Phase 4 (identity/privacy), Phase 6 (state-changing pattern)
**Requirements**: OUT-01, OUT-02, OUT-03
**Success Criteria** (what must be TRUE):
  1. A response containing PII is scrubbed per policy by Hands, not the Brain
  2. Conversation-privacy is enforced by Hands as the canonical owner
  3. Streaming emits no user-visible tokens before Hands' policy review; a privacy red-team passes
**Plans**: TBD

Plans:
- [ ] 07-01: Output-send capability; privacy-shaped result enforcement (Hands owns conversation_privacy)
- [ ] 07-02: Streaming-safety model (Hands approves envelope before user-visible release)
- [ ] 07-03: Privacy red-team suite against the output path

### Phase 8: Shadow Mode
**Goal**: The Brain runs alongside the existing production agent — all inputs mirrored, the Brain proposes decisions and tool calls but executes nothing. Prove parity (decisions, latency, cost, eval scores) and operability before any real traffic moves.
**Depends on**: Phase 7 (Brain is now feature-capable end-to-end)
**Requirements**: SHAD-01, SHAD-02, SHAD-03, SHAD-04
**Success Criteria** (what must be TRUE):
  1. Production inputs are mirrored to the Brain; the Brain proposes but never executes
  2. Brain decisions match or improve on the production agent across a sustained window, with p95 latency within target
  3. Any Brain-side failure is diagnosable within 10 minutes from logs/traces alone
  4. A distributed-system eval suite (privacy red-team, duplicate sends, webhook replay, concurrent conversations, stale identity) passes against Brain output
**Plans**: TBD

Plans:
- [ ] 08-01: Input mirroring + non-executing proposal mode
- [ ] 08-02: Side-by-side decision/latency/cost/eval logging
- [ ] 08-03: Operator-diagnose test (≤10-min failure diagnosis from traces)
- [ ] 08-04: Distributed-system eval suite

### Phase 9: Single Canary Path
**Goal**: Move ONE low-risk path — read-only research/summarization — to Brain-driven production, for the maintainer first, then a small invited group. Rollback is always one switch away.
**Depends on**: Phase 8
**Requirements**: CAN-01, CAN-02, CAN-03, CAN-04
**Success Criteria** (what must be TRUE):
  1. The research path runs Brain-driven in production
  2. It is continuously evaluated against the regression suite with no regressions
  3. A rollback switch instantly returns the path to the production agent
  4. The maintainer spends less time fighting infrastructure and more on agent behavior
**Plans**: TBD

Plans:
- [ ] 09-01: Canary routing for the research path (maintainer-first)
- [ ] 09-02: Continuous eval + rollback switch
- [ ] 09-03: Canary soak + maintainer-metric review

## Progress

**Execution Order:**
Phases execute in numeric order: 1 → 2 → 3 → 4 → 5 → 6 → 7 → 8 → 9

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Orchestration Skeleton | 0/4 | Not started | - |
| 2. Hands Contract + Mock | 0/4 | Not started | - |
| 3. Read-Only Research Tool | 0/3 | Not started | - |
| 4. Identity via Hands | 0/3 | Not started | - |
| 5. Memory Read via Hands | 0/2 | Not started | - |
| 6. Memory Write via Hands | 0/2 | Not started | - |
| 7. Output via Hands | 0/3 | Not started | - |
| 8. Shadow Mode | 0/4 | Not started | - |
| 9. Single Canary Path | 0/3 | Not started | - |

## Post-v1 (not in this roadmap)

Tracked in REQUIREMENTS.md v2 — surfaced here for shape only:
- **Aerys-Voice** (TypeScript streaming runtime, sub-4s) — gated on a Brain↔Voice latency benchmark
- **Self-evolution** (GEPA-style optimization against eval feedback)
- **MCP integration** (scoped, sandboxed, no credential pass-through)
- **Earned migration** — remaining production paths move to Brain one at a time, each with its own canary

---
*Roadmap drafted: 2026-06-01 (DRAFT — pending cross-architecture review before execution)*
