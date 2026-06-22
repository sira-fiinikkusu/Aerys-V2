# Phase 1 — Orchestration Skeleton — PLAN

**Status:** v2 — three-brain reviewed (Gemini + Codex) + verified vs langgraph 1.2.6 docs + grounded against a reference LangGraph implementation. Supersedes the v1 draft (NEEDS-REWORK). Review trail: `01-REVIEW.md`.
**Phase goal (ROADMAP):** A LangGraph Brain runs locally on the Jetson — loads `soul.md`, routes through a swappable model, holds conversation state, and routes ALL user-visible output through a single stream-shaped emit seam. Tracing from the first commit; reproducible ARM build.
**Requirements:** ORCH-01..07 · **Depends on:** nothing (first phase)

## Reference map (where each pattern comes from)
- **Confirmed against a reference LangGraph implementation** (adopt the proven shape): `add_messages` reducer, `InMemorySaver`, `Identity` TypedDict + single accessor, empty-input guard, narrow-except-with-full-traceback + `BaseException` propagates, fail-fast config/prompt, swappable checkpointer seam, `build_model`/`build_graph` factories, caller-supplied thread_id, fake-model graph injection for offline tests.
- **Current-docs only (the reference is silent — it's synchronous)**: the **streaming emit seam** via `graph.astream`. Net-new for Aerys, driven by the future Voice runtime.
- **RESOLVED (external validation complete 2026-06-22)**: S2 caller-context channel = `configurable` + single accessor. A meets all 3 invariants but is **proven, not future-proof** — `context`/`Runtime` is LangGraph's stated long-term direction for typed per-run context (the `config_schema` *declaration* is deprecated, removal targeted v2.0). We never *declare* a `config_schema` (we read the `configurable` bag through one accessor), so we don't trip the deprecation; and `thread_id` rides `configurable` permanently, so `context` never fully replaces it. The single-accessor rule is the hedge: a later swap to `context_schema` stays bounded (~hours, cost scales per-tool, cheapest pre-Phase-5).

---

## Success Criteria (must be TRUE to close)
1. A text message returns a persona-shaped reply on the Jetson (ORCH-01)
2. Model provider swappable via config, no code edit (ORCH-02)
3. `soul.md` visibly shapes the response (ORCH-03)
4. Conversation state persists across turns within a session (ORCH-04)
5. All user-visible output flows through one stream-capable emit seam — chunked/async, mock-approved (ORCH-06)
6. Every run emits an OpenTelemetry trace (ORCH-05)
7. Container builds + boots reproducibly on ARM/Jetson; deps pinned; secrets-loading check passes (ORCH-07)

---

## Architectural Invariants (seams + guardrails — every plan honors these)

**S1 — Stream-shaped emit seam (→ Phase 8). [CORRECTED]** Streaming is a GRAPH-level concern, not a node. Graph nodes are state-transition functions (the chat node returns `{"messages": [reply]}`). The public entry `async def ask(query, *, session_id, identity, ...) -> AsyncGenerator[OutEvent, None]` **wraps `graph.astream(..., stream_mode="messages")`** and translates LangGraph stream events into typed `OutEvent`s. A mock-approval pass-through wraps the OutEvent stream (Phase 8 fills real approval by consuming-before-yielding — no signature change).

**S2 — Injected caller-context seam (→ Phases 5/6). [RESOLVED: `configurable` + single accessor, 2026-06-22]** This is the **authorization** identity — *who an action runs for*. Passed per-call via `config["configurable"]["identity"]`; read ONLY through a single accessor `identity_from_config(config)` returning an `Identity` TypedDict, defaulting to `UNKNOWN_CALLER`. **HARD RULES:** never in checkpointed state; never a model-settable arg; **never read raw — always via the accessor** (this single-accessor discipline is what bounds a future swap to `context_schema`). Phase 1 has no tools yet, so identity is composed into the system message only. **Pairs with attribution** (the Phase-6 "Speaker (verified)" turn format, below) — two deliberately separate channels for two jobs: authorization here vs *who said what* in the transcript.

**S3 — Trace-propagation seam (→ Phases 3/4). [CORRECTED]** OTel + `W3CTraceContextPropagator` configured now — but registering it does NOT auto-propagate. Define a named `PropagationHelper` / injection point so Phase 4's HTTP client has a contract to fill. No claim of automatic cross-boundary propagation.

**S4 — Swappable checkpointer seam (→ Phase 2).** Graph compiled with an injected checkpointer; Phase 1 passes `InMemorySaver`. Durable store (SQLite-vs-Postgres) is a Phase-2 decision — do NOT pick one now. (A reference implementation swaps `PostgresSaver` behind this exact seam in its own Phase 2.)

**Guardrails (reference-confirmed / synthesis):**
- **G1 — state is `messages` only**, `Annotated[list, add_messages]` (NOT `operator.add`). Test asserts no identity field in state.
- **G2 — exact dep pins + committed lockfile** (LangGraph core + prebuilt pinned together).
- **G3 — narrow exception boundary at `ask()`**: log `exc_info=True`, return a fallback `OutEvent`; `BaseException` (SIGTERM/KeyboardInterrupt) propagates. Config/prompt errors fail fast BEFORE the try (matches the reference).
- **G4 — deployable-process contract now**: a real `__main__`/`cli.py` wiring `ask()` to stdin/stdout, a SIGTERM handler, structured logging, and a Docker `HEALTHCHECK` that probes the actual process.
- **G5 — prompt-as-config**: load `soul.md` at runtime, fail-fast if missing, resolve absolute path, log content SHA256.
- **G6 — DI config (`pydantic-settings`) + factories** (`build_model`, `build_graph`, checkpointer). No module globals / scattered `os.environ`.
- **G7 — backpressure-capable stream. [STRENGTHENED]** Consumer `aclose()` must cancel UPSTREAM (the model stream + graph task), proven by test (not just "tokens stop"). `astream` closed in a `finally`. Define SIGTERM-vs-in-flight-stream behavior (stop new work, cancel active `ask()` generators, flush spans, exit). NB a reference implementation deliberately avoided off-thread wall-clock timeouts to prevent abandoned runs writing to shared state — our cancellation must be equally clean.

---

## Plan 01-01 — Scaffold + reproducible ARM build (ORCH-07)
**Tasks:** `uv init`; layout `src/aerys_v2/`, `tests/`, `config/`, `.github/workflows/`. `pyproject.toml` exact-pinned deps (`langgraph`, provider lib, `pydantic-settings`, `opentelemetry-sdk`+exporter) + committed `uv` lockfile (G2). Multi-stage `linux/arm64` Dockerfile (`python:3.11-slim`), install via `uv pip sync` against the lockfile, `HEALTHCHECK` (G4). `Settings` via `pydantic-settings` (env/`.env`) incl. model config + `soul_file_path`; fail-fast secrets check → non-zero exit (G6). SIGTERM handler + structured logging (G4). `Makefile`/`justfile` (`lock/build/test/run`); `pytest` skeleton + GH Actions.
**Acceptance:** reproducible build from the lockfile boots on Jetson; no-key run exits non-zero clearly; HEALTHCHECK passes.

## Plan 01-02 — LangGraph graph + state + swappable model (ORCH-01/02/04)
Built ONCE with the persona-injection point (01-03) and stream entry (01-04) stubbed so they FILL, not rewrite.
**Tasks:** `ChatState(TypedDict)` = `messages: Annotated[list, add_messages]` ONLY (G1); test asserts messages-only. `StateGraph(ChatState)`, single `chat` node returning `{"messages": [reply]}`, `START → chat → END` (no tools in Phase 1; tools are Phase 5). Compile with an injected checkpointer (S4) = `InMemorySaver`. `build_model(settings)` factory — config→client, no baked-in model id (reference pattern), swappable provider (ORCH-02). `ask(query, *, session_id, identity=None, graph=None, settings=None)` — `session_id` caller-supplied (default `"local"` for CLI), passed as `config["configurable"]["thread_id"]`; identity via `config["configurable"]["identity"]` read through `identity_from_config` (S2). The `chat` node composes `system_prompt` + the caller line into the `SystemMessage` each turn (identity from config, never state).
**Acceptance:** a text message returns a reply (ORCH-01); config-swap changes provider, no code edit (ORCH-02); same `session_id` sees prior context, **two different `session_id`s are fully isolated** (ORCH-04 + the identity-leak guard); state-shape test passes (G1).

## Plan 01-03 — soul.md persona + config (ORCH-03)
**Tasks:** `soul_file_path` in `Settings`; loader fails fast if missing, resolves absolute, logs SHA256 (G5). Compose `soul.md` system content + caller line into the chat node's `SystemMessage`. **soul.md public/private (RESOLVED):** public `config/soul.example.md` committed; real `config/soul.md` gitignored, injected at deploy. Lightweight capability-overlay: since Phase 1 has no tools, a startup warn (not crash) if `soul.md` claims tool/action abilities that don't exist yet (synthesis "skeleton-lies" guard).
**Acceptance:** reply reflects `soul.md` (ORCH-03); missing file fails fast; SHA256 logged; `soul.md` gitignored, `soul.example.md` committed.

## Plan 01-04 — Stream-shaped emit seam (ORCH-06) [CORRECTED]
**Tasks:** `ask()` returns an `AsyncGenerator[OutEvent]` wrapping `graph.astream(..., stream_mode="messages")` (S1) — translate LangGraph stream events → typed `OutEvent`s. **OutEvent taxonomy** (typed stubs now, most no-op in Phase 1): `draft_token`, `approval_requested`, `approved_token`, `redacted`, `rejected`, `final`, `cancelled` — each carrying `output_id` + sequence. **Mock-approval pass-through** wrapping the OutEvent stream (Phase 8 signature, no logic). **G7 cancellation:** `aclose()` cancels the upstream model stream + graph task; `astream` in a `finally`; test proves upstream tasks are gone (not just that tokens stopped). Empty-input guard at the `ask()` boundary (matches the reference). Narrow-except → fallback `OutEvent`, `BaseException` propagates (G3).
**Acceptance:** all output flows through the single async/chunked seam (ORCH-06); mock-approval wrapper in path; OutEvent stubs exist; cancellation test proves upstream cancellation (G7); empty input returns the guard event.

## Plan 01-05 — OpenTelemetry + smoke test (ORCH-05)
**Tasks:** OTel SDK in entrypoint (console exporter local, OTLP toggle via `Settings`; tracing always on). Enable LangGraph's OTel instrumentation; top-level span per `ask`. Configure `W3CTraceContextPropagator` + name the `PropagationHelper` injection point (S3). Smoke test asserts a span per `ask`.
**Acceptance:** every run emits a per-`ask` span (ORCH-05); W3C propagator registered; injection point named for Phase 4.

---

## Offline test seam (reference-confirmed, spans plans)
`ask(..., graph=fake_graph)` accepts an injected graph built on a `FakeModel`/`MockLLM` — ALL unit + integration tests run with no network, no API key (reference pattern). Acceptance: full test suite green offline.

## Explicit Deferrals (placeholder each leaves)
- Durable checkpointer store → **Phase 2** (`InMemorySaver` behind S4).
- Hands contract + typed client → **Phase 3** (S3 makes it cheap).
- Identity *resolution* → **Phase 6** (Phase 1: caller-supplied `identity` dict via S2; no resolution logic).
- Real streaming safety / PII / approval → **Phase 8** (S1 seam + 01-04 mock-approval).
- Tool loop / eval harness → **Phase 5** (Phase 1 graph is tools-free: `chat → END`).

## Reference patterns deferred to later phases (noted, NOT built in Phase 1)
- Per-thread serialization lock (concurrency) → Phase 2.
- "Speaker (verified) / Message (untrusted, verbatim)" turn format (prompt-injection-hardened attribution) → Phase 6 (when real identity arrives). **This is the attribution half of the two-identity split** (authorization = S2 `configurable`, never checkpointed; attribution = this sanitized speaker label written INTO the transcript). It's the piece that actually (a) stops a second caller in a shared thread — e.g. a multi-person guild channel — from inheriting the first caller's identity, and (b) lets the model tell speakers apart. Elevate it to a named Phase-6 deliverable, not a footnote (see ROADMAP Phase 6 note).
- Dangling-tool crash healer → Phase 2 (recovery) / relevant once tools + durable checkpointer exist.

## Resolved / Provisional Decisions
- **soul.md:** public template + gitignored real persona. (Confirmed 2026-06-20.)
- **Framework:** LangGraph (ORCH-01 commits to it).
- **Secrets:** `pydantic-settings`, fail-fast (01-01).
- **S2 channel:** `configurable` + single accessor — **RESOLVED** (external validation complete 2026-06-22). Proven-not-future-proof; `thread_id` rides `configurable` permanently; bounded swap to `context_schema` later via the single-accessor hedge (cheapest pre-Phase-5). Amendments: `config_schema` *declaration* is deprecated (removal v2.0) but we don't declare one; tracing identity is a parallel `metadata` write decoupled from the accessor, so it adds nothing to an A→B swap.

## Three-Brain Review Log → see `01-REVIEW.md`
Gemini (structural) + Codex (adversarial, caught 3 blockers) + current-docs verification (langgraph 1.2.6) + reference-implementation grounding (confirmed fixes; refuted Codex's S2 over-call). All folded into this v2.

## Open Operational Item
gemini-cli auth fixed 2026-06-20 (api-key via `~/.gemini/.env`). RESOLVED.
