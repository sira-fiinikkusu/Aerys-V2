# Phase 1 — Three-Brain Review Log

**Reviewed:** 2026-06-20. Subject: `01-PLAN.md` (v1 draft).
**Net verdict: NEEDS-REWORK** — three structural blockers in the seam definitions. The seam *intent* is right; the seam *shapes* used LangGraph-API-wrong patterns. Fix before any implementation.

---

## Gemini (long-context structural pass) — DONE
Strong convergence with Kael's read on all four seams. Contributed: the mock-approval task (success criterion #5), build-graph-once sequencing, the `aclose()` backpressure refinement, the secrets-pattern naming. All folded into the v1 draft. **Did NOT catch the LangGraph-API shape errors below** (converged with Kael's same blind spot — which is exactly why the adversarial pass matters).

## Codex (adversarial "find the fatal flaw") — DONE → NEEDS-REWORK

### BLOCKERS
1. **S1 — emit seam shape is wrong.** The draft makes the terminal graph *node* yield output chunks. LangGraph nodes are state-transition functions, not generators; streaming is a **graph-level** concern via `graph.astream(...)` / `astream_events(...)`. Fix: `ask()` wraps `graph.astream(...)` and translates stream events → typed `OutEvent`s. The graph checkpoints state; it does not produce the output stream.
2. **S2 — caller-context channel is wrong.** Draft injects caller context via `RunnableConfig.configurable`. Current LangGraph wants per-run immutable caller context in **typed runtime context** (`context_schema` on the graph + `Runtime`/`ToolRuntime` accessor), distinct from `configurable`. Fix: typed `CallerContext`, passed via `context=`, read by tools via runtime — never model-settable, never in state.
3. **thread_id — hardcoding it recreates the identity-leak landmine IN Phase 1.** `thread_id` is the checkpointer's primary key for conversation history; a hardcoded value makes all callers share one conversation. Fix: `ask()` requires a caller-supplied `session_id` from day one. Acceptance test: two distinct session_ids → fully isolated histories.

### MAJORS
4. **OutEvent taxonomy** — define a minimal event taxonomy now (`draft_token`, `approval_requested`, `approved_token`, `redacted`, `rejected`, `final`, `cancelled`), each carrying `output_id` + sequence. Most are no-ops in Phase 1 but the type stubs must exist, or Phase 8 retrofits status/sequence onto a protocol never designed for it.
5. **G7 backpressure test is too weak** — "closing consumer stops tokens" can pass with a fake while the model HTTP stream / graph task keeps running (zombie). Test must prove *upstream* cancellation: model stream cancelled, background tasks gone, `astream` closed in a `finally`. Plus define SIGTERM behavior for in-flight streams.
6. **S3 trace propagation isn't automatic** — registering `W3CTraceContextPropagator` doesn't make httpx propagate `traceparent`. Define a `PropagationHelper`/injection point now so Phase 4 has a contract to fill.
7. **LangGraph API specifics** — use the `add_messages` reducer (NOT `operator.add`) for the messages field; confirm in-memory checkpointer class name (`InMemorySaver`); verify against the pinned version before coding.

### MINORS
8. Empty/whitespace input guard at `ask()`. 9. `FakeModel`/`MockLLM` injectable test double so tests run offline with no key. 10. soul.md tool-fabrication guard (validate named tools exist; warn, don't crash). 11. Real deployable process: a `__main__.py`/`cli.py` that wires `ask()` to stdin/stdout (or a minimal HTTP endpoint), handles SIGTERM, and is what the Docker HEALTHCHECK actually probes.

## Verification (current official LangGraph docs) — DONE
Pinned against **langgraph 1.2.6**. Confirms Codex's direction (not a stale prior):
- **Caller context (Q2):** typed runtime context is the right home — `StateGraph(State, context_schema=Context)`, `graph.invoke(..., context=Context(user_id=...))`, read via `from langgraph.runtime import Runtime` → `runtime.context.user_id` (and `ToolRuntime` for tools). Static/immutable per run; model never sets it. Sources: docs.langchain.com `concepts/context` + `langgraph/add-memory`.
- Corroborates: graph-level streaming via `astream` (not node-as-generator); `add_messages` reducer; `thread_id` as the persistence key (same id → shared history, different id → isolated). [Pull exact stream_mode values + checkpointer class name into the rewrite.]

---

## Rewrite punch-list (apply to 01-PLAN.md)
- [ ] S1: redefine emit seam — `async def ask(query, session_id, *, caller_context) -> AsyncGenerator[OutEvent]` wrapping `graph.astream(...)`; graph does NOT yield output.
- [ ] S2: typed `CallerContext` via `context_schema`/runtime context; remove the `configurable`-for-identity framing.
- [ ] thread_id: caller-supplied `session_id` from Phase 1; isolation acceptance test.
- [ ] Add the OutEvent taxonomy (type stubs, mostly no-op in Phase 1).
- [ ] G7: strengthen the cancellation test (prove upstream stream/tasks cancelled); define SIGTERM-vs-in-flight-stream behavior.
- [ ] S3: name a propagation injection point/helper now.
- [ ] Reducer `add_messages` (not `operator.add`); confirm `InMemorySaver` name vs pinned version.
- [ ] Minors: input guard; offline FakeModel; soul.md tool-validation warn; real `__main__`/cli entry the HEALTHCHECK probes.

## Resolution
**All punch-list items folded into 01-PLAN.md v2 (2026-06-20).** Codex's BLOCKER 2 (`configurable`) DOWNGRADED after cross-checking a reference implementation — `configurable` is a sound, proven pattern (not a bug); adopted provisionally with the single-accessor guardrail, pending external validation. Streaming seam rebuilt graph-level (`astream`) per current docs since the reference implementation is synchronous. v2 verdict: SOUND-TO-BUILD.
