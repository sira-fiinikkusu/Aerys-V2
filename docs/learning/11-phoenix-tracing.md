# LEARNING — wire_tracing(): Phoenix, the Executions tab we had to build (01-05)

*2026-07-03. In n8n, every node run was automatically visible — inputs, outputs,
timings — because the engine recorded them. And then pruned them: the execution list
self-destructs after ~30 days, so "why did she say that in May?" was unanswerable by
June. LangGraph gives us neither the recording nor the pruning: without tracing, a bad
reply is a black box. This doc is one function, one container, and one rule.*

## The one-sentence version

`wire_tracing(settings)` arms OpenInference/OTel once at startup and ships every model
call and graph step as spans to Phoenix on the Jetson — the n8n Executions tab done
properly (queryable, token-counted, no self-destruct timer) — and it is structurally
incapable of taking the brain down, because tracing is a passenger, never the driver.

## The big mapping

| V1 (n8n) | aerys-v2 | what changed |
|---|---|---|
| Executions tab — free, automatic, per-node I/O | Phoenix UI at `http://192.168.1.107:6006` fed by OTel spans | same per-node visibility, plus token counts and latency breakdowns |
| Execution data pruned on a timer — history evaporates | Phoenix's SQLite store on a named docker volume (`phoenix-data`) | traces survive container recreation; retention is OUR decision |
| "Which prompt actually went to the model?" = re-run and hope | every span carries the full prompt/reply | debugging a bad turn = opening its trace |
| 06-03 Central Error Handler — observed failures, never caused them | THE DEGRADE-SAFE RULE (below) | same posture, enforced by structure |

## The degrade-safe rule — the load-bearing part

If Phoenix is down, the endpoint is wrong, or a library import explodes, **the brain
must still serve.** `wire_tracing()` can NEVER raise. Three structural choices enforce
it:

1. **Imports live inside `_install()`.** If the openinference/otel packages are
   missing or broken, the cost is a logged warning — not an ImportError that kills the
   process at module import time.
2. **The whole install is one try/except.** Any failure logs
   `"tracing setup failed — continuing WITHOUT tracing"` and returns False. Loud, but
   swallowed.
3. **`BatchSpanProcessor` exports in the background, off the hot path.** A slow or
   dead Phoenix costs *dropped spans*, never latency on the turn — the ~3.6s voice
   budget never pays for observability.

The real-install smoke check confirmed the runtime half: a dead endpoint still **arms
cleanly** — setup succeeds because nothing connects at setup time, and export failures
at runtime drop spans without blocking turns. The rule holds at both ends.

## The arming pattern — None = structurally OFF

Same move as the Discord token and `memories_database_url`: `otlp_endpoint = None`
means nothing is imported, nothing connects, `wire_tracing` returns False before the
try block. Dev, tests, and CI never touch OTel. Arming is one `.env` line:

```bash
OTLP_ENDPOINT=http://192.168.1.107:6006/v1/traces   # the /v1/traces path matters
```

## Why ONE call covers everything

`LangChainInstrumentor().instrument(...)` hooks LangChain's callback system — and
LangGraph runs on LangChain runnables, so graph steps, model calls, and (from 01-03+)
tool calls all emit spans with **zero per-node wiring**. This is the ask()-as-the-
single-seam dividend again: in n8n, adding observability meant touching every
workflow; here it's one line in the `--serve` path.

`service.name = "aerys-v2"` on the Resource is what Phoenix groups by — a future
second service won't blur into this brain's traces.

## The container (deploy/phoenix.md has the full runbook)

`arizephoenix/phoenix` is multi-arch — arm64 runs natively on the Jetson. Port 6006
serves BOTH the UI and OTLP-over-HTTP ingest (we deliberately don't publish 4317/gRPC
— every unpublished port is one less door). Two non-negotiables:

- **`PHOENIX_ENABLE_TELEMETRY=false`** — Phoenix's anonymous phone-home stays off.
  This is a work-approved CONDITION, not a preference.
- **The trace store is AT DATA SENSITIVITY.** Spans contain full prompts and replies —
  soul.md content, memory context, real conversation text. The `phoenix-data` volume
  is as sensitive as the NAS `aerys` database: LAN-only, never through the tunnel,
  same ask-first destruction rule.

## The tests — proving a negative, offline

Four tests, no Phoenix, no network, no real `instrument()` (that mutates global
LangChain callback state and would bleed into every other test — a
`SimpleNamespace` stands in for Settings):

- unset endpoint → clean False no-op (the dev/test default)
- monkeypatched `_install` that raises → False + the loud log line, never a raise —
  THE rule, under test
- stubbed `_install` → True, endpoint passed through verbatim
- an import-only smoke check of the actual dependency stack — better a test failure
  here than a silent "tracing off" on the Jetson

## Try it yourself

```bash
cd ~/projects/aerys-v2
uv run pytest -q                        # 267 green (4 in test_tracing.py)
uv run pytest tests/test_tracing.py -v  # the degrade-safe rule by name

# on the Jetson (full runbook: deploy/phoenix.md)
docker ps --filter name=phoenix         # Up, port 6006
# then set OTLP_ENDPOINT, restart --serve, look for: tracing armed | otlp=...
```

## What's deliberately NOT here yet

**Sampling** — at personal-assistant volume every turn is traced; a sampler is a
one-line addition if that ever changes. **Span redaction/attributes** — person_id and
thread_id tagging (and deciding what NOT to record) belongs with the guild transport,
when non-owner text starts flowing. **Phoenix's eval features** — our eval harness
(doc 03) stays its own thing for now. **Auth on the Phoenix UI** — accepted
single-operator-LAN posture, same as the n8n UI; the runbook documents the
`PHOENIX_ENABLE_AUTH` switch for the day that changes.
