# LEARNING — the extraction worker in shadow mode (workers/extraction.py)

*2026-07-03. The memory WRITE path arrives — but pointed at a staging table, not prod.
Every concept below is mapped to something you already run in n8n.*

## The one-sentence version

`workers/extraction.py` is the n8n batch extraction (IfqY4BrhBGeQrcTC) re-run node-for-node
in Python, reading prod conversations READ-ONLY and writing ONLY to `v2_memories_staging` —
so her extractions can be diffed against what n8n wrote to prod memories BEFORE the writer
lease ever flips.

## Why shadow first — the careful way to take over a write path

The read path (05-read-services, 10-memory-context) could go live immediately: a wrong
read costs one bad answer. A wrong WRITE poisons memory forever. So the takeover is
staged like the lease doctrine demands:

1. **Now:** worker writes to `v2_memories_staging` (migration 002), append-only. n8n keeps
   writing prod. Nobody's memories change.
2. **Next:** diff staging against prod memories for the same window — same facts, same
   key_labels, same privacy calls?
3. **Only then:** the `memory_write` lease flips to `'brain'`, n8n's extraction retires,
   and the staging insert is repointed at prod (plus the dedup/soft-delete branch shadow
   mode deliberately drops — comparing RAW extractions is the point of staging).

This is the same lease discipline as `ha_write` in doc 12, minus the beta exception:
memory has no "structurally impossible double-fire" argument, so no exception. Shadow or
nothing.

## The port, node by node

| n8n node (IfqY4BrhBGeQrcTC) | aerys-v2 | what changed |
|---|---|---|
| Read Last Processed (`$getWorkflowStaticData`) | `read_watermark()` → `v2_extraction_watermark` | durable table, not workflow staticData that dies with the workflow |
| Fetch from n8n_chat_histories | `PROD_MESSAGES_SQL` | ported verbatim + `created_at::text` raw string; READ-ONLY conn |
| Group Messages | `group_by_person()` | person_id FIRST, then 20-message batches — the pre-05.1 misattribution lesson stays enforced |
| Build Extraction Request | `build_transcript()` + `EXTRACTION_SYSTEM_PROMPT` | the prompt is ported VERBATIM — it's the contract being shadow-diffed, so it must not drift from what prod runs |
| Call LLM for Extraction | injected `Llm` seam (`openrouter_chat()` live) | Haiku 4.5 via OpenRouter, temp 0.1, same as v1 |
| Parse Observations | `parse_observations()` + `_compose_content()` | fence-stripping, garbage-tolerant: a chatty model returns `[]`, never crashes the batch |
| Embed Observation | injected `Embedder` seam | SAME model as retrieval (`services.memory.EMBED_MODEL`, 1536-dim) — mismatched embed models make cosine distance compare apples to bananas |
| Insert Memory (+ Dedup/Soft Delete) | `INSERT_STAGING_SQL` only | staging is append-only; the whole dedup branch is prod-writer work, deliberately absent |

## A second source: her own conversations

New vs v1: the worker also reads `v2_turns` — the audit spine from doc 02 — so her V2
conversations feed extraction too. Why v2_turns and not the LangGraph checkpoints?
Because identity never lands in graph state (`test_identity_never_lands_in_state`, the
S2 rule from doc 01) — checkpoints are person-blind, so grouping by person_id would be
impossible there. v2_turns carries `person_id + input_text + created_at`: exactly what
extraction needs, and the audit table earns its keep a second time.

Both source queries return the SAME column tuple, so everything downstream is
source-agnostic — adding a third source someday is one more SQL string.

## Three v1 bugs, fixed in the port instead of ported

- **The millisecond watermark bug.** v1 stored `new Date().toISOString()` — JS truncates
  to milliseconds, timestamptz keeps microseconds, so `>` re-matched the same row forever
  (papered over with a +1ms bump hack). V2 stores the RAW Postgres string
  (`created_at::text`) verbatim; it round-trips exactly through `> $1::timestamptz`.
  Bug AND hack deleted. Corollary: the newest row is picked by the real timestamp, never
  by string-max — Postgres's variable-width fractional seconds sort `'.9' > '.15'`.
- **The backlog-skip bug.** v1's `ORDER BY h.id DESC LIMIT 200` fetched the NEWEST 200 —
  with a >200-row backlog the watermark jumped past the overflow and those rows were
  never extracted. Both V2 queries are `ORDER BY created_at ASC`: the watermark only
  advances through rows actually processed.
- **created_at honesty.** Memories get the original message time (the batch's latest
  `created_at_raw`), not `NOW()` — recency scoring needs "when did this happen", not
  "when did the cron run". Staging keeps `shadow_run_at` separately for run provenance.

## The entrypoint — `python -m aerys_v2.workers extraction`

n8n mapping: the Schedule Trigger node. `--once` is a manual Execute Workflow click;
without it, APScheduler runs the pass hourly (the future worker container's PID 1,
separate from the Brain's serve loop). Wiring rule matches factory.py: connections open
in `__main__.py` and are injected — `extraction.py` never connects on its own, which is
exactly why its tests run offline. The prod connection is set `read_only = True`
(belt-and-braces: the SQL is SELECT-only AND the session refuses writes), and the
staging transaction commits on clean exit / rolls back on a mid-batch crash. The
watermark advances only after a batch fully lands — a crash re-runs it, and since
staging is append-only, a re-run at worst duplicates shadow rows, never loses them.

## The tests — 11 new, all offline

`test_extraction.py` pins the load-bearing choices with fake connections and a fake LLM:
grouping by person before extraction, raw-string watermark round-trip,
newest-by-timestamp-not-string, prod conn receives SELECTs only while every write hits
staging, message-time created_at + provenance flow, empty window = no-op (no LLM spend,
watermark untouched), fence-stripping and garbage tolerance, owner-timezone batch dates.

## Try it yourself

```bash
uv run pytest -q tests/test_extraction.py       # 11 green, no network
# one shadow pass against real DBs (needs DATABASE_URL, MEMORIES_DATABASE_URL,
# EMBEDDINGS_API_KEY in .env; migration 002 applied to aerys_v2 first):
uv run python -m aerys_v2.workers extraction --once
```

## What's deliberately NOT here yet

The diff report itself (staging vs prod — the gate for the lease flip), dedup /
soft-delete-replace (prod-writer work), the `memory_write` lease flip, writing to prod
memories, and any second worker. Shadow rows pile up quietly until the comparison says
the port is trustworthy.
