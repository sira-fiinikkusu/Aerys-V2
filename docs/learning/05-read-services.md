# LEARNING — read services: identity, profile, memory (the READ path lands)

*2026-07-02. Three n8n workflows — roughly 100 nodes of webhooks, IF branches,
Code nodes, and sentinel plumbing — become three small read-only Python modules.
Every concept below is mapped to the V1 workflow it replaces.*

## The one-sentence version

The Core Agent's per-message READ path (who is this → what do I know about them →
what's relevant right now) is now three importable functions that take an injected
connection, and the UNION-ALL sentinels and CTE param wrappers die because a Python
list is allowed to be empty.

## The big mapping

| n8n workflow | ID | aerys-v2 | call shape |
|---|---|---|---|
| 03-01 Identity Resolver | f3eDUPbif0RnhKIn | `identity.resolve_identity(conn, platform, platform_user_id)` | dict or `None` |
| 04-03 Profile API | kQsn28s7NZFvrlfJ | `profile.get_profile(conn, person_id, privacy_context)` | `{"profile": {...}}` |
| 04-02 Memory Retrieval | GXaRTmTCTP9XqQxY | `memory.retrieve_memories(conn, person_id, ...)` + `format_memory_context(rows)` | list of row dicts → prompt block |

No HTTP hop, no Execute Workflow boundary, no webhook. Profile API was literally
`POST /webhook/aerys-profile-api` because the Core Agent could only reach it over
the wire — in Python the "API" is a function call, and the body parsing, the
`person_id`-falls-back-to-`speaker_id` shim, and the respondToWebhook typeVersion
pinning (the "must be 1, not 1.1" quirk) all evaporate. They weren't features;
they were the cost of the wire.

## Why the sentinels die — the core lesson of this port

n8n's foundational rule is "no items, no downstream execution." A Postgres node
returning 0 rows doesn't yield an empty result — it yields *nothing*, and every
node after it silently never runs. That single engine behavior spawned a whole
workaround genus in V1:

| V1 workaround | what it cost | Python replacement |
|---|---|---|
| `UNION ALL SELECT NULL, NULL, ... LIMIT 1` sentinel (identity) | relied on UNION branch *order* so LIMIT 1 picked the real row first — fragile | `row is None` |
| `UNION ALL SELECT NULL,... WHERE NOT EXISTS(...)` sentinel (profile, memory) | duplicated the entire WHERE clause inside the NOT EXISTS | `if not rows:` / empty list |
| `WITH params AS (SELECT $1::uuid AS pid, ...)` CTE wrapper | existed only because n8n's node throws "Syntax error near UNION" when `$1` appears twice across a UNION ALL | psycopg named params (`%(pid)s`) repeat freely |
| `person_id::text` cast in SQL | existed only because IF `notEmpty` fails on UUID-typed values | `str(row[0])` in code |
| downstream "is this the sentinel row?" filters in Code nodes | every consumer had to know the sentinel's shape | gone (one kept as a cheap belt-and-braces check in `format_memory_context`) |

The tests make this a tripwire, not a nostalgia note: `test_resolve_identity_found`
literally asserts `"UNION" not in sql` and `"::text" not in sql`. If someone
reintroduces a sentinel, CI fails.

Memory retrieval collapses further: V1 carried **two whole query variants**
(private/public, differing only in the privacy predicate — duplicated *again*
inside each sentinel's NOT EXISTS, four copies of the same WHERE clause). Now it's
one query with `privacy_level = ANY(%(levels)s)` and a two-line branch that picks
`["public"]` or `["public", "private"]`.

## READ-ONLY BY CONTRACT — the writes we deliberately left behind

This is the honest-deferral section. The V1 Identity Resolver smuggled two writes
into the read path, and porting them faithfully would have ported the bugs:

1. **`DELETE FROM pending_links WHERE expires_at < NOW()`** ran on *every* message,
   even pure lookups. Expiry sweeping belongs in a maintenance cron, not the hot
   path. Not ported.
2. **Create-person-on-miss** did `INSERT persons` + `INSERT platform_identities
   ... ON CONFLICT DO NOTHING` — which leaks an orphan `persons` row when two
   first-messages race. A future **write service** owns this, and must do both
   inserts in one transaction and re-select the winning person_id on conflict.
   Until then, `resolve_identity` returning `None` is the "route to the
   create-person write path" signal.

`test_resolve_identity_miss_returns_none_without_writing` enforces the contract:
exactly one statement runs on a miss, and it starts with `SELECT`.

## The connection is injected — same move as the checkpointer

Every function takes a psycopg-style connection as its first argument (anything
with `.execute()` returning a cursor). No ORM, no global connection, no module
that connects on import. `Settings.database_url` defaults to `None` — DB features
are simply *off* until it's set, and nothing anywhere dials Postgres at startup.
This is the factory.py lesson again: construction is separate from behavior, so
tests inject a `FakeConn` and production injects the real NAS connection, and the
service code can't tell the difference.

The embedder gets the same treatment: `retrieve_memories` takes an `embed` seam
(`text in, vector out`). The V1 workflow called OpenRouter inline via an HTTP
Request node — which, remember, **wipes all item JSON**, forcing the
"Pre-Embed Context" recovery-node pattern. An injected function has no such
problem: the surrounding variables just… stay in scope.

## A latent V1 bug found and fixed by the port

Rows with NULL `m.embedding` produced NULL `combined_score`, and Postgres
`ORDER BY ... DESC` is NULLS FIRST — so memories that were **never embedded sorted
to the TOP of retrieval** in V1. The port adds `AND m.embedding IS NOT NULL` (an
unembedded memory can't be similarity-scored anyway). This is the recurring
pattern from doc 04: rewriting a workflow as testable code is itself an audit.

Two smaller finds in the same vein:
- V1's missing/failed embedding fell through to `'[]'`, which would ERROR in
  Postgres. V2 short-circuits to `[]` without touching the connection.
- The JS memory-line template left an interior double space when
  `source_platform` was missing. Cosmetic, fixed by joining parts with single
  spaces — and pinned by a test so it's a decision, not drift.

## What survives verbatim — because it's policy, not plumbing

The port kills workarounds but preserves *semantics*, character for character:

- **Profile privacy:** P0/P1 sensitivity never leaves the database — only P2/P3.
  Visibility gate: `'all'` always; `'server'` only in public context; `'dm'` only
  in private. Locked claims outrank everything, then confidence DESC NULLS LAST,
  hard cap 15.
- **Display name extraction:** split on the two-char `': '` separator, last
  segment wins — so `"Preferred name: Chris"` → `"Chris"` and values with lone
  colons (URLs, times) survive. Same JS `.split(': ').pop()` behavior.
- **Memory scoring:** `similarity * 0.7 + recency * 0.3`, recency decaying
  linearly to 0 over 30 days, top 20 scored, 5 lines survive dedup into the
  prompt. The weights live as named constants interpolated into the SQL, with a
  `combined_score()` reference implementation so the math is provable offline.
- **JS `Math.round` half-up:** `_age_days` uses `floor(x + 0.5)` because Python's
  `round()` banker's-rounds `.5` to even — a real behavior difference the test
  `test_format_age_rounds_half_up_like_js` pins at exactly 2.5 days.
- Even a weird one: V1 embedded the empty string for empty messages, and V2
  preserves that (`embed('')` is called) rather than silently "improving" it.

## The tests — FakeConn is a pinned Postgres node

`FakeConn` records every `execute(sql, params)` call and replays canned rows —
the exact trick as pinning an n8n Postgres node's output to test the Code nodes
after it. The 25 new tests prove four things without a database or a network:
SQL param shapes, that the quirk-workarounds are dead (grep-the-SQL assertions),
the assembly/formatting rules (category order, dedup-first-wins, cap at 5), and
the scoring math including both clamp ends (future timestamps from clock skew
clamp to 1, not above).

## Try it yourself

```bash
cd ~/projects/aerys-v2
uv run pytest -q                          # 204 green (25 new in test_services.py)
uv run pytest tests/test_services.py -v   # watch the sentinel-killer assertions by name
python3 -c "
from aerys_v2.services.memory import combined_score, RECENCY_WINDOW_S
print(combined_score(0.5, 0))                    # fresh: 0.65
print(combined_score(0.5, RECENCY_WINDOW_S))     # 30d old: 0.35 — pure similarity
"
```

## What's deliberately NOT here yet

The **write paths**: create-person-on-miss, the pending_links expiry sweep, memory
insertion / batch extraction, Guardian — all belong to a future write service with
proper transactions. **Live wiring**: nothing calls these from `ask()` yet, and
`database_url` stays `None` until the real connection lands (NAS Postgres,
Phase 2, alongside the checkpointer swap). The **real embedder**
(`openrouter_embedder`) exists but only parity checks call it — offline tests
inject fakes. One seam at a time.
