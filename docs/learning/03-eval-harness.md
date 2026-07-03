# LEARNING — the eval harness: Targets, a judge, and the migration's measuring stick

*2026-07-02. Teaches `src/aerys_v2/evals/` (runner.py + `__init__.py`), the `--eval` CLI
path, and `tests/test_evals.py`. This is the Python port of the n8n "06-01 Eval Suite"
workflow (g74yHFlCeOf8kxui) — same rubric, same failure semantics, new spine.*

## The one-sentence version

The eval harness runs a dataset of real prompts through *anything that can answer*
(`Target`), has a model score each reply against a rubric (`Judge`), and rolls the
scores into per-category numbers — which makes it the instrument that will tell us,
in numbers, whether V2 actually matches V1 as each capability migrates.

## The whole package, node by node

| n8n node (06-01 Eval Suite) | evals package | what changed |
|---|---|---|
| Load Dataset (Code node, `readFileSync` off a docker volume) | `load_cases()` | crashed if the volume wasn't mounted → now falls back to the committed `example.json`, so CI is always green |
| Execute Workflow → Core Agent | `Target` protocol / `LocalGraphTarget.respond()` | the hop is now *pluggable* — see below, this is the big one |
| Build Judge Request (Code node) | `JUDGE_SYSTEM_PROMPT` / `JUDGE_USER_TEMPLATE` constants | rubric extracted VERBATIM — same 1-5 scale, same wording |
| Parse Score (Code node) | `Judge.score()` + `_parse_judge_output()` | bit-for-bit tolerances: strip ```` ```json ```` fences, unparseable → 3, call failure → 0 |
| SplitInBatches loop + staticData accumulator | `run_eval()` plain for-loop with a list | no staticData reset dance, no microsecond-truncation high-water-mark bug |
| Format Report | `summarize()` + `format_summary_table()` | same avg/min/max per category + latency; the dict is the artifact, the table is cosmetic |

## Target protocol — pointing the same test at two stacks

In n8n, the eval suite could only ever test the n8n Core Agent — the Execute Workflow
node is hard-wired to a workflow ID. If you wanted to compare two implementations,
you'd build a second eval workflow.

Here, the thing being tested is behind a `Protocol` (Python structural typing — "any
object with a `name` and a `respond(case) -> TargetResponse` method qualifies," no
inheritance required, same duck-typing spirit as n8n not caring what's inside a
sub-workflow as long as the output shape matches):

| Target | What it evaluates |
|---|---|
| `LocalGraphTarget` | the real V2 graph through the `ask()` seam — one isolated `thread_id` per case (`eval-{case.id}`), exactly like the n8n harness running every case as a fresh session so case 7's history can't contaminate case 8 |
| `N8nBaselineTarget` | the live n8n V1 — **deliberately a stub** (see deferred section) |
| (future) a replay target | a recorded dict of V1's answers, replayed offline |

Same dataset, same judge, same report format — only the target swaps. That's the whole
trick: `run_eval(LocalGraphTarget(...), cases, judge)` vs
`run_eval(N8nReplayTarget(...), cases, judge)` produces two directly comparable tables.

One wrinkle worth knowing: the golden set includes an empty-input edge case (tc-19).
V2's `ask()` rejects empty input at the seam with a `ValueError` — that IS the system's
designed behavior, so `LocalGraphTarget` catches it and hands
`"[rejected at the ask() seam: ...]"` to the judge as the reply. It gets *scored* as
graceful handling, not miscounted as a score-0 infrastructure failure.

## Judge-as-model vs eyeballing

The alternative to a judge is you, reading 25 replies, deciding "yeah that sounds like
Aerys." That doesn't scale, drifts with your mood, and can't run unattended. The judge
is a model at `temperature=0` scoring each reply 1-5 against explicit criteria
(`persona_expectations` per case) — cheap, consistent, and repeatable enough to diff
across runs.

The score semantics are load-bearing and copied exactly from the n8n Parse Score node,
so old reports and new reports mean the same thing:

| Score | Means |
|---|---|
| 5-1 | quality verdicts, straight from the rubric |
| 3 (defaulted) | the judge's output wouldn't parse (or score out of range) — one flaky judge reply can't tank or inflate a category |
| **0** | **infrastructure failure** (judge call died, or the target itself blew up) — never "the answer was bad" |

And zeros are *included* in the averages, same as n8n: a run with broken plumbing
should look bad in the report, not be quietly excluded from it.

The judge's model is **injected** (`Judge(model)`) — the same seam idea as
`build_graph()` taking a `BaseChatModel`. Tests hand in `GenericFakeChatModel` (which
plays both roles: the target's brain AND a judge that emits parseable rubric JSON);
production uses `Judge.from_settings()` (temp 0, max_tokens 500 — the n8n judge node's
settings, but the model id follows Settings instead of being hard-coded).

## Why golden.json is gitignored

The golden dataset is the owner's *real* DM prompts — personal data. aerys-v2 is a
public repo with a hard boundary (zero personal/employer content), so:

- `evals/cases/*` is gitignored; only the sanitized `example.json` is committed.
- `load_cases()` prefers `golden.json` when present (your machine), falls back to
  `example.json` (fresh clone, CI) — so CI always has at least one case to chew on and
  the harness works with zero personal data present.
- The judge rubric lives **in code** as constants, not in a `judge_rubric.md` next to
  the cases — because that directory is gitignored, and a fresh clone still needs the
  rubric to function.

n8n contrast: the V1 dataset lived on a docker volume outside the workflow — same
"data stays out of the code" instinct, but with a crash instead of a fallback when the
volume was missing.

## How this becomes the parity instrument for the whole migration

The migration's core question, repeated per capability: *does the Brain answer as well
as n8n did?* This harness is how that stops being vibes:

1. **Capture the V1 baseline** (supervised — see deferred) → judge it → per-category table.
2. **Run V2** on the identical cases with the identical judge → per-category table.
3. **Diff the tables.** Category parity within tolerance = cut over (flip the writer
   lease); a regression = a number pointing at exactly which category, which cases.
4. As tools/memory/sub-agents land in V2, add cases per capability — the report grows
   a row, and every later change re-verifies every earlier capability for free.

Same instrument, both stacks, before/after every cutover. That's the plan.

## Try it yourself

```bash
cd ~/projects/aerys-v2

uv run pytest tests/test_evals.py -q      # the whole loop offline — fakes both ends, no key
uv run aerys-v2 --eval                    # the REAL thing: golden.json (or example.json)
                                          # through the live graph, judged by a real model
                                          # call. Needs ANTHROPIC_API_KEY in .env — spends
                                          # tokens: ~2 calls per case (target + judge).
```

The `--eval` output is one line per case (`[score] id (category, latency) — reasoning`)
followed by the same fixed-width category table the n8n workflow used to post.

## What's deliberately NOT here yet

- **`N8nBaselineTarget` is a stub** that raises `NotImplementedError` — honestly, not
  lazily. There is no unsupervised way to drive the n8n Core Agent from here: community
  edition has no execute-via-API endpoint, and temp webhook workflows 404 on this
  instance (documented CLAUDE.md quirk). Capturing the baseline is a *supervised* step —
  run the cases through Discord/Telegram or trigger 06-01 manually in the UI, record the
  replies — and then a simple dict-lookup replay target replaces the stub.
- **No regression gates.** The harness reports numbers; it doesn't yet fail a build when
  a category drops. Thresholds come once there's a captured baseline to threshold against.
- **No result persistence.** Reports print and vanish; wiring runs into `v2_turns` /
  a results table is a Phase 2 conversation.
- **The golden set is small and static** (~25 cases from the n8n suite). Per-capability
  case growth happens as each migration wave lands.
