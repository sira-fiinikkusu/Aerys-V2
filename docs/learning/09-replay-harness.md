# LEARNING — the replay harness: her real traffic as a regression battery

*2026-07-03. Teaches `src/aerys_v2/replay.py`, the `--replay` CLI path, and
`tests/test_replay.py`. Companion to doc 03: the eval harness asks "does V2 answer
*well*?", this one asks the migration question that came first — "does V2 even *accept*
every message shape V1 actually saw in production?"*

## The one-sentence version

Fifty real executions lifted from n8n's execution history — the exact items that hit
the Core Agent and the Voice Adapter — get replayed through V2's `ask()` seam on a
throwaway checkpointer, and pass/fail is simply "a turn came back," which turns V1's
production history into a shape-compatibility regression battery for the new brain.

## Why this exists when the eval harness already does

The eval harness (doc 03) runs *curated* prompts through a *judge*. But curated prompts
are what we *think* traffic looks like. The replay payloads are what traffic *actually
looked like*: the Discord DM shape as the Core Agent's Execute Workflow Trigger really
received it (person_id already stamped by the Identity Resolver, thread_context,
memory_context, cold_start flags…), and the Voice Adapter's fully-enriched item exactly
as it entered Execute Gemini Agent (`message_content` *and* `message_text`,
`_openai_format`, `current_datetime`…). Field bags nobody would have written by hand.

This is a **smoke harness, not a quality harness** — no judge, no rubric, no scores.
It answers: does the seam accept every real shape, does a reply come back, and how fast
(voice cares at ~4s). Quality stays in `evals/runner.py`. The two harnesses are
deliberately different instruments pointed at the same seam.

## The whole module, n8n-side by V2-side

| n8n reality | replay.py | what it does |
|---|---|---|
| Execution history rows (the Core Agent + Voice Adapter runs saved in the `n8n` DB) | `evals/replay/payloads.json` → `ReplayPayload` | 50 captured executions: 47 voice, 3 DM. Gitignored — it's the owner's traffic. `real_text: false` marks payloads whose text was length-preservingly redacted |
| — (n8n has no equivalent) | `load_payloads()` | prefers `payloads.json`, falls back to committed `example_payload.json` — same fresh-clone/CI contract as `load_cases()` in doc 03. Also normalizes `source_execution` (int in the capture, string in the example) so callers never care |
| Normalize Message (Code node), run in reverse | `to_ask_inputs()` | maps a raw channel field bag onto `(text, identity, thread_id)`: person_id → `Identity.user_id` (platform user_id/username as fallbacks for pre-resolution payloads), message_text with message_content fallback (voice set both from the same STT), thread_id always `replay:<payload id>` |
| a Postgres-backed n8n_chat_histories | `build_replay_graph()` | constructs the graph on a **fresh `InMemorySaver`** — there is no parameter through which a Postgres checkpointer could arrive |
| SplitInBatches loop + staticData accumulator | `run_replay()` | plain for-loop; one bad payload records `ok=False` and the loop continues — a 50-payload run reports 1 bad shape, never dies on it |
| Format Report | `summarize_replay()` + `format_replay_summary()` | per-channel counts + latency; failures INCLUDED in averages (broken plumbing looks bad, doesn't vanish) — same philosophy as the eval summarizer |

## The isolation rules — both load-bearing

Replaying captured turns is the one operation in this repo that could *contaminate her
real memory*: 50 payloads full of redacted placeholder text, pushed into durable
conversation threads, would poison the history the real brain reads back. Two
independent walls prevent it:

1. **Throwaway checkpointer by construction.** `build_replay_graph()` builds on a fresh
   `InMemorySaver` and takes no checkpointer parameter at all. The isolation isn't a
   convention or a default someone could override — the Postgres path physically does
   not exist in this function's signature. Everything the replay writes dies with the
   process. (Contrast doc 01, where `build_graph()`'s injectable checkpointer is the
   *feature*; here, removing the injection point is the feature.)

2. **Namespaced thread ids.** `to_ask_inputs()` derives `thread_id` from the payload's
   *own* capture id — `replay:replay-042` — never from the captured `session_key` or
   `channel_id`. So even a hypothetically-durable graph could never address a live
   thread key like `discord:dm:<snowflake>`. Belt and braces.

This is the V1 session-contamination bug class (n8n_chat_histories keyed on person_id,
one shared session), answered the V2 way: not "be careful," but "make the collision
unrepresentable." `test_mapping_thread_id_is_replay_namespaced` and
`test_run_replay_threads_are_isolated_and_namespaced` are the tripwires.

## Where the payloads came from (and why the shapes differ)

The capture pulled saved executions off the live instance: the DM payloads are the
Core Agent's (EfNdaABSJPl6ebFz) trigger input; the voice payloads are the item that
left "Inject Profile Context (Voice)" in the Voice Adapter (IE5u3QgQlfMfTvbT). The two
shapes genuinely differ — voice carries `message_content`, `_openai_format`, and
`current_datetime`; DM carries `guild_id`, `is_mention`, attachment arrays — and *that
difference is the point*. `ReplayPayload.payload` keeps the raw bag untouched; all
mapping decisions live in `to_ask_inputs()` where a test can pin each one.

Privacy handling, layered: most payloads (41 of 50) had their text replaced with
length-preserving redaction before leaving the capture step; the whole file is
gitignored regardless (redacted or not, traffic *shapes* are still the owner's); and
`run_replay()` records `reply_len` instead of reply text, so neither captured input nor
model output ever lands in logs or CI artifacts.

## Try it yourself

```bash
cd ~/projects/aerys-v2

uv run pytest tests/test_replay.py -q     # the whole harness offline — fake model,
                                          # synthetic payloads, no key, no personal data

uv run aerys-v2 --replay                  # the REAL thing: payloads.json (or the two
                                          # committed examples on a fresh clone) through
                                          # a live model. Needs ANTHROPIC_API_KEY —
                                          # spends tokens: 1 call per payload, 50 on
                                          # your machine.
```

`--replay` prints a fixed-width table: per-channel n / ok / fail / avg latency, plus a
TOTAL roll-up. A clean run is 50 ok, 0 fail — and the voice `avg ms` column is the
number to watch against the ~4s budget.

## What's deliberately NOT here yet

- **Only DM and voice shapes.** Guild and Telegram had no saved executions to lift when
  the capture ran — n8n prunes execution history, and those adapters' runs were gone.
  When guild/telegram traffic gets captured (or when those transports go live on V2 and
  can be captured natively), the battery grows two channels; the harness already
  accepts any `channel` string.
- **No judge, on purpose — and that's a limit, not just a scope cut.** A payload can
  "pass" replay with a reply that's wrong, off-persona, or useless. Replay proves the
  pipe; the eval harness proves the water. Read both reports together.
- **No golden-reply comparison.** The capture didn't save V1's *responses*, only its
  inputs — so replay can't diff "what V2 said" against "what V1 said" per payload.
  That's the supervised-baseline problem from doc 03, unchanged.
- **No CI gate on the real capture.** CI only ever sees the two synthetic examples;
  the 50-payload battery runs on the owner's machine, manually. Wiring `--replay` into
  a pre-cutover checklist (or persisting results into `v2_turns`-style tables) is a
  later conversation.
- **Latency numbers are indicative, not budgeted.** The harness records ms but enforces
  nothing — no threshold fails a run yet. The voice budget becomes a real gate once
  there's a stable baseline to threshold against.
