# LEARNING — factory.py, service.py, and the ask() seam (01-02 completion)

*2026-07-02. First doc of the new rhythm: Kael builds, you gate + learn from these. Every
concept below is mapped to something you already run in n8n.*

## The one-sentence version

`factory.py` builds the machine once at startup, `service.py:ask()` is the single door
every channel walks through, and the tests prove the wiring without spending a token.

## factory.py — construction vs behavior

| aerys-v2 | n8n equivalent | why it's better here |
|---|---|---|
| `load_soul(path)` | Load Config reading soul.md via `require('fs')` | Missing file → fallback persona, not a crashed brain |
| `build_model(settings)` | the anthropic credential + model dropdown on an AI Agent node | `timeout` + `max_retries` + `max_tokens` set ONCE, cover every call |
| `build_graph(model, soul, checkpointer)` | the workflow canvas itself | the graph is data you can test; a canvas isn't |
| `chat()` node function | the AI Agent node | receives `state` instead of `$json` — and NOTHING gets stripped (no LangChain context black hole; the #1 n8n quirk just… doesn't exist) |

**The checkpointer is injected** (`InMemorySaver` today, NAS Postgres in Phase 2 — your
call from tonight). Swapping storage never touches graph shape. In n8n terms: imagine
changing where n8n_chat_histories lives without opening a single workflow.

## The identity rule (S2), enforced again

`chat()` reads identity from `config` via your `identity_from_config` accessor — never
from state. Checkpointed identity in a shared thread = user B inherits user A (the
literal Aerys V1 session-contamination bug). The test
`test_identity_never_lands_in_state` makes this a tripwire, not a convention.

## service.py — ask() is the Execute Workflow boundary

Every transport (Discord, Telegram, voice, CLI) will call `ask()` and nothing else.
Anything added here — rails, audit, tracing — covers every channel at once. In n8n the
same fix had to be copy-pasted into each adapter (remember patching retry onto every
Send Discord Message node?).

**Rails (from tonight's Codex+Gemini review):**
- `recursion_limit` = the tool-loop fuse. Dormant until tools land in 01-03, wired now
  so the seam never ships without it.
- `wall_clock_s` = the whole-turn budget. A reply that arrives too late for voice is a
  failure, not a success — so it raises instead of silently degrading.
- Empty input rejected at the door (the n8n UNION-ALL-sentinel class of bug, killed by
  one `if` statement).

## The tests — pinning without spending

`GenericFakeChatModel` = pinning an n8n node's output to test downstream wiring. The 8
new tests prove: reply flows through, one thread accumulates history (checkpointer
replay), threads don't leak into each other, identity stays out of state, soul fallback
works. All offline, no API key, CI-safe.

## Try it yourself

```bash
uv run pytest -q                 # 17 green
uv run aerys-v2 --ask "hello"    # first real Claude call through the seam (needs .env)
```

## What's deliberately NOT here yet

Tools/ToolNode (01-03), the turns audit table + outbox (the review's "migration spine" —
next design doc), Postgres checkpointer (Phase 2), streaming, polisher. One seam at a time.
