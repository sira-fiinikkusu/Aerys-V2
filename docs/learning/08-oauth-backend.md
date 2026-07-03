# LEARNING 08 — the OAuth backend (Max pool, not API tokens)

*2026-07-03. Your June decision ("bank subscription-auth as a V2-Brain design choice"),
landed as ~90 lines.*

## The one-sentence version

`MODEL_BACKEND=oauth` makes `build_model()` return a chat model backed by the Claude
Agent SDK — same auth Kael runs on, zero API tokens for daily conversation — and
nothing else in the system can tell the difference.

## n8n mapping

| n8n | here |
|---|---|
| Swap the credential on the AI Agent node | one Settings field; `build_model()` picks the class |
| Credential = an API key in the vault | "credential" = your Max subscription (Agent SDK → bundled Claude Code CLI) |
| Every tier burns metered tokens | conversation on the pool; **api key stays for evals/CI/fallback** |

## What the adapter deliberately is NOT

The Agent SDK is a whole agent runtime (tools, sessions, MCP). We use it as a **pure
chat backend**: `max_turns=1`, `allowed_tools=[]`. The agent loop belongs to LangGraph
— when tool-calling lands, the SDK-loop-vs-ToolNode fork gets decided on purpose
(flagged in CROSS-REVIEW), not inherited by accident.

## The discovery that simplified deployment

The Python SDK ships a **bundled CLI** (`claude_agent_sdk/_bundled/claude`) — no node,
no npm, no image surgery. The Jetson container needs exactly one new thing: an auth
token (`claude setup-token` → `CLAUDE_CODE_OAUTH_TOKEN` in the .env).

## Proof run (Tachyon, 2026-07-03 9:11am)

Placeholder API key + `MODEL_BACKEND=oauth` → *"I'm Aerys — your personal AI
companion…"* — a real reply with an unusable API key in the slot. The wallet it
billed: your subscription.

## Watch-fors

Shared rate windows (you + Kael + her on one Max pool); transcript-style history (the
SDK takes one prompt, so history rides speaker-labeled — same pattern as thread_context
snippets); latency slightly higher than raw API (CLI spawn per turn — fine for chat,
measure before voice).
