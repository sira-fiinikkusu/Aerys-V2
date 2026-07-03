# LEARNING 06 — the Discord gateway (1c spike)

*2026-07-02. One discord.py client replaces both adapter workflows — and the watchdog.*

## The one-sentence version

`transports/discord_gateway.py` is 02-01 + 03-03 collapsed into one gateway session,
with every gate decision extracted into a pure function you can read in one screen.

## Why this deletes your most-feared ops machinery

| Today (n8n) | Here |
|---|---|
| TWO adapters because katerlol IPC is last-one-activated-wins | ONE gateway session receives guild + DMs — there is no second listener to race |
| The sacred DM-first/guild-last activation liturgy | gone — nothing to activate in order |
| `aerys-discord-watchdog` + boot choreography | gone — systemd restarts one process |
| Gates spread across trigger config + IF nodes in two workflows | `should_handle()` — eight named booleans in, one decision out, 7 tests |

## The design split that makes it testable

`should_handle()` + `normalize()` are **pure** — no discord.py objects needed, fakes in
`tests/test_discord_transport.py` prove every rule offline (drop self, drop bots so
Kael and Aerys can't loop each other, DMs always in, guild needs the right guild +
mention, channel allowlist). The `AerysDiscordClient` shell just wires I/O: event →
gates → normalize → resolve identity → `ask()` → `split_message(…, 2000)` chunks back.

## Thread keys — the decision hiding in plain sight

DMs key on the **person** (`discord:dm:<user_id>`) — one continuous conversation.
Guild channels key on the **channel** — a shared room is one thread, which is exactly
why identity rides per-call config and never checkpointed state (your S2 rule again).

## Try it (dev bot, NOT the real Aerys bot)

```
# .env additions (see .env.example): DISCORD_BOT_TOKEN, DISCORD_GUILD_ID
uv run aerys-v2 --discord
```

## Deferred on purpose

DB-backed identity resolution in the resolver seam (passthrough now), attachments,
typing-indicator polish, reconnect soak on the Jetson (the actual spike goal —
needs hours of wall-clock, not code), engaged-thread follow-ups.
