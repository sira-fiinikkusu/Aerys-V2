# LEARNING — tier routing: the classify sandwich returns (router.py + factory.py)

*2026-07-03. V1's classify → switch → per-tier sub-workflow, folded into one dict lookup —
plus the boot assertions that make the two-database footgun unbootable. Every concept
below is mapped to something you already run in n8n.*

## The one-sentence version

The router's one Haiku call now also grades how much brain the reply deserves
(fast/standard/deep), the chat node picks the model per turn from
`tier_models_for(settings)`, voice stays pinned to standard, and the deep tier spends
against an atomic daily cap in `v2_model_usage` — V1's whole 06-05 tiering architecture,
minus the three sub-workflows and the race condition.

## The sandwich, folded

| aerys-v2 | n8n equivalent | why it's better here |
|---|---|---|
| `RouteDecision.tier` | Classify Intent's tier output | the router already read the message for chat-vs-action; the tier question rides the SAME call — zero extra latency, zero extra spend |
| `tier_models_for(settings)` dict | Load Config's `modelsConfig` + the Execute Sonnet/Opus/Gemini sub-workflows | the sub-workflow split only ever existed because of the n8n task-runner hang (>6 LangChain tools froze Code nodes); Python has no such disease, so it's one graph, one node, one dict lookup |
| `config["configurable"]["tier"]` | the Switch node's route | per-call, never checkpointed — the same channel identity rides (doc 01's S2 rule), so a tier can never contaminate a thread |
| `normalize_tier()` | Parse Classification's `['gemini','sonnet','opus']` safety array | the vocabulary and the normalizer share ONE tuple, so they cannot drift — that array once held a renamed-away `'haiku'` and nobody noticed |

**Tiers are named by ROLE, not vendor** — `fast`/`standard`/`deep`, not
haiku/sonnet/opus. The haiku→gemini rename left a dead name in V1's validation array;
role names survive model swaps by construction.

## Tier is a hint; route is a contract

The two outputs of the router carry different trust levels, deliberately (Chip's
`normalize_tier` doctrine, adopted via the dossier):

- **route** is a correctness input — validated strictly, exactly `"chat"` or
  `"action"`, nothing coerced. A wrong route gaslights the caller.
- **tier** is a hint — unknown/missing/garbage silently normalizes to `standard`. A
  wrong tier costs pennies or a slightly weaker answer, never a wrong route. A garbage
  tier must not throw away a perfectly good route decision, so normalization happens
  inside `parse_route_reply` AND again at the chat node (belt-and-braces: whatever
  reaches config, the node answers with a REAL model and the trace shows which).

The degraded path never spends: `fallback_decision` always emits DEFAULT_TIER — the
heuristic must not burn the rationed tier on a guess.

## ChannelPolicy: voice is pinned

Voice threads never carry a tier — the pin is structural: `_voice_parallel_start`
simply never writes `tier` into config, so the chat node's default (standard) always
applies. Two reasons, both V1 scars: the ~3.6s voice budget can't absorb opus latency,
and fast-tier identity wobbles are exactly what got Haiku demoted in V1 (it called
itself "Claude"). Text threads are where tiering earns: greetings ride fast (pennies),
conversation rides standard, research earns deep.

The backend rule extends the June credit-pool decision: the oauth/SDK client is
single-model, so only the STANDARD tier may ride the subscription pool — fast and deep
are always metered ChatAnthropic. `standard = build_model(settings)` on the oauth
backend, so pre-tier behavior is preserved byte-for-byte.

## The deep tier earns its cap — atomically this time

V1's opus cap (10/day against `aerys_model_usage`) was check-then-increment: two
queries, racy, and it degraded SILENTLY — a documented regret. `deep_gate_for` is one
atomic statement against `v2_model_usage` (migration 003):

```sql
INSERT ... ON CONFLICT (day, tier) DO UPDATE
  SET call_count = call_count + 1
  WHERE call_count < cap
RETURNING call_count
```

No row back = the cap held = `ask()` downgrades the turn to standard AND logs the
downgrade. The gate runs ONLY once a text-thread chat turn actually classified deep —
voice turns and downgrades never burn a credit. Failure direction on DB trouble:
`False` — a broken counter fails toward the cheap tier, never toward uncounted opus
spend. No DATABASE_URL = cap unenforced, but logged at arm time so a metered box
missing its DB is visible, not silent.

## Boot assertions — the two-database footgun, made unbootable

Shipped in the same block because tiering gave the brain its third prod-adjacent write
surface (checkpoints, outbox, now v2_model_usage). `run_boot_assertions()` runs in
`--serve` and `--discord` BEFORE anything binds or connects:

1. **FATAL:** `DATABASE_URL` targeting anything but `aerys_v2` refuses to start with a
   sentence, not a stack trace — pointed at prod `aerys` it would checkpoint V2 threads
   INTO the production database.
2. **LOUD WARNING:** `MEMORIES_DATABASE_URL` targeting `aerys_v2` is the mirror mistake
   — retrieval reads an empty staging DB and every turn quietly knows nothing.
   Survivable (read-only), so warn, don't die.
3. **LOUD WARNING:** duplicate keys in `.env` — dotenv keeps the LAST assignment, so a
   stale line lower in the file silently overrides the one you just edited (the exact
   shape of the 2026-05-05 watchdog .env scare).

n8n mapping: V1 had no equivalent — a misconfigured workflow just ran wrong until
someone noticed. Refusing to boot is the upgrade. The failure direction is the pattern:
write surfaces aimed wrong must never boot; read surfaces aimed odd degrade visibly.

## The tests — 33 new (in this block), all offline

`test_tiering.py` pins: tier parsing + normalization (unknown tier keeps the route),
the degraded path never spending deep, media/CDN heuristics failing toward action, the
router prompt teaching media and tiers, chat turns running on the routed tier through
a real graph with fake models, the cap downgrading + logging, the voice pin, the
no-tier-models backward-compat path, oauth-keeps-standard-on-the-pool, all boot
assertions (accept/refuse/warn/dupes), and both arming halves of `action_tools_for` +
overlay composition.

## Try it yourself

```bash
uv run pytest -q tests/test_tiering.py tests/test_router.py   # the sandwich, pinned
uv run aerys-v2 --serve      # log line: "tiers armed | fast=... standard=... deep=... cap=10/day"
# then: "hey" → fast; "compare these two architectures in depth" → deep (until 10/day)
DATABASE_URL=postgresql://sira@nas/aerys uv run aerys-v2 --serve   # refuses, one sentence
```

## What's deliberately NOT here yet

Per-person or per-channel cap budgets (one global deep counter for now), tier hints
from conversation history (each turn grades alone), a fast-tier identity re-audit
(fast serves only greetings/trivia precisely because of the V1 wobble), the v2_turns
writer persisting the tier decision (the router log IS the persistence until then),
and streaming. The sandwich is back; the rationing stays simple until usage data says
otherwise.
