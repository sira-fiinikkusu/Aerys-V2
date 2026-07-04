# LEARNING — the timer tool: closing the gap she flagged out loud (tools/timer.py)

*2026-07-04. On voice, mid-conversation, Aerys said she couldn't set a timer —
an honest "here's a wall I hit" instead of a confident fake. This is that wall,
removed. Every concept below is about one thing: making the brain reproduce a
built-in Home Assistant voice behavior it had been intercepting away.*

## The one-sentence version

The action subgraph gets a `timer` tool that calls back into Home Assistant's
NATIVE assist-timer intent (`HassStartTimer` / `HassCancelTimer`) on the SAME
device the user spoke to — so the satellite's LED wheel spins and it rings
locally, identical UX to asking the Voice PE directly — while degrading honestly
(never a silent pretend) on the text/DM path that has no device.

## The gap, and why it existed

Ask a Voice PE "set a 5 minute timer" and it just works: it acks, its LED wheel
spins, and it rings on-device when the time's up. That's HA's native timer intent
running LOCALLY on the satellite. But Aerys' pipeline puts her IN FRONT of that:
`ha_custom_components/aerys_conversation` is the conversation agent, so every turn
is forwarded to the Brain's `/ask` BEFORE HA's built-in intent handler ever sees
it (that interception is the whole point — it's how she gets to be the one who
answers). The side effect nobody had closed: the built-in timer handler was now
downstream of a fork the turn never took. She could talk about timers; she
couldn't set one. She noticed, and said so.

The fix is not to stop intercepting — it's to give her a hand that reaches back
into HA and starts the native timer herself, on the device the request came from.

## The HA contract (2026.7, confirmed read-only against HA Green)

Two mechanisms exist and they are NOT interchangeable:

- **`timer.*` services** (`timer.start` / `timer.cancel`) drive generic *helper*
  entities. No LED wheel, no on-device ring — a `timer.finished` event and
  whatever automation you wired to it. Fine as a fallback, wrong as the primary.
- **The `HassStartTimer` / `HassCancelTimer` intents** drive DEVICE-SCOPED assist
  timers — the ones that show the ring and ring on the speaker. This is what "set
  a timer" does when you ask the satellite.

The intent is reachable over REST, and two details make it work AND keep it
honest:

1. `POST /api/intent/handle` accepts a top-level `device_id` and forwards it into
   the intent (`intent.async_handle(..., device_id=...)`). That's the hook — the
   originating satellite's device_id becomes the timer's home.
2. `StartTimerIntentHandler` REFUSES (`TimersNotSupportedError`) unless that
   device_id is a *registered timer device* — i.e. exactly a satellite that shows
   a ring. So targeting the origin device_id is simultaneously what makes the
   wheel spin AND a built-in honesty gate: a non-satellite device_id fails loudly
   with an error IntentResponse, never a silent success. We didn't have to invent
   the "can I even show this here?" check — HA enforces it.

`HassStartTimer`'s slots: at least one of `hours` / `minutes` / `seconds`
(positive ints), optional `name`. `HassCancelTimer`: optional `name` / `area` /
`start_*`, and it scopes to the device's running timer via `device_id`. The
response is the bare `IntentResponse.as_dict()` — top-level `response_type`
(`action_done` / `error`), speech at `speech.plain.speech`, an error code at
`data.code`. HA catches handle/match errors and returns them as an error response
with HTTP 200, so the tool reads the ENVELOPE, not the status code.

## How the device gets to the tool — RunnableConfig injection

The tool must target the originating device, but it must NEVER let the model pick
the device (a one-way voice channel can't answer "which device?", and a guessed
id is the V1 hallucinated-entity bug). So the device is not a model-facing
parameter. `timer(action, duration, name, config: RunnableConfig)` — the `config`
arg is INJECTED by LangChain and hidden from the model's tool schema (verified:
the model sees only `action`/`duration`/`name`). The tool reads
`identity_from_config(config).get("device_id")` — the exact same per-call identity
the `aerys_conversation` component already threads into `/ask`
(`state.Identity.device_id`) and that doc's satellite-routing already uses to
resolve WHERE the spoken follow-up speaks. One source of truth for "which device
is this," now feeding both the announce target and the timer target.

The wiring is proven end-to-end offline: `build_action_graph` threads the
per-call config through `ToolNode` into the tool, which fires `HassStartTimer`
with the caller's `device_id` — `test_action_graph_wires_device_id_from_config_into_timer_intent`
pins it.

## Silent-success alignment — the "Done:" prefix is load-bearing

A successful native start/cancel returns a string that leads with
`WRITE_OK_PREFIX` ("Done:") — the SAME signal `home_control` uses, imported from
it so the two can't drift. That's not cosmetic: `service.py`'s silent-success
rule (`_needs_spoken_followup`) skips the spoken follow-up when every tool note
starts with that prefix. The reasoning transfers exactly — a light changing IS
its own feedback, and a timer's LED wheel spinning IS its own feedback. The
router already spoke the ack the caller heard; a second voice reciting "I started
your timer" on top of a visibly-spinning ring is the noise the rule exists to
kill. A FAILURE (device not timer-capable, HA down) returns a non-"Done:" string,
so the rule speaks it — the user must hear when a timer did NOT start.

`raw_reply` vs `emitted_reply` stays correct on the voice-action path with zero
new code: the caller hears the router's ack (`emitted_reply`), the tool's real
outcome becomes the final AIMessage (`raw_reply`), and the audit records both —
the same split doc 16 built, now carrying a timer outcome.

## Fail-open, and the name-match tax paid up front

A raise inside a `ToolNode` kills the whole action turn (the V1
failed-webhook-kills-execution outage, kept dead). So the tool NEVER raises:
httpx failures, an error IntentResponse, an unparseable duration — all return
honest strings the model relays. And the V1 toolWorkflow name-mismatch lesson: the
`@tool` function is named `timer`, the overlay (`TIMER_OVERLAY`) tells the model
to "call the timer tool", and both say `timer` — mismatch there and the model
hallucinates having called a tool that isn't registered.

## Graceful degrade — honest on the text/DM path

Text, DM, CLI: no originating satellite, so `device_id` is absent. There is no
device-local visual timer to start, and the tool says so rather than faking one.
If `HA_TIMER_FALLBACK_ENTITY` names a generic `timer.*` helper, the tool starts
THAT via `timer.start` as a best-effort background timer and is honest it won't
ring on a speaker or show a light. With no fallback configured, it returns a plain
"I can't tell which voice device you're on from here — ask me out loud on a
satellite." Both are strings, both are true, neither pretends. The fallback arms
by config exactly like every other optional half — absent = the honest refusal.

## Natural-duration parsing — small on purpose

`parse_duration` turns "5 minutes" / "90 seconds" / "1 hour 30 minutes" /
"1.5 hours" / "an hour" / "half an hour" / "1h30m" / a bare number (read as
minutes) into whole seconds. It is deliberately NOT a full NLU: the `duration`
value is authored by the MODEL from the tool description, not raw STT, so it
arrives already tidy; the fuzzy phrases ("half an hour") are a safety net, not a
promise. One sharp edge worth the note: the unit is closed by a NON-LETTER
lookahead `(?![a-z])`, not `\b` — `\b` fails between a unit letter and a following
digit, so "1h30m" would silently read as just "30m". The lookahead lets glued
forms parse while still rejecting a bare unit letter buried in a word ("5 monkeys"
→ the 'm' never matches).

## The tests — 40, all offline

`test_timer.py` proves it with a fake HA on `httpx.MockTransport` (no live HA —
same seam philosophy as home_control / the checkpointer / speak_fn): duration
parsing across every natural form (and the None cases); START drives
`HassStartTimer` at `/api/intent/handle` with the injected `device_id` and
zero-slots omitted; compound durations split into `hours`+`minutes` slots; the
`name` slot rides through; a device that isn't timer-capable relays HA's error and
does NOT claim success; HA-unreachable and a raising transport both come back as
honest strings (never an exception); the no-device path degrades to the fallback
helper or an honest refusal; CANCEL hits `HassCancelTimer`; unknown actions and a
missing config never raise; and the end-to-end graph wiring carries `device_id`
from config through `ToolNode` into the native intent.

## Try it yourself

```bash
uv run pytest -q tests/test_timer.py          # 40 green, no HA, no network
```

Live, armed with `HA_TOKEN` (the timer rides the same HA door as home_control),
the shape of the call the tool makes — a native timer on the originating
satellite, the LED wheel her hand now reaches:

```bash
# READ-ONLY confirmation the intent exists (this changes nothing):
curl -s "$HA_BASE_URL/api/services" -H "Authorization: Bearer $HA_TOKEN" \
  | python3 -c "import sys,json;print([d['domain'] for d in json.load(sys.stdin) if d['domain']=='timer'])"
# what a START turn sends (device_id = the satellite the user spoke on):
#   POST /api/intent/handle
#   {"name":"HassStartTimer","data":{"minutes":5},"device_id":"<origin>"}
```

## What's deliberately NOT here yet

- **No status / query intent.** `HassTimerStatus` ("how long's left?") isn't
  wired — start and cancel are the two the gap was actually about.
- **No increase / decrease / pause.** `HassIncreaseTimer` / `HassDecreaseTimer` /
  `timer.pause` exist; they're a follow-on, not this slice.
- **No multi-timer disambiguation of our own.** If a device has several running
  timers and the cancel is ambiguous, HA's own `_find_timer` raises a match error
  and the tool relays it honestly — we don't add a picker.
- **No outbox row.** Unlike a device write, a timer is HA-durable countdown state
  HA already owns — there's no brain-side intent to reconcile, so the timer skips
  the `v2_outbox` write-ahead the home_control write uses.
- **Fallback is opt-in.** `HA_TIMER_FALLBACK_ENTITY` is unset by default; without
  it the text/DM path is an honest refusal, not a silent generic timer.
