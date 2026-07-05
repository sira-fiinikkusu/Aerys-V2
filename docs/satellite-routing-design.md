# Design: Satellite-Routing for Voice Follow-Ups

*2026-07-04. Status: DESIGN — to be built with the owner in one morning session.
No code changed writing this doc; this is the plan to execute against.*

## Problem statement

The owner has three voice I/O devices: a ReSpeaker Lite satellite
(`assist_satellite.aerys_satellite_assist_satellite`, 192.0.2.109), a Home
Assistant Voice PE, and a phone running a headless Myo-triggered satellite app.
When aerys-v2 speaks a *follow-up* (the async spoken result of a device action,
see below), it always announces to one hardcoded entity — regardless of which
device the owner actually spoke to. Speak to the Voice PE, hear the answer out of
the ReSpeaker. Goal: the follow-up always answers on the device that started the
turn.

## 1. Root cause (with file:line)

The turn has two different response paths, and only one of them is broken:

- **The synchronous reply** (the direct answer HA's Assist pipeline plays back)
  is fine today — HA's own pipeline plays a conversation agent's response on
  whichever satellite is currently running that pipeline. This was never
  satellite-blind.
- **The async spoken follow-up** is what's broken. `service.py`'s
  `_voice_parallel_start` → `_complete_action` (`service.py:251-413`) sends an
  ack immediately, runs the action in a background thread, and — only when the
  silent-success rule says to (`_needs_spoken_followup`, `service.py:222-248`) —
  calls `speak_fn(final)` at `service.py:389`. This call has no request to attach
  a device to; it fires long after the HTTP response that started the turn
  already returned.

- `speak_fn` itself is built **once, at process startup**, in
  `factory.py:107-143` (`speak_fn_for`). It closes over a single static
  `entity = settings.ha_announce_entity` (`config.py:82-86`,
  `HA_ANNOUNCE_ENTITY`) and posts to `assist_satellite/announce` with that one
  `entity_id` every time, for every caller, forever. There is structurally no
  per-call parameter — the target is baked in at construction, not passed at
  invocation. **Verified live on the Jetson deploy (`jetson:~/aerys-brain-src/.env`,
  2026-07-04): `HA_ANNOUNCE_ENTITY=assist_satellite.aerys_satellite_assist_satellite`**
  — the ReSpeaker, exactly as the owner described the symptom (speak to the
  Voice PE or phone, hear the follow-up out of the ReSpeaker in the living room).

- Even if `speak_fn` *could* take a target, nothing upstream carries one. HA's
  "Extended OpenAI Conversation" integration talks to aerys-v2 through the
  OpenAI-compatible shim, `transports/http_api.py:68-107` (`openai_compat`).
  That handler extracts only `last_user` text from `body["messages"]`
  (`http_api.py:76-83`) and calls `ask_fn` with a **hardcoded** thread id,
  `"voice:beta"` (`http_api.py:95`) — no device identity anywhere in the OpenAI
  `chat.completions` wire format, and none is extracted even if it were there.

- **The codebase already correctly diagnosed this** — `factory.py:116-124`
  carries a `KNOWN LIMITATION` comment: *"the OpenAI shim never learns WHICH
  satellite a request came from — HA's Extended OpenAI Conversation sends no
  satellite/device identity in the chat-completions payload... the proper fix
  (satellite identity riding the request) lands with the voice-runtime phase."*
  This design is that fix.

- **A second, independent wrinkle — verified live against HA Green
  (homeassistant.local) on 2026-07-04, load-bearing for the morning session:** the
  two ESPHome satellites aren't even both wired to aerys-v2 today. Each
  satellite has its own per-device pipeline selector
  (`select.<satellite>_assistant`), and they're currently split:

  | Satellite | Pipeline selector state | Pipeline → conversation agent |
  |---|---|---|
  | ReSpeaker (`aerys_satellite`) | `select.aerys_satellite_assistant` = **Aerys-beta** | pipeline `01kwm6wg44vkk80m91hfvxcz82` → `conversation.office_aerys_dev` ("Aerys Dev" config entry) → **aerys-v2 Brain** |
  | Voice PE (`home_assistant_voice_0925b6`) | `select.home_assistant_voice_0925b6_assistant` = **Aerys** (stable) | pipeline `01kmxw572s67aay4x2ryhb21sx` → `conversation.extended_openai_conversation_aerys` ("Aerys" config entry) → **n8n Core Agent** (the old V1 path, unrelated to this codebase) |

  So today, speaking to the Voice PE doesn't reach aerys-v2 at all — it's still
  on the n8n-era pipeline documented in the root `CLAUDE.md`'s "Voice Adapter
  Details." This design only fixes routing for turns that reach the aerys-v2
  Brain. **Testing the fix against the Voice PE requires first flipping
  `select.home_assistant_voice_0925b6_assistant` to `Aerys-beta`** (see §4).
  The phone Myo-bridge app already defaults to a pipeline toggle in-app
  ("Aerys (stable)" / "Aerys-beta (v2)") and can select `01kwm6wg44vkk80m91hfvxcz82`
  directly — it reaches aerys-v2 today when the owner picks the beta option.

One-sentence root cause: **HA's Extended OpenAI Conversation speaks OpenAI's
wire protocol to aerys-v2, which has no field for device identity, so the brain
has never been able to know which satellite a voice turn came from — it always
announces to whatever entity is hardcoded in `.env`. Layered on top: only the
ReSpeaker and the phone (when its in-app toggle is set to beta) currently talk
to aerys-v2 at all — the Voice PE is still on the old n8n pipeline until its
per-device selector is flipped.**

## 2. Candidate approaches

### A. Prompt-template hack (stopgap only)

HA's Extended OpenAI Conversation already renders a `current_device_id`
template variable into its configured system prompt. **Verified directly
against the live-installed component on HA Green** (not just upstream docs —
`ssh ha "grep -n device_id /config/custom_components/extended_openai_conversation/*.py"`,
2026-07-04): `conversation.py:259`, inside `_async_generate_prompt` /
`_generate_system_message`, does exactly this —
`"current_device_id": user_input.device_id` goes into the Jinja render
context for the system prompt string. This is stronger confirmation than a
repo citation: it's the actual code this HA instance runs today, and it
independently proves `ConversationInput.device_id` is real, populated, and
already flowing into this integration before any LLM call — it's just never
forwarded into the outbound `messages[]` payload as a structured field, only
usable if the configured prompt TEXT happens to reference `{{ current_device_id }}`.
Append
`Device ID: {{ current_device_id }}` to the prompt text, then regex the value
back out of the system message in `openai_compat`.

- **Pros:** zero new HA components, a two-line change on each side.
- **Cons:** device identity gets smuggled through model-visible freeform text —
  fragile (an owner prompt edit, an HA template re-render, or the model being
  asked to "repeat your instructions" can all leak or break it), and it violates
  a principle this same codebase has already locked in for identity
  (`.planning/ROADMAP.md` Phase 6: *"Policy-by-architecture: control is
  structural, not promptable"*). Only worth building if B can't ship tonight.

### B. Minimal custom HA conversation component — **RECOMMENDED**

HA's own `ConversationInput` dataclass
(`homeassistant/components/conversation/models.py`, confirmed against HA core
source) already carries the answer structurally:

```python
@dataclass(slots=True)
class ConversationInput:
    text: str
    context: Context
    conversation_id: str | None
    device_id: str | None        # <- already here, before any agent runs
    satellite_id: str | None     # <- already here too
    language: str
    agent_id: str
```

Every `ConversationEntity._async_handle_message(user_input, chat_log)` receives
this object directly — `user_input.device_id` is just... there, populated by
the Assist pipeline before the conversation agent is ever invoked. No
extraction, no regex, no HA-side plumbing to build.

Replace "Extended OpenAI Conversation" with a small custom component
(`aerys_conversation`) that does zero LLM work — it forwards
`{text, device_id, conversation_id}` to aerys-v2's `/ask` endpoint (JSON, not
OpenAI's rigid `messages[]` shape) and speaks back whatever `reply` comes back.
This is exactly the "normalize → ask() → reply" shape every other transport in
this codebase already follows (`http_api.py:1-9`'s own docstring) — it just
runs inside HA instead of inside the aerys-v2 process, because only HA sees
`device_id`.

- **Pros:** structural, not stringly-typed; matches the codebase's existing
  identity doctrine; genuinely small (~80-120 lines total across 3 files)
  because aerys-v2 already owns all the intelligence — this component is a pure
  transport, same shape as `transports/discord_gateway.py`.
- **Cons:** a new artifact to install and maintain on HA Green
  (`custom_components/aerys_conversation/`); more of the morning goes into HA
  setup than option A.
- **Not exotic** — this is the standard community pattern for pointing HA's
  Assist pipeline at a private/local LLM backend: copy the official
  `homeassistant/components/openai_conversation` skeleton and swap the outbound
  call (confirmed via a HA community thread doing exactly this — see Sources).

### C. Brain-side device→entity map alone

Considered and rejected as a *standalone* option: without `device_id` reaching
aerys-v2 at all, a mapping table has nothing to key off of. This isn't a third
alternative — it's a required *component* of option B (§3a.3-4 below), not a
substitute for getting device identity into the request in the first place.

### Recommendation

**B**, with the device identity carried on the existing `Identity` TypedDict
(`state.py:16-27`) rather than inventing new `ask()` plumbing — `Identity` is
already `total=False` and already carries channel-derived metadata this same
way (`privacy_context` was added identically). Minimal diff, no new seam shape.

## 3. Implementation steps

### 3a. aerys-v2 changes

**1. `state.py`** — one additive field on `Identity`:

```python
class Identity(TypedDict, total=False):
    ...
    device_id: str   # HA ConversationInput.device_id — the originating satellite
```

**2. `transports/http_api.py`**

- `AskRequest` (line 18-23): add `device_id: str | None = None`.
- `ask_route` (line 109-120): include it in the identity dict, e.g.
  `identity: Identity = {"user_id": http_user_id, "display_name": body.display_name}`
  then `if body.device_id: identity["device_id"] = body.device_id`.
- **Leave `/v1/chat/completions` untouched.** The new HA component targets
  `/ask` directly — it was never constrained to OpenAI's wire format, and `/ask`
  already has the fields we need. No reason to teach the OpenAI shim about
  device_id when we control both ends of a better contract.

**3. `config.py`** — new setting, same CSV convention as `ha_canary_entities`
(`config.py:74`):

```python
# csv of "device_id=entity_id" pairs mapping a ConversationInput.device_id to
# the assist_satellite entity that should speak follow-ups FOR that device.
# Empty or an unmapped device_id falls back to ha_announce_entity (today's
# single-satellite behavior) — never a silent drop.
ha_satellite_map: str = ""
```

**4. `factory.py`**

- New pure parser, mirrors `tools/home_control.py:60-62`'s `canary_set`:

```python
def satellite_map_from(csv: str) -> dict[str, str]:
    """Parse HA_SATELLITE_MAP ('device_id=entity_id,...') into a dict ('' -> {})."""
    pairs = (p.strip() for p in csv.split(",") if p.strip())
    return dict(p.split("=", 1) for p in pairs)
```

- Rewrite `speak_fn_for` (`factory.py:107-143`) — drop the single fixed
  `entity` capture; the returned closure takes the target per call instead:

```python
def speak_fn_for(settings: Settings) -> Callable[[str, str], None] | None:
    """text, entity_id -> HA announce. entity_id is now resolved PER CALL by
    the caller (service.py), not fixed at construction — see
    resolve_announce_entity below. ha_announce_entity remains required to arm
    the feature at all; it's now the fallback default, not the only target."""
    if settings.ha_token is None or settings.ha_announce_entity is None:
        return None
    import httpx
    base = settings.ha_base_url.rstrip("/")
    headers = {"Authorization": f"Bearer {settings.ha_token.get_secret_value()}"}

    def speak(text: str, entity_id: str) -> None:
        r = httpx.post(
            f"{base}/api/services/assist_satellite/announce",
            headers=headers, json={"entity_id": entity_id, "message": text},
            timeout=15.0,
        )
        r.raise_for_status()
    return speak
```

- New tiny resolver, same shape as `_needs_spoken_followup` (`service.py:222`):

```python
def resolve_announce_entity(
    device_id: str | None, satellite_map: dict[str, str], default_entity: str
) -> str:
    """device_id -> its mapped satellite, or the configured default. None/
    unmapped device_id degrades to today's single-satellite behavior."""
    return satellite_map.get(device_id, default_entity) if device_id else default_entity
```

**5. `service.py`**

- Change the `speak_fn` type everywhere from `Callable[[str], None]` to
  `Callable[[str, str], None]` (text, entity_id) — `ask()`'s signature
  (`service.py:116`), its docstring, and `_voice_parallel_start`'s parameter.
- Add a new optional injected callable to `ask()`'s signature, right next to
  `speak_fn` (`service.py:116`), consistent with how `deep_allowed` already
  rides in as a caller-supplied gate:
  `satellite_for: Callable[[str | None], str] | None = None`. Thread it
  through the existing `_voice_parallel_start(...)` call site
  (`service.py:155-158`, where `speak_fn`/`followup_skip_s` already get
  passed) and add it as a new parameter on `_voice_parallel_start` itself
  (`service.py:251-261`) — no other threading needed, since `_complete_action`
  is a nested closure inside `_voice_parallel_start` and automatically sees
  any of its enclosing parameters, exactly like it already sees `speak_fn` and
  `followup_skip_s` today.
- In `_complete_action` (`service.py:365-411`), right before the existing
  `speak_fn(final)` call at line 389:

```python
device_id = real_configurable.get("identity", {}).get("device_id")
entity_id = satellite_for(device_id) if satellite_for is not None else None
if speak_fn is not None and entity_id is not None and (
    failed or _needs_spoken_followup(result_messages, elapsed, followup_skip_s)
):
    try:
        speak_fn(final, entity_id)
    except Exception:
        log.warning("spoken follow-up delivery failed", exc_info=True)
```

  (`real_configurable` is already in scope in this closure — no new parameter
  threading needed beyond `satellite_for` itself.)

**6. `cli.py`** (`--serve` block, `cli.py:176-192`)

```python
satellite_map = satellite_map_from(settings.ha_satellite_map)
speak_fn = speak_fn_for(settings)
satellite_for = (
    (lambda device_id: resolve_announce_entity(
        device_id, satellite_map, settings.ha_announce_entity))
    if speak_fn is not None else None
)
...
app = build_app(
    lambda text, identity, thread: ask(
        graph, text, identity=identity, thread_id=thread,
        router=router, action_graph=action_graph,
        speak_fn=speak_fn, satellite_for=satellite_for,
        followup_skip_s=settings.voice_followup_skip_s,
        deep_allowed=deep_gate,
    ),
    settings.api_token.get_secret_value(),
    owner_person_id=settings.owner_person_id,
)
```

This entire change is **additive and backward compatible**: any caller that
never sends `device_id` (curl, tests, a future non-satellite HTTP client)
resolves to `settings.ha_announce_entity` exactly as today. Zero regression
risk — this is a pure widening of the existing seam.

### 3b. Home Assistant side — new custom component

Create `custom_components/aerys_conversation/` on HA Green (homeassistant.local):

- `manifest.json` — domain `aerys_conversation`, `config_flow: true` (or skip
  config_flow for tonight, see below). No external requirements — `aiohttp` is
  already part of HA core.
- `__init__.py` — config-entry setup; stores `base_url` / `api_token`.
- `conversation.py` — the entire agent:

```python
class AerysConversationAgent(ConversationEntity, conversation.AbstractConversationAgent):
    _attr_supported_features = ConversationEntityFeature.CONTROL

    @property
    def supported_languages(self) -> list[str] | Literal["*"]:
        return MATCH_ALL

    async def _async_handle_message(self, user_input: ConversationInput, chat_log) -> ConversationResult:
        session = async_get_clientsession(self.hass)
        intent_response = intent.IntentResponse(language=user_input.language)
        try:
            resp = await session.post(
                f"{self._base_url}/ask",
                json={
                    "text": user_input.text,
                    "thread_id": "voice:beta",
                    "display_name": "Chris (Voice)",
                    "device_id": user_input.device_id,
                },
                headers={"Authorization": f"Bearer {self._api_token}"},
                timeout=aiohttp.ClientTimeout(total=15),
            )
            data = await resp.json()
            intent_response.async_set_speech(data["reply"])
        except Exception as err:
            intent_response.async_set_error(
                intent.IntentResponseErrorCode.UNKNOWN, f"Aerys is unreachable: {err}"
            )
        return ConversationResult(response=intent_response, conversation_id=user_input.conversation_id)
```

- `config_flow.py` — fastest legitimate path for tonight: a single form asking
  `base_url` (default `http://jetson.local:8300`) and `api_token`. Even faster:
  hardcode both as module constants for the first cut and add a real config
  flow once it's proven working — this is a one-owner LAN box, not a HACS
  release.
- **Install:** drop the folder at
  `/config/custom_components/aerys_conversation/` on HA Green, restart HA, then
  Settings → Voice Assistants → **the "Aerys-beta" pipeline** (id
  `01kwm6wg44vkk80m91hfvxcz82`, currently pointed at conversation agent
  "Aerys Dev" / `conversation.office_aerys_dev`) → Conversation agent → switch
  from "Aerys Dev" to the new agent.
  **Do NOT touch the "Aerys" (stable) pipeline** (id
  `01kmxw572s67aay4x2ryhb21sx`, conversation agent
  `conversation.extended_openai_conversation_aerys`) — that one is still the
  production n8n Core Agent path (per root `CLAUDE.md`'s Voice Adapter
  Details) and is out of scope here. Mixing these two up would silently
  repoint live production voice at the beta brain — verify the pipeline NAME
  and ID before switching, not just "the one called Aerys."
- This mirrors the exact pattern HA community members use to point Assist at a
  private LLM backend: copy `homeassistant/components/openai_conversation`'s
  skeleton, gut the OpenAI call, point it at your own server (see Sources).

### 3c. Device → entity mapping (the actual data)

**Two of the three device_ids are already known — pulled live via the HA
WebSocket API (`config/entity_registry/list` + `config/device_registry/list`)
on 2026-07-04, no guessing needed:**

| Device | `device_id` | `assist_satellite` entity | Area |
|---|---|---|---|
| ReSpeaker ("Aerys Satellite") | `4f23e5d4672b5a56da3566d3522ccae7` | `assist_satellite.aerys_satellite_assist_satellite` | living_room |
| Voice PE ("Aerys-Voice-PE") | `46100e87ff18621ce195fccf903ef049` | `assist_satellite.home_assistant_voice_0925b6_assist_satellite` | office |
| Phone (Myo-bridge app) | **unknown — see §5, likely N/A** | **none registered (confirmed live, see §5)** | — |

`HA_SATELLITE_MAP` in aerys-v2's `.env`, one CSV of `device_id=entity_id`:

```
HA_SATELLITE_MAP=4f23e5d4672b5a56da3566d3522ccae7=assist_satellite.aerys_satellite_assist_satellite,46100e87ff18621ce195fccf903ef049=assist_satellite.home_assistant_voice_0925b6_assist_satellite
```

That's the ReSpeaker and Voice PE done — no log-and-speak trick needed for
those two, they're both static ESPHome devices already in the registry. The
phone is the only one that needs live verification (§5): if the Myo-bridge
app's `assist_pipeline/run` WS call turns out to pass a stable `device_id`
into `ConversationInput.device_id` (unconfirmed — it isn't a registered HA
device today), add a third pair once known. Until then it falls back to
`ha_announce_entity` like every unmapped/missing device_id — never a crash,
possibly just the wrong satellite for that one device.

## 4. What the owner needs to do in the morning

1. Confirm this doc's recommendation (B) before writing code — 5 min read.
2. **aerys-v2 side** (~45-60 min): the six file changes in §3a. All pure/additive;
   can be TDD'd fast — `resolve_announce_entity` and `satellite_map_from` are
   trivial pure-function unit tests, matching existing test style for
   `canary_set`/`_needs_spoken_followup`.
3. **HA side** (~45-60 min): write the 3-file custom component in §3b, install
   on HA Green, switch **the "Aerys-beta" pipeline's** conversation agent from
   "Aerys Dev" to the new component. Double-check the pipeline ID
   (`01kwm6wg44vkk80m91hfvxcz82`) before saving — do not touch "Aerys" (stable).
4. **Flip the Voice PE onto the beta pipeline** (~2 min, easy to forget and
   the whole reason Voice PE testing would otherwise silently no-op): HA
   Settings → the Voice PE device → `select.home_assistant_voice_0925b6_assistant`
   → change from `Aerys` to `Aerys-beta`. Without this the Voice PE never
   reaches aerys-v2 at all (§1) and the fix will look broken when it isn't.
5. **Fill in `HA_SATELLITE_MAP`** using the table already in §3c (ReSpeaker +
   Voice PE device_ids are already known — no lookup needed). Decide on the
   phone per §5 (scope out for v1, or wire the `notify.mobile_app_sm_f966u`
   fallback if there's time).
6. **Test the actual bug**: speak a device-action command ("turn off the office
   light") from the ReSpeaker, then from the (now-flipped) Voice PE, then from
   the phone with its in-app toggle on "Aerys-beta (v2)". Confirm the
   follow-up (when one fires — remember the silent-success rule skips it for
   fast clean writes, `service.py:222-248`) comes back on the SAME device each
   time. Force a slow/failing action if needed to guarantee a follow-up
   actually fires during the test (e.g. temporarily lower
   `voice_followup_skip_s`, or target an entity outside `HA_CANARY_ENTITIES`
   to force the refusal-speaks-always path).
7. Update `CLAUDE.md`'s Voice Adapter Details section and the aerys-v2 CLAUDE.md
   equivalent once verified — this whole section currently describes the old
   single-satellite n8n-era pipeline and will be stale. Also worth a line
   noting which satellite is on which pipeline going forward, since that split
   is exactly what caused today's confusion.

## 5. Uncertain — needs live testing, not more research

- **CONFIRMED (2026-07-04, live against HA Green's `/api/states`): the phone
  has no `assist_satellite.*` entity today.** Only two exist in the entire
  instance — `assist_satellite.aerys_satellite_assist_satellite` and
  `assist_satellite.home_assistant_voice_0925b6_assist_satellite`, both
  `platform: esphome`. The Myo-bridge app talks to HA's
  `assist_pipeline/run` WebSocket command directly as a raw client — it is
  architecturally NOT a registered satellite device, so
  `assist_satellite.announce` has no entity to target for it regardless of
  what `device_id` (if any) rides along in the pipeline run. This is not a
  gap in research; it's a real architecture gap that needs an owner decision,
  not more digging:
  - **Option 1 (recommended for v1):** scope the phone out. Its own
    synchronous reply already plays back fine (the app renders/plays the TTS
    from the same `assist_pipeline/run` response) — only the *async*
    follow-up (slow/failed actions) has nowhere to go, and per the
    silent-success rule (`service.py:222-248`) most fast writes never trigger
    a follow-up anyway. Unmapped device_id (or none at all) falls back to
    `ha_announce_entity` today — meaning a slow/failed phone-originated
    action would currently announce out of the ReSpeaker. Acceptable known
    gap for one morning session; revisit if it's annoying in practice.
  - **Option 2 (if the owner wants it fixed now):** a push notification via
    **`notify.mobile_app_sm_f966u`** — confirmed to exist live (his Fold's
    HA Companion App registration, `device_tracker.chris_fold_7`). Could
    carry the follow-up text as a push, possibly played via Companion App
    TTS/"Announce" support. This is a different code path from
    `assist_satellite.announce` (notify domain, not assist_satellite) and
    would need its own `satellite_for`-style resolver branch — real but
    small extra work, not designed here; flag as a fast-follow if wanted.
  - Whether the Myo-bridge's `assist_pipeline/run` call even passes a stable,
    app-chosen `device_id` (as opposed to `None`) is itself unverified — check
    by adding the temporary INFO log from the old §3c plan (now only needed
    for this one device) and tapping the phone once.
- Whether `assist_satellite.announce` cleanly interrupts/queues correctly when
  a *different* satellite than the one currently active receives it — should be
  unchanged from today's proven single-satellite behavior, but worth a glance
  once multiple targets are in play.
- Config-flow vs. hardcoded constants for the new HA component — pure
  judgment call for the morning; hardcode first, harden later if it survives
  past the beta.
- Whether to formally deprecate `/v1/chat/completions` once the new component
  is live — low urgency; harmless dead code either way if left in place.
- **Function-calling overlap, worth a 2-minute sanity check before assuming
  parity:** Extended OpenAI Conversation supports its own HA
  service-calling ("functions"/"tools" config, `helpers.py:266-272`,
  targets `entity_id`/`area_id`/`device_id` directly). aerys-v2 already has
  its OWN independent action path (`factory.py`'s `action_stack_for`,
  `tools/home_control.py`, gated by `HA_CANARY_ENTITIES`) that the Brain runs
  itself, separate from whatever Extended OpenAI Conversation's config has
  wired up. If the "Aerys Dev" config entry currently has functions/tools
  configured and in active use for device control, swapping it out for the
  dumb-forwarder component in §3b would silently drop that — worth a quick
  check of the "Aerys Dev" entry's options before assuming the new component
  is a like-for-like replacement instead of just a like-for-like replacement
  *for the parts that route through `/ask`*.

## Sources

**Ground truth, verified live 2026-07-04 (stronger than docs — this is the
actual code/config running today, not upstream reference material):**

- `ssh ha "grep -n device_id /config/custom_components/extended_openai_conversation/*.py"`
  — the real installed Extended OpenAI Conversation source on HA Green.
  Confirms `ConversationInput.device_id` is populated and reaches the
  integration (`conversation.py:259`), and confirms it is used ONLY for
  prompt-template rendering and HA-service function-call target resolution
  (`helpers.py:266-272`) — never forwarded into the outbound LLM request body.
- HA REST `/api/states` (via `~/.config/kael/ha-token.txt`) — enumerated every
  `assist_satellite.*` and `media_player.*` entity live: exactly two
  `assist_satellite` entities exist, both `platform: esphome`; no phone/mobile
  satellite entity exists; `notify.mobile_app_sm_f966u` exists as a fallback
  channel.
- HA WebSocket API (`config/entity_registry/list`,
  `config/device_registry/list`, `assist_pipeline/pipeline/list`,
  `config_entries/get`) — the exact `device_id`/`entity_id` pairs in §3c, the
  4 configured pipelines, and confirmation that the ReSpeaker and Voice PE are
  on *different* pipelines/backends today (§1).
- `jetson:~/aerys-brain-src/.env` (the actual deployed config, read-only,
  key-name and single-value check only) — confirms `HA_ANNOUNCE_ENTITY` is
  live-set to the ReSpeaker entity, matching the reported symptom exactly.

**Upstream reference (used to confirm the general HA mechanisms, corroborated
by the live checks above rather than taken on faith):**

- `developers.home-assistant.io/docs/core/entity/conversation/` —
  `ConversationEntity` / `_async_handle_message` contract.
- `homeassistant/components/conversation/models.py` (HA core) —
  `ConversationInput` dataclass shape (`device_id`/`satellite_id` fields).
- `developers.home-assistant.io/docs/core/entity/assist-satellite/` —
  `assist_satellite.announce` / `ask_question` / `start_conversation` actions,
  and confirmation that `announce` accepts `target: {device_id: ...}` directly
  (HA's standard target-resolution layer, not assist_satellite-specific) — so
  no HA-side device_id→entity_id lookup code is required for the announce call
  itself; only aerys-v2's `HA_SATELLITE_MAP` needs one, to decide who deserves
  an announce for a given device_id in the first place. Also documents that
  phone/browser satellites do NOT get an `assist_satellite` entity for free —
  matching the live-confirmed absence above — and that a custom integration
  (e.g. community's `voice-satellite-card-integration`) is the only way to
  change that.
- `github.com/jekalmin/extended_openai_conversation` upstream — corroborates
  the live-grepped behavior above is the current/intended design, not a
  local misconfiguration.
- HA community thread, "Help With Custom Conversation Agent" (r/homeassistant)
  — confirms copying `homeassistant/components/openai_conversation` and
  swapping the backend call is the standard, previously-proven pattern for a
  custom LLM-backed conversation agent (option B's basis).
