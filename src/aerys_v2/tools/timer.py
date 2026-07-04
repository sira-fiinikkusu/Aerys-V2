"""timer — the TIMER TOOL: start/cancel a NATIVE Home Assistant assist timer on
the ORIGINATING voice device, so the satellite's LED wheel spins and it rings
locally — identical UX to asking the Voice PE directly.

Why this exists (the honest gap Aerys flagged out loud, on voice, 2026-07-04):
the Voice PE satellite runs HA's native timer intent LOCALLY — ask it "set a 5
minute timer" and it acks, spins its LED wheel, and rings on-device. But the
Aerys brain INTERCEPTS the conversation (ha_custom_components/aerys_conversation
→ /ask) BEFORE HA's built-in intent handler ever sees the turn, so the brain had
no way to set a timer at all. She noticed the wall and said so. This tool is the
fix: it calls back into HA to start a native timer on the SAME device the user
spoke to.

n8n mapping: there is none — V1 never had a timer capability. This is net-new,
the first tool that reproduces a built-in HA voice behavior from the brain side.

How the LED wheel gets to spin (HA 2026.7 contract, confirmed read-only against
HA Green): HA's `POST /api/intent/handle` accepts a `device_id` and forwards it
into the intent, and the `HassStartTimer` handler REFUSES (TimersNotSupportedError)
unless that device_id is a registered timer device — i.e. exactly a satellite that
shows the ring. So targeting the originating device_id is both what makes the
wheel spin AND a built-in honesty gate: a non-satellite device_id fails loudly,
never silently pretends. The device_id rides the per-call identity that the
aerys_conversation component already threads through /ask (state.Identity.device_id),
and this tool reads it via RunnableConfig injection — hidden from the model, so
the model never has to (and never can) guess which device it's on.

Contracts every tool here obeys (same as tools/home_control.py, tools/web_search.py):

1. HONEST FAILURE — every error path returns a plain STRING the model must relay.
   NEVER raise out of a tool: an exception inside a ToolNode kills the whole
   action turn (the V1 failed-webhook-kills-execution outage mode). HA
   unreachable, an error IntentResponse, an unparseable duration — all come back
   as honest strings.
2. SILENT-SUCCESS ALIGNMENT — a successful native start/cancel returns a string
   that leads with WRITE_OK_PREFIX ("Done:"), the same signal home_control uses,
   so service.py's silent-success rule skips the redundant spoken follow-up: the
   LED wheel spinning IS the feedback, exactly like the light changing is.
3. GRACEFUL DEGRADE — on the text/DM path there is no originating satellite
   (device_id absent). The tool can't show a ring there and says so honestly; if
   an ha_timer_fallback_entity is configured it starts that generic (non-visual)
   timer helper as a best-effort fallback and is honest that it won't ring on a
   speaker or show a light.
"""

import logging
import re
from typing import Callable

import httpx
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

from aerys_v2.state import identity_from_config
from aerys_v2.tools.home_control import WRITE_OK_PREFIX

log = logging.getLogger(__name__)

# A timer intent call is quick; a hung HA must fail the turn, never stall the
# caller (same safety-rail reasoning as home_control's 10s client timeout).
TIMER_TIMEOUT_S = 10.0

# The two native intents. HassStartTimer starts a device-scoped assist timer (LED
# wheel); HassCancelTimer cancels the one running on that device.
INTENT_START = "HassStartTimer"
INTENT_CANCEL = "HassCancelTimer"

# Fuzzy phrases STT / the model may hand us that the plain <qty><unit> grammar
# would mis-read ("an hour" inside "half an hour" would wrongly score 1 hour).
# Normalized to canonical <digits> <unit> BEFORE the grammar runs.
_PHRASE_FIXUPS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\ban?\s+hour\s+and\s+a\s+half\b"), "90 minutes"),
    (re.compile(r"\bhalf\s+an\s+hour\b"), "30 minutes"),
    (re.compile(r"\bhalf\s+a\s+minute\b"), "30 seconds"),
    (re.compile(r"\bquarter\s+of\s+an\s+hour\b"), "15 minutes"),
    (re.compile(r"\bquarter\s+hour\b"), "15 minutes"),
    # DIGIT-form '…and a half|quarter' tails. Without these the grammar below
    # matches only the leading '<n> hour(s)/minute(s)' fragment and silently
    # DROPS the fractional tail — '1 hour and a half' would score 3600, a
    # confidently-wrong timer. Normalize the tail to an explicit sub-unit so the
    # whole phrase is accounted for (\1 keeps the author's quantity).
    (re.compile(r"\b(\d+)\s+hours?\s+and\s+a\s+half\b"), r"\1 hours 30 minutes"),
    (re.compile(r"\b(\d+)\s+hours?\s+and\s+a\s+quarter\b"), r"\1 hours 15 minutes"),
    (re.compile(r"\b(\d+)\s+minutes?\s+and\s+a\s+half\b"), r"\1 minutes 30 seconds"),
]

# One quantity (digits/decimal, or "a"/"an" = 1) glued to a unit. Findall-summed,
# so "1 hour 30 minutes", "90 seconds" and glued "1h30m" all parse. The unit is
# closed by a NON-LETTER (end / space / digit / punctuation) rather than \b so
# glued forms like "1h30m" (unit letter followed by a digit) still match, while a
# bare unit letter inside a longer word ("5 monkeys" → the 'm') stays rejected.
_QTY_UNIT = re.compile(
    # The article branch is boundary-anchored (\ban?) so 'a'/'an' only matches a
    # standalone word — without \b the 'an' inside 'human minutes' would glue to
    # the unit and score a bogus 60s. The numeric branch stays unanchored so
    # glued forms like '1h30m' still match.
    r"(?P<qty>\d+(?:\.\d+)?|\ban?)\s*"
    r"(?P<unit>hours?|hrs?|h|minutes?|mins?|m|seconds?|secs?|s)(?![a-z])"
)


def parse_duration(text: str) -> int | None:
    """Natural-language duration → whole seconds (>0), or None if none is found.

    Handles '5 minutes', '90 seconds', '1 hour 30 minutes', '1.5 hours',
    'an hour', 'half an hour', and a bare number (read as minutes — the common
    'set a 5 minute timer' shorthand). Deliberately small: the `duration` value
    is authored by the MODEL from the tool description, not raw STT, so it arrives
    already tidy; the fuzzy phrases are a safety net, not a promise of full NLU.
    """
    if not text:
        return None
    t = text.strip().lower()
    for pat, repl in _PHRASE_FIXUPS:
        t = pat.sub(repl, t)

    total = 0.0
    matched = False
    for m in _QTY_UNIT.finditer(t):
        qty_raw = m.group("qty")
        qty = 1.0 if qty_raw in ("a", "an") else float(qty_raw)
        unit = m.group("unit")
        if unit.startswith("h"):
            total += qty * 3600
        elif unit.startswith("m"):
            total += qty * 60
        else:  # s
            total += qty
        matched = True

    if not matched:
        bare = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*", t)
        if bare:  # a bare number = minutes
            total = float(bare.group(1)) * 60
            matched = True

    if not matched:
        return None
    secs = int(round(total))
    return secs if secs > 0 else None


def _hms(total: int) -> tuple[int, int, int]:
    """seconds → (hours, minutes, seconds)."""
    return total // 3600, (total % 3600) // 60, total % 60


def describe_duration(total: int) -> str:
    """Human 'X hour(s) Y minute(s) Z second(s)' for confirmation strings."""
    h, m, s = _hms(total)
    parts = []
    if h:
        parts.append(f"{h} hour" + ("s" if h != 1 else ""))
    if m:
        parts.append(f"{m} minute" + ("s" if m != 1 else ""))
    if s:
        parts.append(f"{s} second" + ("s" if s != 1 else ""))
    return " ".join(parts) if parts else "0 second"


def _intent_error_message(resp: object) -> str | None:
    """None if the IntentResponse succeeded; an honest message if it's an error.

    /api/intent/handle returns the bare IntentResponse.as_dict(): a top-level
    `response_type` ('action_done'/'query_answer'/'error'), the spoken text at
    speech.plain.speech, and (on errors) a code at data.code. HA catches
    IntentHandleError/MatchFailedError and returns them as an error response with
    HTTP 200, so a non-2xx never fires for those — we read the envelope."""
    if not isinstance(resp, dict):
        # A 200 whose body isn't a JSON object (a bare list, or literal null)
        # can't be a valid IntentResponse. Return an honest failure string rather
        # than let resp.get(...) raise AttributeError out of the tool — contract
        # #1: NEVER raise inside a ToolNode (it would kill the whole action turn).
        return "Home Assistant returned an unexpected response."
    if resp.get("response_type") != "error":
        return None
    speech = (((resp.get("speech") or {}).get("plain") or {}).get("speech") or "").strip()
    code = (resp.get("data") or {}).get("code")
    return speech or code or "the timer intent could not be handled"


def build_timer_tool(
    *,
    base_url: str,
    token: str,
    client: httpx.Client | None = None,
    fallback_entity: str | None = None,
):
    """Close over the HA config and return the LangChain timer tool.

    Everything injectable, same seam as build_home_control_tool: tests pass an
    httpx.Client on a MockTransport; the factory passes settings.ha_base_url /
    ha_token (reused from the home-control half — the timer needs the same HA
    door). fallback_entity is an optional generic `timer.*` helper for the
    no-device (text/DM) path; None = honest refusal there instead.
    """
    base = base_url.rstrip("/")
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    http = client or httpx.Client(timeout=TIMER_TIMEOUT_S)

    def _handle_intent(intent_name: str, data: dict, device_id: str) -> str | None:
        """POST the native intent for `device_id`; return an honest error string,
        or None on success. Never raises — httpx failures become strings."""
        try:
            r = http.post(
                f"{base}/api/intent/handle",
                headers=headers,
                json={"name": intent_name, "data": data, "device_id": device_id},
            )
            r.raise_for_status()
            resp = r.json()
        except (httpx.HTTPError, ValueError) as e:
            return f"Home Assistant is unreachable right now ({e})."
        return _intent_error_message(resp)

    def _fallback_service(service: str, body: dict) -> str | None:
        """Best-effort generic timer helper (no LED wheel). Honest error string
        or None on success; never raises."""
        try:
            r = http.post(
                f"{base}/api/services/timer/{service}", headers=headers, json=body
            )
            r.raise_for_status()
        except httpx.HTTPError as e:
            return f"Home Assistant is unreachable right now ({e})."
        return None

    @tool
    def timer(
        action: str,
        duration: str = "",
        name: str = "",
        config: RunnableConfig = None,
    ) -> str:
        """Start or cancel a kitchen/countdown TIMER on the user's voice device.

        CALL THIS TOOL whenever the user asks to set, start, or cancel a timer —
        e.g. "set a 5 minute timer", "start a timer for 10 minutes", "set a timer
        for an hour and a half", "cancel my timer", "stop the timer". This is a
        real device timer: it shows the ring/LED on the voice satellite and rings
        there when it's done, exactly like asking the speaker directly.

        action: "start" or "cancel".
        duration: for start, the length in plain words — "5 minutes", "90 seconds",
        "1 hour 30 minutes", "an hour". Leave empty for cancel.
        name: optional label the user gave the timer, e.g. "pasta" or "laundry".

        The tool targets the device the user is speaking on automatically — NEVER
        ask which device, and NEVER pass a device or entity id. If the user isn't
        on a voice device (a text/DM chat), the tool will tell you honestly that it
        can't show a ring there; relay that, don't claim a timer is visibly running.
        """
        act = action.strip().lower()
        label = name.strip()
        device_id = identity_from_config(config).get("device_id")

        if act not in ("start", "cancel"):
            return (
                f"Unknown timer action {action!r}. Valid actions: start, cancel."
            )

        # ---- START ----------------------------------------------------------
        if act == "start":
            total = parse_duration(duration)
            if total is None:
                return (
                    "I couldn't tell how long to set the timer for — say a duration "
                    "like '5 minutes', '90 seconds', or '1 hour 30 minutes'."
                )
            desc = describe_duration(total)
            h, m, s = _hms(total)

            if device_id:
                data: dict = {}
                if h:
                    data["hours"] = h
                if m:
                    data["minutes"] = m
                if s:
                    data["seconds"] = s
                if label:
                    data["name"] = label
                err = _handle_intent(INTENT_START, data, device_id)
                if err:
                    # Includes the "device does not support timers" case (a
                    # non-satellite device_id) — relayed honestly, never faked.
                    named = f" '{label}'" if label else ""
                    return f"I couldn't start the{named} timer — Home Assistant said: {err}"
                named = f" '{label}'" if label else ""
                # WRITE_OK_PREFIX => silent-success: the LED wheel is the feedback.
                return f"{WRITE_OK_PREFIX} started a {desc}{named} timer on your device."

            # No originating device (text/DM/CLI): best-effort, and honest.
            if fallback_entity:
                err = _fallback_service(
                    "start",
                    {"entity_id": fallback_entity, "duration": f"{h:02d}:{m:02d}:{s:02d}"},
                )
                if err:
                    return f"I couldn't start the fallback timer — {err}"
                return (
                    f"Heads up — I don't know which voice device you're on, so I "
                    f"can't set a timer that rings on a speaker or shows a light. I "
                    f"started a background {desc} timer instead; you'll only see it "
                    f"in Home Assistant."
                )
            return (
                "I can only set a timer that rings on a voice device, and from here "
                "I can't tell which device you're on — ask me out loud on a voice "
                "satellite and the timer will show and ring there."
            )

        # ---- CANCEL ---------------------------------------------------------
        data = {"name": label} if label else {}
        if device_id:
            err = _handle_intent(INTENT_CANCEL, data, device_id)
            if err:
                return f"I couldn't cancel the timer — Home Assistant said: {err}"
            named = f" '{label}'" if label else ""
            return f"{WRITE_OK_PREFIX} cancelled the{named} timer on your device."

        if fallback_entity:
            err = _fallback_service("cancel", {"entity_id": fallback_entity})
            if err:
                return f"I couldn't cancel the fallback timer — {err}"
            return "Cancelled the background timer in Home Assistant."
        return (
            "I can't tell which voice device you're on from here, so there's no "
            "device timer for me to cancel — ask me on the satellite the timer is "
            "running on."
        )

    return timer


# Type alias kept parallel with home_control.ConnFactory for readers wiring the
# factory (the timer needs no DB seam — no outbox: a timer is HA-durable, not a
# brain write, and HA already owns the countdown state).
TimerBuilder = Callable[..., object]
