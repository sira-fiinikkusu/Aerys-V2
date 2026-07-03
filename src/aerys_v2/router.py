"""Route classifier — one fast model call decides chat vs action, and drafts the ack.

n8n mapping: this is the Core Agent's "Classify Intent" node reborn. V1 asked a
cheap model "which tier?" before every turn; V2 asks a cheaper question — "does
this turn need to TOUCH something?" — and gets a bonus for free: because the
router already read the message, it also drafts the immediately-speakable
acknowledgment for the action path ("[warmly] Getting the office light for you")
so voice never sits silent while the tool loop runs.

The acks are GENERATED per request, never templated — a canned "On it!" heard
five times a day reads as a phone tree, not a companion. The ONLY templated ack
is the degraded path below, which fires exclusively when the router itself is
down or speaking garbage.

Failure direction is deliberate: when the router can't be trusted (exception,
unparseable reply), we fall back to a keyword heuristic and on uncertainty fail
TOWARD the action path — the action subgraph is the audited one (outbox rows,
canary allowlist, honest tool errors), so a misroute there is visible and safe,
while a device command misrouted to chat just gaslights the caller ("done!"
while the light stays off — the exact V1 hallucinated-tool-call failure mode).
"""

import json
import logging
from dataclasses import dataclass
from typing import Callable

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

log = logging.getLogger(__name__)

# Haiku at temperature 0: the decision must be deterministic and fast (~300ms) —
# it sits on the voice hot path, racing the chat generation (see service.py).
ROUTER_MODEL = "claude-haiku-4-5"

# Degraded-path ack ONLY — fires when the router call itself failed, so there is
# no generated ack to use. Normal operation never speaks this string.
FALLBACK_ACK = "On it."

# Heuristic for the degraded path: words that make a message *plausibly* a device
# command OR a live-state question. Deliberately broad — the locked failure
# direction is toward action. State words (charge, battery, temperature...)
# matter as much as command words: a state question answered from chat is a
# guess dressed as an answer (the Jolteon-charge live incident, 2026-07-02).
_DEVICE_WORDS = (
    "turn on", "turn off", "switch on", "switch off", "toggle",
    "light", "lights", "lamp", "switch", "plug", "outlet", "fan",
    "dim", "brighten", "thermostat",
    # live-state readings — questions about these need the action path's tools
    "charge", "battery", "temperature", "how warm", "how cold",
    "locked", "unlocked", "sensor",
)

_ROUTER_INSTRUCTIONS = """\
You are the routing layer in front of Aerys's brain. Read the user's message and
decide which path handles it:

- "action": the message needs to TOUCH or READ the smart home. That means
  controlling a device (lights, switches, toggles) — AND any question whose
  honest answer requires the CURRENT state of a device or sensor: battery or
  charge level, temperature, on/off, open/closed, locked/unlocked, presence,
  location. Phrasing does NOT matter: opinion or speculation wording is still
  "action" when live state is needed to answer. "Do you think the car has
  enough charge to get to Tampa?", "I wonder if the office light is still on",
  "would Jolteon be able to make it there and back?" are ALL "action" — the
  answer depends on a reading only the tools can take.
- "chat": pure conversation — feelings, memories, opinions about the world,
  general knowledge, planning that needs no device reading. "Do you think cats
  love us?" is chat; "do you think the bedroom is too warm?" is action.

If you are unsure whether live state is needed, choose "action" — that path can
read as well as act, and a needless reading is harmless, while a chat answer to
a state question is a guess.

Reply with ONLY a JSON object — no prose, no code fences:
{"route": "chat" or "action", "ack": "<acknowledgment>"}

The ack is what Aerys says OUT LOUD immediately, before the action completes.
Write it fresh for THIS message, in Aerys's voice, referencing what was actually
asked (e.g. for "kill the office light": "[softly] Dousing the office light now").
For read-style questions the ack is a natural check-in (e.g. for "does the car
have enough charge for Tampa?": "[warmly] Let me check her charge").
Short and speakable — one clause. Never a generic canned phrase. For "chat"
routes the ack is ignored; an empty string is fine there."""


@dataclass(frozen=True)
class RouteDecision:
    """The router's verdict: which path, and (for action) what to say right now."""

    route: str  # "chat" | "action"
    ack: str


def plausibly_commands_device(text: str) -> bool:
    """Degraded-path heuristic: does this text look like a device command?

    Keyword match, deliberately trigger-happy — same lesson as the V1 tool
    descriptions ("specificity beats generality"), inverted: when the smart
    classifier is unavailable, the dumb one must over-trigger toward the path
    that can't lie (the tool loop returns honest errors; chat hallucinates).
    """
    lowered = text.lower()
    return any(word in lowered for word in _DEVICE_WORDS)


def fallback_decision(text: str) -> RouteDecision:
    """What we do when the router's answer is unusable: heuristic, biased to action."""
    if plausibly_commands_device(text):
        return RouteDecision(route="action", ack=FALLBACK_ACK)
    return RouteDecision(route="chat", ack="")


def parse_route_reply(raw: str, user_text: str) -> RouteDecision:
    """Strict-parse the router JSON; anything off-contract -> heuristic fallback.

    Tolerates code fences / stray prose by slicing first '{' to last '}' — models
    at temp 0 still occasionally wrap JSON — but the OBJECT itself is validated
    strictly: route must be exactly "chat" or "action", nothing coerced.
    """
    try:
        start, end = raw.find("{"), raw.rfind("}")
        if start == -1 or end == -1:
            raise ValueError("no JSON object in router reply")
        data = json.loads(raw[start : end + 1])
        route = data["route"]
        if route not in ("chat", "action"):
            raise ValueError(f"bad route {route!r}")
        ack = str(data.get("ack") or "").strip()
        if route == "action" and not ack:
            # generated-ack contract broken; degrade the ack, keep the route
            ack = FALLBACK_ACK
        return RouteDecision(route=route, ack=ack)
    except Exception:
        log.warning("router reply unparseable: %.200r — using heuristic", raw)
        return fallback_decision(user_text)


def build_router(model: BaseChatModel, soul: str) -> Callable[[str], RouteDecision]:
    """Wrap a chat model into the (text) -> RouteDecision seam service.py consumes.

    The soul rides in the system prompt so the generated acks sound like Aerys,
    not like a JSON classifier. Injectable model = offline tests use fakes.
    """
    system = SystemMessage(content=f"{soul}\n\n{_ROUTER_INSTRUCTIONS}")

    def route(text: str) -> RouteDecision:
        try:
            reply = model.invoke([system, HumanMessage(content=text)])
        except Exception:
            # a dead router must never take the turn down — fail to heuristic
            log.warning("router model call failed — using heuristic", exc_info=True)
            return fallback_decision(text)
        text_attr = getattr(reply, "text", None)
        raw = text_attr if isinstance(text_attr, str) else str(reply.content)
        return parse_route_reply(raw, text)

    return route


def router_for(settings, soul: str) -> Callable[[str], RouteDecision]:
    """Build the real Haiku router from Settings (the --serve wiring).

    Always the API backend — the router is a metered call regardless of
    model_backend, same Option C rule as the tool model. Tight budgets on
    purpose: it races chat generation on the voice path, so a slow router
    erases its own reason to exist.
    """
    from langchain_anthropic import ChatAnthropic

    model = ChatAnthropic(
        model=ROUTER_MODEL,
        api_key=settings.anthropic_api_key,  # SecretStr — unwrapped only by the client
        temperature=0,
        max_tokens=200,      # one small JSON object; anything longer is wrong
        timeout=10.0,        # slow router = fall back to heuristic, not a stall
        max_retries=1,
    )
    return build_router(model, soul)
