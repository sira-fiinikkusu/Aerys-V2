"""Route classifier — one fast model call decides chat vs action, tier, and the ack.

n8n mapping: this is the Core Agent's "Classify Intent" node reborn — the classify
half of V1's classify sandwich (classify → switch → Execute Sonnet/Opus/Gemini
Agent, the 06-05 per-tier sub-workflows). V1 asked a cheap model "which tier?"
before every turn; V2 folds that question INTO the routing call it already makes —
one Haiku round-trip answers "does this turn need to TOUCH something?" AND "how
much brain does the reply deserve?" — and gets a bonus for free: because the
router already read the message, it also drafts the immediately-speakable
acknowledgment for the action path ("[warmly] Getting the office light for you")
so voice never sits silent while the tool loop runs.

The tier is a HINT, not a correctness input (the normalize_tier doctrine):
an unknown or missing tier silently normalizes to
"standard" — a misclassified tier costs pennies or a slightly weaker answer,
never a wrong route. Contrast route, which IS a correctness input and is
validated strictly below.

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

# The RETURN-LOOP contract (owner design, 2026-07-18): the router sees only the
# CURRENT message, so a short follow-up ("yes, go ahead", "what about tomorrow?")
# whose action-ness lives entirely in prior turns is unclassifiable here and lands
# on the chat path — where the model, which DOES see full history, knows the turn
# needs hands. The chat prompt (factory.py capability block) tells it to open its
# reply with this exact token plus one natural handoff line; service.py detects
# the token, discards/patches the chat reply, and re-runs the turn on the action
# graph — the second half of the router's own doctrine ("fail TOWARD the action
# path"), executed late by the one component that had enough context to know.
# One hop only: nothing on the action side knows this token, so escalation can
# never ping-pong. Any emitted text is stripped of the token defensively.
HANDOFF_MARKER = "<<HANDOFF>>"

# The tier vocabulary — V1's gemini/sonnet/opus renamed for what they MEAN, not
# which vendor serves them (the Parse Classification safety-check lesson: the
# 'haiku'→'gemini' rename left a dead name in the validation array; naming tiers
# by ROLE means a model swap never invalidates the contract).
TIERS = ("fast", "standard", "deep")
DEFAULT_TIER = "standard"


def normalize_tier(tier: object) -> str:
    """Any tier value -> a member of TIERS; unknown/missing -> standard.

    The V1 equivalent was Parse Classification's `if (!['gemini','sonnet','opus']
    .includes(...))` fall-through to sonnet — except that array once held a
    renamed-away 'haiku' and nobody noticed. Here the vocabulary and the
    normalizer share one tuple, so they cannot drift.
    """
    return tier if tier in TIERS else DEFAULT_TIER

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

# Media shapes for the degraded path: attachments and "look at this" requests need
# the action path's media tools (analyze_image / read_document / youtube_summary) —
# a chat answer to "what's in this image?" is the V1 hallucinated-vision failure
# (the model pretending it saw the picture). Same failure direction as devices:
# uncertain -> action, where the tools either work or refuse honestly.
_MEDIA_MARKERS = (
    "cdn.discordapp.com/attachments",   # signed Discord CDN attachment URLs
    "media.discordapp.net",             # Discord's other CDN host
    "youtube.com/watch", "youtu.be/",   # video links -> youtube_summary
    ".pdf", ".docx",                    # document extensions -> read_document
    "look at this", "what's in this", "whats in this",
    "read this file", "this image", "this picture", "this screenshot",
    "this video", "this pdf", "this document", "this attachment",
)

# Web-lookup shapes for the degraded path: current-events / "search for" / weather
# asks need the action path's search_web tool — a chat answer to "what's the
# weather this weekend?" is a guess from stale training data (the same
# hallucinated-answer failure as chat-answering an image). Explicit lookup verbs
# and time-sensitive nouns only, to keep timeless general-knowledge chat OUT
# (same over-trigger-toward-action bias as the device/media heuristics, but
# tuned not to swallow "do you think cats love us?"-shaped opinions).
_SEARCH_MARKERS = (
    "search for", "search the web", "look up", "look it up", "google ",
    "find out", "what's the latest", "whats the latest", "latest news",
    "in the news", "current price", "stock price", "exchange rate",
    "weather", "forecast", "who won", "score of", "right now online",
)

# Email / gap-logging shapes for the degraded path — the 2026-07-11 additions.
# Same over-trigger-toward-action bias: a needless action hop costs a tool
# refusal, while a chat route here either hallucinates mail she never read or
# claims a log she never wrote. Discovered the hard way the same day the tools
# shipped: the router's vocabulary is part of every tool's wiring — a tool the
# router can't route to does not exist, no matter how armed the action graph is.
_EMAIL_MARKERS = (
    "email", "e-mail", "inbox", "mailbox", "your mail",
)
_GAP_MARKERS = (
    "log a gap", "log that gap", "log the gap", "file a gap", "log a complaint",
    "file a complaint", "file an issue", "log an issue", "record a gap",
    "for the coding agent", "log it as a gap",
)

# Music shapes for the degraded path — the 2026-07-18 addition (the music tool's
# wiring, same day it shipped: a tool the router can't route to does not exist).
# Same over-trigger-toward-action bias: a needless action hop costs a tool
# refusal, while a chat route here claims playback she never started. "play " is
# deliberately broad (trailing space keeps "player"/"display" out); the chat
# path's return loop catches whatever still slips through.
_MUSIC_MARKERS = (
    "play ", "music", "song", "playlist", "spotify", "album",
    "pause", "skip this", "next track", "next song", "previous track",
    "turn it up", "turn it down", "volume",
    "what's playing", "whats playing", "now playing",
)

_ROUTER_INSTRUCTIONS = """\
You are the routing layer in front of Aerys's brain. Read the user's message and
decide which path handles it:

- "action": the message needs to TOUCH or READ something outside the
  conversation. That means controlling a device (lights, switches, toggles) —
  AND any question whose honest answer requires the CURRENT state of a device
  or sensor: battery or charge level, temperature, on/off, open/closed,
  locked/unlocked, presence, location. Phrasing does NOT matter:
  opinion or speculation wording is still "action" when live state is needed
  to answer. "Do you think the car has enough charge to get to Tampa?",
  "I wonder if the office light is still on", "would Jolteon be able to make
  it there and back?" are ALL "action" — the answer depends on a reading only
  the tools can take.
  MEDIA is "action" too: whenever an attachment or CDN URL appears in the
  message (https://cdn.discordapp.com/attachments/..., media.discordapp.net,
  a .pdf/.docx link, a youtube.com or youtu.be link), or the user asks you to
  look at / read / describe / summarize an image, photo, screenshot, PDF,
  document, or video — you have ZERO eyes without the media tools, and only
  the action path carries them.
  LIVE WEB LOOKUP is "action" too: current events, breaking news, today's
  weather or forecast, sports scores, prices, stock quotes, exchange rates —
  or any "search for / look up / google / find out / what's the latest"
  request, or any fact that could have changed after your training cutoff.
  You cannot know today's world without a web search, and only the action path
  carries the search tool. "What's the weather this weekend?", "search for the
  latest on the merger", "look up who won last night" are ALL "action".
  EMAIL is "action" too: anything about your inbox or mail — did something
  arrive, search/read/summarize an email, draft or send one. "Did the
  confirmation email come in?", "read me that email from the county",
  "send them a reply" are ALL "action" — only the action path carries the
  email tools.
  LOGGING A GAP is "action" too: when the user asks you to log/file/record a
  gap, complaint, issue, limitation, or "note that for the coding agent" —
  the log_gap tool that performs the write lives only on the action path.
  "Log that as a gap", "file a complaint about the lens cutoff" are "action".
  MUSIC is "action" too: playing a song/artist/album/playlist, pausing,
  resuming, skipping, stopping the music, changing volume, or asking what is
  currently playing. "Play some daft punk", "put on my focus playlist",
  "pause the music", "next song", "turn it up" are ALL "action" — the music
  tool lives only on the action path.
- "chat": pure conversation — feelings, memories, opinions about the world,
  timeless general knowledge, planning that needs no device reading, no
  attachment, and no live lookup. "Do you think cats love us?" is chat; "do you
  think the bedroom is too warm?" is action.

If you are unsure whether live state is needed, choose "action" — that path can
read as well as act, and a needless reading is harmless, while a chat answer to
a state question is a guess. The same bias applies to media and to web lookups:
unsure whether an attachment needs the tools or whether an answer needs current
information, choose "action".

Some messages arrive via speech-to-text and can be mangled — words prepended,
dropped, or misheard ("can you play Against the Tide" may arrive as "To play
against the tide."). A message that CONTAINS a command shape — "play <title>",
"turn on/off <thing>", a timer, volume, or lookup ask — is "action" even when
the sentence as a whole reads oddly or poetically. Judge the fragment, not the
grammar.

Also grade how much thinking the reply deserves, as "tier":
- "fast": greetings, one-word acknowledgments, small talk, trivial system
  questions — anything a small model answers perfectly.
- "standard": everyday conversation, questions, code help, creative writing —
  the default. When unsure, say "standard".
- "deep": genuine research or heavy analysis — multi-step reasoning the user
  clearly wants done thoroughly (compare architectures, audit a plan, long
  technical synthesis). Deep is expensive and rationed; reserve it for
  requests that earn it.

Reply with ONLY a JSON object — no prose, no code fences:
{"route": "chat" or "action", "ack": "<acknowledgment>", "tier": "fast" or "standard" or "deep"}

The ack is what Aerys says OUT LOUD immediately, before the action completes.
Write it fresh for THIS message, in Aerys's voice, referencing what was actually
asked (e.g. for "kill the office light": "[softly] Dousing the office light now").
For read-style questions the ack is a natural check-in (e.g. for "does the car
have enough charge for Tampa?": "[warmly] Let me check her charge").
Short and speakable — one clause. Never a generic canned phrase. For "chat"
routes the ack is ignored; an empty string is fine there."""


@dataclass(frozen=True)
class RouteDecision:
    """The router's verdict: which path, how much brain, and what to say right now.

    tier applies to CHAT routes only (the action subgraph runs its own fixed
    tool model) and defaults to standard — the same default the V1 switch fell
    to when Parse Classification saw an unmapped intent.
    """

    route: str  # "chat" | "action"
    ack: str
    tier: str = DEFAULT_TIER  # "fast" | "standard" | "deep" — a hint, pre-normalized


def plausibly_commands_device(text: str) -> bool:
    """Degraded-path heuristic: does this text look like a device command?

    Keyword match, deliberately trigger-happy — same lesson as the V1 tool
    descriptions ("specificity beats generality"), inverted: when the smart
    classifier is unavailable, the dumb one must over-trigger toward the path
    that can't lie (the tool loop returns honest errors; chat hallucinates).
    """
    lowered = text.lower()
    return any(word in lowered for word in _DEVICE_WORDS)


def plausibly_references_media(text: str) -> bool:
    """Degraded-path heuristic: does this text carry an attachment or media ask?

    Same over-trigger bias as the device heuristic: a needless hop through the
    action path costs one tool refusal; a chat route on an image is the model
    describing a picture it never saw.
    """
    lowered = text.lower()
    return any(marker in lowered for marker in _MEDIA_MARKERS)


def plausibly_wants_web_search(text: str) -> bool:
    """Degraded-path heuristic: does this text want a live web lookup?

    Same over-trigger bias as the device/media heuristics, but deliberately
    tighter — a needless search costs one Tavily call, while a chat answer to a
    current-events question is stale training data dressed as fact. Explicit
    lookup verbs and time-sensitive nouns keep timeless opinion/knowledge chat
    ("do you think cats love us?") on the chat path.
    """
    lowered = text.lower()
    return any(marker in lowered for marker in _SEARCH_MARKERS)


def plausibly_wants_email(text: str) -> bool:
    """Degraded-path heuristic: does this text concern her mailbox?"""
    lowered = text.lower()
    return any(marker in lowered for marker in _EMAIL_MARKERS)


def plausibly_logs_a_gap(text: str) -> bool:
    """Degraded-path heuristic: an explicit ask to log/file a gap or complaint."""
    lowered = text.lower()
    return any(marker in lowered for marker in _GAP_MARKERS)


def plausibly_wants_music(text: str) -> bool:
    """Degraded-path heuristic: does this text want playback control?"""
    lowered = text.lower()
    return any(marker in lowered for marker in _MUSIC_MARKERS)


def fallback_decision(text: str) -> RouteDecision:
    """What we do when the router's answer is unusable: heuristic, biased to action.

    Tier is always DEFAULT_TIER here — the degraded path must never spend the
    rationed deep tier on a guess (fail cheap, same direction as the cap).
    """
    if (
        plausibly_commands_device(text)
        or plausibly_references_media(text)
        or plausibly_wants_web_search(text)
        or plausibly_wants_email(text)
        or plausibly_logs_a_gap(text)
        or plausibly_wants_music(text)
    ):
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
        # tier is a hint (see module docstring): normalize, never reject — a
        # garbage tier must not throw away a perfectly good route decision.
        return RouteDecision(route=route, ack=ack, tier=normalize_tier(data.get("tier")))
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
