"""ask() — the single seam every transport calls.

n8n mapping: this is the Execute Workflow boundary into the Core Agent, done as one
function. Discord, Telegram, the voice endpoint, and the CLI will ALL call ask() and
nothing else — so safety rails, auditing, and tracing added here cover every channel
at once (in n8n the same fix had to be patched into each adapter separately).

TOOLS block (Option C hybrid, owner-ratified): ask() optionally takes a router and
an action subgraph. Both None = chat-only, byte-for-byte the old behavior. Both set:

- non-voice threads: router first (sequential) — chat routes to the chat graph,
  action routes to the tool subgraph, whose result becomes the reply.
- voice turns (identity.voice flag — is_voice_turn): PARALLEL-START — the router and
  the chat generation launch concurrently. Router says chat -> the chat result
  (already in flight) is the reply and the router cost vanishes into the
  latency shadow. Router says action -> the caller gets the router's generated
  ack IMMEDIATELY (speakable now, ~3.6s budget intact) while a background
  thread finishes the action and appends the real result to the SAME thread —
  so the next turn's history shows what actually happened, not just the ack.

TIER ROUTING rides the same router verdict: chat routes on TEXT threads carry a
fast/standard/deep tier into the graph (model picked per turn in the chat node);
voice threads stay pinned to standard (ChannelPolicy, locked), and the deep tier
is rationed by the deep_allowed gate — cap reached means the turn quietly runs
standard and the downgrade is logged, never an error to the caller.
"""

import concurrent.futures
import contextlib
import contextvars
import logging
import re
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Callable
from zoneinfo import ZoneInfo

from langgraph.errors import GraphRecursionError
from langchain_core.messages import AIMessage, HumanMessage

from aerys_v2.factory import LOCAL_FALLBACK_FIRED
from aerys_v2.router import (
    DEFAULT_TIER,
    FALLBACK_ACK,
    HANDOFF_MARKER,
    RouteDecision,
    normalize_tier,
    plausibly_asks_for_action,
)
from aerys_v2.services.content_privacy import (
    CONTENT_PRIVACY_KEY,
    PRIVATE,
    PUBLIC,
    redact_private_history,
)
from aerys_v2.state import Identity, is_lens_surface, is_voice_turn
from aerys_v2.turns import build_turn_row, current_trace_id, extract_tool_calls

log = logging.getLogger(__name__)


# ── per-message content-privacy tagging (the short-term privacy gate's write half) ──
# Every human turn is tagged 'public'|'private' in additional_kwargs (checkpointer-
# persisted) so the chat node's public-room gate (services.content_privacy) can drop
# private DM content. The tag is set SYNCHRONOUSLY here (public channel -> public;
# anything else -> fail-closed 'private') with ZERO added latency, then RELAXED
# off the hot path by an optional judge — see _reclassify_content_privacy.
def _origin_privacy(identity: Identity | dict | None) -> str:
    """The ingest-time tag. A public channel is public-by-origin (no classification);
    a DM / the owner's private channels start fail-closed 'private' and only ever get
    relaxed to 'public' by a judge that has read the actual content."""
    return PUBLIC if (identity or {}).get("privacy_context") == "public" else PRIVATE


def _human_turn(text: str, origin_privacy: str, msg_id: str) -> HumanMessage:
    """A tagged HumanMessage with a STABLE id — the id lets the async judge retag THIS
    exact message later (add_messages replaces by id, in place)."""
    return HumanMessage(
        content=text, id=msg_id, additional_kwargs={CONTENT_PRIVACY_KEY: origin_privacy}
    )


# ── v2_turns audit seam (migration 001; recorder wired by factory.turn_recorder_for) ──
# One row per completed ask() turn, on EVERY completion path — chat, action, voice
# chat, voice background action, and the timeout/error exits. Two hard rules:
#   OFF THE HOT PATH — the row is BUILT synchronously here (so trace_id, tool_calls,
#     and latency are captured with the data in hand and inside the turn's OTel span)
#     but WRITTEN on a daemon thread, so the reply returns to the transport without
#     ever waiting on the NAS insert.
#   FAIL-OPEN — building or writing the row can never disturb the turn; both are
#     wrapped so a DB/serialization failure logs and is dropped (the outbox /
#     extraction graceful contract). record_turn=None (dev/CI, no DATABASE_URL)
#     short-circuits the whole thing.
# Cap concurrent audit writer threads. At personal-assistant volume this is never
# neared; it exists as a fuse for a SLOW/DOWN NAS (cross-review hotpath H/M): without
# it, one thread + one fresh DB connection per turn grow without bound while inserts
# hang, marching toward RLIMIT_NPROC / Postgres max_connections until the hot path's
# own DB access on the shared aerys_v2 instance starts failing. Over the cap we DROP
# the audit write (fail-open) rather than pile up — an audit log may lose a row under
# a NAS outage; the live turn may not.
_MAX_INFLIGHT_AUDIT = 32
_audit_inflight = threading.BoundedSemaphore(_MAX_INFLIGHT_AUDIT)


def _safe_record(record_turn: Callable[[dict], None], row: dict) -> None:
    try:
        record_turn(row)
    except Exception:  # pragma: no cover - recorder is already fail-open
        log.warning("v2_turns record failed — turn not audited", exc_info=True)


def _fire_turn_record(
    record_turn: Callable[[dict], None] | None,
    config: dict,
    text: str,
    latency_ms: int | None,
    **fields: object,
) -> None:
    """Build the audit row now (trace/tool/latency captured in-context), write it
    off the hot path. thread_id + identity are read from the per-call config — the
    same S2 channel the graph uses — so the row can never disagree with the turn."""
    if record_turn is None:
        return
    try:
        configurable = (config or {}).get("configurable") or {}
        row = build_turn_row(
            thread_id=str(configurable.get("thread_id", "")),
            identity=configurable.get("identity") or {},
            input_text=text,
            latency_ms=latency_ms,
            trace_id=current_trace_id(),
            **fields,  # type: ignore[arg-type]
        )
    except Exception:
        log.warning("v2_turns row build failed — turn not audited", exc_info=True)
        return

    # Bounded fire-and-forget. The .start() itself was the ONE audit-path line outside
    # a try/except (cross-review hotpath H): under thread exhaustion Thread.start()
    # raises RuntimeError and, unguarded, that unwinds into the live turn and crashes
    # the reply — the exact opposite of the writer's fail-open contract. Acquire a
    # slot first (drop the write if the fuse is blown), and guard the spawn so a failed
    # start can NEVER reach the caller.
    if not _audit_inflight.acquire(blocking=False):
        log.warning(
            "v2_turns audit DROPPED — %d writes already in flight (NAS slow/down?)",
            _MAX_INFLIGHT_AUDIT,
        )
        return

    def _run() -> None:
        try:
            _safe_record(record_turn, row)
        finally:
            _audit_inflight.release()

    try:
        threading.Thread(target=_run, daemon=True).start()
    except RuntimeError:  # can't start new thread — fail open, never crash the turn
        _audit_inflight.release()
        log.warning("v2_turns audit thread could not start — turn not audited", exc_info=True)


def _record_turn_failure(
    record_turn: Callable[[dict], None] | None,
    config: dict,
    text: str,
    started: float | None,
    exc: BaseException,
    *,
    classifier_intent: str | None = None,
    tier: str | None = None,
    tier_override_source: str | None = None,
    base_degraded: list[str] | None = None,
    emitted_reply: str | None = None,
) -> None:
    """One v2_turns row for a turn whose invoke RAISED, fired before the caller
    re-raises OR emits an honest fallback. Degraded marker is 'recursion_limit' for a
    rail trip, else 'turn_failed'; the exception text rides `error`. This is what makes
    the docstring promise — a row on the error exits, not just the timeout exit —
    literally true (cross-review correctness H).

    emitted_reply is set on the FIX-2 path: when the failure is converted to an honest
    rate-limit line the caller returns instead of raising, the row records what the
    user actually heard (no longer NULL) while degraded still carries 'turn_failed'."""
    marker = (
        "recursion_limit" if type(exc).__name__ == "GraphRecursionError" else "turn_failed"
    )
    latency_ms = int((time.monotonic() - started) * 1000) if started is not None else None
    _fire_turn_record(
        record_turn, config, text, latency_ms,
        classifier_intent=classifier_intent,
        tier=tier,
        tier_override_source=tier_override_source,
        extra_degraded=[*(base_degraded or []), marker],
        error=str(exc) or type(exc).__name__,
        emitted_reply=emitted_reply,
    )


# Degrade-safe tracer (same rule as tracing.py: a passenger, never the driver).
# Without a root span at the ask() seam, the parallel-start's worker threads each
# minted their OWN root trace — the router's ack generation showed up in Phoenix
# as an orphan (or not at all) instead of inside the turn. get_tracer before
# wire_tracing() is safe: the proxy resolves the real provider at span time.
try:  # pragma: no cover - import guard
    from opentelemetry import trace as _otel_trace

    _TRACER = _otel_trace.get_tracer("aerys_v2.service")
except Exception:  # pragma: no cover
    _TRACER = None


def _face(
    face_push: Callable[[str, str], None] | None, phase: str, text: str = ""
) -> None:
    """The panel-face seam, double fail-open: FacePusher already swallows its own
    errors, but the seam guards against non-conforming fakes too — her desk face
    is decoration, and decoration never costs a turn anything."""
    if face_push is None:
        return
    try:
        face_push(phase, text)
    except Exception:
        log.debug("face push seam failed (harmless)", exc_info=True)


def _turn_span(thread_id: str, text: str):
    """One span per ask() turn — every model call underneath parents into it."""
    if _TRACER is None:
        return contextlib.nullcontext()
    return _TRACER.start_as_current_span(
        "ask",
        attributes={
            "openinference.span.kind": "CHAIN",
            "input.value": text,
            "thread_id": thread_id,
        },
    )


def _in_ctx(fn: Callable, *args):
    """Bind fn to a COPY of the caller's contextvars (OTel context included).

    threading/ThreadPoolExecutor drop contextvars at the thread boundary, which
    is exactly how the router/chat/action spans fell out of the turn trace.
    One copy per callable — a Context object cannot be entered by two threads.
    """
    ctx = contextvars.copy_context()
    return lambda: ctx.run(fn, *args)


# The reclassify retag is a read-modify-write on a person-keyed thread that is SHARED
# across every surface (DM, guild, telegram, voice) AND every container (soak/telegram/
# brain). A turn that lands between our read and write branches the checkpoint and
# ORPHANS the retag onto a dead sibling (confirmed 2026-07-05: ~95% lost under
# back-to-back turns). Re-read the head and re-apply until the tag is observed on the
# current head, bounded. Human-paced turns leave a gap within a few hundred ms and the
# retag then wins for good — later turns build on the checkpoint that carries it. If it
# never sticks (sustained churn), the message stays 'private' (over-hidden) — fail-safe:
# a lost retag hides a benign thing, it never leaks a private one.
_RETAG_MAX_ATTEMPTS = 6
_RETAG_BACKOFF_S = 0.4


def _retag_landed(graph: object, configurable: dict, msg_id: str) -> bool:
    """True when msg_id is present on the CURRENT head with a 'public' tag."""
    from aerys_v2.services.content_privacy import content_privacy_of

    try:
        msgs = graph.get_state({"configurable": configurable}).values.get("messages", [])
    except Exception:
        return False
    m = next((x for x in msgs if getattr(x, "id", None) == msg_id), None)
    return m is not None and content_privacy_of(m) == PUBLIC


def _reclassify_content_privacy(
    graph: object,
    config: dict,
    msg_id: str,
    text: str,
    reply: str,
    classifier: Callable[[str], str],
) -> None:
    """OFF THE HOT PATH: re-judge a DM turn's CONTENT and, if general, relax its
    fail-closed 'private' tag to 'public' so it may carry into public rooms.

    The judge reads the human turn AND the reply together — so a benign-looking
    question whose ANSWER is private (a balance read, a symptom named back) stays
    private even though the question alone looked general. A 'private' verdict is a
    no-op (the ingest tag is already private); only 'public' rewrites, via update_state
    replacing the message by its stable id (add_messages semantics — content/position
    preserved, only additional_kwargs change).

    Fires on a daemon thread the caller never joins (the same background-update_state
    pattern as the voice _complete_action path). FAIL-OPEN and FAIL-CLOSED at once: any
    trouble — judge error, a thread that won't start, an update_state hiccup — leaves
    the SAFE 'private' tag in place. Worst case a general DM message never carries into
    public (conservative), never a private one leaking."""

    def run() -> None:
        try:
            if classifier(f"{text}\n{reply}") != PUBLIC:
                return  # judge kept it private — the ingest tag already is; nothing to do
            configurable = config["configurable"]
            # Retry-with-verify against the concurrent-turn race (see the constants note).
            for _ in range(_RETAG_MAX_ATTEMPTS):
                graph.update_state(
                    {"configurable": configurable},
                    {"messages": [_human_turn(text, PUBLIC, msg_id)]},
                    as_node="chat",
                )
                if _retag_landed(graph, configurable, msg_id):
                    return  # observed on the head — later turns build on it now
                time.sleep(_RETAG_BACKOFF_S)
            log.warning(
                "content-privacy retag never stuck after %d attempts (thread churn) — "
                "message stays private (fail-safe)", _RETAG_MAX_ATTEMPTS,
            )
        except Exception:
            log.warning("content-privacy reclassification failed — tag stays private", exc_info=True)

    try:
        threading.Thread(target=_in_ctx(run), daemon=True).start()
    except RuntimeError:  # thread exhaustion — never crash the turn over a retag
        log.warning("content-privacy reclassify thread could not start", exc_info=True)


def _reclassify_if_needed(
    graph: object,
    config: dict,
    msg_id: str,
    text: str,
    reply: str,
    classifier: Callable[[str], str] | None,
    origin_privacy: str,
) -> None:
    """Fire the async retag only for a candidate turn: a judge must be wired, and the
    turn must be a fail-closed 'private' DM/voice turn (a public-origin turn is already
    public, nothing to relax). Called from BOTH the non-voice paths and the voice path:
    since person-keying, a private thing said by voice shares the owner's thread with his
    public text turns, so voice content needs the same relax-general/keep-private
    treatment as a DM."""
    if classifier is None or origin_privacy != PRIVATE:
        return
    _reclassify_content_privacy(graph, config, msg_id, text, reply, classifier)


@dataclass(frozen=True)
class Rails:
    """Per-request safety limits (cross-review #13) — enforced at the seam, not by prompts.

    turn_limit went live with the TOOLS block: with tools wired, a confused model
    can loop tool-call → result → tool-call forever; the rail (as LangGraph's
    recursion_limit) makes the 10th hop a hard stop instead of an Opus-budget
    incineration.
    """

    wall_clock_s: float = 90.0
    turn_limit: int = 10


class TurnTimeout(RuntimeError):
    """The whole turn (not just one model call) exceeded its wall-clock budget."""


def _reply_text(message: object) -> str:
    # .text is a property in current langchain-core (calling it is deprecated)
    text_attr = getattr(message, "text", None)
    return text_attr if isinstance(text_attr, str) else str(message.content)


# ── #2 RETURN LOOP: chat→action escalation (owner design, 2026-07-18) ───────────
# The router classifies from the CURRENT message only, so a follow-up whose
# action-ness lives in prior turns ("yes, go ahead" / "what about tomorrow?")
# lands on the chat path — where the model, which sees full history, knows the
# turn needs hands. The chat prompt (factory capability block) has it open such
# a reply with router.HANDOFF_MARKER + one natural handoff line; ask() detects
# the marker and re-runs the turn on the action graph. One hop by construction:
# the action side has never heard of the marker, and every emitted string is
# stripped of it defensively, so escalation cannot ping-pong.

# Degraded markers for the audit pair: the chat row that raised its hand, and
# the action row that did the recovered work.
CHAT_HANDOFF_MARKER = "chat_handoff"
ESCALATED_MARKER = "escalated_from_chat"

# Chat-only surfaces (dev boxes without the TOOLS block; guests with no media
# graph): a handoff has nowhere to go, and emitting the model's "let me get
# that for you" line with nothing behind it would be promise-and-abandon — the
# exact dead end the loop exists to kill. Refuse honestly instead.
HANDOFF_UNARMED_REPLY = (
    "I can't actually do that from here — this surface doesn't have my tools wired."
)


def _strip_handoff(text: str) -> str:
    """Remove every occurrence of the handoff token from emitted text."""
    return text.replace(HANDOFF_MARKER, "").strip()


def _last_ai_message_id(graph: object, configurable: dict) -> str | None:
    """Id of the thread's most recent (just-checkpointed) AI message, or None.

    The escalation paths use it to surgically REPLACE the chat model's handoff
    line in durable history (add_messages replaces by id) with what actually
    got emitted — the marker text must never survive as history the next turn's
    model reads. Degrade-safe: any read failure returns None and the caller
    appends instead (a duplicate-shaped history beats a dead turn).
    """
    try:
        msgs = graph.get_state({"configurable": configurable}).values.get("messages", [])
        if msgs and getattr(msgs[-1], "type", "") == "ai":
            return getattr(msgs[-1], "id", None)
    except Exception:
        log.warning("handoff: last-AI-id read failed — will append instead", exc_info=True)
    return None


# ── FIX 1: the action-honesty gate (the anti-hallucinated-action rail) ──────────
# Production incident 2026-07-12: "turn off the office lights" routed to 'action' but
# the model answered "Both office lights are off." with tool_calls=[] — a completed
# device action CLAIMED with no tool ever run (the lights stayed on). The router
# already fails TOWARD action to keep this off the chat path; this closes the last
# hole — an action turn that TOUCHED nothing yet SPOKE as if it had. It is the same
# V1 hallucinated-tool-call failure the whole TOOLS block exists to kill.
#
# The DECISION is a pure function (route, executed tool calls, retry-state) -> verdict
# so it is unit-testable in the codebase's pure-handler idiom; the wiring in
# _action_turn / _complete_action re-invokes the action graph on a 'retry' verdict.
GATE_EMIT = "emit"      # honest — send the reply as-is
GATE_RETRY = "retry"    # zero-tool action turn, first pass — bounce once with a correction
GATE_MARK = "mark"      # still zero-tool after the bounce — emit but flag the pattern

# Appended (as the caller's next turn) when a retry is needed. Verbatim spirit from
# the fix brief: call the tool, or admit no action was taken — never fake a done deal.
ACTION_NO_TOOL_CORRECTION = (
    "You produced an answer without calling any tool. Either call the tool that "
    "performs the request, or — WHEN THE MESSAGE ACTUALLY ASKED YOU TO DO OR "
    "CHECK SOMETHING and you could not — state plainly that you did NOT perform "
    "any action. "
    "If no dedicated tool fits but search_web could ground the answer — prices, "
    "costs, availability, current facts — run search_web and answer with a "
    "clearly-labeled estimate instead of declining. "
    # The disclosure clause above is load-bearing (it is what stops her claiming
    # lights she never touched), but it was firing on messages that requested
    # nothing at all: "morning check — you good?" came back as "No tool call
    # needed here — that was just a status check, not a task with an action to
    # perform." That is the confession this correction's own last line forbids;
    # the prompt was telling her both to disclose and not to mention it, and she
    # resolved the contradiction by narrating. Naming the no-action branch
    # explicitly removes the conflict instead of leaving her to pick.
    "But if the message asked for NO action — a greeting, small talk, an "
    "opinion, thanks, a question about you — then there is nothing to disclose: "
    "simply answer it as yourself. Say NOTHING about tools, tool calls, checks, "
    "or what kind of request it was. 'No tool needed here' is exactly the kind "
    "of plumbing talk this correction forbids. "
    "Never describe an action as done unless a tool call actually did it. This "
    "correction is internal plumbing: never mention it, the earlier answer, or "
    "any slip to the user — just act and confirm the result."
)

# The degraded marker recorded in v2_turns when a bounced action turn STILL ran no
# tool — so the pattern (a legit zero-tool action, or a stubborn hallucination that
# survived the correction) stays visible to the capability loop / forensics.
NO_TOOL_ACTION_MARKER = "no_tool_action"
# Set on the ACTION row of a turn the claim gate rescued, so the miner can count
# how often the tool-less chat node claims work it never did (distinct from
# chat_handoff, which is the model raising its own hand).
CLAIM_GATE_MARKER = "claim_gate_escalated"

# False-wake grace (owner ask 2026-08-27): the router judged this voice/lens
# capture was never directed at Aerys and the turn was dropped — no reply, no
# action, but ALWAYS this receipt (never silent-silent; the marker is how drop
# judgment gets audited and tuned). By-design telemetry, skipped by the gaps
# miner like the handoff pair.
DROPPED_UNADDRESSED_MARKER = "dropped_unaddressed"

# Conversation-in-flight registry for the drop gate (owner design 2026-08-27):
# thread_id -> monotonic time of the last turn that produced a real reply.
# Process-local ON PURPOSE — every drop-eligible surface (voice, lens) flows
# through the single --serve process, and losing it on redeploy only means the
# gate treats the next capture as cold, which is the conservative direction.
# Injectable in ask() so tests get isolation instead of cross-test bleed.
_THREAD_ACTIVITY: dict[str, float] = {}


def _conversation_in_flight(
    registry: dict[str, float], thread_id: str, window_s: float
) -> bool:
    last = registry.get(thread_id)
    return last is not None and (time.monotonic() - last) <= window_s


# ── FIX 3: the self-perception gate (gap #33, owner-approved 2026-08-10) ────────
# Production incident 2026-08-10: mid voice-outage, Kael told her the satellite fix
# was applied and she replied "I'm hearing you on the speaker now, yeah. Voice is
# live." — a first-person sensory confirmation she is not equipped to make (no
# audio input path exists), and the TTS was in fact still broken. The action gate
# can't catch this: it keys on tool emptiness, and no tool grants hearing at all.
#
# v1 is deliberately AUDIO-ONLY. Hearing is unambiguous — she has no ears, so a
# claim of having heard something is always fabricated. "I can see..." is NOT
# gated: she legitimately sees state through tools (HA reads, camera snapshots),
# and gating sight would fire on honest sentences constantly. The empathy idiom
# ("I hear you, that's rough") survives because the pattern requires an audio-
# output noun within the same clause — bare "I hear you" never matches.
_AUDIO_PERCEPTION_RE = re.compile(
    # "I" + optional auxiliaries/adverbs (ENUMERATED — "I'll"/"I told her about
    # hearing" must not match; intent and reported speech are not claims) +
    # a hearing verb + an audio-output noun inside the same clause.
    r"\bI(?:'m|'ve| am| have| can| could| just| definitely| clearly| really"
    r"| actually| totally| still| now| did)*\s*"
    r"(?:hear(?:ing|d)?|listen(?:ed|ing)?(?:\s+to)?)\b"
    r"[^.!?\n]{0,60}?"
    r"\b(?:speakers?|audio|sound|voice|music|announce(?:ment)?s?|chime|tts|playback)\b",
    re.IGNORECASE,
)

#: Degraded marker for a reply that still claimed hearing after the bounce —
#: emitted (visibility over censorship, same philosophy as GATE_MARK) but
#: countable by the miner and diffable on the operator dash.
AUDIO_CLAIM_MARKER = "unverifiable_audio_claim"

# Appended as the caller's next turn when the gate bounces. Same contract as
# ACTION_NO_TOOL_CORRECTION: internal plumbing, never mentioned to the user.
AUDIO_CLAIM_CORRECTION = (
    "You just claimed to HEAR something — you have no audio input. You cannot "
    "hear the room, the speakers, or your own voice; a claim of having heard "
    "audio is fabricated even when the underlying fact happens to be true. "
    "Answer again honestly: state what tools or receipts actually confirmed "
    "(a command accepted, a service answering) and, when it matters whether "
    "sound really played, ask the human what reached their ears instead of "
    "asserting it. Do not stop being warm — just never wear senses you don't "
    "have. This correction is internal plumbing: never mention it, the earlier "
    "answer, or any slip — simply give your honest answer."
)


def audio_perception_claim(text: str) -> bool:
    """True when the reply asserts first-person HEARING of audio output.

    Pure and deliberately narrow (see the FIX 3 block comment): perception verb
    and an audio-output noun must share one clause. False positives cost her
    voice its warmth; false negatives cost one marker — tuned toward the first.
    """
    return bool(_AUDIO_PERCEPTION_RE.search(text or ""))


_RECORD_CLAIM_RE = re.compile(
    # Three shapes of a COMPLETED record-write claim (gap #47, receipts
    # 8/14): the bare "Logged:" prefix (the incident verbatim); first-person
    # completed logged/filed/recorded (login idioms excluded — "logged in/
    # into/onto" is not a record claim); and "<verb> that/this/it as a gap".
    # Intent ("let me log", "I'll log") deliberately does NOT match — intent
    # is honest; only the claimed completion is a fabrication here.
    r"(?:(?:^|\n)\s*Logged\s*[:\u2014\u2013-])"
    r"|(?:\bI(?:'ve| have| just| already)?\s+"
    r"(?:logged(?!\s+in\b|\s+into\b|\s+on(?:to)?\b)|filed|recorded)\b)"
    r"|(?:\b(?:logged|filed|recorded)\s+(?:that|this|it)\s+as\s+a\s+"
    r"(?:gap|capability|request|issue)\b)",
    re.IGNORECASE,
)

#: Degraded marker for a record-write claim that survived the bounce —
#: emitted (visibility over censorship) but countable by the miner.
RECORD_CLAIM_MARKER = "unverifiable_record_claim"

# Same internal-plumbing contract as the audio correction. Honesty-only v1:
# no re-route instruction — a bounce retry cannot ride the handoff escalation
# (that flag was computed from the original reply), so the honest recovery is
# owning the limitation and steering the human to a path that can write.
RECORD_CLAIM_CORRECTION = (
    "You just claimed to have logged or recorded something — this "
    "conversational path has no record-keeping tool, so nothing was written "
    "anywhere. A 'Logged:' that wrote nothing is a fabrication even when "
    "well-intentioned. Answer again honestly: acknowledge the request "
    "matters, say plainly that you couldn't write it to the board from "
    "here, and either offer to carry it in conversation or ask them to "
    "raise it again in a fresh message so it reaches the path that can "
    "actually record it. Do not stop being warm. This correction is "
    "internal plumbing: never mention it, the earlier answer, or any slip "
    "— simply give your honest answer."
)


def record_action_claim(text: str) -> bool:
    """True when the reply claims a COMPLETED log/file/record write.

    Same tuning philosophy as the audio detector: narrow beats eager. Login
    idioms, stated intent, third-party reports, and reading logs all pass.
    """
    return bool(_RECORD_CLAIM_RE.search(text or ""))


def action_honesty_gate(route: str, tool_calls: list, *, already_retried: bool) -> str:
    """Pure verdict for the action-honesty gate: emit | retry | mark.

    - A non-action route is NEVER gated (chat may legitimately answer with no tool) ->
      emit.
    - An action turn that executed at least one tool is honest -> emit. A FAILED tool
      call still counts as executed (the tool ran and returned an honest error) — we
      only bounce turns that touched nothing at all.
    - An action turn with ZERO executed tool calls, first pass -> retry (bounce once).
    - The same, but AFTER the one allowed bounce -> mark: emit the reply but attach
      NO_TOOL_ACTION_MARKER. Legitimate zero-tool action turns (an honest "I can't see
      that" answer, a compose request misrouted to action) survive the retry because
      the correction explicitly permits stating no action was taken; they land here
      marked, not suppressed.

    tool_calls is the structured list from turns.extract_tool_calls (one entry per
    EXECUTED ToolMessage); only its emptiness matters here.

    Since 2026-09-04 the specialist's first pass is FORCED to call a tool (with
    no_action as the honest exit), so a zero-tool first pass is only reachable with
    ACTION_FORCE_TOOL=false; the retry path stays as the belt under that knob.
    """
    if route != "action":
        return GATE_EMIT
    if tool_calls:
        return GATE_EMIT
    return GATE_MARK if already_retried else GATE_RETRY


def _run_action_gated(
    action_graph: object, seeded_messages: list, config: dict
) -> tuple[dict, list[str]]:
    """Invoke the action graph, then enforce the action-honesty gate.

    Returns (result, extra_degraded). An action turn that ran ZERO tools is bounced
    ONCE — the graph is re-invoked with its own no-tool answer plus a corrective
    message appended — and if it STILL runs no tool, extra_degraded carries
    NO_TOOL_ACTION_MARKER so the reply is emitted but the pattern is audited.

    One extra model call on the rare zero-tool action turn is the accepted cost of
    never again emitting a fabricated "done" (fix brief, 2026-07-12)."""
    result = action_graph.invoke({"messages": seeded_messages}, config)
    verdict = action_honesty_gate(
        "action", extract_tool_calls(result["messages"]), already_retried=False
    )
    if verdict != GATE_RETRY:
        return result, []
    log.info("action-honesty gate: zero tool calls — bouncing once with a correction")
    retry_messages = [*result["messages"], HumanMessage(content=ACTION_NO_TOOL_CORRECTION)]
    result = action_graph.invoke({"messages": retry_messages}, config)
    verdict = action_honesty_gate(
        "action", extract_tool_calls(result["messages"]), already_retried=True
    )
    if verdict == GATE_MARK:
        log.info(
            "action-honesty gate: STILL zero tool calls after the bounce — "
            "emitting with the %s marker", NO_TOOL_ACTION_MARKER,
        )
        return result, [NO_TOOL_ACTION_MARKER]
    return result, []


# ── FIX 2: an honest reply when the turn is rate-limited (never silence) ─────────
# Production incident 2026-07-12: "Are you sure?" died as degraded=['turn_failed'],
# error="oauth backend error: ... result=\"You've hit your session limit · resets
# 7:10pm (UTC)\" ...", emitted_reply=None — the turn RAISED and the user got NOTHING
# on their glasses. The oauth (Max-pool) chat backend surfaces a session/word-budget
# cap as a RuntimeError; rather than re-raising into silence we emit a short, honest,
# in-voice line (with the reset time converted to Eastern when parseable). It rides
# the normal emitted_reply path, so EVERY transport benefits at once.
EASTERN = ZoneInfo("America/New_York")  # Chris's timezone — the clock she reasons in

# "resets 7:10pm (UTC)" / "resets 7pm" / "resets 07:10 pm UTC" — hour, optional
# minutes, am/pm, optional trailing zone. Case-insensitive; scans anywhere in the text.
_RESET_RE = re.compile(
    r"reset[s]?\s+(\d{1,2})(?::(\d{2}))?\s*([ap]m)\s*\(?\s*([A-Za-z_/]+)?",
    re.IGNORECASE,
)
# Signatures that mean "the model's word/usage budget is spent", not a normal error.
_RATE_LIMIT_SIGNALS = ("session limit", "rate limit", "usage limit", "rate_limit")


def _parse_reset_eastern(error_text: str, *, now: datetime | None = None) -> str | None:
    """Pull a 'resets <time> (<zone>)' out of a limit error and render it in Eastern,
    e.g. 'resets 7:10pm (UTC)' -> '3:10pm'. None when nothing parseable is present.

    The zone defaults to UTC (the backend reports UTC); an unknown zone label also
    falls back to UTC rather than failing. If the reset clock time is already past
    relative to `now`, it is rolled to the next day — a limit 'resets 7:10pm' names
    the upcoming boundary, never one in the past. `now` is injectable so the
    conversion is deterministic under test."""
    m = _RESET_RE.search(error_text or "")
    if not m:
        return None
    hour = int(m.group(1))
    minute = int(m.group(2) or 0)
    if not (1 <= hour <= 12) or minute > 59:
        return None
    ampm = m.group(3).lower()
    if ampm == "pm" and hour != 12:
        hour += 12
    elif ampm == "am" and hour == 12:
        hour = 0
    zone = (m.group(4) or "UTC").upper()
    try:
        src_tz = ZoneInfo("UTC") if zone in ("UTC", "GMT", "Z") else ZoneInfo(m.group(4))
    except Exception:
        src_tz = ZoneInfo("UTC")
    now = now or datetime.now(EASTERN)
    now_src = now.astimezone(src_tz)
    reset = now_src.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if reset <= now_src:
        reset = reset + timedelta(days=1)
    eastern = reset.astimezone(EASTERN)
    h12 = eastern.strftime("%I").lstrip("0") or "12"
    suffix = eastern.strftime("%p").lower()
    return f"{h12}:{eastern:%M}{suffix}"


def rate_limit_reply(error_text: str, *, now: datetime | None = None) -> str | None:
    """An honest, in-voice reply for a rate/session-limit failure — or None when the
    error is any OTHER failure class (those keep the historical re-raise; the caller
    decides). No emotion tags on purpose: the SAME string goes to voice AND text
    transports, and a bracketed tag would print literally in a Discord/Telegram
    message (the voice pipeline is the only surface that strips them)."""
    low = (error_text or "").lower()
    if not any(sig in low for sig in _RATE_LIMIT_SIGNALS):
        return None
    reset = _parse_reset_eastern(error_text, now=now)
    if reset:
        return (
            "I'm rate-limited right now — my brain's word budget is tapped until "
            f"about {reset}. Try me again after that."
        )
    return (
        "I'm rate-limited right now — my brain's word budget is tapped for a bit. "
        "Try me again in a little while."
    )


def _honest_reply_for_failure(exc: BaseException) -> str | None:
    """The emittable, in-voice reply for a turn whose model invoke RAISED — or None to
    keep the historical re-raise-into-silence for every other failure class.
    Converted classes: the oauth/session rate-limit cap (FIX 2), and — since 9/04 —
    the recursion rail: a tool loop that hit the wall (every Jolteon entity was
    unavailable and the specialist kept re-reading) surfaced as an HTTP 500 with NO
    reply. The rail is doing its job; the caller still deserves a sentence."""
    if isinstance(exc, GraphRecursionError):
        return (
            "I went in circles on that one and stopped myself — the reading never "
            "came together. Ask me again in a moment, or tell me which device to look at."
        )
    return rate_limit_reply(str(exc) or type(exc).__name__)


def ask(
    graph: object,
    text: str,
    *,
    identity: Identity,
    thread_id: str,
    rails: Rails = Rails(),
    router: Callable[[str], RouteDecision] | None = None,
    action_graph: object | None = None,
    guest_action_graph: object | None = None,
    speak_fn: Callable[[str, str], None] | None = None,
    satellite_for: Callable[[str | None], str] | None = None,
    followup_router: Callable[[str, str | None], None] | None = None,
    display_push: Callable[[str, str | None], None] | None = None,
    followup_skip_s: float = 6.0,
    deep_allowed: Callable[[], bool] | None = None,
    action_allowlist: frozenset[str] | None = None,
    record_turn: Callable[[dict], None] | None = None,
    content_privacy_classifier: Callable[[str], str] | None = None,
    face_push: Callable[[str, str], None] | None = None,
    drop_unaddressed: bool = False,
    drop_conversation_window_s: float = 180.0,
    activity_registry: dict[str, float] | None = None,
) -> str:
    """Run one conversational turn and return the reply text.

    - identity rides `configurable` (the S2 channel) — per-call, never checkpointed.
    - thread_id selects the conversation; the checkpointer replays its history.
    - recursion_limit is the LangGraph-native turn_limit enforcement: each graph
      super-step counts, so a runaway tool loop trips it long before infinity.
    - router + action_graph arm the TOOLS block (see module docstring); either
      missing = the pre-TOOLS chat-only path, unchanged.
    - speak_fn + satellite_for + followup_skip_s: the voice spoken-follow-up
      seam — speak_fn delivers (text, entity_id) to the room (HA announce in
      prod, a fake in tests); satellite_for resolves the originating device_id
      to the announce entity_id (factory.resolve_announce_entity), so the
      follow-up answers on the SAME satellite the turn came from; the
      silent-success rule in _voice_parallel_start decides WHEN it fires. None
      satellite_for = no follow-up target resolves, so speak_fn never fires
      (the pre-satellite-routing default: history-only unless the caller wires
      both halves, exactly as cli.py --serve does).
    - followup_router: when wired (factory.followup_router_for), it OWNS follow-up
      delivery per originating device — a mapped satellite gets an announce, the
      headless Myo phone (unmapped/None device_id) gets an `aerys_followup` HA
      event it turns into speech. Takes precedence over speak_fn/satellite_for;
      None falls back to the legacy announce path above (tests, dev boxes).
    - deep_allowed: the deep-tier cap gate (factory.deep_gate_for) — consulted
      ONLY when a text-thread chat turn actually classified deep, so voice
      turns and downgrades never burn a v2_model_usage credit. None = cap
      unenforced (dev boxes); the gate saying False downgrades to standard.
    - action_allowlist: the AUTH gate for the SENSITIVE tools. House control,
      presence reads, and web search are restricted to an allowlist of person_ids: a
      caller NOT in it is swapped onto guest_action_graph (analyze_image /
      read_document / youtube only) — or fully chat-only if no media graph is armed.
      Reading media someone shares is not sensitive; actuating the house or
      disclosing presence is. The memory boundary makes a stranger's identity COLD
      (no memories) but does NOTHING to the tools — so this gate is the one thing
      between a guild member and the owner's house. The owner is always in the set;
      more can be added by config (e.g. Megan) with no code change
      (factory.action_allowlist_for). None = unenforced (dev boxes).
    - guest_action_graph: the reduced action graph (media tools only) used for
      non-allowlisted callers, from factory.guest_action_graph_for. None = they get
      no tools at all (chat-only), preserving the pre-media-split behavior.
    - record_turn: the v2_turns audit seam (factory.turn_recorder_for). Called
      once per completed turn on EVERY path with the fully-built row, off the hot
      path and fail-open (see _fire_turn_record). None = no auditing (dev/CI, no
      DATABASE_URL), byte-for-byte the old behavior.
    - face_push: the panel-face seam (factory.face_pusher_for) — (phase, text)
      with phase working|speaking|idle, fired at the turn's phase changes so
      her desk avatar mirrors what the brain is doing. Fire-and-forget and
      fail-open by construction; None = no panel (dev/CI), zero cost.
    - content_privacy_classifier: the OFF-hot-path judge (factory.content_privacy_fn_for)
      that relaxes a DM turn's fail-closed 'private' content tag to 'public' when the
      content is general, so general things said in a DM carry into public rooms while
      private-CONTENT things never do. None = feature off: DM turns stay 'private' and
      simply never carry into public. Never touches latency (daemon thread) and never
      loosens the public-origin path (those turns are already 'public').
    """
    if not text or not text.strip():
        raise ValueError("ask() requires non-empty text")

    # Content-privacy tagging (short-term gate, write half): compute THIS turn's ingest
    # tag once, and mint a stable id so the async judge can retag this exact human
    # message. Both ride down into whichever path builds the main-thread human message.
    origin_privacy = _origin_privacy(identity)
    turn_msg_id = str(uuid.uuid4())

    # Gate the action stack BEFORE anything else can arm it. A caller outside the
    # allowlist never reaches home_control / search_entities / get_state — closing
    # both the unauthorized-actuation and the presence-disclosure (reads) risks.
    if action_allowlist is not None and identity.get("user_id") not in action_allowlist:
        # Non-allowlisted callers lose house control + presence + web search, but
        # KEEP media (analyze_image / read_document / youtube) — reading what someone
        # shares is not sensitive. Swap to the media-only graph; if none is armed,
        # fall fully chat-only (router None), exactly the old behavior.
        action_graph = guest_action_graph
        if action_graph is None:
            router = None

    started = time.monotonic()
    config = {
        "configurable": {"thread_id": thread_id, "identity": identity},
        "recursion_limit": rails.turn_limit,
    }

    with _turn_span(str(thread_id), text):
        if router is None or action_graph is None:
            # Chat-only path: either the TOOLS block isn't armed, or the caller was
            # forced off it by the allowlist gate above. No router ran, so
            # classifier_intent/tier stay NULL — the row records what actually
            # happened, not a tier decision that was never made.
            reply, handoff = _chat_turn(
                graph, text, config, rails, started, record_turn=record_turn,
                human_privacy=origin_privacy, human_id=turn_msg_id,
            )
            if handoff:
                # The model raised its hand but there is no action graph to hand
                # to (dev box, or a guest with no media graph). Emitting its "let
                # me get that for you" line would promise work nothing will do —
                # refuse honestly instead. Guests asking for house control land
                # here by design: the allowlist gate stripped their tools. Patch
                # the checkpointed marker line so history matches what was said.
                reply = HANDOFF_UNARMED_REPLY
                msg_id = _last_ai_message_id(graph, config["configurable"])
                if msg_id is not None:
                    graph.update_state(
                        {"configurable": config["configurable"]},
                        {"messages": [AIMessage(content=reply, id=msg_id)]},
                        as_node="chat",
                    )
            _reclassify_if_needed(
                graph, config, turn_msg_id, text, reply,
                content_privacy_classifier, origin_privacy,
            )
            _face(face_push, "idle", reply)
            return reply

        if is_voice_turn(identity, thread_id):
            # Voice detection now rides the EXPLICIT identity.voice flag (is_voice_turn),
            # not the thread prefix — because voice folds into the owner's person-keyed
            # thread ('person:{id}') and no longer names 'voice'. Behavior is unchanged:
            # ChannelPolicy (locked) PINS voice to the standard tier — the ~3.6s budget
            # can't absorb deep latency, and fast-tier identity wobbles are what got
            # Haiku demoted in V1. The pin is structural: this path never writes a tier
            # into config, so the chat node's DEFAULT_TIER (= standard) always applies.
            # Content reclassification NOW runs on voice too: person-keying means a voice
            # turn shares the owner's thread with his public text turns, so a private
            # thing said by voice must be gated out of public exactly like a DM — the
            # fail-closed 'private' ingest tag (below) does the gating, and the async
            # judge relaxes general voice content so it still carries into public rooms.
            voice_reply = _voice_parallel_start(
                graph, text, config, rails, started, router, action_graph,
                speak_fn, satellite_for, followup_skip_s, record_turn=record_turn,
                followup_router=followup_router,
                display_push=display_push,
                content_privacy_classifier=content_privacy_classifier,
                human_privacy=origin_privacy, human_id=turn_msg_id,
                face_push=face_push,
                drop_unaddressed=drop_unaddressed,
                drop_conversation_window_s=drop_conversation_window_s,
                activity_registry=activity_registry,
            )
            if voice_reply:
                registry = (
                    _THREAD_ACTIVITY if activity_registry is None else activity_registry
                )
                registry[thread_id] = time.monotonic()
            return voice_reply

        # Non-voice: nobody is waiting on a speaker, so the router runs first
        # (sequential) and only the chosen path spends model tokens.
        decision = router(text)
        registry = _THREAD_ACTIVITY if activity_registry is None else activity_registry
        if (
            drop_unaddressed
            and decision.unaddressed
            and is_lens_surface(identity)
            and not _conversation_in_flight(
                registry, thread_id, drop_conversation_window_s
            )
        ):
            # False-wake grace on the LENS path (glasses turns arrive voice=False
            # — the G2 has no speaker — but their capture is still a mic that
            # misfires). Typed surfaces never reach this: no lens, no drop.
            log.info(
                "route decision | thread=%s DROPPED (unaddressed voice capture)",
                thread_id,
            )
            _fire_turn_record(
                record_turn, config, text,
                int((time.monotonic() - started) * 1000),
                classifier_intent="unaddressed",
                raw_reply="", emitted_reply="",
                extra_degraded=[DROPPED_UNADDRESSED_MARKER],
            )
            return ""
        if decision.route == "action":
            # add_human=True: the chat graph never saw this turn, so BOTH the human
            # message and the action result must land in the thread history.
            log.info("route decision | thread=%s route=action", thread_id)
            _face(face_push, "working")
            reply = _action_turn(
                action_graph, graph, text, config, add_human=True,
                record_turn=record_turn, started=started,
                human_privacy=origin_privacy, human_id=turn_msg_id,
                tier=normalize_tier(decision.tier),
            )
            _reclassify_if_needed(
                graph, config, turn_msg_id, text, reply,
                content_privacy_classifier, origin_privacy,
            )
            _face(face_push, "idle", reply)
            registry[thread_id] = time.monotonic()
            return reply

        # Chat route on a TEXT thread: the router's tier picks the model. This
        # is where the deep cap bites — the gate is an atomic spend against
        # v2_model_usage, so it runs ONLY once we know this turn is deep.
        tier = normalize_tier(decision.tier)
        override_source: str | None = None
        downgrade_marker: list[str] | None = None
        if tier == "deep" and deep_allowed is not None and not deep_allowed():
            # Cap held: degrade to standard, and say so in the logs (the V1
            # opus cap degraded SILENTLY — a documented regret, not a feature).
            log.info(
                "deep tier cap reached — downgrading to standard | thread=%s", thread_id
            )
            tier = DEFAULT_TIER
            # The turn row now carries this too: the served tier (standard), WHY it
            # differs from the classifier's pick (tier_override_source), and a
            # degraded marker so the capability loop can see a capped deep request.
            override_source = "deep_cap"
            downgrade_marker = ["deep_cap_downgraded"]
        log.info("route decision | thread=%s route=chat tier=%s", thread_id, tier)
        config["configurable"]["tier"] = tier
        reply, handoff = _chat_turn(
            graph, text, config, rails, started, record_turn=record_turn,
            classifier_intent="chat", tier=tier,
            tier_override_source=override_source, extra_degraded=downgrade_marker,
            human_privacy=origin_privacy, human_id=turn_msg_id,
        )
        claim_escalation = False
        if not handoff and action_graph is not None and _claims_effect_without_doing_it(
            text, reply
        ):
            # THE CLAIM GATE (gap #15, owner-approved 2026-07-28). The chat model
            # said it did something while holding no tools. Voice can't reach this
            # state any more — every voice turn runs the tool-armed graph — but
            # text still routes through a tool-less chat node, so this is the
            # remaining door the 7/25 fabrication came through.
            #
            # The response is to ESCALATE, not to scold: bouncing a tool-less node
            # with "call the tool" only teaches it to apologize, whereas the action
            # graph HAS the tool and will either do the thing or refuse honestly.
            # Reuses the return loop's machinery verbatim — the checkpointed claim
            # is replaced by the real outcome, so history never keeps the lie.
            log.info(
                "claim gate: tool-less chat reply claimed an effectful act — "
                "escalating to action | thread=%s", thread_id
            )
            handoff = True
            claim_escalation = True
        if handoff:
            # THE RETURN LOOP (owner design, 2026-07-18): the chat model — the only
            # component that saw full history — says this turn needs hands. Re-run
            # it on the action graph directly (no router re-run: the router already
            # got one vote and lost; the model's marker IS the reclassification).
            # add_human=False because the chat invoke already landed the human turn;
            # replace_message_id swaps the checkpointed handoff line for the action
            # outcome, so history ends up exactly as if the router had said action.
            # One hop: nothing on the action side can emit a live marker, so this
            # branch cannot re-enter itself.
            log.info(
                "chat handoff — escalating to action | thread=%s", thread_id
            )
            _face(face_push, "working")
            reply = _action_turn(
                action_graph, graph, text, config, add_human=False,
                record_turn=record_turn, started=started,
                human_privacy=origin_privacy, human_id=turn_msg_id,
                replace_message_id=_last_ai_message_id(graph, config["configurable"]),
                escalated=True,
                extra_degraded=[CLAIM_GATE_MARKER] if claim_escalation else None,
                tier=normalize_tier(decision.tier),
            )
        _reclassify_if_needed(
            graph, config, turn_msg_id, text, reply,
            content_privacy_classifier, origin_privacy,
        )
        _face(face_push, "idle", reply)
        registry[thread_id] = time.monotonic()
        return reply


def _chat_turn(
    graph: object,
    text: str,
    config: dict,
    rails: Rails,
    started: float,
    *,
    record_turn: Callable[[dict], None] | None = None,
    classifier_intent: str | None = None,
    tier: str | None = None,
    tier_override_source: str | None = None,
    extra_degraded: list[str] | None = None,
    human_privacy: str = PRIVATE,
    human_id: str | None = None,
) -> tuple[str, bool]:
    """The original chat path: invoke, budget-check, extract — now also audited.

    Returns (reply, handoff). handoff=True means the model opened with
    HANDOFF_MARKER — it judged this turn action-shaped (the return loop); the
    returned reply is the marker-stripped handoff line, and it is the CALLER's
    job to escalate (routed path) or refuse honestly (chat-only path). The
    audit row keeps the marker in raw_reply (the receipt a misroute happened)
    and adds CHAT_HANDOFF_MARKER to degraded so misroutes are countable.

    The v2_turns row is fired on BOTH exits: the normal return AND the timeout
    raise (the reply exists either way — a turn that ran past budget is exactly
    the kind of thing forensics need to see). raw_reply == emitted_reply here
    (modulo marker-stripping): the chat path has no separate polish step (V1's
    Gemini polisher is now prompt-side emotion tags), so what the model said IS
    what the channel emits.
    """
    LOCAL_FALLBACK_FIRED.set(False)  # per-turn flag; stamped into degraded below
    try:
        result = graph.invoke(
            {"messages": [_human_turn(text, human_privacy, human_id)]}, config
        )
    except Exception as e:
        # A raised invoke (model 500, recursion-rail trip) is the HIGHEST-value turn
        # for forensics and the capability loop — record it BEFORE re-raising so the
        # 'row on EVERY completion path incl. error' contract actually holds
        # (cross-review correctness H). FIX 2: if the failure is a rate/session-limit
        # cap, emit an honest in-voice line instead of raising into silence (the
        # 2026-07-12 "Are you sure?" glasses turn got NO reply) — the row then records
        # what the user actually heard, with 'turn_failed' still on degraded. Every
        # other failure class re-raises exactly as before.
        honest = _honest_reply_for_failure(e)
        _record_turn_failure(
            record_turn, config, text, started, e,
            classifier_intent=classifier_intent,
            tier=tier,
            tier_override_source=tier_override_source,
            base_degraded=[
                *(extra_degraded or []),
                # Primary died, lifeboat fired, and the turn STILL raised — the row
                # must show both facts (owner condition 8/03: logs reflect fallback).
                *(["local_model_fallback"] if LOCAL_FALLBACK_FIRED.get() else []),
            ] or None,
            emitted_reply=honest,
        )
        if honest is not None:
            return honest, False
        raise
    raw = _reply_text(result["messages"][-1])
    handoff = HANDOFF_MARKER in raw
    reply = _strip_handoff(raw) if handoff else raw

    # FIX 3 (gap #33): a chat reply that claims to have HEARD audio gets one
    # bounce with the correction — the same one-extra-call trade as the action
    # gate. raw keeps the ORIGINAL claim so the operator dash's gate diff shows
    # exactly what she was going to say; a persistent claim is emitted with the
    # marker (visibility, not censorship). Handoff lines skip the gate — they
    # are one-sentence escalations, and the action path audits its own reply.
    #
    # History surgery: the chat graph CHECKPOINTS, so a naive bounce would leave
    # the correction text (internal plumbing) and the fabricated claim in the
    # durable thread for every later turn to read. After the bounce, the claiming
    # AI message is replaced in-place (add_messages replaces by id) with the
    # corrected reply, and the correction + its duplicate answer are removed.
    # Degrade-safe: if surgery fails, the messy history is logged and kept — a
    # readable-but-cluttered thread beats a dead turn.
    audio_gate_marker = False
    if not handoff and audio_perception_claim(reply):
        log.info("self-perception gate: audio claim in chat reply — bouncing once")
        claim_id = _last_ai_message_id(graph, config.get("configurable") or {})
        retry = graph.invoke(
            {"messages": [HumanMessage(content=AUDIO_CLAIM_CORRECTION)]}, config
        )
        reply = _reply_text(retry["messages"][-1])
        result = retry
        if audio_perception_claim(reply):
            log.warning(
                "self-perception gate: audio claim SURVIVED the bounce — "
                "emitting with %s", AUDIO_CLAIM_MARKER,
            )
            audio_gate_marker = True
        try:
            from langchain_core.messages import RemoveMessage

            tail = retry["messages"][-2:]
            removals = [
                RemoveMessage(id=m.id) for m in tail if getattr(m, "id", None)
            ]
            surgery: list = list(removals)
            if claim_id is not None:
                surgery.append(AIMessage(content=reply, id=claim_id))
            if surgery:
                graph.update_state(
                    {"configurable": config.get("configurable") or {}},
                    {"messages": surgery},
                    as_node="chat",
                )
        except Exception:
            log.warning(
                "self-perception gate: history surgery failed — correction "
                "text remains in thread history", exc_info=True,
            )

    # Gap #47: same shape for claimed record-writes ("Logged:" with no tool).
    # One bounce per turn across BOTH gates: if the audio gate already spent
    # it (its claim_id local exists), a record claim in the retry is marked
    # without another model call — two bounces would double latency for the
    # rarest possible overlap.
    record_gate_marker = False
    audio_bounced = "claim_id" in locals()
    if not handoff and record_action_claim(reply):
        if audio_bounced:
            log.warning(
                "record-claim gate: claim present after the audio bounce — "
                "emitting with %s", RECORD_CLAIM_MARKER,
            )
            record_gate_marker = True
        else:
            log.info("record-claim gate: 'Logged:' claim in chat reply — bouncing once")
            claim_id = _last_ai_message_id(graph, config.get("configurable") or {})
            retry = graph.invoke(
                {"messages": [HumanMessage(content=RECORD_CLAIM_CORRECTION)]}, config
            )
            reply = _reply_text(retry["messages"][-1])
            result = retry
            if record_action_claim(reply):
                log.warning(
                    "record-claim gate: claim SURVIVED the bounce — "
                    "emitting with %s", RECORD_CLAIM_MARKER,
                )
                record_gate_marker = True
            try:
                from langchain_core.messages import RemoveMessage

                tail = retry["messages"][-2:]
                removals = [
                    RemoveMessage(id=m.id) for m in tail if getattr(m, "id", None)
                ]
                surgery: list = list(removals)
                if claim_id is not None:
                    surgery.append(AIMessage(content=reply, id=claim_id))
                if surgery:
                    graph.update_state(
                        {"configurable": config.get("configurable") or {}},
                        {"messages": surgery},
                        as_node="chat",
                    )
            except Exception:
                log.warning(
                    "record-claim gate: history surgery failed — correction "
                    "text remains in thread history", exc_info=True,
                )

    elapsed = time.monotonic() - started
    timed_out = elapsed > rails.wall_clock_s
    timeout_msg = (
        f"turn took {elapsed:.1f}s (budget {rails.wall_clock_s}s)" if timed_out else None
    )
    degraded = list(extra_degraded or [])
    if LOCAL_FALLBACK_FIRED.get():
        # The metered model died mid-turn and the local lifeboat answered
        # (owner-approved automatic failover, 2026-08-03 — condition: visible).
        degraded.append("local_model_fallback")
    if handoff:
        degraded.append(CHAT_HANDOFF_MARKER)
    if audio_gate_marker:
        degraded.append(AUDIO_CLAIM_MARKER)
    if record_gate_marker:
        degraded.append(RECORD_CLAIM_MARKER)
    if timed_out:
        degraded.append("wall_clock_exceeded")

    _fire_turn_record(
        record_turn, config, text, int(elapsed * 1000),
        classifier_intent=classifier_intent,
        tier=tier,
        tier_override_source=tier_override_source,
        raw_reply=raw,
        emitted_reply=reply,
        messages=result["messages"],
        extra_degraded=degraded or None,
        error=timeout_msg,
    )

    if timed_out:
        # The reply exists but arrived past budget — surface it loudly rather than
        # silently normalizing a degraded experience (voice cares at ~4s, not 90).
        raise TurnTimeout(timeout_msg)

    return reply, handoff


ACTION_SEED_HUMAN_TURNS = 3      # prior requests kept for referent resolution
ACTION_SEED_TRUNCATE_AT = 300    # chars per kept prior request


def _trim_human_turn(m: HumanMessage) -> HumanMessage:
    """A prior request, shortened for the specialist; tags and id preserved."""
    content = m.content
    if isinstance(content, str) and len(content) > ACTION_SEED_TRUNCATE_AT:
        content = content[:ACTION_SEED_TRUNCATE_AT] + "…"
    return HumanMessage(
        content=content, id=getattr(m, "id", None),
        additional_kwargs=dict(getattr(m, "additional_kwargs", {}) or {}),
    )


def _action_history_seed(
    graph: object, configurable: dict, text: str, *, escalated: bool = False,
    specialist: bool = False,
) -> list:
    """Seed the checkpointer-less action graph with the thread's PRIOR turns.

    The chat graph auto-replays history from its checkpointer; the action graph is
    compiled WITHOUT one (one-shot by design), so on its own it sees ONLY the current
    message — which is exactly why a follow-up device/media command lost its referent:
    "turn them back on", "does it look like Hsin?" both arrived with no earlier turn in
    view (the 2026-07-05 continuity bug). Read the prior turns off the chat graph's
    checkpointer and hand them to the action graph as its state, GATED for the room
    exactly like the chat node: a non-private (public/unknown) context drops
    private-tagged priors before the action model can see them, so restoring continuity
    never reopens the short-term privacy leak the chat gate closes. The current human
    turn is appended LAST so redact_private_history's always-keep-the-current-turn rule
    lands on THIS message, not the last prior one. Degrade-safe: any checkpointer read
    failure falls back to the current message alone (the pre-fix behavior), never a dead
    turn.

    Used by BOTH the text action path (_action_turn) and the voice action path
    (_complete_action, owner ask 2026-07-05: stateless voice commands are annoying).
    The 2026-07-03 STT-garble incident is guarded on the voice path by VOICE_ACK_OVERLAY
    (spoken_ack in action_config) — the model is told NEVER to ask over the one-way
    channel and to resolve garble toward the already-spoken ack — NOT by hiding history.
    """
    identity = (configurable or {}).get("identity") or {}
    try:
        prior = graph.get_state({"configurable": configurable}).values.get("messages", [])
    except Exception:
        log.warning(
            "action-seed history read failed — running action with current turn only",
            exc_info=True,
        )
        prior = []
    if escalated and prior and getattr(prior[-1], "type", "") == "ai":
        # Escalated text turn: the chat invoke already checkpointed BOTH this
        # turn's human message and the marker handoff line. Drop the trailing
        # handoff line (the action model must reason from the request, not from
        # a note about handing off) — the human turn is then already the tail,
        # so appending another copy would duplicate it.
        prior = prior[:-1]
    # ── SPECIALIST SEED (2026-09-04): prior HUMAN turns only ──────────────────
    # The 7/05 continuity fix seeded the WHOLE prior exchange. Traced 9/04: her own
    # earlier line "I can't set a specific brightness" rode into the next command
    # and the model repeated the belief instead of calling the tool (28s, 5 round
    # trips). Referents live in what the USER said ("turn off the office lights"
    # -> "turn them back on"); stale beliefs live in what SHE said. So the
    # specialist sees the last few prior requests, truncated, and never a prior
    # assistant or tool message. Privacy tags travel with each kept turn so the
    # room gate below still applies.
    # specialist=False (voice banter: conversation riding the tool graph) keeps the
    # full prior exchange — her replies ARE the context there.
    current_checkpointed = escalated and prior and getattr(prior[-1], "type", "") == "human"
    private_room = identity.get("privacy_context") == PRIVATE
    if not specialist:
        seeded = list(prior) if current_checkpointed else [*prior, HumanMessage(content=text)]
        # Mirror the chat node's fail-closed gate: redact unless the room is
        # EXPLICITLY private. A public/unknown context drops private-tagged priors;
        # a private DM/voice context passes the owner's full history through.
        return seeded if private_room else redact_private_history(seeded)
    humans = [m for m in prior if getattr(m, "type", "") == "human"]
    if current_checkpointed:
        current = prior[-1]          # already checkpointed by the chat invoke
        humans = humans[:-1]
    else:
        current = HumanMessage(content=text)
    window = humans[-ACTION_SEED_HUMAN_TURNS:] if ACTION_SEED_HUMAN_TURNS > 0 else []
    # Same room gate on the window, BEFORE folding (the gate works on message lists
    # and always keeps the last human, so the current turn rides along as the tail).
    kept = [*window, current] if private_room else redact_private_history([*window, current])
    priors = [_trim_human_turn(m) for m in kept[:-1]]
    # ★ FOLD, don't stack (9/04 bench): seeded as separate human messages with no
    # replies between them, three prior requests read as a BATCH of pending
    # commands — the specialist re-ran "20%", "brighter", "off" before the real
    # one. One message, the priors quoted as already-handled context, the
    # request last: nothing in the prompt looks like an unanswered instruction.
    if not priors:
        return [current]
    note = "\n".join(f"- {_message_text(m)}" for m in priors)
    folded = (
        "Earlier requests in this conversation — ALREADY HANDLED, shown only so "
        "you can resolve references like 'them' or 'that one'. Do not carry any of "
        "them out again unless the request below explicitly asks to repeat one:"
        f"\n{note}\n\nThe request to carry out now:\n{_message_text(current)}"
    )
    return [HumanMessage(
        content=folded, id=getattr(current, "id", None),
        additional_kwargs=dict(getattr(current, "additional_kwargs", {}) or {}),
    )]


def _message_text(m: HumanMessage) -> str:
    """Plain text of a human turn (multimodal content lists collapse to their text
    parts + a note that an attachment was present)."""
    c = m.content
    if isinstance(c, str):
        return c
    parts = []
    for part in c or []:
        if isinstance(part, dict) and part.get("type") == "text":
            parts.append(str(part.get("text", "")))
        elif isinstance(part, dict):
            parts.append(f"[{part.get('type', 'attachment')}]")
    return " ".join(parts) or str(c)


def _action_turn(
    action_graph: object,
    graph: object,
    text: str,
    config: dict,
    *,
    add_human: bool,
    record_turn: Callable[[dict], None] | None = None,
    started: float | None = None,
    human_privacy: str = PRIVATE,
    human_id: str | None = None,
    replace_message_id: str | None = None,
    escalated: bool = False,
    extra_degraded: list[str] | None = None,
    tier: str = "standard",
) -> str:
    """Run the tool subgraph, then land the outcome in the MAIN thread's history.

    replace_message_id + escalated are the RETURN-LOOP wiring: an escalated turn
    seeds from history that already ends in this turn's human message (see
    _action_history_seed), swaps the checkpointed chat handoff line for the
    action outcome (add_messages replaces by id), and stamps its audit row with
    ESCALATED_MARKER so the misroute→recovery pair is countable in v2_turns.

    The action graph is checkpointer-less (one-shot); update_state on the chat
    graph is how the outcome becomes durable conversation history — next turn,
    the chat model sees "I turned the office light on" as its own prior message
    instead of a hole where an action happened. as_node="chat" attributes the
    write to the node that normally speaks.

    The audit row carries the ACTION subgraph's OWN message list (result_messages)
    — the AIMessage tool_calls + ToolMessages — so tool_calls/degraded are mined
    from the real tool loop, not the two-line human/ai summary written to history.
    """
    try:
        # Seed the action graph with the thread's prior turns (gated for the room) so a
        # follow-up command can resolve a reference to an earlier turn — the action graph
        # is checkpointer-less and would otherwise see ONLY this message (2026-07-05
        # continuity bug: "turn them back on" / "does it look like Hsin?"). FIX 1: the
        # gate re-invokes once if the model answered with zero tool calls (a claimed
        # action that touched nothing), and marks the row 'no_tool_action' if it still
        # runs no tool after the correction.
        result, gate_degraded = _run_action_gated(
            action_graph,
            _action_history_seed(
                graph, config["configurable"], text, escalated=escalated, specialist=True
            ),
            {**config, "configurable": {**config["configurable"], "specialist": True, "tier": tier}},
        )
    except Exception as e:
        # Rail trip / tool-loop blowup on the sequential action path: record the
        # failed turn (classifier_intent='action') before re-raising — or, on a
        # rate/session-limit cap, emit the honest line instead of silence (FIX 2),
        # same as the chat path (cross-review correctness H).
        honest = _honest_reply_for_failure(e)
        _record_turn_failure(
            record_turn, config, text, started, e, classifier_intent="action",
            base_degraded=(
                ([ESCALATED_MARKER] if escalated else []) + list(extra_degraded or [])
            ) or None,
            emitted_reply=honest,
        )
        if honest is not None:
            return honest
        raise
    result_messages = result["messages"]
    # Marker-strip on every action final (return-loop one-hop belt): the action
    # side is never TAUGHT the marker, but a model echoing history could still
    # surface it — stripped here, it can neither reach the caller nor persist.
    final = _strip_handoff(_reply_text(result_messages[-1]))
    landed = (
        AIMessage(content=final, id=replace_message_id)
        if replace_message_id is not None
        else AIMessage(content=final)
    )
    messages: list = [landed]
    if add_human:
        # The tagged, stable-id human turn (so the async judge can retag it) — the
        # action graph never touched the main thread, so THIS is where it lands.
        messages.insert(0, _human_turn(text, human_privacy, human_id))
    graph.update_state(
        {"configurable": config["configurable"]}, {"messages": messages}, as_node="chat"
    )

    degraded = list(gate_degraded or [])
    if escalated:
        degraded.append(ESCALATED_MARKER)
    degraded.extend(extra_degraded or [])
    latency_ms = int((time.monotonic() - started) * 1000) if started is not None else None
    _fire_turn_record(
        record_turn, config, text, latency_ms,
        classifier_intent="action",
        raw_reply=final,
        emitted_reply=final,
        messages=result_messages,
        extra_degraded=degraded or None,
    )
    return final


def _needs_spoken_followup(result_messages: list, elapsed_s: float, skip_s: float) -> bool:
    """The SILENT-SUCCESS RULE (owner requirement, 2026-07-03).

    A fast, successful device write needs no spoken follow-up — the light
    changing IS the feedback; a voice reciting "I turned off the light" after
    the light visibly went off is noise. Speak only when the caller can't
    already tell what happened:

    - the action ran SLOW (> skip_s after the ack) — silence would read as a
      dropped command;
    - any tool note is NOT a successful write (refusal, HA failure, or a
      get_state answer — a question's answer IS the follow-up);
    - no tool ran at all — the model's own sentence is the only outcome there is.

    Failures raised as exceptions never reach this helper — the caller speaks
    those unconditionally.
    """
    if elapsed_s > skip_s:
        return True
    from aerys_v2.tools.home_control import WRITE_OK_PREFIX

    tool_notes = [
        str(m.content) for m in result_messages if getattr(m, "type", "") == "tool"
    ]
    if not tool_notes:
        return True
    return not all(note.startswith(WRITE_OK_PREFIX) for note in tool_notes)


def _deliver_followup(
    text: str,
    device_id: str | None,
    followup_router: Callable[[str, str | None], None] | None,
    speak_fn: Callable[[str, str], None] | None,
    satellite_for: Callable[[str | None], str] | None,
) -> None:
    """Best-effort spoken follow-up, one place, fail-open. Prefers followup_router
    (per-device: mapped satellite -> announce, phone -> aerys_followup event); else
    the legacy speak_fn(text, resolved_entity) path (tests, dev). A delivery failure
    is logged and swallowed — the durable history write never depends on it."""
    try:
        if followup_router is not None:
            followup_router(text, device_id)
            return
        if speak_fn is not None and satellite_for is not None:
            entity_id = satellite_for(device_id)
            if entity_id is not None:
                speak_fn(text, entity_id)
    except Exception:
        log.warning("spoken follow-up delivery failed", exc_info=True)


# ---- effectful-claim detection (voice banter branch + text claim gate) ------
# Fires ONLY on a first-person, affirmative claim that an externally-visible act
# is DONE. Three tightenings came out of cross-review (2026-07-28), each closing
# a way normal conversation could have been mistaken for a fabrication:
#   - first person only: "the email was forwarded yesterday" is narration, not a
#     claim about this turn;
#   - affirmative only: a negation in the run-up ("I haven't sent it", "nothing
#     was logged") vetoes the match — refusing honestly is the BEHAVIOR WE WANT
#     and must never be punished by an escalation;
#   - the vocabulary covers how she actually phrases delivery ("passed it along",
#     "let Kael know", "Got it to Kael", a bare "Done."), because the gap this
#     serves was a Kael-relay claim.
_CLAIM_VERBS = (
    r"(?:sent|messaged|delivered|relayed|forwarded|emailed|texted|notified|"
    r"logged|filed|recorded|scheduled|queued|posted|shared|"
    r"passed (?:it |that |them )?(?:along|on)|"
    r"let (?:kael|kayle|cale|kale|kail|cael|him|her|them) know|got it to \w+)"
)
# What a BARE (subject-less) completion claim looks like: the verb takes a
# pronoun/particle object or simply ends. Narration takes a noun instead —
# "Sent messages appear in the log", "Forwarded mail lands in that folder" —
# and must not trip the gate (cross-review round 2, 2026-07-28).
_CLAIM_TAIL = (
    r"(?:\s+(?:it|that|this|them|those))?"
    r"(?:\s+(?:along|over|on|off|through|already|now|just now|for you|"
    r"to \w+|your way))*"
    r"\s*(?:[.!?]|$)"
)
_CLAIM_PATTERNS = (
    # "I sent it", "I've already logged that", "I just let Kael know"
    re.compile(
        rf"\b(?:i|i've|i have)\s+(?:just\s+|already\s+|now\s+)?{_CLAIM_VERBS}\b",
        re.I,
    ),
    # terse, subject-less: "Sent.", "Sent it over.", "— logged.", "Got it to Kael"
    re.compile(rf"(?:^|[.!?]\s+|[—-]\s*){_CLAIM_VERBS}{_CLAIM_TAIL}", re.I),
    # device + timer writes, first person only
    re.compile(
        r"\b(?:i|i've|i have)\s+(?:just\s+)?(?:turned|switched|set)\s+"
        r"(?:it|them|the|a|an|your)\b",
        re.I,
    ),
    # timer/alarm confirmations in either voice ("Timer set", "I set your alarm")
    re.compile(r"\b(?:timer|alarm|reminder)\s+(?:is\s+)?set\b", re.I),
    # a bare completion report
    re.compile(r"^\s*(?:done|all set|taken care of)\b", re.I),
)
# Checked in the run-up to a match, not across the whole reply, so "I sent it,
# but I couldn't reach the second one" still counts as a claim.
_NEGATION_RE = re.compile(
    r"\b(?:haven'?t|hasn'?t|hadn'?t|didn'?t|don'?t|doesn'?t|won'?t|can'?t|"
    r"cannot|couldn'?t|unable|never|not|nothing|no|without|failed|unfortunately)\b",
    re.I,
)
_NEGATION_WINDOW = 45


def claims_effect_done(text: str) -> bool:
    """Does this reply claim, in its own voice, that an effectful act happened?"""
    if not text:
        return False
    for pattern in _CLAIM_PATTERNS:
        for m in pattern.finditer(text):
            window = text[max(0, m.start() - _NEGATION_WINDOW): m.start()]
            if not _NEGATION_RE.search(window):
                return True
    return False


VOICE_EMPTY_REPLY = "Hm — I lost that thought. Ask me again?"

def _claims_effect_without_doing_it(user_text: str, reply: str) -> bool:
    """Did a TOOL-LESS chat reply claim it performed an effectful act?

    The 2026-07-25 incident, text side: asked to pass something to Kael, the
    chat-routed turn answered "Got it to Kael — sent" having called nothing.
    It COULDN'T have called anything: the chat graph carries no tools, so on
    this path a completion claim is unbacked by construction — there is no
    tool-call list to check, only the claim itself.

    Both signals are required, and the first is what keeps ordinary talk out of
    the gate: the user must have actually asked for something DONE (the
    router's own action heuristics), so reminiscing ("remember when you turned
    the lights off?") or hypotheticals never trip it — only a real request that
    the router sent to chat, answered as though it had been carried out.
    """
    return bool(
        plausibly_asks_for_action(user_text) and claims_effect_done(reply)
    )


def _voice_parallel_start(
    graph: object,
    text: str,
    config: dict,
    rails: Rails,
    started: float,
    router: Callable[[str], RouteDecision],
    action_graph: object,
    speak_fn: Callable[[str, str], None] | None,
    satellite_for: Callable[[str | None], str] | None,
    followup_skip_s: float,
    record_turn: Callable[[dict], None] | None = None,
    followup_router: Callable[[str, str | None], None] | None = None,
    display_push: Callable[[str, str | None], None] | None = None,
    content_privacy_classifier: Callable[[str], str] | None = None,
    human_privacy: str = PRIVATE,
    human_id: str | None = None,
    face_push: Callable[[str, str], None] | None = None,
    drop_unaddressed: bool = False,
    drop_conversation_window_s: float = 180.0,
    activity_registry: dict[str, float] | None = None,
) -> str:
    """Voice hot path — VOICE-ALWAYS-ACTION (owner's simplification, 2026-07-25).

    Every voice turn runs the tool-armed ACTION graph; the router's verdict now
    picks only the UX SHAPE, never the capability:

    - route=action: the proven ack-then-act path, unchanged — the router's ack
      goes to the speaker NOW, the tool loop finishes in a background thread,
      the real outcome lands in history + spoken follow-up per the silent-
      success rule.
    - route=chat (banter): the action graph runs SYNCHRONOUSLY on the real
      thread and its reply is spoken directly — single-reply UX, tools in hand
      if the model decides it needs one after all.

    What this deleted (and why): the speculative chat generation on a throwaway
    checkpointer thread, its seeding/copy-back/discard machinery, and the
    <<HANDOFF>> escalation branch for voice. The chat graph is out of the voice
    path entirely — which structurally removes both 2026-07-25 voice failures:
    the fabrication class ("Got it to Kael — sent" with zero tools: the chat
    graph HAD no tools, so it could only confabulate; the action graph has the
    tool in hand, and models with the tool in hand call it) and the chat
    backend's unreliable warm client. A router misroute is now a UX blemish
    (ack-shaped banter or direct-reply-shaped action), never a capability gap.

    Cross-review (Codex, GO-WITH-CHANGES) additions live here too: the banter
    branch carries a narrow EFFECTFUL-CLAIM postcondition (a zero-tool reply
    claiming a send/log/schedule is bounced once, then marked — general banter
    is never gated), and VOICE_BANTER_OVERLAY in factory.py keeps her voice
    persona intact on the tool-armed graph. The name is kept for history's
    sake; nothing races anymore.
    """
    real_configurable = config["configurable"]

    def _launch_background_action(ack: str, *, escalated: bool) -> str:
        """The ack-then-act tail: `ack` goes to the speaker NOW, a background
        thread finishes the tool loop, lands the real outcome in the thread,
        and follows up per the silent-success rule. (escalated= is kept for
        the audit-marker contract; nothing escalates from voice anymore.)"""
        ack_at = time.monotonic()  # the ack leaves for the speaker ~now
        # Her face speaks the ack; the pusher defers the working face until
        # the ack's estimated playback runs out (panel.py owns that timing).
        _face(face_push, "speaking", ack)
        _face(face_push, "working")

        # The ack the caller just heard rides `configurable` into the subgraph
        # (2026-07-03 incident): the action model must execute CONSISTENT with
        # what was already spoken — and must never ask a clarifying question,
        # because the announce channel is one-way. See VOICE_ACK_OVERLAY in
        # factory.py for the prompt-side half of this contract.
        action_config = {
            **config,
            "configurable": {
                **config["configurable"], "spoken_ack": ack, "specialist": True,
                "tier": normalize_tier(decision.tier),
            },
        }

        # Seed BEFORE the human turn lands (the seed appends the current human
        # itself), then write the human turn NOW — at ack time — so a slow
        # background action can never place the USER's words after a newer
        # turn's (cross-review finding 1). Only the AI result appends late,
        # which is semantically true: the action DID finish later. (Residual:
        # a late AI result can still land after a newer turn's pair — the full
        # placeholder-replace fix is deliberately out of tonight's scope.)
        seeded_messages = _action_history_seed(
            graph, real_configurable, text, specialist=True
        )
        graph.update_state(
            {"configurable": real_configurable},
            {"messages": [_human_turn(text, human_privacy, human_id)]},
            as_node="chat",
        )

        def _complete_action() -> None:
            failed = False
            result_messages: list = []
            gate_degraded: list[str] = []
            try:
                # Stateless-voice continuity (owner ask 2026-07-05) rides the
                # precomputed seed; the 2026-07-03 garble incident is guarded by
                # VOICE_ACK_OVERLAY (spoken_ack in action_config). The gate
                # bounces a zero-tool action once and marks 'no_tool_action' if
                # it still touched nothing.
                result, gate_degraded = _run_action_gated(
                    action_graph,
                    seeded_messages,
                    action_config,
                )
                result_messages = result["messages"]
                final = _strip_handoff(_reply_text(result_messages[-1]))
            except Exception as e:  # honest failure into history, never silence
                log.warning("background action turn failed", exc_info=True)
                final = _honest_reply_for_failure(e) or f"(The action didn't complete — {e})"
                failed = True

            # Spoken follow-up: failures ALWAYS speak; otherwise the
            # silent-success rule decides (fast clean write = the device is
            # the feedback, say nothing).
            elapsed = time.monotonic() - ack_at
            device_id = real_configurable.get("identity", {}).get("device_id")
            spoke_followup = failed or _needs_spoken_followup(
                result_messages, elapsed, followup_skip_s
            )
            if spoke_followup:
                _deliver_followup(
                    final, device_id, followup_router, speak_fn, satellite_for
                )
            _face(face_push, "speaking" if spoke_followup else "idle", final)

            # History write happens EITHER WAY (silent record) — the next turn's
            # model must see what actually happened, spoken aloud or not. The
            # human turn already landed at ack time; only the outcome appends.
            graph.update_state(
                {"configurable": real_configurable},
                {"messages": [AIMessage(content=final)]},
                as_node="chat",
            )
            _reclassify_if_needed(
                graph, config, human_id, text, final,
                content_privacy_classifier, human_privacy,
            )

            # Audit — off the hot path by construction (the caller got the ack
            # long ago). emitted_reply is the ACK the caller actually heard;
            # raw_reply is the action's real outcome.
            degraded = ["action_failed"] if failed else list(gate_degraded or [])
            if escalated:
                degraded.append(ESCALATED_MARKER)
            _fire_turn_record(
                record_turn, config, text,
                int((time.monotonic() - started) * 1000),
                classifier_intent="action",
                raw_reply=final,
                emitted_reply=ack,
                messages=result_messages,
                extra_degraded=degraded or None,
                error=final if failed else None,
            )

        threading.Thread(target=_in_ctx(_complete_action), daemon=True).start()
        return ack

    decision = router(text)
    registry = _THREAD_ACTIVITY if activity_registry is None else activity_registry
    thread_key = str(real_configurable.get("thread_id", ""))
    if (
        drop_unaddressed
        and decision.unaddressed
        and not _conversation_in_flight(
            registry, thread_key, drop_conversation_window_s
        )
    ):
        # False-wake grace (owner ask 2026-08-27): the router judged this
        # capture was never directed at Aerys — a wake-word misfire during a
        # human conversation. Drop it Alexa-style: nothing spoken, nothing run,
        # and the human turn is NOT written into the thread history (a stray
        # fragment must not pollute her memory of the conversation) — but the
        # receipt row ALWAYS lands, so drop judgment is auditable and tunable.
        log.info(
            "voice route decision | thread=%s DROPPED (unaddressed capture)",
            real_configurable.get("thread_id"),
        )
        _fire_turn_record(
            record_turn, config, text,
            int((time.monotonic() - started) * 1000),
            classifier_intent="unaddressed",
            raw_reply="", emitted_reply="",
            extra_degraded=[DROPPED_UNADDRESSED_MARKER],
        )
        return ""
    if decision.route == "action":
        log.info(
            "voice route decision | thread=%s route=action",
            real_configurable.get("thread_id"),
        )
        return _launch_background_action(decision.ack, escalated=False)

    # ---- banter branch: synchronous action-graph turn, single spoken reply ----
    log.info(
        "voice route decision | thread=%s route=chat (action graph, synchronous)",
        real_configurable.get("thread_id"),
    )
    _face(face_push, "working")
    gate_degraded: list[str] = []
    try:
        # NO spoken_ack in config: that's the signal (factory.act) to style the
        # reply for voice banter instead of arming the ack-consistency overlay.
        result = action_graph.invoke(
            {"messages": _action_history_seed(graph, real_configurable, text)},
            config,
        )
        result_messages = result["messages"]
        reply = _strip_handoff(_reply_text(result_messages[-1])) or VOICE_EMPTY_REPLY
        # Effectful-claim postcondition (the 2026-07-25 22:08 fabrication class,
        # narrowed per cross-review): a ZERO-tool reply claiming an externally
        # effectful act gets one bounce with the same correction the action gate
        # uses; still claiming tool-free -> emit but mark for the audit trail.
        if not extract_tool_calls(result_messages) and claims_effect_done(reply):
            log.info("voice banter effectful-claim gate: zero tools — bouncing once")
            retry_messages = [
                *result_messages, HumanMessage(content=ACTION_NO_TOOL_CORRECTION),
            ]
            result = action_graph.invoke({"messages": retry_messages}, config)
            result_messages = result["messages"]
            reply = _strip_handoff(_reply_text(result_messages[-1])) or VOICE_EMPTY_REPLY
            if not extract_tool_calls(result_messages) and claims_effect_done(reply):
                log.info(
                    "voice banter effectful-claim gate: STILL claiming with zero "
                    "tools — emitting with the %s marker", NO_TOOL_ACTION_MARKER,
                )
                gate_degraded = [NO_TOOL_ACTION_MARKER]
    except Exception as e:
        # A rate/session-limit cap gets the honest in-voice line (the 2026-07-12
        # empty-glasses fix); anything else records the failure and re-raises.
        honest = _honest_reply_for_failure(e)
        _record_turn_failure(
            record_turn, config, text, started, e,
            classifier_intent="chat", tier=DEFAULT_TIER,
            emitted_reply=honest,
        )
        if honest is None:
            raise
        # The user HEARS this line — so the thread must carry it too, exactly
        # like the background path's honest failures (cross-review finding 4:
        # a spoken reply the next turn's model can't see is a continuity hole).
        graph.update_state(
            {"configurable": real_configurable},
            {"messages": [_human_turn(text, human_privacy, human_id), AIMessage(content=honest)]},
            as_node="chat",
        )
        _face(face_push, "speaking", honest)
        return honest

    # The action graph has no checkpointer — the turn lands on the REAL thread
    # here, exactly one human + one assistant message (cross-review history
    # invariant: a banter turn must never produce a reply AND a follow-up).
    graph.update_state(
        {"configurable": real_configurable},
        {"messages": [_human_turn(text, human_privacy, human_id), AIMessage(content=reply)]},
        as_node="chat",
    )

    elapsed = time.monotonic() - started
    timed_out = elapsed > rails.wall_clock_s
    timeout_msg = (
        f"turn took {elapsed:.1f}s (budget {rails.wall_clock_s}s)"
        if timed_out else None
    )
    # Voice banter stays recorded as intent=chat (the router's honest verdict);
    # the graph it ran on is an implementation detail the audit needn't rename.
    _fire_turn_record(
        record_turn, config, text, int(elapsed * 1000),
        classifier_intent="chat",
        tier=DEFAULT_TIER,
        raw_reply=reply,
        emitted_reply=reply,
        messages=result_messages,
        extra_degraded=(gate_degraded + (["wall_clock_exceeded"] if timed_out else [])) or None,
        error=timeout_msg,
    )
    if timed_out:
        # Late but REAL: the reply exists and is already in history, so return
        # it (the pipeline may still be listening) instead of raising into a
        # 500 — a raise here would leave the audit claiming an emitted reply
        # nobody heard and her face stuck on 'working' (cross-review finding 3).
        log.warning("voice banter turn exceeded wall clock: %s", timeout_msg)
    _reclassify_if_needed(
        graph, config, human_id, text, reply,
        content_privacy_classifier, human_privacy,
    )
    # Gap #48: the pipeline event that mirrors this reply onto an e-ink
    # display is capped at 500 bytes by upstream firmware — push the FULL
    # text through the uncapped display door (closure delays so it lands
    # after the capped write and wins the ink; non-display devices no-op).
    if display_push is not None:
        try:
            display_push(
                reply, real_configurable.get("identity", {}).get("device_id")
            )
        except Exception:
            log.warning("display push failed (spoken reply unaffected)",
                        exc_info=True)
    # The pipeline TTS speaks this return value; the pusher's estimate settles
    # her back to the reply's mood-idle when the words run out.
    _face(face_push, "speaking", reply)
    return reply
