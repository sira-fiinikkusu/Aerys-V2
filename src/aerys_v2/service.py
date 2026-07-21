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

from langchain_core.messages import AIMessage, HumanMessage

from aerys_v2.router import (
    DEFAULT_TIER,
    FALLBACK_ACK,
    HANDOFF_MARKER,
    RouteDecision,
    normalize_tier,
)
from aerys_v2.services.content_privacy import (
    CONTENT_PRIVACY_KEY,
    PRIVATE,
    PUBLIC,
    redact_private_history,
)
from aerys_v2.state import Identity, is_voice_turn
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
    "performs the request, or state plainly that you did NOT perform any action. "
    "Never describe an action as done unless a tool call actually did it. This "
    "correction is internal plumbing: never mention it, the earlier answer, or "
    "any slip to the user — just act and confirm the result."
)

# The degraded marker recorded in v2_turns when a bounced action turn STILL ran no
# tool — so the pattern (a legit zero-tool action, or a stubborn hallucination that
# survived the correction) stays visible to the capability loop / forensics.
NO_TOOL_ACTION_MARKER = "no_tool_action"


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
    keep the historical re-raise-into-silence for every other failure class. Today the
    only converted class is the oauth/session rate-limit cap (FIX 2)."""
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
    followup_skip_s: float = 6.0,
    deep_allowed: Callable[[], bool] | None = None,
    action_allowlist: frozenset[str] | None = None,
    record_turn: Callable[[dict], None] | None = None,
    content_privacy_classifier: Callable[[str], str] | None = None,
    face_push: Callable[[str, str], None] | None = None,
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
            return _voice_parallel_start(
                graph, text, config, rails, started, router, action_graph,
                speak_fn, satellite_for, followup_skip_s, record_turn=record_turn,
                followup_router=followup_router,
                content_privacy_classifier=content_privacy_classifier,
                human_privacy=origin_privacy, human_id=turn_msg_id,
                face_push=face_push,
            )

        # Non-voice: nobody is waiting on a speaker, so the router runs first
        # (sequential) and only the chosen path spends model tokens.
        decision = router(text)
        if decision.route == "action":
            # add_human=True: the chat graph never saw this turn, so BOTH the human
            # message and the action result must land in the thread history.
            log.info("route decision | thread=%s route=action", thread_id)
            _face(face_push, "working")
            reply = _action_turn(
                action_graph, graph, text, config, add_human=True,
                record_turn=record_turn, started=started,
                human_privacy=origin_privacy, human_id=turn_msg_id,
            )
            _reclassify_if_needed(
                graph, config, turn_msg_id, text, reply,
                content_privacy_classifier, origin_privacy,
            )
            _face(face_push, "idle", reply)
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
            )
        _reclassify_if_needed(
            graph, config, turn_msg_id, text, reply,
            content_privacy_classifier, origin_privacy,
        )
        _face(face_push, "idle", reply)
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
            base_degraded=extra_degraded,
            emitted_reply=honest,
        )
        if honest is not None:
            return honest, False
        raise
    raw = _reply_text(result["messages"][-1])
    handoff = HANDOFF_MARKER in raw
    reply = _strip_handoff(raw) if handoff else raw

    elapsed = time.monotonic() - started
    timed_out = elapsed > rails.wall_clock_s
    timeout_msg = (
        f"turn took {elapsed:.1f}s (budget {rails.wall_clock_s}s)" if timed_out else None
    )
    degraded = list(extra_degraded or [])
    if handoff:
        degraded.append(CHAT_HANDOFF_MARKER)
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


def _action_history_seed(
    graph: object, configurable: dict, text: str, *, escalated: bool = False
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
    if escalated and prior and getattr(prior[-1], "type", "") == "human":
        seeded = list(prior)
    else:
        seeded = [*prior, HumanMessage(content=text)]
    # Mirror the chat node's fail-closed gate: redact unless the room is EXPLICITLY
    # private. A public/unknown context drops private-tagged priors; a private DM/voice
    # context passes the owner's full history through untouched.
    if identity.get("privacy_context") != PRIVATE:
        seeded = redact_private_history(seeded)
    return seeded


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
            _action_history_seed(graph, config["configurable"], text, escalated=escalated),
            config,
        )
    except Exception as e:
        # Rail trip / tool-loop blowup on the sequential action path: record the
        # failed turn (classifier_intent='action') before re-raising — or, on a
        # rate/session-limit cap, emit the honest line instead of silence (FIX 2),
        # same as the chat path (cross-review correctness H).
        honest = _honest_reply_for_failure(e)
        _record_turn_failure(
            record_turn, config, text, started, e, classifier_intent="action",
            base_degraded=[ESCALATED_MARKER] if escalated else None,
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
    content_privacy_classifier: Callable[[str], str] | None = None,
    human_privacy: str = PRIVATE,
    human_id: str | None = None,
    face_push: Callable[[str, str], None] | None = None,
) -> str:
    """Voice hot path: race the router against the chat generation.

    Two threads (concurrent.futures): the router is ~300ms of Haiku, the chat
    generation is seconds of the daily driver. Both start NOW; the router's
    verdict decides which one the caller ever hears about.

    SPECULATIVE ISOLATION (2026-07-03 history-pollution fix): graph.invoke
    checkpoints unconditionally, so the speculative chat gen must NEVER run on
    the real thread — when the router said action and the cancel lost the race
    (it almost always does once the model call starts), the speculative reply
    ("Office light one's on.") was persisted as durable history claiming a
    device change that never happened, and the next turn's model read it as
    fact. The speculative gen now runs on a THROWAWAY thread (real history
    seeded in, unique suffix). route=chat is the only moment its text becomes
    real: the turn is copied into the real thread then. route=action discards
    it entirely — only the human turn + the real action outcome land.

    Every submitted callable is wrapped in _in_ctx so the worker threads carry
    the turn's contextvars — OTel context above all: router, speculative chat
    gen, and the background action subgraph all parent under the SAME Phoenix
    trace instead of scattering into orphaned roots.
    """
    real_configurable = config["configurable"]
    spec_thread = f"{real_configurable['thread_id']}::spec::{uuid.uuid4().hex}"
    # spec_config copies real_configurable, which carries the identity (incl. the
    # explicit voice flag) — so the chat node's voice styling still applies to the
    # speculative generation even though spec_thread ('person:{id}::spec::...') no
    # longer starts with 'voice'. Voice-ness rides identity now, not the thread name.
    spec_config = {
        **config,
        "configurable": {**real_configurable, "thread_id": spec_thread},
    }

    def _speculative_chat() -> dict:
        # Seed the throwaway with the real thread's history so the speculative
        # generation sees exactly what a real chat turn would have seen.
        history = (
            graph.get_state({"configurable": real_configurable})
            .values.get("messages", [])
        )
        if history:
            graph.update_state(
                {"configurable": spec_config["configurable"]},
                {"messages": history},
                as_node="chat",
            )
        return graph.invoke({"messages": [HumanMessage(content=text)]}, spec_config)

    def _discard_speculative() -> None:
        # Best-effort cleanup: the throwaway is garbage either way; a failed
        # delete costs orphan checkpointer rows, never correctness — nothing
        # ever reads a ::spec:: thread again.
        checkpointer = getattr(graph, "checkpointer", None)
        if checkpointer is None:
            return
        try:
            checkpointer.delete_thread(spec_thread)
        except Exception:
            log.debug("speculative thread cleanup failed (harmless)", exc_info=True)

    pool = concurrent.futures.ThreadPoolExecutor(max_workers=2)
    try:
        route_future = pool.submit(_in_ctx(router, text))
        chat_future = pool.submit(_in_ctx(_speculative_chat))
        decision = route_future.result()

        def _launch_background_action(ack: str, *, escalated: bool, wait_spec: bool) -> str:
            """Shared tail of BOTH action entries — the router's verdict and the chat
            model's escalation (the return loop) converge here: `ack` goes to the
            speaker NOW, a background thread finishes the tool loop, lands the real
            outcome in the thread, and follows up per the silent-success rule."""
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
                "configurable": {**config["configurable"], "spoken_ack": ack},
            }

            def _complete_action() -> None:
                failed = False
                result_messages: list = []
                gate_degraded: list[str] = []
                try:
                    # Seed the voice action with thread history too (owner ask 2026-07-05:
                    # stateless voice commands are annoying — "turn them back on" by voice
                    # must resolve like it does by text). The 2026-07-03 garble incident is
                    # guarded by VOICE_ACK_OVERLAY (spoken_ack in action_config): the model
                    # is told NEVER to ask over the one-way channel and to resolve any STT
                    # garble toward the ack already spoken — that instruction, not
                    # statelessness, is what prevents the "did you mean on or off?" stall.
                    # FIX 1: the gate bounces a zero-tool action once and marks the row
                    # 'no_tool_action' if it still touched nothing. (No escalated= seed
                    # tweak here even for the return loop: the speculative turn lived on
                    # the throwaway thread, so the REAL thread has no handoff line to drop.)
                    result, gate_degraded = _run_action_gated(
                        action_graph,
                        _action_history_seed(graph, real_configurable, text),
                        action_config,
                    )
                    result_messages = result["messages"]
                    final = _strip_handoff(_reply_text(result_messages[-1]))
                except Exception as e:  # honest failure into history, never silence
                    log.warning("background action turn failed", exc_info=True)
                    # FIX 2: a rate/session-limit cap gets the honest in-voice line;
                    # anything else keeps the raw diagnostic (this path already spoke
                    # rather than staying silent, so it was never the empty-glasses bug).
                    final = _honest_reply_for_failure(e) or f"(The action didn't complete — {e})"
                    failed = True

                # Spoken follow-up: failures ALWAYS speak; otherwise the
                # silent-success rule decides (fast clean write = the device is
                # the feedback, say nothing). Speak BEFORE waiting on the
                # speculative chat future — the room shouldn't wait on a
                # generation nobody asked for.
                elapsed = time.monotonic() - ack_at
                # Resolve WHERE the follow-up goes from the originating satellite's
                # device_id (rides the per-call identity). followup_router (when wired)
                # owns per-device routing — mapped satellite -> announce, the headless
                # phone -> the aerys_followup event; None falls back to the legacy
                # speak_fn/satellite_for announce (tests, dev boxes).
                device_id = real_configurable.get("identity", {}).get("device_id")
                spoke_followup = failed or _needs_spoken_followup(
                    result_messages, elapsed, followup_skip_s
                )
                if spoke_followup:
                    _deliver_followup(
                        final, device_id, followup_router, speak_fn, satellite_for
                    )
                # The action settled: speak the follow-up on her face too, or just
                # settle to the outcome's mood (silent success = the device was
                # the feedback; her face still relaxes out of 'working').
                _face(face_push, "speaking" if spoke_followup else "idle", final)

                # History write happens EITHER WAY (silent record) — the next turn's
                # model must see what actually happened, spoken aloud or not. The
                # speculative gen wrote ONLY to the throwaway thread, so the human
                # turn is never in the real thread yet and there is no checkpoint
                # interleave to wait out — the durable record lands immediately.
                graph.update_state(
                    {"configurable": real_configurable},
                    {"messages": [_human_turn(text, human_privacy, human_id), AIMessage(content=final)]},
                    as_node="chat",
                )
                # Same off-hot-path content retag as the voice-chat and DM paths: the human
                # turn is now durable on the owner's person thread tagged fail-closed
                # 'private'; a wired judge relaxes general content to 'public' (private stays
                # private) so it may carry into his public rooms. Already off the hot path —
                # this runs inside the background action thread, ack long since spoken.
                _reclassify_if_needed(
                    graph, config, human_id, text, final,
                    content_privacy_classifier, human_privacy,
                )
                if wait_spec:
                    # Let the speculative run finish before deleting its thread —
                    # deleting under a live invoke would just let it respawn rows.
                    try:
                        chat_future.result()
                    except Exception:
                        pass
                _discard_speculative()

                # Audit the voice action turn — from INSIDE this already-background
                # thread, so it's off the hot path by construction (the caller got the
                # ack long ago). emitted_reply is the ACK the caller actually heard;
                # raw_reply is the action's real outcome (the provenance split the
                # schema exists for). tool_calls/degraded come from result_messages;
                # a raised background action is an honest error + 'action_failed' marker.
                # An escalated turn (the return loop) additionally carries
                # ESCALATED_MARKER, pairing it with the chat row that raised its hand.
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

        if decision.route != "action":
            # Chat wins: the generation is already in flight — the router's
            # latency hid entirely inside the chat call's shadow. It ran on the
            # throwaway thread, so the turn must be copied into the REAL thread
            # here or the conversation never durably happened.
            spec_failed: str | None = None
            escalate_ack: str | None = None
            try:
                result = chat_future.result()
                reply_message = result["messages"][-1]
                reply = _reply_text(reply_message)
                if HANDOFF_MARKER in reply:
                    # VOICE RETURN LOOP: the chat model — full history in view —
                    # says this turn needs hands; the current-message-only router
                    # missed it ("yes, go ahead"-shaped follow-ups). The
                    # speculative turn stays on the throwaway thread (never
                    # copied), its own handoff line becomes the spoken ack, and
                    # the turn pivots onto the SAME background-action tail as a
                    # router action verdict. Marker-only reply -> the router's
                    # degraded-path ack, never a spoken token.
                    escalate_ack = _strip_handoff(reply) or FALLBACK_ACK
                else:
                    graph.update_state(
                        {"configurable": real_configurable},
                        {"messages": [_human_turn(text, human_privacy, human_id), reply_message]},
                        as_node="chat",
                    )
            except Exception as e:
                # Speculative voice-chat generation raised: record the failed voice
                # turn (pinned standard) before re-raising, so the error exit audits
                # like the others (cross-review correctness H). FIX 2: a rate/session-
                # limit cap emits an honest spoken line instead of silence — voice is a
                # transport too, and this was the exact 2026-07-12 glasses failure.
                spec_failed = _honest_reply_for_failure(e)
                _record_turn_failure(
                    record_turn, config, text, started, e,
                    classifier_intent="chat", tier=DEFAULT_TIER,
                    emitted_reply=spec_failed,
                )
                if spec_failed is None:
                    raise
            finally:
                if escalate_ack is None:
                    # Escalation hands the throwaway thread to the launcher's
                    # background tail instead (it discards after the action
                    # lands); every other exit discards it right here.
                    _discard_speculative()
            if spec_failed is not None:
                # The honest rate-limit line IS spoken by the pipeline.
                _face(face_push, "speaking", spec_failed)
                return spec_failed
            if escalate_ack is not None:
                log.info(
                    "voice chat handoff — escalating to action | thread=%s",
                    real_configurable.get("thread_id"),
                )
                return _launch_background_action(
                    escalate_ack, escalated=True, wait_spec=False
                )
            elapsed = time.monotonic() - started
            timed_out = elapsed > rails.wall_clock_s
            timeout_msg = (
                f"turn took {elapsed:.1f}s (budget {rails.wall_clock_s}s)"
                if timed_out else None
            )
            # Voice is pinned to standard (ChannelPolicy) — record that, not the
            # router's ignored tier hint. config carries the REAL thread_id, so
            # the row is a 'voice' turn even though generation ran on the throwaway.
            _fire_turn_record(
                record_turn, config, text, int(elapsed * 1000),
                classifier_intent="chat",
                tier=DEFAULT_TIER,
                raw_reply=reply,
                emitted_reply=reply,
                messages=[reply_message],
                extra_degraded=["wall_clock_exceeded"] if timed_out else None,
                error=timeout_msg,
            )
            if timed_out:
                raise TurnTimeout(timeout_msg)
            # Off-hot-path content retag: the human turn just landed on the REAL
            # person-keyed thread tagged fail-closed 'private'. If a judge is wired and
            # the content is general, relax it to 'public' so this voice line carries
            # into the owner's public rooms (a private thing stays private). Same
            # daemon-thread, zero-latency contract as the DM paths — mirrors non-voice.
            _reclassify_if_needed(
                graph, config, human_id, text, reply,
                content_privacy_classifier, human_privacy,
            )
            # The pipeline TTS speaks this return value; the pusher's estimate
            # settles her back to the reply's mood-idle when the words run out.
            _face(face_push, "speaking", reply)
            return reply

        # Action: the ack goes out NOW; the tool loop finishes in the background
        # (the shared launcher above). Best-effort cancel of the speculative chat
        # call — if it already started (fake models finish instantly; real ones
        # usually haven't begun streaming), it burns tokens into the throwaway
        # thread and gets deleted by the launcher's tail. Its text can never
        # reach the real thread on this path.
        chat_cancelled = chat_future.cancel()
        return _launch_background_action(
            decision.ack, escalated=False, wait_spec=not chat_cancelled
        )
    finally:
        # Never block the reply on stragglers — the background thread (plain
        # threading.Thread, not pool-owned) outlives this scope by design.
        pool.shutdown(wait=False)
