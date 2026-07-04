"""ask() — the single seam every transport calls.

n8n mapping: this is the Execute Workflow boundary into the Core Agent, done as one
function. Discord, Telegram, the voice endpoint, and the CLI will ALL call ask() and
nothing else — so safety rails, auditing, and tracing added here cover every channel
at once (in n8n the same fix had to be patched into each adapter separately).

TOOLS block (Option C hybrid, owner-ratified): ask() optionally takes a router and
an action subgraph. Both None = chat-only, byte-for-byte the old behavior. Both set:

- non-voice threads: router first (sequential) — chat routes to the chat graph,
  action routes to the tool subgraph, whose result becomes the reply.
- voice threads (thread_id startswith "voice"): PARALLEL-START — the router and
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
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Callable

from langchain_core.messages import AIMessage, HumanMessage

from aerys_v2.router import DEFAULT_TIER, RouteDecision, normalize_tier
from aerys_v2.state import Identity
from aerys_v2.turns import build_turn_row, current_trace_id

log = logging.getLogger(__name__)


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
) -> None:
    """One v2_turns row for a turn whose invoke RAISED, fired before the caller
    re-raises. Degraded marker is 'recursion_limit' for a rail trip, else
    'turn_failed'; the exception text rides `error`. This is what makes the docstring
    promise — a row on the error exits, not just the timeout exit — literally true
    (cross-review correctness H)."""
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


def ask(
    graph: object,
    text: str,
    *,
    identity: Identity,
    thread_id: str,
    rails: Rails = Rails(),
    router: Callable[[str], RouteDecision] | None = None,
    action_graph: object | None = None,
    speak_fn: Callable[[str, str], None] | None = None,
    satellite_for: Callable[[str | None], str] | None = None,
    followup_router: Callable[[str, str | None], None] | None = None,
    followup_skip_s: float = 6.0,
    deep_allowed: Callable[[], bool] | None = None,
    action_allowlist: frozenset[str] | None = None,
    record_turn: Callable[[dict], None] | None = None,
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
    - action_allowlist: the AUTH gate for the TOOLS block. The action stack (house
      control + every tool) is restricted to an allowlist of person_ids: a caller
      NOT in it is forced onto the chat-only path here, regardless of what the
      transport armed. The memory boundary makes a stranger's identity COLD (no
      memories) but does NOTHING to the tools — so this is the one thing between a
      guild member and the owner's house. The owner is always in the set; more can
      be added by config (e.g. Megan) with no code change (factory.action_allowlist_for).
      None = unenforced (dev boxes), same posture as deep_allowed.
    - record_turn: the v2_turns audit seam (factory.turn_recorder_for). Called
      once per completed turn on EVERY path with the fully-built row, off the hot
      path and fail-open (see _fire_turn_record). None = no auditing (dev/CI, no
      DATABASE_URL), byte-for-byte the old behavior.
    """
    if not text or not text.strip():
        raise ValueError("ask() requires non-empty text")

    # Gate the action stack BEFORE anything else can arm it. A caller outside the
    # allowlist never reaches home_control / search_entities / get_state — closing
    # both the unauthorized-actuation and the presence-disclosure (reads) risks.
    if action_allowlist is not None and identity.get("user_id") not in action_allowlist:
        router = None
        action_graph = None

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
            return _chat_turn(graph, text, config, rails, started, record_turn=record_turn)

        if str(thread_id).startswith("voice"):
            # ChannelPolicy (locked): voice is PINNED to the standard tier —
            # the ~3.6s budget can't absorb deep latency, and fast-tier
            # identity wobbles are what got Haiku demoted in V1. The pin is
            # structural: this path never writes a tier into config, so the
            # chat node's DEFAULT_TIER (= standard) always applies.
            return _voice_parallel_start(
                graph, text, config, rails, started, router, action_graph,
                speak_fn, satellite_for, followup_skip_s, record_turn=record_turn,
                followup_router=followup_router,
            )

        # Non-voice: nobody is waiting on a speaker, so the router runs first
        # (sequential) and only the chosen path spends model tokens.
        decision = router(text)
        if decision.route == "action":
            # add_human=True: the chat graph never saw this turn, so BOTH the human
            # message and the action result must land in the thread history.
            log.info("route decision | thread=%s route=action", thread_id)
            return _action_turn(
                action_graph, graph, text, config, add_human=True,
                record_turn=record_turn, started=started,
            )

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
        return _chat_turn(
            graph, text, config, rails, started, record_turn=record_turn,
            classifier_intent="chat", tier=tier,
            tier_override_source=override_source, extra_degraded=downgrade_marker,
        )


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
) -> str:
    """The original chat path: invoke, budget-check, extract — now also audited.

    The v2_turns row is fired on BOTH exits: the normal return AND the timeout
    raise (the reply exists either way — a turn that ran past budget is exactly
    the kind of thing forensics need to see). raw_reply == emitted_reply here:
    the chat path has no separate polish step (V1's Gemini polisher is now
    prompt-side emotion tags), so what the model said IS what the channel emits.
    """
    try:
        result = graph.invoke({"messages": [HumanMessage(content=text)]}, config)
    except Exception as e:
        # A raised invoke (model 500, recursion-rail trip) is the HIGHEST-value turn
        # for forensics and the capability loop — record it BEFORE re-raising so the
        # 'row on EVERY completion path incl. error' contract actually holds
        # (cross-review correctness H). Then propagate unchanged: the caller still sees
        # the failure exactly as before.
        _record_turn_failure(
            record_turn, config, text, started, e,
            classifier_intent=classifier_intent,
            tier=tier,
            tier_override_source=tier_override_source,
            base_degraded=extra_degraded,
        )
        raise
    reply = _reply_text(result["messages"][-1])

    elapsed = time.monotonic() - started
    timed_out = elapsed > rails.wall_clock_s
    timeout_msg = (
        f"turn took {elapsed:.1f}s (budget {rails.wall_clock_s}s)" if timed_out else None
    )
    degraded = list(extra_degraded or [])
    if timed_out:
        degraded.append("wall_clock_exceeded")

    _fire_turn_record(
        record_turn, config, text, int(elapsed * 1000),
        classifier_intent=classifier_intent,
        tier=tier,
        tier_override_source=tier_override_source,
        raw_reply=reply,
        emitted_reply=reply,
        messages=result["messages"],
        extra_degraded=degraded or None,
        error=timeout_msg,
    )

    if timed_out:
        # The reply exists but arrived past budget — surface it loudly rather than
        # silently normalizing a degraded experience (voice cares at ~4s, not 90).
        raise TurnTimeout(timeout_msg)

    return reply


def _action_turn(
    action_graph: object,
    graph: object,
    text: str,
    config: dict,
    *,
    add_human: bool,
    record_turn: Callable[[dict], None] | None = None,
    started: float | None = None,
) -> str:
    """Run the tool subgraph, then land the outcome in the MAIN thread's history.

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
        result = action_graph.invoke({"messages": [HumanMessage(content=text)]}, config)
    except Exception as e:
        # Rail trip / tool-loop blowup on the sequential action path: record the
        # failed turn (classifier_intent='action') before re-raising, same as the
        # chat path (cross-review correctness H).
        _record_turn_failure(
            record_turn, config, text, started, e, classifier_intent="action"
        )
        raise
    result_messages = result["messages"]
    final = _reply_text(result_messages[-1])
    messages: list = [AIMessage(content=final)]
    if add_human:
        messages.insert(0, HumanMessage(content=text))
    graph.update_state(
        {"configurable": config["configurable"]}, {"messages": messages}, as_node="chat"
    )

    latency_ms = int((time.monotonic() - started) * 1000) if started is not None else None
    _fire_turn_record(
        record_turn, config, text, latency_ms,
        classifier_intent="action",
        raw_reply=final,
        emitted_reply=final,
        messages=result_messages,
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
    # keeps the "voice" prefix, so the chat node's voice styling still applies
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

        if decision.route != "action":
            # Chat wins: the generation is already in flight — the router's
            # latency hid entirely inside the chat call's shadow. It ran on the
            # throwaway thread, so the turn must be copied into the REAL thread
            # here or the conversation never durably happened.
            try:
                result = chat_future.result()
                reply_message = result["messages"][-1]
                graph.update_state(
                    {"configurable": real_configurable},
                    {"messages": [HumanMessage(content=text), reply_message]},
                    as_node="chat",
                )
            except Exception as e:
                # Speculative voice-chat generation raised: record the failed voice
                # turn (pinned standard) before re-raising, so the error exit audits
                # like the others (cross-review correctness H).
                _record_turn_failure(
                    record_turn, config, text, started, e,
                    classifier_intent="chat", tier=DEFAULT_TIER,
                )
                raise
            finally:
                _discard_speculative()
            elapsed = time.monotonic() - started
            timed_out = elapsed > rails.wall_clock_s
            timeout_msg = (
                f"turn took {elapsed:.1f}s (budget {rails.wall_clock_s}s)"
                if timed_out else None
            )
            reply = _reply_text(reply_message)
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
            return reply

        # Action: the ack goes out NOW; the tool loop finishes in the background.
        # Best-effort cancel of the speculative chat call — if it already
        # started (fake models finish instantly; real ones usually haven't
        # begun streaming), it burns tokens into the throwaway thread and gets
        # deleted below. Its text can never reach the real thread on this path.
        chat_cancelled = chat_future.cancel()
        ack_at = time.monotonic()  # the ack leaves for the speaker ~now

        # The ack the caller just heard rides `configurable` into the subgraph
        # (2026-07-03 incident): the action model must execute CONSISTENT with
        # what was already spoken — and must never ask a clarifying question,
        # because the announce channel is one-way. See VOICE_ACK_OVERLAY in
        # factory.py for the prompt-side half of this contract.
        action_config = {
            **config,
            "configurable": {**config["configurable"], "spoken_ack": decision.ack},
        }

        def _complete_action() -> None:
            failed = False
            result_messages: list = []
            try:
                result = action_graph.invoke(
                    {"messages": [HumanMessage(content=text)]}, action_config
                )
                result_messages = result["messages"]
                final = _reply_text(result_messages[-1])
            except Exception as e:  # honest failure into history, never silence
                log.warning("background action turn failed", exc_info=True)
                final = f"(The action didn't complete — {e})"
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
            if failed or _needs_spoken_followup(result_messages, elapsed, followup_skip_s):
                _deliver_followup(
                    final, device_id, followup_router, speak_fn, satellite_for
                )

            # History write happens EITHER WAY (silent record) — the next turn's
            # model must see what actually happened, spoken aloud or not. The
            # speculative gen wrote ONLY to the throwaway thread, so the human
            # turn is never in the real thread yet and there is no checkpoint
            # interleave to wait out — the durable record lands immediately.
            graph.update_state(
                {"configurable": real_configurable},
                {"messages": [HumanMessage(content=text), AIMessage(content=final)]},
                as_node="chat",
            )
            if not chat_cancelled:
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
            _fire_turn_record(
                record_turn, config, text,
                int((time.monotonic() - started) * 1000),
                classifier_intent="action",
                raw_reply=final,
                emitted_reply=decision.ack,
                messages=result_messages,
                extra_degraded=["action_failed"] if failed else None,
                error=final if failed else None,
            )

        threading.Thread(target=_in_ctx(_complete_action), daemon=True).start()
        return decision.ack
    finally:
        # Never block the reply on stragglers — the background thread (plain
        # threading.Thread, not pool-owned) outlives this scope by design.
        pool.shutdown(wait=False)
