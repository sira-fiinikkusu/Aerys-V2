"""One-message Jev judgments. Evidence for a future router, never today's driver."""
from __future__ import annotations

import contextvars
import logging
import threading
import time
from typing import Callable

from aerys_v2.config import Settings
from aerys_v2.router import FALLBACK_ACK, RouteDecision, fallback_decision, normalize_tier

# Phase 2 plumbing (live mode). REFLEX_SURFACE is set by ask() before routing so a
# (text) -> RouteDecision router can still tell Jev which surface it is on;
# LAST_REFLEX carries the live decider's record to the audit writer without
# threading a new argument through every record call site.
REFLEX_SURFACE: contextvars.ContextVar[str] = contextvars.ContextVar('reflex_surface', default='unknown')
LAST_REFLEX: contextvars.ContextVar[object] = contextvars.ContextVar('last_reflex', default=None)

log = logging.getLogger(__name__)
_LIVE_WARNED = False
_LIVE_LOCK = threading.Lock()

# Raw typed questions are part of the SDK contract. Keeping the rubric here also
# lets offline fakes exercise the exact payload without importing a provider.
QUESTIONS = {
    'route': {
        'type': 'choice',
        'instructions': 'Which path must handle this message from the owner to his home assistant Aerys?',
        'criteria': {
            'action': "It needs to TOUCH or READ something outside the conversation: control a device, read live device/sensor state (charge, temperature, on/off, locked), look at an attachment/image/PDF/video link, search the web or current events/weather/prices/shopping availability, read or send email, keep a note or reminder for later ('remember that…', 'remind me'), play or control music, set a timer, or message Kael.",
            'chat': "Pure conversation: opinions, feelings, general knowledge that cannot have changed, reminiscing ('remember when…'), greetings, acknowledgments, jokes, planning talk that touches nothing.",
        },
    },
    'tier': {
        'type': 'score',
        'instructions': 'How much thinking does the reply deserve?',
        'criteria': [
            'fast: greeting, one-word acknowledgment, small talk, trivial',
            'standard: an ordinary question or request',
            'deep: multi-step reasoning, analysis, something emotionally weighty or ambiguous',
        ],
    },
    'unaddressed': {
        'type': 'noul',
        'instructions': 'The message is clearly NOT directed at the assistant at all (overheard speech, a wake word landing mid-sentence, talk to another person). Short acknowledgments and follow-ups ARE addressed.',
    },
}


def error_result(exc: Exception) -> dict:
    return {'error': f'{type(exc).__name__}: {str(exc)[:120]}'}


class ReflexClient:
    def __init__(self, *, client: object, timeout_s: float = .6):
        self.client = client
        self.timeout_s = timeout_s
        # A broken transport must not leave an unbounded pile of abandoned calls.
        self._inflight = threading.BoundedSemaphore(32)

    def __call__(self, text: str, context: dict) -> dict:
        started = time.monotonic()
        result = {'error': 'timeout'}
        completed_at = None

        def run():
            nonlocal result, completed_at
            try:
                response = self.client.system_one(
                    state={'message': text[:2000], 'surface': context.get('surface', 'unknown')},
                    questions=QUESTIONS,
                )
                route = response.answers['route']
                score = float(response.answers['tier'].score)
                result = {
                    'route': str(route.choice),
                    'p_action': float(route.probabilities['action']),
                    'confidence': float(route.confidence),
                    'tier': ('fast', 'standard', 'deep')[max(0, min(2, round(score)))],
                    'tier_score': score,
                    'unaddressed': float(response.answers['unaddressed'].noul),
                    'model': str(response.model),
                    'input_tokens': int(response.usage.input_tokens),
                }
            except Exception as exc:
                result = error_result(exc)
            finally:
                completed_at = time.monotonic()
                self._inflight.release()

        try:
            if self._inflight.acquire(blocking=False):
                try:
                    ctx = contextvars.copy_context()
                    thread = threading.Thread(target=lambda: ctx.run(run), daemon=True)
                    thread.start()
                except Exception:
                    self._inflight.release()
                    raise
                thread.join(max(0, self.timeout_s - (time.monotonic() - started)))
            if completed_at is None or completed_at - started > self.timeout_s:
                result = {'error': 'timeout'}
        except Exception as exc:
            result = error_result(exc)
        return {**result, 'latency_ms': int((time.monotonic() - started) * 1000)}


def reflex_for(settings: Settings) -> Callable[[str, dict], dict | None] | None:
    if settings.reflex_mode == 'off' or not settings.typesafe_api_key:
        return None
    # Lazy import keeps the default-off dev/CI path free of provider setup.
    from typesafe_sdk import RetryPolicy, TypeSafeClient

    client = TypeSafeClient(
        api_key=settings.typesafe_api_key.get_secret_value(),
        model=settings.reflex_model,
        timeout=settings.reflex_timeout_s,
        retry=RetryPolicy(max_retries=0),
    )
    return ReflexClient(client=client, timeout_s=settings.reflex_timeout_s)


def _router_verdict(decision: RouteDecision | None) -> dict | None:
    if decision is None:
        return None
    return {'route': decision.route, 'tier': decision.tier, 'unaddressed': decision.unaddressed,
            'ack': decision.ack}


class LiveReflexRecord:
    """What the live decider did on one turn, collected by the audit writer.

    collect() gives the parallel Haiku router a little more time to finish so
    the row still carries BOTH verdicts on chat turns Jev decided alone — the
    comparison that makes the report meaningful survives the cutover.
    """

    def __init__(self, record: dict, router_thread, router_box: dict, join_s: float = 3.0):
        self.record = record
        self._thread = router_thread
        self._box = router_box
        self._join_s = join_s

    def collect(self) -> dict:
        if self.record.get('router') is None and self._thread is not None:
            self._thread.join(self._join_s)
            self.record['router'] = _router_verdict(self._box.get('decision'))
        return self.record


def live_router_for(
    settings: Settings,
    reflex: Callable[[str, dict], dict | None],
    router: Callable[[str], RouteDecision],
    *,
    registered_routes: tuple[str, ...] = ('chat', 'action'),
) -> Callable[[str], RouteDecision]:
    """Phase 2: Jev decides when it is sure; the Haiku router otherwise.

    Both run in parallel. Jev answers in ~0.2–0.4 s; the router keeps running
    and is waited on only when its output is needed: the ACK on action routes
    (J2: the spoken ack stays generated), or the whole decision when Jev is
    unsure, late, or failed. Chat turns Jev decides alone skip the 0.8–2.0 s
    router wait entirely. Doctrine kept from router.py: uncertainty leans toward
    action, and a wrong 'unaddressed' is worse than a wrong route, so it needs a
    higher bar. Never raises: any failure becomes the router's decision.
    """
    conf_bar = settings.reflex_route_confidence
    action_floor = settings.reflex_action_floor
    unaddressed_floor = settings.reflex_unaddressed_floor

    def decide(text: str) -> RouteDecision:
        started = time.monotonic()
        box: dict = {}
        ctx = contextvars.copy_context()

        def run_router():
            try:
                box['decision'] = router(text)
            except Exception as exc:  # build_router already fails to heuristic; belt and braces
                box['error'] = error_result(exc)

        thread = threading.Thread(target=lambda: ctx.run(run_router), daemon=True)
        thread.start()

        try:
            jev = reflex(text, {'surface': REFLEX_SURFACE.get()})
        except Exception as exc:
            jev = error_result(exc)
        if not isinstance(jev, dict):
            jev = {'error': 'no result'}
        record = {'jev': jev, 'router': None, 'mode': 'live', 'decided_by': 'router'}

        decision = None
        confident = (
            'error' not in jev
            and jev.get('route') in registered_routes
            and float(jev.get('confidence', 0)) >= conf_bar
        )
        if confident:
            route = jev['route']
            if route == 'chat' and float(jev.get('p_action', 0)) >= action_floor:
                route = 'action'
            ack = ''
            if route == 'action':
                # The ack is generated by the router (J2). Its own timeout bounds this.
                thread.join()
                rd = box.get('decision')
                record['router'] = _router_verdict(rd)
                ack = rd.ack if rd is not None and rd.ack else FALLBACK_ACK
            decision = RouteDecision(
                route=route, ack=ack, tier=normalize_tier(jev.get('tier')),
                unaddressed=float(jev.get('unaddressed', 0)) >= unaddressed_floor,
            )
            record['decided_by'] = 'jev'
        if decision is None:
            thread.join()
            rd = box.get('decision')
            if rd is None:
                rd = fallback_decision(text)
                record['router_error'] = box.get('error', {'error': 'no decision'})
            record['router'] = _router_verdict(rd)
            decision = rd
        record['decided'] = {'route': decision.route, 'tier': decision.tier,
                             'unaddressed': decision.unaddressed}
        record['latency_ms'] = int((time.monotonic() - started) * 1000)
        LAST_REFLEX.set(LiveReflexRecord(record, thread, box))
        return decision

    return decide
