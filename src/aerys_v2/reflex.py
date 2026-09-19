"""One-message Jev judgments. Evidence for a future router, never today's driver."""
from __future__ import annotations

import contextvars
import logging
import threading
import time
from typing import Callable

from aerys_v2.config import Settings

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
    global _LIVE_WARNED
    if settings.reflex_mode == 'live':
        with _LIVE_LOCK:
            if not _LIVE_WARNED:
                log.warning('REFLEX_MODE=live is reserved; running shadow only')
                _LIVE_WARNED = True
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
