"""One-message Jev judgments. Evidence for a future router, never today's driver."""
from __future__ import annotations

import contextvars
import random
import re
from dataclasses import replace
import logging
import threading
import time
from typing import Callable

from aerys_v2.config import Settings
from aerys_v2.router import (FALLBACK_ACK, RouteDecision, fallback_decision, normalize_tier,
                             plausibly_asks_for_action)

# What Jev is shown of a message. Anything longer is judged on a prefix, and a
# prefix verdict may never execute a write (see plain_device_command).
STATE_MESSAGE_CHARS = 2000

# Phase 2 plumbing (live mode). REFLEX_SURFACE is set by ask() before routing so a
# (text) -> RouteDecision router can still tell Jev which surface it is on;
# LAST_REFLEX carries the live decider's record to the audit writer without
# threading a new argument through every record call site.
REFLEX_SURFACE: contextvars.ContextVar[str] = contextvars.ContextVar('reflex_surface', default='unknown')
LAST_REFLEX: contextvars.ContextVar[object] = contextvars.ContextVar('last_reflex', default=None)
# Option B (shadow, 2026-09-20): the thread's recent exchanges, set by ask() so the
# unaddressed question can be asked WITH context. Recorded beside the plain score.
REFLEX_LAST_REPLY: contextvars.ContextVar[str] = contextvars.ContextVar('reflex_last_reply', default='')
REFLEX_RECENT: contextvars.ContextVar[list] = contextvars.ContextVar('reflex_recent', default=[])
REFLEX_SINCE_S: contextvars.ContextVar[float | None] = contextvars.ContextVar('reflex_since_s', default=None)
REFLEX_ROOM: contextvars.ContextVar[dict] = contextvars.ContextVar('reflex_room', default={})  # device / local_time / previous_capture_outcome

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
    # J10 (Chris 2026-09-19, ruled "silent on voice"): the speaker withdraws the
    # request mid-turn. Asked in the same call; consumed only above a high floor.
    'cancelled': {
        'type': 'noul',
        'instructions': "The speaker is WITHDRAWING the request: says cancel, never mind, forget it, scratch that, or breaks off ('actually no', 'nothing', 'ignore that'). A message that asks for something new, or says never mind about one thing and then asks another, is NOT withdrawn.",
    },
}


# Option B (shadow): the same unaddressed judgment, but with the assistant's previous
# reply in the state. A single fragment is ambiguous ("Is the question I asked." scored
# 0.36 alone); against her last line it is not.
# Round 2 (13:20): background about the house + the last two exchanges + seconds since
# she spoke + examples from the real drops. On 124 real voice turns: floor 0.6 → 7/13
# captures caught, 9/111 real turns falsely dropped; 0.7 → 2/13, 3/111. The remaining
# picture (who is speaking, how sure the satellite was about the wake word) is not in
# the text yet — speaker ID and wake-word confidence are the inputs that finish this.
VOICE_BACKGROUND = ("Household: Chris (the owner; speaks to the assistant Aerys in short casual sentences and often answers "
                    "her previous line with a few words) and his wife Megan. Friends/colleagues who come up by name: Joe, "
                    "Ben, Ricky, Marcus. Satellites are microphones in the office (Chris works and takes calls there) and "
                    "the bedroom (TV often on); phone calls and other people happen near the mic; the wake word sometimes "
                    "fires on background speech and captures a fragment nobody said to Aerys. Fragments tend to come in "
                    "runs: after one false capture, the next capture within a minute is often also background speech. "
                    "When Aerys just spoke seconds ago, a short reply is usually to her.")
UNADDRESSED_CTX_QUESTION = {
    'type': 'noul',
    'instructions': ("The message was NOT said to the assistant: television, video or radio speech, someone else in the "
                     "room being spoken to, third-person narration, or a stray fragment that asks and answers nothing. "
                     "Examples of NOT addressed: 'so I told Megan we would probably leave around nine tomorrow' | "
                     "'and then he just left it in the driveway all weekend' | 'He interrupted Filibuster. Let me ask him "
                     "about his trip to Italy' | 'Marcus, it's a, uh, a way to track outages'. Examples of addressed (a "
                     "request, or a short answer/follow-up to the assistant's previous line): 'yes go ahead' | "
                     "'the sunroom' | 'never mind' | 'can you turn off the dressers please?' | 'what time is it' | "
                     "'So I just want to see if you can hear me on the.' | 'okay thanks' | 'sorry, that wasn't meant for you' | "
                     "'I wasn't talking to you' (a correction TO the assistant after she answered background talk is "
                     "addressed to her)."),
}


# Round 6 (2026-09-20 13:19, Chris: "6 of 16 is not acceptable… this is what Jev is designed
# to do"): the voice gate asks WHO the capture is for, on a TRIMMED state. jev-1.13's own
# jaggedness page (#5, context rot) and rounds 4–6 on 124 real turns agree: the 8-exchange
# history HURT (background median 0.49 with it, 0.96 without). Time is handed over as words
# computed in code (jaggedness #3), never as raw seconds.
VOICE_BACKGROUND_SHORT = ("The microphone sometimes triggers on background speech (Chris on a work call, talking to his "
                          "wife Megan, TV or a video) and captures a fragment nobody said to the assistant Aerys.")
ADDRESSEE_QUESTION = {
    'type': 'choice',
    'instructions': 'Who is this message for?',
    'criteria': {
        'request_to_assistant': ('a request, question, command or greeting to the assistant Aerys, including asking '
                                 'her to pass a message to Kael'),
        'reply_to_assistant': ("a short answer, confirmation or follow-up to the assistant's previous line, or "
                               "something that adds to, narrows or corrects her previous action (naming a room or "
                               "more devices, 'the displays too', 'meant the other one')"),
        'correction_to_assistant': 'telling the assistant an earlier message was not meant for her or that she misheard',
        'another_person': 'speech to another person in the room or on a phone/video call',
        'media_speech': 'television, video, radio or music audio picked up by the microphone',
        'work_meeting_talk': ('explaining or discussing work systems, tickets, outages, code or meetings to '
                              'colleagues; asks the assistant nothing'),
        'fragment_nobody': ("a fragment cut from the middle of someone's sentence that neither asks nor tells the "
                            'assistant anything'),
    },
}
ADDRESSEE_BACKGROUND = ('another_person', 'media_speech', 'work_meeting_talk', 'fragment_nobody')


def time_bucket(seconds: float | None) -> str:
    """Elapsed time as words for the model: jev-1.13 reads numbers as text (jaggedness #3)."""
    if seconds is None:
        return 'never'
    if seconds <= 15:
        return 'just now (within 15 seconds)'
    if seconds <= 120:
        return 'a moment ago (within 2 minutes)'
    if seconds <= 600:
        return 'a few minutes ago'
    return 'not recently (over 10 minutes ago)'


def addressee_state(text: str, context: dict) -> dict:
    """The trimmed voice state (round 6): her last line, when she last spoke, the previous
    capture and how it went, how many background captures lately — all as words."""
    state = {'message': text[:STATE_MESSAGE_CHARS], 'background': VOICE_BACKGROUND_SHORT}
    for k in ('device', 'local_time'):
        if context.get(k):
            state[k] = context[k]
    last_reply = (context.get('last_reply') or '')[:300]
    state['assistant_previous_line'] = last_reply or None
    state['assistant_last_spoke'] = time_bucket(context.get('seconds_since_assistant_spoke'))
    prev = context.get('previous_capture')
    if isinstance(prev, dict) and prev.get('text'):
        state['previous_capture'] = {'how_long_ago': time_bucket(prev.get('seconds_ago')),
                                     'text': str(prev['text'])[:200], 'outcome': prev.get('outcome') or 'answered normally'}
    elif context.get('previous_capture_outcome'):
        state['previous_capture'] = {'outcome': context['previous_capture_outcome']}
    n = int(context.get('background_captures_5m') or 0)
    state['background_captures_in_last_5_minutes'] = 'none' if n == 0 else ('one' if n == 1 else 'several')
    return state


def unaddressed_bypass(text: str, context: dict) -> str | None:
    """Identities enforced in code (jaggedness #8): a reply given right after she asked a
    question, or a few words right after any line of hers, is never dropped as background."""
    since = context.get('seconds_since_assistant_spoke')
    if since is None:
        return None
    last_reply = (context.get('last_reply') or '').rstrip()
    words = len(text.split())
    # Round 6 data: an 11-word background fragment landed 5 s after she asked "how about
    # you?" at P(background)=0.99 — a reply to her question is short; a story is not.
    if last_reply.endswith('?') and since <= 20 and words <= 10:
        return 'question'
    if since <= 30 and words <= 6:
        return 'short_followup'
    return None


# Phase 3 (shadow): the content-privacy question, asked of the turn + reply. Phrased
# to match the metered judge's rubric (services.content_privacy / factory judge):
# DEFAULT PUBLIC; private only for the sensitive categories and any secret.
PRIVACY_QUESTION = {
    'type': 'noul',
    'instructions': (
        "The content may be repeated in a shared, public room without harm: it contains NO health or medical "
        "detail, NO financial specifics, NO relationship struggle or personal trauma, NO sexual orientation, "
        "and NO secret or credential (password, passcode, PIN, door/garage/gate/alarm code, wifi password, "
        "API or private key, seed phrase, account/card/routing number, exact home address). Names, jobs, "
        "hobbies, opinions, plans, general facts and ordinary household talk ARE fine to repeat."
    ),
}


def error_result(exc: Exception) -> dict:
    return {'error': f'{type(exc).__name__}: {str(exc)[:120]}'}


# ---- Phase 4: speculative device questions, asked in the SAME call as the route.
# Extra questions cost no extra latency on this model class; the answers are
# recorded on every action turn and only ACTED on when plain_device_command()
# says the request is one plain thing on one allowed target.
DEVICE_ACTIONS = {
    'turn_on': 'turn it on',
    'turn_off': 'turn it off',
    'toggle': 'toggle / flip it',
    'set_brightness': 'set or change how bright or dim it is, or a percentage',
    'set_color': 'set or change its color',
    'other': 'something else, or more than one of these',
}
NONE_TARGET = 'none_of_these'


def device_questions(target_choices: dict[str, str]) -> dict:
    return {
        'is_device_command': {
            'type': 'noul',
            'instructions': ('The message asks to turn a light, switch, plug or fan on or off, '
                             'toggle it, dim or brighten it, or change its color — RIGHT NOW, as a '
                             'command. Not a question about its state, not a schedule or timer, '
                             'not talk about devices.'),
        },
        'is_state_question': {
            'type': 'noul',
            'instructions': ("The message asks what a device's current state is (is it on, how "
                             'bright, what color, is it locked) rather than asking to change it.'),
        },
        'is_compound': {
            'type': 'noul',
            'instructions': ('The message asks for more than one distinct thing — two devices in '
                             'different rooms, an action plus a question, or an action plus '
                             'anything unrelated.'),
        },
        'device_target': {
            'type': 'choice',
            'instructions': 'Which room or device does the message refer to?',
            'criteria': {**target_choices, NONE_TARGET: 'no listed room or device is named'},
        },
        'device_action': {
            'type': 'choice',
            'instructions': 'What does the message ask to do to it?',
            'criteria': dict(DEVICE_ACTIONS),
        },
    }


_PCT_RE = re.compile(r'(\d+)\s*(?:%|percent)')
_SIGNED_PCT_RE = re.compile(r'[+\-]\s*\d+\s*(?:%|percent)')
# Codex review 2026-09-20 #2: "dim by half" / "from 80% to 20%" / "-20%" / "1001%" are NOT
# absolute levels. Any relative marker, more than one level, or a sign defers to the
# specialist, which reads the current level first.
_RELATIVE_RE = re.compile(r'\b(by|more|less|brighter|dimmer|increase|decrease|raise|lower|than)\b')
# Absolute levels only. "dim"/"dimmer"/"brighter" are RELATIVE asks and must fall
# through to the specialist, which reads the current level first.
_PCT_WORDS = (('full', 100), ('max', 100), ('all the way up', 100), ('half', 50),
              ('quarter', 25), ('low', 20), ('minimum', 10))
_COLOR_WORDS = ('warm white', 'soft white', 'cool white', 'daylight', 'red', 'orange', 'yellow',
                'green', 'blue', 'purple', 'violet', 'pink', 'magenta', 'cyan', 'teal', 'white',
                'amber')
_HEX_RE = re.compile(r'#[0-9a-fA-F]{6}\b')


def brightness_from_text(text: str) -> int | None:
    """Numbers stay in code: the decision model does not do arithmetic.

    Returns a level ONLY for one unambiguous absolute ask; anything relative,
    signed, doubled or out of range returns None (the specialist handles it).
    """
    low = text.lower()
    if _RELATIVE_RE.search(low) or _SIGNED_PCT_RE.search(low):
        return None
    pcts = _PCT_RE.findall(low)
    words = [pct for word, pct in _PCT_WORDS if re.search(r'\b' + re.escape(word) + r'\b', low)]
    # Exactly ONE candidate across numbers and words, and every number in range:
    # "20% or half", "1001% or 40%", "half, no quarter" all defer.
    if len(pcts) + len(words) != 1:
        return None
    if pcts:
        n = int(pcts[0])
        return n if 1 <= n <= 100 else None
    return words[0]


def _finite(value) -> float | None:
    """Codex review 2026-09-20 #4/#9: NaN compares false against every gate and a
    zero threshold admits a substituted zero. A malformed score is None, and the
    caller refuses outright instead of comparing it."""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if f != f or f in (float('inf'), float('-inf')) or not 0.0 <= f <= 1.0:
        return None
    return f


def color_from_text(text: str) -> str | None:
    m = _HEX_RE.search(text)
    if m:
        return m.group(0).lower()
    low = text.lower()
    for name in _COLOR_WORDS:  # multi-word names first (tuple order)
        if re.search(r'\b' + re.escape(name) + r'\b', low):
            return name
    return None


def plain_device_command(record: dict, text: str, settings: Settings, canary_entities,
                         aliases: dict | None = None) -> dict | None:
    """One plain thing, on one allowed target, that the model is sure about — else None.

    Sets record['device']['direct'] = {'ok': bool, 'reason': str} so the row shows
    why a turn did or did not go direct. Never raises.
    """
    from aerys_v2.tools.home_control import resolve_targets

    dev = record.setdefault('device', {})

    def no(reason: str) -> None:
        dev['direct'] = {'ok': False, 'reason': reason}
        return None

    try:
        if not settings.reflex_direct:
            return no('disabled')
        if len(text) > STATE_MESSAGE_CHARS:
            # Codex review 2026-09-20 #1: Jev judged only the first 2,000 chars; the
            # part it never saw could say "do not". Never act on a partial reading.
            return no('request longer than the classified excerpt')
        if record.get('decided', {}).get('route') != 'action' or record.get('decided_by') != 'jev':
            return no('not a jev action route')
        target = dev.get('device_target') or {}
        action = dev.get('device_action') or {}
        scores = {name: _finite(value) for name, value in (
            ('is_device_command', dev.get('is_device_command', 0)),
            ('is_state_question', dev.get('is_state_question', 0)),
            ('is_compound', dev.get('is_compound', 0)),
            ('target_confidence', target.get('confidence', 0)),
            ('action_confidence', action.get('confidence', 0)),
        )}
        bad = [name for name, value in scores.items() if value is None]
        if bad:
            return no(f'malformed score: {bad}')
        if scores['is_device_command'] < settings.reflex_direct_command_floor:
            return no('not clearly a device command')
        if scores['is_state_question'] > 0.3:
            return no('reads as a state question')
        if scores['is_compound'] > 0.3:
            return no('compound request')
        if target.get('choice') in (None, NONE_TARGET) or scores['target_confidence'] < settings.reflex_direct_target_confidence:
            return no('target unsure')
        if action.get('choice') in (None, 'other') or scores['action_confidence'] < settings.reflex_direct_target_confidence:
            return no('action unsure')
        op = action['choice']
        command: dict = {'operation': op, 'entity_id': target['choice']}
        if op == 'set_brightness':
            pct = brightness_from_text(text)
            if pct is None:
                return no('brightness needs the specialist (relative or unstated)')
            command['brightness_pct'] = pct
        if op == 'set_color':
            color = color_from_text(text)
            if color is None:
                return no('color not stated')
            command['color'] = color
        targets, problem = resolve_targets(target['choice'], op, frozenset(canary_entities or ()), aliases)
        if problem or not targets:
            return no('target does not resolve')
        # Precedence (Codex review 2026-09-20 #6): (1) the owner ruling J3 — sensitive
        # DOMAINS never go direct, not even by allowlist; (2) the domain gate; (3) the
        # explicit allowlist, which WINS over the token deny list because the owner
        # named the entity; (4) the token deny list, matched on whole id tokens so
        # "door" does not catch switch.outdoor_lights and "gate" not switch.gateway_plug.
        domains = {e.split('.', 1)[0] for e in targets}
        hard = {d.strip().rstrip('.') for d in settings.reflex_direct_deny_domains.split(',') if d.strip()}
        if domains & hard:
            return no(f'sensitive domain: {sorted(domains & hard)}')
        allowed = {d.strip() for d in settings.reflex_direct_domains.split(',') if d.strip()}
        if not domains <= allowed:
            return no(f'domain not direct: {sorted(domains - allowed)}')
        allow = {e.strip() for e in settings.reflex_direct_allow.split(',') if e.strip()}
        if allow:
            if not set(targets) <= allow:
                return no(f'not on the direct allowlist: {sorted(set(targets) - allow)}')
        else:
            deny = {d.strip().lower() for d in settings.reflex_direct_deny.split(',') if d.strip()}
            hit = [e for e in targets if deny & set(re.split(r'[._\-\s]+', e.lower()))]
            if hit:
                return no(f'deny-listed: {hit}')
        dev['direct'] = {'ok': True, 'reason': 'plain command', 'targets': targets}
        return command
    except Exception as exc:  # the direct path is an optimisation; never a crash
        return no(f'error {type(exc).__name__}')


class ReflexClient:
    def __init__(self, *, client: object, timeout_s: float = .6):
        self.client = client
        self.timeout_s = timeout_s
        # A broken transport must not leave an unbounded pile of abandoned calls.
        self._inflight = threading.BoundedSemaphore(32)

    def judge_privacy(self, text: str) -> dict:
        """Phase 3 shadow: p(public) for one piece of content. Never raises; bounded."""
        started = time.monotonic()
        try:
            response = self.client.system_one(
                state={'content': text[:STATE_MESSAGE_CHARS]}, questions={'public': PRIVACY_QUESTION},
            )
            result = {'p_public': float(response.answers['public'].noul), 'model': str(response.model)}
        except Exception as exc:
            result = error_result(exc)
        return {**result, 'latency_ms': int((time.monotonic() - started) * 1000)}

    def __call__(self, text: str, context: dict) -> dict:
        started = time.monotonic()
        # Codex review 2026-09-20 #5: the worker used to share `result` with the caller,
        # so an answer landing between the deadline check and the return became a
        # late, authoritative verdict. Now the worker writes into its own box and the
        # caller reads it ONLY if `done` was set before the deadline.
        box: dict = {}
        done = threading.Event()
        result: dict = {'error': 'timeout'}

        def run():
            try:
                questions = dict(QUESTIONS)
                targets = context.get('device_targets')
                if targets:
                    questions.update(device_questions(targets))
                state = {'message': text[:STATE_MESSAGE_CHARS], 'surface': context.get('surface', 'unknown')}
                exchanges = context.get('recent_exchanges') or []
                last_reply = (context.get('last_reply') or '')[:600]
                source = context.get('unaddressed_source', 'addressee')
                voice_ctx = bool(exchanges or last_reply or context.get('device') or context.get('previous_capture'))
                if voice_ctx and source == 'addressee':
                    # Round 6: the trimmed state, one Choice — see addressee_state().
                    state.update(addressee_state(text, context))
                    questions['addressee'] = ADDRESSEE_QUESTION
                elif (exchanges or last_reply) and source == 'ctx':
                    # Round 3 (2026-09-20 13:30, Chris: "hit it with data"): the richest
                    # picture — up to 8 prior exchanges with their age, which device, the
                    # local time, how the previous capture went, the household names.
                    state['background'] = VOICE_BACKGROUND
                    state['recent_exchanges'] = exchanges[-8:]
                    if last_reply:
                        state['assistant_previous_reply'] = last_reply
                    if context.get('seconds_since_assistant_spoke') is not None:
                        state['seconds_since_assistant_last_spoke'] = context['seconds_since_assistant_spoke']
                    for k in ('device', 'local_time', 'previous_capture_outcome'):
                        if context.get(k):
                            state[k] = context[k]
                    questions['unaddressed_ctx'] = UNADDRESSED_CTX_QUESTION
                response = self.client.system_one(state=state, questions=questions)
                route = response.answers['route']
                score = float(response.answers['tier'].score)
                device = {}
                if targets:
                    a = response.answers
                    device = {
                        'is_device_command': float(a['is_device_command'].noul),
                        'is_state_question': float(a['is_state_question'].noul),
                        'is_compound': float(a['is_compound'].noul),
                        'device_target': {'choice': str(a['device_target'].choice),
                                          'confidence': float(a['device_target'].confidence)},
                        'device_action': {'choice': str(a['device_action'].choice),
                                          'confidence': float(a['device_action'].confidence)},
                    }
                out = {
                    'route': str(route.choice),
                    'p_action': float(route.probabilities['action']),
                    'confidence': float(route.confidence),
                    'tier': ('fast', 'standard', 'deep')[max(0, min(2, round(score)))],
                    'tier_score': score,
                    'unaddressed': float(response.answers['unaddressed'].noul),
                    'cancelled': float(response.answers['cancelled'].noul),
                    'model': str(response.model),
                    'input_tokens': int(response.usage.input_tokens),
                }
                if 'unaddressed_ctx' in questions:
                    out['unaddressed_ctx'] = float(response.answers['unaddressed_ctx'].noul)
                if 'addressee' in questions:
                    ans = response.answers['addressee']
                    probs = ans.probabilities
                    out['addressee'] = {'choice': str(ans.choice), 'confidence': float(ans.confidence),
                                        'background': float(sum(float(probs[k]) for k in ADDRESSEE_BACKGROUND))}
                if device:
                    out['device'] = device
                box['result'] = out
            except Exception as exc:
                box['result'] = error_result(exc)
            finally:
                box['completed_at'] = time.monotonic()
                done.set()
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
                # Codex review 2026-09-20 #8: a descheduled caller waking late must not
                # accept an answer that landed after the deadline (wait(0) still
                # returns True) — the completion time decides, not the wake time.
                if done.wait(max(0, self.timeout_s - (time.monotonic() - started))):
                    if box.get('completed_at', float('inf')) - started <= self.timeout_s:
                        result = dict(box.get('result') or {'error': 'no result'})
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
            'cancelled': decision.cancelled, 'ack': decision.ack}


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
    device_targets: dict[str, str] | None = None,
    canary_entities=None,
    aliases: dict | None = None,
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
    cancel_floor = settings.reflex_cancel_floor
    router_sample = settings.reflex_router_sample
    unaddressed_source = settings.reflex_unaddressed_source
    join_floor = settings.reflex_unaddressed_join_floor
    agree_floor = settings.reflex_unaddressed_agree_floor
    strong_floor = settings.reflex_unaddressed_strong_floor

    def decide(text: str) -> RouteDecision:
        started = time.monotonic()
        box: dict = {}
        ctx = contextvars.copy_context()
        state: dict = {'thread': None, 'started': False}

        def run_router():
            try:
                box['decision'] = router(text)
            except Exception as exc:  # build_router already fails to heuristic; belt and braces
                box['error'] = error_result(exc)

        def start_router() -> None:
            # Started at most once; a failed start is audited (Codex #7/#10) and the
            # fallback path treats it as "no decision".
            if state['started']:
                return
            state['started'] = True
            t = threading.Thread(target=lambda: ctx.run(run_router), daemon=True)
            try:
                t.start()
                state['thread'] = t
            except Exception as exc:
                box['error'] = error_result(exc)

        def join_router() -> None:
            start_router()
            if state['thread'] is not None:
                state['thread'].join()

        surface = REFLEX_SURFACE.get()
        # Router sampling (approved 2026-09-20): Haiku runs in parallel only when its
        # output is likely needed — voice (spoken ack) — or on a sample for the
        # agreement report. Otherwise it starts lazily, only if Jev turns out unsure.
        sampled = surface == 'voice' or router_sample >= 1.0 or random.random() < router_sample
        if sampled:
            start_router()
        context = {'surface': surface}
        if device_targets:
            context['device_targets'] = device_targets
        last_reply = REFLEX_LAST_REPLY.get()
        if last_reply:
            context['last_reply'] = last_reply
        recent = REFLEX_RECENT.get()
        if recent:
            context['recent_exchanges'] = recent
        since = REFLEX_SINCE_S.get()
        if since is not None:
            context['seconds_since_assistant_spoke'] = since
        context.update({k: v for k, v in (REFLEX_ROOM.get() or {}).items() if v})
        context['unaddressed_source'] = unaddressed_source
        try:
            jev = reflex(text, context)
        except Exception as exc:
            jev = error_result(exc)
        if not isinstance(jev, dict):
            jev = {'error': 'no result'}
        record = {'jev': jev, 'router': None, 'mode': 'live', 'decided_by': 'router',
                  'router_sampled': sampled}
        if box.get('error'):
            record['router_error'] = box['error']  # Codex #10: audited even when Jev decides
        if isinstance(jev.get('device'), dict):
            record['device'] = dict(jev['device'])

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
            if route == 'action' and surface == 'voice':
                # The spoken ack is generated by the router (J2) and only VOICE
                # consumes it (service._launch_background_action). Text and lens
                # surfaces never speak it, so they no longer wait for Haiku.
                join_router()
                rd = box.get('decision')
                record['router'] = _router_verdict(rd)
                ack = rd.ack if rd is not None and rd.ack else FALLBACK_ACK
            # Codex review 2026-09-20 #6: keep the router's command-preservation guard —
            # a command-shaped message is never dropped as unaddressed, whatever Jev said.
            jev_u = float(jev.get('unaddressed', 0))
            if unaddressed_source == 'ctx' and jev.get('unaddressed_ctx') is not None:
                jev_u = float(jev['unaddressed_ctx'])  # the fuller picture, when the thread has one
            elif unaddressed_source == 'addressee' and isinstance(jev.get('addressee'), dict):
                jev_u = float(jev['addressee'].get('background', 0))  # round 6: P(for someone/something else)
            record['unaddressed_score'] = jev_u
            unaddressed = jev_u >= unaddressed_floor
            strong = jev_u >= strong_floor
            bypass = unaddressed_bypass(text, context) if surface == 'voice' else None
            if bypass:
                record['unaddressed_bypass'] = bypass
                unaddressed, strong = False, False
            if surface == 'voice' and not unaddressed and not bypass and jev_u >= join_floor:
                # Option A (Chris 2026-09-20 12:55, "A is a yes"): a suspicious fragment on
                # voice waits for Haiku's verdict; both agreeing is what drops it. Two TV
                # fragments reached her at 12:51/12:52 with Jev at 0.36 and 0.76 while
                # Haiku said unaddressed both times.
                join_router()
                rd = box.get('decision')
                record['router'] = _router_verdict(rd)
                if rd is not None and rd.unaddressed and jev_u >= agree_floor:
                    unaddressed = True
                    strong = jev_u >= strong_floor
                    record['ensemble'] = 'router+jev'
            if plausibly_asks_for_action(text):
                unaddressed, strong = False, False
            decision = RouteDecision(
                route=route, ack=ack, tier=normalize_tier(jev.get('tier')),
                unaddressed=unaddressed, unaddressed_strong=unaddressed and strong,
            )
            record['decided_by'] = 'jev'
        if decision is None:
            join_router()
            rd = box.get('decision')
            if rd is None:
                rd = fallback_decision(text)
                record['router_error'] = box.get('error', {'error': 'no decision'})
            record['router'] = _router_verdict(rd)
            decision = rd
            if decision.unaddressed and float(jev.get('unaddressed', 0)) >= strong_floor:
                decision = replace(decision, unaddressed_strong=True)  # both agree
        # J10: a withdrawn request is dropped whoever decided the route. The
        # cancel Noul rides the same call, so it costs nothing extra and does
        # not depend on route confidence; the floor is high because a wrong
        # "cancelled" IGNORES the user exactly like a wrong "unaddressed".
        if 'error' not in jev and float(jev.get('cancelled', 0)) >= cancel_floor:
            decision = replace(decision, cancelled=True)
        record['decided'] = {'route': decision.route, 'tier': decision.tier,
                             'unaddressed': decision.unaddressed,
                             'unaddressed_strong': decision.unaddressed_strong,
                             'cancelled': decision.cancelled}
        # Phase 4: is this one plain device command the code may carry out itself?
        record['command'] = (
            plain_device_command(record, text, settings, canary_entities, aliases)
            if device_targets and decision.route == 'action' else None
        )
        # J8: a plain command on voice speaks no ack (the device is the feedback).
        record['silent_ack'] = bool(record['command']) and bool(settings.reflex_direct_silent_ack)
        record['latency_ms'] = int((time.monotonic() - started) * 1000)
        LAST_REFLEX.set(LiveReflexRecord(record, state['thread'], box))
        return decision

    return decide
