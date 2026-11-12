"""D27/D28: outside text can inform a turn, but cannot authorize a reach-out.

Only completed reader receipts in the current model input count. Sibling calls share
that input: finishing a read never races a sibling send into a different verdict.
The mutable config holder is audit-only, never a source of tool-read taint.
"""
from __future__ import annotations

import asyncio
import json

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

READERS = frozenset({'search_web', 'read_web_page', 'analyze_image', 'read_document',
                     'youtube_summary', 'search_email', 'read_email', 'search_his_email', 'read_his_email',
                     'check_async_task', 'list_async_tasks'})
# draft_in_his_gmail is deliberately absent: a draft reaches no one; the owner
# reviews and sends it himself. It is also absent from action_executor's allowed set.
REACH_OUT = frozenset({'send_message', 'send_email', 'start_async_task', 'update_async_task', 'message_kael',
                      'sticky_note', 'log_gap', 'control_alarm', 'remember', 'save_contact_email'})
#: D29 #1 (2026-11-11): her own home jobs' results are trusted for her notes and for asking Kael;
#: sending to people and starting or changing jobs after them still waits for his tap.
HOME_JOB_READERS = frozenset({'check_async_task', 'list_async_tasks'})
TRUSTED_AFTER_HOME_JOBS = frozenset({'remember', 'message_kael'})
OUTSIDE_ACTION = 'outside_action'
REFUSAL = 'Not done: I could not check the outside-text guard, so nothing happened.'


def origin_sources(identity, thread_id, owner, shared_surface_ids=(), members=None):
    from aerys_v2.hands_gate import asker, OTHER, VOICE_UNSURE
    from aerys_v2.state import is_lens_surface

    # A household member's own request (D19) is not outside text; an unsure voice still is (D28 #5).
    who = asker(identity, thread_id, owner, members)
    sources = []
    if who in (OTHER, VOICE_UNSURE):
        sources.append('outside speaker')
    speaker = identity.get('speaker') or {}
    if speaker.get('unsure') or speaker.get('band') == 'unsure':
        sources.append('unsure voice')
    if (is_lens_surface(identity) or identity.get('surface') in ('sticky', 'lens')
            or identity.get('device_id') in shared_surface_ids):
        sources.append('shared surface')
    return sources


def sources_for(request):
    config = request.runtime.config.get('configurable') or {}
    holder = config.get('outside_text') or {}
    sources = set(holder.get('origin', ()))
    room = config.get('room_sink') or {}
    if room.get('outside'):
        sources.add('room')
    messages = request.state['messages']
    if not isinstance(messages, (list, tuple)):
        raise ValueError('unreadable messages')
    # Keep the immediately replied-to relay tainted through every tool/model step.
    human = next((i for i in range(len(messages) - 1, -1, -1)
                  if isinstance(messages[i], HumanMessage)), None)
    if human is not None and human > 0:
        prior = messages[human - 1]
        if isinstance(prior, AIMessage) and prior.additional_kwargs.get('reach_source') == 'relay':
            sources.add('relay')
    sources.update(m.name for m in messages if isinstance(m, ToolMessage) and m.name in READERS)
    return tuple(sorted(sources))


def notice(sources):
    # Sources are code-owned names; outside content never appears in this annotation.
    names = sources if isinstance(sources, (tuple, list, set)) else ('outside source',)
    return 'Asked after reading outside text (' + ', '.join(names) + ')'


def _refusal(request, text):
    return ToolMessage(content=text, name=request.tool_call.get('name', ''),
                       tool_call_id=request.tool_call.get('id', ''))


def _answer(request, verdict, conf):
    """The gate's verdict as the tool result. A refusal that is the approval system failing (infra)
    marks the turn guard_error, so it mines as outside_guard_failed; a wait or a no doesn't."""
    if verdict.go:
        return None
    if getattr(verdict, 'infra', False):
        holder = conf.get('outside_text')
        if holder is not None:
            holder['guard_error'] = True
    return _refusal(request, verdict.text)


def _decide(request, gate):
    sources = sources_for(request)
    conf = request.runtime.config.get('configurable') or {}
    holder = conf.get('outside_text')
    name = request.tool_call['name']
    args = dict(request.tool_call.get('args') or {})
    if holder is not None and (sources or name in READERS):
        holder['untrusted_read'] = True
    reaches = name in REACH_OUT or (name == 'kael_tickets' and
                                   args.get('action', '').strip().lower() in ('update', 'cancel'))
    # The message tool reads this same snapshot through ToolRuntime and calls D12 itself.
    # Its button shows the resolved identity and signed words, not an outside_action wrapper.
    if name == 'send_message':
        return None
    if not sources or not reaches:
        return None
    if name in TRUSTED_AFTER_HOME_JOBS and set(sources) <= HOME_JOB_READERS:
        return None
    if name == 'send_email':
        # D28 (11/11, the follow-through): after outside text an email waits for his tap. The button
        # shows the exact mail; his tap is the confirmation, and the watcher sends exactly that, once
        # (action_executor). Her `confirmed` flag never rides along: the tap decides.
        if gate is None:
            return _refusal(request, 'Not sent: after reading outside text, email needs the owner’s '
                            'Approve button, which is unavailable. Nothing was sent.')
        # Shown exactly as send_email will send it (it collapses the subject and trims the body;
        # Codex review 11/11): what he approves is what goes.
        mail = {'to': str(args.get('to') or '').strip(), 'subject': ' '.join(str(args.get('subject') or '').split()),
                'body': str(args.get('body') or '').strip()}
        if not mail['to'] or not mail['body']:
            return _refusal(request, 'Not sent: I need who it is to and what it says before asking the owner.')
        identity, thread = conf.get('identity') or {}, conf.get('thread_id', '')
        from aerys_v2.hands_gate import _key
        payload = {'tool': name, 'tool_args': mail, 'identity': identity,
                   'what': f'your email to {mail["to"]} after reading outside text'}
        verdict = gate._ask(OUTSIDE_ACTION, 'outside:' + _key(thread, payload), thread, payload,
                            kind='email', summary=f'Send email to {mail["to"]}',
                            detail=(f'{notice(sources)}\n\nTo: {mail["to"]}\nSubject: {mail["subject"]}'
                                    f'\n\n{mail["body"]}'),
                            what=payload['what'])
        return _answer(request, verdict, conf)
    if gate is None:
        return _refusal(request, 'Not done: after reading outside text this needs the owner’s '
                        'Approve button, which is unavailable. Nothing happened.')
    identity, thread = conf.get('identity') or {}, conf.get('thread_id', '')
    if name in ('start_async_task', 'update_async_task'):
        verdict = gate.home_job(identity, thread, name, args, request.state, tainted=sources)
    else:
        from aerys_v2.hands_gate import _key
        payload = {'tool': name, 'tool_args': args, 'identity': identity,
                   'what': f'{name} after reading outside text'}
        verdict = gate._ask(OUTSIDE_ACTION, 'outside:' + _key(thread, payload), thread, payload,
                            kind='outside text', summary=f'Approve {name}',
                            detail=notice(sources) + '\n\n' + json.dumps(args, ensure_ascii=False),
                            what=payload['what'])
    return _answer(request, verdict, conf)


def outside_guard(request, handler, *, gate=None):
    """Shared guard for ToolNode and AgentMiddleware. Its own errors never escape."""
    try:
        refused = _decide(request, gate)
    except Exception:
        try:
            holder = request.runtime.config.get('configurable', {}).get('outside_text')
            if holder is not None:
                holder['untrusted_read'] = True
                # Telemetry only (the refusal above is unchanged): the turn record says the guard
                # itself failed, so it mines as a gap even though untrusted_read alone doesn't.
                holder['guard_error'] = True
        except Exception:
            pass
        return _refusal(request, REFUSAL)
    return handler(request) if refused is None else refused


def outside_middleware(gate=None):
    from langchain.agents.middleware import AgentMiddleware

    class OutsideTextGuard(AgentMiddleware):
        def wrap_tool_call(self, request, handler):
            return outside_guard(request, handler, gate=gate)

        async def awrap_tool_call(self, request, handler):
            # Same decision function, without blocking the event loop on approval I/O.
            result = await asyncio.to_thread(outside_guard, request, lambda r: None, gate=gate)
            return await handler(request) if result is None else result

    return OutsideTextGuard()


def action_executor(tools):
    """The watcher invokes only the exact registered tool and args shown on his button."""
    allowed = {t.name: t for t in tools if t.name in REACH_OUT - {
        'send_message', 'start_async_task', 'update_async_task'} or t.name == 'kael_tickets'}

    def execute(row):
        tool = allowed.get(row.args.get('tool'))
        if tool is None:
            return False, 'Not done: the approved tool is unavailable.'
        config = {'configurable': {'thread_id': row.thread_id,
                                   'identity': row.args.get('identity') or {}}}
        # Approval authorizes the write; it does not promote outside words to owner facts.
        from aerys_v2.tools.remember import CURRENT_TURN_TEXT
        # His tap on the button that showed this exact address is his word for it (contacts); for
        # everything else the approved write carries no owner words (remember stays her own inference).
        approved_words = (str(row.args['tool_args'].get('email') or '')
                          if tool.name == 'save_contact_email' else '')
        token = CURRENT_TURN_TEXT.set(approved_words)
        tool_args = dict(row.args['tool_args'])
        if tool.name == 'send_email':
            # His tap on the button that showed this exact mail is the confirmation.
            tool_args = {k: str(tool_args.get(k) or '') for k in ('to', 'subject', 'body')}
            tool_args['confirmed'] = True
        try:
            result = str(tool.invoke(tool_args, config=config))
        finally:
            CURRENT_TURN_TEXT.reset(token)
        prefixes = {
            'remember': ('Kept:', 'Already kept:'),
            'sticky_note': ('Note is up on the sticky displays',),
            'log_gap': ('Logged as gap #',),
            'control_alarm': ('Alarm ',),
            'message_kael': ('Delivered', 'NOT DELIVERED YET:'),
            'kael_tickets': ('Added to ', 'Canceled '),
            'save_contact_email': ('Saved:',),
            'send_email': ('Sent:',),
        }
        return result.startswith(prefixes[tool.name]), result

    return execute


def prepare_origin(config, owner, shared_surface_ids=None, members=None):
    """Called before the model; service owns the fresh holder for each incoming turn."""
    if owner is None:  # standalone builders without deployment identity configuration
        return
    conf = config.setdefault('configurable', {})
    holder = conf.setdefault('outside_text', {})
    holder['origin'] = origin_sources(conf.get('identity') or {}, conf.get('thread_id'),
                                      owner, shared_surface_ids or (), members)
