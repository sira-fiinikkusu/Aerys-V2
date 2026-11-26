"""D27/D28: outside text can inform a turn, but cannot authorize a reach-out.

Completed reader receipts in the current input count for all reach-outs.
Prior outside-read AI replies also count, for skill writes only. Sibling calls share
that input: finishing a read never races a sibling send into a different verdict.
The mutable holder records audit and history provenance, never a source for
the current tool call’s guard verdict.
"""
from __future__ import annotations

import asyncio
import json

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

READERS = frozenset({'search_web', 'read_web_page', 'analyze_image', 'read_document',
                     'youtube_summary', 'search_email', 'read_email', 'search_his_email', 'read_his_email',
                     'check_async_task', 'list_async_tasks'})
# search_history is deliberately absent (D98, Chris 11/25 11:30: "i dont need her asking for permission
# for this at all"): her own past conversations never hold her next action for a tap.
# draft_in_his_gmail is deliberately absent: a draft reaches no one; the owner
# reviews and sends it himself. It is also absent from action_executor's allowed set.
# my_pending_requests stays outside REACH_OUT: withdrawing only narrows permission.
REACH_OUT = frozenset({'send_message', 'send_email', 'start_async_task', 'update_async_task', 'message_kael',
                      'sticky_note', 'log_gap', 'control_alarm', 'remember', 'save_contact_email', 'write_file', 'edit_file'})
#: D29 #1 (2026-11-11): her own home jobs' results are trusted for her notes and for asking Kael;
#: sending to people and starting or changing jobs after them still waits for his tap.
HOME_JOB_READERS = frozenset({'check_async_task', 'list_async_tasks'})
TRUSTED_AFTER_HOME_JOBS = frozenset({'remember', 'message_kael'})
OUTSIDE_ACTION = 'outside_action'
REFUSAL = 'Not done: I could not check the outside-text guard, so nothing happened.'
#: G5: why a reach-out was let through, named for the branch of _decide that let it (written on its receipt
#: as response_metadata['basis'], and from there into the turn's tool_calls). A tool that decides for itself
#: (send_message, through D12) adds its own with note_basis; both together read 'clean_turn+confirmed_draft'.
CLEAN_TURN = 'clean_turn'
D29_OWN_JOB_RESULTS = 'own_home_job_results_D29'
D62_CARVE_OUT = 'owner_carve_out_D62'
D64_CARVE_OUT = 'owner_carve_out_D64'


def origin_sources(identity, thread_id, owner, shared_surface_ids=(), members=None):
    from aerys_v2.hands_gate import asker, OTHER, VOICE_UNSURE

    # A household member's own request (D19) is not outside text; an unsure voice still is (D28 #5).
    who = asker(identity, thread_id, owner, members)
    sources = []
    if who in (OTHER, VOICE_UNSURE):
        sources.append('outside speaker')
    speaker = identity.get('speaker') or {}
    if speaker.get('unsure') or speaker.get('band') == 'unsure':
        sources.append('unsure voice')
    if (identity.get('surface') == 'sticky'
            or identity.get('device_id') in shared_surface_ids):
        sources.append('shared surface')
    # D40/D41 (wakeups.py): a wake-up she set herself runs with nobody there. Whatever it read or
    # was told, reaching anyone from it waits for his tap.
    if identity.get('unattended'):
        sources.append('unattended wake-up')
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
    # D79 (Codex review of 45ea17c #1): a correction pass, or the zero-tool bounce inside one, appends a
    # human-shaped instruction after his turn. The caller says where his real turn is, so the relay right
    # before IT keeps tainting every step. Unset (every other turn), or not a human message there: as above.
    start = config.get('turn_human_index')
    if (isinstance(start, int) and not isinstance(start, bool) and 0 <= start < len(messages)
            and isinstance(messages[start], HumanMessage)):
        human = start
    if human is not None and human > 0:
        prior = messages[human - 1]
        if isinstance(prior, AIMessage) and prior.additional_kwargs.get('reach_source') == 'relay':
            sources.add('relay')
    sources.update(m.name for m in messages if isinstance(m, ToolMessage) and m.name in READERS)
    # D79 (#2): a correction rerun starts from his turn again; the earlier pass's outside reads ride along
    # (code-owned reader names), so what they read still can't authorize a reach-out on the rerun.
    sources.update(str(s) for s in (config.get('carried_sources') or ()) if str(s) in READERS)
    return tuple(sorted(sources))


def notice(sources):
    # Sources are code-owned names; outside content never appears in this annotation.
    names = sources if isinstance(sources, (tuple, list, set)) else ('outside source',)
    return 'Asked after reading outside text (' + ', '.join(names) + ')'


def _refusal(request, text):
    return ToolMessage(content=text, name=request.tool_call.get('name', ''),
                       tool_call_id=request.tool_call.get('id', ''))


def _answer(request, verdict, conf):
    """The gate's verdict as (refusal, basis). A refusal that is the approval system failing (infra)
    marks the turn guard_error, so it mines as outside_guard_failed; a wait or a no doesn't."""
    if verdict.go:
        return None, getattr(verdict, 'basis', '') or 'hands_gate'
    if getattr(verdict, 'infra', False):
        holder = conf.get('outside_text')
        if holder is not None:
            holder['guard_error'] = True
    return _refusal(request, verdict.text), None


def _owner_typed_on_discord(conf, gate):
    """D62: this turn is the owner's own typed Discord message (hands_gate.asker decides who)."""
    if gate is None:
        return False
    from aerys_v2.hands_gate import OWNER_TEXT, asker
    identity = conf.get('identity') or {}
    return (str(identity.get('platform') or '').strip().lower() == 'discord'
            and asker(identity, conf.get('thread_id', ''), gate.owner, gate.members) == OWNER_TEXT)


#: D104 (Chris 11/25 19:58: "i hate the idea of having to type to her to confirm it"), redone under D105 (20:22:
#: "that isnt how anyone remembers anything... why is it me approving her to hold onto something becomes something
#: i explicitly said?"): a tap on a remember card means "keep it", never "I said it". Whose words a fact is stays
#: decided by what the person actually said this turn (the remember tool's quote check), and what she read is
#: recorded on the fact. When the speaker wasn't sure it was him, the card asks "was this you?": his tap settles
#: who was talking, nothing more.
REMEMBER_UNSURE_VOICE = "The speaker wasn't sure it was your voice."
REMEMBER_APPROVE_MEANS = ("Approve = OK to keep it. It stays marked by where it came from: a tap never turns it "
                          "into anyone's own words.")
REMEMBER_WAS_IT_YOU_MEANS = ("Approve = yes, that was me, and keep it. Only words you actually said count as yours; "
                             "anything she added stays marked as hers.")
HEARD_SHOWN = 300                         # how much of what the speaker heard the card shows
REMEMBER_FENCE_REFUSAL = ('Not kept: a fact holding ' + '```' + " can't be shown exactly on his card. "
                          'Say it without the triple backticks.')


def canonical_fact(value) -> str:
    """Codex D104: ONE text for the card, the stored args and the write. Invisible format and control characters
    out (newlines and tabs kept), ends trimmed as remember trims them."""
    import unicodedata
    text = str(value or '')
    return ''.join(c for c in text if c in '\n\t' or unicodedata.category(c) not in ('Cf', 'Cc')).strip()


def _literal(text: str) -> str:
    """Shown in a code block, so no markup (a spoiler, a link, bold) can hide or dress up a word he confirms."""
    return '```\n' + text + '\n```'          # the card refuses a fact holding a fence first


def _remember_card(request, gate, conf, sources, args):
    """D104/D105: a remember after outside text, or from a voice that wasn't sure it was him, waits for a tap on a card
    showing the exact fact. The tap means "keep it": whose words it is stays decided by what was actually said this
    turn, and what she read is recorded on the fact. For an unsure voice the card asks "was this you?" and shows what
    the speaker heard; his tap settles who was talking, nothing more."""
    from aerys_v2.state import VOICE_OWNER_ASSUMED_KEY
    from aerys_v2.tools.remember import CURRENT_TURN_TEXT, refusal_for
    if gate is None:
        return _refusal(request, 'Not done: this needs the owner’s Approve button, which is unavailable. '
                        'Nothing was kept.'), None
    tool_args = dict(args)                    # exactly what she asked to keep; the card shows the fact
    if 'fact' in tool_args:
        tool_args['fact'] = canonical_fact(tool_args.get('fact'))
        refused = refusal_for(tool_args['fact'])
        if refused:
            # The tool would refuse it after his tap anyway; a secret-shaped fact must never be put on a card.
            return _refusal(request, refused), None
        if '```' in tool_args['fact']:
            # Codex D104 r2: the card must show exactly what is kept, and a fence can't be shown inside one.
            return _refusal(request, REMEMBER_FENCE_REFUSAL), None
    shown = tool_args.get('fact') or json.dumps(tool_args, ensure_ascii=False)
    identity, thread = conf.get('identity') or {}, conf.get('thread_id', '')
    owner = str(getattr(gate, 'owner', None) or '')
    was_it_you = bool(identity.get(VOICE_OWNER_ASSUMED_KEY)) and bool(owner) and str(identity.get('user_id') or '') == owner
    # Code-captured, never hers: what the person actually said this turn (the quote check runs against it after the
    # tap) and the code-owned names of what she read.
    turn_words = str(CURRENT_TURN_TEXT.get() or '')
    read_first = sorted(str(n) for n in sources) if isinstance(sources, (set, list, tuple)) else []
    why = [notice(sources)] if sources else []
    if was_it_you:
        heard = ' '.join(canonical_fact(turn_words).replace('```', "'''").split())
        heard = heard if len(heard) <= HEARD_SHOWN else heard[:HEARD_SHOWN - 1].rstrip() + '…'
        why.append(REMEMBER_UNSURE_VOICE + (' It heard:\n' + _literal(heard) if heard else ''))
        summary, means = 'Was this you? Keep this in your memory?', REMEMBER_WAS_IT_YOU_MEANS
    else:
        summary, means = 'Keep this in your memory?', REMEMBER_APPROVE_MEANS
    from aerys_v2.hands_gate import _key
    payload = {'tool': 'remember', 'tool_args': tool_args, 'identity': identity,
               '_turn_words': turn_words, '_was_it_you': was_it_you, '_read_first': read_first,
               'what': 'keeping this in your memory'}
    verdict = gate._ask(OUTSIDE_ACTION, 'outside:' + _key(thread, payload), thread, payload,
                        kind='memory', summary=summary,
                        detail='\n'.join(why) + '\n\nShe\'d keep:\n' + _literal(shown) + '\n' + means,
                        what=payload['what'])
    return _answer(request, verdict, conf)


def _decide(request, gate):
    """(refusal, basis): a ToolMessage when the call is refused or waits; else None and, for a reach-out,
    the G5 basis code naming the branch that let it go (None for anything that isn't a reach-out)."""
    sources = sources_for(request)
    conf = request.runtime.config.get('configurable') or {}
    holder = conf.get('outside_text')
    name = request.tool_call['name']
    args = dict(request.tool_call.get('args') or {})
    if name in ('write_file', 'edit_file') and any(
        isinstance(m, AIMessage) and m.additional_kwargs.get('outside_read')
        for m in request.state['messages']
    ):
        sources = (*sources, 'earlier outside text')
    if holder is not None and name in READERS:
        holder['outside_read'] = True
    if holder is not None and (sources or name in READERS):
        holder['untrusted_read'] = True
    reaches = name in REACH_OUT or (name == 'kael_tickets' and
                                   args.get('action', '').strip().lower() in ('update', 'cancel'))
    # The message tool reads this same snapshot through ToolRuntime and calls D12 itself.
    # Its button shows the resolved identity and signed words, not an outside_action wrapper.
    if name == 'send_message':
        return None, None
    if not reaches:
        return None, None
    from aerys_v2.state import VOICE_OWNER_ASSUMED_KEY
    if name == 'remember' and (conf.get('identity') or {}).get(VOICE_OWNER_ASSUMED_KEY):
        # D104: a voice that wasn't sure it was him asks for his tap instead of telling him to type it.
        return _remember_card(request, gate, conf, sources, args)
    if not sources:
        return None, CLEAN_TURN
    if name in ('write_file', 'edit_file'):
        # 11/19 (Chris tapped Approve on a home-skill edit that could never land, and nobody told him): a
        # write to anything but her own /skills/ shelf asks nobody and runs nothing. It's refused HERE, in
        # her turn, where she can say so; never handed on (Codex 11/19: passing it to the handler would let
        # any other write_file/edit_file implementation write without his tap).
        from aerys_v2.skills import BODY_SKILLS, HOME_SKILL_REFUSAL, writable_skill_path
        path = str(args.get('file_path') or '')
        if not writable_skill_path(path):
            return _refusal(request, HOME_SKILL_REFUSAL if path.startswith(BODY_SKILLS) else
                            'Not saved: I can only write my own skills, under /skills/<name>/. Nothing changed.'), None
    if name in TRUSTED_AFTER_HOME_JOBS and set(sources) <= HOME_JOB_READERS:
        return None, D29_OWN_JOB_RESULTS
    if name == 'remember':
        return _remember_card(request, gate, conf, sources, args)
    # D62 (Chris 11/18 10:20: "she shouldnt be hitting privacy gates when i message her on discord"):
    # after reading outside text, her note to Kael goes straight when HE asked in his own typed words
    # on Discord, as D58 lets her Discord message to him go. Kael reads her notes as observations,
    # never as instructions. Only her own reads taint it here; another asker, a voice, a shared
    # surface, a relayed message, a room or an unattended run still waits for his tap.
    if name == 'message_kael' and set(sources) <= READERS and _owner_typed_on_discord(conf, gate):
        return None, D62_CARVE_OUT
    # D64 (Chris 11/18 11:24, "a"): the same for starting a home job. 11:22 she listed her home jobs
    # (their reports carry web text), and starting the MapTap he asked for raised a card he "shouldnt
    # really need at home". The D10 gate, next in line, still decides who may start one. Adjusting a
    # running job (update_async_task), another asker, a voice, Telegram, a relay or an unattended run
    # still waits for his tap.
    if name == 'start_async_task' and set(sources) <= READERS and _owner_typed_on_discord(conf, gate):
        return None, D64_CARVE_OUT
    if name == 'send_email':
        # D28 (11/11, the follow-through): after outside text an email waits for his tap. The button
        # shows the exact mail; his tap is the confirmation, and the watcher sends exactly that, once
        # (action_executor). Her `confirmed` flag never rides along: the tap decides.
        if gate is None:
            return _refusal(request, 'Not sent: after reading outside text, email needs the owner’s '
                            'Approve button, which is unavailable. Nothing was sent.'), None
        # Shown exactly as send_email will send it (it collapses the subject and trims the body;
        # Codex review 11/11): what he approves is what goes.
        mail = {'to': str(args.get('to') or '').strip(), 'subject': ' '.join(str(args.get('subject') or '').split()),
                'body': str(args.get('body') or '').strip()}
        if not mail['to'] or not mail['body']:
            return _refusal(request, 'Not sent: I need who it is to and what it says before asking the owner.'), None
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
                        'Approve button, which is unavailable. Nothing happened.'), None
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


def _checked(request, gate):
    """_decide, failing closed: (refusal, basis). Its own errors never escape."""
    try:
        return _decide(request, gate)
    except Exception:
        try:
            holder = request.runtime.config.get('configurable', {}).get('outside_text')
            if holder is not None:
                holder['untrusted_read'] = True
                # Telemetry only (the refusal above is unchanged): the turn record says the guard
                # itself failed, its own gap, apart from untrusted_read (which every outside read carries).
                holder['guard_error'] = True
        except Exception:
            pass
        return _refusal(request, REFUSAL), None


def note_basis(config, tool_call_id, code):
    """G5: a tool that decides for itself (send_message through D12) says which branch let it go. Kept on
    the turn's holder by call id; the guard writes it onto that call's receipt. Never raises."""
    try:
        holder = ((config or {}).get('configurable') or {}).get('outside_text')
        if holder is not None and tool_call_id and code:
            holder.setdefault('basis', {})[str(tool_call_id)] = str(code)
    except Exception:
        pass


def _with_basis(message, code):
    return message.model_copy(update={'response_metadata': {**(message.response_metadata or {}), 'basis': code}})


def _after(request, result, basis):
    """What the tool returned, with its G5 basis on this call's receipt (a ToolMessage, or the ToolMessage
    for this call inside a Command). Anything else passes through unchanged; so does a result on error."""
    try:
        call_id = request.tool_call.get('id')
        holder = ((request.runtime.config or {}).get('configurable') or {}).get('outside_text') or {}
        own = (holder.get('basis') or {}).get(str(call_id)) if isinstance(holder, dict) else None
        code = '+'.join(b for b in (basis, own) if b)
        if not code:
            return result
        if isinstance(result, ToolMessage) and result.tool_call_id == call_id:
            return _with_basis(result, code)
        from langgraph.types import Command
        update = result.update if isinstance(result, Command) else None
        if isinstance(update, dict) and isinstance(update.get('messages'), list):
            import dataclasses
            messages = [_with_basis(m, code) if isinstance(m, ToolMessage) and m.tool_call_id == call_id else m
                        for m in update['messages']]
            return dataclasses.replace(result, update={**update, 'messages': messages})
    except Exception:
        pass
    return result


def outside_guard(request, handler, *, gate=None):
    """Shared guard for ToolNode and AgentMiddleware. Its own errors never escape."""
    refused, basis = _checked(request, gate)
    if refused is not None:
        return refused
    return _after(request, handler(request), basis)


def outside_middleware(gate=None):
    from langchain.agents.middleware import AgentMiddleware

    class OutsideTextGuard(AgentMiddleware):
        def wrap_tool_call(self, request, handler):
            return outside_guard(request, handler, gate=gate)

        async def awrap_tool_call(self, request, handler):
            # Same decision function, without blocking the event loop on approval I/O.
            refused, basis = await asyncio.to_thread(_checked, request, gate)
            if refused is not None:
                return refused
            return _after(request, await handler(request), basis)

    return OutsideTextGuard()


def action_executor(tools, *, skill_middleware=None):
    """The watcher invokes only the exact registered tool and args shown on his button."""
    allowed = {t.name: t for t in tools if t.name in REACH_OUT - {
        'send_message', 'start_async_task', 'update_async_task'} or t.name == 'kael_tickets'}

    def execute(row):
        if skill_middleware is not None and row.args.get('tool') in ('write_file', 'edit_file'):
            from aerys_v2.skills import approved_write
            result = approved_write(skill_middleware, row)
            prefixes = {'write_file': ('Updated file ',), 'edit_file': ('Successfully replaced ',)}
            return result.content.startswith(prefixes[row.args['tool']]), result.content
        tool = allowed.get(row.args.get('tool'))
        if tool is None:
            return False, 'Not done: the approved tool is unavailable.'
        identity = dict(row.args.get('identity') or {})
        from aerys_v2.tools.remember import CURRENT_TURN_TEXT, KEPT_AFTER_READING
        # His tap on the button that showed this exact address is his word for it (contacts). A remember card is
        # different (D105): his tap means "keep it", never "I said it". The fact's words are checked against what
        # was actually said in that turn (_turn_words, code-captured at the card); "was this you?" (an unsure
        # voice) only settles who was talking. Any other approved write carries no owner words.
        approved_words = ''
        if tool.name == 'save_contact_email':
            approved_words = str(row.args['tool_args'].get('email') or '')
        read_first = []
        if tool.name == 'remember':
            if row.args.get('_was_it_you') is True:
                from aerys_v2.state import VOICE_OWNER_ASSUMED_KEY
                identity.pop(VOICE_OWNER_ASSUMED_KEY, None)
            approved_words = str(row.args.get('_turn_words') or '')
            read_first = [str(n) for n in (row.args.get('_read_first') or ())]
        config = {'configurable': {'thread_id': row.thread_id, 'identity': identity}}
        if read_first:
            config['configurable'][KEPT_AFTER_READING] = read_first
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
