"""Owner-authorized messages from Aerys's own bot; D12 identity approval and D28 guard."""
from __future__ import annotations

import hashlib
import logging
import threading
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from zoneinfo import ZoneInfo

from langchain_core.tools import tool
from langgraph.prebuilt import ToolRuntime

from aerys_v2.hands_gate import MESSAGE
from aerys_v2.outside_guard import sources_for

log = logging.getLogger(__name__)
WINDOW = timedelta(minutes=10)


def signed(text):
    text = str(text or '').rstrip()
    return text if text.endswith('\n-Aerys') else text + '\n-Aerys'


class MessageSends:
    """Test ledger. Reserve before sending; an interrupted attempt remains pending."""
    def __init__(self):
        self.rows = []
        self.lock = threading.Lock()

    def recent(self, limit):
        with self.lock:
            return [dict(r) for r in sorted(self.rows, key=lambda r: r['time'], reverse=True)[:limit]]

    def reserve(self, person, platform, digest):
        with self.lock:
            now = datetime.now(timezone.utc)
            if any(r['person'] == person and r['platform'] == platform and r['hash'] == digest
                   and (r['status'] == 'pending' or r['time'] >= now - WINDOW) for r in self.rows):
                return None
            self.rows.append(dict(person=person, platform=platform, hash=digest, time=now, status='pending'))
            return len(self.rows)

    def finish(self, attempt, ok):
        with self.lock:
            self.rows[attempt - 1]['status'] = 'ok' if ok else 'failed'


class PostgresMessageSends:
    """029: committed reservations serialize workers, survive crashes, and fail closed.

    Pending attempts block indefinitely: delivery may have happened before a crash. Completed
    attempts (including failures with uncertain network outcomes) block for ten minutes.
    No message text or destination address is stored in this ledger.
    """
    def __init__(self, database_url):
        self.database_url = database_url

    def _connect(self):
        import psycopg
        return psycopg.connect(self.database_url, connect_timeout=5, options='-c statement_timeout=5000')

    def recent(self, limit):
        with self._connect() as c:
            rows = c.execute("SELECT person_id, platform, words_hash, attempted_at, status "
                             "FROM v2_message_sends ORDER BY attempted_at DESC, id DESC LIMIT %s",
                             (limit,)).fetchall()
        return [dict(person=str(p), platform=platform, hash=digest, time=when, status=status)
                for p, platform, digest, when, status in rows]

    def reserve(self, person, platform, digest):
        with self._connect() as c:
            c.execute('SELECT pg_advisory_xact_lock(hashtextextended(%s, 29))',
                      (f'{person}:{platform}:{digest}',))
            existing = c.execute(
                "SELECT id FROM v2_message_sends WHERE person_id = %s AND platform = %s AND words_hash = %s "
                "AND (status = 'pending' OR attempted_at >= now() - interval '10 minutes') LIMIT 1",
                (person, platform, digest)).fetchone()
            if existing:
                return None
            return c.execute('INSERT INTO v2_message_sends (person_id, platform, words_hash) '
                             'VALUES (%s, %s, %s) RETURNING id', (person, platform, digest)).fetchone()[0]

    def finish(self, attempt, ok):
        with self._connect() as c:
            if c.execute("UPDATE v2_message_sends SET status = %s WHERE id = %s AND status = 'pending' "
                         'RETURNING id', ('ok' if ok else 'failed', attempt)).fetchone() is None:
                raise RuntimeError('message reservation unavailable')


class MessageSender:
    def __init__(self, gate, people, ledger, deliver, names=None):
        self.gate, self.people, self.ledger, self.deliver = gate, set(people), ledger, deliver
        self.names = names or (lambda ids: {})
        #: D42: her wake-ups' store, so a wake turn can send the message he asked for (factory sets it).
        self.wake_store = None

    def pin_recipient(self, to, text):
        """D42: who a wake-up's message goes to (resolved now, one person on her list) and its exact
        words, kept when he asks for it. A dict, or why not."""
        words = str(text or '').strip()
        if not words:
            return 'NOT DONE: I need the words to send.'
        if len(signed(words)) > 1990:
            return 'NOT DONE: the signed message is too long for Discord; shorten it.'
        try:
            matches = self.gate.people('discord', str(to).strip().lstrip('@'))
        except Exception:
            return "NOT DONE: I can't check who that is right now; nothing was set."
        destinations = {(str(p['person_id']), str(p.get('address') or '').strip()): p
                        for p in matches or [] if p.get('person_id')}
        if len(destinations) != 1:
            return f'NOT DONE: "{to}" isn\'t exactly one person I can message on Discord; ask him who he means.'
        ((pid, address), p), = destinations.items()
        if not address or (pid not in self.people and pid != str(self.gate.owner)):
            return "NOT DONE: that person isn't on the list of people I can message."
        return {'person_id': pid, 'address': address, 'name': str(p.get('name') or to),
                'handle': str(p.get('handle') or address), 'text': words}

    def _wake_pin(self, thread, destinations, text):
        """D42: the message he asked for when this wake-up was set: the same person at the same address,
        the same words, once per run. Spent before it goes, so a crash or a twin call can't send it twice.
        Anything unreadable is no pin (the button stays)."""
        from aerys_v2.wakeups import own_wake_id

        wid = own_wake_id({'thread_id': thread})
        if self.wake_store is None or wid is None or len(destinations) != 1:
            return None
        try:
            w = self.wake_store.get(wid)
            pin = (w.pinned_send if w is not None and w.status == 'active' else None) or {}
            (pid, address), = destinations
            if (not pin or (pid, address) != (str(pin.get('person_id') or ''), str(pin.get('address') or ''))
                    or signed(str(text or '').strip()) != signed(str(pin.get('text') or '').strip())):
                return None
            if not self.wake_store.spend_pin(wid, w.runs + 1):
                return None
            return str(pin['text'])
        except Exception:
            return None

    def messages_sent(self, limit, identity, thread):
        if str(identity.get('user_id') or '') != str(self.gate.owner) or str(thread).startswith('kael:'):
            return 'Refused: only the owner can read my send record.'
        try:
            limit = max(1, min(10, int(limit)))
            gates = self.gate.store.recent_messages(limit)
            deliveries = self.ledger.recent(limit)
            rows = [dict(r, verdict={'ok': 'sent', 'failed': "didn't go through",
                                    'pending': 'outcome unknown'}[r['status']]) for r in deliveries]
            used = set()
            for g in gates:
                status = g.status
                if status in ('waiting', 'dropped'):
                    status = self.gate.approvals.status(g.approval_id)
                    if status == 'unknown':
                        raise RuntimeError('approval unavailable')
                person, platform = str(g.args.get('person_id') or ''), g.args.get('platform')
                digest = hashlib.sha256(str(g.args.get('text') or '').encode('utf-8')).hexdigest()
                # No shared attempt id in legacy records: coalesce only approved/executed
                # requests with the same person/platform/hash, never an older delivery.
                matches = [i for i, r in enumerate(deliveries) if i not in used
                           and (r['person'], r['platform'], r['hash']) == (person, platform, digest)
                           and r['time'] >= g.created_at
                           and (status != 'failed' or r['status'] != 'ok')
                           and (status != 'done' or r['status'] == 'ok')]
                if status in ('done', 'running', 'failed', 'approved') and matches:
                    used.add(min(matches, key=lambda i: deliveries[i]['time']))
                    continue
                verdict = {'pending': "waiting on Chris's tap", 'declined': 'Chris declined it',
                           'expired': 'the button expired', 'failed': "didn't go through"}.get(
                               status, 'outcome unknown')
                rows.append(dict(person=person, platform=platform, time=g.created_at, verdict=verdict))
            rows = sorted(rows, key=lambda r: r['time'], reverse=True)[:limit]
            try:
                names = self.names({r['person'] for r in rows})
            except Exception:
                names = {}
            lines = []
            for r in rows:
                name = ' '.join(str(names.get(r['person']) or 'someone').split())
                platform = 'Discord' if r['platform'] == 'discord' else 'unknown platform'
                et = r['time'].astimezone(ZoneInfo('America/New_York'))
                when = f"{et:%a} {et.strftime('%I').lstrip('0') or '12'}:{et:%M %p}"   # portable: no %-I
                lines.append(f"{name} · {platform} · {when} · {r['verdict']}")
            return '\n'.join(lines) or 'No message attempts in my send record.'
        except Exception:
            return "I couldn't read my send record just now"


    def _deliver(self, person, platform, address, text):
        if platform != 'discord' or not address or not text.endswith('\n-Aerys') or len(text) > 1990:
            return False, 'the approved destination or signed words are invalid'
        if person not in self.people and person != str(self.gate.owner):
            return False, 'that person is no longer on the message list'
        digest = hashlib.sha256(text.encode('utf-8')).hexdigest()
        try:
            attempt = self.ledger.reserve(person, platform, digest)
        except Exception:
            return False, 'the message record is unavailable; nothing was sent'
        if attempt is None:
            return False, 'these words already have a recent or unresolved send attempt; nothing was sent again'
        try:
            self.deliver(address, text)
        except Exception:
            try:
                self.ledger.finish(attempt, False)
            except Exception:
                return False, 'delivery failed or is uncertain, and its record could not be finished; do not retry'
            return False, 'Discord delivery failed or is uncertain; do not retry now'
        try:
            self.ledger.finish(attempt, True)
        except Exception:
            return False, 'Discord accepted the message but its record could not be finished; do not resend'
        return True, 'sent'

    def execute(self, row):
        args = row.args
        return self._deliver(str(args.get('person_id') or ''), args.get('platform'),
                             args.get('address'), str(args.get('text') or ''))

    def send(self, to, text, platform, identity, thread, sources):
        if str(identity.get('user_id') or '') != str(self.gate.owner) or str(thread).startswith('kael:'):
            return 'Not sent: only the owner can ask me to message someone.'
        if str(platform).strip().lower() != 'discord':
            return 'Not sent: this tool supports Discord only. Use send_email for email; Telegram is parked.'
        if not str(text or '').strip():
            return 'Not sent: I need the words to send.'
        words = signed(text)
        if len(words) > 1990:
            return 'Not sent: the signed message is too long for Discord; shorten it first.'
        try:
            matches = self.gate.people('discord', str(to).strip().lstrip('@'))
            destinations = {(str(p['person_id']), str(p.get('address') or '').strip()): p
                            for p in matches or [] if p.get('person_id')}
            if len(destinations) == 1:
                (pid, address), = destinations
                if pid not in self.people and pid != str(self.gate.owner):
                    return "Not sent: that person isn't on the list of people I can message; ask the owner."
            # D42: a wake turn's only outside source is being unattended; the message he asked for when
            # it was set is his word already, so for exactly that send the wake-up isn't outside text.
            if tuple(sources) == ('unattended wake-up',) and identity.get('unattended'):
                pinned = self._wake_pin(thread, destinations, text)
                if pinned is not None:
                    sources, words = (), signed(pinned)
            # Use one lookup snapshot for both approval and delivery: a name cannot resolve differently
            # between the identity decision and sending. Ambiguous unlisted matches remain visible.
            verdict = self.gate.message(thread, 'discord', to, words, identity, tainted=sources,
                                        resolved=list(destinations.values()))
            if not verdict.go:
                return verdict.text
            if len(destinations) != 1 or not address:
                return 'Not sent: I need an unambiguous Discord address. Ask the owner.'
            ok, reason = self._deliver(pid, 'discord', address, words)
            return 'Sent: the signed message went through Discord.' if ok else f'Not sent: {reason}.'
        except Exception:
            return 'Not sent: I could not check the person or approval; nothing was sent.'


def build_send_message_tool(sender):
    @tool
    def send_message(to: str, text: str, runtime: ToolRuntime, platform: str = 'discord') -> str:
        """Message a listed person as yourself on Discord, signed -Aerys.

        First contact waits for the owner's identity-confirmation button. After outside text,
        even a confirmed contact needs approval. Email uses send_email; Telegram is parked.
        """
        try:
            conf = runtime.config.get('configurable') or {}
            sources = sources_for(SimpleNamespace(runtime=runtime, state=runtime.state))
            return sender.send(to, text, platform, conf.get('identity') or {}, conf.get('thread_id', ''), sources)
        except Exception:
            return 'Not sent: I could not check the outside-text guard.'
    return send_message


def build_messages_sent_tool(sender):
    @tool
    def messages_sent(runtime: ToolRuntime, limit: int = 5) -> str:
        """CALL THIS when he asks whether a message went through, or what she has sent.

        Answer from this record, never from memory or a note. Owner only. Most recent
        attempts first (up to 10), Eastern time. Sent means Discord accepted it;
        outcome unknown means it may have gone, never call it sent.
        """
        try:
            conf = runtime.config.get('configurable') or {}
            return sender.messages_sent(limit, conf.get('identity') or {}, conf.get('thread_id', ''))
        except Exception:
            return "I couldn't read my send record just now"
    return messages_sent


def message_names_for(database_url):
    def names(ids):
        if not database_url or not ids:
            return {}
        import psycopg
        with psycopg.connect(database_url, connect_timeout=5,
                             options='-c statement_timeout=5000 -c default_transaction_read_only=on') as c:
            rows = c.execute("SELECT id, display_name FROM persons "
                             "WHERE deleted_at IS NULL AND id::text = ANY(%s)", (list(ids),)).fetchall()
        return {str(pid): name for pid, name in rows}
    return names


def message_sender_for(settings, gate):
    reasons = []
    if not getattr(settings, 'messages_as_herself', False):
        reasons.append('MESSAGES_AS_HERSELF is off')
    if not getattr(settings, 'discord_bot_token', None):
        reasons.append('bot token missing')
    people = {p.strip() for p in (getattr(settings, 'message_people', '') or '').split(',') if p.strip()}
    if not people:
        reasons.append('MESSAGE_PEOPLE is empty')
    if gate is None or gate.people is None:
        reasons.append('gate or people lookup unavailable')
    if not getattr(settings, 'database_url', None):
        reasons.append('message database unavailable')
    if reasons:
        log.info('send_message not armed: %s', '; '.join(reasons))
        return None
    from aerys_v2.kael_line import _discord_dm_send
    token = settings.discord_bot_token.get_secret_value()
    sender = MessageSender(gate, people, PostgresMessageSends(settings.database_url),
                           lambda address, text: _discord_dm_send(token, address, text),
                           names=message_names_for(getattr(settings, 'memories_database_url', None)))
    log.info('send_message armed')
    return sender
