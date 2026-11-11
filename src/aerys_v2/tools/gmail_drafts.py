"""Owner-only draft creation. No mailbox reads, draft edits, or sending endpoints."""
from __future__ import annotations

import base64
import logging
import re
from email.message import EmailMessage
from email.policy import SMTP

import httpx
from langchain_core.tools import tool
from langgraph.prebuilt import ToolRuntime
from pydantic import SecretStr

log = logging.getLogger(__name__)
TOKEN_URL = 'https://oauth2.googleapis.com/token'
DRAFT_URL = 'https://gmail.googleapis.com/gmail/v1/users/me/drafts'
MAX_BODY = 100_000


def addresses(value: str, *, optional: bool = False) -> str:
    """Conservative bare ASCII mailboxes, comma separated; no display names."""
    if optional and not value:
        return ''
    result = []
    for item in value.split(','):
        item = item.strip()
        if len(item) > 254 or not re.fullmatch(
                r"[A-Za-z0-9!#$%&'*+/=?^_`{|}~.-]+@[A-Za-z0-9.-]+", item):
            raise ValueError('use valid email addresses, separated by commas')
        local, domain = item.rsplit('@', 1)
        if (len(local) > 64 or local.startswith('.') or local.endswith('.') or '..' in local
                or '.' not in domain or any(not label or len(label) > 63 or label.startswith('-')
                                           or label.endswith('-') for label in domain.split('.'))):
            raise ValueError('use valid email addresses, separated by commas')
        result.append(item)
    return ', '.join(result)


class GmailDrafts:
    def __init__(self, *, client_id: str, client_secret: SecretStr, refresh_token: SecretStr,
                 from_address: str, transport=None):
        self.client_id = client_id
        self.client_secret = client_secret
        self.refresh_token = refresh_token
        self.from_address = from_address
        self.transport = transport

    def create(self, to: str, subject: str, body: str, cc: str = '') -> str:
        try:
            if any(any(ord(c) < 32 or ord(c) == 127 for c in value)
                   for value in (to, cc, subject, self.from_address)):
                raise ValueError('headers must not contain control characters')
            to, cc = addresses(to), addresses(cc, optional=True)
            from_address = addresses(self.from_address)
            if ',' in from_address:
                raise ValueError('GMAIL_DRAFTS_FROM must be one email address')
            if not body.strip():
                raise ValueError('the message body is empty')
            if len(body) > MAX_BODY:
                raise ValueError('the message body exceeds 100000 characters')
            msg = EmailMessage(policy=SMTP)
            msg['From'], msg['To'], msg['Subject'] = from_address, to, subject
            if cc:
                msg['Cc'] = cc
            msg.set_content(body)
            raw = base64.urlsafe_b64encode(msg.as_bytes()).decode('ascii')
        except ValueError as exc:
            return f'Not drafted: {exc}.'
        except Exception:
            return 'Not drafted: the message could not be encoded.'
        phase = 'token refresh'
        try:
            # Never follow redirects or retry: only these two endpoints may be used.
            with httpx.Client(timeout=20, follow_redirects=False, transport=self.transport) as client:
                response = client.post(TOKEN_URL, data={
                    'grant_type': 'refresh_token', 'client_id': self.client_id,
                    'client_secret': self.client_secret.get_secret_value(),
                    'refresh_token': self.refresh_token.get_secret_value()})
                response.raise_for_status()
                token = response.json().get('access_token')
                if not isinstance(token, str) or not token.strip():
                    return 'Not drafted: Google returned no access token.'
                phase = 'draft creation'
                response = client.post(DRAFT_URL, headers={'Authorization': f'Bearer {token}'},
                                       json={'message': {'raw': raw}})
                response.raise_for_status()
                draft_id = response.json().get('id')
                if not isinstance(draft_id, str) or not draft_id.strip():
                    return 'Not drafted: Google returned no draft id; check Gmail before trying again.'
        except httpx.HTTPStatusError as exc:
            return f'Not drafted: Google rejected {phase} (HTTP {exc.response.status_code}); check Gmail before trying again.'
        except httpx.TimeoutException:
            return f'Not drafted: {phase} timed out; check Gmail before trying again.'
        except Exception:
            # Never echo response bodies, credentials, or exception text into logs/tools/traces.
            return f'Not drafted: {phase} failed or returned an invalid response; check Gmail before trying again.'
        return f'Drafted: {subject} — to {to}' + (f'; cc {cc}' if cc else '') + '.'


def build_gmail_draft_tool(client: GmailDrafts, *, owner_person_id: str):
    @tool
    def draft_in_his_gmail(to: str, subject: str, body: str, runtime: ToolRuntime, cc: str = '') -> str:
        """Write an unsigned draft in the owner's Gmail for HIM to review and send himself.

        Use when he asks you to write something for HIM to send: a reply in his name,
        an email from him. Use send_email for mail from your own address. Never claim
        this was sent. Supply bare email addresses, comma separated, and no signature
        from you. This creates a draft only; it cannot read his inbox or send mail.
        """
        try:
            conf = runtime.config.get('configurable') or {}
            identity = conf.get('identity') or {}
            if (not owner_person_id or str(identity.get('user_id') or '') != str(owner_person_id)
                    or str(conf.get('thread_id') or '').startswith('kael:')):
                return 'Not drafted: only the owner can ask for a draft in his Gmail.'
        except Exception:
            return 'Not drafted: the owner identity could not be checked.'
        return client.create(to, subject, body, cc)
    return draft_in_his_gmail


def gmail_drafts_for(settings):
    reasons = []
    if not getattr(settings, 'gmail_drafts', False):
        reasons.append('GMAIL_DRAFTS is off')
    for key in ('gmail_drafts_client_id', 'gmail_drafts_client_secret',
                'gmail_drafts_refresh_token', 'gmail_drafts_from', 'owner_person_id'):
        value = getattr(settings, key, None)
        if isinstance(value, SecretStr):
            value = value.get_secret_value()
        if value is None or not str(value).strip():
            reasons.append(key.upper() + ' is missing')
    if not reasons:
        try:
            addr = settings.gmail_drafts_from
            if any(ord(c) < 32 or ord(c) == 127 for c in addr) or ',' in addresses(addr):
                raise ValueError()
        except ValueError:
            reasons.append('GMAIL_DRAFTS_FROM is invalid')
    if reasons:
        log.info('gmail drafts not armed: %s', '; '.join(reasons))
        return None
    client = GmailDrafts(client_id=settings.gmail_drafts_client_id,
                         client_secret=settings.gmail_drafts_client_secret,
                         refresh_token=settings.gmail_drafts_refresh_token,
                         from_address=settings.gmail_drafts_from)
    log.info('gmail drafts armed')
    return build_gmail_draft_tool(client, owner_person_id=str(settings.owner_person_id))
