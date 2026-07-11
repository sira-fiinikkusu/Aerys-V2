"""email_tool — the EMAIL TOOLS: search/read her own Gmail inbox over IMAP, plus
draft-then-confirm sending over SMTP as her own address.

n8n mapping: this file replaces V1's 05-03 Email Sub-Agent (kbKrKBVUgwU6n9gg) —
an AI Agent sub-workflow hung off the Core Agent by a toolWorkflow node, with a
Check Email Auth gate that HAD to be wired before Parse Email Intent (miswire it
and every caller got email access). Here there is no sub-agent and no wiring
order to get wrong: these are four flat LangChain tools the action graph binds,
and the send capability is gated by an explicit `confirmed` parameter instead of
a workflow branch.

DATABASE: NONE. This module receives no conn — neither the prod `aerys` DB nor
the brain's `aerys_v2` DB. The mailbox itself is the durable store (Gmail keeps
the Sent copy), so there is no outbox row in v1; if send auditing is wanted
later, it belongs in a conn_factory seam like home_control's, not here yet.

The injection seam (same philosophy as build_home_control_tool): the integrator
wires credentials by passing two zero-arg factories —

  imap_factory() -> an authenticated imaplib.IMAP4_SSL-shaped object
                    (imap.gmail.com, already logged in; a fake in tests)
  smtp_factory() -> an authenticated smtplib.SMTP_SSL-shaped object
                    (smtp.gmail.com, already logged in; a fake in tests)

and self_address, the ONLY From address sends ever use. The tools NEVER read
Settings and open nothing at import time — construction knows config, behavior
doesn't. Connections are opened per call and closed best-effort, so a stale
socket can never wedge a later turn.

Contracts every tool here obeys (same as tools/home_control.py, tools/web_search.py):

1. HONEST FAILURE — every error path returns a plain STRING the model must
   relay. NEVER raise out of a tool: an exception inside a ToolNode kills the
   whole action turn (the V1 failed-webhook-kills-execution outage mode). A
   dead IMAP server, a malformed message, an SMTP refusal — all come back as
   honest words.
2. DRAFT-THEN-CONFIRM — draft_email has ZERO side effects; it exists so the
   fully-formatted draft lands in the conversation where the owner can read it.
   send_email refuses unless confirmed=true, and its description spells out the
   concrete trigger ("the owner has seen this exact draft and explicitly said
   to send it in their most recent message") — the V1 lesson that generic
   instructions get skipped but unmistakable triggers get followed.
3. ANTI-PLACEHOLDER — the presence-gate-placeholder-trap lesson: a model that
   must fill a `to` field will invent one. Both draft_email and send_email
   describe-and-enforce "never invent an address": descriptions say to ASK when
   the owner didn't give one, and a format backstop refuses example.com-style
   placeholders and non-address strings outright.
4. READ-ONLY READS — search/read select the mailbox readonly, so peeking at
   the inbox can never flip \\Seen flags or otherwise change anything.
"""

import email
import email.policy
import logging
from email.header import decode_header, make_header
from email.message import EmailMessage
from typing import Any, Callable

from langchain_core.tools import tool

log = logging.getLogger(__name__)

# The seams: zero-arg callables returning connected+authenticated clients.
ImapFactory = Callable[[], Any]
SmtpFactory = Callable[[], Any]

# Result-count ceiling for search — enough to disambiguate, few enough to not
# blow the tool-message budget (same knob philosophy as home_control.SEARCH_LIMIT).
SEARCH_LIMIT_MAX = 10
SEARCH_LIMIT_DEFAULT = 5

# read_email body cap — a full email rides inside the model's context window.
BODY_TRUNCATE_AT = 4000
TRUNCATION_NOTE = "\n\n[truncated — the message continues beyond 4000 characters]"

# Per-line header trims for the compact search listing.
FROM_TRUNCATE_AT = 60
SUBJECT_TRUNCATE_AT = 100

# The refusal send_email returns when confirmed isn't true — a fixed sentence so
# tests (and the model) see one stable contract, not paraphrase drift.
UNCONFIRMED_REFUSAL = (
    "NOT SENT: send_email requires confirmed=true. Show the owner the exact "
    "draft (use draft_email) and get an explicit yes in their most recent "
    "message before setting confirmed=true."
)

# Anti-placeholder backstop (reference_presence_gate_placeholder_trap): the
# classic invented addresses a model reaches for when it was never given one.
_PLACEHOLDER_DOMAINS = frozenset({"example.com", "example.org", "example.net"})
_PLACEHOLDER_LOCALPARTS = frozenset(
    {"recipient", "someone", "placeholder", "email", "address", "name", "user"}
)


def _addr_problem(addr: str) -> str | None:
    """None if `addr` looks like a real single address; an honest reason if not.

    Deliberately shallow (contains exactly one @, a dotted domain, no spaces) —
    this is a placeholder trap, not RFC 5322 validation. Gmail rejects genuinely
    bad addresses and that failure comes back honestly from send anyway.
    """
    a = addr.strip()
    if not a:
        return "no recipient address was given"
    # CR/LF inside an address is header injection ('a\nb@c.io' — .strip() only
    # trims the ends) and crashes EmailMessage header assembly (review finding
    # 2026-07-11). Any whitespace, not just literal space.
    if any(c.isspace() for c in a) or a.count("@") != 1:
        return f"'{a}' is not a valid email address"
    local, _, domain = a.partition("@")
    if not local or "." not in domain:
        return f"'{a}' is not a valid email address"
    if domain.lower() in _PLACEHOLDER_DOMAINS or local.lower() in _PLACEHOLDER_LOCALPARTS:
        return (
            f"'{a}' looks like a placeholder, not a real address — the owner "
            "never gave this address. Ask the owner for the recipient's actual "
            "email address; never invent one."
        )
    return None


def _validate_recipients(to: str) -> tuple[list[str], str | None]:
    """Split a comma-separated `to` into clean addresses; (addrs, problem)."""
    addrs = [a.strip() for a in (to or "").split(",") if a.strip()]
    if not addrs:
        return [], "no recipient address was given"
    for a in addrs:
        problem = _addr_problem(a)
        if problem:
            return [], problem
    return addrs, None


def _imap_quote(text: str) -> str:
    """Quote a string for use inside an IMAP SEARCH command.

    CR/LF are stripped, not escaped: a newline inside the quoted string ends
    the IMAP command line and starts a NEW command (injection surface — the
    trigger path is prompt-injected mail steering tool args; review finding
    2026-07-11). Search terms lose nothing meaningful by dropping newlines.
    """
    clean = text.replace("\r", " ").replace("\n", " ")
    return '"' + clean.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _decode_hdr(value: Any) -> str:
    """RFC 2047 header → readable str, never raising (malformed → best effort)."""
    if not value:
        return ""
    try:
        decoded = str(make_header(decode_header(str(value))))
    except Exception:
        decoded = str(value)
    return " ".join(decoded.split())  # collapse folded-header newlines/runs


def _trim(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit].rstrip() + "…"


def _fetch_payload(data: Any) -> bytes | None:
    """Extract the literal bytes from an imaplib FETCH response.

    imaplib returns a list mixing b')' terminators and (envelope, payload)
    tuples; the message bytes are the second element of a tuple item.
    """
    for item in data or []:
        if isinstance(item, tuple) and len(item) >= 2 and isinstance(item[1], (bytes, bytearray)):
            return bytes(item[1])
    return None


def _close_quietly(client: Any, method_names: tuple[str, ...]) -> None:
    """Best-effort teardown — a hung LOGOUT/QUIT must never cost the turn."""
    for name in method_names:
        fn = getattr(client, name, None)
        if callable(fn):
            try:
                fn()
            except Exception:
                pass
            return


def build_email_tools(
    *,
    imap_factory: ImapFactory,
    smtp_factory: SmtpFactory,
    self_address: str,
) -> list:
    """Close over the mailbox seams and return the four email tools.

    Everything injectable, same seam as build_home_control_tool: tests pass
    fakes; the factory passes real imaplib/smtplib constructors closed over the
    Gmail app-password credentials. self_address is the owner-agent's own Gmail
    address — the only From identity sends ever carry.
    """
    if not self_address or "@" not in self_address:
        # Construction-time honesty (the build_web_search_tool blank-key rule):
        # a bogus self address is a wiring bug, caught here, not at first send.
        raise ValueError("build_email_tools needs a real self_address (got %r)" % self_address)
    sender = self_address.strip()

    def _uid_search(imap: Any, query: str) -> list[bytes]:
        """UID SEARCH: Gmail X-GM-RAW first (full gmail search syntax), plain
        SUBJECT/FROM/TEXT OR-search as the fallback. Raises on hard failure —
        callers wrap in the honest-string try/except."""
        q = _imap_quote(query)
        data = None
        try:
            typ, data = imap.uid("SEARCH", "X-GM-RAW", q)
            if typ != "OK":
                data = None
        except Exception:
            # Not Gmail (or the extension refused) — fall through to plain SEARCH.
            data = None
        if data is None:
            typ, data = imap.uid(
                "SEARCH", "OR", "OR", "SUBJECT", q, "FROM", q, "TEXT", q
            )
            if typ != "OK":
                raise RuntimeError(f"IMAP search returned {typ}")
        return (data[0] or b"").split()

    @tool
    def search_email(query: str, limit: int = SEARCH_LIMIT_DEFAULT) -> str:
        """Search the owner's OWN Gmail inbox for messages.

        CALL THIS TOOL whenever the user asks about their email — "did I get an
        email from...", "any mail about the invoice?", "find that message from
        Delta", "check my inbox". You have NO access to the mailbox except
        through this tool; never claim to know what's in the inbox without
        calling it.

        query: what to look for. Gmail search syntax works here — e.g.
        "from:delta.com", "subject:invoice", "newer_than:7d flight" — and plain
        words match sender/subject/body as a fallback.
        limit: how many results to return (default 5, max 10).

        Returns the newest matches first, one per line:
        `uid | from | date | subject`. To read a message's body, call
        read_email with the uid from this listing.
        """
        q = (query or "").strip()
        if not q:
            return "search_email needs a query — there is nothing to search for."
        try:
            count = int(limit)
        except (TypeError, ValueError):
            count = SEARCH_LIMIT_DEFAULT
        count = max(1, min(count, SEARCH_LIMIT_MAX))

        try:
            imap = imap_factory()
        except Exception as e:
            return f"email search failed: the mailbox is unreachable right now ({e})."
        try:
            imap.select("INBOX", readonly=True)
            uids = _uid_search(imap, q)
            if not uids:
                return f"No emails in the inbox match {q!r}."
            # UIDs ascend with arrival — take the newest `count`, newest first.
            picked = uids[-count:][::-1]
            lines: list[str] = []
            for uid in picked:
                uid_s = uid.decode("ascii", "replace")
                typ, data = imap.uid(
                    "FETCH", uid_s, "(BODY.PEEK[HEADER.FIELDS (FROM SUBJECT DATE)])"
                )
                raw = _fetch_payload(data) if typ == "OK" else None
                if raw is None:
                    lines.append(f"{uid_s} | (headers unavailable)")
                    continue
                msg = email.message_from_bytes(raw)
                lines.append(
                    f"{uid_s}"
                    f" | {_trim(_decode_hdr(msg.get('From')), FROM_TRUNCATE_AT) or '(unknown sender)'}"
                    f" | {_decode_hdr(msg.get('Date')) or '(no date)'}"
                    f" | {_trim(_decode_hdr(msg.get('Subject')), SUBJECT_TRUNCATE_AT) or '(no subject)'}"
                )
            return "\n".join(lines)
        except Exception as e:
            return f"email search failed: {e}."
        finally:
            _close_quietly(imap, ("logout", "close"))

    @tool
    def read_email(uid: str) -> str:
        """Read ONE email from the owner's inbox — headers plus the plain-text body.

        CALL THIS TOOL after search_email, passing a uid from its results, when
        the user wants to know what a message actually says. Never summarize an
        email you have not read through this tool.

        uid: the message uid exactly as search_email listed it.

        Returns From/To/Date/Subject headers and the decoded text body
        (truncated past 4000 characters, with a note when that happens).
        """
        uid_s = (uid or "").strip() if isinstance(uid, str) else str(uid)
        if not uid_s:
            return "read_email needs the uid of a message — call search_email first."
        # uids are digits, full stop. Anything else spliced into imap.uid() is
        # protocol text under model control (review finding 2026-07-11).
        if not uid_s.isdigit():
            return (
                f"'{uid_s}' is not a message uid — uids are plain numbers from "
                "search_email's results."
            )
        try:
            imap = imap_factory()
        except Exception as e:
            return f"email read failed: the mailbox is unreachable right now ({e})."
        try:
            imap.select("INBOX", readonly=True)
            typ, data = imap.uid("FETCH", uid_s, "(BODY.PEEK[])")
            raw = _fetch_payload(data) if typ == "OK" else None
            if raw is None:
                return (
                    f"No email with uid {uid_s} was found — the uid may be stale; "
                    "run search_email again for fresh uids."
                )
            msg = email.message_from_bytes(raw, policy=email.policy.default)
            header_lines = [
                f"From: {_decode_hdr(msg.get('From')) or '(unknown)'}",
                f"To: {_decode_hdr(msg.get('To')) or '(unknown)'}",
                f"Date: {_decode_hdr(msg.get('Date')) or '(unknown)'}",
                f"Subject: {_decode_hdr(msg.get('Subject')) or '(no subject)'}",
            ]
            body_part = msg.get_body(preferencelist=("plain",))
            if body_part is None:
                if msg.get_body(preferencelist=("html",)) is not None:
                    body = (
                        "(this message has no plain-text body — it is HTML-only, "
                        "which v1 does not render)"
                    )
                else:
                    body = "(this message has no readable text body)"
            else:
                body = str(body_part.get_content()).strip()
                if len(body) > BODY_TRUNCATE_AT:
                    body = body[:BODY_TRUNCATE_AT] + TRUNCATION_NOTE
            return "\n".join(header_lines) + "\n\n" + body
        except Exception as e:
            return f"email read failed: {e}."
        finally:
            _close_quietly(imap, ("logout", "close"))

    @tool
    def draft_email(to: str, subject: str, body: str) -> str:
        """Compose an email DRAFT. This tool NEVER sends anything — zero side effects.

        CALL THIS TOOL whenever the user asks to email someone, reply to a
        message, or send anything by mail. Then you MUST show the returned draft
        to the owner VERBATIM and ask whether to send it — do not call send_email
        until the owner has read this exact draft and explicitly said yes.

        to: the recipient's email address, EXACTLY as the owner gave it. NEVER
        invent, guess, or fill in a placeholder address — if the owner has not
        given you the recipient's actual address, ASK for it instead of calling
        this tool.
        subject: the subject line.
        body: the full plain-text message body.

        Returns the fully-formatted draft to show the owner.
        """
        addrs, problem = _validate_recipients(to)
        if problem:
            return f"DRAFT NOT CREATED: {problem}"
        subj = (subject or "").strip()
        text = (body or "").strip()
        if not text:
            return "DRAFT NOT CREATED: the body is empty — there is nothing to say yet."
        return (
            "DRAFT (not sent):\n"
            f"From: {sender}\n"
            f"To: {', '.join(addrs)}\n"
            f"Subject: {subj or '(no subject)'}\n"
            "\n"
            f"{text}\n"
            "\n"
            "Show this draft to the owner word-for-word and ask whether to send "
            "it. Only call send_email with confirmed=true after they explicitly "
            "say yes to THIS draft."
        )

    @tool
    def send_email(to: str, subject: str, body: str, confirmed: bool = False) -> str:
        """SEND an email from the owner's own address. IRREVERSIBLE — real mail goes out.

        HARD RULE — confirmed: only set confirmed=true after the owner has SEEN
        this exact draft (shown via draft_email) and EXPLICITLY said to send it
        in their MOST RECENT message — an explicit "yes, send it" / "send that".
        An earlier general request to "email Bob" is NOT confirmation of a draft
        the owner has not read. If in any doubt, call with confirmed=false (or
        show the draft again) — the tool will refuse and nothing is sent.

        to: the recipient's address EXACTLY as the owner gave it. NEVER invent
        or guess an address — if the owner never stated the recipient's actual
        email address, do not call this tool; ask them for the address.
        subject: the subject line, matching the confirmed draft.
        body: the plain-text body, matching the confirmed draft.
        confirmed: true ONLY under the hard rule above; anything else refuses.

        On success returns "Sent: ..." — relay it. On refusal or failure returns
        an honest message; relay that too and never claim the email was sent.
        """
        # Strict truthiness: the model must assert True, not hand back a string
        # that merely LOOKS affirmative ("false" is truthy — the classic trap).
        is_confirmed = confirmed is True or (
            isinstance(confirmed, str) and confirmed.strip().lower() == "true"
        )
        if not is_confirmed:
            return UNCONFIRMED_REFUSAL

        addrs, problem = _validate_recipients(to)
        if problem:
            return f"NOT SENT: {problem}"
        text = (body or "").strip()
        if not text:
            return "NOT SENT: the body is empty — draft the message first."

        # Header assembly must sit inside the tool's never-raise contract:
        # msg['Subject'] raises ValueError on embedded CR/LF ('Header values
        # may not contain linefeed...'), and a raising ToolNode kills the whole
        # action turn (review finding 2026-07-11, reproduced empirically).
        # Subjects are one line by definition — collapse rather than refuse.
        try:
            msg = EmailMessage()
            msg["From"] = sender
            msg["To"] = ", ".join(addrs)
            msg["Subject"] = " ".join((subject or "").split())
            msg.set_content(text)  # EmailMessage handles UTF-8; plain text only in v1
        except Exception as e:
            return f"NOT SENT: couldn't assemble the message headers ({e})."

        try:
            smtp = smtp_factory()
        except Exception as e:
            return f"NOT SENT: the mail server is unreachable right now ({e})."
        try:
            smtp.send_message(msg)
        except Exception as e:
            return f"NOT SENT: the mail server refused the message ({e})."
        finally:
            _close_quietly(smtp, ("quit", "close"))
        # No WRITE_OK_PREFIX here on purpose: an email send has no visible device
        # effect, so the spoken/text confirmation IS the feedback channel (the
        # timer-START reasoning, not the light-toggle one).
        return f"Sent: email to {', '.join(addrs)} — subject {msg['Subject'] or '(no subject)'!r}."

    return [search_email, read_email, draft_email, send_email]
