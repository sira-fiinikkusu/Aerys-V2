"""Offline tests for the email tools — fake IMAP/SMTP objects recording calls.

What these prove: search hits Gmail's X-GM-RAW extension first and falls back to
a plain SUBJECT/FROM/TEXT OR-search, results come back compact and newest-first,
read decodes RFC 2047 headers + the plain body (with visible truncation past
4000 chars), draft_email touches NOTHING (zero factory calls — pure string out),
send_email with confirmed=false NEVER touches smtp_factory, confirmed=true sends
exactly the confirmed content as self_address, placeholder/invented addresses
are refused, and every failure mode is an HONEST string (never a raise — the
ToolNode contract). No network, no real mailbox.
"""

import pytest

from aerys_v2.tools.email_tool import (
    BODY_TRUNCATE_AT,
    SEARCH_LIMIT_MAX,
    UNCONFIRMED_REFUSAL,
    build_email_tools,
)

SELF = "aerys@siravaultlore.work"


# ---- fakes ------------------------------------------------------------------------

def _headers(from_, date, subject):
    return (
        f"From: {from_}\r\nDate: {date}\r\nSubject: {subject}\r\n\r\n"
    ).encode()


PLAIN_MSG = (
    b"From: Delta <no-reply@delta.com>\r\n"
    b"To: Chris <chris@example-owner.dev>\r\n"
    b"Date: Fri, 10 Jul 2026 09:00:00 -0400\r\n"
    b"Subject: =?utf-8?q?Flight_confirmation_=E2=9C=88?=\r\n"
    b"Content-Type: text/plain; charset=utf-8\r\n"
    b"\r\n"
    b"Your flight DL123 departs at 9am.\r\n"
)

HTML_ONLY_MSG = (
    b"From: a@b.co\r\n"
    b"Subject: shiny\r\n"
    b"Content-Type: text/html; charset=utf-8\r\n"
    b"\r\n"
    b"<p>hello</p>\r\n"
)


class FakeIMAP:
    """Records every call; answers uid SEARCH/FETCH like imaplib would."""

    def __init__(self, *, uids=b"3 7 12", fetch_raw=PLAIN_MSG, gm_raw="ok"):
        # gm_raw: "ok" = X-GM-RAW works, "no" = returns NO, "raise" = raises
        self.calls: list[tuple] = []
        self.uids = uids
        self.fetch_raw = fetch_raw
        self.gm_raw = gm_raw
        self.logged_out = False

    def select(self, mailbox, readonly=False):
        self.calls.append(("select", mailbox, readonly))
        return ("OK", [b"3"])

    def uid(self, command, *args):
        self.calls.append(("uid", command, *args))
        if command == "SEARCH":
            if args and args[0] == "X-GM-RAW":
                if self.gm_raw == "raise":
                    raise Exception("X-GM-RAW unknown")
                if self.gm_raw == "no":
                    return ("NO", [None])
                return ("OK", [self.uids])
            return ("OK", [self.uids])  # the plain fallback search
        if command == "FETCH":
            uid_s, spec = args
            if "HEADER.FIELDS" in spec:
                raw = _headers(
                    f"sender{uid_s}@mail.test",
                    "Thu, 9 Jul 2026 12:00:00 -0400",
                    f"subject for {uid_s}",
                )
            else:
                raw = self.fetch_raw
            if raw is None:
                return ("OK", [None])
            return ("OK", [(f"{uid_s} (BODY[] {{{len(raw)}}}".encode(), raw), b")"])
        raise AssertionError(f"unexpected uid command {command}")

    def logout(self):
        self.calls.append(("logout",))
        self.logged_out = True


class FakeSMTP:
    def __init__(self, *, fail=False):
        self.sent = []
        self.fail = fail
        self.quit_called = False

    def send_message(self, msg):
        if self.fail:
            raise Exception("550 relaying denied")
        self.sent.append(msg)

    def quit(self):
        self.quit_called = True


class CountingFactory:
    """Wraps a fake; records how many times the seam was opened."""

    def __init__(self, obj=None, raise_on_call=False):
        self.obj = obj
        self.calls = 0
        self.raise_on_call = raise_on_call

    def __call__(self):
        self.calls += 1
        if self.raise_on_call:
            raise Exception("connection refused")
        return self.obj


def make_tools(imap=None, smtp=None):
    imap_f = CountingFactory(imap or FakeIMAP())
    smtp_f = CountingFactory(smtp or FakeSMTP())
    search, read, draft, send = build_email_tools(
        imap_factory=imap_f, smtp_factory=smtp_f, self_address=SELF
    )
    return search, read, draft, send, imap_f, smtp_f


# ---- search_email -----------------------------------------------------------------

def test_search_uses_gm_raw_and_formats_newest_first():
    imap = FakeIMAP(uids=b"3 7 12")
    search, *_ = make_tools(imap=imap)
    out = search.invoke({"query": "from:delta.com"})
    # X-GM-RAW rode the first SEARCH, quoted
    search_calls = [c for c in imap.calls if len(c) > 1 and c[1] == "SEARCH"]
    assert search_calls[0][2] == "X-GM-RAW"
    assert search_calls[0][3] == '"from:delta.com"'
    # newest (highest uid) first, compact pipe format
    lines = out.splitlines()
    assert lines[0].startswith("12 | sender12@mail.test | ")
    assert lines[-1].startswith("3 | ")
    assert "subject for 12" in lines[0]
    # reads are readonly — peeking never flips \Seen
    assert ("select", "INBOX", True) in imap.calls
    assert imap.logged_out


def test_search_falls_back_to_plain_search_when_gm_raw_refused():
    for mode in ("no", "raise"):
        imap = FakeIMAP(uids=b"5", gm_raw=mode)
        search, *_ = make_tools(imap=imap)
        out = search.invoke({"query": "invoice"})
        search_calls = [c for c in imap.calls if len(c) > 1 and c[1] == "SEARCH"]
        assert len(search_calls) == 2
        # fallback is the SUBJECT/FROM/TEXT OR-search
        assert search_calls[1][2:] == (
            "OR", "OR", "SUBJECT", '"invoice"', "FROM", '"invoice"', "TEXT", '"invoice"'
        )
        assert out.startswith("5 | ")


def test_search_limit_clamped_to_max():
    imap = FakeIMAP(uids=b"1 2 3 4 5 6 7 8 9 10 11 12")
    search, *_ = make_tools(imap=imap)
    out = search.invoke({"query": "q", "limit": 99})
    assert len(out.splitlines()) == SEARCH_LIMIT_MAX


def test_search_no_matches_is_honest_string():
    search, *_ = make_tools(imap=FakeIMAP(uids=b""))
    out = search.invoke({"query": "nonexistent thing"})
    assert "No emails" in out


def test_search_blank_query_never_opens_imap():
    search, _, _, _, imap_f, _ = make_tools()
    out = search.invoke({"query": "   "})
    assert "nothing to search" in out
    assert imap_f.calls == 0


def test_search_dead_mailbox_is_honest_string_not_exception():
    imap_f = CountingFactory(raise_on_call=True)
    search, *_ = build_email_tools(
        imap_factory=imap_f, smtp_factory=CountingFactory(FakeSMTP()), self_address=SELF
    )
    out = search.invoke({"query": "q"})
    assert out.startswith("email search failed:") and "unreachable" in out


# ---- read_email -------------------------------------------------------------------

def test_read_returns_decoded_headers_and_plain_body():
    _, read, *_ = make_tools(imap=FakeIMAP())
    out = read.invoke({"uid": "12"})
    assert "From: Delta <no-reply@delta.com>" in out
    assert "Subject: Flight confirmation ✈" in out  # RFC 2047 decoded
    assert "Your flight DL123 departs at 9am." in out


def test_read_truncates_long_body_with_note():
    long_body = b"x" * (BODY_TRUNCATE_AT + 500)
    raw = (
        b"From: a@b.co\r\nSubject: big\r\n"
        b"Content-Type: text/plain; charset=utf-8\r\n\r\n" + long_body
    )
    _, read, *_ = make_tools(imap=FakeIMAP(fetch_raw=raw))
    out = read.invoke({"uid": "1"})
    assert "truncated" in out
    assert "x" * (BODY_TRUNCATE_AT + 1) not in out


def test_read_html_only_is_honest_note_not_garbage():
    _, read, *_ = make_tools(imap=FakeIMAP(fetch_raw=HTML_ONLY_MSG))
    out = read.invoke({"uid": "1"})
    assert "HTML-only" in out
    assert "<p>" not in out


def test_read_missing_uid_is_honest_string():
    _, read, *_ = make_tools(imap=FakeIMAP(fetch_raw=None))
    out = read.invoke({"uid": "999"})
    assert "999" in out and "stale" in out


def test_read_empty_uid_never_opens_imap():
    _, read, _, _, imap_f, _ = make_tools()
    out = read.invoke({"uid": ""})
    assert "search_email first" in out
    assert imap_f.calls == 0


# ---- draft_email — ZERO side effects ----------------------------------------------

def test_draft_returns_formatted_draft_and_touches_nothing():
    _, _, draft, _, imap_f, smtp_f = make_tools()
    out = draft.invoke({
        "to": "megan@perry-home.dev",
        "subject": "Dinner Friday",
        "body": "Bringing wine — see you at 7.",
    })
    assert out.startswith("DRAFT (not sent):")
    assert f"From: {SELF}" in out
    assert "To: megan@perry-home.dev" in out
    assert "Subject: Dinner Friday" in out
    assert "Bringing wine — see you at 7." in out
    # the draft itself instructs the show-then-ask step
    assert "confirmed=true" in out and "explicitly" in out
    # NO side effects: neither seam was ever opened
    assert imap_f.calls == 0
    assert smtp_f.calls == 0


def test_draft_refuses_placeholder_address():
    _, _, draft, _, imap_f, smtp_f = make_tools()
    for bogus in ("recipient@example.com", "someone@gmail.com", "not-an-address"):
        out = draft.invoke({"to": bogus, "subject": "s", "body": "b"})
        assert out.startswith("DRAFT NOT CREATED:")
    assert smtp_f.calls == 0 and imap_f.calls == 0


def test_draft_refuses_empty_body():
    _, _, draft, *_ = make_tools()
    out = draft.invoke({"to": "real.person@company.io", "subject": "s", "body": "  "})
    assert out.startswith("DRAFT NOT CREATED:") and "empty" in out


# ---- send_email — the confirmation gate --------------------------------------------

def test_send_unconfirmed_never_touches_smtp_factory():
    _, _, _, send, _, smtp_f = make_tools()
    out = send.invoke({
        "to": "real.person@company.io",
        "subject": "s",
        "body": "b",
        "confirmed": False,
    })
    assert out == UNCONFIRMED_REFUSAL
    assert smtp_f.calls == 0  # the seam was NEVER opened — not opened-then-aborted


def test_send_confirmed_missing_defaults_to_refusal():
    _, _, _, send, _, smtp_f = make_tools()
    out = send.invoke({"to": "real.person@company.io", "subject": "s", "body": "b"})
    assert out == UNCONFIRMED_REFUSAL
    assert smtp_f.calls == 0


def test_send_stringly_false_is_still_refused():
    # "false" is truthy in Python — the gate must check identity, not truthiness.
    _, _, _, send, _, smtp_f = make_tools()
    out = send.invoke({
        "to": "real.person@company.io", "subject": "s", "body": "b", "confirmed": "false",
    })
    assert out == UNCONFIRMED_REFUSAL
    assert smtp_f.calls == 0


def test_send_confirmed_true_sends_as_self_address():
    smtp = FakeSMTP()
    _, _, _, send, _, smtp_f = make_tools(smtp=smtp)
    out = send.invoke({
        "to": "real.person@company.io",
        "subject": "Résumé attached",
        "body": "Hi — see below. — Chris",
        "confirmed": True,
    })
    assert out.startswith("Sent:")
    assert smtp_f.calls == 1
    assert len(smtp.sent) == 1
    msg = smtp.sent[0]
    assert msg["From"] == SELF
    assert msg["To"] == "real.person@company.io"
    assert msg["Subject"] == "Résumé attached"
    assert "Hi — see below. — Chris" in msg.get_content()
    assert smtp.quit_called  # connection closed best-effort


def test_send_confirmed_refuses_placeholder_address_without_sending():
    _, _, _, send, _, smtp_f = make_tools()
    out = send.invoke({
        "to": "recipient@example.com", "subject": "s", "body": "b", "confirmed": True,
    })
    assert out.startswith("NOT SENT:") and "placeholder" in out
    assert smtp_f.calls == 0


def test_send_smtp_failure_is_honest_string_not_exception():
    _, _, _, send, _, _ = make_tools(smtp=FakeSMTP(fail=True))
    out = send.invoke({
        "to": "real.person@company.io", "subject": "s", "body": "b", "confirmed": True,
    })
    assert out.startswith("NOT SENT:") and "550" in out


def test_send_dead_smtp_factory_is_honest_string():
    search_t, read_t, draft_t, send = build_email_tools(
        imap_factory=CountingFactory(FakeIMAP()),
        smtp_factory=CountingFactory(raise_on_call=True),
        self_address=SELF,
    )
    out = send.invoke({
        "to": "real.person@company.io", "subject": "s", "body": "b", "confirmed": True,
    })
    assert out.startswith("NOT SENT:") and "unreachable" in out


# ---- construction + name/description contract (the V1 tool-name-mismatch guard) ----

def test_bogus_self_address_rejected_at_construction():
    with pytest.raises(ValueError):
        build_email_tools(
            imap_factory=CountingFactory(), smtp_factory=CountingFactory(),
            self_address="",
        )
    with pytest.raises(ValueError):
        build_email_tools(
            imap_factory=CountingFactory(), smtp_factory=CountingFactory(),
            self_address="not-an-address",
        )


def test_tool_names_match_prompt_references():
    tools = build_email_tools(
        imap_factory=CountingFactory(FakeIMAP()),
        smtp_factory=CountingFactory(FakeSMTP()),
        self_address=SELF,
    )
    assert [t.name for t in tools] == [
        "search_email", "read_email", "draft_email", "send_email",
    ]


def test_descriptions_carry_the_hard_won_trigger_language():
    search, read, draft, send, *_ = make_tools()
    # concrete triggers, not generalities (V1 lesson: specificity beats generality)
    assert "check my inbox" in search.description.lower()
    # draft: always show, always ask, never invent
    d = draft.description.lower()
    assert "never sends" in d
    assert "verbatim" in d
    assert "never" in d and "invent" in d
    # send: explicit-yes-on-this-exact-draft gate + anti-placeholder
    s = send.description.lower()
    assert "explicitly said" in s
    assert "most recent message" in s
    assert "exact draft" in s
    assert "never invent" in s
    assert "ask" in s


# ---- review-hardening regressions (2026-07-11) -------------------------------------

def test_send_multiline_subject_never_raises_and_collapses_to_one_line():
    """EmailMessage raises ValueError on CR/LF in header values; that must
    surface as an honest string (or a clean send), never a raise out of the
    tool (review finding, reproduced empirically)."""
    smtp = FakeSMTP()
    _, _, _, send, _, _ = make_tools(smtp=smtp)
    out = send.invoke({
        "to": "real.person@company.io",
        "subject": "line one\nline two\r\nline three",
        "body": "hello",
        "confirmed": True,
    })
    assert out.startswith("Sent:")
    assert smtp.sent[0]["Subject"] == "line one line two line three"


def test_send_crlf_in_address_is_refused_not_header_injected():
    _, _, _, send, _, smtp_f = make_tools()
    out = send.invoke({
        "to": "a\nb@c.io",
        "subject": "hi",
        "body": "hello",
        "confirmed": True,
    })
    assert out.startswith("NOT SENT")
    assert smtp_f.calls == 0


def test_read_rejects_non_digit_uid_without_touching_imap():
    _, read, _, _, imap_f, _ = make_tools()
    out = read.invoke({"uid": "12 (BODY[])\r\nUID STORE 1 +FLAGS \\Deleted"})
    assert "not a message uid" in out
    assert imap_f.calls == 0


def test_search_strips_crlf_from_query_before_imap():
    imap = FakeIMAP(gm_raw="no")  # force the plain fallback (quoted terms path)
    search, *_ = make_tools(imap=imap)
    search.invoke({"query": 'invoice\r\nUID STORE 1 +FLAGS \\Deleted'})
    for call in imap.calls:
        for part in call:
            if isinstance(part, str):
                assert "\r" not in part and "\n" not in part
