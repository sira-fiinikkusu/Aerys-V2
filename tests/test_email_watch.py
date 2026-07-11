"""Offline tests for the email inbox watcher — no IMAP socket, no Postgres, no Discord.

FakeImap replays a canned mailbox through the same imaplib-shaped subset the core
uses (select/response/uid), including the real `UID N:*` server quirk (a no-new-mail
search still returns the highest-UID message). FakeConn routes by SQL substring,
same pattern as test_extraction. What these prove: first run and UIDVALIDITY reset
ping NOTHING (fresh-start / no-flood posture), incremental fetch pings each new
message exactly once in arrival order, the watermark advances PER MESSAGE so a
mid-burst notify failure defers-but-never-loses pings, the stored watermark is the
server's own integers verbatim, watching never marks mail read (readonly select +
BODY.PEEK), and any IMAP/DB error returns cleanly instead of raising.
"""

from email.message import EmailMessage

import pytest

from aerys_v2.workers.email_watch import (
    EmailWatchError,
    check_inbox,
    format_ping,
    parse_message,
    read_db_watermark,
    run_once,
    save_db_watermark,
)

UV = 1111  # the mailbox's UIDVALIDITY in most tests


def raw_email(sender="Chris Perry <chris@example.com>", subject="Hello there",
              body="Just checking in about the weekend plans.",
              date="Fri, 10 Jul 2026 12:00:00 -0400"):
    return (
        f"From: {sender}\r\nSubject: {subject}\r\nDate: {date}\r\n"
        f"Content-Type: text/plain; charset=utf-8\r\n\r\n{body}"
    ).encode()


def multipart_email(plain="the plain part", html="<b>the html part</b>"):
    msg = EmailMessage()
    msg["From"] = "Megan <megan@example.com>"
    msg["Subject"] = "Multipart"
    msg["Date"] = "Fri, 10 Jul 2026 12:00:00 -0400"
    msg.set_content(plain)
    msg.add_alternative(html, subtype="html")
    return msg.as_bytes()


class FakeImap:
    """imaplib-shaped fake: select/response/uid, canned {uid: raw_bytes} mailbox.

    SEARCH mimics the REAL `UID N:*` behavior: when nothing has a UID >= N, the
    server returns the highest-UID message anyway (`*` resolves to the last
    message) — the quirk the core must filter client-side."""

    def __init__(self, uidvalidity=UV, messages=None, fail_select=False):
        self.uidvalidity = uidvalidity
        self.messages = dict(messages or {})
        self.calls = []
        self.logged_out = False
        self.fail_select = fail_select

    def select(self, mailbox, readonly=False):
        self.calls.append(("select", mailbox, readonly))
        if self.fail_select:
            return ("NO", [b"unavailable"])
        return ("OK", [str(len(self.messages)).encode()])

    def response(self, code):
        self.calls.append(("response", code))
        if code == "UIDVALIDITY" and self.uidvalidity is not None:
            return (code, [str(self.uidvalidity).encode()])
        return (code, [None])

    def uid(self, command, *args):
        self.calls.append(("uid", command) + args)
        if command == "SEARCH":
            criteria = args[-1]
            all_uids = sorted(self.messages)
            if criteria == "ALL":
                found = all_uids
            else:  # "UID N:*"
                start = int(criteria.split()[1].split(":")[0])
                found = [u for u in all_uids if u >= start]
                if not found and all_uids:
                    found = [all_uids[-1]]  # the N:* quirk
            return ("OK", [" ".join(str(u) for u in found).encode()])
        if command == "FETCH":
            u = int(args[0])
            raw = self.messages[u]
            header = f"{u} (UID {u} BODY[] {{{len(raw)}}}".encode()
            return ("OK", [(header, raw), b")"])
        raise AssertionError(f"unexpected uid command {command!r}")

    def logout(self):
        self.logged_out = True
        return ("BYE", [b"logging out"])


class FakeCursor:
    def __init__(self, rows):
        self.rows = rows

    def fetchone(self):
        return self.rows[0] if self.rows else None

    def fetchall(self):
        return self.rows


class FakeConn:
    """Duck-typed psycopg connection: routes by SQL substring, records every call."""

    def __init__(self, routes=()):
        self.routes = list(routes)
        self.calls = []

    def execute(self, sql, params=None):
        self.calls.append((sql, params))
        for needle, rows in self.routes:
            if needle in sql:
                return FakeCursor(rows)
        return FakeCursor([])


class BrokenConn:
    def execute(self, sql, params=None):
        raise RuntimeError("NAS Postgres is down")


class Notify:
    def __init__(self, fail_at=None):
        self.sent = []
        self.fail_at = fail_at  # 0-based index of the ping that raises

    def __call__(self, text):
        if self.fail_at is not None and len(self.sent) == self.fail_at:
            raise RuntimeError("discord webhook down")
        self.sent.append(text)


def conn_with_watermark(uidvalidity, last_uid):
    return FakeConn([("FROM v2_email_watermark", [(uidvalidity, last_uid)])])


def watermark_saves(conn):
    return [p for s, p in conn.calls if "INSERT INTO v2_email_watermark" in s]


# --- pure core: check_inbox ---------------------------------------------------


def test_first_run_marks_current_max_and_returns_nothing():
    """No watermark -> adopt the current max UID, ZERO messages (fresh-start)."""
    imap = FakeImap(messages={1: raw_email(), 2: raw_email(), 7: raw_email()})
    messages, wm = check_inbox(imap, None)
    assert messages == []
    assert wm == {"uidvalidity": UV, "last_uid": 7}


def test_first_run_on_empty_mailbox_marks_zero():
    messages, wm = check_inbox(FakeImap(messages={}), None)
    assert messages == []
    assert wm == {"uidvalidity": UV, "last_uid": 0}


def test_uidvalidity_change_resets_without_flooding():
    """A reindexed mailbox (new UIDVALIDITY) with 500 messages must produce ZERO
    pings — last_uid jumps straight to the new max."""
    imap = FakeImap(uidvalidity=2222, messages={u: raw_email() for u in range(1, 501)})
    messages, wm = check_inbox(imap, {"uidvalidity": UV, "last_uid": 400})
    assert messages == []
    assert wm == {"uidvalidity": 2222, "last_uid": 500}


def test_incremental_fetch_returns_only_new_ascending():
    imap = FakeImap(messages={
        1: raw_email(subject="old"),
        2: raw_email(subject="also old"),
        3: raw_email(subject="third"),
        5: raw_email(subject="fifth"),
    })
    messages, wm = check_inbox(imap, {"uidvalidity": UV, "last_uid": 2})
    assert [m.uid for m in messages] == [3, 5]
    assert [m.subject for m in messages] == ["third", "fifth"]
    assert wm == {"uidvalidity": UV, "last_uid": 5}


def test_uid_star_quirk_no_new_mail_yields_nothing():
    """last_uid == the highest UID: the server still answers `UID 6:*` with
    message 5 — the client-side filter must drop it, watermark unchanged."""
    imap = FakeImap(messages={5: raw_email()})
    messages, wm = check_inbox(imap, {"uidvalidity": UV, "last_uid": 5})
    assert messages == []
    assert wm == {"uidvalidity": UV, "last_uid": 5}


def test_readonly_select_and_body_peek():
    """Watching must never mark her mail as read."""
    imap = FakeImap(messages={3: raw_email()})
    check_inbox(imap, {"uidvalidity": UV, "last_uid": 2})
    select_calls = [c for c in imap.calls if c[0] == "select"]
    assert select_calls == [("select", "INBOX", True)]
    fetch_calls = [c for c in imap.calls if c[:2] == ("uid", "FETCH")]
    assert fetch_calls and all("BODY.PEEK[]" in c[-1] for c in fetch_calls)


def test_missing_uidvalidity_raises():
    """Pure core raises (the shell catches) — no UIDVALIDITY, no safe watermark."""
    imap = FakeImap(uidvalidity=None, messages={1: raw_email()})
    with pytest.raises(EmailWatchError):
        check_inbox(imap, None)


def test_failed_select_raises():
    with pytest.raises(EmailWatchError):
        check_inbox(FakeImap(fail_select=True), None)


# --- parsing + ping format ----------------------------------------------------


def test_ping_carries_sender_subject_snippet():
    msg = parse_message(9, raw_email(
        sender="Chris Perry <chris@example.com>",
        subject="Weekend plans",
        body="Want to grab dinner Saturday?",
    ))
    ping = format_ping(msg)
    assert "Chris Perry <chris@example.com>" in ping
    assert "Subject: Weekend plans" in ping
    assert "Want to grab dinner Saturday?" in ping


def test_rfc2047_subject_is_decoded():
    # "Résumé ✓" RFC2047-encoded — the ping must show the decoded text.
    msg = parse_message(1, raw_email(subject="=?utf-8?b?UsOpc3Vtw6kg4pyT?="))
    assert msg.subject == "Résumé ✓"


def test_multipart_prefers_plain_over_html():
    msg = parse_message(1, multipart_email(plain="the plain part"))
    assert msg.snippet == "the plain part"
    assert "html" not in msg.snippet


def test_snippet_is_one_line_and_truncated():
    body = "line one\r\nline two\r\n" + "x" * 500
    msg = parse_message(1, raw_email(body=body))
    assert "\n" not in msg.snippet and "\r" not in msg.snippet
    assert msg.snippet.startswith("line one line two")
    assert len(msg.snippet) <= 200
    assert msg.snippet.endswith("…")


def test_unparseable_message_yields_stub_not_crash():
    msg = parse_message(4, None)  # not even bytes — parser blows up internally
    assert msg.uid == 4
    assert "unparseable" in msg.subject


def test_missing_headers_get_placeholders():
    msg = parse_message(1, b"Content-Type: text/plain\r\n\r\nbody only")
    assert msg.sender == "(unknown sender)"
    assert msg.subject == "(no subject)"


# --- shell: run_once ------------------------------------------------------------


def test_run_once_first_run_saves_watermark_pings_nothing():
    conn = FakeConn()  # no watermark row
    imap = FakeImap(messages={1: raw_email(), 8: raw_email()})
    notify = Notify()
    stats = run_once(conn, lambda: imap, notify)

    assert notify.sent == []
    assert stats["ok"] and stats["first_run"] and stats["new"] == 0
    saves = watermark_saves(conn)
    assert saves == [{"mailbox": "INBOX", "uidvalidity": UV, "last_uid": 8}]
    assert imap.logged_out


def test_run_once_pings_each_new_message_and_advances():
    conn = conn_with_watermark(UV, 2)
    imap = FakeImap(messages={
        2: raw_email(subject="old"),
        3: raw_email(subject="first new", body="alpha"),
        4: raw_email(subject="second new", body="beta"),
    })
    notify = Notify()
    stats = run_once(conn, lambda: imap, notify)

    assert stats["ok"] and stats["new"] == 2 and stats["pinged"] == 2
    assert len(notify.sent) == 2
    assert "first new" in notify.sent[0] and "second new" in notify.sent[1]
    # watermark advanced PER MESSAGE: one save per ping, server ints verbatim
    saves = watermark_saves(conn)
    assert saves == [
        {"mailbox": "INBOX", "uidvalidity": UV, "last_uid": 3},
        {"mailbox": "INBOX", "uidvalidity": UV, "last_uid": 4},
    ]


def test_run_once_quiet_poll_writes_nothing():
    conn = conn_with_watermark(UV, 5)
    imap = FakeImap(messages={5: raw_email()})  # N:* quirk fires, nothing new
    stats = run_once(conn, lambda: imap, Notify())
    assert stats["ok"] and stats["new"] == 0
    assert watermark_saves(conn) == []


def test_run_once_uidvalidity_reset_saves_new_mark_no_pings():
    conn = conn_with_watermark(UV, 3)
    imap = FakeImap(uidvalidity=2222, messages={u: raw_email() for u in range(1, 51)})
    notify = Notify()
    stats = run_once(conn, lambda: imap, notify)

    assert notify.sent == []
    assert stats["ok"] and stats["reset"] and stats["new"] == 0
    assert watermark_saves(conn) == [
        {"mailbox": "INBOX", "uidvalidity": 2222, "last_uid": 50}
    ]


def test_notify_failure_holds_watermark_below_unpinged_message():
    """3 new messages, the 2nd ping raises: the 1st is pinged AND behind the
    saved mark (never re-pings); the 2nd and 3rd stay AHEAD of the mark
    (deferred to next tick, not lost); run_once returns cleanly."""
    conn = conn_with_watermark(UV, 10)
    imap = FakeImap(messages={
        11: raw_email(subject="one"),
        12: raw_email(subject="two"),
        13: raw_email(subject="three"),
    })
    notify = Notify(fail_at=1)
    stats = run_once(conn, lambda: imap, notify)

    assert len(notify.sent) == 1 and "one" in notify.sent[0]
    assert stats["ok"] is False and stats["pinged"] == 1
    # ONE save, at uid 11 — never past the un-pinged 12
    assert watermark_saves(conn) == [
        {"mailbox": "INBOX", "uidvalidity": UV, "last_uid": 11}
    ]


def test_imap_factory_failure_returns_cleanly():
    def exploding_factory():
        raise ConnectionRefusedError("imap.gmail.com unreachable")

    stats = run_once(FakeConn(), exploding_factory, Notify())
    assert stats["ok"] is False and stats["error"]
    assert stats["pinged"] == 0


def test_imap_protocol_failure_returns_cleanly_and_logs_out():
    imap = FakeImap(fail_select=True)
    stats = run_once(FakeConn(), lambda: imap, Notify())
    assert stats["ok"] is False and stats["error"]
    assert imap.logged_out  # teardown ran even though the check raised


def test_db_failure_returns_cleanly():
    imap = FakeImap(messages={1: raw_email()})
    stats = run_once(BrokenConn(), lambda: imap, Notify())
    assert stats["ok"] is False and "down" in stats["error"]


def test_watermark_save_failure_after_ping_stops_at_least_once():
    """Ping delivered, save fails: run_once stops, flags the pass, and the
    message may re-ping next tick — at-least-once by design."""

    class SaveFailsConn(FakeConn):
        def execute(self, sql, params=None):
            if "INSERT INTO v2_email_watermark" in sql:
                raise RuntimeError("write failed")
            return super().execute(sql, params)

    conn = SaveFailsConn([("FROM v2_email_watermark", [(UV, 1)])])
    imap = FakeImap(messages={2: raw_email(subject="landed"), 3: raw_email()})
    notify = Notify()
    stats = run_once(conn, lambda: imap, notify)

    assert len(notify.sent) == 1  # stopped after the first save blew up
    assert stats["ok"] is False and "save failed" in stats["error"]


# --- DB watermark helpers -------------------------------------------------------


def test_read_db_watermark_none_then_ints():
    assert read_db_watermark(FakeConn(), "INBOX") is None
    got = read_db_watermark(conn_with_watermark(1234, 42), "INBOX")
    assert got == {"uidvalidity": 1234, "last_uid": 42}


def test_save_db_watermark_uses_named_params_verbatim_ints():
    conn = FakeConn()
    save_db_watermark(conn, "INBOX", {"uidvalidity": 999, "last_uid": 7})
    sql, params = conn.calls[0]
    assert "ON CONFLICT (mailbox)" in sql
    assert params == {"mailbox": "INBOX", "uidvalidity": 999, "last_uid": 7}


# ---- downtime backlog collapse (review finding 2026-07-11) --------------------


def _backlog_imap(count, start_uid=3):
    msgs = {2: raw_email(subject="old")}
    for i in range(count):
        msgs[start_uid + i] = raw_email(subject=f"backlog {i}", body=f"msg {i}")
    return FakeImap(messages=msgs)


def test_backlog_past_threshold_collapses_to_one_summary_ping():
    from aerys_v2.workers.email_watch import BACKLOG_COLLAPSE_AT

    n = BACKLOG_COLLAPSE_AT + 5
    conn = conn_with_watermark(UV, 2)
    notify = Notify()
    stats = run_once(conn, lambda: _backlog_imap(n), notify)

    assert stats["ok"] and stats["new"] == n and stats["pinged"] == 1
    assert len(notify.sent) == 1
    assert f"{n} new emails" in notify.sent[0]
    assert f"backlog {n - 1}" in notify.sent[0]  # names the newest
    # ONE save, straight past the whole backlog
    saves = watermark_saves(conn)
    assert saves == [{"mailbox": "INBOX", "uidvalidity": UV, "last_uid": 2 + n}]


def test_backlog_at_threshold_still_pings_individually():
    from aerys_v2.workers.email_watch import BACKLOG_COLLAPSE_AT

    conn = conn_with_watermark(UV, 2)
    notify = Notify()
    stats = run_once(conn, lambda: _backlog_imap(BACKLOG_COLLAPSE_AT), notify)
    assert stats["pinged"] == BACKLOG_COLLAPSE_AT
    assert len(notify.sent) == BACKLOG_COLLAPSE_AT


def test_backlog_summary_notify_failure_holds_the_watermark():
    from aerys_v2.workers.email_watch import BACKLOG_COLLAPSE_AT

    conn = conn_with_watermark(UV, 2)
    notify = Notify(fail_at=0)  # the summary itself fails
    stats = run_once(conn, lambda: _backlog_imap(BACKLOG_COLLAPSE_AT + 5), notify)
    assert not stats["ok"]
    assert watermark_saves(conn) == []  # nothing advanced; whole backlog retries
