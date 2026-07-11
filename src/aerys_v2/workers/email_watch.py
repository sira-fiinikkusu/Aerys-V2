"""Email inbox watcher — Aerys's OWN Gmail inbox, notification-only.

n8n mapping: the notification half of 05-03 Gmail Trigger (48toI7JVcl3MnL4n).
Scope (owner decision 2026-07-11): HER inbox only, ping-the-owner-on-Discord
only. No morning brief, no reading, no sending — those are a separate tool.
This worker's entire job is "new mail arrived, here's who/what/first-200-chars".

Pure core + thin shell, same shape as every module here:

  check_inbox(imap_client, watermark)  — PURE core. `imap_client` is an injected
      object exposing the small imaplib-shaped subset this module needs
      (select / response / uid); it never dials anything itself. Raises
      EmailWatchError on protocol trouble — pure handlers let exceptions raise,
      the shell catches (codebase failure doctrine).
  run_once(conn, imap_factory, notify_fn)  — the shell. Opens NOTHING itself:
      `conn` is an already-open connection to the brain's OWN aerys_v2 database
      (this module reads/writes ONLY v2_email_watermark — migration 006; it
      never touches prod `aerys`), `imap_factory()` constructs+logs-in the IMAP
      client (imaplib.IMAP4_SSL against imap.gmail.com in real life — but built
      by the factory, so tests never dial), `notify_fn(str)` delivers one ping
      (the integrator wires Discord DM delivery; the core never touches the
      network). ANY IMAP/DB error here logs and returns cleanly — the scheduler
      retries next tick. Polling cadence is the scheduler's knob: the integrator
      wires run_once as an interval job in workers/__main__ exactly like
      extraction/gaps-mine (_add_interval_job), so no poll loop lives here.

The high-water mark — the extraction lesson, generalized:

  Watermark = {uidvalidity: int, last_uid: int}, the SERVER'S OWN values stored
  verbatim (v2_email_watermark, BIGINTs). extraction.py's ms-vs-µs bug taught
  us never to persist a re-serialized version of a server cursor — a JS Date
  round-trip re-matched the same row forever. Same doctrine here: we store the
  UIDVALIDITY and UID integers exactly as the IMAP server issued them, never a
  fetch timestamp or any locally-derived stand-in.

  UIDVALIDITY change = the server renumbered EVERY message (a Gmail reindex /
  mailbox recreation). Old UIDs are meaningless, so naive "fetch since last_uid"
  would re-match the entire mailbox and replay hundreds of pings. On a change we
  RESET: last_uid jumps to the current max UID and ZERO messages are returned —
  a reindex must never flood the owner. (Mail arriving during the exact reindex
  moment is the accepted cost; flooding is worse than one rare missed ping.)

  First run (no watermark row): same fresh-start posture — mark the current max
  UID, ping NOTHING. Turning the watcher on must not replay the whole inbox.

  The `UID N:*` quirk: when nothing is newer than N, IMAP servers (Gmail
  included) return the HIGHEST-UID message anyway (`*` resolves to the last
  message and the range degenerates to it). Every search result is therefore
  re-filtered client-side to uid > last_uid — without that, the newest email
  would re-ping on every single poll forever.

Per-message watermark advance (the decided failure posture): messages are pinged
in ascending-UID order and the watermark is saved after EACH successful ping. A
notify failure mid-burst stops the loop — already-pinged messages are behind the
mark (no duplicates), the failed one and everything after it are still ahead of
it (retried next tick, so no ping is ever LOST, only deferred). If the DB save
fails after a ping went out, that one message may ping again next tick —
at-least-once is the chosen direction: a duplicate ping beats a lost one.

Read-only mailbox posture: SELECT is issued readonly and fetches use BODY.PEEK[]
— watching never marks her mail as \\Seen. The future read/send tool owns flags.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from email import message_from_bytes
from email import policy as email_policy
from email.utils import parseaddr
from typing import Any, Callable

log = logging.getLogger(__name__)

DEFAULT_MAILBOX = "INBOX"
SNIPPET_CHARS = 200   # body snippet cap (first text/plain part)
SUBJECT_CHARS = 150   # subject cap in the ping
SENDER_CHARS = 120    # sender cap in the ping
# Past this many accumulated messages in one pass (worker downtime with a valid
# watermark), individual pings collapse into ONE summary ping — a restart after
# days away must not machine-gun the owner (review finding 2026-07-11).
BACKLOG_COLLAPSE_AT = 10


class EmailWatchError(RuntimeError):
    """IMAP protocol trouble in the pure core — raised, never swallowed, so the
    shell (run_once) is the ONE place that decides to log-and-return-cleanly.
    Same split as extraction: pure code raises, the binding layer catches."""


@dataclass(frozen=True)
class EmailSummary:
    """One new email, reduced to exactly what a notification needs. All fields
    are plain decoded strings (RFC2047 handled by the email stdlib) and
    single-line sanitized — nothing here is ever executed or interpolated into
    SQL; it only rides inside a Discord ping as data."""

    uid: int
    sender: str    # display name <addr> when both exist, else whichever does
    subject: str
    snippet: str   # first ~SNIPPET_CHARS chars of the text/plain part, one line
    date: str      # the raw Date header (display only — never a watermark)


# ── watermark persistence (aerys_v2: v2_email_watermark, migration 006) ──────
# Named psycopg params throughout — values are bound, never interpolated.

WATERMARK_GET_SQL = """\
SELECT uidvalidity, last_uid
FROM v2_email_watermark
WHERE mailbox = %(mailbox)s
"""

# Upsert: one row per mailbox. The stored values are the server's own integers,
# verbatim (module docstring — the re-serialization lesson).
WATERMARK_SET_SQL = """\
INSERT INTO v2_email_watermark (mailbox, uidvalidity, last_uid, updated_at)
VALUES (%(mailbox)s, %(uidvalidity)s, %(last_uid)s, now())
ON CONFLICT (mailbox) DO UPDATE SET
  uidvalidity = EXCLUDED.uidvalidity,
  last_uid    = EXCLUDED.last_uid,
  updated_at  = now()
"""


def read_db_watermark(conn: Any, mailbox: str) -> dict | None:
    """The stored {uidvalidity, last_uid} for a mailbox, or None on first run.
    `conn` is the brain's OWN aerys_v2 database — never prod aerys."""
    row = conn.execute(WATERMARK_GET_SQL, {"mailbox": mailbox}).fetchone()
    if row is None:
        return None
    return {"uidvalidity": int(row[0]), "last_uid": int(row[1])}


def save_db_watermark(conn: Any, mailbox: str, watermark: dict) -> None:
    """Persist the server's integers VERBATIM (see the module docstring)."""
    conn.execute(
        WATERMARK_SET_SQL,
        {
            "mailbox": mailbox,
            "uidvalidity": int(watermark["uidvalidity"]),
            "last_uid": int(watermark["last_uid"]),
        },
    )


# ── message parsing (pure — email stdlib only) ───────────────────────────────


def _one_line(text: object, *, limit: int) -> str:
    """Single-line, printable, length-capped — headers and body text both pass
    through here before entering a ping, so a crafted Subject can't smuggle
    newlines/control chars into the Discord message. Cousin of the gap miner's
    _bounded_excerpt, but newlines become SPACES (an email body is multi-line
    prose; deleting the separators would glue words together)."""
    cleaned = " ".join(str(text).split())  # all whitespace runs -> single space
    cleaned = "".join(c for c in cleaned if c.isprintable())
    if len(cleaned) > limit:
        cleaned = cleaned[: limit - 1].rstrip() + "…"
    return cleaned


def _plain_snippet(msg: Any, *, limit: int) -> str:
    """First ~limit chars of the first text/plain part, one line. Prefers
    get_body(preferencelist=('plain',)) — never the HTML alternative — and
    degrades to '' rather than raising: a snippet is garnish, the ping itself
    (sender + subject) must survive any body weirdness."""
    try:
        part = msg.get_body(preferencelist=("plain",))
        if part is None:
            return ""
        text = part.get_content()
    except Exception:
        try:
            payload = msg.get_payload(decode=True)
            text = payload.decode("utf-8", errors="replace") if payload else ""
        except Exception:
            return ""
    return _one_line(text, limit=limit)


def parse_message(uid: int, raw: bytes, *, snippet_chars: int = SNIPPET_CHARS) -> EmailSummary:
    """One fetched RFC822 blob -> EmailSummary. policy.default decodes RFC2047
    headers (=?utf-8?...?= subjects arrive human-readable). A message that
    defeats the parser still yields a STUB summary rather than vanishing —
    notification-only means "you have mail" matters more than parsing it."""
    try:
        msg = message_from_bytes(raw, policy=email_policy.default)
        name, addr = parseaddr(str(msg.get("From", "")))
        if name and addr:
            sender = f"{name} <{addr}>"
        else:
            sender = addr or name or "(unknown sender)"
        subject = _one_line(str(msg.get("Subject", "")), limit=SUBJECT_CHARS) or "(no subject)"
        date = _one_line(str(msg.get("Date", "")), limit=64)
        snippet = _plain_snippet(msg, limit=snippet_chars)
        return EmailSummary(
            uid=uid,
            sender=_one_line(sender, limit=SENDER_CHARS),
            subject=subject,
            snippet=snippet,
            date=date,
        )
    except Exception:
        log.warning("email watch: message uid %s failed to parse — pinging a stub",
                    uid, exc_info=True)
        return EmailSummary(uid=uid, sender="(unparseable sender)",
                            subject="(unparseable message)", snippet="", date="")


def format_ping(msg: EmailSummary) -> str:
    """One notification per email: sender, subject, snippet. Plain and
    informative — this is a system notification, not her voice (scope decision:
    the personality lives in the agent, not in the mail doorbell)."""
    lines = [f"New email from {msg.sender}", f"Subject: {msg.subject}"]
    if msg.snippet:
        lines.append(msg.snippet)
    return "\n".join(lines)


# ── the pure core: incremental UID fetch ─────────────────────────────────────


def _uidvalidity(client: Any) -> int:
    """The selected mailbox's UIDVALIDITY, via imaplib's response() accessor.
    Absent UIDVALIDITY = no safe way to run a UID watermark — refuse (Gmail
    always sends it; a server that doesn't is not one we can watch this way)."""
    _code, data = client.response("UIDVALIDITY")
    value = data[0] if data else None
    if value is None:
        raise EmailWatchError("server sent no UIDVALIDITY for the selected mailbox")
    return int(value.decode() if isinstance(value, (bytes, bytearray)) else value)


def _search_uids(client: Any, criteria: str) -> list[int]:
    """UID SEARCH -> sorted ints. The None charset arg is imaplib's own idiom
    (imaplib._command skips None args)."""
    typ, data = client.uid("SEARCH", None, criteria)
    if typ != "OK":
        raise EmailWatchError(f"UID SEARCH {criteria!r} failed: {typ}")
    if not data or not data[0]:
        return []
    return sorted(int(tok) for tok in data[0].split())


def _max_uid(client: Any) -> int:
    """Highest UID currently in the mailbox, 0 when empty — the fresh-start /
    reset mark."""
    uids = _search_uids(client, "ALL")
    return uids[-1] if uids else 0


def _fetch_raw(client: Any, uid: int) -> bytes:
    """UID FETCH BODY.PEEK[] -> the raw RFC822 bytes. PEEK is load-bearing:
    watching must never mark her mail \\Seen. imaplib returns a mixed list of
    (header, literal) tuples and bare closers — the literal is the payload."""
    typ, data = client.uid("FETCH", str(uid), "(BODY.PEEK[])")
    if typ != "OK":
        raise EmailWatchError(f"UID FETCH {uid} failed: {typ}")
    for item in data or []:
        if isinstance(item, tuple) and len(item) >= 2 and isinstance(item[1], (bytes, bytearray)):
            return bytes(item[1])
    raise EmailWatchError(f"UID FETCH {uid} returned no literal payload")


def check_inbox(
    imap_client: Any,
    watermark: dict | None,
    *,
    mailbox: str = DEFAULT_MAILBOX,
    snippet_chars: int = SNIPPET_CHARS,
) -> tuple[list[EmailSummary], dict]:
    """PURE core: (new_messages ascending by UID, new_watermark). Never touches
    a DB or notify path; raises EmailWatchError on protocol trouble.

    Three postures, per the module docstring:
      - watermark is None (first run): mark current max UID, return NO messages.
      - UIDVALIDITY changed (mailbox reindex): same — reset to current max,
        return NO messages. A reindex must not replay 500 pings.
      - normal: UID SEARCH UID last+1:* , re-filtered to uid > last_uid because
        a no-new-mail search still returns the highest-UID message (the N:*
        quirk) — without the filter the newest email re-pings every poll.
    """
    typ, _data = imap_client.select(mailbox, readonly=True)
    if typ != "OK":
        raise EmailWatchError(f"SELECT {mailbox!r} failed: {typ}")
    uidvalidity = _uidvalidity(imap_client)

    if watermark is None or int(watermark["uidvalidity"]) != uidvalidity:
        # Fresh start or reindex: adopt the server's CURRENT max UID, flood nothing.
        return [], {"uidvalidity": uidvalidity, "last_uid": _max_uid(imap_client)}

    last = int(watermark["last_uid"])
    uids = [u for u in _search_uids(imap_client, f"UID {last + 1}:*") if u > last]
    if not uids:
        return [], {"uidvalidity": uidvalidity, "last_uid": last}

    messages = [
        parse_message(u, _fetch_raw(imap_client, u), snippet_chars=snippet_chars)
        for u in uids  # already sorted ascending — ping in arrival order
    ]
    return messages, {"uidvalidity": uidvalidity, "last_uid": uids[-1]}


# ── the shell ────────────────────────────────────────────────────────────────


def imap_login_factory(
    host: str, user: str, password: str, *, port: int = 993
) -> Callable[[], Any]:
    """A ready-to-inject imap_factory for run_once. Config arrives as plain
    parameters (the integrator wires Settings — this module never imports
    config.py). imaplib is imported INSIDE the returned callable: no network
    and no protocol module at import time, so tests never come near a socket.
    Gmail in real life: host='imap.gmail.com', an app password (or OAuth later)."""

    def _connect() -> Any:
        import imaplib

        client = imaplib.IMAP4_SSL(host, port)
        client.login(user, password)
        return client

    return _connect


def _close_quietly(client: Any) -> None:
    """Best-effort logout — teardown trouble must never eat a pass's outcome."""
    if client is None:
        return
    try:
        client.logout()
    except Exception:
        log.debug("email watch: logout failed (ignored)", exc_info=True)


def run_once(
    conn: Any,
    imap_factory: Callable[[], Any],
    notify_fn: Callable[[str], None],
    mailbox: str = DEFAULT_MAILBOX,
) -> dict:
    """One watch pass: read watermark -> connect -> check -> ping -> advance.

    `conn` — the brain's OWN aerys_v2 database (v2_email_watermark only).
    NEVER prod aerys: this worker stores a cursor, not content. The conn should
    be AUTOCOMMIT: the per-message durability below assumes each save lands as
    it happens — under a deferred-commit conn, a mid-burst failure would roll
    back every earlier save in the pass and the whole burst re-pings next tick
    (still at-least-once, but the "only that one message" guarantee is gone).

    Failure posture (shell = the catching layer):
      - IMAP/DB trouble during the check: log, return {"ok": False, ...} — the
        scheduler retries next tick, nothing is lost (the watermark is durable).
      - notify failure mid-burst: stop pinging. Everything already pinged is
        behind the saved mark (no duplicates); the failed message and everything
        after it stay AHEAD of it and retry next tick (deferred, never lost).
        The watermark is saved PER MESSAGE for exactly this reason.
      - watermark save failure after a ping went out: that message may ping
        again next tick — at-least-once by choice, a duplicate beats a loss.

    Returns a stats dict for logs: {ok, new, pinged, first_run, reset,
    watermark, error}.
    """
    stats: dict[str, Any] = {
        "ok": True, "new": 0, "pinged": 0,
        "first_run": False, "reset": False,
        "watermark": None, "error": None,
    }
    client = None
    try:
        stored = read_db_watermark(conn, mailbox)
        client = imap_factory()
        try:
            new_messages, new_wm = check_inbox(client, stored, mailbox=mailbox)
        finally:
            _close_quietly(client)
    except Exception as e:
        log.warning("email watch: pass failed cleanly (%s) — scheduler retries next tick",
                    e, exc_info=True)
        _close_quietly(client)
        stats["ok"] = False
        stats["error"] = str(e)
        return stats

    stats["new"] = len(new_messages)
    stats["first_run"] = stored is None
    stats["reset"] = stored is not None and int(stored["uidvalidity"]) != int(new_wm["uidvalidity"])
    stats["watermark"] = stored

    if not new_messages:
        # First run / reindex reset / nothing new. Persist only when the mark
        # actually moved — a quiet poll writes nothing.
        if new_wm != stored:
            try:
                save_db_watermark(conn, mailbox, new_wm)
                stats["watermark"] = new_wm
            except Exception as e:
                log.warning("email watch: watermark save failed (%s) — retrying next tick",
                            e, exc_info=True)
                stats["ok"] = False
                stats["error"] = str(e)
        return stats

    # Downtime backlog collapse: first-run and reindex resets already ping
    # nothing, but a VALID watermark after days of worker downtime would fire
    # one ping per accumulated message. Past the threshold, ONE summary ping
    # covers the whole backlog (naming the newest few so it's useful, not just
    # a count) and the mark advances past all of it in a single save.
    if len(new_messages) > BACKLOG_COLLAPSE_AT:
        lines = [f"• {m.sender} — {m.subject}" for m in reversed(new_messages[-3:])]
        summary = (
            f"{len(new_messages)} new emails arrived while I wasn't watching. "
            "Most recent:\n" + "\n".join(lines)
        )
        top = {"uidvalidity": int(new_wm["uidvalidity"]), "last_uid": new_messages[-1].uid}
        try:
            notify_fn(summary)
        except Exception as e:
            log.warning("email watch: backlog summary notify failed (%s) — retrying "
                        "next tick, watermark held", e, exc_info=True)
            stats["ok"] = False
            stats["error"] = f"backlog summary notify failed: {e}"
            return stats
        stats["pinged"] = 1
        try:
            save_db_watermark(conn, mailbox, top)
            stats["watermark"] = top
        except Exception as e:
            log.warning("email watch: watermark save failed after backlog summary (%s) — "
                        "the summary may repeat next tick", e, exc_info=True)
            stats["ok"] = False
            stats["error"] = f"watermark save failed after backlog summary: {e}"
        return stats

    # Normal path only (messages flow ONLY when uidvalidity matched the stored
    # mark, so `stored` is a dict here). Ping ascending, advance per message.
    for msg in new_messages:
        try:
            notify_fn(format_ping(msg))
        except Exception as e:
            remaining = stats["new"] - stats["pinged"]
            log.warning(
                "email watch: notify failed at uid %s (%s) — watermark held below it; "
                "%d ping(s) deferred to next tick", msg.uid, e, remaining, exc_info=True,
            )
            stats["ok"] = False
            stats["error"] = f"notify failed at uid {msg.uid}: {e}"
            break
        advanced = {"uidvalidity": int(new_wm["uidvalidity"]), "last_uid": msg.uid}
        stats["pinged"] += 1
        try:
            save_db_watermark(conn, mailbox, advanced)
        except Exception as e:
            # The ping went OUT but the mark didn't move: this message may ping
            # again next tick (at-least-once — documented above). Stop here; a
            # broken conn would fail every subsequent save anyway.
            log.warning("email watch: watermark save failed after pinging uid %s (%s) — "
                        "it may ping again next tick", msg.uid, e, exc_info=True)
            stats["ok"] = False
            stats["error"] = f"watermark save failed at uid {msg.uid}: {e}"
            break
        stats["watermark"] = advanced

    return stats
