"""The owner's approval button (board #176, Stage 4's shared piece).

Some things she may only do once he taps Approve on a Discord button. The owner's rulings:
  D10 (2026-11-08): "Voice can start anything text can but requires me to approve via discord
      button. Else read only jobs".
  D12 (2026-11-09): she may message people he's told her about, and the first message to each
      waits for his button, which shows exactly who and where ("Sam · Discord: sam_42"), so his
      tap confirms the person too.
This module is only the button. The features that need it ask through Approvals.request() and
read the answer back; the Discord gateway (the soak container) posts the button to his DMs and
records his tap. He doesn't get Discord notifications (11/08: "i dont get discord
notifications"), so every request also goes as an HA notification saying a button is waiting,
retried until it lands; without that leg the button doesn't arm at all.

Fail closed, everywhere: anything but an explicit Approve from HIS Discord account (looked up
fresh at every tap, never remembered) before the request expires is not an approval. A click from
anyone else changes nothing; a second click changes nothing; a click after the expiry changes
nothing; a database that can't be read means "not approved". The decision is one compare-and-set
UPDATE, so two clicks can never both win. The DM is claimed before it's sent, so it's never sent
twice, and whatever the DM shows is corrected until it matches the record.

What he's approving (summary, detail) is the private part: never logged.
"""

from __future__ import annotations

import logging
import re
import time as _time
import uuid
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from typing import Callable

log = logging.getLogger(__name__)

STATUSES = ("pending", "approved", "declined", "expired", "withdrawn")
DEFAULT_FOR = timedelta(minutes=30)
LONGEST_FOR = timedelta(days=7)
SUMMARY_CHARS = 300
DETAIL_CHARS = 1200
POLL_S = 5
PING_RETRY_AFTER = timedelta(minutes=1)   # a request whose HA ping didn't land is pinged again
SHOW_FOR = timedelta(days=7)              # a decided DM that won't take its edit stops being retried
CLAIM = "posting:"
CUSTOM_ID = re.compile(r"aerys-approval:(?P<id>[0-9]+):(?P<verdict>approve|decline)")


@dataclass
class Approval:
    id: int
    kind: str
    summary: str
    detail: str
    requested_at: datetime
    expires_at: datetime
    status: str = "pending"
    decided_at: datetime | None = None
    decided_by: str | None = None
    request_key: str = ""
    message_id: str | None = None
    channel_id: str | None = None
    shown: str | None = None
    pinged_at: datetime | None = None
    approvers: list[str] | None = None
    recipient: str | None = None


def custom_id(approval_id: int, verdict: str) -> str:
    return f"aerys-approval:{int(approval_id)}:{verdict}"


# --- the record ---------------------------------------------------------------------------


class MemoryApprovalStore:
    """In-process store (tests, dev boxes). Same semantics as the Postgres one."""

    def __init__(self):
        self.rows: dict[int, Approval] = {}
        self._next = 1
        self.posts = {}

    def request(self, kind, summary, detail, now, expires_at, key, approvers=None):
        for r in self.rows.values():
            if r.request_key == key:
                return r, False
        a = Approval(self._next, kind, summary, detail, now, expires_at, request_key=key, approvers=approvers)
        self.rows[a.id] = a
        self._next += 1
        return a, True

    def get(self, approval_id):
        return self.rows.get(approval_id)

    def pinged(self, approval_id, now):
        self.rows[approval_id].pinged_at = now

    def unpinged(self, before, now):
        return sorted((r for r in self.rows.values() if r.status == "pending" and r.pinged_at is None
                       and r.requested_at <= before and r.expires_at > now), key=lambda r: r.id)

    def unposted(self, now):
        return sorted((r for r in self.rows.values() if r.status == "pending" and r.approvers is None and r.message_id is None
                       and r.expires_at > now), key=lambda r: r.id)

    def claim(self, approval_id, token):
        r = self.rows.get(approval_id)
        if r is None or r.message_id is not None:
            return False
        r.message_id = token
        return True

    def posted(self, approval_id, token, channel_id, message_id):
        r = self.rows.get(approval_id)
        if r is None or r.message_id != token:
            return False
        r.message_id, r.channel_id = str(message_id), str(channel_id)
        return True

    def release(self, approval_id, token):
        r = self.rows.get(approval_id)
        if r is not None and r.message_id == token:
            r.message_id = None

    def decide(self, approval_id, approve, by, now):
        r = self.rows.get(approval_id)
        if r is None or r.status != "pending" or r.expires_at <= now:
            return None
        r.status, r.decided_at, r.decided_by = ("approved" if approve else "declined"), now, str(by)
        return r

    def withdraw(self, approval_id, now):
        r = self.rows.get(approval_id)
        if r is None or r.status != "pending":
            return False
        r.status, r.decided_at = "withdrawn", now
        return True

    def expire(self, now):
        out = []
        for r in sorted(self.rows.values(), key=lambda r: r.id):
            if r.status == "pending" and r.expires_at <= now:
                r.status, r.decided_at = "expired", now
                out.append(r)
        return out

    def unshown(self, since):
        return sorted((r for r in self.rows.values() if r.status != "pending" and r.message_id and r.channel_id
                       and not r.message_id.startswith(CLAIM) and r.shown != r.status
                       and r.decided_at is not None and r.decided_at >= since), key=lambda r: r.id)

    def set_shown(self, approval_id, status):
        self.rows[approval_id].shown = status

    def post_rows(self, now, since):
        out = []
        for a in self.rows.values():
            for recipient in a.approvers or []:
                post = self.posts.get((a.id, recipient), {})
                if (a.status == "pending" and a.expires_at > now and not post.get("message_id")) or (
                    a.status != "pending" and a.decided_at >= since and post.get("channel_id")
                    and post.get("shown") != a.status):
                    out.append(replace(a, recipient=recipient, message_id=post.get("message_id"),
                                       channel_id=post.get("channel_id")))
        return out

    def claim_post(self, aid, recipient, token):
        key = (aid, recipient)
        if key in self.posts:
            return False
        self.posts[key] = {"message_id": token}
        return True

    def posted_post(self, aid, recipient, token, channel, message):
        p = self.posts.get((aid, recipient), {})
        if p.get("message_id") != token:
            return False
        p.update(channel_id=str(channel), message_id=str(message))
        return True

    def release_post(self, aid, recipient, token):
        if self.posts.get((aid, recipient), {}).get("message_id") == token:
            del self.posts[aid, recipient]

    def shown_post(self, aid, recipient, status):
        self.posts[aid, recipient]["shown"] = status


class PostgresApprovalStore:
    """v2_approvals (db/migrations/022). Fresh short connections per call."""

    COLUMNS = ("id, kind, summary, detail, requested_at, expires_at, status, decided_at, decided_by, "
               "request_key, message_id, channel_id, shown, pinged_at, approvers")

    def __init__(self, database_url: str):
        self.database_url = database_url

    def _connect(self):
        import psycopg

        return psycopg.connect(self.database_url, connect_timeout=5, options="-c statement_timeout=5000")

    @staticmethod
    def _row(r) -> Approval:
        return Approval(*r)

    def _one(self, sql, args):
        with self._connect() as conn:
            return conn.execute(sql, args).fetchone()

    def _all(self, sql, args):
        with self._connect() as conn:
            return [self._row(r) for r in conn.execute(sql, args).fetchall()]

    def request(self, kind, summary, detail, now, expires_at, key, approvers=None):
        with self._connect() as conn:
            row = conn.execute(
                "INSERT INTO v2_approvals (kind, summary, detail, requested_at, expires_at, request_key, approvers) "
                f"VALUES (%s, %s, %s, %s, %s, %s, %s) ON CONFLICT (request_key) DO NOTHING RETURNING {self.COLUMNS}",
                (kind, summary, detail, now, expires_at, key, approvers)).fetchone()
            if row is not None:
                return self._row(row), True
            row = conn.execute(f"SELECT {self.COLUMNS} FROM v2_approvals WHERE request_key = %s", (key,)).fetchone()
        return self._row(row), False

    def get(self, approval_id):
        row = self._one(f"SELECT {self.COLUMNS} FROM v2_approvals WHERE id = %s", (approval_id,))
        return self._row(row) if row else None

    def pinged(self, approval_id, now):
        self._one("UPDATE v2_approvals SET pinged_at = %s WHERE id = %s RETURNING id", (now, approval_id))

    def unpinged(self, before, now):
        return self._all(f"SELECT {self.COLUMNS} FROM v2_approvals WHERE status = 'pending' AND pinged_at IS NULL "
                         "AND requested_at <= %s AND expires_at > %s ORDER BY id", (before, now))

    def unposted(self, now):
        return self._all(f"SELECT {self.COLUMNS} FROM v2_approvals WHERE status = 'pending' "
                         "AND approvers IS NULL AND message_id IS NULL AND expires_at > %s ORDER BY id", (now,))

    def claim(self, approval_id, token):
        """Before the DM goes: one gateway, one DM, ever (a claim left by a failed record stays)."""
        return self._one("UPDATE v2_approvals SET message_id = %s WHERE id = %s AND message_id IS NULL "
                         "RETURNING id", (token, approval_id)) is not None

    def posted(self, approval_id, token, channel_id, message_id):
        return self._one("UPDATE v2_approvals SET message_id = %s, channel_id = %s "
                         "WHERE id = %s AND message_id = %s RETURNING id",
                         (str(message_id), str(channel_id), approval_id, token)) is not None

    def release(self, approval_id, token):
        self._one("UPDATE v2_approvals SET message_id = NULL WHERE id = %s AND message_id = %s RETURNING id",
                  (approval_id, token))

    def decide(self, approval_id, approve, by, now):
        """One compare-and-set: only a pending, unexpired request is decided, once."""
        row = self._one(
            "UPDATE v2_approvals SET status = %s, decided_at = %s, decided_by = %s "
            f"WHERE id = %s AND status = 'pending' AND expires_at > %s RETURNING {self.COLUMNS}",
            ("approved" if approve else "declined", now, str(by), approval_id, now))
        return self._row(row) if row else None

    def withdraw(self, approval_id, now):
        return self._one("UPDATE v2_approvals SET status = 'withdrawn', decided_at = %s "
                         "WHERE id = %s AND status = 'pending' RETURNING id", (now, approval_id)) is not None

    def expire(self, now):
        return self._all("UPDATE v2_approvals SET status = 'expired', decided_at = %s "
                         f"WHERE status = 'pending' AND expires_at <= %s RETURNING {self.COLUMNS}", (now, now))

    def unshown(self, since):
        return self._all(f"SELECT {self.COLUMNS} FROM v2_approvals WHERE status <> 'pending' "
                         "AND message_id IS NOT NULL AND message_id NOT LIKE 'posting:%%' AND channel_id IS NOT NULL "
                         "AND shown IS DISTINCT FROM status AND decided_at >= %s ORDER BY id", (since,))

    def set_shown(self, approval_id, status):
        self._one("UPDATE v2_approvals SET shown = %s WHERE id = %s RETURNING id", (status, approval_id))

    def post_rows(self, now, since):
        columns = ", ".join("a." + c.strip() for c in self.COLUMNS.split(","))
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT {columns}, recipient, p.channel_id, p.message_id FROM v2_approvals a "
                "CROSS JOIN LATERAL unnest(a.approvers) AS recipient "
                "LEFT JOIN v2_approval_posts p ON p.approval_id = a.id AND p.approver = recipient "
                "WHERE (a.status = 'pending' AND a.expires_at > %s AND p.message_id IS NULL) "
                "OR (a.status <> 'pending' AND a.decided_at >= %s AND p.channel_id IS NOT NULL "
                "AND p.shown IS DISTINCT FROM a.status) ORDER BY a.id, recipient", (now, since)).fetchall()
        return [replace(self._row(r[:-3]), recipient=r[-3], channel_id=r[-2], message_id=r[-1]) for r in rows]

    def claim_post(self, aid, recipient, token):
        return self._one("INSERT INTO v2_approval_posts (approval_id, approver, message_id) VALUES (%s, %s, %s) "
                         "ON CONFLICT DO NOTHING RETURNING approval_id", (aid, recipient, token)) is not None

    def posted_post(self, aid, recipient, token, channel, message):
        return self._one("UPDATE v2_approval_posts SET channel_id = %s, message_id = %s "
                         "WHERE approval_id = %s AND approver = %s AND message_id = %s RETURNING approval_id",
                         (str(channel), str(message), aid, recipient, token)) is not None

    def release_post(self, aid, recipient, token):
        self._one("DELETE FROM v2_approval_posts WHERE approval_id = %s AND approver = %s "
                  "AND message_id = %s RETURNING approval_id", (aid, recipient, token))

    def shown_post(self, aid, recipient, status):
        self._one("UPDATE v2_approval_posts SET shown = %s WHERE approval_id = %s AND approver = %s "
                  "RETURNING approval_id", (status, aid, recipient))


# --- the asking side (any process that writes rows) ---------------------------------------


class Approvals:
    """request() -> an id; approved()/status()/wait() read his answer, failing closed."""

    def __init__(self, store, *, notify: Callable[[str, str], None] | None = None,
                 clock: Callable[[], datetime] | None = None, sleep: Callable[[float], None] = _time.sleep):
        self.store, self.notify = store, notify
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        self.sleep = sleep

    def request(self, kind: str, summary: str, detail: str = "", expires_in: timedelta = DEFAULT_FOR,
                key: str | None = None, approvers: list[str] | None = None) -> int:
        """Ask him. The same key twice is one request. Raises if the request can't be written: a
        caller that can't ask can't go ahead. The HA ping is tried now and retried by the gateway."""
        kind = " ".join(str(kind or "").split())[:60]
        summary = " ".join(str(summary or "").split())[:SUMMARY_CHARS]
        if not kind or not summary:
            raise ValueError("an approval needs a kind and a summary")
        if expires_in <= timedelta(0) or expires_in > LONGEST_FOR:
            raise ValueError("an approval lasts more than nothing and at most a week")
        if approvers is not None:
            approvers = sorted(set(str(x) for x in approvers))
            if not approvers or any(not x.isdigit() for x in approvers):
                raise ValueError("approvers must be nonempty Discord ids")
        now = self.clock()
        row, created = self.store.request(kind, summary, str(detail or "")[:DETAIL_CHARS], now, now + expires_in,
                                          key or f"{kind}:{uuid.uuid4().hex}", approvers=approvers)
        if created:
            ping(self.store, self.notify, row, now)
        log.info("approval %s (%s): %s", row.id, kind, "requested" if created else "already asked")
        return row.id

    def status(self, approval_id: int) -> str:
        """pending | approved | declined | expired | withdrawn, or "unknown" when the record can't
        be read (which is never an approval)."""
        try:
            row = self.store.get(approval_id)
        except Exception:
            log.warning("approval %s: couldn't read it; treating it as not approved", approval_id, exc_info=True)
            return "unknown"
        if row is None:
            return "unknown"
        if row.status == "pending" and row.expires_at <= self.clock():
            return "expired"
        return row.status

    def approved(self, approval_id: int) -> bool:
        return self.status(approval_id) == "approved"

    def wait(self, approval_id: int, timeout_s: float, poll_s: float = 2.0) -> str:
        """Block until he answers, it expires, or timeout_s passes; returns the status then."""
        deadline = _time.monotonic() + timeout_s
        while True:
            state = self.status(approval_id)
            if state not in ("pending", "unknown") or _time.monotonic() >= deadline:
                return state
            self.sleep(poll_s)

    def withdraw(self, approval_id: int) -> bool:
        try:
            return self.store.withdraw(approval_id, self.clock())
        except Exception:
            log.warning("approval %s: couldn't withdraw it", approval_id, exc_info=True)
            return False


def ping(store, notify, row: Approval, now: datetime) -> bool:
    """The HA notification that a button is waiting; recorded only when it lands (else retried)."""
    if notify is None:
        return False
    try:
        notify("Aerys needs your OK", f"{row.summary} (tap Approve or Decline in your Discord DMs)")
    except Exception as e:
        log.warning("approval %s: the notification didn't go (%s); retried", row.id, type(e).__name__)
        return False
    try:
        store.pinged(row.id, now)
    except Exception:
        log.warning("approval %s: pinged, but recording it failed (it may ping again)", row.id)
    return True


def notify_for(settings) -> Callable[[str, str], None] | None:
    """The HA notification leg (his phone, watch, glasses): reach_notify_service, else
    storm_notify_service; None when neither is set or HA isn't armed."""
    service = (str(getattr(settings, "reach_notify_service", "") or "").strip()
               or str(getattr(settings, "storm_notify_service", "") or "").strip())
    if not service or getattr(settings, "ha_token", None) is None:
        return None

    def notify(title: str, message: str) -> None:
        import httpx

        r = httpx.post(f"{str(settings.ha_base_url).rstrip('/')}/api/services/notify/{service}",
                       json={"title": title, "message": message},
                       headers={"Authorization": f"Bearer {settings.ha_token.get_secret_value()}"}, timeout=10.0)
        r.raise_for_status()

    return notify


def owner_id_for(settings, *, lookup=None) -> Callable[[], int | None]:
    """The owner's Discord user id, looked up FRESH on every call (a corrected or removed identity
    takes effect at the next tap): approvals_owner_discord_id when set, else his linked Discord
    identity (prod platform_identities). No answer means nobody can approve."""
    fixed = str(getattr(settings, "approvals_owner_discord_id", "") or "").strip()

    def owner() -> int | None:
        if fixed:
            return int(fixed) if fixed.isdigit() else None
        try:
            if lookup is not None:
                raw = lookup(settings)
            else:
                from aerys_v2.kael_line import _owner_platform_id

                raw = _owner_platform_id(settings, "discord")
        except Exception:
            log.warning("approvals: the owner lookup failed; nobody can approve this time", exc_info=True)
            return None
        raw = str(raw or "").strip()
        return int(raw) if raw.isdigit() else None

    return owner


def approvals_for(settings) -> Approvals | None:
    """None unless armed: approvals on, a database, an owner, and the HA notification leg (he doesn't
    get Discord notifications, so a button without a ping would just sit there)."""
    if not getattr(settings, "approvals", False) or not getattr(settings, "database_url", None) \
            or getattr(settings, "owner_person_id", None) is None:
        return None
    notify = notify_for(settings)
    if notify is None:
        log.warning("approvals: no HA notify service (REACH_NOTIFY_SERVICE / STORM_NOTIFY_SERVICE); not armed")
        return None
    return Approvals(PostgresApprovalStore(settings.database_url), notify=notify)


# --- the button (the Discord gateway: the soak container) ---------------------------------


def message_text(a: Approval) -> str:
    when = int(a.expires_at.timestamp())
    body = f"**Aerys needs your OK** ({a.kind})\n{a.summary}"
    if a.detail:
        body += f"\n\n{a.detail}"
    return (body + f"\n\nAnswer by <t:{when}:t> (<t:{when}:R>); no answer is a no.")[:1990]


def outcome_text(a: Approval) -> str:
    said = {"approved": "Approved", "declined": "Declined", "expired": "Expired, nothing done",
            "withdrawn": "Withdrawn, no longer needed"}.get(a.status, a.status)
    return (f"**{said}** ({a.kind})\n{a.summary}")[:1990]


def handle_click(store, approval_id: int, approve: bool, clicker_id: int, owner_id: int | None,
                 now: datetime) -> tuple[str, Approval | None]:
    """The tap, decided. Returns (what to tell the clicker, the decided row or None). Only the
    owner decides; only a pending, unexpired request is decided; and only once."""
    try:
        current = store.get(approval_id)
        allowed = current.approvers if current and current.approvers is not None else [str(owner_id)]
        if str(clicker_id) not in allowed:
            return "Not yours to answer.", None
        row = store.decide(approval_id, approve, clicker_id, now)
    except Exception:
        log.warning("approval %s: couldn't record the answer", approval_id, exc_info=True)
        return "I couldn't record that just now. Nothing was approved; tap again in a moment.", None
    if row is None:
        return "That one was already answered or has expired. Nothing changed.", None
    return ("Approved." if approve else "Declined."), row


async def on_tap(interaction, approval_id: int, approve: bool, *, store, owner_id: Callable[[], int | None],
                 clock: Callable[[], datetime] | None = None) -> None:
    """A button tap: acknowledged FIRST (Discord allows three seconds), then decided off the event
    loop, then the message edited. A failed edit is fixed by the poller (the record says what the
    message should show)."""
    import asyncio

    await interaction.response.defer()
    oid = await asyncio.to_thread(owner_id)
    now = (clock or (lambda: datetime.now(timezone.utc)))()
    reply, row = await asyncio.to_thread(handle_click, store, approval_id, approve, interaction.user.id, oid, now)
    if row is None:
        await interaction.followup.send(reply, ephemeral=True)
        return
    try:
        await interaction.edit_original_response(content=outcome_text(row), view=None)
        if row.approvers is None:
            await asyncio.to_thread(store.set_shown, row.id, row.status)
    except Exception as e:
        log.warning("approval %s: decided, but the message edit failed (%s); the poller retries",
                    row.id, type(e).__name__)


async def poll_once(store, *, send: Callable, edit: Callable, notify=None, now: datetime,
                    unrecorded: dict | None = None) -> dict:
    """One pass of the gateway's poller:
      1. record any DM that went out last time but whose record failed (kept in unrecorded);
      2. close what expired;
      3. make every decided DM show its outcome (retried until it does, for a week);
      4. ping again for any request whose HA notification didn't land;
      5. post a button for each new request, claimed first so it's never sent twice.
    send(approval) -> (channel id, message id); edit(approval, text) edits the DM it was posted in."""
    import asyncio

    out = {"posted": 0, "expired": 0, "shown": 0, "pinged": 0}
    for aid, (token, channel_id, message_id) in list((unrecorded or {}).items()):
        if isinstance(aid, tuple):
            continue
        try:
            await asyncio.to_thread(store.posted, aid, token, channel_id, message_id)
        except Exception:
            continue
        unrecorded.pop(aid)
    out["expired"] = len(await asyncio.to_thread(store.expire, now))
    for a in await asyncio.to_thread(store.unshown, now - SHOW_FOR):
        try:
            await edit(a, outcome_text(a))
        except Exception as e:
            log.warning("approval %s: its message won't take the outcome yet (%s)", a.id, type(e).__name__)
            continue
        await asyncio.to_thread(store.set_shown, a.id, a.status)
        out["shown"] += 1
    if notify is not None:
        for a in await asyncio.to_thread(store.unpinged, now - PING_RETRY_AFTER, now):
            if await asyncio.to_thread(ping, store, notify, a, now):
                out["pinged"] += 1
    for a in await asyncio.to_thread(store.unposted, now):
        token = f"{CLAIM}{uuid.uuid4().hex}"
        if not await asyncio.to_thread(store.claim, a.id, token):
            continue                                      # another pass or gateway has it
        try:
            channel_id, message_id = await send(a)
        except Exception as e:
            log.warning("approval %s: couldn't post the button (%s); tried again next pass", a.id, type(e).__name__)
            await asyncio.to_thread(store.release, a.id, token)
            continue
        try:
            await asyncio.to_thread(store.posted, a.id, token, channel_id, message_id)
        except Exception:
            # Sent: the button works regardless (its id rides in the button); the claim stays so
            # it's never sent again, and the record is retried next pass (until a restart).
            log.warning("approval %s: posted, but recording the message failed; retried", a.id)
            if unrecorded is not None:
                unrecorded[a.id] = (token, channel_id, message_id)
        out["posted"] += 1
    extra = await _poll_posts(store, send=send, edit=edit, now=now, unrecorded=unrecorded)
    for key, value in extra.items():
        out[key] += value
    return out


async def edit_dm(client, a: Approval, text: str) -> None:
    """Edit the DM a button was posted in, by the channel it went to (not whoever the owner is now),
    without fetching it first."""
    import discord

    channel = client.get_partial_messageable(int(a.channel_id), type=discord.ChannelType.private)
    await channel.get_partial_message(int(a.message_id)).edit(content=text, view=None)


def attach_approvals(client, *, store, owner_id: Callable[[], int | None], notify=None, poll_s: float = POLL_S):
    """Wire the button into the gateway: a restart-proof button class (custom_ids carry the
    approval id, registered with add_dynamic_items on every start), and the poller."""
    import asyncio

    import discord

    class ApprovalButton(discord.ui.DynamicItem[discord.ui.Button], template=CUSTOM_ID):
        def __init__(self, approval_id: int, verdict: str):
            super().__init__(discord.ui.Button(
                label="Approve" if verdict == "approve" else "Decline",
                style=discord.ButtonStyle.success if verdict == "approve" else discord.ButtonStyle.danger,
                custom_id=custom_id(approval_id, verdict)))
            self.approval_id, self.verdict = approval_id, verdict

        @classmethod
        async def from_custom_id(cls, interaction, item, match):  # pragma: no cover - live only
            return cls(int(match["id"]), match["verdict"])

        async def callback(self, interaction):  # pragma: no cover - live only (on_tap is tested)
            await on_tap(interaction, self.approval_id, self.verdict == "approve", store=store, owner_id=owner_id)

    def buttons(a: Approval):
        view = discord.ui.View(timeout=None)
        view.add_item(ApprovalButton(a.id, "approve"))
        view.add_item(ApprovalButton(a.id, "decline"))
        return view

    async def owner_user():  # pragma: no cover - live only
        oid = await asyncio.to_thread(owner_id)
        if oid is None:
            raise RuntimeError("no owner Discord account")
        return client.get_user(int(oid)) or await client.fetch_user(int(oid))

    async def send(a: Approval) -> tuple[str, str]:  # pragma: no cover - live only
        user = (client.get_user(int(a.recipient)) or await client.fetch_user(int(a.recipient))) if a.recipient else await owner_user()
        message = await user.send(message_text(a), view=buttons(a))
        return str(message.channel.id), str(message.id)

    async def edit(a: Approval, text: str) -> None:  # pragma: no cover - live only (edit_dm is tested)
        await edit_dm(client, a, text)

    unrecorded: dict = {}                        # DMs sent whose record failed, retried each pass

    async def poller() -> None:  # pragma: no cover - thin loop over poll_once
        await client.wait_until_ready()
        log.info("approval buttons armed | every %ss", poll_s)
        while not client.is_closed():
            try:
                await poll_once(store, send=send, edit=edit, notify=notify, now=datetime.now(timezone.utc),
                                unrecorded=unrecorded)
            except Exception:
                log.warning("approvals poll failed", exc_info=True)
            await asyncio.sleep(poll_s)

    original_setup_hook = client.setup_hook

    async def setup_hook() -> None:
        await original_setup_hook()
        client.add_dynamic_items(ApprovalButton)     # every start: old buttons keep working
        client.loop.create_task(poller())

    client.setup_hook = setup_hook  # type: ignore[method-assign]
    return ApprovalButton


async def _poll_posts(store, *, send, edit, now, unrecorded=None):
    """Per-recipient claims and outcome retries for explicit approver lists."""
    import asyncio

    out = {"posted": 0, "shown": 0}
    for key, values in list((unrecorded or {}).items()):
        if not isinstance(key, tuple):
            continue
        try:
            await asyncio.to_thread(store.posted_post, *key, *values)
        except Exception:
            continue
        unrecorded.pop(key)
    for a in await asyncio.to_thread(store.post_rows, now, now - SHOW_FOR):
        if a.status != "pending":
            try:
                await edit(a, outcome_text(a))
                await asyncio.to_thread(store.shown_post, a.id, a.recipient, a.status)
                out["shown"] += 1
            except Exception:
                log.warning("approval %s: outcome edit will retry", a.id)
            continue
        token = f"{CLAIM}{uuid.uuid4().hex}"
        if not await asyncio.to_thread(store.claim_post, a.id, a.recipient, token):
            continue
        try:
            channel, message = await send(a)
        except Exception:
            await asyncio.to_thread(store.release_post, a.id, a.recipient, token)
            continue
        try:
            await asyncio.to_thread(store.posted_post, a.id, a.recipient, token, channel, message)
        except Exception:
            if unrecorded is not None:
                unrecorded[a.id, a.recipient] = (token, channel, message)
        out["posted"] += 1
    return out


def person_id_for(settings, person_id, *, lookup=None) -> int | None:
    """Resolve a person's Discord id from env overrides or linked identities."""
    from aerys_v2.config import person_mapping
    from aerys_v2.kael_line import _person_platform_id

    if str(person_id) == str(getattr(settings, "owner_person_id", "")):
        return owner_id_for(settings)()
    fixed = person_mapping(getattr(settings, "approvals_person_discord_ids", ""))
    try:
        raw = fixed.get(str(person_id))
        if raw is None:
            raw = (lookup or _person_platform_id)(settings, "discord", person_id)
        raw = str(raw or "").strip()
        return int(raw) if raw.isdigit() else None
    except Exception:
        log.warning("approvals: person lookup failed")
        return None
