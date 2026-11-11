"""She hands him a file her home job made (#183, the owner's ruling 11/10).

Her home body (her own VM) writes files in her workspace there. On 11/10 a how-to her home job
wrote for him never reached him: she couldn't attach a file in his DM. send_file_to_chris reads
the file from her home server (GET {home_url}/files/{path}, the home server's own token) and sends
it to him alone: a Discord attachment in his DM, or a Telegram document when he asked on Telegram.

It only ever goes to him, so it isn't messaging other people (the owner's 11/08 line), and it needs
no tap. Honest by construction: SENT only after Discord or Telegram took the file; anything else is
a plain NOT SENT that says why. Only he is offered the tool (send_file_middleware), and a call from
anyone else is refused anyway. The file's contents are never logged.
"""

from __future__ import annotations

import json
import logging
import posixpath
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Callable
from urllib.parse import quote
from zoneinfo import ZoneInfo

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

from aerys_v2.state import identity_from_config

log = logging.getLogger(__name__)

TOOL_NAME = "send_file_to_chris"
SENT_TOOL_NAME = "files_sent_to_chris"
#: The same file to him again within this window needs him to ask again (#187, 11/10).
RESEND_HOURS = 12
EASTERN = ZoneInfo("America/New_York")
#: The home server refuses more (413); a bot's Discord upload limit is above it.
MAX_BYTES = 8 * 1024 * 1024
#: Her workspace on her home machine: absolute paths inside it are sent as relative ones.
HOME_WORKSPACE = "/home/hands/work"
NOTE_CHARS = 900
NAME_CHARS = 120
DISCORD_API = "https://discord.com/api/v10"
USER_AGENT = "DiscordBot (aerys-files, 1.0)"   # the default UA gets 403 from Discord (kael_line)

Deliver = Callable[[str, str, bytes, str], None]


def _relative(path) -> str:
    """The path as her home server takes it: relative to her workspace."""
    p = str(path or "").strip()
    for prefix in (HOME_WORKSPACE + "/", "~/work/"):
        if p.startswith(prefix):
            p = p[len(prefix):]
    return p.lstrip("/")


def _name(header, rel: str) -> str:
    """A file name, only ever a name: whatever the header says, no directories."""
    name = posixpath.basename(str(header or "").replace("\\", "/").strip()) or posixpath.basename(rel)
    return (name or "file")[:NAME_CHARS]


@dataclass(frozen=True)
class FileSend:
    sent_at: datetime
    path: str
    name: str
    bytes: int
    surface: str


class FileSends:
    """The record of files she handed him (v2_file_sends, migration 026). In memory for tests."""

    def __init__(self):
        self.rows: list[tuple[str, FileSend]] = []

    def record(self, person_id: str, send: FileSend, thread_id: str = "") -> None:
        self.rows.append((str(person_id), send))

    def last(self, person_id: str, path: str, since: datetime) -> FileSend | None:
        hits = [s for p, s in self.rows if p == str(person_id) and s.path == path and s.sent_at >= since]
        return max(hits, key=lambda s: s.sent_at) if hits else None

    def recent(self, person_id: str, since: datetime) -> list[FileSend]:
        return sorted((s for p, s in self.rows if p == str(person_id) and s.sent_at >= since),
                      key=lambda s: s.sent_at, reverse=True)


class PostgresFileSends(FileSends):
    """v2_file_sends in the brain's own database: a short connection per call, like the other ledgers."""

    def __init__(self, database_url: str):
        self.database_url = database_url

    def _connect(self):
        import psycopg

        return psycopg.connect(self.database_url, connect_timeout=5, options="-c statement_timeout=5000")

    def record(self, person_id, send, thread_id=""):
        with self._connect() as c:
            c.execute("INSERT INTO v2_file_sends (sent_at, person_id, path, name, bytes, surface, thread_id) "
                      "VALUES (%s, %s, %s, %s, %s, %s, %s)",
                      (send.sent_at, str(person_id), send.path, send.name, send.bytes, send.surface, thread_id or ""))

    def _rows(self, sql, args):
        with self._connect() as c:
            return [FileSend(*row) for row in c.execute(sql, args).fetchall()]

    def last(self, person_id, path, since):
        rows = self._rows("SELECT sent_at, path, name, bytes, surface FROM v2_file_sends WHERE person_id = %s "
                          "AND path = %s AND sent_at >= %s ORDER BY sent_at DESC LIMIT 1",
                          (str(person_id), path, since))
        return rows[0] if rows else None

    def recent(self, person_id, since):
        return self._rows("SELECT sent_at, path, name, bytes, surface FROM v2_file_sends WHERE person_id = %s "
                          "AND sent_at >= %s ORDER BY sent_at DESC LIMIT 50", (str(person_id), since))


def _when(at: datetime, now: datetime) -> str:
    """'10:14 PM today' / '10:14 PM Mon Nov 9', in the house's time."""
    local, today = at.astimezone(EASTERN), now.astimezone(EASTERN).date()
    clock = local.strftime("%I:%M %p").lstrip("0")
    return f"{clock} today" if local.date() == today else f"{clock} {local.strftime('%a %b')} {local.day}"


def _where(surface: str) -> str:
    return "his Telegram chat" if surface == "telegram" else "his Discord DM"


def build_send_file_tool(*, home_url: str, home_token: str, owner_person_id: str, deliver: Deliver,
                         get: Callable | None = None, ledger: FileSends | None = None,
                         now: Callable[[], datetime] | None = None):
    """The tool, closed over her home server and a delivery leg. get(url, headers) and now() are the
    test seams. ledger: the record of sends (#187); None keeps no record and guards nothing."""
    import httpx

    base = str(home_url).rstrip("/")
    headers = {"Authorization": f"Bearer {home_token}"}
    fetch = get or (lambda url, hdrs: httpx.get(url, headers=hdrs, timeout=30.0))
    clock = now or (lambda: datetime.now(timezone.utc))

    @tool(TOOL_NAME)
    def send_file_to_chris(path: str, config: RunnableConfig, note: str = "", again: bool = False) -> str:
        """Send Chris a file from your home machine: it lands in his DM, as a file he can open.

        CALL THIS when he wants a file your home job made (a write-up, a page, an export, an
        image): "send it to me", "can I have the file", "put it in my DM". The file must be in
        your home workspace; use the path your home job reported (absolute or relative to the
        workspace). It goes to him alone, on the surface he asked from (Telegram if he asked on
        Telegram, else his Discord DM), up to 8 MB. If he asks what you sent him, that's
        files_sent_to_chris, not this.

        path: the file's path in your home workspace. Required.
        note: a short line to send with it (optional).
        again: true ONLY when he asked you to send a file you already sent ("send it again").

        Returns SENT once he has it, ALREADY SENT when it went to him recently, or NOT SENT and
        why; never say he has it otherwise.
        """
        identity = identity_from_config(config)
        if str(identity.get("user_id") or "") != str(owner_person_id):
            return "NOT SENT: files from her home machine go only to Chris, when he asks."
        rel = _relative(path)
        if not rel:
            return "NOT SENT: say which file: its path in your home workspace."
        if ledger is not None and not again:
            try:
                prev = ledger.last(owner_person_id, rel, clock() - timedelta(hours=RESEND_HOURS))
            except Exception as e:       # a record we can't read never blocks a send he asked for
                log.warning("send_file: couldn't read the record of sends (%s)", type(e).__name__)
                prev = None
            if prev is not None:
                return (f"ALREADY SENT: {prev.name} ({prev.bytes} bytes) went to {_where(prev.surface)} at "
                        f"{_when(prev.sent_at, clock())}. It's there; don't send it again unless he asks you "
                        "to send it again (then call with again=true).")
        try:
            r = fetch(f"{base}/files/{quote(rel, safe='/')}", headers)
        except Exception as e:
            log.warning("send_file: home machine unreachable (%s)", type(e).__name__)
            return f"NOT SENT: I couldn't reach my home machine ({type(e).__name__}); try again in a moment."
        if r.status_code == 404:
            return f"NOT SENT: there's no file at {rel} in my home workspace (check the path my job reported)."
        if r.status_code == 413:
            return "NOT SENT: that file is over 8 MB, too big to send in a message."
        if r.status_code in (401, 403):
            return "NOT SENT: my home machine refused the request (its key didn't match)."
        if r.status_code != 200:
            return f"NOT SENT: my home machine answered HTTP {r.status_code}."
        data = bytes(r.content or b"")
        if len(data) > MAX_BYTES:
            return "NOT SENT: that file is over 8 MB, too big to send in a message."
        name = _name((r.headers or {}).get("X-File-Name"), rel)
        surface = "telegram" if str(identity.get("platform") or "").strip().lower() == "telegram" else "discord"
        try:
            deliver(surface, name, data, " ".join(str(note or "").split())[:NOTE_CHARS])
        except Exception as e:
            log.warning("send_file: %s didn't take the file (%s)", surface, type(e).__name__)
            return (f"NOT SENT: I got the file from my home machine, but {surface.title()} wouldn't take it "
                    f"({type(e).__name__}).")
        log.info("send_file: a file went to him on %s (%d bytes)", surface, len(data))
        if ledger is not None:
            try:
                thread = str(((config or {}).get("configurable") or {}).get("thread_id") or "")
                ledger.record(owner_person_id, FileSend(clock(), rel, name, len(data), surface), thread)
            except Exception as e:       # he has the file either way; only the record is missing
                log.warning("send_file: sent, but the record of it failed (%s)", type(e).__name__)
        return f"SENT: {name} ({len(data)} bytes) is in {_where(surface)}."

    return send_file_to_chris


def build_files_sent_tool(*, ledger: FileSends, owner_person_id: str,
                          now: Callable[[], datetime] | None = None):
    """What she handed him, from the record (#187): the evidence for 'what did you send me?'."""
    clock = now or (lambda: datetime.now(timezone.utc))

    @tool(SENT_TOOL_NAME)
    def files_sent_to_chris(config: RunnableConfig, hours: int = 48) -> str:
        """The files you've sent Chris, newest first: when, what, and where it went.

        CALL THIS when he asks what you sent him, whether a file reached him, or when. It is the
        record: every file send_file_to_chris delivered is in it. Answer from it; don't send a
        file again because he asked about it.

        hours: how far back to look (default 48, at most a week).
        """
        identity = identity_from_config(config)
        if str(identity.get("user_id") or "") != str(owner_person_id):
            return "The record of files sent to Chris is only for Chris."
        span = max(1, min(int(hours or 48), 168))
        try:
            sends = ledger.recent(owner_person_id, clock() - timedelta(hours=span))
        except Exception as e:
            log.warning("files_sent: couldn't read the record (%s)", type(e).__name__)
            return "I couldn't read the record of files I sent just now; I don't know, so I won't guess."
        if not sends:
            return f"No files sent to Chris in the last {span} hours."
        lines = [f"- {_when(s.sent_at, clock())}: {s.name} ({s.bytes} bytes), {s.path}, to {_where(s.surface)}"
                 for s in sends]
        return f"Files sent to Chris in the last {span} hours, newest first:\n" + "\n".join(lines)

    return files_sent_to_chris


def deliver_for(settings, *, post: Callable | None = None,
                owner_id: Callable[[str], str | None] | None = None) -> Deliver | None:
    """deliver(surface, name, data, note) over her own bots: Discord (a DM attachment) and Telegram
    (a document). None when neither bot is configured. Raises when the surface can't be reached
    (no bot, no id for him there, a refused upload); the tool turns that into NOT SENT.
    post(url, **httpx_kwargs) and owner_id(platform) are the test seams."""
    discord = getattr(settings, "discord_bot_token", None)
    telegram = getattr(settings, "telegram_bot_token", None)
    if discord is None and telegram is None:
        return None
    import httpx

    send = post or (lambda url, **kw: httpx.post(url, timeout=60.0, **kw))
    if owner_id is None:
        from aerys_v2.kael_line import _owner_platform_id

        owner_id = lambda platform: _owner_platform_id(settings, platform)    # noqa: E731

    def ok(r, what: str):
        if not 200 <= r.status_code < 300:
            raise RuntimeError(f"{what} answered HTTP {r.status_code}")
        return r

    def to_discord(name: str, data: bytes, note: str) -> None:
        if discord is None:
            raise RuntimeError("no Discord bot")
        user = owner_id("discord")
        if not user:
            raise RuntimeError("no Discord id for him")
        hdrs = {"Authorization": f"Bot {discord.get_secret_value()}", "User-Agent": USER_AGENT}
        dm = ok(send(f"{DISCORD_API}/users/@me/channels", headers={**hdrs, "Content-Type": "application/json"},
                     content=json.dumps({"recipient_id": user})), "Discord")
        channel = json.loads(dm.content)["id"]
        payload = {"content": note, "attachments": [{"id": 0, "filename": name}]}
        ok(send(f"{DISCORD_API}/channels/{channel}/messages", headers=hdrs,
                data={"payload_json": json.dumps(payload)},
                files={"files[0]": (name, data, "application/octet-stream")}), "Discord")

    def to_telegram(name: str, data: bytes, note: str) -> None:
        if telegram is None:
            raise RuntimeError("no Telegram bot")
        chat = owner_id("telegram")
        if not chat:
            raise RuntimeError("no Telegram id for him")
        ok(send(f"https://api.telegram.org/bot{telegram.get_secret_value()}/sendDocument",
                data={"chat_id": chat, "caption": note}, files={"document": (name, data)}), "Telegram")

    def deliver(surface: str, name: str, data: bytes, note: str) -> None:
        (to_telegram if surface == "telegram" else to_discord)(name, data, note)

    return deliver


def send_file_middleware(file_tool, owner_person_id: str, sent_tool=None):
    """Mounts the tools for him only: anyone else's model call doesn't see them (and a call from
    anyone else is refused in the tools too). sent_tool: files_sent_to_chris, the record (#187)."""
    from langchain.agents.middleware import AgentMiddleware
    from langgraph.config import get_config

    mine = [file_tool] + ([sent_tool] if sent_tool is not None else [])
    names = {getattr(t, "name", None) for t in mine}

    def his(request) -> bool:
        try:
            config = get_config()
        except RuntimeError:          # not inside a run: nobody to offer it to
            return False
        return str(identity_from_config(config).get("user_id") or "") == str(owner_person_id)

    def offered(request):
        if his(request):
            return request
        return request.override(tools=[t for t in (request.tools or []) if getattr(t, "name", None) not in names])

    class SendFile(AgentMiddleware):
        tools = mine

        def wrap_model_call(self, request, handler):
            return handler(offered(request))

        async def awrap_model_call(self, request, handler):
            return await handler(offered(request))

    return SendFile()
