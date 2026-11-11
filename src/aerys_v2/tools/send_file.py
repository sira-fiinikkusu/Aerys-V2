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
from typing import Callable
from urllib.parse import quote

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

from aerys_v2.state import identity_from_config

log = logging.getLogger(__name__)

TOOL_NAME = "send_file_to_chris"
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


def build_send_file_tool(*, home_url: str, home_token: str, owner_person_id: str, deliver: Deliver,
                         get: Callable | None = None):
    """The tool, closed over her home server and a delivery leg. get(url, headers) is the test seam."""
    import httpx

    base = str(home_url).rstrip("/")
    headers = {"Authorization": f"Bearer {home_token}"}
    fetch = get or (lambda url, hdrs: httpx.get(url, headers=hdrs, timeout=30.0))

    @tool(TOOL_NAME)
    def send_file_to_chris(path: str, config: RunnableConfig, note: str = "") -> str:
        """Send Chris a file from your home machine: it lands in his DM, as a file he can open.

        CALL THIS when he wants a file your home job made (a write-up, a page, an export, an
        image): "send it to me", "can I have the file", "put it in my DM". The file must be in
        your home workspace; use the path your home job reported (absolute or relative to the
        workspace). It goes to him alone, on the surface he asked from (Telegram if he asked on
        Telegram, else his Discord DM), up to 8 MB.

        path: the file's path in your home workspace. Required.
        note: a short line to send with it (optional).

        Returns SENT once he has it, or NOT SENT and why; never say he has it otherwise.
        """
        identity = identity_from_config(config)
        if str(identity.get("user_id") or "") != str(owner_person_id):
            return "NOT SENT: files from her home machine go only to Chris, when he asks."
        rel = _relative(path)
        if not rel:
            return "NOT SENT: say which file: its path in your home workspace."
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
        where = "his Telegram chat" if surface == "telegram" else "his Discord DM"
        return f"SENT: {name} ({len(data)} bytes) is in {where}."

    return send_file_to_chris


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


def send_file_middleware(file_tool, owner_person_id: str):
    """Mounts the tool for him only: anyone else's model call doesn't see it (and a call from
    anyone else is refused in the tool too)."""
    from langchain.agents.middleware import AgentMiddleware
    from langgraph.config import get_config

    def his(request) -> bool:
        try:
            config = get_config()
        except RuntimeError:          # not inside a run: nobody to offer it to
            return False
        return str(identity_from_config(config).get("user_id") or "") == str(owner_person_id)

    def offered(request):
        if his(request):
            return request
        return request.override(tools=[t for t in (request.tools or []) if getattr(t, "name", None) != TOOL_NAME])

    class SendFile(AgentMiddleware):
        tools = [file_tool]

        def wrap_model_call(self, request, handler):
            return handler(offered(request))

        async def awrap_model_call(self, request, handler):
            return await handler(offered(request))

    return SendFile()
