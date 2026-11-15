"""D53 (11/14): her reach into Music Assistant's own MCP server, through LangChain's MCP bridge.

`langchain.mcp` (LangChain 1.4+, beta, built on FastMCP) turns each MCP tool into a LangChain tool that
opens its own session per call. Those tools are async-only and her action graph is synchronous, so every
call runs on ONE private event loop in a daemon thread and the caller waits with a timeout.

Only the tools in ALLOWED are ever loaded; the server can offer more (playback, volume, players stay on
her Home Assistant path, which holds the speaker allowlist). The client carries no elicitation handler,
so a server can never stop a call to ask for input. The token is never logged.
"""
from __future__ import annotations

import asyncio
import json
import logging
import threading
import uuid
from typing import Any, Callable

log = logging.getLogger(__name__)

#: the Music Assistant tools she may use (his D53 permissions: edit on, control off)
ALLOWED = frozenset({
    "media_add_to_favorites", "media_add_to_library",
    "library_search_tracks", "metadata_recently_played",
    "players_list_players", "queue_get_active_queue", "queue_set_shuffle", "queue_set_repeat",
    "playlists_create_playlist", "playlists_add_track",
})


class MusicAssistantError(RuntimeError):
    """A Music Assistant call that didn't work, with a reason she can say."""


def _text(result: Any) -> str:
    """The text of an MCP tool result: LangChain hands back content blocks, a string on a handled error."""
    if isinstance(result, str):
        return result
    if isinstance(result, (list, tuple)):
        return "".join(block.get("text", "") if isinstance(block, dict) else str(block) for block in result)
    return str(result)


def parse(result_text: str) -> Any:
    """A tool's JSON payload, or None when it is empty (an idle player's queue) or not JSON."""
    text = (result_text or "").strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except ValueError:
        return None


class MusicAssistantMCP:
    def __init__(self, url: str, token: str, *, timeout: float = 20.0,
                 client_factory: Callable[[], Any] | None = None):
        self.url, self.timeout = url, timeout
        self._token = token
        self._client_factory = client_factory or self._client
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._tools: dict[str, Any] | None = None

    def _client(self):
        from fastmcp.client import Client
        from fastmcp.client.transports import StreamableHttpTransport

        return Client(StreamableHttpTransport(self.url, headers={"Authorization": f"Bearer {self._token}"}))

    def _run(self, coro):
        with self._lock:
            # One loop, published only once it is running (Codex 11/15: checking is_running() right after
            # Thread.start() raced, and concurrent first calls left a second loop behind).
            if self._loop is None or self._thread is None or not self._thread.is_alive():
                loop, ready = asyncio.new_event_loop(), threading.Event()

                def serve():
                    asyncio.set_event_loop(loop)
                    loop.call_soon(ready.set)
                    loop.run_forever()
                thread = threading.Thread(target=serve, daemon=True, name="ma-mcp")
                thread.start()
                if not ready.wait(5):
                    raise MusicAssistantError("Music Assistant's connection didn't start")
                self._loop, self._thread = loop, thread
            loop = self._loop
        future = asyncio.run_coroutine_threadsafe(coro, loop)
        try:
            return future.result(self.timeout)
        except TimeoutError:
            future.cancel()
            raise MusicAssistantError("Music Assistant didn't answer in time") from None

    async def _load(self) -> dict[str, Any]:
        import warnings

        with warnings.catch_warnings():                      # LangChainBetaWarning: we pin and test it
            warnings.simplefilter("ignore")
            from langchain.mcp import as_langchain_tool
        client = self._client_factory()
        async with client:
            listed = await client.list_tools()
        return {t.name: await as_langchain_tool(t, client) for t in listed if t.name in ALLOWED}

    def call(self, name: str, /, **arguments: Any) -> str:
        """Run one allowed Music Assistant tool and return its text. Raises MusicAssistantError.
        `name` is positional-only: a tool's own arguments may include `name` (playlists_create_playlist)."""
        if name not in ALLOWED:
            raise MusicAssistantError(f"{name} isn't one of her Music Assistant tools")
        try:
            if self._tools is None:
                self._tools = self._run(self._load())
            tool = self._tools.get(name)
            if tool is None:
                raise MusicAssistantError(f"Music Assistant isn't offering {name} (is its permission on?)")
            # A tool CALL (not bare args), so a server-side failure comes back as status="error" instead of
            # text that would read like success (a "liked" that never happened).
            message = self._run(tool.ainvoke({"name": name, "args": arguments, "type": "tool_call",
                                              "id": f"aerys-{uuid.uuid4().hex[:12]}"}))
            text = _text(getattr(message, "content", message))
            if getattr(message, "status", "success") == "error":
                raise MusicAssistantError(f"Music Assistant said no: {text[:160]}")
            return text
        except MusicAssistantError:
            raise
        except Exception as e:                               # transport/protocol: one plain sentence
            self._tools = None                               # rediscover next time (a restart, a new token)
            log.warning("Music Assistant MCP call failed | tool=%s error=%s", name, type(e).__name__)
            raise MusicAssistantError("Music Assistant couldn't be reached") from None


def ma_for(settings) -> MusicAssistantMCP | None:
    if not settings.ma_mcp_url or settings.ma_mcp_token is None:
        return None
    return MusicAssistantMCP(settings.ma_mcp_url, settings.ma_mcp_token.get_secret_value())
