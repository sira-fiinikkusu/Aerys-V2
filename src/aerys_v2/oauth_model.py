"""OAuth-backed chat model — her daily words on the Max pool, not API tokens.

n8n mapping: this is swapping which credential the AI Agent node uses, except the
"credential" here is Chris's Claude subscription (the same auth Kael runs on). The
Claude Agent SDK runs the bundled Claude Code CLI under the hood; we use it as a
PURE chat backend. LangGraph never knows the difference — build_model() returns
"a chat model" either way.

Spare-client design (v3 of this module, 2026-09-20 — Codex review of f7edf77,
reproduced live): a connected ClaudeSDKClient is ONE conversation for the life of
its process. The `session_id` passed to query() is only a field on the message; the
bundled CLI keeps appending to the same conversation, so the v2 claim "warm process,
cold context" was false — turn B could recall turn A. Real isolation is a NEW PROCESS
per turn. To keep the latency of a warm client anyway:
  - Each turn TAKES a pre-connected spare client, uses it for exactly one turn, and
    discards it (disconnect in the background).
  - The moment a turn starts, the NEXT spare begins connecting in the background, so
    the following turn usually finds one ready (~2 s connect hidden behind the reply).
  - Every process gets its own empty temp cwd: no project transcript, no auto-memory,
    no CLAUDE.md can carry between turns (setting_sources=[] alone does not stop
    memory — Codex #1).
  - The full system prompt rides the head of each turn's prompt.
  - A turn is bounded by turn_timeout_s; on timeout the in-flight task is CANCELLED
    and its client dropped (Codex #2 — a timed-out reader must not survive).
"""

import asyncio
import atexit
import contextlib
import logging
import hashlib
import json
import re
import threading
import uuid
from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.utils.function_calling import convert_to_openai_tool


log = logging.getLogger(__name__)


class OAuthBackendError(RuntimeError):
    """The subscription client failed (transport, CLI death, timeout, SDK error).

    One class so the failover wrappers can treat it like a connection failure:
    it is never a programming error and never a refusal.
    """


# Tools are exposed to the CLI as an in-process MCP server under this prefix; the
# model calls them as mcp__lc__<name>. We NEVER let the SDK run one: the
# PreToolUse hook defers every call, the run ends with stop_reason
# "tool_deferred", and LangGraph's ToolNode executes it. Results come back in
# the next turn's prompt text (measured 2026-09-20: 2.6 s, no re-fire; the
# alternative — delivering results by letting the CLI re-fire the tool —
# stalled 78 s on a warm client).
MCP_SERVER = "lc"
MCP_PREFIX = f"mcp__{MCP_SERVER}__"
_FORCE_TOOL_LINE = "[System instructions] You MUST call one of the provided tools before answering."
_RESULTS_FINAL_LINE = ("[System instructions] The tool results above are final: those exact calls already ran, "
                       "do not repeat an identical call. If the request needs a further, different tool call "
                       "(for example reading a state before setting it), make it; otherwise answer the user now.")
_FORCED_REFUSED_LINE = "I couldn't get the tool to run for that, so nothing was changed. Ask me again and I'll try once more."
_MALFORMED_CALL_LINE = "The tool call came back malformed, so I didn't run it and nothing was changed. Ask me again and I'll try once more."


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            (b.get("text", "") if isinstance(b, dict) else str(b)) for b in content
            if not (isinstance(b, dict) and b.get("type") in ("tool_use", "thinking"))
        ).strip()
    return str(content)


# Gemini review 2026-09-20 (blocker): the flattened prompt is ONE user message, so text
# that came from a user, a tool or a web page could open a line with "User:", "Aerys:"
# or "[System instructions]" and impersonate the transcript. Every such line inside
# untrusted text gets a visible quote mark so it reads as content, never as a speaker.
_LABEL_RE = re.compile(r"^(\s*)(User|Aerys|Human|Assistant|System|\[System instructions\]|\[Tool result[^\]]*\])(\s*:?)", re.M | re.I)
# Gemini 2nd pass: untrusted text could also forge "<called tool X with {...}>" — the marker
# the model reads as ITS OWN past action — or the API's own role boundaries.
_MARKER_RE = re.compile(r"<\s*called tool\b", re.I)


def _neutralize(text: str) -> str:
    text = _LABEL_RE.sub(lambda m: f"{m.group(1)}> {m.group(2)}{m.group(3)}", text)
    text = _MARKER_RE.sub("< called-tool", text)
    return text.replace("\n\nHuman:", "\n\n> Human:").replace("\n\nAssistant:", "\n\n> Assistant:")


def _valid_tool_calls(calls: list[dict]) -> tuple[list[dict], int]:
    """Codex #6: a call with no id, or a duplicate id, must never reach the ToolNode
    (it executed a null-id call and ran duplicates twice). Drop and log, keep the rest;
    the count of dropped calls travels so a forced pass can tell "malformed" from "none"."""
    seen: set[str] = set()
    out: list[dict] = []
    dropped = 0
    for c in calls or []:
        cid = c.get("id")
        if not cid or cid in seen or not c.get("name"):
            log.warning("oauth tool call dropped: malformed id/name %r", {k: c.get(k) for k in ("id", "name")})
            dropped += 1
            continue
        seen.add(cid)
        out.append(c)
    return out, dropped


def _sum_usage(*metas: dict) -> dict | None:
    total = {"input_tokens": 0, "output_tokens": 0}
    seen = False
    for m in metas:
        u = (m or {}).get("usage") or {}
        if u:
            seen = True
            total["input_tokens"] += int(u.get("input_tokens", 0) or 0)
            total["output_tokens"] += int(u.get("output_tokens", 0) or 0)
    return total if seen else None


def _flatten(messages: list[BaseMessage], *, force_tool: bool = False) -> str:
    """Serialize LangChain messages into one speaker-labeled prompt.

    System content leads, history follows labeled (same shape as thread_context
    snippets in n8n — the model reads attribution, it doesn't infer it). Tool
    calls and results ride as text too: "<called tool X with {...}>" and
    "[Tool result X]: ...". When the last message is a tool result the prompt
    closes with the results-are-final line; force_tool adds the must-call line
    (the forced first specialist pass, tool_choice="any").
    """
    system_parts: list[str] = []
    lines: list[str] = []
    names: dict[str, str] = {}
    for m in messages:
        if isinstance(m, SystemMessage):
            system_parts.append(str(m.content))
        elif isinstance(m, HumanMessage):
            lines.append(f"User: {_neutralize(_text(m.content))}")
        elif isinstance(m, ToolMessage):
            name = names.get(m.tool_call_id, "tool")
            # Delimited AND neutralized: the model sees a quoted block, not a speaker.
            lines.append(f"[Tool result {name}] <<<\n{_neutralize(_text(m.content))}\n>>> end of tool result {name}")
        elif isinstance(m, AIMessage):
            parts = []
            text = _text(m.content)
            if text:
                parts.append(_neutralize(text))   # her own free text is neutralized like any text…
            for tc in m.tool_calls:
                names[tc.get("id", "")] = tc["name"]
                # …the marker is OURS (built from the structured tool_calls), never neutralized
                parts.append(f"<called tool {tc['name']} with {json.dumps(tc.get('args', {}), sort_keys=True)}>")
            lines.append("Aerys: " + " ".join(parts))
    head = ("[System instructions]\n" + "\n\n".join(system_parts) + "\n\n") if system_parts else ""
    tail = ""
    if messages and isinstance(messages[-1], ToolMessage):
        tail += "\n" + _RESULTS_FINAL_LINE
    if force_tool:
        tail += "\n" + _FORCE_TOOL_LINE
    return head + "\n".join(lines) + tail + "\nAerys:"


class _WarmClient:
    """Owns the event-loop thread and a SPARE connected ClaudeSDKClient (one pool per model + tool set)."""

    def __init__(self, model: str, tool_schemas: list[dict] | None = None, turn_timeout_s: float = 60.0) -> None:
        self.model = model
        self.tool_schemas = list(tool_schemas or [])
        self.turn_timeout_s = turn_timeout_s
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True, name="oauth-model")
        self._thread.start()
        self._spare = None            # a connected, never-used client (with its temp cwd)
        self._spare_task = None       # the background connect in flight, if any
        self._lock = threading.Lock()  # one turn at a time per pool — household-sized
        self.spawns = 0               # audit: processes started
        atexit.register(self.close)   # the pre-connected spare must not outlive the process

    # ---- plumbing -------------------------------------------------------------
    def _run(self, coro, timeout: float | None = None):
        fut = asyncio.run_coroutine_threadsafe(coro, self._loop)
        try:
            return fut.result(timeout=timeout or self.turn_timeout_s)
        except BaseException:
            fut.cancel()  # Codex #2: never leave a reader alive past its deadline
            raise

    def _options(self, cwd: str):
        from claude_agent_sdk import ClaudeAgentOptions

        # AUTH PRECEDENCE TRAP (found 2026-07-03): if ANTHROPIC_API_KEY exists in
        # the process env, the spawned CLI prefers it over subscription auth — the
        # container would silently bill the API while claiming oauth. Neutralize it
        # for the subprocess; subscription login / CLAUDE_CODE_OAUTH_TOKEN remain.
        options = ClaudeAgentOptions(
            model=self.model,
            max_turns=1 if not self.tool_schemas else 2,
            # TOOLS-OFF TRAP (root-caused 2026-07-03): `allowed_tools` only controls
            # AUTO-PERMISSION; `tools=[]` is what actually removes the built-ins.
            tools=[],
            allowed_tools=[],
            permission_mode="default",
            env={"ANTHROPIC_API_KEY": ""},
            # Never load the host's ~/.claude settings (hooks, CLAUDE.md) into her.
            setting_sources=[],
            # A private, empty working directory per PROCESS: no transcript or
            # auto-memory of an earlier turn can be found, let alone loaded.
            cwd=cwd,
        )
        if self.tool_schemas:
            from claude_agent_sdk import HookMatcher, create_sdk_mcp_server, tool as sdk_tool

            async def _never(args: dict) -> dict:  # the hook defers first; belt and braces
                return {"content": [{"type": "text", "text": ""}]}

            async def _defer(input_data: dict, tool_use_id: str | None, context: Any) -> dict:
                return {"hookSpecificOutput": {"hookEventName": "PreToolUse", "permissionDecision": "defer"}}

            sdk_tools = [
                sdk_tool(t["name"], t.get("description", ""), t.get("parameters") or {"type": "object", "properties": {}})(_never)
                for t in self.tool_schemas
            ]
            options.mcp_servers = {MCP_SERVER: create_sdk_mcp_server(name=MCP_SERVER, version="1.0.0", tools=sdk_tools)}
            options.allowed_tools = [f"{MCP_PREFIX}{t['name']}" for t in self.tool_schemas]
            options.hooks = {"PreToolUse": [HookMatcher(matcher=f"{MCP_PREFIX}.*", hooks=[_defer])]}
        return options

    async def _connect(self):
        """Spawn ONE fresh process in its own empty cwd. Returns (client, cwd)."""
        import tempfile

        from claude_agent_sdk import ClaudeSDKClient

        cwd = tempfile.mkdtemp(prefix="aerys-oauth-")
        client = ClaudeSDKClient(options=self._options(cwd))
        try:
            await client.connect()
        except BaseException:
            # Gemini 2nd pass: a failed or cancelled connect must not leak its temp dir
            # (or a half-started process).
            import shutil
            with contextlib.suppress(Exception):
                await asyncio.wait_for(client.disconnect(), timeout=5)
            shutil.rmtree(cwd, ignore_errors=True)
            raise
        self.spawns += 1
        return client, cwd

    async def _discard(self, entry) -> None:
        import shutil

        client, cwd = entry
        try:
            await asyncio.wait_for(client.disconnect(), timeout=10)
        except Exception:
            pass
        shutil.rmtree(cwd, ignore_errors=True)

    def close(self) -> None:
        """Discard the spare (and its temp dir) — atexit and tests."""
        entry, self._spare = self._spare, None
        if entry is None:
            return
        with contextlib.suppress(Exception):
            asyncio.run_coroutine_threadsafe(self._discard(entry), self._loop).result(timeout=10)

    def _prewarm(self) -> None:
        """Start connecting the next spare on the loop (idempotent, fire-and-forget)."""
        async def go():
            try:
                entry = await self._connect()
            except Exception:
                log.warning("oauth spare client failed to connect; next turn connects inline", exc_info=True)
                return
            if self._spare is None:
                self._spare = entry
            else:  # a spare already exists (race) — do not hold two processes
                await self._discard(entry)

        if self._spare_task is None or self._spare_task.done():
            self._spare_task = asyncio.ensure_future(go(), loop=self._loop)

    async def _take(self):
        """A fresh connected client for THIS turn: the spare if ready, else connect now."""
        if self._spare_task is not None and not self._spare_task.done():
            try:
                await asyncio.wait_for(asyncio.shield(self._spare_task), timeout=self.turn_timeout_s / 2)
            except asyncio.TimeoutError:
                raise OAuthBackendError("spare client did not connect within half the turn budget")
            except Exception:
                pass  # the prewarm failed; connect inline below (still inside the budget)
        entry = self._spare
        self._spare = None
        if entry is None:
            entry = await self._connect()
        return entry

    # ---- one turn -------------------------------------------------------------
    async def _turn(self, prompt: str) -> tuple[str, list[dict], dict]:
        from claude_agent_sdk import AssistantMessage, ResultMessage
        from claude_agent_sdk.types import TextBlock, ToolUseBlock

        entry = await self._take()
        client, _cwd = entry
        # The next turn's process starts connecting NOW, behind this reply.
        self._prewarm()
        session = uuid.uuid4().hex
        result_text: str | None = None
        assistant_text: list[str] = []
        tool_calls: list[dict] = []
        meta: dict = {"model": self.model}
        try:
            await client.query(prompt, session_id=session)
            async for message in client.receive_response():
                if type(message).__name__ == "RateLimitEvent":
                    meta["rate_limit"] = {k: getattr(message, k, None) for k in ("status", "type", "utilization", "resets_at")}
                if isinstance(message, AssistantMessage):
                    for block in message.content:
                        if isinstance(block, TextBlock):
                            # Text AFTER a deferred tool call is the model reacting to the
                            # deferral, not an answer — drop it (same rule as upstream).
                            if not tool_calls:
                                assistant_text.append(block.text)
                        elif isinstance(block, ToolUseBlock) and block.name.startswith(MCP_PREFIX):
                            tool_calls.append({"name": block.name[len(MCP_PREFIX):], "args": dict(block.input or {}),
                                               "id": block.id, "type": "tool_call"})
                if isinstance(message, ResultMessage):
                    meta["stop_reason"] = getattr(message, "stop_reason", None)
                    meta["usage"] = getattr(message, "usage", None)
                    meta["total_cost_usd"] = getattr(message, "total_cost_usd", None)
                    meta["session_id"] = getattr(message, "session_id", None)  # the CLI's, not ours
                    if getattr(message, "is_error", False):
                        raise RuntimeError(
                            "oauth backend error: "
                            f"subtype={getattr(message, 'subtype', None)!r} "
                            f"result={message.result!r} "
                            f"num_turns={getattr(message, 'num_turns', None)!r} "
                            f"stop_reason={getattr(message, 'stop_reason', None)!r} "
                            f"permission_denials={getattr(message, 'permission_denials', None)!r}"
                        )
                    result_text = message.result
        finally:
            # Used once, gone: the conversation dies with the process.
            asyncio.ensure_future(self._discard(entry))
        if tool_calls:
            return "".join(assistant_text), tool_calls, meta
        return result_text or "".join(assistant_text) or "", [], meta

    def ask(self, prompt: str) -> tuple[str, list[dict], dict]:
        import concurrent.futures

        with self._lock:
            try:
                return self._run(self._turn(prompt))
            except concurrent.futures.TimeoutError as hung:
                # Cancelled by _run; the process is discarded by _turn's finally when the
                # cancellation lands. Fail this turn now; the next one takes a fresh spare.
                raise OAuthBackendError(f"turn exceeded {self.turn_timeout_s:.0f}s; client dropped") from hung
            except Exception as first:
                # A dead spare (idle timeout, OOM, upgrade) — one retry on a fresh process.
                try:
                    return self._run(self._turn(prompt))
                except Exception as second:
                    raise OAuthBackendError(f"{type(first).__name__}: {str(first)[:160]} / "
                                            f"retry {type(second).__name__}: {str(second)[:160]}") from second


_CLIENTS: dict[str, _WarmClient] = {}
_CLIENTS_LOCK = threading.Lock()


def _schema_key(model: str, schemas: list[dict]) -> str:
    digest = hashlib.sha256(json.dumps(schemas, sort_keys=True).encode()).hexdigest()[:12]
    return f"{model}:{digest}"


def warm_client_for(model: str, schemas: list[dict], turn_timeout_s: float = 60.0) -> _WarmClient:
    """One CLI subprocess per (model, tool set); every ChatModel copy shares it."""
    key = _schema_key(model, schemas)
    with _CLIENTS_LOCK:
        client = _CLIENTS.get(key)
        if client is None:
            client = _CLIENTS[key] = _WarmClient(model, schemas, turn_timeout_s=turn_timeout_s)
        return client


class ClaudeOAuthChatModel(BaseChatModel):
    """LangChain chat model backed by a warm Claude Agent SDK client.

    Deliberately minimal: one turn, no tools, no MCP — the agent loop belongs to
    LangGraph. When tool-calling lands (01-03+), the SDK-loop-vs-ToolNode fork
    gets navigated on purpose, not by accident (see CROSS-REVIEW).
    """

    model: str = "claude-sonnet-5"
    # bind_tools() fills these on a COPY: OpenAI-style schemas (name/description/
    # parameters) and whether the first pass must call a tool (tool_choice="any").
    bound_tools: list[dict] = []
    force_tool: bool = False
    turn_timeout_s: float = 60.0

    _warm: Any = None  # lazily created _WarmClient (pydantic private-ish)

    @property
    def _llm_type(self) -> str:
        return "claude-oauth-sdk"

    def bind_tools(self, tools: list, *, tool_choice: Any = None, **kwargs: Any) -> "ClaudeOAuthChatModel":
        schemas = []
        for t in tools:
            fn = convert_to_openai_tool(t)["function"]
            schemas.append({"name": fn["name"], "description": fn.get("description", ""),
                            "parameters": fn.get("parameters") or {"type": "object", "properties": {}}})
        bound = self.model_copy(update={"bound_tools": schemas, "force_tool": tool_choice in ("any", "required")})
        object.__setattr__(bound, "_warm", None)
        return bound

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        forced = self.force_tool and bool(self.bound_tools)
        out = self._query(_flatten(messages, force_tool=forced))
        if isinstance(out, str):  # tests may stub _query with a bare string
            text, tool_calls, meta = out, [], {}
        else:
            text, tool_calls, meta = out
        tool_calls, dropped = _valid_tool_calls(tool_calls)
        metas = [meta]
        if forced and not tool_calls:
            # Codex #4: the must-call line is advisory; a text answer on the forced pass
            # would read as a fabricated success ("the light is off" with nothing run).
            # One more try, then a deterministic honest line the gate can mark. A pass
            # whose calls were all MALFORMED says so instead (Gemini 2nd pass).
            out = self._query(_flatten(messages, force_tool=True) + "\n" + _FORCE_TOOL_LINE)
            text, tool_calls, meta = (out, [], {}) if isinstance(out, str) else out
            metas.append(meta)
            tool_calls, dropped2 = _valid_tool_calls(tool_calls)
            if not tool_calls:
                meta = {**(meta or {}), "forced_refused": True}
                if dropped or dropped2:
                    meta["malformed_tool_calls"] = dropped + dropped2
                    text = _MALFORMED_CALL_LINE
                else:
                    text = _FORCED_REFUSED_LINE
        total = _sum_usage(*metas)
        usage_metadata = None
        if total:
            usage_metadata = {**total, "total_tokens": total["input_tokens"] + total["output_tokens"]}
        msg = AIMessage(content=text, tool_calls=tool_calls, usage_metadata=usage_metadata,
                        response_metadata={k: v for k, v in (meta or {}).items() if k != "usage"})
        return ChatResult(generations=[ChatGeneration(message=msg)])

    def _query(self, prompt: str):
        if self._warm is None:
            object.__setattr__(self, "_warm", warm_client_for(self.model, self.bound_tools, self.turn_timeout_s))
        return self._warm.ask(prompt)
