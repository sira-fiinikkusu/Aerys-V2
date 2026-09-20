"""OAuth-backed chat model — her daily words on the Max pool, not API tokens.

n8n mapping: this is swapping which credential the AI Agent node uses, except the
"credential" here is Chris's Claude subscription (the same auth Kael runs on). The
Claude Agent SDK runs the bundled Claude Code CLI under the hood; we use it as a
PURE chat backend. LangGraph never knows the difference — build_model() returns
"a chat model" either way.

Warm-client design (v2 of this module — the first version spawned the CLI per turn,
~3-4s of pure process boot on every reply):
  - ONE ClaudeSDKClient is spawned lazily and kept warm on a dedicated event-loop
    thread (the sync-facade-over-async-client pattern).
  - EVERY turn uses a FRESH session_id. The warm client is stateful by design, but
    two history owners (SDK session + LangGraph checkpointer) is the session-
    contamination bug in a new hat — so the process is warm, the context is not.
  - The full system prompt rides the head of each turn's prompt (per-turn caller
    line means it can't be baked into connect-time options).
  - If the warm process died, reconnect once and retry — then fail loudly.
"""

import asyncio
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
_RESULTS_FINAL_LINE = ("[System instructions] The tool results above are final and already executed. "
                       "Do not call those tools again for the same purpose; answer the user now.")


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
_LABEL_RE = re.compile(r"^(\s*)(User|Aerys|\[System instructions\]|\[Tool result[^\]]*\])(\s*:?)", re.M | re.I)


def _neutralize(text: str) -> str:
    return _LABEL_RE.sub(lambda m: f"{m.group(1)}> {m.group(2)}{m.group(3)}", text)


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
                parts.append(text)
            for tc in m.tool_calls:
                names[tc.get("id", "")] = tc["name"]
                parts.append(f"<called tool {tc['name']} with {json.dumps(tc.get('args', {}), sort_keys=True)}>")
            lines.append("Aerys: " + _neutralize(" ".join(parts)))
    head = ("[System instructions]\n" + "\n\n".join(system_parts) + "\n\n") if system_parts else ""
    tail = ""
    if messages and isinstance(messages[-1], ToolMessage):
        tail += "\n" + _RESULTS_FINAL_LINE
    if force_tool:
        tail += "\n" + _FORCE_TOOL_LINE
    return head + "\n".join(lines) + tail + "\nAerys:"


class _WarmClient:
    """Owns the event-loop thread + the connected ClaudeSDKClient (one per model + tool set)."""

    def __init__(self, model: str, tool_schemas: list[dict] | None = None, turn_timeout_s: float = 60.0) -> None:
        self.model = model
        self.tool_schemas = list(tool_schemas or [])
        # Gemini review 2026-09-20: a hung CLI used to hold the lock 120 s, then 120 s
        # more on the retry. One bounded turn; a hang resets the client and raises.
        self.turn_timeout_s = turn_timeout_s
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True, name="oauth-model")
        self._thread.start()
        self._client = None
        self._lock = threading.Lock()  # one turn at a time — household-sized

    def _run(self, coro, timeout: float | None = None):
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result(timeout=timeout or self.turn_timeout_s)

    async def _connect(self):
        from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient

        # AUTH PRECEDENCE TRAP (found 2026-07-03): if ANTHROPIC_API_KEY exists in
        # the process env, the spawned CLI prefers it over subscription auth — the
        # container would silently bill the API while claiming oauth. Neutralize it
        # for the subprocess; subscription login / CLAUDE_CODE_OAUTH_TOKEN remain.
        options = ClaudeAgentOptions(
                model=self.model,
                max_turns=1 if not self.tool_schemas else 2,
                # TOOLS-OFF TRAP (root-caused 2026-07-03, live voice trace): these are
                # TWO different knobs. `allowed_tools` only controls AUTO-PERMISSION;
                # the CLI still exposes every built-in tool (Bash, Skill, ...) to the
                # model. `tools=[]` is what actually removes them. Without it, an
                # action-shaped prompt ("kill the office light" — reaches this chat
                # backend via the voice parallel-start's speculative generation) makes
                # the model CALL a tool; with max_turns=1 the turn then dies as
                # ResultMessage(subtype='error_max_turns', result=None) — on every
                # retry, deterministically, because it's the prompt, not a race.
                tools=[],               # chat backend only — no built-in tools exist
                allowed_tools=[],       # and nothing would be auto-permitted anyway
                permission_mode="default",
                env={"ANTHROPIC_API_KEY": ""},
                # Never load the host's ~/.claude settings (hooks, CLAUDE.md) into her.
                setting_sources=[],
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
        client = ClaudeSDKClient(options=options)
        await client.connect()
        return client

    async def _turn(self, prompt: str) -> tuple[str, list[dict], dict]:
        from claude_agent_sdk import AssistantMessage, ResultMessage
        from claude_agent_sdk.types import TextBlock, ToolUseBlock

        if self._client is None:
            self._client = await self._connect()
        # Fresh session per turn: warm process, cold context (see module doc).
        session = uuid.uuid4().hex
        await self._client.query(prompt, session_id=session)
        result_text: str | None = None
        assistant_text: list[str] = []
        tool_calls: list[dict] = []
        meta: dict = {"session_id": session, "model": self.model}
        async for message in self._client.receive_response():
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
                if getattr(message, "is_error", False):
                    # result is None on error subtypes — surface the fields that
                    # actually diagnose it (a bare "error: None" left us blind on
                    # the 2026-07-03 voice trace; never again).
                    raise RuntimeError(
                        "oauth backend error: "
                        f"subtype={getattr(message, 'subtype', None)!r} "
                        f"result={message.result!r} "
                        f"num_turns={getattr(message, 'num_turns', None)!r} "
                        f"stop_reason={getattr(message, 'stop_reason', None)!r} "
                        f"permission_denials={getattr(message, 'permission_denials', None)!r}"
                    )
                result_text = message.result
        if tool_calls:
            return "".join(assistant_text), tool_calls, meta
        return result_text or "".join(assistant_text) or "", [], meta

    async def _reset(self):
        try:
            if self._client is not None:
                await self._client.disconnect()
        except Exception:
            pass
        self._client = None

    def ask(self, prompt: str) -> tuple[str, list[dict], dict]:
        import concurrent.futures

        with self._lock:
            try:
                return self._run(self._turn(prompt))
            except concurrent.futures.TimeoutError as hung:
                # A hang is not retried: drop the process so the next caller starts
                # clean, and fail this turn now (the lifeboat/honest-failure path decides).
                try:
                    self._run(self._reset(), timeout=10)
                except Exception:
                    pass
                self._client = None
                raise OAuthBackendError(f"turn exceeded {self.turn_timeout_s:.0f}s; client reset") from hung
            except Exception as first:
                # Warm process may have died (idle timeout, OOM, upgrade) —
                # reconnect once, then let a second failure surface loudly.
                try:
                    self._run(self._reset(), timeout=10)
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
        out = self._query(_flatten(messages, force_tool=self.force_tool and bool(self.bound_tools)))
        if isinstance(out, str):  # tests may stub _query with a bare string
            text, tool_calls, meta = out, [], {}
        else:
            text, tool_calls, meta = out
        usage = (meta or {}).get("usage") or {}
        usage_metadata = None
        if usage:
            usage_metadata = {"input_tokens": int(usage.get("input_tokens", 0) or 0),
                              "output_tokens": int(usage.get("output_tokens", 0) or 0),
                              "total_tokens": int((usage.get("input_tokens", 0) or 0) + (usage.get("output_tokens", 0) or 0))}
        msg = AIMessage(content=text, tool_calls=tool_calls, usage_metadata=usage_metadata,
                        response_metadata={k: v for k, v in (meta or {}).items() if k != "usage"})
        return ChatResult(generations=[ChatGeneration(message=msg)])

    def _query(self, prompt: str):
        if self._warm is None:
            object.__setattr__(self, "_warm", warm_client_for(self.model, self.bound_tools, self.turn_timeout_s))
        return self._warm.ask(prompt)
