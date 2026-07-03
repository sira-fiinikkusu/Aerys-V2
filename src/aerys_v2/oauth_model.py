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
import threading
import uuid
from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult


def _flatten(messages: list[BaseMessage]) -> str:
    """Serialize LangChain messages into one speaker-labeled prompt.

    System content leads, history follows labeled (same shape as thread_context
    snippets in n8n — the model reads attribution, it doesn't infer it).
    """
    system_parts: list[str] = []
    lines: list[str] = []
    for m in messages:
        if isinstance(m, SystemMessage):
            system_parts.append(str(m.content))
        elif isinstance(m, HumanMessage):
            lines.append(f"User: {m.content}")
        elif isinstance(m, AIMessage):
            lines.append(f"Aerys: {m.content}")
    head = ("[System instructions]\n" + "\n\n".join(system_parts) + "\n\n") if system_parts else ""
    return head + "\n".join(lines) + "\nAerys:"


class _WarmClient:
    """Owns the event-loop thread + the connected ClaudeSDKClient (one per model)."""

    def __init__(self, model: str) -> None:
        self.model = model
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True, name="oauth-model")
        self._thread.start()
        self._client = None
        self._lock = threading.Lock()  # one turn at a time — household-sized

    def _run(self, coro):
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result(timeout=120)

    async def _connect(self):
        from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient

        # AUTH PRECEDENCE TRAP (found 2026-07-03): if ANTHROPIC_API_KEY exists in
        # the process env, the spawned CLI prefers it over subscription auth — the
        # container would silently bill the API while claiming oauth. Neutralize it
        # for the subprocess; subscription login / CLAUDE_CODE_OAUTH_TOKEN remain.
        client = ClaudeSDKClient(
            options=ClaudeAgentOptions(
                model=self.model,
                max_turns=1,
                allowed_tools=[],       # chat backend only — no file/bash/tool access
                permission_mode="default",
                env={"ANTHROPIC_API_KEY": ""},
            )
        )
        await client.connect()
        return client

    async def _turn(self, prompt: str) -> str:
        from claude_agent_sdk import AssistantMessage, ResultMessage
        from claude_agent_sdk.types import TextBlock

        if self._client is None:
            self._client = await self._connect()
        # Fresh session per turn: warm process, cold context (see module doc).
        session = uuid.uuid4().hex
        await self._client.query(prompt, session_id=session)
        result_text: str | None = None
        assistant_text: list[str] = []
        async for message in self._client.receive_response():
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        assistant_text.append(block.text)
            if isinstance(message, ResultMessage):
                if getattr(message, "is_error", False):
                    raise RuntimeError(f"oauth backend error: {message.result!r}")
                result_text = message.result
        return result_text or "".join(assistant_text) or ""

    async def _reset(self):
        try:
            if self._client is not None:
                await self._client.disconnect()
        except Exception:
            pass
        self._client = None

    def ask(self, prompt: str) -> str:
        with self._lock:
            try:
                return self._run(self._turn(prompt))
            except Exception:
                # Warm process may have died (idle timeout, OOM, upgrade) —
                # reconnect once, then let a second failure surface loudly.
                self._run(self._reset())
                return self._run(self._turn(prompt))


class ClaudeOAuthChatModel(BaseChatModel):
    """LangChain chat model backed by a warm Claude Agent SDK client.

    Deliberately minimal: one turn, no tools, no MCP — the agent loop belongs to
    LangGraph. When tool-calling lands (01-03+), the SDK-loop-vs-ToolNode fork
    gets navigated on purpose, not by accident (see CROSS-REVIEW).
    """

    model: str = "claude-opus-4-8"

    _warm: Any = None  # lazily created _WarmClient (pydantic private-ish)

    @property
    def _llm_type(self) -> str:
        return "claude-oauth-sdk"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        text = self._query(_flatten(messages))
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])

    def _query(self, prompt: str) -> str:
        if self._warm is None:
            object.__setattr__(self, "_warm", _WarmClient(self.model))
        return self._warm.ask(prompt)
