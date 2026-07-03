"""OAuth-backed chat model — her daily words on the Max pool, not API tokens.

n8n mapping: this is swapping which credential the AI Agent node uses, except the
"credential" here is Chris's Claude subscription (the same auth Kael runs on). The
Claude Agent SDK spawns the Claude Code CLI under the hood; we use it as a PURE
chat backend: one turn, zero tools, system prompt passed through. LangGraph never
knows the difference — build_model() returns "a chat model" either way.

The June decision this cashes in: don't retrofit n8n Aerys with subscription auth;
bank it as a V2-Brain design choice. This module is that choice, landed.
"""

import asyncio
from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult


def _flatten(messages: list[BaseMessage]) -> tuple[str, str]:
    """Split LangChain messages into (system_prompt, transcript prompt).

    The SDK takes one prompt string per query, so history is serialized as a
    speaker-labeled transcript (same shape as thread_context snippets in n8n —
    and the same reason: the model reads attribution, it doesn't infer it).
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
    # The trailing "Aerys:" cue keeps the transcript framing coherent for the model.
    prompt = "\n".join(lines) + "\nAerys:"
    return "\n\n".join(system_parts), prompt


class ClaudeOAuthChatModel(BaseChatModel):
    """LangChain chat model backed by the Claude Agent SDK (subscription auth).

    Deliberately minimal: max_turns=1, no tools, no MCP — the agent loop belongs
    to LangGraph, not the SDK. When tool-calling lands (01-03+), that design fork
    gets navigated on purpose, not by accident (see CROSS-REVIEW / scoping notes).
    """

    model: str = "claude-opus-4-8"

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
        system, prompt = _flatten(messages)
        text = asyncio.run(self._query(system, prompt))
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])

    async def _query(self, system: str, prompt: str) -> str:
        from claude_agent_sdk import AssistantMessage, ClaudeAgentOptions, ResultMessage, query
        from claude_agent_sdk.types import TextBlock

        options = ClaudeAgentOptions(
            system_prompt=system,
            model=self.model,
            max_turns=1,
            allowed_tools=[],       # chat backend only — no file/bash/tool access
            permission_mode="default",
        )
        result_text: str | None = None
        assistant_text: list[str] = []
        async for message in query(prompt=prompt, options=options):
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        assistant_text.append(block.text)
            if isinstance(message, ResultMessage):
                # is_error covers refused/failed runs; surface loudly, never blank.
                if getattr(message, "is_error", False):
                    raise RuntimeError(f"oauth backend error: {message.result!r}")
                result_text = message.result
        return result_text or "".join(assistant_text) or ""
