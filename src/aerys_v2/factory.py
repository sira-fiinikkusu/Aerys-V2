"""Builders for the model and the graph — construction lives here, behavior lives in the graph.

n8n mapping: this file is the "Load Config" node's job done properly. In n8n the model
choice, prompt, and wiring were assembled per-execution inside Code nodes; here they are
built ONCE at startup into objects the rest of the app calls. The graph is the workflow
canvas; each node function is a Code node that receives state instead of $json.
"""

from pathlib import Path

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from aerys_v2.config import Settings
from aerys_v2.state import ChatState, identity_from_config

FALLBACK_SOUL = "You are Aerys, a personal AI companion. Be warm, direct, and honest."


def load_soul(path: Path) -> str:
    """Read the persona prompt from disk, with a safe fallback.

    Same contract as n8n's Load Config reading soul.md via require('fs'): edit the file,
    restart nothing (n8n) / restart the service (here — cheap), no redeploy. A missing
    file degrades to a minimal persona instead of crashing the brain at startup.
    """
    try:
        text = path.read_text(encoding="utf-8").strip()
        return text if text else FALLBACK_SOUL
    except OSError:
        return FALLBACK_SOUL


def build_model(settings: Settings, *, timeout_s: float = 60.0) -> ChatAnthropic:
    """One place that knows how to turn Settings into a chat model.

    The request `timeout` is a safety rail (cross-review #13): a hung provider call
    must fail the turn, never hang the caller. max_tokens caps the spend per reply.
    """
    return ChatAnthropic(
        model=settings.model,
        api_key=settings.anthropic_api_key,  # SecretStr — unwrapped only by the client
        max_tokens=4096,
        timeout=timeout_s,
        max_retries=2,
    )


def build_graph(
    model: BaseChatModel,
    soul: str,
    checkpointer: BaseCheckpointSaver | None = None,
) -> object:
    """START → chat → END, checkpointed.

    The checkpointer is INJECTED (pluggable — cross-review #9): InMemorySaver for tests
    and the CLI today, PostgresSaver on the NAS when Phase 2 wires durability. The graph
    shape doesn't change when the storage does — that's the point of the seam.
    """

    def chat(state: ChatState, config: RunnableConfig) -> dict:
        # Identity comes from per-call config (the S2 channel), NEVER from state —
        # in a shared thread, checkpointed identity would leak user A onto user B
        # (the session-contamination bug Aerys V1 actually had).
        identity = identity_from_config(config)
        caller_line = (
            f"The current caller is {identity.get('display_name', 'Unknown Caller')}."
        )
        system = SystemMessage(content=f"{soul}\n\n{caller_line}")
        # n8n mapping: this is the AI Agent node's invoke — prompt + history in, one
        # AIMessage out. add_messages in ChatState appends it to the thread history.
        reply = model.invoke([system, *state["messages"]])
        return {"messages": [reply]}

    graph = StateGraph(ChatState)
    graph.add_node("chat", chat)
    graph.add_edge(START, "chat")
    graph.add_edge("chat", END)
    return graph.compile(checkpointer=checkpointer or InMemorySaver())
