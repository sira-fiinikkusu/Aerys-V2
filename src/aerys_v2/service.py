"""ask() — the single seam every transport calls.

n8n mapping: this is the Execute Workflow boundary into the Core Agent, done as one
function. Discord, Telegram, the voice endpoint, and the CLI will ALL call ask() and
nothing else — so safety rails, auditing, and tracing added here cover every channel
at once (in n8n the same fix had to be patched into each adapter separately).
"""

import time
from dataclasses import dataclass

from langchain_core.messages import HumanMessage

from aerys_v2.state import Identity


@dataclass(frozen=True)
class Rails:
    """Per-request safety limits (cross-review #13) — enforced at the seam, not by prompts.

    turn_limit is dormant until the ToolNode loop lands (01-03+): with tools wired, a
    confused model can loop tool-call → result → tool-call forever; the rail makes the
    10th hop a hard stop instead of an Opus-budget incineration.
    """

    wall_clock_s: float = 90.0
    turn_limit: int = 10


class TurnTimeout(RuntimeError):
    """The whole turn (not just one model call) exceeded its wall-clock budget."""


def ask(
    graph: object,
    text: str,
    *,
    identity: Identity,
    thread_id: str,
    rails: Rails = Rails(),
) -> str:
    """Run one conversational turn and return the reply text.

    - identity rides `configurable` (the S2 channel) — per-call, never checkpointed.
    - thread_id selects the conversation; the checkpointer replays its history.
    - recursion_limit is the LangGraph-native turn_limit enforcement: each graph
      super-step counts, so a runaway tool loop trips it long before infinity.
    """
    if not text or not text.strip():
        raise ValueError("ask() requires non-empty text")

    started = time.monotonic()
    config = {
        "configurable": {"thread_id": thread_id, "identity": identity},
        "recursion_limit": rails.turn_limit,
    }
    result = graph.invoke({"messages": [HumanMessage(content=text)]}, config)

    elapsed = time.monotonic() - started
    if elapsed > rails.wall_clock_s:
        # The reply exists but arrived past budget — surface it loudly rather than
        # silently normalizing a degraded experience (voice cares at ~4s, not 90).
        raise TurnTimeout(f"turn took {elapsed:.1f}s (budget {rails.wall_clock_s}s)")

    reply = result["messages"][-1]
    return reply.text() if callable(getattr(reply, "text", None)) else str(reply.content)
