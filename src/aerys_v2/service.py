"""ask() — the single seam every transport calls.

n8n mapping: this is the Execute Workflow boundary into the Core Agent, done as one
function. Discord, Telegram, the voice endpoint, and the CLI will ALL call ask() and
nothing else — so safety rails, auditing, and tracing added here cover every channel
at once (in n8n the same fix had to be patched into each adapter separately).

TOOLS block (Option C hybrid, owner-ratified): ask() optionally takes a router and
an action subgraph. Both None = chat-only, byte-for-byte the old behavior. Both set:

- non-voice threads: router first (sequential) — chat routes to the chat graph,
  action routes to the tool subgraph, whose result becomes the reply.
- voice threads (thread_id startswith "voice"): PARALLEL-START — the router and
  the chat generation launch concurrently. Router says chat -> the chat result
  (already in flight) is the reply and the router cost vanishes into the
  latency shadow. Router says action -> the caller gets the router's generated
  ack IMMEDIATELY (speakable now, ~3.6s budget intact) while a background
  thread finishes the action and appends the real result to the SAME thread —
  so the next turn's history shows what actually happened, not just the ack.
"""

import concurrent.futures
import logging
import threading
import time
from dataclasses import dataclass
from typing import Callable

from langchain_core.messages import AIMessage, HumanMessage

from aerys_v2.router import RouteDecision
from aerys_v2.state import Identity

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Rails:
    """Per-request safety limits (cross-review #13) — enforced at the seam, not by prompts.

    turn_limit went live with the TOOLS block: with tools wired, a confused model
    can loop tool-call → result → tool-call forever; the rail (as LangGraph's
    recursion_limit) makes the 10th hop a hard stop instead of an Opus-budget
    incineration.
    """

    wall_clock_s: float = 90.0
    turn_limit: int = 10


class TurnTimeout(RuntimeError):
    """The whole turn (not just one model call) exceeded its wall-clock budget."""


def _reply_text(message: object) -> str:
    # .text is a property in current langchain-core (calling it is deprecated)
    text_attr = getattr(message, "text", None)
    return text_attr if isinstance(text_attr, str) else str(message.content)


def ask(
    graph: object,
    text: str,
    *,
    identity: Identity,
    thread_id: str,
    rails: Rails = Rails(),
    router: Callable[[str], RouteDecision] | None = None,
    action_graph: object | None = None,
) -> str:
    """Run one conversational turn and return the reply text.

    - identity rides `configurable` (the S2 channel) — per-call, never checkpointed.
    - thread_id selects the conversation; the checkpointer replays its history.
    - recursion_limit is the LangGraph-native turn_limit enforcement: each graph
      super-step counts, so a runaway tool loop trips it long before infinity.
    - router + action_graph arm the TOOLS block (see module docstring); either
      missing = the pre-TOOLS chat-only path, unchanged.
    """
    if not text or not text.strip():
        raise ValueError("ask() requires non-empty text")

    started = time.monotonic()
    config = {
        "configurable": {"thread_id": thread_id, "identity": identity},
        "recursion_limit": rails.turn_limit,
    }

    if router is None or action_graph is None:
        return _chat_turn(graph, text, config, rails, started)

    if str(thread_id).startswith("voice"):
        return _voice_parallel_start(
            graph, text, config, rails, started, router, action_graph
        )

    # Non-voice: nobody is waiting on a speaker, so the router runs first
    # (sequential) and only the chosen path spends model tokens.
    decision = router(text)
    if decision.route == "action":
        # add_human=True: the chat graph never saw this turn, so BOTH the human
        # message and the action result must land in the thread history.
        return _action_turn(action_graph, graph, text, config, add_human=True)
    return _chat_turn(graph, text, config, rails, started)


def _chat_turn(graph: object, text: str, config: dict, rails: Rails, started: float) -> str:
    """The original chat path: invoke, budget-check, extract."""
    result = graph.invoke({"messages": [HumanMessage(content=text)]}, config)

    elapsed = time.monotonic() - started
    if elapsed > rails.wall_clock_s:
        # The reply exists but arrived past budget — surface it loudly rather than
        # silently normalizing a degraded experience (voice cares at ~4s, not 90).
        raise TurnTimeout(f"turn took {elapsed:.1f}s (budget {rails.wall_clock_s}s)")

    return _reply_text(result["messages"][-1])


def _action_turn(
    action_graph: object, graph: object, text: str, config: dict, *, add_human: bool
) -> str:
    """Run the tool subgraph, then land the outcome in the MAIN thread's history.

    The action graph is checkpointer-less (one-shot); update_state on the chat
    graph is how the outcome becomes durable conversation history — next turn,
    the chat model sees "I turned the office light on" as its own prior message
    instead of a hole where an action happened. as_node="chat" attributes the
    write to the node that normally speaks.
    """
    result = action_graph.invoke({"messages": [HumanMessage(content=text)]}, config)
    final = _reply_text(result["messages"][-1])
    messages: list = [AIMessage(content=final)]
    if add_human:
        messages.insert(0, HumanMessage(content=text))
    graph.update_state(
        {"configurable": config["configurable"]}, {"messages": messages}, as_node="chat"
    )
    return final


def _voice_parallel_start(
    graph: object,
    text: str,
    config: dict,
    rails: Rails,
    started: float,
    router: Callable[[str], RouteDecision],
    action_graph: object,
) -> str:
    """Voice hot path: race the router against the chat generation.

    Two threads (concurrent.futures): the router is ~300ms of Haiku, the chat
    generation is seconds of the daily driver. Both start NOW; the router's
    verdict decides which one the caller ever hears about.
    """
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=2)
    try:
        route_future = pool.submit(router, text)
        chat_future = pool.submit(
            graph.invoke, {"messages": [HumanMessage(content=text)]}, config
        )
        decision = route_future.result()

        if decision.route != "action":
            # Chat wins: the generation is already in flight — the router's
            # latency hid entirely inside the chat call's shadow. Discard it.
            result = chat_future.result()
            elapsed = time.monotonic() - started
            if elapsed > rails.wall_clock_s:
                raise TurnTimeout(
                    f"turn took {elapsed:.1f}s (budget {rails.wall_clock_s}s)"
                )
            return _reply_text(result["messages"][-1])

        # Action: the ack goes out NOW; the tool loop finishes in the background.
        # Best-effort cancel of the speculative chat call — if it already
        # started (fake models finish instantly; real ones usually haven't
        # begun streaming), its reply simply lands in the thread and the action
        # result lands after it, superseding it. Slightly chatty history beats
        # blocking the ack, and the checkpointer race is avoided by waiting for
        # the chat future before writing (below).
        chat_cancelled = chat_future.cancel()

        def _complete_action() -> None:
            try:
                result = action_graph.invoke(
                    {"messages": [HumanMessage(content=text)]}, config
                )
                final = _reply_text(result["messages"][-1])
            except Exception as e:  # honest failure into history, never silence
                log.warning("background action turn failed", exc_info=True)
                final = f"(The action didn't complete — {e})"
            if not chat_cancelled:
                # The speculative chat invoke owns the checkpoint until it
                # finishes — wait so our update_state can't interleave with it.
                try:
                    chat_future.result()
                except Exception:
                    pass
            messages: list = [AIMessage(content=final)]
            if chat_cancelled:
                # chat never ran -> the human turn isn't in the thread yet
                messages.insert(0, HumanMessage(content=text))
            graph.update_state(
                {"configurable": config["configurable"]},
                {"messages": messages},
                as_node="chat",
            )

        threading.Thread(target=_complete_action, daemon=True).start()
        return decision.ack
    finally:
        # Never block the reply on stragglers — the background thread (plain
        # threading.Thread, not pool-owned) outlives this scope by design.
        pool.shutdown(wait=False)
