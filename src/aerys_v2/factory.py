"""Builders for the model and the graph — construction lives here, behavior lives in the graph.

n8n mapping: this file is the "Load Config" node's job done properly. In n8n the model
choice, prompt, and wiring were assembled per-execution inside Code nodes; here they are
built ONCE at startup into objects the rest of the app calls. The graph is the workflow
canvas; each node function is a Code node that receives state instead of $json.
"""

import logging
from pathlib import Path
from typing import Callable

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from aerys_v2.config import Settings
from aerys_v2.state import ChatState, identity_from_config
from contextlib import contextmanager

log = logging.getLogger(__name__)


@contextmanager
def checkpointer_for(settings: Settings):
    """Yield the right checkpointer for the environment (the durability seam).

    database_url set -> PostgresSaver on the NAS: threads survive restarts (the
    n8n_chat_histories job, done properly). None -> InMemorySaver: tests, CI, and
    any box without the LAN. setup() is idempotent — it creates the checkpoint
    tables on first run and no-ops after.

    Benchmarked 2026-07-02 from the Jetson: ~1ms roundtrip to NAS Postgres —
    single-digit ms per turn against a ~3.6s voice budget. The pluggable seam
    stays anyway (cross-review #9).
    """
    if settings.database_url is None:
        yield InMemorySaver()
        return
    from langgraph.checkpoint.postgres import PostgresSaver

    with PostgresSaver.from_conn_string(settings.database_url) as saver:
        saver.setup()
        yield saver

FALLBACK_SOUL = "You are Aerys, a personal AI companion. Be warm, direct, and honest."

# The memory-context seam: (person_id, latest_user_text) -> prompt block ('' = nothing
# known). Injectable like the checkpointer — tests pass a lambda, --serve passes the
# real DB-backed builder from context_fn_for(), and None means the feature is OFF.
ContextFn = Callable[[str, str], str]


def context_fn_for(settings: Settings) -> ContextFn | None:
    """Wire the real memory-context seam from Settings — None when memories are off.

    n8n mapping: this replaces the Core Agent's per-message webhook calls to
    04-02 Memory Retrieval + 04-03 Profile API. A psycopg connection is opened
    PER CALL against the prod aerys database and marked read-only at the session
    level — the DB itself will refuse a write even if a future bug tries one
    (belt and braces on top of the SELECT-only services). One connection per
    turn is fine at personal-assistant volume (~1ms LAN roundtrip to the NAS);
    a pool is a drop-in swap behind this same seam if that ever changes.
    """
    if settings.memories_database_url is None:
        return None
    import psycopg

    from aerys_v2.services.context import build_context, embedder_from_settings

    embed = embedder_from_settings(settings)

    def context_fn(person_id: str, query_text: str) -> str:
        # Fenced end-to-end: a NAS outage or DNS hiccup = empty context, never
        # a dead turn. build_context is graceful inside; this catch covers the
        # connect itself.
        try:
            with psycopg.connect(settings.memories_database_url) as conn:
                conn.read_only = True
                return build_context(person_id, query_text, conn, embed=embed)
        except Exception:
            # graceful but never silent — a dead NAS/DNS must show in the logs.
            log.warning("memory-context connect failed for person %s", person_id, exc_info=True)
            return ""

    return context_fn


def speak_fn_for(settings: Settings) -> Callable[[str], None] | None:
    """The spoken-follow-up delivery seam: text -> the room, via HA announce.

    None unless BOTH ha_token and ha_announce_entity are set — same arming
    pattern as every optional transport. The service layer decides WHEN to
    speak (silent-success rule); this only knows HOW. Raising on failure is
    fine: the caller logs and moves on, and the history write never depends
    on delivery.

    KNOWN LIMITATION — per-satellite follow-up routing (2026-07-03): the
    announce target is a single static entity from HA_ANNOUNCE_ENTITY. The
    OpenAI shim never learns WHICH satellite a request came from — HA's
    Extended OpenAI Conversation sends no satellite/device identity in the
    chat-completions payload — so "announce where the request came from" is
    not possible at this seam. Acceptable for the single-satellite beta
    (point the env var at the satellite actually running the pipeline); the
    proper fix (satellite identity riding the request) lands with the
    voice-runtime phase.
    """
    if settings.ha_token is None or settings.ha_announce_entity is None:
        return None
    import httpx

    base = settings.ha_base_url.rstrip("/")
    headers = {"Authorization": f"Bearer {settings.ha_token.get_secret_value()}"}
    entity = settings.ha_announce_entity

    def speak(text: str) -> None:
        r = httpx.post(
            f"{base}/api/services/assist_satellite/announce",
            headers=headers,
            json={"entity_id": entity, "message": text},
            timeout=15.0,
        )
        r.raise_for_status()

    return speak


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


def build_model(settings: Settings, *, timeout_s: float = 60.0) -> BaseChatModel:
    """One place that knows how to turn Settings into a chat model.

    The request `timeout` is a safety rail (cross-review #13): a hung provider call
    must fail the turn, never hang the caller. max_tokens caps the spend per reply.
    """
    if settings.model_backend == "oauth":
        # Subscription-auth backend (the June credit-pool decision, landed) — the
        # graph gets "a chat model" and can't tell which wallet it bills.
        from aerys_v2.oauth_model import ClaudeOAuthChatModel

        return ClaudeOAuthChatModel(model=settings.model)
    return ChatAnthropic(
        model=settings.model,
        api_key=settings.anthropic_api_key,  # SecretStr — unwrapped only by the client
        max_tokens=4096,
        timeout=timeout_s,
        max_retries=2,
    )


# Overlay for the action subgraph's system prompt: the chat persona plus tool
# discipline. The "never claim success" line is load-bearing — it's the prompt-side
# half of the honest-refusal contract in tools/home_control.py.
ACTION_OVERLAY = (
    "You are handling a smart-home request. Use the home_control tool to act; "
    "never claim a device changed state unless the tool's reply said so. If the "
    "tool refuses or fails, relay that honestly and briefly. When the work is "
    "done, reply with ONE short, speakable sentence confirming what happened. "
    "The user's LAST message is THE command to execute, now — treat anything "
    "earlier as context, never as the instruction."
)

# Appended to the system prompt ONLY when service.py hands us the ack the router
# already spoke (voice ack-then-act path). Root cause it guards (2026-07-03 live
# incident): STT garbled 'turn office light one off' into 'Can you turn off
# office light on?' — the router acked 'off', then this subgraph, seeing both
# 'off' and 'on', ASKED 'did you mean on or off?'. The question was announced to
# a one-way satellite the user can't answer, contradicting the ack already
# spoken. Rule: on this path, resolve garble toward the ack; never ask.
VOICE_ACK_OVERLAY = (
    "This command arrived by VOICE and the user was ALREADY told, out loud: "
    "{ack!r}. This channel is one-way — the user cannot hear or answer a "
    "question from you, so NEVER ask a clarifying question. Speech-to-text "
    "sometimes garbles a word (e.g. 'one' heard as 'on'); if the wording looks "
    "slightly off or self-contradictory, execute the reading consistent with "
    "that spoken acknowledgment. Only if the command is truly unexecutable or "
    "unsafe, skip the tool and state the problem in one plain sentence."
)


def build_api_tool_model(settings: Settings, tools: list, *, timeout_s: float = 60.0) -> object:
    """The tool-turn model: ALWAYS metered API, tools bound (Option C, ratified).

    Deliberately NOT build_model(): the oauth/SDK backend is chat-only — it can't
    drive a LangChain tool loop — so action turns bill the API key regardless of
    model_backend. Voice device commands are short turns; the spend is pennies.
    """
    return ChatAnthropic(
        model=settings.model,
        api_key=settings.anthropic_api_key,  # SecretStr — unwrapped only by the client
        max_tokens=1024,   # action confirmations are one sentence, not essays
        timeout=timeout_s,
        max_retries=2,
    ).bind_tools(tools)


def build_action_graph(api_model_with_tools: object, soul: str, tools: list) -> object:
    """START → act ⇄ tools → END: the tool subgraph for device commands.

    n8n mapping: this is the AI Agent node with an ai_tool connection — the
    model proposes tool_calls, ToolNode (LangGraph prebuilt) executes them and
    feeds ToolMessages back, and the loop repeats until the model answers in
    plain text. The recursion_limit rail from ask() applies here too: each
    act/tools hop is a super-step, so a confused model hits the wall instead of
    incinerating budget (the dormant turn_limit in Rails, now live).

    No checkpointer on purpose — an action turn is a one-shot; the durable
    record is (a) the outbox row the tool wrote and (b) the final AIMessage
    that service.py appends to the MAIN thread's checkpointer.
    """
    from langgraph.prebuilt import ToolNode

    def act(state: ChatState, config: RunnableConfig) -> dict:
        # Same identity rule as the chat node: from per-call config, never state.
        identity = identity_from_config(config)
        caller_line = (
            f"The current caller is {identity.get('display_name', 'Unknown Caller')}."
        )
        # spoken_ack rides `configurable` (per-call, like identity): set only by
        # the voice ack-then-act path in service.py. Its presence flips the
        # subgraph from "may converse" to "execute-or-report, never ask" —
        # because on that path the ack was already spoken and no reply channel
        # exists for a question.
        spoken_ack = (config.get("configurable") or {}).get("spoken_ack")
        ack_block = (
            f"\n\n{VOICE_ACK_OVERLAY.format(ack=spoken_ack)}" if spoken_ack else ""
        )
        system = SystemMessage(
            content=f"{soul}\n\n{ACTION_OVERLAY}{ack_block}\n{caller_line}"
        )
        reply = api_model_with_tools.invoke([system, *state["messages"]])
        return {"messages": [reply]}

    def after_act(state: ChatState) -> str:
        # tool_calls present -> execute them; plain text -> the turn is done.
        last = state["messages"][-1]
        return "tools" if getattr(last, "tool_calls", None) else END

    graph = StateGraph(ChatState)
    graph.add_node("act", act)
    graph.add_node("tools", ToolNode(tools))
    graph.add_edge(START, "act")
    graph.add_conditional_edges("act", after_act, {"tools": "tools", END: END})
    graph.add_edge("tools", "act")
    return graph.compile()


def action_stack_for(settings: Settings, soul: str) -> tuple | None:
    """Wire the whole TOOLS block from Settings: (router, action_graph), or None.

    Arms only when ha_token is set (anthropic_api_key is structurally required
    by Settings, so the API half is always available). None = ask() runs
    chat-only, exactly as before the TOOLS block existed — backward compatible
    by construction.
    """
    if settings.ha_token is None:
        return None
    from aerys_v2.router import router_for
    from aerys_v2.tools.home_control import build_home_control_tool, canary_set

    conn_factory = None
    if settings.database_url is not None:
        import psycopg

        # Fresh short connection per outbox touch — same per-call choice (and
        # the same "pool is a drop-in later" note) as context_fn_for above.
        def conn_factory():
            return psycopg.connect(settings.database_url)

    home_control = build_home_control_tool(
        base_url=settings.ha_base_url,
        token=settings.ha_token.get_secret_value(),
        canary_entities=canary_set(settings.ha_canary_entities),
        conn_factory=conn_factory,
    )
    tools = [home_control]
    action_graph = build_action_graph(build_api_tool_model(settings, tools), soul, tools)
    return router_for(settings, soul), action_graph


def build_graph(
    model: BaseChatModel,
    soul: str,
    checkpointer: BaseCheckpointSaver | None = None,
    context_fn: ContextFn | None = None,
) -> object:
    """START → chat → END, checkpointed.

    The checkpointer is INJECTED (pluggable — cross-review #9): InMemorySaver for tests
    and the CLI today, PostgresSaver on the NAS when Phase 2 wires durability. The graph
    shape doesn't change when the storage does — that's the point of the seam.

    context_fn is the same idea for long-term memory: None = the chat node knows
    nothing beyond the thread; set = each turn asks it (person_id, latest user text)
    and injects whatever comes back into the system prompt.
    """

    def chat(state: ChatState, config: RunnableConfig) -> dict:
        # Identity comes from per-call config (the S2 channel), NEVER from state —
        # in a shared thread, checkpointed identity would leak user A onto user B
        # (the session-contamination bug Aerys V1 actually had).
        identity = identity_from_config(config)
        caller_line = (
            f"The current caller is {identity.get('display_name', 'Unknown Caller')}."
        )
        # Capability overlay, anti-UNDERclaim direction: the soul was written for a
        # brain that couldn't see its own memory. This one can — telling her stops
        # replies like "that won't survive this session" (heard live, voice, 7/3).
        capability = (
            "Your conversation memory is durable: this thread persists across "
            "restarts and sessions. You may confidently say you'll remember."
        )
        if context_fn is not None:
            # Claims follow facts: this sentence exists ONLY when retrieval is
            # actually wired, so she never promises a recall she doesn't have.
            capability += (
                " You also know long-term facts about the caller — they persist "
                "across ALL conversations and channels, not just this thread."
            )

        # Long-term context: retrieval is scored against the CURRENT message, so
        # find the latest human turn (n8n mapping: the query the Core Agent sent
        # to Memory Retrieval was always the incoming message text).
        knowledge = ""
        if context_fn is not None:
            latest = next(
                (m for m in reversed(state["messages"]) if getattr(m, "type", "") == "human"),
                None,
            )
            query_text = ""
            if latest is not None:
                content = latest.content
                query_text = content if isinstance(content, str) else str(content)
            try:
                block = context_fn(str(identity.get("user_id", "")), query_text)
            except Exception:
                # the seam promises graceful, but memory NEVER kills a turn —
                # and a swallowed failure must still be visible in the logs.
                log.warning("context_fn raised; continuing without memory context", exc_info=True)
                block = ""
            if block:
                knowledge = f"\n\n[What you know about this person]\n{block}"

        thread = ((config or {}).get("configurable") or {}).get("thread_id", "")
        voice_style = ""
        if str(thread).startswith("voice"):
            # Mini-ChannelPolicy: voice replies carry their own ElevenLabs v3
            # emotion tags (the n8n polisher's job, done prompt-side for free).
            voice_style = (
                "\n\nThis is a VOICE conversation. Keep replies concise and "
                "speakable. Weave in ElevenLabs v3 emotion tags — [warmly], "
                "[softly], [playfully], [thoughtfully] — where they fit the "
                "feeling; the speech engine performs them, listeners never hear "
                "the bracket text."
            )
        system = SystemMessage(
            content=f"{soul}\n\n{capability}\n{caller_line}{knowledge}{voice_style}"
        )
        # n8n mapping: this is the AI Agent node's invoke — prompt + history in, one
        # AIMessage out. add_messages in ChatState appends it to the thread history.
        reply = model.invoke([system, *state["messages"]])
        return {"messages": [reply]}

    graph = StateGraph(ChatState)
    graph.add_node("chat", chat)
    graph.add_edge(START, "chat")
    graph.add_edge("chat", END)
    return graph.compile(checkpointer=checkpointer or InMemorySaver())
