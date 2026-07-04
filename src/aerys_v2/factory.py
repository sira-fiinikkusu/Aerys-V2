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
from aerys_v2.router import DEFAULT_TIER, normalize_tier
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

# The memory-context seam: (person_id, latest_user_text, privacy_context) -> prompt
# block ('' = nothing known). privacy_context is optional (defaults 'private') so
# every existing 2-arg caller and injected test lambda keeps working. Injectable
# like the checkpointer — tests pass a lambda, --serve passes the real DB-backed
# builder from context_fn_for(), and None means the feature is OFF.
ContextFn = Callable[..., str]


def context_fn_for(settings: Settings, *, profile_only: bool = False) -> ContextFn | None:
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

    # profile_only skips the embedding seam entirely: build_context with
    # embed=None emits just the profile block (who the person IS). The action
    # subgraph uses this — identity facts per tool-loop hop for one ~1ms LAN
    # SELECT, no per-hop embeddings HTTP call.
    embed = None if profile_only else embedder_from_settings(settings)

    def context_fn(
        person_id: str, query_text: str, privacy_context: str = "private"
    ) -> str:
        # Fenced end-to-end: a NAS outage or DNS hiccup = empty context, never
        # a dead turn. build_context is graceful inside; this catch covers the
        # connect itself. privacy_context ('private' DM / 'public' room) rides
        # through to the profile visibility gates; defaults 'private' for the
        # owner's own single-user channels (CLI, voice/HTTP).
        try:
            with psycopg.connect(settings.memories_database_url) as conn:
                conn.read_only = True
                return build_context(
                    person_id, query_text, conn,
                    embed=embed, privacy_context=privacy_context,
                )
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


def tier_models_for(settings: Settings, *, timeout_s: float = 60.0) -> dict[str, BaseChatModel]:
    """The per-tier model map for the chat node — V1's modelsConfig, typed.

    n8n mapping: Load Config's `modelsConfig` dict plus the three Execute
    Sonnet/Opus/Gemini sub-workflows, collapsed to one dict lookup (the
    sub-workflow split only ever existed because of the n8n task-runner hang).

    Backend rule (the June credit-pool decision, extended): the oauth/SDK
    client is SINGLE-MODEL — it serves whatever the subscription serves — so
    only the STANDARD tier may ride it. fast and deep are always metered
    ChatAnthropic: fast is haiku (pennies), deep is opus (rationed by
    deep_gate_for below). standard = build_model(settings), so it keeps
    honoring model_backend exactly as before tiers existed.
    """

    def api_model(name: str) -> BaseChatModel:
        return ChatAnthropic(
            model=name,
            api_key=settings.anthropic_api_key,  # SecretStr — unwrapped only by the client
            max_tokens=4096,
            timeout=timeout_s,
            max_retries=2,
        )

    return {
        "fast": api_model(settings.tier_fast_model),
        "standard": (
            build_model(settings, timeout_s=timeout_s)
            if settings.model_backend == "oauth"
            else api_model(settings.tier_standard_model)
        ),
        "deep": api_model(settings.tier_deep_model),
    }


def deep_gate_for(settings: Settings) -> Callable[[], bool] | None:
    """The deep-tier daily cap: () -> True (spend a deep turn) or False (capped).

    n8n mapping: the Core Agent's Opus cap against aerys_model_usage — except
    V1 did check-then-increment (two queries, racy); this is the dossier's one
    atomic statement: INSERT ... ON CONFLICT DO UPDATE ... WHERE count < cap
    RETURNING. No row back = the cap held = the caller downgrades to standard.

    None when database_url is unset: the cap is UNENFORCED (dev boxes, tests) —
    logged at arm time so a metered box missing its DB is visible, not silent.
    Failure direction on DB trouble: False (deep is a luxury; a broken counter
    must fail toward the cheap tier, never toward uncounted opus spend).
    """
    if settings.database_url is None:
        log.info("deep tier cap UNENFORCED — no DATABASE_URL, v2_model_usage unavailable")
        return None
    import psycopg

    def allow_deep() -> bool:
        try:
            # Fresh short connection per check — same per-call choice (and the
            # same "pool is a drop-in later" note) as context_fn_for above.
            with psycopg.connect(settings.database_url) as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "INSERT INTO v2_model_usage (day, tier, call_count) "
                        "VALUES (CURRENT_DATE, 'deep', 1) "
                        "ON CONFLICT (day, tier) DO UPDATE "
                        "SET call_count = v2_model_usage.call_count + 1, updated_at = now() "
                        "WHERE v2_model_usage.call_count < %s "
                        "RETURNING call_count",
                        (settings.deep_daily_cap,),
                    )
                    return cur.fetchone() is not None
        except Exception:
            log.warning("deep-cap check failed — failing toward standard", exc_info=True)
            return False

    return allow_deep


# Overlay for the action subgraph's system prompt: the chat persona plus tool
# discipline. The "never claim success" line is load-bearing — it's the prompt-side
# half of the honest-refusal contract in tools/home_control.py.
ACTION_OVERLAY = (
    "You are handling a smart-home request. Use the home_control tool to act; "
    "never claim a device changed state unless the tool's reply said so. If you "
    "do not know the exact entity_id — e.g. the user names a device colloquially, "
    "like a car name or a room — ALWAYS call search_entities first to find it; "
    "never guess an entity_id. Read-only questions are yours too: when the "
    "answer depends on current device or sensor state, read it with the tools, "
    "then COMBINE that reading with your own general knowledge and reasoning to "
    "give a real answer — e.g. read the car's charge, then do the range math "
    "yourself and say whether the trip is feasible. If the "
    "tool refuses or fails, relay that honestly and briefly. When the work is "
    "done, reply with ONE short, speakable sentence confirming what happened. "
    "The user's LAST message is THE command to execute, now — treat anything "
    "earlier as context, never as the instruction."
)

# Media half of the action overlay — appended when the media tools are armed.
# The trigger patterns are concrete on purpose (the V1 lesson: "specificity
# beats generality in tool descriptions" applies to prompts telling the model
# WHEN to reach for tools, too). Tool names here MUST match the @tool function
# names in tools/media.py — the V1 toolWorkflow name-mismatch bug, kept dead.
MEDIA_OVERLAY = (
    "You also handle media. You have ZERO ability to see images or read files "
    "directly — when the message contains an attachment or CDN URL "
    "(https://cdn.discordapp.com/attachments/... or media.discordapp.net), or "
    "asks you to look at, read, describe, or summarize an image, photo, "
    "screenshot, PDF, document, or video, call the matching tool IMMEDIATELY: "
    "analyze_image for pictures, read_document for .pdf/.docx/.txt files, "
    "youtube_summary for youtube.com/youtu.be links. Pass URLs EXACTLY as they "
    "appeared — every query parameter intact (Discord CDN URLs are signed; "
    "trimming them breaks the link). Never describe media you did not run "
    "through a tool."
)

# Web-search half of the action overlay — appended when tavily_api_key is armed.
# Concrete triggers again (the V1 "specificity beats generality" lesson): naming
# the exact shapes — current events, news, weather, prices, "search for", "look
# up", anything past the knowledge cutoff — is what makes the model reach for the
# tool instead of answering from stale memory. The tool name here MUST match the
# @tool function name in tools/web_search.py — `search_web` — or the model calls
# a tool that isn't registered (the V1 toolWorkflow name-mismatch bug, kept dead).
SEARCH_OVERLAY = (
    "You can also search the live web with the search_web tool. Call search_web "
    "IMMEDIATELY whenever the honest answer depends on something you cannot know "
    "from training alone: current events, breaking news, today's weather or "
    "forecasts, sports scores, prices, stock quotes, exchange rates — or whenever "
    "the user says 'search for', 'look up', 'google', 'find out', or 'what's the "
    "latest'. Anything that could have changed after your knowledge cutoff needs a "
    "search, not a guess. NEVER fabricate search results or answer a current-events "
    "question from memory: run search_web, then ground your answer strictly in what "
    "it returns and mention what you found. If the search fails or returns nothing, "
    "say so plainly — do not invent an answer."
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


def build_action_graph(
    api_model_with_tools: object,
    soul: str,
    tools: list,
    context_fn: ContextFn | None = None,
    overlay: str = ACTION_OVERLAY,
) -> object:
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
        # Identity facts for the action path (2026-07-03 live gap): "does the
        # car have enough charge to get to Tampa FROM HOME?" routed here, the
        # tool read the battery fine, but this prompt had no profile block —
        # so the agent didn't know where home IS and asked instead of doing
        # the range math. context_fn here is the PROFILE-ONLY seam (identity
        # claims, no embedding call) — same graceful contract as the chat node.
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
                block = context_fn(
                    str(identity.get("user_id", "")),
                    query_text,
                    identity.get("privacy_context", "private"),
                )
            except Exception:
                log.warning("action context_fn raised; continuing without profile", exc_info=True)
                block = ""
            if block:
                knowledge = f"\n\n[What you know about this person]\n{block}"
        system = SystemMessage(
            content=f"{soul}\n\n{overlay}{ack_block}\n{caller_line}{knowledge}"
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


def action_tools_for(settings: Settings) -> list:
    """Every tool the action subgraph gets, assembled from what Settings arms.

    Two independently-armed halves, same pattern as every optional transport:
    - HOME (ha_token set): home_control + search_entities — the write half,
      canary-gated and outbox-audited.
    - MEDIA (embeddings_api_key set): analyze_image + read_document +
      youtube_summary — the read half, replacing V1's Tool: Image node and the
      06-05 extractor sub-workflows (HE7zmxKeWoxjvM9L / yuHzxHqqWz93xwYj /
      tJwLt494G1VugToU). The OpenRouter credential is the embedder's, reused —
      exactly like n8n credential gvgPllzFhLSds5Qv serving both jobs.

    Empty list = nothing armed = the caller keeps ask() chat-only.
    """
    tools: list = []

    if settings.ha_token is not None:
        from aerys_v2.tools.home_control import (
            build_home_control_tool,
            build_search_entities_tool,
            canary_set,
        )

        conn_factory = None
        if settings.database_url is not None:
            import psycopg

            # Fresh short connection per outbox touch — same per-call choice (and
            # the same "pool is a drop-in later" note) as context_fn_for above.
            def conn_factory():
                return psycopg.connect(settings.database_url)

        tools.append(
            build_home_control_tool(
                base_url=settings.ha_base_url,
                token=settings.ha_token.get_secret_value(),
                canary_entities=canary_set(settings.ha_canary_entities),
                conn_factory=conn_factory,
            )
        )
        tools.append(
            build_search_entities_tool(
                base_url=settings.ha_base_url,
                token=settings.ha_token.get_secret_value(),
            )
        )

    if settings.embeddings_api_key is not None:
        from aerys_v2.tools.media import build_media_tools

        tools.extend(
            build_media_tools(
                api_key=settings.embeddings_api_key.get_secret_value(),
                base_url=settings.embeddings_base_url,
            )
        )

    if settings.tavily_api_key is not None:
        from aerys_v2.tools.web_search import build_web_search_tool

        tools.append(
            build_web_search_tool(api_key=settings.tavily_api_key.get_secret_value())
        )

    return tools


def action_overlay_for(settings: Settings) -> str:
    """Compose the action-subgraph system overlay from the armed tool halves.

    The prompt must only mention tools that EXIST this boot — telling the model
    to "use the home_control tool" on a box without ha_token is asking for the
    V1 hallucinated-tool-call failure with extra steps.
    """
    parts = []
    if settings.ha_token is not None:
        parts.append(ACTION_OVERLAY)
    if settings.embeddings_api_key is not None:
        parts.append(MEDIA_OVERLAY)
    if settings.tavily_api_key is not None:
        parts.append(SEARCH_OVERLAY)
    return "\n\n".join(parts)


def action_stack_for(settings: Settings, soul: str) -> tuple | None:
    """Wire the whole TOOLS block from Settings: (router, action_graph), or None.

    Arms when ANY tool half exists — ha_token (home) and/or embeddings_api_key
    (media); the API model half is structurally required by Settings. None =
    ask() runs chat-only, exactly as before the TOOLS block existed — backward
    compatible by construction.
    """
    tools = action_tools_for(settings)
    if not tools:
        return None
    from aerys_v2.router import router_for

    action_graph = build_action_graph(
        build_api_tool_model(settings, tools),
        soul,
        tools,
        context_fn=context_fn_for(settings, profile_only=True),
        overlay=action_overlay_for(settings),
    )
    return router_for(settings, soul), action_graph


def build_graph(
    model: BaseChatModel,
    soul: str,
    checkpointer: BaseCheckpointSaver | None = None,
    context_fn: ContextFn | None = None,
    tier_models: dict[str, BaseChatModel] | None = None,
) -> object:
    """START → chat → END, checkpointed.

    The checkpointer is INJECTED (pluggable — cross-review #9): InMemorySaver for tests
    and the CLI today, PostgresSaver on the NAS when Phase 2 wires durability. The graph
    shape doesn't change when the storage does — that's the point of the seam.

    context_fn is the same idea for long-term memory: None = the chat node knows
    nothing beyond the thread; set = each turn asks it (person_id, latest user text)
    and injects whatever comes back into the system prompt.

    tier_models is the model-as-a-per-call-parameter seam (Chip's tiering
    pattern via the dossier): the router's tier rides `configurable` — per
    call, never checkpointed, same channel as identity — and the chat node
    picks the model for THIS turn from the map. None (or a tier missing from
    the map) = `model`, so every pre-tier caller behaves byte-for-byte as
    before. One graph, one node; the V1 three-sub-workflow split stays dead.
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
                block = context_fn(
                    str(identity.get("user_id", "")),
                    query_text,
                    identity.get("privacy_context", "private"),
                )
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
        # Tier -> model, resolved per turn (normalize_tier at the node too, not
        # just ask() — Chip's belt-and-braces: whatever garbage reaches config,
        # the node answers with a REAL model and the trace shows which).
        tier = normalize_tier(((config or {}).get("configurable") or {}).get("tier", DEFAULT_TIER))
        turn_model = (tier_models or {}).get(tier, model)
        # n8n mapping: this is the AI Agent node's invoke — prompt + history in, one
        # AIMessage out. add_messages in ChatState appends it to the thread history.
        reply = turn_model.invoke([system, *state["messages"]])
        return {"messages": [reply]}

    graph = StateGraph(ChatState)
    graph.add_node("chat", chat)
    graph.add_edge(START, "chat")
    graph.add_edge("chat", END)
    return graph.compile(checkpointer=checkpointer or InMemorySaver())
