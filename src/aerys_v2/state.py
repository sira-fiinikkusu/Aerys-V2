from typing import Annotated, TypedDict

from langchain_core.runnables import RunnableConfig
from langgraph.graph.message import add_messages


class ChatState(TypedDict):
    """Checkpointed conversation state — the running message list ONLY.

    `add_messages` appends so history accumulates per thread. The AUTHORIZATION identity is intentionally absent here (per-call only); only the conversational record of who-said-what lives in the messages.
    """

    messages: Annotated[list, add_messages]


class Identity(TypedDict, total=False):
    """Represents the identity of the user making the request."""

    user_id: str
    display_name: str
    email: str
    # Room-scoped privacy, set by the transport resolver: 'private' for a 1:1 DM,
    # 'public' for a group/guild. Threaded into build_context so the profile
    # service's visibility gates keep dm-only claims out of shared rooms. Absent =
    # 'private' (the owner's own single-user channels — CLI, voice/HTTP).
    privacy_context: str
    # HA ConversationInput.device_id — the originating satellite. Set by the HA
    # aerys_conversation component (via /ask); lets the spoken follow-up answer on
    # the SAME device the voice turn came from. Absent = fall back to the single
    # configured announce entity (today's single-satellite behavior).
    device_id: str
    # Human room label for shared channels (e.g. Discord "#general", a Telegram
    # group title). Set by the transport resolver from the event; display-only,
    # feeds the "where you're talking" line so she can name the public channel
    # she's in. Absent/"" for DMs and single-user channels.
    channel_name: str
    # The room's WHERE, threaded gateway->resolver->identity alongside channel_name
    # (track/memory-continuity). Since the checkpointer thread is now person-keyed
    # ('person:{id}' for every surface), thread_id no longer encodes the surface —
    # so the resolver carries it here instead: platform ('discord'|'telegram'),
    # channel_kind ('dm'|'guild'|'group'), and channel_id (the raw platform room id,
    # a discord channel snowflake or telegram chat id). Consumed by the chat/action
    # nodes (the "where you're talking" line + the public-channel room block) and by
    # the v2_turns audit row (channel enum + channel_id column). Absent for the
    # owner's single-user channels (CLI, voice/HTTP), whose thread_id still names the
    # surface directly.
    platform: str
    channel_kind: str
    channel_id: str
    # EXPLICIT voice signal (track/memory-continuity): set True by the voice
    # transport (http_api's /v1/chat/completions shim, and /ask when voice=True).
    # It is what arms the three voice behaviors — parallel-start (service.py),
    # ElevenLabs emotion tags (the chat node), and the standard-tier pin — now that
    # voice folds into the owner's person-keyed thread ('person:{id}') and no longer
    # names 'voice' in the thread_id. Per-call only (rides config, never checkpointed),
    # so the SAME person thread is a voice turn or a text turn purely by this flag.
    voice: bool
    # Which PHYSICAL SURFACE renders this turn, when the transport knows and it
    # changes how she should write. Today's only value: 'lens' (the G2 glasses
    # text display, set by the g2-bridge on /ask) — a tiny screen where her reply
    # is READ, not spoken, and anything past ~350 chars gets machine-summarized
    # by the bridge (another model paraphrasing her; owner and Kael agreed
    # 2026-08-26 that her own words should reach the lens instead). Per-call
    # only, like voice — rides config, never checkpointed.
    surface: str


UNKNOWN_CALLER: Identity = {"display_name": "Unknown Caller"}

# The Identity key the explicit voice flag lives under — one constant so the writer
# (the voice transport) and every reader (is_voice_turn) can never drift on spelling.
VOICE_KEY = "voice"

# The Identity key the rendering-surface hint lives under, and the one surface
# value with behavior attached — same writer/reader-can-never-drift rationale.
SURFACE_KEY = "surface"
LENS_SURFACE = "lens"


def is_lens_surface(identity: Identity | dict | None) -> bool:
    """Is this turn rendered on the glasses lens? Single source of truth.

    Set by the g2-bridge (surface='lens' on /ask). Deliberately independent of
    is_voice_turn: a lens turn IS usually a voice turn (spoken in via STT), but
    the lens styling is about where the reply is READ, and the voice flag is
    about how the words came in — the two compose, they don't imply each other.
    """
    return (identity or {}).get(SURFACE_KEY) == LENS_SURFACE


def identity_from_config(config: RunnableConfig | None) -> Identity:
    """Returns the identity of the user making the request, or the unknown caller if not found."""
    configurable = (config or {}).get("configurable") or {}
    identity = configurable.get("identity") or {}
    return identity if identity else UNKNOWN_CALLER


def is_voice_turn(identity: Identity | dict | None, thread_id: object) -> bool:
    """Is THIS turn a voice turn? Single source of truth for voice detection.

    The PRIMARY signal is the explicit per-call `voice` flag on identity, set by the
    voice transport. Person-keying folded voice into the owner's 'person:{id}' thread,
    so the thread_id no longer names 'voice' — the flag is what carries voice-ness now.

    The legacy thread-prefix check is retained as a FALLBACK: the single-user raw-thread
    surfaces (and direct callers/tests) that still name a 'voice:*' thread keep working
    unchanged. A person-keyed voice thread never starts with 'voice', so the fallback
    never fires there — the flag does. No false positives: the '::spec::' speculative
    voice thread is 'person:{id}::spec::...' (flag-detected), and a text 'discord:*'/
    'telegram:*'/'person:*' thread has no flag and no 'voice' prefix.
    """
    if (identity or {}).get(VOICE_KEY):
        return True
    return str(thread_id or "").startswith("voice")
