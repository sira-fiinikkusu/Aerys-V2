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


UNKNOWN_CALLER: Identity = {"display_name": "Unknown Caller"}


def identity_from_config(config: RunnableConfig | None) -> Identity:
    """Returns the identity of the user making the request, or the unknown caller if not found."""
    configurable = (config or {}).get("configurable") or {}
    identity = configurable.get("identity") or {}
    return identity if identity else UNKNOWN_CALLER
