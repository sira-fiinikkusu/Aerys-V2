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


UNKNOWN_CALLER: Identity = {"display_name": "Unknown Caller"}


def identity_from_config(config: RunnableConfig | None) -> Identity:
    """Returns the identity of the user making the request, or the unknown caller if not found."""
    configurable = (config or {}).get("configurable") or {}
    identity = configurable.get("identity") or {}
    return identity if identity else UNKNOWN_CALLER
