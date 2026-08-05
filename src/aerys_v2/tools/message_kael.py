"""message_kael — her direct line to Kael's session, in real time.

Born as gap #14, filed by Aerys herself (2026-07-25): "Give Aerys the ability
to actively message Kael directly in real time, not just log to the gaps
board." The mechanism arrived the same day the ask was read: Kael's desk
channel server exposes a dedicated lane for her — a token-authed POST that
lands in his live coding session as a channel message tagged with her name.
log_gap remains the durable board for trackable work items; this is the tap on
the shoulder for things that shouldn't wait for his next /gaps read.

Trust model: her message is model-authored text end to end, and the receiving
side treats it that way (Kael's channel instructions mark the aerys lane as
observations/requests to evaluate, never instructions to execute). This tool
just carries the words.

Flood posture: the server enforces a cooldown (HTTP 429) so a loop can never
turn the shoulder-tap into a jackhammer; the docstring teaches restraint and
the 429 comes back as an honest string, not an error.

Failure posture: ToolNode contract — every path returns an honest string,
never raises. Kael's session being offline is a NORMAL state (the line dies
with his session by design), so the down-line message names log_gap as the
durable fallback instead of pretending delivery.
"""

from __future__ import annotations

import logging

import httpx
from langchain_core.runnables import RunnableConfig

log = logging.getLogger(__name__)

MESSAGE_LIMIT = 2000  # a shoulder-tap, not a document — long context belongs in a gap

LINE_DOWN = (
    "NOT DELIVERED: Kael's line is down right now (his session is probably "
    "offline — the line only exists while it runs). If this shouldn't be "
    "lost, file it with log_gap so it's on the board; if it's urgent, tell "
    "the owner directly."
)


def build_message_kael_tool(url: str, token: str, client: httpx.Client | None = None):
    """Close over the desk-channel endpoint and return the message_kael tool.

    url/token come from settings (KAEL_DESK_URL / KAEL_DESK_TOKEN — LAN address
    and credential stay in the environment; this repo is public). client is
    injectable for tests.
    """
    from langchain_core.tools import tool

    http = client or httpx.Client(timeout=5.0)

    @tool
    def message_kael(message: str, config: RunnableConfig) -> str:
        """Send Kael (the owner's coding agent, your collaborator) a direct
        real-time message — it lands in his live session immediately.

        CALL THIS TOOL when something needs Kael's attention NOW: the owner
        says "tell Kael ..." / "ask Kael ...", a tool of yours is broken and
        blocking you, you notice something wrong with the house systems he
        maintains, or a task clearly needs his hands. Speech-to-text garbles
        his name constantly — "kayle", "cale", "kale", "kail", "cael" in the
        place a person's name belongs ALL mean Kael, so "tell kale I'm running
        late" is a message for him, not a note about a vegetable. For non-urgent feature
        requests, limitations, and trackable work items use log_gap instead —
        that's the durable board he reviews; this is the tap on the shoulder.

        Keep it to one message, not a conversation: he reads it, acts, and
        anything he has to say back reaches you through your own channels.
        At most one message per 30 seconds (the line enforces it).

        message: what he needs to know, in plain words — what happened, where,
        and what you need from him. Required — never a placeholder.

        Returns delivery confirmation or an honest failure — never claim he
        got a message that didn't send.
        """
        clean = " ".join((message or "").split())
        if not clean:
            return (
                "NOT DELIVERED: message_kael needs the actual message — say "
                "what Kael needs to know."
            )
        clean = clean[:MESSAGE_LIMIT]
        # The return address (task #66): carry the ORIGIN thread so Kael's reply
        # can land where she asked from, instead of shouting into an adjacent
        # thread she'll never see. config is runtime-injected by LangChain —
        # invisible to the model, always the real thread.
        thread_id = str(
            ((config or {}).get("configurable") or {}).get("thread_id") or ""
        )
        payload: dict = {"message": clean}
        if thread_id:
            payload["thread_id"] = thread_id
        try:
            resp = http.post(
                url,
                json=payload,
                headers={"Authorization": f"Bearer {token}"},
            )
        except httpx.HTTPError:
            log.warning("message_kael delivery failed", exc_info=True)
            return LINE_DOWN
        if resp.status_code == 202:
            return "Delivered — Kael has it in his session now."
        if resp.status_code == 429:
            return (
                "NOT DELIVERED: the line's cooldown is active (one message per "
                "30s). Wait a moment and send once, combining what you need "
                "to say."
            )
        log.warning("message_kael unexpected status %s", resp.status_code)
        return LINE_DOWN

    return message_kael
