"""Model-only history views; the checkpointer keeps the literal record."""

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage


def window_messages(messages: list[BaseMessage], limit: int) -> list[BaseMessage]:
    """Keep the tail, dropping any orphan reply/tool result at its boundary.

    Measured 2026-09-19: replaying 824 messages made one voice turn 62,588
    input tokens. Limit what she sees, never what the checkpointer remembers.
    Zero disables the window.
    """
    if limit <= 0:
        return list(messages)
    if len(messages) <= limit:
        kept = list(messages)
    else:
        # Stepped boundary: the cut index is aligned to a multiple of limit/4, so
        # the window's leading edge moves only every limit/4 messages instead of
        # every turn. A boundary that slid by two messages per turn would re-shape
        # the cached history prefix on every request — the history breakpoint
        # would write every turn and read never, the exact failure the
        # 2026-09-19 measurements found for the clock line. The view stays
        # between 3/4 of the limit and the limit; the stored thread only ever
        # appends, so absolute indexes are a stable thing to align on.
        step = max(1, limit // 4)
        low_water = max(1, limit - step)
        cut = ((len(messages) - low_water) // step) * step
        kept = messages[cut:]
    start = next((i for i, message in enumerate(kept) if isinstance(message, HumanMessage)), len(kept))
    return kept[start:]


def prompt_with_context(static: str, messages: list[BaseMessage], dynamic: str) -> list[BaseMessage]:
    """Attach live context to a COPY so privacy retagging still sees the original.

    The minute clock ahead of history caused cache writes every turn (2026-09-19).
    Keep the system prefix stable, including on later passes through a tool loop.
    """
    prompt = list(messages)
    for i in range(len(prompt) - 1, -1, -1):
        current = prompt[i]
        if isinstance(current, HumanMessage):
            blocks = ([{"type": "text", "text": current.content}]
                      if isinstance(current.content, str) else list(current.content))
            prompt[i] = current.model_copy(update={"content": [
                {"type": "text", "text": f"[Context for this turn]\n{dynamic}"}, *blocks,
            ]})
            return [SystemMessage(content=static), *prompt]
    # Defensive: even malformed/no-human inputs must retain their live context.
    return [SystemMessage(content=f"{static}\n{dynamic}"), *prompt]
