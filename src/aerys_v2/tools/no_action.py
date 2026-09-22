"""no_action — the forced first pass's honest exit (2026-09-04).

The specialist's first pass is bound with tool_choice="any": the router already
decided this turn needs a tool, and letting the model decline on pass one is how
a stale belief ("I can't set brightness") turned a two-call job into a 28s turn.
But forcing a tool on a graph that also holds EFFECTFUL tools (message_kael,
timer, music, log_gap) means a misrouted turn — chit-chat the router sent to
action, a question the model genuinely needs to ask — would have to pick one of
them. This tool is the harmless option: it does nothing, records the reason, and
tells the model to say exactly that. It is never a claim of work done.
"""
from langchain_core.tools import tool

NO_ACTION_PREFIX = "No action was taken."


def build_no_action_tool():
    @tool
    def no_action(reason: str) -> str:
        """Call this when the request needs NO device or tool action from you:
        it is a question you must ask the user back (which room? which one?),
        it is outside every tool you have, or there is genuinely nothing to do.

        NEVER call this because you believe a room or device does not exist.
        You do not hold the list — home_control does, and it matches names
        loosely: spacing, plurals and word order all resolve, so "sun room",
        "the sunroom" and "sunroom lights" all reach the same four bulbs. If
        the user names something to turn on, off, dim or check, CALL
        home_control and let it answer. When a name really matches nothing it
        says so, and that answer is worth far more than your guess.
        (2026-09-21: "Turn off the sun room, please." was declined here with
        "I don't actually have control over a sun room device" while four
        sunroom lights sat on the allowlist — and stayed on.)

        reason: the one plain sentence you will say to the user — the question
        itself, or what you cannot do and why. Never call this AFTER another
        tool already did the work; never use it to skip a tool that applies.
        """
        text = (reason or "").strip() or "Nothing to do."
        return f"{NO_ACTION_PREFIX} Tell the user exactly this, in your own voice: {text}"

    return no_action
