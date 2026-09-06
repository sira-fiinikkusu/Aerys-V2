"""remember — she keeps a fact on purpose, through the memory service, never SQL.

Origin (2026-09-06, the portable-body design conversation): the owner can add a
memory explicitly (/aerys-tell) and the extractor mines turns after the fact,
but SHE had no way to decide "this matters, keep it" mid-conversation, and no
way to know whether "got it" was true. This tool is that ability. The write
goes through the same writer the extractor uses (workers.extraction.triage_memory:
provenance, dedup, supersession, embedding) — the owner's rule from the same
morning: memory writes hit a service, never a table.

Trust is NOT self-declared. A fact that quotes what the owner just said is
owner-trust; anything else she keeps is her own inference (assistant-trust).
The check is mechanical — token overlap between the fact and the current turn's
text — so the model cannot promote its own inference to the owner's word.

Two back ends, one tool: on the house the writer is triage_memory over the
prod memories DB; on the portable body the writer appends a memory event to the
local store and the sync carries it home through the door's judge. The model
sees the same tool name and the same replies either way.

Failure posture: ToolNode contract — every path returns an honest string, never
raises; "Kept:" is said ONLY when the writer confirmed the write.
"""
from __future__ import annotations

import contextvars
import hashlib
import logging
import re
from typing import Any, Callable

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

from aerys_v2.state import identity_from_config

log = logging.getLogger(__name__)

# The current turn's user text, set by service.ask() before any graph runs.
# LangGraph copies the context into its tool threads, so the tool can compare
# the fact against what the owner actually said this turn.
CURRENT_TURN_TEXT: contextvars.ContextVar[str] = contextvars.ContextVar("aerys_current_turn_text", default="")

FACT_LIMIT = 500
OWNER_QUOTE_RATIO = 0.8  # share of the fact's tokens that must appear in the turn text
_UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.I)
_TOKEN_RE = re.compile(r"\w+")  # Unicode-aware: a Japanese or Arabic quote is still the owner's word
_STOP = frozenset("the a an and or of to in on at for is are was were be that this it my his her their our".split())

KEPT_PREFIX = "Kept:"
ALREADY_PREFIX = "Already kept:"
NOT_KEPT = "I couldn't save that — the memory store didn't confirm the write. Nothing was kept."
NOT_LINKED = "I can only keep memories for a linked person; nothing was kept."
EMPTY = "Tell me the thing to keep — I got an empty fact."

# writer(record) -> 'insert' | 'update' | 'replace' | 'skipped'; raises on failure.
# record keys: person_id, fact, key_label, privacy_level, trust, source_platform, channel
Writer = Callable[[dict], str]


def _tokens(text: str) -> set[str]:
    return {t for t in _TOKEN_RE.findall(text.casefold()) if (len(t) >= 3 or not t.isascii()) and t not in _STOP}


def trust_for(fact: str, turn_text: str) -> str:
    """owner when the fact is (near-)verbatim what the owner said this turn; else assistant."""
    fact_tokens = _tokens(fact)
    if not fact_tokens or not turn_text:
        return "assistant"
    overlap = len(fact_tokens & _tokens(turn_text)) / len(fact_tokens)
    return "owner" if overlap >= OWNER_QUOTE_RATIO else "assistant"


def key_label_for(fact: str) -> str:
    """Stable per fact text: re-keeping the same fact is idempotent; different facts never collide."""
    normalized = " ".join(_TOKEN_RE.findall(fact.casefold()))
    return "remember." + hashlib.sha1(normalized.encode()).hexdigest()[:10]


def build_remember_tool(writer: Writer):
    @tool
    def remember(fact: str, config: RunnableConfig = None) -> str:
        """KEEP a fact for the future, on purpose, in your long-term memory.

        CALL THIS TOOL IMMEDIATELY when the user asks you to remember, keep, note,
        or not forget something — "remember that…", "keep in mind…", "make a note
        that…", "for next time…", "don't forget…" — and when you yourself decide a
        fact is worth keeping. Pass the fact in plain words; when the user stated
        it, pass it as they said it (that is what makes it THEIR word on record).
        Say it was kept ONLY if this tool replied "Kept:". Recalling what you
        already know needs no tool.
        """
        text = (fact or "").strip()
        if not text:
            return EMPTY
        if len(text) > FACT_LIMIT:
            text = text[:FACT_LIMIT].rstrip()
        identity = identity_from_config(config) if config else {}
        person_id = str(identity.get("user_id") or "")
        if not _UUID_RE.match(person_id):
            return NOT_LINKED
        record = {
            "person_id": person_id,
            "fact": text,
            "key_label": key_label_for(text),
            "privacy_level": "private" if identity.get("privacy_context") == "private" else "public",
            "trust": trust_for(text, CURRENT_TURN_TEXT.get()),
            "source_platform": str(identity.get("platform") or "unknown"),
            "channel": identity.get("channel_id"),
        }
        try:
            action = writer(record)
        except Exception:
            log.warning("remember: writer failed for person %s", person_id, exc_info=True)
            return NOT_KEPT
        if action == "skipped":
            return f"{ALREADY_PREFIX} {text}"
        if action in ("insert", "update", "replace"):
            return f"{KEPT_PREFIX} {text}"
        log.warning("remember: writer returned an unknown action %r", action)
        return NOT_KEPT

    return remember
