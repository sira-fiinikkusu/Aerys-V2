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
import queue
import threading
import logging
import re
from typing import Callable

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

from aerys_v2.state import identity_from_config
from aerys_v2.workers.extraction import KEY_LABEL_PATTERN, question_shaped

log = logging.getLogger(__name__)

# The current turn's user text, set by service.ask() before any graph runs.
# LangGraph copies the context into its tool threads, so the tool can compare
# the fact against what the owner actually said this turn.
CURRENT_TURN_TEXT: contextvars.ContextVar[str] = contextvars.ContextVar("aerys_current_turn_text", default="")

#: Characters, and NOT arbitrary — it is bounded by the smaller of the two
#: embedders that have to read the same memory. The house embeds with
#: openai/text-embedding-3-small (8k tokens, effectively unbounded here); her
#: PORTABLE body embeds on-device with all-MiniLM-L6-v2, whose max sequence is
#: 256 tokens and which SILENTLY TRUNCATES past it. 500 characters is roughly
#: 125 tokens, comfortably inside that. Raising this much past ~1000 would put
#: the same mid-thought cut back, one layer down and invisible, inside the
#: vector rather than the text — which is worse, because nothing would show it.
#: If it ever needs to rise, raise the portable embedder first.
FACT_LIMIT = 500
KEY_LABEL_TIMEOUT_S = 2.0
OWNER_QUOTE_RATIO = 0.8  # share of the fact's tokens that must appear in the turn text
_UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.I)
_TOKEN_RE = re.compile(r"\w+")  # Unicode-aware: a Japanese or Arabic quote is still the owner's word
_STOP = frozenset("the a an and or of to in on at for is are was were be that this it my his her their our".split())

KEPT_PREFIX = "Kept:"
ALREADY_PREFIX = "Already kept:"
NOT_KEPT = "I couldn't save that — the memory store didn't confirm the write. Nothing was kept."
NOT_LINKED = "I can only keep memories for a linked person; nothing was kept."
EMPTY = "Tell me the thing to keep — I got an empty fact."
#: An over-long "fact" is almost never one long fact — it is several that were
#: never separated. That is exactly what arrived on 2026-09-13: four distinct
#: things about home control, isolation, gap reporting and the dreaming pass, in
#: one call, cut at 500 characters mid-word and stored as if whole. Refusing says
#: so and costs nothing, because she still has the text in front of her and can
#: call again per fact — which stores better and recalls better, since each one
#: gets its own key and its own embedding.
TOO_LONG = (
    f"Nothing was kept: that is longer than {FACT_LIMIT} characters, which usually "
    "means it is several facts at once. Keep them one at a time, each in its own "
    "sentence, and call me again for each."
)

# writer(record) -> 'insert' | 'update' | 'replace' | 'duplicate' | 'skipped'.
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
    """The original stable, local fallback key when canonical labeling fails."""
    normalized = " ".join(_TOKEN_RE.findall(fact.casefold()))
    return "remember." + hashlib.sha1(normalized.encode()).hexdigest()[:10]


def _bounded_label(labeler, fact):
    # A daemon worker bounds even a custom labeler that ignores its own timeout.
    # It can finish labeling later, but only this calling thread can write memory.
    result = queue.SimpleQueue()
    def run():
        try:
            result.put(labeler(fact))
        except Exception:
            result.put(None)
    threading.Thread(target=contextvars.copy_context().run, args=(run,), daemon=True).start()
    label = result.get(timeout=KEY_LABEL_TIMEOUT_S)
    if not isinstance(label, str) or not re.fullmatch(KEY_LABEL_PATTERN, label):
        raise ValueError("invalid memory key label")
    return label


def build_remember_tool(writer: Writer, *, key_labeler: Callable[[str], str] | None = None):
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

        ONE fact per call, and keep it under 500 characters. If there are several
        things to keep, call this once for each — they are stored and found
        separately, so a paragraph holding four facts is four calls, not one. A
        longer one is refused rather than trimmed; nothing is ever half-kept.
        """
        text = (fact or "").strip()
        if not text:
            return EMPTY
        if len(text) > FACT_LIMIT:
            # Refuse; never keep half. The old behaviour sliced at FACT_LIMIT and
            # stored the fragment as though it were the whole fact — no word
            # boundary, no marker, and the EMBEDDING built from the surviving half,
            # so the part that was lost was also lost to search. Chris found one of
            # these on 2026-09-13, cut mid-word at "poisoned by ba". A refusal is
            # visible and recoverable; a silent half is neither.
            log.warning("remember refused: fact of %d chars exceeds the %d limit",
                        len(text), FACT_LIMIT)
            return TOO_LONG
        identity = identity_from_config(config) if config else {}
        person_id = str(identity.get("user_id") or "")
        if not _UUID_RE.match(person_id):
            return NOT_LINKED
        if question_shaped(text):
            log.warning("remember skipped: question-shaped observation")
            return "Nothing was kept: questions are not facts. State the fact to remember."
        try:
            if key_labeler is None:
                raise ValueError("memory key labeler is not configured")
            label = _bounded_label(key_labeler, text)
        except Exception:
            log.warning("remember: key labeling failed; using stable hash key")
            label = key_label_for(text)
        record = {
            "person_id": person_id,
            "fact": text,
            "key_label": label,
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
        if action == "duplicate":
            return f"{ALREADY_PREFIX} {text}"
        if action == "skipped":
            return "Nothing was kept: the fact did not pass memory checks. State a specific fact."
        if action in ("insert", "update", "replace"):
            return f"{KEPT_PREFIX} {text}"
        log.warning("remember: writer returned an unknown action %r", action)
        return NOT_KEPT

    return remember
