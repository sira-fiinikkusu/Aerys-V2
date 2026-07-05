"""Per-message CONTENT privacy — the short-term (checkpointer) privacy gate.

Two jobs live here, both PURE (no DB, no network — same testable-with-fakes stance
as services/context.py and turns.py):

  1. CLASSIFY (Part 2): decide whether a piece of conversation is 'private' CONTENT
     (health / finances / relationship struggles / trauma / orientation) or 'public'
     general content (name / job / location / hobbies / a number he mentioned). This
     is the SAME rule workers/extraction.py's `privacy_level` prompt applies to
     long-term memories — copied here as the short-term twin. Privacy is by CONTENT,
     NOT by origin room: a general thing said in a DM may carry into a public room,
     while a private thing said ANYWHERE (even a DM) must never surface in public.

  2. GATE (Part 3, the security-critical piece): given a thread's message history and
     a PUBLIC viewing context, drop every human turn tagged 'private' AND its paired
     assistant response, so the model in a public room never sees private DM content
     — nor a reply that references it. Structural, not a prompt request.

The tag rides `additional_kwargs["content_privacy"]` on each HumanMessage (the
checkpointer persists additional_kwargs verbatim). FAIL-CLOSED is the load-bearing
contract: anything NOT explicitly tagged 'public' is treated as private when viewed
in public. A DM message is tagged 'private' at ingest and only ever RELAXED to
'public' by an off-the-hot-path classifier that has looked at the actual content —
so an unclassified (or unclassifiable) message reveals LESS, never more.
"""

from __future__ import annotations

import logging
import re

log = logging.getLogger(__name__)

# The additional_kwargs key the tag lives under — one constant so service.py (writer)
# and the gate (reader) can never drift on the spelling.
CONTENT_PRIVACY_KEY = "content_privacy"
PUBLIC = "public"
PRIVATE = "private"

# Keyword fast-path for the classifier: a small, high-precision set of markers for the
# five private CATEGORIES (extraction.py's privacy_level rules). A hit is a confident
# 'private' short-circuit that skips the LLM entirely — it can only ADD privacy, never
# remove it, so a false positive over-hides (annoying, safe) and never leaks. Nuanced
# private content with NO obvious marker ("I've been really struggling lately") is
# exactly what the LLM judge is for; keyword-only mode leaves those to the fail-closed
# default. Word-boundary anchored so "therapist" doesn't fire on "the rapist"-style
# substrings and "meds" doesn't fire inside "immeds"/"comedy".
_PRIVATE_MARKERS = re.compile(
    r"\b("
    # health / medical
    r"diagnos\w*|symptoms?|therap(?:y|ist)|depress\w*|anxiety|anxious|"
    r"medications?|meds|prescriptions?|disorders?|disabilit\w*|"
    r"suicid\w*|self[- ]harm|cancer|chronic|illness|mental health|"
    # finances (specifics — not "I bought a truck", which is a general life fact)
    r"salary|my income|in debt|owe\w*|bankrupt\w*|credit score|net worth|"
    r"bank account|savings account|social security number|ssn|"
    # relationship struggles
    r"divorce\w*|break(?:ing)?[- ]?up|broke up|cheat(?:ed|ing)|affair|"
    r"custody|marriage problems|couples? (?:counsel\w*|therapy)|"
    # trauma
    r"trauma\w*|ptsd|abus(?:e|ed|ive)|assault\w*|molest\w*|"
    # orientation
    r"gay|lesbian|bisexual|transgender|closeted|coming out|my orientation|"
    # secrets / credentials — a "number" or "location" that is a SECRET is NOT a
    # general fact. These must never depend on the LLM judge's discretion (the
    # 2026-07-05 review's confirmed leak: "the garage code is 4482" relaxed to public).
    r"passwords?|passcodes?|passwd|\bpins?\b|"
    r"(?:garage|gate|door|alarm|access|entry|security|lock|vault|keypad|safe|house|building|combo|combination)[- ]?codes?|"
    r"\bwi[- ]?fi\b|\bssid\b|network password|"
    r"api[- ]?keys?|secret[- ]?keys?|private[- ]?keys?|\baccess[- ]?keys?|seed phrase|recovery phrase|\bwallet\b|"
    r"account numbers?|routing numbers?|credit[- ]?cards?|debit[- ]?cards?|\bcvv\b|card numbers?|"
    r"(?:home|street|mailing|physical|my)[- ]?address"
    r")\b",
    re.IGNORECASE,
)


def keyword_verdict(text: str) -> str | None:
    """PRIVATE if any private-category marker appears, else None ("no opinion").

    None is deliberately NOT 'public': the keyword pass only ever ASSERTS privacy.
    Whether a marker-free message is general-enough to carry into public is a
    judgment the LLM (or, absent one, the fail-closed default) makes — see
    classify_content_privacy.
    """
    return PRIVATE if _PRIVATE_MARKERS.search(text or "") else None


def normalize_verdict(raw: object) -> str:
    """Coerce a judge's free-text answer to 'public'|'private', FAIL-CLOSED.

    The classifier LLM is told to answer with one word, but models wander
    ("This looks private."). We look for an explicit 'public' and default to
    'private' on ANYTHING ambiguous — a mis-read judge must hide, never reveal.
    """
    low = str(raw or "").strip().lower()
    # 'private' wins ties and anything unclear; only an unambiguous 'public' relaxes.
    if PRIVATE in low:
        return PRIVATE
    if PUBLIC in low:
        return PUBLIC
    return PRIVATE


def classify_content_privacy(text: str, llm=None) -> str:
    """Classify one piece of conversation as 'public' or 'private' CONTENT.

    Order (the keyword+LLM combo): a keyword hit short-circuits to 'private' with no
    model spend. Otherwise, if an `llm` judge is wired (Callable[[str], str] returning
    a verdict word), its normalized answer decides — a judge exception fails CLOSED to
    'private'. With NO judge, a marker-free message defaults to 'public' (the honest
    "keyword classifier alone" behavior); callers that must not risk a keyword false
    negative (the async retag path) only ARM this with a judge present, so the
    unqualified 'public' default is never on the leak-critical path.
    """
    if keyword_verdict(text) == PRIVATE:
        return PRIVATE
    if llm is not None:
        try:
            return normalize_verdict(llm(text))
        except Exception:
            log.warning("content-privacy judge raised — failing closed to private", exc_info=True)
            return PRIVATE
    return PUBLIC


def content_privacy_of(message: object) -> str | None:
    """Read a message's content-privacy tag, or None when it carries none.

    Duck-typed on additional_kwargs so fakes and every message class work without
    importing concrete langchain types (same stance as turns.extract_tool_calls).
    """
    kwargs = getattr(message, "additional_kwargs", None) or {}
    return kwargs.get(CONTENT_PRIVACY_KEY)


def redact_private_history(messages: list) -> list:
    """The GATE: strip private human turns + their responses for a PUBLIC viewer.

    FAIL-CLOSED: a human turn is KEPT only when its tag is EXACTLY 'public'. A turn
    tagged 'private', OR carrying no tag at all (legacy history from before this
    feature, or a DM turn the async classifier hasn't relaxed yet), is dropped —
    along with every following non-human message (its assistant reply, and any tool
    messages that reply generated), up to the next human turn. That pairing is what
    keeps a reply that QUOTES private content out of the public model input, not just
    the private message itself.

    Called by the chat node whenever the room is NOT explicitly private (fail-closed:
    public OR unknown context redacts; only a private DM passes through untouched).
    Pure and order-preserving.

    The CURRENT (final) human turn is ALWAYS kept regardless of tag — it is the
    message the user just said IN THIS room, so it is appropriate to answer here (and
    in production it is tagged 'public' at ingest anyway; this is belt-and-braces so a
    fail-closed unknown context can never drop the very message being answered). It's
    not a leak: its privacy is the current room's, which is the room we're serving.
    """
    last_human = max(
        (i for i, m in enumerate(messages) if getattr(m, "type", "") == "human"),
        default=-1,
    )
    kept: list = []
    keeping_response = False  # are we inside a KEPT (public) human turn's response span?
    for i, m in enumerate(messages):
        if getattr(m, "type", "") == "human":
            # keep a PRIOR human only when explicitly tagged 'public'; ALWAYS keep the
            # current turn (i == last_human). Private/untagged priors drop with their
            # whole response span (keeping_response=False until the next kept human).
            keeping_response = content_privacy_of(m) == PUBLIC or i == last_human
            if keeping_response:
                kept.append(m)
            continue
        if keeping_response:
            kept.append(m)
    return kept
