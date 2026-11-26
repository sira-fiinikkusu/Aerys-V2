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

from aerys_v2.services.location_guard import location_memory_blocked, LOCATION_REFUSAL
from aerys_v2.services.secret_guard import SECRET_REFUSAL, secret_kind
from aerys_v2.state import VOICE_OWNER_ASSUMED_KEY, identity_from_config
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
#: D105 r9 (Codex): there is no overlap ratio any more. A fact is his words only when EVERY content word in it is
#: in what he said (filler, pronouns, contraction endings and his own name excepted): a 20% allowance was where a
#: meaning-changing word (a homoglyph "n\u043et", "dont", "non-smoking", a split "n ot") could hide.
_UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.I)
_TOKEN_RE = re.compile(r"\w+")  # Unicode-aware: a Japanese or Arabic quote is still the owner's word
_STOP = frozenset("the a an and or of to in on at for is are was were be that this it my his her their our".split())
#: Codex D105 r1/r5: word overlap ignores "not" ("I do not like coffee" -> "I like coffee" read as his words).
#: Until the judged check (#261), the negations (each with the word it negates) must match exactly.
#: A side agent's catch (11/25): "Don't forget my appointment is Dec 19" or "No, remember that I..." ask her to keep
#: something; that negation is the request, not the fact. Codex D105 r3: only a REQUEST at the start of a sentence
#: of HIS turn is removed ("I never forget birthdays" keeps its negation), never anything in the fact. Codex r4: a
#: leading "No," is NOT removed ("I like coffee? No." is a correction); "No, remember that I..." stays hers.
_REMEMBER_ASK_RE = re.compile(
    r"(?:^|(?<=[.!?]))\s*(?:(?:hey|ok|okay)\s+aerys[,!]?\s*)?(?:(?:and|also|oh|please)[,]?\s+)*"
    r"(?:(?:don['\u2019]?t|do\s+not|never)\s+(?:you\s+)?(?:ever\s+)?(?:let\s+(?:me|you|us)\s+)?forget\b[\s,.:;!\u2014\u2013-]*)",
    re.I | re.M)


_NEG_WORDS = frozenset("not no cannot never nothing none nobody nowhere neither nor without".split())
_NEG_SAME = frozenset("not no cannot".split())       # don't / doesn't / do not / cannot / no ... all read as "not"
#: Codex r6/r7: a word, plus an apostrophe only when it's a real contraction ending (don't, it's, we're), so quote
#: marks split words instead of joining them ("not\u2018like\u2019" is "not", "like").
_WORDS_RE = re.compile(r"\w+(?:'(?:t|s|re|ve|ll|d|m)\b)?")
#: Codex r8: every apostrophe-like mark reads as one apostrophe first (don\u2018t, don\u2019t, don`t), then only a real
#: contraction ending joins the word ("not\u2018like\u2019" still splits into "not", "like").
_APOSTROPHES_RE = re.compile("[\u2018\u2019`\u00b4\u02bc\u2032]")


def _negations(text: str) -> set[tuple[str, str]]:
    """Codex D105 r5: each negation WITH the word it negates ("not like", "never forget"), so an unrelated "No," or
    "I don't know" elsewhere can't vouch for a negation in the fact, and a negation in the fact needs the same one
    in what he said."""
    words = _WORDS_RE.findall(_plain(text))
    out = set()
    for i, w in enumerate(words):
        if w in _NEG_WORDS or w.endswith("n't"):
            head = "not" if (w in _NEG_SAME or w.endswith("n't")) else w
            nxt = next((x for x in words[i + 1:] if x not in _STOP), "")
            out.add((head, nxt))
    return out

KEPT_PREFIX = "Kept:"
ALREADY_PREFIX = "Already kept:"
NOT_KEPT = "I couldn't save that — the memory store didn't confirm the write. Nothing was kept."
NOT_LINKED = "I can only keep memories for a linked person; nothing was kept."
#: D2 (Chris 2026-11-08): a voice that wasn't clearly his can't write his memory.
NOT_SURE_WHO = ("Nothing was kept: I couldn't tell for sure that was Chris's voice. "
                "If it was you, Chris, say it again close to the speaker or text it to me.")
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


#: Contractions spelled out, with or without the apostrophe (Chris types "dont", "cant", "doesnt"), so
#: "don't" / "dont" / "do not" all read the same in both texts.
_CONTRACTIONS = [
    (re.compile(r"(?i)\bcan'?t\b"), "can not"), (re.compile(r"(?i)\bwon'?t\b"), "will not"),
    (re.compile(r"(?i)\bshan'?t\b"), "shall not"), (re.compile(r"(?i)\bain'?t\b"), "am not"),
    (re.compile(r"(?i)\b(do|does|did|is|are|was|were|has|have|had|should|would|could|must|need|might)n'?t\b"), r"\1 not"),
    (re.compile(r"(?i)n't\b"), " not"),
    # Codex r10: the other contractions are spelled out too, so no leftover piece needs an exemption.
    (re.compile(r"(?i)'s\b"), ""), (re.compile(r"(?i)'m\b"), " am"), (re.compile(r"(?i)'re\b"), " are"),
    (re.compile(r"(?i)'ve\b"), " have"), (re.compile(r"(?i)'ll\b"), " will"), (re.compile(r"(?i)'d\b"), " would"),
]


def _plain(text: str) -> str:
    """Codex r6-r9, the whole typography class at once: compatibility forms folded (full-width letters and
    apostrophes), invisible format characters dropped, every apostrophe-like mark read as one apostrophe, and
    contractions spelled out. Whatever still looks odd (a homoglyph, an accented letter, a split word) simply
    doesn't match his words, and an unmatched word means the fact isn't his."""
    return _spell(_normal(text).casefold())


def _normal(text: str) -> str:
    import unicodedata
    text = _APOSTROPHES_RE.sub("'", text or "")      # before NFKC: it turns \u00b4 into a space + accent
    text = unicodedata.normalize("NFKC", text)
    text = "".join(c for c in text if unicodedata.category(c) not in ("Cf", "Mn"))
    return _APOSTROPHES_RE.sub("'", text)


def _spell(text: str) -> str:
    """Contractions spelled out (case kept: "Don't" -> "Do not")."""
    for pattern, spelled in _CONTRACTIONS:
        text = pattern.sub(spelled, text)
    return text


#: Codex r10: the only words a fact may carry that he didn't say are his own name (passed in) and first-person
#: pronouns (she keeps "Chris ..." for his "I ..."). Filler, letters and stopwords all count: "vitamin A" never
#: vouches for "vitamin D", nor "I work" for "I work in IT".
#: Codex r12: no pronoun is exempt ("The cabin is yours" must never vouch for "The cabin is mine").
_LIVE_QUOTE_RATIO = 0.8   # the rule live until D105 (885e3e5), kept as a gate: see _live_rule
#: Numbers whole, with their sign, decimal point, fraction or time ("-20", "1.5", "1/5", "8:30" all differ); then
#: words; then any symbol that isn't ordinary sentence punctuation as a token of its own ("<", ">", "\u00ac", "$").
_WORD_TOKEN_RE = re.compile(
    r"[-\u2212+\u00b1]?(?:\d+(?:[.,/:]\d+)*|[.,]\d+)%?"          # numbers whole: -20, 1.5, 1/5, 8:30, .5
    r"|\w+"
    r"|[^\w\s.,!?;:'\"()\[\]{}\u2026\u2013\u2014*`]")            # any other mark is a token ("<", "\u00ac", "+", "-")
_ACRONYM_RE = re.compile(r"[A-Z]{1,3}")
#: Codex r11: "I" after these words is a Roman numeral ("type I", "stage I"), never the pronoun.
_ROMAN_AFTER = frozenset("type stage phase grade class level part chapter war act volume vol tier step round "
                         "schedule category".split())


def _words(text: str) -> set[str]:
    """Every token of the text: numbers whole, words casefolded except short all-caps acronyms (IT, A, D, TV),
    meaningful symbols on their own."""
    return set(_tokens_in_order(text))


def _tokens_in_order(text: str) -> list[str]:
    plain = _spell(_normal(text))
    # Codex r11: a spaced minus stays on its number ("- 20" is "-20"); a hyphen inside a word just splits it
    # ("non-smoking" -> "non", "smoking"); any other minus or plus is a token of its own ("O-", "O+").
    plain = re.sub(r"(?<![\w.])([-\u2212])\s+(?=[.,]?\d)", r"\1", plain)
    plain = re.sub(r"(?<=\w)[-\u2010](?=\w)", " ", plain)
    out: list[str] = []
    prev = ""
    for m in _WORD_TOKEN_RE.finditer(plain):
        tok = m.group()
        # A capital "A" that starts a sentence is the article, not an acronym ("A friend of mine...").
        sentence_start = not plain[:m.start()].strip() or plain[:m.start()].rstrip()[-1:] in ".!?"
        if tok == "I" and prev in _ROMAN_AFTER:
            out.append("^I")
        elif tok != "I" and _ACRONYM_RE.fullmatch(tok) and not (len(tok) == 1 and sentence_start):
            out.append("^" + tok)
        else:
            out.append(tok.casefold())
        prev = tok.casefold()
    return out


def _marks(text: str) -> set[str]:
    """Codex r11: the meaning-bearing marks (comparison, polarity, sign, currency...) — like negations, they must
    be the same on both sides: a fact may not add one, and may not drop one ("\u00aclike" -> "like", ">5" -> "5")."""
    import unicodedata
    return {t for t in _words(text) if len(t) == 1 and not t.isalnum()
            and (unicodedata.category(t)[0] == "S" or t in "+-\u2212\u00b1<>=~^|&%#@/\\")}


def _odd_characters(text: str) -> bool:
    """Codex r9: characters the word check can't see (enclosed letters like \U0001F15D, emoji, private-use,
    unassigned, control) could carry a word; a fact holding one is never counted as his words."""
    import unicodedata
    return any(unicodedata.category(c) in ("So", "Sk", "Co", "Cn", "Cs") or
               (unicodedata.category(c) == "Cc" and c not in "\n\t\r") for c in _plain(text))


def _live_rule(fact: str, turn_text: str) -> str:
    """The quote check exactly as it was live before D105 (885e3e5): 80% of the fact's longer words in the turn.
    Codex r12: kept as a GATE, so the new rules can only ever say "their words" where the old one already did —
    never a new false attribution, by construction. Whatever slips past both is a gap live already had (#261)."""
    def toks(text: str) -> set[str]:
        return {w for w in _TOKEN_RE.findall((text or "").casefold()) if (len(w) >= 3 or not w.isascii()) and w not in _STOP}
    fact_tokens = toks(fact)
    if not fact_tokens or not turn_text:
        return "assistant"
    return "owner" if len(fact_tokens & toks(turn_text)) / len(fact_tokens) >= _LIVE_QUOTE_RATIO else "assistant"


def trust_for(fact: str, turn_text: str, names=()) -> str:
    """owner only when the fact says what he said this turn: the same negations (each with the word it negates)
    and every content word of the fact found in his words (his name and pronouns excepted); else assistant. His
    sentence-leading "Don't forget ..." request is not part of what he stated. Ambiguity is always assistant."""
    if not (fact or "").strip() or not (turn_text or "").strip():
        return "assistant"
    if _odd_characters(fact):
        return "assistant"
    stated = _REMEMBER_ASK_RE.sub(" ", _spell(_normal(turn_text)))
    if _negations(fact) != _negations(stated) or _marks(fact) != _marks(stated):
        return "assistant"
    content = _words(fact)
    if not content:
        return "assistant"
    own_names = {w.casefold() for n in names or () for w in _TOKEN_RE.findall(_normal(str(n or "")))}
    if content - _words(stated) - own_names:
        return "assistant"
    return _live_rule(fact, turn_text)


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


#: D105: set ONLY by outside_guard.action_executor, from the card that was tapped (never by a turn): the
#: code-owned names of what she read before asking to keep this (a web page, an email...). Recorded on the fact
#: so recall can say where it came from; it never changes whose words it is.
KEPT_AFTER_READING = "kept_after_reading"
_SOURCE_NAME = re.compile(r"[a-z_]{1,40}")


def refusal_for(text: str) -> str | None:
    """The refusals remember gives before it knows who is asking (the card checks these first, so a fact that
    can't be kept never reaches his button, and a secret never reaches a card)."""
    if location_memory_blocked(text):
        return LOCATION_REFUSAL
    if secret_kind(text):
        return SECRET_REFUSAL
    if not text:
        return EMPTY
    if len(text) > FACT_LIMIT:
        return TOO_LONG
    if question_shaped(text):
        return QUESTION_REFUSAL
    return None


QUESTION_REFUSAL = "Nothing was kept: questions are not facts. State the fact to remember."


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
        if location_memory_blocked(text):
            return LOCATION_REFUSAL
        # Before the length check (TOO_LONG asks her to split it, and a key in pieces
        # loses the shape that marks it) and before the key labeler (a model call).
        kind = secret_kind(text)
        if kind:
            log.warning("remember refused: secret-shaped fact (%s)", kind)
            return SECRET_REFUSAL
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
        if identity.get(VOICE_OWNER_ASSUMED_KEY):
            return NOT_SURE_WHO
        person_id = str(identity.get("user_id") or "")
        if not _UUID_RE.match(person_id):
            return NOT_LINKED
        if question_shaped(text):
            log.warning("remember skipped: question-shaped observation")
            return QUESTION_REFUSAL
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
            "trust": trust_for(text, CURRENT_TURN_TEXT.get(), names=(identity.get("display_name"),)),
            "read": sorted({str(n) for n in (((config or {}).get("configurable") or {}).get(KEPT_AFTER_READING)
                                            or ()) if _SOURCE_NAME.fullmatch(str(n))}),
            "source_platform": str(identity.get("platform") or "unknown"),
            "channel": identity.get("channel_id"),
        }
        try:
            action = writer(record)
        except Exception:
            log.warning("remember: writer failed for person %s", person_id, exc_info=True)
            return NOT_KEPT
        if action == LOCATION_REFUSAL:
            return LOCATION_REFUSAL
        if action == "duplicate":
            return f"{ALREADY_PREFIX} {text}"
        if action == "skipped":
            return "Nothing was kept: the fact did not pass memory checks. State a specific fact."
        if action in ("insert", "update", "replace"):
            return f"{KEPT_PREFIX} {text}"
        log.warning("remember: writer returned an unknown action %r", action)
        return NOT_KEPT

    return remember
