"""Secrets and card numbers never become memories.

A memory is recalled into later prompts, synced to her other bodies and backed up,
so a key or a card number that lands in one is copied everywhere she goes. Until
2026-11-24 nothing stopped that: a key pasted into chat could be mined by the
extractor or kept on purpose with the remember tool; only locations were fenced
(location_guard). This is the same kind of fence for secrets, used by the remember
tool, parse_observations and triage_memory alike.

What it refuses, each by its shape, never by guessing:
  - card numbers: 13-19 digits in a card's shape, a major network's prefix, and the
    Luhn check. Mirrors the Portable body's home wall (aerys_portable/hands/home_wall.py,
    card_number and friends), INCLUDING its deliberately narrow network list (Visa,
    Mastercard, Amex, Discover): order numbers, tracking numbers and timestamps pass.
  - JWTs, PEM private keys, and `scheme://user:pass@` links.
  - obvious API-key formats (sk-, ghp_/github_pat_, xox?-, AKIA/ASIA, AIza, sk_live_),
    and a long value with letters and digits right after a token/secret/password/API-key
    label, read whole (punctuation and all). A bare "key" counts only as an assignment
    ("key: ...") and never for a dashed word identifier ("part-number-2026-revision-2").

A short human password ("the wifi password is bluefish") is NOT caught: it has no
shape to tell it from an ordinary word. Whether she may keep those is Chris's call.
Log the kind, never the value.
"""
from __future__ import annotations

import re
import unicodedata

SECRET_REFUSAL = ("Nothing was kept: I don't keep secrets or card numbers in my memory "
                  "(passwords, keys, tokens, card numbers), so none of it was saved.")


# --- card numbers (mirrors home_wall.py) -----------------------------------------------

def _norm(text: str) -> str:
    """NFKC, invisible format characters dropped (a zero-width space between digits),
    casefolded, and runs of whitespace made one space: a number wrapped onto two lines
    or double-spaced is still one run of digit groups (home_wall reads form fields,
    which never wrap; chat does)."""
    text = unicodedata.normalize("NFKC", text)
    text = "".join(c for c in text if unicodedata.category(c) != "Cf").casefold()
    return " ".join(text.split())


def _digits(text: str) -> str:
    """Every decimal digit in TEXT, in order, as ASCII."""
    return "".join(str(unicodedata.decimal(c)) for c in text if c.isdecimal())


def luhn(number: str) -> bool:
    total, odd = 0, False
    for ch in reversed(number):
        d = int(ch)
        if odd:
            d = d * 2 - 9 if d > 4 else d * 2
        total += d
        odd = not odd
    return total % 10 == 0


def card_network(number: str) -> str | None:
    """The network a 13-19 digit NUMBER's prefix belongs to, or None: Visa 4; Mastercard
    51-55 and 2221-2720; Amex 34/37; Discover 6011, 65, 644-649. Narrow on purpose."""
    if not (13 <= len(number) <= 19 and number.isdigit()):
        return None
    if number[0] == "4":
        return "visa"
    if 51 <= int(number[:2]) <= 55 or 2221 <= int(number[:4]) <= 2720:
        return "mastercard"
    if number[:2] in ("34", "37"):
        return "amex"
    if number.startswith(("6011", "65")) or 644 <= int(number[:3]) <= 649:
        return "discover"
    return None


_RUN = re.compile(r"\d+(?:[ \-]\d+)*")


def _card_spans(groups: list[str]) -> set[str]:
    """The card-shaped candidates in a run of digit groups: a group of 13-19 alone, or two
    or more groups led by a group of four (4-4-4-4, Amex 4-6-5), middle groups of 4-6, the
    last of 1-6. Never a sliding window: a ZIP then a phone is not a card."""
    out = {g for g in groups if 13 <= len(g) <= 19}
    for i, lead in enumerate(groups):
        if len(lead) != 4:
            continue
        for j in range(i + 1, len(groups)):
            if not 1 <= len(groups[j]) <= 6:
                break
            out.add("".join(groups[i:j + 1]))
            if len(groups[j]) < 4:
                break                       # a short group can only end a card
    return out


def card_number_in(text: str) -> bool:
    """A number in a card's shape, from a card network's prefix, that passes Luhn."""
    for run in _RUN.finditer(_norm(text)):
        groups = [_digits(g) for g in re.split(r"[ \-]", run.group(0)) if g]
        if any(card_network(c) and luhn(c) for c in _card_spans(groups)):
            return True
    return False


# --- keys, tokens, credentials -----------------------------------------------------------

_PRIVATE_KEY = re.compile(r"-{3,}\s*BEGIN [A-Z0-9 ]*PRIVATE KEY", re.I)
_JWT = re.compile(r"(?<![A-Za-z0-9_-])eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]*")
_URL_CREDENTIALS = re.compile(r"\b[a-z][a-z0-9+.\-]{1,20}://[^\s/:@]*:[^\s/@]+@[^\s/@]", re.I)
#: Each needs a non-alphanumeric character before it, so "risk-averse" or "Ask-..." never
#: trip sk-. The two whose prefix is also an ordinary word start ("sk-", "xoxo-") must
#: also look random (letters AND digits); the rest are distinctive by prefix and length.
_API_KEYS = [(re.compile(p), needs_random) for p, needs_random in (
    (r"(?<![A-Za-z0-9_-])sk-[A-Za-z0-9_\-]{20,}", True),                      # OpenAI, Anthropic, OpenRouter
    (r"(?<![A-Za-z0-9_-])xox[a-z]-[A-Za-z0-9-]{10,}", True),                  # Slack
    (r"(?<![A-Za-z0-9_])(?:sk|rk)_(?:live|test)_[A-Za-z0-9]{16,}", False),    # Stripe secret / restricted
    (r"(?<![A-Za-z0-9_])gh[pousr]_[A-Za-z0-9]{30,}", False),                  # GitHub
    (r"(?<![A-Za-z0-9_])github_pat_[A-Za-z0-9_]{30,}", False),
    (r"(?<![A-Za-z0-9])(?:AKIA|ASIA)[A-Z0-9]{16}(?![A-Za-z0-9])", False),     # AWS access key id
    (r"(?<![A-Za-z0-9_-])AIza[0-9A-Za-z_\-]{35}", False),                     # Google API key
)]
#: The value after a label, read WHOLE (Codex review round 1: a password with "!" in it
#: was cut at the "!" and then too short to count): a quoted run, or a bare run up to
#: whitespace. Never across whitespace, even quoted. A quote mark or backtick opens a
#: quoted run only at the value's START and only if it closes at the value's end
#: (round 2: one inside a bare value used to end it early); otherwise it is part of it.
_VALUE_END = r"""(?=[\s.,;:!?)\]}>]|$)"""
_VALUE = (r"""(?:(?:"([^"\s]{1,200})"|'([^'\s]{1,200})'|`([^`\s]{1,200})`)""" + _VALUE_END
          + r"|(\S{1,200}))")
#: Labels whose value IS a credential: "my token is ...", "password: ...", "API_KEY=...".
_LABEL = (r"(?:api[ _-]?key|access[ _-]?key|secret[ _-]?key|private[ _-]?key|client[ _-]?secret"
          r"|secret|token|bearer|password|passwd|passphrase|pwd)")
_LABELLED = re.compile(r"(?<![a-z])" + _LABEL + r"(?![a-z])[\"']?\s*(?:[:=]|\bis\b|\bwas\b)?\s*" + _VALUE,
                       re.I)
#: A bare "key" is everyday English ("the key is to keep the RPM down"), so it counts only
#: as an assignment ("key: ...", "SERVICE_KEY=..."), and only for a random-looking value.
_BARE_KEY = re.compile(r"(?<![a-z])key(?![a-z])[\"']?\s*[:=]\s*" + _VALUE, re.I)
#: Sentence punctuation after a bare value is the sentence's, not the value's.
_TRAILING = ".,;:)]}>"


def _whole_value(match: re.Match) -> str:
    quoted = match.group(1) or match.group(2) or match.group(3)
    return quoted if quoted else match.group(4).rstrip(_TRAILING)


def _structured(value: str) -> bool:
    """Words and numbers joined by - _ . / ("part-number-2026-revision-v2"): an
    identifier someone typed, not a random one. Judged by runs: a piece that switches
    between letters and digits twice or more ("a1B2", "123e4567") reads as random; "v2"
    or "2026" does not (Codex review round 1)."""
    pieces = [p for p in re.split(r"[-_./]+", value) if p]
    return bool(pieces) and all(len(re.findall(r"[^\W\d_]+|\d+", p)) < 3 and p.isalnum() for p in pieces)


def _random_looking(token: str) -> bool:
    """A long value with both letters and digits: what a key looks like, and what a
    word or a plain number does not."""
    token = token.rstrip("=")
    return len(token) >= 20 and any(c.isdigit() for c in token) and any(c.isalpha() for c in token)


def secret_kind(text: object) -> str | None:
    """The kind of secret TEXT carries ('card number', 'private key', 'access token',
    'credentials in a link', 'API key', 'labelled secret'), or None. Safe to log."""
    if not isinstance(text, str) or not text:
        return None
    if card_number_in(text):
        return "card number"
    if _PRIVATE_KEY.search(text):
        return "private key"
    if _JWT.search(text):
        return "access token"
    if _URL_CREDENTIALS.search(text):
        return "credentials in a link"
    if any(not needs_random or _random_looking(m.group(0))
           for pattern, needs_random in _API_KEYS for m in pattern.finditer(text)):
        return "API key"
    if any(_random_looking(_whole_value(m)) for m in _LABELLED.finditer(text)):
        return "labelled secret"
    if any(_random_looking(v) and not _structured(v) for v in map(_whole_value, _BARE_KEY.finditer(text))):
        return "labelled secret"
    return None


def secret_shaped(text: object) -> bool:
    return secret_kind(text) is not None
