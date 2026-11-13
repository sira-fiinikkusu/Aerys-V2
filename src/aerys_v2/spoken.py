"""What she says aloud after a spoken ack (Chris 11/13 18:10, "would not shut up").

A researched voice answer was read aloud whole: 998 characters, bullets and all. What's SPOKEN after an
ack is plain speech, cut at a sentence near SPOKEN_MAX, then one short line that there's more. A
sentence that says how things stand (waiting on his tap, it failed, nothing was sent) is never the
part that gets cut. The full text still goes wherever it is READ (her history, an ink panel).
"""
from __future__ import annotations

import re

SPOKEN_MAX = 300
SPOKEN_MORE = "There's more if you want it."
_MD = re.compile(r"(\*\*|__|`+|^#+\s*|^\s*[-*•]\s+|^\s*\d+[.)]\s+)", re.M)
_LINK = re.compile(r"\[([^\]]+)\]\((?:https?://)[^)]+\)")
_URL = re.compile(r"https?://\S+")
_STATUS = re.compile(r"\b(tap|approve|approval|waiting|failed|couldn't|could not|didn't go|did not go|"
                     r"not sent|wasn't sent|nothing (?:was|has been|went))\b", re.I)


def spoken(text: str, limit: int = SPOKEN_MAX) -> str:
    plain = _URL.sub("", _LINK.sub(r"\1", _MD.sub("", str(text or ""))))
    lines = [ln.strip() for ln in plain.splitlines() if ln.strip()]
    joined = " ".join(ln if (ln[-1] in ".!?:" or len(lines) == 1) else ln + "." for ln in lines)
    plain = re.sub(r"\s{2,}", " ", joined).strip()
    if len(plain) <= limit:
        return str(text) if plain == str(text or "").strip() else plain   # nothing to change: as it was
    sentences = re.split(r"(?<=[.!?])\s+", plain)
    head: list[str] = []
    for sentence in sentences:
        if len(" ".join([*head, sentence])) > limit:
            break
        head.append(sentence)
    rest = sentences[len(head):]
    if not head:                                   # one long first sentence: cut it at a word
        head, rest = [sentences[0][:limit].rsplit(" ", 1)[0].rstrip(",;:") + "."], sentences[1:]
    status = next((s for s in rest if _STATUS.search(s) and len(s) <= 200), None)
    return " ".join([*head, *([status] if status else []), SPOKEN_MORE])
