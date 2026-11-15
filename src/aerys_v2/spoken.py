"""What she says aloud after a spoken ack: plain speech.

11/13 (Chris, "would not shut up"): a researched voice answer was read aloud whole, 998 characters,
bullets and all. The fix then also cut every spoken follow-up near 300 characters and added "There's
more if you want it."; on 11/15 that chopped his morning greeting, and his ruling (D55) was: "we
shouldnt be doing hard 300 character cuts by code. we want her to speak less but if she speaks longer
it should be allowed. its taught not coded". So nothing here shortens what she says: how long she
speaks is taught (factory.VOICE_ACK_OVERLAY, VOICE_BANTER_OVERLAY). What stays is the cleanup no
speaker should read out: markdown, bullets, links and bare URLs. The full text still goes wherever it
is READ (her history, an ink panel).
"""
from __future__ import annotations

import re

_MD = re.compile(r"(\*\*|__|`+|^#+\s*|^\s*[-*•]\s+|^\s*\d+[.)]\s+)", re.M)
_LINK = re.compile(r"\[([^\]]+)\]\((?:https?://)[^)]+\)")
_URL = re.compile(r"https?://\S+")
_TAG_ONLY = re.compile(r"^(\[[a-z ]+\]\s*)+$", re.I)


def spoken(text: str) -> str:
    plain = _URL.sub("", _LINK.sub(r"\1", _MD.sub("", str(text or ""))))
    lines: list[str] = []
    for ln in (ln.strip() for ln in plain.splitlines()):
        if not ln:
            continue
        if lines and _TAG_ONLY.match(lines[-1]):      # a tag on its own line belongs to the words after it
            lines[-1] = f"{lines[-1]} {ln}"
        else:
            lines.append(ln)
    joined = " ".join(ln if (ln[-1] in ".!?:" or len(lines) == 1) else ln + "." for ln in lines)
    plain = re.sub(r"\s{2,}", " ", joined).strip()
    return str(text) if plain == str(text or "").strip() else plain   # nothing to change: as it was
