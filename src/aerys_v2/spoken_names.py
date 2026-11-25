"""11/25 (Chris: "She wasn't understanding Eris is Aerys when im speaking"): speech recognition hears her name
as "Eris" (Even's on the glasses wrote "ERIS satellite" for the "Aerys satellite" her music tool lists), so on a
SPOKEN turn (voice, or the glasses' lens) those spellings are read as Aerys before she sees the words. Typed text
is left exactly as written. Narrow on purpose: Iris and Aries are real names and words, so they stay out.
"""
from __future__ import annotations

import re

SPOKEN_AS_AERYS = ("Eris", "Erys", "Aeris", "Errys", "Ayris")
_AS_AERYS = re.compile(r"\b(?:" + "|".join(SPOKEN_AS_AERYS) + r")\b", re.IGNORECASE)


def fix_spoken_names(text: str) -> str:
    return _AS_AERYS.sub("Aerys", text or "")
