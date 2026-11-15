"""Shared, conservative location fence for durable memory and turn disclosure."""
import contextvars
import re
import time
from collections.abc import Iterable

# A fresh mutable marker per ask; copied execution contexts share this object,
# including graph/tool workers. Place strings live only in this request-local marker.
LOCATION_TURN: contextvars.ContextVar[set[str] | None] = contextvars.ContextVar(
    "aerys_location_turn", default=None)
LOCATION_REFUSAL = "Nothing was kept: current whereabouts and coordinates cannot be saved."


# Only timestamps survive a request; no place data is stored in this process map.
RECENT_LOCATION_TURNS: dict[str, float] = {}
LOCATION_LOCK_SECONDS = 24 * 60 * 60


def mark_location_turn(values: Iterable[str] = (), *, person_id: str | None = None) -> None:
    marker = LOCATION_TURN.get()
    if marker is not None:
        marker.add("location")
        for value in values:
            if isinstance(value, str) and value.strip():
                value = value.strip()
                marker.add("place:" + value)
                if address_shaped(value):
                    marker.add("street:" + re.sub(r"^\d+\s+", "", value))   # never excused by a date
    # Call sites pass the identity only after their owner authorization gate.
    if person_id:
        RECENT_LOCATION_TURNS[person_id] = time.monotonic()


def recent_location_turn(person_id: str) -> bool:
    at = RECENT_LOCATION_TURNS.get(person_id)
    return at is not None and time.monotonic() - at < LOCATION_LOCK_SECONDS


_NUMBER = r"[+-]?(?:\d+(?:\.\d+)?|\.\d+)"
_PAIR = re.compile(rf"(?<![\w.])({_NUMBER})(?:\s*,\s*|\s+)({_NUMBER})(?![\w.])")
_LABEL = re.compile(r"\b(?:lat(?:itude)?|lon(?:gitude)?|lng)\b[\s\"']*[:=]?\s*[+-]?\d", re.I)
_JSON_KEY = re.compile(r'["\'](?:latitude|longitude)["\']\s*:', re.I)
#: A bare pair counts as coordinates when both numbers carry 2+ decimals and one carries
#: 3+ (the nearby tool always prints 3; a phone gives 4-7), or when the second is a
#: western longitude (<= -60 with 2+ decimals). Version lists, prices, scores and
#: temperatures ("3.11, 3.12", "98.75, 87.50", "-5.25, 12.50") stay memories (reviews
#: 11/02). Labelled pairs ("lat", "coordinates:") are caught at any precision.
_WHEREABOUTS = re.compile(
    r"\bChris\b[^.!?\n]{0,80}(?:\bis\s+currently\s+(?:in|at)\b|"
    r"\bis\s+in\s+[^.!?\n]+\bright\s+now\b|\bright\s+now\s+in\b|"
    r"\bcurrently\s+staying\s+at\b|\bis\s+near\b)", re.I)
# A place must end before punctuation, end-of-text, or a location qualifier.
_PLACE = r"[A-Z][\w'’-]*(?:\s+[A-Z][\w'’-]*)*(?=\s*(?:$|[,.]|\b(?i:for|on|until|this|today|tonight|with|right)\b))"
_NAMELESS = re.compile(r"\b(?i:currently|right\s+now)\s+(?i:located\s+|staying\s+)?(?i:in|at)\s+(?i:the\s+)?" + _PLACE)
_STAYING = re.compile(r"\b(?i:staying\s+at)\s+(?i:the\s+)?" + _PLACE)
_CURRENT = re.compile(r"\bcurrent\s+(?:city|location|address|hotel)\b|\bphone says he is away from home, in\b", re.I)
_COORDS = re.compile(rf"\b(?:coordinates|coords)\s*:\s*({_NUMBER})(?:\s*,\s*|\s+)({_NUMBER})", re.I)
_MONTH = r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
_END_DATE = re.compile(rf"\b(?:{_MONTH}\s+\d{{1,2}}\s*[-–]\s*\d{{1,2}},?\s+\d{{4}}|until\s+{_MONTH}\s+\d{{1,2}},?\s+\d{{4}})\b", re.I)


#: A street suffix as an address writes it (capitalized). The short forms that are also everyday words
#: or titles ("Dr Smith", "St Pete", "on the way") count only where an address ends them: at the end or
#: before punctuation (Kael 11/15: "2 appointments with Dr Smith" must stay a memory).
def _cased(words: str) -> str:
    """Each suffix as an address writes it: Capitalized, or ALL CAPS (some geocoders), never lowercase."""
    return "|".join(f"{w}|{w.upper()}" for w in words.split())


_STREET_WORD = (r"(?:" + _cased("Street Avenue Ave Road Rd Boulevard Blvd Drive Lane Ln Court Place Highway Hwy "
                                "Parkway Pkwy Circle Terrace Trail Trl") + r")\b")
#: ...unless a Capitalized name follows (St Pete, Dr. Smith); a unit after the street still counts, and so
#: does a new sentence ("1 Example St. He enjoyed it": Codex 11/15), told by its usual first words.
_STARTERS = ("He|She|It|They|We|I|You|The|A|An|This|That|These|Those|Then|There|Here|His|Her|Their|Our|My|"
             "And|But|So|After|Before|Later|Today|Tonight|Yesterday|Tomorrow|Also|Afterward|Afterwards")
_STREET_SHORT = (r"(?:" + _cased("St Dr Pl Ct Ter Cir Way") + r")\b"
                 r"(?!\.?\s+(?!(?:Apt|Apartment|Suite|Ste|Unit|APT|SUITE|STE|UNIT|" + _STARTERS + r")\b)[A-Z])")
_ADDRESS = re.compile(r"\b\d+\s+(?:[\w'’-]+\s+){1,4}(?:" + _STREET_WORD + "|" + _STREET_SHORT + ")")


def address_shaped(text: str | None) -> bool:
    """Recognize a numbered US street address, without blocking place names."""
    return isinstance(text, str) and bool(_ADDRESS.search(text))


def location_memory_blocked(text: str | None) -> bool:
    """Shape checks always apply; dated trips exempt only place provenance."""
    if location_shaped(text):
        return True
    if not isinstance(text, str):
        return False
    marker = LOCATION_TURN.get()
    if not marker or "location" not in marker:
        return False
    def said(value: str) -> bool:
        return bool(re.search(r"(?<!\w)" + re.escape(value) + r"(?!\w)", text, re.I))

    dated = bool(_END_DATE.search(text))
    return any(said(v[6:]) for v in tuple(marker) if v.startswith("place:")
               and (not dated or any(c.isdigit() for c in v[6:]))) \
        or any(said(v[7:]) for v in tuple(marker) if v.startswith("street:"))


def location_shaped(text: str | None) -> bool:
    """Recognize location payloads before embedding or writing decoded fields.

    Dated trips alone are allowed; a date elsewhere never exempts coordinates or
    an explicit current-location statement. This is a shape guard, not an NLU claim.
    """
    if not isinstance(text, str):
        return False
    if _JSON_KEY.search(text) or _LABEL.search(text) or _WHEREABOUTS.search(text) or _NAMELESS.search(text) or _STAYING.search(text) or _CURRENT.search(text) or _COORDS.search(text):
        return True
    for match in _PAIR.finditer(text):
        a, b = match.groups()
        places = [len(n.split('.')[1]) if '.' in n else 0 for n in (a, b)]
        if abs(float(a)) <= 90 and abs(float(b)) <= 180 and (
                (min(places) >= 2 and max(places) >= 3) or (float(b) <= -60 and places[1] >= 2)):
            return True
    return False
