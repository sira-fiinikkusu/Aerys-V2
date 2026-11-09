"""Bodies whose turns are tests, never his words (board #176, 11/09).

Every portable body records its turns at the door as the owner's, `portable:<body>:<seq>`
under his person id. Kael's gate body (ember) does that on every gate run, and the home
body's test jobs did it too: by 11/09 her portable block (newest 100) held 63 ember lines
shown to her as "[ember] Chris: ...", and the extractor read them as his. The turns stay in
v2_turns (a record of what ran); only the readers that treat a turn as his speech or her
context skip these bodies.

PORTABLE_TEST_BODIES: the body names, comma-separated (default "ember,home"). A name is
lowercase letters, digits, '-' or '_'; anything else is ignored, never interpolated.
"""
from __future__ import annotations

import logging
import re

log = logging.getLogger(__name__)

DEFAULT = "ember,home"
_NAME = re.compile(r"^[a-z0-9][a-z0-9_-]{0,31}$")

#: The SQL each reader adds (named or positional parameter): a portable turn of a test body
#: never counts. A turn with no channel id is unaffected.
CLAUSE_NAMED = "NOT (COALESCE({col}::text, '') LIKE ANY(%(test_bodies)s::text[]))"
CLAUSE_POSITIONAL = "NOT (COALESCE({col}::text, '') LIKE ANY(%s::text[]))"


def names(raw: str | None = None) -> list[str]:
    """The test bodies' names from a comma list (DEFAULT when None)."""
    out: list[str] = []
    for part in str(DEFAULT if raw is None else raw).split(","):
        name = part.strip().lower()
        if not name:
            continue
        if not _NAME.match(name):
            log.warning("PORTABLE_TEST_BODIES: %r isn't a body name; ignored", name[:40])
            continue
        if name not in out:
            out.append(name)
    return out


def patterns(raw: str | None = None) -> list[str]:
    """LIKE patterns for their portable channel ids, 'portable:<body>:%' ('_' escaped)."""
    return ["portable:" + name.replace("_", "\\_") + ":%" for name in names(raw)]


def patterns_for(settings) -> list[str]:
    """The patterns from Settings (DEFAULT when the setting isn't there)."""
    return patterns(getattr(settings, "portable_test_bodies", None))


DEFAULT_PATTERNS = patterns(DEFAULT)
