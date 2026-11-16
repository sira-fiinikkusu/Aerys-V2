"""D14 (11/09; kept as Stage 4 work 11/15): she reads her own code, so a failure report can point at the
exact line. His words: "she can be more proactive in her reports as to why something failed when she
reports it to you". Read-only and her own package only; owner turns only (her wake-ups run as him).
D22: she never changes her own code. tests/test_my_code.py probes that, and fails if a tool that
writes files ever appears unregistered."""
from __future__ import annotations

import stat
from itertools import islice
from pathlib import Path

from langchain_core.tools import tool
from langgraph.prebuilt import ToolRuntime

MAX_HITS = 30         # a search past this asks her to narrow it
AROUND = 30           # lines either side of the line she asks for
MAX_LINES = 120       # a whole-file read stops here and says how long the file is
_LINE_CHARS = 200
MAX_FILE_BYTES = 1_000_000   # her biggest file is under 200 KB; anything past this isn't opened
MAX_FILES = 2_000             # her package is about 115 files; a search never walks past this


def package_root() -> Path:
    import aerys_v2

    return Path(aerys_v2.__file__).resolve().parent


def _readable(f: Path, root: Path) -> bool:
    """A regular file inside root and small enough to read. A pipe or device named *.py, or a giant file,
    is never opened (Codex review 11/15)."""
    try:
        st = f.stat()
        return stat.S_ISREG(st.st_mode) and st.st_size <= MAX_FILE_BYTES and f.resolve().is_relative_to(root)
    except (OSError, RuntimeError):
        return False


def _resolve(root: Path, path: str) -> Path | None:
    """A .py file inside root, or None. Absolute paths and anything resolving outside (.., symlinks) are refused."""
    raw = str(path or "").strip().replace("\\", "/")
    if not raw or raw.startswith("/") or ":" in raw:
        return None
    for prefix in ("src/", "aerys_v2/"):
        if raw.startswith(prefix):
            raw = raw[len(prefix):]
    try:
        target = (root / raw).resolve()
    except (OSError, RuntimeError):
        return None
    if not target.is_relative_to(root) or target.suffix != ".py" or not _readable(target, root):
        return None
    return target


def _lines(target: Path) -> list[str]:
    with open(target, encoding="utf-8", errors="replace") as f:
        return f.read().splitlines()


def search(root: Path, text: str) -> str:
    needle = str(text or "").strip().casefold()
    if len(needle) < 3:
        return "Search for at least 3 characters: a function name, or words from the error."
    hits, more = [], False
    for f in sorted(islice(root.rglob("*.py"), MAX_FILES)):
        if "__pycache__" in f.parts or not _readable(f, root):
            continue
        for i, line in enumerate(_lines(f), 1):
            if needle in line.casefold():
                if len(hits) == MAX_HITS:
                    more = True
                    break
                hits.append(f"{f.relative_to(root)}:{i}: {line.strip()[:_LINE_CHARS]}")
        if more:
            break
    if not hits:
        return f'Nothing in my code matches "{text}".'
    tail = f"\n(More than {MAX_HITS} matches; search for something more specific.)" if more else ""
    return "\n".join(hits) + tail


def read(root: Path, path: str, line: int = 0) -> str:
    target = _resolve(root, path)
    if target is None:
        return (f'"{path}" isn\'t one of my code files. Use a path from a search result, '
                "like tools/web_read.py.")
    lines = _lines(target)
    if not lines:
        return f"{target.relative_to(root)} is empty."
    if line and 1 <= int(line) <= len(lines):
        start, end = max(1, int(line) - AROUND), min(len(lines), int(line) + AROUND)
    else:
        start, end = 1, min(len(lines), MAX_LINES)
    body = "\n".join(f"{n:>5}| {lines[n - 1][:_LINE_CHARS]}" for n in range(start, end + 1))
    return f"{target.relative_to(root)}, lines {start}-{end} of {len(lines)}:\n{body}"


def build_read_my_code_tool(owner_person_id, *, root: Path | None = None):
    owner = str(owner_person_id or "")
    base = root or package_root()

    @tool
    def read_my_code(runtime: ToolRuntime, search_for: str = "", path: str = "", line: int = 0) -> str:
        """Read my own code (read-only). CALL THIS when something of mine failed and I'm explaining why,
        to Chris or in a report to Kael: search for the function name or the error's words, then read the
        lines around the hit, and name the file and line in my report.

        search_for: words to find (returns file:line matches). path: a file from a search result, like
        tools/web_read.py. line: the line to center on (0 reads the start of the file).
        I can't change my own code; a fix goes through Kael.
        """
        try:
            ident = (runtime.config.get("configurable") or {}).get("identity") or {}
        except Exception:
            ident = {}
        if not owner or str(ident.get("user_id") or "") != owner:
            return "Not available: only Chris's turns and my own wake-ups can read my code."
        try:
            if path:
                return read(base, path, line)
            if search_for:
                return search(base, search_for)
            return "Give search_for (words to find) or path (a file to read)."
        except Exception:
            return "I couldn't read my code just now; say so rather than guessing at the cause."

    return read_my_code
