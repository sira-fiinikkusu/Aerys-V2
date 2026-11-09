"""The names of the people in Chris's life that her prompts use, kept out of this public repo.

The owner, 2026-11-09: "i would appreciate we dont air out anyone but me". So the names that
her voice gate, her router and her house notes say out loud (his wife, the friends and colleagues
who come up by name, the names in the examples) live in the private .env as one JSON object,
AERYS_HOUSE_NAMES, and never in tracked files:

  wife       his wife's first name
  friends    the friends/colleagues her voice gate knows come up by name, as the one line it says
             ("A, B, C, D")
  friend     the friend in the router's example of talking to someone else ("hang on ___, ...")
  colleague  the colleague in the voice gate's example ("___, it's a, uh, a way to track outages")
  aside      the name in the voice gate's TV-and-radio example ("He interrupted ___. ...")

Read once at import, like the rest of her prompts (they're module constants). With the real values
the prompts are byte-identical to what they were when the names were written into the code.

Missing or unreadable: one loud warning, and every prompt falls back to wording that names nobody.
Never a made-up name for a real person: a wrong name for his wife is worse than none.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from dataclasses import dataclass

log = logging.getLogger(__name__)

ENV = "AERYS_HOUSE_NAMES"
#: The env file Settings reads (config.Settings: env_file=".env", from where she runs).
ENV_FILE = ".env"
KEYS = ("wife", "friends", "friend", "colleague", "aside")


@dataclass(frozen=True)
class HouseNames:
    wife: str = ""
    friends: str = ""
    friend: str = ""
    colleague: str = ""
    aside: str = ""


def _from_env_file(path) -> str:
    """AERYS_HOUSE_NAMES as the env file Settings reads would give it ('' when absent or
    unreadable). Docker's --env-file puts it in the environment before she starts; a CLI run reads
    .env through Settings, which a prompt module imported first never sees (Codex, 11/09)."""
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except (OSError, ValueError):
        return ""
    for line in lines:
        key, sep, value = line.partition("=")
        if sep and key.strip().removeprefix("export ").strip() == ENV:
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
                value = value[1:-1]
            return value
    return ""


def load(env=None, env_file=None) -> HouseNames:
    """The names from AERYS_HOUSE_NAMES; empty ones (and a warning) when it's missing or broken.
    The variable wins; without it, the env file (Settings' .env when reading the real environment).
    Never raises: she starts without the names rather than not at all."""
    if env is None:
        env, env_file = os.environ, ENV_FILE if env_file is None else env_file
    raw = env.get(ENV, "")
    if (not isinstance(raw, str) or not raw.strip()) and env_file is not None:
        raw = _from_env_file(env_file)
    if not isinstance(raw, str) or not raw.strip():
        log.warning("house names: %s isn't set; her prompts name nobody (no wife's name, no friends) "
                    "until it is", ENV)
        return HouseNames()
    try:
        data = json.loads(raw)
    except (ValueError, RecursionError):        # deeply nested JSON raises RecursionError (Codex)
        log.warning("house names: %s isn't valid JSON; her prompts name nobody until it's fixed", ENV)
        return HouseNames()
    if not isinstance(data, dict):
        log.warning("house names: %s isn't a JSON object; her prompts name nobody until it's fixed", ENV)
        return HouseNames()
    picked = {k: data[k].strip() for k in KEYS if isinstance(data.get(k), str) and data[k].strip()}
    missing = [k for k in KEYS if k not in picked]
    if missing:
        log.warning("house names: %s has no %s; those parts of her prompts name nobody", ENV, ", ".join(missing))
    return HouseNames(**picked)


#: Her house's names, read once when she starts.
NAMES = load()


def wife_or(names: HouseNames, fallback: str = "his wife") -> str:
    """His wife's name where a sentence needs a who, else the fallback ("his wife")."""
    return names.wife or fallback
