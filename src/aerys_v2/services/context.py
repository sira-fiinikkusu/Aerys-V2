"""Context assembly — profile + memories folded into one prompt block.

n8n mapping: in V1 the Core Agent's prompt-builder Code node called TWO webhooks
per message (04-03 Profile API, 04-02 Memory Retrieval) and spliced their outputs
into the system prompt. Here both are plain function calls against one injected
connection, and the splice is a string join.

The GRACEFUL contract is the load-bearing part: a broken memory pipeline must
degrade to "no context" — it must NEVER take the conversational turn down with
it. (In n8n a failed webhook killed the whole execution; Aerys went mute because
a SELECT hiccupped. That failure class is deleted here.) So: no connection, no
recognizable person, no rows, or an exception in EITHER half → the other half
still emits, and the worst case is an empty string the chat node simply skips.
"""

import json
import logging
import urllib.request
import uuid
from typing import Any

from aerys_v2.services.memory import (
    EMBED_MODEL,
    Embedder,
    format_memory_context,
    retrieve_memories,
)
from aerys_v2.services.profile import get_profile

log = logging.getLogger(__name__)


def _is_uuid(value: str) -> bool:
    """person_id must be a real UUID before it goes anywhere near `::uuid` SQL.

    Transports mint non-UUID identities ("cli-operator", "discord:12345") until
    the DB resolver maps them; those are "no person" here — skip the roundtrip
    entirely instead of letting Postgres throw on the cast every single turn.
    """
    try:
        uuid.UUID(str(value))
        return True
    except (ValueError, AttributeError, TypeError):
        return False


def build_context(
    person_id: str,
    query_text: str,
    conn: Any,
    *,
    embed: Embedder | None = None,
    privacy_context: str = "private",
) -> str:
    """Everything the brain knows about this person, as one prompt-ready block.

    Profile first (who they ARE — stable claims), memories second (what happened
    lately, scored against the current message) — the same order the V1 prompt
    builder spliced them. Returns '' when there is nothing to say; the caller
    injects nothing rather than an empty header.

    privacy_context defaults to 'private' because today's only wired caller is
    the owner's own channel (voice / HTTP behind the Bearer token). A future
    guild transport passes 'public' and the P-level/visibility gates in the
    services do the filtering — the assembly logic here doesn't change.
    """
    if conn is None or not _is_uuid(person_id):
        return ""

    parts: list[str] = []

    # Each half is independently fenced: profile trouble must not cost the
    # memories, and vice versa. Losing context is annoying; losing the TURN
    # (an exception bubbling into the chat node) is the V1 outage mode.
    try:
        profile = get_profile(conn, person_id, privacy_context)["profile"]
        if profile["lines"]:
            parts.append("\n".join(profile["lines"]))
    except Exception:
        # graceful: no profile, not a dead turn — but degrade-graceful must not
        # mean degrade-SILENT: an invisible fence hid a broken pipeline once.
        log.warning("profile context failed for person %s", person_id, exc_info=True)

    try:
        # No embed seam = no way to score memories against the query; the
        # profile half still stands. (retrieve_memories would raise ValueError
        # here — this guard makes the degradation intentional, not accidental.)
        if embed is not None:
            rows = retrieve_memories(
                conn,
                person_id,
                query_text=query_text,
                embed=embed,
                privacy_context=privacy_context,
            )
            memory_block = format_memory_context(rows)
            if memory_block:
                parts.append(f"Relevant memories:\n{memory_block}")
    except Exception:
        # graceful: no memories, not a dead turn — logged for the same reason
        # as the profile fence (embed HTTP failures land here too).
        log.warning("memory context failed for person %s", person_id, exc_info=True)

    return "\n\n".join(parts)


def embedder_from_settings(settings: Any) -> Embedder | None:
    """Build the real embed seam from Settings, or None when unconfigured.

    Mirrors memory.openrouter_embedder (the n8n "Generate Embedding" node) but
    honors embeddings_base_url, so any OpenAI-compatible /embeddings host works.
    stdlib urllib on purpose — same zero-dependency choice as memory.py; offline
    tests inject a fake Embedder and never reach this function.
    """
    if settings.embeddings_api_key is None:
        return None
    api_key = settings.embeddings_api_key.get_secret_value()
    base_url = settings.embeddings_base_url.rstrip("/")

    def embed(text: str) -> list[float]:
        request = urllib.request.Request(
            f"{base_url}/embeddings",
            data=json.dumps({"model": EMBED_MODEL, "input": text}).encode(),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
        )
        with urllib.request.urlopen(request, timeout=30.0) as resp:
            payload = json.load(resp)
        return payload["data"][0]["embedding"]

    return embed
