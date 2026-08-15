"""A2A exchanges leave a durable memory (capability gap #37, owner-approved).

Found live 2026-08-14, demo night: the owner asked her to check in with Kael,
Kael answered through /ask on a kael:* thread — and on her person thread she
knew she *reached* him but not what was said. Threads are isolated rooms by
design; the kael_line (task #66) can put his words into ONE thread's context,
but nothing made the exchange part of what she durably *knows*.

This is door 1 of the family-memory design (the lightweight one, the only one
approved to build solo): after an A2A exchange completes on a kael:* thread,
write a compact record into her existing `memories` table — same embedding
model, same retrieval surface — so "Kael told me X" can surface on ANY thread
the way any other remembered fact does. Doors 2 and 3 (labeled turns in the
owner's own thread, a true shared family room) are design-together and NOT
built here.

Trust model: the /ask door is Bearer-authed owner infrastructure; a kael:*
thread is Kael's line by convention (the same convention kael_line documents).
The memory is attributed in content as an exchange WITH Kael — it never
impersonates the owner's own words. Tests and smoke checks must use non-kael
threads (http:default et al.) exactly so they never enter the family lore —
the same "tests do poison memory" rule the kael_line was built around.

Posture: fail-open, off the hot path. The reply must never be delayed or
crashed by a memory write — embed + insert run on a daemon thread and swallow
everything with a warning, the FacePusher/turn-recorder stance.
"""

from __future__ import annotations

import logging
import threading
from datetime import datetime
from typing import Callable

from aerys_v2.services.memory import embedding_to_pgvector, openrouter_embedder

log = logging.getLogger(__name__)

# One live row per (person_id, key_label): the timestamped label makes every
# exchange its own row instead of a rolling latest — these are rare (a few per
# week) and the dream/consolidation pass owns any later dedup.
_KEY_FMT = "a2a_kael_%Y%m%d_%H%M%S"

# Compact by construction: both sides are clipped so one exchange stays one
# glanceable memory, not a transcript. Retrieval needs the gist; the full text
# lives in v2_turns for anyone doing forensics.
_CLIP = 420

_INSERT_SQL = """\
INSERT INTO memories
  (person_id, content, key_label, context, event_date, embedding,
   source_platform, privacy_level, created_at, updated_at)
VALUES
  (%(person_id)s::uuid, %(content)s, %(key_label)s, %(context)s, %(event_date)s,
   %(embedding)s::vector, 'kael_line', 'private', now(), now())
"""


def _clip(text: str) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= _CLIP else text[: _CLIP - 1] + "…"


def a2a_memory_writer_for(
    settings,
    *,
    embed: Callable | None = None,
    connect: Callable | None = None,
    synchronous: bool = False,
) -> Callable[[str, str, str], None] | None:
    """Build the writer the /ask door fires after a kael:* exchange, or None.

    Unarmed (returns None) unless the durable store, the owner identity, and
    the embeddings credential are all configured — the same three things the
    memory this writes would need to ever be retrieved. embed/connect are
    injectable for offline tests; synchronous=True runs the write inline so
    tests assert without racing a daemon thread.
    """
    database_url = settings.database_url
    owner = settings.owner_person_id
    key = settings.embeddings_api_key
    if not (database_url and owner and key):
        return None

    if embed is None:
        embed = openrouter_embedder(key.get_secret_value())
    if connect is None:
        import psycopg

        connect = psycopg.connect

    def _write_now(thread_id: str, kael_text: str, reply_text: str) -> None:
        try:
            now = datetime.now().astimezone()
            content = (
                f"Talked with Kael on his line ({now:%Y-%m-%d %H:%M}): "
                f'Kael said: "{_clip(kael_text)}" — I replied: "{_clip(reply_text)}"'
            )
            vector = embedding_to_pgvector(embed(content))
            with connect(database_url, connect_timeout=5) as conn:
                conn.execute(
                    _INSERT_SQL,
                    {
                        "person_id": owner,
                        "content": content,
                        "key_label": now.strftime(_KEY_FMT),
                        "context": f"A2A exchange on {thread_id}",
                        "event_date": now.date(),
                        "embedding": vector,
                    },
                )
                conn.commit()
            log.info("a2a memory written for %s", thread_id)
        except Exception:
            log.warning("a2a memory write failed (reply already delivered)",
                        exc_info=True)

    def write(thread_id: str, kael_text: str, reply_text: str) -> None:
        if not str(thread_id).startswith("kael:"):
            return
        if not str(kael_text).strip() or not str(reply_text).strip():
            return
        if synchronous:
            _write_now(thread_id, kael_text, reply_text)
            return
        try:
            threading.Thread(
                target=_write_now,
                args=(thread_id, kael_text, reply_text),
                daemon=True,
            ).start()
        except RuntimeError:  # thread exhaustion — drop the memory, never the turn
            log.warning("a2a memory thread could not start", exc_info=True)

    return write
