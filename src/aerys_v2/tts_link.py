"""Where the house is playing her voice from.

When she answers aloud, Home Assistant's voice pipeline turns her words into
sound and hands the satellite a short-lived link to it. This module asks Home
Assistant for the link of the pipeline run that is happening right now, so the
desk mirror can be told where her real voice is (owner ask 2026-11-27: the
light in her hands should move with her actual voice, not a stand-in).

It only reads. It is called from the face pusher's daemon thread, after the
panel has been told, and every failure returns None: a house that cannot be
asked costs the turn and the panel nothing.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone

log = logging.getLogger(__name__)

#: A run older than this is not the reply being spoken now.
MAX_AGE_S = 90.0
#: A run that ENDED longer ago than this is finished business. A reply she speaks by the pipeline has its
#: run end within milliseconds of her answering, so the lookup always arrives inside this window. A follow-up
#: she speaks by announcement, seconds or a minute after an earlier acknowledgement, has no run of its own;
#: without this the acknowledgement's sound would be handed over as if it were the follow-up's.
ENDED_GRACE_S = 4.0
#: How many of the newest runs to look at. One with no sound (a typed question, a wake word that led nowhere)
#: must not hide the spoken one beside it.
LOOK_AT = 3


def _when(stamp) -> datetime | None:
    try:
        when = datetime.fromisoformat(stamp)
    except (TypeError, ValueError):
        return None
    return when if when.tzinfo is not None else when.replace(tzinfo=timezone.utc)


def pick_link(runs_with_events: list[tuple[str, list[dict]]], *, now: datetime,
              max_age_s: float = MAX_AGE_S, ended_grace_s: float = ENDED_GRACE_S) -> str | None:
    """The audio path of the newest pipeline run that is the reply being spoken now.

    ``runs_with_events`` is ``[(run timestamp ISO, events), ...]``. The link is
    in the run-start event (``data.tts_output.url``), present before she has
    even answered. A run counts only if it started recently, and has either
    not ended or ended moments ago. Returns the path as Home Assistant gave it,
    or None.
    """
    best: tuple[datetime, str] | None = None
    for stamp, events in runs_with_events:
        when = _when(stamp)
        if when is None:
            continue
        age = (now - when).total_seconds()
        if age < -5 or age > max_age_s:
            continue
        url, ended = None, None
        for ev in events or []:
            if ev.get("type") == "run-start":
                got = ((ev.get("data") or {}).get("tts_output") or {}).get("url")
                if isinstance(got, str) and got.startswith("/api/tts_proxy/"):
                    url = got
            elif ev.get("type") == "run-end":
                ended = _when(ev.get("timestamp")) or when
        if url is None:
            continue
        if ended is not None and (now - ended).total_seconds() > ended_grace_s:
            continue
        if best is None or when > best[0]:
            best = (when, url)
    return best[1] if best else None


async def _ask(base_url: str, token: str, timeout_s: float) -> str | None:
    import websockets

    ws_url = base_url.rstrip("/").replace("http", "ws", 1) + "/api/websocket"
    n = 0

    async def call(ws, payload: dict):
        nonlocal n
        n += 1
        await ws.send(json.dumps({"id": n, **payload}))
        while True:
            msg = json.loads(await ws.recv())
            if msg.get("id") == n:
                return msg.get("result") if msg.get("success") else None

    # The connection gives up well inside the overall limit and closes without waiting on the house, so an
    # unreachable or hung house ends as one quiet exception here, never as a long wait or a stray task.
    async with websockets.connect(ws_url, max_size=None, open_timeout=timeout_s * 0.5,
                                  close_timeout=0.5) as ws:
        await ws.recv()
        await ws.send(json.dumps({"type": "auth", "access_token": token}))
        if json.loads(await ws.recv()).get("type") != "auth_ok":
            return None
        listed = await call(ws, {"type": "assist_pipeline/pipeline/list"}) or {}
        now = datetime.now(timezone.utc)
        newest: list[tuple[str, str, str]] = []         # (timestamp, pipeline id, run id)
        for p in listed.get("pipelines") or []:
            runs = (await call(ws, {"type": "assist_pipeline/pipeline_debug/list",
                                    "pipeline_id": p["id"]}) or {}).get("pipeline_runs") or []
            newest.extend((r["timestamp"], p["id"], r["pipeline_run_id"]) for r in runs[-LOOK_AT:])
        newest.sort(reverse=True)
        seen: list[tuple[str, list[dict]]] = []
        for stamp, pipeline_id, run_id in newest[:LOOK_AT]:
            when = _when(stamp)
            if when is None or (now - when).total_seconds() > MAX_AGE_S:
                continue
            got = await call(ws, {"type": "assist_pipeline/pipeline_debug/get",
                                  "pipeline_id": pipeline_id, "pipeline_run_id": run_id}) or {}
            seen.append((stamp, got.get("events") or []))
        path = pick_link(seen, now=datetime.now(timezone.utc))
        return base_url.rstrip("/") + path if path else None


def current_voice_link(base_url: str, token: str, *, timeout_s: float = 2.0) -> str | None:
    """Full address of the sound being spoken right now, or None. Never raises."""
    try:
        return asyncio.run(asyncio.wait_for(_ask(base_url, token, timeout_s), timeout_s))
    except Exception:
        log.debug("voice link lookup failed (harmless)", exc_info=True)
        return None
