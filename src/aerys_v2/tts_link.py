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


def pick_link(runs_with_events: list[tuple[str, list[dict]]], *, now: datetime,
              max_age_s: float = MAX_AGE_S) -> str | None:
    """The audio path of the newest pipeline run that is recent enough.

    ``runs_with_events`` is ``[(run timestamp ISO, events), ...]``. The link is
    in the run-start event (``data.tts_output.url``), present before she has
    even answered. Returns the path as Home Assistant gave it, or None.
    """
    best: tuple[datetime, str] | None = None
    for stamp, events in runs_with_events:
        try:
            when = datetime.fromisoformat(stamp)
        except (TypeError, ValueError):
            continue
        if when.tzinfo is None:
            when = when.replace(tzinfo=timezone.utc)
        age = (now - when).total_seconds()
        if age < -5 or age > max_age_s:
            continue
        for ev in events or []:
            if ev.get("type") != "run-start":
                continue
            url = ((ev.get("data") or {}).get("tts_output") or {}).get("url")
            if isinstance(url, str) and url.startswith("/api/tts_proxy/"):
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

    # The connection gives up well inside the overall limit, so an unreachable house ends as one
    # quiet exception here, never as a cancelled task that the event loop reports on its own.
    async with websockets.connect(ws_url, max_size=None, open_timeout=timeout_s * 0.5) as ws:
        await ws.recv()
        await ws.send(json.dumps({"type": "auth", "access_token": token}))
        if json.loads(await ws.recv()).get("type") != "auth_ok":
            return None
        listed = await call(ws, {"type": "assist_pipeline/pipeline/list"}) or {}
        now = datetime.now(timezone.utc)
        newest: tuple[str, str, str] | None = None      # (timestamp, pipeline id, run id)
        for p in listed.get("pipelines") or []:
            runs = (await call(ws, {"type": "assist_pipeline/pipeline_debug/list",
                                    "pipeline_id": p["id"]}) or {}).get("pipeline_runs") or []
            if runs and (newest is None or runs[-1]["timestamp"] > newest[0]):
                newest = (runs[-1]["timestamp"], p["id"], runs[-1]["pipeline_run_id"])
        if newest is None:
            return None
        got = await call(ws, {"type": "assist_pipeline/pipeline_debug/get",
                              "pipeline_id": newest[1], "pipeline_run_id": newest[2]}) or {}
        path = pick_link([(newest[0], got.get("events") or [])], now=now)
        return base_url.rstrip("/") + path if path else None


def current_voice_link(base_url: str, token: str, *, timeout_s: float = 3.0) -> str | None:
    """Full address of the sound being spoken right now, or None. Never raises."""
    try:
        return asyncio.run(asyncio.wait_for(_ask(base_url, token, timeout_s), timeout_s))
    except Exception:
        log.debug("voice link lookup failed (harmless)", exc_info=True)
        return None
