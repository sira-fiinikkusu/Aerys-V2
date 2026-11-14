"""Satellite voice — she speaks through the satellite's own speaker, and a stop reaches it there.

11/14 (the Pebble): the office satellite gets a speaker back, so its voice moves off the desk
redirect and onto the satellite, where the XVF3800 hears her own output and cancels it. The
satellite's firmware already cuts her off itself on "stop" or her wake word while she talks; this
is the brain's half: the office button stops her there, and a stop holds what's still on its way.

Probed 11/14 on the office satellite, silently (no speaker yet): its assist_satellite entity reads
"responding" from the moment an answer or announce starts (before any sound) until it ends. The
satellite's own ``stop_va`` action cuts either in about 0.25 s; ``media_player.media_stop`` does not
(it left the announcement playing to its end). The action is named for the ESPHome node, which the
entity id carries: ``assist_satellite.<node>_assist_satellite`` -> ``esphome.<node>_stop_va``. A
satellite without that action (the Voice PE) fails the stop, and the caller says so.
"""

from __future__ import annotations

import logging
import threading
import time

import httpx

log = logging.getLogger(__name__)

_PREFIX, _SUFFIX = "assist_satellite.", "_assist_satellite"
#: states that mean she isn't speaking there; anything else but "responding" is unknown (Codex 11/14)
QUIET = frozenset({"idle", "listening", "processing"})
#: a follow-up that passed its hush check just before the stop can still reach HA a moment after it
#: (Codex 11/14); for this long after a stop, a satellite that starts talking again for no newer turn
#: is stopped again. HA reads "responding" ~0.3 s after an announce is sent.
GUARD_S = 3.0


def stop_action(entity_id: str) -> str | None:
    """The satellite's own stop action, from its entity id; None when the id isn't in that shape."""
    if not (entity_id.startswith(_PREFIX) and entity_id.endswith(_SUFFIX)):
        return None
    node = entity_id[len(_PREFIX):-len(_SUFFIX)]
    return f"{node}_stop_va" if node else None


class SatelliteVoice:
    def __init__(self, *, ha_base_url: str, ha_token: str, satellites: dict[str, str],
                 client: httpx.Client | None = None, settle_s: float = 1.5, guard_s: float = GUARD_S,
                 async_guard: bool = True):
        self._ha = ha_base_url.rstrip("/")
        self._headers = {"Authorization": f"Bearer {ha_token}"}
        self._satellites = dict(satellites)
        self._client = client or httpx.Client(timeout=5.0)
        self._settle_s, self._guard_s, self._async_guard = settle_s, guard_s, async_guard
        self._guards: dict[str, tuple[float, float]] = {}   # device -> (hush floor, deadline)
        self._lock = threading.Lock()

    def handles(self, device_id: str | None) -> bool:
        return bool(device_id) and device_id in self._satellites

    def talking(self, device_id: str | None, timeout: float = 1.5) -> bool | None:
        """Is she answering or announcing there? True, False, or None when HA can't say (a failed
        read, or the satellite unavailable): never guessed as quiet."""
        if not self.handles(device_id):
            return False
        try:
            r = self._client.get(f"{self._ha}/api/states/{self._satellites[device_id]}",
                                 headers=self._headers, timeout=timeout)
            r.raise_for_status()
            state = r.json().get("state")
        except Exception:
            log.debug("satellite talking check failed", exc_info=True)
            return None
        if not isinstance(state, str):
            return None
        if state == "responding":
            return True
        return False if state in QUIET else None

    def stop(self, device_id: str | None) -> bool:
        """Stop her there; True only once the satellite reads quiet again."""
        if not self.handles(device_id):
            return False
        action = stop_action(self._satellites[device_id])
        if action is None:
            return False
        try:
            r = self._client.post(f"{self._ha}/api/services/esphome/{action}", headers=self._headers,
                                  json={}, timeout=3.0)
            r.raise_for_status()
        except Exception:
            log.debug("satellite stop failed", exc_info=True)
            return False
        deadline = time.monotonic() + self._settle_s
        while True:
            if self.talking(device_id, timeout=0.5) is False:
                return True
            if time.monotonic() >= deadline:
                return False
            time.sleep(0.15)

    def guard(self, device_id: str | None, hushed: float) -> None:
        """For a few seconds after a stop, stop her again whenever she starts talking there for no
        turn that began after it (a follow-up already on its way to HA when he stopped her). Best
        effort, not a cancellation: HA has no turn to check a clip against. One guard per satellite;
        another stop moves its floor and deadline (Codex 11/14)."""
        if not self.handles(device_id):
            return
        with self._lock:
            running = device_id in self._guards
            floor = max(hushed, self._guards[device_id][0]) if running else hushed
            self._guards[device_id] = (floor, time.monotonic() + self._guard_s)
        if running:
            return
        if self._async_guard:
            threading.Thread(target=self._guard, args=(device_id,), daemon=True).start()
        else:
            self._guard(device_id)

    def _guard(self, device_id: str) -> None:
        from .speaker_redirect import last_turn

        try:
            while True:
                with self._lock:
                    hushed, deadline = self._guards[device_id]
                    if time.monotonic() >= deadline or last_turn(device_id) > hushed:
                        del self._guards[device_id]  # done, or his new ask: its answer is his
                        return
                time.sleep(0.3)
                if self.talking(device_id, timeout=0.5) is not True:
                    continue
                with self._lock:
                    hushed = self._guards[device_id][0]
                if last_turn(device_id) > hushed:
                    continue                    # he asked something new while HA answered
                log.info("voice: %s started talking again after the stop, for nothing newer; stopped", device_id)
                self.stop(device_id)
        except Exception:
            log.warning("voice: the guard on %s failed", device_id, exc_info=True)
            with self._lock:
                self._guards.pop(device_id, None)


def satellite_voice_for(settings) -> SatelliteVoice | None:
    """Armed when HA and a satellite map are configured — the standard optional-seam pattern."""
    from .factory import satellite_map_from

    satellites = satellite_map_from(getattr(settings, "ha_satellite_map", "") or "")
    if not satellites or settings.ha_token is None:
        return None
    return SatelliteVoice(ha_base_url=settings.ha_base_url, ha_token=settings.ha_token.get_secret_value(),
                          satellites=satellites)
