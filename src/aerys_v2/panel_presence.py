"""panel_presence — her circadian rhythm (owner ask, 2026-07-19, from the bath).

The desk panel shouldn't glow at an empty room all night. This watcher tethers
her display to the office presence sensors the owner installed the same day:

- room vacated (occupancy group off AND the office lights are out — i.e. the
  motion-lights automation's cleared branch has fired): she visibly dozes off
  (eyes_closed for a beat) and the screen goes dark.
- owner returns (occupancy on): screen wakes, she does a little surprised
  "oh!" and settles back to idle.
- movie mode falls out for free: lights killed manually while someone is
  still in the room leaves occupancy ON, so she stays awake — the same
  distinction the lights automation's override latch draws.

Trust, but verify (added 2026-07-25 after the "sabbatical" morning): the panel
can report a healthy state server while its video player sits wedged on a dark
canvas. Every wake is verified against /health's decoded-frame counter — if
frames aren't moving, she gets one clean /reboot and a re-wake. The same
counter is watched while she's awake (three stalled ticks = reboot), with the
player-stopped case excluded so an in-flight OTA is never interrupted.

Deliberately a dumb poll loop (20s): presence changes on human timescales,
HA's REST reads are cheap, and a poll survives restarts/outages statelessly.
Every network touch is fail-open — a dark HA or dark panel just means no
transition this tick. Runs as ONE daemon thread in the --serve container
only (three transports must not fight over her eyelids).
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Callable

log = logging.getLogger(__name__)

WAKE_EMOTE_STATE = "surprised"
SLEEP_DOZE_STATE = "eyes_closed"
IDLE_STATE = "neutral_idle"

#: Consecutive daytime ticks (20s apart) with the panel's health endpoint
#: unreachable before escalating — ~3 minutes. The panel rides office WiFi
#: with a known-flappy radio hop upstream, so a short blip must not page
#: anyone; the failure this exists for (the app on the P4 dying while the
#: ESP co-processor keeps answering pings) lasts HOURS. 2026-07-31: dark
#: ~15h before a human noticed.
DAYTIME_UNKNOWABLE_STRIKES = 9
#: Consecutive WAKE verifications that came back unknowable before escalating.
#: Wake events are sparse (occupancy transitions), so two in a row spans a
#: real stretch of the day — exactly the 7/31 shape: 01:59 and 10:55 both
#: logged "health unknowable — trusting the wake" into a log nobody reads,
#: and the trust was misplaced both times.
WAKE_UNKNOWABLE_STRIKES = 2
#: Hysteresis (8/02, the Nanoleaf-test yo-yo). The movie-mode design — manual
#: lights-off with someone present keeps her awake — assumes occupancy holds
#: "on" while the owner is in the room. The office sensor drops him during
#: desk-stillness, so with the lights manually off she cycled sleep/wake three
#: times in six minutes and it read as a reboot loop from the chair. Occupancy
#: must be off for 3 consecutive ticks (~60s) before she MAY sleep, and a wake
#: buys 5 minutes of guaranteed-awake — one sensor flicker no longer closes
#: her eyes, and a flapping sensor cannot yo-yo her. Semantics unchanged:
#: what sleep/wake MEAN is untouched, only how twitchy they are.
SLEEP_DEBOUNCE_TICKS = 3
WAKE_DWELL_S = 300.0


class PanelPresenceWatcher:
    def __init__(
        self,
        *,
        panel_state_url: str,
        ha_base_url: str,
        ha_token: str,
        occupancy_entity: str,
        light_entities: list[str],
        client=None,
        poll_s: float = 20.0,
        emote_s: float = 2.5,
        doze_s: float = 3.0,
        sleep_fn: Callable[[float], None] = time.sleep,
        notify_fn: Callable[[str], None] | None = None,
    ) -> None:
        import httpx

        base = panel_state_url.rstrip("/")
        if base.endswith("/state"):
            base = base[: -len("/state")]
        self._panel_state = f"{base}/state"
        self._panel_display = f"{base}/display"
        self._panel_health = f"{base}/health"
        self._panel_reboot = f"{base}/reboot"
        self._ha_base = ha_base_url.rstrip("/")
        self._headers = {"Authorization": f"Bearer {ha_token}"}
        self._occupancy = occupancy_entity
        self._lights = light_entities
        self._client = client or httpx.Client(timeout=5.0)
        self._poll_s = poll_s
        self._emote_s = emote_s
        self._doze_s = doze_s
        self._sleep = sleep_fn
        self.asleep = False
        # Daytime wedge detection: consecutive ticks where the player claims
        # "playing" but the frame counter hasn't moved. At ~30fps a 20s tick
        # should advance ~600 frames — equal twice running is a dead decode
        # task wearing a "playing" badge (the 7/24 sabbatical signature).
        self._last_frames: int | None = None
        self._stall_strikes = 0
        self._last_reboot_monotonic: float | None = None
        # Escalation (task #61, owner-approved 2026-08-02). "Unknowable" is not
        # "fine": a verification that CANNOT verify must get louder, not
        # quieter. The 7/31 panel death produced two polite log lines and 15
        # dark hours. notify_fn is the shoulder-tap (the kael-desk lane —
        # same endpoint her message_kael tool uses); None = log-only (tests,
        # boxes without the desk line).
        self._notify = notify_fn
        self._daytime_unknowable = 0
        self._wake_unknowable = 0
        self._escalated = False
        # Hysteresis state (the Nanoleaf-test yo-yo, 8/02)
        self._occ_off_ticks = 0
        self._last_wake_monotonic: float | None = None

    # -- HA reads (fail-open: unknown never causes a transition) ----------
    def _entity_state(self, entity_id: str) -> str:
        try:
            r = self._client.get(
                f"{self._ha_base}/api/states/{entity_id}", headers=self._headers
            )
            r.raise_for_status()
            return r.json().get("state", "unknown")
        except Exception:
            log.debug("presence read failed for %s (harmless)", entity_id, exc_info=True)
            return "unknown"

    # -- panel writes (fail-open) -----------------------------------------
    def _push_state(self, state: str) -> None:
        try:
            self._client.post(self._panel_state, json={"state": state})
        except Exception:
            log.debug("panel state push failed (harmless)", exc_info=True)

    def _push_display(self, on: bool) -> None:
        try:
            self._client.post(self._panel_display, json={"on": on})
        except Exception:
            log.debug("panel display push failed (harmless)", exc_info=True)

    # -- panel health (fail-open: None = unknowable, never a transition) ---
    def _health(self) -> dict | None:
        try:
            r = self._client.get(self._panel_health)
            r.raise_for_status()
            h = r.json()
            if "frames" not in h:
                return None  # pre-reboot1 firmware — nothing to verify against
            return h
        except Exception:
            log.debug("panel health read failed (harmless)", exc_info=True)
            return None

    def _frames_advancing(self) -> bool | None:
        """Two health samples a beat apart. True = she's really rendering;
        False = playing-but-stalled (or not playing at all when she should
        be); None = can't know (old firmware / panel unreachable)."""
        h1 = self._health()
        self._sleep(2.0)
        h2 = self._health()
        if h1 is None or h2 is None:
            return None
        if h2.get("player") != "playing":
            return False
        return h2.get("frames", 0) > h1.get("frames", 0)

    def _reboot_panel_and_recover(self) -> bool:
        """POST /reboot, wait for her to come back, restore display+idle.
        Bounded and fail-open — a panel that never returns just logs."""
        now = time.monotonic()
        if (
            self._last_reboot_monotonic is not None
            and now - self._last_reboot_monotonic < 600.0
        ):
            log.warning("panel wedged again within cooldown — not reboot-looping her")
            return False
        self._last_reboot_monotonic = now
        try:
            self._client.post(self._panel_reboot)
        except Exception:
            log.warning("panel /reboot unreachable", exc_info=True)
            return False
        for _ in range(24):  # ~2 min at 5s — boot takes ~30s
            self._sleep(5.0)
            if self._health() is not None:
                self._push_display(True)
                self._push_state(IDLE_STATE)
                return True
        log.error("panel did not return within 2 min of /reboot")
        return False

    # -- escalation (task #61): unknowable gets LOUDER, not quieter ---------
    def _escalate(self, why: str) -> None:
        """One shoulder-tap per outage episode, plus a WARNING that always
        lands in the log. Fail-open like everything else here — a dead desk
        line must not take the watcher down with it."""
        if self._escalated:
            return  # one episode, one page; recovery re-arms it
        self._escalated = True
        msg = (
            f"PANEL ESCALATION: {why} The panel may be dead while looking "
            "alive (ESP answers pings after the P4 app dies — the 7/31 "
            "signature). /reboot needs the HTTP server, so if it is truly "
            "down, someone has to press the physical reset."
        )
        log.warning(msg)
        if self._notify is not None:
            try:
                self._notify(msg)
            except Exception:
                log.warning("panel escalation notify failed", exc_info=True)

    def _note_health_answered(self) -> None:
        """Any successful health read ends the episode: counters reset, and if
        we had paged, say so — a page with no all-clear teaches people to
        drive home for nothing."""
        self._daytime_unknowable = 0
        self._wake_unknowable = 0
        if not self._escalated:
            return
        self._escalated = False
        msg = "PANEL RECOVERED: health endpoint answering again."
        log.info(msg)
        if self._notify is not None:
            try:
                self._notify(msg)
            except Exception:
                log.warning("panel recovery notify failed", exc_info=True)

    def _verify_wake(self) -> None:
        """The 7/24 lesson: never trust the panel's self-report on wake.
        Confirm frames are actually moving; if not, one clean reboot."""
        check = self._frames_advancing()
        if check is None:
            self._wake_unknowable += 1
            if self._wake_unknowable >= WAKE_UNKNOWABLE_STRIKES:
                log.warning(
                    "wake verify: health unknowable %d wakes running — no longer trusting",
                    self._wake_unknowable,
                )
                self._escalate(
                    f"{self._wake_unknowable} consecutive wake verifications "
                    "could not reach panel health — the wakes are being "
                    "trusted blind."
                )
            else:
                log.info("wake verify: panel health unknowable — trusting the wake")
            return
        self._note_health_answered()
        if check:
            log.info("wake verify: frames advancing — she's really awake")
            return
        log.warning("wake verify: display on but frames stalled — rebooting panel")
        if self._reboot_panel_and_recover() and self._frames_advancing():
            log.info("wake verify: recovered after reboot")
        else:
            log.error("wake verify: panel still not rendering after reboot")

    def _check_daytime_wedge(self) -> None:
        """Cheap per-tick stall detector while she's awake. Three strikes
        (playing + display on + frame counter frozen) = reboot. Player
        'stopped'/'paused' resets the count — that's an OTA or a nap, not
        a wedge, and rebooting mid-OTA would corrupt the push."""
        h = self._health()
        if h is None:
            # Unreachable is a DIFFERENT no-op than stopped/paused: a healthy
            # panel that answers "paused" ends an episode; one that answers
            # nothing at all builds toward a page. ~3 min of silence beats a
            # WiFi blip; the target failure lasts hours.
            self._last_frames = None
            self._stall_strikes = 0
            self._daytime_unknowable += 1
            if self._daytime_unknowable >= DAYTIME_UNKNOWABLE_STRIKES:
                self._escalate(
                    "panel health unreachable for "
                    f"{self._daytime_unknowable} consecutive ticks (~"
                    f"{int(self._daytime_unknowable * self._poll_s // 60)} min) "
                    "while she should be awake."
                )
            return
        self._note_health_answered()
        if h.get("player") != "playing" or not h.get("display"):
            self._last_frames = None
            self._stall_strikes = 0
            return
        frames = h.get("frames", 0)
        if self._last_frames is not None and frames == self._last_frames:
            self._stall_strikes += 1
        else:
            self._stall_strikes = 0
        self._last_frames = frames
        if self._stall_strikes >= 3:
            log.warning("panel wedged mid-day (3 stalled ticks) — rebooting")
            self._stall_strikes = 0
            self._last_frames = None
            self._reboot_panel_and_recover()

    # -- transitions -------------------------------------------------------
    def _fall_asleep(self) -> None:
        log.info("office empty + lights out — panel going to sleep")
        self._push_state(SLEEP_DOZE_STATE)
        self._sleep(self._doze_s)  # let her visibly doze off first
        self._push_display(False)
        self.asleep = True

    def _wake_up(self) -> None:
        log.info("owner is back — panel waking up")
        self._push_display(True)
        self._push_state(WAKE_EMOTE_STATE)  # the little "oh!" he asked for
        self._sleep(self._emote_s)
        self._push_state(IDLE_STATE)
        self.asleep = False
        self._last_wake_monotonic = time.monotonic()
        self._occ_off_ticks = 0
        self._verify_wake()

    def tick(self) -> None:
        """One poll cycle — separated from the loop so tests drive it directly."""
        occupied = self._entity_state(self._occupancy)
        if self.asleep:
            if occupied == "on":
                self._wake_up()
            return
        self._check_daytime_wedge()
        if occupied != "off":
            # occupied, or HA unreadable — never sleep on uncertainty, and any
            # sighting of the owner restarts the debounce from zero.
            self._occ_off_ticks = 0
            return
        self._occ_off_ticks += 1
        if self._occ_off_ticks < SLEEP_DEBOUNCE_TICKS:
            return  # one flicker is not an empty room
        if (
            self._last_wake_monotonic is not None
            and time.monotonic() - self._last_wake_monotonic < WAKE_DWELL_S
        ):
            return  # she JUST woke — a flapping sensor doesn't get to yo-yo her
        lights = [self._entity_state(e) for e in self._lights]
        # "Lights out" = no light KNOWN-ON and at least one KNOWN-OFF. The
        # original all-off test kept her awake all night the first time a
        # bulb was physically unplugged (office light 2, 2026-08-10): an
        # unavailable entity is never "off", so a light that does not exist
        # pinned her eyelids open. An unplugged light cannot be on — it must
        # not block sleep. The known-off requirement preserves the fail-open
        # doctrine: HA down = every read "unknown" = no transition, exactly
        # as before.
        if (
            lights
            and not any(s == "on" for s in lights)
            and any(s == "off" for s in lights)
        ):
            # occupancy off AND lights out = the vacancy automation has fired;
            # manual lights-off with someone present keeps occupancy on.
            self._fall_asleep()

    def run_forever(self) -> None:  # pragma: no cover - thin loop over tick()
        log.info(
            "panel presence watcher up | occupancy=%s lights=%s poll=%.0fs",
            self._occupancy, ",".join(self._lights), self._poll_s,
        )
        while True:
            try:
                self.tick()
            except Exception:
                log.warning("panel presence tick failed", exc_info=True)
            self._sleep(self._poll_s)


def start_panel_presence(settings) -> threading.Thread | None:
    """Arm-and-forget: None unless the panel, HA, and an occupancy entity are
    all configured — the standard optional-seam pattern."""
    if not settings.panel_state_url or settings.ha_token is None:
        return None
    if not settings.panel_presence_entity:
        return None
    lights = [e.strip() for e in settings.panel_presence_lights.split(",") if e.strip()]
    # Escalation line: the same desk-channel lane message_kael uses (Kael's
    # session treats the aerys lane as observations, and this is exactly one).
    # Optional like everything else — no desk line configured means the
    # escalation still lands in the log, just not in a session.
    notify_fn = None
    if settings.kael_desk_url and settings.kael_desk_token is not None:
        import httpx

        _url = settings.kael_desk_url
        _tok = settings.kael_desk_token.get_secret_value()

        def notify_fn(message: str) -> None:
            httpx.post(
                _url,
                json={"message": f"[panel_presence] {message}"},
                headers={"Authorization": f"Bearer {_tok}"},
                timeout=5.0,
            )

    watcher = PanelPresenceWatcher(
        panel_state_url=settings.panel_state_url,
        ha_base_url=settings.ha_base_url,
        ha_token=settings.ha_token.get_secret_value(),
        occupancy_entity=settings.panel_presence_entity,
        light_entities=lights,
        notify_fn=notify_fn,
    )
    thread = threading.Thread(target=watcher.run_forever, daemon=True, name="panel-presence")
    thread.start()
    return thread
