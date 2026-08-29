"""storm_watch — hurricane-season eyes that find the owner first (owner ask,
2026-08-17, from the bath the night the Sticky fleet shipped).

The house had radar iframes and spaghetti-model dashboards — all PULL. Nothing
came and found anyone. The owner's actual requirement is beating the crowd:
official watches post ~48 hours out, when the gas lines already exist. The
tiers map to that timeline:

- tier ① OUTLOOK (the beat-the-lines tier): NHC's tropical weather outlook
  flags disturbances with formation odds up to 7 days ahead — back when a
  supply run is quiet and boring. One calm nudge when an area's odds first
  cross a band (50/70/90%), never a repeat inside a band.
- named storms: anything new the NHC starts tracking in the Atlantic basin
  gets one heads-up by name.
- tier ③ WATCH/WARNING (the loud tier): official NWS alerts for the exact
  home point — hurricane / tropical storm / surge / tornado families — land
  the moment they post, radar picture attached.

Delivery reuses the proactive plumbing the kael line already proved: her own
Discord DM to the owner, plus (when configured) a Home Assistant phone
notification with the imagery attached. Both legs are independent and
fail-open — a dead Discord must not cost the phone ping, and vice versa.

Deliberately a dumb poll loop (60s base tick; alerts every 10 min, outlook
every 6 h, both once at startup). All state is in-memory: a redeploy at worst
repeats one nudge, which beats carrying a table for four events a season.
Every network touch is fail-open — a dark feed just means no news this tick.
Runs as ONE daemon thread in the --serve container only.
"""

from __future__ import annotations

import logging
import re
import threading
import time
from typing import Callable

log = logging.getLogger(__name__)

#: NWS requires a descriptive User-Agent and will 403 anonymous defaults.
_NWS_HEADERS = {
    "User-Agent": "aerys-v2 storm-watch (personal home installation)",
    "Accept": "application/geo+json",
}

ALERTS_URL = "https://api.weather.gov/alerts/active?point={point}"
TWO_TEXT_URL = "https://tgftp.nws.noaa.gov/data/raw/ab/abnt20.kmia.two.at.txt"
TWO_FALLBACK_URL = "https://www.nhc.noaa.gov/text/MIATWOAT.shtml"
STORMS_URL = "https://www.nhc.noaa.gov/CurrentStorms.json"
#: Regional NEXRAD frame attached to loud-tier alerts (same station the rain
#: nudge uses). Swap for your region if you are not on the eastern Gulf.
RADAR_IMAGE_URL = "https://radar.weather.gov/ridge/standard/KTBW_0.gif"
OUTLOOK_IMAGE_URL = "https://www.nhc.noaa.gov/xgtwo/two_atl_7d0.png"

#: Default loud-tier event matching: substring families (an NWS event named
#: "Hurricane Force Wind Warning" or "Tropical Storm Watch" should alarm
#: without enumerating every variant) plus two exact single-event names.
DEFAULT_EVENT_FAMILIES = ("Hurricane", "Tropical Storm", "Storm Surge", "Tornado")
DEFAULT_EVENT_EXACT = ("Extreme Wind Warning", "Flash Flood Warning")

#: Outlook nudge bands: first crossing of each band gets one nudge. Reset to
#: quiet only after the basin calms below this floor, so a re-brewing system
#: re-alerts instead of hiding under an old high-water mark.
OUTLOOK_BANDS = (50, 70, 90)
OUTLOOK_RESET_BELOW = 40

ALERT_EVERY_TICKS = 10     # 10 min at the 60s base tick
OUTLOOK_EVERY_TICKS = 360  # 6 h — NHC issues four outlooks a day

#: "Formation chance through 7 days...medium...60 percent." (also matches
#: "near 0 percent"). The 48-hour lines deliberately do NOT match.
_SEVEN_DAY_RE = re.compile(
    r"formation chance through 7 days[.\s\w]*?(\d+)\s*percent", re.IGNORECASE
)
#: Numbered area headers in the outlook text ("1. Eastern Tropical Atlantic:")
_AREA_SPLIT_RE = re.compile(r"^\s*\d+\.\s", re.MULTILINE)


def _fetch_text(url: str) -> str:
    import httpx

    r = httpx.get(url, headers=_NWS_HEADERS, timeout=15.0, follow_redirects=True)
    r.raise_for_status()
    return r.text


def _fetch_json(url: str) -> dict:
    import httpx

    r = httpx.get(url, headers=_NWS_HEADERS, timeout=15.0, follow_redirects=True)
    r.raise_for_status()
    return r.json()


def _strip_pre(html: str) -> str:
    """The shtml fallback wraps the raw product in a <pre> block."""
    m = re.search(r"<pre[^>]*>(.*?)</pre>", html, re.DOTALL | re.IGNORECASE)
    return m.group(1) if m else html


def parse_outlook_areas(text: str) -> list[tuple[int, str]]:
    """(seven_day_percent, first_sentence) per numbered outlook area.

    Areas without a 7-day line (or with unparseable text) are skipped —
    fail-open per area, so one malformed block never hides the others.
    """
    areas: list[tuple[int, str]] = []
    blocks = _AREA_SPLIT_RE.split(text)[1:]  # text before area 1 is boilerplate
    for block in blocks:
        m = _SEVEN_DAY_RE.search(block)
        if not m:
            continue
        pct = int(m.group(1))
        # First sentence of the descriptive text: skip the "Area Name:" header
        # line, then take up to the first period.
        body = block.split(":", 1)[1] if ":" in block.split("\n", 1)[0] else block
        sentence = " ".join(body.split(".")[0].split()).strip()
        areas.append((pct, sentence))
    return areas


def band_of(pct: int) -> int:
    """The highest outlook band this percentage has reached (0 = below all)."""
    return max((b for b in OUTLOOK_BANDS if pct >= b), default=0)


def event_matches(event: str, families: tuple, exact: tuple) -> bool:
    return any(fam in event for fam in families) or event in exact


def compose_legs(*legs: Callable) -> Callable:
    """One deliver(title, message, image_url) fanned across legs, each leg
    independently fail-open — a dead Discord must not cost the phone ping."""

    def deliver(title: str, message: str, image_url: str) -> None:
        for leg in legs:
            try:
                leg(title, message, image_url)
            except Exception:
                log.warning("storm delivery leg failed (continuing)", exc_info=True)

    return deliver


class StormWatcher:
    def __init__(
        self,
        *,
        point: str,
        deliver: Callable[[str, str, str], None],
        event_csv: str = "",
        fetch_json: Callable[[str], dict] = _fetch_json,
        fetch_text: Callable[[str], str] = _fetch_text,
        sleep_fn: Callable[[float], None] = time.sleep,
        tick_s: float = 60.0,
        store: "StormSeenStore | None" = None,
    ) -> None:
        self._point = point
        self._deliver = deliver
        self._fetch_json = fetch_json
        self._fetch_text = fetch_text
        self._sleep = sleep_fn
        self._tick_s = tick_s
        exact_override = tuple(
            e.strip() for e in event_csv.split(",") if e.strip()
        )
        # A csv override is an EXACT allowlist; empty keeps the built-ins.
        self._families = () if exact_override else DEFAULT_EVENT_FAMILIES
        self._exact = exact_override or DEFAULT_EVENT_EXACT
        # Seen-state, durable when a store is wired (migration 009): the
        # original in-memory-only design ("a redeploy at worst repeats one
        # nudge") was written for rare deploys — the 2026-08-27 six-deploy
        # day re-announced Dolly on every boot. The store is fail-open in
        # BOTH directions: load failure = start empty (today's behavior),
        # write failure = logged and ignored (a nudge repeat, never a crash).
        self._store = store
        loaded = store.load() if store is not None else None
        self._seen_alerts: set[str] = set(loaded["alerts"]) if loaded else set()
        self._seen_storms: set[str] = set(loaded["storms"]) if loaded else set()
        self._outlook_high_water = int(loaded["outlook_hw"]) if loaded else 0
        # Countdowns start at 0 so both polls run once immediately at startup.
        self._alert_countdown = 0
        self._outlook_countdown = 0

    # -- tier ③: official alerts for the home point (fail-open) ------------
    def poll_alerts(self) -> None:
        try:
            data = self._fetch_json(ALERTS_URL.format(point=self._point))
        except Exception:
            log.debug("alert poll failed (harmless)", exc_info=True)
            return
        current: set[str] = set()
        for feature in data.get("features", []):
            props = feature.get("properties", {}) or {}
            alert_id = str(props.get("id") or feature.get("id") or "")
            event = str(props.get("event") or "")
            if not alert_id or not event_matches(event, self._families, self._exact):
                continue
            current.add(alert_id)
            if alert_id in self._seen_alerts:
                continue
            headline = props.get("headline") or event
            severity = props.get("severity") or ""
            detail = (props.get("instruction") or props.get("description") or "").strip()
            if len(detail) > 300:
                detail = detail[:297] + "..."
            message = f"{headline}"
            if severity:
                message += f" (severity: {severity})"
            if detail:
                message += f"\n{detail}"
            log.warning("NWS alert active: %s", event)
            self._deliver(f"⚠️ {event}", message, RADAR_IMAGE_URL)
        # Every current id is now either old news or just delivered; dropping
        # expired ids keeps the set from growing across a whole season.
        self._seen_alerts = current
        if self._store is not None:
            self._store.replace_alerts(current)

    # -- tier ①: outlook bands + new named storms (fail-open) --------------
    def poll_outlook(self) -> None:
        text = None
        try:
            text = self._fetch_text(TWO_TEXT_URL)
        except Exception:
            log.debug("primary outlook fetch failed — trying fallback", exc_info=True)
            try:
                text = _strip_pre(self._fetch_text(TWO_FALLBACK_URL))
            except Exception:
                log.debug("outlook fallback failed (harmless)", exc_info=True)
        if text:
            self._check_outlook(text)
        try:
            storms = self._fetch_json(STORMS_URL)
        except Exception:
            log.debug("current-storms fetch failed (harmless)", exc_info=True)
            return
        self._check_new_storms(storms)

    def _check_outlook(self, text: str) -> None:
        areas = parse_outlook_areas(text)
        if not areas:
            return
        pct, sentence = max(areas, key=lambda a: a[0])
        if pct < OUTLOOK_RESET_BELOW:
            # Basin calmed down — re-arm so the next brewing system alerts.
            if self._outlook_high_water != 0 and self._store is not None:
                self._store.set_outlook_high_water(0)
            self._outlook_high_water = 0
            return
        band = band_of(pct)
        if band <= self._outlook_high_water:
            return
        self._outlook_high_water = band
        if self._store is not None:
            self._store.set_outlook_high_water(band)
        log.info("tropical outlook crossed the %d%% band (max %d%%)", band, pct)
        self._deliver(
            "🌀 Tropical outlook",
            f"An area is at {pct}% formation odds over the next 7 days — "
            f"{sentence}. Worth staying ahead of the crowd.",
            OUTLOOK_IMAGE_URL,
        )

    def _check_new_storms(self, data: dict) -> None:
        for storm in data.get("activeStorms", []) or []:
            sid = str(storm.get("id") or "")
            bin_number = str(storm.get("binNumber") or "")
            atlantic = sid.lower().startswith("al") or bin_number.upper().startswith("AT")
            if not sid or not atlantic or sid in self._seen_storms:
                continue
            self._seen_storms.add(sid)
            if self._store is not None:
                self._store.add_storm(sid)
            name = storm.get("name") or sid
            classification = storm.get("classification") or ""
            intensity = storm.get("intensity") or ""
            detail = " ".join(
                str(part) for part in (classification, intensity and f"{intensity} kt")
                if part
            )
            log.info("new named system in the Atlantic: %s", name)
            self._deliver(
                f"🌀 {name}",
                f"The NHC is now tracking {name}"
                + (f" ({detail})" if detail else "")
                + " in the Atlantic basin. Calm heads-up — track and cone attached.",
                OUTLOOK_IMAGE_URL,
            )

    # -- loop --------------------------------------------------------------
    def tick(self) -> None:
        """One base tick — separated from the loop so tests drive it."""
        if self._alert_countdown <= 0:
            self._alert_countdown = ALERT_EVERY_TICKS
            self.poll_alerts()
        if self._outlook_countdown <= 0:
            self._outlook_countdown = OUTLOOK_EVERY_TICKS
            self.poll_outlook()
        self._alert_countdown -= 1
        self._outlook_countdown -= 1

    def run_forever(self) -> None:  # pragma: no cover - thin loop over tick()
        log.info(
            "storm watcher up | alerts every %d min, outlook every %d h",
            ALERT_EVERY_TICKS * self._tick_s // 60,
            OUTLOOK_EVERY_TICKS * self._tick_s // 3600,
        )
        while True:
            try:
                self.tick()
            except Exception:
                log.warning("storm watch tick failed", exc_info=True)
            self._sleep(self._tick_s)


def _deliver_for(settings) -> Callable[[str, str, str], None]:
    """The two real legs, composed. Discord = her own DM to the owner (the
    kael-line plumbing, owner id resolved fresh per event — storms are rare
    enough that a lookup per nudge is nothing, and it survives restarts of
    either side). Phone = an HA notify service with the imagery attached."""

    def discord_leg(title: str, message: str, image_url: str) -> None:
        token = settings.discord_bot_token
        if token is None:
            return
        from aerys_v2.kael_line import _discord_dm_send, _owner_platform_id

        owner_id = _owner_platform_id(settings, "discord")
        if not owner_id:
            return
        _discord_dm_send(
            token.get_secret_value(), owner_id, f"**{title}**\n{message}\n{image_url}"
        )

    def phone_leg(title: str, message: str, image_url: str) -> None:
        if not settings.storm_notify_service or settings.ha_token is None:
            return
        import httpx

        httpx.post(
            f"{settings.ha_base_url.rstrip('/')}/api/services/notify/"
            f"{settings.storm_notify_service}",
            json={
                "title": title,
                "message": message,
                "data": {"image": image_url},
            },
            headers={
                "Authorization": f"Bearer {settings.ha_token.get_secret_value()}"
            },
            timeout=10.0,
        )

    return compose_legs(discord_leg, phone_leg)


class StormSeenStore:
    """Durable seen-state over v2_storm_seen (migration 009). Every method is
    fail-open: storms are rare and a repeated nudge is annoying, but a watcher
    that dies because the NAS blinked is the outage class this codebase keeps
    paying down. A fresh connection per call — the checkpointer-pool lesson:
    long-lived connections are how outages outlive their cause."""

    def __init__(self, database_url: str) -> None:
        self._url = database_url.replace("postgresql+psycopg://", "postgresql://")

    def _exec(self, sql: str, params: tuple = ()) -> list | None:
        import psycopg

        with psycopg.connect(self._url, connect_timeout=5) as conn:
            cur = conn.execute(sql, params)
            rows = cur.fetchall() if cur.description else None
            conn.commit()
            return rows

    def load(self) -> dict | None:
        try:
            rows = self._exec("SELECT kind, key, value FROM v2_storm_seen") or []
            state = {"alerts": set(), "storms": set(), "outlook_hw": 0}
            for kind, key, value in rows:
                if kind == "alert":
                    state["alerts"].add(key)
                elif kind == "storm":
                    state["storms"].add(key)
                elif kind == "outlook_hw":
                    state["outlook_hw"] = int(value or 0)
            return state
        except Exception:
            log.warning("storm seen-state load failed — starting empty", exc_info=True)
            return None

    def add_storm(self, sid: str) -> None:
        try:
            self._exec(
                "INSERT INTO v2_storm_seen (kind, key) VALUES ('storm', %s)"
                " ON CONFLICT DO NOTHING",
                (sid,),
            )
        except Exception:
            log.warning("storm seen-state write failed (harmless)", exc_info=True)

    def replace_alerts(self, current: set[str]) -> None:
        try:
            self._exec("DELETE FROM v2_storm_seen WHERE kind = 'alert'")
            for alert_id in current:
                self._exec(
                    "INSERT INTO v2_storm_seen (kind, key) VALUES ('alert', %s)"
                    " ON CONFLICT DO NOTHING",
                    (alert_id,),
                )
        except Exception:
            log.warning("alert seen-state write failed (harmless)", exc_info=True)

    def set_outlook_high_water(self, band: int) -> None:
        try:
            self._exec(
                "INSERT INTO v2_storm_seen (kind, key, value)"
                " VALUES ('outlook_hw', 'high_water', %s)"
                " ON CONFLICT (kind, key) DO UPDATE SET value = EXCLUDED.value,"
                " seen_at = now()",
                (str(band),),
            )
        except Exception:
            log.warning("outlook seen-state write failed (harmless)", exc_info=True)


def start_storm_watch(settings) -> threading.Thread | None:
    """Arm-and-forget: None unless a home point is configured — the standard
    optional-seam pattern. The point stays in the environment: it is a street
    address in disguise and this repo is public."""
    point = (settings.storm_watch_latlon or "").strip()
    if not point:
        return None
    store = None
    if settings.database_url:
        store = StormSeenStore(settings.database_url)
    watcher = StormWatcher(
        point=point,
        deliver=_deliver_for(settings),
        event_csv=settings.storm_alert_events,
        store=store,
    )
    thread = threading.Thread(
        target=watcher.run_forever, daemon=True, name="storm-watch"
    )
    thread.start()
    return thread
