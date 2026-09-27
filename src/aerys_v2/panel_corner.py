"""The panel's top-right corner (owner ask 2026-09-26): a sky glyph and her status line.

The panel stays dumb (aerys-os ``POST /corner/glyph`` + ``POST /corner/status``); the
brain draws and decides:

- **Glyph** — the real sky over the house, in the current world's light: tonight's moon
  phase (computed, with a faint earthshine so the shape reads), the sun by day, the sun
  on the horizon around dusk, and a cloud / rain / storm over it when HA's weather says
  so. Rendered with numpy distance fields (no image library in the brain image) and sent
  as LVGL TRUE_COLOR_ALPHA: RGB565 little-endian + an 8-bit alpha per pixel.
- **Status** — what she is actually doing, named from the TOOLS she is calling, never
  from the words of the request: the panel sits in a shared room, so "reading your mail"
  may show, the subject line may not. It clears itself after a quiet spell.

Everything network-facing is fail-open, like every other panel seam.
"""

from __future__ import annotations

import datetime as dt
import logging
import math
import threading
from typing import Callable, Iterable

import numpy as np

log = logging.getLogger(__name__)

SIZE = 110

# ---- sky ----------------------------------------------------------------------------------------

#: A reference new moon (2000-01-06 18:14 UTC) and the mean synodic month. Good to well
#: under a day for decades — plenty for a glyph.
_NEW_MOON = dt.datetime(2000, 1, 6, 18, 14, tzinfo=dt.timezone.utc)
_SYNODIC_DAYS = 29.530588853


def moon_phase(now: dt.datetime) -> float:
    """0 = new, 0.25 = first quarter, 0.5 = full, 0.75 = last quarter."""
    days = (now - _NEW_MOON).total_seconds() / 86400.0
    return (days / _SYNODIC_DAYS) % 1.0


def sky_kind(elevation: float | None, rising: bool | None) -> str:
    """moon at night, sun by day, 'dusk' when the sun sits on the horizon either way."""
    if elevation is None:
        return "sun"
    if elevation < -8.0:
        return "moon"
    if elevation <= 6.0:
        return "dusk"
    return "sun"


def weather_overlay(condition: str | None) -> str | None:
    c = (condition or "").lower()
    if c in ("lightning", "lightning-rainy"):
        return "storm"
    if c in ("rainy", "pouring", "snowy-rainy", "hail", "snowy"):
        return "rain"
    if c in ("cloudy", "fog", "partlycloudy"):
        return "cloud"
    return None


#: glow tint per world, so the glyph belongs to the scene behind it
WORLD_GLOW = {
    "starfield": (150, 200, 255),
    "rain_window": (170, 190, 215),
    "gulf_dusk": (255, 190, 130),
    "rack_glow": (255, 205, 150),
}


def glyph_key(kind: str, overlay: str | None, phase: float, world: str | None) -> str:
    """Changes only when the drawing would: the phase in ~1-day steps."""
    return f"{kind}:{overlay or '-'}:{round(phase * 30) % 30 if kind == 'moon' else 0}:{world or '-'}"


# ---- drawing (numpy distance fields, 4x supersampled) --------------------------------------------

_SS = 4


def _grid(n: int = SIZE * _SS):
    y, x = np.mgrid[0:n, 0:n].astype(np.float32)
    return x / _SS, y / _SS   # coordinates in output pixels


def _disk(x, y, cx, cy, r):
    return np.clip(r - np.hypot(x - cx, y - cy) + 0.5, 0.0, 1.0)


def _over(dst_rgb, dst_a, rgb, a):
    """Premultiplied 'over'. dst_rgb is premultiplied."""
    a = a[..., None]
    return dst_rgb * (1 - a) + np.float32(rgb) * a, dst_a * (1 - a[..., 0]) + a[..., 0]


def _glow(x, y, cx, cy, r, sigma, strength):
    d = np.hypot(x - cx, y - cy)
    return np.where(d > r, np.exp(-((d - r) ** 2) / (2 * sigma ** 2)), 1.0) * strength


def render_glyph(kind: str, overlay: str | None, phase: float, world: str | None) -> bytes:
    x, y = _grid()
    n = x.shape[0]
    rgb = np.zeros((n, n, 3), np.float32)
    alpha = np.zeros((n, n), np.float32)
    tint = WORLD_GLOW.get(world or "", (200, 210, 230))
    c = SIZE / 2
    r = SIZE * 0.26
    if kind == "moon":
        rgb, alpha = _over(rgb, alpha, tint, _glow(x, y, c, c, r, 11, 0.30))
        body = _disk(x, y, c, c, r)
        rgb, alpha = _over(rgb, alpha, (235, 228, 210), body * 0.16)          # earthshine
        # terminator: lit where the illuminated half-ellipse covers the disk
        dx, dy = x - c, y - c
        w = np.sqrt(np.clip(r * r - dy * dy, 0, None))
        k = math.cos(2 * math.pi * phase)
        edge = (dx - k * w) if phase < 0.5 else (-k * w - dx)
        lit = np.clip(edge + 0.5, 0, 1) * body
        rgb, alpha = _over(rgb, alpha, (244, 236, 214), lit)
        # faint maria (the familiar near-side shading), only where the moon is lit
        maria = np.maximum.reduce([
            _disk(x, y, c - r * 0.30, c - r * 0.25, r * 0.30),
            _disk(x, y, c + r * 0.22, c - r * 0.35, r * 0.20),
            _disk(x, y, c + r * 0.05, c + r * 0.25, r * 0.26),
            _disk(x, y, c - r * 0.45, c + r * 0.30, r * 0.15),
        ])
        rgb, alpha = _over(rgb, alpha, (205, 198, 182), maria * lit * 0.55)
    else:
        warm = (255, 196, 96) if kind == "sun" else (255, 150, 80)
        cy = c if kind == "sun" else c + r * 0.35
        rgb, alpha = _over(rgb, alpha, warm, _glow(x, y, c, cy, r, 13, 0.38))
        body = _disk(x, y, c, cy, r)
        if kind == "dusk":
            horizon = c + r * 0.55
            body = body * np.clip(horizon - y + 0.5, 0, 1)
            line = np.clip(1.2 - np.abs(y - horizon), 0, 1) * np.clip(1 - np.abs(x - c) / (r * 1.9), 0, 1)
            rgb, alpha = _over(rgb, alpha, (255, 200, 140), line * 0.85)
        rgb, alpha = _over(rgb, alpha, (255, 210, 120) if kind == "sun" else (255, 170, 95), body)
    if overlay is not None:
        # a soft cloud across the lower right of the body
        ox, oy = c + r * 0.25, c + r * 0.55
        cloud = np.maximum.reduce([
            _disk(x, y, ox - r * 0.55, oy + r * 0.1, r * 0.42),
            _disk(x, y, ox, oy - r * 0.2, r * 0.55),
            _disk(x, y, ox + r * 0.55, oy + r * 0.12, r * 0.4),
            np.clip(np.minimum.reduce([x - (ox - r * 0.55), (ox + r * 0.55) - x,
                                       y - oy, (oy + r * 0.52) - y]) + 0.5, 0, 1),
        ])
        shade = (198, 206, 222) if overlay != "storm" else (150, 158, 176)
        rgb, alpha = _over(rgb, alpha, shade, cloud * 0.96)
        base = oy + r * 0.52
        if overlay == "rain":
            drops = np.zeros_like(x)
            for i, fx in enumerate((-0.45, 0.0, 0.45)):
                x0 = ox + fx * r
                y0 = base + 4 + (i % 2) * 5
                t = np.clip((y - y0) / 12.0, 0, 1)
                line_x = x0 - t * 3.0
                drops = np.maximum(drops, np.clip(1.4 - np.abs(x - line_x), 0, 1) * ((y >= y0) & (y <= y0 + 12)))
            rgb, alpha = _over(rgb, alpha, (170, 205, 240), drops)
        elif overlay == "storm":
            # a small lightning bolt under the cloud
            bx, by = ox, base + 2
            pts = [(bx + 3, by), (bx - 5, by + 12), (bx + 1, by + 12), (bx - 4, by + 24), (bx + 8, by + 9), (bx + 2, by + 9), (bx + 7, by)]
            bolt = _polygon(x, y, pts)
            rgb, alpha = _over(rgb, alpha, (255, 214, 90), bolt)
    # downsample the supersampled layer
    rgb = rgb.reshape(SIZE, _SS, SIZE, _SS, 3).mean(axis=(1, 3))
    alpha = alpha.reshape(SIZE, _SS, SIZE, _SS).mean(axis=(1, 3))
    straight = np.where(alpha[..., None] > 1e-4, rgb / np.maximum(alpha[..., None], 1e-4), 0)
    return pack_rgb565a8(straight, alpha)


def _polygon(x, y, pts):
    """Even-odd fill of a polygon (supersampling does the antialiasing)."""
    inside = np.zeros(x.shape, bool)
    j = len(pts) - 1
    for i in range(len(pts)):
        xi, yi = pts[i]
        xj, yj = pts[j]
        cond = ((yi > y) != (yj > y)) & (x < (xj - xi) * (y - yi) / ((yj - yi) or 1e-6) + xi)
        inside ^= cond
        j = i
    return inside.astype(np.float32)


def pack_rgb565a8(rgb: np.ndarray, alpha: np.ndarray) -> bytes:
    """LVGL 8 TRUE_COLOR_ALPHA at 16-bit depth, no byte swap: [lo, hi, a] per pixel."""
    r = np.clip(rgb[..., 0], 0, 255).astype(np.uint16)
    g = np.clip(rgb[..., 1], 0, 255).astype(np.uint16)
    b = np.clip(rgb[..., 2], 0, 255).astype(np.uint16)
    v = ((r >> 3) << 11) | ((g >> 2) << 5) | (b >> 3)
    out = np.empty(rgb.shape[:2] + (3,), np.uint8)
    out[..., 0] = (v & 0xFF).astype(np.uint8)
    out[..., 1] = (v >> 8).astype(np.uint8)
    out[..., 2] = np.clip(alpha * 255 + 0.5, 0, 255).astype(np.uint8)
    return out.tobytes()


# ---- status ---------------------------------------------------------------------------------------

#: tool -> what the room may read. Named from the TOOL, never the request text.
TOOL_PHRASES = {
    "home_control": "running the house",
    "search_entities": "checking the house",
    "control_alarm": "on the alarm",
    "calendar_events": "checking your calendar",
    "search_email": "going through mail",
    "read_email": "reading your mail",
    "draft_email": "drafting an email",
    "send_email": "sending an email",
    "analyze_image": "looking at a picture",
    "read_document": "reading a document",
    "youtube_summary": "watching a video",
    "music": "on the music",
    "timer": "setting a timer",
    "manage_list": "updating a list",
    "get_weather": "checking the weather",
    "search_web": "searching the web",
    "remember": "remembering that",
    "sticky_note": "writing the note",
    "log_gap": "noting it for Kael",
    "message_kael": "telling Kael",
}
_SILENT = {"no_action"}


def status_for(tool_names: Iterable[str]) -> str:
    """One short line for a batch of tool calls; '' when nothing worth showing."""
    phrases: list[str] = []
    for name in tool_names:
        if name in _SILENT:
            continue
        p = TOOL_PHRASES.get(name, "working on it")
        if p not in phrases:
            phrases.append(p)
    if not phrases:
        return ""
    return phrases[0] if len(phrases) == 1 else f"{phrases[0]} + {len(phrases) - 1} more"


class CornerStatus:
    """Fire-and-forget status line. ``report(names)`` shows a line; it clears itself
    ``linger_s`` after the last report (a turn usually ends well before that)."""

    def __init__(self, status_url: str, *, client=None, linger_s: float = 25.0, async_send: bool = True) -> None:
        import httpx

        self._url = status_url
        self._client = client or httpx.Client(timeout=3.0)
        self._linger = linger_s
        self._async = async_send
        self._lock = threading.Lock()
        self._gen = 0
        self._timer: threading.Timer | None = None

    def report(self, tool_names: Iterable[str]) -> None:
        text = status_for(tool_names)
        if not text:
            return
        with self._lock:
            self._gen += 1
            gen = self._gen
            if self._timer is not None:
                self._timer.cancel()
            self._timer = threading.Timer(self._linger, self._clear, args=(gen,))
            self._timer.daemon = True
            self._timer.start()
        self._send(text)

    def _clear(self, gen: int) -> None:
        with self._lock:
            if gen != self._gen:
                return   # a newer report owns the line
        self._send("")

    def _send(self, text: str) -> None:
        if self._async:
            threading.Thread(target=self._post, args=(text,), daemon=True).start()
        else:
            self._post(text)

    def _post(self, text: str) -> None:
        try:
            self._client.post(self._url, json={"text": text})
        except Exception:
            log.debug("corner status push failed (harmless)", exc_info=True)


def corner_base(panel_state_url: str) -> str:
    base = panel_state_url.rstrip("/")
    return base[: -len("/state")] if base.endswith("/state") else base


_ACTIVITY_SINK: Callable[[list[str]], None] | None = None


def set_activity_sink(fn: Callable[[list[str]], None] | None) -> None:
    """The action graph reports tool names here (armed once per process by the factory)."""
    global _ACTIVITY_SINK
    _ACTIVITY_SINK = fn


def report_activity(tool_names: list[str]) -> None:
    fn = _ACTIVITY_SINK
    if fn is None:
        return
    try:
        fn(tool_names)
    except Exception:
        log.debug("activity sink failed (harmless)", exc_info=True)
