"""The panel corner (owner ask 2026-09-26): her sky glyph and status line, offline."""

import datetime as dt
import threading

import numpy as np

from aerys_v2 import panel_corner as pc


def test_moon_phase_matches_known_nights():
    at = lambda s: pc.moon_phase(dt.datetime.fromisoformat(s))  # noqa: E731
    assert abs(at("2026-09-26T17:00:00+00:00") - 0.5) < 0.02      # the 2026 Harvest Moon
    assert abs(at("2026-09-18T12:00:00+00:00") - 0.25) < 0.04     # first quarter, give or take a day
    ph = at("2026-09-11T03:30:00+00:00")                           # new moon
    assert min(ph, 1 - ph) < 0.04


def test_sky_kind_and_weather_overlay():
    assert pc.sky_kind(-20, False) == "moon"
    assert pc.sky_kind(-20, True) == "moon"
    assert pc.sky_kind(2, False) == "dusk"
    assert pc.sky_kind(2, True) == "dusk"
    assert pc.sky_kind(40, False) == "sun"
    assert pc.sky_kind(None, None) == "sun"
    assert pc.weather_overlay("clear-night") is None
    assert pc.weather_overlay("partlycloudy") == "cloud"
    assert pc.weather_overlay("pouring") == "rain"
    assert pc.weather_overlay("lightning-rainy") == "storm"


def _alpha(buf):
    return np.frombuffer(buf, np.uint8).reshape(pc.SIZE, pc.SIZE, 3)[..., 2]


def test_glyph_is_panel_sized_and_the_moon_follows_its_phase():
    full = pc.render_glyph("moon", None, 0.5, "starfield")
    new = pc.render_glyph("moon", None, 0.0, "starfield")
    quarter = pc.render_glyph("moon", None, 0.25, "starfield")
    assert len(full) == pc.SIZE * pc.SIZE * 3
    c = pc.SIZE // 2
    lit = lambda b: _alpha(b)[c - 20:c + 20, c - 20:c + 20].mean()  # noqa: E731
    assert lit(full) > lit(quarter) > lit(new)       # earthshine keeps the new moon faint, not gone
    q = _alpha(quarter)
    assert q[c, c + 18] > q[c, c - 18]               # waxing: lit on the right
    assert _alpha(full)[0, 0] == 0                   # corners stay transparent over her sky


def test_every_kind_and_overlay_renders():
    for kind in ("sun", "dusk", "moon"):
        for overlay in (None, "cloud", "rain", "storm"):
            buf = pc.render_glyph(kind, overlay, 0.3, "rack_glow")
            assert len(buf) == pc.SIZE * pc.SIZE * 3 and _alpha(buf).max() == 255


def test_pack_is_lvgl_true_color_alpha_rgb565_little_endian():
    rgb = np.array([[[255, 0, 0], [0, 0, 255]]], np.float32)
    out = pc.pack_rgb565a8(rgb, np.array([[1.0, 0.5]], np.float32))
    assert out == bytes([0x00, 0xF8, 255, 0x1F, 0x00, 128])


def test_glyph_key_only_moves_when_the_drawing_would():
    assert pc.glyph_key("moon", None, 0.500, "starfield") == pc.glyph_key("moon", None, 0.505, "starfield")
    assert pc.glyph_key("moon", None, 0.50, "starfield") != pc.glyph_key("moon", None, 0.56, "starfield")
    assert pc.glyph_key("sun", None, 0.1, "rack_glow") == pc.glyph_key("sun", None, 0.9, "rack_glow")
    assert pc.glyph_key("sun", "rain", 0.1, "rack_glow") != pc.glyph_key("sun", None, 0.1, "rack_glow")


def test_status_names_tools_never_request_text():
    assert pc.status_for(["read_email"]) == "reading your mail"
    assert pc.status_for(["home_control", "home_control"]) == "running the house"
    assert pc.status_for(["search_web", "get_weather"]) == "searching the web + 1 more"
    assert pc.status_for(["no_action"]) == ""
    assert pc.status_for(["some_new_tool"]) == "working on it"


class FakeClient:
    def __init__(self):
        self.sent, self.event = [], threading.Event()

    def post(self, url, json=None, **kw):
        self.sent.append(json["text"]); self.event.set()


def test_status_shows_then_clears_itself_and_a_newer_report_wins():
    client = FakeClient()
    s = pc.CornerStatus("http://panel/corner/status", client=client, linger_s=0.15, async_send=False)
    s.report(["read_email"])
    s.report(["music"])                     # newer report: the first timer must not clear it early
    assert client.sent == ["reading your mail", "on the music"]
    deadline = threading.Event(); deadline.wait(0.4)
    assert client.sent[-1] == "" and client.sent.count("") == 1
    s.report(["no_action"])                 # nothing worth showing -> nothing sent
    assert client.sent.count("") == 1 and len(client.sent) == 3


def test_activity_sink_is_fail_open():
    seen = []
    pc.set_activity_sink(seen.append)
    pc.report_activity(["timer"])
    pc.set_activity_sink(lambda _n: 1 / 0)
    pc.report_activity(["timer"])           # a broken sink never reaches the graph
    pc.set_activity_sink(None)
    pc.report_activity(["timer"])
    assert seen == [["timer"]]
