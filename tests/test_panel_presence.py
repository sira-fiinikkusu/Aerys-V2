"""Panel presence watcher — her circadian rhythm, offline.

Proves: sleep only when occupancy is off AND all lights are out (the vacancy
automation's fingerprint); movie mode (manual lights-off, still present) never
sleeps her; wake fires the emote sequence; HA unreadable = no transition
(fail-open); arming rules."""

import httpx

from aerys_v2.panel_presence import (
    SLEEP_DEBOUNCE_TICKS,
    IDLE_STATE,
    SLEEP_DOZE_STATE,
    SLEEP_HOLD_STATE,
    WAKE_EMOTE_STATE,
    PanelPresenceWatcher,
    start_panel_presence,
)

OCC = "binary_sensor.office_occupancy"
L1 = "light.office_light_1"
L2 = "light.office_light_2"


class FakeWorld:
    """Fake HA + fake panel behind one httpx MockTransport."""

    def __init__(self, states=None):
        self.states = states or {}
        self.panel_calls: list[tuple[str, dict]] = []

    def handle(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path.startswith("/api/states/"):
            entity = path.split("/api/states/")[1]
            if entity not in self.states:
                return httpx.Response(404)
            return httpx.Response(200, json={"state": self.states[entity]})
        if path in ("/state", "/display"):
            import json

            self.panel_calls.append((path, json.loads(request.content)))
            return httpx.Response(200, json={})
        return httpx.Response(404)

    def client(self):
        return httpx.Client(transport=httpx.MockTransport(self.handle))


def watcher(world, **kwargs):
    return PanelPresenceWatcher(
        panel_state_url="http://panel:8300/state",
        ha_base_url="http://ha:8123",
        ha_token="t",
        occupancy_entity=OCC,
        light_entities=[L1, L2],
        client=world.client(),
        sleep_fn=lambda _s: None,  # no real waiting in tests
        **kwargs,
    )


def test_vacant_room_with_lights_out_puts_her_to_sleep():
    world = FakeWorld({OCC: "off", L1: "off", L2: "off"})
    w = watcher(world)
    for _ in range(SLEEP_DEBOUNCE_TICKS):
        w.tick()
    assert w.asleep is True
    assert world.panel_calls == [
        ("/state", {"state": SLEEP_DOZE_STATE}),   # she yawns and dozes off first
        ("/state", {"state": SLEEP_HOLD_STATE}),   # then her sleeping is held (Chris 10/26)
        ("/display", {"on": False}),
    ]


def test_movie_mode_keeps_her_awake():
    # lights manually killed but someone is still in the room
    world = FakeWorld({OCC: "on", L1: "off", L2: "off"})
    w = watcher(world)
    w.tick()
    assert w.asleep is False
    assert world.panel_calls == []


def test_vacant_but_lights_still_on_means_grace_period_not_bedtime():
    world = FakeWorld({OCC: "off", L1: "on", L2: "off"})
    w = watcher(world)
    w.tick()
    assert w.asleep is False
    assert world.panel_calls == []


def test_return_wakes_screen_then_emotes_then_settles():
    world = FakeWorld({OCC: "on"})
    w = watcher(world)
    w.asleep = True
    w.tick()
    assert w.asleep is False
    assert world.panel_calls == [
        ("/display", {"on": True}),
        ("/state", {"state": WAKE_EMOTE_STATE}),
        ("/state", {"state": IDLE_STATE}),
    ]


def test_asleep_stays_asleep_while_room_stays_empty():
    world = FakeWorld({OCC: "off", L1: "off", L2: "off"})
    w = watcher(world)
    w.asleep = True
    for _ in range(SLEEP_DEBOUNCE_TICKS):
        w.tick()
    assert w.asleep is True
    assert world.panel_calls == []


def test_unreadable_ha_never_causes_a_transition():
    world = FakeWorld({})  # every entity 404s -> "unknown"
    awake = watcher(world)
    awake.tick()
    assert awake.asleep is False
    dozing = watcher(world)
    dozing.asleep = True
    dozing.tick()
    assert dozing.asleep is True
    assert world.panel_calls == []


def test_dead_panel_never_raises():
    class DeadPanelWorld(FakeWorld):
        def client(self):
            def handler(request: httpx.Request) -> httpx.Response:
                if request.url.path.startswith("/api/states/"):
                    return httpx.Response(200, json={"state": "off"})
                raise httpx.ConnectError("panel dark")

            return httpx.Client(transport=httpx.MockTransport(handler))

    w = watcher(DeadPanelWorld({OCC: "off", L1: "off", L2: "off"}))
    for _ in range(SLEEP_DEBOUNCE_TICKS):
        w.tick()  # no raise = fail-open held; state machine still advanced
    assert w.asleep is True


class HealthWorld(FakeWorld):
    """FakeWorld plus a panel /health + /reboot surface.

    frames advance by 100 per read while `advancing`; a /reboot flips
    `advancing` True (the wedge cleared by a restart)."""

    def __init__(self, states=None, *, player="playing", display=True, advancing=True,
                 brightness=None, state="neutral_idle"):
        super().__init__(states)
        self.player = player
        self.display = display
        self.advancing = advancing
        self.brightness = brightness   # None = pre-os24 firmware (no brightness field)
        self.state = state
        self.frames = 1000
        self.reboots = 0

    def handle(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/health":
            if self.advancing:
                self.frames += 100
            body = {
                "ok": True, "player": self.player, "state": self.state,
                "frames": self.frames, "display": self.display,
            }
            if self.brightness is not None:
                body["brightness"] = self.brightness
            return httpx.Response(200, json=body)
        if path == "/reboot":
            self.reboots += 1
            self.advancing = True
            return httpx.Response(200, text="ok — rebooting\n")
        return super().handle(request)


def test_wake_verify_reboots_a_wedged_panel_and_rewakes_her():
    # display claims on, player claims playing, frames frozen = the sabbatical
    world = HealthWorld({OCC: "on"}, advancing=False)
    w = watcher(world)
    w.asleep = True
    w.tick()
    assert world.reboots == 1
    # after the reboot she gets display-on + idle re-pushed
    assert world.panel_calls[-2:] == [
        ("/display", {"on": True}),
        ("/state", {"state": IDLE_STATE}),
    ]


def test_wake_verify_trusts_wake_when_health_is_unknowable():
    # FakeWorld has no /health (old firmware) — wake must not reboot-loop
    world = FakeWorld({OCC: "on"})
    w = watcher(world)
    w.asleep = True
    w.tick()
    assert w.asleep is False
    assert all(path != "/reboot" for path, _ in world.panel_calls)


def test_daytime_wedge_needs_three_stalled_ticks_then_reboots():
    world = HealthWorld({OCC: "on"}, advancing=False)
    w = watcher(world)
    for _ in range(3):
        w.tick()
    assert world.reboots == 0  # strikes 0,1,2 — still patient
    w.tick()
    assert world.reboots == 1  # third stalled comparison = reboot


def test_stopped_player_never_counts_as_a_wedge():
    # player "stopped" = an OTA in flight or a deliberate stop — a reboot
    # here would corrupt the push. Frames frozen is EXPECTED then.
    world = HealthWorld({OCC: "on"}, player="stopped", advancing=False)
    w = watcher(world)
    for _ in range(6):
        w.tick()
    assert world.reboots == 0


def test_reboot_cooldown_prevents_a_reboot_loop():
    world = HealthWorld({OCC: "on"}, advancing=False)
    w = watcher(world)
    for _ in range(4):
        w.tick()
    assert world.reboots == 1
    world.advancing = False  # it wedges AGAIN right after recovering
    for _ in range(6):
        w.tick()
    assert world.reboots == 1  # within cooldown — no reboot-looping her


def test_arming_requires_all_three_halves():
    class S:
        panel_state_url = None
        ha_token = None
        panel_presence_entity = None
        panel_presence_lights = ""
        ha_base_url = "http://ha:8123"

    assert start_panel_presence(S()) is None
    S.panel_state_url = "http://panel:8300/state"
    assert start_panel_presence(S()) is None  # still no token/entity


# --- escalation: unknowable gets louder, not quieter (task #61) --------------
# 7/31: the panel died in the evening; wake verification ran twice, said
# "health unknowable — trusting the wake" into a log nobody reads, and she sat
# dark ~15 hours until a human pressed reset. These pin the fix: silence past
# a threshold pages the desk line once, recovery says so once, and a dead desk
# line can't take the watcher down.


class ToggleWorld(HealthWorld):
    """HealthWorld whose /health can be taken down and brought back."""

    def __init__(self, states=None, **kw):
        super().__init__(states, **kw)
        self.health_up = True

    def handle(self, request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health" and not self.health_up:
            return httpx.Response(404)
        return super().handle(request)


def notes():
    sent: list[str] = []
    return sent, sent.append


def test_daytime_silence_pages_once_after_the_threshold():
    from aerys_v2.panel_presence import DAYTIME_UNKNOWABLE_STRIKES

    sent, notify = notes()
    world = ToggleWorld({OCC: "on"})
    world.health_up = False
    w = watcher(world, notify_fn=notify)

    for _ in range(DAYTIME_UNKNOWABLE_STRIKES - 1):
        w.tick()
    assert sent == []  # patient through a WiFi blip's worth of silence

    w.tick()  # threshold
    assert len(sent) == 1 and "PANEL ESCALATION" in sent[0]

    for _ in range(5):
        w.tick()  # continued silence must not jackhammer
    assert len(sent) == 1


def test_recovery_sends_the_all_clear_and_rearms():
    from aerys_v2.panel_presence import DAYTIME_UNKNOWABLE_STRIKES

    sent, notify = notes()
    world = ToggleWorld({OCC: "on"})
    world.health_up = False
    w = watcher(world, notify_fn=notify)
    for _ in range(DAYTIME_UNKNOWABLE_STRIKES):
        w.tick()
    assert len(sent) == 1

    world.health_up = True
    w.tick()
    assert len(sent) == 2 and "PANEL RECOVERED" in sent[1]

    # a SECOND outage pages again — recovery re-armed the episode
    world.health_up = False
    for _ in range(DAYTIME_UNKNOWABLE_STRIKES):
        w.tick()
    assert len(sent) == 3 and "PANEL ESCALATION" in sent[2]


def test_two_unknowable_wake_verifies_stop_trusting():
    from aerys_v2.panel_presence import WAKE_UNKNOWABLE_STRIKES

    sent, notify = notes()
    world = FakeWorld({OCC: "on"})  # no /health surface at all
    w = watcher(world, notify_fn=notify)

    for i in range(WAKE_UNKNOWABLE_STRIKES):
        w.asleep = True  # simulate the next morning's wake event
        w.tick()
        if i < WAKE_UNKNOWABLE_STRIKES - 1:
            assert sent == []  # first unknowable wake is still trusted
    assert len(sent) == 1 and "wake verifications" in sent[0]


def test_healthy_panel_never_pages():
    sent, notify = notes()
    world = HealthWorld({OCC: "on"})
    w = watcher(world, notify_fn=notify)
    for _ in range(30):
        w.tick()
    assert sent == []


def test_stopped_player_is_a_healthy_answer_not_silence():
    """Paused/stopped (a nap, an OTA) is the panel ANSWERING — it must end an
    episode-in-progress, not extend it."""
    from aerys_v2.panel_presence import DAYTIME_UNKNOWABLE_STRIKES

    sent, notify = notes()
    world = ToggleWorld({OCC: "on"}, player="stopped")
    world.health_up = False
    w = watcher(world, notify_fn=notify)
    for _ in range(DAYTIME_UNKNOWABLE_STRIKES - 1):
        w.tick()
    world.health_up = True  # answers again — as "stopped"
    w.tick()
    world.health_up = False
    w.tick()
    assert sent == []  # counter was reset by the stopped-but-alive answer


def test_dead_desk_line_does_not_take_the_watcher_down():
    from aerys_v2.panel_presence import DAYTIME_UNKNOWABLE_STRIKES

    def broken_notify(_msg):
        raise RuntimeError("desk line offline")

    world = ToggleWorld({OCC: "on"})
    world.health_up = False
    w = watcher(world, notify_fn=broken_notify)
    for _ in range(DAYTIME_UNKNOWABLE_STRIKES + 2):
        w.tick()  # no raise = fail-open held


# --- hysteresis: the Nanoleaf-test yo-yo (8/02) -------------------------------
# Chris manually killed the office lights while staying at his desk; the
# occupancy sensor drops him during stillness, so she cycled sleep/wake three
# times in six minutes and it read as a reboot loop. Movie mode's assumption
# (occupancy holds while present) gets hysteresis instead of trust.


def test_one_occupancy_flicker_does_not_sleep_her():
    world = FakeWorld({OCC: "off", L1: "off", L2: "off"})
    w = watcher(world)
    w.tick()                     # one flicker
    assert w.asleep is False
    world.states[OCC] = "on"     # owner seen again — debounce restarts
    w.tick()
    world.states[OCC] = "off"
    for _ in range(SLEEP_DEBOUNCE_TICKS - 1):
        w.tick()
    assert w.asleep is False     # still one short of the debounce
    w.tick()
    assert w.asleep is True      # sustained absence finally sleeps her


def test_fresh_wake_holds_five_minutes_against_a_flapping_sensor():
    import time as _time

    world = FakeWorld({OCC: "on", L1: "off", L2: "off"})
    w = watcher(world)
    w.asleep = True
    w.tick()                     # wake
    assert w.asleep is False
    world.states[OCC] = "off"    # sensor immediately drops him again
    for _ in range(SLEEP_DEBOUNCE_TICKS + 2):
        w.tick()
    assert w.asleep is False     # dwell holds her awake

    # ...but once the dwell has genuinely elapsed, sustained absence sleeps her
    w._last_wake_monotonic = _time.monotonic() - 301.0
    for _ in range(SLEEP_DEBOUNCE_TICKS):
        w.tick()
    assert w.asleep is True


def test_unplugged_light_cannot_hold_her_awake():
    # The 2026-08-10 regression: office light 2 physically unplugged reads
    # "unavailable" forever — never "off" — and the all-off test kept her up
    # all night. A light that does not exist cannot be on; sleep proceeds on
    # the lights we can actually see.
    world = FakeWorld({OCC: "off", L1: "off", L2: "unavailable"})
    w = watcher(world)
    for _ in range(SLEEP_DEBOUNCE_TICKS):
        w.tick()
    assert w.asleep is True


def test_all_lights_unknowable_blocks_sleep():
    # Fail-open doctrine intact: HA down = every light reads unknown = no
    # KNOWN-off light exists = no transition. She holds her current state
    # rather than acting on a world she cannot see.
    world = FakeWorld({OCC: "off", L1: "unknown", L2: "unavailable"})
    w = watcher(world)
    for _ in range(SLEEP_DEBOUNCE_TICKS + 2):
        w.tick()
    assert w.asleep is False


def test_unavailable_light_plus_a_lit_one_still_means_awake():
    world = FakeWorld({OCC: "off", L1: "on", L2: "unavailable"})
    w = watcher(world)
    for _ in range(SLEEP_DEBOUNCE_TICKS + 1):
        w.tick()
    assert w.asleep is False


class BootsLateWorld(HealthWorld):
    """A REAL wake is a boot: /health is dead for the first few reads (the P4
    app hasn't started its HTTP server), then comes up healthy and advancing.
    Before 2026-09-03 the verify sampled at ~2s/5s and called every genuine
    boot 'unknowable' — 54 unknowable vs 17 verified in one week."""

    def __init__(self, states=None, *, dead_reads=3, **kw):
        super().__init__(states, **kw)
        self.dead_reads = dead_reads
        self.health_reads = 0

    def handle(self, request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            self.health_reads += 1
            if self.health_reads <= self.dead_reads:
                return httpx.Response(503)  # booting: nothing listening yet
        return super().handle(request)


def test_wake_verify_waits_for_a_booting_panel_then_verifies():
    sent, notify = notes()
    world = BootsLateWorld({OCC: "on"}, dead_reads=3)
    w = watcher(world, notify_fn=notify)
    w.asleep = True
    w.tick()
    assert w.asleep is False
    assert world.reboots == 0                 # a booting panel is not a wedged panel
    assert sent == []                         # and it must not page
    assert w._wake_unknowable == 0            # it VERIFIED, not "trusted blind"
    assert world.health_reads > world.dead_reads  # it actually waited her out


def test_wake_verify_still_unknowable_when_boot_never_answers():
    from aerys_v2.panel_presence import WAKE_HEALTH_POLL_S, WAKE_HEALTH_WAIT_S

    world = BootsLateWorld({OCC: "on"}, dead_reads=10_000)  # never comes up
    w = watcher(world)
    w.asleep = True
    w.tick()
    assert w.asleep is False                  # the wake itself is still trusted
    assert world.reboots == 0                 # /reboot can't help a dead HTTP
    assert w._wake_unknowable == 1            # counted toward the strikes
    # bounded: it gave up after the window, not forever
    assert world.health_reads <= int(WAKE_HEALTH_WAIT_S / WAKE_HEALTH_POLL_S) + 2


# ---- wake gate (owner ask 2026-09-03): wake like the lights do --------------
P1 = "binary_sensor.office_motion_occupancy"
P2 = "binary_sensor.office_motion_2_occupancy"


def test_mmwave_only_occupancy_does_not_wake_her():
    # The AC kicks on at 3 AM, the mmWave half of the group flips ON, the PIRs
    # see nothing. 13-19 phantom wakes a night before this gate existed.
    world = FakeWorld({OCC: "on", P1: "off", P2: "off"})
    w = watcher(world, wake_entities=[P1, P2])
    w.asleep = True
    w.tick()
    assert w.asleep is True
    assert world.panel_calls == []  # no display-on, no emote, nothing


def test_a_pir_sighting_with_occupancy_wakes_her():
    world = FakeWorld({OCC: "on", P1: "on", P2: "off"})
    w = watcher(world, wake_entities=[P1, P2])
    w.asleep = True
    w.tick()
    assert w.asleep is False
    assert world.panel_calls[0] == ("/display", {"on": True})


def test_mmwave_alone_still_holds_her_awake():
    # He's sitting still at the desk: PIRs quiet, mmWave holds the group ON.
    # The group must keep her awake even though it could not have woken her.
    world = FakeWorld({OCC: "on", P1: "off", P2: "off", L1: "off", L2: "off"})
    w = watcher(world, wake_entities=[P1, P2])
    for _ in range(SLEEP_DEBOUNCE_TICKS + 2):
        w.tick()
    assert w.asleep is False


def test_dead_pir_fails_safe_to_asleep():
    # A PIR that HA can't read is never "on" — an unplugged sensor must not
    # become a phantom wake source (same fail-open doctrine as the lights).
    world = FakeWorld({OCC: "on"})  # P1/P2 absent -> 404 -> unknown
    w = watcher(world, wake_entities=[P1, P2])
    w.asleep = True
    w.tick()
    assert w.asleep is True


def test_no_wake_entities_keeps_legacy_wake_on_occupancy():
    world = FakeWorld({OCC: "on"})
    w = watcher(world)  # no wake_entities -> occupancy alone wakes (pre-9/03)
    w.asleep = True
    w.tick()
    assert w.asleep is False


# ---- her worlds (owner ask 2026-09-26) ---------------------------------------------------------
import json as _json

import aerys_v2.panel_presence as _pp
from aerys_v2.panel_presence import pick_world


def test_pick_world_follows_chris_mapping():
    assert pick_world(40.0, False, "rainy") == "rain_window"          # rain beats everything
    assert pick_world(-30.0, False, "pouring") == "rain_window"
    assert pick_world(-20.0, False, "clear-night") == "starfield"     # full night
    assert pick_world(-20.0, True, "clear-night") == "starfield"      # still night before dawn
    assert pick_world(3.0, False, "sunny") == "gulf_dusk"             # sunset
    assert pick_world(-5.0, False, "partlycloudy") == "gulf_dusk"     # dusk, not yet dark
    assert pick_world(3.0, True, "sunny") == "rack_glow"              # morning twilight is not dusk
    assert pick_world(45.0, False, "sunny") == "rack_glow"            # daytime
    assert pick_world(None, None, None) == "rack_glow"                # HA told us nothing useful


class WorldHA(FakeWorld):
    def __init__(self, elevation, rising, weather):
        super().__init__({OCC: "on", L1: "on", L2: "on"})
        self.sun = {"state": "x", "attributes": {"elevation": elevation, "rising": rising}}
        self.weather = weather
        self.worlds: list[str] = []
        self.ha_down = False

    def handle(self, request):
        path = request.url.path
        if self.ha_down and path.startswith("/api/states/"):
            return httpx.Response(500)
        if path == "/api/states/sun.sun":
            return httpx.Response(200, json=self.sun)
        if path == "/api/states/weather.home":
            return httpx.Response(200, json={"state": self.weather})
        if path == "/world":
            self.worlds.append(_json.loads(request.content)["world"])
            return httpx.Response(200, json={})
        return super().handle(request)


def test_world_is_pushed_on_change_and_only_then(monkeypatch):
    ha = WorldHA(-20.0, False, "clear-night")
    clock = [1000.0]
    monkeypatch.setattr(_pp.time, "monotonic", lambda: clock[0])
    w = watcher(ha, world_weather_entity="weather.home")
    w.tick()
    assert ha.worlds == ["starfield"]
    clock[0] += 30; w.tick()                       # inside the 60 s check window: no HA reads
    clock[0] += 60; w.tick()                       # checked, unchanged, not stale: no push
    assert ha.worlds == ["starfield"]
    ha.weather = "rainy"
    clock[0] += 61; w.tick()
    assert ha.worlds == ["starfield", "rain_window"]


def test_world_is_reasserted_after_the_repush_interval(monkeypatch):
    ha = WorldHA(45.0, False, "sunny")
    clock = [1000.0]
    monkeypatch.setattr(_pp.time, "monotonic", lambda: clock[0])
    w = watcher(ha, world_weather_entity="weather.home")
    w.tick()
    clock[0] += _pp.WORLD_REPUSH_S + 1; w.tick()   # a rebooted panel forgot it: say it again
    assert ha.worlds == ["rack_glow", "rack_glow"]


def test_worlds_are_opt_in_and_fail_open(monkeypatch):
    ha = WorldHA(-20.0, False, "clear-night")
    watcher(ha).tick()                             # no weather entity -> never touches /world
    assert ha.worlds == []
    ha.ha_down = True
    watcher(ha, world_weather_entity="weather.home").tick()
    assert ha.worlds == []                         # HA unreadable -> keep whatever world she has


def test_arming_passes_the_weather_entity_only_when_worlds_are_on(monkeypatch):
    from aerys_v2.config import Settings
    seen = {}

    class Spy:
        def __init__(self, **kw):
            seen.update(kw)

        def run_forever(self):
            pass

    monkeypatch.setattr(_pp, "PanelPresenceWatcher", Spy)
    base = dict(_env_file=None, anthropic_api_key="t", panel_state_url="http://p:8300/state",
                ha_token="t", panel_presence_entity=OCC, ha_weather_entity="weather.home")
    _pp.start_panel_presence(Settings(**base))
    assert seen["world_weather_entity"] is None
    _pp.start_panel_presence(Settings(**base, panel_worlds=True))
    assert seen["world_weather_entity"] == "weather.home"


def test_a_panel_reboot_makes_world_and_glyph_resend_on_the_next_tick(monkeypatch):
    ha = WorldHA(-20.0, False, "clear-night")
    clock = [1000.0]
    monkeypatch.setattr(_pp.time, "monotonic", lambda: clock[0])
    w = watcher(ha, world_weather_entity="weather.home")
    w.tick()
    assert ha.worlds == ["starfield"]
    clock[0] += 70; w.tick()                       # unchanged: nothing re-sent
    assert ha.worlds == ["starfield"]
    w._forget_panel_state()                        # what every reboot path calls
    clock[0] += 5; w.tick()                        # well inside the 60 s check window
    assert ha.worlds == ["starfield", "starfield"]


def test_frame_counter_going_backwards_counts_as_a_reboot():
    ha = WorldHA(-20.0, False, "clear-night")
    w = watcher(ha, world_weather_entity="weather.home")
    w._world_pushed, w._glyph_key, w._world_checked_at = "starfield", "k", 1.0
    w._last_frames = 5000
    w._health = lambda: {"player": "playing", "display": True, "frames": 120}
    w._check_daytime_wedge()
    assert w._world_pushed is None and w._glyph_key is None and w._world_checked_at is None


def test_wake_and_settle_go_through_the_face_map():
    # Owner ask 2026-10-24: photoreal faces in place of the sprites. The watcher's own
    # pushes (wake emote, settle) must wear the same remap as the turn's pushes.
    world = FakeWorld({OCC: "on"})
    w = watcher(world, face_map={IDLE_STATE: "wideidle"})
    w.asleep = True
    w.tick()
    assert world.panel_calls == [
        ("/display", {"on": True}),
        ("/state", {"state": WAKE_EMOTE_STATE}),   # not in the map: unchanged
        ("/state", {"state": "wideidle"}),
    ]


# -- dimmed sleep + screen reconcile (owner ask / incident, 2026-10-25) -------

def test_sleep_dims_her_when_a_sleep_brightness_is_set():
    # "dimming the screen with her sleepy face when the room is vacant"
    world = FakeWorld({OCC: "off", L1: "off", L2: "off"})
    w = watcher(world, sleep_brightness=15)
    for _ in range(SLEEP_DEBOUNCE_TICKS):
        w.tick()
    assert w.asleep is True
    assert world.panel_calls == [
        ("/state", {"state": SLEEP_DOZE_STATE}),
        ("/state", {"state": SLEEP_HOLD_STATE}),
        ("/display", {"on": True, "brightness": 15}),
    ]


def test_a_lost_wake_is_resent_on_the_next_tick():
    # 22:30 incident: the wake push timed out behind a face upload; /health still
    # shows the dark screen and the doze face while she's meant to be awake.
    world = HealthWorld({OCC: "on"}, display=False, state=SLEEP_DOZE_STATE)
    w = watcher(world)
    w.tick()
    assert world.panel_calls == [
        ("/display", {"on": True}),
        ("/state", {"state": IDLE_STATE}),
    ]


def test_a_lost_wake_resend_leaves_her_current_face_alone():
    # Dark but showing a mood (a turn pushed it) — bring the screen back, keep the face.
    world = HealthWorld({OCC: "on"}, display=False, state="happy_idle")
    w = watcher(world)
    w.tick()
    assert world.panel_calls == [("/display", {"on": True})]


def test_awake_and_lit_at_full_sends_nothing():
    world = HealthWorld({OCC: "on"}, brightness=100)
    w = watcher(world)
    for _ in range(3):
        w.tick()
    assert world.panel_calls == []


def test_awake_but_still_dimmed_is_brought_back_to_full():
    world = HealthWorld({OCC: "on"}, brightness=15)
    w = watcher(world, sleep_brightness=15)
    w.tick()
    assert ("/display", {"on": True}) in world.panel_calls


def test_a_lost_doze_is_resent_while_she_sleeps():
    # asleep, but the dim never landed: she's lit at full over an empty room
    world = HealthWorld({OCC: "off", L1: "off", L2: "off"}, brightness=100)
    w = watcher(world, sleep_brightness=15)
    w.asleep = True
    w.tick()
    assert world.panel_calls == [
        ("/state", {"state": SLEEP_HOLD_STATE}),   # the held sleep, not a second yawn
        ("/display", {"on": True, "brightness": 15}),
    ]


def test_dimmed_and_asleep_as_asked_sends_nothing():
    world = HealthWorld({OCC: "off", L1: "off", L2: "off"}, brightness=15)
    w = watcher(world, sleep_brightness=15)
    w.asleep = True
    for _ in range(3):
        w.tick()
    assert world.panel_calls == []


def test_wake_verify_does_not_call_a_dark_screen_awake():
    # frames advance with the backlight off, so the verify must read the screen too
    world = HealthWorld({OCC: "on"}, display=False)   # the fake never lights up
    w = watcher(world)
    w.asleep = True
    w.tick()
    assert w.asleep is False
    assert world.panel_calls.count(("/display", {"on": True})) == 2   # the wake + the re-send


def test_old_firmware_without_brightness_still_reconciles_on_off():
    world = HealthWorld({OCC: "off", L1: "off", L2: "off"}, display=True)
    w = watcher(world)            # dark sleep (no brightness configured)
    w.asleep = True
    w.tick()
    assert world.panel_calls == [
        ("/state", {"state": SLEEP_HOLD_STATE}),
        ("/display", {"on": False}),
    ]


def test_a_lost_wake_from_the_held_sleep_settles_her_too():
    # 10/26: the hold is a sleep face as well, so a wake lost behind it gets the idle re-sent
    world = HealthWorld({OCC: "on"}, display=False, state=SLEEP_HOLD_STATE)
    w = watcher(world)
    w.tick()
    assert world.panel_calls == [
        ("/display", {"on": True}),
        ("/state", {"state": IDLE_STATE}),
    ]


# ── 10/28 rare surprises ────────────────────────────────────────────────────
# Chris: "a couple... amusing idles that can play rarely throughout the day (maybe
# like... 10x a day)": knocking on the glass, fists on the glass, leaving for a drink.
# Only while she's awake, the room is occupied and her plain idle is on the panel;
# never twice within SURPRISE_MIN_GAP_S, never more than per_day in 24 h, never the
# same one twice running, and the settle back to idle never clobbers a newer face.

from aerys_v2.panel_presence import SURPRISE_AWAKE_DAY_S, SURPRISE_MIN_GAP_S, parse_surprises

SURPRISES = [("surprise_knock", 7.0), ("surprise_drink", 46.0)]


class Dice:
    """Deterministic rng: random() pops queued rolls (1.0 = never), choice() pops indexes."""

    def __init__(self, rolls, picks=None):
        self.rolls, self.picks = list(rolls), list(picks or [])

    def random(self):
        return self.rolls.pop(0) if self.rolls else 1.0

    def choice(self, seq):
        return seq[self.picks.pop(0)] if self.picks else seq[0]


class SurpriseWorld(HealthWorld):
    """HealthWorld whose /health reports the last state the panel was sent."""

    def handle(self, request):
        if request.url.path == "/state":
            import json
            self.state = json.loads(request.content)["state"]
        return super().handle(request)


def surprise_watcher(world, rolls, *, picks=None, per_day=10, clock=None, on_sleep=None, **kwargs):
    t = {"now": 0.0}
    sleeps = []

    def sleep(s):
        sleeps.append(s)
        if on_sleep:
            on_sleep(s)

    w = PanelPresenceWatcher(
        panel_state_url="http://panel:8300/state", ha_base_url="http://ha:8123", ha_token="t",
        occupancy_entity=OCC, light_entities=[L1, L2], client=world.client(), sleep_fn=sleep,
        surprises=SURPRISES, surprises_per_day=per_day, rng=Dice(rolls, picks),
        clock=clock or (lambda: t["now"]), **kwargs,
    )
    return w, t, sleeps


def states_sent(world):
    return [body["state"] for path, body in world.panel_calls if path == "/state"]


def test_parse_surprises_reads_state_and_seconds():
    assert parse_surprises("surprise_knock:7, surprise_drink:46 ,junk, x:, :3") == SURPRISES
    assert parse_surprises("") == []


def test_a_lucky_roll_plays_a_surprise_then_settles_back_to_idle():
    world = SurpriseWorld({OCC: "on"}, state=IDLE_STATE)
    w, _t, sleeps = surprise_watcher(world, [0.0])
    w.tick()
    assert states_sent(world) == ["surprise_knock", IDLE_STATE]
    assert 7.0 in sleeps                     # held for the clip's own length


def test_an_unlucky_roll_plays_nothing():
    world = SurpriseWorld({OCC: "on"}, state=IDLE_STATE)
    w, _t, _s = surprise_watcher(world, [0.99])
    w.tick()
    assert states_sent(world) == []


def test_the_roll_targets_per_day_over_an_awake_day():
    chance = 10 * 20.0 / SURPRISE_AWAKE_DAY_S   # per_day x poll_s / awake seconds
    for roll, plays in ((chance * 0.9, True), (chance * 1.1, False)):
        world = SurpriseWorld({OCC: "on"}, state=IDLE_STATE)
        w, _t, _s = surprise_watcher(world, [roll])
        w.tick()
        assert bool(states_sent(world)) is plays


def test_only_her_plain_idle_can_be_interrupted():
    for busy in ("neutral_speaking", "happy_idle", "thinking", SLEEP_HOLD_STATE):
        world = SurpriseWorld({OCC: "on"}, state=busy)
        w, _t, _s = surprise_watcher(world, [0.0])
        w.tick()
        assert states_sent(world) == [], busy


def test_never_in_an_empty_room_or_while_asleep():
    world = SurpriseWorld({OCC: "off", L1: "on", L2: "off"}, state=IDLE_STATE)
    w, _t, _s = surprise_watcher(world, [0.0])
    w.tick()
    assert states_sent(world) == []
    world = SurpriseWorld({OCC: "off", L1: "off", L2: "off"}, state=IDLE_STATE)
    w, _t, _s = surprise_watcher(world, [0.0])
    w.asleep = True
    w.tick()
    assert "surprise_knock" not in states_sent(world)


def test_surprises_keep_their_distance():
    world = SurpriseWorld({OCC: "on"}, state=IDLE_STATE)
    w, t, _s = surprise_watcher(world, [0.0, 0.0, 0.0], picks=[0, 0, 0])
    w.tick()
    t["now"] = SURPRISE_MIN_GAP_S - 1
    w.tick()                                 # too soon after the last one
    assert states_sent(world).count(IDLE_STATE) == 1
    t["now"] = SURPRISE_MIN_GAP_S + 1
    w.tick()
    assert len([s for s in states_sent(world) if s.startswith("surprise_")]) == 2


def test_daily_cap():
    world = SurpriseWorld({OCC: "on"}, state=IDLE_STATE)
    w, t, _s = surprise_watcher(world, [0.0] * 5, per_day=2)
    for i in range(4):
        t["now"] = i * (SURPRISE_MIN_GAP_S + 1)
        w.tick()
    assert len([s for s in states_sent(world) if s.startswith("surprise_")]) == 2
    t["now"] = 24 * 3600 + 10                # a day after the first one, the cap reopens
    w.tick()
    assert len([s for s in states_sent(world) if s.startswith("surprise_")]) == 3


def test_never_the_same_surprise_twice_running():
    world = SurpriseWorld({OCC: "on"}, state=IDLE_STATE)
    w, t, _s = surprise_watcher(world, [0.0, 0.0])   # Dice.choice defaults to index 0
    w.tick()
    t["now"] = SURPRISE_MIN_GAP_S + 1
    w.tick()
    assert [s for s in states_sent(world) if s.startswith("surprise_")] == ["surprise_knock", "surprise_drink"]


def test_a_newer_face_during_the_surprise_is_never_clobbered():
    world = SurpriseWorld({OCC: "on"}, state=IDLE_STATE)

    def he_talks_to_her(_s):
        world.state = "neutral_speaking"     # a turn's face push lands mid-surprise

    w, _t, _s = surprise_watcher(world, [0.0], on_sleep=he_talks_to_her)
    w.tick()
    assert states_sent(world) == ["surprise_knock"]


def test_surprises_wear_the_face_map():
    world = SurpriseWorld({OCC: "on"}, state="m_idle")
    w, _t, _s = surprise_watcher(world, [0.0],
                                 face_map={"surprise_knock": "m_knock", IDLE_STATE: "m_idle"})
    w.tick()
    assert states_sent(world) == ["m_knock", "m_idle"]


def test_no_surprises_configured_means_none_ever():
    world = SurpriseWorld({OCC: "on"}, state=IDLE_STATE)
    w = watcher(world)
    for _ in range(50):
        w.tick()
    assert states_sent(world) == []
