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
        ("/state", {"state": SLEEP_DOZE_STATE}),   # she dozes off first
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

    def __init__(self, states=None, *, player="playing", display=True, advancing=True):
        super().__init__(states)
        self.player = player
        self.display = display
        self.advancing = advancing
        self.frames = 1000
        self.reboots = 0

    def handle(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/health":
            if self.advancing:
                self.frames += 100
            return httpx.Response(200, json={
                "ok": True, "player": self.player, "state": "neutral_idle",
                "frames": self.frames, "display": self.display,
            })
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
