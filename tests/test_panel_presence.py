"""Panel presence watcher — her circadian rhythm, offline.

Proves: sleep only when occupancy is off AND all lights are out (the vacancy
automation's fingerprint); movie mode (manual lights-off, still present) never
sleeps her; wake fires the emote sequence; HA unreadable = no transition
(fail-open); arming rules."""

import httpx

from aerys_v2.panel_presence import (
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

    def client(self):
        def handler(request: httpx.Request) -> httpx.Response:
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

        return httpx.Client(transport=httpx.MockTransport(handler))


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
    w.tick()  # no raise = fail-open held; state machine still advanced
    assert w.asleep is True


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
