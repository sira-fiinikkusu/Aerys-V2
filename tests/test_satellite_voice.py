"""11/14 (the Pebble): the office satellite speaks through its own speaker again. The button stops her
there, a stop the satellite heard itself holds what's still on its way, and the desk-redirected path is
untouched. Placeholder ids only."""

from __future__ import annotations

import contextvars

import httpx
import pytest
from fastapi.testclient import TestClient

from aerys_v2 import speaker_redirect as sr
from aerys_v2.satellite_voice import SatelliteVoice, satellite_voice_for, stop_action
from aerys_v2.transports import http_api
from aerys_v2.transports.http_api import build_app

ENTITY = "assist_satellite.office_satellite_assist_satellite"


@pytest.fixture(autouse=True)
def fresh_hushes(monkeypatch):
    monkeypatch.setattr(sr, "_HUSH_AT", {})
    monkeypatch.setattr(sr, "_TURN_AT", {})
    t = {"now": 1_000.0}

    def now():
        t["now"] += 1.0
        return t["now"]
    monkeypatch.setattr(sr.time, "time", now)


class HA:
    """HA's side: the satellite's state, and its stop action (which quiets it unless told otherwise).
    after_stop: what reads return after the first stop, in order (a 500 for "fail"), then the state."""

    def __init__(self, state="responding", stops=True, fail=False, after_stop=()):
        self.state, self.stops, self.fail, self.calls = state, stops, fail, []
        self.after_stop, self.stopped = list(after_stop), 0

    def client(self):
        ha = self

        def handle(request):
            ha.calls.append((request.method, request.url.path))
            if ha.fail:
                return httpx.Response(500)
            if request.method == "POST":
                ha.stopped += 1
                if ha.stopped > 1:
                    ha.after_stop = []
                if ha.stops:
                    ha.state = "idle"
                return httpx.Response(200, json=[])
            if ha.stopped and ha.after_stop:
                state = ha.after_stop.pop(0)
                return httpx.Response(500) if state == "fail" else httpx.Response(200, json={"state": state})
            return httpx.Response(200, json={"state": ha.state})
        return httpx.Client(transport=httpx.MockTransport(handle))


def satellite(ha, **kw):
    return SatelliteVoice(ha_base_url="http://ha", ha_token="t", satellites={"office": ENTITY},
                          client=ha.client(), settle_s=kw.get("settle_s", 0.3), guard_s=kw.get("guard_s", 0.35),
                          async_guard=False)


def test_the_stop_action_is_named_for_the_node():
    assert stop_action(ENTITY) == "office_satellite_stop_va"
    assert stop_action("media_player.office") is None and stop_action("assist_satellite._assist_satellite") is None


def test_responding_is_talking_quiet_states_are_quiet_and_the_rest_is_unknown():
    assert satellite(HA("responding")).talking("office") is True
    assert satellite(HA("idle")).talking("office") is False
    assert satellite(HA("listening")).talking("office") is False
    assert satellite(HA("unavailable")).talking("office") is None      # never guessed as quiet (Codex)
    assert satellite(HA(fail=True)).talking("office") is None
    assert satellite(HA("responding")).talking("garage") is False


def test_stop_calls_the_satellites_own_action_and_waits_for_quiet():
    ha = HA("responding")
    assert satellite(ha).stop("office") is True
    assert ("POST", "/api/services/esphome/office_satellite_stop_va") in ha.calls


def test_a_stop_that_leaves_her_talking_or_fails_is_not_a_stop():
    assert satellite(HA("responding", stops=False)).stop("office") is False
    assert satellite(HA(fail=True)).stop("office") is False
    # the stop went through but HA can't confirm she's quiet: not called a stop (Codex 11/14)
    assert satellite(HA("responding", after_stop=["fail"] * 20)).stop("office") is False
    odd = SatelliteVoice(ha_base_url="http://ha", ha_token="t", satellites={"office": "media_player.x"},
                         client=HA().client())
    assert odd.stop("office") is False


def test_armed_only_with_ha_and_a_satellite_map():
    from aerys_v2.config import Settings
    assert satellite_voice_for(Settings(_env_file=None, anthropic_api_key="x", ha_token="t",
                                        ha_satellite_map=f"office={ENTITY}")) is not None
    assert satellite_voice_for(Settings(_env_file=None, anthropic_api_key="x", ha_token="t")) is None
    assert satellite_voice_for(Settings(_env_file=None, anthropic_api_key="x",
                                        ha_satellite_map=f"office={ENTITY}")) is None


# --- the office button and the satellite's own stop, at the door ---


class Desk:
    """The desk redirect, for the precedence test."""

    def __init__(self, playing=False):
        self.is_playing, self.stops = playing, []

    def handles(self, device_id):
        return device_id == "office"

    def playing(self, device_id):
        return self.is_playing

    def stop(self, device_id, hush_at=None):
        self.stops.append(device_id)
        return True


def settings():
    from aerys_v2.config import Settings
    return Settings(_env_file=None, anthropic_api_key="x", ha_token="t", ha_satellite_map=f"office={ENTITY}")


def door(monkeypatch, sat, desk=None):
    opened = []
    monkeypatch.setattr(http_api, "_open_mic", lambda s, entity: opened.append(entity))
    c = TestClient(build_app(lambda *a: "ok", "sekrit", owner_person_id="owner-uuid", speaker_redirect=desk,
                             satellite_voice=sat, settings=settings()))
    return c, opened


AUTH = {"Authorization": "Bearer sekrit"}


def test_while_she_talks_on_the_satellite_the_button_stops_her_there(monkeypatch):
    ha = HA("responding")
    c, opened = door(monkeypatch, satellite(ha))
    r = c.post("/voice/hush-or-listen", json={"device_id": "office"}, headers=AUTH)
    assert r.json() == {"did": "hushed"} and opened == [] and ha.state == "idle"
    assert sr.hushed_at("office") > 0                            # what's on its way is held too


def test_a_satellite_that_wont_stop_is_not_called_hushed(monkeypatch):
    c, opened = door(monkeypatch, satellite(HA("responding", stops=False)))
    r = c.post("/voice/hush-or-listen", json={"device_id": "office"}, headers=AUTH)
    assert r.json()["did"] == "nothing" and opened == []


def test_when_ha_cant_say_the_button_neither_stops_nor_opens_the_mic(monkeypatch):
    for ha in (HA(fail=True), HA("unavailable")):
        c, opened = door(monkeypatch, satellite(ha))
        r = c.post("/voice/hush-or-listen", json={"device_id": "office"}, headers=AUTH)
        assert r.json()["did"] == "nothing" and opened == [] and sr.hushed_at("office") == 0


def test_a_follow_up_already_on_its_way_when_he_pressed_is_stopped_again(monkeypatch):
    # its hush check passed a moment before the press; HA starts it right after the stop (Codex 11/14)
    ha = HA("responding", after_stop=["idle", "responding"])
    c, _ = door(monkeypatch, satellite(ha))
    r = c.post("/voice/hush-or-listen", json={"device_id": "office"}, headers=AUTH)
    assert r.json() == {"did": "hushed"} and ha.stopped == 2


def test_the_second_stop_spares_the_answer_to_his_new_ask():
    ha = HA("responding")
    sv = satellite(ha)
    hushed = sr.hush("office")
    sr.turn_started("office")                                    # he asked something new after the stop
    sv.guard("office", hushed)
    assert ha.stopped == 0


def test_when_the_satellite_is_quiet_the_button_still_opens_the_mic(monkeypatch):
    ha = HA("idle")
    c, opened = door(monkeypatch, satellite(ha))
    r = c.post("/voice/hush-or-listen", json={"device_id": "office"}, headers=AUTH)
    assert r.json() == {"did": "listening"} and opened == [ENTITY]
    assert not any(m == "POST" for m, _ in ha.calls) and sr.hushed_at("office") == 0


def test_while_the_office_speaks_through_the_desk_the_satellite_is_never_asked(monkeypatch):
    # until the env flip, the satellite "responds" silently beside the desk speaker: the desk decides
    ha = HA("responding")
    c, opened = door(monkeypatch, satellite(ha), desk=Desk(playing=False))
    r = c.post("/voice/hush-or-listen", json={"device_id": "office"}, headers=AUTH)
    assert r.json() == {"did": "listening"} and ha.calls == []


@pytest.mark.parametrize("word,state,changed_s,did", [
    ("stop", "idle", 40.0, "held"),               # only armed while she talks
    ("hey Aerys", "responding", 2.0, "held"),     # her name over her voice: the satellite cut her off
    ("hey Aerys", "idle", 0.4, "held"),           # ...and its own stop beat HA's read (Codex 11/14)
    ("hey Aerys", "idle", 40.0, "nothing"),       # she was quiet: a new ask, its own turn
    ("hey Aerys", "listening", 0.4, "nothing"),
    ("hey Aerys", "", None, "nothing"),
])
def test_a_stop_the_satellite_heard_itself_holds_what_is_on_its_way(monkeypatch, word, state, changed_s, did):
    c, _ = door(monkeypatch, satellite(HA("idle")))
    r = c.post("/voice/stopped", json={"device_id": "office", "wake_word": word, "state": state,
                                       "changed_s": changed_s}, headers=AUTH)
    assert r.json()["did"] == did
    assert (sr.hushed_at("office") > 0) is (did == "held")


def test_the_stopped_door_only_holds_her_own_satellites(monkeypatch):
    # any authed caller could otherwise grow the hush table without end (Codex 11/14)
    c, _ = door(monkeypatch, satellite(HA("idle")))
    for n in range(5):
        c.post("/voice/stopped", json={"device_id": f"made-up-{n}", "wake_word": "stop"}, headers=AUTH)
    c, _ = door(monkeypatch, None)
    c.post("/voice/stopped", json={"device_id": "office", "wake_word": "stop"}, headers=AUTH)
    assert sr._HUSH_AT == {}


def test_the_stopped_door_leaves_a_desk_redirected_satellite_alone_and_needs_the_token(monkeypatch):
    c, _ = door(monkeypatch, satellite(HA("idle")), desk=Desk())
    r = c.post("/voice/stopped", json={"device_id": "office", "wake_word": "stop"}, headers=AUTH)
    assert r.json()["did"] == "nothing" and sr.hushed_at("office") == 0
    c, _ = door(monkeypatch, satellite(HA("idle")))
    r = c.post("/voice/stopped", json={"device_id": "office", "wake_word": "stop"},
               headers={"Authorization": "Bearer wrong"})
    assert r.status_code == 401 and sr.hushed_at("office") == 0


# --- follow-ups on the satellite path ---


def router(monkeypatch):
    from aerys_v2.config import Settings
    from aerys_v2.factory import followup_router_for

    monkeypatch.setattr(sr, "speaker_redirect_for", lambda _s: None)       # the office is off the desk
    posts = []
    monkeypatch.setattr(httpx, "post", lambda *a, **k: posts.append((a, k))
                        or httpx.Response(200, request=httpx.Request("POST", "http://x")))
    route = followup_router_for(Settings(_env_file=None, anthropic_api_key="t", ha_token="h",
                                         ha_satellite_map=f"office={ENTITY}"))
    return route, posts


def turn():
    ctx = contextvars.copy_context()
    ctx.run(sr.turn_started, "office")
    return ctx


def test_a_follow_up_from_a_turn_he_stopped_is_never_announced(monkeypatch):
    route, posts = router(monkeypatch)
    old = turn()
    sr.hush("office")
    old.run(route, "the flight search finished", "office")
    assert posts == []


def test_the_real_follow_up_thread_carries_the_turn_it_came_from(monkeypatch):
    # service.py hands a turn's follow-up to a thread through _in_ctx: the hush must still see that turn
    import threading

    from aerys_v2.service import _in_ctx

    route, posts = router(monkeypatch)

    def follow_up_of_an_old_turn():
        sr.turn_started("office")
        sr.hush("office")
        t = threading.Thread(target=_in_ctx(route, "the flight search finished", "office"))
        t.start()
        t.join()
    contextvars.copy_context().run(follow_up_of_an_old_turn)
    assert posts == []


def test_his_next_turn_and_speech_from_no_turn_still_announce(monkeypatch):
    route, posts = router(monkeypatch)
    sr.hush("office")
    turn().run(route, "the answer to his new ask", "office")
    route("your 3 PM reminder", "office")                    # a wake-up's tell belongs to no turn
    assert [k["json"]["message"] for _, k in posts] == ["the answer to his new ask", "your 3 PM reminder"]
