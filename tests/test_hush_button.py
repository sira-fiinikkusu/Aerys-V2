"""Chris 11/13 18:18: "my button switch in the office has 4 buttons... could that button be a shut up and
listen toggle? like if she is speaking and i hit the button she shuts up, if she isnt speaking it opens the
mic like i said her wake word". POST /voice/hush-or-listen, from HA on button 4. Placeholder ids only."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from aerys_v2.transports import http_api
from aerys_v2.transports.http_api import build_app


class Speaker:
    def __init__(self, playing):
        self.is_playing, self.stops = playing, []

    def handles(self, device_id):
        return device_id == "office"

    def playing(self, device_id):
        return self.is_playing

    def stop(self, device_id):
        self.stops.append(device_id)
        return True

    def say(self, device_id, text, then=None):
        pass


def settings():
    from aerys_v2.config import Settings
    return Settings(_env_file=None, anthropic_api_key="x", ha_token="t", ha_satellite_map="office=assist_satellite.office")


def press(speaker, monkeypatch, device="office", token="sekrit", fail=False):
    opened = []

    def open_mic(s, satellite):
        if fail:
            raise RuntimeError("HA down")
        opened.append(satellite)

    monkeypatch.setattr(http_api, "_open_mic", open_mic)
    asked = []
    c = TestClient(build_app(lambda *a: asked.append(a) or "ok", "sekrit", owner_person_id="owner-uuid",
                             speaker_redirect=speaker, settings=settings()))
    r = c.post("/voice/hush-or-listen", json={"device_id": device}, headers={"Authorization": f"Bearer {token}"})
    return r, opened, asked


def test_while_she_talks_the_button_hushes_her(monkeypatch):
    sp = Speaker(playing=True)
    r, opened, asked = press(sp, monkeypatch)
    assert r.json() == {"did": "hushed"} and sp.stops == ["office"] and opened == [] and asked == []


def test_when_she_is_quiet_the_button_opens_the_mic(monkeypatch):
    sp = Speaker(playing=False)
    r, opened, asked = press(sp, monkeypatch)
    assert r.json() == {"did": "listening"} and opened == ["assist_satellite.office"] and sp.stops == []
    assert asked == []                                          # no model call either way


def test_a_mic_that_wont_open_says_so(monkeypatch):
    r, opened, asked = press(Speaker(playing=False), monkeypatch, fail=True)
    assert r.json()["did"] == "nothing"


def test_an_unknown_device_does_nothing_and_the_door_needs_the_token(monkeypatch):
    r, opened, asked = press(Speaker(playing=True), monkeypatch, device="garage")
    assert r.json()["did"] == "nothing" and opened == []
    r, opened, asked = press(Speaker(playing=True), monkeypatch, token="wrong")
    assert r.status_code == 401 and opened == []


def test_queued_speech_counts_as_talking():
    from aerys_v2.speaker_redirect import SpeakerRedirect

    class Client:
        def __init__(self, body):
            self.body = body

        def get(self, *a, **kw):
            body = self.body

            class R:
                def raise_for_status(self):
                    pass

                def json(self):
                    return body
            return R()

    def redirect(body):
        return SpeakerRedirect(ha_base_url="http://ha", ha_token="t", engine="e", voice="v", language="en",
                               targets={"office": "http://desk"}, speaker_token="s", client=Client(body))

    assert redirect({"playing": False, "queued": 1}).playing("office")
    assert not redirect({"playing": False, "queued": 0}).playing("office")
