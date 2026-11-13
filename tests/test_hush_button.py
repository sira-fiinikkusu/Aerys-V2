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


# --- Codex 11/13: a hush ends what the turns before it were going to say, and nothing undoes that ---

@pytest.fixture(autouse=True)
def _fresh_generations():
    from aerys_v2 import speaker_redirect as sr
    sr._GEN.clear()
    yield
    sr._GEN.clear()


class Stuck(Speaker):
    def stop(self, device_id):
        self.stops.append(device_id)
        return False


def test_a_stop_the_speaker_did_not_confirm_is_not_called_a_hush(monkeypatch):
    r, opened, asked = press(Stuck(playing=True), monkeypatch)
    assert r.json() == {"did": "nothing", "why": "the speaker didn't stop"}


def redirect_recording(on_play=None):
    from aerys_v2.speaker_redirect import SpeakerRedirect
    posts = []

    class R:
        def __init__(self, body):
            self.body = body

        def raise_for_status(self):
            pass

        def json(self):
            return self.body

    class Client:
        def post(self, url, **kw):
            posts.append(url)
            if url.endswith("/play") and on_play is not None:
                on_play()
            if url.endswith("tts_get_url"):
                return R({"url": "http://tts/x.mp3"})
            return R({"played": True, "stopped": True})

        def get(self, *a, **kw):
            return R({"playing": False, "queued": 0})

    r = SpeakerRedirect(ha_base_url="http://ha", ha_token="t", engine="e", voice="v", language="en",
                        targets={"office": "http://desk"}, speaker_token="s", client=Client(), async_send=False)
    return r, posts


def in_a_turn(fn):
    """Run fn inside a voice turn that began now (its own context, like a request or its follow-up)."""
    import contextvars
    from aerys_v2.speaker_redirect import turn_started

    def go():
        turn_started("office")
        return fn
    ctx = contextvars.copy_context()
    ctx.run(go)
    return ctx


def test_a_late_answer_from_before_the_hush_never_plays():
    from aerys_v2.speaker_redirect import hush
    r, posts = redirect_recording()
    turn = in_a_turn(None)
    hush("office")
    reopened = []
    turn.run(r.say, "office", "the research answer, arriving late", then=lambda: reopened.append(1))
    assert posts == [] and reopened == []


def test_his_next_turn_never_revives_the_old_one():
    from aerys_v2.speaker_redirect import hush
    r, posts = redirect_recording()
    old = in_a_turn(None)
    hush("office")
    new = in_a_turn(None)                                       # he speaks again after the hush
    old.run(r.say, "office", "the old answer")
    assert not any(p.endswith("/play") for p in posts)
    new.run(r.say, "office", "his new answer")
    assert any(p.endswith("/play") for p in posts)


def test_a_clip_being_made_when_the_hush_lands_never_plays():
    from aerys_v2.speaker_redirect import generation, hush
    r, posts = redirect_recording()
    gen = generation("office")
    hush("office")
    reopened = []
    r._say("office", "a clip already being made", lambda: reopened.append(1), gen)
    assert not any(p.endswith("/play") for p in posts) and reopened == []


def test_a_hush_landing_while_play_is_on_its_way_stops_it():
    from aerys_v2.speaker_redirect import hush
    r, posts = None, None
    r, posts = redirect_recording(on_play=lambda: hush("office"))
    reopened = []
    r.say("office", "slips past the check", then=lambda: reopened.append(1))
    assert posts[-1].endswith("/stop") and reopened == []


def test_speech_that_belongs_to_no_turn_still_speaks_after_a_hush():
    # the reach desk (a wake-up's TELL, a ticket answer): not what he hushed, and never logged as spoken
    # without being said
    from aerys_v2.speaker_redirect import hush
    r, posts = redirect_recording()
    hush("office")
    r.say("office", "your 3 PM reminder")
    assert any(p.endswith("/play") for p in posts)


def test_a_voice_stop_also_ends_what_was_on_its_way(monkeypatch):
    from aerys_v2.speaker_redirect import generation
    sp = Speaker(playing=True)
    c = TestClient(build_app(lambda *a: "ok", "sekrit", owner_person_id="owner-uuid",
                             speaker_redirect=sp, settings=settings()))
    before = generation("office")
    c.post("/ask", json={"text": "Aerys, stop!", "thread_id": "voice:beta", "voice": True, "device_id": "office"},
           headers={"Authorization": "Bearer sekrit"})
    assert sp.stops == ["office"] and generation("office") == before + 1


def test_the_button_hush_moves_the_generation_on(monkeypatch):
    from aerys_v2.speaker_redirect import generation
    before = generation("office")
    r, opened, asked = press(Speaker(playing=True), monkeypatch)
    assert r.json() == {"did": "hushed"} and generation("office") == before + 1
