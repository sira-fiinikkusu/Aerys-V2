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

    def stop(self, device_id, hush_at=None):
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


# --- Codex 11/13: a hush ends what the turns before it were going to say; the speaker is the judge ----

@pytest.fixture(autouse=True)
def _fresh_hushes():
    from aerys_v2 import speaker_redirect as sr
    sr._HUSH_AT.clear()
    yield
    sr._HUSH_AT.clear()


class Stuck(Speaker):
    def stop(self, device_id, hush_at=None):
        self.stops.append(device_id)
        return False


def test_a_stop_the_speaker_did_not_confirm_is_not_called_a_hush(monkeypatch):
    r, opened, asked = press(Stuck(playing=True), monkeypatch)
    assert r.json() == {"did": "nothing", "why": "the speaker didn't stop"}


class Desk:
    """The desk speaker's own rule (aerys-speaker.py): a clip from before its last hush is refused."""

    def __init__(self, on_play=None):
        self.floor, self.played, self.posts, self.on_play = 0.0, [], [], on_play

    def client(self):
        desk = self

        class R:
            def __init__(self, body):
                self.body = body

            def raise_for_status(self):
                pass

            def json(self):
                return self.body

        class Client:
            def post(self, url, json=None, **kw):
                desk.posts.append(url)
                if url.endswith("tts_get_url"):
                    return R({"url": "http://tts/x.mp3"})
                if url.endswith("/stop"):
                    desk.floor = max(desk.floor, (json or {}).get("hush_at") or 0.0)
                    return R({"stopped": True})
                if desk.on_play is not None:
                    desk.on_play()
                since = (json or {}).get("since")
                if since is not None and since < desk.floor:
                    return R({"played": False, "stale": True})
                desk.played.append(since)
                return R({"played": True})

            def get(self, *a, **kw):
                return R({"playing": False, "queued": 0})
        return Client()


def redirect_on(desk):
    from aerys_v2.speaker_redirect import SpeakerRedirect
    return SpeakerRedirect(ha_base_url="http://ha", ha_token="t", engine="e", voice="v", language="en",
                           targets={"office": "http://desk"}, speaker_token="s", client=desk.client(),
                           async_send=False)


def in_a_turn():
    """A voice turn that began now, in its own context (like a request, or its background follow-up)."""
    import contextvars
    from aerys_v2.speaker_redirect import turn_started
    ctx = contextvars.copy_context()
    ctx.run(turn_started, "office")
    return ctx


def tick(monkeypatch):
    """Each call to the clock is a little later (the hush must land after the turn began)."""
    from aerys_v2 import speaker_redirect as sr
    t = {"now": 1_000.0}

    def now():
        t["now"] += 1.0
        return t["now"]
    monkeypatch.setattr(sr.time, "time", now)


def test_a_late_answer_from_before_the_hush_never_plays(monkeypatch):
    from aerys_v2.speaker_redirect import hush
    tick(monkeypatch)
    desk = Desk()
    r = redirect_on(desk)
    turn = in_a_turn()
    hush("office")
    reopened = []
    turn.run(r.say, "office", "the research answer, arriving late", then=lambda: reopened.append(1))
    assert desk.played == [] and reopened == [] and desk.posts == []      # never even made


def test_his_next_turn_never_revives_the_old_one(monkeypatch):
    from aerys_v2.speaker_redirect import hush
    tick(monkeypatch)
    desk = Desk()
    r = redirect_on(desk)
    old = in_a_turn()
    hush("office")
    new = in_a_turn()
    old.run(r.say, "office", "the old answer")
    new.run(r.say, "office", "his new answer")
    assert len(desk.played) == 1


def test_the_speaker_refuses_a_clip_that_slipped_past_the_check(monkeypatch):
    # Codex: the clip passed the brain's check, then the hush + /stop landed, then /play arrived.
    from aerys_v2.speaker_redirect import hush
    tick(monkeypatch)
    r = None
    desk = Desk(on_play=lambda: r.stop("office", hush_at=hush("office")))
    r = redirect_on(desk)
    reopened = []
    in_a_turn().run(r.say, "office", "slips past the check", then=lambda: reopened.append(1))
    assert desk.played == [] and reopened == []


def test_an_old_clip_never_stops_a_newer_one(monkeypatch):
    from aerys_v2.speaker_redirect import hush
    tick(monkeypatch)
    desk = Desk()
    r = redirect_on(desk)
    old = in_a_turn()
    r.stop("office", hush_at=hush("office"))
    new = in_a_turn()
    new.run(r.say, "office", "his new answer")
    old.run(r._say, "office", "the old clip, finishing late", None, None)   # no turn time: plays as now
    assert not any(p.endswith("/stop") for p in desk.posts[1:])


def test_a_hush_on_any_speaker_holds_that_turns_speech_there(monkeypatch):
    from aerys_v2.speaker_redirect import SpeakerRedirect, hush
    tick(monkeypatch)
    desk = Desk()
    r = SpeakerRedirect(ha_base_url="http://ha", ha_token="t", engine="e", voice="v", language="en",
                        targets={"office": "http://desk", "kitchen": "http://desk"}, speaker_token="s",
                        client=desk.client(), async_send=False)
    turn = in_a_turn()                                         # the turn began in the office
    hush("kitchen")
    turn.run(r.say, "kitchen", "its follow-up, routed to the kitchen")
    assert desk.played == []


def test_speech_that_belongs_to_no_turn_still_speaks_after_a_hush(monkeypatch):
    # the reach desk (a wake-up's TELL, a ticket answer): not what he hushed, never logged as said unsaid
    from aerys_v2.speaker_redirect import hush
    tick(monkeypatch)
    desk = Desk()
    r = redirect_on(desk)
    r.stop("office", hush_at=hush("office"))
    r.say("office", "your 3 PM reminder")
    assert len(desk.played) == 1


def test_a_voice_stop_and_the_button_both_hush(monkeypatch):
    from aerys_v2.speaker_redirect import hushed_at

    class Sp(Speaker):
        def stop(self, device_id, hush_at=None):
            self.stops.append((device_id, hush_at))
            return True
    sp = Sp(playing=True)
    c = TestClient(build_app(lambda *a: "ok", "sekrit", owner_person_id="owner-uuid",
                             speaker_redirect=sp, settings=settings()))
    c.post("/ask", json={"text": "Aerys, stop!", "thread_id": "voice:beta", "voice": True, "device_id": "office"},
           headers={"Authorization": "Bearer sekrit"})
    assert sp.stops and sp.stops[-1][1] == hushed_at("office") > 0
    sp2 = Sp(playing=True)
    r, opened, asked = press(sp2, monkeypatch)
    assert r.json() == {"did": "hushed"} and sp2.stops[-1][1] == hushed_at("office")
