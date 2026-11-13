"""11/13 18:07: she read a 998-character researched answer aloud on the desk speakers and "AERYS STOP" did
nothing. The office mic heard her own voice and his stop as one sentence ("…Air traffic control stop."),
which the whole-phrase check missed; "On it." then queued behind the answer. Now, while she's talking on a
redirected speaker, anything said there cuts her off (with a hush, so what's on its way stays quiet). A
stop at the END of what the mic heard, or her own voice coming back, ends it with no model call; anything
else is answered. Codex 11/13 kept the ordinary questions safe ("why is the office so quiet?", "were those
bright lights moving around?"). What's SPOKEN after an ack is short plain speech; a display still gets it
all. Placeholder ids and text only."""

from __future__ import annotations

import httpx
import pytest
from fastapi.testclient import TestClient

from aerys_v2.transports.http_api import build_app

SAID = ("I couldn't find a story about the FAA grounding lights, Chris. Here's what did come up: a business "
        "jet, Executive Jet Flight 146, reported bright lights moving around and randomly appearing and "
        "disappearing. Air traffic control logged it.")
ECHO_AND_STOP = ("Jet Flight 146 reported bright lights moving around and randomly appearing and disappearing. "
                 "Air traffic control stop.")


@pytest.fixture(autouse=True)
def _fresh():
    from aerys_v2 import speaker_redirect as sr
    sr._HUSH_AT.clear()
    sr._SAID.clear()
    sr.note_said("office", SAID)
    yield
    sr._HUSH_AT.clear()
    sr._SAID.clear()


class Speaker:
    def __init__(self, playing=True):
        self.is_playing, self.stops = playing, []

    def handles(self, device_id):
        return device_id == "office"

    def quick_playing(self, device_id):
        return self.is_playing

    def playing(self, device_id, timeout=1.5):
        return self.is_playing

    def stop(self, device_id, hush_at=None):
        self.stops.append(device_id)
        self.is_playing = False
        return True

    def say(self, device_id, text, then=None):
        pass


def door(speaker):
    asked = []

    def ask(text, identity, thread_id):
        asked.append(text)
        return "ok"

    c = TestClient(build_app(ask, "sekrit", owner_person_id="owner-uuid", speaker_redirect=speaker))

    def say(text, device="office"):
        body = {"text": text, "thread_id": "voice:beta", "voice": True, "device_id": device}
        return c.post("/ask", json=body, headers={"Authorization": "Bearer sekrit"}).json()

    return say, asked


def test_tonights_capture_her_voice_then_his_stop_cuts_her_off():
    from aerys_v2.speaker_redirect import hushed_at
    sp = Speaker()
    say, asked = door(sp)
    assert say(ECHO_AND_STOP)["reply"] == ""
    assert sp.stops == ["office"] and asked == [] and hushed_at("office") > 0


@pytest.mark.parametrize("words", ["Aerys, stop!", "okay stop", "that's enough, Aerys", "shut up", "stop talking"])
def test_a_plain_stop_still_works(words):
    sp = Speaker()
    say, asked = door(sp)
    say(words)
    assert sp.stops == ["office"] and asked == []


def test_her_own_voice_in_the_mic_is_never_answered():
    sp = Speaker()
    say, asked = door(sp)
    say("Jet Flight 146 reported bright lights moving around and randomly appearing")
    assert sp.stops == ["office"] and asked == []


@pytest.mark.parametrize("words", ["Why is the office so quiet?", "Were those bright lights moving around?",
                                   "where's the next bus stop", "is that enough", "what's the weather tomorrow"])
def test_his_own_words_over_her_stop_her_and_are_answered(words):
    # Codex 11/13: an ordinary question ending in "quiet", a follow-up quoting her, a bus stop
    sp = Speaker()
    say, asked = door(sp)
    say(words)
    assert sp.stops == ["office"] and asked == [words]


def test_when_she_is_quiet_nothing_is_cut_and_stop_reaches_her():
    sp = Speaker(playing=False)
    say, asked = door(sp)
    say("stop")
    assert sp.stops == [] and asked == ["stop"]


def test_other_speakers_and_test_doubles_are_left_alone():
    from unittest.mock import Mock
    sp = Speaker()
    say, asked = door(sp)
    say("stop", device="kitchen")
    assert sp.stops == [] and asked == ["stop"]
    say2, asked2 = door(Mock())                      # a Mock's truthy "playing" never counts as talking
    say2("stop", device="office")
    assert asked2 == ["stop"]


def test_the_shared_record_of_what_she_said_is_what_the_echo_check_reads():
    from aerys_v2.speaker_redirect import SpeakerRedirect, _SAID, recent_said

    class Client:
        def post(self, url, **kw):
            body = {"url": "http://tts/x.mp3"} if url.endswith("tts_get_url") else {"played": True}
            return httpx.Response(200, json=body, request=httpx.Request("POST", url))

    _SAID.clear()
    follow_up_side = SpeakerRedirect(ha_base_url="http://ha", ha_token="t", engine="e", voice="v", language="en",
                                     targets={"office": "http://desk"}, speaker_token="s", client=Client(),
                                     async_send=False)
    follow_up_side.say("office", SAID)              # the follow-up router's own redirect instance
    assert "Executive Jet Flight 146" in recent_said("office")


def test_what_she_says_after_an_ack_is_short_plain_speech():
    from aerys_v2.factory import VOICE_ACK_OVERLAY
    from aerys_v2.spoken import SPOKEN_MAX, SPOKEN_MORE, spoken
    long = ("I couldn't find a story about the FAA grounding lights, Chris. Two searches turned up nothing that "
            "matches.\n\nHere's what did come up:\n- **Unusual lights, Nov 1:** Executive Jet Flight 146 reported "
            "bright lights moving around and randomly appearing and disappearing, and air traffic control logged "
            "it, and the FAA said it would look into it.\n- **Drones:** several airports paused departures, "
            "[details](https://example.com/x).\n\nYour message to Megan is waiting on your tap.")
    out = spoken(long)
    assert out.endswith(SPOKEN_MORE) and out.startswith("I couldn't find")
    assert "**" not in out and "- " not in out and "http" not in out
    assert "waiting on your tap" in out                         # a status line is never the part cut
    assert len(out) <= SPOKEN_MAX + 200 + len(SPOKEN_MORE) + 2
    assert spoken("[warmly] Done, Chris.") == "[warmly] Done, Chris."
    assert "SPOKEN aloud" in VOICE_ACK_OVERLAY and "markdown" in VOICE_ACK_OVERLAY


def test_speakers_get_the_short_version_and_a_display_gets_it_all(monkeypatch):
    # Codex 11/13: the first cut also shortened the ink panel's text
    from aerys_v2.config import Settings
    from aerys_v2.factory import followup_router_for
    from aerys_v2.spoken import spoken
    posts = []

    def post(url, **kwargs):
        posts.append((url, kwargs["json"]))
        return httpx.Response(200, request=httpx.Request("POST", url))

    monkeypatch.setattr(httpx, "post", post)
    monkeypatch.setattr(httpx, "get", lambda *a, **k: httpx.Response(200, json={"state": "off"},
                                                                      request=httpx.Request("GET", "http://x")))
    settings = Settings(_env_file=None, anthropic_api_key="x", ha_token="t", ha_base_url="http://ha.example",
                        ha_satellite_map="sat=assist_satellite.example",
                        ha_display_followups="display=esphome.example_show")
    long = "First sentence here. " * 30
    route = followup_router_for(settings)
    route(long, "sat")
    route(long, "display")
    sat = next(b for u, b in posts if u.endswith("/announce"))
    ink = next(b for u, b in posts if u.endswith("/example_show"))
    assert sat["message"] == spoken(long) and len(sat["message"]) < len(long)
    assert ink["message"] == long
