"""11/13 18:08: she read a 998-character answer aloud on the desk speakers and "AERYS STOP" did nothing.
The mic heard her own voice and his stop together ("…Air traffic control stop."), which the whole-phrase
check missed, and "On it." queued behind the answer. Now, while she's talking on a redirected speaker
(a clip playing or more queued), anything said there cuts her off first; a stop at the end of what the
mic heard, or a capture that is mostly her own words, ends it with no model call. What's spoken after an
ack is short plain speech. Placeholder ids and text only."""

from __future__ import annotations

from fastapi.testclient import TestClient

from aerys_v2.transports.http_api import build_app

SAID = ("I couldn't find a story about the FAA grounding lights, Chris. Here's what did come up: a business "
        "jet, Executive Jet Flight 146, reported bright lights moving around and randomly appearing and "
        "disappearing. Air traffic control logged it.")


class Speaker:
    def __init__(self, playing=True):
        self.is_playing, self.stops, self.said = playing, [], {"office": SAID}

    def handles(self, device_id):
        return device_id == "office"

    def playing(self, device_id):
        return self.is_playing

    def stop(self, device_id):
        self.stops.append(device_id)
        self.is_playing = False
        return True

    def last_said(self, device_id):
        return self.said.get(device_id, "")

    def say(self, device_id, text, then=None):
        self.said[device_id] = text


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
    sp = Speaker()
    say, asked = door(sp)
    r = say("Jet Flight 146 reported bright lights moving around and randomly appearing and disappearing. "
            "Air traffic control stop.")
    assert sp.stops == ["office"] and asked == [] and r["reply"] == ""


def test_a_plain_stop_still_works():
    for words in ("Aerys, stop!", "okay stop", "that's enough, Aerys"):
        sp = Speaker()
        say, asked = door(sp)
        say(words)
        assert sp.stops == ["office"] and asked == [], words


def test_her_own_voice_in_the_mic_is_never_answered():
    sp = Speaker()
    say, asked = door(sp)
    say("Jet Flight 146 reported bright lights moving around and randomly appearing")
    assert sp.stops == ["office"] and asked == []


def test_his_new_question_over_her_stops_her_and_is_answered():
    sp = Speaker()
    say, asked = door(sp)
    say("what's the weather tomorrow")
    assert sp.stops == ["office"] and asked == ["what's the weather tomorrow"]


def test_when_she_is_quiet_nothing_is_cut_and_stop_reaches_her():
    sp = Speaker(playing=False)
    say, asked = door(sp)
    say("stop")
    assert sp.stops == [] and asked == ["stop"]


def test_other_speakers_are_left_alone():
    sp = Speaker()
    say, asked = door(sp)
    say("stop", device="kitchen")
    assert sp.stops == [] and asked == ["stop"]


def test_queued_speech_counts_as_talking_and_say_remembers_its_words():
    from aerys_v2.speaker_redirect import SpeakerRedirect

    class Resp:
        def __init__(self, body):
            self.body = body

        def raise_for_status(self):
            pass

        def json(self):
            return self.body

    class Client:
        def __init__(self, body):
            self.body = body

        def get(self, *a, **kw):
            return Resp(self.body)

    def redirect(body):
        return SpeakerRedirect(ha_base_url="http://ha", ha_token="t", engine="e", voice="v", language="en",
                               targets={"office": "http://desk"}, speaker_token="s", client=Client(body),
                               async_send=False)

    assert redirect({"playing": False, "queued": 2}).playing("office")
    assert redirect({"playing": True, "queued": 0}).playing("office")
    assert not redirect({"playing": False, "queued": 0}).playing("office")
    r = redirect({"playing": False, "queued": 0})
    r._say = lambda *a: None
    r.say("office", "hello there")
    assert r.last_said("office") == "hello there" and r.last_said("kitchen") == ""


def test_what_she_says_after_an_ack_is_short_plain_speech():
    from aerys_v2.factory import VOICE_ACK_OVERLAY
    from aerys_v2.service import SPOKEN_MORE, _deliver_followup, spoken_followup
    long = ("I couldn't find a story about the FAA grounding lights, Chris. Two searches turned up nothing that "
            "matches.\n\nHere's what did come up:\n- **Unusual lights, Nov 1:** Executive Jet Flight 146 reported "
            "bright lights moving around and randomly appearing and disappearing, and air traffic control logged "
            "it.\n- **Drones:** several airports paused departures, [details](https://example.com/x).")
    out = spoken_followup(long)
    assert len(out) <= 300 + len(SPOKEN_MORE) + 1 and out.endswith(SPOKEN_MORE)
    assert "**" not in out and "- " not in out and "http" not in out and out.startswith("I couldn't find")
    assert spoken_followup("[warmly] Done, Chris.") == "[warmly] Done, Chris."
    routed = []
    _deliver_followup(long, "office", lambda text, dev: routed.append(text), None, None)
    assert routed == [out]
    assert "SPOKEN aloud" in VOICE_ACK_OVERLAY and "markdown" in VOICE_ACK_OVERLAY
