"""Speaker redirect (owner ask 2026-09-26): the office satellite's speaker died, so her
office replies are rendered by HA in her pipeline voice and played on Leviathan."""

import httpx

from aerys_v2.config import Settings
from aerys_v2.speaker_redirect import SpeakerRedirect, parse_targets, speaker_redirect_for

OFFICE = "185bd720dd074d798a6094ad4f22e525"


class Recorder:
    def __init__(self, tts_status=200):
        self.calls, self.tts_status = [], tts_status

    def handle(self, request: httpx.Request) -> httpx.Response:
        import json

        body = json.loads(request.content or b"{}")
        self.calls.append((str(request.url), request.headers.get("authorization"), body))
        if request.url.path == "/api/tts_get_url":
            if self.tts_status != 200:
                return httpx.Response(self.tts_status)
            return httpx.Response(200, json={"url": "http://192.168.1.155:8123/api/tts_proxy/x.mp3"})
        if request.url.path == "/play":
            return httpx.Response(202, json={"ok": True})
        return httpx.Response(404)


def redirect(rec, **kw):
    return SpeakerRedirect(ha_base_url="http://ha:8123", ha_token="hat", engine="tts.eleven",
                           voice="arabella", language="en", targets={OFFICE: "http://lev:8410"},
                           speaker_token="spk", client=httpx.Client(transport=httpx.MockTransport(rec.handle)),
                           async_send=False, **kw)


def test_parse_targets():
    assert parse_targets(f"{OFFICE}=http://lev:8410/ , bad, =x, y=") == {OFFICE: "http://lev:8410"}
    assert parse_targets("") == {}


def test_office_reply_is_rendered_in_her_voice_and_sent_to_the_desk():
    rec = Recorder()
    redirect(rec).say(OFFICE, "[warmly] That's **K-PAX** (2001).")
    (tts_url, tts_auth, tts_body), (play_url, play_auth, play_body) = rec.calls
    assert tts_url == "http://ha:8123/api/tts_get_url" and tts_auth == "Bearer hat"
    assert tts_body == {"engine_id": "tts.eleven", "message": "[warmly] That's K-PAX (2001).",
                        "language": "en", "options": {"voice": "arabella"}}   # tags kept, markdown gone
    assert play_url == "http://lev:8410/play" and play_auth == "Bearer spk"
    assert play_body == {"url": "http://192.168.1.155:8123/api/tts_proxy/x.mp3"}


def test_unmapped_devices_empty_text_and_failures_are_quiet():
    rec = Recorder()
    r = redirect(rec)
    r.say("some-other-satellite", "hi")
    r.say(None, "hi")
    r.say(OFFICE, "   ")
    assert rec.calls == []
    bad = Recorder(tts_status=500)
    redirect(bad).say(OFFICE, "hi")               # HA can't render -> nothing played, nothing raised
    assert [c[0] for c in bad.calls] == ["http://ha:8123/api/tts_get_url"]


def test_arming_needs_map_token_ha_and_voice():
    base = dict(_env_file=None, anthropic_api_key="t")
    assert speaker_redirect_for(Settings(**base)) is None
    armed = dict(base, ha_token="h", voice_speaker_redirect=f"{OFFICE}=http://lev:8410",
                 voice_speaker_token="s", ha_tts_voice="arabella")
    assert speaker_redirect_for(Settings(**armed)) is not None
    assert speaker_redirect_for(Settings(**{**armed, "ha_tts_voice": ""})) is None
    assert speaker_redirect_for(Settings(**{k: v for k, v in armed.items() if k != "voice_speaker_token"})) is None


def test_followup_router_sends_office_followups_to_the_desk(monkeypatch):
    import aerys_v2.speaker_redirect as sr
    from aerys_v2.factory import followup_router_for

    said = []

    class Fake:
        def handles(self, d):
            return d == OFFICE

        def say(self, d, t):
            said.append((d, t))

    monkeypatch.setattr(sr, "speaker_redirect_for", lambda _s: Fake())
    posts = []
    monkeypatch.setattr(httpx, "post", lambda *a, **k: posts.append((a, k)) or httpx.Response(200, request=httpx.Request("POST", "http://x")))
    route = followup_router_for(Settings(_env_file=None, anthropic_api_key="t", ha_token="h",
                                         ha_satellite_map=f"{OFFICE}=assist_satellite.office,bed=assist_satellite.bed"))
    route("lights are off", OFFICE)
    assert said == [(OFFICE, "lights are off")] and posts == []      # not the dead speaker
    route("lights are off", "bed")
    assert len(posts) == 1 and posts[0][1]["json"]["entity_id"] == "assist_satellite.bed"
