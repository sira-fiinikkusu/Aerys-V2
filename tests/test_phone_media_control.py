"""Board #264 (11/26): he asked for his phone as a place she can control. She can pause, resume, stop
and skip whatever app is playing there, through the Home Assistant phone app's media command, for him
only. She cannot start a song there, and she never guesses which app."""

import json

import httpx

from aerys_v2.tools.home_control import WRITE_OK_PREFIX
from aerys_v2.tools.music import build_music_tool, is_phone_target, phone_session

OWNER = "person-1"
SENSOR = "sensor.test_phone_media_session"
NOTIFY = "mobile_app_test_phone"


def attrs(**apps):
    """apps: package (dots as __) -> (state, title)."""
    out = {"total_media_session_count": len(apps), "friendly_name": "Phone Media session"}
    for key, (state, title) in apps.items():
        package = key.replace("__", ".")
        out[f"playback_state_{package}"] = state
        out[f"title_{package}"] = title
    return out


class HA:
    def __init__(self, state="Playing", attributes=None, sensor_status=200, notify_status=200):
        self.state, self.attributes = state, attributes
        self.sensor_status, self.notify_status = sensor_status, notify_status
        self.calls: list[tuple[str, dict]] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        body = json.loads(request.content) if request.content else {}
        self.calls.append((path, body))
        if path == f"/api/states/{SENSOR}":
            return httpx.Response(self.sensor_status, json={"state": self.state, "attributes": self.attributes or {}})
        if path == f"/api/services/notify/{NOTIFY}":
            return httpx.Response(self.notify_status, json=[])
        return httpx.Response(404, json={})

    @property
    def sent(self):
        return [b for p, b in self.calls if p.startswith("/api/services/")]


def tool(ha, notify=NOTIFY, sensor=SENSOR):
    return build_music_tool(
        base_url="http://ha.test", token="t", config_entry_id="entry-1",
        players={"dev-office": "media_player.office_satellite"}, default_player="media_player.office_satellite",
        client=httpx.Client(transport=httpx.MockTransport(ha)), owner_person_id=OWNER,
        phone_notify=notify, phone_media_sensor=sensor,
    )


def him(**extra):
    return {"configurable": {"identity": {"user_id": OWNER, "platform": "discord", **extra}, "thread_id": "discord:dm:1"}}


SPOTIFY = attrs(com__spotify__music=("Playing", "Example Song"))


def call(ha, op, config=None, target="phone", **kw):
    return tool(ha, **kw).invoke({"operation": op, "target": target}, config=config or him())


def test_pause_goes_to_the_app_that_is_playing():
    ha = HA(attributes=SPOTIFY)
    out = call(ha, "pause")
    assert out.startswith(WRITE_OK_PREFIX) and "phone" in out and "spotify" in out and "Example Song" in out
    assert ha.sent == [{"message": "command_media",
                        "data": {"media_command": "pause", "media_package_name": "com.spotify.music"}}]


def test_each_operation_sends_its_own_command():
    for op, command in (("next", "next"), ("previous", "previous"), ("stop", "stop")):
        ha = HA(attributes=SPOTIFY)
        assert call(ha, op).startswith(WRITE_OK_PREFIX)
        assert ha.sent[0]["data"]["media_command"] == command
    ha = HA(state="Paused", attributes=attrs(com__audible__application=("Paused", "A Book")))
    assert call(ha, "resume").startswith(WRITE_OK_PREFIX)
    assert ha.sent[0]["data"] == {"media_command": "play", "media_package_name": "com.audible.application"}


def test_it_says_sent_and_never_claims_the_phone_did_it():
    out = call(HA(attributes=SPOTIFY), "pause")
    assert "sent to his phone" in out
    assert "paused" not in out.lower()


def test_nothing_playing_sends_nothing():
    for state, attributes in (("unavailable", {}), ("unknown", {}), ("Paused", attrs(com__spotify__music=("Paused", "S"))),
                              ("Stopped", attrs(com__spotify__music=("Stopped", "S")))):
        ha = HA(state=state, attributes=attributes)
        out = call(ha, "pause")
        assert not out.startswith(WRITE_OK_PREFIX), out
        assert ha.sent == []


def test_resume_with_nothing_paused_sends_nothing():
    ha = HA(attributes=SPOTIFY)
    assert "already playing" in call(ha, "resume")
    assert ha.sent == []


def test_two_apps_playing_is_asked_about_not_guessed():
    ha = HA(attributes=attrs(com__spotify__music=("Playing", "S"), com__google__android__youtube=("Playing", "V")))
    out = call(ha, "pause")
    assert "which one" in out and "spotify" in out and "youtube" in out
    assert ha.sent == []


def test_pause_picks_the_playing_app_over_a_paused_one():
    ha = HA(attributes=attrs(com__spotify__music=("Paused", "S"), com__google__android__youtube=("Playing", "V")))
    assert call(ha, "pause").startswith(WRITE_OK_PREFIX)
    assert ha.sent[0]["data"]["media_package_name"] == "com.google.android.youtube"


def test_only_he_can_control_his_phone():
    someone = {"configurable": {"identity": {"user_id": "person-2", "platform": "discord"}, "thread_id": "discord:dm:2"}}
    kael = {"configurable": {"identity": {"user_id": OWNER, "platform": "discord"}, "thread_id": "kael:aerys-line"}}
    unsure_voice = {"configurable": {"identity": {"user_id": OWNER, "platform": "voice", "device_id": "dev-office",
                                                  "speaker": {"id": "unknown"}}, "thread_id": "voice:dev-office"}}
    wake = him(platform="wake", unattended=True)
    for config in (someone, kael, unsure_voice, wake):
        ha = HA(attributes=SPOTIFY)
        out = call(ha, "pause", config=config)
        assert not out.startswith(WRITE_OK_PREFIX), out
        assert ha.calls == [], "not even a read of his phone for someone else"


def test_she_cannot_start_a_song_or_set_volume_on_the_phone():
    for op in ("play", "volume", "play_next", "shuffle"):
        ha = HA(attributes=SPOTIFY)
        out = tool(ha).invoke({"operation": op, "target": "phone", "query": "something"}, config=him())
        assert not out.startswith(WRITE_OK_PREFIX) and "can't start a song" in out
        assert ha.calls == []


def test_without_the_phone_configured_it_is_out_of_reach():
    for kw in ({"notify": ""}, {"sensor": ""}):
        ha = HA(attributes=SPOTIFY)
        out = call(ha, "pause", **kw)
        assert "can't reach" in out and ha.calls == []


def test_failures_are_said_plainly():
    ha = HA(attributes=SPOTIFY, notify_status=500)
    assert "FAILED" in call(ha, "pause")
    ha = HA(attributes=SPOTIFY, sensor_status=500)
    out = call(ha, "pause")
    assert "nothing was sent" in out and ha.sent == []


def test_a_room_name_is_not_the_phone_and_a_strange_key_is_not_an_app():
    assert is_phone_target("phone") and is_phone_target("my phone") and is_phone_target("on his Phone")
    assert not is_phone_target("") and not is_phone_target("office") and not is_phone_target("headphones")
    odd = {"playback_state_": "Playing", "playback_state_not a package": "Playing", "playback_state_x": "Playing"}
    assert phone_session(odd, "pause") == "Nothing is playing on the phone right now."


def test_the_office_speaker_still_pauses_as_before():
    ha = HA(attributes=SPOTIFY)
    out = tool(ha).invoke({"operation": "pause", "target": "office"}, config=him())
    assert [p for p, _ in ha.calls if "notify" in p] == []
    assert "office" in out


def test_her_instructions_name_the_phone():
    from aerys_v2.factory import MUSIC_OVERLAY

    assert "target='phone'" in MUSIC_OVERLAY and "can't start a song on the phone" in MUSIC_OVERLAY
