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
        if path.startswith("/api/services/media_player/") or path.startswith("/api/states/media_player."):
            return httpx.Response(200, json=[] if "services" in path else {"state": "playing", "attributes": {}})
        return httpx.Response(404, json={})

    @property
    def sent(self):
        return [b for p, b in self.calls if p.startswith("/api/services/")]


def tool(ha, notify=NOTIFY, sensor=SENSOR, players=None, **more):
    return build_music_tool(
        base_url="http://ha.test", token="t", config_entry_id="entry-1",
        players=players or {"dev-office": "media_player.office_satellite"}, default_player="media_player.office_satellite",
        **more,
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
    wake_unmarked = him(platform="wake")                      # a wake-up without the flag is still not him asking
    unattended = him(unattended=True)
    for config in (someone, kael, unsure_voice, wake, wake_unmarked, unattended):
        ha = HA(attributes=SPOTIFY)
        out = call(ha, "pause", config=config)
        assert not out.startswith(WRITE_OK_PREFIX), out
        assert ha.calls == [], "not even a read of his phone for someone else"


class Recorder:
    """Stands in for Music Assistant and the DJ store: any use at all is recorded."""

    def __init__(self):
        self.used = []

    def call(self, *a, **k):
        self.used.append(("call", a, k))
        return "[]"

    def put(self, *a, **k):
        self.used.append(("put", a, k))

    def get(self, *a, **k):
        self.used.append(("get", a, k))


def test_every_other_operation_on_the_phone_is_refused_before_anything_happens():
    for op in ("play", "volume", "play_next", "add_to_queue", "shuffle", "repeat", "now_playing", "up_next",
               "more_like_this", "dj", "like", "add_to_library", "new_playlist", "add_to_playlist",
               "my_playlists", "my_taste", "recently_played", "queue", "nonsense"):
        ha, ma, dj = HA(attributes=SPOTIFY), Recorder(), Recorder()
        out = tool(ha, ma=ma, dj_store=dj).invoke(
            {"operation": op, "target": "phone", "query": "on", "playlist": "Focus"}, config=him())
        assert not out.startswith(WRITE_OK_PREFIX) and "can't start a song" in out, (op, out)
        assert ha.calls == [] and ma.used == [] and dj.used == [], op


def test_a_malformed_reading_is_a_plain_refusal():
    for attributes in (["unexpected"], "text", 7):
        ha = HA(attributes=attributes)
        out = call(ha, "pause")
        assert "nothing was sent" in out and ha.sent == []


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
    for yes in ("phone", "my phone", "on his Phone", "my cellphone", "the smartphone", "my cell phone", "mobile"):
        assert is_phone_target(yes), yes
    for no in ("", "office", "headphones", "phone booth", "phone room", "the office phone speaker"):
        assert not is_phone_target(no), no
    odd = {"playback_state_": "Playing", "playback_state_not a package": "Playing", "playback_state_x": "Playing"}
    assert phone_session(odd, "pause") == "Nothing is playing on the phone right now."


def test_the_office_speaker_still_pauses_as_before():
    ha = HA(attributes=SPOTIFY)
    out = tool(ha).invoke({"operation": "pause", "target": "office"}, config=him())
    assert out == f"{WRITE_OK_PREFIX} pause sent to office satellite."
    assert ha.sent == [{"entity_id": "media_player.office_satellite"}]
    assert [p for p, _ in ha.calls if p.startswith("/api/services/")] == ["/api/services/media_player/media_pause"]


def test_a_speaker_that_is_called_phone_stays_a_speaker():
    ha = HA(attributes=SPOTIFY)
    players = {"dev-office": "media_player.office_satellite", "dev-hall": "media_player.phone"}
    out = tool(ha, players=players).invoke({"operation": "pause", "target": "phone"}, config=him())
    assert out == f"{WRITE_OK_PREFIX} pause sent to phone."
    assert ha.sent == [{"entity_id": "media_player.phone"}]


def test_her_instructions_name_the_phone():
    from aerys_v2.factory import MUSIC_OVERLAY

    assert "target='phone'" in MUSIC_OVERLAY and "can't start a song on the phone" in MUSIC_OVERLAY


def test_a_speaker_with_phone_inside_its_name_does_not_take_his_phone():
    for entity in ("media_player.headphones", "media_player.phone_booth", "media_player.office_phone_speaker"):
        ha, ma = HA(attributes=SPOTIFY), Recorder()
        players = {"dev-office": "media_player.office_satellite", "dev-x": entity}
        out = tool(ha, players=players).invoke({"operation": "pause", "target": "my phone"}, config=him())
        assert "sent to his phone" in out, (entity, out)
        assert ha.sent == [{"message": "command_media",
                            "data": {"media_command": "pause", "media_package_name": "com.spotify.music"}}]
        out = tool(ha, players=players, ma=ma).invoke({"operation": "new_playlist", "target": "phone", "query": "X"}, config=him())
        assert "can't start a song" in out and ma.used == []
