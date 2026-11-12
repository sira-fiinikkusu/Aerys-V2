"""Offline keep-listening contract; call the real /ask handler without a socketpair."""
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import pytest

from aerys_v2.config import Settings
from aerys_v2.factory import followup_router_for
from aerys_v2.transports.http_api import AskRequest, build_app


@pytest.fixture
def rig(monkeypatch):
    from aerys_v2 import keep_listening as kl
    monkeypatch.setattr(kl, "opened_mic", {})
    now = [100.0]
    monkeypatch.setattr(kl, "time", SimpleNamespace(monotonic=lambda: now[0]))
    state = ["on"]
    gets, posts = [], []

    def get(url, **kwargs):
        gets.append((url, kwargs))
        if isinstance(state[0], Exception):
            raise state[0]
        return httpx.Response(200, json={"state": state[0]}, request=httpx.Request("GET", url))

    def post(url, **kwargs):
        posts.append((url, kwargs["json"]))
        return httpx.Response(200, request=httpx.Request("POST", url))

    monkeypatch.setattr(httpx, "get", get)
    monkeypatch.setattr(httpx, "post", post)
    settings = Settings(_env_file=None, anthropic_api_key="placeholder", ha_token="placeholder", ha_base_url="http://ha.example",
                        ha_satellite_map="sat=assist_satellite.example",
                        ha_display_followups="display=esphome.example_show")
    return kl, settings, now, state, gets, posts


def handler(settings, ask, **kwargs):
    app = build_app(ask, "placeholder", settings=settings, **kwargs)
    return next(r.endpoint for r in app.routes if r.path == "/ask")


@pytest.mark.parametrize("state", ["off", "unavailable", RuntimeError("offline")])
def test_disabled_preserves_delivery_and_turn(rig, state):
    kl, settings, _, states, _, posts = rig
    states[0] = state
    followup_router_for(settings)("Ready?", "sat")
    assert posts == [("http://ha.example/api/services/assist_satellite/announce",
                      {"entity_id": "assist_satellite.example", "message": "Ready?", "preannounce": False})]
    kl.mark_opened("sat")
    ask = Mock(return_value="Ready?")
    reply = handler(settings, ask)(AskRequest(text="stop", voice=True, device_id="sat"))
    assert reply.reply == "Ready?"
    assert reply.keep_listening is False
    ask.assert_called_once()


def test_toggle_cache_and_fail_closed(rig):
    kl, settings, now, states, gets, _ = rig
    reader = kl.ToggleReader(settings)
    assert reader.enabled()
    states[0] = RuntimeError("offline")
    now[0] += 4.9
    assert reader.enabled()
    assert len(gets) == 1
    now[0] += .2
    assert not reader.enabled()
    assert len(gets) == 2
    assert gets[0][0] == "http://ha.example/api/states/input_boolean.aerys_keep_listening"
    assert gets[0][1]["headers"] == {"Authorization": "Bearer placeholder"}
    assert not kl.ToggleReader(None).enabled()


@pytest.mark.parametrize("device,text,path,payload", [
    ("sat", " Ready? ", "services/assist_satellite/start_conversation",
     {"entity_id": "assist_satellite.example", "start_message": " Ready? ", "preannounce": False}),
    ("sat", "Ready.", "services/assist_satellite/announce",
     {"entity_id": "assist_satellite.example", "message": "Ready.", "preannounce": False}),
    ("display", "Ready?", "services/esphome/example_show", {"message": "Ready?"}),
    ("phone", "Ready?", "events/aerys_followup", {"text": "Ready?"}),
    (None, "Ready?", "events/aerys_followup", {"text": "Ready?"}),
])
def test_reachout_routes(rig, device, text, path, payload):
    kl, settings, _, _, _, posts = rig
    followup_router_for(settings)(text, device)
    assert posts == [("http://ha.example/api/" + path, payload)]
    assert bool(kl.opened_mic) == path.endswith("start_conversation")


@pytest.mark.parametrize("text,enabled", [("Ready?", True), ("Ready.", False)])
def test_office_redirect(rig, monkeypatch, text, enabled):
    kl, settings, _, _, _, posts = rig
    redirect = Mock()
    redirect.handles.return_value = True
    monkeypatch.setattr("aerys_v2.speaker_redirect.speaker_redirect_for", lambda _: redirect)
    followup_router_for(settings)(text, "sat")
    redirect.say.assert_called_once_with("sat", text)
    assert bool(posts) == enabled
    if enabled:
        assert posts[0] == ("http://ha.example/api/services/assist_satellite/start_conversation",
                            {"entity_id": "assist_satellite.example", "start_message": text, "preannounce": False})


@pytest.mark.parametrize("text,voice,expected", [
    ("Ready?", True, True), ("Ready.", True, False),
    ("[curious] Ready? [softly]", True, True), ("Ready?", False, False),
])
def test_reply_flag(rig, text, voice, expected):
    kl, settings, _, _, _, _ = rig
    reply = handler(settings, Mock(return_value=text))(AskRequest(text="hi", voice=voice, device_id="sat"))
    assert reply.reply == text
    assert reply.keep_listening is expected
    assert ("sat" in kl.opened_mic) is expected


@pytest.mark.parametrize("text", ["stop", "Never mind!", "nevermind", "cancel", "That's all.", "thats all", "That’s all!"])
def test_quiet_dismissal_shared_with_reachout(rig, text):
    kl, settings, _, _, _, _ = rig
    followup_router_for(settings)("Ready?", "sat")
    ask, record, redirect = Mock(), Mock(), Mock()
    route = handler(settings, ask, a2a_memory_fn=record, speaker_redirect=redirect)
    reply = route(AskRequest(text=text, voice=True, device_id="sat"))
    assert reply.reply == ""
    assert not reply.keep_listening
    ask.assert_not_called()
    record.assert_not_called()
    redirect.say.assert_not_called()
    assert "sat" not in kl.opened_mic


@pytest.mark.parametrize("age,device,opened,text,voice", [
    (30.1, "sat", True, "never mind", True), (1, "other", True, "never mind", True),
    (1, "sat", False, "never mind", True), (1, None, True, "never mind", True),
    (1, "sat", True, "stop the music", True), (1, "sat", True, "stop", False),
])
def test_dismissal_scope(rig, age, device, opened, text, voice):
    kl, settings, now, _, _, _ = rig
    if opened:
        kl.mark_opened("sat")
    now[0] += age
    ask = Mock(return_value="Normal reply.")
    reply = handler(settings, ask)(AskRequest(text=text, voice=voice, device_id=device))
    assert reply.reply == "Normal reply."
    assert not reply.keep_listening
    ask.assert_called_once()


def test_answer_consumes_open_mic(rig):
    kl, settings, _, _, _, _ = rig
    ask = Mock(side_effect=["Ready?", "Okay.", "Stopping music."])
    route = handler(settings, ask)
    for text in ["hello", "yes", "stop"]:
        reply = route(AskRequest(text=text, voice=True, device_id="sat"))
    assert reply.reply == "Stopping music."
    assert ask.call_count == 3


@pytest.mark.parametrize("status,payload", [(401, {}), (404, {}), (500, {}), (200, []), (200, {}), (200, {"state": "ON"})])
def test_bad_toggle_response_is_off(rig, monkeypatch, status, payload):
    kl, settings, _, _, _, _ = rig
    monkeypatch.setattr(httpx, "get", lambda url, **kw: httpx.Response(
        status, json=payload, request=httpx.Request("GET", url)))
    assert not kl.ToggleReader(settings).enabled()


def test_failed_delivery_does_not_arm_dismissal(rig, monkeypatch):
    kl, settings, _, _, _, _ = rig
    monkeypatch.setattr(httpx, "post", lambda url, **kw: httpx.Response(
        500, request=httpx.Request("POST", url)))
    with pytest.raises(httpx.HTTPStatusError):
        followup_router_for(settings)("Ready?", "sat")
    assert not kl.opened_mic


def test_reply_opens_window_and_dismissal_is_once(rig):
    _, settings, _, _, _, _ = rig
    ask = Mock(side_effect=["Ready?", "Normal stop."])
    route = handler(settings, ask)
    first = route(AskRequest(text="hi", voice=True, device_id="sat"))
    assert first.model_dump() == {"reply": "Ready?", "thread_id": "voice:beta", "keep_listening": True}
    assert route(AskRequest(text="cancel", voice=True, device_id="sat")).reply == ""
    assert route(AskRequest(text="cancel", voice=True, device_id="sat")).reply == "Normal stop."
    assert ask.call_count == 2


def test_toggle_off_after_open_and_expiry(rig):
    _, settings, now, states, _, _ = rig
    ask = Mock(side_effect=["Ready?", "Normal stop."])
    route = handler(settings, ask)
    route(AskRequest(text="hi", voice=True, device_id="sat"))
    states[0] = "off"
    now[0] += 6
    assert route(AskRequest(text="stop", voice=True, device_id="sat")).reply == "Normal stop."
    assert ask.call_count == 2


def test_her_voice_switch_flips_his_toggle_and_says_which():
    from types import SimpleNamespace

    from pydantic import SecretStr

    from aerys_v2.keep_listening import ENTITY, build_keep_listening_tool

    calls = []

    class Ok:
        def raise_for_status(self):
            pass

    settings = SimpleNamespace(ha_token=SecretStr("placeholder"), ha_base_url="http://ha.example:8123/")
    tool = build_keep_listening_tool(settings, post=lambda url, **kw: calls.append((url, kw["json"])) or Ok())
    assert tool.invoke({"on": True}).startswith("Keep listening is on")
    assert tool.invoke({"on": False}).startswith("Keep listening is off")
    assert calls == [("http://ha.example:8123/api/services/input_boolean/turn_on", {"entity_id": ENTITY}),
                     ("http://ha.example:8123/api/services/input_boolean/turn_off", {"entity_id": ENTITY})]

    def broken(url, **kw):
        raise ConnectionError("ha away")

    assert build_keep_listening_tool(settings, post=broken).invoke({"on": True}).startswith("Couldn't change it")
    assert build_keep_listening_tool(SimpleNamespace(ha_token=None), post=broken).invoke(
        {"on": True}).startswith("Couldn't change it")
