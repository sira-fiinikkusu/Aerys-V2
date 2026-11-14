"""D47 (11/14): her HA read is the whole reading, the way Kael reads it (units, class, attributes, and
history over time), and his own phone/watch readings answer only to him in his own rooms. Whereabouts
never ride along. Placeholder ids only."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from types import SimpleNamespace

import httpx
import pytest

from aerys_v2.tools.home_control import (
    HISTORY_POINTS,
    build_home_control_tool,
    build_search_entities_tool,
    history_points,
)

OWNER = "owner-uuid"
PRIVATE = ("sensor.phone_", "sensor.watch_")
NOW = datetime(2026, 1, 10, 13, 0, tzinfo=timezone.utc)          # 8:00 AM Eastern

STATES = {
    "sensor.phone_sleep_duration": {
        "entity_id": "sensor.phone_sleep_duration", "state": "172.0", "last_updated": "2026-01-10T07:28:00Z",
        "last_changed": "2026-01-10T07:28:00Z",
        "attributes": {"friendly_name": "Phone Sleep duration", "unit_of_measurement": "min",
                       "device_class": "duration", "startTime": "2026-01-10T04:04:00Z",
                       "endTime": "2026-01-10T07:26:00Z", "icon": "mdi:sleep"}},
    "sensor.watch_heart_rate": {
        "entity_id": "sensor.watch_heart_rate", "state": "64", "last_changed": "2026-01-10T12:55:00Z",
        "attributes": {"friendly_name": "Watch Heart rate", "unit_of_measurement": "bpm",
                       "state_class": "measurement"}},
    "sensor.office_temperature": {
        "entity_id": "sensor.office_temperature", "state": "72", "last_changed": "2026-01-10T12:00:00Z",
        "attributes": {"friendly_name": "Office temperature", "unit_of_measurement": "°F"}},
    "sensor.phone_battery": {
        "entity_id": "sensor.phone_battery", "state": "80", "last_changed": "2026-01-10T12:00:00Z",
        "attributes": {"friendly_name": "Phone battery", "unit_of_measurement": "%",
                       "note": "lat 26.9512, -82.3511", "latitude": 26.9512, "charger": "wireless",
                       "big": "x" * 5000}},
    "device_tracker.phone": {
        "entity_id": "device_tracker.phone", "state": "home", "last_changed": "2026-01-10T12:00:00Z",
        "attributes": {"friendly_name": "Phone", "latitude": 26.9512, "longitude": -82.3511, "gps_accuracy": 5,
                       "source_type": "gps"}},
}
HISTORY = [[
    {"state": "60", "last_changed": "2026-01-10T03:00:00Z"},
    {"state": "60", "last_changed": "2026-01-10T03:05:00Z"},          # a repeat: dropped
    {"state": "unavailable", "last_changed": "2026-01-10T03:10:00Z"},  # dead: dropped
    {"state": "58", "last_changed": "2026-01-10T07:00:00Z"},
    {"state": "64", "last_changed": "2026-01-10T12:55:00Z"},
]]


def ha(calls=None):
    def handle(request):
        if calls is not None:
            calls.append(str(request.url))
        path = request.url.path
        if path.startswith("/api/history/period/"):
            return httpx.Response(200, json=HISTORY)
        if path == "/api/states":
            return httpx.Response(200, json=list(STATES.values()))
        entity = path.rsplit("/", 1)[-1]
        return httpx.Response(200, json=STATES[entity]) if entity in STATES else httpx.Response(404)
    return httpx.Client(transport=httpx.MockTransport(handle))


def tool(calls=None, private=PRIVATE):
    return build_home_control_tool(base_url="http://ha", token="t", canary_entities=frozenset(), client=ha(calls),
                                   private_entities=private, owner_person_id=OWNER, clock=lambda: NOW)


def search(private=PRIVATE):
    return build_search_entities_tool(base_url="http://ha", token="t", client=ha(), private_entities=private,
                                      owner_person_id=OWNER)


def rt(identity, thread="person:owner"):
    return SimpleNamespace(config={"configurable": {"identity": identity, "thread_id": thread}})


HIS_DM = {"user_id": OWNER, "platform": "discord", "privacy_context": "private", "channel_kind": "dm"}


def read(t, entity, identity=HIS_DM, thread="person:owner", op="get_state", **kw):
    out = t.func(operation=op, entity_id=entity, runtime=rt(identity, thread), **kw)
    try:
        return json.loads(out)
    except ValueError:
        return out


def test_a_reading_comes_whole_with_its_unit_and_details():
    r = read(tool(), "sensor.phone_sleep_duration")
    assert r["state"] == "172.0" and r["unit_of_measurement"] == "min" and r["device_class"] == "duration"
    assert r["attributes"] == {"startTime": "2026-01-10T04:04:00Z", "endTime": "2026-01-10T07:26:00Z"}
    assert r["last_updated"] == "2026-01-10T07:28:00Z"


def test_whereabouts_never_ride_along():
    r = read(tool(), "sensor.phone_battery")
    assert "latitude" not in r["attributes"] and "note" not in r["attributes"]          # coordinates, any shape
    assert r["attributes"]["charger"] == "wireless"
    assert len(r["attributes"]["big"]) == 200                                          # one value clipped
    tracker = read(tool(), "device_tracker.phone")
    assert tracker["state"] == "home" and "attributes" not in tracker                    # today's state-only read
    assert "No history" in read(tool(), "device_tracker.phone", op="history")


def test_a_reading_with_many_details_is_capped():
    from aerys_v2.tools.home_control import DETAIL_MAX, reading_details
    data = {"attributes": {f"field_{n}": "y" * 150 for n in range(40)}}
    details = reading_details("sensor.something", data)["attributes"]
    assert details["more"].startswith("further attributes left out")
    assert sum(len(k) + len(str(v)) for k, v in details.items() if k != "more") <= DETAIL_MAX


def test_history_shows_how_a_reading_moved_in_his_clock():
    calls = []
    h = read(tool(calls), "sensor.watch_heart_rate", op="history", hours=12)
    assert h["changes"] == [["Fri 10:00 PM", "60"], ["2:00 AM", "58"], ["7:55 AM", "64"]]
    assert h["unit_of_measurement"] == "bpm" and h["hours"] == 12
    url = next(c for c in calls if "/api/history/period/" in c)
    assert "%2B00%3A00" in url and "filter_entity_id=sensor.watch_heart_rate" in url    # quoted, not a space


def test_history_is_thinned_to_a_page_and_hours_are_bounded():
    rows = [{"state": str(n), "last_changed": f"2026-01-10T{n // 60:02d}:{n % 60:02d}:00Z"} for n in range(600)]
    points = history_points(rows, NOW)
    assert len(points) == HISTORY_POINTS and points[0][1] == "0" and points[-1][1] == "599"
    assert read(tool(), "sensor.office_temperature", op="history", hours=5000)["hours"] == 168


@pytest.mark.parametrize("identity,thread,reads", [
    (HIS_DM, "person:owner", True),
    ({"user_id": OWNER, "voice": True, "speaker": {"id": "owner-voice"}, "privacy_context": "private"}, "voice:office", True),
    ({"user_id": OWNER, "surface": "lens", "privacy_context": "private"}, "glasses:abc", True),          # his glasses
    ({"user_id": OWNER, "platform": "wake", "unattended": True, "privacy_context": "private"}, "wake:1", True),
    ({"user_id": OWNER, "voice": True, "voice_owner_assumed": True, "privacy_context": "private"}, "voice:office", False),
    ({"user_id": OWNER, "privacy_context": "private"}, "owner:panel", False),                               # shared surface
    ({**HIS_DM, "privacy_context": "public", "channel_kind": "guild"}, "guild:x", False),
    ({**HIS_DM, "platform": "telegram", "privacy_context": "public", "channel_kind": "group"}, "tg:x", False),
    ({**HIS_DM, "sealed_room": "family"}, "sealed:x", False),
    ({"user_id": "someone-else", "platform": "discord", "privacy_context": "private", "channel_kind": "dm"}, "person:x", False),
    (HIS_DM, "kael:test", False),
])
def test_his_own_readings_answer_only_in_his_own_rooms(identity, thread, reads):
    for op in ("get_state", "history"):
        out = read(tool(), "sensor.watch_heart_rate", identity, thread, op=op)
        assert isinstance(out, dict) is reads, (op, out)
    # anything else in the house reads anywhere, as before
    assert read(tool(), "sensor.office_temperature", identity, thread)["state"] == "72"


def test_a_room_others_read_hears_why_and_no_runtime_fails_closed():
    guild = {**HIS_DM, "privacy_context": "public", "channel_kind": "guild"}
    assert "others can read this room" in read(tool(), "sensor.watch_heart_rate", guild, "guild:x")
    out = tool().func(operation="get_state", entity_id="sensor.watch_heart_rate")
    assert out.startswith("Couldn't read that")
    mixed = read(tool(), "sensor.office_temperature,sensor.watch_heart_rate", guild, "guild:x")
    assert isinstance(mixed, str) and mixed.startswith("Couldn't read that")         # one private target holds the call
    assert read(tool(private=()), "sensor.watch_heart_rate", guild, "guild:x")["state"] == "64"   # unset = as before


def test_search_shows_his_readings_names_but_not_their_values_outside_his_rooms():
    guild = {**HIS_DM, "privacy_context": "public", "channel_kind": "guild"}
    there = search().func(query="watch heart", runtime=rt(guild, "guild:x"))
    assert "sensor.watch_heart_rate | Watch Heart rate | (his own reading: not here)" in there and "64" not in there
    here = search().func(query="watch heart", runtime=rt(HIS_DM))
    assert "sensor.watch_heart_rate | Watch Heart rate | 64 bpm" in here
    assert "72 °F" in search().func(query="office temperature", runtime=rt(guild, "guild:x"))


def test_the_factory_hands_both_tools_his_private_prefixes(monkeypatch):
    from aerys_v2 import factory
    from aerys_v2.config import Settings
    from aerys_v2.tools import home_control as hc

    seen = {}
    real_control, real_search = hc.build_home_control_tool, hc.build_search_entities_tool
    monkeypatch.setattr(hc, "build_home_control_tool", lambda **kw: seen.setdefault("control", kw) and real_control(**kw))
    monkeypatch.setattr(hc, "build_search_entities_tool", lambda **kw: seen.setdefault("search", kw) and real_search(**kw))
    factory._action_tools_armed(Settings(_env_file=None, anthropic_api_key="x", ha_token="t", owner_person_id=OWNER,
                                         ha_private_entities="sensor.phone_, sensor.watch_"))
    for name in ("control", "search"):
        assert seen[name]["private_entities"] == ("sensor.phone_", "sensor.watch_")
        assert seen[name]["owner_person_id"] == OWNER
