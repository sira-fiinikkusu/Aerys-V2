"""His sleep, read to him only (his ask 11/13: "the same full view of me"). Placeholder ids only."""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

import httpx
import pytest

from aerys_v2.tools.his_sleep import NOT_HIM, SHARED, UNSURE, build_his_sleep_tool, describe, parse_sensors

OWNER = "owner-uuid"
NOW = datetime(2026, 1, 10, 13, 0, tzinfo=timezone.utc)          # 8:00 AM Eastern

STATES = {
    "sensor.night": {"state": "232", "attributes": {"last_end": "2026-01-10T07:26:00Z", "last_mins": "172.0",
                                                    "sessions": "2", "unit_of_measurement": "min"}},
    "sensor.last": {"state": "172.0", "attributes": {"startTime": "2026-01-10T04:04:00Z",
                                                     "endTime": "2026-01-10T07:26:00Z",
                                                     "unit_of_measurement": "min"}},
    "sensor.now": {"state": "94", "attributes": {"timestamp": "1768047600000", "unit_of_measurement": "%"},
                   "last_changed": "2026-01-10T12:20:00Z"},
}


def settings(**kw):
    from aerys_v2.config import Settings
    kw.setdefault("ha_sleep_sensors", "night=sensor.night,last=sensor.last,now=sensor.now")
    return Settings(_env_file=None, anthropic_api_key="x", ha_token="t", owner_person_id=OWNER, **kw)


def tool_on(states, fail=False):
    def handle(request):
        entity = request.url.path.rsplit("/", 1)[-1]
        if fail or entity not in states:
            return httpx.Response(404 if not fail else 500)
        return httpx.Response(200, json=states[entity])
    return build_his_sleep_tool(settings(), client=httpx.Client(transport=httpx.MockTransport(handle)),
                                clock=lambda: NOW)


def ask(t, identity, thread="person:owner"):
    return t.func(runtime=SimpleNamespace(config={"configurable": {"identity": identity, "thread_id": thread}}))


HIM_TYPING = {"user_id": OWNER, "platform": "discord", "privacy_context": "private", "channel_kind": "dm"}
HIS_VOICE = {"user_id": OWNER, "voice": True, "privacy_context": "private"}


def test_the_sensors_parse_and_unknown_keys_are_dropped():
    assert parse_sensors("night=sensor.a, last=sensor.b,now=sensor.c,steps=sensor.d,junk") == {
        "night": "sensor.a", "last": "sensor.b", "now": "sensor.c"}
    assert parse_sensors("") == {}


def test_his_night_reads_in_his_clock_with_units():
    out = ask(tool_on(STATES), HIM_TYPING)
    assert "Last night: 3 h 52 min asleep, in 2 stretches; the last one ended 2:26 AM." in out
    assert "Latest stretch: Fri 11:04 PM to 2:26 AM (2 h 52 min)." in out     # began the evening before
    assert "Right now: his phone is 94% sure he's asleep (as of 7:20 AM)." in out
    assert "syncs late" in out


def test_a_sensor_that_is_down_or_missing_is_left_out_and_none_says_so():
    down = {**STATES, "sensor.now": {"state": "unavailable", "attributes": {}}}
    out = ask(tool_on(down), HIM_TYPING)
    assert "Right now" not in out and "Last night" in out
    assert "hasn't reported" in ask(tool_on({}), HIM_TYPING)
    assert "hasn't reported" in ask(tool_on(STATES, fail=True), HIM_TYPING)


def test_an_odd_reading_is_left_out_and_its_siblings_still_read():
    for bad in ("not-a-number", "nan", "-40", "99999"):
        odd = {**STATES, "sensor.night": {"state": bad, "attributes": {}}}
        out = ask(tool_on(odd), HIM_TYPING)
        assert "Last night" not in out and "Latest stretch" in out and "Right now" in out
    over = {"sensor.now": {"state": "125", "attributes": {}}}
    assert "can't tell" in ask(tool_on(over), HIM_TYPING)


def test_an_old_reading_carries_its_date_and_is_never_last_night_or_right_now():
    week_old = {
        "sensor.night": {"state": "300", "attributes": {"last_end": "2026-01-03T12:00:00Z", "sessions": "1"}},
        "sensor.now": {"state": "80", "attributes": {"timestamp": "1767960000000"}},   # Jan 9, 12:00 UTC
    }
    out = ask(tool_on(week_old), HIM_TYPING)
    assert "Latest night his phone reported (ended Jan 3, 7:00 AM): 5 h 0 min asleep." in out
    assert "Last reading (Fri 7:00 AM): his phone was 80% sure he was asleep." in out
    assert "Last night" not in out and "Right now" not in out


def test_last_night_only_for_a_night_that_ended_today():
    # a missed nightly sync: yesterday morning's night is not "last night" (Codex 11/14)
    yesterday = {"sensor.night": {"state": "300", "attributes": {"last_end": "2026-01-09T12:00:00Z"}}}
    out = ask(tool_on(yesterday), HIM_TYPING)
    assert "Latest night his phone reported (ended Fri 7:00 AM): 5 h 0 min asleep." in out and "Last night" not in out


def test_units_are_honored_and_an_unknown_unit_is_left_out():
    hours = {**STATES, "sensor.night": {"state": "7.5", "attributes": {**STATES["sensor.night"]["attributes"],
                                                                       "unit_of_measurement": "h"}}}
    assert "Last night: 7 h 30 min asleep" in ask(tool_on(hours), HIM_TYPING)
    odd = {**STATES, "sensor.last": {"state": "3", "attributes": {**STATES["sensor.last"]["attributes"],
                                                                  "unit_of_measurement": "d"}},
           "sensor.now": {"state": "0.94", "attributes": {"unit_of_measurement": "ratio",
                                                          "timestamp": "1768046400000"}}}   # fresh, 7:00 AM
    out = ask(tool_on(odd), HIM_TYPING)
    assert "Latest stretch" not in out and "Right now" not in out and "Last night" in out


def test_a_zoneless_time_is_read_as_utc():
    naive = {"sensor.last": {"state": "60", "attributes": {"startTime": "2026-01-10T06:00:00",
                                                           "endTime": "2026-01-10T07:00:00"}}}
    assert "Latest stretch: 1:00 AM to 2:00 AM (1 h 0 min)." in ask(tool_on(naive), HIM_TYPING)


@pytest.mark.parametrize("identity,thread,expected", [
    ({**HIS_VOICE, "speaker": {"id": "owner-voice"}}, "voice:office", None),                  # his voice, sure
    ({**HIS_VOICE, "speaker": {"id": "owner-voice", "unsure": True}}, "voice:office", UNSURE),
    ({**HIS_VOICE, "voice_owner_assumed": True}, "voice:office", UNSURE),
    ({"user_id": OWNER, "privacy_context": "private"}, "owner:panel", UNSURE),   # a shared surface's /ask
    ({"user_id": "someone-else", "platform": "discord", "privacy_context": "private"}, "person:x", NOT_HIM),
    (HIM_TYPING, "kael:test", NOT_HIM),
    # his own typed words, but in a room others read (Codex 11/14)
    ({**HIM_TYPING, "privacy_context": "public", "channel_kind": "guild"}, "guild:x", SHARED),
    ({**HIM_TYPING, "platform": "telegram", "privacy_context": "public", "channel_kind": "group"}, "tg:x", SHARED),
    ({**HIM_TYPING, "sealed_room": "family"}, "sealed:x", SHARED),
    ({**HIM_TYPING, "channel_kind": "guild"}, "guild:x", SHARED),               # a guild, whatever it claims
    ({"user_id": OWNER, "platform": "discord"}, "person:owner", SHARED),        # no privacy said: not private
])
def test_only_he_hears_it(identity, thread, expected):
    out = ask(tool_on(STATES), identity, thread)
    assert out == expected if expected else out.startswith("His sleep")


def test_she_has_the_tool_only_when_its_sensors_are_set_and_never_for_a_guest():
    from aerys_v2.factory import _action_tools_armed

    names = lambda s, guest: {t.name for t in _action_tools_armed(s, guest=guest)}
    assert "his_sleep" in names(settings(), False)
    assert "his_sleep" not in names(settings(), True)
    assert "his_sleep" not in names(settings(ha_sleep_sensors=""), False)
