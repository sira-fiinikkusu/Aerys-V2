"""Travel-zone regression coverage using synthetic sensor data only."""
import json
from datetime import UTC, datetime, timedelta

import httpx
import pytest

from aerys_v2 import factory
from aerys_v2.config import Settings
from aerys_v2.services import presence, location_guard
from aerys_v2.tools.calendar_events import build_calendar_tool

ZONE = (timedelta(hours=-8), "Pacific Standard Time")


@pytest.fixture(autouse=True)
def isolated(monkeypatch):
    token = location_guard.LOCATION_TURN.set(set())
    zone_token = presence.TRAVEL_ZONE.set({})
    monkeypatch.setattr(location_guard, "RECENT_LOCATION_TURNS", {})
    monkeypatch.setattr(factory, "_OWNER_ID", "synthetic-owner")
    yield
    presence.TRAVEL_ZONE.reset(zone_token)
    location_guard.LOCATION_TURN.reset(token)


def snapshot(monkeypatch, offset=-28800000, name=ZONE[1], state="not_home", entity="sensor.synthetic_zone"):
    calls = []
    def handler(request):
        calls.append(json.loads(request.content)["template"])
        return httpx.Response(200, json={"rooms": [], "person": state, "room": "",
            "local_zone": {"name": name, "utc_offset": offset}})
    real = httpx.Client
    monkeypatch.setattr(httpx, "Client", lambda **kw: real(transport=httpx.MockTransport(handler), **kw))
    settings = Settings(_env_file=None, anthropic_api_key="synthetic", ha_token="synthetic",
        presence_context=True, presence_person_entity="person.synthetic", presence_timezone_entity=entity)
    return factory.presence_fn_for(settings)(), calls


@pytest.mark.parametrize("offset", [True, "-28800000", 50400001, -50400001, 1000, None])
def test_bad_offsets(monkeypatch, offset):
    result, _ = snapshot(monkeypatch, offset=offset)
    assert result.local_zone is None


@pytest.mark.parametrize("name", ["", "  ", "unknown", "UNAVAILABLE", "none", "null", "x" * 65, None, 5])
def test_bad_names(monkeypatch, name):
    result, _ = snapshot(monkeypatch, name=name)
    assert result.local_zone is None


@pytest.mark.parametrize("offset", [-50400000, 50400000, 19800000, 0, -28800000])
def test_valid_offsets(monkeypatch, offset):
    result, _ = snapshot(monkeypatch, offset=offset)
    assert result.local_zone == (timedelta(milliseconds=offset), ZONE[1])


@pytest.mark.parametrize("state,entity", [("home", "sensor.synthetic_zone"), ("unknown", "sensor.synthetic_zone"), ("not_home", "")])
def test_zone_disabled_or_home(monkeypatch, state, entity):
    result, _ = snapshot(monkeypatch, state=state, entity=entity)
    assert result.local_zone is None


def test_template_gates_sensor_reads(monkeypatch):
    result, calls = snapshot(monkeypatch)
    assert result.local_zone == ZONE and len(calls) == 1
    # Exercise the Python-compatible conditional independently of HA filters.
    expression = calls[0].split("'local_zone': ", 1)[1].split(", 'geocoded_state':", 1)[0]
    reads = []
    for home in ("home", "not_home"):
        reads.clear()
        def states(entity):
            if entity == "person.synthetic": return home
            reads.append(entity)
            return ZONE[1]
        def attr(entity, key):
            reads.append(entity)
            return -28800000
        rendered = eval(expression, {"__builtins__": {}, "states": states, "state_attr": attr})
        assert reads.count("sensor.synthetic_zone") == (0 if home == "home" else 2)
        assert bool(rendered) == (home != "home")
    assert calls[0].count("sensor.synthetic_zone") == 2


@pytest.mark.parametrize("instant,gap,example", [
    ("2026-10-31T04:05:00+00:00", "4 hours behind", "6:00 AM"),
    ("2026-11-02T04:05:00+00:00", "3 hours behind", "7:00 AM"),
])
def test_dst_clock_and_example(monkeypatch, instant, gap, example):
    class Clock(datetime):
        @classmethod
        def now(cls, tz=None): return datetime.fromisoformat(instant).astimezone(tz)
    monkeypatch.setattr(presence, "datetime", Clock)
    text = "\n".join(presence.format_whereabouts("not_home", "", local_zone=ZONE))
    assert f"His local time there is 8:05 PM ({ZONE[1]}), {gap} Eastern." in text
    assert f"10:00 AM Eastern is {example} for him" in text
    assert "always tell him times in his local time" in text
    assert presence.NO_TIMES_ZONE in text and presence.NO_TIMES not in text and "never save" in text


@pytest.mark.parametrize("offset,comparison,example", [(5.5, "10.5 hours ahead of Eastern", "8:30 PM"), (-5, "the same as Eastern", "10:00 AM")])
def test_half_hour_and_same_zone(monkeypatch, offset, comparison, example):
    class Clock(datetime):
        @classmethod
        def now(cls, tz=None): return datetime(2026, 11, 2, 12, tzinfo=UTC).astimezone(tz)
    monkeypatch.setattr(presence, "datetime", Clock)
    text = "\n".join(presence.format_whereabouts("not_home", "", {"locality": "Example City"}, (timedelta(hours=offset), "Synthetic Zone")))
    assert comparison in text and f"10:00 AM Eastern is {example} for him" in text
    assert "Use this city for suggestions." in text
    assert "Use this city for local time" not in text


def test_home_and_voice_suppress_zone():
    assert presence.format_whereabouts("home", "", local_zone=ZONE) == presence.format_whereabouts("home", "")
    snap = factory.PresenceSnapshot([], ("not_home", ""), local_zone=ZONE)
    identity = {"user_id": "synthetic-owner", "privacy_context": "private"}
    assert "local time there" in factory.presence_block(identity, lambda: snap)
    assert location_guard.LOCATION_TURN.get() == {"location"}
    location_guard.LOCATION_TURN.get().clear()
    assert "local time there" not in factory.presence_block(dict(identity, voice=True), lambda: snap)
    assert not location_guard.LOCATION_TURN.get()
    assert "local time there" not in presence.format_presence([], "office", ("not_home", ""), local_zone=ZONE)


EVENTS = [
    {"summary": "Meeting", "start": {"dateTime": "2026-11-02T10:00:00-05:00"}, "end": {"dateTime": "2026-11-02T10:30:00-05:00"}},
    {"summary": "Rollover", "start": {"dateTime": "2026-11-02T05:30:00+00:00"}, "end": {"dateTime": "2026-11-02T06:00:00+00:00"}},
    {"summary": "Day", "start": {"date": "2026-11-02"}, "end": {"date": "2026-11-03"}},
]


def calendar(**kwargs):
    return build_calendar_tool(base_url="http://ha.test", token="synthetic", entities=("calendar.synthetic",),
        client=httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200, json=EVENTS))), **kwargs).invoke({})


def test_calendar_conversion():
    text = calendar(local_zone_fn=lambda: (timedelta(hours=-7), "Synthetic Zone"))
    assert text.startswith("Times shown in his local time, Synthetic Zone, with Eastern in brackets.\n")
    assert "8:00 AM–8:30 AM his time (10:00 AM–10:30 AM Eastern) — Meeting" in text
    assert "Sun Nov 1:\n  10:30 PM–11:00 PM his time (12:30 AM–1:00 AM Eastern) — Rollover" in text
    assert "Mon Nov 2:\n  all day — Day [synthetic]" in text
    assert location_guard.LOCATION_TURN.get() == {"location"}


def raises():
    raise RuntimeError("synthetic")


@pytest.mark.parametrize("fn", [None, raises, lambda: None, lambda: "garbage", lambda: (True, "zone"),
    lambda: (timedelta(hours=15), "zone"), lambda: (timedelta(seconds=1), "zone"), lambda: (timedelta(), "unknown")])
def test_calendar_fallback_identical(fn):
    expected = "Mon Nov 2:\n  all day — Day [synthetic]\n  5:30 AM–6:00 AM — Rollover [synthetic]\n  10:00 AM–10:30 AM — Meeting [synthetic]"
    assert calendar() == expected
    assert calendar(local_zone_fn=fn) == expected
    assert not location_guard.LOCATION_TURN.get()


@pytest.mark.parametrize("payload", [{}, {"name": "Synthetic Zone"}, {"utc_offset": 0}, None, "garbage"])
def test_missing_sensor_fields(monkeypatch, payload):
    real = httpx.Client
    monkeypatch.setattr(httpx, "Client", lambda **kw: real(transport=httpx.MockTransport(
        lambda r: httpx.Response(200, json={"rooms": [], "person": "not_home", "room": "", "local_zone": payload})), **kw))
    settings = Settings(_env_file=None, anthropic_api_key="synthetic", ha_token="synthetic",
        presence_context=True, presence_timezone_entity="sensor.synthetic_zone")
    assert factory.presence_fn_for(settings)().local_zone is None


@pytest.mark.parametrize("provider", [None, raises, lambda: factory.PresenceSnapshot([], local_zone=ZONE)])
def test_factory_wires_zone_provider(monkeypatch, provider):
    import aerys_v2.tools.calendar_events as module
    captured = {}
    class Captured(Exception): pass
    def capture(**kwargs):
        captured.update(kwargs)
        raise Captured
    # The calendar never reads presence itself; it gets the zone presence_block chose.
    monkeypatch.setattr(factory, "presence_fn_for", lambda _: pytest.fail("calendar must not read presence"))
    monkeypatch.setattr(module, "build_calendar_tool", capture)
    settings = Settings(_env_file=None, anthropic_api_key="synthetic", ha_token="synthetic",
        ha_calendar_entities="calendar.synthetic", presence_timezone_entity="sensor.synthetic_zone")
    with pytest.raises(Captured):
        factory._action_tools_armed(settings)
    fn = captured["local_zone_fn"]
    assert fn is presence.current_travel_zone and fn() is None
    if provider is not None and provider is not raises:
        presence.TRAVEL_ZONE.get()["zone"] = provider().local_zone
        assert fn() == ZONE


def owner(**extra):
    return dict({"user_id": "synthetic-owner", "privacy_context": "private"}, **extra)


@pytest.mark.parametrize("identity,shown", [
    (owner(), True),
    (owner(voice=True), False),
    (owner(privacy_context="public"), False),
    ({"user_id": "someone-else", "privacy_context": "private"}, False),
])
def test_calendar_follows_presence_fence(identity, shown):
    snap = factory.PresenceSnapshot([], ("not_home", ""), local_zone=ZONE)
    factory.presence_block(identity, lambda: snap)
    text = calendar(local_zone_fn=presence.current_travel_zone)
    assert ("his time" in text) is shown
    assert ("Pacific Standard Time" in text) is shown


def test_calendar_zone_held_when_home():
    snap = factory.PresenceSnapshot([], ("home", ""), local_zone=ZONE)
    factory.presence_block(owner(), lambda: snap)
    assert presence.current_travel_zone() is None
    assert "his time" not in calendar(local_zone_fn=presence.current_travel_zone)


def test_no_holder_outside_ask():
    token = presence.TRAVEL_ZONE.set(None)
    try:
        factory.presence_block(owner(), lambda: factory.PresenceSnapshot([], ("not_home", ""), local_zone=ZONE))
        assert presence.current_travel_zone() is None
    finally:
        presence.TRAVEL_ZONE.reset(token)


def test_factory_feature_off_does_not_read_presence(monkeypatch):
    import aerys_v2.tools.calendar_events as module
    class Captured(Exception): pass
    def capture(**kwargs):
        assert kwargs["local_zone_fn"] is None
        raise Captured
    monkeypatch.setattr(factory, "presence_fn_for", lambda _: pytest.fail("feature off"))
    monkeypatch.setattr(module, "build_calendar_tool", capture)
    with pytest.raises(Captured):
        factory._action_tools_armed(Settings(_env_file=None, anthropic_api_key="synthetic",
            ha_token="synthetic", ha_calendar_entities="calendar.synthetic"))
