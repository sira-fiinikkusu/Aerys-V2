"""calendar_events (gap #27) — offline, MockTransport all the way.

What these prove: only ALLOWLISTED calendars are ever fetched (the leak-proof
property), events merge chronologically across calendars under day headers,
all-day events render with HA's exclusive-end quirk handled, a dead calendar
degrades to a partial answer (or an honest unreachable when nothing loaded),
the days window clamps, the event cap announces truncation, and an empty
window says so plainly. No network.
"""

import json

import httpx

from aerys_v2.tools.calendar_events import (
    DAYS_MAX,
    EVENT_CAP,
    build_calendar_tool,
    calendar_set,
)

PERSONAL = "calendar.personal"
HOME = "calendar.home"


def timed(day, start, end, summary, location=None):
    return {
        "summary": summary,
        "start": {"dateTime": f"2026-08-{day:02d}T{start}:00-04:00"},
        "end": {"dateTime": f"2026-08-{day:02d}T{end}:00-04:00"},
        "location": location,
    }


def all_day(start_day, end_day, summary):
    return {
        "summary": summary,
        "start": {"date": f"2026-08-{start_day:02d}"},
        "end": {"date": f"2026-08-{end_day:02d}"},  # HA end is EXCLUSIVE
    }


def make_tool(payloads, *, entities=(PERSONAL, HOME), fail_for=()):
    """payloads: {entity_id: [events]}; fail_for: entities answering 500."""
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        entity = request.url.path.rsplit("/", 1)[-1]
        seen.append(entity)
        if entity in fail_for:
            return httpx.Response(500, text="boom")
        return httpx.Response(200, json=payloads.get(entity, []))

    tool = build_calendar_tool(
        base_url="http://ha.test",
        token="tok",
        entities=entities,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    return tool, seen


def test_calendar_set_parses_csv():
    assert calendar_set(" calendar.a, ,calendar.b ") == ("calendar.a", "calendar.b")
    assert calendar_set("") == ()


def test_only_allowlisted_calendars_are_fetched():
    tool, seen = make_tool({PERSONAL: [timed(10, "09:00", "10:00", "Standup")]})
    tool.invoke({"days": 7})
    # the transport only ever saw the two allowlisted entity ids — nothing else
    assert sorted(set(seen)) == sorted({PERSONAL, HOME})


def test_events_merge_across_calendars_in_time_order_with_day_headers():
    tool, _ = make_tool(
        {
            PERSONAL: [timed(11, "14:00", "15:00", "Endo appointment")],
            HOME: [
                timed(10, "18:00", "19:00", "Game night"),
                timed(11, "09:00", "09:30", "Trash pickup"),
            ],
        }
    )
    out = tool.invoke({"days": 7})
    lines = out.splitlines()
    assert lines[0].startswith("Mon Aug 10")
    order = [l for l in lines if "—" in l]
    assert "Game night" in order[0]
    assert "Trash pickup" in order[1]
    assert "Endo appointment" in order[2]
    assert "[home]" in order[0] and "[personal]" in order[2]


def test_all_day_exclusive_end_and_multi_day_span():
    tool, _ = make_tool(
        {PERSONAL: [all_day(10, 11, "Anniversary"), all_day(12, 15, "Trip")]},
        entities=(PERSONAL,),
    )
    out = tool.invoke({"days": 14})
    assert "all day — Anniversary" in out            # one-day event: no span text
    assert "all day through Fri Aug 14 — Trip" in out  # exclusive end handled


def test_location_rides_the_line():
    tool, _ = make_tool(
        {PERSONAL: [timed(10, "09:00", "10:00", "VA visit", location="St. Petersburg FL")]},
        entities=(PERSONAL,),
    )
    out = tool.invoke({"days": 7})
    assert "@ St. Petersburg FL" in out


def test_empty_window_is_honest():
    tool, _ = make_tool({})
    assert "Nothing on the calendar" in tool.invoke({"days": 7})


def test_dead_calendar_degrades_to_partial_answer():
    tool, _ = make_tool(
        {PERSONAL: [timed(10, "09:00", "10:00", "Standup")]},
        fail_for=(HOME,),
    )
    out = tool.invoke({"days": 7})
    assert "Standup" in out
    assert "couldn't read: home" in out


def test_all_dead_is_honest_unreachable():
    tool, _ = make_tool({}, fail_for=(PERSONAL, HOME))
    out = tool.invoke({"days": 7})
    assert "unreachable" in out and "personal" in out and "home" in out


def test_days_clamped_and_default_window():
    # Non-int input never reaches the tool body — LangChain's schema validation
    # rejects it and the model retries; the clamp guards the int range only.
    tool, seen = make_tool({})
    tool.invoke({"days": 999})   # clamps to DAYS_MAX, still runs
    tool.invoke({})              # default window
    assert len(seen) == 4  # two calendars, two invocations — no crash
    assert DAYS_MAX == 31


def test_event_cap_announces_truncation():
    many = [timed(10, "09:00", "10:00", f"Event {i}") for i in range(EVENT_CAP + 5)]
    tool, _ = make_tool({PERSONAL: many}, entities=(PERSONAL,))
    out = tool.invoke({"days": 7})
    assert f"first {EVENT_CAP} events" in out


def test_factory_arms_calendar_tool_only_when_entities_set():
    from aerys_v2.config import Settings
    from aerys_v2.factory import action_tools_for

    base = dict(_env_file=None, anthropic_api_key="sk-test", ha_token="tok")
    names_without = {t.name for t in action_tools_for(Settings(**base))}
    names_with = {
        t.name
        for t in action_tools_for(
            Settings(**base, ha_calendar_entities="calendar.personal")
        )
    }
    assert "calendar_events" not in names_without
    assert "calendar_events" in names_with
