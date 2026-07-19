"""get_weather — the house's own forecast (the Rotonda-Switzerland fix).

Offline proof against a fake HA: current conditions, daily/today/tomorrow,
hourly, the no-invented-percentages rule (precipitation inches only), span
fallback, and the honest-failure contract (HTTP errors and unreachable HA
come back as strings, never raises)."""

import httpx

from aerys_v2.tools.weather import build_weather_tool

ENTITY = "weather.forecast_home"

STATE = {
    "state": "partlycloudy",
    "attributes": {
        "temperature": 88.0,
        "temperature_unit": "°F",
        "humidity": 74,
        "wind_speed": 9.2,
        "wind_speed_unit": "mph",
    },
}

DAILY = [
    {"datetime": "2026-07-19T10:00:00+00:00", "condition": "rainy",
     "temperature": 88.0, "templow": 76.0, "precipitation": 0.42},
    {"datetime": "2026-07-20T10:00:00+00:00", "condition": "sunny",
     "temperature": 91.0, "templow": 75.0, "precipitation": 0.0},
]

HOURLY = [
    {"datetime": "2026-07-19T21:00:00+00:00", "condition": "rainy",
     "temperature": 84.0, "precipitation": 0.15},
    {"datetime": "2026-07-19T22:00:00+00:00", "condition": "cloudy",
     "temperature": 82.0, "precipitation": 0.0},
]


def fake_ha(*, state=STATE, daily=DAILY, hourly=HOURLY, status=200):
    def handler(request: httpx.Request) -> httpx.Response:
        if status != 200:
            return httpx.Response(status)
        if request.url.path == f"/api/states/{ENTITY}":
            return httpx.Response(200, json=state)
        if request.url.path == "/api/services/weather/get_forecasts":
            import json

            kind = json.loads(request.content)["type"]
            fc = daily if kind == "daily" else hourly
            return httpx.Response(
                200, json={"changed_states": [], "service_response": {ENTITY: {"forecast": fc}}}
            )
        return httpx.Response(404)

    return httpx.Client(transport=httpx.MockTransport(handler))


def tool(**kwargs):
    return build_weather_tool(
        base_url="http://ha:8123", token="t", entity_id=ENTITY,
        client=fake_ha(**kwargs),
    )


def test_now_reads_current_conditions():
    out = tool().invoke({"span": "now"})
    assert out == "Right now: partlycloudy, 88°F, 74% humidity, wind 9.2 mph."


def test_today_combines_forecast_and_current():
    out = tool().invoke({"span": "today"})
    assert out.startswith('Today: rainy, 76–88°, 0.42" precipitation')
    assert "Right now: partlycloudy" in out


def test_tomorrow_reads_second_day_and_skips_zero_precip():
    out = tool().invoke({"span": "tomorrow"})
    assert out == "Tomorrow: sunny, 75–91°"


def test_hourly_lists_hours_with_utc_times():
    out = tool().invoke({"span": "hourly"})
    assert out.startswith("Hourly (UTC): ")
    assert '21:00 rainy 84° 0.15"' in out
    assert "22:00 cloudy 82°" in out


def test_no_probability_is_ever_invented():
    for span in ("today", "tomorrow", "hourly"):
        assert "%" not in tool().invoke({"span": span}).replace("% humidity", "")


def test_unknown_span_falls_back_to_now():
    out = tool().invoke({"span": "next week"})
    assert out.startswith("Right now:")


def test_http_error_is_an_honest_string():
    out = tool(status=503).invoke({"span": "now"})
    assert out == "Weather read failed: HA returned 503."


def test_unreachable_ha_is_an_honest_string():
    def handler(_request):
        raise httpx.ConnectError("nope")

    t = build_weather_tool(
        base_url="http://ha:8123", token="t", entity_id=ENTITY,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    out = t.invoke({"span": "today"})
    assert out == "Weather read failed: Home Assistant is unreachable right now."


def test_empty_forecast_is_reported_not_fabricated():
    out = tool(daily=[]).invoke({"span": "today"})
    assert out == "The weather provider returned no daily forecast."
