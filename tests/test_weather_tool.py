"""get_weather — the house's own forecast (the Rotonda-Switzerland fix).

Offline proof against a fake HA: current conditions, daily/today/tomorrow,
hourly, the no-invented-percentages rule (precipitation inches only), span
fallback, and the honest-failure contract (HTTP errors and unreachable HA
come back as strings, never raises)."""

from datetime import datetime, timedelta

import httpx
import pytest

from aerys_v2.tools import weather
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
    out = tool().invoke({"span": "last month"})
    assert out.startswith("Right now:")


# 10/27: "he hears it'll be wet all week" -- the provider carries ~6 days of
# daily forecast and the tool stopped at tomorrow. "week" reads all of them.
def test_week_lists_every_forecast_day_with_its_weekday():
    out = tool().invoke({"span": "week"})
    assert out == (
        'Next 2 days: Sun 7/19 rainy, 76–88°, 0.42" precipitation; '
        "Mon 7/20 sunny, 75–91°"
    )


def test_week_phrasings_reach_the_week_span():
    for span in ("this week", "next week", "weekly", "WEEK"):
        assert tool().invoke({"span": span}).startswith("Next 2 days: ")


def test_week_never_invents_a_probability():
    assert "%" not in tool().invoke({"span": "week"})


def test_week_with_empty_forecast_is_reported_not_fabricated():
    out = tool(daily=[]).invoke({"span": "week"})
    assert out == "The weather provider returned no daily forecast."


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


# The weekend span (Chris, Python lesson 2, 10/31): the real forecast HA gave on Fri 10/30.
FRI_TO_WED = [
    {"datetime": "2026-10-30T16:00:00+00:00", "condition": "partlycloudy", "temperature": 80, "templow": 76, "precipitation": 0.0},
    {"datetime": "2026-10-31T16:00:00+00:00", "condition": "rainy", "temperature": 85, "templow": 74, "precipitation": 0.11},
    {"datetime": "2026-11-01T17:00:00+00:00", "condition": "rainy", "temperature": 81, "templow": 71, "precipitation": 0.83},
    {"datetime": "2026-11-02T17:00:00+00:00", "condition": "rainy", "temperature": 78, "templow": 70, "precipitation": 0.39},
    {"datetime": "2026-11-03T17:00:00+00:00", "condition": "rainy", "temperature": 84, "templow": 70, "precipitation": 0.03},
    {"datetime": "2026-11-04T17:00:00+00:00", "condition": "sunny", "temperature": 84, "templow": 71, "precipitation": 0.0},
]
WEEKEND = ('This weekend: Sat 10/31 rainy, 74–85°, 0.11" precipitation; '
           'Sun 11/1 rainy, 71–81°, 0.83" precipitation')


def test_weekend_lists_only_saturday_and_sunday():
    assert tool(daily=FRI_TO_WED).invoke({"span": "weekend"}) == WEEKEND


def test_weekend_is_checked_before_week():
    # "week" is inside "weekend": every weekend phrasing must reach the weekend, not the week list.
    for span in ("weekend", "this weekend", "Weekend", "WEEKEND", "next weekend"):
        out = tool(daily=FRI_TO_WED).invoke({"span": span})
        assert out == WEEKEND, span
        assert "Next 6 days" not in out and "Fri 10/30" not in out


def test_week_still_lists_every_day_with_a_weekend_in_it():
    assert tool(daily=FRI_TO_WED).invoke({"span": "week"}).startswith(
        "Next 6 days: Fri 10/30 partlycloudy, 76–80°; Sat 10/31 rainy")


def test_weekend_says_so_when_the_forecast_stops_before_it():
    mon_to_fri = FRI_TO_WED[3:5]  # Mon 11/2, Tue 11/3: no Saturday or Sunday
    assert tool(daily=mon_to_fri).invoke({"span": "weekend"}) == "The forecast doesn't reach this weekend yet."


def test_weekend_skips_an_odd_row_instead_of_failing():
    odd = [{"datetime": "garbage", "condition": "rainy"}, *FRI_TO_WED]
    assert tool(daily=odd).invoke({"span": "weekend"}) == WEEKEND


def test_weekend_with_empty_forecast_is_reported_not_fabricated():
    assert tool(daily=[]).invoke({"span": "weekend"}) == "The weather provider returned no daily forecast."


# NWS fixtures use symbolic points and endpoint names, never actual locations.
@pytest.fixture
def nws():
    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 10, 30, 15, tzinfo=tz)

    periods = []
    for offset, name in enumerate(("This Afternoon", "Tonight", "Saturday", "Saturday Night",
                                   "Sunday", "Sunday Night", "Monday", "Monday Night")):
        start = datetime(2026, 10, 30) + timedelta(days=offset // 2)
        periods.append({"name": name, "startTime": f"{start:%Y-%m-%d}T{12 if offset % 2 == 0 else 18}:00:00-04:00",
                        "probabilityOfPrecipitation": {"value": None if offset == 3 else offset * 10},
                        "shortForecast": "Cloudy"})
    hours = [{"startTime": f"2026-10-30T{15 + i:02}:00:00-04:00",
              "probabilityOfPrecipitation": {"value": i}, "shortForecast": "Cloudy"}
             for i in range(9)]
    calls = []

    def fetch(url):
        calls.append(url)
        if "/points/" in url:
            return {"properties": {"forecast": "https://example.invalid/daily",
                                   "forecastHourly": "https://example.invalid/hourly"}}
        return {"properties": {"periods": hours if url.endswith("hourly") else periods}}

    return Clock, fetch, calls


def nws_tool(fetch, **kwargs):
    return build_weather_tool(base_url="http://ha:8123", token="t", entity_id=ENTITY,
                              client=fake_ha(**kwargs), nws_point="LAT,LON", fetch_json=fetch)


@pytest.mark.parametrize("span,expected", [
    ("now", "This Afternoon 0% (Cloudy); Tonight 10% (Cloudy)"),
    ("today", "This Afternoon 0% (Cloudy); Tonight 10% (Cloudy)"),
    ("tomorrow", "Saturday 20% (Cloudy); Saturday Night no figure (Cloudy)"),
    ("weekend", "Saturday 20%; Saturday Night no figure; Sunday 40%; Sunday Night 50%"),
    ("week", "This Afternoon 0%; Tonight 10%; Saturday 20%; Saturday Night no figure; Sunday 40%; Sunday Night 50%; Monday 60%; Monday Night 70%"),
    ("hourly", "15:00 0%; 16:00 1%; 17:00 2%; 18:00 3%; 19:00 4%; 20:00 5%; 21:00 6%; 22:00 7%"),
])
def test_nws_spans_preserve_ha_and_select_periods(monkeypatch, nws, span, expected):
    clock, fetch, _ = nws
    monkeypatch.setattr(weather, "datetime", clock)
    assert nws_tool(fetch).invoke({"span": span}) == (
        tool().invoke({"span": span}) + "\nRain chance (National Weather Service): " + expected)


def test_nws_points_cached_across_tools_but_forecasts_fresh(nws):
    _, fetch, calls = nws
    for _ in range(2):
        nws_tool(fetch).invoke({"span": "now"})
    assert calls.count("https://api.weather.gov/points/LAT,LON") == 1
    assert calls.count("https://example.invalid/daily") == 2


@pytest.mark.parametrize("body", [None, {}, {"properties": {"periods": []}},
                                  {"properties": {"periods": [{"startTime": "bad"}]}}])
def test_nws_failure_is_honest_and_preserves_ha(body):
    def fetch(url):
        if body is None:
            raise httpx.ConnectError("offline")
        if "/points/" in url:
            return {"properties": {"forecast": "https://example.invalid/daily",
                                   "forecastHourly": "https://example.invalid/hourly"}}
        return body
    assert nws_tool(fetch).invoke({"span": "now"}) == (
        tool().invoke({"span": "now"}) + "\nRain chance: not available right now.")


def test_no_point_never_fetches_nws():
    def fetch(url):
        pytest.fail("NWS must not be fetched without a point")
    t = build_weather_tool(base_url="http://ha:8123", token="t", entity_id=ENTITY,
                           client=fake_ha(), fetch_json=fetch)
    for span in weather._SPANS:
        assert t.invoke({"span": span}) == tool().invoke({"span": span})


def test_nws_default_fetch_sets_headers_and_checks_status(monkeypatch):
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(503)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    monkeypatch.setattr(weather.httpx, "get", lambda url, **kw: client.get(
        url, headers=kw["headers"], timeout=kw["timeout"], follow_redirects=kw["follow_redirects"]))
    with pytest.raises(httpx.HTTPStatusError):
        weather._fetch_nws_json("https://example.invalid/daily")
    assert "aerys-v2" in requests[0].headers["User-Agent"]
    assert requests[0].headers["Accept"] == "application/geo+json"
    assert "Authorization" not in requests[0].headers


def test_nws_point_failure_retries_and_ha_failure_still_has_chance(nws):
    _, fetch, calls = nws
    failed = False

    def flaky(url):
        nonlocal failed
        if not failed:
            failed = True
            raise httpx.ConnectError("offline")
        return fetch(url)

    t = nws_tool(flaky, status=503)
    assert t.invoke({"span": "now"}) == (
        "Weather read failed: HA returned 503.\nRain chance: not available right now.")
    assert "Tonight 10% (Cloudy)" in t.invoke({"span": "now"})
    assert calls.count("https://api.weather.gov/points/LAT,LON") == 1


def test_tomorrow_when_first_period_is_tonight(monkeypatch, nws):
    clock, fetch, _ = nws
    monkeypatch.setattr(weather, "datetime", clock)

    def evening(url):
        body = fetch(url)
        if url.endswith("daily"):
            body["properties"]["periods"] = body["properties"]["periods"][1:]
        return body

    out = nws_tool(evening).invoke({"span": "tomorrow"})
    assert out.endswith("Saturday 20% (Cloudy); Saturday Night no figure (Cloudy)")
