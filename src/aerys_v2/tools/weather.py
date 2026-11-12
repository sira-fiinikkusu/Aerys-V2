"""weather — the house's own forecast, never a web search.

Born from a receipt (2026-07-19): "is it gonna rain today?" went to search_web,
Tavily resolved "Rotonda" to Rotonda, SWITZERLAND, and the model — honesty
gates working — refused the 14.8°C garbage and had nothing better to offer.
Local weather is not a search problem: Home Assistant sits on the same action
graph with a weather entity configured for THIS house. This tool reads it.

Two HA doors, both read-only:
- current conditions: GET /api/states/{entity} (state = condition, attributes
  carry temperature/humidity/wind in the instance's configured units — this
  house is imperial, so °F and mph).
- forecasts: POST /api/services/weather/get_forecasts?return_response — the
  modern HA shape (forecast attributes on the state are long gone). NOTE from
  the HA-side rain automation build (2026-06-28): this provider reports
  precipitation in INCHES and gives NO precipitation_probability — so answers
  speak in condition + precipitation amount, never invented percentages.

Honest-failure contract matches home_control: every failure is a plain string
back to the model, never a raise (an exception inside a ToolNode kills the
whole action turn). Reads only — no canary, no outbox, nothing to allowlist.
"""

import logging
from datetime import date, datetime, timedelta
from functools import lru_cache
from typing import Callable

import httpx
from langchain_core.tools import tool

log = logging.getLogger(__name__)

_SPANS = ("now", "today", "tomorrow", "hourly", "week", "weekend")


_NWS_HEADERS = {
    "User-Agent": "aerys-v2 weather (personal home installation)",
    "Accept": "application/geo+json",
}


def _fetch_nws_json(url: str) -> dict:
    # 5 s, not storm watch's 15: this runs inside a (voice) turn, and a slow NWS must only cost the line.
    response = httpx.get(url, headers=_NWS_HEADERS, timeout=5.0, follow_redirects=True)
    response.raise_for_status()
    return response.json()


@lru_cache(maxsize=128)
def _nws_endpoints(point: str, fetch: Callable[[str], dict]) -> tuple[str, str]:
    """Cache successful point discovery across tool instances in this process."""
    properties = fetch(f"https://api.weather.gov/points/{point}")["properties"]
    daily, hourly = properties["forecast"], properties["forecastHourly"]
    if not isinstance(daily, str) or not daily or not isinstance(hourly, str) or not hourly:
        raise ValueError("Missing NWS forecast endpoints")
    return daily, hourly


def _rain_chance(point: str, span: str, fetch: Callable[[str], dict]) -> str:
    try:
        daily, hourly = _nws_endpoints(point, fetch)
        # Only point discovery is cached; forecasts are fetched anew.
        periods = fetch(hourly if span == "hourly" else daily)["properties"]["periods"]
        if span in ("now", "today"):
            periods = periods[:2]
        elif span == "hourly":
            periods = periods[:8]
        elif span == "tomorrow":
            zone = datetime.fromisoformat(periods[0]["startTime"]).tzinfo
            if zone is None:
                raise ValueError("Missing NWS local offset")
            tomorrow = datetime.now(zone).date() + timedelta(days=1)
            periods = [p for p in periods
                       if datetime.fromisoformat(p["startTime"]).date() == tomorrow]
        elif span == "weekend":
            periods = [p for p in periods
                       if datetime.fromisoformat(p["startTime"]).weekday() >= 5]
        if not periods:
            return "Rain chance: not available right now."
        pieces = []
        for period in periods:
            if span == "hourly":
                start = datetime.fromisoformat(period["startTime"])
                if start.tzinfo is None:
                    raise ValueError("Missing NWS local offset")
                label = start.strftime("%H:%M")
            else:
                label = period["name"]
            value = period["probabilityOfPrecipitation"]["value"]
            if value is not None and (isinstance(value, bool) or
                                     not isinstance(value, (int, float)) or not 0 <= value <= 100):
                raise ValueError("Invalid NWS probability")
            chance = "no figure" if value is None else f"{_fmt_num(value)}%"
            piece = f"{label} {chance}"
            if span in ("now", "today", "tomorrow"):
                piece += f" ({period['shortForecast']})"
            pieces.append(piece)
        return "Rain chance (National Weather Service): " + "; ".join(pieces)
    except Exception:
        # An NWS failure must never discard the HA answer or expose fetch errors.
        return "Rain chance: not available right now."


def _day_of(f: dict) -> date | None:
    """A forecast row's date, or None for an odd row (one must never sink a whole answer)."""
    try:
        return date.fromisoformat(str(f.get("datetime", ""))[:10])
    except ValueError:
        return None


def _fmt_num(v: object) -> str:
    if isinstance(v, float) and v == int(v):
        return str(int(v))
    return str(v)


def build_weather_tool(
    *,
    base_url: str,
    token: str,
    entity_id: str,
    client: httpx.Client | None = None,
    nws_point: str | None = None,
    fetch_json: Callable[[str], dict] | None = None,
):
    """The armed get_weather tool. entity_id is the house's weather entity
    (settings.ha_weather_entity); client and the NWS JSON fetch are injectable for tests."""
    http = client or httpx.Client(timeout=10.0)
    base = base_url.rstrip("/")
    headers = {"Authorization": f"Bearer {token}"}

    def _current() -> str:
        r = http.get(f"{base}/api/states/{entity_id}", headers=headers)
        r.raise_for_status()
        s = r.json()
        a = s.get("attributes", {})
        t_unit = a.get("temperature_unit", "°")
        parts = [f"Right now: {s.get('state', 'unknown')}"]
        if a.get("temperature") is not None:
            parts.append(f"{_fmt_num(a['temperature'])}{t_unit}")
        if a.get("humidity") is not None:
            parts.append(f"{_fmt_num(a['humidity'])}% humidity")
        if a.get("wind_speed") is not None:
            parts.append(
                f"wind {_fmt_num(a['wind_speed'])} {a.get('wind_speed_unit', '')}".strip()
            )
        return ", ".join(parts) + "."

    def _forecast(kind: str) -> list[dict]:
        r = http.post(
            f"{base}/api/services/weather/get_forecasts?return_response",
            headers=headers,
            json={"entity_id": entity_id, "type": kind},
        )
        r.raise_for_status()
        body = r.json()
        # REST wraps service data under service_response; tolerate both shapes.
        data = body.get("service_response", body) or {}
        return (data.get(entity_id) or {}).get("forecast") or []

    def _fmt_day(f: dict) -> str:
        bits = [str(f.get("condition", "unknown"))]
        if f.get("temperature") is not None:
            hi = _fmt_num(f["temperature"])
            lo = f.get("templow")
            bits.append(f"{_fmt_num(lo)}–{hi}°" if lo is not None else f"high {hi}°")
        precip = f.get("precipitation")
        if precip:  # inches; 0/None = nothing meaningful to report
            bits.append(f'{_fmt_num(precip)}" precipitation')
        return ", ".join(bits)

    def _fmt_hour(f: dict) -> str:
        # "2026-07-19T21:00:00+00:00" -> "21:00" (UTC; the model localizes)
        dt = str(f.get("datetime", ""))
        clock = dt[11:16] if len(dt) >= 16 else dt
        piece = f"{clock} {f.get('condition', '?')}"
        if f.get("temperature") is not None:
            piece += f" {_fmt_num(f['temperature'])}°"
        if f.get("precipitation"):
            piece += f' {_fmt_num(f["precipitation"])}"'
        return piece

    def _ha_weather(want: str) -> str:
        try:
            if want == "now":
                return _current()
            if want == "hourly":
                hours = _forecast("hourly")[:8]
                if not hours:
                    return "The weather provider returned no hourly forecast."
                return "Hourly (UTC): " + "; ".join(_fmt_hour(f) for f in hours)
            days = _forecast("daily")
            if not days:
                return "The weather provider returned no daily forecast."
            if want == "weekend":
                weekend = [(day, f) for f in days if (day := _day_of(f)) and day.weekday() >= 5]
                if not weekend:
                    return "The forecast doesn't reach this weekend yet."
                return "This weekend: " + "; ".join(f"{day:%a} {day.month}/{day.day} {_fmt_day(f)}" for day, f in weekend)
            if want == "week":
                entries = []
                for f in days:
                    raw = str(f.get("datetime", ""))[:10]
                    try:
                        day = date.fromisoformat(raw)
                        label = f"{day:%a} {day.month}/{day.day}"
                    except ValueError:  # one odd row must not sink the week
                        label = raw or "?"
                    entries.append(f"{label} " + _fmt_day(f))
                return f"Next {len(days)} days: " + "; ".join(entries)
            if want == "tomorrow":
                if len(days) < 2:
                    return "The weather provider returned no forecast for tomorrow."
                return "Tomorrow: " + _fmt_day(days[1])
            return "Today: " + _fmt_day(days[0]) + " (" + _current() + ")"
        except httpx.HTTPStatusError as e:
            return f"Weather read failed: HA returned {e.response.status_code}."
        except httpx.TransportError:
            return "Weather read failed: Home Assistant is unreachable right now."
        except Exception as e:  # honest string, never a raise inside ToolNode
            log.warning("get_weather failed", exc_info=True)
            return f"Weather read failed: {e}"

    @tool
    def get_weather(span: str = "now") -> str:
        """ALWAYS use this for ANY weather question about home or the local
        area — rain, forecast, temperature, wind, "should I bring a jacket".
        NEVER use search_web for local weather: web results resolve the town
        name wrong and return data for the wrong continent. This reads the
        house's own Home Assistant weather station/provider.

        span: "now" (current conditions), "today" (today's forecast),
        "tomorrow", "hourly" (next hours: HA times in UTC, NWS times in local time),
        "week" (every forecast day the provider has, about 6), or
        "weekend" (only the Saturday and Sunday in the forecast; it says so
        plainly when the forecast doesn't reach them).
        Note: the house's provider reports precipitation in inches only. The
        rain chance (percent per period) comes from the National Weather Service
        line; quote it as given. When it says not available or no figure, say
        there is no figure; never invent a percentage.
        """
        want = (span or "now").strip().lower()
        # "week" is inside "weekend": ask the narrower question first (Chris, lesson 2).
        if "weekend" in want:
            want = "weekend"
        elif "week" in want:
            want = "week"
        if want not in _SPANS:
            want = "now"
        answer = _ha_weather(want)
        point = (nws_point or "").strip()
        if point:
            answer += "\n" + _rain_chance(point, want, fetch_json or _fetch_nws_json)
        return answer

    return get_weather
