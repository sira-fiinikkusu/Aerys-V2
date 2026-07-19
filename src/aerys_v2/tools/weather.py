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

import httpx
from langchain_core.tools import tool

log = logging.getLogger(__name__)

_SPANS = ("now", "today", "tomorrow", "hourly")


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
):
    """The armed get_weather tool. entity_id is the house's weather entity
    (settings.ha_weather_entity); client is injectable for tests."""
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

    @tool
    def get_weather(span: str = "now") -> str:
        """ALWAYS use this for ANY weather question about home or the local
        area — rain, forecast, temperature, wind, "should I bring a jacket".
        NEVER use search_web for local weather: web results resolve the town
        name wrong and return data for the wrong continent. This reads the
        house's own Home Assistant weather station/provider.

        span: "now" (current conditions), "today" (today's forecast),
        "tomorrow", or "hourly" (next hours, times in UTC).
        Note: precipitation is reported in inches; this provider has no
        rain-probability percentage — do not invent one.
        """
        want = (span or "now").strip().lower()
        if want not in _SPANS:
            want = "now"
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

    return get_weather
