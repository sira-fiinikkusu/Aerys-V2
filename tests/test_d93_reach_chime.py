"""D93 (the owner, 11/22 20:21): her unsolicited spoken lines open with HA's chime.

#253: the office Pebbles sleep after a quiet stretch and lost her first word. On 11/22 a plain line, a
silent lead-in and a near-silent hiss all clipped; HA's default chime then the line came through whole.
His ruling: "having her chime like that for every unsolicited poke from her (outside of DJing) is actually
fine. it signals to me that she is about to say something and i should pay attention". Answers in a
conversation and DJ breaks (their own Music Assistant path) stay without it."""

from __future__ import annotations

import ast
import pathlib
from datetime import datetime, timezone
from types import SimpleNamespace

import httpx
import pytest

import aerys_v2.factory as factory
from aerys_v2.config import Settings
from aerys_v2.factory import followup_router_for


@pytest.fixture
def posts(monkeypatch):
    sent = []

    def post(url, **kw):
        sent.append((url, kw.get("json")))
        return httpx.Response(200, request=httpx.Request("POST", url))

    def get(url, **kw):          # the keep-listening toggle: on
        return httpx.Response(200, json={"state": "on"}, request=httpx.Request("GET", url))

    monkeypatch.setattr(httpx, "post", post)
    monkeypatch.setattr(httpx, "get", get)
    return sent


def router(**kw):
    return followup_router_for(Settings(_env_file=None, anthropic_api_key="placeholder", ha_token="placeholder",
                                        ha_base_url="http://ha.example", ha_satellite_map="sat=assist_satellite.example",
                                        **kw))


def test_the_router_chimes_only_when_asked(posts):
    route = router()
    route("Heads up.", "sat")
    route("Heads up.", "sat", chime=True)
    assert [p[1]["preannounce"] for p in posts] == [False, True]
    assert all(p[0].endswith("/assist_satellite/announce") for p in posts)


def test_a_chimed_question_that_opens_the_mic_chimes_too(posts):
    router()("Want the porch light on?", "sat", chime=True)
    (url, body), = [p for p in posts if "start_conversation" in p[0]]
    assert body["preannounce"] is True and body["start_message"] == "Want the porch light on?"


def test_the_setting_is_on_by_default():
    assert Settings(_env_file=None, anthropic_api_key="x").reach_chime is True


class FakeRouter:
    def __init__(self):
        self.calls = []

    def __call__(self, settings):
        def route(text, device_id, **kw):
            self.calls.append((text, device_id, kw))
        return route


@pytest.mark.parametrize("on", [True, False])
def test_owner_reach_speaks_with_the_chime_the_setting_says(monkeypatch, on):
    from aerys_v2.kael_line import owner_reach_for

    fake = FakeRouter()
    monkeypatch.setattr(factory, "followup_router_for", fake)
    settings = SimpleNamespace(owner_person_id="owner-1", telegram_bot_token=None, discord_bot_token="t",
                               owner_quiet_start_hour=21, owner_quiet_end_hour=9, reach_chime=on)
    noon = datetime(2026, 11, 23, 17, 0, tzinfo=timezone.utc)
    reach = owner_reach_for(settings, lookup_surface=lambda s: ("device", "sat"), clock=lambda: noon,
                            hold_fn=lambda: None, wake_watch=None)
    assert reach("The laundry's done.")
    assert fake.calls == [("The laundry's done.", "sat", {"chime": on})]


def test_the_reach_desk_speaks_with_the_chime(monkeypatch):
    from aerys_v2.kael_line import reach_desk_for

    fake = FakeRouter()
    monkeypatch.setattr(factory, "followup_router_for", fake)
    settings = Settings(_env_file=None, anthropic_api_key="x", database_url="postgresql://u:p@db.invalid/aerys",
                        owner_person_id="6e6bcbed-03ef-4d17-95d2-89c467414335", discord_bot_token="t",
                        ha_token="h", ha_base_url="http://ha.example")
    desk = reach_desk_for(settings, None)
    assert desk is not None
    desk.deliverer.speak("Your 3 PM is in ten minutes.", "sat")
    assert fake.calls == [("Your 3 PM is in ten minutes.", "sat", {"chime": True})]


def test_every_unsolicited_caller_passes_the_chime_and_the_router_defaults_off():
    """The three paths she speaks on her own (Kael notes, owner reach, the reach desk) all pass chime; the
    router's own default stays off, so her answers in a conversation are unchanged."""
    src = pathlib.Path(__file__).resolve().parents[1] / "src" / "aerys_v2"
    tree = ast.parse((src / "kael_line.py").read_text())
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "route"]
    assert len(calls) == 3 and all(any(k.arg == "chime" for k in c.keywords) for c in calls)
    ftree = ast.parse((src / "factory.py").read_text())
    route = next(n for n in ast.walk(ftree) if isinstance(n, ast.FunctionDef) and n.name == "route"
                 and any(a.arg == "chime" for a in n.args.kwonlyargs))
    default = route.args.kw_defaults[[a.arg for a in route.args.kwonlyargs].index("chime")]
    assert isinstance(default, ast.Constant) and default.value is False


def test_dj_breaks_keep_their_own_no_chime_path():
    src = pathlib.Path(__file__).resolve().parents[1] / "src" / "aerys_v2" / "dj.py"
    text = src.read_text()
    assert '"use_pre_announce": False' in text and "followup_router_for" not in text
