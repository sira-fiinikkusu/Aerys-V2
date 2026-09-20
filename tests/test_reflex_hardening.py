"""Codex adversarial review 2026-09-20 (five Jev commits): the seven should-fix findings, pinned."""
import math
import threading
import time
from types import SimpleNamespace

from aerys_v2.config import Settings
from aerys_v2.reflex import (LAST_REFLEX, STATE_MESSAGE_CHARS, ReflexClient, brightness_from_text,
                             live_router_for, plain_device_command)
from aerys_v2.router import RouteDecision


def settings(**kw):
    return Settings(_env_file=None, anthropic_api_key="test", reflex_mode="live",
                    typesafe_api_key="k", **kw)


def record(target="switch.office_light_1", action="turn_off", cmd=0.95, state=0.05, compound=0.05,
           tconf=0.9, aconf=0.9):
    return {"decided": {"route": "action"}, "decided_by": "jev",
            "device": {"is_device_command": cmd, "is_state_question": state, "is_compound": compound,
                       "device_target": {"choice": target, "confidence": tconf},
                       "device_action": {"choice": action, "confidence": aconf}}}


CANARY = ("switch.office_light_1", "switch.front_door_lock", "switch.ev_charger", "switch.heater")


# 1. truncated requests never execute
def test_request_longer_than_the_classified_excerpt_never_goes_direct():
    text = "turn off the office light " + ("x" * STATE_MESSAGE_CHARS) + " do not execute"
    rec = record()
    assert plain_device_command(rec, text, settings(), CANARY) is None
    assert "longer" in rec["device"]["direct"]["reason"]
    assert plain_device_command(record(), "turn off the office light", settings(), CANARY) is not None


# 2. brightness parsing: absolute only
def test_brightness_accepts_only_one_unambiguous_absolute_level():
    assert brightness_from_text("set the lamp to 40%") == 40
    assert brightness_from_text("lamp to 40 percent") == 40
    assert brightness_from_text("all the way up") == 100
    assert brightness_from_text("dim by half") is None
    assert brightness_from_text("from 80% to 20%") is None
    assert brightness_from_text("set it to -20%") is None
    assert brightness_from_text("set it to 1001%") is None
    assert brightness_from_text("a bit brighter, 60%") is None
    assert brightness_from_text("dim the lights") is None


# 3. sensitive proxies on the switch domain
def test_sensitive_switch_proxies_are_denied_by_default_and_by_allowlist():
    for proxy in ("switch.front_door_lock", "switch.ev_charger", "switch.heater"):
        rec = record(target=proxy)
        assert plain_device_command(rec, "turn it on", settings(), CANARY) is None
        assert "deny" in rec["device"]["direct"]["reason"]
    # an explicit allowlist refuses everything it does not name, even a plain light
    rec = record()
    assert plain_device_command(rec, "turn off the office light", settings(reflex_direct_allow="switch.other"), CANARY) is None
    assert "allowlist" in rec["device"]["direct"]["reason"]
    assert plain_device_command(record(), "turn off the office light", settings(reflex_direct_allow="switch.office_light_1"), CANARY)


# 4. malformed scores fail CLOSED
def test_non_finite_scores_refuse_instead_of_passing_every_gate():
    nan = float("nan")
    for rec in (record(cmd=nan), record(state=nan), record(compound=nan), record(tconf=nan),
                record(aconf=math.inf), record(cmd=1.7)):
        assert plain_device_command(rec, "turn off the office light", settings(), CANARY) is None
        assert rec["device"]["direct"]["ok"] is False


# 5. a late Jev answer can no longer overwrite a timeout
def test_late_answer_after_deadline_stays_a_timeout():
    release = threading.Event()

    class Slow:
        def system_one(self, **kw):
            release.wait(2)
            return SimpleNamespace(answers={
                "route": SimpleNamespace(choice="action", probabilities={"action": .9, "chat": .1}, confidence=.95),
                "tier": SimpleNamespace(score=1.0), "unaddressed": SimpleNamespace(noul=.0),
                "cancelled": SimpleNamespace(noul=.0)}, model="jev", usage=SimpleNamespace(input_tokens=1))

    client = ReflexClient(client=Slow(), timeout_s=0.05)
    out = client("turn off the light", {})
    release.set()
    time.sleep(0.05)
    assert out["error"] == "timeout" and "route" not in out


# 6. command-shaped text is never dropped as unaddressed, whatever Jev said
def test_confident_command_is_not_dropped_as_unaddressed():
    router = lambda t: RouteDecision(route="action", ack="ok")  # noqa: E731
    jev = lambda t, c: {"route": "action", "p_action": .9, "confidence": .95, "tier": "fast",  # noqa: E731
                        "tier_score": .2, "unaddressed": .99, "cancelled": .0, "model": "jev",
                        "input_tokens": 1, "latency_ms": 5}
    d = live_router_for(settings(), jev, router)("turn off the office lights")
    assert d.unaddressed is False
    d2 = live_router_for(settings(), jev, router)("Megan can you grab the towel")
    assert d2.unaddressed is True


# 7. an unstartable router thread falls back instead of escaping
def test_router_thread_start_failure_falls_back(monkeypatch):
    class Boom(threading.Thread):
        def start(self):
            raise RuntimeError("can't start new thread")

    monkeypatch.setattr("aerys_v2.reflex.threading.Thread", Boom)
    jev = lambda t, c: {"route": "chat", "p_action": .1, "confidence": .3, "tier": "fast",  # noqa: E731
                        "tier_score": .2, "unaddressed": .0, "cancelled": .0, "model": "jev",
                        "input_tokens": 1, "latency_ms": 5}
    d = live_router_for(settings(), jev, lambda t: RouteDecision(route="chat", ack=""))("hello there")
    assert d.route in ("chat", "action")
    rec = LAST_REFLEX.get().collect()
    assert rec["decided_by"] == "router" and "router_error" in rec
