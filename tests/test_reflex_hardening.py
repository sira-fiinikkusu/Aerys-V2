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
    # Codex second pass (#7): signs with spaces, mixed candidates, any oversize number
    assert brightness_from_text("set it to +20%") is None
    assert brightness_from_text("set it to - 20%") is None
    assert brightness_from_text("20% or half") is None
    assert brightness_from_text("1001% or 40%") is None
    assert brightness_from_text("half, no quarter") is None


# 3. sensitive proxies on the switch domain
def test_sensitive_switch_proxies_are_denied_by_default_and_by_allowlist():
    for proxy in ("switch.front_door_lock", "switch.ev_charger", "switch.heater", "switch.front_lock",
                  "switch.deadbolt"):
        rec = record(target=proxy)
        assert plain_device_command(rec, "turn it on", settings(), CANARY + (proxy,)) is None
        assert "deny" in rec["device"]["direct"]["reason"]
    # token match: harmless ids that merely CONTAIN a deny word still go direct (#6)
    for harmless in ("switch.outdoor_lights", "switch.gateway_plug"):
        rec = record(target=harmless)
        assert plain_device_command(rec, "turn it on", settings(), CANARY + (harmless,)) is not None
    # precedence: the explicit allowlist beats the token deny list, never the domain rule
    rec = record(target="switch.heater")
    assert plain_device_command(rec, "turn it on", settings(reflex_direct_allow="switch.heater"), CANARY) is not None
    rec = record(target="lock.front")
    assert plain_device_command(rec, "unlock it", settings(reflex_direct_allow="lock.front", reflex_direct_domains="lock,switch"), CANARY + ("lock.front",)) is None
    assert "sensitive domain" in rec["device"]["direct"]["reason"]
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
        assert "malformed" in rec["device"]["direct"]["reason"]
    # Codex second pass (#9): a zero threshold must not admit a malformed score either
    rec = record(cmd=nan)
    assert plain_device_command(rec, "turn off the office light", settings(reflex_direct_command_floor=0.0), CANARY) is None


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


def test_answer_completed_after_the_deadline_is_refused_even_if_the_caller_woke_late():
    # Codex second pass (#8): wait(0) returns True once done — completion time decides.
    class Fast:
        def system_one(self, **kw):
            time.sleep(0.08)
            return SimpleNamespace(answers={
                "route": SimpleNamespace(choice="action", probabilities={"action": .9, "chat": .1}, confidence=.95),
                "tier": SimpleNamespace(score=1.0), "unaddressed": SimpleNamespace(noul=.0),
                "cancelled": SimpleNamespace(noul=.0)}, model="jev", usage=SimpleNamespace(input_tokens=1))

    client = ReflexClient(client=Fast(), timeout_s=0.02)
    real_wait = threading.Event.wait

    def slow_wait(self, timeout=None):  # the caller gets descheduled past the deadline
        time.sleep(0.12)
        return real_wait(self, 0)

    threading.Event.wait = slow_wait
    try:
        out = client("turn off the light", {})
    finally:
        threading.Event.wait = real_wait
    assert out["error"] == "timeout"


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
    # Codex second pass (#10): the start error is audited even when Jev decides the turn
    # (with the router sampled in — at sample 0 a confident turn never starts it)
    confident = lambda t, c: {**jev(t, c), "confidence": .95}  # noqa: E731
    live_router_for(settings(reflex_router_sample=1.0), confident, lambda t: RouteDecision(route="chat", ack=""))("hello there")
    rec = LAST_REFLEX.get().collect()
    assert rec["decided_by"] == "jev" and "router_error" in rec
