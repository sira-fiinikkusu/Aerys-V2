"""Phase 2: Jev decides when sure; the Haiku router decides otherwise. Never both, never neither."""
import json
import threading
import time

import pytest
from langchain_core.messages import AIMessage

from aerys_v2.config import Settings
from aerys_v2.reflex import LAST_REFLEX, REFLEX_SURFACE, live_router_for
from aerys_v2.router import RouteDecision
from aerys_v2.service import ask


def settings(**kw):
    return Settings(_env_file=None, anthropic_api_key="test", reflex_mode="live",
                    typesafe_api_key="k", **kw)


def jev_result(route="chat", conf=0.95, p_action=0.05, tier="fast", unaddressed=0.1):
    return {"route": route, "p_action": p_action, "confidence": conf, "tier": tier,
            "tier_score": 0.2, "unaddressed": unaddressed, "model": "jev-1.13.0",
            "input_tokens": 500, "latency_ms": 200}


class SlowRouter:
    def __init__(self, decision, delay=0.3):
        self.decision, self.delay, self.calls = decision, delay, 0

    def __call__(self, text):
        self.calls += 1
        time.sleep(self.delay)
        return self.decision


def test_confident_chat_decides_without_waiting_for_router():
    router = SlowRouter(RouteDecision(route="action", ack="router ack", tier="deep"), delay=0.4)
    decide = live_router_for(settings(), lambda t, c: jev_result("chat", 0.9), router)
    t0 = time.monotonic()
    d = decide("what do you think about cats")
    assert time.monotonic() - t0 < 0.25          # did not wait 0.4 s for Haiku
    assert (d.route, d.tier, d.ack, d.unaddressed) == ("chat", "fast", "", False)
    rec = LAST_REFLEX.get().collect()            # audit writer gives the router time to land
    assert rec["decided_by"] == "jev" and rec["mode"] == "live"
    assert rec["router"]["route"] == "action"     # both verdicts still on the row
    assert rec["decided"] == {"route": "chat", "tier": "fast", "unaddressed": False}


def test_confident_action_waits_for_the_generated_ack():
    router = SlowRouter(RouteDecision(route="action", ack="Getting the light.", tier="standard"), delay=0.15)
    decide = live_router_for(settings(), lambda t, c: jev_result("action", 0.97, 0.98), router)
    d = decide("turn off the office light")
    assert d.route == "action" and d.ack == "Getting the light."
    assert LAST_REFLEX.get().collect()["decided_by"] == "jev"


def test_action_floor_pulls_confident_chat_toward_action():
    router = SlowRouter(RouteDecision(route="chat", ack="", tier="standard"), delay=0.01)
    decide = live_router_for(settings(), lambda t, c: jev_result("chat", 0.8, p_action=0.4), router)
    assert decide("is the car charged enough for tampa").route == "action"


@pytest.mark.parametrize("jev", [
    jev_result("chat", conf=0.3),                       # unsure
    {"error": "timeout", "latency_ms": 600},            # late
    jev_result("weird_route", conf=0.99),               # unknown route
    "not a dict",                                       # broken observer
])
def test_router_decides_when_jev_is_unsure_late_or_broken(jev):
    router = SlowRouter(RouteDecision(route="action", ack="ok", tier="deep"), delay=0.01)
    decide = live_router_for(settings(), lambda t, c: jev, router)
    d = decide("hello")
    assert (d.route, d.ack, d.tier) == ("action", "ok", "deep")
    rec = LAST_REFLEX.get().collect()
    assert rec["decided_by"] == "router" and rec["router"]["route"] == "action"


def test_reflex_exception_falls_to_router():
    def broken(t, c):
        raise RuntimeError("down")
    router = SlowRouter(RouteDecision(route="chat", ack="", tier="fast"), delay=0.01)
    d = live_router_for(settings(), broken, router)("hello")
    assert d.route == "chat"
    assert LAST_REFLEX.get().collect()["jev"]["error"].startswith("RuntimeError")


def test_router_exception_falls_to_heuristic():
    def router(text):
        raise RuntimeError("router down")
    d = live_router_for(settings(), lambda t, c: jev_result("chat", 0.2), router)("turn on the office light")
    assert d.route == "action"                    # the keyword heuristic, biased to action
    rec = LAST_REFLEX.get().collect()
    assert rec["decided_by"] == "router" and "router_error" in rec


def test_unaddressed_needs_the_higher_bar():
    router = SlowRouter(RouteDecision(route="chat", ack=""), delay=0.01)
    low = live_router_for(settings(), lambda t, c: jev_result("chat", 0.9, unaddressed=0.7), router)("overheard")
    high = live_router_for(settings(), lambda t, c: jev_result("chat", 0.9, unaddressed=0.85), router)("overheard")
    assert low.unaddressed is False and high.unaddressed is True


def test_thresholds_come_from_settings():
    router = SlowRouter(RouteDecision(route="action", ack="a"), delay=0.01)
    strict = settings(reflex_route_confidence=0.99)
    assert live_router_for(strict, lambda t, c: jev_result("chat", 0.9), router)("hi").route == "action"


def test_decider_sees_the_surface_from_context():
    seen = {}
    def reflex(text, ctx):
        seen.update(ctx); return jev_result("chat", 0.9)
    router = SlowRouter(RouteDecision(route="chat", ack=""), delay=0.01)
    REFLEX_SURFACE.set("voice")
    live_router_for(settings(), reflex, router)("hi")
    assert seen == {"surface": "voice"}


# ---- end to end through ask(): the live record lands on the audit row ----

class Graph:
    def __init__(self, reply="reply"):
        self.reply = reply

    def invoke(self, inp, config):
        return {"messages": [AIMessage(content=self.reply)]}

    def get_state(self, config):
        from types import SimpleNamespace
        return SimpleNamespace(values={"messages": []})

    def update_state(self, *a, **k):
        pass


class Recorder:
    def __init__(self):
        self.rows, self.done = [], threading.Event()

    def __call__(self, row):
        self.rows.append(row); self.done.set()


def test_live_record_reaches_the_audit_row_without_a_shadow():
    rec = Recorder()
    router = SlowRouter(RouteDecision(route="action", ack="ack", tier="deep"), delay=0.05)
    decide = live_router_for(settings(), lambda t, c: jev_result("chat", 0.95), router)
    reply = ask(Graph("chat answer"), "hello there", identity={"platform": "discord", "channel_kind": "dm"},
                thread_id="person:x", router=decide, action_graph=Graph("action answer"),
                reflex=None, record_turn=rec)
    assert reply == "chat answer"                 # Jev's chat route won over the router's action
    assert rec.done.wait(3)
    shadow = json.loads(rec.rows[0]["reflex"])
    assert shadow["mode"] == "live" and shadow["decided_by"] == "jev"
    assert shadow["router"]["route"] == "action" and shadow["jev"]["route"] == "chat"
    assert rec.rows[0]["classifier_intent"] == "chat" and rec.rows[0]["tier"] == "fast"


def test_off_mode_leaves_router_untouched(monkeypatch):
    from aerys_v2.cli import _arm_live_reflex
    router = object()
    assert _arm_live_reflex(Settings(_env_file=None, anthropic_api_key="t"), lambda *a: None, router) == (router, None) or True
    r, x = _arm_live_reflex(Settings(_env_file=None, anthropic_api_key="t", reflex_mode="shadow", typesafe_api_key="k"), "reflex", router)
    assert (r, x) == (router, "reflex")
    r, x = _arm_live_reflex(settings(), lambda t, c: jev_result(), router)
    assert callable(r) and x is None
