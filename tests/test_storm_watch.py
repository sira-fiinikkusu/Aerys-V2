"""Storm watch (owner ask 8/17) — outlook bands, official alerts, delivery.

The watcher's promise is calm and non-repeating: one nudge per band crossing,
one per new alert id, one per new named storm — and silence on any feed
failure. These tests pin the band ladder, the dedup/prune behavior, the
event-name matching (families vs csv override), the TWO text parser against a
realistic product sample, and the two-legs-never-couple delivery rule.
"""

import pytest

from aerys_v2.storm_watch import (
    DEFAULT_EVENT_EXACT,
    DEFAULT_EVENT_FAMILIES,
    StormWatcher,
    band_of,
    compose_legs,
    event_matches,
    parse_outlook_areas,
    start_storm_watch,
)

TWO_SAMPLE = """
Tropical Weather Outlook
NWS National Hurricane Center Miami FL
800 AM EDT Sun Aug 17 2025

For the North Atlantic...Caribbean Sea and the Gulf of America:

1. Eastern Tropical Atlantic:
A tropical wave located several hundred miles south-southwest of the
Cabo Verde Islands continues to produce disorganized showers. Environmental
conditions are expected to become more conducive.
* Formation chance through 48 hours...low...10 percent.
* Formation chance through 7 days...medium...60 percent.

2. Central Atlantic:
A weak area of low pressure is producing limited convection. Development
is not expected.
* Formation chance through 48 hours...low...near 0 percent.
* Formation chance through 7 days...low...20 percent.

Forecaster Example
"""


class Recorder:
    def __init__(self):
        self.sent = []

    def __call__(self, title, message, image_url):
        self.sent.append((title, message, image_url))


def alert_feature(alert_id, event, **props):
    return {
        "properties": {"id": alert_id, "event": event, "headline": f"{event} headline",
                       "severity": "Severe", "instruction": "Take cover.", **props}
    }


def make_watcher(deliver=None, *, event_csv="", alerts=None, outlook=None, storms=None):
    calls = {"alerts": alerts or {"features": []},
             "outlook": outlook or TWO_SAMPLE,
             "storms": storms or {"activeStorms": []}}

    def fetch_json(url):
        if "alerts" in url:
            v = calls["alerts"]
        else:
            v = calls["storms"]
        if isinstance(v, Exception):
            raise v
        return v

    def fetch_text(url):
        v = calls["outlook"]
        if isinstance(v, Exception):
            raise v
        return v

    w = StormWatcher(
        point="0.0,0.0", deliver=deliver or Recorder(), event_csv=event_csv,
        fetch_json=fetch_json, fetch_text=fetch_text, sleep_fn=lambda s: None,
    )
    return w, calls


# -- arming gate ----------------------------------------------------------

def test_arming_gate_requires_a_point():
    class S:
        storm_watch_latlon = ""
    assert start_storm_watch(S()) is None


# -- TWO text parsing ------------------------------------------------------

def test_parse_outlook_takes_seven_day_lines_not_48h():
    areas = parse_outlook_areas(TWO_SAMPLE)
    assert [pct for pct, _ in areas] == [60, 20]
    assert "tropical wave" in areas[0][1].lower()


def test_band_ladder():
    assert band_of(49) == 0
    assert band_of(50) == 50
    assert band_of(69) == 50
    assert band_of(70) == 70
    assert band_of(95) == 90


# -- outlook band crossings ------------------------------------------------

def outlook_at(pct):
    return TWO_SAMPLE.replace("medium...60 percent", f"medium...{pct} percent")


def test_outlook_nudges_on_band_crossings_only():
    rec = Recorder()
    w, calls = make_watcher(rec)

    calls["outlook"] = outlook_at(60)
    w.poll_outlook()
    assert len(rec.sent) == 1 and "60%" in rec.sent[0][1]

    calls["outlook"] = outlook_at(65)   # same band — silent
    w.poll_outlook()
    assert len(rec.sent) == 1

    calls["outlook"] = outlook_at(75)   # 70 band crossed
    w.poll_outlook()
    assert len(rec.sent) == 2

    calls["outlook"] = outlook_at(20)   # calm — resets the high-water
    w.poll_outlook()
    assert len(rec.sent) == 2

    calls["outlook"] = outlook_at(55)   # re-brewing system re-alerts
    w.poll_outlook()
    assert len(rec.sent) == 3


# -- named storms ----------------------------------------------------------

def test_new_atlantic_storm_nudges_once():
    rec = Recorder()
    w, calls = make_watcher(rec)
    calls["outlook"] = outlook_at(10)   # keep the outlook quiet
    calls["storms"] = {"activeStorms": [
        {"id": "al052026", "binNumber": "AT5", "name": "Erin",
         "classification": "HU", "intensity": "105"},
        {"id": "ep032026", "binNumber": "EP3", "name": "Pacific"},
    ]}
    w.poll_outlook()
    assert len(rec.sent) == 1
    assert "Erin" in rec.sent[0][0]
    w.poll_outlook()                    # same storm — silent
    assert len(rec.sent) == 1


# -- alert matching / dedup / prune ---------------------------------------

def test_default_event_set_families_and_exact():
    fams, exact = DEFAULT_EVENT_FAMILIES, DEFAULT_EVENT_EXACT
    assert event_matches("Hurricane Warning", fams, exact)
    assert event_matches("Tropical Storm Watch", fams, exact)
    assert event_matches("Storm Surge Warning", fams, exact)
    assert event_matches("Tornado Watch", fams, exact)
    assert event_matches("Extreme Wind Warning", fams, exact)
    assert event_matches("Flash Flood Warning", fams, exact)
    assert not event_matches("Rip Current Statement", fams, exact)
    assert not event_matches("Flood Advisory", fams, exact)


def test_alert_dedup_and_prune_across_polls():
    rec = Recorder()
    w, calls = make_watcher(rec)
    calls["alerts"] = {"features": [alert_feature("a1", "Hurricane Watch")]}
    w.poll_alerts()
    w.poll_alerts()                     # same alert — one delivery
    assert len(rec.sent) == 1
    assert rec.sent[0][0] == "⚠️ Hurricane Watch"
    assert "Take cover." in rec.sent[0][1]

    # a1 expires, then REISSUES under the same id after pruning: it is new
    # news again (the prune bounds memory; NWS reuses nothing within life).
    calls["alerts"] = {"features": []}
    w.poll_alerts()
    calls["alerts"] = {"features": [alert_feature("a1", "Hurricane Watch")]}
    w.poll_alerts()
    assert len(rec.sent) == 2


def test_event_csv_override_is_exact_only():
    rec = Recorder()
    w, calls = make_watcher(rec, event_csv="Red Flag Warning")
    calls["alerts"] = {"features": [
        alert_feature("a1", "Hurricane Warning"),   # not in the override
        alert_feature("a2", "Red Flag Warning"),
    ]}
    w.poll_alerts()
    assert len(rec.sent) == 1
    assert rec.sent[0][0] == "⚠️ Red Flag Warning"


# -- fail-open -------------------------------------------------------------

def test_feed_failures_deliver_nothing_and_do_not_raise():
    rec = Recorder()
    w, calls = make_watcher(rec)
    calls["alerts"] = RuntimeError("nws down")
    calls["outlook"] = RuntimeError("nhc down")
    calls["storms"] = RuntimeError("nhc down")
    w.poll_alerts()
    w.poll_outlook()
    assert rec.sent == []


def test_tick_runs_both_polls_at_startup_then_spaces_them():
    rec = Recorder()
    w, calls = make_watcher(rec)
    calls["alerts"] = {"features": [alert_feature("a1", "Tornado Warning")]}
    calls["outlook"] = outlook_at(90)
    w.tick()
    assert len(rec.sent) == 2           # both polls fired on the first tick
    calls["alerts"] = {"features": [alert_feature("a1", "Tornado Warning"),
                                    alert_feature("a2", "Hurricane Warning")]}
    w.tick()                            # tick 2 — neither countdown elapsed
    assert len(rec.sent) == 2


# -- delivery legs ---------------------------------------------------------

def test_delivery_legs_never_couple():
    landed = []

    def bad_leg(title, message, image_url):
        raise RuntimeError("discord down")

    def good_leg(title, message, image_url):
        landed.append(title)

    deliver = compose_legs(bad_leg, good_leg)
    deliver("t", "m", "i")              # must not raise
    assert landed == ["t"]

    deliver2 = compose_legs(good_leg, bad_leg)
    deliver2("t2", "m", "i")
    assert landed == ["t", "t2"]


# ---- durable seen-state (migration 009; the 8/27 six-deploy Dolly spam) ----


class FakeStore:
    def __init__(self, preloaded=None):
        self.state = preloaded or {"alerts": set(), "storms": set(), "outlook_hw": 0}
        self.storm_adds: list[str] = []

    def load(self):
        return {
            "alerts": set(self.state["alerts"]),
            "storms": set(self.state["storms"]),
            "outlook_hw": self.state["outlook_hw"],
        }

    def add_storm(self, sid):
        self.storm_adds.append(sid)
        self.state["storms"].add(sid)

    def replace_alerts(self, current):
        self.state["alerts"] = set(current)

    def set_outlook_high_water(self, band):
        self.state["outlook_hw"] = band


def _watcher_with_store(store, delivered):
    return StormWatcher(
        point="26.9,-82.3",
        deliver=lambda t, m, i: delivered.append(t),
        event_csv="",
        fetch_json=lambda url: {"activeStorms": [
            {"id": "al052026", "binNumber": "AT5", "name": "Dolly",
             "classification": "TS", "intensity": "45"},
        ]},
        fetch_text=lambda url: "",
        store=store,
    )


def test_new_storm_is_recorded_in_the_store():
    store, delivered = FakeStore(), []
    w = _watcher_with_store(store, delivered)
    w.poll_outlook()
    assert delivered  # announced once
    assert store.storm_adds == ["al052026"]


def test_restart_with_store_does_not_reannounce():
    # The 8/27 failure, pinned: same storm, fresh watcher (a "redeploy"),
    # but the store remembers — zero deliveries the second time around.
    store, delivered = FakeStore(), []
    _watcher_with_store(store, delivered).poll_outlook()
    assert len(delivered) == 1
    delivered2: list[str] = []
    _watcher_with_store(store, delivered2).poll_outlook()
    assert delivered2 == []


def test_no_store_keeps_todays_behavior():
    delivered: list[str] = []
    w = StormWatcher(
        point="26.9,-82.3",
        deliver=lambda t, m, i: delivered.append(t),
        event_csv="",
        fetch_json=lambda url: {"activeStorms": []},
        fetch_text=lambda url: "",
    )
    w.poll_outlook()
    assert delivered == []
