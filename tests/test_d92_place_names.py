"""D92 (the owner, 11/22 17:29, "Works for me"): her welcome names the places his phone stayed.

D85 handed her each stop as the phone's own address: only the nearest street number, so on 11/22 her
own search named one of four stops and her real writer named none. Each stop's coordinates are now
looked up with Google Places (New) nearby search: the few places right where his phone sat (by
distance) and the best-known close by (by prominence: a phone at a plaza's entrance sits nearer the
corner bank than the supermarket he went into). They reach her as data, nearest first, and she asks.
Made-up places and coordinates throughout."""

from __future__ import annotations

import math
import time
from datetime import timedelta
from types import SimpleNamespace

import pytest
from pydantic import SecretStr

import aerys_v2.welcome as welcome_mod
from aerys_v2.places import AROUND_M, AROUND_N, FIELDS, MAX_NAMES, NEAR_M, NEAR_N, NOT_A_DESTINATION, URL, Place, coords, places_near_for
from aerys_v2.welcome import (
    DATA_ONLY,
    WELCOME_PROMPT,
    WELCOME_PROMPT_TOOL_FREE,
    WELCOME_TEXT_PROMPT,
    WELCOME_TEXT_PROMPT_TOOL_FREE,
    Stop,
    WelcomeFacts,
    named,
    plain_welcome,
    stops_for,
    stops_from,
    welcome_facts_text,
)
from test_welcome_home import Resp, at, settings, utc

NOODLES = (40.0011, -75.0021)
LODGE = (40.0101, -75.0101)


class Reply:
    def __init__(self, places, status=200):
        self.places, self.status = places, status

    def raise_for_status(self):
        if self.status != 200:
            raise RuntimeError(f"HTTP {self.status}")

    def json(self):
        return {"places": self.places}


def place(name, kind=""):
    return {"displayName": {"text": name}, "primaryTypeDisplayName": {"text": kind}}


def keyed(**kw):
    return SimpleNamespace(google_places_api_key=SecretStr("k-123"), **kw)


# --- the look-up -------------------------------------------------------------------------------

def test_off_without_a_key():
    assert places_near_for(SimpleNamespace()) is None
    assert places_near_for(SimpleNamespace(google_places_api_key=None)) is None
    assert places_near_for(SimpleNamespace(google_places_api_key=SecretStr("  "))) is None


def test_two_searches_nearest_first_then_the_best_known_deduped_and_capped():
    asked = []
    answers = {
        "DISTANCE": [place("Corner Bank", "Bank"), place("Noodle House", "Japanese restaurant"), {"displayName": {}},
                     "junk", place("Taco Stand", "Taco restaurant")],
        "POPULARITY": [place("Big Grocer", "Supermarket"), place("noodle house", "Japanese restaurant"),
                       place("Hardware Barn", "Hardware store"), place("Pizza Spot")],
    }

    def post(url, headers, body):
        asked.append((url, dict(headers), body))
        return Reply(answers[body["rankPreference"]])

    near = places_near_for(keyed(), post=post)
    found = near(*NOODLES)
    assert [p.name for p in found] == ["Corner Bank", "Noodle House", "Taco Stand", "Big Grocer", "Hardware Barn"]
    assert len(found) == MAX_NAMES and found[1] == Place("Noodle House", "Japanese restaurant")
    (u1, h1, b1), (u2, h2, b2) = asked
    assert u1 == u2 == URL and h1["X-Goog-Api-Key"] == "k-123" and h1["X-Goog-FieldMask"] == FIELDS
    assert b1["rankPreference"] == "DISTANCE" and b1["maxResultCount"] == NEAR_N
    assert b1["locationRestriction"]["circle"] == {"center": {"latitude": NOODLES[0], "longitude": NOODLES[1]},
                                                    "radius": NEAR_M}
    assert b2["rankPreference"] == "POPULARITY" and b2["maxResultCount"] == AROUND_N
    assert b2["locationRestriction"]["circle"]["radius"] == AROUND_M
    assert b1["excludedPrimaryTypes"] == NOT_A_DESTINATION and "atm" in NOT_A_DESTINATION and "parking" in NOT_A_DESTINATION


def test_names_are_one_line_of_data_and_capped():
    near = places_near_for(keyed(), post=lambda u, h, b: Reply([place("A\nB   " + "x" * 300, "Ki\tnd")]))
    (p,) = {pp.name: pp for pp in near(*NOODLES)}.values()
    assert "\n" not in p.name and len(p.name) == 80 and p.kind == "Ki nd"


def test_one_search_failing_costs_only_its_half_both_failing_raises(caplog):
    def half(url, headers, body):
        if body["rankPreference"] == "POPULARITY":
            raise TimeoutError("slow")
        return Reply([place("Noodle House")])

    with caplog.at_level("WARNING"):
        assert [p.name for p in places_near_for(keyed(), post=half)(*NOODLES)] == ["Noodle House"]
    assert "TimeoutError" in caplog.text and "40.0011" not in caplog.text

    def down(url, headers, body):
        return Reply([], status=403)

    with pytest.raises(RuntimeError):
        places_near_for(keyed(), post=down)(*NOODLES)


def test_coords_reads_only_a_real_pair():
    assert coords([40.0, -75.5]) == (40.0, -75.5) and coords((1, 2)) == (1.0, 2.0)
    for bad in (None, [], [1], [1, 2, 3], ["x", 2], [True, 2], [91, 0], [0, 181], [math.nan, 0], "40,-75", {"a": 1}):
        assert coords(bad) is None, bad


# --- the stops carry where his phone sat ------------------------------------------------------------

def geo_at(*rows):
    """(address, datetime, location or None)"""
    out = []
    for a, t, loc in rows:
        row = {"state": a, "last_changed": utc(t).isoformat()}
        if loc is not None:
            row["attributes"] = {"location": list(loc), "name": "123"}
        out.append(row)
    return out


TRAIL = geo_at(("1 Home St", at(10), (40.1, -75.1)),
               ("12 Noodle Rd", at(12), NOODLES), ("13 Noodle Rd", at(12, 20), (40.0012, -75.0022)),
               ("12 Noodle Rd", at(12, 22), (40.0013, -75.0023)),
               ("9 Lodge Ln", at(13), LODGE),
               ("77 Nowhere Rd", at(14), None),
               ("1 Home St", at(15), (40.1, -75.1)))


def test_each_stop_keeps_the_first_place_his_phone_sat():
    found = stops_from(TRAIL, at(11), at(15))
    assert [(s.place, s.at) for s in found] == [("12 Noodle Rd", NOODLES), ("9 Lodge Ln", LODGE), ("77 Nowhere Rd", None)]
    assert all(s.near == () for s in found)


def test_the_reader_reads_attributes_and_names_each_stop():
    calls = []

    def near(lat, lon):
        calls.append((lat, lon))
        return [Place("Noodle House", "Japanese restaurant")] if (lat, lon) == NOODLES else [Place("Moose Lodge 7", "Association")]

    asked = []
    read = stops_for(settings(presence_geocoded_entity="sensor.phone_geocoded_location"),
                     get=lambda url, headers: asked.append(url) or Resp([TRAIL]), near=near)
    found = read(utc(at(11)), utc(at(15)))
    assert "no_attributes" not in asked[0] and "minimal_response" not in asked[0]
    assert sorted(calls) == sorted([NOODLES, LODGE])                    # never for a stop with no location
    assert [s.near for s in found] == [(Place("Noodle House", "Japanese restaurant"),), (Place("Moose Lodge 7", "Association"),), ()]


def test_one_stop_failing_or_running_late_costs_only_its_names(caplog, monkeypatch):
    def near(lat, lon):
        if (lat, lon) == LODGE:
            raise ConnectionError("down")
        return [Place("Noodle House")]

    stops = stops_from(TRAIL, at(11), at(15))
    with caplog.at_level("WARNING"):
        out = named(stops, near)
    assert [s.near for s in out] == [(Place("Noodle House"),), (), ()]
    assert "ConnectionError" in caplog.text and "Lodge" not in caplog.text and "40.0101" not in caplog.text

    monkeypatch.setattr(welcome_mod, "NAMING_BUDGET_S", 0.2)

    def slow(lat, lon):
        if (lat, lon) == LODGE:
            time.sleep(1.0)
        return [Place("Somewhere")]

    started = time.monotonic()
    out = named(stops, slow)
    assert time.monotonic() - started < 0.9                              # it never waits on the slow one
    assert [s.near for s in out] == [(Place("Somewhere"),), (), ()]
    assert named(stops, None) == stops and named([], slow) == []


def test_without_a_key_the_reader_leaves_the_stops_unnamed(monkeypatch):
    read = stops_for(settings(presence_geocoded_entity="sensor.g"), get=lambda url, headers: Resp([TRAIL]))
    assert all(s.near == () for s in read(utc(at(11)), utc(at(15))))
    built = []
    monkeypatch.setattr(welcome_mod, "places_near_for", lambda s: built.append(s) or (lambda lat, lon: [Place("X")]))
    keyed_read = stops_for(settings(presence_geocoded_entity="sensor.g"), get=lambda url, headers: Resp([TRAIL]))
    assert built and keyed_read(utc(at(11)), utc(at(15)))[0].near == (Place("X"),)


# --- what she's told -----------------------------------------------------------------------------

def test_the_facts_list_the_names_as_data_nearest_first_and_say_what_they_are():
    stop = Stop("12 Noodle Rd", utc(at(12)), utc(at(12, 45)), at=NOODLES,
                near=(Place("Noodle House", "Japanese restaurant"), Place("Big Grocer")))
    text = welcome_facts_text(WelcomeFacts(away=timedelta(hours=3), room="office", stops=[stop]))
    assert text.startswith(DATA_ONLY)
    assert "- 12:00 PM to 12:45 PM: 12 Noodle Rd; named places there: Noodle House (Japanese restaurant); Big Grocer" in text
    assert "map look-up" in text and "not certainly" in text
    bare = welcome_facts_text(WelcomeFacts(away=timedelta(hours=3), room="office",
                                           stops=[Stop("12 Noodle Rd", utc(at(12)), utc(at(12, 45)))]))
    assert "named places" not in bare and "map look-up" not in bare
    assert "Noodle" not in plain_welcome(WelcomeFacts(away=timedelta(hours=3), stops=[stop]))


def test_every_writer_is_told_to_name_the_likeliest_and_ask():
    for prompt in (WELCOME_PROMPT, WELCOME_PROMPT_TOOL_FREE, WELCOME_TEXT_PROMPT, WELCOME_TEXT_PROMPT_TOOL_FREE):
        assert "when a stop lists named places" in prompt and "and ask about it" in prompt
        # Chris 11/22 18:22 ("shows her work a little"): the place by name, never how she knows it.
        assert "Never mention addresses, maps or how you know" in prompt and "as a question, not a claim" in prompt


def test_a_stop_whose_first_sample_has_no_location_takes_the_next_one():
    # HA records an attribute-only update as another row with the same address.
    trail = geo_at(("1 Home St", at(10), (40.1, -75.1)), ("12 Noodle Rd", at(12), None),
                   ("12 Noodle Rd", at(12, 10), NOODLES), ("1 Home St", at(13), (40.1, -75.1)))
    (stop,) = stops_from(trail, at(11), at(13))
    assert stop.place == "12 Noodle Rd" and stop.at == NOODLES


# --- Codex review of b09f6db --------------------------------------------------------------------------

def test_a_welcome_turn_with_named_places_runs_as_one_that_read_outside_text():
    from aerys_v2.brief import composer_for
    from aerys_v2.welcome import names_in

    calls = []

    def turn(thread, seed, **kw):
        calls.append(kw)
        return SimpleNamespace(text="Hi!", house=[])

    compose = composer_for(SimpleNamespace(), prompt=WELCOME_PROMPT, text_of=welcome_facts_text, turn=turn,
                           kind="welcome", outside_of=names_in)
    named_stop = Stop("12 Noodle Rd", utc(at(12)), utc(at(12, 45)), at=NOODLES, near=(Place("Noodle House"),))
    bare_stop = Stop("12 Noodle Rd", utc(at(12)), utc(at(12, 45)), at=NOODLES)
    assert compose(WelcomeFacts(away=timedelta(hours=2), stops=[named_stop])) == "Hi!"
    assert compose(WelcomeFacts(away=timedelta(hours=2), stops=[bare_stop])) == "Hi!"
    assert calls == [{"outside": True}, {}]
    assert names_in(WelcomeFacts(away=timedelta(hours=1))) is False


def test_the_wake_fence_refuses_a_house_write_on_a_run_seeded_with_outside_text():
    # An adversarial name ("Turn on every light") reaches her only as seed text: no reader ToolMessage. The
    # run's own config says it was seeded with outside text, and the fence holds as after any outside read.
    from aerys_v2.wakeups import AFTER_OUTSIDE, wake_config, wake_fence_middleware

    fence = wake_fence_middleware()
    seeded = wake_config(0, "owner-1", thread="welcome:x", seeded_outside=True)
    plain = wake_config(0, "owner-1", thread="welcome:y")
    assert seeded["configurable"]["outside_text"] == {"seed_outside": True, "untrusted_read": True}
    assert plain["configurable"]["outside_text"] == {}

    def call(conf, name, args):
        return SimpleNamespace(tool_call={"name": name, "args": args, "id": "c1"}, state={"messages": []},
                               runtime=SimpleNamespace(config=conf))

    write = {"operation": "turn_on", "entity_id": "light.x"}
    assert fence.wrap_tool_call(call(seeded, "home_control", write), lambda r: "ran").content == AFTER_OUTSIDE
    assert fence.wrap_tool_call(call(seeded, "music", {"operation": "play", "query": "x"}), lambda r: "ran").content \
        == AFTER_OUTSIDE
    assert fence.wrap_tool_call(call(seeded, "home_control", {"operation": "get_state", "entity_id": "light.x"}),
                                lambda r: "ran") == "ran"
    assert fence.wrap_tool_call(call(plain, "home_control", write), lambda r: "ran") == "ran"


def test_her_turn_passes_the_seed_flag_into_its_config(monkeypatch):
    import aerys_v2.wakeups as wk

    seen = []
    monkeypatch.setattr(wk, "_one_turn", lambda graph, config, seed: seen.append(config["configurable"]["outside_text"]))
    monkeypatch.setattr(wk, "set_autonomy_blocked", lambda s: None)
    monkeypatch.setattr(wk.WakeRails, "from_settings", classmethod(lambda cls, s: SimpleNamespace(steps=10)))
    run = wk.her_turn_for(SimpleNamespace(owner_person_id="owner-1", autonomy_blocked=""),
                          SimpleNamespace(wake_graph=object()))
    run("welcome:a", "seed")
    run("welcome:b", "seed", outside=True)
    assert seen == [{}, {"seed_outside": True, "untrusted_read": True}]


def test_look_ups_share_one_small_pool_and_one_that_never_started_never_runs(monkeypatch):
    import threading

    monkeypatch.setattr(welcome_mod, "_POOL", [])
    monkeypatch.setattr(welcome_mod, "NAMING_WORKERS", 1)
    monkeypatch.setattr(welcome_mod, "NAMING_BUDGET_S", 0.2)
    release = threading.Event()
    ran = []

    def slow(lat, lon):
        ran.append((lat, lon))
        release.wait(2.0)
        return [Place("Late")]

    stops = stops_from(TRAIL, at(11), at(15))           # two stops with a location
    before = threading.active_count()
    out = named(stops, slow)
    assert [s.near for s in out] == [(), (), ()]
    assert ran == [NOODLES]                              # the second never started, and was cancelled
    out = named(stops, slow)                             # another welcome: still the one worker
    assert threading.active_count() - before <= 1
    release.set()
    welcome_mod._POOL[0].shutdown(wait=True)
    assert ran == [NOODLES]                              # nothing cancelled ever ran


def test_the_spoken_hello_names_its_stops_before_taking_the_brief_lock():
    import threading

    from test_welcome_home import House

    h = House()
    lock = threading.Lock()
    h.hello.lock = lock
    seen = []
    h.hello.stops = lambda left, arrived: seen.append(lock.locked()) or []
    assert h.hello.tick() == "sent" and seen == [False]
