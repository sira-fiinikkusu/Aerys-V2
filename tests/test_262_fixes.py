"""#262 (Gemini full-repo review, 11/25): three fixes in live code.

(3) A model-chosen `replaces` never retires a taste unless it is the very same taste. The prompt already says
    "two tastes never share a key and one never replaces another"; with ~4x more tastes kept (D107) a mistaken
    replace could retire an unrelated one. A refused replace keeps both rows; nothing is lost.
(1) The stated-as-true check's said_on seam is detected from the signature, not caught: a real TypeError
    inside the check is never retried, and any failure of the check is 'unavailable' (keeps nothing).
(4) The Jev timeout threads run in a copy of the caller's context, so tracing spans stay attached.
"""

from __future__ import annotations

import contextvars
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from aerys_v2.workers.extraction import is_taste_key, quoted_observations, replace_allowed
from test_d97_her_words import _live, obs as d97_obs
from test_extraction import CHRIS, row

T0 = datetime(2026, 11, 20, 15, 0, tzinfo=timezone.utc)


# ── (3) tastes ─────────────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("new_key,old_key,allowed", [
    ("preference.hotel.rooftop_pool", "preference.hotel.air_conditioning", False),  # same subject, other thing
    ("preference.food.cilantro", "preference.hotel.rooftop_pool", False),           # unrelated taste
    ("preference.hotel.rooftop_pool", "preference.hotel.rooftop_pool", True),       # the same taste
    ("preference.coffee.oat_milk", "preference.favorite_drink", False),             # a taste never retires a favorite
    ("interest.beverage", "preference.hotel.rooftop_pool", False),                  # nor is retired by one
    ("preference.favorite_drink", "interest.beverage", True),                       # favorites: unchanged (Codex r2)
    ("interest.beverage", "preference.favorite_drink", True),
    ("work.role", "basic.job", True),                                               # other state keys: unchanged
    ("relationship.dad.visit", "relationship.family", True),
    ("event.concert", "event.concert", False),                                      # events never
    ("aerys.told.move", "basic.location", False),                                   # her words never
])
def test_replace_allowed(new_key, old_key, allowed):
    assert replace_allowed(new_key, old_key) is allowed


def test_a_taste_is_three_parts_under_preference():
    assert is_taste_key("preference.hotel.rooftop_pool")
    assert not is_taste_key("preference.favorite_drink")
    assert not is_taste_key("interest.hotel.pool") and not is_taste_key(None)


def test_a_taste_replacing_an_unrelated_taste_is_ignored_and_both_stay():
    summary, write = _live(
        [d97_obs("preference.food.cilantro", "Hates cilantro", "I hate cilantro", replaces="k1")],
        [("k1", "preference.hotel.rooftop_pool", "preference.hotel.rooftop_pool: Loves a rooftop pool")],
        [row(CHRIS, "I hate cilantro")])
    stats = summary["sources"]["prod_chat"]
    assert stats["replaces_ignored"] == 1 and stats.get("replaces_used", 0) == 0
    assert not [p for s, p in write.calls if "RETURNING id, key_label" in s]   # nothing retired


def test_the_same_taste_may_still_be_replaced():
    summary, write = _live(
        [d97_obs("preference.hotel.rooftop_pool", "No longer cares for rooftop pools",
                 "I don't care about rooftop pools anymore", replaces="k1")],
        [("k1", "preference.hotel.rooftop_pool", "preference.hotel.rooftop_pool: Loves a rooftop pool")],
        [row(CHRIS, "I don't care about rooftop pools anymore")])
    assert [p["old_id"] for s, p in write.calls if "RETURNING id, key_label" in s] == ["k1"]
    assert summary["sources"]["prod_chat"]["replaces_used"] == 1


# ── (1) the stated-as-true seam ───────────────────────────────────────────────────────────────────

RAM = "I just bought a Ram 1500, what oil should I use?"


def _group(text):
    msgs = [{"id": "0", "content": text, "reply": None, "created_at": T0}]
    return {"person_id": CHRIS, "messages": msgs, "privacy_level": "private", "latest": msgs[-1]}


def _obs():
    return [{"key_label": "vehicle.truck", "value_text": "Bought a Ram 1500", "source_text": "I just bought a Ram 1500",
             "context": "", "event_date": "", "privacy_level": "private", "replaces": ""}]


def test_a_check_that_takes_said_on_gets_the_date():
    seen = []

    def stated(message, facts, said_on=None):
        seen.append(said_on)
        return [0.97] * len(facts)

    assert [o["key_label"] for o in quoted_observations(_obs(), _group(RAM), {}, stated)] == ["vehicle.truck"]
    assert seen and seen[0] and "2026" in seen[0]


def test_a_seam_without_said_on_is_called_once_without_it():
    calls = []

    def stated(message, facts):
        calls.append(message)
        return [0.97] * len(facts)

    assert quoted_observations(_obs(), _group(RAM), {}, stated)
    assert calls == [RAM]


@pytest.mark.parametrize("error", [TypeError("bad operand inside the check"), RuntimeError("down")])
def test_a_failing_check_is_unavailable_never_retried_never_escapes(error):
    calls = []

    def stated(message, facts, said_on=None):
        calls.append(said_on)
        raise error

    stats = {}
    assert quoted_observations(_obs(), _group(RAM), stats, stated) == []
    assert len(calls) == 1 and stats["stated_unavailable"] == 1


# ── (4) tracing context ───────────────────────────────────────────────────────────────────────────

TRACE = contextvars.ContextVar("trace", default=None)


class _AnyNoul(dict):
    def __missing__(self, key):
        return SimpleNamespace(noul=0.1)


@pytest.mark.parametrize("make,args", [
    (lambda c: __import__("aerys_v2.their_words", fromlist=["x"]).TheirWordsCheck(c, timeout_s=2.0),
     ("Chris likes tea", "I like tea", "Chris")),
    (lambda c: __import__("aerys_v2.looking_back", fromlist=["x"]).LookingBackCheck(c, timeout_s=2.0),
     ("remember when we talked about the trip?",)),
    (lambda c: __import__("aerys_v2.addressed_check", fromlist=["x"]).AddressedCheck(c, timeout_s=2.0),
     ("turn the lights off", "Want the lights off?", "it's late")),
])
def test_each_jev_thread_sees_the_callers_context(make, args):
    seen = []

    class Client:
        def system_one(self, **_):
            seen.append(TRACE.get())
            return SimpleNamespace(answers=_AnyNoul())

    token = TRACE.set("span-1")
    try:
        make(Client())(*args)
    finally:
        TRACE.reset(token)
    assert seen == ["span-1"]


def test_the_real_stated_check_gets_the_date_through_quoted_observations():
    from aerys_v2.workers.extraction import jev_stated_check

    states = []

    class Client:
        def system_one(self, *, state, questions, model):
            states.append(state)
            return SimpleNamespace(answers={q: SimpleNamespace(noul=0.97) for q in questions})

    stats = {}
    kept = quoted_observations(_obs(), _group(RAM), stats, jev_stated_check("x", client=Client()))
    assert [o["key_label"] for o in kept] == ["vehicle.truck"]
    assert states and "2026" in (states[0].get("said_on") or "")
