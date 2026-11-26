"""Preferences (c), 2026-11-25: the extractor keeps a personal taste said in passing ("loved the
rooftop pool and the balcony view at our last hotel"), in his words, keyed by what it is about and
which aspect. A preference question ("suggest a hotel for my trip") can only use a taste that was
saved; before this rule only his preferences FOR Aerys and interests with signal were kept.

The taste-plus-request case rides the existing gates unchanged: the whole message ends in "?", so it
is question-shaped, and its taste is kept only through M1 (Jev says he states it as true). Without
the check it is dropped at the quote gate, as before."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone

from aerys_v2.workers import extraction as ex
from aerys_v2.workers.extraction import (
    EXTRACTION_SYSTEM_PROMPT as PROMPT, KEY_LABEL_PATTERN, is_state_key, quoted_observations,
)
from test_extraction import CHRIS, FakeLlm

T0 = datetime(2026, 11, 20, 15, 0, tzinfo=timezone.utc)
TASTE = "I loved the rooftop pool at our last hotel"
ASKING = TASTE + ", can you suggest a hotel for my trip?"


def extract_section():
    return PROMPT.split("## What to EXTRACT")[1].split("## What to REJECT")[0]


def key_rules():
    return PROMPT.split("## key_label rules:")[1].split("## source_text rules:")[0]


def test_tastes_said_in_passing_are_on_the_extract_list():
    rule = next(l for l in extract_section().splitlines() if l.startswith("- Personal tastes"))
    for phrase in ("even in passing", "what they like or dislike", "hotels", "food and drink",
                   "products and gear", "what they own or use", "what they tried and how it went"):
        assert phrase in rule


def test_a_taste_is_kept_in_his_words_never_as_a_rule_drawn_from_it():
    rule = next(l for l in extract_section().splitlines() if l.startswith("- Personal tastes"))
    assert "Keep what they said, not a rule drawn from it" in rule
    assert '"loved the rooftop pool and the balcony view at our last hotel"' in rule
    assert '"Loved the rooftop pool at their last hotel"' in rule and '"Loved the balcony view at their last hotel"' in rule
    assert 'never "Prefers hotels with rooftop pools"' in rule


def test_a_taste_is_only_one_he_states_never_from_a_question_or_from_aerys():
    rule = next(l for l in extract_section().splitlines() if l.startswith("- Personal tastes"))
    assert "only one they state themselves" in rule
    assert "never inferred from a question alone or from what Aerys said" in rule
    assert "A taste said in the same message as a request is still a statement worth keeping" in rule
    assert "quote only the stated part" in rule
    # the old refusal of question-derived observations still stands beside it
    assert "observations derived only from questions (never infer an answer)" in PROMPT


def test_a_taste_is_keyed_by_the_thing_itself_so_a_second_taste_does_not_replace_the_first():
    # preference.* is a state key: a different value under the SAME key replaces the old row (triage_memory).
    # Codex prefs r1: "amenities" would have let "disliked the noisy AC" retire "loved the rooftop pool". Each
    # taste gets its own key (the specific thing), and a compound statement is split, one taste per fact.
    rules = key_rules()
    assert "keyed by its subject and the specific thing it is about, never by the value" in rules
    assert "two tastes never share a key and one never replaces another" in rules
    assert "The same thing again (a changed opinion of it) keeps its key." in rules
    assert "A favorite or an interest keeps the key the rules above give it." in rules
    assert "one taste per fact" in PROMPT and "amenities" not in PROMPT
    for key in ("preference.hotel.rooftop_pool", "preference.hotel.balcony_view", "preference.hotel.air_conditioning",
                "preference.food.cilantro"):
        assert key in PROMPT
        assert re.fullmatch(KEY_LABEL_PATTERN, key) and is_state_key(key)


def test_two_tastes_about_one_hotel_both_survive_successive_writes():
    """Codex prefs r1 regression: compatible tastes must not replace each other in triage."""
    from aerys_v2.workers.extraction import triage_memory
    rows = {}

    class Conn:
        def execute(self, sql, params=None):
            head = sql.lstrip().upper()
            found = [(i, c) for i, (k, c) in rows.items() if params and k == params.get("key_label")]
            if head.startswith("INSERT"):
                rows[f"r{len(rows) + 1}"] = (params["key_label"], params["content"])
                return type("R", (), {"fetchone": lambda self: (f"r{len(rows)}",), "fetchall": lambda self: []})()
            return type("R", (), {"fetchone": lambda self: found[0] if found else None,
                                  "fetchall": lambda self: found})()
    for key, value in [("preference.hotel.rooftop_pool", "Loved the rooftop pool at their last hotel"),
                       ("preference.hotel.air_conditioning", "Disliked the noisy air conditioning at their last hotel")]:
        action = triage_memory(Conn(), person_id="p", key_label=key, value_text=value, content=f"{key}: {value}",
                               context=None, event_date=None, embedding="[0.1]", source_platform="discord",
                               privacy_level="public", created_at="2026-11-25 12:00:00+00")
        assert action == "insert", (key, action)
    assert sorted(k for k, _ in rows.values()) == ["preference.hotel.air_conditioning", "preference.hotel.rooftop_pool"]


# --- the taste-plus-request message through the existing gates ------------------------------------

def group(*texts, reply=None):
    msgs = [{"id": str(i), "content": t, "reply": reply, "created_at": T0} for i, t in enumerate(texts)]
    return {"person_id": CHRIS, "messages": msgs, "privacy_level": "private", "latest": msgs[-1]}


def taste(quote=TASTE):
    return {"key_label": "preference.hotel.rooftop_pool", "value_text": "Loved the rooftop pool at their last hotel",
            "source_text": quote, "context": "", "event_date": "", "privacy_level": "private", "replaces": ""}


def test_without_the_check_a_lone_taste_plus_request_never_reaches_the_model():
    assert ex.question_shaped(ASKING)
    llm = FakeLlm([json.dumps([taste()])])
    observations, _ = ex._extract_group(llm, group(ASKING), {}, asking_ok=False)
    assert observations == [] and llm.calls == []


def test_without_the_check_her_reply_still_brings_it_to_the_model_and_the_quote_gate_drops_it():
    # v2_turns rows carry her reply, so the model IS asked; the taste dies after it, at the quote gate.
    g = group(ASKING, reply="Here are a few hotels with rooftop pools.")
    llm = FakeLlm([json.dumps([taste()])])
    observations, _ = ex._extract_group(llm, g, {}, asking_ok=False)
    assert len(llm.calls) == 1 and "(asking; a fact he states inside may be quoted" in llm.calls[0][1]
    stats = {}
    assert quoted_observations(observations, g, stats, None) == [] and stats["unquoted"] == 1


def test_with_the_check_armed_the_model_sees_it_marked_as_asking():
    llm = FakeLlm([json.dumps([taste()])])
    observations, _ = ex._extract_group(llm, group(ASKING), {}, asking_ok=True)
    assert len(llm.calls) == 1 and "(asking; a fact he states inside may be quoted" in llm.calls[0][1]
    assert [o["key_label"] for o in observations] == ["preference.hotel.rooftop_pool"]


def test_without_the_check_the_quote_gate_drops_it_as_before():
    stats = {}
    assert quoted_observations([taste()], group(ASKING), stats, None) == []
    assert stats["unquoted"] == 1


def test_with_the_check_it_is_kept_when_jev_says_he_states_it():
    calls = []

    def stated(message, facts):
        calls.append((message, list(facts)))
        return [0.95 for _ in facts]

    stats = {}
    kept = quoted_observations([taste()], group(ASKING), stats, stated)
    assert [o["key_label"] for o in kept] == ["preference.hotel.rooftop_pool"]
    assert calls == [(ASKING, ["Loved the rooftop pool at their last hotel"])] and stats["stated_kept"] == 1


def test_a_quote_that_carries_the_request_is_never_kept():
    stats = {}
    assert quoted_observations([taste(ASKING)], group(ASKING), stats, lambda m, f: [0.99] * len(f)) == []
    assert stats["unquoted"] == 1
