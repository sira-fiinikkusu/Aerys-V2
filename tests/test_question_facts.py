"""A fact said inside a question is still a fact; the question never is.

Found comparing our extractor with another open-source agent's (2026-11-24): every
owner message shaped like a question was dropped BEFORE the extraction model saw it,
and the prompt rejected "requests for information" outright. So "I just bought a new
truck, what oil should I use?" lost the truck: the one thing in that message worth
keeping went out with the question around it.

The line now sits on what is EXTRACTED, not on what is said. Question-shaped turns
reach the model; the prompt tells it a request often reveals a fact while a question
alone never confirms one; and the existing checks on the extracted values
(parse_observations, triage_memory) still refuse any value, context or quote that is
itself a question. Probe-shaped turns ("do you remember my birthday") are still
dropped before the model, exactly as before.

Codex review round 1 found the hole in that: a quote can be his words without being
his claim. "Could my birthday be April 12?" contains "my birthday be April 12", which
is not itself question-shaped. So the quote's span is judged by the clause around it,
in his punctuation (extraction.source_quote_status): a clause that asks, supposes or
doubts is not asserted, while a clause stated beside a question still is.

All fixtures are synthetic.
"""
import json

import pytest

from aerys_v2.workers import extraction as ex
from test_extraction import FakeConn, FakeLlm, fake_embedder, lease_conn, row
from test_memory_hygiene import PERSON

TRUCK_MESSAGE = "I just bought a new truck, what oil should I use?"
TRUCK_QUOTE = "I just bought a new truck"
WEATHER = "what's the weather tomorrow?"


def obs(value, source, *, key="user.vehicle", context=""):
    return {"key_label": key, "value_text": value, "context": context, "source_text": source,
            "event_date": "", "privacy_level": "private"}


def live(message, reply):
    """One live pass over one owner message, with the model's reply canned."""
    prod = FakeConn([("FROM n8n_chat_histories", [row(PERSON, message)])])
    staging = lease_conn(holder="brain")
    prod_write = FakeConn([("FROM memories", [])])
    llm = FakeLlm([json.dumps(reply)])
    summary = ex.run_live_extraction(prod, staging, prod_write, llm, fake_embedder)
    inserts = [p for s, p in prod_write.calls if s.lstrip().upper().startswith("INSERT")]
    return summary, inserts, llm


def test_a_question_bearing_message_reaches_the_model():
    seen = []

    def model(system, user):
        seen.append(user)
        return ex.LlmReply("[]", False)

    observations, reply = ex._extract_group(model, {"messages": [{"content": TRUCK_MESSAGE}]}, {})
    assert observations == [] and not reply.truncated
    assert len(seen) == 1 and TRUCK_MESSAGE in seen[0]


def test_the_truck_inside_the_question_becomes_a_memory():
    summary, inserts, llm = live(TRUCK_MESSAGE, [obs("Just bought a new truck", TRUCK_QUOTE)])
    assert TRUCK_MESSAGE in llm.calls[0][1]
    assert summary["inserted_total"] == 1
    assert inserts[0]["content"].startswith("user.vehicle: Just bought a new truck")


def test_quoting_the_whole_question_still_loses_the_fact_and_says_so():
    """Why the prompt asks for only the stating words: the check on the extracted
    quote is unchanged, and a quote that carries the question mark is a question.
    Codex review round 1: that loss was invisible (observations, skipped and
    unquoted all 0). The parser's refusal is counted now, by reason."""
    summary, inserts, _ = live(TRUCK_MESSAGE, [obs("Just bought a new truck", TRUCK_MESSAGE)])
    assert summary["inserted_total"] == 0 and inserts == []
    assert summary["sources"]["prod_chat"]["rejected"] == {"question": 1}


def test_a_paraphrased_quote_is_counted_not_silent():
    summary, inserts, _ = live(TRUCK_MESSAGE, [obs("Just bought a new truck", "I just purchased a new truck")])
    assert inserts == [] and summary["sources"]["prod_chat"]["unquoted"] == 1


# A guessed answer is not a fact (Codex review round 1, the blocker): the quote's words
# are in the message, but inside a question or a supposition. Message, quote, value, key.
NOT_ASSERTED = [
    ("Could my birthday be April 12?", "my birthday be April 12", "Birthday is April 12", "basic.birth_date"),
    ("Is my birthday April 12?", "my birthday April 12", "Birthday is April 12", "basic.birth_date"),
    ("My birthday might be April 12", "My birthday might be April 12", "Birthday is April 12",
     "basic.birth_date"),
    ("Maybe my birthday is April 12", "my birthday is April 12", "Birthday is April 12", "basic.birth_date"),
    ("I wonder if my birthday is April 12", "my birthday is April 12", "Birthday is April 12",
     "basic.birth_date"),
    ("My birthday is April 12, right?", "My birthday is April 12", "Birthday is April 12", "basic.birth_date"),
    ("If I get the job, I'll move to Exampleville", "I'll move to Exampleville", "Moving to Exampleville",
     "event.move"),
    ("Tell me if the package came", "the package came", "The package arrived", "event.package"),
    (WEATHER, "the weather tomorrow", "Wants to know the weather tomorrow", "interest.weather"),
]


@pytest.mark.parametrize("message,quote,value,key", NOT_ASSERTED)
def test_a_quote_from_inside_a_question_or_supposition_never_becomes_a_memory(message, quote, value, key):
    summary, inserts, _ = live(message, [obs(value, quote, key=key)])
    stats = summary["sources"]["prod_chat"]
    assert inserts == [] and summary["inserted_total"] == 0
    assert stats["quote_in_question"] == 1 and stats["skipped"] == 1 and stats["unquoted"] == 0


# What a request reveals, or what is said beside a question, is still asserted.
ASSERTED = [
    (TRUCK_MESSAGE, TRUCK_QUOTE),
    ("I just bought a new truck, any tips?", TRUCK_QUOTE),
    ("I just bought a new truck\nthoughts on synthetic oil?", TRUCK_QUOTE),     # a line break ends it
    ("I just bought a new truck what oil should I use?", TRUCK_QUOTE),          # no comma at all
    ("Is the oil change due? I just bought a new truck", TRUCK_QUOTE),
    ("[Portable observation time: 2026-11-20T10:00:00Z] I just bought a new truck, any tips?", TRUCK_QUOTE),
    ("Can you remind me Thursday, my blood test is Friday", "my blood test is Friday"),
    ("I would like you to call me Captain from now on", "I would like you to call me Captain"),
    ("May 3 is the appointment with the vet", "May 3 is the appointment with the vet"),
    ("When I moved to Exampleville I got a dog", "I got a dog"),
    ("I only eat eggs if they're scrambled", "I only eat eggs if they're scrambled"),
    ("Is it true my truck is red? My truck is red, I checked.", "my truck is red"),   # asked, then said
]


@pytest.mark.parametrize("message,quote", ASSERTED)
def test_an_asserted_clause_is_kept_even_beside_a_question(message, quote):
    summary, inserts, _ = live(message, [obs("A fact he stated", quote, key="user.note")])
    assert len(inserts) == 1, summary["sources"]["prod_chat"]


@pytest.mark.parametrize("reply", [
    [],                                                                  # the model follows the prompt
    [obs(WEATHER, WEATHER, key="interest.weather")],                     # the question stored as a value
    [obs("Tomorrow will be sunny", WEATHER, key="event.weather")],       # an answer inferred from it
    [obs("Interested in the weather", "the weather tomorrow", key="interest.weather",
         context="what is the weather tomorrow")],                       # a question as the context
])
def test_a_pure_question_yields_nothing(reply):
    summary, inserts, llm = live(WEATHER, reply)
    assert len(llm.calls) == 1                     # it is asked now, not dropped unseen
    assert summary["inserted_total"] == 0 and inserts == []


def test_a_memory_probe_is_still_dropped_before_the_model():
    llm = FakeLlm([])
    observations, _ = ex._extract_group(
        llm, {"messages": [{"content": "do you remember what my favorite drink is?"}]}, {})
    assert observations == [] and llm.calls == []


def test_the_prompt_keeps_facts_a_request_reveals_and_never_answers_a_question():
    prompt = ex.EXTRACTION_SYSTEM_PROMPT
    assert "requests for information" not in prompt          # the blanket rejection is gone
    assert TRUCK_MESSAGE in prompt and TRUCK_QUOTE in prompt  # the worked example
    assert "a question never confirms a fact" in prompt.lower()
    assert "never infer" in prompt.lower()
    # the quote must stop before the question, or the unchanged quote check drops it
    assert "only the words that state the fact" in prompt
