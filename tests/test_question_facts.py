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


def test_quoting_the_whole_question_still_loses_the_fact():
    """Why the prompt asks for only the stating words: the check on the extracted
    quote is unchanged, and a quote that carries the question mark is a question."""
    summary, inserts, _ = live(TRUCK_MESSAGE, [obs("Just bought a new truck", TRUCK_MESSAGE)])
    assert summary["inserted_total"] == 0 and inserts == []


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


def test_known_gap_a_fragment_lifted_from_a_pure_question_is_kept():
    """PINNED GAP, not a goal (flagged to Chris 2026-11-24). A quote that is a
    fragment of a question ("the weather tomorrow") is not itself question-shaped,
    so nothing in code stops it; only the prompt does. A clause-level check would
    also refuse the facts a request reveals ("what oil does my 2019 pickup take?"
    says he has a 2019 pickup), so it is not built. If this starts failing because
    the gap was closed on purpose, update it."""
    _, inserts, _ = live(WEATHER, [obs("Wants to know the weather tomorrow", "the weather tomorrow",
                                       key="interest.weather")])
    assert len(inserts) == 1


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
