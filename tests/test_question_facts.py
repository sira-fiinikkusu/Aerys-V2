"""A question never becomes a fact, and for now neither does a fact said inside one.

Status quo, on purpose (2026-11-25). Earlier in this batch question-shaped owner
messages were let through to the extraction model, so that "I just bought a new truck,
what oil should I use?" could keep the truck. Two review rounds showed that judging
"is this clause stated, or asked?" with punctuation and word lists keeps chasing cases
both ways: it stored "Maybe, my birthday is April 12" and refused "I bought a truck and
maybe you can help with the oil". So that change was taken out. Question-shaped
messages are dropped before the model again, and the truck is lost. It comes back
later as a model judgment ("does his message state this as true?") with the decision
in code; when it does, the first test here flips on purpose.

What stays from that work: when the model's own output is question-shaped, the refusal
is counted (stats["rejected"]["question"]) instead of vanishing, and a run of passes
that loses most of what the model proposed reaches the desk (see test_source_quote.py).

All fixtures are synthetic.
"""
import json

import pytest

from aerys_v2.workers import extraction as ex
from test_extraction import FakeConn, FakeLlm, fake_embedder, lease_conn, row
from test_memory_hygiene import PERSON

TRUCK_MESSAGE = "I just bought a new truck, what oil should I use?"
TRUCK_STATEMENT = "I just bought a new truck."
TRUCK_QUOTE = "I just bought a new truck"


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


def test_a_fact_inside_a_question_is_lost_for_now():
    """The accepted status quo: the message never reaches the model."""
    summary, inserts, llm = live(TRUCK_MESSAGE, [obs("Just bought a new truck", TRUCK_QUOTE)])
    assert llm.calls == [] and inserts == []
    assert summary["sources"]["prod_chat"]["rows"] == 1      # read, and the line still moves


def test_the_same_words_without_the_question_reach_the_model_and_are_kept():
    summary, inserts, llm = live(TRUCK_STATEMENT, [obs("Just bought a new truck", TRUCK_QUOTE)])
    assert len(llm.calls) == 1 and len(inserts) == 1


@pytest.mark.parametrize("observation", [
    obs("What oil should he use?", TRUCK_QUOTE),                                   # a question as the value
    obs("Just bought a new truck", TRUCK_QUOTE, context="which oil should he use"),  # as the context
    obs("Just bought a new truck", TRUCK_QUOTE + ", what oil should I use?"),      # as the quote
])
def test_a_question_shaped_observation_is_refused_and_counted(observation):
    """Codex review round 1: this refusal used to leave observations, skipped and
    unquoted all at 0, invisible to anyone reading the pass summary."""
    summary, inserts, _ = live(TRUCK_STATEMENT, [observation])
    assert inserts == []
    assert summary["sources"]["prod_chat"]["rejected"] == {"question": 1}
