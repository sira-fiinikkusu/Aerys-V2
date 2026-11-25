"""No quote, no memory: a mined fact must quote the speaker's own words.

The extraction model is asked for `source_text`, the words a fact came from. Until
2026-11-24 a quote that was not in the messages was simply dropped to None and the
fact inserted anyway; the quote only fed the "at least as specific" check. So a
fact the model made up, or remembered from somewhere else, landed with nothing he
ever said behind it. A thin memory is worse than none.

Now a mined fact whose quote is not found in one of the messages the model was
shown is skipped, counted (`unquoted`, and `skipped` on the live path) and logged by
its key only. The match forgives what copying changes: case, runs of whitespace,
edge punctuation, curly quotes, the transcript's own "[Speaker]: " prefix. It does
not forgive changed words. The explicit remember tool is not affected: when he says
"remember that...", his words are the source.

All fixtures are synthetic.
"""
import json
import logging
from unittest.mock import Mock

import pytest

from aerys_v2.tools.remember import CURRENT_TURN_TEXT, build_remember_tool
from aerys_v2.workers import extraction as ex
from test_extraction import FakeConn, FakeLlm, fake_embedder, lease_conn, row
from test_memory_hygiene import PERSON

MESSAGE = "Big news: I adopted a gray tabby named Pickle last week!"
VALUE = "Adopted a gray tabby cat named Pickle"


def obs(source, *, value=VALUE, key="relationship.pet"):
    return {"key_label": key, "value_text": value, "context": "", "source_text": source,
            "event_date": "", "privacy_level": "private"}


def live(messages, reply, *, platform="discord"):
    rows = []
    for i, text in enumerate(messages):
        r = list(row(PERSON, text, id=str(i + 1)))
        r[3] = platform
        rows.append(tuple(r))
    source_sql = "FROM v2_turns" if platform == "portable" else "FROM n8n_chat_histories"
    prod = FakeConn([] if platform == "portable" else [(source_sql, rows)])
    staging = lease_conn(holder="brain", routes=[(source_sql, rows)] if platform == "portable" else [])
    prod_write = FakeConn([("FROM memories", [])])
    summary = ex.run_live_extraction(prod, staging, prod_write, FakeLlm([json.dumps(reply)]), fake_embedder)
    inserts = [p for s, p in prod_write.calls if s.lstrip().upper().startswith("INSERT")]
    return summary, inserts


def stats_of(summary):
    return summary["sources"]["prod_chat"]


def test_a_quoted_fact_is_kept():
    summary, inserts = live([MESSAGE], [obs("I adopted a gray tabby named Pickle last week")])
    assert summary["inserted_total"] == 1 and len(inserts) == 1
    assert stats_of(summary)["unquoted"] == 0


def test_a_fact_whose_quote_is_not_in_his_messages_is_skipped_counted_and_logged_by_key(caplog):
    caplog.set_level(logging.WARNING)
    made_up = "I adopted a black kitten named Biscuit"
    summary, inserts = live([MESSAGE], [obs(made_up, value="Adopted a black kitten named Biscuit")])
    assert inserts == [] and summary["inserted_total"] == 0
    assert stats_of(summary)["unquoted"] == 1
    assert stats_of(summary)["skipped"] == 1 and summary["skipped_total"] == 1
    assert "relationship.pet" in caplog.text and "quote" in caplog.text
    assert "Biscuit" not in caplog.text and "kitten" not in caplog.text   # the key, never the fact


@pytest.mark.parametrize("source", [None, "", "   ", "...", "\u201c\u201d"])
def test_a_missing_or_empty_quote_is_not_found(source):
    summary, inserts = live([MESSAGE], [obs(source)])
    assert inserts == [] and stats_of(summary)["unquoted"] == 1


@pytest.mark.parametrize("source", [
    "i adopted a GRAY tabby named pickle last week",          # case
    "I adopted  a gray\ntabby   named Pickle last week",     # whitespace
    "\"I adopted a gray tabby named Pickle last week.\"",    # edge punctuation and wrapping quotes
    "I adopted a gray tabby named Pickle last week!!",
    "[Unknown]: I adopted a gray tabby named Pickle",        # the transcript's own line prefix
])
def test_a_near_miss_quote_still_matches(source):
    summary, inserts = live([MESSAGE], [obs(source)])
    assert len(inserts) == 1 and stats_of(summary)["unquoted"] == 0


def test_curly_apostrophes_match_straight_ones():
    summary, inserts = live(["I'm allergic to shellfish."],
                            [obs("I\u2019m allergic to shellfish", value="Allergic to shellfish",
                                 key="basic.allergy")])
    assert len(inserts) == 1


@pytest.mark.parametrize("source", [
    "I adopted a gray tabby named Pickles last week",   # a changed word is a different quote
    "I adopted a tabby named Pickle",                    # words dropped from the middle
    "named Pickle last week! She sleeps all day",       # runs across two messages
])
def test_a_changed_or_stitched_quote_does_not_match(source):
    summary, inserts = live([MESSAGE, "She sleeps all day"], [obs(source)])
    assert inserts == [] and stats_of(summary)["unquoted"] == 1


def test_a_quote_matches_whole_words_only():
    summary, inserts = live(["My trucks are loud"], [obs("My truck", value="Owns a truck", key="user.vehicle")])
    assert inserts == []


def test_shadow_staging_holds_the_same_line():
    prod = FakeConn([("FROM n8n_chat_histories", [row(PERSON, MESSAGE)])])
    staging = FakeConn()
    summary = ex.run_extraction(prod, staging, FakeLlm([json.dumps([obs("I adopted a black kitten")])]),
                                fake_embedder)
    assert not any("v2_memories_staging" in s for s, _ in staging.calls)
    assert summary["sources"]["prod_chat"]["unquoted"] == 1
    assert any("v2_extraction_watermark" in s for s, _ in staging.calls)   # the line still moves


def test_an_unquoted_portable_fact_never_reaches_the_review_inbox(monkeypatch):
    hook = Mock()
    monkeypatch.setattr(ex, "portable_quarantine_hook", lambda: hook)
    summary, inserts = live([MESSAGE], [obs("I adopted a black kitten")], platform="portable")
    hook.assert_not_called()
    assert inserts == [] and summary["sources"]["v2_turns"]["unquoted"] == 1


def test_a_quoted_portable_fact_still_goes_to_the_review_inbox(monkeypatch):
    hook = Mock()
    monkeypatch.setattr(ex, "portable_quarantine_hook", lambda: hook)
    live([MESSAGE], [obs("I adopted a gray tabby named Pickle")], platform="portable")
    hook.assert_called_once()


def test_the_triage_writer_receives_the_quote_that_matched(monkeypatch):
    """Before, only an EXACT substring was passed on; a case-different quote became
    None and the fact went in as unsourced. Now a matched quote always rides along."""
    prod = FakeConn([("FROM n8n_chat_histories", [row(PERSON, MESSAGE)])])
    staging = lease_conn(holder="brain")
    prod_write = FakeConn([("FROM memories", [])])
    seen = {}
    real = ex.triage_memory

    def spy(conn, **kw):
        seen.update(kw)
        return real(conn, **kw)

    monkeypatch.setattr(ex, "triage_memory", spy)
    quote = "i adopted a gray tabby named pickle"   # case differs from the message
    ex.run_live_extraction(prod, staging, prod_write, FakeLlm([json.dumps([obs(quote)])]), fake_embedder)
    assert seen["source_text"] == quote and seen["mined_from_turn"] is True


def test_the_remember_tool_needs_no_quote():
    """When he tells her to remember something, his words ARE the source: the tool's
    write goes straight to triage_memory with no source_text and still lands."""
    conn = Mock()
    conn.execute.return_value.fetchone.return_value = None

    def writer(record):
        return ex.triage_memory(conn, person_id=record["person_id"], key_label=record["key_label"],
                                value_text=record["fact"], content=record["fact"], context=None,
                                event_date=None, embedding="[0.1]", source_platform="house",
                                privacy_level="private", created_at="2026-11-24T00:00:00+00:00")

    tool = build_remember_tool(writer, key_labeler=lambda fact: "relationship.pet")
    token = CURRENT_TURN_TEXT.set("remember that my cat is named Pickle")
    try:
        reply = tool.invoke({"fact": "My cat is named Pickle"},
                            config={"configurable": {"identity": {"user_id": PERSON}}})
    finally:
        CURRENT_TURN_TEXT.reset(token)
    assert reply.startswith("Kept:")
    assert any("INSERT INTO memories" in c.args[0] for c in conn.execute.call_args_list)


@pytest.mark.parametrize("quote,messages,found", [
    ("I adopted a gray tabby", [{"content": MESSAGE}], True),
    ("I adopted a gray tabby", [{"content": "unrelated"}, {"content": MESSAGE}], True),
    ("I adopted a gray tabby", [], False),
    ("tabby named", [{"content": None}], False),
])
def test_source_quote_found_unit(quote, messages, found):
    assert ex.source_quote_found(quote, messages) is found


def test_the_prompt_asks_for_an_exact_quote_and_says_what_happens_without_one():
    prompt = ex.EXTRACTION_SYSTEM_PROMPT
    assert "copied exactly from one message" in prompt
    assert "never paraphrase" in prompt
    assert "A fact you cannot quote this way is not kept" in prompt
