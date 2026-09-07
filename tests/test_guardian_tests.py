"""Test chatter must not become a person fact; all fixtures are synthetic."""
import json
from unittest.mock import Mock

import pytest

from aerys_v2.workers import extraction as ex
from aerys_v2.tools.remember import build_remember_tool
from test_a2a_memory import writer_with
from test_memory_hygiene import PERSON


PROBES = [
    "Can you confirm that you know my favorite drink",
    "Could you confirm you remember my birthday",
    "Do you remember what I said", "Do you recall my birthday", "Do you know the answer",
    "Memory test: favorite drink", "Recall test", "Testing your memory",
    "Testing the recall", "Just reply yes", "Just say no", "Just answer yes or no",
    "Without telling me what it is", "What do you know about me",
    "What do you remember about me", "hey", "HI!", "hello", "yo", "ping", "test",
    "hello world", "hey, hi!", "test test test",
]
FACTS = [
    "I have a blood test Friday", "the tests passed at work",
    "remember that my blood test is Friday",
    "Remember that I am testing your memory tomorrow",
    "I like the movie Test", "My job is testing medical devices",
    "Hello, I adopted a dog", "Hello world is my favorite program",
    "I said just reply yes during the meeting", "I run a memory test on hardware at work",
    "hey 世界", "world", "test results",
]


@pytest.mark.parametrize("text", PROBES)
def test_probe_source_never_reaches_model(text):
    assert ex.test_shaped(text)
    llm = Mock()
    observations, reply = ex._extract_group(llm, {"messages": [{"content": text}]}, {})
    assert observations == [] and reply == ex.LlmReply("[]", False)
    llm.assert_not_called()


@pytest.mark.parametrize("text", FACTS)
def test_person_statement_is_not_a_probe(text):
    assert not ex.test_shaped(text)


def parse(key, value, **extra):
    return ex.parse_observations(json.dumps([dict(key_label=key, value_text=value, **extra)]))


ARTIFACTS = [
    ("decision.testing_plan", "Completed memory testing; will stop recursive memory tests"),
    ("interest.ai_memory_testing", "Engaged in testing AI memory capabilities"),
    ("technical.uat", "Conducting UAT with the assistant"),
    ("technical.uat", "Conducting UAT with Claude"),
    ("decision.phase", "Completed memory testing phase"),
    ("preference.recall", "Likes to test memory recall with specific items"),
    ("emotional.testing_frustration", "Frustrated with testing the assistant's memory"),
    ("interest.memory_testing", "Engaged in evaluating recall capabilities"),
    ("technical.probing", "Probing the system"),
    ("work.task", "Running live verification of action graph"),
    ("work.task", "Testing assistant memory"),
]


@pytest.mark.parametrize("key,value", ARTIFACTS)
def test_model_cannot_launder_test_activity_into_person_fact(key, value):
    assert parse(key, value) == []
    conn = Mock()
    assert triage(conn, key, value, mined_from_turn=True) == "skipped"
    conn.execute.assert_not_called()


KEPT = [
    ("event.health", "Has a blood test Friday"),
    ("decision.health", "Chose a blood test Friday"),
    ("work.task", "The tests passed at work"),
    ("interest.testing", "Enjoys testing recipes for family dinners"),
    ("technical.testing", "Tests medical devices at work"),
    ("decision.health", "Scheduled a clinical memory test Friday"),
    ("work.task", "Completed hardware memory testing at work"),
    ("work.task", "Running live verification of action graph; works as a reliability engineer"),
    ("work.task", "Testing the assistant; prefers concise replies"),
    ("preference.memory_testing", "Prefers concise replies"),
]


def triage(conn, key, value, **extra):
    return ex.triage_memory(conn, person_id=PERSON, key_label=key, value_text=value,
        content=f"{key}: {value}", context=None, event_date=None, embedding="[0.1]",
        source_platform="portable", privacy_level="private", created_at="2026-09-07",
        **extra)


@pytest.mark.parametrize("key,value", KEPT)
def test_real_facts_survive_model_and_write_guards(key, value):
    assert len(parse(key, value)) == 1
    conn = Mock()
    conn.execute.return_value.fetchone.return_value = None
    assert triage(conn, key, value, mined_from_turn=True) == "insert"


def test_source_probe_cannot_be_rewritten_as_a_fact():
    assert parse("interest.beverage", "Likes hibiscus tea", source_text="Just reply yes") == []


def test_explicit_remember_source_survives_extraction():
    source = "remember that my blood test is Friday"
    llm = Mock(return_value=ex.LlmReply(json.dumps([dict(key_label="event.health",
        value_text="Has a blood test Friday", source_text=source)]), False))
    observations, _ = ex._extract_group(llm, {"messages": [{"content": source}]}, {})
    assert len(observations) == 1
    llm.assert_called_once()
    assert parse("interest.memory_testing", "Testing assistant memory",
                 source_text="remember that I am testing assistant memory")


@pytest.mark.parametrize("text", ["Memory test", "[Portable observation time: 2026-09-07] Testing your memory"])
def test_remote_session_note_source_is_filtered(text):
    conn = Mock()
    assert triage(conn, "session.remote_body.check", "Reviewed this session",
        source_text=text, category=["source:portable"]) == "skipped"
    conn.execute.assert_not_called()
    llm = Mock()
    assert ex._extract_group(llm, {"messages": [{"content": text}], "source_platform": "portable"}, {})[0] == []
    llm.assert_not_called()


def test_real_remote_session_note_survives():
    conn = Mock()
    conn.execute.return_value.fetchone.return_value = None
    assert triage(conn, "session.remote_body.check", "Started a ceramics class",
        source_text="I started a ceramics class", category=["source:portable"]) == "insert"


def test_remote_session_note_with_turn_as_value_is_filtered():
    conn = Mock()
    assert triage(conn, "session.remote_body.check", "Testing your memory",
        category=["source:portable"]) == "skipped"
    conn.execute.assert_not_called()


def test_mixed_batch_only_sends_person_fact_to_model():
    def model(system, user):
        assert "Testing your memory" not in user
        assert "I have a blood test Friday" in user
        return ex.LlmReply(json.dumps([dict(key_label="event.health",
            value_text="Has a blood test Friday", source_text="I have a blood test Friday")]), False)
    observations, reply = ex._extract_group(model, {"messages": [
        {"content": "Testing your memory"}, {"content": "I have a blood test Friday"},
    ]}, {})
    assert len(observations) == 1 and not reply.truncated


@pytest.mark.parametrize("live", [False, True])
def test_remote_probe_advances_watermark_without_embedding_or_writing(live, monkeypatch):
    from test_extraction import FakeConn, row
    source = FakeConn()
    raw = "2026-09-07 12:00:00+00"
    remote = list(row(PERSON, "[Portable observation time: 2026-09-07] Memory test", raw=raw))
    remote[3] = "portable"
    staging = FakeConn([("FROM v2_turns", [tuple(remote)])])
    llm, embed, hook = Mock(), Mock(), Mock()
    monkeypatch.setattr(ex, "ensure_lease_holder", lambda _: "brain")
    monkeypatch.setattr(ex, "acquire_write_mutex", lambda _: True)
    monkeypatch.setattr(ex, "portable_quarantine_hook", lambda: hook)
    prod = Mock()
    summary = (ex.run_live_extraction(source, staging, prod, llm, embed) if live
               else ex.run_extraction(source, staging, llm, embed))
    assert summary["sources"]["v2_turns"]["watermark"] == raw
    assert summary["inserted_total"] == 0
    assert any("INSERT INTO v2_extraction_watermark" in sql and params["raw"] == raw
               for sql, params in staging.calls)
    assert not any("INSERT INTO v2_memories_staging" in sql for sql, _ in staging.calls)
    llm.assert_not_called()
    embed.assert_not_called()
    hook.assert_not_called()
    prod.execute.assert_not_called()


@pytest.mark.parametrize("message,reply", [
    ("hey", "Got it."), ("hey", "I am here and ready"),
    ("Testing your memory", "I remember your favorite drink"),
    ("The article relay is complete", "Done with that."),
])
def test_operator_ping_or_short_reply_never_embeds_or_writes(message, reply):
    log = []
    embed = Mock()
    writer_with(log, embed=embed)("kael:checkin", message, reply)
    assert log == []
    embed.assert_not_called()


def test_real_article_relay_is_written():
    log = []
    writer_with(log)("kael:relay", "Relaying the user's question about realtor articles",
        "You relayed the question about realtor articles; I found three useful sources.")
    assert log[-1] == ("commit", None)
    assert "realtor articles" in log[0][1]["content"]


@pytest.mark.parametrize("fact,key", [
    ("remember that my blood test is Friday", "event.health"),
    ("Testing assistant memory", "interest.memory_testing"),
])
def test_explicit_remember_writer_remains_unguarded(fact, key):
    conn = Mock()
    conn.execute.return_value.fetchone.return_value = None
    tool = build_remember_tool(lambda record: triage(conn, record["key_label"], record["fact"]),
                               key_labeler=lambda _: key)
    assert tool.invoke({"fact": fact}, {"configurable": {"identity": {"user_id": PERSON}}}).startswith("Kept:")
    assert any("INSERT INTO memories" in c.args[0] for c in conn.execute.call_args_list)


def test_probe_behind_a_greeting_and_name_is_still_a_probe():
    from aerys_v2.workers.extraction import test_shaped
    live = ("hey Aerys, without telling me what it is, can you confirm that you know what car i drive "
            "and the nickname for it? just reply with yes or no, dont tell me what it is")
    assert test_shaped(live)
    assert test_shaped("Hey Aerys, do you remember what my favorite drink is?")
    assert not test_shaped("hey Aerys, my blood test is Friday, can you remind me Thursday night")
    assert not test_shaped("hey Aerys, remember that I have a test drive on Saturday")


def test_multi_word_address_and_event_probe():
    from aerys_v2.workers.extraction import test_shaped, _test_only_observation
    assert test_shaped("hey my friend, do you remember what my car is?")
    assert test_shaped("Aerys dear, can you confirm you know my birthday? yes or no")
    assert _test_only_observation("event.evaluation", "event.evaluation: Ran a memory test on the assistant today")
    assert not _test_only_observation("event.milestone", "event.milestone: Passed the driving test on Saturday")
