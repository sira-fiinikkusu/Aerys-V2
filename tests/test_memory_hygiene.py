"""Regression fixtures use synthetic identities and dates."""
import json
from unittest.mock import Mock

import pytest

from aerys_v2.workers import extraction as ex
from aerys_v2.tools.remember import build_remember_tool, CURRENT_TURN_TEXT, NOT_KEPT

PERSON = '11111111-1111-4111-8111-111111111111'
QUESTION = "what is Rowan's favorite drink?"
BEVERAGE = "Rowan's favorite drink is iced hibiscus tea with lime"
BIRTHDAY = 'My birthday is April 12, 1990'


def triage(conn, value, **kw):
    return ex.triage_memory(conn, person_id=PERSON, key_label='interest.beverage',
        value_text=value, content=value, context=None, event_date=None,
        embedding='[0.1]', source_platform='house', privacy_level='private',
        created_at='2026-09-06T00:00:00+00:00', **kw)


@pytest.mark.parametrize('value', [QUESTION, 'Which drink does she like', 'Can she drink tea?'])
def test_question_never_replaces_july_beverage(value, caplog):
    conn = Mock()
    conn.execute.return_value.fetchone.return_value = ('july-memory', BEVERAGE)
    assert triage(conn, value) == 'skipped'
    conn.execute.assert_not_called()
    assert 'question' in caplog.text


def test_question_parse_skips_without_holding_watermark(caplog):
    assert ex.parse_observations(json.dumps([dict(key_label='interest.beverage', value_text=QUESTION)])) == []
    assert 'question' in caplog.text


def test_question_source_cannot_be_laundered_into_statement():
    assert ex.parse_observations(json.dumps([dict(key_label='interest.beverage',
        value_text='Likes drinks', source_text=QUESTION)])) == []


def test_weak_statement_preserves_specific_value_even_when_similar():
    conn = Mock()
    conn.execute.return_value.fetchone.return_value = ('july-memory', BEVERAGE)
    assert triage(conn, "Rowan's favorite drink is iced hibiscus tea", mined_from_turn=True) == 'skipped'
    assert conn.execute.call_count == 1


def test_provenance_records_source_message_id():
    conn = Mock()
    conn.execute.return_value.fetchone.return_value = None
    assert triage(conn, BEVERAGE, source_message_id='origin-turn') == 'insert'
    assert conn.execute.call_args.args[1]['source_message_id'] == 'origin-turn'
    assert 'source_message_id' in conn.execute.call_args.args[0]


def test_birthday_aliases_and_remember_share_one_key():
    llm = Mock(return_value=ex.LlmReply(json.dumps([dict(key_label='event.birthday', value_text=BIRTHDAY)]), False))
    key = ex.key_label_for(BIRTHDAY, llm=llm)
    assert key == 'basic.birth_date'
    parsed = ex.parse_observations(json.dumps([
        dict(key_label='basic.birth_date', value_text=BIRTHDAY),
        dict(key_label='event.birthday', value_text=BIRTHDAY)]))
    assert {o['key_label'] for o in parsed} == {key}


def test_tool_same_fact_twice_one_row_and_label_failure_is_honest():
    rows = {}
    def write(record):
        action = 'update' if record['key_label'] in rows else 'insert'
        rows[record['key_label']] = record['fact']
        return action
    labeler = Mock(return_value='basic.birth_date')
    tool = build_remember_tool(write, key_labeler=labeler)
    cfg = {'configurable': {'identity': {'user_id': PERSON}}}
    token = CURRENT_TURN_TEXT.set(BIRTHDAY)
    try:
        for _ in range(2):
            assert tool.invoke({'fact': BIRTHDAY}, config=cfg).startswith('Kept:')
        assert rows == {'basic.birth_date': BIRTHDAY}
        labeler.side_effect = ValueError('unlabeled')
        assert tool.invoke({'fact': BIRTHDAY}, config=cfg).startswith('Kept:')
    finally:
        CURRENT_TURN_TEXT.reset(token)


def test_birthday_tool_then_two_extractor_aliases_leave_one_database_row():
    import re
    import sqlite3
    database = sqlite3.connect(':memory:')
    database.execute('''CREATE TABLE memories (id INTEGER PRIMARY KEY,
        person_id TEXT, key_label TEXT, content TEXT, context TEXT, event_date TEXT,
        embedding TEXT, source_platform TEXT, privacy_level TEXT, created_at TEXT,
        updated_at TEXT, deleted_at TEXT)''')
    class Connection:
        def execute(self, sql, params):
            sql = re.sub(r'::(?:uuid|vector|timestamptz)', '', sql)
            sql = re.sub(r'%\((\w+)\)s', r':\1', sql).replace('now()', 'CURRENT_TIMESTAMP')
            return database.execute(sql, params)
    def write(record):
        return ex.triage_memory(Connection(), person_id=PERSON, key_label=record['key_label'],
            value_text=record['fact'], content=record['fact'], context=None, event_date=None,
            embedding='[0.1]', source_platform='house', privacy_level='private',
            created_at='2026-09-06T00:00:00+00:00')
    labeler = lambda fact: ex.key_label_for(fact, llm=lambda *args:
        ex.LlmReply(json.dumps([dict(key_label='event.birthday', value_text=fact)]), False))
    tool = build_remember_tool(write, key_labeler=labeler)
    for _ in range(2):
        assert tool.invoke({'fact': BIRTHDAY}, config={'configurable': {'identity': {'user_id': PERSON}}}).startswith('Kept:')
    observations = ex.parse_observations(json.dumps([
        dict(key_label='basic.birth_date', value_text=BIRTHDAY),
        dict(key_label='event.birthday', value_text=BIRTHDAY)]))
    for obs in observations:
        assert write(dict(key_label=obs['key_label'], fact=obs['value_text'])) == 'update'
    assert database.execute('SELECT key_label, content FROM memories WHERE deleted_at IS NULL').fetchall() == [('basic.birth_date', BIRTHDAY)]
    assert database.execute('SELECT COUNT(*) FROM memories').fetchone()[0] == 1
    database.close()


@pytest.mark.parametrize('source', [QUESTION, '[Portable observation time: 2026-09-06] ' + QUESTION])
def test_source_question_skipped_before_llm(source):
    llm = Mock()
    observations, reply = ex._extract_group(llm, {'messages': [{'content': source}]}, {})
    assert observations == [] and not reply.truncated
    llm.assert_not_called()


@pytest.mark.parametrize('value', ['Has two dogs', 'Was born on April 12, 1990', 'Likes iced tea'])
def test_declarative_fragments_are_not_questions(value):
    assert not ex.question_shaped(value)


@pytest.mark.parametrize('reply', [ex.LlmReply('[]', False), ex.LlmReply('not json', False),
    ex.LlmReply('[{"key_label":"basic.birth_date","value_text":"Born in April"}]', True),
    ex.LlmReply('[{"key_label":"remember.hash","value_text":"Born in April"}]', False)])
def test_key_labeler_never_falls_back_to_hash_on_failure(reply):
    with pytest.raises(ValueError):
        ex.key_label_for(BIRTHDAY, llm=lambda *args: reply)


def test_labeler_factory_uses_shared_extractor_contract(monkeypatch):
    from aerys_v2.config import Settings
    from aerys_v2.factory import memory_key_labeler_for
    from aerys_v2.workers.extraction import EXTRACTION_SYSTEM_PROMPT
    model = Mock()
    model.invoke.return_value = Mock(content=json.dumps([dict(key_label='event.birthday', value_text=BIRTHDAY)]),
                                    response_metadata={'stop_reason': 'end_turn'})
    monkeypatch.setattr('langchain_anthropic.ChatAnthropic', Mock(return_value=model))
    labeler = memory_key_labeler_for(Settings(_env_file=None, anthropic_api_key='offline-placeholder'))
    assert labeler(BIRTHDAY) == 'basic.birth_date'
    assert model.invoke.call_args.args[0][0] == ('system', EXTRACTION_SYSTEM_PROMPT)


def test_skipped_tool_write_never_claims_already_kept():
    tool = build_remember_tool(lambda record: 'skipped', key_labeler=lambda fact: 'interest.beverage')
    reply = tool.invoke({'fact': 'Likes drinks'}, config={'configurable': {'identity': {'user_id': PERSON}}})
    assert reply.startswith('Nothing was kept:')
