"""Offline tests for the Discord slash-command handlers (n8n 03-02 port).

No network, no database, no discord.py: a routed FakeConn (the test_extraction
pattern) records every (sql, params) call, and embedders are plain callables.
What's proven: each handler's reply strings, that every write is parameterized
(user text never lands in SQL), that forget/correct only touch live rows, that
a raising embedder means ZERO writes, and that correct() preserves the old
row's created_at/key_label/context/privacy_level.
"""

import json
from datetime import datetime, timezone

import pytest

from aerys_v2.transports.discord_commands import (
    correct,
    forget,
    recall,
    set_profile_name,
    status,
    tell,
)

PERSON = "6e6bcbed-03ef-4d17-95d2-89c467414335"
CREATED = datetime(2026, 1, 15, 12, 0, 0, tzinfo=timezone.utc)

# A found row for FIND_MEMORY_SQL:
# (id, content, key_label, context, privacy_level, created_at)
FOUND = ("mem-1", "Drives a Dodge Ram", "vehicle.car", "guild chat", "public", CREATED)


class FakeCursor:
    def __init__(self, rows):
        self.rows = rows

    def fetchone(self):
        return self.rows[0] if self.rows else None

    def fetchall(self):
        return self.rows


class FakeConn:
    """Duck-typed psycopg connection: routes by SQL substring, records every call."""

    def __init__(self, routes=()):
        self.routes = list(routes)  # [(sql_substring, rows), ...] first match wins
        self.calls = []  # [(sql, params), ...]

    def execute(self, sql, params=None):
        self.calls.append((sql, params))
        for needle, rows in self.routes:
            if needle in sql:
                return FakeCursor(rows)
        return FakeCursor([])

    # -- assertion helpers ----------------------------------------------------

    def calls_containing(self, needle):
        return [(sql, params) for sql, params in self.calls if needle in sql]

    def writes(self):
        """Every non-SELECT statement issued."""
        return [
            (sql, params)
            for sql, params in self.calls
            if not sql.lstrip().upper().startswith("SELECT")
        ]

    def audit_entries(self):
        return [
            (params["action"], json.loads(params["details"]))
            for sql, params in self.calls_containing("INSERT INTO audit_log")
        ]


def embedder(text):
    return [0.1, 0.2, 0.3]


def broken_embedder(text):
    raise RuntimeError("embedding API down")


# --- recall -------------------------------------------------------------------


def test_recall_empty_gives_the_honest_default():
    conn = FakeConn()
    assert recall(conn, PERSON) == "I don't have any stored memories about you yet."


def test_recall_lists_content_and_locks_private_rows():
    conn = FakeConn([("FROM memories", [
        ("Drives a Dodge Ram", "public"),
        ("Takes medication for ADHD", "private"),
    ])])
    reply = recall(conn, PERSON)
    assert reply.startswith("**What I remember about you:**")
    assert "• Drives a Dodge Ram" in reply
    assert "• Takes medication for ADHD \U0001f512" in reply
    # only the private row is marked
    assert "Dodge Ram \U0001f512" not in reply


def test_recall_query_is_live_rows_recency_ordered_capped_at_10():
    conn = FakeConn([("FROM memories", [("x", "public")])])
    recall(conn, PERSON)
    sql, params = conn.calls_containing("FROM memories")[0]
    assert "deleted_at IS NULL" in sql
    assert "COALESCE(updated_at, created_at) DESC" in sql
    assert "LIMIT 10" in sql
    assert params == {"person_id": PERSON}


# --- forget -------------------------------------------------------------------


def test_forget_not_found_names_the_search_and_writes_nothing():
    conn = FakeConn()  # search returns no row
    reply = forget(conn, PERSON, "unicorn ownership")
    assert "unicorn ownership" in reply
    assert conn.writes() == []


def test_forget_soft_deletes_audits_and_quotes_the_memory():
    conn = FakeConn([("ILIKE", [FOUND])])
    reply = forget(conn, PERSON, "dodge")
    assert "Drives a Dodge Ram" in reply

    # soft delete, not a hard DELETE, keyed by the found row's id AND the
    # person (defense-in-depth: the person boundary is structural, not
    # call-site-dependent — review finding 2026-07-11)
    (sql, params), = conn.calls_containing("SET deleted_at = now()")
    assert sql.lstrip().startswith("UPDATE memories")
    assert "person_id = %(person_id)s" in sql
    assert params == {"id": "mem-1", "person_id": PERSON}
    assert conn.calls_containing("DELETE FROM") == []

    assert conn.audit_entries() == [
        ("memory.forget",
         {"person_id": PERSON, "memory_id": "mem-1", "content": "Drives a Dodge Ram"})
    ]


def test_forget_search_hits_only_live_rows_of_that_person():
    conn = FakeConn([("ILIKE", [FOUND])])
    forget(conn, PERSON, "dodge")
    sql, params = conn.calls_containing("ILIKE")[0]
    assert "deleted_at IS NULL" in sql
    assert "LIMIT 1" in sql
    # The predicate itself, not just the param: psycopg tolerates unused dict
    # keys, so params alone can't prove the person filter exists (the cardinal
    # invariant — review finding 2026-07-11).
    assert "person_id = %(person_id)s" in sql
    assert params == {"person_id": PERSON, "pattern": "%dodge%"}


def test_forget_empty_fact_refuses_instead_of_matching_everything():
    # V1's '%%' pattern matched an arbitrary row — a destructive footgun.
    conn = FakeConn([("ILIKE", [FOUND])])
    reply = forget(conn, PERSON, "  ")
    assert "search" in reply.lower()
    assert conn.calls == []  # not even the SELECT ran


def test_forget_never_interpolates_user_text_into_sql():
    fact = "'; DROP TABLE memories; --"
    conn = FakeConn([("ILIKE", [FOUND])])
    forget(conn, PERSON, fact)
    for sql, _params in conn.calls:
        assert fact not in sql


# --- correct ------------------------------------------------------------------


def test_correct_not_found_writes_nothing():
    conn = FakeConn()
    reply = correct(conn, PERSON, "unicorn", "a horse", embedder)
    assert "unicorn" in reply
    assert conn.writes() == []


def test_correct_replaces_atomically_preserving_the_old_rows_identity():
    conn = FakeConn([("ILIKE", [FOUND])])
    reply = correct(conn, PERSON, "dodge", "Drives a Ford F-150", embedder)
    assert "Drives a Dodge Ram" in reply and "Drives a Ford F-150" in reply

    (sql, params), = conn.calls_containing("WITH soft_del AS")
    # one atomic statement: soft-delete CTE + insert together, person-scoped
    assert "UPDATE memories SET deleted_at = now()" in sql
    assert "person_id = %(person_id)s" in sql.split("INSERT INTO")[0]  # the CTE half
    assert "INSERT INTO memories" in sql
    assert "'discord'" in sql  # source_platform is the new surface
    assert params["old_id"] == "mem-1"
    assert params["content"] == "Drives a Ford F-150"
    # identity + history carried from the OLD row
    assert params["key_label"] == "vehicle.car"
    assert params["context"] == "guild chat"
    assert params["privacy_level"] == "public"
    assert params["created_at"] == CREATED  # NOT reset — "when this fact first landed"
    # embedding serialized the services/memory way, cast in-query
    assert params["embedding"] == "[0.1,0.2,0.3]"
    assert "%(embedding)s::vector" in sql

    assert conn.audit_entries() == [(
        "memory.correct",
        {"person_id": PERSON, "memory_id": "mem-1",
         "old_content": "Drives a Dodge Ram",
         "new_content": "Drives a Ford F-150"},
    )]


def test_correct_broken_embedder_means_zero_writes_and_an_honest_reply():
    conn = FakeConn([("ILIKE", [FOUND])])
    reply = correct(conn, PERSON, "dodge", "Drives a Ford", broken_embedder)
    assert "couldn't store that right now" in reply.lower()
    assert conn.writes() == []  # the search SELECT ran; nothing else


def test_correct_embeds_the_new_value_not_the_search_term():
    seen = []

    def spy(text):
        seen.append(text)
        return [0.5]

    conn = FakeConn([("ILIKE", [FOUND])])
    correct(conn, PERSON, "dodge", "Drives a Ford F-150", spy)
    assert seen == ["Drives a Ford F-150"]


def test_correct_empty_fact_refuses():
    conn = FakeConn([("ILIKE", [FOUND])])
    reply = correct(conn, PERSON, "", "anything", embedder)
    assert conn.calls == []
    assert "search" in reply.lower()


def test_correct_db_errors_raise_to_the_binding_layer():
    class Boom(FakeConn):
        def execute(self, sql, params=None):
            if "WITH soft_del" in sql:
                raise RuntimeError("NAS is down")
            return super().execute(sql, params)

    conn = Boom([("ILIKE", [FOUND])])
    with pytest.raises(RuntimeError):
        correct(conn, PERSON, "dodge", "Drives a Ford", embedder)


# --- tell ---------------------------------------------------------------------


def test_tell_inserts_a_user_stated_memory_and_audits():
    conn = FakeConn()
    reply = tell(conn, PERSON, "My favorite color is teal", embedder, "public")
    assert "My favorite color is teal" in reply

    (sql, params), = conn.calls_containing("INSERT INTO memories")
    assert "'discord'" in sql
    assert "now(), now()" in sql  # created_at = now(): the statement just happened
    assert params["person_id"] == PERSON
    assert params["content"] == "My favorite color is teal"
    assert params["privacy_level"] == "public"
    assert params["embedding"] == "[0.1,0.2,0.3]"
    assert "%(embedding)s::vector" in sql
    # Unique suffix per tell: a bare 'user.stated' collides with the live-
    # uniqueness index (person_id, key_label) WHERE deleted_at IS NULL and the
    # command would work exactly once per person (review finding 2026-07-11).
    assert params["key_label"].startswith("user.stated.")
    assert len(params["key_label"]) > len("user.stated.")

    (action, details), = conn.audit_entries()
    assert action == "memory.tell"
    assert details["person_id"] == PERSON
    assert details["content"] == "My favorite color is teal"
    assert details["privacy_level"] == "public"
    assert details["key_label"] == params["key_label"]


def test_tell_twice_uses_distinct_key_labels():
    conn = FakeConn()
    tell(conn, PERSON, "fact one", embedder, "public")
    tell(conn, PERSON, "fact two", embedder, "public")
    inserts = conn.calls_containing("INSERT INTO memories")
    labels = [params["key_label"] for _sql, params in inserts]
    assert len(labels) == 2 and labels[0] != labels[1]


def test_tell_empty_fact_refuses():
    conn = FakeConn()
    reply = tell(conn, PERSON, "   ", embedder, "public")
    assert conn.calls == []
    assert "empty" in reply.lower()


def test_tell_privacy_level_is_the_callers_call():
    # 'private' from a DM, 'public' from a guild room — passed through verbatim.
    conn = FakeConn()
    tell(conn, PERSON, "secret", embedder, "private")
    (_sql, params), = conn.calls_containing("INSERT INTO memories")
    assert params["privacy_level"] == "private"


def test_tell_broken_embedder_means_zero_writes():
    conn = FakeConn()
    reply = tell(conn, PERSON, "a fact", broken_embedder, "public")
    assert "couldn't store that right now" in reply.lower()
    assert conn.calls == []


# --- status -------------------------------------------------------------------


def test_status_formats_name_and_linked_platforms():
    conn = FakeConn([("FROM platform_identities", [
        ("discord", "123456789", "chris_p", "Chris"),
        ("telegram", "987654", None, "Chris"),
    ])])
    reply = status(conn, PERSON)
    # username preferred, platform_user_id as the fallback (the n8n format)
    assert reply == (
        "**Identity:** Chris\n**Linked platforms:** discord (chris_p), telegram (987654)"
    )
    _sql, params = conn.calls[0]
    assert params == {"person_id": PERSON}


def test_status_with_no_rows_degrades_to_unknown_and_none():
    conn = FakeConn()
    assert status(conn, PERSON) == "**Identity:** Unknown\n**Linked platforms:** none"


# --- set_profile_name -----------------------------------------------------------


def test_set_profile_name_updates_persons_and_confirms():
    conn = FakeConn()
    reply = set_profile_name(conn, PERSON, "Captain")
    assert "Captain" in reply

    (sql, params), = conn.calls_containing("UPDATE persons")
    assert "SET display_name" in sql
    assert params == {"name": "Captain", "person_id": PERSON}

    assert conn.audit_entries() == [
        ("profile.set_name", {"person_id": PERSON, "display_name": "Captain"})
    ]


def test_set_profile_name_empty_refuses():
    conn = FakeConn()
    reply = set_profile_name(conn, PERSON, "  ")
    assert conn.calls == []
    assert "name" in reply.lower()


def test_set_profile_name_never_interpolates_the_name():
    name = "x'); DROP TABLE persons; --"
    conn = FakeConn()
    set_profile_name(conn, PERSON, name)
    for sql, _params in conn.calls:
        assert name not in sql
