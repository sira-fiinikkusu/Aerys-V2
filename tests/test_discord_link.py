"""Offline tests for the Discord identity slash-command handlers.

No network, no database, no discord.py: a duck-typed FakeConn (same pattern
as test_extraction.FakeConn) records every (sql, params) call and routes
SELECTs by substring. What's proven:

  - statement ORDER of the merge chains (the caller wraps them in one
    transaction, so order = the n8n node order),
  - the AUTH BOUNDARY: a failed lookup / non-admin call executes ZERO
    mutating SQL and never falls through to any other person's id,
  - winner/loser orientation (code issuer wins; Discord person wins the
    admin merge),
  - reply texts stay close to the n8n originals.
"""

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from aerys_v2.transports.discord_link import (
    ADMIN_LINK_DM_TEXT,
    CODE_CHARS,
    admin_link,
    admin_unlink,
    generate_link_code,
    link,
    sweep_expired,
    unlink,
)

ISSUER = "aaaaaaaa-1111-2222-3333-444444444444"  # person who issued the code
CALLER = "bbbbbbbb-5555-6666-7777-888888888888"  # person redeeming / running the command
OWNER = "6e6bcbed-03ef-4d17-95d2-89c467414335"  # prod owner id — must never appear


class FakeCursor:
    def __init__(self, rows):
        self.rows = rows
        self.rowcount = len(rows)

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


def executed(conn, needle):
    return [(sql, params) for sql, params in conn.calls if needle in sql]


def statement_order(conn, needles):
    """Indices into conn.calls for each needle (asserts each appears exactly once)."""
    order = []
    for needle in needles:
        hits = [i for i, (sql, _) in enumerate(conn.calls) if needle in sql]
        assert len(hits) == 1, f"expected exactly one {needle!r} statement, got {len(hits)}"
        order.append(hits[0])
    return order


# ---- /link — code issue ------------------------------------------------------


def test_link_no_code_stores_pending_link_and_replies_with_instructions():
    conn = FakeConn()
    reply = link(conn, CALLER, "discord-123", code=None, code_gen=lambda: "ABC234")
    inserts = executed(conn, "INSERT INTO pending_links")
    assert len(inserts) == 1
    _, params = inserts[0]
    assert params["code"] == "ABC234"
    assert params["person_id"] == CALLER
    # the stored platform is hardcoded 'discord' in the SQL itself
    assert "'discord'" in inserts[0][0]
    # redemption instructions carry the code and the 10-minute TTL
    assert "ABC234" in reply
    assert "10 minutes" in reply
    assert "/link ABC234" in reply


def test_link_code_expiry_is_ten_minutes_out():
    conn = FakeConn()
    before = datetime.now(timezone.utc)
    link(conn, CALLER, "discord-123", code=None, code_gen=lambda: "ABC234")
    after = datetime.now(timezone.utc)
    (_, params), = executed(conn, "INSERT INTO pending_links")
    assert before + timedelta(minutes=10) <= params["expires_at"] <= after + timedelta(minutes=10)


def test_default_code_gen_shape():
    for _ in range(20):
        code = generate_link_code()
        assert len(code) == 6
        assert all(c in CODE_CHARS for c in code)
        # lookalikes excluded by the alphabet
        assert not set(code) & set("IO01")


def test_link_empty_string_code_issues_a_new_code():
    # n8n Check Link Code treats '' the same as absent
    conn = FakeConn()
    reply = link(conn, CALLER, "discord-123", code="", code_gen=lambda: "ZZZZ99")
    assert executed(conn, "INSERT INTO pending_links")
    assert "ZZZZ99" in reply


# ---- /link — redemption ------------------------------------------------------


def _lookup_route(rows):
    return ("FROM pending_links pl", rows)


def test_link_unknown_or_expired_code_runs_zero_merges():
    conn = FakeConn(routes=[_lookup_route([])])
    reply = link(conn, CALLER, "discord-123", code="NOPE22")
    assert "expired" in reply or "wasn't found" in reply
    # AUTH BOUNDARY: nothing mutated — no UPDATE, no DELETE, no audit
    assert not [c for c in conn.calls if "UPDATE" in c[0]]
    assert not executed(conn, "DELETE FROM pending_links WHERE code")
    assert not executed(conn, "audit_log")
    # and the reply names no person id
    assert CALLER not in reply and OWNER not in reply


def test_link_code_is_uppercased_for_lookup():
    conn = FakeConn(routes=[_lookup_route([])])
    link(conn, CALLER, "discord-123", code="abc234")
    (_, params), = executed(conn, "FROM pending_links pl")
    assert params["code"] == "ABC234"


def test_link_same_person_short_circuits():
    conn = FakeConn(routes=[_lookup_route([(CALLER, "telegram")])])
    reply = link(conn, CALLER, "discord-123", code="ABC234")
    assert "already linked" in reply
    assert not [c for c in conn.calls if "UPDATE" in c[0]]


def test_link_merge_statement_order_and_winner_loser():
    conn = FakeConn(routes=[_lookup_route([(ISSUER, "telegram")])])
    reply = link(conn, CALLER, "discord-123", code="ABC234")

    order = statement_order(
        conn,
        [
            "FROM pending_links pl",  # Lookup Link Code
            "UPDATE platform_identities SET person_id = %(winner)s WHERE",
            "UPDATE conversations SET person_id = %(winner)s WHERE",
            "UPDATE messages SET person_id = %(winner)s WHERE",
            "UPDATE memories SET person_id = %(winner)s WHERE",
            "UPDATE persons SET deleted_at = NOW() WHERE id = %(loser)s",
            "DELETE FROM pending_links WHERE code",
            "DELETE FROM pending_links WHERE person_id = %(loser)s",
            "INSERT INTO audit_log",
        ],
    )
    assert order == sorted(order), "merge statements ran out of n8n node order"

    # winner = the code ISSUER; loser = the redeeming CALLER — on every statement
    for sql, params in conn.calls:
        if "%(winner)s" in sql:
            assert params["winner"] == ISSUER
            assert params["loser"] == CALLER
    (_, del_params), = executed(conn, "DELETE FROM pending_links WHERE code")
    assert del_params == {"code": "ABC234"}
    assert "linked" in reply.lower()
    assert "telegram" in reply  # names the source platform


def test_link_merge_soft_deletes_the_caller_not_the_issuer():
    conn = FakeConn(routes=[_lookup_route([(ISSUER, "telegram")])])
    link(conn, CALLER, "discord-123", code="ABC234")
    (_, params), = executed(conn, "UPDATE persons SET deleted_at")
    assert params["loser"] == CALLER
    assert params["loser"] != ISSUER


def test_link_merge_kills_every_outstanding_code_of_the_loser():
    """A stale code the loser issued pre-merge must die with the merge —
    otherwise a third person redeeming it gets merged INTO a soft-deleted
    person (review finding 2026-07-11)."""
    conn = FakeConn(routes=[_lookup_route([(ISSUER, "telegram")])])
    link(conn, CALLER, "discord-123", code="ABC234")
    (_, params), = executed(conn, "DELETE FROM pending_links WHERE person_id")
    assert params == {"loser": CALLER}


def test_link_audit_records_the_merge():
    conn = FakeConn(routes=[_lookup_route([(ISSUER, "telegram")])])
    link(conn, CALLER, "discord-123", code="ABC234")
    (sql, params), = executed(conn, "INSERT INTO audit_log")
    assert "('user'" in sql and "::jsonb" in sql
    assert params["action"] == "link_merge"
    assert ISSUER in params["details"] and CALLER in params["details"]


# ---- /unlink -----------------------------------------------------------------


def test_unlink_sole_identity_is_refused():
    conn = FakeConn(routes=[("count(*)", [(1,)])])
    reply = unlink(conn, CALLER, "discord", "discord-123")
    assert "can't unlink" in reply.lower() or "cannot" in reply.lower()
    assert not [c for c in conn.calls if "DELETE" in c[0]]


def test_unlink_zero_identities_is_refused_too():
    # count 0 shouldn't happen for a resolved person, but must not delete
    conn = FakeConn(routes=[("count(*)", [(0,)])])
    unlink(conn, CALLER, "discord", "discord-123")
    assert not [c for c in conn.calls if "DELETE" in c[0]]


def test_unlink_zero_row_delete_is_an_honest_miss_not_a_success():
    """The count guard is person-level; the narrowed DELETE can still match 0
    rows (platform_user_id not actually this person's). No success claim, no
    audit row for a mutation that never happened (review finding 2026-07-11)."""
    conn = FakeConn(routes=[("count(*)", [(2,)])])  # DELETE unrouted -> rowcount 0
    reply = unlink(conn, CALLER, "discord", "someone-elses-account")
    assert "isn't linked" in reply
    assert not [c for c in conn.calls if "audit_log" in c[0]]


def test_unlink_with_two_identities_deletes_exactly_this_account():
    conn = FakeConn(
        routes=[("count(*)", [(2,)]),
                ("DELETE FROM platform_identities", [(1,)])]  # rowcount 1: the row died
    )
    reply = unlink(conn, CALLER, "discord", "discord-123")
    (sql, params), = executed(conn, "DELETE FROM platform_identities")
    # tightened WHERE: platform AND person AND platform_user_id
    assert params == {
        "platform": "discord",
        "person_id": CALLER,
        "platform_user_id": "discord-123",
    }
    assert "platform_user_id" in sql
    assert "Unlinked" in reply
    (_, audit), = executed(conn, "INSERT INTO audit_log")
    assert audit["action"] == "unlink"


# ---- /admin-link -------------------------------------------------------------


def _both_route(discord_person, telegram_person):
    return ("AS discord_person_id", [(discord_person, telegram_person)])


def test_admin_link_not_admin_runs_zero_sql():
    conn = FakeConn()
    result = admin_link(conn, "d-1", "t-1", is_admin=False)
    assert result["notify"] == []
    assert "Aerys Admin" in result["reply"]
    assert conn.calls == []  # AUTH BOUNDARY: not even the lookup runs


def test_admin_link_missing_discord_identity():
    conn = FakeConn(routes=[_both_route(None, ISSUER)])
    result = admin_link(conn, "d-1", "t-1", is_admin=True)
    assert "Discord user d-1" in result["reply"]
    assert result["notify"] == []
    assert not [c for c in conn.calls if "UPDATE" in c[0]]


def test_admin_link_missing_telegram_identity():
    conn = FakeConn(routes=[_both_route(CALLER, None)])
    result = admin_link(conn, "d-1", "t-1", is_admin=True)
    assert "Telegram user t-1" in result["reply"]
    assert result["notify"] == []
    assert not [c for c in conn.calls if "UPDATE" in c[0]]


def test_admin_link_both_missing_names_both():
    conn = FakeConn(routes=[_both_route(None, None)])
    result = admin_link(conn, "d-1", "t-1", is_admin=True)
    assert "Discord user d-1" in result["reply"]
    assert "Telegram user t-1" in result["reply"]
    assert not [c for c in conn.calls if "UPDATE" in c[0]]


def test_admin_link_already_same_person_is_a_noop():
    conn = FakeConn(routes=[_both_route(CALLER, CALLER)])
    result = admin_link(conn, "d-1", "t-1", is_admin=True)
    assert "already" in result["reply"]
    assert result["notify"] == []
    assert not [c for c in conn.calls if "UPDATE" in c[0]]


def test_admin_link_merge_order_discord_wins_and_dm_notify():
    # Discord person is winner; Telegram is loser (n8n Prepare Admin Merge)
    conn = FakeConn(routes=[_both_route(ISSUER, CALLER)])
    result = admin_link(conn, "d-1", "t-1", is_admin=True)

    order = statement_order(
        conn,
        [
            "AS discord_person_id",  # Lookup Both Identities
            "UPDATE platform_identities SET person_id = %(winner)s::uuid",
            "UPDATE conversations SET person_id = %(winner)s::uuid",
            "UPDATE messages SET person_id = %(winner)s::uuid",
            "UPDATE memories SET person_id = %(winner)s::uuid",
            "UPDATE persons SET deleted_at = NOW() WHERE id = %(loser)s::uuid",
            "DELETE FROM pending_links WHERE person_id = %(loser)s::uuid",
            "INSERT INTO audit_log",
        ],
    )
    assert order == sorted(order), "admin merge statements ran out of n8n node order"
    for sql, params in conn.calls:
        if "%(winner)s" in sql:
            assert params["winner"] == ISSUER  # the discord person
            assert params["loser"] == CALLER  # the telegram person
    # notify: exactly one Discord DM, verbatim n8n text; binding layer sends it
    assert result["notify"] == [
        {"platform": "discord", "user_id": "d-1", "text": ADMIN_LINK_DM_TEXT}
    ]
    assert "unified" in result["notify"][0]["text"]
    assert "Linked" in result["reply"]


# ---- /admin-unlink -----------------------------------------------------------


def test_admin_unlink_not_admin_runs_zero_sql():
    conn = FakeConn()
    reply = admin_unlink(conn, "telegram", "t-9", is_admin=False)
    assert "Aerys Admin" in reply
    assert conn.calls == []


def test_admin_unlink_target_not_found():
    conn = FakeConn(routes=[("LIMIT 1", [])])
    reply = admin_unlink(conn, "telegram", "t-9", is_admin=True)
    assert "No telegram identity found" in reply
    assert not [c for c in conn.calls if "DELETE" in c[0]]


def test_admin_unlink_deletes_and_reports():
    conn = FakeConn(routes=[("LIMIT 1", [(CALLER,)])])
    reply = admin_unlink(conn, "telegram", "t-9", is_admin=True)
    (sql, params), = executed(conn, "DELETE FROM platform_identities")
    assert params == {"platform": "telegram", "target_id": "t-9"}
    assert "Unlinked telegram account t-9" in reply
    assert CALLER in reply  # names the person it belonged to
    (_, audit), = executed(conn, "INSERT INTO audit_log")
    assert audit["action"] == "admin_unlink"
    assert CALLER in audit["details"]


# ---- sweep -------------------------------------------------------------------


def test_sweep_expired_deletes_only_expired_rows():
    conn = FakeConn(routes=[("expires_at < NOW()", [("x",), ("y",)])])
    deleted = sweep_expired(conn)
    (sql, params), = conn.calls
    assert sql == "DELETE FROM pending_links WHERE expires_at < NOW()"
    assert params is None
    assert deleted == 2  # rowcount passthrough for cron logging


def test_handlers_never_swallow_db_errors():
    # PURE handler posture: exceptions raise; the binding layer catches and
    # rolls the transaction back.
    class Boom:
        def execute(self, *a, **k):
            raise RuntimeError("NAS is down")

    import pytest

    with pytest.raises(RuntimeError):
        link(Boom(), CALLER, "discord-123", code="ABC234")
    with pytest.raises(RuntimeError):
        unlink(Boom(), CALLER, "discord", "discord-123")
    with pytest.raises(RuntimeError):
        admin_link(Boom(), "d-1", "t-1", is_admin=True)
    with pytest.raises(RuntimeError):
        admin_unlink(Boom(), "telegram", "t-9", is_admin=True)
