"""Offline tests for the slash-command binding layer's PURE pieces.

The discord.py shell (attach_interactions) is exercised live like the rest of
the gateway; what CI pins here is the auth-adjacent logic that must never
regress: create-on-miss can't orphan or misattribute a person, the admin gate
fails closed, and link codes come from a sane alphabet.
"""

from contextlib import contextmanager
from types import SimpleNamespace

from aerys_v2.transports.discord_interactions import (
    _CODE_CHARS,
    ensure_person,
    generate_link_code,
    is_admin_member,
)

OWNER = "6e6bcbed-03ef-4d17-95d2-89c467414335"
NEW = "99999999-8888-7777-6666-555555555555"


class FakeResult:
    def __init__(self, row):
        self._row = row

    def fetchone(self):
        return self._row


class FakeConn:
    """Scripted conn: maps SQL substrings -> queued results, records execution
    order, and tracks whether the transaction block rolled back."""

    def __init__(self, script):
        # script: list of (substring, row) consumed in order per matching call
        self.script = list(script)
        self.executed = []
        self.rolled_back = False

    def execute(self, sql, params=None):
        self.executed.append((sql, params))
        for i, (needle, row) in enumerate(self.script):
            if needle in sql:
                self.script.pop(i)
                return FakeResult(row)
        return FakeResult(None)

    @contextmanager
    def transaction(self):
        try:
            yield
        except BaseException:
            self.rolled_back = True
            raise


# ---- ensure_person ----------------------------------------------------------


def test_existing_identity_short_circuits():
    conn = FakeConn([("SELECT person_id::text FROM platform_identities", (OWNER,))])
    assert ensure_person(conn, "discord", "123", "Chris") == OWNER
    # No INSERT ever ran.
    assert all("INSERT" not in sql for sql, _ in conn.executed)


def test_fresh_account_mints_person_and_identity():
    conn = FakeConn(
        [
            ("SELECT person_id::text FROM platform_identities", None),
            ("INSERT INTO persons", (NEW,)),
            ("INSERT INTO platform_identities", (NEW,)),
        ]
    )
    assert ensure_person(conn, "discord", "456", "Somebody") == NEW
    assert not conn.rolled_back


def test_lost_race_rolls_back_and_returns_winner():
    """Two first-contacts race: our identity INSERT conflicts (RETURNING empty).
    The transaction must roll back (no orphan persons row) and the winner's
    person_id — NOT our freshly minted one — must be returned."""
    winner = "11111111-2222-3333-4444-555555555555"
    conn = FakeConn(
        [
            ("SELECT person_id::text FROM platform_identities", None),  # initial miss
            ("INSERT INTO persons", (NEW,)),
            ("INSERT INTO platform_identities", None),  # ON CONFLICT DO NOTHING -> no row
            ("SELECT person_id::text FROM platform_identities", (winner,)),  # re-select
        ]
    )
    assert ensure_person(conn, "discord", "789", "Racer") == winner
    assert conn.rolled_back  # the orphan persons row died with the transaction


# ---- admin gate --------------------------------------------------------------


def _member(*role_ids):
    return SimpleNamespace(roles=[SimpleNamespace(id=r) for r in role_ids])


def test_admin_role_grants():
    assert is_admin_member(_member(42, 1421594197147910194), 1421594197147910194)


def test_wrong_roles_fail_closed():
    assert not is_admin_member(_member(42, 43), 1421594197147910194)


def test_dm_user_without_roles_fails_closed():
    assert not is_admin_member(SimpleNamespace(name="dm-user"), 1421594197147910194)


def test_unconfigured_role_id_fails_closed():
    """DISCORD_ADMIN_ROLE_ID unset -> nobody is admin, even with matching-ish roles."""
    assert not is_admin_member(_member(1421594197147910194), None)


# ---- link codes ---------------------------------------------------------------


def test_link_code_shape():
    for _ in range(50):
        code = generate_link_code()
        assert len(code) == 6
        assert all(c in _CODE_CHARS for c in code)
    # The ambiguous glyphs the n8n alphabet excluded stay excluded.
    assert not set("01OI") & set(_CODE_CHARS)
