"""Offline tests for the transport identity resolver — the AUTH BOUNDARY.

No network, no database: the pure mapper (identity_from_lookup) carries every
safety invariant, and the db_resolver shell is exercised with a fake connection
and a fake that raises. What's proven here is the property the whole migration
leans on — a stranger, or any second user, can NEVER resolve to the owner's
person_id and read the owner's memories.
"""

from types import SimpleNamespace

from aerys_v2.services.context import _is_uuid, build_context
from aerys_v2.transports.resolver import (
    db_resolver,
    identity_from_lookup,
)

# Real owner id (from prod) + a distinct second person. Both valid UUIDs so the
# build_context hydration gate treats them as real people.
OWNER = "6e6bcbed-03ef-4d17-95d2-89c467414335"
OTHER = "11111111-2222-3333-4444-555555555555"


def event(**overrides):
    base = dict(
        platform="discord",
        platform_user_id="123456789",
        display_name="Somebody",
        channel_kind="dm",
    )
    base.update(overrides)
    return SimpleNamespace(**base)


# ---- the pure mapper: identity_from_lookup ---------------------------------


def test_known_account_resolves_to_its_person_id():
    row = {"person_id": OWNER, "display_name": "Chris", "is_new": False}
    ident = identity_from_lookup(row, event())
    assert ident["user_id"] == OWNER
    assert _is_uuid(ident["user_id"])  # a real person -> build_context hydrates
    assert ident["display_name"] == "Chris"  # DB name wins over the platform name


def test_stranger_resolves_cold_and_never_owner():
    ident = identity_from_lookup(None, event(platform_user_id="999"))
    # cold handle: namespaced, NOT a UUID, and provably not the owner
    assert ident["user_id"] == "discord:999"
    assert not _is_uuid(ident["user_id"])
    assert ident["user_id"] != OWNER


def test_second_user_gets_their_own_id_not_the_owner():
    # The resolver has no owner concept at all — a known second person resolves to
    # THEIR row, structurally incapable of inheriting Chris. This test documents it.
    row = {"person_id": OTHER, "display_name": "Megan", "is_new": False}
    ident = identity_from_lookup(row, event(platform_user_id="222"))
    assert ident["user_id"] == OTHER
    assert ident["user_id"] != OWNER


def test_cold_display_name_falls_back_to_platform():
    ident = identity_from_lookup(None, event(display_name="ScreenName"))
    assert ident["display_name"] == "ScreenName"


def test_known_missing_db_name_falls_back_to_platform():
    row = {"person_id": OWNER, "display_name": None, "is_new": False}
    ident = identity_from_lookup(row, event(display_name="ScreenName"))
    assert ident["display_name"] == "ScreenName"


def test_privacy_context_follows_the_room():
    assert identity_from_lookup(None, event(channel_kind="dm"))["privacy_context"] == "private"
    assert identity_from_lookup(None, event(channel_kind="guild"))["privacy_context"] == "public"
    assert identity_from_lookup(None, event(channel_kind="group"))["privacy_context"] == "public"
    # even a KNOWN person in a public room is public — dm-only claims stay hidden
    row = {"person_id": OWNER, "display_name": "Chris", "is_new": False}
    assert identity_from_lookup(row, event(channel_kind="guild"))["privacy_context"] == "public"


def test_cold_handle_is_inert_in_build_context():
    # The whole point of the non-UUID handle: build_context refuses to hydrate it,
    # so even a truthy connection is never touched. No profile, no memories leak.
    cold = identity_from_lookup(None, event())["user_id"]
    assert build_context(cold, "any query", conn=object()) == ""


# ---- the I/O shell: db_resolver (fake connection) --------------------------


class _FakeCursor:
    def __init__(self, row):
        self._row = row

    def fetchone(self):
        return self._row


class _FakeConn:
    """Minimal psycopg-shaped stand-in: context manager + settable read_only + execute."""

    def __init__(self, row):
        self._row = row
        self.read_only = False

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, *args, **kwargs):
        return _FakeCursor(self._row)


def _connect_returning(row):
    def connect(url):
        return _FakeConn(row)

    return connect


def test_db_resolver_maps_a_known_row():
    # resolve_identity reads a (person_id, display_name) tuple; str()'d to a UUID str.
    resolve = db_resolver("postgres://x", connect=_connect_returning((OWNER, "Chris")))
    ident = resolve(event())
    assert ident["user_id"] == OWNER
    assert ident["display_name"] == "Chris"


def test_db_resolver_cold_for_unknown_row():
    resolve = db_resolver("postgres://x", connect=_connect_returning(None))
    ident = resolve(event(platform_user_id="404"))
    assert ident["user_id"] == "discord:404"
    assert not _is_uuid(ident["user_id"])


def test_db_resolver_degrades_cold_when_db_is_down_never_owner():
    def _boom(url):
        raise RuntimeError("NAS is down")

    resolve = db_resolver("postgres://x", connect=_boom)
    ident = resolve(event(platform_user_id="500"))
    # a dead database resolves a stranger, NOT the owner — the safe degradation
    assert ident["user_id"] == "discord:500"
    assert ident["user_id"] != OWNER
    assert not _is_uuid(ident["user_id"])


def test_db_resolver_sets_read_only_before_query():
    # belt-and-braces: the connection is marked read-only (a write can't slip in).
    seen = {}

    class _Recording(_FakeConn):
        def execute(self, *a, **k):
            seen["read_only"] = self.read_only
            return _FakeCursor((OWNER, "Chris"))

    def connect(url):
        return _Recording(None)

    resolve = db_resolver("postgres://x", connect=connect)
    resolve(event())
    assert seen["read_only"] is True
