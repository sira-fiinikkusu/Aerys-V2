"""The checkpointer must SURVIVE Postgres going away and coming back.

Regression cover for two outages inside 24 hours (2026-07-30/31). The NAS wedged
on a full LVM-thin pool; `PostgresSaver.from_conn_string` holds ONE connection
for the life of the process, so when Postgres returned the connection stayed
dead and every turn raised `OperationalError: the connection is closed` — while
the container reported healthy and `/health` returned 200. The second occurrence
served 500s for ~13.5 hours overnight and was only caught because the daily loop
ran a REAL turn instead of trusting a health endpoint.

These tests pin the SHAPE of the fix, not the plumbing of psycopg: a pool, with
a liveness check, configured the way PostgresSaver requires. A future refactor
that quietly drops `check=` would restore the outage, and that is exactly the
kind of regression a test should make impossible.
"""

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from aerys_v2 import factory
from aerys_v2.config import Settings


class FakePool:
    """Records how the pool was constructed; PostgresSaver never sees a real DB."""

    instances: list["FakePool"] = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.opened = None
        self.closed = False
        FakePool.instances.append(self)

    # psycopg_pool's real API surface that we exercise
    def open(self, wait=False, timeout=None):
        self.opened = {"wait": wait, "timeout": timeout}

    def close(self):
        self.closed = True

    @staticmethod
    def check_connection(conn):  # the sentinel we assert on
        return None


@pytest.fixture(autouse=True)
def _reset():
    FakePool.instances.clear()
    yield
    FakePool.instances.clear()


def _patch(monkeypatch, saver_factory=None):
    """Swap ConnectionPool + PostgresSaver at their import sites."""
    import psycopg_pool
    import langgraph.checkpoint.postgres as lg_pg

    monkeypatch.setattr(psycopg_pool, "ConnectionPool", FakePool)

    class FakeSaver:
        def __init__(self, conn, *a, **kw):
            self.conn = conn
            self.setup_called = False

        def setup(self):
            self.setup_called = True

    monkeypatch.setattr(lg_pg, "PostgresSaver", saver_factory or FakeSaver)


def test_no_database_url_still_yields_in_memory(monkeypatch):
    """The offline path is untouched — tests and any box without the LAN."""
    settings = Settings(anthropic_api_key="sk-test", database_url=None)  # type: ignore[arg-type]
    with factory.checkpointer_for(settings) as saver:
        assert isinstance(saver, InMemorySaver)
    assert not FakePool.instances, "no pool should be built without a database_url"


def test_pool_carries_the_liveness_check(monkeypatch):
    """★ THE REGRESSION GUARD. Without `check`, the pool hands back the same
    dead connection forever and a NAS blip becomes a multi-hour outage."""
    _patch(monkeypatch)
    settings = Settings(anthropic_api_key="sk-test", database_url="postgresql://u:p@host/db")  # type: ignore[arg-type]
    with factory.checkpointer_for(settings):
        pass

    pool = FakePool.instances[0]
    assert pool.kwargs["check"] is FakePool.check_connection, (
        "the pool MUST validate a connection before handing it out"
    )


def test_pool_configures_what_postgressaver_requires(monkeypatch):
    """PostgresSaver manages its own transactions, reads dict rows, and must not
    cache prepared statements across pooled connections."""
    _patch(monkeypatch)
    from psycopg.rows import dict_row

    settings = Settings(anthropic_api_key="sk-test", database_url="postgresql://u:p@host/db")  # type: ignore[arg-type]
    with factory.checkpointer_for(settings):
        pass

    kw = FakePool.instances[0].kwargs["kwargs"]
    assert kw["autocommit"] is True
    assert kw["prepare_threshold"] == 0
    assert kw["row_factory"] is dict_row


def test_boot_fails_fast_and_setup_runs(monkeypatch):
    """A brain that cannot reach its own memory should NOT come up quietly —
    open(wait=True) preserves from_conn_string's fail-fast behaviour."""
    _patch(monkeypatch)
    settings = Settings(anthropic_api_key="sk-test", database_url="postgresql://u:p@host/db")  # type: ignore[arg-type]
    with factory.checkpointer_for(settings) as saver:
        assert saver.setup_called, "checkpoint tables must be ensured on boot"

    pool = FakePool.instances[0]
    assert pool.opened["wait"] is True
    assert pool.opened["timeout"] == settings.checkpoint_pool_open_timeout_s


def test_pool_is_closed_even_when_the_body_raises(monkeypatch):
    """No leaked connections when the server dies mid-serve."""
    _patch(monkeypatch)
    settings = Settings(anthropic_api_key="sk-test", database_url="postgresql://u:p@host/db")  # type: ignore[arg-type]
    with pytest.raises(RuntimeError):
        with factory.checkpointer_for(settings):
            raise RuntimeError("server died")
    assert FakePool.instances[0].closed is True
