"""Gap #37 — A2A exchanges on kael:* threads leave a durable memory.

The writer is the unit under test (arming, gating, content shape, fail-open);
the route tests prove the /ask door fires it for kael:* and ONLY kael:* —
tests themselves ride http:default, the same "tests do poison memory" rule
the kael_line suite enforces.
"""

from types import SimpleNamespace

from fastapi.testclient import TestClient

from aerys_v2.services.a2a_memory import a2a_memory_writer_for
from aerys_v2.transports.http_api import build_app


class _Secret:
    def __init__(self, value: str) -> None:
        self._value = value

    def get_secret_value(self) -> str:
        return self._value


def settings_with(db="postgresql://x/y", owner="0" * 32, key="k") -> SimpleNamespace:
    # memories_database_url, NOT database_url: the memories table lives in the
    # prod aerys DB, not the v2 spine (first-deploy lesson, 2026-08-15).
    return SimpleNamespace(
        memories_database_url=db,
        owner_person_id=owner,
        embeddings_api_key=_Secret(key) if key else None,
    )


class _FakeConn:
    def __init__(self, log: list) -> None:
        self._log = log

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql, params):
        self._log.append((sql, params))

    def commit(self):
        self._log.append(("commit", None))


def writer_with(log: list, *, embed=None, **kw):
    return a2a_memory_writer_for(
        settings_with(**kw),
        embed=embed or (lambda text: [0.1, 0.2]),
        connect=lambda url, connect_timeout: _FakeConn(log),
        synchronous=True,
    )


# --- arming ------------------------------------------------------------------


def test_unarmed_without_memories_db_owner_or_embed_key():
    assert a2a_memory_writer_for(settings_with(db=None)) is None
    assert a2a_memory_writer_for(settings_with(owner=None)) is None
    assert a2a_memory_writer_for(settings_with(key=None)) is None


def test_armed_with_full_config():
    assert writer_with([]) is not None


# --- gating ------------------------------------------------------------------


def test_non_kael_threads_write_nothing():
    log: list = []
    write = writer_with(log)
    write("http:default", "hi", "hello")
    write("person:owner", "hi", "hello")
    write("voice:beta", "hi", "hello")
    assert log == []


def test_blank_sides_write_nothing():
    log: list = []
    write = writer_with(log)
    write("kael:checkin", "   ", "reply")
    write("kael:checkin", "ask", "")
    assert log == []


# --- the write ---------------------------------------------------------------


def test_kael_thread_inserts_compact_attributed_memory():
    log: list = []
    write = writer_with(log)
    write("kael:checkin", "the demo went well", "glad to hear it")
    assert log[-1] == ("commit", None)
    sql, params = log[0]
    assert "INSERT INTO memories" in sql
    assert "'kael_line'" in sql and "'private'" in sql
    assert params["person_id"] == "0" * 32
    assert params["key_label"].startswith("a2a_kael_")
    assert params["context"] == "A2A exchange on kael:checkin"
    assert params["embedding"] == "[0.1,0.2]"
    content = params["content"]
    assert 'Kael said: "the demo went well"' in content
    assert 'I replied: "glad to hear it"' in content
    assert content.startswith("Talked with Kael on his line (")


def test_long_sides_are_clipped():
    log: list = []
    write = writer_with(log)
    write("kael:checkin", "x" * 2000, "y" * 2000)
    content = log[0][1]["content"]
    # both sides clipped to ~420 + ellipsis; whole memory stays glanceable
    assert len(content) < 1000
    assert "…" in content


def test_embed_failure_is_swallowed_and_writes_nothing():
    log: list = []

    def boom(text: str):
        raise RuntimeError("embeddings endpoint down")

    write = writer_with(log, embed=boom)
    write("kael:checkin", "ask", "reply")  # must not raise
    assert log == []


# --- the door ----------------------------------------------------------------


def app_with(a2a_fn):
    return build_app(
        lambda text, identity, thread: "ok",
        "sekrit",
        owner_person_id=None,
        a2a_memory_fn=a2a_fn,
    )


def test_ask_route_fires_writer_with_final_thread_and_both_sides():
    calls: list = []
    c = TestClient(app_with(lambda tid, ask, reply: calls.append((tid, ask, reply))))
    r = c.post(
        "/ask",
        json={"text": "hello her", "thread_id": "kael:checkin"},
        headers={"Authorization": "Bearer sekrit"},
    )
    assert r.status_code == 200
    assert calls == [("kael:checkin", "hello her", "ok")]


def test_ask_route_still_fires_writer_on_plain_threads_writer_gates():
    # The route passes every completed /ask through; the WRITER owns the
    # kael:* gate (proven above). This pins the division of labor.
    calls: list = []
    c = TestClient(app_with(lambda tid, ask, reply: calls.append(tid)))
    c.post(
        "/ask",
        json={"text": "hi"},
        headers={"Authorization": "Bearer sekrit"},
    )
    assert calls == ["http:default"]


def test_ask_route_unarmed_is_unchanged():
    c = TestClient(app_with(None))
    r = c.post(
        "/ask",
        json={"text": "hi", "thread_id": "kael:x"},
        headers={"Authorization": "Bearer sekrit"},
    )
    assert r.status_code == 200 and r.json()["reply"] == "ok"
