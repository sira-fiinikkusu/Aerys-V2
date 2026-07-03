"""Offline tests for the MEMORY-RETRIEVAL context seam — no Postgres, no network.

Three layers proven here, all with fakes:
  1. build_context assembly: profile + memories become one block, and EVERY
     failure path (no conn, unknown person, zero rows, a half that throws)
     degrades to less context — never an exception into the turn.
  2. Graph injection: the chat node asks context_fn (identity person_id + the
     latest user text) and splices the block under the
     "[What you know about this person]" header; the capability line only
     claims cross-conversation recall when retrieval is actually wired.
  3. HTTP owner mapping: with owner_person_id configured, voice + /ask callers
     resolve to the owner's persons.id — so voice-Chris gets HIS memories.
"""

from datetime import datetime, timedelta, timezone
import uuid

from fastapi.testclient import TestClient
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from aerys_v2.factory import build_graph
from aerys_v2.service import ask
from aerys_v2.services.context import build_context
from aerys_v2.transports.http_api import build_app

NOW = datetime.now(timezone.utc)
PERSON = "6e6bcbed-03ef-4d17-95d2-89c467414335"  # a real UUID — passes the guard
FAKE_EMBED = lambda text: [0.5, 0.5]  # noqa: E731 — "text in, vector out", offline


def stamp(days_ago: int) -> str:
    """The '(YYYY-MM-DD, Nd ago)' suffix format_memory_context appends per line."""
    created = NOW - timedelta(days=days_ago)
    return f"({created.date().isoformat()}, {days_ago}d ago)"


# --- fakes (same duck-typed psycopg trick as test_services.py) ----------------


class FakeCursor:
    def __init__(self, rows):
        self.rows = rows

    def fetchone(self):
        return self.rows[0] if self.rows else None

    def fetchall(self):
        return self.rows


class FakeConn:
    """Duck-typed psycopg connection: records every execute, replays canned rows."""

    def __init__(self, results=None):
        self.results = list(results or [])
        self.calls = []  # [(sql, params), ...]

    def execute(self, sql, params=None):
        self.calls.append((sql, params))
        return FakeCursor(self.results.pop(0) if self.results else [])


class ExplodingConn(FakeConn):
    """First execute raises (profile half dies), later ones replay canned rows —
    proves each half of build_context is independently fenced."""

    def __init__(self, results=None):
        super().__init__(results)
        self.exploded = False

    def execute(self, sql, params=None):
        if not self.exploded:
            self.exploded = True
            raise RuntimeError("boom: NAS fell over mid-query")
        return super().execute(sql, params)


def profile_row(key_label, claim_text):
    # column order of profile.PROFILE_SQL
    return (uuid.uuid4(), key_label, claim_text, "approved", False, 0.9, "P3", "all")


def memory_row(content, source="discord", days_ago=3):
    # column order of memory.MEMORY_SQL
    return (
        uuid.uuid4(),
        uuid.UUID(PERSON),
        content,
        source,
        "public",
        NOW - timedelta(days=days_ago),
        0.8,
    )


# --- build_context: assembly ---------------------------------------------------


def test_context_assembles_profile_then_memories():
    conn = FakeConn(
        [
            [profile_row("basic.name", "Preferred name: Chris")],  # get_profile runs first
            [memory_row("likes: black coffee")],  # retrieve_memories runs second
        ]
    )
    block = build_context(PERSON, "coffee?", conn, embed=FAKE_EMBED)
    # profile lines first (who they ARE), memories second (what happened lately)
    assert block == (
        "• Preferred name: Chris"
        f"\n\nRelevant memories:\n* black coffee [discord] {stamp(3)}"
    )


def test_context_defaults_to_private_privacy():
    # Today's only wired caller is the owner's own channel — private context:
    # profile sees dm-visibility claims, memories see both privacy levels.
    conn = FakeConn([[], []])
    build_context(PERSON, "hi", conn, embed=FAKE_EMBED)
    assert conn.calls[0][1]["pctx"] == "private"  # profile query
    assert conn.calls[1][1]["levels"] == ["public", "private"]  # memory query


def test_context_profile_only_when_no_memory_rows():
    conn = FakeConn([[profile_row("interests.gaming", "Plays Subnautica with Joe")], []])
    assert build_context(PERSON, "hi", conn, embed=FAKE_EMBED) == (
        "• Plays Subnautica with Joe"
    )


def test_context_memories_only_when_profile_cold_start():
    conn = FakeConn([[], [memory_row("no colon memory", source=None, days_ago=0)]])
    assert build_context(PERSON, "hi", conn, embed=FAKE_EMBED) == (
        f"Relevant memories:\n* no colon memory {stamp(0)}"
    )


# --- build_context: graceful paths (never raises into the turn) ----------------


def test_context_no_conn_is_empty():
    assert build_context(PERSON, "hi", None, embed=FAKE_EMBED) == ""


def test_context_non_uuid_person_skips_db_entirely():
    # Transport-minted ids ("cli-operator", "discord:12345") = "no person":
    # no roundtrip, no ::uuid cast error, empty context.
    conn = FakeConn()
    for pid in ("cli-operator", "discord:12345", "", None):
        assert build_context(pid, "hi", conn, embed=FAKE_EMBED) == ""
    assert conn.calls == []


def test_context_zero_rows_everywhere_is_empty_string():
    assert build_context(PERSON, "hi", FakeConn([[], []]), embed=FAKE_EMBED) == ""


def test_context_without_embed_seam_still_serves_profile():
    # No embedder configured → the memory half is skipped ON PURPOSE (not via
    # the ValueError retrieve_memories would raise) — profile still stands.
    conn = FakeConn([[profile_row("basic.name", "Preferred name: Chris")]])
    assert build_context(PERSON, "hi", conn) == "• Preferred name: Chris"
    assert len(conn.calls) == 1  # only the profile SELECT ran


def test_context_survives_a_query_exploding(caplog):
    # Profile SELECT raises; memories still emit — halves are fenced separately.
    conn = ExplodingConn([[memory_row("likes: espresso", days_ago=0)]])
    assert build_context(PERSON, "hi", conn, embed=FAKE_EMBED) == (
        f"Relevant memories:\n* espresso [discord] {stamp(0)}"
    )
    # Degrade-graceful is NOT degrade-silent: the swallowed failure must log.
    assert any(
        "profile context failed" in r.message and r.levelname == "WARNING"
        for r in caplog.records
    )


# --- graph injection (RecordingModel pattern from test_service.py) -------------

CHRIS = {"user_id": PERSON, "display_name": "Chris"}
HEADER = "[What you know about this person]"


class RecordingModel(GenericFakeChatModel):
    """Fake that records the system prompt each turn (for prompt-shape tests)."""

    seen: list = []

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        type(self).seen.append(str(messages[0].content))
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)


def recording_graph(context_fn, *replies):
    RecordingModel.seen = []
    model = RecordingModel(messages=iter([AIMessage(content=r) for r in replies]))
    return build_graph(model, soul="test soul", context_fn=context_fn)


def test_context_block_lands_under_header_in_system_prompt():
    graph = recording_graph(lambda pid, text: "• Preferred name: Chris", "ok")
    ask(graph, "hi", identity=CHRIS, thread_id="t1")
    assert f"{HEADER}\n• Preferred name: Chris" in RecordingModel.seen[0]


def test_context_fn_receives_person_id_and_latest_user_text():
    calls = []

    def spy(person_id, query_text):
        calls.append((person_id, query_text))
        return ""

    graph = recording_graph(spy, "one", "two")
    ask(graph, "first message", identity=CHRIS, thread_id="t1")
    ask(graph, "second message", identity=CHRIS, thread_id="t1")
    # person_id = identity user_id; query = the LATEST human turn, not history
    assert calls == [(PERSON, "first message"), (PERSON, "second message")]


def test_empty_context_means_no_header():
    graph = recording_graph(lambda pid, text: "", "ok")
    ask(graph, "hi", identity=CHRIS, thread_id="t1")
    assert HEADER not in RecordingModel.seen[0]


def test_no_context_fn_means_no_header_and_no_recall_claim():
    graph = recording_graph(None, "ok")
    ask(graph, "hi", identity=CHRIS, thread_id="t1")
    assert HEADER not in RecordingModel.seen[0]
    # claims follow facts: without retrieval wired, no cross-conversation promise
    assert "ALL conversations" not in RecordingModel.seen[0]
    assert "memory is durable" in RecordingModel.seen[0]  # thread claim stays


def test_wired_context_fn_extends_capability_claim():
    graph = recording_graph(lambda pid, text: "", "ok")
    ask(graph, "hi", identity=CHRIS, thread_id="t1")
    # even with nothing retrieved THIS turn, the capability is real → claimed
    assert "ALL conversations" in RecordingModel.seen[0]


def test_raising_context_fn_never_kills_the_turn():
    def bomb(person_id, query_text):
        raise RuntimeError("NAS unplugged")

    graph = recording_graph(bomb, "still alive")
    assert ask(graph, "hi", identity=CHRIS, thread_id="t1") == "still alive"
    assert HEADER not in RecordingModel.seen[0]


# --- HTTP owner identity mapping ------------------------------------------------

OWNER = PERSON


def spy_app(owner_person_id):
    """TestClient whose ask_fn records the identity each route built."""
    identities = []

    def fake_ask(text, identity, thread_id):
        identities.append(identity)
        return "ok"

    return TestClient(build_app(fake_ask, "sekrit", owner_person_id)), identities


AUTH = {"Authorization": "Bearer sekrit"}


def test_ask_route_resolves_to_owner_person_id():
    client, identities = spy_app(OWNER)
    client.post("/ask", json={"text": "hi", "display_name": "Chris (HTTP)"}, headers=AUTH)
    # user_id = owner persons.id (the memory retrieval key); display_name untouched
    assert identities == [{"user_id": OWNER, "display_name": "Chris (HTTP)"}]


def test_voice_route_resolves_to_owner_person_id():
    client, identities = spy_app(OWNER)
    client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}]},
        headers=AUTH,
    )
    assert identities == [{"user_id": OWNER, "display_name": "Chris (Voice)"}]


def test_no_owner_configured_keeps_anonymous_http_caller():
    client, identities = spy_app(None)
    client.post("/ask", json={"text": "hi"}, headers=AUTH)
    client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}]},
        headers=AUTH,
    )
    assert [i["user_id"] for i in identities] == ["http-caller", "http-caller"]
