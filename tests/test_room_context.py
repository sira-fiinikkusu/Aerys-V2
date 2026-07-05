"""Offline tests for the cross-surface continuity plumbing — no network, no DB.

Covers:
  1. Part 1 person-keyed threading: person_thread_key is cross-surface + per-person.
  2. Part 4 room formatter (services.room_context.format_room_context).
  3. The v2_turns channel derivation from the resolved surface (turns.channel_enum /
     build_turn_row) — the migration-005 channel_id + display_name columns.
  4. The chat node: a PUBLIC turn injects the room block and answers "where" from the
     identity surface (person-keyed thread no longer encodes it); a DM gets neither.
"""

from datetime import datetime, timezone

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from aerys_v2.factory import _channel_phrase, _surface_thread_for_phrase, build_graph
from aerys_v2.service import ask
from aerys_v2.services.room_context import format_room_context
from aerys_v2.transports.discord_gateway import person_thread_key
from aerys_v2.turns import build_turn_row, channel_enum, derive_channel

NOW = datetime.now(timezone.utc)
OWNER_UUID = "6e6bcbed-03ef-4d17-95d2-89c467414335"
PUBLIC_GUILD = {
    "user_id": "person-1", "display_name": "Chris", "privacy_context": "public",
    "platform": "discord", "channel_kind": "guild", "channel_id": "555",
    "channel_name": "general",
}
PRIVATE_DM = {"user_id": "person-1", "display_name": "Chris", "privacy_context": "private"}


def fake_model(*replies: str) -> GenericFakeChatModel:
    return GenericFakeChatModel(messages=iter([AIMessage(content=r) for r in replies]))


# ── Part 1: person-keyed threading ───────────────────────────────────────────

def test_person_thread_key_is_cross_surface_and_per_person():
    # his DM and his guild resolve to ONE thread (continuity); two people are distinct
    assert person_thread_key(OWNER_UUID) == f"person:{OWNER_UUID}"
    assert person_thread_key("uuid-a") != person_thread_key("uuid-b")
    # a cold stranger handle keys its OWN inert thread — never merges with anyone
    assert person_thread_key("discord:999") == "person:discord:999"


# ── Part 4: the room formatter ───────────────────────────────────────────────

def test_format_room_context_renders_chronologically_with_replies():
    # ROOM_TURNS_SQL returns NEWEST-first; the formatter reverses to chronological.
    rows = [
        ("Megan", "uuid-m", "what's for dinner?", "Pizza, probably.", NOW),   # newest
        ("Chris", "uuid-c", "hey room", None, NOW),                            # older
    ]
    block = format_room_context(rows)
    lines = block.split("\n")
    assert lines[0] == "Chris: hey room"                  # older turn first
    assert "Megan: what's for dinner?" in block
    assert "Aerys: Pizza, probably." in block             # her reply rendered when present


def test_format_room_context_empty_is_blank():
    assert format_room_context([]) == ""


def test_format_room_context_speaker_fallbacks():
    # no display_name -> short person handle; no name AND no id -> neutral 'Someone'
    assert "person·1234" in format_room_context([(None, "abcd-1234", "hi", None, NOW)])
    assert "Someone" in format_room_context([("", None, "hi", None, NOW)])


def test_format_room_context_clips_long_fields():
    long = "x" * 5000
    block = format_room_context([("Chris", "c", long, None, NOW)])
    assert len(block) < 400 and block.endswith("…")


# ── v2_turns channel derivation from the resolved surface (channel_enum) ──────

def test_channel_enum_maps_surface_to_the_v2_turns_enum():
    assert channel_enum("discord", "dm") == "discord_dm"
    assert channel_enum("discord", "guild") == "guild"
    assert channel_enum("telegram", "dm") == "telegram_dm"
    assert channel_enum("telegram", "group") == "telegram_group"
    # single-user surfaces set nothing -> '' signals derive_channel(thread_id) fallback
    assert channel_enum("voice", "beta") == ""
    assert channel_enum(None, None) == ""


def test_build_turn_row_channel_and_columns_from_identity_surface():
    ident = {
        "user_id": OWNER_UUID, "display_name": "Chris",
        "platform": "discord", "channel_kind": "guild", "channel_id": "555",
    }
    # thread_id is person-keyed and no longer names the surface — the row's channel
    # must come from the identity surface, not derive_channel('person:...').
    row = build_turn_row(
        thread_id=f"person:{OWNER_UUID}", identity=ident, input_text="hi",
        latency_ms=1, messages=[],
    )
    assert row["channel"] == "guild"                 # NOT 'person'
    assert row["channel_id"] == "555"                # migration-005 room key
    assert row["display_name"] == "Chris"            # migration-005 speaker label


def test_build_turn_row_falls_back_to_thread_for_single_user_surfaces():
    row = build_turn_row(
        thread_id="voice:beta",
        identity={"user_id": OWNER_UUID, "display_name": "Chris"},
        input_text="hi", latency_ms=1, messages=[],
    )
    assert row["channel"] == derive_channel("voice:beta") == "voice"
    assert row["channel_id"] is None                 # no room id on a single-user surface


# ── the where-line synthesis (person-keyed thread no longer encodes the room) ─

def test_surface_thread_for_phrase_rebuilds_channel_key_from_identity():
    ident = {"platform": "discord", "channel_kind": "guild", "channel_id": "555"}
    assert _surface_thread_for_phrase("person:xyz", ident) == "discord:guild:555"
    # no surface on identity (CLI/voice) -> the raw thread_id is used verbatim
    assert _surface_thread_for_phrase("voice:beta", {}) == "voice:beta"
    # and _channel_phrase then produces the familiar clickable-room phrasing
    assert "<#555>" in _channel_phrase(_surface_thread_for_phrase("person:xyz", ident), "general")


# ── the chat node: room block on public turns only ───────────────────────────

class RecordingModel(GenericFakeChatModel):
    """Fake that records the system prompt each invoke sees (prompt-shape tests)."""

    seen: list = []

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        type(self).seen.append(str(messages[0].content))
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)


def test_public_turn_injects_room_block_dm_does_not():
    RecordingModel.seen = []
    calls = []

    def room_fn(channel_id, channel):
        calls.append((channel_id, channel))
        return "Megan: anyone around?"

    m = RecordingModel(messages=iter([AIMessage(content="a"), AIMessage(content="b")]))
    graph = build_graph(m, soul="s", room_context_fn=room_fn)

    ask(graph, "hey", identity=PUBLIC_GUILD, thread_id="person:p1")   # public -> room block
    ask(graph, "hey", identity=PRIVATE_DM, thread_id="person:p2")     # DM -> no room block

    assert "Recent activity in this channel" in RecordingModel.seen[0]
    assert "Megan: anyone around?" in RecordingModel.seen[0]
    assert "Recent activity in this channel" not in RecordingModel.seen[1]
    # queried once (public only), with the raw channel_id + derived channel enum
    assert calls == [("555", "guild")]


def test_room_block_degrades_safe_when_fn_raises():
    RecordingModel.seen = []

    def boom(_cid, _ch):
        raise RuntimeError("NAS down")

    m = RecordingModel(messages=iter([AIMessage(content="a")]))
    graph = build_graph(m, soul="s", room_context_fn=boom)
    # a raising room fn must NOT kill the turn — the block is simply omitted
    out = ask(graph, "hey", identity=PUBLIC_GUILD, thread_id="person:p1")
    assert out == "a"
    assert "Recent activity in this channel" not in RecordingModel.seen[0]


def test_where_line_uses_identity_surface_for_person_keyed_thread():
    RecordingModel.seen = []
    m = RecordingModel(messages=iter([AIMessage(content="a")]))
    graph = build_graph(m, soul="s")
    # person-keyed thread_id, but the resolver carried the surface on identity
    ask(graph, "where am I", identity=PUBLIC_GUILD, thread_id="person:p1")
    assert "#general" in RecordingModel.seen[0]      # names the room despite 'person:' thread
    assert "<#555>" in RecordingModel.seen[0]        # clickable channel link


def test_room_context_fn_for_arming():
    import types

    from aerys_v2.factory import room_context_fn_for

    # None without a DB — the room feature is off (degrade-safe)
    assert room_context_fn_for(types.SimpleNamespace(database_url=None)) is None
    # a callable when database_url is set (does not connect until called)
    fn = room_context_fn_for(
        types.SimpleNamespace(database_url="postgresql://x@nas/aerys_v2", room_context_limit=50)
    )
    assert callable(fn)
