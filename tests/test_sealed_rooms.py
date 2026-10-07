"""Sealed rooms: a guild channel whose conversation never leaves it.

Her privacy model has two rooms: a 1:1 DM is private, every shared room is public,
and privacy is judged by CONTENT (health, money, relationships) rather than by room.
That leaves a gap for a channel that is private by MEMBERSHIP: a locked channel
someone uses for something everyday-sounding but sensitive (a job search, say) is
just another public room to her, and because a person has ONE thread across every
surface, what they said there could come up in the next shared room they talk to her in.

A sealed room closes that structurally, the same way the DM gate does:
  - every turn said in a sealed room is stamped with that room at write time;
  - in any OTHER shared room, those turns (and her replies to them) are dropped from
    what the model sees; in the sealed room itself, and in the person's own 1:1
    surfaces, they stay;
  - memories extracted from a sealed room are always 'private', whatever the
    extraction model thinks of the content, and are never batched with unsealed turns.
"""

from __future__ import annotations

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage

from aerys_v2.factory import build_graph
from aerys_v2.service import ask
from aerys_v2.services.content_privacy import (
    CONTENT_PRIVACY_KEY,
    SEALED_ROOM_KEY,
    gate_for_room,
    redact_sealed_history,
    sealed_room_of,
)

SEALED = "777"
OTHER = "555"

SEALED_ROOM = {
    "user_id": "person-2", "display_name": "Guest", "privacy_context": "public",
    "platform": "discord", "channel_kind": "guild", "channel_id": SEALED,
    "sealed_room": SEALED,
}
OTHER_ROOM = {
    "user_id": "person-2", "display_name": "Guest", "privacy_context": "public",
    "platform": "discord", "channel_kind": "guild", "channel_id": OTHER,
}
THEIR_DM = {
    "user_id": "person-2", "display_name": "Guest", "privacy_context": "private",
    "platform": "discord", "channel_kind": "dm", "channel_id": "999",
}


def _human(content: str, *, privacy: str = "public", sealed: str = "") -> HumanMessage:
    kw = {CONTENT_PRIVACY_KEY: privacy}
    if sealed:
        kw[SEALED_ROOM_KEY] = sealed
    return HumanMessage(content=content, additional_kwargs=kw)


# ── the pure gate ────────────────────────────────────────────────────────────

def test_sealed_turn_and_its_reply_drop_in_another_room():
    history = [
        _human("sealed question", sealed=SEALED), AIMessage(content="sealed answer"),
        _human("ordinary hello"), AIMessage(content="hi back"),
        _human("current turn"),
    ]
    kept = [m.content for m in redact_sealed_history(history, current_room=OTHER)]
    assert kept == ["ordinary hello", "hi back", "current turn"]


def test_sealed_turn_stays_in_its_own_room():
    history = [_human("sealed question", sealed=SEALED), AIMessage(content="sealed answer"),
               _human("current turn", sealed=SEALED)]
    kept = [m.content for m in redact_sealed_history(history, current_room=SEALED)]
    assert kept == ["sealed question", "sealed answer", "current turn"]


def test_one_sealed_room_never_sees_another():
    history = [_human("from room A", sealed="111"), AIMessage(content="a"),
               _human("current", sealed=SEALED)]
    kept = [m.content for m in redact_sealed_history(history, current_room=SEALED)]
    assert kept == ["current"]


def test_the_current_turn_is_always_kept():
    history = [_human("said just now", sealed=SEALED)]
    assert [m.content for m in redact_sealed_history(history, current_room=OTHER)] == ["said just now"]


def test_gate_for_room_composes_both_gates():
    history = [
        _human("a private dm line", privacy="private"), AIMessage(content="dm reply"),
        _human("sealed line", sealed=SEALED), AIMessage(content="sealed reply"),
        _human("public line"), AIMessage(content="public reply"),
        _human("current"),
    ]
    in_other = [m.content for m in gate_for_room(history, OTHER_ROOM)]
    assert in_other == ["public line", "public reply", "current"]
    in_sealed = [m.content for m in gate_for_room(history, SEALED_ROOM)]
    assert in_sealed == ["sealed line", "sealed reply", "public line", "public reply", "current"], \
        "the sealed room is still a shared room: private DM content stays out"
    in_dm = [m.content for m in gate_for_room(history, THEIR_DM)]
    assert in_dm == [m.content for m in history], "their own 1:1 surface sees everything"


def test_an_unknown_room_context_still_drops_sealed_turns():
    history = [_human("sealed line", sealed=SEALED), AIMessage(content="r"), _human("current")]
    kept = [m.content for m in gate_for_room(history, {"user_id": "person-2"})]
    assert kept == ["current"]


# ── write side: ask() stamps the room ────────────────────────────────────────

def fake_model(*replies: str) -> GenericFakeChatModel:
    return GenericFakeChatModel(messages=iter([AIMessage(content=r) for r in replies]))


def _human_on_thread(graph, thread_id: str, needle: str):
    state = graph.get_state({"configurable": {"thread_id": thread_id}})
    return next(m for m in state.values["messages"]
                if getattr(m, "type", "") == "human" and needle in str(m.content))


def test_a_turn_in_a_sealed_room_is_stamped_with_it():
    graph = build_graph(fake_model("ok", "ok"), soul="s")
    ask(graph, "sealed words", identity=SEALED_ROOM, thread_id="person:p2")
    ask(graph, "open words", identity=OTHER_ROOM, thread_id="person:p2")
    assert sealed_room_of(_human_on_thread(graph, "person:p2", "sealed words")) == SEALED
    assert sealed_room_of(_human_on_thread(graph, "person:p2", "open words")) == ""


class CapturingModel(GenericFakeChatModel):
    seen: list = []

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        type(self).seen.append(
            [str(m.content) for m in messages if getattr(m, "type", "") != "system"]
        )
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)


def test_what_is_said_in_a_sealed_room_stays_there():
    CapturingModel.seen = []
    model = CapturingModel(messages=iter([AIMessage(content=r) for r in
                                          ("noted-1", "hello-2", "back-3", "dm-4")]))
    graph = build_graph(model, soul="s")
    ask(graph, "codeword pelican", identity=SEALED_ROOM, thread_id="person:p2")
    ask(graph, "hi everyone", identity=OTHER_ROOM, thread_id="person:p2")
    ask(graph, "back again", identity=SEALED_ROOM, thread_id="person:p2")
    ask(graph, "just us", identity=THEIR_DM, thread_id="person:p2")
    in_other, in_sealed, in_dm = CapturingModel.seen[1], CapturingModel.seen[2], CapturingModel.seen[3]
    assert not any("pelican" in s for s in in_other)
    assert "noted-1" not in in_other, "her reply to the sealed turn is gone too"
    assert any("hi everyone" in s for s in in_other)
    assert any("pelican" in s for s in in_sealed)
    assert any("pelican" in s for s in in_dm)


def test_the_action_path_seeds_through_the_same_gate():
    """The action graph has no checkpointer; _action_history_seed hands it the prior
    turns, and both of its branches (the folded specialist seed and the windowed
    exchange) must pass through the room gate."""
    from aerys_v2.service import _action_history_seed

    graph = build_graph(fake_model("noted-1", "hello-2"), soul="s")
    ask(graph, "codeword pelican", identity=SEALED_ROOM, thread_id="person:p2")
    ask(graph, "the lamp by the window", identity=OTHER_ROOM, thread_id="person:p2")

    def seed(identity, *, specialist):
        configurable = {"thread_id": "person:p2", "identity": identity}
        return " ".join(str(m.content) for m in
                        _action_history_seed(graph, configurable, "turn it on", specialist=specialist))

    for specialist in (True, False):
        elsewhere = seed(OTHER_ROOM, specialist=specialist)
        assert "pelican" not in elsewhere and "lamp by the window" in elsewhere, specialist
        assert "pelican" in seed(SEALED_ROOM, specialist=specialist), specialist
        assert "pelican" in seed(THEIR_DM, specialist=specialist), specialist


def test_a_retag_never_strips_the_seal():
    from aerys_v2.service import _human_turn

    m = _human_turn("words", "public", "id-1", sealed_room=SEALED)
    assert sealed_room_of(m) == SEALED and m.additional_kwargs[CONTENT_PRIVACY_KEY] == "public"


# ── transport: the gateway stamps the room ───────────────────────────────────

def test_the_gateway_stamps_only_sealed_channels():
    from aerys_v2.transports.discord_gateway import sealed_room_for

    sealed = frozenset({777})
    assert sealed_room_for(777, sealed) == "777"
    assert sealed_room_for(555, sealed) == ""
    assert sealed_room_for(777, frozenset()) == ""


def test_a_thread_inside_a_sealed_channel_is_sealed_as_that_channel():
    """Gemini (review): a Discord thread or forum post has its own channel id. It
    inherits its parent's seal and records the parent, the room people were let into."""
    from aerys_v2.transports.discord_gateway import sealed_room_for

    sealed = frozenset({777})
    assert sealed_room_for(888, sealed, parent_id=777) == "777"
    assert sealed_room_for(888, sealed, parent_id=555) == ""
    assert sealed_room_for(888, sealed, parent_id=None) == ""


def test_the_turn_row_records_the_sealed_room():
    from aerys_v2.turns import INSERT_TURN_SQL, build_turn_row

    def row_for(identity):
        return build_turn_row(thread_id="person:p2", identity=identity, input_text="x",
                              raw_reply="y", emitted_reply="y", latency_ms=1)

    assert row_for(SEALED_ROOM)["sealed_room"] == SEALED
    assert row_for(OTHER_ROOM)["sealed_room"] is None
    assert "%(sealed_room)s" in INSERT_TURN_SQL


def test_the_migration_adds_the_column():
    from pathlib import Path

    sql = (Path(__file__).resolve().parents[1] / "db" / "migrations"
           / "012_sealed_room_on_the_turn.sql").read_text()
    assert "ADD COLUMN IF NOT EXISTS sealed_room TEXT" in sql


def test_settings_carry_sealed_channels_off_by_default():
    from aerys_v2.config import Settings

    assert Settings.model_fields["discord_sealed_channel_ids"].default == ""


# ── long-term memory: sealed turns extract as private, never mixed ───────────

def _row(pid: str, content: str, channel_id: str | None, platform: str = "guild",
         sealed_room: str | None = None) -> dict:
    from datetime import datetime, timezone

    return {
        "id": content, "person_id": pid, "content": content, "source_platform": platform,
        "privacy_level": "public" if platform == "guild" else "private",
        "created_at": datetime(2026, 10, 7, tzinfo=timezone.utc),
        "created_at_raw": "2026-10-07", "speaker_name": "Unknown",
        "source_thread": "person:x", "channel_id": channel_id, "sealed_room": sealed_room,
    }


def test_source_queries_carry_the_channel():
    from aerys_v2.workers import extraction

    assert "channel_id" in extraction.SOURCE_COLUMNS
    assert "AS channel_id" in extraction.V2_TURNS_SQL or "t.channel_id" in extraction.V2_TURNS_SQL
    assert "NULL::text AS channel_id" in extraction.PROD_MESSAGES_SQL
    assert "sealed_room" in extraction.SOURCE_COLUMNS
    assert "t.sealed_room AS sealed_room" in extraction.V2_TURNS_SQL
    assert "NULL::text AS sealed_room" in extraction.PROD_MESSAGES_SQL


def test_sealed_rows_extract_private_and_batch_alone():
    from aerys_v2.workers.extraction import group_by_person, seal_rows

    rows = seal_rows([_row("p2", "sealed fact", SEALED), _row("p2", "open fact", OTHER),
                      _row("p2", "v1 row", None, platform="discord")],
                     frozenset({SEALED}))
    groups = group_by_person(rows)
    sealed = [g for g in groups if g.get("sealed")]
    assert len(sealed) == 1
    assert [m["content"] for m in sealed[0]["messages"]] == ["sealed fact"]
    assert sealed[0]["privacy_level"] == "private"
    open_groups = [g for g in groups if not g.get("sealed")]
    assert all("sealed fact" not in [m["content"] for m in g["messages"]] for g in open_groups)


def test_a_turn_stamped_at_ingest_is_sealed_even_when_its_channel_is_not_listed():
    """A thread under a sealed channel has its own channel id; the stamp on the row
    (written from the room the gateway sealed) is what makes it sealed."""
    from aerys_v2.workers.extraction import seal_rows

    (row,) = seal_rows([_row("p2", "thread fact", "888", sealed_room=SEALED)], frozenset())
    assert row["sealed"] is True and row["privacy_level"] == "private"


def test_the_extraction_model_cannot_loosen_a_sealed_memory():
    from aerys_v2.workers.extraction import observation_privacy

    assert observation_privacy({"privacy_level": "public"}, {"sealed": True, "privacy_level": "private"}) == "private"
    assert observation_privacy({"privacy_level": "public"}, {"privacy_level": "public"}) == "public"
    assert observation_privacy({}, {"privacy_level": "private"}) == "private"
    assert observation_privacy({"privacy_level": "private"}, {"privacy_level": "public"}) == "private"
