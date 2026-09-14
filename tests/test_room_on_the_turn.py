"""The room she was standing in, kept ON the turn she answered (board #17).

Chris, 2026-09-13, approving this shape: "middle works for me then you are approved
to work that."

His own words already follow him from room to room — his house thread is person-keyed,
so a public Discord message of his is already in the stick's window. What is missing is
the room AROUND him. When a line of his was a reply to something someone else said, the
stick sees his line without the thing it answered, and it reads as a non sequitur.

The obvious fix is to record every public channel message into a table. He raised the
three objections against that himself and they are all correct: it creates a retention
obligation over other people's messages, the channel is usually quiet so it buys
little, and if it got busy it would drown the rolling hundred. It is also the option I
turned down on 2026-09-13 morning when fixing room blindness — she reads the channel
live when summoned and stores nothing.

So: attach the room to HIS turn instead. She already reads the channel at summon time;
this keeps what she read, with the turn she read it for. Retention shrinks to a few
lines hanging off rows already kept under the existing policy; a quiet channel costs
nothing; and a busy one cannot drown anything, because this never creates rows — the
volume is bounded by how often he speaks to her, not by how loud the room is.
"""
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from aerys_v2.factory import build_graph
from aerys_v2.service import ask
from aerys_v2.turns import build_turn_row

PUBLIC = {"user_id": "person-1", "display_name": "Chris", "privacy_context": "public",
          "platform": "discord", "channel_kind": "guild", "channel_id": "555",
          "channel_name": "resonance"}
PRIVATE = {"user_id": "person-1", "display_name": "Chris", "privacy_context": "private",
           "platform": "discord", "channel_kind": "dm", "channel_id": "9"}

ROOM = "Stratus: is anything forming in the Atlantic\nChris: probably worth a look"


def fake(*replies):
    return GenericFakeChatModel(messages=iter([AIMessage(content=r) for r in replies]))


def recorded(identity, room_fn):
    rows = []
    graph = build_graph(fake("ok"), soul="s", room_context_fn=room_fn)
    ask(graph, "can you look?", identity=identity, thread_id="person:p1",
        record_turn=rows.append)
    return rows[0] if rows else None


# ── the row carries what she read ───────────────────────────────────────────

def test_a_public_turn_keeps_the_room_it_was_answered_in():
    row = recorded(PUBLIC, lambda cid, ch: ROOM)
    assert row is not None
    assert row["room_context"] is not None
    assert "is anything forming in the Atlantic" in row["room_context"]


def test_a_DM_turn_keeps_no_room():
    assert recorded(PRIVATE, lambda cid, ch: ROOM)["room_context"] is None


def test_an_empty_room_is_stored_as_nothing_not_as_an_empty_string():
    assert recorded(PUBLIC, lambda cid, ch: "")["room_context"] is None


def test_a_raising_room_fn_still_records_the_turn():
    def boom(_cid, _ch):
        raise RuntimeError("NAS down")

    row = recorded(PUBLIC, boom)
    assert row is not None and row["room_context"] is None, 'the turn matters more'


def test_it_stores_what_she_READ_not_the_heading_around_it():
    """The prompt wrapper is her framing for that turn. What belongs on the row is
    the room itself, so the stick can frame it its own way."""
    room = recorded(PUBLIC, lambda cid, ch: ROOM)["room_context"]
    assert "Recent activity in this channel" not in room
    assert room.strip() == ROOM


# ── the row builder ─────────────────────────────────────────────────────────

def test_build_turn_row_defaults_room_context_to_none():
    row = build_turn_row(thread_id="t", identity={}, input_text="hi", latency_ms=1)
    assert row["room_context"] is None, 'every existing caller keeps working'


def test_build_turn_row_bounds_a_huge_room():
    from aerys_v2.turns import ROOM_CONTEXT_LIMIT

    row = build_turn_row(thread_id="t", identity={}, input_text="hi", latency_ms=1,
                         room_context="x" * (ROOM_CONTEXT_LIMIT * 3))
    assert len(row["room_context"]) <= ROOM_CONTEXT_LIMIT


def test_clipping_never_leaves_half_a_line_attributed_to_someone():
    """A character cut makes the first line a fragment, and on the stick a fragment
    reads as something a real person said. One line fewer is cheaper than that.
    Adversarial review, 2026-09-14."""
    from aerys_v2.turns import ROOM_CONTEXT_LIMIT, build_turn_row

    lines = [f"Stratus: message number {n} with some words after it" for n in range(400)]
    row = build_turn_row(thread_id="t", identity={}, input_text="hi", latency_ms=1,
                         room_context="\n".join(lines))
    kept = row["room_context"]
    assert len(kept) <= ROOM_CONTEXT_LIMIT
    assert kept.splitlines()[0].startswith("Stratus: message number "), kept.splitlines()[0]
    assert kept.splitlines()[-1] == lines[-1], 'the newest line survives'


def test_the_shared_holder_is_load_bearing_and_this_test_is_its_canary():
    """The room reaches the audit row through a mutable dict on the per-call config,
    which works because LangGraph copies `configurable` SHALLOWLY. That is an
    assumption about a dependency, not a guarantee it makes. If an upgrade ever
    deep-copies, the feature degrades silently — the turn still records, just with no
    room — so the only thing that would catch it is a test that runs the REAL graph
    and looks at the REAL row. That is this file's first test, and this one says so
    out loud next to it."""
    row = recorded(PUBLIC, lambda cid, ch: ROOM)
    assert row["room_context"], 'if this fails, check whether LangGraph now deep-copies config'
