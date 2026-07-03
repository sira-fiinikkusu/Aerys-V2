"""Offline tests for the replay harness — no API key, no network, no personal data.

GenericFakeChatModel stands in for the brain (same pin-the-node-output trick as
test_service.py / test_evals.py). What these prove: payloads load (including the
captured→example fallback CI relies on), the n8n→ask() mapping is correct
(person_id becomes Identity.user_id, thread ids live in the "replay:" namespace),
one bad payload doesn't kill the run, and summarize_replay does its math right.
"""

import json

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from aerys_v2.replay import (
    ReplayPayload,
    build_replay_graph,
    format_replay_summary,
    load_payloads,
    run_replay,
    summarize_replay,
    to_ask_inputs,
)


def fake_model(*replies: str) -> GenericFakeChatModel:
    return GenericFakeChatModel(messages=iter([AIMessage(content=r) for r in replies]))


def make_record(id: str = "replay-t1", channel: str = "dm", **payload_overrides) -> ReplayPayload:
    # Minimal DM-shaped payload; tests override fields to build voice/bad shapes.
    payload = {
        "source_channel": "discord",
        "person_id": "person-uuid-1",
        "user_id": "5551212",
        "username": "chris",
        "display_name": "Chris",
        "message_text": "hello there",
        "session_key": "person-uuid-1",
        "channel_id": "999",
    }
    payload.update(payload_overrides)
    return ReplayPayload(
        id=id, channel=channel, captured_at="2026-07-01T00:00:00.000Z",
        source_execution="15000", real_text=False, payload=payload,
    )


def write_payloads(path, records: list[dict]) -> None:
    path.write_text(json.dumps(records), encoding="utf-8")


RAW = {
    "id": "replay-r1", "channel": "voice", "captured_at": "2026-07-01T00:00:00.000Z",
    "source_execution": 15611,  # captured file stores ints — loader must normalize
    "real_text": True,
    "payload": {"message_text": "Play music.", "person_id": "p1", "display_name": "Chris"},
}


# --- load_payloads: the capture loader ---------------------------------------


def test_repo_example_payloads_load():
    # The committed synthetic examples must always parse — it's all CI ever sees.
    records = load_payloads()
    assert len(records) >= 1
    assert all(isinstance(r, ReplayPayload) for r in records)
    assert all(r.id and r.channel in ("dm", "voice", "guild", "telegram") for r in records)
    assert len({r.id for r in records}) == len(records)  # ids are unique


def test_fallback_to_example_when_capture_absent(tmp_path):
    # Fresh-clone situation: payloads.json is gitignored (owner traffic), so only
    # example_payload.json exists — the loader must quietly use it.
    write_payloads(tmp_path / "example_payload.json", [RAW])
    records = load_payloads(tmp_path)
    assert [r.id for r in records] == ["replay-r1"]
    assert records[0].source_execution == "15611"  # int normalized to string


def test_capture_preferred_over_example(tmp_path):
    # On the owner's machine both exist — the real capture wins.
    write_payloads(tmp_path / "example_payload.json", [RAW])
    write_payloads(tmp_path / "payloads.json", [
        dict(RAW, id="replay-c1"), dict(RAW, id="replay-c2"),
    ])
    assert [r.id for r in load_payloads(tmp_path)] == ["replay-c1", "replay-c2"]


# --- to_ask_inputs: the mapping seam ------------------------------------------


def test_mapping_person_id_becomes_identity():
    # person_id is what the V1 Identity Resolver stamped — it must become
    # Identity.user_id, with display_name riding along.
    text, identity, thread_id = to_ask_inputs(make_record())
    assert text == "hello there"
    assert identity == {"user_id": "person-uuid-1", "display_name": "Chris"}


def test_mapping_falls_back_to_platform_user_id():
    # A payload predating identity resolution has no person_id — the raw
    # platform user_id is the fallback, username covers display_name.
    record = make_record(person_id=None, display_name=None)
    _, identity, _ = to_ask_inputs(record)
    assert identity == {"user_id": "5551212", "display_name": "chris"}


def test_mapping_thread_id_is_replay_namespaced():
    # The isolation rule: thread ids come from the payload's OWN id under the
    # "replay:" namespace — never from session_key/channel_id, so no replay can
    # collide with a live thread key like "discord:dm:<snowflake>".
    _, _, thread_id = to_ask_inputs(make_record(id="replay-042"))
    assert thread_id == "replay:replay-042"


def test_mapping_voice_message_content_fallback():
    # The Voice Adapter set both message_text and message_content from the same
    # STT — if a capture only carried message_content, the mapping still works.
    record = make_record(channel="voice", message_text=None, message_content="Play music.")
    text, _, _ = to_ask_inputs(record)
    assert text == "Play music."


# --- run_replay: the loop, isolation, and summary ------------------------------


def test_run_replay_happy_path():
    graph = build_replay_graph(fake_model("reply one", "reply two"), soul="test soul")
    records = [make_record(id="replay-1"), make_record(id="replay-2", channel="voice")]
    results, summary = run_replay(graph, records)

    assert [r["ok"] for r in results] == [True, True]
    assert results[0]["reply_len"] == len("reply one")
    assert results[0]["error"] is None
    assert all(r["latency_ms"] >= 0 for r in results)
    assert summary["payloads"] == 2 and summary["ok"] == 2 and summary["failed"] == 0


def test_run_replay_threads_are_isolated_and_namespaced():
    # Each payload lands in its own "replay:" thread on the throwaway
    # InMemorySaver — exactly one human+ai pair per thread, nothing shared.
    graph = build_replay_graph(fake_model("a", "b"), soul="test soul")
    records = [make_record(id="replay-1"), make_record(id="replay-2")]
    run_replay(graph, records)
    for rid in ("replay-1", "replay-2"):
        state = graph.get_state({"configurable": {"thread_id": f"replay:{rid}"}})
        assert len(state.values["messages"]) == 2


def test_run_replay_one_bad_payload_does_not_kill_the_run():
    # An empty message_text trips ask()'s non-empty rail (ValueError). The run
    # must record that payload as ok=False and still replay the others.
    graph = build_replay_graph(fake_model("fine", "also fine"), soul="test soul")
    records = [
        make_record(id="replay-good-1"),
        make_record(id="replay-bad", message_text="   "),
        make_record(id="replay-good-2"),
    ]
    results, summary = run_replay(graph, records)

    assert [r["ok"] for r in results] == [True, False, True]
    bad = results[1]
    assert bad["reply_len"] == 0
    assert "ValueError" in bad["error"]
    assert summary["ok"] == 2 and summary["failed"] == 1


def test_summarize_replay_math():
    results = [
        {"id": "a", "channel": "dm", "ok": True, "reply_len": 5, "latency_ms": 100.0, "error": None},
        {"id": "b", "channel": "dm", "ok": False, "reply_len": 0, "latency_ms": 300.0, "error": "x"},
        {"id": "c", "channel": "voice", "ok": True, "reply_len": 9, "latency_ms": 200.0, "error": None},
    ]
    summary = summarize_replay(results)
    assert summary["payloads"] == 3
    assert summary["ok"] == 2 and summary["failed"] == 1
    assert summary["by_channel"]["dm"] == {
        "count": 2, "ok": 1, "failed": 1, "avg_latency_ms": 200.0,
    }
    assert summary["by_channel"]["voice"]["count"] == 1
    assert summary["avg_latency_ms"] == 200.0


def test_summarize_replay_empty():
    assert summarize_replay([]) == {
        "payloads": 0, "ok": 0, "failed": 0, "by_channel": {}, "avg_latency_ms": 0.0,
    }


def test_format_replay_summary_renders():
    # The CLI's --replay print path must not crash and must show every channel
    # plus the total roll-up line.
    results, summary = run_replay(
        build_replay_graph(fake_model("hi", "yo"), soul="test soul"),
        [make_record(id="replay-1"), make_record(id="replay-2", channel="voice")],
    )
    table = format_replay_summary(summary)
    assert "dm" in table and "voice" in table and "TOTAL" in table
    assert format_replay_summary(summarize_replay([])) == "No payloads were replayed."
