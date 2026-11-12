"""The replay harness itself (board #176): synthetic rows only, never her real turns."""

from __future__ import annotations

import pytest

pytest.importorskip("langchain.agents")

from langchain_core.load import dumpd
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

import aerys_v2.factory as factory
from aerys_v2.replay_traces import compare, turn_from_row


@pytest.fixture(autouse=True)
def fixed_clock(monkeypatch):
    monkeypatch.setattr(factory, "_where_when_line", lambda thread, identity: "\nIt is noon.")


def row(passes, **kw):
    return {"id": 1, "channel": "person", "intent": "action", "person_id": "u1", "display_name": "Chris",
            "passes": [{"input": [dumpd(m) for m in i], "output": dumpd(o), "model": "claude-sonnet-5-5"}
                       for i, o in passes], **kw}


SYS = SystemMessage(content="recorded system prompt")
ASK = HumanMessage(content="what's the weather")
CALL = AIMessage(content="", tool_calls=[{"id": "t1", "name": "get_weather", "args": {"when": "today"}}])
RESULT = ToolMessage(content="Sunny, 82.", tool_call_id="t1", name="get_weather")
DONE = AIMessage(content="Sunny and 82 today.")


def test_a_two_pass_turn_replays_with_fidelity():
    report = compare(turn_from_row(row([([SYS, ASK], CALL), ([SYS, ASK, CALL, RESULT], DONE)])))
    assert report["fidelity"], report["fidelity_diffs"]
    assert [p["slot"] for p in report["replay_log"]] == ["std", "std"]


def test_the_honesty_gate_retry_is_replayed_from_its_recorded_input():
    claim = AIMessage(content="Done, it's sunny.")
    correction = HumanMessage(content="(you did not call a tool)")
    report = compare(turn_from_row(row([([SYS, ASK], claim), ([SYS, ASK, claim, correction], CALL),
                                        ([SYS, ASK, claim, correction, CALL, RESULT], DONE)])))
    assert report["fidelity"]
    assert len(report["replay_log"]) == 3


def test_a_record_the_graph_could_not_have_produced_fails_fidelity():
    stray = HumanMessage(content="a line the graph never wrote")
    report = compare(turn_from_row(row([([SYS, ASK], CALL), ([SYS, ASK, CALL, RESULT, stray], DONE)])))
    assert not report["fidelity"]
    assert report["fidelity_diffs"] == ["pass 2: history differs from the record"]


def test_a_second_run_the_gate_would_not_have_made_fails_fidelity():
    # The record continues after a finished run, but not as "this run + one correction".
    other = HumanMessage(content="something else entirely")
    report = compare(turn_from_row(row([([SYS, ASK], DONE), ([SYS, other], DONE)])))
    assert not report["fidelity"]
    assert any("without a gate retry" in d for d in report["fidelity_diffs"])


def test_an_error_result_replays_as_an_error():
    failed = ToolMessage(content="Error: HA unreachable", tool_call_id="t1", name="get_weather", status="error")
    report = compare(turn_from_row(row([([SYS, ASK], CALL), ([SYS, ASK, CALL, failed], DONE)])))
    assert report["fidelity"]
    tool_seen = [m for m in report["replay_log"][1]["messages"] if m[0] == "tool"]
    assert tool_seen[0][5] == "error"


def test_a_call_the_record_does_not_have_fails_fidelity():
    other_args = AIMessage(content="", tool_calls=[{"id": "t1", "name": "get_weather", "args": {"when": "tomorrow"}}])
    rec = row([([SYS, ASK], CALL), ([SYS, ASK, CALL, RESULT], DONE)])
    turn = turn_from_row(rec)
    turn.outputs[0] = other_args           # the loop now asks for something never recorded
    report = compare(turn)
    assert not report["fidelity"]


def test_cli_reports_only_fidelity_and_show_opts_into_content(tmp_path, capsys):
    import json
    from aerys_v2.replay_traces import main
    corpus = tmp_path / "synthetic.jsonl"
    corpus.write_text(json.dumps(row([([SYS, ASK], CALL), ([SYS, ASK, CALL, RESULT], DONE)])) + "\n")
    assert main([str(corpus)]) == 0
    report = capsys.readouterr().out
    assert "1 real action turns, 2 model passes" in report
    assert "fidelity (agent loop replays the record)" in report
    assert "parity" not in report and ASK.content not in report and RESULT.content not in report
    assert main([str(corpus), "--show", "1"]) == 0
    assert ASK.content in capsys.readouterr().out


def test_cli_load_failure_is_content_free(tmp_path, capsys):
    import json
    from aerys_v2.replay_traces import main
    corpus = tmp_path / "invalid.jsonl"
    corpus.write_text(json.dumps({"id": 2, "passes": [{"input": ["PRIVATE_SENTINEL"]}]}) + "\n")
    assert main([str(corpus)]) == 1
    report = capsys.readouterr().out
    assert "load failed:" in report and "PRIVATE_SENTINEL" not in report
