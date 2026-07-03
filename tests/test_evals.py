"""Offline tests for the eval harness — no API key, no network, no personal data.

GenericFakeChatModel plays BOTH roles: the target's brain (any reply) and the
judge (a reply that happens to be parseable rubric JSON). Same pin-the-node-output
trick as test_service.py. What these prove: cases load (including the
golden→example fallback CI relies on), LocalGraphTarget round-trips through the
real graph, the judge parses/defaults/zeroes exactly like n8n's Parse Score node,
and summarize() does the Format Report math correctly.
"""

import json

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from aerys_v2.evals.runner import (
    EvalCase,
    Judge,
    LocalGraphTarget,
    N8nBaselineTarget,
    load_cases,
    run_eval,
    summarize,
)
from aerys_v2.factory import build_graph

CASE = EvalCase(
    id="tc-x",
    input="hello?",
    persona_expectations="warm greeting",
    category="normal_conversation",
)


def fake_model(*replies: str) -> GenericFakeChatModel:
    return GenericFakeChatModel(messages=iter([AIMessage(content=r) for r in replies]))


def write_cases(path, cases: list[dict]) -> None:
    path.write_text(json.dumps(cases), encoding="utf-8")


# --- load_cases: the Load Dataset node -------------------------------------


def test_repo_example_json_loads():
    # The committed sanitized case must always parse — it's all CI ever sees.
    # Force the fallback by pointing at a dir where golden.json may or may not
    # exist; either file must yield valid EvalCase objects.
    cases = load_cases()
    assert len(cases) >= 1
    assert all(isinstance(c, EvalCase) for c in cases)
    # id/category always present; input may legitimately be empty (tc-19 in the
    # golden set is an empty-message edge case), so don't assert on it.
    assert all(c.id and c.category for c in cases)
    assert len({c.id for c in cases}) == len(cases)  # ids are unique


def test_fallback_to_example_when_golden_absent(tmp_path):
    # Fresh-clone situation: golden.json is gitignored (personal data), so only
    # example.json exists — the loader must quietly use it instead of crashing.
    write_cases(tmp_path / "example.json", [
        {"id": "e1", "input": "hi", "persona_expectations": "warm", "category": "normal_conversation"},
    ])
    cases = load_cases(tmp_path)
    assert [c.id for c in cases] == ["e1"]


def test_golden_preferred_over_example(tmp_path):
    # When both exist (the owner's machine), golden wins — example is the stand-in.
    write_cases(tmp_path / "example.json", [
        {"id": "e1", "input": "hi", "persona_expectations": "x", "category": "edge_case"},
    ])
    write_cases(tmp_path / "golden.json", [
        {"id": "g1", "input": "hi", "persona_expectations": "x", "category": "edge_case"},
        {"id": "g2", "input": "yo", "persona_expectations": "x", "category": "research"},
    ])
    cases = load_cases(tmp_path)
    assert [c.id for c in cases] == ["g1", "g2"]


# --- LocalGraphTarget: the Execute Workflow hop -----------------------------


def test_local_graph_target_round_trips():
    graph = build_graph(fake_model("hey Chris!"), soul="test soul")
    response = LocalGraphTarget(graph).respond(CASE)
    assert response.reply == "hey Chris!"
    assert response.latency_ms >= 0  # the Date.now() bracket actually measured


def test_local_graph_target_isolates_threads():
    # Two cases through one target must NOT share history — each case gets its
    # own thread_id, so each thread holds exactly its own human+ai pair.
    graph = build_graph(fake_model("a", "b"), soul="test soul")
    target = LocalGraphTarget(graph)
    target.respond(EvalCase(id="c1", input="one", persona_expectations="x", category="edge_case"))
    target.respond(EvalCase(id="c2", input="two", persona_expectations="x", category="edge_case"))
    for tid in ("eval-c1", "eval-c2"):
        state = graph.get_state({"configurable": {"thread_id": tid}})
        assert len(state.values["messages"]) == 2


def test_local_graph_target_empty_input_becomes_judgeable_reply():
    # tc-19 in the golden set sends an empty message. ask() rejects it at the
    # seam (ValueError) — the target must surface that rejection as a reply for
    # the judge to score, NOT crash and NOT get score-0'd as an infra failure.
    graph = build_graph(fake_model("never used"), soul="test soul")
    case = EvalCase(id="c-empty", input="", persona_expectations="graceful", category="edge_case")
    response = LocalGraphTarget(graph).respond(case)
    assert "rejected at the ask() seam" in response.reply


def test_n8n_baseline_target_is_a_stub():
    with pytest.raises(NotImplementedError):
        N8nBaselineTarget().respond(CASE)


# --- Judge: Build Judge Request + Parse Score --------------------------------


def test_judge_parses_clean_json():
    judge = Judge(fake_model('{"reasoning": "warm and on-voice", "score": 5}'))
    verdict = judge.score(CASE, "hello there, Chris!")
    assert verdict == {"score": 5, "reasoning": "warm and on-voice"}


def test_judge_strips_markdown_fences():
    # Models fence JSON in markdown even when told "JSON only" — the n8n Parse
    # Score node stripped ```json fences before JSON.parse; so do we.
    judge = Judge(fake_model('```json\n{"reasoning": "ok", "score": 4}\n```'))
    assert judge.score(CASE, "hi")["score"] == 4


def test_judge_parse_failure_defaults_to_3():
    # Bit-for-bit n8n semantics: garbage judge output → 3, never a crash.
    judge = Judge(fake_model("I refuse to emit JSON today"))
    verdict = judge.score(CASE, "hi")
    assert verdict["score"] == 3
    assert "Parse error" in verdict["reasoning"]


def test_judge_out_of_range_score_defaults_to_3():
    judge = Judge(fake_model('{"reasoning": "over-enthusiastic", "score": 11}'))
    assert judge.score(CASE, "hi")["score"] == 3


def test_judge_call_failure_scores_0():
    # n8n semantics: HTTP errors scored 0 — 0 means "infrastructure broke",
    # never "the answer was bad". An exhausted fake model raises on invoke.
    judge = Judge(fake_model())  # no messages left → invoke raises
    verdict = judge.score(CASE, "hi")
    assert verdict["score"] == 0
    assert "failed" in verdict["reasoning"].lower()


# --- run_eval + summarize: the loop and Format Report ------------------------


def make_results():
    # Hand-built per-case results with known math: normal avg (5+4)/2 = 4.5,
    # research avg 3.0, overall (5+4+3)/3 = 4.0.
    return [
        {"id": "a", "category": "normal_conversation", "input": "x", "reply": "r",
         "latency_ms": 100.0, "score": 5, "reasoning": ""},
        {"id": "b", "category": "normal_conversation", "input": "x", "reply": "r",
         "latency_ms": 300.0, "score": 4, "reasoning": ""},
        {"id": "c", "category": "research", "input": "x", "reply": "r",
         "latency_ms": 200.0, "score": 3, "reasoning": ""},
    ]


def test_summary_math():
    summary = summarize(make_results())
    assert summary["cases"] == 3
    assert summary["overall"] == {"avg": 4.0, "min": 3, "max": 5}
    normal = summary["by_category"]["normal_conversation"]
    assert normal == {"count": 2, "avg": 4.5, "min": 4, "max": 5, "avg_latency_ms": 200.0}
    assert summary["by_category"]["research"]["count"] == 1
    assert summary["avg_latency_ms"] == 200.0


def test_summary_empty():
    assert summarize([]) == {"cases": 0, "overall": None, "by_category": {}}


def test_format_summary_table_renders():
    # The CLI's --eval print path must not crash on a real summary shape and
    # must show every category plus the overall roll-up line.
    from aerys_v2.evals.runner import format_summary_table

    table = format_summary_table(summarize(make_results()))
    assert "normal_conversation" in table
    assert "research" in table
    assert "OVERALL" in table
    assert format_summary_table(summarize([])) == "No cases were run."


def test_run_eval_end_to_end_offline():
    # Full pipeline with fakes on both ends: 2 cases → 2 target replies →
    # 2 judge verdicts → results + summary. This is the whole n8n eval loop
    # running in-process with zero network.
    cases = [
        EvalCase(id="c1", input="hi", persona_expectations="warm", category="normal_conversation"),
        EvalCase(id="c2", input="search this", persona_expectations="cites", category="research"),
    ]
    graph = build_graph(fake_model("reply one", "reply two"), soul="test soul")
    judge = Judge(fake_model(
        '{"reasoning": "great", "score": 5}',
        '{"reasoning": "meh", "score": 3}',
    ))
    results, summary = run_eval(LocalGraphTarget(graph), cases, judge)

    assert [r["score"] for r in results] == [5, 3]
    assert results[0]["reply"] == "reply one"
    assert summary["cases"] == 2
    assert summary["overall"]["avg"] == 4.0
    assert set(summary["by_category"]) == {"normal_conversation", "research"}


def test_run_eval_target_failure_scores_0():
    # A dead target must produce a 0-scored result, not abort the run —
    # the report should show broken plumbing, not hide it.
    cases = [CASE]
    judge = Judge(fake_model('{"reasoning": "unused", "score": 5}'))
    results, summary = run_eval(N8nBaselineTarget(), cases, judge)
    assert results[0]["score"] == 0
    assert "Target failed" in results[0]["reasoning"]
    assert summary["overall"]["avg"] == 0
