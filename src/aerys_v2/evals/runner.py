"""Eval runner — load cases, run a target, judge the replies, summarize.

n8n mapping: this file IS the "06-01 Eval Suite" workflow, but with the moving parts
separated so each one is testable offline:

- In n8n the dataset lived OUTSIDE the workflow (a Code node did
  require('fs').readFileSync('/home/node/aerys-evals/baseline.json') off a docker
  volume). Here load_cases() reads evals/cases/golden.json the same way — data stays
  out of the code, and out of git (golden.json is gitignored; only the sanitized
  example.json is committed, so CI still has one case to chew on).
- The "Execute Workflow → Core Agent" hop becomes the Target protocol: anything with
  a respond() method can be evaluated. LocalGraphTarget wraps our real graph.
- "Build Judge Request" + "Parse Score" become the Judge class. Same rubric, same
  1-5 scale, same failure semantics (parse failure → 3, call failure → 0).
- "Format Report" becomes summarize() — per-category avg/min/max plus latency,
  the same numbers the staticData accumulator emitted, minus the staticData.
"""

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from aerys_v2.config import Settings
from aerys_v2.service import ask
from aerys_v2.state import Identity

# ---------------------------------------------------------------------------
# The judge rubric — extracted VERBATIM from the n8n "Build Judge Request" Code
# node. It lives in code (not read from evals/cases/judge_rubric.md) because that
# directory is gitignored apart from example.json; embedding it means the harness
# works on a fresh clone with zero personal data present.
# ---------------------------------------------------------------------------

JUDGE_SYSTEM_PROMPT = """You are an expert evaluator for an AI assistant named Aerys. Given a user message, the expected behavior criteria, and Aerys's actual response, score the response 1-5:

5 = Perfect: accurate, natural, in Aerys's voice, directly addresses the question, meets all expected behavior criteria
4 = Good: mostly correct, minor issues with tone or completeness
3 = Acceptable: correct but mechanical, missing personality, or slightly off-topic
2 = Poor: partially wrong, ignores the question, or very out of character
1 = Fail: wrong, harmful, complete non-answer, or error message

Respond with JSON only:
{"reasoning": "<1-2 sentence analysis>", "score": <1-5>}"""

JUDGE_USER_TEMPLATE = """User message: {user_message}

Expected behavior: {expected_behavior}

Aerys response: {agent_response}"""

# The n8n harness ran every case as a private DM from the owner (context was
# {"person_id": "6e6bcbed-..."} on all 25 cases). The V2 equivalent of "owner DM"
# is this identity riding the configurable channel — see service.ask().
EVAL_IDENTITY: Identity = {"user_id": "eval-harness", "display_name": "Chris"}


# ---------------------------------------------------------------------------
# Cases — the "Load Dataset" node
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EvalCase:
    """One row of the dataset, in the normalized schema.

    Mirrors the JSON in evals/cases/*.json: the n8n source schema
    {id, user_message, expected_behavior, context, category} was normalized to
    {id, input, persona_expectations, category, notes} at extraction time.
    """

    id: str
    input: str
    persona_expectations: str
    category: str
    notes: str = ""


def default_cases_dir() -> Path:
    """Repo-root evals/cases/, located relative to this file.

    src/aerys_v2/evals/runner.py → parents[3] is the repo root (uv installs the
    project editable, so __file__ points into src/ — walking up works in dev and
    in CI). Callers can always pass an explicit directory instead.
    """
    return Path(__file__).resolve().parents[3] / "evals" / "cases"


def load_cases(cases_dir: Path | None = None) -> list[EvalCase]:
    """Load golden.json if present, else fall back to example.json.

    n8n mapping: the Load Dataset Code node's readFileSync — except that node
    crashed the run if the docker volume wasn't mounted. Here the fallback keeps
    CI green: golden.json is gitignored (personal data — the owner's real DM
    prompts), so a fresh clone only has the sanitized example.json.
    """
    directory = cases_dir or default_cases_dir()
    golden = directory / "golden.json"
    example = directory / "example.json"
    path = golden if golden.exists() else example
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [
        EvalCase(
            id=item["id"],
            input=item["input"],
            persona_expectations=item["persona_expectations"],
            category=item["category"],
            notes=item.get("notes", ""),
        )
        for item in raw
    ]


# ---------------------------------------------------------------------------
# Targets — the "Execute Workflow → Core Agent" hop, made pluggable
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TargetResponse:
    """What a target hands back: the reply plus how long it took.

    The n8n harness measured Date.now() around the Execute Workflow call and
    carried response_time_ms into the report — latency was a first-class metric
    (voice cares at ~4s), so it stays one here.
    """

    reply: str
    latency_ms: float


class Target(Protocol):
    """Anything that can answer an eval case.

    A Protocol instead of a base class (duck typing, checked structurally):
    the runner doesn't care whether replies come from our local graph, a future
    HTTP endpoint, or a recorded n8n baseline — same seam idea as service.ask()
    being the one door every transport uses.
    """

    name: str

    def respond(self, case: EvalCase) -> TargetResponse: ...


class LocalGraphTarget:
    """Runs each case through the real V2 graph via the ask() seam.

    n8n mapping: the Execute Workflow node calling the Core Agent. Each case gets
    its OWN thread_id — the n8n harness likewise ran every case as an isolated
    conversation (fresh session, no history bleed between cases). Reusing one
    thread would let case 7's chat history contaminate case 8's answer.
    """

    name = "local-graph"

    def __init__(self, graph: object, *, identity: Identity = EVAL_IDENTITY) -> None:
        self._graph = graph
        self._identity = identity

    def respond(self, case: EvalCase) -> TargetResponse:
        started = time.monotonic()  # same Date.now() bracket as the n8n harness
        try:
            reply = ask(
                self._graph,
                case.input,
                identity=self._identity,
                thread_id=f"eval-{case.id}",  # isolated thread per case
            )
        except ValueError as exc:
            # The golden set includes an empty-input edge case (tc-19: "should
            # handle gracefully, not crash"). V2's answer to empty input is a
            # deliberate rejection at the ask() seam — transports filter blanks
            # before the model ever runs. That IS the system's behavior, so we
            # hand it to the judge as the reply instead of letting run_eval
            # mark it score-0 infra failure (nothing broke; it refused cleanly).
            reply = f"[rejected at the ask() seam: {exc}]"
        return TargetResponse(reply=reply, latency_ms=(time.monotonic() - started) * 1000)


class N8nBaselineTarget:
    """Placeholder for scoring the LIVE n8n Aerys V1 as a comparison baseline.

    Why this is a stub and not an HTTP client: there is no unsupervised way to
    drive the n8n Core Agent from here.

    - The n8n public API has no execute-workflow endpoint on this instance
      (community edition — POST /workflows/{id}/run is not available).
    - The usual workaround, a temp webhook workflow, is BROKEN on this instance:
      webhook URLs 404 after PUT/activate cycles (documented n8n quirk in
      CLAUDE.md — "Temp webhook workflows broken").

    So capturing a baseline is a SUPERVISED step: a human runs the cases through
    Discord/Telegram (or triggers the 06-01 Eval Suite workflow manually in the
    n8n UI) and records the replies. Once captured, the recording could be
    replayed here as a simple dict-lookup target — that replay class can replace
    this stub when the capture exists.
    """

    name = "n8n-baseline"

    def respond(self, case: EvalCase) -> TargetResponse:
        raise NotImplementedError(
            "n8n baseline capture is a supervised step — see the class docstring. "
            "This instance has no execute-via-API, and temp webhook workflows 404."
        )


# ---------------------------------------------------------------------------
# Judge — the "Build Judge Request" + "Parse Score" nodes
# ---------------------------------------------------------------------------


class Judge:
    """LLM-as-judge: scores one (case, reply) pair against the rubric.

    The model is INJECTED (same seam as build_graph taking a BaseChatModel):
    tests hand in GenericFakeChatModel, production uses from_settings(). Failure
    semantics are kept bit-for-bit compatible with the n8n Parse Score node so
    old and new reports mean the same thing:

    - unparseable judge output → score 3 ("Parse error - defaulting to 3")
    - judge call blew up       → score 0 (0 = infrastructure failure, NOT quality)
    """

    def __init__(self, model: BaseChatModel) -> None:
        self._model = model

    @classmethod
    def from_settings(cls, settings: Settings) -> "Judge":
        """Build the real judge from Settings — mirrors factory.build_model().

        Imported lazily so offline tests never touch the anthropic client. The
        n8n judge was claude-sonnet-4-5 via OpenRouter at temperature 0 /
        max_tokens 500; we keep temp 0 and the token cap, but take the model id
        from Settings so the judge follows the same config as the brain.
        """
        from langchain_anthropic import ChatAnthropic

        return cls(
            ChatAnthropic(
                model=settings.model,
                api_key=settings.anthropic_api_key,
                temperature=0,  # deterministic-ish judging — same as the n8n node
                max_tokens=500,
                timeout=60.0,
                max_retries=2,
            )
        )

    def score(self, case: EvalCase, reply: str) -> dict:
        """Return {"score": 0-5, "reasoning": str} for one reply."""
        user_prompt = JUDGE_USER_TEMPLATE.format(
            user_message=case.input,
            expected_behavior=case.persona_expectations,
            agent_response=reply,
        )
        try:
            response = self._model.invoke(
                [SystemMessage(content=JUDGE_SYSTEM_PROMPT), HumanMessage(content=user_prompt)]
            )
        except Exception as exc:  # noqa: BLE001 — any transport failure means score 0
            # n8n mapping: HTTP errors from the judge scored 0 in Parse Score.
            return {"score": 0, "reasoning": f"Judge call failed: {exc}"}

        text = (
            response.text() if callable(getattr(response, "text", None)) else str(response.content)
        )
        return _parse_judge_output(text)


def _parse_judge_output(text: str) -> dict:
    """Parse the judge's JSON, with the n8n Parse Score node's exact tolerances.

    The n8n version did .replace(/```json/g,'').replace(/```/g,'') then
    JSON.parse — models love wrapping JSON in markdown fences even when told
    "JSON only". Anything that still won't parse (or has a score outside 1-5)
    defaults to 3, so one flaky judge reply can't tank or inflate a category.
    """
    cleaned = text.replace("```json", "").replace("```", "").strip()
    try:
        parsed = json.loads(cleaned)
        score = int(parsed["score"])
        if not 1 <= score <= 5:
            raise ValueError(f"score {score} outside 1-5")
        return {"score": score, "reasoning": str(parsed.get("reasoning", ""))}
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return {"score": 3, "reasoning": "Parse error - defaulting to 3"}


# ---------------------------------------------------------------------------
# Runner + report — the SplitInBatches loop and the "Format Report" node
# ---------------------------------------------------------------------------


def run_eval(
    target: Target, cases: list[EvalCase], judge: Judge
) -> tuple[list[dict], dict]:
    """Run every case through the target, judge each reply, return (results, summary).

    n8n mapping: the SplitInBatches loop that fed one case at a time into
    Execute Workflow → Build Judge Request → Parse Score, accumulating scores in
    staticData. A plain for-loop with a results list does the same job without
    the staticData reset dance (and without the microsecond-truncation bug that
    pattern is famous for).
    """
    results: list[dict] = []
    for case in cases:
        try:
            response = target.respond(case)
            reply, latency_ms = response.reply, response.latency_ms
            verdict = judge.score(case, reply)
        except Exception as exc:  # noqa: BLE001 — a dead target is an infra failure
            # Same semantics as the judge's transport failures: score 0 means
            # "the pipeline broke", never "the answer was bad".
            reply, latency_ms = f"[target error: {exc}]", 0.0
            verdict = {"score": 0, "reasoning": f"Target failed: {exc}"}

        results.append(
            {
                "id": case.id,
                "category": case.category,
                "input": case.input,
                "reply": reply,
                "latency_ms": latency_ms,
                "score": verdict["score"],
                "reasoning": verdict["reasoning"],
            }
        )

    return results, summarize(results)


def summarize(results: list[dict]) -> dict:
    """Aggregate per-case results into the report shape Format Report emitted.

    Overall avg/min/max plus, per category: count, avg, min, max, and average
    latency. Zeros (infra failures) are INCLUDED, exactly like the n8n report —
    a run with broken plumbing should look bad, not be quietly excluded.
    """
    if not results:
        return {"cases": 0, "overall": None, "by_category": {}}

    scores = [r["score"] for r in results]
    by_category: dict[str, dict] = {}
    for cat in sorted({r["category"] for r in results}):
        cat_results = [r for r in results if r["category"] == cat]
        cat_scores = [r["score"] for r in cat_results]
        by_category[cat] = {
            "count": len(cat_results),
            "avg": round(sum(cat_scores) / len(cat_scores), 2),
            "min": min(cat_scores),
            "max": max(cat_scores),
            "avg_latency_ms": round(
                sum(r["latency_ms"] for r in cat_results) / len(cat_results), 1
            ),
        }

    return {
        "cases": len(results),
        "overall": {
            "avg": round(sum(scores) / len(scores), 2),
            "min": min(scores),
            "max": max(scores),
        },
        "by_category": by_category,
        "avg_latency_ms": round(sum(r["latency_ms"] for r in results) / len(results), 1),
    }


def format_summary_table(summary: dict) -> str:
    """Render the summary as a fixed-width table for the CLI (--eval).

    Purely cosmetic — the dict from summarize() is the real artifact; this is
    the human-readable projection of it, like the report message the n8n
    workflow posted at the end of a run.
    """
    if not summary["cases"]:
        return "No cases were run."

    lines = [
        f"{'category':<24} {'n':>3} {'avg':>5} {'min':>4} {'max':>4} {'avg ms':>8}",
        "-" * 52,
    ]
    for cat, stats in summary["by_category"].items():
        lines.append(
            f"{cat:<24} {stats['count']:>3} {stats['avg']:>5} "
            f"{stats['min']:>4} {stats['max']:>4} {stats['avg_latency_ms']:>8}"
        )
    overall = summary["overall"]
    lines.append("-" * 52)
    lines.append(
        f"{'OVERALL':<24} {summary['cases']:>3} {overall['avg']:>5} "
        f"{overall['min']:>4} {overall['max']:>4} {summary['avg_latency_ms']:>8}"
    )
    return "\n".join(lines)
