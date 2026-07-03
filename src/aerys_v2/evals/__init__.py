"""Eval harness — the Python port of n8n's "06-01 Eval Suite" (workflow g74yHFlCeOf8kxui).

n8n mapping, whole-package view:

    n8n node               →  this package
    ---------------------     ------------------------------------------
    Load Dataset (fs read) →  runner.load_cases()
    Execute Workflow       →  runner.Target / LocalGraphTarget.respond()
    Build Judge Request    →  runner.Judge (rubric embedded as constants)
    Parse Score            →  runner.Judge.score() JSON parsing + defaults
    SplitInBatches loop    →  runner.run_eval() plain for-loop
    Format Report          →  runner.summarize()

The rubric lives IN CODE here (not in evals/cases/judge_rubric.md) because that
directory is gitignored — everything except example.json contains personal data.
"""

from aerys_v2.evals.runner import (
    EvalCase,
    Judge,
    LocalGraphTarget,
    N8nBaselineTarget,
    Target,
    TargetResponse,
    load_cases,
    run_eval,
    summarize,
)

__all__ = [
    "EvalCase",
    "Judge",
    "LocalGraphTarget",
    "N8nBaselineTarget",
    "Target",
    "TargetResponse",
    "load_cases",
    "run_eval",
    "summarize",
]
