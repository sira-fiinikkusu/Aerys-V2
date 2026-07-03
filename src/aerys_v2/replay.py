"""Replay harness — run captured n8n traffic through the V2 graph (dossier 1g).

n8n mapping: the payloads in evals/replay/ are REAL executions lifted from the live
instance — the Core Agent's "Execute Workflow Trigger" input (Discord DM shape) and
the Voice Adapter's "Inject Profile Context (Voice)" output (the fully-enriched item
that entered Execute Gemini Agent). Replaying them here answers the migration
question the eval suite can't: does V2's ask() seam accept the exact shapes V1's
adapters actually produced, and does a turn come back for every one of them?

This is a SMOKE harness, not a quality harness — no judge, no rubric. Pass/fail is
"a reply came back" (plus latency, because voice cares at ~4s). Quality scoring
stays in evals/runner.py; replay proves shape-compatibility and crash-freedom.

Isolation rules (both load-bearing):
- build_replay_graph() constructs the graph on a FRESH InMemorySaver — never the
  Postgres checkpointer. Replaying 50 captured turns into NAS-durable threads would
  poison real conversation history with redacted placeholder text.
- Every thread_id is namespaced "replay:<payload id>" — belt and braces, so even a
  mistakenly-durable graph could never collide with a live thread key like
  "discord:dm:<snowflake>" (see transports.discord_gateway.thread_key).
"""

import json
import time
from dataclasses import dataclass
from pathlib import Path

from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.checkpoint.memory import InMemorySaver

from aerys_v2.factory import build_graph
from aerys_v2.service import ask
from aerys_v2.state import Identity

# ---------------------------------------------------------------------------
# Payloads — the capture artifact (see evals/replay/, git-ignored except example)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReplayPayload:
    """One captured n8n execution, in the capture schema.

    Mirrors the JSON in evals/replay/*.json: {id, channel, captured_at,
    source_execution, real_text, payload}. `payload` keeps the raw channel-specific
    field bag (DM and voice shapes differ — that difference is the point of the
    harness), so mapping to graph inputs happens in to_ask_inputs(), not at load.
    """

    id: str
    channel: str            # "dm" | "voice" (guild/telegram had no saved executions)
    captured_at: str
    source_execution: str
    real_text: bool         # False → message_text is length-preserving redaction
    payload: dict


def default_replay_dir() -> Path:
    """Repo-root evals/replay/, located relative to this file.

    src/aerys_v2/replay.py → parents[2] is the repo root (same editable-install
    walk-up as evals.runner.default_cases_dir). Callers can pass an explicit
    directory instead.
    """
    return Path(__file__).resolve().parents[2] / "evals" / "replay"


def load_payloads(replay_dir: Path | None = None) -> list[ReplayPayload]:
    """Load payloads.json if present, else fall back to example_payload.json.

    Same contract as evals.runner.load_cases: the real capture is gitignored
    (redacted or not, it's the owner's traffic shapes), so a fresh clone only has
    the two fully-synthetic examples — CI still gets one payload per channel shape.
    """
    directory = replay_dir or default_replay_dir()
    captured = directory / "payloads.json"
    example = directory / "example_payload.json"
    path = captured if captured.exists() else example
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [
        ReplayPayload(
            id=item["id"],
            channel=item["channel"],
            captured_at=item["captured_at"],
            # source_execution is an int in the captured file, a string in the
            # example — normalize so callers never care which file they got.
            source_execution=str(item["source_execution"]),
            real_text=bool(item["real_text"]),
            payload=item["payload"],
        )
        for item in raw
    ]


# ---------------------------------------------------------------------------
# Mapping — the "Normalize Message" job, run in reverse
# ---------------------------------------------------------------------------


def to_ask_inputs(record: ReplayPayload) -> tuple[str, Identity, str]:
    """Map one captured n8n payload onto the ask() seam: (text, identity, thread_id).

    n8n mapping, field by field:
    - text ← message_text (both shapes carry it; voice also has message_content,
      kept as a fallback because the Voice Adapter set both from the same STT).
    - identity ← person_id + display_name: person_id is what the Identity Resolver
      (03-01) stamped on the item, i.e. the V1 equivalent of Identity.user_id.
      The raw platform user_id is the fallback for payloads that predate resolution.
    - thread_id ← "replay:<payload id>": NEVER derived from session_key or
      channel_id, so no replay can address a real conversation thread.
    """
    p = record.payload
    text = p.get("message_text") or p.get("message_content") or ""
    identity: Identity = {
        "user_id": p.get("person_id") or p.get("user_id") or "replay-unknown",
        "display_name": p.get("display_name") or p.get("username") or "Unknown Caller",
    }
    return text, identity, f"replay:{record.id}"


# ---------------------------------------------------------------------------
# Graph + runner — the throwaway brain and the SplitInBatches loop
# ---------------------------------------------------------------------------


def build_replay_graph(model: BaseChatModel, soul: str) -> object:
    """A graph on a FRESH InMemorySaver — the only graph replay is allowed to use.

    Explicit rather than relying on build_graph's default, so the isolation rule
    is enforced by construction: there is no parameter through which a Postgres
    checkpointer could arrive. Everything the replay writes dies with the process.
    """
    return build_graph(model, soul, checkpointer=InMemorySaver())


def run_replay(graph: object, payloads: list[ReplayPayload]) -> tuple[list[dict], dict]:
    """Replay every payload through the graph, return (results, summary).

    Error isolation matches run_eval's semantics: one payload that trips the
    ask() rails (or a model failure) records ok=False and the loop keeps going —
    a 50-payload run must report 1 bad shape, not die on it. reply_len instead of
    reply text keeps redacted/real content out of logs and CI output entirely.
    """
    results: list[dict] = []
    for record in payloads:
        text, identity, thread_id = to_ask_inputs(record)
        started = time.monotonic()  # same Date.now() bracket as the eval runner
        try:
            reply = ask(graph, text, identity=identity, thread_id=thread_id)
            ok, reply_len, error = True, len(reply), None
        except Exception as exc:  # noqa: BLE001 — isolate, record, continue
            ok, reply_len, error = False, 0, f"{type(exc).__name__}: {exc}"
        results.append(
            {
                "id": record.id,
                "channel": record.channel,
                "ok": ok,
                "reply_len": reply_len,
                "latency_ms": (time.monotonic() - started) * 1000,
                "error": error,
            }
        )
    return results, summarize_replay(results)


def summarize_replay(results: list[dict]) -> dict:
    """Aggregate per-payload results — the replay's "Format Report".

    Counts and latency only (no scores — replay has no judge). Failures are
    INCLUDED in latency averages, same philosophy as summarize() in the eval
    runner: broken plumbing should make the numbers look bad, not vanish.
    """
    if not results:
        return {"payloads": 0, "ok": 0, "failed": 0, "by_channel": {}, "avg_latency_ms": 0.0}

    by_channel: dict[str, dict] = {}
    for ch in sorted({r["channel"] for r in results}):
        ch_results = [r for r in results if r["channel"] == ch]
        by_channel[ch] = {
            "count": len(ch_results),
            "ok": sum(1 for r in ch_results if r["ok"]),
            "failed": sum(1 for r in ch_results if not r["ok"]),
            "avg_latency_ms": round(
                sum(r["latency_ms"] for r in ch_results) / len(ch_results), 1
            ),
        }

    return {
        "payloads": len(results),
        "ok": sum(1 for r in results if r["ok"]),
        "failed": sum(1 for r in results if not r["ok"]),
        "by_channel": by_channel,
        "avg_latency_ms": round(sum(r["latency_ms"] for r in results) / len(results), 1),
    }


def format_replay_summary(summary: dict) -> str:
    """Render the summary as a fixed-width table for the CLI (--replay).

    Cosmetic projection of summarize_replay()'s dict — same relationship as
    format_summary_table has to summarize() in the eval runner.
    """
    if not summary["payloads"]:
        return "No payloads were replayed."

    lines = [
        f"{'channel':<12} {'n':>4} {'ok':>4} {'fail':>5} {'avg ms':>9}",
        "-" * 38,
    ]
    for ch, stats in summary["by_channel"].items():
        lines.append(
            f"{ch:<12} {stats['count']:>4} {stats['ok']:>4} "
            f"{stats['failed']:>5} {stats['avg_latency_ms']:>9}"
        )
    lines.append("-" * 38)
    lines.append(
        f"{'TOTAL':<12} {summary['payloads']:>4} {summary['ok']:>4} "
        f"{summary['failed']:>5} {summary['avg_latency_ms']:>9}"
    )
    return "\n".join(lines)
