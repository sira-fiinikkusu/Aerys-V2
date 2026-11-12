"""Replay recorded action history through the agent loop and check fidelity.

Each turn uses recorded model replies and stub tool results. Replayed model inputs
must match the recorded history, excluding the system prompt's live context.
Offline: no real model or tool runs. Corpora stay private; the default report
prints only turn ids, channels, counts, and diff kinds. --show opts into content.

  python -m aerys_v2.replay_traces <corpus.jsonl> [--show TURN_ID]
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import BaseTool
from pydantic import ConfigDict


def _load(obj: dict) -> BaseMessage:
    import warnings

    from langchain_core.load import load

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")      # "load is in beta": known, and it's exact here
        return load(obj)


def _norm(m: BaseMessage) -> tuple:
    """What the model is shown of a message: comparable with the record."""
    calls = tuple((c.get("name"), json.dumps(c.get("args"), sort_keys=True), c.get("id"))
                  for c in (getattr(m, "tool_calls", None) or []))
    content = m.content if isinstance(m.content, str) else json.dumps(m.content, sort_keys=True)
    extra = json.dumps(getattr(m, "additional_kwargs", None) or {}, sort_keys=True, default=str)
    return (m.type, content, calls, getattr(m, "tool_call_id", None), getattr(m, "name", None),
            getattr(m, "status", None), extra)


@dataclass
class RecordedTurn:
    id: Any
    channel: str
    seed: list
    outputs: list
    recorded_inputs: list          # per pass, the recorded messages minus the system prompt
    tool_results: dict             # (name, args json) -> deque of contents
    tool_names: list
    config: dict
    fast: bool = False
    issues: list = field(default_factory=list)


def turn_from_row(row: dict) -> RecordedTurn:
    passes = row["passes"]
    inputs = [[_load(m) for m in p["input"]] for p in passes]
    outputs = [_load(p["output"]) for p in passes]
    strip = [msgs[1:] if msgs and isinstance(msgs[0], SystemMessage) else msgs for msgs in inputs]
    # A tool's recorded result is the ToolMessage the NEXT pass was handed for that call.
    by_id = {m.tool_call_id: m for msgs in strip for m in msgs if isinstance(m, ToolMessage)}
    results: dict = defaultdict(deque)
    names: list = []
    for out in outputs:
        for c in getattr(out, "tool_calls", None) or []:
            names.append(c["name"])
            done = by_id.get(c.get("id"))
            results[(c["name"], json.dumps(c.get("args"), sort_keys=True))].append(
                (done.content, getattr(done, "status", "success")) if done is not None
                else ("(no recorded result)", "error"))
    voice = row.get("channel") == "voice"
    identity = {"user_id": row.get("person_id") or "replay", "display_name": row.get("display_name") or "Someone"}
    if voice:
        identity["voice"] = True
    configurable = {"thread_id": f"replay:{row['id']}", "identity": identity,
                    "specialist": row.get("intent") == "action"}
    fast = "haiku" in str(passes[0].get("model") or "")
    if fast:
        configurable["tier"] = "fast"
    return RecordedTurn(id=row["id"], channel=row.get("channel") or "", seed=strip[0], outputs=outputs,
                        recorded_inputs=strip, tool_results=results, tool_names=sorted(set(names)),
                        config={"configurable": configurable, "recursion_limit": 10}, fast=fast)


class _Recording(BaseChatModel):
    """Answers with the recorded outputs in order; records what each call was handed."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    label: str
    outputs: Any
    log: Any
    bound: Any = ()
    choice: Any = None

    @property
    def _llm_type(self) -> str:
        return "recording"

    def bind_tools(self, tools, *, tool_choice=None, **kwargs):
        return self.model_copy(update={"bound": tuple(t.name for t in tools), "choice": tool_choice})

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.log.append({"slot": self.label, "tools": self.bound, "choice": self.choice,
                         "messages": [_norm(m) for m in messages if not isinstance(m, SystemMessage)],
                         "system": next((m.content for m in messages if isinstance(m, SystemMessage)), None)})
        if not self.outputs:
            raise RuntimeError("the loop asked the model for more passes than were recorded")
        return ChatResult(generations=[ChatGeneration(message=self.outputs.popleft())])


class _Stub(BaseTool):
    """A recorded tool: answers each call with the recorded result (content AND status) for
    that exact call, and keeps a ledger of what it was asked."""

    results: Any
    ledger: Any
    log: Any            # the model log: its length is the pass a call belongs to
    args_schema: Any = {"type": "object", "properties": {}, "additionalProperties": True}

    def _take(self, args: dict) -> tuple[str, str]:
        key = json.dumps(args, sort_keys=True)
        # Calls in one pass run concurrently in the agent loop, so their order within the
        # pass is not part of the comparison; the pass they belong to is.
        self.ledger.append((len(self.log), self.name, key))
        exact = self.results.get((self.name, key))
        if exact:
            return exact.popleft()
        self.ledger.append((len(self.log), "<inexact>", self.name))   # fails fidelity
        return ("(no recorded result for these arguments)", "error")

    def invoke(self, input, config=None, **kwargs):
        # ToolNode hands the whole tool call; answer with a ToolMessage so the recorded
        # status (an error result stays an error) survives, exactly as recorded.
        if isinstance(input, dict) and input.get("type") == "tool_call":
            content, status = self._take(input.get("args") or {})
            return ToolMessage(content=content, tool_call_id=input["id"], name=self.name, status=status)
        return self._take(dict(input or {}))[0]

    def _run(self, **kwargs):
        return self._take(kwargs)[0]


def _run(turn: RecordedTurn, all_tools: list | None = None) -> dict:
    """Replay one turn: {log, out, error, ledger, unexplained, leftover}.

    all_tools: every tool name in the corpus. Production binds her whole tool set on
    every pass, never only the tools a turn happened to call (an empty list binds
    differently), so each turn gets them all.

    A recorded turn can span more than one graph run: when the honesty gate in
    service.py bounces a zero-tool claim, it runs the graph again with its correction
    appended. A second run is replayed only when the record shows exactly that: the
    next recorded input is this run's messages plus one human line. Anything else is
    an unexplained run, and fails fidelity."""
    from aerys_v2.action_loop import AgentToolModels
    from aerys_v2.factory import build_action_graph

    log: list = []
    ledger: list = []
    outputs = deque(m.model_copy() for m in turn.outputs)
    results = {k: deque(v) for k, v in turn.tool_results.items()}
    names = sorted(set(all_tools or ()) | set(turn.tool_names))
    tools = [_Stub(name=n, description=f"recorded {n}", results=results, ledger=ledger, log=log) for n in names]
    conv, std, fast = (_Recording(label=n, outputs=outputs, log=log) for n in ("conv", "std", "fast"))
    graph = build_action_graph(AgentToolModels(conv, (std, True), (fast, True)), "SOUL", tools)
    run = {"log": log, "out": None, "error": None, "ledger": ledger, "unexplained": None}
    try:
        messages = list(turn.seed)
        while True:
            before = len(log)
            out = graph.invoke({"messages": messages}, turn.config)
            run["out"] = [_norm(m) for m in out["messages"]]
            if len(log) == before or len(log) >= len(turn.recorded_inputs):
                break
            nxt = turn.recorded_inputs[len(log)]
            if [_norm(m) for m in nxt[:-1]] == run["out"] and getattr(nxt[-1], "type", "") == "human":
                messages = list(nxt)            # the gate's correction, exactly as recorded
                continue
            run["unexplained"] = f"run ended after pass {len(log)}; the record continues without a gate retry"
            break
    except Exception as e:
        run["error"] = type(e).__name__
    run["leftover"] = sum(len(q) for q in results.values()) + len(outputs)
    return run


def compare(turn: RecordedTurn, all_tools: list | None = None) -> dict:
    run = _run(turn, all_tools)
    replay_log = run["log"]
    fidelity = []
    if run["error"]:
        fidelity.append("the replay failed")
    if run["unexplained"]:
        fidelity.append(run["unexplained"])
    if len(replay_log) != len(turn.recorded_inputs):
        fidelity.append(f"passes replayed={len(replay_log)} recorded={len(turn.recorded_inputs)}")
    for i, (a, rec) in enumerate(zip(replay_log, turn.recorded_inputs)):
        if a["messages"] != [_norm(m) for m in rec]:
            fidelity.append(f"pass {i + 1}: history differs from the record")
    if any(entry[1] == "<inexact>" for entry in run["ledger"]):
        fidelity.append("a tool was called with arguments the record doesn't have")
    if run["leftover"]:
        fidelity.append(f"{run['leftover']} recorded outputs/results never used")
    return {"id": turn.id, "channel": turn.channel, "passes": len(turn.outputs),
            "fidelity": not fidelity, "fidelity_diffs": fidelity,
            "replay_log": replay_log}


def main(argv: list[str]) -> int:
    import aerys_v2.factory as factory

    if not argv:
        print(__doc__)
        return 2
    show = int(argv[argv.index("--show") + 1]) if "--show" in argv else None
    factory._where_when_line = lambda thread, identity: "\nIt is noon."   # deterministic prompt clock
    rows = [json.loads(line) for line in open(argv[0])]
    all_tools = sorted({t["name"] for row in rows for t in row.get("tools", []) if t.get("name")})
    reports = []
    for row in rows:
        try:
            reports.append(compare(turn_from_row(row), all_tools))
        except Exception as e:
            reports.append({"id": row.get("id"), "channel": row.get("channel"), "passes": 0,
                            "fidelity": False, "fidelity_diffs": [f"load failed: {type(e).__name__}"]})
    fidelity = sum(r["fidelity"] for r in reports)
    passes = sum(r["passes"] for r in reports)
    print(f"{len(reports)} real action turns, {passes} model passes")
    print(f"fidelity (agent loop replays the record):      {fidelity}/{len(reports)}")
    for r in reports:
        if not r["fidelity"]:
            print(f"  turn {r['id']} ({r['channel']}, {r['passes']} passes): "
                  f"{'; '.join(r['fidelity_diffs'])}")
    if show is not None:
        r = next((r for r in reports if r["id"] == show), None)
        print(json.dumps(r, indent=1, default=str)[:20000] if r else f"no turn {show}")
    return 0 if fidelity == len(reports) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
