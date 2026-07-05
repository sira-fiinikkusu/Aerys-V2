"""Offline test for the ``--discord`` CLI runner — fakes only, no discord.py, no network.

The transport's PURE core (event dropping/normalization) is covered by
test_discord_transport.py. This proves the ONE wiring fact that regressed the
n8n->brain cutover: the --discord runner threads long-term memory context into
build_graph exactly as --serve does, so Discord text chats recall memory too
(before this fix only voice/--serve passed context_fn). Every heavy seam is
monkeypatched to a sentinel — pure wiring verification.
"""

import sys
from contextlib import nullcontext

import pytest

import aerys_v2.cli as cli
import aerys_v2.factory as factory
import aerys_v2.service as service
import aerys_v2.transports.discord_gateway as dg


def test_discord_runner_wires_context_fn(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setenv("DISCORD_BOT_TOKEN", "dev-bot-token")
    # Keep the run hermetic: no DB (cold resolver), no wrong-database boot trap.
    for stray in ("DATABASE_URL", "MEMORIES_DATABASE_URL", "TELEGRAM_BOT_TOKEN"):
        monkeypatch.delenv(stray, raising=False)
    monkeypatch.setattr(cli, "run_boot_assertions", lambda *a, **k: None)

    # Factory seams -> sentinels (no models, no checkpointer, no DB).
    monkeypatch.setattr(factory, "build_model", lambda s: "MODEL")
    monkeypatch.setattr(factory, "load_soul", lambda p: "SOUL")
    monkeypatch.setattr(factory, "checkpointer_for", lambda s: nullcontext("CP"))
    monkeypatch.setattr(factory, "tier_models_for", lambda s: None)
    monkeypatch.setattr(factory, "deep_gate_for", lambda s: "DEEPGATE")
    monkeypatch.setattr(factory, "turn_recorder_for", lambda s: "RECORDER")
    monkeypatch.setattr(factory, "action_allowlist_for", lambda s: "ALLOW")
    monkeypatch.setattr(factory, "action_stack_for", lambda s, soul: None)  # chat-only
    # The seam under test: build_graph must receive context_fn_for(settings), the
    # SAME long-term memory wiring --serve has.
    monkeypatch.setattr(factory, "context_fn_for", lambda s: "CTXFN")

    graph_calls = {}

    def fake_build_graph(model, *, soul, checkpointer, context_fn, tier_models):
        graph_calls.update(
            model=model, soul=soul, checkpointer=checkpointer,
            context_fn=context_fn, tier_models=tier_models,
        )
        return "GRAPH"

    monkeypatch.setattr(factory, "build_graph", fake_build_graph)
    monkeypatch.setattr(service, "ask", lambda *a, **k: "ok")

    made = []

    class FakeDiscordClient:
        def __init__(self, *, ask_fn, resolve_fn, allowed_guild_id=None,
                     allowed_channel_ids=frozenset()):
            self.ask_fn = ask_fn
            self.resolve_fn = resolve_fn
            self.run_token = None
            made.append(self)

        def run(self, token):  # discord.py's run() is sync
            self.run_token = token

    monkeypatch.setattr(dg, "AerysDiscordClient", FakeDiscordClient)

    monkeypatch.setattr(sys, "argv", ["aerys-v2", "--discord"])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 0

    assert len(made) == 1
    assert made[0].run_token == "dev-bot-token"
    # The regression guard: context_fn is threaded from context_fn_for(settings)
    # into build_graph, so Discord text chats recall long-term memory.
    assert graph_calls == {
        "model": "MODEL",
        "soul": "SOUL",
        "checkpointer": "CP",
        "context_fn": "CTXFN",
        "tier_models": None,
    }
