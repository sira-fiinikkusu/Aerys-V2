"""Offline tests for the ``--telegram`` CLI runner — fakes only, no aiogram, no network.

Mirrors the ``--discord`` runner's contract at the wiring level (the transport's
PURE core is covered by test_telegram_transport.py). What's proven here without a
model, a database, or a live Bot:

  - the token-missing gate prints an instructive line and exits(1);
  - the runner instantiates AerysTelegramClient with the ask() seam wired
    identically to --discord (graph + router/action_graph + deep_allowed +
    action_allowlist + record_turn), the group allowlist parsed from
    TELEGRAM_CHAT_IDS, and drives the client's async run() with the real token;
  - identity resolution is the shared AUTH BOUNDARY: DB-less it resolves COLD and
    room-scoped (dm=private, group=public), never the owner.

Every heavy seam (factory builders, service.ask, the client) is monkeypatched to
a sentinel so this is pure wiring verification — the same fakes-and-seams style as
test_http_api.py and test_discord_transport.py.
"""

import sys
from contextlib import nullcontext

import pytest

import aerys_v2.cli as cli
import aerys_v2.factory as factory
import aerys_v2.service as service
import aerys_v2.transports.telegram_gateway as tg
from aerys_v2.transports.discord_gateway import NormalizedEvent


def test_telegram_missing_token_exits(monkeypatch, capsys):
    # No TELEGRAM_BOT_TOKEN -> instructive message + exit(1), exactly like --discord.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.setattr(sys, "argv", ["aerys-v2", "--telegram"])

    with pytest.raises(SystemExit) as exc:
        cli.main()

    assert exc.value.code == 1
    assert "TELEGRAM_BOT_TOKEN" in capsys.readouterr().out


def test_telegram_runner_wires_client(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123:abc")
    monkeypatch.setenv("TELEGRAM_CHAT_IDS", "-1001,-1002")
    # Keep the run hermetic: no DB (cold resolver), no wrong-database boot trap.
    for stray in ("DATABASE_URL", "MEMORIES_DATABASE_URL", "DISCORD_BOT_TOKEN"):
        monkeypatch.delenv(stray, raising=False)
    # Boot assertions have their own tests (test_config.py); no-op here so the
    # runner wiring is what's under test, not the DB gate.
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
    # Long-term memory context seam: a sentinel so the assertion proves the runner
    # calls context_fn_for(settings) and threads its result into build_graph — the
    # SAME memory wiring --serve has (text chats must recall memory too, not only voice).
    monkeypatch.setattr(factory, "context_fn_for", lambda s: "CTXFN")
    # Cross-surface continuity seams (track/memory-continuity) — wired like --discord.
    monkeypatch.setattr(factory, "room_context_fn_for", lambda s: "ROOMFN")
    monkeypatch.setattr(factory, "content_privacy_fn_for", lambda s: "CPFN")

    graph_calls = {}

    def fake_build_graph(model, *, soul, checkpointer, context_fn, tier_models,
                         room_context_fn):
        graph_calls.update(
            model=model, soul=soul, checkpointer=checkpointer,
            context_fn=context_fn, tier_models=tier_models,
            room_context_fn=room_context_fn,
        )
        return "GRAPH"

    monkeypatch.setattr(factory, "build_graph", fake_build_graph)

    ask_calls = []

    def fake_ask(graph, text, **kwargs):
        ask_calls.append((graph, text, kwargs))
        return f"REPLY:{text}"

    monkeypatch.setattr(service, "ask", fake_ask)

    made = []

    class FakeTelegramClient:
        def __init__(self, *, ask_fn, resolve_fn, allowed_chat_ids=frozenset(), bot_username=None):
            self.ask_fn = ask_fn
            self.resolve_fn = resolve_fn
            self.allowed_chat_ids = allowed_chat_ids
            self.bot_username = bot_username
            self.run_token = None
            made.append(self)

        async def run(self, token):  # driven by the runner's asyncio.run(...)
            self.run_token = token

    monkeypatch.setattr(tg, "AerysTelegramClient", FakeTelegramClient)

    monkeypatch.setattr(sys, "argv", ["aerys-v2", "--telegram"])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 0

    assert len(made) == 1
    client = made[0]

    # The async run() actually ran, with the real secret token (asyncio.run bridge).
    assert client.run_token == "123:abc"
    # Group allowlist parsed from TELEGRAM_CHAT_IDS — negative group ids preserved.
    assert client.allowed_chat_ids == frozenset({-1001, -1002})
    # bot_username left for getMe to resolve live (client's own design).
    assert client.bot_username is None
    # Graph built on the fake model/soul/checkpointer + tier map, same as --discord.
    # context_fn is now wired from context_fn_for(settings) — long-term memory recall
    # reaches Telegram text chats, closing the n8n->brain regression gap.
    assert graph_calls == {
        "model": "MODEL",
        "soul": "SOUL",
        "checkpointer": "CP",
        "context_fn": "CTXFN",
        "tier_models": None,
        "room_context_fn": "ROOMFN",
    }

    # ask() seam: invoking the injected ask_fn routes a Telegram turn through the
    # SAME tool/tier/audit wiring a Discord or voice turn gets.
    reply = client.ask_fn(
        "hello",
        {"user_id": "u1", "display_name": "Chris", "privacy_context": "private"},
        "telegram:dm:1",
    )
    assert reply == "REPLY:hello"
    g, text, kw = ask_calls[-1]
    assert g == "GRAPH"
    assert text == "hello"
    assert kw["thread_id"] == "telegram:dm:1"
    assert kw["identity"]["user_id"] == "u1"
    assert kw["router"] is None and kw["action_graph"] is None  # action_stack_for -> None
    assert kw["deep_allowed"] == "DEEPGATE"
    assert kw["action_allowlist"] == "ALLOW"
    assert kw["record_turn"] == "RECORDER"
    assert kw["content_privacy_classifier"] == "CPFN"  # cross-surface privacy judge wired


def _event(channel_kind: str, channel_id: str) -> NormalizedEvent:
    return NormalizedEvent(
        platform="telegram",
        platform_user_id="123",
        display_name="Chris",
        channel_kind=channel_kind,
        channel_id=channel_id,
        thread_id=f"telegram:{channel_kind}:{channel_id}",
        text="hi",
    )


def test_telegram_runner_resolver_is_cold_and_room_scoped(monkeypatch):
    """DB-less: the injected resolve_fn resolves COLD and sets room-scoped privacy.

    This is the AUTH BOUNDARY the runner shares with --discord: without
    MEMORIES_DATABASE_URL every account resolves to a non-owner cold handle, and
    privacy follows the room (dm=private, group=public) — never person-scoped.
    """
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123:abc")
    monkeypatch.setenv("TELEGRAM_CHAT_IDS", "")
    for stray in ("DATABASE_URL", "MEMORIES_DATABASE_URL", "DISCORD_BOT_TOKEN"):
        monkeypatch.delenv(stray, raising=False)
    monkeypatch.setattr(cli, "run_boot_assertions", lambda *a, **k: None)

    monkeypatch.setattr(factory, "build_model", lambda s: "MODEL")
    monkeypatch.setattr(factory, "load_soul", lambda p: "SOUL")
    monkeypatch.setattr(factory, "checkpointer_for", lambda s: nullcontext("CP"))
    monkeypatch.setattr(factory, "tier_models_for", lambda s: None)
    monkeypatch.setattr(factory, "deep_gate_for", lambda s: None)
    monkeypatch.setattr(factory, "turn_recorder_for", lambda s: None)
    monkeypatch.setattr(factory, "action_allowlist_for", lambda s: None)
    monkeypatch.setattr(factory, "action_stack_for", lambda s, soul: None)
    monkeypatch.setattr(factory, "build_graph", lambda *a, **k: "GRAPH")
    monkeypatch.setattr(service, "ask", lambda *a, **k: "ok")

    made = []

    class FakeTelegramClient:
        def __init__(self, *, ask_fn, resolve_fn, allowed_chat_ids=frozenset(), bot_username=None):
            self.resolve_fn = resolve_fn
            self.allowed_chat_ids = allowed_chat_ids
            made.append(self)

        async def run(self, token):
            return None

    monkeypatch.setattr(tg, "AerysTelegramClient", FakeTelegramClient)
    monkeypatch.setattr(sys, "argv", ["aerys-v2", "--telegram"])
    with pytest.raises(SystemExit):
        cli.main()

    resolve = made[0].resolve_fn
    dm = resolve(_event("dm", "123"))
    grp = resolve(_event("group", "-1001"))
    assert dm["privacy_context"] == "private"       # 1:1 DM
    assert grp["privacy_context"] == "public"        # shared room
    assert dm["user_id"] == "telegram:123"           # COLD handle, never the owner's UUID
    # empty TELEGRAM_CHAT_IDS -> no group allowlist (DMs in; groups fail closed until a chat id is added)
    assert made[0].allowed_chat_ids == frozenset()
