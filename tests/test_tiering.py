"""Offline tests for tier routing + boot assertions + media-tool registration.

Fake models and stub routers all the way down — no API key spent, no network.
What these prove: the router's tier field parses as a HINT (unknown -> standard,
never a rejected turn), the chat node picks the model the tier names, the deep
daily cap downgrades to standard instead of erroring, voice threads stay PINNED
to standard regardless of what the router said (ChannelPolicy, locked), the boot
assertions refuse a wrong-database DATABASE_URL with a sentence instead of a
stack trace, and the media tools actually register into the action stack.
"""

import logging
from pathlib import Path

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from aerys_v2.config import (
    BootConfigError,
    Settings,
    database_name,
    duplicate_env_keys,
    run_boot_assertions,
)
from aerys_v2.factory import (
    MEDIA_OVERLAY,
    action_overlay_for,
    action_stack_for,
    action_tools_for,
    build_graph,
    tier_models_for,
)
from aerys_v2.router import (
    DEFAULT_TIER,
    TIERS,
    RouteDecision,
    fallback_decision,
    normalize_tier,
    parse_route_reply,
)
from aerys_v2.service import ask

CHRIS = {"user_id": "person-1", "display_name": "Chris"}


def fake_model(*replies: str) -> GenericFakeChatModel:
    return GenericFakeChatModel(messages=iter([AIMessage(content=r) for r in replies]))


def settings_with(**overrides) -> Settings:
    """A Settings that never reads a .env file — offline, explicit config only."""
    return Settings(_env_file=None, anthropic_api_key="sk-test", **overrides)


def chat_router_with_tier(tier: str):
    return lambda _text: RouteDecision(route="chat", ack="", tier=tier)


class StubActionGraph:
    def invoke(self, inp, config):
        return {"messages": [AIMessage(content="acted")]}


# ---- tier parsing: a hint, normalized — never a rejected turn ---------------------

def test_tier_parsed_from_router_json():
    decision = parse_route_reply(
        '{"route": "chat", "ack": "", "tier": "deep"}', "audit this plan"
    )
    assert decision == RouteDecision(route="chat", ack="", tier="deep")


def test_missing_tier_defaults_standard():
    # pre-tier router replies (no tier key) must keep working unchanged
    decision = parse_route_reply('{"route": "chat", "ack": ""}', "hello")
    assert decision.tier == DEFAULT_TIER


def test_unknown_tier_normalizes_to_standard_without_losing_the_route():
    # tier is a hint, not a correctness input — garbage tier must NOT throw
    # away a perfectly good route (contrast route, which validates strictly)
    decision = parse_route_reply(
        '{"route": "action", "ack": "On it now", "tier": "opus"}', "lights off"
    )
    assert decision.route == "action"
    assert decision.ack == "On it now"
    assert decision.tier == DEFAULT_TIER


def test_normalize_tier_vocabulary():
    for t in TIERS:
        assert normalize_tier(t) == t
    assert normalize_tier("haiku") == DEFAULT_TIER   # the V1 dead-name lesson
    assert normalize_tier(None) == DEFAULT_TIER
    assert normalize_tier(3) == DEFAULT_TIER


def test_fallback_decision_never_spends_deep():
    # the degraded path must fail CHEAP — heuristics never earn the rationed tier
    assert fallback_decision("turn on the lamp").tier == DEFAULT_TIER
    assert fallback_decision("tell me a story").tier == DEFAULT_TIER


# ---- media routing: attachments fail toward the action path ----------------------

def test_cdn_url_fails_toward_action_in_heuristic():
    decision = fallback_decision(
        "https://cdn.discordapp.com/attachments/1/2/photo.png?ex=a&is=b&hm=c"
    )
    assert decision.route == "action"


def test_media_phrases_fail_toward_action():
    assert fallback_decision("can you look at this image for me?").route == "action"
    assert fallback_decision("summarize https://youtu.be/dQw4w9WgXcQ").route == "action"
    assert fallback_decision("read this file: report.pdf").route == "action"


def test_router_prompt_teaches_media_and_tiers():
    # prompt-level regression guard, same style as the Jolteon state-question
    # guard in test_router.py: the media triggers and the tier vocabulary must
    # stay in the instructions verbatim enough to keep firing.
    from aerys_v2.router import _ROUTER_INSTRUCTIONS

    text = _ROUTER_INSTRUCTIONS.lower()
    assert "cdn.discordapp.com/attachments" in text
    assert "youtu" in text
    assert ".pdf" in text
    assert '"tier"' in text
    for tier in TIERS:
        assert f'"{tier}"' in text


def test_media_overlay_names_the_real_tools_and_url_sanctity():
    lowered = MEDIA_OVERLAY.lower()
    # tool names MUST match tools/media.py @tool functions (the V1 toolWorkflow
    # name-mismatch bug: prompt says "media", engine registers garbage, model
    # hallucinates having called it)
    assert "analyze_image" in lowered
    assert "read_document" in lowered
    assert "youtube_summary" in lowered
    assert "query parameter" in lowered  # signed-CDN-URL sanctity rule


# ---- ask() tier plumbing: the chat node answers on the tier's model ---------------

def tiered_graph():
    """A graph whose three tier models give themselves away by their reply text."""
    return build_graph(
        fake_model("base reply"),
        soul="s",
        tier_models={
            "fast": fake_model("fast reply"),
            "standard": fake_model("standard reply"),
            "deep": fake_model("deep reply"),
        },
    )


def test_chat_turn_runs_on_the_routed_tier():
    graph = tiered_graph()
    out = ask(graph, "hi", identity=CHRIS, thread_id="t1",
              router=chat_router_with_tier("fast"), action_graph=StubActionGraph())
    assert out == "fast reply"


def test_deep_tier_runs_deep_when_gate_allows():
    graph = tiered_graph()
    out = ask(graph, "audit the plan", identity=CHRIS, thread_id="t1",
              router=chat_router_with_tier("deep"), action_graph=StubActionGraph(),
              deep_allowed=lambda: True)
    assert out == "deep reply"


def test_deep_tier_without_gate_is_unenforced():
    # no DATABASE_URL -> deep_gate_for returns None -> deep runs uncapped
    graph = tiered_graph()
    out = ask(graph, "audit the plan", identity=CHRIS, thread_id="t1",
              router=chat_router_with_tier("deep"), action_graph=StubActionGraph())
    assert out == "deep reply"


def test_cap_exceeded_downgrades_to_standard_and_logs(caplog):
    graph = tiered_graph()
    with caplog.at_level(logging.INFO, logger="aerys_v2.service"):
        out = ask(graph, "audit the plan", identity=CHRIS, thread_id="t1",
                  router=chat_router_with_tier("deep"), action_graph=StubActionGraph(),
                  deep_allowed=lambda: False)
    assert out == "standard reply"  # quietly cheaper, never an error
    assert any("downgrading to standard" in r.message for r in caplog.records)


def test_tier_decision_lands_in_router_logs(caplog):
    # the router log IS the tier persistence until the v2_turns writer lands
    graph = tiered_graph()
    with caplog.at_level(logging.INFO, logger="aerys_v2.service"):
        ask(graph, "hey", identity=CHRIS, thread_id="t9",
            router=chat_router_with_tier("fast"), action_graph=StubActionGraph())
    assert any("route=chat tier=fast" in r.message for r in caplog.records)


def test_voice_thread_stays_pinned_standard():
    # ChannelPolicy (locked): the router said deep, but voice never leaves
    # standard — the ~3.6s budget can't absorb opus latency
    graph = tiered_graph()
    out = ask(graph, "audit the plan", identity=CHRIS, thread_id="voice:beta",
              router=chat_router_with_tier("deep"), action_graph=StubActionGraph(),
              deep_allowed=lambda: True)
    assert out == "standard reply"


def test_no_tier_models_keeps_old_behavior():
    # pre-tier callers (no tier_models) run the base model, byte-for-byte
    graph = build_graph(fake_model("base reply"), soul="s")
    out = ask(graph, "hi", identity=CHRIS, thread_id="t1",
              router=chat_router_with_tier("deep"), action_graph=StubActionGraph())
    assert out == "base reply"


def test_tier_models_for_api_backend_maps_the_settings_knobs():
    models = tier_models_for(settings_with())
    assert set(models) == set(TIERS)
    assert models["fast"].model == "claude-haiku-4-5"
    assert models["standard"].model == "claude-sonnet-5"
    assert models["deep"].model == "claude-opus-4-8"


def test_tier_models_for_oauth_backend_keeps_standard_on_the_pool():
    # standard = the daily driver (subscription, single-model); fast and deep
    # are ALWAYS metered ChatAnthropic — the SDK client can't switch models
    from langchain_anthropic import ChatAnthropic

    from aerys_v2.oauth_model import ClaudeOAuthChatModel

    models = tier_models_for(settings_with(model_backend="oauth"))
    assert isinstance(models["standard"], ClaudeOAuthChatModel)
    assert isinstance(models["fast"], ChatAnthropic)
    assert isinstance(models["deep"], ChatAnthropic)


# ---- boot assertions: the env-scare prevention ------------------------------------

V2_URL = "postgresql://user:pw@localhost:5432/aerys_v2"
PROD_URL = "postgresql://user:pw@localhost:5432/aerys"


def test_database_name_parses_url_shapes():
    assert database_name(V2_URL) == "aerys_v2"
    assert database_name(PROD_URL + "?sslmode=disable") == "aerys"
    assert database_name("postgresql://h") == ""


def test_boot_accepts_correct_databases():
    run_boot_assertions(
        settings_with(database_url=V2_URL, memories_database_url=PROD_URL),
        env_file=None,
    )  # no raise = pass


def test_boot_accepts_no_databases_at_all():
    run_boot_assertions(settings_with(), env_file=None)  # DB-less dev box boots


def test_boot_refuses_prod_database_as_brain_db():
    # DATABASE_URL at prod `aerys` would checkpoint V2 threads INTO production
    with pytest.raises(BootConfigError, match="aerys_v2"):
        run_boot_assertions(settings_with(database_url=PROD_URL), env_file=None)


def test_boot_refuses_n8n_engine_database():
    with pytest.raises(BootConfigError):
        run_boot_assertions(
            settings_with(database_url="postgresql://sira:pw@nas:5432/n8n"),
            env_file=None,
        )


def test_boot_warns_on_backwards_memories_url(caplog):
    # survivable (read-only surface) -> loud warning, not a refusal
    with caplog.at_level(logging.WARNING, logger="aerys_v2.config"):
        run_boot_assertions(
            settings_with(memories_database_url=V2_URL), env_file=None
        )
    assert any("swapped" in r.message for r in caplog.records)


def test_boot_warns_on_duplicate_env_keys(tmp_path, caplog):
    env = tmp_path / ".env"
    env.write_text(
        "ANTHROPIC_API_KEY=sk-new\n"
        "# comment lines never count\n"
        "MODEL=claude-sonnet-5\n"
        "export API_PORT=8300\n"
        "ANTHROPIC_API_KEY=sk-stale-line-that-wins\n"
    )
    with caplog.at_level(logging.WARNING, logger="aerys_v2.config"):
        run_boot_assertions(settings_with(), env_file=env)
    warned = [r.message for r in caplog.records if "more than once" in r.message]
    assert warned and "ANTHROPIC_API_KEY" in warned[0]


def test_clean_env_file_warns_nothing(tmp_path, caplog):
    env = tmp_path / ".env"
    env.write_text("ANTHROPIC_API_KEY=sk-one\nMODEL=claude-sonnet-5\n")
    with caplog.at_level(logging.WARNING, logger="aerys_v2.config"):
        run_boot_assertions(settings_with(), env_file=env)
    assert not caplog.records


def test_duplicate_env_keys_handles_missing_file(tmp_path):
    assert duplicate_env_keys(tmp_path / "nope.env") == []


# ---- media tools registered into the action stack ---------------------------------

def tool_names(tools: list) -> set:
    return {t.name for t in tools}


def test_media_half_arms_from_embeddings_key_alone():
    tools = action_tools_for(settings_with(embeddings_api_key="or-key"))
    assert tool_names(tools) == {"analyze_image", "read_document", "youtube_summary"}


def test_home_half_arms_from_ha_token_alone():
    tools = action_tools_for(settings_with(ha_token="ha-token"))
    # the timer tool rides the same HA door as home_control/search_entities
    assert tool_names(tools) == {"home_control", "search_entities", "timer"}


def test_both_halves_arm_together():
    tools = action_tools_for(
        settings_with(ha_token="ha-token", embeddings_api_key="or-key")
    )
    assert tool_names(tools) == {
        "home_control", "search_entities", "timer",
        "analyze_image", "read_document", "youtube_summary",
    }


def test_nothing_armed_keeps_ask_chat_only():
    assert action_tools_for(settings_with()) == []
    assert action_stack_for(settings_with(), soul="s") is None


def test_action_stack_arms_with_media_only():
    # embeddings key alone is enough for the action path to exist — image
    # questions route to tools even on a box with no Home Assistant
    stack = action_stack_for(settings_with(embeddings_api_key="or-key"), soul="s")
    assert stack is not None
    router, action_graph = stack
    assert callable(router) and hasattr(action_graph, "invoke")


def test_overlay_only_mentions_armed_tools():
    # the prompt must never tell the model to use a tool that doesn't exist
    media_only = action_overlay_for(settings_with(embeddings_api_key="or-key"))
    assert "analyze_image" in media_only and "home_control" not in media_only
    assert "timer tool" not in media_only  # timer rides the HA half, not media
    home_only = action_overlay_for(settings_with(ha_token="ha-token"))
    assert "home_control" in home_only and "analyze_image" not in home_only
    assert "timer tool" in home_only  # armed with ha_token alongside home_control
    both = action_overlay_for(
        settings_with(ha_token="ha-token", embeddings_api_key="or-key")
    )
    assert "home_control" in both and "analyze_image" in both


# ---- web-search tool registered into the action stack -----------------------------

def test_search_half_arms_from_tavily_key_alone():
    tools = action_tools_for(settings_with(tavily_api_key="tvly-key"))
    assert tool_names(tools) == {"search_web"}


def test_search_absent_when_tavily_key_is_none():
    # the default Settings (no tavily key) must NOT carry search_web — the whole
    # arming pattern is "no key, no tool"
    assert "search_web" not in tool_names(action_tools_for(settings_with()))
    assert "search_web" not in tool_names(
        action_tools_for(settings_with(ha_token="ha-token", embeddings_api_key="or-key"))
    )


def test_all_three_halves_arm_together():
    tools = action_tools_for(
        settings_with(
            ha_token="ha-token", embeddings_api_key="or-key", tavily_api_key="tvly-key"
        )
    )
    assert tool_names(tools) == {
        "home_control", "search_entities", "timer",
        "analyze_image", "read_document", "youtube_summary",
        "search_web",
    }


def test_action_stack_arms_with_search_only():
    # the tavily key alone is enough for the action path to exist — current-events
    # questions route to tools even on a box with no HA and no media
    stack = action_stack_for(settings_with(tavily_api_key="tvly-key"), soul="s")
    assert stack is not None
    router, action_graph = stack
    assert callable(router) and hasattr(action_graph, "invoke")


def test_search_overlay_names_search_web_and_only_when_armed():
    from aerys_v2.factory import SEARCH_OVERLAY

    # the overlay names the real @tool function (V1 name-mismatch guard) and the
    # concrete triggers (specificity beats generality)
    lowered = SEARCH_OVERLAY.lower()
    assert "search_web" in lowered
    assert "current events" in lowered
    assert "search for" in lowered
    assert "never fabricate" in lowered

    # armed -> the search clause appears; unarmed -> it must not
    search_only = action_overlay_for(settings_with(tavily_api_key="tvly-key"))
    assert "search_web" in search_only and "home_control" not in search_only
    home_only = action_overlay_for(settings_with(ha_token="ha-token"))
    assert "search_web" not in home_only
