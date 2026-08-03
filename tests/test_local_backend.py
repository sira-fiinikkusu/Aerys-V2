"""Offline tests for the local model backend + metered->local failover.

No network, no Ollama: the ChatOpenAI constructions are inspected, not invoked,
and the failover wrapper is exercised with fake chat models. What these prove:
MODEL_BACKEND=local builds an OpenAI-compatible client aimed at the configured
base_url, local mode collapses ALL tiers onto one local model, the failover
wrapper answers from the lifeboat when the primary dies (and ONLY then), the
LOCAL_FALLBACK_FIRED contextvar + WARNING log fire on fallback (the owner's
8/03 condition: automatic failover is fine PROVIDED the logs reflect it), and
the service stamps 'local_model_fallback' into the turn's degraded list.
"""

import logging
import time

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from aerys_v2.config import Settings
from aerys_v2.factory import (
    LOCAL_FALLBACK_FIRED,
    LocalFailoverModel,
    build_model,
    local_model_for,
    tier_models_for,
)


def settings(**overrides) -> Settings:
    return Settings(_env_file=None, anthropic_api_key="sk-test", **overrides)


def fake(text: str) -> GenericFakeChatModel:
    return GenericFakeChatModel(messages=iter([AIMessage(content=text)]))


class ExplodingModel(GenericFakeChatModel):
    def _generate(self, *a, **k):  # noqa: D102 — dies like a metered 500 would
        raise RuntimeError("meter unreachable")


# ---------------------------------------------------------------- local backend


def test_local_backend_builds_openai_client_at_configured_url():
    s = settings(
        model_backend="local",
        local_model_base_url="http://10.0.0.5:11434/v1",
        local_model_name="hermes3:8b",
    )
    m = build_model(s)
    assert type(m).__name__ == "ChatOpenAI"
    assert "10.0.0.5:11434" in str(m.openai_api_base or m.client._client.base_url)
    assert m.model_name == "hermes3:8b"


def test_local_mode_collapses_all_tiers_onto_one_local_model():
    s = settings(model_backend="local")
    tiers = tier_models_for(s)
    assert set(tiers) == {"fast", "standard", "deep"}
    assert tiers["fast"] is tiers["standard"] is tiers["deep"]
    assert type(tiers["fast"]).__name__ == "ChatOpenAI"


def test_api_backend_without_fallback_url_is_unwrapped():
    tiers = tier_models_for(settings())
    for name, model in tiers.items():
        assert not isinstance(model, LocalFailoverModel), name


def test_fallback_url_arms_failover_on_every_metered_tier():
    tiers = tier_models_for(settings(local_fallback_url="http://127.0.0.1:11434/v1"))
    for name, model in tiers.items():
        assert isinstance(model, LocalFailoverModel), name


# ---------------------------------------------------------------- failover wrapper


def test_healthy_primary_answers_and_no_flag_is_set():
    LOCAL_FALLBACK_FIRED.set(False)
    m = LocalFailoverModel(primary=fake("from primary"), lifeboat=fake("from lifeboat"))
    reply = m.invoke([("human", "hi")])
    assert reply.content == "from primary"
    assert LOCAL_FALLBACK_FIRED.get() is False


def test_dead_primary_falls_back_sets_flag_and_logs(caplog):
    LOCAL_FALLBACK_FIRED.set(False)
    m = LocalFailoverModel(
        primary=ExplodingModel(messages=iter([])), lifeboat=fake("from lifeboat")
    )
    with caplog.at_level(logging.WARNING):
        reply = m.invoke([("human", "hi")])
    assert reply.content == "from lifeboat"
    assert LOCAL_FALLBACK_FIRED.get() is True
    assert any("local_model_fallback" in r.message for r in caplog.records)


def test_dead_primary_and_dead_lifeboat_still_raises():
    LOCAL_FALLBACK_FIRED.set(False)
    m = LocalFailoverModel(
        primary=ExplodingModel(messages=iter([])),
        lifeboat=ExplodingModel(messages=iter([])),
    )
    with pytest.raises(RuntimeError):
        m.invoke([("human", "hi")])
    # The flag still fired: forensics must see the lifeboat was attempted.
    assert LOCAL_FALLBACK_FIRED.get() is True


# ---------------------------------------------------------------- service stamping


def test_chat_turn_stamps_local_model_fallback_into_degraded():
    from aerys_v2.service import Rails, _chat_turn

    class FallbackGraph:
        def invoke(self, _state, _config):
            LOCAL_FALLBACK_FIRED.set(True)  # what a mid-turn lifeboat does
            return {"messages": [AIMessage(content="answered locally")]}

    rows = []
    reply, handoff = _chat_turn(
        FallbackGraph(),
        "hello",
        {"configurable": {"thread_id": "t"}},
        Rails(),
        time.monotonic(),
        record_turn=rows.append,
    )
    assert reply == "answered locally"
    assert not handoff
    assert rows and "local_model_fallback" in (rows[0].get("degraded") or [])


def test_chat_turn_resets_stale_flag_from_prior_turn():
    from aerys_v2.service import Rails, _chat_turn

    class CleanGraph:
        def invoke(self, _state, _config):
            return {"messages": [AIMessage(content="all metered")]}

    LOCAL_FALLBACK_FIRED.set(True)  # stale from an earlier turn in this task
    rows = []
    _chat_turn(
        CleanGraph(),
        "hello",
        {"configurable": {"thread_id": "t"}},
        Rails(),
        time.monotonic(),
        record_turn=rows.append,
    )
    assert "local_model_fallback" not in (rows[0].get("degraded") or [])


def test_local_model_for_prefers_explicit_base_url():
    s = settings(local_model_base_url="http://127.0.0.1:11434/v1")
    m = local_model_for(s, base_url="http://192.0.2.9:8080/v1")
    assert "192.0.2.9" in str(m.openai_api_base or m.client._client.base_url)
