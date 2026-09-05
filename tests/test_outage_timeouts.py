"""Inspect real SDK clients and requests; provider I/O stays offline."""

import asyncio
import time

import anthropic
import httpx
import pytest
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from aerys_v2.anthropic_model import build_metered_model
from aerys_v2.config import Settings
from aerys_v2.evals.runner import Judge
from aerys_v2.factory import (
    LOCAL_FALLBACK_FIRED,
    build_api_tool_model,
    build_model,
    content_privacy_fn_for,
    tier_models_for,
)
from aerys_v2.router import fallback_decision, router_for


@tool
def test_light() -> str:
    """Read the fake light."""
    return "off"


def settings(armed=True, **overrides):
    return Settings(
        _env_file=None, anthropic_api_key="sk-test",
        local_fallback_url="http://127.0.0.1:11434/v1" if armed else None,
        **overrides,
    )


@pytest.fixture(autouse=True)
def reset_fallback_flag():
    token = LOCAL_FALLBACK_FIRED.set(False)
    yield
    LOCAL_FALLBACK_FIRED.reset(token)


@pytest.mark.parametrize("armed", [False, True])
@pytest.mark.parametrize("timeout_s", [60.0, 83.0])
@pytest.mark.parametrize("pin,force", [("", True), ("pinned", True), ("", False)])
def test_all_metered_clients_preserve_read_budget_and_set_connection_policy(
    monkeypatch, armed, timeout_s, pin, force,
):
    s = settings(armed, action_model=pin, action_force_tool=force)
    clients = []

    def collect(model, read, retries=2):
        if armed:
            model = model.primary
        clients.append((model, read, retries))

    collect(build_model(s, timeout_s=timeout_s), timeout_s)
    for model in tier_models_for(s, timeout_s=timeout_s).values():
        collect(model, timeout_s)
    pair = build_api_tool_model(s, [test_light], timeout_s=timeout_s)
    for binding in (pair._conv, pair._auto, pair._forced, pair._fast_auto, pair._fast_forced):
        if armed:
            binding = binding.primary
        clients.append((binding.bound, timeout_s, 2))
    empty = build_api_tool_model(s, [], timeout_s=timeout_s)
    clients.append(((empty.primary if armed else empty).bound, timeout_s, 2))

    # Exercise the real closures to capture their actual client, not a factory stub.
    def record(self, *args, **kwargs):
        clients.append((self, 10.0 if self.max_tokens == 200 else 15.0, 1))
        return AIMessage(content="public")

    monkeypatch.setattr(ChatAnthropic, "invoke", record)
    router_for(s, "soul")("hello")
    assert content_privacy_fn_for(s)("hello") == "public"
    clients.append((Judge.from_settings(s)._model, 60.0, 2))

    for model, read, retries in clients:
        assert isinstance(model, ChatAnthropic)
        if not armed:
            assert type(model) is ChatAnthropic
        assert model.default_request_timeout == read
        assert model.max_retries == (0 if armed else retries)
        # SDK request options override the shared transport's scalar default.
        for sdk in (model._client, model._async_client):
            expected = httpx.Timeout(read, connect=5.0) if armed else read
            assert sdk.timeout == expected
            assert sdk.max_retries == (0 if armed else retries)
            assert sdk._client.timeout == httpx.Timeout(read)


@pytest.mark.parametrize("text", ["turn off the light", "hello"])
def test_router_api_timeout_returns_heuristic_promptly(monkeypatch, text):
    calls = []

    def timeout(self, *args, **kwargs):
        calls.append(self)
        raise anthropic.APITimeoutError(request=httpx.Request("POST", "https://cloud.test"))

    monkeypatch.setattr(ChatAnthropic, "invoke", timeout)
    route = router_for(settings(), "soul")
    started = time.monotonic()
    assert route(text) == fallback_decision(text)
    assert time.monotonic() - started < 1.0
    assert len(calls) == 1


def test_armed_sdk_options_do_not_change_unarmed_shared_transport(monkeypatch):
    requests = []

    def blackhole(self, request, **kwargs):
        requests.append(request)
        raise httpx.ConnectTimeout("connection timed out", request=request)

    monkeypatch.setattr(httpx.Client, "send", blackhole)
    kwargs = dict(
        model="test-model", api_key="sk-test", timeout=83.0, max_retries=0,
        base_url="https://cloud.test", default_headers={"X-Test-Policy": "preserved"},
    )
    unarmed = build_metered_model(settings(False), **kwargs)
    armed = build_metered_model(settings(), **kwargs)
    assert unarmed._client._client is armed._client._client
    assert unarmed._async_client._client is armed._async_client._client
    for model, connect in ((armed, 5.0), (unarmed, 83.0), (armed, 5.0)):
        with pytest.raises(anthropic.APITimeoutError):
            model.invoke([HumanMessage(content="hello")])
        request = requests[-1]
        assert request.url.host == "cloud.test"
        assert request.headers["X-Test-Policy"] == "preserved"
        assert request.extensions["timeout"] == {
            "connect": connect, "read": 83.0, "write": 83.0, "pool": 83.0,
        }
    assert len(requests) == 3


def test_async_sdk_request_uses_split_timeout_without_retries(monkeypatch):
    requests = []

    async def blackhole(self, request, **kwargs):
        requests.append(request)
        raise httpx.ConnectTimeout("connection timed out", request=request)

    monkeypatch.setattr(httpx.AsyncClient, "send", blackhole)
    primary = build_model(settings(), timeout_s=83.0).primary
    with pytest.raises(anthropic.APITimeoutError):
        asyncio.run(primary._async_client.messages.create(
            model=primary.model, max_tokens=16,
            messages=[{"role": "user", "content": "hello"}],
        ))
    assert len(requests) == 1
    assert requests[0].extensions["timeout"] == {
        "connect": 5.0, "read": 83.0, "write": 83.0, "pool": 83.0,
    }


@pytest.mark.parametrize("lane", ["chat", "fast", "standard", "deep", "tool", "router", "privacy"])
def test_connect_timeout_reaches_real_sdk_once_and_preserves_failure_policy(monkeypatch, lane):
    requests, local_calls = [], []

    def blackhole(self, request, **kwargs):
        requests.append(request)
        # This is the transport exception raised when SYNs exceed connect timeout.
        raise httpx.ConnectTimeout("connection timed out", request=request)

    def local(self, *args, **kwargs):
        local_calls.append(self)
        return AIMessage(content="local reply")

    monkeypatch.setattr(httpx.Client, "send", blackhole)
    monkeypatch.setattr(ChatOpenAI, "invoke", local)
    s = settings()
    read = 60.0
    if lane == "router":
        invoke = lambda: router_for(s, "soul")("turn off the light")
        expected = fallback_decision("turn off the light")
        read = 10.0
    elif lane == "privacy":
        invoke = lambda: content_privacy_fn_for(s)("hello there")
        expected = "private"  # A failed judge must never relax privacy.
        read = 15.0
    else:
        if lane == "chat":
            model = build_model(s)
        elif lane == "tool":
            model = build_api_tool_model(s, [test_light])._forced
        else:
            model = tier_models_for(s)[lane]
        invoke = lambda: model.invoke([HumanMessage(content="hello")]).content
        expected = "local reply"
    started = time.monotonic()
    assert invoke() == expected
    assert time.monotonic() - started < 2.0
    assert len(requests) == 1  # Actual SDK retries, not a mocked model invocation.
    assert requests[0].extensions["timeout"] == {
        "connect": 5.0, "read": read, "write": read, "pool": read,
    }
    fallback_expected = lane not in ("router", "privacy")
    assert len(local_calls) == int(fallback_expected)
    assert LOCAL_FALLBACK_FIRED.get() is fallback_expected
