"""Offline tool lifeboat: real bindings/graphs, fake provider calls and tools."""

import json
import logging
import threading

import anthropic
import httpx
import pytest
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from aerys_v2.config import Settings
from aerys_v2.factory import (
    LOCAL_ACTION_HARDENING,
    LOCAL_FALLBACK_FIRED,
    LocalToolFailoverModel,
    build_action_graph,
    build_api_tool_model,
    build_graph,
    build_model,
    track_local_tool_fallback,
)
from aerys_v2.router import build_router, fallback_decision
from aerys_v2.service import ask


@tool
def home_control(operation: str) -> str:
    """Control the test light."""
    return f"Done: {operation} verified"


def settings(**overrides):
    return Settings(_env_file=None, anthropic_api_key="sk-test", **{
        "local_fallback_url": "http://127.0.0.1:11434/v1",
        "local_tool_model_name": "qwen3:4b-instruct-2507-q4_K_M",
        **overrides,
    })


def connection_error():
    return anthropic.APIConnectionError(request=httpx.Request("POST", "https://cloud.test"))


def status_error(status, cls=None):
    response = httpx.Response(status, request=httpx.Request("POST", "https://cloud.test"))
    cls = cls or (anthropic.BadRequestError if status == 400 else anthropic.APIStatusError)
    return cls("API rejected the request", response=response, body={})


@pytest.fixture(autouse=True)
def clean_fallback_flag():
    token = LOCAL_FALLBACK_FIRED.set(False)
    yield
    LOCAL_FALLBACK_FIRED.reset(token)


def bindings(pair):
    return [pair._conv, pair._auto, pair._forced, pair._fast_auto, pair._fast_forced]


@pytest.mark.parametrize("name", [None, "qwen3:4b-instruct-2507-q4_K_M"])
@pytest.mark.parametrize("pin,force", [("", True), ("pinned-model", True), ("", False)])
def test_every_binding_has_same_unforced_local_tools(name, pin, force):
    pair = build_api_tool_model(settings(
        local_tool_model_name=name, action_model=pin, action_force_tool=force,
        local_model_timeout_s=231,
    ), [home_control])
    for binding in bindings(pair):
        assert isinstance(binding, LocalToolFailoverModel)
        local = binding.lifeboat
        assert local.bound.model_name == (name or "hermes3:8b")
        assert local.bound.openai_api_base == "http://127.0.0.1:11434/v1"
        assert local.bound.request_timeout == 231
        assert local.bound.max_retries == 1
        assert "tool_choice" not in local.kwargs
        assert [t["function"]["name"] for t in local.kwargs["tools"]] == ["home_control"]
        assert [t["name"] for t in binding.primary.kwargs["tools"]] == ["home_control"]
    assert pair._forced.primary.kwargs.get("tool_choice") == ({"type": "any"} if force else None)


def test_unarmed_bindings_are_plain_and_do_not_construct_local_client(monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("unarmed tool lane constructed a local client")

    monkeypatch.setattr("langchain_openai.ChatOpenAI", unexpected)
    pair = build_api_tool_model(settings(local_fallback_url=None), [home_control])
    for binding in bindings(pair):
        assert isinstance(binding.bound, ChatAnthropic)
    assert pair._forced.kwargs["tool_choice"] == {"type": "any"}
    assert pair._fast_forced.kwargs["tool_choice"] == {"type": "any"}
    assert pair._conv.bound.model == pair._auto.bound.model == "claude-sonnet-5"


@pytest.mark.parametrize("specialist,fast,after_tool", [
    (False, False, False), (True, False, False), (True, False, True),
    (True, True, False), (True, True, True),
])
def test_outage_invokes_each_lifeboat_with_hardening_and_no_force(
    monkeypatch, caplog, specialist, fast, after_tool,
):
    cloud_calls, local_calls = [], []

    def cloud(self, messages, config=None, **kwargs):
        cloud_calls.append((messages, kwargs))
        raise connection_error()

    def local(self, messages, config=None, **kwargs):
        local_calls.append((messages, kwargs))
        return AIMessage(content="local result")

    monkeypatch.setattr(ChatAnthropic, "invoke", cloud)
    monkeypatch.setattr(ChatOpenAI, "invoke", local)
    pair = build_api_tool_model(settings(), [home_control])
    system = SystemMessage(content="Charter, tools, identity, profile, clock, shared context.", id="s")
    prompt = [system, HumanMessage(content="turn on the light")]
    if after_tool:
        prompt += [AIMessage(content="", tool_calls=[{
            "name": "home_control", "args": {"operation": "turn_on"}, "id": "c1",
        }]), ToolMessage(content="Done", tool_call_id="c1")]
    with caplog.at_level(logging.WARNING):
        reply = pair.invoke(prompt, specialist=specialist, fast=fast, stop=["STOP"])
    assert reply.content == "local result"
    assert len(cloud_calls) == len(local_calls) == 1
    assert cloud_calls[0][0] is prompt
    assert cloud_calls[0][0][0].content == system.content
    local_prompt, kwargs = local_calls[0]
    assert local_prompt[0].content == system.content + LOCAL_ACTION_HARDENING
    name_rule = (
        '(0b) For home_control, entity_id is the room or device NAME exactly as the '
        'user said it ("guest room lights", "office"); never write or guess an '
        'entity id like light.xxx unless a tool result in THIS conversation showed '
        'that exact id. '
    )
    assert "call the tool, then report its result. " + name_rule + "(1)" in local_prompt[0].content
    assert local_prompt[0].content.endswith("/no_think")
    assert local_prompt[0].id == "s"
    assert local_prompt[1:] == prompt[1:]
    assert kwargs["tools"][0]["function"]["name"] == "home_control"
    assert "tool_choice" not in kwargs
    assert kwargs["stop"] == ["STOP"]
    assert LOCAL_FALLBACK_FIRED.get() is True
    assert len([r for r in caplog.records if "local_model_fallback" in r.message]) == 1


@pytest.mark.parametrize("error", [
    connection_error(),
    anthropic.APITimeoutError(request=httpx.Request("POST", "https://cloud.test")),
    httpx.ConnectError("offline"), httpx.ConnectTimeout("offline"), httpx.ReadTimeout("offline"),
    status_error(500, anthropic.InternalServerError),
    status_error(500), status_error(503), status_error(529),
])
def test_connection_and_server_failures_fall_back(monkeypatch, error):
    calls = []

    def cloud(self, messages, config=None, **kwargs):
        raise error

    def local(self, messages, config=None, **kwargs):
        calls.append(messages)
        return AIMessage(content="local")

    monkeypatch.setattr(ChatAnthropic, "invoke", cloud)
    monkeypatch.setattr(ChatOpenAI, "invoke", local)
    pair = build_api_tool_model(settings(), [home_control])
    assert pair.invoke([HumanMessage(content="light on")], specialist=True).content == "local"
    assert calls[0][0].content == LOCAL_ACTION_HARDENING
    assert LOCAL_FALLBACK_FIRED.get() is True


@pytest.mark.parametrize("error", [
    status_error(400), status_error(401), status_error(403), status_error(404),
    status_error(429), status_error(429, anthropic.RateLimitError), status_error(499),
    RuntimeError("bug"), httpx.ReadError("not an eligible connection failure"),
])
def test_ineligible_failures_reraise_unchanged_without_local(monkeypatch, error):
    def cloud(self, messages, config=None, **kwargs):
        raise error

    def local(*args, **kwargs):
        pytest.fail("ineligible error invoked local model")

    monkeypatch.setattr(ChatAnthropic, "invoke", cloud)
    monkeypatch.setattr(ChatOpenAI, "invoke", local)
    pair = build_api_tool_model(settings(), [home_control])
    with pytest.raises(type(error)) as caught:
        pair.invoke([HumanMessage(content="light on")], specialist=True)
    assert caught.value is error
    assert LOCAL_FALLBACK_FIRED.get() is False


def test_healthy_cloud_keeps_prompt_and_never_invokes_local(monkeypatch):
    seen = []

    def cloud(self, messages, config=None, **kwargs):
        seen.append(messages)
        return AIMessage(content="cloud")

    def local(*args, **kwargs):
        pytest.fail("healthy cloud invoked local")

    monkeypatch.setattr(ChatAnthropic, "invoke", cloud)
    monkeypatch.setattr(ChatOpenAI, "invoke", local)
    pair = build_api_tool_model(settings(), [home_control])
    prompt = [SystemMessage(content="cloud prompt"), HumanMessage(content="hello")]
    assert pair.invoke(prompt).content == "cloud"
    assert seen == [prompt] and seen[0] is prompt
    assert LOCAL_FALLBACK_FIRED.get() is False


def test_receipt_crosses_copied_context_without_tainting_other_invocations():
    from contextvars import copy_context
    from unittest.mock import Mock

    primary = Mock()
    primary.invoke.side_effect = connection_error()
    model = LocalToolFailoverModel(primary, Mock())
    prompt = [SystemMessage(content=[{"type": "text", "text": "charter"}])]
    with track_local_tool_fallback():
        copy_context().run(model.invoke, prompt)
        assert LOCAL_FALLBACK_FIRED.get() is False  # node has its own context
    assert LOCAL_FALLBACK_FIRED.get() is True
    forwarded = model.lifeboat.invoke.call_args.args[0]
    assert prompt[0].content == [{"type": "text", "text": "charter"}]
    assert forwarded[0].content[-1] == {"type": "text", "text": LOCAL_ACTION_HARDENING}
    LOCAL_FALLBACK_FIRED.set(False)
    with track_local_tool_fallback():
        pass
    assert LOCAL_FALLBACK_FIRED.get() is False


@pytest.mark.parametrize("text", ["turn off the light", "hello"])
def test_router_survives_connection_error(monkeypatch, text):
    def cloud(self, messages, config=None, **kwargs):
        raise connection_error()

    monkeypatch.setattr(ChatAnthropic, "invoke", cloud)
    router = build_router(ChatAnthropic(model="test", api_key="sk-test"), "soul")
    assert router(text) == fallback_decision(text)


@pytest.mark.parametrize("voice,command", [(False, True), (True, True), (True, False)])
@pytest.mark.parametrize("local_fails", [False, True])
def test_real_graph_stamps_fallback_on_action_and_voice_turns(
    monkeypatch, voice, command, local_fails,
):
    from langchain_core.language_models.fake_chat_models import GenericFakeChatModel

    calls = []

    def cloud(self, messages, config=None, **kwargs):
        raise connection_error()

    def local(self, messages, config=None, **kwargs):
        calls.append(messages)
        if local_fails:
            raise RuntimeError("local unavailable")
        if command and not any(isinstance(m, ToolMessage) for m in messages):
            return AIMessage(content="", tool_calls=[{
                "name": "home_control", "args": {"operation": "turn_on"}, "id": "c1",
            }])
        return AIMessage(content="The light is on." if command else "Hello there.")

    monkeypatch.setattr(ChatAnthropic, "invoke", cloud)
    monkeypatch.setattr(ChatOpenAI, "invoke", local)
    pair = build_api_tool_model(settings(), [home_control])
    action = build_action_graph(pair, "soul", [home_control])
    graph = build_graph(GenericFakeChatModel(messages=iter([])), "soul")
    router = build_router(ChatAnthropic(model="test", api_key="sk-test"), "soul")
    recorded, rows = threading.Event(), []

    def record(row):
        rows.append(row)
        recorded.set()

    LOCAL_FALLBACK_FIRED.set(True)  # prove each service path resets stale flags below
    kwargs = dict(
        identity={"user_id": "p1", "display_name": "Tester", "voice": voice},
        thread_id="offline-test", router=router, action_graph=action, record_turn=record,
    )
    text = "turn on the light" if command else "hello"
    if local_fails and not (voice and command):
        with pytest.raises(RuntimeError, match="local unavailable"):
            ask(graph, text, **kwargs)
    else:
        ask(graph, text, **kwargs)
    assert recorded.wait(3), "audit row never arrived"
    assert len(rows) == 1
    assert "local_model_fallback" in rows[0]["degraded"]
    if local_fails:
        failure = "action_failed" if voice and command else "turn_failed"
        assert {"local_model_fallback", failure} <= set(json.loads(rows[0]["degraded"]))
    elif command:
        assert len(calls) == 2
        assert any(isinstance(m, ToolMessage) for m in calls[-1])
    for prompt in calls:
        assert prompt[0].content.endswith(LOCAL_ACTION_HARDENING)

    # Next turn recovers to cloud; the previous turn must not taint its receipt.
    monkeypatch.setattr(ChatAnthropic, "invoke", local if not local_fails else (
        lambda self, messages, config=None, **kwargs: AIMessage(content="Hello there.")
    ))
    kwargs["router"] = fallback_decision
    recorded.clear()
    rows.clear()
    ask(graph, "hello" if voice else "turn on the light", **kwargs)
    assert recorded.wait(3), "recovery audit row never arrived"
    assert "local_model_fallback" not in (rows[0].get("degraded") or [])


@pytest.mark.parametrize("first_reply,local_fails", [
    (None, False), (None, True),
    ("I can hear your voice.", False),
    ("Logged: your preference.", False),
])
def test_real_graph_stamps_fallback_on_chat_turns(monkeypatch, first_reply, local_fails):
    cloud_calls, local_calls = [], []
    recovered = False

    def cloud(self, messages, config=None, **kwargs):
        cloud_calls.append(messages)
        if recovered:
            return AIMessage(content="Cloud recovered.")
        if first_reply is not None and len(cloud_calls) == 1:
            return AIMessage(content=first_reply)
        raise connection_error()

    def local(self, messages, config=None, **kwargs):
        local_calls.append(messages)
        if local_fails:
            raise RuntimeError("local unavailable")
        return AIMessage(content="Hello there.")

    monkeypatch.setattr(ChatAnthropic, "invoke", cloud)
    monkeypatch.setattr(ChatOpenAI, "invoke", local)
    graph = build_graph(build_model(settings()), "soul")
    recorded, rows = threading.Event(), []

    def record(row):
        rows.append(row)
        recorded.set()

    kwargs = dict(
        identity={"user_id": "p1", "display_name": "Tester"},
        thread_id="chat-fallback-test", record_turn=record,
    )
    if local_fails:
        with pytest.raises(RuntimeError, match="local unavailable"):
            ask(graph, "hello", **kwargs)
    else:
        assert ask(graph, "hello", **kwargs) == "Hello there."
    assert recorded.wait(3), "chat audit row never arrived"
    assert len(rows) == 1
    assert len(local_calls) == 1
    assert len(cloud_calls) == (2 if first_reply else 1)
    assert "local_model_fallback" in rows[0]["degraded"]
    if local_fails:
        assert {"local_model_fallback", "turn_failed"} <= set(json.loads(rows[0]["degraded"]))

    recovered = True
    recorded.clear()
    rows.clear()
    assert ask(graph, "hello again", **kwargs) == "Cloud recovered."
    assert recorded.wait(3), "chat recovery audit row never arrived"
    assert "local_model_fallback" not in (rows[0].get("degraded") or [])
    assert len(local_calls) == 1
