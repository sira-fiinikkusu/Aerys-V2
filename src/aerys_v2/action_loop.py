"""The sole action engine, built on LangChain's create_agent.

Middleware builds the action prompt, selects unbound models and tool choice per
pass, retries eligible outages on the local house-tool lifeboat, and reports
panel activity. The honesty retry stays inside the model node, preserving the
recursion rail. Synthetic malformed-call receipts never count as executed tools.
Plain .invoke models remain supported for offline callers and tests.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from langchain_core.messages import SystemMessage

log = logging.getLogger(__name__)


@dataclass
class AgentToolModels:
    """The unbound chat models one surface's action loop chooses from, per call.

    conversational: voice banter and other non-specialist turns (free to answer
    without a tool). standard / fast: (model, may_force) for specialist turns; may_force
    = the first pass is forced to call a tool (tool_choice="any").
    fast None = the standard pair serves the fast tier too
    (a pinned action_model). lifeboat / lifeboat_tools: the local tool model and the
    house tools it gets, or None when no lifeboat is armed.
    """

    conversational: object
    standard: tuple[object, bool]
    fast: tuple[object, bool] | None = None
    lifeboat: object | None = None
    lifeboat_tools: list = field(default_factory=list)

    def choose(self, prompt: list, *, specialist: bool, fast: bool) -> tuple[object, str | None]:
        if not specialist:
            return self.conversational, None
        model, may_force = (self.fast or self.standard) if fast else self.standard
        first_pass = not any(getattr(m, "type", "") in ("ai", "tool") for m in prompt)
        return model, ("any" if may_force and first_pass else None)


@dataclass
class SurfaceSplitAgentModels:
    """Text rides the plan, voice stays metered, selected by REFLEX_SURFACE."""

    text: AgentToolModels
    voice: AgentToolModels

    def current(self) -> AgentToolModels:
        from aerys_v2.reflex import REFLEX_SURFACE

        return self.voice if REFLEX_SURFACE.get() == "voice" else self.text


def build_agent_tool_models(settings, tools: list, *, timeout_s: float = 60.0):
    """Build unbound action models, forcing policy, and optional local lifeboat."""
    from aerys_v2.factory import CLI_BLOCKED

    if settings.model_backend == "cli":
        raise RuntimeError(CLI_BLOCKED)
    if settings.model_backend == "oauth" and settings.oauth_tool_backend == "oauth":
        text = _agent_models(settings, tools, timeout_s=timeout_s, backend="oauth")
        if settings.oauth_voice_backend == "oauth":
            return text
        return SurfaceSplitAgentModels(text, _agent_models(settings, tools, timeout_s=timeout_s, backend="api"))
    return _agent_models(settings, tools, timeout_s=timeout_s, backend="api")


def _agent_models(settings, tools: list, *, backend: str, timeout_s: float = 60.0) -> AgentToolModels:
    from aerys_v2.anthropic_model import build_metered_model, supports_forced_tool_choice
    from aerys_v2.factory import LIFEBOAT_TOOLS

    lifeboat, house = None, []
    if settings.local_fallback_url is not None:
        from langchain_openai import ChatOpenAI

        house = [t for t in tools if getattr(t, "name", None) in LIFEBOAT_TOOLS] or list(tools)
        lifeboat = ChatOpenAI(
            model=settings.local_tool_model_name or settings.local_model_name,
            base_url=settings.local_fallback_url,
            api_key="local",
            max_tokens=1024,
            timeout=settings.local_model_timeout_s,
            max_retries=1,
        )

    def chat(model_name: str) -> object:
        if backend == "oauth":
            from aerys_v2.oauth_model import ClaudeOAuthChatModel

            return ClaudeOAuthChatModel(model=model_name, turn_timeout_s=settings.oauth_turn_timeout_s)
        return build_metered_model(
            settings, model=model_name, api_key=settings.anthropic_api_key,
            max_tokens=1024, timeout=timeout_s, max_retries=2, cache_prefix=True,
        )

    def pair(model_name: str) -> tuple[object, bool]:
        can_force = backend != "api" or supports_forced_tool_choice(model_name)
        return chat(model_name), bool(settings.action_force_tool and can_force)

    conversational = chat(settings.model)
    if settings.action_model:
        return AgentToolModels(conversational, pair(settings.action_model),
                               lifeboat=lifeboat, lifeboat_tools=house)
    return AgentToolModels(conversational, pair(settings.model), pair(settings.tier_fast_model),
                           lifeboat=lifeboat, lifeboat_tools=house)


def _honesty_middleware():
    """Bounce once inside the model node, without spending another graph step."""
    from langchain.agents.middleware import AgentMiddleware, ModelResponse
    from langchain_core.messages import AIMessage, HumanMessage
    from langgraph.config import get_config

    from aerys_v2.turns import ACTION_NO_TOOL_CORRECTION, extract_tool_calls

    class ActionHonesty(AgentMiddleware):
        def wrap_model_call(self, request, handler):
            first = handler(request)
            last = next((m for m in reversed(first.result) if isinstance(m, AIMessage)), None)
            if (not get_config().get("configurable", {}).get("honesty_gate")
                    or last is None or last.tool_calls
                    or extract_tool_calls(request.messages)
                    or any(isinstance(m, HumanMessage) and m.content == ACTION_NO_TOOL_CORRECTION
                           for m in request.messages)):
                return first
            log.info("action-honesty gate: zero tool calls — bouncing once with a correction")
            fix = HumanMessage(content=ACTION_NO_TOOL_CORRECTION)
            again = handler(request.override(messages=[*request.messages, *first.result, fix]))
            return ModelResponse(result=[*first.result, fix, *again.result],
                                 structured_response=again.structured_response)

    return ActionHonesty()


def _act_middleware(build_prompt, models):
    """One wrap_model_call hook: the action prompt, the per-call model, the
    lifeboat, the panel tap. A wrap hook adds no graph node (see the module doc)."""
    from langchain.agents.middleware import AgentMiddleware, ModelResponse
    from langgraph.config import get_config

    from aerys_v2.factory import (
        LOCAL_ACTION_HARDENING,
        ToolModelPair,
        is_lifeboat_failure,
        lifeboat_fired,
        lifeboat_prompt,
    )

    from aerys_v2.turns import invalid_call_ids, is_repair_receipt

    native = isinstance(models, (AgentToolModels, SurfaceSplitAgentModels))

    def _native(request, handler, prompt: list, specialist: bool, fast: bool):
        surface = models.current() if isinstance(models, SurfaceSplitAgentModels) else models
        model, tool_choice = surface.choose(prompt, specialist=specialist, fast=fast)
        system, rest = (prompt[0], prompt[1:]) if isinstance(prompt[0], SystemMessage) else (None, prompt)
        try:
            return handler(request.override(model=model, system_message=system, messages=list(rest),
                                             tool_choice=tool_choice))
        except Exception as e:
            if surface.lifeboat is None or not is_lifeboat_failure(e):
                raise
            lifeboat_fired(e)
            return handler(request.override(
                model=surface.lifeboat, tools=list(surface.lifeboat_tools), tool_choice=None,
                system_message=None, messages=list(lifeboat_prompt(prompt, hardening=LOCAL_ACTION_HARDENING)),
            ))

    def _direct(prompt: list, specialist: bool, fast: bool):
        if isinstance(models, ToolModelPair):
            reply = models.invoke(prompt, specialist=specialist, fast=fast)
        else:
            reply = models.invoke(prompt)
        return ModelResponse(result=[reply])

    class AerysAct(AgentMiddleware):
        def wrap_model_call(self, request, handler):
            # create_agent's model node has already answered any malformed tool call with a
            # synthetic error receipt. Hide it from the model: it is not a tool result
            # from execution (turns.invalid_call_ids).
            repairs = invalid_call_ids(request.messages)
            history = [m for m in request.messages if not is_repair_receipt(m, repairs)]
            prompt, specialist, fast = build_prompt(history, get_config())
            response = (_native(request, handler, prompt, specialist, fast) if native
                        else _direct(prompt, specialist, fast))
            last = response.result[-1] if response.result else None
            calls = getattr(last, "tool_calls", None)
            if calls:
                # her panel status line names what she is about to do (tool names only)
                from aerys_v2.panel_corner import report_activity

                report_activity([c.get("name", "") for c in calls if isinstance(c, dict)])
            return response

    return AerysAct()


def build_agent_action_graph(models, soul: str, tools: list, **kw) -> object:
    """Build the action loop with message input/output and service attributes."""
    from langchain.agents import create_agent
    from langchain_core.language_models.fake_chat_models import FakeListChatModel

    from aerys_v2.factory import ACTION_OVERLAY, SPECIALIST_CHARTER, _action_prompt_builder

    presence_fn = kw.get("presence_fn")
    build_prompt = _action_prompt_builder(
        soul,
        charter=kw.get("charter", SPECIALIST_CHARTER),
        overlay=kw.get("overlay", ACTION_OVERLAY),
        context_fn=kw.get("context_fn"),
        family_notes_fn=kw.get("family_notes_fn"),
        room_context_fn=kw.get("room_context_fn"),
        portable_context_fn=kw.get("portable_context_fn"),
        presence_fn=presence_fn,
        voice_room_fn=kw.get("voice_room_fn"),
        shared_surface_ids=kw.get("shared_surface_ids"),
        context_trailing=kw.get("context_trailing", False),
    )
    if isinstance(models, SurfaceSplitAgentModels):
        default = models.text.conversational
    elif isinstance(models, AgentToolModels):
        default = models.conversational
    else:
        # Never invoked: the hook calls the supplied invoke model directly. A fake that answers
        # with an error string would be a lie, so it has nothing to say.
        default = FakeListChatModel(responses=[])
    from aerys_v2.outside_guard import outside_middleware, prepare_origin
    original_prompt = build_prompt

    def build_prompt(history, config):
        prepare_origin(config, kw.get("guard_owner"), kw.get("shared_surface_ids"),
                       getattr(kw.get("outside_gate"), "members", None))
        return original_prompt(history, config)

    compiled = create_agent(
        default,
        tools=list(tools),
        # extra_middleware: official middleware mounted after hers, e.g. her home body's long
        # jobs (home_jobs.py, s3-02). The honesty retry must rebuild the prompt.
        middleware=[outside_middleware(kw.get("outside_gate")), _honesty_middleware(), _act_middleware(build_prompt, models),
                    *kw.get("extra_middleware", ())],
        checkpointer=kw.get("checkpointer"),
    )
    compiled.presence_fn = presence_fn
    compiled.action_loop = "agent"
    return compiled
