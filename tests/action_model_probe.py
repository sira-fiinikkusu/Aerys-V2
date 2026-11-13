"""Run model-policy tests through create_agent with an already-built prompt."""
from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware

from aerys_v2.action_loop import AgentToolModels, SurfaceSplitAgentModels, _act_middleware


def invoke_models(models, prompt, tools, *, specialist=False, fast=False, **model_settings):
    # Isolate model policy from context enrichment while retaining real agent binding,
    # model execution, tool execution, and middleware error handling.
    surface = models.current() if isinstance(models, SurfaceSplitAgentModels) else models

    class Settings(AgentMiddleware):
        def wrap_model_call(self, request, handler):
            return handler(request.override(model_settings=model_settings))

    graph = create_agent(surface.conversational, tools=tools, middleware=[
        _act_middleware(lambda history, config: (history, specialist, fast), models), Settings(),
    ])
    return graph.invoke({"messages": prompt})["messages"][-1]


def failing_models(error):
    from test_action_loop_agent import Boom, Rec
    log = []
    primary = Boom(label="primary", script=[], log=log, error=error)
    from langchain_core.messages import AIMessage
    boat = Rec(label="boat", script=[AIMessage(content="local")], log=log)
    return AgentToolModels(primary, (primary, True), lifeboat=boat), log
