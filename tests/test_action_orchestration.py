"""Offline tests for the TOOLS-block orchestration in ask() + the action subgraph.

Fakes all the way down: GenericFakeChatModel scripts the tool-calling loop, a
stub action graph and plain-callable routers drive the ask() branching, and the
home_control tool runs against an httpx MockTransport. What these prove: the
subgraph's act ⇄ tools loop terminates into a final AIMessage, non-voice threads
route sequentially, voice threads parallel-start (chat verdict -> chat reply,
action verdict -> immediate ack + background result landing in the SAME thread),
and ask() without router/action_graph is byte-for-byte the old chat path.
"""

import time

import httpx
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage

from aerys_v2.factory import build_action_graph, build_graph
from aerys_v2.router import RouteDecision
from aerys_v2.service import ask
from aerys_v2.tools.home_control import build_home_control_tool, canary_set

CHRIS = {"user_id": "person-1", "display_name": "Chris"}


def fake_model(*replies) -> GenericFakeChatModel:
    msgs = [r if isinstance(r, AIMessage) else AIMessage(content=r) for r in replies]
    return GenericFakeChatModel(messages=iter(msgs))


def ha_client() -> httpx.Client:
    return httpx.Client(
        transport=httpx.MockTransport(lambda req: httpx.Response(200, json=[]))
    )


def home_tool():
    return build_home_control_tool(
        base_url="http://ha.test:8123",
        token="t",
        canary_entities=canary_set("light.desk"),
        client=ha_client(),
    )


class StubActionGraph:
    """Stands in for the compiled subgraph in ask() branching tests."""

    def __init__(self, final: str = "the light is on now"):
        self.final = final
        self.calls: list[str] = []

    def invoke(self, inp: dict, config: dict) -> dict:
        self.calls.append(inp["messages"][-1].content)
        return {"messages": [AIMessage(content=self.final)]}


def chat_router(_text: str) -> RouteDecision:
    return RouteDecision(route="chat", ack="")


def action_router(_text: str) -> RouteDecision:
    return RouteDecision(route="action", ack="[warmly] Getting that light for you")


def wait_for_messages(graph, thread_id: str, count: int, timeout_s: float = 3.0) -> list:
    """Poll the checkpointer until the background action lands (or fail loudly)."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        state = graph.get_state({"configurable": {"thread_id": thread_id}})
        msgs = state.values.get("messages", [])
        if len(msgs) >= count:
            return msgs
        time.sleep(0.02)
    raise AssertionError(f"thread never reached {count} messages: {msgs}")


# ---- the action subgraph itself -------------------------------------------------

def test_action_graph_runs_tool_loop_to_final_message():
    tool = home_tool()
    # turn 1: the model calls the tool; turn 2: it answers in plain text
    model = fake_model(
        AIMessage(
            content="",
            tool_calls=[{
                "name": "home_control",
                "args": {"operation": "turn_on", "entity_id": "light.desk"},
                "id": "call-1",
                "type": "tool_call",
            }],
        ),
        "Desk light's on.",
    )
    graph = build_action_graph(model, soul="s", tools=[tool])
    result = graph.invoke(
        {"messages": [HumanMessage(content="turn on the desk light")]},
        {"configurable": {"thread_id": "t", "identity": CHRIS}, "recursion_limit": 10},
    )
    assert result["messages"][-1].content == "Desk light's on."
    # the loop actually executed the tool — a ToolMessage is in the transcript
    assert any(getattr(m, "type", "") == "tool" for m in result["messages"])
    assert "Done: turn_on sent to light.desk" in next(
        m.content for m in result["messages"] if getattr(m, "type", "") == "tool"
    )


def test_action_graph_no_tool_call_ends_immediately():
    graph = build_action_graph(fake_model("nothing to do"), soul="s", tools=[home_tool()])
    result = graph.invoke(
        {"messages": [HumanMessage(content="hi")]},
        {"configurable": {"identity": CHRIS}, "recursion_limit": 10},
    )
    assert len(result["messages"]) == 2  # human + one AI reply, no loop


# ---- ask() branching: non-voice (sequential) ------------------------------------

def test_backward_compatible_without_router():
    # no router/action_graph kwargs = the pre-TOOLS path, untouched
    graph = build_graph(fake_model("plain chat"), soul="s")
    assert ask(graph, "hi", identity=CHRIS, thread_id="t1") == "plain chat"


def test_nonvoice_chat_route_uses_chat_graph():
    graph = build_graph(fake_model("chat reply"), soul="s")
    stub = StubActionGraph()
    out = ask(graph, "how are you?", identity=CHRIS, thread_id="t1",
              router=chat_router, action_graph=stub)
    assert out == "chat reply"
    assert stub.calls == []  # action path never touched


def test_nonvoice_action_route_returns_action_result_and_lands_in_thread():
    graph = build_graph(fake_model("never spoken"), soul="s")
    stub = StubActionGraph("light is on")
    out = ask(graph, "turn on the light", identity=CHRIS, thread_id="t1",
              router=action_router, action_graph=stub)
    assert out == "light is on"
    assert stub.calls == ["turn on the light"]
    # history is coherent: the human turn AND the outcome are in the thread
    msgs = graph.get_state({"configurable": {"thread_id": "t1"}}).values["messages"]
    assert [m.content for m in msgs] == ["turn on the light", "light is on"]


# ---- ask() branching: voice (parallel-start) ------------------------------------

def test_voice_chat_route_returns_chat_result():
    graph = build_graph(fake_model("spoken chat reply"), soul="s")
    stub = StubActionGraph()
    out = ask(graph, "tell me something nice", identity=CHRIS, thread_id="voice:beta",
              router=chat_router, action_graph=stub)
    assert out == "spoken chat reply"
    assert stub.calls == []


def test_voice_action_route_returns_ack_immediately_then_result_lands():
    graph = build_graph(fake_model("speculative chat"), soul="s")
    stub = StubActionGraph("done — desk light on")
    out = ask(graph, "turn on the desk light", identity=CHRIS, thread_id="voice:beta",
              router=action_router, action_graph=stub)
    # the caller hears the router's GENERATED ack, not the action result
    assert out == "[warmly] Getting that light for you"
    # ...and the real outcome arrives in the SAME thread shortly after
    msgs = wait_for_messages(graph, "voice:beta", 2)
    assert msgs[-1].content == "done — desk light on"
    assert msgs[0].content == "turn on the desk light"  # human turn present exactly once
    assert sum(1 for m in msgs if getattr(m, "type", "") == "human") == 1


def test_voice_action_failure_lands_honestly_in_thread():
    class ExplodingActionGraph:
        def invoke(self, inp, config):
            raise RuntimeError("HA melted")

    graph = build_graph(fake_model("speculative chat"), soul="s")
    out = ask(graph, "toggle the desk light", identity=CHRIS, thread_id="voice:x",
              router=action_router, action_graph=ExplodingActionGraph())
    assert out == "[warmly] Getting that light for you"  # ack still speaks
    msgs = wait_for_messages(graph, "voice:x", 2)
    assert "didn't complete" in msgs[-1].content  # honest failure, never silence


# ---- spoken follow-up: the silent-success rule (owner, 2026-07-03) ---------------
# If the device action succeeds fast, the light changing IS the feedback — skip the
# spoken follow-up. Slow, failed, refused, or question-shaped actions always speak.
# The thread history gets the outcome EITHER WAY (silent record).

from langchain_core.messages import ToolMessage  # noqa: E402


class ToolNoteActionGraph:
    """Returns a tool trace shaped like the real subgraph: ToolMessages + final AI."""

    def __init__(self, notes: list[str], final: str = "all set", delay_s: float = 0.0):
        self.notes, self.final, self.delay_s = notes, final, delay_s

    def invoke(self, inp: dict, config: dict) -> dict:
        if self.delay_s:
            time.sleep(self.delay_s)
        msgs = [ToolMessage(content=n, tool_call_id=f"c{i}") for i, n in enumerate(self.notes)]
        return {"messages": [*inp["messages"], *msgs, AIMessage(content=self.final)]}


def recording_speaker():
    calls: list[str] = []
    return calls, calls.append


def test_voice_fast_clean_write_skips_spoken_followup():
    calls, speak = recording_speaker()
    graph = build_graph(fake_model("speculative chat"), soul="s")
    stub = ToolNoteActionGraph(
        ["Done: turn_off sent to light.desk (HA responded 200)."], final="Light's off."
    )
    out = ask(graph, "turn off the desk light", identity=CHRIS, thread_id="voice:skip",
              router=action_router, action_graph=stub, speak_fn=speak, followup_skip_s=6.0)
    assert out == "[warmly] Getting that light for you"
    msgs = wait_for_messages(graph, "voice:skip", 2)
    assert any(m.content == "Light's off." for m in msgs)  # silent record still lands
    assert calls == []  # the light changing IS the feedback — say nothing


def test_voice_slow_action_speaks_followup():
    calls, speak = recording_speaker()
    graph = build_graph(fake_model("speculative chat"), soul="s")
    stub = ToolNoteActionGraph(
        ["Done: turn_off sent to light.desk (HA responded 200)."],
        final="Light's off.", delay_s=0.02,
    )
    ask(graph, "turn off the desk light", identity=CHRIS, thread_id="voice:slow",
        router=action_router, action_graph=stub, speak_fn=speak, followup_skip_s=0.0)
    wait_for_messages(graph, "voice:slow", 2)
    assert calls == ["Light's off."]  # slow = silence would read as a dropped command


def test_voice_failed_action_always_speaks():
    class ExplodingActionGraph:
        def invoke(self, inp, config):
            raise RuntimeError("HA melted")

    calls, speak = recording_speaker()
    graph = build_graph(fake_model("speculative chat"), soul="s")
    ask(graph, "toggle the desk light", identity=CHRIS, thread_id="voice:fail",
        router=action_router, action_graph=ExplodingActionGraph(),
        speak_fn=speak, followup_skip_s=6.0)
    wait_for_messages(graph, "voice:fail", 2)
    assert len(calls) == 1 and "didn't complete" in calls[0]  # failures NEVER silent


def test_voice_refusal_speaks_followup():
    calls, speak = recording_speaker()
    graph = build_graph(fake_model("speculative chat"), soul="s")
    stub = ToolNoteActionGraph(
        ["Refused: light.garage is not on the beta write allowlist."],
        final="I can't touch that one yet.",
    )
    ask(graph, "turn off the garage light", identity=CHRIS, thread_id="voice:refuse",
        router=action_router, action_graph=stub, speak_fn=speak, followup_skip_s=6.0)
    wait_for_messages(graph, "voice:refuse", 2)
    # nothing visibly changed in the room — the honest refusal MUST be spoken
    assert calls == ["I can't touch that one yet."]


def test_voice_state_query_speaks_answer():
    calls, speak = recording_speaker()
    graph = build_graph(fake_model("speculative chat"), soul="s")
    stub = ToolNoteActionGraph(
        ['{"entity_id": "light.desk", "state": "on", "friendly_name": "Desk"}'],
        final="It's on.",
    )
    ask(graph, "is the desk light on?", identity=CHRIS, thread_id="voice:query",
        router=action_router, action_graph=stub, speak_fn=speak, followup_skip_s=6.0)
    wait_for_messages(graph, "voice:query", 2)
    assert calls == ["It's on."]  # a question's answer IS the follow-up


def test_voice_no_tool_run_speaks_followup():
    calls, speak = recording_speaker()
    graph = build_graph(fake_model("speculative chat"), soul="s")
    ask(graph, "turn on the desk light", identity=CHRIS, thread_id="voice:notool",
        router=action_router, action_graph=StubActionGraph("nothing I could do"),
        speak_fn=speak, followup_skip_s=6.0)
    wait_for_messages(graph, "voice:notool", 2)
    assert calls == ["nothing I could do"]  # no device change = the sentence is all there is


def test_speak_failure_never_blocks_history():
    def broken_speaker(_text: str) -> None:
        raise RuntimeError("satellite offline")

    graph = build_graph(fake_model("speculative chat"), soul="s")
    stub = ToolNoteActionGraph(["Refused: not allowed."], final="couldn't do it")
    ask(graph, "turn off the lamp", identity=CHRIS, thread_id="voice:deaf",
        router=action_router, action_graph=stub, speak_fn=broken_speaker,
        followup_skip_s=6.0)
    msgs = wait_for_messages(graph, "voice:deaf", 2)
    assert any(m.content == "couldn't do it" for m in msgs)  # durable record survives


# ---- context propagation into parallel-start threads (Phoenix trace unity) ------

def test_parallel_start_propagates_contextvars_into_workers():
    """OTel context rides contextvars; the parallel-start worker threads must carry
    a COPY of the caller's context or Phoenix gets orphaned root traces (the
    router's ack generation was invisible in the turn trace, owner-observed
    2026-07-03)."""
    import contextvars

    turn_var: contextvars.ContextVar = contextvars.ContextVar("turn_var", default=None)
    turn_var.set("trace-123")
    seen: dict = {}

    def recording_router(text: str) -> RouteDecision:
        seen["router"] = turn_var.get()
        return RouteDecision(route="action", ack="on it")

    class RecordingActionGraph(StubActionGraph):
        def invoke(self, inp, config):
            seen["action"] = turn_var.get()
            return super().invoke(inp, config)

    graph = build_graph(fake_model("speculative chat"), soul="s")
    ask(graph, "toggle the desk light", identity=CHRIS, thread_id="voice:ctx",
        router=recording_router, action_graph=RecordingActionGraph())
    wait_for_messages(graph, "voice:ctx", 2)
    assert seen == {"router": "trace-123", "action": "trace-123"}


# ---- regression: 2026-07-03 voice garble incident --------------------------------
# Live bug: STT turned "turn office light one off" into "Can you turn off office
# light on?"; the router acked "off", then the action subgraph asked "on or off?"
# over the one-way announce channel. These tests pin the contract that prevents it:
# the subgraph gets ONLY the current turn, plus the already-spoken ack with an
# explicit never-ask instruction.


class RecordingActionGraph:
    """Stub that captures BOTH the input messages and the config it was invoked with."""

    def __init__(self, final: str = "done"):
        self.final = final
        self.inputs: list[list] = []
        self.configs: list[dict] = []

    def invoke(self, inp: dict, config: dict) -> dict:
        self.inputs.append(list(inp["messages"]))
        self.configs.append(config)
        return {"messages": [AIMessage(content=self.final)]}


def test_voice_action_passes_spoken_ack_to_subgraph():
    graph = build_graph(fake_model("unused"), soul="s")
    stub = RecordingActionGraph()
    out = ask(graph, "turn off office light 1", identity=CHRIS, thread_id="voice:ack",
              router=action_router, action_graph=stub)
    assert out == "[warmly] Getting that light for you"
    wait_for_messages(graph, "voice:ack", 2)
    # the ack the caller already heard rides configurable into the subgraph
    assert stub.configs[0]["configurable"]["spoken_ack"] == out
    # and the checkpointer thread_id still flows (history write targets the thread)
    assert stub.configs[0]["configurable"]["thread_id"] == "voice:ack"


def test_action_subgraph_sees_only_current_command_despite_toggle_history():
    # Thread full of on/off ping-pong — the exact history live on voice:beta when
    # the incident fired. The subgraph must still receive ONLY the current turn.
    graph = build_graph(fake_model("unused"), soul="s")
    ping_pong = []
    for i in range(3):
        ping_pong += [
            HumanMessage(content=f"turn on office light {i}"),
            AIMessage(content="Office light is on."),
            HumanMessage(content="Can you turn off office light on?"),
            AIMessage(content="Quick check — did you mean on or off?"),
        ]
    graph.update_state(
        {"configurable": {"thread_id": "voice:pingpong"}},
        {"messages": ping_pong},
        as_node="chat",
    )
    stub = RecordingActionGraph(final="Office light 1 is off.")
    ask(graph, "turn off office light 1", identity=CHRIS, thread_id="voice:pingpong",
        router=action_router, action_graph=stub)
    wait_for_messages(graph, "voice:pingpong", len(ping_pong) + 2)
    # exactly ONE message reached the subgraph: the current command, verbatim
    assert len(stub.inputs[0]) == 1
    assert stub.inputs[0][0].content == "turn off office light 1"


class RecordingToolModel:
    """Fake tool-model for build_action_graph that records the prompt it was given."""

    def __init__(self, reply: str = "Office light one is off."):
        self.reply = reply
        self.prompts: list[list] = []

    def invoke(self, messages: list) -> AIMessage:
        self.prompts.append(list(messages))
        return AIMessage(content=self.reply)


def test_spoken_ack_flips_subgraph_prompt_to_never_ask():
    model = RecordingToolModel()
    graph = build_action_graph(model, soul="persona", tools=[home_tool()])
    graph.invoke(
        {"messages": [HumanMessage(content="Can you turn off office light on?")]},
        {"configurable": {"identity": CHRIS, "spoken_ack": "Turning off the office light."}},
    )
    system = model.prompts[0][0].content
    assert "Turning off the office light." in system
    assert "NEVER ask a clarifying question" in system
    # the garbled text arrives as the single human turn, untouched
    assert model.prompts[0][1].content == "Can you turn off office light on?"


def test_action_graph_injects_profile_context_block():
    # Live gap (2026-07-03): "enough charge to get to Tampa and back from home?"
    # routed to the action path, which had NO profile block — the agent read the
    # battery but didn't know where home is. context_fn output must ride the
    # action system prompt.
    model = RecordingToolModel()
    seen = []

    def fake_context(person_id, query_text, privacy_context="private"):
        seen.append((person_id, query_text))
        return "• basic.location: Rotonda West, Florida"

    graph = build_action_graph(model, soul="persona", tools=[home_tool()], context_fn=fake_context)
    graph.invoke(
        {"messages": [HumanMessage(content="enough charge to reach Tampa from home?")]},
        {"configurable": {"identity": CHRIS}},
    )
    system = model.prompts[0][0].content
    assert "[What you know about this person]" in system
    assert "Rotonda West" in system
    # called with the caller's id and the latest human text
    assert seen[0] == (CHRIS["user_id"], "enough charge to reach Tampa from home?")


def test_action_graph_context_fn_failure_never_kills_the_turn():
    def broken_context(person_id, query_text, privacy_context="private"):
        raise RuntimeError("NAS is down")

    model = RecordingToolModel(reply="done")
    graph = build_action_graph(model, soul="persona", tools=[home_tool()], context_fn=broken_context)
    result = graph.invoke(
        {"messages": [HumanMessage(content="lights off")]},
        {"configurable": {"identity": CHRIS}},
    )
    assert result["messages"][-1].content == "done"
    assert "[What you know about this person]" not in model.prompts[0][0].content


def test_action_graph_without_context_fn_prompt_unchanged():
    model = RecordingToolModel(reply="done")
    graph = build_action_graph(model, soul="persona", tools=[home_tool()])
    graph.invoke(
        {"messages": [HumanMessage(content="lights off")]},
        {"configurable": {"identity": CHRIS}},
    )
    assert "[What you know about this person]" not in model.prompts[0][0].content


# ---- regression: speculative chat must NEVER pollute the real thread -------------
# Live bug (flagged 2026-07-03): on route=action, the speculative chat gen — running
# via graph.invoke on the REAL thread — checkpointed replies like "Office light
# one's on." claiming device changes it never made; the next turn's model read that
# false history. The speculative gen now runs on a throwaway thread: route=action
# discards it entirely; route=chat copies the turn into the real thread.


def slow_action_router(_text: str) -> RouteDecision:
    # Slow verdict on purpose: the instant fake chat gen FINISHES first, so
    # chat_future.cancel() is guaranteed to fail — the exact live-race shape
    # that used to leak the speculative reply into the thread.
    time.sleep(0.05)
    return RouteDecision(route="action", ack="[warmly] Getting that light for you")


def all_thread_ids(graph) -> set:
    return {c.config["configurable"]["thread_id"] for c in graph.checkpointer.list(None)}


def test_voice_action_route_never_persists_speculative_chat_text():
    graph = build_graph(fake_model("Office light one's on."), soul="s")
    stub = StubActionGraph("Done: office light 1 is off.")
    out = ask(graph, "turn office light 1 off", identity=CHRIS, thread_id="voice:clean",
              router=slow_action_router, action_graph=stub)
    assert out == "[warmly] Getting that light for you"
    msgs = wait_for_messages(graph, "voice:clean", 2)
    contents = [m.content for m in msgs]
    # the false claim NEVER lands — only the human turn and the real outcome
    assert "Office light one's on." not in contents
    assert contents == ["turn office light 1 off", "Done: office light 1 is off."]
    # ...and the throwaway thread is cleaned up shortly after (background)
    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline:
        tids = all_thread_ids(graph)
        if tids == {"voice:clean"}:
            break
        time.sleep(0.02)
    assert tids == {"voice:clean"}


def test_voice_chat_route_persists_reply_and_leaves_no_speculative_thread():
    graph = build_graph(fake_model("spoken chat reply"), soul="s")
    out = ask(graph, "tell me something nice", identity=CHRIS, thread_id="voice:persist",
              router=chat_router, action_graph=StubActionGraph())
    assert out == "spoken chat reply"
    # route=chat: the turn IS copied into the real thread (durable conversation)
    msgs = graph.get_state({"configurable": {"thread_id": "voice:persist"}}).values["messages"]
    assert [m.content for m in msgs] == ["tell me something nice", "spoken chat reply"]
    # cleanup is synchronous on the chat path — no ::spec:: thread survives
    assert all_thread_ids(graph) == {"voice:persist"}


class RecordingChatModel:
    """Fake chat model recording the exact messages each invoke received."""

    def __init__(self, reply: str = "ok"):
        self.reply = reply
        self.prompts: list[list] = []

    def invoke(self, messages: list) -> AIMessage:
        self.prompts.append(list(messages))
        return AIMessage(content=self.reply)


def test_voice_speculative_chat_sees_real_thread_history():
    # isolation must not cost context: the throwaway thread is SEEDED with the
    # real history, so the speculative gen answers with full conversation memory.
    model = RecordingChatModel("and that's the story")
    graph = build_graph(model, soul="s")
    graph.update_state(
        {"configurable": {"thread_id": "voice:hist"}},
        {"messages": [HumanMessage(content="remember the lighthouse"),
                      AIMessage(content="I remember.")]},
        as_node="chat",
    )
    out = ask(graph, "tell me more", identity=CHRIS, thread_id="voice:hist",
              router=chat_router, action_graph=StubActionGraph())
    assert out == "and that's the story"
    prompt = model.prompts[0]  # [system, *history, human]
    assert [m.content for m in prompt[1:]] == [
        "remember the lighthouse", "I remember.", "tell me more"]
    msgs = graph.get_state({"configurable": {"thread_id": "voice:hist"}}).values["messages"]
    assert [m.content for m in msgs] == [
        "remember the lighthouse", "I remember.", "tell me more", "and that's the story"]


def test_no_spoken_ack_means_no_voice_overlay():
    # Non-voice sequential path never sets spoken_ack — the subgraph may still
    # converse there (its reply returns to a caller who CAN answer).
    model = RecordingToolModel()
    graph = build_action_graph(model, soul="persona", tools=[home_tool()])
    graph.invoke(
        {"messages": [HumanMessage(content="turn off office light 1")]},
        {"configurable": {"identity": CHRIS}},
    )
    system = model.prompts[0][0].content
    assert "NEVER ask a clarifying question" not in system


# ---- ask() ACTION AUTHORIZATION — the owner/allowlist gate (CRITICAL) -----------
# Cross-review (2026-07-04) found the action stack had NO owner check: any caller
# who reached ask() could actuate the house / read presence. These lock the gate:
# only person_ids in the allowlist reach the tools; everyone else gets chat-only.

STRANGER = {"user_id": "stranger:404", "display_name": "Nosy Guildmate"}
MEGAN = {"user_id": "person-megan", "display_name": "Megan"}


def test_action_gate_blocks_non_allowlisted_caller():
    # A caller outside the allowlist NEVER reaches the action stack — even when the
    # router would route action and the transport armed it. The house is untouched.
    graph = build_graph(fake_model("chat-only for strangers"), soul="s")
    stub = StubActionGraph("light is on")
    out = ask(graph, "turn on the office light", identity=STRANGER, thread_id="t1",
              router=action_router, action_graph=stub,
              action_allowlist=frozenset({CHRIS["user_id"]}))
    assert out == "chat-only for strangers"
    assert stub.calls == []  # the action graph was never invoked


def test_action_gate_allows_owner():
    graph = build_graph(fake_model("never spoken"), soul="s")
    stub = StubActionGraph("light is on")
    out = ask(graph, "turn on the light", identity=CHRIS, thread_id="t1",
              router=action_router, action_graph=stub,
              action_allowlist=frozenset({CHRIS["user_id"]}))
    assert out == "light is on"
    assert stub.calls == ["turn on the light"]


def test_action_gate_allows_second_allowlisted_person():
    # Adding Megan's person_id to the allowlist grants IDENTICAL house access —
    # the config-only extension path (house_control_person_ids), proven.
    graph = build_graph(fake_model("never spoken"), soul="s")
    stub = StubActionGraph("light is on")
    allow = frozenset({CHRIS["user_id"], MEGAN["user_id"]})
    out = ask(graph, "turn on the light", identity=MEGAN, thread_id="t1",
              router=action_router, action_graph=stub, action_allowlist=allow)
    assert out == "light is on"
    assert stub.calls == ["turn on the light"]


def test_action_gate_unenforced_when_no_allowlist():
    # None = dev/unenforced (no owner configured), matching deep_allowed's posture:
    # tools work for anyone. The gate is opt-in via config, not on by default.
    graph = build_graph(fake_model("never spoken"), soul="s")
    stub = StubActionGraph("light is on")
    out = ask(graph, "turn on the light", identity=STRANGER, thread_id="t1",
              router=action_router, action_graph=stub, action_allowlist=None)
    assert out == "light is on"
    assert stub.calls == ["turn on the light"]


def test_action_gate_blocks_stranger_on_voice_thread_too():
    # The gate fires BEFORE the voice parallel-start branch — a non-owner on a
    # voice thread still gets chat-only, never a speculative action.
    graph = build_graph(fake_model("chat-only"), soul="s")
    stub = StubActionGraph("done")
    out = ask(graph, "turn on the light", identity=STRANGER, thread_id="voice:x",
              router=action_router, action_graph=stub,
              action_allowlist=frozenset({CHRIS["user_id"]}))
    assert out == "chat-only"
    assert stub.calls == []
