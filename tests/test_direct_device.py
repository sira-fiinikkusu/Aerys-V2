"""Phase 4 end to end: the decider's plain command is carried out through home_control,
the specialist only speaks, and every safety seam sees a real executed tool."""
import json
import threading
import time

import httpx
from langchain_core.messages import AIMessage

from aerys_v2.config import Settings
from aerys_v2.reflex import LAST_REFLEX, LiveReflexRecord, REFLEX_SURFACE, live_router_for
from aerys_v2.router import RouteDecision
from aerys_v2.service import _direct_device_seed, _needs_spoken_followup, ask
from aerys_v2.tools.home_control import WRITE_OK_PREFIX, build_home_control_tool, canary_set, device_target_choices

CANARY = canary_set("light.sunroom_light_1,light.sunroom_light_2,lock.jolteon_door_lock")


class FakeHA:
    def __init__(self, changed=True):
        self.bodies = []
        self.changed = changed

    def handler(self, request):
        if request.url.path.startswith("/api/services/"):
            self.bodies.append(json.loads(request.content or b"{}"))
            ids = self.bodies[-1]["entity_id"]
            ids = ids if isinstance(ids, list) else [ids]
            return httpx.Response(200, json=[{"entity_id": i, "state": "off"} for i in ids] if self.changed else [])
        return httpx.Response(200, json={"state": "off", "attributes": {"friendly_name": "x"}})

    def tool(self):
        return build_home_control_tool(base_url="http://ha.test:8123", token="t", canary_entities=CANARY,
                                       client=httpx.Client(transport=httpx.MockTransport(self.handler)))


class Graph:
    """Chat graph stand-in: minimal state so history landing works."""
    def __init__(self):
        self.updates = []

    def invoke(self, inp, config):
        return {"messages": [AIMessage(content="chat")]}

    def get_state(self, config):
        from types import SimpleNamespace
        return SimpleNamespace(values={"messages": []})

    def update_state(self, config, values, as_node=None):
        self.updates.append(values["messages"])


class ActionGraph:
    """Records what it was seeded with; answers with the specialist's sentence."""
    def __init__(self, tool):
        self.home_control_tool = tool
        self.seeds = []

    def invoke(self, inp, config):
        self.seeds.append(inp["messages"])
        return {"messages": [*inp["messages"], AIMessage(content="Sunroom's off.")]}


class Recorder:
    def __init__(self):
        self.rows, self.done = [], threading.Event()

    def __call__(self, row):
        self.rows.append(row); self.done.set()


def settings(**kw):
    return Settings(_env_file=None, anthropic_api_key="t", reflex_mode="live", typesafe_api_key="k", **kw)


def jev(route="action", **device):
    d = {"is_device_command": 0.97, "is_state_question": 0.02, "is_compound": 0.03,
         "device_target": {"choice": "sunroom", "confidence": 0.95},
         "device_action": {"choice": "turn_off", "confidence": 0.96}}
    d.update(device)
    return {"route": route, "p_action": 0.99, "confidence": 0.98, "tier": "fast", "tier_score": 0.1,
            "unaddressed": 0.05, "model": "jev-1.13.0", "input_tokens": 700, "latency_ms": 220, "device": d}


def decider(ha_tool, **device):
    targets = device_target_choices(CANARY)
    router = lambda text: RouteDecision(route="action", ack="On it.", tier="standard")
    return live_router_for(settings(), lambda t, c: jev(**device), router,
                           device_targets=targets, canary_entities=CANARY)


def test_text_path_writes_once_then_the_specialist_only_speaks():
    ha = FakeHA(); tool = ha.tool()
    action = ActionGraph(tool); graph = Graph(); rec = Recorder()
    reply = ask(graph, "turn off the sunroom", identity={"platform": "discord", "channel_kind": "dm",
                                                        "privacy_context": "private"},
                thread_id="person:x", router=decider(tool), action_graph=action, reflex=None, record_turn=rec)
    assert reply == "Sunroom's off."
    # ONE write, through the real tool, for both sunroom lights in one call.
    assert len(ha.bodies) == 1 and set(ha.bodies[0]["entity_id"]) == {"light.sunroom_light_1", "light.sunroom_light_2"}
    # The specialist saw the executed call, not a request to choose one.
    seed = action.seeds[0]
    assert seed[-2].tool_calls[0]["name"] == "home_control"
    assert seed[-2].tool_calls[0]["args"] == {"operation": "turn_off", "entity_id": "sunroom"}
    assert seed[-1].type == "tool" and seed[-1].content.startswith(WRITE_OK_PREFIX)
    # History landed, audit row carries the real tool call and the direct verdict.
    assert rec.done.wait(3)
    row = rec.rows[0]
    assert json.loads(row["tool_calls"])[0]["name"] == "home_control"
    reflex = json.loads(row["reflex"])
    assert reflex["device"]["direct"]["ok"] is True and reflex["device"]["direct"]["executed"] is True
    assert graph.updates and graph.updates[-1][-1].content == "Sunroom's off."


def test_deny_listed_and_unsure_turns_take_the_old_path():
    ha = FakeHA(); tool = ha.tool(); action = ActionGraph(tool); graph = Graph(); rec = Recorder()
    ask(graph, "unlock the door", identity={"privacy_context": "private"}, thread_id="person:x",
        router=decider(tool, device_target={"choice": "jolteon door lock", "confidence": 0.99},
                       device_action={"choice": "turn_on", "confidence": 0.99}),
        action_graph=action, reflex=None, record_turn=rec)
    assert ha.bodies == []                      # nothing written by the code path
    assert action.seeds[0][-1].type == "human"  # the specialist got the plain request
    assert rec.done.wait(3)
    assert "sensitive domain" in json.loads(rec.rows[0]["reflex"])["device"]["direct"]["reason"]


def test_kill_switch_keeps_live_routing_but_not_direct_writes():
    ha = FakeHA(); tool = ha.tool(); action = ActionGraph(tool)
    targets = device_target_choices(CANARY)
    router = lambda text: RouteDecision(route="action", ack="On it.")
    decide = live_router_for(settings(reflex_direct=False), lambda t, c: jev(), router,
                             device_targets=None, canary_entities=CANARY)
    REFLEX_SURFACE.set("discord_dm")
    d = decide("turn off the sunroom")
    assert d.route == "action"
    assert LAST_REFLEX.get().record["command"] is None
    assert _direct_device_seed(action, ["seed"]) == ["seed"] and ha.bodies == []


def test_direct_seed_without_a_command_or_tool_is_untouched():
    LAST_REFLEX.set(None)
    assert _direct_device_seed(ActionGraph(None), ["seed"]) == ["seed"]
    LAST_REFLEX.set(LiveReflexRecord({"command": {"operation": "turn_off", "entity_id": "sunroom"}}, None, {}))
    assert _direct_device_seed(ActionGraph(None), ["seed"]) == ["seed"]


def test_refusal_reaches_the_specialist_and_a_raise_becomes_an_honest_failure():
    class Raising:
        def invoke(self, cmd):
            raise RuntimeError("HA exploded")
    LAST_REFLEX.set(LiveReflexRecord({"command": {"operation": "turn_off", "entity_id": "sunroom"}, "device": {}}, None, {}))
    seed = _direct_device_seed(ActionGraph(Raising()), ["seed"])
    assert seed[-1].type == "tool" and "FAILED" in seed[-1].content and "HA exploded" in seed[-1].content
    # a refusal string is still a completed tool note -> the voice follow-up SPEAKS it
    assert _needs_spoken_followup(seed, 0.5, 5.0) is True
    # and a fast clean write stays SILENT exactly as before
    ha = FakeHA(); tool = ha.tool()
    LAST_REFLEX.set(LiveReflexRecord({"command": {"operation": "turn_off", "entity_id": "sunroom"}, "device": {}}, None, {}))
    ok_seed = _direct_device_seed(ActionGraph(tool), ["seed"])
    assert _needs_spoken_followup(ok_seed, 0.5, 5.0) is False


def test_voice_plain_command_speaks_no_ack_and_still_acts():
    from aerys_v2.factory import build_graph
    ha = FakeHA(); tool = ha.tool(); action = ActionGraph(tool); rec = Recorder()
    # A real chat graph is needed for the voice path's history landing; use the
    # in-memory one from factory with a fake model.
    from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
    chat = build_graph(GenericFakeChatModel(messages=iter([AIMessage(content="chat")])), "soul")
    decide = decider(tool)
    reply = ask(chat, "turn off the sunroom", identity={"voice": True, "privacy_context": "private"},
                thread_id="person:v", router=decide, action_graph=action, reflex=None, record_turn=rec)
    assert reply == ""                       # no ack spoken
    assert rec.done.wait(3)
    assert len(ha.bodies) == 1               # the write still happened, once
    row = rec.rows[0]
    assert row["emitted_reply"] == "" and json.loads(row["reflex"])["silent_ack"] is True
    assert json.loads(row["tool_calls"])[0]["name"] == "home_control"


def test_voice_keeps_the_ack_when_silent_ack_is_off_or_no_command():
    from aerys_v2.factory import build_graph
    from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
    ha = FakeHA(); tool = ha.tool(); action = ActionGraph(tool); rec = Recorder()
    chat = build_graph(GenericFakeChatModel(messages=iter([AIMessage(content="chat")])), "soul")
    targets = device_target_choices(CANARY)
    router = lambda text: RouteDecision(route="action", ack="On it.")
    decide = live_router_for(settings(reflex_direct_silent_ack=False), lambda t, c: jev(), router,
                             device_targets=targets, canary_entities=CANARY)
    reply = ask(chat, "turn off the sunroom", identity={"voice": True, "privacy_context": "private"},
                thread_id="person:v2", router=decide, action_graph=action, reflex=None, record_turn=rec)
    assert reply == "On it."
    assert rec.done.wait(3) and len(ha.bodies) == 1
