"""HA specialist v1 (2026-09-04) — offline tests.

The tool subgraph became a stateless SPECIALIST: seeded with the request plus a
few prior REQUESTS (never prior assistant turns), its first pass forced to call a
tool, on its own model knob, with room-level targeting and read-back verification
in home_control. Traced motivation: "dim the sunroom lights by half" took 28s and
five Sonnet round-trips because a prior "I can't set brightness" line rode into
the tool loop and the tool took one guessed id per call.
"""
import json

import httpx
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from aerys_v2.factory import SPECIALIST_CHARTER, ToolModelPair, build_action_graph
from aerys_v2.service import ACTION_SEED_TRUNCATE_AT, _action_history_seed
from aerys_v2.services.content_privacy import CONTENT_PRIVACY_KEY
import aerys_v2.tools.home_control as hc
from aerys_v2.tools.home_control import (
    NOOP_OK_PREFIX,
    WRITE_OK_PREFIX,
    build_home_control_tool,
    canary_set,
)

hc._VERIFY_DELAYS_S = (0.0, 0.0)  # the read-back grace is real time; tests don't wait

# ---- ToolModelPair: conversation free, specialist forced-then-free, fast pair ----


class Recorder:
    def __init__(self, name):
        self.name = name
        self.calls = []

    def invoke(self, messages, **kw):
        self.calls.append(list(messages))
        return AIMessage(content=self.name)


def _trio():
    conv, auto, forced = Recorder("conv"), Recorder("auto"), Recorder("forced")
    fa, ff = Recorder("fast_auto"), Recorder("fast_forced")
    return conv, auto, forced, fa, ff, ToolModelPair(conv, auto, forced, fa, ff)


FIRST = [HumanMessage(content="sys"), HumanMessage(content="dim the sunroom")]
AFTER_TOOL = [
    *FIRST,
    AIMessage(content="", tool_calls=[{"name": "home_control", "args": {}, "id": "c1"}]),
    ToolMessage(content="Done: set_brightness 50% sent", tool_call_id="c1"),
]


def test_pair_forces_the_first_pass_only():
    conv, auto, forced, fa, ff, pair = _trio()
    assert pair.invoke(FIRST, specialist=True).content == "forced"
    assert pair.invoke(AFTER_TOOL, specialist=True).content == "auto"
    assert not conv.calls and not fa.calls and not ff.calls


def test_pair_fast_tier_uses_the_fast_models():
    conv, auto, forced, fa, ff, pair = _trio()
    assert pair.invoke(FIRST, specialist=True, fast=True).content == "fast_forced"
    assert pair.invoke(AFTER_TOOL, specialist=True, fast=True).content == "fast_auto"
    assert not auto.calls and not forced.calls


def test_pair_conversation_is_never_forced_and_never_fast():
    # Voice banter rides this graph as conversation: the daily driver, free.
    conv, auto, forced, fa, ff, pair = _trio()
    assert pair.invoke(FIRST).content == "conv"
    assert pair.invoke(FIRST, fast=True).content == "conv"
    assert not forced.calls and not ff.calls


def test_pair_without_fast_models_falls_back_to_standard():
    conv, auto, forced = Recorder("conv"), Recorder("auto"), Recorder("forced")
    pair = ToolModelPair(conv, auto, forced)
    assert pair.invoke(FIRST, specialist=True, fast=True).content == "forced"


def test_pair_honesty_bounce_is_a_free_pass():
    # The gate's bounce re-invokes with the model's own no-tool answer appended —
    # an assistant turn is in view, so forcing must NOT apply (the loop must be
    # able to end on a plain answer).
    conv, auto, forced, fa, ff, pair = _trio()
    bounced = [HumanMessage(content="x"), AIMessage(content="I can't"), HumanMessage(content="use a tool")]
    assert pair.invoke(bounced, specialist=True).content == "auto"


# ---- the charter rides the prompt slot the soul used to -------------------------


class RecordingToolModel:
    def __init__(self):
        self.prompts = []

    def invoke(self, messages, **kw):
        self.prompts.append(list(messages))
        return AIMessage(content="done")


def test_specialist_prompt_leads_with_charter_not_soul():
    model = RecordingToolModel()
    graph = build_action_graph(model, "SOUL: a Curious Sentinel", tools=[])
    ident = {"user_id": "p1", "display_name": "Chris"}
    graph.invoke(
        {"messages": [HumanMessage(content="turn off the sunroom lights")]},
        {"configurable": {"identity": ident, "specialist": True}},
    )
    system = model.prompts[0][0].content
    assert system.startswith(SPECIALIST_CHARTER)
    assert "Curious Sentinel" not in system  # nothing of the soul leaks in
    # voice banter (no flag) keeps the soul — conversation still sounds like her
    graph.invoke(
        {"messages": [HumanMessage(content="tell me more")]},
        {"configurable": {"identity": ident}},
    )
    assert model.prompts[1][0].content.startswith("SOUL: a Curious Sentinel")


def test_specialist_prompt_carries_the_fast_tier_into_the_pair():
    class TierRecorder(ToolModelPair):
        def __init__(self):
            self.seen = []
        def invoke(self, messages, *, specialist=False, fast=False, **kw):
            self.seen.append((specialist, fast))
            return AIMessage(content="done")
    model = TierRecorder()
    graph = build_action_graph(model, "soul", tools=[])
    ident = {"user_id": "p1", "display_name": "Chris"}
    graph.invoke({"messages": [HumanMessage(content="x")]},
                 {"configurable": {"identity": ident, "specialist": True, "tier": "fast"}})
    graph.invoke({"messages": [HumanMessage(content="x")]},
                 {"configurable": {"identity": ident, "specialist": True, "tier": "standard"}})
    graph.invoke({"messages": [HumanMessage(content="x")]},
                 {"configurable": {"identity": ident, "tier": "fast"}})
    assert model.seen == [(True, True), (True, False), (False, True)]


# ---- the seed: prior REQUESTS only, truncated, current last ------------------------


class StubGraph:
    def __init__(self, messages):
        self._messages = messages

    def get_state(self, config):
        class S:
            values = {"messages": self._messages}
        return S()


def test_seed_folds_last_three_prior_requests_into_one_message():
    prior = []
    for i in range(1, 6):
        prior.append(HumanMessage(content=f"request {i}", additional_kwargs={CONTENT_PRIVACY_KEY: "public"}))
        prior.append(AIMessage(content=f"I can't do {i}"))  # the stale-belief carrier
    seeded = _action_history_seed(
        StubGraph(prior), {"thread_id": "t", "identity": {"privacy_context": "private"}}, "turn them back on", specialist=True
    )
    assert len(seeded) == 1 and seeded[0].type == "human"
    body = seeded[0].content
    assert "ALREADY HANDLED" in body and "Do NOT redo" in body
    assert "- request 3\n- request 4\n- request 5" in body
    assert "request 2" not in body and "I can't do" not in body
    assert body.endswith("The request to carry out now:\nturn them back on")


def test_seed_with_no_priors_is_just_the_request():
    seeded = _action_history_seed(StubGraph([]), {"identity": {"privacy_context": "private"}}, "dim the sunroom", specialist=True)
    assert [m.content for m in seeded] == ["dim the sunroom"]


def test_seed_truncates_long_prior_requests_and_keeps_current_tags():
    long = "x" * (ACTION_SEED_TRUNCATE_AT + 50)
    prior = [HumanMessage(content=long, additional_kwargs={CONTENT_PRIVACY_KEY: "public"}), AIMessage(content="ok")]
    seeded = _action_history_seed(StubGraph(prior), {"identity": {"privacy_context": "private"}}, "now", specialist=True)
    assert ("x" * ACTION_SEED_TRUNCATE_AT + "…") in seeded[0].content
    assert ("x" * (ACTION_SEED_TRUNCATE_AT + 1)) not in seeded[0].content
    assert seeded[0].content.endswith("\nnow")


def test_seed_public_room_still_drops_private_prior_requests():
    prior = [
        HumanMessage(content="my resting HR was 48", additional_kwargs={CONTENT_PRIVACY_KEY: "private"}),
        AIMessage(content="noted"),
        HumanMessage(content="turn off the office lights", additional_kwargs={CONTENT_PRIVACY_KEY: "public"}),
        AIMessage(content="off"),
    ]
    seeded = _action_history_seed(StubGraph(prior), {"identity": {"privacy_context": "public"}}, "turn them back on", specialist=True)
    body = seeded[0].content
    assert "my resting HR" not in body and "- turn off the office lights" in body
    assert body.endswith("turn them back on")


def test_seed_without_specialist_keeps_the_full_exchange():
    # Voice banter rides the tool graph as conversation: her replies are context.
    prior = [HumanMessage(content="remember the lighthouse"), AIMessage(content="I remember.")]
    seeded = _action_history_seed(StubGraph(prior), {"identity": {"privacy_context": "private"}}, "tell me more")
    assert [m.content for m in seeded] == ["remember the lighthouse", "I remember.", "tell me more"]


def test_seed_escalated_turn_uses_the_checkpointed_current_request():
    # Escalated: the chat invoke already checkpointed the human turn + a handoff line.
    prior = [
        HumanMessage(content="earlier ask"), AIMessage(content="sure"),
        HumanMessage(content="dim the sunroom by half"), AIMessage(content="<<handoff>>"),
    ]
    seeded = _action_history_seed(
        StubGraph(prior), {"identity": {"privacy_context": "private"}}, "dim the sunroom by half", escalated=True, specialist=True
    )
    assert len(seeded) == 1
    assert "- earlier ask" in seeded[0].content and seeded[0].content.endswith("dim the sunroom by half")
    assert "<<handoff>>" not in seeded[0].content and "sure" not in seeded[0].content


# ---- home_control: room targeting, one call, read-back verification -----------------

SUNROOM = "light.sunroom_light_1,light.sunroom_light_2,light.sunroom_light_3,light.sunroom_light_4"
OFFICE = "switch.office_light_1,switch.office_light_2"


class FakeHA:
    """Records requests; GET state is scriptable per entity; POST echoes changed ids."""

    def __init__(self, states=None, changed_ids=None):
        self.requests = []
        self.bodies = []
        self.states = states or {}           # entity -> (state, brightness_pct|None)
        self.changed_ids = changed_ids        # None = echo every requested entity

    def handler(self, req: httpx.Request) -> httpx.Response:
        self.requests.append((req.method, req.url.path))
        if req.url.path.startswith("/api/states/"):
            eid = req.url.path.rsplit("/", 1)[-1]
            if eid not in self.states:
                return httpx.Response(404)
            state, pct = self.states[eid]
            attrs = {"friendly_name": eid}
            if pct is not None:
                attrs["brightness"] = round(pct / 100 * 255)
            return httpx.Response(200, json={"state": state, "attributes": attrs})
        body = json.loads(req.content or b"{}")
        self.bodies.append(body)
        ids = body["entity_id"] if isinstance(body["entity_id"], list) else [body["entity_id"]]
        changed = ids if self.changed_ids is None else [i for i in ids if i in self.changed_ids]
        return httpx.Response(200, json=[{"entity_id": i, "state": "on"} for i in changed])

    def tool(self, canary=SUNROOM + "," + OFFICE):
        return build_home_control_tool(
            base_url="http://ha.test:8123", token="t",
            canary_entities=canary_set(canary),
            client=httpx.Client(transport=httpx.MockTransport(self.handler)),
        )


def test_room_name_resolves_every_match_into_one_call():
    ha = FakeHA()
    out = ha.tool().invoke({"operation": "set_brightness", "entity_id": "sunroom lights", "brightness_pct": 50})
    assert out.startswith(WRITE_OK_PREFIX)
    assert [r for r in ha.requests if r[0] == "POST"] == [("POST", "/api/services/light/turn_on")]
    assert ha.bodies[0] == {"entity_id": SUNROOM.split(","), "brightness_pct": 50}


def test_room_name_variants_and_plurals_resolve():
    for name in ("sun room", "the sunroom", "Sunroom light", "sunroom"):
        ha = FakeHA()
        ha.tool().invoke({"operation": "turn_off", "entity_id": name})
        assert ha.bodies[0]["entity_id"] == SUNROOM.split(","), name
    ha = FakeHA()
    ha.tool().invoke({"operation": "turn_off", "entity_id": "office lights"})
    assert ha.bodies[0]["entity_id"] == OFFICE.split(",")
    assert ha.requests[-1] == ("POST", "/api/services/switch/turn_off")


def test_comma_list_of_exact_ids_is_one_call():
    ha = FakeHA()
    ha.tool().invoke({"operation": "turn_on", "entity_id": "light.sunroom_light_1, light.sunroom_light_2"})
    assert ha.bodies[0]["entity_id"] == ["light.sunroom_light_1", "light.sunroom_light_2"]


def test_generic_only_name_becomes_a_question_never_a_house_wide_write():
    # "turn the lights off" with no room: every allowlisted entity contains "light".
    for name in ("the lights", "lights", "all the lights", "switches"):
        ha = FakeHA()
        out = ha.tool().invoke({"operation": "turn_off", "entity_id": name})
        assert out.startswith("Which ones?"), name
        assert "office" in out and "sunroom" in out  # the choices, by place
        assert ha.requests == []                      # nothing written


def test_unknown_room_is_an_honest_refusal_and_ha_untouched():
    ha = FakeHA()
    out = ha.tool().invoke({"operation": "turn_on", "entity_id": "garage"})
    assert out.startswith("Refused:") and "light.sunroom_light_1" in out
    assert ha.requests == []


def test_unknown_name_on_a_read_points_at_search_entities():
    ha = FakeHA()
    out = ha.tool().invoke({"operation": "get_state", "entity_id": "jolteon"})
    assert "search_entities" in out and ha.requests == []


def test_partial_allowlist_refuses_the_whole_call():
    ha = FakeHA()
    out = ha.tool(canary="light.sunroom_light_1").invoke(
        {"operation": "turn_on", "entity_id": "light.sunroom_light_1,light.sunroom_light_2"}
    )
    assert out.startswith("Refused: light.sunroom_light_2 is not")
    assert ha.requests == []


def test_mixed_domains_become_one_call_per_domain():
    ha = FakeHA()
    out = ha.tool().invoke({"operation": "turn_on", "entity_id": "light.sunroom_light_1,switch.office_light_1"})
    assert out.startswith(WRITE_OK_PREFIX)
    posts = [r for r in ha.requests if r[0] == "POST"]
    assert posts == [("POST", "/api/services/light/turn_on"), ("POST", "/api/services/switch/turn_on")]


def test_brightness_on_a_mixed_name_dims_the_lights_and_says_switches_were_left():
    canary = SUNROOM + ",switch.sunroom_fan"
    ha = FakeHA()
    out = ha.tool(canary=canary).invoke({"operation": "set_brightness", "entity_id": "sunroom", "brightness_pct": 40})
    assert out.startswith(WRITE_OK_PREFIX) and "switch.sunroom_fan" in out and "left as is" in out
    assert [r for r in ha.requests if r[0] == "POST"] == [("POST", "/api/services/light/turn_on")]
    assert ha.bodies[0]["entity_id"] == SUNROOM.split(",")


def test_whole_token_matching_is_by_token_not_substring():
    # "office" is a whole token of switch.office_light_* AND light.office_closet_1
    # (a closet light in the office IS an office light) — both go, one call per
    # domain. "closet" alone reaches only the closet; "off" (a substring of
    # "office") reaches nothing.
    canary = OFFICE + ",light.office_closet_1"
    ha = FakeHA()
    out = ha.tool(canary=canary).invoke({"operation": "turn_off", "entity_id": "office lights"})
    assert out.startswith(WRITE_OK_PREFIX)
    assert [b["entity_id"] for b in ha.bodies] == ["light.office_closet_1", OFFICE.split(",")]
    ha = FakeHA()
    ha.tool(canary=canary).invoke({"operation": "turn_off", "entity_id": "closet"})
    assert ha.bodies[0]["entity_id"] == "light.office_closet_1"
    ha = FakeHA()
    out = ha.tool(canary=canary).invoke({"operation": "turn_off", "entity_id": "off"})
    assert out.startswith("Refused:") and ha.requests == []


def test_malformed_state_json_is_an_honest_string():
    class Odd(FakeHA):
        def handler(self, req):
            if req.url.path.startswith("/api/states/"):
                return httpx.Response(200, content=b"<html>not json")
            return super().handler(req)
    ha = Odd()
    out = ha.tool().invoke({"operation": "get_state", "entity_id": "light.sunroom_light_1"})
    assert "unreadable" in out


def test_already_there_is_verified_by_reading_the_device_back():
    # HA accepts, reports nothing changed; read-back says every light is already at 50%.
    ha = FakeHA(states={e: ("on", 50) for e in SUNROOM.split(",")}, changed_ids=[])
    out = ha.tool().invoke({"operation": "set_brightness", "entity_id": "sunroom", "brightness_pct": 50})
    assert out.startswith(NOOP_OK_PREFIX) and "already on at 50%" in out
    gets = [r for r in ha.requests if r[0] == "GET"]
    assert len(gets) == 4  # one read-back per unverified target — the check can fail


def test_late_reporting_light_is_caught_by_the_second_read():
    # First read-back still shows the old state (cloud light lagging), the retry
    # shows it applied -> "already there", not a false "not applied".
    class Lagging(FakeHA):
        def __init__(self):
            super().__init__(states={"light.sunroom_light_1": ("off", None)}, changed_ids=[])
            self.reads = 0
        def handler(self, req):
            if req.url.path.startswith("/api/states/"):
                self.reads += 1
                if self.reads >= 2:
                    self.states["light.sunroom_light_1"] = ("on", 50)
            return super().handler(req)
    ha = Lagging()
    out = ha.tool().invoke({"operation": "set_brightness", "entity_id": "light.sunroom_light_1", "brightness_pct": 50})
    assert out.startswith(NOOP_OK_PREFIX) and ha.reads == 2


def test_dropped_command_is_reported_not_claimed():
    # HA accepts, reports nothing changed; read-back shows one light still OFF.
    states = {e: ("on", 50) for e in SUNROOM.split(",")}
    states["light.sunroom_light_3"] = ("off", None)
    ha = FakeHA(states=states, changed_ids=[])
    out = ha.tool().invoke({"operation": "set_brightness", "entity_id": "sunroom", "brightness_pct": 50})
    assert not out.startswith(WRITE_OK_PREFIX) and not out.startswith(NOOP_OK_PREFIX)
    assert "light.sunroom_light_3" in out and "NO state change" in out


def test_partial_change_verifies_only_the_rest():
    ha = FakeHA(states={"light.sunroom_light_2": ("off", None)}, changed_ids=["light.sunroom_light_1"])
    out = ha.tool().invoke({"operation": "turn_off", "entity_id": "light.sunroom_light_1,light.sunroom_light_2"})
    assert out.startswith(WRITE_OK_PREFIX) and "light.sunroom_light_2 already off" in out
    assert ("GET", "/api/states/light.sunroom_light_2") in ha.requests
    assert ("GET", "/api/states/light.sunroom_light_1") not in ha.requests


def test_toggle_with_no_change_cannot_verify_so_it_warns():
    ha = FakeHA(states={"light.sunroom_light_1": ("on", None)}, changed_ids=[])
    out = ha.tool().invoke({"operation": "toggle", "entity_id": "light.sunroom_light_1"})
    assert "NO state change" in out
    assert not any(r[0] == "GET" for r in ha.requests)


def test_get_state_by_room_returns_a_list():
    ha = FakeHA(states={e: ("on", 30) for e in SUNROOM.split(",")})
    out = json.loads(ha.tool().invoke({"operation": "get_state", "entity_id": "sunroom"}))
    assert [o["entity_id"] for o in out] == SUNROOM.split(",")
    assert all(o["brightness_pct"] == 30 for o in out)


# ---- no_action: the forced first pass's honest exit --------------------------------

from aerys_v2.tools.no_action import NO_ACTION_PREFIX, build_no_action_tool  # noqa: E402
from aerys_v2.config import Settings  # noqa: E402
from aerys_v2.factory import action_tools_for  # noqa: E402


def test_no_action_returns_the_sentence_to_say_and_claims_nothing():
    out = build_no_action_tool().invoke({"reason": "Which room did you mean?"})
    assert out.startswith(NO_ACTION_PREFIX) and "Which room did you mean?" in out
    assert not out.startswith(WRITE_OK_PREFIX)


def test_no_action_is_registered_only_when_something_real_is_armed():
    assert action_tools_for(Settings(_env_file=None, anthropic_api_key="k")) == []
    armed = Settings(_env_file=None, anthropic_api_key="k", ha_token="t")
    names = [getattr(t, "name", "") for t in action_tools_for(armed)]
    assert "home_control" in names and names[-1] == "no_action"
