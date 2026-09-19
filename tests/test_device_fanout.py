"""Phase 4 verdicts are pure functions: numbers and colors from code, targets from the allowlist."""
import pytest

from aerys_v2.config import Settings
from aerys_v2.reflex import (NONE_TARGET, brightness_from_text, color_from_text, device_questions,
                             plain_device_command)
from aerys_v2.tools.home_control import canary_set, device_target_choices, resolve_targets

CANARY = canary_set("light.sunroom_light_1,light.sunroom_light_2,switch.office_light_1,switch.office_fan,"
                    "lock.jolteon_door_lock,switch.jolteon_ev_charging,switch.jolteon_climate,"
                    "fan.omnibreeze_tower_fan_5m_f")


def settings(**kw):
    return Settings(_env_file=None, anthropic_api_key="t", reflex_mode="live", typesafe_api_key="k", **kw)


@pytest.mark.parametrize("text,pct", [("set it to 40%", 40), ("30 percent please", 30), ("go to half", 50),
                                      ("full brightness", 100), ("make it dim", None), ("a bit dimmer", None), ("keep it low", 20),
                                      ("200%", None), ("turn on", None)])
def test_brightness_from_text(text, pct):
    assert brightness_from_text(text) == pct


@pytest.mark.parametrize("text,color", [("turn the sunroom red", "red"), ("make it warm white", "warm white"),
                                        ("#FF8000 please", "#ff8000"), ("turn it on", None),
                                        ("whiteboard mode", None)])
def test_color_from_text(text, color):
    assert color_from_text(text) == color


def test_every_target_choice_resolves_through_the_tool_resolver():
    choices = device_target_choices(CANARY)
    assert "sunroom" in choices and "office fan" in choices and NONE_TARGET not in choices
    for name in choices:
        targets, problem = resolve_targets(name, "turn_on", CANARY)
        assert targets and not problem, name
    q = device_questions(choices)
    assert NONE_TARGET in q["device_target"]["criteria"]
    assert set(q) == {"is_device_command", "is_state_question", "is_compound", "device_target", "device_action"}


def record(target="sunroom", action="turn_off", cmd=0.95, state=0.05, compound=0.05, tconf=0.9, aconf=0.9,
           route="action", by="jev"):
    return {"decided": {"route": route}, "decided_by": by, "device": {
        "is_device_command": cmd, "is_state_question": state, "is_compound": compound,
        "device_target": {"choice": target, "confidence": tconf},
        "device_action": {"choice": action, "confidence": aconf}}}


def test_plain_command_happy_path_and_reason_on_the_row():
    rec = record()
    cmd = plain_device_command(rec, "turn off the sunroom", settings(), CANARY)
    assert cmd == {"operation": "turn_off", "entity_id": "sunroom"}
    assert rec["device"]["direct"]["ok"] is True
    assert set(rec["device"]["direct"]["targets"]) == {"light.sunroom_light_1", "light.sunroom_light_2"}


@pytest.mark.parametrize("kw,reason", [
    (dict(cmd=0.6), "not clearly a device command"),
    (dict(state=0.5), "state question"),
    (dict(compound=0.5), "compound"),
    (dict(target=NONE_TARGET), "target unsure"),
    (dict(tconf=0.5), "target unsure"),
    (dict(action="other"), "action unsure"),
    (dict(aconf=0.4), "action unsure"),
    (dict(route="chat"), "not a jev action route"),
    (dict(by="router"), "not a jev action route"),
    (dict(target="jolteon door lock"), "deny-listed"),
    (dict(target="jolteon ev charging"), "deny-listed"),
    (dict(target="jolteon climate"), "deny-listed"),
])
def test_plain_command_gates(kw, reason):
    rec = record(**kw)
    assert plain_device_command(rec, "turn off the sunroom", settings(), CANARY) is None
    assert reason in rec["device"]["direct"]["reason"]


def test_brightness_and_color_come_from_the_text_or_fall_through():
    assert plain_device_command(record(action="set_brightness"), "sunroom to 40%", settings(), CANARY) == {
        "operation": "set_brightness", "entity_id": "sunroom", "brightness_pct": 40}
    rec = record(action="set_brightness")
    assert plain_device_command(rec, "sunroom a little dimmer", settings(), CANARY) is None
    assert "specialist" in rec["device"]["direct"]["reason"]
    assert plain_device_command(record(action="set_color"), "turn the sunroom red", settings(), CANARY) == {
        "operation": "set_color", "entity_id": "sunroom", "color": "red"}
    assert plain_device_command(record(action="set_color"), "change the sunroom color", settings(), CANARY) is None


def test_domain_gate_and_kill_switch():
    rec = record(target="office fan")
    assert plain_device_command(rec, "office fan off", settings(reflex_direct_domains="light"), CANARY) is None
    assert "domain not direct" in rec["device"]["direct"]["reason"]
    rec = record()
    assert plain_device_command(rec, "sunroom off", settings(reflex_direct=False), CANARY) is None
    assert rec["device"]["direct"]["reason"] == "disabled"
