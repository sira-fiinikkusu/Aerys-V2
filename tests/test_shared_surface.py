"""Owner ask 8/16 — household-surface awareness on Sticky voice turns.

The Bearer authenticates the DEVICE as owner infrastructure; the SPEAKER may
be Megan or a guest. Turns from designated shared devices carry a prompt
block telling her to stay identity-humble and keep his private specifics off
the shared speaker. These tests pin the gate (device-id membership) and both
absences (other devices; feature off).
"""

from langchain_core.messages import AIMessage, HumanMessage

from aerys_v2.factory import _shared_surface_note, build_action_graph

STICKY = "b9cde1126a975e27d4e4b852f2e798b8"


class RecordingModel:
    def __init__(self) -> None:
        self.systems: list[str] = []

    def invoke(self, messages):
        self.systems.append(str(messages[0].content))
        return AIMessage(content="hi there")


def run_turn(shared_ids, device_id):
    model = RecordingModel()
    graph = build_action_graph(
        model, "SOUL", tools=[], shared_surface_ids=frozenset(shared_ids)
    )
    identity = {"user_id": "owner-1", "display_name": "Chris", "voice": True}
    if device_id:
        identity["device_id"] = device_id
    graph.invoke(
        {"messages": [HumanMessage(content="what's the weather?")]},
        {"configurable": {"thread_id": "t", "identity": identity}},
    )
    return model.systems[0]


def test_sticky_turns_carry_the_household_block():
    system = run_turn({STICKY}, STICKY)
    assert "HOUSEHOLD surface" in system
    assert "not necessarily Chris" in system
    assert "identity-neutral" in system


def test_other_devices_and_unset_feature_stay_clean():
    assert "HOUSEHOLD" not in run_turn({STICKY}, "office-satellite-dev")
    assert "HOUSEHOLD" not in run_turn({STICKY}, None)
    assert "HOUSEHOLD" not in run_turn(set(), STICKY)


def test_note_helper_is_pure_and_gated():
    assert _shared_surface_note({}, frozenset()) == ""
    assert _shared_surface_note({"device_id": STICKY}, frozenset({STICKY})) != ""
    assert _shared_surface_note({"device_id": "x"}, frozenset({STICKY})) == ""
