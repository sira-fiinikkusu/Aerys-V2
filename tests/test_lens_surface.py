"""Owner ask 8/26 — lens-native replies on the G2 glasses (the 'is it really you' fix).

The g2-bridge display-summarizes any reply too long for the lens: another model
paraphrasing her, invisible to her by construction (she denied a summarizer
existed the night the owner asked — she was right from where she stands). The
fix teaches her the surface instead: identity.surface == 'lens' (sent by the
bridge on /ask) appends LENS_SURFACE_OVERLAY so she writes short natively and
the summarizer demotes to a rare overrun fallback.

These tests pin: the gate (surface value, both graphs), both absences (no
surface / other surfaces stay byte-for-byte clean), the ordering contract
(lens text follows the voice styling it must override), and the helper.
"""

from langchain_core.messages import AIMessage, HumanMessage

from aerys_v2.factory import (
    LENS_SURFACE_OVERLAY,
    VOICE_BANTER_OVERLAY,
    build_action_graph,
    build_graph,
)
from aerys_v2.state import is_lens_surface

LENS_MARK = "READ, NOT HEARD"


class RecordingModel:
    def __init__(self) -> None:
        self.systems: list[str] = []

    def invoke(self, messages):
        self.systems.append(str(messages[0].content))
        return AIMessage(content="hi there")


def action_system(identity: dict) -> str:
    model = RecordingModel()
    graph = build_action_graph(model, "SOUL", tools=[])
    graph.invoke(
        {"messages": [HumanMessage(content="how's the house?")]},
        {"configurable": {"thread_id": "t", "identity": identity}},
    )
    return model.systems[0]


def chat_system(identity: dict) -> str:
    model = RecordingModel()
    graph = build_graph(model, "SOUL")
    graph.invoke(
        {"messages": [HumanMessage(content="how's the house?")]},
        {"configurable": {"thread_id": "t", "identity": identity}},
    )
    return model.systems[0]


def test_lens_voice_turn_carries_the_overlay_on_the_action_graph():
    system = action_system(
        {"user_id": "owner-1", "display_name": "Chris", "voice": True, "surface": "lens"}
    )
    assert LENS_MARK in system
    assert "350 characters" in system


def test_lens_overlay_follows_voice_banter_so_its_rules_win():
    system = action_system(
        {"user_id": "owner-1", "display_name": "Chris", "voice": True, "surface": "lens"}
    )
    banter_head = VOICE_BANTER_OVERLAY[:40]
    lens_head = LENS_SURFACE_OVERLAY[:40]
    assert banter_head in system and lens_head in system
    assert system.index(lens_head) > system.index(banter_head)


def test_lens_without_voice_still_styles_the_action_graph():
    system = action_system({"user_id": "owner-1", "display_name": "Chris", "surface": "lens"})
    assert LENS_MARK in system


def test_non_lens_turns_stay_byte_for_byte_clean_on_the_action_graph():
    assert LENS_MARK not in action_system(
        {"user_id": "owner-1", "display_name": "Chris", "voice": True}
    )
    assert LENS_MARK not in action_system(
        {"user_id": "owner-1", "display_name": "Chris", "voice": True, "surface": "desk"}
    )


def test_lens_overlay_reaches_the_chat_graph_too():
    # Today's lens turns are voice turns and run the action graph — but the
    # surface styling must not depend on which graph the router picked.
    assert LENS_MARK in chat_system(
        {"user_id": "owner-1", "display_name": "Chris", "surface": "lens"}
    )
    assert LENS_MARK not in chat_system({"user_id": "owner-1", "display_name": "Chris"})


def test_is_lens_surface_single_source_of_truth():
    assert is_lens_surface({"surface": "lens"})
    assert not is_lens_surface({"surface": "desk"})
    assert not is_lens_surface({})
    assert not is_lens_surface(None)
