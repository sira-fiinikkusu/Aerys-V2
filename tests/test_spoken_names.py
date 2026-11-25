"""11/25 (Chris: "She wasn't understanding Eris is Aerys when im speaking"): on the glasses Even's speech
recognition wrote "ERIS satellite" for the "Aerys satellite"; spoken turns read those spellings as Aerys."""
from aerys_v2.spoken_names import fix_spoken_names
from test_d79_text_one_loop import Action, dm, route, run, tool_then


def test_her_name_as_speech_recognition_writes_it():
    assert fix_spoken_names("The ERIS satellite is your bedroom.") == "The Aerys satellite is your bedroom."
    assert fix_spoken_names("Hey Eris, what's Eris's favourite?") == "Hey Aerys, what's Aerys's favourite?"
    for kept in ("Iris is coming over", "I'm an Aries", "Erisa called", "Paris"):
        assert fix_spoken_names(kept) == kept


def test_a_glasses_turn_reaches_her_as_aerys_and_a_typed_one_stays_as_written():
    def asked(identity):
        action = Action(tool_then("music", "playing"))
        run("play it on the ERIS satellite", identity=identity, action=action, router=route("action"))
        return action.calls[0][0][-1].content
    assert asked(dm(surface="lens")).endswith("play it on the Aerys satellite")
    assert asked(dm()).endswith("play it on the ERIS satellite")
