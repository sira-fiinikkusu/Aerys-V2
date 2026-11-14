"""#198 (11/13 18:07): "What is this about the FAA grounding all these flights?" reached her as "lights"; she
searched "FAA grounding lights" three times and said there was no such story, on the day of a nationwide
ground stop. Both voice overlays now tell her to retry the most likely misheard word before saying it
doesn't exist."""

from aerys_v2.factory import VOICE_ACK_OVERLAY, VOICE_BANTER_OVERLAY


def test_both_voice_overlays_say_to_retry_a_misheard_word():
    for overlay in (VOICE_ACK_OVERLAY, VOICE_BANTER_OVERLAY):
        assert "'lights' for 'flights'" in overlay and "doesn't exist" in overlay
    assert "{ack!r}" in VOICE_ACK_OVERLAY                       # still a format string
    VOICE_ACK_OVERLAY.format(ack="Let me look into that.")
