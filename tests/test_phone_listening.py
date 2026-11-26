"""11/26: asked "what am i listening to right now?" she checked only the office speaker. Her music
instructions now say to read the phone's media session sensor too (her general Home Assistant read;
no special tool, and the D47 rail on his phone readings is unchanged)."""

from aerys_v2.factory import MUSIC_OVERLAY


def test_the_music_instructions_point_at_the_phones_media_session():
    assert "media session sensor" in MUSIC_OVERLAY
    assert "search_entities 'media session'" in MUSIC_OVERLAY and "home_control get_state" in MUSIC_OVERLAY
    assert "If nothing is playing anywhere, say that." in MUSIC_OVERLAY


def test_no_private_entity_id_is_written_into_the_public_prompt():
    assert "sm_f" not in MUSIC_OVERLAY and "sensor." not in MUSIC_OVERLAY


def test_the_older_rules_are_still_there():
    assert "NEVER ask which speaker" in MUSIC_OVERLAY
    assert "Never claim music is playing or changed unless the tool's reply said so." in MUSIC_OVERLAY
