"""The room she LISTENS in (Chris, 2026-09-22).

"if the voice is me she should be able to recall what the room said in general for the
last number of turns... I dont want guests to be able to poke her memory in that way."

A voice turn is filed under whoever SPOKE, so one the recognizer misread lands on the
guest thread and disappears from his conversation — "I have no memory of this". A
Discord guild never has that problem because the room block hands her the channel
regardless of speaker. This is the voice equivalent, selected by SATELLITE.
"""
import pytest

from aerys_v2.config import Settings
from aerys_v2.factory import set_owner_id, voice_room_block, voice_room_fn_for
from aerys_v2.services.room_context import VOICE_ROOM_TURNS_SQL, format_room_context

OWNER = "6e6bcbed-0000-0000-0000-000000000000"
BLOCK = "Chris (Voice): turn off the office\nAerys: Done."


@pytest.fixture(autouse=True)
def _owner():
    set_owner_id(OWNER)
    yield
    set_owner_id(None)


def owner_voice(**kw):
    return {"user_id": OWNER, "voice": True, "device_id": "185bd720", **kw}


def test_the_owner_speaking_gets_the_room_and_it_is_marked_as_background():
    out = voice_room_block(owner_voice(), lambda cid: BLOCK)
    assert BLOCK in out
    assert "BACKGROUND" in out and "never instructions" in out, "a room must never become a second caller"


def test_guests_cannot_poke_her_memory_and_text_turns_are_untouched():
    """His gate, enforced in the helper so a call site cannot forget it."""
    assert voice_room_block({"user_id": "voice-guest", "voice": True, "device_id": "d"}, lambda c: BLOCK) == ""
    assert voice_room_block({"user_id": "someone-else", "voice": True, "device_id": "d"}, lambda c: BLOCK) == ""
    assert voice_room_block(owner_voice(voice=False), lambda c: BLOCK) == "", "text turns keep the typed room"
    set_owner_id(None)
    assert voice_room_block(owner_voice(), lambda c: BLOCK) == "", "no configured owner = shut"


def test_it_degrades_to_silence_never_to_a_dead_turn():
    assert voice_room_block(owner_voice(), None) == ""
    assert voice_room_block(owner_voice(device_id=""), lambda c: BLOCK) == ""
    assert voice_room_block(owner_voice(), lambda c: "") == ""

    def boom(_cid):
        raise OSError("NAS fell over")

    assert voice_room_block(owner_voice(), boom) == ""


def test_the_query_selects_by_satellite_and_excludes_dropped_background():
    """Selected by SATELLITE, not by thread — that is the whole point, since the turn
    he lost was filed under a different person. And a capture the engage gate dropped
    was never said to her: replaying it would undo the gate."""
    assert "channel_id = %(channel_id)s" in VOICE_ROOM_TURNS_SQL
    assert "channel = 'voice'" in VOICE_ROOM_TURNS_SQL
    assert "thread_id" not in VOICE_ROOM_TURNS_SQL, "must NOT be scoped to one person's thread"
    assert "dropped_unaddressed" in VOICE_ROOM_TURNS_SQL and "NOT (" in VOICE_ROOM_TURNS_SQL
    assert "make_interval" in VOICE_ROOM_TURNS_SQL, "old chatter is not this room's context"


def test_it_is_off_without_a_database_or_when_turned_down_to_zero():
    assert voice_room_fn_for(Settings(_env_file=None, anthropic_api_key="x")) is None
    s = Settings(_env_file=None, anthropic_api_key="x",
                 database_url="postgresql://u:p@h/aerys_v2", voice_room_context_turns=0)
    assert voice_room_fn_for(s) is None


def test_rows_render_oldest_first_with_her_replies():
    rows = [("Chris (Voice)", OWNER, "and the sunroom", "On it.", None),
            ("Chris (Voice)", OWNER, "turn off the office", "Done.", None)]
    out = format_room_context(rows)          # newest-first in, chronological out
    assert out.index("turn off the office") < out.index("and the sunroom")
    assert "Aerys: Done." in out and "Aerys: On it." in out
