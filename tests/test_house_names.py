"""Her house's names live in the private .env, never in this public repo (house_names.py; the owner,
2026-11-09: "i would appreciate we dont air out anyone but me"). Generic names only here."""
import json
import logging
import os
import subprocess
import sys

from aerys_v2.factory import _shared_surface_note
from aerys_v2.house_names import ENV, HouseNames, load
from aerys_v2.reflex import unaddressed_instructions, voice_background
from aerys_v2.router import named_address_example

FULL = HouseNames(wife="Robin", friends="Alex, Jordan", friend="Alex", colleague="Riley", aside="Morgan")
NONE = HouseNames()


def test_the_names_come_from_one_json_object():
    env = {ENV: json.dumps({"wife": " Robin ", "friends": "Alex, Jordan", "friend": "Alex",
                            "colleague": "Riley", "aside": "Morgan"})}
    assert load(env) == HouseNames(wife="Robin", friends="Alex, Jordan", friend="Alex", colleague="Riley",
                                   aside="Morgan")


def test_missing_or_broken_names_never_stop_her_and_say_so(caplog):
    caplog.set_level(logging.WARNING)
    for env in ({}, {ENV: ""}, {ENV: "{not json"}, {ENV: '["Robin"]'}):
        caplog.clear()
        assert load(env) == NONE
        assert caplog.records and ENV in caplog.text


def test_a_partial_object_keeps_what_it_has_and_names_what_it_lacks(caplog):
    caplog.set_level(logging.WARNING)
    names = load({ENV: json.dumps({"wife": "Robin", "friends": 3, "aside": ""})})
    assert names == HouseNames(wife="Robin")
    assert "friends" in caplog.text and "aside" in caplog.text


def test_with_names_her_voice_gate_says_them():
    text = voice_background(FULL)
    assert ("with a few words) and his wife Robin. Friends/colleagues who come up by name: Alex, Jordan. "
            "Satellites are microphones") in text
    ask = unaddressed_instructions(FULL)
    assert ("Examples of NOT addressed: 'so I told Robin we would probably leave around nine tomorrow' | "
            "'and then he just left it in the driveway all weekend' | 'He interrupted Morgan. Let me ask him about "
            "his trip to Italy' | 'Riley, it's a, uh, a way to track outages'. Examples of addressed") in ask


def test_without_names_nothing_is_invented():
    text = voice_background(NONE)
    assert "and his wife. Satellites are microphones" in text and "Friends/colleagues" not in text
    ask = unaddressed_instructions(NONE)
    assert ("Examples of NOT addressed: 'so I told her we would probably leave around nine tomorrow' | "
            "'and then he just left it in the driveway all weekend'. Examples of addressed") in ask
    assert "interrupted" not in ask and "track outages" not in ask
    assert named_address_example(NONE) == '("<name>, can you grab my towel?", "hang on <name>, I\'ll be right there")'
    assert named_address_example(HouseNames(wife="Robin")) == named_address_example(NONE)   # one name isn't the pair
    assert named_address_example(FULL) == '("Robin, can you grab my towel?", "hang on Alex, I\'ll be right there")'


def test_the_shared_sticky_note_names_the_wife_from_the_private_names():
    note = _shared_surface_note({"device_id": "sticky-1"}, {"sticky-1": ""})
    assert "it may be Robin, or a guest." in note        # the suite's generic names (conftest)


def test_she_starts_without_the_names_and_warns_once():
    env = {k: v for k, v in os.environ.items() if k != ENV}
    env["PYTHONPATH"] = os.pathsep.join(p for p in sys.path if p)
    run = subprocess.run([sys.executable, "-c",
                          "import aerys_v2.reflex as r, aerys_v2.router, aerys_v2.factory, aerys_v2.services.presence\n"
                          "assert 'and his wife. Satellites' in r.VOICE_BACKGROUND\n"
                          "assert 'Friends/colleagues' not in r.VOICE_BACKGROUND\n"
                          "print('started')"],
                         env=env, capture_output=True, text=True, timeout=120)
    assert run.returncode == 0, run.stderr
    assert "started" in run.stdout
    assert run.stderr.count(f"{ENV} isn't set") == 1
