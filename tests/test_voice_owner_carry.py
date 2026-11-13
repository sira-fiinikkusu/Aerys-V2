"""D43 (2026-11-13): a short clip right after he spoke on the same speaker counts as his by default.

The week before: 3 of 9 clips answered as a guest came 16-30 s after him on the same satellite ("Hold off
on the diagram.", "Stop.", "Thank you."), and all three were him. A clip the recognizer calls unknown, on a
device that heard him within voice_owner_carry_s, is his turn by default only (the unsure band: answered as
his, memory never takes it, the hands gate treats it as an unsure voice). A named voice (Megan's print) is
never carried onto him, and a carried clip never extends the window. Placeholder ids and text only."""

from __future__ import annotations

from fastapi.testclient import TestClient

from aerys_v2.state import VOICE_OWNER_ASSUMED_KEY
from aerys_v2.transports.http_api import build_app

OWNER = "11111111-1111-1111-1111-111111111111"
ROBIN = "22222222-2222-2222-2222-222222222222"


def settings(**kw):
    from aerys_v2.config import Settings
    return Settings(_env_file=None, anthropic_api_key="x", owner_person_id=OWNER,
                    voice_speaker_persons=f"chris={OWNER},robin={ROBIN}", **kw)


class Door:
    def __init__(self, **kw):
        self.now, self.seen = {"t": 1000.0}, []

        def ask(text, identity, thread_id):
            self.seen.append((dict(identity), thread_id))
            return "ok"

        self.c = TestClient(build_app(ask, "sekrit", owner_person_id=OWNER, settings=settings(**kw),
                                      guest_clock=lambda: self.now["t"]))

    def say(self, speaker, device="office", after=0):
        self.now["t"] += after
        body = {"text": "hi", "thread_id": "voice:beta", "voice": True, "display_name": "Chris (Voice)"}
        if device is not None:
            body["device_id"] = device
        if speaker is not None:
            body["speaker"] = speaker
        self.c.post("/ask", json=body, headers={"Authorization": "Bearer sekrit"})
        return self.seen[-1]


HIM = {"id": "chris", "confidence": 0.8}
UNSURE = {"id": "unknown", "confidence": 0.31}
STRAY = {"id": "unknown", "confidence": 0.1}


def his_by_default(turn):
    identity, thread = turn
    return (thread == f"person:{OWNER}" and identity.get(VOICE_OWNER_ASSUMED_KEY) is True
            and identity["speaker"].get("carried") is True and identity["speaker"]["demoted"] is False)


def guest(turn):
    return turn[1].startswith("person:voice-guest:")


def test_a_short_clip_right_after_him_is_his_by_default():
    d = Door()
    d.say(HIM)
    assert his_by_default(d.say(STRAY, after=30))            # "Thank you." 30 s after his ask


def test_past_the_window_or_on_another_speaker_it_is_a_guest():
    d = Door()
    d.say(HIM)
    assert guest(d.say(STRAY, device="kitchen", after=10))
    assert guest(d.say(STRAY, after=61))


def test_an_unsure_turn_of_his_opens_the_window_too():
    d = Door()
    assert not guest(d.say(UNSURE))                           # the unsure band: his by default
    assert his_by_default(d.say(STRAY, after=16))             # "Stop. Harris." then "Hold off on the diagram."


def test_a_carried_clip_never_extends_the_window():
    d = Door()
    d.say(HIM)
    assert his_by_default(d.say(STRAY, after=50))
    assert guest(d.say(STRAY, after=50))                      # 100 s after him: the TV doesn't chain on


def test_a_named_voice_is_never_carried_onto_him():
    d = Door()
    d.say(HIM)
    identity, thread = d.say({"id": "robin", "confidence": 0.7}, after=5)
    assert thread == f"person:{ROBIN}" and not identity.get(VOICE_OWNER_ASSUMED_KEY)
    d.say(HIM, after=1)
    assert guest(d.say({"id": "dimitri", "confidence": 0.9}, after=5))   # an unlisted name stays a guest


def test_no_device_untagged_turns_and_the_switch():
    d = Door()
    d.say(HIM, device=None)
    assert guest(d.say(STRAY, device=None, after=5))          # no device: nothing to carry on
    d = Door()
    d.say(None)                                               # untagged: no verdict heard him
    assert guest(d.say(STRAY, after=5))
    d = Door(voice_owner_carry_s=0)
    d.say(HIM)
    assert guest(d.say(STRAY, after=5))                       # 0 turns it off


def test_a_guest_never_opens_the_window():
    d = Door()
    d.say(STRAY)
    assert guest(d.say(STRAY, after=5))


def test_the_window_runs_from_when_she_finished_answering_him():
    # her answer to him takes 40 s; his "thank you" lands 30 s after she finished (70 s after he asked)
    d = Door()

    def ask_slow(text, identity, thread_id):
        d.seen.append((dict(identity), thread_id))
        if (identity.get("speaker") or {}).get("id") == "chris":
            d.now["t"] += 40
        return "ok"

    d.c = TestClient(build_app(ask_slow, "sekrit", owner_person_id=OWNER, settings=settings(),
                               guest_clock=lambda: d.now["t"]))
    d.say(HIM)
    assert his_by_default(d.say(STRAY, after=30))


def test_her_hands_still_treat_a_carried_clip_as_an_unsure_voice():
    from aerys_v2.hands_gate import VOICE_UNSURE, asker
    from aerys_v2.outside_guard import origin_sources
    d = Door()
    d.say(HIM)
    identity, thread = d.say(STRAY, after=10)
    assert asker(identity, thread, OWNER) == VOICE_UNSURE
    assert "outside speaker" in origin_sources(identity, thread, OWNER)    # a send waits for his tap


def test_words_that_were_his_by_default_never_become_his_on_a_later_turn():
    # Codex 11/13 (D43 review): a carried "The pool guy comes on Tuesdays", then his text "remember that":
    # the remember window took the previous human turn as his words and saved it owner-trusted (D2).
    from langchain_core.messages import AIMessage, HumanMessage
    from aerys_v2.service import PRIVATE, _human_turn, _owner_assumed, _remember_window_for
    carried_config = {"configurable": {"identity": {"user_id": OWNER, VOICE_OWNER_ASSUMED_KEY: True}}}
    assert _owner_assumed(carried_config) and not _owner_assumed({"configurable": {"identity": {"user_id": OWNER}}})
    said = _human_turn("The pool guy comes on Tuesdays", PRIVATE, "m1", owner_assumed=_owner_assumed(carried_config))
    assert said.additional_kwargs[VOICE_OWNER_ASSUMED_KEY] is True
    his = {"configurable": {"identity": {"user_id": OWNER, "privacy_context": "private", "platform": "discord"}}}
    seed = [said, AIMessage(content="Noted."), HumanMessage(content="remember that")]
    assert _remember_window_for(seed, "remember that", his) == "remember that"
    plain = [HumanMessage(content="The pool guy comes on Tuesdays"), AIMessage(content="Noted."),
             HumanMessage(content="remember that")]
    assert "pool guy" in _remember_window_for(plain, "remember that", his)      # his own words still count


def test_the_carry_cache_is_bounded():
    from aerys_v2.transports.http_api import OWNER_HEARD_MAX
    d = Door()
    for i in range(OWNER_HEARD_MAX + 40):
        d.say(HIM, device=f"dev-{i}")
    assert guest(d.say(STRAY, device="dev-0", after=1))               # the oldest was let go
    assert his_by_default(d.say(STRAY, device=f"dev-{OWNER_HEARD_MAX + 39}", after=1))


def test_every_writer_of_his_turn_keeps_the_by_default_stamp():
    # Like the sealed room: a writer that rebuilt the message by id without it (the privacy retag) would
    # quietly make his-by-default words his own again.
    import inspect
    import re
    import aerys_v2.service as service
    source = inspect.getsource(service)
    starts = [m.start() for m in re.finditer(r"_human_turn\(text,", source)]
    assert len(starts) >= 2
    for a, b in zip(starts, starts[1:] + [len(source)]):
        call = source[a:min(b, a + 400)]
        assert "owner_assumed=_owner_assumed(" in call, call
