"""Gap #47 — the record-claim gate: 'Logged:' with no tool behind it.

The incident (v2_turns 8/14 23:56-57, voice): the owner reported a display
bug, she replied 'Got it — let me log that as a gap' then 'Logged: sticky
display truncates long replies…' with tool_calls=[] on both turns. Nothing
was written; the board stayed empty until he asked a day later. These tests
mirror the audio-gate suite (FIX 3) — same machinery, new fabrication family.
"""

import json

from aerys_v2.factory import build_graph
from aerys_v2.service import (
    RECORD_CLAIM_CORRECTION,
    RECORD_CLAIM_MARKER,
    ask,
    record_action_claim,
)
from test_honesty_fixes import CHRIS, Recorder, chat_router, fake_model

# Her 8/14 sentence, verbatim from the turns audit.
THE_INCIDENT = (
    "Logged: sticky display truncates long replies with no scroll — "
    "swipe up/down suggested as the fix. Keeping it on the board."
)


def test_record_claim_detector_positive_table():
    for text in (
        THE_INCIDENT,
        "Logged: panel wifi drops after the rebuild.",
        "I've logged that as a gap for the next loop.",
        "I just filed it as a capability request.",
        "Done — I recorded that as an issue on the board.",
        "All set, I have logged the display bug you mentioned.",
    ):
        assert record_action_claim(text), text


def test_record_claim_detector_negative_table():
    # Intent, login idioms, third parties, reading logs, and plain warmth
    # all pass — narrow beats eager, same tuning as the audio detector.
    for text in (
        "Got it — let me log that as a gap.",
        "I'll log it as soon as I'm on the right path.",
        "I logged into the router UI and checked the leases.",
        "I've logged in to Home Assistant just fine.",
        "Chris logged the issue himself this morning.",
        "The log shows three reconnects overnight.",
        "Noted — I'll keep an eye on it.",
        "You should log that one on the board when you get a chance.",
    ):
        assert not record_action_claim(text), text


def test_record_claim_bounces_once_and_emits_the_honest_retry():
    rec = Recorder()
    honest = (
        "That display bug matters — I can't write to the gap board from this "
        "path, so ask me again in a fresh message and it'll get logged for real."
    )
    graph = build_graph(fake_model(THE_INCIDENT, honest), soul="s")
    out = ask(graph, "long replies get cut off on the sticky", identity=CHRIS,
              thread_id="t-record-bounce", router=chat_router, record_turn=rec)
    assert out == honest
    row = rec.wait(1)[0]
    # raw keeps the original claim — the gate diff shows the save
    assert row["raw_reply"] == THE_INCIDENT
    assert row["emitted_reply"] == honest
    assert not row["degraded"] or RECORD_CLAIM_MARKER not in json.loads(row["degraded"])


def test_record_claim_surviving_the_bounce_is_emitted_with_the_marker():
    rec = Recorder()
    stubborn = "I've logged it as a gap, promise."
    graph = build_graph(fake_model(THE_INCIDENT, stubborn), soul="s")
    out = ask(graph, "long replies get cut off on the sticky", identity=CHRIS,
              thread_id="t-record-mark", router=chat_router, record_turn=rec)
    assert out == stubborn  # visibility, not censorship
    row = rec.wait(1)[0]
    assert RECORD_CLAIM_MARKER in json.loads(row["degraded"])


def test_record_claim_history_surgery_leaves_no_plumbing_behind():
    graph = build_graph(
        fake_model(THE_INCIDENT, "Honest answer, no board access."), soul="s"
    )
    ask(graph, "long replies get cut off on the sticky", identity=CHRIS,
        thread_id="t-record-surgery", router=chat_router)
    msgs = graph.get_state(
        {"configurable": {"thread_id": "t-record-surgery", "identity": CHRIS}}
    ).values["messages"]
    joined = " || ".join(str(getattr(m, "content", "")) for m in msgs)
    assert RECORD_CLAIM_CORRECTION[:40] not in joined  # plumbing never persists
    assert THE_INCIDENT not in joined                  # the fabricated claim is gone
    assert joined.count("Honest answer, no board access.") == 1


def test_record_claim_correction_never_reaches_the_user():
    assert "internal plumbing" in RECORD_CLAIM_CORRECTION
    assert "never mention it" in RECORD_CLAIM_CORRECTION


def test_clean_replies_never_bounce():
    rec = Recorder()
    clean = "That's a real annoyance — tell me more about when it happens?"
    graph = build_graph(fake_model(clean), soul="s")
    out = ask(graph, "the sticky cuts off replies", identity=CHRIS,
              thread_id="t-record-clean", router=chat_router, record_turn=rec)
    assert out == clean
    row = rec.wait(1)[0]
    assert not row["degraded"] or RECORD_CLAIM_MARKER not in json.loads(row["degraded"])
