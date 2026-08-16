"""Gap #45 — the proactive turn when a family-visible Kael note lands.

The owner's spec: after Kael answers on the line, "i'd like for her to adhoc
respond to me" — her voice, her call. These tests pin the gate (owner thread
+ family_visible only), both verdicts (speak → send + history append; HOLD →
silence), fail-open on a broken model, and that the note's own delivery is
never hostage to the proactive step.
"""

from types import SimpleNamespace

from langchain_core.messages import AIMessage

from aerys_v2.kael_line import PROACTIVE_DECIDE_PROMPT, kael_note_for

OWNER = "6e6bcbed-0000-0000-0000-000000000000"


class _Secret:
    def __init__(self, value: str) -> None:
        self._value = value

    def get_secret_value(self) -> str:
        return self._value


def settings_with(owner=OWNER, token="tg-token"):
    return SimpleNamespace(
        database_url=None,               # note receipts off — not under test
        memories_database_url=None,      # chat id comes from the injected lookup
        owner_person_id=owner,
        telegram_bot_token=_Secret(token) if token else None,
        soul_file_path=__import__("pathlib").Path("/nonexistent/soul.md"),
    )


class FakeGraph:
    def __init__(self) -> None:
        self.updates: list = []

    def update_state(self, config, values, **kw):
        self.updates.append((config["configurable"]["thread_id"], values["messages"]))


class SpeakModel:
    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.prompts: list = []

    def invoke(self, messages):
        self.prompts.append(messages)
        return AIMessage(content=self.reply)


class BoomModel:
    def invoke(self, messages):
        raise RuntimeError("model down")


def note_fn(graph, model, sends, *, owner=OWNER, token="tg-token"):
    return kael_note_for(
        graph,
        settings_with(owner=owner, token=token),
        proactive_model=model,
        proactive_send=lambda cid, t: sends.append((cid, t)),
        proactive_lookup_chat_id=lambda s: "7113937380",
        proactive_sync=True,
    )


def test_speak_sends_telegram_and_lands_in_history():
    graph, sends = FakeGraph(), []
    model = SpeakModel("Chris — Kael says the demo notes landed. Proud of you.")
    note = note_fn(graph, model, sends)
    note(f"person:{OWNER}", "demo went great, tell him", True)
    assert sends == [("7113937380", "Chris — Kael says the demo notes landed. Proud of you.")]
    # two history writes: the injected note, then HER message (she remembers)
    assert len(graph.updates) == 2
    thread, msgs = graph.updates[1]
    assert thread == f"person:{OWNER}"
    assert msgs[0].content == sends[0][1]
    # the deciding prompt rode her soul, with the tone guard in it
    system = model.prompts[0][0].content
    assert PROACTIVE_DECIDE_PROMPT in system and "help, never hover" in system


def test_hold_verdict_sends_nothing_and_leaves_history_clean():
    graph, sends = FakeGraph(), []
    for verdict in ("HOLD", "hold", " HOLD. "):
        g = FakeGraph()
        note = note_fn(g, SpeakModel(verdict), sends)
        note(f"person:{OWNER}", "minor deploy chatter", True)
        assert len(g.updates) == 1  # only the injected note itself
    assert sends == []


def test_private_notes_and_foreign_threads_never_consult_the_model():
    graph, sends = FakeGraph(), []
    model = SpeakModel("should never be called")
    note = note_fn(graph, model, sends)
    note(f"person:{OWNER}", "smoke test", False)          # not family-visible
    note("kael:checkin", "family news elsewhere", True)   # not his thread
    assert model.prompts == [] and sends == []


def test_unarmed_without_token_still_delivers_the_note():
    graph, sends = FakeGraph(), []
    note = note_fn(graph, SpeakModel("hi"), sends, token=None)
    msg_id = note(f"person:{OWNER}", "note body", True)
    assert msg_id.startswith("kael-note-")
    assert len(graph.updates) == 1 and sends == []


def test_model_failure_is_swallowed_and_note_survives():
    graph, sends = FakeGraph(), []
    note = note_fn(graph, BoomModel(), sends)
    msg_id = note(f"person:{OWNER}", "note body", True)  # must not raise
    assert msg_id.startswith("kael-note-")
    assert len(graph.updates) == 1 and sends == []
