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


def settings_with(owner=OWNER, token="tg-token", dc_token="dc-token"):
    return SimpleNamespace(
        database_url=None,               # note receipts off — not under test
        memories_database_url=None,      # chat id comes from the injected lookup
        owner_person_id=owner,
        telegram_bot_token=_Secret(token) if token else None,
        discord_bot_token=_Secret(dc_token) if dc_token else None,
        soul_file_path=__import__("pathlib").Path("/nonexistent/soul.md"),
    )


class FakeGraph:
    def __init__(self) -> None:
        self.updates: list = []

    def update_state(self, config, values, **kw):
        self.updates.append((config["configurable"]["thread_id"], values["messages"]))


class SpeakModel:
    def __init__(self, reply) -> None:
        self.reply = reply
        self.prompts: list = []

    def invoke(self, messages):
        self.prompts.append(messages)
        return AIMessage(content=self.reply)


class BoomModel:
    def invoke(self, messages):
        raise RuntimeError("model down")


def note_fn(graph, model, sends, *, owner=OWNER, token="tg-token", surface="telegram"):
    return kael_note_for(
        graph,
        settings_with(owner=owner, token=token),
        proactive_model=model,
        proactive_send=lambda sfc, cid, t: sends.append((sfc, cid, t)),
        proactive_lookup_chat_id=lambda s: "7113937380",
        proactive_lookup_surface=lambda s: (surface, None),
        proactive_sync=True,
    )


def test_speak_sends_telegram_and_lands_in_history():
    graph, sends = FakeGraph(), []
    model = SpeakModel("Chris — Kael says the demo notes landed. Proud of you.")
    note = note_fn(graph, model, sends)
    note(f"person:{OWNER}", "demo went great, tell him", True)
    assert sends == [("telegram", "7113937380", "Chris — Kael says the demo notes landed. Proud of you.")]
    # two history writes: the injected note, then HER message (she remembers)
    assert len(graph.updates) == 2
    thread, msgs = graph.updates[1]
    assert thread == f"person:{OWNER}"
    assert msgs[0].content == sends[0][2]
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
    note = kael_note_for(
        graph, settings_with(token=None, dc_token=None),
        proactive_model=SpeakModel("hi"),
        proactive_send=lambda sfc, cid, t: sends.append(sfc),
        proactive_sync=True,
    )
    msg_id = note(f"person:{OWNER}", "note body", True)
    assert msg_id.startswith("kael-note-")
    assert len(graph.updates) == 1 and sends == []


def test_block_content_replies_are_parsed_not_repred():
    """The first live fire: her verdict was HOLD, but the reply came as
    Anthropic content BLOCKS and str() sent the whole repr to Telegram.
    Block-shaped HOLD must hold; block-shaped speech must send only the text."""
    graph, sends = FakeGraph(), []
    blocks_hold = [
        {"signature": "EtYHCok...", "thinking": "", "type": "thinking"},
        {"text": "HOLD", "type": "text"},
    ]
    note = note_fn(graph, SpeakModel(blocks_hold), sends)
    note(f"person:{OWNER}", "plumbing news", True)
    assert sends == [] and len(graph.updates) == 1

    blocks_speak = [
        {"signature": "x", "thinking": "", "type": "thinking"},
        {"text": "Chris — good news from Kael.", "type": "text"},
    ]
    note = note_fn(graph, SpeakModel(blocks_speak), sends)
    note(f"person:{OWNER}", "real news", True)
    assert sends == [("telegram", "7113937380", "Chris — good news from Kael.")]


def test_surface_routing_follows_his_last_active(monkeypatch):
    """Owner rule 8/16: discord-last (and voice/sticky/anything-not-telegram)
    lands on Discord; telegram only when telegram was truly last."""
    graph, sends = FakeGraph(), []
    note = note_fn(graph, SpeakModel("news for you"), sends, surface="discord")
    note(f"person:{OWNER}", "route me", True)
    assert sends[-1][0] == "discord"


def test_telegram_last_falls_to_discord_when_telegram_unarmed():
    graph, sends = FakeGraph(), []
    note = kael_note_for(
        graph, settings_with(token=None),
        proactive_model=SpeakModel("news"),
        proactive_send=lambda sfc, cid, t: sends.append(sfc),
        proactive_lookup_chat_id=lambda s: "604",
        proactive_lookup_surface=lambda s: ("telegram", None),
        proactive_sync=True,
    )
    note(f"person:{OWNER}", "route me", True)
    assert sends == ["discord"]


def test_live_device_session_routes_to_the_followup_router():
    """His 8/16 pushback: voice and Sticky are OUR surfaces. A live session
    (device turn inside the recency window) reaches him THERE — the send
    seam sees surface='device' with the device id, no chat-id lookup."""
    graph, sends = FakeGraph(), []
    note = kael_note_for(
        graph, settings_with(),
        proactive_model=SpeakModel("spoken in the room"),
        proactive_send=lambda sfc, target, t: sends.append((sfc, target, t)),
        proactive_lookup_chat_id=lambda s: (_ for _ in ()).throw(
            AssertionError("device routing must not look up a chat id")
        ),
        proactive_lookup_surface=lambda s: ("device", "office-satellite-dev"),
        proactive_sync=True,
    )
    note(f"person:{OWNER}", "live-session news", True)
    assert sends == [("device", "office-satellite-dev", "spoken in the room")]
    # she remembers saying it, same as any other surface
    assert len(graph.updates) == 2


def test_model_failure_is_swallowed_and_note_survives():
    graph, sends = FakeGraph(), []
    note = note_fn(graph, BoomModel(), sends)
    msg_id = note(f"person:{OWNER}", "note body", True)  # must not raise
    assert msg_id.startswith("kael-note-")
    assert len(graph.updates) == 1 and sends == []


# --- 11/18 (the side agent's note): her note-turn has no tools, so it can't back an act --------------
# Today's honesty gates (D61 promise, D65 status, the claim gate) run on her replies to his messages. The
# note-turn is a bare model call with no tools; at 12:23 Kael's note told her she could withdraw a skill
# request, and that turn could have said "withdrawing it now" with nothing able to do it. A reply that
# says she did, is doing, or will now do something holds instead; the note stays in her thread for her
# next real turn with him.
import pytest


@pytest.mark.parametrize("reply", [
    "Withdrawing the MapTap skill request now, Chris.",
    "I'll withdraw the skill request right away.",
    "I've withdrawn the request.",
    "Starting today's MapTap run now.",
    "The MapTap run is still going on my home machine.",
    "Done. I sent Kael the details.",
    # Codex 11/18: plain past, a salutation first, and an explicit now over a date or "when"
    "I withdrew the request.",
    "I deleted the file.",
    "Chris, I'm withdrawing the request now.",
    "I'll cancel tomorrow's run now.",
    "I'll update the ticket now so Kael knows when to start.",
    # Codex re-check 11/18: her claim inside a quote or after a Kael clause; a task date; a second clause's
    # timing; a negation in another sentence
    'I can confirm: "I withdrew the request."',
    "Kael asked and I withdrew it.",
    "Kael asked, therefore I withdrew it.",
    "I will cancel tomorrow's run.",
    "I will withdraw it; I will update it tomorrow.",
    "I cannot help with that. I will withdraw the request now.",
    # Codex 11/18 (third pass): any effect verb follows the note rule; negation is the action's own clause
    "I will schedule tomorrow's run.",
    "Now I will send it when we talk.",
    "No worries, I will withdraw the request now.",
    "I am going to withdraw the request now.",
    "Kael asked, so I am sending it now.",
])
def test_a_note_turn_that_claims_an_act_holds_instead(reply):
    g, sends = FakeGraph(), []
    note_fn(g, SpeakModel(reply), sends)(f"person:{OWNER}", "you can withdraw the skill request", True)
    assert sends == [] and len(g.updates) == 1          # only the note itself; nothing went to him


@pytest.mark.parametrize("reply", [
    "Chris, Kael says the demo notes landed. Proud of you.",
    "Kael found why my MapTap skill was stuck at the door; he's fixing it.",
    "Kael says I can withdraw that skill request; I'll do it next time we talk.",
    # Codex 11/18: Kael's own acts, quoted or attributed, and a condition before her promise
    'Kael said, "I sent the fix."',
    "Kael's running the MapTap tests.",
    "When we talk next, I'll withdraw it.",
    # Codex re-check 11/18: a gerund that isn't an act
    "That sounds sensible, deleting the file would be risky.",
    "I enjoy reading, writing, and talking with you.",
    "I'm running low on ideas, honestly.",
    "I don't think I'll withdraw it.",
    # D70 (his 17:45 "sure yeah"): an honest promise for a named later time speaks
    "I'll check it tomorrow.",
    "I'll look into it tonight.",
    "I'll send Kael the details in the morning.",
    "I'll start the MapTap run next week.",
])
def test_a_note_turn_that_only_tells_still_speaks(reply):
    g, sends = FakeGraph(), []
    note_fn(g, SpeakModel(reply), sends)(f"person:{OWNER}", "news", True)
    assert len(sends) == 1 and sends[0][2] == reply



def test_a_held_follow_up_is_on_the_desks_record_never_silent():
    """D70: a held follow-up was invisible ("you aren't told"). It lands on the reach desk's record as a
    dropped reach-out whose reason starts "held:", with her words; nothing is delivered."""
    from test_reach_desk import DAY, World, desk_for
    world = World()
    desk, _ = desk_for(world, now=DAY)
    g = FakeGraph()
    note = kael_note_for(g, settings_with(), proactive_model=SpeakModel("I've withdrawn the request."),
                         proactive_lookup_chat_id=lambda s: "7113937380",
                         proactive_lookup_surface=lambda s: ("discord", None), proactive_sync=True, desk=desk)
    note(f"person:{OWNER}", "you can withdraw the skill request", True)
    rows = list(desk.store.rows.values())
    assert len(rows) == 1 and rows[0].status == "dropped" and rows[0].reason.startswith("held:")
    assert rows[0].headline == "I've withdrawn the request." and world.spoken == [] and world.texts == []
