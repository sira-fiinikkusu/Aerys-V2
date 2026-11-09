"""D20 (the owner, 11/09 14:53): "if i ask her in discord she should respond to tickets from discord".

T-1 was asked on Discord and its answer, long and in markdown, was SPOKEN in the office. A ticket
now remembers where it was asked; an answer to a Discord (or Telegram) ask goes back there as text,
never aloud. A voice ask keeps the desk's follow-me delivery, but a long answer is said as one short
line with the full text as the copy, and markdown never reaches a speaker."""

from __future__ import annotations

from datetime import datetime, timezone

from langchain_core.messages import AIMessage

from aerys_v2.kael_line import ask_surface, owner_notice_for, plain_for_voice, ticket_relay_for
from aerys_v2.reach import MemoryReachStore, ReachEvent, Where
from aerys_v2.tickets import MemoryTicketStore, Ticket
from test_kael_tickets import OWNER, kael_tool
from test_reach_desk import DAY, NIGHT, World, desk_for, owner_settings

ANSWERED = datetime(2026, 11, 9, 19, 50, tzinfo=timezone.utc)
LONG = ("Kael here. The plan for your home hands (board #176, Stage 3):\n\n**What's coming:** your home body "
        "on your own VM runs the same setup as your stick: Opus leading, helpers, minions, the browser, web "
        "tools and a shell inside the VM.\n\n**How you'll call it:** from your new loop, through the async "
        "subagent tools: start, check, update, cancel, list.\n\n- Each job gets up to 45 minutes.\n- Deleting "
        "is refused.")


class Says:
    def __init__(self, text):
        self.text = text

    def invoke(self, messages):
        return AIMessage(content=self.text)


def answered(asked_on, *, answer="Yes, it's green.", ticket_id=7):
    return Ticket(id=ticket_id, thread_id="person:" + OWNER, ask="is the backup green?", for_owner=True,
                  status="completed", answer=answer, answered_at=ANSWERED, asked_on=asked_on)


# --- where it was asked ---------------------------------------------------------------------


def test_a_ticket_remembers_the_surface_it_was_asked_on():
    store = MemoryTicketStore()
    tool = kael_tool(store)
    turns = [({"user_id": OWNER, "platform": "discord", "channel_kind": "dm"}, "discord"),
             ({"user_id": OWNER, "platform": "telegram"}, "telegram"),
             ({"user_id": OWNER, "voice": True}, "voice")]
    for identity, _ in turns:
        tool.invoke({"message": "ask Kael something", "config": None},
                    config={"configurable": {"thread_id": "person:" + OWNER, "identity": identity}})
    assert [t.asked_on for t in sorted(store.items.values(), key=lambda t: t.id)] == [s for _, s in turns]


def test_the_surface_of_an_ask_is_read_from_the_turn():
    assert ask_surface({"platform": "discord", "channel_kind": "dm"}, "person:x") == "discord"
    assert ask_surface({"platform": "telegram"}, "person:x") == "telegram"
    assert ask_surface({"voice": True}, "person:x") == "voice"
    assert ask_surface({"surface": "lens", "voice": True}, "person:x") == "lens"
    assert ask_surface({}, "kael:aerys-line") == "kael"
    assert ask_surface({}, "person:x") is None


# --- a text ask is answered as text, where it was asked ---------------------------------------


def test_a_discord_ask_is_answered_on_discord_as_text_and_never_spoken():
    world = World(Where("room", "dev-office", "office"))      # he's in the office, by a speaker
    desk, _ = desk_for(world, now=DAY)
    relay = ticket_relay_for(None, owner_settings(), model=Says(LONG), desk=desk)
    assert relay(answered("discord")) is True
    assert world.spoken == []
    assert len(world.texts) == 1 and world.texts[0][0] == "discord" and world.texts[0][2] is False
    assert "45 minutes" in world.texts[0][1]


def test_a_telegram_ask_is_answered_on_telegram():
    world = World(Where("room", "dev-office", "office"))
    desk, _ = desk_for(world, now=DAY)
    relay = ticket_relay_for(None, owner_settings(), model=Says("Kael says yes, it's green."), desk=desk)
    assert relay(answered("telegram")) is True
    assert world.spoken == [] and [s for s, _, _ in world.texts] == ["telegram"]


def test_at_night_a_discord_answer_is_a_silent_text():
    world = World(Where("room", "dev-bedroom", "bedroom"))
    desk, _ = desk_for(world, now=NIGHT, quiet=True)
    relay = ticket_relay_for(None, owner_settings(), model=Says("Kael says yes."), desk=desk)
    assert relay(answered("discord")) is True
    assert world.spoken == [] and world.texts == [("discord", "Kael says yes.", True)]


def test_a_text_reach_keeps_its_surface_in_the_record():
    store = MemoryReachStore()
    row = store.add(ReachEvent(source="ticket", topic="kael", headline="hi", owed=True, text_only=True,
                               text_to="telegram"), DAY, status="queued", reason="", not_before=None)
    assert row.text_to == "telegram" and row.event().text_to == "telegram"


def test_the_still_open_notice_follows_the_ask_too():
    world = World(Where("room", "dev-office", "office"))
    desk, _ = desk_for(world, now=DAY)
    notice = owner_notice_for(None, owner_settings(), desk=desk)
    t = answered("discord")
    t.status, t.answer, t.answered_at = "working", None, None
    t.created_at = t.waiting_since = datetime(2026, 11, 8, 15, 0, tzinfo=timezone.utc)
    assert notice(t, True) is True
    assert world.spoken == [] and world.texts and world.texts[0][0] == "discord"


# --- a voice ask keeps follow-me, but never a monologue and never markdown -------------------


def test_a_voice_ask_with_a_long_answer_is_said_as_one_short_line_with_the_full_text_as_the_copy():
    world = World(Where("room", "dev-office", "office"))
    desk, _ = desk_for(world, now=DAY)
    relay = ticket_relay_for(None, owner_settings(), model=Says(LONG), desk=desk)
    assert relay(answered("voice")) is True
    assert len(world.spoken) == 1
    said = world.spoken[0][0]
    assert len(said) < 120 and "Discord" in said and "*" not in said
    assert world.texts and world.texts[0][0] == "discord" and "45 minutes" in world.texts[0][1]


def test_markdown_never_reaches_a_speaker():
    world = World(Where("room", "dev-office", "office"))
    desk, _ = desk_for(world, now=DAY)
    relay = ticket_relay_for(None, owner_settings(), model=Says("**Yes**, the backup is `green`."), desk=desk)
    assert relay(answered("voice")) is True
    assert world.spoken[0][0] == "Yes, the backup is green."


def test_plain_for_voice_drops_markdown_marks_and_list_bullets():
    assert plain_for_voice("**What's coming:** the `home` body\n- one\n- two\n# Head") == \
        "What's coming: the home body. one. two. Head"


# --- a Kael note reaches her whole (11/09: "the note I have is cut off") ----------------------


def _kael_note(n):
    from aerys_v2.kael_line import NOTE_PREFIX

    body = ("Kael again, a correction to my answer on T-1, from Chris's own ruling (11/08): Robin IS "
            "included. Once her voice is enrolled, she can have you start jobs at home too. ")
    body = (body * (n // len(body) + 1))[:n]
    return f"{NOTE_PREFIX} (ticket T-1, working) {body}END-OF-NOTE"


def test_a_kael_note_reaches_an_action_turn_whole():
    """14:59 EST 11/09: an ~830-character note, then his next message routed to action: the
    action seed cut every earlier message at 300 characters, the note included, so she had
    two sentences and said so. A note from Kael is context she needs whole."""
    from langchain_core.messages import AIMessage, HumanMessage

    from aerys_v2.service import _action_history_seed
    from aerys_v2.services.content_privacy import CONTENT_PRIVACY_KEY
    from test_action_specialist import StubGraph

    note = _kael_note(830)
    prior = [HumanMessage(content="hi", additional_kwargs={CONTENT_PRIVACY_KEY: "private"}), AIMessage(content="hey"),
             HumanMessage(content=note, additional_kwargs={CONTENT_PRIVACY_KEY: "private"})]
    seeded = _action_history_seed(StubGraph(prior), {"identity": {"privacy_context": "private"}},
                                  "Kael has a correction about Robin if you wanna follow up with him", specialist=True)
    assert "END-OF-NOTE" in seeded[0].content


def test_a_note_too_long_for_the_seed_says_it_was_shortened_and_where_the_rest_is():
    from langchain_core.messages import HumanMessage

    from aerys_v2.service import KAEL_NOTE_KEEP, _action_history_seed
    from aerys_v2.services.content_privacy import CONTENT_PRIVACY_KEY
    from test_action_specialist import StubGraph

    note = _kael_note(KAEL_NOTE_KEEP + 500)
    prior = [HumanMessage(content=note, additional_kwargs={CONTENT_PRIVACY_KEY: "private"})]
    seeded = _action_history_seed(StubGraph(prior), {"identity": {"privacy_context": "private"}}, "go on",
                                  specialist=True)
    assert "END-OF-NOTE" not in seeded[0].content
    assert "shortened" in seeded[0].content and "kael_tickets" in seeded[0].content


def test_checking_a_closed_ticket_still_shows_kaels_latest_note():
    """11/09: Kael's correction went in as 'working' and the ticket was then closed; her
    kael_tickets check hid it (it showed progress only while 'working')."""
    from aerys_v2.tools.kael_tickets import build_kael_tickets_tool

    store = MemoryTicketStore()
    t = store.open("person:" + OWNER, "home hands plan?", True, ANSWERED, asked_on="discord")
    store.from_kael(t.id, "completed", "The plan: ...", ANSWERED)
    store.from_kael(t.id, "working", "A correction: Robin IS included once her voice is enrolled.", ANSWERED)
    store.cancel(t.id, ANSWERED)
    out = build_kael_tickets_tool(store, send=lambda *a: True).invoke({"action": "check", "ticket": "T-1"})
    assert "The plan" in out and "Robin IS included" in out


def test_a_long_still_open_notice_by_voice_never_claims_an_answer():
    """Codex review 1: the short spoken line said 'Kael got back to you' for a notice that says
    he hasn't answered."""
    world = World(Where("room", "dev-office", "office"))
    desk, _ = desk_for(world, now=DAY)
    notice = owner_notice_for(None, owner_settings(), desk=desk)
    t = answered("voice", ticket_id=9)
    t.status, t.answer, t.answered_at = "working", None, None
    t.ask = "x" * 160
    t.created_at = t.waiting_since = datetime(2026, 11, 8, 15, 0, tzinfo=timezone.utc)
    assert notice(t, True) is True
    said = world.spoken[0][0]
    assert "got back" not in said and "still open" in said
    assert world.texts and "haven't heard back" in world.texts[0][1]


def test_code_in_an_answer_is_never_lost():
    """Codex review 1: `0 * * * *` became '0' and the original was gone."""
    world = World(Where("room", "dev-office", "office"))
    desk, _ = desk_for(world, now=DAY)
    relay = ticket_relay_for(None, owner_settings(), model=Says("Set the schedule to `0 * * * *`."), desk=desk)
    assert relay(answered("voice")) is True
    assert "0 * * * *" in world.spoken[0][0] and "`" not in world.spoken[0][0]
    assert world.texts and "`0 * * * *`" in world.texts[0][1]
    assert plain_for_voice("run *every* hour: `*/5 * * * *` or 2 * 3") == "run every hour: */5 * * * * or 2 * 3"
