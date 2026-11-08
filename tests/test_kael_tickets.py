"""Kael tickets (board #176): what she asks Kael comes back, and the owner's asks come back
to him. Offline: the in-memory store, a fake line (httpx.MockTransport), fake relays.
The owner's complaint these pin (2026-11-07): "when i tell her to ask you something, she
never follows up with me. the conversation is 1 way."
"""

from __future__ import annotations

import json
import threading
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import httpx
import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

from aerys_v2.kael_line import (
    TICKET_RELAY_PROMPT,
    in_quiet_hours,
    kael_note_for,
    owner_notice_for,
    owner_reach_for,
    ticket_relay_for,
)
from aerys_v2.tickets import (
    MemoryTicketStore,
    Ticket,
    kael_update_for,
    label,
    line_sender,
    parse_id,
    run_once,
    TicketDoor,
)
from aerys_v2.tools.kael_tickets import build_kael_tickets_tool
from aerys_v2.tools.message_kael import build_message_kael_tool
from aerys_v2.transports.http_api import build_app

OWNER = "6e6bcbed-0000-0000-0000-000000000000"
URL = "http://desk.test:8399/aerys/message"
T0 = datetime(2026, 11, 8, 16, 0, tzinfo=timezone.utc)     # 11:00 Eastern
NAG = timedelta(hours=4)
TELL = timedelta(hours=24)


def owner_cfg(thread="person:" + OWNER, user=OWNER):
    return {"configurable": {"thread_id": thread, "identity": {"user_id": user}}}


def line(status=202, seen=None):
    def handler(req):
        if seen is not None:
            seen.append(json.loads(req.read().decode()))
        if isinstance(status, Exception):
            raise status
        return httpx.Response(status)
    return httpx.Client(transport=httpx.MockTransport(handler))


def kael_tool(store, status=202, seen=None, clock=lambda: T0):
    return build_message_kael_tool(URL, "tok", client=line(status, seen), tickets=store,
                                   owner_person_id=OWNER, clock=clock)


class Recorder:
    def __init__(self, ok=True):
        self.calls, self.ok = [], ok

    def __call__(self, *args):
        self.calls.append(args)
        return self.ok


def tick(store, now, send=None, relay=None, tell=None, relayed=None):
    return run_once(store, now, send=send or Recorder(), nag_after=NAG, tell_owner_after=TELL,
                    relay=relay, tell_owner=tell, relayed=relayed)


# --- ids ---------------------------------------------------------------------------------

@pytest.mark.parametrize("text", ["T-12", "t12", "#12", "ticket 12", " 12 "])
def test_every_way_of_saying_a_ticket_id_parses(text):
    assert parse_id(text) == 12


@pytest.mark.parametrize("text", ["", "T-", "twelve", "12a", None])
def test_nonsense_is_not_an_id(text):
    assert parse_id(text) is None


# --- message_kael opens the ticket --------------------------------------------------------

def test_an_owner_ask_opens_a_ticket_and_the_id_rides_in_the_message():
    store, seen = MemoryTicketStore(), []
    out = kael_tool(store, seen=seen).invoke({"message": "is the backup green?"}, config=owner_cfg())
    t = store.get(1)
    assert t.for_owner and t.thread_id == "person:" + OWNER and t.ask == "is the backup green?"
    assert t.outbox is None and t.delivered_at == T0          # the line took it
    assert seen[0]["message"] == "(ticket T-1) is the backup green?"
    assert seen[0]["ticket_id"] == "T-1" and seen[0]["thread_id"] == "person:" + OWNER
    assert "T-1" in out and "brought to the owner" in out


def test_her_own_ask_from_a_guest_or_background_thread_is_not_the_owners():
    store = MemoryTicketStore()
    out = kael_tool(store).invoke({"message": "my weather tool is down"},
                                  config=owner_cfg(thread="kael:aerys-line", user="aerys"))
    assert store.get(1).for_owner is False
    assert "come back to this conversation" in out and "owner" not in out


def test_a_dead_line_keeps_the_ticket_owed_and_says_so_honestly():
    store = MemoryTicketStore()
    out = kael_tool(store, status=httpx.ConnectError("refused")).invoke(
        {"message": "ping"}, config=owner_cfg())
    t = store.get(1)
    assert t.outbox == "ping" and t.delivered_at is None
    assert out.startswith("NOT DELIVERED YET") and "T-1" in out and "Don't send it again" in out
    assert "owner hears" in out


def test_a_busy_line_is_queued_not_a_retry_order():
    store = MemoryTicketStore()
    out = kael_tool(store, status=429).invoke({"message": "ping"}, config=owner_cfg())
    assert "busy" in out and "T-1" in out and store.get(1).outbox == "ping"


def test_a_ledger_that_cannot_be_reached_never_blocks_the_line():
    class Down:
        def open(self, *a):
            raise RuntimeError("NAS down")

    seen = []
    out = kael_tool(Down(), seen=seen).invoke({"message": "ping"}, config=owner_cfg())
    assert out == "Delivered — Kael has it in his session now."
    assert seen[0]["message"] == "ping" and "ticket_id" not in seen[0]


# --- the watcher ----------------------------------------------------------------------------

def owed(store, **kw):
    t = store.open("person:" + OWNER, kw.pop("ask", "is the backup green?"), kw.pop("for_owner", True), T0)
    for k, v in kw.items():
        setattr(t, k, v)
    return t


def test_an_undelivered_ask_is_resent_after_ten_minutes_not_before():
    store, send = MemoryTicketStore(), Recorder()
    owed(store)
    assert tick(store, T0 + timedelta(minutes=9), send=send)["resent"] == []
    assert tick(store, T0 + timedelta(minutes=10), send=send)["resent"] == ["T-1"]
    assert send.calls[0][1] == "is the backup green?" and store.get(1).outbox is None


def test_a_resend_that_fails_waits_another_ten_minutes():
    store, send = MemoryTicketStore(), Recorder(ok=False)
    owed(store)
    tick(store, T0 + timedelta(minutes=10), send=send)
    tick(store, T0 + timedelta(minutes=15), send=send)
    assert len(send.calls) == 1
    tick(store, T0 + timedelta(minutes=20), send=send)
    assert len(send.calls) == 2


def test_resending_stops_after_three_days():
    store, send = MemoryTicketStore(), Recorder(ok=False)
    owed(store)
    tick(store, T0 + timedelta(hours=72), send=send)
    assert send.calls == []


def test_kael_is_reminded_once_four_hours_after_he_could_see_it():
    store, send = MemoryTicketStore(), Recorder()
    owed(store, outbox=None, delivered_at=T0)
    assert tick(store, T0 + timedelta(hours=3, minutes=59), send=send)["nagged"] == []
    assert tick(store, T0 + NAG, send=send)["nagged"] == ["T-1"]
    assert "still open after 4 h" in send.calls[0][1] and "is the backup green?" in send.calls[0][1]
    assert tick(store, T0 + NAG + timedelta(hours=1), send=send)["nagged"] == []


def test_an_ask_that_sat_behind_a_dead_line_is_news_to_kael_not_overdue():
    store, send = MemoryTicketStore(), Recorder()
    owed(store, last_try_at=T0)
    tick(store, T0 + timedelta(hours=5), send=send)            # delivered now, 5 h late
    assert [c[1] for c in send.calls] == ["is the backup green?"]
    assert tick(store, T0 + timedelta(hours=8), send=send)["nagged"] == []
    assert tick(store, T0 + timedelta(hours=9), send=send)["nagged"] == ["T-1"]


def test_a_ticket_kael_answered_is_never_nagged():
    store, send = MemoryTicketStore(), Recorder()
    owed(store, outbox=None, delivered_at=T0, status="completed")
    assert tick(store, T0 + timedelta(hours=10), send=send)["nagged"] == []


def test_the_owner_hears_once_after_a_day_when_his_ask_is_still_open():
    store, tell = MemoryTicketStore(), Recorder()
    owed(store, outbox=None, delivered_at=T0)
    assert tick(store, T0 + timedelta(hours=23), tell=tell)["told_owner"] == []
    assert tick(store, T0 + TELL, tell=tell)["told_owner"] == ["T-1"]
    assert tell.calls[0][1] is True                           # reached: Kael had it
    assert tick(store, T0 + TELL + timedelta(hours=2), tell=tell)["told_owner"] == []


def test_the_owner_hears_when_it_never_got_through():
    store, tell = MemoryTicketStore(), Recorder()
    owed(store)
    tick(store, T0 + TELL, send=Recorder(ok=False), tell=tell)
    assert tell.calls[0][1] is False


def test_her_own_asks_never_go_to_the_owner():
    store, tell = MemoryTicketStore(), Recorder()
    owed(store, for_owner=False, outbox=None, delivered_at=T0)
    assert tick(store, T0 + TELL * 2, tell=tell)["told_owner"] == [] and tell.calls == []


def test_kaels_answer_to_the_owners_ask_is_relayed_once():
    store, relay = MemoryTicketStore(), Recorder()
    owed(store, outbox=None, delivered_at=T0)
    store.from_kael(1, "completed", "yes, 03:00 run was green", T0 + timedelta(hours=1))
    assert tick(store, T0 + timedelta(hours=1), relay=relay)["relayed"] == ["T-1"]
    assert relay.calls[0][0].answer == "yes, 03:00 run was green"
    assert tick(store, T0 + timedelta(hours=2), relay=relay)["relayed"] == []
    assert len(relay.calls) == 1


def test_a_relay_that_fails_is_retried_next_tick():
    store = MemoryTicketStore()
    owed(store, outbox=None, delivered_at=T0)
    store.from_kael(1, "completed", "yes", T0)
    assert tick(store, T0, relay=Recorder(ok=False))["failed"] == ["T-1"]
    assert tick(store, T0 + timedelta(minutes=1), relay=Recorder())["relayed"] == ["T-1"]


def test_relayed_but_not_recorded_is_never_relayed_twice():
    class FlakyMark(MemoryTicketStore):
        broken = True

        def mark(self, ticket_id, column, now):
            if self.broken:
                raise RuntimeError("NAS hiccup")
            super().mark(ticket_id, column, now)

    store, relay, relayed = FlakyMark(), Recorder(), set()
    owed(store, outbox=None, delivered_at=T0)
    store.from_kael(1, "completed", "yes", T0)
    tick(store, T0, relay=relay, relayed=relayed)
    store.broken = False
    tick(store, T0 + timedelta(minutes=1), relay=relay, relayed=relayed)
    assert len(relay.calls) == 1 and store.get(1).relayed_at is not None and relayed == set()


def test_a_question_back_from_kael_goes_to_the_owner_but_progress_does_not():
    store, relay = MemoryTicketStore(), Recorder()
    owed(store, outbox=None, delivered_at=T0)
    store.from_kael(1, "working", "on it, about an hour", T0)
    assert tick(store, T0, relay=relay)["relayed"] == []
    store.from_kael(1, "input-required", "which phone, the work one?", T0 + timedelta(minutes=5))
    assert tick(store, T0 + timedelta(minutes=5), relay=relay)["relayed"] == ["T-1"]


def test_an_old_unrelayed_answer_is_dropped_after_a_day():
    store, relay = MemoryTicketStore(), Recorder()
    owed(store, outbox=None, delivered_at=T0)
    store.from_kael(1, "completed", "yes", T0)
    assert tick(store, T0 + timedelta(hours=25), relay=relay)["relayed"] == []


def test_a_naive_now_is_refused():
    with pytest.raises(ValueError):
        tick(MemoryTicketStore(), datetime(2026, 11, 8, 12, 0))


# --- Kael's side of the door -------------------------------------------------------------------

def door_app(store, notes, wake=None):
    def note_fn(thread_id, text, family_visible=False, *, proactive=True):
        notes.append((thread_id, text, family_visible, proactive))
        return "kn-1"

    door = TicketDoor(store=store, update=kael_update_for(store, note_fn, wake, clock=lambda: T0))
    return TestClient(build_app(lambda *a, **k: "unused", "sekrit", kael_tickets=door))


AUTH = {"Authorization": "Bearer sekrit"}


def test_kaels_answer_lands_in_her_thread_without_a_second_proactive_turn():
    store, notes, wake = MemoryTicketStore(), [], threading.Event()
    owed(store, outbox=None, delivered_at=T0)
    c = door_app(store, notes, wake)
    r = c.post("/kael-ticket/1", json={"status": "completed", "text": "yes, green"}, headers=AUTH)
    assert r.status_code == 200 and r.json()["status"] == "completed" and r.json()["label"] == "T-1"
    assert notes == [("person:" + OWNER, "(ticket T-1, completed) yes, green", False, False)]
    assert store.get(1).relay_owed and wake.is_set()        # the watcher brings it now


def test_the_ticket_door_needs_the_token_a_real_status_and_words_for_an_answer():
    store, notes = MemoryTicketStore(), []
    owed(store)
    c = door_app(store, notes)
    assert c.post("/kael-ticket/1", json={"status": "completed", "text": "x"}).status_code in (401, 403)
    assert c.post("/kael-ticket/1", json={"status": "done", "text": "x"}, headers=AUTH).status_code == 422
    assert c.post("/kael-ticket/1", json={"status": "completed"}, headers=AUTH).status_code == 422
    assert c.post("/kael-ticket/9", json={"status": "working"}, headers=AUTH).status_code == 404
    assert c.post("/kael-ticket/1", json={"status": "working"}, headers=AUTH).status_code == 200
    assert notes == []                                        # progress without words: no note


def test_kael_lists_open_tickets_and_reads_one():
    store = MemoryTicketStore()
    owed(store)
    owed(store, ask="second", status="completed")
    c = door_app(store, [])
    assert [t["label"] for t in c.get("/kael-tickets", headers=AUTH).json()["tickets"]] == ["T-1"]
    assert len(c.get("/kael-tickets?open_only=false", headers=AUTH).json()["tickets"]) == 2
    assert c.get("/kael-ticket/2", headers=AUTH).json()["ask"] == "second"
    assert c.get("/kael-ticket/7", headers=AUTH).status_code == 404


def test_an_unarmed_door_is_an_honest_503():
    c = TestClient(build_app(lambda *a, **k: "unused", "sekrit"))
    assert c.get("/kael-tickets", headers=AUTH).status_code == 503


# --- her tool --------------------------------------------------------------------------------

def her_tool(store, send=None):
    return build_kael_tickets_tool(store, send or Recorder(), clock=lambda: T0)


def test_she_can_check_a_ticket_and_read_kaels_words():
    store = MemoryTicketStore()
    owed(store, outbox=None, delivered_at=T0)
    store.from_kael(1, "completed", "yes, green", T0)
    out = her_tool(store).invoke({"action": "check", "ticket": "T-1"})
    assert "answered" in out and "yes, green" in out


def test_she_adds_to_an_open_ticket_instead_of_resending():
    store, send = MemoryTicketStore(), Recorder()
    owed(store, outbox=None, delivered_at=T0, status="input-required")
    out = her_tool(store, send).invoke({"action": "update", "ticket": "1", "message": "the work phone"})
    t = store.get(1)
    assert "Kael has it now" in out and t.status == "submitted" and t.outbox is None
    assert send.calls[0][1] == "update: the work phone"


def test_a_closed_ticket_takes_no_updates():
    store, send = MemoryTicketStore(), Recorder()
    owed(store, status="completed")
    out = her_tool(store, send).invoke({"action": "update", "ticket": "T-1", "message": "also"})
    assert out.startswith("NOT DONE") and "message_kael" in out and send.calls == []


def test_cancel_closes_it_and_tells_kael_only_if_he_had_it():
    store, send = MemoryTicketStore(), Recorder()
    owed(store, outbox=None, delivered_at=T0)
    owed(store, ask="never sent")
    tool = her_tool(store, send)
    assert tool.invoke({"action": "cancel", "ticket": "T-1"}) == "Canceled T-1."
    assert tool.invoke({"action": "cancel", "ticket": "T-2"}) == "Canceled T-2."
    assert store.get(1).status == store.get(2).status == "canceled"
    assert [c[0].id for c in send.calls] == [1]
    assert "already closed" in tool.invoke({"action": "cancel", "ticket": "T-1"})


def test_her_list_and_honest_failures():
    store = MemoryTicketStore()
    assert her_tool(store).invoke({"action": "list"}) == "You haven't asked Kael anything yet."
    owed(store)
    assert "T-1" in her_tool(store).invoke({"action": "list"})
    assert her_tool(store).invoke({"action": "check", "ticket": "nope"}).startswith("NOT DONE")
    assert her_tool(store).invoke({"action": "check", "ticket": "T-9"}).startswith("NOT DONE")
    assert her_tool(store).invoke({"action": "shout"}).startswith("NOT DONE")


# --- the line ----------------------------------------------------------------------------

def test_the_line_carries_the_ticket_and_its_return_address():
    seen = []
    send = line_sender(URL, "tok", client=line(202, seen))
    t = Ticket(id=4, thread_id="person:x", ask="a")
    assert send(t, "still open") is True
    assert seen == [{"message": "(ticket T-4) still open", "thread_id": "person:x", "ticket_id": "T-4"}]
    assert line_sender(URL, "tok", client=line(429))(t, "x") is False
    assert line_sender(URL, "tok", client=line(httpx.ConnectError("x")))(t, "x") is False


# --- reaching the owner ------------------------------------------------------------------------

class _Secret:
    def __init__(self, v):
        self.v = v

    def get_secret_value(self):
        return self.v


def owner_settings(**kw):
    base = dict(owner_person_id=OWNER, telegram_bot_token=_Secret("tg"), discord_bot_token=_Secret("dc"),
                owner_quiet_start_hour=21, owner_quiet_end_hour=9, database_url=None,
                memories_database_url=None, soul_file_path=__import__("pathlib").Path("/nonexistent/soul.md"))
    base.update(kw)
    return SimpleNamespace(**base)


def eastern(h, m=0):
    """11/08 at h:m Eastern (EST, UTC-5)."""
    return datetime(2026, 11, 8, tzinfo=timezone.utc) + timedelta(hours=h + 5, minutes=m)


@pytest.mark.parametrize("h,quiet", [(2, True), (8, True), (9, False), (14, False), (20, False),
                                     (21, True), (23, True)])
def test_quiet_hours_are_nine_pm_to_nine_am_eastern(h, quiet):
    assert in_quiet_hours(eastern(h), 21, 9) is quiet


def test_no_quiet_hours_when_start_equals_end():
    assert in_quiet_hours(eastern(23), 9, 9) is False


def reach_with(surface, now, sent, device=None):
    return owner_reach_for(owner_settings(), send=lambda *a: sent.append(a),
                           lookup_surface=lambda s: (surface, device),
                           lookup_chat_id=lambda s, p: f"{p}-id", clock=lambda: now)


def test_by_day_it_goes_where_he_was_last_out_loud_included():
    sent = []
    assert reach_with("device", eastern(14), sent, device="office")("hi") is True
    assert sent == [("device", "office", "hi", False)]


def test_at_night_nothing_is_spoken_and_the_text_is_silent():
    sent = []
    assert reach_with("device", eastern(23), sent, device="bedroom")("hi") is True
    assert sent == [("discord", "discord-id", "hi", True)]


def test_a_device_that_fails_falls_back_to_his_discord():
    calls = []

    def send(surface, target, text, quiet):
        calls.append(surface)
        if surface == "device":
            raise RuntimeError("satellite offline")

    reach = owner_reach_for(owner_settings(), send=send, lookup_surface=lambda s: ("device", "office"),
                            lookup_chat_id=lambda s, p: "id", clock=lambda: eastern(14))
    assert reach("hi") is True and calls == ["device", "discord"]


def test_no_owner_means_no_reach():
    assert owner_reach_for(owner_settings(owner_person_id=None)) is None


class FakeGraph:
    def __init__(self):
        self.updates = []

    def update_state(self, config, values, **kw):
        self.updates.append((config["configurable"]["thread_id"], values["messages"]))


class SpeakModel:
    def __init__(self, reply):
        self.reply, self.prompts = reply, []

    def invoke(self, messages):
        self.prompts.append(messages)
        if isinstance(self.reply, Exception):
            raise self.reply
        return AIMessage(content=self.reply)


def answered(text="yes, green"):
    return Ticket(id=3, thread_id="person:" + OWNER, ask="is the backup green?", for_owner=True,
                  status="completed", answer=text, created_at=T0)


def test_the_relay_is_her_words_with_no_hold_and_the_thread_remembers():
    graph, sent, model = FakeGraph(), [], SpeakModel("Kael says last night's backup was green!")
    relay = ticket_relay_for(graph, owner_settings(), model=model, reach=lambda t: sent.append(t) or True)
    assert relay(answered()) is True
    assert sent == ["Kael says last night's backup was green!"]
    system, human = model.prompts[0]
    assert system.content.endswith(TICKET_RELAY_PROMPT) and "HOLD" not in TICKET_RELAY_PROMPT
    assert "is the backup green?" in human.content and "yes, green" in human.content
    assert graph.updates == [("person:" + OWNER, [AIMessage(content=sent[0])])]


@pytest.mark.parametrize("reply", [RuntimeError("model down"), "", "HOLD"])
def test_when_her_words_fail_kaels_answer_still_goes_plainly(reply):
    sent = []
    relay = ticket_relay_for(FakeGraph(), owner_settings(), model=SpeakModel(reply),
                             reach=lambda t: sent.append(t) or True)
    assert relay(answered()) is True
    assert sent == ["Kael got back to you (T-3): yes, green"]


def test_a_relay_that_cannot_reach_him_reports_false_and_leaves_the_thread_alone():
    graph = FakeGraph()
    relay = ticket_relay_for(graph, owner_settings(), model=SpeakModel("hi"), reach=lambda t: False)
    assert relay(answered()) is False and graph.updates == []


def test_the_owner_notice_says_exactly_what_is_true():
    sent = []
    notice = owner_notice_for(FakeGraph(), owner_settings(), reach=lambda t: sent.append(t) or True)
    t = answered()
    assert notice(t, True) and notice(t, False)
    assert "haven't heard back" in sent[0] and "Sun 11/8 at 11:00 AM" in sent[0] and "T-3" in sent[0]
    assert "couldn't get through" in sent[1] and "keep trying" in sent[1]


def test_a_ticket_note_never_earns_the_hold_able_proactive_turn():
    graph, spoke = FakeGraph(), []
    note = kael_note_for(graph, owner_settings(), proactive_model=SpeakModel("hi"),
                         proactive_send=lambda *a: spoke.append(a),
                         proactive_lookup_chat_id=lambda s: "id",
                         proactive_lookup_surface=lambda s: ("discord", None), proactive_sync=True)
    note("person:" + OWNER, "(ticket T-3, completed) yes", True, proactive=False)
    assert spoke == [] and len(graph.updates) == 1
    note("person:" + OWNER, "a family note", True)
    assert len(spoke) == 1                                    # the old path is unchanged


# --- Codex review 11/08: the four it found ----------------------------------------------------

def test_an_update_queued_while_a_resend_is_out_is_not_erased():
    store = MemoryTicketStore()
    owed(store)

    def send(t, text):                     # while the old text is on the wire, she adds more
        if text == "is the backup green?":
            store.from_her(1, "update: and the NAS copy?", T0 + timedelta(minutes=10))
        return True

    tick(store, T0 + timedelta(minutes=10), send=send)
    assert store.get(1).outbox == "is the backup green?\nupdate: and the NAS copy?"


def test_updates_while_the_line_is_down_keep_everything_in_order():
    store = MemoryTicketStore()
    owed(store)
    tool = her_tool(store, Recorder(ok=False))
    tool.invoke({"action": "update", "ticket": "T-1", "message": "first"})
    tool.invoke({"action": "update", "ticket": "T-1", "message": "second"})
    assert store.get(1).outbox == "is the backup green?\nupdate: first\nupdate: second"


def test_progress_never_replaces_a_question_still_owed_to_the_owner():
    store, relay = MemoryTicketStore(), Recorder()
    owed(store, outbox=None, delivered_at=T0)
    store.from_kael(1, "input-required", "which phone?", T0)
    store.from_kael(1, "working", "looking meanwhile", T0 + timedelta(minutes=1))
    tick(store, T0 + timedelta(minutes=1), relay=relay)
    assert relay.calls[0][0].answer == "which phone?"
    assert store.get(1).progress == "looking meanwhile"


def test_cancel_tells_kael_when_he_had_the_ask_even_with_an_update_still_queued():
    store, send = MemoryTicketStore(), Recorder()
    owed(store, outbox="update: more", delivered_at=T0)
    her_tool(store, send).invoke({"action": "cancel", "ticket": "T-1"})
    assert [c[0].id for c in send.calls] == [1]


def test_an_update_that_would_not_fit_in_one_send_is_refused_not_cut():
    from aerys_v2.tickets import LINE_LIMIT, OUTBOX_LIMIT, line_text

    store = MemoryTicketStore()
    owed(store, ask="x" * 2000)
    tool = her_tool(store, Recorder(ok=False))
    assert "Added to T-1" in tool.invoke({"action": "update", "ticket": "T-1", "message": "y" * 1400})
    out = tool.invoke({"action": "update", "ticket": "T-1", "message": "z" * 1400})
    assert out.startswith("NOT DONE") and "z" not in store.get(1).outbox
    assert len(store.get(1).outbox) <= OUTBOX_LIMIT
    assert line_text(store.get(1), store.get(1).outbox).endswith("y" * 1400)   # nothing cut
    assert OUTBOX_LIMIT + len("(ticket T-999999) ") < LINE_LIMIT
    assert tool.invoke({"action": "update", "ticket": "T-1", "message": "w" * 1600}).startswith("NOT DONE")


def test_a_refused_append_is_never_reported_as_added_even_if_the_text_matches():
    store = MemoryTicketStore()
    owed(store, ask="x" * 3790)
    store.items[1].outbox = "x" * 3780 + "\nupdate: hi"      # same tail, but no room left
    out = her_tool(store, Recorder()).invoke({"action": "update", "ticket": "T-1", "message": "hi"})
    assert out.startswith("NOT DONE")
