"""The owner's approval button (board #176, Stage 4's shared piece). D10 (11/08): from voice,
"anything text can but requires me to approve via discord button"; D12 (11/09): the first message
to anyone new waits for his tap. Fail closed: only an explicit Approve from him, in time, counts."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from aerys_v2.approvals import (
    CUSTOM_ID,
    Approvals,
    MemoryApprovalStore,
    approvals_for,
    attach_approvals,
    custom_id,
    edit_dm,
    PING_RETRY_AFTER,
    handle_click,
    message_text,
    on_tap,
    outcome_text,
    owner_id_for,
    poll_once,
)

NOW = datetime(2026, 11, 9, 15, 0, tzinfo=timezone.utc)
OWNER = 100000000000000001
SOMEONE = 111


class Desk:
    def __init__(self, now=NOW):
        self.clock = {"now": now}
        self.store = MemoryApprovalStore()
        self.pinged = []
        self.approvals = Approvals(self.store, notify=lambda title, msg: self.pinged.append((title, msg)),
                                   clock=lambda: self.clock["now"], sleep=lambda s: None)


def test_asking_twice_for_the_same_thing_is_one_request_and_one_ping():
    d = Desk()
    a = d.approvals.request("message", "First message to Sam · Discord: sam_42", key="msg:sam:1")
    b = d.approvals.request("message", "First message to Sam · Discord: sam_42", key="msg:sam:1")
    assert a == b and len(d.store.rows) == 1
    assert len(d.pinged) == 1 and "Sam" in d.pinged[0][1] and "Discord" in d.pinged[0][1]
    assert d.approvals.status(a) == "pending" and not d.approvals.approved(a)


def test_a_request_needs_words_and_a_sane_expiry():
    d = Desk()
    with pytest.raises(ValueError):
        d.approvals.request("message", "   ")
    with pytest.raises(ValueError):
        d.approvals.request("message", "x", expires_in=timedelta(days=8))
    with pytest.raises(ValueError):
        d.approvals.request("message", "x", expires_in=timedelta(0))


def test_a_failed_ping_never_fails_the_request():
    d = Desk()
    d.approvals.notify = lambda title, msg: (_ for _ in ()).throw(ConnectionError("HA down"))
    aid = d.approvals.request("job", "Start the plane tracker")
    assert d.approvals.status(aid) == "pending"


def test_only_his_tap_decides_and_only_once():
    d = Desk()
    aid = d.approvals.request("job", "Start the plane tracker")
    reply, row = handle_click(d.store, aid, True, SOMEONE, OWNER, NOW)
    assert reply == "Not yours to answer." and row is None and d.approvals.status(aid) == "pending"
    reply, row = handle_click(d.store, aid, True, OWNER, OWNER, NOW)
    assert reply == "Approved." and row.status == "approved" and row.decided_by == str(OWNER)
    reply, row = handle_click(d.store, aid, False, OWNER, OWNER, NOW)       # a second tap
    assert row is None and "already answered" in reply and d.approvals.approved(aid)


def test_no_owner_known_means_nobody_can_approve():
    d = Desk()
    aid = d.approvals.request("job", "x")
    reply, row = handle_click(d.store, aid, True, OWNER, None, NOW)
    assert row is None and d.approvals.status(aid) == "pending"


def test_a_tap_after_the_expiry_changes_nothing_and_silence_is_a_no():
    d = Desk()
    aid = d.approvals.request("job", "x", expires_in=timedelta(minutes=10))
    later = NOW + timedelta(minutes=11)
    d.clock["now"] = later
    assert d.approvals.status(aid) == "expired" and not d.approvals.approved(aid)
    reply, row = handle_click(d.store, aid, True, OWNER, OWNER, later)
    assert row is None and not d.approvals.approved(aid)


def test_a_decline_is_a_no():
    d = Desk()
    aid = d.approvals.request("job", "x")
    handle_click(d.store, aid, False, OWNER, OWNER, NOW)
    assert d.approvals.status(aid) == "declined" and not d.approvals.approved(aid)


def test_withdrawn_requests_cant_be_approved_later():
    d = Desk()
    aid = d.approvals.request("job", "x")
    assert d.approvals.withdraw(aid)
    reply, row = handle_click(d.store, aid, True, OWNER, OWNER, NOW)
    assert row is None and d.approvals.status(aid) == "withdrawn"


def test_a_record_that_cant_be_read_is_never_an_approval():
    d = Desk()
    aid = d.approvals.request("job", "x")
    handle_click(d.store, aid, True, OWNER, OWNER, NOW)
    d.store.get = lambda i: (_ for _ in ()).throw(ConnectionError("NAS down"))
    assert d.approvals.status(aid) == "unknown" and not d.approvals.approved(aid)


def test_a_tap_that_cant_be_recorded_approves_nothing():
    d = Desk()
    aid = d.approvals.request("job", "x")
    d.store.decide = lambda *a: (_ for _ in ()).throw(ConnectionError("NAS down"))
    reply, row = handle_click(d.store, aid, True, OWNER, OWNER, NOW)
    assert row is None and "Nothing was approved" in reply and d.store.rows[aid].status == "pending"


def test_wait_returns_his_answer_or_gives_up():
    d = Desk()
    aid = d.approvals.request("job", "x")
    d.approvals.sleep = lambda s: handle_click(d.store, aid, True, OWNER, OWNER, NOW)   # he taps meanwhile
    assert d.approvals.wait(aid, timeout_s=5) == "approved"
    other = d.approvals.request("job", "y")
    d.approvals.sleep = lambda s: None
    assert d.approvals.wait(other, timeout_s=0) == "pending"


def test_the_poller_posts_each_button_once_and_closes_what_expired():
    d = Desk()
    a = d.approvals.request("job", "Start the plane tracker", expires_in=timedelta(minutes=10))
    b = d.approvals.request("message", "First message to Sam · Discord: sam_42")
    sent, edits = [], []

    async def send(row):
        sent.append(row.id)
        return "dm", f"m{row.id}"

    async def edit(row, text):
        edits.append((row.message_id, text))

    out = asyncio.run(poll_once(d.store, send=send, edit=edit, now=NOW))
    assert out["posted"] == 2 and sent == [a, b] and d.store.rows[a].channel_id == "dm"
    assert asyncio.run(poll_once(d.store, send=send, edit=edit, now=NOW))["posted"] == 0      # never twice
    asyncio.run(poll_once(d.store, send=send, edit=edit, now=NOW + timedelta(minutes=11)))
    assert edits == [(f"m{a}", outcome_text(d.store.rows[a]))] and d.store.rows[a].status == "expired"
    assert d.store.rows[b].status == "pending"
    asyncio.run(poll_once(d.store, send=send, edit=edit, now=NOW + timedelta(minutes=12)))
    assert len(edits) == 1                                                  # shown once it took


def test_a_button_that_couldnt_be_posted_is_tried_again():
    d = Desk()
    aid = d.approvals.request("job", "x")
    calls = {"n": 0}

    async def flaky(row):
        calls["n"] += 1
        if calls["n"] == 1:
            raise ConnectionError("discord down")
        return "dm", "m1"

    async def edit(row, text):
        pass

    assert asyncio.run(poll_once(d.store, send=flaky, edit=edit, now=NOW))["posted"] == 0
    assert d.store.rows[aid].message_id is None
    assert asyncio.run(poll_once(d.store, send=flaky, edit=edit, now=NOW))["posted"] == 1
    assert d.store.rows[aid].message_id == "m1"


def test_every_gateway_start_registers_the_buttons_again():
    class FakeClient:
        def __init__(self):
            self.registered, self.tasks = [], []
            self.loop = SimpleNamespace(create_task=lambda coro: (self.tasks.append(coro), coro.close()))

        async def setup_hook(self):
            pass

        def add_dynamic_items(self, *items):
            self.registered.extend(items)

    client = FakeClient()
    button = attach_approvals(client, store=MemoryApprovalStore(), owner_id=lambda: OWNER)
    for _ in range(2):                              # a restart is just another start
        asyncio.run(client.setup_hook())
    assert client.registered == [button, button] and len(client.tasks) == 2
    assert button.__discord_ui_compiled_template__.fullmatch(custom_id(42, "approve"))
    assert CUSTOM_ID.fullmatch(custom_id(42, "decline"))["id"] == "42"
    assert not CUSTOM_ID.fullmatch("aerys-approval:42:maybe")


def test_the_message_says_what_and_until_when():
    d = Desk()
    aid = d.approvals.request("message", "First message to Sam · Discord: sam_42", detail="Hey Sam, Chris asked me to say hi.")
    text = message_text(d.store.rows[aid])
    assert "Sam · Discord: sam_42" in text and "Hey Sam" in text and "no answer is a no" in text
    assert f"<t:{int((NOW + timedelta(minutes=30)).timestamp())}:t>" in text


def test_the_owner_is_looked_up_fresh_at_every_tap():
    answers = iter([None, str(OWNER), "999", ConnectionError("db down"), str(OWNER)])

    def lookup(settings):
        a = next(answers)
        if isinstance(a, Exception):
            raise a
        return a

    owner = owner_id_for(SimpleNamespace(approvals_owner_discord_id=""), lookup=lookup)
    # a corrected identity takes effect at once; a failed lookup means nobody, never the last answer
    assert [owner(), owner(), owner(), owner(), owner()] == [None, OWNER, 999, None, OWNER]
    assert owner_id_for(SimpleNamespace(approvals_owner_discord_id="123"), lookup=lookup)() == 123


ARMED = dict(approvals=True, database_url="postgresql://x", owner_person_id="p", reach_notify_service="",
             storm_notify_service="svc", ha_token=SimpleNamespace(get_secret_value=lambda: "t"),
             ha_base_url="http://ha:8123")


@pytest.mark.parametrize("kw", [dict(approvals=False), dict(database_url=None), dict(owner_person_id=None),
                                dict(storm_notify_service=""), dict(ha_token=None)])
def test_off_unless_armed(kw):
    assert approvals_for(SimpleNamespace(**{**ARMED, **kw})) is None       # no HA ping, no button either


def test_armed():
    a = approvals_for(SimpleNamespace(**ARMED))
    assert isinstance(a, Approvals) and a.notify is not None


# --- after review: acknowledge first, claim before sending, keep correcting, keep pinging ---


class Tap:
    """A Discord interaction, recording what was called in what order."""

    def __init__(self, user_id, *, edit_fails=False):
        self.calls, self.user = [], SimpleNamespace(id=user_id)
        self.edit_fails = edit_fails
        tap = self

        class Response:
            async def defer(self):
                tap.calls.append(("defer",))

        class Followup:
            async def send(self, text, ephemeral=False):
                tap.calls.append(("followup", text, ephemeral))

        self.response, self.followup = Response(), Followup()

    async def edit_original_response(self, content=None, view="unset"):
        if self.edit_fails:
            raise ConnectionError("discord hiccup")
        self.calls.append(("edit", content, view))


def test_a_tap_is_acknowledged_before_anything_slow():
    d = Desk()
    aid = d.approvals.request("job", "Start the plane tracker")
    tap = Tap(OWNER)

    def owner():
        assert tap.calls == [("defer",)], "Discord gets its answer before the owner lookup or the database"
        return OWNER

    asyncio.run(on_tap(tap, aid, True, store=d.store, owner_id=owner, clock=lambda: NOW))
    assert tap.calls[1] == ("edit", outcome_text(d.store.rows[aid]), None)       # buttons gone
    assert d.store.rows[aid].status == "approved" and d.store.rows[aid].shown == "approved"


def test_a_tap_from_anyone_else_is_told_privately_and_changes_nothing():
    d = Desk()
    aid = d.approvals.request("job", "x")
    tap = Tap(SOMEONE)
    asyncio.run(on_tap(tap, aid, True, store=d.store, owner_id=lambda: OWNER))
    assert tap.calls == [("defer",), ("followup", "Not yours to answer.", True)]
    assert d.store.rows[aid].status == "pending"


def test_a_tap_whose_edit_failed_is_corrected_by_the_poller():
    d = Desk()
    aid = d.approvals.request("job", "x")
    d.store.rows[aid].message_id, d.store.rows[aid].channel_id = "m1", "dm"
    asyncio.run(on_tap(Tap(OWNER, edit_fails=True), aid, False, store=d.store, owner_id=lambda: OWNER,
                       clock=lambda: NOW))
    assert d.store.rows[aid].status == "declined" and d.store.rows[aid].shown is None
    edits = []

    async def edit(row, text):
        edits.append((row.message_id, text))

    async def send(row):
        raise AssertionError("nothing to post")

    asyncio.run(poll_once(d.store, send=send, edit=edit, now=NOW))
    assert edits == [("m1", outcome_text(d.store.rows[aid]))] and d.store.rows[aid].shown == "declined"


def test_an_expiry_edit_that_failed_is_tried_again_until_it_takes():
    d = Desk()
    aid = d.approvals.request("job", "x", expires_in=timedelta(minutes=10))
    d.store.rows[aid].message_id, d.store.rows[aid].channel_id = "m1", "dm"
    tries = []

    async def edit(row, text):
        tries.append(row.message_id)
        if len(tries) == 1:
            raise ConnectionError("discord down")

    async def send(row):
        raise AssertionError("nothing to post")

    later = NOW + timedelta(minutes=11)
    asyncio.run(poll_once(d.store, send=send, edit=edit, now=later))
    assert d.store.rows[aid].status == "expired" and d.store.rows[aid].shown is None
    asyncio.run(poll_once(d.store, send=send, edit=edit, now=later))
    asyncio.run(poll_once(d.store, send=send, edit=edit, now=later))
    assert tries == ["m1", "m1"] and d.store.rows[aid].shown == "expired"


def test_two_passes_at_once_send_one_dm():
    d = Desk()
    aid = d.approvals.request("job", "x")
    sent = []

    async def send(row):
        sent.append(row.id)
        await asyncio.sleep(0.05)                   # the other pass runs meanwhile
        return "dm", "m1"

    async def edit(row, text):
        pass

    async def both():
        return await asyncio.gather(poll_once(d.store, send=send, edit=edit, now=NOW),
                                    poll_once(d.store, send=send, edit=edit, now=NOW))

    asyncio.run(both())
    assert sent == [aid] and d.store.rows[aid].message_id == "m1"


def test_a_dm_that_went_but_wasnt_recorded_is_never_sent_again():
    d = Desk()
    aid = d.approvals.request("job", "x")
    sent = []

    async def send(row):
        sent.append(row.id)
        return "dm", "m1"

    async def edit(row, text):
        pass

    d.store.posted = lambda *a: (_ for _ in ()).throw(ConnectionError("NAS down"))
    for _ in range(3):
        asyncio.run(poll_once(d.store, send=send, edit=edit, now=NOW))
    assert sent == [aid]


def test_a_ping_that_didnt_land_is_sent_again_until_it_does():
    d = Desk()
    d.approvals.notify = lambda title, msg: (_ for _ in ()).throw(ConnectionError("HA down"))
    aid = d.approvals.request("job", "Start the plane tracker")
    d.approvals.request("job", "y", expires_in=timedelta(seconds=30))     # gone before its retry
    assert d.store.rows[aid].pinged_at is None
    pings = []

    async def send(row):
        return "dm", f"m{row.id}"

    async def edit(row, text):
        pass

    def notify(title, msg):
        pings.append(msg)

    asyncio.run(poll_once(d.store, send=send, edit=edit, notify=notify, now=NOW))
    assert pings == []                                       # the first try was just now
    later = NOW + PING_RETRY_AFTER
    asyncio.run(poll_once(d.store, send=send, edit=edit, notify=notify, now=later))
    asyncio.run(poll_once(d.store, send=send, edit=edit, notify=notify, now=later))
    assert len(pings) == 1 and "plane tracker" in pings[0] and d.store.rows[aid].pinged_at == later


def test_a_ping_that_landed_is_recorded():
    d = Desk()
    aid = d.approvals.request("job", "x")
    assert len(d.pinged) == 1 and d.store.rows[aid].pinged_at == NOW


def test_a_dm_whose_record_failed_is_recorded_later_and_still_shows_its_outcome():
    d = Desk()
    aid = d.approvals.request("job", "x", expires_in=timedelta(minutes=10))
    sent, edits, unrecorded = [], [], {}
    real_posted = d.store.posted
    d.store.posted = lambda *a: (_ for _ in ()).throw(ConnectionError("NAS down"))

    async def send(row):
        sent.append(row.id)
        return "dm", "m1"

    async def edit(row, text):
        edits.append((row.channel_id, row.message_id, text))

    asyncio.run(poll_once(d.store, send=send, edit=edit, now=NOW, unrecorded=unrecorded))
    assert d.store.rows[aid].message_id.startswith("posting:") and aid in unrecorded
    d.store.posted = real_posted                             # the NAS is back
    later = NOW + timedelta(minutes=11)
    asyncio.run(poll_once(d.store, send=send, edit=edit, now=later, unrecorded=unrecorded))
    assert sent == [aid] and unrecorded == {}
    assert edits == [("dm", "m1", outcome_text(d.store.rows[aid]))] and d.store.rows[aid].shown == "expired"


def test_an_outcome_edit_goes_to_the_dm_it_was_posted_in():
    """Not to whoever the owner is now: a corrected owner (or a failed lookup) can't strand it."""
    calls = []

    class Message:
        def __init__(self, mid):
            self.mid = mid

        async def edit(self, content=None, view="unset"):
            calls.append(("edit", self.mid, content, view))

    class Channel:
        def __init__(self, cid):
            self.cid = cid

        def get_partial_message(self, mid):
            calls.append(("message", self.cid, mid))
            return Message(mid)

    class Client:
        def get_partial_messageable(self, cid, *, type=None):
            calls.append(("channel", cid))
            return Channel(cid)

    d = Desk()
    aid = d.approvals.request("job", "x")
    row = d.store.rows[aid]
    row.channel_id, row.message_id = "222", "333"
    asyncio.run(edit_dm(Client(), row, "**Approved** (job)"))
    assert calls == [("channel", 222), ("message", 222, 333), ("edit", 333, "**Approved** (job)", None)]


def test_multiple_approvers_first_tap_wins_and_all_posts_update():
    d = Desk()
    aid = d.approvals.request("job", "Shared request", approvers=[str(OWNER), str(SOMEONE)])
    sent, edited = [], []

    async def send(a):
        sent.append(a.recipient)
        return a.recipient, str(len(sent))

    async def edit(a, text):
        edited.append((a.channel_id, text))

    asyncio.run(poll_once(d.store, send=send, edit=edit, now=NOW))
    assert set(sent) == {str(OWNER), str(SOMEONE)}
    assert handle_click(d.store, aid, True, 999, OWNER, NOW)[1] is None
    assert handle_click(d.store, aid, True, SOMEONE, OWNER, NOW)[1].decided_by == str(SOMEONE)
    assert handle_click(d.store, aid, False, OWNER, OWNER, NOW)[1] is None
    asyncio.run(poll_once(d.store, send=send, edit=edit, now=NOW))
    assert {channel for channel, _ in edited} == set(sent)
    assert all("Approved" in text for _, text in edited)
    assert d.store.get(aid).status == "approved"
