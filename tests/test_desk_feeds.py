"""Storm and email pings through the reach-out desk (board #176, Stage 2: "Storm, Kael-note and
email pings routed through it"). The desk's rules then apply to them: a storm WARNING breaks
through the night (D4), other storm news goes now as quiet text at night, and email waits
for the morning brief like any routine item."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

import aerys_v2.storm_watch as sw
from aerys_v2.reach import MemoryReachStore, Prefs, ReachDesk, ReachEvent, Where
from aerys_v2.storm_watch import RADAR_IMAGE_URL, StormNotice, StormWatcher, storm_reach_for
from aerys_v2.workers.email_watch import format_ping, reach_notify_for, run_once, split_ping, EmailSummary
from test_email_watch import UV, FakeImap, conn_with_watermark, raw_email
from test_morning_brief import IN_OFFICE, NIGHT, House
from test_reach_desk import World

ET = timezone(timedelta(hours=-5))
DAY = datetime(2026, 11, 9, 14, 0, tzinfo=ET)


def quiet(t):
    hour = t.astimezone(ET).hour
    return hour >= 21 or hour < 9


def desk_at(now, world=None, *, store=None, budget=3, muted=()):
    world = world or World(Where("room", "dev-bedroom", "bedroom"))
    store = store or MemoryReachStore(prefs=Prefs(budget=budget, muted=list(muted)))
    clock = {"now": now}
    desk = ReachDesk(store, world.deliverer(), budget_default=budget, quiet=quiet, morning_hour=13,
                     replied_since=lambda when: world.replied,
                     said=lambda row, body: world.said.append(body), clock=lambda: clock["now"])
    return desk, world, clock


# --- storms ---------------------------------------------------------------------------------

ALERT = {"features": [{"properties": {"id": "urn:nws:1", "event": "Hurricane Warning",
                                      "headline": "Hurricane Warning issued", "severity": "Extreme",
                                      "instruction": "Finish preparations."}}]}
WATCH = {"features": [{"properties": {"id": "urn:nws:2", "event": "Tropical Storm Watch",
                                      "headline": "Tropical Storm Watch issued"}}]}


def watcher(desk, legs, *, alerts=None, storms=None, fallback=None):
    return StormWatcher(point="0,0", deliver=lambda t, m, i: legs.append((t, m, i)),
                        fetch_json=lambda url: alerts if "alerts" in url else (storms or {"activeStorms": []}),
                        fetch_text=lambda url: "", sleep_fn=lambda s: None,
                        reach=storm_reach_for(desk, fallback=fallback))


def test_a_storm_warning_breaks_through_the_night_spoken_where_he_is_and_the_phone_keeps_its_picture():
    desk, world, _ = desk_at(NIGHT)
    phone = []
    watcher(desk, phone, alerts=ALERT).poll_alerts()
    assert world.spoken == [("Heads up: the weather service has a Hurricane Warning out for here.", "dev-bedroom")]
    assert "Finish preparations." in world.texts[0][1] and RADAR_IMAGE_URL in world.texts[0][1]
    assert phone == [("⚠️ Hurricane Warning", phone[0][1], RADAR_IMAGE_URL)]      # the image leg, unchanged
    row = desk.store.rows[1]
    assert row.urgency == "safety" and row.topic == "weather" and row.dedupe_key == "storm:alert:urn:nws:1"


def test_a_watch_at_night_goes_as_quiet_text_never_spoken():
    desk, world, _ = desk_at(NIGHT)
    watcher(desk, [], alerts=WATCH).poll_alerts()
    assert world.spoken == [] and len(world.texts) == 1 and world.texts[0][2] is True     # silent
    assert desk.store.rows[1].urgency == "timely"


def test_the_budget_never_drops_a_storm_heads_up_but_he_can_mute_weather():
    desk, world, _ = desk_at(DAY, budget=0)
    storms = {"activeStorms": [{"id": "al052026", "name": "Ernesto", "classification": "TS"}]}
    w = watcher(desk, [], storms=storms)
    w._check_new_storms(storms)
    assert world.spoken == [("The hurricane center is now tracking Ernesto.", "dev-bedroom")]
    desk2, world2, _ = desk_at(DAY, muted=["weather"])
    watcher(desk2, [], storms=storms)._check_new_storms(storms)
    assert world2.spoken == [] and desk2.store.rows[1].status == "dropped"


def test_a_warning_is_never_muted():
    desk, world, _ = desk_at(DAY, muted=["weather"])
    watcher(desk, [], alerts=ALERT).poll_alerts()
    assert len(world.spoken) == 1


def test_if_the_desk_fails_the_old_dm_goes():
    class Broken:
        def reach_out(self, event):
            raise ConnectionError("NAS down")

    dms = []
    watcher(Broken(), [], alerts=ALERT, fallback=lambda t, m, i: dms.append((t, i))).poll_alerts()
    assert dms == [("⚠️ Hurricane Warning", RADAR_IMAGE_URL)]


def test_with_a_desk_the_discord_leg_is_the_desk_and_the_phone_leg_stays(monkeypatch):
    built = {}
    monkeypatch.setattr(sw, "StormWatcher", lambda **kw: built.update(kw) or SimpleNamespace(run_forever=lambda: None))
    calls = []
    monkeypatch.setattr(sw, "_legs_for", lambda s: (lambda *a: calls.append("discord"), lambda *a: calls.append("phone")))
    settings = SimpleNamespace(storm_watch_latlon="1,2", database_url=None, storm_alert_events="")
    assert sw.start_storm_watch(settings, desk=object()) is not None
    built["deliver"]("t", "m", "i")
    assert calls == ["phone"] and built["reach"] is not None
    calls.clear()
    sw.start_storm_watch(settings)
    built["deliver"]("t", "m", "i")
    assert calls == ["discord", "phone"] and built["reach"] is None


# --- email ----------------------------------------------------------------------------------

def test_what_she_says_about_an_email_and_what_she_texts():
    ping = format_ping(EmailSummary(uid=3, sender="Chris Perry <chris@example.com>", subject="Dinner?",
                                    snippet="are we still on", date=""))
    said, text = split_ping(ping)
    assert said == "New email from Chris Perry: Dinner?"
    assert text == "From: Chris Perry <chris@example.com>\nare we still on"
    said, text = split_ping("12 new emails arrived while I wasn't watching. Most recent:\n• a — b")
    assert said.startswith("12 new emails") and text == "• a — b"


def email_settings():
    return SimpleNamespace(database_url="postgresql://x", owner_person_id="00000000-0000-0000-0000-000000000001")


def test_the_watcher_queues_each_mail_once_for_the_desk():
    store = MemoryReachStore()
    notify = reach_notify_for(email_settings(), store=store, clock=lambda: DAY)
    conn = conn_with_watermark(UV, 2)
    imap = FakeImap(messages={3: raw_email(subject="first"), 4: raw_email(subject="second")})
    stats = run_once(conn, lambda: imap, notify)
    assert stats["ok"] and stats["pinged"] == 2
    rows = sorted(store.rows.values(), key=lambda r: r.id)
    assert [r.dedupe_key for r in rows] == [f"email:INBOX:{UV}:3", f"email:INBOX:{UV}:4"]
    assert all(r.status == "queued" and r.topic == "email" and r.urgency == "routine" for r in rows)
    notify(format_ping(EmailSummary(uid=3, sender="x", subject="first", snippet="", date="")), key=f"INBOX:{UV}:3")
    assert len(store.rows) == 2                                # a re-ping of the same mail isn't queued twice


def test_a_ledger_that_cant_be_written_holds_the_watermark():
    class Down(MemoryReachStore):
        def add(self, *a, **kw):
            raise ConnectionError("NAS down")

    notify = reach_notify_for(email_settings(), store=Down(), clock=lambda: DAY)
    stats = run_once(conn_with_watermark(UV, 2), lambda: FakeImap(messages={3: raw_email()}), notify)
    assert not stats["ok"] and stats["pinged"] == 0


def test_the_desk_delivers_it_by_day():
    desk, world, _ = desk_at(DAY, World())
    reach_notify_for(email_settings(), store=desk.store, clock=lambda: DAY)(
        "New email from Chris Perry <chris@example.com>\nSubject: Dinner?\nare we still on", key="INBOX:1:3")
    assert desk.tick()["sent"] == 1
    assert world.spoken == [("New email from Chris Perry: Dinner?", "dev-office")]
    assert "chris@example.com" in world.texts[0][1]


def test_at_night_it_waits_and_rides_in_the_morning_brief():
    h = House(NIGHT)
    reach_notify_for(email_settings(), store=h.desk.store, clock=lambda: NIGHT)(
        "New email from Chris Perry <chris@example.com>\nSubject: Dinner?\nare we still on", key="INBOX:1:3")
    h.desk.tick()
    assert h.world.spoken == [] and h.desk.store.rows[1].not_before == datetime(2026, 11, 10, 13, 0, tzinfo=ET)
    h.at(IN_OFFICE)
    assert h.brief.tick() == "sent"
    assert [r.headline for r in h.composed[0].held] == ["New email from Chris Perry: Dinner?"]
    assert h.desk.store.rows[1].status == "folded" and "are we still on" in h.world.texts[0][1]


@pytest.mark.parametrize("on, has_owner, want", [(True, True, "desk"), (False, True, "dm"), (True, False, "dm")])
def test_the_switch_picks_the_path(monkeypatch, on, has_owner, want):
    import aerys_v2.factory as factory
    from aerys_v2.workers import __main__ as workers

    monkeypatch.setattr(factory, "discord_dm_notify_for", lambda s: "dm")
    monkeypatch.setattr("aerys_v2.workers.email_watch.reach_notify_for", lambda s: "desk" if has_owner else None)
    settings = SimpleNamespace(reach_email_via_desk=on)
    assert workers._email_notify_for(settings) == want


# --- Codex review 11/08: each of these failed on the first cut ---------------------------------

def test_a_warning_never_waits_on_a_busy_speaker_it_goes_as_text_now():
    desk, world, _ = desk_at(NIGHT)
    world.busy = True
    watcher(desk, [], alerts=ALERT).poll_alerts()
    assert world.spoken == [] and len(world.texts) == 1 and "Finish preparations." in world.texts[0][1]
    assert desk.store.rows[1].status == "sent"


def test_a_warning_whose_speaker_fails_goes_as_text_now():
    desk, world, _ = desk_at(NIGHT)
    world.fail = 99
    watcher(desk, [], alerts=ALERT).poll_alerts()
    assert len(world.texts) == 1 and desk.store.rows[1].status == "sent"


def test_if_the_house_cant_place_him_it_goes_as_text():
    desk, world, _ = desk_at(DAY)

    def broken():
        raise ConnectionError("presence down")

    desk.deliverer.locate = broken
    assert desk.reach_out(ReachEvent(source="kael_note", topic="kael", headline="Backup green.")) == "sent"
    assert world.spoken == [] and world.texts[0][1] == "Backup green."


def test_no_direct_dm_for_a_storm_the_desk_already_took():
    class MarkFailsOnce(MemoryReachStore):
        fail = 1

        def mark(self, *a, **kw):
            if self.fail:
                self.fail -= 1
                raise ConnectionError("synthetic")
            return super().mark(*a, **kw)

    desk, world, clock = desk_at(DAY, store=MarkFailsOnce())
    world.busy = True                                  # a watch: queued for a retry, whose record then fails
    dms = []
    watcher(desk, [], alerts=WATCH, fallback=lambda t, m, i: dms.append(t)).poll_alerts()
    assert dms == []                                   # the desk has it
    world.busy = False
    clock["now"] = DAY + timedelta(minutes=2)
    desk.tick()
    assert len(world.spoken) == 1 and dms == []


def test_an_email_is_never_queued_twice_whatever_became_of_the_first():
    store = MemoryReachStore()
    notify = reach_notify_for(email_settings(), store=store, clock=lambda: DAY)
    notify("New email from a <a@x.com>\nSubject: s\nbody", key="INBOX:1:3")
    store.rows[1].status = "folded"                    # a brief carried it
    notify("New email from a <a@x.com>\nSubject: s\nbody", key="INBOX:1:3")
    later = reach_notify_for(email_settings(), store=store, clock=lambda: DAY + timedelta(days=2))
    later("New email from a <a@x.com>\nSubject: s\nbody", key="INBOX:1:3")   # a retry two days on
    assert len(store.rows) == 1


def test_the_backlog_summary_keeps_its_key_when_more_mail_arrives_before_the_retry():
    from test_email_watch import Notify, _backlog_imap
    from aerys_v2.workers.email_watch import BACKLOG_COLLAPSE_AT

    first, second = Notify(), Notify()
    run_once(conn_with_watermark(UV, 2), lambda: _backlog_imap(BACKLOG_COLLAPSE_AT + 2), first)
    run_once(conn_with_watermark(UV, 2), lambda: _backlog_imap(BACKLOG_COLLAPSE_AT + 3), second)   # the save had failed
    assert first.keys == second.keys == [f"INBOX:{UV}:after:2"]


def test_a_bare_address_is_never_said_out_loud():
    said, text = split_ping("New email from sender@example.com\nSubject: hello\nhi")
    assert said == "New email from example.com: hello" and "sender@example.com" in text
    assert split_ping("New email from (unparseable sender)\nSubject: (unparseable message)\n")[0].startswith("New email")


# --- Codex recheck 11/08 -----------------------------------------------------------------------

def test_a_busy_speaker_warning_is_texted_once_even_when_that_text_fails():
    desk, world, _ = desk_at(NIGHT)
    world.busy = True
    attempts = []

    def failing_text(surface, body, silent):
        attempts.append(body)
        raise TimeoutError("discord timed out")

    desk.deliverer.text = failing_text
    watcher(desk, [], alerts=ALERT).poll_alerts()
    assert len(attempts) == 1 and desk.store.rows[1].status == "queued"   # the desk retries it, later


@pytest.mark.parametrize("alerts, direct", [(ALERT, True), (WATCH, False)])
def test_when_the_ledger_cant_say_only_a_warning_goes_direct(alerts, direct):
    class Dark:
        class store:
            @staticmethod
            def recently_sent(*a):
                raise ConnectionError("NAS down")

        def reach_out(self, event):
            raise ConnectionError("NAS down")

    dms, phone = [], []
    watcher(Dark(), phone, alerts=alerts, fallback=lambda t, m, i: dms.append(t)).poll_alerts()
    assert bool(dms) is direct and len(phone) == 1          # the phone leg went either way
