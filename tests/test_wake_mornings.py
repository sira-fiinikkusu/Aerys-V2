"""D74 (the owner, 2026-11-19, board #177): his mornings follow his phone.

  call 1 "i like this": AWAKE = what she held overnight arrives as quiet text, nothing spoken;
         UP (out of the bedroom after waking) = her voice comes back and the morning brief plays.
  call 2 "B works for me": quiet hours end when he's up, 09:00 the fallback when the phone gives
         nothing; 21:00 stays the start.
  naps "your nap bit is good too": daytime sleep holds her like a nap; no morning routine after one.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from aerys_v2.brief import Facts, MorningBrief, Reading, plain_brief
from aerys_v2.reach import (OVERNIGHT_SOURCE, Deliverer, MemoryReachStore, Prefs, ReachDesk, ReachEvent, Where,
                            told_quietly)
from aerys_v2.wake import (Morning, WakeWatch, night_minutes, quiet_now, read_morning, up_time, wake_time,
                           wake_watch_for)
from test_reach_desk import World, ev

ET = timezone(timedelta(hours=-5))                       # EST on these dates
EVE = datetime(2026, 11, 18, 22, 30, tzinfo=ET)          # Wednesday night, in bed
NIGHT = datetime(2026, 11, 19, 1, 0, tzinfo=ET)
WOKE = datetime(2026, 11, 19, 7, 2, tzinfo=ET)           # the phone's wake (Thursday)
UP = datetime(2026, 11, 19, 7, 24, tzinfo=ET)            # out of the bedroom
BEDROOM = Where("room", "dev-bed", "bedroom")
OFFICE = Where("room", "dev-office", "office")


def at(h, m=0, day=19):
    return datetime(2026, 11, day, h, m, tzinfo=ET)


def fixed(t):
    hour = t.astimezone(ET).hour
    return hour >= 21 or hour < 9


# --- the wake rule (pure) ----------------------------------------------------------------

CONF = [(at(22, 0, 18), "12"), (at(23, 10, 18), "91"), (at(6, 50), "95"), (at(7, 5), "20")]
SEG = [(at(23, 0, 18), "unknown"), (WOKE, "28200000"), (at(13, 30), "3600000")]


def test_the_night_wake_is_the_first_morning_segment_after_a_night_block():
    assert wake_time(CONF, SEG, at(0).date(), at(12)) == WOKE
    assert wake_time(CONF, SEG, at(0).date(), at(6, 59)) is None            # not yet
    short = [(at(4, 0), "90"), (at(5, 30), "10")]                            # 90 minutes isn't a night
    assert wake_time(short, SEG, at(0).date(), at(12)) is None


def test_a_daytime_nap_never_makes_a_wake():
    nap = [(at(13, 0), "92"), (at(15, 0), "10")]
    assert wake_time(nap, [(at(15, 1), "7200000")], at(0).date(), at(16)) is None


def test_up_is_the_first_change_out_of_the_bedroom_after_the_wake():
    room = [(at(21, 0, 18), "Bedroom"), (at(5, 10), "Office"), (at(5, 20), "Bedroom"), (UP, "Kitchen")]
    assert up_time(room, WOKE, at(12), bedroom="bedroom") == UP              # the 05:10 trip was before the wake
    assert up_time(room, WOKE, at(7, 20), bedroom="bedroom") is None
    # Codex 11/19 #3: a stale 'office' from last night proves nothing; only a change after the wake does.
    assert up_time([(at(6, 0), "Office")], WOKE, at(12), bedroom="bedroom") is None
    assert up_time([(UP, "unavailable")], WOKE, at(12), bedroom="bedroom") is None    # unknown isn't up


def test_unreadable_is_no_evidence_and_no_evidence_is_the_fixed_hours():
    """Codex 11/19 #3: missing or dark room history, or a stale non-bedroom room, is no wake."""
    day, bed = at(0).date(), [(at(21, 0, 18), "Bedroom")]
    assert read_morning(CONF, SEG, bed, day, at(9, 30), bedroom="bedroom") == Morning(WOKE, None)    # valid
    assert read_morning(CONF, SEG, [], day, at(9, 30), bedroom="bedroom") == Morning()              # no history
    stale = [(at(20, 0, 18), "Office")]
    assert read_morning(CONF, SEG, stale, day, at(9, 30), bedroom="bedroom") == Morning()           # not in bed
    dark = bed + [(at(7, 10), "unavailable")]
    assert read_morning(CONF, SEG, dark, day, at(9, 30), bedroom="bedroom") == Morning()            # gone dark
    assert read_morning(CONF, SEG, bed + [(UP, "Office")], day, at(9, 30), bedroom="bedroom") == Morning(WOKE, UP)
    assert not quiet_now(at(9, 30), 21, 9, Morning())


def test_a_reading_that_is_not_a_number_breaks_the_night_block():
    gap = [(at(22, 0, 18), "95"), (at(23, 0, 18), "unavailable"), (at(6, 0), "95"), (at(7, 0), "10")]
    assert night_minutes(gap, at(21, 0, 18), at(7, 0)) == 60 + 60    # 22-23 and 06-07, not the dark 7 hours


def test_quiet_hours_end_when_he_is_up_and_09_00_is_the_fallback():
    m = Morning(WOKE, UP)
    assert quiet_now(at(7, 10), 21, 9, m) and not quiet_now(at(7, 30), 21, 9, m)      # ends at up, not 9
    assert quiet_now(at(8, 59), 21, 9, Morning()) and not quiet_now(at(9, 0), 21, 9, Morning())   # no wake
    assert quiet_now(at(9, 30), 21, 9, Morning(WOKE, None))      # awake in bed past 9: still nothing aloud
    assert quiet_now(at(15, 0), 21, 9, Morning(WOKE, None))      # ...until he's up, no afternoon cutoff (#4)
    assert quiet_now(at(3, 59), 21, 9, Morning(at(4, 0), at(4, 0)))  # never ends before 04:00
    assert quiet_now(at(21, 0), 21, 9, m) and quiet_now(EVE, 21, 9, m)                # the start stays
    assert not quiet_now(at(7, 0), 9, 9, m)                                           # no quiet hours set


# --- the desk ----------------------------------------------------------------------------

class Phone:
    def __init__(self, woke=None, up=None, asleep=False):
        self.woke, self.up, self.asleep_now = woke, up, asleep

    def morning(self, now):
        return Morning(self.woke if self.woke and self.woke <= now else None,
                       self.up if self.up and self.up <= now else None)

    def quiet(self, now):
        return quiet_now(now, 21, 9, self.morning(now))

    def awake(self, now):
        m = self.morning(now)
        return m.woke_at is not None and m.up_at is None

    def asleep(self, now):
        if self.asleep_now == "boom":
            raise ConnectionError("phone unreadable")
        return self.asleep_now


def desk_with(phone, world, *, now, store=None, morning_hour=9):
    clock = {"now": now}
    desk = ReachDesk(store or MemoryReachStore(), world.deliverer(), budget_default=3, quiet=phone.quiet,
                     morning_hour=morning_hour, replied_since=lambda when: world.replied,
                     said=lambda row, body: world.said.append(body), clock=lambda: clock["now"],
                     awake=phone.awake, asleep=phone.asleep, fixed_morning_hour=9)
    return desk, clock


def test_awake_what_the_night_held_arrives_as_one_quiet_text_and_nothing_is_spoken():
    world, phone = World(BEDROOM), Phone(WOKE, UP)
    desk, clock = desk_with(phone, world, now=EVE)
    assert desk.reach_out(ev(headline="Kael finished the backup.", detail="All three.")) == "queued"
    clock["now"] = NIGHT
    assert desk.reach_out(ev(headline="Your package shipped.", topic="mail")) == "queued"
    clock["now"] = at(7, 0)
    desk.tick()
    assert world.texts == [] and world.spoken == []                  # still asleep: nothing
    clock["now"] = at(7, 5)
    desk.tick()
    assert world.spoken == []
    assert len(world.texts) == 1
    surface, body, silent = world.texts[0]
    assert silent and body.startswith("While you slept, 2 things came in:")
    assert "Kael finished the backup." in body and "All three." in body and "Your package shipped." in body
    note = next(r for r in desk.store.rows.values() if r.source == OVERNIGHT_SOURCE)
    assert note.status == "sent" and note.text_only and note.brief
    assert {r.status for r in desk.store.rows.values() if r.source != OVERNIGHT_SOURCE} == {"folded"}
    assert desk.store.sent_since(at(0)) == 0                         # the night's items don't eat the day
    desk.tick()
    clock["now"] = at(7, 10)
    desk.tick()
    assert len(world.texts) == 1                                     # once


def test_awake_something_new_goes_as_quiet_text_and_up_brings_her_voice_back():
    world, phone = World(BEDROOM), Phone(WOKE, UP)
    desk, clock = desk_with(phone, world, now=at(7, 10))
    assert desk.reach_out(ev(headline="The dryer's done.", topic="house")) == "sent"
    assert world.spoken == [] and world.texts[-1] == ("discord", "The dryer's done.", True)
    clock["now"] = at(7, 30)                                         # up, before 9: her voice is back
    world.where = OFFICE
    assert desk.reach_out(ev(headline="Kael says hi.")) == "sent"
    assert world.spoken == [("Kael says hi.", "dev-office")]


def test_a_message_relayed_overnight_goes_on_its_own_as_quiet_text_at_his_wake():
    world, phone = World(BEDROOM), Phone(WOKE, UP)
    desk, clock = desk_with(phone, world, now=NIGHT)
    assert desk.reach_out(ReachEvent(source="relay", topic="messages", headline="Your friend says goodnight.",
                                     dedupe_key="relay:1")) == "queued"
    clock["now"] = at(7, 5)
    desk.tick()
    assert world.spoken == [] and ("discord", "Your friend says goodnight.", True) in world.texts
    assert not any(r.source == OVERNIGHT_SOURCE for r in desk.store.rows.values())   # nothing to fold


def test_no_wake_from_the_phone_keeps_the_09_00_fallback():
    world, phone = World(OFFICE), Phone()
    desk, clock = desk_with(phone, world, now=NIGHT)
    assert desk.reach_out(ev()) == "queued"
    clock["now"] = at(8, 59)
    assert desk.tick()["sent"] == 0 and world.texts == []
    clock["now"] = at(9, 1)
    assert desk.tick()["sent"] == 1 and len(world.spoken) == 1


def test_an_unreadable_phone_falls_back_to_the_fixed_hours_and_holds_nothing():
    calls = []

    def get(path):
        calls.append(path)
        raise ConnectionError("HA down")

    watch = WakeWatch(get, confidence="sensor.conf", segment="sensor.seg", room="sensor.room",
                      clock=lambda: at(7, 30))
    assert watch.morning() == Morning()
    q = watch.quiet(21, 9)
    assert q(at(7, 30)) and not q(at(9, 5))
    assert watch.asleep(at(14, 0)) is False
    world = World(OFFICE)
    desk, clock = desk_with(Phone(asleep="boom"), world, now=at(14, 0))
    assert desk.reach_out(ev()) == "sent" and world.spoken                # a phone that can't be read holds nothing


def test_a_nap_holds_her_like_a_named_nap_and_no_morning_routine_follows_it():
    world, phone = World(BEDROOM), Phone(WOKE, UP, asleep=True)
    desk, clock = desk_with(phone, world, now=at(14, 0))
    assert desk.voice_held() == "asleep"
    assert desk.reach_out(ev(headline="news")) == "queued" and world.spoken == []
    assert "asleep" in desk.store.rows[1].reason
    assert desk.reach_out(ev(urgency="timely", headline="the oven timer")) == "sent"
    assert world.texts[-1] == ("discord", "the oven timer", True)
    # Owed, during the phone's nap: quiet text now, never waiting for him to wake (Codex 11/19 #7).
    assert desk.reach_out(ev(owed=True, headline="Kael answered.")) == "sent" and world.spoken == []
    assert world.texts[-1] == ("discord", "Kael answered.", True)
    assert desk.release_overnight() == "not awake"                   # waking from a nap is no morning
    phone.asleep_now = False
    clock["now"] = at(15, 10)
    desk.tick()
    assert world.spoken == [("news", "dev-bed")]
    assert not any(r.source == OVERNIGHT_SOURCE for r in desk.store.rows.values())


def test_over_the_budget_while_awake_it_waits_for_the_morning_as_the_night_did_and_owed_still_goes():
    world, phone = World(BEDROOM), Phone(WOKE, UP)
    desk, clock = desk_with(phone, world, now=at(7, 10))
    for i in range(3):
        assert desk.reach_out(ev(headline=f"item {i}", dedupe_key=f"k{i}")) == "sent"
    assert desk.reach_out(ev(headline="one more", dedupe_key="k9")) == "queued"       # no new drops (#8)
    row = max(desk.store.rows.values(), key=lambda r: r.id)
    assert row.reason == "night: waits for the morning" and row.not_before == at(9, 0)
    assert desk.reach_out(ev(owed=True, headline="Kael's answer.", dedupe_key="ka")) == "sent"
    assert world.spoken == [] and all(silent for *_, silent in world.texts)


def test_a_muted_overnight_note_lets_the_items_go_on_their_own_later():
    world, phone = World(OFFICE), Phone(WOKE, UP)
    store = MemoryReachStore()
    desk, clock = desk_with(phone, world, now=NIGHT, store=store, morning_hour=13)   # a brief armed
    store._prefs = Prefs(muted=["overnight"])
    assert desk.reach_out(ev(headline="Backup done.")) == "queued"
    clock["now"] = at(7, 5)
    desk.tick()
    assert world.texts == []                                         # the note was muted...
    row = store.rows[1]                                              # ...its hold lifted (#6): 9, not 13
    assert row.not_before == at(9, 0) and not row.reason.startswith("carried by")
    clock["now"] = at(9, 1)
    desk.tick()
    assert world.spoken == [("Backup done.", "dev-office")]          # ...the item still reached him


def test_a_late_wake_never_holds_the_night_until_tomorrow():
    world, phone, store = World(OFFICE), Phone(at(9, 40), None), MemoryReachStore()
    store._prefs = Prefs(muted=["overnight"])                        # the note won't go
    desk, clock = desk_with(phone, world, now=NIGHT, store=store)
    desk.reach_out(ev(headline="Backup done."))
    clock["now"] = at(9, 45)                                         # past 9, awake in bed: still quiet
    desk.tick()
    row = store.rows[1]
    assert row.status == "queued" and row.not_before <= at(10, 45)


def test_a_restart_mid_morning_never_sends_the_overnight_note_twice():
    world, phone, store = World(BEDROOM), Phone(WOKE, UP), MemoryReachStore()
    desk, clock = desk_with(phone, world, now=NIGHT, store=store)
    desk.reach_out(ev(headline="Backup done."))
    clock["now"] = at(7, 5)
    desk.tick()
    assert len(world.texts) == 1
    again, clock2 = desk_with(phone, world, now=at(7, 15), store=store)   # a new process, same record
    assert again.release_overnight() == "done"
    again.tick()
    assert len(world.texts) == 1


# --- the brief ---------------------------------------------------------------------------

def test_the_brief_counts_what_the_overnight_note_carried():
    world, phone = World(OFFICE), Phone(WOKE, UP)
    store = MemoryReachStore()
    desk, clock = desk_with(phone, world, now=NIGHT, store=store, morning_hour=13)
    desk.reach_out(ev(headline="Backup done."))
    desk.reach_out(ev(headline="Package shipped.", topic="mail"))
    clock["now"] = at(7, 5)
    desk.tick()                                                      # the note: 2 things, quietly
    seen = []
    brief = MorningBrief(desk, read=lambda: Reading(room="office", confirmed=True, woke_at=WOKE),
                         gather=lambda now, held, quiet: Facts(held=held, quiet=quiet),
                         compose=lambda facts: seen.append(facts) or "", clock=lambda: clock["now"])
    clock["now"] = at(7, 40)                                         # up, in the office
    assert brief.tick() == "sent"
    facts = seen[0]
    assert [r.source for r in facts.quiet] == [OVERNIGHT_SOURCE] and facts.held == []   # carried once
    assert "2 things came in overnight" in plain_brief(facts)
    assert world.spoken and world.spoken[-1][1] == "dev-office"


def test_a_spoken_brief_waits_while_he_is_awake_in_bed_and_plays_once_he_is_up():
    """Codex 11/19 #5: a brief configured for the bedroom, his wake seen by the house: nothing aloud
    until he's up, asked again at the moment of speech."""
    world, phone = World(BEDROOM), Phone(WOKE, None)
    desk, clock = desk_with(phone, world, now=at(7, 10), morning_hour=13)
    brief = MorningBrief(desk, read=lambda: Reading(room="bedroom", confirmed=True, woke_at=WOKE),
                         gather=lambda now, held, quiet: Facts(held=held, quiet=quiet),
                         compose=lambda facts: "Morning!", room="bedroom", clock=lambda: clock["now"])
    assert brief.tick() == "held: awake" and world.spoken == []
    # Straight at the desk too: decided before he was up, it's never spoken while he's awake.
    assert desk.reach_out(ReachEvent(source="brief", topic="brief", headline="Morning!", room="bedroom",
                                     brief=True, dedupe_key="b")) == "queued"
    assert world.spoken == []
    phone.up = at(7, 30)
    clock["now"] = at(7, 31)
    desk.tick()
    assert world.spoken == [("Morning!", "dev-bed")]


def test_a_reservation_is_exclusive_each_item_is_carried_once():
    """Codex 11/19 #1: the overnight note holds what it carries while it retries; the brief or the
    welcome-home hello can't take the same item."""
    phone, store, texts = Phone(WOKE, UP), MemoryReachStore(), []
    fail = {"n": 1}

    def text(surface, body, silent):
        if fail["n"]:
            fail["n"] -= 1
            raise ConnectionError("discord down")
        texts.append(body)

    clock = {"now": NIGHT}
    desk = ReachDesk(store, Deliverer(locate=lambda: OFFICE, speak=lambda t, d: None, text=text), quiet=phone.quiet,
                     morning_hour=13, clock=lambda: clock["now"], awake=phone.awake, asleep=phone.asleep,
                     fixed_morning_hour=9)
    desk.reach_out(ev(headline="Backup done."))
    clock["now"] = at(7, 5)
    assert desk.release_overnight() == "queued"                      # the note couldn't go yet
    from datetime import datetime as _dt

    since = _dt(2026, 11, 18, 18, 0, tzinfo=ET)
    assert desk.reserve_for_brief(since, at(13, 0)) == []            # the brief can't take it meanwhile
    clock["now"] = at(7, 20)                                         # the retry lands
    desk.tick()
    desk.tick()
    assert len(texts) == 1 and "Backup done." in texts[0]
    assert store.rows[1].status == "folded"
    clock["now"] = at(9, 10)                                         # a lapsed hold is fair game again
    assert desk.reserve_for_brief(since, at(13, 0)) == []            # (it's folded: nothing left to take)


def test_a_lapsed_reservation_is_fair_game():
    store = MemoryReachStore()
    from aerys_v2.reach import Reach

    store.rows[1] = Reach(id=1, source="kael_note", topic="kael", urgency="routine", owed=False, headline="x",
                          status="queued", not_before=at(7, 0), reason="carried by the morning brief",
                          created_at=NIGHT)
    assert [r.id for r in store.foldable(at(0), at(13), at(7, 30))] == [1]          # lapsed at 7
    store.rows[1].not_before = at(13, 0)
    assert store.foldable(at(0), at(13), at(7, 30)) == []                           # held by another
    assert [r.id for r in store.foldable(at(0), at(13))] == [1]                     # no `now`: as before


def test_an_email_sent_quietly_stays_out_of_the_brief_but_the_overnight_note_counts():
    rows = [type("R", (), {"text_only": True, "source": "email"})(),
            type("R", (), {"text_only": True, "source": OVERNIGHT_SOURCE})(),
            type("R", (), {"text_only": False, "source": "kael_note"})()]
    assert [r.source for r in told_quietly(rows)] == [OVERNIGHT_SOURCE, "kael_note"]


# --- reading his phone from HA ------------------------------------------------------------

def _hist(entity, pts):
    return [{"entity_id": entity, "state": pts[0][1], "last_changed": pts[0][0].isoformat()}] + \
        [{"state": s, "last_changed": t.isoformat()} for t, s in pts[1:]]


def test_the_watch_reads_his_morning_from_history_and_a_restart_reaches_the_same_answer():
    room = [(at(21, 0, 18), "Bedroom"), (UP, "Office")]
    paths = []

    def get(path):
        paths.append(path)
        return [_hist("sensor.conf", CONF), _hist("sensor.seg", SEG), _hist("sensor.room", room)]

    for _ in range(2):                                               # two processes, same history
        watch = WakeWatch(get, confidence="sensor.conf", segment="sensor.seg", room="sensor.room")
        assert watch.morning(at(7, 30)) == Morning(WOKE.astimezone(timezone.utc), UP.astimezone(timezone.utc))
        assert watch.quiet(21, 9)(at(7, 30)) is False
        assert watch.awake(at(7, 10)) is True and watch.awake(at(7, 30)) is False
    n = len(paths)
    watch.morning(at(7, 30))
    assert len(paths) == n                                           # cached within the minute
    assert "filter_entity_id=sensor.conf,sensor.seg,sensor.room" in paths[0]
    assert watch.morning(at(15, 0)) == Morning(WOKE.astimezone(timezone.utc), UP.astimezone(timezone.utc))
    assert watch.morning(at(21, 30)) == Morning()                    # the evening's quiet hours: nothing to look for


def test_the_watch_holds_for_steady_daytime_sleep_only():
    now = at(14, 30)

    def get_for(state, changed, recent):
        def get(path):
            if path.startswith("/api/states/"):
                return {"state": state, "last_changed": changed.isoformat(), "last_reported": now.isoformat()}
            return [_hist("sensor.conf", recent)]
        return get

    steady = get_for("94", at(14, 0), [(at(14, 0), "94")])
    assert WakeWatch(steady, confidence="sensor.conf", segment="s", room="r").asleep(now) is True
    just_down = get_for("90", at(14, 25), [(at(14, 10), "30"), (at(14, 25), "90")])
    assert WakeWatch(just_down, confidence="sensor.conf", segment="s", room="r").asleep(now) is False
    awake = get_for("12", at(14, 0), [(at(14, 0), "12")])
    assert WakeWatch(awake, confidence="sensor.conf", segment="s", room="r").asleep(now) is False

    def stale(path):
        return {"state": "95", "last_changed": at(11, 0).isoformat(), "last_reported": at(11, 0).isoformat()}

    assert WakeWatch(stale, confidence="sensor.conf", segment="s", room="r").asleep(now) is False


def test_unarmed_unless_both_phone_sensors_are_named():
    from types import SimpleNamespace

    from pydantic import SecretStr

    base = dict(presence_settled_entity="sensor.room", ha_token=SecretStr("t"), ha_base_url="http://ha")
    assert wake_watch_for(SimpleNamespace(**base)) is None
    assert wake_watch_for(SimpleNamespace(**base, wake_sleep_confidence_entity="sensor.c")) is None
    armed = wake_watch_for(SimpleNamespace(**base, wake_sleep_confidence_entity="sensor.c",
                                           wake_sleep_segment_entity="sensor.s"), get=lambda p: [])
    assert armed is not None and armed.room == "sensor.room"
    assert wake_watch_for(SimpleNamespace(presence_settled_entity="sensor.room", ha_token=None,
                                          wake_sleep_confidence_entity="sensor.c",
                                          wake_sleep_segment_entity="sensor.s")) is None


def test_owner_reach_uses_the_same_wake_aware_hours_when_the_phone_is_wired():
    """Codex 11/19 #9: the desk-less path to him follows his mornings and naps too."""
    from types import SimpleNamespace

    from pydantic import SecretStr

    from aerys_v2.kael_line import owner_reach_for

    sent = []
    settings = SimpleNamespace(owner_person_id="p", telegram_bot_token=None, discord_bot_token=SecretStr("t"),
                               owner_quiet_start_hour=21, owner_quiet_end_hour=9)
    phone = Phone(WOKE, None)
    watch = SimpleNamespace(quiet=lambda s, e: phone.quiet, asleep=phone.asleep)
    clock = {"now": at(9, 30)}                                       # past 9, awake in bed
    reach = owner_reach_for(settings, send=lambda surface, cid, text, quiet: sent.append((surface, quiet)),
                            lookup_surface=lambda s: ("discord", None), lookup_chat_id=lambda s, surf: "c",
                            clock=lambda: clock["now"], hold_fn=lambda: None, wake_watch=watch)
    assert reach("hi") and sent[-1] == ("discord", True)            # quiet: no buzz, nothing spoken
    phone.up = at(9, 40)
    clock["now"] = at(9, 45)
    assert reach("hi again") and sent[-1] == ("discord", False)
