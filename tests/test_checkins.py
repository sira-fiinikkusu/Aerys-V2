"""Kael check-ins (owner ask 2026-10-12). The firing rule and most of these cases come
from Aerys's own draft (her workspace, the first Kael sitting, 26 tests), carried over
to the store the brain uses; the created-at and miss-reporting cases settle the two
leftovers from Kael's review of that draft."""

from datetime import date, datetime, timedelta, timezone
from types import SimpleNamespace

import httpx
import pytest
from pydantic import SecretStr

from aerys_v2.checkins import (
    EASTERN, TAP_PREFIX, Checkin, CheckinWatcher, MemoryCheckinStore, lane_sender, run_once,
    start_checkins,
)
from aerys_v2.tools.kael_checkins import build_kael_checkins_tool


def at(y, m, d, hh, mm, ss=0, fold=0):
    return datetime(y, m, d, hh, mm, ss, tzinfo=EASTERN, fold=fold)


class FakeSend:
    """Records taps. fail_times: how many calls fail first (raise or False)."""

    def __init__(self, fail_times=0, mode="raise"):
        self.sent, self.notes, self.fail_times, self.mode = [], [], fail_times, mode

    def __call__(self, item, day, *, text=None):
        if text is not None:
            self.notes.append(text)
            return True
        if self.fail_times > 0:
            self.fail_times -= 1
            if self.mode == "raise":
                raise RuntimeError("fake send failure")
            return False
        self.sent.append(item.message)
        return True


def store(*items, created=at(2026, 10, 1, 0, 0)):
    return MemoryCheckinStore([Checkin(name=n, days=d, at=t, message=m, created_at=created)
                               for n, d, t, m in items])


@pytest.fixture
def plants():
    # 2026-10-12 is a Monday.
    return store(("plant-watering", ["mon", "thu"], "08:47", "water the ferns"))


# --- firing (her draft) ---------------------------------------------------------

def test_fires_at_its_time(plants):
    send = FakeSend()
    r = run_once(plants, at(2026, 10, 12, 8, 47), send)
    assert r["fired"] == ["plant-watering"] and r["late"] == [] and send.sent == ["water the ferns"]


def test_not_before_its_time(plants):
    send = FakeSend()
    assert run_once(plants, at(2026, 10, 12, 8, 46, 59), send)["fired"] == [] and send.sent == []


def test_not_twice_the_same_day(plants):
    send = FakeSend()
    for t in (at(2026, 10, 12, 8, 47), at(2026, 10, 12, 8, 48), at(2026, 10, 12, 9, 30)):
        run_once(plants, t, send)
    assert len(send.sent) == 1


def test_not_on_other_days(plants):
    send = FakeSend()
    for day in (13, 14, 16, 17, 18):  # tue wed fri sat sun
        assert run_once(plants, at(2026, 10, day, 9, 0), send)["fired"] == []
    assert send.sent == []


def test_fires_again_on_the_next_listed_day(plants):
    send = FakeSend()
    run_once(plants, at(2026, 10, 12, 8, 47), send)
    run_once(plants, at(2026, 10, 15, 8, 47), send)  # Thursday
    assert len(send.sent) == 2


def test_a_stopped_one_never_fires(plants):
    assert plants.set_active("plant-watering", False)
    send = FakeSend()
    assert run_once(plants, at(2026, 10, 12, 8, 47), send)["fired"] == [] and send.sent == []


# --- late and retry (her draft) ---------------------------------------------------

def test_late_within_two_hours_fires_once(plants):
    send = FakeSend()
    r = run_once(plants, at(2026, 10, 12, 10, 30), send)
    assert r["fired"] == ["plant-watering"] and r["late"] == ["plant-watering"]
    assert run_once(plants, at(2026, 10, 12, 10, 31), send)["fired"] == [] and len(send.sent) == 1


def test_exactly_two_hours_still_fires(plants):
    assert run_once(plants, at(2026, 10, 12, 10, 47), FakeSend())["fired"] == ["plant-watering"]


def test_after_two_hours_it_skips_says_so_once_and_never_fires_that_day(plants):
    send = FakeSend()
    r = run_once(plants, at(2026, 10, 12, 10, 47, 1), send)
    assert r["fired"] == [] and send.sent == [] and [s["name"] for s in r["skipped"]] == ["plant-watering"]
    assert "skipped for 2026-10-12" in r["skipped"][0]["reason"]
    r2 = run_once(plants, at(2026, 10, 12, 12, 0), send)
    assert r2["skipped"] == [] and send.sent == []


def test_a_skipped_day_does_not_block_the_next_one(plants):
    send = FakeSend()
    run_once(plants, at(2026, 10, 12, 12, 0), send)
    run_once(plants, at(2026, 10, 15, 8, 47), send)
    assert len(send.sent) == 1


def test_a_send_that_raises_retries_next_tick(plants):
    send = FakeSend(fail_times=1, mode="raise")
    r = run_once(plants, at(2026, 10, 12, 8, 47), send)
    assert r["fired"] == [] and len(r["failed"]) == 1 and plants.items["plant-watering"].last_fired is None
    assert run_once(plants, at(2026, 10, 12, 8, 48), send)["fired"] == ["plant-watering"]


def test_a_send_that_returns_false_retries_next_tick(plants):
    send = FakeSend(fail_times=2, mode="false")
    run_once(plants, at(2026, 10, 12, 8, 47), send)
    run_once(plants, at(2026, 10, 12, 8, 48), send)
    assert run_once(plants, at(2026, 10, 12, 8, 49), send)["fired"] == ["plant-watering"]
    assert len(send.sent) == 1


def test_retries_stop_after_the_window(plants):
    send = FakeSend(fail_times=99, mode="raise")
    run_once(plants, at(2026, 10, 12, 8, 47), send)
    r = run_once(plants, at(2026, 10, 12, 11, 0), send)
    assert len(r["skipped"]) == 1 and send.sent == []


def test_a_naive_now_is_refused(plants):
    with pytest.raises(ValueError):
        run_once(plants, datetime(2026, 10, 12, 8, 47), FakeSend())


# --- midnight (her fix after Kael's first review) --------------------------------------

def test_caught_after_midnight_fires_late_once():
    s = store(("nightly", ["mon"], "23:30", "b"))
    send = FakeSend()
    r = run_once(s, at(2026, 10, 13, 0, 30), send)
    assert r["fired"] == ["nightly"] and r["late"] == ["nightly"]
    assert run_once(s, at(2026, 10, 13, 0, 31), send)["fired"] == [] and len(send.sent) == 1


def test_caught_after_midnight_past_the_window_skips_once():
    s = store(("nightly", ["mon"], "23:30", "b"))
    r = run_once(s, at(2026, 10, 13, 2, 0), FakeSend())
    assert [x["day"] for x in r["skipped"]] == ["2026-10-12"]
    assert run_once(s, at(2026, 10, 13, 2, 1), FakeSend())["skipped"] == []


def test_fired_on_time_is_not_repeated_after_midnight():
    s = store(("nightly", ["mon", "tue"], "23:30", "b"))
    send = FakeSend()
    run_once(s, at(2026, 10, 12, 23, 30), send)
    run_once(s, at(2026, 10, 13, 0, 30), send)
    assert len(send.sent) == 1


# --- DST (her draft) --------------------------------------------------------------

def test_spring_forward_a_time_that_does_not_exist_fires_at_0330_edt():
    s = store(("window-check", ["sun"], "02:30", "check the windows"), created=at(2026, 3, 1, 0, 0))
    send = FakeSend()
    assert run_once(s, at(2026, 3, 8, 1, 59), send)["fired"] == []
    assert run_once(s, at(2026, 3, 8, 3, 0), send)["fired"] == []
    assert run_once(s, at(2026, 3, 8, 3, 30), send)["fired"] == ["window-check"]
    run_once(s, at(2026, 3, 8, 4, 0), send)
    assert len(send.sent) == 1


def test_spring_forward_a_regular_time_after_the_gap_is_on_time():
    s = store(("sunday-sweep", ["sun"], "08:00", "sweep the porch"), created=at(2026, 3, 1, 0, 0))
    r = run_once(s, at(2026, 3, 8, 8, 0), FakeSend())
    assert r["fired"] == ["sunday-sweep"] and r["late"] == []


def test_fall_back_the_repeated_hour_fires_once():
    s = store(("lamp-check", ["sun"], "01:30", "check the lamps"))
    send = FakeSend()
    assert run_once(s, at(2026, 11, 1, 1, 30, fold=0), send)["fired"] == ["lamp-check"]
    for t in (at(2026, 11, 1, 1, 30, fold=1), at(2026, 11, 1, 1, 45, fold=1), at(2026, 11, 1, 3, 0)):
        run_once(s, t, send)
    assert len(send.sent) == 1


def test_fall_back_uses_real_elapsed_time():
    s = store(("lamp-check", ["sun"], "01:30", "check the lamps"))
    r = run_once(s, at(2026, 11, 1, 3, 45), FakeSend())   # 3 h 15 m after 01:30 EDT
    assert r["fired"] == [] and len(r["skipped"]) == 1


# --- the two leftovers from Kael's review --------------------------------------------

def test_a_new_check_in_never_reports_times_before_it_existed():
    s = store(("twice-weekly", ["mon", "thu"], "08:47", "a"), created=at(2026, 10, 12, 12, 0))
    r = run_once(s, at(2026, 10, 12, 12, 1), FakeSend())
    assert r == {"fired": [], "late": [], "skipped": [], "failed": []}


def test_a_miss_more_than_a_day_back_is_reported_once():
    s = store(("twice-weekly", ["mon", "thu"], "08:47", "a"))
    s.items["twice-weekly"].last_fired = date(2026, 10, 8)            # Thu: on time
    missed = []
    r = run_once(s, at(2026, 10, 14, 12, 0), FakeSend(), missed=lambda item, day: missed.append(day))
    assert [x["day"] for x in r["skipped"]] == ["2026-10-12"] and missed == [date(2026, 10, 12)]
    r2 = run_once(s, at(2026, 10, 14, 12, 5), FakeSend(), missed=lambda item, day: missed.append(day))
    assert r2["skipped"] == [] and len(missed) == 1


def test_a_long_outage_reports_only_the_newest_miss():
    s = store(("twice-weekly", ["mon", "thu"], "08:47", "a"))
    s.items["twice-weekly"].last_fired = date(2026, 10, 1)            # then nothing for two weeks
    r = run_once(s, at(2026, 10, 14, 12, 0), FakeSend())
    assert [x["day"] for x in r["skipped"]] == ["2026-10-12"]


# --- the lane, the watcher, the tool ---------------------------------------------

def _lane(status):
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(status)
    return lane_sender("http://desk/aerys/message", "t0k", client=httpx.Client(transport=httpx.MockTransport(handler))), seen


def test_the_tap_is_labelled_and_202_is_the_only_delivery():
    send, seen = _lane(202)
    item = Checkin(name="twice-weekly", days=["mon"], at="08:47", message="run  the\nthing")
    assert send(item, date(2026, 10, 12)) is True
    body = __import__("json").loads(seen[0].content)
    assert body["message"] == f"{TAP_PREFIX} twice-weekly [2026-10-12]: run the thing"
    assert body["thread_id"] == "checkin:twice-weekly:2026-10-12", "the receiver dedupes on (name, day)"
    assert seen[0].headers["authorization"] == "Bearer t0k"
    for status in (429, 500):
        assert _lane(status)[0](item, date(2026, 10, 12)) is False


def test_a_line_that_is_down_is_a_retry_not_a_crash():
    def handler(request):
        raise httpx.ConnectError("down")
    send = lane_sender("http://desk/x", "t", client=httpx.Client(transport=httpx.MockTransport(handler)))
    assert send(Checkin(name="n", days=["mon"], at="08:47", message="m"), date(2026, 10, 12)) is False


def test_the_watcher_tells_kael_about_a_missed_day_labelled_as_missed(plants):
    send = FakeSend()
    CheckinWatcher(plants, send, clock=lambda: at(2026, 10, 12, 12, 0)).tick()
    assert send.sent == [] and len(send.notes) == 1
    assert send.notes[0].startswith(f"{TAP_PREFIX} plant-watering [2026-10-12]: MISSED")
    assert "Nothing was started." in send.notes[0]


def test_it_arms_only_with_the_database_and_kaels_line():
    for missing in ({"database_url": None}, {"kael_desk_url": None}, {"kael_desk_token": None}):
        settings = SimpleNamespace(database_url="postgresql://x", kael_desk_url="http://desk",
                                   kael_desk_token=SecretStr("t"))
        for key, value in missing.items():
            setattr(settings, key, value)
        assert start_checkins(settings) is None


def test_her_tool_lists_stops_and_resumes(plants):
    tool = build_kael_checkins_tool(plants)
    listed = tool.invoke({"action": "list"})
    assert "plant-watering: mon/thu at 08:47 (Eastern), on; last tapped Kael: never" in listed
    assert tool.invoke({"action": "stop", "name": "plant-watering"}).startswith("Stopped plant-watering")
    assert plants.items["plant-watering"].active is False and "OFF" in tool.invoke({"action": "list"})
    assert tool.invoke({"action": "resume", "name": "plant-watering"}).startswith("Resumed plant-watering")
    assert plants.items["plant-watering"].active is True


@pytest.mark.parametrize("args,said", [
    ({"action": "stop"}, "needs the check-in's name"),
    ({"action": "stop", "name": "nope"}, "no check-in named 'nope'"),
    ({"action": "add", "name": "x"}, 'action must be "list", "stop" or "resume"'),
])
def test_her_tool_says_plainly_what_it_did_not_do(plants, args, said):
    out = build_kael_checkins_tool(plants).invoke(args)
    assert out.startswith("NOT DONE") and said in out


def test_a_store_that_is_down_changes_nothing_and_says_so():
    class Down:
        def list(self):
            raise OSError("db down")

        def set_active(self, name, active):
            raise OSError("db down")
    tool = build_kael_checkins_tool(Down())
    assert tool.invoke({"action": "list"}).startswith("NOT DONE: the check-ins store couldn't be reached")


def test_the_migration_matches_the_store_columns():
    from pathlib import Path

    from aerys_v2.checkins import PostgresCheckinStore
    sql = (Path(__file__).resolve().parents[1] / "db/migrations/014_kael_checkins.sql").read_text()
    for column in PostgresCheckinStore.COLUMNS.split(", "):
        assert f"    {column} " in sql or f"    {column}\n" in sql, column


# --- Codex review (2026-10-12) ----------------------------------------------------

def test_created_after_the_spring_gap_still_fires_at_0330():
    # Created at 03:00 EDT on the spring-forward Sunday; 02:30 reads as 03:30 EDT, after creation.
    s = store(("window-check", ["sun"], "02:30", "w"), created=at(2026, 3, 8, 3, 0))
    assert run_once(s, at(2026, 3, 8, 3, 30), FakeSend())["fired"] == ["window-check"]


def test_created_in_the_repeated_hour_never_fires_for_its_first_pass():
    # Created at 01:15 EST (the second 01:xx); the 01:30 occurrence is the EDT one, before it.
    s = store(("lamp-check", ["sun"], "01:30", "l"), created=at(2026, 11, 1, 1, 15, fold=1))
    send = FakeSend()
    for t in (at(2026, 11, 1, 1, 30, fold=1), at(2026, 11, 1, 2, 0), at(2026, 11, 1, 4, 0)):
        r = run_once(s, t, send)
        assert r["fired"] == [] and r["skipped"] == []
    assert send.sent == []


def test_a_record_that_fails_never_means_a_second_tap(plants):
    class FlakyRecord(MemoryCheckinStore):
        fails = 2

        def mark_fired(self, name, day):
            if self.fails:
                self.fails -= 1
                raise OSError("db down")
            super().mark_fired(name, day)
    s = FlakyRecord(plants.list())
    send, delivered = FakeSend(), set()
    for minute in (47, 48, 49, 50):
        run_once(s, at(2026, 10, 12, 8, minute), send, delivered=delivered)
    assert send.sent == ["water the ferns"], "one tap, though the record failed twice"
    assert s.items["plant-watering"].last_fired == date(2026, 10, 12), "the record was retried and landed"
    assert delivered == set(), "once recorded, the watcher stops holding it (bounded memory)"


def test_a_timed_out_answer_is_retried_and_the_tap_names_its_day_for_the_receiver(plants):
    # The line may have accepted it; a lost answer reads as not delivered, so it is sent
    # again (never lost). Both taps carry the same (name, day): Kael's run-book dedupes.
    texts = []

    def handler(request):
        texts.append(__import__("json").loads(request.content)["message"])
        if len(texts) == 1:
            raise httpx.ReadTimeout("answer lost")
        return httpx.Response(202)
    send = lane_sender("http://desk/x", "t", client=httpx.Client(transport=httpx.MockTransport(handler)))
    run_once(plants, at(2026, 10, 12, 8, 47), send)
    run_once(plants, at(2026, 10, 12, 8, 48), send)
    assert len(texts) == 2 and texts[0] == texts[1] and "[2026-10-12]" in texts[0]
    assert run_once(plants, at(2026, 10, 12, 8, 49), send)["fired"] == [] and len(texts) == 2


def test_what_fell_due_while_it_was_off_is_not_fired_on_resume(plants, monkeypatch):
    plants.set_active("plant-watering", False)                       # off before 08:47
    import aerys_v2.checkins as checkins
    real = datetime

    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return at(2026, 10, 12, 9, 0).astimezone(tz) if tz else at(2026, 10, 12, 9, 0)
    monkeypatch.setattr(checkins, "datetime", Clock)
    plants.set_active("plant-watering", True)                        # resumed at 09:00
    monkeypatch.setattr(checkins, "datetime", real)
    send = FakeSend()
    for t in (at(2026, 10, 12, 9, 1), at(2026, 10, 12, 11, 0)):
        r = run_once(plants, t, send)
        assert r["fired"] == [] and r["skipped"] == []
    assert run_once(plants, at(2026, 10, 15, 8, 47), send)["fired"] == ["plant-watering"]


def test_resuming_one_that_was_on_moves_nothing(plants):
    before = plants.items["plant-watering"].resumed_at
    assert plants.set_active("plant-watering", True)
    assert plants.items["plant-watering"].resumed_at == before
