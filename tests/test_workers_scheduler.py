"""Offline tests for the worker loop-mode scheduler — no DB, no models, no network.

The regression these lock down: the loop-mode ``scheduler.add_job(...)`` calls in
``aerys_v2.workers.__main__`` passed ``next_run_time=None``, which APScheduler v3
treats as "add the job PAUSED" (see BaseScheduler.add_job: "pass ``None`` to add
the job as paused"). A paused interval job never fires — the worker ran ONE
startup pass and then sat idle for 22h, silently killing memory extraction and
the gaps-miner cadence.

Two levels of proof, all fakes-and-seams (same style as test_cli_telegram_runner):

  - the ``_add_interval_job`` seam lands a real job SCHEDULED (next_run_time set to
    ~now+interval), never paused — with a contrast test that pins the exact v3
    footgun the fix removes;
  - the extraction and gaps-mine loop branches drive the REAL worker code with a
    fake scheduler and assert add_job is invoked once, on the configured interval,
    with NO next_run_time kwarg, plus the one immediate T0 pass before start().

Real wall-clock firing is deliberately NOT exercised (that would be flaky and
slow); next_run_time-is-not-None is the honest, deterministic proxy for "the
interval is live". Coverage gap noted: these prove the job is scheduled and the
immediate pass runs, not that N passes actually execute over N intervals.
"""

from datetime import datetime, timezone
from types import SimpleNamespace

import apscheduler.schedulers.blocking as apsblocking
from apscheduler.schedulers.background import BackgroundScheduler

import aerys_v2.factory as factory
import aerys_v2.workers.__main__ as workers_main


def _next_run_time(scheduler, job):
    """Read a pending job's computed next_run_time via a paused start.

    An UNSTARTED scheduler hasn't processed pending jobs yet (the attribute
    isn't even set). start(paused=True) runs _real_add_job — computing
    next_run_time from the trigger — without starting the execution loop.
    """
    scheduler.start(paused=True)
    try:
        return scheduler.get_job(job.id).next_run_time
    finally:
        scheduler.shutdown(wait=False)


def test_add_interval_job_lands_scheduled_not_paused():
    # The fix: omitting next_run_time -> the interval trigger computes the first
    # fire at now+interval, so next_run_time is a real datetime (job WILL fire).
    bg = BackgroundScheduler(timezone="UTC")
    before = datetime.now(timezone.utc)
    job = workers_main._add_interval_job(bg, lambda: None, minutes=5)
    nrt = _next_run_time(bg, job)

    assert nrt is not None, "interval job must be scheduled, not paused"
    delta = (nrt - before).total_seconds()
    # First fire is ~now+interval (not immediate): the manual pass before
    # scheduler.start() is what covers T0, so no double-run at T0.
    assert 5 * 60 - 5 <= delta <= 5 * 60 + 5


def test_explicit_none_next_run_time_is_paused_the_original_bug():
    # Documents WHY the helper must omit the kwarg: passing None explicitly is
    # exactly the 22h silent-stall footgun — the job lands paused (never fires).
    bg = BackgroundScheduler(timezone="UTC")
    job = bg.add_job(lambda: None, "interval", minutes=5, next_run_time=None)
    nrt = _next_run_time(bg, job)

    assert nrt is None, "next_run_time=None is APScheduler v3's 'add paused' — the bug"


class _FakeScheduler:
    """Stand-in for BlockingScheduler: records add_job, start() never blocks."""

    def __init__(self, *args, **kwargs):
        self.timezone = kwargs.get("timezone")
        self.jobs = []
        self.started = False

    def add_job(self, func, trigger=None, **kwargs):
        self.jobs.append(SimpleNamespace(func=func, trigger=trigger, kwargs=kwargs))
        return SimpleNamespace(id="fake-job")

    def start(self):  # BlockingScheduler.start() would block until SIGTERM
        self.started = True


def _assert_scheduled_not_paused(sched, *, expected_minutes):
    assert len(sched.jobs) == 1
    job = sched.jobs[0]
    assert job.trigger == "interval"
    assert job.kwargs.get("minutes") == expected_minutes
    # The whole point: the real worker path must NOT reintroduce next_run_time.
    assert "next_run_time" not in job.kwargs
    assert sched.started is True


def test_extraction_loop_schedules_interval_and_runs_immediate_pass(monkeypatch):
    monkeypatch.setattr(workers_main, "run_boot_assertions", lambda *a, **k: None)
    monkeypatch.setattr(apsblocking, "BlockingScheduler", _FakeScheduler)

    passes = []
    monkeypatch.setattr(
        workers_main, "_run_once", lambda settings, *, live=False: passes.append(live) or {}
    )

    captured = {}
    orig_add = workers_main._add_interval_job

    def spy_add(scheduler, func, *, minutes):
        captured["scheduler"] = scheduler
        return orig_add(scheduler, func, minutes=minutes)

    monkeypatch.setattr(workers_main, "_add_interval_job", spy_add)

    settings = SimpleNamespace(
        database_url="postgresql://x/aerys_v2",
        memories_database_url="postgresql://x/aerys",
        embeddings_api_key="sk-test",
        n8n_api_key="",
        extraction_interval_minutes=15,
    )
    args = SimpleNamespace(once=False, live=False)

    rc = workers_main._extraction_main(settings, args)

    assert rc == 0
    _assert_scheduled_not_paused(captured["scheduler"], expected_minutes=15)
    # Exactly one immediate T0 pass ran before start() — no double-run at T0.
    assert passes == [False]


def test_gaps_mine_loop_schedules_interval_and_runs_immediate_pass(monkeypatch):
    monkeypatch.setattr(workers_main, "run_boot_assertions", lambda *a, **k: None)
    monkeypatch.setattr(apsblocking, "BlockingScheduler", _FakeScheduler)
    # Owner scope present -> the miner is armed (None would refuse at startup).
    monkeypatch.setattr(factory, "action_allowlist_for", lambda s: "ALLOW")

    passes = []
    monkeypatch.setattr(
        workers_main, "_mine_gaps_once", lambda settings, allow: passes.append(allow) or {}
    )

    captured = {}
    orig_add = workers_main._add_interval_job

    def spy_add(scheduler, func, *, minutes):
        captured["scheduler"] = scheduler
        return orig_add(scheduler, func, minutes=minutes)

    monkeypatch.setattr(workers_main, "_add_interval_job", spy_add)

    settings = SimpleNamespace(
        database_url="postgresql://x/aerys_v2",
        extraction_interval_minutes=15,
    )
    args = SimpleNamespace(once=False)

    rc = workers_main._gaps_mine_main(settings, args)

    assert rc == 0
    _assert_scheduled_not_paused(captured["scheduler"], expected_minutes=15)
    # The immediate _safe_pass() ran exactly once, with the owner allowlist.
    assert passes == ["ALLOW"]
