"""Worker entrypoint — the batch jobs that run beside the Brain, not inside a turn.

Subcommands:
  extraction [--once] [--live]  — memory extraction (n8n 04-02 port; shadow/live)
  gaps-mine  [--once]           — capability-request miner (self-iteration, Phase A)
  email-watch [--once]          — her-inbox arrival pings (n8n 05-03 Gmail Trigger, IMAP rebuild)
  gaps       [--status] [--limit] — the owner READ path for mined gaps (/gaps)
  signals    [--window] [--quiet] — invariants over real traffic; the checks that
                                    would have caught the week of 2026-09-13
  gap-board                       — publish her own filed gaps to the shared board

n8n mapping: the Schedule Trigger node. `--once` is a manual "Execute Workflow"
click (one pass, exit code says whether anything landed); without it, APScheduler
runs the same pass on an interval — this is the process a future worker container
runs as PID 1, separate from the Brain's serve loop.

`--live` (extraction only) swaps the write target from shadow staging
(v2_memories_staging) to prod `memories`, via run_live_extraction()'s triage
(insert/update/replace) instead of run_extraction()'s append-only insert. Default
(no --live) is UNCHANGED shadow mode — this flag opt-INS into the two hard gates
(n8n-inactive, writer-lease held by 'brain'), never opts out of anything.

Wiring rule (matches factory.py): connections are opened HERE and injected — the
worker logic in extraction.py / capability_requests.py never connects on its own,
which is why their tests run offline. Fresh short connections per pass, same "pool
is a drop-in later" stance as the memory-context seam.
"""

import argparse
import json
import logging
import sys
from typing import Callable

from ..config import BootConfigError, Settings, run_boot_assertions
from ..services.memory import openrouter_embedder
from .capability_requests import (
    GapMiningRefused,
    format_gaps,
    read_gaps,
    run_gap_mining,
)
from .extraction import openrouter_chat, run_extraction, run_live_extraction

log = logging.getLogger("aerys_v2.workers")


def _add_interval_job(scheduler, func, *, minutes: int):
    """Add ``func`` to ``scheduler`` on a fixed interval — scheduled, never paused.

    APScheduler v3 treats an EXPLICIT ``next_run_time=None`` as "add this job
    PAUSED" (BaseScheduler.add_job docstring: "pass ``None`` to add the job as
    paused"). A paused interval job never fires: empirically the worker ran ONE
    startup pass and then sat idle for 22h, silently killing both memory
    extraction and the gaps-miner cadence. The fix is to OMIT the kwarg entirely
    so the interval trigger computes the first fire at now+interval. Each caller
    still does one manual pass right before ``scheduler.start()`` for the
    immediate T0 run — so the net cadence is: immediate pass + every-interval
    passes, with no double-run at T0.

    Kept as a tiny seam so a unit test can prove the job lands SCHEDULED
    (next_run_time is not None) rather than paused.
    """
    return scheduler.add_job(func, "interval", minutes=minutes)


def _run_once(settings: Settings, *, live: bool = False) -> dict:
    """One pass: shadow staging by default, or prod triage when --live."""
    import psycopg

    llm = openrouter_chat(
        settings.embeddings_api_key.get_secret_value(),
        model=settings.extraction_model,
        base_url=settings.embeddings_base_url,
    )
    embedder = openrouter_embedder(settings.embeddings_api_key.get_secret_value())

    # prod aerys (READ-ONLY — the same belt-and-braces as factory's memory-context
    # connection) + aerys_v2 (v2_turns reads and every write). `with` commits the
    # staging transaction on clean exit, rolls back if the pass blew up mid-batch.
    with psycopg.connect(settings.memories_database_url) as source_conn:
        source_conn.read_only = True
        with psycopg.connect(settings.database_url) as staging_conn:
            if live:
                # A THIRD connection to the SAME url as source_conn — read_only
                # is a per-connection posture, so writing to prod `memories`
                # needs its own connection object, never source_conn itself.
                with psycopg.connect(settings.memories_database_url) as prod_write_conn:
                    summary = run_live_extraction(
                        source_conn,
                        staging_conn,
                        prod_write_conn,
                        llm,
                        embedder,
                        lookback_hours=settings.extraction_lookback_hours,
                        batch_limit=settings.extraction_batch_limit,
                    )
            else:
                summary = run_extraction(
                    source_conn,
                    staging_conn,
                    llm,
                    embedder,
                    lookback_hours=settings.extraction_lookback_hours,
                    batch_limit=settings.extraction_batch_limit,
                )
    log.info("extraction pass (%s): %s", "live" if live else "shadow", json.dumps(summary))
    _check_stalled(summary)
    return summary


#: Consecutive passes that read rows and stored nothing before we say so out loud.
#: 3 (hourly loop) = ~3h, long enough that a run of genuinely unmemorable chatter
#: doesn't cry wolf, short enough to catch a stall the same day.
STALL_PASSES = 3

#: Module-level because the scheduler calls _run_once repeatedly in one process.
_zero_insert_streak = 0
#: Set by main() from KAEL_DESK_URL/TOKEN: the stall line ALSO goes to Kael's live
#: session. 2026-09-20: the WARNING below fired for 11 hours into a log nobody
#: reads while memory formation was stopped — a repeat of the 7/29 lesson at a
#: smaller scale. A log line is not a voice.
_stall_alarm: Callable[[str], None] | None = None


def kael_desk_alarm_for(settings: Settings) -> Callable[[str], None] | None:
    if not settings.kael_desk_url or settings.kael_desk_token is None:
        return None
    import httpx

    url, token = settings.kael_desk_url, settings.kael_desk_token.get_secret_value()

    def alarm(text: str) -> None:
        try:
            httpx.post(url, json={"message": text[:1800]}, headers={"Authorization": f"Bearer {token}"}, timeout=5.0)
        except Exception:
            log.warning("stall alarm delivery failed", exc_info=True)

    return alarm


def _check_stalled(summary: dict) -> None:
    """Say it OUT LOUD when the worker is reading rows and storing nothing.

    The 2026-07-29 outage was not really a parse bug. The parse bug cost 24 days
    because `inserted_total: 0`, pass after pass, existed ONLY in a log line
    nobody was reading — every container reported healthy, the brain stayed up,
    and the watermark guard was working exactly as designed while memory formation
    was completely stopped. Absence of writes is a signal; it needs a voice.

    Fires on the PATTERN, never a single quiet hour: a window with no rows is a
    genuinely idle interval, not a stall. WARNING (not INFO) so it separates from
    the routine pass log, and it carries the watermarks + parse_failures so the
    first question ("what is it stuck behind?") is answered in the line itself.
    """
    global _zero_insert_streak
    sources = summary.get("sources") or {}
    had_rows = any((s or {}).get("rows") for s in sources.values())
    if summary.get("inserted_total") or not had_rows:
        _zero_insert_streak = 0
        return
    _zero_insert_streak += 1
    if _zero_insert_streak < STALL_PASSES:
        return
    detail = json.dumps({
        "consecutive_passes": _zero_insert_streak,
        "rows_read": {n: (s or {}).get("rows") for n, s in sources.items()},
        "parse_failures": {n: (s or {}).get("parse_failures") for n, s in sources.items()},
        "watermarks": {n: (s or {}).get("watermark") for n, s in sources.items()},
    })
    if _stall_alarm is not None and _zero_insert_streak in (STALL_PASSES, STALL_PASSES * 4, STALL_PASSES * 8):
        _stall_alarm(f"[aerys-extractor] extraction stalled: {detail}")  # first, then ~12 h and ~24 h in
    log.warning(
        "extraction stalled: %s",
        detail,
    )


def _extraction_main(settings: Settings, args: argparse.Namespace) -> int:
    """`extraction [--once] [--live]` — unchanged from before subcommands existed."""
    # Same arming pattern as every optional transport: missing config = the worker
    # refuses loudly at startup, not quietly mid-pass.
    missing = [
        name
        for name, value in (
            ("DATABASE_URL", settings.database_url),
            ("MEMORIES_DATABASE_URL", settings.memories_database_url),
            ("EMBEDDINGS_API_KEY", settings.embeddings_api_key),
        )
        if not value
    ]
    if missing:
        print(f"extraction worker needs: {', '.join(missing)}", file=sys.stderr)
        return 2

    # Same env-scare gate as --serve/--discord/--telegram: this worker WRITES to
    # database_url (staging + watermark), so a URL aimed at prod `aerys`
    # must refuse to run, not fail halfway through a pass.
    try:
        run_boot_assertions(settings)
    except BootConfigError as e:
        print(f"extraction worker refusing to start: {e}", file=sys.stderr)
        return 2

    if args.once:
        summary = _run_once(settings, live=args.live)
        print(json.dumps(summary, indent=2))
        return 0

    # Loop mode — the future container's steady state. Imported lazily so the
    # scheduler is a loop-mode-only dependency (tests and --once never touch it).
    from apscheduler.schedulers.blocking import BlockingScheduler

    scheduler = BlockingScheduler(timezone="UTC")
    _add_interval_job(
        scheduler,
        lambda: _run_once(settings, live=args.live),
        minutes=settings.extraction_interval_minutes,
    )
    log.info("extraction loop armed (%s): every %s min",
              "live" if args.live else "shadow", settings.extraction_interval_minutes)
    _run_once(settings, live=args.live)  # fire immediately, then settle into the interval
    scheduler.start()  # blocks until SIGINT/SIGTERM
    return 0


def _mine_gaps_once(settings: Settings, allowlist) -> dict:
    """One capability-mining pass over the brain's OWN aerys_v2 database.

    ONE connection — the miner reads v2_turns and writes the two capability tables
    + the watermark, all in aerys_v2 (unlike extraction, no prod aerys connection).
    `with` commits on clean exit, rolls back if the pass blew up mid-batch."""
    import psycopg

    # Loop-mode self-defense (cross-review). The miner is OFFLINE, so a wedged NAS
    # Postgres can never crash a live turn — but in loop mode it would hang the
    # BlockingScheduler's single job thread. Bound the connect, and cap any single
    # statement, so a stuck connect/query surfaces as a caught error and the next
    # interval retries instead of the pass blocking forever. run_gap_mining holds ONE
    # transaction across the batch, so the SET is a per-STATEMENT ceiling (each SELECT/
    # INSERT/watermark write), not a whole-pass one — enough to unstick a hung DB.
    # The SET also guarantees a transaction is open before the per-turn SAVEPOINTs,
    # reinforcing the non-autocommit invariant run_gap_mining now asserts.
    with psycopg.connect(settings.database_url, connect_timeout=10) as conn:
        conn.execute("SET statement_timeout = '120s'")
        summary = run_gap_mining(
            conn,
            allowlist=allowlist,
            lookback_hours=settings.extraction_lookback_hours,
            batch_limit=settings.extraction_batch_limit,
        )
    log.info("gaps mining pass: %s", json.dumps(summary))
    return summary


def _gaps_mine_main(settings: Settings, args: argparse.Namespace) -> int:
    """`gaps-mine [--once]` — the self-iteration miner (Phase A)."""
    # Owner scope is a HARD requirement (the design's None-defeatable caveat made a
    # boot assertion): action_allowlist_for is None when no owner is configured, and
    # mining without an owner scope is exactly what H2 forbids. Refuse loudly. Imported
    # lazily (the codebase's CLI-branch convention) so the lean `gaps` read path never
    # pulls in the factory/langchain stack just to print a table.
    from ..factory import action_allowlist_for

    allow = action_allowlist_for(settings)
    missing = [n for n, v in (("DATABASE_URL", settings.database_url),) if not v]
    if allow is None:
        missing.append("OWNER_PERSON_ID")
    if missing:
        print(f"gaps miner needs: {', '.join(missing)}", file=sys.stderr)
        return 2

    # Same env-scare gate as extraction: this worker WRITES to database_url, so a
    # URL aimed at prod `aerys` must refuse to run, not fail halfway through.
    try:
        run_boot_assertions(settings)
    except BootConfigError as e:
        print(f"gaps miner refusing to start: {e}", file=sys.stderr)
        return 2

    if args.once:
        # The parity gate / empty-allowlist gate raise GapMiningRefused — a --once
        # run surfaces that as a distinct nonzero exit, not a stack trace.
        try:
            summary = _mine_gaps_once(settings, allow)
        except GapMiningRefused as e:
            print(f"gaps miner refused: {e}", file=sys.stderr)
            return 3
        print(json.dumps(summary, indent=2))
        return 0

    # Loop mode — a GapMiningRefused (e.g. the writer not yet landed, or a
    # momentarily empty table) is logged and the pass SKIPPED, so the loop keeps
    # retrying every interval instead of dying; the writer may land later.
    from apscheduler.schedulers.blocking import BlockingScheduler

    def _safe_pass() -> None:
        try:
            _mine_gaps_once(settings, allow)
        except GapMiningRefused as e:
            log.warning("gaps mining pass skipped: %s", e)

    scheduler = BlockingScheduler(timezone="UTC")
    _add_interval_job(scheduler, _safe_pass, minutes=settings.extraction_interval_minutes)
    log.info("gaps miner loop armed: every %s min", settings.extraction_interval_minutes)
    _safe_pass()  # fire immediately, then settle into the interval
    scheduler.start()  # blocks until SIGINT/SIGTERM
    return 0


def _email_watch_once(settings: Settings, notify_fn) -> dict:
    """One watch pass over her inbox. AUTOCOMMIT connection on purpose: the
    watcher saves the watermark PER pinged message, and that durability story
    (module docstring) only holds if each save lands as it happens — a deferred
    commit would roll back every earlier save when a mid-burst failure aborts
    the transaction."""
    import psycopg

    from .email_watch import imap_login_factory, run_once

    imap_factory = imap_login_factory(
        settings.email_imap_host,
        settings.email_address,
        settings.email_app_password.get_secret_value(),
    )
    with psycopg.connect(settings.database_url, connect_timeout=10,
                         autocommit=True) as conn:
        conn.execute("SET statement_timeout = '30s'")
        stats = run_once(conn, imap_factory, notify_fn)
    log.info("email watch pass: %s", json.dumps(stats))
    return stats


def _email_watch_main(settings: Settings, args: argparse.Namespace) -> int:
    """`email-watch [--once]` — her-inbox arrival pings (the notification half
    of n8n 05-03 Gmail Trigger, rebuilt on IMAP; scope decided 2026-07-11)."""
    from ..factory import discord_dm_notify_for

    notify_fn = discord_dm_notify_for(settings)
    missing = [
        name
        for name, value in (
            ("DATABASE_URL", settings.database_url),
            ("EMAIL_ADDRESS", settings.email_address),
            ("EMAIL_APP_PASSWORD", settings.email_app_password),
        )
        if not value
    ]
    if notify_fn is None:
        missing.append("DISCORD_BOT_TOKEN + EMAIL_NOTIFY_DISCORD_USER_ID")
    if missing:
        print(f"email watcher needs: {', '.join(missing)}", file=sys.stderr)
        return 2

    # Same env-scare gate as the other workers: this one WRITES the watermark
    # to database_url, so a URL aimed at prod `aerys` must refuse to run.
    try:
        run_boot_assertions(settings)
    except BootConfigError as e:
        print(f"email watcher refusing to start: {e}", file=sys.stderr)
        return 2

    if args.once:
        stats = _email_watch_once(settings, notify_fn)
        print(json.dumps(stats, indent=2))
        return 0 if stats.get("ok") else 1

    from apscheduler.schedulers.blocking import BlockingScheduler

    scheduler = BlockingScheduler(timezone="UTC")
    # Seconds cadence (not the minutes seam) — mail pings want the 3-minute
    # default, and run_once already catches everything, so a failed pass just
    # waits for the next tick. Same no-next_run_time rule as _add_interval_job.
    scheduler.add_job(lambda: _email_watch_once(settings, notify_fn),
                      "interval", seconds=settings.email_poll_seconds)
    log.info("email watch loop armed: every %ss on %s",
             settings.email_poll_seconds, settings.email_address)
    _email_watch_once(settings, notify_fn)  # immediate T0 pass, then the interval
    scheduler.start()  # blocks until SIGINT/SIGTERM
    return 0


def _gaps_read_main(settings: Settings, args: argparse.Namespace) -> int:
    """`gaps [--status] [--limit]` — the owner READ path (/gaps). Read-only."""
    if not settings.database_url:
        print("gaps read needs: DATABASE_URL", file=sys.stderr)
        return 2
    try:
        run_boot_assertions(settings)
    except BootConfigError as e:
        print(f"gaps read refusing to start: {e}", file=sys.stderr)
        return 2
    import psycopg

    with psycopg.connect(settings.database_url) as conn:
        # Belt-and-braces: the /gaps surface never writes; the DB refuses one too.
        conn.read_only = True
        rows = read_gaps(conn, status=args.status, limit=args.limit)
    print(format_gaps(rows))
    return 0


def _reflex_report_main(settings: Settings, args: argparse.Namespace) -> int:
    if not settings.database_url:
        print("reflex-report needs: DATABASE_URL", file=sys.stderr)
        return 2
    try:
        run_boot_assertions(settings)
    except BootConfigError as e:
        print(f"reflex-report refusing to start: {e}", file=sys.stderr)
        return 2
    import psycopg

    from .reflex_report import format_report, read_reflex_rows

    with psycopg.connect(settings.database_url, connect_timeout=10) as conn:
        conn.read_only = True
        conn.execute("SET statement_timeout = '30s'")
        rows = read_reflex_rows(conn, args.window)
    print(format_report(rows))
    return 0


def _privacy_report_main(settings: Settings, args: argparse.Namespace) -> int:
    if not settings.database_url:
        print("privacy-report needs: DATABASE_URL", file=sys.stderr)
        return 2
    import psycopg

    from .privacy_report import format_report, read_rows

    with psycopg.connect(settings.database_url, connect_timeout=10) as conn:
        conn.read_only = True
        conn.execute("SET statement_timeout = '30s'")
        rows = read_rows(conn, args.window)
    print(format_report(rows))
    return 0


def _signals_main(settings: Settings, args: argparse.Namespace) -> int:
    """`signals [--window] [--quiet]` — invariants over the traffic that happened.

    READ-ONLY, by design and by connection. It plants nothing: a plant-and-ask probe
    has to write a synthetic turn into his thread and junk into her memory, so testing
    the real path would mean polluting it. Exit 1 when a signal fails, so a scheduler
    or a cron can treat this like any other check.
    """
    if not settings.database_url:
        print("signals needs: DATABASE_URL", file=sys.stderr)
        return 2
    try:
        run_boot_assertions(settings)
    except BootConfigError as e:
        print(f"signals refusing to start: {e}", file=sys.stderr)
        return 2
    import psycopg

    from ..factory import owner_room_names_for
    from .signals import format_report, run_signals, should_speak

    # The names his own account answers to on a platform. If a room line carries one
    # of these as a speaker, identity resolution was bypassed somewhere.
    aliases = args.alias or []

    with psycopg.connect(settings.database_url) as turns_conn:
        turns_conn.read_only = True
        memories_conn = None
        try:
            if settings.memories_database_url:
                memories_conn = psycopg.connect(settings.memories_database_url)
                memories_conn.read_only = True
            results = run_signals(
                turns_conn=turns_conn, memories_conn=memories_conn,
                person_id=settings.owner_person_id, aliases=aliases, window=args.window)
        finally:
            if memories_conn is not None:
                memories_conn.close()

    report = format_report(results)
    if should_speak(results) or not args.quiet:
        print(report)
    return 1 if should_speak(results) else 0


def _gap_board_main(settings: Settings, args: argparse.Namespace) -> int:
    """`gap-board [--once]` — put her unpublished open gaps on the shared board.

    The table stays the source of truth; this is the pass that makes a gap WORKABLE
    by putting it where comments, labels and closes happen. Idempotent by the
    board_issue column (migration 011), bounded per run, private repo only, and
    anything credential-shaped is held back rather than redacted and sent.
    """
    if not settings.database_url:
        print("gap-board needs: DATABASE_URL", file=sys.stderr)
        return 2
    if not (settings.board_repo and settings.board_token):
        print("gap-board needs: BOARD_REPO and BOARD_TOKEN (her GitHub identity)",
              file=sys.stderr)
        return 2
    try:
        run_boot_assertions(settings)
    except BootConfigError as e:
        print(f"gap-board refusing to start: {e}", file=sys.stderr)
        return 2
    import psycopg

    from .gap_board import GitHubBoard, run_publish

    board = GitHubBoard(repo=settings.board_repo, token=settings.board_token,
                        private=True)
    with psycopg.connect(settings.database_url) as conn:
        filed = run_publish(conn, board=board)
    print(f"filed {filed} gap(s) on {settings.board_repo}")
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(prog="python -m aerys_v2.workers")
    sub = parser.add_subparsers(dest="worker", required=True)

    extraction = sub.add_parser("extraction", help="shadow memory extraction")
    extraction.add_argument("--once", action="store_true", help="single pass, then exit")
    extraction.add_argument(
        "--live",
        action="store_true",
        help="write triaged memories to PROD instead of shadow staging "
             "(refuses unless the writer lease is held by 'brain')",
    )

    gaps_mine = sub.add_parser(
        "gaps-mine", help="mine v2_turns for capability gaps (self-iteration, Phase A)"
    )
    gaps_mine.add_argument("--once", action="store_true", help="single pass, then exit")

    email_watch = sub.add_parser(
        "email-watch", help="poll her Gmail inbox over IMAP, ping the owner on new mail"
    )
    email_watch.add_argument("--once", action="store_true", help="single pass, then exit")

    gaps = sub.add_parser("gaps", help="read the mined capability gaps (owner /gaps path)")
    gaps.add_argument(
        "--status", default=None,
        help="filter to one status (open|surfaced|diagnosing|proposed|approved|"
             "building|built|rejected|wont_fix); default: all",
    )
    gaps.add_argument("--limit", type=int, default=50, help="max rows (default 50)")

    privacy_report = sub.add_parser("privacy-report", help="Phase 3 shadow: Jev vs the metered privacy judge")
    privacy_report.add_argument("--window", default="24 hours")
    reflex_report = sub.add_parser("reflex-report", help="read Jev shadow agreement")
    reflex_report.add_argument("--window", default="24 hours",
                               help="lookback as a Postgres interval (default 24 hours)")

    signals = sub.add_parser(
        "signals", help="check invariants over recent real traffic (read-only)")
    signals.add_argument("--window", default="24 hours",
                         help="how far back to look, as a Postgres interval (default 24 hours)")
    signals.add_argument("--alias", action="append", default=None,
                         help="a display name his own account answers to (repeatable); "
                              "a room line speaking as one means identity resolution was skipped")
    signals.add_argument("--quiet", action="store_true",
                         help="print only when something failed — a green run says nothing")

    sub.add_parser(
        "gap-board",
        help="put her unpublished open gaps on the shared board (private repo only)")

    args = parser.parse_args(argv)
    settings = Settings()

    if args.worker == "extraction":
        global _stall_alarm
        _stall_alarm = kael_desk_alarm_for(settings)
        return _extraction_main(settings, args)
    if args.worker == "gaps-mine":
        return _gaps_mine_main(settings, args)
    if args.worker == "email-watch":
        return _email_watch_main(settings, args)
    if args.worker == "gaps":
        return _gaps_read_main(settings, args)
    if args.worker == "reflex-report":
        return _reflex_report_main(settings, args)
    if args.worker == "privacy-report":
        return _privacy_report_main(settings, args)
    if args.worker == "signals":
        return _signals_main(settings, args)
    if args.worker == "gap-board":
        return _gap_board_main(settings, args)
    return 2  # unreachable: subparsers is required=True


if __name__ == "__main__":
    sys.exit(main())
