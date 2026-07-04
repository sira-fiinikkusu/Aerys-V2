"""Worker entrypoint — the batch jobs that run beside the Brain, not inside a turn.

Subcommands:
  extraction [--once] [--live]  — memory extraction (n8n 04-02 port; shadow/live)
  gaps-mine  [--once]           — capability-request miner (self-iteration, Phase A)
  gaps       [--status] [--limit] — the owner READ path for mined gaps (/gaps)

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

from ..config import BootConfigError, Settings, run_boot_assertions
from ..services.memory import openrouter_embedder
from .capability_requests import (
    GapMiningRefused,
    format_gaps,
    read_gaps,
    run_gap_mining,
)
from .extraction import n8n_workflow_active, openrouter_chat, run_extraction, run_live_extraction

log = logging.getLogger("aerys_v2.workers")


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
                n8n_active = n8n_workflow_active(
                    settings.n8n_api_key.get_secret_value(),
                    base_url=settings.n8n_base_url,
                )
                with psycopg.connect(settings.memories_database_url) as prod_write_conn:
                    summary = run_live_extraction(
                        source_conn,
                        staging_conn,
                        prod_write_conn,
                        llm,
                        embedder,
                        n8n_active,
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
    return summary


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
    if args.live and not settings.n8n_api_key:
        # Only --live needs this — shadow mode never calls the n8n API.
        missing.append("N8N_API_KEY")
    if missing:
        print(f"extraction worker needs: {', '.join(missing)}", file=sys.stderr)
        return 2

    # Same env-scare gate as --serve/--discord: this worker WRITES to
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
    scheduler.add_job(
        lambda: _run_once(settings, live=args.live),
        "interval",
        minutes=settings.extraction_interval_minutes,
        next_run_time=None,
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
    scheduler.add_job(
        _safe_pass, "interval",
        minutes=settings.extraction_interval_minutes, next_run_time=None,
    )
    log.info("gaps miner loop armed: every %s min", settings.extraction_interval_minutes)
    _safe_pass()  # fire immediately, then settle into the interval
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
             "(requires N8N_API_KEY; refuses unless the writer lease is held by 'brain' "
             "and the n8n batch-extraction workflow is inactive)",
    )

    gaps_mine = sub.add_parser(
        "gaps-mine", help="mine v2_turns for capability gaps (self-iteration, Phase A)"
    )
    gaps_mine.add_argument("--once", action="store_true", help="single pass, then exit")

    gaps = sub.add_parser("gaps", help="read the mined capability gaps (owner /gaps path)")
    gaps.add_argument(
        "--status", default=None,
        help="filter to one status (open|surfaced|diagnosing|proposed|approved|"
             "building|built|rejected|wont_fix); default: all",
    )
    gaps.add_argument("--limit", type=int, default=50, help="max rows (default 50)")

    args = parser.parse_args(argv)
    settings = Settings()

    if args.worker == "extraction":
        return _extraction_main(settings, args)
    if args.worker == "gaps-mine":
        return _gaps_mine_main(settings, args)
    if args.worker == "gaps":
        return _gaps_read_main(settings, args)
    return 2  # unreachable: subparsers is required=True


if __name__ == "__main__":
    sys.exit(main())
