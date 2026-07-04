"""Worker entrypoint — `python -m aerys_v2.workers extraction [--once] [--live]`.

n8n mapping: the Schedule Trigger node. `--once` is a manual "Execute Workflow"
click (one pass, exit code says whether anything landed); without it, APScheduler
runs the same pass on an interval — this is the process a future worker container
runs as PID 1, separate from the Brain's serve loop.

`--live` swaps the write target from shadow staging (v2_memories_staging) to
prod `memories`, via run_live_extraction()'s triage (insert/update/replace)
instead of run_extraction()'s append-only insert. Default (no --live) is
UNCHANGED shadow mode — this flag opt-INS into the two hard gates
(n8n-inactive, writer-lease held by 'brain'), never opts out of anything.

Wiring rule (matches factory.py): connections are opened HERE and injected — the
worker logic in extraction.py never connects on its own, which is why its tests
run offline. Fresh short connections per pass, same "pool is a drop-in later"
stance as the memory-context seam.
"""

import argparse
import json
import logging
import sys

from ..config import BootConfigError, Settings, run_boot_assertions
from ..services.memory import openrouter_embedder
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
    args = parser.parse_args(argv)

    settings = Settings()
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


if __name__ == "__main__":
    sys.exit(main())
