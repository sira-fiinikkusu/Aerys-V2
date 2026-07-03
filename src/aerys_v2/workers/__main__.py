"""Worker entrypoint — `python -m aerys_v2.workers extraction [--once]`.

n8n mapping: the Schedule Trigger node. `--once` is a manual "Execute Workflow"
click (one pass, exit code says whether anything landed); without it, APScheduler
runs the same pass on an interval — this is the process a future worker container
runs as PID 1, separate from the Brain's serve loop.

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
from .extraction import openrouter_chat, run_extraction

log = logging.getLogger("aerys_v2.workers")


def _run_once(settings: Settings) -> dict:
    """One shadow pass: open both conns, run, commit, report."""
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
            summary = run_extraction(
                source_conn,
                staging_conn,
                llm,
                embedder,
                lookback_hours=settings.extraction_lookback_hours,
                batch_limit=settings.extraction_batch_limit,
            )
    log.info("extraction pass: %s", json.dumps(summary))
    return summary


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(prog="python -m aerys_v2.workers")
    sub = parser.add_subparsers(dest="worker", required=True)
    extraction = sub.add_parser("extraction", help="shadow memory extraction")
    extraction.add_argument("--once", action="store_true", help="single pass, then exit")
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
        summary = _run_once(settings)
        print(json.dumps(summary, indent=2))
        return 0

    # Loop mode — the future container's steady state. Imported lazily so the
    # scheduler is a loop-mode-only dependency (tests and --once never touch it).
    from apscheduler.schedulers.blocking import BlockingScheduler

    scheduler = BlockingScheduler(timezone="UTC")
    scheduler.add_job(
        lambda: _run_once(settings),
        "interval",
        minutes=settings.extraction_interval_minutes,
        next_run_time=None,
    )
    log.info("extraction loop armed: every %s min", settings.extraction_interval_minutes)
    _run_once(settings)  # fire immediately, then settle into the interval
    scheduler.start()  # blocks until SIGINT/SIGTERM
    return 0


if __name__ == "__main__":
    sys.exit(main())
