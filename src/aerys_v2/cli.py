import logging, sys
import signal, threading
from pydantic import ValidationError
from aerys_v2.config import Settings

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
log = logging.getLogger("aerys_v2")


def main() -> None:
    health = (
        "--health" in sys.argv
    )  # check for health check flag before loading settings
    wants_eval = "--eval" in sys.argv  # checked pre-Settings so we can instruct on failure
    try:
        settings = Settings()
    except ValidationError as e:
        if wants_eval:
            # --eval needs a REAL judge (an actual model call scoring each reply) —
            # there is no offline mode here; the offline path is tests/test_evals.py
            # with fake models. So a missing key gets instructions, not a stacktrace.
            print(
                "--eval requires a configured ANTHROPIC_API_KEY (the judge is a real "
                "model call).\nAdd it to .env (see config.py Settings) and rerun."
            )
            sys.exit(1)
        log.error(f"Error validating settings: {e}")
        sys.exit(1)

    if health:  # only after a clean load
        print("ok")  # confirm that we loaded successfully, which means we're healthy
        sys.exit(0)  # exit with success

    if "--ask" in sys.argv:  # one-shot turn: aerys-v2 --ask "hello" — the first REAL call path
        from aerys_v2.factory import build_graph, build_model, load_soul
        from aerys_v2.service import ask

        from aerys_v2.factory import checkpointer_for

        text = sys.argv[sys.argv.index("--ask") + 1]
        with checkpointer_for(settings) as cp:  # Postgres when DATABASE_URL set → durable
            graph = build_graph(
                build_model(settings), soul=load_soul(settings.soul_file_path), checkpointer=cp
            )
            reply = ask(
                graph,
                text,
                # CLI caller = the operator; real transports resolve identity properly (S2)
                identity={"user_id": "cli-operator", "display_name": "Chris (CLI)"},
                thread_id="cli",  # durable with DATABASE_URL: separate runs SHARE this thread
            )
        print(reply)
        sys.exit(0)

    if wants_eval:  # run the eval harness against the local graph: aerys-v2 --eval
        # n8n mapping: this is "manually execute the 06-01 Eval Suite workflow",
        # except the dataset/target/judge are library code we can also unit-test.
        from aerys_v2.evals.runner import (
            Judge,
            LocalGraphTarget,
            format_summary_table,
            load_cases,
            run_eval,
        )
        from aerys_v2.factory import build_graph, build_model, load_soul

        cases = load_cases()  # golden.json locally; example.json on a fresh clone/CI
        log.info("eval: %d case(s) loaded, judging with model=%s", len(cases), settings.model)
        graph = build_graph(build_model(settings), soul=load_soul(settings.soul_file_path))
        results, summary = run_eval(LocalGraphTarget(graph), cases, Judge.from_settings(settings))
        for r in results:  # one line per case — the per-item view before the rollup
            print(f"[{r['score']}] {r['id']} ({r['category']}, {r['latency_ms']:.0f}ms) — {r['reasoning']}")
        print()
        print(format_summary_table(summary))
        sys.exit(0)

    if "--serve" in sys.argv:  # run the HTTP door (deploy target: the Jetson container)
        if settings.api_token is None:
            print("--serve needs API_TOKEN in .env (Bearer token for /ask).")
            sys.exit(1)
        import uvicorn

        from aerys_v2.factory import build_graph, build_model, checkpointer_for, load_soul
        from aerys_v2.service import ask
        from aerys_v2.transports.http_api import build_app

        with checkpointer_for(settings) as cp:
            graph = build_graph(
                build_model(settings), soul=load_soul(settings.soul_file_path), checkpointer=cp
            )
            app = build_app(
                lambda text, identity, thread: ask(graph, text, identity=identity, thread_id=thread),
                settings.api_token.get_secret_value(),
            )
            uvicorn.run(app, host="0.0.0.0", port=settings.api_port, log_level="info")
        sys.exit(0)

    if "--discord" in sys.argv:  # run the 1c gateway spike (needs DISCORD_BOT_TOKEN in .env)
        if settings.discord_bot_token is None:
            print("--discord needs DISCORD_BOT_TOKEN in .env (dev bot token).")
            sys.exit(1)
        from aerys_v2.factory import build_graph, build_model, load_soul
        from aerys_v2.service import ask
        from aerys_v2.factory import checkpointer_for
        from aerys_v2.transports.discord_gateway import AerysDiscordClient

        cp_ctx = checkpointer_for(settings)
        cp = cp_ctx.__enter__()  # held for the life of the gateway process
        graph = build_graph(build_model(settings), soul=load_soul(settings.soul_file_path), checkpointer=cp)

        def resolve(event):  # spike resolver: display-name passthrough (DB resolver later)
            return {"user_id": f"discord:{event.platform_user_id}", "display_name": event.display_name}

        channel_ids = frozenset(
            int(c) for c in settings.discord_reply_channel_ids.split(",") if c.strip()
        )
        client = AerysDiscordClient(
            ask_fn=lambda text, identity, thread: ask(graph, text, identity=identity, thread_id=thread),
            resolve_fn=resolve,
            allowed_guild_id=settings.discord_guild_id,
            allowed_channel_ids=channel_ids,
        )
        client.run(settings.discord_bot_token.get_secret_value())
        sys.exit(0)

    log.info(
        "aerys-v2 ready | model=%s soul=%s otlp=%s",
        settings.model,
        settings.soul_file_path,
        "on" if settings.otlp_endpoint else "off",
    )

    stop = threading.Event()  # create an event to signal shutdown

    def _shutdown(signum, frame):  # signal handler for SIGINT/SIGTERM
        log.info(
            "received signal %s, shutting down", signum
        )  # signal handler sets the event, which unblocks stop.wait() below
        stop.set()  # set the event to signal shutdown

    signal.signal(signal.SIGTERM, _shutdown)  # handle SIGTERM for graceful shutdown
    signal.signal(
        signal.SIGINT, _shutdown
    )  # handle SIGINT (Ctrl+C) for graceful shutdown

    stop.wait()  # block here until the event is set by the signal handler
    log.info(
        "aerys-v2 has stopped gracefully"
    )  # once we get here, we know the shutdown signal was received and we can exit cleanly
