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
    try:
        settings = Settings()
    except ValidationError as e:
        log.error(f"Error validating settings: {e}")
        sys.exit(1)

    if health:  # only after a clean load
        print("ok")  # confirm that we loaded successfully, which means we're healthy
        sys.exit(0)  # exit with success

    if "--ask" in sys.argv:  # one-shot turn: aerys-v2 --ask "hello" — the first REAL call path
        from aerys_v2.factory import build_graph, build_model, load_soul
        from aerys_v2.service import ask

        text = sys.argv[sys.argv.index("--ask") + 1]
        graph = build_graph(build_model(settings), soul=load_soul(settings.soul_file_path))
        reply = ask(
            graph,
            text,
            # CLI caller = the operator; real transports resolve identity properly (S2)
            identity={"user_id": "cli-operator", "display_name": "Chris (CLI)"},
            thread_id="cli",  # InMemorySaver → each CLI run is a fresh thread for now
        )
        print(reply)
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
