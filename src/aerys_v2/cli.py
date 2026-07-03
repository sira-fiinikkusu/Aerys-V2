import logging, sys
import signal, threading
from pydantic import ValidationError
from aerys_v2.config import BootConfigError, Settings, run_boot_assertions

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
log = logging.getLogger("aerys_v2")


def main() -> None:
    health = (
        "--health" in sys.argv
    )  # check for health check flag before loading settings
    wants_eval = "--eval" in sys.argv  # checked pre-Settings so we can instruct on failure
    wants_replay = "--replay" in sys.argv  # same pre-check, same reason
    try:
        settings = Settings()
    except ValidationError as e:
        if wants_replay:
            # --replay drives the REAL model with captured traffic (the point is
            # proving live payload shapes survive the ask() seam end-to-end) —
            # the offline path is tests/test_replay.py with fake models.
            print(
                "--replay requires a configured ANTHROPIC_API_KEY (each payload is a "
                "real model call).\nAdd it to .env (see config.py Settings) and rerun."
            )
            sys.exit(1)
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

    if wants_replay:  # replay captured n8n traffic through the graph: aerys-v2 --replay
        # n8n mapping: feed the Core Agent the exact items its Execute Workflow
        # Trigger / Voice Adapter actually received — but into the V2 brain, on a
        # THROWAWAY InMemorySaver (never checkpointer_for: replaying redacted
        # captures into NAS-durable threads would poison real history).
        from aerys_v2.factory import build_model, load_soul
        from aerys_v2.replay import (
            build_replay_graph,
            format_replay_summary,
            load_payloads,
            run_replay,
        )

        payloads = load_payloads()  # payloads.json locally; example on a fresh clone/CI
        log.info("replay: %d payload(s) loaded, model=%s", len(payloads), settings.model)
        graph = build_replay_graph(build_model(settings), soul=load_soul(settings.soul_file_path))
        results, summary = run_replay(graph, payloads)
        for r in results:  # one line per payload — the per-item view before the rollup
            status = "ok" if r["ok"] else f"FAIL {r['error']}"
            print(f"[{status}] {r['id']} ({r['channel']}, {r['latency_ms']:.0f}ms, reply={r['reply_len']} chars)")
        print()
        print(format_replay_summary(summary))
        sys.exit(0 if summary["failed"] == 0 else 1)  # nonzero when any payload broke

    if "--serve" in sys.argv:  # run the HTTP door (deploy target: the Jetson container)
        if settings.api_token is None:
            print("--serve needs API_TOKEN in .env (Bearer token for /ask).")
            sys.exit(1)
        # Boot assertions BEFORE anything binds or connects: wrong-database
        # config refuses to start with a sentence, not a stack trace (the
        # env-scare prevention — see run_boot_assertions in config.py).
        try:
            run_boot_assertions(settings)
        except BootConfigError as e:
            log.error(str(e))
            sys.exit(1)
        import uvicorn

        from aerys_v2.factory import (
            action_stack_for,
            build_graph,
            build_model,
            checkpointer_for,
            context_fn_for,
            deep_gate_for,
            load_soul,
            speak_fn_for,
            tier_models_for,
        )
        from aerys_v2.service import ask
        from aerys_v2.transports.http_api import build_app

        # [01-05 PHOENIX] one line, degrade-safe: no-op unless OTLP_ENDPOINT set; any failure logs and serves anyway
        from aerys_v2.tracing import wire_tracing; wire_tracing(settings)

        soul = load_soul(settings.soul_file_path)  # shared: chat graph, action graph, router acks
        with checkpointer_for(settings) as cp:
            # Tier routing: the per-tier model map + the deep daily-cap gate
            # (v2_model_usage; None = unenforced on DB-less boxes, logged).
            tier_models = tier_models_for(settings)
            deep_gate = deep_gate_for(settings)
            log.info("tiers armed | fast=%s standard=%s deep=%s cap=%d/day",
                     settings.tier_fast_model,
                     settings.model if settings.model_backend == "oauth" else settings.tier_standard_model,
                     settings.tier_deep_model, settings.deep_daily_cap)
            graph = build_graph(
                build_model(settings),
                soul=soul,
                checkpointer=cp,
                # long-term memory context: ON only when MEMORIES_DATABASE_URL is
                # set (read-only prod aerys DB); None keeps the graph memory-free
                context_fn=context_fn_for(settings),
                tier_models=tier_models,
            )
            # TOOLS block (Option C): arms when HA_TOKEN (home) and/or
            # EMBEDDINGS_API_KEY (media) is set (the api key the router/tool
            # model needs is structurally required by Settings). None = ask()
            # runs chat-only, exactly as before tools existed.
            router = action_graph = None
            stack = action_stack_for(settings, soul)
            if stack is not None:
                router, action_graph = stack
                log.info("action stack armed | ha=%s canary=[%s] media=%s",
                         settings.ha_base_url if settings.ha_token else "(off)",
                         settings.ha_canary_entities,
                         "on" if settings.embeddings_api_key else "off")
            # Spoken follow-up seam: None = history-only (no announce entity).
            speak_fn = speak_fn_for(settings)
            if speak_fn is not None:
                log.info("spoken follow-ups armed | entity=%s skip<=%.1fs",
                         settings.ha_announce_entity, settings.voice_followup_skip_s)
            app = build_app(
                lambda text, identity, thread: ask(
                    graph, text, identity=identity, thread_id=thread,
                    router=router, action_graph=action_graph,
                    speak_fn=speak_fn,
                    followup_skip_s=settings.voice_followup_skip_s,
                    deep_allowed=deep_gate,
                ),
                settings.api_token.get_secret_value(),
                # authed HTTP callers ARE the owner when configured — voice-Chris
                # retrieves HIS memories (identity user_id = owner persons.id)
                owner_person_id=settings.owner_person_id,
            )
            uvicorn.run(app, host="0.0.0.0", port=settings.api_port, log_level="info")
        sys.exit(0)

    if "--discord" in sys.argv:  # run the 1c gateway spike (needs DISCORD_BOT_TOKEN in .env)
        if settings.discord_bot_token is None:
            print("--discord needs DISCORD_BOT_TOKEN in .env (dev bot token).")
            sys.exit(1)
        # Same env-scare gate as --serve — the gateway checkpoints turns too,
        # so a wrong-database DATABASE_URL must refuse here just as hard.
        try:
            run_boot_assertions(settings)
        except BootConfigError as e:
            log.error(str(e))
            sys.exit(1)
        from aerys_v2.factory import (
            action_stack_for,
            build_graph,
            build_model,
            checkpointer_for,
            deep_gate_for,
            load_soul,
            tier_models_for,
        )
        from aerys_v2.service import ask
        from aerys_v2.transports.discord_gateway import AerysDiscordClient

        # [01-05 PHOENIX] same degrade-safe arming as --serve: the soak
        # container's turns must trace too (gap found 2026-07-03 — only
        # --serve called wire_tracing, so aerys-soak turns never reached
        # Phoenix). No-op unless OTLP_ENDPOINT is set; failures log and run on.
        from aerys_v2.tracing import wire_tracing; wire_tracing(settings)

        soul = load_soul(settings.soul_file_path)
        cp_ctx = checkpointer_for(settings)
        cp = cp_ctx.__enter__()  # held for the life of the gateway process
        # Discord IS the text channel tier routing exists for: greetings ride
        # fast, conversation rides standard (oauth pool when configured), and
        # research earns deep until the daily cap says otherwise.
        graph = build_graph(
            build_model(settings), soul=soul, checkpointer=cp,
            tier_models=tier_models_for(settings),
        )
        deep_gate = deep_gate_for(settings)
        router = action_graph = None
        stack = action_stack_for(settings, soul)
        if stack is not None:
            router, action_graph = stack

        def resolve(event):  # spike resolver: display-name passthrough (DB resolver later)
            return {"user_id": f"discord:{event.platform_user_id}", "display_name": event.display_name}

        channel_ids = frozenset(
            int(c) for c in settings.discord_reply_channel_ids.split(",") if c.strip()
        )
        client = AerysDiscordClient(
            ask_fn=lambda text, identity, thread: ask(
                graph, text, identity=identity, thread_id=thread,
                router=router, action_graph=action_graph,
                deep_allowed=deep_gate,
            ),
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
