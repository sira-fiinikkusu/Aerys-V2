"""wire_tracing() — Phoenix/OTel observability, wired once at startup (dossier 01-05).

n8n mapping: this is the Executions tab, done properly. In n8n every node run was
automatically visible — inputs, outputs, timings — because the engine recorded them.
LangGraph gives us none of that for free: without tracing, a bad reply is a black box
("which prompt actually went to the model? which tool fired?"). OpenInference hooks
LangChain's callback system and ships every model call / graph step as OTel spans to
Phoenix — the same per-node visibility, but queryable and with token counts.

THE DEGRADE-SAFE RULE: tracing is a passenger, never the driver. If Phoenix is down,
the endpoint is wrong, or a library import explodes — the brain must still serve.
Every failure path here logs and returns False; wire_tracing() can NEVER raise.
(Mirror of the n8n lesson: the 06-03 Central Error Handler observed failures, it
never caused them.)
"""

import logging

log = logging.getLogger("aerys_v2.tracing")


def _install(endpoint: str) -> None:
    """The actual OTel wiring — separated so wire_tracing() can wrap ALL of it in
    the degrade-safe try/except (and so tests can monkeypatch a boom right here).

    Imports live inside the function on purpose: if the openinference/otel packages
    are missing or broken, the cost is a logged warning, not an ImportError that
    kills the whole process at module import time.
    """
    from openinference.instrumentation.langchain import LangChainInstrumentor
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    # service.name is what Phoenix shows as the project/source — "aerys-v2" groups
    # every span from this brain, so a future second service won't blur into it.
    provider = TracerProvider(resource=Resource.create({"service.name": "aerys-v2"}))
    # BatchSpanProcessor exports in the background off the hot path — a slow or dead
    # Phoenix costs dropped spans, never latency on the voice turn (~3.6s budget).
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint)))
    # One instrument() call covers EVERYTHING: LangGraph runs on LangChain runnables,
    # so graph steps, model calls, and (later, 01-03+) tool calls all emit spans —
    # no per-node wiring, which is exactly what ask()-as-the-single-seam buys us.
    LangChainInstrumentor().instrument(tracer_provider=provider)


def wire_tracing(settings) -> bool:
    """Arm Phoenix tracing if OTLP_ENDPOINT is configured. Returns True when live.

    Same arming pattern as the Discord transport (config.py): the field is None →
    the feature is structurally OFF, nothing is imported, nothing connects. Set
    OTLP_ENDPOINT in .env (see deploy/phoenix.md for the value) and restart.
    """
    if settings.otlp_endpoint is None:
        return False  # not configured — silent no-op, the common dev/test case
    try:
        _install(settings.otlp_endpoint)
    except Exception:
        # The degrade-safe rule: ANY failure (bad endpoint, missing package,
        # exporter bug) is logged loudly and swallowed. Tracing never takes the
        # brain down.
        log.exception("tracing setup failed — continuing WITHOUT tracing")
        return False
    log.info("tracing armed | otlp=%s", settings.otlp_endpoint)
    return True
