"""Offline tests for wire_tracing() — the degrade-safe rule, proven.

No Phoenix, no network, no real OTel registration (instrumenting for real would
mutate global LangChain callback state and bleed into every other test). A bare
SimpleNamespace stands in for Settings — wire_tracing only reads .otlp_endpoint.
What these prove: unset endpoint = clean no-op, any setup explosion = logged False
(never a raise), and a working install path reports True.
"""

from types import SimpleNamespace

from aerys_v2 import tracing
from aerys_v2.tracing import wire_tracing


def test_no_endpoint_is_a_noop():
    # OTLP_ENDPOINT unset (the dev/test default) — feature structurally OFF
    assert wire_tracing(SimpleNamespace(otlp_endpoint=None)) is False


def test_failure_degrades_never_raises(monkeypatch, caplog):
    # THE rule under test: a broken tracing stack must not take the brain down.
    def boom(endpoint):
        raise RuntimeError("exporter exploded")

    monkeypatch.setattr(tracing, "_install", boom)
    result = wire_tracing(SimpleNamespace(otlp_endpoint="http://phoenix:6006/v1/traces"))
    assert result is False  # returned, didn't raise — serve path continues
    assert "continuing WITHOUT tracing" in caplog.text  # and it was loud about it


def test_successful_install_reports_true(monkeypatch):
    # Stub the wiring (real instrument() is global state we don't want in tests);
    # prove wire_tracing passes the endpoint through and reports armed.
    seen = []
    monkeypatch.setattr(tracing, "_install", lambda endpoint: seen.append(endpoint))
    assert wire_tracing(SimpleNamespace(otlp_endpoint="http://x:6006/v1/traces")) is True
    assert seen == ["http://x:6006/v1/traces"]


def test_real_install_path_imports_cleanly():
    # Smoke-check the actual dependency stack: if openinference/otel imports are
    # broken, better a test failure here than a silent "tracing off" on the Jetson.
    # (Import-only — we build nothing, register nothing.)
    from openinference.instrumentation.langchain import LangChainInstrumentor  # noqa: F401
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import (  # noqa: F401
        OTLPSpanExporter,
    )
