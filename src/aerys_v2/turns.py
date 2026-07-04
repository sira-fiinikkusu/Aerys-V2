"""v2_turns audit row construction — the writer's pure, testable half.

n8n mapping: this is the record every workflow execution left behind in the n8n
Executions tab, done properly and durably. `db/migrations/001_turns_and_outbox.sql`
defines the table; `service.py` fires one row per completed ask() turn through the
recorder seam wired in `factory.turn_recorder_for`; the ONLY reader today is
`workers/extraction.py` (V2_TURNS_SQL). Everything here is pure — no DB, no network,
no OTel required — so the row shape is unit-tested with fakes exactly like the rest
of the codebase (tools, resolver, router).

Two fields are LOAD-BEARING for the capability-request loop
(docs/capability-request-loop-design.md §1) and must stay STRUCTURED, never prose:

  - tool_calls : list[{name, ok: bool, error_class: str|null}] — the self-iteration
    loop fingerprints structural gap signals on (tool_name, error_class). A prose
    string here defeats the whole downstream feature (cross-review M1).
  - degraded   : list[str] of markers (e.g. ["ha_unreachable"]) — the high-trust
    structural signal the model cannot fake.

Both are emitted as JSON strings here (ready for the `::jsonb` cast in the INSERT),
so the recorder factory stays a dumb `conn.execute(sql, row)`.
"""

from __future__ import annotations

import json
import logging
import uuid
from typing import Any

log = logging.getLogger(__name__)

# The INSERT lives with the row builder so the columns and the dict keys can never
# drift apart. id + created_at are table-defaulted (GENERATED / DEFAULT now()).
# tool_calls / degraded ride in as JSON strings, cast to jsonb (same %s::jsonb
# convention as tools/home_control.py's outbox writes). person_id is cast to uuid;
# a None value passes as NULL::uuid, which is exactly the "cold caller" case.
INSERT_TURN_SQL = """\
INSERT INTO v2_turns
  (thread_id, channel, person_id, platform_identity, resolver_version,
   classifier_intent, tier, tier_override_source, guard_verdict,
   input_text, raw_reply, emitted_reply, tool_calls, degraded,
   error, latency_ms, trace_id)
VALUES
  (%(thread_id)s, %(channel)s, %(person_id)s::uuid, %(platform_identity)s,
   %(resolver_version)s, %(classifier_intent)s, %(tier)s, %(tier_override_source)s,
   %(guard_verdict)s, %(input_text)s, %(raw_reply)s, %(emitted_reply)s,
   %(tool_calls)s::jsonb, %(degraded)s::jsonb, %(error)s, %(latency_ms)s,
   %(trace_id)s)
"""


def _is_uuid(value: object) -> bool:
    """person_id must be a real UUID before it goes anywhere near `::uuid` SQL.

    Same rule as services.context._is_uuid: transports mint non-UUID identities
    ("cli-operator", "discord:12345") until the DB resolver maps them to a real
    persons.id. A resolved user_id IS the UUID; a cold one is a "platform:id"
    handle. This split is how the row separates person_id (UUID column) from
    platform_identity (the audit handle).
    """
    try:
        uuid.UUID(str(value))
        return True
    except (ValueError, AttributeError, TypeError):
        return False


# Channel taxonomy keyed off the checkpointer thread_id prefix — the one field
# every transport already sets. Order matters: 'voice' first because the voice
# thread ("voice:beta") is what arms ack-then-act in service.py. Longest/most
# specific prefixes before their shorter parents.
def derive_channel(thread_id: str) -> str:
    """thread_id -> the migration's channel enum ('discord_dm'|'guild'|'voice'|'cli'|...).

    Mirrors the thread_key builders in the transports (discord_gateway.thread_key,
    telegram_gateway.telegram_thread_key, http_api's 'voice:beta'/'http:default',
    cli's 'cli'). Unknown shapes degrade to the prefix before the first ':' rather
    than raising — channel is NOT NULL, so this must always return something.
    """
    tid = str(thread_id or "")
    lowered = tid.lower()
    if lowered.startswith("voice"):
        return "voice"
    if lowered.startswith("discord:dm"):
        return "discord_dm"
    if lowered.startswith("discord:guild"):
        return "guild"
    if lowered.startswith("telegram:dm"):
        return "telegram_dm"
    if lowered.startswith("telegram:group"):
        return "telegram_group"
    if lowered.startswith("telegram"):
        return "telegram"
    if lowered.startswith("cli"):
        return "cli"
    if lowered.startswith("http"):
        return "http"
    head = tid.split(":", 1)[0].strip()
    return head or "unknown"


# ── tool-call classification ────────────────────────────────────────────────
# The action-stack tools (tools/home_control.py, media.py, web_search.py) share
# ONE contract by design: they NEVER raise (an exception inside a ToolNode kills
# the whole action turn — the V1 failed-webhook-kills-execution outage). So a real
# infrastructure failure comes back as an honest STRING, not a ToolMessage with
# status='error'. That means status alone can't tell a good call from a failed one
# for these tools — we also match a small, curated set of failure sentinels these
# tools emit verbatim. Coupling note: these substrings track the tools' failure
# wording. It degrades SAFELY — a reworded failure under-reports (a missed gap),
# never over-reports, and the status=='error' path still catches any tool that
# genuinely raises. Refusals ("Refused: ... not on the allowlist") and empty
# results ("returned no results") are NOT failures: the tool did its job, so they
# stay ok=True and surface (if at all) via the complaint/reply-phrase detector.
_TOOL_FAILURE_SIGNALS: tuple[tuple[str, str], ...] = (
    ("failed — home assistant", "ha_write_failed"),  # home_control write POST failed
    ("home assistant has no entity named", "no_entity"),  # home_control write can't-fulfil
    ("home assistant is unreachable", "unreachable"),     # home_control / search read
    ("vision service is unreachable", "unreachable"),     # media analyze_image
    ("summarization service is unreachable", "unreachable"),  # media youtube_summary
    ("search service is unreachable", "unreachable"),     # web_search
    ("web search failed", "search_failed"),
    ("failed with http", "http_error"),                   # media doc / youtube HTTP >=400
    ("couldn't fetch the document", "fetch_failed"),
    ("couldn't extract its text", "extract_failed"),
    ("came back malformed", "malformed_response"),
    ("returned a malformed response", "malformed_response"),
)

# When a tool genuinely RAISES (status=='error') the ToolNode-caught exception
# message is machine-set (not an attacker payload), so we can refine the coarse
# 'exception' class into a cause the capability loop can fingerprint distinctly
# (cross-review sharp-3 M: a timeout and an auth error must not merge). Order is
# most-specific-first; anything unmatched stays the honest 'exception'.
_EXCEPTION_CAUSE_SIGNALS: tuple[tuple[str, str], ...] = (
    ("timed out", "timeout"),
    ("timeout", "timeout"),
    ("connection refused", "unreachable"),
    ("connection reset", "unreachable"),
    ("unreachable", "unreachable"),
    ("name or service not known", "unreachable"),
    ("unauthorized", "auth_error"),
    ("forbidden", "auth_error"),
    ("401", "auth_error"),
    ("403", "auth_error"),
    ("jsondecode", "malformed_response"),
    ("expecting value", "malformed_response"),
)

# Turn-level degraded markers derived from the SAME tool contents — coarser than
# the per-tool error_class, and named per-subsystem so the capability loop can key
# on the marker directly (the migration's canonical example is ["ha_unreachable"]).
_DEGRADED_SIGNALS: tuple[tuple[str, str], ...] = (
    # A reachable-but-refusing HA (4xx write rejection) is a STRUCTURALLY different
    # gap from an unreachable HA — keep them as distinct markers so the capability
    # loop doesn't fingerprint a permissions/allowlist problem as a connectivity one
    # (cross-review sharp-3 L).
    ("failed — home assistant", "ha_write_failed"),
    ("home assistant is unreachable", "ha_unreachable"),
    ("vision service is unreachable", "vision_unreachable"),
    ("summarization service is unreachable", "summarizer_unreachable"),
    ("search service is unreachable", "search_unreachable"),
)


def _first_line(content: object) -> str:
    """The tools' honest failure strings are single-line and lead the content — a
    successful payload prepends nothing to them. Scanning ONLY the first line is the
    anti-forgery boundary (cross-review sharp-3 H): read_document success prefixes
    'Contents of <file>:\\n\\n<body>' and web_search success is '\\n'.join(results),
    so third-party/attacker text riding in those payloads lands on line 2+ and can
    never be substring-matched into a forged failure. Every real tool failure keeps
    its sentinel on line 1, so true-positive detection is unchanged."""
    return str(content).split("\n", 1)[0].lower()


def classify_tool_result(content: str, status: object) -> str | None:
    """None when the call succeeded; an error_class string when it failed.

    status=='error' is LangChain's authoritative signal that ToolNode caught an
    exception (the tool raised) — highest trust; the exception message is machine-set
    so we refine it into a cause. Otherwise scan the FIRST LINE of the returned string
    for the never-raise tools' curated failure sentinels (see _first_line — a payload's
    echoed third-party text can't forge a failure). Anything else (a normal answer, a
    policy refusal, an empty result) is a success.
    """
    if status == "error":
        low = _first_line(content)
        for needle, error_class in _EXCEPTION_CAUSE_SIGNALS:
            if needle in low:
                return error_class
        return "exception"
    low = _first_line(content)
    for needle, error_class in _TOOL_FAILURE_SIGNALS:
        if needle in low:
            return error_class
    return None


def extract_tool_calls(messages: list) -> list[dict]:
    """Structured per-tool record from an action subgraph's message list.

    One entry per ToolMessage (the executed call), in execution order:
    {name, ok: bool, error_class: str|null}. The name prefers the ToolMessage's
    own .name and falls back to the requesting AIMessage.tool_calls entry (mapped
    by tool_call_id) — belt-and-braces, since .name has been None across
    langchain-core versions. Duck-typed on `type`/attributes so fakes and every
    message class work without importing concrete langchain types.
    """
    id_to_name: dict[str, str] = {}
    for m in messages:
        for tc in (getattr(m, "tool_calls", None) or []):
            tid = tc.get("id") if isinstance(tc, dict) else getattr(tc, "id", None)
            name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", None)
            if tid and name:
                id_to_name[tid] = name

    calls: list[dict] = []
    for m in messages:
        if getattr(m, "type", "") != "tool":
            continue
        content = m.content if isinstance(getattr(m, "content", None), str) else str(getattr(m, "content", ""))
        name = (
            getattr(m, "name", None)
            or id_to_name.get(getattr(m, "tool_call_id", None))
        )
        if not name:
            # An unresolved tool name silently collapses distinct tools into one
            # (tool_error, unknown) fingerprint — make the blindness VISIBLE in logs
            # rather than letting the loop conflate subsystems (cross-review sharp-3 L).
            log.warning(
                "v2_turns: tool name unresolved (no .name, id=%r not in AIMessage map) "
                "— fingerprint collapses to 'unknown'",
                getattr(m, "tool_call_id", None),
            )
            name = "unknown"
        error_class = classify_tool_result(content, getattr(m, "status", None))
        calls.append({"name": name, "ok": error_class is None, "error_class": error_class})
    return calls


def degraded_markers(messages: list, extra: list[str] | None = None) -> list[str]:
    """Turn-level degraded markers — subsystem markers mined from tool contents
    (e.g. 'ha_unreachable') PLUS any caller-supplied markers (deep-cap downgrade,
    wall-clock overrun, a failed background action). Deduped, insertion order
    preserved so the highest-signal marker leads.
    """
    ordered: list[str] = []

    def _add(marker: str) -> None:
        if marker and marker not in ordered:
            ordered.append(marker)

    for marker in (extra or []):
        _add(marker)
    for m in messages:
        if getattr(m, "type", "") != "tool":
            continue
        low = _first_line(getattr(m, "content", ""))  # same anti-forgery boundary
        for needle, marker in _DEGRADED_SIGNALS:
            if needle in low:
                _add(marker)
    return ordered


def build_turn_row(
    *,
    thread_id: str,
    identity: dict | None,
    input_text: str,
    latency_ms: int | None,
    classifier_intent: str | None = None,
    tier: str | None = None,
    tier_override_source: str | None = None,
    guard_verdict: str | None = None,
    raw_reply: str | None = None,
    emitted_reply: str | None = None,
    messages: list | None = None,
    extra_degraded: list[str] | None = None,
    error: str | None = None,
    trace_id: str | None = None,
    resolver_version: str | None = None,
) -> dict:
    """Assemble the parameter dict for INSERT_TURN_SQL — the whole row, one place.

    person_id vs platform_identity: identity['user_id'] is EITHER a resolved
    persons.id (a UUID) OR a cold "platform:id" handle. A UUID lands in person_id
    (the column extraction/the capability loop filter on); anything else lands in
    platform_identity as the audit handle, with person_id NULL. That keeps the
    UUID-typed column clean while still recording who the cold caller was.
    """
    messages = messages or []
    identity = identity or {}
    user_id = str(identity.get("user_id") or "").strip()
    is_uuid = _is_uuid(user_id)
    person_id = user_id if is_uuid else None
    platform_identity = None if (is_uuid or not user_id) else user_id

    return {
        "thread_id": str(thread_id or ""),
        "channel": derive_channel(thread_id),
        "person_id": person_id,
        "platform_identity": platform_identity,
        "resolver_version": resolver_version,
        "classifier_intent": classifier_intent,
        "tier": tier,
        "tier_override_source": tier_override_source,
        "guard_verdict": guard_verdict,
        "input_text": input_text,
        "raw_reply": raw_reply,
        "emitted_reply": emitted_reply,
        # STRUCTURED, never prose — see module docstring / capability-loop design.
        "tool_calls": json.dumps(extract_tool_calls(messages)),
        "degraded": json.dumps(degraded_markers(messages, extra_degraded)),
        "error": error,
        "latency_ms": latency_ms,
        "trace_id": trace_id,
    }


# ── trace_id capture ────────────────────────────────────────────────────────
# Degrade-safe, exactly like service._TRACER: reading the current span is a
# passenger, never the driver. If OTel isn't installed or no span is active,
# trace_id is simply NULL and the row still writes.
try:  # pragma: no cover - import guard
    from opentelemetry import trace as _otel_trace
except Exception:  # pragma: no cover
    _otel_trace = None  # type: ignore[assignment]


def current_trace_id() -> str | None:
    """The active Phoenix/OTel trace id as 32 lowercase hex (schema note: trace_id
    joins to Phoenix once tracing lands, 01-05). None when tracing is off or no
    span is current — this must NEVER raise into the turn."""
    if _otel_trace is None:
        return None
    try:
        ctx = _otel_trace.get_current_span().get_span_context()
        if ctx is None or not ctx.is_valid:
            return None
        return format(ctx.trace_id, "032x")
    except Exception:  # pragma: no cover - defensive
        return None
