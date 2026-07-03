"""home_control — Tool v1: the Brain's first native write capability.

n8n mapping: this replaces workflow 07-01 "HA Action: Play Music (owner-gated)"
as the pattern-setter — an HTTP Request node to Home Assistant's REST API, done
as a LangChain tool the action subgraph can call. The function is literally
named `home_control` because of the V1 toolWorkflow lesson: the tool name the
LLM sees MUST match what prompts call it, or the model hallucinates having
called it.

Three safety layers, outermost first:

1. CANARY ALLOWLIST — writes are refused (honest error string back to the
   model, so it can tell the caller the truth) for any entity not in
   ha_canary_entities. Reads are unrestricted: looking at a state can't break
   anything. This is the same crawl-walk-run gate as the V1 owner-gated action.
2. OUTBOX-INLINE — every write that reaches HA is recorded in v2_outbox
   (INSERT intent as 'executing' -> call HA -> UPDATE receipt/status), the
   write-ahead pattern from db/migrations/001. A crash mid-call leaves an
   'executing' row the sweeper can reconcile — never a silent mystery write.
3. HONEST FAILURE — HA unreachable / 4xx / refused all come back as plain
   error strings the model must relay. Never raise out of the tool: an
   exception inside a ToolNode kills the whole action turn (the V1
   failed-webhook-kills-execution outage mode, again).
"""

import json
import logging
import uuid
from typing import Any, Callable

import httpx
from langchain_core.tools import tool

log = logging.getLogger(__name__)

WRITE_OPS = frozenset({"turn_on", "turn_off", "toggle"})

# The ONLY string prefix a successful write returns — service.py's silent-success
# rule keys on it (a fast turn whose every tool note starts with this = the device
# visibly changed = skip the spoken follow-up). Change it here and nowhere else.
WRITE_OK_PREFIX = "Done:"
# v1 scope: only domains where a misfire is an annoyance, not a hazard.
# (No locks, no covers, no climate — those arrive with confirmation semantics.)
WRITABLE_DOMAINS = frozenset({"light", "switch"})

# The connection seam: a zero-arg callable returning a DB connection usable as
# a context manager (psycopg.connect in prod, a fake in tests). None = no
# database_url = the outbox layer is simply absent (spike/dev boxes).
ConnFactory = Callable[[], Any]

# search_entities knobs: enough matches to disambiguate, few enough to not
# blow the tool-message budget; long states (weather blobs) get elided.
SEARCH_LIMIT = 15
STATE_TRUNCATE_AT = 60
# HA's "nothing home" states — noise in a discovery listing, filtered unless
# they're literally all we found (then honesty beats tidiness: show them).
DEAD_STATES = frozenset({"unavailable", "unknown"})


def canary_set(csv: str) -> frozenset[str]:
    """Parse the HA_CANARY_ENTITIES csv into the allowlist set ('' -> empty)."""
    return frozenset(e.strip() for e in csv.split(",") if e.strip())


def build_home_control_tool(
    *,
    base_url: str,
    token: str,
    canary_entities: frozenset[str],
    client: httpx.Client | None = None,
    conn_factory: ConnFactory | None = None,
):
    """Close over the config and return the LangChain tool object.

    Everything injectable, same seam philosophy as the checkpointer: tests pass
    an httpx.Client on a MockTransport and a fake conn_factory; --serve passes
    the real things. The tool NEVER reads Settings — construction knows config,
    behavior doesn't.
    """
    base = base_url.rstrip("/")
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    http = client or httpx.Client(timeout=10.0)

    def _outbox_open(payload: dict) -> int | None:
        """INSERT the write-ahead intent row; returns its id (None = outbox off/failed).

        Lease check lives here because it must be decided BEFORE the intent row
        is written — the exception marker travels IN the payload, auditable.

        ── ONE-ARMED-WRITER EXCEPTION (deliberate, bounded, beta only) ─────────
        The v2_writer_lease rule (design doc 2026-07-02-turns-outbox-spine.md,
        cross-review #11) says a write capability REFUSES unless it holds the
        lease for its kind. During the voice beta, ha_write's lease still says
        'n8n' — but the callers reaching THIS tool are satellite-scoped voice
        threads that n8n never serves, so a double-fire is structurally
        impossible for this slice. Owner-ratified: execute anyway, but mark the
        payload {"lease_exception": "beta-canary"} so every such write is
        queryable (SELECT .. WHERE payload ? 'lease_exception') and the
        exception dies loudly when the lease flips to 'brain' and the marker
        stops appearing. This is the ONLY capability allowed to bend the rule.
        """
        if conn_factory is None:
            return None
        try:
            with conn_factory() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT holder FROM v2_writer_lease WHERE kind = 'ha_write'"
                    )
                    row = cur.fetchone()
                    holder = row[0] if row else None
                    if holder != "brain":
                        payload["lease_exception"] = "beta-canary"
                    cur.execute(
                        "INSERT INTO v2_outbox (kind, payload, idempotency_key, status) "
                        "VALUES ('ha_write', %s::jsonb, %s, 'executing') RETURNING id",
                        (json.dumps(payload), str(uuid.uuid4())),
                    )
                    return cur.fetchone()[0]
        except Exception:
            # Audit trouble must not cost the turn (the graceful contract) —
            # but an unaudited write is a real event, so it logs loudly.
            log.warning("outbox INSERT failed — HA write proceeds unaudited", exc_info=True)
            return None

    def _outbox_close(
        outbox_id: int | None, status: str, receipt: dict | None = None, error: str | None = None
    ) -> None:
        """UPDATE the intent row with what actually happened (receipt = evidence).

        Separate short connection from _outbox_open on purpose: holding a conn
        across the HA HTTP call buys nothing, and a crash between the two
        leaves exactly the 'executing' row the sweeper contract expects.
        """
        if outbox_id is None:
            return
        try:
            with conn_factory() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "UPDATE v2_outbox SET status = %s, receipt = %s::jsonb, "
                        "last_error = %s, attempts = attempts + 1, updated_at = now() "
                        "WHERE id = %s",
                        (
                            status,
                            json.dumps(receipt) if receipt is not None else None,
                            error,
                            outbox_id,
                        ),
                    )
        except Exception:
            log.warning("outbox UPDATE failed for row %s", outbox_id, exc_info=True)

    @tool
    def home_control(operation: str, entity_id: str) -> str:
        """Control or inspect the smart home via Home Assistant.

        CALL THIS TOOL whenever the user asks to turn something on or off,
        toggle a device, or asks whether a light/switch is currently on.

        operation: one of "get_state", "turn_on", "turn_off", "toggle".
        entity_id: the full Home Assistant entity id, e.g. "light.office_lamp"
        or "switch.desk_fan". This must be EXACT — if you do not already know
        the exact entity id, call the search_entities tool FIRST to find it;
        never guess an entity id.

        get_state works on any entity. Writes only work on lights and switches
        on the beta allowlist — if the tool refuses, tell the user honestly;
        NEVER claim a device changed state unless this tool said so.
        """
        op = operation.strip().lower()
        entity = entity_id.strip()

        # ---- reads: unrestricted (looking can't break anything) -------------
        if op == "get_state":
            try:
                r = http.get(f"{base}/api/states/{entity}", headers=headers)
                if r.status_code == 404:
                    return f"Home Assistant has no entity named {entity}."
                r.raise_for_status()
                data = r.json()
                attrs = data.get("attributes") or {}
                return json.dumps(
                    {
                        "entity_id": entity,
                        "state": data.get("state"),
                        "friendly_name": attrs.get("friendly_name"),
                    }
                )
            except httpx.HTTPError as e:
                return f"Home Assistant is unreachable right now ({e})."

        if op not in WRITE_OPS:
            return (
                f"Unknown operation '{operation}'. "
                "Valid operations: get_state, turn_on, turn_off, toggle."
            )

        # ---- writes: domain gate, then canary gate --------------------------
        domain = entity.split(".", 1)[0]
        if domain not in WRITABLE_DOMAINS:
            return (
                f"Refused: writes to '{domain}' entities aren't enabled yet — "
                "only lights and switches can be controlled in this beta."
            )
        if entity not in canary_entities:
            # Honest refusal STRING back to the model — never an exception, and
            # never a lie. The model relays this so the caller learns the truth.
            allowed = ", ".join(sorted(canary_entities)) or "(none configured)"
            return (
                f"Refused: {entity} is not on the beta write allowlist. "
                f"I can read its state, but the only entities I may control are: {allowed}."
            )

        # ---- the audited write: intent -> HA -> receipt ----------------------
        payload = {"operation": op, "entity_id": entity, "domain": domain}
        outbox_id = _outbox_open(payload)
        try:
            # HA REST: POST /api/services/<domain>/<service> — the same endpoint
            # the V1 HTTP Request node hit, minus the workflow around it.
            r = http.post(
                f"{base}/api/services/{domain}/{op}",
                headers=headers,
                json={"entity_id": entity},
            )
            r.raise_for_status()
        except httpx.HTTPError as e:
            _outbox_close(outbox_id, "failed", error=str(e))
            return f"The {op} on {entity} FAILED — Home Assistant said: {e}."
        # Receipt with evidence, not a bare ok (hands-contract rule 4): HA
        # returns the list of states the call changed — that's the proof.
        try:
            changed = r.json()
        except ValueError:
            changed = None
        _outbox_close(
            outbox_id, "succeeded", receipt={"status_code": r.status_code, "changed": changed}
        )
        return f"{WRITE_OK_PREFIX} {op} sent to {entity} (HA responded {r.status_code})."

    return home_control


def build_search_entities_tool(
    *,
    base_url: str,
    token: str,
    client: httpx.Client | None = None,
):
    """Close over the config and return the READ-ONLY entity discovery tool.

    Why this exists (observed live, 2026-07-03): home_control's get_state needs
    an EXACT entity id. Asked "what is jolteon's charge level?", the model
    guessed ids, got 404s, and had to ask the user — a discovery gap, not a
    reasoning gap. This tool is a fuzzy index over GET /api/states so the model
    can find the id itself. No canary allowlist on purpose: the allowlist gates
    WRITES; listing names and states can't break anything.
    """
    base = base_url.rstrip("/")
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    http = client or httpx.Client(timeout=10.0)

    @tool
    def search_entities(query: str) -> str:
        """Find Home Assistant entity ids by name. READ-ONLY — never changes anything.

        ALWAYS call this BEFORE home_control's get_state when you lack the
        exact entity id — i.e. whenever the user names a device colloquially:
        a car name ("jolteon"), a room ("the office light"), a person's phone,
        a nickname. NEVER guess an entity id — guesses 404; searching works.

        query: one or more words to match, e.g. "jolteon battery" or
        "office lamp". Matches entity ids and friendly names,
        case-insensitive. Returns up to 15 matches, one per line:
        "entity_id | friendly_name | state" (units included when known).
        Then call home_control get_state with the exact entity_id you picked,
        or answer directly from the state shown here.
        """
        terms = [t for t in query.lower().split() if t]
        if not terms:
            return "search_entities needs at least one word to search for."
        try:
            r = http.get(f"{base}/api/states", headers=headers)
            r.raise_for_status()
            states = r.json()
        except (httpx.HTTPError, ValueError) as e:
            # Honest failure string, never a raise — same ToolNode contract
            # as home_control (an exception kills the whole action turn).
            return f"Home Assistant is unreachable right now ({e})."

        scored: list[tuple[int, str, str, dict]] = []
        for item in states:
            entity = item.get("entity_id") or ""
            attrs = item.get("attributes") or {}
            friendly = str(attrs.get("friendly_name") or "")
            haystack = f"{entity} {friendly}".lower()
            hits = sum(1 for t in terms if t in haystack)
            if hits:
                scored.append((hits, entity, friendly, item))
        if not scored:
            return f"No Home Assistant entities match '{query}'."

        # Rank: most query terms matched first, then entity_id for stability.
        scored.sort(key=lambda s: (-s[0], s[1]))
        live = [s for s in scored if str(s[3].get("state")) not in DEAD_STATES]
        picked = (live or scored)[:SEARCH_LIMIT]

        lines = []
        for _, entity, friendly, item in picked:
            state = str(item.get("state"))
            if len(state) > STATE_TRUNCATE_AT:
                state = state[:STATE_TRUNCATE_AT] + "…"
            # Battery/temperature sensors are meaningless without the unit —
            # "78" vs "78 %" is exactly the EV6 charge-level use case.
            unit = (item.get("attributes") or {}).get("unit_of_measurement")
            if unit:
                state = f"{state} {unit}"
            lines.append(f"{entity} | {friendly or '(no name)'} | {state}")
        return "\n".join(lines)

    return search_entities
