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
import re
import time
import uuid
from typing import Any, Callable

import httpx
from langchain_core.tools import tool

log = logging.getLogger(__name__)

WRITE_OPS = frozenset({"turn_on", "turn_off", "toggle", "set_brightness"})

# The ONLY string prefix a successful write returns — service.py's silent-success
# rule keys on it (a fast turn whose every tool note starts with this = the device
# visibly changed = skip the spoken follow-up). Change it here and nowhere else.
WRITE_OK_PREFIX = "Done:"
# A write HA accepted where a read-back shows the device was ALREADY in the
# requested state. Deliberately NOT WRITE_OK_PREFIX: nothing visibly changed, so
# the spoken follow-up should say so ("it was already off") instead of staying
# silent. (2026-09-04: replaces the old "NO state change — verify" note that
# cost the model a whole extra round-trip to read the state itself.)
NOOP_OK_PREFIX = "OK (already there):"
# Words that carry no target information in a room/device name.
_NAME_STOPWORDS = frozenset({"the", "all", "of", "in", "my", "every", "please"})
# Words that name a KIND of thing, not WHICH thing. A name made only of these
# ("the lights", "switches") would match every controllable entity — a
# house-wide write from an under-specified request. It becomes a question.
_GENERIC_TERMS = frozenset({"light", "lights", "lamp", "lamps", "switch", "switches", "plug", "plugs"})
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

# A transient transport failure (connection refused / DNS / timeout — a network
# blip or HA mid-restart) gets ONE quick retry after a short backoff before we
# fall back to the honest "unreachable" message. httpx.TransportError is the
# connect/timeout family ONLY — it excludes HTTPStatusError, so a real 404/500
# is never retried (that's a genuine answer, not a blip). Reads only: writes
# keep their single-shot + outbox path so a retry can never double-fire a toggle.
_READ_RETRY_BACKOFF_S = 0.6
# Read-back after a write HA reported no state change for: these lights (Tuya-class
# cloud devices) update their HA state a beat AFTER the service call returns —
# observed 9/04: the same lights answered "Done" for two bulbs and an empty
# changed-list for the other two in one command. So the read-back waits, and
# retries once, before calling a command dropped. Only the no-change path pays.
_VERIFY_DELAYS_S = (0.8, 1.5)


def _get_with_retry(http: httpx.Client, url: str, headers: dict) -> httpx.Response:
    """GET, retrying ONCE on a transient transport error (never on a status).

    On a momentary HA outage (today's SSD-reboot blip is the canonical case) the
    second attempt after a short backoff usually succeeds — so a brief restart
    no longer surfaces as an unreachable/degraded turn. If both attempts fail the
    final error propagates to the caller's existing graceful except-block.
    """
    try:
        return http.get(url, headers=headers)
    except httpx.TransportError:
        time.sleep(_READ_RETRY_BACKOFF_S)
        return http.get(url, headers=headers)  # final attempt; may raise → caught upstream


def _room_of(entity_id: str) -> str:
    """Best-effort place label from an entity id: the object-id words before the
    first generic term / trailing number. light.sunroom_light_1 -> "sunroom"."""
    obj = entity_id.split(".", 1)[-1]
    words = [w for w in obj.split("_") if w and not w.isdigit() and w not in _GENERIC_TERMS]
    return " ".join(words)


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

    def _resolve_targets(raw: str, op: str) -> tuple[list[str], str | None]:
        """WHAT the call is about: exact id, comma list of ids, or a room/device
        NAME matched against the controllable set. Returns (targets, problem).

        Why (2026-09-04 trace): "dim the sunroom lights by half" cost five model
        round-trips because the tool took ONE exact id per call — the model
        guessed ids, hit the allowlist refusal, then wrote four lights one by one.
        A name resolves here, against the allowlist, into ONE call."""
        raw = raw.strip()
        if "," in raw:
            ids = [e.strip() for e in raw.split(",") if e.strip()]
            bad = [e for e in ids if "." not in e]
            if bad:
                return [], (
                    f"'{', '.join(bad)}' aren't entity ids — a list must be exact ids "
                    "like light.a, light.b (or give ONE room/device name instead)."
                )
            return ids, None
        if "." in raw:
            return [raw], None
        terms = [
            w for w in re.split(r"[\s_\-]+", raw.lower()) if w and w not in _NAME_STOPWORDS
        ]
        if not terms:
            return [], "home_control needs an entity id or a room/device name."
        if all(w in _GENERIC_TERMS for w in terms):
            rooms = sorted({_room_of(e) for e in canary_entities} - {""})
            return [], (
                f"Which ones? '{raw}' names a kind of device, not a place. Ask the user "
                f"ONE short question naming the choices: {', '.join(rooms) or 'none configured'}."
            )

        # WHOLE-TOKEN matching (Codex review 9/04): "office" must not reach
        # light.office_closet_1 by substring. Tokens of the object id; a term
        # matches a token exactly (plural-stripped), and "sun room" also matches
        # "sunroom" via the joined non-generic terms.
        joined = "".join(w for w in terms if w not in _GENERIC_TERMS)

        def hit(eid: str) -> bool:
            toks = set(re.split(r"[._\-\s]+", eid.lower()))
            return all(w in toks or w.rstrip("s") in toks for w in terms) or (
                bool(joined) and joined in toks
                and all(w in toks or w.rstrip("s") in toks for w in terms if w in _GENERIC_TERMS)
            )

        matches = sorted(e for e in canary_entities if hit(e))
        if matches:
            log.info("home_control resolved %r -> %s", raw, matches)
            return matches, None
        allowed = ", ".join(sorted(canary_entities)) or "(none configured)"
        if op == "get_state":
            return [], (
                f"No controllable entity matches '{raw}'. To read other devices "
                "(cars, sensors, phones) call search_entities with that name to get "
                f"the exact id, then get_state with it. Entities I can control: {allowed}."
            )
        return [], (
            f"Refused: nothing I may control matches '{raw}'. "
            f"The entities I may control are: {allowed}."
        )

    def _read_state(entity: str) -> dict | str:
        """GET one entity's state: a dict, or an honest error STRING."""
        try:
            r = _get_with_retry(http, f"{base}/api/states/{entity}", headers)
            if r.status_code == 404:
                return f"Home Assistant has no entity named {entity}."
            r.raise_for_status()
            data = r.json()
            if not isinstance(data, dict):
                # Never raise inside a ToolNode (kills the turn): an odd body is an
                # honest string, and a read-back caller treats it as "unverified".
                return f"Home Assistant returned an unexpected reply for {entity}."
            attrs = data.get("attributes") or {}
            out = {
                "entity_id": entity,
                "state": data.get("state"),
                "friendly_name": attrs.get("friendly_name"),
            }
            # Lights report brightness 0-255; the model (and Chris) think in
            # percent — translate here so relative dimming math ("down by
            # half") never has to know about the 255 scale.
            if attrs.get("brightness") is not None:
                out["brightness_pct"] = round(attrs["brightness"] / 255 * 100)
            return out
        except httpx.HTTPError as e:
            return f"Home Assistant is unreachable right now ({e})."
        except (ValueError, TypeError) as e:
            # Malformed JSON / odd attribute types: honest string, never a raise.
            return f"Home Assistant returned an unreadable reply for {entity} ({e})."

    def _desired_phrase(op: str, pct: int | None) -> str:
        if op == "turn_off":
            return "off"
        if pct is not None:
            return f"on at {pct}%"
        return "on"

    def _verify_after_write(
        entities: list[str], op: str, pct: int | None
    ) -> tuple[list[str], list[str]]:
        """Read back entities HA reported no new state for: (already_there, not_applied).

        A verification that can FAIL: it reads the device, it does not trust the
        command. toggle has no knowable target state, so it can never verify."""
        if op == "toggle":
            return [], list(entities)

        def matches(e: str) -> bool:
            st = _read_state(e)
            if not isinstance(st, dict):
                return False
            want_on = op != "turn_off"
            ok = (st.get("state") == "on") == want_on
            if ok and want_on and pct is not None:
                ok = abs(int(st.get("brightness_pct") or 0) - pct) <= 2
            return ok

        already: list[str] = []
        pending = list(entities)
        for delay in _VERIFY_DELAYS_S:
            if not pending:
                break
            time.sleep(delay)
            still = []
            for e in pending:
                (already if matches(e) else still).append(e)
            pending = still
        return already, pending

    @tool
    def home_control(operation: str, entity_id: str, brightness_pct: int | None = None) -> str:
        """Control or inspect the smart home via Home Assistant.

        CALL THIS TOOL whenever the user asks to turn something on or off,
        toggle a device, dim/brighten a light, or asks whether a light/switch
        is currently on.

        operation: one of "get_state", "turn_on", "turn_off", "toggle",
        "set_brightness".
        entity_id: WHAT to control — any ONE of:
          - an exact entity id ("light.office_lamp", "switch.desk_fan");
          - a comma-separated list of exact ids (one call controls them all);
          - a ROOM or DEVICE NAME ("sunroom lights", "office", "sun room"):
            resolved against the controllable entities, and EVERY match is
            controlled in this ONE call. Use this for "the sunroom lights"
            style requests — do NOT call search_entities and do NOT loop
            over lights one by one.
          To READ something you can't control (a car, a sensor, a phone), call
          search_entities first to get the exact id, then get_state with it.
          Never invent an id.
        brightness_pct: 1-100, lights only. For an ABSOLUTE target ("half",
        "50%", "to 20 percent") call set_brightness with the number DIRECTLY —
        no get_state first. Only a RELATIVE ask ("a bit dimmer", "a little
        brighter", "down by half from where it is") needs get_state first: it
        returns the current brightness_pct; do the math, then set_brightness.
        For "off"/0%, use turn_off.

        get_state works on any entity. Writes only work on lights and switches
        on the beta allowlist — if the tool refuses, tell the user honestly and
        exactly why; NEVER claim a device changed state unless this tool said
        so. A reply starting "OK (already there)" means the device was already
        in the requested state — say that plainly.
        """
        op = operation.strip().lower()
        targets, problem = _resolve_targets(entity_id, op)
        if problem:
            return problem
        label = ", ".join(targets)

        # ---- reads: unrestricted (looking can't break anything) -------------
        if op == "get_state":
            results = [_read_state(e) for e in targets]
            if len(results) == 1:
                return results[0] if isinstance(results[0], str) else json.dumps(results[0])
            return json.dumps(results)

        if op not in WRITE_OPS:
            return (
                f"Unknown operation '{operation}'. "
                "Valid operations: get_state, turn_on, turn_off, toggle, set_brightness."
            )

        # ---- writes: domain gate, then canary gate — over EVERY target -------
        domains = {e.split(".", 1)[0] for e in targets}
        bad = sorted(d for d in domains if d not in WRITABLE_DOMAINS)
        if bad:
            return (
                f"Refused: writes to '{bad[0]}' entities aren't enabled yet — "
                "only lights and switches can be controlled in this beta."
            )
        not_allowed = [e for e in targets if e not in canary_entities]
        if not_allowed:
            # Honest refusal STRING back to the model — never an exception, and
            # never a lie. The model relays this so the caller learns the truth.
            allowed = ", ".join(sorted(canary_entities)) or "(none configured)"
            verb = "is" if len(not_allowed) == 1 else "are"
            return (
                f"Refused: {', '.join(not_allowed)} {verb} not on the beta write allowlist. "
                f"I can read state, but the only entities I may control are: {allowed}."
            )

        # ---- brightness: validate BEFORE any outbox row exists ---------------
        # set_brightness is sugar over HA's light/turn_on + brightness_pct —
        # there is no set_brightness service. Both ops share the same checks.
        wants_pct = op == "set_brightness" or (op == "turn_on" and brightness_pct is not None)
        if wants_pct:
            if op == "set_brightness" and brightness_pct is None:
                return (
                    "set_brightness needs brightness_pct (1-100). For a relative "
                    "change, call get_state first to read the current level."
                )
            if brightness_pct is not None and not 1 <= brightness_pct <= 100:
                return (
                    f"Refused: brightness_pct must be 1-100 (got {brightness_pct}). "
                    "For 0% / off, use turn_off instead."
                )
            if "light" not in domains:
                return (
                    f"Refused: brightness only applies to lights — {label} "
                    f"{'is a' if len(targets) == 1 else 'are'} {'/'.join(sorted(domains))}. "
                    "Use turn_on/turn_off for it."
                )
        # A name can span lights AND switches ("living room"). One HA call per
        # domain (Gemini review 9/04: a refusal here was a dead end — the model
        # can only address them by that same name). Brightness applies to the
        # lights; switches in the same name are left alone and SAID so.
        groups = {d: [e for e in targets if e.split(".", 1)[0] == d] for d in sorted(domains)}
        skipped: list[str] = []
        if wants_pct:
            skipped = [e for d, es in groups.items() if d != "light" for e in es]
            groups = {"light": groups["light"]}
        did = f"{op} {brightness_pct}%" if brightness_pct is not None else op
        done: list[str] = []
        already: list[str] = []
        dropped: list[str] = []
        for domain, group in groups.items():
            service = "turn_on" if wants_pct else op
            entity_field: Any = group if len(group) > 1 else group[0]
            body: dict[str, Any] = {"entity_id": entity_field}
            if wants_pct:
                body["brightness_pct"] = brightness_pct
            # ---- the audited write: intent -> HA -> receipt ------------------
            payload = {"operation": op, "entity_id": entity_field, "domain": domain}
            if brightness_pct is not None:
                payload["brightness_pct"] = brightness_pct
            outbox_id = _outbox_open(payload)
            try:
                # HA REST: POST /api/services/<domain>/<service> — the same
                # endpoint the V1 HTTP Request node hit, minus the workflow
                # around it. ONE call per domain: HA takes an entity_id list.
                r = http.post(
                    f"{base}/api/services/{domain}/{service}",
                    headers=headers,
                    json=body,
                )
                r.raise_for_status()
            except httpx.HTTPError as e:
                _outbox_close(outbox_id, "failed", error=str(e))
                return f"The {op} on {', '.join(group)} FAILED — Home Assistant said: {e}."
            # Receipt with evidence, not a bare ok (hands-contract rule 4): HA
            # returns the list of states the call changed — that's the proof.
            try:
                changed = r.json()
            except ValueError:
                changed = None
            changed_ids = (
                {c.get("entity_id") for c in changed if isinstance(c, dict)}
                if isinstance(changed, list) else set()
            )
            unverified = [e for e in group if e not in changed_ids]
            done += [e for e in group if e in changed_ids]
            # HA accepted the command (200) but reported no new state for some
            # target. Two honest readings: it was ALREADY in that state, or the
            # device dropped the command (observed live 2026-08-10 with a Tuya
            # light: "on at 30%" while it stayed dark). Instead of handing the
            # model a "go verify" note (a whole extra round-trip), READ THE
            # DEVICE BACK here and say which it was.
            g_already, g_dropped = (
                _verify_after_write(unverified, op, brightness_pct) if unverified else ([], [])
            )
            already += g_already
            dropped += g_dropped
            _outbox_close(
                outbox_id, "succeeded",
                receipt={
                    "status_code": r.status_code, "changed": changed,
                    "already_there": g_already, "not_applied": g_dropped,
                },
            )
        note = f" ({', '.join(skipped)} can't take a brightness — left as is.)" if skipped else ""
        if dropped:
            return (
                f"Sent {did} to {label}; Home Assistant accepted it (200) but "
                f"{', '.join(dropped)} reported NO state change and a read-back does not "
                "show the requested state — the device may not have applied it (cloud "
                "lights sometimes drop commands). Do not tell the user it definitely "
                f"happened.{note}"
            )
        if not done:
            verb = "was" if len(already) == 1 else "were"
            return (
                f"{NOOP_OK_PREFIX} {', '.join(already)} {verb} already "
                f"{_desired_phrase(op, brightness_pct)} — nothing to change.{note}"
            )
        extra = f" {', '.join(already)} already {_desired_phrase(op, brightness_pct)}." if already else ""
        return f"{WRITE_OK_PREFIX} {did} sent to {', '.join(done)} (HA responded 200).{extra}{note}"

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

        Call this BEFORE home_control's get_state when you lack the exact
        entity id of something you are READING (to CONTROL lights/switches,
        give home_control the room or device name directly — no search) —
        i.e. whenever the user names a device colloquially:
        a car name ("jolteon"), a sensor ("office temperature"), a person's phone,
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
            r = _get_with_retry(http, f"{base}/api/states", headers)
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
