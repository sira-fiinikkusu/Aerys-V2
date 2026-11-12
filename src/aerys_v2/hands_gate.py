"""What waits for his tap: what her voice may start at home (D10) and her first message to anyone
new (D12). Board #176, Stage 4.

The owner's rulings:
  D10 (2026-11-08): "Voice can start anything text can but requires me to approve via discord
      button. Else read only jobs". His text starts any home job. From his voice, on a strong
      match, she may start a READ-ONLY job directly; anything that could write, and any voice turn
      the recognizer wasn't sure was him, waits for his Approve button. Guests and voices that
      aren't his are refused.
  D17 (2026-11-09), the details: his typed words and his voice on a strong match stop a job
      directly; a voice it wasn't sure of may check on and list his jobs, but stopping one waits
      for his button ("it could be someone else in the house, and a cancel can end a long job"). His own lines that
      carry his id through /ask (the G2 lens, the panel, Sticky, Kael's line) can't show he typed
      it: starting, changing or stopping a job from them waits for his button instead of a no.
      After he declines, the same request isn't put to him again that day (Eastern) unless he
      raises it himself ("keep it adaptive. She can resurface it later if it makes sense to ask
      again"); from the next day she may.
  D12 (2026-11-09): she may message people he's told her about; the first message to each waits
      for his button, which shows exactly who and where ("Sam · Discord: sam_42"), so his tap
      confirms the person too. An address that matches nobody, or more than one person: she asks
      him first, and nothing is sent.

The button is approvals.py. What it guards is a row in v2_gated_actions (db/migrations/023): the
exact job or message, which the watcher here does once, and only after his Approve. She never
does it herself after the tap (asking again only tells her where it stands), so nothing happens
twice, and a declined, expired or withdrawn request is dropped and she's told.

"Read-only" is never her word for it: it's the home agent she picked, and only agents listed in
home_read_only_agents count, which should list one only once the home body enforces read-only
for it. Until one is listed, every voice-started job waits for the button.

Fail closed, everywhere: an asker it can't place is refused; a gate that can't write the request,
or read the answer, says no; a request nothing could carry out (no executor wired for it) is
refused, never queued. The job's words and the message are the private part: never logged.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Callable
from zoneinfo import ZoneInfo

from aerys_v2.approvals import DEFAULT_FOR, DETAIL_CHARS, approvals_for, person_id_for
from aerys_v2.state import VOICE_OWNER_ASSUMED_KEY, identity_from_config, is_lens_surface, is_voice_turn

log = logging.getLogger(__name__)

# Who is asking. OWNER_TEXT: him, typing, on Discord or Telegram (the doors that resolve the
# account). OWNER_VOICE: his voice on a strong match. VOICE_UNSURE: a voice turn that is his only by
# default (untagged, or the recognizer unsure). OWNER_ASK: his id through /ask without voice (the
# G2 lens, the panel, Sticky, Kael's line): his own infrastructure, but not proof he typed it.
# OTHER: everyone and everything else.
OWNER_TEXT, OWNER_VOICE, VOICE_UNSURE, OWNER_ASK, OTHER = (
    "owner_text", "owner_voice", "voice_unsure", "owner_ask", "other")
MEMBER, MEMBER_VOICE = "member", "member_voice"
#: Whoever this is, it's him raising it himself: never held by an earlier no (D17 Q5).
HIMSELF = (OWNER_TEXT, OWNER_VOICE)
TEXT_PLATFORMS = ("discord", "telegram")

# deepagents' async-subagent tools (AsyncSubAgentMiddleware): the two that set a home job working
# are gated by D10; stopping one is gated by D17 Q3; checking and listing only read his jobs, so
# they're his any way he asks.
START, UPDATE, CANCEL = "start_async_task", "update_async_task", "cancel_async_task"
READS = ("check_async_task", "list_async_tasks")
HOME_TOOLS = (START, UPDATE, *READS, CANCEL)
EASTERN = ZoneInfo("America/New_York")

HOME_JOB, MESSAGE = "home_job", "message"
PLATFORMS = {"discord": "Discord", "telegram": "Telegram", "email": "email"}
OUTCOME_CHARS = 500
TICK_S = 5

REFUSED_HOME = ("Not done: only Chris can start or change work on my home machine, and this didn't "
                "come from him (a guest, a voice that isn't his, or a line that isn't his own).")
CANT_ASK_HOME = "Not started: I can't put a home job to Chris for his OK yet, so from voice it waits until he types it."
CANT_ASK_MESSAGE = "Not sent: I can't put a first message to Chris for his OK yet, so it waits until I can."
GATE_DOWN = "Not done: I couldn't check whether this needs Chris's OK just now, so nothing happened. Try again in a moment."


@dataclass(frozen=True)
class Verdict:
    go: bool
    text: str = ""


GO = Verdict(True)


@dataclass
class GatedAction:
    id: int
    approval_id: int
    action: str
    base_key: str
    attempt: int
    thread_id: str
    args: dict = field(default_factory=dict)
    status: str = "waiting"
    created_at: datetime | None = None
    finished_at: datetime | None = None
    outcome: str | None = None


def asker(identity, thread_id, owner_person_id, members=None) -> str:
    """Who is asking, from what the doors already decided (never re-judged here)."""
    ident = identity if isinstance(identity, dict) else {}
    person = str(ident.get("user_id") or "")
    if person in (members or {}):
        if is_voice_turn(ident, thread_id):
            speaker = ident.get("speaker") or {}
            strong = (speaker.get("id") not in (None, "", "unknown")
                      and not speaker.get("demoted") and not speaker.get("unsure"))
            return MEMBER_VOICE if strong else VOICE_UNSURE
        return MEMBER if str(ident.get("platform") or "").lower() == "discord" else OTHER
    if not owner_person_id or person != str(owner_person_id):
        return OTHER
    if is_voice_turn(ident, thread_id):
        spk = ident.get("speaker")
        sid = str(spk.get("id") or "").strip().lower() if isinstance(spk, dict) else ""
        strong = (bool(sid) and sid != "unknown" and not spk.get("demoted")
                  and not ident.get(VOICE_OWNER_ASSUMED_KEY) and not spk.get("unsure")
                  and spk.get("band") != "unsure")
        return OWNER_VOICE if strong else VOICE_UNSURE
    if str(ident.get("platform") or "").strip().lower() in TEXT_PLATFORMS:
        return OWNER_TEXT
    return OWNER_ASK        # his id, no voice, no Discord/Telegram account: an /ask caller (D17 Q4)


def _line(identity, thread_id) -> str:
    """Which of his own lines an /ask turn came on, in his words, for his button."""
    if str(thread_id or "").startswith("kael:"):
        return "Kael's line"
    if is_lens_surface(identity if isinstance(identity, dict) else None):
        return "the G2 lens"
    return "the panel or Sticky"


def _key(*parts) -> str:
    return hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()[:32]


_QUOTES = str.maketrans({"\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"'})
_EDGES = " \t\n.,!?;:'\"\u2026"
#: A twin finished this recently is still "in hand": its job was just dispatched and is running at
#: home, so the same words typed now would start it twice (Codex recheck, 11/09).
RECENT_TWIN = timedelta(minutes=10)


def _norm(value, loose=False):
    """The same words however they're dressed: spacing, curly quotes and the punctuation at either end
    don't make a new request (Codex, 11/09: a trailing '.' escaped his no). Case counts (two paths
    that differ only in case are different work) unless `loose`, which his-no uses."""
    if isinstance(value, str):
        text = " ".join(value.translate(_QUOTES).split()).strip(_EDGES)
        return text.casefold() if loose else text
    if isinstance(value, dict):
        return {k: _norm(v, loose) for k, v in value.items()}
    if isinstance(value, list):
        return [_norm(v, loose) for v in value]
    return value


def _request(args, loose=False) -> dict:
    """What was asked, without the gate's own wording ('what'), normalized: two asks for the same
    thing match, from any thread, however they're punctuated (and, `loose`, however they're cased)."""
    return {k: _norm(v, loose) for k, v in dict(args or {}).items() if k not in ("what", "_start_candidates")}


def _tasks(state) -> dict:
    tasks = state.get("async_tasks") if isinstance(state, dict) else getattr(state, "async_tasks", None)
    return tasks if isinstance(tasks, dict) else {}


# --- what waits (v2_gated_actions) --------------------------------------------------------


class MemoryGatedActionStore:
    """In-process store (tests, dev boxes). Same semantics as the Postgres one."""

    def __init__(self):
        self.rows: dict[int, GatedAction] = {}
        self._next = 1

    def recent_messages(self, limit):
        return sorted((r for r in self.rows.values() if r.action == MESSAGE),
                      key=lambda r: (r.created_at, r.id), reverse=True)[:limit]

    def attempts(self, base_key):
        return sorted((r for r in self.rows.values() if r.base_key == base_key), key=lambda r: r.attempt)

    def same_request(self, action, args, loose=False):
        """Every row asking for this (the same action and args, from any thread), oldest first."""
        want = _request(args, loose)
        return sorted((r for r in self.rows.values() if r.action == action and _request(r.args, loose) == want),
                      key=lambda r: r.id)

    def add(self, approval_id, action, base_key, attempt, thread_id, args, now):
        for r in self.rows.values():
            if (r.base_key, r.attempt) == (base_key, attempt) or r.approval_id == approval_id:
                return r
        r = GatedAction(self._next, approval_id, action, base_key, attempt, thread_id, dict(args), created_at=now)
        self.rows[r.id] = r
        self._next += 1
        return r

    def waiting(self):
        return sorted((r for r in self.rows.values() if r.status == "waiting"), key=lambda r: r.id)

    def claim(self, action_id):
        r = self.rows.get(action_id)
        if r is None or r.status != "waiting":
            return False
        r.status = "running"
        return True

    def finish(self, action_id, status, outcome, now):
        r = self.rows.get(action_id)
        if r is not None and r.status == "running":
            r.status, r.outcome, r.finished_at = status, outcome, now

    def drop(self, action_id, outcome, now):
        r = self.rows.get(action_id)
        if r is None or r.status != "waiting":
            return False
        r.status, r.outcome, r.finished_at = "dropped", outcome, now
        return True


class PostgresGatedActionStore:
    """v2_gated_actions (db/migrations/023). Fresh short connections per call."""

    COLUMNS = "id, approval_id, action, base_key, attempt, thread_id, args, status, created_at, finished_at, outcome"

    def __init__(self, database_url: str):
        self.database_url = database_url

    def _connect(self):
        import psycopg

        return psycopg.connect(self.database_url, connect_timeout=5, options="-c statement_timeout=5000")

    @staticmethod
    def _row(r) -> GatedAction:
        return GatedAction(*r)

    def _one(self, sql, args):
        with self._connect() as conn:
            return conn.execute(sql, args).fetchone()

    def _all(self, sql, args):
        with self._connect() as conn:
            return [self._row(r) for r in conn.execute(sql, args).fetchall()]

    def recent_messages(self, limit):
        return self._all(f"SELECT {self.COLUMNS} FROM v2_gated_actions WHERE action = %s "
                         "ORDER BY created_at DESC, id DESC LIMIT %s", (MESSAGE, limit))

    def attempts(self, base_key):
        return self._all(f"SELECT {self.COLUMNS} FROM v2_gated_actions WHERE base_key = %s ORDER BY attempt",
                         (base_key,))

    def same_request(self, action, args, loose=False):
        """Candidates by action and tool, matched in Python: the match is normalized (_request), which
        a JSONB equality can't do."""
        want = _request(args, loose)
        rows = self._all(f"SELECT {self.COLUMNS} FROM v2_gated_actions WHERE action = %s "
                         "AND (args ->> 'tool') IS NOT DISTINCT FROM %s ORDER BY id",
                         (action, (args or {}).get("tool")))
        return [r for r in rows if _request(r.args, loose) == want]

    def add(self, approval_id, action, base_key, attempt, thread_id, args, now):
        from psycopg.types.json import Jsonb

        with self._connect() as conn:
            row = conn.execute(
                "INSERT INTO v2_gated_actions (approval_id, action, base_key, attempt, thread_id, args, created_at) "
                f"VALUES (%s, %s, %s, %s, %s, %s, %s) ON CONFLICT DO NOTHING RETURNING {self.COLUMNS}",
                (approval_id, action, base_key, attempt, thread_id, Jsonb(dict(args)), now)).fetchone()
            if row is None:
                row = conn.execute(f"SELECT {self.COLUMNS} FROM v2_gated_actions WHERE base_key = %s AND attempt = %s",
                                   (base_key, attempt)).fetchone()
        if row is None:
            raise RuntimeError("the gated action was neither written nor found")
        return self._row(row)

    def waiting(self):
        return self._all(f"SELECT {self.COLUMNS} FROM v2_gated_actions WHERE status = 'waiting' ORDER BY id", ())

    def claim(self, action_id):
        """Before it's done: one watcher does it, once (a claim left by a crash is never redone)."""
        return self._one("UPDATE v2_gated_actions SET status = 'running' WHERE id = %s AND status = 'waiting' "
                         "RETURNING id", (action_id,)) is not None

    def finish(self, action_id, status, outcome, now):
        self._one("UPDATE v2_gated_actions SET status = %s, outcome = %s, finished_at = %s "
                  "WHERE id = %s AND status = 'running' RETURNING id", (status, outcome, now, action_id))

    def drop(self, action_id, outcome, now):
        return self._one("UPDATE v2_gated_actions SET status = 'dropped', outcome = %s, finished_at = %s "
                         "WHERE id = %s AND status = 'waiting' RETURNING id", (outcome, now, action_id)) is not None


# --- the gate ------------------------------------------------------------------------------


class HandsGate:
    """home_job() for D10, message() for D12: GO, or the words she gets instead (and, when it's
    his to approve, the button already asked)."""

    def __init__(self, approvals, store, *, owner_person_id, people=None,
                 executors: dict[str, Callable[[GatedAction], tuple[bool, str]]] | None = None,
                 read_only_agents=(), clock: Callable[[], datetime] | None = None,
                 ask_for: timedelta = DEFAULT_FOR, members=None, discord_id=None):
        self.approvals, self.store, self.owner = approvals, store, owner_person_id
        self.members = dict(members or {})
        self.discord_id = discord_id or (lambda person: None)
        self.people = people
        self.executors = dict(executors or {})
        self.read_only_agents = frozenset(a.strip() for a in read_only_agents if str(a).strip())
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        self.ask_for = ask_for

    # D10 ---------------------------------------------------------------------------------

    def home_job(self, identity, thread_id, name: str, args: dict, state=None, *, tainted=()) -> Verdict:
        who = asker(identity, thread_id, self.owner, self.members)
        if who == OTHER or name not in HOME_TOOLS:
            return Verdict(False, REFUSED_HOME)
        person = str((identity or {}).get("user_id") or "")
        member = person in self.members
        args = {k: v for k, v in args.items() if not k.startswith("_")}
        approvers = None
        if member:
            if name == START:
                if args.get("subagent_type") != self.members[person]:
                    return Verdict(False, "Not started: use your own home project.")
            elif name == "list_async_tasks":
                # Her own list (slice 3): the ledger hands her turns her jobs only; anything else in
                # sight means it didn't, so nothing is listed.
                if any(not isinstance(t, dict) or t.get("owner") != person for t in _tasks(state).values()):
                    return Verdict(False, "Not done: only your own identified home jobs are accessible.")
            else:
                task = _tasks(state).get(args.get("task_id"))
                if not isinstance(task, dict) or task.get("owner") != person:
                    return Verdict(False, "Not done: only your own identified home jobs are accessible.")
            if name in READS:
                return GO
            args["_job_owner"] = person
            if (who == MEMBER and (not tainted or name == CANCEL)) or (who == MEMBER_VOICE and name == CANCEL):
                return self._direct(name, args)
            recipients = [person]
        else:
            recipients = []
            if name == START and who != VOICE_UNSURE and args.get("subagent_type") in set(self.members.values()):
                # A household member's project is hers (D19 slice 4); an unsure voice's project is
                # whoever taps first, set by the watcher.
                return Verdict(False, "Not started: that's a household member's own project. "
                                      "Chris's jobs go to home.")
        if who == VOICE_UNSURE and name == START and self.members:
            recipients = [self.owner, *self.members]
        if tainted:
            recipients = []  # outside text always needs the owner, never a member tap
        if recipients:
            candidates = {}
            for recipient in recipients:
                address = self.discord_id(recipient)
                if address is None or str(address) in candidates:
                    return Verdict(False, "Not done: approval recipients are unavailable or ambiguous.")
                candidates[str(address)] = {"person": recipient,
                    "project": self.members.get(recipient, "home")}
            approvers = list(candidates)
            if name == START:
                args["_start_candidates"] = candidates
        if name in READS:
            return GO
        if who == OWNER_TEXT and (not tainted or name == CANCEL):
            return self._direct(name, args)
        agent = self._agent(name, args, state)
        if who == OWNER_VOICE and (name == CANCEL or (not tainted and agent in self.read_only_agents)):
            return self._direct(name, args)
        if tainted:
            how, heard = "asked after reading outside text", ""
        elif who == OWNER_ASK:
            line = _line(identity, thread_id)
            how, heard = f"asked on {line}", f"Asked on {line}, which can't show it was him typing"
        else:
            how = "asked by voice"
            heard = ("Asked by a strong match on his voice" if who == OWNER_VOICE
                     else "Asked by a voice I wasn't sure was his")
        if name == START:
            words = str(args.get("description") or "").strip()
            summary = f"Start a home job, {how}: {words}"
            what = f"the home job you {how.replace('asked', 'asked for')}"
        elif name == UPDATE:
            words = str(args.get("message") or "").strip()
            summary = f"Change a running home job ({agent or 'unknown job'}), {how}: {words}"
            what = f"the change to the home job you {how.replace('asked', 'asked for')}"
        else:
            task = str(args.get("task_id") or "").strip()
            words = f"Stop task {task or 'unknown'}."
            summary = f"Stop a running home job ({agent or 'unknown job'}), {how}"
            what = f"stopping the home job, {how}"
        if member:
            heard = "Asked by a member voice; approval is required"
        if tainted:
            from aerys_v2.outside_guard import notice
            heard = notice(tainted)
        detail = f"{heard}. Home agent: {agent or 'unknown'}.\n\n{words}"
        if approvers is not None and name == START:
            detail += "\nApproving starts this job in your own home project."
        base_key = f"home:{_key(str(thread_id or ''), name, args)}"
        verdict = self._ask(HOME_JOB, base_key, str(thread_id or ""), {"tool": name, **args, "what": what},
                         kind="home job", summary=summary, detail=detail, what=what,
                         himself=who in HIMSELF and not tainted, approvers=approvers)
        if approvers is not None and verdict.text.startswith(("Waiting for", "Still waiting")):
            return Verdict(False, "Waiting for approval on Discord. Nothing has been done; "
                           "the first approval decides whose home project runs the job.")
        return verdict

    def _direct(self, name: str, args: dict) -> Verdict:
        """He asked directly for something that may already be queued for his tap (Codex, 11/09: an
        approved voice start and the same job typed before the watcher ticked both ran). One of
        them runs, never both: a queued twin still waiting is retired atomically (its button
        withdrawn) and this request is the one execution; a twin the watcher already took is in
        hand, and this one is refused. Unreadable: refused, never run blind."""
        try:
            twins = self.store.same_request(HOME_JOB, {"tool": name, **dict(args or {})})
        except Exception:
            log.warning("hands gate: couldn't check what's queued for %s; refused", name, exc_info=True)
            return Verdict(False, GATE_DOWN)
        now = self.clock()
        for row in twins:
            if row.status == "running" or (row.status == "done" and row.finished_at is not None
                                           and now - row.finished_at < RECENT_TWIN):
                return Verdict(False, f"That's already being done ({row.args.get('what') or 'the job he approved'}); "
                                      "it isn't done twice. Check on it instead.")
            if row.status != "waiting":
                continue
            try:
                retired = self.store.drop(row.id, "asked for directly; done that way", now)
            except Exception:
                log.warning("hands gate: couldn't retire queued action %s; refused", row.id, exc_info=True)
                return Verdict(False, GATE_DOWN)
            if not retired:                                # the watcher took it between the read and now
                return Verdict(False, f"That's already being done ({row.args.get('what') or 'the job he approved'}); "
                                      "it isn't done twice.")
            self.approvals.withdraw(row.approval_id)        # pending: a later tap does nothing
            log.info("hands gate: %s asked for directly; queued action %s retired", name, row.id)
        return GO

    @staticmethod
    def _agent(name, args, state) -> str:
        if name == START:
            return str(args.get("subagent_type") or "").strip()
        task = _tasks(state).get(str(args.get("task_id") or "").strip()) or {}
        return str(task.get("agent_name") or "").strip() if isinstance(task, dict) else ""

    # D12 ---------------------------------------------------------------------------------

    def message(self, thread_id, platform: str, to: str, text: str, identity=None, *, tainted=(), resolved=None) -> Verdict:
        """Before she messages anyone: GO for him and for anyone he's already confirmed at this
        exact address; otherwise his button (one match) or a question for him (none, or several)."""
        platform = str(platform or "").strip().lower()
        to = str(to or "").strip().lstrip("@")
        if platform not in PLATFORMS or not to:
            return Verdict(False, "Not sent: I need a platform (Discord, Telegram or email) and who it's for. Ask Chris.")
        if self.people is None:
            return Verdict(False, "Not sent: I can't check who that is right now, so nothing went. Ask Chris.")
        try:
            found = self.people(platform, to) if resolved is None else resolved
        except Exception:
            log.warning("hands gate: the people lookup failed; nothing sent", exc_info=True)
            return Verdict(False, "Not sent: I can't check who that is right now, so nothing went. Ask Chris.")
        # One destination is one person at one address; anything more is his to pick.
        found = [p for p in found or [] if str(p.get("person_id") or "")]
        destinations: dict[tuple[str, str], dict] = {}
        for p in found:
            destinations.setdefault((str(p["person_id"]), str(p.get("address") or "").strip().lower()), p)
        label = PLATFORMS[platform]
        if not destinations:
            return Verdict(False, f"Not sent: I don't know who {label} {to} is. Ask Chris who it is before "
                                  "sending anything.")
        if len(destinations) > 1:
            names = ", ".join(f"{p.get('name') or 'someone'} ({label}: {p.get('handle') or p.get('address')})"
                              for p in destinations.values())
            return Verdict(False, f"Not sent: {to} could be more than one person or address ({names}). Ask "
                                  "Chris which one he means.")
        (p,) = destinations.values()
        pid = str(p["person_id"])
        if pid == str(self.owner) and not tainted:
            return GO
        address = str(p.get("address") or "").strip()
        if not address:
            return Verdict(False, f"Not sent: I don't have a {label} address for {p.get('name') or to}. Ask Chris.")
        name, handle = str(p.get("name") or "someone"), str(p.get("handle") or address)
        base_key = f"contact:{pid}:{platform}:{address.lower()}"
        # Who and where ride in the detail, which is shown whole or refused (the summary may be cut).
        to_line = f"To: {name} · {label}: {handle}" + (f" ({address})" if address != handle else "")
        try:
            rows = self.store.attempts(base_key)
            approved = [r for r in rows if self.approvals.status(r.approval_id) == "approved"]
        except Exception:
            log.warning("hands gate: couldn't read the contact's confirmations; nothing sent", exc_info=True)
            return Verdict(False, GATE_DOWN)
        if tainted:
            from aerys_v2.outside_guard import notice
            what = f"your message to {name} after reading outside text"
            return self._ask(MESSAGE, f"outside:{base_key}:{_key(text)}", str(thread_id or ""),
                             {"platform": platform, "address": address, "person_id": pid,
                              "name": name, "handle": handle, "text": str(text or ""), "what": what},
                             kind="message", summary=f"Message to {name} · {label}: {handle}",
                             detail=f"{notice(tainted)}\n{to_line}\n\n{text}", what=what)
        if approved:
            # Confirmed. The approved first message itself is the watcher's to send, before and
            # after it goes: her asking again never sends it a second time.
            if any(r.status in ("waiting", "running") for r in approved):
                return Verdict(False, f"Chris approved {approved[-1].args.get('what')}; it's being done now. "
                                      "Don't send it again.")
            if any(r.status == "done" and r.args.get("text") == str(text or "") for r in approved):
                return Verdict(False, "That exact message already went when Chris approved it; it isn't sent twice.")
            if any(r.status == "failed" and r.args.get("text") == str(text or "") for r in approved):
                # It may have gone before it failed: sending the same words again is his call.
                what = f"the resend of your message to {name} ({label}: {handle})"
                return self._ask(MESSAGE, base_key, str(thread_id or ""),
                                 {"platform": platform, "address": address, "person_id": pid, "name": name,
                                  "handle": handle, "text": str(text or ""), "what": what},
                                 himself=asker(identity, thread_id, self.owner) in HIMSELF,
                                 kind="message again",
                                 summary=f"Send again to {name} · {label}: {handle} (the last try may have gone through)",
                                 detail=f"{to_line}\nThe last try may have gone through; this sends the same "
                                        f"words again.\n\nWhat she'd send again:\n{text}", what=what)
            return GO
        what = f"your first message to {name} ({label}: {handle})"
        return self._ask(MESSAGE, base_key, str(thread_id or ""),
                         {"platform": platform, "address": address, "person_id": str(pid), "name": name,
                          "handle": handle, "text": str(text or ""), "what": what},
                         himself=asker(identity, thread_id, self.owner) in HIMSELF,
                         kind="first message", summary=f"First message to {name} · {label}: {handle}",
                         detail=f"{to_line}\n\nWhat she'd send:\n{text}", what=what)

    # the button --------------------------------------------------------------------------

    def _no_today(self, rows, args) -> str | None:
        """When his latest answer to this same request (the same words, from any thread) was a no
        today (Eastern):
        the time he said it, e.g. "10:42 AM"; else None. Ran out or withdrawn isn't a no. Raises
        when an answer can't be read, so the caller fails closed."""
        same = [r for r in rows if _request(r.args, loose=True) == _request(args, loose=True)]
        for r in reversed(same):
            state = self.approvals.status(r.approval_id)
            if state == "unknown":
                raise RuntimeError("an earlier answer can't be read")
            if state not in ("approved", "declined"):
                continue                                   # unanswered, ran out, or withdrawn: not his word
            if state == "approved":
                return None
            decided = self.approvals.store.get(r.approval_id).decided_at
            today = self.clock().astimezone(EASTERN).date()
            if decided is None or decided.astimezone(EASTERN).date() != today:
                return None
            return decided.astimezone(EASTERN).strftime("%-I:%M %p")
        return None

    def _ask(self, action, base_key, thread_id, args, *, kind, summary, detail, what, himself=False, approvers=None) -> Verdict:
        if action not in self.executors:
            return Verdict(False, CANT_ASK_HOME if action == HOME_JOB else CANT_ASK_MESSAGE)
        if len(detail) > DETAIL_CHARS:
            # His button shows the words whole or not at all: what he approves is what goes.
            return Verdict(False, f"Not done: that's too long to show Chris whole on his button (about "
                                  f"{DETAIL_CHARS - 200} characters fit). Shorten it, or ask him in words.")
        try:
            rows = self.store.attempts(base_key)
            last = rows[-1] if rows else None
            if last is not None:
                state = self.approvals.status(last.approval_id)
                if state == "unknown":
                    return Verdict(False, GATE_DOWN)
                if last.status == "waiting" and state == "pending":
                    return Verdict(False, f"Still waiting: Chris hasn't answered the button for {last.args.get('what') or what} "
                                          "yet. Nothing has been done; it goes the moment he taps Approve.")
                if last.status in ("waiting", "running") and state == "approved":
                    return Verdict(False, f"Chris approved {last.args.get('what') or what}; it's being done now. "
                                          "Don't ask again.")
                if last.status == "done" and action in (HOME_JOB, "outside_action"):
                    return Verdict(False, f"Chris approved that already and it was done ({last.outcome or 'started'}). "
                                          "It isn't done twice.")
            # The same request from another thread (D17 Q5's "same request"): one live button for it,
            # wherever it was asked, so two approvals can't do it twice (Gemini recheck).
            same = self.store.same_request(action, args)
            for r in same:
                if r.base_key == base_key or r.status not in ("waiting", "running"):
                    continue
                state = self.approvals.status(r.approval_id)
                if state == "unknown":
                    return Verdict(False, GATE_DOWN)
                if r.status == "waiting" and state == "pending":
                    return Verdict(False, f"Still waiting: Chris hasn't answered the button for {r.args.get('what') or what} "
                                          "yet (asked from another conversation). Nothing has been done; it goes the "
                                          "moment he taps Approve.")
                if state == "approved":
                    return Verdict(False, f"Chris approved {r.args.get('what') or what}; it's being done now. "
                                          "Don't ask again.")
            if not himself:
                said_no = self._no_today(self.store.same_request(action, args, loose=True), args)
                if said_no is not None:
                    return Verdict(False, f"Chris said no to {what} at {said_no}; I'll leave it for today unless he "
                                          "brings it up himself.")
            attempt = last.attempt + 1 if last is not None else 1
            # The button is keyed by exactly what it shows: two turns asking different things at
            # once get two buttons, and only the one whose row is recorded stays.
            approval_id = self.approvals.request(kind, summary, detail, expires_in=self.ask_for,
                                                 key=f"{base_key}:{attempt}:{_key(args)}",
                                                 **({"approvers": approvers} if approvers is not None else {}))
            try:
                row = self.store.add(approval_id, action, base_key, attempt, thread_id, args, self.clock())
            except Exception:
                self.approvals.withdraw(approval_id)    # a button with nothing behind it would be a lie
                raise
            if row.approval_id != approval_id:          # another turn's request got there first
                self.approvals.withdraw(approval_id)
                return Verdict(False, f"Still waiting: Chris hasn't answered the button for {row.args.get('what') or what} "
                                      "yet. Nothing has been done; ask again after he answers.")
        except Exception:
            log.warning("hands gate: couldn't put %s to him; nothing done", action, exc_info=True)
            return Verdict(False, "Not done: I couldn't put this to Chris for his OK just now, so nothing happened.")
        log.info("hands gate: %s waits for his button (approval %s)", action, approval_id)
        return Verdict(False, f"Waiting for Chris: I've asked him on Discord (a button, and a notification on his "
                              f"phone) to OK {what}. Nothing has been done; it goes only when he taps Approve. "
                              "Tell him it's waiting on his tap.")


# --- the watcher: does what he approved, once ---------------------------------------------


def _note_for(row: GatedAction, status: str, outcome: str = "") -> str:
    what = row.args.get("what") or ("the home job" if row.action == HOME_JOB else "the message")
    if status == "done":
        if row.action == MESSAGE:
            return (f"Chris tapped Approve, and {what} was sent. Later messages to "
                    f"{row.args.get('name') or 'them'} at that address go without asking unless the turn reads outside text.")
        if row.action == "outside_action":
            return f"The owner tapped Approve for {what}. {outcome}".strip()
        return f"Chris tapped Approve, and {what} has started. {outcome}".strip()
    if status == "failed":
        return (f"Chris approved {what}, but it didn't go through ({outcome}). It may not have happened; "
                "tell him, and don't say it's done.")
    said = {"declined": "Chris declined", "expired": "Chris didn't answer in time about",
            "withdrawn": "Withdrawn:"}.get(status, "Not approved:")
    return f"{said} {what}, so it wasn't done."


class HandsWatcher:
    """tick(): every waiting action whose button was answered is done (approved, once) or dropped
    (declined, expired, withdrawn); she's told either way. Pending or unreadable: it waits.
    An executor returns (True, what happened) when it went and (False, why) when it didn't;
    anything else, a refusal string included, counts as not done."""

    def __init__(self, gate: HandsGate, *, note: Callable[[str, str], None] | None = None, tick_s: float = TICK_S):
        self.gate, self.note, self.tick_s = gate, note, tick_s
        self._stop = threading.Event()

    def _tell(self, row: GatedAction, text: str) -> None:
        if self.note is None or not row.thread_id:
            return
        try:
            self.note(row.thread_id, text)
        except Exception:
            log.warning("hands watcher: couldn't note action %s's outcome in her thread", row.id, exc_info=True)

    def tick(self) -> dict:
        g, out = self.gate, {"done": 0, "failed": 0, "dropped": 0}
        for row in g.store.waiting():
            state = g.approvals.status(row.approval_id)
            now = g.clock()
            if state == "approved":
                run = g.executors.get(row.action)
                if run is None:
                    log.warning("hands watcher: action %s approved but nothing can carry out %s", row.id, row.action)
                    continue
                if not g.store.claim(row.id):
                    continue                                # another watcher has it
                try:
                    candidates = row.args.get("_start_candidates")
                    if row.action == HOME_JOB and row.args.get("tool") == START and candidates:
                        winner = candidates.get(g.approvals.decided_by(row.approval_id))
                        if winner is None:
                            raise ValueError("approval has no authorized winner")
                        row.args = {**row.args, "_job_owner": winner["person"],
                                    "subagent_type": winner["project"]}
                    result = run(row)
                    ok, said = result if isinstance(result, tuple) and len(result) == 2 else (False, "no clear result")
                    outcome = " ".join(str(said or "").split())[:OUTCOME_CHARS]
                    status = "done" if ok is True else "failed"
                except Exception as e:
                    outcome, status = f"{type(e).__name__}", "failed"
                if status == "failed":
                    log.warning("hands watcher: action %s (%s) didn't go through", row.id, row.action)
                try:
                    g.store.finish(row.id, status, outcome, now)
                except Exception:
                    log.warning("hands watcher: action %s %s, but recording it failed (never redone)", row.id, status)
                out[status] += 1
                self._tell(row, _note_for(row, status, outcome))
            elif state in ("declined", "expired", "withdrawn"):
                if g.store.drop(row.id, state, now):
                    out["dropped"] += 1
                    self._tell(row, _note_for(row, state))
        return out

    def run_forever(self) -> None:  # pragma: no cover - thin loop over tick
        log.info("hands watcher armed | every %ss", self.tick_s)
        while not self._stop.is_set():
            try:
                self.tick()
            except Exception:
                log.warning("hands watcher tick failed", exc_info=True)
            self._stop.wait(self.tick_s)

    def start(self) -> threading.Thread:  # pragma: no cover - live only
        t = threading.Thread(target=self.run_forever, name="hands-watcher", daemon=True)
        t.start()
        return t

    def stop(self) -> None:
        self._stop.set()


# --- D10 in her loop: the middleware beside AsyncSubAgentMiddleware -----------------------


def home_job_middleware(gate: HandsGate):
    """An AgentMiddleware that puts every home-job tool call through the gate before it runs.
    Anything the gate can't decide is refused, never run."""
    from langchain.agents.middleware import AgentMiddleware
    from langchain_core.messages import ToolMessage

    def refusal(request):
        call = request.tool_call or {}
        name = call.get("name")
        if name not in HOME_TOOLS:
            return None
        try:
            config = getattr(request.runtime, "config", None) or {}
            thread_id = (config.get("configurable") or {}).get("thread_id")
            verdict = gate.home_job(identity_from_config(config), thread_id, name, dict(call.get("args") or {}),
                                    request.state)
        except Exception:
            log.warning("hands gate: deciding %s failed; refused", name, exc_info=True)
            verdict = Verdict(False, GATE_DOWN)
        if verdict.go:
            return None
        return ToolMessage(content=verdict.text, tool_call_id=call.get("id") or "", name=name)

    class HomeJobGate(AgentMiddleware):
        def wrap_tool_call(self, request, handler):
            refused = refusal(request)
            return handler(request) if refused is None else refused

        async def awrap_tool_call(self, request, handler):
            refused = await asyncio.to_thread(refusal, request)
            return await handler(request) if refused is None else refused

    return HomeJobGate()


# --- arming --------------------------------------------------------------------------------


def people_lookup_for(settings) -> Callable[[str, str], list[dict]] | None:
    """Her people records (prod persons + platform_identities), read-only: who an address is.
    A Discord or Telegram account by its id or username; an email by persons.email; else, by the
    name he gave them, anyone with that name who has an address on that platform."""
    db = getattr(settings, "memories_database_url", None)
    if not db:
        return None

    def lookup(platform: str, to: str) -> list[dict]:
        import psycopg

        def rows(conn, sql, args):
            return [{"person_id": str(r[0]), "name": r[1] or "", "address": str(r[2] or ""), "handle": str(r[3] or r[2] or "")}
                    for r in conn.execute(sql, args).fetchall()]

        with psycopg.connect(db, connect_timeout=5,
                             options="-c statement_timeout=5000 -c default_transaction_read_only=on") as conn:
            if platform == "email":
                found = rows(conn, "SELECT id, display_name, btrim(email), btrim(email) FROM persons "
                                   "WHERE deleted_at IS NULL AND lower(btrim(email)) = lower(%s)", (to,))
                if not found:
                    found = rows(conn, "SELECT id, display_name, btrim(email), btrim(email) FROM persons "
                                       "WHERE deleted_at IS NULL AND lower(btrim(display_name)) = lower(%s) "
                                       "AND coalesce(btrim(email), '') <> ''", (to,))
                return found
            sql = ("SELECT p.id, p.display_name, pi.platform_user_id, nullif(btrim(pi.username), '') "
                   "FROM platform_identities pi JOIN persons p ON p.id = pi.person_id "
                   "WHERE p.deleted_at IS NULL AND pi.platform = %s AND ")
            found = rows(conn, sql + "(pi.platform_user_id = %s OR lower(btrim(pi.username)) = lower(%s))",
                         (platform, to, to))
            if not found:
                found = rows(conn,
                             "SELECT p.id, p.display_name, pi.platform_user_id, nullif(btrim(pi.username), '') "
                             "FROM persons p LEFT JOIN platform_identities pi "
                             "ON pi.person_id = p.id AND pi.platform = %s "
                             "WHERE p.deleted_at IS NULL AND lower(btrim(p.display_name)) = lower(%s)",
                             (platform, to))
            return found

    return lookup


def hands_gate_for(settings, *, executors: dict[str, Callable[[GatedAction], tuple[bool, str]]] | None = None
                   ) -> HandsGate | None:
    """None unless armed: hands_gates on, the approval button armed, a database and an owner. The
    executors (what carries out an approved home job or message) come from whatever wires the job
    and message paths in; without one, that kind of request is refused rather than asked."""
    if not getattr(settings, "hands_gates", False) or not getattr(settings, "database_url", None) \
            or getattr(settings, "owner_person_id", None) is None:
        return None
    approvals = approvals_for(settings)
    if approvals is None:
        log.warning("hands gates: the approval button isn't armed (APPROVALS + an HA notify service); not armed")
        return None
    from aerys_v2.config import home_job_member_map

    read_only = [a for a in str(getattr(settings, "home_read_only_agents", "") or "").split(",") if a.strip()]
    gate = HandsGate(approvals, PostgresGatedActionStore(settings.database_url),
                     owner_person_id=str(settings.owner_person_id), people=people_lookup_for(settings),
                     executors=executors, read_only_agents=read_only,
                     members=home_job_member_map(settings),
                     discord_id=lambda person: person_id_for(settings, person))

    if getattr(settings, "messages_as_herself", False):
        from aerys_v2.tools.send_message import message_sender_for
        sender = message_sender_for(settings, gate)
        if sender is not None:
            gate.message_sender = sender
            gate.executors[MESSAGE] = sender.execute
    return gate
