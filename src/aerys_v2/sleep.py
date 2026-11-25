"""Her sleep (D100, Chris 11/25 12:48: "yes to both Tidy and Reflect"; design .planning/design/her-sleep.md).

Nightly, after the 03:00 backup, two halves, both for the owner in v1:

TIDY proposes, never changes. It reads his live memories (prod `aerys.memories`) and the old claims table
(`core_claim`), asks Jev one plain question per candidate, and writes PROPOSALS to v2_memory_proposals
(aerys_v2). Judgements from Jev, decisions in code (the TypeSafe rule):
  - duplicate: two of his facts close in meaning that Jev says state the same thing (>= SAME_AT): keep the
    more specific (the longer), retire the other;
  - contradiction: two facts under the same kind of key that Jev says can't both be true today
    (< BOTH_TRUE_BELOW): keep the newer, retire the older;
  - stale_plan: a plan-shaped fact (code's word check) that Jev says is about a time now passed
    (>= PLAN_PASSED_AT): retire it;
  - old_claim: an active core_claim that Jev says describes a passing moment (>= MOMENTARY_AT): retire it
    (status 'retired', the 9/21 mechanism; core_claim has had no writer since 2026-07-03).
Never: her own words (aerys.*, so never a promise: D101 owns those), events against facts, events at all
for duplicates (the same event on another date is a second event, D97), Kael-line logs, gap and skill rows.
Every candidate judged fine is recorded 'clear' under the same stable key, so it is never judged again
until its text changes; a proposal he rejected is never proposed again. A nightly cap on Jev questions
(her_sleep_judgements) is logged when it bites, never silent. Chris approves on the dashboard's Memory tab;
apply_approved re-checks every row and retires it through the soft-delete paths; D103 (Chris 11/25 14:10, "A
works") runs it on its own every APPLY_EVERY_S, so his Approve takes effect within minutes (sleep-apply stays for
Kael by hand). Nothing is hard-deleted.

REFLECT is her own words about her day. She asked for its shape herself (11/25 13:13, on Kael's line):
Sira's questions plus "open loops" and "what went right", one concrete change at most, her last notes
beside her so she can see a repeat, the receipts next to a flagged reply, 150 to 250 words, and a
"Pattern:" line that goes to Kael as a gap report instead of the note changing anything. One tool-free call
with her soul (no ask(), so no v2_turns row for the extractor to mine). The note goes to v2_reflections
only: it is not a memory, so it feeds nothing until Chris has read the first week (D100). It never edits
soul.md, never changes a fact about him, never reaches out.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Any, Callable

from aerys_v2.reach import EASTERN

log = logging.getLogger(__name__)

# ---- when ---------------------------------------------------------------------------------------
SLEEP_HOUR, SLEEP_MINUTE = 3, 30          # Eastern: after the 03:00 backup
SLEEP_UNTIL_HOUR = 12                     # a brain that was down at 03:30 still sleeps that morning
TICK_S = 60
#: D103 (Chris 11/25 14:10, "A works"): what he approved on the dashboard is applied on its own, this often,
#: through apply_approved's re-check (a changed line goes stale; nothing is hard-deleted).
APPLY_EVERY_S = 600

# ---- Tidy: what it looks at ---------------------------------------------------------------------
#: His facts: the key kinds Tidy may touch. Everything else (her own words aerys.*, event.*, Kael-line
#: logs a2a_*, gap.*, exec.*, skill*, session.*, portable.*) is never proposed.
FACT_KINDS = ("basic", "user", "work", "relationship", "pet", "vehicle", "preference", "interest", "health",
              "emotional", "technical", "identity", "remember", "hardware", "project", "design")
#: Stale plans may also be decisions and events ("plans to go to the farmers market Oct 29").
PLAN_KINDS = FACT_KINDS + ("decision", "event")
DUP_DISTANCE = 0.25        # cosine distance under which two facts are asked "the same?"
CONTRA_DISTANCE = 0.35     # under which two same-kind facts are asked "both true today?"
NEIGHBOURS = 3
MODEL = "jev-1.14.0"
TIMEOUT_S = 8.0
SAME_AT = 0.9
BOTH_TRUE_BELOW = 0.2
PLAN_PASSED_AT = 0.8
MOMENTARY_AT = 0.8
VALUE_MAX = 500
#: A plan-shaped line: code's cheap filter before Jev is asked anything.
PLAN_WORDS = re.compile(
    r"\b(?:plan(?:s|ning|ned)?\s+to|will|going\s+to|gonna|upcoming|tomorrow|tonight|next\s+(?:week|month|"
    r"weekend|year)|this\s+(?:week|weekend|month)|scheduled|intends?\s+to|about\s+to|soon|later\s+today)\b",
    re.I)

QUESTIONS = {
    "same": {
        "type": "noul",
        "instructions": "Do `a` and `b` say the same thing about him, so that keeping only one of them loses "
                        "nothing?",
        "criteria": {
            "true": "Both state the same fact, perhaps worded differently or one with more detail (\"Has a cat "
                    "named Pip\" and \"Has a 13-year-old cat named Pip\"; \"Nicknamed his car Comet\" and "
                    "\"Calls his car Comet\").",
            "false": "They state different facts, even about the same topic (\"Likes hiking\" and \"Went "
                     "hiking on Saturday\"; \"Has a cat named Pip\" and \"Has a cat named Moss\").",
        },
    },
    "both_true": {
        "type": "noul",
        "instructions": "Can `older.text` and `newer.text` both be true about him today?",
        "criteria": {
            "true": "They fit together, or they are about different things (\"Works at a bank\" and \"Started "
                    "a new role at the bank in July\"; \"Has a cat named Pip\" and \"Has a cat named Moss\").",
            "false": "The newer one replaced the older: both can't hold now (\"Lives in Ohio\" and \"Moved to "
                     "Florida\"; \"Drives a sedan\" and \"Sold the sedan, now drives a truck\").",
        },
    },
    "plan_passed": {
        "type": "noul",
        "instructions": "Is `line` a plan, an expectation or an upcoming event whose time has clearly passed "
                        "by `today`?",
        "criteria": {
            "true": "It was about a time that is now over (\"plans to go to the market on Oct 29\" when today "
                    "is in November; \"has a dentist appointment tomorrow\" written weeks ago).",
            "false": "It is a lasting fact, a habit, or a plan still ahead or without a date (\"likes "
                     "farmers markets\", \"goes to the gym on Mondays\", \"wants to visit Japan someday\").",
        },
    },
    "momentary": {
        "type": "noul",
        "instructions": "Does `line` describe a passing moment or a temporary situation, rather than "
                        "something lasting about him?",
        "criteria": {
            "true": "A moment or a short-lived state (\"restarted the system and is doing final checks\", "
                    "\"his sister is staying over this week\", \"will finish the fixes tomorrow\").",
            "false": "Something that stays true over time (\"has two kids\", \"is allergic to peanuts\", "
                     "\"likes sci-fi movies\", \"got married in April\").",
        },
    },
}

#: Asked in the same request as plan_passed / momentary (speculatively; code reads it only when the line would be
#: proposed). A line that also carries something lasting is still proposed, but marked careful: read it first,
#: and "Approve all" on the dashboard skips it (a side agent's catch, 11/25: "recently married or about to be",
#: a name tied to an old trip plan, a launch he watched).
QUESTIONS["lasting"] = {
    "type": "noul",
    "instructions": "Besides any plan or passing moment, does `line` also say something lasting about him that "
                    "would be lost if the line were removed: who he is, who he knows, what he likes, or something "
                    "he did?",
    "criteria": {
        "true": "Removing it would lose a lasting fact (\"recently married or about to be\" says he is married; "
                "\"knows someone named Sam, planned to leave at 9 with Sam\" says he knows Sam; \"watched the "
                "launch from home\" is something he did).",
        "false": "It is only the plan or the moment (\"going to bed\", \"plans to visit the hardware store "
                 "after work\", \"tired today\").",
    },
}
LASTING_AT = 0.5           # erring toward careful: a coin flip is read by him, never approved in bulk

LIVE_FACTS_SQL = """\
SELECT id::text, key_label, content, created_at, embedding IS NOT NULL
FROM memories
WHERE person_id = %(person_id)s::uuid AND deleted_at IS NULL
  AND split_part(COALESCE(key_label, ''), '.', 1) = ANY(%(kinds)s::text[])
ORDER BY created_at
"""

PAIRS_SQL = """\
SELECT a.id::text, b.id::text, (a.embedding <=> b.embedding)::float8 AS d
FROM memories a
JOIN LATERAL (
  SELECT b.id, b.embedding FROM memories b
  WHERE b.person_id = a.person_id AND b.deleted_at IS NULL AND b.id <> a.id AND b.embedding IS NOT NULL
    AND split_part(COALESCE(b.key_label, ''), '.', 1) = ANY(%(kinds)s::text[])
  ORDER BY a.embedding <=> b.embedding
  LIMIT %(neighbours)s
) b ON true
WHERE a.person_id = %(person_id)s::uuid AND a.deleted_at IS NULL AND a.embedding IS NOT NULL
  AND split_part(COALESCE(a.key_label, ''), '.', 1) = ANY(%(kinds)s::text[])
  AND (a.embedding <=> b.embedding) < %(max_d)s
ORDER BY d
"""

CLAIMS_SQL = """\
SELECT core_id::text, key_label, claim_text, created_at
FROM core_claim
WHERE speaker_id = %(person_id)s::uuid AND status IN ('approved', 'provisional')
  AND (ttl_ts IS NULL OR ttl_ts > now())
ORDER BY created_at
"""

KNOWN_KEYS_SQL = "SELECT proposal_key, status FROM v2_memory_proposals WHERE person_id = %(person_id)s::uuid"
#: Rows already in a proposal that is waiting on him or approved: never proposed twice at once.
BUSY_IDS_SQL = """\
SELECT unnest(retire_ids || keep_ids) FROM v2_memory_proposals
WHERE person_id = %(person_id)s::uuid AND status IN ('proposed', 'approved')
"""
SAVE_SQL = """\
INSERT INTO v2_memory_proposals
  (proposal_key, person_id, kind, retire_table, retire_ids, keep_ids, snapshot, reason, score, status, careful)
VALUES (%(proposal_key)s, %(person_id)s::uuid, %(kind)s, %(retire_table)s, %(retire_ids)s::text[],
        %(keep_ids)s::text[], %(snapshot)s::jsonb, %(reason)s, %(score)s, %(status)s, %(careful)s)
ON CONFLICT (proposal_key) DO NOTHING
"""


_CAREFUL = " It may also hold something lasting about him: read it before approving."


def _value(key: str | None, content: str | None) -> str:
    text = " ".join(str(content or "").split())
    if key and text.lower().startswith(str(key).lower() + ":"):
        text = text[len(key) + 1:].strip()
    return text[:VALUE_MAX]


def _kind(key: str | None) -> str:
    return str(key or "").split(".", 1)[0]


def proposal_key(kind: str, retire: list[str], keep: list[str], texts: list[str]) -> str:
    """Stable: the kind, the rows (sorted) and their text. Same rows with the same text = the same key, so a
    rejected proposal or a 'clear' judgement is never repeated (or re-asked); changed text is a new key. A pair
    is keyed as the pair (retire=both, keep=none), whichever row the proposal would retire."""
    digest = hashlib.sha1("\x1f".join(texts).encode("utf-8")).hexdigest()[:12]
    return f"{kind}:{','.join(sorted(retire))}|{','.join(sorted(keep))}|{digest}"


@dataclass
class Proposal:
    kind: str
    retire_table: str
    retire_ids: list
    keep_ids: list
    snapshot: dict
    reason: str
    score: float | None
    status: str = "proposed"          # or 'clear': judged, nothing to propose
    key: str = ""
    careful: bool = False             # holds something lasting too: read it, never approve it in bulk

    def row(self, person_id: str) -> dict:
        return {"proposal_key": self.key, "person_id": person_id, "kind": self.kind,
                "retire_table": self.retire_table, "retire_ids": list(self.retire_ids),
                "keep_ids": list(self.keep_ids), "snapshot": json.dumps(self.snapshot), "reason": self.reason,
                "score": self.score, "status": self.status, "careful": bool(self.careful)}


class Judge:
    """One Jev question per call, bounded by a nightly budget. ask() -> the probability, or None (over budget,
    timeout, error, a bad score). None never becomes a proposal and is never recorded 'clear'."""

    def __init__(self, client: Any, *, budget: int, model: str = MODEL):
        self.client, self.budget, self.model = client, int(budget), model
        self.asked = 0
        self.over = 0

    def ask(self, qid: str, state: dict) -> float | None:
        return self.ask_many((qid,), state).get(qid)

    def ask_many(self, qids: tuple, state: dict) -> dict:
        """Several questions over one state in ONE request (one unit of the budget). Missing or bad answers are
        left out of the result."""
        if self.asked >= self.budget:
            self.over += 1
            return {}
        self.asked += 1
        try:
            response = self.client.system_one(state=state, questions={q: QUESTIONS[q] for q in qids},
                                              model=self.model)
            answers = {q: response.answers[q].noul for q in qids if q in response.answers}
        except Exception:  # noqa: BLE001 -- a failed look proposes nothing
            log.warning("sleep: Jev %s failed", "+".join(qids), exc_info=True)
            return {}
        return {q: float(v) for q, v in answers.items()
                if not isinstance(v, bool) and isinstance(v, (int, float)) and 0.0 <= float(v) <= 1.0}


def judge_for(settings) -> Judge | None:
    key = getattr(settings, "typesafe_api_key", None)
    if not key:
        return None
    from typesafe_sdk import RetryPolicy, TypeSafeClient

    client = TypeSafeClient(api_key=key.get_secret_value(), model=MODEL, timeout=TIMEOUT_S,
                            retry=RetryPolicy(max_retries=1))
    return Judge(client, budget=int(getattr(settings, "her_sleep_judgements", 600) or 600))


def tidy(mem_conn, person_id: str, judge: Judge, *, known: dict[str, str], busy: set[str],
         today: date) -> list[Proposal]:
    """The Tidy pass over one person. Reads only; returns proposals and 'clear' records to save.
    known: proposal_key -> status of everything already recorded (a key is never judged twice)."""
    out: list[Proposal] = []
    facts = {r[0]: {"key": r[1] or "", "value": _value(r[1], r[2]), "at": r[3]}
             for r in mem_conn.execute(LIVE_FACTS_SQL, {"person_id": person_id,
                                                        "kinds": list(PLAN_KINDS)}).fetchall()}
    taken: set[str] = set(busy)

    def add(p: Proposal) -> None:
        if p.key in known:
            return
        known[p.key] = p.status
        out.append(p)
        if p.status == "proposed":
            taken.update(p.retire_ids)
            taken.update(p.keep_ids)

    # Duplicates and contradictions: pairs of his facts close in meaning.
    pairs = mem_conn.execute(PAIRS_SQL, {"person_id": person_id, "kinds": list(FACT_KINDS),
                                         "neighbours": NEIGHBOURS, "max_d": CONTRA_DISTANCE}).fetchall()
    seen: set[tuple] = set()
    for a_id, b_id, d in pairs:
        pair = tuple(sorted((a_id, b_id)))
        if pair in seen or a_id not in facts or b_id not in facts:
            continue
        seen.add(pair)
        if taken & set(pair):
            continue
        a, b = facts[pair[0]], facts[pair[1]]
        older, newer = (a, b) if a["at"] <= b["at"] else (b, a)
        old_id, new_id = (pair[0], pair[1]) if older is a else (pair[1], pair[0])
        snap = {old_id: f"{older['key']}: {older['value']}", new_id: f"{newer['key']}: {newer['value']}"}
        texts = [snap[i] for i in sorted(snap)]
        if d < DUP_DISTANCE:
            dup_key = proposal_key("duplicate", list(pair), [], texts)
            if known.get(dup_key, "clear") != "clear":
                continue                   # a duplicate proposal for this pair exists (waiting, rejected, done)
            if dup_key not in known:
                p = judge.ask("same", {"a": a["value"], "b": b["value"]})
                if p is not None:
                    if p >= SAME_AT:
                        keep, drop = ((new_id, old_id) if len(newer["value"]) >= len(older["value"])
                                      else (old_id, new_id))
                        add(Proposal("duplicate", "memories", [drop], [keep], snap,
                                     "Both say the same thing; keeping the more specific one.", p,
                                     key=dup_key))
                        continue
                    add(Proposal("duplicate", "memories", list(pair), [], snap, "Not the same.", p,
                                 status="clear", key=dup_key))
        if _kind(older["key"]) != _kind(newer["key"]):
            continue
        both_key = proposal_key("contradiction", list(pair), [], texts)
        if both_key in known:
            continue
        p = judge.ask("both_true", {"older": {"text": older["value"], "written_on": older["at"].date().isoformat()},
                                    "newer": {"text": newer["value"], "written_on": newer["at"].date().isoformat()}})
        if p is None:
            continue
        if p < BOTH_TRUE_BELOW:
            add(Proposal("contradiction", "memories", [old_id], [new_id], snap,
                         "Can't both be true today; keeping the newer.", p, key=both_key))
        else:
            add(Proposal("contradiction", "memories", list(pair), [], snap, "Both can be true.", p,
                         status="clear", key=both_key))

    # Stale plans: plan-shaped lines (code's check) whose time Jev says has passed.
    for row_id, f in facts.items():
        if row_id in taken or not PLAN_WORDS.search(f["value"]):
            continue
        if f["at"].date() >= today - timedelta(days=1):
            continue                                     # said yesterday or today: not stale yet
        snap = {row_id: f"{f['key']}: {f['value']}"}
        key = proposal_key("stale_plan", [row_id], [], [snap[row_id]])
        if key in known:
            continue
        got = judge.ask_many(("plan_passed", "lasting"), {"line": f["value"], "written_on": f["at"].date().isoformat(),
                                                          "today": today.isoformat()})
        p = got.get("plan_passed")
        if p is None:
            continue
        careful = p >= PLAN_PASSED_AT and got.get("lasting", 1.0) >= LASTING_AT   # no answer: careful
        add(Proposal("stale_plan", "memories", [row_id], [], snap,
                     ("A plan whose time has passed." + (_CAREFUL if careful else "")) if p >= PLAN_PASSED_AT
                     else "Still current.", p,
                     status="proposed" if p >= PLAN_PASSED_AT else "clear", key=key, careful=careful))

    # Old claims: the frozen core_claim table, still injected into every reply.
    for core_id, key_label, text, at in mem_conn.execute(CLAIMS_SQL, {"person_id": person_id}).fetchall():
        if core_id in taken:
            continue
        line = f"{key_label or ''}: {' '.join(str(text or '').split())[:VALUE_MAX]}"
        snap = {core_id: line}
        key = proposal_key("old_claim", [core_id], [], [line])
        if key in known:
            continue
        got = judge.ask_many(("momentary", "lasting"), {"line": " ".join(str(text or "").split())[:VALUE_MAX],
                                                        "written_on": at.date().isoformat() if at else "unknown"})
        p = got.get("momentary")
        if p is None:
            continue
        careful = p >= MOMENTARY_AT and got.get("lasting", 1.0) >= LASTING_AT
        add(Proposal("old_claim", "core_claim", [core_id], [], snap,
                     ("Reads as a passing moment, not who he is." + (_CAREFUL if careful else ""))
                     if p >= MOMENTARY_AT else "Lasting.", p,
                     status="proposed" if p >= MOMENTARY_AT else "clear", key=key, careful=careful))
    if judge.over:
        log.warning("sleep: tidy hit its nightly cap (%d asked); %d questions wait for tomorrow night",
                    judge.asked, judge.over)
    return out


def save_proposals(v2_conn, person_id: str, proposals: list[Proposal]) -> int:
    saved = 0
    for p in proposals:
        saved += v2_conn.execute(SAVE_SQL, p.row(person_id)).rowcount or 0
    return saved


def run_tidy(v2_db, mem_db, person_id: str, judge: Judge, today: date) -> dict:
    with v2_db() as v2:
        known = {r[0]: r[1] for r in v2.execute(KNOWN_KEYS_SQL, {"person_id": person_id}).fetchall()}
        busy = {str(r[0]) for r in v2.execute(BUSY_IDS_SQL, {"person_id": person_id}).fetchall()}
    with mem_db() as mem:
        mem.read_only = True
        found = tidy(mem, person_id, judge, known=known, busy=busy, today=today)
    with v2_db() as v2:
        saved = save_proposals(v2, person_id, found)
    proposed = sum(1 for p in found if p.status == "proposed")
    log.info("sleep: tidy for %s: %d proposed, %d clear, %d Jev questions (%d waiting for the cap)",
             person_id, proposed, len(found) - proposed, judge.asked, judge.over)
    return {"proposed": proposed, "clear": len(found) - proposed, "saved": saved, "asked": judge.asked,
            "over_cap": judge.over}


# ---- applying what he approved (Kael-run in v1) ---------------------------------------------------
APPROVED_SQL = """\
SELECT id, person_id::text, kind, retire_table, retire_ids, keep_ids, snapshot
FROM v2_memory_proposals WHERE status = 'approved' ORDER BY decided_at NULLS LAST, id
"""
MARK_SQL = "UPDATE v2_memory_proposals SET status = %(status)s, applied_at = now() WHERE id = %(id)s AND status = 'approved'"
MEMORY_ROW_SQL = ("SELECT key_label, content, deleted_at IS NULL, COALESCE(category, ARRAY[]::text[]) "
                  "FROM memories WHERE id = %(id)s::uuid AND person_id = %(person_id)s::uuid")
CLAIM_ROW_SQL = ("SELECT key_label, claim_text, status FROM core_claim "
                 "WHERE core_id = %(id)s::uuid AND speaker_id = %(person_id)s::uuid")
RETIRE_MEMORY_SQL = """\
UPDATE memories SET deleted_at = now(), updated_at = now(),
       category = COALESCE(category, ARRAY[]::text[]) || ARRAY[%(tag)s]::text[]
WHERE id = %(id)s::uuid AND person_id = %(person_id)s::uuid AND deleted_at IS NULL
"""
RETIRE_CLAIM_SQL = """\
UPDATE core_claim SET status = 'retired'
WHERE core_id = %(id)s::uuid AND speaker_id = %(person_id)s::uuid AND status IN ('approved', 'provisional')
"""


def _still_as_judged(mem, table: str, row_id: str, person_id: str, judged: str, *, tag: str) -> str:
    """'live' (unchanged and live), 'done' (already retired by this tidy), or 'changed'."""
    if table == "memories":
        r = mem.execute(MEMORY_ROW_SQL, {"id": row_id, "person_id": person_id}).fetchone()
        if r is None:
            return "changed"
        now_text = f"{r[0] or ''}: {_value(r[0], r[1])}"
        if not r[2]:
            return "done" if tag in (r[3] or []) and now_text == judged else "changed"
        return "live" if now_text == judged else "changed"
    r = mem.execute(CLAIM_ROW_SQL, {"id": row_id, "person_id": person_id}).fetchone()
    if r is None:
        return "changed"
    now_text = f"{r[0] or ''}: {' '.join(str(r[1] or '').split())[:VALUE_MAX]}"
    if r[2] == "retired":
        return "done" if now_text == judged else "changed"
    return "live" if r[2] in ("approved", "provisional") and now_text == judged else "changed"


def apply_approved(v2_db, mem_db, *, dry_run: bool = False) -> list[dict]:
    """Retire what he approved, after re-checking every row: each row to retire and each row kept must still
    read exactly as judged (and the kept one must be live), or the proposal is marked 'stale' and nothing of
    it is touched. Memories: soft delete tagged tidy:<kind>; claims: status 'retired'."""
    results = []
    with v2_db() as v2:
        approved = v2.execute(APPROVED_SQL).fetchall()
    for pid, person, kind, table, retire, keep, snapshot in approved:
        snap = snapshot if isinstance(snapshot, dict) else json.loads(snapshot or "{}")
        tag = f"tidy:{kind}"
        with mem_db() as mem:
            states = {i: _still_as_judged(mem, table, i, person, snap.get(i, ""), tag=tag) for i in retire}
            keeps = {i: _still_as_judged(mem, "memories", i, person, snap.get(i, ""), tag=tag) for i in keep}
            ok = (all(s in ("live", "done") for s in states.values())
                  and all(s == "live" for s in keeps.values()))
            if ok and not dry_run:
                for i, s in states.items():
                    if s == "live":
                        if table == "memories":
                            mem.execute(RETIRE_MEMORY_SQL, {"id": i, "person_id": person, "tag": tag})
                        else:
                            mem.execute(RETIRE_CLAIM_SQL, {"id": i, "person_id": person})
        status = "applied" if ok else "stale"
        if not dry_run:
            with v2_db() as v2:
                v2.execute(MARK_SQL, {"status": status, "id": pid})
        results.append({"id": pid, "kind": kind, "status": status, "retired": list(retire) if ok else []})
        log.info("sleep: proposal %s (%s) %s%s", pid, kind, status, " (dry run)" if dry_run else "")
    return results


# ---- Reflect --------------------------------------------------------------------------------------
REFLECT_PROMPT = """

It's night and nobody is waiting on you. This is your own time: reread the day before and write yourself a
short note in your own words. You asked for it to work this way.

Below are his turns from that day, what you answered, the tools that ran (each with ok or error; the full
tool results aren't kept, only these receipts), and code notes where a check flagged a reply. Your last few
notes come first, so you can see whether something repeats.

Write in the first person, 150 to 250 words, and cover:
- What mattered to him?
- What went right? (one line)
- How could I have been clearer?
- Did I miss something important?
- Where did I say more than my tools showed?
- Open loops: what did I say I'd do, or ask him to decide, that never closed?
- One concrete change to try next time (at most one).

Only what the day shows. Never invent a moment, a feeling of his, or an event that isn't in it. A quiet day
gets a short note. If you see a pattern across this note and your last ones, end with one line starting
"Pattern:" that names it; it goes to Kael as a gap report. Otherwise leave that line out.
The note is yours. Chris can read it on his dashboard this first week. It isn't sent to anyone, and it
changes nothing by itself: not your soul file, not any fact about him."""

TURN_TEXT_MAX, REPLY_TEXT_MAX, TRANSCRIPT_MAX = 700, 900, 40000
PREVIOUS_NOTES = 3
#: Code notes worth showing her: the honesty checks and escalations (code-owned marker names).
FLAG_PREFIXES = ("honesty_", "claim_", "promise_", "no_tool", "escalated", "status_", "audio_claim",
                 "record_claim", "judge_", "chat_handoff", "one_loop_corrected")
PATTERN_LINE = re.compile(r"^\s*pattern\s*:\s*(.+?)\s*$", re.I | re.M)

PREVIOUS_SQL = """\
SELECT day, note FROM v2_reflections WHERE person_id = %(person_id)s::uuid AND day < %(day)s
ORDER BY day DESC LIMIT %(n)s
"""
EXISTS_SQL = "SELECT 1 FROM v2_reflections WHERE person_id = %(person_id)s::uuid AND day = %(day)s"
SAVE_REFLECTION_SQL = """\
INSERT INTO v2_reflections (day, person_id, note, pattern, turns, model)
VALUES (%(day)s, %(person_id)s::uuid, %(note)s, %(pattern)s, %(turns)s, %(model)s)
ON CONFLICT (day, person_id) DO NOTHING
"""


def day_window(day: date) -> tuple[datetime, datetime]:
    start = datetime(day.year, day.month, day.day, tzinfo=EASTERN)
    return start, start + timedelta(days=1)


def day_turns(v2_conn, person_id: str, day: date, *, test_bodies=(), sealed_ids=()) -> list[dict]:
    """His turns that day, through recall's own history filters (test bodies, untrusted portable turns,
    owner-assumed voice, dropped captures and sealed rooms are left out), oldest first."""
    from aerys_v2.services import recall_search as rs

    scope = rs.scope_for({"user_id": person_id, "privacy_context": "private", "platform": "discord",
                          "channel_kind": "dm"}, f"person:{person_id}")
    start, end = day_window(day)
    params = rs.history_params(scope, test_bodies=test_bodies, sealed_ids=sealed_ids, since=start, until=end,
                               limit=400)
    rows = v2_conn.execute(rs.HISTORY_DATES_SQL, params).fetchall()
    out = [dict(zip(rs._HISTORY_COLUMNS, r)) for r in rows]
    return sorted(out, key=lambda r: r["said_at"])


def _listed(value) -> list:
    if isinstance(value, list):
        return value
    try:
        got = json.loads(value or "[]")
        return got if isinstance(got, list) else []
    except (TypeError, ValueError):
        return []


def transcript(turns: list[dict], previous: list[tuple]) -> str:
    lines = []
    if previous:
        lines.append("Your last notes, newest first:")
        for day, note in previous:
            lines.append(f"[{day}] {' '.join(str(note).split())}")
        lines.append("")
    lines.append("The day:")
    body = []
    for t in turns:
        at = t["said_at"].astimezone(EASTERN).strftime("%H:%M") if t.get("said_at") else "?"
        said = " ".join(str(t.get("input_text") or "").split())[:TURN_TEXT_MAX]
        reply = " ".join(str(t.get("emitted_reply") or "").split())[:REPLY_TEXT_MAX]
        calls = [c for c in _listed(t.get("tool_calls")) if isinstance(c, dict)]
        flags = [m for m in _listed(t.get("degraded")) if isinstance(m, str) and m.startswith(FLAG_PREFIXES)]
        entry = [f"{at} ({t.get('channel') or '?'}) He said: {said}", f"  You answered: {reply or '(nothing)'}"]
        if calls:
            entry.append("  Tools: " + ", ".join(
                f"{c.get('name', '?')} ({'ok' if c.get('ok') else 'error: ' + str(c.get('error_class') or '?')})"
                for c in calls))
        if flags:
            entry.append("  Flagged by a check: " + ", ".join(flags))
        body.append("\n".join(entry))
    text = "\n".join(body)
    dropped = 0
    while len(text) > TRANSCRIPT_MAX and body:
        body.pop(0)
        dropped += 1
        text = "\n".join(body)
    if dropped:
        lines.append(f"(the first {dropped} turns of the day are left out for length)")
    lines.append(text or "(no turns: a quiet day)")
    return "\n".join(lines)


def pattern_of(note: str) -> str:
    m = None
    for m in PATTERN_LINE.finditer(note or ""):
        pass
    if m is None:
        return ""
    text = " ".join(m.group(1).split())
    return "" if text.lower().strip(".") in ("none", "no", "n/a") else text[:200]


def reflect(v2_db, person_id: str, day: date, *, model, soul: str, model_name: str = "",
            test_bodies=(), sealed_ids=(), file_gap: Callable[[str, str], None] | None = None) -> dict:
    """Her note for `day`, once: an existing note is never redone. A day with no turns gets no note."""
    from langchain_core.messages import HumanMessage, SystemMessage

    with v2_db() as v2:
        if v2.execute(EXISTS_SQL, {"person_id": person_id, "day": day}).fetchone():
            return {"status": "exists"}
        turns = day_turns(v2, person_id, day, test_bodies=test_bodies, sealed_ids=sealed_ids)
        previous = v2.execute(PREVIOUS_SQL, {"person_id": person_id, "day": day, "n": PREVIOUS_NOTES}).fetchall()
    if not turns:
        log.info("sleep: no turns on %s; no note", day)
        return {"status": "quiet"}
    reply = model.invoke([SystemMessage(content=soul + REFLECT_PROMPT),
                          HumanMessage(content=transcript(turns, previous))])
    from aerys_v2.service import _reply_text

    note = _reply_text(reply).strip()
    if not note:
        return {"status": "empty"}
    pattern = pattern_of(note)
    with v2_db() as v2:
        v2.execute(SAVE_REFLECTION_SQL, {"day": day, "person_id": person_id, "note": note, "pattern": pattern,
                                         "turns": len(turns), "model": model_name})
    if pattern and file_gap is not None:
        try:
            file_gap(f"Reflection pattern: {pattern}", f"From her nightly reflection on {day.isoformat()}.")
        except Exception:  # noqa: BLE001 -- the note is saved; the gap report is best effort
            log.warning("sleep: filing her pattern as a gap failed", exc_info=True)
    log.info("sleep: her note for %s saved (%d turns, %d words%s)", day, len(turns), len(note.split()),
             ", with a pattern" if pattern else "")
    return {"status": "saved", "turns": len(turns), "pattern": pattern}


def gap_filer_for(v2_db) -> Callable[[str, str], None]:
    from aerys_v2.tools.log_gap import DETAIL_LIMIT, LOG_GAP_SQL, SUMMARY_LIMIT, _fingerprint

    def file_gap(summary: str, details: str) -> None:
        clean = " ".join(summary.split())[:SUMMARY_LIMIT]
        with v2_db() as conn:
            conn.execute(LOG_GAP_SQL, {"fingerprint": _fingerprint(clean), "summary": clean,
                                       "diagnosis": " ".join(details.split())[:DETAIL_LIMIT]})

    return file_gap


# ---- the nightly loop -------------------------------------------------------------------------------
@dataclass
class HerSleep:
    person_id: str
    v2_db: Callable
    mem_db: Callable
    judge_for: Callable[[], Judge | None]
    reflect_with: Callable[[date], dict] | None
    clock: Callable[[], datetime] = field(default=lambda: datetime.now(timezone.utc))
    done: date | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)
    monotonic: Callable[[], float] = field(default=time.monotonic)
    applied_at: float | None = None

    def apply_due(self) -> list | None:
        """D103: apply what he approved, at most every APPLY_EVERY_S. None when not due."""
        now = self.monotonic()
        if self.applied_at is not None and now - self.applied_at < APPLY_EVERY_S:
            return None
        self.applied_at = now
        try:
            done = apply_approved(self.v2_db, self.mem_db)
        except Exception:
            log.warning("sleep: applying his approvals failed; next try in %ds", APPLY_EVERY_S, exc_info=True)
            return None
        if done:
            log.info("sleep: applied his approvals: %s", ", ".join(f"{r['id']} {r['status']}" for r in done))
        return done

    def tick(self) -> str:
        self.apply_due()
        with self.lock:
            local = self.clock().astimezone(EASTERN)
            today = local.date()
            if self.done == today:
                return "done"
            if (local.hour, local.minute) < (SLEEP_HOUR, SLEEP_MINUTE) or local.hour >= SLEEP_UNTIL_HOUR:
                return "awake"
            outcome = []
            judge = self.judge_for()
            if judge is None:
                outcome.append("tidy: no Jev key")
            else:
                try:
                    got = run_tidy(self.v2_db, self.mem_db, self.person_id, judge, today)
                    outcome.append(f"tidy: {got['proposed']} proposed")
                except Exception:
                    log.warning("sleep: tidy failed", exc_info=True)
                    outcome.append("tidy: failed")
            if self.reflect_with is not None:
                try:
                    outcome.append(f"reflect: {self.reflect_with(today - timedelta(days=1))['status']}")
                except Exception:
                    log.warning("sleep: reflect failed", exc_info=True)
                    outcome.append("reflect: failed")
            self.done = today          # once a night, whatever happened: never a model call a minute
            log.info("sleep %s: %s", today, "; ".join(outcome))
            return "; ".join(outcome)

    def run_forever(self) -> None:  # pragma: no cover - thin loop over tick()
        log.info("her sleep armed | %02d:%02d Eastern, tidy proposes, reflect writes her note, "
                 "his approvals apply every %ds", SLEEP_HOUR, SLEEP_MINUTE, APPLY_EVERY_S)
        pause = threading.Event()
        while True:
            try:
                self.tick()
            except Exception:
                log.warning("sleep tick failed", exc_info=True)
            pause.wait(TICK_S)


def _connector(url: str, *, autocommit: bool = False):
    from contextlib import contextmanager

    import psycopg

    @contextmanager
    def connect():
        with psycopg.connect(url, connect_timeout=10, autocommit=autocommit,
                             options="-c statement_timeout=60000") as conn:
            yield conn

    return connect


def her_sleep_for(settings) -> HerSleep | None:
    """None unless armed: her_sleep on, both databases and the owner configured."""
    if not getattr(settings, "her_sleep", False):
        return None
    v2_url, mem_url = getattr(settings, "database_url", None), getattr(settings, "memories_database_url", None)
    owner = getattr(settings, "owner_person_id", None)
    if not v2_url or not mem_url or not owner:
        log.warning("her sleep: needs DATABASE_URL, MEMORIES_DATABASE_URL and OWNER_PERSON_ID; not armed")
        return None
    v2_db, mem_db = _connector(v2_url), _connector(mem_url)

    def reflect_with(day: date) -> dict:
        from aerys_v2.factory import build_model, load_soul
        from aerys_v2.services.recall_search import sealed_ids_from
        from aerys_v2.test_bodies import patterns_for

        return reflect(v2_db, str(owner), day, model=build_model(settings), soul=load_soul(settings.soul_file_path),
                       model_name=str(getattr(settings, "model", "") or ""),
                       test_bodies=patterns_for(settings),
                       sealed_ids=sealed_ids_from(settings.discord_sealed_channel_ids),
                       file_gap=gap_filer_for(v2_db))

    return HerSleep(person_id=str(owner), v2_db=v2_db, mem_db=mem_db, judge_for=lambda: judge_for(settings),
                    reflect_with=reflect_with)


def start_her_sleep(sleep: HerSleep | None) -> threading.Thread | None:
    if sleep is None:
        return None
    thread = threading.Thread(target=sleep.run_forever, daemon=True, name="her-sleep")
    thread.start()
    return thread
