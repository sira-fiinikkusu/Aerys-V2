"""D100 (Chris 11/25 12:48, "yes to both Tidy and Reflect"): her sleep.

Tidy only proposes (judgements from Jev, decisions in code), never touches her own words, events or
promises, never repeats a judged or rejected proposal, and stops at its nightly cap. The apply step re-checks
every row and retires only what still reads as judged. Reflect writes her note once a day, in the shape she
asked for (11/25 13:13), to its own table, with her last notes and the turn receipts beside her.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from aerys_v2 import sleep as sl

ME = "11111111-1111-4111-8111-111111111111"
A, B, C, D, E = (f"{c * 8}-0000-4000-8000-000000000000" for c in "abcde")
T0 = datetime(2026, 10, 1, 12, tzinfo=timezone.utc)
TODAY = date(2026, 11, 26)
NOW = datetime(2026, 11, 26, 8, tzinfo=timezone.utc)


class Jev:
    """TypeSafe faked: answer(qid, state) -> a probability, or an exception to raise. One request may carry
    several questions; asked records each (qid, state), requests counts the calls."""

    def __init__(self, answer):
        self.answer, self.asked, self.requests = answer, [], 0

    def system_one(self, *, state, questions, model=None):
        self.requests += 1
        out = {}
        for qid in questions:
            self.asked.append((qid, state))
            v = self.answer(qid, state)
            if isinstance(v, BaseException):
                raise v
            out[qid] = SimpleNamespace(noul=v)
        return SimpleNamespace(answers=out)


class Mem:
    def __init__(self, facts=(), pairs=(), claims=()):
        self.facts, self.pairs, self.claims, self.calls = list(facts), list(pairs), list(claims), []

    def execute(self, sql, params=None):
        self.calls.append((sql, params))
        rows = {sl.LIVE_FACTS_SQL: self.facts, sl.PAIRS_SQL: self.pairs, sl.CLAIMS_SQL: self.claims}.get(sql, [])
        return SimpleNamespace(fetchall=lambda: rows)


def fact(row_id, key, value, days_ago=40):
    return (row_id, key, f"{key}: {value}", NOW - timedelta(days=days_ago), True)


def judge(answer, budget=150):
    return sl.Judge(Jev(answer), budget=budget)


def run(mem, answer, **kw):
    j = judge(answer, kw.pop("budget", 150))
    out = sl.tidy(mem, ME, j, known=kw.pop("known", {}), busy=kw.pop("busy", set()), today=TODAY)
    return out, j


def proposed(out):
    return [p for p in out if p.status == "proposed"]


# ---- Tidy -------------------------------------------------------------------------------------------

def test_a_duplicate_keeps_the_more_specific_and_retires_the_other():
    mem = Mem([fact(A, "pet.cat_pip", "Has a cat named Pip", 50),
               fact(B, "relationship.pet_pip", "Has a 13-year-old tabby named Pip", 10)], [(A, B, 0.2)])
    out, j = run(mem, lambda q, s: .95 if q == "same" else .9)
    (p,) = proposed(out)
    assert (p.kind, p.retire_table, p.retire_ids, p.keep_ids) == ("duplicate", "memories", [A], [B])
    assert p.snapshot == {A: "pet.cat_pip: Has a cat named Pip", B: "relationship.pet_pip: Has a 13-year-old tabby named Pip"}
    assert [q for q, _ in j.client.asked] == ["same"]           # a duplicate is never also asked "both true?"


def test_not_the_same_is_recorded_clear_and_same_kind_pairs_are_asked_both_true():
    mem = Mem([fact(A, "basic.home", "Lives in Ohio", 300), fact(B, "basic.home_new", "Moved to Florida", 5)],
              [(A, B, 0.2)])
    out, j = run(mem, lambda q, s: .1 if q == "same" else .05)
    assert [(p.kind, p.status) for p in out] == [("duplicate", "clear"), ("contradiction", "proposed")]
    contra = out[1]
    assert contra.retire_ids == [A] and contra.keep_ids == [B]   # the older goes, the newer stays
    assert j.client.asked[1][1]["older"]["text"] == "Lives in Ohio"


def test_different_kinds_are_never_asked_both_true():
    mem = Mem([fact(A, "work.role", "Works nights"), fact(B, "interest.night", "Likes the night sky")],
              [(A, B, 0.3)])
    out, j = run(mem, lambda q, s: .0)
    assert out == [] and j.client.asked == []


def test_her_words_events_and_logs_are_never_candidates():
    mem = Mem()
    run(mem, lambda q, s: 1.0)
    params = {sql: p for sql, p in mem.calls}
    for kinds in (params[sl.PAIRS_SQL]["kinds"], params[sl.LIVE_FACTS_SQL]["kinds"]):
        assert not {"aerys", "a2a_kael", "gap", "exec", "skill", "session", "portable"} & set(kinds)
    assert "event" not in params[sl.PAIRS_SQL]["kinds"]          # the same event on another date is a second one
    assert "event" in params[sl.LIVE_FACTS_SQL]["kinds"]         # but a passed plan may be stale


def test_a_stale_plan_is_proposed_only_when_plan_shaped_old_and_passed():
    mem = Mem([fact(A, "event.market", "Plans to go to the farmers market Oct 29", 40),
               fact(B, "interest.market", "Likes farmers markets", 40),
               fact(C, "decision.trip", "Going to book the trip tomorrow", 0)])
    out, j = run(mem, lambda q, s: .1 if q == "lasting" else .93)
    (p,) = proposed(out)
    assert (p.kind, p.retire_ids, p.careful) == ("stale_plan", [A], False)
    asked = [s["line"] for q, s in j.client.asked if q == "plan_passed"]
    assert asked == ["Plans to go to the farmers market Oct 29"]   # not shaped like a plan / said today: never asked
    assert j.client.asked[0][1]["today"] == "2026-11-26"


def test_an_old_momentary_claim_is_proposed_for_the_claims_table():
    mem = Mem(claims=[(C, "session.status", "Restarted the system; doing final checks", T0),
                      (D, "basic.kids", "Has two kids", T0)])
    out, _ = run(mem, lambda q, s: .05 if q == "lasting" else (.9 if "Restarted" in s["line"] else .05))
    assert [(p.kind, p.retire_table, p.retire_ids, p.status) for p in out] == [
        ("old_claim", "core_claim", [C], "proposed"), ("old_claim", "core_claim", [D], "clear")]


def test_a_line_that_also_holds_something_lasting_is_proposed_but_marked_careful():
    """A side agent's catch (11/25): "recently married or about to be", a name tied to an old trip plan and a
    launch he watched were among the first proposals. Still his to decide, never approved in bulk."""
    mem = Mem([fact(A, "relationship.status", "Recently married or about to be married, upcoming honeymoon", 200),
               fact(B, "decision.bed", "Going to bed", 30)],
              claims=[(C, "event.launch", "Watched the launch from home", T0)])
    lasting = {"Recently married or about to be married, upcoming honeymoon": .93, "Going to bed": .04,
               "Watched the launch from home": .88}
    out, j = run(mem, lambda q, s: lasting[s["line"]] if q == "lasting" else .95)
    marks = {p.retire_ids[0]: p.careful for p in proposed(out)}
    assert marks == {A: True, B: False, C: True}
    assert all("read it before approving" in p.reason for p in proposed(out) if p.careful)
    assert j.client.requests == 3                     # the lasting question rides in the same request
    assert all(p.row(ME)["careful"] is p.careful for p in out)


def test_no_lasting_answer_counts_as_careful():
    mem = Mem([fact(A, "decision.bed", "Going to bed", 30)])
    out, _ = run(mem, lambda q, s: .95 if q == "plan_passed" else None)   # Jev left "lasting" unanswered
    (p,) = proposed(out)
    assert p.careful is True


def test_judged_and_busy_rows_are_never_asked_again():
    mem = Mem([fact(A, "pet.a", "Has a cat named Pip", 50), fact(B, "pet.b", "Has a cat named Pip!", 10)],
              [(A, B, 0.1)])
    first, _ = run(mem, lambda q, s: .97)
    assert [p.status for p in first] == ["proposed"]
    again, j = run(mem, lambda q, s: .97, known={p.key: "rejected" for p in first})   # he said no
    assert again == [] and j.client.asked == []
    busy, j2 = run(mem, lambda q, s: .97, busy={A})
    assert busy == [] and j2.client.asked == []


def test_the_key_is_stable_and_changes_with_the_text():
    k1 = sl.proposal_key("duplicate", [B, A], [], ["x", "y"])
    assert k1 == sl.proposal_key("duplicate", [A, B], [], ["x", "y"])
    assert k1 != sl.proposal_key("duplicate", [A, B], [], ["x", "z"])


def test_the_nightly_cap_bites_loudly_and_records_nothing_it_did_not_judge():
    mem = Mem(claims=[(f"{i:08d}-0000-4000-8000-000000000000", "k", f"claim {i}", T0) for i in range(5)])
    out, j = run(mem, lambda q, s: .9, budget=2)
    assert len(out) == 2 and j.asked == 2 and j.over == 3


@pytest.mark.parametrize("bad", [RuntimeError("jev down"), True, 1.5, float("nan")])
def test_a_failed_or_bad_judgement_proposes_nothing(bad):
    mem = Mem(claims=[(C, "k", "Restarted the system", T0)])
    out, _ = run(mem, lambda q, s: bad)
    assert out == []


# ---- applying what he approved --------------------------------------------------------------------------

class V2:
    def __init__(self, approved=()):
        self.approved, self.calls = list(approved), []

    def execute(self, sql, params=None):
        self.calls.append((sql, params))
        return SimpleNamespace(fetchall=lambda: self.approved if sql == sl.APPROVED_SQL else [],
                               fetchone=lambda: None, rowcount=1)


class Rows:
    """The prod side for apply: memory and claim rows by id."""

    def __init__(self, memories=None, claims=None):
        self.memories, self.claims, self.calls = dict(memories or {}), dict(claims or {}), []

    def execute(self, sql, params=None):
        self.calls.append((sql, params))
        if sql == sl.MEMORY_ROW_SQL:
            row = self.memories.get(params["id"])
        elif sql == sl.CLAIM_ROW_SQL:
            row = self.claims.get(params["id"])
        else:
            row = None
        return SimpleNamespace(fetchone=lambda: row)


def db(conn):
    @contextmanager
    def connect():
        yield conn
    return connect


def approved(kind="duplicate", table="memories", retire=(A,), keep=(B,), snapshot=None):
    snap = snapshot or {A: "pet.a: Has a cat named Pip", B: "pet.b: Has a 13-year-old cat named Pip"}
    return (7, ME, kind, table, list(retire), list(keep), snap)


LIVE_A = ("pet.a", "pet.a: Has a cat named Pip", True, [])
LIVE_B = ("pet.b", "pet.b: Has a 13-year-old cat named Pip", True, [])


def retires(rows):
    return [p for s, p in rows.calls if s in (sl.RETIRE_MEMORY_SQL, sl.RETIRE_CLAIM_SQL)]


def test_an_approved_proposal_that_still_reads_as_judged_is_retired_and_marked_applied():
    v2, rows = V2([approved()]), Rows({A: LIVE_A, B: LIVE_B})
    (r,) = sl.apply_approved(db(v2), db(rows))
    assert r["status"] == "applied" and retires(rows) == [{"id": A, "person_id": ME, "tag": "tidy:duplicate"}]
    assert (sl.MARK_SQL, {"status": "applied", "id": 7}) in v2.calls


@pytest.mark.parametrize("memories", [
    {A: ("pet.a", "pet.a: Has a cat named Pip and a dog", True, []), B: LIVE_B},   # its text changed
    {A: LIVE_A, B: ("pet.b", "pet.b: Has a 13-year-old cat named Pip", False, [])},  # the kept one is gone
    {B: LIVE_B},                                                                      # the row is gone
])
def test_anything_changed_since_it_was_judged_marks_it_stale_and_touches_nothing(memories):
    v2, rows = V2([approved()]), Rows(memories)
    (r,) = sl.apply_approved(db(v2), db(rows))
    assert r["status"] == "stale" and retires(rows) == []
    assert (sl.MARK_SQL, {"status": "stale", "id": 7}) in v2.calls


def test_a_row_this_tidy_already_retired_counts_as_applied_without_a_second_write():
    done = ("pet.a", "pet.a: Has a cat named Pip", False, ["tidy:duplicate"])
    v2, rows = V2([approved()]), Rows({A: done, B: LIVE_B})
    (r,) = sl.apply_approved(db(v2), db(rows))
    assert r["status"] == "applied" and retires(rows) == []


def test_an_old_claim_is_retired_by_status_the_9_21_way():
    snap = {C: "session.status: Restarted the system"}
    v2 = V2([approved("old_claim", "core_claim", (C,), (), snap)])
    rows = Rows(claims={C: ("session.status", "Restarted the system", "approved")})
    (r,) = sl.apply_approved(db(v2), db(rows))
    assert r["status"] == "applied" and retires(rows) == [{"id": C, "person_id": ME}]
    assert "status = 'retired'" in sl.RETIRE_CLAIM_SQL and "deleted_at" not in sl.RETIRE_CLAIM_SQL


def test_a_dry_run_changes_nothing():
    v2, rows = V2([approved()]), Rows({A: LIVE_A, B: LIVE_B})
    (r,) = sl.apply_approved(db(v2), db(rows), dry_run=True)
    assert r["status"] == "applied" and retires(rows) == [] and not [s for s, _ in v2.calls if s == sl.MARK_SQL]


def test_nothing_is_ever_hard_deleted():
    import re
    for sql in (sl.RETIRE_MEMORY_SQL, sl.RETIRE_CLAIM_SQL, sl.SAVE_SQL, sl.MARK_SQL, sl.SAVE_REFLECTION_SQL):
        assert not re.search(r"\bDELETE\s+FROM\b|\bTRUNCATE\b|\bDROP\b", sql, re.I)


# ---- Reflect ----------------------------------------------------------------------------------------------

class Model:
    def __init__(self, reply="Today he cared about the router.\nPattern: I promise checks before running them."):
        self.reply, self.calls = reply, []

    def invoke(self, messages):
        self.calls.append(messages)
        return SimpleNamespace(content=self.reply)


def turn(text="the router keeps dropping", reply="I'll check it", at=None, tools=(), degraded=()):
    return {"id": 1, "channel": "discord_dm", "channel_id": "9", "person_id": ME, "display_name": "Chris",
            "input_text": text, "emitted_reply": reply, "sealed_room": None, "guard_verdict": None,
            "degraded": list(degraded), "tool_calls": list(tools), "said_at": at or datetime(2026, 11, 25, 19, tzinfo=timezone.utc)}


class ReflectDb:
    def __init__(self, turns=(), previous=(), exists=False):
        from aerys_v2.services import recall_search as rs
        self.rs, self.turns, self.previous, self.exists, self.calls = rs, list(turns), list(previous), exists, []

    def execute(self, sql, params=None):
        self.calls.append((sql, params))
        if sql == self.rs.HISTORY_DATES_SQL:
            rows = [tuple(t[c] for c in self.rs._HISTORY_COLUMNS) for t in self.turns]
        elif sql == sl.PREVIOUS_SQL:
            rows = self.previous
        else:
            rows = []
        one = (1,) if (sql == sl.EXISTS_SQL and self.exists) else None
        return SimpleNamespace(fetchall=lambda: rows, fetchone=lambda: one, rowcount=1)


def test_her_note_is_written_once_in_the_shape_she_asked_for():
    conn, model, gaps = ReflectDb([turn()], previous=[(date(2026, 11, 24), "Yesterday I rushed.")]), Model(), []
    got = sl.reflect(db(conn), ME, date(2026, 11, 25), model=model, soul="SOUL",
                     file_gap=lambda s, d: gaps.append((s, d)))
    assert got["status"] == "saved" and got["pattern"] == "I promise checks before running them."
    system, human = model.calls[0]
    for asked_for in ("What went right? (one line)", "Open loops", "One concrete change to try next time (at most one)",
                      "150 to 250 words", "Pattern:", "never invent", "Chris can read it on his dashboard"):
        assert asked_for.lower() in system.content.lower(), asked_for
    assert system.content.startswith("SOUL")
    assert "[2026-11-24] Yesterday I rushed." in human.content and "He said: the router keeps dropping" in human.content
    (save,) = [p for s, p in conn.calls if s == sl.SAVE_REFLECTION_SQL]
    assert save["turns"] == 1 and save["pattern"] == "I promise checks before running them."
    assert gaps == [("Reflection pattern: I promise checks before running them.",
                     "From her nightly reflection on 2026-11-25.")]


def test_an_existing_note_is_never_redone_and_a_quiet_day_gets_none():
    model = Model()
    assert sl.reflect(db(ReflectDb([turn()], exists=True)), ME, date(2026, 11, 25), model=model, soul="")["status"] == "exists"
    assert sl.reflect(db(ReflectDb([])), ME, date(2026, 11, 25), model=model, soul="")["status"] == "quiet"
    assert model.calls == []


def test_the_day_is_read_through_recalls_own_filters_for_his_eastern_day():
    conn = ReflectDb([turn()])
    sl.reflect(db(conn), ME, date(2026, 11, 25), model=Model("ok"), soul="", test_bodies=["%ember%"], sealed_ids={"5"})
    (params,) = [p for s, p in conn.calls if s == conn.rs.HISTORY_DATES_SQL]
    assert params["person_id"] == ME and params["test_bodies"] == ["%ember%"] and params["sealed_ids"] == ["5"]
    assert params["current_sealed"] == ""                         # sealed rooms stay out
    assert params["since"].isoformat() == "2026-11-25T00:00:00-05:00"
    assert params["until"].isoformat() == "2026-11-26T00:00:00-05:00"


def test_flagged_replies_carry_their_receipts_and_noise_markers_stay_out():
    t = turn(tools=[{"name": "search_web", "ok": True}, {"name": "home_control", "ok": False, "error_class": "timeout"}],
             degraded=["honesty_unsure", "untrusted_read", "text_one_loop"])
    text = sl.transcript([t], [])
    assert "Tools: search_web (ok), home_control (error: timeout)" in text
    assert "Flagged by a check: honesty_unsure" in text and "untrusted_read" not in text


def test_a_long_day_drops_its_earliest_turns_and_says_so():
    turns = [turn(text=f"turn {i} " + "x" * 600, at=datetime(2026, 11, 25, 12, i, tzinfo=timezone.utc)) for i in range(60)]
    text = sl.transcript(turns, [])
    assert len(text) <= sl.TRANSCRIPT_MAX + 200 and "turns of the day are left out for length" in text
    assert "turn 59" in text and "turn 0 " not in text


@pytest.mark.parametrize("note,pattern", [("no pattern here", ""), ("x\nPattern: none", ""),
                                          ("x\npattern:  I over-promise  \n", "I over-promise")])
def test_the_pattern_line(note, pattern):
    assert sl.pattern_of(note) == pattern


# ---- the nightly loop ------------------------------------------------------------------------------------

def sleeper(at, *, tidy_ok=True, judge_obj=object()):
    calls = []
    clock = {"now": at}
    s = sl.HerSleep(person_id=ME, v2_db=None, mem_db=None, judge_for=lambda: judge_obj,
                    reflect_with=lambda day: calls.append(("reflect", day)) or {"status": "saved"},
                    clock=lambda: clock["now"])
    return s, calls, clock


def test_it_sleeps_once_a_night_after_three_thirty_eastern(monkeypatch):
    ran = []
    monkeypatch.setattr(sl, "run_tidy", lambda *a: ran.append(a[-1]) or {"proposed": 2})
    s, calls, clock = sleeper(datetime(2026, 11, 26, 8, 0, tzinfo=timezone.utc))      # 03:00 Eastern
    assert s.tick() == "awake" and ran == [] and calls == []
    clock["now"] = datetime(2026, 11, 26, 8, 31, tzinfo=timezone.utc)                  # 03:31 Eastern
    assert s.tick() == "tidy: 2 proposed; reflect: saved"
    assert ran == [date(2026, 11, 26)] and calls == [("reflect", date(2026, 11, 25))]
    assert s.tick() == "done"


def test_a_failed_tidy_still_lets_her_reflect_and_no_key_skips_tidy(monkeypatch):
    monkeypatch.setattr(sl, "run_tidy", lambda *a: (_ for _ in ()).throw(RuntimeError("nas down")))
    s, calls, _ = sleeper(datetime(2026, 11, 26, 9, 0, tzinfo=timezone.utc))
    assert s.tick() == "tidy: failed; reflect: saved" and calls
    s2, _, _ = sleeper(datetime(2026, 11, 26, 9, 0, tzinfo=timezone.utc), judge_obj=None)
    assert s2.tick() == "tidy: no Jev key; reflect: saved"


def test_after_noon_a_late_start_waits_for_the_next_night():
    s, calls, _ = sleeper(datetime(2026, 11, 26, 18, 0, tzinfo=timezone.utc))         # 13:00 Eastern
    assert s.tick() == "awake" and calls == []


def test_it_is_armed_only_with_both_databases_and_the_owner():
    base = dict(her_sleep=True, database_url="postgresql://v2", memories_database_url="postgresql://aerys",
                owner_person_id=ME)
    assert sl.her_sleep_for(SimpleNamespace(**base)) is not None
    for missing in ("database_url", "memories_database_url", "owner_person_id"):
        assert sl.her_sleep_for(SimpleNamespace(**{**base, missing: None})) is None
    assert sl.her_sleep_for(SimpleNamespace(**{**base, "her_sleep": False})) is None


def test_the_migration_keeps_proposals_stable_and_reflections_one_a_day():
    sql = (Path(__file__).resolve().parents[1] / "db/migrations/034_her_sleep.sql").read_text()
    assert "proposal_key TEXT NOT NULL UNIQUE" in sql and "PRIMARY KEY (day, person_id)" in sql
    assert "'proposed', 'approved', 'rejected', 'applied', 'stale', 'clear'" in sql
    import re
    assert not re.search(r"\bDELETE\s+FROM\b|\bTRUNCATE\b|\bDROP\s+(?:TABLE|COLUMN|INDEX)\b", sql, re.I)
