"""M1 (#259 step 3, 2026-11-25): a fact he STATES inside a question is kept when Jev says he states it
as true (>= 0.7). "I just bought a Ram, what oil should I use?" keeps the truck; "is my dad visiting?",
"what if I bought a Ram?" and a probe keep nothing. Jev judges, this code decides; any Jev failure keeps
nothing (the old rule)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from aerys_v2.workers import extraction as ex
from aerys_v2.workers.extraction import jev_stated_check, quoted_observations, run_live_extraction
from test_extraction import CHRIS, FakeConn, FakeLlm, lease_conn, row

T0 = datetime(2026, 11, 20, 15, 0, tzinfo=timezone.utc)


def group(*texts):
    msgs = [{"id": str(i), "content": t, "reply": None, "created_at": T0} for i, t in enumerate(texts)]
    return {"person_id": CHRIS, "messages": msgs, "privacy_level": "private", "latest": msgs[-1]}


def obs(key, value, quote):
    return {"key_label": key, "value_text": value, "source_text": quote, "context": "", "event_date": "",
            "privacy_level": "private", "replaces": ""}


class Stated:
    def __init__(self, *scores):
        self.scores, self.calls = list(scores), []

    def __call__(self, message, facts):
        self.calls.append((message, list(facts)))
        return [self.scores.pop(0) if self.scores else None for _ in facts]


RAM = "I just bought a Ram 1500, what oil should I use?"


def test_a_fact_stated_inside_a_question_is_kept_when_jev_says_he_states_it():
    stated, stats = Stated(0.97), {}
    kept = quoted_observations([obs("vehicle.truck", "Bought a Ram 1500", "I just bought a Ram 1500")],
                               group(RAM), stats, stated)
    assert [o["key_label"] for o in kept] == ["vehicle.truck"]
    assert stated.calls == [(RAM, ["Bought a Ram 1500"])] and stats["stated_kept"] == 1


@pytest.mark.parametrize("score,counter", [(0.2, "stated_refused"), (0.5, "stated_refused"), (None, "stated_unavailable")])
def test_below_the_bar_or_unknown_keeps_nothing(score, counter):
    stats = {}
    assert quoted_observations([obs("vehicle.truck", "Bought a Ram 1500", "I just bought a Ram 1500")],
                               group(RAM), stats, Stated(score)) == []
    assert stats[counter] == 1


def test_without_the_check_nothing_from_a_question_line_is_kept_the_old_rule():
    stats = {}
    assert quoted_observations([obs("vehicle.truck", "Bought a Ram 1500", "I just bought a Ram 1500")],
                               group(RAM), stats, None) == []
    assert stats["unquoted"] == 1


def test_a_quote_that_is_itself_a_question_is_never_even_asked_about():
    stated, stats = Stated(0.99), {}
    assert quoted_observations([obs("relationship.dad.visit", "Dad is visiting", "is my dad visiting this week?")],
                               group("is my dad visiting this week?"), stats, stated) == []
    assert stated.calls == []


def test_a_fact_from_an_ordinary_line_never_waits_on_jev():
    stated = Stated()
    kept = quoted_observations([obs("work.shift", "Works nights", "I work nights now")],
                               group("I work nights now."), {}, stated)
    assert len(kept) == 1 and stated.calls == []


def test_a_memory_probe_line_is_never_a_candidate():
    stated = Stated(0.99)
    probe = "do you remember what my truck is? I just bought a Ram 1500"
    assert quoted_observations([obs("vehicle.truck", "Bought a Ram 1500", "I just bought a Ram 1500")],
                               group(probe), {}, stated) == []
    assert stated.calls == []


def test_one_request_per_message_with_one_question_per_fact():
    stated = Stated(0.9, 0.1)
    msg = "I just bought a Ram 1500 and sold my Civic, what oil should I use?"
    kept = quoted_observations([obs("vehicle.truck", "Bought a Ram 1500", "I just bought a Ram 1500"),
                                obs("vehicle.old", "Sold his Civic", "sold my Civic")], group(msg), {}, stated)
    assert stated.calls == [(msg, ["Bought a Ram 1500", "Sold his Civic"])]
    assert [o["key_label"] for o in kept] == ["vehicle.truck"]


def test_the_jev_client_failure_is_none_for_every_fact():
    class Boom:
        def system_one(self, **kw):
            raise TimeoutError("slow")
    assert jev_stated_check("k", client=Boom())("msg", ["a", "b"]) == [None, None]


def test_the_jev_client_reads_each_noul_and_refuses_bad_numbers():
    class Fake:
        def system_one(self, state, questions, model):
            assert state == {"message": RAM, "said_on": "Friday, Nov 20, 2026"} and set(questions) == {"f0", "f1", "f2"}
            assert "Bought a Ram 1500" in questions["f0"]["instructions"]
            return SimpleNamespace(answers={"f0": SimpleNamespace(noul=0.98), "f1": SimpleNamespace(noul=float("nan")),
                                            "f2": SimpleNamespace(noul=True)})
    assert jev_stated_check("k", client=Fake())(RAM, ["Bought a Ram 1500", "x", "y"],
                                                said_on="Friday, Nov 20, 2026") == [0.98, None, None]


def test_the_live_pass_keeps_the_stated_fact_through_the_seam():
    prod = FakeConn([("FROM memories m", []), ("FROM n8n_chat_histories", [row(CHRIS, RAM)])])
    write = FakeConn([("RETURNING id", [("n1",)]), ("FROM memories", [])])
    summary = run_live_extraction(prod, lease_conn(holder="brain"), write,
                                  FakeLlm([json.dumps([obs("vehicle.truck", "Bought a Ram 1500", "I just bought a Ram 1500")])]),
                                  lambda t: [0.5], stated=Stated(0.95))
    inserts = [p for s, p in write.calls if s.lstrip().startswith("INSERT INTO memories")]
    assert [p["key_label"] for p in inserts] == ["vehicle.truck"]
    assert summary["sources"]["prod_chat"]["stated_kept"] == 1


def test_the_worker_arms_the_check_only_with_a_key_and_the_switch_on():
    from aerys_v2.workers.__main__ import stated_check
    key = SimpleNamespace(get_secret_value=lambda: "k")
    assert stated_check(SimpleNamespace(typesafe_api_key=None, extraction_facts_in_questions=True)) is None
    assert stated_check(SimpleNamespace(typesafe_api_key=key, extraction_facts_in_questions=False)) is None
    assert callable(stated_check(SimpleNamespace(typesafe_api_key=key, extraction_facts_in_questions=True)))


def test_the_check_is_told_when_he_said_it_so_a_worked_out_date_can_be_judged():
    seen = []
    def stated(message, facts, said_on=None):
        seen.append(said_on)
        return [0.9] * len(facts)
    quoted_observations([obs("vehicle.truck", "Bought a Ram 1500 around Nov 20, 2026", "I just bought a Ram 1500")],
                        group(RAM), {}, stated)
    assert seen == ["Friday, Nov 20, 2026"]


def test_only_the_judged_fact_is_stored_never_unchecked_context_or_date():
    """Codex M1 r1: a hypothetical riding in the context must not become stored testimony."""
    msg = "I just bought a Ram 1500; what if my dad visits for Christmas?"
    o = obs("vehicle.truck", "Bought a Ram 1500", "I just bought a Ram 1500")
    o["context"], o["event_date"] = "His dad is visiting him", "2026-12-25"
    kept = quoted_observations([o], group(msg), {}, Stated(0.95))
    assert kept and kept[0]["context"] == "" and kept[0]["event_date"] == ""
    content, date = ex._compose_content(kept[0], "Nov 20")
    assert "dad" not in content.lower() and "12-25" not in content and date == "Nov 20"


def test_the_shadow_worker_passes_the_check_too():
    import inspect
    from aerys_v2.workers import __main__ as w
    src = inspect.getsource(w._run_once)
    assert src.count("stated=stated_check(settings)") == 2
