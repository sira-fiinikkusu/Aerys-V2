"""Phase 3 in shadow: Jev's privacy verdict is recorded beside the metered judge's, never
the content, and never on the judge's path. Chris 2026-09-20: Phase 3 after the 10 am build."""
import threading
import time
from types import SimpleNamespace

from aerys_v2.config import Settings
from aerys_v2.privacy_shadow import privacy_shadow_for, shadow_privacy_fn
from aerys_v2.reflex import PRIVACY_CATEGORY_QUESTION, PRIVACY_QUESTION, ReflexClient
from aerys_v2.workers.privacy_report import format_report, summarize


class FakeReflex:
    def __init__(self, p=0.95, error=None, delay=0.0):
        self.p, self.error, self.delay, self.seen = p, error, delay, []

    def judge_privacy(self, text):
        self.seen.append(text)
        time.sleep(self.delay)
        if self.error:
            return {"error": self.error, "latency_ms": 5}
        return {"p_public": self.p, "model": "jev", "latency_ms": 7, "category": "health", "p_ordinary": 0.03}


class Sink:
    def __init__(self):
        self.rows, self.event = [], threading.Event()

    def __call__(self, row):
        self.rows.append(row); self.event.set()


def test_judge_verdict_is_returned_first_and_unchanged_and_both_are_recorded_without_content():
    reflex, sink = FakeReflex(p=0.95), Sink()
    classify = shadow_privacy_fn(lambda t: "private", reflex, sink)
    assert classify("my blood pressure was 150 over 95") == "private"
    assert sink.event.wait(2)
    row = sink.rows[0]
    assert row["judge"] == "private" and row["jev_p_public"] == 0.95 and row["jev_error"] is None
    assert row["keyword_hit"] is False and row["sample_len"] == len("my blood pressure was 150 over 95")
    assert row["jev_category"] == "health" and row["jev_p_ordinary"] == 0.03
    assert "blood" not in str(row)                        # no content in the audit row


def test_a_slow_or_dead_jev_never_delays_or_changes_the_verdict():
    reflex, sink = FakeReflex(error="timeout", delay=0.3), Sink()
    classify = shadow_privacy_fn(lambda t: "public", reflex, sink)
    t0 = time.monotonic()
    assert classify("we should grab tacos thursday") == "public"
    assert time.monotonic() - t0 < 0.2                    # the judge path did not wait for Jev
    assert sink.event.wait(2) and sink.rows[0]["jev_error"] == "timeout"


def test_keyword_hit_is_recorded():
    reflex, sink = FakeReflex(), Sink()
    classify = shadow_privacy_fn(lambda t: "private", reflex, sink)
    classify("the wifi password is hunter2")
    assert sink.event.wait(2) and sink.rows[0]["keyword_hit"] is True


def test_arming_needs_reflex_and_the_flag_and_a_judge():
    judge = lambda t: "public"  # noqa: E731
    on = Settings(_env_file=None, anthropic_api_key="t", reflex_mode="live", typesafe_api_key="k")
    off = Settings(_env_file=None, anthropic_api_key="t", reflex_privacy_shadow=False)
    assert privacy_shadow_for(on, judge, FakeReflex()) is not judge
    assert privacy_shadow_for(off, judge, FakeReflex()) is judge
    assert privacy_shadow_for(on, judge, None) is judge
    assert privacy_shadow_for(on, None, FakeReflex()) is None


def test_reflex_client_privacy_question_is_bounded_and_never_raises():
    class Client:
        def system_one(self, **kw):
            assert kw["questions"]["public"] is PRIVACY_QUESTION                  # one call, both questions
            assert kw["questions"]["category"] is PRIVACY_CATEGORY_QUESTION
            assert "content" in kw["state"]
            return SimpleNamespace(answers={"public": SimpleNamespace(noul=0.12),
                                            "category": SimpleNamespace(choice="money", probabilities={"ordinary": 0.2, "money": 0.8})},
                                   model="jev-1.13.0")

    out = ReflexClient(client=Client()).judge_privacy("x" * 5000)
    assert out["p_public"] == 0.12 and "latency_ms" in out
    assert out["category"] == "money" and out["p_ordinary"] == 0.2

    class NoCategory:  # a missing or malformed category answer never costs the noul
        def system_one(self, **kw):
            return SimpleNamespace(answers={"public": SimpleNamespace(noul=0.5)}, model="jev-1.13.0")

    out = ReflexClient(client=NoCategory()).judge_privacy("hi")
    assert out["p_public"] == 0.5 and "category" not in out and "category_error" in out

    class Boom:
        def system_one(self, **kw):
            raise RuntimeError("down")

    assert "error" in ReflexClient(client=Boom()).judge_privacy("hi")


def test_report_counts_the_dangerous_direction_separately():
    rows = [
        {"judge": "private", "keyword_hit": False, "jev_p_public": 0.95, "jev_error": None, "jev_latency_ms": 200, "sample_len": 10},  # would leak
        {"judge": "public", "keyword_hit": False, "jev_p_public": 0.3, "jev_error": None, "jev_latency_ms": 210, "sample_len": 10},   # over-hide
        {"judge": "public", "keyword_hit": False, "jev_p_public": 0.97, "jev_error": None, "jev_latency_ms": 190, "sample_len": 10},  # agree
        {"judge": "private", "keyword_hit": True, "jev_p_public": 0.1, "jev_error": None, "jev_latency_ms": 205, "sample_len": 10},   # agree
        {"judge": "private", "keyword_hit": False, "jev_p_public": None, "jev_error": "timeout", "jev_latency_ms": 600, "sample_len": 10},
    ]
    s = summarize(rows)
    assert s["n"] == 5 and s["judged"] == 4 and s["errors"] == 1 and s["keyword_hits"] == 1
    assert s["agreement"] == 0.5 and s["jev_public_but_judge_private"] == 1 and s["jev_private_but_judge_public"] == 1
    text = format_report(rows)
    assert "would leak) = 1" in text and "agreement (jev public at p>=0.9)=50.0%" in text
    assert summarize([])["agreement"] is None
    assert s["category_scored"] == 0                      # rows from before migration 013


def test_report_scores_the_category_candidate_both_directions():
    base = {"keyword_hit": False, "jev_p_public": 0.9, "jev_error": None, "jev_latency_ms": 200, "sample_len": 10}
    rows = [
        {**base, "judge": "public", "jev_category": "health", "jev_p_ordinary": 0.03},    # candidate hides, judge missed
        {**base, "judge": "private", "jev_category": "ordinary", "jev_p_ordinary": 0.9},  # candidate public, judge private
        {**base, "judge": "public", "jev_category": "ordinary", "jev_p_ordinary": 0.95},  # agree
        {**base, "judge": "public", "jev_category": None, "jev_p_ordinary": None},        # not scored
    ]
    s = summarize(rows)
    assert s["category_scored"] == 3
    assert s["category_private_but_judge_public"] == 1 and s["category_public_but_judge_private"] == 1
    assert s["category_private_by_kind"] == {"health": 1}
    assert "by kind={'health': 1}" in format_report(rows)
