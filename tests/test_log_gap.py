"""Offline tests for the log_gap tool — the deliberate half of the gaps pipeline.

What's pinned: the trust lane is structural (complaint/stringent/self_reported
in the SQL itself, matching migration 007's provenance CHECK), fingerprints are
'self::'-prefixed (disjoint from every miner fingerprint), refiles bump instead
of fragment, empty summaries refuse without touching the DB, and DB failure is
an honest NOT-LOGGED string — never a raise (ToolNode contract), never a fake
success (her own bar: "I can't fake a success on this one").
"""

from contextlib import contextmanager

from aerys_v2.tools.log_gap import LOG_GAP_SQL, _fingerprint, build_log_gap_tool


class FakeResult:
    def __init__(self, row):
        self._row = row

    def fetchone(self):
        return self._row


class FakeConn:
    def __init__(self, row=(41, 1)):
        self.calls = []
        self.row = row

    def execute(self, sql, params=None):
        self.calls.append((sql, params))
        return FakeResult(self.row)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def make_tool(conn):
    return build_log_gap_tool(lambda: conn), conn


# ---- the write ---------------------------------------------------------------


def test_logs_a_gap_on_the_self_reported_complaint_lane():
    tool, conn = make_tool(FakeConn(row=(41, 1)))
    out = tool.invoke({"summary": "G2 lens replies get cut off mid-render",
                       "details": "long replies truncate on the glasses display"})
    assert "Logged as gap #41" in out
    (sql, params), = conn.calls
    # The lane is in the SQL itself — matching the 007 provenance CHECK.
    assert "'self_reported'" in sql
    assert "'complaint'" in sql
    assert "'stringent'" in sql
    assert "'owner'" in sql
    assert params["summary"] == "G2 lens replies get cut off mid-render"
    assert params["diagnosis"].startswith("long replies truncate")


def test_fingerprint_is_self_prefixed_and_disjoint_from_miner_space():
    fp = _fingerprint("G2 lens replies get cut off mid-render")
    assert fp.startswith("self::")
    assert fp == "self::g2-lens-replies-get-cut-off-mid-render"
    # miner fingerprints: 'degraded::…', 'tool_error::…', phrase heads — no overlap
    assert not fp.startswith(("degraded::", "tool_error::"))


def test_refile_bumps_instead_of_fragmenting():
    tool, conn = make_tool(FakeConn(row=(41, 3)))
    out = tool.invoke({"summary": "G2 lens replies get cut off mid-render"})
    assert "already on the board" in out and "3x" in out
    (sql, _), = conn.calls
    assert "ON CONFLICT (fingerprint) DO UPDATE" in sql
    assert "how_often + 1" in sql


def test_summary_is_collapsed_and_capped():
    tool, conn = make_tool(FakeConn())
    tool.invoke({"summary": "  a\n\ngap   with\tmessy   whitespace  " + "x" * 500})
    (_, params), = conn.calls
    assert "\n" not in params["summary"]
    assert "  " not in params["summary"]
    assert len(params["summary"]) <= 200


# ---- refusals + failure posture ------------------------------------------------


def test_empty_summary_refuses_without_touching_the_db():
    conn = FakeConn()
    tool, _ = make_tool(conn)
    out = tool.invoke({"summary": "   "})
    assert out.startswith("NOT LOGGED")
    assert conn.calls == []


def test_db_failure_is_an_honest_string_never_a_raise_never_a_fake_success():
    @contextmanager
    def dead_factory():
        raise RuntimeError("NAS is down")
        yield  # pragma: no cover

    tool = build_log_gap_tool(dead_factory)
    out = tool.invoke({"summary": "a real gap"})
    assert out.startswith("NOT LOGGED")
    assert "write failed" in out or "isn't reachable" in out
    assert "Logged" not in out.replace("NOT LOGGED", "")


# ---- description discipline (the specificity lesson) ---------------------------


def test_description_carries_concrete_triggers_and_the_no_fake_rule():
    tool, _ = make_tool(FakeConn())
    d = " ".join(tool.description.lower().split())  # docstring line-wraps collapse
    assert "log a gap" in d
    assert "call this tool immediately" in d
    assert "never claim" in d or "never pretend" in d


def test_sql_matches_migration_007_lane_exactly():
    # Belt-and-braces: if someone edits the SQL onto a softer lane, this fails
    # before Postgres ever gets the chance to reject it.
    assert "'complaint'" in LOG_GAP_SQL and "'stringent'" in LOG_GAP_SQL
    assert "'self_reported'" in LOG_GAP_SQL
    assert "'error'" not in LOG_GAP_SQL and "'standard'" not in LOG_GAP_SQL
