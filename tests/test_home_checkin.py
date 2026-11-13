"""Home check-ins stay owed; only an attended owner can continue one."""
import json
from pathlib import Path

import httpx
import pytest
from langchain_core.messages import AIMessage

from test_home_jobs import Box, home, call, tool_texts, TOKEN, OWNER
from test_home_notices import Home, note, NIGHT
from aerys_v2.home_notices import Job
from aerys_v2.wakeups import fenced

IDENTITY = {"user_id": OWNER, "platform": "discord"}
MESSAGE = "He said keep going. Continue the job from your workspace and notes."


def test_checkin_is_stored_clipped_and_owed():
    h = Home(now=NIGHT, job=Job(task="Build a sample", error="not the report"))
    payload = {**note("check_in"), "summary": "progress " * 400, "reason": "limit " * 100,
               "seconds_used": 88200, "steps_used": 123.5}
    assert h.notices.receive(payload) == "new"
    assert h.notices.receive(payload) == "duplicate"
    meta = h.notices.store.rows[1].metadata
    assert len(meta["summary"]) <= 2000 and meta["summary"].startswith("progress")
    assert len(meta["reason"]) <= 300 and meta["reason"].startswith("limit")
    assert meta["seconds_used"] == 88200 and meta["steps_used"] == 123.5
    assert h.notices.tick()["handed"] == 1
    event = h.desk.store.rows[1]
    assert event.owed and event.dedupe_key == "home-job:run-1:check_in"
    assert event.detail == meta["summary"]
    assert event.headline == "That job at home has been going 24 hours: Build a sample. It's paused until you say: keep going, or stop."
    assert not h.composed


def test_checkin_member_delivery_and_desk_retry():
    h = Home()
    sent = []
    h.notices.whose = lambda n: "member-placeholder"
    h.notices.owner = OWNER
    h.notices.tell_member = lambda *args: sent.append(args)
    h.notices.receive({**note("check_in"), "summary": "Done: first part. Left: second part.", "seconds_used": 90000})
    assert h.notices.tick()["handed"] == 1
    assert sent[0][0] == "member-placeholder" and "25 hours" in sent[0][1]
    assert "Left: second part." in sent[0][1] and not h.desk.store.rows
    h = Home()
    h.notices.receive({**note("check_in"), "seconds_used": 86400})
    original = h.desk.reach_out
    h.desk.reach_out = lambda e: (_ for _ in ()).throw(ConnectionError())
    assert h.notices.tick()["waiting"] == 1 and h.notices.store.open()
    h.desk.reach_out = original
    assert h.notices.tick()["handed"] == 1


def setup_job(home, monkeypatch):
    b = Box(members={"member-placeholder": "sample-project"})
    b.turn([call("start_async_task", {"description": "Sample job", "subagent_type": "home"}), AIMessage(content="Started")], IDENTITY)
    tid = next(iter(b.ledger.rows))
    home.calls.clear()
    status = {"code": 200}
    def extend(request):
        assert request.headers["Authorization"] == f"Bearer {TOKEN}"
        home.calls.append((request.method, request.url.path, json.loads(request.content)))
        return httpx.Response(status["code"], json={"job_status": "ready"})
    client = httpx.Client(transport=httpx.MockTransport(extend))
    monkeypatch.setattr(httpx, "post", client.post)
    # GET /threads/{id}: what the tool reads after a 409 (not recorded in home.calls).
    reader = httpx.Client(transport=httpx.MockTransport(
        lambda request: httpx.Response(200, json={"values": {"job_status": status.get("job_status", "check_in")}})))
    monkeypatch.setattr(httpx, "get", reader.get)
    return b, tid, status


@pytest.mark.parametrize("identity,thread", [
    (IDENTITY, "person:owner"),
    ({"user_id": OWNER}, "panel:owner"),
    ({"user_id": OWNER, "voice": True, "speaker": {"id": "owner-placeholder"}}, "voice:room"),
])
def test_continue_extends_then_uses_official_update(home, monkeypatch, identity, thread):
    b, tid, _ = setup_job(home, monkeypatch)
    old_run = b.ledger.rows[tid][1]["run_id"]
    out, log = b.turn([call("continue_home_job", {"task_id": tid}), AIMessage(content="Continuing")], identity, thread)
    assert "continue_home_job" in log[0]["tools"]
    assert [c[1] for c in home.calls] == [f"/threads/{tid}/extend", f"/threads/{tid}/runs"]
    assert home.calls[0][2] == {}
    body = home.calls[1][2]
    assert body["input"] == {"messages": [{"role": "user", "content": MESSAGE}]}
    assert body["multitask_strategy"] == "interrupt" and body["assistant_id"] == "hands"
    assert b.ledger.rows[tid][1]["run_id"] != old_run
    assert not any(t.startswith("NOT DONE") for t in tool_texts(out))


@pytest.mark.parametrize("identity,thread", [
    ({**IDENTITY, "unattended": True}, "wake:sample"),
    ({"user_id": "member-placeholder", "platform": "discord"}, "person:member"),
    ({"user_id": "guest-placeholder"}, "person:guest"),
    ({**IDENTITY, "voice": True, "speaker": {"id": "unknown", "unsure": True}}, "voice:room"),
    (IDENTITY, "kael:sample"),
])
def test_continue_refuses_without_http(home, monkeypatch, identity, thread):
    b, tid, _ = setup_job(home, monkeypatch)
    out, _ = b.turn([call("continue_home_job", {"task_id": tid}), AIMessage(content="Refused")], identity, thread)
    assert any("NOT DONE:" in t for t in tool_texts(out))
    assert not home.calls


@pytest.mark.parametrize("code,message", [
    (409, "NOT DONE: that job isn't waiting at a check-in."),
    (404, "NOT DONE: no such job."),
    (401, "NOT DONE: couldn't reach the home machine; nothing changed."),
    (500, "NOT DONE: couldn't reach the home machine; nothing changed."),
])
def test_continue_http_failures(home, monkeypatch, code, message):
    b, tid, status = setup_job(home, monkeypatch)
    status["code"] = code
    out, _ = b.turn([call("continue_home_job", {"task_id": tid}), AIMessage(content="Refused")], IDENTITY)
    assert message in tool_texts(out)
    assert len(home.calls) == 1


def test_checkin_fence_and_migration():
    assert fenced("continue_home_job", {"task_id": "placeholder"})
    sql = (Path(__file__).parents[1] / "db/migrations/031_home_checkin.sql").read_text()
    assert "'check_in'" in sql and "DROP CONSTRAINT IF EXISTS v2_home_notices_status_check" in sql


@pytest.mark.parametrize("stage", ["extend", "run"])
def test_continue_connection_failures_return_plain_refusal(home, monkeypatch, stage):
    b, tid, _ = setup_job(home, monkeypatch)
    old = dict(b.ledger.rows[tid][1])
    if stage == "extend":
        def unavailable(*args, **kwargs):
            raise httpx.ConnectError("offline")
        monkeypatch.setattr(httpx, "post", unavailable)
    else:
        original = home.handler
        def failed_run(request):
            if request.url.path.endswith("/runs"):
                return httpx.Response(503, json={"detail": "offline"})
            return original(request)
        home.handler = failed_run
    out, _ = b.turn([call("continue_home_job", {"task_id": tid}), AIMessage(content="Refused")], IDENTITY)
    # Before the extend nothing changed; after a successful extend the job HAS more time (Kael, 11/12:
    # saying "nothing changed" there was false), and asking again starts the run.
    expected = ("NOT DONE: couldn't reach the home machine; nothing changed." if stage == "extend" else
                "NOT DONE: the job has more time now, but its next run didn't start. Tell him; an update to the job (update_async_task) starts it.")
    assert expected in tool_texts(out)
    assert b.ledger.rows[tid][1] == old


def test_continue_unknown_task_does_not_extend(home, monkeypatch):
    b, _, _ = setup_job(home, monkeypatch)
    out, _ = b.turn([call("continue_home_job", {"task_id": "unknown-placeholder"}), AIMessage(content="Refused")], IDENTITY)
    assert "NOT DONE: no such job." in tool_texts(out)
    assert not home.calls


@pytest.mark.parametrize("value", ["86400", None, True, float("inf"), float("nan")])
def test_checkin_invalid_counters_cannot_break_delivery(value):
    h = Home()
    h.notices.receive({**note("check_in"), "seconds_used": value, "steps_used": value})
    assert "seconds_used" not in h.notices.store.rows[1].metadata
    assert "steps_used" not in h.notices.store.rows[1].metadata
    assert h.notices.tick()["handed"] == 1


def test_other_status_does_not_interpret_checkin_counters():
    h = Home()
    h.notices.receive(note("error", metadata={"seconds_used": "not a counter"}))
    assert h.notices.tick()["handed"] == 1


def test_a_second_continue_never_queues_another_run(home, monkeypatch):
    b, tid, status = setup_job(home, monkeypatch)
    b.turn([call("continue_home_job", {"task_id": tid}), AIMessage(content="Continued")], IDENTITY)
    status["code"] = 409                                  # running now: no longer at a check-in
    out, _ = b.turn([call("continue_home_job", {"task_id": tid}), AIMessage(content="Refused")], IDENTITY)
    assert "NOT DONE: that job isn't waiting at a check-in." in tool_texts(out)
    assert [c[1].rsplit("/", 1)[-1] for c in home.calls].count("runs") == 1


def test_continue_refuses_after_outside_text(home, monkeypatch):
    b, tid, _ = setup_job(home, monkeypatch)
    out, _ = b.turn([call("read_web_page", {"url": "https://example.test/page"}),
                     call("continue_home_job", {"task_id": tid}), AIMessage(content="Refused")], IDENTITY)
    assert any("fresh 'keep going'" in text for text in tool_texts(out))
    assert not any(c[1].endswith("/extend") for c in home.calls)
