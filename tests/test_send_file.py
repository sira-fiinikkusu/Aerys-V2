"""She hands him a file her home job made (#183, 11/10).

11/10 her home job wrote the how-to he asked for, and it never reached him: she couldn't attach a
file in his DM. send_file_to_chris reads the file from her home server (GET /files/{path}, its own
token) and sends it to him alone: a Discord attachment in his DM, or a Telegram document when he
asked on Telegram. These pin: it says SENT only once Discord or Telegram took the file, every other
outcome is a plain NOT SENT, and only he is ever offered the tool.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

pytest.importorskip("langchain.agents")
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import ConfigDict, SecretStr

from aerys_v2.action_loop import AgentToolModels, build_agent_action_graph
from aerys_v2.home_jobs import HOME_JOBS_OVERLAY
from aerys_v2.tools.send_file import (
    MAX_BYTES,
    TOOL_NAME,
    build_send_file_tool,
    deliver_for,
    send_file_middleware,
)

OWNER = "6e6bcbed-0000-4000-8000-000000000001"
HOME_URL = "http://home.test:2024"
DISCORD = {"user_id": OWNER, "display_name": "Chris", "platform": "discord", "channel_kind": "dm"}
TELEGRAM = {"user_id": OWNER, "display_name": "Chris", "platform": "telegram", "channel_kind": "dm"}
GUEST = {"user_id": "someone-else", "display_name": "Sam", "platform": "discord", "channel_kind": "guild"}
HOWTO = b"# Fixed IP for the voice satellite\n\nmanual_ip: ...\n"


class Resp:
    def __init__(self, status_code=200, content=b"", headers=None):
        self.status_code, self.content, self.headers = status_code, content, headers or {}


class Rig:
    def __init__(self, resp=None, deliver_fails=None):
        self.resp = resp or Resp(200, HOWTO, {"X-File-Name": "esphome-voice-satellite-fixed-ip.md"})
        self.got, self.sent = [], []
        self.deliver_fails = deliver_fails
        self.tool = build_send_file_tool(home_url=HOME_URL, home_token="home-api-token", owner_person_id=OWNER,
                                         deliver=self.deliver, get=self.get)

    def get(self, url, headers):
        self.got.append((url, headers))
        if isinstance(self.resp, Exception):
            raise self.resp
        return self.resp

    def deliver(self, surface, name, data, note):
        if self.deliver_fails:
            raise self.deliver_fails
        self.sent.append((surface, name, data, note))

    def call(self, identity, path="esphome-voice-satellite-fixed-ip.md", note=""):
        return self.tool.invoke({"path": path, "note": note},
                                config={"configurable": {"identity": identity, "thread_id": f"person:{OWNER}"}})


def test_a_file_from_her_home_workspace_lands_in_his_discord_dm():
    rig = Rig()
    said = rig.call(DISCORD, path="/home/hands/work/esphome-voice-satellite-fixed-ip.md", note="Here's the how-to.")
    assert said.startswith("SENT") and "esphome-voice-satellite-fixed-ip.md" in said
    ((url, headers),) = rig.got
    assert url == f"{HOME_URL}/files/esphome-voice-satellite-fixed-ip.md"      # workspace paths go relative
    assert headers == {"Authorization": "Bearer home-api-token"}
    assert rig.sent == [("discord", "esphome-voice-satellite-fixed-ip.md", HOWTO, "Here's the how-to.")]


def test_asked_on_telegram_it_goes_to_his_telegram():
    rig = Rig()
    assert rig.call(TELEGRAM).startswith("SENT")
    assert rig.sent[0][0] == "telegram"


@pytest.mark.parametrize("status, words", [(404, "no file"), (413, "over 8 MB"), (401, "refused"), (500, "HTTP 500")])
def test_a_file_her_home_machine_wont_give_is_never_called_sent(status, words):
    rig = Rig(Resp(status))
    said = rig.call(DISCORD)
    assert said.startswith("NOT SENT") and words in said and rig.sent == []


def test_an_unreachable_home_machine_is_not_sent_too():
    rig = Rig(ConnectionError("vm away"))
    said = rig.call(DISCORD)
    assert said.startswith("NOT SENT") and "reach my home machine" in said and rig.sent == []


def test_a_file_over_the_cap_is_refused_even_if_the_server_sent_it():
    rig = Rig(Resp(200, b"x" * (MAX_BYTES + 1), {"X-File-Name": "big.bin"}))
    assert "over 8 MB" in rig.call(DISCORD) and rig.sent == []


def test_when_discord_wont_take_it_she_says_so():
    rig = Rig(deliver_fails=RuntimeError("discord 403"))
    said = rig.call(DISCORD)
    assert said.startswith("NOT SENT") and "Discord" in said


def test_a_file_name_from_her_home_machine_is_only_ever_a_name():
    rig = Rig(Resp(200, HOWTO, {"X-File-Name": "../../etc/passwd"}))
    rig.call(DISCORD)
    assert rig.sent[0][1] == "passwd"


def test_no_path_no_call():
    rig = Rig()
    assert rig.call(DISCORD, path="  ").startswith("NOT SENT") and rig.got == []


def test_anyone_but_him_is_refused_at_the_call():
    rig = Rig()
    said = rig.call(GUEST)
    assert said.startswith("NOT SENT") and "only to Chris" in said and rig.got == [] and rig.sent == []


# --- only he is offered it (through a real create_agent loop) -----------------------------------


class Rec(BaseChatModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    script: Any
    log: Any
    bound: Any = ()

    @property
    def _llm_type(self) -> str:
        return "rec"

    def bind_tools(self, tools, *, tool_choice=None, **kwargs):
        return self.model_copy(update={"bound": tuple(convert_to_openai_tool(t)["function"]["name"] for t in tools)})

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.log.append({"tools": self.bound})
        return ChatResult(generations=[ChatGeneration(message=self.script.pop(0))])


def _turn(identity, script):
    log = []
    model = Rec(script=list(script), log=log)
    rig = Rig()
    graph = build_agent_action_graph(AgentToolModels(model, (model, True)), "SOUL", [],
                                     extra_middleware=[send_file_middleware(rig.tool, OWNER)])
    out = graph.invoke({"messages": [HumanMessage(content="send me the how-to")]},
                       {"configurable": {"thread_id": f"person:{OWNER}", "identity": identity}, "recursion_limit": 8})
    return out, log, rig


def test_he_is_offered_the_tool_and_it_sends():
    call = AIMessage(content="", tool_calls=[{"id": "c1", "name": TOOL_NAME,
                                              "args": {"path": "esphome-voice-satellite-fixed-ip.md"}}])
    out, log, rig = _turn(DISCORD, [call, AIMessage(content="It's in your DM.")])
    assert TOOL_NAME in log[0]["tools"]
    (sent,) = [m.content for m in out["messages"] if isinstance(m, ToolMessage)]
    assert sent.startswith("SENT") and rig.sent and rig.sent[0][0] == "discord"


def test_anyone_else_is_never_offered_it():
    out, log, rig = _turn(GUEST, [AIMessage(content="I can't do that.")])
    assert TOOL_NAME not in log[0]["tools"]


def test_her_overlay_teaches_it():
    assert TOOL_NAME in HOME_JOBS_OVERLAY


# --- the delivery legs: Discord attachment, Telegram document -----------------------------------


class Post:
    def __init__(self):
        self.calls = []

    def __call__(self, url, **kw):
        self.calls.append((url, kw))
        if url.endswith("/users/@me/channels"):
            return Resp(200, b'{"id": "dm-channel-1"}', {})
        return Resp(200, b"{}", {})


def _settings(**over):
    base = dict(discord_bot_token=SecretStr("discord-bot-token"), telegram_bot_token=SecretStr("telegram-bot-token"),
                owner_person_id=OWNER, memories_database_url="postgresql://x/aerys")
    return SimpleNamespace(**{**base, **over})


def test_discord_delivery_opens_his_dm_and_attaches_the_file():
    post = Post()
    deliver = deliver_for(_settings(), post=post, owner_id=lambda platform: f"{platform}-user-1")
    deliver("discord", "howto.md", HOWTO, "Here's the how-to.")
    (open_dm, _), (send, kw) = post.calls
    assert open_dm.endswith("/users/@me/channels")
    assert send.endswith("/channels/dm-channel-1/messages")
    assert kw["headers"]["Authorization"] == "Bot discord-bot-token"
    (field, (name, data, _ctype)), = kw["files"].items()
    assert field == "files[0]" and name == "howto.md" and data == HOWTO
    assert "Here's the how-to." in kw["data"]["payload_json"]


def test_telegram_delivery_sends_a_document_to_his_chat():
    post = Post()
    deliver = deliver_for(_settings(), post=post, owner_id=lambda platform: f"{platform}-user-1")
    deliver("telegram", "howto.md", HOWTO, "Here's the how-to.")
    ((url, kw),) = post.calls
    assert url == "https://api.telegram.org/bottelegram-bot-token/sendDocument"
    assert kw["data"] == {"chat_id": "telegram-user-1", "caption": "Here's the how-to."}
    assert kw["files"]["document"][0] == "howto.md"


def test_a_surface_she_cant_reach_raises_so_the_tool_says_not_sent():
    deliver = deliver_for(_settings(telegram_bot_token=None), post=Post(), owner_id=lambda platform: None)
    with pytest.raises(RuntimeError):
        deliver("telegram", "howto.md", HOWTO, "")


def test_no_bot_at_all_means_no_delivery():
    assert deliver_for(_settings(discord_bot_token=None, telegram_bot_token=None)) is None


# --- the record of sends, and no re-send he didn't ask for (#187, 11/10) ----------------------
# 23:13: asked "what file did you send me tonight?", her hands looked in the home-job records,
# didn't find the 22:14 send (a direct send, never a job), decided it hadn't happened, and sent the
# file again. These pin: every send is recorded, the record answers "what did you send", and the
# same file again within hours needs him to ask for it again.

from datetime import datetime, timedelta, timezone

from aerys_v2.factory import SPECIALIST_CHARTER
from aerys_v2.tools.send_file import (
    RESEND_HOURS,
    SENT_TOOL_NAME,
    FileSend,
    FileSends,
    build_files_sent_tool,
)

T0 = datetime(2026, 11, 11, 3, 14, tzinfo=timezone.utc)    # 10:14 PM Eastern, 11/10


class Clock:
    def __init__(self, at=T0):
        self.at = at

    def __call__(self):
        return self.at


class RecordedRig(Rig):
    def __init__(self, ledger=None, clock=None, **kw):
        super().__init__(**kw)
        self.ledger, self.clock = ledger if ledger is not None else FileSends(), clock or Clock()
        self.tool = build_send_file_tool(home_url=HOME_URL, home_token="home-api-token", owner_person_id=OWNER,
                                         deliver=self.deliver, get=self.get, ledger=self.ledger, now=self.clock)

    def call(self, identity, path="esphome-voice-satellite-fixed-ip.md", note="", again=False):
        return self.tool.invoke({"path": path, "note": note, "again": again},
                                config={"configurable": {"identity": identity, "thread_id": f"person:{OWNER}"}})


def test_a_send_is_recorded_once_he_has_it():
    rig = RecordedRig()
    assert rig.call(DISCORD).startswith("SENT")
    ((person, send),) = rig.ledger.rows
    assert person == OWNER and send.path == "esphome-voice-satellite-fixed-ip.md"
    assert (send.name, send.bytes, send.surface) == ("esphome-voice-satellite-fixed-ip.md", len(HOWTO), "discord")


def test_a_send_that_didnt_happen_is_never_recorded():
    rig = RecordedRig(deliver_fails=RuntimeError("Discord answered HTTP 500"))
    assert rig.call(DISCORD).startswith("NOT SENT") and rig.ledger.rows == []


def test_the_same_file_again_is_refused_and_says_when_it_went():
    clock = Clock()
    rig = RecordedRig(clock=clock)
    assert rig.call(DISCORD).startswith("SENT")
    clock.at = T0 + timedelta(minutes=59)                    # 11:13 PM: the voice question
    said = rig.call(DISCORD)
    assert said.startswith("ALREADY SENT") and "10:14 PM today" in said and "his Discord DM" in said
    assert len(rig.sent) == 1 and len(rig.got) == 1          # nothing fetched, nothing sent the second time


def test_he_can_ask_for_it_again():
    rig = RecordedRig()
    rig.call(DISCORD)
    assert rig.call(DISCORD, again=True).startswith("SENT")
    assert len(rig.sent) == 2 and len(rig.ledger.rows) == 2


def test_after_the_window_it_sends_without_asking_again():
    clock = Clock()
    rig = RecordedRig(clock=clock)
    rig.call(DISCORD)
    clock.at = T0 + timedelta(hours=RESEND_HOURS, minutes=1)
    assert rig.call(DISCORD).startswith("SENT") and len(rig.sent) == 2


def test_a_different_file_is_its_own_send():
    rig = RecordedRig()
    rig.call(DISCORD)
    assert rig.call(DISCORD, path="/home/hands/work/notes/other.md").startswith("SENT")


def test_a_record_it_cant_read_never_blocks_a_send_he_asked_for():
    class Broken(FileSends):
        def last(self, *a):
            raise RuntimeError("database down")
    rig = RecordedRig(ledger=Broken())
    assert rig.call(DISCORD).startswith("SENT")


def test_the_record_answers_what_she_sent():
    clock = Clock()
    rig = RecordedRig(clock=clock)
    rig.call(DISCORD)
    clock.at = T0 + timedelta(minutes=59)
    sent = build_files_sent_tool(ledger=rig.ledger, owner_person_id=OWNER, now=clock)
    said = sent.invoke({}, config={"configurable": {"identity": DISCORD}})
    assert "10:14 PM today" in said and "esphome-voice-satellite-fixed-ip.md" in said and "Discord DM" in said


def test_an_empty_record_says_so_and_guesses_nothing():
    sent = build_files_sent_tool(ledger=FileSends(), owner_person_id=OWNER, now=Clock())
    said = sent.invoke({"hours": 24}, config={"configurable": {"identity": DISCORD}})
    # Codex review: a missing line is never stated as "nothing was sent" (the 11/10 premise).
    assert said.startswith("No recorded sends") and "isn't proof" in said and "don't send it again" in said


def test_the_record_is_only_his():
    sent = build_files_sent_tool(ledger=FileSends(), owner_person_id=OWNER, now=Clock())
    assert "only for Chris" in sent.invoke({}, config={"configurable": {"identity": GUEST}})


def test_both_tools_are_offered_to_him_and_neither_to_anyone_else():
    ledger = FileSends()
    sent = build_files_sent_tool(ledger=ledger, owner_person_id=OWNER, now=Clock())

    def offered(identity):
        log = []
        model = Rec(script=[AIMessage(content="ok")], log=log)
        graph = build_agent_action_graph(AgentToolModels(model, (model, True)), "SOUL", [],
                                         extra_middleware=[send_file_middleware(Rig().tool, OWNER, sent_tool=sent)])
        graph.invoke({"messages": [HumanMessage(content="what did you send me?")]},
                     {"configurable": {"thread_id": "t", "identity": identity}, "recursion_limit": 6})
        return set(log[0]["tools"])

    assert {TOOL_NAME, SENT_TOOL_NAME} <= offered(DISCORD)
    assert not ({TOOL_NAME, SENT_TOOL_NAME} & offered(GUEST))


def test_her_rules_say_a_question_about_what_she_did_is_not_a_redo():
    assert "never by doing it again" in SPECIALIST_CHARTER
    assert SENT_TOOL_NAME in HOME_JOBS_OVERLAY and "don't send the file again" in HOME_JOBS_OVERLAY


# --- Codex review of 9826512: one file, one send, whatever the path spelling or the timing -------

import threading

from aerys_v2.tools.send_file import PostgresFileSends


@pytest.mark.parametrize("spelling", ["./esphome-voice-satellite-fixed-ip.md",
                                      "notes/../esphome-voice-satellite-fixed-ip.md",
                                      "/home/hands/work/esphome-voice-satellite-fixed-ip.md"])
def test_another_spelling_of_the_same_file_is_still_already_sent(spelling):
    rig = RecordedRig()
    rig.call(DISCORD)
    assert rig.call(DISCORD, path=spelling).startswith("ALREADY SENT") and len(rig.sent) == 1


def test_a_path_out_of_her_workspace_is_refused_before_anything_is_fetched():
    rig = RecordedRig()
    said = rig.call(DISCORD, path="../.ssh/id_ed25519")
    assert said.startswith("NOT SENT") and "leaves my home workspace" in said and rig.got == []


def test_two_overlapping_sends_of_one_file_deliver_once():
    rig = RecordedRig()
    entered, release = threading.Event(), threading.Event()
    first = rig.deliver

    def slow(surface, name, data, note):
        entered.set()
        release.wait(5)
        first(surface, name, data, note)
    rig.deliver = slow
    rig.tool = build_send_file_tool(home_url=HOME_URL, home_token="home-api-token", owner_person_id=OWNER,
                                    deliver=lambda *a: rig.deliver(*a), get=rig.get, ledger=rig.ledger, now=rig.clock)
    results = []
    a = threading.Thread(target=lambda: results.append(rig.call(DISCORD)))
    a.start()
    assert entered.wait(5)
    b = threading.Thread(target=lambda: results.append(rig.call(DISCORD)))
    b.start()
    release.set()
    a.join(5); b.join(5)
    assert sorted(r.split(":")[0] for r in results) == ["ALREADY SENT", "SENT"] and len(rig.sent) == 1


class FlakyPg(PostgresFileSends):
    """The database answers reads (with nothing) but every write fails."""

    def __init__(self):
        super().__init__("postgresql://unused")

    def _connect(self, options=""):
        raise RuntimeError("database write failed")

    def _rows(self, sql, args):
        return []


def test_a_send_whose_record_write_failed_is_still_known_to_this_process():
    ledger = FlakyPg()
    rig = RecordedRig(ledger=ledger)
    assert rig.call(DISCORD).startswith("SENT")              # the claim fell back to the process lock
    sent = build_files_sent_tool(ledger=ledger, owner_person_id=OWNER, now=rig.clock)
    said = sent.invoke({}, config={"configurable": {"identity": DISCORD}})
    assert "esphome-voice-satellite-fixed-ip.md" in said and not said.startswith("No recorded")
    assert rig.call(DISCORD).startswith("ALREADY SENT")
