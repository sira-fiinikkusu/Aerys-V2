"""Panel face pusher — her desk avatar mirrors the brain's phases.

Offline proof of: mood extraction from emotion tags, phase->state mapping, the
consecutive-dedup rule, the speaking->idle auto-flip, working deferred behind a
playing ack, the fail-open contract (a dead panel costs nothing), and the ask()
seams firing at the right moments on the text chat, text action, and voice paths.
"""

import time

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

import aerys_v2.panel as panel
from aerys_v2.factory import build_graph
from aerys_v2.panel import FacePusher, build_face_pusher, mood_of, speaking_estimate_s
from aerys_v2.router import RouteDecision
from aerys_v2.service import ask

CHRIS = {"user_id": "person-1", "display_name": "Chris"}


class RecordingClient:
    def __init__(self):
        self.states: list[str] = []

    def post(self, url, json=None, **kwargs):
        self.states.append(json["state"])


class ExplodingClient:
    def post(self, url, json=None, **kwargs):
        raise ConnectionError("panel is dark")


def pusher(client=None) -> tuple[FacePusher, RecordingClient]:
    client = client or RecordingClient()
    return FacePusher("http://panel:8300/state", client=client, async_send=False), client


def fake_model(*replies) -> GenericFakeChatModel:
    msgs = [r if isinstance(r, AIMessage) else AIMessage(content=r) for r in replies]
    return GenericFakeChatModel(messages=iter(msgs))


# ── mood extraction ─────────────────────────────────────────────────────────


def test_mood_of_reads_the_first_recognized_tag():
    assert mood_of("[warmly] Morning, Chris!") == "happy"
    assert mood_of("[softly] [teasingly] sure") == "neutral"  # first tag wins
    assert mood_of("[teasingly] oh really?") == "playful"
    assert mood_of("[sighs] fine.") == "grumpy"
    assert mood_of("[gasps] no way!") == "surprised"
    assert mood_of("[lovingly] you did great") == "affection"


def test_mood_of_defaults_neutral_and_hearts_read_affection():
    assert mood_of("The lights are off.") == "neutral"
    assert mood_of("") == "neutral"
    assert mood_of("goodnight ❤") == "affection"
    assert mood_of("[unrecognized tag] hello") == "neutral"


def test_speaking_estimate_bounds():
    assert speaking_estimate_s("hi") == 2.5  # floor: one-word acks hold a beat
    assert speaking_estimate_s("x" * 10_000) == 20.0  # ceiling: rambles don't pin
    # tags are spoken silently — they must not inflate the estimate
    assert speaking_estimate_s("[warmly] hi") == speaking_estimate_s("hi")


# ── phase -> state mapping + dedup ──────────────────────────────────────────


def test_phases_map_to_panel_states():
    p, client = pusher()
    p("working")
    p("idle", "[warmly] all set!")
    p("idle", "plain text")
    assert client.states == ["working", "happy_idle", "neutral_idle"]


def test_moods_without_speaking_variants_fall_back():
    p, client = pusher()
    p("speaking", "[teasingly] make me")  # playful speaks as happy
    p("speaking", "[sighs] fine, done")   # grumpy speaks as deadpan neutral
    assert client.states[:2] == ["happy_speaking", "neutral_speaking"]


def test_consecutive_duplicate_states_are_not_resent():
    p, client = pusher()
    p("idle", "one")
    p("idle", "two")
    p("idle", "three")
    assert client.states == ["neutral_idle"]


# ── speaking auto-flip + deferred working ───────────────────────────────────


def test_speaking_flips_back_to_mood_idle(monkeypatch):
    monkeypatch.setattr(panel, "speaking_estimate_s", lambda _t: 0.05)
    p, client = pusher()
    p("speaking", "[warmly] here's your forecast")
    assert client.states == ["happy_speaking"]
    time.sleep(0.2)
    assert client.states == ["happy_speaking", "happy_idle"]


def test_working_defers_until_the_ack_finishes(monkeypatch):
    monkeypatch.setattr(panel, "speaking_estimate_s", lambda _t: 0.1)
    p, client = pusher()
    p("speaking", "On it — one sec.")
    p("working")
    # the ack is still 'playing': the working face must NOT preempt it
    assert client.states == ["neutral_speaking"]
    time.sleep(0.3)
    assert client.states == ["neutral_speaking", "working"]


def test_newer_push_cancels_the_pending_flip(monkeypatch):
    monkeypatch.setattr(panel, "speaking_estimate_s", lambda _t: 0.1)
    p, client = pusher()
    p("speaking", "[warmly] sure thing")
    p("idle", "[sighs] done I guess")  # settles the face NOW
    time.sleep(0.3)  # the cancelled flip must never land happy_idle after this
    assert client.states == ["happy_speaking", "grumpy_idle"]


# ── fail-open contract ──────────────────────────────────────────────────────


def test_dead_panel_never_raises():
    p, _ = pusher(client=ExplodingClient())
    p("working")
    p("speaking", "hello")
    p("idle", "done")  # no exception = contract held


def test_build_face_pusher_arming():
    assert build_face_pusher(None) is None
    assert build_face_pusher("") is None
    assert callable(build_face_pusher("http://panel:8300/state"))


# ── ask() seams ─────────────────────────────────────────────────────────────


class FaceLog:
    def __init__(self):
        self.calls: list[tuple[str, str]] = []

    def __call__(self, phase: str, text: str = "") -> None:
        self.calls.append((phase, text))


class StubActionGraph:
    def __init__(self, final: str = "light is off now"):
        self.final = final

    def invoke(self, inp: dict, config: dict) -> dict:
        return {"messages": [AIMessage(content=self.final)]}


def test_text_chat_turn_pushes_mood_idle():
    graph = build_graph(fake_model("[warmly] hey you!"), soul="s")
    face = FaceLog()
    reply = ask(graph, "morning", identity=CHRIS, thread_id="t-chat", face_push=face)
    assert reply == "[warmly] hey you!"
    assert face.calls == [("idle", "[warmly] hey you!")]


def test_text_action_turn_pushes_working_then_idle():
    graph = build_graph(fake_model(), soul="s")
    face = FaceLog()
    reply = ask(
        graph, "turn off the light", identity=CHRIS, thread_id="t-act",
        router=lambda _t: RouteDecision(route="action", ack=""),
        action_graph=StubActionGraph(), face_push=face,
    )
    assert reply == "light is off now"
    assert face.calls == [("working", ""), ("idle", "light is off now")]


def test_voice_chat_turn_pushes_speaking():
    graph = build_graph(fake_model("[warmly] it's sunny all day"), soul="s")
    face = FaceLog()
    reply = ask(
        graph, "what's the weather", identity={**CHRIS, "voice": True},
        thread_id="person:person-1",
        router=lambda _t: RouteDecision(route="chat", ack=""),
        action_graph=StubActionGraph(), face_push=face,
    )
    assert reply == "[warmly] it's sunny all day"
    assert ("speaking", "[warmly] it's sunny all day") in face.calls


def test_voice_action_turn_speaks_ack_then_works():
    graph = build_graph(fake_model("(speculative, discarded)"), soul="s")
    face = FaceLog()
    ack = ask(
        graph, "kill the lights", identity={**CHRIS, "voice": True},
        thread_id="person:person-1",
        router=lambda _t: RouteDecision(route="action", ack="On it."),
        action_graph=StubActionGraph(), face_push=face,
    )
    assert ack == "On it."
    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline and len(face.calls) < 3:
        time.sleep(0.02)
    phases = [phase for phase, _ in face.calls]
    assert phases[:2] == ["speaking", "working"]
    # background action settled -> her face settles too (spoken or mood-idle)
    assert phases[-1] in ("speaking", "idle")


def test_seam_survives_a_raising_face_push():
    def bomb(_phase: str, _text: str = "") -> None:
        raise RuntimeError("misbehaving fake")

    graph = build_graph(fake_model("still fine"), soul="s")
    reply = ask(graph, "hello", identity=CHRIS, thread_id="t-bomb", face_push=bomb)
    assert reply == "still fine"
