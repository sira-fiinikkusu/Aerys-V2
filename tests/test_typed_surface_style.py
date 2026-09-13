"""The typed surfaces must say they are typed.

Live defect 2026-09-13, #resonance. Chris typed "Okay how about now" in Discord
and she answered "[thoughtfully] Still not landing ... is the glasses garbling
again?" — an ElevenLabs emotion tag and a speech-to-text excuse, on a channel
where he had typed every character himself.

Nothing was misconfigured. Her thread is person-keyed, so voice, glasses and
Discord all share one history, and that history is full of voice-styled replies.
The voice branch styles the CURRENT turn; the text branch said nothing at all, so
the thread's own precedent was the only style signal she had.

So the text branch has to speak: this is typed, their words arrive exactly as
written, no mis-hearing to blame, no stage directions to emit. The fence is on
BOTH minds — the chat node and the action node share the thread and the bleed.
"""

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage

from aerys_v2.factory import TYPED_SURFACE_STYLE, build_action_graph, build_graph
from aerys_v2.service import ask

TYPED_DISCORD = {
    "user_id": "person-1", "display_name": "Chris", "privacy_context": "public",
    "platform": "discord", "channel_kind": "guild", "channel_id": "555",
    "channel_name": "resonance",
}
TYPED_DM = {"user_id": "person-1", "display_name": "Chris", "privacy_context": "private",
            "platform": "telegram", "channel_kind": "dm", "channel_id": "9"}
VOICE = {"user_id": "person-1", "display_name": "Chris", "voice": True}
LENS = {"user_id": "person-1", "display_name": "Chris", "surface": "lens"}


class RecordingModel(GenericFakeChatModel):
    seen: list = []

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        type(self).seen.append(str(messages[0].content))
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)


class RecordingToolModel:
    def __init__(self):
        self.prompts = []

    def invoke(self, messages, **kw):
        self.prompts.append(list(messages))
        return AIMessage(content="done")


def _chat_prompt(identity, thread="person:p1"):
    RecordingModel.seen = []
    model = RecordingModel(messages=iter([AIMessage(content="ok")]))
    ask(build_graph(model, soul="s"), "hey", identity=identity, thread_id=thread)
    return RecordingModel.seen[0]


def _action_prompt(identity):
    model = RecordingToolModel()
    graph = build_action_graph(model, soul="s", tools=[])
    graph.invoke({"messages": [HumanMessage(content="hey")]},
                 {"configurable": {"identity": identity}})
    return model.prompts[0][0].content


# ── the chat mind ────────────────────────────────────────────────────────────

def test_typed_discord_turn_is_told_it_is_typed():
    system = _chat_prompt(TYPED_DISCORD)
    assert TYPED_SURFACE_STYLE.strip() in system
    assert "VOICE conversation" not in system     # and not styled for speech


def test_typed_dm_turn_is_told_too():
    # the bleed is per-thread, not per-room: his DM shares the same history
    assert TYPED_SURFACE_STYLE.strip() in _chat_prompt(TYPED_DM, thread="person:p2")


def test_voice_turn_keeps_voice_styling_and_never_gets_the_typed_fence():
    system = _chat_prompt(VOICE)
    assert "VOICE conversation" in system
    assert TYPED_SURFACE_STYLE.strip() not in system


def test_lens_turn_is_not_called_typed():
    # a glasses turn is read, but it is SPOKEN in — mis-hearing is real there
    assert TYPED_SURFACE_STYLE.strip() not in _chat_prompt(LENS)


def test_the_fence_names_both_symptoms_it_exists_for():
    low = TYPED_SURFACE_STYLE.lower()
    assert "typed" in low
    assert "[thoughtfully]" in low or "emotion tag" in low   # no stage directions
    assert "speech-to-text" in low or "mis-hearing" in low or "misheard" in low


# ── the action mind (same thread, same bleed) ────────────────────────────────

def test_action_node_fences_a_typed_turn():
    system = _action_prompt(TYPED_DISCORD)
    assert TYPED_SURFACE_STYLE.strip() in system


def test_action_node_leaves_voice_and_lens_turns_alone():
    assert TYPED_SURFACE_STYLE.strip() not in _action_prompt(VOICE)
    assert TYPED_SURFACE_STYLE.strip() not in _action_prompt(LENS)


def test_legacy_voice_thread_is_voice_on_BOTH_minds():
    """The two minds must agree on what a voice turn is.

    is_voice_turn honours a legacy 'voice:*' thread_id as well as the identity
    flag; the action node's own styling chain only ever read the flag. Before the
    typed fence that mismatch was harmless (the action node simply said nothing),
    but a fence that asserts "there is no speech-to-text in this path" would have
    contradicted the chat node's speech styling on the same turn. Caught in
    adversarial review, 2026-09-13.
    """
    flagless = {"user_id": "person-1", "display_name": "Chris"}   # voice by THREAD only
    RecordingModel.seen = []
    model = RecordingModel(messages=iter([AIMessage(content="ok")]))
    ask(build_graph(model, soul="s"), "hey", identity=flagless, thread_id="voice:beta")
    assert "VOICE conversation" in RecordingModel.seen[0]
    assert TYPED_SURFACE_STYLE.strip() not in RecordingModel.seen[0]

    tool_model = RecordingToolModel()
    graph = build_action_graph(tool_model, soul="s", tools=[])
    graph.invoke({"messages": [HumanMessage(content="hey")]},
                 {"configurable": {"identity": flagless, "thread_id": "voice:beta"}})
    assert TYPED_SURFACE_STYLE.strip() not in tool_model.prompts[0][0].content


def test_the_fence_never_names_a_particular_person():
    """Guests type in her rooms too. A fence that says "you and Chris share one
    thread" tells a stranger they are someone else. Adversarial review, 2026-09-13."""
    assert "Chris" not in TYPED_SURFACE_STYLE


def test_the_fence_gives_the_specialist_no_conversational_order():
    """The action mind is her hands: the charter says act or report, never ask.
    The fence must not slip an "ask them about it" into that prompt."""
    assert "ask " not in TYPED_SURFACE_STYLE.lower()
