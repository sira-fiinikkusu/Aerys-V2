"""The fence told her not to copy the tags. Her own history kept telling her to.

Measured after the typed-surface fence shipped (6568b36): one of fourteen typed turns
still carried an emotion tag — down from routine, but not gone. The one that slipped
was nine characters in and nineteen out: "wonderful" -> "[warmly] Good test."

The reason is a loop, not a weak sentence. Tags are stripped at the door for a screen,
but the message she keeps is the one the model produced, tags and all. So her thread
fills with tagged replies, and on a short turn — where there is little else in the
immediate exchange to go on — the precedent in her own history outweighs an instruction
further up the prompt. She is imitating herself.

Two fixes, both about telling the truth rather than pushing harder.

1. On a TYPED turn, prior replies of hers arrive with the tags removed, exactly as the
   reader saw them. She is not being asked to ignore a pattern; the pattern is not
   there. The checkpoint keeps the literal record, so nothing is lost — this is what
   she is SHOWN, the same seam that already redacts private history.

2. The fence claimed those tags were "history from another surface". Some of them are
   not: she tagged a Discord DM reply on 2026-09-06. Telling her something her own
   context disproves is how the portable heading failed the same day — she has no
   reason to trust the rest of the block.
"""
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage

from aerys_v2.factory import TYPED_SURFACE_STYLE, build_graph
from aerys_v2.service import ask

TYPED = {"user_id": "person-1", "display_name": "Chris", "privacy_context": "private",
         "platform": "discord", "channel_kind": "dm", "channel_id": "9"}
#: Voice by THREAD rather than by the flag: the flag routes through the speculative
#: thread, which is a different question from the one this file is asking.
VOICE = {"user_id": "person-1", "display_name": "Chris", "privacy_context": "private"}
VOICE_THREAD = "voice:beta"


class Recorder(GenericFakeChatModel):
    seen: list = []

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        type(self).seen.append(list(messages))
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)


def _turn(identity, prior_reply, thread):
    # A checkpointer, because the whole point is what her OWN HISTORY shows her.
    from langgraph.checkpoint.memory import InMemorySaver

    Recorder.seen = []
    model = Recorder(messages=iter([AIMessage(content=prior_reply), AIMessage(content="ok")]))
    graph = build_graph(model, soul="s", checkpointer=InMemorySaver())
    ask(graph, "first", identity=identity, thread_id=thread)
    ask(graph, "wonderful", identity=identity, thread_id=thread)
    return Recorder.seen[1]


def test_a_typed_turn_never_sees_its_own_tags_in_the_history():
    prompt = _turn(TYPED, "[warmly] Good to see you.", "person:typed")
    prior = [m for m in prompt if isinstance(m, AIMessage)]
    assert prior, 'the prior reply must still be there'
    assert "Good to see you." in prior[0].content
    assert "[warmly]" not in prior[0].content, 'the pattern is gone, not merely forbidden'


def test_a_voice_turn_keeps_them_because_they_were_spoken():
    prompt = _turn(VOICE, "[warmly] Good to see you.", VOICE_THREAD)
    prior = [m for m in prompt if isinstance(m, AIMessage)]
    assert "[warmly]" in prior[0].content


def test_only_real_tags_go_and_his_words_are_never_touched():
    prompt = _turn(TYPED, "I read [July] and [reply] as words.", "person:words")
    prior = [m for m in prompt if isinstance(m, AIMessage)]
    assert "[July]" in prior[0].content and "[reply]" in prior[0].content
    human = [m for m in prompt if isinstance(m, HumanMessage)]
    assert any(m.content == "first" for m in human)


def test_the_fence_no_longer_claims_the_tags_came_from_elsewhere():
    """She tagged a Discord DM reply on 2026-09-06. A fence that says those came from
    another surface is contradicted by her own context, which is how the portable
    heading failed the same day."""
    low = TYPED_SURFACE_STYLE.lower()
    assert "another surface" not in low
    assert "stage direction" in low or "speech engine" in low


# ── what the rewrite must NOT disturb (adversarial review, 2026-09-14) ───────

def _untag(messages):
    from aerys_v2.factory import _untag_own_replies

    return _untag_own_replies(messages)


def test_the_id_and_the_tool_calls_survive_the_rewrite():
    """It is a prompt-time copy, not a new message. Anything keyed on identity —
    tool-result matching, the checkpointer, a replay — must not notice."""
    original = AIMessage(content="[warmly] checking", id="msg-1",
                         tool_calls=[{"name": "search", "args": {"q": "x"}, "id": "call-1"}])
    out = _untag([original])[0]
    assert out.content == "checking"
    assert out.id == "msg-1"
    assert out.tool_calls == original.tool_calls


def test_a_multimodal_reply_is_cleaned_inside_its_text_blocks():
    """An assistant reply can be a list of blocks rather than a string. Skipping
    those would leave the imitation loop open on exactly the turns that carry an
    image, which is not a rare shape on Discord."""
    out = _untag([AIMessage(content=[{"type": "text", "text": "[warmly] here it is"},
                                     {"type": "image_url", "image_url": {"url": "u"}}])])[0]
    assert out.content[0]["text"] == "here it is"
    assert out.content[1] == {"type": "image_url", "image_url": {"url": "u"}}


def test_nothing_to_do_returns_the_very_same_list():
    """The usual turn pays nothing: no walk, no copies, no new list."""
    messages = [HumanMessage(content="hi"), AIMessage(content="Hello.")]
    assert _untag(messages) is messages


def test_his_words_are_never_rewritten_even_when_they_look_like_tags():
    messages = [HumanMessage(content="[warmly] is what you keep saying")]
    assert _untag(messages) is messages
