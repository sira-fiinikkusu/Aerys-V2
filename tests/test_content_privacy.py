"""Offline tests for the short-term content-privacy gate — no network, no DB.

Three layers, all with fakes:
  1. The PURE classifier + gate (services.content_privacy): keyword short-circuit,
     LLM path, fail-closed defaults, and redact_private_history's drop-the-turn logic.
  2. Ingest TAGGING through ask(): a public-channel turn is tagged 'public', a
     DM/private turn is tagged fail-closed 'private' — verified on the checkpointer.
  3. The SECURITY property (Part 3): a private DM turn is ABSENT from a later public
     turn's model input (and so is its reply), yet PRESENT in a private turn — because
     his DM and his public channel now share one person-keyed thread.
  4. The async retag (Part 2): a wired judge relaxes a general DM turn to 'public' so
     it carries into public rooms; a 'private' verdict leaves the fail-closed tag.
"""

import time

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage

from aerys_v2.factory import build_graph
from aerys_v2.service import ask
from aerys_v2.services.content_privacy import (
    CONTENT_PRIVACY_KEY,
    classify_content_privacy,
    content_privacy_of,
    keyword_verdict,
    normalize_verdict,
    redact_private_history,
)

PRIVATE_DM = {"user_id": "person-1", "display_name": "Chris", "privacy_context": "private"}
PUBLIC_GUILD = {
    "user_id": "person-1", "display_name": "Chris", "privacy_context": "public",
    "platform": "discord", "channel_kind": "guild", "channel_id": "555",
}


def fake_model(*replies: str) -> GenericFakeChatModel:
    return GenericFakeChatModel(messages=iter([AIMessage(content=r) for r in replies]))


def _tagged(content: str, tag: str | None) -> HumanMessage:
    kw = {CONTENT_PRIVACY_KEY: tag} if tag is not None else {}
    return HumanMessage(content=content, additional_kwargs=kw)


# ── PURE: keyword_verdict / normalize_verdict / classify_content_privacy ──────

def test_keyword_verdict_flags_private_categories():
    assert keyword_verdict("my therapist says I'm improving") == "private"
    assert keyword_verdict("I'm buried in debt right now") == "private"
    assert keyword_verdict("we're going through a divorce") == "private"
    assert keyword_verdict("the PTSD flashbacks are rough") == "private"
    assert keyword_verdict("I'm bisexual, by the way") == "private"
    # general content has no marker — the keyword pass has "no opinion" (None, NOT public)
    assert keyword_verdict("the number is 42") is None
    assert keyword_verdict("I drive a Dodge Ram and live in Florida") is None


def test_normalize_verdict_is_fail_closed():
    assert normalize_verdict("public") == "public"
    assert normalize_verdict("PRIVATE") == "private"
    assert normalize_verdict("This looks public to me") == "public"
    # ambiguous / garbage / empty -> private (a confused judge must hide, never reveal)
    assert normalize_verdict("hmm not sure") == "private"
    assert normalize_verdict("") == "private"
    assert normalize_verdict(None) == "private"
    # 'private' wins when both words appear (least-disclosure tiebreak)
    assert normalize_verdict("not public, it's private") == "private"


def test_classify_keyword_short_circuits_before_llm():
    calls = []

    def judge(_t):
        calls.append(_t)
        return "public"

    # a keyword hit returns private WITHOUT ever consulting the (would-say-public) judge
    assert classify_content_privacy("my depression is bad", llm=judge) == "private"
    assert calls == []


def test_classify_uses_judge_for_marker_free_text():
    assert classify_content_privacy("how's the weather", llm=lambda _t: "public") == "public"
    assert classify_content_privacy("something subtle", llm=lambda _t: "private") == "private"


def test_classify_judge_error_fails_closed():
    def boom(_t):
        raise RuntimeError("judge down")

    assert classify_content_privacy("marker-free text", llm=boom) == "private"


def test_classify_no_judge_defaults_public_for_marker_free():
    # keyword classifier alone: no marker + no judge -> public (the honest default;
    # the leak-critical async path only ARMS this with a judge present).
    assert classify_content_privacy("just a normal sentence") == "public"


# ── PURE: the gate (redact_private_history) ──────────────────────────────────

def test_gate_drops_private_human_and_its_reply_keeps_public():
    history = [
        _tagged("my therapy is going well", "private"),
        AIMessage(content="glad to hear it"),          # the private turn's reply
        _tagged("what's the weather", "public"),
        AIMessage(content="sunny"),
    ]
    kept = redact_private_history(history)
    contents = [str(m.content) for m in kept]
    assert "my therapy is going well" not in contents
    assert "glad to hear it" not in contents          # reply that referenced private -> gone
    assert "what's the weather" in contents
    assert "sunny" in contents


def test_gate_is_fail_closed_on_untagged_history():
    # legacy/untagged turns (pre-feature) are treated as private in a public view
    history = [_tagged("old message", None), AIMessage(content="old reply")]
    assert redact_private_history(history) == []


def test_gate_keeps_only_explicitly_public():
    history = [_tagged("keep me", "public"), AIMessage(content="kept reply")]
    kept = redact_private_history(history)
    assert [str(m.content) for m in kept] == ["keep me", "kept reply"]


# ── ingest TAGGING through ask() (verified on the checkpointer) ──────────────

def _human_tag(graph, thread_id, needle):
    state = graph.get_state({"configurable": {"thread_id": thread_id}})
    for m in state.values.get("messages", []):
        if getattr(m, "type", "") == "human" and needle in str(m.content):
            return content_privacy_of(m)
    return "<not found>"


def test_public_origin_human_tagged_public():
    graph = build_graph(fake_model("ok"), soul="s")
    ask(graph, "hi all", identity=PUBLIC_GUILD, thread_id="person:p1")
    assert _human_tag(graph, "person:p1", "hi all") == "public"


def test_private_origin_human_tagged_private_failclosed():
    graph = build_graph(fake_model("ok"), soul="s")
    ask(graph, "hey there", identity=PRIVATE_DM, thread_id="person:p1")
    assert _human_tag(graph, "person:p1", "hey there") == "private"


# ── the SECURITY property: private DM content is walled out of a public turn ──

class CapturingModel(GenericFakeChatModel):
    """Fake that records the NON-system messages each invoke actually saw."""

    seen: list = []

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        type(self).seen.append(
            [str(m.content) for m in messages if getattr(m, "type", "") != "system"]
        )
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)


def test_private_dm_content_absent_from_public_turn_present_in_private():
    CapturingModel.seen = []
    m = CapturingModel(
        messages=iter([AIMessage(content="a"), AIMessage(content="b"), AIMessage(content="c")])
    )
    graph = build_graph(m, soul="s")

    # 1) a private DM turn on the person thread (no judge -> stays fail-closed private)
    ask(graph, "my depression has been rough lately", identity=PRIVATE_DM, thread_id="person:p1")
    # 2) a PUBLIC guild turn on the SAME person-keyed thread
    ask(graph, "hey everyone", identity=PUBLIC_GUILD, thread_id="person:p1")
    # 3) a private DM turn again on the same thread
    ask(graph, "still just me", identity=PRIVATE_DM, thread_id="person:p1")

    public_seen = CapturingModel.seen[1]   # what the model saw on the public turn
    private_seen = CapturingModel.seen[2]  # what it saw on the later private turn

    # THE GATE: private DM content — and the reply it produced — never reach public.
    assert not any("depression" in s for s in public_seen)
    assert "a" not in public_seen                     # the private turn's reply is gone too
    assert any("hey everyone" in s for s in public_seen)  # the current message is still there

    # ...but in a private DM the owner sees his own history in full.
    assert any("depression" in s for s in private_seen)


# ── async retag: a judge relaxes general DM content to 'public' ──────────────

def _wait_for_tag(graph, thread_id, needle, want, timeout=3.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if _human_tag(graph, thread_id, needle) == want:
            return want
        time.sleep(0.02)
    return _human_tag(graph, thread_id, needle)


def test_judge_relaxes_general_dm_turn_to_public():
    graph = build_graph(fake_model("noted"), soul="s")
    ask(graph, "the alarm code is 4231", identity=PRIVATE_DM, thread_id="person:p1",
        content_privacy_classifier=lambda _t: "public")
    # the daemon-thread retag flips the fail-closed 'private' to 'public'
    assert _wait_for_tag(graph, "person:p1", "alarm code", "public") == "public"


def test_judge_private_verdict_leaves_failclosed_tag():
    graph = build_graph(fake_model("noted"), soul="s")
    ask(graph, "my anxiety spiked today", identity=PRIVATE_DM, thread_id="person:p1",
        content_privacy_classifier=lambda _t: "private")
    time.sleep(0.2)   # give any (there should be none) retag a chance to land
    assert _human_tag(graph, "person:p1", "anxiety") == "private"


def test_public_origin_never_consults_judge():
    calls = []
    graph = build_graph(fake_model("ok"), soul="s")
    ask(graph, "hi all", identity=PUBLIC_GUILD, thread_id="person:p1",
        content_privacy_classifier=lambda t: calls.append(t) or "private")
    time.sleep(0.2)
    assert calls == []                                 # a public turn is already public
    assert _human_tag(graph, "person:p1", "hi all") == "public"


def test_action_path_tags_and_can_relax_human_turn():
    # the ACTION path lands the human turn in history via _action_turn — it must be
    # tagged (fail-closed private) and eligible for the same async relaxation.
    from aerys_v2.router import RouteDecision

    class StubAction:
        def invoke(self, inp, config):
            return {"messages": [AIMessage(content="Done — noted.")]}

    graph = build_graph(fake_model("unused"), soul="s")
    ask(graph, "remember the wifi is guest1234", identity=PRIVATE_DM, thread_id="person:p1",
        router=lambda _t: RouteDecision(route="action", ack="on it"),
        action_graph=StubAction(),
        content_privacy_classifier=lambda _t: "public")
    assert _wait_for_tag(graph, "person:p1", "wifi", "public") == "public"


def test_content_privacy_fn_for_arming():
    import types

    from pydantic import SecretStr

    from aerys_v2.factory import content_privacy_fn_for

    # None without a judge (no anthropic key) — DM content stays fail-closed private
    assert content_privacy_fn_for(types.SimpleNamespace(anthropic_api_key=None)) is None
    # a callable classifier when a key can arm the judge (no network at construction)
    fn = content_privacy_fn_for(
        types.SimpleNamespace(anthropic_api_key=SecretStr("sk-test"), tier_fast_model="claude-haiku-4-5")
    )
    assert callable(fn)
