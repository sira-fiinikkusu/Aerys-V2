"""remember tool: trust is mechanical, writes go through the writer, replies are honest."""
from unittest.mock import Mock

import pytest

from aerys_v2 import router
from aerys_v2.tools import remember as mod
from aerys_v2.tools.remember import (ALREADY_PREFIX, CURRENT_TURN_TEXT, EMPTY, KEPT_PREFIX, NOT_KEPT,
                                     NOT_LINKED, build_remember_tool, key_label_for, trust_for)

PERSON = "6e6bcbed-03ef-4d17-95d2-89c467414335"


def cfg(user_id=PERSON, privacy="private", platform="discord", channel_id="123"):
    return {"configurable": {"identity": {"user_id": user_id, "privacy_context": privacy,
                                           "platform": platform, "channel_id": channel_id}}}


def test_trust_is_owner_only_when_the_fact_quotes_the_turn():
    turn = "hey, remember that the pool guy comes on Tuesdays"
    assert trust_for("the pool guy comes on Tuesdays", turn) == "owner"
    assert trust_for("Pool maintenance is scheduled weekly", turn) == "assistant"
    assert trust_for("the pool guy comes on Tuesdays", "") == "assistant"
    assert trust_for("", turn) == "assistant"
    # Unicode quotes are the owner's word too (tokens are not ASCII-only).
    assert trust_for("犬の名前はポチ", "覚えておいて、犬の名前はポチ") == "owner"
    assert trust_for("Le chien s'appelle Émile", "retiens que le chien s'appelle Émile") == "owner"


def test_key_label_is_stable_per_fact_and_distinct_across_facts():
    assert key_label_for("The pool guy comes Tuesdays!") == key_label_for("the pool guy comes tuesdays")
    assert key_label_for("pool guy Tuesdays") != key_label_for("pool guy Wednesdays")
    assert key_label_for("x").startswith("remember.")


def test_kept_only_when_the_writer_confirms():
    writer = Mock(return_value="insert")
    tool = build_remember_tool(writer)
    CURRENT_TURN_TEXT.set("remember that the pool guy comes on Tuesdays")
    reply = tool.invoke({"fact": "the pool guy comes on Tuesdays"}, config=cfg())
    assert reply.startswith(KEPT_PREFIX)
    record = writer.call_args.args[0]
    assert record["person_id"] == PERSON and record["trust"] == "owner"
    assert record["privacy_level"] == "private" and record["source_platform"] == "discord"
    assert record["key_label"] == key_label_for("the pool guy comes on Tuesdays")
    writer.return_value = "skipped"
    assert tool.invoke({"fact": "the pool guy comes on Tuesdays"}, config=cfg()).startswith(ALREADY_PREFIX)


def test_inference_is_assistant_trust_and_public_rooms_stay_public():
    writer = Mock(return_value="update")
    tool = build_remember_tool(writer)
    CURRENT_TURN_TEXT.set("the pool is cleaned weekly I think")
    reply = tool.invoke({"fact": "Owner prefers the pool cleaned on Tuesdays"}, config=cfg(privacy="public"))
    assert reply.startswith(KEPT_PREFIX)
    record = writer.call_args.args[0]
    assert record["trust"] == "assistant" and record["privacy_level"] == "public"


@pytest.mark.parametrize("failure", [Exception("db down"), None])
def test_failure_and_unknown_actions_never_claim_kept(failure):
    writer = Mock(side_effect=failure) if failure else Mock(return_value="weird")
    tool = build_remember_tool(writer)
    assert tool.invoke({"fact": "anything"}, config=cfg()) == NOT_KEPT


def test_gates_empty_fact_and_unlinked_caller():
    writer = Mock(return_value="insert")
    tool = build_remember_tool(writer)
    assert tool.invoke({"fact": "   "}, config=cfg()) == EMPTY
    assert tool.invoke({"fact": "x"}, config=cfg(user_id="http-caller")) == NOT_LINKED
    assert tool.invoke({"fact": "x"}, config=None) == NOT_LINKED
    writer.assert_not_called()


def test_long_facts_are_truncated_not_refused():
    writer = Mock(return_value="insert")
    tool = build_remember_tool(writer)
    reply = tool.invoke({"fact": "word " * 400}, config=cfg())
    assert reply.startswith(KEPT_PREFIX) and len(writer.call_args.args[0]["fact"]) <= mod.FACT_LIMIT


def test_tool_schema_has_no_trust_or_confirmation_parameter():
    tool = build_remember_tool(Mock())
    params = set(tool.args_schema.model_json_schema()["properties"])
    assert params == {"fact"}  # config is injected, never model-visible


@pytest.mark.parametrize("text", [
    "remember that the pool guy comes Tuesdays", "Keep in mind I hate cilantro",
    "make a note that the gate code changed", "don't forget the vet is Friday",
    "for next time, Megan takes oat milk", "note that for later", "remind me the gate code is 4411",
])
def test_router_fallback_routes_keep_requests_to_action(text):
    assert router.plausibly_wants_remember(text)
    assert router.fallback_decision(text).route == "action"


@pytest.mark.parametrize("text", [
    "remember when we drove to Tampa?", "do you remember my birthday?",
    "I can't remember the movie name", "what do you remember about the wedding",
    "remember that time we drove to Tampa?", "do you remember that I hate cilantro?",
])
def test_router_fallback_keeps_recall_and_reminiscing_in_chat(text):
    assert not router.plausibly_wants_remember(text)
    assert router.fallback_decision(text).route == "chat"


def test_router_prompt_advertises_the_distinction():
    assert "REMEMBERING" in router._ROUTER_INSTRUCTIONS
    assert "remember when" in router._ROUTER_INSTRUCTIONS


def test_armed_only_with_memories_and_embeddings_and_never_for_guests():
    from aerys_v2.config import Settings
    from aerys_v2.factory import REMEMBER_OVERLAY, _action_tools_armed, action_overlay_for, remember_writer_for

    bare = Settings(_env_file=None, anthropic_api_key="k")
    assert remember_writer_for(bare) is None
    assert REMEMBER_OVERLAY not in action_overlay_for(bare)
    assert "remember" not in [t.name for t in _action_tools_armed(bare)]
    half = Settings(_env_file=None, anthropic_api_key="k", memories_database_url="postgresql://x/aerys")
    assert remember_writer_for(half) is None and REMEMBER_OVERLAY not in action_overlay_for(half)
    armed = Settings(_env_file=None, anthropic_api_key="k", memories_database_url="postgresql://x/aerys",
                     embeddings_api_key="e")
    assert callable(remember_writer_for(armed))
    assert REMEMBER_OVERLAY in action_overlay_for(armed)
    assert "remember" in [t.name for t in _action_tools_armed(armed)]
    assert "remember" not in [t.name for t in _action_tools_armed(armed, guest=True)]
    assert REMEMBER_OVERLAY not in action_overlay_for(armed, guest=True)


def test_previous_human_turn_completes_the_owner_quote():
    from langchain_core.messages import AIMessage, HumanMessage
    from aerys_v2.service import _remember_window
    seed = [HumanMessage("do you remember when my birthday is?"), AIMessage("I don't have it."),
            HumanMessage("can you remember its on August 24th, 1987")]
    window = _remember_window(seed, "can you remember its on August 24th, 1987")
    assert trust_for("my birthday is on August 24th, 1987", window) == "owner"
    assert trust_for("Chris was born in the eighties", window) == "assistant"
    assert _remember_window([HumanMessage("only this")], "only this") == "only this"
    from aerys_v2.service import _remember_window_for
    private = {"configurable": {"identity": {"privacy_context": "private"}}}
    public = {"configurable": {"identity": {"privacy_context": "public"}}}
    assert "birthday" in _remember_window_for(seed, "the date line", private)
    assert _remember_window_for(seed, "the date line", public) == "the date line"


def test_emotion_tags_are_stripped_for_text_surfaces_only():
    from aerys_v2.service import _for_surface, strip_emotion_tags
    tagged = "[thoughtfully] I don't have your birthday on record.\n\n[warmly] When is it?"
    assert strip_emotion_tags(tagged) == "I don't have your birthday on record.\n\nWhen is it?"
    assert strip_emotion_tags("[the office] is a room, [laughs] fine") == "[the office] is a room, fine"
    assert strip_emotion_tags("no brackets here") == "no brackets here"
    assert strip_emotion_tags("see my [reply] in [July], [only] once") == "see my [reply] in [July], [only] once"
    assert strip_emotion_tags("Hello [warmly]. Then [softly] , yes") == "Hello. Then, yes"
    text_cfg = {"configurable": {"thread_id": "person:x", "identity": {"user_id": "u", "platform": "discord"}}}
    voice_cfg = {"configurable": {"thread_id": "person:x", "identity": {"user_id": "u", "voice": True}}}
    assert _for_surface(tagged, text_cfg).startswith("I don't")
    assert _for_surface(tagged, voice_cfg) == tagged
