"""The lifeboat prompt (gap #118, board #156): her full prompt (~21.5k tokens on 10/25) can't fit the
local lifeboat (4096 tokens), so a turn that falls back gets a slim prompt instead: a short core
system message, the recent turns that fit a budget, and the last thing said, always. A prompt that
already fits passes through unchanged."""

from unittest.mock import Mock

import anthropic
import httpx
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from aerys_v2.factory import (
    LIFEBOAT_MAX_CHARS,
    LIFEBOAT_SYSTEM,
    LOCAL_ACTION_HARDENING,
    LocalFailoverModel,
    LocalToolFailoverModel,
    lifeboat_prompt,
)


def chars(messages):
    return sum(len(m.content) if isinstance(m.content, str) else len(str(m.content)) for m in messages)


def big_system():
    return SystemMessage(content="Charter and profile. " * 6000, id="s")   # ~126k chars, like her real prompt


def test_a_prompt_that_fits_passes_through_unchanged():
    prompt = [SystemMessage(content="Short charter.", id="s"), HumanMessage(content="turn on the office light")]
    assert lifeboat_prompt(prompt) == prompt


def test_an_oversized_prompt_gets_the_core_system_and_the_last_message():
    out = lifeboat_prompt([big_system(), HumanMessage(content="turn on the office light")])
    assert isinstance(out[0], SystemMessage)
    assert out[0].content.startswith(LIFEBOAT_SYSTEM)
    assert "Charter and profile." not in out[0].content
    assert out[-1] == HumanMessage(content="turn on the office light")
    assert chars(out) <= LIFEBOAT_MAX_CHARS


def test_recent_turns_are_kept_newest_first_until_the_budget_and_start_on_a_human_turn():
    history = []
    for i in range(40):
        history += [HumanMessage(content=f"question {i} " + "x" * 400), AIMessage(content=f"answer {i} " + "y" * 400)]
    history.append(HumanMessage(content="and the last one"))
    out = lifeboat_prompt([big_system(), *history])
    kept = out[1:]
    assert isinstance(kept[0], HumanMessage)
    assert kept == history[-len(kept):]
    assert len(kept) > 1
    assert chars(out) <= LIFEBOAT_MAX_CHARS


def test_a_tool_exchange_is_never_split():
    tool_turn = [
        HumanMessage(content="turn on the light " + "z" * 300),
        AIMessage(content="", tool_calls=[{"name": "home_control", "args": {"operation": "on"}, "id": "c1"}]),
        ToolMessage(content="Done: on verified", tool_call_id="c1"),
    ]
    older = [HumanMessage(content="older " + "o" * 3000), AIMessage(content="ok " + "o" * 3000)] * 4
    out = lifeboat_prompt([big_system(), *older, *tool_turn])
    kept = out[1:]
    assert isinstance(kept[0], HumanMessage)
    assert kept[-3:] == tool_turn
    for i, m in enumerate(kept):
        if isinstance(m, ToolMessage):
            assert isinstance(kept[i - 1], AIMessage) and kept[i - 1].tool_calls


def test_an_oversized_current_tool_exchange_keeps_its_tool_result():
    # mid-tool-loop on the lifeboat: dropping the ToolMessage would make it repeat the action
    tool_turn = [
        HumanMessage(content="turn on the light"),
        AIMessage(content="", tool_calls=[{"name": "home_control", "args": {"operation": "on"}, "id": "c1"}]),
        ToolMessage(content="Done: on verified " + "r" * 50000, tool_call_id="c1"),
    ]
    out = lifeboat_prompt([big_system(), *tool_turn])
    kept = out[1:]
    assert [type(m) for m in kept] == [HumanMessage, AIMessage, ToolMessage]
    assert kept[0].content == "turn on the light"
    assert kept[1].tool_calls[0]["id"] == "c1"
    assert kept[2].content.startswith("Done: on verified")
    assert chars(out) <= LIFEBOAT_MAX_CHARS


def test_a_huge_last_message_is_cut_down_but_kept():
    out = lifeboat_prompt([big_system(), HumanMessage(content="please read " + "w" * 200000)])
    assert isinstance(out[-1], HumanMessage)
    assert out[-1].content.startswith("please read ")
    assert chars(out) <= LIFEBOAT_MAX_CHARS


def test_hardening_rides_on_the_core_system_for_the_tool_path():
    out = lifeboat_prompt([big_system(), HumanMessage(content="lights off")], hardening=LOCAL_ACTION_HARDENING)
    assert out[0].content == LIFEBOAT_SYSTEM + LOCAL_ACTION_HARDENING


def connection_error():
    return anthropic.APIConnectionError(request=httpx.Request("POST", "https://cloud.test"))


def test_the_chat_lifeboat_receives_the_slim_prompt():
    primary, lifeboat = Mock(), Mock()
    primary.invoke.side_effect = connection_error()
    lifeboat.invoke.return_value = AIMessage(content="I'm offline right now.")
    model = LocalFailoverModel.model_construct(primary=primary, lifeboat=lifeboat)
    model.invoke([big_system(), HumanMessage(content="what's the weather")])
    sent = lifeboat.invoke.call_args.args[0]
    assert sent[0].content.startswith(LIFEBOAT_SYSTEM)
    assert sent[-1].content == "what's the weather"
    assert chars(sent) <= LIFEBOAT_MAX_CHARS


def test_the_tool_lifeboat_receives_the_slim_prompt_with_hardening():
    primary, lifeboat = Mock(), Mock()
    primary.invoke.side_effect = connection_error()
    model = LocalToolFailoverModel(primary, lifeboat)
    model.invoke([big_system(), HumanMessage(content="office lights off")])
    sent = lifeboat.invoke.call_args.args[0]
    assert sent[0].content == LIFEBOAT_SYSTEM + LOCAL_ACTION_HARDENING
    assert sent[-1].content == "office lights off"
    assert chars(sent) <= LIFEBOAT_MAX_CHARS + len(LOCAL_ACTION_HARDENING)
