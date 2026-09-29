"""Offline tests for the OAuth backend — the SDK boundary is faked, nothing spawns."""

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from aerys_v2.config import Settings
from aerys_v2.factory import build_model
from aerys_v2.oauth_model import ClaudeOAuthChatModel, _flatten, _prompt_blocks, _prompt_text, _sum_usage


def settings(backend: str) -> Settings:
    return Settings(anthropic_api_key="sk-test", model_backend=backend)  # type: ignore[arg-type]


def test_factory_picks_oauth_backend():
    m = build_model(settings("oauth"))
    assert isinstance(m, ClaudeOAuthChatModel)


def test_factory_default_stays_api():
    m = build_model(settings("api"))
    assert not isinstance(m, ClaudeOAuthChatModel)


def test_flatten_system_leads_and_speakers_labeled():
    prompt = _flatten(
        [
            SystemMessage(content="be aerys"),
            HumanMessage(content="hi"),
            AIMessage(content="hey"),
            HumanMessage(content="what number?"),
        ]
    )
    assert prompt.startswith("[System instructions]\nbe aerys\n\n")
    assert prompt.endswith("User: hi\nAerys: hey\nUser: what number?\nAerys:")


def test_generate_uses_result_message(monkeypatch):
    model = ClaudeOAuthChatModel()

    def fake_query(prompt):
        assert "User: ping" in _prompt_text(prompt)
        return "pong"

    monkeypatch.setattr(model, "_query", fake_query)
    out = model.invoke([SystemMessage(content="s"), HumanMessage(content="ping")])
    assert out.content == "pong"


def test_error_result_raises(monkeypatch):
    model = ClaudeOAuthChatModel()

    def fake_query(prompt):
        raise RuntimeError("oauth backend error: 'refused'")

    monkeypatch.setattr(model, "_query", fake_query)
    with pytest.raises(RuntimeError):
        model.invoke([HumanMessage(content="x")])


def test_connect_disables_all_builtin_tools(monkeypatch):
    """Regression guard for the 2026-07-03 voice bug: `allowed_tools=[]` is only
    auto-permission — the CLI still exposes every built-in tool unless `tools=[]`
    is set, and a tool attempt under max_turns=1 dies as error_max_turns."""
    import claude_agent_sdk

    captured = {}

    class FakeClient:
        def __init__(self, options=None):
            captured["options"] = options

        async def connect(self):
            pass

    monkeypatch.setattr(claude_agent_sdk, "ClaudeSDKClient", FakeClient)
    from aerys_v2.oauth_model import _WarmClient

    w = _WarmClient("claude-sonnet-5")
    client, cwd = w._run(w._connect())          # v3: one fresh process, in its own empty cwd
    assert isinstance(client, FakeClient)
    opts = captured["options"]
    assert opts.cwd == cwd and cwd.startswith("/tmp") and opts.setting_sources == []
    assert opts.tools == []          # no built-in tools EXIST for the chat backend
    assert opts.allowed_tools == []  # and none would be auto-permitted anyway
    assert opts.max_turns == 1


def test_rate_limit_reads_the_events_info():
    """2026-09-28: the fields live on rate_limit_info; read off the event itself, the
    meter was all None on every call."""
    from claude_agent_sdk.types import RateLimitEvent, RateLimitInfo

    from aerys_v2.oauth_model import _rate_limit

    event = RateLimitEvent(rate_limit_info=RateLimitInfo(status="allowed_warning", resets_at=1790000000,
                                                         rate_limit_type="five_hour", utilization=0.82,
                                                         overage_status="rejected"),
                           uuid="u", session_id="s")
    assert _rate_limit(event) == {"status": "allowed_warning", "type": "five_hour", "utilization": 0.82,
                                  "resets_at": 1790000000, "overage_status": "rejected"}


def test_blocks_are_the_same_prompt_and_the_mark_sits_before_her_last_user_line():
    """2026-09-28: as one string, no call reused the previous call's cache. The blocks must
    say exactly what the string said."""
    msgs = [SystemMessage(content="SOUL"), HumanMessage(content="hi"), AIMessage(content="hey"),
            HumanMessage(content="what number?")]
    blocks = _prompt_blocks(msgs)
    assert _prompt_text(blocks) == _flatten(msgs)
    marked = [i for i, b in enumerate(blocks) if "cache_control" in b]
    assert marked == [2] and blocks[2]["text"] == "\nAerys: hey"          # before "\nUser: what number?"
    assert blocks[2]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}
    assert blocks[-1]["text"] == "\nAerys:" and all(b["text"] for b in blocks)
    only = _prompt_blocks([SystemMessage(content="SOUL"), HumanMessage(content="hi")])
    assert "cache_control" in only[0], "no history yet: the system block is the prefix"
    assert _prompt_text(_prompt_blocks(msgs, extra="\nMORE")).endswith("\nAerys:\nMORE")


def test_each_turn_starts_with_the_last_turns_cached_prefix():
    """The property the cache needs, on her real layout: this turn's context rides a COPY
    of the last user message, the stored thread keeps it plain."""
    from aerys_v2.history import prompt_with_context

    def cached(blocks):
        cut = next(i for i, b in enumerate(blocks) if "cache_control" in b)
        return [b["text"] for b in blocks[:cut + 1]]

    thread = [HumanMessage(content="hi"), AIMessage(content="hey")]
    for turn in range(3):
        thread.append(HumanMessage(content=f"question {turn}"))
        now = _prompt_blocks(prompt_with_context("SOUL", thread, f"time 10:0{turn}"))
        thread.append(AIMessage(content=f"answer {turn}"))
        thread.append(HumanMessage(content=f"question {turn + 1}"))
        after = _prompt_blocks(prompt_with_context("SOUL", thread, f"time 10:1{turn}"))
        thread.pop()
        assert [b["text"] for b in after[:len(cached(now))]] == cached(now)


def test_usage_counts_what_the_plan_cached():
    total = _sum_usage({"usage": {"input_tokens": 2, "output_tokens": 20, "cache_read_input_tokens": 10848,
                                  "cache_creation_input_tokens": 236}})
    assert total["input_tokens"] == 11086 and total["output_tokens"] == 20
    assert total["input_token_details"] == {"cache_read": 10848, "cache_creation": 236}
    assert _sum_usage({}) is None


def test_a_refused_cache_mark_is_retried_once_without_it(monkeypatch):
    """Gemini review 2026-09-28: past the API's four breakpoints, every turn would 400.
    The retry is the same text, uncached."""
    model = ClaudeOAuthChatModel()
    seen = []

    def fake_query(prompt):
        seen.append(prompt)
        if len(seen) == 1:
            raise RuntimeError("oauth backend error: result='API Error: 400 messages.0.content.2.cache_control: "
                               "A maximum of 4 blocks with cache_control may be provided'")
        return "pong"

    monkeypatch.setattr(model, "_query", fake_query)
    assert model.invoke([SystemMessage(content="s"), HumanMessage(content="hi"), AIMessage(content="hey"),
                         HumanMessage(content="ping")]).content == "pong"
    assert any("cache_control" in b for b in seen[0]) and not any("cache_control" in b for b in seen[1])
    assert _prompt_text(seen[0]) == _prompt_text(seen[1])

    def other_error(prompt):
        raise RuntimeError("oauth backend error: result='overloaded'")

    monkeypatch.setattr(model, "_query", other_error)
    with pytest.raises(RuntimeError, match="overloaded"):
        model.invoke([HumanMessage(content="x")])
