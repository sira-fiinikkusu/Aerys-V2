"""The extractor on the Anthropic API (Chris, 2026-10-07).

Max plans now include monthly API credit; the extractor was the one Claude caller
still billed through OpenRouter. anthropic_chat is the same Llm seam as
openrouter_chat: schema-enforced JSON in, the observations array out, and an
honest `truncated` flag so a cut-off reply holds the watermark.
"""
import json
from types import SimpleNamespace

from aerys_v2.workers.extraction import _RESPONSE_FORMAT, anthropic_chat


class FakeMessages:
    def __init__(self, text, stop_reason="end_turn"):
        self.calls = []
        self._reply = SimpleNamespace(content=[SimpleNamespace(type="text", text=text)],
                                      stop_reason=stop_reason)

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self._reply


def fake_client(text, stop_reason="end_turn"):
    return SimpleNamespace(messages=FakeMessages(text, stop_reason))


def test_it_returns_the_observations_array_and_enforces_the_schema():
    client = fake_client(json.dumps({"observations": [{"key_label": "pet", "value_text": "a cat"}]}))
    llm = anthropic_chat("sk-test", model="claude-haiku-5-5", client=client)
    reply = llm("system prompt", "transcript")
    assert json.loads(reply.text) == [{"key_label": "pet", "value_text": "a cat"}]
    assert reply.truncated is False
    (call,) = client.messages.calls
    assert call["model"] == "claude-haiku-5-5" and call["system"] == "system prompt"
    assert call["messages"] == [{"role": "user", "content": "transcript"}]
    assert call["output_config"] == {"format": {"type": "json_schema",
                                                "schema": _RESPONSE_FORMAT["json_schema"]["schema"]}}
    assert "temperature" not in call, "Haiku 5.5 rejects temperature (deprecated for the model)"


def test_a_cut_off_reply_is_reported_truncated():
    llm = anthropic_chat("sk-test", client=fake_client('{"observations": [', stop_reason="max_tokens"))
    assert llm("s", "u").truncated is True


def test_plain_mode_drops_the_schema_for_the_fallback_retry():
    client = fake_client("[]")
    anthropic_chat("sk-test", client=client)("s", "u", plain=True)
    assert "output_config" not in client.messages.calls[0]


def test_the_worker_picks_the_anthropic_seam_by_default(monkeypatch):
    from aerys_v2.workers import __main__ as workers

    picked = {}
    monkeypatch.setattr(workers, "anthropic_chat", lambda key, **kw: picked.setdefault("anthropic", kw) or "LLM")
    monkeypatch.setattr(workers, "openrouter_chat", lambda key, **kw: picked.setdefault("openrouter", kw) or "LLM")
    settings = SimpleNamespace(extraction_backend="anthropic", extraction_anthropic_model="claude-haiku-5-5",
                               anthropic_api_key=SimpleNamespace(get_secret_value=lambda: "sk"),
                               embeddings_api_key=SimpleNamespace(get_secret_value=lambda: "or"),
                               extraction_model="anthropic/claude-haiku-5.5", embeddings_base_url="https://x")
    workers.extraction_llm(settings)
    assert picked == {"anthropic": {"model": "claude-haiku-5-5"}}
    settings.extraction_backend = "openrouter"
    picked.clear()
    workers.extraction_llm(settings)
    assert "openrouter" in picked


def test_a_refusal_is_an_empty_reply_so_it_cannot_stall_the_watermark():
    """A declined group would be declined again every pass; reported as truncated
    it would hold the watermark forever and stop extraction for everyone."""
    llm = anthropic_chat("sk-test", client=fake_client("", stop_reason="refusal"))
    reply = llm("s", "u")
    assert (reply.text, reply.truncated) == ("[]", False)
