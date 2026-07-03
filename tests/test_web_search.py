"""Offline tests for the web-search tool — fake Tavily via httpx.MockTransport.

What these prove: the request hits Tavily's /search with the documented body
(api_key, query, max_results, search_depth=basic, include_answer=True), the
reply is formatted compactly (answer first, then numbered `title | url | snippet`
with snippets truncated), empty results and every failure mode come back as an
HONEST string (never a raise — the ToolNode contract), and a blank key is caught
at construction.
"""

import json

import httpx
import pytest

from aerys_v2.tools.web_search import (
    SNIPPET_MAX_CHARS,
    TAVILY_SEARCH_URL,
    build_web_search_tool,
)


class FakeTavily:
    """Records every request; answers /search like Tavily would."""

    def __init__(self, payload: dict | None = None, status: int = 200):
        self.requests: list[httpx.Request] = []
        self.status = status
        self.payload = payload

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if self.status >= 400:
            return httpx.Response(self.status, json={"error": "nope"})
        return httpx.Response(200, json=self.payload or {})

    def client(self) -> httpx.Client:
        return httpx.Client(transport=httpx.MockTransport(self.handler))

    def sent_json(self, idx: int = -1) -> dict:
        return json.loads(self.requests[idx].content)


def make_tool(fake: FakeTavily):
    return build_web_search_tool(api_key="tvly-test", client=fake.client())


# ---- request shape ---------------------------------------------------------------

def test_request_hits_tavily_with_documented_body():
    fake = FakeTavily(payload={"answer": "42.", "results": []})
    make_tool(fake).invoke({"query": "meaning of life", "max_results": 3})
    req = fake.requests[0]
    assert str(req.url) == TAVILY_SEARCH_URL
    body = fake.sent_json()
    assert body == {
        "api_key": "tvly-test",
        "query": "meaning of life",
        "max_results": 3,
        "search_depth": "basic",
        "include_answer": True,
    }


def test_max_results_defaults_to_five_and_clamps():
    fake = FakeTavily(payload={"results": []})
    make_tool(fake).invoke({"query": "x"})
    assert fake.sent_json()["max_results"] == 5
    fake2 = FakeTavily(payload={"results": []})
    make_tool(fake2).invoke({"query": "x", "max_results": 99})
    assert fake2.sent_json()["max_results"] == 10  # clamped to a sane ceiling
    fake3 = FakeTavily(payload={"results": []})
    make_tool(fake3).invoke({"query": "x", "max_results": 0})
    assert fake3.sent_json()["max_results"] == 1   # clamped up from zero


# ---- result formatting -----------------------------------------------------------

def test_formats_answer_then_numbered_sources():
    fake = FakeTavily(payload={
        "answer": "Rotonda West sees scattered storms this weekend.",
        "results": [
            {"title": "Weather.gov", "url": "https://weather.gov/x", "content": "Saturday: 30% rain, high 89."},
            {"title": "Weather Channel", "url": "https://weather.com/y", "content": "Sunday: sunny, high 91."},
        ],
    })
    out = make_tool(fake).invoke({"query": "Rotonda West weekend weather"})
    # the answer leads
    assert out.startswith("Rotonda West sees scattered storms this weekend.")
    # then the sources, numbered, pipe-delimited
    assert "Sources:" in out
    assert "1. Weather.gov | https://weather.gov/x | Saturday: 30% rain, high 89." in out
    assert "2. Weather Channel | https://weather.com/y | Sunday: sunny, high 91." in out


def test_formats_results_without_an_answer():
    fake = FakeTavily(payload={
        "results": [{"title": "Only Hit", "url": "https://a.test", "content": "some text"}],
    })
    out = make_tool(fake).invoke({"query": "obscure thing"})
    assert "Sources:" in out
    assert "1. Only Hit | https://a.test | some text" in out


def test_snippet_is_truncated():
    long = "z" * (SNIPPET_MAX_CHARS + 200)
    fake = FakeTavily(payload={
        "results": [{"title": "T", "url": "https://a.test", "content": long}],
    })
    out = make_tool(fake).invoke({"query": "q"})
    assert "z" * (SNIPPET_MAX_CHARS + 1) not in out  # the full snippet never rides
    assert "…" in out                                # truncation is visible


def test_answer_only_no_results_still_returns_the_answer():
    fake = FakeTavily(payload={"answer": "Yes.", "results": []})
    out = make_tool(fake).invoke({"query": "is water wet"})
    assert out == "Yes."


# ---- honest failure --------------------------------------------------------------

def test_empty_query_is_honest_string():
    fake = FakeTavily(payload={"results": []})
    assert "nothing to look up" in make_tool(fake).invoke({"query": "   "})
    assert fake.requests == []  # a blank query never spends a Tavily call


def test_no_results_is_honest_string():
    fake = FakeTavily(payload={"answer": "", "results": []})
    out = make_tool(fake).invoke({"query": "asdkjfhaskjdfh no such thing"})
    assert "no results" in out


def test_http_500_is_honest_string_not_exception():
    out = make_tool(FakeTavily(status=500)).invoke({"query": "q"})
    assert out.startswith("web search failed:") and "500" in out


def test_timeout_is_honest_string_not_exception():
    def boom(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectTimeout("slow")

    tool = build_web_search_tool(
        api_key="tvly-test", client=httpx.Client(transport=httpx.MockTransport(boom))
    )
    out = tool.invoke({"query": "q"})
    assert out.startswith("web search failed:") and "unreachable" in out


def test_malformed_json_is_honest_string():
    def bad(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"not json at all")

    tool = build_web_search_tool(
        api_key="tvly-test", client=httpx.Client(transport=httpx.MockTransport(bad))
    )
    out = tool.invoke({"query": "q"})
    assert out.startswith("web search failed:") and "malformed" in out


def test_blank_key_is_rejected_at_construction():
    with pytest.raises(ValueError):
        build_web_search_tool(api_key="")
    with pytest.raises(ValueError):
        build_web_search_tool(api_key="   ")


# ---- name/description contract (the V1 tool-name-mismatch guard) ------------------

def test_tool_name_matches_prompt_reference():
    # the @tool function name MUST be search_web — factory.SEARCH_OVERLAY tells
    # the model to call "search_web" by that exact name (the V1 toolWorkflow
    # name-mismatch bug, kept dead).
    tool = build_web_search_tool(api_key="tvly-test")
    assert tool.name == "search_web"
    # concrete trigger patterns ride the description (specificity beats generality)
    lowered = tool.description.lower()
    assert "current events" in lowered
    assert "search for" in lowered
