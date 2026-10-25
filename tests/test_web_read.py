"""Offline tests for read_web_page — the page-reading tool, fake Firecrawl via MockTransport.

Owner-approved 2026-10-25 ("both approved"): search_web only returns snippets, so
she could find a page but never read it — a recipe, an article, a link someone
pasted. read_web_page hands the URL to Firecrawl's hosted scrape and returns the
page's main text as markdown. What these prove: the documented request shape, a
compact honest reply (title, source, text, a cut marker when trimmed), every
failure is a string (never a raise — the ToolNode contract), links that belong to
other tools are routed to them, and nothing on the home network is ever sent to
an outside service.
"""

import json

import httpx
import pytest

from aerys_v2.tools.web_read import (
    FIRECRAWL_SCRAPE_URL,
    PAGE_MAX_CHARS,
    build_read_web_page_tool,
)


class FakeFirecrawl:
    def __init__(self, payload=None, status: int = 200, exc: Exception | None = None):
        self.requests: list[httpx.Request] = []
        self.payload = payload
        self.status = status
        self.exc = exc

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if self.exc is not None:
            raise self.exc
        if self.status >= 400:
            return httpx.Response(self.status, json={"success": False, "error": "nope"})
        return httpx.Response(200, json=self.payload)

    def client(self) -> httpx.Client:
        return httpx.Client(transport=httpx.MockTransport(self.handler))


def ok_payload(markdown="# Lemon bars\n\n1. Heat the oven to 350F.", title="Lemon bars", url="https://example.com/lemon"):
    return {"success": True, "data": {"markdown": markdown, "metadata": {"title": title, "sourceURL": url}}}


def make_tool(fake: FakeFirecrawl):
    return build_read_web_page_tool(api_key="fc-test", client=fake.client())


def read(fake: FakeFirecrawl, url: str) -> str:
    return make_tool(fake).invoke({"url": url})


def test_tool_name_matches_the_overlay():
    assert make_tool(FakeFirecrawl(ok_payload())).name == "read_web_page"


def test_request_hits_firecrawl_scrape_with_documented_body():
    fake = FakeFirecrawl(ok_payload())
    read(fake, "https://example.com/lemon")
    req = fake.requests[0]
    assert str(req.url) == FIRECRAWL_SCRAPE_URL
    assert req.headers["authorization"] == "Bearer fc-test"
    body = json.loads(req.content)
    assert body["url"] == "https://example.com/lemon"
    assert body["formats"] == ["markdown"]
    assert body["onlyMainContent"] is True


def test_reply_carries_title_source_and_text_marked_as_page_content():
    out = read(FakeFirecrawl(ok_payload()), "https://example.com/lemon")
    assert "Lemon bars" in out and "https://example.com/lemon" in out
    assert "Heat the oven to 350F" in out
    # Page text is content to read, never instructions to follow (prompt-injection guard).
    assert "not instructions" in out.lower()


def test_long_pages_are_cut_and_say_so():
    long_md = "word " * (PAGE_MAX_CHARS // 2)
    out = read(FakeFirecrawl(ok_payload(markdown=long_md)), "https://example.com/long")
    assert len(out) < PAGE_MAX_CHARS + 600
    assert "cut" in out.lower() or "continues" in out.lower()


@pytest.mark.parametrize("status", [401, 402, 429, 500])
def test_http_errors_come_back_as_honest_strings(status):
    out = read(FakeFirecrawl(status=status), "https://example.com/x")
    assert out.lower().startswith("reading the page failed")


def test_success_false_is_an_honest_failure():
    out = read(FakeFirecrawl({"success": False, "error": "blocked"}), "https://example.com/x")
    assert out.lower().startswith("reading the page failed")


def test_timeout_is_an_honest_failure_not_a_raise():
    out = read(FakeFirecrawl(exc=httpx.ReadTimeout("slow")), "https://example.com/x")
    assert out.lower().startswith("reading the page failed")


def test_empty_page_says_there_was_nothing_to_read():
    out = read(FakeFirecrawl(ok_payload(markdown="   ")), "https://example.com/empty")
    assert "no readable text" in out.lower()


@pytest.mark.parametrize("url", ["ftp://example.com/a", "file:///etc/passwd", "javascript:alert(1)", "example.com/no-scheme"])
def test_only_http_links_are_read(url):
    fake = FakeFirecrawl(ok_payload())
    out = read(fake, url)
    assert fake.requests == [], "nothing may be sent for a non-web link"
    assert "http" in out.lower()


@pytest.mark.parametrize("url", [
    "http://192.168.1.155:8123/lovelace",
    "http://10.0.0.5/",
    "http://172.16.4.2/",
    "http://127.0.0.1:8300/health",
    "http://localhost:5678/",
    "http://nas.local/share",
    "http://[::1]/",
    "http://169.254.169.254/latest/meta-data",
    "http://homeassistant:8123/",
    "http://jetson/",
])
def test_home_network_addresses_are_never_sent_out(url):
    fake = FakeFirecrawl(ok_payload())
    out = read(fake, url)
    assert fake.requests == [], f"{url} must never reach an outside service"
    assert "home network" in out.lower() or "private" in out.lower()


def test_document_links_go_to_read_document():
    fake = FakeFirecrawl(ok_payload())
    out = read(fake, "https://example.com/menu.pdf?v=2")
    assert fake.requests == []
    assert "read_document" in out


def test_youtube_links_go_to_youtube_summary():
    fake = FakeFirecrawl(ok_payload())
    out = read(fake, "https://www.youtube.com/watch?v=dQw4w9WgXcQ")
    assert fake.requests == []
    assert "youtube_summary" in out


def test_blank_key_is_caught_at_construction():
    with pytest.raises(ValueError):
        build_read_web_page_tool(api_key="  ")


# ---- wiring -------------------------------------------------------------------

def _settings(**kw):
    from aerys_v2.config import Settings
    return Settings(_env_file=None, anthropic_api_key="sk-test", **kw)


def test_factory_arms_read_web_page_only_with_a_firecrawl_key():
    from aerys_v2.factory import action_tools_for
    assert "read_web_page" not in {t.name for t in action_tools_for(_settings())}
    armed = _settings(firecrawl_api_key="fc-test")
    assert "read_web_page" in {t.name for t in action_tools_for(armed)}


def test_guests_can_read_pages_too():
    # Public information, like search_web (owner decision 2026-07-05: web search is
    # available to everyone, guest included). Only the HOME half stays allowlisted.
    from aerys_v2.factory import action_tools_for
    armed = _settings(firecrawl_api_key="fc-test")
    assert "read_web_page" in {t.name for t in action_tools_for(armed, guest=True)}


def test_overlay_teaches_the_tool_by_name_when_armed():
    from aerys_v2.factory import action_overlay_for
    assert "read_web_page" not in action_overlay_for(_settings())
    assert "read_web_page" in action_overlay_for(_settings(firecrawl_api_key="fc-test"))
