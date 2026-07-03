"""web_search — the WEB SEARCH TOOL: current-events lookup via Tavily.

n8n mapping: this file replaces V1's `tavilyTool` community node
(@tavily/n8n-nodes-tavily.tavilyTool, credentials iZxeoPSLwObXXEGN /
PRAECj0Em1imOqmW) that hung off the research sub-agent's AI Agent by an ai_tool
connection. There the model got a single `query` param and n8n's node made the
Tavily call; here the builder closes over the API key and an injectable
httpx.Client (the home_control / media seam philosophy) and returns a LangChain
tool the action graph binds.

Contracts every tool here obeys (same as tools/home_control.py and tools/media.py):

1. READ-ONLY — a web search changes nothing outside the conversation. No outbox
   row, nothing to audit; it never touches conn_factory.
2. HONEST FAILURE — every error path returns a plain string the model must
   relay. NEVER raise out of a tool: an exception inside a ToolNode kills the
   whole turn (the V1 failed-webhook-kills-execution outage mode). A dead
   Tavily, a 500, a timeout — all come back as "web search failed: <reason>".
3. NEVER FABRICATE — the prompt overlay (factory.SEARCH_OVERLAY) tells the model
   to ground its answer in what this tool returns and never invent results; the
   tool's own job is to hand back the real results (or an honest "no results")
   so there is something true to ground on.
"""

import logging

import httpx
from langchain_core.tools import tool

log = logging.getLogger(__name__)

# Tavily's public search endpoint. One host, one path — no OpenAI-compat base_url
# dance here (Tavily is its own API, not an OpenAI-shaped one).
TAVILY_SEARCH_URL = "https://api.tavily.com/search"

# Snippet cap per result — a search result rides inside the model's context
# window, so ten verbose page dumps must not blow the turn. Truncation is silent
# per-result (the whole compact block says "here are the top hits", not "the full
# page"); the model follows up with read_document if it needs the full text.
SNIPPET_MAX_CHARS = 300

# A search request must fail the turn rather than hang the caller (the same
# safety-rail reasoning as build_model's request timeout). 20s is generous for
# basic depth; a slow Tavily becomes an honest error, not a stall.
SEARCH_TIMEOUT_S = 20.0


def _truncate(text: str, limit: int = SNIPPET_MAX_CHARS) -> str:
    text = (text or "").strip().replace("\n", " ")
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "…"


def build_web_search_tool(*, api_key: str, client: httpx.Client | None = None):
    """Close over the Tavily API key and return the search tool.

    Everything injectable, same seam as build_home_control_tool /
    build_analyze_image_tool: tests pass an httpx.Client on a MockTransport; the
    factory passes settings.tavily_api_key. The tool NEVER reads Settings —
    construction knows config, behavior doesn't.
    """
    if not api_key or not api_key.strip():
        # Construction-time honesty, mirroring build_auth_headers in the vision
        # ladder: a blank key is a wiring bug, caught here, not at first call.
        raise ValueError("build_web_search_tool needs a non-empty Tavily API key")
    http = client or httpx.Client(timeout=SEARCH_TIMEOUT_S)

    @tool
    def search_web(query: str, max_results: int = 5) -> str:
        """Search the live web for current, up-to-the-minute information.

        CALL THIS TOOL IMMEDIATELY when the answer depends on anything you cannot
        know from training alone:
        - current events, breaking news, "what happened with..."
        - today's weather, forecasts, scores, prices, stock quotes, exchange rates
        - the user says "search for", "look up", "google", "find out", "what's the
          latest on..."
        - ANY fact that could have changed after your knowledge cutoff, or that
          you are not certain about.

        You have NO live web access without this tool. Do not answer current-events
        or lookup questions from memory — call this tool and ground your answer in
        what it returns.

        query: what to search for, as a natural search query.
        max_results: how many results to pull (default 5).

        Returns: a short answer (when available) plus the top results as
        `title | url | snippet`. Never fabricate results — use only what comes back.
        """
        q = query.strip() if isinstance(query, str) else ""
        if not q:
            return "search_web needs a search query — there is nothing to look up."

        # Clamp the count defensively: a model can hand back anything, and Tavily
        # bills per result. 1..10 is plenty for grounding one reply.
        try:
            count = int(max_results)
        except (TypeError, ValueError):
            count = 5
        count = max(1, min(count, 10))

        body = {
            "api_key": api_key,
            "query": q,
            "max_results": count,
            "search_depth": "basic",
            "include_answer": True,
        }
        try:
            r = http.post(TAVILY_SEARCH_URL, json=body)
        except httpx.HTTPError as e:
            # timeout, connect error, DNS — all honest words, never a raise.
            return f"web search failed: the search service is unreachable right now ({e})."
        if r.status_code >= 400:
            return f"web search failed: the search service returned HTTP {r.status_code}."
        try:
            data = r.json()
        except ValueError:
            return "web search failed: the search service returned a malformed response."

        answer = str(data.get("answer") or "").strip()
        results = data.get("results") or []

        lines: list[str] = []
        if answer:
            lines.append(answer)
        if isinstance(results, list) and results:
            if answer:
                lines.append("")  # blank line between the answer and the sources
            lines.append("Sources:")
            for i, res in enumerate(results, start=1):
                if not isinstance(res, dict):
                    continue
                title = str(res.get("title") or "untitled").strip()
                url = str(res.get("url") or "").strip()
                snippet = _truncate(str(res.get("content") or ""))
                lines.append(f"{i}. {title} | {url} | {snippet}")

        if not lines:
            return f"web search for {q!r} returned no results."
        return "\n".join(lines)

    return search_web
