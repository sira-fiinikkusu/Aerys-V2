"""media — the MEDIA TOOLS: vision, document reading, YouTube summaries.

n8n mapping: this file replaces the whole 06-05 media stack — the `Tool: Image`
httpRequestTool node (vision via OpenRouter) plus the three extractor
sub-workflows: YouTube (HE7zmxKeWoxjvM9L), PDF (yuHzxHqqWz93xwYj) and DOCX
(tJwLt494G1VugToU). The pure logic already lives in channels/media_detect.py and
channels/vision_ladder.py; this module is the LIVE half — the builders close
over config and an injectable httpx.Client (the home_control seam philosophy)
and return LangChain tools the action/chat graphs can bind.

Contracts every tool here obeys (same as tools/home_control.py):

1. READ-ONLY — nothing in this file changes any external state. No outbox
   needed: there is no write to audit.
2. HONEST FAILURE — every error path returns a plain string the model must
   relay. NEVER raise out of a tool: an exception inside a ToolNode kills the
   whole turn (the V1 failed-webhook-kills-execution outage mode).
3. URL SANCTITY — signed Discord CDN URLs (`?ex=&is=&hm=`) pass through
   VERBATIM to the vision API and to fetches. Never `split('?')[0]`, never
   download-and-re-encode for vision (the proven V1 pattern is handing the
   signed URL straight to the API; the n8n sandbox couldn't even read binary).
4. EXTENSIONS BEFORE IMAGE — routing reuses detect_media_kind(), where
   .pdf/.docx/.txt are checked before the image catch-all. Flipping that order
   sent every attachment to vision in V1 (CLAUDE.md quirk).
"""

import logging
from io import BytesIO
from typing import Callable

import httpx
from langchain_core.tools import tool

from aerys_v2.channels.media_detect import filename_from_url, find_youtube_id
from aerys_v2.channels.vision_ladder import (
    VISION_MODEL,
    build_auth_headers,
    build_vision_body,
    detect_media_kind,
    explain_vision_http_error,
    extract_description,
    looks_like_signed_discord_url,
)

log = logging.getLogger(__name__)

# Default OpenAI-compatible host. The live wiring passes Settings values instead:
# embeddings_api_key + embeddings_base_url — the SAME OpenRouter key/base the
# memory embedder uses (one credential, two OpenAI-compatible endpoints), exactly
# like n8n credential gvgPllzFhLSds5Qv served both jobs.
OPENROUTER_BASE = "https://openrouter.ai/api/v1"

# Document text cap — a tool message rides inside the model's context window, so
# a 300-page PDF must not blow the turn. Truncation says so honestly (contract 2).
DOC_MAX_CHARS = 8_000

# Transcript cap before summarization — same context-budget reasoning. ~12k chars
# is roughly 30-40 minutes of speech; longer videos summarize from the front.
TRANSCRIPT_MAX_CHARS = 12_000

# Summaries run on the same model as vision (one constant, no ladder — the
# vision_ladder module docstring explains why tier fallback lives in the router,
# never here). Undated slug: OpenRouter rejects the dated ID format.
SUMMARY_MODEL = VISION_MODEL
SUMMARY_MAX_TOKENS = 1024

# The transcript seam: video_id -> plain transcript text. Injectable like the
# httpx.Client — tests pass a lambda, prod uses _default_transcript_fetcher
# (youtube-transcript-api: caption download only, no yt-dlp, no video bytes).
TranscriptFetcher = Callable[[str], str]


def _default_transcript_fetcher(video_id: str) -> str:
    """Fetch the caption transcript via youtube-transcript-api (no downloads).

    n8n mapping: the YouTube extractor sub-workflow's transcript HTTP call. The
    import is deferred so tests (which always inject a fake) never need the
    package's network-touching machinery loaded.
    """
    from youtube_transcript_api import YouTubeTranscriptApi

    fetched = YouTubeTranscriptApi().fetch(video_id)
    return " ".join(snippet.text for snippet in fetched)


# ---------------------------------------------------------------------------
# analyze_image — the Tool: Image node, live.
# ---------------------------------------------------------------------------


def build_analyze_image_tool(
    *,
    api_key: str,
    base_url: str = OPENROUTER_BASE,
    client: httpx.Client | None = None,
):
    """Close over the OpenRouter config and return the vision tool.

    Everything injectable, same seam as build_home_control_tool: tests pass an
    httpx.Client on a MockTransport; the factory passes Settings values. The
    tool NEVER reads Settings — construction knows config, behavior doesn't.
    """
    chat_url = f"{base_url.rstrip('/')}/chat/completions"
    headers = build_auth_headers(api_key)  # raises at CONSTRUCTION on a blank key
    http = client or httpx.Client(timeout=60.0)

    @tool
    def analyze_image(url: str, question: str = "") -> str:
        """Analyze an image. CALL THIS TOOL IMMEDIATELY when:
        - A Discord CDN URL appears (https://cdn.discordapp.com/attachments/...)
        - The user asks you to look at, describe, or analyze an image
        - Any image URL is shared in conversation

        You have ZERO visual capability without this tool. You cannot see
        images directly. Even if you think you know what the image contains,
        you MUST call this tool.

        url: the image URL EXACTLY as it appeared — keep every query parameter
        (Discord CDN URLs are signed; trimming them breaks the link).
        question: what to find out about the image; empty means "describe it".

        Returns: a description and analysis of the image.
        """
        target = url.strip() if isinstance(url, str) else ""
        if not target:
            return "analyze_image needs an image URL — there is nothing to look at."

        # build_vision_body passes the URL through VERBATIM (contract 3) and
        # promotes an empty question to the default describe prompt.
        body = build_vision_body(target, question)
        try:
            r = http.post(chat_url, headers=headers, json=body)
        except httpx.HTTPError as e:
            return f"The vision service is unreachable right now ({e})."
        if r.status_code >= 400:
            # 403/404 on a signed CDN URL gets the expired-signature explanation
            # instead of a bare status the model would just retry into.
            return explain_vision_http_error(r.status_code, target)
        try:
            return extract_description(r.json())
        except ValueError as e:
            # extract_description fails LOUDLY on malformed payloads — but a
            # tool must relay, not raise (contract 2).
            return f"The vision call came back malformed: {e}"

    return analyze_image


# ---------------------------------------------------------------------------
# read_document — the PDF / DOCX / TXT extractor sub-workflows, live.
# ---------------------------------------------------------------------------


def _pdf_text(data: bytes) -> str:
    """Extract text from PDF bytes — the PDF extractor sub-workflow's job."""
    from pypdf import PdfReader

    reader = PdfReader(BytesIO(data))
    return "\n\n".join((page.extract_text() or "") for page in reader.pages)


def _docx_text(data: bytes) -> str:
    """Extract text from DOCX bytes.

    n8n mapping: the @mazix converter community node, whose output hid at
    `$json.files[0].text` (CLAUDE.md quirk). python-docx replaces it: paragraphs
    joined by newlines — same plain-text shape that node produced.
    """
    from docx import Document

    doc = Document(BytesIO(data))
    return "\n".join(paragraph.text for paragraph in doc.paragraphs)


def build_read_document_tool(*, client: httpx.Client | None = None):
    """Return the document-reading tool. No credentials — it only fetches URLs."""
    http = client or httpx.Client(timeout=30.0, follow_redirects=True)

    @tool
    def read_document(url: str) -> str:
        """Read the text out of a document file (.pdf, .docx, or .txt).

        CALL THIS TOOL IMMEDIATELY when a URL ending in .pdf, .docx, or .txt
        appears in the conversation (including Discord CDN attachment URLs like
        https://cdn.discordapp.com/attachments/...file.pdf?ex=...). You cannot
        read file contents without this tool.

        url: the document URL EXACTLY as it appeared — keep every query
        parameter (Discord CDN URLs are signed; trimming them breaks the link).

        Returns: the document's extracted text (long documents are truncated).
        """
        target = url.strip() if isinstance(url, str) else ""
        if not target:
            return "read_document needs a document URL — there is nothing to read."

        # Routing honesty: YouTube links and images have their own tools. The
        # kind check reuses detect_media_kind, where document extensions are
        # matched BEFORE the image catch-all (contract 4) — and the extension is
        # read from the URL *path*, so `report.pdf?ex=...` still says pdf.
        if find_youtube_id(target):
            return (
                "That is a YouTube link, not a document — call the "
                "youtube_summary tool instead."
            )
        kind = detect_media_kind(target)
        if kind == "image":
            return (
                f"{filename_from_url(target, 'That URL')} does not look like a "
                "document (.pdf/.docx/.txt) — for an image, call the "
                "analyze_image tool instead."
            )

        # The fetch uses the FULL URL, signature params intact (contract 3).
        try:
            r = http.get(target)
        except httpx.HTTPError as e:
            return f"Couldn't fetch the document ({e})."
        if r.status_code >= 400:
            if r.status_code in (403, 404) and looks_like_signed_discord_url(target):
                return (
                    f"Fetching the document failed with HTTP {r.status_code}: the "
                    "Discord CDN signature on this attachment URL has likely "
                    "expired (signed URLs carry ?ex=&is=&hm= params with a short "
                    "lifetime). Re-fetch the attachment to get a fresh URL — and "
                    "never strip the query string."
                )
            return f"Fetching the document failed with HTTP {r.status_code}."

        filename = filename_from_url(target, "document")
        try:
            if kind == "pdf":
                text = _pdf_text(r.content)
            elif kind == "docx":
                text = _docx_text(r.content)
            else:  # text_file — decode defensively, mojibake beats a dead turn
                text = r.content.decode("utf-8", errors="replace")
        except Exception as e:
            # Corrupt files must come back as words, not a raise (contract 2).
            log.warning("document extraction failed for %s", filename, exc_info=True)
            return f"Fetched {filename} but couldn't extract its text ({e})."

        text = text.strip()
        if not text:
            return (
                f"Fetched {filename} fine, but it contained no extractable text "
                "(it may be a scanned/image-only file)."
            )
        if len(text) > DOC_MAX_CHARS:
            text = (
                text[:DOC_MAX_CHARS]
                + f"\n\n[truncated at {DOC_MAX_CHARS} characters — the document continues]"
            )
        return f"Contents of {filename}:\n\n{text}"

    return read_document


# ---------------------------------------------------------------------------
# youtube_summary — the YouTube extractor sub-workflow, live.
# ---------------------------------------------------------------------------


def build_youtube_summary_tool(
    *,
    api_key: str,
    base_url: str = OPENROUTER_BASE,
    client: httpx.Client | None = None,
    transcript_fetcher: TranscriptFetcher | None = None,
):
    """Return the YouTube summary tool: caption transcript -> LLM summary.

    Transcript only — no yt-dlp, no video downloads (the V1 extractor made the
    same choice: captions are kilobytes, videos are gigabytes). The summarize
    call rides the SAME OpenRouter client/credential as the vision tool.
    """
    chat_url = f"{base_url.rstrip('/')}/chat/completions"
    headers = build_auth_headers(api_key)
    http = client or httpx.Client(timeout=60.0)
    fetch_transcript = transcript_fetcher or _default_transcript_fetcher

    @tool
    def youtube_summary(url: str) -> str:
        """Summarize a YouTube video from its transcript.

        CALL THIS TOOL IMMEDIATELY when a youtube.com/watch or youtu.be link
        appears in the conversation. You cannot watch videos — this tool reads
        the video's caption transcript and summarizes it for you.

        url: the YouTube URL as shared (long or short form both work).

        Returns: a summary of what the video covers.
        """
        video_id = find_youtube_id(url.strip() if isinstance(url, str) else "")
        if not video_id:
            return (
                "That doesn't look like a YouTube link — I need a "
                "youtube.com/watch?v=... or youtu.be/... URL."
            )

        # The fetcher raises on missing/disabled captions — relay, never raise
        # (contract 2). Broad catch on purpose: youtube-transcript-api throws a
        # zoo of exception types (TranscriptsDisabled, NoTranscriptFound, ...).
        try:
            transcript = fetch_transcript(video_id)
        except Exception as e:
            return (
                f"Couldn't get a transcript for video {video_id} ({e}). The video "
                "may have captions disabled — I can't summarize it without them."
            )
        transcript = (transcript or "").strip()
        if not transcript:
            return f"Video {video_id} has an empty transcript — nothing to summarize."

        truncated = len(transcript) > TRANSCRIPT_MAX_CHARS
        if truncated:
            transcript = transcript[:TRANSCRIPT_MAX_CHARS]

        prompt = (
            "Summarize this YouTube video transcript. Cover the main points, "
            "any conclusions, and anything actionable. Be concise."
            + (" (Transcript truncated — summarize what is here.)" if truncated else "")
            + f"\n\nTranscript:\n{transcript}"
        )
        body = {
            "model": SUMMARY_MODEL,
            "max_tokens": SUMMARY_MAX_TOKENS,
            "messages": [{"role": "user", "content": prompt}],
        }
        try:
            r = http.post(chat_url, headers=headers, json=body)
        except httpx.HTTPError as e:
            return f"The summarization service is unreachable right now ({e})."
        if r.status_code >= 400:
            return f"Summarizing the transcript failed with HTTP {r.status_code}."
        try:
            return extract_description(r.json())
        except ValueError as e:
            return f"The summarization call came back malformed: {e}"

    return youtube_summary


# ---------------------------------------------------------------------------
# Aggregate builder — what the factory registers.
# ---------------------------------------------------------------------------


def build_media_tools(
    *,
    api_key: str,
    base_url: str = OPENROUTER_BASE,
    vision_client: httpx.Client | None = None,
    doc_client: httpx.Client | None = None,
    transcript_fetcher: TranscriptFetcher | None = None,
) -> list:
    """All three media tools, wired from one OpenRouter credential.

    Factory wiring (the arming pattern): api_key comes from
    settings.embeddings_api_key.get_secret_value() and base_url from
    settings.embeddings_base_url — the memory embedder's OpenRouter credential,
    reused for chat completions. embeddings_api_key None = media tools simply
    don't exist, same as ha_token gating the home_control stack.
    """
    return [
        build_analyze_image_tool(api_key=api_key, base_url=base_url, client=vision_client),
        build_read_document_tool(client=doc_client),
        build_youtube_summary_tool(
            api_key=api_key,
            base_url=base_url,
            client=vision_client,
            transcript_fetcher=transcript_fetcher,
        ),
    ]
