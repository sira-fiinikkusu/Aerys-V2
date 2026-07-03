"""Media-type detection — the "Detect Media Type" Code node ported to pure functions.

n8n mapping: this is the classifier from workflow nqTeANyy5jlZgWJI ("05-01 Media Agent").
The live per-tier agents (Sonnet lGjy9sHqbwOh7J50, Gemini PDpiLfZCXqEGGbiD) replaced the
switch-based routing with LLM tool-calling (Tool: Image / PDF / DOCX / Text File /
YouTube), but the detection RULES below are still the canonical logic for "what kind of
media is this message carrying?". In V2 the transport layer calls `detect_media()` and
routes on the result — same job, but now it runs in <1ms with a test suite instead of
inside a sandboxed Code node you can only debug by sending yourself Discord messages.

Quirk rules that MUST survive (each has a golden test):

1. NEVER strip query params from media URLs. Discord CDN URLs carry `?ex=&is=&hm=`
   signature params that expire — stripping them 404s the download. `split('?')[0]` is
   applied ONLY when deriving a `filename` for display, never to the URL itself.
2. Guild-adapter attachments store the MIME type under `type`, not `content_type`
   (Telegram/others use `content_type` / `mime_type`). The coalesce order is
   content_type → mime_type → type.
3. In the query-URL fallback, document extensions (.pdf/.docx/.txt) are checked BEFORE
   the image catch-all. The Core Agent passes attachments as a bare CDN URL string in
   `query`, and every Discord CDN URL matches the image catch-all — without the
   ordering, all attachments route to the image branch.
4. `attachments` may arrive as a JSON string (toolWorkflow callers serialize it) or as
   outright garbage — parse defensively, fall back to [].
"""

import json
import re
from typing import Any

# The 11-char YouTube video ID, from either the long or short link form.
# Same pattern the Code node used: youtube.com/watch?v=<id> OR youtu.be/<id>.
YOUTUBE_RE = re.compile(r"(?:youtube\.com/watch\?v=|youtu\.be/)([a-zA-Z0-9_-]{11})")

# First http(s) URL in a string. \S+ keeps EVERYTHING up to whitespace — including the
# ?ex=&is=&hm= signature params (quirk rule 1: the URL is never truncated).
URL_RE = re.compile(r"(https?://\S+)")

# Image detection by filename extension — used when the MIME type is missing.
IMAGE_EXT_RE = re.compile(r"\.(jpg|jpeg|png|webp|gif)$", re.IGNORECASE)

# Query-URL branch patterns. `(\?|$)` means "extension at end of URL OR followed by a
# query string" — signed CDN URLs look like .../file.pdf?ex=...&hm=..., so a plain
# end-anchor would miss them.
QUERY_PDF_RE = re.compile(r"\.pdf(\?|$)", re.IGNORECASE)
QUERY_DOCX_RE = re.compile(r"\.docx(\?|$)", re.IGNORECASE)
QUERY_TXT_RE = re.compile(r"\.txt(\?|$)", re.IGNORECASE)
QUERY_IMAGE_RE = re.compile(r"\.(jpg|jpeg|png|gif|webp)(\?|$)", re.IGNORECASE)
# Image catch-all: ANY Discord CDN attachment URL that didn't match a document
# extension above is treated as an image (quirk rule 3 — order matters).
DISCORD_CDN_RE = re.compile(r"cdn\.discordapp\.com/attachments/", re.IGNORECASE)

DOCX_MIME = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"


def parse_attachments(raw: Any) -> list[dict]:
    """Normalize the `attachments` field to a list of dicts, no matter what arrived.

    n8n mapping: toolWorkflow callers serialize workflowInputs, so `attachments` often
    arrives as a JSON STRING instead of an array (quirk rule 4). The Code node did
    `JSON.parse` in a try/catch and an `Array.isArray` check; this is the same ladder.
    Non-dict entries are dropped to {} so a malformed element can't crash detection
    (in JS, `att.content_type` on a string is just `undefined` — we match that).
    """
    attachments = raw if raw is not None else []
    if isinstance(attachments, str):
        try:
            attachments = json.loads(attachments)
        except (json.JSONDecodeError, ValueError):
            attachments = []
    if not isinstance(attachments, list):
        attachments = []
    return [a if isinstance(a, dict) else {} for a in attachments]


def _mime_of(att: dict) -> str:
    """Coalesce the attachment MIME type across adapter dialects (quirk rule 2).

    n8n mapping: `(att.content_type || att.mime_type || att.type || '').toLowerCase()`.
    The guild adapter stores it under `type`; Telegram and the DM adapter use
    `content_type` / `mime_type`. JS `||` skips falsy values (empty string), so we use
    `or`-chaining on .get() the same way.
    """
    return str(att.get("content_type") or att.get("mime_type") or att.get("type") or "").lower()


def _fname_of(att: dict) -> str:
    """Lowercased filename for extension fallback: `filename || file_name || ''`."""
    return str(att.get("filename") or att.get("file_name") or "").lower()


def _is_image_attachment(att: dict) -> bool:
    """Image if the MIME says so, or the filename extension does when MIME is missing."""
    return _mime_of(att).startswith("image/") or bool(IMAGE_EXT_RE.search(_fname_of(att)))


def _url_of(att: dict) -> str | None:
    """URL preference: `att.url || att.proxy_url` — full signed URL, never trimmed."""
    return att.get("url") or att.get("proxy_url") or None


def find_youtube_id(text: str) -> str | None:
    """Extract an 11-char YouTube video ID from free text, or None.

    Handles both youtube.com/watch?v=<id> and youtu.be/<id> short links.
    """
    match = YOUTUBE_RE.search(text or "")
    return match.group(1) if match else None


def filename_from_url(url: str, default: str) -> str:
    """Derive a display filename from a URL's last path segment.

    n8n mapping: `(mediaUrl.split('/').pop() || 'document').split('?')[0]`. This is the
    ONE place `split('?')` is allowed — it trims signature params off the display NAME,
    never off the URL we fetch with (quirk rule 1).
    """
    return (url.split("/")[-1] or default).split("?")[0]


def _empty_result(payload: dict, content: str, platform: str, context: str) -> dict:
    """The output shape with everything unset — mirrors the Code node's `let` block.

    n8n mapping: the Code node initialized every field up front so the returned item
    always had a stable shape for downstream Switch/IF nodes. Same contract here:
    every key present on every call, `unknown` routes to the graceful fallback branch.
    """
    return {
        "mediaType": "unknown",
        "mediaUrl": None,
        "mediaUrls": [],  # multi-image support (Discord allows several per message)
        "videoId": None,
        "filename": None,
        "needsTelegramDownload": False,
        "telegramFileId": None,
        "fileMimeType": None,
        "platform": platform,
        "content": content,
        "context": context,
        # Passthrough identity/routing fields — sub-workflows lose upstream context
        # (the n8n pairedItem quirk), so the classifier re-emits them explicitly.
        "person_id": payload.get("person_id"),
        "source_channel": payload.get("source_channel"),
        "conversation_privacy": payload.get("conversation_privacy"),
    }


def _classify_attachment(result: dict, attachments: list[dict], platform: str) -> None:
    """Step 1: classify by attachments[0] ONLY — image → pdf → docx → txt.

    Multiple mixed attachments are typed by the FIRST one (matching the Code node),
    but the image branch still collects ALL image URLs into mediaUrls so a
    two-screenshot message reaches the vision API whole.
    """
    att = attachments[0]
    ct = _mime_of(att)
    fname = _fname_of(att)

    if _is_image_attachment(att):
        result["mediaType"] = "image"
        if platform == "telegram" or att.get("file_id"):
            # Telegram files aren't public URLs — downstream must call getFile with
            # the file_id first. Flag it and hand over what we know.
            result["needsTelegramDownload"] = True
            result["telegramFileId"] = att.get("file_id")
            result["fileMimeType"] = ct or "image/jpeg"  # Telegram photos omit MIME
            result["mediaUrl"] = _url_of(att)
        else:
            # Discord: the signed CDN URL is directly fetchable — pass it UNTOUCHED
            # (quirk rule 1). Collect every image attachment for multi-image support;
            # non-image attachments (e.g. a PDF riding along) are skipped here.
            result["mediaUrl"] = _url_of(att)
            result["mediaUrls"] = [
                url for a in attachments if _is_image_attachment(a) and (url := _url_of(a))
            ]
        result["filename"] = att.get("filename") or att.get("file_name") or "image"
    elif ct == "application/pdf" or fname.endswith(".pdf"):
        result["mediaType"] = "pdf"
        result["mediaUrl"] = _url_of(att)
        result["filename"] = att.get("filename") or att.get("file_name") or "document.pdf"
    elif ct == DOCX_MIME or fname.endswith(".docx"):
        result["mediaType"] = "docx"
        result["mediaUrl"] = _url_of(att)
        result["filename"] = att.get("filename") or att.get("file_name") or "document.docx"
    elif ct == "text/plain" or fname.endswith(".txt"):
        result["mediaType"] = "txt"
        result["mediaUrl"] = _url_of(att)
        result["filename"] = att.get("filename") or att.get("file_name") or "document.txt"
    # Anything else (audio/mpeg, video/mp4, ...) stays 'unknown' and detection falls
    # through to the text scans — exactly like the Code node's if-ladder.


def _classify_query_url(result: dict, query: str) -> None:
    """Step 3: the bare-URL fallback for `query`.

    n8n mapping: the Core Agent's toolWorkflow calls pass attachments as a CDN URL
    STRING inside `query` (the LangChain tool collapses inputs to {query: "..."}), so
    there's no attachments array to inspect. Order is load-bearing (quirk rule 3):
    YouTube → .pdf → .docx → .txt → image catch-all. Every Discord CDN URL matches the
    catch-all, so documents must be recognized first or they all route to vision.
    """
    video_id = find_youtube_id(query)
    if video_id:
        result["mediaType"] = "youtube"
        result["videoId"] = video_id
        return

    for media_type, pattern, default in (
        ("pdf", QUERY_PDF_RE, "document"),
        ("docx", QUERY_DOCX_RE, "document"),
        ("txt", QUERY_TXT_RE, "document"),
    ):
        if pattern.search(query):
            result["mediaType"] = media_type
            url_match = URL_RE.search(query)
            # Full URL with signature params intact; the raw query if no URL found.
            result["mediaUrl"] = url_match.group(1) if url_match else query
            result["filename"] = filename_from_url(result["mediaUrl"], default)
            return

    if QUERY_IMAGE_RE.search(query) or DISCORD_CDN_RE.search(query):
        # Image catch-all — reached ONLY after the document extensions missed.
        result["mediaType"] = "image"
        url_match = URL_RE.search(query)
        result["mediaUrl"] = url_match.group(1) if url_match else query
        result["mediaUrls"] = [result["mediaUrl"]]
        result["filename"] = filename_from_url(result["mediaUrl"], "image")


def detect_media(payload: dict) -> dict:
    """Classify what media (if any) a message payload carries. Pure — no I/O.

    n8n mapping: the whole "Detect Media Type" Code node. Input is the Execute
    Workflow Trigger item JSON; output is the single item the Switch node routed on.

    Detection order (preserved exactly from the Code node):
      1. attachments[0] — image / pdf / docx / txt, MIME first, extension fallback
      2. YouTube ID scan across content → context → message_content
      3. bare-URL fallback on `query` — youtube, then documents, THEN image catch-all

    'unknown' is a valid answer, not an error — downstream routes it to a graceful
    "I couldn't tell what that file is" branch.
    """
    attachments = parse_attachments(payload.get("attachments"))

    # JS `input.content || input.query || ''` — content falls back to query when the
    # caller only sent a query string (the toolWorkflow collapse case).
    content = payload.get("content") or payload.get("query") or ""
    platform = payload.get("platform") or "discord"
    context = payload.get("context") or ""
    message_content = payload.get("message_content") or ""

    result = _empty_result(payload, content, platform, context)

    # --- Step 1: attachments array wins when present -----------------------------
    if attachments:
        _classify_attachment(result, attachments, platform)

    # --- Step 2: YouTube URL scan across the text fields, first hit wins ---------
    # (content first, then context, then message_content — same order as the node)
    if result["mediaType"] == "unknown":
        for field in (content, context, message_content):
            video_id = find_youtube_id(field)
            if video_id:
                result["mediaType"] = "youtube"
                result["videoId"] = video_id
                break

    # --- Step 3: bare file-URL fallback in `query` --------------------------------
    if result["mediaType"] == "unknown" and payload.get("query"):
        _classify_query_url(result, str(payload["query"]))

    return result
