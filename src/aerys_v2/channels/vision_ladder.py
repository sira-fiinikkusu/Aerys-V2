"""Vision tool logic — the `Tool: Image` httpRequestTool node as pure functions.

n8n mapping: in Aerys V1, image analysis is an `n8n-nodes-base.httpRequestTool`
(typeVersion 4.2) wired to each tier's AI Agent via an ai_tool connection. The whole
request build is one `jsonBody` expression evaluated at tool-call time, with the LLM
supplying `image_url` and `prompt` through `$fromAI(...)`. Here that expression becomes
`build_vision_body()`, the node's credential + URL + method become
`build_vision_request()`, and the upstream "Detect Media Type" Code node becomes
`detect_media_kind()`.

Despite the filename, there is deliberately NO model ladder in this module: all three
tier agents (Sonnet lGjy9sHqbwOh7J50, Opus nGKpHpzfZZ1XRSxV, Gemini PDpiLfZCXqEGGbiD)
carry byte-identical Tool: Image nodes. Vision is ALWAYS `VISION_MODEL` via OpenRouter
regardless of which conversation tier is running. The only fallback chain in V1 is at
the Core Agent router level (opus -> sonnet -> gemini), which swaps which AGENT
sub-workflow runs — never which vision model is called. So: one constant, no fallback
logic here; tier fallback stays in the router.

Everything in this file is PURE — no network, no filesystem. A transport layer takes
the dict from `build_vision_request()`, performs the actual POST, and feeds the parsed
JSON back through `extract_description()` / `explain_vision_http_error()`.
"""

from urllib.parse import urlsplit

# ---------------------------------------------------------------------------
# Constants — the parts of the n8n node that were configuration, not code.
# ---------------------------------------------------------------------------

# The tool's function name as the LLM sees it. CLAUDE.md quirk: the system prompt says
# "use the matching tool (image, youtube, pdf, ...)" — if the registered tool name
# doesn't match, the model can't find it and hallucinates having called it. In n8n this
# matching is fragile (the toolWorkflow `name` property is hidden at typeVersion 2.2);
# in Python we just name it correctly and pin it with a test.
TOOL_NAME = "image"

# Verbatim from the live node's toolDescription — load-bearing for reliable invocation.
# CLAUDE.md quirk: generic "call this for images" fails; the concrete
# `https://cdn.discordapp.com/attachments/` trigger string and the "ZERO visual
# capability" framing are what make the model actually call the tool. Do not soften.
TOOL_DESCRIPTION = """Analyze an image. CALL THIS TOOL IMMEDIATELY when:
- A Discord CDN URL appears (https://cdn.discordapp.com/attachments/...)
- The user asks you to look at, describe, or analyze an image
- Any image URL is shared in conversation

You have ZERO visual capability without this tool. You cannot see images directly. Even if you think you know what the image contains, you MUST call this tool.

Returns: A description and analysis of the image."""

# Must be the UNDATED slug. CLAUDE.md quirk: OpenRouter routes
# "anthropic/claude-sonnet-4.6" fine but the dated ID format fails. One constant,
# no ladder — see module docstring.
VISION_MODEL = "anthropic/claude-sonnet-4.6"

# The n8n node's `url` parameter. OpenAI-compatible chat completions endpoint.
OPENROUTER_CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"

# The n8n jsonBody's max_tokens — caps spend per image description.
VISION_MAX_TOKENS = 1024

# In n8n this string was only a HINT inside $fromAI('prompt', 'Describe this image in
# detail', 'string') — a description shown to the LLM, not an enforced default. The
# port promotes it to a real default so an empty prompt still produces a useful call.
DEFAULT_PROMPT = "Describe this image in detail"

# ---------------------------------------------------------------------------
# Media dispatch — the "Detect Media Type" node before the tool ever fires.
# ---------------------------------------------------------------------------

# Extension routing table. CLAUDE.md quirk: the Core Agent passes attachments as a bare
# CDN URL string, so Detect Media Type MUST check extensions BEFORE falling through to
# the image branch — otherwise every attachment routes to the image tool. Order here is
# expressed as "match these positively; image is the catch-all, never a positive match".
_EXTENSION_KINDS = {
    ".pdf": "pdf",
    ".docx": "docx",
    ".txt": "text_file",
}

# MIME fallback for URLs with no telling extension. CLAUDE.md quirk: the guild adapter
# stores the attachment's MIME under `type` (NOT `content_type`) — see attachment_mime().
_MIME_KINDS = {
    "application/pdf": "pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
    "text/plain": "text_file",
}


def attachment_mime(attachment: dict) -> str | None:
    """Read the MIME type off a guild-adapter attachment object.

    CLAUDE.md quirk: the guild adapter stores content type as `type`, not
    `content_type`. In n8n, Detect Media Type nodes that read `$json.content_type`
    silently got undefined and mis-routed everything. Centralizing the field name here
    means exactly one place breaks (loudly, via tests) if the adapter shape changes.
    """
    return attachment.get("type")


def detect_media_kind(url: str, mime_type: str | None = None) -> str:
    """Route a media URL to a tool kind: 'pdf' | 'docx' | 'text_file' | 'image'.

    n8n mapping: the Detect Media Type Code node's switch. Ordering is load-bearing
    (CLAUDE.md quirk): extension check FIRST, then MIME hint, then image as the
    catch-all. Image is what's left over, not something we positively detect —
    flipping that order sends PDFs to the vision API.

    The extension is read from the URL *path* only. Discord CDN URLs carry
    `?ex=...&is=...&hm=...` signature params, so a naive `url.endswith('.pdf')` misses
    `report.pdf?ex=...`. We parse the path for CLASSIFICATION only — the URL itself is
    never modified (see build_vision_body for why stripping the query is fatal).
    """
    path = urlsplit(url).path.lower()
    for extension, kind in _EXTENSION_KINDS.items():
        if path.endswith(extension):
            return kind
    if mime_type is not None:
        # Bare MIME match ('application/pdf'), ignoring any '; charset=...' suffix.
        kind = _MIME_KINDS.get(mime_type.split(";")[0].strip().lower())
        if kind is not None:
            return kind
    return "image"  # the catch-all branch — everything unmatched is vision's problem


# ---------------------------------------------------------------------------
# Request building — the jsonBody expression + node parameters.
# ---------------------------------------------------------------------------


def build_vision_body(image_url: str, prompt: str | None = None) -> dict:
    """Build the OpenAI-style multimodal chat body — the n8n jsonBody expression.

    Both arguments were `$fromAI(...)` slots in n8n: the calling LLM supplies them at
    tool-call time. Contract preserved from V1:

    - The URL passes through VERBATIM. Never download + re-encode (the n8n sandbox
      couldn't — `$helpers.getBinaryDataBuffer()` is blocked and binary storage returns
      "filesystem-v2" refs — and the proven pattern is handing the signed CDN URL
      straight to the vision API). Never strip query params: Discord CDN signatures
      (`?ex=&is=&hm=`) live there, and `split('?')[0]` earns a 403/404.
    - `image_url` content part comes FIRST, the `text` part second — same order as the
      live node, kept byte-identical so responses stay comparable across V1/V2.
    - Empty/missing prompt falls back to DEFAULT_PROMPT (the $fromAI hint, promoted).
    """
    url = image_url.strip() if isinstance(image_url, str) else ""
    if not url:
        raise ValueError("image_url is required — the vision tool has nothing to look at")

    text = (prompt or "").strip() or DEFAULT_PROMPT

    return {
        "model": VISION_MODEL,
        "max_tokens": VISION_MAX_TOKENS,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": url}},
                    {"type": "text", "text": text},
                ],
            }
        ],
    }


def build_auth_headers(api_key: str) -> dict:
    """OpenRouter header auth — the n8n `httpHeaderAuth` credential, materialized.

    n8n mapping: credential gvgPllzFhLSds5Qv "OpenRouter Header Auth" injected exactly
    this header; the node itself only declared `nodeCredentialType: httpHeaderAuth`.
    """
    key = api_key.strip() if isinstance(api_key, str) else ""
    if not key:
        raise ValueError("OpenRouter API key is required for the vision tool")
    return {"Authorization": f"Bearer {key}"}


def build_vision_request(
    image_url: str, prompt: str | None = None, *, api_key: str
) -> dict:
    """The full HTTP call as data — everything the transport layer needs to POST.

    n8n mapping: this is the whole Tool: Image node flattened into a dict:
    method/url from the node parameters, headers from the credential, json from the
    jsonBody expression. `responseFormat: json` from the node's options means the
    transport should parse the response as JSON and hand it to extract_description().
    """
    return {
        "method": "POST",
        "url": OPENROUTER_CHAT_URL,
        "headers": build_auth_headers(api_key),
        "json": build_vision_body(image_url, prompt),
    }


# ---------------------------------------------------------------------------
# Response handling — what the agent reads out of the tool result.
# ---------------------------------------------------------------------------


def extract_description(response: dict) -> str:
    """Pull the image description out of an OpenRouter completion response.

    n8n mapping: the raw completion JSON was returned to the agent wholesale
    (responseFormat json) and the agent read `choices[0].message.content` itself.
    The port does that read explicitly and fails LOUDLY on malformed payloads —
    in n8n a missing field became `undefined` flowing silently downstream.
    """
    error = response.get("error")
    if error:
        # OpenRouter error payloads are {"error": {"message": ..., "code": ...}}.
        message = error.get("message") if isinstance(error, dict) else str(error)
        raise ValueError(f"OpenRouter vision call returned an error: {message}")

    choices = response.get("choices")
    if not choices:
        raise ValueError("OpenRouter vision response had no choices — malformed payload")

    content = (choices[0].get("message") or {}).get("content")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("OpenRouter vision response had no message content")
    return content


def looks_like_signed_discord_url(url: str) -> bool:
    """True when the URL is a Discord CDN attachment carrying signature params.

    Discord signs attachment URLs with `?ex=...&is=...&hm=...` and the signature
    EXPIRES. Used to make 403/404 errors say the useful thing instead of the generic
    thing. Query-string sniffing only — no network.
    """
    parts = urlsplit(url)
    if "cdn.discordapp.com" not in parts.netloc and "media.discordapp.net" not in parts.netloc:
        return False
    query = parts.query
    return all(marker in query for marker in ("ex=", "is=", "hm="))


def explain_vision_http_error(status: int, image_url: str) -> str:
    """Turn an HTTP failure status into a message worth showing the agent.

    Fail-fast contract from the spec: a 403/404 on a signed Discord CDN URL is almost
    always an EXPIRED SIGNATURE (or someone stripped the query params upstream — the
    exact bug `split('?')[0]` causes). Say so, instead of a bare status code the agent
    would just retry into. Everything else gets a plain, honest failure line.
    """
    if status in (403, 404) and looks_like_signed_discord_url(image_url):
        return (
            f"Vision call failed with HTTP {status}: the Discord CDN signature on this "
            "attachment URL has likely expired (signed URLs carry ?ex=&is=&hm= params "
            "with a short lifetime). Re-fetch the attachment to get a fresh URL — and "
            "never strip the query string."
        )
    return f"OpenRouter vision call failed with HTTP {status}."
