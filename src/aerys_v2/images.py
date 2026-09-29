"""Let her SEE an image in her own chat turn.

Before (found 2026-09-28): an image in a Discord message went down the action path,
where a separate vision tool (analyze_image, OpenRouter) described it and she relayed
the transcript -- Selyra's words to her came back as "That image reads: ...". Now the
chat node inlines the image itself, so the model answering him is the one looking.

The stored message is unchanged: its text still carries the signed CDN URL, as it
always has. Only the prompt for THIS call changes, at the model boundary:
- the LAST human message: each Discord CDN image URL is fetched and becomes a base64
  image block after the text (the CLI on the plan accepts base64 but cannot fetch URLs:
  measured on 9/28), and the URL in the text becomes "[image attached]";
- earlier human messages: a CDN image URL becomes "[an image shared earlier]" (the
  signed URL has expired by then anyway, and old images stay out of the prompt).
A fetch that fails becomes a note that the image didn't load, so she can say so or hand
off to the action path (which reads the stored message, URL intact, with analyze_image)
instead of guessing.
"""
from __future__ import annotations

import base64
import logging
import re
from contextvars import ContextVar
from typing import Callable

import httpx
from langchain_core.messages import BaseMessage, HumanMessage

log = logging.getLogger(__name__)

#: Discord's CDN only: the fetch is ours, so it never follows a URL of anyone's choosing.
IMAGE_URL_RE = re.compile(
    r"https://(?:cdn\.discordapp\.com|media\.discordapp\.net)/attachments/\S+?\.(?:png|jpe?g|gif|webp)(?:\?\S*)?",
    re.IGNORECASE,
)
MAX_IMAGES = 4
MAX_BYTES = 5 * 1024 * 1024          # the API's per-image limit
FETCH_TIMEOUT_S = 10.0
ATTACHED = "[image attached]"
EARLIER = "[an image shared earlier]"
NOT_LOADED = "[an image was attached but didn't load]"
#: A Telegram photo in the stored text (2026-09-29). Never its URL: a Telegram file URL
#: carries the bot token. The gateway downloads the bytes and hands them to THIS turn
#: through TURN_IMAGES, so the chat node can inline them like a Discord image.
PHOTO_MARKER = "[a photo]"
#: ((media_type, bytes), ...) for the current turn only (telegram_gateway sets it).
TURN_IMAGES: ContextVar[tuple] = ContextVar("aerys_turn_images", default=())


def has_photo(text: str) -> bool:
    """A Telegram photo: the gateway always puts PHOTO_MARKER FIRST, so the same words
    typed mid-message are just words (Gemini review, 2026-09-29)."""
    return text.lstrip().startswith(PHOTO_MARKER)


def carries_image(text: str) -> bool:
    """Does TEXT carry an image she can see (a Discord CDN image link or a Telegram photo)?"""
    return bool(IMAGE_URL_RE.search(text)) or has_photo(text)


def without_image_refs(text: str) -> str:
    """TEXT with its image links and its leading photo marker taken out (what his words say)."""
    text = IMAGE_URL_RE.sub(" ", text)
    return text.lstrip()[len(PHOTO_MARKER):] if has_photo(text) else text


def with_turn_images(images, fn, *args, **kwargs):
    """Run FN with IMAGES as this turn's images (a transport's executor thread)."""
    token = TURN_IMAGES.set(tuple(images))
    try:
        return fn(*args, **kwargs)
    finally:
        TURN_IMAGES.reset(token)
UNSEEN = "[an image was attached, but this model can't see it]"

_MAGIC = (
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"GIF87a", "image/gif"),
    (b"GIF89a", "image/gif"),
)


def media_type_of(data: bytes) -> str | None:
    """The image type from its first bytes (never from the URL or a header)."""
    for magic, kind in _MAGIC:
        if data.startswith(magic):
            return kind
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return None


def fetch_image(url: str) -> tuple[str, bytes] | None:
    """(media_type, bytes), or None when it isn't a loadable image within the limits."""
    try:
        with httpx.Client(timeout=FETCH_TIMEOUT_S, follow_redirects=False) as client:
            with client.stream("GET", url, headers={"User-Agent": "Mozilla/5.0 (aerys)"}) as response:
                if response.status_code != 200:
                    log.warning("image fetch: HTTP %s", response.status_code)
                    return None
                data = b""
                for chunk in response.iter_bytes():
                    data += chunk
                    if len(data) > MAX_BYTES:
                        log.warning("image fetch: over %d bytes, skipped", MAX_BYTES)
                        return None
    except httpx.HTTPError as exc:
        log.warning("image fetch failed: %s", type(exc).__name__)
        return None
    kind = media_type_of(data)
    if kind is None:
        log.warning("image fetch: not a png/jpeg/gif/webp")
        return None
    return kind, data


def _swap_marker(content, replacement: str):
    """CONTENT with its leading PHOTO_MARKER replaced (a string, or its first text block)."""
    def swap(text):
        return text.replace(PHOTO_MARKER, replacement, 1) if has_photo(text) else text
    if isinstance(content, str):
        return swap(content)
    first = next((i for i, b in enumerate(content) if isinstance(b, dict) and b.get("type") == "text"), None)
    return [{**b, "text": swap(b.get("text", ""))} if i == first else b for i, b in enumerate(content)]


def _texts(content) -> list[str]:
    if isinstance(content, str):
        return [content]
    return [block.get("text", "") for block in content if isinstance(block, dict) and block.get("type") == "text"]


def _replace_urls(content, replacement: Callable[[re.Match], str]):
    """CONTENT with every CDN image URL in its text replaced; other blocks untouched."""
    if isinstance(content, str):
        return IMAGE_URL_RE.sub(replacement, content)
    return [
        {**block, "text": IMAGE_URL_RE.sub(replacement, block.get("text", ""))}
        if isinstance(block, dict) and block.get("type") == "text" else block
        for block in content
    ]


def inline_current_images(prompt: list[BaseMessage], *, fetch: Callable[[str], tuple[str, bytes] | None] = fetch_image
                          ) -> list[BaseMessage]:
    """The prompt for this call, with the last human message's images inlined (see the
    module docstring). Returns the prompt unchanged when there is nothing to do."""
    last = max((i for i, m in enumerate(prompt) if isinstance(m, HumanMessage)), default=None)
    if last is None:
        return prompt
    out = list(prompt)
    for i in range(last):
        if isinstance(out[i], HumanMessage) and any(carries_image(t) for t in _texts(out[i].content)):
            content = _replace_urls(out[i].content, lambda _m: EARLIER)
            out[i] = out[i].model_copy(update={"content": _swap_marker(content, EARLIER)})
    urls = [m.group(0) for t in _texts(out[last].content) for m in IMAGE_URL_RE.finditer(t)][:MAX_IMAGES]
    photo = any(has_photo(t) for t in _texts(out[last].content)[:1])
    if not urls and not photo:
        return out
    images, loaded = [], set()
    if photo:
        # A Telegram photo: its bytes came with this turn (never a URL). Missing bytes
        # (a failed download) say so, like a Discord image that didn't load.
        sent = [(kind, data) for kind, data in TURN_IMAGES.get() if media_type_of(data) == kind][:MAX_IMAGES]
        for kind, data in sent:
            images.append({"type": "image", "source": {"type": "base64", "media_type": kind,
                                                       "data": base64.b64encode(data).decode("ascii")}})
        out[last] = out[last].model_copy(update={"content": _swap_marker(out[last].content,
                                                                         ATTACHED if sent else NOT_LOADED)})
        urls = urls[:max(0, MAX_IMAGES - len(images))]
    for url in urls:
        got = fetch(url)
        if got is None:
            continue
        kind, data = got
        images.append({"type": "image", "source": {"type": "base64", "media_type": kind,
                                                   "data": base64.b64encode(data).decode("ascii")}})
        loaded.add(url)
    content = _replace_urls(out[last].content, lambda m: ATTACHED if m.group(0) in loaded else NOT_LOADED)
    blocks = [{"type": "text", "text": content}] if isinstance(content, str) else list(content)
    out[last] = out[last].model_copy(update={"content": [*blocks, *images]})
    return out


def without_images(messages: list[BaseMessage]) -> list[BaseMessage]:
    """MESSAGES with the inlined image blocks taken out and their "[image attached]"
    marker saying the image can't be seen: for the local lifeboat (a text model that
    errors on an image), so a failed image turn still gets an honest answer instead
    of the error apology."""
    out = []
    for m in messages:
        if isinstance(m, HumanMessage) and isinstance(m.content, list) and any(
                isinstance(b, dict) and b.get("type") == "image" for b in m.content):
            kept = [{**b, "text": b.get("text", "").replace(ATTACHED, UNSEEN)}
                    if isinstance(b, dict) and b.get("type") == "text" else b
                    for b in m.content if not (isinstance(b, dict) and b.get("type") == "image")]
            m = m.model_copy(update={"content": kept})
        out.append(m)
    return out
