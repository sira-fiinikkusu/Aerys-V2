"""Output formatting + chunking — the Output Router's Code nodes as pure functions.

n8n mapping: this is workflow `02-04-output-router` (V67KVguBAJG1sOij), which took the
agent's polished text and turned it into platform-ready sends:

    Set Polished Response → Platform Formatter → Message Splitter
        → Loop Over Chunks (splitInBatches, batch=1, sequential)
        → Switch: Route by Platform
            discord  → Build Discord Body → HTTP POST (retry x3)
            telegram → native Telegram node (parse_mode HTML)
            voice    → Strip Markdown (Voice) → TTS

Each Code node becomes one function here:

    Platform Formatter     → format_for_platform()  (+ to_telegram_html())
    Message Splitter       → split_message() / chunk_for_platform()
    Build Discord Body     → build_discord_body()
    Strip Markdown (Voice) → strip_markdown_for_voice()
    Switch node            → route_platform()

Everything is PURE — no HTTP, no credentials. The transport layers own the sending
(and the sequential in-order delivery that splitInBatches gave us for free: send
chunk 0, await, send chunk 1, ...).

Quirk rules carried over from CLAUDE.md / the live workflow:

1. "HTTP Request nodes wipe item JSON" — in n8n, Platform Formatter and Strip Markdown
   read `$('Set Polished Response').item.json` instead of `$json`, because the SQL
   Write-Back node upstream destroyed all item fields. The Python equivalent: callers
   pass the PERSISTED response context dict into these functions — never whatever a
   later pipeline step happened to return.
2. Discord sends MUST retry (3 tries, 2000ms apart) — DNS to discord.com transiently
   fails on this network stack. That's transport-layer work, but the policy constants
   live here (DISCORD_SEND_MAX_TRIES / DISCORD_SEND_RETRY_WAIT_MS) so the transport
   can't "forget" them.
3. The n8n Switch matched `source_channel` with strict case-sensitive equality and
   SILENTLY DROPPED unknown channels (no fallback output). We deliberately fix that:
   route_platform() raises ValueError so a typo'd channel fails loudly instead of
   eating the reply. (Documented deviation, see the function.)
4. The Telegram node applied `parse_mode = $json.parse_mode || 'HTML'` — so Telegram
   is ALWAYS HTML mode even when the formatter set none. effective_telegram_parse_mode()
   preserves that.
5. Signed Discord CDN URLs (`?ex=&is=&hm=`) must never be truncated — nothing in this
   module splits on `?` or strips query strings; golden tests pin that.

Chunking faithfully reproduces JS `String.lastIndexOf(needle, fromIndex)` semantics,
including its off-by-the-delimiter quirks — see split_message() for the gory details.
"""

import json
import re

# Platform message-size limits (hard API limits, not preferences).
DISCORD_LIMIT = 2000
TELEGRAM_LIMIT = 4096

# Quirk rule 2 — the documented-mandatory Discord retry policy. The transport layer
# that actually POSTs to discord.com must honor these (n8n: retryOnFail:true,
# maxTries:3, waitBetweenTries:2000 on the Send Discord Message node).
DISCORD_SEND_MAX_TRIES = 3
DISCORD_SEND_RETRY_WAIT_MS = 2000

# Quirk rule 4 — Telegram node default: `{{ $json.parse_mode || 'HTML' }}`.
TELEGRAM_DEFAULT_PARSE_MODE = "HTML"

# The channels the Switch node had outputs for. Anything else was silently dropped
# in n8n; here it raises (quirk rule 3).
KNOWN_CHANNELS = ("discord", "telegram", "voice")

# --- Telegram markdown → HTML regexes (verbatim ports of the JS) -------------------
# Fenced code block: optional language hint (captured but DISCARDED — Telegram's
# <pre><code> doesn't take a class in this port, same as the JS), optional newline,
# then a lazy body. [\s\S] is the JS idiom for "any char including newline";
# re.DOTALL + `.` is the Python equivalent.
_FENCED_RE = re.compile(r"```(\w+)?\n?(.*?)```", re.DOTALL)
_INLINE_CODE_RE = re.compile(r"`([^`]+)`")
_BOLD_RE = re.compile(r"\*\*([^*]+)\*\*")
# Negative lookbehind/lookahead so `*i*` matches but the leftover `**` of an already
# consumed bold never produces a stray <i>. Bold MUST run before italic.
_ITALIC_RE = re.compile(r"(?<!\*)\*([^*]+)\*(?!\*)")
# Placeholder tokens: \x00N\x00. NUL can't appear in chat text, so extracted code
# blocks survive the entity-escape + bold/italic passes untouched.
_PLACEHOLDER_RE = re.compile(r"\x00(\d+)\x00")


def _escape_html(text: str) -> str:
    """Telegram-HTML entity escape — exactly the three the JS escaped, in order.

    & first (or we'd double-escape the & in &lt;), then < and >. Note: this hits
    URLs too ("?ex=a&is=b" → "?ex=a&amp;is=b"). That is CORRECT for Telegram HTML
    mode — Telegram unescapes entities before display/linkification, so the user
    still sees and clicks the full signed URL (quirk rule 5: nothing truncates it).
    """
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def pick_response_text(ctx: dict) -> str:
    """The `polished_response || output || raw_response || text` fallback chain.

    n8n mapping: every consumer of Set Polished Response used this exact JS `||`
    chain. `or` mirrors JS truthiness here: missing keys AND empty strings both fall
    through to the next candidate. Final fallback is "" (the JS voice node's `|| ''`),
    so downstream string ops never explode on None.
    """
    return (
        ctx.get("polished_response")
        or ctx.get("output")
        or ctx.get("raw_response")
        or ctx.get("text")
        or ""
    )


def to_telegram_html(md: str) -> str:
    """Markdown → Telegram-safe HTML. Verbatim port of the `toTelegramHtml` JS.

    The ORDER is load-bearing (same as the Code node):
      1. Extract fenced code blocks → <pre><code>, entity-escaped, swapped for
         \\x00N\\x00 placeholders so later passes can't touch their contents.
      2. Extract inline code → <code>, same treatment.
      3. Entity-escape ALL remaining plain text (safe now — code is stashed away).
      4. **bold** → <b>, then *italic* → <i> (bold first, or `**x**` would parse as
         an italic containing asterisks).
      5. Restore the placeholders.

    Deliberately narrow, like the original: headers, links, strikethrough, and
    underscore-italics are NOT converted — they pass through escaped-as-text.
    """
    parts: list[str] = []

    def stash_fenced(m: re.Match) -> str:
        # m.group(1) is the language hint — captured but discarded, same as the JS.
        code = _escape_html(m.group(2))
        parts.append(f"<pre><code>{code}</code></pre>")
        return f"\x00{len(parts) - 1}\x00"

    def stash_inline(m: re.Match) -> str:
        code = _escape_html(m.group(1))
        parts.append(f"<code>{code}</code>")
        return f"\x00{len(parts) - 1}\x00"

    md = _FENCED_RE.sub(stash_fenced, md)
    md = _INLINE_CODE_RE.sub(stash_inline, md)
    md = _escape_html(md)
    md = _BOLD_RE.sub(r"<b>\1</b>", md)
    md = _ITALIC_RE.sub(r"<i>\1</i>", md)
    md = _PLACEHOLDER_RE.sub(lambda m: parts[int(m.group(1))], md)
    return md


def format_for_platform(ctx: dict) -> dict:
    """The `Platform Formatter` Code node.

    Takes the PERSISTED response context (quirk rule 1 — the n8n node read
    $('Set Polished Response').item.json, never $json, because the SQL Write-Back
    HTTP node wiped item fields). Returns a NEW dict: all of ctx spread through
    (source_channel, channel_id, person_id, ... survive) plus:

      formatted_response — Telegram: HTML-converted; everyone else: text untouched
                           (Discord renders its own markdown natively)
      parse_mode         — "HTML" for Telegram, None otherwise (JS: undefined)
    """
    text = pick_response_text(ctx)
    if ctx.get("source_channel") == "telegram":
        formatted = to_telegram_html(text)
        parse_mode = "HTML"
    else:
        formatted = text
        parse_mode = None
    return {**ctx, "formatted_response": formatted, "parse_mode": parse_mode}


def _js_last_index_of(haystack: str, needle: str, from_index: int) -> int:
    """JS `haystack.lastIndexOf(needle, fromIndex)` — the exact semantics.

    JS returns the largest index i <= fromIndex where needle STARTS — the match may
    extend PAST fromIndex. Python's rfind(sub, start, end) instead requires the match
    to END within [start, end). The translation: a match starting at i <= fromIndex
    is exactly a match ending within [0, fromIndex + len(needle)). Getting this wrong
    changes which delimiter wins near the limit (test: delimiter at exactly `limit`).
    Returns -1 when not found, like both languages.
    """
    return haystack.rfind(needle, 0, from_index + len(needle))


def split_message(text: str, limit: int) -> list[str]:
    """The `Message Splitter` Code node's splitMessage() — faithful port.

    Splits at natural boundaries, best-first: paragraph break > single newline >
    sentence end > hard cut at exactly `limit`. The acceptance windows differ by
    tier: newline splits must land at >= 50% of the limit, a sentence split is
    accepted down to 30%, otherwise hard cut. There is NO word-boundary fallback —
    a 2000+ char run with no \\n\\n / \\n / ". " in the window gets cut mid-word,
    same as production.

    Faithfully-preserved JS quirks (each pinned by a golden test):

    * text within the limit returns [text] UNTRIMMED — only split chunks get strip().
    * lastIndexOf semantics: a delimiter STARTING exactly at `limit` is eligible
      (see _js_last_index_of).
    * The split index sits at the START of the delimiter, so `\\n\\n` / `\\n` land at
      the head of `remaining` and vanish under strip() — but for ". " the sentence's
      final period travels WITH the delimiter: the previous chunk loses its trailing
      "." and the next chunk begins ". Next sentence..." (strip only removes
      whitespace, not the dot). Production has sent messages like this since Phase 2;
      the port reproduces it rather than silently changing live behavior. Fix it on
      purpose someday, with a test change, not by accident.
    * A remaining that strips to "" is dropped (JS `if (remaining)` truthiness).
    """
    if len(text) <= limit:
        return [text]
    chunks: list[str] = []
    remaining = text
    while len(remaining) > limit:
        # Tier 1: paragraph break at or before the limit.
        split_at = _js_last_index_of(remaining, "\n\n", limit)
        # Tier 2: single newline — only if the paragraph break was in the first half.
        if split_at < limit * 0.5:
            split_at = _js_last_index_of(remaining, "\n", limit)
        # Tier 3: sentence end — only if the newline was ALSO in the first half.
        if split_at < limit * 0.5:
            split_at = _js_last_index_of(remaining, ". ", limit)
        # Tier 4: hard cut. Note the looser 30% window — a sentence split between
        # 30% and 50% of the limit IS accepted (the 50% checks above only gated the
        # newline tiers).
        if split_at < 0 or split_at < limit * 0.3:
            split_at = limit
        chunks.append(remaining[:split_at].strip())
        remaining = remaining[split_at:].strip()
    if remaining:
        chunks.append(remaining)
    return chunks


def chunk_for_platform(item: dict) -> list[dict]:
    """The `Message Splitter` node's item fan-out: 1 item in → N chunk items out.

    Chunks the FORMATTED text — so Telegram is chunked post-HTML-conversion, on the
    <b>/<pre> form that actually gets sent (the 4096 limit applies to the payload,
    not the source markdown). Limits: discord → 2000; everything else (telegram,
    voice) → 4096.

    n8n mapping: returning N items here is what made Loop Over Chunks
    (splitInBatches, batch=1) iterate N times downstream. In V2 the transport layer
    owns that sequencing: send chunks IN ORDER, awaiting each, or Discord/Telegram
    will interleave them.
    """
    text = item.get("formatted_response") or ""
    limit = DISCORD_LIMIT if item.get("source_channel") == "discord" else TELEGRAM_LIMIT
    chunks = split_message(text, limit)
    return [
        {**item, "chunk": chunk, "chunk_index": i, "total_chunks": len(chunks)}
        for i, chunk in enumerate(chunks)
    ]


def build_discord_body(chunk: str) -> str:
    """The `Build Discord Body` Code node — always a plain-content message.

    Returns the JSON string for POST https://discord.com/api/v10/channels/
    {channel_id}/messages. separators=(",", ":") matches JS JSON.stringify
    byte-for-byte (no spaces); ensure_ascii=False because JSON.stringify emits
    real UTF-8, not \\uXXXX escapes. The transport sending this MUST apply the
    DISCORD_SEND_* retry policy above (quirk rule 2).
    """
    return json.dumps({"content": chunk}, separators=(",", ":"), ensure_ascii=False)


# --- Strip Markdown (Voice) regexes — one per JS .replace(), in the JS order -------
# ORDER MATTERS: fenced blocks go first (deleted outright — you don't read code
# aloud), bold before single-star italic, double-underscore before single.
_VOICE_PASSES: list[tuple[re.Pattern, str]] = [
    (re.compile(r"```.*?```", re.DOTALL), ""),  # fenced code: DELETED, not unwrapped
    (re.compile(r"\*\*([^*]+)\*\*"), r"\1"),  # **bold** → bold
    (re.compile(r"\*([^*]+)\*"), r"\1"),  # *italic* → italic
    (re.compile(r"__([^_]+)__"), r"\1"),  # __underline__ → underline
    (re.compile(r"_([^_]+)_"), r"\1"),  # _italic_ → italic
    # Header markers. NOTE: the JS had no ^ anchor — `#{1,6}\s*` fires ANYWHERE, so
    # "C# code" → "Ccode". Faithful port of a live quirk; TTS has said "Ccode".
    (re.compile(r"#{1,6}\s*"), ""),
    (re.compile(r"\[([^\]]+)\]\([^)]+\)"), r"\1"),  # [text](url) → text
    (re.compile(r"`([^`]+)`"), r"\1"),  # `inline` → inline
    (re.compile(r"^[\s]*[-*+]\s", re.MULTILINE), ""),  # bullet markers
    (re.compile(r"^[\s]*\d+\.\s", re.MULTILINE), ""),  # numbered-list markers
    (re.compile(r"\n{3,}"), "\n\n"),  # collapse 3+ newlines
]


def strip_markdown_for_voice(ctx: dict) -> dict:
    """The `Strip Markdown (Voice)` Code node — markdown → speakable plain text.

    Voice BYPASSES the chunker entirely: in n8n this node re-read the FULL unchunked
    text from $('Set Polished Response') (quirk rule 1 again), because TTS wants one
    continuous utterance, not 2000-char fragments. Same contract here: pass the
    persisted response ctx, get {**ctx, "tts_text": clean} back.

    Bare URLs (e.g. signed Discord CDN links) are NOT markdown links, so they pass
    through with every query param intact (quirk rule 5) — awkward to hear, but
    never corrupted.
    """
    clean = pick_response_text(ctx)
    for pattern, replacement in _VOICE_PASSES:
        clean = pattern.sub(replacement, clean)
    return {**ctx, "tts_text": clean.strip()}


def route_platform(source_channel: object) -> str:
    """The `Switch: Route by Platform` node — strict, case-sensitive matching.

    DOCUMENTED DEVIATION from n8n: the Switch had no fallback output, so an unknown
    source_channel meant the reply silently evaporated — the worst possible failure
    mode for a companion (Aerys just... doesn't answer). Here it raises instead, so
    a typo'd channel surfaces in logs/alerts on the FIRST occurrence. The matching
    itself stays strict and case-sensitive ("Discord" != "discord"), same as the
    Switch — normalizing case is the adapters' job at ingest, not ours at egress.
    """
    if source_channel not in KNOWN_CHANNELS:
        raise ValueError(
            f"Unroutable source_channel {source_channel!r} — "
            f"expected one of {KNOWN_CHANNELS} (case-sensitive)"
        )
    return source_channel


def effective_telegram_parse_mode(item: dict) -> str:
    """Quirk rule 4: the Telegram node's `{{ $json.parse_mode || 'HTML' }}` default.

    Even when the formatter set parse_mode to None (it never does for Telegram, but
    belt-and-braces), the send falls back to HTML — Telegram is effectively ALWAYS
    HTML mode. The transport should also mirror the node's other settings:
    disable_web_page_preview=True, no attribution appended.
    """
    return item.get("parse_mode") or TELEGRAM_DEFAULT_PARSE_MODE
