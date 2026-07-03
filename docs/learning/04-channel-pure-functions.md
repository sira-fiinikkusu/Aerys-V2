# LEARNING — channels/: the Output Router and Detect Media Type as pure functions

*2026-07-02. The `src/aerys_v2/channels/` package. This is the doc where three of the
gnarliest Code nodes in Aerys V1 — Output Router formatting/chunking, Detect Media
Type, and the Tool: Image vision call — become plain Python functions with golden
tests. Every quirk that lived as a warning paragraph in CLAUDE.md is now a test that
fails if anyone breaks it.*

## The one-sentence version

`splitter.py`, `media_detect.py`, and `vision_ladder.py` are the channel-facing Code
nodes rewritten as pure functions (no I/O, no network, no n8n sandbox), and the ~130
tests pinning them mean the 2000-char chunking rules, signed-CDN-URL preservation, and
extension-before-image ordering can never silently regress again.

## The big idea: quirks as tests, not comments

In V1, the rules that kept messages from silently vanishing lived in two places:
inside sandboxed Code nodes you could only debug by sending yourself Discord messages,
and as prose warnings in CLAUDE.md ("never strip query params", "extensions before
image catch-all", "guild adapter stores MIME under `type`"). Prose can't fail a build.
A new session that skims past the warning re-introduces the bug, and you find out when
a PDF routes to the vision API or a chunk arrives truncated.

Now each of those rules is a named test. `test_query_discord_cdn_pdf_routes_to_pdf_not_image`
IS the extension-ordering rule. `test_signed_url_passes_through_verbatim` IS the
CDN-signature rule. `test_chunk_discord_uses_2000_limit` IS the chunking rule. Break
one and `uv run pytest` goes red before anything ships — CLAUDE.md becomes the story
of why, not the only line of defense.

## splitter.py — the Output Router (02-04, V67KVguBAJG1sOij)

Each Code node in the workflow became exactly one function:

| n8n node | aerys-v2 function | what it does |
|---|---|---|
| `$('Set Polished Response')` reads | `pick_response_text()` | the `polished_response \|\| output \|\| raw_response \|\| text` fallback chain, JS-truthiness included |
| Platform Formatter | `format_for_platform()` + `to_telegram_html()` | Telegram gets markdown→HTML (placeholder-protected code blocks); Discord/voice pass through |
| Message Splitter | `split_message()` + `chunk_for_platform()` | paragraph > newline > sentence > hard cut; discord=2000, everything else=4096 |
| Build Discord Body | `build_discord_body()` | `JSON.stringify` parity — no spaces, real UTF-8 |
| Strip Markdown (Voice) | `strip_markdown_for_voice()` | full unchunked text → speakable; fenced code deleted, not read aloud |
| Switch: Route by Platform | `route_platform()` | strict case-sensitive match — but unknown channels now RAISE (see deviations) |
| Telegram node's `parse_mode \|\| 'HTML'` | `effective_telegram_parse_mode()` | Telegram is always HTML mode, even if the formatter set None |

Two things worth noticing:

- **The chunker is a faithful port of JS, quirks included.** `_js_last_index_of()`
  reproduces `String.lastIndexOf(needle, fromIndex)` exactly — a delimiter *starting*
  at the limit is eligible even though it extends past, which a naive Python
  `rfind(..., 0, limit)` misses. And the ". " split moves the sentence's final period
  onto the head of the next chunk (`"Alpha beta gam"` / `". Delta..."`). Production
  has sent messages like that since Phase 2, so the port reproduces it and the test
  says so out loud: fix it on purpose someday, with a test change, not by accident.
- **The retry policy is data, not folklore.** `DISCORD_SEND_MAX_TRIES = 3` /
  `DISCORD_SEND_RETRY_WAIT_MS = 2000` live in the module and are pinned by a test —
  the CLAUDE.md "Discord DNS transiently fails, retry 3x/2000ms" rule can't be
  "simplified" away when someone writes the transport layer.

## media_detect.py — Detect Media Type (from 05-01, nqTeANyy5jlZgWJI)

`detect_media(payload)` is the whole classifier: attachments[0] first (MIME, then
filename extension), then the YouTube scan across content → context → message_content,
then the bare-URL fallback on `query`. `'unknown'` is a valid answer, not an error.

The four quirk rules from CLAUDE.md, each now a golden test:

| Quirk (was a CLAUDE.md warning) | The test that enforces it |
|---|---|
| Never strip `?ex=&is=&hm=` from CDN URLs — `split('?')[0]` only for the display *name* | `test_discord_image_attachment_keeps_signed_url`, `test_query_discord_cdn_pdf_routes_to_pdf_not_image` (checks both URL intact AND name trimmed) |
| Guild adapter stores MIME under `type`, not `content_type` — coalesce content_type → mime_type → type | `test_guild_adapter_type_field_detected` |
| Documents (.pdf/.docx/.txt) checked BEFORE the image catch-all, or every CDN URL routes to vision | `test_query_docx_and_txt_before_image_catch_all` |
| `attachments` may arrive as a JSON string or garbage (toolWorkflow serialization) — parse defensively | `test_attachments_as_json_string`, `test_attachments_non_array_garbage_becomes_empty` |

In n8n, verifying any of these meant uploading a real file to Discord and reading the
execution log. Now it's a <1ms function call in pytest.

## vision_ladder.py — Tool: Image, flattened into data

The `httpRequestTool` node's `jsonBody` expression becomes `build_vision_body()`, the
node parameters + credential become `build_vision_request()` / `build_auth_headers()`,
and the agent's read of `choices[0].message.content` becomes `extract_description()` —
which raises loudly on malformed payloads instead of n8n's silent-`undefined` flow.

The parts of the node that were *prompt engineering* are pinned too:
`TOOL_NAME == "image"` (the CLAUDE.md name-mismatch quirk — wrong name and the LLM
hallucinates having called the tool) and `TOOL_DESCRIPTION` still contains the literal
`https://cdn.discordapp.com/attachments/` trigger and "ZERO visual capability" framing,
because specificity-beats-generality is what makes the model actually call it. There's
even a test asserting the module imports no HTTP library — purity enforced by grep.

**Despite the filename, there is no model ladder here.** All three tier agents carry
byte-identical Tool: Image nodes; vision is always `anthropic/claude-sonnet-4.6` (the
undated slug — the dated ID fails on OpenRouter, and a test pins that too). The only
fallback chain in V1 is at the router level (opus → sonnet → gemini), which swaps
*agents*, never the vision model.

## Deliberate deviations from V1 (documented, tested)

The port isn't blind. Three places where V1's behavior was a bug we chose to fix:

- **Unknown channels raise.** The n8n Switch had no fallback output — a typo'd
  `source_channel` meant the reply silently evaporated, the worst failure mode for a
  companion. `route_platform()` raises `ValueError` so it surfaces on the first
  occurrence. (`test_route_unknown_channel_raises` documents the decision.)
- **Malformed vision responses raise.** `undefined` no longer flows downstream.
- **The `$fromAI` prompt hint is now a real default.** In n8n, "Describe this image in
  detail" was only a description shown to the LLM; the port makes an empty prompt
  actually produce it.

Everything else — including the awkward stuff like the unanchored header regex that
turns "C# code" into "Ccode" for TTS, and snake_case losing its underscores — is
faithfully preserved and pinned, because changing live behavior by accident during a
port is how migrations lose trust.

## What the transport layer still owes

Pure functions can't send. The (future) transport layer must honor the contracts these
modules hand it: send chunks **in order, awaiting each** (what splitInBatches gave V1
for free), apply the `DISCORD_SEND_*` retry policy, pass the **persisted** response
context into the formatters (quirk rule 1 — never whatever the last pipeline step
returned; that's the "HTTP Request nodes wipe item JSON" bug in Python clothing), and
POST `build_vision_request()`'s dict then feed the JSON back through
`extract_description()` / `explain_vision_http_error()`.

## Try it yourself

```bash
cd ~/projects/aerys-v2
uv run pytest tests/test_splitter.py tests/test_media_detect.py tests/test_vision_ladder.py -q

# Watch a quirk rule catch a regression: strip the query params like the old bug did
uv run python -c "
from aerys_v2.channels.media_detect import detect_media
url = 'https://cdn.discordapp.com/attachments/1/2/report.pdf?ex=a&is=b&hm=c'
out = detect_media({'query': url})
print(out['mediaType'], '->', out['mediaUrl'])   # pdf -> full signed URL, untouched
print('display name:', out['filename'])          # report.pdf — ?ex= trimmed from NAME only
"

# The period-migration chunker quirk, live
uv run python -c "
from aerys_v2.channels.splitter import split_message
print(split_message('Alpha beta gam. Delta epsilon zeta eta.', 20))
"
```

## What's deliberately NOT here yet

The transport layer itself (Discord/Telegram/voice clients that actually send), the
YouTube/PDF/DOCX *extractors* (detection routes to them; fetching and parsing is I/O
and comes with transports), Telegram `getFile` download handling
(`needsTelegramDownload` is flagged, not acted on), and any wiring of these functions
into the graph — that's the next seam. Pure first, plumbed second.
