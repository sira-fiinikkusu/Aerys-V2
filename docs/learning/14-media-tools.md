# LEARNING — the media tools: she can see now (tools/media.py)

*2026-07-03. Vision, documents, YouTube — the read half of the toolbox. Every concept
below is mapped to something you already run in n8n.*

## The one-sentence version

`tools/media.py` replaces the whole 06-05 media stack — the `Tool: Image`
httpRequestTool node plus the YouTube (HE7zmxKeWoxjvM9L), PDF (yuHzxHqqWz93xwYj) and
DOCX (tJwLt494G1VugToU) extractor sub-workflows — as three LangChain tools riding the
action subgraph, wired from ONE OpenRouter credential and obeying the same contracts as
home_control.

## The four contracts (same file, same discipline as home_control)

| Contract | What it means | V1 scar it protects |
|---|---|---|
| READ-ONLY | nothing here changes external state — no outbox, because there's no write to audit | — |
| HONEST FAILURE | every error path returns a plain string the model must relay; NEVER raise out of a tool | an exception inside ToolNode kills the whole turn — the failed-webhook-kills-execution outage mode |
| URL SANCTITY | signed Discord CDN URLs (`?ex=&is=&hm=`) pass through VERBATIM — never `split('?')[0]`, never download-and-re-encode for vision | stripped signature params = dead link; the n8n sandbox couldn't even read binary, so signed-URL-straight-to-API is the proven pattern |
| EXTENSIONS BEFORE IMAGE | routing reuses `detect_media_kind()`, where .pdf/.docx/.txt match before the image catch-all — and the extension reads from the URL *path*, so `report.pdf?ex=...` still says pdf | flipping that order sent every attachment to vision in V1 (the Detect Media Type quirk) |

## The signed-URL lesson, taught to the model too

The tool docstrings — which ARE the tool descriptions the LLM reads — say "pass the URL
EXACTLY as it appeared, keep every query parameter." And when a fetch comes back 403/404
on a URL that `looks_like_signed_discord_url()`, the error string explains the expired
signature instead of returning a bare status the model would just retry into. The same
lesson lives in three layers: the docstring (prevention), the pass-through (correctness),
and the error message (diagnosis).

## The three tools

- **analyze_image** — the `Tool: Image` node, live. `build_vision_body()` (pure, from
  doc 04) builds the multimodal message; the signed URL goes straight into
  `image_url.url`. Empty question promotes to "describe it".
- **read_document** — the PDF/DOCX/TXT extractors, live. pypdf replaces the PDF
  sub-workflow, python-docx replaces the @mazix converter node (whose output hid at
  `$json.files[0].text` — that quirk is now just `"\n".join(paragraph.text ...)`), and
  text files decode with `errors="replace"` (mojibake beats a dead turn). Routing
  honesty: a YouTube link or an image gets a redirect string naming the RIGHT tool.
  Long documents truncate at 8k chars and SAY so — a tool message rides inside the
  context window, and a 300-page PDF must not blow the turn.
- **youtube_summary** — caption transcript → LLM summary. Transcript only, no yt-dlp,
  no video bytes (v1's extractor made the same call: captions are kilobytes, videos are
  gigabytes). The fetcher is an injectable seam (`TranscriptFetcher`) — tests pass a
  lambda; prod defers the youtube-transcript-api import so tests never load its
  network machinery. Transcripts cap at ~12k chars (30-40 min of speech) and the
  summary prompt admits the truncation.

## One credential, three jobs — the arming pattern again

`build_media_tools()` wires all three from `settings.embeddings_api_key` +
`embeddings_base_url` — the memory embedder's OpenRouter credential, reused for chat
completions, exactly like n8n credential gvgPllzFhLSds5Qv served both vision and
embeddings. No key = the media tools simply don't exist, same as `ha_token` gating the
home_control stack.

Registration lives in `factory.action_tools_for()`: two independently-armed halves —
HOME (ha_token → home_control + search_entities, the write half) and MEDIA
(embeddings_api_key → these three, the read half). Either half alone arms the action
stack; both empty = ask() stays chat-only. And `action_overlay_for()` composes the
system prompt from ONLY the armed halves — telling the model to "use home_control" on a
box without ha_token is the V1 hallucinated-tool-call failure with extra steps. The
MEDIA_OVERLAY names the exact tool names (`analyze_image`, `read_document`,
`youtube_summary`) with concrete CDN-URL trigger patterns — the toolWorkflow
name-mismatch bug and the "specificity beats generality" lesson, both kept dead by
`test_build_media_tools_names_match_what_prompts_call` and
`test_media_overlay_names_the_real_tools_and_url_sanctity`.

## Everything injectable — the seam philosophy, third verse

Builders close over config and an optional `httpx.Client`; the tools never read
Settings (construction knows config, behavior doesn't — doc 01's factory rule). Tests
run every path against `httpx.MockTransport`: no network, no API key, and the vision
tests literally assert the `?ex=&is=&hm=` params arrived at the fake API intact.

## The tests — 19 new, all offline

`test_media_tools.py` covers: vision call shape + verbatim signature pass-through +
default prompt + expired-signature explanation + unreachable/malformed honesty; real
PDF/DOCX bytes extracted (pypdf and python-docx run for real against generated files),
truncation, extension-beats-image routing, YouTube/image redirects, corrupt-PDF
honesty; transcript happy path (long and short link forms), truncation, and every
failure mode returning words instead of a raise.

## Try it yourself

```bash
uv run pytest -q tests/test_media_tools.py tests/test_media_detect.py \
  tests/test_vision_ladder.py                     # the pure + live halves together
# live: EMBEDDINGS_API_KEY in .env arms the media half; then over Discord or voice,
# drop an attachment: "what's in this image?" → router says action → analyze_image
```

## What's deliberately NOT here yet

Video download or frame analysis (transcript-only is the deliberate boundary), OCR for
scanned/image-only PDFs (the empty-text error says so honestly), audio transcription,
media WRITES (image generation etc. — those would need the outbox), and per-file-size
guards beyond the char caps. The read half of the toolbox stays cheap and honest.
