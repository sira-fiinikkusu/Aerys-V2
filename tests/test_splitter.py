"""Golden tests for the Output Router port (channels/splitter.py) — no I/O, no network.

These pin the LIVE behavior of workflow 02-04 (V67KVguBAJG1sOij), quirks included:
the JS lastIndexOf boundary semantics, the ". "-split period migration, the unanchored
header regex, the always-HTML Telegram default. Where the port deliberately deviates
(unknown channel raises instead of silently dropping), the test documents the decision.
If one of these goldens starts failing, either the port drifted or someone changed
production behavior on purpose — both deserve a diff review, not a quick fix.
"""

import json

import pytest

from aerys_v2.channels.splitter import (
    DISCORD_LIMIT,
    DISCORD_SEND_MAX_TRIES,
    DISCORD_SEND_RETRY_WAIT_MS,
    TELEGRAM_LIMIT,
    build_discord_body,
    chunk_for_platform,
    effective_telegram_parse_mode,
    format_for_platform,
    pick_response_text,
    route_platform,
    split_message,
    strip_markdown_for_voice,
    to_telegram_html,
)

# A signed Discord CDN URL — the ?ex=&is=&hm= params expire but are REQUIRED for the
# download to work. Nothing in the router may truncate or split them (quirk rule 5).
SIGNED_URL = "https://cdn.discordapp.com/attachments/1/2/photo.png?ex=6612aa&is=6611bb&hm=deadbeef"

# The persisted response context — what n8n's $('Set Polished Response').item.json
# held. Passthrough fields must survive every stage untouched.
CTX = {
    "polished_response": "hello there",
    "source_channel": "discord",
    "channel_id": "1480365115684688127",
    "person_id": "6e6bcbed-03ef-4d17-95d2-89c467414335",
    "session_id": "s-1",
    "model_tier": "sonnet",
    "_jailbreak_detected": False,
}


# --- pick_response_text: the `||` fallback chain ------------------------------------


def test_fallback_prefers_polished_response():
    ctx = {"polished_response": "polished", "output": "raw", "text": "last"}
    assert pick_response_text(ctx) == "polished"


def test_fallback_empty_string_falls_through():
    # JS `||` treats "" as falsy — an empty polished_response must NOT win.
    ctx = {"polished_response": "", "output": "", "raw_response": "raw", "text": "t"}
    assert pick_response_text(ctx) == "raw"


def test_fallback_all_missing_yields_empty_string():
    # The voice node's `|| ''` terminator: never None, downstream string ops are safe.
    assert pick_response_text({}) == ""


# --- to_telegram_html: markdown → Telegram-safe HTML --------------------------------


def test_telegram_html_escapes_entities():
    assert to_telegram_html("a < b & c > d") == "a &lt; b &amp; c &gt; d"


def test_telegram_html_bold_then_italic():
    assert (
        to_telegram_html("**bold** and *ital*") == "<b>bold</b> and <i>ital</i>"
    )


def test_telegram_html_fenced_block_escaped_and_lang_discarded():
    # Language hint (`python`) is captured but discarded; entities inside the block
    # are escaped; the trailing newline of the code body is preserved.
    assert (
        to_telegram_html("```python\nx < 1 & y\n```")
        == "<pre><code>x &lt; 1 &amp; y\n</code></pre>"
    )


def test_telegram_html_inline_code_protects_markdown():
    # The \x00N\x00 placeholder swap: **…** inside backticks must NOT become <b>.
    assert (
        to_telegram_html("use `**not bold**` now")
        == "use <code>**not bold**</code> now"
    )


def test_telegram_html_fenced_protects_markdown():
    assert to_telegram_html("```\n*stars*\n```") == "<pre><code>*stars*\n</code></pre>"


def test_telegram_html_multiple_placeholders_restore_in_order():
    assert to_telegram_html("`a` and `b`") == "<code>a</code> and <code>b</code>"


def test_telegram_html_narrow_by_design():
    # Headers/links are NOT converted — they pass through escaped-as-text,
    # exactly like the JS. (Telegram renders them as literal text.)
    assert to_telegram_html("# Head [x](http://a)") == "# Head [x](http://a)"


def test_telegram_html_signed_url_only_ampersand_escaped():
    # Quirk rule 5: the full query string survives. & → &amp; is CORRECT for HTML
    # parse mode — Telegram unescapes it before display/linkification.
    assert to_telegram_html(SIGNED_URL) == SIGNED_URL.replace("&", "&amp;")


# --- format_for_platform: the Platform Formatter node -------------------------------


def test_format_telegram_converts_and_sets_parse_mode():
    ctx = {**CTX, "source_channel": "telegram", "polished_response": "**hi** `x<y`"}
    out = format_for_platform(ctx)
    assert out["formatted_response"] == "<b>hi</b> <code>x&lt;y</code>"
    assert out["parse_mode"] == "HTML"


def test_format_discord_passthrough():
    ctx = {**CTX, "polished_response": "**raw markdown** stays"}
    out = format_for_platform(ctx)
    # Discord renders its own markdown natively — no conversion, no parse_mode.
    assert out["formatted_response"] == "**raw markdown** stays"
    assert out["parse_mode"] is None


def test_format_voice_passthrough():
    out = format_for_platform({**CTX, "source_channel": "voice"})
    assert out["formatted_response"] == "hello there"
    assert out["parse_mode"] is None


def test_format_spreads_ctx_through():
    # Quirk rule 1: the node spread ctx (the persisted response), so channel_id,
    # person_id, etc. survive into the send stage.
    out = format_for_platform(CTX)
    for key, value in CTX.items():
        assert out[key] == value


# --- split_message: faithful JS chunker ---------------------------------------------


def test_split_within_limit_returns_untrimmed():
    # Only SPLIT chunks are trimmed; a text within the limit passes through as-is.
    assert split_message("  hi  ", 10) == ["  hi  "]


def test_split_prefers_paragraph_break():
    text = "one two three\n\nfour five six seven eight"
    assert split_message(text, 20) == [
        "one two three",  # split at the \n\n (index 13, past the 50% window)
        "four five six seven",  # no delimiter in window → hard cut at 20, trimmed
        "eight",
    ]


def test_split_falls_back_to_newline():
    text = "line one here\nline two goes on and on"
    assert split_message(text, 20) == [
        "line one here",  # \n at 13 ≥ 50% of limit → accepted
        "line two goes on and",  # hard cut at 20
        "on",
    ]


def test_split_sentence_period_migrates_to_next_chunk():
    # THE ". " QUIRK, pinned: the split index sits at the '.', so the sentence's
    # final period leaves chunk 1 and arrives at the head of chunk 2 (strip removes
    # whitespace, not dots). Production has always sent this; the port must too.
    text = "Alpha beta gam. Delta epsilon zeta eta."
    assert split_message(text, 20) == [
        "Alpha beta gam",  # NO trailing period
        ". Delta epsilon zeta",  # period at the head; ". " at 0 < 30% → hard cut at 20
        "eta.",
    ]


def test_split_sentence_accepted_in_30_to_50_percent_window():
    # The newline tiers demand ≥ 50% of the limit; the sentence tier only ≥ 30%.
    # ". " at index 7 (35% of 20) is rejected nowhere and wins.
    text = "Aa bb c. " + "d" * 13
    assert split_message(text, 20) == ["Aa bb c", ". " + "d" * 13]


def test_split_js_lastindexof_boundary_semantics():
    # JS lastIndexOf(needle, fromIndex) accepts a match STARTING at fromIndex even
    # though it extends past. The \n\n starts exactly at index 10 (= limit): JS
    # finds it → clean paragraph split. A naive Python rfind(..., 0, limit) port
    # would miss it and split at the \n at index 6 instead.
    text = "abcdef\nghi\n\nzzzzzz"
    assert split_message(text, 10) == ["abcdef\nghi", "zzzzzz"]


def test_split_early_delimiters_rejected_then_hard_cut():
    # \n\n at index 2 and \n at 3 are both < 50% of the limit; no ". " → hard cut.
    # Internal newlines inside the cut chunk survive (strip only touches the ends).
    text = "ab\n\n" + "c" * 30
    assert split_message(text, 20) == ["ab\n\n" + "c" * 16, "c" * 14]


def test_split_hard_cut_mid_word_no_space_fallback():
    # There is NO word-boundary fallback — an unbroken run gets cut mid-"word".
    assert split_message("x" * 25, 10) == ["x" * 10, "x" * 10, "x" * 5]


def test_split_empty_trailing_remainder_dropped():
    # remaining strips to "" → JS `if (remaining)` falsiness drops it.
    assert split_message("aaaa aaaa\n\n   ", 10) == ["aaaa aaaa"]


# --- chunk_for_platform: item fan-out with per-platform limits ----------------------


def test_chunk_discord_uses_2000_limit():
    item = {**CTX, "formatted_response": "y" * (DISCORD_LIMIT + 1)}
    out = chunk_for_platform(item)
    assert [c["chunk"] for c in out] == ["y" * 2000, "y"]
    assert [c["chunk_index"] for c in out] == [0, 1]
    assert all(c["total_chunks"] == 2 for c in out)


def test_chunk_telegram_uses_4096_limit():
    # The same 2001-char text that split on Discord is a single Telegram chunk.
    item = {**CTX, "source_channel": "telegram", "formatted_response": "y" * 2001}
    out = chunk_for_platform(item)
    assert len(out) == 1 and out[0]["total_chunks"] == 1


def test_chunk_voice_falls_into_else_limit():
    # The JS ternary: discord → 2000, EVERYTHING else → 4096 (voice included).
    item = {**CTX, "source_channel": "voice", "formatted_response": "y" * 2001}
    assert len(chunk_for_platform(item)) == 1
    assert TELEGRAM_LIMIT == 4096 and DISCORD_LIMIT == 2000  # pin the constants


def test_chunk_operates_on_formatted_text():
    # Chunking happens POST-formatting — Telegram splits the HTML payload it will
    # actually send, not the source markdown.
    item = format_for_platform(
        {**CTX, "source_channel": "telegram", "polished_response": "**b**\n\n" + "x" * 4200}
    )
    out = chunk_for_platform(item)
    assert out[0]["chunk"].startswith("<b>b</b>")


def test_chunk_single_passes_through_unmodified():
    item = {**CTX, "formatted_response": "hi\n"}
    assert chunk_for_platform(item)[0]["chunk"] == "hi\n"  # no trim on the fast path


def test_chunk_spreads_item_fields():
    item = {**CTX, "formatted_response": "z" * 2001, "parse_mode": None}
    for c in chunk_for_platform(item):
        assert c["channel_id"] == CTX["channel_id"]
        assert c["person_id"] == CTX["person_id"]


def test_chunk_missing_text_yields_one_empty_chunk():
    # `formatted_response || ''` — one (empty) item still flows, same as n8n.
    out = chunk_for_platform({**CTX, "formatted_response": None})
    assert len(out) == 1 and out[0]["chunk"] == ""


# --- build_discord_body: JSON.stringify parity --------------------------------------


def test_discord_body_matches_json_stringify():
    # separators=(",", ":") → no spaces, byte-for-byte what JSON.stringify emitted.
    assert build_discord_body("hi") == '{"content":"hi"}'


def test_discord_body_keeps_utf8():
    # JSON.stringify emits real UTF-8, not \uXXXX escapes.
    assert build_discord_body("héllo — ok") == '{"content":"héllo — ok"}'


def test_discord_body_round_trips():
    chunk = 'she said "hi"\nthen left'
    assert json.loads(build_discord_body(chunk)) == {"content": chunk}


def test_discord_retry_policy_constants():
    # Quirk rule 2 (CLAUDE.md): DNS to discord.com transiently fails on this network
    # stack — the transport MUST retry 3x / 2000ms. Pinned so nobody "simplifies" it.
    assert DISCORD_SEND_MAX_TRIES == 3
    assert DISCORD_SEND_RETRY_WAIT_MS == 2000


# --- strip_markdown_for_voice: TTS text ----------------------------------------------


def voice(text: str) -> str:
    return strip_markdown_for_voice({"polished_response": text})["tts_text"]


def test_voice_fenced_code_deleted_not_unwrapped():
    # You don't read code aloud — fenced blocks vanish entirely.
    assert voice("before\n```js\ncode\n```\nafter") == "before\n\nafter"


def test_voice_emphasis_unwrapped():
    assert voice("**b** *i* __u__ _e_") == "b i u e"


def test_voice_headers_removed():
    assert voice("## Heading\ntext") == "Heading\ntext"


def test_voice_header_regex_unanchored_quirk():
    # The JS had no ^ anchor on `#{1,6}\s*` — it fires mid-line too. "C# and #tag"
    # loses both hashes AND the space after "C#". Faithful port of live behavior;
    # fix deliberately (anchor + test change), never by accident.
    assert voice("C# and #tag") == "Cand tag"


def test_voice_links_unwrapped_to_text():
    assert voice("see [Chip](https://example.com/a?b=c) now") == "see Chip now"


def test_voice_inline_code_unwrapped():
    assert voice("run `ls -la` now") == "run ls -la now"


def test_voice_bullets_and_numbers_removed():
    assert voice("- one\n- two\nplain") == "one\ntwo\nplain"
    assert voice("1. first\n2. second") == "first\nsecond"


def test_voice_collapses_3plus_newlines():
    assert voice("a\n\n\n\nb") == "a\n\nb"


def test_voice_signed_url_preserved_verbatim():
    # Quirk rule 5: no markdown pass touches a bare signed CDN URL — every query
    # param survives into the spoken text (awkward to hear, never corrupted).
    assert voice(f"Look: {SIGNED_URL}") == f"Look: {SIGNED_URL}"


def test_voice_underscore_pass_mangles_snake_case_quirk():
    # Known JS quirk, pinned: the _italic_ pass eats paired underscores anywhere,
    # so snake_case_words lose their underscores. Faithful to production.
    assert voice("a_b_c") == "abc"


def test_voice_reads_persisted_text_not_chunks():
    # Voice bypasses the chunker: it re-reads the FULL text from the persisted ctx
    # (quirk rule 1), ignoring formatted_response/chunk fields entirely.
    ctx = {
        **CTX,
        "polished_response": "**full** text",
        "formatted_response": "IGNORED",
        "chunk": "IGNORED TOO",
    }
    out = strip_markdown_for_voice(ctx)
    assert out["tts_text"] == "full text"
    assert out["person_id"] == CTX["person_id"]  # ctx spread through


# --- route_platform + Telegram parse-mode default -----------------------------------


@pytest.mark.parametrize("channel", ["discord", "telegram", "voice"])
def test_route_known_channels(channel):
    assert route_platform(channel) == channel


@pytest.mark.parametrize("channel", ["Discord", "sms", "", None])
def test_route_unknown_channel_raises(channel):
    # DELIBERATE DEVIATION: the n8n Switch silently dropped these (message just
    # never arrived). The port fails loudly — see route_platform's docstring.
    with pytest.raises(ValueError):
        route_platform(channel)


def test_telegram_parse_mode_defaults_to_html():
    # Quirk rule 4: `$json.parse_mode || 'HTML'` at the Telegram node — Telegram is
    # effectively ALWAYS HTML mode, even if the formatter set None.
    assert effective_telegram_parse_mode({"parse_mode": None}) == "HTML"
    assert effective_telegram_parse_mode({}) == "HTML"
    assert effective_telegram_parse_mode({"parse_mode": "MarkdownV2"}) == "MarkdownV2"


# --- End-to-end: formatter → chunker → body (the whole Discord leg) -----------------


def test_pipeline_discord_end_to_end():
    long_text = ("Paragraph one is short.\n\n" + "word " * 500).strip()
    ctx = {**CTX, "polished_response": long_text}
    item = format_for_platform(ctx)
    chunks = chunk_for_platform(item)
    assert len(chunks) > 1
    # Every chunk fits the platform limit and round-trips through the body builder.
    for c in chunks:
        assert len(c["chunk"]) <= DISCORD_LIMIT
        assert json.loads(build_discord_body(c["chunk"])) == {"content": c["chunk"]}
    # Order metadata is intact for the transport's sequential send loop.
    assert [c["chunk_index"] for c in chunks] == list(range(len(chunks)))
    assert all(c["total_chunks"] == len(chunks) for c in chunks)
