"""Golden tests for the vision tool port — every CLAUDE.md quirk pinned in pytest.

These are the tests the n8n version never had: the Tool: Image node could only be
verified by uploading a real image to Discord and watching the execution log. Each
test below pins one behavior extracted from the live instance (Sonnet lGjy9sHqbwOh7J50,
cross-checked against Opus and Gemini — all three carry byte-identical nodes).
"""

import json

import pytest

from aerys_v2.channels.vision_ladder import (
    DEFAULT_PROMPT,
    OPENROUTER_CHAT_URL,
    TOOL_DESCRIPTION,
    TOOL_NAME,
    VISION_MAX_TOKENS,
    VISION_MODEL,
    attachment_mime,
    build_auth_headers,
    build_vision_body,
    build_vision_request,
    detect_media_kind,
    explain_vision_http_error,
    extract_description,
    looks_like_signed_discord_url,
)

# A realistic signed Discord CDN URL — the ?ex=&is=&hm= params are the signature and
# they MUST survive every code path untouched.
SIGNED_URL = (
    "https://cdn.discordapp.com/attachments/1421592889687015424/1234567890/photo.png"
    "?ex=66aa11bb&is=66a9c02b&hm=deadbeefcafe0123456789abcdef"
)


# ---------------------------------------------------------------------------
# Golden body — the jsonBody expression, byte-for-byte.
# ---------------------------------------------------------------------------


def test_body_matches_n8n_json_body_golden():
    # Exactly what JSON.stringify({...}) produced in the live node, as a Python dict.
    assert build_vision_body(SIGNED_URL, "What breed is this dog?") == {
        "model": "anthropic/claude-sonnet-4.6",
        "max_tokens": 1024,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": SIGNED_URL}},
                    {"type": "text", "text": "What breed is this dog?"},
                ],
            }
        ],
    }


def test_image_part_ordered_before_text_part():
    # The live node puts image_url first, text second — kept identical so V1/V2
    # responses stay comparable.
    content = build_vision_body(SIGNED_URL, "hi")["messages"][0]["content"]
    assert [part["type"] for part in content] == ["image_url", "text"]


def test_single_user_message():
    body = build_vision_body(SIGNED_URL)
    assert len(body["messages"]) == 1
    assert body["messages"][0]["role"] == "user"


def test_model_is_the_undated_slug():
    # CLAUDE.md quirk: OpenRouter needs "anthropic/claude-sonnet-4.6", NOT the dated
    # ID format. A date suffix sneaking in (e.g. "-20260115") must fail this test.
    assert VISION_MODEL == "anthropic/claude-sonnet-4.6"
    assert build_vision_body(SIGNED_URL)["model"] == "anthropic/claude-sonnet-4.6"


def test_max_tokens_is_1024():
    assert VISION_MAX_TOKENS == 1024
    assert build_vision_body(SIGNED_URL)["max_tokens"] == 1024


# ---------------------------------------------------------------------------
# Signed-URL preservation — never strip, never re-encode, never download.
# ---------------------------------------------------------------------------


def test_signed_url_passes_through_verbatim():
    body = build_vision_body(SIGNED_URL)
    url_out = body["messages"][0]["content"][0]["image_url"]["url"]
    assert url_out == SIGNED_URL  # full string, signature params and all
    assert "ex=" in url_out and "is=" in url_out and "hm=" in url_out


def test_signed_url_survives_json_serialization():
    # The transport will json-encode the body; the signature must survive that too
    # (no percent-encoding, no truncation at the '?').
    wire = json.dumps(build_vision_body(SIGNED_URL))
    assert SIGNED_URL in wire


def test_url_with_encoded_chars_not_mangled():
    tricky = "https://cdn.discordapp.com/attachments/1/2/a%20b.png?ex=1&is=2&hm=ab%2Fcd"
    url_out = build_vision_body(tricky)["messages"][0]["content"][0]["image_url"]["url"]
    assert url_out == tricky


def test_outer_whitespace_trimmed_but_query_kept():
    # LLM tool args sometimes arrive with stray whitespace; trimming edges is safe,
    # touching the query string is not.
    url_out = build_vision_body(f"  {SIGNED_URL}\n")["messages"][0]["content"][0][
        "image_url"
    ]["url"]
    assert url_out == SIGNED_URL


# ---------------------------------------------------------------------------
# Prompt default — the $fromAI hint promoted to a real default.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("empty_prompt", [None, "", "   ", "\n\t"])
def test_empty_prompt_falls_back_to_default(empty_prompt):
    text = build_vision_body(SIGNED_URL, empty_prompt)["messages"][0]["content"][1]["text"]
    assert text == DEFAULT_PROMPT == "Describe this image in detail"


def test_custom_prompt_used_verbatim():
    text = build_vision_body(SIGNED_URL, "Read the whiteboard")["messages"][0]["content"][
        1
    ]["text"]
    assert text == "Read the whiteboard"


@pytest.mark.parametrize("bad_url", ["", "   ", None])
def test_missing_image_url_rejected(bad_url):
    with pytest.raises(ValueError):
        build_vision_body(bad_url)


# ---------------------------------------------------------------------------
# Request envelope — node parameters + credential.
# ---------------------------------------------------------------------------


def test_request_envelope_golden():
    request = build_vision_request(SIGNED_URL, "hi", api_key="sk-or-test")
    assert request["method"] == "POST"
    assert request["url"] == OPENROUTER_CHAT_URL
    assert request["url"] == "https://openrouter.ai/api/v1/chat/completions"
    assert request["headers"] == {"Authorization": "Bearer sk-or-test"}
    assert request["json"] == build_vision_body(SIGNED_URL, "hi")


def test_auth_header_is_plain_bearer():
    # n8n cred gvgPllzFhLSds5Qv "OpenRouter Header Auth" -> Authorization: Bearer <key>
    assert build_auth_headers("sk-or-abc") == {"Authorization": "Bearer sk-or-abc"}


@pytest.mark.parametrize("bad_key", ["", "  ", None])
def test_missing_api_key_rejected(bad_key):
    with pytest.raises(ValueError):
        build_auth_headers(bad_key)


# ---------------------------------------------------------------------------
# Tool identity — name + description are load-bearing, not cosmetic.
# ---------------------------------------------------------------------------


def test_tool_name_matches_system_prompt_reference():
    # The tier system prompts say "use the matching tool (image, ...)". CLAUDE.md
    # quirk: name mismatch -> the LLM hallucinates having called the tool.
    assert TOOL_NAME == "image"


def test_tool_description_keeps_the_concrete_triggers():
    # CLAUDE.md quirk: specificity beats generality. The CDN trigger string and the
    # "ZERO visual capability" framing are what make the model call the tool.
    assert "https://cdn.discordapp.com/attachments/" in TOOL_DESCRIPTION
    assert "ZERO visual capability" in TOOL_DESCRIPTION
    assert "CALL THIS TOOL IMMEDIATELY" in TOOL_DESCRIPTION


# ---------------------------------------------------------------------------
# Media dispatch — extension FIRST, image is the catch-all.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("url", "kind"),
    [
        ("https://cdn.discordapp.com/attachments/1/2/report.pdf", "pdf"),
        ("https://cdn.discordapp.com/attachments/1/2/resume.docx", "docx"),
        ("https://cdn.discordapp.com/attachments/1/2/notes.txt", "text_file"),
    ],
)
def test_document_extensions_route_positively(url, kind):
    assert detect_media_kind(url) == kind


def test_extension_detected_despite_signature_query_params():
    # The '?ex=...' tail must not hide the extension — classification parses the URL
    # path, a naive url.endswith('.pdf') would miss this and mis-route to image.
    signed_pdf = "https://cdn.discordapp.com/attachments/1/2/report.pdf?ex=1&is=2&hm=3"
    assert detect_media_kind(signed_pdf) == "pdf"


def test_extension_check_is_case_insensitive():
    assert detect_media_kind("https://x.example/REPORT.PDF") == "pdf"


def test_extension_beats_image_catch_all_ordering():
    # THE ordering quirk: a Discord attachment ending .pdf must never fall through to
    # the image branch just because it lives on the image-heavy CDN.
    assert detect_media_kind(SIGNED_URL.replace("photo.png", "doc.pdf")) == "pdf"


def test_extension_wins_over_conflicting_mime():
    # Extension check runs BEFORE the MIME hint — same precedence as the n8n node.
    assert detect_media_kind("https://x.example/file.pdf", "image/png") == "pdf"


@pytest.mark.parametrize(
    "url",
    [
        SIGNED_URL,  # .png
        "https://cdn.discordapp.com/attachments/1/2/pic.jpg?ex=1&is=2&hm=3",
        "https://cdn.discordapp.com/attachments/1/2/anim.webp",
        "https://cdn.discordapp.com/attachments/1/2/mystery",  # no extension
        "https://cdn.discordapp.com/attachments/1/2/data.zip",  # unknown extension
    ],
)
def test_everything_else_falls_through_to_image(url):
    # Image is the CATCH-ALL, never a positive match — .zip lands here too, exactly
    # like the live Detect Media Type node.
    assert detect_media_kind(url) == "image"


def test_mime_hint_rescues_extensionless_documents():
    url = "https://cdn.discordapp.com/attachments/1/2/paste"
    assert detect_media_kind(url, "application/pdf") == "pdf"
    assert detect_media_kind(url, "text/plain; charset=utf-8") == "text_file"


def test_attachment_mime_reads_type_not_content_type():
    # CLAUDE.md quirk: the guild adapter stores MIME under `type`. Reading
    # `content_type` returns None and mis-routes everything.
    attachment = {"type": "application/pdf", "content_type": "WRONG-FIELD"}
    assert attachment_mime(attachment) == "application/pdf"
    assert attachment_mime({"content_type": "application/pdf"}) is None


# ---------------------------------------------------------------------------
# Response handling — the agent's read of choices[0].message.content.
# ---------------------------------------------------------------------------


def test_extract_description_happy_path():
    response = {
        "id": "gen-123",
        "choices": [{"message": {"role": "assistant", "content": "A corgi on a beach."}}],
    }
    assert extract_description(response) == "A corgi on a beach."


def test_extract_description_surfaces_openrouter_error_payload():
    response = {"error": {"message": "Invalid API key", "code": 401}}
    with pytest.raises(ValueError, match="Invalid API key"):
        extract_description(response)


@pytest.mark.parametrize(
    "malformed",
    [
        {},
        {"choices": []},
        {"choices": [{}]},
        {"choices": [{"message": {}}]},
        {"choices": [{"message": {"content": None}}]},
        {"choices": [{"message": {"content": "   "}}]},
    ],
)
def test_extract_description_fails_loudly_on_malformed(malformed):
    # In n8n a missing field became `undefined` flowing silently downstream; the
    # port raises instead.
    with pytest.raises(ValueError):
        extract_description(malformed)


# ---------------------------------------------------------------------------
# Fail-fast on expired signatures — the useful 403/404 message.
# ---------------------------------------------------------------------------


def test_signed_discord_url_detection():
    assert looks_like_signed_discord_url(SIGNED_URL)
    assert looks_like_signed_discord_url(
        "https://media.discordapp.net/attachments/1/2/a.png?ex=1&is=2&hm=3"
    )
    # Missing signature params -> not signed (already stripped upstream, perhaps).
    assert not looks_like_signed_discord_url(SIGNED_URL.split("?")[0])
    # Same params on a non-Discord host -> not a Discord signature.
    assert not looks_like_signed_discord_url("https://imgur.example/a.png?ex=1&is=2&hm=3")


@pytest.mark.parametrize("status", [403, 404])
def test_403_404_on_signed_url_explains_expiry(status):
    message = explain_vision_http_error(status, SIGNED_URL)
    assert "expired" in message
    assert str(status) in message


def test_403_on_plain_url_stays_generic():
    # No signature -> no expiry theory; don't invent explanations.
    message = explain_vision_http_error(403, "https://imgur.example/cat.png")
    assert "expired" not in message
    assert "403" in message


def test_other_statuses_stay_generic_even_on_signed_urls():
    message = explain_vision_http_error(500, SIGNED_URL)
    assert "expired" not in message
    assert "500" in message


# ---------------------------------------------------------------------------
# Purity — this module must never do I/O; the transport layer owns the network.
# ---------------------------------------------------------------------------


def test_module_imports_no_network_or_io_libraries():
    import aerys_v2.channels.vision_ladder as module

    source = open(module.__file__, encoding="utf-8").read()
    for forbidden in ("import requests", "import httpx", "import aiohttp", "urlopen"):
        assert forbidden not in source
