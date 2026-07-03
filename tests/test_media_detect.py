"""Golden tests for media detection — every quirk rule from the Code node, pinned.

These are offline and pure (no network, no n8n). Each test names the V1 failure it
guards against: signed-URL stripping, guild-adapter `type` field, document-extension-
before-image-catch-all ordering, JSON-string attachments from toolWorkflow callers.
"""

from aerys_v2.channels.media_detect import (
    detect_media,
    filename_from_url,
    find_youtube_id,
    parse_attachments,
)

# A realistic signed Discord CDN URL — the ?ex=&is=&hm= params are the expiring
# signature. If any test sees these stripped, the port broke quirk rule 1.
SIGNED_PNG = (
    "https://cdn.discordapp.com/attachments/123/456/photo.png"
    "?ex=66aa11bb&is=66aa00aa&hm=deadbeef"
)
SIGNED_PDF = (
    "https://cdn.discordapp.com/attachments/123/457/report.pdf"
    "?ex=66aa11bb&is=66aa00aa&hm=cafef00d"
)


# --- Step 1: attachments array ---------------------------------------------------


def test_discord_image_attachment_keeps_signed_url():
    out = detect_media(
        {"attachments": [{"content_type": "image/png", "url": SIGNED_PNG, "filename": "photo.png"}]}
    )
    assert out["mediaType"] == "image"
    assert out["mediaUrl"] == SIGNED_PNG  # signature params intact — quirk rule 1
    assert out["mediaUrls"] == [SIGNED_PNG]
    assert out["filename"] == "photo.png"
    assert out["needsTelegramDownload"] is False


def test_guild_adapter_type_field_detected():
    # Guild adapter stores MIME under `type` (quirk rule 2) — no content_type at all.
    out = detect_media({"attachments": [{"type": "image/jpeg", "url": SIGNED_PNG}]})
    assert out["mediaType"] == "image"


def test_mime_missing_filename_extension_fallback():
    out = detect_media({"attachments": [{"filename": "Vacation.JPG", "url": SIGNED_PNG}]})
    assert out["mediaType"] == "image"
    assert out["filename"] == "Vacation.JPG"  # original case preserved in output


def test_multi_image_collects_all_image_urls_and_skips_non_images():
    url2 = "https://cdn.discordapp.com/attachments/123/458/two.webp?ex=1&is=2&hm=3"
    out = detect_media(
        {
            "attachments": [
                {"content_type": "image/png", "url": SIGNED_PNG},
                {"content_type": "application/pdf", "url": SIGNED_PDF},  # rides along, skipped
                {"content_type": "image/webp", "url": url2},
            ]
        }
    )
    assert out["mediaType"] == "image"
    assert out["mediaUrls"] == [SIGNED_PNG, url2]  # both images, PDF not collected


def test_classification_is_by_first_attachment_only():
    # attachment[0] is a PDF → the whole message is 'pdf' even with an image behind it.
    out = detect_media(
        {
            "attachments": [
                {"content_type": "application/pdf", "url": SIGNED_PDF, "filename": "report.pdf"},
                {"content_type": "image/png", "url": SIGNED_PNG},
            ]
        }
    )
    assert out["mediaType"] == "pdf"
    assert out["mediaUrl"] == SIGNED_PDF
    assert out["mediaUrls"] == []  # image branch never ran


def test_telegram_platform_flags_download():
    out = detect_media(
        {
            "platform": "telegram",
            "attachments": [{"mime_type": "image/jpeg", "file_id": "tg-file-42"}],
        }
    )
    assert out["mediaType"] == "image"
    assert out["needsTelegramDownload"] is True
    assert out["telegramFileId"] == "tg-file-42"
    assert out["fileMimeType"] == "image/jpeg"
    assert out["mediaUrls"] == []  # multi-image collection is Discord-branch only


def test_file_id_alone_triggers_telegram_branch():
    # JS: `platform === 'telegram' || att.file_id` — file_id wins even on 'discord'.
    out = detect_media({"attachments": [{"filename": "pic.png", "file_id": "tg-9"}]})
    assert out["needsTelegramDownload"] is True
    assert out["fileMimeType"] == "image/jpeg"  # no MIME anywhere → default


def test_pdf_docx_txt_attachment_classification():
    docx_mime = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    cases = [
        ({"content_type": "application/pdf", "url": "u"}, "pdf", "document.pdf"),
        ({"filename": "notes.PDF", "url": "u"}, "pdf", "notes.PDF"),
        ({"content_type": docx_mime, "url": "u"}, "docx", "document.docx"),
        ({"file_name": "spec.docx", "url": "u"}, "docx", "spec.docx"),
        ({"content_type": "text/plain", "url": "u"}, "txt", "document.txt"),
        ({"filename": "log.txt", "url": "u"}, "txt", "log.txt"),
    ]
    for att, expected_type, expected_name in cases:
        out = detect_media({"attachments": [att]})
        assert out["mediaType"] == expected_type, att
        assert out["filename"] == expected_name, att


def test_proxy_url_fallback_when_url_missing():
    out = detect_media(
        {"attachments": [{"content_type": "image/png", "proxy_url": SIGNED_PNG}]}
    )
    assert out["mediaUrl"] == SIGNED_PNG


def test_unrecognized_attachment_mime_stays_unknown():
    out = detect_media({"attachments": [{"content_type": "audio/mpeg", "url": "u"}]})
    assert out["mediaType"] == "unknown"


def test_unknown_attachment_falls_through_to_youtube_scan():
    out = detect_media(
        {
            "attachments": [{"content_type": "audio/mpeg", "url": "u"}],
            "content": "listen: https://youtu.be/dQw4w9WgXcQ",
        }
    )
    assert out["mediaType"] == "youtube"
    assert out["videoId"] == "dQw4w9WgXcQ"


# --- Attachments arriving as JSON string / garbage (quirk rule 4) -----------------


def test_attachments_as_json_string():
    import json

    raw = json.dumps([{"content_type": "image/png", "url": SIGNED_PNG}])
    out = detect_media({"attachments": raw})
    assert out["mediaType"] == "image"
    assert out["mediaUrl"] == SIGNED_PNG


def test_attachments_invalid_json_string_becomes_empty():
    out = detect_media({"attachments": "not json at all ["})
    assert out["mediaType"] == "unknown"


def test_attachments_non_array_garbage_becomes_empty():
    for garbage in ({"url": "x"}, 42, True, "\"a string\""):  # last one: valid JSON, not a list
        out = detect_media({"attachments": garbage})
        assert out["mediaType"] == "unknown", garbage


def test_parse_attachments_drops_non_dict_entries():
    assert parse_attachments('[null, "str", {"url": "u"}]') == [{}, {}, {"url": "u"}]


# --- Step 2: YouTube scan across content → context → message_content --------------


def test_youtube_long_link_in_content():
    out = detect_media({"content": "watch https://www.youtube.com/watch?v=abc123XYZ_-"})
    assert out["mediaType"] == "youtube"
    assert out["videoId"] == "abc123XYZ_-"


def test_youtube_short_link_in_context():
    out = detect_media({"context": "earlier: https://youtu.be/dQw4w9WgXcQ ok"})
    assert (out["mediaType"], out["videoId"]) == ("youtube", "dQw4w9WgXcQ")


def test_youtube_in_message_content():
    out = detect_media({"message_content": "https://youtube.com/watch?v=dQw4w9WgXcQ"})
    assert (out["mediaType"], out["videoId"]) == ("youtube", "dQw4w9WgXcQ")


def test_youtube_content_wins_over_context():
    out = detect_media(
        {
            "content": "https://youtu.be/AAAAAAAAAAA",
            "context": "https://youtu.be/BBBBBBBBBBB",
        }
    )
    assert out["videoId"] == "AAAAAAAAAAA"


def test_youtube_id_must_be_exactly_11_chars():
    # 10-char tail → no match (boundary of the {11} quantifier).
    assert find_youtube_id("https://youtu.be/short10ch") is None
    # 12+ chars → the first 11 are captured, same as the JS regex.
    assert find_youtube_id("https://youtu.be/dQw4w9WgXcQ9") == "dQw4w9WgXcQ"


# --- Step 3: bare-URL fallback in `query` -----------------------------------------


def test_query_discord_cdn_pdf_routes_to_pdf_not_image():
    # THE ordering quirk (rule 3): a CDN URL matches the image catch-all too, so the
    # .pdf check must fire first or every document becomes an image.
    out = detect_media({"query": SIGNED_PDF})
    assert out["mediaType"] == "pdf"
    assert out["mediaUrl"] == SIGNED_PDF  # signature intact
    assert out["filename"] == "report.pdf"  # ?ex=... trimmed from the NAME only


def test_query_docx_and_txt_before_image_catch_all():
    for ext, expected in (("docx", "docx"), ("txt", "txt")):
        url = f"https://cdn.discordapp.com/attachments/1/2/file.{ext}?ex=a&hm=b"
        out = detect_media({"query": url})
        assert out["mediaType"] == expected, ext
        assert out["mediaUrl"] == url, ext


def test_query_extension_at_end_without_query_string():
    # The (\?|$) boundary: extension flush at end-of-string also matches.
    out = detect_media({"query": "https://example.com/paper.pdf"})
    assert out["mediaType"] == "pdf"
    assert out["filename"] == "paper.pdf"


def test_query_cdn_url_without_extension_is_image_catch_all():
    url = "https://cdn.discordapp.com/attachments/1/2/blob?ex=a&is=b&hm=c"
    out = detect_media({"query": url})
    assert out["mediaType"] == "image"
    assert out["mediaUrl"] == url  # full signed URL, untouched
    assert out["mediaUrls"] == [url]
    assert out["filename"] == "blob"


def test_query_image_extension_on_non_cdn_host():
    out = detect_media({"query": "look at https://example.com/cat.JPEG?w=200 please"})
    assert out["mediaType"] == "image"
    assert out["mediaUrl"] == "https://example.com/cat.JPEG?w=200"
    assert out["filename"] == "cat.JPEG"


def test_query_youtube_wins_over_extensions():
    out = detect_media({"query": "https://youtu.be/dQw4w9WgXcQ and also file.pdf"})
    assert out["mediaType"] == "youtube"


def test_query_with_extension_but_no_url_uses_raw_query():
    # JS: `mediaUrl = urlMatch ? urlMatch[1] : q` — no http URL → the whole query.
    out = detect_media({"query": "summarize notes.txt"})
    assert out["mediaType"] == "txt"
    assert out["mediaUrl"] == "summarize notes.txt"
    # No '/' in the string → split('/').pop() is the WHOLE string (JS parity).
    assert out["filename"] == "summarize notes.txt"


def test_query_unrecognized_url_stays_unknown():
    out = detect_media({"query": "read https://example.com/page.html"})
    assert out["mediaType"] == "unknown"


def test_attachment_classification_beats_query_scan():
    # Step 1 already resolved → step 3 never runs, even with a PDF URL in query.
    out = detect_media(
        {
            "attachments": [{"content_type": "image/png", "url": SIGNED_PNG}],
            "query": SIGNED_PDF,
        }
    )
    assert out["mediaType"] == "image"


# --- Output shape & passthrough ----------------------------------------------------


def test_empty_payload_yields_stable_unknown_shape():
    out = detect_media({})
    assert out == {
        "mediaType": "unknown",
        "mediaUrl": None,
        "mediaUrls": [],
        "videoId": None,
        "filename": None,
        "needsTelegramDownload": False,
        "telegramFileId": None,
        "fileMimeType": None,
        "platform": "discord",  # default platform
        "content": "",
        "context": "",
        "person_id": None,
        "source_channel": None,
        "conversation_privacy": None,
    }


def test_passthrough_fields_survive():
    out = detect_media(
        {
            "query": SIGNED_PDF,
            "person_id": "6e6bcbed-03ef-4d17-95d2-89c467414335",
            "source_channel": "discord_dm",
            "conversation_privacy": "private",
            "platform": "telegram",
            "context": "ctx",
        }
    )
    assert out["person_id"] == "6e6bcbed-03ef-4d17-95d2-89c467414335"
    assert out["source_channel"] == "discord_dm"
    assert out["conversation_privacy"] == "private"
    assert out["platform"] == "telegram"
    assert out["context"] == "ctx"


def test_content_falls_back_to_query():
    # JS: `input.content || input.query || ''` — the toolWorkflow-collapse case.
    out = detect_media({"query": "hello"})
    assert out["content"] == "hello"


def test_filename_from_url_trailing_slash_defaults():
    assert filename_from_url("https://example.com/dir/", "document") == "document"
