"""Offline tests for the media tools — fake OpenRouter + fake CDN (httpx.MockTransport).

What these prove: signed Discord CDN URLs travel VERBATIM (query params intact —
the split('?')[0] regression), document extensions route BEFORE the image
catch-all, extraction works on real pdf/docx/txt bytes, and every failure mode
comes back as an honest string, never an exception (the ToolNode contract).
"""

from io import BytesIO

import httpx

from aerys_v2.channels.vision_ladder import VISION_MODEL
from aerys_v2.tools.media import (
    DOC_MAX_CHARS,
    TRANSCRIPT_MAX_CHARS,
    build_analyze_image_tool,
    build_media_tools,
    build_read_document_tool,
    build_youtube_summary_tool,
)

# A realistic signed Discord CDN URL — the ?ex=&is=&hm= params are the signature
# that MUST survive end-to-end (quirk rule 1).
SIGNED_IMAGE_URL = (
    "https://cdn.discordapp.com/attachments/123/456/photo.png"
    "?ex=66aa11bb&is=66aa00aa&hm=deadbeefcafe"
)
SIGNED_PDF_URL = (
    "https://cdn.discordapp.com/attachments/123/456/report.pdf"
    "?ex=66aa11bb&is=66aa00aa&hm=deadbeefcafe"
)


# ---- fixture bytes ---------------------------------------------------------------

def make_pdf_bytes(text: str) -> bytes:
    """A minimal valid one-page PDF whose page shows `text` (pypdf-extractable)."""
    stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode()
    objs = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R "
        b"/Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length %d >>\nstream\n%s\nendstream" % (len(stream), stream),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for i, body in enumerate(objs, start=1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % i + body + b"\nendobj\n"
    xref_pos = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objs) + 1)
    for off in offsets:
        out += b"%010d 00000 n \n" % off
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (
        len(objs) + 1,
        xref_pos,
    )
    return bytes(out)


def make_docx_bytes(text: str) -> bytes:
    from docx import Document

    buf = BytesIO()
    doc = Document()
    doc.add_paragraph(text)
    doc.save(buf)
    return buf.getvalue()


# ---- fakes -----------------------------------------------------------------------

class FakeOpenRouter:
    """Records every request; answers /chat/completions like OpenRouter would."""

    def __init__(self, reply: str = "A cat on a keyboard.", status: int = 200,
                 payload: dict | None = None):
        self.requests: list[httpx.Request] = []
        self.reply = reply
        self.status = status
        self.payload = payload  # overrides the default completion shape

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if self.status >= 400:
            return httpx.Response(self.status, json={"error": {"message": "nope"}})
        body = self.payload if self.payload is not None else {
            "choices": [{"message": {"role": "assistant", "content": self.reply}}]
        }
        return httpx.Response(200, json=body)

    def client(self) -> httpx.Client:
        return httpx.Client(transport=httpx.MockTransport(self.handler))

    def sent_json(self, idx: int = -1) -> dict:
        import json
        return json.loads(self.requests[idx].content)


class FakeCDN:
    """Serves document bytes; records the FULL URL each fetch used."""

    def __init__(self, content: bytes, status: int = 200):
        self.content = content
        self.status = status
        self.urls: list[str] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.urls.append(str(request.url))
        if self.status >= 400:
            return httpx.Response(self.status)
        return httpx.Response(200, content=self.content)

    def client(self) -> httpx.Client:
        return httpx.Client(transport=httpx.MockTransport(self.handler))


def make_vision(openrouter: FakeOpenRouter):
    return build_analyze_image_tool(api_key="or-key", client=openrouter.client())


def make_reader(cdn: FakeCDN):
    return build_read_document_tool(client=cdn.client())


# ---- analyze_image: vision via OpenRouter ----------------------------------------

def test_vision_call_shape_and_reply():
    router = FakeOpenRouter(reply="Two screenshots of a stack trace.")
    out = make_vision(router).invoke({"url": SIGNED_IMAGE_URL, "question": "what error?"})
    assert out == "Two screenshots of a stack trace."
    req = router.requests[0]
    assert str(req.url) == "https://openrouter.ai/api/v1/chat/completions"
    assert req.headers["authorization"] == "Bearer or-key"
    body = router.sent_json()
    assert body["model"] == VISION_MODEL  # one constant, no ladder
    parts = body["messages"][0]["content"]
    assert parts[0]["type"] == "image_url" and parts[1]["text"] == "what error?"


def test_vision_url_signature_preserved_verbatim():
    # THE regression test: the signed URL reaches the API byte-identical —
    # query params intact, nothing stripped, nothing re-encoded (quirk rule 1).
    router = FakeOpenRouter()
    make_vision(router).invoke({"url": SIGNED_IMAGE_URL, "question": ""})
    sent_url = router.sent_json()["messages"][0]["content"][0]["image_url"]["url"]
    assert sent_url == SIGNED_IMAGE_URL
    assert "?ex=66aa11bb&is=66aa00aa&hm=deadbeefcafe" in sent_url


def test_vision_empty_question_gets_default_prompt():
    router = FakeOpenRouter()
    make_vision(router).invoke({"url": SIGNED_IMAGE_URL})
    assert router.sent_json()["messages"][0]["content"][1]["text"] == (
        "Describe this image in detail"
    )


def test_vision_403_on_signed_url_explains_expired_signature():
    out = make_vision(FakeOpenRouter(status=403)).invoke(
        {"url": SIGNED_IMAGE_URL, "question": "?"}
    )
    assert "expired" in out and "never strip the query string" in out


def test_vision_unreachable_and_malformed_are_honest_strings():
    def boom(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no route")

    dead = build_analyze_image_tool(
        api_key="or-key", client=httpx.Client(transport=httpx.MockTransport(boom))
    )
    assert "unreachable" in dead.invoke({"url": SIGNED_IMAGE_URL, "question": ""})
    # malformed completion payload -> relayed as words, never raised
    mangled = FakeOpenRouter(payload={"choices": []})
    assert "malformed" in make_vision(mangled).invoke({"url": SIGNED_IMAGE_URL})
    assert "nothing to look at" in make_vision(FakeOpenRouter()).invoke({"url": "  "})


def test_vision_custom_base_url():
    router = FakeOpenRouter()
    tool = build_analyze_image_tool(
        api_key="or-key", base_url="http://local-llm:4000/v1/", client=router.client()
    )
    tool.invoke({"url": SIGNED_IMAGE_URL})
    assert str(router.requests[0].url) == "http://local-llm:4000/v1/chat/completions"


# ---- read_document: pdf / docx / txt ---------------------------------------------

def test_read_pdf_extracts_text_from_signed_url():
    cdn = FakeCDN(make_pdf_bytes("Hello from the PDF fixture"))
    out = make_reader(cdn).invoke({"url": SIGNED_PDF_URL})
    assert "Hello from the PDF fixture" in out
    assert out.startswith("Contents of report.pdf:")  # display name, params trimmed
    # signature preservation on the FETCH side: the full signed URL was requested
    assert cdn.urls == [SIGNED_PDF_URL]


def test_read_docx_extracts_text():
    cdn = FakeCDN(make_docx_bytes("Quarterly notes from the DOCX fixture"))
    out = make_reader(cdn).invoke({"url": "https://cdn.discordapp.com/attachments/1/2/notes.docx?ex=a&is=b&hm=c"})
    assert "Quarterly notes from the DOCX fixture" in out


def test_read_txt_decodes_plain_text():
    cdn = FakeCDN("line one\nline two — with unicode".encode())
    out = make_reader(cdn).invoke({"url": "https://example.com/readme.txt"})
    assert "line two — with unicode" in out


def test_read_document_truncates_long_text():
    cdn = FakeCDN(b"x" * (DOC_MAX_CHARS + 500))
    out = make_reader(cdn).invoke({"url": "https://example.com/big.txt"})
    assert f"[truncated at {DOC_MAX_CHARS} characters" in out
    assert "x" * (DOC_MAX_CHARS + 1) not in out


def test_extension_beats_image_catchall():
    # THE ordering regression (quirk rule 3): a signed CDN URL ending .pdf?ex=...
    # matches the image catch-all pattern too — it must route as a DOCUMENT.
    cdn = FakeCDN(make_pdf_bytes("routed as pdf"))
    out = make_reader(cdn).invoke({"url": SIGNED_PDF_URL})
    assert "routed as pdf" in out
    # ...while a CDN URL with NO document extension is the image branch's —
    # read_document refuses honestly and points at analyze_image, no fetch.
    cdn2 = FakeCDN(b"")
    out2 = make_reader(cdn2).invoke({"url": SIGNED_IMAGE_URL})
    assert "analyze_image" in out2 and cdn2.urls == []


def test_read_document_redirects_youtube_links():
    cdn = FakeCDN(b"")
    out = make_reader(cdn).invoke({"url": "https://youtu.be/dQw4w9WgXcQ"})
    assert "youtube_summary" in out and cdn.urls == []


def test_read_document_expired_signature_and_plain_404():
    out = make_reader(FakeCDN(b"", status=404)).invoke({"url": SIGNED_PDF_URL})
    assert "expired" in out and "never strip the query string" in out
    out = make_reader(FakeCDN(b"", status=404)).invoke({"url": "https://example.com/gone.pdf"})
    assert "HTTP 404" in out and "expired" not in out


def test_read_document_corrupt_pdf_is_honest_string_not_exception():
    cdn = FakeCDN(b"this is not a pdf at all")
    out = make_reader(cdn).invoke({"url": "https://example.com/broken.pdf"})
    assert "couldn't extract" in out


def test_read_document_unreachable_and_empty_url():
    def boom(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no route")

    dead = build_read_document_tool(client=httpx.Client(transport=httpx.MockTransport(boom)))
    assert "Couldn't fetch" in dead.invoke({"url": "https://example.com/a.pdf"})
    assert "nothing to read" in dead.invoke({"url": ""})


# ---- youtube_summary: transcript -> LLM ------------------------------------------

def make_yt(router: FakeOpenRouter, transcript="welcome to the video about seams"):
    fetched: list[str] = []

    def fetcher(video_id: str) -> str:
        fetched.append(video_id)
        if isinstance(transcript, Exception):
            raise transcript
        return transcript

    tool = build_youtube_summary_tool(
        api_key="or-key", client=router.client(), transcript_fetcher=fetcher
    )
    return tool, fetched


def test_youtube_summary_happy_path_long_and_short_links():
    router = FakeOpenRouter(reply="It's about dependency seams.")
    tool, fetched = make_yt(router)
    out = tool.invoke({"url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ"})
    assert out == "It's about dependency seams."
    out = tool.invoke({"url": "https://youtu.be/dQw4w9WgXcQ"})
    assert out == "It's about dependency seams."
    assert fetched == ["dQw4w9WgXcQ", "dQw4w9WgXcQ"]  # the 11-char ID, both forms
    body = router.sent_json(0)
    assert body["model"] == VISION_MODEL
    assert "seams" in body["messages"][0]["content"]  # transcript rode the prompt


def test_youtube_summary_truncates_long_transcripts():
    router = FakeOpenRouter()
    tool, _ = make_yt(router, transcript="y" * (TRANSCRIPT_MAX_CHARS + 999))
    tool.invoke({"url": "https://youtu.be/dQw4w9WgXcQ"})
    prompt = router.sent_json()["messages"][0]["content"]
    assert "Transcript truncated" in prompt
    assert "y" * (TRANSCRIPT_MAX_CHARS + 1) not in prompt


def test_youtube_summary_failure_modes_are_honest_strings():
    router = FakeOpenRouter()
    # not a youtube link at all -> no fetcher call, no HTTP
    tool, fetched = make_yt(router)
    out = tool.invoke({"url": "https://example.com/watch?v=nope"})
    assert "doesn't look like a YouTube link" in out and fetched == []
    # captions disabled -> the fetcher's raise becomes words (ToolNode contract)
    tool, _ = make_yt(router, transcript=RuntimeError("Subtitles are disabled"))
    out = tool.invoke({"url": "https://youtu.be/dQw4w9WgXcQ"})
    assert "Couldn't get a transcript" in out and "disabled" in out
    # empty transcript -> honest, no LLM spend
    tool, _ = make_yt(router, transcript="   ")
    assert "empty transcript" in tool.invoke({"url": "https://youtu.be/dQw4w9WgXcQ"})
    assert router.requests == []  # none of the failure paths hit OpenRouter
    # summarizer HTTP failure -> honest string
    tool, _ = make_yt(FakeOpenRouter(status=500))
    assert "HTTP 500" in tool.invoke({"url": "https://youtu.be/dQw4w9WgXcQ"})


# ---- aggregate builder -------------------------------------------------------------

def test_build_media_tools_names_match_what_prompts_call():
    # CLAUDE.md quirk: the tool name the LLM sees MUST match what prompts say —
    # V1's hidden toolWorkflow `name` property caused hallucinated tool calls.
    tools = build_media_tools(api_key="or-key")
    assert [t.name for t in tools] == ["analyze_image", "read_document", "youtube_summary"]
    # descriptions keep the concrete CDN trigger string (load-bearing for invocation)
    assert "cdn.discordapp.com/attachments" in tools[0].description
