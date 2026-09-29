"""Her own eyes (2026-09-28): the current turn's Discord images inlined at the model
boundary. Selyra's screenshot came back as "That image reads: ..." because a separate tool
looked at it; now the chat model does."""
import base64

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from aerys_v2 import images
from aerys_v2.images import ATTACHED, EARLIER, NOT_LOADED, inline_current_images, media_type_of

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 20
URL = "https://cdn.discordapp.com/attachments/1/2/image.png?ex=6abc703e&is=6abb1ebe&hm=78cb93"
OLD = "https://cdn.discordapp.com/attachments/1/9/old.jpg?ex=1&is=2&hm=3"


def fetched(url):
    return ("image/png", PNG) if url == URL else None


def test_the_current_image_is_inlined_and_the_stored_message_is_untouched():
    stored = HumanMessage(content=f"Selyra's reply to you\n{URL}")
    prompt = [SystemMessage(content="soul"), HumanMessage(content=f"look {OLD}"), AIMessage(content="nice"), stored]
    out = inline_current_images(prompt, fetch=fetched)
    blocks = out[-1].content
    assert blocks[0] == {"type": "text", "text": f"Selyra's reply to you\n{ATTACHED}"}
    assert blocks[1]["type"] == "image" and blocks[1]["source"]["media_type"] == "image/png"
    assert base64.b64decode(blocks[1]["source"]["data"]) == PNG
    assert out[1].content == f"look {EARLIER}", "an earlier image is named, never refetched"
    assert stored.content == f"Selyra's reply to you\n{URL}" and prompt[1].content == f"look {OLD}"


def test_list_content_from_the_context_layout_keeps_its_blocks():
    last = HumanMessage(content=[{"type": "text", "text": "[Context for this turn]\nnow"},
                                 {"type": "text", "text": URL}])
    out = inline_current_images([last], fetch=fetched)
    assert [b["type"] for b in out[0].content] == ["text", "text", "image"]
    assert out[0].content[1]["text"] == ATTACHED


def test_a_failed_fetch_says_so_and_adds_no_image():
    out = inline_current_images([HumanMessage(content=f"hi {URL}")], fetch=lambda url: None)
    assert out[0].content == [{"type": "text", "text": f"hi {NOT_LOADED}"}]


def test_only_discord_images_are_ever_fetched():
    asked = []
    for text in ("https://evil.example/attachments/x.png", "http://cdn.discordapp.com/attachments/1/2/a.png",
                 "https://cdn.discordapp.com/attachments/1/2/report.pdf?ex=1", "no link at all"):
        prompt = [HumanMessage(content=text)]
        assert inline_current_images(prompt, fetch=lambda url: asked.append(url)) == prompt
    assert asked == []


def test_at_most_four_images_and_types_come_from_the_bytes():
    many = " ".join(f"https://cdn.discordapp.com/attachments/1/{i}/p.png" for i in range(6))
    seen = []
    inline_current_images([HumanMessage(content=many)], fetch=lambda url: seen.append(url))
    assert len(seen) == images.MAX_IMAGES
    assert media_type_of(PNG) == "image/png" and media_type_of(b"\xff\xd8\xff\xe0") == "image/jpeg"
    assert media_type_of(b"RIFF\x00\x00\x00\x00WEBPVP8") == "image/webp" and media_type_of(b"<html>") is None


def test_the_plan_prompt_carries_the_image_right_after_its_line():
    from aerys_v2.oauth_model import _flatten, _prompt_blocks, _prompt_text

    prompt = inline_current_images([SystemMessage(content="soul"), HumanMessage(content="hi"), AIMessage(content="hey"),
                                    HumanMessage(content=f"from Selyra {URL}")], fetch=fetched)
    blocks = _prompt_blocks(prompt)
    kinds = [b["type"] for b in blocks]
    assert kinds == ["text", "text", "text", "text", "image", "text"], kinds
    assert "cache_control" in blocks[2] and "cache_control" not in blocks[4], "the image stays out of the cached prefix"
    assert _prompt_text(blocks) == _flatten(prompt)


def test_the_local_lifeboat_gets_the_text_and_is_told_it_cannot_see_the_image():
    """LOCAL_FALLBACK_URL is set on the rack: a failed plan call on an image turn used to
    hand the text-only local model an image it errors on."""
    from aerys_v2.factory import LocalFailoverModel
    from aerys_v2.images import UNSEEN
    from langchain_core.language_models.fake_chat_models import GenericFakeChatModel

    class Down(GenericFakeChatModel):
        def _generate(self, *a, **k):
            raise RuntimeError("plan window rejected")

    seen = []

    class Lifeboat(GenericFakeChatModel):
        def _generate(self, messages, *a, **k):
            seen.append(messages)
            return super()._generate(messages, *a, **k)

    prompt = inline_current_images([SystemMessage(content="soul"), HumanMessage(content=f"look {URL}")],
                                   fetch=lambda url: ("image/png", PNG))
    model = LocalFailoverModel(primary=Down(messages=iter([])), lifeboat=Lifeboat(messages=iter([AIMessage(content="ok")])))
    assert model.invoke(prompt).content == "ok"
    (last,) = [m for m in seen[0] if isinstance(m, HumanMessage)]
    assert last.content == [{"type": "text", "text": f"look {UNSEEN}"}]
    assert any(b.get("type") == "image" for b in prompt[-1].content)   # the primary's prompt is untouched


def test_a_telegram_photo_is_seen_from_the_bytes_that_came_with_this_turn():
    """9/29: the gateway hands the photo's bytes to the turn (TURN_IMAGES); no URL exists."""
    from aerys_v2.images import PHOTO_MARKER, with_turn_images

    def boom(url):
        raise AssertionError("a Telegram photo is never fetched by URL")

    prompt = [SystemMessage(content="soul"), HumanMessage(content=f"{PHOTO_MARKER} earlier one"),
              AIMessage(content="nice"), HumanMessage(content=f"{PHOTO_MARKER} what do you think")]
    seen = with_turn_images([("image/png", PNG)], inline_current_images, prompt, fetch=boom)
    blocks = seen[-1].content
    assert blocks[0]["text"] == f"{ATTACHED} what do you think" and blocks[1]["type"] == "image"
    assert seen[1].content == f"{EARLIER} earlier one"
    assert prompt[-1].content == f"{PHOTO_MARKER} what do you think"          # the stored message is untouched
    failed = inline_current_images(prompt, fetch=boom)                       # the download failed: no bytes
    assert failed[-1].content == [{"type": "text", "text": f"{NOT_LOADED} what do you think"}]


def test_the_photo_bytes_reach_the_chat_node_through_the_real_graph_and_executor():
    """Gemini 2026-09-29 doubted a ContextVar set in the transport's executor thread is
    visible inside LangGraph's chat node. Through build_graph + ask() + run_in_executor: it is."""
    import asyncio

    from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
    from langgraph.checkpoint.memory import InMemorySaver

    from aerys_v2.factory import build_graph
    from aerys_v2.images import PHOTO_MARKER, with_turn_images
    from aerys_v2.service import ask

    seen = []

    class Recorder(FakeMessagesListChatModel):
        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            seen.append(list(messages))
            return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)

    graph = build_graph(Recorder(responses=[AIMessage(content="lovely")]), soul="s", checkpointer=InMemorySaver())
    identity = {"user_id": "u1", "display_name": "Chris", "privacy_context": "private", "platform": "telegram",
                "trust": "owner"}

    async def transport():
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: with_turn_images(
            [("image/png", PNG)], ask, graph, f"{PHOTO_MARKER} look", identity=identity, thread_id="person:u1",
            router=lambda _t: __import__("aerys_v2.router", fromlist=["RouteDecision"]).RouteDecision("chat", "")))

    asyncio.run(transport())
    last = [m for m in seen[-1] if isinstance(m, HumanMessage)][-1]
    assert any(isinstance(b, dict) and b.get("type") == "image" for b in last.content), last.content


def test_the_photo_marker_counts_only_where_the_gateway_puts_it():
    """Gemini 2026-09-29: someone typing the marker words mid-message must not become a photo."""
    from aerys_v2.images import PHOTO_MARKER, carries_image
    assert carries_image(f"{PHOTO_MARKER} look") and not carries_image(f"she said {PHOTO_MARKER} was missing")
    prompt = [HumanMessage(content=f"she said {PHOTO_MARKER} was missing")]
    assert inline_current_images(prompt, fetch=fetched) == prompt


def test_her_chat_capability_says_she_sees_images_he_attaches():
    """Live 2026-09-29 (plan, Sonnet 5): with "no eyes on attachments" still in her chat
    capability, "can you actually see this image?" handed the turn to the vision tool and
    "do you have eyes on images now?" got "No -- not in this mode". Images come with his
    message since a5aafe2; the capability must say so."""
    from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel

    from aerys_v2.factory import build_graph
    from aerys_v2.router import RouteDecision
    from aerys_v2.service import ask

    seen = []

    class Capture(FakeMessagesListChatModel):
        def _generate(self, messages, *a, **k):
            seen.append(messages)
            return super()._generate(messages, *a, **k)

    identity = {"user_id": "u1", "display_name": "Chris", "privacy_context": "private", "platform": "discord",
                "trust": "owner"}
    ask(build_graph(Capture(responses=[AIMessage(content="x")]), soul="s"), "hi", identity=identity,
        thread_id="person:u1", router=lambda _t: RouteDecision("chat", ""))
    system = seen[0][0].content
    assert "no eyes on attachments" not in system and "IMAGE he attaches" in system and "you see it yourself" in system
