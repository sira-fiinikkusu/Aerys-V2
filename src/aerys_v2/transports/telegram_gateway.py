"""Telegram gateway transport — the aiogram v3 mirror of discord_gateway (1c spike, cont'd).

n8n mapping: replaces workflow 02-02 Telegram Adapter (K1jR1tpKZTOiid8N). That
adapter was a webhook wired straight into the (retiring) n8n Core Agent; this
gateway session absorbs it the same way discord_gateway absorbed BOTH Discord
adapters — normalize → resolve → ask() → chunked reply, no n8n in the loop.

Design split for testability (identical shape to discord_gateway.py):
  - `should_handle()`, `normalize()`, and `telegram_thread_key()` are PURE —
    every gating decision and field mapping is unit-tested offline with
    SimpleNamespace fakes (see tests/test_telegram_transport.py).
  - `AerysTelegramClient` is the thin I/O shell around them; it is exercised
    live, not in CI. Built tonight with no bot token in hand — nothing here is
    wired into cli.py or activated; it awaits BotFather.

NOTE: NormalizedEvent is imported from discord_gateway rather than redefined —
both platforms produce the identical transport-neutral shape. A future refactor
may hoist NormalizedEvent (and this per-platform thread_key's shared
DM-follows-person / room-follows-channel rationale) into a shared base module
once a third transport needs the same contract.
"""

import asyncio
import io
import logging

from aiogram import Bot, Dispatcher
from aiogram.types import Message

from aerys_v2.channels.splitter import split_message
from aerys_v2.images import MAX_BYTES, PHOTO_MARKER, media_type_of, with_turn_images
from aerys_v2.state import Identity
from aerys_v2.transports.discord_gateway import (
    EMPTY_PING,
    NormalizedEvent,
    person_thread_key,
)

log = logging.getLogger(__name__)

# Telegram's hard per-message limit — NOT Discord's 2000 (see splitter.py's
# TELEGRAM_LIMIT; duplicated here as a plain constant so this transport doesn't
# need to reach into the Output Router's module for one number).
TELEGRAM_MESSAGE_LIMIT = 4096


def telegram_thread_key(channel_kind: str, platform_user_id: str, chat_id: str) -> str:
    """Conversation key for the checkpointer — same rationale as discord's thread_key.

    DMs follow the PERSON (one continuous conversation regardless of which
    Telegram client sent it); groups follow the CHAT (a shared room is one
    thread — identity stays per-call, which is exactly why it must never live
    in checkpointed state).
    """
    if channel_kind == "dm":
        return f"telegram:dm:{platform_user_id}"
    return f"telegram:group:{chat_id}"


def should_handle(
    *,
    is_bot: bool,
    is_dm: bool,
    chat_id: int,
    allowed_chat_ids: frozenset[int],
    mentions_me: bool,
) -> bool:
    """Every drop/accept rule in one pure function.

    Mirrors discord's should_handle FAIL-CLOSED group posture: bots never get to
    summon Aerys (the same rule that keeps Kael and Aerys from looping each
    other), DMs always in, and a group is served only when its chat_id is in an
    EXPLICIT allowlist AND the message mentions her. An empty allowed_chat_ids
    means "serve NO groups" — the direct analogue of discord requiring
    DISCORD_GUILD_ID before any guild is served (discord_gateway.should_handle:
    `allowed_guild_id is None → return False`). Telegram has no separate
    guild-id gate, so the chat-id allowlist IS that primary gate; leaving it
    empty must DENY every group, not open every group the bot is @mentioned in
    (otherwise anyone who knows the @username could add the bot to any group and
    burn the owner's model budget). Telegram has no self-message problem the way
    Discord does (long-polling never redelivers the bot's own sends), so there's
    no author_is_self check here.
    """
    if is_bot:
        return False
    if is_dm:
        return True
    if not allowed_chat_ids or chat_id not in allowed_chat_ids:
        return False
    return mentions_me


def normalize(message: object, *, bot_username: str) -> NormalizedEvent:
    """Map an aiogram Message to the neutral event (pure — fakes in tests).

    Strips every `@{bot_username}` occurrence from group text the way discord's
    normalize strips `<@id>`/`<@!id>`, so the model sees "what's up" not
    "@aerys_bot what's up". display_name prefers from_user.full_name, falling
    back to from_user.username (a bare display name can be blank; a username
    can't, but both are optional on the wire so we guard both).
    """
    is_dm = message.chat.type == "private"
    channel_kind = "dm" if is_dm else "group"
    # A photo's words are its caption (dropped until 2026-09-29), and the photo itself
    # rides as PHOTO_MARKER: its bytes reach the chat turn through TURN_IMAGES, never a
    # file URL (a Telegram file URL carries the bot token).
    text = (message.text or getattr(message, "caption", None) or "").replace(f"@{bot_username}", "")
    if getattr(message, "photo", None):
        text = f"{PHOTO_MARKER} {text}"
    user = message.from_user
    platform_user_id = str(user.id)
    channel_id = str(message.chat.id)
    return NormalizedEvent(
        platform="telegram",
        platform_user_id=platform_user_id,
        display_name=user.full_name or user.username or "",
        channel_kind=channel_kind,
        channel_id=channel_id,
        thread_id=telegram_thread_key(channel_kind, platform_user_id, channel_id),
        text=text.strip(),
    )


class AerysTelegramClient:
    """The I/O shell: aiogram polling session in, ask() out, chunked replies back.

    Same injected seams as AerysDiscordClient: ask_fn and resolve_fn are the
    only things this class knows about the outside world — never models,
    souls, or checkpointers. Unlike discord.py's Client subclass, aiogram v3
    favors composition over inheritance (Bot = credentials/HTTP, Dispatcher =
    routing), so this class owns a Dispatcher rather than being one.
    """

    def __init__(
        self,
        *,
        ask_fn,
        resolve_fn,
        allowed_chat_ids: frozenset[int] = frozenset(),
        bot_username: str | None = None,
    ) -> None:
        self._ask = ask_fn
        self._resolve = resolve_fn
        self._chat_ids = allowed_chat_ids
        self._bot_username = bot_username
        self._dp = Dispatcher()
        # aiogram v3 injects handler params by name from its per-update context
        # dict (bot, event_chat, ...) — `bot: Bot` below arrives that way, not
        # via a manual lookup.
        self._dp.message.register(self._on_message)

    def _mentions_me(self, message: Message, *, bot_id: int) -> bool:  # pragma: no cover - live only
        """An @mention OR a reply to one of our own messages counts as a summon.

        Kept simple and correct rather than clever: aiogram exposes message
        entities that could locate an exact @mention span, but a substring
        check on `@{bot_username}` makes the same string-match trade discord's
        mention-strip already makes, and reply-to-bot needs no entity parsing
        at all — just comparing the replied-to message's author id.
        """
        text = message.text or message.caption or ""
        if self._bot_username and f"@{self._bot_username}" in text:
            return True
        reply = message.reply_to_message
        return bool(reply is not None and reply.from_user is not None and reply.from_user.id == bot_id)

    async def _on_message(self, message: Message, bot: Bot) -> None:  # pragma: no cover - live only
        user = message.from_user
        if user is None:
            return  # channel posts / anonymous-admin sends have no from_user — nothing to resolve
        if not should_handle(
            is_bot=user.is_bot,
            is_dm=message.chat.type == "private",
            chat_id=message.chat.id,
            allowed_chat_ids=self._chat_ids,
            mentions_me=self._mentions_me(message, bot_id=bot.id),
        ):
            return
        event = normalize(message, bot_username=self._bot_username or "")
        identity: Identity = self._resolve(event)
        # Person-keyed threading (parity with discord_gateway): the checkpointer key
        # is derived from the RESOLVED identity, so a Telegram DM/group joins the SAME
        # 'person:{id}' thread as his Discord surfaces (cross-surface continuity).
        thread_id = person_thread_key(identity["user_id"])
        # Empty text (bare @mention, sticker-only) rides the model via EMPTY_PING so
        # the acknowledgement is contextual, not a fixed string (parity with discord).
        turn_text = event.text.strip() or EMPTY_PING
        # ask() is sync (same seam and same caveat as discord_gateway: fine for
        # a one-user spike, the soak test will tell us whether it needs more);
        # run_in_executor keeps a slow LLM turn from blocking aiogram's loop.
        images = await self._photo(message, bot)
        loop = asyncio.get_running_loop()
        try:
            reply = await loop.run_in_executor(
                None, lambda: with_turn_images(images, self._ask, turn_text, identity, thread_id)
            )
        except Exception:
            # Any failure inside the turn becomes a short apology, never dead air.
            log.exception("ask() failed for thread %s", thread_id)
            await message.answer(
                "Sorry — something broke on my end handling that. Try me again in a moment?"
            )
            return
        # Telegram's hard limit is 4096 chars — NOT Discord's 2000.
        for chunk in split_message(reply, TELEGRAM_MESSAGE_LIMIT):
            await message.answer(chunk)

    async def _photo(self, message: Message, bot: Bot) -> list[tuple[str, bytes]]:  # pragma: no cover - live only
        """The message's photo as [(media_type, bytes)], or [] (no photo, or it would not
        load: the chat turn then says so). The largest size within MAX_BYTES, fetched in
        memory; nothing is written to disk and no file URL leaves this function."""
        # An unknown size is tried: Telegram photos are compressed and getFile serves at
        # most 20 MB, and the bytes are checked against MAX_BYTES after the download.
        sizes = [p for p in (message.photo or []) if (p.file_size or 0) <= MAX_BYTES]
        if not sizes:
            return []
        try:
            buffer = await bot.download(sizes[-1].file_id, destination=io.BytesIO(), timeout=20)
        except Exception as exc:  # noqa: BLE001 -- a failed download becomes "didn't load"
            log.warning("telegram photo download failed (%s)", type(exc).__name__)
            return []
        data = buffer.getvalue() if buffer is not None else b""
        kind = media_type_of(data)
        return [(kind, data)] if kind and len(data) <= MAX_BYTES else []

    async def run(self, token: str) -> None:  # pragma: no cover - live only
        """Starts long-polling. No webhook — matches discord_gateway's gateway-session
        model (one persistent connection, no adapter-IPC race to watchdog around).

        bot_username is resolved from Telegram itself (getMe) when not supplied
        at construction, so the constructor never has to guess it.
        """
        bot = Bot(token=token)
        if self._bot_username is None:
            me = await bot.get_me()
            self._bot_username = me.username
        print(f"telegram polling up as @{self._bot_username}")
        await self._dp.start_polling(bot)
