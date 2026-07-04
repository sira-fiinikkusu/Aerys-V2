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

from aiogram import Bot, Dispatcher
from aiogram.types import Message

from aerys_v2.channels.splitter import split_message
from aerys_v2.state import Identity
from aerys_v2.transports.discord_gateway import NormalizedEvent

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

    Mirrors discord's should_handle: bots never get to summon Aerys (the same
    rule that keeps Kael and Aerys from looping each other), DMs always in,
    groups only when allowlisted (an empty allowlist means "no chat_id
    restriction, judge by mention alone" — same non-empty-only enforcement as
    discord's channel allowlist) AND the message mentions her. Telegram has no
    self-message problem the way Discord does (long-polling never redelivers
    the bot's own sends), so there's no author_is_self check here.
    """
    if is_bot:
        return False
    if is_dm:
        return True
    if allowed_chat_ids and chat_id not in allowed_chat_ids:
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
    text = (message.text or "").replace(f"@{bot_username}", "")
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
        text = message.text or ""
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
        # ask() is sync (same seam and same caveat as discord_gateway: fine for
        # a one-user spike, the soak test will tell us whether it needs more);
        # run_in_executor keeps a slow LLM turn from blocking aiogram's loop.
        loop = asyncio.get_running_loop()
        reply = await loop.run_in_executor(
            None, lambda: self._ask(event.text, identity, event.thread_id)
        )
        # Telegram's hard limit is 4096 chars — NOT Discord's 2000.
        for chunk in split_message(reply, TELEGRAM_MESSAGE_LIMIT):
            await message.answer(chunk)

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
