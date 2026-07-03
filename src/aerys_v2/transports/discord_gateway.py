"""Discord gateway transport (1c spike) — ONE client for guild + DMs.

n8n mapping: replaces BOTH adapter workflows (02-01 guild pmwONooCIEPQrlbJ and
03-03 DM 0UH3pW7AWwkG1HOs). Those had to be two workflows because katerlol's IPC
is last-one-activated-wins — one listener at a time, hence the watchdog and the
sacred DM-first/guild-last activation liturgy. discord.py holds one gateway
session that receives everything; that entire failure class is deleted, not fixed.

Design split for testability:
  - `should_handle()` + `normalize()` are PURE — every gating decision and field
    mapping is unit-tested offline with fakes (see tests/test_discord_transport.py).
  - `AerysDiscordClient` is the thin I/O shell around them; it is exercised live,
    not in CI (the spike's reconnect soak happens on real hardware).
"""

from dataclasses import dataclass

import discord

from aerys_v2.channels.splitter import split_message
from aerys_v2.state import Identity


@dataclass(frozen=True)
class NormalizedEvent:
    """The transport-neutral shape every adapter produces (the Normalize Message node)."""

    platform: str            # "discord"
    platform_user_id: str    # snowflake as string
    display_name: str
    channel_kind: str        # "dm" | "guild"
    channel_id: str
    thread_id: str           # checkpointer key — see thread_key()
    text: str


def thread_key(channel_kind: str, platform_user_id: str, channel_id: str) -> str:
    """Conversation key for the checkpointer.

    DMs follow the PERSON (one continuous conversation regardless of client);
    guild channels follow the CHANNEL (a shared room is one thread — identity
    stays per-call, which is exactly why it must never live in checkpointed state).
    """
    if channel_kind == "dm":
        return f"discord:dm:{platform_user_id}"
    return f"discord:guild:{channel_id}"


def should_handle(
    *,
    author_is_self: bool,
    author_is_bot: bool,
    is_dm: bool,
    guild_id: int | None,
    allowed_guild_id: int | None,
    channel_id: int,
    allowed_channel_ids: frozenset[int],
    mentions_me: bool,
) -> bool:
    """Every drop/accept rule in one pure function.

    Mirrors the live adapters' gates: never self, never bots (bots don't get to
    summon Aerys — the same rule that keeps Kael and Aerys from looping each
    other), DMs always in, guild only in the configured guild (+channel allowlist
    when set), and guild messages must mention her (DMs don't need to).
    """
    if author_is_self or author_is_bot:
        return False
    if is_dm:
        return True
    if allowed_guild_id is None or guild_id != allowed_guild_id:
        return False
    if allowed_channel_ids and channel_id not in allowed_channel_ids:
        return False
    return mentions_me


def normalize(message: object, *, self_id: int) -> NormalizedEvent:
    """Map a discord.py Message to the neutral event (pure — fakes in tests).

    Strips the leading bot-mention from guild text the way the guild adapter's
    Normalize node did, so the model sees "what's the weather" not "<@123> what's
    the weather".
    """
    is_dm = message.guild is None
    text = message.content or ""
    mention_tokens = (f"<@{self_id}>", f"<@!{self_id}>")
    for tok in mention_tokens:
        text = text.replace(tok, "")
    return NormalizedEvent(
        platform="discord",
        platform_user_id=str(message.author.id),
        display_name=getattr(message.author, "display_name", None)
        or message.author.name,
        channel_kind="dm" if is_dm else "guild",
        channel_id=str(message.channel.id),
        thread_id=thread_key(
            "dm" if is_dm else "guild", str(message.author.id), str(message.channel.id)
        ),
        text=text.strip(),
    )


class AerysDiscordClient(discord.Client):
    """The I/O shell: gateway session in, ask() out, chunked replies back.

    ask_fn is injected (same seam as everywhere else) so this class never knows
    about models, souls, or checkpointers. resolve_fn turns a platform user into
    an Identity — the DB-backed resolver when configured, display-name passthrough
    when not (the spike runs fine with no database).
    """

    def __init__(
        self,
        *,
        ask_fn,
        resolve_fn,
        allowed_guild_id: int | None,
        allowed_channel_ids: frozenset[int] = frozenset(),
    ) -> None:
        intents = discord.Intents.default()
        intents.message_content = True  # privileged — enabled on the dev bot app page
        super().__init__(intents=intents)
        self._ask = ask_fn
        self._resolve = resolve_fn
        self._guild_id = allowed_guild_id
        self._channel_ids = allowed_channel_ids

    async def on_ready(self) -> None:  # pragma: no cover - live only
        print(f"gateway up as {self.user} (guild={self._guild_id})")

    async def on_message(self, message: discord.Message) -> None:  # pragma: no cover - live only
        if not should_handle(
            author_is_self=(message.author.id == self.user.id),
            author_is_bot=message.author.bot,
            is_dm=message.guild is None,
            guild_id=message.guild.id if message.guild else None,
            allowed_guild_id=self._guild_id,
            channel_id=message.channel.id,
            allowed_channel_ids=self._channel_ids,
            mentions_me=self.user in message.mentions,
        ):
            return
        event = normalize(message, self_id=self.user.id)
        identity: Identity = self._resolve(event)
        async with message.channel.typing():
            # ask() is sync (fine for a one-user spike); the soak test will tell
            # us whether it needs a thread executor before this leaves spike-hood.
            reply = await self.loop.run_in_executor(
                None, lambda: self._ask(event.text, identity, event.thread_id)
            )
        for chunk in split_message(reply, 2000):
            await message.channel.send(chunk)
