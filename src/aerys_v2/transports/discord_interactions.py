"""Discord slash-command interactions — the gateway-native replacement for n8n 03-02.

n8n mapping: 03-02 Discord Slash Commands (OGqzkWSpgK9XN8ZM, 100 nodes) + 04-03
Memory Commands (OWezq9kumsXisw7R) + the Cloudflare sig-verify worker
(aerys-discord-verify). Those three exist only because n8n received interactions
as WEBHOOKS: Discord demands ed25519 signature verification on a public HTTPS
endpoint, hence the CF worker in front, hence PING handling, hence deferred
PATCH-followup gymnastics in every branch. discord.py receives interactions over
the gateway session this process already holds — the entire webhook chain
(worker + tunnel exposure + signature dance) is deleted, not ported.

Registration replaces the run-once 03-02 Register Commands workflow
(PqOG9hskuDA19GnZ): `CommandTree.sync(guild=...)` PUTs the command set on login,
so the code below is the single source of truth for what exists. Deliberately
DROPPED from V1's set: /aerys-pin — V2's `memories` table has no locked column
(pin/lock semantics are a backlog design item, not a silent stub).

Split (same seam discipline as discord_gateway.py):
  - PURE command logic lives in discord_commands.py / discord_link.py — conn in,
    reply string out, unit-tested offline.
  - THIS module is the I/O shell: defer-ephemeral, resolve the invoker, run the
    pure handler on a worker thread with a fresh short-lived prod-aerys
    connection, followup.send the reply. It is exercised live, not in CI, except
    for the pure helpers at the bottom (ensure_person, _is_admin).

AUTH BOUNDARY notes (the invariants that make this file reviewed-not-delegated):
  - Memory/profile/status commands require a KNOWN person — an unknown invoker
    gets a friendly "I don't know you yet", never a fallthrough to any other
    person's rows. Handlers receive the invoker's OWN resolved person_id only.
  - /link is the ONE command allowed to mint a person (ensure_person below —
    the create-on-miss write that services/identity.py deliberately refused to
    own; single transaction, conflict re-select, no orphan persons row).
  - /gaps is owner-only here. V1's webhook flow had no gate (anyone in the guild
    could pull the mined-gaps list); that was an oversight, not a feature —
    it is operator telemetry, and replies are ephemeral anyway.
  - Admin commands gate on the Aerys Admin role read from the interaction's
    Member object — DM invocations have no roles and therefore always fail
    closed to unauthorized.

Failure posture: any exception inside a handler becomes an ephemeral apology,
logged — a slash command must never eat the interaction (Discord shows a scary
"application did not respond" after 3s undeferred / 15min deferred).
"""

from __future__ import annotations

import asyncio
import logging
import secrets
from contextlib import contextmanager
from typing import Any, Callable, Iterator, Sequence

import discord
from discord import app_commands

log = logging.getLogger(__name__)

# Same alphabet as the n8n Generate Link Code node (no 0/O/1/I ambiguity), but
# secrets.choice instead of Math.random — the n8n sandbox blocked crypto; Python
# has no such excuse for an auth code.
_CODE_CHARS = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"


def generate_link_code() -> str:
    return "".join(secrets.choice(_CODE_CHARS) for _ in range(6))


# ---------------------------------------------------------------- pure helpers


def ensure_person(conn: Any, platform: str, platform_user_id: str, username: str) -> str:
    """Create-on-miss, done right — the write half n8n 03-01 got wrong.

    The n8n flow's `INSERT persons` + `INSERT platform_identities ON CONFLICT DO
    NOTHING` leaked an orphan persons row when two first-contacts raced. Here both
    inserts share ONE transaction: if the identity insert loses the race (RETURNING
    comes back empty), the transaction rolls back — taking the would-be orphan
    persons row with it — and we re-select the winner. Returns person_id as str.
    """
    row = conn.execute(
        "SELECT person_id::text FROM platform_identities"
        " WHERE platform = %s AND platform_user_id = %s",
        (platform, platform_user_id),
    ).fetchone()
    if row is not None:
        return row[0]

    class _LostRace(Exception):
        pass

    try:
        with conn.transaction():
            pid = conn.execute(
                "INSERT INTO persons (display_name) VALUES (%s) RETURNING id::text",
                (username,),
            ).fetchone()[0]
            claimed = conn.execute(
                "INSERT INTO platform_identities (person_id, platform, platform_user_id, username)"
                " VALUES (%s::uuid, %s, %s, %s)"
                " ON CONFLICT (platform, platform_user_id) DO NOTHING"
                " RETURNING person_id::text",
                (pid, platform, platform_user_id, username),
            ).fetchone()
            if claimed is None:
                raise _LostRace()  # rollback removes the orphan persons row
    except _LostRace:
        row = conn.execute(
            "SELECT person_id::text FROM platform_identities"
            " WHERE platform = %s AND platform_user_id = %s",
            (platform, platform_user_id),
        ).fetchone()
        return row[0]
    return pid


def is_admin_member(user: Any, admin_role_id: int | None) -> bool:
    """PURE gate: does the interaction's user carry the Aerys Admin role?

    DM interactions hand us a User (no .roles) — getattr fails CLOSED to False,
    so admin commands are guild-only by construction. admin_role_id None (the
    setting unconfigured) also fails closed: nobody is admin.
    """
    if admin_role_id is None:
        return False
    roles = getattr(user, "roles", None) or []
    return any(getattr(r, "id", None) == admin_role_id for r in roles)


NOT_KNOWN_REPLY = (
    "I don't know you yet — say hi to me in the server first (so I have someone "
    "to attach this to), or use /link if you've talked to me on Telegram."
)


# ------------------------------------------------------------------ the shell


def attach_interactions(
    client: discord.Client,
    *,
    guild_id: int,
    admin_role_id: int | None,
    conn_factory: Callable[[], Any],
    embedder: Callable[[str], Sequence[float]] | None,
    gaps_fn: Callable[[], str] | None,
    owner_person_id: str | None,
    telegram_notify: Callable[[str, str], bool] | None = None,
) -> app_commands.CommandTree:
    """Register the full slash-command set on `client` and sync it on login.

    conn_factory: context manager yielding a fresh READ-WRITE psycopg connection
    to PROD aerys (memories/persons/platform_identities/pending_links/audit_log
    live there — NOT the brain's own aerys_v2 DB). Fresh-per-command like every
    other seam (personal-assistant volume; a wedged NAS can't hold a pool slot).

    telegram_notify(chat_id, text) -> bool delivers admin-link notifications to
    the Telegram side; None degrades to a note in the admin's ephemeral reply.
    """
    # Imports deferred to attach-time so this module imports without the pure
    # handler modules present (they land in the same deploy; belt-and-braces
    # for partial checkouts and keeps import-time free of sibling coupling).
    from aerys_v2.services.identity import resolve_identity
    from aerys_v2.transports import discord_commands as cmds
    from aerys_v2.transports import discord_link as linkmod

    tree = app_commands.CommandTree(client)
    guild = discord.Object(id=guild_id)

    def _resolve(user_id: int) -> str | None:
        with conn_factory() as conn:
            row = resolve_identity(conn, "discord", str(user_id))
        return row["person_id"] if row else None

    async def _run(interaction: discord.Interaction, work: Callable[[], str]) -> None:
        """defer → thread → followup, with the apology posture on any failure."""
        await interaction.response.defer(ephemeral=True)
        try:
            reply = await asyncio.to_thread(work)
        except Exception:
            log.exception("slash command %s failed", interaction.command and interaction.command.name)
            reply = "Sorry — something broke on my end handling that. Try again in a moment?"
        await interaction.followup.send(reply[:1990], ephemeral=True)

    def _known_person_work(user_id: int, fn: Callable[[Any, str], str]) -> Callable[[], str]:
        """Wrap a pure handler that needs (conn, person_id) for a KNOWN invoker."""

        def work() -> str:
            with conn_factory() as conn:
                row = resolve_identity(conn, "discord", str(user_id))
                if row is None:
                    return NOT_KNOWN_REPLY
                return fn(conn, row["person_id"])

        return work

    # ---- memory family (04-03 ported onto the V2 `memories` table) ----

    @tree.command(name="aerys-recall", description="See what Aerys remembers about you", guild=guild)
    async def aerys_recall(interaction: discord.Interaction) -> None:
        await _run(interaction, _known_person_work(interaction.user.id, cmds.recall))

    @tree.command(name="aerys-forget", description="Ask Aerys to forget a specific memory", guild=guild)
    @app_commands.describe(fact="Search for the memory to forget")
    async def aerys_forget(interaction: discord.Interaction, fact: str) -> None:
        await _run(
            interaction,
            _known_person_work(
                interaction.user.id, lambda conn, pid: cmds.forget(conn, pid, fact)
            ),
        )

    @tree.command(name="aerys-correct", description="Correct a memory Aerys has about you", guild=guild)
    @app_commands.describe(fact="Search for the memory to correct", value="The correct value")
    async def aerys_correct(interaction: discord.Interaction, fact: str, value: str) -> None:
        if embedder is None:
            await interaction.response.send_message(
                "Memory writes are offline right now (no embedder configured).", ephemeral=True
            )
            return
        await _run(
            interaction,
            _known_person_work(
                interaction.user.id,
                lambda conn, pid: cmds.correct(conn, pid, fact, value, embedder),
            ),
        )

    @tree.command(name="aerys-tell", description="Tell Aerys something about yourself to remember", guild=guild)
    @app_commands.describe(fact="What should Aerys remember?")
    async def aerys_tell(interaction: discord.Interaction, fact: str) -> None:
        if embedder is None:
            await interaction.response.send_message(
                "Memory writes are offline right now (no embedder configured).", ephemeral=True
            )
            return
        # Room-scoped privacy, same convention as the resolver: only a true DM
        # is private; a guild room is public.
        privacy = "private" if interaction.guild is None else "public"
        await _run(
            interaction,
            _known_person_work(
                interaction.user.id,
                lambda conn, pid: cmds.tell(conn, pid, fact, embedder, privacy),
            ),
        )

    # ---- identity family (03-02 ported) ----

    @tree.command(name="status", description="View your linked accounts and display name", guild=guild)
    async def status(interaction: discord.Interaction) -> None:
        await _run(interaction, _known_person_work(interaction.user.id, cmds.status))

    @tree.command(name="profile", description="Update your display name", guild=guild)
    @app_commands.describe(name="Your new display name")
    async def profile(interaction: discord.Interaction, name: str) -> None:
        await _run(
            interaction,
            _known_person_work(
                interaction.user.id, lambda conn, pid: cmds.set_profile_name(conn, pid, name)
            ),
        )

    @tree.command(
        name="link",
        description="Link your Discord and Telegram accounts (or redeem a code from Telegram)",
        guild=guild,
    )
    @app_commands.describe(code="Verification code from Telegram")
    async def link(interaction: discord.Interaction, code: str | None = None) -> None:
        user = interaction.user

        def work() -> str:
            with conn_factory() as conn:
                # The ONE create-on-miss door: /link must work for someone whose
                # first-ever touch is the linking itself.
                pid = ensure_person(
                    conn, "discord", str(user.id), getattr(user, "display_name", None) or user.name
                )
                return linkmod.link(conn, pid, str(user.id), code, generate_link_code)

        await _run(interaction, work)

    @tree.command(name="unlink", description="Unlink your Discord and Telegram accounts", guild=guild)
    async def unlink(interaction: discord.Interaction) -> None:
        user_id = interaction.user.id
        await _run(
            interaction,
            _known_person_work(
                user_id,
                lambda conn, pid: linkmod.unlink(conn, pid, "discord", str(user_id)),
            ),
        )

    @tree.command(
        name="admin-link",
        description="Force-link a Discord account to a Telegram account (admin only)",
        guild=guild,
    )
    @app_commands.describe(discord_user="Discord user to link", telegram_id="Telegram user ID to link")
    async def admin_link(
        interaction: discord.Interaction, discord_user: discord.User, telegram_id: str
    ) -> None:
        admin = is_admin_member(interaction.user, admin_role_id)

        def work() -> str:
            with conn_factory() as conn:
                result = linkmod.admin_link(conn, str(discord_user.id), telegram_id, admin)
            reply = result["reply"]
            notes: list[str] = []
            for n in result.get("notify", []):
                if n["platform"] == "telegram":
                    if telegram_notify is not None and telegram_notify(n["user_id"], n["text"]):
                        continue
                    notes.append(f"(couldn't notify Telegram user {n['user_id']} — tell them yourself)")
                # Discord-side notify is delivered async below via the client.
            if notes:
                reply += "\n" + "\n".join(notes)
            return reply

        await interaction.response.defer(ephemeral=True)
        try:
            reply = await asyncio.to_thread(work)
            # Discord DM notify rides the gateway client we already are.
            if admin:
                try:
                    await discord_user.send(
                        "Your Discord account was linked to a Telegram account by an admin. "
                        "Your conversations and memories now follow you on both."
                    )
                except Exception:
                    log.warning("admin-link DM notify failed for %s", discord_user.id)
        except Exception:
            log.exception("admin-link failed")
            reply = "Sorry — something broke on my end handling that. Try again in a moment?"
        await interaction.followup.send(reply[:1990], ephemeral=True)

    @tree.command(
        name="admin-unlink",
        description="Force-unlink a platform account from its unified identity (admin only)",
        guild=guild,
    )
    @app_commands.describe(
        platform="Platform to unlink from",
        discord_user="Discord user to unlink (Discord only)",
        id="User ID string (Telegram only, or Discord fallback)",
    )
    @app_commands.choices(
        platform=[
            app_commands.Choice(name="Discord", value="discord"),
            app_commands.Choice(name="Telegram", value="telegram"),
        ]
    )
    async def admin_unlink(
        interaction: discord.Interaction,
        platform: app_commands.Choice[str],
        discord_user: discord.User | None = None,
        id: str | None = None,
    ) -> None:
        admin = is_admin_member(interaction.user, admin_role_id)
        target = str(discord_user.id) if discord_user is not None else (id or "")

        def work() -> str:
            if not target:
                return "Give me either a Discord user or an ID string to unlink."
            with conn_factory() as conn:
                return linkmod.admin_unlink(conn, platform.value, target, admin)

        await _run(interaction, work)

    # ---- gaps (self-iteration Phase A read path) ----

    @tree.command(name="gaps", description="Show Aerys's mined capability gaps", guild=guild)
    async def gaps(interaction: discord.Interaction) -> None:
        user_id = interaction.user.id

        def work() -> str:
            if gaps_fn is None:
                return "Gaps aren't wired on this deployment (no database)."
            pid = _resolve(user_id)
            if owner_person_id is None or pid != owner_person_id:
                return "That one's operator telemetry — owner only."
            return gaps_fn()

        await _run(interaction, work)

    # ---- registration on login ----

    # Instance-attribute assignment shadows the bound method (deliberate): the
    # existing AerysDiscordClient doesn't know about trees, and subclassing it
    # for one sync call would couple the gateway to interactions. Chain the
    # original so future setup_hook logic in the client class still runs.
    original_setup_hook = client.setup_hook

    async def setup_hook() -> None:
        await original_setup_hook()
        synced = await tree.sync(guild=guild)
        log.info("slash commands synced: %d commands -> guild %s", len(synced), guild_id)

    client.setup_hook = setup_hook  # type: ignore[method-assign]
    return tree
