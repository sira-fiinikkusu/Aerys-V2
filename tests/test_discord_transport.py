"""Offline tests for the Discord transport's pure core — no gateway, no network.

SimpleNamespace fakes stand in for discord.py Message objects (same pinned-data
trick as everywhere else). What's proven: every gate decision in should_handle,
mention stripping, DM-vs-guild thread keys, and the normalize field mapping.
"""

from types import SimpleNamespace

from aerys_v2.transports.discord_gateway import (
    normalize,
    should_handle,
    thread_key,
)

SELF_ID = 999


def gate(**overrides) -> bool:
    base = dict(
        author_is_self=False,
        author_is_bot=False,
        is_dm=False,
        guild_id=42,
        allowed_guild_id=42,
        channel_id=7,
        allowed_channel_ids=frozenset(),
        mentions_me=True,
    )
    base.update(overrides)
    return should_handle(**base)


def test_drops_own_messages():
    assert gate(author_is_self=True) is False


def test_drops_other_bots():
    # bots never summon Aerys — the Kael/Aerys loop-prevention rule
    assert gate(author_is_bot=True) is False


def test_dm_always_in_no_mention_needed():
    assert gate(is_dm=True, guild_id=None, mentions_me=False) is True


def test_wrong_guild_dropped():
    assert gate(guild_id=41) is False


def test_no_guild_configured_drops_guild_traffic():
    assert gate(allowed_guild_id=None) is False


def test_channel_allowlist_enforced_when_set():
    assert gate(allowed_channel_ids=frozenset({8})) is False
    assert gate(allowed_channel_ids=frozenset({7})) is True


def test_guild_requires_mention():
    assert gate(mentions_me=False) is False


def fake_message(*, content: str, guild: object | None):
    return SimpleNamespace(
        guild=guild,
        content=content,
        author=SimpleNamespace(id=123, name="chris", display_name="Chris"),
        channel=SimpleNamespace(id=555),
    )


def test_normalize_dm():
    ev = normalize(fake_message(content="hey", guild=None), self_id=SELF_ID)
    assert ev.channel_kind == "dm"
    assert ev.thread_id == "discord:dm:123"  # DMs follow the person
    assert ev.display_name == "Chris"
    assert ev.text == "hey"


def test_normalize_guild_strips_mention_both_forms():
    for tok in (f"<@{SELF_ID}>", f"<@!{SELF_ID}>"):
        ev = normalize(
            fake_message(content=f"{tok} what's up", guild=SimpleNamespace(id=42)),
            self_id=SELF_ID,
        )
        assert ev.text == "what's up"
        assert ev.thread_id == "discord:guild:555"  # guild follows the channel


def test_thread_keys_are_distinct_namespaces():
    assert thread_key("dm", "1", "9") != thread_key("guild", "1", "9")
