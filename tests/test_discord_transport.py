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


def fake_message(*, content: str, guild: object | None, attachments=()):
    return SimpleNamespace(
        guild=guild,
        content=content,
        author=SimpleNamespace(id=123, name="chris", display_name="Chris"),
        channel=SimpleNamespace(id=555),
        attachments=[SimpleNamespace(url=u) for u in attachments],
    )


CDN = "https://cdn.discordapp.com/attachments/1/2/pic.png?ex=abc&is=def&hm=deadbeef"


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


def test_normalize_appends_attachment_url():
    ev = normalize(
        fake_message(content="what's in this", guild=None, attachments=(CDN,)),
        self_id=SELF_ID,
    )
    assert ev.text == f"what's in this\n{CDN}"  # router keys off the CDN URL in text


def test_normalize_image_only_is_not_empty():
    # image with no caption used to crash ask() with "requires non-empty text"
    ev = normalize(
        fake_message(content="", guild=None, attachments=(CDN,)), self_id=SELF_ID
    )
    assert ev.text == CDN
    assert ev.text.strip()  # non-empty — the crash case is gone


def test_normalize_preserves_cdn_signature():
    ev = normalize(
        fake_message(content="", guild=None, attachments=(CDN,)), self_id=SELF_ID
    )
    assert "?ex=abc&is=def&hm=deadbeef" in ev.text  # signature is load-bearing


def test_normalize_multiple_attachments_all_appended():
    a = "https://cdn.discordapp.com/attachments/1/2/a.png?hm=1"
    b = "https://cdn.discordapp.com/attachments/1/2/b.pdf?hm=2"
    ev = normalize(
        fake_message(content="look", guild=SimpleNamespace(id=42), attachments=(a, b)),
        self_id=SELF_ID,
    )
    assert ev.text == f"look\n{a}\n{b}"


def test_normalize_bare_mention_is_empty():
    # a naked @mention (no text, no attachment) normalizes to empty text — the
    # gateway nudges instead of handing "" to ask() (the crash the guard prevents)
    ev = normalize(
        fake_message(content=f"<@{SELF_ID}>", guild=SimpleNamespace(id=42)),
        self_id=SELF_ID,
    )
    assert ev.text == ""


# --- A dropped summon must not be silent (2026-09-13) ---------------------------
#
# Stratus asked her a direct question in a guild channel on 09-11 and nothing
# happened: no turn, no reply, no refusal, and nothing in any log, because
# on_message returns on a False gate without a word. Chris read the silence as her
# choosing not to answer. Nobody could tell him otherwise, which is the actual
# defect: the drop is correct sometimes, invisible always.

def test_drop_reason_names_why_each_message_was_not_handled():
    from aerys_v2.transports.discord_gateway import drop_reason
    assert drop_reason(**gate_args()) is None, 'a handled message has no reason'
    assert drop_reason(**gate_args(author_is_self=True)) == 'self'
    assert drop_reason(**gate_args(author_is_bot=True)) == 'bot'
    assert drop_reason(**gate_args(guild_id=7)) == 'other-guild'
    assert drop_reason(**gate_args(allowed_channel_ids=frozenset({5}))) == 'channel-not-allowed'
    assert drop_reason(**gate_args(mentions_me=False)) == 'no-mention'
    assert drop_reason(**gate_args(is_dm=True, mentions_me=False)) is None, 'a DM needs no mention'


def test_a_message_that_names_her_without_a_real_mention_is_flagged_loudly():
    """The likeliest cause of a vanished summon: the name typed, not the mention picked."""
    from aerys_v2.transports.discord_gateway import looks_like_a_summon
    assert looks_like_a_summon('Aerys - tell me what changed this month', names=('aerys',))
    assert looks_like_a_summon('hey AERYS you around?', names=('aerys',))
    assert looks_like_a_summon('@Aerys - Resonant Span', names=('aerys', 'resonant span'))
    assert not looks_like_a_summon('the barometric pressure is falling', names=('aerys',))
    assert not looks_like_a_summon('', names=('aerys',))


def gate_args(**overrides):
    base = dict(
        author_is_self=False,
        author_is_bot=False,
        is_dm=False,
        guild_id=42,
        allowed_guild_id=42,
        channel_id=1,
        allowed_channel_ids=frozenset(),
        mentions_me=True,
    )
    base.update(overrides)
    return base


def test_the_gate_and_the_reason_never_disagree():
    from aerys_v2.transports.discord_gateway import drop_reason
    for overrides in ({}, {'author_is_self': True}, {'author_is_bot': True},
                      {'guild_id': 7}, {'allowed_channel_ids': frozenset({5})},
                      {'mentions_me': False}, {'is_dm': True, 'mentions_me': False},
                      {'is_dm': True, 'author_is_bot': True}):
        args = gate_args(**overrides)
        assert should_handle(**args) is (drop_reason(**args) is None), overrides
