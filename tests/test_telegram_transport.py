"""Offline tests for the Telegram transport's pure core — no aiogram Dispatcher, no network.

SimpleNamespace fakes stand in for aiogram Message/User/Chat objects (same
pinned-data trick as test_discord_transport.py). What's proven: every gate
decision in should_handle, mention stripping, DM-vs-group thread keys, and the
normalize field mapping.
"""

from types import SimpleNamespace

from aerys_v2.transports.telegram_gateway import (
    normalize,
    should_handle,
    telegram_thread_key,
)

BOT_USERNAME = "aerys_test_bot"


def gate(**overrides) -> bool:
    base = dict(
        is_bot=False,
        is_dm=False,
        chat_id=-1001,
        allowed_chat_ids=frozenset(),
        mentions_me=True,
    )
    base.update(overrides)
    return should_handle(**base)


def test_drops_bots():
    # bots never summon Aerys — the Kael/Aerys loop-prevention rule
    assert gate(is_bot=True) is False


def test_dm_always_in_no_mention_needed():
    assert gate(is_dm=True, mentions_me=False) is True


def test_group_dropped_by_default_when_no_allowlist_configured():
    # FAIL-CLOSED, mirroring discord's guild lock: with no chat-id allowlist a
    # group is never served — even a direct @mention is dropped. This is the
    # analogue of discord refusing every guild until DISCORD_GUILD_ID is set.
    assert gate(allowed_chat_ids=frozenset()) is False
    assert gate(allowed_chat_ids=frozenset(), mentions_me=True) is False


def test_group_allowlist_enforced_when_set():
    assert gate(allowed_chat_ids=frozenset({-2002})) is False  # -1001 not allowlisted
    assert gate(allowed_chat_ids=frozenset({-1001})) is True   # allowlisted + mentioned


def test_group_requires_mention_even_when_allowlisted():
    # An allowlisted group still needs an @mention to summon her.
    assert gate(allowed_chat_ids=frozenset({-1001}), mentions_me=False) is False
    assert gate(allowed_chat_ids=frozenset({-1001}), mentions_me=True) is True


def fake_user(*, user_id: int, full_name: str = "Chris", username: str | None = "chrisp", is_bot: bool = False):
    return SimpleNamespace(id=user_id, full_name=full_name, username=username, is_bot=is_bot)


def fake_message(*, text: str, chat_type: str, chat_id: int = 555, user=None):
    return SimpleNamespace(
        text=text,
        chat=SimpleNamespace(id=chat_id, type=chat_type),
        from_user=user or fake_user(user_id=123),
    )


def test_normalize_dm():
    ev = normalize(fake_message(text="hey", chat_type="private"), bot_username=BOT_USERNAME)
    assert ev.platform == "telegram"
    assert ev.platform_user_id == "123"
    assert ev.display_name == "Chris"
    assert ev.channel_kind == "dm"
    assert ev.channel_id == "555"
    assert ev.thread_id == "telegram:dm:123"  # DMs follow the person
    assert ev.text == "hey"


def test_normalize_group_strips_mention():
    ev = normalize(
        fake_message(text=f"@{BOT_USERNAME} what's up", chat_type="group", chat_id=-777),
        bot_username=BOT_USERNAME,
    )
    assert ev.channel_kind == "group"
    assert ev.channel_id == "-777"
    assert ev.text == "what's up"
    assert ev.thread_id == "telegram:group:-777"  # groups follow the chat


def test_supergroup_maps_to_group_channel_kind():
    ev = normalize(
        fake_message(text="hi", chat_type="supergroup", chat_id=-888),
        bot_username=BOT_USERNAME,
    )
    assert ev.channel_kind == "group"


def test_normalize_display_name_falls_back_to_username():
    ev = normalize(
        fake_message(
            text="hi",
            chat_type="private",
            user=fake_user(user_id=9, full_name="", username="nobody"),
        ),
        bot_username=BOT_USERNAME,
    )
    assert ev.display_name == "nobody"


def test_thread_keys_are_distinct_namespaces():
    assert telegram_thread_key("dm", "1", "9") != telegram_thread_key("group", "1", "9")
