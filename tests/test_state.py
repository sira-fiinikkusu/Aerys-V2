from aerys_v2.state import (
    ChatState,
    UNKNOWN_CALLER,
    identity_from_config,
)  # mirrors test_config's `from aerys_v2.config import Settings`


def test_state_is_messages_only():
    assert set(ChatState.__annotations__.keys()) == {"messages"}


def test_returns_identity_when_present():
    config = {"configurable": {"identity": {"user_id": "u1"}}}
    assert identity_from_config(config) == {"user_id": "u1"}


def test_returns_unknown_caller_when_identity_is_missing():
    config = {"configurable": {}}
    assert identity_from_config(config) == UNKNOWN_CALLER


def test_returns_unknown_caller_when_identity_is_none():
    config = {"configurable": {"identity": None}}
    assert identity_from_config(config) == UNKNOWN_CALLER


def test_returns_unknown_caller_when_configurable_is_missing():
    config = {}
    assert identity_from_config(config) == UNKNOWN_CALLER


def test_returns_unknown_caller_when_configurable_is_none():
    config = {"configurable": None}
    assert identity_from_config(config) == UNKNOWN_CALLER


def test_returns_unknown_caller_when_config_is_none():
    config = None
    assert identity_from_config(config) == UNKNOWN_CALLER
