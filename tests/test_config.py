import pytest
from pydantic import ValidationError
from aerys_v2.config import Settings


def test_settings_loads_with_key(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    settings = Settings(_env_file=None)
    assert settings.model == "claude-sonnet-5-5"
    assert settings.otlp_endpoint is None


def test_settings_requires_api_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_telegram_settings_default_off(monkeypatch):
    # Telegram transport is OFF by default — same arming pattern as discord_bot_token.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    s = Settings(_env_file=None)
    assert s.telegram_bot_token is None
    assert s.telegram_chat_ids == ""


def test_telegram_token_read_as_secret(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123:abc")
    monkeypatch.setenv("TELEGRAM_CHAT_IDS", "-1001,-1002")
    s = Settings(_env_file=None)
    assert s.telegram_bot_token is not None
    assert s.telegram_bot_token.get_secret_value() == "123:abc"
    assert s.telegram_chat_ids == "-1001,-1002"


# ---- empty API_TOKEN must fail closed (cross-review CRITICAL, 2026-07-04) -------


def test_blank_api_token_coerces_to_none(monkeypatch):
    # "API_TOKEN=" parses to SecretStr('') not None; the validator coerces it to
    # None so the --serve gate refuses to start rather than accept "Bearer " as auth.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setenv("API_TOKEN", "")
    assert Settings(_env_file=None).api_token is None


def test_whitespace_api_token_coerces_to_none(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setenv("API_TOKEN", "   ")
    assert Settings(_env_file=None).api_token is None


def test_real_api_token_is_kept(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setenv("API_TOKEN", "a-real-token-value")
    s = Settings(_env_file=None)
    assert s.api_token is not None and s.api_token.get_secret_value() == "a-real-token-value"


def test_malformed_owner_person_id_refuses_to_boot(monkeypatch):
    from aerys_v2.config import BootConfigError, run_boot_assertions

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setenv("OWNER_PERSON_ID", "not-a-uuid")
    s = Settings(_env_file=None)
    with pytest.raises(BootConfigError):
        run_boot_assertions(s, env_file=None)


def test_valid_owner_person_id_boots_clean(monkeypatch):
    from aerys_v2.config import run_boot_assertions

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setenv("OWNER_PERSON_ID", "6e6bcbed-03ef-4d17-95d2-89c467414335")
    run_boot_assertions(Settings(_env_file=None), env_file=None)  # no raise


def test_voice_confirm_writes_defaults_on_and_can_be_turned_off(monkeypatch):
    # Owner ruling 2026-10-01: device writes are confirmed aloud by default.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    assert Settings(_env_file=None).voice_confirm_writes is True
    monkeypatch.setenv("VOICE_CONFIRM_WRITES", "false")
    assert Settings(_env_file=None).voice_confirm_writes is False


# ---- G2: the database URLs carry the NAS password: secrets like every other credential -------------

_V2 = "postgresql://placeholder-user:placeholder-secret-1@db.invalid:5432/aerys_v2"
_PROD = "postgresql://placeholder-user:placeholder-secret-2@db.invalid:5432/aerys"
_DB_FIELDS = ("database_url", "memories_database_url")


def _db_settings(**extra):
    return Settings(_env_file=None, anthropic_api_key="sk-test", database_url=_V2, memories_database_url=_PROD,
                    **extra)


def test_settings_never_print_the_database_urls():
    import logging
    s = _db_settings()
    for shown in (repr(s), str(s), s.model_dump_json(), f"{s}", logging.Formatter().format(
            logging.LogRecord("t", logging.INFO, __file__, 1, "%s", (s,), None))):
        assert "placeholder-secret" not in shown
    assert "placeholder-secret" not in repr(s.database_url) + str(s.memories_database_url)


def test_db_url_hands_every_consumer_the_plain_url():
    from pydantic import SecretStr
    from aerys_v2.config import db_url
    s = _db_settings()
    assert db_url(s.database_url) == _V2 and db_url(s.memories_database_url) == _PROD
    assert db_url(SecretStr("postgresql://x")) == "postgresql://x"
    assert db_url("postgresql://x") == "postgresql://x"          # test doubles and plain strings pass through
    assert db_url(None) is None
    unset = Settings(_env_file=None, anthropic_api_key="sk-test")
    assert unset.database_url is None and unset.memories_database_url is None
    assert db_url(unset.database_url) is None


def _db_url_reads():
    """Every read of a database URL off settings in src, with whether it's safe: unwrapped by db_url(...),
    or only tested (is None / is not None / truthiness). Anything else would hand a SecretStr to a driver."""
    import ast
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "src" / "aerys_v2"
    found = []
    for path in sorted(root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}

        def tested(node):
            up = parents.get(node)
            while isinstance(up, ast.BoolOp):                       # a and b / a or b, used as a test
                node, up = up, parents.get(up)
            if isinstance(up, ast.UnaryOp) and isinstance(up.op, ast.Not):
                return True
            if isinstance(up, (ast.If, ast.While, ast.IfExp, ast.Assert)) and up.test is node:
                return True
            return (isinstance(up, ast.Compare) and len(up.ops) == 1 and isinstance(up.ops[0], (ast.Is, ast.IsNot))
                    and isinstance(up.comparators[0], ast.Constant) and up.comparators[0].value is None)

        def unwrapped(node):
            up = parents.get(node)
            return (isinstance(up, ast.Call) and up.args and up.args[0] is node
                    and getattr(up.func, "id", getattr(up.func, "attr", None)) == "db_url")

        for node in ast.walk(tree):
            read = None
            if (isinstance(node, ast.Attribute) and node.attr in _DB_FIELDS and isinstance(node.ctx, ast.Load)
                    and not (isinstance(node.value, ast.Name) and node.value.id == "self")):
                read = node
            elif (isinstance(node, ast.Call) and getattr(node.func, "id", None) == "getattr" and len(node.args) >= 2
                  and isinstance(node.args[1], ast.Constant) and node.args[1].value in _DB_FIELDS):
                read = node
            if read is not None:
                found.append((f"{path.relative_to(root)}:{read.lineno}", unwrapped(read) or tested(read)))
    return found


def test_every_reader_of_a_database_url_unwraps_it():
    reads = _db_url_reads()
    assert len(reads) > 80, "the scan found the readers"
    assert [where for where, safe in reads if not safe] == []


def _captured_conninfo(monkeypatch):
    """psycopg.connect, replaced: records what each consumer would dial (and dials nothing)."""
    import psycopg
    seen = []

    def connect(conninfo="", *args, **kwargs):
        seen.append(conninfo)
        raise ConnectionError("offline test")

    monkeypatch.setattr(psycopg, "connect", connect)
    return seen


def test_consumers_still_get_the_plain_url_from_real_settings(monkeypatch):
    """Real Settings (SecretStr URLs) through real consumers: each dials the plain URL, never a SecretStr."""
    from aerys_v2 import factory, followups, kael_line
    from aerys_v2.hands_gate import hands_gate_for, people_lookup_for

    owner = "6e6bcbed-0000-4000-8000-000000000001"
    s = _db_settings(owner_person_id=owner, approvals=True, hands_gates=True)
    seen = _captured_conninfo(monkeypatch)
    assert factory.room_names_for(s) == {}                           # fail-open read of the memories DB
    for call in (lambda: followups.plan_miner_turns(s, None), lambda: kael_line.owner_replied_since(s, None),
                 lambda: people_lookup_for(s)("email", "someone@example.net")):
        with pytest.raises(ConnectionError):
            call()
    assert seen == [_PROD, _V2, _V2, _PROD] and all(type(c) is str for c in seen)
    monkeypatch.setattr("aerys_v2.hands_gate.approvals_for", lambda settings: object())   # the button, armed
    gate = hands_gate_for(s)
    assert gate is not None and gate.store.database_url == _V2 and type(gate.store.database_url) is str


def test_boot_assertions_read_the_secret_urls():
    from aerys_v2.config import BootConfigError, run_boot_assertions
    run_boot_assertions(_db_settings(), env_file=None)
    with pytest.raises(BootConfigError):
        run_boot_assertions(Settings(_env_file=None, anthropic_api_key="sk-test",
                                     database_url=_PROD), env_file=None)
