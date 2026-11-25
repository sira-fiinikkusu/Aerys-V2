"""Secrets and card numbers never become memories.

Found comparing our memory with another open-source agent's (2026-11-24): nothing
stopped a key or a card number pasted into chat from becoming a lasting memory, mined
by the extractor or kept on purpose by the remember tool. Only locations were fenced.
A memory is recalled into later prompts, synced to her other bodies and backed up;
a secret there is a secret copied everywhere she goes.

One shared guard (services/secret_guard.py) now refuses, in the remember tool, in
parse_observations and in triage_memory: card numbers (13-19 digits, a major network's
prefix and the Luhn check, mirroring the Portable body's home wall and its narrow
network list, so order and tracking numbers pass), JWTs, PEM private keys,
`scheme://user:pass@` links, and obvious API-key formats. Ordinary numbers stay
memories: phones, order numbers, years, dates.

Every value here is synthetic. Token-shaped values are assembled at run time so no
token-shaped literal sits in this public file; card numbers are the networks'
published test numbers.
"""
import json
import logging
from unittest.mock import Mock

import pytest

from aerys_v2.services import secret_guard as sg
from aerys_v2.tools.remember import CURRENT_TURN_TEXT, FACT_LIMIT, build_remember_tool
from aerys_v2.workers import extraction as ex
from test_extraction import FakeConn, FakeLlm, fake_embedder, lease_conn, row
from test_memory_hygiene import PERSON

RANDOM = "a1B2c3D4e5F6g7H8i9J0k1L2m3N4o5P6q7R8"           # 36 mixed characters, no meaning
PUNCTUATED = "Xk9!mP2v#L8qR4$nT6w&"                          # 20, with password punctuation
INSIDE = "Xk9mP2vL8qR4nT6w7Zb3c"                             # 21; one quote mark inside makes 22
HEX = "0123456789abcdef" * 3

SECRETS = [
    ("card number", "4111 1111 1111 1111"),
    ("card number", "my card is 4111-1111-1111-1111, exp 12/29"),
    ("card number", "4111111111111111"),
    ("card number", "use the amex 3782 822463 10005 for that"),
    ("card number", "5555 5555 5555 4444"),
    ("card number", "6011111111111117"),
    ("card number", "2223003122003222"),
    ("card number", "4111 1111\n1111 1111"),                            # wrapped onto two lines
    ("card number", "４１１１ 1111 1111 1111"),         # full-width digits
    ("access token", "eyJ" + "hbGciOiJIUzI1NiJ9" + "." + "eyJzdWIiOiJ0ZXN0In0" + "." + RANDOM[:22]),
    ("private key", "-----BEGIN " + "RSA PRIVATE KEY-----\n" + RANDOM),
    ("private key", "-----BEGIN " + "OPENSSH PRIVATE KEY-----"),
    ("private key", "-----BEGIN " + "PRIVATE KEY-----"),
    ("private key", "-----BEGIN " + "ENCRYPTED PRIVATE KEY-----"),
    ("private key", "-----BEGIN " + "PGP PRIVATE KEY BLOCK-----"),
    ("credentials in a link", "postgresql://appuser:" + "examplepass" + "@db.example.com:5432/app"),
    ("credentials in a link", "redis://:" + "examplepass" + "@cache.example.com"),
    ("credentials in a link", "the dashboard is https://admin:" + "hunter22" + "@example.com/x"),
    ("API key", "sk-" + "proj-" + RANDOM),
    ("API key", "sk-" + "ant-api03-" + RANDOM),
    ("API key", "sk-" + "or-v1-" + HEX),
    ("API key", "gh" + "p_" + RANDOM),
    ("API key", "github" + "_pat_" + RANDOM),
    ("API key", "xox" + "b-" + "1234567890-" + RANDOM),
    ("API key", "AKIA" + "IOSFODNN7EXAMPLE"),                         # the provider's documented example
    ("API key", "AIza" + RANDOM[:35]),
    ("API key", "sk_" + "live_" + RANDOM),
    ("labelled secret", "api key: " + HEX),
    ("labelled secret", "OPENAI_API_KEY=" + RANDOM),
    ("labelled secret", "my token is " + RANDOM),
    ("labelled secret", "the wifi password is " + RANDOM[:20]),
    ("labelled secret", "client_secret = '" + RANDOM + "'"),
    ("labelled secret", "Bearer " + RANDOM),
    # Codex review round 1: the value is read whole, punctuation and all, then shape-checked.
    ("labelled secret", "the password is " + PUNCTUATED),
    ("labelled secret", "my password is " + PUNCTUATED + "."),         # a sentence's full stop is not in it
    ("labelled secret", 'password: "' + PUNCTUATED + '"'),
    ("labelled secret", "pwd=" + PUNCTUATED),
    ("labelled secret", "key: " + RANDOM),                             # a bare "key" with : or =
    ("labelled secret", "SOME_SERVICE_KEY=" + RANDOM),
    # Codex review round 2: a quote mark or backtick INSIDE a bare value is part of it;
    # only at the value's start does it open a quoted run. 22 characters each.
    ("labelled secret", "the password is " + INSIDE[:10] + "'" + INSIDE[10:]),
    ("labelled secret", "the password is " + INSIDE[:10] + '"' + INSIDE[10:]),
    ("labelled secret", "the password is " + INSIDE[:10] + "`" + INSIDE[10:]),
    ("labelled secret", "password: `" + INSIDE + "`"),                 # a backtick-quoted value
    ("labelled secret", "password: '" + INSIDE[:6] + "'" + INSIDE[6:]),   # a quote that closes early
    # PINNED (reported 2026-11-24): words and a year after a password label still count;
    # only after a bare "key" does a dashed word identifier pass.
    ("labelled secret", "the wifi password is Sunshine-Garden-2024"),
]

ORDINARY = [
    "My phone number is 555-0100",
    "Call the shop at (555) 555-0100",
    "His number is +1 555 555 0100",
    "Order 112-4567890-1234567 ships Friday",
    "order #1234567890123456",                       # 16 digits, no card network
    "4111 1111 1111 1112",                           # a card's shape that fails Luhn
    "771234567890125",                               # tracking-number shaped
    "1732190000000",                                 # a millisecond timestamp
    "30569309025904",                                # Diners: not on the narrow list
    "3530111333300000",                              # JCB: likewise
    "ZIP 12345, call 555-555-0100",                  # numbers side by side are not a card
    "Born in 1990", "Started the job on 2026-11-24", "Has 2 dogs and 3 cats",
    "The key is to keep the RPM under 3000",
    "Reset my password today; the password manager is great",
    "Got a new key fob for the truck",
    "A risk-averse investor", "Ask-me-anything sessions", "Uses sk-learn at work",
    "Shared the risk-assessment-checklist-2026-v2 doc",   # "sk-" inside a word, long and with digits
    "Repo is at https://example.com/team/project:main",
    "Clone it with ssh://git@example.com/team/project.git",
    "Email me at someone@example.com",
    "-----BEGIN " + "PUBLIC KEY-----", "-----BEGIN " + "CERTIFICATE-----",
    "Commit " + HEX[:40] + " fixed it",              # an unlabelled hash is not a secret
    "Key West trip in 2027",
    "Map key: midnight-blue-with-silver-accents",    # a label, then a long value that is plain words
    # Codex review round 1: a bare "key" needs : or =, and a dashed word identifier is not random.
    "The key is part-number-2026-revision-2",
    "key: part-number-2026-revision-2", "key: part-number-2026-revision-v2",
    "my password is " + PUNCTUATED[:11],                              # short: the short-password line holds
    "Token of appreciation for 20 years of service",
    "",
]


@pytest.mark.parametrize("kind,text", SECRETS)
def test_secrets_are_recognized_by_kind(kind, text):
    assert sg.secret_kind(text) == kind
    assert sg.secret_shaped(text)


@pytest.mark.parametrize("text", ORDINARY)
def test_ordinary_numbers_and_words_are_not_secrets(text):
    assert sg.secret_kind(text) is None and not sg.secret_shaped(text)


def test_non_text_is_not_a_secret():
    assert sg.secret_kind(None) is None and sg.secret_kind(4111111111111111) is None


# The Portable body's own network table (home_wall.card_network), mirrored on purpose.
@pytest.mark.parametrize("number,network", [
    ("4111111111111111", "visa"), ("4012888888881881", "visa"), ("4222222222222", "visa"),
    ("5555555555554444", "mastercard"), ("5105105105105100", "mastercard"),
    ("2223003122003222", "mastercard"), ("2720990000000007", "mastercard"),
    ("378282246310005", "amex"), ("371449635398431", "amex"), ("340000000000009", "amex"),
    ("6011111111111117", "discover"), ("6011000990139424", "discover"),
    ("6500000000000002", "discover"), ("6445644564456445", "discover"),
])
def test_each_network_is_known_by_its_prefix_and_passes_luhn(number, network):
    assert sg.card_network(number) == network and sg.luhn(number)
    assert sg.secret_kind(number) == "card number"


@pytest.mark.parametrize("number", [
    "2220990000000008", "2721000000000004", "5600000000000003", "6440000000000006",
    "3300000000000000", "9000123412341234",
])
def test_numbers_off_the_network_list_or_failing_luhn_are_not_cards(number):
    assert sg.secret_kind(number) is None


# --- the remember tool -------------------------------------------------------------

def remember(fact, writer=None, labeler=None):
    writer = writer or Mock(return_value="insert")
    labeler = labeler or Mock(return_value="basic.note")
    tool = build_remember_tool(writer, key_labeler=labeler)
    token = CURRENT_TURN_TEXT.set(fact)
    try:
        reply = tool.invoke({"fact": fact}, config={"configurable": {"identity": {"user_id": PERSON}}})
    finally:
        CURRENT_TURN_TEXT.reset(token)
    return reply, writer, labeler


@pytest.mark.parametrize("kind,text", SECRETS)
def test_the_remember_tool_refuses_a_secret_plainly_and_never_writes_or_labels_it(kind, text):
    reply, writer, labeler = remember(text)
    assert reply == sg.SECRET_REFUSAL
    writer.assert_not_called()
    labeler.assert_not_called()          # the labeler is a model call: a secret never goes there either


def test_the_refusal_tells_her_plainly_what_she_does_not_keep():
    assert sg.SECRET_REFUSAL.startswith("Nothing was kept:")
    assert "secrets" in sg.SECRET_REFUSAL and "card numbers" in sg.SECRET_REFUSAL


def test_a_pasted_private_key_is_refused_as_a_secret_not_as_too_long():
    """TOO_LONG asks her to split it and call again per piece. Pieces of a key carry
    no header, so the secret check must come first."""
    pem = "-----BEGIN " + "OPENSSH PRIVATE KEY-----\n" + (RANDOM + "\n") * 20
    assert len(pem) > FACT_LIMIT
    reply, writer, _ = remember(pem)
    assert reply == sg.SECRET_REFUSAL
    writer.assert_not_called()


@pytest.mark.parametrize("fact", ["My phone number is 555-0100", "Order 112-4567890-1234567 ships Friday",
                                  "Born in 1990", "Reset my password today"])
def test_the_remember_tool_still_keeps_ordinary_numbers(fact):
    reply, writer, _ = remember(fact)
    assert reply.startswith("Kept:")
    writer.assert_called_once()


# --- extraction -----------------------------------------------------------------

@pytest.mark.parametrize("field", ["value_text", "context", "source_text", "key_label"])
def test_parse_drops_a_secret_in_any_field_and_logs_only_its_kind(field, caplog):
    caplog.set_level(logging.WARNING)
    card = "4111 1111 1111 1111"
    observation = {"key_label": "basic.payment", "value_text": "Uses a Visa card",
                   "source_text": "I use a Visa card", "context": ""}
    observation[field] = card if field != "key_label" else "basic.card_" + card.replace(" ", "")
    assert ex.parse_observations(json.dumps([observation])) == []
    assert "secret-shaped" in caplog.text and "card number" in caplog.text
    assert "4111" not in caplog.text


def test_parse_reads_a_bare_number_as_text():
    assert ex.parse_observations(json.dumps([{"key_label": "basic.payment", "value_text": 4111111111111111,
                                              "source_text": "I use a Visa card"}])) == []


def test_triage_refuses_a_secret_before_any_query():
    conn = Mock()
    action = ex.triage_memory(conn, person_id=PERSON, key_label="basic.payment",
                              value_text="Card is 4111 1111 1111 1111", content="basic.payment: Card",
                              context=None, event_date=None, embedding="[0.1]", source_platform="house",
                              privacy_level="private", created_at="2026-11-24T00:00:00+00:00")
    assert action == "skipped"
    conn.execute.assert_not_called()


def test_the_house_remember_path_through_triage_never_writes_a_secret():
    conn = Mock()

    def writer(record):
        return ex.triage_memory(conn, person_id=record["person_id"], key_label=record["key_label"],
                                value_text=record["fact"], content=record["fact"], context=None,
                                event_date=None, embedding="[0.1]", source_platform="house",
                                privacy_level="private", created_at="2026-11-24T00:00:00+00:00")

    assert ex.triage_memory(conn, person_id=PERSON, key_label="basic.note", value_text="api key: " + HEX,
                            content="basic.note: api key: " + HEX, context=None, event_date=None,
                            embedding="[0.1]", source_platform="house", privacy_level="private",
                            created_at="2026-11-24T00:00:00+00:00") == "skipped"
    reply, _, _ = remember("my token is " + RANDOM, writer=writer)
    assert reply == sg.SECRET_REFUSAL
    conn.execute.assert_not_called()


def test_a_card_pasted_in_chat_never_becomes_a_mined_memory():
    message = "Here's my card for the order: 4111 1111 1111 1111"
    prod = FakeConn([("FROM n8n_chat_histories", [row(PERSON, message)])])
    staging = lease_conn(holder="brain")
    prod_write = FakeConn([("FROM memories", [])])
    reply = json.dumps([{"key_label": "basic.payment", "value_text": "Card number 4111 1111 1111 1111",
                         "context": "", "source_text": "my card for the order: 4111 1111 1111 1111",
                         "event_date": "", "privacy_level": "private"}])
    summary = ex.run_live_extraction(prod, staging, prod_write, FakeLlm([reply]), fake_embedder)
    assert summary["inserted_total"] == 0
    assert not any(s.lstrip().upper().startswith("INSERT") for s, _ in prod_write.calls)
