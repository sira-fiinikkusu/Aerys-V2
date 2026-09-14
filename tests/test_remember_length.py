"""A fact too long to keep must be refused, not quietly halved.

Live 2026-09-13: a memory from ember came back through the door flagged unclear, and
Chris was unhappy it had been cut. It was exactly 500 characters and ended mid-word —
`... a "dreaming" process that helps ensure she doesn't get poisoned by ba`. The only
memory in the database at exactly 500; every other one is shorter.

The cause was here:

    if len(text) > FACT_LIMIT:
        text = text[:FACT_LIMIT].rstrip()

A flat character slice. No word boundary, no ellipsis, nothing telling her or him that
anything was lost, and the embedding then gets built from the fragment — so the half
that survived is also what she searches by.

The content that got cut was four distinct facts run together in one call, which is
also WHY the judge called it unclear. That is the real shape of an over-long fact: not
one long fact, but several that were never separated. So the fix is to refuse and say
so, the way this tool already refuses a question. She still has the text in front of
her and can call again, once per fact, which stores better and recalls better.

Silently keeping half is worse than keeping none: none is visible.
"""
import pytest

from aerys_v2.tools.remember import FACT_LIMIT, TOO_LONG, build_remember_tool

OWNER = "6e6bcbed-03ef-4d17-95d2-89c467414335"


def tool(written):
    def writer(record):
        written.append(record)
        return "insert"

    return build_remember_tool(writer, key_labeler=lambda text: "remember.test")


def call(tool_obj, fact):
    return tool_obj.invoke(
        {"fact": fact},
        config={"configurable": {"identity": {"user_id": OWNER}}},
    )


def test_an_over_long_fact_is_refused_and_nothing_is_written():
    written = []
    out = call(tool(written), "x " * (FACT_LIMIT))       # comfortably over
    assert written == [], "a fact we will not keep whole must not be kept in part"
    assert out == TOO_LONG


def test_the_refusal_says_what_to_do_and_names_the_limit():
    low = TOO_LONG.lower()
    assert "nothing was kept" in low, "match the vocabulary of the other refusals"
    assert str(FACT_LIMIT) in TOO_LONG, "tell her the actual number"
    # it must point at the real cause: several facts in one call
    assert "one" in low and ("separate" in low or "split" in low or "each" in low)


def test_a_fact_at_the_limit_is_still_kept():
    written = []
    fact = "a" * FACT_LIMIT
    assert call(tool(written), fact).startswith("Kept:")
    assert written[0]["fact"] == fact, "exactly at the limit is not over it"


def test_nothing_is_ever_stored_truncated():
    """The property that actually matters, whatever the limit becomes."""
    written = []
    obj = tool(written)
    for fact in ("short one", "b" * (FACT_LIMIT - 1), "c" * FACT_LIMIT, "d" * (FACT_LIMIT + 1)):
        call(obj, fact)
    assert all(len(record["fact"]) <= FACT_LIMIT for record in written)
    assert all(record["fact"][-1] != "…" for record in written), "no cut marker either"
    assert len(written) == 3, "the over-long one was refused, not trimmed"


def test_the_tool_tells_her_the_rule_before_she_hits_it():
    """A refusal she could have avoided is a wasted turn. The description carries
    the rule, so the usual path is one fact per call rather than a retry loop."""
    doc = build_remember_tool(lambda r: "insert").description
    assert "ONE fact per call" in doc
    assert str(FACT_LIMIT) in doc
    assert "refused" in doc and ("trim" in doc or "half" in doc)


def test_the_limit_stays_inside_the_SMALLEST_embedder():
    """A guard against a future me raising this because 500 looks small.

    Two embedders read the same memory. The house uses text-embedding-3-small, which
    could take far more. Her portable body embeds on-device with all-MiniLM-L6-v2,
    max sequence 256 tokens, and it truncates past that WITHOUT SAYING SO. Raising
    the character limit past roughly a thousand would move the mid-thought cut from
    the text (visible, fixable) into the vector (invisible, unfixable) — the same bug
    one layer down. Adversarial review, 2026-09-13, asked why 500; this is the answer.
    """
    conservative_chars_per_token = 4          # English prose runs ~4; shorter is safer
    smallest_window_tokens = 256              # all-MiniLM-L6-v2
    assert FACT_LIMIT <= smallest_window_tokens * conservative_chars_per_token


def test_a_multibyte_fact_is_measured_the_same_way_as_any_other():
    """len() counts code points, and the store takes text, not bytes. A fact of
    emoji or accented script must not be refused for being under the limit in
    characters but over it in bytes."""
    written = []
    obj = tool(written)
    fact = "🏳️‍🌈" * 40                        # far over FACT_LIMIT in BYTES
    assert len(fact) <= FACT_LIMIT, 'precondition: under the limit in characters'
    assert call(obj, fact).startswith("Kept:")
    assert written[0]["fact"] == fact, 'stored whole, not re-encoded or clipped'
