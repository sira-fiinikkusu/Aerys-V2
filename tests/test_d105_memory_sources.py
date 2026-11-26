"""D105 (Chris 11/25 20:27): every memory she recalls says where it came from.

Since 7/04 recall introduced every memory as "user-reported facts". From 9/06 she could also keep things herself
(the remember tool) and her home body writes its own notes; Kael's approved notes on his records are memories too.
None of that was ever shown, so all of it read as things he said ("im not sure how we got on the whole path that her
memories of everything is funneled as spoken by me"). Each mark comes from what the row already carries (its key,
its context, its tags); nothing is guessed from the text, and a line in their own words stays unmarked.
"""
from datetime import datetime, timezone
from types import SimpleNamespace

from aerys_v2.services.memory import (HER_PHRASING_LABEL, HER_WORDS_LABEL, KAEL_NOTES_LABEL, KAEL_TALK_LABEL,
                                      OWN_NOTE_LABEL,
                                      SOURCE_NOTES, format_memory_context_flagged, format_memory_context_marked,
                                      source_mark, source_notes)

NOW = datetime(2026, 11, 25, 12, tzinfo=timezone.utc)
DAY = datetime(2026, 11, 20, 12, tzinfo=timezone.utc)


def row(content, **extra):
    return {**dict(content=content, source_platform=None, created_at=DAY), **extra}


def test_each_origin_gets_its_mark_from_what_the_row_carries():
    cases = [
        (row("aerys.said: I'll check the attic fan", key_label="aerys.said"), "hers", HER_WORDS_LABEL),
        (row("basic.wife: Dana is his wife", category=["trust:owner", "remember"]), "", ""),
        (row("x: The library opens at noon", category=["trust:assistant", "remember", "from:read_web_page"]),
         "read", "(saved after she read: a web page)"),
        (row("x: Mail says the order shipped", category=["trust:assistant", "remember", "from:read_email",
                                                         "from:read_web_page"]),
         "read", "(saved after she read: an email, a web page)"),
        (row("gap: a turn ran no tool", context="portable gap report", category=["trust:assistant", "remember"]),
         "own_note", OWN_NOTE_LABEL),
        (row("exec.home.exec-1: note", context="note to self from unattended run exec-1"), "own_note", OWN_NOTE_LABEL),
        (row("skillreview.x: reviewed", context="skill review mark"), "own_note", OWN_NOTE_LABEL),
        (row("Chris pinged me", key_label="session.stick.2026-09-12"), "own_note", OWN_NOTE_LABEL),
        (row("basic.pets: three cats", context="from Kael's notes on Chris's records, approved by Chris"),
         "kael", KAEL_NOTES_LABEL),
        (row("pref.voice: shorter voice replies", category=["trust:assistant", "remember"]),
         "her_phrasing", HER_PHRASING_LABEL),
        (row("interest.music: likes synthwave", context="Mentioned while sharing a playlist"), "", ""),
        (row("plain memory"), "", ""),
        # Replay 11/25: a talk with Kael came back unmarked in per-message recall.
        (row("12): Kael said: hi", source_platform="kael_line"), "kael_talk", KAEL_TALK_LABEL),
        (row("x: y", context="A2A exchange on kael:warmup"), "kael_talk", KAEL_TALK_LABEL),
    ]
    for r, kind, label in cases:
        assert source_mark(r) == (kind, label), r


def test_an_unknown_read_source_is_still_marked_as_read_never_as_theirs():
    assert source_mark(row("x: y", category=["from:something_new"])) == (
        "read", "(saved after she read: an outside source)")


def test_the_recalled_lines_carry_their_marks_and_the_flag_api_still_works():
    rows = [row("pref.voice: shorter voice replies", category=["trust:assistant", "remember"]),
            row("basic.pets: three cats", context="from Kael's notes on Chris's records, approved by Chris"),
            row("interest.music: likes synthwave"),
            row("aerys.said: I'll check the fan")]
    block, marks = format_memory_context_marked(rows, now=NOW)
    lines = block.split("\n")
    assert lines[0].startswith(f"* {HER_PHRASING_LABEL} shorter voice replies")
    assert lines[1].startswith(f"* {KAEL_NOTES_LABEL} three cats")
    assert lines[2].startswith("* likes synthwave (")
    assert lines[3].startswith(f"* {HER_WORDS_LABEL} I'll check the fan")
    assert marks == {"her_phrasing", "kael", "hers"}
    assert format_memory_context_flagged(rows, now=NOW) == (block, True)


def test_notes_explain_only_the_marks_shown_and_never_call_her_phrasing_not_theirs():
    notes = source_notes({"her_phrasing", "kael"})
    assert notes == SOURCE_NOTES["kael"] + "\n" + SOURCE_NOTES["her_phrasing"]
    # The side agent's catch (11/25 20:31): many lines she saved in her own words restate what he told her. The note
    # must not make her disown them, only stop her quoting them as his exact words.
    assert "many restate what they told you" in SOURCE_NOTES["her_phrasing"]
    # Codex D105 r1: reading something before saving isn't proof the fact came from it: no categorical denial.
    assert "may come from it or from them" in SOURCE_NOTES["read"] and "never say" not in SOURCE_NOTES["read"]
    assert source_notes(set()) == ""


def test_the_prompt_block_says_where_lines_came_from(monkeypatch):
    from aerys_v2.services import context
    rows = [row("pref.voice: shorter voice replies", category=["trust:assistant", "remember"])]
    monkeypatch.setattr(context, "retrieve_memories", lambda *a, **k: rows)
    monkeypatch.setattr(context, "get_profile", lambda *a, **k: [])
    block = context.build_context("6e6bcbed-03ef-4d17-95d2-89c467414335", "voice?", object(),
                                  embed=lambda q: [0.0])
    assert "user-reported" not in block
    assert "unmarked lines were drawn from what they said" in block
    assert HER_PHRASING_LABEL in block and SOURCE_NOTES["her_phrasing"] in block


def test_search_results_carry_the_same_marks():
    from aerys_v2.services import recall_search as rs
    scope = SimpleNamespace(person_id="p1", own=True, unsure="", levels=("public",))
    rows = [dict(person_id="p1", content="x: The library opens at noon", key_label="x", created_at=DAY,
                 category=["trust:assistant", "remember", "from:read_web_page"]),
            dict(person_id="p1", content="basic.wife: Dana is his wife", key_label="basic.wife", created_at=DAY,
                 category=["trust:owner", "remember"])]
    out = rs.format_memories(rows, "library", scope)
    assert "facts people reported" not in out
    assert "(saved after she read: a web page) The library opens at noon" in out
    assert "· Dana is his wife" in out
    assert out.endswith(SOURCE_NOTES["read"])


def test_the_people_block_carries_the_marks_too():
    from aerys_v2.services.people_memory import PEOPLE_SQL, WHO_SQL, person_section

    class Conn:
        def execute(self, sql, params=None):
            rows = [] if sql == WHO_SQL else [("Sam likes escape rooms", None, DAY, "interest.sam",
                                                "portable gap report", ["trust:assistant", "remember"])]
            return SimpleNamespace(fetchall=lambda: rows)
    out = person_section(Conn(), "p1", "Sam", "public")
    assert f"{OWN_NOTE_LABEL} Sam likes escape rooms" in out and out.endswith(SOURCE_NOTES["own_note"])


def test_old_three_column_people_rows_still_read_unmarked():
    from aerys_v2.services.people_memory import WHO_SQL, person_section

    class Conn:
        def execute(self, sql, params=None):
            rows = [] if sql == WHO_SQL else [("Sam likes escape rooms", None, DAY)]
            return SimpleNamespace(fetchall=lambda: rows)
    out = person_section(Conn(), "p1", "Sam", "public")
    assert "* Sam likes escape rooms (" in out and "Lines marked" not in out


# ---- Codex D105 r1 regressions ----

def test_a_negation_on_one_side_only_is_never_his_words():
    """Codex r1 #1: he says "I do not like coffee", she keeps "I like coffee": word overlap called it his."""
    from aerys_v2.tools.remember import trust_for
    assert trust_for("I like coffee.", "I do not like coffee.") == "assistant"
    assert trust_for("Chris eats pork", "I don't eat pork") == "assistant"
    assert trust_for("I do not like coffee", "I do not like coffee") == "owner"
    assert trust_for("I like coffee", "remember I like coffee") == "owner"


def test_a_rewrite_clears_the_old_texts_provenance_in_the_same_statement():
    """Codex r1 #3: an extraction refresh rewrote a row's text from his words but kept its old trust:/from: tags,
    so his quoted statement stayed labeled as web-derived. The rewrite clears them in the same UPDATE, and a
    writer with provenance (remember) replaces both kinds of tag, never stacking an old from: on new text."""
    from aerys_v2.workers.extraction import TRIAGE_UPDATE_SQL, triage_memory
    assert "NOT LIKE 'trust:%%'" in TRIAGE_UPDATE_SQL and "NOT LIKE 'from:%%'" in TRIAGE_UPDATE_SQL

    class Conn:
        def __init__(self):
            self.calls = []

        def execute(self, sql, params=None):
            self.calls.append((sql, params))
            rows = [("existing-id", "basic.location: Lives in Exampleville")] if "FROM memories" in sql and \
                sql.lstrip().upper().startswith("SELECT") else [("existing-id",)]
            return type("R", (), {"fetchone": lambda self: rows[0] if rows else None,
                                  "fetchall": lambda self: rows})()
    conn = Conn()
    triage_memory(conn, person_id="p", key_label="basic.location", value_text="Lives in Exampleville",
                  content="basic.location: Lives in Exampleville", context=None, event_date=None,
                  embedding="[0.1]", source_platform="discord", privacy_level="public",
                  created_at="2026-11-25 12:00:00+00", category=["trust:owner", "remember"])
    tagged = [s for s, p in conn.calls if s.lstrip().upper().startswith("UPDATE") and "%(category)s" in s]
    assert tagged and all("NOT LIKE 'from:%%'" in s for s in tagged)


def test_a_refresh_that_keeps_the_text_keeps_its_source_tags():
    """Codex r2: an undated restatement of a dated event only refreshes the row (its text stays): its own
    trust:/from: tags still describe it, so the writer's provenance tags must not replace them."""
    from aerys_v2.workers.extraction import triage_memory

    class Conn:
        def __init__(self):
            self.calls = []

        def execute(self, sql, params=None):
            self.calls.append((sql, params))
            if sql.lstrip().upper().startswith("SELECT"):
                rows = [("ev-1", "event.concert: Saw the band at the park", "2026-11-01")]
            else:
                rows = [("ev-1",)]
            return type("R", (), {"fetchone": lambda self: rows[0], "fetchall": lambda self: rows})()
    conn = Conn()
    action = triage_memory(conn, person_id="p", key_label="event.concert", value_text="Saw the band at the park",
                           content="event.concert: Saw the band at the park", context=None, event_date=None,
                           embedding="[0.1]", source_platform="portable", privacy_level="public",
                           created_at="2026-11-25 12:00:00+00", category=["trust:owner", "portable"])
    assert action == "update"
    tagged = [(s, p) for s, p in conn.calls if s.lstrip().upper().startswith("UPDATE") and "%(category)s" in s]
    assert len(tagged) == 1 and tagged[0][1]["kept_text"] is True


def test_a_remember_request_phrased_with_dont_forget_still_counts_as_his_words():
    """A side agent's catch (11/25): "don't forget ..." / "never forget ..." / "No, remember ..." are requests, not
    negated facts; a real negation inside the fact still decides."""
    from aerys_v2.tools.remember import trust_for
    assert trust_for("my tattoo appointment is Dec 19", "don't forget my tattoo appointment is Dec 19") == "owner"
    assert trust_for("my tattoo appointment is Dec 19", "Don’t let me forget: my tattoo appointment is Dec 19") == "owner"
    # Codex r4: a leading "No," is never removed (it can be a correction), so this one stays hers.
    assert trust_for("I like coffee", "No, remember that I like coffee") == "assistant"
    assert trust_for("I don't like coffee", "never forget: I don't like coffee") == "owner"
    assert trust_for("I like coffee", "don't forget I don't like coffee") == "assistant"
    assert trust_for("coffee is fine", "No coffee for me, remember that") == "assistant"
    assert trust_for("my appointment is Dec 19", "It moved. Don't forget my appointment is Dec 19") == "owner"


def test_a_negation_that_belongs_to_his_statement_is_never_stripped():
    """Codex D105 r3: only a sentence-leading request is removed; a declarative "never/do not/not to forget" stays."""
    from aerys_v2.tools.remember import trust_for
    for fact, said in [("I forget birthdays", "I never forget birthdays"),
                       ("I forget my medication", "I do not forget my medication"),
                       ("I try to forget birthdays", "I try not to forget birthdays"),
                       ("I forget birthdays", "Remember that I never forget birthdays"),
                       ("I forget birthdays", "Never forget birthdays")]:
        assert trust_for(fact, said) == "assistant", (fact, said)


def test_a_no_or_a_changed_number_is_a_correction_not_a_quote():
    """Codex r4: "I like coffee? No." and "Dec 19? No, Dec 20." are corrections; and a number is content."""
    from aerys_v2.tools.remember import trust_for
    assert trust_for("I like coffee", "I like coffee? No.") == "assistant"
    assert trust_for("My appointment is Dec 19", "My appointment is Dec 19? No, Dec 20.") == "assistant"
    assert trust_for("My appointment is Dec 19", "My appointment is Dec 20") == "assistant"
    assert trust_for("My appointment is Dec 19", "remember my appointment is Dec 19") == "owner"


def test_a_negation_must_match_the_word_it_negates():
    """Codex r5: "No, remember that I like dark roast coffee" must not vouch for "I do not like dark roast coffee"."""
    from aerys_v2.tools.remember import trust_for
    assert trust_for("I do not like dark roast coffee", "No, remember that I like dark roast coffee") == "assistant"
    assert trust_for("I do not like dark roast coffee", "I don't know, but I like dark roast coffee") == "assistant"
    assert trust_for("I don't like dark roast coffee", "remember I do not like dark roast coffee") == "owner"
    assert trust_for("I cannot eat shellfish", "remember I cannot eat shellfish") == "owner"


def test_a_quoted_negator_still_counts():
    """Codex r6: "I do 'not' like dark roast coffee" lost its negation to the quote marks."""
    from aerys_v2.tools.remember import trust_for
    for said in ["I do 'not' like dark roast coffee", "I 'don't' like dark roast coffee",
                 "I do \u2018not\u2019 like dark roast coffee", 'I do "not" like dark roast coffee']:
        assert trust_for("I like dark roast coffee", said) == "assistant", said
    assert trust_for("I don't like dark roast coffee", "remember I 'don't' like dark roast coffee") == "owner"


def test_quote_marks_split_words_and_never_hide_a_negation():
    """Codex r7: "I do not\u2018like\u2019 dark roast coffee" became one word, "not'like", and lost its negation."""
    from aerys_v2.tools.remember import _negations, trust_for
    for said in ["I do not\u2018like\u2019 dark roast coffee", "I do not'like' dark roast coffee",
                 "I do not\u201clike\u201d dark roast coffee", "I don\u2019t like dark roast coffee",
                 "I won't like dark roast coffee", "I do 'not' like dark roast coffee"]:
        assert trust_for("I like dark roast coffee", said) == "assistant", said
    assert _negations("we're fine and it's good") == set()
    assert _negations("I can't swim") == {("not", "swim")}


def test_every_apostrophe_mark_in_a_contraction_still_negates():
    """Codex r8: a left-curly apostrophe split "don\u2018t" into "don", "t" and the negation vanished."""
    from aerys_v2.tools.remember import _negations, trust_for
    for said in ["I don\u2018t like dark roast coffee", "I don\u2019t like dark roast coffee", "I don`t like dark roast coffee",
                 "I don\u00b4t like dark roast coffee", "I don\u02bct like dark roast coffee", "I do not\u2018like\u2019 dark roast coffee"]:
        assert trust_for("I like dark roast coffee", said) == "assistant", said
    assert _negations("can\u2018t swim") == {("not", "swim")} and _negations("won\u2018t eat peanuts") == {("not", "eat")}


def test_typography_never_hides_a_negation_or_fakes_a_match():
    """Codex r6-r8 class, closed at once: full-width forms, invisible characters, odd apostrophes."""
    from aerys_v2.tools.remember import trust_for
    for said in ["I do n\u200bot like dark roast coffee", "I do \uff4e\uff4f\uff54 like dark roast coffee",
                 "I don\uff07t like dark roast coffee", "I don\u2032t like dark roast coffee"]:
        assert trust_for("I like dark roast coffee", said) == "assistant", repr(said)
    assert trust_for("I like dark roast coffee", "remember I like dark\u200b roast coffee") == "owner"
    # Codex r12: full-width letters in HIS message never matched under the live rule, which is now a gate.
    assert trust_for("I like dark roast coffee", "remember I like \uff44\uff41\uff52\uff4b roast coffee") == "assistant"


def test_every_content_word_must_be_his_so_no_trick_can_flip_the_meaning():
    """Codex r9's sweep: unmapped apostrophes, combining accents, homoglyphs, split words, invisible characters
    and quoted suffixes all hid a flipped meaning inside the old 20% allowance. Now every content word of the fact
    must be in his words; anything unexplained means it's hers."""
    from aerys_v2.tools.remember import trust_for
    said = "remember I like dark roast coffee"
    flipped = ["I donʻt like dark roast coffee", "I donʹt like dark roast coffee", "I don“t like dark roast coffee",
               "I ńot like dark roast coffee", "I nót like dark roast coffee",
               "I nоt like dark roast coffee", "I nοt like dark roast coffee", "I \U0001F15D\U0001F15E\U0001F163 like dark roast coffee",
               "I n-ot like dark roast coffee", "I n ot like dark roast coffee", "I don t like dark roast coffee",
               "I dont like dark roast coffee", "I nᅟot like dark roast coffee", "I nㅤot like dark roast coffee",
               "I do not​like dark roast coffee"]
    for fact in flipped:
        assert trust_for(fact, said) == "assistant", repr(fact)
    assert trust_for("I prefer non-smoking hotel rooms near central station",
                     "remember I prefer smoking hotel rooms near central station") == "assistant"
    assert trust_for("I take vitamins without‘d’ every single morning",
                     "remember I take vitamins with D every single morning") == "assistant"


def test_his_own_typing_and_name_still_count_as_his_words():
    """Chris types "dont"; she writes "Chris" for his "I". Neither should cost him his words, and neither can flip one."""
    from aerys_v2.tools.remember import trust_for
    assert trust_for("I don't like dark roast coffee", "remember i dont like dark roast coffee") == "owner"
    assert trust_for("I like dark roast coffee", "i dont like dark roast coffee") == "assistant"
    # Codex r12: the live rule is a gate, and it never matched a name for "my": this stays hers until #261's judged
    # check can vouch for a paraphrase.
    assert trust_for("Dana is Chris's wife", "for the record, Dana is my wife", names=("Chris",)) == "assistant"
    assert trust_for("Dana is Chris's wife", "for the record, Dana is my wife") == "assistant"
    assert trust_for("Dana is Sam's wife", "for the record, Dana is my wife", names=("Chris",)) == "assistant"


def test_numbers_symbols_and_every_word_count():
    """Codex r10: signs, comparison symbols, fractions, standalone letters and filler words all carry meaning."""
    from aerys_v2.tools.remember import trust_for
    for fact, said in [("The temperature is −20 degrees", "The temperature is 20 degrees"),
                       ("The temperature is -20 degrees", "The temperature is 20 degrees"),
                       ("The value is >5", "The value is <5"),
                       ("I ¬like coffee", "I like coffee"),
                       ("I take 1/5 tablet", "I take 1.5 tablet"),
                       ("I take vitamin D daily", "I take vitamin A daily"),
                       ("I take vitamin D daily", "I take vitamin daily"),
                       ("I work in IT", "I work"),
                       ("I like tea and coffee", "I like tea or coffee"),
                       ("My appointment is at 8:30", "remember my appointment is at 8:45")]:
        assert trust_for(fact, said) == "assistant", (fact, said)
    assert trust_for("A friend of mine lives in Ohio", "remember a friend of mine lives in Ohio") == "owner"
    assert trust_for("My appointment is at 8:30", "remember my appointment is at 8:30") == "owner"
    assert trust_for("I take vitamin D daily", "remember I take vitamin D daily") == "owner"


def test_marks_and_numbers_must_survive_both_ways_and_roman_numerals_count():
    """Codex r11: a fact may not drop a mark ("¬like" -> "like", ">5" -> "5"), lose a leading decimal or a spaced
    minus, flip a blood type's sign, or read a Roman "I" as the pronoun."""
    from aerys_v2.tools.remember import trust_for
    for said, fact in [("I ¬like coffee", "I like coffee"),
                       ("The value is >5", "The value is 5"),
                       ("I take .5 mg", "I take 5 mg"), ("I take 5 mg", "I take .5 mg"),
                       ("The temperature is 20 degrees", "The temperature is - 20 degrees"),
                       ("The temperature is - 20 degrees", "The temperature is 20 degrees"),
                       ("My blood type is O+", "My blood type is O-"),
                       ("I have type II diabetes", "I have type I diabetes"),
                       ("My cancer is stage II", "My cancer is stage I")]:
        assert trust_for(fact, said) == "assistant", (said, fact)
    assert trust_for("My blood type is O+", "remember my blood type is O+") == "owner"
    assert trust_for("I take .5 mg", "remember I take .5 mg") == "owner"
    assert trust_for("I have type II diabetes", "remember I have type II diabetes") == "owner"
    assert trust_for("I prefer non-smoking rooms", "remember I prefer non-smoking rooms") == "owner"


#: Codex r13: an explicit table (no source scan) of every pair these reviews raised, positives included.
INCLUSION_CASES = [
    ("I like coffee.", "I do not like coffee."), ("I do not like coffee", "I do not like coffee"),
    ("I like coffee", "remember I like coffee"), ("my tattoo appointment is Dec 19", "don't forget my tattoo appointment is Dec 19"),
    ("I forget birthdays", "I never forget birthdays"), ("I forget birthdays", "Never forget birthdays"),
    ("I like coffee", "I like coffee? No."), ("My appointment is Dec 19", "My appointment is Dec 19? No, Dec 20."),
    ("I do not like dark roast coffee", "No, remember that I like dark roast coffee"),
    ("I like dark roast coffee", "I do \u2018not\u2019 like dark roast coffee"), ("I like dark roast coffee", "I do not\u2018like\u2019 dark roast coffee"),
    ("I like dark roast coffee", "I don\u2018t like dark roast coffee"), ("I don\u02bbt like dark roast coffee", "remember I like dark roast coffee"),
    ("I n\u043et like dark roast coffee", "remember I like dark roast coffee"), ("I \U0001F15D\U0001F15E\U0001F163 like dark roast coffee", "remember I like dark roast coffee"),
    ("I n ot like dark roast coffee", "remember I like dark roast coffee"), ("I dont like dark roast coffee", "remember I like dark roast coffee"),
    ("I do not\u200blike dark roast coffee", "remember I like dark roast coffee"),
    ("I prefer non-smoking hotel rooms near central station", "remember I prefer smoking hotel rooms near central station"),
    ("The temperature is \u221220 degrees", "The temperature is 20 degrees"), ("The value is >5", "The value is <5"),
    ("I like coffee", "I \u00aclike coffee"), ("The value is 5", "The value is >5"), ("I take 5 mg", "I take .5 mg"),
    ("My blood type is O-", "My blood type is O+"), ("I have type I diabetes", "I have type II diabetes"),
    ("I take vitamin D daily", "I take vitamin A daily"), ("I work in IT", "I work"), ("I like tea and coffee", "I like tea or coffee"),
    ("The cabin is mine", "The cabin is yours"), ("I like coffee", "I like \uff43\uff4f\uff46\uff46\uff45\uff45"),
    ("Dana is Chris's wife", "for the record, Dana is my wife"), ("I don't like dark roast coffee", "remember i dont like dark roast coffee"),
    ("My blood type is O+", "remember my blood type is O+"), ("I take .5 mg", "remember I take .5 mg"),
    ("A friend of mine lives in Ohio", "remember a friend of mine lives in Ohio"),
    ("the pool guy comes on Tuesdays", "hey, remember that the pool guy comes on Tuesdays"),
]


def test_never_their_words_where_the_live_rule_said_otherwise():
    """Codex r12/r13: the branch is at least as safe as live for every input (the live rule is the last gate); the
    explicit table above holds every case the reviews raised, and each one obeys new-owner => live-owner."""
    from aerys_v2.tools.remember import _live_rule, trust_for
    assert trust_for("The cabin is mine", "The cabin is yours") == "assistant"
    owners = 0
    for fact, said in INCLUSION_CASES:
        if trust_for(fact, said, names=("Chris",)) == "owner":
            owners += 1
            assert _live_rule(fact, said) == "owner", (fact, said)
    assert owners >= 5, "the table must hold positives too"


def test_her_search_guidance_says_to_look_before_advice_about_their_things():
    """Preferences (11/25): "any tips?" about his phone didn't read as a memory question, so she never looked up the
    power bank she had saved; and where she knew a taste she padded with options that clash with it."""
    from aerys_v2.factory import search_recall_overlay
    text = search_recall_overlay(memory=True, history=True)
    assert "Before advice or a recommendation about their own things" in text
    assert "leave out options that clash with it" in text
    assert search_recall_overlay(memory=False, history=False) == ""
