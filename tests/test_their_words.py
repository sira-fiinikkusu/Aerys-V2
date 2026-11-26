"""#261: the judged "does this fact say what they said?" check (Jev), beside remember's strict word check.

Calibrated 11/25 on live Jev (jev-1.14.0), with the speaker named in the state: 12/12 on the tuning set and 13/14
held out (the miss was "unsure" on a true paraphrase, which leaves the strict word check standing). Every changed
meaning was vetoed: swapped roles, "might" made certain, an added detail, "or" made "and", a flipped negation, a
corrected date, a Roman numeral, a homoglyph "nоt", "usually" made "always", "sold" made "bought".
"""
import time
from types import SimpleNamespace

from aerys_v2 import their_words as tw


class Client:
    def __init__(self, scores=None, *, delay=0.0, boom=False):
        self.scores, self.delay, self.boom, self.calls = scores or {}, delay, boom, []

    def system_one(self, *, state, questions, model):
        self.calls.append((state, questions, model))
        if self.delay:
            time.sleep(self.delay)
        if self.boom:
            raise RuntimeError("down")
        return SimpleNamespace(answers={q: SimpleNamespace(noul=self.scores.get(q, 0.5)) for q in questions})


def test_the_state_names_the_speaker_and_carries_only_the_two_texts():
    client = Client({"restates": 0.99, "adds": 0.05, "changes": 0.05})
    j = tw.TheirWordsCheck(client)("Sam gets up at 6", "I'm up at 6", "Sam")
    state, questions, model = client.calls[0]
    assert state == {"speaker": "Sam", "their_message": "I'm up at 6", "fact": "Sam gets up at 6"}
    assert set(questions) == {"restates", "adds", "changes"} and model == tw.MODEL
    assert j.theirs and not j.veto


def test_code_decides_theirs_veto_or_unsure():
    assert tw.Judgement(0.9, 0.1, 0.1).theirs
    assert not tw.Judgement(0.9, 0.1, 0.27).theirs and not tw.Judgement(0.9, 0.1, 0.27).veto   # unsure
    assert tw.Judgement(0.9, 0.6, 0.1).veto and tw.Judgement(0.1, 0.1, 0.9).veto


def test_any_failure_is_none_and_the_word_check_stands():
    assert tw.TheirWordsCheck(Client(boom=True))("fact", "message", "Sam") is None
    assert tw.TheirWordsCheck(Client({"restates": 0.9}, delay=0.3), timeout_s=0.05)("fact", "message") is None
    bad = Client({"restates": float("nan"), "adds": 0.1, "changes": 0.1})
    assert tw.TheirWordsCheck(bad)("fact", "message") is None
    assert tw.TheirWordsCheck(Client())("", "message") is None
    assert tw.their_words_for(SimpleNamespace(typesafe_api_key=None)) is None


def test_a_missing_sdk_means_no_veto_never_a_failed_build(monkeypatch):
    import builtins
    real_import = builtins.__import__
    def fake_import(name, *a, **k):
        if name.startswith("typesafe_sdk"):
            raise ImportError("not installed")
        return real_import(name, *a, **k)
    monkeypatch.setattr(builtins, "__import__", fake_import)
    key = SimpleNamespace(get_secret_value=lambda: "k")
    assert tw.their_words_for(SimpleNamespace(typesafe_api_key=key)) is None


def test_the_decision_lives_in_one_place():
    """The veto/upgrade decision is remember._judged; the module only measures (no second decision path)."""
    assert not hasattr(tw, "combine")


def test_the_remember_tool_vetoes_a_changed_meaning_and_upgrades_a_clear_paraphrase():
    """#261 + D107: same words with a changed meaning go back to hers; a clear restatement becomes theirs; unsure,
    unavailable and odd characters leave the word check alone."""
    from aerys_v2.tools.remember import CURRENT_TURN_TEXT, build_remember_tool
    owner = "6e6bcbed-03ef-4d17-95d2-89c467414335"
    config = {"configurable": {"identity": {"user_id": owner, "platform": "discord", "display_name": "Sam"}}}

    def kept(fact, said, judged):
        records, calls = [], []
        def check(f, s, speaker):
            calls.append((f, s, speaker))
            return judged
        tool = build_remember_tool(lambda r: records.append(r) or "insert", their_words=check)
        token = CURRENT_TURN_TEXT.set(said)
        try:
            tool.invoke({"fact": fact}, config=config)
        finally:
            CURRENT_TURN_TEXT.reset(token)
        return records[0]["trust"], calls

    veto, theirs, unsure = tw.Judgement(0.1, 0.9, 0.9), tw.Judgement(0.99, 0.05, 0.05), tw.Judgement(0.9, 0.1, 0.3)
    trust, calls = kept("Alice trusts Bob", "remember Bob trusts Alice", veto)
    assert trust == "assistant" and calls == [("Alice trusts Bob", "remember Bob trusts Alice", "Sam")]
    assert kept("Bob trusts Alice", "remember Bob trusts Alice", theirs)[0] == "owner"
    assert kept("Bob trusts Alice", "remember Bob trusts Alice", None)[0] == "owner"        # Jev unavailable
    assert kept("Sam gets up at 6", "I'm up at 6", theirs)[0] == "owner"                    # D107: a paraphrase
    assert kept("Sam gets up at 6", "I'm up at 6", unsure)[0] == "assistant"
    assert kept("Sam gets up at 6", "I'm up at 6", None)[0] == "assistant"
    trust, calls = kept("Sam does n\u043et like coffee", "I like coffee", theirs)              # odd characters: never
    assert trust == "assistant" and calls == []


def test_a_clipped_message_never_upgrades_a_trailing_correction_can_not_be_lost():
    """Codex 11/26 MAJOR: "I like coffee ... (past the clip) Correction: I do not like coffee" -> Jev sees only the
    head. A judgement on clipped text may veto but never makes a fact theirs."""
    head = "I like coffee. " + "We talked about the weekend and the weather for a long while. " * 25
    said = head + "Correction: I do not like coffee."
    assert len(said) > tw.MESSAGE_MAX and not tw.fits("Sam likes coffee", said)
    check = tw.TheirWordsCheck(Client({"restates": 0.99, "adds": 0.01, "changes": 0.01}), timeout_s=2.0)
    j = check("Sam likes coffee", said, "Sam")
    assert j is not None and j.complete is False and j.theirs is False
    from aerys_v2.tools.remember import _judged
    assert _judged("assistant", check, "Sam likes coffee", said, "Sam") == "assistant"
    short = check("Sam likes coffee", "I like coffee", "Sam")
    assert short.complete and short.theirs                                   # whole text: unchanged
    vetoing = tw.TheirWordsCheck(Client({"restates": 0.1, "adds": 0.9, "changes": 0.9}), timeout_s=2.0)
    assert _judged("owner", vetoing, "Sam likes coffee", said, "Sam") == "assistant"   # a veto still counts


def test_hidden_characters_cant_split_a_mixed_alphabet_word():
    """Codex 11/26 MAJOR: a zero-width space or a combining mark between the letters hid Cyrillic \u043e."""
    from aerys_v2.tools.remember import _judged, _mixed_script
    for fact in ("Sam does n\u200b\u043e\u200bt like coffee", "Sam does n\u0301\u043et like coffee",
                 "Sam does n\u043et like coffee"):
        assert _mixed_script(fact)
        assert _judged("assistant", lambda *a: tw.Judgement(0.99, 0.01, 0.01), fact, "I like coffee", "Sam") == "assistant"
    assert not _mixed_script("Sam does not like caf\u00e9 au lait")
    # Codex r2: circled and full-width letters fold to Latin under NFKC around a Cyrillic o; a squared letter has no
    # fold at all and is caught as a symbol.
    for fact in ("Sam does \u24dd\u043e\u24e3 like coffee", "Sam does \uff4e\u043e\uff54 like coffee",
                 "Sam does \U0001f15d\u043e\U0001f163 like coffee"):
        assert _judged("assistant", lambda *a: tw.Judgement(0.99, 0.01, 0.01), fact, "I like coffee", "Sam") == "assistant"
    assert _mixed_script("Sam does \u24dd\u043e\u24e3 like coffee")
    assert _judged("assistant", lambda *a: tw.Judgement(0.99, 0.01, 0.01), "Sam likes coffee", "I like coffee",
                   "Sam") == "owner"                                       # a plain paraphrase still upgrades
    assert not _mixed_script("Сэм любит кофе")                                 # one alphabet per word is fine


def test_the_upgrade_needs_every_character_ascii_or_typed_by_them():
    """Codex r3: whole-word homoglyphs, separators and combining letters passed the denylist guards. The allowlist:
    each character of the fact is printable ASCII or appears in their own message or name."""
    from aerys_v2.tools.remember import _foreign_characters, _judged
    yes = lambda *a: tw.Judgement(0.99, 0.01, 0.01)
    for fact in ("Sam does \u043f\u043e\u0442 like coffee",               # whole word, Cyrillic
                 "Sam does \u03bd\u03bf\u03c4 like coffee",               # whole word, Greek
                 "Sam does \U0001d6ce\U0001d6d0\U0001d6d5 like coffee",   # mathematical letters
                 "Sam does n\u00b7\u043e\u00b7t like coffee", "Sam does n-\u043e-t like coffee",
                 "Sam does n \u043e t like coffee",                           # separators around a Cyrillic o
                 "Sam does n\u2deat like coffee", "Sam does n\u0345t like coffee"):  # combining letters
        assert _foreign_characters(fact, "I like coffee", "Sam")
        assert _judged("assistant", yes, fact, "I like coffee", "Sam") == "assistant", fact
    # what they typed is fine, accents and all; so is their own name
    assert not _foreign_characters("Sam loves caf\u00e9 au lait", "I love caf\u00e9 au lait", "Sam")
    assert _judged("assistant", yes, "Sam loves caf\u00e9 au lait", "I love caf\u00e9 au lait", "Sam") == "owner"
    assert _judged("assistant", yes, "Jos\u00e9 gets up at 6", "I'm up at 6", "Jos\u00e9") == "owner"
    assert _judged("assistant", yes, "Sam gets up at 6", "I'm up at 6", "Sam") == "owner"
    # the allowlist only gates the upgrade: a veto and the word check's own verdict are untouched
    assert _judged("owner", lambda *a: tw.Judgement(0.1, 0.9, 0.9), "Bob trusts Alice", "remember Bob trusts Alice",
                   "Sam") == "assistant"


def test_the_name_exempts_only_itself_and_equivalent_spellings_pass():
    """Codex r4: a Cyrillic name supplied every letter of a look-alike word; and equivalent spellings (cafe + accent vs
    caf\u00e9, CRLF, a curly apostrophe) blocked real paraphrases."""
    from aerys_v2.tools.remember import _foreign_characters, _judged
    yes = lambda *a: tw.Judgement(0.99, 0.01, 0.01)
    name = "\u0421\u0442\u0435\u043f\u0430\u043d \u041f\u043e\u043f\u043e\u0432"
    assert _judged("assistant", yes, name + " does \u043f\u043e\u0442 like coffee", "I like coffee", name) == "assistant"
    assert _judged("assistant", yes, name + " likes coffee", "I like coffee", name) == "owner"   # the name itself is fine
    assert _judged("assistant", yes, "\u041f\u043e\u043f\u043e\u0432 likes coffee", "I like coffee", name) == "assistant"   # Codex r5: only the full name, as the leading words, is exempt
    for fact, said in (("Sam loves caf\u00e9", "I love cafe\u0301"), ("Sam loves cafe\u0301", "I love caf\u00e9"),
                       ("Sam gets\r\nup at 6", "I'm up at 6"), ("Sam doesn\u2019t like coffee", "I don't like coffee"),
                       ("Sam doesn't like coffee", "I don\u2019t like coffee"), ("Sam said \u201cnever again\u201d", 'I said "never again"')):
        assert not _foreign_characters(fact, said, "Sam"), (fact, said)
        assert _judged("assistant", yes, fact, said, "Sam") == "owner", (fact, said)


def test_the_name_is_exempt_once_as_the_subject_and_no_letterlike_sign_folds_in():
    """Codex r5: a one-letter Cyrillic name made 'n-\u041e-t' pass (hyphens made it a whole word); a common-word name
    was exempt everywhere; NFC turned the Kelvin sign into an ASCII K."""
    from aerys_v2.tools.remember import _judged
    yes = lambda *a: tw.Judgement(0.99, 0.01, 0.01)
    o = "\u041e"
    assert _judged("assistant", yes, o + " does n-" + o + "-t like coffee", "I like coffee", o) == "assistant"
    net = "\u041d\u0435\u0442"
    assert _judged("assistant", yes, net + " likes coffee. " + net + " coffee after noon.", "I like coffee", net) == "assistant"
    assert _judged("assistant", yes, net + " likes coffee", "I like coffee", net) == "owner"     # the subject is fine
    assert _judged("assistant", yes, "Sam takes vitamin \u212a", "I take vitamin D", "Sam") == "assistant"   # Kelvin sign
    assert _judged("assistant", yes, "Sam takes vitamin K", "I take vitamin K", "Sam") == "owner"
    assert _judged("assistant", yes, "Jos\u00e9's alarm is at 6", "my alarm is at 6", "Jos\u00e9") == "owner"


def test_a_malformed_judgement_never_blocks_the_write():
    """Codex 11/26 MINOR: reading .veto / .theirs on a bad result raised outside the try."""
    from aerys_v2.tools.remember import _judged

    class Raises:
        @property
        def veto(self):
            raise ValueError("bad")

        theirs = veto

    for bad in (object(), Raises(), SimpleNamespace(veto="yes", theirs="yes")):
        assert _judged("owner", lambda *a, b=bad: b, "Bob trusts Alice", "remember Bob trusts Alice", "Sam") == "owner"
        assert _judged("assistant", lambda *a, b=bad: b, "Sam gets up at 6", "I'm up at 6", "Sam") == "assistant"


def test_a_failing_check_or_an_empty_turn_never_blocks_the_write_and_never_changes_trust():
    """Gemini review: a check that raises keeps the word check; an empty turn never asks Jev."""
    from aerys_v2.tools.remember import _judged
    def boom(*a):
        raise RuntimeError("down")
    called = []
    def spy(*a):
        called.append(a)
        return tw.Judgement(0.99, 0.05, 0.05)
    assert _judged("owner", boom, "Bob trusts Alice", "remember Bob trusts Alice", "Sam") == "owner"
    assert _judged("assistant", boom, "Sam gets up at 6", "I'm up at 6", "Sam") == "assistant"
    assert _judged("assistant", spy, "Sam gets up at 6", "   ", "Sam") == "assistant" and called == []
    assert _judged("assistant", None, "Sam gets up at 6", "I'm up at 6", "Sam") == "assistant"
