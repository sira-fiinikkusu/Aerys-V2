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


def test_the_module_has_no_upgrade_path():
    """Gemini review: only the veto is wired; a paraphrase upgrade (making something theirs) is the owner's call."""
    assert not hasattr(tw, "combine")


def test_the_remember_tool_uses_it_as_a_veto_only():
    """#261 wired VETO ONLY: same words with a changed meaning go back to hers; nothing is ever made theirs."""
    from aerys_v2.tools.remember import CURRENT_TURN_TEXT, build_remember_tool
    owner = "6e6bcbed-03ef-4d17-95d2-89c467414335"
    config = {"configurable": {"identity": {"user_id": owner, "platform": "discord", "display_name": "Sam"}}}

    def kept(fact, said, judged):
        records = []
        calls = []
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

    veto, theirs = tw.Judgement(0.1, 0.9, 0.9), tw.Judgement(0.99, 0.05, 0.05)
    trust, calls = kept("Alice trusts Bob", "remember Bob trusts Alice", veto)
    assert trust == "assistant" and calls == [("Alice trusts Bob", "remember Bob trusts Alice", "Sam")]
    assert kept("Bob trusts Alice", "remember Bob trusts Alice", theirs)[0] == "owner"
    assert kept("Bob trusts Alice", "remember Bob trusts Alice", None)[0] == "owner"        # Jev unavailable
    trust, calls = kept("Sam gets up at 6", "I'm up at 6", theirs)                          # never an upgrade
    assert trust == "assistant" and calls == []
