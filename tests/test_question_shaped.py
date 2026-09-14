"""English opens declarative sentences with interrogative words constantly.

Found 2026-09-13 while restoring a memory my truncation bug had mangled. Four of the
five facts wrote; the fifth was refused:

    What makes Aerys's shared memory safe is a surface she cannot quite see, plus a
    "dreaming" process that helps make sure she does not get poisoned by bad memories.
       -> memory skipped: question-shaped observation

That is a statement. It was rejected for starting with the word "What".

The guard protects something real: the extractor mines turns for facts, and a question
stored as a fact poisons recall with something nobody ever asserted. But the rule was
`^(what|which|who|where|when|why|how|do|does|...)` plus "ends with ?", and the first
half fires on every wh-cleft — "what makes it safe is", "where the checkpointer lives
matters", "why she refused is in the log".

The signal that actually separates them is punctuation, not the opening word: a
sentence that opens with an interrogative and ends in a full stop is a cleft. A real
question either carries its question mark or, typed in a hurry, carries no terminal
punctuation at all.

Yes/no questions are different and keep the old treatment: they begin with the
auxiliary itself ("do you know my birthday"), which no declarative does.
"""
from aerys_v2.workers.extraction import question_shaped

DECLARATIVES = [
    'What makes Aerys\'s shared memory safe is a surface she cannot quite see.',
    'Where the checkpointer lives matters for gap reporting.',
    'Why she refused is in the log.',
    'How he wants approvals handled changed on 2026-09-13.',
    'What he asked for was one fact per call.',
    'Who owns home control is settled: the home body does.',
    'When the stick syncs is every ten minutes.',
    'Which body answered is recorded on the turn.',
]

QUESTIONS = [
    'What is my birthday?',
    'what do you know about me',
    'how do i get to Tampa',
    'where are my keys',
    'who is coming on Thursday',
    'What time is it',
    'do you know my birthday',
    'is the pool guy coming Tuesday',
    # (a yes/no question opening with 'was'/'has' is NOT caught, and deliberately:
    # memory values are declarative fragments like "Has two dogs" — see _YES_NO)
    'can you check the weather?',
    'Do you remember what I told you last night?',
]


def test_a_statement_that_opens_with_an_interrogative_is_a_statement():
    missed = [t for t in DECLARATIVES if question_shaped(t)]
    assert not missed, missed


def test_a_question_is_still_a_question():
    missed = [t for t in QUESTIONS if not question_shaped(t)]
    assert not missed, missed


def test_the_portable_observation_prefix_is_still_stripped_first():
    assert question_shaped('[Portable observation time: 2026-09-12T00:30:00Z] what is my birthday?')
    assert not question_shaped(
        '[Portable observation time: 2026-09-12T00:30:00Z] What makes it safe is the filter.')


def test_empty_and_none_are_not_questions():
    assert not question_shaped(None) and not question_shaped('') and not question_shaped('   ')
