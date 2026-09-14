"""Her own gaps reach the board she and Chris actually work from.

Chris, 2026-09-14: "since she has her own gh identity ... if we could utilize that
further for general gaps reporting in addition to what we have."

She already files gaps (log_gap) and the miner files what it reads from her turns.
Both land in v2_capability_requests, which only `/gaps` ever reads. The board is where
the work happens — comments, labels, closing, linked commits — so a gap that never
reaches it is a gap nobody works.

So: the table stays the source of truth and the trust lane is untouched; this is a
separate pass that puts unpublished gaps on the board and remembers the issue number.

Five things it must get right, and every one of them is a way this could hurt:

  - IDEMPOTENT. A gap is filed once. Re-running the pass files nothing new.
  - BOUNDED. A bad day cannot become forty issues.
  - PRIVATE. Her text is model-authored; it goes to the private repo, never a public one.
  - SCANNED. She writes file paths and tokens constantly. Nothing credential-shaped
    leaves this machine.
  - HONEST. The issue says on its face that a person did not write it, so nobody reads
    a model's account of a failure as an established fact.
"""
import pytest

publish = pytest.importorskip('aerys_v2.workers.gap_board',
                              reason='the gap publisher is not built yet')


def gap(**over):
    row = {'id': 1, 'summary': 'browse returned nothing on three tries',
           'diagnosis': 'the tool answered but the page body was empty',
           'origin_class': 'complaint', 'signal_kind': 'self_reported',
           'how_often': 3, 'status': 'open', 'board_issue': None}
    row.update(over)
    return row


class FakeBoard:
    def __init__(self, fail=False):
        self.opened = []
        self.fail = fail

    def open_issue(self, *, title, body, labels):
        if self.fail:
            raise RuntimeError('github down')
        self.opened.append({'title': title, 'body': body, 'labels': labels})
        return 100 + len(self.opened)


def test_an_open_gap_reaches_the_board_and_is_remembered():
    board, marked = FakeBoard(), []
    published = publish.publish_gaps([gap()], board=board, mark=lambda i, n: marked.append((i, n)))
    assert len(board.opened) == 1
    assert marked == [(1, 101)], 'the issue number goes back on the row'
    assert published == 1


def test_a_gap_already_on_the_board_is_never_filed_twice():
    board = FakeBoard()
    assert publish.publish_gaps([gap(board_issue=57)], board=board, mark=lambda *_: None) == 0
    assert board.opened == []


def test_only_open_gaps_go():
    board = FakeBoard()
    for status in ('built', 'wont_fix', 'rejected'):
        publish.publish_gaps([gap(status=status)], board=board, mark=lambda *_: None)
    assert board.opened == []


def test_a_bad_day_cannot_become_forty_issues():
    board = FakeBoard()
    many = [gap(id=n) for n in range(40)]
    published = publish.publish_gaps(many, board=board, mark=lambda *_: None,
                                     limit=publish.PER_RUN_LIMIT)
    assert published == publish.PER_RUN_LIMIT <= 5
    assert len(board.opened) == publish.PER_RUN_LIMIT


def test_nothing_credential_shaped_leaves_the_machine():
    """She writes paths and tokens constantly. This text is hers, and it is going to
    a remote service."""
    board = FakeBoard()
    leaky = gap(diagnosis='it failed with api_key: sk-proj-AbC123dEf456GhI789jKlMnO')
    published = publish.publish_gaps([leaky], board=board, mark=lambda *_: None)
    assert published == 0 and board.opened == [], 'held back, not redacted and sent'


def test_a_path_in_her_diagnosis_is_not_a_credential():
    """The detector that holds her own build reports is the failure we already had
    once; this must not repeat it."""
    board = FakeBoard()
    ordinary = gap(diagnosis='write_file was denied for /home/aerys/work/weather/run.sh')
    assert publish.publish_gaps([ordinary], board=board, mark=lambda *_: None) == 1


def test_the_issue_says_a_person_did_not_write_it():
    board = FakeBoard()
    publish.publish_gaps([gap()], board=board, mark=lambda *_: None)
    body = board.opened[0]['body'].lower()
    assert 'aerys' in body and ('model-authored' in body or 'she wrote' in body
                               or 'not a person' in body)
    assert 'from:aerys' in board.opened[0]['labels']


def test_the_issue_carries_what_a_reader_needs_to_act():
    board = FakeBoard()
    publish.publish_gaps([gap(how_often=3)], board=board, mark=lambda *_: None)
    issue = board.opened[0]
    assert 'browse returned nothing' in issue['title']
    assert 'the page body was empty' in issue['body']
    assert '3' in issue['body'], 'how often it has happened is the triage signal'


def test_a_board_that_is_down_marks_nothing():
    board, marked = FakeBoard(fail=True), []
    assert publish.publish_gaps([gap()], board=board, mark=lambda i, n: marked.append(i)) == 0
    assert marked == [], 'never record a publish that did not happen'


def test_the_repo_must_be_private():
    with pytest.raises(ValueError):
        publish.GitHubBoard(repo='owner/public-repo', token='x', private=False)


def test_the_worker_refuses_without_her_identity_configured():
    """Unset is not broken: the publisher simply does not run, and nothing changes."""
    from aerys_v2.config import Settings

    fields = Settings.model_fields
    assert 'board_repo' in fields and fields['board_repo'].default is None
    assert 'board_token' in fields and fields['board_token'].default is None


def test_the_subcommand_exists_and_is_wired():
    import inspect

    from aerys_v2.workers import __main__ as workers_main

    source = inspect.getsource(workers_main)
    assert '"gap-board"' in source and '_gap_board_main' in source


# ── adversarial review, 2026-09-14: this writes OUTWARD, under her name ─────

def test_only_gaps_she_filed_herself_go_automatically():
    """Her self-reported gaps are her own account of her own failure, written to be
    read. A MINED gap is an excerpt of a conversation — his words, or a third party's
    in a room — and that is not hers to publish on a judgement nobody made. Mined ones
    need the caller to say so explicitly."""
    board = FakeBoard()
    mined = gap(signal_kind='reply_phrase')
    assert publish.publish_gaps([mined], board=board, mark=lambda *_: None) == 0
    assert publish.publish_gaps([mined], board=board, mark=lambda *_: None,
                                include_mined=True) == 1


def test_a_held_gap_does_not_block_the_queue_behind_it():
    """The poison pill: a held gap is never marked, sorts to the top forever, and
    once enough of them accumulate the pass re-reads the same blocked rows every run
    and never reaches a good one again."""
    board, held = FakeBoard(), []
    rows = [gap(id=1, diagnosis='api_key: sk-proj-AbC123dEf456GhI789jKl'),
            gap(id=2, summary='browse returns nothing')]
    filed = publish.publish_gaps(rows, board=board, mark=lambda *_: None,
                                 hold=lambda gap_id, why: held.append((gap_id, why)))
    assert filed == 1, 'the good one still goes'
    assert held and held[0][0] == 1, 'the poisoned one is taken OUT of the queue'


def test_it_will_not_file_a_gap_that_is_already_on_the_board_under_another_number():
    """mark() runs after the issue is created, so a timeout that loses the response
    leaves a filed issue with no marker and the next pass files it twice. Asking the
    board first makes the common case genuinely once."""
    class Existing(FakeBoard):
        def find_issue(self, gap_id):
            return 77 if gap_id == 1 else None

    board, marked = Existing(), []
    assert publish.publish_gaps([gap(id=1)], board=board,
                                mark=lambda i, n: marked.append((i, n))) == 0
    assert board.opened == [], 'not filed again'
    assert marked == [(1, 77)], 'and the marker is repaired'


def test_a_body_too_long_for_the_board_is_cut_not_rejected():
    board = FakeBoard()
    publish.publish_gaps([gap(diagnosis='x' * 80_000)], board=board, mark=lambda *_: None)
    assert len(board.opened[0]['body']) <= publish.BODY_LIMIT


@pytest.mark.parametrize('text', [
    'the key is AKIAIOSFODNN7EXAMPLE',
    'postgres://aerys:hunter2swordfish@nas:5432/aerys',
    '-----BEGIN OPENSSH PRIVATE KEY-----',
    'it rejected AIzaSyA1B2C3D4E5F6G7H8I9J0K1L2M3N4O5P6Q',
    'Authorization: Bearer abcdefghijklmnopqrstuvwxyz012345',
])
def test_the_scanner_catches_the_shapes_it_missed(text):
    assert publish.looks_like_a_secret(text), text


@pytest.mark.parametrize('text', [
    'write_file was denied for /home/aerys/work/weather_news/run_report.sh',
    'browse returned nothing for https://docs.langchain.com/oss/python/overview',
    'the cron line 0 9-21 * * * did not install',
])
def test_the_scanner_still_does_not_hold_her_ordinary_work(text):
    assert not publish.looks_like_a_secret(text), text
