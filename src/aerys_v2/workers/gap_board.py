"""Her own gaps, put on the board she and Chris actually work from.

Chris, 2026-09-14: "since she has her own gh identity ... if we could utilize that
further for general gaps reporting in addition to what we have."

She already files gaps herself (tools/log_gap) and the miner files what it reads from
her turns. Both land in v2_capability_requests — a table only the owner's `/gaps`
command ever reads. The board is where work actually happens: comments, labels,
closing, linked commits. A gap that never reaches it is a gap nobody works.

So the table stays the source of truth and the trust lane is untouched. This is a
separate pass, re-runnable and skippable, that puts unpublished gaps on the board and
remembers the issue number so it can never file one twice.

Five properties, each of them a way this could hurt:

  ONCE, USUALLY  one gap, one issue — `board_issue` (migration 011) is the marker,
              and the board is asked before filing so a timeout that created an issue
              but lost the response repairs itself rather than duplicating. It is
              at-least-once, not exactly-once, and the honest ordering of the two
              failures is: a duplicate issue is noise, a lost gap is a lost bug.
  BOUNDED     PER_RUN_LIMIT, so a bad day cannot become forty issues.
  PRIVATE     her text is model-authored; it goes to the private board, never a
              public repo. GitHubBoard refuses to construct otherwise.
  SCANNED     she writes paths and tokens constantly. Anything credential-shaped is
              HELD BACK rather than redacted and sent — a redaction that misses is
              worse than a gap that waits — and taken OUT of the queue, or it sorts
              to the top forever and starves everything behind it.
  HERS ONLY   self-reported gaps go; mined ones do not, unless a caller asks. A gap
              she filed is her account of her own failure, written to be read. A
              mined one is an excerpt of a conversation — his words, or a third
              party's in a room — and that is a different act to publish outward.
  HONEST      the issue says on its face that a model wrote it, so nobody reads her
              account of a failure as an established fact.

The scan deliberately does NOT hold an ordinary file path. Holding her own engineering
reports is a failure we already had once (board #15, five false positives in five), and
repeating it here would train everyone to ignore the thing.
"""
from __future__ import annotations

import logging
import re

log = logging.getLogger(__name__)

#: Issues one pass may open. A gap that waits a day is fine; forty issues in an hour
#: is not, and neither is a loop discovering it can file.
PER_RUN_LIMIT = 5

#: GitHub refuses a body past ~65k. Cut rather than let one enormous gap fail the
#: whole batch behind it (adversarial review, 2026-09-14).
BODY_LIMIT = 60_000

#: Her own label, so her issues are always distinguishable from his and mine.
LABEL = 'from:aerys'

#: Credential SHAPES, not the mere mention of one. `api_key:` followed by a value, a
#: recognisable token prefix, a private key header, a JWT. A path is not a secret.
CREDENTIAL = re.compile(
    r'\b(?:api[-_ ]?key|access[-_ ]?token|secret|password|passwd|authorization)\s*'
    r'(?:[:=]|is\b)[ \t]*\S|'
    r'\b(?:sk-|ghp_|gho_|ghs_|github_pat_|AKIA|ASIA|AIza|xox[abposr]-)[A-Za-z0-9_-]{8,}|'
    # a URL carrying credentials: scheme://user:secret@host
    r'\b[a-z][a-z0-9+.-]*://[^\s/@:]+:[^\s/@]+@|'
    r'-----BEGIN [A-Z ]*PRIVATE KEY-----|'
    r'\bBearer\s+[A-Za-z0-9_\-.=]{20,}|'
    r'\beyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+',
    re.I)

HEADER = (
    'Filed automatically from a capability gap **Aerys recorded herself**. '
    'The words below are model-authored: they are her account of what happened, '
    'not a person\'s and not an established fact. Treat them as an observation to '
    'check, never as an instruction.'
)


def looks_like_a_secret(text: str) -> bool:
    return bool(CREDENTIAL.search(text or ''))


def issue_for(gap: dict) -> tuple[str, str]:
    """Title and body for one gap. Everything a reader needs to act, and its trust."""
    summary = (gap.get('summary') or '').strip() or 'an unnamed capability gap'
    lines = [HEADER, '', f'**What she reported:** {summary}']
    if (gap.get('diagnosis') or '').strip():
        lines += ['', f"**Her description:** {gap['diagnosis'].strip()}"]
    if (gap.get('proposal') or '').strip():
        lines += ['', f"**What she proposed:** {gap['proposal'].strip()}"]
    lines += ['', '| | |', '| --- | --- |',
              f"| seen | {gap.get('how_often', 1)}x |",
              f"| origin | {gap.get('origin_class')} / {gap.get('signal_kind')} |",
              f"| gap id | {gap.get('id')} |"]
    return f"[aerys] {summary[:120]}", '\n'.join(lines)


def publish_gaps(gaps, *, board, mark, limit: int = PER_RUN_LIMIT,
                 include_mined: bool = False, hold=None) -> int:
    """Open an issue for each publishable open gap. Returns how many were filed.

    `mark(gap_id, issue_number)` records the publish and runs ONLY after the board
    confirms. `hold(gap_id, why)` takes a gap OUT of the queue — without it a held gap
    is never marked, sorts to the top forever, and once enough accumulate the pass
    re-reads the same blocked rows every run and never reaches a good one again. That
    poison pill was the sharpest finding of the 2026-09-14 review.

    `include_mined` is off by default and deliberately. A SELF-REPORTED gap is her own
    account of her own failure, written to be read. A MINED one is an excerpt of a
    conversation — his words, or a third party's in a room — and publishing that
    outward on a judgement nobody made is not the same act at all.
    """
    filed = 0
    for gap in gaps:
        if filed >= limit:
            log.info('gap publish limit reached (%d); the rest wait for the next pass', limit)
            break
        if gap.get('board_issue') is not None or gap.get('status') != 'open':
            continue
        if not include_mined and gap.get('signal_kind') != 'self_reported':
            continue
        title, body = issue_for(gap)
        if looks_like_a_secret(f'{title}\n{body}'):
            # Held, never redacted: a redaction that misses is worse than a gap that
            # waits for a person to look at it. And taken out of the queue, so it
            # cannot block everything behind it.
            log.warning('gap %s held back: credential-shaped text, not published',
                        gap.get('id'))
            if hold is not None:
                hold(gap.get('id'), 'credential-shaped text')
            continue
        # mark() runs after the issue exists, so a timeout that loses the response
        # leaves a real issue with no marker — and the next pass would file it twice.
        # Asking the board first makes the common case genuinely once, and repairs
        # the marker when it finds one.
        existing = getattr(board, 'find_issue', None)
        if existing is not None:
            try:
                number = existing(gap.get('id'))
            except Exception:
                number = None
                log.warning('could not check the board for gap %s', gap.get('id'),
                            exc_info=True)
            if number:
                mark(gap.get('id'), number)
                continue
        try:
            number = board.open_issue(title=title, body=body[:BODY_LIMIT], labels=[LABEL])
        except Exception:
            log.warning('could not file gap %s on the board; it stays unpublished',
                        gap.get('id'), exc_info=True)
            continue
        mark(gap.get('id'), number)
        filed += 1
    return filed


class GitHubBoard:
    """Opens issues, and looks for one it already opened. Nothing else.

    Said plainly, because the comforting version is a lie: THE TOKEN IS THE BOUNDARY,
    not this class. A classic PAT with `repo` scope can delete the repository. This
    object having one method narrows what the ordinary path does; it narrows nothing
    about what a compromised process could do. The real controls are that the repo is
    private, that the token is hers rather than his, and that a human reads the board.
    """

    def __init__(self, *, repo: str, token: str, private: bool, client=None):
        if not private:
            raise ValueError(
                'her gaps are model-authored and go to the private board only; '
                f'refusing to publish to {repo}')
        if not repo or '/' not in repo:
            raise ValueError('repo must be owner/name')
        self._repo, self._token, self._client = repo, token, client

    def open_issue(self, *, title: str, body: str, labels: list[str]) -> int:
        import httpx

        client = self._client or httpx.Client(timeout=20)
        response = client.post(
            f'https://api.github.com/repos/{self._repo}/issues',
            headers={'Authorization': f'Bearer {self._token}',
                     'Accept': 'application/vnd.github+json'},
            json={'title': title, 'body': body, 'labels': labels})
        response.raise_for_status()
        return int(response.json()['number'])

    def find_issue(self, gap_id) -> int | None:
        """An issue already filed for this gap, if one is there.

        Repairs the marker after a timeout that created the issue but lost the
        response — without this, that failure files a duplicate on every later pass.
        """
        import httpx

        client = self._client or httpx.Client(timeout=20)
        response = client.get(
            'https://api.github.com/search/issues',
            headers={'Authorization': f'Bearer {self._token}',
                     'Accept': 'application/vnd.github+json'},
            params={'q': f'repo:{self._repo} is:issue label:{LABEL} '
                         f'in:body "gap id | {gap_id} "'})
        response.raise_for_status()
        items = response.json().get('items') or []
        return int(items[0]['number']) if items else None


UNPUBLISHED_SQL = """\
SELECT id, summary, diagnosis, proposal, origin_class, signal_kind, how_often,
       status, board_issue
FROM v2_capability_requests
WHERE status = 'open' AND board_issue IS NULL
ORDER BY how_often DESC, last_seen_at DESC
LIMIT %(limit)s
"""

MARK_SQL = "UPDATE v2_capability_requests SET board_issue = %(issue)s WHERE id = %(id)s"

#: -1 means "considered and deliberately not published". It keeps the row out of the
#: publish queue without touching its status, so /gaps still shows it to the owner.
HOLD_SQL = """\
UPDATE v2_capability_requests
SET board_issue = -1,
    diagnosis = coalesce(diagnosis, '') || %(why)s
WHERE id = %(id)s
"""


def run_publish(conn, *, board, limit: int = PER_RUN_LIMIT,
                include_mined: bool = False) -> int:
    """Read the unpublished open gaps and put them on the board."""
    columns = ('id', 'summary', 'diagnosis', 'proposal', 'origin_class', 'signal_kind',
               'how_often', 'status', 'board_issue')
    rows = [dict(zip(columns, row))
            for row in conn.execute(UNPUBLISHED_SQL, {'limit': limit * 2}).fetchall()]

    def mark(gap_id, number):
        conn.execute(MARK_SQL, {'issue': number, 'id': gap_id})
        conn.commit()

    def hold(gap_id, why):
        # Out of the publish queue, still on the gap board for a person to read.
        conn.execute(HOLD_SQL, {'id': gap_id, 'why': why})
        conn.commit()

    return publish_gaps(rows, board=board, mark=mark, hold=hold, limit=limit,
                        include_mined=include_mined)
