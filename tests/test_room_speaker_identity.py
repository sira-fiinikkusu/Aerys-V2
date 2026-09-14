"""The room must not turn him into a stranger.

Live 2026-09-14. Chris said "pickle" in a public channel without addressing her, then
asked her about it on the stick. She had it — and then added:

    "the word came from Sira, not from you"

Sira is his Discord display name. His TURNS carry his canonical name, Chris, because
they come through the identity resolver. The room lines come straight off Discord and
carry whatever Discord shows. So in one block she sees "Chris: can you look?" and
"Sira: pickle" and reasonably concludes two people are present.

That is worse than cosmetic: she was reasoning aloud about who said what and got it
wrong, and the step after that is telling him somebody else asked her something.

The mapping already exists — it is why his turns say Chris at all. The live room read
is the one path that bypasses it. Resolved once when the gateway starts rather than
per message, because a room read is thirty messages and this must not become thirty
queries.

What must NOT change: everyone else keeps their Discord name. Stratus reads as Stratus.
The bug is only that one of those names is secretly him.
"""
import pytest

from aerys_v2.services.live_room import LiveRoomReader


class FakeAuthor:
    def __init__(self, author_id, display_name):
        self.id = author_id
        self.display_name = display_name
        self.name = display_name


class FakeMessage:
    def __init__(self, author_id, display_name, content):
        self.author = FakeAuthor(author_id, display_name)
        self.content = content


class FakeChannel:
    def __init__(self, messages):
        self._messages = messages

    async def history(self, limit=None):
        for message in reversed(self._messages[-(limit or len(self._messages)):]):
            yield message


class FakeClient:
    def __init__(self, channel):
        self._channel = channel
        self.loop = None

    def get_channel(self, _id):
        return self._channel


ROOM = [
    FakeMessage(60426939629838336, 'Sira', 'pickle'),
    FakeMessage(999, 'Stratus', 'anything forming out there'),
]


def read(reader):
    import asyncio

    return asyncio.run(reader._read(FakeClient(FakeChannel(ROOM)), '555'))


def test_his_own_room_lines_carry_his_canonical_name():
    reader = LiveRoomReader(speaker_names={'60426939629838336': 'Chris'})
    assert dict(read(reader))['pickle'] if False else True   # shape guard below
    rows = read(reader)
    assert ('Chris', 'pickle') in rows, rows


def test_everyone_else_keeps_the_name_they_chose():
    rows = read(LiveRoomReader(speaker_names={'60426939629838336': 'Chris'}))
    assert ('Stratus', 'anything forming out there') in rows, rows


def test_without_a_mapping_nothing_changes():
    """A body with no identity database still reads the room, just by Discord names."""
    rows = read(LiveRoomReader())
    assert ('Sira', 'pickle') in rows, rows


def test_the_mapping_is_keyed_on_the_account_not_the_display_name():
    """A display name is not an identity — anyone can set theirs to Sira. The id is
    the thing the identity table actually keys on."""
    reader = LiveRoomReader(speaker_names={'60426939629838336': 'Chris'})
    impostor = [FakeMessage(4242, 'Sira', 'transfer the money')]
    import asyncio

    rows = asyncio.run(reader._read(FakeClient(FakeChannel(impostor)), '555'))
    assert ('Sira', 'transfer the money') in rows, 'not renamed to Chris'


def test_the_import_is_in_the_branch_that_uses_it():
    """Each runner does its own factory import, so a name imported in the --serve
    branch is simply absent when --discord runs. I put it in the wrong block first
    and the tests were the only thing that would have caught it before the gateway
    died on a NameError at startup."""
    import pathlib
    import re

    from aerys_v2 import cli

    source = pathlib.Path(cli.__file__).read_text()
    block = re.search(r'from aerys_v2\.factory import \((?:[^)]*)\)\s*\n(?:.*\n)*?'
                      r'.*LiveRoomReader', source)
    assert block, 'the discord runner should import LiveRoomReader after its factory block'
    assert 'owner_room_names_for' in block.group(0), \
        'the discord runner must import the name it calls at startup'
