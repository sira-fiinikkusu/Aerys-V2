"""DJ contracts against fake HA; all library names are invented placeholders."""
import json

import httpx
import pytest
from langchain_core.messages import AIMessage, ToolMessage

from aerys_v2.tools.music import build_music_tool

PLAYER = 'media_player.example'
OWNER = 'owner-example'
#: his own DM: the room where his library may be read back (D47's rule, via rooms.his_private_read)
PRIVATE = {'configurable': {'thread_id': 'person:owner-example', 'identity': {
    'user_id': OWNER, 'platform': 'discord', 'privacy_context': 'private', 'channel_kind': 'dm'}}}


class Private:
    """The music tool, invoked from his DM unless a test says otherwise."""
    def __init__(self, tool):
        self.tool = tool

    def invoke(self, args, config=None):
        return self.tool.invoke(args, config=config or PRIVATE)
LIBRARY = [
    {'name': 'Liked Songs Example User', 'uri': 'library://playlist/1'},
    {'name': 'All favorited tracks', 'uri': 'library://playlist/2'},
    {'name': 'Example Mix', 'uri': 'library://playlist/3'},
]

class HA:
    def __init__(self):
        self.library = list(LIBRARY)
        self.tracks = []
        self.artists = []
        self.calls = []
        self.state = {'state': 'playing', 'attributes': {'media_content_id': 'spotify://track/example'}}
        self.queue = {'current_item': {'name': 'Current Example'}, 'next_item': {
            'name': 'Next Example', 'media_item': {'name': 'Next Example', 'artists': [{'name': 'Example Artist'}]}}}
        self.fail = ''

    def __call__(self, req):
        op = req.url.path.rsplit('/', 1)[-1]
        data = json.loads(req.content) if req.content else {}
        self.calls.append((op, data))
        if op == self.fail:
            return httpx.Response(503)
        if op == 'get_library':
            assert req.url.query == b'return_response'
            pool = {'playlist': self.library, 'track': self.tracks, 'artist': self.artists}[data['media_type']]
            return httpx.Response(200, json={'service_response': {'items': pool[data['offset']:data['offset'] + data['limit']]}})
        if op == 'search':
            return httpx.Response(200, json={'service_response': {'tracks': [{'name': 'Example Track', 'uri': 'spotify://track/example'}]}})
        if op == 'get_queue':
            assert data == {'entity_id': PLAYER}
            return httpx.Response(200, json={'service_response': {PLAYER: self.queue}})
        if req.method == 'GET':
            return httpx.Response(200, json=self.state)
        return httpx.Response(200, json=[])

    def tool(self, **kwargs):
        return Private(build_music_tool(base_url='http://ha.test', token='example', config_entry_id='example',
                                        players={'example': PLAYER}, default_player=PLAYER, owner_person_id=OWNER,
                                        client=httpx.Client(transport=httpx.MockTransport(self)), **kwargs))

    def plays(self):
        return [body for op, body in self.calls if op == 'play_media']

@pytest.mark.parametrize('query', ['liked songs', 'my liked songs', 'my favorites', 'my favourites', 'life songs'])
def test_liked_family(query):
    ha = HA()
    assert ha.tool().invoke({'operation': 'play', 'query': query}).startswith('Done:')
    assert ha.plays()[0]['media_id'] == LIBRARY[0]['uri']
    assert not any(op == 'search' for op, _ in ha.calls)

@pytest.mark.parametrize('query', ['Example Mix', 'exampel mix'])
def test_library_exact_and_stt(query):
    ha = HA()
    ha.tool().invoke({'operation': 'play', 'query': query})
    assert ha.plays()[0]['media_id'] == LIBRARY[2]['uri']


def test_favorites_fallback():
    ha = HA(); ha.library = LIBRARY[1:]
    ha.tool().invoke({'operation': 'play', 'query': 'liked songs'})
    assert ha.plays()[0]['media_id'] == LIBRARY[1]['uri']

# 0.80 admits life/liked songs (0.857) and exampel/example mix (0.909),
# but rejects example moon/example mix (0.783) and bare mix (0.429).
@pytest.mark.parametrize('query', ['unrelated tune', 'mix', 'example moon'])
def test_no_clear_match_searches(query):
    ha = HA()
    ha.tool().invoke({'operation': 'play', 'query': query})
    assert ha.plays()[0]['media_id'] == 'spotify://track/example'


def test_ambiguous_fuzzy_match_searches():
    ha = HA(); ha.library = [{'name': n, 'uri': 'library://playlist/' + str(i)} for i, n in enumerate(['Example Mix A', 'Example Mix B'])]
    ha.tool().invoke({'operation': 'play', 'query': 'Example Mix'})
    assert ha.plays()[0]['media_id'] == 'spotify://track/example'


def test_explicit_track_does_not_become_playlist():
    ha = HA()
    ha.tool().invoke({'operation': 'play', 'query': 'Example Mix', 'media_type': 'track'})
    assert ha.plays()[0]['media_type'] == 'track'


def test_cache_expiry_and_pagination(monkeypatch):
    import importlib
    module = importlib.import_module('aerys_v2.tools.music')
    now = [0.0]
    monkeypatch.setattr(module.time, 'monotonic', lambda: now[0])
    ha = HA(); ha.library += [{'name': f'Example List {i}', 'uri': f'library://playlist/{i+10}'} for i in range(110)]
    tool = ha.tool()
    for _ in range(2):
        assert 'Example List 109' in tool.invoke({'operation': 'my_playlists'})
    assert len([1 for op, _ in ha.calls if op == 'get_library']) == 2
    now[0] = 301
    tool.invoke({'operation': 'my_playlists'})
    assert len([1 for op, _ in ha.calls if op == 'get_library']) == 4


def test_playlist_filter_without_speaker():
    ha = HA()
    tool = Private(build_music_tool(base_url='http://ha.test', token='example', config_entry_id='example', owner_person_id=OWNER, players={}, client=httpx.Client(transport=httpx.MockTransport(ha))))
    out = tool.invoke({'operation': 'my_playlists', 'query': 'mix'})
    assert 'Example Mix' in out and 'Liked Songs' not in out and not out.startswith('Done:')

@pytest.mark.parametrize('op,enqueue', [('play_next', 'next'), ('add_to_queue', 'add')])
@pytest.mark.parametrize('query,kind', [('Example Mix', 'playlist'), ('unrelated tune', 'track')])
def test_enqueue(op, enqueue, query, kind):
    ha = HA()
    out = ha.tool().invoke({'operation': op, 'query': query})
    assert out.startswith('Done:') and 'queued' in out
    assert ha.plays() == [{'entity_id': PLAYER, 'media_id': LIBRARY[2]['uri'] if kind == 'playlist' else 'spotify://track/example', 'media_type': kind, 'enqueue': enqueue}]

@pytest.mark.parametrize('op,value,key,expected', [('shuffle','on','shuffle',True), ('shuffle','off','shuffle',False), ('repeat','off','repeat','off'), ('repeat','one','repeat','one'), ('repeat','all','repeat','all')])
def test_modes(op, value, key, expected):
    ha = HA()
    assert ha.tool().invoke({'operation': op, 'query': value}).startswith('Done:')
    assert ha.calls == [(op + '_set', {'entity_id': PLAYER, key: expected})]

@pytest.mark.parametrize('op,value', [('shuffle','yes'), ('shuffle',''), ('repeat','on'), ('repeat','')])
def test_bad_modes_refused(op, value):
    ha = HA()
    assert not ha.tool().invoke({'operation': op, 'query': value}).startswith('Done:')
    assert ha.calls == []


def test_up_next():
    ha = HA()
    out = ha.tool().invoke({'operation': 'up_next'})
    assert 'Next Example by Example Artist' in out and 'Current Example' not in out
    assert 'only the next item' in out and not out.startswith('Done:')


def test_empty_queue():
    ha = HA(); ha.queue = {'next_item': None}
    assert 'No next item' in ha.tool().invoke({'operation': 'up_next'})


def test_more_like_this():
    ha = HA()
    assert ha.tool().invoke({'operation': 'more_like_this'}).startswith('Done:')
    assert ha.plays() == [{'entity_id': PLAYER, 'media_id': 'spotify://track/example', 'media_type': 'track', 'radio_mode': True}]

@pytest.mark.parametrize('state,attrs', [('idle', {'media_content_id': 'stale'}), ('paused', {'media_content_id': 'stale'}), ('playing', {})])
def test_no_radio_without_current_track(state, attrs):
    ha = HA(); ha.state = {'state': state, 'attributes': attrs}
    assert not ha.tool().invoke({'operation': 'more_like_this'}).startswith('Done:')
    assert ha.plays() == []

@pytest.mark.parametrize('op,fail', [('my_playlists','get_library'), ('up_next','get_queue'), ('shuffle','shuffle_set'), ('repeat','repeat_set'), ('more_like_this','play_media')])
def test_failures_honest(op, fail):
    ha = HA(); ha.fail = fail
    out = ha.tool().invoke({'operation': op, 'query': 'off'})
    assert not out.startswith('Done:') and ('FAILED' in out or "Couldn't" in out)


def test_library_failure_preserves_search():
    ha = HA(); ha.fail = 'get_library'
    ha.tool().invoke({'operation': 'play', 'query': 'unrelated tune'})
    assert ha.plays()[0]['media_id'] == 'spotify://track/example'


def test_quiet_cap_guards():
    from aerys_v2.service import _quiet_music_reply
    text = 'Example Mix is playing. Here is a long explanation.'
    music = ToolMessage(content='Done: playing', name='music', tool_call_id='1')
    assert _quiet_music_reply(text, [music]) == 'Example Mix is playing.'
    assert len(_quiet_music_reply(' '.join(['word'] * 30), [music]).split()) == 12
    for notes in [[], [ToolMessage(content='Done: on', name='home_control', tool_call_id='2')], [music, ToolMessage(content='FAILED', name='music', tool_call_id='3')], [music, ToolMessage(content='Done: on', name='home_control', tool_call_id='4')], [ToolMessage(content='Now playing', name='music', tool_call_id='5')]]:
        assert _quiet_music_reply(text, notes) == text

@pytest.mark.parametrize('voice,route,mixed', [(True,'action',False), (True,'chat',False), (True,'action',True), (False,'action',False)])
def test_quiet_cap_actual_delivery(voice, route, mixed):
    from test_action_orchestration import fake_model, wait_for_messages
    from aerys_v2.factory import build_graph
    from aerys_v2.router import RouteDecision
    from aerys_v2.service import ask
    text = 'Example Mix is playing. Here is a long explanation.'
    class Action:
        def invoke(self, inp, config):
            notes = [ToolMessage(content='Done: playing', name='music', tool_call_id='1')]
            if mixed:
                notes.append(ToolMessage(content='Done: on', name='home_control', tool_call_id='2'))
            return {'messages': [*inp['messages'], *notes, AIMessage(content=text)]}
    graph = build_graph(fake_model('unused'), soul='s')
    calls = []
    thread = 'voice:dj' if voice else 'text:dj'
    result = ask(graph, 'play Example Mix', identity={'user_id': 'example'}, thread_id=thread,
                 router=lambda _: RouteDecision(route=route, ack='Starting.'), action_graph=Action(),
                 followup_router=lambda text, device: calls.append(text), confirm_writes=True)
    wait_for_messages(graph, thread, 2)
    heard = calls[0] if voice and route == 'action' else result
    assert heard == ('Example Mix is playing.' if voice and not mixed else text)

@pytest.mark.parametrize('library,query', [
    ([{'name': 'Example Mix', 'uri': 'library://playlist/a'}, {'name': 'Example Mix', 'uri': 'library://playlist/b'}], 'Example Mix'),
    ([{'name': 'Liked Songs Example A', 'uri': 'library://playlist/a'}, {'name': 'Liked Songs Example B', 'uri': 'library://playlist/b'}], 'my favorites'),
    ([{'name': 'Example Mix', 'uri': ''}], 'Example Mix'),
    ([{'name': 'All favorited tracks', 'uri': 'library://playlist/a'}, {'name': 'All favorited tracks', 'uri': 'library://playlist/b'}], 'my favorites'),
])
def test_ambiguous_or_unplayable_library_is_not_selected(library, query):
    ha = HA(); ha.library = library
    ha.tool().invoke({'operation': 'play', 'query': query})
    assert ha.plays()[0]['media_id'] == 'spotify://track/example'


def test_empty_library_is_cached():
    ha = HA(); ha.library = []
    tool = ha.tool()
    assert tool.invoke({'operation': 'my_playlists'}) == 'No matching library playlists.'
    tool.invoke({'operation': 'my_playlists'})
    assert len(ha.calls) == 1


def test_failed_library_read_is_not_cached():
    ha = HA(); ha.fail = 'get_library'; tool = ha.tool()
    assert "Couldn't" in tool.invoke({'operation': 'my_playlists'})
    ha.fail = ''
    assert 'Example Mix' in tool.invoke({'operation': 'my_playlists'})

@pytest.mark.parametrize('response', [[], {}, {'service_response': {'items': None}}, {'service_response': {'items': ['bad']}}])
def test_malformed_library_refused(response):
    tool = Private(build_music_tool(base_url='http://ha.test', token='example', config_entry_id='example', owner_person_id=OWNER, players={}, client=httpx.Client(transport=httpx.MockTransport(lambda req: httpx.Response(200, json=response)))))
    assert "Couldn't read" in tool.invoke({'operation': 'my_playlists'})


def test_library_pagination_safety_limit():
    calls = []
    def repeated(req):
        calls.append(req)
        return httpx.Response(200, json={'service_response': {'items': [LIBRARY[0]] * 100}})
    tool = Private(build_music_tool(base_url='http://ha.test', token='example', config_entry_id='example', owner_person_id=OWNER, players={}, client=httpx.Client(transport=httpx.MockTransport(repeated))))
    assert 'pagination limit' in tool.invoke({'operation': 'my_playlists'})
    assert len(calls) == 100


def test_radio_uri_fallback_and_failure():
    ha = HA(); ha.state['attributes'] = {'uri': 'library://track/example'}
    assert ha.tool().invoke({'operation': 'more_like_this'}).startswith('Done:')
    assert ha.plays()[0]['media_id'] == 'library://track/example'
    ha.fail = PLAYER
    assert 'FAILED' in ha.tool().invoke({'operation': 'more_like_this'})


def test_queue_missing_player_is_unavailable_not_empty():
    tool = Private(build_music_tool(base_url='http://ha.test', token='example', config_entry_id='example', owner_person_id=OWNER, players={}, default_player=PLAYER, client=httpx.Client(transport=httpx.MockTransport(lambda req: httpx.Response(200, json={'service_response': {}})))))
    assert "Couldn't read" in tool.invoke({'operation': 'up_next'})


def test_queue_top_level_name():
    ha = HA(); ha.queue = {'next_item': {'name': 'Next Example'}}
    assert 'Next Example' in ha.tool().invoke({'operation': 'up_next'})

@pytest.mark.parametrize('op', ['play_next', 'add_to_queue', 'up_next', 'shuffle', 'repeat', 'more_like_this'])
def test_new_player_ops_respect_allowlist(op):
    ha = HA()
    assert "don't have a speaker" in ha.tool().invoke({'operation': op, 'query': 'Example Mix', 'target': 'unknown'})
    assert ha.calls == []

@pytest.mark.parametrize('op', ['play_next', 'add_to_queue'])
def test_enqueue_requires_query(op):
    ha = HA()
    assert 'what to play' in ha.tool().invoke({'operation': op})
    assert ha.calls == []


def test_quiet_prefix_and_sentence_boundaries():
    from aerys_v2.service import _quiet_music_reply
    note = ToolMessage(content='Done: playing', name='music', tool_call_id='1')
    for text, expected in [('Playing Example! More words.', 'Playing Example!'), ('Playing Example.\nMore words.', 'Playing Example.'), ('Playing Example.Next sentence.', 'Playing Example.Next sentence.'), ('Playing Example', 'Playing Example')]:
        assert _quiet_music_reply(text, [note]) == expected
    text = 'Playing Example. More words.'
    for content in ['Not Done: failed', ' Done: playing', ['Done: playing']]:
        assert _quiet_music_reply(text, [ToolMessage(content=content, name='music', tool_call_id='1')]) == text
    assert _quiet_music_reply(text, [AIMessage(content='Done: playing', name='music')]) == text


def test_exact_library_name_beats_near_duplicate():
    ha = HA(); ha.library.append({'name': 'Example Mix A', 'uri': 'library://playlist/near'})
    ha.tool().invoke({'operation': 'play', 'query': 'Example Mix'})
    assert ha.plays()[0]['media_id'] == LIBRARY[2]['uri']


def test_ambiguous_liked_accounts_do_not_fall_back_to_other_favorites():
    ha = HA(); ha.library.append({'name': 'Liked Songs Another Example', 'uri': 'library://playlist/other'})
    ha.tool().invoke({'operation': 'play', 'query': 'my favorites'})
    assert ha.plays()[0]['media_id'] == 'spotify://track/example'


# --- 11/14 20:05: his saved songs and artists come before a Spotify-wide search ---------------------------
SAVED = [
    {'name': 'Beneath the Copper Lantern', 'uri': 'library://track/10', 'artists': [{'name': 'Example Band'}]},
    {'name': 'The Copper Lantern', 'uri': 'library://track/11', 'artists': [{'name': 'Example Band'}]},
    {'name': 'Quiet Example', 'uri': 'library://track/12', 'artists': [{'name': 'Other Example'}]},
]


def saved_ha():
    ha = HA(); ha.tracks = [dict(t) for t in SAVED]
    ha.artists = [{'name': 'Example Band', 'uri': 'library://artist/1'}]
    return ha


@pytest.mark.parametrize('query', ['Beneath the Copper Lantern', 'beneath the cooper lantern',
                                   'beneath the copper lanterns', 'Beneath the Copper Lantern Example Band'])
def test_saved_song_beats_spotify_even_misheard(query):
    ha = saved_ha()
    ha.tool().invoke({'operation': 'play', 'query': query})
    assert ha.plays()[0]['media_id'] == 'library://track/10'
    exact = query in ('Beneath the Copper Lantern', 'Beneath the Copper Lantern Example Band')
    assert ('search' in [op for op, _ in ha.calls]) is not exact     # only a near name asks Spotify too


def test_saved_artist_and_explicit_types():
    ha = saved_ha()
    ha.tool().invoke({'operation': 'play', 'query': 'example band'})
    assert ha.plays()[0] == {'entity_id': PLAYER, 'media_id': 'library://artist/1', 'media_type': 'artist'}
    ha = saved_ha()
    ha.tool().invoke({'operation': 'play', 'query': 'Quiet Example', 'media_type': 'album'})
    assert 'search' in [op for op, _ in ha.calls]          # an album ask skips his saved songs


def test_two_saved_copies_are_one_choice_and_near_names_are_not():
    ha = saved_ha(); ha.tracks.append({'name': 'Quiet Example', 'uri': 'library://track/13',
                                       'artists': [{'name': 'Other Example'}]})
    ha.tool().invoke({'operation': 'play', 'query': 'quiet exampel'})
    assert ha.plays()[0]['media_id'] == 'library://track/12'
    ha = saved_ha(); ha.tracks = [{'name': 'Copper Lantern A', 'uri': 'u1'}, {'name': 'Copper Lantern B', 'uri': 'u2'}]
    ha.tool().invoke({'operation': 'play', 'query': 'copper lantern'})
    assert ha.plays()[0]['media_id'] == 'spotify://track/example'    # tied: no guess, search instead


def test_playlist_name_beats_saved_song():
    ha = saved_ha(); ha.library.append({'name': 'Quiet Example', 'uri': 'library://playlist/9'})
    ha.tool().invoke({'operation': 'play', 'query': 'Quiet Example'})
    assert ha.plays()[0]['media_id'] == 'library://playlist/9'


def test_saved_outage_falls_back_to_search():
    ha = saved_ha()
    real = ha.__call__
    ha.__call__ = None
    def flaky(req):
        if req.url.path.endswith('get_library') and json.loads(req.content)['media_type'] == 'track':
            return httpx.Response(503)
        return real(req)
    tool = build_music_tool(base_url='http://ha.test', token='example', config_entry_id='example',
                            players={'example': PLAYER}, default_player=PLAYER,
                            client=httpx.Client(transport=httpx.MockTransport(flaky)))
    tool.invoke({'operation': 'play', 'query': 'beneath the copper lantern'})
    assert ha.plays()[0]['media_id'] == 'spotify://track/example'


def test_my_taste_reads_his_library():
    ha = saved_ha()
    out = ha.tool().invoke({'operation': 'my_taste'})
    assert '3 saved songs, 3 playlists' in out
    assert 'Example Band (2), Other Example (1)' in out
    assert 'Example Mix' in out


@pytest.mark.parametrize('text,expected', [
    ('Playing Mr. Example Sky now. Want more?', 'Playing Mr. Example Sky now.'),
    ('Here is P.S. I Example You. Enjoy it.', 'Here is P.S. I Example You.'),
    ('Example Vol. 2 is on. It has more.', 'Example Vol. 2 is on.'),
    ('[warmly] Here\'s "Example!" Long story after.', '[warmly] Here\'s "Example!"'),
])
def test_quiet_cap_keeps_title_dots(text, expected):
    from aerys_v2.service import _quiet_music_reply
    note = ToolMessage(content='Done: playing', name='music', tool_call_id='1')
    assert _quiet_music_reply(text, [note]) == expected


def test_a_close_rival_below_the_bar_still_blocks_a_guess():
    # best 0.80, rival 0.727 (under the bar but within the 0.10 margin): no guess, search instead
    ha = HA(); ha.tracks = [{'name': 'abcdefghxy', 'uri': 'u1'}, {'name': 'abcdefghzzzz', 'uri': 'u2'}]
    ha.tool().invoke({'operation': 'play', 'query': 'abcdefghij'})
    assert ha.plays()[0]['media_id'] == 'spotify://track/example'


def test_same_title_by_different_artists_is_not_a_guess():
    ha = HA(); ha.tracks = [{'name': 'Example Night', 'uri': 'u0', 'artists': [{'name': 'Band Zero'}]},
                            {'name': 'Example Night', 'uri': 'u1', 'artists': [{'name': 'Band One'}]}]
    ha.tool().invoke({'operation': 'play', 'query': 'Example Nights'})
    assert ha.plays()[0]['media_id'] == 'spotify://track/example'
    ha.calls.clear()
    ha.tool().invoke({'operation': 'play', 'query': 'Example Night'})
    assert ha.plays()[0]['media_id'] == 'spotify://track/example'
    ha.calls.clear()
    ha.tool().invoke({'operation': 'play', 'query': 'Example Night Band One'})
    assert ha.plays()[0]['media_id'] == 'u1'


def test_a_near_saved_name_yields_to_an_exact_spotify_name():
    ha = HA(); ha.tracks = [{'name': 'Example Tracks', 'uri': 'library://track/near', 'artists': []}]
    ha.tool().invoke({'operation': 'play', 'query': 'Example Track'})      # fake search returns "Example Track"
    assert ha.plays()[0]['media_id'] == 'spotify://track/example'


def test_search_down_still_plays_his_near_saved_song():
    ha = saved_ha(); ha.fail = 'search'
    ha.tool().invoke({'operation': 'play', 'query': 'beneath the cooper lantern'})
    assert ha.plays()[0]['media_id'] == 'library://track/10'


@pytest.mark.parametrize('identity,thread,word', [
    ({'user_id': OWNER, 'platform': 'discord', 'privacy_context': 'public', 'channel_kind': 'guild'}, 'discord:1', 'Not here'),
    ({'user_id': 'guest-example', 'platform': 'discord', 'privacy_context': 'private'}, 'person:g', 'only he can ask'),
    ({'user_id': OWNER, 'voice': True, 'speaker': {'id': 'unknown'}, 'privacy_context': 'private'}, 'person:o', 'not sure'),
])
@pytest.mark.parametrize('op', ['my_playlists', 'my_taste'])
def test_his_library_reads_stay_his(op, identity, thread, word):
    ha = saved_ha()
    out = ha.tool().invoke({'operation': op}, config={'configurable': {'thread_id': thread, 'identity': identity}})
    assert word in out and 'Example Mix' not in out and 'Example Band' not in out
    assert ha.calls == []                           # refused before anything is fetched


def test_playing_music_needs_no_private_room():
    ha = saved_ha()
    guild = {'configurable': {'thread_id': 'discord:1', 'identity': {'user_id': OWNER, 'platform': 'discord',
                                                                     'privacy_context': 'public'}}}
    assert ha.tool().invoke({'operation': 'play', 'query': 'Beneath the Copper Lantern'}, config=guild).startswith('Done:')


@pytest.mark.parametrize('text,note,expected', [
    ('Want me to queue another song?', 'Done: playing Example Song on example. Keep it short.', 'Playing Example Song.'),
    ('Playing Example; I can queue more if you like.', 'Done: playing Example on example.', 'Playing Example.'),
    ('Playing Example… Want more?', 'Done: playing Example on example.', 'Playing Example…'),
    ('Playing Example — enjoy it, and let me know.', 'Done: playing Example on example.', 'Playing Example'),
    ('I can add more after this one.', 'Done: queued next Example Two on example.', 'Example Two is up next.'),
    ('Should I keep going?', 'Done: similar-track radio started on example.', 'More like this, coming up.'),
    ('Tell me what you think.', 'Done: pause sent to example.', 'Done.'),
])
def test_quiet_cap_never_offers_or_asks(text, note, expected):
    from aerys_v2.service import _quiet_music_reply
    assert _quiet_music_reply(text, [ToolMessage(content=note, name='music', tool_call_id='1')]) == expected


def test_a_near_saved_name_only_yields_to_an_exact_name_playback_will_use():
    def run(search, media_type=''):
        ha = HA(); ha.tracks = [{'name': 'Example Tracks', 'uri': 'library://track/near', 'artists': []}]
        real = ha.__call__
        def fake(req):
            if req.url.path.endswith('/search'):
                ha.calls.append(('search', {}))
                return httpx.Response(200, json={'service_response': search})
            return real(req)
        tool = Private(build_music_tool(base_url='http://ha.test', token='example', config_entry_id='example',
                                        players={'example': PLAYER}, default_player=PLAYER, owner_person_id=OWNER,
                                        client=httpx.Client(transport=httpx.MockTransport(fake))))
        tool.invoke({'operation': 'play', 'query': 'Example Track', 'media_type': media_type})
        return ha.plays()[0]['media_id']
    unrelated = [{'name': f'Unrelated {n}', 'uri': f'spotify://track/u{n}'} for n in range(3)]
    # an exact name only in a category the ask excluded: keep his saved song
    assert run({'artists': [{'name': 'Example Track', 'uri': 'spotify://artist/x'}], 'tracks': unrelated[:1]},
               'track') == 'library://track/near'
    # the exact name is ranked 4th: still found, still played
    assert run({'tracks': unrelated + [{'name': 'Example Track', 'uri': 'spotify://track/exact'}]}) == 'spotify://track/exact'
    # punctuation is not a different name
    assert run({'tracks': unrelated[:1] + [{'name': 'Example Track!', 'uri': 'spotify://track/bang'}]}) == 'spotify://track/bang'
    # nothing exact anywhere: his saved song
    assert run({'tracks': unrelated}) == 'library://track/near'


@pytest.mark.parametrize('text,note', [
    ('Want more?', 'Done: playing An Example Song With A Very Long Title Indeed by The Example Ensemble Players on example.'),
    ('Could I queue another song for you after this current song has finished playing on the speaker?',
     'Done: playing Example on example.'),
])
def test_quiet_fallback_is_capped_and_questions_are_judged_before_the_cut(text, note):
    from aerys_v2.service import _quiet_music_reply
    out = _quiet_music_reply(text, [ToolMessage(content=note, name='music', tool_call_id='1')])
    assert len(out.split()) <= 12 and '?' not in out and out.startswith('Playing')


def test_a_failed_exact_play_still_falls_back_to_his_saved_song():
    ha = HA(); ha.tracks = [{'name': 'Example Tracks', 'uri': 'library://track/near', 'artists': []}]
    real = ha.__call__
    def fake(req):
        if req.url.path.endswith('/search'):
            return httpx.Response(200, json={'service_response': {'tracks': [
                {'name': 'Example Track', 'uri': 'spotify://track/exact'},
                {'name': 'Unrelated', 'uri': 'spotify://track/other'}]}})
        if req.url.path.endswith('/play_media') and json.loads(req.content)['media_id'] == 'spotify://track/exact':
            ha.calls.append(('play_media_failed', {}))
            return httpx.Response(503)
        return real(req)
    tool = Private(build_music_tool(base_url='http://ha.test', token='example', config_entry_id='example',
                                    players={'example': PLAYER}, default_player=PLAYER, owner_person_id=OWNER,
                                    client=httpx.Client(transport=httpx.MockTransport(fake))))
    out = tool.invoke({'operation': 'play', 'query': 'Example Track'})
    assert out.startswith('Done:') and ha.plays()[-1]['media_id'] == 'library://track/near'
