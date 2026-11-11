from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from aerys_v2.factory import _reach_feedback_line
from aerys_v2.kael_line import _said_to_owner
from aerys_v2.reach import MemoryReachStore, Prefs, muted_topics
from aerys_v2.tools.reach_prefs import build_reach_prefs_tool

NOW = datetime(2026, 11, 11, 18, tzinfo=timezone.utc)


def test_delivery_metadata_and_legacy_delivery():
    calls = []
    graph = SimpleNamespace(update_state=lambda config, state: calls.append(state['messages'][0]))
    settings = SimpleNamespace(owner_person_id='owner')
    _said_to_owner(graph, settings, 'hello', SimpleNamespace(topic='weather', source='forecast'))
    assert calls[-1].additional_kwargs == {'reach_topic': 'weather', 'reach_source': 'forecast'}
    _said_to_owner(graph, settings, 'answer')
    assert calls[-1].additional_kwargs == {}
    _said_to_owner(graph, settings, 'hello', SimpleNamespace(topic='', source='notice'))
    assert calls[-1].content == 'hello'


def test_quiet_lifecycle_and_audit():
    store = MemoryReachStore()
    tool = build_reach_prefs_tool(store, clock=lambda: NOW)
    assert tool.invoke(dict(action='quiet', topic='weather', his_words='ugh, not now')).startswith('Changed')
    prefs = store.prefs(3)
    assert muted_topics(prefs, NOW) == {'weather'}
    assert muted_topics(prefs, NOW + timedelta(hours=3)) == set()
    assert store.changes[-1][2] == 'ugh, not now'
    shown = tool.invoke(dict(action='show'))
    assert 'weather' in shown and '16:00' in shown and 'EST' in shown
    assert tool.invoke(dict(action='unmute', topic='weather')).startswith('Changed')
    assert store.prefs(3).muted == []
    tool.invoke(dict(action='quiet', topic='weather', hours=100))
    assert muted_topics(store.prefs(3), NOW + timedelta(hours=24)) == set()
    later = build_reach_prefs_tool(store, clock=lambda: NOW + timedelta(days=2))
    later.invoke(dict(action='less', his_words='less please'))
    assert store.prefs(3).muted == []


def test_bad_entries_ignored_and_logged_once(caplog):
    prefs = Prefs(muted=['email@bad-date-feedback-test', 'weather@2026-11-12T12:00:00', 'calendar'])
    assert muted_topics(prefs, NOW) == {'calendar'}
    assert muted_topics(prefs, NOW) == {'calendar'}
    assert len(caplog.records) == 2


def test_quiet_never_shortens_or_replaces_a_permanent_mute():
    store = MemoryReachStore(prefs=Prefs(muted=['weather']))
    build_reach_prefs_tool(store, clock=lambda: NOW).invoke(dict(action='quiet', topic='weather'))
    assert store.prefs(3).muted == ['weather']


def test_hint_requires_adjacent_stamped_reachout():
    stamped = AIMessage(content='forecast', additional_kwargs={'reach_topic': 'weather', 'reach_source': 'forecast'})
    human = HumanMessage(content='not now')
    hint = _reach_feedback_line([stamped, human])
    for word in ('weather', 'quiet', 'mute', 'never more', 'his_words', 'undo'):
        assert word in hint
    for history in ([], [human], [AIMessage(content='hello'), human], [stamped, human, human], [stamped]):
        assert _reach_feedback_line(history) == ''


@pytest.mark.parametrize('action', [False, True])
def test_actual_prompt_hint_and_unchanged_without_stamp(monkeypatch, action):
    from aerys_v2.factory import build_graph, build_action_graph
    from test_prompt_layout import Recorder
    monkeypatch.setattr('aerys_v2.factory._where_when_line', lambda *_: '\nCLOCK')
    def prompt(stamped):
        model = Recorder()
        graph = build_action_graph(model, 'soul', []) if action else build_graph(model, 'soul')
        history = [AIMessage(content='forecast', id='prior', additional_kwargs={'reach_topic': 'weather'} if stamped else {}),
                   HumanMessage(content='not now', id='current')]
        graph.invoke({'messages': history}, {'configurable': {'thread_id': 'feedback', 'identity': {'privacy_context': 'private'}}})
        return [m.model_dump() for m in model.prompts[0]]
    assert 'reach-out about weather' in str(prompt(True))
    unchanged = prompt(False)
    assert 'reach-out about' not in str(unchanged)
    monkeypatch.setattr('aerys_v2.factory._reach_feedback_line', lambda *_: '')
    assert prompt(False) == unchanged


@pytest.mark.parametrize('specialist', [False, True])
@pytest.mark.parametrize('escalated', [False, True])
def test_service_seed_preserves_immediate_reachout(specialist, escalated):
    from aerys_v2.service import _action_history_seed
    prior = [HumanMessage(content='earlier'), AIMessage(content='forecast', additional_kwargs={'reach_topic': 'weather'})]
    if escalated:
        prior += [HumanMessage(content='not now'), AIMessage(content='handoff')]
    graph = SimpleNamespace(get_state=lambda _: SimpleNamespace(values={'messages': prior}))
    seed = _action_history_seed(graph, {'identity': {'privacy_context': 'private'}}, 'not now',
                                specialist=specialist, escalated=escalated)
    assert 'reach-out about weather' in _reach_feedback_line(seed)


def test_timed_quiet_preserves_safety_and_owed_answers():
    from test_reach_desk import d, ev, DAY
    prefs = Prefs(muted=[f'weather@{(DAY + timedelta(hours=3)).isoformat()}'])
    assert d(ev(topic='weather'), prefs=prefs).action == 'drop'
    assert d(ev(topic='weather', owed=True), prefs=prefs).action == 'send'
    assert d(ev(topic='weather', urgency='safety'), prefs=prefs).action == 'send'


def test_brief_honors_timed_quiet():
    from test_morning_brief import House
    h = House()
    build_reach_prefs_tool(h.desk.store, clock=lambda: h.clock['now']).invoke(dict(action='quiet', topic='brief'))
    assert h.brief.tick() == 'muted' and not h.composed


def test_welcome_honors_timed_quiet():
    from test_welcome_home import House, SETTLED, utc
    h = House()
    build_reach_prefs_tool(h.desk.store, clock=lambda: utc(SETTLED)).invoke(dict(action='quiet', topic='welcome'))
    assert h.hello.tick() == 'muted' and not h.composed


def test_followups_honor_quiet_but_keep_owed_checkbacks():
    from test_followups import Life, appt, DAY
    life = Life()
    life.appts = [appt()]
    life.prefs.invoke(dict(action='quiet', topic='follow-ups'))
    life.tool.invoke(dict(action='add', when='2026-11-10T14:05', about='the appointment'))
    life.at(DAY + timedelta(minutes=6))
    assert life.runner.tick()['raised'] == 1
    assert life.calls == ['asked']
    assert all(r.owed for r in life.desk.store.rows.values())


def test_unmute_clears_plain_and_timed_entries_and_quiet_does_not_shorten():
    end = NOW + timedelta(hours=12)
    store = MemoryReachStore(prefs=Prefs(muted=[f'weather@{end.isoformat()}']))
    tool = build_reach_prefs_tool(store, clock=lambda: NOW)
    tool.invoke(dict(action='quiet', topic='weather', hours=1))
    assert store.prefs(3).muted == [f'weather@{end.isoformat()}']
    store.set_prefs(Prefs(muted=['weather', f'weather@{end.isoformat()}', 'email']), NOW, 'setup', '')
    tool.invoke(dict(action='unmute', topic='weather'))
    assert store.prefs(3).muted == ['email']
    assert 'never more or checkins_more' in tool.description


def test_desk_said_hook_passes_row(monkeypatch):
    from aerys_v2.kael_line import reach_desk_for
    from test_reach_desk import owner_settings
    recorded = []
    monkeypatch.setattr('aerys_v2.reach.PostgresReachStore', lambda _: MemoryReachStore())
    monkeypatch.setattr('aerys_v2.factory.presence_fn_for', lambda _: None)
    graph = SimpleNamespace(update_state=lambda config, state: recorded.extend(state['messages']))
    desk = reach_desk_for(owner_settings(database_url='postgresql://example.invalid/test'), graph)
    desk.said(SimpleNamespace(topic='weather', source='forecast'), 'forecast text')
    assert recorded[0].additional_kwargs == {'reach_topic': 'weather', 'reach_source': 'forecast'}


def test_specialist_does_not_restore_a_private_reachout_in_a_public_room():
    from aerys_v2.service import _action_history_seed
    prior = [AIMessage(content='forecast', additional_kwargs={'reach_topic': 'weather', 'content_privacy': 'private'})]
    graph = SimpleNamespace(get_state=lambda _: SimpleNamespace(values={'messages': prior}))
    seed = _action_history_seed(graph, {'identity': {'privacy_context': 'public'}}, 'not now', specialist=True)
    assert _reach_feedback_line(seed) == ''
