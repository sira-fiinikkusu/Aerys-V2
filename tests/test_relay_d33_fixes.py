from types import SimpleNamespace
from datetime import timedelta

import pytest

from aerys_v2.reach import MemoryReachStore, ReachDesk, ReachEvent
from aerys_v2.tools.pass_to_chris import Relay
from test_reach_desk import DAY
from test_pass_to_chris import CONTACT


def make_relay(names):
    from aerys_v2.tools.pass_to_chris import ReachQueue
    store = MemoryReachStore()
    queue = ReachQueue(store, clock=lambda: DAY)
    relay = Relay('owner', {'contact'}, queue, names.get)
    sent = []
    deliverer = SimpleNamespace(deliver=lambda row, decision: (sent.append(row) or SimpleNamespace(
        status='sent', surface='discord', spoken=False, texted=True)))
    desk = ReachDesk(store, deliverer, clock=lambda: DAY)
    return relay, store, desk, sent


def test_only_serving_desk_delivers_once():
    relay, store, desk, sent = make_relay({'contact': 'Sam', 'owner': 'Chris'})
    assert relay.send('hello', dict(CONTACT, display_name='Chris'), '') == "Queued for Chris. He gets it as soon as he's free (a meeting or his quiet hours can hold it a while)."
    row = next(r for r in store.rows.values() if r.source == 'relay')
    assert row.status == 'queued' and row.headline == 'Sam says: "hello"'
    assert sent == []
    desk.tick()
    desk.tick()
    assert sent == [row] and row.status == 'sent'


@pytest.mark.parametrize('name', [None, '', '  ', 'Chris', ' cHRis ', 'Chris.', 'chris p', '(Chris)', 'Mr Chris'])
def test_record_name_required_and_owner_name_reserved(name):
    relay, store, desk, sent = make_relay({'contact': name, 'owner': 'Chris'})
    result = relay.send('hello', dict(CONTACT, display_name='Sam'), '')
    assert result.startswith('Not passed on:')
    if not name or not name.strip():
        assert result == "Not passed on: I don't have your name on record"
    assert not store.rows and not sent


def test_relay_never_reserved_or_settled_as_fold():
    relay, store, desk, sent = make_relay({'contact': 'Sam', 'owner': 'Chris'})
    relay.send('hello', CONTACT, '')
    row = next(r for r in store.rows.values() if r.source == 'relay')
    ordinary = store.add(ReachEvent(source='test', topic='home', headline='ordinary'), DAY,
                         status='queued', reason='waiting', not_before=None)
    assert desk.reserve_for_brief(DAY - timedelta(hours=12), DAY + timedelta(hours=1)) == [ordinary]
    store.settle(ordinary.id, DAY, status='sent', folds=(row.id,))
    assert row.status == 'queued'
    desk.tick()
    assert sent == [row] and sent[0].source == 'relay'


def test_factory_uses_shared_queue_and_memories_names(monkeypatch):
    import psycopg
    import aerys_v2.hands_gate as gates
    from aerys_v2.tools.pass_to_chris import relay_for, relay_queue_for, ReachQueue
    from aerys_v2.reach import PostgresReachStore
    settings = SimpleNamespace(messages_as_herself=True, database_url='reach-db',
                               memories_database_url='memories-db', owner_person_id='owner')
    queue = relay_queue_for(settings)
    assert isinstance(queue, ReachQueue) and isinstance(queue.store, PostgresReachStore)
    assert not hasattr(queue, 'deliverer') and not hasattr(queue, 'tick')
    queue.store = MemoryReachStore()
    queue.clock = lambda: DAY
    lookups = []
    class Conn:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def execute(self, sql, args):
            assert self.read_only
            assert sql == 'SELECT display_name FROM persons WHERE id = %s::uuid'
            lookups.append(args)
            return SimpleNamespace(fetchone=lambda: ({'contact': 'Record Sam', 'owner': 'Chris'}[args[0]],))
    def connect(db):
        assert db == 'memories-db'
        return Conn()
    monkeypatch.setattr(psycopg, 'connect', connect)
    monkeypatch.setattr(gates, 'hands_gate_for', lambda s: SimpleNamespace(message_sender=SimpleNamespace(
        people={'contact'}, gate=SimpleNamespace(owner='owner'))))
    relay = relay_for(settings, queue)
    assert relay.send('hello', dict(CONTACT, display_name='Platform Sam'), '') == "Queued for Chris. He gets it as soon as he's free (a meeting or his quiet hours can hold it a while)."
    assert lookups == [('contact',), ('owner',)]
    assert next(r for r in queue.store.rows.values() if r.source == 'relay').headline == 'Record Sam says: "hello"'


def test_lookup_missing_record_and_database_failure(monkeypatch):
    import psycopg
    from aerys_v2.tools.pass_to_chris import person_name_lookup
    relay, store, desk, sent = make_relay({})
    class Conn:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def execute(self, *args):
            return SimpleNamespace(fetchone=lambda: None)
    monkeypatch.setattr(psycopg, 'connect', lambda db: Conn())
    relay.person_name = person_name_lookup('memories-db')
    assert relay.send('hello', CONTACT, '') == "Not passed on: I don't have your name on record"
    def failed(db):
        raise OSError('unavailable')
    monkeypatch.setattr(psycopg, 'connect', failed)
    assert relay.send('hello', CONTACT, '').startswith('Not passed on:')
    assert not store.rows


def test_postgres_fold_queries_exclude_relay(monkeypatch):
    from aerys_v2.reach import PostgresReachStore
    queries = []
    class Conn:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def execute(self, sql, args):
            queries.append(sql)
            return SimpleNamespace(fetchall=lambda: [])
    store = PostgresReachStore('unused')
    monkeypatch.setattr(store, '_connect', Conn)
    assert store.foldable(DAY, DAY + timedelta(hours=1)) == []
    store.settle(1, DAY, folds=(2,))
    assert len(queries) == 2
    assert all("source <> 'relay'" in sql for sql in queries)


def test_overnight_relay_stays_out_of_morning_brief():
    from test_morning_brief import House, NIGHT, IN_OFFICE
    from aerys_v2.tools.pass_to_chris import ReachQueue
    h = House(NIGHT)
    queue = ReachQueue(h.desk.store, clock=lambda: h.clock['now'])
    relay = Relay('owner', {'contact'}, queue, {'owner': 'Chris', 'contact': 'Sam'}.get)
    assert relay.send('hello', CONTACT, '') == "Queued for Chris. He gets it as soon as he's free (a meeting or his quiet hours can hold it a while)."
    row = next(r for r in h.desk.store.rows.values() if r.source == 'relay')
    assert h.desk.tick()['sent'] == 0
    h.at(IN_OFFICE)
    assert h.brief.tick() == 'sent'
    assert not h.composed[0].held
    assert row.status == 'queued'
    assert all(row.id not in r.folds for r in h.desk.store.rows.values())
    h.at(IN_OFFICE.replace(hour=13, minute=1))
    provenance = []
    h.desk.said = lambda r, body: provenance.append(r.source)
    assert h.desk.tick()['sent'] == 1
    assert row.status == 'sent' and provenance == ['relay']


def test_a_name_sharing_no_word_with_the_owner_is_fine():
    relay, store, desk, sent = make_relay({'contact': 'Christopher Robin', 'owner': 'Chris'})
    assert relay.send('hello', CONTACT, '') == "Queued for Chris. He gets it as soon as he's free (a meeting or his quiet hours can hold it a while)."


def test_a_shared_surname_is_fine():
    relay, store, desk, sent = make_relay({'contact': 'Sam Rivera', 'owner': 'Alex Rivera'})
    assert relay.send('hello', CONTACT, '') == "Queued for Chris. He gets it as soon as he's free (a meeting or his quiet hours can hold it a while)."
    relay, store, desk, sent = make_relay({'contact': 'Alex Smith', 'owner': 'Alex Rivera'})
    assert relay.send('hello', CONTACT, '').startswith('Not passed on')
