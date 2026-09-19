from aerys_v2.workers.reflex_report import summarize, format_report, read_reflex_rows


def test_three_row_agreement_math():
    def row(route, confidence, tier, unaddressed, latency):
        return {'input_text': 'hello', 'reflex': {
            'mode': 'shadow',
            'router': {'route': 'chat', 'tier': 'standard', 'unaddressed': False},
            'jev': {'route': route, 'confidence': confidence, 'p_action': .8,
                    'tier': tier, 'unaddressed': unaddressed, 'latency_ms': latency}}}
    rows = [row('chat', .9, 'standard', .1, 100), row('action', .8, 'deep', .9, 200),
            row('chat', .4, 'fast', .1, 300)]
    report = summarize(rows)
    assert report['n'] == 3 and report['errors'] == 0
    assert report['route_agreement'] == 2 / 3
    assert report['confident_route_agreement'] == 1 / 2
    assert report['tier_agreement'] == 1 / 3
    assert report['unaddressed_agreement'] == 2 / 3
    assert report['p50_ms'] == 200 and report['p90_ms'] == 280
    output = format_report(rows)
    assert 'truth=chat jev=action p=0.8 conf=0.8 :: hello' in output
    assert '66.7%' in output and '50.0%' in output


def test_errors_and_missing_router_are_not_agreements():
    rows = [{'input_text': '', 'reflex': {'jev': {'error': 'timeout'}, 'router': {}}}]
    report = summarize(rows)
    assert report['errors'] == 1
    assert report['route_agreement'] is None
    assert report['p50_ms'] is None
    assert 'n/a' in format_report(rows)
    assert summarize([])['n'] == 0


def test_query_is_parameterized_and_read_only():
    class Conn:
        def execute(self, sql, params):
            assert sql.lstrip().startswith('SELECT')
            assert 'reflex IS NOT NULL' in sql
            assert params == ('2 hours',)
            return self

        def fetchall(self):
            return [('hello', {'jev': {'error': 'timeout'}})]

    assert read_reflex_rows(Conn(), '2 hours') == [
        {'input_text': 'hello', 'reflex': {'jev': {'error': 'timeout'}}}]


def test_worker_dispatch_and_read_only_connection(monkeypatch, capsys):
    from types import SimpleNamespace
    import psycopg
    from aerys_v2.workers import __main__ as worker

    class Conn:
        read_only = False

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def execute(self, sql, params=None):
            assert self.read_only
            if sql.startswith('SELECT'):
                assert params == ('2 hours',)
            else:
                assert sql == "SET statement_timeout = '30s'"
            return self

        def fetchall(self):
            return []

    monkeypatch.setattr(worker, 'Settings', lambda: SimpleNamespace(database_url='fake'))
    monkeypatch.setattr(worker, 'run_boot_assertions', lambda _: None)
    monkeypatch.setattr(psycopg, 'connect', lambda *a, **k: Conn())
    assert worker.main(['reflex-report', '--window', '2 hours']) == 0
    assert 'n=0 errors=0' in capsys.readouterr().out
