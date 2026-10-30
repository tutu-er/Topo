"""Source/CLI compatibility only; no physical producer, fitting or real data."""
import argparse
import json
import sys
import types
import pytest

from research_experiments.wzzt_dr_selection.dr_coded_selection import actual_worker, controller


def case(job='flynn', bundle=7, sigma=.0002):
    return dict(job_id=job, history_row=dict(job_id=job, case='flynn16', bundle=bundle,
        voltage_sigma=sigma, condition='reference'))


def actual(**updates):
    result = dict(labels=[201, 202], status='complete', training_event_ids=[0, 1],
        validation_event_ids=[2, 3], base_kw=100., base_kv=.4, base_mva=.1, root_label=1)
    result.update(updates)
    return result


def metadata(value, case_name='flynn16'):
    return actual_worker.observation_metadata(value, case=case_name,
        expected_lock_sha='declared', policy='opposed', path=0, dependence='local')


def test_explicit_job_precedes_legacy_bundle_noise_filter():
    chosen = case()
    legacy = case('legacy', 8, .001)
    assert controller.select_prepared_case([legacy, chosen], 'flynn') is chosen
    assert controller.select_prepared_case([legacy, chosen]) is legacy
    with pytest.raises(ValueError, match='explicitly selected'):
        controller.select_prepared_case([chosen, chosen], 'flynn')
    with pytest.raises(ValueError, match='default pilot'):
        controller.select_prepared_case([chosen])


def test_nonlegacy_history_requires_explicit_binding():
    c = case()
    with pytest.raises(ValueError, match='explicit base_kw'):
        controller.history_loader_kwargs(c)
    c.update(base_kw=100., source_metadata=dict(path='locked-metadata', sha256='declared', case_name='declared-case'))
    kwargs = controller.history_loader_kwargs(c)
    assert kwargs == dict(expected_case='flynn16', base_kw=100., source_metadata=c['source_metadata'])
    assert controller.history_loader_kwargs(dict(history_row=dict(case='paper15'))) == dict(
        expected_case='paper15', base_kw=200., source_metadata=None)


def test_configured_history_forwards_frozen_kwargs_without_reading(monkeypatch):
    from research_experiments.wzzt_dr_selection.dr_competition_bridge import physical_bridge
    calls = []
    monkeypatch.setattr(physical_bridge, 'load_history', lambda row, **kw: calls.append((row, kw)) or 'observed')
    kwargs = dict(expected_case='flynn16', base_kw=100., source_metadata={'sha256': 'declared'})
    cfg = dict(history_row={'case': 'flynn16'}, history_loader_kwargs=kwargs)
    assert controller.configured_history(cfg) == 'observed'
    assert calls == [(cfg['history_row'], kwargs)]
    controller.configured_history(dict(history_row={'case': 'paper15'}))
    assert calls[-1] == ({'case': 'paper15'}, {})


def test_worker_metadata_uses_guarded_producer_bases():
    m = metadata(actual())
    assert (m['base_kw'], m['base_kv'], m['base_mva'], m['root_label']) == (100., .4, .1, 1)
    assert m['observed_only'] and m['case'] == 'flynn16'
    assert m['training_event_ids'] == [0, 1] and 'physical_network' not in m


def test_worker_rejects_missing_or_inconsistent_nonlegacy_bases():
    missing = actual()
    for key in ('base_kw', 'base_kv', 'base_mva', 'root_label'):
        missing.pop(key)
    with pytest.raises(ValueError, match='explicit bases'):
        metadata(missing)
    legacy = metadata(missing, 'paper15')
    assert legacy['base_kw'] == 200. and legacy['base_mva'] == .2
    with pytest.raises(ValueError, match='explicit bases'):
        metadata({**missing, 'base_kw': 200.}, 'paper15')
    with pytest.raises(ValueError, match='power bases disagree'):
        metadata(actual(base_kw=200.))
    with pytest.raises(ValueError, match='root cannot be a terminal'):
        metadata(actual(root_label=201))


def test_worker_cli_forwards_case_before_any_physical_work(tmp_path, monkeypatch):
    captured = {}
    module = types.ModuleType('coded_physical')
    source = tmp_path / 'fake_coded_physical.py'
    source.write_text('# test stub, no producer\n', encoding='utf-8')
    module.__file__ = str(source)
    def stop(*args, **kwargs):
        captured.update(kwargs)
        raise RuntimeError('test stop before any physical generation')
    module.generate_actual_events = stop
    module.evaluate_supports = lambda *args: pytest.fail('no evaluator should run')
    monkeypatch.setitem(sys.modules, 'coded_physical', module)
    out = tmp_path / 'output'
    old_path = list(sys.path)
    try:
        status = actual_worker.main(['--plan-lock', str(tmp_path / 'not-opened.json'),
            '--expected-lock-sha', 'declared', '--policy', 'opposed', '--path', '0',
            '--dependence', 'local', '--sigma', '.0002', '--case', 'flynn16',
            '--source-root', str(tmp_path), '--output', str(out)])
    finally:
        sys.path[:] = old_path
    assert status == 2 and captured['case'] == 'flynn16'
    receipt = json.loads((out / 'events_receipt.json').read_text(encoding='utf-8'))
    assert receipt['case'] == 'flynn16' and receipt['status'] == 'worker_failed'
    assert not (out / 'packet.pkl').exists()
