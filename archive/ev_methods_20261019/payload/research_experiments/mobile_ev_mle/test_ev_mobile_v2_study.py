"""Independent driver wiring checks; no optimization-policy or formal batch run."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, replace
from hashlib import sha256
from types import SimpleNamespace

import numpy as np
import pytest

from . import ev_mobile_v2_study as study
from . import ev_mobile_cases_v2 as newcases
from . import ev_systematic_cases as oldcases


def _same(left, right):
    if isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _same(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            _same(a, b)
    elif isinstance(left, np.ndarray):
        np.testing.assert_array_equal(left, right)
    else:
        assert left == right


def _trained(bank, observations, meter, cfg, seed):
    fits = {candidate.name: SimpleNamespace(candidate=candidate.name,
        r=np.full(candidate.m, .003), x=np.full(candidate.m, .0008),
        source_p=np.full(observations.t, .05), source_q=np.full(observations.t, .02),
        loss=1., success=True) for candidate in bank}
    return fits, dict(all_complete=True, fits=[], failures={})


def _spec(case_id, seed=101140001, fi=3):
    return dict(physical_bank='n4' if case_id == 'twinblock12' else 'n3',
        phase='development', index=0, case_id=case_id, seed=seed, family_index=fi)


@pytest.mark.parametrize('case_id', ['ordinary96', 'remote8_H8', 'remote8_H32', 'remote8_H96', 'twinblock12'])
def test_driver_case_mapping_forecast_and_public_domains(case_id):
    case = study.make_case(case_id, 101140002, 3)
    cfg = case['config']
    fits, _ = _trained(case['bank'], case['train'], case['tm'], cfg, 0)
    actual = study.oldcases.forecast_for(case['train'], case['tm'], fits, cfg)
    expected = (newcases.forecast_for if case['case_id'] in newcases.CASES else oldcases.forecast_for)(
        case['train'], case['tm'], fits, cfg)
    _same(actual, expected)
    assert cfg['experiment_case_id'] == case_id
    if case_id.startswith('remote8_H'):
        assert case['train'].t == int(case_id.split('_H')[-1])
    T, n = cfg['horizon'], case['train'].n
    domains = study.domains_for(case)
    for name, domain in domains.items():
        mobile = name == 'mobile'
        expected_domain = (newcases.service_domain(case['case_id'], mobile=mobile)
            if case['case_id'] in newcases.CASES else
            oldcases.service_domain(case['case_id'], T, n, mobile=mobile))
        _same(asdict(domain.config), asdict(expected_domain.config))
        assert domain.indices.u.shape == (T, n)
        assert domain.config.initial_site == 0
        assert domain.config.max_moves == 1
        assert domain.config.terminal_energy_kwh == pytest.approx(26.85)
        assert domain.config.mobile is mobile
    for candidate in case['bank']:
        # These are complete edge supports, not only optional internal clades.
        assert len(candidate.supports) == candidate.m
        assert tuple(range(n)) in candidate.supports
        assert all((j,) in candidate.supports for j in range(n))
    assert {2*k.m for k in case['bank']} == ({10, 12, 14} if n == 4 else {8, 10})


@pytest.mark.parametrize('case_id', ['remote8_H8', 'remote8_H96', 'twinblock12'])
def test_planning_boundary_ignores_validation_truth_and_future_noise(monkeypatch, case_id):
    original = study.make_case(case_id, 101140003, 2)
    changed = deepcopy(original)
    start = changed['future_start']
    for key in ('p', 'q', 'source_p', 'source_q', 'root_v'):
        changed[key][start:] = 987.
    for values in changed['white'].values():
        values[start:] = -765.
    changed['truth']['r'] *= 2.
    changed['truth']['x'] *= 3.
    changed['validation'] = replace(changed['validation'], v=changed['validation'].v + .8)
    captured = []

    def select(training, bank, fits, tm, fm, forecast, **kwargs):
        domains = kwargs.pop('domains')
        captured.append(dict(training=study.io.observations_record(training),
            bank=[(k.name, k.n, k.supports) for k in bank],
            fits={k: vars(v) for k, v in fits.items()},
            tm=study.meter_to_record(tm), fm=study.meter_to_record(fm), forecast=forecast,
            public=kwargs, domains={k: asdict(v.config) for k, v in domains.items()}))
        assert not {'truth', 'validation', 'white', 'case', 'family_index', 'true_name'} & set(kwargs)
        raise StopIteration('stop before execution/inference')

    monkeypatch.setattr(study.rx, 'train_bank', _trained)
    monkeypatch.setattr(study.design, 'select_plans', select)
    for case in (original, changed):
        monkeypatch.setattr(study, 'make_case', lambda *args, _case=case: _case)
        with pytest.raises(StopIteration, match='stop before execution'):
            study._run(_spec(case_id, 101140003, 2))
    _same(captured[0], captured[1])


def test_seed_groups_are_independent_and_development_pairing_is_explicit():
    groups = {}
    for bank in ('n3', 'n4'):
        for phase, count in (('development', 20 if bank == 'n3' else 8),
                             ('calibration', 96), ('evaluation', 48)):
            rows = [study.specification(bank, phase, i) for i in range(count)]
            groups[(bank, phase)] = {row['seed'] for row in rows}
            assert len(groups[(bank, phase)]) == (8 if bank == 'n3' and phase == 'development' else count)
            for row in rows:
                assert row['physical_bank'] == bank and row['family_index'] in range(4)
                if phase in ('calibration', 'evaluation'):
                    rng = np.random.default_rng(np.random.SeedSequence([row['seed'], 94991]))
                    expected = study.N3_CASES[int(rng.integers(len(study.N3_CASES)))] if bank == 'n3' else 'twinblock12'
                    assert row['case_id'] == expected
                    assert row['family_index'] == int(rng.integers(4))
        for i in range(3 if bank == 'n3' else 1):
            row = study.specification(bank, 'pilot', i)
            assert all(row['seed'] not in values for values in groups.values())
    keys = list(groups)
    for i, key in enumerate(keys):
        for other in keys[i+1:]:
            assert not groups[key] & groups[other]
    # Ordinary H8/32/96 and remote H8/96 are paired regimes, not new histories.
    for fi in range(4):
        assert len({study.specification('n3', 'development', fi + offset)['seed'] for offset in (0, 4, 8)}) == 1
        assert study.specification('n3', 'development', fi+12)['seed'] == study.specification('n3', 'development', fi+16)['seed']


def test_training_endpoint_and_public_noise_budget_survive_driver_mapping():
    cases = [study.make_case(k, 101140004, 2) for k in ('ordinary96', 'train32', 'train8')]
    remote = [study.make_case(k, 101140005, 2) for k in ('remote8_H96', 'remote8_H32', 'remote8_H8')]
    for group in (cases, remote):
        whole = group[0]
        for part in group[1:]:
            for key in ('p', 'q', 'v', 'master_p', 'master_q', 'root_v'):
                np.testing.assert_array_equal(getattr(part['train'], key), getattr(whole['train'], key)[-part['train'].t:])
                np.testing.assert_array_equal(getattr(part['validation'], key), getattr(whole['validation'], key))
                np.testing.assert_array_equal(getattr(study.observe_future(part, np.zeros((part['config']['horizon'], 3))), key),
                    getattr(study.observe_future(whole, np.zeros((whole['config']['horizon'], 3))), key))
    for case in (cases[0], remote[0], study.make_case('twinblock12', 101140006, 2)):
        T, n = case['config']['horizon'], case['train'].n
        for key in ('p', 'q', 'v', 'master_p', 'master_q'):
            old_sigma = float(np.asarray(getattr(case['tm'], key+'_sigma')).ravel()[0])
            shape = (T, n) if key in ('p', 'q', 'v') else (T,)
            sigma = np.broadcast_to(getattr(case['fm'], key+'_sigma'), shape)
            np.testing.assert_allclose(np.sum((old_sigma/sigma)**2, axis=0), 30., atol=1e-12)


def _mock_full_pipeline(monkeypatch, case_id, *, only_true_fit=False):
    case = study.make_case(case_id, 101140007, 3)
    T, n = case['config']['horizon'], case['train'].n
    constant = np.zeros((T, n)); constant[:, 0] = 3. / (.25*T)
    mobile = np.zeros((T, n)); charging = 3.05 / .95 / ((T-2)*.25)
    mobile[:2, 0] = charging; mobile[4:, 2] = charging
    plans = {name: (np.zeros((T, n)) if name == 'none' else
        mobile.copy() if name in ('scalar_mobile', 'cap_mobile', 'jointgrad_mobile') else constant.copy())
        for name in study.METHODS}
    metadata = {name: dict(success=True, planning_actual_seconds=.01*(j+1),
        planning_seconds=.1*(j+1), dependencies=[]) for j, name in enumerate(study.METHODS)}
    compare_calls = []
    monkeypatch.setattr(study, 'make_case', lambda *args: case)
    monkeypatch.setattr(study.rx, 'train_bank', _trained)
    monkeypatch.setattr(study.design, 'select_plans', lambda *args, **kwargs:
        (plans, dict(methods=metadata, context=dict(pairs=[])), kwargs['domains']))

    def prefit(candidate, *args, **kwargs):
        return SimpleNamespace(fixed_pre=dict(loss=1.), joint_pre=dict(loss=.5, seconds=2.), seconds=5.)

    def compare(candidate, training, validation, future, tm, vm, fm, **kwargs):
        compare_calls.append(dict(candidate=candidate.name, p=future.p.copy(), q=future.q.copy(),
            training=training, validation=validation, tm=tm, vm=vm, fm=fm))
        truth_name = case['bank'][3].name
        if only_true_fit and candidate.name != truth_name:
            raise RuntimeError('deliberate missing competitor fit')
        j = next(i for i, k in enumerate(case['bank']) if k.name == candidate.name)
        return dict(fixed=dict(post=dict(success=True), fixed_total=100.+j, fixed_holdout=50.+j),
            joint=dict(post=dict(success=True, state=dict(r=np.full(candidate.m, .003), x=np.full(candidate.m, .0008))),
                joint_total=80.+j), cost=dict(fixed_future_seconds=.4, joint_post_seconds=.7))

    monkeypatch.setattr(study.refit, 'prepare_refit_history', prefit)
    monkeypatch.setattr(study.refit, 'compare_fixed_and_joint', compare)
    return case, plans, metadata, compare_calls


@pytest.mark.parametrize('case_id', ['ordinary96', 'twinblock12'])
def test_all_eleven_methods_share_actual_data_across_three_modes_and_pay_dependencies(monkeypatch, case_id):
    case, plans, metadata, calls = _mock_full_pipeline(monkeypatch, case_id)
    record = study._run(_spec(case_id, 101140007, 3))
    assert record['status'] == 'complete'
    assert len(record['methods']) == len(study.METHODS) == 11
    assert record['unique_actual_curves'] == 3
    assert len(calls) == 3*len(case['bank'])
    assert len(record['future_records']) == 3
    assert record['candidate_miss'] is False
    assert [tuple(map(tuple, row['supports'])) for row in record['candidates']] == [k.supports for k in case['bank']]
    common = record['common_cost']
    assert common['fixed_history_seconds'] == 12.
    assert common['joint_history_seconds'] == 8.
    for method in record['methods']:
        name = method['name']
        np.testing.assert_array_equal(method['power_kw'], plans[name])
        assert method['planning_seconds'] == metadata[name]['planning_seconds']
        assert method['operation']['success'] is True
        future = record['future_records'][method['action_key']]
        np.testing.assert_array_equal(future['power_kw'], plans[name])
        direct = study.observe_future(case, plans[name])
        np.testing.assert_array_equal(future['observations']['p'], direct.p)
        np.testing.assert_array_equal(future['observations']['v'], direct.v)
        if name == 'none':
            assert method['operation']['battery_task_applicable'] is False
        else:
            audit = method['operation']['audit']
            assert audit['feasible']
            assert audit['energy_kwh'][-1] == pytest.approx(26.85)
            assert method['operation']['driver_cost'] == pytest.approx(audit['driver_cost'])
            if name in ('scalar_mobile', 'cap_mobile', 'jointgrad_mobile'):
                assert audit['travel_energy_kwh'] == pytest.approx(.2)
                assert audit['driver_cost'] == pytest.approx(30.)
                assert audit['moves'] == 1
        for mode, row in method['inference'].items():
            own = 4*(.4+.7) if mode == 'joint_total' else 4*.4
            expected = common['training_seconds'] + metadata[name]['planning_seconds'] + own + 12.
            if mode == 'joint_total':
                expected += 8.
            assert row['seconds'] == pytest.approx(own)
            assert row['total_seconds'] == pytest.approx(expected)
            assert row['complete']
            assert row['joint_pays_fixed_source_witness_dependencies'] is (mode == 'joint_total')
    for call in calls:
        assert call['training'] is case['train'] and call['validation'] is case['validation']
        assert call['tm'] is case['tm'] and call['vm'] is case['vm'] and call['fm'] is case['fm']


def test_missing_competitors_remain_partial_failures_with_undefined_margin(monkeypatch):
    _, _, _, _ = _mock_full_pipeline(monkeypatch, 'ordinary96', only_true_fit=True)
    record = study._run(_spec('ordinary96', 101140007, 3))
    assert record['status'] == 'partial_failure'
    for method in record['methods']:
        assert all(not row['complete'] and row['M'] is None for row in method['inference'].values())
    assert all(len(row['failures']) == 3 for row in record['future_records'].values())


def test_failure_slots_keep_all_methods_modes_and_margin_guard():
    rows = study.failed_methods('deliberate failure')
    assert [row['name'] for row in rows] == list(study.METHODS)
    assert all(set(row['inference']) == set(study.MODES) for row in rows)
    assert all(not mode['complete'] for row in rows for mode in row['inference'].values())
    assert study._margin({'a': 1.}, 'a') is None
    assert study._margin({'b': 1.}, 'a') is None
    assert study._margin({'a': 3., 'b': 1.}, 'a') == -2.


def test_source_hashes_bind_real_mobile_case_design_driver_and_unchanged_old_closure():
    current = study.source_hashes()
    old = study.old.source_hashes()
    assert len(old) == 110
    assert all(current[name] == value for name, value in old.items())
    for name in ('ev_mobile_v2_study.py', 'ev_mobile_v2_design.py', 'ev_mobile_cases_v2.py',
                 'ev_mobile_v2_protocol.md', 'ev_mobile_v2_math.md'):
        assert current[name] == sha256((study.HERE/name).read_bytes()).hexdigest()
