"""Case coverage, physical scale and observed-information contracts."""
from collections import Counter
from copy import deepcopy
from dataclasses import fields
import json

import numpy as np
import pytest

from . import ev_bayes_response_cases as cases
from . import ev_mobile_cases_v2 as original
from . import likelihood as lk
from .simulation import model


def test_exact_matrix_independent_seeds_and_predeclared_legacy_subset():
    phases = {p: cases.specifications(p) for p in cases.SEED_BASES}
    assert {p: len(v) for p, v in phases.items()} == dict(development=16, calibration=64, evaluation=128)
    all_seeds = [r['seed'] for rows in phases.values() for r in rows]
    assert len(set(all_seeds)) == len(all_seeds)
    assert min(all_seeds) > 2026102707
    for phase, repetitions in (('calibration', 1), ('evaluation', 2)):
        rows = phases[phase]
        assert Counter(r['stratum_id'] for r in rows) == Counter({i: repetitions for i in range(64)})
        assert all(r['innovation_multiplier'] == cases.INNOVATIONS[r['innovation']] for r in rows)
    legacy = [r for r in phases['evaluation'] if r['legacy_audit']]
    assert len(legacy) == 16
    assert len({(r['bank'], r['family_index'], r['H']) for r in legacy}) == 16
    assert {r['bank'] for r in phases['development'][:8]} == {'n3', 'n4'}
    assert {r['H'] for r in phases['development'][:8]} == {32, 96}
    json.dumps(dict(config=cases.CASE_CONFIG, generator=cases.GENERATOR_CONFIG, strata=cases.STRATA, specs=phases), allow_nan=False)
    changed = cases.specifications('evaluation'); changed[0]['H'] = -1
    assert cases.specifications('evaluation')[0]['H'] == 32


@pytest.fixture(scope='module', params=[('n3', 32), ('n4', 96)])
def paired(request):
    bank, H = request.param
    spec = next(r for r in cases.specifications('development') if r['bank'] == bank and r['H'] == H)
    return tuple(cases.make_case(spec, regime) for regime in cases.REGIMES)


def test_complete_length_and_paired_regimes_identical_history_noise_assets(paired):
    regular, shift = paired; H = regular['spec']['H']; start = H+20; n = regular['train'].n
    assert regular['train'].t == H and regular['evidence'].t == 20
    assert regular['validation'] is regular['evidence'] and regular['vm'] is regular['em']
    assert regular['future_start'] == start and regular['p'].shape == (H+20+8, n)
    for field in ('p', 'q', 'source_p', 'source_q', 'root_v'):
        np.testing.assert_array_equal(regular[field][:start], shift[field][:start])
        if field != 'root_v':
            assert np.max(abs(regular[field][start:]-shift[field][start:])) > 0
    for section in ('train', 'evidence'):
        for field in fields(lk.Observations):
            a, b = getattr(regular[section], field.name), getattr(shift[section], field.name)
            if a is None:
                assert b is None
            else:
                np.testing.assert_array_equal(a, b)
    for channel in regular['white']:
        np.testing.assert_array_equal(regular['white'][channel], shift['white'][channel])
    for channel in ('r', 'x'):
        np.testing.assert_array_equal(regular['truth'][channel], shift['truth'][channel])


def test_physical_assets_original_domain_root_and_uniform_precision(paired):
    regular, _ = paired; n = regular['train'].n; meter = model(n)
    reference = original.case_config(regular['case_id'], training_points=96)
    for field in ('r_bounds', 'x_bounds'):
        assert regular['config']['training'][field] == reference['training'][field]
    assert regular['config']['ratio_bounds'] == reference['ratio_bounds']
    for channel in ('p', 'q', 'v', 'master_p', 'master_q'):
        actual = np.asarray(getattr(regular['fm'], channel+'_sigma'))
        target = np.asarray(getattr(meter, channel+'_sigma'))/np.sqrt(30/8)
        np.testing.assert_allclose(actual, np.broadcast_to(target, actual.shape))
    np.testing.assert_array_equal(regular['root_v'], 1.)
    assert len(regular['bank']) == 4 and all(c.n == n for c in regular['bank'])
    assert regular['provenance']['EV_included'] is False
    assert regular['provenance']['source_interface'] == 1
    truth = regular['truth']
    physical = lk.ac_predict(truth['candidate'], truth['r'], truth['x'], regular['p'], regular['q'],
        regular['source_p'], regular['source_q'], regular['root_v'])
    assert .85 < physical['v'].min() and physical['v'].max() < 1.15
    assert physical['power_balance_residual'] < 2e-10


def test_observed_only_input_view_excludes_future_truth_and_regime(paired):
    regular, shift = paired
    expected = {'training', 'evidence', 'train_meter', 'evidence_meter', 'future_meter', 'candidates', 'horizon', 'base_kw'}
    assert set(cases.planning_inputs(regular)) == expected
    poisoned = dict(regular, p=np.full_like(regular['p'], np.nan), truth=object(), regime='forbidden')
    a, b = cases.planning_inputs(regular), cases.planning_inputs(poisoned)
    assert a == b
    for channel in ('p', 'q', 'v', 'master_p', 'master_q'):
        np.testing.assert_array_equal(getattr(cases.planning_inputs(regular)['evidence'], channel),
                                      getattr(cases.planning_inputs(shift)['evidence'], channel))


@pytest.mark.parametrize('bank', cases.BANKS)
@pytest.mark.parametrize('family', range(4))
def test_all_predeclared_truths_have_complete_meters_without_truth_filtering(bank, family):
    spec = next(r for r in cases.specifications('development') if r['bank'] == bank and r['family_index'] == family)
    case = cases.make_case(spec)
    assert sum(c.supports == case['truth']['candidate'].supports for c in case['bank']) == 1
    assert len(case['bank']) == 4
    assert case['train'].source_p is None and case['evidence'].source_q is None


def test_ordinary_and_correlated_loads_retain_PF_variation_and_reasonable_scale():
    correlations = {}
    for excitation in cases.EXCITATIONS:
        values = cases.generate_trajectory(2026110907, 4, 96, excitation, 'triple')
        p, ratio = values['p_kw'], values['q_over_p']
        assert 1.5 < p.min() < p.max() < 3.5
        assert .2 <= ratio.min() <= ratio.max() <= .45
        assert np.all(p.std(axis=0) > .08) and np.all(ratio.std(axis=0) > .025)
        correlations[excitation] = np.corrcoef(p.T)[np.triu_indices(4, 1)].mean()
        assert np.max(abs(values['q_kvar']/p-ratio)) < 1e-15
    assert correlations['correlated'] > .9
    assert correlations['correlated'] > correlations['ordinary']+.2


def test_threefold_innovations_apply_to_training_evidence_and_future():
    a = cases.generate_trajectory(2026110908, 3, 96, innovation='nominal')
    b = cases.generate_trajectory(2026110908, 3, 96, innovation='triple')
    for section in (slice(0, 96), slice(96, 116), slice(116, 124)):
        for channel in ('p_kw', 'source_p_kw', 'source_q_kvar'):
            assert np.max(abs(a[channel][section]-b[channel][section])) > 0
    for key in ('p_kw', 'q_over_p', 'source_p_kw', 'source_q_kvar'):
        assert b['metadata']['stationary_innovation_scales'][key] == pytest.approx(3*a['metadata']['stationary_innovation_scales'][key])
    assert a['metadata']['child_seeds'] == b['metadata']['child_seeds']


@pytest.mark.parametrize('change', [dict(bank='n5'), dict(H=80), dict(family_index=4),
    dict(excitation='oracle'), dict(innovation='oracle'), dict(innovation_multiplier=9.), dict(seed=-1)])
def test_invalid_case_options_rejected(change):
    spec = deepcopy(cases.specifications('development')[0]); spec.update(change)
    with pytest.raises(ValueError):
        cases.make_case(spec)


def test_invalid_phase_rejected():
    with pytest.raises(ValueError):
        cases.specifications('previous_evaluation')
