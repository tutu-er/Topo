"""Public-case, physical-unit and selector-isolation checks; no formal study."""
from __future__ import annotations

from copy import deepcopy
import inspect
from types import SimpleNamespace

import numpy as np
import pytest

from . import ev_mobile_cases_v2 as cases
from . import ev_systematic_cases as old
from . import ev_fixed_train_design as fixed
from . import likelihood as lk


@pytest.mark.parametrize('case_id', cases.CASES)
@pytest.mark.parametrize('fi', range(4))
def test_all_declared_truths_in_same_public_bank_with_original_asset_domain(case_id, fi):
    case = cases.make_case(case_id, 100711001, fi)
    cfg = case['config']
    assert case['train'].t == cfg['training_points'] == 96
    assert case['validation'].t == 4
    assert len(case['bank']) == 4
    assert len({k.name for k in case['bank']}) == 4
    assert sum(k.supports == case['truth']['candidate'].supports for k in case['bank']) == 1
    old_cfg = old.case_config('ordinary96')
    for field in ('r_bounds', 'x_bounds'):
        assert cfg['training'][field] == old_cfg['training'][field]
    assert cfg['ratio_bounds'] == old_cfg['ratio_bounds']
    for key in ('r', 'x'):
        low, high = cfg['training'][key + '_bounds']
        assert np.min(case['truth'][key]) >= low
        assert np.max(case['truth'][key]) <= high
    lo, hi = cfg['ratio_bounds']
    ratio = case['truth']['x'] / case['truth']['r']
    assert np.min(ratio) >= lo - 1e-15
    assert np.max(ratio) <= hi + 1e-15
    assert cfg['source_interface'] == case['provenance']['source_interface'] == 1
    assert case['provenance']['field_PQV_available'] is False
    assert case['truth_used_only_for_generation_and_evaluation'] is True
    T, n = cfg['horizon'], case['train'].n
    future = cases.observe_future(case, np.zeros((T, n)))
    assert future.t == T and future.n == n
    assert np.isfinite(future.v).all()
    assert np.min(future.v) > .9 and np.max(future.v) < 1.1


def test_twin_bank_is_two_declared_disjoint_blocks_and_never_truth_filtered():
    bank = cases.candidate_bank('twinblock12')
    assert [k.name for k in bank] == ['star', 'clade_01', 'clade_23', 'clade_01_23']
    families = [tuple(s for s in k.supports if 1 < len(s) < k.n) for k in bank]
    assert families == list(cases.FAMILIES['twinblock12'])
    for fi in range(4):
        case = cases.make_case('twinblock12', 100711002, fi)
        assert [(k.name, k.supports) for k in case['bank']] == [(k.name, k.supports) for k in bank]


def test_training_subsets_share_assets_endpoint_future_and_noise():
    paths = [cases.make_case('remote8', 100711003, 3, training_points=h) for h in (96, 32, 8)]
    whole = paths[0]
    for part in paths[1:]:
        H = part['train'].t
        for key in ('p', 'q', 'v', 'master_p', 'master_q'):
            np.testing.assert_array_equal(getattr(part['train'], key), getattr(whole['train'], key)[-H:])
            np.testing.assert_array_equal(getattr(part['validation'], key), getattr(whole['validation'], key))
            np.testing.assert_array_equal(part['white'][key], whole['white'][key])
        for key in ('p', 'q', 'source_p', 'source_q', 'root_v'):
            np.testing.assert_array_equal(part[key], whole[key])
        for key in ('r', 'x'):
            np.testing.assert_array_equal(part['truth'][key], whole['truth'][key])


@pytest.mark.parametrize('fi', range(4))
def test_remote_case_preserves_old_past_assets_and_future_background_prefix(fi):
    new = cases.make_case('remote8', 100711004, fi)
    previous = old.make_case('ordinary96', 100711004, fi)
    for key in ('p', 'q', 'v', 'master_p', 'master_q'):
        np.testing.assert_array_equal(getattr(new['train'], key), getattr(previous['train'], key))
        np.testing.assert_array_equal(getattr(new['validation'], key), getattr(previous['validation'], key))
        np.testing.assert_array_equal(new['white'][key], previous['white'][key])
    for key in ('p', 'q', 'source_p', 'source_q', 'root_v'):
        np.testing.assert_array_equal(new[key], previous[key])
    for key in ('r', 'x'):
        np.testing.assert_array_equal(new['truth'][key], previous['truth'][key])


@pytest.mark.parametrize('case_id', cases.CASES)
def test_uniform_precision_has_same_total_integration_budget(case_id):
    case = cases.make_case(case_id, 100711005, 0)
    T = case['config']['horizon']
    assert sum(case['config']['public_voltage_repeat_counts']) == 30.
    assert case['config']['future_metering_integration_seconds_total'] == 3000.
    seconds = case['config']['future_metering_integration_seconds_per_slot']
    assert seconds * T == 3000. and seconds <= 900. and float(seconds).is_integer()
    for channel in ('p', 'q', 'v', 'master_p', 'master_q'):
        old_sigma = np.asarray(getattr(case['tm'], channel + '_sigma'))
        new_sigma = np.asarray(getattr(case['fm'], channel + '_sigma'))
        assert np.max(new_sigma) == pytest.approx(float(np.max(old_sigma)) * np.sqrt(T / 30.))
        assert np.min(new_sigma) == pytest.approx(np.max(new_sigma))
    assert case['provenance']['metering']['weights_are_integer_sample_counts'] is False


@pytest.mark.parametrize('case_id', cases.CASES)
def test_fixed_parent_and_costed_one_move_plan_are_both_executable(case_id):
    T, n = cases.DIMENSIONS[case_id]
    parent = np.zeros((T, n))
    parent[:, 0] = 3. / (.25 * T)
    fixed_domain = cases.service_domain(case_id, T, n, mobile=False)
    mobile_domain = cases.service_domain(case_id, T, n, mobile=True)
    assert fixed._realize(fixed_domain, parent)['success']
    assert fixed._realize(mobile_domain, parent)['success']
    assert mobile_domain.config.initial_site == fixed_domain.config.initial_site == 0
    assert mobile_domain.config.max_moves == 1
    assert np.all(mobile_domain.config.arrays()['access'])
    # Two initial connected slots, driving slot 2, setup slot 3, remote slots 4:T.
    power = np.zeros((T, n))
    charge = (2.85 + .2) / .95 / ((T - 2) * .25)
    power[:2, 0] = charge
    power[4:, 2] = charge
    realised = fixed._realize(mobile_domain, power)
    assert realised['success'], realised
    audit = mobile_domain.check(realised['x'])
    assert audit['feasible']
    assert audit['moves'] == 1
    assert audit['driver_cost'] == pytest.approx(30.)
    assert audit['travel_energy_kwh'] == pytest.approx(.2)
    assert audit['energy_kwh'][-1] == pytest.approx(26.85)
    assert audit['grid_energy_kwh'] == pytest.approx(3.05 / .95)
    assert audit['throughput_kwh'] < 7.5
    assert not fixed._realize(fixed_domain, power)['success']


@pytest.mark.parametrize('case_id', cases.CASES)
def test_ev_enters_total_p_once_and_uses_same_noise_for_all_actions(case_id):
    case = cases.make_case(case_id, 100711006, 3)
    T, n = cases.DIMENSIONS[case_id]
    zero = np.zeros((T, n))
    power = zero.copy()
    power[:, 2] = np.linspace(-1., 2., T)
    clean = cases.observe_future(case, power, clean=True)
    noisy = cases.observe_future(case, power)
    empty = cases.observe_future(case, zero)
    rows = slice(case['future_start'], case['future_start'] + T)
    np.testing.assert_allclose(clean.p - case['fm'].bias_p, case['p'][rows] + power / 20., rtol=0, atol=3e-17)
    np.testing.assert_allclose(clean.q - case['fm'].bias_q, case['q'][rows], rtol=0, atol=3e-17)
    np.testing.assert_allclose(noisy.p - empty.p, power / 20., rtol=0, atol=1e-16)
    assert case['truth']['candidate'].n == n
    predicted = lk.ac_predict(case['truth']['candidate'], case['truth']['r'], case['truth']['x'],
        case['p'][rows] + power / 20., case['q'][rows], case['source_p'][rows],
        case['source_q'][rows], case['root_v'][rows])
    np.testing.assert_allclose(clean.v - case['fm'].bias_v, predicted['v'], rtol=0, atol=3e-17)
    np.testing.assert_array_equal(predicted['loads'][:, 0], case['source_p'][rows] + 1j * case['source_q'][rows])


def test_forecast_and_operation_api_exclude_true_and_future_inputs():
    assert tuple(inspect.signature(cases.forecast_for).parameters) == ('tr', 'tm', 'fits', 'cfg')
    assert tuple(inspect.signature(cases.service_domain).parameters) == ('case_id', 'T', 'n', 'mobile')
    case = cases.make_case('remote8', 100711007, 2, training_points=8)
    fits = {k.name: SimpleNamespace(source_p=np.zeros(8), source_q=np.zeros(8), success=True)
            for k in case['bank']}
    forecast = cases.forecast_for(case['train'], case['tm'], fits, case['config'])
    case['p'][100:] = 999.
    case['q'][100:] = 999.
    case['source_p'][100:] = 999.
    case['white']['v'][100:] = 999.
    case['truth']['r'][:] = 999.
    second = cases.forecast_for(case['train'], case['tm'], fits, case['config'])
    for key in ('p', 'q', 'root'):
        np.testing.assert_array_equal(forecast[key], second[key])
    assert forecast['metadata']['uses_truth'] is False
    assert forecast['metadata']['uses_future_noise'] is False
    assert forecast['metadata']['uses_validation'] is False
    assert forecast['metadata']['training_only'] is True


def test_remote_block_has_no_first_order_response_to_start_site_injection():
    """Audit the physical reason for twinblock12 independently of its truth draws.

    At zero base power, root and terminal impedances are identical in the two
    models. The only extra line is the (2,3) internal line. Injecting at 0 does
    not send current through it; injecting at 2 moves both 2 and 3 voltages.
    This local identity motivates the case, not a guarantee after refitting.
    """
    bank = cases.candidate_bank('twinblock12')
    star, cherry = bank[0], bank[2]
    root_r, leaf_r, internal_r = .00104, .008025, .00208
    root_x, leaf_x, internal_x = .0004, .0010375, .0008

    def values(candidate, j, epsilon):
        r = [root_r if len(s) == 4 else leaf_r if len(s) == 1 else internal_r
             for s in candidate.supports]
        x = [root_x if len(s) == 4 else leaf_x if len(s) == 1 else internal_x
             for s in candidate.supports]
        p, q = np.zeros((1, 4)), np.zeros((1, 4))
        p[0, j] = epsilon
        return lk.ac_predict(candidate, r, x, p, q)['v'][0]

    epsilon = 1e-5
    np.testing.assert_allclose(values(star, 0, epsilon), values(cherry, 0, epsilon),
        rtol=0., atol=1e-14)
    derivative_difference = (values(cherry, 2, epsilon) - values(star, 2, epsilon)) / epsilon
    np.testing.assert_allclose(derivative_difference, [0., 0., -internal_r, -internal_r],
        rtol=0., atol=2e-8)


def test_configs_and_case_bank_validation_preserve_frozen_defaults():
    before = deepcopy(old.case_config('ordinary96'))
    modified = cases.case_config('remote8', training_points=32)
    modified['training']['r_bounds'] = [0., 0.]
    assert old.case_config('ordinary96') == before
    assert cases.case_config('remote8', training_points=32)['training']['r_bounds'] == before['training']['r_bounds']
    for unknown in ('not_a_case', 'ordinary96'):
        with pytest.raises(ValueError):
            cases.candidate_bank(unknown)
    for wrong_h in (7, True, 12):
        with pytest.raises(ValueError):
            cases.case_config('remote8', training_points=wrong_h)
    with pytest.raises(ValueError):
        cases.case_config('twinblock12', training_points=32)
    for bad_fi in (-1, 4, True):
        with pytest.raises(ValueError):
            cases.make_case('remote8', 100711008, bad_fi)
    with pytest.raises(ValueError):
        cases.service_domain('remote8', 12, 3, mobile=True)
    with pytest.raises(ValueError):
        cases.service_domain('remote8', 8, 4, mobile=True)
    with pytest.raises(ValueError):
        cases.observe_future(cases.make_case('remote8', 100711008, 0), np.zeros((8, 4)))
