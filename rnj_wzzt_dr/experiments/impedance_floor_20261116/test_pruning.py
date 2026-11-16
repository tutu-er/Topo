"""Small fixed-support tests; no topology search, physical AC or grid batch."""
from copy import deepcopy
import json
from types import SimpleNamespace

import numpy as np
import pytest

from ...observations import Observation
from ...identification.optimization import QPFitter, response_matrices
from . import pruning


def training(n=2):
    # Full-column-rank, deterministic P/Q design separates both response blocks.
    inputs = np.eye(2*n)
    return Observation(inputs[:, :n], inputs[:, n:], np.zeros((2*n, n)),
                       tuple(101+i for i in range(n)))


def point(supports, r, x=None):
    return dict(supports=deepcopy(supports), r=list(r), x=list(np.zeros(len(r)) if x is None else x))


def fitter_for(p):
    obs = training(max(max(C) for C in p['supports'])+1)
    R, X = response_matrices(p, obs.n)
    return QPFitter(Observation(obs.p, obs.q, obs.p@R.T+obs.q@X.T, obs.labels), reference_mse=.1)


def simple_calibration(n=2):
    return dict(a=1., b=1., median_root_depth=1., terminal_count=n)


def test_zero_threshold_preserves_exact_point_even_zero_supports_and_metadata():
    p = point([[0, 1], [1], [0]], [0., .1, 0.], [0., 0., .2])
    p['source_metadata'] = {'exact_order': [3, 1], 'do_not_rewrite': 'native'}
    encoded = json.dumps(p, separators=(',', ':'))
    fit = fitter_for(p)
    result, receipt = pruning.prune(p, fit, 0.)
    assert result is p
    assert json.dumps(result, separators=(',', ':')) == encoded
    assert fit.records == [] and fit.cache == {}
    assert receipt['baseline_identity_preserved'] and receipt['deletions'] == 0


def test_calibration_matches_independent_distance_formula_and_channel_units_cancel():
    R = np.array([[.4, .1], [.1, .8]])
    X = np.array([[.3, .05], [.05, .5]])
    c = pruning.calibration_from_matrices(R, X)
    assert c['mu_r'] == pytest.approx(1.)
    assert c['mu_x'] == pytest.approx(.7)
    assert c['a'] == pytest.approx(.75)
    assert c['b'] == pytest.approx(.25/.7)
    assert c['median_root_depth'] == pytest.approx(.75*.6+.25*.4/.7)
    p = point([[0]], [.07], [.02])
    c2 = pruning.calibration_from_matrices(7*R, 3*X)
    transformed = point([[0]], [.49], [.06])
    np.testing.assert_allclose(pruning.strengths(p, calibration=c),
                               pruning.strengths(transformed, calibration=c2), rtol=1e-14)
    with pytest.raises(ValueError, match='median root depth'):
        pruning.calibration_from_matrices(np.zeros((2, 2)), np.zeros((2, 2)))


def test_training_calibration_calls_current_fit_once_and_cannot_move_between_inputs(monkeypatch):
    calls = []
    def fake_fit(train):
        calls.append(train)
        return np.eye(2), 2*np.eye(2), {'solver': 'test-only pre-tree fit'}
    monkeypatch.setattr(pruning.trees, 'fit_sensitivity', fake_fit)
    train = training()
    c = pruning.calibrate(train)
    assert calls == [train]
    assert c['labels'] == [101, 102] and c['training_rows'] == 4
    altered = Observation(train.p, train.q, train.y+.001, train.labels)
    with pytest.raises(ValueError, match='different training input'):
        pruning.prune(point([[0]], [.1]), QPFitter(altered), .2, calibration=c)
    assert len(calls) == 1


def test_physical_impedance_has_squared_voltage_factor_two():
    p = point([[0]], [.3], [.4])
    # r=.3,x=.4 are 2*per-unit impedance: |Z|=4/2*.5=1 ohm.
    assert pruning.strengths(p, metric='ohm', zbase_ohm=4.)[0] == pytest.approx(1.)
    with pytest.raises(ValueError, match='Zbase'):
        pruning.strengths(p, metric='ohm', zbase_ohm=0.)


def test_positive_floor_can_remove_singletons_and_empty_family_keeps_all_measurements():
    p = point([[0], [1], [0, 1]], [.03, .04, .01], [.02, .01, .005])
    fit = fitter_for(p)
    result, receipt = pruning.prune(p, fit, 5., metric='ohm', zbase_ohm=2.)
    assert result['supports'] == [] and result['r'] == [] and result['x'] == []
    assert result['mse'] == pytest.approx(float(np.mean(fit.data.y**2)))
    assert result['objective'] == pytest.approx(result['mse']/.1)
    assert receipt['observed_labels'] == [101, 102]
    assert sorted(v['label'] for v in receipt['removed_singletons']) == [101, 102]
    assert receipt['removed_common'] == [[0, 1]]
    assert receipt['deletions'] == 3 and receipt['checks'] == 4
    empty_record = receipt['steps'][-1]['fit']['record']
    assert empty_record['solver'] == 'closed_form_empty_support'
    assert empty_record['certificate_gap'] == 0.
    assert empty_record['lower_bound'] == result['objective']
    assert all(e['support_count_after'] == e['support_count_before']-1 for e in receipt['steps'])
    assert receipt['minimum_final_strength'] is None
    json.dumps(receipt, allow_nan=False)


def test_non_singleton_scope_protects_zero_leaves_but_removes_common_and_reuses_cache():
    p = point([[0], [1], [0, 1]], [0., 0., .01])
    fit = fitter_for(p)
    result, receipt = pruning.prune(p, fit, .1, scope='non_singleton', calibration=simple_calibration())
    assert result['supports'] == [[0], [1]]
    assert receipt['removed_singletons'] == [] and receipt['removed_common'] == [[0, 1]]
    assert all(not v['eligible'] for v in receipt['final_strengths'])
    count = len(fit.records)
    again, repeated = pruning.prune(p, fit, .1, scope='non_singleton', calibration=simple_calibration())
    assert again is result
    assert len(fit.records) == count and repeated['refits'] == 0
    assert repeated['steps'][0]['fit']['cache_hit']
    assert repeated['steps'][0]['fit']['certificate_gap'] is not None


def test_final_coefficients_rechecked_after_refit_and_exact_tie_is_canonical():
    # A strong common edge becomes weak after the first deletion. A single-pass
    # filter would retain it, so the final threshold check must remove it too.
    p = point([[1], [0], [0, 1], [0, 1, 2]], [.9, .9, .1, .7])
    fit = SimpleNamespace(data=training(3), reference_mse=1., bound=2., cache={}, records=[])
    def refit(family):
        result = point([list(C) for C in family], [.05 if len(C) == 3 else .9 for C in family])
        fit.cache[family] = result
        fit.records.append(dict(family=family, certificate_gap=0.))
        return result
    fit.fit = refit
    result, receipt = pruning.prune(p, fit, .2, calibration=simple_calibration(3))
    assert [s['removed_support'] for s in receipt['steps']] == [[0, 1], [0, 1, 2]]
    assert result['supports'] == [[0], [1]]
    assert receipt['minimum_final_strength'] == pytest.approx(.9)
    # Equal smallest strengths: canonical support ordering, not input ordering.
    tied = point([[1], [0], [0, 1]], [.01, .01, .2])
    _, trace = pruning.prune(tied, fitter_for(tied), .05, calibration=simple_calibration())
    assert trace['steps'][0]['removed_support'] == [0]


def test_boundary_tolerance_scales_with_threshold_and_does_not_clip_weights():
    f = .01
    p = point([[0]], [f*(1-.5e-10)])
    out, receipt = pruning.prune(p, fitter_for(p), f, metric='ohm', zbase_ohm=2.)
    assert out is p and receipt['deletions'] == 0
    assert receipt['numerical_absolute_tolerance'] == pytest.approx(1e-12)
    p2 = point([[0]], [f*(1-2e-10)])
    out2, receipt2 = pruning.prune(p2, fitter_for(p2), f, metric='ohm', zbase_ohm=2.)
    assert out2['supports'] == [] and receipt2['deletions'] == 1


def test_failure_preserves_trace_and_does_not_return_a_false_success():
    p = point([[0], [1]], [.1, .5])
    fit = fitter_for(p)
    def fail(family):
        raise TimeoutError('test soft-QP budget exceeded')
    fit.fit = fail
    with pytest.raises(pruning.PruningError) as captured:
        pruning.prune(p, fit, .2, calibration=simple_calibration())
    exc = captured.value
    assert exc.last_point is p
    assert exc.receipt['status'] == 'failed' and not exc.receipt['floor_satisfied']
    assert exc.receipt['steps'][0]['removed_support'] == [0]
    assert exc.receipt['error_type'] == 'TimeoutError'


def test_actual_qp_refit_loss_and_certificate_match_direct_prediction():
    p = point([[0], [1], [0, 1]], [.04, .3, .01], [.03, .2, .005])
    fit = fitter_for(p)
    out, receipt = pruning.prune(p, fit, .06, metric='ohm', zbase_ohm=2.)
    R, X = response_matrices(out, 2)
    residual = fit.data.y-fit.data.p@R.T-fit.data.q@X.T
    assert receipt['final_training_loss']['mse'] == pytest.approx(np.mean(residual**2))
    assert receipt['final_training_loss']['mse'] >= receipt['initial_training_loss']['mse']
    assert receipt['minimum_final_strength'] >= .06-6e-12
    assert all(s['fit']['certificate_gap'] <= 1e-6 for s in receipt['steps'])
    assert len(fit.data.labels) == 2  # A missing singleton is not a missing meter.
