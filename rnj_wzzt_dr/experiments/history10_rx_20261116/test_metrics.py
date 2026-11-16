"""Independent small algebra checks; no network, optimization or random experiment."""
import json
import numpy as np
import pytest

from .metrics import metrics, design_diagnostics, fixed_support_diagnostics


def test_small_prediction_error_can_hide_equal_opposite_rx_errors():
    p = np.array([[1., 2.], [2., 1.], [-1., 3.]])
    truth = np.c_[2*np.eye(2), np.eye(2)]
    fitted = np.c_[3*np.eye(2), np.zeros((2, 2))]
    result = metrics(fitted, truth, p, p)
    assert result['R']['relative_error'] == pytest.approx(.5)
    assert result['X']['relative_error'] == pytest.approx(1.)
    pred = result['prediction']
    assert pred['total_error_RMS'] == pytest.approx(0.)
    assert pred['R_error_RMS'] > 0
    assert pred['cross_R_X_MSE'] == pytest.approx(-2*pred['R_error_MSE'])
    assert pred['identity_absolute_error'] < 1e-14


def test_conditional_q_energy_uses_independent_part_not_q_rms():
    p = np.array([[1.], [-1.], [1.], [-1.]])
    orthogonal = np.array([[1.], [1.], [-1.], [-1.]])
    result = design_diagnostics(p, p+orthogonal)
    raw = result['uncentered']['Q_given_P']
    assert raw['block_energy'] == pytest.approx(8.)
    assert raw['residual_energy'] == pytest.approx(4.)
    assert raw['residual_energy_fraction'] == pytest.approx(.5)
    assert raw['projection_orthogonality_norm'] < 1e-13


def test_rank_deficiency_is_not_reported_as_finite_condition_number():
    p = np.array([[1.], [2.], [3.], [4.]])
    result = design_diagnostics(p, 3*p)
    geometry = result['uncentered']['design']
    for key in ('raw', 'column_normalized'):
        assert geometry[key]['rank'] == 1
        assert geometry[key]['nullity'] == 1
        assert geometry[key]['condition_number'] is None
    assert result['uncentered']['Q_given_P']['residual_energy_fraction'] < 1e-28
    assert result['uncentered']['Q_given_P']['residual_geometry']['rank'] == 0
    assert result['uncentered']['Q_given_P']['residual_geometry']['nullity'] == 1


def test_conditioning_uses_column_span_and_keeps_measurement_units():
    p = np.array([[1., 0.], [0., 1.], [1., -1.], [2., 1.]])
    q = np.array([[2., 1.], [1., -2.], [0., 3.], [4., -1.]])
    transform = np.array([[2., 1.], [0., 3.]])
    a = design_diagnostics(p, q)['uncentered']['Q_given_P']
    b = design_diagnostics(p@transform, q)['uncentered']['Q_given_P']
    c = design_diagnostics(10*p, 10*q)['uncentered']['Q_given_P']
    assert a['residual_energy'] == pytest.approx(b['residual_energy'])
    assert c['residual_energy'] == pytest.approx(100*a['residual_energy'])
    assert c['residual_energy_fraction'] == pytest.approx(a['residual_energy_fraction'])


def test_centered_view_does_not_silently_replace_no_intercept_qp():
    p = np.arange(1., 5.)[:, None]
    q = np.ones_like(p)
    result = design_diagnostics(p, q)
    assert result['uncentered']['Q_given_P']['residual_energy'] > .1
    assert result['centered']['Q_given_P']['block_energy'] == 0
    assert result['centered']['Q_given_P']['residual_energy_fraction'] is None


def test_resource_columns_follow_labels_and_do_not_count_devices_twice():
    p = np.ones((3, 2)); q = np.arange(6.).reshape(3, 2)
    reference = np.c_[np.eye(2), 2*np.eye(2)]
    h = reference.copy(); h[:, 2] += np.array([.2, 0.])
    result = metrics(h, reference, p, q, {'P': [304], 'Q': [303, 303]}, labels=[303, 304])
    assert result['resource_columns']['Q']['node_count'] == 1
    assert result['resource_columns']['Q']['response']['relative_error'] == pytest.approx(.1)
    assert result['resource_columns']['P']['node_labels'] == [304]
    assert result['columns'][0]['registered_Q_resource']
    assert not result['columns'][1]['registered_Q_resource']
    json.dumps(result, allow_nan=False)


def test_zero_reference_column_has_absolute_error_but_no_relative_percentage():
    p = np.ones((2, 1)); q = np.zeros_like(p)
    result = metrics(np.array([[1., .2]]), np.array([[1., 0.]]), p, q)
    assert result['X']['error_norm'] == pytest.approx(.2)
    assert result['X']['relative_error'] is None
    assert result['X']['angle_degrees'] is None
    json.dumps(result, allow_nan=False)


def test_resource_indices_and_labels_require_an_explicit_unambiguous_mapping():
    p = np.ones((2, 2)); h = np.c_[np.eye(2), np.eye(2)]
    a = metrics(h, h, p, p, {'Q': [0]}, labels=[303, 304], resource_node_mode='indices')
    b = metrics(h, h, p, p, {'Q': [303]}, labels=[303, 304])
    assert a['resource_columns'] == b['resource_columns']
    with pytest.raises(ValueError):
        metrics(h, h, p, p, {'Q': [303]}, resource_node_mode='labels')
    with pytest.raises(ValueError):
        metrics(h, h, p, p, {'Q': [False]}, resource_node_mode='indices')


def test_fixed_support_geometry_matches_hand_built_design_ordering():
    p = np.array([[1., 2.], [3., 4.], [2., -1.]])
    q = np.array([[2., -1.], [1., 3.], [-1., 2.]])
    # Terminal 0 and the common support: rows are t0/n0,t0/n1,... .
    fr = np.array([[1., 3.], [0., 3.], [3., 7.], [0., 7.], [2., 1.], [0., 1.]])
    fx = np.array([[2., 1.], [0., 1.], [1., 4.], [0., 4.], [-1., 1.], [0., 1.]])
    result = fixed_support_diagnostics(p, q, [[0], [0, 1]])
    expected = np.linalg.svd(np.c_[fr, fx], compute_uv=False)
    assert result['design']['raw']['singular_values'] == pytest.approx(expected)
    assert result['residual_cells'] == 6
    projection = fr@np.linalg.lstsq(fr, fx, rcond=None)[0]
    assert result['X_given_R']['residual_energy'] == pytest.approx(np.sum((fx-projection)**2))


def test_inputs_unchanged_and_invalid_geometry_fails_closed():
    p = np.array([[1., 2.], [3., 4.]])
    q = p.copy(); h = np.c_[np.eye(2), np.eye(2)]
    saved = [a.copy() for a in (p, q, h)]
    metrics(h, h, p, q)
    for a, b in zip((p, q, h), saved):
        assert np.array_equal(a, b)
    with pytest.raises(ValueError): metrics(h, h, p, q, {'Q': [999]})
    with pytest.raises(ValueError): metrics(h[:, :-1], h, p, q)
    with pytest.raises(ValueError): fixed_support_diagnostics(p, q, [[0], [0]])
    with pytest.raises(ValueError): fixed_support_diagnostics(p, q, [[True]])
