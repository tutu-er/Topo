"""Pure open-node adapter checks: fake AC/meter calls only, no graph creation."""
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from ac_bridge import execute_policy


def _fixture(*, converged=True, voltage_ok=True, numerical_nan=False):
    labels = [101, 102, 103, 104]
    index = pd.RangeIndex(96)
    background = SimpleNamespace(
        p_base=pd.DataFrame(.01, index=index, columns=labels),
        q_base=pd.DataFrame(.0035, index=index, columns=labels),
        root_true=pd.Series(1.02, index=index), scenario_index=1,
        replicate=950000, profile='cloudy_variable_pv', shifted_seed=7+100003*950000)
    calls = dict(ac=[], meter=[])

    def ac(net, p, q, **kwargs):
        calls['ac'].append((p.copy(), q.copy(), kwargs))
        ok = np.ones(96, bool)
        if not converged:
            ok[0] = False
        voltage = pd.DataFrame(1.01 if voltage_ok else 1.12, index=index, columns=labels)
        if numerical_nan:
            voltage.iloc[0, 0] = np.nan
        return dict(V_bus_mag=voltage, converged=pd.Series(ok, index=index))

    def observe(context, base, p, q, v, **kwargs):
        calls['meter'].append(kwargs)
        return dict(p_measured=p+.001, q_measured=q-.002,
                    voltage_measured=v+.003, root_observed=base.root_true+.004)

    def checker(*args):
        return dict(electrical_equations_pass=not numerical_nan, voltage_limits_pass=voltage_ok,
                    voltage_violation_cells=0 if voltage_ok else 96*len(labels),
                    max_nodal_p_residual_pu=np.nan if numerical_nan else 0.,
                    max_nodal_q_residual_pu=0., max_distflow_identity_error_pu2=0.)

    def scene(p, q, v, root, labels, name, day_offset):
        return dict(P_terminal=np.array(p), Q_terminal=np.array(q), V_terminal=np.array(v),
                    V_root_observed=np.array(root), labels=labels, name=name, day_offset=day_offset)

    runtime = dict(sim=SimpleNamespace(solve_ac_power_flow_timeseries=ac,
                                      _observe_terminal_channels=observe),
                   context=SimpleNamespace(net=object()), checker=checker,
                   legacy=SimpleNamespace(scene=scene), metadata=dict(base_kw=200.),
                   labels=labels, case='paper15', voltage_sigma=.0005, pq_sigma=.005)
    return runtime, background, calls


def test_three_resources_map_in_given_order_and_share_scaled_capacity_contract():
    runtime, base, calls = _fixture()
    before = base.p_base.copy()
    allocation = np.array([.5, 0., 1.5])
    u = np.zeros((3, 96)); u[0, 0] = -3.; u[2] = .3
    result = execute_policy(runtime, base, u, np.zeros_like(u),
                            resource_labels=[103, 101, 104], capacity_kw=allocation)
    assert len(calls['ac']) == 1
    assert result['summary']['status'] == 'complete'
    assert result['summary']['resource_count'] == 3
    assert result['summary']['total_capacity_kw'] == 2.
    assert result['summary']['allocated_zero_resources'] == 1
    assert result['summary']['resource_violation_cells'] == 1
    assert result['summary']['component_min_kw'] == -2.5
    assert result['summary']['terminal_negative_cells'] == 1
    assert result['summary']['planned_cost'] == pytest.approx(.25*(9/.25+96*.09/.75))
    assert np.all(result['private']['prices'][1] == 0.)
    p, q, options = calls['ac'][0]
    assert p.loc[0, 103] == pytest.approx(.01-3/200)
    assert p.loc[0, 104] == pytest.approx(.01+.3/200)
    assert np.all(p[101] == .01)
    assert q.loc[0, 103] == pytest.approx(.0035-.35*3/200)
    assert options['max_iter'] == 100 and options['tol'] == 1e-10
    pd.testing.assert_frame_equal(base.p_base, before)


def test_four_noise_versions_reuse_one_ac_and_meter_call_with_four_resources():
    runtime, base, calls = _fixture()
    with np.errstate(all='raise'):
        result = execute_policy(runtime, base, np.zeros((4, 96)), np.zeros((4, 96)),
            resource_labels=[101, 102, 103, 104], capacity_kw=[.25, .75, 0., 1.], day_offset=21)
    assert len(calls['ac']) == len(calls['meter']) == 1
    assert result['summary']['planned_cost'] == 0.
    s = result['scenes']; true_p = base.p_base.to_numpy()
    assert np.array_equal(s['none']['P_terminal'], true_p)
    assert np.array_equal(s['v']['P_terminal'], true_p)
    assert np.array_equal(s['pq']['P_terminal'], s['both']['P_terminal'])
    assert np.array_equal(s['pq']['P_terminal'], true_p+.001)
    assert np.array_equal(s['none']['V_terminal'], s['pq']['V_terminal'])
    assert np.array_equal(s['v']['V_terminal'], s['both']['V_terminal'])
    assert np.array_equal(s['none']['V_root_observed'], base.root_true.to_numpy())
    assert np.array_equal(s['pq']['V_root_observed'], base.root_true.to_numpy())
    assert np.array_equal(s['v']['V_root_observed'], s['both']['V_root_observed'])
    assert all(v['day_offset'] == 21 for v in s.values())
    assert calls['meter'][0] == dict(pq_noise_rel=.005, v_noise_rel=.0005)


def test_single_resource_misspecified_multiplier_changes_delivery_but_not_fee():
    runtime, base, calls = _fixture()
    u = np.full((1, 96), .2); Z = np.ones_like(u)
    regular = execute_policy(runtime, base, u, Z, resource_labels=[102], capacity_kw=[.5], cv=.1)
    misspec = execute_policy(runtime, base, u, Z, resource_labels=[102], capacity_kw=[.5],
                             cv=.1, multiplier_mean=.9)
    assert np.allclose(regular['private']['delivered_kw'], .22)
    assert np.allclose(misspec['private']['delivered_kw'], .2)
    assert regular['summary']['planned_cost'] == misspec['summary']['planned_cost']
    assert misspec['summary']['planned_cost'] == pytest.approx(.25*96*.04/.25)
    assert len(calls['ac']) == 2


def test_common_signed_noise_preserves_opposite_action_cancellation():
    runtime, base, calls = _fixture()
    u = np.stack([np.full(96, .4), np.full(96, -.4)])
    Z = np.stack([np.linspace(-3., 3., 96)]*2)
    result = execute_policy(runtime, base, u, Z, resource_labels=[103, 101],
                            capacity_kw=[1., 1.], cv=np.array([.2, .2]))
    expected = u*(1+.2*Z)
    assert np.array_equal(result['private']['delivered_kw'], expected)
    assert np.array_equal(expected.sum(axis=0), np.zeros(96))
    assert len(calls['ac']) == 1


def test_zero_capacity_and_command_cost_zero_but_nonzero_command_is_invalid():
    runtime, base, calls = _fixture()
    result = execute_policy(runtime, base, np.zeros((1, 96)), np.full((1, 96), 100.),
                            resource_labels=[103], capacity_kw=[0.])
    assert result['summary']['planned_cost'] == 0.
    assert not result['summary']['resource_violation']
    assert np.all(result['private']['prices'] == 0.)
    assert np.all(result['private']['delivered_kw'] == 0.)
    with pytest.raises(ValueError, match='zero allocated capacity'):
        execute_policy(runtime, base, np.ones((1, 96)), np.zeros((1, 96)),
                       resource_labels=[103], capacity_kw=[0.])
    assert len(calls['ac']) == 1


def test_small_capacity_realized_excursion_reaches_ac_without_clipping():
    runtime, base, calls = _fixture()
    result = execute_policy(runtime, base, np.full((1, 96), .25), np.ones((1, 96)),
                            resource_labels=[103], capacity_kw=[.25], cv=.2)
    assert result['summary']['resource_violation_cells'] == 96
    assert result['summary']['component_max_kw'] == pytest.approx(.55)
    assert np.allclose(result['private']['delivered_kw'], .3)
    assert len(calls['ac']) == 1 and result['summary']['status'] == 'complete'


def test_nonconverged_nan_diagnostics_preserve_private_and_json_safe_receipt():
    runtime, base, calls = _fixture(converged=False, numerical_nan=True)
    result = execute_policy(runtime, base, np.zeros((1, 96)), np.zeros((1, 96)),
                            resource_labels=[103], capacity_kw=[1.])
    assert result['summary']['status'] == 'physical_unresolved'
    assert result['summary']['ac_converged_samples'] == 95
    assert result['summary']['max_nodal_p_residual_pu'] is None
    assert np.isnan(result['private']['AC']['V_bus_mag'].iloc[0, 0])
    assert all(v is None for v in result['scenes'].values())
    json.dumps(result['summary'], allow_nan=False)
    assert len(calls['ac']) == 1 and calls['meter'] == []


def test_voltage_violation_stays_observed_and_invalid_controls_do_not_call_ac():
    runtime, base, calls = _fixture(voltage_ok=False)
    result = execute_policy(runtime, base, np.zeros((1, 96)), np.zeros((1, 96)),
                            resource_labels=[103], capacity_kw=[1.])
    assert result['summary']['status'] == 'complete'
    assert not result['summary']['voltage_limits_pass']
    assert not result['summary']['all_operation_checks_pass']
    assert all(s is not None for s in result['scenes'].values())
    bad = np.zeros((1, 96)); bad[0, 0] = np.nan
    with pytest.raises(ValueError):
        execute_policy(runtime, base, bad, np.zeros_like(bad), resource_labels=[103], capacity_kw=[1.])
    assert len(calls['ac']) == 1
