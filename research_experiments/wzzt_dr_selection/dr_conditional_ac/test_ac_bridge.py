"""Pure adapter checks using fake AC/meter functions, never a real feeder solve."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from ac_bridge import execute_policy


def _fixture(*, converged=True, voltage_ok=True):
    labels = [101, 102, 103]
    index = pd.RangeIndex(96)
    base = SimpleNamespace(p_base=pd.DataFrame(.01, index=index, columns=labels),
        q_base=pd.DataFrame(.0035, index=index, columns=labels),
        root_true=pd.Series(1.02, index=index), scenario_index=1,
        replicate=930000, profile='cloudy_variable_pv', shifted_seed=7+100003*930000)
    calls = dict(ac=[], meter=[])

    def ac(net, p, q, **kwargs):
        calls['ac'].append((p.copy(), q.copy(), kwargs))
        ok = np.ones(96, bool)
        if not converged:
            ok[0] = False
        return dict(V_bus_mag=pd.DataFrame(1.01 if voltage_ok else 1.12, index=index, columns=labels),
                    converged=pd.Series(ok, index=index))

    def observe(context, background, p, q, v, **kwargs):
        calls['meter'].append(kwargs)
        return dict(p_measured=p+.001, q_measured=q-.002,
                    voltage_measured=v+.003, root_observed=background.root_true+.004)

    def checker(*args):
        return dict(electrical_equations_pass=True, voltage_limits_pass=voltage_ok,
                    voltage_violation_cells=0 if voltage_ok else 96*3,
                    max_nodal_p_residual_pu=0., max_nodal_q_residual_pu=0.,
                    max_distflow_identity_error_pu2=0.)

    def scene(p, q, v, root, labels, name, day_offset):
        return dict(P_terminal=np.array(p), Q_terminal=np.array(q),
                    V_terminal=np.array(v), V_root_observed=np.array(root),
                    drop_target=np.asarray(root)[:, None]**2-np.asarray(v)**2,
                    labels=labels, name=name, day_offset=day_offset)

    runtime = dict(sim=SimpleNamespace(solve_ac_power_flow_timeseries=ac,
                                      _observe_terminal_channels=observe),
                   context=SimpleNamespace(net=object()), checker=checker,
                   legacy=SimpleNamespace(scene=scene), metadata=dict(base_kw=200.),
                   labels=labels, case='paper15', voltage_sigma=.0005, pq_sigma=.005)
    return runtime, base, calls


def test_violating_component_and_negative_terminal_still_reach_original_ac():
    runtime, base, calls = _fixture()
    original = base.p_base.copy()
    u = np.zeros((2, 96)); u[0, 0] = -3.
    result = execute_policy(runtime, base, u, np.zeros_like(u))
    assert len(calls['ac']) == 1
    assert result['summary']['status'] == 'complete'
    assert result['summary']['resource_violation']
    assert result['summary']['resource_violation_cells'] == 1
    assert result['summary']['component_min_kw'] == -2.
    assert result['summary']['terminal_negative_cells'] == 1
    assert result['private']['delivered_kw'][0, 0] == -3.
    p, q, options = calls['ac'][0]
    assert p.loc[0, 103] == pytest.approx(.01-3/200)
    assert q.loc[0, 103] == pytest.approx(.0035-.35*3/200)
    assert options['max_iter'] == 100 and options['tol'] == 1e-10
    pd.testing.assert_frame_equal(base.p_base, original)
    assert all(result['scenes'][key] is not None for key in ('none', 'v', 'pq', 'both'))


def test_four_noise_versions_share_one_ac_and_one_standard_meter_stream():
    runtime, base, calls = _fixture()
    result = execute_policy(runtime, base, np.zeros((2, 96)), np.zeros((2, 96)), day_offset=20)
    assert len(calls['ac']) == len(calls['meter']) == 1
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
    assert calls['meter'][0] == dict(pq_noise_rel=.005, v_noise_rel=.0005)
    assert all(v['day_offset'] == 20 for v in s.values())
    assert result['summary']['root_meter_seed'] == 70000019+100003*930000+1


def test_shared_signed_execution_noise_is_not_replaced_by_absolute_command():
    runtime, base, calls = _fixture()
    u = np.stack([np.ones(96)*.4, -np.ones(96)*.4])
    Z = np.stack([np.linspace(-3., 3., 96)]*2)
    result = execute_policy(runtime, base, u, Z, cv=.2)
    expected = u*(1+.2*Z)
    assert np.array_equal(result['private']['delivered_kw'], expected)
    assert np.array_equal(expected.sum(axis=0), np.zeros(96))
    assert len(calls['ac']) == 1


def test_nonconverged_ac_arrays_are_retained_without_observed_payload():
    runtime, base, calls = _fixture(converged=False)
    result = execute_policy(runtime, base, np.zeros((2, 96)), np.zeros((2, 96)))
    assert result['summary']['status'] == 'physical_unresolved'
    assert result['summary']['ac_converged_samples'] == 95
    assert result['summary']['failure_retained']
    assert 'AC' in result['private']
    assert all(v is None for v in result['scenes'].values())
    assert len(calls['ac']) == 1 and len(calls['meter']) == 0


def test_voltage_operating_limit_violation_is_scored_without_dropping_day():
    runtime, base, calls = _fixture(voltage_ok=False)
    result = execute_policy(runtime, base, np.zeros((2, 96)), np.zeros((2, 96)))
    assert result['summary']['status'] == 'complete'
    assert not result['summary']['voltage_limits_pass']
    assert not result['summary']['all_operation_checks_pass']
    assert result['summary']['voltage_violation_cells'] == 288
    assert all(s is not None for s in result['scenes'].values())
    assert len(calls['ac']) == 1


def test_nonfinite_control_is_rejected_before_any_call():
    runtime, base, calls = _fixture()
    u = np.zeros((2, 96)); u[0, 0] = np.nan
    with pytest.raises(ValueError):
        execute_policy(runtime, base, u, np.zeros_like(u))
    assert calls['ac'] == []
