"""Check device physics, units, co-location and the retained evidence adapter."""
from copy import deepcopy

import numpy as np
import pytest

from .model import CustomerCase, window


@pytest.fixture
def case():
    return CustomerCase()


def witness(case):
    direction = np.zeros(case.shape)
    direction[np.flatnonzero(window(case.clock, 12, 13))[0]] = -1.
    return case.linear_optimum(direction, least_fee=True)[0]


def test_zero_keeps_ev_energy_and_water_temperature(case):
    audit = case.audit(np.zeros(case.shape))
    assert audit['passed'] and audit['service_fee_yuan'] == 0.
    assert np.allclose(list(audit['ev_delivered_battery_kwh'].values()), 7.5, atol=1e-12)
    assert all(np.array_equal(v, [47.5, 47.5]) for v in audit['thermal_temperature_ranges_C'].values())
    assert len(set(case.node)) == 6 and case.R == 8


def test_pv_ac_clipping_and_no_manufactured_curtailment(case):
    for j, r in enumerate(case.resources):
        if r['kind'] != 'pv':
            continue
        assert case.solar_available[:, j].max() == 3.75
        assert case.solar_available[window(case.clock, 0, 6), j].sum() == 0.
        assert np.min(case.lower[:, j]) == 0.  # no extra generation above available baseline
        assert np.allclose(case.baseline[:, j], -case.solar_available[:, j])
    c = deepcopy(case.config)
    c['resources'][0]['baseline_available_fraction'] = .8
    with pytest.raises(ValueError, match='provenance'):
        CustomerCase(c)
    c['resources'][0]['baseline_curtailment_provenance'] = 'unit-test declared cap, not the main benchmark'
    limited = CustomerCase(c)
    assert limited.lower[:, 0].min() < 0 < limited.upper[:, 0].max()


def test_ev_energy_limit_rejects_power_feasible_overcharging(case):
    u = np.zeros(case.shape)
    j = next(i for i, r in enumerate(case.resources) if r['id'] == 'ev104')
    u[:, j] = case.upper[:, j]
    assert np.all(u <= case.upper) and np.all(u >= case.lower)
    assert not case.audit(u)['passed']
    assert case.audit(u)['ev_delivered_battery_kwh']['ev104'] > 7.5


def test_thermal_forward_recursion_matches_eliminated_linear_states(case):
    u = witness(case)
    values = case.temperatures(u)
    for j, info in case.thermal.items():
        from_rows = 47.5+info['rows']@u.ravel()/info['capacity_kwh_per_C']
        assert np.allclose(from_rows, values[case.resources[j]['id']][1:], atol=1e-12)
        assert abs(values[case.resources[j]['id']][-1]-47.5) < 1e-7


def test_terminal_heat_requirement_is_not_only_a_power_bound(case):
    u = np.zeros(case.shape)
    j = next(iter(case.thermal))
    u[-1, j] = .1
    result = case.audit(u)
    assert not result['passed'] and result['max_linear_inequality_violation'] == 0.
    assert result['thermal_terminal_error_C'] > .1


def test_colocated_devices_have_distinct_reactive_response_and_correct_units(case):
    u = np.zeros(case.shape)
    pv = next(i for i, r in enumerate(case.resources) if r['id'] == 'pv106')
    hp = next(i for i, r in enumerate(case.resources) if r['id'] == 'hp106')
    h = np.flatnonzero(window(case.clock, 12, 13))[0]
    u[h, pv] = 1.
    u[h, hp] = -.5
    p, q = case.aggregate(case.delivery(u))
    node = case.labels.index(106)
    assert np.allclose(p[4*h:4*h+4, node], .5)
    assert np.allclose(q[4*h:4*h+4, node], -.5*np.tan(np.arccos(.95)))
    shift = case.shift_pq_pu(u, 200.)
    assert np.allclose(shift[:, :case.N]*200, p)
    assert np.allclose(shift[:, case.N:]*200, q)


def test_reordered_terminal_labels_preserve_physical_node_delivery(case):
    other = CustomerCase(labels=tuple(reversed(case.labels)))
    u = witness(case)
    p1, q1 = case.aggregate(case.delivery(u))
    p2, q2 = other.aggregate(other.delivery(u))
    assert np.array_equal(p1, p2[:, ::-1])
    assert np.array_equal(q1, q2[:, ::-1])
    with pytest.raises(ValueError, match='Terminal IDs'):
        CustomerCase(labels=tuple(range(15)))


def test_retained_evidence_matches_full_h_pq_formula(case):
    from ..expected_logodds_admm_20261113.evidence import direct_gains
    rng = np.random.default_rng(2026111317)
    H = rng.normal(size=(3, case.N, 2*case.N))*.01
    forecast = rng.uniform(.001, .05, (96, 2*case.N))
    u = witness(case)
    e = case.evidence(H, 0, [1, 2], forecast, 1e-5, 200., .1)
    original = direct_gains(H, 0, [1, 2], forecast, 1e-5, case, 200., case.delivery(u))
    assert np.allclose(e.gains(case.delivery(u)), original, rtol=1e-11, atol=1e-10)


def test_gurobi_linear_domain_fee_and_qcqp_connection(case):
    gp = pytest.importorskip('gurobipy')
    try:
        env = gp.Env(empty=True)
        env.setParam('OutputFlag', 0)
        env.start()
    except gp.GurobiError as exc:
        pytest.skip('Gurobi license unavailable: '+str(exc))
    u = witness(case)
    rng = np.random.default_rng(2026111318)
    H = rng.normal(size=(2, case.N, 2*case.N))*.01
    # Align a nonzero baseline with the shift, so both the linear and quadratic
    # parts matter: the correct gain is three times the squared shift distance.
    e = case.evidence(H, 0, [1], case.shift_pq_pu(u, 200.), 1e-5, 200., .01)
    e.gain_required = float(e.gains(case.delivery(u))[0]/2)
    try:
        with gp.Model('fixed_resource_evidence_interface', env=env) as m:
            m.Params.OutputFlag = 0
            m.Params.Threads = 1
            m.Params.NonConvex = 2
            m.Params.FeasibilityTol = 1e-8
            m.Params.TimeLimit = 5.
            decision, fee = case.attach(m)
            case.attach_evidence(m, decision, e)
            m.addConstr(decision == u)
            m.setObjective(fee)
            m.optimize()
            assert m.Status == gp.GRB.OPTIMAL
            assert m.NumIntVars == 0 and m.NumQConstrs == 1
            assert np.isclose(m.ObjVal, case.fee(u), atol=1e-6)
    finally:
        env.dispose()
