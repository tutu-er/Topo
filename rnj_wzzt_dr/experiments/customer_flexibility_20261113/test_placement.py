"""Validate the smaller energy contract and reuse of the original guards."""
import numpy as np
import pytest

from .model import CustomerCase
from .placement import CandidateEvidence, configurations, directional_configurations


def test_electric_buffer_is_a_linear_smaller_thermal_contract():
    case = CustomerCase(configurations()['compact'])
    direction = np.zeros(case.shape)
    direction[4] = -1.
    u, receipt = case.linear_optimum(direction, least_fee=True)
    assert receipt['resource_audit']['passed']
    for j, info in case.thermal.items():
        assert np.allclose(info['electric_buffer_kwh'], [-.85, .85])
        e = 0.
        trajectory = []
        for h in range(24):
            e = .98*e+u[h, j]
            trajectory.append(e)
        assert max(abs(np.array(trajectory))) <= .85+1e-7
        assert abs(e) < 1e-7
    assert all(v.min() >= 40 and v.max() <= 55 for v in case.temperatures(u).values())


def test_added_contracts_contain_the_old_control_domain_with_a_common_forecast():
    configs = configurations()
    old, new = CustomerCase(configs['compact']), CustomerCase(configs['focused'])
    d = np.zeros(old.shape)
    d[4] = -1.
    u, _ = old.linear_optimum(d, least_fee=True)
    expanded = np.zeros(new.shape)
    expanded[:, :old.R] = u
    assert new.audit(expanded)['passed']
    assert np.isclose(old.fee(u), new.fee(expanded))
    assert np.allclose(old.shift_pq_pu(u, 200), new.shift_pq_pu(expanded, 200))


def test_general_resource_guard_quadratic_matches_the_owning_full_h_formula():
    case = CustomerCase(configurations()['focused'])
    rng = np.random.default_rng(2026111321)
    H = rng.normal(size=(3, 15, 30))*.01
    K = H*rng.uniform(.95, 1.05, (3, 15, 1))
    forecast = rng.uniform(.001, .05, (96, 30))
    d = np.zeros(case.shape)
    d[4] = -1.
    u, _ = case.linear_optimum(d, least_fee=True)
    e = CandidateEvidence(case, H, 0, [1, 2], forecast, 1e-5, 200., K)
    assert np.allclose(e.values(u), e.direct(u), rtol=1e-10, atol=1e-9)
    assert np.allclose(e.values(u)[:2], e.nominal.gains(case.delivery(u)), rtol=1e-10, atol=1e-9)
    assert np.all(e.direct(u)[:2] <= e.power_box_upper()+1e-8)


def test_reservation_parameter_is_not_silently_ignored():
    config = configurations()['compact']
    config['pricing']['reservation_fee_yuan_per_kw_day'] = .5
    with pytest.raises(ValueError, match='reservation'):
        CustomerCase(config)


def test_directional_contracts_keep_device_counts_and_offers():
    configs = directional_configurations()
    old, changed = configs['compact']['resources'], configs['relocated']['resources']
    assert len(old) == len(changed) == 8
    for a, b in zip(old, changed, strict=True):
        assert a['kind'] == b['kind']
        assert a['fee_positive_yuan_per_kwh'] == b['fee_positive_yuan_per_kwh']
        assert a['fee_negative_yuan_per_kwh'] == b['fee_negative_yuan_per_kwh']
    assert {r['node_label'] for r in changed if r['kind'] == 'pv'} == {101, 106, 110}
    assert {r['node_label'] for r in changed if r['kind'] == 'thermal'} == {102, 106}
    assert len(configs['augmented']['resources']) == 10
