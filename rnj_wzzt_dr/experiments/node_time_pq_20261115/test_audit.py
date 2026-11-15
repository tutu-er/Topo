"""Independent arithmetic checks; no QCQP or AC is invoked."""
import numpy as np
import pytest

from . import audit


def sample():
    from datetime import datetime, timedelta, timezone
    stamps = [(datetime(2024, 1, 4, 7, tzinfo=timezone.utc)+timedelta(minutes=30*i)).isoformat()
              for i in range(96)]
    t = np.arange(96, dtype=float)
    p = np.c_[2+np.sin(t), -1+np.cos(t*.2)]
    q = np.c_[3+.2*p[:, 0]+.05*np.cos(t*.7), -2-.3*p[:, 1]+.1*np.sin(t*.4)]
    return p, q, stamps, stamps[::2][:24]


def test_local_hours_use_timestamp_and_not_row_number():
    p, q, stamps, decisions = sample()
    result = audit.reconstruct_band(p, q, stamps, decisions)
    assert result['decision_clock_hour'][0] == 8
    assert result['decision_clock_hour'][-1] == 7
    assert np.array_equal(result['training_count_by_hour'], np.repeat(4, 24))
    for hour in range(24):
        residual = result['residual_pu'][result['training_clock_hour'] == hour]
        np.testing.assert_array_equal(result['width_by_clock_hour_pu'][hour], residual.max(0)-residual.min(0))


def test_affine_slope_keeps_offset_and_no_ratio_division():
    p, q, stamps, decisions = sample()
    first = audit.reconstruct_band(p, q, stamps, decisions)
    second = audit.reconstruct_band(p+100, q-7, stamps, decisions)
    np.testing.assert_allclose(first['slope'], second['slope'], atol=1e-13)
    np.testing.assert_allclose(first['width_pu'], second['width_pu'], atol=1e-13)


def test_band_allows_independent_q_but_checks_every_node_hour():
    p = np.zeros((24, 2)); q = p.copy(); width = np.ones_like(p)*.01
    q[3, 1] = .009
    assert audit.direct_band(p, q, np.array([.3, -.2]), width)['passed']
    q[3, 1] = .011
    result = audit.direct_band(p, q, np.array([.3, -.2]), width)
    assert not result['passed'] and result['violated_cells'] == 1


def test_exact_zero_width_not_replaced_by_floor():
    p = np.zeros((24, 1)); q = p.copy(); q[0, 0] = 1e-5
    assert not audit.direct_band(p, q, np.array([0.]), np.zeros_like(p))['passed']


def test_positive_slack_is_zero_violation_not_a_negative_violation():
    p=np.zeros((24,1));q=p.copy();b=np.ones_like(p)*.01
    result=audit.direct_band(p,q,np.array([.2]),b)
    assert result['minimum_slack_pu']==pytest.approx(.01)
    assert result['maximum_violation_pu']==0
    q[0,0]=.012
    result=audit.direct_band(p,q,np.array([.2]),b)
    assert result['maximum_violation_pu']==pytest.approx(.002)
    assert result['minimum_slack_pu']==pytest.approx(-.002)


def test_timezones_and_empty_hours_rejected():
    with pytest.raises(ValueError, match='timezone'):
        audit.local_hours(['2024-01-01T00:00:00'])
    p, q, stamps, decisions = sample()
    with pytest.raises(ValueError, match='Every local hour'):
        audit.reconstruct_band(p[:3], q[:3], stamps[:3], decisions)


def test_sdp_numerical_value_not_a_certified_lower_bound():
    assert not audit.no_sdp_certificate({'status': 'optimal_inaccurate', 'objective': 2.})['certified_lower_bound']
    with pytest.raises(ValueError, match='no certified'):
        audit.no_sdp_certificate({'used_as_qcqp_lower_bound': True})
