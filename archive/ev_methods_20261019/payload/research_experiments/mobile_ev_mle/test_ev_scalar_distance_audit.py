"""Independent audit of scalar chords, two directions and moving/off periods.

The exact values below are obtained by clipping diagonal source coordinates,
without calling the source QP.  Action sampling is only an audit of the claimed
upper bound; it is not the action selector or a claim of discrete optimality.
"""
from types import SimpleNamespace

import numpy as np
import pytest

from . import ev_fixed_train_design as fixed
from . import ev_scalar_distance_milp as scalar
from .ev_mobile_service_domain import MobileServiceConfig, build_mobile_service_domain
from .ev_original_domain_milp import OriginalDomainPair


def asymmetric_context(T=3, *, zero_control=False):
    """Two different directed responses, root channels and active source boxes."""
    n = 2
    m = T * (n + 2)
    pairs = []
    for k in range(2):
        c = np.zeros(m)
        B = np.zeros((m, T * n))
        G = np.zeros((m, 2 * T))
        lower = np.tile([-.4 + .1 * k, .1], T)
        upper = np.tile([.2 + .1 * k, .1], T)
        center = np.tile([.15, -.2 + .1 * k], T)
        for t in range(T):
            rows = np.r_[np.arange(t * n, (t + 1) * n), T * n + t, T * (n + 1) + t]
            c[rows] = np.array([.13, -.24, .31, -.11]) * (k + 1) + .07 * t
            B[np.ix_(rows, [2 * t, 2 * t + 1])] = np.array([
                [.9 + .1 * t, -.5], [.3, 1.1 - .1 * t], [.2, -.4], [-.3, .6]
            ]) * (1. if k == 0 else -.7)
            G[rows[0], 2 * t] = .6 if k == 0 else 1.3
            G[rows[1], 2 * t + 1] = -1.2 if k == 0 else .8
        if zero_control:
            B[:] = 0.
        pairs.append(OriginalDomainPair(
            ['a_to_b', 'b_to_a'][k], c, B, G, np.zeros((T * n, m, 2 * T)),
            lower, upper, center, np.zeros((2 * T, T * n))))
    return dict(success=True, bank=[SimpleNamespace(name='a'), SimpleNamespace(name='b')],
                control_shape=(T, n), pairs=pairs)


def exact_directed_half_sse(pair, power):
    """Analytic box projection for a diagonal source map, including fixed sQ."""
    residual = pair.c + pair.B @ np.asarray(power).ravel()
    for column in range(pair.A0.shape[1]):
        rows = np.flatnonzero(pair.A0[:, column])
        assert len(rows) == 1
        row = rows[0]
        gain = pair.A0[row, column]
        source = np.clip(pair.base_center[column] - residual[row] / gain,
                         pair.lower[column], pair.upper[column])
        residual[row] += gain * (source - pair.base_center[column])
    return .5 * float(residual @ residual)


def exact_score(context, power):
    return float(np.mean([exact_directed_half_sse(p, power) for p in context['pairs']]))


def operation_domain(T=3, *, moving=False, zero_control=False):
    return build_mobile_service_domain(MobileServiceConfig(
        n_slots=T, n_sites=2, dt_hours=1., p_charge_max_kw=1.3,
        p_discharge_max_kw=.7, battery_capacity_kwh=10., initial_soc=.5,
        terminal_energy_kwh=5., eta_charge=1., eta_discharge=1.,
        mobile=moving, initial_site=0, terminal_site=1 if moving else 0,
        terminal_connected=moving, setup_slots=1, travel_slots=1,
        travel_energy_kwh=.2, max_moves=1 if moving else 0,
        max_throughput_kwh=3.))


def test_asymmetric_multinode_time_partition_and_directed_average():
    context = asymmetric_context()
    domain = operation_domain()
    power = np.array([[0., .43], [-.28, 0.], [0., .81]])
    exact = exact_score(context, power)
    result = scalar.action_profile_bounds(context, power)
    assert result['success']
    assert result['primal_upper'] == pytest.approx(exact, rel=1e-12, abs=1e-12)
    assert result['dual_lower'] <= exact + 1e-11
    assert exact - result['dual_lower'] < 1e-9

    table = scalar.build_scalar_tables(context, domain, grid_points=9)
    # Recover the score by independent one-time actions, subtracting all other
    # background periods.  This also checks root-P and root-Q channel indexing.
    zero = exact_score(context, np.zeros_like(power))
    total = zero
    for t, j in ((0, 1), (1, 0), (2, 1)):
        one = np.zeros_like(power)
        one[t, j] = power[t, j]
        total += exact_score(context, one) - zero
    assert total == pytest.approx(exact, abs=1e-12)
    assert sum(x['primal_upper'] for x in table['background']) == pytest.approx(zero, abs=1e-12)


@pytest.mark.parametrize('points', [3, 5, 9])
def test_every_asymmetric_chord_has_correct_upper_direction_and_error(points):
    context = asymmetric_context()
    table = scalar.build_scalar_tables(context, operation_domain(), grid_points=points)
    T, n = context['control_shape']
    zero = np.zeros((T, n))
    background = exact_score(context, zero)
    for t in range(T):
        for j in range(n):
            row = table['tables'][t][j]
            for p in np.linspace(-.7, 1.3, 71):
                power = zero.copy()
                power[t, j] = p
                exact_time = exact_score(context, power) - background + table['background'][t]['primal_upper']
                upper = np.interp(p, row['grid'], row['primal_upper'])
                assert exact_time <= upper + 1e-11
                assert upper - exact_time <= row['total_error_upper'] + 1e-10
            assert row['curvature_upper'] > 0.
    # Asymmetric charge/discharge limits can insert zero as an extra grid node.
    if points > 3:
        assert table['tables'][0][0]['actual_node_count'] == points + 1


@pytest.mark.parametrize('points', [3, 5])
def test_forced_move_and_setup_keep_background_exactly_once(points):
    context = asymmetric_context(T=4, zero_control=True)
    domain = operation_domain(T=4, moving=True)
    solved = scalar.solve_scalar_distance_milp(context, domain, grid_points=points,
        backend='scipy', solver_seconds=5.)
    assert solved['success'] and solved['operation']['feasible']
    assert solved['operation']['moves'] == 1
    assert np.count_nonzero(np.max(np.abs(solved['power_kw']), axis=1) < 1e-8) >= 2
    exact = exact_score(context, np.zeros((4, 2)))
    assert solved['solver_proxy_incumbent'] == pytest.approx(exact, abs=1e-9)
    assert solved['frozen_objective_upper'] == pytest.approx(exact, abs=1e-9)
    assert solved['frozen_objective_lower'] <= exact + 1e-10
    assert solved['frozen_absolute_gap'] < 1e-8
    assert solved['added_binary_count'] == (0 if points == 3 else 4 * 2 * 5)


@pytest.mark.parametrize('points', [3, 5, 9])
def test_mobile_bound_covers_independent_feasible_actions(points):
    context = asymmetric_context(T=4)
    domain = operation_domain(T=4, moving=True)
    solved = scalar.solve_scalar_distance_milp(context, domain, grid_points=points,
        backend='scipy', solver_seconds=5.)
    assert solved['success'] and solved['bound_consistent']
    returned = exact_score(context, solved['power_kw'])
    assert solved['frozen_objective_lower'] <= returned + 1e-9
    assert returned <= solved['frozen_objective_upper'] + 1e-9
    # Declared actions share one executable travel/setup schedule.  They merely
    # audit the MILP upper bound; they are not enumerated by the selector.
    for first in (-.7, -.2, 0., .2, .9):
        power = np.zeros((4, 2))
        power[2:, 1] = [first, .2 - first]
        feasible = fixed._realize(domain, power)
        assert feasible['success']
        assert exact_score(context, power) <= solved['frozen_objective_upper'] + 1e-8


def test_external_zero_tables_rejected_before_they_can_certify_false_zero_gap(monkeypatch):
    old = asymmetric_context(T=4, zero_control=True)
    new = asymmetric_context(T=4)
    domain = operation_domain(T=4, moving=True)
    table = scalar.build_scalar_tables(old, domain, grid_points=3)
    def must_not_solve(*args, **kwargs):
        raise AssertionError('stale tables must be rejected before MILP')
    monkeypatch.setattr(scalar, '_solve_mip', must_not_solve)
    with pytest.raises(ValueError, match='different frozen'):
        scalar.solve_scalar_distance_milp(new, domain, tables=table, backend='scipy')
