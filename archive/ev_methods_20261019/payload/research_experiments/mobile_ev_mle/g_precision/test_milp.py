"""Small independent analytic audits of the shared-square MILP wrapper."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from research_experiments.mobile_ev_mle.ev_mobile_service_domain import (
    MobileServiceConfig, build_mobile_service_domain,
)
from research_experiments.mobile_ev_mle.g_precision.milp import solve_tasks
from research_experiments.mobile_ev_mle.g_precision.coefficients import QuadraticTask as Task


def domain(mobile=False):
    cfg = MobileServiceConfig(n_slots=2, n_sites=2, dt_hours=1.,
        p_charge_max_kw=2., p_discharge_max_kw=2., allow_discharge=False,
        battery_capacity_kwh=10., initial_soc=.5, soc_min=0., soc_max=1.,
        eta_charge=1., eta_discharge=1., terminal_energy_kwh=7., mobile=mobile,
        travel_slots=1, setup_slots=0, travel_energy_kwh=0., max_moves=1)
    return build_mobile_service_domain(cfg)


def complete(dom, power):
    model = dom.model.clone()
    for index, value in zip(dom.indices.u.ravel(), np.asarray(power).ravel()):
        model.add_row({int(index): 1.}, lower=float(value), upper=float(value))
    result = model.solve(time_limit=3.)
    assert result.x is not None
    x = np.asarray(result.x[:dom.model.n_variables])
    assert dom.check(x)['feasible']
    return x


def solve(dom, tasks, parents, **options):
    opts = dict(iterations=4, segments=4, max_segments=16,
                solver_seconds=2., total_seconds=8., backend='scipy',
                relative_gap=1e-7, absolute_gap=1e-7)
    opts.update(options)
    return solve_tasks(dom, tasks, parents, **opts)


def test_zero_g_site_never_receives_fictitious_score():
    dom = domain(mobile=True)
    parent = complete(dom, [[1., 0.], [1., 0.]])
    q = np.array([[0., 0.], [0., 1.]])
    task = Task('remote', 0., np.zeros((2, 2)), q)
    out = solve(dom, [task], {'parent': parent})
    assert out['success'] and out['score'] == pytest.approx(4., abs=1e-7)
    assert out['task_values']['remote'] == pytest.approx(task.value(np.array(out['power_kw'])))
    assert np.asarray(out['power_kw'])[1, 1] == pytest.approx(2.)
    assert out['objective_lower'] <= 4. <= out['objective_upper']
    assert out['operation_check']['moves'] == 1
    assert out['nuisance_dimension'] == 0
    assert out['score_gain'] == pytest.approx(4.)
    json.dumps(out, allow_nan=False)


def test_linear_diagonal_task_matches_analytic_endpoint_and_tiny_audit_grid():
    dom = domain()
    parent = complete(dom, [[1., 0.], [1., 0.]])
    task = Task('response', .4, np.array([[.8, 0.], [.2, 0.]]),
                np.array([[.25, 0.], [.5, 0.]]))
    # With fixed position u_0+u_1=2 and nonnegative power, the polynomial is
    # convex in u_0. Its maximum is one of the two analytic endpoints.
    endpoint = max(task.value(np.array([[p, 0.], [2-p, 0.]])) for p in (0., 2.))
    audit = max(task.value(np.array([[p, 0.], [2-p, 0.]]))
                for p in np.linspace(0., 2., 9))
    assert endpoint == pytest.approx(3.) and audit == pytest.approx(endpoint)
    out = solve(dom, [task], {'constant': parent})
    assert out['score'] == pytest.approx(endpoint, abs=1e-7)
    assert out['objective_lower'] <= endpoint <= out['objective_upper']
    assert out['inheritance_protected']
    assert out['completed'] and not out['unfinished']
    assert out['objective_scale'] != 1.
    assert all(row['chord_incumbent_is_original_lower'] is False for row in out['iterations'])


def test_maxmin_keeps_the_weak_task_instead_of_averaging_it_away():
    dom = domain()
    parent = complete(dom, [[1., 0.], [1., 0.]])
    easy = Task('easy', 100., np.zeros((2, 2)), np.array([[100., 0.], [0., 0.]]))
    weak = Task('weak', 0., np.zeros((2, 2)), np.array([[0., 0.], [1., 0.]]))
    out = solve(dom, [easy, weak], {'parent': parent})
    assert out['score'] == pytest.approx(4., abs=1e-7)
    assert out['task_values']['weak'] == pytest.approx(4.)
    assert out['task_values']['easy'] >= 100.-1e-7
    assert np.asarray(out['power_kw'])[1, 0] == pytest.approx(2.)
    assert out['objective_lower'] <= 4. <= out['objective_upper']


def test_best_of_all_parents_is_retained_despite_a_tiny_budget():
    dom = domain()
    balanced = complete(dom, [[1., 0.], [1., 0.]])
    endpoint = complete(dom, [[2., 0.], [0., 0.]])
    task = Task('gain', -2., np.zeros((2, 2)), np.array([[2., 0.], [0., 0.]]))
    out = solve(dom, [task], {'balanced': balanced, 'endpoint': endpoint}, total_seconds=1e-9)
    assert out['score'] == pytest.approx(6.)
    assert out['best_parent_score'] == pytest.approx(6.)
    assert out['score_gain'] == pytest.approx(0.)
    assert out['inheritance_protected']
    assert all(r['success'] for r in out['parents'].values())
    assert len(out['operation_x']) == dom.model.n_variables


def test_constant_zero_tasks_have_zero_nuisance_and_no_artificial_gain():
    dom = domain()
    parent = complete(dom, [[1., 0.], [1., 0.]])
    task = Task('zero', 0., np.zeros((2, 2)), np.zeros((2, 2)))
    out = solve(dom, [task], {'parent': parent})
    assert out['score'] == out['score_gain'] == 0.
    assert out['nuisance_dimension'] == 0
    assert out['objective_lower'] <= 0. <= out['objective_upper']
    assert out['completed']


def test_two_symmetric_tasks_have_a_nonendpoint_maxmin_solution():
    dom = domain()
    parent = complete(dom, [[1., 0.], [1., 0.]])
    tasks = [Task('first', 0., np.zeros((2, 2)), np.array([[1., 0.], [0., 0.]])),
             Task('second', 0., np.zeros((2, 2)), np.array([[0., 0.], [1., 0.]]))]
    out = solve(dom, tasks, {'balanced': parent})
    assert out['score'] == pytest.approx(1., abs=1e-7)
    assert out['objective_lower'] <= 1. <= out['objective_upper']
    assert out['inheritance_protected']


@pytest.mark.parametrize('quadratic,linear', [
    (np.array([[-1., 0.], [0., 0.]]), np.zeros((2, 2))),
    (np.zeros((2, 2)), np.array([[1., 0.], [0., 0.]])),
])
def test_negative_curvature_or_unsupported_pure_linear_terms_rejected(quadratic, linear):
    dom = domain()
    unchecked = SimpleNamespace(name='invalid', constant=0., linear=linear,
                                quadratic=quadratic, value=lambda u: 0.)
    with pytest.raises(ValueError):
        solve(dom, [unchecked], {})


def test_infeasible_parent_is_reported_and_legal_parent_survives():
    dom = domain()
    parent = complete(dom, [[1., 0.], [1., 0.]])
    invalid = parent.copy(); invalid[dom.indices.u[0, 0]] = 20.
    task = Task('power', 0., np.zeros((2, 2)), np.ones((2, 2)))
    out = solve(dom, [task], {'bad': invalid, 'legal': parent})
    assert out['success'] and out['inheritance_protected']
    assert not out['parents']['bad']['success']
    assert out['parents']['legal']['success']
    assert any(f['kind'] == 'parent_rejected' for f in out['failures'])


def test_large_completion_constant_does_not_replace_original_scoring():
    dom = domain()
    parent = complete(dom, [[1., 0.], [1., 0.]])
    task = Task('small_curvature', .3, np.array([[1., 0.], [.2, 0.]]),
                np.array([[1e-15, 0.], [1e-15, 0.]]))
    out = solve(dom, [task], {'parent': parent})
    assert out['success'] and out['score'] >= task.value(np.array([[1., 0.], [1., 0.]]))
    assert out['score'] == pytest.approx(task.value(np.asarray(out['power_kw'])), abs=1e-12)
    assert out['objective_upper'] >= out['objective_lower']
    assert out['completion_error_bound'] > 0.
