"""Analytic and independent grid audits of continuous quadratic chord MILPs."""
from types import SimpleNamespace
import numpy as np
import pytest
from scipy.optimize import LinearConstraint
from scipy.sparse import csr_matrix

from research_experiments.mobile_ev_mle.ev_profile_chord_milp import (
    FrozenQuadratic, build_profile_chord_milp, check_control, chord_value,
    residual_intervals, solve_profile_chord_milp,
)


@pytest.mark.parametrize('segments', [1, 4, 8, 16])
def test_two_time_same_energy_maximum_variance(segments):
    q = FrozenQuadratic('contrast', [0], [[1, -1]])
    energy = LinearConstraint([[1, 1]], 1, 1)
    out = solve_profile_chord_milp([q], [0, 0], [1, 1], energy, segments=segments)
    assert out['physical_validation']['feasible']
    assert sum(out['u']) == pytest.approx(1, abs=1e-7)
    assert out['objective_lower'] == pytest.approx(1, abs=1e-7)
    assert out['objective_upper'] >= 1-1e-7
    assert out['original_gap'] < 1e-6
    assert not out['original_global_optimum_certified']


def test_coarse_optimal_milp_does_not_certify_original_exactness():
    qs = [FrozenQuadratic('left', [0], [[1]]), FrozenQuadratic('right', [1], [[-1]])]
    out = solve_profile_chord_milp(qs, [0], [1], segments=1)
    assert out['solver_success']
    assert out['u'][0] == pytest.approx(.5, abs=1e-7)
    assert out['objective_lower'] == pytest.approx(.25, abs=1e-7)
    assert out['objective_upper'] == pytest.approx(.5, abs=1e-7)
    assert out['original_gap'] == pytest.approx(.25, abs=1e-7)
    assert out['secant_error_max'] == pytest.approx(.25, abs=1e-10)
    assert not out['original_global_optimum_certified']
    fine = solve_profile_chord_milp(qs, [0], [1], segments=2)
    assert fine['original_gap'] < 1e-6


@pytest.mark.parametrize('segments', [1, 3, 8])
def test_signed_residual_chord_bound_including_zero(segments):
    q = FrozenQuadratic('signed', [-.3], [[1, -.5]])
    low, high, variable = residual_intervals(q, [-2, -.4], [3, .7])
    assert variable[0]
    knots = np.linspace(low[0], high[0], segments+1)
    error = np.max(np.diff(knots))**2/4
    for r in np.r_[np.linspace(low[0], high[0], 500), 0]:
        difference = chord_value(r, knots)-r*r
        assert -1e-12 <= difference <= error+1e-12
    out = solve_profile_chord_milp([q], [-2, -.4], [3, .7], segments=segments)
    analytic = max((-.3-2-.5*.7)**2, (-.3+3-.5*(-.4))**2)
    assert out['objective_lower'] == pytest.approx(analytic, abs=1e-6)
    assert out['objective_upper'] >= analytic-1e-6


def test_continuous_optimum_not_preselected_power_grid():
    qs = [FrozenQuadratic('a', [0], [[1]]), FrozenQuadratic('b', [1], [[-1]], .1)]
    out = solve_profile_chord_milp(qs, [0], [1], segments=16)
    # Original optimum is the crossing u=.45, not a chord knot.
    assert abs(out['u'][0]-.45) < .002
    assert not np.any(np.isclose(out['u'][0], np.linspace(0, 1, 17), atol=1e-4))
    assert out['objective_lower'] <= .45**2+1e-8
    assert out['objective_upper'] >= .45**2-1e-8
    assert out['original_gap'] <= out['secant_error_max']+1e-6


def test_offsets_and_negative_original_objective():
    qs = [FrozenQuadratic('constant', [2], [[0]], 10), FrozenQuadratic('variable', [0], [[1]], 10)]
    out = solve_profile_chord_milp(qs, [-1], [1], segments=4)
    assert out['objective_lower'] == pytest.approx(-9, abs=1e-7)
    assert out['objective_upper'] == pytest.approx(-9, abs=1e-7)
    assert out['chord_incumbent'] < 0


def test_nonuniform_d_offset_and_sparse_constraints_against_independent_grid():
    qs = [FrozenQuadratic('a', [.15, -.3], [[1, .2], [.7, -1.3]], .35),
          FrozenQuadratic('b', [-.8, .12, .4], [[-.4, 1.1], [.2, .8], [1.2, -.2]], .15)]
    constraints = [LinearConstraint(csr_matrix([[1, 1]]), .5, 1.2),
                   LinearConstraint([[1, -1]], -.6, .8)]
    out = solve_profile_chord_milp(qs, [0, 0], [1, 1], constraints, segments=8)
    # The grid is solely an external lower-bound audit, never the selector.
    scores = []
    for x in np.linspace(0, 1, 101):
        for y in np.linspace(0, 1, 101):
            if .5-1e-12 <= x+y <= 1.2+1e-12 and -.6-1e-12 <= x-y <= .8+1e-12:
                scores.append(min(sum((d+row[0]*x+row[1]*y)**2 for d, row in zip(q.d, q.T))-q.offset for q in qs))
    assert out['objective_upper'] >= max(scores)-1e-7
    assert out['objective_lower'] == pytest.approx(min(q.value(out['u']) for q in qs))
    assert out['original_gap'] <= out['secant_error_max']+1e-5
    assert out['physical_validation']['max_linear_violation'] < 1e-7


def test_fixed_coordinates_and_all_constant_model():
    q = FrozenQuadratic('fixed', [.3, 1.1], [[2, 3], [-1, 0]], .7)
    out = solve_profile_chord_milp([q], [1, 2], [1, 2], segments=4)
    assert out['n_binary'] == 0
    assert out['objective_lower'] == pytest.approx((.3+2+6)**2+(.1)**2-.7)
    assert out['original_gap'] < 1e-6


def test_residual_independent_of_controls_and_empty_residual():
    qs = [FrozenQuadratic('empty', [], np.zeros((0, 2)), -2),
          FrozenQuadratic('constant', [2], [[0, 0]], 1)]
    out = solve_profile_chord_milp(qs, [0, 0], [1, 1], segments=8)
    assert out['objective_lower'] == pytest.approx(2)
    assert out['objective_upper'] == pytest.approx(2)
    assert out['secant_error_max'] == 0


def test_infeasible_operations_have_no_incumbent():
    out = solve_profile_chord_milp([FrozenQuadratic('q', [0], [[1]])], [0], [1],
                                   LinearConstraint([[1]], 2, np.inf))
    assert out['status'] == 2
    assert out['u'] is None and out['objective_lower'] is None
    assert out['original_gap'] is None


def test_timeout_keeps_provided_feasible_incumbent_and_analytical_upper(monkeypatch):
    from research_experiments.mobile_ev_mle import ev_profile_chord_milp as module
    monkeypatch.setattr(module, 'milp', lambda *a, **kw: SimpleNamespace(
        x=None, fun=None, status=1, message='Time limit reached', success=False))
    out = module.solve_profile_chord_milp([FrozenQuadratic('q', [0], [[1, -1]])], [0, 0], [1, 1],
                                        LinearConstraint([[1, 1]], 1, 1), feasible_incumbents=[[.25, .75]])
    assert out['time_limit_reached']
    assert out['selected_origin'] == 'retained_feasible_incumbent'
    assert out['objective_lower'] == pytest.approx(.25)
    assert out['objective_upper'] >= 1
    assert out['solver_best_bound'] is None


def test_timeout_best_bound_is_used_instead_of_milp_incumbent(monkeypatch):
    from research_experiments.mobile_ev_mle import ev_profile_chord_milp as module
    built = build_profile_chord_milp([FrozenQuadratic('q', [0], [[1]])], [0], [1], segments=2)
    x = np.zeros(built['n_variables']); x[0] = .25
    monkeypatch.setattr(module, 'milp', lambda *a, **kw: SimpleNamespace(
        x=x, fun=-.125/built['scale'], mip_dual_bound=-.8/built['scale'], status=1,
        message='Time limit reached', success=False, mip_gap=5.4, mip_node_count=1))
    out = module.solve_profile_chord_milp([FrozenQuadratic('q', [0], [[1]])], [0], [1], segments=2)
    assert out['objective_lower'] == pytest.approx(.0625)
    assert out['objective_upper'] == pytest.approx(.8)
    assert out['chord_incumbent'] == pytest.approx(.125)


def test_invalid_solver_control_is_not_original_lower_bound(monkeypatch):
    from research_experiments.mobile_ev_mle import ev_profile_chord_milp as module
    monkeypatch.setattr(module, 'milp', lambda *a, **kw: SimpleNamespace(
        x=np.array([2.0]), fun=None, status=4, message='bad incumbent', success=False))
    out = module.solve_profile_chord_milp([FrozenQuadratic('q', [0], [[1]])], [0], [1])
    assert out['u'] is None
    assert not out['solver_physical_validation']['feasible']
    assert out['objective_lower'] is None


def test_contradictory_numeric_solver_bound_falls_back_to_box(monkeypatch):
    from research_experiments.mobile_ev_mle import ev_profile_chord_milp as module
    monkeypatch.setattr(module, 'milp', lambda *a, **kw: SimpleNamespace(
        x=np.array([1.0]), fun=-1.0, mip_dual_bound=-.2, status=0, message='optimal', success=True))
    out = module.solve_profile_chord_milp([FrozenQuadratic('q', [0], [[1]])], [0], [1])
    assert not out['bound_consistent']
    assert out['objective_upper'] >= out['objective_lower']


def test_error_decreases_by_four_when_segments_double():
    q = FrozenQuadratic('q', [.7, -.2], [[1, -2], [3, .4]])
    small = build_profile_chord_milp([q], [-1, 0], [2, 1], segments=4)
    large = build_profile_chord_milp([q], [-1, 0], [2, 1], segments=8)
    assert small['secant_error_max'] == pytest.approx(4*large['secant_error_max'])
    assert small['n_binary'] == 2*4 and large['n_binary'] == 2*8


def test_input_arrays_are_copied_and_not_mutated():
    d, T = np.array([1.0]), np.array([[2.0]])
    q = FrozenQuadratic('q', d, T); d[0] = 5; T[0, 0] = 7
    assert q.value([1]) == 9
    assert not q.d.flags.writeable and not q.T.flags.writeable


@pytest.mark.parametrize('segments', [0, -1, 2.5, True])
def test_bad_segments_rejected(segments):
    with pytest.raises(ValueError, match='segments'):
        build_profile_chord_milp([FrozenQuadratic('q', [0], [[1]])], [0], [1], segments=segments)


def test_invalid_domains_dimensions_or_incumbents_rejected():
    q = FrozenQuadratic('q', [0], [[1]])
    with pytest.raises(ValueError, match='finite ordered'):
        build_profile_chord_milp([q], [0], [np.inf])
    with pytest.raises(ValueError, match='dimension'):
        build_profile_chord_milp([q], [0, 0], [1, 1])
    with pytest.raises(ValueError, match='constraint control dimension'):
        build_profile_chord_milp([q], [0], [1], LinearConstraint([[1, 1]], 1, 1))
    with pytest.raises(ValueError, match='incumbent violates'):
        solve_profile_chord_milp([q], [0], [1], feasible_incumbents=[[2]])
    with pytest.raises(ValueError, match='matching residual'):
        FrozenQuadratic('bad', [0, 0], [[1]])
    with pytest.raises(ValueError, match='unique'):
        build_profile_chord_milp([q, q], [0], [1])


def test_physical_check_reports_each_violation_without_repair():
    out = check_control([1.1, .4], [0, 0], [1, 1], LinearConstraint([[1, 1]], 1, 1))
    assert not out['feasible']
    assert out['max_box_violation'] == pytest.approx(.1)
    assert out['max_linear_violation'] == pytest.approx(.5)
