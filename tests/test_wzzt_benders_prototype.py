"""Small independent oracles for the full-domain, one-atom Benders prototype.

The oracle constructs a dense fixed-support LAD regression directly, without
using either the production MILP builders or the prototype LP construction.
Exhaustion is confined to these tiny test cases, never the implementation.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pytest
from scipy.optimize import linprog

from experiments.wzzt_benders_prototype import (
    BendersExtensionProblem,
    solve_best_laminar_extension_benders,
)


def _legal_extensions(n, supports):
    """Independent set-theoretic enumeration of the complete feasible domain."""
    old = [frozenset(s) for s in supports]
    for size in range(1, n + 1):
        for support in combinations(range(n), size):
            new = frozenset(support)
            if new in old:
                continue
            if all(not (new & prev) or new <= prev or prev <= new for prev in old):
                yield support


def _oracle_lp(scenarios, supports, r_bound, x_bound):
    """Build min mean(abs(d - A beta)) with shared R/X and free intercepts."""
    n = np.asarray(scenarios[0]["P_terminal"]).shape[1]
    k = len(supports)
    ns = len(scenarios)
    rows = sum(np.asarray(s["P_terminal"]).size for s in scenarios)
    design = np.zeros((rows, 2 * k + ns * n))
    observed = np.empty(rows)
    row = 0
    for s_index, scenario in enumerate(scenarios):
        p = np.asarray(scenario["P_terminal"])
        q = np.asarray(scenario["Q_terminal"])
        d = np.asarray(scenario["drop_target"])
        for time in range(len(p)):
            for terminal in range(n):
                for atom, support in enumerate(supports):
                    if terminal in support:
                        design[row, atom] = sum(p[time, j] for j in support)
                        design[row, k + atom] = sum(q[time, j] for j in support)
                design[row, 2 * k + s_index * n + terminal] = 1.0
                observed[row] = d[time, terminal]
                row += 1
    identity = np.eye(rows)
    constraints = np.block([[design, -identity], [-design, -identity]])
    rhs = np.concatenate([observed, -observed])
    objective = np.r_[np.zeros(design.shape[1]), np.full(rows, 1.0 / rows)]
    bounds = (
        [(0.0, r_bound)] * k
        + [(0.0, x_bound)] * k
        + [(None, None)] * (ns * n)
        + [(0.0, None)] * rows
    )
    answer = linprog(objective, A_ub=constraints, b_ub=rhs, bounds=bounds, method="highs")
    assert answer.success, answer.message
    return float(answer.fun), answer.x[: design.shape[1]]


def _case(n=3, supports=(), new=(0, 1), *, seed=11, noise=0.0, lengths=(9,)):
    rng = np.random.default_rng(seed)
    atoms = tuple(supports) + (tuple(new),)
    r = rng.uniform(0.12, 0.72, len(atoms))
    x = rng.uniform(0.09, 0.52, len(atoms))
    scenarios = []
    for scenario_index, length in enumerate(lengths):
        # Signed injections deliberately exercise both signs of the design.
        p = rng.normal(size=(length, n))
        q = rng.normal(size=(length, n))
        intercept = rng.normal(size=n) * (scenario_index + 1.0)
        d = np.broadcast_to(intercept, p.shape).copy()
        for atom, support in enumerate(atoms):
            contribution = r[atom] * p[:, support].sum(axis=1)
            contribution += x[atom] * q[:, support].sum(axis=1)
            for terminal in support:
                d[:, terminal] += contribution
        d += noise * rng.normal(size=d.shape)
        scenarios.append({"name": f"s{scenario_index}", "P_terminal": p,
                          "Q_terminal": q, "drop_target": d})
    return scenarios


def _y(problem, support):
    support = set(support)
    return np.asarray([float(i in support and j in support) for i, j in problem.pairs])


def _exhaustive_objectives(scenarios, supports, r_bound=1.0, x_bound=0.8):
    n = np.asarray(scenarios[0]["P_terminal"]).shape[1]
    return {new: _oracle_lp(scenarios, tuple(supports) + (new,), r_bound, x_bound)[0]
            for new in _legal_extensions(n, supports)}


@pytest.mark.parametrize("n,supports,new,noise,lengths,bounds", [
    (3, (), (0, 2), 0.0, (9,), (1.0, 0.8)),
    (4, ((0, 1),), (0, 1, 2), 0.035, (10,), (1.0, 0.8)),
    (4, ((0,), (2, 3)), (0, 1, 2, 3), 0.025, (7, 11), (1.0, 0.8)),
    # Deliberately tight bounds ensure the comparison also checks bounded fitting.
    (3, ((0, 1, 2),), (1, 2), 0.01, (10,), (0.2, 0.15)),
])
def test_benders_matches_complete_independent_enumeration(n, supports, new, noise, lengths, bounds):
    scenarios = _case(n, supports, new, noise=noise, lengths=lengths)
    expected = _exhaustive_objectives(scenarios, supports, *bounds)
    result = solve_best_laminar_extension_benders(
        scenarios, supports, r_upper_bound=bounds[0], x_upper_bound=bounds[1],
        max_iterations=200, time_limit=60.0,
        absolute_tolerance=1e-8, relative_tolerance=1e-8,
    )
    optimum = min(expected.values())
    assert result.proven_optimal, result.stop_reason
    assert tuple(result.support) in expected
    assert result.objective == pytest.approx(optimum, abs=3e-7)
    assert expected[tuple(result.support)] == pytest.approx(optimum, abs=3e-7)
    assert result.lower_bound <= optimum + 3e-7
    assert result.upper_bound >= optimum - 3e-7
    assert result.absolute_gap <= 3e-7


@pytest.mark.parametrize("lengths,noise", [((9,), 0.0), ((6, 10), 0.04)])
def test_subproblem_refits_old_and_new_weights_with_shared_scenario_parameters(lengths, noise):
    supports = ((0, 1),)
    new = (0, 1, 2)
    scenarios = _case(3, supports, new, noise=noise, lengths=lengths)
    problem = BendersExtensionProblem(scenarios, supports, r_upper_bound=1.0, x_upper_bound=0.8)
    result = problem.solve_subproblem(_y(problem, new))
    expected, _ = _oracle_lp(scenarios, supports + (new,), 1.0, 0.8)
    assert result.objective == pytest.approx(expected, abs=1e-8)
    assert np.asarray(result.r_values).shape == (2,)
    assert np.asarray(result.x_values).shape == (2,)
    assert np.asarray(result.intercepts).shape == (len(scenarios), 3)
    assert np.all(np.asarray(result.r_values) >= -1e-9)
    assert np.all(np.asarray(result.r_values) <= 1.0 + 1e-9)
    assert np.all(np.asarray(result.x_values) >= -1e-9)
    assert np.all(np.asarray(result.x_values) <= 0.8 + 1e-9)
    error_sum = 0.0
    count = 0
    for s_index, scenario in enumerate(scenarios):
        p, q, d = (scenario[key] for key in ("P_terminal", "Q_terminal", "drop_target"))
        prediction = np.broadcast_to(result.intercepts[s_index], d.shape).copy()
        for atom, support in enumerate(supports + (new,)):
            contribution = result.r_values[atom] * p[:, support].sum(axis=1)
            contribution += result.x_values[atom] * q[:, support].sum(axis=1)
            for terminal in support:
                prediction[:, terminal] += contribution
        error_sum += np.abs(d - prediction).sum()
        count += d.size
    assert error_sum / count == pytest.approx(result.objective, abs=1e-8)


@pytest.mark.parametrize("supports", [(), ((0, 1),)])
def test_every_generated_dual_cut_is_global_and_tight_on_its_origin(supports):
    scenarios = _case(3, supports, (0, 1, 2), noise=0.065, lengths=(7, 8))
    problem = BendersExtensionProblem(scenarios, supports, r_upper_bound=0.5, x_upper_bound=0.35)
    oracle = _exhaustive_objectives(scenarios, supports, 0.5, 0.35)
    for origin in oracle:
        origin_y = _y(problem, origin)
        result = problem.solve_subproblem(origin_y)
        gradient = np.asarray(result.cut_gradient)
        assert gradient.shape == origin_y.shape
        assert result.objective == pytest.approx(oracle[origin], abs=1e-8)
        assert result.dual_objective == pytest.approx(result.objective, abs=1e-8)
        assert abs(result.primal_dual_gap) < 1e-8
        cut_at_origin = result.cut_constant + np.dot(gradient, origin_y)
        assert cut_at_origin == pytest.approx(result.objective, abs=1e-8)
        for target, target_objective in oracle.items():
            cut_at_target = result.cut_constant + np.dot(gradient, _y(problem, target))
            assert cut_at_target <= target_objective + 2e-8, (origin, target, cut_at_target, target_objective)


def test_scenarios_cannot_fit_inconsistent_impedances_independently():
    # With one terminal and two differently scaled copies, free intercepts cannot
    # absorb incompatible slopes. An erroneous per-scenario fit would return 0.
    p = np.arange(-3.0, 4.0).reshape(-1, 1)
    scenarios = [{"P_terminal": p, "Q_terminal": np.zeros_like(p), "drop_target": slope * p + offset}
                 for slope, offset in [(0.2, 4.0), (0.8, -2.0)]]
    problem = BendersExtensionProblem(scenarios, (), r_upper_bound=1.0, x_upper_bound=1.0)
    result = problem.solve_subproblem(_y(problem, (0,)))
    expected, _ = _oracle_lp(scenarios, ((0,),), 1.0, 1.0)
    assert expected > 0.25
    assert result.objective == pytest.approx(expected, abs=1e-9)


def test_no_legal_extension_reports_absence_without_an_optimality_claim():
    scenarios = _case(1, (), (0,))
    result = solve_best_laminar_extension_benders(
        scenarios, ((0,),), r_upper_bound=1.0, x_upper_bound=1.0,
    )
    assert result.support is None
    assert result.objective is None
    assert result.stop_reason == "no_feasible_extension"
    assert not result.proven_optimal
    assert result.lower_bound == float("inf")
    assert result.upper_bound == float("inf")
    assert result.absolute_gap is None


def test_iteration_limit_preserves_a_valid_bound_and_does_not_claim_convergence():
    scenarios = _case(4, (), (0, 2, 3), noise=0.1, lengths=(12,))
    oracle = _exhaustive_objectives(scenarios, ())
    optimum = min(oracle.values())
    assert optimum > 1e-3  # The initial zero lower bound cannot certify this case.
    result = solve_best_laminar_extension_benders(
        scenarios, (), r_upper_bound=1.0, x_upper_bound=0.8,
        max_iterations=1, absolute_tolerance=1e-10, relative_tolerance=1e-10,
    )
    assert result.iterations == 1
    assert not result.proven_optimal
    assert result.stop_reason == "iteration_limit"
    assert result.lower_bound <= optimum + 1e-8
    assert result.upper_bound >= optimum - 1e-8
    assert result.objective == pytest.approx(oracle[tuple(result.support)], abs=1e-8)


@pytest.mark.parametrize("kind", ["unequal_shapes", "empty_times", "nonfinite_p", "nonfinite_q", "nonfinite_d"])
def test_invalid_observation_data_is_rejected(kind):
    scenarios = _case()
    if kind == "unequal_shapes":
        scenarios[0]["Q_terminal"] = np.zeros((2, 3))
    elif kind == "empty_times":
        for key in ("P_terminal", "Q_terminal", "drop_target"):
            scenarios[0][key] = np.zeros((0, 3))
    else:
        key = {"nonfinite_p": "P_terminal", "nonfinite_q": "Q_terminal", "nonfinite_d": "drop_target"}[kind]
        scenarios[0][key][0, 0] = np.nan if kind == "nonfinite_p" else np.inf
    with pytest.raises(ValueError):
        BendersExtensionProblem(scenarios, (), r_upper_bound=1.0, x_upper_bound=1.0)


@pytest.mark.parametrize("r_bound,x_bound", [(0.0, 1.0), (1.0, -1.0), (np.inf, np.nan)])
def test_nonpositive_or_nonfinite_impedance_bounds_are_rejected(r_bound, x_bound):
    with pytest.raises(ValueError):
        BendersExtensionProblem(_case(), (), r_upper_bound=r_bound, x_upper_bound=x_bound)


@pytest.mark.parametrize("supports", [((0, 1), (1, 2)), ((0,), (0,)), ((),), ((3,),)])
def test_invalid_existing_support_families_are_rejected(supports):
    with pytest.raises(ValueError):
        BendersExtensionProblem(_case(), supports, r_upper_bound=1.0, x_upper_bound=1.0)


def test_iteration_budget_and_tolerances_require_valid_values():
    for kwargs in ({"max_iterations": 0}, {"max_iterations": -1}, {"max_iterations": 1.5},
                   {"absolute_tolerance": -1.0}, {"relative_tolerance": np.nan}):
        with pytest.raises(ValueError):
            solve_best_laminar_extension_benders(
                _case(), (), r_upper_bound=1.0, x_upper_bound=1.0, **kwargs,
            )


def test_dataframe_time_misalignment_is_rejected():
    pd = pytest.importorskip("pandas")
    scenarios = _case()
    for key in ("P_terminal", "Q_terminal", "drop_target"):
        scenarios[0][key] = pd.DataFrame(scenarios[0][key], columns=["a", "b", "c"])
    scenarios[0]["Q_terminal"].index = np.arange(100, 109)
    with pytest.raises(ValueError):
        BendersExtensionProblem(scenarios, (), r_upper_bound=1.0, x_upper_bound=1.0)


def test_fractional_generation_points_produce_global_tight_dual_cuts():
    supports = ((0, 1),)
    scenarios = _case(3, supports, (0, 1, 2), noise=0.04, lengths=(7, 9))
    problem = BendersExtensionProblem(scenarios, supports, r_upper_bound=0.5, x_upper_bound=0.35)
    # These RHS parameters need not be products of any z. Test the continuous
    # value function beyond integer y, including a plainly inconsistent matrix.
    rng = np.random.default_rng(927)
    points = [rng.uniform(0.03, 0.97, size=len(problem.pairs)) for _ in range(5)]
    points.append(np.array([0.10, 0.85, 0.65, 0.25, 0.90, 0.05]))
    fits = [problem.solve_subproblem(point) for point in points]
    integer_oracle = _exhaustive_objectives(scenarios, supports, 0.5, 0.35)
    for point, fit in zip(points, fits):
        assert fit.cut_constant + fit.cut_gradient @ point == pytest.approx(fit.objective, abs=2e-8)
        assert abs(fit.primal_dual_gap) < 2e-8
        for target, target_fit in zip(points, fits):
            cut_value = fit.cut_constant + fit.cut_gradient @ target
            assert cut_value <= target_fit.objective + 2e-8
        # An independently formulated fixed-support oracle also checks every
        # fractional-origin cut at every legal integer support.
        for support, objective in integer_oracle.items():
            cut_value = fit.cut_constant + fit.cut_gradient @ _y(problem, support)
            assert cut_value <= objective + 2e-8


def test_total_time_budget_can_expire_before_first_solver_call():
    result = solve_best_laminar_extension_benders(
        _case(), (), r_upper_bound=1.0, x_upper_bound=0.8, time_limit=1e-12,
    )
    assert result.stop_reason == "time_limit"
    assert not result.proven_optimal
    assert result.support is None
    assert result.objective is None
    assert result.iterations == 0
    assert result.cuts == []
    assert result.lower_bound == 0.0
    assert result.upper_bound == float("inf")
    assert result.absolute_gap is None


def test_interrupted_subproblem_never_generates_an_optimality_cut(monkeypatch):
    from types import SimpleNamespace
    import experiments.wzzt_benders_prototype as prototype

    calls = []

    def interrupted_lp(c, **kwargs):
        calls.append(len(c))
        # A finite incumbent and apparently perfect objective do not make
        # a status-1 solve optimal.
        return SimpleNamespace(status=1, message="injected LP time limit",
                               x=np.zeros(len(c)), fun=0.0)

    def forbidden_dual_cut(*args, **kwargs):
        pytest.fail("An interrupted LP must not be turned into an optimal cut")

    monkeypatch.setattr(prototype, "linprog", interrupted_lp)
    monkeypatch.setattr(prototype.BendersExtensionProblem, "_dual_cut", forbidden_dual_cut)
    result = solve_best_laminar_extension_benders(
        _case(), (), r_upper_bound=1.0, x_upper_bound=0.8, time_limit=60.0,
    )
    assert len(calls) == 1
    assert result.stop_reason == "subproblem_time_limit"
    assert not result.proven_optimal
    assert result.cuts == []
    assert result.iterations == 0
    assert result.objective is None
    assert result.support is None
    assert result.upper_bound == float("inf")
    assert result.absolute_gap is None
    assert result.trace[-1]["status"] == "subproblem_time_limit"


def test_inverted_bounds_above_requested_tolerance_cannot_certify_optimality(monkeypatch):
    problem_type = BendersExtensionProblem
    original_subproblem = problem_type.solve_subproblem
    original_master = problem_type.solve_master
    fitted_objectives = []
    master_calls = []

    def record_subproblem(self, y, time_limit=None):
        fit = original_subproblem(self, y, time_limit=time_limit)
        fitted_objectives.append(fit.objective)
        return fit

    def inject_second_master_bound(self, cuts, time_limit=None):
        result = original_master(self, cuts, time_limit=time_limit)
        master_calls.append(result)
        if len(master_calls) == 2:
            assert fitted_objectives and fitted_objectives[0] > 1e-3
            result.mip_dual_bound = fitted_objectives[0] + 5e-8
        return result

    monkeypatch.setattr(problem_type, "solve_subproblem", record_subproblem)
    monkeypatch.setattr(problem_type, "solve_master", inject_second_master_bound)
    tolerance = 1e-10
    result = solve_best_laminar_extension_benders(
        _case(3, (), (0, 2), noise=0.10, lengths=(10,)), (),
        r_upper_bound=1.0, x_upper_bound=0.8, max_iterations=10,
        absolute_tolerance=tolerance, relative_tolerance=0.0,
    )
    assert len(master_calls) == 2
    assert result.stop_reason == "inconsistent_bounds"
    assert not result.proven_optimal
    assert result.lower_bound > result.upper_bound + tolerance
    assert result.absolute_gap > tolerance
    assert result.absolute_gap == pytest.approx(abs(result.upper_bound - result.lower_bound), abs=1e-12)
