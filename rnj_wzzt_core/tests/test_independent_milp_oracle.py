"""Independent, tiny-instance LP oracles for the bounded laminar estimator.

The oracle uses a dense equality formulation with positive/negative residuals,
``scipy.optimize.linprog`` and Python-set enumeration. It deliberately does not
call the production atom builder, fixed-support solver, or laminar predicate.
HiGHS is shared, so this checks formulations and search domains, not an
independent numerical optimizer. Random cases have fixed seeds and small n.
"""

from dataclasses import replace
from itertools import combinations

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import linprog

from rnj_wzzt.estimation import laminar_l1_milp as core


def _supports(n):
    return tuple(s for size in range(1, n + 1) for s in combinations(range(n), size))


def _laminar(family):
    sets = [set(s) for s in family]
    return all(not a & b or a <= b or b <= a for a, b in combinations(sets, 2))


def _matrix(n, supports, weights):
    return np.array([
        [sum(w for s, w in zip(supports, weights) if i in s and j in s)
         for j in range(n)]
        for i in range(n)
    ])


def _scenarios(n, supports, r, x, *, seed=1, noise=0.0, sizes=(13,), mode="normal"):
    rng = np.random.default_rng(seed)
    resistance = _matrix(n, supports, r)
    reactance = _matrix(n, supports, x)
    result = []
    for k, samples in enumerate(sizes):
        p = rng.normal(size=(samples, n))
        q = rng.normal(size=(samples, n))
        if mode == "collinear":
            q = p.copy()
        elif mode == "no_q":
            q[:] = 0.0
        elif mode == "constant":
            p[:] = 0.4
            q[:] = -0.7
        target = p @ resistance + q @ reactance
        target += rng.laplace(0, noise, size=target.shape)
        result.append({"name": f"scenario_{k}", "P_terminal": p,
                       "Q_terminal": q, "drop_target": target})
    return result


def _lp_oracle(scenarios, family, r_bound=2.0, x_bound=2.0):
    """Minimize sum(u+v)/N subject to F theta + u - v = y."""
    n = scenarios[0]["P_terminal"].shape[1]
    atom_count = len(family)
    rows, targets = [], []
    for scenario in scenarios:
        p, q, y = (scenario[key] for key in ("P_terminal", "Q_terminal", "drop_target"))
        for t in range(len(p)):
            for output in range(n):
                row = [sum(p[t, j] for j in support) if output in support else 0.0
                       for support in family]
                row += [sum(q[t, j] for j in support) if output in support else 0.0
                        for support in family]
                rows.append(row)
                targets.append(y[t, output])
    design = np.asarray(rows)
    count, coefficients = design.shape
    objective = np.r_[np.zeros(coefficients), np.full(2 * count, 1.0 / count)]
    bounds = ([(0.0, r_bound)] * atom_count + [(0.0, x_bound)] * atom_count
              + [(0.0, None)] * (2 * count))
    solved = linprog(objective, A_eq=np.c_[design, np.eye(count), -np.eye(count)],
                     b_eq=targets, bounds=bounds, method="highs-ds")
    assert solved.success, solved.message
    direct_mae = np.mean(np.abs(design @ solved.x[:coefficients] - targets))
    assert direct_mae == pytest.approx(solved.fun, abs=2e-8)
    return float(solved.fun)


def _extension_oracle(scenarios, initial, r_bound=2.0, x_bound=2.0):
    n = scenarios[0]["P_terminal"].shape[1]
    return {s: _lp_oracle(scenarios, (*initial, s), r_bound, x_bound)
            for s in _supports(n) if s not in initial and _laminar((*initial, s))}


def _direct_mae(scenarios, family, solution):
    n = scenarios[0]["P_terminal"].shape[1]
    r = _matrix(n, family, solution.r_values)
    x = _matrix(n, family, solution.x_values)
    residuals = [np.abs(s["drop_target"] - s["P_terminal"] @ r - s["Q_terminal"] @ x).ravel()
                 for s in scenarios]
    return float(np.mean(np.concatenate(residuals)))


FIXED_FAMILIES = [
    (1, ()), (1, ((0,),)), (3, ((0,), (1,), (2,))),
    (3, ((0, 1, 2), (0, 1), (0,))), (4, ((0, 1), (2, 3), (0,))),
]


@pytest.mark.parametrize("n,family", FIXED_FAMILIES)
@pytest.mark.parametrize("noise", [0.0, 0.03, 0.5])
@pytest.mark.parametrize("seed", [1701, 1702])
def test_fixed_family_matches_independent_residual_split_lp(n, family, noise, seed):
    count = len(family)
    scenarios = _scenarios(n, family, np.linspace(0.15, 0.7, count),
                          np.linspace(0.5, 0.1, count), seed=seed, noise=noise, sizes=(7, 11))
    solved = core.solve_fixed_support_l1(scenarios, family, r_upper_bound=2, x_upper_bound=2)
    expected = _lp_oracle(scenarios, family)
    assert solved.objective == pytest.approx(expected, abs=3e-8)
    assert _direct_mae(scenarios, family, solved) == pytest.approx(expected, abs=3e-8)


@pytest.mark.parametrize("initial", [(), ((0,), (1,), (2,)), ((0, 1, 2),), ((0, 1),)])
@pytest.mark.parametrize("mode", ["normal", "collinear", "no_q"])
@pytest.mark.parametrize("noise", [0.0, 0.06])
@pytest.mark.parametrize("seed", [1801, 1802])
def test_extension_matches_independent_enumeration(initial, mode, noise, seed):
    scenarios = _scenarios(3, ((0, 1, 2), (1, 2), (0,)), (.15, .6, .2), (.2, .35, .1),
                          seed=seed, noise=noise, sizes=(9,), mode=mode)
    expected = _extension_oracle(scenarios, initial)
    solved = core.solve_best_laminar_extension_l1(
        scenarios, initial, r_upper_bound=2, x_upper_bound=2,
        time_limit=20, mip_rel_gap=0)
    assert core.solver_diagnostics_prove_optimality(solved.diagnostics)
    best = min(expected.values())
    assert solved.objective == pytest.approx(best, abs=5e-8)
    assert solved.support in expected
    assert expected[solved.support] == pytest.approx(best, abs=5e-8)
    assert _direct_mae(scenarios, (*initial, solved.support), solved) == pytest.approx(best, abs=5e-8)


@pytest.mark.parametrize("mode", ["collinear", "no_q", "constant"])
@pytest.mark.parametrize("noise", [0.0, .02])
def test_rank_deficiency_certifies_prediction_not_separate_rx_identifiability(mode, noise):
    family = ((0,), (1,), (0, 1))
    scenarios = _scenarios(2, family, (.3, .5, .7), (.6, .2, .4),
                          mode=mode, noise=noise, sizes=(15,))
    solved = core.solve_fixed_support_l1(scenarios, family, r_upper_bound=2, x_upper_bound=2)
    assert solved.objective == pytest.approx(_lp_oracle(scenarios, family), abs=2e-8)
    if noise == 0:
        assert _direct_mae(scenarios, family, solved) < 1e-8
    if mode == "collinear":
        r = np.array([.3, .5, .7])
        x = np.array([.6, .2, .4])
        p, q = (scenarios[0][key] for key in ("P_terminal", "Q_terminal"))
        assert np.allclose(p @ _matrix(2, family, r) + q @ _matrix(2, family, x),
                           p @ _matrix(2, family, r + .1) + q @ _matrix(2, family, x - .1))


@pytest.mark.parametrize("r_bound,x_bound", [(.1, .2), (.4, .5), (2., 2.)])
def test_explicit_coefficient_domain_matches_oracle(r_bound, x_bound):
    scenarios = _scenarios(3, ((0, 2),), (.7,), (.6,), sizes=(12,))
    solved = core.solve_best_laminar_extension_l1(
        scenarios, (), r_upper_bound=r_bound, x_upper_bound=x_bound)
    expected = _extension_oracle(scenarios, (), r_bound=r_bound, x_bound=x_bound)
    assert solved.objective == pytest.approx(min(expected.values()), abs=3e-8)
    assert np.all(solved.r_values <= r_bound + 1e-8)
    assert np.all(solved.x_values <= x_bound + 1e-8)


def test_unequal_scenario_lengths_use_observation_weighted_mae():
    short = {"name": "short", "P_terminal": np.zeros((3, 1)), "Q_terminal": np.zeros((3, 1)),
             "drop_target": np.array([[0.], [1.], [100.]])}
    long = {"name": "long", "P_terminal": np.zeros((21, 1)), "Q_terminal": np.zeros((21, 1)),
            "drop_target": np.zeros((21, 1))}
    solved = core.solve_fixed_support_l1([short, long], (), r_upper_bound=1, x_upper_bound=1)
    assert solved.objective == pytest.approx(101.0 / 24)
    assert solved.objective == pytest.approx(_lp_oracle([short, long], ()))
    assert solved.objective != pytest.approx((101.0 / 3) / 2)


def test_false_fixed_prior_blocks_true_clade_while_free_search_recovers_it():
    true_support, wrong_support = (1, 2), (0, 1)
    scenarios = _scenarios(3, (true_support,), (.8,), (.4,), seed=1901, sizes=(25,))
    hard = core.fit_laminar_l1_sensitivity(
        scenarios, initial_supports=(wrong_support,), max_atoms=3,
        r_upper_bound=2, x_upper_bound=2)
    free = core.fit_laminar_l1_sensitivity(
        scenarios, max_atoms=3,
        r_upper_bound=2, x_upper_bound=2)
    assert all(wrong_support in point.support_indices for point in hard.path)
    assert true_support not in hard.support_indices
    assert hard.train_mae > .05
    assert free.train_mae < 1e-8
    assert free.support_indices == (true_support,)


def test_extension_searches_all_subsets_without_candidate_input():
    scenarios = _scenarios(3, ((0, 2),), (.8,), (.4,), sizes=(20,))
    solved = core.solve_best_laminar_extension_l1(
        scenarios, (), r_upper_bound=2, x_upper_bound=2)
    expected = _extension_oracle(scenarios, ())
    assert core.solver_diagnostics_prove_optimality(solved.diagnostics)
    assert solved.support == (0, 2)
    assert solved.objective == pytest.approx(min(expected.values()), abs=1e-8)
    assert solved.objective < 1e-8
    assert min(loss for support, loss in expected.items() if support != (0, 2)) > .05


@pytest.mark.parametrize("shift", [-2.0, .5, 3.0])
def test_heldout_offset_is_not_refitted_and_changes_validation_loss(shift):
    scenarios = _scenarios(2, ((0, 1),), (.6,), (.3,), sizes=(24,))
    solved = core.solve_fixed_support_l1(scenarios, ((0, 1),), r_upper_bound=2, x_upper_bound=2)
    heldout = [{**scenarios[0], "drop_target": scenarios[0]["drop_target"] + shift}]
    r = _matrix(2, solved.supports, solved.r_values)
    x = _matrix(2, solved.supports, solved.x_values)
    mae, _, _ = core.evaluate_l1_matrices(heldout, r, x)
    assert mae == pytest.approx(abs(shift), abs=1e-8)


def test_constant_scenarios_identify_physical_slopes_without_centering():
    scenarios = [
        {"name": "p_only", "P_terminal": np.ones((4, 1)), "Q_terminal": np.zeros((4, 1)),
         "drop_target": np.full((4, 1), .7)},
        {"name": "q_only", "P_terminal": np.zeros((5, 1)), "Q_terminal": np.ones((5, 1)),
         "drop_target": np.full((5, 1), .3)},
    ]
    assert core.estimate_atom_upper_bounds(scenarios) == pytest.approx((1.4, .6))
    solved = core.solve_fixed_support_l1(scenarios, ((0,),), r_upper_bound=2, x_upper_bound=2)
    assert solved.r_values == pytest.approx([.7])
    assert solved.x_values == pytest.approx([.3])
    assert solved.objective < 1e-10


def test_validation_can_use_different_scenario_count_names_and_operating_points():
    train = _scenarios(2, ((0, 1),), (.8,), (.4,), sizes=(15,), seed=1971)
    validation = _scenarios(2, ((0, 1),), (.8,), (.4,), sizes=(7, 11, 9), seed=1972)
    for i, scenario in enumerate(validation):
        scenario["name"] = f"new_operating_point_{i}"
    solved = core.fit_laminar_l1_sensitivity(
        train, validation_scenarios=validation, max_atoms=1, r_upper_bound=2, x_upper_bound=2,
    )
    assert solved.support_indices == ((0, 1),)
    assert solved.validation_mae < 1e-8
    mae, _, _ = core.evaluate_l1_matrices(validation, solved.r_matrix, solved.x_matrix)
    assert mae == pytest.approx(solved.validation_mae, abs=1e-12)


def test_regular_extension_reuses_joint_milp_weights_without_another_lp(monkeypatch):
    initial = ((0,), (1,), (2,))
    family = (*initial, (0, 1))
    scenarios = _scenarios(3, family, (.2, .3, .4, .6), (.1, .2, .1, .4), sizes=(16,))
    actual = core._run_milp
    solve_kinds = []

    def counted(*args, **kwargs):
        result, diagnostics = actual(*args, **kwargs)
        solve_kinds.append("MILP" if diagnostics.binary_variable_count else "LP")
        return result, diagnostics

    monkeypatch.setattr(core, "_run_milp", counted)
    solved = core.fit_laminar_l1_sensitivity(
        scenarios, initial_supports=initial, max_atoms=4, r_upper_bound=2, x_upper_bound=2,
    )
    assert solved.support_indices == family
    assert solved.train_mae < 1e-8
    assert solve_kinds == ["LP", "MILP"]
    assert solved.r_values == pytest.approx(solved.attempted_extensions[0].r_values)
    assert solved.x_values == pytest.approx(solved.attempted_extensions[0].x_values)


def test_inconsistent_physical_milp_prediction_is_rejected(monkeypatch):
    scenarios = _scenarios(2, ((0, 1),), (.6,), (.3,), sizes=(16,))
    actual = core._run_milp

    def corrupted(variables, *args, **kwargs):
        result, diagnostics = actual(variables, *args, **kwargs)
        if diagnostics.binary_variable_count:
            result.x[variables.slices["new_r"]] += .1
        return result, diagnostics

    monkeypatch.setattr(core, "_run_milp", corrupted)
    with pytest.raises(RuntimeError, match="physical prediction"):
        core.solve_best_laminar_extension_l1(scenarios, (), r_upper_bound=2, x_upper_bound=2)


@pytest.mark.parametrize("extension", [False, True])
@pytest.mark.parametrize("truth,returned,expected", [
    (0.0, -3.2455e-13, 0.0), (2.0, 2.0 + 3e-13, 2.0),
    (0.0, -1e-3, None), (2.0, 2.01, None),
])
def test_solver_weight_extraction_repairs_only_boundary_roundoff(
    monkeypatch, extension, truth, returned, expected,
):
    scenarios = _scenarios(1, ((0,),), (truth,), (.4,), sizes=(16,))
    actual = core._run_milp

    def perturbed(variables, *args, **kwargs):
        result, diagnostics = actual(variables, *args, **kwargs)
        result.x[variables.slices["new_r" if extension else "r"]] = returned
        return result, diagnostics

    monkeypatch.setattr(core, "_run_milp", perturbed)
    solve = core.solve_best_laminar_extension_l1 if extension else core.solve_fixed_support_l1
    family = () if extension else ((0,),)
    if expected is None:
        with pytest.raises(RuntimeError, match="coefficients.*bounds"):
            solve(scenarios, family, r_upper_bound=2, x_upper_bound=2)
    else:
        solved = solve(scenarios, family, r_upper_bound=2, x_upper_bound=2)
        assert solved.r_values == pytest.approx([expected], abs=1e-14)
        r, x = core.build_matrices_from_atoms(1, ((0,),), solved.r_values, solved.x_values)
        mae, _, _ = core.evaluate_l1_matrices(scenarios, r, x)
        assert mae < 1e-8


def test_public_milp_apis_align_shuffled_time_labels_before_fitting_and_evaluation():
    index = pd.Index(["t1", "t0", "t3", "t2"])
    p = pd.DataFrame({50: [1., 2., 3., 4.]}, index=index)
    q = pd.DataFrame({50: [2., 1., 0., 3.]}, index=index)
    y = 2 * p + .5 * q
    scenarios = [{"P_terminal": p, "Q_terminal": q.iloc[[2, 0, 3, 1]],
                  "drop_target": y.iloc[::-1]}]
    for solve, family in ((core.solve_fixed_support_l1, ((0,),)),
                          (core.solve_best_laminar_extension_l1, ())):
        solved = solve(scenarios, family, r_upper_bound=3, x_upper_bound=3)
        assert solved.r_values == pytest.approx([2.])
        assert solved.x_values == pytest.approx([.5])
        assert solved.objective < 1e-10
    shifted = [{**scenarios[0], "drop_target": scenarios[0]["drop_target"] + .25}]
    mae, _, _ = core.evaluate_l1_matrices(shifted, np.array([[2.]]), np.array([[.5]]))
    assert mae == pytest.approx(.25)


@pytest.mark.parametrize("key,indices", [
    ("P_terminal", [0, 0, 2]),
    ("Q_terminal", [0, 1, 1]),
    ("drop_target", [0, 1, 3]),
    ("drop_target", [0, 1]),
])
def test_milp_preparation_rejects_ambiguous_or_mismatched_time_labels(key, indices):
    frame = pd.DataFrame({0: [1., 2., 3.]})
    scenario = {name: frame.copy() for name in ("P_terminal", "Q_terminal", "drop_target")}
    scenario[key] = pd.DataFrame({0: np.ones(len(indices))}, index=indices)
    with pytest.raises(ValueError, match="time indices"):
        core.solve_fixed_support_l1([scenario], ((0,),), r_upper_bound=2, x_upper_bound=2)


def test_validation_selects_base_when_training_signal_disappears():
    train = _scenarios(2, ((0, 1),), (.8,), (.4,), seed=1951, sizes=(32,))
    # Symmetric observations identify the physical slopes without any offset.
    for key in ("P_terminal", "Q_terminal"):
        train[0][key] = np.r_[train[0][key], -train[0][key]]
    train[0]["drop_target"] = train[0]["P_terminal"] @ np.full((2, 2), .8) + train[0]["Q_terminal"] @ np.full((2, 2), .4)
    validation = [{**train[0], "drop_target": np.zeros_like(train[0]["drop_target"])}]
    solved = core.fit_laminar_l1_sensitivity(
        train, validation_scenarios=validation,
        max_atoms=1, r_upper_bound=2, x_upper_bound=2)
    assert len(solved.path) == 2
    assert solved.path[-1].train_mae < 1e-8
    assert solved.selected_path_index == 0
    assert solved.support_indices == ()


@pytest.mark.parametrize("dual_mode", ["missing", "loose", "no_gain"])
def test_timeout_never_accepts_incumbent_and_only_valid_dual_bound_certifies_no_gain(monkeypatch, dual_mode):
    r, x = (0.0, 0.0) if dual_mode == "no_gain" else (.5, .3)
    scenarios = _scenarios(2, ((0, 1),), (r,), (x,), sizes=(12,))
    base = _lp_oracle(scenarios, ())
    actual_run = core._run_milp

    def limited(*args, **kwargs):
        result, diagnostics = actual_run(*args, **kwargs)
        if diagnostics.binary_variable_count:
            # Deliberately simulate a time-limited solve with an incumbent;
            # avoid brittle millisecond wall-clock races in CI.
            dual = {"missing": None, "loose": 0.0, "no_gain": base}[dual_mode]
            # The no-gain case represents a separate admissible solver outcome:
            # incumbent=base, lower bound=base, but solver status is still 1.
            objective = base if dual_mode == "no_gain" else diagnostics.objective
            result.fun = objective
            diagnostics = replace(diagnostics, status=1, success=False, objective=objective,
                                  dual_bound=dual, mip_gap=None, message="simulated time limit")
        return result, diagnostics

    monkeypatch.setattr(core, "_run_milp", limited)
    solved = core.fit_laminar_l1_sensitivity(scenarios, max_atoms=2, r_upper_bound=2, x_upper_bound=2)
    assert solved.support_indices == ()
    assert len(solved.path) == 1
    assert solved.attempted_extensions[0].support is None
    assert solved.attempted_extensions[0].objective is not None
    expected = ("no_significant_one_atom_gain_certified_by_dual_bound"
                if dual_mode == "no_gain" else "extension_not_proven_optimal")
    assert solved.stop_reason == expected


def test_exact_noiseless_full_rank_counterexample_to_global_two_atom_optimality():
    """Public-API feasible-domain witness, not a production-default feeder.

    This first minimal example has zero pendant weights for terminals 0/1,
    empty initialization, and a two-atom budget. The separate positive-pendant
    witness below tests frozen leaf supports under the unrestricted API.
    """
    scenarios = _scenarios(3, ((0, 1), (2,)), (.6, .9), (.3, .4),
                          seed=2152, sizes=(9,))
    scenario = scenarios[0]
    design = np.c_[scenario["P_terminal"], scenario["Q_terminal"]]
    assert np.linalg.matrix_rank(design) == 6
    first = _extension_oracle(scenarios, ())
    assert min(first, key=first.get) == (0, 1, 2)
    feasible_families = [()] + [(s,) for s in _supports(3)]
    feasible_families += [family for family in combinations(_supports(3), 2) if _laminar(family)]
    global_objective = min(_lp_oracle(scenarios, family) for family in feasible_families)
    greedy = core.fit_laminar_l1_sensitivity(
        scenarios, max_atoms=2, r_upper_bound=2, x_upper_bound=2)
    assert global_objective < 1e-8
    assert greedy.path[1].support_indices == ((0, 1, 2),)
    assert greedy.train_mae == pytest.approx(_lp_oracle(scenarios, greedy.support_indices), abs=2e-8)
    assert greedy.train_mae > global_objective + .1
    assert all(core.solver_diagnostics_prove_optimality(point.solver) for point in greedy.path)


@pytest.mark.parametrize("seed", [2201, 2202, 2203])
def test_every_corrected_path_point_matches_independent_lp(seed):
    initial = ((0,), (1,), (2,))
    scenarios = _scenarios(3, (*initial, (0, 1)), (.2, .3, .4, .6), (.3, .2, .1, .4),
                          seed=seed, noise=.03, sizes=(12, 8))
    result = core.fit_laminar_l1_sensitivity(
        scenarios, initial_supports=initial, max_atoms=5, r_upper_bound=2, x_upper_bound=2)
    objectives = []
    for point in result.path:
        expected = _lp_oracle(scenarios, point.support_indices)
        assert point.train_mae == pytest.approx(expected, abs=3e-8)
        assert _laminar(point.support_indices)
        assert set(initial) <= set(point.support_indices)
        objectives.append(expected)
    assert np.all(np.diff(objectives) <= 1e-8)


@pytest.mark.parametrize("seed", [2301, 2302, 2303])
def test_terminal_permutation_preserves_enumerated_extension_loss(seed):
    family = ((0, 2), (1,))
    scenarios = _scenarios(3, family, (.8, .3), (.4, .2), seed=seed, noise=.01, sizes=(13,))
    permutation = [2, 0, 1]
    permuted = [{key: value if key == "name" else value[:, permutation]
                 for key, value in scenario.items()} for scenario in scenarios]
    original = core.solve_best_laminar_extension_l1(
        scenarios, (), r_upper_bound=2, x_upper_bound=2)
    transformed = core.solve_best_laminar_extension_l1(
        permuted, (), r_upper_bound=2, x_upper_bound=2)
    mapped = tuple(sorted(permutation[i] for i in transformed.support))
    oracle = _extension_oracle(scenarios, ())
    assert transformed.objective == pytest.approx(original.objective, abs=3e-8)
    assert oracle[mapped] == pytest.approx(min(oracle.values()), abs=3e-8)


def test_inactive_hard_prior_remains_a_structural_constraint():
    scenarios = _scenarios(3, (), (), (), seed=2401, sizes=(14,))
    result = core.fit_laminar_l1_sensitivity(
        scenarios, initial_supports=((0, 1),), max_atoms=2, r_upper_bound=2, x_upper_bound=2)
    assert result.support_indices == ((0, 1),)
    assert np.max(result.r_values) < 1e-8
    assert np.max(result.x_values) < 1e-8
    assert result.train_mae < 1e-8
    assert not _laminar((*result.support_indices, (1, 2)))


@pytest.mark.parametrize("initial", [((0, 1), (2, 3)), ((0, 1, 2), (0, 1)),
                                     ((0, 1, 2, 3), (0,), (2, 3))])
@pytest.mark.parametrize("seed", [2501, 2502, 2503])
def test_four_terminal_nested_and_disjoint_constraints_match_all_subsets(initial, seed):
    scenarios = _scenarios(4, ((0, 1, 2, 3), (0, 2), (1,)), (.2, .5, .4), (.1, .3, .6),
                          seed=seed, noise=.03, sizes=(11,))
    expected = _extension_oracle(scenarios, initial)
    solved = core.solve_best_laminar_extension_l1(
        scenarios, initial, r_upper_bound=2, x_upper_bound=2)
    assert core.solver_diagnostics_prove_optimality(solved.diagnostics)
    assert solved.objective == pytest.approx(min(expected.values()), abs=5e-8)
    assert expected[solved.support] == pytest.approx(solved.objective, abs=5e-8)


@pytest.mark.parametrize("initial_bound", [.1, .25])
def test_bound_expansion_final_path_matches_final_domain_oracle(initial_bound):
    scenarios = _scenarios(2, ((0, 1),), (.7,), (.3,), sizes=(16,))
    solved = core.fit_laminar_l1_sensitivity(
        scenarios, max_atoms=1, r_upper_bound=initial_bound, x_upper_bound=initial_bound)
    assert solved.bound_expansions > 0
    assert solved.train_mae < 1e-8
    assert _lp_oracle(scenarios, solved.support_indices,
                      solved.r_upper_bound, solved.x_upper_bound) == pytest.approx(solved.train_mae, abs=1e-8)
    assert len(solved.path) == 2
    assert solved.r_upper_bound > .7
    assert solved.x_upper_bound > .3


def test_bound_expansion_limit_keeps_previously_certified_path():
    scenarios = _scenarios(2, ((0, 1),), (.7,), (.3,), sizes=(16,))
    solved = core.fit_laminar_l1_sensitivity(
        scenarios, max_atoms=1, r_upper_bound=.1, x_upper_bound=.1, max_bound_expansions=0)
    assert solved.stop_reason == "bound_expansion_limit"
    assert solved.support_indices == ()
    assert len(solved.path) == 1
    assert solved.train_mae == pytest.approx(_lp_oracle(scenarios, ()))
    assert solved.attempted_extensions[0].objective < solved.train_mae



def test_positive_pendants_and_fixed_leaf_initialization_do_not_prove_global_budget_optimum():
    """A complete-search witness with a finite two-extension budget.

    All true leaf edges are positive and all leaf supports are frozen. The
    finite budget permits two additional atoms. This does not claim failure
    of an unlimited path or an empirical AC feeder experiment.
    """
    initial = tuple((i,) for i in range(4))
    truth = (*initial, (0, 1), (2, 3))
    scenarios = _scenarios(4, truth, (.2, .3, .4, .5, .6, .8),
                          (.4, .3, .2, .1, .4, .3), seed=2712, sizes=(9,))
    scenario = scenarios[0]
    design = np.c_[scenario["P_terminal"], scenario["Q_terminal"]]
    assert np.linalg.matrix_rank(design) == 8
    all_extensions = _extension_oracle(scenarios, initial)
    assert min(all_extensions, key=all_extensions.get) == (0, 1, 2, 3)
    greedy = core.fit_laminar_l1_sensitivity(
        scenarios, initial_supports=initial, max_atoms=6,
        r_upper_bound=2, x_upper_bound=2)
    assert _lp_oracle(scenarios, truth) < 1e-8
    assert greedy.path[1].support_indices == (*initial, (0, 1, 2, 3))
    assert greedy.train_mae == pytest.approx(_lp_oracle(scenarios, greedy.support_indices), abs=2e-8)
    assert greedy.train_mae > .1
    assert all(set(initial) <= set(point.support_indices) for point in greedy.path)
    assert all(core.solver_diagnostics_prove_optimality(point.solver) for point in greedy.path)


def test_complete_search_can_refit_and_prune_a_provisional_root_stem():
    """A provisional root atom need not prevent later exact recovery.

    A single noisy-free, finite-sample linear scenario has correlated inputs
    and full-rank [P,Q]. All true pendant edges
    are positive, all singletons are frozen, and every nonempty subset is
    searched. The first greedy root atom becomes inactive after adding both
    true internal supports and can then be pruned without increasing loss.
    """
    initial = tuple((i,) for i in range(4))
    truth = (*initial, (0, 1), (2, 3))
    r = (.05, .1, .15, .2, .3, .4)
    x = (.2, .15, .1, .05, .25, .15)
    scenarios = _scenarios(4, truth, r, x, seed=3812, sizes=(9,))
    scenario = scenarios[0]
    rng = np.random.default_rng(503812)
    scenario["P_terminal"] = .5 * scenario["P_terminal"] + rng.normal(size=(9, 1))
    scenario["Q_terminal"] = .5 * scenario["Q_terminal"] + rng.normal(size=(9, 1))
    scenario["drop_target"] = (scenario["P_terminal"] @ _matrix(4, truth, r)
                               + scenario["Q_terminal"] @ _matrix(4, truth, x))
    design = np.c_[scenario["P_terminal"], scenario["Q_terminal"]]
    assert np.linalg.matrix_rank(design) == 8
    first = _extension_oracle(scenarios, initial)
    assert min(first, key=first.get) == (0, 1, 2, 3)
    greedy = core.fit_laminar_l1_sensitivity(
        scenarios, initial_supports=initial,
        r_upper_bound=2, x_upper_bound=2)
    # Zero is also the global lower bound of a nonnegative absolute loss.
    assert _lp_oracle(scenarios, truth) < 1e-8
    assert greedy.path[1].support_indices == (*initial, (0, 1, 2, 3))
    assert greedy.support_indices == truth
    assert greedy.train_mae < 1e-8
    for point in greedy.path:
        assert point.train_mae == pytest.approx(_lp_oracle(scenarios, point.support_indices), abs=2e-8)
    assert min(_extension_oracle(scenarios, greedy.support_indices).values()) < 1e-8
    assert all(core.solver_diagnostics_prove_optimality(point.solver) for point in greedy.path)
