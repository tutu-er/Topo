"""Tests for exact laminar one-atom L1 MILP regression."""

from itertools import combinations

import networkx as nx
import numpy as np
import pytest

scipy_optimize = pytest.importorskip(
    "scipy.optimize", reason="requires scipy.optimize.milp/HiGHS"
)
if not hasattr(scipy_optimize, "milp"):
    pytest.skip("requires scipy.optimize.milp/HiGHS", allow_module_level=True)

from terminal_case33.data.small_terminal_lv import build_small_terminal_lv_case
from terminal_case33.estimation.laminar_l1_milp import (
    SolverDiagnostics,
    build_matrices_from_atoms,
    evaluate_l1_matrices,
    fit_laminar_l1_sensitivity,
    is_admissible_extension,
    is_laminar_family,
    normalize_support_pool,
    solver_diagnostics_prove_optimality,
    solve_best_laminar_extension_l1,
    solve_fixed_support_l1,
)
from terminal_case33.models.lin_distflow import build_reduced_sensitivity_matrices


def _scenario_from_atoms(
    *,
    n: int,
    supports: tuple[tuple[int, ...], ...],
    r_values: tuple[float, ...],
    x_values: tuple[float, ...],
    samples: int,
    seed: int,
    noise_scale: float = 0.0,
) -> dict:
    rng = np.random.default_rng(seed)
    p = rng.normal(size=(samples, n))
    q = rng.normal(size=(samples, n))
    r_matrix, x_matrix = build_matrices_from_atoms(
        n, supports, r_values, x_values
    )
    intercept = np.linspace(-0.2, 0.2, n)
    target = p @ r_matrix.T + q @ x_matrix.T + intercept
    if noise_scale:
        target += rng.laplace(0.0, noise_scale, size=target.shape)
    return {
        "name": "synthetic",
        "P_terminal": p,
        "Q_terminal": q,
        "drop_target": target,
    }


def _all_nonempty_supports(n: int):
    for size in range(1, n + 1):
        yield from combinations(range(n), size)


def test_laminar_predicate_and_atom_matrix_properties() -> None:
    family = ((0, 1, 2), (0, 1), (0,), (1,), (3,))
    assert is_laminar_family(family)
    assert not is_laminar_family(((0, 1), (1, 2)))

    r_matrix, x_matrix = build_matrices_from_atoms(
        4,
        family,
        (0.4, 0.3, 0.2, 0.1, 0.5),
        (0.2, 0.15, 0.1, 0.05, 0.25),
    )
    for matrix in (r_matrix, x_matrix):
        assert np.allclose(matrix, matrix.T)
        assert np.min(matrix) >= 0.0
        assert np.min(np.linalg.eigvalsh(matrix)) >= -1e-12
    assert np.isclose(r_matrix[0, 0], 0.9)
    assert np.isclose(r_matrix[0, 1], 0.7)
    assert np.isclose(r_matrix[0, 2], 0.4)


def test_singleton_changes_only_diagonal_and_atom_inputs_are_validated() -> None:
    r_matrix, x_matrix = build_matrices_from_atoms(4, ((2,),), (0.7,), (0.3,))
    expected_r = np.zeros((4, 4))
    expected_x = np.zeros((4, 4))
    expected_r[2, 2] = 0.7
    expected_x[2, 2] = 0.3
    assert np.array_equal(r_matrix, expected_r)
    assert np.array_equal(x_matrix, expected_x)

    with pytest.raises(ValueError, match="nonnegative"):
        build_matrices_from_atoms(4, ((0, 1),), (-0.1,), (0.2,))
    with pytest.raises(ValueError, match="duplicate"):
        build_matrices_from_atoms(4, ((0, 0),), (0.1,), (0.2,))
    with pytest.raises(ValueError, match="terminal range"):
        build_matrices_from_atoms(4, ((0, 4),), (0.1,), (0.2,))


def test_solver_certificate_requires_reported_gap_or_primal_dual_bound() -> None:
    common = {
        "status": 0,
        "success": True,
        "message": "optimal",
        "objective": 1.0,
        "dual_bound": None,
        "mip_gap": None,
        "node_count": 0,
        "runtime_seconds": 0.0,
        "variable_count": 3,
        "binary_variable_count": 1,
        "constraint_count": 2,
    }
    assert not solver_diagnostics_prove_optimality(SolverDiagnostics(**common))
    assert solver_diagnostics_prove_optimality(
        SolverDiagnostics(**{**common, "mip_gap": 0.0})
    )
    assert solver_diagnostics_prove_optimality(
        SolverDiagnostics(
            **{
                **common,
                "dual_bound": 1.0 - 5e-11,
                "mip_gap": 0.5,
            }
        )
    )
    assert not solver_diagnostics_prove_optimality(
        SolverDiagnostics(**{**common, "status": 1, "mip_gap": 0.0})
    )
    assert not solver_diagnostics_prove_optimality(
        SolverDiagnostics(
            **{
                **common,
                "dual_bound": 2.0,
                "mip_gap": 0.5,
            }
        )
    )
    assert not solver_diagnostics_prove_optimality(
        SolverDiagnostics(**{**common, "mip_gap": -1e-12})
    )
    assert solver_diagnostics_prove_optimality(
        SolverDiagnostics(
            **{
                **common,
                "binary_variable_count": 0,
                "dual_bound": None,
                "mip_gap": None,
            }
        )
    )


def test_exact_single_atom_recovers_support_diagonal_and_values() -> None:
    scenario = _scenario_from_atoms(
        n=4,
        supports=((0, 2, 3),),
        r_values=(0.7,),
        x_values=(0.3,),
        samples=64,
        seed=11,
    )
    solution = solve_best_laminar_extension_l1(
        [scenario],
        (),
        r_upper_bound=2.0,
        x_upper_bound=2.0,
        mip_rel_gap=0.0,
    )
    assert solution.diagnostics.status == 0
    assert solver_diagnostics_prove_optimality(solution.diagnostics)
    assert solution.support == (0, 2, 3)
    assert np.isclose(solution.r_values[-1], 0.7, atol=1e-8)
    assert np.isclose(solution.x_values[-1], 0.3, atol=1e-8)
    assert solution.objective is not None and solution.objective < 1e-9

    r_matrix, x_matrix = build_matrices_from_atoms(
        4, (solution.support,), solution.r_values[-1:], solution.x_values[-1:]
    )
    expected_r, expected_x = build_matrices_from_atoms(
        4, ((0, 2, 3),), (0.7,), (0.3,)
    )
    assert np.allclose(r_matrix, expected_r, atol=1e-8)
    assert np.allclose(x_matrix, expected_x, atol=1e-8)
    assert r_matrix[0, 0] == r_matrix[0, 2] == r_matrix[2, 3]


def test_finite_candidate_pool_is_canonical_and_exactly_enforced() -> None:
    assert normalize_support_pool(((2, 0), (0, 2), (1, 2)), 3) == (
        (0, 2),
        (1, 2),
    )
    scenario = _scenario_from_atoms(
        n=4,
        supports=((0, 2, 3),),
        r_values=(0.7,),
        x_values=(0.3,),
        samples=48,
        seed=13,
    )
    pool = ((0, 1), (0, 2, 3), (1, 3))

    solution = solve_best_laminar_extension_l1(
        [scenario],
        (),
        candidate_supports=pool,
        r_upper_bound=2.0,
        x_upper_bound=2.0,
        mip_rel_gap=0.0,
    )

    assert solver_diagnostics_prove_optimality(solution.diagnostics)
    assert solution.support == (0, 2, 3)
    assert solution.support in pool


def test_finite_candidate_pool_reports_exhaustion_after_laminar_filter() -> None:
    scenario = _scenario_from_atoms(
        n=3,
        supports=((0, 1),),
        r_values=(0.5,),
        x_values=(0.2,),
        samples=24,
        seed=15,
    )
    solution = solve_best_laminar_extension_l1(
        [scenario],
        ((0, 1),),
        candidate_supports=((0, 1), (1, 2)),
        r_upper_bound=2.0,
        x_upper_bound=2.0,
    )

    assert solution.diagnostics.status == 2
    assert solution.support is None


def test_milp_objective_matches_exhaustive_subset_oracle() -> None:
    existing = ((0, 1, 2),)
    scenario = _scenario_from_atoms(
        n=4,
        supports=((0, 1, 2), (0, 1)),
        r_values=(0.25, 0.8),
        x_values=(0.15, 0.35),
        samples=40,
        seed=17,
        noise_scale=0.002,
    )
    milp_solution = solve_best_laminar_extension_l1(
        [scenario],
        existing,
        r_upper_bound=2.0,
        x_upper_bound=2.0,
        mip_rel_gap=0.0,
    )
    assert milp_solution.diagnostics.status == 0
    assert milp_solution.objective is not None

    candidates: list[tuple[tuple[int, ...], float]] = []
    for candidate in _all_nonempty_supports(4):
        if not is_admissible_extension(candidate, existing, 4):
            continue
        fit = solve_fixed_support_l1(
            [scenario],
            (*existing, candidate),
            r_upper_bound=2.0,
            x_upper_bound=2.0,
        )
        candidates.append((candidate, fit.objective))
    best_objective = min(value for _, value in candidates)
    best_supports = {
        support for support, value in candidates if value <= best_objective + 1e-9
    }
    assert np.isclose(milp_solution.objective, best_objective, atol=1e-9)
    assert milp_solution.support in best_supports
    assert solver_diagnostics_prove_optimality(milp_solution.diagnostics)

    r_matrix, x_matrix = build_matrices_from_atoms(
        4,
        (*existing, milp_solution.support),
        milp_solution.r_values,
        milp_solution.x_values,
    )
    p = np.asarray(scenario["P_terminal"], dtype=float)
    q = np.asarray(scenario["Q_terminal"], dtype=float)
    target = np.asarray(scenario["drop_target"], dtype=float)
    prediction = (
        p @ r_matrix.T
        + q @ x_matrix.T
        + milp_solution.intercepts[0][None, :]
    )
    assert np.isclose(
        np.mean(np.abs(target - prediction)),
        milp_solution.objective,
        atol=1e-9,
    )


def test_crossing_extension_is_excluded() -> None:
    existing = ((0, 1),)
    scenario = _scenario_from_atoms(
        n=3,
        supports=((0, 1), (1, 2)),
        r_values=(0.1, 1.0),
        x_values=(0.05, 0.4),
        samples=48,
        seed=23,
    )
    solution = solve_best_laminar_extension_l1(
        [scenario],
        existing,
        r_upper_bound=2.0,
        x_upper_bound=2.0,
    )
    assert solution.diagnostics.status == 0
    assert solution.support != (1, 2)
    assert solution.support is not None
    assert is_admissible_extension(solution.support, existing, 3)


def test_no_admissible_extension_reports_infeasible_without_support() -> None:
    scenario = _scenario_from_atoms(
        n=1,
        supports=((0,),),
        r_values=(0.4,),
        x_values=(0.2,),
        samples=8,
        seed=27,
    )
    solution = solve_best_laminar_extension_l1(
        [scenario],
        ((0,),),
        r_upper_bound=1.0,
        x_upper_bound=1.0,
    )
    assert solution.diagnostics.status == 2
    assert not solution.diagnostics.success
    assert solution.support is None
    assert solution.objective is None


def test_deterministic_three_atom_path_is_genuinely_fully_corrective() -> None:
    n = 4
    identity = np.eye(n)
    zero = np.zeros_like(identity)
    p = np.vstack([identity, -identity, zero, zero])
    q = np.vstack([zero, zero, identity, -identity])
    supports = ((0, 1, 2, 3), (0, 1, 2), (0,))
    r_values = (10.0, 2.0, 0.5)
    x_values = (8.0, 1.5, 0.4)
    r_true, x_true = build_matrices_from_atoms(
        n, supports, r_values, x_values
    )
    scenario = {
        "name": "deterministic_hierarchy",
        "P_terminal": p,
        "Q_terminal": q,
        "drop_target": p @ r_true.T + q @ x_true.T,
    }
    result = fit_laminar_l1_sensitivity(
        [scenario],
        max_atoms=3,
        r_upper_bound=20.0,
        x_upper_bound=20.0,
        max_bound_expansions=0,
        improvement_abs_tol=1e-12,
        improvement_rel_tol=1e-12,
    )
    assert result.path[1].support_indices == (supports[0],)
    assert result.path[2].support_indices == supports[:2]
    assert result.path[3].support_indices == supports
    assert np.allclose(result.path[1].r_values, (12.0,), atol=1e-8)
    assert np.allclose(result.path[1].x_values, (9.5,), atol=1e-8)
    assert np.allclose(result.path[2].r_values, (10.0, 2.0), atol=1e-8)
    assert np.allclose(result.path[2].x_values, (8.0, 1.5), atol=1e-8)
    assert np.allclose(result.r_values, r_values, atol=1e-8)
    assert np.allclose(result.x_values, x_values, atol=1e-8)
    assert np.allclose(result.r_matrix, r_true, atol=1e-8)
    assert np.allclose(result.x_matrix, x_true, atol=1e-8)
    assert result.train_mae < 1e-10


def test_forward_path_is_fully_corrective_monotone_and_laminar() -> None:
    scenario = _scenario_from_atoms(
        n=4,
        supports=((0, 1, 2, 3), (0, 1), (0,), (1,), (2,), (3,)),
        r_values=(0.15, 0.55, 0.20, 0.25, 0.30, 0.35),
        x_values=(0.08, 0.25, 0.10, 0.12, 0.14, 0.16),
        samples=72,
        seed=29,
        noise_scale=0.001,
    )
    result = fit_laminar_l1_sensitivity(
        [scenario],
        max_atoms=6,
        r_upper_bound=2.0,
        x_upper_bound=2.0,
        improvement_abs_tol=1e-9,
    )
    objectives = np.asarray([point.train_mae for point in result.path])
    assert np.all(np.diff(objectives) <= 1e-9)
    assert all(is_laminar_family(point.support_indices) for point in result.path)
    for point in result.path:
        fixed = solve_fixed_support_l1(
            [scenario],
            point.support_indices,
            r_upper_bound=2.0,
            x_upper_bound=2.0,
        )
        assert np.isclose(fixed.objective, point.train_mae, atol=1e-9)
        assert np.allclose(fixed.r_values, point.r_values, atol=1e-8)
        assert np.allclose(fixed.x_values, point.x_values, atol=1e-8)
        for matrix in (point.r_matrix, point.x_matrix):
            assert np.allclose(matrix, matrix.T)
            assert np.min(matrix) >= -1e-10
            assert np.min(np.linalg.eigvalsh(matrix)) >= -1e-8
    assert all(
        solver_diagnostics_prove_optimality(attempt.diagnostics)
        for attempt in result.attempted_extensions
    )


def test_validation_path_reuses_training_intercept() -> None:
    n = 2
    p_train = np.zeros((6, n))
    q_train = np.zeros((6, n))
    train_level = np.array([1.0, -2.0])
    validation_shift = np.array([7.0, -5.0])
    training = {
        "name": "same_scenario",
        "P_terminal": p_train,
        "Q_terminal": q_train,
        "drop_target": np.tile(train_level, (6, 1)),
    }
    validation = {
        "name": "same_scenario",
        "P_terminal": np.zeros((4, n)),
        "Q_terminal": np.zeros((4, n)),
        "drop_target": np.tile(train_level + validation_shift, (4, 1)),
    }
    result = fit_laminar_l1_sensitivity(
        [training],
        validation_scenarios=[validation],
        max_atoms=0,
        r_upper_bound=1.0,
        x_upper_bound=1.0,
    )
    fixed_mae, _, _ = evaluate_l1_matrices(
        [validation],
        result.r_matrix,
        result.x_matrix,
        fixed_intercepts=result.intercepts,
    )
    profiled_mae, _, _ = evaluate_l1_matrices(
        [validation], result.r_matrix, result.x_matrix
    )
    assert np.allclose(result.intercepts[0], train_level, atol=1e-9)
    assert np.isclose(fixed_mae, np.mean(np.abs(validation_shift)))
    assert np.isclose(result.validation_mae, fixed_mae)
    assert profiled_mae < 1e-12


def test_zero_signal_stops_without_accepting_an_atom() -> None:
    rng = np.random.default_rng(41)
    scenario = {
        "name": "zero_signal",
        "P_terminal": rng.normal(size=(16, 3)),
        "Q_terminal": rng.normal(size=(16, 3)),
        "drop_target": np.tile(np.array([0.4, -0.2, 0.1]), (16, 1)),
    }
    result = fit_laminar_l1_sensitivity(
        [scenario],
        max_atoms=3,
        r_upper_bound=2.0,
        x_upper_bound=2.0,
        improvement_abs_tol=1e-10,
        improvement_rel_tol=1e-10,
    )
    assert result.support_indices == ()
    assert len(result.path) == 1
    assert result.stop_reason == "no_significant_one_atom_gain"
    assert len(result.attempted_extensions) == 1
    assert solver_diagnostics_prove_optimality(
        result.attempted_extensions[0].diagnostics
    )


def test_initial_support_is_fitted_and_cannot_be_pruned() -> None:
    scenario = _scenario_from_atoms(
        n=3,
        supports=((0, 1),),
        r_values=(0.7,),
        x_values=(0.3,),
        samples=48,
        seed=143,
        noise_scale=0.0,
    )
    result = fit_laminar_l1_sensitivity(
        [scenario],
        initial_supports=((0, 1),),
        max_atoms=1,
        r_upper_bound=2.0,
        x_upper_bound=2.0,
    )
    assert result.support_indices == ((0, 1),)
    assert result.path[0].iteration == 1
    assert result.attempted_extensions == ()
    assert np.allclose(result.r_values, (0.7,), atol=1e-8)
    assert np.allclose(result.x_values, (0.3,), atol=1e-8)
    assert result.train_mae < 1e-10


def test_initial_supports_must_be_laminar_and_fit_atom_budget() -> None:
    scenario = _scenario_from_atoms(
        n=3,
        supports=((0, 1),),
        r_values=(0.2,),
        x_values=(0.1,),
        samples=16,
        seed=149,
        noise_scale=0.0,
    )
    with pytest.raises(ValueError, match="laminar"):
        fit_laminar_l1_sensitivity(
            [scenario],
            initial_supports=((0, 1), (1, 2)),
            max_atoms=2,
            r_upper_bound=1.0,
            x_upper_bound=1.0,
        )
    with pytest.raises(ValueError, match="cannot exceed"):
        fit_laminar_l1_sensitivity(
            [scenario],
            initial_supports=((0,), (1,)),
            max_atoms=1,
            r_upper_bound=1.0,
            x_upper_bound=1.0,
        )


def test_small6_physical_edge_atoms_reconstruct_exact_sensitivity() -> None:
    net = build_small_terminal_lv_case()
    terminals = (
        net.buses.loc[
            net.buses["bus_type"].eq("observed_terminal"), "bus_id"
        ]
        .astype(int)
        .tolist()
    )
    terminal_set = set(terminals)
    terminal_position = {terminal: index for index, terminal in enumerate(terminals)}
    graph = net.to_networkx_graph()
    oriented = nx.bfs_tree(graph, net.root_bus)
    impedance_base = net.base_kv**2 / net.base_mva
    grouped: dict[tuple[int, ...], list[float]] = {}
    for parent, child in oriented.edges():
        descendants = {
            int(child),
            *(int(node) for node in nx.descendants(oriented, child)),
        }
        support_labels = tuple(sorted(descendants & terminal_set))
        if not support_labels:
            continue
        support = tuple(terminal_position[label] for label in support_labels)
        values = grouped.setdefault(support, [0.0, 0.0])
        values[0] += 2.0 * float(graph.edges[parent, child]["r_ohm"]) / impedance_base
        values[1] += 2.0 * float(graph.edges[parent, child]["x_ohm"]) / impedance_base

    supports = tuple(grouped)
    reconstructed_r, reconstructed_x = build_matrices_from_atoms(
        len(terminals),
        supports,
        tuple(grouped[support][0] for support in supports),
        tuple(grouped[support][1] for support in supports),
    )
    true_r, true_x = build_reduced_sensitivity_matrices(
        net, terminals, voltage_model="squared-voltage"
    )
    assert is_laminar_family(supports)
    assert np.allclose(reconstructed_r, true_r, atol=1e-12)
    assert np.allclose(reconstructed_x, true_x, atol=1e-12)
