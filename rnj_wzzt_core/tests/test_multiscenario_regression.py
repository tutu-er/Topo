"""Regression contracts and independent small constrained-LS reference cases."""

from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from rnj_wzzt.estimation.multiscenario import fit_projected_sensitivity


def _orthogonal_scenario(r_matrix, x_matrix, *, name="orthogonal"):
    """Use a centered design with A.T @ A = 2 I, independent of the fitter."""

    n = len(r_matrix)
    design = np.vstack([np.eye(2 * n), -np.eye(2 * n)])
    columns = [10 * (i + 1) for i in range(n)]
    index = pd.Index([f"sample_{i}" for i in range(len(design))])
    return {
        "name": name,
        "P_terminal": pd.DataFrame(design[:, :n], index=index, columns=columns),
        "Q_terminal": pd.DataFrame(design[:, n:], index=index, columns=columns),
        "drop_target": pd.DataFrame(
            design[:, :n] @ np.asarray(r_matrix).T
            + design[:, n:] @ np.asarray(x_matrix).T,
            index=index,
            columns=columns,
        ),
    }


def _squared_residual(scenario, r_matrix, x_matrix):
    p = scenario["P_terminal"].to_numpy()
    q = scenario["Q_terminal"].to_numpy()
    target = scenario["drop_target"].to_numpy()
    residual = p @ r_matrix.T + q @ x_matrix.T - target
    return float(np.sum(residual**2))


@pytest.mark.parametrize("mode", ["basic", "ordered"])
def test_active_constraint_matches_independent_analytic_optimum(mode):
    """The ordered optimum has all four R entries equal to 1/2.

    For the ordered problem, symmetry permits R=[[d,t],[t,d]], d>=t>=0.
    The objective is proportional to 2*d**2 + 2*(t-1)**2, whose minimum is
    d=t=1/2. The legacy tree_covariance path promises feasibility only and is
    deliberately excluded from this exact-optimality contract.
    """

    raw_r = np.array([[0.0, 1.0], [1.0, 0.0]])
    true_x = np.eye(2)
    scenario = _orthogonal_scenario(raw_r, true_x)
    r_hat, x_hat, _, _ = fit_projected_sensitivity(
        [scenario], constraint_mode=mode, diagonal_margin_ratio=0.0,
    )
    expected_r = raw_r if mode == "basic" else np.full((2, 2), 0.5)
    np.testing.assert_allclose(r_hat, expected_r, atol=2e-6)
    np.testing.assert_allclose(x_hat, true_x, atol=2e-6)
    expected_sse = 0.0 if mode == "basic" else 2.0
    assert _squared_residual(scenario, r_hat, x_hat) == pytest.approx(
        expected_sse, abs=2e-6,
    )


def test_ordered_margin_is_fixed_from_initial_ols_scale():
    scenario = _orthogonal_scenario(np.array([[1.0, 3.0], [3.0, 1.0]]), np.eye(2))
    r_hat, x_hat, _, _ = fit_projected_sensitivity(
        [scenario], constraint_mode="ordered", diagonal_margin_ratio=0.1,
    )
    # Initial OLS positive-diagonal median is 1, so the fixed margin is 0.1.
    # Minimize 2*(d-1)^2 + 2*(t-3)^2 with d=t+0.1: (d,t)=(2.05,1.95).
    np.testing.assert_allclose(r_hat, [[2.05, 1.95], [1.95, 2.05]], atol=2e-6)
    np.testing.assert_allclose(x_hat, np.eye(2), atol=2e-6)


@pytest.mark.parametrize("mode", ["basic", "ordered", "tree_covariance"])
def test_recovers_shared_matrices_with_distinct_scenario_power_means(mode):
    rng = np.random.default_rng(20260907)
    r_true = np.array([[2.0, 0.4], [0.4, 1.4]])
    x_true = np.array([[1.1, 0.2], [0.2, 0.8]])
    scenarios = []
    for i, samples in enumerate([37, 53]):
        p = rng.normal(size=(samples, 2)) + np.array([3.0, -2.0]) * i
        q = rng.normal(size=(samples, 2)) + np.array([-1.0, 4.0]) * i
        target = p @ r_true.T + q @ x_true.T
        scenarios.append({
            "name": f"scenario_{i}",
            "P_terminal": pd.DataFrame(p, columns=[10, 20]),
            "Q_terminal": pd.DataFrame(q, columns=[10, 20]),
            "drop_target": pd.DataFrame(target, columns=[10, 20]),
        })
    r_hat, x_hat, r2, _ = fit_projected_sensitivity(scenarios, constraint_mode=mode)
    np.testing.assert_allclose(r_hat, r_true, atol=2e-6)
    np.testing.assert_allclose(x_hat, x_true, atol=2e-6)
    assert r2 == pytest.approx(1.0, abs=1e-10)


def test_ridge_keeps_orthogonal_meter_offset_in_residual():
    r_true = np.array([[2.0, 0.4], [0.4, 1.4]])
    x_true = np.array([[1.1, 0.2], [0.2, 0.8]])
    scenario = _orthogonal_scenario(r_true, x_true)
    scenario["drop_target"] += [13.0, -7.0]
    alpha = 3.0
    diagnostics = {}
    r_hat, x_hat, _, _ = fit_projected_sensitivity(
        [scenario], alpha=alpha, diagonal_margin_ratio=0.0, diagnostics=diagnostics,
    )
    # A.T @ A=2 I; unconstrained ridge is 2/(2+alpha) times the true slopes.
    scale = 2.0 / (2.0 + alpha)
    np.testing.assert_allclose(r_hat, scale * r_true, atol=2e-6)
    np.testing.assert_allclose(x_hat, scale * x_true, atol=2e-6)
    expected_sse = 2 * (1 - scale)**2 * (np.sum(r_true**2) + np.sum(x_true**2))
    expected_sse += len(scenario["drop_target"]) * (13.0**2 + 7.0**2)
    assert diagnostics["squared_residual_sum"] == pytest.approx(expected_sse)


@pytest.mark.parametrize("mode", ["basic", "ordered", "tree_covariance"])
def test_rows_and_columns_align_by_labels(mode):
    r_true = np.array([[2.0, 0.4], [0.4, 1.4]])
    x_true = np.array([[1.1, 0.2], [0.2, 0.8]])
    original = _orthogonal_scenario(r_true, x_true)
    shuffled = deepcopy(original)
    shuffled["Q_terminal"] = shuffled["Q_terminal"].iloc[::-1, ::-1]
    shuffled["drop_target"] = shuffled["drop_target"].sample(
        frac=1.0, random_state=18,
    ).iloc[:, ::-1]
    r_hat, x_hat, r2, _ = fit_projected_sensitivity([shuffled], constraint_mode=mode)
    np.testing.assert_allclose(r_hat, r_true, atol=2e-6)
    np.testing.assert_allclose(x_hat, x_true, atol=2e-6)
    assert r2 == pytest.approx(1.0, abs=1e-10)


@pytest.mark.parametrize("mode", ["basic", "ordered"])
def test_terminal_permutation_preserves_active_constraint_solution(mode):
    raw_r = np.array([[0.0, 1.0], [1.0, 0.0]])
    raw_x = np.eye(2)
    original = _orthogonal_scenario(raw_r, raw_x)
    permutation = [1, 0]
    permuted = {"name": "permuted"}
    for key in ("P_terminal", "Q_terminal", "drop_target"):
        permuted[key] = original[key].iloc[:, permutation]
    r_base, x_base, _, _ = fit_projected_sensitivity(
        [original], constraint_mode=mode, diagonal_margin_ratio=0.0,
    )
    r_permuted, x_permuted, _, _ = fit_projected_sensitivity(
        [permuted], constraint_mode=mode, diagonal_margin_ratio=0.0,
    )
    np.testing.assert_allclose(
        r_permuted, r_base[np.ix_(permutation, permutation)], atol=3e-6,
    )
    np.testing.assert_allclose(
        x_permuted, x_base[np.ix_(permutation, permutation)], atol=3e-6,
    )


def test_ordered_mode_does_not_impose_psd():
    ordered_indefinite = np.array([[1.0, 0.9, 0.9], [0.9, 1.0, 0.0], [0.9, 0.0, 1.0]])
    scenario = _orthogonal_scenario(ordered_indefinite, np.eye(3))
    r_ordered, _, _, _ = fit_projected_sensitivity([scenario], constraint_mode="ordered")
    r_psd, _, _, _ = fit_projected_sensitivity([scenario], constraint_mode="tree_covariance")
    np.testing.assert_allclose(r_ordered, ordered_indefinite, atol=2e-6)
    assert np.linalg.eigvalsh(r_ordered).min() < -0.2
    assert np.linalg.eigvalsh(r_psd).min() >= -1e-8


@pytest.mark.parametrize("kwargs", [
    {"alpha": -1.0},
    {"alpha": np.nan},
    {"alpha": np.inf},
    {"constraint_mode": "unknown"},
    {"diagonal_margin_ratio": -1.0},
    {"diagonal_margin_ratio": np.nan},
    {"diagonal_margin_ratio": np.inf},
    {"constraint_refine_iterations": -1},
    {"constraint_refine_iterations": 1.5},
])
def test_invalid_parameters_fail_explicitly(kwargs):
    scenario = _orthogonal_scenario(np.eye(2), np.eye(2))
    with pytest.raises(ValueError):
        fit_projected_sensitivity([scenario], **kwargs)


@pytest.mark.parametrize("defect", [
    "empty_scenarios", "empty_rows", "empty_columns", "missing_column",
    "extra_column", "duplicate_columns", "duplicate_index", "different_index",
    "nan", "infinity",
])
def test_invalid_data_fail_explicitly(defect):
    scenario = _orthogonal_scenario(np.eye(2), np.eye(2))
    scenarios = [scenario]
    if defect == "empty_scenarios":
        scenarios = []
    elif defect == "empty_rows":
        for key in ("P_terminal", "Q_terminal", "drop_target"):
            scenario[key] = scenario[key].iloc[:0]
    elif defect == "empty_columns":
        for key in ("P_terminal", "Q_terminal", "drop_target"):
            scenario[key] = scenario[key].iloc[:, :0]
    elif defect == "missing_column":
        scenario["Q_terminal"] = scenario["Q_terminal"].iloc[:, :1]
    elif defect == "extra_column":
        scenario["Q_terminal"][30] = 0.0
    elif defect == "duplicate_columns":
        scenario["P_terminal"].columns = [10, 10]
    elif defect == "duplicate_index":
        scenario["Q_terminal"].index = ["same"] * len(scenario["Q_terminal"])
    elif defect == "different_index":
        scenario["drop_target"].index = pd.RangeIndex(len(scenario["drop_target"]))
    elif defect == "nan":
        scenario["drop_target"].iloc[0, 0] = np.nan
    elif defect == "infinity":
        scenario["Q_terminal"].iloc[0, 0] = np.inf
    with pytest.raises(ValueError):
        fit_projected_sensitivity(scenarios)


def test_fit_does_not_mutate_input_frames():
    scenario = _orthogonal_scenario(np.eye(2), np.eye(2))
    original = deepcopy(scenario)
    fit_projected_sensitivity([scenario])
    for key in ("P_terminal", "Q_terminal", "drop_target"):
        pd.testing.assert_frame_equal(scenario[key], original[key])


@pytest.mark.parametrize("defect", ["failed_status", "nonfinite_solution", "infeasible_solution"])
def test_qp_failure_is_reported_instead_of_returning_a_fit(monkeypatch, defect):
    from types import SimpleNamespace
    import rnj_wzzt.estimation.constrained_least_squares as qp

    scenario = _orthogonal_scenario(np.eye(2), np.eye(2))
    # Force the original-coordinate path so injected values have direct meaning.
    def no_cholesky(_matrix):
        raise np.linalg.LinAlgError("test rank-deficient route")

    def fake_minimize(_fun, values, **_kwargs):
        values = np.asarray(values).copy()
        if defect == "nonfinite_solution":
            values[0] = np.nan
        elif defect == "infeasible_solution":
            values[0] = -1.0
        return SimpleNamespace(
            success=defect != "failed_status", x=values, nit=1,
            message="injected solver failure",
        )

    monkeypatch.setattr(qp.np.linalg, "cholesky", no_cholesky)
    monkeypatch.setattr(qp, "minimize", fake_minimize)
    with pytest.raises(RuntimeError, match="failed|infeasible"):
        fit_projected_sensitivity([scenario])


@pytest.mark.parametrize("reactive_design", ["same_as_p", "zero"])
def test_rank_deficient_design_fits_identifiable_combination(reactive_design):
    true_r = np.array([[2.0, 0.4], [0.4, 1.4]])
    true_x = np.array([[1.1, 0.2], [0.2, 0.8]])
    scenario = _orthogonal_scenario(true_r, true_x)
    p = scenario["P_terminal"]
    scenario["Q_terminal"] = p.copy() if reactive_design == "same_as_p" else p * 0.0
    scenario["drop_target"] = pd.DataFrame(
        p.to_numpy() @ true_r + scenario["Q_terminal"].to_numpy() @ true_x,
        index=p.index, columns=p.columns,
    )
    diagnostics = {}
    r_hat, x_hat, r2, condition = fit_projected_sensitivity([scenario], diagnostics=diagnostics)
    if reactive_design == "same_as_p":
        # R and X individually are not identifiable; only their sum is known.
        np.testing.assert_allclose(r_hat + x_hat, true_r + true_x, atol=2e-6)
    else:
        np.testing.assert_allclose(r_hat, true_r, atol=2e-6)
        np.testing.assert_allclose(x_hat, np.zeros((2, 2)), atol=2e-6)
    assert r2 == pytest.approx(1.0, abs=1e-10)
    assert condition > 1e14
    assert diagnostics["success"] is True
    assert diagnostics["squared_residual_sum"] < 1e-10


@pytest.mark.parametrize("varying_target", [False, True])
def test_zero_power_cannot_explain_nonzero_voltage_drop(varying_target):
    scenario = _orthogonal_scenario(np.eye(2), np.eye(2))
    scenario["P_terminal"].iloc[:, :] = 0.0
    scenario["Q_terminal"].iloc[:, :] = 0.0
    if varying_target:
        target = scenario["drop_target"] + [5.0, -8.0]
    else:
        target = scenario["drop_target"] * 0.0 + [5.0, -8.0]
    scenario["drop_target"] = target
    diagnostics = {}
    r_hat, x_hat, r2, _ = fit_projected_sensitivity([scenario], diagnostics=diagnostics)
    np.testing.assert_array_equal(r_hat, np.zeros((2, 2)))
    np.testing.assert_array_equal(x_hat, np.zeros((2, 2)))
    expected_sse = float(np.sum(target.to_numpy()**2))
    assert diagnostics["squared_residual_sum"] == pytest.approx(expected_sse)
    assert "intercepts" not in diagnostics
    if varying_target:
        expected_r2 = 1.0 - expected_sse / float(np.sum((target - target.mean()).to_numpy()**2))
        assert r2 == pytest.approx(expected_r2)
        assert r2 < 0.0
    else:
        assert r2 == 0.0
    assert diagnostics["coordinates"] == "original_rank_deficient"
    assert diagnostics["success"] is True


@pytest.mark.parametrize("power_scale", [1e-8, 1e8])
def test_changing_power_units_preserves_predictions_and_fixed_margin(power_scale):
    original = _orthogonal_scenario(np.array([[1.0, 3.0], [3.0, 1.0]]), np.eye(2))
    changed = deepcopy(original)
    changed["P_terminal"] *= power_scale
    changed["Q_terminal"] *= power_scale
    diagnostics = {}
    r_hat, x_hat, _, _ = fit_projected_sensitivity(
        [changed], diagonal_margin_ratio=0.1, diagnostics=diagnostics,
    )
    np.testing.assert_allclose(r_hat * power_scale, [[2.05, 1.95], [1.95, 2.05]], atol=2e-6)
    np.testing.assert_allclose(x_hat * power_scale, np.eye(2), atol=2e-6)
    assert diagnostics["r_margin"] * power_scale == pytest.approx(0.1, abs=1e-12)
    assert diagnostics["x_margin"] * power_scale == pytest.approx(0.1, abs=1e-12)


@pytest.mark.parametrize("data_scale", [1.0, 1e-8])
def test_whitened_ridge_qp_keeps_active_optimum_under_small_data_scale(data_scale):
    scenario = _orthogonal_scenario(np.array([[1.0, 3.0], [3.0, 1.0]]), np.eye(2))
    for key in ("P_terminal", "Q_terminal", "drop_target"):
        scenario[key] *= data_scale
    diagnostics = {}
    r_hat, x_hat, _, _ = fit_projected_sensitivity(
        [scenario], alpha=3.0 * data_scale**2, diagonal_margin_ratio=0.0,
        diagnostics=diagnostics,
    )
    # Scaling A,Y by c and alpha by c^2 multiplies the full loss by c^2.
    # With A.T@A=2I, ridge scales the unconstrained solution by 2/5;
    # the ordered optimum is therefore R=(4/5)*ones and X=(2/5)*I.
    np.testing.assert_allclose(r_hat, np.full((2, 2), 0.8), atol=2e-6)
    np.testing.assert_allclose(x_hat, 0.4 * np.eye(2), atol=2e-6)
    assert diagnostics["coordinates"] == "Cholesky_whitened"


def test_tiny_excitation_with_fixed_ridge_has_bounded_known_slopes():
    true_r = np.array([[2.0, 0.4], [0.4, 1.4]])
    true_x = np.array([[1.1, 0.2], [0.2, 0.8]])
    scenario = _orthogonal_scenario(true_r, true_x)
    data_scale = 1e-8
    for key in ("P_terminal", "Q_terminal", "drop_target"):
        scenario[key] *= data_scale
    alpha = 3.0
    expected_scale = 2.0 * data_scale**2 / (2.0 * data_scale**2 + alpha)
    r_hat, x_hat, _, _ = fit_projected_sensitivity(
        [scenario], alpha=alpha, diagonal_margin_ratio=0.0,
    )
    np.testing.assert_allclose(r_hat, expected_scale * true_r, rtol=2e-6, atol=1e-28)
    np.testing.assert_allclose(x_hat, expected_scale * true_x, rtol=2e-6, atol=1e-28)


def test_diagnostics_match_unshifted_objective_and_raw_design_condition():
    true_r = np.array([[1.0, 3.0], [3.0, 1.0]])
    true_x = np.eye(2)
    first = _orthogonal_scenario(true_r, true_x, name="first")
    second = deepcopy(first)
    second["name"] = "second"
    second["P_terminal"] += [3.0, -2.0]
    second["Q_terminal"] += [-1.0, 4.0]
    for scenario, intercept in [(first, np.array([7.0, -9.0])), (second, np.array([-4.0, 6.0]))]:
        scenario["drop_target"].iloc[:, :] = (
            scenario["P_terminal"].to_numpy() @ true_r
            + scenario["Q_terminal"].to_numpy() @ true_x + intercept
        )
    diagnostics = {}
    alpha = 0.7
    r_hat, x_hat, r2, condition = fit_projected_sensitivity(
        [first, second], alpha=alpha, diagonal_margin_ratio=0.1,
        diagnostics=diagnostics,
    )
    sse = 0.0
    for scenario in [first, second]:
        p = scenario["P_terminal"].to_numpy()
        q = scenario["Q_terminal"].to_numpy()
        y = scenario["drop_target"].to_numpy()
        residual = p @ r_hat + q @ x_hat - y
        sse += float(np.sum(residual**2))
    expected_objective = 0.5 * (sse + alpha * (np.sum(r_hat**2) + np.sum(x_hat**2)))
    assert diagnostics["squared_residual_sum"] == pytest.approx(sse, abs=1e-10)
    assert diagnostics["objective"] == pytest.approx(expected_objective, abs=1e-10)
    design = np.vstack([
        np.hstack([scenario["P_terminal"], scenario["Q_terminal"]])
        for scenario in [first, second]
    ])
    target = np.vstack([scenario["drop_target"] for scenario in [first, second]])
    initial = np.linalg.solve(design.T @ design + alpha * np.eye(4), design.T @ target)
    for key, block in zip(["r_margin", "x_margin"], [initial[:2], initial[2:]]):
        diagonal = np.diag(block)
        assert diagnostics[key] == pytest.approx(0.1 * np.median(diagonal[diagonal > 0]), abs=1e-10)
    assert condition == pytest.approx(np.linalg.cond(design))
    assert "intercepts" not in diagnostics
    assert "centered_design_condition_number" not in diagnostics
    assert diagnostics["method"] == "SLSQP_convex_QP"
    assert diagnostics["success"] is True
    assert np.isfinite(r2)


def test_zero_iterations_returns_feasible_initializer_without_calling_qp(monkeypatch):
    import rnj_wzzt.estimation.multiscenario as multiscenario

    def unexpected_solver(*_args, **_kwargs):
        pytest.fail("zero iterations must not call the QP solver")

    monkeypatch.setattr(multiscenario, "solve_symmetric_least_squares", unexpected_solver)
    scenario = _orthogonal_scenario(np.array([[0.0, 1.0], [1.0, 0.0]]), np.eye(2))
    diagnostics = {}
    r_hat, x_hat, _, _ = fit_projected_sensitivity(
        [scenario], diagonal_margin_ratio=0.0, constraint_refine_iterations=0,
        diagnostics=diagnostics,
    )
    # This initializer only raises diagonals. It is feasible but not the QP optimum.
    np.testing.assert_allclose(r_hat, np.ones((2, 2)), atol=1e-12)
    np.testing.assert_allclose(x_hat, np.eye(2), atol=1e-12)
    assert diagnostics["method"] == "feasible_initialization_only"
    assert diagnostics["success"] is False
    assert diagnostics["iterations"] == 0
    assert diagnostics["objective"] == pytest.approx(2.0, abs=1e-12)



def test_collinear_noisy_pq_does_not_use_unstable_whitening():
    rng = np.random.default_rng(18)
    p = rng.normal(size=(30, 2)) * 1e-3
    q = 0.5 * p
    shared = np.eye(2) + 0.2
    y = p @ shared + q @ (0.5 * shared) + rng.normal(size=(30, 2)) * 1e-5
    scenario = {
        "name": "collinear_noisy",
        "P_terminal": pd.DataFrame(p, columns=[10, 20]),
        "Q_terminal": pd.DataFrame(q, columns=[10, 20]),
        "drop_target": pd.DataFrame(y, columns=[10, 20]),
    }
    diagnostics = {}
    r_hat, x_hat, _, _ = fit_projected_sensitivity([scenario], diagnostics=diagnostics)
    assert diagnostics["success"] is True
    assert diagnostics["coordinates"] == "spectral_preconditioned"
    for matrix, margin in [(r_hat, diagnostics["r_margin"]), (x_hat, diagnostics["x_margin"])]:
        assert np.all(np.isfinite(matrix))
        assert matrix.min() >= 0.0
        np.testing.assert_allclose(matrix, matrix.T, atol=1e-12)
        assert matrix[0, 0] - matrix[0, 1] >= margin - 1e-10
        assert matrix[1, 1] - matrix[1, 0] >= margin - 1e-10



@pytest.mark.parametrize("epsilon", [1e-5, 1e-6, 1e-8])
def test_nearly_collinear_nonnegative_fit_matches_boundary_oracle(epsilon):
    rng = np.random.default_rng(2)
    p = rng.normal(size=(40, 1))
    noise_direction = rng.normal(size=(40, 1))
    q = p + epsilon * noise_direction
    y = noise_direction
    scenario = {
        "name": "nearly_collinear",
        "P_terminal": pd.DataFrame(p, columns=[10]),
        "Q_terminal": pd.DataFrame(q, columns=[10]),
        "drop_target": pd.DataFrame(y, columns=[10]),
    }
    expected_x = max(float((q.T @ y).item() / (q.T @ q).item()), 0.0)
    oracle_residual = q * expected_x - y
    oracle_sse = float(np.sum(oracle_residual**2))
    # The unconstrained solution is R=-1/epsilon, X=1/epsilon. At R=0,
    # X is its one-dimensional LS optimum and the R directional derivative
    # is nonnegative, so this feasible boundary point is globally optimal.
    assert float((p.T @ oracle_residual).item()) >= -1e-12
    diagnostics = {}
    r_hat, x_hat, _, _ = fit_projected_sensitivity([scenario], diagnostics=diagnostics)
    fitted_residual = p @ r_hat + q @ x_hat - y
    fitted_sse = float(np.sum(fitted_residual**2))
    assert fitted_sse == pytest.approx(oracle_sse, abs=1e-10, rel=0.0)
    assert r_hat.min() >= 0.0
    assert x_hat.min() >= 0.0
    assert diagnostics["success"] is True


@pytest.mark.parametrize("seed", [5, 18, 61])
def test_collinear_signed_targets_return_feasible_finite_fit(seed):
    rng = np.random.default_rng(seed)
    p = rng.normal(size=(40, 3)) * 1e-3
    q = 0.5 * p
    # Signed, asymmetric coefficients intentionally violate the matrix bounds.
    true_r = rng.normal(size=(3, 3))
    true_x = rng.normal(size=(3, 3))
    y = p @ true_r.T + q @ true_x.T + rng.normal(size=(40, 3)) * 1e-5
    columns = [10, 20, 30]
    scenario = {
        "name": f"signed_collinear_{seed}",
        "P_terminal": pd.DataFrame(p, columns=columns),
        "Q_terminal": pd.DataFrame(q, columns=columns),
        "drop_target": pd.DataFrame(y, columns=columns),
    }
    diagnostics = {}
    r_hat, x_hat, _, _ = fit_projected_sensitivity([scenario], diagnostics=diagnostics)
    assert diagnostics["success"] is True
    assert np.isfinite(diagnostics["objective"])
    for matrix, margin in [(r_hat, diagnostics["r_margin"]), (x_hat, diagnostics["x_margin"])]:
        assert np.all(np.isfinite(matrix))
        assert matrix.min() >= 0.0
        np.testing.assert_allclose(matrix, matrix.T, atol=1e-12)
        diagonal_gaps = np.diag(matrix)[:, None] - matrix
        assert diagonal_gaps[~np.eye(3, dtype=bool)].min() >= margin - 1e-10
