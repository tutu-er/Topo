"""Independent numerical oracles and falsifiable estimator limitations.

These tests use physical arrays, elementary two-variable active faces, and
explicit temporal operators. They do not import solver packing/projection
helpers. Counterexamples deliberately distinguish a good fit from an identified
R/X pair or an estimator that is robust to corrupted meter observations.
"""

from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from rnj_wzzt.estimation.multiscenario import (
    fit_projected_sensitivity,
    preprocess_scenarios,
)
from rnj_wzzt.estimation.preprocessing import (
    RECIPE,
    apply_preprocessing_recipe,
    daily_demean,
    squared_voltage_drop_from_observed_root,
)


def _scenario(p, q, y, name="oracle", labels=None):
    p, q, y = (np.asarray(a, dtype=float) for a in (p, q, y))
    columns = labels if labels is not None else [f"meter_{i}" for i in range(p.shape[1])]
    return {
        "name": name,
        "P_terminal": pd.DataFrame(p, columns=columns),
        "Q_terminal": pd.DataFrame(q, columns=columns),
        "drop_target": pd.DataFrame(y, columns=columns),
    }


def _two_variable_nonnegative_oracle(design, target, alpha):
    """Enumerate the interior, both axes, and origin of the positive quadrant.

    On each axis the minimizer is a scalar dot-product quotient. A feasible
    unconstrained least-squares point covers the interior. Convexity makes
    the best of these four faces a global oracle even if A is rank deficient.
    """
    a = design
    y = target
    augmented = np.vstack([a, np.sqrt(alpha) * np.eye(2)])
    candidates = [np.zeros(2)]
    unconstrained = np.linalg.lstsq(augmented, np.r_[y, 0.0, 0.0], rcond=None)[0]
    if np.all(unconstrained >= 0.0):
        candidates.append(unconstrained)
    for j in range(2):
        vector = np.zeros(2)
        denominator = a[:, j] @ a[:, j] + alpha
        vector[j] = max(float(a[:, j] @ y) / denominator, 0.0) if denominator else 0.0
        candidates.append(vector)
    losses = [float(np.sum((a @ beta - y)**2) + alpha * (beta @ beta)) for beta in candidates]
    best = int(np.argmin(losses))
    return candidates[best], losses[best]


@pytest.mark.parametrize("style", ["independent", "collinear", "opposite", "near_collinear", "weak_q", "zero_q"])
@pytest.mark.parametrize("alpha", [0.0, 0.25, 10.0])
@pytest.mark.parametrize("data_scale", [1e-5, 1.0, 1e5])
def test_scalar_constrained_ridge_matches_all_active_faces(style, alpha, data_scale):
    rng = np.random.default_rng(731)
    p, independent = rng.normal(size=(2, 37))
    q = {
        "independent": independent,
        "collinear": 0.4 * p,
        "opposite": -0.7 * p,
        "near_collinear": p + 1e-6 * independent,
        "weak_q": 1e-5 * independent,
        "zero_q": np.zeros_like(p),
    }[style]
    design = np.column_stack([p, q])
    y = -0.6 * p + 1.3 * q + 0.17 * rng.normal(size=len(p)) + 2.0
    expected, loss = _two_variable_nonnegative_oracle(design, y, alpha)
    scenario = _scenario(data_scale * p[:, None], data_scale * q[:, None], data_scale * y[:, None])
    diagnostics = {}
    r, x, _, _ = fit_projected_sensitivity(
        [scenario], alpha=alpha * data_scale**2,
        diagonal_margin_ratio=0.0, diagnostics=diagnostics,
    )
    fitted = np.array([r.item(), x.item()])
    assert np.min(fitted) >= 0.0
    assert diagnostics["success"] is True
    assert 2.0 * diagnostics["objective"] / data_scale**2 == pytest.approx(loss, rel=2e-7, abs=2e-7)
    # Null directions need not select the oracle's particular minimizer.
    np.testing.assert_allclose(design @ fitted, design @ expected, rtol=2e-5, atol=2e-5)
    if alpha > 0:
        np.testing.assert_allclose(fitted, expected, rtol=2e-5, atol=2e-5)


@pytest.mark.parametrize("n", [2, 4, 8])
@pytest.mark.parametrize("ratios", [(0.0, 1.0), (0.4, 1.2), (-0.6, 0.6)])
def test_multiple_power_factors_identify_shared_rx_when_each_scenario_cannot(n, ratios):
    rng = np.random.default_rng(912 + n)
    r_true = 0.3 * np.ones((n, n)) + np.diag(np.linspace(1.0, 2.0, n))
    x_true = 0.12 * np.ones((n, n)) + np.diag(np.linspace(0.5, 0.9, n))
    scenarios, designs = [], []
    for k, ratio in enumerate(ratios):
        p = rng.normal(size=(3 * n + 9, n)) + 3.0 * k
        q = ratio * p
        y = p @ r_true + q @ x_true
        scenarios.append(_scenario(p, q, y, name=f"factor_{k}"))
        a = np.column_stack([p, q])
        assert np.linalg.matrix_rank(a) == n
        designs.append(a)
    assert np.linalg.matrix_rank(np.vstack(designs)) == 2 * n
    diagnostics = {}
    r, x, _, _ = fit_projected_sensitivity(scenarios, diagnostics=diagnostics)
    np.testing.assert_allclose(r, r_true, atol=3e-6, rtol=3e-6)
    np.testing.assert_allclose(x, x_true, atol=3e-6, rtol=3e-6)
    assert diagnostics["squared_residual_sum"] < 1e-8


@pytest.mark.parametrize("n", [2, 5])
@pytest.mark.parametrize("p_scale,q_scale", [(1e-6, 1e6), (1e6, 1e-6), (1.0, 1.0)])
def test_independent_active_and_reactive_units_preserve_physical_predictions(n, p_scale, q_scale):
    rng = np.random.default_rng(188 + n)
    p, q = rng.normal(size=(2, 6 * n + 20, n))
    r_true = np.eye(n) + 0.1
    x_true = 0.4 * np.eye(n) + 0.05
    y = p @ r_true + q @ x_true
    r, x, _, _ = fit_projected_sensitivity(
        [_scenario(p_scale * p, q_scale * q, y)], diagonal_margin_ratio=0.0,
    )
    np.testing.assert_allclose(p_scale * r, r_true, atol=3e-6, rtol=3e-6)
    np.testing.assert_allclose(q_scale * x, x_true, atol=3e-6, rtol=3e-6)
    np.testing.assert_allclose(p_scale * p @ r + q_scale * q @ x, y, atol=2e-5)


@pytest.mark.parametrize("alpha", [0.0, 0.4])
def test_replicating_observations_is_equivalent_to_scaling_ridge(alpha):
    rng = np.random.default_rng(867)
    p, q = rng.normal(size=(2, 31, 3))
    y = rng.normal(size=(31, 3)) + [2.0, -4.0, 8.0]
    scenario = _scenario(p, q, y)
    replicated = _scenario(np.tile(p, (4, 1)), np.tile(q, (4, 1)), np.tile(y, (4, 1)))
    first = fit_projected_sensitivity([scenario], alpha=alpha, diagonal_margin_ratio=0.0)
    second = fit_projected_sensitivity([replicated], alpha=4 * alpha, diagonal_margin_ratio=0.0)
    np.testing.assert_allclose(second[:2], first[:2], atol=3e-6, rtol=3e-6)


@pytest.mark.parametrize("shift", [0.2, 1.0, 5.0])
def test_correlated_unobserved_common_mode_is_indistinguishable_from_shared_path(shift):
    rng = np.random.default_rng(43)
    n = 4
    p, q = rng.normal(size=(2, 70, n))
    r_true, x_true = np.eye(n) + 0.2, 0.5 * np.eye(n) + 0.1
    common = shift * p.sum(axis=1)
    y = p @ r_true + q @ x_true + common[:, None]
    r, x, r2, _ = fit_projected_sensitivity([_scenario(p, q, y)])
    # The error is precisely P*(shift*11.T), so no residual statistic can
    # distinguish root-meter corruption from a longer common upstream path.
    np.testing.assert_allclose(r, r_true + shift * np.ones((n, n)), atol=3e-6)
    np.testing.assert_allclose(x, x_true, atol=3e-6)
    assert r2 == pytest.approx(1.0, abs=1e-10)
    assert np.linalg.norm(r - r_true, "fro") == pytest.approx(n * shift, rel=3e-6)


@pytest.mark.parametrize("outlier", [0.0, 1.0, 100.0])
def test_single_voltage_outlier_has_unbounded_least_squares_influence(outlier):
    # Centered orthogonal P/Q with P.T P = Q.T Q = 2. The first observation
    # has P=1, Q=0; an additive voltage error b raises R by exactly b/2.
    p = np.array([1.0, -1.0, 0.0, 0.0])[:, None]
    q = np.array([0.0, 0.0, 1.0, -1.0])[:, None]
    y = 2 * p + q
    y[0, 0] += outlier
    r, x, _, _ = fit_projected_sensitivity([_scenario(p, q, y)])
    assert r.item() == pytest.approx(2.0 + outlier / 2.0, abs=1e-6)
    assert x.item() == pytest.approx(1.0, abs=1e-6)


def test_scenario_offsets_are_not_explained_by_zero_power():
    p = np.zeros((4, 1))
    fluctuations = np.array([1.0, -1.0, 1.0, -1.0])[:, None]
    scenarios = [_scenario(p, p, fluctuations + mean, name=str(mean)) for mean in (-100.0, 100.0)]
    diagnostics = {}
    r, x, r2, _ = fit_projected_sensitivity(scenarios, diagnostics=diagnostics)
    np.testing.assert_array_equal([r.item(), x.item()], [0.0, 0.0])
    assert diagnostics["squared_residual_sum"] == pytest.approx(8.0 * 10001.0)
    assert r2 == pytest.approx(0.0)
    assert "intercepts" not in diagnostics


@pytest.mark.parametrize("n", [1, 3])
def test_constant_nonzero_loads_identify_rx_across_operating_points(n):
    """Each day is constant, but the combined operating points span every input."""
    r_true = np.eye(n) + 0.2
    x_true = 0.3 * np.eye(n) + 0.1
    operating_points = np.diag(np.arange(2.0, 2.0 + 2 * n))
    scenarios = []
    for index, point in enumerate(operating_points):
        p = np.tile(point[:n], (5, 1))
        q = np.tile(point[n:], (5, 1))
        scenarios.append(_scenario(p, q, p @ r_true + q @ x_true, name=str(index)))
    transformed = preprocess_scenarios(scenarios, RECIPE)
    assert RECIPE["kind"] == "raw"
    for original, fitted in zip(scenarios, transformed):
        for key in ("P_terminal", "Q_terminal", "drop_target"):
            pd.testing.assert_frame_equal(original[key], fitted[key])
    r, x, r2, condition = fit_projected_sensitivity(transformed)
    np.testing.assert_allclose(r, r_true, atol=1e-8)
    np.testing.assert_allclose(x, x_true, atol=1e-8)
    assert r2 == pytest.approx(1.0)
    assert condition == pytest.approx((2 * n + 1.0) / 2.0)


def test_time_varying_observed_root_cancels_before_rx_regression():
    rng = np.random.default_rng(314)
    p, q = rng.uniform(0.002, 0.015, size=(2, 60, 2))
    r_true = np.array([[0.8, 0.2], [0.2, 0.6]])
    x_true = np.array([[0.3, 0.1], [0.1, 0.4]])
    physical_drop = p @ r_true + q @ x_true
    # Root variation is deliberately correlated with power and is observed.
    root = pd.Series(1.02 + 2.0 * p.sum(axis=1))
    terminal = pd.DataFrame(np.sqrt(root.to_numpy()[:, None]**2 - physical_drop), columns=[10, 20])
    scenario = _scenario(p, q, squared_voltage_drop_from_observed_root(terminal, root), labels=[10, 20])
    r, x, r2, _ = fit_projected_sensitivity([scenario])
    np.testing.assert_allclose(r, r_true, atol=1e-8)
    np.testing.assert_allclose(x, x_true, atol=1e-8)
    assert r2 == pytest.approx(1.0)


def test_independent_terminal_offsets_remain_in_the_physical_residual():
    design = np.vstack([np.eye(4), -np.eye(4)])
    p, q = design[:, :2], design[:, 2:]
    r_true, x_true = np.eye(2) + 0.2, 0.4 * np.eye(2) + 0.1
    offsets = [np.array([0.7, -0.3]), np.array([-0.2, 0.9])]
    scenarios = [_scenario(p, q, p @ r_true + q @ x_true + offset, name=str(index))
                 for index, offset in enumerate(offsets)]
    diagnostics = {}
    r, x, _, _ = fit_projected_sensitivity(scenarios, diagnostics=diagnostics)
    # The constant offsets are orthogonal to every column of the design.
    np.testing.assert_allclose(r, r_true, atol=1e-8)
    np.testing.assert_allclose(x, x_true, atol=1e-8)
    expected_sse = len(design) * sum(offset @ offset for offset in offsets)
    assert diagnostics["squared_residual_sum"] == pytest.approx(expected_sse)
    assert "intercepts" not in diagnostics


@pytest.mark.parametrize("power,target,expected", [(0.0, 0.0, 1.0), (0.0, 4.0, 0.0), (2.0, 4.0, 1.0)])
def test_constant_target_r2_uses_finite_residual_convention(power, target, expected):
    p, q, y = np.full((7, 1), power), np.zeros((7, 1)), np.full((7, 1), target)
    _, _, r2, _ = fit_projected_sensitivity([_scenario(p, q, y)])
    assert r2 == expected


@pytest.mark.parametrize("day_size", [1, 3, 7, 20])
@pytest.mark.parametrize("as_series", [False, True])
def test_daily_demean_matches_block_projection_including_partial_final_day(day_size, as_series):
    rng = np.random.default_rng(903)
    values = rng.normal(size=(11, 2)) + [4.0, -9.0]
    index = pd.Index([f"t{i * 3}" for i in range(11)], name="meter_time")
    frame = pd.DataFrame(values, index=index, columns=["a", "b"])
    data = frame["a"] if as_series else frame
    original = data.copy(deep=True)
    operator = np.eye(len(frame))
    for start in range(0, len(frame), day_size):
        stop = min(start + day_size, len(frame))
        operator[start:stop, start:stop] -= 1.0 / (stop - start)
    result = daily_demean(data, samples_per_day=day_size)
    expected = operator @ data.to_numpy()
    np.testing.assert_allclose(result, expected, atol=5e-15)
    np.testing.assert_allclose(daily_demean(result, day_size), result, atol=2e-15)
    assert result.index.equals(index)
    if as_series:
        pd.testing.assert_series_equal(data, original)
        assert result.name == data.name
    else:
        pd.testing.assert_frame_equal(data, original)


@pytest.mark.parametrize("recipe", [
    {"kind": "raw"},
    {"kind": "demean"},
    {"kind": "difference"},
    {"kind": "rolling_highpass", "window": 5},
    {"kind": "chain", "steps": [{"kind": "difference"}, {"kind": "rolling_highpass", "window": 3}]},
])
@pytest.mark.parametrize("reorder_inputs", [False, True])
def test_shared_temporal_operator_preserves_exact_physical_regression(recipe, reorder_inputs):
    rng = np.random.default_rng(871)
    p, q = rng.normal(size=(2, 45, 3))
    r_true, x_true = np.eye(3) + 0.3, 0.5 * np.eye(3) + 0.1
    scenario = _scenario(p, q, p @ r_true + q @ x_true)
    if reorder_inputs:
        for key in ("Q_terminal", "drop_target"):
            scenario[key] = scenario[key].iloc[rng.permutation(len(p))]
    original = deepcopy(scenario)
    transformed = preprocess_scenarios([scenario], recipe)
    r, x, _, _ = fit_projected_sensitivity(transformed)
    np.testing.assert_allclose(r, r_true, atol=3e-6)
    np.testing.assert_allclose(x, x_true, atol=3e-6)
    for key in ("P_terminal", "Q_terminal", "drop_target"):
        pd.testing.assert_frame_equal(scenario[key], original[key])


@pytest.mark.parametrize("key", ["P_terminal", "Q_terminal", "drop_target"])
@pytest.mark.parametrize("invalid_index", ["missing", "extra", "duplicate"])
def test_preprocessing_rejects_mismatched_or_ambiguous_time_indices(key, invalid_index):
    values = np.arange(6, dtype=float).reshape(3, 2)
    scenario = _scenario(values, 2 * values, 3 * values)
    frame = scenario[key]
    if invalid_index == "missing":
        scenario[key] = frame.iloc[:-1]
    elif invalid_index == "extra":
        scenario[key] = pd.concat([frame, frame.iloc[:1].rename(index={0: 3})])
    else:
        scenario[key] = pd.concat([frame, frame.iloc[:1]])
    with pytest.raises(ValueError, match="time indices"):
        preprocess_scenarios([scenario], {"kind": "difference"})


@pytest.mark.parametrize("window", [2, 3, 5, 21])
def test_rolling_highpass_matches_explicit_finite_sample_weights(window):
    values = np.array([[1.0, 10.0], [4.0, -3.0], [2.0, 8.0], [9.0, 1.0], [-4.0, 5.0], [3.0, 7.0]])
    frame = pd.DataFrame(values, columns=["x", "y"])
    operator = np.eye(len(frame))
    # pandas' centered even window places the extra sample on the left.
    for t in range(len(frame)):
        start = max(0, t - window // 2)
        stop = min(len(frame), t + (window + 1) // 2)
        operator[t, start:stop] -= 1.0 / (stop - start)
    actual = apply_preprocessing_recipe(frame, {"kind": "rolling_highpass", "window": window}, 6)
    np.testing.assert_allclose(actual, operator @ values, atol=2e-15)


@pytest.mark.parametrize("root_error", [0.0, 0.005, -0.01])
def test_observed_root_error_is_exact_common_squared_voltage_shift(root_error):
    index = pd.Index(["late", "early", "mid"])
    root = pd.Series([1.02, 1.01, 1.03], index=index)
    drop = np.array([[0.04, 0.03], [0.02, 0.01], [0.05, 0.02]])
    terminal = pd.DataFrame(np.sqrt(root.to_numpy()[:, None]**2 - drop), index=index, columns=[10, 40])
    # Deliberately reorder the root meter and include an unmatched observation.
    observed = pd.concat([root + root_error, pd.Series([9.0], index=["unmatched"])]).iloc[::-1]
    actual = squared_voltage_drop_from_observed_root(terminal, observed)
    expected_error = 2.0 * root.to_numpy() * root_error + root_error**2
    np.testing.assert_allclose(actual, drop + expected_error[:, None], atol=3e-16)
    assert actual.index.equals(index)
    np.testing.assert_allclose(np.diff(actual.to_numpy() - drop, axis=1), 0.0, atol=3e-16)
