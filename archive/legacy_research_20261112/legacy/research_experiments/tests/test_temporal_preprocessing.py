"""Independent oracles for historical temporal preprocessing recipes."""

from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from rnj_wzzt.estimation.multiscenario import fit_projected_sensitivity
from research_experiments.rnj.temporal_preprocessing import (
    apply_preprocessing_recipe, daily_demean, preprocess_scenarios,
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
