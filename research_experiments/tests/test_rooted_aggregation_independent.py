"""Independent deembedding oracles for the historical aggregation research path."""

import numpy as np
import pandas as pd
import pytest

from research_experiments.rnj.rooted_aggregation import PseudoCluster, aggregate_rooted_scenarios


@pytest.mark.parametrize("weight", [0.0, 0.5, 1.0])
@pytest.mark.parametrize("order", [[1, 2, 3], [3, 2, 1]])
def test_cherry_deembedding_matches_independently_calculated_boundary_voltage(weight, order):
    # Root -- stem -- {cherry boundary, terminal 3}; external power contributes
    # to the stem drop but must disappear from the cherry's differential drop.
    p = np.array([[0.1, 0.2, 0.3], [0.4, -0.1, 0.5], [-0.2, 0.3, -0.1]])
    q = np.array([[0.02, 0.03, 0.01], [-0.01, 0.02, 0.03], [0.03, -0.02, 0.01]])
    r = np.array([[0.10, 0.06, 0.02], [0.06, 0.13, 0.02], [0.02, 0.02, 0.09]])
    x = np.array([[0.06, 0.04, 0.01], [0.04, 0.09, 0.01], [0.01, 0.01, 0.07]])
    root_sq = np.array([1.0, 1.02, 0.98])
    boundary_sq = root_sq - 0.02 * p.sum(axis=1) - 0.01 * q.sum(axis=1)
    boundary_sq -= 0.04 * p[:, :2].sum(axis=1) + 0.03 * q[:, :2].sum(axis=1)
    terminal_sq = root_sq[:, None] - p @ r.T - q @ x.T
    index = pd.Index([7, 13, 19], name="timestamp")
    p_frame = pd.DataFrame(p, columns=[1, 2, 3], index=index)
    q_frame = pd.DataFrame(q, columns=[1, 2, 3], index=index)
    voltage = pd.DataFrame(np.sqrt(terminal_sq), columns=[1, 2, 3], index=index)
    scenario = {"name": "independent", "P_terminal": p_frame, "Q_terminal": q_frame,
                "V_terminal": voltage, "root_voltage": pd.Series(np.sqrt(root_sq), index=index)}
    cluster = PseudoCluster(900001, frozenset({1, 2}), 1.0, (frozenset({1, 2}),), ((1, 2),))
    positions = [terminal - 1 for terminal in order]
    aggregated, mapping = aggregate_rooted_scenarios(
        [scenario], order, r[np.ix_(positions, positions)], x[np.ix_(positions, positions)],
        [cluster], "deembedded_vsq", weight)
    actual = aggregated[0]
    mean_sq = terminal_sq[:, :2].mean(axis=1)
    expected_sq = (1 - weight) * mean_sq + weight * boundary_sq
    np.testing.assert_allclose(actual["V_terminal"][900001] ** 2, expected_sq, rtol=1e-14)
    np.testing.assert_allclose(actual["P_terminal"][900001], p[:, :2].sum(axis=1), rtol=1e-14)
    np.testing.assert_allclose(actual["Q_terminal"][900001], q[:, :2].sum(axis=1), rtol=1e-14)
    singleton = next(pseudo for pseudo, members in mapping.items() if members == frozenset({3}))
    np.testing.assert_array_equal(actual["V_terminal"][singleton], voltage[3])
    np.testing.assert_allclose(actual["drop_target"][900001], root_sq - expected_sq, atol=1e-15)
    assert actual["V_terminal"].index.equals(index)
    assert not np.allclose(mean_sq, boundary_sq)


@pytest.mark.parametrize("weight", [0.0, 0.5, 1.0])
@pytest.mark.parametrize("order", [[2, 7, 11, 20], [20, 11, 7, 2]])
def test_deembedding_matches_scalar_formula_for_asymmetric_estimates(weight, order):
    # Imperfect estimates need not be symmetric. Distinct row coefficients
    # expose a missing transpose; the last terminal is outside the clade.
    labels = [2, 7, 11, 20]
    r = np.array([[0.7, 0.1, 0.4, 0.9], [-0.2, 0.8, 0.2, 0.3],
                  [0.3, 0.05, 0.9, 0.4], [0.1, 0.2, 0.3, 0.8]])
    x = np.array([[0.4, 0.05, 0.1, 0.1], [0.08, 0.5, 0.2, 0.3],
                  [0.12, 0.09, 0.6, 0.1], [0.1, 0.2, 0.1, 0.4]])
    p = np.array([[0.1, 0.2, 0.3, 7.0], [-0.2, 0.3, 0.1, -4.0]])
    q = np.array([[0.02, 0.01, -0.03, -2.0], [-0.01, 0.03, 0.02, 5.0]])
    v = np.array([[0.95, 0.96, 0.94, 0.99], [0.97, 0.95, 0.93, 0.98]])
    scenario = {
        "name": "asymmetric",
        "P_terminal": pd.DataFrame(p, columns=labels),
        "Q_terminal": pd.DataFrame(q, columns=labels),
        "V_terminal": pd.DataFrame(v, columns=labels),
        "root_voltage": pd.Series([1.01, 1.02]),
    }
    cluster = PseudoCluster(900000, frozenset(labels[:3]), 1.0, (), ())
    positions = [labels.index(label) for label in order]
    actual, _ = aggregate_rooted_scenarios(
        [scenario], order, r[np.ix_(positions, positions)], x[np.ix_(positions, positions)],
        [cluster], "deembedded_vsq", weight,
    )
    # Upper-triangle 20th percentiles are 0.14 for R and 0.07 for X.
    recovered = [[v[t, j] ** 2 + sum(
        p[t, i] * max(r[j, i] - 0.14, 0.0)
        + q[t, i] * max(x[j, i] - 0.07, 0.0)
        for i in range(3)
    ) for j in range(3)] for t in range(2)]
    mean_sq = np.mean(v[:, :3] ** 2, axis=1)
    expected = mean_sq + weight * (np.median(recovered, axis=1) - mean_sq)
    np.testing.assert_allclose(actual[0]["V_terminal"][900000] ** 2, expected, atol=1e-14)
