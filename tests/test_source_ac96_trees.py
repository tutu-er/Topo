"""Independent checks for source-aware observed-root NJ/RNJ controls."""

import numpy as np
import pandas as pd
import pytest

import research_experiments.rnj.source_ac96_trees as module
from rnj_wzzt.estimation.multiscenario import fit_projected_sensitivity
from rnj_wzzt.estimation.unmetered_inputs import SourceInputs, prepare_source_inputs


def _problem(rank=2, seed=7):
    rng = np.random.default_rng(seed)
    count, n = 100, 3
    columns = [11, 22, 33]
    index = pd.Index(1000 + np.arange(count), name="time")
    p, q = rng.uniform(.1, .8, (count, n)), rng.uniform(.1, .7, (count, n))
    u_p = .2 + .1 * np.sin(np.arange(count) / 5)
    u_q = .4 * u_p if rank == 1 else .1 + .05 * np.cos(np.arange(count) / 7)
    if rank == 0:
        u_p, u_q = np.zeros(count), np.zeros(count)
    r = np.array([[1.7, .4, .2], [.4, 1.8, .2], [.2, .2, 1.5]])
    x = np.array([[1.1, .3, .1], [.3, 1.3, .1], [.1, .1, 1.]])
    g = np.array([[.2, .4, .1], [.1, .2, .15]])
    y = p @ r + q @ x + np.column_stack((u_p, u_q)) @ g
    frames = [pd.DataFrame(v, index=index, columns=columns) for v in (p, q, y)]
    scenario = dict(zip(("P_terminal", "Q_terminal", "drop_target"), frames))
    scenario["name"] = "observed_only"
    source = SourceInputs(pd.Series(u_p, index=index), pd.Series(u_q, index=index), "external")
    return [scenario], [source], r, x, g


def _explicit_symmetric_joint_lstsq(scenarios, sources):
    """Independent joint LS in symmetric R/X and free source coefficients."""
    scenario, source = scenarios[0], sources[0]
    p, q, y = [scenario[k].to_numpy() for k in ("P_terminal", "Q_terminal", "drop_target")]
    u = np.column_stack((source.p_pu.to_numpy(), source.q_pu.to_numpy()))
    n = p.shape[1]
    basis = []
    matrices = []
    for i in range(n):
        for j in range(i, n):
            matrix = np.zeros((n, n))
            matrix[i, j] = matrix[j, i] = 1
            matrices.append(matrix)
    basis.extend((p @ matrix).ravel() for matrix in matrices)
    basis.extend((q @ matrix).ravel() for matrix in matrices)
    for k in range(2):
        for j in range(n):
            response = np.zeros_like(y)
            response[:, j] = u[:, k]
            basis.append(response.ravel())
    coefficient = np.linalg.lstsq(np.column_stack(basis), y.ravel(), rcond=None)[0]
    width = len(matrices)
    r = sum(c * matrix for c, matrix in zip(coefficient[:width], matrices))
    x = sum(c * matrix for c, matrix in zip(coefficient[width:2 * width], matrices))
    return r, x, coefficient[2 * width:].reshape(2, n)


@pytest.mark.parametrize("rank", [1, 2])
def test_fwl_ordered_fit_matches_independent_explicit_nuisance_ls(rank):
    scenarios, sources, _, _, _ = _problem(rank)
    rng = np.random.default_rng(99)
    scenarios[0]["drop_target"] += rng.normal(0, .0001, (100, 3))
    expected_r, expected_x, expected_g = _explicit_symmetric_joint_lstsq(scenarios, sources)
    # This independent unconstrained symmetric optimum is strictly interior to
    # the ordered QP domain, so it is also the constrained joint-LS optimum.
    for matrix in (expected_r, expected_x):
        assert np.all(matrix > 0)
        assert np.all(np.diag(matrix)[:, None] > matrix - np.diag(np.diag(matrix)))
    result = module.fit_tree(scenarios, sources, "RNJ")
    np.testing.assert_allclose(result["r_matrix"], expected_r, atol=2e-8)
    np.testing.assert_allclose(result["x_matrix"], expected_x, atol=2e-8)
    np.testing.assert_allclose(result["source_response_g"], expected_g, atol=2e-8)
    assert result["source_rank"] == rank
    assert result["diagnostics"]["source_location_used"] is False
    assert result["diagnostics"]["source_path_constraints"] is False


@pytest.mark.parametrize("method", ["NJ", "RNJ"])
def test_no_source_is_core_raw_regression_with_observed_root_anchor(method):
    scenarios, sources, _, _, _ = _problem(0)
    expected = fit_projected_sensitivity(scenarios)
    result = module.fit_tree(scenarios, sources, method, root_bus=12345)
    np.testing.assert_allclose(result["r_matrix"], expected[0], atol=1e-12)
    np.testing.assert_allclose(result["x_matrix"], expected[1], atol=1e-12)
    np.testing.assert_array_equal(result["source_response_g"], np.zeros((2, 3)))
    assert result["source_rank"] == 0
    assert result["tree"].root == 12345
    assert any(12345 in edge[:2] for edge in result["tree"].edges)
    assert result["diagnostics"]["intercept_added"] is False


def test_source_and_all_channels_resample_in_sync_after_label_alignment():
    count = 12
    values = np.arange(count, dtype=float)
    index = pd.Index(100 + np.arange(count))
    scenario = {"name": "sync", "P_terminal": pd.DataFrame({7: values}, index=index),
                "Q_terminal": pd.DataFrame({7: values * 10}, index=index),
                "drop_target": pd.DataFrame({7: values * 100}, index=index)}
    sources = [SourceInputs(pd.Series(values * 1000, index=index).iloc[::-1],
                            pd.Series(values * 10000, index=index).iloc[::-1], "external")]
    aligned, prepared = module._aligned_inputs([scenario], sources)
    sampled, sampled_sources = module._moving_block_source_copy(
        aligned, prepared, np.random.default_rng(28), 4,
    )
    p = sampled[0]["P_terminal"].to_numpy()[:, 0]
    assert not np.array_equal(p, values)
    np.testing.assert_array_equal(sampled[0]["Q_terminal"].to_numpy()[:, 0], p * 10)
    np.testing.assert_array_equal(sampled[0]["drop_target"].to_numpy()[:, 0], p * 100)
    np.testing.assert_array_equal(sampled_sources[0].p_pu, p * 1000)
    np.testing.assert_array_equal(sampled_sources[0].q_pu, p * 10000)
    assert sampled[0]["P_terminal"].index.is_unique


def test_source_labels_reordered_and_prepared_inputs_give_same_estimate():
    scenarios, sources, _, _, _ = _problem(2)
    source = sources[0]
    reverse = [SourceInputs(source.p_pu.iloc[::-1], source.q_pu.iloc[::-1], "external")]
    prepared = prepare_source_inputs(scenarios, reverse)
    a = module.fit_tree(scenarios, sources, "NJ")
    b = module.fit_tree(scenarios, prepared, "RNJ")
    np.testing.assert_allclose(a["R"], b["R"], atol=1e-12)
    np.testing.assert_allclose(a["X"], b["X"], atol=1e-12)


def test_bootstrap_retains_failure_and_fixed_requested_denominator(monkeypatch):
    scenarios, sources, _, _, _ = _problem(2)
    clade = frozenset({11, 22})
    calls = []

    def fake_fit(scenarios, sources, method, *, root_bus):
        calls.append((scenarios, sources))
        if len(calls) == 2:
            raise RuntimeError("deliberate bootstrap failure")
        return {"method": "RNJ", "root_bus": root_bus, "clades": {clade},
                "terminals": (11, 22, 33), "elapsed_seconds": .01}

    monkeypatch.setattr(module, "fit_tree", fake_fit)
    result = module.select_rnj_initial_supports(scenarios, sources, replicates=4, threshold=.75)
    assert result["confidence"][clade] == .75
    assert result["selected"] == [clade]
    assert result["selected_support_indices"] == [(0, 1)]
    assert result["bootstrap_successes"] == 3
    assert result["bootstrap_failures"][0]["replicate"] == 0
    assert result["status"] == "complete"
    assert result["truth_used_for_selection"] is False


def test_bootstrap_wall_limit_preserves_unattempted_denominator(monkeypatch):
    scenarios, sources, _, _, _ = _problem(2)
    clade = frozenset({11, 22})
    full_fit = {"method": "RNJ", "root_bus": 1, "clades": {clade},
                "terminals": (11, 22, 33), "elapsed_seconds": .01}
    clock = iter([0, 0, 2, 2, 2])
    monkeypatch.setattr(module, "perf_counter", lambda: next(clock))
    result = module.select_rnj_initial_supports(
        scenarios, sources, replicates=4, wall_seconds=1, full_fit=full_fit,
    )
    assert result["status"] == "bootstrap_time_limit"
    assert result["bootstrap_attempted"] == 0
    assert result["bootstrap_unattempted"] == 4
    assert result["bootstrap_successes"] == 0
    assert result["confidence"][clade] == 0
    assert result["selected"] == []


def test_invalid_source_and_root_are_rejected():
    scenarios, sources, _, _, _ = _problem()
    source = sources[0]
    with pytest.raises(ValueError, match="same unique labels"):
        module.fit_tree(scenarios, [SourceInputs(source.p_pu.iloc[:-1], source.q_pu, "external")], "NJ")
    with pytest.raises(ValueError, match="distinct"):
        module.fit_tree(scenarios, sources, "RNJ", root_bus=11)
    with pytest.raises(ValueError, match="method"):
        module.fit_tree(scenarios, sources, "WZZT")
