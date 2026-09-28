"""Independent truth, split-isolation and clustered paired-statistic checks."""

from copy import deepcopy
import importlib.util
import inspect
import json
import os
from types import SimpleNamespace
from itertools import combinations
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest


# The standalone core's experiments folder is intentionally not an installed
# package. Do not accidentally import the parent repository's experiments.
_path = Path(__file__).resolve().parents[1] / "experiments" / "stress_support.py"
_spec = importlib.util.spec_from_file_location("independent_stress_protocol_support", _path)
support = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = support
_spec.loader.exec_module(support)


def _kernel_clades(matrix, terminals):
    """Read shared-path clusters from level sets, without the support list.

    At each positive off-diagonal shared depth, connected components are the
    descendants of a branching point. This gives an independent tree oracle.
    """
    n = len(terminals)
    result = set()
    thresholds = np.unique(matrix[np.triu_indices(n, 1)])
    for threshold in thresholds[thresholds > 0]:
        remaining = set(range(n))
        while remaining:
            first = remaining.pop()
            component, frontier = {first}, [first]
            while frontier:
                vertex = frontier.pop()
                neighbors = {j for j in remaining if matrix[vertex, j] >= threshold - 1e-12}
                component |= neighbors
                remaining -= neighbors
                frontier.extend(neighbors)
            if 1 < len(component) < n:
                result.add(frozenset(terminals[i] for i in component))
    return result


@pytest.mark.parametrize("shape", ["balanced", "caterpillar", "random", "multifurcating", "star"])
@pytest.mark.parametrize("n", [2, 3, 8, 17])
@pytest.mark.parametrize("seed", [11, 91])
def test_generated_truth_is_a_reduced_positive_tree_by_independent_geometry(shape, n, seed):
    truth = support.generate_tree(shape, n, seed)
    assert len(set(truth.terminals)) == n
    assert set(truth.supports) >= {(i,) for i in range(n)}
    assert len(set(truth.supports)) == len(truth.supports)
    assert all(1 <= len(s) < n for s in truth.supports)
    assert np.all(truth.r_weights > 0) and np.all(truth.x_weights > 0)
    if shape in ("balanced", "caterpillar", "random"):
        assert len(truth.supports) == 2 * n - 2
    if shape == "star":
        assert len(truth.supports) == n
        assert truth.clades == set()
    for matrix in (truth.r, truth.x):
        np.testing.assert_allclose(matrix, matrix.T, atol=0.0)
        assert np.linalg.eigvalsh(matrix).min() > 0.0
        assert matrix.min() >= 0.0
        off = matrix[np.triu_indices(n, 1)]
        assert off.min() == 0.0  # The experiment intentionally has no root stem.
        assert np.min((np.diag(matrix)[:, None] - matrix)[~np.eye(n, dtype=bool)]) > 0.0
        assert _kernel_clades(matrix, truth.terminals) == truth.clades
        distance = np.diag(matrix)[:, None] + np.diag(matrix)[None, :] - 2 * matrix
        for a, b, c in combinations(range(n), 3):
            shared_depths = sorted([matrix[a, b], matrix[a, c], matrix[b, c]])
            assert shared_depths[0] == pytest.approx(shared_depths[1], abs=1e-12)
        # The additive-tree four-point condition independently checks path
        # geometry rather than repeating the edge-contribution constructor.
        for a, b, c, d in combinations(range(n), 4):
            sums = sorted([distance[a, b] + distance[c, d],
                           distance[a, c] + distance[b, d],
                           distance[a, d] + distance[b, c]])
            assert sums[1] == pytest.approx(sums[2], abs=1e-11)


@pytest.mark.parametrize("shape", ["balanced", "caterpillar", "multifurcating", "star"])
def test_weak_internal_edges_preserve_truth_and_pendant_edge_lengths(shape):
    regular = support.generate_tree(shape, 9, 88)
    weak = support.generate_tree(shape, 9, 88, internal_scale=0.01)
    assert regular.supports == weak.supports
    assert regular.clades == weak.clades
    pendant = np.array([len(s) == 1 for s in regular.supports])
    for a, b in ((regular.r_weights, weak.r_weights), (regular.x_weights, weak.x_weights)):
        np.testing.assert_array_equal(a[pendant], b[pendant])
        np.testing.assert_allclose(0.01 * a[~pendant], b[~pendant])


@pytest.mark.parametrize("stress", ["exact", "weak_internal", "noise_02", "noise_10", "x_noisy", "r_noisy", "diagonal_only"])
def test_matrix_perturbation_is_paired_reproducible_and_does_not_mutate_truth(stress):
    truth = support.generate_tree("random", 7, 91)
    original = deepcopy(truth)
    r, x = support.perturb_matrices(truth, stress, 444)
    repeat = support.perturb_matrices(truth, stress, 444)
    np.testing.assert_array_equal([r, x], repeat)
    np.testing.assert_array_equal(truth.r, original.r)
    np.testing.assert_array_equal(truth.x, original.x)
    assert truth.supports == original.supports
    for observed, clean in ((r, truth.r), (x, truth.x)):
        np.testing.assert_array_equal(observed, observed.T)
        assert not np.shares_memory(observed, clean)
        if stress in ("exact", "weak_internal"):
            np.testing.assert_array_equal(observed, clean)
        elif stress == "diagonal_only":
            mask = ~np.eye(len(r), dtype=bool)
            np.testing.assert_array_equal(observed[mask], clean[mask])
        else:
            assert not np.array_equal(observed, clean)


@pytest.mark.parametrize("stress", ["reference", "low_sample", "high_noise", "pq_collinear", "terminal_correlated", "outliers", "weak_internal", "common_noise"])
def test_synthetic_splits_are_reproducible_disjoint_and_physically_self_consistent(stress):
    truth = support.generate_tree("balanced", 6, 12)
    original = deepcopy(truth)
    splits, matrices = support.synthetic_splits(truth, stress, 444, samples=23)
    repeat, _ = support.synthetic_splits(truth, stress, 444, samples=23)
    changed, _ = support.synthetic_splits(truth, stress, 445, samples=23)
    np.testing.assert_array_equal(matrices, [0.001 * truth.r, 0.001 * truth.x])
    np.testing.assert_array_equal(truth.r, original.r)
    np.testing.assert_array_equal(truth.x, original.x)
    assert truth.supports == original.supports
    assert list(splits) == ["train", "validation", "test"]
    all_frames = []
    for name, scenarios in splits.items():
        assert len(scenarios) == 3
        expected_count = 96 if name == "test" else (8 if stress == "low_sample" else 23)
        for day, scenario in enumerate(scenarios):
            assert scenario["name"] == f"scenario_{day}"
            for key in ("P_terminal", "Q_terminal", "drop_target", "V_terminal"):
                frame = scenario[key]
                assert frame.shape == (expected_count, 6)
                assert list(frame.columns) == truth.terminals
                assert np.isfinite(frame.to_numpy()).all()
                pd.testing.assert_frame_equal(frame, repeat[name][day][key])
                assert not frame.equals(changed[name][day][key])
            squared_drop = scenario["root_voltage"].to_numpy()[:, None]**2 - scenario["V_terminal"].to_numpy()**2
            np.testing.assert_allclose(squared_drop, scenario["drop_target"], atol=3e-16)
            all_frames.append(scenario["P_terminal"])
    for first, second in combinations(all_frames, 2):
        assert not np.shares_memory(first.to_numpy(), second.to_numpy())
        count = min(len(first), len(second))
        assert not np.array_equal(first.to_numpy()[:count], second.to_numpy()[:count])
    saved_validation = splits["validation"][0]["P_terminal"].copy(deep=True)
    splits["train"][0]["P_terminal"].iloc[0, 0] = 1e20
    pd.testing.assert_frame_equal(saved_validation, splits["validation"][0]["P_terminal"])


@pytest.mark.parametrize("samples", [9, 32, 65])
def test_test_set_does_not_change_when_training_sample_budget_changes(samples):
    truth = support.generate_tree("random", 5, 96)
    original, _ = support.synthetic_splits(truth, "reference", 788, samples=32)
    resized, _ = support.synthetic_splits(truth, "reference", 788, samples=samples)
    for left, right in zip(original["test"], resized["test"]):
        for key in ("P_terminal", "Q_terminal", "drop_target", "V_terminal"):
            pd.testing.assert_frame_equal(left[key], right[key])


def test_power_factor_stress_has_exact_ambiguity_in_every_split_and_scenario():
    truth = support.generate_tree("random", 5, 96)
    splits, _ = support.synthetic_splits(truth, "pq_collinear", 788)
    for scenarios in splits.values():
        designs = []
        for scenario in scenarios:
            p = scenario["P_terminal"].to_numpy()
            q = scenario["Q_terminal"].to_numpy()
            np.testing.assert_array_equal(q, 0.6 * p)
            a = np.column_stack([p - p.mean(axis=0), q - q.mean(axis=0)])
            assert np.linalg.matrix_rank(a) == len(truth.terminals)
            designs.append(a)
        assert np.linalg.matrix_rank(np.vstack(designs)) == len(truth.terminals)


def _row(method, job, cluster, score, *, tier="linear", stress="reference"):
    return dict(tier=tier, stress=stress, method=method, job_id=job, cluster_id=cluster, clade_f1=score)


def _summary(rows, **kwargs):
    items = support.paired_summary(rows, "reference", **kwargs)
    assert len(items) == 1
    return items[0]


def test_paired_summary_matches_jobs_despite_row_reordering_and_preserves_input():
    rows = [_row("candidate", "b", "tree_b", 0.7), _row("reference", "a", "tree_a", 0.2),
            _row("candidate", "a", "tree_a", 0.5), _row("reference", "b", "tree_b", 0.8)]
    original = deepcopy(rows)
    summary = _summary(rows)
    assert summary == _summary(list(reversed(rows)))
    assert rows == original
    assert summary["attempted"] == summary["paired_available"] == 2
    assert summary["missing"] == 0
    assert (summary["wins"], summary["ties"], summary["losses"]) == (1, 0, 1)
    assert summary["mean_delta"] == pytest.approx(0.1)
    assert summary["cluster_count"] == 2


@pytest.mark.parametrize("missing_kind", ["absent_row", "none_score", "nan_score"])
def test_paired_summary_counts_missing_comparisons_on_union_of_jobs(missing_kind):
    rows = [_row("reference", "a", "tree_a", 0.4), _row("candidate", "a", "tree_a", 0.6),
            _row("reference", "b", "tree_b", 0.5), _row("candidate", "c", "tree_c", 0.7)]
    if missing_kind != "absent_row":
        missing_score = None if missing_kind == "none_score" else np.nan
        rows.extend([_row("candidate", "b", "tree_b", missing_score),
                     _row("reference", "c", "tree_c", missing_score)])
    summary = _summary(rows)
    assert summary["attempted"] == 3
    assert summary["paired_available"] == 1
    assert summary["missing"] == 2
    assert summary["mean_delta"] == pytest.approx(0.2)


@pytest.mark.parametrize("method", ["reference", "candidate"])
def test_duplicate_method_job_rows_are_rejected_instead_of_cartesian_paired(method):
    rows = [_row("reference", "a", "tree_a", 0.4), _row("candidate", "a", "tree_a", 0.6)]
    rows.append(deepcopy(next(row for row in rows if row["method"] == method)))
    with pytest.raises(ValueError, match="duplicat|unique"):
        support.paired_summary(rows, "reference")


@pytest.mark.parametrize("cluster", [None, np.nan, "different_tree"])
def test_invalid_or_disagreeing_cluster_assignment_is_rejected(cluster):
    rows = [_row("reference", "a", "tree_a", 0.4), _row("candidate", "a", cluster, 0.6)]
    with pytest.raises(ValueError, match="cluster"):
        support.paired_summary(rows, "reference")


def test_cluster_resampling_retains_whole_trees_and_row_weighted_estimand():
    rows = []
    for job in ("a1", "a2", "a3"):
        rows.extend([_row("reference", job, "tree_a", 0.8), _row("candidate", job, "tree_a", 1.0)])
    rows.extend([_row("reference", "b1", "tree_b", 0.8), _row("candidate", "b1", "tree_b", 0.0)])
    summary = _summary(rows, seed=444, bootstrap=2000)
    assert summary["cluster_count"] == 2
    # Mean across fixed-bank rows: (3*0.2 - 0.8)/4 = -0.05. A mean of
    # per-cluster means would be -0.3 and is a different estimand.
    assert summary["mean_delta"] == pytest.approx(-0.05)
    # Resampling two entire clusters produces only -0.8, -0.05, or 0.2;
    # each endpoint has probability 1/4, so the percentile CI spans both.
    assert summary["ci_low"] == pytest.approx(-0.8)
    assert summary["ci_high"] == pytest.approx(0.2)
    replicated = [dict(row, job_id=f'{row["job_id"]}_{i}') for i in range(5) for row in rows]
    again = _summary(replicated, seed=444, bootstrap=2000)
    assert again["cluster_count"] == 2
    assert again["mean_delta"] == pytest.approx(summary["mean_delta"])
    assert again["ci_low"] == pytest.approx(summary["ci_low"])
    assert again["ci_high"] == pytest.approx(summary["ci_high"])


def test_same_cluster_rows_are_not_reported_as_independent_trees():
    rows = []
    for i in range(20):
        rows.extend([_row("reference", str(i), "one_tree", 0.1),
                     _row("candidate", str(i), "one_tree", 0.3)])
    summary = _summary(rows)
    assert summary["paired_available"] == 20
    assert summary["cluster_count"] == 1
    # This degeneracy is a property of the bootstrap, not cross-tree evidence.
    assert summary["ci_low"] == pytest.approx(0.2)
    assert summary["ci_high"] == pytest.approx(0.2)


def test_empty_or_entirely_missing_comparison_does_not_create_a_numeric_ci():
    assert support.paired_summary([], "reference") == []
    rows = [_row("reference", "a", "tree_a", None), _row("candidate", "a", "tree_a", np.nan)]
    summary = _summary(rows)
    assert summary["paired_available"] == 0
    assert summary["missing"] == 1
    assert summary.get("mean_delta") is None
    assert summary.get("ci_low") is None
    assert summary.get("ci_high") is None


@pytest.mark.parametrize("bootstrap", [0, -1, 1.5])
def test_invalid_bootstrap_count_is_rejected(bootstrap):
    rows = [_row("reference", "a", "tree_a", 0.2), _row("candidate", "a", "tree_a", 0.5)]
    with pytest.raises(ValueError):
        support.paired_summary(rows, "reference", bootstrap=bootstrap)




@pytest.fixture
def runner(monkeypatch):
    """Import the CLI without leaking its path/thread-setting side effects."""
    monkeypatch.setattr(sys, "path", sys.path.copy())
    monkeypatch.syspath_prepend(str(_path.parent))
    for key in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
        monkeypatch.setenv(key, os.environ.get(key, "1"))
    spec = importlib.util.spec_from_file_location(
        "independent_advantage_stress_protocol", _path.parent / "run_advantage_stress.py",
    )
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("tier", ["geometry", "synthetic", "ac"])
def test_top_level_job_failure_retains_every_predeclared_method(runner, monkeypatch, tmp_path, tier):
    def fail(_job):
        raise RuntimeError("deliberate generator failure before the first method")

    monkeypatch.setattr(runner, "geometry_job", fail)
    monkeypatch.setattr(runner, "measurement_job", fail)
    job = dict(tier=tier, stress="reference", shape="balanced", n=4, repeat=0,
               job_id="failed_job", cluster_id="tree_0", seed=7, output=str(tmp_path))
    rows = runner.safe_job(job)
    expected = runner.GEOMETRY_METHODS if tier == "geometry" else runner.MEASUREMENT_METHODS
    assert {row["method"] for row in rows} == set(expected)
    assert len(rows) == len(expected)
    for row in rows:
        assert row["status"] == "job_failed"
        assert row["job_id"] == "failed_job"
        assert row["cluster_id"] == "tree_0"
        assert row["clade_f1"] is None
        assert row["clade_exact"] is None
        assert row["elapsed_seconds"] is None
    failures = list((tmp_path / "job_failures").glob("*.json"))
    assert len(failures) == 1
    assert "deliberate generator failure" in json.loads(failures[0].read_text(encoding="utf-8"))["error"]
    comparisons = support.paired_summary(rows, "rnj_RX75")
    assert len(comparisons) == len(expected) - 1
    assert all(item["attempted"] == item["missing"] == 1 for item in comparisons)
    assert all(item["paired_available"] == 0 for item in comparisons)


@pytest.mark.parametrize("offset", [0.0, 0.2, 100.0])
def test_validation_mae_preserves_unexplained_offsets_without_truth(runner, monkeypatch, offset):
    def forbidden(*_args, **_kwargs):
        pytest.fail("validation-only selection must not call truth/test reporting")

    monkeypatch.setattr(runner, "prediction_metrics", forbidden)
    r = np.array([[2.0, 0.3], [0.8, 1.0]])
    x = np.array([[0.7, 0.1], [0.4, 0.9]])
    p1 = np.array([[1.0, 2.0], [-3.0, 4.0]])
    q1 = np.array([[0.5, -2.0], [1.0, 3.0]])
    p2, q2 = np.array([[4.0, 2.0]]), np.array([[1.0, 7.0]])
    errors = [np.array([[1.0, -2.0], [-3.0, 4.0]]) + offset,
              np.array([[5.0, -6.0]]) + offset]
    scenarios = []
    for p, q, error in zip((p1, p2), (q1, q2), errors):
        scenarios.append({"P_terminal": pd.DataFrame(p), "Q_terminal": pd.DataFrame(q),
                          "drop_target": pd.DataFrame(p @ r.T + q @ x.T + error)})
    original = deepcopy(scenarios)
    actual = runner.validation_mae(scenarios, r, x)
    expected = np.mean(np.abs(np.vstack(errors)))
    assert actual == pytest.approx(expected, abs=1e-12)
    assert "truth" not in inspect.signature(runner.validation_mae).parameters
    for scenario, saved in zip(scenarios, original):
        for key in scenario:
            pd.testing.assert_frame_equal(scenario[key], saved[key])


@pytest.mark.parametrize("stress", ["reference", "low_sample", "high_noise"])
def test_ac_split_calls_match_declared_noise_replica_and_sample_protocol(runner, monkeypatch, stress):
    import rnj_wzzt.scenario.simulation as simulation
    import rnj_wzzt.models.lin_distflow as model
    import rnj_wzzt.reporting as reporting

    simulator_signature = inspect.signature(simulation._simulate_pool)
    calls, raw_pools = [], []
    net = SimpleNamespace(root_bus=0, branches=pd.DataFrame({"from_bus": [0, 0], "to_bus": [10, 20]}))

    def simulate(*args, **kwargs):
        bound = simulator_signature.bind(*args, **kwargs)
        bound.apply_defaults()
        values = bound.arguments
        calls.append(dict(values))
        raw = []
        for day in range(3):
            data = np.arange(192).reshape(96, 2) + 1000 * values["replicate"] + day
            settings = {key: values[key] for key in ("root_observation", "root_meter_noise_rel", "scenario_suite")}
            raw.append({"name": f"original_{day}", "P_terminal": pd.DataFrame(data, columns=[10, 20]),
                        "Q_terminal": pd.DataFrame(data / 2, columns=[10, 20]),
                        "drop_target": pd.DataFrame(data / 10000, columns=[10, 20]),
                        "diagnostics": {"source_replicate": values["replicate"]}, "scenario_settings": settings})
        raw_pools.append(raw)
        return net, raw

    monkeypatch.setattr(simulation, "_simulate_pool", simulate)
    monkeypatch.setattr(simulation, "_terminal_buses", lambda _net: [10, 20])
    monkeypatch.setattr(model, "build_reduced_sensitivity_matrices", lambda *_args, **_kwargs: (np.eye(2), 0.5 * np.eye(2)))
    monkeypatch.setattr(reporting, "_truth_nontrivial_clades", lambda *_args: set())
    sets, terminals, truth, clades, metadata = runner.ac_splits({
        "shape": "paper15", "stress": stress, "repeat": 4, "samples": 24,
    })
    assert len(calls) == 3
    expected_noise = 0.001 if stress == "high_noise" else 0.0002
    for split_id, (split, scenarios) in enumerate(sets.items()):
        call = calls[split_id]
        assert call["case_key"] == "paper15"
        assert call["t_count"] == 96
        assert call["replicate"] == 12 + split_id
        assert call["maximum_scenarios"] == 3
        assert call["pq_noise_rel"] == 0.005
        assert call["v_noise_rel"] == expected_noise
        assert call["scenario_suite"] == "reference"
        assert call["root_observation"] == "noisy"
        assert call["root_meter_noise_rel"] == 0.0002
        count = 96 if split == "test" else (8 if stress == "low_sample" else 24)
        assert len(scenarios) == 3
        for day, scenario in enumerate(scenarios):
            assert scenario["name"] == f"scenario_{day}"
            assert len(scenario["P_terminal"]) == count
            assert scenario["P_terminal"].index.is_unique
            assert not np.shares_memory(scenario["P_terminal"].to_numpy(), raw_pools[split_id][day]["P_terminal"].to_numpy())
            settings = metadata["scenario_settings"][split][day]
            assert settings["root_observation"] == "noisy"
            assert settings["root_meter_noise_rel"] == 0.0002
    assert metadata["terminal_meter_noise_rel"] == expected_noise
    assert terminals == [10, 20]
    assert clades == set()
    np.testing.assert_array_equal(truth, [np.eye(2), 0.5 * np.eye(2)])

