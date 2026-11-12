"""Mock-only protocol tests for the post-result matched RNJ tuning script."""
from copy import deepcopy
import importlib.util
import inspect
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def tuning(monkeypatch):
    monkeypatch.setattr(sys, "path", sys.path.copy())
    for key in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
        monkeypatch.setenv(key, os.environ.get(key, "1"))
    path = Path(__file__).resolve().parents[1] / "rnj" / "scripts" / "run_matched_rnj_tuning.py"
    spec = importlib.util.spec_from_file_location("independent_matched_rnj_tuning", path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def _job():
    return dict(tier="synthetic", shape="balanced", n=3, repeat=0, stress="reference",
                seed=91, cluster_id="tree_0", job_id="job_0", solve_seconds=1.5)


def _write_cache(directory):
    directory.mkdir(parents=True)
    terminals = [10, 30, 20]
    arrays = {"terminals": terminals, "R_true": np.eye(3), "X_true": 0.5 * np.eye(3)}
    metadata = {"root": 0, "truth_clades": [[10, 30]], "job": _job(),
                "source_sample_indices": {}, "snapshots": {}}
    for split_id, (split, count) in enumerate((("train", 5), ("validation", 7), ("test", 9))):
        metadata["source_sample_indices"][split] = []
        metadata["snapshots"][split] = 2 * count
        for day in range(2):
            index = list(np.arange(count, dtype=int) * 3 + 1)
            metadata["source_sample_indices"][split].append([int(i) for i in index])
            p = np.arange(count * 3, dtype=float).reshape(count, 3) + 100 * split_id + 10 * day
            for key, value in (("P_terminal", p), ("Q_terminal", p / 2),
                               ("drop_target", p / 1000), ("root_voltage", np.ones(count))):
                arrays[f"{split}_{day}_{key}"] = value
    np.savez_compressed(directory / "inputs.npz", **arrays)
    (directory / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    return arrays, metadata


def test_cached_arrays_restore_original_labels_indices_and_values(tuning, tmp_path):
    arrays, metadata = _write_cache(tmp_path / "job")
    cached = tuning.load_cached_job(tmp_path / "job")
    assert cached["terminals"] == [10, 30, 20]
    assert cached["true_clades"] == {frozenset({10, 30})}
    assert cached["root"] == 0
    for split, scenarios in cached["sets"].items():
        for day, scenario in enumerate(scenarios):
            assert scenario["name"] == f"scenario_{day}"
            for key in tuning.OBSERVATION_KEYS:
                np.testing.assert_array_equal(scenario[key], arrays[f"{split}_{day}_{key}"])
                assert list(scenario[key].columns) == [10, 30, 20]
                assert list(scenario[key].index) == metadata["source_sample_indices"][split][day]
            assert isinstance(scenario["root_voltage"], pd.Series)
    assert len(cached["input_sha256"]) == len(cached["metadata_sha256"]) == 64
    repeated = tuning.load_cached_job(tmp_path / "job")
    cached["sets"]["train"][0]["P_terminal"].iloc[0, 0] = -9999
    np.testing.assert_array_equal(repeated["sets"]["train"][0]["P_terminal"], arrays["train_0_P_terminal"])


@pytest.mark.parametrize("defect", ["missing", "shape", "nan", "duplicate_labels", "duplicate_index", "count"])
def test_invalid_cache_fails_explicitly(tuning, tmp_path, defect):
    arrays, metadata = _write_cache(tmp_path / "job")
    if defect == "missing":
        del arrays["validation_0_Q_terminal"]
    elif defect == "shape":
        arrays["train_0_drop_target"] = np.zeros((5, 2))
    elif defect == "nan":
        arrays["test_1_P_terminal"][0, 0] = np.nan
    elif defect == "duplicate_labels":
        arrays["terminals"] = [10, 10, 20]
    elif defect == "duplicate_index":
        metadata["source_sample_indices"]["train"][0] = [1] * 5
    else:
        metadata["snapshots"]["test"] = 1000
    np.savez_compressed(tmp_path / "job" / "inputs.npz", **arrays)
    (tmp_path / "job" / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError):
        tuning.load_cached_job(tmp_path / "job")


def _selection_inputs(target=0.08):
    scenarios = [{"name": "scenario_0", "P_terminal": pd.DataFrame(np.ones((4, 2)), columns=[10, 20]),
                  "Q_terminal": pd.DataFrame(np.zeros((4, 2)), columns=[10, 20]),
                  "drop_target": pd.DataFrame(np.full((4, 2), target), columns=[10, 20])}]
    return deepcopy(scenarios), deepcopy(scenarios)


def _mock_models(tuning, monkeypatch, *, fail=(), tie=False, nonfinite=False):
    def regression(_training, *, diagnostics):
        diagnostics["success"] = True
        return np.eye(2), np.eye(2), 1.0, 1.0

    def infer(_r, _x, _terminals, _root, factor):
        if factor in fail:
            raise RuntimeError(f"injected inference failure {factor}")
        return {frozenset({int(round(100 * factor)), 100})}

    def fixed(train, clades, terminals, solve_seconds):
        assert solve_seconds == 1.5
        factor = min(next(iter(clades))) / 100
        coefficient = 0.0 if tie else factor
        if nonfinite:
            coefficient = np.nan
        diagnostics = SimpleNamespace(certified=True, to_dict=lambda: {"status": 0, "success": True})
        solution = SimpleNamespace(diagnostics=diagnostics)
        return solution, coefficient * np.eye(2), np.zeros((2, 2)), clades

    monkeypatch.setattr(tuning, "fit_projected_sensitivity", regression)
    monkeypatch.setattr(tuning, "infer_clades", infer)
    monkeypatch.setattr(tuning, "fixed_fit", fixed)
    monkeypatch.setattr(tuning, "solver_diagnostics_prove_optimality", lambda d: d.certified)


@pytest.mark.parametrize("tie", [False, True])
def test_selection_uses_validation_only_and_matches_nj_tie_rule(tuning, monkeypatch, tie):
    _mock_models(tuning, monkeypatch, tie=tie)
    train, validation = _selection_inputs()
    before = deepcopy((train, validation))

    def forbidden(*_args, **_kwargs):
        pytest.fail("truth/test metrics must not enter candidate selection")

    monkeypatch.setattr(tuning, "prediction_metrics", forbidden)
    monkeypatch.setattr(tuning, "rooted_scores", forbidden)
    selected = tuning.choose_rnj_on_validation(train, validation, [10, 20], 0, 1.5)
    assert selected["selected_factor"] == (0.0 if tie else 0.08)
    assert [c["factor"] for c in selected["candidates"]] == [0.0, 0.04, 0.08, 0.16]
    assert all("test_mae" not in c and "clade_f1" not in c for c in selected["candidates"])
    assert all(c["status"] == "complete" for c in selected["candidates"])
    assert "test" not in inspect.signature(tuning.choose_rnj_on_validation).parameters
    assert "truth" not in inspect.signature(tuning.choose_rnj_on_validation).parameters
    for frames, saved in zip((train, validation), before):
        for frame, copy in zip(frames, saved):
            for key in tuning.OBSERVATION_KEYS:
                pd.testing.assert_frame_equal(frame[key], copy[key])


def test_partial_candidate_failure_is_retained_without_losing_successful_selection(tuning, monkeypatch):
    _mock_models(tuning, monkeypatch, fail=(0.0, 0.04))
    train, validation = _selection_inputs()
    selected = tuning.choose_rnj_on_validation(train, validation, [10, 20], 0, 1.5)
    assert selected["selected_factor"] == 0.08
    assert [c["status"] for c in selected["candidates"]] == ["failed", "failed", "complete", "complete"]
    assert "injected inference failure" in selected["candidates"][0]["error"]


@pytest.mark.parametrize("failure", ["exception", "nonfinite"])
def test_all_failed_candidates_keep_four_diagnostic_rows(tuning, monkeypatch, failure):
    _mock_models(tuning, monkeypatch, fail=tuning.FACTORS if failure == "exception" else (),
                 nonfinite=failure == "nonfinite")
    train, validation = _selection_inputs()
    with pytest.raises(tuning.NoValidCandidate) as caught:
        tuning.choose_rnj_on_validation(train, validation, [10, 20], 0, 1.5)
    assert len(caught.value.candidates) == 4
    assert all(c["status"] == "failed" and c["validation_mae"] is None for c in caught.value.candidates)


def test_primary_completion_gate_prevents_any_fitting_or_output(tuning, monkeypatch, tmp_path):
    source, output = tmp_path / "unfinished", tmp_path / "new_output"
    source.mkdir()

    def forbidden(*_args, **_kwargs):
        pytest.fail("an incomplete primary run must not trigger model fitting")

    monkeypatch.setattr(tuning, "run_cached_job", forbidden)
    with pytest.raises(SystemExit) as error:
        tuning.main(["--source", str(source), "--output", str(output)])
    assert error.value.code == 2
    assert not output.exists()


def test_test_and_truth_changes_do_not_change_the_selected_candidate(tuning, monkeypatch, tmp_path):
    _mock_models(tuning, monkeypatch)
    train, validation = _selection_inputs()
    test = deepcopy(validation)
    metadata = {"job": _job()}
    cached = dict(sets={"train": train, "validation": validation, "test": test},
                  terminals=[10, 20], root=0, truth=(np.eye(2), np.eye(2)),
                  true_clades=set(), metadata=metadata, input_sha256="a", metadata_sha256="b")
    monkeypatch.setattr(tuning, "load_cached_job", lambda _path: cached)
    observed = []

    def evaluate(selected, sets, truth, true_clades, terminals):
        observed.append(selected["selected_factor"])
        return {"clade_f1": float(bool(true_clades)), "clade_exact": 0.0,
                "validation_mae": selected["validation_mae"],
                "test_mae": float(sets["test"][0]["drop_target"].to_numpy().mean())}

    monkeypatch.setattr(tuning, "evaluate_selected", evaluate)
    first = tuning.run_cached_job(tmp_path, tmp_path / "first", _job())
    cached["sets"]["test"][0]["drop_target"] += 1000
    cached["truth"] = (10000 * np.eye(2), 10000 * np.eye(2))
    cached["true_clades"] = {frozenset({10, 20})}
    second = tuning.run_cached_job(tmp_path, tmp_path / "second", _job())
    assert first["status"] == second["status"] == "complete"
    assert first["selected_factor"] == second["selected_factor"] == 0.08
    assert observed == [0.08, 0.08]
    assert first["test_mae"] != second["test_mae"]
    assert first["clade_f1"] != second["clade_f1"]


def test_all_job_failures_still_write_summary_pairs_and_completion(tuning, monkeypatch, tmp_path):
    source = tmp_path / "source"
    _write_cache(source / "jobs" / "job_0")
    (source / "completion.json").write_text("{}", encoding="utf-8")
    (source / "protocol.json").write_text(json.dumps({"source_sha256": {}, "jobs": [_job()]}), encoding="utf-8")
    reference = dict(_job(), method="classical_nj_validation", status="complete", clade_f1=1.0)
    (source / "rows.jsonl").write_text(json.dumps(reference) + "\n", encoding="utf-8")
    monkeypatch.setattr(tuning, "source_fingerprint", lambda _core: {})

    def fail(*_args, **_kwargs):
        raise tuning.NoValidCandidate([{"factor": f, "status": "failed", "error": "mock"} for f in tuning.FACTORS])

    monkeypatch.setattr(tuning, "choose_rnj_on_validation", fail)
    output = tmp_path / "output"
    tuning.main(["--source", str(source), "--output", str(output)])
    completion = json.loads((output / "completion.json").read_text(encoding="utf-8"))
    assert completion["jobs"] == completion["failed"] == 1
    assert completion["complete"] == 0
    assert completion["exploratory_post_result"] is True
    details = json.loads((output / "jobs" / "job_0" / "selection.json").read_text(encoding="utf-8"))
    assert len(details["candidates"]) == 4
    assert pd.read_csv(output / "summary.csv").iloc[0]["available"] == 0
    pairs = pd.read_csv(output / "paired_differences.csv")
    assert len(pairs) == 3
    assert pairs.clade_f1_delta.isna().all()


def test_missing_baseline_is_retained_in_paired_differences(tuning):
    current = dict(_job(), method=tuning.METHOD, status="complete", clade_f1=0.8, selected_factor=0.04)
    pairs = tuning.paired_differences([current], [])
    assert len(pairs) == 3
    assert all(pair["reference_status"] == "missing" and pair["clade_f1_delta"] is None for pair in pairs)

