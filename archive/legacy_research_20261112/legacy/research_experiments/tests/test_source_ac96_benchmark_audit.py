"""Independent evidence reconstruction detects saved benchmark corruption."""
import json
import hashlib
import shutil

import numpy as np
import pandas as pd
import pytest

from research_experiments.rnj.audit_source_ac96_benchmark import (
    Audit,
    CASES,
    VARIANTS,
    METHODS,
    METRICS,
    audit_benchmark,
    check_atom_model,
    clade_metrics,
    optimality_certificate,
    physical_truth,
)


def _fixture():
    supports = [(0, 1), (2, 3), (0,), (1,), (2,), (3,)]
    r = np.array([.2, .3, .4, .5, .6, .7])
    x = r / 2
    z = np.zeros((6, 4))
    for index, support in enumerate(supports):
        z[index, list(support)] = 1
    h = np.array([1, 0, 0, 0, 0, 0])
    generator = np.random.default_rng(12)
    p, q = generator.uniform(0, .03, (2, 96, 4))
    source_p = np.linspace(.01, .02, 96)
    source_q = .4 * source_p
    r_matrix, x_matrix = (z.T * r) @ z, (z.T * x) @ z
    g_r, g_x = (r * h) @ z, (x * h) @ z
    y = p @ r_matrix.T + q @ x_matrix.T + source_p[:, None] * g_r + source_q[:, None] * g_x
    model = {
        "terminal_labels": [4, 5, 6, 7], "support_indices": supports,
        "r_values": r.tolist(), "x_values": x.tolist(), "source_h": h.tolist(),
        "source_g_r": g_r.tolist(), "source_g_x": g_x.tolist(), "train_mae": 0.,
        "r_upper_bound": 2., "x_upper_bound": 2., "clades": [[4, 5], [6, 7]],
        "matrix_origin": "actual_accepted_atoms",
        "solver_events": [{"accepted": True, "status": 0, "success": True,
                           "binary_variable_count": 6, "objective": 0., "dual_bound": 0., "mip_gap": None}],
    }
    arrays = {"R": r_matrix, "X": x_matrix, "P": p, "Q": q, "Y": y,
              "P_source": source_p, "Q_source": source_q}
    return model, arrays


def _failed_checks(audit):
    return [item["check"] for item in audit.checks if not item["passed"]]


def test_physical_path_truth_uses_units_and_squared_voltage_factor(tmp_path):
    metadata = {"root_bus": 1, "base_kv": .4, "base_mva": .2}
    (tmp_path / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    pd.DataFrame({"bus_id": list(range(1, 8))}).to_csv(tmp_path / "buses.csv", index=False)
    r = [.08, .12, .16, .20, .24, .28]
    edges = [(1, 2), (1, 3), (2, 4), (2, 5), (3, 6), (3, 7)]
    pd.DataFrame([{"from_bus": a, "to_bus": b, "r_ohm": rr, "x_ohm": rr / 2,
                   "is_true_closed": True} for (a, b), rr in zip(edges, r)]).to_csv(tmp_path / "branches.csv", index=False)
    result = physical_truth(tmp_path, [4, 5, 6, 7])
    expected = np.array([[.6, .2, 0, 0], [.2, .7, 0, 0], [0, 0, .9, .3], [0, 0, .3, 1.]])
    np.testing.assert_allclose(result["R"], expected)
    np.testing.assert_allclose(result["X"], expected / 2)
    assert result["clades"] == {frozenset((4, 5)), frozenset((6, 7))}
    assert result["zbase_ohm"] == pytest.approx(.8)


def test_saved_atom_model_passes_independent_reconstruction():
    model, arrays = _fixture()
    audit = Audit()
    value = check_atom_model(model, arrays, audit)
    assert value == pytest.approx(0.)
    assert audit.result()["passed"], _failed_checks(audit)


@pytest.mark.parametrize("corruption, expected_check", [
    ("matrix", "R reconstructed from atoms"),
    ("source_g", "source g_r from atom h"),
    ("source_path", "complete ancestor and disjoint source path constraints"),
    ("objective", "full training mean absolute objective"),
    ("bound", "r bounded atom weights"),
    ("certificate", "accepted solve 0 certified"),
    ("clade", "reported clades match active atoms"),
])
def test_corrupted_evidence_is_rejected(corruption, expected_check):
    model, arrays = _fixture()
    if corruption == "matrix":
        arrays["R"][0, 0] += .01
    elif corruption == "source_g":
        model["source_g_r"][0] += .01
    elif corruption == "source_path":
        model["source_h"] = [0, 0, 1, 0, 0, 0]
    elif corruption == "objective":
        model["train_mae"] = .01
    elif corruption == "bound":
        model["r_upper_bound"] = .1
    elif corruption == "certificate":
        model["solver_events"][0]["dual_bound"] = None
    elif corruption == "clade":
        model["clades"] = [[4, 5]]
    audit = Audit()
    check_atom_model(model, arrays, audit)
    assert any(expected_check in value for value in _failed_checks(audit)), _failed_checks(audit)


def test_source_nuisance_is_independently_refit_for_rank_one_design():
    model, arrays = _fixture()
    for key in ("support_indices", "r_values", "x_values", "source_h"):
        model[key] = None
    model["matrix_origin"] = "source_adjusted_ordered_regression"
    source = np.column_stack((arrays["P_source"], arrays["Q_source"]))
    residual = arrays["Y"] - arrays["P"] @ arrays["R"].T - arrays["Q"] @ arrays["X"].T
    coefficients = np.linalg.lstsq(source, residual, rcond=None)[0]
    model["source_g_r"], model["source_g_x"] = coefficients.tolist()
    audit = Audit()
    check_atom_model(model, arrays, audit)
    assert audit.result()["passed"], _failed_checks(audit)
    model["source_g_r"][0] += .1
    audit = Audit()
    check_atom_model(model, arrays, audit)
    assert any("Source nuisance g independently refit" in value for value in _failed_checks(audit))


def test_status_zero_without_gap_or_bound_is_not_a_mip_certificate():
    event = {"status": 0, "success": True, "binary_variable_count": 2}
    assert not optimality_certificate(event)
    assert not optimality_certificate({**event, "mip_gap": 1e-6})
    assert optimality_certificate({**event, "mip_gap": 0.})
    assert optimality_certificate({**event, "objective": .01, "dual_bound": .01})
    assert optimality_certificate({**event, "binary_variable_count": 0})
    assert not optimality_certificate({**event, "status": 1, "mip_gap": 0.})


def test_f1_uses_nontrivial_terminal_clades_and_keeps_zero_recovery():
    truth = [[4, 5], [6, 7]]
    score = clade_metrics([[4], [4, 5, 6, 7], [4, 5]], truth, [4, 5, 6, 7])
    assert score["precision"] == 1
    assert score["recall"] == .5
    assert score["clade_f1"] == pytest.approx(2 / 3)
    assert clade_metrics([], truth, [4, 5, 6, 7])["clade_f1"] == 0.


def test_incomplete_and_failed_jobs_keep_64_rows_with_null_metrics(tmp_path):
    job = tmp_path / "jobs" / "paper15__constant__WZZT"
    job.mkdir(parents=True)
    result = {"case": "paper15", "variant": "constant", "method": "WZZT", "status": "failed",
              "stop_reason": "numerical_failure", "elapsed_seconds": 2., "search_complete": False,
              "metrics": {"clade_f1": None, "r_relative_error": 0.}}
    (job / "result.json").write_text(json.dumps(result), encoding="utf-8")
    output = audit_benchmark(tmp_path, tmp_path / "dataset")
    assert len(output["metrics_recomputed"]) == 64
    row = next(value for value in output["metrics_recomputed"] if value["case"] == "paper15"
               and value["variant"] == "constant" and value["method"] == "WZZT")
    assert row["status"] == "failed"
    assert row["r_relative_error"] is None
    failed = [value["check"] for value in output["checks"] if not value["passed"]]
    assert "all 64 attempted jobs retained" in failed
    assert "paper15__constant__WZZT: missing output metrics remain null" in failed


def _failed_directory_fixture(tmp_path):
    """Create a complete saved-output protocol without fitting any estimator."""
    output, dataset = tmp_path / "run", tmp_path / "data"
    model, fixture_arrays = _fixture()
    labels = model["terminal_labels"]
    for case in CASES:
        for variant in VARIANTS:
            arrays = {name: value.copy() for name, value in fixture_arrays.items() if name not in {"R", "X"}}
            if variant == "no_source":
                arrays["P_source"].fill(0)
                arrays["Q_source"].fill(0)
            condition = dataset / case / variant
            observed, source = condition / "observed", condition / "source"
            observed.mkdir(parents=True)
            source.mkdir()
            for name, filename in (("P", "P_terminal.csv"), ("Q", "Q_terminal.csv"), ("Y", "drop_target.csv")):
                pd.DataFrame(arrays[name], columns=labels).to_csv(observed / filename, float_format="%.17g")
            pd.DataFrame(np.ones((96, 4)), columns=labels).to_csv(observed / "V_terminal.csv")
            for filename in ("root_voltage.csv", "P0_measured.csv", "Q0_measured.csv"):
                pd.DataFrame({"value": np.ones(96)}).to_csv(observed / filename)
            (observed / "metadata.json").write_text("{}", encoding="utf-8")
            pd.DataFrame({"P_pu": arrays["P_source"], "Q_pu": arrays["Q_source"],
                          "P_kw": 200 * arrays["P_source"], "Q_kvar": 200 * arrays["Q_source"]}).to_csv(source / "amplitudes.csv", float_format="%.17g")
            (source / "metadata.json").write_text("{}", encoding="utf-8")
            copied = output / "inputs" / f"{case}__{variant}"
            shutil.copytree(condition, copied)
            file_hashes = {path.relative_to(condition).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                           for path in condition.rglob("*") if path.is_file()}
            for method in METHODS:
                job = output / "jobs" / f"{case}__{variant}__{method}"
                job.mkdir(parents=True)
                result = {"case": case, "variant": variant, "method": method,
                          "status": "failed", "stop_reason": "test_failure", "elapsed_seconds": 1.,
                          "has_model": False, "search_complete": False, "metrics": dict.fromkeys(METRICS)}
                saved_model = {"terminal_labels": labels, "has_model": False, "status": "failed",
                               "clades": [], "source_g_r": None, "source_g_x": None}
                (job / "result.json").write_text(json.dumps(result), encoding="utf-8")
                (job / "model.json").write_text(json.dumps(saved_model), encoding="utf-8")
                (job / "inputs_hash.json").write_text(json.dumps({"file_sha256": file_hashes}), encoding="utf-8")
                np.savez_compressed(job / "model.npz", R=np.empty((0, 0)), X=np.empty((0, 0)), **arrays)
    return output, dataset


def test_directory_audit_smoke_preserves_all_64_failure_checkpoints(tmp_path):
    output, dataset = _failed_directory_fixture(tmp_path)
    result = audit_benchmark(output, dataset)
    assert result["passed"], [check for check in result["checks"] if not check["passed"]]
    assert result["jobs_present"] == 64
    assert len(result["metrics_recomputed"]) == 64
    assert all(row["clade_f1"] is None for row in result["metrics_recomputed"])


@pytest.mark.parametrize("corruption, expected_check", [
    ("nonempty_matrix", "unavailable model uses empty R/X placeholders"),
    ("zero_metric", "unavailable model metrics remain null"),
])
def test_failure_with_files_rejects_false_matrix_or_metric(tmp_path, corruption, expected_check):
    output, dataset = _failed_directory_fixture(tmp_path)
    job = output / "jobs" / "paper15__constant__WZZT"
    if corruption == "nonempty_matrix":
        with np.load(job / "model.npz") as saved:
            arrays = {name: saved[name].copy() for name in saved.files}
        arrays["R"] = np.eye(4)
        np.savez_compressed(job / "model.npz", **arrays)
    else:
        path = job / "result.json"
        result = json.loads(path.read_text(encoding="utf-8"))
        result["metrics"]["clade_f1"] = 0.
        path.write_text(json.dumps(result), encoding="utf-8")
    result = audit_benchmark(output, dataset)
    assert any(expected_check in check["check"] for check in result["checks"] if not check["passed"])


def test_copied_estimator_input_hash_mismatch_is_rejected(tmp_path):
    output, dataset = _failed_directory_fixture(tmp_path)
    path = output / "inputs" / "paper15__constant" / "source" / "amplitudes.csv"
    path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    result = audit_benchmark(output, dataset)
    assert any("copied estimator input sha256 source/amplitudes.csv" in check["check"]
               for check in result["checks"] if not check["passed"])
