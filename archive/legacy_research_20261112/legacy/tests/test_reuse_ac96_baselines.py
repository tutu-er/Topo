"""Hand-built saved fits: reuse must neither fit nor mutate its producers."""
import importlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from research_experiments.rnj import reuse_ac96_baselines as reuse


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2)+"\n", encoding="utf-8")


def inventory(folder):
    return {p.relative_to(folder).as_posix(): reuse._sha(p) for p in folder.rglob("*") if p.is_file()}


@pytest.fixture
def archived(tmp_path):
    original, dataset, output = tmp_path/"original", tmp_path/"dataset", tmp_path/"new"
    original.mkdir()
    labels, n = [11, 12, 13, 14], 4
    z = np.array([1., 1., 1., 0.])
    R, X = .03*np.eye(n)+.08*np.outer(z, z), .02*np.eye(n)+.05*np.outer(z, z)
    rng = np.random.default_rng(6124)
    for case in reuse.CASES:
        net = dataset/case/"network"
        net.mkdir(parents=True)
        write(net/"metadata.json", {"root_bus": 1, "base_kv": 1., "base_mva": 1.})
        pd.DataFrame({"bus_id": [1, 2, *labels]}).to_csv(net/"buses.csv", index=False)
        pd.DataFrame([(1, 2, .04, .025, True), (2, 11, .015, .01, True), (2, 12, .015, .01, True),
                      (2, 13, .015, .01, True), (1, 14, .015, .01, True)],
                     columns=["from_bus", "to_bus", "r_ohm", "x_ohm", "is_true_closed"]).to_csv(net/"branches.csv", index=False)
        P, Q = rng.uniform(.01, .05, (96, n)), rng.uniform(.005, .02, (96, n))
        index = pd.Index(range(96), name="t")
        for variant in reuse.VARIANTS:
            condition = dataset/case/variant
            observed, source = condition/"observed", condition/"source"
            observed.mkdir(parents=True)
            source.mkdir()
            if variant == "no_source":
                U = np.zeros((96, 2))
            elif variant == "constant":
                U = np.tile([.01, .004], (96, 1))
            elif variant == "intermittent":
                U = np.tile([.01, .004], (96, 1))*(np.arange(96) % 4 != 0)[:, None]
            else:
                U = np.column_stack((.01*(1+.3*np.sin(np.arange(96))), .004*(1+.2*np.cos(np.arange(96)))))
            G = np.outer([1., .4], [.02, .02, .02, 0.]) if U.any() else np.zeros((2, n))
            Y = P@R.T+Q@X.T+U@G
            for name, values in (("P_terminal", P), ("Q_terminal", Q), ("drop_target", Y), ("V_terminal", np.sqrt(1-Y))):
                pd.DataFrame(values, index=index, columns=labels).to_csv(observed/(name+".csv"), float_format="%.17g")
            for name, values in (("root_voltage", np.ones(96)), ("P0_measured", P.sum(1)), ("Q0_measured", Q.sum(1))):
                pd.Series(values, index=index, name=name).to_csv(observed/(name+".csv"), float_format="%.17g")
            write(observed/"metadata.json", {"root_observation": "noisy", "scenario_settings": {}})
            pd.DataFrame(U, index=index, columns=["P_pu", "Q_pu"]).to_csv(source/"amplitudes.csv", float_format="%.17g")
            write(source/"metadata.json", {"information": "known schedules; unknown location"})
    write(dataset/"manifest.json", {"settings": {"v_noise_rel": .0002, "root_meter_noise_rel": .0002}, "file_sha256": inventory(dataset)})
    sources = {
        "research_experiments/rnj/source_ac96_trees.py": "# a frozen producer fixture, never imported\n",
        "rnj_wzzt_core/rnj_wzzt/estimation/multiscenario.py": "def fit_projected_sensitivity(scenarios, constraint_refine_iterations=500):\n    pass\n",
    }
    for relative, content in sources.items():
        target = original/"source_snapshot"/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    source_hashes = {relative: reuse._sha(original/"source_snapshot"/relative) for relative in sources}
    protocol = {"cases": list(reuse.CASES), "variants": list(reuse.VARIANTS), "samples_per_fit": 96, "validation_samples": 0,
                "rx75_weight": .75, "tree_tolerance_factor": .16, "dataset_manifest_sha256": reuse._sha(dataset/"manifest.json"),
                "source_sha256": source_hashes}
    write(original/"protocol.json", protocol)
    write(original/"source_snapshot_manifest.json", [{"path": p, "sha256": h} for p, h in source_hashes.items()])
    write(original/"environment.json", {"git_head": "fixture-producer", "packages": {"numpy": "archived"}})
    signature = reuse._sha(original/"protocol.json")
    for job in reuse._jobs():
        folder = original/"jobs"/job["job_id"]
        copied = original/"inputs"/(job["case"]+"__"+job["variant"])
        condition = dataset/job["case"]/job["variant"]
        if not copied.exists():
            for sub in ("observed", "source"):
                shutil.copytree(condition/sub, copied/sub)
        hashes = inventory(copied)
        P = pd.read_csv(condition/"observed/P_terminal.csv", index_col="t", float_precision="round_trip").to_numpy()
        Q = pd.read_csv(condition/"observed/Q_terminal.csv", index_col="t", float_precision="round_trip").to_numpy()
        Y = pd.read_csv(condition/"observed/drop_target.csv", index_col="t", float_precision="round_trip").to_numpy()
        U = pd.read_csv(condition/"source/amplitudes.csv", index_col="t", float_precision="round_trip").to_numpy()
        G = np.outer([1., .4], [.02, .02, .02, 0.]) if U.any() else np.zeros((2, n))
        clades = [[11, 12], [11, 12, 13]]
        diag = {"source_location_used": False, "source_rank": int(np.linalg.matrix_rank(U)), "success": True,
                "iterations": 2, "intercept_added": False, "source_path_constraints": False, "collapse_or_tolerance_factor": .16}
        model = {"terminal_labels": labels, "has_model": True, "status": "complete", "matrix_origin": "source_adjusted_ordered_regression",
                 "source_h": None, "support_indices": None, "r_values": None, "x_values": None,
                 "source_g_r": G[0].tolist(), "source_g_x": G[1].tolist(), "regression_diagnostics": diag,
                 "clades": clades, "tree": {"root": 1, "terminals": labels, "edges": [[1, -1, 1.], [-1, -2, .001],
                     [-2, 11, .1], [-2, 12, .1], [-1, 13, .1], [1, 14, .1]]}, "timing": {"sensitivity_seconds": .125},
                 "train_mae": float(np.mean(abs(Y-(P@R.T+Q@X.T+U@G))))}
        metrics = {"clade_f1": 2/3, "precision": .5, "recall": 1., "r_relative_error": 0., "x_relative_error": 0.}
        result = {**job, "status": "complete", "has_model": True, "search_complete": True, "metrics": metrics,
                  "matrix_origin": model["matrix_origin"], "elapsed_seconds": .3, "compute_seconds": .25, "protocol_sha256": signature}
        write(folder/"model.json", model)
        np.savez_compressed(folder/"model.npz", R=R, X=X, P=P, Q=Q, Y=Y, P_source=U[:, 0], Q_source=U[:, 1])
        write(folder/"result.json", result)
        write(folder/"inputs_hash.json", {"file_sha256": hashes})
        write(folder/"payload.json", {"job": job, "config": protocol, "protocol_sha256": signature, "input_dir": str(copied)})
    return original, dataset, output


def test_complete_32_reuse_exact_npz_original_runtime_and_resolution(archived):
    original, dataset, output = archived
    before = inventory(original)
    evidence = reuse.reuse_baselines(original, dataset, output)
    assert len(evidence["baseline_jobs"]) == evidence["baseline_denominator"] == 32
    assert evidence["original_version"]["historical_QP_max_iterations"] == 500
    assert evidence["original_version"]["historical_QP_iteration_counts"] == [2]
    assert evidence["estimation_performed"] is False
    assert evidence["baseline_provenance_sha256"] == reuse._sha(output/evidence["baseline_provenance_path"])
    report = reuse._read(output/"baseline_reuse/reuse_audit.json")
    assert report["failures"] == [] and report["baseline_denominator"] == 32
    for job in evidence["baseline_jobs"]:
        folder = output/job["directory"]
        old = original/"jobs"/job["job_id"]
        assert reuse._sha(folder/"model.npz") == reuse._sha(old/"model.npz")
        result, model = reuse._read(folder/"result.json"), reuse._read(folder/"model.json")
        assert result["original_metrics"]["clade_f1"] == 2/3
        assert result["metrics"]["family_clade_f1"] == result["metrics"]["active_clade_f1"] == 2/3
        assert result["metrics"]["resolution_clade_f1"] == result["metrics"]["clade_f1"] == 1.
        assert result["compute_seconds"] == .25 and result["elapsed_seconds"] == .3
        assert result["postprocess_seconds"] >= 0
        assert model["source_h"] is None and model["support_indices"] is None
        assert model["training_rows"] == 96 and model["validation_supplied"] is False
        assert model["resolution_clades"] == [[11, 12, 13]]
        assert (folder/"reuse_provenance.json").is_file()
    assert before == inventory(original)


def test_idempotent_reuse_keeps_derived_bytes_and_times(archived):
    original, dataset, output = archived
    first = reuse.reuse_baselines(original, dataset, output)
    frozen = inventory(output)
    second = reuse.reuse_baselines(original, dataset, output)
    assert first == second and frozen == inventory(output)


@pytest.mark.parametrize("target", ["original_input", "copied_input", "snapshot", "payload"])
def test_rejects_changed_producer_or_inputs_without_estimation(archived, target):
    original, dataset, output = archived
    key = reuse._jobs()[0]["job_id"]
    if target == "original_input":
        path = dataset/reuse.CASES[0]/reuse.VARIANTS[0]/"observed/P_terminal.csv"
    elif target == "copied_input":
        path = original/"inputs"/(reuse.CASES[0]+"__"+reuse.VARIANTS[0])/"observed/P_terminal.csv"
    elif target == "snapshot":
        path = original/"source_snapshot/research_experiments/rnj/source_ac96_trees.py"
    else:
        path = original/"jobs"/key/"payload.json"
        value = reuse._read(path)
        value["protocol_sha256"] = "changed"
        write(path, value)
    if target != "payload":
        path.write_text(path.read_text()+"\n", encoding="utf-8")
    with pytest.raises(reuse.ReuseValidationError, match="audit failed"):
        reuse.reuse_baselines(original, dataset, output)
    assert not (output/"baseline_reuse/jobs").exists()
    assert reuse._read(output/"baseline_reuse/reuse_audit.json")["failures"]


def test_rejects_changed_derived_npz_on_resume(archived):
    original, dataset, output = archived
    evidence = reuse.reuse_baselines(original, dataset, output)
    path = output/evidence["baseline_jobs"][0]["directory"]/"model.npz"
    path.write_bytes(path.read_bytes()+b"changed")
    with pytest.raises(reuse.ReuseValidationError, match="immutable reused artifact"):
        reuse.reuse_baselines(original, dataset, output)


def test_protects_original_directory(archived):
    original, dataset, output = archived
    before = inventory(original)
    with pytest.raises(ValueError, match="separate"):
        reuse.reuse_baselines(original, dataset, original)
    assert inventory(original) == before


def test_import_does_not_activate_estimator_runtime():
    result = subprocess.run([sys.executable, "-B", "-c", "import sys; import research_experiments.rnj.reuse_ac96_baselines; assert not any(n=='rnj_wzzt' or n.startswith('rnj_wzzt.') for n in sys.modules)"],
                            cwd=reuse.ROOT, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
