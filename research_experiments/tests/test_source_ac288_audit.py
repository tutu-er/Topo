"""Saved-evidence corruption tests; no estimator or solver is executed."""

import copy
import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from research_experiments.rnj.audit_source_ac288 import (
    Audit, atom_reconstruction, audit_baseline, audit_path, audit_point,
    audit_raw_primal, audit_screen, audit_stopping, canonical_partition,
    expected_partition, optimality_certificate, prediction_and_loss,
)
from research_experiments.rnj import audit_source_ac288 as module


LABELS = [301, 302, 303, 304, 305, 306]
INITIAL = [(i,) for i in range(6)] + [(0, 1, 2)]
GROUPS = [(0, 1, 2), (3,), (4,), (5,)]


def fixture(source=True):
    rng = np.random.default_rng(17)
    p = rng.uniform(.01, .1, (288, 6))
    q = rng.uniform(.005, .05, (288, 6))
    up = np.tile(np.linspace(.01, .04, 96), 3) if source else np.zeros(288)
    uq = .4 * up
    point = {
        "support_indices": [list(s) for s in [*INITIAL, (0, 1, 2, 3)]],
        "support_labels": [[LABELS[i] for i in s] for s in [*INITIAL, (0, 1, 2, 3)]],
        "r_values": [.1] * 8, "x_values": [.05] * 8,
        "validation_mae": None, "validation_se": None,
        "source_h": [0] * 6 + [1, 1] if source else None,
    }
    value = atom_reconstruction(point, LABELS)
    for name in ("r", "x"):
        response = ((value[name] * np.asarray(point["source_h"])) @ value["z"]
                    if source else np.zeros(6))
        point[f"source_response_{name}"] = response.tolist() if source else None
        point[f"{name}_matrix"] = value[name.upper()].tolist()
    arrays = {"P": p, "Q": q, "P_source": up, "Q_source": uq,
              "R": value["R"], "X": value["X"]}
    gr = np.asarray(point["source_response_r"]) if source else np.zeros(6)
    gx = np.asarray(point["source_response_x"]) if source else np.zeros(6)
    arrays["Y"] = p @ value["R"].T + q @ value["X"].T + up[:, None] * gr + uq[:, None] * gx
    point["train_mae"] = 0.
    point["solver"] = {"status": 0, "success": True, "binary_variable_count": 8 if source else 0,
                       "objective": 0., "dual_bound": 0., "mip_gap": 0.}
    return point, arrays


def failed(audit):
    return [c["check"] for c in audit.checks if not c["passed"]]


def test_partition_is_complete_in_original_terminal_positions():
    assert expected_partition([[301, 302, 303]], LABELS) == canonical_partition(GROUPS, 6)
    with pytest.raises(ValueError, match="partition every"):
        canonical_partition([(0, 1, 2), (3,), (4,)], 6)
    with pytest.raises(ValueError, match="partition every"):
        canonical_partition([(0, 1, 2), (2, 3), (4,), (5,)], 6)
    with pytest.raises(ValueError, match="size >= 3"):
        expected_partition([[301, 302]], LABELS)


@pytest.mark.parametrize("source", [False, True])
def test_original_dimension_atoms_and_full_source_path_pass(source):
    point, arrays = fixture(source)
    audit = Audit()
    value = audit_point(point, arrays, LABELS, INITIAL, GROUPS, 2., 2., audit, "model")
    assert value["R"].shape == (6, 6)
    assert audit.result()["passed"], failed(audit)


def test_atomic_group_refinement_is_rejected_even_though_laminar():
    point, arrays = fixture(False)
    point["support_indices"][-1] = [0, 1]
    point["support_labels"][-1] = LABELS[:2]
    audit = Audit()
    audit_point(point, arrays, LABELS, INITIAL, GROUPS, 2., 2., audit, "model")
    assert any("union of complete terminal groups" in c for c in failed(audit))


@pytest.mark.parametrize("corruption, expected", [
    ("ancestor", "ancestor complete"), ("shape", "covers every original atom"),
    ("response", "independently rebuilt"), ("bound", "coefficient bound"),
    ("objective", "complete 288-row"), ("validation", "validation MAE is absent"),
    ("certificate", "strict LP/MIP certificate"),
])
def test_source_and_certificate_corruption_rejected(corruption, expected):
    point, arrays = fixture()
    if corruption == "ancestor":
        point["source_h"][-1] = 0
    elif corruption == "shape":
        point["source_h"] = [point["source_h"]] * 3
    elif corruption == "response":
        point["source_response_r"][0] += .01
    elif corruption == "bound":
        point["r_values"][0] = 3.
    elif corruption == "objective":
        point["train_mae"] = .01
    elif corruption == "validation":
        point["validation_mae"] = .01
    elif corruption == "certificate":
        point["solver"].update(dual_bound=None, mip_gap=None)
    audit = Audit()
    try:
        audit_point(point, arrays, LABELS, INITIAL, GROUPS, 2., 2., audit, "model")
    except ValueError:
        pass
    assert any(expected in c for c in failed(audit)), failed(audit)


def test_no_validation_selects_last_certified_point():
    point, arrays = fixture(False)
    initial = copy.deepcopy(point)
    for field in ("support_indices", "support_labels", "r_values", "x_values"):
        initial[field] = initial[field][:-1]
    value = atom_reconstruction(initial, LABELS)
    for name in ("r", "x"):
        initial[f"{name}_matrix"] = value[name.upper()].tolist()
    _, initial["train_mae"] = prediction_and_loss(arrays, value["R"], value["X"], np.zeros(6), np.zeros(6))
    initial["solver"]["objective"] = initial["solver"]["dual_bound"] = initial["train_mae"]
    initial["accepted_gain"] = None
    point["accepted_gain"] = initial["train_mae"]
    audit = Audit()
    result = audit_path([initial, point], arrays, LABELS, INITIAL, GROUPS, 2., 2., audit, "path")
    assert result["selected_path_index"] == 1
    assert audit.result()["passed"], failed(audit)


def test_collinear_source_fwl_nuisance_recomputed_over_all_288_rows():
    point, arrays = fixture()
    r, x = arrays["R"], arrays["X"]
    source = np.column_stack((arrays["P_source"], arrays["Q_source"]))
    response = np.linalg.lstsq(source, arrays["Y"] - arrays["P"] @ r.T - arrays["Q"] @ x.T, rcond=None)[0]
    model = {"matrix_origin": "source_adjusted_ordered_regression_before_tree_inference", "source_h": None,
             "source_g_r": response[0].tolist(), "source_g_x": response[1].tolist(), "train_mae": 0.,
             "tree": {"root": 1, "edges": [[1, label] for label in LABELS]}}
    audit = Audit()
    audit_baseline(model, arrays, LABELS, audit, "baseline")
    assert audit.result()["passed"], failed(audit)
    model["source_g_r"][0] += .1
    audit = Audit()
    audit_baseline(model, arrays, LABELS, audit, "baseline")
    assert any("FWL Source nuisance independently refit" in c for c in failed(audit))


def test_status_zero_without_gap_is_not_a_source_mip_certificate():
    event = {"status": 0, "success": True, "binary_variable_count": 7}
    assert not optimality_certificate(event)
    assert optimality_certificate({**event, "objective": .1, "dual_bound": .1})
    assert not optimality_certificate({**event, "objective": .1, "dual_bound": .09})


def test_raw_primal_checks_original_objective_constraints_bounds_and_integrality():
    saved = {"primal": np.array([1., .5]), "objective_coefficients": np.array([.1, .2]),
             "lower_bounds": np.zeros(2), "upper_bounds": np.ones(2), "integrality": np.array([1, 0]),
             "constraint_row_index": np.array([0, 0]), "constraint_col_index": np.array([0, 1]),
             "constraint_data": np.ones(2), "constraint_lower": np.array([1.5]), "constraint_upper": np.array([1.5])}
    event = {"variable_count": 2, "objective": .2, "dual_bound": .2}
    audit = Audit()
    audit_raw_primal(event, saved, audit, "solve")
    assert audit.result()["passed"], failed(audit)
    bad = dict(saved, primal=np.array([1.1, .5]))
    audit = Audit()
    audit_raw_primal(event, bad, audit, "solve")
    errors = failed(audit)
    assert any("variable bounds" in c for c in errors)
    assert any("integrality" in c for c in errors)
    assert any("linear constraints" in c for c in errors)
    assert any("objective independently" in c for c in errors)


def screen_fixture():
    peers = [[301, 302, 303], [304, 305, 306]]
    screen = {"training_sample_count": 288, "validation_used": False,
              "parameters": {**module.SELECTION_PARAMETERS, "seed": 20260903},
              "full_clades": peers, "confidence_records": [
                  {"clade": peers[0], "frequency": .8}, {"clade": peers[1], "frequency": .9}],
              "selected_support_labels": [peers[0]], "selected_support_indices": [[0, 1, 2]],
              "diagnostics": {"full_peers": peers, "valid_bootstrap_replicates": 90,
                  "unidentifiable_bootstrap_replicates": 10, "source_adjusted_identifiable": True,
                  "subclade_frequency_records": [{"clade": peers[0], "frequency": .1},
                                                 {"clade": peers[1], "frequency": 1 / 3}],
                  "full_fine_subclades": [], "eligible": [peers[0]]}}
    model = {"requested_terminal_groups": [[0, 1, 2]], "terminal_groups": GROUPS, "initial_supports": INITIAL}
    return screen, model


def test_final_atomic_screen_uses_peer_and_fine_veto_with_correct_denominators():
    screen, model = screen_fixture()
    audit = Audit()
    assert audit_screen(screen, model, LABELS, audit, "screen") == [[301, 302, 303]]
    assert audit.result()["passed"], failed(audit)
    screen["diagnostics"]["full_fine_subclades"] = [[301, 302, 303]]
    audit = Audit()
    audit_screen(screen, model, LABELS, audit, "screen")
    assert any("refinement veto" in c for c in failed(audit))
    screen, model = screen_fixture()
    screen["diagnostics"]["subclade_frequency_records"][0]["frequency"] = .11
    audit = Audit()
    audit_screen(screen, model, LABELS, audit, "screen")
    assert any("only identifiable replicates" in c for c in failed(audit))


def test_iteration_limit_is_not_misread_as_current_family_size_or_no_gain():
    point, _ = fixture(False)
    point["iteration"] = 35
    model = {"stop_reason": "max_atoms", "path": [point], "terminal_labels": LABELS,
             "core_parameters": {"max_atoms": 35}}
    audit = Audit()
    audit_stopping(model, audit, "stop")
    assert len(point["support_indices"]) < 35
    assert audit.result()["passed"], failed(audit)


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def failed_directory(tmp_path, monkeypatch):
    directory, dataset = tmp_path / "run", tmp_path / "data"
    jobs = [{"case": module.CASE, "variant": variant, "method": method,
             "job_id": f"{module.CASE}__{variant}__{method}",
             "per_solve_seconds": 6000. if "WZZT" in method else None,
             "method_wall_seconds": 30000. if "WZZT" in method else 300.}
            for variant in module.VARIANTS for method in module.METHODS]
    protocol = {"schema_version": 3, "case": module.CASE, "variants": list(module.VARIANTS),
                "methods": list(module.METHODS), "expected_jobs": 16, "total_samples": 288,
                "training_samples": 288, "validation_samples": 0, "validation_supplied": False,
                "independent_test_samples": 0, "scenario_count": 3, "samples_per_scenario": 96,
                "selection": "last_certified_path_point_without_validation", "observed_root_label": 1,
                "rx75_weight": .75, "tree_tolerance_factor": .16, "minimum_atomic_group_size": 3,
                "rnj_atomic_block_rule": "complete_direct_terminal_children", "solver_threads": 1,
                "blas_threads": 1, "truth_used_for_selection": False, "core_parameters": module.CORE_PARAMETERS,
                "screen_parameters": {**module.SELECTION_PARAMETERS, "seed": 20260903}, "jobs": jobs,
                "runtime_directory": str(tmp_path / "runtime")}
    _write_json(directory / "protocol.json", protocol)
    key = "pengwah18__constant__WZZT"
    job = directory / "jobs" / key
    point, arrays = fixture()
    inventory = {}
    for i in range(3):
        for name in module.INPUT_NAMES:
            relative = f"scenario_{i}/{name}"
            target = dataset / module.CASE / "constant" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if name in {"observed/P_terminal.csv", "observed/Q_terminal.csv", "observed/drop_target.csv"}:
                field = {"P_terminal": "P", "Q_terminal": "Q", "drop_target": "Y"}[target.stem]
                pd.DataFrame(arrays[field][i*96:(i+1)*96], columns=LABELS).rename_axis("t").to_csv(target)
            elif name == "source/amplitudes.csv":
                pd.DataFrame({"P_pu": arrays["P_source"][i*96:(i+1)*96],
                              "Q_pu": arrays["Q_source"][i*96:(i+1)*96]}).rename_axis("t").to_csv(target)
            else:
                target.write_text("{}" if target.suffix == ".json" else "t,v\n0,0\n", encoding="utf-8")
            duplicate = directory / "inputs/constant" / relative
            duplicate.parent.mkdir(parents=True, exist_ok=True)
            duplicate.write_bytes(target.read_bytes())
            inventory[relative] = hashlib.sha256(target.read_bytes()).hexdigest()
    _write_json(job / "inputs_hash.json", {"file_sha256": inventory})
    _write_json(job / "data_use.json", {"total_samples": 288, "training_samples": 288, "validation_samples": 0,
                "validation_supplied": False, "terminal_labels": LABELS,
                "selection": "last_certified_path_point_without_validation",
                "scenarios": [{"row_range": [96*i, 96*(i+1)]} for i in range(3)]})
    _write_json(job / "runtime_import.json", {"core_file": str(tmp_path / "runtime/rnj_wzzt/estimation/laminar_l1_milp.py")})
    _write_json(job / "model.json", {"has_model": False, "terminal_labels": LABELS})
    _write_json(job / "result.json", {"status": "failed", "has_model": False, "search_complete": False,
                "protocol_sha256": module.file_hash(directory / "protocol.json"), "elapsed_seconds": .2, "metrics": {}})
    arrays.update(R=np.empty((0, 0)), X=np.empty((0, 0)), row_scenario=np.repeat(np.arange(3), 96),
                  row_position=np.tile(np.arange(96), 3))
    np.savez_compressed(job / "model.npz", **arrays)
    monkeypatch.setattr(module, "audit_fingerprints", lambda *_: None)
    return directory, dataset, key


def test_failure_with_input_files_preserves_no_model_without_zero_metrics(tmp_path, monkeypatch):
    directory, dataset, key = failed_directory(tmp_path, monkeypatch)
    result = module.audit_benchmark(directory, dataset, only_jobs=[key])
    assert result["jobs_present"] == 1
    assert result["passed"], [c for c in result["checks"] if not c["passed"]]
    assert result["metrics_recomputed"][0]["metrics_recomputed"] is None
    path = directory / "jobs" / key / "result.json"
    corrupt = module.read_json(path)
    corrupt["metrics"] = {"family_clade_f1": 0.}
    _write_json(path, corrupt)
    result = module.audit_benchmark(directory, dataset, only_jobs=[key])
    assert any("unavailable model keeps metrics" in c["check"] for c in result["checks"] if not c["passed"])


def test_directory_audit_keeps_all_sixteen_denominators(tmp_path, monkeypatch):
    directory, dataset, _ = failed_directory(tmp_path, monkeypatch)
    result = module.audit_benchmark(directory, dataset)
    assert result["jobs_expected"] == 16 and len(result["metrics_recomputed"]) == 16
    assert result["jobs_present"] == 1
    assert sum(row["status"] == "missing" for row in result["metrics_recomputed"]) == 15
    assert all("outcome or explicit failure retained" in c["check"] for c in result["checks"] if not c["passed"])


def test_source_row_permutation_is_detected_before_any_truth_access(tmp_path, monkeypatch):
    directory, dataset, key = failed_directory(tmp_path, monkeypatch)
    path = directory / "jobs" / key / "model.npz"
    with np.load(path, allow_pickle=False) as saved:
        arrays = {name: saved[name] for name in saved.files}
    arrays["P_source"] = np.roll(arrays["P_source"], 1)
    np.savez_compressed(path, **arrays)
    result = module.audit_benchmark(directory, dataset, only_jobs=[key])
    assert any("P_source uses all 3x96 rows" in c["check"] for c in result["checks"] if not c["passed"])
