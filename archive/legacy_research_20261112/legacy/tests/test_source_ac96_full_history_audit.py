"""Small arithmetic fixtures for the independent full-history auditor."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from research_experiments.rnj.audit_source_ac96_full_history import (
    ARRAY_NAMES, CORE_PARAMETERS, JOBS, _job_id, atom_matrices, audit_full_history,
    audit_path, audit_protocol, audit_restart_trace, one_se_selection, tree_clades,
    validation_statistics,
)
from research_experiments.rnj.audit_source_ac96_benchmark import Audit, clade_metrics, physical_truth, relative_error


def _diagnostics(objective, *, binary=0):
    return {"status": 0, "success": True, "objective": objective, "dual_bound": objective,
            "mip_gap": None if binary == 0 else 0., "binary_variable_count": binary,
            "variable_count": 2, "runtime_seconds": .001, "solver": "fixture", "raw_status": 2}


def _arrays_and_points():
    rng = np.random.default_rng(53)
    p, q = rng.uniform(.01, .04, (96, 3)), rng.uniform(.004, .015, (96, 3))
    labels, initial = [101, 102, 103], [[0], [1], [2]]
    definitions = [(initial, [.2, .3, .4], [.1, .1, .1]),
                   (initial + [[0, 1]], [.2, .3, .4, .01], [.1, .1, .1, .005])]
    target_point = {"support_indices": definitions[1][0], "r_values": definitions[1][1],
                    "x_values": definitions[1][2]}
    true_r, true_x, _, _ = atom_matrices(target_point, labels)
    y = p @ true_r + q @ true_x
    arrays = {"R": true_r, "X": true_x, "P_all": p, "Q_all": q, "Y_all": y,
              "P_source_all": np.zeros(96), "Q_source_all": np.zeros(96),
              "train_positions": np.arange(64), "validation_positions": np.arange(64, 96)}
    for suffix, rows in (("train", slice(0, 64)), ("validation", slice(64, 96))):
        arrays.update({f"{name}_{suffix}": value[rows] for name, value in (("P", p), ("Q", q), ("Y", y))})
    points, previous = [], None
    for i, (supports, rw, xw) in enumerate(definitions):
        point = {"support_indices": supports, "support_labels": [[labels[j] for j in s] for s in supports],
                 "r_values": rw, "x_values": xw, "iteration": 3 + i}
        r, x, _, _ = atom_matrices(point, labels)
        tr = validation_statistics(arrays["P_train"], arrays["Q_train"], arrays["Y_train"], r, x)
        va = validation_statistics(arrays["P_validation"], arrays["Q_validation"], arrays["Y_validation"], r, x)
        point.update(r_matrix=r.tolist(), x_matrix=x.tolist(), train_mae=tr["mae"],
                     validation_mae=va["mae"], validation_se=va["se"],
                     accepted_gain=None if previous is None else previous - tr["mae"],
                     solver=_diagnostics(tr["mae"], binary=0 if i == 0 else 2))
        points.append(point)
        previous = tr["mae"]
    assert set(arrays) == ARRAY_NAMES
    return labels, initial, arrays, points


def _protocol():
    jobs = []
    for method, budget in JOBS:
        multiplier = {"t1": 1, "t100": 100}.get(budget)
        jobs.append({"case": "pengwah18", "variant": "no_source", "method": method,
                     "job_id": _job_id(method, budget), "budget_multiplier": multiplier,
                     "per_solve_seconds": None if multiplier is None else 60. * multiplier,
                     "method_wall_seconds": 300. if multiplier is None else 300. * multiplier})
    return {"case": "pengwah18", "variant": "no_source", "jobs": jobs, "expected_jobs": 6,
            "core_parameters": CORE_PARAMETERS, "total_samples": 96, "training_samples": 64,
            "validation_samples": 32, "training_positions": list(range(64)),
            "validation_positions": list(range(64, 96)), "observed_root_label": 1,
            "rx75_weight": .75, "tree_tolerance_factor": .16,
            "bootstrap_replicates": 100, "bootstrap_block_length": 4, "bootstrap_threshold": .75,
            "maximum_frozen_blocks": 2, "bootstrap_seed": 20260903,
            "bootstrap_selector": "rnj_wzzt.pipeline._select_boundary_blocks",
            "validation_used_for_bootstrap": False, "active_atom_tolerance": 1e-8,
            "solver_threads": 1, "blas_threads": 1}


def test_four_chronological_validation_blocks_have_independent_ddof1_se():
    y = np.repeat(np.arange(1., 5.), 8)[:, None]
    stats = validation_statistics(np.zeros_like(y), np.zeros_like(y), y, [[0]], [[0]])
    assert stats["mae"] == 2.5
    assert stats["block_maes"] == [1., 2., 3., 4.]
    assert stats["se"] == pytest.approx(np.sqrt(5 / 3) / 2)


def test_one_se_uses_best_points_own_se_then_minimum_support_count():
    path = [dict(validation_mae=.100, validation_se=.001, support_indices=[[0], [1], [2]]),
            dict(validation_mae=.105, validation_se=100., support_indices=[[0]]),
            dict(validation_mae=.090, validation_se=.020, support_indices=[[0], [1]])]
    chosen = one_se_selection(path)
    assert chosen["best_validation_index"] == 2
    assert chosen["threshold"] == pytest.approx(.110)
    assert chosen["selected_path_index"] == 1
    assert chosen["selected_path_index"] != len(path) - 1


def test_one_se_ties_use_first_index_and_invalid_se_is_rejected():
    path = [dict(validation_mae=.1, validation_se=.02, support_indices=[[0]]),
            dict(validation_mae=.09, validation_se=.02, support_indices=[[1]])]
    assert one_se_selection(path)["selected_path_index"] == 0
    path[0]["validation_se"] = -1.
    with pytest.raises(ValueError, match="nonnegative"):
        one_se_selection(path)


def test_full_path_losses_se_and_gap_are_audited_without_fitting():
    labels, initial, arrays, points = _arrays_and_points()
    audit = Audit()
    choice = audit_path(points, arrays, labels, initial, 2., 2., audit, "fixture")
    assert audit.result()["passed"]
    assert choice["selected_path_index"] == 1
    bad = deepcopy(points)
    bad[1]["validation_se"] = .1
    tampered = Audit()
    audit_path(bad, arrays, labels, initial, 2., 2., tampered, "fixture")
    assert any(not c["passed"] and "validation SE" in c["check"] for c in tampered.checks)
    bad = deepcopy(points)
    bad[1]["solver"].update(dual_bound=None, mip_gap=None)
    tampered = Audit()
    audit_path(bad, arrays, labels, initial, 2., 2., tampered, "fixture")
    assert any(not c["passed"] and "optimality" in c["check"] for c in tampered.checks)


def test_family_and_active_clades_keep_zero_tolerance_distinct():
    point = {"support_indices": [[0], [1], [2], [0, 1]],
             "r_values": [.1, .1, .1, 5e-9], "x_values": [0., 0., 0., 0.]}
    _, _, family, active = atom_matrices(point, [101, 102, 103])
    assert 5e-9 > CORE_PARAMETERS["zero_tolerance"]
    assert family == {frozenset({101, 102})}
    assert active == set()


def test_bound_restart_retains_old_path_and_resets_to_initial_family():
    labels, initial, arrays, points = _arrays_and_points()
    attempt = {"support": [0, 1], "r_values": [.2, .3, .4, 2.],
               "x_values": [.1, .1, .1, .005], "diagnostics": _diagnostics(.001, binary=2)}
    base = dict(terminal_labels=labels, initial_supports=initial, path=[points[0]],
                selected_path_index=0, r_upper_bound=2., x_upper_bound=2., bound_expansions=0)
    snapshots = [dict(base, event="initialized", attempted_extensions=[]),
                 dict(base, event="bounds_expansion_requested", attempted_extensions=[attempt]),
                 dict(base, event="bounds_restarted", attempted_extensions=[attempt],
                      r_upper_bound=4., bound_expansions=1)]
    model = {**snapshots[-1]}
    audit = Audit()
    assert audit_restart_trace(snapshots, model, arrays, labels, audit, "fixture")["successful_bound_restarts"] == 1
    assert audit.result()["passed"]
    tampered = Audit()
    audit_restart_trace([snapshots[0], snapshots[-1]], model, arrays, labels, tampered, "fixture")
    assert any(not c["passed"] and "pre-expansion" in c["check"] for c in tampered.checks)


def test_protocol_has_exactly_100x_time_change_and_authoritative_seed():
    audit = Audit()
    protocol = _protocol()
    audit_protocol(protocol, audit)
    assert audit.result()["passed"]
    protocol = deepcopy(protocol)
    protocol["bootstrap_seed"] = 20261018
    protocol["jobs"][-1]["method_wall_seconds"] = 300.
    tampered = Audit()
    audit_protocol(protocol, tampered)
    assert sum(not c["passed"] for c in tampered.checks) == 2


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _saved_fixture(tmp_path):
    labels, initial, arrays, points = _arrays_and_points()
    run, dataset = tmp_path / "run", tmp_path / "dataset"
    protocol = _protocol()
    _write(dataset / "manifest.json", {"fixture": True})
    source_code = run / "source_snapshot" / "fixture_version.py"
    source_code.parent.mkdir(parents=True, exist_ok=True)
    source_code.write_text("# non-executable saved fixture version\n", encoding="utf-8")
    protocol["source_sha256"] = {"fixture_version.py": hashlib.sha256(source_code.read_bytes()).hexdigest()}
    protocol["dataset_manifest_sha256"] = hashlib.sha256((dataset / "manifest.json").read_bytes()).hexdigest()
    _write(run / "protocol.json", protocol)
    signature = hashlib.sha256((run / "protocol.json").read_bytes()).hexdigest()
    source = dataset / "pengwah18" / "no_source"
    copies = run / "inputs" / "pengwah18__no_source"
    for name, filename in (("P", "P_terminal"), ("Q", "Q_terminal"), ("Y", "drop_target")):
        frame = pd.DataFrame(arrays[f"{name}_all"], columns=labels)
        frame.index.name = "t"
        for base in (source, copies):
            path = base / "observed" / f"{filename}.csv"
            path.parent.mkdir(parents=True, exist_ok=True)
            frame.to_csv(path, float_format="%.17g")
    for base in (source, copies):
        path = base / "source" / "amplitudes.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"P_pu": np.zeros(96), "Q_pu": np.zeros(96)}).rename_axis("t").to_csv(path)
    network = dataset / "pengwah18" / "network"
    _write(network / "metadata.json", {"root_bus": 1, "base_kv": .4, "base_mva": .2})
    pd.DataFrame({"bus_id": [1, 2, *labels]}).to_csv(network / "buses.csv", index=False)
    branches = pd.DataFrame({"from_bus": [1, 2, 2, 1], "to_bus": [2, 101, 102, 103],
                             "r_ohm": np.asarray([.01, .2, .3, .4]) * .4,
                             "x_ohm": np.asarray([.005, .1, .1, .1]) * .4})
    branches.to_csv(network / "branches.csv", index=False)
    key = _job_id("WZZT", "t1")
    job = next(j for j in protocol["jobs"] if j["job_id"] == key)
    folder = run / "jobs" / key
    selected = points[1]
    r, x, family, active = atom_matrices(selected, labels)
    attempt = {"support": [0, 1], "r_values": selected["r_values"], "x_values": selected["x_values"],
               "objective": selected["train_mae"], "diagnostics": selected["solver"]}
    snapshots = [dict(event="initialized", terminal_labels=labels, initial_supports=initial,
                      path=[points[0]], attempted_extensions=[], selected_path_index=0,
                      r_upper_bound=2., x_upper_bound=2., bound_expansions=0),
                 dict(event="path_accepted", terminal_labels=labels, initial_supports=initial,
                      path=points, attempted_extensions=[attempt], selected_path_index=1,
                      r_upper_bound=2., x_upper_bound=2., bound_expansions=0)]
    events = []
    for i, point in enumerate(points):
        event = {**point["solver"], "sequence": i + 1, "requested_time_limit_seconds": 60.,
                 "time_limit_seconds": 60., "remaining_wall_before_solve": 300.,
                 "optimality_certified": True, "has_incumbent": True,
                 "primal_file": f"solver/{i + 1:06d}.npz"}
        events.append(event)
        _write(folder / "solver" / f"{i + 1:06d}.json", event)
        np.savez_compressed(folder / event["primal_file"], primal=np.zeros(2))
    model = {"terminal_labels": labels, **{name: selected[name] for name in
             ("support_indices", "support_labels", "r_values", "x_values", "train_mae", "validation_mae", "validation_se")},
             "path": points, "selected_path_index": 1, "attempted_extensions": [attempt],
             "initial_supports": initial, "r_upper_bound": 2., "x_upper_bound": 2., "bound_expansions": 0,
             "clades": [sorted(c) for c in family], "family_clades": [sorted(c) for c in family],
             "active_clades": [sorted(c) for c in active], "core_parameters": CORE_PARAMETERS,
             "solver_events": events, "budget": job}
    for i, snapshot in enumerate(snapshots):
        _write(folder / "progress" / f"{i + 1:06d}.json", snapshot)
    _write(folder / "model.json", model)
    np.savez_compressed(folder / "model.npz", **arrays)
    _write(folder / "split.json", {"total_samples": 96, "training_samples": 64, "validation_samples": 32,
           "training_positions": list(range(64)), "validation_positions": list(range(64, 96)),
           "all_time_labels": list(range(96)), "training_time_labels": list(range(64)),
           "validation_time_labels": list(range(64, 96)),
           "validation_used_for_rnj_selection": False, "new_observations_generated": False})
    truth = physical_truth(network, labels)
    metrics = {}
    for name, clades in (("family", family), ("active", active)):
        score = clade_metrics(clades, truth["clades"], labels)
        metrics.update({f"{name}_{key}": score[key] for key in ("clade_f1", "precision", "recall")})
    metrics.update(clade_f1=metrics["family_clade_f1"], r_relative_error=relative_error(r, truth["R"]),
                   x_relative_error=relative_error(x, truth["X"]))
    _write(folder / "result.json", {"status": "complete", "has_model": True, "elapsed_seconds": .1,
                                   "metrics": metrics, "protocol_sha256": signature})
    return run, dataset, folder


def test_saved_fixture_recomputes_arrays_path_selection_and_physical_metrics(tmp_path):
    run, dataset, folder = _saved_fixture(tmp_path)
    result = audit_full_history(run, dataset, only_jobs=[folder.name])
    assert result["passed"], [c for c in result["checks"] if not c["passed"]]
    assert result["metrics_recomputed"][0]["one_se_recomputed"]["selected_path_index"] == 1
    with np.load(folder / "model.npz") as saved:
        arrays = {k: saved[k].copy() for k in saved.files}
    arrays["P_train"][0] = arrays["P_validation"][0]
    np.savez_compressed(folder / "model.npz", **arrays)
    result = audit_full_history(run, dataset, only_jobs=[folder.name])
    assert not result["passed"]
    assert any(not c["passed"] and "first 64" in c["check"] for c in result["checks"])


def test_failed_missing_model_remains_in_denominator(tmp_path):
    run, dataset, folder = _saved_fixture(tmp_path)
    (folder / "model.json").unlink()
    (folder / "model.npz").unlink()
    _write(folder / "result.json", {"status": "failed", "has_model": False,
                                   "elapsed_seconds": 3., "metrics": {"clade_f1": None},
                                   "protocol_sha256": hashlib.sha256((run / "protocol.json").read_bytes()).hexdigest()})
    result = audit_full_history(run, dataset, only_jobs=[folder.name])
    assert result["passed"]
    assert result["jobs_expected"] == result["jobs_present"] == 1
    assert result["metrics_recomputed"][0]["status"] == "failed"


def test_tree_clades_come_from_edges_and_reject_cycles():
    tree = {"root": 1, "edges": [[1, -1, .1], [-1, 101, .2], [-1, 102, .2], [1, 103, .3]]}
    assert tree_clades(tree, [101, 102, 103]) == {frozenset({101, 102})}
    tree["edges"].append([101, 102, .2])
    with pytest.raises(ValueError, match="cycle"):
        tree_clades(tree, [101, 102, 103])
