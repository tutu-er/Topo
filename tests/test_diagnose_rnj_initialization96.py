"""Saved, hand-written stage evidence only; no estimator or solver calls."""
from copy import deepcopy
import csv
import json
from pathlib import Path

import numpy as np
import pytest

from research_experiments.rnj import diagnose_rnj_initialization96 as subject


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def network(directory):
    write_json(directory/"metadata.json", {"root_bus": 1, "base_kv": 1., "base_mva": 1.})
    (directory/"buses.csv").write_text("bus_id\n1\n2\n3\n11\n12\n13\n14\n15\n", encoding="utf-8")
    (directory/"branches.csv").write_text(
        "from_bus,to_bus,r_ohm,x_ohm,is_true_closed\n1,2,.2,.1,true\n1,3,.15,.075,true\n"
        "2,11,.05,.025,true\n2,12,.05,.025,true\n2,13,.05,.025,true\n3,14,.05,.025,true\n3,15,.05,.025,true\n", encoding="utf-8")


def example(directory, *, later_improves=False, source_active=False):
    network(directory)
    labels, a, b, fake = [11, 12, 13, 14, 15], [0, 1, 2], [3, 4], [0, 1]
    leaves = [[i] for i in range(5)]
    initial = leaves+[a, fake] if later_improves else leaves+[a, b]
    final = initial+[b] if later_improves else initial+[fake]
    def point(supports):
        r = [.1 if len(s)==1 else .4 if s==a else .3 if s==b else .12 for s in supports]
        x = [.05 if len(s)==1 else .2 if s==a else .15 if s==b else .06 for s in supports]
        h = [int(s==a) for s in supports] if source_active else None
        z = np.array([[float(i in s) for i in range(5)] for s in supports])
        return {"support_indices": supports, "support_labels": [[labels[i] for i in s] for s in supports],
                "r_values": r, "x_values": x, "source_h": h,
                "source_response_r": ((np.array(r)*h)@z).tolist() if h else None,
                "source_response_x": ((np.array(x)*h)@z).tolist() if h else None,
                "iteration": len(supports), "validation_mae": None, "validation_se": None}
    points = [point(initial), point(final)]
    rng = np.random.default_rng(317)
    arrays = {"P": .1+.02*rng.random((96, 5)), "Q": .04+.01*rng.random((96, 5)),
              "P_source": .02*np.sin(np.arange(96)/5) if source_active else np.zeros(96),
              "Q_source": .01*np.cos(np.arange(96)/7) if source_active else np.zeros(96)}
    last_atom = subject.evidence.atom_reconstruction(points[-1], labels)
    gr = np.asarray(points[-1]["source_response_r"]) if source_active else np.zeros(5)
    gx = np.asarray(points[-1]["source_response_x"]) if source_active else np.zeros(5)
    arrays["Y"] = arrays["P"]@last_atom["R"].T+arrays["Q"]@last_atom["X"].T+arrays["P_source"][:, None]*gr+arrays["Q_source"][:, None]*gx
    arrays.update(R=last_atom["R"], X=last_atom["X"])
    events = []
    for index, current in enumerate(points):
        atom = subject.evidence.atom_reconstruction(current, labels)
        gr = np.asarray(current["source_response_r"]) if source_active else np.zeros(5)
        gx = np.asarray(current["source_response_x"]) if source_active else np.zeros(5)
        loss = subject.evidence.prediction_and_loss(arrays, atom["R"], atom["X"], gr, gx)[1]
        diagnostic = {"status": 0, "success": True, "objective": loss, "dual_bound": loss,
                      "mip_gap": 0., "binary_variable_count": 2, "variable_count": 100+index,
                      "constraint_count": 200+index, "runtime_seconds": .1+index}
        current.update(train_mae=loss, solver=diagnostic, accepted_gain=points[0]["train_mae"]-loss if index else None)
        events.append({**diagnostic, "sequence": index+1, "elapsed_seconds": 5.+index*3.,
                       "has_incumbent": True, "optimality_certified": True, "is_extension": bool(index),
                       "old_support_indices": initial, "variable_slices": {"z": [0, 5]} if index else {"r": [0, len(initial)]}})
    calibration = {"a": 1., "b": 1., "tau": .1}
    model = {"terminal_labels": labels, "has_model": True, "status": "complete", "path": points,
             "solver_events": events, "initial_supports": initial, "selected_path_index": 1,
             "rx75_calibration": calibration, "bound_expansions": 0}
    truth = subject.physical_truth(directory, labels)
    baseline_arrays = dict(arrays) | {"R": truth["R"], "X": truth["X"]}
    tree_edges = [[1, -1, .6], [-1, 11, .15], [-1, 12, .15], [-1, 13, .15],
                  [1, -2, .45], [-2, 14, .15], [-2, 15, .15]]
    if later_improves:
        tree_edges = [[1, -1, .6], [-1, -2, .18], [-2, 11, .15], [-2, 12, .15], [-1, 13, .15], [1, 14, .15], [1, 15, .15]]
    baseline = {"has_model": True, "terminal_labels": labels, "tree": {"root": 1, "edges": tree_edges},
                "rx75_calibration": calibration, "matrix_origin": "source_adjusted_ordered_regression",
                "source_g_r": [.4, .4, .4, 0., 0.] if source_active else [0.]*5,
                "source_g_x": [.2, .2, .2, 0., 0.] if source_active else [0.]*5}
    return model, arrays, baseline, baseline_arrays


def test_path0_perfect_final_point8_both_preserved_and_false_refinement_named(tmp_path):
    model, arrays, baseline, baseline_arrays = example(tmp_path/"network")
    report = subject.diagnose_condition(model, arrays, baseline, baseline_arrays, tmp_path/"network")
    assert report["stages"]["path0"]["metrics"]["resolution_clade_f1"] == 1
    assert report["stages"]["final"]["metrics"]["resolution_clade_f1"] == pytest.approx(.8)
    assert [p["selected_last_point"] for p in report["path"]] == [False, True]
    assert report["gains"]["greedy_gain_abs"] > 0 and report["gains"]["greedy_gain_rel"] == pytest.approx(1)
    changes = next(row for row in report["changes"] if row["comparison"] == "fixed_family_to_saved_greedy_final" and row["clade_kind"] == "resolution")
    assert changes["added_false"] == [[11, 12]] and changes["added_true"] == []
    assert report["truth_used_for_selection"] is False


def test_later_improvement_and_added_true_clade_also_preserved(tmp_path):
    model, arrays, baseline, baseline_arrays = example(tmp_path/"network", later_improves=True)
    report = subject.diagnose_condition(model, arrays, baseline, baseline_arrays, tmp_path/"network")
    assert report["stages"]["path0"]["metrics"]["resolution_clade_f1"] == pytest.approx(.5)
    assert report["stages"]["final"]["metrics"]["resolution_clade_f1"] == pytest.approx(.8)
    changes = next(row for row in report["changes"] if row["comparison"] == "fixed_family_to_saved_greedy_final" and row["clade_kind"] == "family")
    assert changes["added_true"] == [[14, 15]] and not changes["added_false"]


def test_complete_source_h_rebuilt_and_no_refit_of_location_or_response(tmp_path):
    model, arrays, baseline, baseline_arrays = example(tmp_path/"network", source_active=True)
    report = subject.diagnose_condition(model, arrays, baseline, baseline_arrays, tmp_path/"network")
    for stage in ("path0", "final"):
        assert np.allclose(report["stages"][stage]["artifact"]["source_g_r"], [.4, .4, .4, 0., 0.])
        assert np.allclose(report["stages"][stage]["artifact"]["source_g_x"], [.2, .2, .2, 0., 0.])
    model["path"][0]["source_h"] = [1]*len(model["path"][0]["source_h"])
    broken = subject.diagnose_condition(model, arrays, baseline, baseline_arrays, tmp_path/"network")
    assert "path0" not in broken["stages"] and "0" in broken["errors"]
    assert "final" in broken["stages"]


def test_uncertified_initial_event_is_missing_not_best_f1_selected(tmp_path):
    model, arrays, baseline, baseline_arrays = example(tmp_path/"network")
    model["solver_events"][0]["mip_gap"] = .2
    model["solver_events"][0]["dual_bound"] -= .1
    report = subject.diagnose_condition(model, arrays, baseline, baseline_arrays, tmp_path/"network")
    assert "path0" not in report["stages"] and "final" in report["stages"]
    assert report["gains"]["greedy_gain_abs"] is None
    model["selected_path_index"] = 0
    with pytest.raises(ValueError, match="saved last-certified"):
        subject.diagnose_condition(model, arrays, baseline, baseline_arrays, tmp_path/"network")


def test_elapsed_binds_correct_final_bound_fixed_event_not_first_global_event(tmp_path):
    model, arrays, baseline, baseline_arrays = example(tmp_path/"network")
    earlier = deepcopy(model["solver_events"][0])
    earlier["runtime_seconds"] += 3.
    earlier["elapsed_seconds"] = 1.
    earlier["sequence"] = 1
    model["solver_events"][0]["sequence"] = 2
    model["solver_events"][1]["sequence"] = 3
    model["solver_events"].insert(0, earlier)
    model["bound_expansions"] = 1
    report = subject.diagnose_condition(model, arrays, baseline, baseline_arrays, tmp_path/"network")
    assert report["stages"]["path0"]["solver_sequence"] == 2
    assert report["stages"]["path0"]["elapsed_seconds"] == 5.


def test_production_gate_precedes_models_and_rejects_active_or_unfinished(tmp_path, monkeypatch):
    original = Path.read_text
    def guarded(path, *args, **kwargs):
        assert path.name not in {"model.json", "solver_events.json", "result.json"}
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "read_text", guarded)
    with pytest.raises(ValueError, match="execution_complete"):
        subject.completed_summary(tmp_path, tmp_path/"summary")
    write_json(tmp_path/"controller.lock", {})
    with pytest.raises(ValueError, match="controller"):
        subject.completed_summary(tmp_path, tmp_path/"summary")


def final_fixture(root):
    jobs, baselines, rows, bindings = [], [], [], {}
    model, arrays, rnj, rnj_arrays = example(root/"evaluation_network/paper15")
    target = "paper15__no_source__"+subject.METHOD
    for case in subject.summary.CASES:
        for variant in subject.summary.VARIANTS:
            for method in (*subject.summary.NEW_METHODS, "NJ", "RNJ"):
                key = f"{case}__{variant}__{method}"
                job = {"job_id": key, "case": case, "variant": variant, "method": method}
                directory = root/("baseline_reuse/jobs" if method in ("NJ", "RNJ") else "jobs")/key
                if method in ("NJ", "RNJ"):
                    job["directory"] = directory.relative_to(root).as_posix()
                    baselines.append(job)
                else:
                    jobs.append(job)
                available = case == "paper15" and variant == "no_source" and method in (subject.METHOD, "RNJ")
                result = {**job, "status": "complete" if available else "failed", "has_model": available,
                          "elapsed_seconds": 10. if available else .1, "compute_seconds": 10. if available else .1,
                          "stop_reason": "no_gain" if available else "no_model"}
                write_json(directory/"result.json", result)
                if available:
                    write_json(directory/"model.json", model if method == subject.METHOD else rnj)
                    np.savez_compressed(directory/"model.npz", **(arrays if method == subject.METHOD else rnj_arrays))
                bindings[key] = {"directory": str(directory.resolve()), "artifact_sha256": {
                    path.name: subject._sha(path) for path in directory.iterdir() if path.is_file()}}
                rows.append({**job, "status": result["status"], "result_directory": str(directory.resolve()),
                             "recovery_applied": False, "original_status": result["status"], "combined_spent_seconds": result["elapsed_seconds"]})
    for case in subject.summary.CASES:
        if case != "paper15":
            network(root/"evaluation_network"/case)
    dataset_files = {f"{case}/network/{path.name}": subject._sha(path)
                     for case in subject.summary.CASES for path in (root/"evaluation_network"/case).iterdir() if path.is_file()}
    write_json(root/"dataset_manifest.json", {"file_sha256": dataset_files})
    dataset_sha = subject._sha(root/"dataset_manifest.json")
    write_json(root/"protocol.json", {"jobs": jobs, "baseline_jobs": baselines, "dataset_manifest_sha256": dataset_sha})
    write_json(root/"execution_complete.json", {"completion": {"expected": 64, "finished": 64}})
    manifest = {"partial_fixture_only": False, "denominator": 96, "new_jobs": 64, "reused_baselines": 32,
                "original_root": str(root.resolve()), "original_protocol_sha256": subject._sha(root/"protocol.json"),
                "execution_complete_sha256": subject._sha(root/"execution_complete.json"),
                "dataset_manifest_sha256": dataset_sha,
                "original_attempts": bindings, "recovery_attempts": []}
    report = root/"final_summary"
    write_json(report/"recording_recovery_manifest.json", manifest)
    subject._write_csv(report/"effective_results.csv", rows)
    return report, target


def test_final_saved_fixture_writes16denominator_and_keeps_missing_blank(tmp_path):
    summary, target = final_fixture(tmp_path/"run")
    result = subject.diagnose(tmp_path/"run", summary, tmp_path/"diagnosis")
    rows = read_csv(tmp_path/"diagnosis/initialization_stages.csv")
    assert result["condition_denominator"] == 16 and len(rows) == 16
    assert result["conditions_with_path0"] == 1
    actual = next(row for row in rows if row["job_id"] == target)
    assert float(actual["path0_resolution_clade_f1"]) == 1 and float(actual["final_resolution_clade_f1"]) == pytest.approx(.8)
    assert actual["path0_elapsed_seconds"] == "5.0"
    assert sum(row["path0_resolution_clade_f1"] == "" for row in rows) == 15
    with np.load(tmp_path/"diagnosis/reconstructed/paper15__no_source.npz", allow_pickle=False) as values:
        assert {"rnj_R", "path0_R", "final_R", "path0_source_g_r", "final_prediction"} <= set(values.files)
    with pytest.raises(FileExistsError): subject.diagnose(tmp_path/"run", summary, tmp_path/"diagnosis")


def test_summary_cannot_ignore_eligible_recovery_or_modify_bound_artifact(tmp_path):
    summary, target = final_fixture(tmp_path/"run")
    manifest_path = summary/"recording_recovery_manifest.json"
    manifest = subject._read(manifest_path)
    manifest["recovery_attempts"] = [{"job_id": target, "eligible": True, "recovery_root": str(tmp_path/"retry")}]
    write_json(manifest_path, manifest)
    with pytest.raises(ValueError, match="exact manifest recovery"):
        subject.completed_summary(tmp_path/"run", summary)
    manifest["recovery_attempts"] = []
    write_json(manifest_path, manifest)
    directory = Path(manifest["original_attempts"][target]["directory"])
    (directory/"model.json").write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="artifact hash changed"):
        subject.diagnose(tmp_path/"run", summary, tmp_path/"diagnosis")


def test_completed_mapping_uses_lower_scoring_eligible_recovery_and_blocks_active_retry(tmp_path):
    summary, target = final_fixture(tmp_path/"run")
    manifest_path = summary/"recording_recovery_manifest.json"
    manifest = subject._read(manifest_path)
    retry_root = (tmp_path/"retry").resolve()
    manifest["recovery_attempts"] = [{"job_id": target, "eligible": True, "recovery_root": str(retry_root)}]
    write_json(manifest_path, manifest)
    rows = read_csv(summary/"effective_results.csv")
    row = next(row for row in rows if row["job_id"] == target)
    row.update(result_directory=str(retry_root/"jobs"/target), recovery_applied="True",
               original_resolution_clade_f1="1.0", resolution_clade_f1="0.8")
    subject._write_csv(summary/"effective_results.csv", rows)
    protocol, _, chosen = subject.completed_summary(tmp_path/"run", summary)
    assert chosen[target]["directory"] == retry_root/"jobs"/target
    # Gating the chosen inactive run precedes any attempt/model inspection.
    write_json(retry_root/"controller.lock", {})
    with pytest.raises(ValueError, match="recovery controller is active"):
        subject.verify_chosen(tmp_path/"run", protocol, chosen[target])


def test_partial_manifest_never_passes_production_gate(tmp_path, monkeypatch):
    write_json(tmp_path/"run/execution_complete.json", {"completion": {"expected": 64, "finished": 64}})
    write_json(tmp_path/"summary/recording_recovery_manifest.json", {"partial_fixture_only": True})
    (tmp_path/"summary/effective_results.csv").write_text("job_id\n", encoding="utf-8")
    def forbidden(*args):
        raise AssertionError("result/model inspection must not precede final-manifest gate")
    monkeypatch.setattr(subject.summary, "_gate", forbidden)
    with pytest.raises(ValueError, match="summary is partial"):
        subject.completed_summary(tmp_path/"run", tmp_path/"summary")


def test_recovery_checks_frozen_original_artifact_before_retesting_eligibility(tmp_path, monkeypatch):
    summary, target = final_fixture(tmp_path/"run")
    protocol, manifest, chosen = subject.completed_summary(tmp_path/"run", summary)
    selection = chosen[target]
    selection["recovery"] = {"recovery_root": str(tmp_path/"retry")}
    selection["directory"] = tmp_path/"retry/jobs"/target
    original = Path(selection["first_binding"]["directory"])/"model.json"
    value = subject._read(original)
    value["irrelevant_metadata_edit"] = True  # Would leave same inputs/configuration.
    write_json(original, value)
    def forbidden(*args, **kwargs):
        raise AssertionError("eligibility cannot replace the frozen original hash")
    monkeypatch.setattr(subject.summary, "_recovery_check", forbidden)
    with pytest.raises(ValueError, match="artifact hash changed"):
        subject.verify_chosen(tmp_path/"run", protocol, selection)


@pytest.mark.parametrize("damage", ["coefficient", "missing_file", "extra_file", "changed_manifest"])
def test_truth_network_binding_is_checked_before_model_loading(tmp_path, monkeypatch, damage):
    summary, _ = final_fixture(tmp_path/"run")
    truth = tmp_path/"run/evaluation_network/paper15"
    if damage == "coefficient":
        text = (truth/"branches.csv").read_text(encoding="utf-8")
        (truth/"branches.csv").write_text(text.replace(".2,.1", ".3,.1"), encoding="utf-8")
    elif damage == "missing_file":
        (truth/"buses.csv").unlink()
    elif damage == "extra_file":
        (truth/"unbound.csv").write_text("unbound\n", encoding="utf-8")
    else:
        manifest = subject._read(tmp_path/"run/dataset_manifest.json")
        manifest["extra"] = "changed"
        write_json(tmp_path/"run/dataset_manifest.json", manifest)
    def forbidden(*args, **kwargs):
        raise AssertionError("no saved model may be loaded before truth hash checks")
    monkeypatch.setattr(subject, "_load", forbidden)
    with pytest.raises(ValueError, match="network|dataset manifest"):
        subject.diagnose(tmp_path/"run", summary, tmp_path/"diagnosis")
