"""Validation-only E-gate calibration, followed by independent post-evaluation.

This analyzer imports no native solver or optimizer. The single mapped endpoint
determines calibration: it is never replaced by a minimum over its path library.
An exclusive selection.json is written before any truth or test file is opened.
Only genuinely new fixed families are checked with independent SciPy bounded LS.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import pandas as pd

from research_experiments.miqp_loss_benchmark_20261111 import analyze as independent
from . import gate


ROOT = Path(__file__).resolve().parents[2]
GRID = (0., .0001, .001, .003, .01)
PREFERENCE = (.001, .003, .0001, .01, 0.)
TIE_TOLERANCE = 1e-12
VARIANT_FOR_TAU = {tau: "tau_"+format(tau, "g") for tau in GRID}
VARIANTS = ("old_J", *(VARIANT_FOR_TAU[tau] for tau in GRID))


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n",
                    encoding="utf-8", newline="\n")


def require_close(left, right, *, atol=1e-12, rtol=1e-8, name="saved value"):
    left, right = float(left), float(right)
    if not np.isfinite(left) or not np.isfinite(right) or abs(left-right) > atol+rtol*max(abs(left), abs(right)):
        raise ValueError(f"{name} does not reproduce: {left!r} versus {right!r}")


def assert_no_native_imports():
    forbidden = [name for name in sys.modules if name == "gurobipy" or name.startswith("gurobipy.")
                 or name.endswith(".squared_backend") or name.endswith(".replay_worker")]
    if forbidden:
        raise RuntimeError("Post-analysis imported a native optimizer: "+", ".join(forbidden))


def canonical_point(point, n):
    supports = independent.family(point, n)
    rows = sorted(zip(supports, point["r"], point["x"]), key=lambda row: (len(row[0]), row[0]))
    return dict(supports=[list(C) for C, _, _ in rows], r=[float(r) for _, r, _ in rows],
                x=[float(x) for _, _, x in rows])


def choose_tau(validation_rows):
    """Equal-input mean of single mapped-endpoint validation-MSE ratios."""
    if not validation_rows:
        raise ValueError("Cannot calibrate on an empty input set.")
    scores = {tau: float(np.mean([row[VARIANT_FOR_TAU[tau]] for row in validation_rows])) for tau in GRID}
    if not all(np.isfinite(value) and value >= 0 for value in scores.values()):
        raise ValueError("Endpoint validation ratios must be finite and nonnegative.")
    best = min(scores.values())
    chosen = next(tau for tau in PREFERENCE if scores[tau] <= best+TIE_TOLERANCE)
    return dict(tau=chosen, variant=VARIANT_FOR_TAU[chosen], training_inputs=len(validation_rows),
                mean_endpoint_validation_mse_ratio=scores[chosen],
                grid_scores=[dict(tau=tau, variant=VARIANT_FOR_TAU[tau], mean_ratio=scores[tau]) for tau in GRID],
                tie_tolerance=TIE_TOLERANCE, tie_preference=list(PREFERENCE))


def selection_payload(validation_rows, metadata, provenance):
    """Pure calibration; its arguments contain neither truth nor test scores."""
    jobs = [row["job"] for row in validation_rows]
    if len(jobs) != 48 or len(set(jobs)) != 48:
        raise ValueError("The fixed 48 input panel is required.")
    strata = Counter((metadata[j]["case"], float(metadata[j]["voltage_sigma"])) for j in jobs)
    if len(strata) != 12 or set(strata.values()) != {4}:
        raise ValueError("Expected four cases by three noise levels by four shared seeds.")
    seeds = sorted({int(metadata[j]["seed_index"]) for j in jobs})
    if seeds != [0, 1, 2, 3]:
        raise ValueError("Expected four globally shared seed groups.")
    full = choose_tau(validation_rows)
    folds = []
    for heldout in seeds:
        training = [row for row in validation_rows if int(metadata[row["job"]]["seed_index"]) != heldout]
        testing = [row["job"] for row in validation_rows if int(metadata[row["job"]]["seed_index"]) == heldout]
        if len(training) != 36 or len(testing) != 12:
            raise ValueError("Leave-seed-out folds must contain 36 calibration and 12 held-out inputs.")
        folds.append(dict(heldout_seed_index=heldout, calibration_jobs=[r["job"] for r in training],
                          heldout_jobs=testing, selection=choose_tau(training)))
    return dict(schema_version=1, phase="validation_only_selection_frozen_before_postevaluation",
                selection_metric="mapped_endpoint_validation_mse_ratio",
                endpoint_definition="one .24-contracted/QP-refitted final E endpoint, without library minimization",
                denominator="same input's .24-contracted/QP-refitted before-E endpoint validation MSE",
                weighting="equal input; complete panel and every fold remain balanced over case and noise",
                candidate_tau_grid=list(GRID), tie_preference=list(PREFERENCE), tie_tolerance=TIE_TOLERANCE,
                old_regularized_gate_is_calibration_option=False,
                full48=full, leave_shared_seed_out=folds, validation_rows=validation_rows,
                truth_opened=False, test_opened=False, truth_used_in_selection=False,
                future_DR_used=False, provenance=provenance,
                interpretation="historical validation calibration and four-shared-seed cross-fitting; no new prospective test panel")


def freeze_selection(path, payload):
    """Never overwrite an earlier calibration decision."""
    body = deepcopy(payload); body["selection_sha256"] = digest(payload)
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        old = read(path)
        if old != body:
            raise FileExistsError("Existing immutable selection differs; preserve it and use a new study output.")
        return old
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(body, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    return body


def choose_saved(rows, models, *, score, K):
    if not rows:
        raise ValueError("A saved output library is empty.")
    values = {int(row["index"]): float(models[row["model_id"]][score])*(K if score == "validation_mse" else 1.)
              for row in rows}
    best = min(values.values())
    eligible = [row for row in rows if values[int(row["index"])] <= best+1e-12]
    return min(eligible, key=lambda row: (int(row["internal"]), values[int(row["index"])], int(row["index"])))


def guarded_history(output):
    """Verify history without parsing any historical F1/test-score table.

    Parent per-input audit JSONs also contain previously scored F1/test values;
    their parsing is deliberately deferred until after selection.json exists.
    The frozen replay protocol already pins every parent T003 result hash.
    """
    protocol_path, catalog_path = output/"protocol.json", output/"catalog.json"
    protocol, catalog = read(protocol_path), read(catalog_path)
    parent = Path(protocol["parent_output"]).resolve()
    parent_protocol_path, parent_audit_path = parent/"protocol.json", parent/"analysis/verification.json"
    parent_protocol, parent_audit = read(parent_protocol_path), read(parent_audit_path)
    if protocol["planned"] != 48 or len(protocol["jobs"]) != 48 or set(protocol["jobs"]) != set(catalog):
        raise ValueError("Expected 48 completed T003 E replays.")
    if protocol["parameter_selection_metric"] != "mapped_endpoint_validation_mse_ratio":
        raise ValueError("Protocol declares a different validation calibration metric.")
    if (protocol["source_method"] != "T003" or tuple(protocol["tau_grid"]) != GRID
            or tuple(protocol["tie_preference"]) != PREFERENCE
            or protocol["selection_tolerance"] != TIE_TOLERANCE or set(protocol["variants"]) != set(VARIANTS)):
        raise ValueError("Predeclared gate grid or calibration tie rule changed.")
    if not (parent_audit.get("passed") is True and parent_audit.get("complete") is True
            and parent_audit.get("new_native_verified") == parent_audit.get("reused_parent_verified") == 96):
        raise RuntimeError("The complete independently verified parent MIQP panel is required.")
    if parent_audit["protocol_sha256"] != sha(parent_protocol_path):
        raise ValueError("Parent native protocol hash changed.")
    if parent_audit["analyzer_sha256"] != sha(Path(independent.__file__)):
        raise ValueError("Independent helper differs from the verified parent analyzer.")
    if protocol["parent_protocol_sha256"] != sha(parent_protocol_path):
        raise ValueError("Replay protocol references a different parent protocol.")
    if protocol["parent_verification_sha256"] != sha(parent_audit_path):
        raise ValueError("Replay protocol references a different parent numerical audit.")
    if protocol["catalog_sha256"] != sha(catalog_path):
        raise ValueError("Replay catalog changed.")
    for name, expected in {**parent_protocol["source_hashes"], **parent_protocol["source_artifact_hashes"]}.items():
        if sha(parent/"source"/name) != expected:
            raise ValueError("Frozen parent optimizer source changed: "+name)
    if protocol["source_hashes"] != parent_protocol["source_hashes"] or protocol["source_artifact_hashes"] != parent_protocol["source_artifact_hashes"]:
        raise ValueError("Replay declares a different native source manifest.")
    for name, expected in protocol["controller_hashes"].items():
        if sha(output/"controller_source"/name) != expected:
            raise ValueError("Frozen replay controller changed: "+name)
    if sha(Path(gate.__file__)) != protocol["controller_hashes"]["gate.py"]:
        raise ValueError("Current pure gate helper differs from its frozen replay version.")
    result_hashes, parent_result_hashes, input_hashes = {}, {}, {}
    history, validation_rows = {}, []
    for job in protocol["jobs"]:
        meta = catalog[job]
        file = output/"jobs"/job/"result.json"
        if not file.is_file():
            raise RuntimeError("Wait for all 48 replay results; missing "+job)
        result = read(file)
        if (result.get("truth_opened") is not False or result.get("test_opened") is not False
                or result.get("validation_used_for_gate") is not False or result.get("new_MIQP_calls") != 0
                or result.get("native_source_modified") is not False):
            raise ValueError("Replay provenance does not exclude truth/test.")
        parent_file = parent/"jobs"/job/"T003/result.json"
        original = read(parent_file)
        if original["protocol_sha256"] != sha(parent_protocol_path):
            raise ValueError("Parent T003 result references a different frozen protocol.")
        references = result["provenance"]
        if (references["protocol_sha256"] != sha(protocol_path)
                or references["parent_protocol_sha256"] != sha(parent_protocol_path)
                or references["parent_result_sha256"] != sha(parent_file)
                or protocol["parent_result_sha256"][job] != sha(parent_file)
                or references["source_hashes"] != protocol["source_hashes"]
                or references["controller_hashes"] != protocol["controller_hashes"]
                or references["gate_sha256"] != protocol["controller_hashes"]["gate.py"]
                or references["worker_sha256"] != protocol["controller_hashes"]["replay_worker.py"]):
            raise ValueError("Replay references different parent inputs or frozen controllers.")
        common_file = parent/"common"/job/"common.json"
        if sha(common_file) != original["common_sha256"] or references["parent_common_sha256"] != sha(common_file):
            raise ValueError("Parent common preprocessing changed.")
        common = read(common_file)
        if references["parent_candidates_sha256"] != original["candidate_export_sha256"] or sha(parent_file.with_name("candidates.json")) != original["candidate_export_sha256"]:
            raise ValueError("Parent frozen candidate export changed.")
        for module in result["loaded_native_sources"].values():
            path = Path(module["file"]).resolve()
            if (not path.is_relative_to(parent/"source") or path.relative_to(parent/"source").as_posix() != module["relative_path"]
                    or module["sha256"] != protocol["source_hashes"].get(module["relative_path"])):
                raise ValueError("A replay native module escaped the immutable source manifest.")
        splits = {split: independent.load_split(meta[split]) for split in ("train", "validation")}
        n = len(meta["labels"])
        for split in splits:
            if splits[split]["labels"] != meta["labels"]:
                raise ValueError("Historical terminal order changed.")
        measured = {split: sha(meta[split]) for split in splits}
        if measured != {split: parent_protocol["input_hashes"][job][split] for split in splits}:
            raise ValueError("Historical observations differ from the frozen parent panel.")
        if references["input_hashes"] != measured or measured != {split: protocol["input_hashes"][job][split] for split in splits}:
            raise ValueError("Replay historical-input hashes differ.")
        models = {row["id"]: row for row in result["models"]}
        if len(models) != len(result["models"]):
            raise ValueError("Replay model IDs are not unique.")
        K = float(original["normalization"]["K"])
        for model in models.values():
            point = canonical_point(model["point"], n)
            if digest(point) != model["parameter_sha256"] or digest(point["supports"]) != model["family_sha256"]:
                raise ValueError("A replay model's parameter/family hash does not reproduce.")
            for split in splits:
                score = independent.scores(splits[split], point)
                for metric in ("mae", "mse"):
                    require_close(model[split+"_"+metric], score[metric], name=split+" model "+metric)
            R, X = independent.matrices(point, n)
            if (np.asarray(model["R"]).shape != (n, n) or np.asarray(model["X"]).shape != (n, n)
                    or not np.allclose(model["R"], R, rtol=0., atol=1e-12)
                    or not np.allclose(model["X"], X, rtol=0., atol=1e-12)):
                raise ValueError("Exported replay matrices do not match their saved coefficients.")
            residual = splits["validation"]["y"]-splits["validation"]["p"]@R.T-splits["validation"]["q"]@X.T
            days = np.split(residual, 3)
            for field, values in (("validation_day_mae", [float(np.mean(abs(day))) for day in days]),
                                  ("validation_day_mse", [float(np.mean(day**2)) for day in days])):
                if len(model[field]) != 3:
                    raise ValueError("Historical validation must contain three complete days.")
                for saved_value, actual in zip(model[field], values):
                    require_close(saved_value, actual, name=field)
            require_close(model["point"]["objective"], K*model["train_mse"], name="native scaled training objective")
        variants = result["variants"]
        if set(variants) != set(VARIANTS):
            raise ValueError("Replay variants differ from the fixed old_J plus five-tau grid.")
        before = models[result["before_raw_model_id"]]
        before_endpoint = models[result["before_endpoint_model_id"]]
        raw = models[result["raw_model_id"]]
        original_before, before_index, _ = gate._before_exit(original)
        if independent.fingerprint(before["point"]) != independent.fingerprint(original_before):
            raise ValueError("Replay before-E point differs from the exact control-derived parent point.")
        if independent.fingerprint(raw["point"]) != independent.fingerprint(original["exit_record"]["raw_point"]):
            raise ValueError("Replay changed the original count-regularized E proposal.")
        if independent.fingerprint(before_endpoint["point"]) != independent.fingerprint(original["output_candidates"][before_index]["point"]):
            raise ValueError("Calibration denominator is not the mapped before-E endpoint.")
        denominator = float(before_endpoint["validation_mse"])
        if not np.isfinite(denominator) or denominator <= 0:
            raise ValueError("Mapped before-E validation MSE must be positive.")
        ratio_row = dict(job=job)
        for variant_id, variant in variants.items():
            if variant_id == "old_J":
                expected = gate.old_regularized_gate(before["train_mse"], raw["train_mse"],
                    independent.support_count(before["point"], n), independent.support_count(raw["point"], n),
                    K=K, f0_mse=original["normalization"]["f0_mse"])
            else:
                tau = next(t for t in GRID if VARIANT_FOR_TAU[t] == variant_id)
                require_close(variant["tau"], tau, name="variant tau")
                expected = gate.pure_mse_gate(before["train_mse"], raw["train_mse"], K=K, tau=tau)
            if bool(variant["accepted"]) != bool(expected["accepted"]):
                raise ValueError("Counterfactual E acceptance does not reproduce.")
            if variant["gate"].get("time_eligible") is not True or variant["gate"].get("feasibility_eligible") is not True:
                raise ValueError("A target replay lacks a feasible/time-eligible saved E proposal.")
            if variant["path_length"] != len(variant["path_model_ids"]) or variant["prefix_length"] != before_index+1:
                raise ValueError("Replay path metadata does not reproduce.")
            if variant["endpoint_raw_model_id"] != variant["path_model_ids"][-1] or variant["endpoint_model_id"] != variant["library"][variant["path_length"]-1]["model_id"]:
                raise ValueError("Calibration endpoint is not the single last path endpoint.")
            if len(variant["library"]) != variant["path_length"]+1:
                raise ValueError("Replay output library must map every path point and one singleton model.")
            for row in variant["library"]:
                if row["model_id"] not in models:
                    raise ValueError("An output-library row references an absent fitted model.")
                if int(row["internal"]) != independent.support_count(models[row["model_id"]]["point"], n):
                    raise ValueError("Output-library tie count does not reproduce.")
                index = int(row["index"])
                if index < variant["path_length"]:
                    if row["source_model_id"] != variant["path_model_ids"][index]:
                        raise ValueError("A mapped output row references a different raw path point.")
                    source_point = models[row["source_model_id"]]["point"]
                    cal = common["calibration"]
                    allowed = {tuple(C) for C, r, x in zip(source_point["supports"], source_point["r"], source_point["x"])
                               if len(C) == 1 or cal["a"]*r+cal["b"]*x+1e-10 >= .24*cal["median_root_depth"]}
                    if not set(independent.family(models[row["model_id"]]["point"], n)) <= allowed:
                        raise ValueError("Mapped output uses a support excluded by the fixed .24 contraction.")
                elif index != variant["path_length"] or any(len(C) != 1 for C in models[row["model_id"]]["point"]["supports"]):
                    raise ValueError("The last output-library row is not its singleton baseline.")
            for key, score in (("selected_mae", "validation_mae"), ("selected_mse", "validation_mse")):
                chosen = choose_saved(variant["library"], models, score=score, K=K)
                if chosen["model_id"] != variant[key]["model_id"] or chosen["index"] != variant[key]["index"]:
                    raise ValueError("Saved strict output-library selector does not replay.")
            endpoint = models[variant["endpoint_model_id"]]
            ratio = float(endpoint["validation_mse"])/denominator
            require_close(variant["endpoint_validation_mse"], endpoint["validation_mse"], name="single endpoint MSE")
            require_close(variant["endpoint_relative_to_before"], ratio, name="single endpoint validation ratio")
            ratio_row[variant_id] = ratio
        baseline = variants["old_J"]
        if len(baseline["library"]) != len(original["output_candidates"]):
            raise ValueError("Replayed old-J output library changed length.")
        for row, old_row in zip(baseline["library"], original["output_candidates"]):
            model = models[row["model_id"]]
            if row["index"] != old_row["index"] or independent.fingerprint(model["point"]) != independent.fingerprint(old_row["point"]):
                raise ValueError("Replayed old-J library altered saved coefficients.")
            require_close(model["validation_mae"], old_row["validation"], name="old-J validation MAE")
        selected_old = models[baseline["selected_mae"]["model_id"]]
        if independent.fingerprint(selected_old["point"]) != independent.fingerprint(original["selected"]["point"]):
            raise ValueError("Replayed old-J final model differs from the original T003 selected model.")
        if result["new_QP_calls"] != len(result["new_fit_audits"]):
            raise ValueError("Replay new-QP call count differs from its fixed-family audit records.")
        for audit in result["new_fit_audits"]:
            if audit["returned_model_id"] not in models or audit.get("preserve") is not True:
                raise ValueError("A new QP audit lacks its untrimmed fitted model.")
            if independent.canonical(audit["requested_family"]) != independent.canonical(models[audit["returned_model_id"]]["point"]["supports"]):
                raise ValueError("A new QP returned a different fixed support family.")
        result_hashes[job], parent_result_hashes[job], input_hashes[job] = sha(file), sha(parent_file), measured
        history[job] = dict(result=result, models=models, splits=splits, original=original,
                            audit_file=parent/"analysis/jobs"/job/"verified.json", K=K)
        validation_rows.append(ratio_row)
    assert_no_native_imports()
    provenance = dict(protocol_sha256=sha(protocol_path), catalog_sha256=sha(catalog_path),
                      parent_protocol_sha256=sha(parent_protocol_path), parent_analysis_verification_sha256=sha(parent_audit_path),
                      replay_result_sha256=result_hashes, parent_T003_result_sha256=parent_result_hashes,
                      historical_input_sha256=input_hashes, analyzer_sha256=sha(__file__),
                      independent_helper_sha256=sha(Path(independent.__file__)), gate_sha256=sha(Path(gate.__file__)))
    return protocol, catalog, parent, history, validation_rows, provenance


def aggregate(rows, keys):
    frame = pd.DataFrame(rows); output = []
    for grouping, group in frame.groupby(keys, dropna=False, sort=False):
        if not isinstance(grouping, tuple):
            grouping = (grouping,)
        record = dict(zip(keys, grouping))
        record.update(planned=len(group), evaluated=len(group), F1=float(group.clade_f1.mean()),
                      exact=int(group.exact.sum()), recovery_rate=float(group.exact.mean()))
        for key in ("train_mae", "train_mse", "validation_mae", "validation_mse", "test_mae", "test_mse"):
            record[key] = float(group[key].mean())
        output.append(record)
    return output


def analyze(output):
    started = perf_counter(); output = Path(output).resolve()
    protocol, catalog, parent, history, validation_rows, provenance = guarded_history(output)
    # No test arrays or physical truth have been loaded by this point.
    payload = selection_payload(validation_rows, catalog, provenance)
    selection_file = output/"selection.json"
    selection = freeze_selection(selection_file, payload)
    selection_file_sha = sha(selection_file)
    review_directory = output/"recovery/selection_read_order_review_20261111"
    read_order_review = dict(applicable=False)
    old_selection_file = review_directory/"selection_before.json"
    if old_selection_file.exists():
        previous = read(old_selection_file)
        semantic_fields = ("selection_metric", "endpoint_definition", "denominator", "weighting",
                           "candidate_tau_grid", "tie_preference", "tie_tolerance",
                           "old_regularized_gate_is_calibration_option", "full48", "leave_shared_seed_out",
                           "validation_rows")
        if any(previous[key] != selection[key] for key in semantic_fields):
            raise ValueError("Read-order correction changed the previously frozen validation-only calibration decision.")
        read_order_review = dict(applicable=True, prior_selection_sha256=sha(old_selection_file),
            prior_script_sha256=sha(review_directory/"analyze_before.py"),
            semantic_calibration_fields_identical=True, compared_fields=list(semantic_fields),
            full_selection_bytes_expected_identical=False,
            differing_provenance="actual analyzer source hash changes with the read-order correction",
            first_pass_boundary="parent per-input audit JSONs containing historical F1/test scores were parsed for hash/error guards before selection; those score fields were not used in calibration",
            clean_pass_boundary="historical F1/test-score JSON parsing, physical truth and test observations all deferred until selection.json was written")
    selected_tau = float(selection["full48"]["tau"])
    fold_for_job = {job: fold for fold in selection["leave_shared_seed_out"] for job in fold["heldout_jobs"]}
    rows, stages, gates, crossfit_rows, fit_checks, replay_rows = [], [], [], [], [], []
    truth_hashes, test_hashes, profile_cache = {}, {}, {}
    new_native_qp_fits = 0
    for job in protocol["jobs"]:
        meta, saved = catalog[job], history[job]
        result, models, splits = saved["result"], saved["models"], saved["splits"]
        audit = read(saved["audit_file"])
        if (audit["errors"] or audit["raw_hashes"]["scaled_mse/T003"] != provenance["parent_T003_result_sha256"][job]
                or audit["protocol_sha256"] != provenance["parent_protocol_sha256"]
                or audit["analyzer_sha256"] != provenance["independent_helper_sha256"]):
            raise ValueError("Parent T003 result differs from its independent numerical audit.")
        saved["audit"] = audit
        parent_protocol = read(parent/"protocol.json")
        if sha(meta["test"]) != parent_protocol["input_hashes"][job]["test"]:
            raise ValueError("Frozen test observations changed after calibration.")
        test_hashes[job] = sha(meta["test"])
        splits["test"] = independent.load_split(meta["test"])
        if splits["test"]["labels"] != meta["labels"]:
            raise ValueError("Test terminal order differs from frozen historical labels.")
        network = Path(meta["truth_network"])
        hashes = {name: sha(network/name) for name in ("metadata.json", "buses.csv", "branches.csv")}
        if hashes != saved["audit"]["truth_source_hashes"]:
            raise ValueError("True physical feeder differs from the independently verified parent.")
        truth_hashes[job] = hashes
        truth = independent.physical_truth(parent, meta, meta["labels"])
        metrics = {key: independent.metrics(model["point"], splits, truth) for key, model in models.items()}
        header = dict(job=job, case=meta["case"], bundle=meta["bundle"], seed_index=meta["seed_index"],
                      voltage_sigma=meta["voltage_sigma"])
        baseline = result["variants"]["old_J"]
        old_model = baseline["selected_mae"]["model_id"]
        old_metrics = metrics[old_model]
        prior = next(row for row in saved["audit"]["rows"] if row["fit_loss"] == "scaled_mse" and row["method"] == "T003")
        for key in ("clade_f1", "train_mae", "train_mse", "validation_mae", "validation_mse", "test_mae", "test_mse"):
            require_close(old_metrics[key], prior[key], name="old-J post-evaluation "+key)
        if bool(old_metrics["exact"]) != bool(prior["exact"]):
            raise ValueError("Replayed old-J exact recovery differs from its independent parent audit.")
        replay_rows.append(dict(**header, new_QP_calls=result["new_QP_calls"],
                                cache_reuses=len(result["qp_cache_reuses"]),
                                replay_seconds=result["timing"]["replay_seconds"],
                                new_QP_native_seconds=result["timing"]["new_QP_native_seconds"],
                                reused_parent_successful_seconds=result["timing"]["reused_parent_successful_seconds"],
                                reused_parent_operational_seconds=result["timing"]["reused_parent_operational_seconds"],
                                new_MIQP_calls=0))
        for variant_id, variant in result["variants"].items():
            final_id, endpoint_id = variant["selected_mae"]["model_id"], variant["endpoint_model_id"]
            score = metrics[final_id]
            row = dict(**header, variant=variant_id, tau=variant.get("tau"), gate_accepted=bool(variant["accepted"]),
                       changed_acceptance_vs_old_J=bool(variant["accepted"]) != bool(baseline["accepted"]),
                       final_model_id=final_id, endpoint_model_id=endpoint_id,
                       selected_by_full48_validation=variant_id == VARIANT_FOR_TAU[selected_tau],
                       endpoint_relative_to_before=variant["endpoint_relative_to_before"],
                       final_parameters_changed_vs_old_J=independent.fingerprint(models[final_id]["point"]) != independent.fingerprint(models[old_model]["point"]),
                       delta_F1_vs_old_J=score["clade_f1"]-old_metrics["clade_f1"],
                       new_exact=bool(score["exact"] and not old_metrics["exact"]),
                       lost_exact=bool(old_metrics["exact"] and not score["exact"]), **score)
            rows.append(row)
            for stage, model_id in (("endpoint_raw", variant["endpoint_raw_model_id"]),
                                    ("endpoint_mapped", endpoint_id), ("selected_final", final_id)):
                stages.append(dict(**header, variant=variant_id, stage=stage, model_id=model_id, **metrics[model_id]))
            gate_row = variant["gate"]
            gates.append(dict(**header, variant=variant_id, tau=variant.get("tau"), accepted=bool(variant["accepted"]),
                              relative_improvement=gate_row["relative_improvement"], delta_mse=gate_row["delta_mse"],
                              gain_for_acceptance_physical=gate_row.get("regularized_gain_physical", gate_row["delta_mse"]),
                              count_reward_physical=gate_row.get("count_reward_physical", 0.),
                              threshold_physical=gate_row["threshold_physical"],
                              margin_physical=gate_row["margin_physical"],
                              endpoint_relative_to_before=variant["endpoint_relative_to_before"]))
            fold = fold_for_job[job]
            if variant_id == fold["selection"]["variant"]:
                crossfit_rows.append(dict(row, heldout_seed_index=fold["heldout_seed_index"],
                                           calibration_tau=fold["selection"]["tau"],
                                           calibration_inputs=fold["selection"]["training_inputs"]))
        for stage, key in (("before_raw", "before_raw_model_id"), ("before_mapped", "before_endpoint_model_id"),
                           ("proposal_raw", "raw_model_id")):
            stages.append(dict(**header, variant="shared", stage=stage, model_id=result[key], **metrics[result[key]]))
        for audit in result["new_fit_audits"]:
            new_native_qp_fits += 1
            model_id = audit["returned_model_id"]
            model = models[model_id]
            requested = independent.canonical(audit["requested_family"])
            cache_key = (job, requested)
            if cache_key not in profile_cache:
                profile_cache[cache_key] = independent.independent_qp(splits["train"], requested)
            check = independent.fixed_qp_check(model["point"], splits["train"], profile_cache[cache_key], saved["K"])
            fit_checks.append(dict(**header, model_id=model_id, family_sha256=model["family_sha256"],
                                   solver="independent SciPy bounded LS", **check))
    frame = pd.DataFrame(rows)
    if len(frame) != 288 or len(crossfit_rows) != 48:
        raise ValueError("Final evaluation lost the fixed 48 by six-variant denominator.")
    baseline_frame = frame[frame.variant == "old_J"]
    if int(baseline_frame.exact.sum()) != 40 or abs(float(baseline_frame.clade_f1.mean())-.9731043262293263) > 1e-12:
        raise ValueError("Post-evaluation old-J baseline does not reproduce the original 40/48 T003 result.")
    if sha(selection_file) != selection_file_sha:
        raise ValueError("The calibration decision changed during post-evaluation.")
    for job in protocol["jobs"]:
        if (sha(output/"jobs"/job/"result.json") != provenance["replay_result_sha256"][job]
                or sha(parent/"jobs"/job/"T003/result.json") != provenance["parent_T003_result_sha256"][job]):
            raise ValueError("A replay/parent result changed during post-evaluation.")
    assert_no_native_imports()
    destination = output/"analysis"; destination.mkdir(parents=True, exist_ok=True)
    selected_rows = [row for row in rows if row["variant"] == VARIANT_FOR_TAU[selected_tau]]
    summaries = aggregate(rows, ["variant"])
    for summary in summaries:
        subset = frame[frame.variant == summary["variant"]]
        summary.update(E_accepted=int(subset.gate_accepted.sum()),
                       changed_E_decisions_vs_old_J=int(subset.changed_acceptance_vs_old_J.sum()),
                       final_parameters_changed_vs_old_J=int(subset.final_parameters_changed_vs_old_J.sum()),
                       F1_wins_vs_old_J=int((subset.delta_F1_vs_old_J > 1e-12).sum()),
                       F1_losses_vs_old_J=int((subset.delta_F1_vs_old_J < -1e-12).sum()),
                       new_exact=int(subset.new_exact.sum()), lost_exact=int(subset.lost_exact.sum()),
                       mean_endpoint_validation_mse_ratio=float(subset.endpoint_relative_to_before.mean()))
    calibration_summary = aggregate([dict(row, policy="full48_validation_calibrated") for row in selected_rows]
                                    +[dict(row, policy="leave_shared_seed_out") for row in crossfit_rows], ["policy"])
    artifacts = {"per_model.csv": frame, "stages.csv": pd.DataFrame(stages), "gates.csv": pd.DataFrame(gates),
                 "crossfit_inputs.csv": pd.DataFrame(crossfit_rows), "summary.csv": pd.DataFrame(summaries),
                 "calibration_summary.csv": pd.DataFrame(calibration_summary),
                 "cases.csv": pd.DataFrame(aggregate(rows, ["variant", "case"])),
                 "noise.csv": pd.DataFrame(aggregate(rows, ["variant", "voltage_sigma"])),
                 "new_QP_checks.csv": pd.DataFrame(fit_checks), "replay_costs.csv": pd.DataFrame(replay_rows)}
    if not fit_checks:
        artifacts["new_QP_checks.csv"] = pd.DataFrame(columns=("job", "model_id", "family_sha256", "solver",
            "native_mse", "independent_mse", "scaled_objective_excess", "scaled_certificate_gap", "scaled_tolerance"))
    for name, table in artifacts.items():
        table.to_csv(destination/name, index=False, lineterminator="\n")
    verification = dict(passed=True, complete=True, inputs=48, variant_rows=288, crossfit_rows=48,
                        baseline_exact=40, baseline_F1=float(baseline_frame.clade_f1.mean()),
                        baseline_coefficients_and_output_library_unchanged=True,
                        selection_written_before_truth_or_test=True, selection_file_sha256=selection_file_sha,
                        historical_F1_and_test_scores_parsed_only_after_selection=True,
                        selection_read_order_review=read_order_review,
                        selection_payload_sha256=selection["selection_sha256"],
                        full48_selected_tau=selected_tau,
                        LOSO_selected_taus=[fold["selection"]["tau"] for fold in selection["leave_shared_seed_out"]],
                        calibration_uses_F1=False, calibration_uses_test=False, calibration_uses_library_minimum=False,
                        new_native_optimizer_calls=0, new_native_MIQP_calls=0,
                        replay_new_native_QP_fits=new_native_qp_fits,
                        replay_new_QP_native_seconds=sum(row["new_QP_native_seconds"] for row in replay_rows),
                        replay_seconds=sum(row["replay_seconds"] for row in replay_rows),
                        replay_cost_scope="shared replay across all six gate views, not an end-to-end runtime comparison or a MIQP speedup",
                        independent_SciPy_QP_profiles=len(profile_cache), independent_fixed_QP_checks=len(fit_checks),
                        immutable_parent_source_unchanged=True, unchanged_saved_count_regularized_E_proposals=True,
                        raw_gate_and_mapped_endpoint_calibration_distinguished=True,
                        prospective_generalization_claimed=False, independent_random_seed_groups=4,
                        provenance=provenance, truth_source_hashes=truth_hashes, test_input_sha256=test_hashes,
                        result_hashes={name: sha(destination/name) for name in artifacts},
                        seconds=perf_counter()-started)
    write(destination/"verification.json", verification)
    print(pd.DataFrame(summaries).to_string(index=False))
    print(pd.DataFrame(calibration_summary).to_string(index=False))
    return verification


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", nargs="?", type=Path, default=ROOT/"outputs/mse_exit_tuning_20261111")
    analyze(parser.parse_args().output)
