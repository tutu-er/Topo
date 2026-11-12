"""Bounded, paired mechanism diagnostic on three cached measurement windows.

No topology search, AC simulation, calibration or old-file writes are run.
Modified trees receive no valid alarm: old-threshold transfer is descriptive.
"""
import argparse
import copy
from dataclasses import replace
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
THEFT = REPO / "theft_wzzt"
sys.path.insert(0, str(THEFT))
sys.dont_write_bytecode = True

from theft_wzzt.theft.identified_tree import load_identified, to_theft_tree

spec = importlib.util.spec_from_file_location("wg_conservative", THEFT / "detect_conservative.py")
entry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(entry)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def support_for(tree, label):
    if label is None:
        return None
    return next(sorted(s) for s in tree.supports if tree.label_of(s) == label)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE / "wrong_graph_pilot_results.json")
    args = parser.parse_args()
    target = args.output.resolve()
    if not target.is_relative_to(HERE):
        raise ValueError("Output must remain in this isolated research directory")
    if target.exists():
        raise FileExistsError(target)
    start = time.perf_counter()
    original = load_identified(THEFT / "outputs/theft/identified_tree.json")
    cal_path = THEFT / "outputs/theft_paper_v3/calibration_paper15.json"
    calibration = json.loads(cal_path.read_text())
    frozen = json.loads((THEFT / "outputs/theft_paper_v3/frozen_protocol.json").read_text())
    cases = [("null", "null", None), ("exp2", "internal", [101, 102, 103, 104, 105]), ("exp3", "terminal", [108])]
    files = [THEFT / "outputs/theft/identified_tree.json", cal_path]
    files += [THEFT / f"outputs/theft_simplification/measurements_{name}.json" for name, _, _ in cases]
    files += [THEFT / f"outputs/theft_paper_v3/example_{name}.json" for _, name, _ in cases]
    files += [THEFT / name for name in frozen["source_sha256"]]
    files += [THEFT / name for name in calibration["inference_source_sha256"]]
    files = sorted(set(files))
    before = {str(p.relative_to(REPO)): digest(p) for p in files}
    output = dict(status="started", scope="three cached paired mechanism examples; not a V3 rerun or independent performance estimate",
                  window=[32, 64], parameter_model="unchanged conservative strength10 fixed-location LP; re-estimate R/X within original relative bounds",
                  budget_seconds=150, source_sha256=before, script_sha256=digest(Path(__file__)), rows=[], failures=[],
                  modifications_predeclared=["delete clade {108,109,110} and its weights", "swap labels 101 and 108 in every support, retaining per-atom weights"],
                  baseline_tolerance=1e-6, threshold_transfer="Old threshold and location cutoff are intentionally transferred as uncalibrated descriptive stress metrics; modified-tree valid alarm is null.")
    try:
        output["frozen_hash_matches"] = {name: digest(THEFT / name) == expected for name, expected in frozen["source_sha256"].items()}
        if not all(output["frozen_hash_matches"].values()):
            raise RuntimeError("Frozen baseline source mismatch")
        measured = {name: entry.entry.load_measurements(THEFT / f"outputs/theft_simplification/measurements_{name}.json") for name, _, _ in cases}
        threshold = max(max(v) for v in calibration["null_gain_bank"].values())
        output["old_gain_threshold"] = threshold
        output["old_location_cutoff"] = calibration["location_cutoff"]
        golden_results = {}
        for name, golden_name, truth in cases:
            if time.perf_counter() - start > 150:
                raise TimeoutError("Diagnostic budget reached")
            result = entry.detect(copy.deepcopy(measured[name]), original, calibration=calibration)
            golden = json.loads((THEFT / f"outputs/theft_paper_v3/example_{golden_name}.json").read_text())
            checks = {k: abs(result["fit"][k] - golden["fit"][k]) for k in ("gain", "h0_loss", "h1_loss", "gain_voltage", "gain_balance", "gain_penalty")}
            profile_error = max(abs(a["loss"] - b["loss"]) for a, b in zip(result["fit"]["profile"], golden["fit"]["profile"]))
            identical = (result["protocol_signature"] == golden["protocol_signature"] and result["fit"]["best_location"] == golden["fit"]["best_location"] and result["decision"]["alarm"] == golden["decision"]["alarm"] and result["calibrated_candidate_regions"] == golden["calibrated_candidate_regions"])
            check = dict(numeric_differences=checks, profile_max_abs_difference=profile_error, labels_signature_alarm_regions_identical=identical)
            output.setdefault("baseline_replay", {})[name] = check
            if max([profile_error, *checks.values()]) > 1e-6 or not identical:
                raise RuntimeError(f"Baseline replay failed for {name}; modifications not run")
            golden_results[name] = result
        keep = [i for i, s in enumerate(original.supports) if s != frozenset((108, 109, 110))]
        dropped = replace(original, supports=tuple(original.supports[i] for i in keep), r_values=tuple(original.r_values[i] for i in keep), x_values=tuple(original.x_values[i] for i in keep))
        mapping = {101: 108, 108: 101}
        swapped = replace(original, supports=tuple(frozenset(mapping.get(v, v) for v in s) for s in original.supports))
        variants = [("baseline", original), ("drop_108_109_110", dropped), ("swap_101_108", swapped)]
        for variant, model in variants:
            tree, _ = to_theft_tree(model)
            tree.incidence()  # validates connected acyclic rooted representation
            for name, _, truth in cases:
                if time.perf_counter() - start > 150:
                    raise TimeoutError("Diagnostic budget reached")
                result = golden_results[name] if variant == "baseline" else entry.detect(copy.deepcopy(measured[name]), model, calibration=None)
                fit = result["fit"]
                minimum = min(p["loss"] for p in fit["profile"])
                transferred_set = [support_for(model, p["location"]) for p in fit["profile"] if p["loss"] - minimum <= calibration["location_cutoff"] + 1e-7]
                alarm_transfer = fit["gain"] > threshold + 1e-7
                output["rows"].append(dict(variant=variant, example=name, replicate=int(measured[name]["scenario_settings"]["replicate"]),
                    support_count=len(model.supports), protocol_signature=result["protocol_signature"],
                    true_original_support=truth, truth_support_present=None if truth is None else frozenset(truth) in model.supports,
                    best_support=support_for(model, fit["best_location"]), valid_alarm=result["decision"]["alarm"],
                    uncalibrated_old_threshold_alarm=alarm_transfer,
                    uncalibrated_old_threshold_false_alarm=alarm_transfer if name == "null" else None,
                    uncalibrated_old_cutoff_set=transferred_set,
                    uncalibrated_old_cutoff_contains_original_truth=None if truth is None else truth in transferred_set,
                    amplitude_max_abs_change_from_baseline_pu=max(abs(a-b) for a,b in zip(result["amplitude_point_pu"], golden_results[name]["amplitude_point_pu"])),
                    fit=fit))
        output["status"] = "complete"
    except Exception as error:
        output["status"] = "failed"
        output["failures"].append(dict(error=repr(error), traceback=traceback.format_exc()))
    output["elapsed_seconds"] = time.perf_counter() - start
    output["changed_old_files"] = [str(p.relative_to(REPO)) for p in files if digest(p) != before[str(p.relative_to(REPO))]]
    with target.open("x", encoding="utf-8") as stream:
        json.dump(output, stream, ensure_ascii=False, indent=2)
    print(json.dumps({"status": output["status"], "elapsed_seconds": output["elapsed_seconds"], "rows": len(output["rows"]), "failures": output["failures"], "changed_old_files": output["changed_old_files"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
