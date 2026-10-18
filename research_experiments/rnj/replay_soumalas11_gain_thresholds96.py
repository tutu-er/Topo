"""Replay four completed Souma traces using saved solves; never import estimators.

Writes only this task's tmp JSON/CSV.  Preconditions intentionally reject bound
restarts, eligible zero pruning, missing terminal attempts, or family changes.
The output reproduces decisions for the archived solver solutions, not a fresh
solver's tie breaking or wall time.  Truth and F1 do not enter replay.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "tmp/rnj_blocks96_soumalas11_gain_sensitivity_versioned_20261019"
NATIVE = ROOT / "outputs/source_ac288_runtime_20261018/rnj_wzzt/estimation/laminar_l1_milp.py"
RELATIVE_TOLERANCES = (1e-4, 1e-3, 1e-2)
EXPECTED_CORE_SHA = "e3a020bc40f7bb46b5875ac9db9e0d4c962c5f24ebae78a0c30743aae4b24fbd"
JOBS = (
    ("no_source", "outputs/rnj_blocks96_io_retry_20261019", "isolated_io_retry"),
    ("constant", "outputs/rnj_blocks96_20261019", "original"),
    ("intermittent", "outputs/rnj_blocks96_20261019", "original"),
    ("smooth", "outputs/rnj_blocks96_20261019", "original"),
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path, hashes: dict) -> dict:
    hashes[path.relative_to(ROOT).as_posix()] = sha(path)
    return json.loads(path.read_text(encoding="utf-8"))


def finite(value) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(value)


def certified(diagnostics: dict) -> bool:
    """Independent transcription of the frozen numeric certificate predicate."""
    if diagnostics.get("status") != 0 or not diagnostics.get("success"):
        return False
    if diagnostics.get("binary_variable_count") == 0:
        return True
    primal, dual = diagnostics.get("objective"), diagnostics.get("dual_bound")
    if finite(primal) and finite(dual) and abs(primal - dual) <= 1e-10:
        return True
    gap = diagnostics.get("mip_gap")
    return finite(gap) and 0 <= gap <= 1e-10


def close(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-15)


def preconditions(model: dict, result: dict, progress: list[dict]) -> dict:
    """Check sufficient conditions for this restricted one-add trace replay."""
    failures = []

    def check(condition: bool, reason: str) -> None:
        if not condition:
            failures.append(reason)

    path, attempts = model.get("path", []), model.get("attempted_extensions", [])
    parameters = model.get("core_parameters", {})
    frozen = len(model.get("initial_supports", []))
    zero_tolerance = parameters.get("zero_tolerance", math.nan)
    check(result.get("status") == model.get("status") == "complete", "job_not_complete")
    check(result.get("error") is None, "result_has_error")
    check(model.get("training_rows") == 96 and not model.get("validation_supplied"), "split_not_full96")
    check(all(p.get("validation_mae") is None and p.get("validation_se") is None for p in path), "validation_present")
    check(model.get("bound_expansions") == 0, "bounds_expanded")
    check(len(path) > 0 and frozen > 0, "missing_initial_path")
    check(len(attempts) == len(path), "missing_or_extra_terminal_attempt")
    check(model.get("selected_path_index") == model.get("last_path_index") == len(path) - 1, "selection_not_last")
    check(model.get("stop_reason") == result.get("stop_reason") == "no_significant_one_atom_gain", "unexpected_native_stop")
    check(parameters.get("improvement_abs_tol") == 1e-10 and parameters.get("improvement_rel_tol") == 1e-8, "native_gain_parameters_changed")
    check(bool(progress) and progress[0].get("event") == "initialized" and progress[-1].get("event") == "stopped", "incomplete_progress")
    check(all(p.get("bound_expansions") == 0 and not p.get("event", "").startswith("bounds_") for p in progress), "restart_in_progress")
    if progress:
        check(progress[-1].get("path") == path and progress[-1].get("attempted_extensions") == attempts, "final_snapshot_mismatch")
        check(sum(p.get("event") == "extension_attempted" for p in progress) == len(attempts), "attempt_event_count_mismatch")
        check(sum(p.get("event") == "path_accepted" for p in progress) == len(path) - 1, "accept_event_count_mismatch")

    solver_events = model.get("solver_events", [])
    check(len(solver_events) == 1 + len(attempts), "unexpected_fixed_refit_or_missing_solve")
    if solver_events:
        check(solver_events[0].get("is_extension") is False, "initial_solve_not_fixed")
    check(all(e.get("optimality_certified") and certified(e) for e in solver_events), "uncertified_solver_event")
    check(all(certified(p.get("solver", {})) for p in path), "uncertified_path_point")

    zero_counts = []
    maximum_coefficients = []
    if path:
        check(path[0].get("support_indices") == model.get("initial_supports"), "initial_family_mismatch")
    for index, attempt in enumerate(attempts):
        check(certified(attempt.get("diagnostics", {})), f"uncertified_attempt_{index}")
        if index >= len(path):
            continue
        current = path[index]
        old_supports = current["support_indices"]
        candidate_supports = old_supports + [attempt.get("support")]
        r, x = attempt.get("r_values", []), attempt.get("x_values", [])
        check(len(r) == len(x) == len(candidate_supports), f"coefficient_dimensions_{index}")
        eligible = [j for j, (rv, xv) in enumerate(zip(r, x)) if j >= frozen and rv <= zero_tolerance and xv <= zero_tolerance]
        zero_counts.append(len(eligible))
        check(not eligible, f"zero_pruning_branch_possible_{index}")
        maximum_coefficients.append({"r": max(r, default=0), "x": max(x, default=0)})
        check(max(r, default=0) < parameters["bound_active_fraction"] * model["r_upper_bound"], f"r_bound_touch_{index}")
        check(max(x, default=0) < parameters["bound_active_fraction"] * model["x_upper_bound"], f"x_bound_touch_{index}")
        if index + 1 < len(path):
            accepted = path[index + 1]
            check(candidate_supports == accepted["support_indices"], f"family_divergence_{index}")
            check(r == accepted["r_values"] and x == accepted["x_values"], f"refit_or_coefficient_divergence_{index}")
            check(attempt.get("source_h") == accepted.get("source_h"), f"source_path_divergence_{index}")
            check(close(attempt["objective"], accepted["train_mae"]), f"objective_divergence_{index}")
            check(close(current["train_mae"] - attempt["objective"], accepted["accepted_gain"]), f"accepted_gain_mismatch_{index}")
        if index + 1 < len(solver_events):
            event = solver_events[index + 1]
            check(event.get("is_extension") is True and event.get("old_support_indices") == old_supports, f"extension_metadata_mismatch_{index}")
    if path and attempts:
        delta = parameters["improvement_abs_tol"] + parameters["improvement_rel_tol"] * max(abs(path[-1]["train_mae"]), 2.220446049250313e-16)
        check(path[-1]["train_mae"] - attempts[-1]["objective"] <= delta, "terminal_attempt_does_not_stop_native")
    return {"eligible": not failures, "failures": failures,
            "bound_expansions": model.get("bound_expansions"),
            "zero_pruning_eligible_atom_counts_by_attempt": zero_counts,
            "maximum_coefficients_by_attempt": maximum_coefficients,
            "complete_terminal_attempt": len(attempts) == len(path),
            "solver_event_count": len(solver_events), "attempt_count": len(attempts),
            "accepted_path_length": len(path)}


def replay(model: dict, relative: float, checks: dict) -> dict:
    if not checks["eligible"]:
        return {"status": "diverged", "reasons": checks["failures"], "relative_tolerance": relative}
    path, attempts = model["path"], model["attempted_extensions"]
    absolute = model["core_parameters"]["improvement_abs_tol"]
    decisions = []
    for index, attempt in enumerate(attempts):
        current = path[index]
        delta = absolute + relative * max(abs(current["train_mae"]), 2.220446049250313e-16)
        gain = current["train_mae"] - attempt["objective"]
        accepted = gain > delta
        decisions.append({"attempt_index": index, "current_path_index": index,
                          "current_mae": current["train_mae"], "candidate_mae": attempt["objective"],
                          "gain": gain, "gain_tolerance": delta, "accept": accepted})
        if not accepted:
            return {"status": "replayed_archived_trace", "relative_tolerance": relative,
                    "absolute_tolerance": absolute, "selected_path_index": index,
                    "selected_support_count": len(current["support_indices"]),
                    "selected_mae": current["train_mae"], "stop_reason": "no_significant_one_atom_gain",
                    "stopping_attempt_index": index, "decisions": decisions}
        if index + 1 >= len(path):
            return {"status": "diverged", "relative_tolerance": relative,
                    "reasons": ["would_accept_terminal_candidate_without_archived_suffix"], "decisions": decisions}
    return {"status": "diverged", "relative_tolerance": relative, "reasons": ["no_stopping_attempt"]}


def main() -> None:
    hashes = {NATIVE.relative_to(ROOT).as_posix(): sha(NATIVE)}
    if sha(NATIVE) != EXPECTED_CORE_SHA:
        raise ValueError("Frozen native core differs from the audited source")
    runtime_manifest = NATIVE.parents[2] / "runtime_manifest.json"
    hashes[runtime_manifest.relative_to(ROOT).as_posix()] = sha(runtime_manifest)
    cases, rows = [], []
    for variant, relative_root, run_kind in JOBS:
        directory = ROOT / relative_root / "jobs" / f"soumalas11__{variant}__RNJ+WZZT_tree_init"
        model, result = read(directory / "model.json", hashes), read(directory / "result.json", hashes)
        payload = read(directory / "payload.json", hashes)
        if sha(runtime_manifest) != payload["config"]["runtime_manifest_sha256"]:
            raise ValueError("Runtime manifest differs from the job protocol")
        for relative in ("research_experiments/rnj/run_rnj_blocks96.py", "research_experiments/rnj/rx75_edge_floor.py"):
            frozen_source = ROOT / relative_root / "source_snapshot" / relative
            actual_sha = sha(frozen_source)
            hashes[frozen_source.relative_to(ROOT).as_posix()] = actual_sha
            if actual_sha != payload["config"]["source_sha256"][relative]:
                raise ValueError(f"Frozen source differs from the job protocol: {frozen_source}")
        snapshots = [read(p, hashes) for p in sorted((directory / "progress").glob("*.json"))]
        checks = preconditions(model, result, snapshots)
        native = replay(model, model["core_parameters"]["improvement_rel_tol"], checks)
        if checks["eligible"] and native.get("selected_path_index") != model["selected_path_index"]:
            checks["eligible"] = False
            checks["failures"].append("native_replay_does_not_reproduce_saved_selection")
        sensitivity = [replay(model, relative, checks) for relative in RELATIVE_TOLERANCES]
        case = {"variant": variant, "run_kind": run_kind,
                "job_directory": directory.relative_to(ROOT).as_posix(),
                "protocol_sha256": payload["protocol_sha256"], "preconditions": checks,
                "native_replay": native, "sensitivity": sensitivity,
                "native_path_mae": [p["train_mae"] for p in model["path"]],
                "native_path_support_counts": [len(p["support_indices"]) for p in model["path"]]}
        cases.append(case)
        for selection in sensitivity:
            rows.append({"variant": variant, "run_kind": run_kind,
                         **{key: selection.get(key) for key in (
                             "relative_tolerance", "absolute_tolerance", "status", "selected_path_index",
                             "selected_support_count", "selected_mae", "stop_reason", "stopping_attempt_index")},
                         "reasons": ";".join(selection.get("reasons", [])),
                         "truth_used_for_selection": False})
    output = {"schema_version": 1, "scope": "four completed soumalas11 tree_init traces only; no_source uses isolated I/O retry",
              "operation": "arithmetic replay of archived certified single-new-atom solves",
              "equivalence_conditions": "same saved solver solutions and tie breaking; checked no restarts or zero-pruning branches",
              "fresh_solver_invoked": False, "truth_used_for_selection": False,
              "wall_time_counterfactual_estimated": False, "predeclared_relative_tolerances": RELATIVE_TOLERANCES,
              "script_sha256": sha(Path(__file__)), "input_sha256": hashes, "cases": cases}
    OUTPUT.with_suffix(".json").write_text(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    with OUTPUT.with_suffix(".csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"json": str(OUTPUT.with_suffix('.json')), "csv": str(OUTPUT.with_suffix('.csv')),
                      "eligible_cases": sum(c['preconditions']['eligible'] for c in cases),
                      "rows": rows}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
