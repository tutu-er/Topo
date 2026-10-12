"""Post-hoc 2s -> 10s WZZT budget diagnostic on two fixed v9 histories.

The formal algorithm/protocol/results are read-only. No independent trial or
new success-rate estimate is claimed. A process timeout is reported as an
external diagnostic stop, never as the algorithm's mathematical stop reason.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from time import perf_counter

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CASES = ("n6_ordinary_rich_jitter_dominated_9101",
         "n12_ordinary_partial_jitter_dominated_9101")
sys.path[:0] = [str(ROOT), str(ROOT / "rnj_wzzt_core"), str(HERE)]
import numpy as np
import scipy
from research_experiments.mobile_ev_probing import passive_bridge_v9 as bridge


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def frozen_sources(formal):
    protocol = read(formal / "protocol.json")
    for relative, expected in protocol["source_sha256"].items():
        path = ROOT / relative.replace("\\", "/")
        if sha(path) != expected:
            raise RuntimeError("Frozen formal source changed: " + relative)
    return protocol


def load_history(formal, name):
    folder = formal / name
    result = read(folder / "result.json")
    original = read(folder / "passive.json")
    if result["proposals"] != original:
        raise RuntimeError("Original passive file does not match result")
    baseline_key = result["policies"]["no_EV"]["action"]
    baseline_path = folder / (baseline_key + ".npz")
    if sha(baseline_path) != result["inputs"][baseline_path.name]:
        raise RuntimeError("Baseline observation hash mismatch")
    with np.load(baseline_path, allow_pickle=False) as saved:
        observations = {key: saved[key].copy() for key in saved.files}
    mask = observations["batch_id"] == 0
    if not mask.any() or np.any(mask[np.count_nonzero(mask):]):
        raise RuntimeError("Expected a leading historical batch")
    count = int(mask.sum())
    checked = []
    for filename, expected in result["inputs"].items():
        path = folder / filename
        if sha(path) != expected:
            raise RuntimeError("Observed action file hash mismatch: " + filename)
        with np.load(path, allow_pickle=False) as saved:
            for key, values in observations.items():
                if not np.array_equal(saved[key][:count], values[:count]):
                    raise RuntimeError("History is not strictly paired: " + filename + "/" + key)
        checked.append(filename)
    history = {key: values[:count].copy() for key, values in observations.items()}
    # These are frozen observation-side declarations, not source truth.
    if not (np.all(history["source_nominal_eff"] == .138)
            and np.all(history["source_error_eff"] == .115)):
        raise RuntimeError("Unexpected frozen nominal-source declaration")
    audit = dict(history_rows=count, terminal_count=history["P"].shape[1],
                 strict_history_pairing=True, paired_files=checked,
                 original_result_sha256=sha(folder / "result.json"),
                 original_passive_sha256=sha(folder / "passive.json"),
                 baseline_observation_sha256=sha(baseline_path),
                 truth_used_for_candidate_generation=False)
    return history, original, result, audit


def clade_metrics(supports, truth, n):
    a = {tuple(sorted(s)) for s in supports if 1 < len(s) < n}
    b = {tuple(sorted(s)) for s in truth if 1 < len(s) < n}
    return dict(exact=a == b, f1=2 * len(a & b) / (len(a) + len(b)) if a or b else 1.,
                nontrivial_clade_count=len(a), true_clade_count=len(b))


def summarize(bank, truth):
    n = bank["n_terminal"]
    wzzt = bank["diagnostics"]["wzzt"]
    attempts = wzzt.get("solver_attempts", [])
    have_path = wzzt.get("path_length", 0) > 0
    candidates = [clade_metrics(row["supports"], truth, n) for row in bank["candidates"]]
    return dict(per_solve_seconds=bank["config"]["wzzt_time_limit_per_solve"],
        bridge_seconds=bank["seconds"], wzzt_stop_reason=wzzt.get("stop_reason"),
        wzzt_error=wzzt.get("error"), wzzt_path_length=wzzt.get("path_length", 0),
        wzzt_selected_family_metrics=clade_metrics(wzzt.get("support_indices", []), truth, n) if have_path else None,
        wzzt_selected_supports=wzzt.get("support_indices", []),
        wzzt_attempt_count=len(attempts), wzzt_accepted_stage_count=sum(a["accepted"] for a in attempts),
        wzzt_attempts=[{k: a.get(k) for k in ("stage", "accepted", "added_support", "status",
            "objective", "dual_bound", "mip_gap", "runtime_seconds", "message")} for a in attempts],
        candidate_truth_coverage=any(row["exact"] for row in candidates),
        candidate_top_metrics=candidates[0], candidate_max_f1=max(row["f1"] for row in candidates),
        generated_count=bank["generated_count"], retained_count=bank["retained_count"],
        ranking_is_AC_evidence=False, global_topology_optimality=False)


def worker(formal, output, name):
    source_protocol = frozen_sources(formal)
    history, old, original_result, input_audit = load_history(formal, name)
    folder = output / name
    folder.mkdir(parents=True, exist_ok=True)
    dump(folder / "input_audit.json", input_audit)
    config = dict(old["config"])
    old_limit = config.pop("wzzt_time_limit_per_solve")
    if old_limit != 2.:
        raise RuntimeError("Expected saved 2-second baseline")
    config.update(source_p_nominal=.12, source_q_nominal=.036,
                  source_p_error=.1, source_q_error=.03,
                  wzzt_time_limit=10., prior_proposals=())
    dump(folder / "effective_config.json", dict(
        case=name, kwargs=config,
        one_changed_parameter="wzzt_time_limit per solve: saved 2.0 -> diagnostic 10.0 seconds",
        source_sha256=source_protocol["source_sha256"],
        diagnostic_driver_sha256=sha(__file__),
        original_protocol_sha256=sha(formal / "protocol.json")))
    began = perf_counter()
    new = bridge.build_passive_candidates(history["P"], history["Q"], history["Y_drop"], **config)
    runtime = perf_counter() - began
    dump(folder / "passive_10s.json", new)
    frozen_sources(formal)
    # Evaluation truth is read only after the new candidate bank is frozen.
    baseline_key = original_result["policies"]["no_EV"]["action"]
    truth_file = formal / name / (baseline_key + "_truth.json")
    if sha(truth_file) != original_result["truth_files"][truth_file.name]:
        raise RuntimeError("Evaluation truth hash mismatch")
    truth = read(truth_file)["candidate"]["supports"]
    before, after = summarize(old, truth), summarize(new, truth)
    comparison = dict(case=name, status="completed", wall_seconds=runtime,
        evaluation_truth_sha256=sha(truth_file), saved_2s=before, diagnostic_10s=after,
        changed_parameters=["wzzt_time_limit"],
        old_candidate_ids=[row["id"] for row in old["candidates"]],
        new_candidate_ids=[row["id"] for row in new["candidates"]],
        interpretation="Post-hoc paired computational diagnostic on exactly the saved historical observations; not a new independent success rate.")
    dump(folder / "comparison.json", comparison)
    print(json.dumps(dict(case=name, status="completed", seconds=runtime,
        old_stop=before["wzzt_stop_reason"], new_stop=after["wzzt_stop_reason"],
        old_top_f1=before["candidate_top_metrics"]["f1"], new_top_f1=after["candidate_top_metrics"]["f1"],
        old_coverage=before["candidate_truth_coverage"], new_coverage=after["candidate_truth_coverage"])), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--formal", type=Path, default=HERE / "results_v9" / "formal")
    parser.add_argument("--output", type=Path, default=HERE / "results_v9" / "wzzt_budget_diagnostic")
    parser.add_argument("--worker")
    args = parser.parse_args()
    if args.worker:
        if args.worker not in CASES:
            raise ValueError("Only the two predeclared diagnostic cases are authorized")
        worker(args.formal, args.output, args.worker)
        return
    if (args.output / "protocol.json").exists():
        raise FileExistsError("Use a fresh diagnostic output directory")
    protocol = frozen_sources(args.formal)
    args.output.mkdir(parents=True, exist_ok=True)
    dump(args.output / "protocol.json", dict(
        cases=CASES, changed_parameter=dict(wzzt_seconds_per_solve=[2., 10.]),
        maximum_process_wall_seconds=180, formal_protocol_sha256=sha(args.formal / "protocol.json"),
        source_sha256=protocol["source_sha256"], diagnostic_driver_sha256=sha(__file__),
        python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
        evaluation_scope="Post-hoc rerun of two selected recorded histories; no independent success-rate interpretation."))
    executions = []
    environment = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    for name in CASES:
        folder = args.output / name
        folder.mkdir(parents=True, exist_ok=True)
        command = [sys.executable, "-u", str(Path(__file__).resolve()), "--formal", str(args.formal.resolve()),
                   "--output", str(args.output.resolve()), "--worker", name]
        began = perf_counter()
        with (folder / "execution.log").open("w", encoding="utf-8") as stream:
            try:
                process = subprocess.run(command, env=environment, stdout=stream,
                    stderr=subprocess.STDOUT, timeout=180, check=False)
                status = "completed" if process.returncode == 0 else "worker_failed"
                execution = dict(case=name, status=status, returncode=process.returncode)
            except subprocess.TimeoutExpired:
                execution = dict(case=name, status="externally_stopped_after_180s",
                    returncode=None, mathematical_algorithm_stop_reason=None,
                    note="Subprocess timeout; partial files and log retained. This is not an infeasibility or algorithmic stop certificate.")
        execution.update(wall_seconds=perf_counter() - began, command=command,
                         log_sha256=sha(folder / "execution.log"))
        dump(folder / "execution.json", execution)
        executions.append(execution)
        print(json.dumps(execution), flush=True)
    frozen_sources(args.formal)
    comparisons = [read(args.output / name / "comparison.json") for name in CASES
                   if (args.output / name / "comparison.json").exists()]
    dump(args.output / "results.json", dict(executions=executions, comparisons=comparisons,
        complete=all(item["status"] == "completed" for item in executions),
        changes_formal_algorithm=False, result_interpretation="Post-hoc computational budget diagnostic only."))


if __name__ == "__main__":
    main()
