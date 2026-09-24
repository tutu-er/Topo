"""Exploratory, post-result RNJ threshold tuning on the primary run's cache.

Use the same RX75 regression, four factors, training-only fixed-tree L1 fits,
validation MAE and tie rule as the primary validation-tuned NJ control. This
script never regenerates observations and refuses to start until the primary
run has a completion.json marker. This is not a preregistered experiment.
"""
from __future__ import annotations

import os
for _key in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[_key] = "1"

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
import time
import traceback

CORE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CORE))
sys.path.insert(0, str(CORE / "experiments"))

import numpy as np
import pandas as pd
from rnj_wzzt.estimation.multiscenario import fit_projected_sensitivity, preprocess_scenarios
from rnj_wzzt.estimation.preprocessing import RECIPE
from rnj_wzzt.estimation.laminar_l1_milp import solver_diagnostics_prove_optimality
from rnj_wzzt.graph.sensitivity_geometry import sensitivity_geometry
from rnj_wzzt.graph.rooted_neighbor_joining import rooted_neighbor_joining
from rnj_wzzt.graph.rooted_hierarchy import rooted_clades
from rooted_ablation_support import rooted_scores, serialized
from run_rooted_ablation import fixed_fit, prediction_metrics
from stress_support import plain, write_json, paired_summary, source_fingerprint

FACTORS = (0.0, 0.04, 0.08, 0.16)
METHOD = "rnj_validation_matched"
BASELINES = ("classical_nj_validation", "rnj_RX75", "rnj_fixed_tree_lp")
OBSERVATION_KEYS = ("P_terminal", "Q_terminal", "drop_target")


class NoValidCandidate(RuntimeError):
    def __init__(self, candidates):
        super().__init__("all four validation candidates failed")
        self.candidates = candidates


def validation_mae(scenarios, r, x, fitted_intercepts):
    """Use frozen training intercepts; this function has no test/truth inputs."""
    residual = np.vstack([
        scenario["drop_target"].to_numpy()
        - scenario["P_terminal"].to_numpy() @ r.T
        - scenario["Q_terminal"].to_numpy() @ x.T - fitted_intercepts[day]
        for day, scenario in enumerate(scenarios)
    ])
    return float(np.mean(np.abs(residual)))


def infer_clades(r, x, terminals, root, factor):
    geometry = sensitivity_geometry(r, x, "RX_75R_25X")
    tau = factor * max(float(np.median(geometry.root_depths)), 1e-12)
    tree = rooted_neighbor_joining(
        geometry.shared_paths, geometry.root_depths, terminals, root,
        group_tolerance=tau,
    )
    return rooted_clades(tree.edges, root, terminals)


def choose_rnj_on_validation(train, validation, terminals, root, solve_seconds):
    """Return a frozen model chosen only from training and validation data.

    LP result acceptance matches the primary NJ control: a returned fixed_fit
    model is eligible, and certification is recorded without imposing an extra
    selection filter absent from that control. Nonfinite validation is rejected.
    """
    started = time.perf_counter()
    regression_diagnostics = {}
    r, x, _, _ = fit_projected_sensitivity(
        preprocess_scenarios(train, RECIPE), diagnostics=regression_diagnostics,
    )
    regression_seconds = time.perf_counter() - started
    candidates, models = [], {}
    for factor in FACTORS:
        candidate_started = time.perf_counter()
        record = {"factor": factor, "status": "failed", "validation_mae": None}
        try:
            clades = infer_clades(r, x, terminals, root, factor)
            solution, rr, xx, active = fixed_fit(train, clades, terminals, solve_seconds)
            error = validation_mae(validation, rr, xx, solution.intercepts)
            if not np.isfinite(error):
                raise ValueError("validation MAE is not finite")
            record.update(
                status="complete", validation_mae=error,
                clades=serialized(clades), active_weight_clades=serialized(active),
                fixed_solver=solution.diagnostics.to_dict(),
                fixed_refit_certified=solver_diagnostics_prove_optimality(solution.diagnostics),
            )
            models[factor] = dict(clades=clades, r=rr, x=xx, intercepts=solution.intercepts)
        except Exception as exc:
            record.update(error=repr(exc))
        record["elapsed_seconds"] = time.perf_counter() - candidate_started
        candidates.append(record)
    valid = [item for item in candidates if item["status"] == "complete"]
    if not valid:
        raise NoValidCandidate(candidates)
    # Exactly the primary NJ rule, including ties: smaller factor wins.
    selected = min(valid, key=lambda item: (item["validation_mae"], item["factor"]))
    return dict(
        **models[selected["factor"]], selected_factor=selected["factor"],
        validation_mae=selected["validation_mae"],
        selected_fixed_refit_certified=selected["fixed_refit_certified"],
        candidates=candidates, regression_diagnostics=regression_diagnostics,
        regression_seconds=regression_seconds,
        elapsed_seconds=time.perf_counter() - started,
    )


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_cached_job(job_directory):
    """Restore original row labels when recorded; never regenerate a sample."""
    directory = Path(job_directory)
    metadata_path, input_path = directory / "metadata.json", directory / "inputs.npz"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    with np.load(input_path, allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    terminals = arrays["terminals"].tolist()
    if not terminals or len(set(terminals)) != len(terminals):
        raise ValueError("cached terminal labels must be nonempty and unique")
    n = len(terminals)
    truth = tuple(arrays[key] for key in ("R_true", "X_true"))
    if any(matrix.shape != (n, n) or not np.isfinite(matrix).all() for matrix in truth):
        raise ValueError("cached truth matrices have invalid shapes or values")
    pattern = re.compile(r"^(train|validation|test)_(\d+)_(.+)$")
    grouped = {}
    for key, array in arrays.items():
        match = pattern.match(key)
        if match:
            split, day, field = match.groups()
            grouped.setdefault(split, {}).setdefault(int(day), {})[field] = array
    sets = {}
    scenario_count = None
    for split in ("train", "validation", "test"):
        days = grouped.get(split, {})
        if not days or sorted(days) != list(range(len(days))):
            raise ValueError(f"{split} scenario indices must be nonempty and contiguous")
        if scenario_count is None:
            scenario_count = len(days)
        elif len(days) != scenario_count:
            raise ValueError("train/validation/test must contain the same scenario identities")
        scenarios = []
        indices = metadata.get("source_sample_indices", {}).get(split, [None] * len(days))
        if len(indices) != len(days):
            raise ValueError(f"{split} row-index metadata does not match scenarios")
        for day in range(len(days)):
            fields = days[day]
            missing = set(OBSERVATION_KEYS) - set(fields)
            if missing:
                raise ValueError(f"{split}/{day} missing observation arrays: {sorted(missing)}")
            rows = fields["P_terminal"].shape[0]
            index = pd.RangeIndex(rows) if indices[day] is None else pd.Index(indices[day])
            if len(index) != rows or not index.is_unique:
                raise ValueError(f"{split}/{day} cached row indices are invalid")
            scenario = {"name": f"scenario_{day}"}
            for field, array in fields.items():
                if field in OBSERVATION_KEYS and (
                    array.shape != (rows, n) or not np.isfinite(array).all()
                ):
                    raise ValueError(f"{split}/{day}/{field} is not a finite samples-by-terminals array")
                if array.shape == (rows, n):
                    scenario[field] = pd.DataFrame(array.copy(), index=index, columns=terminals)
                elif array.shape == (rows,):
                    scenario[field] = pd.Series(array.copy(), index=index)
                else:
                    raise ValueError(f"{split}/{day}/{field} has an unsupported array shape")
            scenarios.append(scenario)
        expected_rows = metadata.get("snapshots", {}).get(split)
        if expected_rows is not None and expected_rows != sum(len(s["P_terminal"]) for s in scenarios):
            raise ValueError(f"{split} observation count differs from metadata")
        sets[split] = scenarios
    return dict(
        sets=sets, terminals=terminals, truth=truth,
        true_clades={frozenset(clade) for clade in metadata["truth_clades"]},
        root=int(metadata["root"]), metadata=metadata,
        input_sha256=_sha256(input_path), metadata_sha256=_sha256(metadata_path),
    )


def evaluate_selected(selected, sets, truth, true_clades, terminals):
    """Truth and test enter only after threshold and model have been frozen."""
    return {
        **rooted_scores(selected["clades"], true_clades, terminals),
        **prediction_metrics(sets, selected["r"], selected["x"], selected["intercepts"], truth),
    }


def run_cached_job(source, output, job):
    started = time.perf_counter()
    base = {key: job[key] for key in ("tier", "stress", "shape", "repeat", "job_id", "cluster_id")}
    row = dict(base, method=METHOD, status="failed", n=job["n"], clade_f1=None,
               clade_exact=None, selected_factor=None, elapsed_seconds=None, test_mae=None, validation_mae=None)
    details = {"job": job, "exploratory_post_result": True}
    directory = Path(output) / "jobs" / job["job_id"]
    directory.mkdir(parents=True, exist_ok=False)
    try:
        cached = load_cached_job(Path(source) / "jobs" / job["job_id"])
        if cached["metadata"]["job"]["job_id"] != job["job_id"]:
            raise ValueError("cached metadata job_id does not match the planned job")
        row["n"] = len(cached["terminals"])
        details.update(input_sha256=cached["input_sha256"], metadata_sha256=cached["metadata_sha256"])
        # Deliberately pass only these two sets into model selection.
        selected = choose_rnj_on_validation(
            cached["sets"]["train"], cached["sets"]["validation"], cached["terminals"],
            cached["root"], job["solve_seconds"],
        )
        # From this statement onwards model/threshold selection is already final.
        row.update(evaluate_selected(selected, cached["sets"], cached["truth"],
                                     cached["true_clades"], cached["terminals"]))
        row.update(
            status="complete", selected_factor=selected["selected_factor"],
            elapsed_seconds=selected["elapsed_seconds"],
            regression_seconds=selected["regression_seconds"],
            selected_fixed_refit_certified=selected["selected_fixed_refit_certified"],
            candidates_available=sum(item["status"] == "complete" for item in selected["candidates"]),
        )
        details.update(candidates=selected["candidates"], selected_factor=selected["selected_factor"],
                       selected_clades=serialized(selected["clades"]),
                       regression_diagnostics=selected["regression_diagnostics"])
        np.savez_compressed(directory / "selected_model.npz", R=selected["r"], X=selected["x"],
                            intercepts=selected["intercepts"], terminals=cached["terminals"])
    except Exception as exc:
        row["error"] = repr(exc)
        details.update(error=repr(exc), traceback=traceback.format_exc())
        if isinstance(exc, NoValidCandidate):
            details["candidates"] = exc.candidates
    details.update(row=row, wall_seconds=time.perf_counter() - started)
    write_json(directory / "selection.json", details)
    return row


def paired_differences(rows, source_rows):
    """One output for every planned new row, including missing baseline scores."""
    prior = pd.DataFrame(source_rows)
    if not prior.empty and prior.duplicated(["tier", "stress", "method", "job_id"]).any():
        raise ValueError("source result rows contain duplicate method/job comparisons")
    indexed = {(r["job_id"], r["method"]): r for r in source_rows}
    comparisons = []
    for current in rows:
        for baseline in BASELINES:
            reference = indexed.get((current["job_id"], baseline), {})
            out = {key: current[key] for key in ("tier", "stress", "shape", "n", "repeat", "job_id", "cluster_id")}
            out.update(method=METHOD, reference=baseline, current_status=current["status"],
                       reference_status=reference.get("status", "missing"),
                       selected_factor=current.get("selected_factor"),
                       reference_selected_factor=reference.get("selected_collapse_factor"))
            for metric in ("clade_f1", "test_mae", "validation_mae", "elapsed_seconds"):
                a, b = current.get(metric), reference.get(metric)
                finite = a is not None and b is not None and np.isfinite(a) and np.isfinite(b)
                out[metric + "_delta"] = float(a - b) if finite else None
            comparisons.append(out)
    return comparisons


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    # A hard gate protects concurrent primary wall-clock-limited experiments.
    if not (args.source / "completion.json").is_file():
        parser.error("primary run is not complete; completion.json is required before any fitting")
    if args.output.exists():
        parser.error("output already exists; use a new directory")
    protocol = json.loads((args.source / "protocol.json").read_text(encoding="utf-8"))
    if source_fingerprint(CORE) != protocol["source_sha256"]:
        parser.error("primary experiment/core source hashes changed; matched tuning requires the recorded implementation")
    jobs = [job for job in protocol["jobs"] if job["tier"] in ("synthetic", "ac")]
    if not jobs or len({job["job_id"] for job in jobs}) != len(jobs):
        parser.error("source protocol must contain unique synthetic/AC jobs")
    source_rows = [json.loads(line) for line in (args.source / "rows.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    args.output.mkdir(parents=True)
    write_json(args.output / "protocol.json", {
        "exploratory_post_result": True,
        "status": "Added after examining primary results; not preregistered.",
        "source": str(args.source.resolve()), "source_protocol_sha256": _sha256(args.source / "protocol.json"),
        "script_sha256": _sha256(__file__), "primary_source_sha256": protocol["source_sha256"],
        "jobs": jobs, "factors": FACTORS, "method": METHOD, "baselines": BASELINES,
        "selection": "Minimum validation MAE; ties choose smaller factor, identical to primary NJ.",
        "observations": "Original inputs.npz and metadata.json only; no simulation or extra observations.",
        "fit": "Same daily-demeaned ordered RX75 fit and raw original-terminal fixed-tree L1 fits as primary NJ.",
        "solver_policy": "Same per-candidate solve_seconds and fixed_fit acceptance; certification recorded.",
        "score": "Structural clades retained even when a refitted edge coefficient is zero.",
        "test_truth_policy": "test observations and truth are passed only to post-selection evaluation.",
        "timing": "Includes regression and all four candidate inference/refit/validation costs; excludes disk/reporting. Timed in a later run with different contention, not a controlled runtime comparison.",
        "statistics": "Post-result exploratory paired differences; cluster bootstrap is descriptive for this same bank, not independent confirmation.",
    })
    rows = []
    for number, job in enumerate(jobs, 1):
        row = run_cached_job(args.source, args.output, job)
        rows.append(row)
        with (args.output / "rows.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(plain(row), ensure_ascii=False, allow_nan=False) + "\n")
        print(f"{number}/{len(jobs)} {job['job_id']}: {row['status']}", flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(args.output / "results.csv", index=False)
    frame.groupby(["tier", "stress"], dropna=False).agg(
        attempted=("status", "size"), available=("clade_f1", "count"),
        mean_f1=("clade_f1", "mean"), exact_rate=("clade_exact", "mean"),
        mean_test_mae=("test_mae", "mean"), mean_seconds=("elapsed_seconds", "mean"),
    ).reset_index().to_csv(args.output / "summary.csv", index=False)
    frame.groupby(["tier", "stress", "selected_factor"], dropna=False).size().reset_index(name="count").to_csv(
        args.output / "threshold_counts.csv", index=False,
    )
    differences = paired_differences(rows, source_rows)
    pd.DataFrame(differences).to_csv(args.output / "paired_differences.csv", index=False)
    planned = {job["job_id"] for job in jobs}
    pairs = []
    for baseline in BASELINES:
        references = [row for row in source_rows if row["job_id"] in planned and row["method"] == baseline]
        pairs.extend(paired_summary([*rows, *references], baseline))
    write_json(args.output / "paired_comparisons.json", pairs)
    pd.DataFrame(pairs).to_csv(args.output / "paired_comparisons.csv", index=False)
    write_json(args.output / "completion.json", {
        "jobs": len(jobs), "complete": int((frame.status == "complete").sum()),
        "failed": int((frame.status != "complete").sum()), "exploratory_post_result": True,
        "primary_source_unchanged": source_fingerprint(CORE) == protocol["source_sha256"],
    })


if __name__ == "__main__":
    main()

