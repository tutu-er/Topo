"""Paired R/X and RNJ comparison with free or ridge-regularized common modes.

Run from the standalone core, for example:
    python experiments/run_common_mode_comparison.py --repeats 5 --workers 2
The default production pipeline and its regression configuration are unchanged.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from time import perf_counter

# Each worker solves small dense QPs; avoid nested BLAS thread pools.
for _thread_option in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[_thread_option] = "1"
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from rnj_wzzt.data.paper_style_case_bank import CASE_BUILDERS
from rnj_wzzt.estimation.constrained_least_squares import solve_symmetric_least_squares
from rnj_wzzt.estimation.multiscenario import _aligned_arrays, fit_projected_sensitivity, preprocess_scenarios
from rnj_wzzt.estimation.preprocessing import RECIPE, squared_voltage_drop_from_observed_root
from rnj_wzzt.graph.rooted_hierarchy import rooted_clades
from rnj_wzzt.graph.rooted_neighbor_joining import rooted_neighbor_joining
from rnj_wzzt.graph.sensitivity_geometry import sensitivity_geometry
from rnj_wzzt.models.lin_distflow import build_reduced_sensitivity_matrices, impedance_distance_from_reduced_R
from rnj_wzzt.reporting import _family_score, _serialize_family, _truth_nontrivial_clades
from rnj_wzzt.scenario.simulation import _simulate_pool, _terminal_buses


ALPHA_GRID = (1e-8, 1e-6, 1e-4, 1e-2, 1.0)
GAMMA_GRID = (1e-3, 0.03, 1.0, 30.0, 1e3, 1e6)
MODES = ("current", "common_free", "common_dual_ridge")


def write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def root_measurements(raw: list[dict], relative_noise: float, replicate: int) -> list[dict]:
    """Add only root-meter error; the physical AC scenarios stay paired."""
    measured = []
    for index, scenario in enumerate(raw):
        item = dict(scenario)
        truth = scenario["root_voltage"].copy()
        rng = np.random.default_rng(70_000_019 + 100_003 * replicate + index)
        observed = truth + rng.normal(0.0, relative_noise * np.abs(truth), size=len(truth))
        item["root_voltage_true"] = truth
        item["root_voltage"] = observed
        item["drop_target"] = squared_voltage_drop_from_observed_root(item["V_terminal"], observed)
        measured.append(item)
    return measured


def arrays(raw: list[dict]) -> dict:
    designs, targets = _aligned_arrays(raw)
    return {
        "A": np.vstack(designs), "Y": np.vstack(targets),
        "A_mean": np.vstack([a.mean(axis=0) for a in designs]),
        "Y_mean": np.vstack([y.mean(axis=0) for y in targets]),
        "scenario": np.repeat(np.arange(len(raw)), [len(a) for a in designs]),
        "root_true": np.concatenate([s["root_voltage_true"].to_numpy() for s in raw]),
        "root_observed": np.concatenate([s["root_voltage"].to_numpy() for s in raw]),
    }


def output_transform(n: int, gamma: float) -> np.ndarray:
    common = np.ones((n, n)) / n
    return np.eye(n) - common + np.sqrt(gamma / (1.0 + gamma)) * common


def training_intercepts(data: dict, blocks: np.ndarray) -> np.ndarray:
    return data["Y_mean"] - data["A_mean"] @ np.vstack(blocks)


def errors(data: dict, blocks: np.ndarray, intercepts: np.ndarray, gamma: float):
    residual = data["Y"] - data["A"] @ np.vstack(blocks) - intercepts[data["scenario"]]
    common = np.zeros(len(residual)) if math.isinf(gamma) else residual.mean(axis=1) / (1.0 + gamma)
    adjusted = residual - common[:, None]
    denominator = np.sum((data["Y"] - data["Y"].mean(axis=0))**2)
    return {
        "prediction_rmse": float(np.sqrt(np.mean(residual**2))),
        "prediction_mae": float(np.mean(np.abs(residual))),
        "profile_fit_rmse": float(np.sqrt(np.mean(adjusted**2))),
        "prediction_r2": float(1.0 - np.sum(residual**2) / denominator),
        "profile_fit_r2": float(1.0 - np.sum(adjusted**2) / denominator),
        "common_rms": float(np.sqrt(np.mean(common**2))),
    }, residual, common


def relative_error(estimate, truth) -> float:
    return float(np.linalg.norm(estimate-truth) / max(np.linalg.norm(truth), 1e-15))


def rnj_metrics(blocks, terminals, root, truth_clades, fixed_tolerances=None):
    values, traces, tolerances = {}, {}, {}
    for mode, label in (("R", "r"), ("X", "x"), ("RX_75R_25X", "rx75")):
        geometry = sensitivity_geometry(*blocks, mode)
        tau = 0.16 * max(float(np.median(geometry.root_depths)), 1e-12)
        tree = rooted_neighbor_joining(geometry.shared_paths, geometry.root_depths, terminals, root, group_tolerance=tau)
        clades = rooted_clades(tree.edges, root, terminals)
        score = _family_score(clades, truth_clades)
        values.update({f"rnj_{label}_{key.removeprefix('nontrivial_support_')}": value for key, value in score.items()})
        values[f"rnj_{label}_exact"] = int(clades == truth_clades)
        values[f"rnj_{label}_tau"] = tau
        values[f"rnj_{label}_clades"] = _serialize_family(clades)
        tolerances[label] = tau
        traces[f"{label}_shared"] = geometry.shared_paths
        traces[f"{label}_root_depths"] = geometry.root_depths
        traces[f"{label}_edges"] = np.asarray(tree.edges)
        if fixed_tolerances is not None:
            fixed = rooted_neighbor_joining(geometry.shared_paths, geometry.root_depths, terminals, root, group_tolerance=fixed_tolerances[label])
            fixed_clades = rooted_clades(fixed.edges, root, terminals)
            values[f"rnj_{label}_fixed_baseline_tau_f1"] = _family_score(fixed_clades, truth_clades)["nontrivial_support_f1"]
    return values, traces, tolerances


def run_job(job: dict) -> dict:
    started = perf_counter()
    out = Path(job["output"]) / job["case"] / f"repeat_{job['repeat']:02d}" / job["regime"]
    out.mkdir(parents=True, exist_ok=True)
    if (out / "result.json").exists():
        previous = json.loads((out / "result.json").read_text(encoding="utf-8"))
        if previous["job"] == job:
            return previous
        raise ValueError(f"Existing results have a different configuration: {out}")
    datasets, raw_sets = {}, {}
    for offset, split in enumerate(("train", "tune", "test")):
        replicate = 3 * job["repeat"] + offset
        net, raw = _simulate_pool(job["case"], 96, replicate, 3, 0.005, 0.0002)
        measured = root_measurements(raw, job["root_noise"], replicate)
        raw_sets[split] = measured
        datasets[split] = arrays(measured)
    terminals = _terminal_buses(net)
    truth = np.asarray(build_reduced_sensitivity_matrices(net, terminals, voltage_model="squared-voltage"))
    truth_clades = _truth_nontrivial_clades(net, terminals)
    training = preprocess_scenarios(raw_sets["train"], RECIPE)
    designs, targets = _aligned_arrays(training)
    design = np.vstack([a-a.mean(axis=0) for a in designs])
    target = np.vstack([y-y.mean(axis=0) for y in targets])
    n = target.shape[1]
    baseline_diagnostics = {}
    r, x, _, _ = fit_projected_sensitivity(training, diagnostics=baseline_diagnostics)
    baseline = np.asarray([r, x])
    margins = np.asarray([baseline_diagnostics["r_margin"], baseline_diagnostics["x_margin"]])
    alpha_scale = float(np.sum(design**2) / (2*n))
    baseline_rnj, _, fixed_tolerances = rnj_metrics(baseline, terminals, net.root_bus, truth_clades)
    input_archive = {f"{split}_{key}": value for split, data in datasets.items() for key, value in data.items()}
    input_archive.update(R_true=truth[0], X_true=truth[1], terminals=np.asarray(terminals), margins=margins, centered_A=design, centered_Y=target)
    np.savez_compressed(out / "data.npz", **input_archive)

    def fit(alpha: float, gamma: float):
        return solve_symmetric_least_squares(design, target, baseline, margins, alpha, 1500, output_transform=output_transform(n, gamma))

    free_raw, free_diagnostics = fit(0.0, 0.0)
    # For gamma=alpha=0, the two uniform 11' shifts are unidentifiable.
    # Choose the smallest-Frobenius representative along those shifts, without truth.
    shifts = np.min(free_raw, axis=(1, 2))
    free = free_raw - shifts[:, None, None]
    assert np.allclose((design @ np.vstack(free_raw)-target) @ output_transform(n, 0), (design @ np.vstack(free)-target) @ output_transform(n, 0), atol=1e-10)
    candidates, failures, best = [], [], None
    for a in job["alpha_grid"]:
        for gamma in job["gamma_grid"]:
            try:
                blocks, diagnostics = fit(a * alpha_scale, gamma)
                intercepts = training_intercepts(datasets["train"], blocks)
                tuning, _, _ = errors(datasets["tune"], blocks, intercepts, gamma)
                candidate = {"alpha_dimensionless": a, "alpha": a*alpha_scale, "gamma": gamma, **tuning, **diagnostics}
                candidates.append(candidate)
                # No tuning/test common-mode profiling, no truth/topology in this selection.
                key = (tuning["prediction_rmse"], -gamma, a)
                if best is None or key < best[0]:
                    best = (key, blocks.copy(), candidate)
            except RuntimeError as exc:
                failures.append({"alpha_dimensionless": a, "gamma": gamma, "error": str(exc)})
    if best is None:
        raise RuntimeError(f"Every ridge fit failed in {out}")
    write_json(out / "tuning_grid.json", {"candidates": candidates, "failures": failures})
    pd.DataFrame(candidates).to_csv(out / "tuning_grid.csv", index=False)
    methods = [
        ("current", baseline, math.inf, 0.0, 0.0, baseline_diagnostics),
        ("common_free", free, 0.0, 0.0, 0.0, free_diagnostics),
        ("common_dual_ridge", best[1], best[2]["gamma"], best[2]["alpha"], best[2]["alpha_dimensionless"], best[2]),
    ]
    rows = []
    for name, blocks, gamma, alpha, a, diagnostics in methods:
        intercepts = training_intercepts(datasets["train"], blocks)
        row = {"case": job["case"], "repeat": job["repeat"], "regime": job["regime"], "root_meter_noise_relative": job["root_noise"], "method": name, "alpha": alpha, "alpha_dimensionless": a, "gamma": None if math.isinf(gamma) else gamma, "terminal_count": n}
        archive = {"R": blocks[0], "X": blocks[1], "R_true": truth[0], "X_true": truth[1], "intercepts": intercepts, "margins": margins}
        for split, data in datasets.items():
            metrics, residual, common = errors(data, blocks, intercepts, gamma)
            row.update({f"{split}_{key}": value for key, value in metrics.items()})
            archive[f"{split}_residual_before_common"] = residual
            archive[f"{split}_common"] = common
        for index, label in enumerate(("r", "x")):
            row[f"{label}_relative_error"] = relative_error(blocks[index], truth[index])
            row[f"d_{label}_relative_error"] = relative_error(impedance_distance_from_reduced_R(blocks[index]), impedance_distance_from_reduced_R(truth[index]))
            row[f"{label}_minimum_entry"] = float(blocks[index].min())
        rnj, traces, _ = rnj_metrics(blocks, terminals, net.root_bus, truth_clades, fixed_tolerances)
        row.update(rnj)
        row["true_clades"] = _serialize_family(truth_clades)
        archive.update(traces)
        if name == "common_free":
            before_gauge, _, _ = rnj_metrics(free_raw, terminals, net.root_bus, truth_clades)
            row["rnj_rx75_f1_before_gauge_fix"] = before_gauge["rnj_rx75_f1"]
            archive.update(R_solver_raw=free_raw[0], X_solver_raw=free_raw[1], gauge_shifts=shifts)
        row["solver_success"] = diagnostics["success"]
        row["solver_iterations"] = diagnostics["iterations"]
        row["solver_coordinates"] = diagnostics["coordinates"]
        np.savez_compressed(out / f"{name}.npz", **archive)
        write_json(out / f"{name}_diagnostics.json", diagnostics)
        rows.append(row)
    result = {"job": job, "rows": rows, "seconds": perf_counter()-started, "ridge_candidates": len(candidates), "ridge_failures": failures, "alpha_scale": alpha_scale, "gauge_convention": "free-only: subtract each matrix minimum, preserving the profiled fit"}
    write_json(out / "result.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/common_mode_comparison"))
    parser.add_argument("--cases", nargs="+", choices=tuple(CASE_BUILDERS), default=list(CASE_BUILDERS))
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--regimes", nargs="+", choices=("root_exact", "root_noisy"), default=["root_exact", "root_noisy"])
    args = parser.parse_args()
    if args.repeats < 1 or args.workers < 1:
        parser.error("repeats and workers must be positive")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    jobs = [dict(output=str(output), case=case, repeat=repeat, regime=regime, root_noise=0.0 if regime=="root_exact" else 0.0002, alpha_grid=list(ALPHA_GRID), gamma_grid=list(GAMMA_GRID)) for case in args.cases for repeat in range(args.repeats) for regime in args.regimes]
    config = {"cases": args.cases, "repeats": args.repeats, "regimes": args.regimes, "alpha_grid": ALPHA_GRID, "gamma_grid": GAMMA_GRID, "samples_per_split": "3 scenarios x 96", "split_replicates": "train=3*k, tune=3*k+1, test=3*k+2", "pq_noise_relative": 0.005, "terminal_voltage_noise_relative": 0.0002, "physical_root_voltage_sigma": 0.0008, "tuning_criterion": "unprofiled absolute voltage-drop RMSE with training-only intercepts", "profile_metric": "uses the evaluated Y to infer c; conditional fitting, not prediction", "alpha_scaling": "alpha = a * sum(centered_A**2)/(2*n)", "objective": "0.5*SSE + 0.5*alpha*(||R||F^2+||X||F^2) + 0.5*n*gamma*||c||2^2", "solver": "existing ordered symmetric convex QP, optional right output transform", "free_gauge": "subtract min entry separately from R and X; raw solutions also retained"}
    source_root = Path(__file__).resolve().parents[1]
    paths = [Path(__file__), *sorted((source_root/"rnj_wzzt").rglob("*.py"))]
    manifest = {str(p.relative_to(source_root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    manifest_path = output / "source_manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
        raise ValueError("Source changed since this output was created; choose a fresh --output directory")
    if not manifest_path.exists() and any(output.rglob("result.json")):
        raise ValueError("Cached results have no source manifest; choose a fresh --output directory")
    write_json(output / "config.json", config)
    write_json(manifest_path, manifest)
    completed = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(run_job, job): job for job in jobs}
        for future in as_completed(futures):
            result = future.result()
            completed.append(result)
            records = [row for item in completed for row in item["rows"]]
            pd.DataFrame(records).sort_values(["regime", "case", "repeat", "method"]).to_csv(output/"metrics.csv", index=False)
            print(json.dumps({"done": len(completed), "total": len(jobs), "case": result["job"]["case"], "repeat": result["job"]["repeat"], "regime": result["job"]["regime"], "seconds": result["seconds"], "rx75_f1": {r["method"]: r["rnj_rx75_f1"] for r in result["rows"]}, "ridge_failures": len(result["ridge_failures"])}), flush=True)
    frame = pd.read_csv(output/"metrics.csv")
    numeric = [c for c in frame.select_dtypes(include="number").columns if c != "repeat"]
    frame.groupby(["regime", "method"])[numeric].agg(["mean", "std"]).to_csv(output/"aggregate.csv")
    write_json(output/"run_summary.json", {"jobs": len(completed), "model_rows": len(frame), "ridge_candidates": sum(x["ridge_candidates"] for x in completed), "ridge_failures": sum(len(x["ridge_failures"]) for x in completed)})


if __name__ == "__main__":
    main()
