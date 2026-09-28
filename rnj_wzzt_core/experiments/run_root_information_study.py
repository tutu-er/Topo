"""Small paired root-information study; no bootstrap or downstream MILP.

Reference days share physical curves and terminal observations across exact,
noisy, and unobserved root conditions. Hyperparameters use a separate tuning
split; truth is only read for post-fit matrix and topology evaluation.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from time import perf_counter

for _option in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[_option] = "1"
sys.dont_write_bytecode = True
CORE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CORE))

import numpy as np
import pandas as pd

from run_common_mode_comparison import (
    errors, output_transform, relative_error, rnj_metrics, training_intercepts,
)
from run_scenario_benchmark import write_json
from rnj_wzzt.data.paper_style_case_bank import CASE_BUILDERS
from rnj_wzzt.estimation.constrained_least_squares import solve_symmetric_least_squares
from rnj_wzzt.estimation.multiscenario import _aligned_arrays, fit_projected_sensitivity, preprocess_scenarios
from rnj_wzzt.estimation.preprocessing import squared_voltage_drop_from_observed_root

# Keep this independent common-mode/intercept study on its declared recipe.
RECIPE = {"name": "daily_demean", "kind": "demean"}
from rnj_wzzt.models.lin_distflow import build_reduced_sensitivity_matrices, impedance_distance_from_reduced_R
from rnj_wzzt.reporting import _serialize_family, _truth_nontrivial_clades
from rnj_wzzt.scenario.simulation import _simulate_pool, _terminal_buses


ALPHA_GRID = (1e-6, 1e-3, 0.1)
GAMMA_GRID = (0.001, 0.1, 1e4)
OBSERVATIONS = ("exact", "noisy", "unobserved")
METHODS = ("current", "common_free", "common_dual_ridge", "meter_anchored_rx_ridge")


def observation_view(raw: list[dict], mode: str, noise: float, replicate: int) -> list[dict]:
    """Change root information only, after the paired 96-point meter draws."""
    result = []
    for index, scenario in enumerate(raw):
        item = dict(scenario)
        if mode == "exact":
            observed = scenario["root_voltage"].copy()
        elif mode == "noisy":
            measured_reference = scenario["root_voltage"]
            rng = np.random.default_rng(70_000_019 + 100_003 * replicate + index)
            observed = measured_reference + rng.normal(
                0.0, noise * np.abs(measured_reference), size=len(measured_reference),
            )
            observed.name = "V_root_measured"
        else:
            observed = None
        item["root_voltage"] = observed
        item["root_observation"] = mode
        item["drop_target"] = (
            1.02**2 - item["V_terminal"].pow(2) if observed is None else
            squared_voltage_drop_from_observed_root(item["V_terminal"], observed)
        )
        item["scenario_settings"] = deepcopy(scenario["scenario_settings"])
        item["scenario_settings"].update(root_observation=mode, root_meter_noise_rel=noise if mode == "noisy" else 0.0)
        result.append(item)
    return result


def selected(raw: list[dict], count: int) -> list[dict]:
    indices = np.arange(0, 96, 96 // count)
    return [{"name": s["name"], **{
        key: s[key].iloc[indices].copy() for key in ("P_terminal", "Q_terminal", "drop_target")
    }} for s in raw]


def data_arrays(raw: list[dict]) -> dict:
    designs, targets = _aligned_arrays(raw)
    return {
        "A": np.vstack(designs), "Y": np.vstack(targets),
        "A_mean": np.vstack([a.mean(axis=0) for a in designs]),
        "Y_mean": np.vstack([y.mean(axis=0) for y in targets]),
        "scenario": np.repeat(np.arange(len(raw)), [len(a) for a in designs]),
    }


def meter_gamma(raw: list[dict], count: int, relative_noise: float, mode: str) -> float:
    """Approximate homoskedastic GLS weight, using only available meter values.

    For Gaussian multiplicative magnitude error, Var(V_meas^2-V^2) is
    V^4 * (4*sigma_rel^2 + 2*sigma_rel^4). Meter values plug in for V here.
    The existing common-mode penalty is n*gamma*||c||^2/2, hence the factor n.
    """
    if mode == "exact":
        return math.inf
    if mode != "noisy":
        raise ValueError("A meter anchor requires an observed root")
    indices = np.arange(0, 96, 96 // count)
    terminal = np.vstack([np.asarray(s["V_terminal"].iloc[indices]) for s in raw])
    root = np.concatenate([np.asarray(s["root_voltage"].iloc[indices]) for s in raw])
    factor = 4 * relative_noise**2 + 2 * relative_noise**4
    terminal_variance = factor * float(np.mean(terminal**4))
    root_variance = factor * float(np.mean(root**4))
    return terminal_variance / (terminal.shape[1] * root_variance)


def invariant_matrix_error(estimate: np.ndarray, truth: np.ndarray) -> float:
    """Relative error after removing only the uniform 11' gauge direction."""
    difference = estimate - truth
    difference -= difference.mean()
    return float(np.linalg.norm(difference) / max(np.linalg.norm(truth - truth.mean()), 1e-15))


def score_data(data: dict, blocks: np.ndarray, intercepts: np.ndarray, gamma: float):
    metrics, residual, common = errors(data, blocks, intercepts, gamma)
    contrast = residual - residual.mean(axis=1, keepdims=True)
    metrics["contrast_rmse"] = float(np.sqrt(np.mean(contrast**2)))
    return metrics, residual, common, contrast


def run_job(job: dict) -> dict:
    started = perf_counter()
    out = Path(job["output"]) / job["case"] / f"repeat_{job['repeat']:02d}" / f"voltage_noise_{job['noise']:.6f}"
    out.mkdir(parents=True, exist_ok=True)
    previous = out / "result.json"
    if previous.exists():
        result = json.loads(previous.read_text(encoding="utf-8"))
        if result["job"] != job:
            raise ValueError(f"Cached configuration differs: {out}")
        return result
    all_sets, source_arrays = {}, {}
    for offset, split in enumerate(("train", "tune", "test")):
        replicate = 3 * job["repeat"] + offset
        net, raw = _simulate_pool(job["case"], 96, replicate, 3, job["pq_noise"], job["noise"],
                                  scenario_suite="reference", root_observation="exact")
        all_sets[split] = {mode: observation_view(raw, mode, job["noise"], replicate) for mode in OBSERVATIONS}
        for index, scenario in enumerate(raw):
            for key in ("P_terminal", "Q_terminal", "V_terminal", "P_true", "Q_true", "V_terminal_true", "root_voltage_true"):
                source_arrays[f"{split}_{index}_{key}"] = np.asarray(scenario[key])
            for mode in OBSERVATIONS:
                observed = all_sets[split][mode][index]["root_voltage"]
                source_arrays[f"{split}_{index}_{mode}_drop_target"] = np.asarray(all_sets[split][mode][index]["drop_target"])
                if observed is not None:
                    source_arrays[f"{split}_{index}_{mode}_root_observed"] = np.asarray(observed)
    terminals = _terminal_buses(net)
    truth = np.asarray(build_reduced_sensitivity_matrices(net, terminals, voltage_model="squared-voltage"))
    truth_clades = _truth_nontrivial_clades(net, terminals)
    source_arrays.update(R_true=truth[0], X_true=truth[1], terminals=terminals)
    np.savez_compressed(out / "observations.npz", **source_arrays)
    rows, failed_fits, attempted_fits = [], [], 0
    for count in job["samples"]:
        for mode in OBSERVATIONS:
            condition_out = out / f"N{count}" / mode
            condition_out.mkdir(parents=True, exist_ok=True)
            datasets = {
                split: data_arrays(selected(all_sets[split][mode], count) if split != "test" else all_sets[split][mode])
                for split in ("train", "tune", "test")
            }
            training = selected(all_sets["train"][mode], count)
            centered = preprocess_scenarios(training, RECIPE)
            design_blocks, target_blocks = _aligned_arrays(centered)
            design, target = np.vstack(design_blocks), np.vstack(target_blocks)
            n = target.shape[1]
            base_row = {"case": job["case"], "repeat": job["repeat"], "observation": mode,
                        "voltage_noise_relative": job["noise"], "root_meter_noise_relative": job["noise"] if mode == "noisy" else 0.0,
                        "samples_per_scenario": count, "training_rows": 3 * count,
                        "tuning_rows": 3 * count, "test_rows": 288, "terminal_count": n}
            diagnostics = {}
            attempted_fits += 1
            try:
                r, x, _, _ = fit_projected_sensitivity(centered, diagnostics=diagnostics)
            except (RuntimeError, ValueError, np.linalg.LinAlgError) as exc:
                for method in METHODS:
                    if method != "meter_anchored_rx_ridge" or mode != "unobserved":
                        rows.append({**base_row, "method": method, "status": "failed", "error": str(exc)})
                failed_fits.append({**base_row, "method": "current", "error": str(exc)})
                continue
            baseline = np.asarray([r, x])
            margins = np.asarray([diagnostics["r_margin"], diagnostics["x_margin"]])
            alpha_scale = float(np.sum(design**2) / (2 * n))
            _, _, fixed_tolerances = rnj_metrics(baseline, terminals, net.root_bus, truth_clades)
            methods = [("current", baseline, math.inf, 0.0, 0.0, diagnostics, None)]

            def fit(alpha: float, gamma: float):
                return solve_symmetric_least_squares(
                    design, target, baseline, margins, alpha, 1500,
                    output_transform=None if math.isinf(gamma) else output_transform(n, gamma),
                )

            attempted_fits += 1
            try:
                free_raw, free_diag = fit(0.0, 0.0)
                shifts = np.min(free_raw, axis=(1, 2))
                free = free_raw - shifts[:, None, None]
                np.testing.assert_allclose((design @ np.vstack(free) - target) @ output_transform(n, 0.0),
                                           (design @ np.vstack(free_raw) - target) @ output_transform(n, 0.0), atol=1e-9)
                methods.append(("common_free", free, 0.0, 0.0, 0.0, free_diag,
                                {"R_solver_raw": free_raw[0], "X_solver_raw": free_raw[1], "gauge_shifts": shifts}))
            except (RuntimeError, ValueError, np.linalg.LinAlgError) as exc:
                rows.append({**base_row, "method": "common_free", "status": "failed", "error": str(exc)})
                failed_fits.append({**base_row, "method": "common_free", "error": str(exc)})

            anchor = meter_gamma(all_sets["train"][mode], count, job["noise"], mode) if mode != "unobserved" else None
            grids = [("common_dual_ridge", list(job["gamma_grid"]))]
            if anchor is not None:
                grids.append(("meter_anchored_rx_ridge", [anchor]))
            for method, gammas in grids:
                candidates, failures, best = [], [], None
                for a in job["alpha_grid"]:
                    for gamma in gammas:
                        attempted_fits += 1
                        try:
                            blocks, diag = fit(a * alpha_scale, gamma)
                            intercepts = training_intercepts(datasets["train"], blocks)
                            tuning, _, _, _ = score_data(datasets["tune"], blocks, intercepts, gamma)
                            candidate = {"alpha_dimensionless": a, "alpha": a * alpha_scale,
                                         "gamma": None if math.isinf(gamma) else gamma, **tuning, **diag}
                            candidates.append(candidate)
                            key = (tuning["prediction_rmse"], -gamma, a)
                            if best is None or key < best[0]:
                                best = (key, blocks, gamma, a * alpha_scale, a, diag)
                        except (RuntimeError, ValueError, np.linalg.LinAlgError) as exc:
                            failures.append({"alpha_dimensionless": a, "gamma": gamma, "error": str(exc)})
                write_json(condition_out / f"{method}_tuning.json", {"selection": "tuning unprofiled prediction RMSE; no truth/test access", "candidates": candidates, "failures": failures})
                failed_fits.extend({**base_row, "method": method, **failure} for failure in failures)
                if best is None:
                    rows.append({**base_row, "method": method, "status": "failed", "error": "All tuning candidates failed"})
                else:
                    methods.append((method, best[1], best[2], best[3], best[4], best[5], None))

            for method, blocks, gamma, alpha, a, diag, extra in methods:
                intercepts = training_intercepts(datasets["train"], blocks)
                row = {**base_row, "method": method, "status": "success", "alpha": alpha,
                       "alpha_dimensionless": a, "gamma": None if math.isinf(gamma) else gamma,
                       "gamma_is_infinite": math.isinf(gamma), "meter_anchor_gamma": anchor,
                       "solver_coordinates": diag["coordinates"], "solver_iterations": diag["iterations"]}
                archive = {"R": blocks[0], "X": blocks[1], "intercepts": intercepts, "margins": margins}
                if extra is not None:
                    archive.update(extra)
                for split, data in datasets.items():
                    metrics, residual, common, contrast = score_data(data, blocks, intercepts, gamma)
                    row.update({f"{split}_{key}": value for key, value in metrics.items()})
                    archive.update({f"{split}_residual": residual, f"{split}_common_profiled": common,
                                    f"{split}_contrast": contrast})
                for index, label in enumerate(("r", "x")):
                    row[f"{label}_relative_error"] = relative_error(blocks[index], truth[index])
                    row[f"{label}_uniform_shift_invariant_relative_error"] = invariant_matrix_error(blocks[index], truth[index])
                    row[f"d_{label}_relative_error"] = relative_error(impedance_distance_from_reduced_R(blocks[index]), impedance_distance_from_reduced_R(truth[index]))
                topology, traces, _ = rnj_metrics(blocks, terminals, net.root_bus, truth_clades, fixed_tolerances)
                row.update(topology, true_clades=_serialize_family(truth_clades))
                archive.update(traces)
                np.savez_compressed(condition_out / f"{method}.npz", **archive)
                write_json(condition_out / f"{method}_diagnostics.json", diag)
                rows.append(row)
    result = {"job": job, "rows": rows, "failed_fits": failed_fits,
              "attempted_fits": attempted_fits, "seconds": perf_counter() - started}
    write_json(previous, result)
    return result


def summarize(results: list[dict], output: Path) -> None:
    frame = pd.DataFrame([row for result in results for row in result["rows"]])
    frame = frame.sort_values(["voltage_noise_relative", "observation", "case", "repeat", "samples_per_scenario", "method"])
    frame.to_csv(output / "metrics.csv", index=False)
    success = frame.loc[frame["status"].eq("success")]
    metrics = ["test_prediction_rmse", "test_profile_fit_rmse", "test_contrast_rmse", "r_relative_error", "x_relative_error",
               "r_uniform_shift_invariant_relative_error", "x_uniform_shift_invariant_relative_error",
               "d_r_relative_error", "d_x_relative_error", "rnj_r_f1", "rnj_x_f1", "rnj_rx75_f1", "rnj_rx75_exact"]
    success.groupby(["voltage_noise_relative", "observation", "samples_per_scenario", "method"])[metrics].agg(["mean", "std", "count"]).to_csv(output / "aggregate.csv")
    pairs = []
    keys = ["case", "repeat", "voltage_noise_relative", "observation", "samples_per_scenario"]
    for _, group in success.groupby(keys):
        baseline = group.loc[group["method"].eq("current")]
        if baseline.empty:
            continue
        baseline = baseline.iloc[0]
        for _, row in group.loc[~group["method"].eq("current")].iterrows():
            pair = {key: row[key] for key in keys}
            pair["method"] = row["method"]
            for metric in metrics:
                pair[f"{metric}_difference"] = row[metric] - baseline[metric]
            pair["test_prediction_rmse_relative_change"] = row["test_prediction_rmse"] / baseline["test_prediction_rmse"] - 1.0
            pairs.append(pair)
    pd.DataFrame(pairs).to_csv(output / "paired_changes.csv", index=False)
    write_json(output / "run_summary.json", {"jobs": len(results), "model_rows": len(frame),
        "failed_models": int((frame["status"] != "success").sum()), "attempted_fits": sum(r["attempted_fits"] for r in results),
        "failed_fits": sum(len(r["failed_fits"]) for r in results), "worker_seconds": sum(r["seconds"] for r in results)})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=CORE.parent / "artifacts/root_information_study_20260908/root_model")
    parser.add_argument("--cases", nargs="+", choices=tuple(CASE_BUILDERS), default=list(CASE_BUILDERS))
    parser.add_argument("--samples", nargs="+", type=int, default=[16, 32])
    parser.add_argument("--noise-levels", nargs="+", type=float, default=[0.0002, 0.001])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--pq-noise", type=float, default=0.005)
    args = parser.parse_args()
    if args.repeats < 1 or args.workers < 1 or any(n < 4 or 96 % n for n in args.samples):
        parser.error("repeats/workers must be positive; samples must be >=4 and divide 96")
    if any(not np.isfinite(v) or v <= 0.0 for v in args.noise_levels) or not np.isfinite(args.pq_noise) or args.pq_noise < 0.0:
        parser.error("voltage noise levels must be finite and positive; P/Q noise must be finite and nonnegative")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    config = {**vars(args), "output": str(output), "suite": "reference", "observations": OBSERVATIONS,
        "alpha_grid": ALPHA_GRID, "gamma_grid": GAMMA_GRID, "split_replicates": "train=3*k; tune=3*k+1; test=3*k+2",
        "sample_budget": "train=3*N; extra tuning=3*N; independent test=3*96 per condition",
        "noise": "P/Q fixed relative noise; terminal and noisy-root magnitude errors use the chosen voltage level",
        "tuning": "unprofiled tuning prediction RMSE, training-only intercepts, no test/truth selection",
        "anchor": "gamma=Var(terminal squared-voltage error)/(n*Var(root squared-voltage error)); exact gamma=infinity",
        "objective": "0.5*SSE + 0.5*alpha*(||R||F^2+||X||F^2) + 0.5*n*gamma*||c||2^2",
        "common_free_gauge": "subtract each matrix minimum; raw representative is archived",
        "profile_and_contrast": "conditional voltage fitting at evaluation time; not future common-mode prediction",
        "truth_access": "matrix/distance/clade scoring only; not used in fitting or parameter selection"}
    files = [Path(__file__), CORE / "experiments/run_common_mode_comparison.py", CORE / "experiments/run_scenario_benchmark.py", *sorted((CORE / "rnj_wzzt").rglob("*.py"))]
    manifest = {str(p.relative_to(CORE)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    for filename, value in (("config.json", config), ("source_manifest.json", manifest)):
        destination = output / filename
        # JSON normalization turns the tuples in the declared configuration into lists.
        expected = json.loads(json.dumps(value))
        if destination.exists() and json.loads(destination.read_text(encoding="utf-8")) != expected:
            raise ValueError(f"{filename} changed; choose a fresh output directory")
        write_json(destination, value)
    jobs = [{"output": str(output), "case": case, "repeat": repeat, "noise": noise, "samples": args.samples,
             "pq_noise": args.pq_noise, "alpha_grid": list(ALPHA_GRID), "gamma_grid": list(GAMMA_GRID)}
            for case in args.cases for repeat in range(args.repeats) for noise in args.noise_levels]
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for future in as_completed([executor.submit(run_job, job) for job in jobs]):
            result = future.result()
            results.append(result)
            summarize(results, output)
            print(json.dumps({"done": len(results), "total": len(jobs), "case": result["job"]["case"],
                              "repeat": result["job"]["repeat"], "noise": result["job"]["noise"],
                              "seconds": result["seconds"], "failed_fits": len(result["failed_fits"])}), flush=True)


if __name__ == "__main__":
    main()
