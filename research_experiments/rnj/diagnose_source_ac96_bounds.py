"""Independent fixed-singleton LP diagnostic on the saved Pengwah18 AC day.

Exactly 3 x 18 independent two-coefficient LAD regressions are solved, at
predeclared coefficient bounds 2, 4, and 32.  This is not a topology search.
All 54 fits finish before physical network truth is opened for posthoc errors.
"""
from __future__ import annotations

import os
for _thread_name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_thread_name] = "1"

import argparse
from collections import deque
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
from time import perf_counter
import warnings

import numpy as np
import pandas as pd
import scipy
from scipy.optimize import OptimizeWarning, linprog


CASE = "pengwah18"
VARIANT = "no_source"
BOUNDS = (2.0, 4.0, 32.0)
ROOT = Path(__file__).resolve().parents[2]


def plain(value):
    if isinstance(value, dict):
        return {str(key): plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return plain(value.tolist())
    if isinstance(value, np.generic):
        return plain(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path, value):
    Path(path).write_text(json.dumps(plain(value), ensure_ascii=False, indent=2,
                                   allow_nan=False) + "\n", encoding="utf-8")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def table(path):
    return pd.read_csv(path, index_col=0, float_precision="round_trip")


def independently_construct_truth(network_directory, labels):
    """Use saved physical path incidence, not a core sensitivity/scorer helper."""
    metadata = json.loads((network_directory / "metadata.json").read_text())
    buses = pd.read_csv(network_directory / "buses.csv")
    branches = pd.read_csv(network_directory / "branches.csv")
    root = int(metadata["root_bus"])
    adjacency = {int(bus): [] for bus in buses["bus_id"]}
    zbase = float(metadata["base_kv"]) ** 2 / float(metadata["base_mva"])
    impedances = []
    for edge in branches.to_dict("records"):
        if str(edge.get("is_true_closed", True)).lower() not in {"true", "1"}:
            continue
        a, b, index = int(edge["from_bus"]), int(edge["to_bus"]), len(impedances)
        adjacency[a].append((b, index))
        adjacency[b].append((a, index))
        impedances.append((float(edge["r_ohm"]) / zbase, float(edge["x_ohm"]) / zbase))
    paths, queue = {root: set()}, deque([root])
    while queue:
        node = queue.popleft()
        for neighbor, edge in adjacency[node]:
            if neighbor not in paths:
                paths[neighbor] = paths[node] | {edge}
                queue.append(neighbor)
    if len(paths) != len(adjacency) or len(impedances) != len(adjacency) - 1:
        raise ValueError("physical truth is not a connected radial network")
    incidence = np.array([[edge in paths[int(label)] for edge in range(len(impedances))]
                          for label in labels], dtype=float)
    rx = np.asarray(impedances)
    return (incidence * (2 * rx[:, 0])) @ incidence.T, (incidence * (2 * rx[:, 1])) @ incidence.T


def fit_terminal(p, q, y, bound, terminal_label):
    """LAD epigraph plus independently checked primal and marginal dual data."""
    count = len(y)
    design = np.column_stack((p, q))
    a_ub = np.block([[design, -np.eye(count)], [-design, -np.eye(count)]])
    b_ub = np.r_[y, -y]
    objective = np.r_[np.zeros(2), np.full(count, 1.0 / count)]
    lower = np.zeros(count + 2)
    upper = np.r_[bound, bound, np.full(count, np.inf)]
    record = {"terminal_label": int(terminal_label), "coefficient_bound": bound,
              "sample_count": count, "variable_count": count + 2,
              "inequality_count": count * 2, "status": None, "success": False}
    started = perf_counter()
    try:
        # SciPy forwards the native HiGHS thread option and may warn because
        # it is not listed in linprog's generic option schema. Record this.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", OptimizeWarning)
            result = linprog(objective, A_ub=a_ub, b_ub=b_ub,
                             bounds=list(zip(lower, upper)), method="highs",
                             options={"threads": 1, "presolve": True,
                                      "primal_feasibility_tolerance": 1e-9,
                                      "dual_feasibility_tolerance": 1e-9})
        record.update(status=int(result.status), success=bool(result.success), message=str(result.message),
                      iterations=int(result.nit), warnings=[str(item.message) for item in caught])
        if result.x is None:
            record.update(certified=False, failure="no_primal_solution")
            return record
        primal = np.asarray(result.x, dtype=float)
        record.update(primal_vector=primal.tolist(), a=float(primal[0]), b=float(primal[1]),
                      solver_objective=float(result.fun),
                      primal_objective=float(objective @ primal),
                      train_mae_recomputed=float(np.mean(abs(y - design @ primal[:2]))))
        ineq = np.asarray(result.ineqlin.marginals)
        low = np.asarray(result.lower.marginals)
        high = np.asarray(result.upper.marginals)
        finite_upper = np.isfinite(upper)
        # Marginals are derivatives with respect to rhs/lower/upper bounds.
        # For this minimization LP: lambda_inequality <= 0, lower >= 0,
        # upper <= 0, c = A.T lambda + lower + upper.
        dual_parts = {"inequality_rhs": float(ineq @ b_ub),
                      "finite_lower_bounds": float(low @ lower),
                      "finite_upper_bounds": float(high[finite_upper] @ upper[finite_upper])}
        dual = sum(dual_parts.values())
        slack = b_ub - a_ub @ primal
        stationarity = objective - a_ub.T @ ineq - low - high
        violation = max(float(np.max(-slack, initial=0)),
                        float(np.max(lower - primal, initial=0)),
                        float(np.max(primal[finite_upper] - upper[finite_upper], initial=0)))
        signs = max(float(np.max(ineq, initial=0)), float(np.max(-low, initial=0)),
                    float(np.max(high, initial=0)))
        complementary = max(float(np.max(abs(ineq * slack), initial=0)),
                            float(np.max(abs(low * (primal - lower)), initial=0)),
                            float(np.max(abs(high[finite_upper] * (primal[finite_upper] - upper[finite_upper])), initial=0)))
        objective_gap = float(objective @ primal - dual)
        stationarity_error = float(np.max(abs(stationarity), initial=0))
        epigraph_gap = abs(float(objective @ primal) - record["train_mae_recomputed"])
        record.update(dual_objective=dual, dual_objective_parts=dual_parts,
                      primal_minus_dual=objective_gap, absolute_gap=abs(objective_gap),
                      max_constraint_violation=violation, max_dual_sign_violation=signs,
                      max_stationarity_residual=stationarity_error,
                      max_complementarity_residual=complementary, epigraph_objective_gap=epigraph_gap,
                      inequality_marginals=ineq.tolist(), lower_marginals=low.tolist(), upper_marginals=high.tolist(),
                      certified=bool(result.success and result.status == 0 and violation <= 1e-8
                                     and signs <= 1e-8 and stationarity_error <= 1e-8
                                     and complementary <= 1e-8 and abs(objective_gap) <= 1e-9 and epigraph_gap <= 1e-9))
    except Exception as error:
        record.update(certified=False, failure=f"{type(error).__name__}: {error}")
    finally:
        record["elapsed_seconds"] = perf_counter() - started
    return record


def run(dataset, reference_directory, output_directory):
    output_directory.mkdir(parents=True, exist_ok=True)
    output_json = output_directory / "bound_lp_diagnostic.json"
    if output_json.exists() or (output_directory / "bound_lp_diagnostic_fits.json").exists():
        raise FileExistsError("diagnostic outputs already exist; preserve the authorized 54-fit run")
    condition = dataset / CASE / VARIANT
    frames = {name: table(condition / "observed" / filename) for name, filename in
              (("P", "P_terminal.csv"), ("Q", "Q_terminal.csv"), ("Y", "drop_target.csv"))}
    labels = [int(value) for value in frames["P"].columns]
    if len(labels) != 18 or any(value.shape != (96, 18) for value in frames.values()):
        raise ValueError("only saved Pengwah18 with 96 points and 18 terminals is authorized")
    if any(not value.index.equals(frames["P"].index) or list(value.columns) != list(frames["P"].columns)
           for value in frames.values()):
        raise ValueError("P/Q/Y labels and times must agree")
    source = table(condition / "source" / "amplitudes.csv")
    if np.any(source[["P_pu", "Q_pu"]].to_numpy() != 0):
        raise ValueError("diagnostic is no-source only")
    arrays = {name: value.to_numpy(dtype=float) for name, value in frames.items()}
    input_paths = [condition / "observed" / name for name in ("P_terminal.csv", "Q_terminal.csv", "drop_target.csv")]
    input_paths.append(condition / "source" / "amplitudes.csv")
    input_hashes = {path.relative_to(dataset).as_posix(): sha256(path) for path in input_paths}
    fits = []
    for bound in BOUNDS:
        for terminal_position, label in enumerate(labels):
            fit = fit_terminal(arrays["P"][:64, terminal_position], arrays["Q"][:64, terminal_position],
                               arrays["Y"][:64, terminal_position], bound, label)
            fits.append(fit)
        write_json(output_directory / "bound_lp_diagnostic_fits.json", {
            "phase": "all fits before physical truth scoring", "fits_expected": 54,
            "fits_attempted": len(fits), "truth_opened": False, "fits": fits})

    # Network truth is used only after all fitting choices and coefficients
    # have been fixed, never for selecting B or the training coefficients.
    true_r, true_x = independently_construct_truth(dataset / CASE / "network", labels)
    summaries, saved_arrays = [], {**arrays, "R_true": true_r, "X_true": true_x,
                                   "train_positions": np.arange(64), "validation_positions": np.arange(64, 96)}
    for bound in BOUNDS:
        group = [fit for fit in fits if fit["coefficient_bound"] == bound]
        summary = {"coefficient_bound": bound, "terminals_expected": 18,
                   "terminals_attempted": len(group), "terminals_certified": sum(fit["certified"] for fit in group),
                   "elapsed_seconds": sum(fit["elapsed_seconds"] for fit in group),
                   "train_mae": None, "validation_mae": None,
                   "r_relative_error": None, "x_relative_error": None}
        if all(fit.get("a") is not None and fit.get("b") is not None for fit in group):
            r, x = np.array([fit["a"] for fit in group]), np.array([fit["b"] for fit in group])
            r_matrix, x_matrix = np.diag(r), np.diag(x)
            predicted = arrays["P"] * r + arrays["Q"] * x
            summary.update(train_mae=float(np.mean(abs(arrays["Y"][:64] - predicted[:64]))),
                           validation_mae=float(np.mean(abs(arrays["Y"][64:] - predicted[64:]))),
                           r_relative_error=float(np.linalg.norm(r_matrix - true_r) / np.linalg.norm(true_r)),
                           x_relative_error=float(np.linalg.norm(x_matrix - true_x) / np.linalg.norm(true_x)),
                           r_max=float(r.max()), x_max=float(x.max()),
                           r_at_bound_count=int(np.sum(r >= bound - 1e-7)), x_at_bound_count=int(np.sum(x >= bound - 1e-7)),
                           max_absolute_dual_gap=max(fit["absolute_gap"] for fit in group),
                           max_constraint_violation=max(fit["max_constraint_violation"] for fit in group))
            saved_arrays[f"R_B{int(bound)}"], saved_arrays[f"X_B{int(bound)}"] = r_matrix, x_matrix
            saved_arrays[f"prediction_B{int(bound)}"] = predicted
        summaries.append(summary)
    comparisons = []
    b2_fits = [fit for fit in fits if fit["coefficient_bound"] == 2]
    r_b2, x_b2 = np.array([fit.get("a", np.nan) for fit in b2_fits]), np.array([fit.get("b", np.nan) for fit in b2_fits])
    for multiplier in (1, 100):
        path = reference_directory / "jobs" / f"pengwah18__no_source__WZZT__t{multiplier}" / "progress" / "000001.json"
        if not path.is_file():
            comparisons.append({"reference": str(path), "available": False})
            continue
        reference = json.loads(path.read_text())
        point = reference["path"][0]
        supports = point["support_indices"]
        if supports != [[i] for i in range(18)] or reference["r_upper_bound"] != 2 or reference["x_upper_bound"] != 2:
            raise ValueError("reference is not the bound-2 singleton initialization")
        r_error = float(np.max(abs(r_b2 - np.array(point["r_values"]))))
        x_error = float(np.max(abs(x_b2 - np.array(point["x_values"]))))
        objective_error = abs(summaries[0]["train_mae"] - point["train_mae"])
        comparisons.append({"reference": str(path), "available": True, "sha256": sha256(path),
                            "objective_absolute_difference": objective_error,
                            "r_weights_max_abs_difference": r_error, "x_weights_max_abs_difference": x_error,
                            "objective_reproduced": objective_error <= 1e-9,
                            "weights_reproduced": max(r_error, x_error) <= 1e-6})
    result = {
        "diagnostic": "Fixed singleton family only; independent terminal LAD LPs; not topology search or a full WZZT method.",
        "case": CASE, "variant": VARIANT, "training_samples": 64, "validation_samples": 32,
        "bounds_predeclared": BOUNDS, "bounds_selected_using_truth": False,
        "solver": "scipy.optimize.linprog(method='highs')", "solver_threads": 1,
        "fits_expected": 54, "fits_attempted": len(fits),
        "fits_certified": sum(fit["certified"] for fit in fits),
        "physical_truth_opened_after_all_fits": True,
        "dual_formula": "inequality_marginals @ rhs + finite_lower_marginals @ lower_bounds + finite_upper_marginals @ upper_bounds",
        "input_sha256": input_hashes, "inputs_unchanged": all(sha256(dataset / relative) == expected for relative, expected in input_hashes.items()),
        "source_sha256": sha256(Path(__file__)),
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, cwd=ROOT).strip(),
        "environment": {"python": sys.version, "numpy": np.__version__, "pandas": pd.__version__,
                        "scipy": scipy.__version__, "platform": platform.platform(),
                        "threads": {name: os.environ[name] for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")}},
        "summary": summaries, "formal_B2_initialization_comparisons": comparisons, "fits": fits,
    }
    write_json(output_json, result)
    pd.DataFrame(summaries).to_csv(output_directory / "bound_lp_diagnostic.csv", index=False, float_format="%.17g")
    scalar_rows = [{key: value for key, value in fit.items() if not isinstance(value, (list, dict))} for fit in fits]
    pd.DataFrame(scalar_rows).to_csv(output_directory / "bound_lp_diagnostic_terminals.csv", index=False, float_format="%.17g")
    np.savez_compressed(output_directory / "bound_lp_diagnostic.npz", **saved_arrays)
    print(json.dumps(plain({"fits_attempted": len(fits), "fits_certified": result["fits_certified"],
                            "inputs_unchanged": result["inputs_unchanged"], "summary": summaries,
                            "B2_comparisons": comparisons}), ensure_ascii=False))
    return 0 if result["fits_certified"] == 54 and result["inputs_unchanged"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=ROOT / "outputs/source_ac96_20261018")
    parser.add_argument("--reference-dir", type=Path, default=ROOT / "outputs/source_ac96_full_history_pengwah18_20261018")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/source_ac96_diagnosis_20261018")
    args = parser.parse_args()
    return run(args.dataset_dir.resolve(), args.reference_dir.resolve(), args.output_dir.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
