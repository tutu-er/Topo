"""Isolated negative off-diagonal / log-det experiment; never patches core code.

Run with an existing environment containing CVXPY + CLARABEL and core dependencies:
  python -B experiments/test_negative_offdiag_logdet.py --self-test
  python -B experiments/test_negative_offdiag_logdet.py --repeats 3 --workers 3

Physical matrices remain symmetric, nonnegative and ordered. Thus -|M_ij|
is exactly -M_ij on the feasible domain, and the objective remains convex.
This rewards off-diagonal magnitude; it is NOT a sparsity penalty.
"""
from __future__ import annotations

import os
for _key in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[_key] = "1"

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter
import unittest

sys.dont_write_bytecode = True
CORE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CORE))
import cvxpy as cp
import numpy as np
import pandas as pd

from rnj_wzzt.data.paper_style_case_bank import CASE_BUILDERS
from rnj_wzzt.estimation.multiscenario import (
    _aligned_arrays, fit_projected_sensitivity, preprocess_scenarios,
)
from rnj_wzzt.estimation.preprocessing import RECIPE
from rnj_wzzt.estimation.laminar_l1_milp import fit_laminar_l1_sensitivity
from rnj_wzzt.graph.rooted_hierarchy import rooted_clades
from rnj_wzzt.graph.rooted_neighbor_joining import rooted_neighbor_joining
from rnj_wzzt.graph.sensitivity_geometry import sensitivity_geometry
from rnj_wzzt.models.lin_distflow import build_reduced_sensitivity_matrices
from rnj_wzzt.pipeline import _rnj_reduced_candidate_pool, _expand_pseudo_result_clades
from rnj_wzzt.reporting import _truth_nontrivial_clades
from rnj_wzzt.scenario.simulation import _simulate_pool, _terminal_buses

ETA_GRID = (1e-3, 1e-2, 3e-2, 1e-1)
MU_GRID = (1e-4, 1e-3, 1e-2)
SPD_FLOOR = 1e-6
METHODS = ("current", "spd_only", "negative_only", "logdet_only", "combined")


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean(payload), indent=2, ensure_ascii=False,
                               allow_nan=False), encoding="utf-8")


def fingerprints():
    return {str(p.relative_to(CORE)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((CORE / "rnj_wzzt").rglob("*.py"))}


def family_score(predicted, truth):
    hit = len(predicted & truth)
    precision = hit / len(predicted) if predicted else float(not truth)
    recall = hit / len(truth) if truth else 1.0
    return {"f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            "precision": precision, "recall": recall, "exact": int(predicted == truth)}


def centered(scenarios):
    designs, targets = _aligned_arrays(scenarios)
    return (np.vstack([a - a.mean(axis=0) for a in designs]),
            np.vstack([y - y.mean(axis=0) for y in targets]))


def intercepts(scenarios, blocks):
    return np.vstack([s["drop_target"].mean().to_numpy()
                      - s["P_terminal"].mean().to_numpy() @ blocks[0]
                      - s["Q_terminal"].mean().to_numpy() @ blocks[1] for s in scenarios])


def prediction(scenarios, blocks, fitted_intercepts):
    residual = np.vstack([s["P_terminal"].to_numpy() @ blocks[0]
                          + s["Q_terminal"].to_numpy() @ blocks[1]
                          + fitted_intercepts[i] - s["drop_target"].to_numpy()
                          for i, s in enumerate(scenarios)])
    return {"rmse": float(np.sqrt(np.mean(residual**2))),
            "mae": float(np.mean(abs(residual)))}


class RegularizedRegression:
    """CVXPY model in fixed, training-only dimensionless coordinates.

    J = SSE/SSE0 - eta/2 sum_M sum_{i<j}(M_ij)/o_M
                   - mu/(2n) sum_M logdet(M/s_M).
    o_M = baseline off-diagonal sum (fallback k*s_M if numerically zero).
    s_M = baseline positive-diagonal median. Both fixed across the whole grid.
    The omitted QR residual is constant; actual SSE is recomputed for reporting.
    """

    def __init__(self, design, target, baseline, margins, *, spd=False,
                 logdet=False, sse_scale=None):
        self.a, self.y = np.asarray(design), np.asarray(target)
        self.baseline = np.asarray(baseline)
        self.n = target.shape[1]
        self.off_indices = np.triu_indices(self.n, 1)
        self.scales = np.array([max(float(np.median(np.diag(m)[np.diag(m) > 0])), 1e-12)
                                if np.any(np.diag(m) > 0) else 1.0 for m in baseline])
        sums = np.array([m[self.off_indices].sum() for m in baseline])
        fallback = max(len(self.off_indices[0]), 1) * self.scales
        self.off_scales = np.where(sums > 1e-10 * fallback, sums, fallback)
        original_sse = float(np.sum((design @ np.vstack(baseline) - target)**2))
        self.sse_scale = max(original_sse, 1e-12 * float(np.sum(target**2)), 1e-30)
        if sse_scale is not None:
            self.sse_scale = float(sse_scale)
        self.margins = np.asarray(margins)
        self.spd, self.logdet = bool(spd or logdet), logdet
        self.eta, self.mu = cp.Parameter(nonneg=True), cp.Parameter(nonneg=True)
        self.variables = [cp.Variable((self.n, self.n), symmetric=True) for _ in range(2)]
        physical = [m * scale for m, scale in zip(self.variables, self.scales)]
        q, r = np.linalg.qr(design, mode="reduced")
        residual = (r @ cp.vstack(physical) - q.T @ target) / np.sqrt(self.sse_scale)
        objective = cp.sum_squares(residual)
        upper = np.triu(np.ones((self.n, self.n)), 1)
        off_reward = sum(cp.sum(cp.multiply(upper, m)) / denom
                         for m, denom in zip(physical, self.off_scales)) / 2
        objective -= self.eta * off_reward
        if logdet:
            objective -= self.mu * sum(cp.log_det(m) for m in self.variables) / (2 * self.n)
        constraints = []
        for m, margin, scale in zip(self.variables, margins, self.scales):
            constraints.extend([m >= 0,
                cp.reshape(cp.diag(m), (self.n, 1), order="C") - m
                >= margin / scale * (np.ones((self.n, self.n)) - np.eye(self.n))])
            if self.spd:
                constraints.append(m >> SPD_FLOOR * np.eye(self.n))
        self.problem = cp.Problem(cp.Minimize(objective), constraints)
        if not self.problem.is_dcp():
            raise AssertionError("The nonnegative-domain objective must be convex")

    def solve(self, eta, mu):
        if eta < 0 or mu < 0 or (mu > 0 and not self.logdet):
            raise ValueError("invalid regularization coefficients")
        self.eta.value, self.mu.value = float(eta), float(mu)
        started = perf_counter()
        self.problem.solve(solver="CLARABEL", max_iter=200,
                           tol_gap_abs=1e-8, tol_gap_rel=1e-8, tol_feas=1e-8)
        if self.problem.status != cp.OPTIMAL:
            raise RuntimeError(f"CLARABEL returned {self.problem.status}")
        blocks = np.array([m.value * scale for m, scale in zip(self.variables, self.scales)])
        if not np.all(np.isfinite(blocks)):
            raise RuntimeError("nonfinite regression solution")
        normalized = blocks / self.scales[:, None, None]
        violation = max(0.0, -float(normalized.min()))
        for m, margin, scale in zip(normalized, self.margins, self.scales):
            order = np.diag(m)[:, None] - m - margin / scale * (1 - np.eye(self.n))
            violation = max(violation, -float(order.min()))
            if self.spd:
                violation = max(violation, SPD_FLOOR - float(np.linalg.eigvalsh(m)[0]))
        if violation > 2e-7:
            raise RuntimeError(f"scaled constraint violation {violation}")
        sse = float(np.sum((self.a @ np.vstack(blocks) - self.y)**2))
        reward = float(sum(m[self.off_indices].sum() / d
                           for m, d in zip(blocks, self.off_scales)) / 2)
        logvolume = (float(sum(np.linalg.slogdet(m)[1] for m in normalized) / (2 * self.n))
                     if self.spd else None)
        return blocks, {"status": self.problem.status, "iterations": self.problem.solver_stats.num_iters,
            "seconds": perf_counter() - started, "scaled_constraint_violation": violation,
            "sse": sse, "sse_ratio": sse / self.sse_scale, "normalized_off_reward": reward,
            "normalized_logvolume": logvolume, "objective": sse / self.sse_scale - eta * reward
            - (mu * logvolume if self.logdet else 0), "eta": eta, "mu": mu,
            "physical_negative_l1_coefficients": eta * self.sse_scale / (2 * self.off_scales),
            "physical_logdet_coefficient": mu * self.sse_scale / (2 * self.n)}


def regression_metrics(blocks, baseline, truth, sets, terminals, root, truth_clades, fixed_tau=None):
    b = intercepts(sets["train"], blocks)
    row = {}
    for split, scenarios in sets.items():
        row.update({f"{split}_{k}": v for k, v in prediction(scenarios, blocks, b).items()})
    off = np.triu_indices(len(terminals), 1)
    for name, m, base, true in zip(("r", "x"), blocks, baseline, truth):
        eigen = np.linalg.eigvalsh(m)
        row.update({f"{name}_relative_error": float(np.linalg.norm(m - true) / np.linalg.norm(true)),
                    f"{name}_norm_ratio": float(np.linalg.norm(m) / max(np.linalg.norm(base), 1e-15)),
                    f"{name}_min_eigenvalue": float(eigen[0]),
                    f"{name}_off_sum": float(m[off].sum()),
                    f"{name}_off_near_zero_fraction": float(np.mean(abs(m[off]) <= 1e-4 * max(np.median(np.diag(base)), 1e-12)))})
    clades_rx = set()
    for mode, label in (("R", "r"), ("X", "x"), ("RX_75R_25X", "rx75")):
        geometry = sensitivity_geometry(*blocks, mode)
        tau = 0.16 * max(float(np.median(geometry.root_depths)), 1e-12)
        tree = rooted_neighbor_joining(geometry.shared_paths, geometry.root_depths, terminals, root,
                                      group_tolerance=tau)
        clades = rooted_clades(tree.edges, root, terminals)
        row.update({f"rnj_{label}_{k}": v for k, v in family_score(clades, truth_clades).items()})
        if label == "rx75":
            clades_rx = clades
            row["rnj_tau"] = tau
            if fixed_tau is not None:
                tree = rooted_neighbor_joining(geometry.shared_paths, geometry.root_depths, terminals, root,
                                              group_tolerance=fixed_tau)
                row["rnj_fixed_tau_f1"] = family_score(rooted_clades(tree.edges, root, terminals), truth_clades)["f1"]
    return row, b, clades_rx


def admissible(row):
    """Predeclared training-only bounds; never use truth or test observations."""
    return (row["sse_ratio"] <= 1.10 and row["r_norm_ratio"] <= 1.25
            and row["x_norm_ratio"] <= 1.25 and row["normalized_off_reward"] <= 1.25)


def run_job(job):
    out = Path(job["output"]) / f"{job['case']}_n{job['samples']}_repeat{job['repeat']}"
    out.mkdir(parents=True, exist_ok=True)
    result_path = out / "result.json"
    if result_path.exists():
        saved = json.loads(result_path.read_text(encoding="utf-8"))
        if saved["job"] != job:
            raise ValueError(f"configuration mismatch: {out}")
        return saved
    started = perf_counter()
    sets, inputs = {}, {}
    for offset, split in enumerate(("train", "tune", "path_validation", "test")):
        net, raw = _simulate_pool(job["case"], 96 if split == "test" else job["samples"],
            4 * job["repeat"] + offset, 3, 0.005, 0.0002, scenario_suite="reference")
        sets[split] = [{**s, "name": f"scenario_{i}"} for i, s in enumerate(raw)]
        for i, s in enumerate(raw):
            for field in ("P_terminal", "Q_terminal", "drop_target", "root_voltage", "root_voltage_true"):
                inputs[f"{split}_{i}_{field}"] = s[field].to_numpy()
    terminals, root = _terminal_buses(net), int(net.root_bus)
    truth = np.array(build_reduced_sensitivity_matrices(net, terminals, voltage_model="squared-voltage"))
    truth_clades = _truth_nontrivial_clades(net, terminals)
    inputs.update(R_true=truth[0], X_true=truth[1], terminals=np.array(terminals))
    np.savez_compressed(out / "inputs.npz", **inputs)
    training = preprocess_scenarios(sets["train"], RECIPE)
    a, y = centered(training)
    baseline_diag = {}
    st = perf_counter()
    baseline = np.array(fit_projected_sensitivity(training, diagnostics=baseline_diag)[:2])
    baseline_seconds = perf_counter() - st
    margins = np.array([baseline_diag["r_margin"], baseline_diag["x_margin"]])
    models = {"negative_only": RegularizedRegression(a, y, baseline, margins),
              "spd_only": RegularizedRegression(a, y, baseline, margins, spd=True),
              "logdet": RegularizedRegression(a, y, baseline, margins, logdet=True)}
    base, b, clades = regression_metrics(baseline, baseline, truth, sets, terminals, root, truth_clades)
    base.update(method="current", eta=0.0, mu=0.0, sse_ratio=1.0, normalized_off_reward=1.0,
                eligible=True, seconds=baseline_seconds, status="optimal")
    selected = {"current": (base, baseline, b, clades)}
    all_rows, failures, archive = [base], [], {"current_R": baseline[0], "current_X": baseline[1]}
    candidates = [("spd_only", 0.0, 0.0)]
    candidates += [("negative_only", eta, 0.0) for eta in ETA_GRID]
    candidates += [("logdet_only", 0.0, mu) for mu in MU_GRID]
    candidates += [("combined", eta, mu) for eta in ETA_GRID for mu in MU_GRID]
    for idx, (method, eta, mu) in enumerate(candidates):
        model = models[method if method in models else "logdet"]
        try:
            blocks, diagnostics = model.solve(eta, mu)
            row, b, clades = regression_metrics(blocks, baseline, truth, sets, terminals, root,
                                               truth_clades, fixed_tau=base["rnj_tau"])
            row.update(diagnostics, method=method, candidate_index=idx)
            row["eligible"] = admissible(row)
            all_rows.append(row)
            archive[f"candidate_{idx}_R"], archive[f"candidate_{idx}_X"] = blocks
            if row["eligible"] or method == "spd_only":
                previous = selected.get(method)
                key = (row["tune_rmse"], eta + mu, eta, mu)
                if previous is None or key < (previous[0]["tune_rmse"], previous[0]["eta"] + previous[0]["mu"], previous[0]["eta"], previous[0]["mu"]):
                    selected[method] = (row, blocks, b, clades)
        except Exception as exc:
            failures.append({"method": method, "eta": eta, "mu": mu, "error": repr(exc)})
    write_json(out / "grid.json", {"rows": all_rows, "failures": failures,
        "scales": models["negative_only"].scales, "off_scales": models["negative_only"].off_scales,
        "sse_scale": models["negative_only"].sse_scale, "margins": margins,
        "centered_design_rank": np.linalg.matrix_rank(a), "centered_design_columns": a.shape[1]})
    np.savez_compressed(out / "grid_matrices.npz", **archive)
    rows = []
    for method in METHODS:
        if method not in selected:
            rows.append({"method": method, "status": "no_admissible_candidate"})
            continue
        row, blocks, b, clades = selected[method]
        row = row.copy()
        np.savez_compressed(out / f"{method}.npz", R=blocks[0], X=blocks[1], intercepts=b)
        row["rnj_clades"] = sorted([sorted(c) for c in clades], key=lambda c: (len(c), c))
        if job["milp"]:
            identity = {t: frozenset({t}) for t in terminals}
            pool = _rnj_reduced_candidate_pool(clades, terminals, identity, include_one_edit=False)
            pool_clades = {frozenset(terminals[i] for i in s) for s in pool}
            row["candidate_count"] = len(pool)
            row["candidate_truth_recall"] = len(pool_clades & truth_clades) / max(len(truth_clades), 1)
            st = perf_counter()
            try:
                # Direct core call, no monkeypatch, no altered solver, no frozen RNJ blocks.
                # Existing structurally known singleton leaf atoms remain fixed.
                fitted = fit_laminar_l1_sensitivity(training,
                    validation_scenarios=preprocess_scenarios(sets["path_validation"], RECIPE),
                    initial_supports=[(i,) for i in range(len(terminals))], candidate_supports=pool,
                    r_upper_bound=2.0, x_upper_bound=2.0, time_limit=job["milp_seconds"])
                recovered = _expand_pseudo_result_clades(fitted, identity, len(terminals))
                row.update({f"milp_{k}": v for k, v in family_score(recovered, truth_clades).items()})
                row.update(milp_seconds=perf_counter() - st, milp_stop=fitted.stop_reason,
                           milp_path_length=len(fitted.path), milp_selected_path=fitted.selected_path_index)
                # Reconstruct only the intercept on raw training means for raw test predictions.
                mb = np.array([fitted.r_matrix, fitted.x_matrix])
                raw_b = intercepts(sets["train"], mb) + fitted.intercepts
                row.update({f"milp_test_{k}": v for k, v in prediction(sets["test"], mb, raw_b).items()})
                write_json(out / f"{method}_milp.json", {"summary": fitted.summary(),
                    "clades": sorted([sorted(c) for c in recovered], key=lambda c: (len(c), c)),
                    "attempts": [e.diagnostics.to_dict() for e in fitted.attempted_extensions],
                    "path": [{"iteration": p.iteration, "train_mae": p.train_mae,
                              "validation_mae": p.validation_mae, "validation_se": p.validation_se,
                              "solver": p.solver.to_dict()} for p in fitted.path]})
                np.savez_compressed(out / f"{method}_milp.npz", R=mb[0], X=mb[1], intercepts=raw_b)
            except Exception as exc:
                row.update(milp_error=repr(exc), milp_seconds=perf_counter() - st)
        rows.append(row)
    result = {"job": job, "rows": rows, "failures": failures, "seconds": perf_counter() - started,
              "truth_clades": sorted([sorted(c) for c in truth_clades], key=lambda c: (len(c), c))}
    write_json(result_path, result)
    return result


class AnalyticalTests(unittest.TestCase):
    def test_negative_offdiagonal_has_positive_closed_form_shift(self):
        baseline = np.array([[[2.0, .2], [.2, 2.0]], [[1.0, .1], [.1, 1.0]]])
        model = RegularizedRegression(np.eye(4), np.vstack(baseline), baseline, np.zeros(2), sse_scale=1)
        got, _ = model.solve(.02, 0)
        expected = baseline.copy()
        for j, denom in enumerate((.2, .1)):
            expected[j, 0, 1] = expected[j, 1, 0] = baseline[j, 0, 1] + .02 / (8 * denom)
        np.testing.assert_allclose(got, expected, atol=2e-5)

    def test_logdet_matches_diagonal_analytic_solution(self):
        baseline = np.array([2 * np.eye(2), np.eye(2)])
        model = RegularizedRegression(np.eye(4), np.vstack(baseline), baseline, np.zeros(2),
                                      logdet=True, sse_scale=1)
        got, _ = model.solve(0, .4)
        for m, value in zip(got, (2., 1.)):
            expected = .5 * (value + np.sqrt(value**2 + .4 / 2))
            np.testing.assert_allclose(np.diag(m), expected, atol=3e-5)
            self.assertLess(abs(m[0, 1]), 5e-5)

    def test_zero_regularization_matches_core_qp(self):
        rng = np.random.default_rng(510)
        p, q = rng.normal(size=(2, 90, 3))
        y = p @ (np.eye(3) + .2) + q @ (.5 * np.eye(3) + .1) + rng.normal(0, .01, (90, 3))
        scenario = {"name": "s", "P_terminal": pd.DataFrame(p), "Q_terminal": pd.DataFrame(q),
                    "drop_target": pd.DataFrame(y)}
        diag = {}
        baseline = np.array(fit_projected_sensitivity([scenario], diagnostics=diag)[:2])
        a, y = centered([scenario])
        model = RegularizedRegression(a, y, baseline, [diag["r_margin"], diag["x_margin"]])
        got, _ = model.solve(0, 0)
        np.testing.assert_allclose(got, baseline, atol=2e-6)

    def test_coefficient_guard_uses_no_truth_or_test_fields(self):
        row = {"sse_ratio": 1.02, "r_norm_ratio": 1.10, "x_norm_ratio": 1.05,
               "normalized_off_reward": 1.20}
        self.assertTrue(admissible(row))
        self.assertFalse(admissible({**row, "r_norm_ratio": 1.26}))
        self.assertFalse(admissible({**row, "sse_ratio": 1.11}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=CORE / "outputs/negative_offdiag_logdet_20260911")
    parser.add_argument("--cases", nargs="+", choices=tuple(CASE_BUILDERS), default=list(CASE_BUILDERS))
    parser.add_argument("--samples", nargs="+", type=int, default=[16, 96])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--milp-seconds", type=float, default=5.0)
    parser.add_argument("--skip-milp", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(AnalyticalTests)
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        raise SystemExit(0 if result.wasSuccessful() else 1)
    before = fingerprints()
    output = args.output.resolve()
    protocol = {"cases": args.cases, "samples": args.samples, "repeats": args.repeats,
        "eta_grid": ETA_GRID, "mu_grid": MU_GRID, "spd_floor_normalized": SPD_FLOOR,
        "objective": "SSE/SSE0 - eta/2 sum_M offsum(M)/offsum(M0) - mu/(2n) sum_M logdet(M/s_M)",
        "coefficient_guard": "SSE/SSE0<=1.10; each matrix Frobenius ratio<=1.25; normalized off reward<=1.25",
        "tuning": "Minimum tune raw prediction RMSE among eligible candidates within each ablation; no truth/test use",
        "splits": "Per repeat: independent replicates 4r=train,4r+1=tune,4r+2=path validation,4r+3=test; test has 96 points/scenario",
        "downstream": "Direct unchanged finite-pool L1-MILP; RX75 RNJ candidates; singleton initial supports; no bootstrap, freezing or contraction",
        "milp_time_limit_per_extension": args.milp_seconds, "milp": not args.skip_milp,
        "core_before": before, "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "python": sys.executable, "cvxpy": cp.__version__, "numpy": np.__version__}
    pp = output / "protocol.json"
    if pp.exists() and json.loads(pp.read_text(encoding="utf-8")) != clean(protocol):
        raise ValueError("Output protocol differs; choose a new output directory")
    write_json(pp, protocol)
    jobs = [{"output": str(output), "case": case, "samples": samples, "repeat": repeat,
             "milp": not args.skip_milp, "milp_seconds": args.milp_seconds}
            for case in args.cases for samples in args.samples for repeat in range(args.repeats)]
    results, errors = [], []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        pending = {pool.submit(run_job, job): job for job in jobs}
        for future in as_completed(pending):
            job = pending[future]
            try:
                result = future.result()
                results.append(result)
                print(json.dumps({"done": len(results), "total": len(jobs), "case": job["case"],
                    "samples": job["samples"], "repeat": job["repeat"], "seconds": result["seconds"],
                    "fit_failures": len(result["failures"])}, ensure_ascii=False), flush=True)
            except Exception as exc:
                errors.append({"job": job, "error": repr(exc)})
                print(json.dumps(errors[-1]), flush=True)
            flat = [{**{k: result["job"][k] for k in ("case", "samples", "repeat")}, **row}
                    for result in results for row in result["rows"]]
            if flat:
                pd.DataFrame(flat).to_csv(output / "selected_results.csv", index=False)
    after = fingerprints()
    write_json(output / "audit.json", {"core_unchanged": before == after,
        "core_after": after, "completed": len(results), "expected": len(jobs), "job_errors": errors,
        "solver_failures": sum(len(r["failures"]) for r in results)})
    if before != after or errors:
        raise RuntimeError("See audit.json for unchanged-code check or job errors")
    print(f"Completed {len(results)} conditions; core_unchanged={before == after}", flush=True)


if __name__ == "__main__":
    main()
