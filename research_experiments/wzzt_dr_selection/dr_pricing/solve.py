"""Solve or replay one portable DR pricing case using NumPy and SciPy only.

Run ``python solve.py --case CASE_FOLDER --out PRIVATE_OUTPUT``.
Use ``--quote quote.npz --check-only`` to verify a frozen schedule without
optimization. Independent physical evaluation belongs to the caller.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import time
import traceback
import zipfile

for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(name, "1")

import numpy as np
import scipy
from scipy.optimize import minimize

if __package__:
    from .problem import load_problem
else:
    from problem import load_problem


def plain(value):
    if isinstance(value, dict):
        return {str(key): plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(plain(value), ensure_ascii=False, indent=2,
                                    allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_arrays(path, **arrays):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def reference(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def optimize(problem, out, initial=None, *, maxiter=60, seconds=300.,
             restore_evaluations=10, restore_seconds=30., ftol=1e-8):
    """Run scaled-fee SLSQP from a possibly infeasible request, then restore.

    The probability constraint is never relaxed. All distinct evaluated request
    vectors are retained in checkpoint.npz and their scalar records in trace.json.
    A solver failure is recorded separately from constraint qualification.
    """
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if (maxiter <= 0 or seconds <= 0 or restore_evaluations < 0
            or restore_seconds <= 0 or ftol <= 0
            or not np.isfinite([seconds, restore_seconds, ftol]).all()):
        raise ValueError("Positive finite numerical limits are required")
    target = float(problem.target)
    records, points, statuses = [], [], []
    started = time.perf_counter()
    stage, deadline = "start", float("inf")

    def observe(values, label):
        if time.perf_counter() > deadline:
            raise TimeoutError(stage + " wall limit")
        point = problem.evaluate(values)
        if not points or not np.array_equal(point["values"], points[-1]["values"]):
            row = dict(index=len(records), stage=stage, label=label,
                       probability=point["probability"], fee=point["fee"],
                       elapsed=time.perf_counter()-started,
                       evaluation_seconds=point.get("seconds"),
                       moments=point.get("moments"),
                       analytic_tail_bound=point.get("analytic_tail_bound"))
            records.append(row)
            points.append(point)
            write_json(out/"trace.json", records)
            write_arrays(out/"checkpoint.npz",
                         values=np.stack([p["values"] for p in points]),
                         probability=np.array([p["probability"] for p in points]),
                         fee=np.array([p["fee"] for p in points]))
        return point

    zero = observe(problem.zero, "zero")
    warm_values = problem.warm if initial is None else np.asarray(initial, float)
    if (warm_values.shape != problem.zero.shape or not np.isfinite(warm_values).all()
            or np.any(warm_values < problem.lower) or np.any(warm_values > problem.upper)):
        raise ValueError("Warm request violates the declared active request domain")
    warm = observe(warm_values, "warm")
    cost_scale = float(warm["fee"]) if warm["fee"] > 0 else 1.
    if not np.isfinite(cost_scale):
        raise ValueError("The fixed objective scale must be finite and positive")

    def feasible():
        return [point for point in points if point["probability"] >= target]

    qualified = feasible()
    # Since costs are nonnegative, a feasible zero fee is already sufficient.
    if not any(point["fee"] == 0 for point in qualified):
        start = min(qualified, key=lambda point:point["fee"]) if qualified else warm
        stage = "cost"
        began = time.perf_counter()
        deadline = began + seconds
        iterations = None
        try:
            result = minimize(
                lambda x:observe(x, "objective")["fee"] / cost_scale,
                start["values"],
                jac=lambda x:observe(x, "objective")["fee_gradient"] / cost_scale,
                bounds=problem.bounds, method="SLSQP",
                constraints=[dict(type="ineq",
                    fun=lambda x:observe(x, "constraint")["probability"]-target,
                    jac=lambda x:observe(x, "constraint")["gradient"])],
                options=dict(maxiter=maxiter, ftol=ftol))
            iterations = int(result.nit)
            observe(result.x, "solver_return")
            status, success = str(result.message), bool(result.success)
        except Exception as exc:
            status, success = repr(exc), False
            write_json(out/"cost_failure.json",
                       dict(error=status, traceback=traceback.format_exc()))
        statuses.append(dict(stage=stage, status=status, solver_success=success,
                             iterations=iterations, seconds=time.perf_counter()-began))

    if not feasible() and restore_evaluations:
        stage = "restore"
        began = time.perf_counter()
        deadline = began + restore_seconds
        current = max(points, key=lambda point:point["probability"])
        scale, accepted, count = 1., 0, 0
        status = "evaluation_limit"
        try:
            while count < restore_evaluations:
                if current["probability"] >= target:
                    status = "target_reached"
                    break
                values = current["values"]
                direction = current["gradient"].copy()
                direction[(values <= problem.lower+1e-8) & (direction < 0)] = 0.
                direction[(values >= problem.upper-1e-8) & (direction > 0)] = 0.
                norm_squared = float(direction@direction)
                if norm_squared <= 1e-24:
                    status = "zero_projected_gradient"
                    break
                step = scale*(1.2*(target-current["probability"])+1e-6)/norm_squared
                trial = np.clip(values+step*direction, problem.lower, problem.upper)
                if np.array_equal(trial, values):
                    status = "zero_projected_step"
                    break
                candidate = observe(trial, "probability_gradient_trial")
                count += 1
                if candidate["probability"] > current["probability"]+1e-12:
                    current, scale = candidate, 1.
                    accepted += 1
                else:
                    scale *= .5
            if current["probability"] >= target:
                status = "target_reached"
        except Exception as exc:
            status = repr(exc)
            write_json(out/"restore_failure.json",
                       dict(error=status, traceback=traceback.format_exc()))
        statuses.append(dict(stage=stage, status=status, accepted_steps=accepted,
                             trial_evaluations=count, seconds=time.perf_counter()-began))

    qualified = feasible()
    best = (min(qualified, key=lambda point:point["fee"]) if qualified
            else max(points, key=lambda point:point["probability"]))
    return best, dict(statuses=statuses, qualified_by_search=bool(qualified),
        target=target, fixed_cost_scale=cost_scale, evaluations=len(records),
        feasible_evaluations=len(qualified), seconds=time.perf_counter()-started,
        optimization_performed=True, global_optimal=False)


def freeze(problem, point, search, out, request_reference):
    """Recheck the saved point and export numeric arrays without pickle."""
    values = point["values"]
    nominal = problem.nominal(values)
    audit = problem.audit_probability(values)
    device = problem.audit_delivery(nominal["mean_u"])
    reward = problem.unit_reward(nominal["mean_u"])
    qualified = bool(audit["numerical_pass"] and device["passed"]
                     and audit["probability"] >= problem.target)
    outcome = ("device_unresolved" if not device["passed"] else
               "numerically_unresolved" if not audit["numerical_pass"] else
               "qualified" if qualified else "target_not_met")
    out = Path(out)
    write_arrays(out/"quote.npz", x=nominal["x"], mean_u=nominal["mean_u"],
                 unit_reward=reward)
    summary = dict(qualified=qualified, outcome=outcome,
        numerical_resolved=bool(audit["numerical_pass"]),
        probability_target_met=bool(audit["probability"] >= problem.target),
        device_feasible=bool(device["passed"]), probability=audit["probability"],
        fee=nominal["fee"], target=problem.target, threshold=problem.threshold,
        candidate_ids=problem.candidate_ids, probability_audit=audit,
        device_audit=device, search=search, global_optimal=False,
        probability_scope="forecast mean and nominal delivery; independent residual voltage innovations",
        case_input_reference=request_reference["case"], arrays_input_reference=request_reference["arrays"])
    write_json(out/"summary.json", summary)
    write_json(out/"price_freeze.json", dict(quote=reference(out/"quote.npz"),
        request=reference(out/"request.json"), qualified=qualified,
        additional_simulator_or_truth_access=False,
        upstream_information_boundary_verified=False))
    write_json(out/"complete.json", dict(status="complete", qualified=qualified,
        summary=reference(out/"summary.json"), freeze=reference(out/"price_freeze.json")))
    return summary


def read_quote(problem, path):
    """Validate a frozen request against this case's dimensions and bounds."""
    with np.load(path, allow_pickle=False) as saved:
        x = np.asarray(saved["x"], float)
        expected_u = np.asarray(saved["mean_u"], float)
    if x.shape != problem.shape or not np.isfinite(x).all():
        raise ValueError("Quote request dimensions differ from this case")
    values = x.ravel()[problem.active]
    if (np.any(values < problem.lower) or np.any(values > problem.upper)
            or not np.array_equal(problem.expand(values), x)):
        raise ValueError("Quote violates request bounds or uses an unavailable hour")
    point = problem.evaluate(values)
    if expected_u.shape != point["mean_u"].shape or not np.allclose(
            expected_u, point["mean_u"], atol=1e-10, rtol=0):
        raise ValueError("Quote delivery does not replay this case's device rules")
    return point


def main(args):
    problem = load_problem(args.case)
    if (problem.config.get("approved_for_new_optimization") is False
            and not args.check_only):
        raise ValueError("This legacy case is locked for replay while its settings are reviewed; use --check-only")
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out/"complete.json").exists():
        print("Already complete; use a new output directory for a new run.")
        return
    if (out/"request.json").exists():
        raise ValueError("An incomplete run is preserved; use a new output directory")
    if args.check_only and args.quote is None:
        raise ValueError("--check-only requires --quote")
    lock = out/"running.lock"
    with lock.open("x", encoding="utf-8") as stream:
        stream.write(str(os.getpid()))
    try:
        folder = Path(__file__).resolve().parent
        source_files = [folder/name for name in ("problem.py", "probability.py", "solve.py")]
        inputs = dict(case=reference(problem.case_path), arrays=reference(problem.arrays_path))
        controls = dict(maxiter=60, seconds=300., restore_evaluations=10,
                        restore_seconds=30., ftol=1e-8)
        controls.update(problem.config.get("solver", {}))
        allowed = {"maxiter", "seconds", "restore_evaluations", "restore_seconds", "ftol"}
        if set(controls) != allowed:
            raise ValueError("Unknown solver control")
        request = dict(inputs=inputs, config=problem.config,
            sources={path.name:reference(path)["sha256"] for path in source_files},
            quote=None if args.quote is None else reference(args.quote),
            check_only=bool(args.check_only), solver=controls,
            python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
            additional_simulator_or_truth_access=False,
            information_boundary="Only supplied case and optional quote are read; upstream preparation is caller-declared")
        write_json(out/"request.json", request)
        with zipfile.ZipFile(out/"source_snapshot.zip", "x", zipfile.ZIP_DEFLATED) as archive:
            for path in source_files:
                archive.write(path, path.name)
        initial = None
        if args.quote is not None:
            checked = read_quote(problem, args.quote)
            initial = checked["values"]
        if args.check_only:
            point, search = checked, dict(optimization_performed=False, mode="check_only")
        else:
            point, search = optimize(problem, out, initial, **controls)
        summary = freeze(problem, point, search, out, inputs)
        print(json.dumps(plain(summary), ensure_ascii=False))
    except Exception as exc:
        write_json(out/"failure.json", dict(error=repr(exc), traceback=traceback.format_exc()))
        raise
    finally:
        lock.unlink()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--quote", type=Path)
    parser.add_argument("--check-only", action="store_true")
    main(parser.parse_args())
