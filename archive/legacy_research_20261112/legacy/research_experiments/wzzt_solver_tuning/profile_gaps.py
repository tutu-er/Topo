"""Replay frozen first-extension matrices with matched MIP-gap settings.

No old primal is supplied as a warm start. No truth is read by this driver.
The models retain all original bounds, integer domains and RX75 constraints.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
from time import perf_counter

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"

import numpy as np
from scipy.optimize import Bounds, LinearConstraint
from scipy.sparse import coo_matrix

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "outputs/rnj_blocks96_20261019"
RUNTIME = ROOT / "outputs/source_ac288_runtime_20261018"
JOBS = ("soumalas11__no_source__RNJ+WZZT_tree_init",
        "paper15__no_source__WZZT", "pengwah18__no_source__WZZT")
GAPS = (0., .01, .05)


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def primal_audit(raw, primal):
    if primal is None:
        return {"feasible": False, "reason": "no_incumbent"}
    activity = np.bincount(raw["constraint_row_index"],
        weights=raw["constraint_data"] * primal[raw["constraint_col_index"]],
        minlength=len(raw["constraint_lower"]))
    violation = max(0., np.max(raw["constraint_lower"]-activity, initial=0.),
                    np.max(activity-raw["constraint_upper"], initial=0.))
    bound = max(0., np.max(raw["lower_bounds"]-primal, initial=0.),
                np.max(primal-raw["upper_bounds"], initial=0.))
    integer = raw["integrality"] != 0
    integral = np.max(abs(primal[integer]-np.rint(primal[integer])), initial=0.)
    return {"feasible": bool(np.isfinite(primal).all() and max(violation, bound, integral) <= 1e-7),
            "max_constraint_violation": float(violation), "max_bound_violation": float(bound),
            "max_integrality_violation": float(integral),
            "objective_recomputed": float(raw["objective_coefficients"] @ primal)}


def prepare(output, seconds):
    evidence, models = {}, []
    for job in JOBS:
        directory = RUN / "jobs" / job
        events = read(directory / "solver_events.json")
        event = next(e for e in events if "z" in e["variable_slices"])
        raw = directory / event["primal_file"]
        models.append({"job_id": job, "raw": str(raw.relative_to(ROOT)),
            "event": event, "historical_seconds": event["runtime_seconds"],
            "historical_objective": event["objective"],
            "current_objective": events[0]["objective"]})
        for path in (raw, directory/"solver_events.json", directory/"payload.json",
                     directory/"initialization.json", directory/"rx75_calibration.json"):
            evidence[str(path.relative_to(ROOT))] = sha(path)
    for path in (Path(__file__), RUNTIME/"rnj_wzzt/estimation/gurobi_milp.py"):
        evidence[str(path.relative_to(ROOT))] = sha(path)
    protocol = {"schema_version": 1, "models": models, "gaps": GAPS,
        "per_solve_seconds": seconds, "threads": 1, "warm_start": False,
        "source": "no_source only; fixed archived first-extension matrices",
        "selection": "observed objectives, bounds and AC residuals only; no physical truth",
        "experimental_scope": "single-step solver settings; no full-path performance claim",
        "versions": {n: importlib.metadata.version(n) for n in ("numpy", "scipy", "gurobipy")},
        "input_sha256": evidence}
    output.mkdir(parents=True, exist_ok=False)
    write(output/"protocol.json", protocol)
    return protocol


def run(output, protocol):
    sys.path.insert(0, str(RUNTIME))
    from rnj_wzzt.estimation.gurobi_milp import solve_gurobi_milp
    results = []
    for model in protocol["models"]:
        raw = dict(np.load(ROOT/model["raw"], allow_pickle=False))
        matrix = coo_matrix((raw["constraint_data"],
            (raw["constraint_row_index"], raw["constraint_col_index"])),
            shape=(len(raw["constraint_lower"]), len(raw["objective_coefficients"]))).tocsr()
        for gap in protocol["gaps"]:
            target = output / f"{model['job_id']}__gap_{gap:g}"
            target.mkdir()
            print(f"START {target.name}", flush=True)
            started = perf_counter()
            result = solve_gurobi_milp(c=raw["objective_coefficients"], integrality=raw["integrality"],
                bounds=Bounds(raw["lower_bounds"], raw["upper_bounds"]),
                constraints=LinearConstraint(matrix, raw["constraint_lower"], raw["constraint_upper"]),
                options={"disp": False, "presolve": True, "time_limit": protocol["per_solve_seconds"],
                         "mip_rel_gap": gap, "mip_abs_gap": 1e-10})
            primal = None if result.x is None else np.asarray(result.x)
            audit = primal_audit(raw, primal)
            ub, lb = result.fun, result.mip_dual_bound
            row = {"job_id": model["job_id"], "requested_gap": gap,
                "runtime_seconds": perf_counter()-started, "raw_status": result.raw_status,
                "message": result.message, "objective": ub, "bound": lb, "gap": result.mip_gap,
                "absolute_gap": None if ub is None or lb is None else ub-lb,
                "node_count": result.mip_node_count, "primal_audit": audit,
                "gap_target_met": bool(audit["feasible"] and result.mip_gap is not None
                    and (result.mip_gap <= gap+1e-12 or abs(ub-lb) <= 1e-10)),
                "historical_objective": model["historical_objective"],
                "objective_excess_over_historical": None if ub is None else ub-model["historical_objective"],
                "gain_lower": None if ub is None else model["current_objective"]-ub,
                "gain_upper": None if lb is None else model["current_objective"]-lb}
            if primal is not None:
                np.savez_compressed(target/"primal.npz", primal=primal)
            write(target/"result.json", row)
            results.append(row)
            write(output/"results.json", results)
            print(json.dumps(row, ensure_ascii=False), flush=True)
    write(output/"completion.json", {"expected": len(protocol["models"])*len(protocol["gaps"]),
        "returned": len(results), "feasible": sum(r["primal_audit"]["feasible"] for r in results),
        "target_met": sum(r["gap_target_met"] for r in results)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT/"outputs/wzzt_solver_tuning_20261027/gap_profiles")
    parser.add_argument("--seconds", type=float, default=90.)
    args = parser.parse_args()
    if not np.isfinite(args.seconds) or args.seconds <= 0:
        parser.error("--seconds must be positive")
    protocol = prepare(args.output_dir, args.seconds)
    run(args.output_dir, protocol)


if __name__ == "__main__":
    main()
