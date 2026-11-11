"""Train-only numerical replay of one historical T-bound contradiction.

This is an optimization-correctness diagnostic, not a topology benchmark.
Only flynn16/b1001/v0002/train_observed.npz is opened as observations.
No native project module, validation/test arrays, or physical truth is used.
"""
from __future__ import annotations

import argparse
import hashlib
from itertools import combinations
import json
import os
from pathlib import Path
import shutil
import sys
from time import perf_counter
from datetime import datetime, timezone

import numpy as np
from scipy.optimize import lsq_linear
import gurobipy as gp


JOB = "day96__flynn16__b1001__reference__v0002__N96"
TRAIN = "data/flynn16/b1001/v0002/train_observed.npz"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def family(rows):
    return tuple(sorted({tuple(sorted(C)) for C in rows}, key=lambda C: (len(C), C)))


def compatible(C, D):
    a, b = set(C), set(D)
    return not a & b or a <= b or b <= a


def features(p, q, supports, reference):
    n = p.shape[1]
    z = np.array([[i in C for i in range(n)] for C in supports], float)
    return tuple(np.column_stack([((v@row)[:, None]*row).ravel() for row in z])
                 / np.sqrt(reference) for v in (p, q))


def audit(point, fixture, p, q, y):
    supports = list(map(tuple, point["supports"]))
    Fr, Fx = features(p, q, supports, fixture["reference"])
    F = np.column_stack([Fr, Fx]); theta = np.r_[point["r"], point["x"]]
    residual = F.astype(np.longdouble)@theta.astype(np.longdouble)-y.astype(np.longdouble)
    objective = float(np.mean(residual**2))
    gradient = np.asarray(2*F.astype(np.longdouble).T@residual/len(y), float)
    B = fixture["bound"]
    lower = max(0., objective+float(np.minimum(-gradient*theta, gradient*(B-theta)).sum()))
    n = p.shape[1]; proper = [C for C in supports if 1 < len(C) < n]
    keep, pool = set(map(tuple, fixture["keep"])), set(map(tuple, fixture["pool"]))
    count_violation = max([sum(set(C) < set(W) for C in proper)-(len(W)-2)
                           for W in pool if len(W) > 1], default=0)
    violations = dict(box=max(0., float(np.max(-theta)), float(np.max(theta-B))),
        missing_keep=len(keep-set(supports)), outside_pool=len(set(supports)-pool),
        crossings=sum(not compatible(C, D) for C, D in combinations(supports, 2)),
        hierarchy_count=max(0, count_violation))
    return dict(normalized_mse=objective, train_mse=objective*fixture["reference"],
        J=objective+fixture["alpha"]*len(proper), proper_count=len(proper),
        box_lower_bound=lower, box_certificate_gap=max(0., objective-lower),
        violations=violations, finite_pool_feasible=all(v <= 1e-7 for v in violations.values()))


def attr(model, name):
    try:
        value = float(model.getAttr(name))
        return value if np.isfinite(value) and abs(value) < 1e90 else None
    except (AttributeError, gp.GurobiError):
        return None


def solve(fixture, p, q, target, output, *, selection, name, presolve=-1, focus=0, limit=300.):
    started = perf_counter(); n = p.shape[1]
    supports = family(fixture["pool"] if selection else fixture["raw"]["supports"])
    proper = [C for C in supports if 1 < len(C) < n]
    positions = {C: i for i, C in enumerate(supports)}
    selected = {C: i for i, C in enumerate(proper)}
    fixed = set(map(tuple, fixture["keep"]))
    Fr, Fx = features(p, q, supports, 1.)
    scale = np.sqrt(fixture["reference"]); y = target.ravel()/scale
    with gp.Env(empty=True) as env:
        env.setParam("OutputFlag", 0); env.start()
        with gp.Model("RTE_laminar_selection" if selection else "fixed_family_QP", env=env) as m:
            settings = dict(Threads=1, OutputFlag=1, LogToConsole=0, NonConvex=0, Seed=20261111,
                TimeLimit=limit, MIPGap=1e-4, MIPGapAbs=1e-7,
                FeasibilityTol=1e-9, IntFeasTol=1e-9, OptimalityTol=1e-9,
                Presolve=presolve, NumericFocus=focus, LogFile=str(output/(name+".log")))
            for key, value in settings.items():
                m.setParam(key, value)
            r = m.addMVar(len(supports), lb=0., ub=fixture["bound"], name="r")
            x = m.addMVar(len(supports), lb=0., ub=fixture["bound"], name="x")
            if selection:
                a = m.addMVar(len(proper), vtype="B", name="selected")
                for C, j in selected.items():
                    i = positions[C]
                    m.addConstr(r[i] <= fixture["bound"]*a[j])
                    m.addConstr(x[i] <= fixture["bound"]*a[j])
                    if C in fixed:
                        a[j].LB = 1.
                for C, D in combinations(proper, 2):
                    if not compatible(C, D):
                        m.addConstr(a[selected[C]]+a[selected[D]] <= 1.)
                for W in [*proper, tuple(range(n))]:
                    inside = [selected[C] for C in proper if set(C) < set(W)]
                    if inside:
                        m.addConstr(a[inside].sum() <= len(W)-2)
            residual = m.addMVar(len(y), lb=-float("inf"), name="scaled_residual")
            m.addConstr(residual == (Fr@r+Fx@x-target.ravel())/scale)
            penalty = fixture["alpha"]*(a.sum() if selection else len(proper))
            m.setObjective((residual@residual)/len(y)+penalty)
            if selection:
                warm = fixture["warm"]
                lookup = {tuple(C): (rr, xx) for C, rr, xx in
                          zip(warm["supports"], warm["r"], warm["x"], strict=True)}
                r.Start = [lookup.get(C, (0., 0.))[0] for C in supports]
                x.Start = [lookup.get(C, (0., 0.))[1] for C in supports]
                a.Start = [float(C in lookup) for C in proper]
                z = np.array([[i in C for i in range(n)] for C in warm["supports"]], float)
                R = (z.T*np.asarray(warm["r"]))@z
                X = (z.T*np.asarray(warm["x"]))@z
                residual.Start = ((p@R.T+q@X.T-target)/scale).ravel()
            m.optimize()
            record = dict(name=name, selection=selection, settings=settings, status=int(m.Status),
                solution_count=int(m.SolCount), objective=attr(m, "ObjVal"), bound=attr(m, "ObjBound"),
                relative_gap=attr(m, "MIPGap"), runtime=attr(m, "Runtime"), nodes=attr(m, "NodeCount"),
                variables=int(m.NumVars), constraints=int(m.NumConstrs),
                constraint_violation=attr(m, "ConstrVio"), bound_violation=attr(m, "BoundVio"),
                integrality_violation=attr(m, "IntVio"))
            if m.SolCount:
                active = [i for i, C in enumerate(supports)
                          if not selection or C in fixed or a.X[selected[C]] > .5]
                point = dict(supports=[list(supports[i]) for i in active],
                    r=np.clip(r.X[active], 0., fixture["bound"]).tolist(),
                    x=np.clip(x.X[active], 0., fixture["bound"]).tolist())
                record.update(point=point, independent=audit(point, fixture, p, q, y))
            record["wall_seconds"] = perf_counter()-started
    write(output/(name+".json"), record)
    print(json.dumps({k: record[k] for k in ("name", "status", "objective", "bound", "relative_gap", "wall_seconds")}), flush=True)
    return record


def main(output, benchmark, limit):
    output = output.resolve(); output.mkdir(parents=True, exist_ok=False)
    source = Path(__file__).resolve(); shutil.copy2(source, output/source.name)
    raw_file = benchmark/"runs"/JOB/"010"/"result.json"
    result = json.loads(raw_file.read_text(encoding="utf-8"))["result"]
    event = next(e for e in result["events"] if e["stage"] == "T")
    warm = result["path"][0]
    for e in result["events"]:
        if e["stage"] == "T":
            break
        if e.get("accepted"):
            warm = e["point"]
    fixture = dict(pool=event["pool"], keep=event["keep"], warm=warm,
        raw=event["raw_point"], closed=event["point"], native=event["solver"],
        reference=result["reference_mse"], bound=result["config"]["response_bound"],
        alpha=result["config"]["alpha"])
    write(output/"training_fixture.json", fixture)
    train_file = benchmark/"runtime"/TRAIN
    shutil.copy2(train_file, output/"train_observed.npz")
    with np.load(train_file, allow_pickle=False) as f:
        n = len(f["labels"])
        p, q, voltage = [np.asarray(f[k], float).reshape(-1, n) for k in ("P", "Q", "V")]
        target = np.asarray(f["root"], float).reshape(-1, 1)**2-voltage**2
    assert len(p) == 96 and n == 16
    y = target.ravel()/np.sqrt(fixture["reference"])
    report = dict(scope="Optimization numerical debug only, not new identification scores",
        job=JOB, source_sha256=sha(source), historical_record_sha256=sha(raw_file),
        train_sha256=sha(train_file), observation_npz_opened=[str(train_file)],
        total_observed_time_points=96, validation_test_truth_opened=False,
        python=sys.version, executable=sys.executable, numpy=np.__version__,
        gurobi_version=gp.gurobi.version(), thread_environment={k: os.environ.get(k)
            for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")},
        started_utc=datetime.now(timezone.utc).isoformat(), per_solve_limit=limit,
        historical_raw=audit(fixture["raw"], fixture, p, q, y),
        historical_closed=audit(fixture["closed"], fixture, p, q, y), runs=[])
    assert report["historical_closed"]["finite_pool_feasible"]
    report["historical_bound_conflict"] = fixture["native"]["bound"]-report["historical_closed"]["J"]
    write(output/"report.json", report)
    Fr, Fx = features(p, q, fixture["raw"]["supports"], fixture["reference"])
    F = np.column_stack([Fr, Fx]); started = perf_counter()
    solved = lsq_linear(F, y, bounds=(0., fixture["bound"]), method="bvls", tol=1e-12, max_iter=5000)
    m = len(fixture["raw"]["supports"])
    bvls = dict(supports=fixture["raw"]["supports"], r=solved.x[:m].tolist(), x=solved.x[m:].tolist())
    report["BVLS"] = dict(success=bool(solved.success), status=int(solved.status),
        iterations=int(solved.nit), wall_seconds=perf_counter()-started,
        point=bvls, independent=audit(bvls, fixture, p, q, y))
    write(output/"report.json", report)
    fixed = solve(fixture, p, q, target, output, selection=False, name="fixed_gurobi_QP", limit=limit)
    report["runs"].append(fixed); write(output/"report.json", report)
    for name, presolve, focus in [("T_auto", -1, 0), ("T_presolve0", 0, 0), ("T_numericfocus3", -1, 3)]:
        record = solve(fixture, p, q, target, output, selection=True, name=name,
                       presolve=presolve, focus=focus, limit=limit)
        record["bound_minus_known_feasible_J"] = (record["bound"]-report["historical_closed"]["J"]
            if record["bound"] is not None else None)
        report["runs"].append(record); write(output/"report.json", report)
    report["finished_utc"] = datetime.now(timezone.utc).isoformat()
    report["source_unchanged"] = sha(source) == report["source_sha256"]
    write(output/"report.json", report)
    print("REPORT "+str(output/"report.json"), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--benchmark", type=Path,
        default=Path(__file__).resolve().parents[1]/"outputs/full48_mse_20261111")
    parser.add_argument("--limit", type=float, default=300.)
    args = parser.parse_args()
    if not 0 < args.limit <= 300:
        parser.error("This diagnostic caps every native solve at 300 seconds")
    main(args.output, args.benchmark.resolve(), args.limit)
