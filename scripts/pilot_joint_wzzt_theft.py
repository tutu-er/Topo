"""Small controlled joint topology/weight/theft MILP feasibility study.

Not the production mainline: four terminals, three balanced candidate trees,
one stationary source, exact externally observed total extra P and common PF,
linear voltage, zero technical losses. No AC or calibration claim.
"""
from itertools import combinations
from pathlib import Path
from time import perf_counter
import argparse
import hashlib
import json

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

ROOT = Path(__file__).resolve().parents[1]
SUPPORTS = tuple(frozenset([j]) for j in range(4)) + tuple(
    frozenset(c) for c in combinations(range(4), 2)) + (frozenset(range(4)),)
Z = np.array([[j in s for j in range(4)] for s in SUPPORTS], float)
A = np.array([[h <= e for h in SUPPORTS] for e in SUPPORTS], float)
E = len(Z)
OBJECTIVE_SCALE = 1e6  # Keep solver absolute tolerance below the reporting scale.
TRUE_PAIRS = (SUPPORTS.index(frozenset([0, 1])), SUPPORTS.index(frozenset([2, 3])))
SOURCE = TRUE_PAIRS[0]
R0 = np.zeros(E); X0 = np.zeros(E)
R0[:4] = [.03, .04, .05, .045]; X0[:4] = [.025, .02, .015, .03]
R0[list(TRUE_PAIRS)] = [.04, .06]; X0[list(TRUE_PAIRS)] = [.025, .02]
R0[-1] = .015; X0[-1] = .012


def voltage(p, q, r, x, amplitude, source):
    b = np.zeros(E) if source is None else A[:, source]
    return ((p @ Z.T + amplitude[:, None] * b) * r
            + (q @ Z.T + .4 * amplitude[:, None] * b) * x) @ Z


def fit(p, q, y, amplitude, *, joint, intercept=False):
    t, n = p.shape
    # [r, x, topology switches, location switches, r*b, x*b, intercept, |residual|]
    r = np.arange(E); x = r + E; u = x + E; s = u + E
    vr = s + E; vx = vr + E
    offset = np.arange(6 * E, 6 * E + n)
    residual = np.arange(6 * E + n, 6 * E + n + t * n)
    size = int(residual[-1] + 1)
    lower = np.zeros(size); upper = np.full(size, np.inf)
    costs = np.zeros(size); costs[residual] = OBJECTIVE_SCALE / y.size
    integrality = np.zeros(size, dtype=int)
    upper[np.r_[r, x, vr, vx]] = .25
    upper[np.r_[u, s]] = 1
    integrality[np.r_[u, s]] = 1
    lower[u[:4]] = 1; lower[u[-1]] = 1
    if intercept:
        lower[offset] = -np.inf
    else:
        upper[offset] = 0
    if not joint or not np.any(amplitude > 0):
        upper[s] = 0
    ri, ci, values, lows, highs = [], [], [], [], []

    def add(coeff, lo=-np.inf, hi=np.inf):
        row = len(lows)
        for index, value in coeff.items():
            if value:
                ri.append(row); ci.append(int(index)); values.append(float(value))
        lows.append(lo); highs.append(hi)

    # Exactly two disjoint pair clades: three admissible balanced trees.
    add({int(u[j]): 1 for j in range(4, 10)}, 2, 2)
    for i, j in combinations(range(4, 10), 2):
        if SUPPORTS[i] & SUPPORTS[j]:
            add({int(u[i]): 1, int(u[j]): 1}, hi=1)
    active = int(joint and np.any(amplitude > 0))
    add({int(v): 1 for v in s}, active, active)
    for e in range(E):
        add({int(s[e]): 1, int(u[e]): -1}, hi=0)
        for weight in (r[e], x[e]):
            add({int(weight): 1, int(u[e]): -.25}, hi=0)
            add({int(weight): 1, int(u[e]): -.005}, lo=0)
        b = {int(s[h]): A[e, h] for h in range(E) if A[e, h]}
        for weight, product in ((r[e], vr[e]), (x[e], vx[e])):
            # b is binary because the location switches sum to zero or one.
            add({int(product): 1, **{k: -.25 * v for k, v in b.items()}}, hi=0)
            add({int(product): 1, int(weight): -1}, hi=0)
            add({int(product): 1, int(weight): -1,
                 **{k: -.25 * v for k, v in b.items()}}, lo=-.25)
    fp, fq = p @ Z.T, q @ Z.T
    for k in range(t):
        for j in range(n):
            coef = {int(offset[j]): 1}
            for e in range(E):
                if Z[e, j]:
                    coef[int(r[e])] = fp[k, e]
                    coef[int(x[e])] = fq[k, e]
                    coef[int(vr[e])] = amplitude[k]
                    coef[int(vx[e])] = .4 * amplitude[k]
            error = int(residual[k * n + j])
            add({**coef, error: -1}, hi=y[k, j])
            add({**coef, error: 1}, lo=y[k, j])
    matrix = coo_matrix((values, (ri, ci)), shape=(len(lows), size)).tocsc()
    started = perf_counter()
    opt = milp(costs, integrality=integrality, bounds=Bounds(lower, upper),
               constraints=LinearConstraint(matrix, lows, highs),
               options={"time_limit": 30., "mip_rel_gap": 1e-9})
    if opt.status != 0 or opt.x is None:
        return dict(status="failed", message=opt.message, seconds=perf_counter() - started)
    selected = np.rint(opt.x[s]).astype(int)
    chosen = np.flatnonzero(selected)
    source = int(chosen[0]) if len(chosen) else None
    r_fit, x_fit = opt.x[r], opt.x[x]
    prediction = voltage(p, q, r_fit, x_fit, amplitude, source) + opt.x[offset]
    direct = float(np.mean(np.abs(y - prediction)))
    activity = matrix @ opt.x
    feasibility = max(float(np.max(np.maximum(np.asarray(lows) - activity, 0))),
                      float(np.max(np.maximum(activity - np.asarray(highs), 0))),
                      float(np.max(np.maximum(lower - opt.x, 0))),
                      float(np.max(np.maximum(opt.x - upper, 0))),
                      float(np.max(np.abs(opt.x[np.r_[u, s]] - np.rint(opt.x[np.r_[u, s]])))))
    b = A @ selected
    product_error = float(max(np.max(np.abs(opt.x[vr] - r_fit * b)),
                              np.max(np.abs(opt.x[vx] - x_fit * b))))
    assert feasibility < 2e-6 and product_error < 2e-6
    assert abs(direct - opt.fun / OBJECTIVE_SCALE) < 2e-6
    assert float(opt.mip_gap) < 1e-7
    pair_indices = np.flatnonzero(opt.x[u[4:10]] > .5) + 4
    rm = Z.T @ (r_fit[:, None] * Z); xm = Z.T @ (x_fit[:, None] * Z)
    rt = Z.T @ (R0[:, None] * Z); xt = Z.T @ (X0[:, None] * Z)
    return dict(status="optimal", selected_pairs=[sorted(SUPPORTS[i]) for i in pair_indices],
                topology_correct=set(pair_indices) == set(TRUE_PAIRS),
                source_support=None if source is None else sorted(SUPPORTS[source]),
                source_correct=source == SOURCE if np.any(amplitude > 0) else source is None,
                train_mae=direct, r_relative_error=float(np.linalg.norm(rm - rt) / np.linalg.norm(rt)),
                x_relative_error=float(np.linalg.norm(xm - xt) / np.linalg.norm(xt)),
                r=r_fit.tolist(), x=x_fit.tolist(), intercept=opt.x[offset].tolist(),
                mip_gap=float(opt.mip_gap), feasibility=feasibility,
                product_error=product_error, objective_error=abs(direct - float(opt.fun) / OBJECTIVE_SCALE),
                seconds=perf_counter() - started)


def main(output, seeds):
    output = output.resolve()
    if not output.is_relative_to(ROOT):
        raise ValueError("Output must be in workspace")
    if output.exists():
        raise FileExistsError(output)
    rows = []
    for seed in range(seeds):
        rng = np.random.default_rng(20260920 + seed)
        p = rng.uniform(.003, .021, (32, 4)); q = rng.uniform(.002, .012, (32, 4))
        pv = rng.uniform(.003, .021, (64, 4)); qv = rng.uniform(.002, .012, (64, 4))
        clean_validation = voltage(pv, qv, R0, X0, np.zeros(len(pv)), None)
        noise = rng.normal(0, .00002, p.shape)
        gate = np.zeros(32); gate[rng.choice(32, 16, replace=False)] = 1
        cases = dict(clean=np.zeros(32), intermittent=.035 * gate,
                     correlated=2.5 * p[:, 2], persistent=np.full(32, .025))
        for name, amplitude in cases.items():
            y = voltage(p, q, R0, X0, amplitude, SOURCE) + noise
            for intercept in ([False, True] if name == "persistent" else [False]):
                fits = {}
                for method, joint in (("topology_only", False), ("joint", True)):
                    fits[method] = fit(p, q, y, amplitude, joint=joint, intercept=intercept)
                    item = fits[method]
                    if item["status"] == "optimal":
                        normal_prediction = voltage(pv, qv, np.asarray(item["r"]),
                                                    np.asarray(item["x"]), np.zeros(len(pv)), None)
                        item["heldout_clean_mae_without_intercept"] = float(np.mean(abs(normal_prediction - clean_validation)))
                rows.append(dict(seed=seed, condition=name, free_intercepts=intercept, fits=fits))
                print(json.dumps({"seed": seed, "condition": name, "intercept": intercept,
                                  "status": {k: v["status"] for k, v in fits.items()}}, ensure_ascii=False), flush=True)
    summary = []
    for name, intercept in sorted(set((row["condition"], row["free_intercepts"]) for row in rows)):
        selected = [row for row in rows if (row["condition"], row["free_intercepts"]) == (name, intercept)]
        for method in ("topology_only", "joint"):
            success = [row["fits"][method] for row in selected if row["fits"][method]["status"] == "optimal"]
            summary.append(dict(condition=name, free_intercepts=intercept, method=method,
                                attempts=len(selected), optimal=len(success),
                                correct_topology=sum(v["topology_correct"] for v in success),
                                correct_source=sum(v["source_correct"] for v in success),
                                mean_r_relative_error=float(np.mean([v["r_relative_error"] for v in success])) if success else None,
                                mean_x_relative_error=float(np.mean([v["x_relative_error"] for v in success])) if success else None,
                                mean_heldout_clean_mae=float(np.mean([v["heldout_clean_mae_without_intercept"] for v in success])) if success else None))
    result = dict(protocol=dict(terminals=4, topology_pool="three balanced trees, all candidates included",
                               samples=32, holdout_samples=64, seeds=seeds, source="one stationary clade {0,1}",
                               voltage_noise_std=.00002, external_amplitude="exact ideal meter balance",
                               q_over_p=.4, forward_model="linear squared voltage, no AC losses",
                               source_presence="given by ideal total balance, not an alarm test",
                               weight_bounds=[.005, .25], objective_scale=OBJECTIVE_SCALE, script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                               limitations="Controlled feasibility evidence, not blind topology recovery, multi-source validation, or calibrated detection."),
                  summary=summary, rows=rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, default=5)
    args = parser.parse_args()
    if args.seeds < 1:
        raise ValueError("Positive seed count required")
    main(args.output, args.seeds)
