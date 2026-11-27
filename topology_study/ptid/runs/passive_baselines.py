"""End-to-end terminal-meter baselines for the four-feeder comparison.

RG, NJ, RNJ use the shared-resistance front end and the blind tolerances
registered on random development feeders in case_library_20261125/PROTOCOL_ID.md.
Park, Deka & Chertkov (2018), arXiv:1710.10727, uses measured magnitude drops,
per-meter P/Q moments (5), resistance distances (6), then finite-sample RG.
All four specify the known PCC as root and return nontrivial meter groups.
"""
from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path
from time import perf_counter

import gurobipy as gp
import numpy as np

from ptid.method import builders
from ptid.method.model import gls_gamma
from ptid.method.sets import lines_of

# The study runs as "python -m ptid.runs..." from topology_study. The sibling
# research package owns the registered, nonzero-tolerance implementations.
REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.append(str(REPO))
from rnj_wzzt_dr.readable import sources

TOLERANCES = {"rnj": .04, "nj": .04, "rg": .16}
PARK_EPS = 1e-4                    # resistance pu, the starting value in the paper's IEEE test


def park_moments(P, Q, drop, diagnostics):
    """Estimate the two sensitivity columns independently for each injection meter."""
    p, q, v = (a - a.mean(axis=0) for a in (P, Q, drop))
    n = p.shape[1]
    R, X = np.empty((n, n)), np.empty((n, n))
    conditions = []
    for b in range(n):
        z = np.column_stack((p[:, b], q[:, b]))
        covariance = z.T @ z / len(z)
        cond = np.linalg.cond(covariance)
        conditions.append(float(cond) if np.isfinite(cond) else None)
        if not np.isfinite(cond) or cond >= 1 / np.finfo(float).eps:
            diagnostics.update(singular_meter=b, pq_condition=conditions)
            raise ValueError(f"unidentifiable P/Q covariance at meter {b}")
        R[:, b], X[:, b] = np.linalg.solve(covariance, z.T @ v / len(z))
    diagnostics["pq_condition"] = conditions
    diagnostics["max_pq_condition"] = max(conditions)
    z = np.column_stack((p, q))
    norm = np.linalg.norm(z, axis=0)
    cor = (z.T @ z) / np.outer(norm, norm)
    different = np.tile(np.arange(n), 2)
    diagnostics["max_abs_cross_meter_pq_corr"] = float(np.max(np.abs(cor[different[:, None] != different])))
    diagnostics["max_raw_resistance_asymmetry"] = float(np.max(np.abs(R - R.T)))
    return (R + R.T) / 2, (X + X.T) / 2


def park_grouping(R, diagnostics):
    """Finite-sample RG (§III-B). Tau is infinite, using all available witnesses.

    Inconsistent approximate relations can overlap. Resolve them deterministically
    by merging the lowest-discrepancy compatible cliques first. Increase epsilon
    by 1.5 until a relation is found and reset it at every iteration (§IV).
    Keep inferred edges as selected, even with a negative fitted length; count
    negative sample distances explicitly. Only the topology is scored.
    """
    depth = np.diag(R)
    n = len(depth)
    D = np.zeros((n + 1, n + 1))
    D[0, 1:] = D[1:, 0] = depth
    D[1:, 1:] = depth[:, None] + depth[None, :] - 2 * R
    diagnostics.update(negative_input_distances=int((D[np.triu_indices(n + 1, 1)] < 0).sum()),
                       epsilon_start_pu=PARK_EPS, witness_cutoff="all", tolerance_by_round=[])
    active, edges, hidden = [-1, *range(n)], [], -2
    while len(active) > 2:
        m = len(active)
        idx = np.arange(m)
        phi = D[:, None, :] - D[None, :, :]
        phi[idx, :, idx] = phi[:, idx, idx] = np.nan
        spread = np.nanmax(phi, axis=2) - np.nanmin(phi, axis=2)
        # parent_error[i,j] measures whether j can be the parent of i.
        parent_error = np.nanmax(np.abs(phi - D[:, :, None]), axis=2)
        error = np.minimum(spread, np.minimum(parent_error, parent_error.T))
        pairs = sorted(combinations(range(m), 2), key=lambda p: (error[p], p))
        eps = PARK_EPS
        while error[pairs[0]] > eps:
            eps *= 1.5
        diagnostics["tolerance_by_round"].append(float(eps))
        parts = [[i] for i in range(m)]
        for i, j in pairs:
            if error[i, j] > eps:
                break
            a = next(g for g in parts if i in g)
            b = next(g for g in parts if j in g)
            if a is not b and np.all(error[np.ix_(a, b)] <= eps):
                a.extend(b)
                parts.remove(b)
        new_active, representatives, offsets = [], [], []
        mean_phi = np.nanmean(phi, axis=2)
        for members in parts:
            parent = next((j for j in members if all(
                i == j or parent_error[i, j] <= eps for i in members)), None)
            if parent is not None:
                new_active.append(active[parent])
                representatives.append([parent])
                offsets.append(np.zeros(1))
                edges.extend((active[parent], active[i], D[parent, i]) for i in members if i != parent)
            else:
                lengths = np.array([np.mean([.5 * (D[i, j] + mean_phi[i, j])
                                             for j in members if j != i]) for i in members])
                edges.extend((hidden, active[i], d) for i, d in zip(members, lengths))
                new_active.append(hidden)
                hidden -= 1
                representatives.append(members)
                offsets.append(lengths)
        next_D = np.zeros((len(parts), len(parts)))
        for i, j in combinations(range(len(parts)), 2):
            block = D[np.ix_(representatives[i], representatives[j])]
            next_D[i, j] = next_D[j, i] = block.mean() - offsets[i].mean() - offsets[j].mean()
        active, D = new_active, next_D
    if len(active) == 2:
        edges.append((active[0], active[1], D[0, 1]))
    diagnostics["negative_fitted_edges"] = sum(float(length) < 0 for _, _, length in edges)
    return builders.tree_family(edges, n, -np.inf)


def run_baseline(method, P, Q, Y, V0sq):
    """Return one case result including its own preprocessing time; tree=None on failure."""
    start = perf_counter()
    method = method.lower()
    diagnostics = {}
    try:
        P, Q, Y, V0sq = (np.asarray(a, dtype=float) for a in (P, Q, Y, V0sq))
        if P.ndim != 2 or P.shape != Q.shape or P.shape != Y.shape or V0sq.shape != (len(P),):
            raise ValueError("expected T-by-n P/Q/Y and T root squared voltages")
        if not all(np.isfinite(a).all() for a in (P, Q, Y, V0sq)):
            raise ValueError("nonfinite measurement")
        if method == "park":
            v_sq = V0sq[:, None] - Y
            if np.any(v_sq <= 0) or np.any(V0sq <= 0):
                raise ValueError("nonpositive measured squared voltage")
            # The archive uses positive consumption, so the regressand is root magnitude minus bus magnitude.
            drop = np.sqrt(V0sq)[:, None] - np.sqrt(v_sq)
            R, _ = park_moments(P, Q, drop, diagnostics)
            family = park_grouping(R, diagnostics)
        elif method in TOLERANCES:
            gamma = gls_gamma(V0sq, Y)
            demean = builders.d1(None)
            R = builders.r_hat(demean(P), demean(Y - gamma * Y.mean(1, keepdims=True)), gamma)
            shared, depth = builders.geometry(R)
            tolerance = TOLERANCES[method]
            family = getattr(sources, method)(shared, depth, tolerance)
            diagnostics.update(tolerance_fraction=tolerance, gls_gamma=float(gamma))
        else:
            raise ValueError(f"unknown passive method {method}")
        tree = list(lines_of(family, P.shape[1]))
        return dict(method=method, status="ok", tree=tree, diagnostics=diagnostics, seconds=perf_counter()-start)
    except (ValueError, RuntimeError, gp.GurobiError, np.linalg.LinAlgError) as exc:
        return dict(method=method, status="failed", tree=None, error=str(exc), diagnostics=diagnostics,
                    seconds=perf_counter()-start)


def selfcheck():
    """Exactly uncorrelated meter injections and non-collinear local P/Q on a known feeder."""
    n, samples = 5, 192
    truth = {(0, 1), (2, 3, 4), (3, 4)}
    R = np.diag([.12, .09, .10, .11, .12])
    for group, resistance in zip(sorted(truth), (.06, .07, .05)):
        R[np.ix_(group, group)] += resistance
    X = .25 * R
    rng = np.random.default_rng(20261128)
    raw = rng.normal(size=(samples, 2*n))
    u, _ = np.linalg.qr(raw - raw.mean(0))
    P = .001 + .0001*u[:, :n]
    Q = .0003 + .0001*u[:, n:]
    v0 = np.ones(samples)
    drop = P @ R + Q @ X
    Y = v0[:, None]**2 - (v0[:, None] - drop)**2
    diagnostics = {}
    Rest, Xest = park_moments(P, Q, drop, diagnostics)
    err_R, err_X = float(abs(Rest-R).max()), float(abs(Xest-X).max())
    rows = [run_baseline(m, P, Q, Y, v0**2) for m in (*TOLERANCES, "park")]
    singular = run_baseline("park", P, .25*P, Y, v0**2)
    passed = (max(err_R, err_X) < 1e-9 and singular["status"] == "failed"
              and all(r["status"] == "ok" and set(r["tree"]) == truth for r in rows))
    return dict(passed=passed, source="https://arxiv.org/pdf/1710.10727", samples=samples, seed=20261128,
                known_groups=sorted(truth), max_R_error=err_R, max_X_error=err_X, conditions=diagnostics,
                methods=rows, fixed_power_factor_control=singular)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    result = selfcheck()
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print({k: result[k] for k in ("passed", "max_R_error", "max_X_error")}, flush=True)
    raise SystemExit(0 if result["passed"] else 1)
