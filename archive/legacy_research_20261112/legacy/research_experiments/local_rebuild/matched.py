"""Identical finite domains and gain gates for repair and joint reconstruction.

The LP bank retains primal/dual certificates and audits the original P/Q/Y
prediction. Search receives observations, a fixed pool and mandatory supports;
validation, test and topology truth are evaluator-only inputs.
"""
from __future__ import annotations

from itertools import combinations
from time import perf_counter

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csc_matrix, eye, hstack, vstack

from .run import core
from .model import canonical, compatible, prediction_loss


def gate(objective, relative=.001):
    if not np.isfinite(relative) or relative < 0.:
        raise ValueError("relative gate must be finite and nonnegative")
    return 1e-10 + relative * objective


class FiniteLP:
    """One fixed feature bank, with zero upper bounds on inactive columns."""

    def __init__(self, prepared, pool, *, mandatory=(), bound=2., seconds=3.):
        self.prepared = core._prepare_scenarios(prepared)
        self.pool = canonical(pool)
        self.mandatory = canonical(mandatory)
        if not set(self.mandatory) <= set(self.pool):
            raise ValueError("mandatory supports missing from pool")
        self.bound, self.seconds = float(bound), float(seconds)
        if min(self.bound, self.seconds) <= 0.:
            raise ValueError("positive bound and LP time limit required")
        self.lookup = {s: i for i, s in enumerate(self.pool)}
        fp, fq = core._fixed_atom_features(self.prepared, self.pool)
        self.features = np.column_stack((fp, fq))
        self.y = np.concatenate([y.ravel() for y in self.prepared.target])
        self.m, self.k = len(self.y), len(self.pool)
        identity = eye(self.m, format="csc")
        self.A = vstack([hstack([csc_matrix(self.features), -identity]),
                         hstack([-csc_matrix(self.features), -identity])], format="csc")
        self.rhs = np.r_[self.y, -self.y]
        self.c = np.r_[np.zeros(2 * self.k), np.ones(self.m) / self.m]
        self.cache, self.solves = {}, []

    def fit(self, family):
        family = canonical([*self.mandatory, *family])
        core.normalize_supports(family, self.prepared.n)
        if family in self.cache:
            return self.cache[family]
        if not set(family) <= set(self.pool):
            raise ValueError("requested support outside common domain")
        active = [self.lookup[s] for s in family]
        upper = np.zeros(2 * self.k)
        upper[active] = upper[np.asarray(active, dtype=int) + self.k] = self.bound
        upper = np.r_[upper, np.full(self.m, np.inf)]
        started = perf_counter()
        result = linprog(self.c, A_ub=self.A, b_ub=self.rhs,
                         bounds=np.column_stack((np.zeros(len(upper)), upper)), method="highs-ds",
                         options=dict(time_limit=self.seconds, primal_feasibility_tolerance=1e-9,
                                      dual_feasibility_tolerance=1e-9))
        record = dict(family=family, status=int(result.status), message=result.message,
                      seconds=perf_counter() - started, valid=False, dual_bound=None,
                      objective=None if result.fun is None else float(result.fun))
        self.solves.append(record)
        if result.status != 0 or result.x is None:
            raise RuntimeError(f"fixed LP uncertified: {record}")
        values = result.x
        primal_violation = float(max(np.max(self.A @ values - self.rhs, initial=0.),
                                     np.max(-values, initial=0.), np.max(values - upper, initial=0.)))
        mu, lo, hi = result.ineqlin.marginals, result.lower.marginals, result.upper.marginals
        finite = np.isfinite(upper)
        dual = float(self.rhs @ mu + upper[finite] @ hi[finite])
        stationarity = float(np.max(abs(self.c - self.A.T @ mu - lo - hi), initial=0.))
        signs = float(max(np.max(mu, initial=0.), np.max(-lo, initial=0.), np.max(hi, initial=0.)))
        r, x = np.maximum(values[active], 0.), np.maximum(values[np.asarray(active, dtype=int) + self.k], 0.)
        retained = [i for i, s in enumerate(family) if s in self.mandatory or max(r[i], x[i]) > 1e-9]
        point = dict(supports=canonical(family[i] for i in retained), r=r[retained].tolist(),
                     x=x[retained].tolist())
        physical = prediction_loss(self.prepared, point["supports"], point["r"], point["x"])
        record.update(dual_bound=dual, absolute_gap=abs(float(result.fun) - dual),
                      constraint_violation=primal_violation, dual_stationarity=stationarity,
                      dual_sign_violation=signs, physical_objective=physical,
                      reproduction_error=abs(physical - float(result.fun)))
        record["valid"] = max(primal_violation, stationarity, signs, record["absolute_gap"],
                               record["reproduction_error"]) <= 1e-7
        if not record["valid"]:
            raise RuntimeError(f"fixed LP audit failed: {record}")
        point.update(objective=physical, solve_index=len(self.solves) - 1)
        self.cache[family] = point
        return point


def single_moves(bank, point):
    """Every one-candidate replacement in the common domain, including refit."""
    family = canonical(point["supports"])
    moves = []
    for candidate in bank.pool:
        if candidate in family or candidate in bank.mandatory:
            continue
        if any(not compatible(candidate, s) for s in bank.mandatory):
            continue
        retained = [s for s in family if compatible(s, candidate)]
        fitted = bank.fit([*retained, candidate])
        moves.append(dict(candidate=candidate, point=fitted,
                          gain=point["objective"] - fitted["objective"]))
    return sorted(moves, key=lambda m: (m["point"]["objective"], len(m["point"]["supports"]), m["candidate"]))


def single_closure(bank, initial, *, relative=.001, max_steps=64, seconds=None):
    """A complete-sweep stop is required to certify single-move stagnation."""
    started = perf_counter()
    current = bank.fit(initial)
    path, sweeps = [current], []
    reason = "step_limit"
    for _ in range(max_steps):
        if seconds is not None and perf_counter() - started >= seconds:
            reason = "wall_limit"
            break
        before = len(bank.solves)
        try:
            moves = single_moves(bank, current)
        except RuntimeError as exc:
            sweeps.append(dict(complete=False, error=str(exc)))
            reason = "LP_failure"
            break
        best = moves[0] if moves else None
        threshold = gate(current["objective"], relative)
        accepted = best is not None and best["gain"] > threshold
        sweeps.append(dict(complete=True, current=current["objective"], threshold=threshold,
                           best_gain=None if best is None else best["gain"], accepted=accepted,
                           candidates=[dict(candidate=m["candidate"], objective=m["point"]["objective"],
                                            gain=m["gain"]) for m in moves], new_LPs=len(bank.solves) - before))
        if not accepted:
            reason = "single_stationary"
            break
        current = best["point"]
        path.append(current)
    return dict(point=current, path=path, sweeps=sweeps, stop_reason=reason,
                seconds=perf_counter() - started, relative_gate=relative,
                stationary=reason == "single_stationary")


def all_hierarchies(n):
    """All proper nontrivial laminar families; 4 leaves: 26, 5 leaves: 236."""
    candidates = [s for k in range(2, n) for s in combinations(range(n), k)]
    families = [()]
    for candidate in candidates:
        families += [(*f, candidate) for f in families
                     if len(f) < n - 2 and all(compatible(candidate, s) for s in f)]
    return sorted((canonical(f) for f in families), key=lambda f: (len(f), f))


def rnj_initial(prepared, *, tolerance=.16):
    """Current raw/RX75 RNJ seed, estimated exclusively from P/Q/Y."""
    import pandas as pd
    from rnj_wzzt.estimation.multiscenario import fit_projected_sensitivity
    from rnj_wzzt.graph.rooted_neighbor_joining import rooted_neighbor_joining
    from rnj_wzzt.graph.rooted_hierarchy import rooted_clades
    from research_experiments.rnj.rx75_edge_floor import calibrate_rx75_floor
    labels = list(range(1, prepared.n + 1))
    scenes = [dict(P_terminal=pd.DataFrame(p, columns=labels), Q_terminal=pd.DataFrame(q, columns=labels),
                   drop_target=pd.DataFrame(y, columns=labels))
              for p, q, y in zip(prepared.p, prepared.q, prepared.target)]
    R, X, _, _ = fit_projected_sensitivity(scenes)
    cal = calibrate_rx75_floor(R, X)
    kernel = .75 * R / cal["mu_r"] + .25 * X / cal["mu_x"]
    tree = rooted_neighbor_joining(kernel, np.diag(kernel), labels, 0,
                                  group_tolerance=tolerance * max(float(np.median(np.diag(kernel))), 1e-12))
    family = canonical([*((i,) for i in range(prepared.n)), tuple(range(prepared.n)),
                        *(tuple(v - 1 for v in c) for c in rooted_clades(tree.edges, 0, labels))])
    return dict(family=family, calibration=cal, tolerance=tolerance)
