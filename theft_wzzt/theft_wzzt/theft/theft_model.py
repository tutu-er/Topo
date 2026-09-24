"""Module 2 (version B): fixed-amplitude theft MILP with joint R/X refit.

Promoted and generalized from ``rnj_wzzt_core/tests/meter_theft_pilot/model.py``.
On a GIVEN rooted tree, jointly re-estimate all edge weights r/x and select at
most one theft location per time step. The theft amplitude is a fixed input
(from the master-meter balance route); the products r_e * b_et are linearized
exactly with the standard four-constraint big-M form, so the model stays a MILP
with global optimality certificates.

Conventions: edge weights absorb the squared-voltage factor 2, i.e. the model
predicts ``drop_target = V_root^2 - V_terminal^2``; loads are positive
consumptions in per unit. Every accepted result must be globally optimal and
pass direct, unlinearized feasibility/product/objective checks.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

from theft_wzzt.models.lin_distflow import ohm_to_pu


@dataclass(frozen=True)
class TheftTree:
    """Rooted tree with observed terminals and candidate theft locations."""

    edges: tuple[tuple[int, int], ...]  # oriented (parent, child), away from root
    observed: tuple[int, ...]
    candidates: tuple[int, ...]

    def incidence(self):
        parent = {v: u for u, v in self.edges}
        if len(parent) != len(self.edges):
            raise ValueError("A node has multiple parents")
        nodes = {v for edge in self.edges for v in edge}
        roots = nodes - set(parent)
        if len(roots) != 1 or not self.observed or not self.candidates:
            raise ValueError("Require one root, observed nodes and candidates")
        root = next(iter(roots))
        if (len(set(self.observed)) != len(self.observed)
                or len(set(self.candidates)) != len(self.candidates)
                or not set(self.observed + self.candidates) <= nodes - {root}):
            raise ValueError("Observation/candidate labels must be unique non-root nodes")

        def path(node):
            seen = set()
            while node != root:
                if node in seen or node not in parent:
                    raise ValueError("Tree has a cycle or disconnected node")
                seen.add(node)
                node = parent[node]
            return seen

        for node in nodes:
            path(node)
        z = np.array([[v in path(o) for o in self.observed] for _, v in self.edges], float)
        c = np.array([[v in path(h) for h in self.candidates] for _, v in self.edges], float)
        return z, c


@dataclass(frozen=True)
class TheftData:
    p: np.ndarray          # (T, n) reported terminal P, pu
    q: np.ndarray          # (T, n) reported terminal Q, pu
    y: np.ndarray          # (T, n) squared-voltage drop target, pu^2
    balance: np.ndarray    # (T,) master - sum(submeters) - EXTERNAL loss; signed
    amplitude: np.ndarray  # (T,) fixed nonnegative unmetered-P input
    extra_q: np.ndarray    # (T,) fixed unmetered-Q input, e.g. kappa * amplitude
    voltage_scale: float
    balance_scale: float

    def validate(self, n):
        t = len(self.p)
        if t == 0 or any(np.shape(a) != (t, n) for a in (self.p, self.q, self.y)):
            raise ValueError("P, Q and voltage must have shape (time, observed)")
        if any(np.shape(a) != (t,) for a in (self.balance, self.amplitude, self.extra_q)):
            raise ValueError("Balance and fixed amplitudes must have shape (time,)")
        if not all(np.all(np.isfinite(a)) for a in
                   (self.p, self.q, self.y, self.balance, self.amplitude, self.extra_q)):
            raise ValueError("Inputs must be finite")
        if np.any(self.amplitude < 0):
            raise ValueError("Unmetered active load cannot be negative")
        if not all(np.isfinite(s) and s > 0 for s in (self.voltage_scale, self.balance_scale)):
            raise ValueError("Noise scales must be positive and fixed")


@dataclass
class TheftFit:
    r: np.ndarray
    x: np.ndarray
    selection: np.ndarray  # (T, H) 0/1
    prediction: np.ndarray
    balance_prediction: np.ndarray
    objective: float
    diagnostics: dict

    def selected_locations(self, tree: TheftTree) -> list[int | None]:
        return [
            None if not row.any() else tree.candidates[int(np.argmax(row))]
            for row in self.selection
        ]


def predict(tree: TheftTree, data: TheftData, r, x, selection):
    """Evaluate the ORIGINAL r*b products, independent of auxiliary v variables."""

    z, c = tree.incidence()
    downstream = selection @ c.T
    flow_p = data.p @ z.T + data.amplitude[:, None] * downstream
    flow_q = data.q @ z.T + data.extra_q[:, None] * downstream
    return (flow_p * r + flow_q * x) @ z


def fit_theft_milp(tree: TheftTree, data: TheftData, *, allow_theft=True,
                   force_active=False, fixed_locations=None, fixed_rx=None,
                   weight_bounds=(0.0, 2.0), time_limit=300.0) -> TheftFit:
    """Jointly refit r/x and at-most-one location EACH time using an exact MILP.

    fixed_locations maps time index to a candidate index, or -1 for no source.
    force_active is only for trusted positive amplitude; otherwise H1 includes H0.
    H0 (allow_theft=False) uses the same objective, observations, scales and
    r/x bounds as H1, so the comparison is fair and nested.

    weight_bounds is either a global ``(lower, upper)`` pair or a per-edge
    ``(lower_array, upper_array)`` of length E. Tight per-edge bounds (e.g. from
    a prior mainline R/X fit) drastically strengthen the LP relaxation; loose
    global bounds make the search intractable in practice.
    """

    z, c = tree.incidence()
    e, n = z.shape
    h = len(tree.candidates)
    data.validate(n)
    t = len(data.p)
    lo_arr, hi_arr = (np.asarray(weight_bounds[0], dtype=float),
                      np.asarray(weight_bounds[1], dtype=float))
    if lo_arr.ndim == 0:
        lo_arr = np.full(e, float(lo_arr))
    if hi_arr.ndim == 0:
        hi_arr = np.full(e, float(hi_arr))
    if lo_arr.shape != (e,) or hi_arr.shape != (e,):
        raise ValueError("Per-edge weight bounds must have length E")
    if not (np.all(np.isfinite(lo_arr)) and np.all(np.isfinite(hi_arr))
            and np.all(lo_arr >= 0) and np.all(lo_arr < hi_arr)):
        raise ValueError("Require finite 0 <= lower < upper weight bounds")
    if not np.isfinite(time_limit) or time_limit <= 0:
        raise ValueError("Positive time limit required")
    if force_active and (not allow_theft or np.any(data.amplitude <= 0)):
        raise ValueError("Forced source requires trusted positive amplitude")

    size = 0

    def alloc(shape):
        nonlocal size
        count = int(np.prod(shape))
        result = np.arange(size, size + count).reshape(shape)
        size += count
        return result

    ri, xi = alloc((e,)), alloc((e,))
    si = alloc((t, h))
    vr, vx = alloc((t, e)), alloc((t, e))
    vy, vm = alloc((t, n)), alloc((t,))
    lb, ub, objective = np.zeros(size), np.full(size, np.inf), np.zeros(size)
    integrality = np.zeros(size, dtype=np.uint8)
    for ids in (ri, xi):
        lb[ids], ub[ids] = lo_arr, hi_arr
    ub[si] = int(allow_theft)
    integrality[si] = 1
    # A zero-amplitude location has no meaning and must not be reported as detected.
    ub[si[data.amplitude == 0]] = 0
    ub[vr], ub[vx] = hi_arr, hi_arr
    objective[vy], objective[vm] = 1 / data.voltage_scale, 1 / data.balance_scale
    if fixed_rx is not None:
        for ids, values in zip((ri, xi), fixed_rx, strict=True):
            values = np.asarray(values)
            if values.shape != (e,) or np.any(values < lo_arr) or np.any(values > hi_arr):
                raise ValueError("Fixed parameters outside common bounds")
            lb[ids] = ub[ids] = values
    for k, location in (fixed_locations or {}).items():
        if not 0 <= k < t or not -1 <= location < h:
            raise ValueError("Invalid fixed location")
        if location >= 0 and (not allow_theft or data.amplitude[k] == 0):
            raise ValueError("Cannot force an inactive source")
        if force_active and location == -1:
            raise ValueError("Conflicting source constraints")
        ub[si[k]] = 0
        if location >= 0:
            lb[si[k, location]] = ub[si[k, location]] = 1

    row_ids, col_ids, values, row_lb, row_ub = [], [], [], [], []

    def add(coeffs, low=-np.inf, high=np.inf):
        row = len(row_lb)
        for col, value in coeffs.items():
            if value != 0:
                row_ids.append(row)
                col_ids.append(int(col))
                values.append(float(value))
        row_lb.append(low)
        row_ub.append(high)

    for k in range(t):
        add(dict.fromkeys(si[k], 1), low=int(force_active), high=1)
        for edge in range(e):
            lower, upper = lo_arr[edge], hi_arr[edge]
            b = {si[k, j]: c[edge, j] for j in range(h) if c[edge, j]}
            for weight, product in ((ri[edge], vr[k, edge]), (xi[edge], vx[k, edge])):
                # lower*b <= v <= upper*b; r-upper*(1-b) <= v <= r-lower*(1-b)
                add({product: 1, **{j: -lower * v for j, v in b.items()}}, low=0)
                add({product: 1, **{j: -upper * v for j, v in b.items()}}, high=0)
                add({product: 1, weight: -1, **{j: -upper * v for j, v in b.items()}}, low=-upper)
                add({product: 1, weight: -1, **{j: -lower * v for j, v in b.items()}}, high=-lower)
        for i in range(n):
            coeffs = {}
            for edge in range(e):
                if z[edge, i]:
                    coeffs[ri[edge]] = float(z[edge] @ data.p[k])
                    coeffs[xi[edge]] = float(z[edge] @ data.q[k])
                    coeffs[vr[k, edge]] = data.amplitude[k]
                    coeffs[vx[k, edge]] = data.extra_q[k]
            add({**coeffs, vy[k, i]: -1}, high=data.y[k, i])
            add({**coeffs, vy[k, i]: 1}, low=data.y[k, i])
        coeffs = dict.fromkeys(si[k], data.amplitude[k])
        add({**coeffs, vm[k]: -1}, high=data.balance[k])
        add({**coeffs, vm[k]: 1}, low=data.balance[k])

    matrix = coo_matrix((values, (row_ids, col_ids)), shape=(len(row_lb), size)).tocsc()
    started = perf_counter()
    result = milp(objective, integrality=integrality, bounds=Bounds(lb, ub),
                  constraints=LinearConstraint(matrix, row_lb, row_ub),
                  options={"time_limit": time_limit, "mip_rel_gap": 1e-9})
    elapsed = perf_counter() - started
    if result.status != 0 or not result.success or result.x is None:
        raise RuntimeError(f"Search incomplete or infeasible: {result.message}")
    raw = result.x
    selection = np.rint(raw[si]).astype(int)
    mu = predict(tree, data, raw[ri], raw[xi], selection)
    balance_mu = data.amplitude * selection.sum(axis=1)
    score = float(np.sum(np.abs(data.y - mu)) / data.voltage_scale
                  + np.sum(np.abs(data.balance - balance_mu)) / data.balance_scale)
    activity = matrix @ raw
    feasibility = max(float(np.max(np.maximum(np.asarray(row_lb) - activity, 0))),
                      float(np.max(np.maximum(activity - np.asarray(row_ub), 0))),
                      float(np.max(np.maximum(lb - raw, 0))),
                      float(np.max(np.maximum(raw - ub, 0))),
                      float(np.max(np.abs(raw[si] - selection))))
    b = selection @ c.T
    products = max(float(np.max(np.abs(raw[vr] - b * raw[ri]))),
                   float(np.max(np.abs(raw[vx] - b * raw[xi]))))
    gap = float(getattr(result, "mip_gap", 0) or 0)
    # HiGHS internal scaling makes the reported objective differ from a direct
    # recomputation by ~1e-5 relative; the check guards correctness, not rounding.
    if (feasibility > 2e-6 or products > 2e-6 or gap > 1e-7
            or not np.isclose(score, result.fun, atol=1e-4, rtol=1e-4)):
        raise RuntimeError(
            "Direct feasibility/product/objective check failed: "
            f"feasibility={feasibility:.3e} products={products:.3e} gap={gap:.3e} "
            f"score={score:.9e} fun={float(result.fun):.9e}"
        )
    diagnostics = {"status": int(result.status), "message": result.message,
                   "mip_gap": gap, "dual_bound": float(getattr(result, "mip_dual_bound", result.fun)),
                   "seconds": elapsed, "variables": size, "binary_variables": int(si.size),
                   "constraints": len(row_lb), "max_feasibility_error": feasibility,
                   "max_product_error": products, "direct_objective": score}
    return TheftFit(raw[ri].copy(), raw[xi].copy(), selection, mu, balance_mu,
                    score, diagnostics)


def tree_from_network(net, candidates: tuple[int, ...] | None = None) -> TheftTree:
    """Build a TheftTree from a TerminalizedNetwork (true or identified tree)."""

    oriented = net.orient_from_root()
    edges = tuple((int(r.parent), int(r.child)) for r in oriented.itertuples())
    observed = tuple(
        net.buses.loc[net.buses["bus_type"].eq("observed_terminal"), "bus_id"].astype(int)
    )
    if candidates is None:
        hidden = net.buses.loc[net.buses["bus_type"].eq("hidden_internal"), "bus_id"].astype(int)
        candidates = tuple(hidden) + tuple(observed)
    return TheftTree(edges, observed, tuple(candidates))


def nominal_edge_weights(net) -> tuple[np.ndarray, np.ndarray]:
    """Squared-voltage edge weights (2*r_pu, 2*x_pu) in tree edge order."""

    oriented = net.orient_from_root()
    r = np.array([ohm_to_pu(net, float(v)) for v in oriented["r_ohm"]])
    x = np.array([ohm_to_pu(net, float(v)) for v in oriented["x_ohm"]])
    return 2.0 * r, 2.0 * x


def data_from_scenario(scenario: dict, *, loss_p_estimate=None, loss_q_estimate=None,
                       voltage_scale: float | None = None,
                       balance_scale: float | None = None) -> TheftData:
    """Build TheftData from a theft_simulation scenario.

    The amplitude is fixed to the clipped master-meter balance
    ``clip(P0_measured - sum(P_terminal) - loss_estimate, 0, None)``. The default
    loss estimate is the counterfactual no-theft technical loss, an IDEALIZED
    external estimate; loss-estimation error sensitivity is a separate experiment.
    The signed balance is always preserved for module-1 scoring.
    """

    p = scenario["P_terminal"].to_numpy(dtype=float)
    q = scenario["Q_terminal"].to_numpy(dtype=float)
    y = scenario["drop_target"].to_numpy(dtype=float)
    loss_p = scenario["loss_p_notheft"] if loss_p_estimate is None else loss_p_estimate
    loss_q = scenario["loss_q_notheft"] if loss_q_estimate is None else loss_q_estimate
    balance = (scenario["P0_measured"] - scenario["P_terminal"].sum(axis=1)
               - pd.Series(loss_p, index=scenario["P_terminal"].index)).to_numpy(dtype=float)
    q_balance = (scenario["Q0_measured"] - scenario["Q_terminal"].sum(axis=1)
                 - pd.Series(loss_q, index=scenario["Q_terminal"].index)).to_numpy(dtype=float)
    amplitude = np.maximum(balance, 0.0)
    extra_q = np.maximum(q_balance, 0.0)
    settings = scenario.get("scenario_settings", {})
    if voltage_scale is None:
        root_mean = float(settings.get("root_mean", 1.02))
        v_noise = float(settings.get("v_noise_rel", 0.0002))
        voltage_scale = 4.0 * root_mean * v_noise  # 2*V*sigma per squaring, margin 2
    if balance_scale is None:
        master = float(settings.get("master_noise_rel", 0.002))
        pq = float(settings.get("pq_noise_rel", 0.005))
        p0_level = float(np.abs(scenario["P0_measured"]).mean())
        pq_sum = pq * float(np.sqrt(np.square(scenario["P_terminal"]).sum(axis=1).mean()))
        balance_scale = 2.0 * (master * p0_level + pq_sum)
    return TheftData(p, q, y, balance, amplitude, extra_q,
                     float(voltage_scale), float(balance_scale))
