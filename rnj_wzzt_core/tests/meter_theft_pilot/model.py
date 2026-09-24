"""Fixed-amplitude wzzT extension on a GIVEN rooted tree, for tiny experiments.

No intercepts, centering, unknown P/Q errors or AC equations are introduced.
Edge r/x already include the coefficient in the chosen voltage-drop convention.
Every accepted result must be optimal and pass direct, unlinearized checks.
"""

from dataclasses import dataclass
from time import perf_counter

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix


@dataclass(frozen=True)
class Tree:
    edges: tuple[tuple[str, str], ...]
    observed: tuple[str, ...]
    candidates: tuple[str, ...]

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
class Data:
    p: np.ndarray
    q: np.ndarray
    y: np.ndarray
    balance: np.ndarray  # master - sum(submeters) - EXTERNAL loss; signed
    amplitude: np.ndarray  # fixed nonnegative input for this solve
    extra_q: np.ndarray  # fixed too, e.g. known kappa * amplitude
    voltage_scale: float = 0.002
    balance_scale: float = 0.03

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
class Fit:
    r: np.ndarray
    x: np.ndarray
    selection: np.ndarray
    prediction: np.ndarray
    balance_prediction: np.ndarray
    objective: float
    diagnostics: dict


def predict(tree, data, r, x, selection):
    """Evaluate the ORIGINAL r*b products, independent of auxiliary v variables."""
    z, c = tree.incidence()
    downstream = selection @ c.T
    flow_p = data.p @ z.T + data.amplitude[:, None] * downstream
    flow_q = data.q @ z.T + data.extra_q[:, None] * downstream
    return (flow_p * r + flow_q * x) @ z


def fit(tree, data, *, allow_theft=True, force_active=False, fixed_locations=None,
        fixed_rx=None, weight_bounds=(0.01, 0.3), time_limit=30.0):
    """Jointly refit r/x and at-most-one location EACH time using an exact MILP.

    fixed_locations maps time to a candidate index, or -1 for no source.
    force_active is only for trusted positive amplitude; otherwise H1 includes H0.
    H0 uses the same objective, observations, scales and r/x bounds as H1.
    """
    z, c = tree.incidence()
    e, n = z.shape
    h = len(tree.candidates)
    data.validate(n)
    t = len(data.p)
    lower, upper = weight_bounds
    if not (np.isfinite(lower) and np.isfinite(upper) and 0 <= lower < upper):
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
        lb[ids], ub[ids] = lower, upper
    ub[si] = int(allow_theft)
    integrality[si] = 1
    # A zero-amplitude location has no meaning and must not be reported as detected.
    ub[si[data.amplitude == 0]] = 0
    ub[vr], ub[vx] = upper, upper
    objective[vy], objective[vm] = 1 / data.voltage_scale, 1 / data.balance_scale
    if fixed_rx is not None:
        for ids, values in zip((ri, xi), fixed_rx, strict=True):
            values = np.asarray(values)
            if values.shape != (e,) or np.any(values < lower) or np.any(values > upper):
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
    if (feasibility > 2e-6 or products > 2e-6 or gap > 1e-7
            or not np.isclose(score, result.fun, atol=1e-5, rtol=1e-7)):
        raise RuntimeError("Direct feasibility/product/objective check failed")
    diagnostics = {"status": int(result.status), "message": result.message,
                   "mip_gap": gap, "dual_bound": float(getattr(result, "mip_dual_bound", result.fun)),
                   "seconds": elapsed, "variables": size, "binary_variables": int(si.size),
                   "constraints": len(row_lb), "max_feasibility_error": feasibility,
                   "max_product_error": products, "direct_objective": score}
    return Fit(raw[ri].copy(), raw[xi].copy(), selection, mu, balance_mu, score, diagnostics)
