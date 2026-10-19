"""General terminal-tree AC evidence layer; research interface, not production.

Known common nonnegative X/R, source at the first common junction, balanced
constant-power AC and a declared voltage lower bound.  A positive verified
outer-LP lower bound excludes one specified candidate in this model domain.
An LP feasible point, a zero lower bound, or a failed local AC fit certifies
neither original AC feasibility nor topology correctness.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction as F
from time import perf_counter

import numpy as np
from scipy.optimize import least_squares, linprog
from scipy.sparse import csr_matrix

# Reuse only the already audited directed rounding / square-root primitives.
# The v8 module and its archived protocol are not modified by this module.
from ac_shared_v8 import sqrt_up
from local_shared_source_v6 import _down, _up


def f(value):
    return value if isinstance(value, F) else F(float(value))


@dataclass(frozen=True)
class RadialCandidate:
    """Canonical physical tree induced by a laminar terminal-support family.

    Edge e ends at node e+1; node 0 is the voltage-controlled grid root.
    A full-set edge (index 0) leads to the common source junction.  Singleton
    edges end at observed terminals.  Every other edge ends at a zero-load
    junction. Supports must use contiguous, zero-based terminal indices.
    Unary chains with identical descendant sets are aggregated; their internal
    unobserved injection/impedance distribution is outside this interface.
    """
    name: str
    n: int
    supports: tuple[tuple[int, ...], ...]
    parents: tuple[int, ...]
    terminal_edges: tuple[int, ...]
    B: np.ndarray = field(repr=False, compare=False)
    M: np.ndarray = field(repr=False, compare=False)
    descendants: tuple[tuple[int, ...], ...]

    @property
    def m(self):
        return len(self.supports)

    @property
    def edges(self):
        return tuple((0 if parent < 0 else parent + 1, e + 1)
                     for e, parent in enumerate(self.parents))

    @classmethod
    def from_supports(cls, name, n_terminal, supports=()):
        n = int(n_terminal)
        if n != n_terminal or n < 2:
            raise ValueError("At least two zero-based terminals are required")
        universe = frozenset(range(n))
        family = {universe, *(frozenset((j,)) for j in range(n))}
        for raw in supports:
            values = tuple(raw)
            if any(int(j) != j for j in values):
                raise ValueError("Support indices must be integers")
            item = frozenset(int(j) for j in values)
            if not item or not item <= universe:
                raise ValueError("Empty or out-of-range support")
            family.add(item)
        for left in family:
            for right in family:
                if left & right and not (left <= right or right <= left):
                    raise ValueError("Crossing supports do not define a radial tree")
        ordered = tuple(sorted(family, key=lambda s: (-len(s), tuple(sorted(s)))))
        parents = tuple(-1 if item == universe else min(
            (e for e, sup in enumerate(ordered) if item < sup),
            key=lambda e: len(ordered[e])) for item in ordered)
        terminal_edges = tuple(ordered.index(frozenset((j,))) for j in range(n))
        b = np.array([[int(child <= ancestor) for ancestor in ordered]
                      for child in ordered], dtype=int)
        a = b[list(terminal_edges)].copy()
        b.setflags(write=False)
        a.setflags(write=False)
        descendants = tuple(tuple(g for g in range(len(ordered))
                                  if g != e and b[g, e]) for e in range(len(ordered)))
        return cls(str(name), n, tuple(tuple(sorted(s)) for s in ordered),
                   parents, terminal_edges, b, a, descendants)

    def metadata(self):
        return dict(name=self.name, n_terminal=self.n, supports=self.supports,
                    parents=self.parents, terminal_edges=self.terminal_edges,
                    source_node=1, root_voltage=1.0,
                    edge_parameter="theta_e=2*r_e; x_e=kappa*r_e")


def _physical_arrays(candidate, theta, p, q, source_p, source_q, kappa):
    theta = np.asarray(theta, float)
    p, q = np.asarray(p, float), np.asarray(q, float)
    if p.ndim == 1:
        p, q = p[None, :], q[None, :]
    if (theta.shape != (candidate.m,) or p.ndim != 2 or q.shape != p.shape
            or p.shape[1] != candidate.n or not len(p)):
        raise ValueError("AC parameter/measurement shape mismatch")
    sp = np.broadcast_to(np.asarray(source_p, float), (len(p),))
    sq = np.broadcast_to(np.asarray(source_q, float), (len(p),))
    if (not all(np.isfinite(v).all() for v in (theta, p, q, sp, sq))
            or (theta < 0).any() or not np.isfinite(float(kappa)) or kappa < 0):
        raise ValueError("Nonfinite AC input or negative impedance ratio/length")
    loads = np.zeros((len(p), candidate.m), complex)
    loads[:, 0] = sp + 1j * sq
    loads[:, candidate.terminal_edges] += p + 1j * q
    z = theta * (1 + 1j * float(kappa)) / 2
    return theta, p, q, sp, sq, loads, z


def ac_batch(candidate, theta, p, q, source_p, source_q, *, kappa=.5):
    """Batched numerical AC witness; checked residual, not interval existence."""
    theta, p, q, sp, sq, loads, z = _physical_arrays(
        candidate, theta, p, q, source_p, source_q, kappa)
    zbus = (candidate.B * z) @ candidate.B.T
    voltage = np.ones_like(loads)
    for _ in range(2000):
        new = 1 - np.conj(loads / voltage) @ zbus.T
        if not np.isfinite(new).all():
            raise RuntimeError("Nonfinite AC fixed-point iterate")
        if np.max(abs(new - voltage)) < 2e-14:
            voltage = new
            break
        voltage = new
    else:
        raise RuntimeError("AC fixed point did not converge; no exclusion follows")
    residual = float(np.max(abs(voltage - 1 + np.conj(loads / voltage) @ zbus.T)))
    if residual > 1e-11:
        raise RuntimeError("AC residual exceeds numerical witness tolerance")
    return 1 - abs(voltage[:, candidate.terminal_edges])**2, voltage, residual


def ac_forward(candidate, theta, p, q, source_p, source_q, *, kappa=.5):
    theta, pb, qb, sp, sq, loads, z = _physical_arrays(
        candidate, theta, p, q, source_p, source_q, kappa)
    if len(pb) != 1:
        raise ValueError("ac_forward expects a single time sample")
    y, voltage, residual = ac_batch(candidate, theta, p, q, source_p,
                                    source_q, kappa=kappa)
    current = candidate.B.T @ np.conj(loads[0] / voltage[0])
    ell = abs(current)**2
    eta = np.array([abs(z[e])**2 * ell[e] + 2 * sum(
        (z[e].real * z[g].real + z[e].imag * z[g].imag) * ell[g]
        for g in candidate.descendants[e]) for e in range(candidate.m)])
    linear = candidate.M @ (theta * (candidate.M.T @ (pb[0] + float(kappa)*qb[0])))
    linear += theta[0] * (sp[0] + float(kappa)*sq[0])
    error = float(np.max(abs(y[0] - linear - candidate.M @ eta)))
    if error > 1e-11:
        raise RuntimeError("Exact DistFlow identity mismatch")
    return dict(Y=y[0], voltage=voltage[0], ell=ell, eta=eta,
                equation_residual=residual, identity_error=error)


def _observations(observed, candidate, box, kappa, vmin):
    p, q, y = (np.asarray(observed[k], float) for k in ("P", "Q", "Y_drop"))
    if p.ndim != 2 or q.shape != p.shape or y.shape != p.shape or not len(p) or p.shape[1] != candidate.n:
        raise ValueError("P, Q, Y_drop must have shape (time, n_terminal)")
    shape, count = p.shape, len(p)
    values = dict(P=p, Q=q, Y_drop=y)
    for key in ("P_error", "Q_error", "drop_error_bound"):
        values[key] = np.broadcast_to(np.asarray(observed.get(key, 0.), float), shape)
    for key in ("source_p_abs_cap", "source_q_abs_cap", "source_nominal_eff", "source_error_eff"):
        values[key] = np.broadcast_to(np.asarray(observed[key], float), (count,))
    for key, value in values.items():
        if not np.isfinite(value).all():
            raise ValueError(f"Nonfinite {key}")
        if ("error" in key or "cap" in key) and np.min(value) < 0:
            raise ValueError(f"Negative bound {key}")
    bounds = np.asarray(box, float)
    if (bounds.shape != (candidate.m, 2) or not np.isfinite(bounds).all()
            or (bounds[:, 0] < 0).any() or (bounds[:, 1] < bounds[:, 0]).any()):
        raise ValueError("One finite nonnegative ordered theta box per edge is required")
    if not (np.isfinite(float(kappa)) and float(kappa) >= 0 and 0 < float(vmin) <= 1):
        raise ValueError("Known nonnegative X/R and 0 < vmin <= 1 required")
    return values, tuple((f(lo), f(hi)) for lo, hi in bounds), f(kappa), f(vmin)


def envelope(observed, candidate, box, *, kappa=.5, vmin=.9):
    """Rational current caps and two valid affine loss upper planes per edge."""
    # Y/error fields are immaterial here, but validation remains identical.
    data = dict(observed)
    data.setdefault("Y_drop", np.zeros_like(data["P"], dtype=float))
    data.setdefault("source_nominal_eff", 0.)
    data.setdefault("source_error_eff", 0.)
    data, bounds, ratio, vfloor = _observations(data, candidate, box, kappa, vmin)
    p, q, a = data["P"], data["Q"], candidate.M
    factor = (1 + ratio**2) / 4
    planes, caps = [], []
    for t in range(len(p)):
        s = [sqrt_up((abs(f(p[t, j])) + f(data["P_error"][t, j]))**2 +
                     (abs(f(q[t, j])) + f(data["Q_error"][t, j]))**2) for j in range(candidate.n)]
        source = sqrt_up(f(data["source_p_abs_cap"][t])**2 + f(data["source_q_abs_cap"][t])**2)
        ell = [((sum((s[j] for j in range(candidate.n) if a[j, e]), F()) +
                 (source if e == 0 else F())) / vfloor)**2 for e in range(candidate.m)]
        caps.append(ell)
        edge_planes = []
        for e in range(candidate.m):
            both = []
            lo, hi = bounds[e]
            for orientation in (0, 1):
                co, const = [F()] * candidate.m, -ell[e]*lo*hi
                co[e] = ell[e]*(lo+hi)
                for g in candidate.descendants[e]:
                    lg, ug = bounds[g]
                    if orientation == 0:
                        co[e] += 2*ell[g]*ug
                        co[g] += 2*ell[g]*lo
                        const -= 2*ell[g]*lo*ug
                    else:
                        co[e] += 2*ell[g]*lg
                        co[g] += 2*ell[g]*hi
                        const -= 2*ell[g]*hi*lg
                both.append((tuple(factor*v for v in co), factor*const))
            edge_planes.append(both)
        planes.append(edge_planes)
    return planes, caps


def solve_outer(observed, candidate, *, box, kappa=.5, vmin=.9,
                sharing="path", linear_only=False, time_limit=None,
                shared_voltage_gain=None):
    """Verified lower/upper bounds on minimum extra slack of an AC outer LP.

    upper bounds the OUTER problem only; no AC feasibility inference follows.
    shared_voltage_gain optionally declares fixed per-meter gains across ALL
    samples, with bounded amplitude jitter; it replaces independent voltage
    error boxes. Root physical magnitude remains exactly one. Unknown common
    root-gain relationships are outer-relaxed, never fixed to fitted values.
    linear_only is an intentionally invalid negative control, never a valid
    AC candidate exclusion rule. Numerical failure always returns unresolved.
    """
    began = perf_counter()
    if sharing not in ("path", "output"):
        raise ValueError("Unknown sharing mode")
    data, bounds, ratio, _ = _observations(observed, candidate, box, kappa, vmin)
    planes, caps = envelope(data, candidate, box, kappa=kappa, vmin=vmin)
    p, q, y, a = data["P"], data["Q"], data["Y_drop"], candidate.M
    count, m, n = len(p), candidate.m, candidate.n
    budget, nominal = data["source_error_eff"], data["source_nominal_eff"]
    nk = m if sharing == "path" else n
    basevar = m + count + count*nk
    calibrated = shared_voltage_gain is not None
    nvar = basevar + (2*n if calibrated else 0) + 1
    lower = [v[0] for v in bounds] + [-f(v)*bounds[0][1] for v in budget]
    upper = [v[1] for v in bounds] + [f(v)*bounds[0][1] for v in budget]
    eta_caps = []
    for t in range(count):
        ec = [min(sum((co[k]*bounds[k][1] for k in range(m)), const)
                  for co, const in planes[t][e]) for e in range(m)]
        eta_caps.extend(ec if sharing == "path" else
                        [sum((ec[e] for e in range(m) if a[i, e]), F()) for i in range(n)])
    if linear_only:
        eta_caps = [F()]*len(eta_caps)
    # Finite cap guaranteed to contain the feasible theta midpoint,w=eta=0
    # witness after increasing tau; avoids a silent fixed tau<=1 restriction.
    cap = F(1) + max(abs(f(y[t, i])) + sum(
        bounds[e][1]*(sum((abs(f(p[t, j])) + ratio*abs(f(q[t, j]))
                          for j in range(n) if a[j, e]), F()) +
                      (abs(f(nominal[t])) if e == 0 else F()))
        for e in range(m) if a[i, e]) for t in range(count) for i in range(n))
    lower += [F()]*len(eta_caps)
    upper += eta_caps
    gain_meta = None
    if calibrated:
        glo, ghi = map(f, shared_voltage_gain["gain_bounds"])
        jitter0 = f(shared_voltage_gain["root_jitter"])
        jitter = np.broadcast_to(np.asarray(shared_voltage_gain["terminal_jitter"], float), (n,))
        vmax = f(shared_voltage_gain["vmax"])
        if (not 0 < glo <= ghi or jitter0 < 0 or not np.isfinite(jitter).all()
                or np.min(jitter) < 0 or vmax < f(vmin)):
            raise ValueError("Positive gain bounds and valid voltage/jitter bounds required")
        hmin, hmax = 1/ghi**2, 1/glo**2
        bmin, bmax = 1-hmax*ghi**2, 1-hmin*glo**2
        # Keep h_i multiplying this known jitter cap: this is still linear
        # and avoids replacing every h_i by the less informative hmax.
        jitter_cap = [2*ghi*jitter0+jitter0**2 +
                      2*ghi*vmax*f(ji)+f(ji)**2 for ji in jitter]
        lower += [hmin]*n + [bmin]*n
        upper += [hmax]*n + [bmax]*n
        cap += hmax*max(abs(f(value)) for value in y.ravel()) + max(abs(bmin), abs(bmax))
        gain_meta = dict(gain_bounds=[float(glo), float(ghi)], root_jitter=float(jitter0),
                         terminal_jitter=jitter.tolist(), vmax=float(vmax),
                         shared_across_all_rows=True,
                         common_root_gain_relation="outer relaxed separately for each terminal")
    lower += [F()]
    upper += [cap]
    rows, rhs, exactrows, exactrhs, guards = [], [], [], [], []

    def add(coefficients, value):
        co = {k: v for k, v in coefficients.items() if v}
        rounded = {k: float(v) for k, v in co.items()}
        guard = sum((abs(f(rounded[k])-v)*max(abs(lower[k]), abs(upper[k]))
                     for k, v in co.items()), F())
        rows.append(rounded)
        rhs.append(_up(value+guard))
        exactrows.append(co)
        exactrhs.append(value)
        guards.append(guard)

    if calibrated:
        for i in range(n):
            # 1-h_i*ghi^2 <= b_i <= 1-h_i*glo^2.
            add({basevar+i: glo**2, basevar+n+i: F(1)}, F(1))
            add({basevar+i: -ghi**2, basevar+n+i: F(-1)}, F(-1))
    for t in range(count):
        wcol = m+t
        for sign in (1, -1):
            add({0: -f(budget[t]), wcol: F(sign)}, F())
        for e in range(nk):
            included = [e] if sharing == "path" else [g for g in range(m) if a[e, g]]
            for orientation in (0, 1):
                co = [sum((planes[t][g][orientation][0][k] for g in included), F()) for k in range(m)]
                const = sum((planes[t][g][orientation][1] for g in included), F())
                add({**{k: -v for k, v in enumerate(co) if v}, m+count+t*nk+e: F(1)}, const)
        for i in range(n):
            pred = {e: sum((f(p[t, j]) + ratio*f(q[t, j]) for j in range(n) if a[j, e]), F())
                    for e in range(m) if a[i, e]}
            pred[0] += f(nominal[t])
            pred[wcol] = F(1)
            for e in range(nk):
                if (sharing == "path" and a[i, e]) or (sharing == "output" and i == e):
                    pred[m+count+t*nk+e] = F(1)
            eiv = {e: sum((f(data["P_error"][t, j]) + ratio*f(data["Q_error"][t, j])
                            for j in range(n) if a[j, e]), F()) for e in range(m) if a[i, e]}
            if calibrated:
                pred[basevar+i] = -f(y[t, i])
                pred[basevar+n+i] = F(-1)
            for sign in (1, -1):
                co = {k: sign*v for k, v in pred.items()}
                for k, v in eiv.items():
                    co[k] = co.get(k, F())-v
                co[nvar-1] = F(-1)
                if calibrated:
                    co[basevar+i] = co.get(basevar+i, F())-jitter_cap[i]
                value = F() if calibrated else sign*f(y[t, i])+f(data["drop_error_bound"][t, i])
                add(co, value)
    rr, cc, vv = [], [], []
    for i, row in enumerate(rows):
        for k, value in row.items():
            rr.append(i); cc.append(k); vv.append(value)
    matrix = csr_matrix((vv, (rr, cc)), shape=(len(rows), nvar))
    objective = np.r_[np.zeros(nvar-1), 1.]
    lo = np.array([_down(v) for v in lower])
    hi = np.array([_up(v) for v in upper])
    options = dict(primal_feasibility_tolerance=1e-9, dual_feasibility_tolerance=1e-9)
    if time_limit is not None:
        options["time_limit"] = float(time_limit)
    result = linprog(objective, A_ub=matrix, b_ub=rhs, bounds=list(zip(lo, hi)),
                     method="highs", options=options)
    common = dict(candidate=candidate.name, sharing=sharing, linear_only=linear_only,
                  solver_status=int(result.status), solver_message=result.message,
                  max_rounding_guard=float(max(guards, default=F())),
                  n_variable=nvar, n_constraint=len(rows),
                  certificate_scope=("invalid linear negative control; no valid AC exclusion"
                                     if linear_only else
                                     "one candidate in stated balanced AC/source/measurement domain"),
                  certificate_valid_for_AC=not linear_only,
                  outer_feasibility_is_not_AC_feasibility=True,
                  voltage_model="fixed_gain_bounded_jitter" if calibrated else "independent_drop_boxes",
                  shared_voltage_gain=gain_meta)
    if not result.success or not np.isfinite(result.x).all():
        return dict(common, status="solver_failure_unresolved", lower=None, upper=None,
                    gap=None, seconds=perf_counter()-began)
    # Weak duality on the coefficient-guarded, outward-rounded LP.  Clipping
    # multipliers to <=0 and minimizing every reduced cost over the finite box
    # makes this valid even if floating-point stationarity is imperfect.
    lam = [f(min(0., v)) for v in result.ineqlin.marginals]
    reduced = [F()]*nvar
    reduced[-1] = F(1)
    bound = sum((f(b)*v for b, v in zip(rhs, lam)), F())
    for row, v in zip(rows, lam):
        if v:
            for k, coefficient in row.items():
                reduced[k] -= f(coefficient)*v
    bound += sum((min(v*f(l), v*f(u)) for v, l, u in zip(reduced, lo, hi)), F())
    bound = max(F(), bound)
    # Exact feasible repair for the ORIGINAL rational outer model.
    point = [min(u, max(l, f(v))) for v, l, u in zip(result.x, lower, upper)]
    if calibrated:
        for i in range(n):
            hcol, bcol = basevar+i, basevar+n+i
            point[bcol] = min(1-point[hcol]*glo**2, max(1-point[hcol]*ghi**2, point[bcol]))
    for t in range(count):
        wc = m+t
        limit = f(budget[t])*point[0]
        point[wc] = min(limit, max(-limit, point[wc]))
        for e in range(nk):
            included = [e] if sharing == "path" else [g for g in range(m) if a[e, g]]
            limits = [sum((sum((co[k]*point[k] for k in range(m)), c)
                           for g in included for co, c in [planes[t][g][orientation]]), F())
                      for orientation in (0, 1)]
            index = m+count+t*nk+e
            point[index] = max(F(), min(point[index], *limits))
    point[-1] = F()
    for row, b in zip(exactrows, exactrhs):
        violation = sum((v*point[k] for k, v in row.items()), F())-b
        if row.get(nvar-1) == -1:
            point[-1] += max(F(), violation)
        elif violation > 0:
            raise RuntimeError("Exact non-observation repair failed")
    ub = point[-1]
    if ub > cap or bound > ub:
        raise RuntimeError("Invalid exact certificate interval")
    if not all(sum((v*point[k] for k, v in row.items()), F()) <= b
               for row, b in zip(exactrows, exactrhs)):
        raise RuntimeError("Exact primal verification failed")
    answer = dict(common, status="excluded" if bound > 0 else "unresolved", lower=_down(bound),
                upper=_up(ub), gap=_up(ub-bound), theta=list(map(float, point[:m])),
                exact_lower=str(bound), exact_upper=str(ub), exact_primal_verified=True,
                exact_box_dual_verified=True, max_current_cap=float(max(max(v) for v in caps)),
                seconds=perf_counter()-began)
    if calibrated:
        answer["inverse_gain_squared"] = list(map(float, point[basevar:basevar+n]))
        answer["gain_offset"] = list(map(float, point[basevar+n:basevar+2*n]))
    return answer


def fit_ac_reference(history, candidate, *, box, kappa=.5, vmin=.9,
                     vmax=1.1, source_q_ratio=.3, max_nfev=100, outer_result=None):
    """Two-start local numerical AC fit used for design, NEVER for exclusion.

    Source Q/P is fixed only for this representative fit, not for solve_outer.
    P,Q equal their measured centers (which lie inside their declared boxes).
    Every returned witness is checked against original error/source/voltage
    bounds. Failure leaves the candidate unresolved.
    """
    began = perf_counter()
    data, bounds, ratio, _ = _observations(history, candidate, box, kappa, vmin)
    count, m = len(data["P"]), candidate.m
    qratio = float(source_q_ratio)
    if not np.isfinite(qratio) or qratio < 0:
        raise ValueError("Nonnegative finite representative source Q/P required")
    eff = 1 + float(kappa)*qratio
    splo = np.maximum(-data["source_p_abs_cap"],
                     (data["source_nominal_eff"]-data["source_error_eff"])/eff)
    sphi = np.minimum(data["source_p_abs_cap"],
                     (data["source_nominal_eff"]+data["source_error_eff"])/eff)
    if qratio > 0:
        splo = np.maximum(splo, -data["source_q_abs_cap"]/qratio)
        sphi = np.minimum(sphi, data["source_q_abs_cap"]/qratio)
    low = np.r_[[float(b[0]) for b in bounds], splo]
    high = np.r_[[float(b[1]) for b in bounds], sphi]
    common = dict(candidate=candidate.name, source_q_ratio=qratio,
                  witness_scope="numerical AC residual <=1e-11; not interval existence",
                  interpretation="local design representative; fit failure never excludes")
    if np.any(low > high):
        return dict(common, historical_numeric_witness=False, status="no_representative_source_domain",
                    theta=None, seconds=perf_counter()-began)
    free = high > low
    initial_source = np.clip(data["source_nominal_eff"]/eff, splo, sphi)
    starts = [np.r_[(low[:m]+high[:m])/2, initial_source]]
    if outer_result is None:
        outer_result = solve_outer(data, candidate, box=box, kappa=kappa, vmin=vmin)
    if outer_result.get("theta") is not None:
        starts.append(np.r_[np.clip(outer_result["theta"], low[:m], high[:m]), initial_source])
    scale = max(float(np.max(data["drop_error_bound"])), 1e-4)

    def unpack(z):
        x = low.copy()
        x[free] = z
        return x

    def residual(z):
        x = unpack(z)
        try:
            pred, _, _ = ac_batch(candidate, x[:m], data["P"], data["Q"], x[m:],
                                  qratio*x[m:], kappa=kappa)
            return ((pred-data["Y_drop"])/scale).ravel()
        except RuntimeError:
            return np.full(count*candidate.n, 1e6)
    fits = []
    for start in starts:
        try:
            if free.any():
                result = least_squares(residual, start[free], bounds=(low[free], high[free]),
                                       ftol=1e-10, xtol=1e-10, gtol=1e-10, max_nfev=max_nfev)
                x, success = unpack(result.x), bool(result.success)
            else:
                x, success = low.copy(), True
            pred, voltage, equation_error = ac_batch(candidate, x[:m], data["P"], data["Q"],
                                                     x[m:], qratio*x[m:], kappa=kappa)
            fit_error = abs(pred-data["Y_drop"])
            source_eff = eff*x[m:]
            feasible = bool(np.all(fit_error+1e-11 <= data["drop_error_bound"])
                            and abs(voltage).min() >= vmin and abs(voltage).max() <= vmax
                            and np.all(abs(x[m:]) <= data["source_p_abs_cap"])
                            and np.all(abs(qratio*x[m:]) <= data["source_q_abs_cap"])
                            and np.all(abs(source_eff-data["source_nominal_eff"])
                                       <= data["source_error_eff"]+1e-14))
            fits.append((not feasible, float(np.sum(fit_error**2)), dict(
                common, theta=x[:m].tolist(), source_p=x[m:].tolist(), source_q=(qratio*x[m:]).tolist(),
                historical_numeric_witness=feasible, status="numeric_witness" if feasible else "fit_only_unresolved",
                max_abs_residual=float(fit_error.max()), min_voltage=float(abs(voltage).min()),
                max_voltage=float(abs(voltage).max()), ac_equation_residual=equation_error,
                solver_success=success)))
        except (RuntimeError, ValueError) as error:
            continue
    if not fits:
        return dict(common, historical_numeric_witness=False, status="local_fit_failed_unresolved",
                    theta=None, seconds=perf_counter()-began)
    answer = min(fits, key=lambda item: item[:2])[2]
    answer["seconds"] = perf_counter()-began
    return answer


def squared_drop_error(voltage_measured, voltage_absolute_error, root_error=0.):
    """Convert |Vtrue-Vmeter|<=d to |Vtrue^2-Vmeter^2|<=2|Vmeter|d+d^2.

    If root voltage is also metered, add its analogous error bound. The AC
    solver still fixes root magnitude to one: root_error is valid here only
    for constructing a measured voltage-drop bound when physical root=1.
    An actually varying unknown root needs a different physical model.
    """
    v, d = np.broadcast_arrays(np.asarray(voltage_measured, float),
                               np.asarray(voltage_absolute_error, float))
    if (d < 0).any() or root_error < 0 or not np.isfinite(v).all() or not np.isfinite(d).all():
        raise ValueError("Finite nonnegative meter error required")
    # Outward-rounded rational evaluation prevents an underestimated meter box.
    return np.array([_up(2*abs(f(vi))*f(di)+f(di)**2 +
                         2*f(root_error)+f(root_error)**2)
                     for vi, di in zip(v.ravel(), d.ravel())]).reshape(v.shape)
