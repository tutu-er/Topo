"""Label-free candidate rejection from a convex DistFlow residual bound.

Inputs contain only one candidate, raw noisy measurements and an independently
declared physical/noise domain.  No truth label, true impedance, or calibration
sample is accepted.  ``ac_sse_lower`` bounds the full raw whitened AC SSE, i.e.
twice the half-SSE likelihood objective.  Floating-point guards are reported;
these are numerical certificates, not interval-arithmetic proofs.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping, Sequence
import time

import numpy as np
from scipy.optimize import linprog, minimize
from scipy.stats import chi2


@dataclass(frozen=True)
class TreeSpec:
    root: object
    edges: Sequence[tuple[object, object]]
    observed_nodes: Sequence[object]
    source_node: object | None = None


@dataclass(frozen=True)
class PhysicalDomain:
    z_min: float | Sequence[float]
    z_max: float | Sequence[float]
    v_min: float
    v_max: float
    power_caps: float | Sequence[float] | Mapping[object, float]
    ratio_min: float | None = None
    ratio_max: float | None = None
    source_power_cap: float = 0.0


@dataclass(frozen=True)
class Measurements:
    p: np.ndarray
    q: np.ndarray
    v: np.ndarray
    root_v: float | np.ndarray
    sigma_p: float | np.ndarray
    sigma_q: float | np.ndarray
    sigma_v: float | np.ndarray
    sigma_root: float | np.ndarray | None = None


@dataclass
class Certificate:
    status: str
    reject: bool
    alpha: float
    raw_noise_dimension: int
    chi_square_threshold: float
    qp_primal: float
    qp_solver_lower: float | None
    qp_support_lower: float
    qp_support_guard: float
    qp_solver: str
    qp_status: str
    support_lp_status: str
    ac_sse_lower: float
    remainder_norm: float
    remainder_per_node: list[float]
    lipschitz_k: float
    qp_rejection_threshold: float
    residual_chi_reject: bool
    residual_chi_threshold: float
    residual_chi_noise_radius: float
    residual_chi_linear_radius: float
    residual_chi_quadratic_radius: float
    residual_chi_df: int
    residual_chi_alpha_split: list[float]
    theta: list[float]
    qp_theta: list[float]
    projection_used: bool
    common_prefix_edges_removed: list[int]
    elapsed_seconds: float
    diagnostics: dict

    def as_dict(self):
        return asdict(self)


def _array(value, shape, name, *, positive=False, nonnegative=False):
    try:
        out = np.broadcast_to(np.asarray(value, dtype=float), shape).copy()
    except ValueError as exc:
        raise ValueError(f"{name} cannot broadcast to {shape}") from exc
    if not np.all(np.isfinite(out)):
        raise ValueError(f"{name} must be finite")
    if positive and np.any(out <= 0):
        raise ValueError(f"{name} must be positive")
    if nonnegative and np.any(out < 0):
        raise ValueError(f"{name} must be nonnegative")
    return out


def _tree_arrays(tree: TreeSpec):
    edges = tuple(tuple(e) for e in tree.edges)
    observed = tuple(tree.observed_nodes)
    if not edges or not observed or len(set(observed)) != len(observed):
        raise ValueError("a nonempty tree and distinct observed nodes are required")
    parent, edge_for, children = {}, {}, {}
    nodes = {tree.root}
    for e, pair in enumerate(edges):
        if len(pair) != 2:
            raise ValueError("each edge needs parent and child")
        a, b = pair
        if b == tree.root or a == b or b in parent:
            raise ValueError("edges must define a rooted tree")
        parent[b], edge_for[b] = a, e
        children.setdefault(a, []).append(b)
        nodes.update((a, b))
    if len(edges) != len(nodes) - 1:
        raise ValueError("edges must form one connected tree")
    paths = {tree.root: ()}
    pending = [tree.root]
    while pending:
        a = pending.pop()
        for b in children.get(a, ()):
            if b in paths:
                raise ValueError("cycle in candidate")
            paths[b] = paths[a] + (edge_for[b],)
            pending.append(b)
    if set(paths) != nodes or any(j == tree.root or j not in nodes for j in observed):
        raise ValueError("observations must be nonroot nodes of a connected tree")
    if tree.source_node is not None and tree.source_node not in nodes:
        raise ValueError("source_node is absent from candidate")
    a = np.zeros((len(observed), len(edges)))
    for j, node in enumerate(observed):
        a[j, list(paths[node])] = 1.0
    descendants = np.zeros((len(edges), len(edges)), dtype=bool)
    for f, (_, child) in enumerate(edges):
        for e in paths[child][:-1]:
            descendants[e, f] = True
    return edges, observed, paths, a, descendants


def path_support(tree: TreeSpec) -> np.ndarray:
    """Rows are observed root paths; observed nodes may carry internal loads."""
    return _tree_arrays(tree)[3]


def linear_voltage_matrix(tree: TreeSpec, r, x) -> np.ndarray:
    """H=[2 A diag(r) A.T, 2 A diag(x) A.T], before projection."""
    a = path_support(tree)
    r = _array(r, (a.shape[1],), "r", nonnegative=True)
    x = _array(x, (a.shape[1],), "x", nonnegative=True)
    return 2 * np.concatenate(((a * r) @ a.T, (a * x) @ a.T), axis=1)


def _domain_arrays(tree, domain, observed, m):
    zmin = _array(domain.z_min, (m,), "z_min", positive=True)
    zmax = _array(domain.z_max, (m,), "z_max", positive=True)
    if np.any(zmax < zmin):
        raise ValueError("z_max must cover z_min")
    if not (0 < domain.v_min <= domain.v_max < np.inf):
        raise ValueError("invalid all-bus voltage domain")
    if isinstance(domain.power_caps, Mapping):
        unknown = set(domain.power_caps) - set(observed)
        if any(float(domain.power_caps[j]) != 0 for j in unknown):
            raise ValueError("unobserved injections must be zero except declared source")
        caps = np.array([domain.power_caps.get(j, 0.0) for j in observed], dtype=float)
    else:
        caps = _array(domain.power_caps, (len(observed),), "power_caps", nonnegative=True)
    if np.any(~np.isfinite(caps)) or np.any(caps < 0):
        raise ValueError("power_caps must be finite and nonnegative")
    lo, hi = domain.ratio_min, domain.ratio_max
    if lo is not None and (not np.isfinite(lo) or lo < 0):
        raise ValueError("ratio_min must be finite and nonnegative")
    if hi is not None and (not np.isfinite(hi) or hi < 0):
        raise ValueError("ratio_max must be finite and nonnegative")
    if lo is not None and hi is not None and lo > hi:
        raise ValueError("ratio_min exceeds ratio_max")
    if not np.isfinite(domain.source_power_cap) or domain.source_power_cap < 0:
        raise ValueError("invalid source_power_cap")
    if tree.source_node is None and domain.source_power_cap:
        raise ValueError("source_power_cap needs source_node")
    return zmin, zmax, caps


def _outer_constraints(zmin, zmax, ratio_min, ratio_max):
    m = len(zmin)
    rows, rhs = [], []
    for e in range(m):
        row = np.zeros(2*m)
        row[e] = row[m+e] = 1
        rows.append(row); rhs.append(zmin[e])
        if ratio_min is not None:
            row = np.zeros(2*m); row[e] = -ratio_min; row[m+e] = 1
            rows.append(row); rhs.append(0.)
        if ratio_max is not None:
            row = np.zeros(2*m); row[e] = ratio_max; row[m+e] = -1
            rows.append(row); rhs.append(0.)
    return np.asarray(rows), np.asarray(rhs), np.zeros(2*m), np.tile(zmax, 2)


def _physical_theta(theta, zmin, zmax, ratio_min, ratio_max):
    """Produce a physical-domain point for a separate feasible AC backscore."""
    m = len(zmin)
    r, x = np.maximum(theta[:m], 0), np.maximum(theta[m:], 0)
    angle = np.arctan2(x, r)
    amin = 0 if ratio_min is None else np.arctan(ratio_min)
    amax = np.pi/2 if ratio_max is None else np.arctan(ratio_max)
    angle = np.clip(angle, amin, amax)
    radius = np.clip(np.hypot(r, x), zmin, zmax)
    return np.r_[radius*np.cos(angle), radius*np.sin(angle)]


def qp_with_support_bound(a, b, c, d, lower, upper, *, time_limit=30.,
                          solver="gurobi", numerical_guard=1e-10):
    """min ||A theta-b||² subject to C theta>=d and a finite box.

    The independent bound uses any LP multipliers lambda>=0:
      min g.theta >= lambda.d + min_box (g-C.T lambda).theta.
    Residual dual infeasibility is therefore paid explicitly in box support.
    """
    scale = max(float(np.linalg.norm(b) / np.sqrt(max(len(b), 1))), 1e-8)
    aa, bb = a/scale, b/scale
    start = linprog(np.zeros(a.shape[1]), A_ub=-c, b_ub=-d,
                    bounds=list(zip(lower, upper)), method="highs")
    if not start.success:
        raise ValueError("declared outer parameter domain is empty")
    theta, solver_bound, status, used = start.x, None, "not_run", solver
    if solver == "gurobi":
        try:
            import gurobipy as gp
            model = gp.Model("label_free_candidate_bound")
            model.Params.OutputFlag = 0
            model.Params.TimeLimit = float(time_limit)
            model.Params.OptimalityTol = 1e-9
            model.Params.FeasibilityTol = 1e-9
            v = model.addMVar(len(lower), lb=lower, ub=upper)
            model.addConstr(c @ v >= d)
            gram, lin = aa.T @ aa, -2 * (aa.T @ bb)
            model.setObjective(v @ gram @ v + lin @ v + float(bb @ bb))
            model.optimize()
            status = str(model.Status)
            if model.SolCount:
                theta = np.asarray(v.X)
            if np.isfinite(model.ObjBound):
                solver_bound = float(model.ObjBound * scale**2)
        except (ImportError, Exception) as exc:
            # A valid support bound remains available after solver/license errors.
            status = f"gurobi_error:{type(exc).__name__}:{exc}"
            used = "scipy_slsqp"
    if solver != "gurobi" or used == "scipy_slsqp":
        result = minimize(lambda x: np.linalg.norm(aa@x-bb)**2, theta,
                          jac=lambda x: 2*aa.T@(aa@x-bb), method="SLSQP",
                          bounds=list(zip(lower, upper)),
                          constraints={"type":"ineq", "fun":lambda x:c@x-d,
                                       "jac":lambda x:c},
                          options={"maxiter":1000,"ftol":1e-12})
        theta = np.asarray(result.x)
        status += f";slsqp:{result.status}"
        used = "scipy_slsqp"
    residual = aa @ theta - bb
    value, gradient = float(residual@residual), 2*aa.T@residual
    support = linprog(gradient, A_ub=-c, b_ub=-d,
                      bounds=list(zip(lower, upper)), method="highs")
    multipliers = np.zeros(len(d))
    if support.success:
        multipliers = np.maximum(-np.asarray(support.ineqlin.marginals), 0)
    gbox = gradient - c.T @ multipliers
    box_terms = np.where(gbox >= 0, gbox*lower, gbox*upper)
    dual_support = float(multipliers@d + box_terms.sum())
    tangent = value + dual_support - float(gradient@theta)
    magnitude = max(1., abs(value), abs(dual_support),
                    float(np.sum(np.abs(gradient*theta))),
                    float(np.sum(np.abs(multipliers*d))),
                    float(np.sum(np.abs(box_terms))))
    guard = numerical_guard * magnitude * scale**2
    bound = max(0., tangent*scale**2 - guard)
    return dict(theta=theta, primal=float(np.linalg.norm(a@theta-b)**2),
                support_lower=bound, guard=guard, solver_lower=solver_bound,
                solver=used, status=status,
                support_status=str(support.status),
                domain_violation=float(max(0., np.max(d-c@theta),
                                           np.max(lower-theta), np.max(theta-upper))))


def certify_candidate(tree: TreeSpec, obs: Measurements, domain: PhysicalDomain,
                      *, alpha=.05, source_projection=False, solver_time_limit=30.,
                      solver="gurobi", numerical_guard=1e-10) -> Certificate:
    started = time.perf_counter()
    if not (0 < alpha < 1):
        raise ValueError("alpha must lie in (0,1)")
    edges, observed, paths, a_load, descendant = _tree_arrays(tree)
    n, m = a_load.shape
    p = np.asarray(obs.p, dtype=float)
    if p.ndim != 2 or p.shape[1] != n or p.shape[0] < 1:
        raise ValueError("p must have shape (time, observed_node)")
    shape, tcount = p.shape, p.shape[0]
    p = _array(p, shape, "p")
    q, v = _array(obs.q, shape, "q"), _array(obs.v, shape, "v")
    if np.any(v <= 0):
        raise ValueError("observed voltage magnitudes must be positive")
    root = _array(obs.root_v, (tcount,), "root_v", positive=True)
    sp = _array(obs.sigma_p, shape, "sigma_p", positive=True)
    sq = _array(obs.sigma_q, shape, "sigma_q", positive=True)
    sv = _array(obs.sigma_v, shape, "sigma_v", positive=True)
    sr = _array(0. if obs.sigma_root is None else obs.sigma_root,
                (tcount,), "sigma_root", nonnegative=True)
    zmin, zmax, caps = _domain_arrays(tree, domain, observed, m)
    projection = np.eye(n)
    a_voltage = a_load.copy()
    removed = ()
    if source_projection:
        if tree.source_node is None:
            raise ValueError("common-source projection requires a declared source_node")
        prefix = set(paths[tree.source_node])
        if any(not prefix.issubset(paths[j]) for j in observed):
            raise ValueError("source PCC must be an ancestor of every observed node")
        projection -= np.ones((n,n))/n
        removed = paths[tree.source_node]
        a_voltage[:, list(removed)] = 0
    elif tree.source_node is not None and domain.source_power_cap > 0:
        raise ValueError("nonzero unmetered source needs valid common-PCC projection")
    # Constant source power at the common PCC and its entire upstream drop
    # disappear exactly from P(v_root² 1-v²). Downstream currents contain no
    # PCC source. This removes its common-path nonlinear losses as well.
    pa = projection @ a_voltage
    psum, qsum = p @ a_load, q @ a_load
    aa = np.empty((tcount*n, 2*m))
    for e in range(m):
        aa[:, e] = (2*psum[:,e,None]*pa[None,:,e]).reshape(-1)
        aa[:, m+e] = (2*qsum[:,e,None]*pa[None,:,e]).reshape(-1)
    yy = root[:,None]**2-v**2
    bb = (yy @ projection.T).reshape(-1)
    c, d, lower, upper = _outer_constraints(zmin,zmax,domain.ratio_min,domain.ratio_max)
    qp = qp_with_support_bound(aa,bb,c,d,lower,upper,
                               time_limit=solver_time_limit,solver=solver,
                               numerical_guard=numerical_guard)
    # Uniform exact DistFlow loss envelope, including downstream propagation.
    downstream_cap = a_load.T @ caps
    if tree.source_node is not None and not source_projection:
        downstream_cap[list(paths[tree.source_node])] += domain.source_power_cap
    ell = (downstream_cap/domain.v_min)**2
    eta = np.zeros(n)
    for j in range(n):
        for e in np.flatnonzero(a_voltage[j]):
            eta[j] += zmax[e]**2*ell[e]
            eta[j] += 2*zmax[e]*np.sum(zmax[descendant[e]]*ell[descendant[e]])
    # eta lies in the componentwise box [0,eta_bar]. Centering this box makes
    # common-mode projection tighter; triangle inequality remains conservative.
    eta_norm = .5*(np.linalg.norm(projection@eta)+np.linalg.norm(eta))
    delta = float(np.sqrt(tcount)*eta_norm)
    hbar = 2*(a_voltage*zmax)@a_load.T
    hbar = np.concatenate((hbar,hbar),axis=1)
    klist, k0list=[], []
    for t in range(tcount):
        pqscale=np.r_[sp[t],sq[t]]
        # Nonnegative H<=Hbar gives ||H D||<=||Hbar D||. P is a contraction.
        kpq=float(np.linalg.norm(hbar*pqscale,2))
        kv=float(np.max((np.abs(v[t])+domain.v_max)*sv[t]))
        kr=0. if source_projection else float((abs(root[t])+domain.v_max)*np.sqrt(n)*sr[t])
        klist.append(np.sqrt(kpq**2+kv**2+kr**2))
        kv0=float(2*domain.v_max*np.max(sv[t]))
        kr0=0. if source_projection else float(2*domain.v_max*np.sqrt(n)*sr[t])
        k0list.append(np.sqrt(kpq**2+kv0**2+kr0**2))
    kval=float(max(klist))
    dim=int(3*n*tcount+np.count_nonzero(sr))
    threshold=float(chi2.isf(alpha,dim))
    ac_lower=float((max(np.sqrt(qp["support_lower"])-delta,0)/kval)**2)
    if n == 1 and source_projection:
        ac_lower=0.
    # A separate, tighter residual test. At the fixed TRUE physical point,
    # r(theta_true)=P eta + L_true epsilon + P(e_root² 1-e_v²).
    # L_true is deterministic, rank<=rank(P)*T, ||L_true||<=K0. Thus its
    # Gaussian norm is dominated by K0*sqrt(chi2_rank), without pretending that
    # the fitted regression with noisy P/Q has ordinary LS degrees of freedom.
    # Two additional events bound max|e_v| and ||e_v||. No independence between
    # these three events is required; their failure probabilities are summed.
    alpha_parts=[float(alpha/3)]*3
    rank=int((n-1 if source_projection else n)*tcount)
    linear_radius=float(max(k0list)*np.sqrt(chi2.isf(alpha_parts[0],rank))) if rank else 0.
    root_active=bool(not source_projection and np.any(sr>0))
    n_voltage=int(n*tcount+(np.count_nonzero(sr) if root_active else 0))
    voltage_sigma=float(max(np.max(sv),np.max(sr) if root_active else 0.))
    # With noisy root the quadratic mapping [e_v²;e_root²] -> e_root² 1-e_v²
    # has norm sqrt(n+1); common-mode projection removes its root column.
    quadratic_map_norm=np.sqrt(n+1) if root_active else 1.
    quadratic_radius=float(quadratic_map_norm*voltage_sigma**2*np.sqrt(
        2*np.log(2*n_voltage/alpha_parts[1])*chi2.isf(alpha_parts[2],n_voltage)))
    if rank==0:
        quadratic_radius=0.
    noise_radius=linear_radius+quadratic_radius
    residual_threshold=float((delta+noise_radius)**2)
    residual_reject=bool(qp["support_lower"]>residual_threshold)
    physical=_physical_theta(qp["theta"],zmin,zmax,domain.ratio_min,domain.ratio_max)
    return Certificate(
        status="rejected" if ac_lower>threshold else "not_rejected",
        reject=bool(ac_lower>threshold),alpha=float(alpha),raw_noise_dimension=dim,
        chi_square_threshold=threshold,qp_primal=qp["primal"],
        qp_solver_lower=qp["solver_lower"],qp_support_lower=qp["support_lower"],
        qp_support_guard=qp["guard"],qp_solver=qp["solver"],qp_status=qp["status"],
        support_lp_status=qp["support_status"],ac_sse_lower=ac_lower,
        remainder_norm=delta,remainder_per_node=eta.tolist(),lipschitz_k=kval,
        qp_rejection_threshold=float((delta+kval*np.sqrt(threshold))**2),
        residual_chi_reject=residual_reject,residual_chi_threshold=residual_threshold,
        residual_chi_noise_radius=float(noise_radius),
        residual_chi_linear_radius=linear_radius,
        residual_chi_quadratic_radius=quadratic_radius,residual_chi_df=rank,
        residual_chi_alpha_split=alpha_parts,
        theta=physical.tolist(),qp_theta=qp["theta"].tolist(),
        projection_used=bool(source_projection),common_prefix_edges_removed=list(removed),
        elapsed_seconds=time.perf_counter()-started,
        diagnostics={"qp_domain_violation":qp["domain_violation"],
                     "qp_domain":"first_quadrant_box_with_r_plus_x_norm_outer_cut",
                     "physical_norm_bounds":[zmin.tolist(),zmax.tolist()],
                     "noise_model":"independent_zero_mean_raw_gaussian_known_sigmas",
                     "sigma_estimated_from_candidate_residuals":False,
                     "confidence_scope":"declared_domain_and_candidate_pool",
                     "bound_validation":"floating_point_guard_not_interval_arithmetic",
                     "residual_chi_scope":"fixed_physical_mean_independent_of_within_block_meter_noise",
                     "residual_chi_df_is_not_fitted_parameter_df":True,
                     "unobserved_internal_injection":"zero_except_declared_source"})
