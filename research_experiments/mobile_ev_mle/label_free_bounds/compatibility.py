"""Observed-only AC feasible explanations, complementing rejection lower bounds.

For a candidate with every nonroot physical bus metered, let D contain the
root-path supports (node by edge), S = Pobs + i Qobs, and
Z = D diag(R+iX) D.T.  We solve V = V0 - Z conj(S/V).  Setting latent P/Q to
their noisy readings is one allowed state choice in the raw P/Q/V error model;
its P/Q residual is zero.  Thus a physically legal point gives an *upper*
bound U = sum(((abs(V)-Vobs)/sigmaV)**2) on the full raw whitened AC SSE.
U inside the joint Gaussian noise ball proves compatibility, not topology
correctness.  U outside the ball cannot reject a candidate.  No MLE or global
optimality claim is made, and no source scenario is fitted in this module.

R+iX = exp(a+i phi), with a=log(radius).  Implicit differentiation gives
  dV - Z diag(conj(S)/conj(V)**2) conj(dV) = -dZ conj(S/V).
The real 2n system is solved for all 2m parameter directions together.  The
search uses only noisy meters, common public equipment limits, and the QP
candidate points.  Simulator truth is not an input.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.optimize import least_squares
from scipy.stats import chi2

from .bounds import Measurements, PhysicalDomain, TreeSpec, path_support


VERSION = "label_free_fixed_meter_PQ_AC_compatibility_v1"


def _broadcast(value, shape, name, *, positive=False):
    out = np.broadcast_to(np.asarray(value, float), shape).copy()
    if not np.all(np.isfinite(out)) or (positive and np.any(out <= 0)):
        raise ValueError(f"invalid {name}")
    return out


def polar_limits(tree, domain):
    m = len(tree.edges)
    zlo = _broadcast(domain.z_min, (m,), "z_min", positive=True)
    zhi = _broadcast(domain.z_max, (m,), "z_max", positive=True)
    if np.any(zhi <= zlo):
        raise ValueError("compatibility search needs z_min < z_max")
    if domain.ratio_min is None or domain.ratio_max is None:
        raise ValueError("finite X/R limits are required")
    lo, hi = float(domain.ratio_min), float(domain.ratio_max)
    if not np.isfinite([lo, hi]).all() or not 0 <= lo < hi:
        raise ValueError("invalid X/R limits")
    return (np.r_[np.log(zlo), np.full(m, np.arctan(lo))],
            np.r_[np.log(zhi), np.full(m, np.arctan(hi))])


def unpack_polar(parameters):
    parameters = np.asarray(parameters, float)
    if parameters.ndim != 1 or len(parameters) % 2:
        raise ValueError("polar parameters need equal radius and angle blocks")
    m = len(parameters)//2
    return np.exp(parameters[:m]+1j*parameters[m:])


def ac_voltage(tree, p, q, root_v, parameters, *, derivatives=False,
               tol=2e-13, max_iter=1000):
    """Exact nodal fixed point and optional analytic magnitude Jacobian.

    This native-tree implementation includes injections at internal observed
    buses.  It does not insert virtual meter edges or read physical line truth.
    """
    d = path_support(tree)
    n, m = d.shape
    physical_nodes = {node for edge in tree.edges for node in edge}
    if set(tree.observed_nodes) != physical_nodes-{tree.root} or n != m:
        raise ValueError("all nonroot physical buses must be observed")
    p = np.asarray(p, float)
    q = np.asarray(q, float)
    if p.ndim != 2 or p.shape != q.shape or p.shape[1] != n or not len(p):
        raise ValueError("P/Q must be matching time-by-observed-bus matrices")
    if not np.isfinite(p).all() or not np.isfinite(q).all():
        raise ValueError("finite P/Q required")
    root = _broadcast(root_v, (len(p),), "root_v", positive=True)
    z = unpack_polar(parameters)
    if z.shape != (m,) or not np.isfinite(z).all():
        raise ValueError("invalid impedance vector")
    zbus = (d*z) @ d.T
    s = p+1j*q
    v = np.broadcast_to(root[:, None], p.shape).astype(complex).copy()
    for iteration in range(1, max_iter+1):
        if np.min(abs(v)) < .1:
            raise RuntimeError("AC fixed point reached low voltage")
        new = root[:, None]-np.conj(s/v) @ zbus.T
        if not np.isfinite(new).all():
            raise RuntimeError("AC fixed point is nonfinite")
        error = float(np.max(abs(new-v)))
        v = new
        if error <= tol:
            break
    else:
        raise RuntimeError("AC fixed point did not converge")
    currents = np.conj(s/v)
    branch_currents = currents @ d
    index = {bus: j for j, bus in enumerate(tree.observed_nodes)}
    equation = 0.
    for e, (parent, child) in enumerate(tree.edges):
        vp = root if parent == tree.root else v[:, index[parent]]
        equation = max(equation, float(np.max(abs(
            vp-v[:, index[child]]-z[e]*branch_currents[:, e]))))
    master = root*np.conj(currents.sum(axis=1))
    loss = np.sum(z[None, :]*abs(branch_currents)**2, axis=1)
    balance = float(np.max(abs(master-s.sum(axis=1)-loss)))
    out = dict(voltage=v, magnitude=abs(v), impedance=z,
               branch_currents=branch_currents, iterations=iteration,
               ac_equation_residual=equation, power_balance_residual=balance,
               master_p=master.real, master_q=master.imag)
    if derivatives:
        c = zbus[None, :, :]*(np.conj(s)/np.conj(v)**2)[:, None, :]
        eye = np.broadcast_to(np.eye(n), c.shape)
        real_system = np.concatenate((
            np.concatenate((eye-c.real, -c.imag), axis=2),
            np.concatenate((-c.imag, eye+c.real), axis=2)), axis=1)
        dz = np.concatenate((z, 1j*z))
        de = np.concatenate((d, d), axis=1)
        branch = np.concatenate((branch_currents, branch_currents), axis=1)
        rhs = -de[None, :, :]*dz[None, None, :]*branch[:, None, :]
        solution = np.linalg.solve(real_system,
                                   np.concatenate((rhs.real, rhs.imag), axis=1))
        dv = solution[:, :n, :]+1j*solution[:, n:, :]
        out["complex_jacobian"] = dv
        out["magnitude_jacobian"] = np.real(np.conj(v)[:, :, None]*dv)/abs(v)[:, :, None]
    return out


def physical_checks(tree, obs, domain, point):
    """Every check is needed before a point supplies an AC SSE upper bound."""
    p, q = np.asarray(obs.p), np.asarray(obs.q)
    if isinstance(domain.power_caps, dict):
        caps = np.array([domain.power_caps[bus] for bus in tree.observed_nodes])
    else:
        caps = _broadcast(domain.power_caps, (p.shape[1],), "power_caps")
    z, v = point["impedance"], point["magnitude"]
    radius = abs(z)
    zlo = _broadcast(domain.z_min, z.shape, "z_min", positive=True)
    zhi = _broadcast(domain.z_max, z.shape, "z_max", positive=True)
    root = _broadcast(obs.root_v, (len(p),), "root_v", positive=True)
    tolerance = 1e-10
    return dict(
        nodal_power=bool(np.all(np.hypot(p, q) <= caps[None, :]+tolerance)),
        all_bus_voltage=bool(min(float(v.min()), float(root.min())) >= domain.v_min-tolerance
                             and max(float(v.max()), float(root.max())) <= domain.v_max+tolerance),
        impedance_modulus=bool(np.all(radius >= zlo-tolerance) and np.all(radius <= zhi+tolerance)),
        impedance_ratio=bool(np.all(z.imag/z.real >= domain.ratio_min-tolerance)
                             and np.all(z.imag/z.real <= domain.ratio_max+tolerance)),
        ac_equations=bool(point["ac_equation_residual"] <= 1e-9),
        power_balance=bool(point["power_balance_residual"] <= 1e-9),
        source=bool(tree.source_node is None and domain.source_power_cap == 0),
        known_root=bool(obs.sigma_root is None or np.all(np.asarray(obs.sigma_root) == 0)))


class _CompatiblePointFound(Exception):
    pass


def fit_compatible_point(tree, obs, domain, qp_theta, *, alpha=.05,
                         max_nfev=100, tolerance=1e-7, early_stop=True):
    """Search two common starts; return only physically checked feasible bounds.

    The QP start is projected to the actual modulus/angle domain.  The second
    start uses minimum radius and common midpoint angle for every edge, using
    the same rule for every candidate.  Failed starts and invalid AC trials
    remain in the returned record.
    """
    began = perf_counter()
    if tree.source_node is not None or domain.source_power_cap:
        return dict(status="unsupported_source", available=False, reject=False,
                    reason="this audit fits no unmetered source", seconds=perf_counter()-began)
    lower, upper = polar_limits(tree, domain)
    p, q, observed_v = [np.asarray(a, float) for a in (obs.p, obs.q, obs.v)]
    if p.shape != q.shape or p.shape != observed_v.shape:
        raise ValueError("P/Q/V observation shapes differ")
    sv = _broadcast(obs.sigma_v, p.shape, "sigma_v", positive=True)
    _broadcast(obs.sigma_p, p.shape, "sigma_p", positive=True)
    _broadcast(obs.sigma_q, p.shape, "sigma_q", positive=True)
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0,1)")
    threshold = float(chi2.isf(alpha, 3*p.size))
    m = len(tree.edges)
    theta = np.asarray(qp_theta, float)
    if theta.shape != (2*m,) or not np.isfinite(theta).all():
        raise ValueError("QP theta must contain finite R then X")
    qp = np.r_[np.log(np.maximum(np.hypot(theta[:m], theta[m:]), np.exp(lower[:m]))),
               np.arctan2(np.maximum(theta[m:], 0), np.maximum(theta[:m], 0))]
    qp = np.clip(qp, lower, upper)
    safe = np.r_[lower[:m], (lower[m:]+upper[m:])/2]
    starts = [("QP_projected", qp), ("uniform_minimum_radius", safe)]
    best = None
    records = []
    for label, start in starts:
        calls = invalid = jac_calls = 0
        cache_x = cache = None
        start_best = None

        def evaluate(x, need_jac=False):
            nonlocal calls, invalid, jac_calls, cache_x, cache, best, start_best
            if cache_x is None or not np.array_equal(cache_x, x) or (need_jac and "magnitude_jacobian" not in cache):
                if need_jac:
                    jac_calls += 1
                else:
                    calls += 1
                try:
                    point = ac_voltage(tree, p, q, obs.root_v, x, derivatives=need_jac)
                except (RuntimeError, np.linalg.LinAlgError, FloatingPointError):
                    invalid += 1
                    raise
                cache_x, cache = x.copy(), point
                residual = (point["magnitude"]-observed_v)/sv
                score = float(np.sum(residual**2))
                checks = physical_checks(tree, obs, domain, point)
                if all(checks.values()):
                    point_record = dict(raw_whitened_AC_sse_upper=score,
                        parameters_polar=x.tolist(),
                        r=point["impedance"].real.tolist(), x=point["impedance"].imag.tolist(),
                        checks=checks, voltage_min=float(point["magnitude"].min()),
                        voltage_max=float(point["magnitude"].max()),
                        ac_equation_residual=point["ac_equation_residual"],
                        power_balance_residual=point["power_balance_residual"], start=label)
                    if start_best is None or score < start_best["raw_whitened_AC_sse_upper"]:
                        start_best = point_record
                    if best is None or score < best["raw_whitened_AC_sse_upper"]:
                        best = point_record
                    if early_stop and score <= threshold:
                        raise _CompatiblePointFound
            return cache

        def fun(x):
            return ((evaluate(x)["magnitude"]-observed_v)/sv).reshape(-1)

        def jac(x):
            point = evaluate(x, need_jac=True)
            return (point["magnitude_jacobian"]/sv[:, :, None]).reshape(p.size, 2*m)

        record = dict(start=label)
        try:
            result = least_squares(fun, start, jac=jac, bounds=(lower, upper),
                method="trf", tr_solver="exact", max_nfev=int(max_nfev),
                ftol=tolerance, xtol=tolerance, gtol=tolerance)
            evaluate(result.x)
            record.update(status="optimizer_stopped", success=bool(result.success),
                optimizer_status=int(result.status), message=str(result.message),
                nfev=int(result.nfev), optimality=float(result.optimality))
        except _CompatiblePointFound:
            record.update(status="compatible_point_found", success=True,
                          message="stopped after a legal point entered the raw noise ball")
        except Exception as exc:
            record.update(status="search_failure", success=False, failure=repr(exc))
        record.update(ac_calls=calls, jacobian_calls=jac_calls, invalid_trials=invalid,
                      best_legal_point=start_best)
        records.append(record)
    return dict(version=VERSION, status="compatible" if best is not None and
                best["raw_whitened_AC_sse_upper"] <= threshold else
                ("legal_point_outside_noise_ball" if best is not None else "no_legal_point_found"),
        available=best is not None, reject=False, fitted_or_MLE=False,
        best_point=best, starts=records, chi_square_threshold=threshold,
        raw_noise_dimension=3*p.size, alpha=alpha,
        ac_noise_ball_feasible=bool(best is not None and best["raw_whitened_AC_sse_upper"] <= threshold),
        latent_PQ_choice="noisy P/Q readings; zero P/Q residual, not P/Q truth",
        source_PQ_choice=[0., 0.], optimization_scope="local feasible-explanation search only",
        seconds=perf_counter()-began)


def _read(path):
    path = Path(path)
    raw = path.read_bytes()
    return json.loads(gzip.decompress(raw) if path.suffix == ".gz" else raw)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _save(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def run_pilot(meter_path, asset_path, qp_path, destination, *, max_nfev=100):
    """Read exactly the three declared data inputs, never a truth file."""
    began = perf_counter()
    meter, assets, qp = map(_read, (meter_path, asset_path, qp_path))
    if tuple(meter["observed_nodes"]) != tuple(assets["observed_nodes"]):
        raise ValueError("observation order differs")
    if meter["id"] != qp["history_id"]:
        raise ValueError("QP result belongs to another history")
    domain_data = dict(assets["domain"])
    domain_data["power_caps"] = {int(k): v for k, v in domain_data["power_caps"].items()}
    if meter["source_mode"] == "none":
        domain_data["source_power_cap"] = 0.
    domain = PhysicalDomain(**domain_data)
    obs = Measurements(**{key: np.asarray(meter[key], float) for key in
        ("p", "q", "v", "root_v", "sigma_p", "sigma_q", "sigma_v")}, sigma_root=meter["sigma_root"])
    qp_by_id = {row["candidate_id"]: row for row in qp["candidate_results"]}
    rows = []
    destination = Path(destination)
    source = Path(__file__).resolve()
    metadata = dict(version=VERSION, history_id=meter["id"], truth_inputs=False,
        selector_read_paths=[str(Path(p).resolve()) for p in (meter_path, asset_path, qp_path)],
        input_sha256={str(Path(p).name): _sha(p) for p in (meter_path, asset_path, qp_path)},
        source_sha256=_sha(source), max_nfev=max_nfev, tolerance=1e-7,
        early_stop_on_compatible_point=True,
        thread_environment={key: os.environ.get(key) for key in
            ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")})
    for candidate in assets["candidates"]:
        cid = candidate["id"]
        tree = TreeSpec(root=candidate["root"], edges=tuple(map(tuple, candidate["edges"])),
            observed_nodes=tuple(candidate["observed_nodes"]),
            source_node=None if meter["source_mode"] == "none" else candidate["source_node"])
        qprec = qp_by_id[cid]
        try:
            if not qprec["success"]:
                raise ValueError("QP point is unavailable")
            result = fit_compatible_point(tree, obs, domain, qprec["result"]["theta"],
                alpha=qp["alpha"], max_nfev=max_nfev)
            if result["available"]:
                upper = result["best_point"]["raw_whitened_AC_sse_upper"]
                lower = qprec["result"]["ac_sse_lower"]
                result["lower_bound_crosscheck"] = dict(ac_sse_lower=lower,
                    ac_sse_upper=upper, ordered=bool(lower <= upper+1e-7*max(1., upper)))
        except Exception as exc:
            result = dict(status="audit_failure", available=False, reject=False, failure=repr(exc))
        row = dict(candidate_id=cid, result=result)
        rows.append(row)
        _save(destination/(cid+".json"), row)
        _save(destination/"summary.json", dict(metadata, status="running",
            completed_candidates=len(rows), total_candidates=len(assets["candidates"]),
            candidate_results=rows, seconds=perf_counter()-began))
        print(cid, result["status"],
              None if result.get("best_point") is None else result["best_point"]["raw_whitened_AC_sse_upper"],
              flush=True)
    summary = dict(metadata, status="complete", completed_candidates=len(rows),
        total_candidates=len(assets["candidates"]), candidate_results=rows,
        compatible_candidates=[row["candidate_id"] for row in rows if row["result"].get("ac_noise_ball_feasible")],
        seconds=perf_counter()-began)
    _save(destination/"summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--meters", required=True)
    parser.add_argument("--assets", required=True)
    parser.add_argument("--qp-results", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-nfev", type=int, default=100)
    args = parser.parse_args()
    run_pilot(args.meters, args.assets, args.qp_results, args.output, max_nfev=args.max_nfev)


if __name__ == "__main__":
    main()
