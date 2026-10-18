"""Observed-only AC feasible points with a direct per-edge resistance floor.

For each edge, write kappa=X/R and h=sqrt(1+kappa**2).  An exact
parameterization of the declared first-quadrant physical domain is

    kappa in [kappa_min, min(kappa_max, sqrt((z_max/r_min)**2-1))],
    r_lo=max(r_min,z_min/h), r_hi=z_max/h,
    R=r_lo+s*(r_hi-r_lo), X=kappa*R, s in [0,1].

The AC backend uses a=log(abs(R+iX)), phi=atan(kappa).  The max in r_lo
is piecewise smooth; the derivative at its kink is explicitly the r_min
branch derivative (zero).  No smoothness or global-optimality claim is made.

This module reads noisy meters, public equipment limits and candidate QP
points only.  A physically valid point supplies an upper bound on minimum
AC SSE.  A large upper bound never rejects a candidate.  The direct R floor
is an additional, declared equipment assumption rather than a threshold
chosen from topology labels.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.optimize import least_squares
from scipy.stats import chi2

from .bounds import Measurements, PhysicalDomain, TreeSpec
from .compatibility import (
    _CompatiblePointFound, _broadcast, _read, _save, _sha,
    ac_voltage, physical_checks,
)


VERSION = "label_free_direct_resistance_AC_compatibility_v1"


def resistance_limits(tree, domain):
    """Bounds for [s, kappa]; empty or zero-width ratio domains are explicit."""
    m = len(tree.edges)
    zlo = _broadcast(domain.z_min, (m,), "z_min", positive=True)
    zhi = _broadcast(domain.z_max, (m,), "z_max", positive=True)
    rmin = _broadcast(domain.r_min, (m,), "r_min", positive=True)
    if np.any(zhi <= zlo):
        raise ValueError("resistance search needs z_min < z_max")
    if domain.ratio_min is None or domain.ratio_max is None:
        raise ValueError("finite X/R limits are required")
    klo, khi = float(domain.ratio_min), float(domain.ratio_max)
    if not np.isfinite([klo, khi]).all() or not 0 <= klo < khi:
        raise ValueError("invalid X/R limits")
    if np.any(rmin >= zhi):
        raise ValueError("empty or zero-width resistance domain: r_min >= z_max")
    cap = np.minimum(khi, np.sqrt((zhi/rmin)**2-1))
    if np.any(cap <= klo):
        raise ValueError("empty or zero-width X/R domain under r_min")
    return np.r_[np.zeros(m), np.full(m, klo)], np.r_[np.ones(m), cap]


def resistance_to_polar(parameters, domain, *, derivatives=False):
    """Map [s,kappa] to backend [log|z|,phi], optionally its exact chain matrix.

    The returned chain matrix has rows (a,phi) and columns (s,kappa).
    At z_min/sqrt(1+kappa**2)==r_min, the r_min branch is used.  Away
    from this joining surface, all derivatives are ordinary derivatives.
    """
    x = np.asarray(parameters, float)
    if x.ndim != 1 or len(x) % 2 or not np.isfinite(x).all():
        raise ValueError("parameters need finite equal s and kappa blocks")
    m = len(x)//2
    s, kappa = x[:m], x[m:]
    zlo = _broadcast(domain.z_min, (m,), "z_min", positive=True)
    zhi = _broadcast(domain.z_max, (m,), "z_max", positive=True)
    rmin = _broadcast(domain.r_min, (m,), "r_min", positive=True)
    h = np.sqrt(1+kappa*kappa)
    radius_lower_r = zlo/h
    rlo = np.maximum(rmin, radius_lower_r)
    rhi = zhi/h
    r = rlo+s*(rhi-rlo)
    if np.any(r <= 0):
        raise ValueError("mapped resistance must be positive")
    polar = np.r_[np.log(r)+np.log(h), np.arctan(kappa)]
    if not derivatives:
        return polar
    drlo = np.where(radius_lower_r > rmin, -zlo*kappa/h**3, 0.)
    drhi = -zhi*kappa/h**3
    drdk = (1-s)*drlo+s*drhi
    chain = np.zeros((2*m, 2*m))
    index = np.arange(m)
    chain[index, index] = (rhi-rlo)/r
    chain[index, m+index] = drdk/r+kappa/h**2
    chain[m+index, m+index] = 1/h**2
    return polar, chain


def ac_voltage_resistance(tree, p, q, root_v, parameters, domain, *, derivatives=False):
    """Exact AC voltage with Jacobian columns ordered as s then kappa."""
    mapped = resistance_to_polar(parameters, domain, derivatives=derivatives)
    polar, chain = mapped if derivatives else (mapped, None)
    point = ac_voltage(tree, p, q, root_v, polar, derivatives=derivatives)
    if derivatives:
        point["magnitude_jacobian"] = point["magnitude_jacobian"] @ chain
        point["complex_jacobian"] = point["complex_jacobian"] @ chain
    point["parameters_polar"] = polar
    return point


def project_qp_start(tree, domain, qp_theta):
    """Project an R/X QP point into the exact domain without using labels."""
    lower, upper = resistance_limits(tree, domain)
    m = len(tree.edges)
    theta = np.asarray(qp_theta, float)
    if theta.shape != (2*m,) or not np.isfinite(theta).all():
        raise ValueError("QP theta must contain finite R then X")
    kappa = np.clip(theta[m:]/np.maximum(theta[:m], np.finfo(float).tiny),
                    lower[m:], upper[m:])
    h = np.sqrt(1+kappa*kappa)
    rlo = np.maximum(_broadcast(domain.r_min, (m,), "r_min", positive=True),
                      _broadcast(domain.z_min, (m,), "z_min", positive=True)/h)
    rhi = _broadcast(domain.z_max, (m,), "z_max", positive=True)/h
    width = rhi-rlo
    r = np.clip(theta[:m], rlo, rhi)
    s = np.divide(r-rlo, width, out=np.full(m, .5), where=width > 1e-14)
    return np.r_[np.clip(s, 0., 1.), kappa]


def fit_resistance_compatible_point(tree, obs, domain, qp_theta, *, alpha=.05,
                                    max_nfev=100, tolerance=1e-7, early_stop=True):
    """Two starts, fixed meter P/Q, checked AC feasible points, no rejection."""
    began = perf_counter()
    if tree.source_node is not None or domain.source_power_cap:
        return dict(version=VERSION, status="unsupported_source", available=False,
                    reject=False, reason="this audit fits no unmetered source",
                    seconds=perf_counter()-began)
    lower, upper = resistance_limits(tree, domain)
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
    safe = np.r_[np.zeros(m), (lower[m:]+upper[m:])/2]
    starts = [("QP_projected", project_qp_start(tree, domain, qp_theta)),
              ("common_ratio_low_radius", safe)]
    best, records = None, []
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
                    point = ac_voltage_resistance(tree, p, q, obs.root_v, x, domain,
                                                  derivatives=need_jac)
                except (RuntimeError, np.linalg.LinAlgError, FloatingPointError):
                    invalid += 1
                    raise
                cache_x, cache = x.copy(), point
                score = float(np.sum(((point["magnitude"]-observed_v)/sv)**2))
                checks = physical_checks(tree, obs, domain, point)
                rmin = _broadcast(domain.r_min, (m,), "r_min", positive=True)
                checks["resistance_floor"] = bool(np.all(point["impedance"].real >= rmin-1e-10))
                if all(checks.values()):
                    record = dict(raw_whitened_AC_sse_upper=score,
                        parameters_resistance=x.tolist(),
                        parameters_polar=point["parameters_polar"].tolist(),
                        r=point["impedance"].real.tolist(), x=point["impedance"].imag.tolist(),
                        checks=checks, voltage_min=float(point["magnitude"].min()),
                        voltage_max=float(point["magnitude"].max()),
                        ac_equation_residual=point["ac_equation_residual"],
                        power_balance_residual=point["power_balance_residual"], start=label)
                    if start_best is None or score < start_best["raw_whitened_AC_sse_upper"]:
                        start_best = record
                    if best is None or score < best["raw_whitened_AC_sse_upper"]:
                        best = record
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
                nfev=int(result.nfev), optimality=float(result.optimality),
                evaluation_limit_reached=bool(result.status == 0))
        except _CompatiblePointFound:
            record.update(status="compatible_point_found", success=True,
                          message="stopped after a legal point entered the raw noise ball")
        except Exception as exc:
            record.update(status="search_failure", success=False, failure=repr(exc))
        record.update(ac_calls=calls, jacobian_calls=jac_calls, invalid_trials=invalid,
                      best_legal_point=start_best)
        records.append(record)
    inside = bool(best is not None and best["raw_whitened_AC_sse_upper"] <= threshold)
    return dict(version=VERSION, status="compatible" if inside else
        ("legal_point_outside_noise_ball" if best is not None else "no_legal_point_found"),
        available=best is not None, reject=False, fitted_or_MLE=False,
        best_point=best, starts=records, chi_square_threshold=threshold,
        raw_noise_dimension=3*p.size, alpha=alpha, ac_noise_ball_feasible=inside,
        r_min=_broadcast(domain.r_min, (m,), "r_min", positive=True).tolist(),
        latent_PQ_choice="noisy P/Q readings; zero P/Q residual, not P/Q truth",
        source_PQ_choice=[0., 0.], optimization_scope="local feasible-explanation search only",
        max_derivative_at_kink="r_min branch", seconds=perf_counter()-began)


def run_pilot(meter_path, asset_path, qp_path, destination, *, max_nfev=100):
    """Only three observed/public inputs; assets must declare direct r_min."""
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
    destination = Path(destination)
    metadata = dict(version=VERSION, history_id=meter["id"], truth_inputs=False,
        selector_read_paths=[str(Path(p).resolve()) for p in (meter_path, asset_path, qp_path)],
        input_sha256={str(Path(p).name): _sha(p) for p in (meter_path, asset_path, qp_path)},
        source_sha256=_sha(Path(__file__)), max_nfev=max_nfev, tolerance=1e-7,
        early_stop_on_compatible_point=True, direct_r_min=domain_data.get("r_min", 0),
        thread_environment={key: os.environ.get(key) for key in
            ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")})
    rows = []
    for candidate in assets["candidates"]:
        cid = candidate["id"]
        tree = TreeSpec(root=candidate["root"], edges=tuple(map(tuple, candidate["edges"])),
            observed_nodes=tuple(candidate["observed_nodes"]),
            source_node=None if meter["source_mode"] == "none" else candidate["source_node"])
        qprec = qp_by_id[cid]
        try:
            if not qprec["success"]:
                raise ValueError("QP point is unavailable")
            result = fit_resistance_compatible_point(tree, obs, domain, qprec["result"]["theta"],
                alpha=qp.get("alpha", qprec["result"]["alpha"]), max_nfev=max_nfev)
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
        print(cid, result["status"], None if result.get("best_point") is None else
              result["best_point"]["raw_whitened_AC_sse_upper"], flush=True)
    summary = dict(metadata, status="complete", completed_candidates=len(rows),
        total_candidates=len(assets["candidates"]), candidate_results=rows,
        compatible_candidates=[row["candidate_id"] for row in rows
                               if row["result"].get("ac_noise_ball_feasible")],
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
