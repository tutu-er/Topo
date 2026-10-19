"""Independent Gaussian/projection audits of the EV marginal-price theory.

All functions here are toy mathematical checks. The Monte Carlo draws do not
enter the selector, its prediction samples, or statistical thresholds.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import platform

import numpy as np
import scipy
from scipy.special import ndtr


def _vector(value, name):
    x = np.asarray(value, float)
    if x.ndim != 1 or not np.isfinite(x).all():
        raise ValueError(f"{name} must be a finite vector")
    return x


def _symmetric(value, size, name, *, positive_semidefinite=False):
    x = np.asarray(value, float)
    if x.shape != (size, size) or not np.isfinite(x).all():
        raise ValueError(f"{name} must be a finite square matrix matching the vector")
    if not np.allclose(x, x.T, rtol=1e-11, atol=1e-11):
        raise ValueError(f"{name} must be symmetric")
    x = (x+x.T)/2
    if positive_semidefinite and np.linalg.eigvalsh(x).min(initial=0.) < -1e-10*max(1., np.linalg.norm(x, 2)):
        raise ValueError(f"{name} must be positive semidefinite")
    return x


def _projection(value, size):
    x = _symmetric(value, size, "projection")
    if not np.allclose(x@x, x, rtol=1e-10, atol=1e-10):
        raise ValueError("projection must be idempotent")
    return x


def gaussian_quadratic_moments(constant, linear, matrix, covariance):
    """Moments of constant+b.T eps+.5 eps.T C eps, eps~N(0,K).

    C is symmetric and K may be singular (fixed history has zero noise).
    Odd Gaussian moments vanish; Isserlis' fourth-moment identity gives the
    quadratic variance .5 tr(C K C K).
    """
    b = _vector(linear, "linear")
    c = _symmetric(matrix, len(b), "quadratic matrix")
    k = _symmetric(covariance, len(b), "covariance", positive_semidefinite=True)
    if not np.isfinite(constant):
        raise ValueError("constant must be finite")
    expectation = float(constant+.5*np.trace(c@k))
    linear_variance = float(b@k@b)
    quadratic_variance = float(.5*np.trace(c@k@c@k))
    variance = linear_variance+quadratic_variance
    if variance < -1e-9:
        raise ArithmeticError("computed Gaussian variance is negative")
    return dict(mean=expectation, variance=max(0., variance),
                linear_variance=max(0., linear_variance),
                quadratic_variance=max(0., quadratic_variance))


def projection_loss_moments(mean, center, projection, covariance):
    """Moments of .5||Pi(mean+eps-center)||^2 for a fixed projection Pi."""
    m, a = _vector(mean, "mean"), _vector(center, "center")
    if a.shape != m.shape:
        raise ValueError("mean and center dimensions differ")
    pi = _projection(projection, len(m))
    d = m-a
    result = gaussian_quadratic_moments(.5*d@pi@d, pi@d, pi, covariance)
    result.update(mean_prediction_loss=float(.5*d@pi@d),
                  noise_offset=float(result["mean"]-.5*d@pi@d))
    return result


def projection_margin_moments(mean, center_reference, center_competitor,
                              projection_reference, projection_competitor,
                              covariance):
    """Moments of competitor loss minus reference loss, both fixed faces."""
    m = _vector(mean, "mean")
    ar, ac = _vector(center_reference, "reference center"), _vector(center_competitor, "competitor center")
    if ar.shape != m.shape or ac.shape != m.shape:
        raise ValueError("mean and center dimensions differ")
    pr, pc = _projection(projection_reference, len(m)), _projection(projection_competitor, len(m))
    dr, dc = m-ar, m-ac
    constant = float(.5*(dc@pc@dc-dr@pr@dr))
    result = gaussian_quadratic_moments(constant, pc@dc-pr@dr, pc-pr, covariance)
    result.update(mean_prediction_margin=constant, noise_offset=float(result["mean"]-constant))
    return result


def projection_margin_price(mean, mean_jacobian, center_reference,
                            center_competitor, projection_reference,
                            projection_competitor):
    """Mean and expected marginal prices coincide when Pi and K stay fixed."""
    m = _vector(mean, "mean")
    ar, ac = _vector(center_reference, "reference center"), _vector(center_competitor, "competitor center")
    if ar.shape != m.shape or ac.shape != m.shape:
        raise ValueError("mean and center dimensions differ")
    pr, pc = _projection(projection_reference, len(m)), _projection(projection_competitor, len(m))
    j = np.asarray(mean_jacobian, float)
    if j.ndim != 2 or j.shape[0] != len(m) or not np.isfinite(j).all():
        raise ValueError("mean_jacobian must have matching finite rows")
    return j.T@(pc@(m-ac)-pr@(m-ar))


def active_boundary_expected_loss(u, sigma=1.):
    """E[.5 max(u+sigma*Z,0)^2], the profiled fit under theta<=0."""
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("sigma must be positive and finite")
    u = np.asarray(u, float)
    if not np.isfinite(u).all():
        raise ValueError("u must be finite")
    z = u/sigma
    density = np.exp(-.5*z*z)/np.sqrt(2*np.pi)
    return .5*((u*u+sigma*sigma)*ndtr(z)+u*sigma*density)


def active_boundary_expected_price(u, sigma=1.):
    """Derivative of active_boundary_expected_loss w.r.t. u."""
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("sigma must be positive and finite")
    u = np.asarray(u, float)
    if not np.isfinite(u).all():
        raise ValueError("u must be finite")
    z = u/sigma
    return u*ndtr(z)+sigma*np.exp(-.5*z*z)/np.sqrt(2*np.pi)


def fixed_gaussian_llr_moments(mean_reference, mean_competitor, covariance):
    """Fixed-model log(p_ref/p_comp), data drawn under ref, equal covariance.

    gamma=d.T Sigma^-1 d; the LLR is N(gamma/2,gamma). Error at gamma=0
    uses randomized ties. This distribution is not a refitted-model test.
    """
    a, b = _vector(mean_reference, "reference mean"), _vector(mean_competitor, "competitor mean")
    if a.shape != b.shape:
        raise ValueError("Gaussian mean dimensions differ")
    sigma = _symmetric(covariance, len(a), "covariance", positive_semidefinite=True)
    if np.linalg.eigvalsh(sigma).min() <= 0:
        raise ValueError("fixed Gaussian covariance must be positive definite")
    d = a-b
    gamma = max(0., float(d@np.linalg.solve(sigma, d)))
    return dict(gamma=gamma, KL=.5*gamma, llr_mean=.5*gamma,
                llr_variance=gamma, equal_prior_error=float(ndtr(-np.sqrt(gamma)/2)))


def energy_transfer_price(power_gradient, dt_hours, source_slot, destination_slot):
    """Price per kWh moved from one slot to another, net grid energy fixed."""
    price = _vector(power_gradient, "power gradient")
    dt = np.broadcast_to(np.asarray(dt_hours, float), price.shape)
    if not np.isfinite(dt).all() or np.any(dt <= 0):
        raise ValueError("positive finite slot durations required")
    for index in (source_slot, destination_slot):
        if not isinstance(index, (int, np.integer)) or not 0 <= index < len(price):
            raise ValueError("slot outside price vector")
    return float(price[destination_slot]/dt[destination_slot]-price[source_slot]/dt[source_slot])


def _mc_check(samples, expected, standard_errors=6.):
    x = np.asarray(samples, float)
    estimate = float(np.mean(x))
    se = float(np.std(x, ddof=1)/np.sqrt(len(x)))
    difference = estimate-float(expected)
    return dict(expected=float(expected), estimate=estimate, standard_error=se,
                difference=difference, tolerance=standard_errors*se+1e-11,
                passed=bool(abs(difference) <= standard_errors*se+1e-11))


def run_theory_checks(*, sample_count=100000, seed=17031013, standard_errors=6.):
    """Run independent audits and return small JSON-safe numerical evidence."""
    if not isinstance(sample_count, (int, np.integer)) or not 100 <= sample_count <= 200000:
        raise ValueError("sample_count must be an integer between 100 and 200000")
    if not np.isfinite(standard_errors) or standard_errors <= 0:
        raise ValueError("positive Monte Carlo tolerance required")
    rng = np.random.default_rng(seed)
    n, historical_rows = 6, 2
    # Two fixed linear least-squares fitting spaces, with shared columns that
    # simultaneously touch history and future. Fixed history has zero noise.
    qr, _ = np.linalg.qr(rng.normal(size=(n, 2)))
    qc, _ = np.linalg.qr(rng.normal(size=(n, 3)))
    pr, pc = np.eye(n)-qr@qr.T, np.eye(n)-qc@qc.T
    covariance = np.diag([0., 0., 1., 1., 1., 1.])
    mean = np.array([.7, -.4, 1.2, -.3, .8, .2])
    ar = np.array([.1, -.1, .3, .1, -.2, .0])
    ac = np.array([-.2, .2, -.1, .2, .0, -.3])
    jac = np.array([[0., 0.], [0., 0.], [.8, -.2], [-.1, .4], [.3, .6], [-.2, .5]])
    eps = rng.normal(size=(sample_count, n))*np.sqrt(np.diag(covariance))
    dr, dc = mean-ar, mean-ac
    sample_ref = .5*np.einsum("ni,ij,nj->n", dr+eps, pr, dr+eps)
    sample_comp = .5*np.einsum("ni,ij,nj->n", dc+eps, pc, dc+eps)
    sample_margin = sample_comp-sample_ref
    ref_moments = projection_loss_moments(mean, ar, pr, covariance)
    margin_moments = projection_margin_moments(mean, ar, ac, pr, pc, covariance)
    projection_checks = dict(
        reference_mean=_mc_check(sample_ref, ref_moments["mean"], standard_errors),
        reference_variance=_mc_check((sample_ref-ref_moments["mean"])**2,
                                     ref_moments["variance"], standard_errors),
        margin_mean=_mc_check(sample_margin, margin_moments["mean"], standard_errors),
        margin_variance=_mc_check((sample_margin-margin_moments["mean"])**2,
                                  margin_moments["variance"], standard_errors))
    price = projection_margin_price(mean, jac, ar, ac, pr, pc)
    price_samples = ((dc+eps)@pc-(dr+eps)@pr)@jac
    price_checks = [_mc_check(price_samples[:, j], price[j], standard_errors) for j in range(jac.shape[1])]
    # At fixed Pi/K, the mean-to-expected correction is constant in control.
    h = 1e-5
    expected_fd, mean_fd = [], []
    for j in range(jac.shape[1]):
        plus = projection_margin_moments(mean+h*jac[:, j], ar, ac, pr, pc, covariance)
        minus = projection_margin_moments(mean-h*jac[:, j], ar, ac, pr, pc, covariance)
        expected_fd.append((plus["mean"]-minus["mean"])/(2*h))
        mean_fd.append((plus["mean_prediction_margin"]-minus["mean_prediction_margin"])/(2*h))
    exact_gradient_check = bool(np.allclose(price, expected_fd, rtol=1e-8, atol=1e-9) and
                                np.allclose(price, mean_fd, rtol=1e-8, atol=1e-9))
    increments = []
    direction = jac[:, 0]
    curvature = float(direction@(pc-pr)@direction)
    for delta in (.1, .03, .01, .003):
        shifted_ref = .5*np.einsum("ni,ij,nj->n", dr+eps+delta*direction, pr, dr+eps+delta*direction)
        shifted_comp = .5*np.einsum("ni,ij,nj->n", dc+eps+delta*direction, pc, dc+eps+delta*direction)
        unit_increment = (shifted_comp-shifted_ref-sample_margin)/delta
        finite_mean = float(price[0]+.5*delta*curvature)
        check = _mc_check(unit_increment, finite_mean, standard_errors)
        increments.append(dict(delta=delta, analytic_price=float(price[0]),
            finite_increment_mean=finite_mean, finite_to_price_error=finite_mean-float(price[0]),
            common_random_numbers=True, check=check))
    z = rng.normal(size=sample_count)
    boundary = []
    for u in (-1., 0., 1.):
        clipped = np.maximum(u+z, 0.)
        boundary.append(dict(u=u, mean_prediction_loss=.5*max(u, 0.)**2,
            mean_prediction_price=max(u, 0.),
            expected_loss=float(active_boundary_expected_loss(u)),
            expected_price=float(active_boundary_expected_price(u)),
            loss_check=_mc_check(.5*clipped**2, active_boundary_expected_loss(u), standard_errors),
            price_check=_mc_check(clipped, active_boundary_expected_price(u), standard_errors)))
    ga = np.array([.8, -.6]); gb = np.zeros(2)
    llr_moments = fixed_gaussian_llr_moments(ga, gb, np.eye(2))
    llr = .5*llr_moments["gamma"]+rng.normal(size=(sample_count, 2))@(ga-gb)
    llr_checks = dict(mean=_mc_check(llr, llr_moments["llr_mean"], standard_errors),
        variance=_mc_check((llr-llr_moments["llr_mean"])**2, llr_moments["llr_variance"], standard_errors),
        equal_prior_error=_mc_check((llr < 0).astype(float), llr_moments["equal_prior_error"], standard_errors))
    energy_price = np.array([2., 3.]); dt = np.array([.25, .5])
    transfer = energy_transfer_price(energy_price, dt, 0, 1)
    transfer_energy = .01
    du = np.array([-transfer_energy/dt[0], transfer_energy/dt[1]])
    energy_check = dict(power_price=energy_price.tolist(), dt_hours=dt.tolist(),
        source_slot=0, destination_slot=1, price_per_kwh=transfer,
        energy_transfer_kwh=transfer_energy, power_change_kw=du.tolist(),
        energy_change_kwh=float(dt@du), score_change_linear=float(energy_price@du),
        predicted_score_change=float(transfer_energy*transfer),
        passed=bool(abs(dt@du) < 1e-12 and abs(energy_price@du-transfer_energy*transfer) < 1e-12))
    checks = [*projection_checks.values(), *price_checks,
              *(i["check"] for i in increments),
              *(b[k] for b in boundary for k in ("loss_check", "price_check")),
              *llr_checks.values()]
    config = dict(sample_count=int(sample_count), seed=int(seed), mc_standard_errors=float(standard_errors),
        dimension=n, historical_rows=historical_rows, future_covariance="identity",
        history_covariance="zero", monte_carlo_usage="independent_audit_only")
    source_paths = [Path(__file__), Path(__file__).with_name("test_gradient_price_theory_checks.py")]
    hashes = {p.name:sha256(p.read_bytes()).hexdigest() for p in source_paths if p.exists()}
    return dict(config=config, config_sha256=sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
        source_sha256=hashes, created_utc=datetime.now(timezone.utc).isoformat(),
        environment=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__),
        assumptions=dict(projection="fixed symmetric idempotent fitting projection",
            noise="zero-mean Gaussian with control-independent covariance",
            boundary="changing active face is a separate counterexample",
            llr="two fixed Gaussian models, not joint re-fitting",
            gradient="limit of the expectation of coupled unit finite increments",
            energy="net grid energy fixed; efficiencies/SOC constrain which transfers are feasible"),
        projection_moments=dict(reference=ref_moments, margin=margin_moments, checks=projection_checks),
        projection_prices=dict(analytic=price.tolist(), expected_central_difference=expected_fd,
            mean_central_difference=mean_fd, monte_carlo=price_checks, deterministic_check=exact_gradient_check),
        unit_random_increments=increments, active_boundary=boundary,
        fixed_gaussian_llr=dict(analytic=llr_moments, checks=llr_checks),
        equal_energy_transfer=energy_check, monte_carlo_check_count=len(checks),
        monte_carlo_failure_count=sum(not c["passed"] for c in checks),
        passed=bool(all(c["passed"] for c in checks) and exact_gradient_check and energy_check["passed"]),
        affects_policy_or_thresholds=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=17031013)
    parser.add_argument("--output", type=Path,
        default=Path(__file__).parent/"evidence"/"gradient_price"/"theory_checks.json")
    args = parser.parse_args(argv)
    evidence = run_theory_checks(sample_count=args.samples, seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(dict(output=str(args.output.resolve()), passed=evidence["passed"],
        samples=args.samples, monte_carlo_checks=evidence["monte_carlo_check_count"],
        failures=evidence["monte_carlo_failure_count"]), ensure_ascii=False))
    return 0 if evidence["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
