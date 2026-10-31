"""Independent whole-episode sampling of a frozen continuous-price plan.

This evaluates excitation and resource/budget delivery under the declared law.
It neither creates voltages nor estimates topology-recovery probabilities.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import beta

from .model import PriceProblem, evaluate_plan


def realize_factors(problem: PriceProblem, amplitudes, factors):
    """Map externally supplied factor rows to full trajectories without clipping."""
    x = np.asarray(amplitudes, dtype=float)
    z = np.asarray(factors, dtype=float)
    if x.shape != (problem.n_contracts,) or not np.isfinite(x).all() or (x < 0).any():
        raise ValueError("one finite nonnegative amplitude per contract required")
    if (z.ndim != 2 or z.shape[1] != problem.n_factors or not np.isfinite(z).all()
            or (np.abs(z) > 1).any()):
        raise ValueError("factor rows must be finite and in [-1,1]")
    actual = x[None, :]*(1+z@problem.response_factors.T)
    increments = actual@problem.profiles.T
    background = problem.mu0[None, :]+z@problem.background_factors.T
    total = background+increments
    # Evaluate the algebraically identical affine benefit directly. This
    # retains exact deterministic boundary values instead of introducing an
    # avoidable floating-point comparison difference from matrix reassociation.
    response_factor = (problem.profiles*x)@problem.response_factors
    if problem.benefit_mode == "incremental":
        mean = problem.benefit_directions@(problem.profiles@x)
        loading = problem.benefit_directions@response_factor
    else:
        mean = problem.benefit_offsets+problem.benefit_directions@(problem.mu0+problem.profiles@x)
        loading = problem.benefit_directions@(problem.background_factors+response_factor)
    benefits = mean[None, :]+z@loading.T
    prices = x/problem.elasticities
    payment = actual@(prices*problem.reduction_hours)
    return dict(factors=z, actual_amplitudes=actual, increments=increments,
                net_load=total, background=background, benefits=benefits,
                payments=payment)


def validate_frozen_plan(problem: PriceProblem, amplitudes, *, threshold=None,
                         samples=20000, seed=41031, batch_size=512,
                         numerical_tolerance=1e-7):
    """Sample once after plan freeze, keeping every complete episode in denominator.

    The returned Clopper-Pearson interval describes Monte Carlo success under
    the explicit factor law. Cantelli is the separate analytic guarantee.
    Only aggregate summaries are returned; raw P/Q trajectories stay private.
    """
    for name, value in (("samples", samples), ("batch_size", batch_size)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value <= 0:
            raise ValueError(name+" must be a positive integer")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if not np.isfinite(numerical_tolerance) or numerical_tolerance <= 0:
        raise ValueError("numerical_tolerance must be finite and positive")
    analytic = evaluate_plan(problem, amplitudes, tolerance=numerical_tolerance)
    no_dr_analytic = evaluate_plan(problem, np.zeros(problem.n_contracts), tolerance=numerical_tolerance)
    if not analytic["numerical_feasible"]:
        raise ValueError("validation requires a numerically feasible frozen plan")
    if threshold is None:
        threshold = analytic["reliable_benefit"]
    threshold = float(threshold)
    if not np.isfinite(threshold):
        raise ValueError("threshold must be finite")
    rng = np.random.default_rng(int(seed))
    k = problem.benefit_directions.shape[0]
    sums, squares = np.zeros(k), np.zeros(k)
    success = zero_success = budget_fail = device_fail = amplitude_fail = 0
    zero_sums = np.zeros(k)
    payment_sum = payment_max = maximum_violation = 0.
    left = int(samples)
    while left:
        size = min(int(batch_size), left)
        z = rng.uniform(-1., 1., (size, problem.n_factors))
        realization = realize_factors(problem, amplitudes, z)
        values, actual = realization["benefits"], realization["actual_amplitudes"]
        # The benefit event uses its exact threshold, not a relaxed tolerance.
        success += int(np.count_nonzero(np.all(values >= threshold, axis=1)))
        if problem.benefit_mode == "incremental":
            zero_values = np.zeros_like(values)
        else:
            zero_mean = problem.benefit_offsets+problem.benefit_directions@problem.mu0
            zero_loading = problem.benefit_directions@problem.background_factors
            zero_values = zero_mean[None, :]+z@zero_loading.T
        zero_success += int(np.count_nonzero(np.all(zero_values >= threshold, axis=1)))
        zero_sums += zero_values.sum(axis=0)
        sums += values.sum(axis=0)
        squares += (values*values).sum(axis=0)
        payments = realization["payments"]
        payment_sum += float(payments.sum())
        payment_max = max(payment_max, float(payments.max(initial=0.)))
        bviol = payments-problem.budget
        budget_fail += int(np.count_nonzero(bviol > numerical_tolerance))
        if len(problem.device_rhs):
            residual = actual@problem.device_matrix.T-problem.device_rhs
            dviol = residual.max(axis=1)
            device_fail += int(np.count_nonzero(dviol > numerical_tolerance))
            maximum_violation = max(maximum_violation, float(dviol.max(initial=0.)))
        aviol = np.maximum((-actual).max(axis=1),
                           (actual-problem.actual_amplitude_upper).max(axis=1))
        amplitude_fail += int(np.count_nonzero(aviol > numerical_tolerance))
        maximum_violation = max(maximum_violation, float(bviol.max(initial=0.)),
                                float(aviol.max(initial=0.)))
        left -= size
    mean = sums/samples
    std = np.sqrt(np.maximum(0., squares/samples-mean*mean))
    alpha = .05
    interval = [0. if success == 0 else float(beta.ppf(alpha/2, success, samples-success+1)),
                1. if success == samples else float(beta.ppf(1-alpha/2, success+1, samples-success))]
    return dict(status="complete", sample_unit="one complete joint factor/response trajectory",
                samples=int(samples), seed=int(seed), threshold=threshold,
                successes=success, failures=int(samples)-success,
                empirical_success_rate=success/samples,
                monte_carlo_95pct_interval=interval,
                benefit_means=mean.tolist(), benefit_standard_deviations=std.tolist(),
                analytic_benefit_means=analytic["benefit_means"],
                analytic_benefit_standard_deviations=analytic["benefit_standard_deviations"],
                mean_payment=payment_sum/samples, maximum_payment=payment_max,
                analytic_expected_payment=analytic["expected_cost"],
                analytic_hard_budget_bound=analytic["hard_budget_bound"],
                paired_no_dr=dict(samples=int(samples), successes=zero_success,
                                  failures=int(samples)-zero_success,
                                  numerical_resource_feasible=no_dr_analytic["numerical_feasible"],
                                  reliable_benefit=no_dr_analytic["reliable_benefit"],
                                  mean_benefits=(zero_sums/samples).tolist(), payment=0.,
                                  mean_benefit_change=(mean-zero_sums/samples).tolist(),
                                  comparison="same background factors, no purchased response"),
                budget_failures=budget_fail, device_failures=device_fail,
                amplitude_failures=amplitude_fail, maximum_violation=maximum_violation,
                risk_scope="excitation under the declared factor law; not topology recovery",
                sampling_law="independent uniform[-1,1] common factors, shared within each episode",
                raw_trajectory_exported=False)
