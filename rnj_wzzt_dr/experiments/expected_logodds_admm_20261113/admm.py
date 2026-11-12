"""Two-block sharing ADMM; the nonconvex update is an exact sphere projection.

The device block is convex only for convex device domains. The original
hourly actuator is retained by MIQP device steps; these bounded inexact
updates do not inherit a general ADMM convergence or global-optimality claim.
"""
from time import perf_counter
import numpy as np


def project_outside(value, radius, fallback=None):
    norm = float(np.linalg.norm(value))
    if norm >= radius:
        return value.copy()
    if norm > 0:
        return value*(radius/norm)
    direction = np.zeros_like(value) if fallback is None else np.array(fallback, copy=True)
    size = float(np.linalg.norm(direction))
    if size == 0:
        direction.flat[0] = 1.; size = 1.
    return direction*(radius/size)


def sharing_projection(evidence, values):
    """argmin sum_r ||Z_r-V_r||² with ||b+sum_r Z_r||>=radius.

    At a prescribed total correction the least squared change is its equal
    distribution among active copies. No candidate independence is assumed.
    """
    result = np.zeros_like(values)
    for j in range(len(evidence.base)):
        active = evidence.active[j]
        if not np.any(active):
            raise ValueError("No resource changes this positive-gain constraint")
        total = evidence.base[j]+values[j, active].sum(axis=0)
        projected = project_outside(total, evidence.radii[j], evidence.base[j])
        result[j, active] = values[j, active]+(projected-total)/active.sum()
    return result


def run(evidence, steps, initial, fee, *, seconds=30., iterations=300, rho=10.,
        checkpoint=None, setup_seconds=0.):
    started = perf_counter(); deadline = started+max(0., seconds-setup_seconds)
    delivery = np.array(initial, copy=True)
    contributions = evidence.contributions(delivery)
    copies = contributions.copy(); dual = np.zeros_like(copies)
    quadratic = np.sum(evidence.effect**2, axis=(0, 1))
    trace = []; best = None; first_feasible = None; status = "iteration_limit"
    initial_fee = float(fee(delivery)); initial_gains = evidence.gains(delivery)

    def retain(iteration):
        nonlocal best, first_feasible
        gains = evidence.gains(delivery); cost = float(fee(delivery))
        if np.min(gains) >= evidence.gain_required-1e-8:
            if first_feasible is None:
                first_feasible = setup_seconds+perf_counter()-started
            if best is None or cost < best["fee"]:
                best = dict(iteration=iteration, fee=cost, gains=gains.copy(), delivery=delivery.copy(),
                            requests=[s.requests.copy() if s is not None else None for s in steps])
        return gains, cost

    retain(-1)
    completed = 0
    for iteration in range(iterations):
        if perf_counter() >= deadline:
            status = "time_limit"; break
        target = copies-dual
        linear = np.einsum("knr,krtn->rt", evidence.effect, target)
        reports = []; partial = False
        for r, step in enumerate(steps):
            if step is None:
                continue
            if perf_counter() >= deadline:
                partial = True; break
            delivery[r], report = step.solve(quadratic[r], linear[r], rho, deadline)
            reports.append(report)
        gains, cost = retain(iteration)
        if partial:
            trace.append(dict(iteration=iteration, partial_device_pass=True, fee=cost,
                              gains=gains, elapsed=setup_seconds+perf_counter()-started))
            status = "time_limit_partial_device_pass"; break
        contributions = evidence.contributions(delivery)
        previous = copies.copy()
        copies = sharing_projection(evidence, contributions+dual)
        residual = contributions-copies
        dual += residual
        active_size = int(np.sum(evidence.active))*evidence.base.shape[1]*evidence.base.shape[2]
        primal = float(np.linalg.norm(residual))
        dual_vector = rho*np.einsum("knr,krtn->rt", evidence.effect, copies-previous)
        dual_residual = float(np.linalg.norm(dual_vector))
        eps_primal = 1e-7*np.sqrt(active_size)+1e-5*max(np.linalg.norm(contributions), np.linalg.norm(copies))
        eps_dual = 1e-7*np.sqrt(delivery.size)+1e-5*np.linalg.norm(rho*np.einsum("knr,krtn->rt", evidence.effect, dual))
        row = dict(iteration=iteration, fee=cost, gains=gains, min_gain=float(min(gains)),
            primal=primal, dual=dual_residual, eps_primal=float(eps_primal), eps_dual=float(eps_dual), rho=rho,
            elapsed=setup_seconds+perf_counter()-started,
            local_optimal=sum(r["optimal"] for r in reports), local_updates=len(reports),
            local_without_incumbent=sum(r.get("no_incumbent", False) for r in reports))
        trace.append(row); completed += 1
        if checkpoint and (iteration % 10 == 0 or best is not None):
            checkpoint(row)
        if primal <= eps_primal and dual_residual <= eps_dual and np.min(gains) >= evidence.gain_required-1e-8:
            status = "residuals_small_and_original_gain_feasible"; break
        if (iteration+1) % 10 == 0:
            updated = min(rho*2, 1e5) if primal > 10*dual_residual else max(rho/2, .1) if dual_residual > 10*primal else rho
            if updated != rho:
                dual *= rho/updated
                rho = updated
    return dict(status=status, feasible=best is not None, best=best, initial_fee=initial_fee,
        initial_gains=initial_gains, first_feasible_seconds=first_feasible, iterations=completed, trace=trace,
        total_seconds=setup_seconds+perf_counter()-started,
        optimal_device_updates=sum(r.get("local_optimal", 0) for r in trace),
        total_device_updates=sum(r.get("local_updates", 0) for r in trace),
        global_lower_bound=None, global_optimality_proved=False)


class ScalarConvexDevice:
    """Analytic convex QP for independent one-dimensional checks."""
    def __init__(self, lower, upper, cost=1., initial=0.):
        self.lower, self.upper, self.cost = lower, upper, cost
        self.requests = np.array([initial], float)

    def solve(self, quadratic, linear, rho, deadline):
        value = float(np.clip(rho*linear[0]/(2*self.cost+rho*quadratic), self.lower, self.upper))
        self.requests[:] = value
        return np.array([value]), dict(optimal=True, no_incumbent=False)
