"""Bounded feasibility restoration of the original finite probability target.

This append-only solver does not modify the previously frozen penalty solver.
Phase I gives probability slack lexicographic priority over movement.  Phase II
minimizes perspective cost with a HARD linearized probability constraint.  Every
accepted step is checked by problem.evaluate, using only its fixed integration
sample.  No diagnostic/validation probability is read here.
"""
from __future__ import annotations

from pathlib import Path
import sys
import time

import gurobipy as gp
from gurobipy import GRB
import numpy as np

SOURCE = Path(__file__).resolve().parents[1]
SOURCE = SOURCE/"src" if (SOURCE/"src").is_dir() else SOURCE
sys.path.insert(0, str(SOURCE))
from spatial_optimizer import cost, feasible


PROBABILITY_TOLERANCE = 1e-7
IMPROVEMENT_TOLERANCE = 1e-10


def _step(u0, a0, p0, gradient, caps, fixed_capacity, target, radius, phase, seconds):
    """One convex numerical subproblem, not a bound on the nonlinear risk."""
    m, T = u0.shape
    model = gp.Model("bounded_probability_restoration")
    model.Params.OutputFlag = 0
    model.Params.Threads = 1
    model.Params.TimeLimit = max(0., float(seconds))
    model.Params.FeasibilityTol = 1e-9
    model.Params.OptimalityTol = 1e-9
    try:
        a = model.addMVar(m, lb=np.zeros(m), ub=caps, name="capacity")
        if fixed_capacity is not None:
            a.lb = fixed_capacity
            a.ub = fixed_capacity
        u = model.addMVar((m, T), lb=-np.broadcast_to(caps[:, None], (m, T)),
                         ub=np.broadcast_to(caps[:, None], (m, T)), name="mean")
        model.addConstr(a.sum() <= 2.)
        for i in range(m):
            model.addConstr(u[i, :] <= a[i])
            model.addConstr(u[i, :] >= -a[i])
        model.addConstr(u <= u0+radius)
        model.addConstr(u >= u0-radius)
        linear_probability = p0+(gradient*(u-u0)).sum()
        slack = None
        if phase == "restore":
            slack = model.addVar(lb=0., name="probability_shortfall")
            model.addConstr(linear_probability+slack >= target)
            movement_u = model.addMVar((m, T), lb=0., name="abs_mean_move")
            movement_a = model.addMVar(m, lb=0., name="abs_capacity_move")
            model.addConstr(movement_u >= u-u0)
            model.addConstr(movement_u >= u0-u)
            model.addConstr(movement_a >= a-a0)
            model.addConstr(movement_a >= a0-a)
            # Both objectives are linear.  No finite coefficient trades target
            # violation for a lower fee.  Numerical objective tolerances only.
            model.setObjectiveN(slack, 0, priority=2, weight=1., abstol=1e-12, reltol=0.)
            model.setObjectiveN(movement_u.sum()+movement_a.sum(), 1, priority=1,
                                weight=1., abstol=1e-12, reltol=0.)
        else:
            model.addConstr(linear_probability >= target)
            epi = model.addMVar(m, lb=0., name="energy_over_capacity")
            for i in range(m):
                model.addQConstr(gp.quicksum(u[i, t].item()*u[i, t].item() for t in range(T))
                                 <= a[i].item()*epi[i].item())
            model.setObjective(.5*epi.sum(), GRB.MINIMIZE)
        model.optimize()
        info = dict(status=int(model.Status), runtime=float(model.Runtime), solutions=int(model.SolCount), phase=phase)
        if model.SolCount == 0:
            return None, info
        raw_u, raw_a = np.asarray(u.X), np.asarray(a.X)
        aa = np.clip(raw_a, 0., caps)
        if fixed_capacity is not None:
            aa = fixed_capacity.copy()
        elif aa.sum() > 2.:
            aa *= 2./aa.sum()
        uu = np.clip(raw_u, -aa[:, None], aa[:, None])
        uu[aa == 0.] = 0.
        info.update(linearized_probability=float(p0+np.sum(gradient*(uu-u0))),
                    slack=float(slack.X) if slack is not None else 0.,
                    movement=float(np.abs(uu-u0).sum()+np.abs(aa-a0).sum()),
                    projection_max=float(max(np.max(np.abs(uu-raw_u), initial=0.),
                                             np.max(np.abs(aa-raw_a), initial=0.))))
        return (uu, aa), info
    finally:
        model.dispose()


def refine(problem, caps, u, a, fixed_capacity=None, target=.95, maxiter=12, seconds=60):
    """Restore feasibility, then reduce cost within one bounded common budget.

    problem.evaluate(u) must return probability and a flattened or shaped
    gradient.  Its integration nodes are never changed.  The probability
    target is checked with the original 1e-7 numerical feasibility tolerance.
    Returns best/status/trace plus scope and feasibility flags.  A nonfeasible
    result is a diagnostic, not evidence of physical or mathematical infeasibility.
    """
    u, a, caps = np.asarray(u, float).copy(), np.asarray(a, float).copy(), np.asarray(caps, float).copy()
    if not np.isfinite([target, seconds]).all() or not 0 < target <= 1 or seconds <= 0:
        raise ValueError("finite target in (0,1] and positive wall budget required")
    if isinstance(maxiter, bool) or int(maxiter) != maxiter or maxiter < 0:
        raise ValueError("nonnegative integer iteration budget required")
    if not feasible(u, a, caps):
        raise ValueError("initial resources are infeasible")
    if fixed_capacity is not None:
        fixed_capacity = np.asarray(fixed_capacity, float).copy()
        if fixed_capacity.shape != a.shape or not np.allclose(a, fixed_capacity, atol=1e-10, rtol=0):
            raise ValueError("initial allocation must equal the fixed allocation")
        if not feasible(u, fixed_capacity, caps):
            raise ValueError("fixed allocation is infeasible")
        a = fixed_capacity.copy()
    started = time.perf_counter()
    radius = .3
    trace = []
    status = "iteration_limit"
    accepted_steps = 0
    objective_checks = 0
    best = None

    def evaluate(point):
        nonlocal objective_checks
        result = problem.evaluate(point)
        objective_checks += 1
        probability = float(result["probability"])
        if not np.isfinite(probability) or not 0 <= probability <= 1:
            raise ArithmeticError("probability evaluator returned invalid value")
        return result, probability

    def retain(point, allocation, probability):
        nonlocal best
        if not feasible(point, allocation, caps):
            raise AssertionError("evaluated candidate violates original resources")
        candidate = dict(mean_u=point.tolist(), capacity_kw=allocation.tolist(),
                         cost=cost(point, allocation), planning_probability=float(probability))
        qualified = probability >= target-PROBABILITY_TOLERANCE
        best_qualified = best is not None and best["planning_probability"] >= target-PROBABILITY_TOLERANCE
        if (best is None or (qualified and not best_qualified)
            or (qualified and best_qualified and candidate["cost"] < best["cost"])
            or (not qualified and not best_qualified and
                (probability, -candidate["cost"]) > (best["planning_probability"], -best["cost"]))):
            best = candidate

    result, probability = evaluate(u)
    retain(u, a, probability)
    for iteration in range(int(maxiter)):
        remaining = seconds-(time.perf_counter()-started)
        if remaining <= 0:
            status = "time_limit"
            break
        gradient = result.get("gradient")
        if gradient is None:
            status = "probability_gradient_unresolved"
            break
        gradient = np.asarray(gradient, float).reshape(u.shape)
        if not np.isfinite(gradient).all():
            status = "probability_gradient_unresolved"
            break
        phase = "reduce_cost" if probability >= target-PROBABILITY_TOLERANCE else "restore"
        if phase == "reduce_cost" and cost(u, a) == 0:
            status = "zero_cost_feasible"
            break
        step, info = _step(u, a, probability, gradient, caps, fixed_capacity,
                           target, radius, phase, min(15., remaining))
        row = dict(iteration=iteration+1, phase=phase, probability=probability,
                   cost=cost(u, a), radius=radius, subproblem=info, trials=[])
        if step is None:
            row["accepted"] = False
            trace.append(row)
            status = "convex_subproblem_unresolved"
            break
        trial_u, trial_a = step
        predicted_gain = float(np.sum(gradient*(trial_u-u)))
        if phase == "restore" and predicted_gain <= IMPROVEMENT_TOLERANCE:
            row["accepted"] = False
            trace.append(row)
            status = "linearized_restoration_stalled"
            break
        if phase == "reduce_cost" and cost(trial_u, trial_a) >= cost(u, a)-1e-9:
            row["accepted"] = False
            trace.append(row)
            status = "no_cost_decrease_in_local_step"
            break
        accepted = False
        old_u, old_a, old_p = u, a, probability
        old_cost = cost(u, a)
        # Interpolation preserves the convex original capacity/resource domain.
        for backtrack in range(6):
            if time.perf_counter()-started >= seconds:
                status = "time_limit"
                break
            fraction = 2.**(-backtrack)
            candidate_u = old_u+fraction*(trial_u-old_u)
            candidate_a = old_a+fraction*(trial_a-old_a)
            candidate_result, candidate_p = evaluate(candidate_u)
            candidate_cost = cost(candidate_u, candidate_a)
            retain(candidate_u, candidate_a, candidate_p)
            if phase == "restore":
                accept = candidate_p > old_p+IMPROVEMENT_TOLERANCE
            else:
                accept = candidate_p >= target-PROBABILITY_TOLERANCE and candidate_cost < old_cost-1e-9
            row["trials"].append(dict(fraction=fraction, probability=candidate_p,
                                      cost=candidate_cost, accepted=bool(accept),
                                      gradient_available=candidate_result.get("gradient") is not None))
            if accept:
                u, a, probability, result = candidate_u, candidate_a, candidate_p, candidate_result
                accepted_steps += 1
                accepted = True
                radius = min(.8, radius*1.25) if backtrack == 0 else max(1e-6, radius*.5)
                break
        row.update(accepted=accepted, elapsed=time.perf_counter()-started)
        trace.append(row)
        if status == "time_limit":
            break
        if not accepted:
            radius *= .5
        if radius < 1e-6:
            status = "trust_region_stalled"
            break
    return dict(best=best, status=status, trace=trace,
                planning_feasible=best["planning_probability"] >= target-PROBABILITY_TOLERANCE,
                target=float(target), target_tolerance=PROBABILITY_TOLERANCE,
                elapsed=time.perf_counter()-started, iterations=len(trace), accepted_steps=accepted_steps,
                probability_evaluations=objective_checks, fixed_capacity=fixed_capacity is not None,
                global_optimum_proven=False, infeasibility_proven=False,
                algorithm="lexicographic slack/movement restoration then hard linearized cost steps; original finite probability acceptance",
                probability_scope="unchanged evaluator sample only; diagnostic samples and actual future never read")
