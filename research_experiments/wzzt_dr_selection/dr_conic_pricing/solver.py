"""Convex conic price planning and binary contract-type selection.

Gurobi is imported at solve time.  The explicit norm epigraphs are standard
second-order cones; NonConvex=0 prevents silently invoking a nonconvex solver.
"""
from __future__ import annotations

import math
from pathlib import Path
import time

import numpy as np

from .model import PriceProblem, evaluate_plan


_STATUS = {1: "loaded", 2: "optimal", 3: "infeasible", 4: "infeasible_or_unbounded",
           5: "unbounded", 6: "cutoff", 7: "iteration_limit", 8: "node_limit",
           9: "time_limit", 10: "solution_limit", 11: "interrupted", 12: "numeric",
           13: "suboptimal", 14: "in_progress", 15: "objective_limit",
           16: "work_limit", 17: "memory_limit"}


def _finite_attr(model, name):
    try:
        value = float(model.getAttr(name))
        return value if math.isfinite(value) and abs(value) < 1e90 else None
    except Exception:
        return None


def _stage_record(model, sense):
    code = int(model.Status)
    incumbent = _finite_attr(model, "ObjVal") if model.SolCount else None
    bound = _finite_attr(model, "ObjBound")
    # Continuous conic optimal models do not always expose ObjBound.  At an
    # OPTIMAL receipt ObjVal is the solver's optimum, but no unavailable field
    # is fabricated as zero gap.
    absolute_gap = None if incumbent is None or bound is None else abs(incumbent-bound)
    relative_gap = None if absolute_gap is None else absolute_gap/max(abs(incumbent), 1e-10)
    return dict(status=_STATUS.get(code, f"status_{code}"), status_code=code,
                objective_sense=sense, objective_value=incumbent, best_bound=bound,
                absolute_gap=absolute_gap, relative_gap=relative_gap,
                solver_reported_optimal=(code == 2), solution_count=int(model.SolCount),
                runtime=_finite_attr(model, "Runtime"), work=_finite_attr(model, "Work"),
                node_count=_finite_attr(model, "NodeCount"),
                barrier_iterations=_finite_attr(model, "BarIterCount"),
                simplex_iterations=_finite_attr(model, "IterCount"))


def _allowed(problem, allowed_contracts, mode):
    n = problem.n_contracts
    if allowed_contracts is None:
        if mode == "socp" and problem.exclusive_groups:
            raise ValueError("SOCP with mutually exclusive types needs explicit allowed_contracts; use MISOCP to select types")
        indices = set(range(n))
    else:
        supplied = list(allowed_contracts)
        if (any(isinstance(i, (bool, np.bool_)) or not isinstance(i, (int, np.integer))
                or not 0 <= i < n for i in supplied) or len(set(supplied)) != len(supplied)):
            raise ValueError("allowed_contracts must contain distinct valid contract indices")
        indices = set(int(i) for i in supplied)
    if mode == "socp" and any(len(indices.intersection(group)) > 1 for group in problem.exclusive_groups):
        raise ValueError("SOCP allowed_contracts cannot contain multiple types for one physical device")
    return indices


def solve_price(problem: PriceProblem, *, mode="socp", allowed_contracts=None,
                time_limit=60., relative_gap=1e-6, feasibility_tolerance=1e-7,
                lex_absolute_tolerance=1e-7, lex_relative_tolerance=1e-7,
                threads=1, seed=0, log_path=None):
    """Maximize a Cantelli reliable-benefit bound, then minimize expected fee.

    Stage two runs only after stage one reports optimality, and retains its
    benefit up to the declared lexicographic tolerance.  Time limit is shared.
    A time-limited incumbent is returned as such, never reported as optimal.
    """
    if not isinstance(problem, PriceProblem):
        raise TypeError("problem must be a validated PriceProblem")
    if mode not in ("socp", "misocp"):
        raise ValueError("mode must be 'socp' or 'misocp'")
    numbers = (time_limit, relative_gap, feasibility_tolerance, lex_absolute_tolerance,
               lex_relative_tolerance)
    if (not np.isfinite(numbers).all() or time_limit <= 0 or relative_gap < 0
            or not 1e-9 <= feasibility_tolerance <= 1e-2
            or lex_absolute_tolerance < 0 or lex_relative_tolerance < 0):
        raise ValueError("invalid solve limits or tolerances")
    if (isinstance(threads, bool) or not isinstance(threads, (int, np.integer)) or threads < 1
            or isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or not 0 <= seed <= 2000000000):
        raise ValueError("threads must be a positive integer and seed an integer in [0,2000000000]")
    allowed = _allowed(problem, allowed_contracts, mode)
    import gurobipy as gp
    from gurobipy import GRB

    start = time.monotonic()
    parameters = dict(time_limit=float(time_limit), relative_gap=float(relative_gap),
                      feasibility_tolerance=float(feasibility_tolerance),
                      lex_absolute_tolerance=float(lex_absolute_tolerance),
                      lex_relative_tolerance=float(lex_relative_tolerance),
                      threads=int(threads), seed=int(seed), nonconvex=0,
                      barrier_qcp_convergence_tolerance=min(1e-8, float(feasibility_tolerance)),
                      dual_reductions=0, numeric_focus=2,
                      allowed_contracts=sorted(allowed))
    result = dict(schema_version=1, mode=mode, status="not_solved", has_incumbent=False,
                  feasible_incumbent=False, amplitudes=None, prices=None,
                  achieved_benefit_lower_bound=None, primary=None, secondary=None,
                  evaluation=None, binary_activations=None, positive_contract_ids=[],
                  parameters=parameters, gurobi_version=list(gp.gurobi.version()),
                  objective_scope="maximize a conservative reliable-excitation lower bound, not exact quantile or topology recovery",
                  confidence=problem.confidence, benefit_mode=problem.benefit_mode,
                  secondary_scope="minimum expected fee within declared primary benefit tolerance")
    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.start()
    model = gp.Model("reliable_excitation_price", env=env)
    try:
        model.Params.OutputFlag = 0
        if log_path is not None:
            path = Path(log_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            model.Params.LogFile = str(path)
            model.Params.LogToConsole = 0
            model.Params.OutputFlag = 1
        model.Params.Threads = int(threads)
        model.Params.Seed = int(seed)
        model.Params.NonConvex = 0
        # MISOCP outer approximation can report an incumbent within its own
        # scaled feasibility checks that fails the original conic benefit
        # floor on recomputation. Emphasize numerical accuracy and still run
        # the independent validation/fallback; never relabel a rejected point.
        model.Params.NumericFocus = 2
        model.Params.FeasibilityTol = float(feasibility_tolerance)
        model.Params.OptimalityTol = min(1e-8, float(feasibility_tolerance))
        # An unnecessarily tighter 1e-9 barrier gap caused numerical status 13
        # on an elementary one-contract cone despite <1e-8 primal/dual gaps.
        # This documented 1e-8 convergence setting changes no model constraint;
        # every returned point is still checked against the declared FeasTol.
        model.Params.BarQCPConvTol = min(1e-8, float(feasibility_tolerance))
        model.Params.DualReductions = 0
        model.Params.MIPGap = float(relative_gap)
        model.Params.TimeLimit = max(1e-9, float(time_limit)-(time.monotonic()-start))

        n, factors = problem.n_contracts, problem.n_factors
        caps = problem.amplitude_upper.copy()
        caps[[i for i in range(n) if i not in allowed]] = 0.
        x = model.addVars(n, lb=0., ub={i: float(caps[i]) for i in range(n)}, name="expected_amplitude")
        tau = model.addVar(lb=-GRB.INFINITY, name="reliable_benefit")
        y = None
        if mode == "misocp":
            y = model.addVars(n, vtype=GRB.BINARY, name="contract_type")
            for j in range(n):
                model.addConstr(x[j] <= float(caps[j])*y[j], name=f"activate_{j}")
                if j not in allowed or caps[j] == 0:
                    y[j].UB = 0.
            for group_index, group in enumerate(problem.exclusive_groups):
                model.addConstr(gp.quicksum(y[j] for j in group) <= 1,
                                name=f"one_type_{group_index}")

        g_h = problem.benefit_directions@problem.profiles
        f_g = problem.benefit_directions@problem.background_factors
        base = problem.benefit_offsets+problem.benefit_directions@problem.mu0
        if problem.benefit_mode == "incremental":
            base = np.zeros_like(base)
            f_g = np.zeros_like(f_g)
        active_contracts = [j for j in range(n) if caps[j] > 0]
        compilation = dict(skipped_zero_device_rows=0, active_risk_factor_terms=0,
                           active_device_factor_terms=0, standard_soc_count=0,
                           absolute_risk_norm_count=0)
        for k in range(len(problem.risk_allocations)):
            mean = float(base[k])+gp.quicksum(float(g_h[k,j])*x[j] for j in active_contracts if g_h[k,j] != 0)
            expressions = []
            for a in range(factors):
                coefficients = [(j, float(g_h[k,j]*problem.response_factors[j,a]))
                                for j in active_contracts
                                if g_h[k,j]*problem.response_factors[j,a] != 0]
                if coefficients or f_g[k,a] != 0:
                    expressions.append(float(f_g[k,a])+gp.quicksum(value*x[j] for j, value in coefficients))
            compilation["active_risk_factor_terms"] += len(expressions)
            if expressions:
                norm = model.addVar(lb=0., name=f"factor_norm_{k}")
                if len(expressions) == 1:
                    # The Euclidean norm of one scalar is exactly its absolute
                    # value.  Use its two linear inequalities, no degenerate QC.
                    model.addConstr(norm >= expressions[0], name=f"scalar_norm_positive_{k}")
                    model.addConstr(norm >= -expressions[0], name=f"scalar_norm_negative_{k}")
                    compilation["absolute_risk_norm_count"] += 1
                else:
                    v = model.addVars(len(expressions), lb=-GRB.INFINITY, name=f"benefit_factors_{k}")
                    for a, expression in enumerate(expressions):
                        model.addConstr(v[a] == expression, name=f"benefit_factor_{k}_{a}")
                    model.addQConstr(gp.quicksum(v[a]*v[a] for a in range(len(expressions))) <= norm*norm,
                                     name=f"benefit_soc_{k}")
                    compilation["standard_soc_count"] += 1
                gamma = math.sqrt((1-problem.risk_allocations[k])/problem.risk_allocations[k]/3.)
                model.addConstr(mean-gamma*norm >= tau, name=f"reliable_benefit_{k}")
            else:
                model.addConstr(mean >= tau, name=f"reliable_benefit_{k}")

        hard_cost = gp.quicksum(float(problem.hard_cost_coefficients[j])*x[j]*x[j] for j in range(n))
        expected_cost = gp.quicksum(float(problem.expected_cost_coefficients[j])*x[j]*x[j] for j in range(n))
        model.addQConstr(hard_cost <= problem.budget, name="hard_episode_budget")
        for row in range(len(problem.device_rhs)):
            arow = problem.device_matrix[row]
            nonzero = [j for j in active_contracts if arow[j] != 0]
            if not nonzero and problem.device_rhs[row] >= 0:
                compilation["skipped_zero_device_rows"] += 1
                continue
            mean = gp.quicksum(float(arow[j])*x[j] for j in nonzero)
            for a in range(factors):
                coefficients = [(j, float(arow[j]*problem.response_factors[j,a]))
                                for j in nonzero if arow[j]*problem.response_factors[j,a] != 0]
                if not coefficients:
                    continue
                absolute = model.addVar(lb=0., name=f"device_abs_{row}_{a}")
                expression = gp.quicksum(value*x[j] for j, value in coefficients)
                model.addConstr(absolute >= expression, name=f"device_positive_{row}_{a}")
                model.addConstr(absolute >= -expression, name=f"device_negative_{row}_{a}")
                mean += absolute
                compilation["active_device_factor_terms"] += 1
            model.addConstr(mean <= float(problem.device_rhs[row]), name=f"device_bound_{row}")

        model.setObjective(tau, GRB.MAXIMIZE)
        model.Params.TimeLimit = max(1e-9, float(time_limit)-(time.monotonic()-start))
        model.optimize()
        primary = _stage_record(model, "maximize_reliable_benefit")
        result["primary"] = primary
        result["status"] = primary["status"]
        result["model_size"] = dict(variables=int(model.NumVars), linear_constraints=int(model.NumConstrs),
                                    quadratic_constraints=int(model.NumQConstrs), binary_variables=int(model.NumBinVars))
        result["compilation"] = compilation
        if not model.SolCount:
            result["elapsed_seconds"] = time.monotonic()-start
            result["secondary_skipped_reason"] = "no primary incumbent"
            return result

        chosen = np.array([x[j].X for j in range(n)])
        chosen_binary = None if y is None else [float(y[j].X) for j in range(n)]
        chosen_stage = "primary"
        primary_tau = float(tau.X)
        primary_eval = evaluate_plan(problem, chosen, tolerance=feasibility_tolerance*10)
        # Use the lower of the solver epigraph and recomputed benefit.  This
        # avoids propagating a loose numerical epigraph into stage two.
        primary_achieved = min(primary_tau, primary_eval["reliable_benefit"])
        floor = primary_achieved-(lex_absolute_tolerance+lex_relative_tolerance*abs(primary_achieved))
        result["primary_recomputed_lower_bound"] = primary_eval["reliable_benefit"]
        result["primary_tau"] = primary_tau
        result["secondary_benefit_floor"] = floor
        remaining = float(time_limit)-(time.monotonic()-start)
        if primary["solver_reported_optimal"] and primary_eval["numerical_feasible"] and remaining > .001:
            model.addConstr(tau >= floor, name="retain_primary_benefit")
            # Minimize sqrt(expected fee), an equivalent increasing transform.
            # A linear norm-radius objective is better conditioned at zero than
            # a quadratic objective with vanishing gradient, and removes tiny
            # gratuitous incentives without post-solve clipping.
            fee_radius = model.addVar(lb=0., name="expected_fee_radius")
            cost_indices = [j for j in range(n) if caps[j] > 0]
            if len(cost_indices) == 1:
                j = cost_indices[0]
                model.addConstr(math.sqrt(problem.expected_cost_coefficients[j])*x[j] <= fee_radius,
                                name="expected_fee_norm_1d")
            elif cost_indices:
                model.addQConstr(expected_cost <= fee_radius*fee_radius, name="expected_fee_soc")
            model.setObjective(fee_radius, GRB.MINIMIZE)
            model.Params.TimeLimit = remaining
            model.optimize()
            secondary = _stage_record(model, "minimize_sqrt_expected_fee")
            secondary["expected_fee_objective_value"] = (None if secondary["objective_value"] is None
                                                         else max(0., secondary["objective_value"])**2)
            secondary["expected_fee_lower_bound"] = (None if secondary["best_bound"] is None
                                                      else max(0., secondary["best_bound"])**2)
            result["secondary"] = secondary
            if model.SolCount:
                candidate = np.array([x[j].X for j in range(n)])
                evaluation = evaluate_plan(problem, candidate, tolerance=feasibility_tolerance*10)
                if (evaluation["numerical_feasible"]
                        and evaluation["reliable_benefit"] >= floor-feasibility_tolerance*10):
                    chosen = candidate
                    chosen_binary = None if y is None else [float(y[j].X) for j in range(n)]
                    chosen_stage = "secondary"
                else:
                    result["secondary_incumbent_rejected"] = "independent numerical constraint or benefit-floor check failed"
                    result["secondary_rejected_candidate"] = dict(
                        amplitudes=candidate.tolist(),
                        recomputed_benefit=evaluation["reliable_benefit"],
                        benefit_floor_violation=max(0., floor-evaluation["reliable_benefit"]),
                        constraint_violations=evaluation["violations"],
                        expected_cost=evaluation["expected_cost"])
            result["status"] = ("primary_optimal_secondary_rejected" if "secondary_incumbent_rejected" in result
                                else "optimal" if secondary["solver_reported_optimal"] and chosen_stage == "secondary"
                                else "primary_optimal_secondary_"+secondary["status"])
        else:
            result["secondary_skipped_reason"] = ("primary not solver-reported optimal" if not primary["solver_reported_optimal"]
                else "primary incumbent failed independent numerical validation" if not primary_eval["numerical_feasible"]
                else "shared time limit exhausted")

        evaluation = evaluate_plan(problem, chosen, tolerance=feasibility_tolerance*10)
        if not evaluation["numerical_feasible"]:
            result["status"] = "incumbent_failed_numerical_validation"
        result.update(has_incumbent=True, feasible_incumbent=evaluation["numerical_feasible"],
                      amplitudes=chosen.tolist(), prices=(chosen/problem.elasticities).tolist(),
                      achieved_benefit_lower_bound=evaluation["reliable_benefit"],
                      evaluation=evaluation, binary_activations=chosen_binary,
                      positive_contract_ids=[problem.contract_ids[j] for j in range(n)
                                             if chosen[j] > feasibility_tolerance*10],
                      returned_stage=chosen_stage, elapsed_seconds=time.monotonic()-start,
                      primary_solver_reported_optimal=primary["solver_reported_optimal"],
                      secondary_solver_reported_optimal=(result["secondary"] is not None
                                                         and result["secondary"]["solver_reported_optimal"]),
                      numerical_check_tolerance=feasibility_tolerance*10)
        return result
    finally:
        model.dispose()
        env.dispose()
