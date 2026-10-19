"""Fixed-competitor-witness exchange for continuous bilinear design.

For each pair, r(u,z)=c+B u+(A0+sum_t u_t At)z, with a common fixed
nuisance box. A feasible fixed z gives a quadratic UPPER response. A
piecewise-square chord MILP bounds max_u min_(pair,witness) ||r(u,z)||^2.
Every proposed continuous u is independently evaluated by all box QPs.
Their tangent lower bounds give a feasible lower bound for the true outer
maximum. The algorithm retains bounds and reports unfinished work honestly.

Scope: declared bilinear residual, finite linear operations domain, floating
arithmetic and solver tolerances. No nonlinear AC or self-profile subtraction.
"""
from __future__ import annotations
from time import perf_counter
import numpy as np
from .ev_bilinear_support_milp import BilinearPair, evaluate_qp, fixed_witness_upper
from .ev_profile_chord_milp import FrozenQuadratic, solve_profile_chord_milp, check_control, _constraints


def witness_quadratic(pair, z, *, name=None):
    """One globally feasible fixed explanation, hence an upper bound on q_pair."""
    z = np.asarray(z, float)
    if (z.shape != pair.lower.shape or not np.isfinite(z).all()
            or np.any(z < pair.lower) or np.any(z > pair.upper)):
        raise ValueError('fixed witness must lie in the declared nuisance box')
    return FrozenQuadratic(pair.name if name is None else name,
        pair.c+pair.A0@z, pair.B+np.einsum('tmn,n->mt', pair.At, z))


def exchange_bilinear(pairs, control_lower, control_upper, constraints=None, *,
                      initial_u=None, max_iterations=8, segments=4, max_segments=32,
                      solver_seconds=5., total_seconds=60., absolute_gap=1e-5,
                      relative_gap=1e-3, mip_rel_gap=1e-8, feasibility_tolerance=1e-7):
    began = perf_counter(); pairs = list(pairs)
    low, high = np.asarray(control_lower, float), np.asarray(control_upper, float)
    if (not pairs or low.ndim != 1 or len(low) < 1 or high.shape != low.shape
            or not np.isfinite(low).all() or not np.isfinite(high).all() or np.any(low > high)
            or any(not isinstance(p, BilinearPair) or p.control_dimension != len(low) for p in pairs)
            or len({p.name for p in pairs}) != len(pairs)):
        raise ValueError('unique compatible pairs and finite ordered control box required')
    for name, value in [('max_iterations', max_iterations), ('segments', segments), ('max_segments', max_segments)]:
        if isinstance(value, bool) or int(value) != value or value < 1:
            raise ValueError(name+' must be a positive integer')
    if max_segments < segments:
        raise ValueError('max_segments must be at least segments')
    if any(not np.isfinite(x) or x <= 0 for x in (solver_seconds, total_seconds)):
        raise ValueError('positive finite time budgets required')
    if any(not np.isfinite(x) or x < 0 for x in (absolute_gap, relative_gap, mip_rel_gap, feasibility_tolerance)):
        raise ValueError('finite nonnegative tolerances required')
    physical = _constraints(constraints, len(low))
    witnesses = [[(p.lower+p.upper)/2] for p in pairs]
    initial_upper = min(fixed_witness_upper(p, low, high, zs[0])**2 for p,zs in zip(pairs,witnesses))
    upper = float(initial_upper); lower = None; best = None
    iterations, evaluations, failures = [], [], []
    qp_calls = 0; current_segments = int(segments)

    def add_witness(k, z):
        z = np.asarray(z, float)
        if z.shape != pairs[k].lower.shape or np.any(z < pairs[k].lower) or np.any(z > pairs[k].upper):
            raise ValueError('QP witness outside the nuisance box')
        if any(np.array_equal(z, old) for old in witnesses[k]):
            return False
        witnesses[k].append(z.copy()); return True

    def evaluate(u, origin):
        nonlocal lower, best, qp_calls
        audit = check_control(u, low, high, physical, tolerance=feasibility_tolerance)
        if not audit['feasible']:
            raise ValueError('candidate fails declared operations: '+str(audit))
        u = np.asarray(u, float).copy()
        solutions = []
        for pair in pairs:
            qp_calls += 1
            solved = evaluate_qp(pair, u)
            solutions.append(solved)
        qlo = float(min(s['q_tangent_lower'] for s in solutions))
        qhi = float(min(s['q_primal_upper'] for s in solutions))
        added = [add_witness(k,s['z']) for k,s in enumerate(solutions)]
        entry = dict(origin=origin, u=u, q_lower=qlo, q_upper=qhi,
            qp_results=solutions, added_witness=added, physical_validation=audit,
            all_qp_converged=all(s['success'] for s in solutions))
        evaluations.append(entry)
        if lower is None or qlo > lower:
            lower, best = qlo, entry
        if not entry['all_qp_converged']:
            failures.append(dict(origin=origin, kind='inner_qp_not_converged',
                                 pairs=[p.name for p,s in zip(pairs,solutions) if not s['success']]))
        return entry

    def gap_target():
        return float(absolute_gap+relative_gap*max(1e-12, abs(upper), 0. if lower is None else abs(lower)))

    def converged():
        return lower is not None and upper >= lower-1e-9*max(1.,upper) and upper-lower <= gap_target()

    if initial_u is not None:
        evaluate(initial_u, 'initial_feasible_control')
    status = 'iteration_limit'
    if converged():
        status = 'gap_reached'
    else:
        for iteration in range(int(max_iterations)):
            remaining = total_seconds-(perf_counter()-began)
            if remaining <= 0:
                status='time_budget'; break
            quadratics = [witness_quadratic(p,z,name=p.name+'__w'+str(j))
                          for p,zs in zip(pairs,witnesses) for j,z in enumerate(zs)]
            master = solve_profile_chord_milp(quadratics,low,high,physical,
                segments=current_segments,time_limit=min(solver_seconds,remaining),mip_rel_gap=mip_rel_gap,
                feasible_incumbents=[] if best is None else [best['u']],
                feasibility_tolerance=feasibility_tolerance)
            before_lower, before_upper = lower, upper
            upper = min(upper,float(master['objective_upper']))
            record = dict(iteration=iteration, segments=current_segments,
                witnesses_before=[len(zs) for zs in witnesses], master=master,
                previous_lower=before_lower, previous_upper=before_upper, global_upper=upper,
                global_lower=lower, selected_u=master['u'], repeated_control=False,
                refinement_next=False)
            iterations.append(record)
            if master['u'] is None:
                status = 'operations_infeasible' if master['status']==2 and best is None else 'master_without_valid_incumbent'
                failures.append(dict(iteration=iteration, kind=status, message=master['message']))
                break
            repeated = any(np.allclose(master['u'],e['u'],rtol=0.,atol=1e-9*max(1.,float(np.max(abs(high-low))))) for e in evaluations)
            try:
                evaluation=evaluate(master['u'],'master_'+str(iteration))
            except (ValueError,RuntimeError,FloatingPointError,np.linalg.LinAlgError) as error:
                failures.append(dict(iteration=iteration,kind='candidate_qp_failure',message=str(error)))
                status='candidate_qp_failure';break
            record.update(global_lower=lower, evaluation_index=len(evaluations)-1,
                witnesses_after=[len(zs) for zs in witnesses], repeated_control=repeated,
                gap=upper-lower, target_gap=gap_target())
            if upper < lower-1e-9*max(1.,upper):
                failures.append(dict(iteration=iteration,kind='numerical_bound_inconsistency',lower=lower,upper=upper))
                status='numerical_bound_inconsistency';break
            if converged():
                status='gap_reached';break
            no_new_witness = not any(evaluation['added_witness'])
            if (repeated or no_new_witness) and current_segments < max_segments:
                current_segments = min(2*current_segments,int(max_segments))
                record['refinement_next']=True
            elif repeated and no_new_witness and current_segments >= max_segments:
                status='repeated_control_with_gap';break
    gap = None if lower is None else upper-lower
    return dict(u=None if best is None else best['u'], objective_lower=lower, objective_upper=upper,
        gap=None if gap is None else max(0.,gap), raw_gap=gap, target_gap=gap_target(),
        status=status, completed=(status=='gap_reached'), unfinished=(status!='gap_reached'),
        initial_fixed_witness_upper=float(initial_upper), iterations=iterations,evaluations=evaluations,
        witnesses=witnesses, witness_counts=[len(zs) for zs in witnesses], qp_calls=qp_calls,
        inner_qp_failures=sum(not e['all_qp_converged'] for e in evaluations), failures=failures,
        best_evaluation_index=None if best is None else next(i for i,e in enumerate(evaluations) if e is best),
        seconds=perf_counter()-began, final_segments=current_segments,
        full_curve_enumeration=False, self_profile_subtracted=False,
        original_global_optimum_certified=False,
        scope='numerical gap for declared bilinear squared distance over fixed boxes and linear operations; no nonlinear AC certificate')
