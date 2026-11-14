"""Exact elimination of distance auxiliaries where the feasible J set is simple.

The original numerical buffer, frozen model, guards, objective and domain stay
unchanged. Two-interval cases (including the c=0 isolated zero) use the original
solver. Returned plans are always checked by the original direct model.
"""
from time import perf_counter
import numpy as np

from ..absolute_identification_20261114.solver import (
    SymmetricRecovery, plan as original_plan, seed_items)


BUFFER = 1e-8


def pair_reduction(h, z, tie, buffer=BUFFER):
    """Feasible J>=0 for h+J-z*sqrt(2J)>=tie+buffer."""
    if not all(np.isfinite(v) for v in (h, z, tie, buffer)) or z <= 0 or buffer < 0:
        raise ValueError('Finite inputs, positive normal quantile and nonnegative buffer required')
    c = float(h-tie-buffer)
    if c >= z*z/2:
        return dict(kind='all_J', c=c, lower_J=None, upper_small_branch_J=None)
    disc = np.sqrt(z*z-2*c)
    upper = float(.5*(z+disc)**2)
    if c < 0:
        return dict(kind='lower_bound', c=c, lower_J=upper, upper_small_branch_J=None)
    # The stable expression also gives zero at c=0; the zero point must remain.
    lower = float(2*c*c/(z+disc)**2)
    return dict(kind='two_intervals', c=c, lower_J=upper, upper_small_branch_J=lower)


def reductions(evidence):
    return [dict(pair=list(pair), **pair_reduction(float(h), evidence.z, evidence.tie))
        for pair, h in zip(evidence.pairs, evidence.h, strict=True)]


def plan(case, score, validation, forecast, base_kw, *, mode='symmetric_guard',
         seconds=180., env, seed_points=None, target_mass=.99):
    """Same problem and initial-point retention as original_plan; no ADMM."""
    began = perf_counter()
    evidence = SymmetricRecovery(case, score, validation, forecast, base_kw, mode, target_mass)
    reduction = reductions(evidence)
    if any(row['kind'] == 'two_intervals' for row in reduction):
        receipt, point = original_plan(case, score, validation, forecast, base_kw,
            mode=mode, seconds=max(.001, seconds-(perf_counter()-began)), env=env,
            seed_points=seed_points, target_mass=target_mass)
        receipt.update(reformulation='original_solver_fallback_for_two_intervals',
            pair_reduction=reduction, seconds=perf_counter()-began,
            time_scope='Includes reduction analysis and original solver; caller adds external seed generation')
        return receipt, point
    baseline = evidence.verify_expansion(np.zeros(case.shape))
    common = dict(mode=mode, target_mass=target_mass, targets=evidence.targets,
        target_topology_ids=[score['models'][i]['topology_id'] for i in evidence.targets],
        historical_selected_mass=float(evidence.weights[evidence.targets].sum()),
        candidates=evidence.K, pair_count=len(evidence.pairs), guard_count=len(evidence.guards),
        individual_comparison_risk=evidence.alpha, total_comparison_risk=.1, baseline=baseline,
        constrained_bank='frozen representative models; native-bank outcomes reported separately',
        reformulation='exact_auxiliary_elimination_no_domain_restriction', pair_reduction=reduction,
        numerical_gate_buffer=BUFFER, dropped_absolute_gates=sum(r['kind'] == 'all_J' for r in reduction),
        remaining_absolute_gates=sum(r['kind'] == 'lower_bound' for r in reduction),
        confidence_scope='Same conditional frozen-candidate iid Gaussian model as original solver',
        time_scope='Includes model, seed auditing and native solve; caller adds external seed generation')
    if baseline['feasible']:
        return dict(common, feasible=True, status='zero_sufficient', source='zero', cost_yuan=0.,
            lower_bound_yuan=0., relative_gap=0., global_optimality_verified=True,
            seconds=perf_counter()-began), np.zeros(case.shape)
    box_checks = []
    for k, row in enumerate(reduction):
        if row['kind'] != 'lower_bound':
            continue
        required, upper = row['lower_J'], float(evidence.upper[k])
        margin = 1e-7*max(1., required, upper)
        box_checks.append(dict(pair=row['pair'], required_J=required,
            relaxed_power_box_upper_J=upper, comparison_margin=margin,
            impossible=bool(required > upper+margin)))
    common['resource_box_checks'] = box_checks
    if any(row['impossible'] for row in box_checks):
        return dict(common, feasible=False, status='infeasible_by_resource_box_bound',
            native_status=None, infeasibility_proved=True,
            infeasibility_source='analytic_relaxed_resource_power_box',
            infeasibility_scope='Declared buffered QCQP; a required single-high-branch distance exceeds a valid resource-box upper bound, even without energy coupling or mean guards',
            optimizer_invoked=False, cost_yuan=None, lower_bound_yuan=None,
            global_optimality_verified=False, seconds=perf_counter()-began), None
    point, warm, chosen, merit = None, None, None, -np.inf
    initial_records = []; tick = perf_counter()
    for name, u in seed_items(seed_points):
        try:
            u = np.asarray(u, float); domain = case.audit(u); info = evidence.verify_expansion(u)
            valid = domain['passed'] and info['feasible']; fee = case.fee(u)
            value = min(v for v in (info['minimum_pair_margin'], info['minimum_guard_gain']) if v is not None)
            initial_records.append(dict(name=name, resource_feasible=domain['passed'],
                qcqp_feasible=info['feasible'], cost_yuan=fee, minimum_margin=value))
            if valid and (point is None or fee < case.fee(point)):
                point, chosen = u.copy(), name
            if domain['passed'] and value > merit:
                warm, merit = u.copy(), value
        except Exception as exc:
            initial_records.append(dict(name=name, error=repr(exc), qcqp_feasible=False))
    if point is not None: warm = point.copy()
    seed = None if point is None else point.copy()
    initial = dict(source='caller_common_seed_pool', feasible=seed is not None, name=chosen,
        cost_yuan=None if seed is None else case.fee(seed), audit_seconds=perf_counter()-tick,
        candidates=initial_records)
    import gurobipy as gp
    with gp.Model('symmetric_recovery_exact_distance_reduction', env=env) as model:
        for key, value in dict(OutputFlag=0, Threads=1, Seed=0, NonConvex=2,
                FeasibilityTol=1e-9, NumericFocus=1, MIPGap=1e-4).items():
            model.setParam(key, value)
        u, fee = case.attach(model)
        for k, reduced in enumerate(reduction):
            if reduced['kind'] == 'lower_bound':
                J = sum(u[h]@evidence.q[k]@u[h] for h in range(case.H))+2*(evidence.a[k]*u).sum()+evidence.c[k]
                model.addConstr(J >= reduced['lower_J'])
        for left, right, K, Q, a in evidence.guards:
            gain = sum(u[h]@Q@u[h] for h in range(case.H))+2*(a*u).sum()
            model.addConstr(gain >= 0.)
        model.setObjective(fee)
        if warm is not None: u.Start = warm
        model.Params.TimeLimit = max(.001, seconds-(perf_counter()-began))
        model.optimize()
        try:
            bound = float(model.ObjBound)
            if not np.isfinite(bound): bound = None
        except (AttributeError, gp.GurobiError):
            bound = None
        source = 'verified_initial_retained' if seed is not None else None
        if model.SolCount:
            candidate = np.asarray(u.X)
            if case.audit(candidate)['passed'] and evidence.verify_expansion(candidate)['feasible']:
                if point is None or case.fee(candidate) < case.fee(point)-1e-9:
                    point = candidate
                    source = 'native_found' if seed is None else 'native_improved_initial'
        if point is None:
            return dict(common, feasible=False, status='native_infeasible' if model.Status == 3 else 'unresolved',
                native_status=int(model.Status), initial=initial, infeasibility_proved=model.Status == 3,
                cost_yuan=None, lower_bound_yuan=bound, global_optimality_verified=False,
                seconds=perf_counter()-began), None
        cost = case.fee(point)
        conflict = bound is not None and bound > cost+1e-7*max(1., cost)
        lower = 0. if bound is None or conflict else max(0., min(cost, bound))
        gap = (cost-lower)/max(cost, 1e-12)
        return dict(common, feasible=True, status='feasible', source=source, native_status=int(model.Status),
            initial=initial, cost_yuan=cost, lower_bound_yuan=lower, relative_gap=gap,
            lower_bound_withdrawn=conflict, native_infeasibility_withdrawn=model.Status == 3,
            global_optimality_verified=gap <= 1e-4,
            improvement_from_verified_initial_yuan=None if seed is None else case.fee(seed)-cost,
            final=evidence.verify_expansion(point), resource_audit=case.audit(point),
            seconds=perf_counter()-began), point
