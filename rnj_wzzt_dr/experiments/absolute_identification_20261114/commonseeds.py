"""Shared resource-feasible starts; no truth, response realization or AC inputs."""
from time import perf_counter

import numpy as np
from scipy.optimize import linprog


def build_portfolio(case, score, forecast, base_kw, *, env=None, max_pairs=3,
                    gamma=2., seconds=2., fractions=(.25, .5, .75, 1.)):
    """Return (name->u, receipt); caller charges full generation to EACH budget.

    A cached pool may be reused across paired methods. Fractions are fixed before
    outcomes; every returned action satisfies only the original resource domain.
    Each consuming optimizer separately checks its own evidence constraints.
    """
    began = perf_counter()
    if seconds <= 0 or base_kw <= 0 or not 1 <= max_pairs:
        raise ValueError('Positive generation budget, power base and pair count required')
    if not all(0 < f <= 1 for f in fractions):
        raise ValueError('Ray fractions must belong to (0,1]')
    forecast = np.asarray(forecast, float)
    H = np.asarray([p['H'] for p in score['parameters']], float)
    target = score['map_index']
    history = np.array([p['history_sse'] for p in score['parameters']], float)
    order = sorted((j for j in range(len(H)) if j != target), key=lambda j:(history[j],j))
    sigma2 = float(score['sigma2'])
    if forecast.shape != (case.H*case.repeat, 2*case.N) or sigma2 <= 0:
        raise ValueError('Frozen forecast/score shape or scale invalid')
    points, records, seen = {}, [], {}
    zero = np.zeros(case.shape)
    if not case.audit(zero)['passed']:
        raise ValueError('Ray portfolio requires a feasible zero resource action')

    def add(name, u):
        if u is None:
            return
        u = np.asarray(u, float)
        audit = case.audit(u)
        if not audit['passed']:
            records.append(dict(name=name, accepted=False, reason='original_resource_domain_failed'))
            return
        key = np.round(u, 12).tobytes()
        if key in seen:
            records.append(dict(name=name, accepted=False, duplicate_of=seen[key]))
            return
        points[name] = u.copy()
        seen[key] = name
        records.append(dict(name=name, accepted=True, cost_yuan=case.fee(u)))

    def rays(name, u):
        if u is not None:
            for f in fractions:
                add(name+f'_fraction{f:g}', f*u)

    lps = []
    def linear(direction, name, extra_A=None, extra_b=None, least_fee=False):
        remaining = seconds-(perf_counter()-began)
        if remaining <= .001:
            lps.append(dict(name=name, status='generation_budget_exhausted', seconds=0.))
            return None
        tick = perf_counter()
        try:
            n = case.dimension
            bounds = list(zip(case.lower.ravel(), case.upper.ravel()))
            if least_fee:
                eye = np.eye(n)
                A = np.block([[case.A_ub, np.zeros_like(case.A_ub)], [eye, -eye], [-eye, -eye]])
                b = np.r_[case.b_ub, np.zeros(2*n)]
                if extra_A is not None:
                    A = np.r_[A, np.c_[extra_A, np.zeros_like(extra_A)]]
                    b = np.r_[b, extra_b]
                eq = np.c_[case.A_eq, np.zeros_like(case.A_eq)]
                cost = case.dt*np.r_[(case.fee_plus-case.fee_minus).ravel()/2,
                                    (case.fee_plus+case.fee_minus).ravel()/2]
                result = linprog(cost, A_ub=A, b_ub=b, A_eq=eq, b_eq=case.b_eq,
                    bounds=bounds+[(0.,None)]*n, method='highs', options={'time_limit':remaining})
            else:
                result = linprog(np.asarray(direction).ravel(), A_ub=case.A_ub, b_ub=case.b_ub,
                    A_eq=case.A_eq, b_eq=case.b_eq, bounds=bounds, method='highs', options={'time_limit':remaining})
            point = None if result.x is None else result.x[:n].reshape(case.shape)
            valid = point is not None and case.audit(point)['passed']
            lps.append(dict(name=name, status=int(result.status), original_domain_feasible=bool(valid),
                optimal=bool(result.success), seconds=perf_counter()-tick, message=result.message))
            return point if valid else None
        except Exception as exc:
            lps.append(dict(name=name, status='failed', error=repr(exc), seconds=perf_counter()-tick))
            return None

    add('zero', zero)
    rotation = np.where((np.arange(case.H)[:,None]//2+np.arange(case.R)[None,:])%2 == 0, 1., -1.)
    rotate = linear(rotation, 'fixed_rotation_LP')
    rays('fixed_rotation', rotate)
    if rotate is not None and case.audit(-rotate)['passed']:
        rays('negative_fixed_rotation', -rotate)
    gradients = []
    for j in order:
        D = H[target]-H[j]
        difference = forecast@D.T
        effect = (case.response_effect(D, base_kw) if hasattr(case, 'response_effect')
                  else (D[:,case.node]+D[:,case.N+case.node]*case.qratio)/base_kw)
        grad = difference.reshape(case.H,case.repeat,case.N).sum(1)@effect/sigma2
        gradients.append(grad)
        if j in order[:max_pairs] and np.max(abs(grad)) > 1e-14:
            for sign in (-1.,1.):
                name = f'pair{target}_{j}_direction{sign:g}'
                rays(name, linear(sign*grad, name))
    if gradients:
        # Same minimum-fee supporting LP recipe as the old gamma2 warm start;
        # this candidate is shared with every consuming model.
        extra_A = -np.array(gradients).reshape(len(gradients),case.dimension)
        extra_b = np.full(len(gradients), -gamma*(1+1e-7)-1e-8)
        supporting = linear(None, 'old_supporting_LP', extra_A, extra_b, least_fee=True)
        rays('old_supporting_LP', supporting)
    receipt = dict(seconds=perf_counter()-began, budget_seconds=seconds,
        budget_exceeded=perf_counter()-began > seconds+.05, points=len(points),
        selected_history_pairs=[(target,j) for j in order[:max_pairs]],
        fractions=list(fractions), target=target, gamma_supporting_LP=gamma,
        source='shared_resource_domain_portfolio', records=records, LPs=lps,
        evidence_constraints_checked=False, physical_or_truth_inputs=False,
        charge_rule='Charge this full actual seconds to each paired optimization budget, even when points are cached')
    return points, receipt


def solve_old_with_seeds(evidence, gamma, *, seed_points, seconds, env):
    """Existing gamma+guard QCQP with caller-supplied starts, same constraints."""
    import gurobipy as gp
    began = perf_counter()
    case = evidence.case
    required = evidence.required(gamma)
    point = None
    warm = None
    merit = -np.inf
    chosen = None
    records = []
    tick = perf_counter()
    for name, u in (seed_points or {}).items():
        u = np.asarray(u, float)
        domain = case.audit(u)['passed']
        actual = evidence.direct(u)
        value = float(np.min(actual-required))
        feasible = domain and value >= -1e-7
        records.append(dict(name=name, resource_feasible=domain, qcqp_feasible=feasible,
            minimum_margin=value, cost_yuan=case.fee(u)))
        if feasible and (point is None or case.fee(u)<case.fee(point)):
            point, chosen = u.copy(), name
        if domain and value > merit:
            warm, merit = u.copy(), value
    seed = None if point is None else point.copy()
    if point is not None:
        warm = point.copy()
    initial = dict(source='caller_common_seed_pool', feasible=seed is not None, name=chosen,
        cost_yuan=None if seed is None else case.fee(seed), audit_seconds=perf_counter()-tick, candidates=records)
    with gp.Model('old_gain_QCQP_shared_initialization', env=env) as m:
        for k,v in dict(OutputFlag=0, Threads=1, Seed=0, NonConvex=2, FeasibilityTol=1e-9,
                        NumericFocus=1, MIPGap=1e-5).items():
            m.setParam(k,v)
        u, fee = case.attach(m)
        for Q,a,rhs in zip(evidence.Q,evidence.a,required,strict=True):
            expression = sum(u[h]@Q@u[h] for h in range(case.H))+2*(a*u).sum()
            m.addConstr(expression >= rhs+1e-8)
        if warm is not None:
            u.Start = warm
        m.setObjective(fee)
        m.Params.TimeLimit = max(.001, seconds-(perf_counter()-began))
        m.optimize()
        try:
            bound = float(m.ObjBound)
            if not np.isfinite(bound):
                bound = None
        except (AttributeError,gp.GurobiError):
            bound = None
        source = 'verified_initial_retained' if seed is not None else None
        if m.SolCount:
            candidate = np.asarray(u.X)
            if case.audit(candidate)['passed'] and np.min(evidence.direct(candidate)-required) >= -1e-7:
                if point is None or case.fee(candidate) < case.fee(point)-1e-9:
                    point = candidate
                    source = 'native_found' if seed is None else 'native_improved_initial'
        if point is None:
            return dict(feasible=False, status='native_infeasible' if m.Status==3 else 'unresolved',
                native_status=int(m.Status), initial=initial, lower_bound_yuan=bound, cost_yuan=None,
                infeasibility_proved=m.Status==3, seconds=perf_counter()-began), None
        if not np.allclose(evidence.direct(point),evidence.values(point),rtol=1e-7,atol=1e-8):
            raise ArithmeticError('Original old-gain constraints and quadratic expansion disagree')
        cost = case.fee(point)
        conflict = bound is not None and bound>cost+1e-7*max(1.,cost)
        lower = 0. if bound is None or conflict else max(0.,min(cost,bound))
        gap = (cost-lower)/max(cost,1e-12)
        return dict(feasible=True, status='feasible', source=source, native_status=int(m.Status), initial=initial,
            cost_yuan=cost, lower_bound_yuan=lower, relative_gap=gap,
            global_optimality_verified=gap<=1e-5, lower_bound_withdrawn=conflict,
            native_infeasibility_withdrawn=m.Status==3,
            improvement_from_verified_initial_yuan=None if seed is None else case.fee(seed)-cost,
            minimum_margin=float(np.min(evidence.direct(point)-required)),
            resource_audit=case.audit(point), seconds=perf_counter()-began,
            time_scope='includes seed auditing and solve; external shared-seed generation must be charged by caller'), point
