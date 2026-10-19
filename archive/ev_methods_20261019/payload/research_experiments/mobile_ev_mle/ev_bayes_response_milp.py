"""Parent-tight BC upper surrogate for the frozen-covariance Gaussian model.

The old square construction and original EV operation domain are reused.  For
each pair, the lower distance is the exact MILP graph of the maximum of its
covariance floor, parent tangent, and shifted square chord.  Exponential chord
nodes include the parent distance.  Thus the mathematical upper surrogate
touches the frozen BC objective at the parent; tiny recorded arithmetic guards
only affect that equality at floating-point scale.  Complete nonlinear AC and
the original risk still decide whether a proposed action is accepted.
"""
from __future__ import annotations

from time import perf_counter

import numpy as np

from . import ev_bayes_small_milp as base
from .ev_operations_milp import LinearMILP
from .ev_original_mip_backend import solve_linear_mip


frozen_bc_value = base.frozen_bc_value


def _exp_chords(grid):
    # expm1 retains a valid slope when the inserted parent is extremely close
    # to an existing node.  Direct subtraction of two near-equal exponentials
    # can otherwise spoil the convex epigraph far outside that tiny segment.
    for lo, hi in zip(grid[:-1], grid[1:]):
        left = float(np.exp(-lo))
        slope = float(left*np.expm1(-(hi-lo))/(hi-lo))
        yield lo, hi, slope, float(left-slope*lo)


def _exp_error(grid):
    error = 0.
    for lo, hi, slope, intercept in _exp_chords(grid):
        if slope == 0.:
            continue
        point = float(np.clip(-np.log(-slope), lo, hi))
        error = max(error, intercept+slope*point-float(np.exp(-point)))
    return max(0., float(error))


def _copy_square_model(old):
    """Keep operation/square/shift rows; remove the old clip/exp graphs."""
    keep = set(range(old['operation_dimension']))
    for mode in old['modes']:
        keep.update(np.asarray(mode['lambda_indices']).ravel().tolist())
        keep.update(np.asarray(mode['segment_indices']).ravel().tolist())
    keep.update(a['shifted_index'] for a in old['pair_audits'])
    keep = sorted(keep)
    original = old['model']
    model = LinearMILP()
    indices = model.add_variables('operation_and_square', len(keep),
        lower=np.asarray(original.lower)[keep], upper=np.asarray(original.upper)[keep],
        integer=np.asarray(original.integer)[keep])
    mapping = dict(zip(keep, map(int, indices)))
    for row, lo, hi in zip(original.rows, original.row_lower, original.row_upper):
        if all(i in mapping for i in row):
            model.add_row({mapping[i]:v for i, v in row.items()}, lo, hi)
    modes = [{**mode, 'lambda_indices':np.asarray([mapping[int(i)] for i in mode['lambda_indices']]),
              'segment_indices':np.asarray([mapping[int(i)] for i in mode['segment_indices']])}
             for mode in old['modes']]
    return model, mapping, modes


def _linear_bounds(coefficient, lower, upper, constant=0.):
    coefficient = np.asarray(coefficient, float)
    lo = float(constant+np.where(coefficient >= 0., coefficient*lower, coefficient*upper).sum())
    hi = float(constant+np.where(coefficient >= 0., coefficient*upper, coefficient*lower).sum())
    return lo, hi


def _exact_max_graph(model, branches, lower, upper, name):
    """Each branch is (constant, sparse linear coefficients, certified lower)."""
    value = int(model.add_variables(name, lower=lower, upper=upper))
    for constant, coefficients, _ in branches:
        row = {value:1.}
        row.update({i:-v for i, v in coefficients.items()})
        model.add_row(row, lower=constant)
    if len(branches) == 1:
        constant, coefficients, _ = branches[0]
        row = {value:1.}
        row.update({i:-v for i, v in coefficients.items()})
        model.add_row(row, upper=constant)
        return value, 0, []
    selectors = model.add_variables(name+'_active', len(branches), upper=1., integer=True)
    model.add_row({int(i):1. for i in selectors}, 1., 1.)
    big_m = []
    for z, (constant, coefficients, branch_lower) in zip(selectors, branches):
        M = max(0., float(upper-branch_lower))
        row = {value:1., int(z):M}
        row.update({i:-v for i, v in coefficients.items()})
        model.add_row(row, upper=constant+M)
        big_m.append(M)
    return value, len(selectors), big_m


def build_bc_master(domain, bank, jacobian, anchor, *, segments=4, exp_segments=8, power_radius=None):
    """Build a MIN MILP; all certificates concern the frozen Gaussian proxy."""
    began = perf_counter()
    old = base.build_bc_master(domain, bank, jacobian, anchor, segments=segments,
                               exp_segments=exp_segments, power_radius=power_radius)
    model, mapping, modes = _copy_square_model(old)
    op, p = old['operation'], old['data']['anchor'].ravel()
    objective, audits, errors = {}, [], []
    n_max_binary = 0
    for index, (pair, original) in enumerate(zip(old['pairs'], old['pair_audits'])):
        a = dict(original)
        a['B_coefficient'] = {mapping[i]:v for i, v in original['B_coefficient'].items()}
        shifted = mapping[original['shifted_index']]
        a['shifted_index'] = shifted
        residual = pair.c+pair.B@p
        Bp = float(.5*(residual@residual)+pair.covariance_term)
        gradient = pair.B.T@residual
        tangent_constant = float(Bp-gradient@p)
        ubar = np.maximum(abs(op['control_lower']), abs(op['control_upper']))
        tangent_guard = float(128*np.finfo(float).eps*(1.+abs(Bp)+abs(tangent_constant)+abs(gradient)@ubar))
        tangent_constant -= tangent_guard
        tlo, thi = _linear_bounds(gradient, op['control_lower'], op['control_upper'], tangent_constant)
        tangent_coeff = {j:float(v) for j, v in enumerate(np.asarray(gradient@op['U']).ravel()) if v}
        floor = float(pair.covariance_term)
        old_lower, old_upper = original['shifted_bounds']
        arithmetic = 128*np.finfo(float).eps*(1.+abs(old_upper)+abs(Bp))
        upper = float(max(floor, old_upper, Bp)+arithmetic)
        branches = [(floor, {}, floor)]
        if thi > floor:
            branches.append((tangent_constant, tangent_coeff, tlo))
        if old_upper > floor:
            branches.append((0., {shifted:1.}, float(old_lower)))
        lower_index, n_binary, big_m = _exact_max_graph(model, branches, floor, upper, 'pair_'+str(index)+'_lower')
        n_max_binary += n_binary
        # Existing exp nodes already adapt to its curvature.  Shift them to
        # the covariance floor and additionally include the mathematical
        # parent distance.  Its guarded lower value differs only at roundoff.
        grid = base._exp_grid(max(0., upper-floor), exp_segments)+floor
        grid = np.unique(np.r_[grid, np.clip(Bp, floor, upper)])
        values = np.exp(-grid)
        epi = int(model.add_variables('pair_'+str(index)+'_exp', lower=float(values[-1]), upper=float(values[0])))
        if len(grid) == 1:
            model.add_row({epi:1.}, float(values[0]), float(values[0]))
        else:
            for _, _, slope, intercept in _exp_chords(grid):
                model.add_row({epi:1., lower_index:-slope}, lower=intercept)
        objective[epi] = pair.weight
        exp_error = _exp_error(grid)
        overlap_error = float(np.exp(-floor)*(-np.expm1(-original['epsilon_B'])))
        uniform_error = min(float(np.exp(-floor)), overlap_error+exp_error)
        errors.append(pair.weight*uniform_error)
        a.update(clip_index=lower_index, exp_index=epi, exp_grid=grid,
                 exp_chord_error=exp_error, weighted_uniform_error=pair.weight*uniform_error,
                 parent_B=Bp, parent_gradient=gradient, tangent_constant=tangent_constant,
                 tangent_roundoff_guard=tangent_guard, tangent_bounds=[tlo, thi],
                 lower_distance_bounds=[floor, upper], max_branch_count=len(branches),
                 max_big_m=big_m, lower_distance_rule='max_covariance_floor_parent_tangent_shifted_chord')
        audits.append(a)
    arrays = model.arrays(objective, cmax=False)
    result = {**old, **arrays, 'model':model, 'modes':modes, 'pair_audits':audits,
              'n_clip_binaries':n_max_binary, 'n_max_binaries':n_max_binary,
              'proxy_uniform_error':float(sum(errors)), 'anchor_envelope':True,
              'build_seconds':perf_counter()-began,
              'scope':'parent-tight upper surrogate of affine means and frozen covariance Gaussian BC sum'}
    at_parent = evaluate_proxy(result, old['data']['anchor'])
    result['anchor_proxy_value'] = at_parent['proxy']
    result['anchor_proxy_error'] = at_parent['actual_proxy_error']
    return result


def evaluate_proxy(master, power_kw):
    """Arithmetic audit; shared products require a feasible operation vector."""
    u = np.asarray(power_kw, float).ravel()
    op = master['operation']
    if u.shape != (master['data']['anchor'].size,) or not np.isfinite(u).all():
        raise ValueError('matching finite power required')
    if np.any(u < op['control_lower']-1e-7) or np.any(u > op['control_upper']+1e-7):
        raise ValueError('power outside proxy control bounds')
    rows = []
    for pair, a in zip(master['pairs'], master['pair_audits']):
        if a['form'] == 'shared_control_square':
            quadratic = sum(float(d*np.interp(f['coefficient']@u, f['grid'], np.asarray(f['grid'])**2))
                            for d, f in zip(a['diagonal'], a['factors']))
        else:
            quadratic = sum(float(.5*np.interp(f['coefficient']@u, f['grid'], np.asarray(f['grid'])**2))
                            for f in a['factors'])
        chord = float(quadratic+a['linear']@u+a['constant']+a['rounding_guard']+a['numeric_guard'])
        tangent = float(a['tangent_constant']+a['parent_gradient']@u)
        lower = max(float(pair.covariance_term), tangent, chord-a['epsilon_B'])
        residual = pair.c+pair.B@u
        exact_B = float(.5*(residual@residual)+pair.covariance_term)
        approx = float(np.interp(lower, a['exp_grid'], np.exp(-a['exp_grid'])))
        exact = float(np.exp(-exact_B))
        rows.append(dict(scenario=pair.scenario, i=pair.i, j=pair.j, B=exact_B,
                         B_chord=chord, B_tangent=tangent, B_lower=lower,
                         epsilon_B=a['epsilon_B'], exact_BC=exact, proxy_BC=approx,
                         weight=pair.weight, weighted_proxy=pair.weight*approx,
                         weighted_exact=pair.weight*exact))
    value = float(sum(r['weighted_proxy'] for r in rows))
    exact = float(sum(r['weighted_exact'] for r in rows))
    return dict(proxy=value, C=exact, actual_proxy_error=value-exact,
                uniform_error=master['proxy_uniform_error'], pairs=rows)


def _operation_start(master,domain,operation_start):
    if operation_start is None:
        return None,None
    x=np.asarray(operation_start,float)
    if x.shape!=(master['operation_dimension'],) or not np.isfinite(x).all():
        raise ValueError('operation_start must be a complete finite original operation vector')
    op_audit=base.operations._audit(x,master['operation'],1e-6)
    physical=domain.check(x,tolerance=1e-6) if hasattr(domain,'check') else op_audit
    if not op_audit['feasible'] or not physical['feasible']:
        raise ValueError('operation_start violates the original operation domain or trust region')
    start=np.full(len(master['c']),np.nan)
    start[:master['operation_dimension']]=x
    return start,dict(operation=op_audit,physical=physical,
                      policy='complete parent operation; all proxy auxiliaries unspecified')


def _solve_master(domain,bank,jacobian,anchor,master,evaluator,seconds,backend,operation_start,began):
    """Identical solve/audit path for legacy and parent-tight proxy matrices."""
    start,start_audit=_operation_start(master,domain,operation_start)
    solved = solve_linear_mip(master['c'], master['integrality'], master['bounds'], master['constraints'],
                              time_limit=float(seconds), mip_rel_gap=1e-5, backend=backend,start=start)
    status = int(solved.status)
    incumbent = base._finite(getattr(solved, 'fun', None))
    best = base._finite(getattr(solved, 'mip_dual_bound', None)) if status in (0, 1) else None
    result = dict(success=False, status=status, stop_reason=str(solved.message),
        backend=getattr(solved, 'backend_used', backend), solver_incumbent=incumbent,
        solver_best_bound=best, solver_relative_gap=base._finite(getattr(solved, 'mip_gap', None)),
        operation_x=None, power_kw=None, physical_audit=None, operation_audit=None, master_audit=None,
        frozen_C=None, proxy_value=None, proxy_uniform_error=master['proxy_uniform_error'],
        frozen_C_anchor=frozen_bc_value(bank, jacobian, anchor, anchor)['C'],
        anchor_proxy_value=master['anchor_proxy_value'], anchor_proxy_error=master['anchor_proxy_error'],
        anchor_envelope=bool(master.get('anchor_envelope',False)), shared_control_square=master['shared_control_square'],
        n_square_binaries=master['n_square_binaries'], n_clip_binaries=master['n_clip_binaries'],
        n_max_binaries=master.get('n_max_binaries',master['n_clip_binaries']), n_exp_binaries=0,
        model_size=dict(variables=len(master['c']), rows=master['constraints'].A.shape[0],
                        operation_variables=master['operation_dimension'],
                        integer_variables=int(np.count_nonzero(master['integrality'])), pair_terms=len(master['pairs'])),
        original_operation_constraints_preserved=True, build_seconds=master['build_seconds'],
        solver_seconds=getattr(solved, 'backend_seconds', None), full_AC_optimality_claim=False,
        scope=master['scope'], failure=None,operation_start_audit=start_audit,
        start_supplied=bool(start is not None),start_provided_count=0 if start is None else int(np.isfinite(start).sum()),
        start_used=bool(getattr(solved,'start_used',False)),
        start_policy=getattr(solved,'start_policy','backend did not confirm MIP-start submission'),
        start_used_meaning=getattr(solved,'start_used_meaning','submission does not establish start acceptance or improvement'))
    if getattr(solved, 'x', None) is None:
        result.update(failure='no_solver_incumbent', seconds=perf_counter()-began)
        return result
    full = np.asarray(solved.x, float)
    all_op = dict(lower=master['bounds'].lb, upper=master['bounds'].ub,
                  integrality=master['integrality'], constraints=[master['constraints']])
    audit = base.operations._audit(full, all_op, 1e-6)
    x = full[:master['operation_dimension']]
    op_audit = base.operations._audit(x, master['operation'], 1e-6)
    physical = domain.check(x, tolerance=1e-6) if hasattr(domain, 'check') else op_audit
    result.update(master_audit=audit, operation_audit=op_audit, physical_audit=physical)
    if not all(a['feasible'] for a in (audit, op_audit, physical)):
        result.update(failure='incumbent_constraint_audit_failed', seconds=perf_counter()-began)
        return result
    u = np.asarray(master['operation']['U']@x).reshape(np.asarray(anchor).shape)
    value = evaluator(master, u)
    tolerance = 1e-6*(1.+abs(value['proxy']))
    failure = None
    if value['actual_proxy_error'] < -tolerance or value['actual_proxy_error'] > master['proxy_uniform_error']+tolerance:
        failure = 'proxy_uniform_error_audit_failed'
    elif incumbent is not None and incumbent < value['proxy']-tolerance:
        failure = 'solver_epigraph_objective_audit_failed'
    elif best is not None and best > value['proxy']+tolerance:
        failure = 'solver_best_bound_audit_failed'
    if failure is not None:
        result.update(failure=failure, seconds=perf_counter()-began)
        return result
    result.update(success=True, operation_x=x.copy(), power_kw=u.copy(), frozen_C=value['C'],
        proxy_value=value['proxy'], proxy_actual_error=value['actual_proxy_error'], pair_audits=value['pairs'],
        proxy_certificate=[max(0., value['proxy']-master['proxy_uniform_error']), value['proxy']],
        frozen_C_optimum_lower=None if best is None else max(0., best-master['proxy_uniform_error']),
        attained_frozen_risk_upper=value['proxy'],
        proxy_absolute_gap=None if best is None else max(0., value['proxy']-best),
        frozen_C_nonincrease=value['C'] <= result['frozen_C_anchor']+tolerance,
        error_audits=[{k:a[k] for k in ('scenario','i','j','weight','logweight','epsilon_B','chord_error',
            'rounding_guard','numeric_guard','exp_chord_error','weighted_uniform_error','form',
            'parent_B','tangent_roundoff_guard','max_branch_count','max_big_m') if k in a} for a in master['pair_audits']],
        seconds=perf_counter()-began)
    return result


def solve_bc_milp(domain,bank,jacobian,anchor,segments=4,exp_segments=8,
                  seconds=5,power_radius=None,*,backend='gurobi',operation_start=None):
    """Parent-tight proposal; a supplied feasible operation is a partial start."""
    began=perf_counter()
    if not np.isfinite(seconds) or seconds<=0.:
        raise ValueError('positive finite solver seconds required')
    master=build_bc_master(domain,bank,jacobian,anchor,segments=segments,
                           exp_segments=exp_segments,power_radius=power_radius)
    return _solve_master(domain,bank,jacobian,anchor,master,evaluate_proxy,
                         seconds,backend,operation_start,began)


def solve_legacy_bc_milp(domain,bank,jacobian,anchor,segments=4,exp_segments=8,
                         seconds=5,power_radius=None,*,backend='gurobi',operation_start=None):
    """Unchanged old master/evaluator with the same optional operation start.

    No legacy scientific file is edited or monkeypatched.  Objectives, rows,
    approximation errors and time limits stay those of the old construction.
    """
    began=perf_counter()
    if not np.isfinite(seconds) or seconds<=0.:
        raise ValueError('positive finite solver seconds required')
    master=base.build_bc_master(domain,bank,jacobian,anchor,segments=segments,
                                exp_segments=exp_segments,power_radius=power_radius)
    at_parent=base.evaluate_proxy(master,anchor)
    master.update(anchor_proxy_value=at_parent['proxy'],anchor_proxy_error=at_parent['actual_proxy_error'],anchor_envelope=False)
    return _solve_master(domain,bank,jacobian,anchor,master,base.evaluate_proxy,
                         seconds,backend,operation_start,began)
