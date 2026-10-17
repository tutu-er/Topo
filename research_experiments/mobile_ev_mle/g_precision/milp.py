"""Max-min diagonal quadratic tasks on the existing single-EV operation MILP.

Only the frozen surrogate is bounded here.  Feasible plans are scored with
their original coefficients, never by the square-chord incumbent.  The
operation graph, SOC, travel, setup and charge modes are reused unchanged.
"""
from __future__ import annotations

from collections.abc import Mapping
from time import perf_counter

import numpy as np

from ..ev_original_domain_milp import OriginalDomainPair, exchange_original_domain


def _plain(value):
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, np.generic):
        return _plain(value.item())
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _task_arrays(tasks, shape):
    tasks = tuple(tasks)
    if not tasks:
        raise ValueError('at least one task is required')
    result = []
    names = set()
    for task in tasks:
        name = task.name
        if not isinstance(name, str) or not name or name in names:
            raise ValueError('task names must be unique nonempty strings')
        a = float(task.constant)
        b = np.asarray(task.linear, float).copy()
        d = np.asarray(task.quadratic, float).copy()
        if (not np.isfinite(a) or b.shape != shape or d.shape != shape
                or not np.isfinite(b).all() or not np.isfinite(d).all()):
            raise ValueError('task coefficients must be finite and match the control shape')
        if np.any(d < 0):
            raise ValueError('negative quadratic coefficients are unsupported')
        if np.any((d == 0) & (b != 0)):
            raise ValueError('zero quadratic coefficient requires zero linear coefficient')
        if not callable(getattr(task, 'value', None)):
            raise ValueError('task must expose value(power)')
        result.append(dict(name=name, constant=a, linear=b, quadratic=d, task=task))
        names.add(name)
    return result


def _values(tasks, power):
    """Evaluate original coefficients in extended precision before converting."""
    ld = np.longdouble
    u = np.asarray(power, ld)
    values, guards = {}, []
    eps = np.finfo(float).eps
    for task in tasks:
        a = ld(task['constant'])
        linear = np.asarray(task['linear'], ld) * u
        quadratic = np.asarray(task['quadratic'], ld) * u * u
        value = float(a + linear.sum() + quadratic.sum())
        magnitude = float(abs(a) + abs(linear).sum() + abs(quadratic).sum())
        guard = 512 * eps * (1 + magnitude)
        declared = float(task['task'].value(np.asarray(power, float)))
        if not np.isfinite([value, declared, guard]).all():
            raise ValueError('nonfinite task value')
        if abs(value - declared) > guard:
            raise ValueError('task.value disagrees with its original coefficients')
        values[task['name']] = value
        guards.append(guard)
    score = min(values.values())
    lower = min(values[t['name']] - g for t, g in zip(tasks, guards))
    return values, float(score), float(lower)


def _box_upper(tasks, low, high):
    ld = np.longdouble
    l, h = np.asarray(low, ld), np.asarray(high, ld)
    uppers = []
    for task in tasks:
        b, d = (np.asarray(task[key], ld).ravel() for key in ('linear', 'quadratic'))
        left, right = b*l + d*l*l, b*h + d*h*h
        value = float(ld(task['constant']) + np.maximum(left, right).sum())
        magnitude = float(abs(ld(task['constant'])) + np.maximum(abs(left), abs(right)).sum())
        uppers.append(value + 512*np.finfo(float).eps*(1+magnitude))
    return float(min(uppers))


def _pairs(tasks, low, high):
    """Complete squares with one common scale and an explicit rounding bound."""
    ubar = np.maximum(abs(low), abs(high))
    magnitude = max(abs(t['constant']) + float(np.sum(abs(t['linear'].ravel())*ubar
                    + t['quadratic'].ravel()*ubar**2)) for t in tasks)
    scale = max(float(magnitude), 1e-12) if magnitude else 1.
    if not np.isfinite(scale):
        raise ValueError('task magnitude overflows the control domain')
    pairs, offsets, errors = [], {}, []
    n = len(low)
    ld = np.longdouble
    for task in tasks:
        a = task['constant']/scale
        b = task['linear'].ravel()/scale
        d = task['quadratic'].ravel()/scale
        if (np.any((task['quadratic'].ravel() != 0) & (d == 0))
                or np.any((task['linear'].ravel() != 0) & (b == 0))):
            raise ValueError('common scaling underflows nonzero task coefficients')
        root = np.sqrt(d)
        intercept = np.divide(b, 2*root, out=np.zeros_like(b), where=d > 0)
        offset = float(np.sum(np.asarray(intercept, ld)**2) - ld(a))
        if not np.isfinite(intercept).all() or not np.isfinite(offset):
            raise ValueError('completing squares overflows; task is numerically unsupported')
        pair = OriginalDomainPair(task['name'], intercept, np.diag(root),
            np.zeros((n, 0)), np.zeros((n, n, 0)), np.zeros(0), np.zeros(0),
            np.zeros(0), np.zeros((0, n)))
        # Stored floating-point completion can differ from the supplied
        # polynomial. Enclose that coefficient difference on the full box.
        v, rr = np.asarray(intercept, ld), np.asarray(root, ld)
        da = abs(np.sum(v*v) - ld(offset) - ld(a))
        db = abs(2*v*rr - np.asarray(b, ld))
        dd = abs(rr*rr - np.asarray(d, ld))
        error = float(da + db @ np.asarray(ubar, ld) + dd @ np.asarray(ubar, ld)**2)
        error += 256*np.finfo(float).eps*(1+abs(offset)+float(np.sum(abs(v*v))))
        pairs.append(pair)
        offsets[task['name']] = offset
        errors.append(error)
    return pairs, offsets, scale, float(max(errors)*scale)


def _iteration_log(entry, scale, completion_error):
    master = entry.get('master', {})
    upper = (None if entry.get('global_upper') is None else
             float(entry['global_upper'])*scale+completion_error)
    lower = (None if entry.get('global_lower') is None else
             float(entry['global_lower'])*scale-completion_error)
    return dict(iteration=entry.get('iteration'), global_upper=upper,
        global_lower=lower, gap=None if upper is None or lower is None else upper-lower,
        control_segments=entry.get('control_segments'),
        repeated_control=entry.get('repeated_control'),
        upper_update_accepted=entry.get('upper_update_accepted'),
        master={k: master.get(k) for k in ('status', 'message', 'solver_success',
            'solver_mip_gap', 'solver_nodes', 'backend', 'master_form', 'n_variables',
            'n_rows', 'n_binary', 'n_shared_square_binary', 'n_shared_square_controls',
            'bound_consistent', 'seconds', 'backend_details')},
        master_objective_upper=(None if master.get('objective_upper') is None else
                                float(master['objective_upper'])*scale+completion_error),
        master_solver_best_bound=(None if master.get('solver_best_bound') is None else
                                  float(master['solver_best_bound'])*scale+completion_error),
        chord_incumbent=(None if master.get('surrogate_incumbent') is None else
                         float(master['surrogate_incumbent'])*scale),
        square_segment_error_upper=(None if master.get(
            'pwl_error_upper_against_finite_witness_objective') is None else
            float(master['pwl_error_upper_against_finite_witness_objective'])*scale),
        chord_incumbent_is_original_lower=False)


def solve_tasks(domain, tasks, parents, *, iterations=2, segments=4, max_segments=16,
                solver_seconds=2, total_seconds=12, backend='gurobi',
                relative_gap=.01, absolute_gap=1e-5):
    """Return the best original-score feasible plan, always retaining parents.

    Tasks implement name/constant/linear[T,J]/quadratic[T,J]/value(power).
    Quadratic coefficients must be nonnegative; a zero coefficient requires
    its corresponding linear coefficient also to vanish.  Arbitrary negative
    constants are allowed. All task scores and returned bounds use the input
    units; normalization divides every task by the same positive scale.
    """
    began = perf_counter()
    for value, name in ((iterations, 'iterations'), (segments, 'segments'),
                        (max_segments, 'max_segments')):
        if isinstance(value, bool) or int(value) != value or value < 1:
            raise ValueError(name+' must be a positive integer')
    if max_segments < segments:
        raise ValueError('max_segments must cover segments')
    if (not np.isfinite([solver_seconds, total_seconds, relative_gap, absolute_gap]).all()
            or min(solver_seconds, total_seconds) <= 0
            or min(relative_gap, absolute_gap) < 0):
        raise ValueError('finite positive budgets and nonnegative gaps required')
    if backend not in ('scipy', 'gurobi'):
        raise ValueError('backend must be scipy or gurobi')
    if not isinstance(parents, Mapping):
        raise ValueError('parents must map names to full operation vectors')
    shape = tuple(np.asarray(domain.indices.u).shape)
    task_data = _task_arrays(tasks, shape)
    operation = domain.as_dict()
    low = np.asarray(operation['control_lower'], float)
    high = np.asarray(operation['control_upper'], float)
    pairs, offsets, scale, completion_error = _pairs(task_data, low, high)
    failures, candidates, parent_logs = [], [], {}

    def keep(x, origin):
        x = np.asarray(x, float)
        check = domain.check(x)
        if not check['feasible']:
            raise ValueError('operation plan is infeasible: '+str(check.get('issues')))
        power = np.asarray(domain.decode(x)['u_kw'], float)
        if power.shape != shape or not np.isfinite(power).all():
            raise ValueError('decoded power is invalid')
        values, score, lower = _values(task_data, power)
        value = dict(origin=origin, operation_x=x.copy(), power_kw=power.copy(),
                     task_values=values, score=score, score_lower=lower,
                     operation_check=check)
        candidates.append(value)
        return value

    for name, x in parents.items():
        if not isinstance(name, str) or not name:
            raise ValueError('parent names must be nonempty strings')
        try:
            row = keep(x, 'parent:'+name)
            parent_logs[name] = dict(success=True, **row)
        except Exception as exc:
            reason = type(exc).__name__+': '+str(exc)
            parent_logs[name] = dict(success=False, failure=reason)
            failures.append(dict(kind='parent_rejected', origin=name, message=reason))
    valid_parents = list(candidates)
    best_parent = max(valid_parents, key=lambda r: r['score']) if valid_parents else None
    exchange = None
    remaining = total_seconds - (perf_counter()-began)
    if remaining > 0:
        try:
            exchange = exchange_original_domain(pairs, low, high,
                operation=operation, offsets=offsets, aggregation='worst', score_scale=1.,
                zero_product_pairs=[], initial_operation_points=[r['operation_x'] for r in valid_parents],
                max_iterations=int(iterations), control_segments=int(segments),
                residual_segments=int(segments), max_segments=int(max_segments),
                solver_seconds=min(float(solver_seconds), remaining), total_seconds=remaining,
                absolute_gap=float(absolute_gap)/scale, relative_gap=float(relative_gap),
                backend=backend, master_form='shared_control_square', witness_policy='affine_preferred')
            for i, row in enumerate(exchange.get('evaluations', [])):
                try:
                    keep(row['operation_x'], 'exchange:'+str(i))
                except Exception as exc:
                    failures.append(dict(kind='exchange_plan_rejected', index=i,
                                         message=type(exc).__name__+': '+str(exc)))
            if exchange.get('operation_x') is not None:
                try:
                    keep(exchange['operation_x'], 'exchange:selected')
                except Exception as exc:
                    failures.append(dict(kind='exchange_selected_rejected',
                                         message=type(exc).__name__+': '+str(exc)))
            for failure in exchange.get('failures', []):
                failures.append({k: failure[k] for k in ('kind', 'origin', 'pair', 'pairs',
                    'index', 'iteration', 'message') if k in failure})
        except Exception as exc:
            failures.append(dict(kind='exchange_exception', message=type(exc).__name__+': '+str(exc)))
    else:
        failures.append(dict(kind='exchange_skipped_total_budget'))

    winner = max(candidates, key=lambda r: r['score']) if candidates else None
    upper = _box_upper(task_data, low, high)
    coarse_upper = upper
    represented_upper = None
    if exchange is not None and exchange.get('objective_upper') is not None:
        represented_upper = float(exchange['objective_upper'])*scale
        if np.isfinite(represented_upper):
            restored = represented_upper + completion_error
            restored += 256*np.finfo(float).eps*(1+abs(restored))
            upper = min(upper, restored)
    lower = None if winner is None else winner['score_lower']
    if lower is not None and upper < lower:
        failures.append(dict(kind='numerical_upper_below_original_lower',
                             rejected_upper=upper, lower=lower))
        upper = coarse_upper
    gap = None if lower is None else float(upper-lower)
    target = None if lower is None else float(absolute_gap + relative_gap*max(
        abs(upper), abs(lower), 1e-12))
    completed = bool(winner is not None and gap >= 0 and gap <= target)
    logs = [] if exchange is None else [_iteration_log(r, scale, completion_error)
                                      for r in exchange.get('iterations', [])]
    answer = dict(success=winner is not None,
        operation_x=None if winner is None else winner['operation_x'],
        power_kw=None if winner is None else winner['power_kw'],
        task_values=None if winner is None else winner['task_values'],
        score=None if winner is None else winner['score'],
        selected_origin=None if winner is None else winner['origin'],
        best_parent_score=None if best_parent is None else best_parent['score'],
        score_gain=None if winner is None or best_parent is None else winner['score']-best_parent['score'],
        inheritance_protected=bool(winner is not None and (best_parent is None or
                                  winner['score'] >= best_parent['score'])),
        objective_lower=lower, objective_upper=upper, gap=gap, target_gap=target,
        completed=completed, unfinished=not completed,
        coarse_box_upper=coarse_upper, represented_surrogate_upper=represented_upper,
        completion_error_bound=completion_error, objective_scale=scale,
        scale_meaning='all task coefficients divided by this one common value; all result bounds restored',
        nuisance_dimension=0, task_names=[t['name'] for t in task_data],
        parents=parent_logs, iterations=logs, n_iterations=len(logs),
        status='exchange_unavailable' if exchange is None else exchange.get('status'),
        failures=failures, backend=backend, seconds=perf_counter()-began,
        exchange_seconds=None if exchange is None else exchange.get('seconds'),
        operation_check=None if winner is None else winner['operation_check'],
        bound_scope='fixed affine plus diagonal quadratic surrogate; numerical bounds, no nonlinear AC certificate',
        uses_nuisance_refit=False, incumbent_is_original_score_lower=False,
        full_route_or_curve_enumeration=False)
    return _plain(answer)


__all__ = ['solve_tasks']
