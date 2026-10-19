"""Continuous bilinear max-min design with fixed historical cost offsets.

No future observation, true topology or noise is accepted. All supplied
pairs remain in every inner evaluation and master. Bounds concern the
declared bilinear residual boxes and linear control domain only. If a common
outer nuisance box is supplied, they concern that relaxed proxy. Original
action-dependent nuisance boxes can be audited separately.
"""
from __future__ import annotations

from collections.abc import Mapping
from time import perf_counter
import numpy as np
from scipy.optimize import LinearConstraint
from .ev_bilinear_support_milp import BilinearPair, evaluate_qp, fixed_witness_upper
from .ev_bilinear_exchange import witness_quadratic
from .ev_profile_chord_milp import FrozenQuadratic, solve_profile_chord_milp, check_control, _constraints


def all_ordered_pairs(names):
    """Keep all distinct ordered topology pairs; no data-based pair deletion."""
    names = tuple(names)
    if len(names) < 2 or len(set(names)) != len(names) or any(not isinstance(n, str) or not n for n in names):
        raise ValueError('at least two unique nonempty candidate names required')
    return tuple((a, b) for a in names for b in names if a != b)


def history_score_offsets(ordered_pairs, history_losses, *, mode='added'):
    """Unscaled SSE offset: competitor H cost for added, source H cost for joint.

    Historical costs may be locally attained costs. Calling them global
    minima requires independent optimization evidence. The mode changes
    the design question and can change its optimal curve.
    """
    if mode not in ('added', 'joint'):
        raise ValueError("mode must be 'added' or 'joint'")
    answer = {}
    for a, b in ordered_pairs:
        if a == b:
            raise ValueError('distinct candidates required')
        if a not in history_losses or b not in history_losses:
            raise ValueError('every required historical cost must be supplied')
        values = [history_losses[a], history_losses[b]]
        if not np.isfinite(values).all() or np.any(np.asarray(values) < 0):
            raise ValueError('historical SSE costs must be finite and nonnegative')
        name = a+'__to__'+b
        if name in answer:
            raise ValueError('duplicate ordered pair')
        answer[name] = float(history_losses[b if mode == 'added' else a])
    if not answer:
        raise ValueError('nonempty pair list required')
    return answer


def difference_interval(comp_lower, comp_upper, own_lower, own_upper, *, scale=.5, offset=0.):
    """Interval for scale*(Q_comp-Q_own-offset); endpoints never mix lower-lower."""
    values = np.asarray([comp_lower, comp_upper, own_lower, own_upper, scale, offset], float)
    if not np.isfinite(values).all() or scale <= 0 or comp_lower > comp_upper or own_lower > own_upper:
        raise ValueError('ordered finite cost intervals and positive scale required')
    return dict(lower=float(scale*(comp_lower-own_upper-offset)),
                upper=float(scale*(comp_upper-own_lower-offset)))


def witness_score_quadratic(pair, z, offset, scale, control_lower, control_upper, *, name=None):
    """Exact reduced QR of [T,d], with a numerical upward guard for the box.

    All constant columns are retained. A fixed feasible z gives an upper
    cost at every control in the supplied box. QR only changes residual
    coordinates, not the mathematical sum of squares.
    """
    if not np.isfinite([offset, scale]).all() or scale <= 0:
        raise ValueError('finite offset and positive scale required')
    original = witness_quadratic(pair, z, name=name)
    low, high = np.asarray(control_lower, float), np.asarray(control_upper, float)
    if low.shape != (pair.control_dimension,) or high.shape != low.shape or not np.isfinite([low, high]).all() or np.any(low > high):
        raise ValueError('finite ordered control box required')
    joined = np.column_stack((original.T, original.d))
    _, triangular = np.linalg.qr(joined, mode='reduced')
    gram = joined.T@joined
    discrepancy = triangular.T@triangular-gram
    # ||delta_Gram||_2 <= ||delta_Gram||_infinity for a symmetric matrix.
    norm_bound = float(np.max(np.sum(abs(discrepancy), axis=1), initial=0.))
    round_guard = 128*np.finfo(float).eps*(1+float(np.max(np.sum(abs(gram), axis=1), initial=0.)))
    vector_norm_upper = 1+float(np.maximum(low*low, high*high).sum())
    upward_guard = float(scale*(norm_bound+round_guard)*vector_norm_upper)
    quadratic = FrozenQuadratic(original.name, np.sqrt(scale)*triangular[:, -1],
        np.sqrt(scale)*triangular[:, :-1], scale*offset-upward_guard)
    return quadratic, dict(original_rows=len(original.d), compressed_rows=len(quadratic.d),
        constant_column_retained=True, gram_max_abs_error=float(np.max(abs(discrepancy), initial=0.)),
        gram_norm_bound=norm_bound, box_numerical_upward_guard=upward_guard)


def evaluate_action_box(pair, u, lower_center, upper_center, lower_control=None, upper_control=None):
    """Audit the original fixed-u QP when nuisance bounds are affine in u.

    Does not change the outer master's common box or claim an AC bound.
    """
    u = np.asarray(u, float)
    if u.shape != (pair.control_dimension,) or not np.isfinite(u).all():
        raise ValueError('finite control vector of matching dimension required')
    lo, hi = np.asarray(lower_center, float), np.asarray(upper_center, float)
    shape = (len(pair.lower), pair.control_dimension)
    L = np.zeros(shape) if lower_control is None else np.asarray(lower_control, float)
    U = np.zeros(shape) if upper_control is None else np.asarray(upper_control, float)
    if lo.shape != pair.lower.shape or hi.shape != pair.upper.shape or L.shape != shape or U.shape != shape:
        raise ValueError('action box dimensions must match nuisance and control dimensions')
    if not all(np.isfinite(a).all() for a in (lo, hi, L, U)):
        raise ValueError('finite affine nuisance bounds required')
    low, high = lo+L@u, hi+U@u
    if np.any(low > high):
        raise ValueError('original action nuisance box is empty')
    inside_outer = bool(np.all(low >= pair.lower-1e-12) and np.all(high <= pair.upper+1e-12))
    audit_pair = BilinearPair(pair.name, pair.c, pair.B, pair.A0, pair.At, low, high)
    answer = evaluate_qp(audit_pair, u)
    answer.update(original_action_lower=low, original_action_upper=high,
        original_box_inside_declared_outer=inside_outer,
        scope='declared bilinear fixed-action original nuisance box; no nonlinear AC certificate')
    return answer


def charging_service_domain(site, n_slots, n_sites, service, *, power_scale=1.):
    """Pure-charge control box and linear SOC/task constraints in chosen units.

    power_scale=BASE_KW means controls are kW; power_scale=1 means pu. Arrival,
    setup, access, driving energy and terminal battery service are included.
    """
    from .ev_service_milp import operational_data, _service_precheck, constant_site_plan
    if not np.isfinite(power_scale) or power_scale <= 0:
        raise ValueError('positive finite power scale required')
    data = operational_data(site, n_slots, n_sites, service)
    reason = _service_precheck(data, service)
    constant = constant_site_plan(site, n_slots, n_sites, service)
    if reason or data['required_charging_gain'] < -1e-12:
        return dict(success=False, reason=reason or 'fixed battery task requires discharge', operations=data)
    gain_per_power = service.dt_hours*service.eta_charge/(service.battery_capacity*power_scale)
    cumulative = gain_per_power*np.tril(np.ones((n_slots, n_slots)))
    initial = data['post_drive_energy']/service.battery_capacity
    required_terminal = service.terminal_energy/service.battery_capacity-initial
    matrix = np.vstack((cumulative, cumulative[-1]))
    lower = np.r_[np.full(n_slots, service.soc_min-initial), required_terminal]
    upper = np.r_[np.full(n_slots, service.soc_max-initial), required_terminal]
    return dict(success=True, lower=np.zeros(n_slots), upper=data['charge']*power_scale,
        constraints=LinearConstraint(matrix, lower, upper),
        constant_u=constant['plan'].p[:, site]*power_scale if constant['success'] else None,
        constant_plan=constant['plan'] if constant['success'] else None, constant_status=constant['status'],
        operations=data, power_scale=float(power_scale), pure_charging=True,
        terminal_battery_service=float(service.terminal_energy))


def exchange_bilinear_scores(pairs, control_lower, control_upper, constraints=None, *,
        offsets=None, score_scale=.5, initial_u=None, max_iterations=8,
        segments=4, max_segments=32, solver_seconds=5., total_seconds=60.,
        absolute_gap=1e-5, relative_gap=1e-3, mip_rel_gap=1e-8, feasibility_tolerance=1e-7):
    """Maximize min_k scale*(q_k(u)-offset_k) with signed numerical bounds."""
    began = perf_counter(); pairs = tuple(pairs)
    low, high = np.asarray(control_lower, float), np.asarray(control_upper, float)
    if (not pairs or low.ndim != 1 or not len(low) or high.shape != low.shape
            or not np.isfinite([low, high]).all() or np.any(low > high)
            or any(not isinstance(p, BilinearPair) or p.control_dimension != len(low) for p in pairs)
            or len({p.name for p in pairs}) != len(pairs)):
        raise ValueError('unique compatible pairs and finite ordered control box required')
    if offsets is None:
        offsets = {p.name: 0. for p in pairs}
    if not isinstance(offsets, Mapping) or set(offsets) != {p.name for p in pairs}:
        raise ValueError('exactly one fixed offset for every required pair must be supplied')
    offset_values = np.asarray([offsets[p.name] for p in pairs], float)
    if not np.isfinite(offset_values).all() or not np.isfinite(score_scale) or score_scale <= 0:
        raise ValueError('finite offsets and positive finite score scale required')
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
    initial_upper = min(score_scale*(fixed_witness_upper(p, low, high, zs[0])**2-o)
                        for p, zs, o in zip(pairs, witnesses, offset_values))
    upper = float(initial_upper); lower = None; best = None
    iterations, evaluations, failures = [], [], []
    qp_calls = 0; current_segments = int(segments)

    def evaluate(u, origin):
        nonlocal lower, best, qp_calls
        audit = check_control(u, low, high, physical, tolerance=feasibility_tolerance)
        if not audit['feasible']:
            raise ValueError('candidate fails declared operations: '+str(audit))
        u = np.asarray(u, float).copy()
        solutions, added = [], []
        for k, pair in enumerate(pairs):
            qp_calls += 1
            solution = evaluate_qp(pair, u); solutions.append(solution)
            z = solution['z']
            novel = not any(np.array_equal(z, old) for old in witnesses[k])
            if novel:
                witnesses[k].append(z.copy())
            added.append(novel)
        pair_lower = score_scale*(np.asarray([s['q_tangent_lower'] for s in solutions])-offset_values)
        pair_upper = score_scale*(np.asarray([s['q_primal_upper'] for s in solutions])-offset_values)
        entry = dict(origin=origin, u=u, score_lower=float(min(pair_lower)), score_upper=float(min(pair_upper)),
            pair_score_lower=pair_lower, pair_score_upper=pair_upper, qp_results=solutions,
            added_witness=added, physical_validation=audit,
            all_qp_converged=all(s['success'] for s in solutions))
        evaluations.append(entry)
        if lower is None or entry['score_lower'] > lower:
            lower, best = entry['score_lower'], entry
        if not entry['all_qp_converged']:
            failures.append(dict(origin=origin, kind='inner_qp_not_converged',
                pairs=[p.name for p, s in zip(pairs, solutions) if not s['success']]))
        return entry

    def target():
        return float(absolute_gap+relative_gap*max(1e-12, abs(upper), 0. if lower is None else abs(lower)))

    def converged():
        consistency = 1e-9*max(1., abs(upper), 0. if lower is None else abs(lower))
        return lower is not None and upper >= lower-consistency and upper-lower <= target()

    if initial_u is not None:
        evaluate(initial_u, 'initial_feasible_control')
    status = 'gap_reached' if converged() else 'iteration_limit'
    if not converged():
        for iteration in range(int(max_iterations)):
            remaining = total_seconds-(perf_counter()-began)
            if remaining <= 0:
                status = 'time_budget'; break
            quadratics, compressions = [], []
            for p, zs, offset in zip(pairs, witnesses, offset_values):
                for j, z in enumerate(zs):
                    q, comp = witness_score_quadratic(p, z, offset, score_scale, low, high,
                                                      name=p.name+'__w'+str(j))
                    quadratics.append(q); compressions.append(dict(pair=p.name, witness=j, **comp))
            master = solve_profile_chord_milp(quadratics, low, high, physical,
                segments=current_segments, time_limit=min(solver_seconds, remaining), mip_rel_gap=mip_rel_gap,
                feasible_incumbents=[] if best is None else [best['u']], feasibility_tolerance=feasibility_tolerance)
            before_lower, before_upper = lower, upper
            upper = min(upper, float(master['objective_upper']))
            record = dict(iteration=iteration, segments=current_segments,
                witnesses_before=[len(zs) for zs in witnesses], master=master, compressions=compressions,
                previous_lower=before_lower, previous_upper=before_upper, global_upper=upper,
                global_lower=lower, selected_u=master['u'], repeated_control=False, refinement_next=False)
            iterations.append(record)
            if master['u'] is None:
                status = 'operations_infeasible' if master['status'] == 2 and best is None else 'master_without_valid_incumbent'
                failures.append(dict(iteration=iteration, kind=status, message=master['message'])); break
            repeated = any(np.allclose(master['u'], e['u'], rtol=0., atol=1e-9*max(1., float(np.max(abs(high-low))))) for e in evaluations)
            try:
                entry = evaluate(master['u'], 'master_'+str(iteration))
            except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError) as error:
                failures.append(dict(iteration=iteration, kind='candidate_qp_failure', message=str(error)))
                status = 'candidate_qp_failure'; break
            record.update(global_lower=lower, evaluation_index=len(evaluations)-1,
                witnesses_after=[len(zs) for zs in witnesses], repeated_control=repeated,
                gap=upper-lower, target_gap=target())
            if upper < lower-1e-9*max(1., abs(upper), abs(lower)):
                failures.append(dict(iteration=iteration, kind='numerical_bound_inconsistency', lower=lower, upper=upper))
                status = 'numerical_bound_inconsistency'; break
            if converged():
                status = 'gap_reached'; break
            no_new_witness = not any(entry['added_witness'])
            if (repeated or no_new_witness) and current_segments < max_segments:
                current_segments = min(2*current_segments, int(max_segments)); record['refinement_next'] = True
            elif repeated and no_new_witness and current_segments >= max_segments:
                status = 'repeated_control_with_gap'; break
    gap = None if lower is None else upper-lower
    return dict(u=None if best is None else best['u'], objective_lower=lower, objective_upper=upper,
        gap=None if gap is None else max(0., gap), raw_gap=gap, target_gap=target(),
        status=status, completed=(status == 'gap_reached'), unfinished=(status != 'gap_reached'),
        initial_fixed_witness_upper=float(initial_upper), iterations=iterations, evaluations=evaluations,
        witnesses=witnesses, witness_counts=[len(zs) for zs in witnesses], qp_calls=qp_calls,
        inner_qp_failures=sum(not e['all_qp_converged'] for e in evaluations), failures=failures,
        best_evaluation_index=None if best is None else next(i for i, e in enumerate(evaluations) if e is best),
        seconds=perf_counter()-began, final_segments=current_segments, score_scale=float(score_scale),
        offsets=dict(zip([p.name for p in pairs], offset_values.tolist())),
        required_pairs=[p.name for p in pairs], full_curve_enumeration=False,
        self_profile_subtracted=False, fixed_historical_offsets=True, original_global_optimum_certified=False,
        scope='signed score bounds for declared bilinear proxy over fixed nuisance boxes and linear operations; no nonlinear AC certificate')
