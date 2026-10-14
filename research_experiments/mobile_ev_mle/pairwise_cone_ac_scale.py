"""Gate 2: fixed H-only source means versus refitted legal AC competitors.

All nine directed cherry-to-other-cherry/star pairs, two positive engineering
R/X directions, three scales and two fixed node0 actions are retained. No new
selector, future truth, formal calibration or budget recovery is used.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace
import numpy as np
from scipy.optimize import lsq_linear, nnls
from . import likelihood, information_context
from . import information_fit_recovery as recovery_module
from .information_context import meter_from_record, meter_to_record, history_white_jacobian
from .information_fit_recovery import (read_json, write_json, sha, immutable_manifest, exact_candidate,
    independent_replay, run_fit)
from .information_replay import audit_source, _obs
from .pairwise_cone_checks import pairwise_cone_distance
from .structured_information import project_columns
from .single_site_milp import (SingleSiteConstraints, SingleSitePlan, constant_site_plan,
    solve_site_step, check_single_site_plan)

HERE = Path(__file__).resolve().parent
FROZEN = HERE/'evidence/structured_information/v3_frozen'
PILOT = HERE/'evidence/structured_information/v2_targeted/pilot'
DIRECTIONS = {'R': np.array([.003, .00003]), 'X': np.array([.00003, .003])}
SCALES = (1., .5, .25)
SUPPORTS = ((0, 1), (0, 2), (1, 2))


def select_saved_input(pilot=PILOT, *, history_id=None):
    """Explicit H id, then filename and anchor-id only; no truth access."""
    paths = sorted(Path(pilot).glob('trial_*.json.gz'))
    unique_history_ids = sorted({path.name.split('_')[1] for path in paths})
    if history_id is not None:
        history_id = str(history_id)
        if not history_id.isdecimal(): raise ValueError('history id must be numeric')
        paths = [path for path in paths if path.name.startswith('trial_'+history_id+'_')]
    if not paths: raise ValueError('requested history id has no saved pilot record')
    path = paths[0]
    saved = read_json(path)
    if history_id is not None and str(saved['history_id']) != history_id: raise ValueError('filename H id differs from saved H identity')
    return _decode_pilot_record(saved, path, history_id=history_id, unique_history_ids=unique_history_ids)


def _decode_pilot_record(saved, path, *, history_id=None, unique_history_ids):
    centres = saved['selection']['centres']
    anchor = sorted({row['anchor_id'] for row in centres})[0]
    centres = [row for row in centres if row['anchor_id'] == anchor]
    if len(centres) != 3 or {tuple(row['target_support']) for row in centres} != set(SUPPORTS):
        raise ValueError('the first pilot has no common three-cherry H-only anchor')
    by_support = {tuple(row['target_support']): row for row in centres}
    first = by_support[SUPPORTS[0]]
    candidates = {row['name']: exact_candidate(row, 3) for row in centres}
    star = likelihood.RadialCandidate.from_supports('star', 3)
    for row in centres:
        for key in ('history_p', 'history_q', 'history_source_p', 'history_source_q', 'history_root_v'):
            if not np.array_equal(row[key], first[key]): raise ValueError('H states differ between zero-edge embeddings')
        for key in ('p', 'q', 'source_p', 'source_q', 'root_v'):
            if not np.array_equal(row['future'][key], first['future'][key]): raise ValueError('H-only forecast differs between embeddings')
        candidate = candidates[row['name']]
        edge = candidate.supports.index(tuple(row['target_support']))
        if row['r'][edge] != 0 or row['x'][edge] != 0: raise ValueError('inserted target must be exactly zero')
    base_r = np.array([first['r'][candidates[first['name']].supports.index(s)] for s in star.supports])
    base_x = np.array([first['x'][candidates[first['name']].supports.index(s)] for s in star.supports])
    for row in centres:
        candidate = candidates[row['name']]
        for key, reference in (('r', base_r), ('x', base_x)):
            values = np.array([row[key][candidate.supports.index(s)] for s in star.supports])
            if not np.array_equal(values, reference): raise ValueError('star common-line values differ')
    return dict(path=str(path), input_sha256=sha(path), history_id=saved['history_id'], quality=saved['quality'],
        requested_history_id=history_id, unique_saved_H_ids=unique_history_ids,
        unique_H_order_position=unique_history_ids.index(str(saved['history_id']))+1,
        anchor_id=anchor, centres=centres, star=star, candidates=candidates,
        star_r=base_r, star_x=base_x, history_observations=saved['history_observations'],
        models=saved['measurement_models'], device_constraints=saved['device_constraints'],
        policy_seed=int(saved['seeds']['policy']), source_uses_H_only=True,
        input_mode='saved_pilot_filename_order',
        future_truth_accessed=False, future_realized_measurements_accessed=False)


def load_compact_inputs(path, *, history_id=None):
    """Explicit archived-input mode; never replaces normal filename selection."""
    path = Path(path)
    compact = read_json(path)
    selected = str(compact['history_id'])
    if history_id is not None and str(history_id) != selected:
        raise ValueError('requested H id differs from compact evidence identity')
    saved = dict(history_id=selected, quality=compact['quality'],
        selection=dict(centres=compact['centres']), history_observations=compact['history_observations'],
        measurement_models=compact['models'], device_constraints=compact['device_constraints'],
        seeds=dict(policy=compact['policy_seed']))
    decoded = _decode_pilot_record(saved, path, history_id=history_id,
        unique_history_ids=compact.get('unique_saved_H_ids', [selected]))
    for key in ('anchor_id', 'star_r', 'star_x'):
        if key == 'anchor_id':
            if decoded[key] != compact[key]: raise ValueError('compact anchor identity differs')
        elif not np.array_equal(decoded[key], compact[key]):
            raise ValueError('compact star coordinates differ from zero-edge embeddings')
    first = decoded['centres'][0]
    for channel in ('p', 'q', 'source_p', 'source_q', 'root_v'):
        historical = np.asarray(first['history_'+channel], float)
        future = np.asarray(first['future'][channel], float)
        expected = np.broadcast_to(historical.mean(axis=0), future.shape)
        if not np.array_equal(future, expected):
            raise ValueError('compact forecast is not the H-only mean:'+channel)
    history = _obs(decoded['history_observations'])
    if history.t != 12 or history.n != 3 or not np.array_equal(first['history_root_v'], history.root_v):
        raise ValueError('compact H12/n3/root calibration differs')
    models = {name: meter_from_record(decoded['models'][name]) for name in ('history', 'future', 'joint')}
    if meter_to_record(models['joint']) != compact['actual_joint_meter']:
        raise ValueError('compact duplicate joint meter differs')
    for channel in ('p', 'q', 'v', 'master_p', 'master_q', 'source_p', 'source_q'):
        sigmas = [getattr(models[key], channel+'_sigma') for key in ('history', 'future', 'joint')]
        if getattr(history, channel) is None:
            if any(value is not None for value in sigmas): raise ValueError('sigma for absent channel:'+channel)
            continue
        tail = (3,) if channel in ('p', 'q', 'v') else ()
        h, n, j = [np.broadcast_to(np.asarray(value, float), (length, *tail))
                   for value, length in zip(sigmas, (12, 4, 16))]
        if any(not np.isfinite(value).all() or np.any(value <= 0) for value in (h, n, j)):
            raise ValueError('compact sigma must be positive finite:'+channel)
        if not np.array_equal(j, np.concatenate((h, n))):
            raise ValueError('compact joint sigma does not join H/N:'+channel)
    for channel in ('p', 'q', 'v'):
        biases = [np.broadcast_to(getattr(models[key], 'bias_'+channel), (3,)) for key in ('history', 'future', 'joint')]
        if not all(np.array_equal(value, biases[0]) for value in biases[1:]):
            raise ValueError('compact calibrated bias differs across H/N/joint:'+channel)
    constraints = SingleSiteConstraints(**decoded['device_constraints'])
    constraints.arrays(4, 3)
    if set(compact['actions']) != {'constant0', 'alternating0'}:
        raise ValueError('compact fixed action set differs')
    for name, action in compact['actions'].items():
        values = action['plan']
        plan = SingleSitePlan(int(values['site']), *[np.asarray(values[key]) for key in ('p', 'q', 'sites', 'soc')], values.get('solver', {}))
        if plan.site != 0 or not check_single_site_plan(plan, constraints)['feasible']:
            raise ValueError('compact action/constraints mismatch:'+name)
        actual = total_states(first, plan)
        if any(not np.array_equal(actual[key], compact['total_H_N_states'][name][key]) for key in actual):
            raise ValueError('compact action/H-only states mismatch or EV counted twice:'+name)
        common_bounds(common_point(decoded, actual))
    decoded.update(input_mode='archived_compact_evidence', compact_input_path=str(path),
        compact_input_sha256=sha(path), original_saved_input_file=compact['path'],
        original_saved_input_sha256=compact['input_sha256'], original_raw_file_required=False,
        compact_input_validation_passed=True, original_filename_selection_replayed=False)
    return decoded


def fixed_actions(constraints):
    constant = constant_site_plan(0, 4, 3, constraints)
    alternating = solve_site_step(0, 4, 3, constraints, intercepts=[0.],
        gradients=np.array([[1., -1., 1., -1.]]), time_limit=5.)
    records = {}
    for name, result in (('constant0', constant), ('alternating0', alternating)):
        if not result['success']: raise ValueError('required public operation failed:'+name)
        plan = result['plan']
        audit = check_single_site_plan(plan, constraints)
        if not audit['feasible']: raise ValueError('public operation infeasible:'+name)
        records[name] = dict(plan=plan, audit=audit, operation={key: value for key, value in result.items() if key != 'plan'})
    return records


def total_states(centre, plan):
    future = centre['future']
    return dict(p=np.r_[centre['history_p'], np.asarray(future['p'])+plan.p],
        q=np.r_[centre['history_q'], np.asarray(future['q'])+plan.q],
        source_p=np.r_[centre['history_source_p'], future['source_p']],
        source_q=np.r_[centre['history_source_q'], future['source_q']],
        root_v=np.r_[centre['history_root_v'], future['root_v']])


def means_observations(physical, meter):
    values = {key: physical[key]+getattr(meter, 'bias_'+key) for key in ('p', 'q', 'v')}
    values.update(root_v=physical['root_v'])
    for key in ('master_p', 'master_q', 'source_p', 'source_q'):
        values[key] = None if getattr(meter, key+'_sigma') is None else physical[key]
    return likelihood.Observations(**values)



def time_major_residual(physical, observed, meter):
    blocks = []
    for channel in ('p', 'q', 'v', 'master_p', 'master_q', 'source_p', 'source_q'):
        data = getattr(observed, channel)
        if data is None: continue
        mean = np.asarray(physical[channel])
        if channel in ('p', 'q', 'v'): mean = mean+getattr(meter, 'bias_'+channel)
        sigma = np.broadcast_to(getattr(meter, channel+'_sigma'), data.shape)
        blocks.append(((mean-data)/sigma).reshape(observed.t, -1))
    return np.concatenate(blocks, axis=1).ravel()


def common_point(input_record, states):
    return np.r_[input_record['star_r'], input_record['star_x'],
        np.c_[states['p'], states['q'], states['source_p'], states['source_q']].ravel()]


def common_bounds(point):
    m = 4
    lower = np.r_[np.full(2*m, 1e-6), np.tile(np.r_[np.full(6, -1.), -.5, -.5], 16)]
    upper = np.r_[np.full(2*m, .1), np.tile(np.r_[np.full(6, 1.), .5, .5], 16)]
    if point.shape != lower.shape: raise ValueError('common coordinates must be eight shared lines plus 16x8 latent states')
    if np.any(point < lower-1e-12) or np.any(point > upper+1e-12): raise ValueError('common H-only point outside original box')
    return lower, upper


def boundary_record(point, lower, upper):
    active = []
    for side, mask, bound in (('lower', abs(point-lower) <= 1e-10, lower), ('upper', abs(point-upper) <= 1e-10, upper)):
        for index in np.flatnonzero(mask):
            active.append(dict(index=int(index), side=side, value=float(point[index]), bound=float(bound[index]),
                coordinate='shared_R_X' if index < 8 else 'time_local_P_Q_same_junction_source'))
    return active


def geometry(input_record, states, meter, counter):
    obs = likelihood.Observations(p=np.zeros((16, 3)), q=np.zeros((16, 3)), v=np.ones((16, 3)),
        root_v=states['root_v'], master_p=np.zeros(16), master_q=np.zeros(16))
    star = input_record['star']
    fit = SimpleNamespace(r=input_record['star_r'], x=input_record['star_x'], **{k: states[k] for k in ('p', 'q', 'source_p', 'source_q')})
    original = information_context.ac_predict
    information_context.ac_predict = counter
    try:
        b, star_physical = history_white_jacobian(star, fit, obs, meter)
        targets, mappings = {}, []
        for centre in input_record['centres']:
            candidate = input_record['candidates'][centre['name']]
            fit = SimpleNamespace(r=np.asarray(centre['r']), x=np.asarray(centre['x']),
                **{key: states[key] for key in ('p', 'q', 'source_p', 'source_q')})
            full, physical = history_white_jacobian(candidate, fit, obs, meter)
            edge = candidate.supports.index(tuple(centre['target_support']))
            target = (edge, candidate.m+edge)
            mapped = [candidate.supports.index(s) for s in star.supports]
            columns = mapped+[candidate.m+j for j in mapped]+list(range(2*candidate.m, full.shape[1]))
            nuisance = full[:, columns]
            difference = float(np.max(abs(nuisance-b)))
            means_difference = max(float(np.max(abs(physical[key]-star_physical[key]))) for key in ('v', 'master_p', 'master_q'))
            same = bool(np.allclose(nuisance, b, atol=1e-8, rtol=1e-11) and means_difference <= 2e-12)
            mappings.append(dict(name=candidate.name, target_columns=target, star_common_columns=columns,
                common_B_max_absolute_error=difference, physical_mean_max_error=means_difference, same_common_nuisance=same))
            if not same: raise ValueError('candidate zero-edge mapping changes common B or physical means')
            targets[candidate.name] = full[:, target]
    finally:
        information_context.ac_predict = original
    joined = np.column_stack([targets[name] for name in sorted(targets)])
    projected = project_columns(joined, b, rcond=1e-10, rank_guard_factor=3.)
    scale = np.linalg.norm(b, axis=0); scale[scale == 0] = 1.
    u, singular, vt = np.linalg.svd(b/scale, full_matrices=False)
    rank = int(projected['rank'])
    inverse = ((vt[:rank].T/singular[:rank])@u[:, :rank].T)/scale[:, None]
    independent_projected = joined-b@(inverse@joined)
    projection_error = float(np.max(abs(independent_projected-projected['projected'])))
    if not np.allclose(independent_projected, projected['projected'], atol=1e-8, rtol=1e-7):
        raise ValueError('SVD projection versus explicit numerical nuisance compensation disagrees')
    projected_targets = {name: projected['projected'][:, 2*i:2*i+2] for i, name in enumerate(sorted(targets))}
    projected_targets['star'] = np.zeros((len(b), 2))
    targets['star'] = np.zeros((len(b), 2))
    point = common_point(input_record, states)
    lower, upper = common_bounds(point)
    active = boundary_record(point, lower, upper)
    return dict(B=b, A=targets, projected_A=projected_targets, B_pseudoinverse=inverse,
        B_rank=rank, B_columns=b.shape[1], rows=b.shape[0], singular_values=singular,
        rank_cutoff=projected['rank_cutoff'], projection_rcond=1e-10,
        rank_guard_factor=3., projection_error=projection_error,
        normalized_nuisance_orthogonality_error=projected['nuisance_orthogonality_error'],
        zero_edge_mapping=mappings, common_point=point, common_lower=lower, common_upper=upper,
        common_active_boundaries=active, B_identity='same star shared R/X and H/N time-local P/Q/source derivatives',
        free_projection_identity='relaxation_of_original_one_sided_competitor_tangent' if active else 'original_interior_nuisance_tangent')


def closure_qp(g, source, competitor, zeta):
    result = pairwise_cone_distance(g['projected_A'][source], g['projected_A'][competitor],
        np.empty((g['rows'], 0)), zeta)
    eta = np.asarray(result['eta'])
    target = g['A'][source]@zeta
    compensation = g['B_pseudoinverse']@(target-g['A'][competitor]@eta)
    residual = target-g['A'][competitor]@eta-g['B']@compensation
    kkt_c = g['A'][competitor].T@residual
    kkt_b = g['B'].T@residual
    y = g['projected_A'][source]@zeta
    eta_oracle, norm = nnls(g['projected_A'][competitor], y)
    cost = float(residual@residual)
    result.update(full_matrix_primal_cost_numeric=cost, common_competitor_step=compensation,
        full_matrix_KKT=dict(competitor_positive_violation=float(max(0., np.max(kkt_c))),
            nuisance_dot_inf=float(np.max(abs(kkt_b))), complementarity_inf=float(np.max(abs(eta*kkt_c)))),
        independent_nnls_cost=float(norm**2), independent_nnls_eta=eta_oracle,
        exact_certificate_identity='exact binary-rational projected two-column problem; SVD whitening/projection is numerical',
        full_original_matrix_unconditional_lower_bound=0.,
        full_original_matrix_positive_rigorous_lower_bound=None,
        relaxation_of_original_tangent=bool(g['common_active_boundaries']))
    result['numeric_KKT_passed'] = bool(np.all(eta >= 0) and max(0., np.max(kkt_c)) <= 1e-6 and
        np.max(abs(kkt_b)) <= 1e-6 and np.max(abs(eta*kkt_c)) <= 1e-7 and
        np.isclose(norm**2, result['distance_squared'], atol=1e-8, rtol=1e-7))
    return result


def bounded_local_qp(g, source, competitor, zeta, *, tangent=False):
    has_edge = competitor != 'star'
    ar = g['A'][competitor][:, :2] if has_edge else np.empty((g['rows'], 0))
    matrix = np.c_[ar, g['B']]
    target = g['A'][source]@zeta
    lower = g['common_lower']-g['common_point']
    upper = g['common_upper']-g['common_point']
    if tangent:
        active_low, active_high = abs(lower) <= 1e-10, abs(upper) <= 1e-10
        lower = np.where(active_low, 0., -np.inf)
        upper = np.where(active_high, 0., np.inf)
    lower = np.r_[np.full(2, 0. if tangent else 1e-6) if has_edge else [], lower]
    upper = np.r_[np.full(2, np.inf if tangent else .1) if has_edge else [], upper]
    scale = np.r_[np.full(2, .01) if has_edge else [], np.full(8, .01), np.tile(np.r_[np.full(6, .001), .01, .01], 16)]
    solved = lsq_linear(matrix*scale, target, bounds=(lower/scale, upper/scale),
        method='bvls', lsq_solver='exact', tol=1e-12, max_iter=300)
    step = solved.x*scale
    residual = matrix@step-target
    cost = float(residual@residual)
    gradient = 2*matrix.T@residual
    mask = solved.active_mask
    kkt = gradient.copy()
    kkt[mask == -1] = np.minimum(kkt[mask == -1], 0.)
    kkt[mask == 1] = np.maximum(kkt[mask == 1], 0.)
    tolerance = 1e-5
    if tangent:
        numeric_lower, gap = None, None
    else:
        corner = np.where(gradient >= 0, lower, upper)
        minorant = cost+float(gradient@(corner-step))
        numeric_lower = min(cost, max(0., minorant))
        gap = cost-numeric_lower
    return dict(success=bool(solved.success and np.all(step >= lower-1e-10) and np.all(step <= upper+1e-10)),
        solver_success=bool(solved.success), status=int(solved.status), message=str(solved.message),
        iterations=int(solved.nit), optimality=float(solved.optimality), cost=cost,
        eta=step[:2] if has_edge else np.zeros(2), common_competitor_step=step[2:] if has_edge else step,
        active_mask=mask, residual=residual, KKT_inf=float(np.max(abs(kkt))), numeric_KKT_passed=bool(np.max(abs(kkt)) <= tolerance),
        lower=lower, upper=upper, coordinate_scale=scale, numeric_tangent_minorant_lower_bound=numeric_lower,
        numeric_primal_lower_gap=gap, unconditional_rigorous_lower_bound=0.,
        bound_identity='floating-point convex finite-box tangent minorant; numerical diagnostic, not interval certificate' if not tangent else 'numerical original one-sided competitor tangent; no finite-box dual claim',
        domain='original_active_tangent' if tangent else 'original_finite_positive_box',
        global_AC_lower_bound=None)


def competitor_initial(input_record, states, competitor, qp=None):
    common = common_point(input_record, states)
    eta = np.full(2, 1e-6)
    if qp is not None:
        common = common+np.asarray(qp['common_competitor_step'])
        eta = np.asarray(qp['eta'])
    candidate = input_record['star'] if competitor == 'star' else input_record['candidates'][competitor]
    r, x = np.zeros(candidate.m), np.zeros(candidate.m)
    for index, support in enumerate(candidate.supports):
        if support in input_record['star'].supports:
            j = input_record['star'].supports.index(support)
            r[index], x[index] = common[j], common[4+j]
        else:
            r[index], x[index] = eta
    latent = common[8:].reshape(16, 8)
    return candidate, dict(r=r, x=x, p=latent[:, :3], q=latent[:, 3:6], source_p=latent[:, 6], source_q=latent[:, 7])


def point_record(candidate, point, meter):
    return dict(name=candidate.name, supports=candidate.supports, **point,
        **{key: np.broadcast_to(getattr(meter, key), (3,)) for key in ('bias_p', 'bias_q', 'bias_v')})



def checked_replay(candidate, point, observations, meter):
    if not all(np.isfinite(np.asarray(point.get(key), float)).all() for key in ('r', 'x', 'p', 'q', 'source_p', 'source_q')):
        return dict(success=False, replayable=False, errors=['nonfinite_failed_state_retained'], failure_state_retained=True)
    try:
        return independent_replay(candidate, point, observations, meter)
    except (ValueError, RuntimeError, TypeError, KeyError, FloatingPointError, np.linalg.LinAlgError) as error:
        return dict(success=False, replayable=False, errors=['replay_exception:'+type(error).__name__+':'+str(error)],
                    failure_state_retained=True)


def safe_fit(candidate, observed, meter, *, initial, mode, seed):
    # The original instrumentation retains every restart. Make its independent
    # replay exception-safe so an invalid/NaN failed fit cannot erase that record
    # or prevent the second predeclared start.
    original_replay = recovery_module.independent_replay
    recovery_module.independent_replay = checked_replay
    try:
        return run_fit(candidate, observed, meter, initial=initial, mode=mode,
            seed=seed, max_nfev=200, multistart=1)
    finally:
        recovery_module.independent_replay = original_replay


def planned_tasks(names):
    return [dict(id=f'{action}_{source}_to_{competitor}_{direction}_e{int(100*epsilon)}', action=action,
        source=source, competitor=competitor, direction=direction, epsilon=epsilon)
        for action in ('constant0', 'alternating0') for source in names
        for competitor in [name for name in names if name != source]+['star']
        for direction in DIRECTIONS for epsilon in SCALES]


def run(output=None, *, history_id=None, inputs_from_evidence=None):
    output = Path(output or HERE/'evidence/structured_information/pairwise_cone_ac_scale/H930001')
    if (output/'summary.json').exists(): raise ValueError('completed evidence exists; use a fresh output')
    freeze = read_json(FROZEN/'freeze.json')
    original_manifest = immutable_manifest(FROZEN)
    original_sources = audit_source(freeze['source_sha256'], snapshot=FROZEN/'science_source_snapshot.json.gz')
    if not original_sources['success'] or original_sources['checked_count'] != 19: raise ValueError('frozen science19 bytes differ')
    input_record = (select_saved_input(history_id=history_id) if inputs_from_evidence is None
                    else load_compact_inputs(inputs_from_evidence, history_id=history_id))
    tasks = planned_tasks(sorted(input_record['candidates']))
    if len(tasks) != 108: raise ValueError('exact 108-task scope required')
    protocol = dict(stage='gate_2_explicit_saved_pilot_H_only_AC_scale', created_at_UTC=datetime.now(timezone.utc).isoformat(),
        required_tasks=108, planned_tasks=tasks, selected_input=input_record['path'], input_sha256=input_record['input_sha256'],
        selected_history=input_record['history_id'], selected_anchor=input_record['anchor_id'],
        input_selection=('explicit archived compact evidence; original filename selection not rerun'
            if inputs_from_evidence is not None else 'explicit history id when given; first filename of that H; first common anchor id; no truth selection'),
        input_mode=input_record['input_mode'], inputs_from_evidence=None if inputs_from_evidence is None else str(inputs_from_evidence),
        requested_history_id=history_id, unique_H_order_position=input_record['unique_H_order_position'],
        unique_saved_H_ids=input_record['unique_saved_H_ids'],
        source_fixed=True, competitor_only_refitted=True, actual_future_truth_or_noise_used=False,
        adopted_residual_sign='competitor minus fixed source = A_r eta + B w - A_s zeta',
        canonical_mapping='canonical delta_nu = -w; lower-bound competitor X gives w_X>=0 and canonical delta_nu_X<=0',
        directions_pu=DIRECTIONS, scales=SCALES, fixed_actions='node0 constant and original positive alternating operation seed',
        max_nfev_per_start=200, required_starts=2, multistart_per_explicit_initial=1,
        explicit_initials=['original_finite_box_local_QP_witness', 'common_H_only_center_with_target_at_positive_lower_bound'],
        tol=1e-10, linear_solver='exact', r_x_bounds=[1e-6, .1], P_Q_bounds=[-1., 1.], source_bounds=[-.5, .5],
        voltage_bounds=[.85, 1.15], rank_rcond=1e-10, rank_guard_factor=3., QP_KKT_tolerance=1e-5,
        scale_ratio_reference_band=[.85, 1.15], scale_ratio_band_is_diagnostic_only=True,
        failure_policy='retain every task and both fixed starts; no budget expansion, H replacement or failed-pair deletion',
        exact_QP_scope='two-column projected floating arrays treated as binary-rational data; full B projection remains numerical',
        frozen_sources_before=original_sources, immutable_original_manifest=original_manifest,
        new_source_sha256={name: sha(HERE/name) for name in ('pairwise_cone_ac_scale.py', 'test_pairwise_cone_ac_scale.py')},
        supporting_sources_sha256={name: sha(HERE/name) for name in ('pairwise_cone_checks.py', 'information_fit_recovery.py', 'information_geometry_audit.py', 'structured_information_ac_audit.py')})
    write_json(output/'protocol.json', protocol)
    write_json(output/'config.json', protocol)
    began = perf_counter()
    ac_counts = dict(attempted=0, success=0, failure=0)
    original_ac = likelihood.ac_predict
    def counted(*args, **kwargs):
        ac_counts['attempted'] += 1
        try: physical = original_ac(*args, **kwargs)
        except Exception:
            ac_counts['failure'] += 1
            raise
        ac_counts['success'] += 1
        return physical
    records, errors = [], []
    try:
        constraints = SingleSiteConstraints(**input_record['device_constraints'])
        actions = fixed_actions(constraints)
        meter = meter_from_record(input_record['models']['joint'])
        centres = {row['name']: row for row in input_record['centres']}
        action_geometries, state_records = {}, {}
        for action, info in actions.items():
            states = total_states(input_record['centres'][0], info['plan'])
            state_records[action] = states
            action_geometries[action] = geometry(input_record, states, meter, counted)
            write_json(output/('geometry_'+action+'.json.gz'), action_geometries[action])
        write_json(output/'inputs.json.gz', {key: value for key, value in input_record.items() if key not in ('star', 'candidates')} |
            dict(actions={name: dict(plan=row['plan'].summary(), audit=row['audit'], operation=row['operation']) for name, row in actions.items()},
                 total_H_N_states=state_records, actual_joint_meter=meter_to_record(meter)))
        for index, task in enumerate(tasks):
            started, before = perf_counter(), ac_counts['attempted']
            row = dict(**task, success=False, fit_attempts=[], source_side_fixed=True, expected_object='fixed source task versus single refitted competitor',
                source_truth_not_used=True, global_AC_lower_bound=None, global_AC_optimality_gap=None)
            records.append(row)
            try:
                g = action_geometries[task['action']]
                states = state_records[task['action']]
                direction = DIRECTIONS[task['direction']]
                epsilon = task['epsilon']
                zeta = epsilon*direction
                if np.any(zeta < 1e-6) or np.any(zeta > .1):
                    row.update(failure='source_engineering_edge_outside_original_domain', scale_applicable=False)
                    continue
                centre = centres[task['source']]
                source = input_record['candidates'][task['source']]
                edge = source.supports.index(tuple(centre['target_support']))
                source_r, source_x = np.asarray(centre['r']).copy(), np.asarray(centre['x']).copy()
                source_r[edge], source_x[edge] = zeta
                source_point = dict(r=source_r, x=source_x, **{key: states[key] for key in ('p', 'q', 'source_p', 'source_q')})
                physical = counted(source, **source_point, root_v=states['root_v'])
                observed = means_observations(physical, meter)
                source_row = point_record(source, source_point, meter)
                source_row.update(loss=0.)
                source_replay = independent_replay(source, source_row, observed, meter)
                if not source_replay['success']: raise ValueError('source mean physical replay failed')
                if source_replay['voltage_min'] < .9 or source_replay['voltage_max'] > 1.1: raise ValueError('source mean outside operational voltage domain')
                closure_unit = closure_qp(g, task['source'], task['competitor'], direction)
                closure = closure_qp(g, task['source'], task['competitor'], zeta)
                tangent = bounded_local_qp(g, task['source'], task['competitor'], zeta, tangent=True)
                finite = bounded_local_qp(g, task['source'], task['competitor'], zeta)
                candidate, projected_initial = competitor_initial(input_record, states, task['competitor'], finite)
                candidate, base_initial = competitor_initial(input_record, states, task['competitor'])
                finite_witness = point_record(candidate, projected_initial, meter)
                finite_replay = checked_replay(candidate, finite_witness, observed, meter)
                finite_taylor = dict(success=False, failure='finite physical witness unavailable')
                if finite_replay['success']:
                    witness_physical = counted(candidate, **projected_initial, root_v=states['root_v'])
                    actual_vector = time_major_residual(witness_physical, observed, meter)
                    linear_vector = np.asarray(finite['residual'])
                    remainder = actual_vector-linear_vector
                    finite_taylor = dict(success=True, residual_sign='competitor minus fixed source',
                        actual_time_major_AC_residual=actual_vector, linear_time_major_residual=linear_vector,
                        remainder_vector=remainder, remainder_l2=float(np.linalg.norm(remainder)),
                        remainder_l2_over_epsilon_squared=float(np.linalg.norm(remainder)/epsilon**2),
                        actual_SSE=float(actual_vector@actual_vector), linear_SSE=float(linear_vector@linear_vector),
                        common_competitor_step_over_epsilon=finite['common_competitor_step']/epsilon,
                        eta_over_epsilon=finite['eta']/epsilon, step_norm_over_epsilon=float(np.linalg.norm(finite['common_competitor_step'])/epsilon),
                        ratio_finite_witness_AC_to_local=None if finite['cost'] < 1e-14 else finite_replay['full_SSE']/finite['cost'],
                        uniform_remainder_or_near_optimal_sequence_certified=False)
                row.update(scale_applicable=True, direction_pu=direction, zeta_pu=zeta,
                    common_active_boundaries=g['common_active_boundaries'], free_projection_identity=g['free_projection_identity'],
                    closure_QP=closure, closure_unit_QP=closure_unit,
                    original_one_sided_tangent_QP=tangent, original_finite_box_QP=finite,
                    source_fixed_point=source_row, source_independent_replay=source_replay,
                    mean_observations={key: getattr(observed, key) for key in ('p', 'q', 'v', 'root_v', 'master_p', 'master_q', 'source_p', 'source_q')},
                    finite_QP_AC_witness=finite_witness, finite_QP_witness_replay=finite_replay, finite_QP_Taylor_replay=finite_taylor,
                    finite_QP_common_step_over_epsilon=finite['common_competitor_step']/epsilon,
                    finite_QP_eta_over_epsilon=finite['eta']/epsilon,
                    finite_QP_step_norm_over_epsilon=float(np.linalg.norm(finite['common_competitor_step'])/epsilon),
                    witness_is_original_domain_feasible=finite_replay['success'],
                    numerical_geometry_validation_passed=closure['numeric_KKT_passed'] and finite['numeric_KKT_passed'] and tangent['numeric_KKT_passed'],
                    closure_homogeneity_absolute_error=abs(closure['distance_squared']-epsilon**2*closure_unit['distance_squared']))
                if not finite['success']:
                    row['failure_stage'] = 'finite_box_local_QP'
                    row['failure'] = 'finite local QP did not return a validated numerical optimum'
                if not finite_replay['success']:
                    row['failure_stage'] = 'finite_QP_witness_original_domain_or_physical'
                    row['failure'] = 'finite local compensation witness failed original domain/physical replay'
                legal_finite_costs = [finite_replay['full_SSE']] if finite_replay['success'] else []
                attempts = []
                for initial_name, initial in (('finite_local_witness', projected_initial), ('H_only_base', base_initial)):
                    fit_before = ac_counts['attempted']
                    saved_outer = likelihood.ac_predict
                    likelihood.ac_predict = counted
                    try:
                        if initial_name == 'finite_local_witness' and not finite_replay['success']:
                            result = dict(candidate=candidate.name, mode=initial_name, initial=initial, starts=[],
                                error='predeclared finite initial violates original physical domain; retained without illegal fit start',
                                returned_best_fit=None, returned_best_fit_replay=None, attempted_AC_calls=dict(attempted=0),
                                all_replays_passed=False, fixed_start_failure_retained=True, failure_stage='initial_domain')
                        else:
                            result = safe_fit(candidate, observed, meter, initial=initial, mode=initial_name,
                                seed=input_record['policy_seed']+1000+2*index+len(attempts))
                    except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError) as error:
                        result = dict(candidate=candidate.name, mode=initial_name, initial=initial, starts=[],
                            error=type(error).__name__+':'+str(error), returned_best_fit=None, returned_best_fit_replay=None,
                            attempted_AC_calls=dict(attempted=ac_counts['attempted']-fit_before),
                            all_replays_passed=False, fixed_start_failure_retained=True)
                    finally:
                        likelihood.ac_predict = saved_outer
                    result['outer_attempted_AC_calls'] = ac_counts['attempted']-fit_before
                    if result['outer_attempted_AC_calls'] != result['attempted_AC_calls']['attempted']:
                        raise ValueError('fit actual attempted AC count mismatch')
                    attempts.append(result)
                row['fit_attempts'] = attempts
                accepted = [result['returned_best_fit'] for result in attempts if result['returned_best_fit'] is not None and
                    result['returned_best_fit']['success'] and result['returned_best_fit_replay'] and result['returned_best_fit_replay']['success']]
                finite_fits = [result['returned_best_fit'] for result in attempts if result['returned_best_fit'] is not None and np.isfinite(result['returned_best_fit']['loss'])]
                row['both_required_starts_successful'] = all(result['returned_best_fit'] and result['returned_best_fit']['success'] for result in attempts)
                row['best_finite_solver_cost'] = min((fit['loss'] for fit in finite_fits), default=None)
                legal_solver_witnesses = [result['returned_best_fit']['loss'] for result in attempts
                    if result['returned_best_fit'] is not None and np.isfinite(result['returned_best_fit']['loss']) and
                    result['returned_best_fit_replay'] is not None and result['returned_best_fit_replay']['success']]
                row['attained_AC_witness_upper_bound'] = min(legal_finite_costs+legal_solver_witnesses, default=None)
                row['legal_solver_witness_count'] = len(legal_solver_witnesses)
                if accepted:
                    best = min(accepted, key=lambda fit: fit['loss'])
                    achieved = best['loss']
                    closure_prediction = epsilon**2*closure_unit['distance_squared']
                    best_common = np.r_[np.array([best['r'][candidate.supports.index(s)] for s in input_record['star'].supports]),
                        np.array([best['x'][candidate.supports.index(s)] for s in input_record['star'].supports]),
                        np.c_[best['p'], best['q'], best['source_p'], best['source_q']].ravel()]
                    delta = best_common-g['common_point']
                    row.update(success=bool(row['numerical_geometry_validation_passed'] and finite_replay['success'] and finite['success']), best_successful_local_AC_fit=best, attained_successful_local_AC_cost=achieved,
                        common_shared_compensation_norm_over_epsilon=float(np.linalg.norm(delta[:8])/epsilon),
                        time_local_compensation_norm_over_epsilon=float(np.linalg.norm(delta[8:])/epsilon),
                        finite_QP_compensation_norm_over_epsilon=float(np.linalg.norm(finite['common_competitor_step'])/epsilon),
                        finite_QP_eta_over_epsilon=finite['eta']/epsilon,
                        ratio_AC_to_epsilon_squared_closure=None if closure_prediction < 1e-14 else achieved/closure_prediction,
                        ratio_AC_to_finite_box_local_value=None if finite['cost'] < 1e-14 else achieved/finite['cost'],
                        ratio_AC_to_original_tangent=None if tangent['cost'] < 1e-14 else achieved/tangent['cost'],
                        original_tangent_cost_over_epsilon_squared=tangent['cost']/epsilon**2,
                        AC_cost_over_epsilon_squared=achieved/epsilon**2,
                        finite_local_cost_over_epsilon_squared=finite['cost']/epsilon**2,
                        closure_local_cost_over_epsilon_squared=closure['distance_squared']/epsilon**2)
                else:
                    row['failure'] = 'both_fixed_max200_local_AC_starts_failed; all available legal finite witnesses remain saved'
                    row['failure_stage'] = 'optimizer_budget_or_numerical'
            except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError) as error:
                row['failure'] = type(error).__name__+':'+str(error)
                row['failure_stage'] = row.get('failure_stage', 'source_or_local_geometry_physical')
            finally:
                row.update(actual_attempted_AC_calls=ac_counts['attempted']-before, seconds=perf_counter()-started)
                write_json(output/'tasks'/(task['id']+'.json.gz'), row)
                if not row['success']: errors.append(dict(task=task['id'], failure=row.get('failure')))
                write_json(output/'progress.json', dict(completed=len(records), planned=108, failed=len(errors), last_task=task['id']))
                if (index+1) % 18 == 0: print(dict(completed=index+1, failed=len(errors), AC_calls=ac_counts['attempted']), flush=True)
        status = 'all108_tasks_completed_with_failures_retained'
    except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError) as error:
        status = 'object_or_setup_failure'
        errors.append(dict(stage='setup', failure=type(error).__name__+':'+str(error)))
    sources_after = audit_source(freeze['source_sha256'], snapshot=FROZEN/'science_source_snapshot.json.gz')
    summary = dict(stage='gate2_fixed_H_scale_only', status=status, planned=108, completed=len(records),
        successful=sum(row['success'] for row in records), failures=errors,
        necessary_start_count=sum(len(row['fit_attempts']) for row in records),
        failed_fit_starts=sum(not (result.get('returned_best_fit') and result['returned_best_fit']['success']) for row in records for result in row['fit_attempts']),
        all108_tasks_retained=len(records) == 108, no_budget_expansion=True, no_selector_or_formal_run=True,
        active_bound_identity='free B distance is a relaxation; original one-sided tangent and finite positive box separately saved',
        sources19_unchanged=original_sources == sources_after,
        original_frozen119_records_and_existing_evidence_unchanged=original_manifest == immutable_manifest(FROZEN),
        actual_AC_counts=ac_counts, seconds=perf_counter()-began,
        AC_global_distance_lower_bound=None, original_lower_bound_preserved=1e-6,
        input_file=input_record['path'], input_sha256=input_record['input_sha256'],
        history_id=input_record['history_id'], unique_H_order_position=input_record['unique_H_order_position'],
        positive_QP_bounds_scope='exact projected two-column binary-rational problem only; SVD and full finite-box minorants are numerical',
        local_scale_acceptance_not_inferred_from_solver_success=True)
    write_json(output/'summary.json', summary)
    return summary


def independent_saved_replay(output):
    output = Path(output)
    inputs = read_json(output/'inputs.json.gz')
    meter = meter_from_record(inputs['actual_joint_meter'])
    audits, task_checks = [], []
    for path in sorted((output/'tasks').glob('*.json.gz')):
        row = read_json(path)
        if 'mean_observations' not in row:
            task_checks.append(dict(task=row['id'], replayable=False, original_failure=row.get('failure'), all_passed=False, retained_required_stage_failure=True))
            continue
        obs = _obs(row['mean_observations'])
        saved = [('source', row['source_fixed_point']), ('finite_QP_witness', row['finite_QP_AC_witness'])]
        for i, attempt in enumerate(row['fit_attempts']):
            for start in attempt['starts']:
                for key in ('initial', 'returned', 'last_finite_state'):
                    if key in start: saved.append((f'start{i}_{key}', start[key]))
            if attempt['returned_best_fit'] is not None: saved.append((f'fit{i}_best', attempt['returned_best_fit']))
        local = []
        for kind, point in saved:
            candidate = exact_candidate(point, 3)
            audit = checked_replay(candidate, point, obs, meter)
            item = dict(task=row['id'], kind=kind, replay=audit)
            audits.append(item); local.append(audit)
        task_checks.append(dict(task=row['id'], states=len(local), all_passed=all(a['success'] for a in local)))
    protocol = read_json(output/'protocol.json')
    planned_keys = {row['id'] for row in protocol['planned_tasks']}
    actual_keys = {row['task'] for row in task_checks}
    coverage = len(task_checks) == 108 and len(actual_keys) == 108 and planned_keys == actual_keys
    result = dict(success=coverage and bool(audits) and all(row.get('all_passed', False) for row in task_checks),
        all_available_saved_states_passed=bool(audits) and all(a['replay']['success'] for a in audits),
        planned_task_count=108, actual_unique_task_count=len(actual_keys), task_key_coverage_passed=coverage,
        planned_task_keys=sorted(planned_keys), actual_task_keys=sorted(actual_keys),
        missing_task_keys=sorted(planned_keys-actual_keys), unexpected_task_keys=sorted(actual_keys-planned_keys),
        retained_required_stage_failure_count=sum(row.get('retained_required_stage_failure', False) for row in task_checks),
        recorded_task_count=len(task_checks),
        actual_saved_state_replay_count=len(audits),
        source_and_finite_witness_replay_count=sum(row['kind'] in ('source', 'finite_QP_witness') for row in audits),
        source_and_finite_witness_failure_count=sum(row['kind'] in ('source', 'finite_QP_witness') and not row['replay']['success'] for row in audits),
        failed_or_invalid_returned_state_count=sum(row['kind'] not in ('source', 'finite_QP_witness') and not row['replay']['success'] for row in audits),
        optimizer_invoked=False, audits=audits, tasks=task_checks)
    write_json(output/'independent_replay.json.gz', result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=HERE/'evidence/structured_information/pairwise_cone_ac_scale/H930001')
    parser.add_argument('--replay-only', action='store_true')
    parser.add_argument('--history-id', help='saved independent H id; first filename within that H is used')
    parser.add_argument('--inputs-from-evidence', type=Path, help='explicit committed inputs.json.gz; no raw-pilot selection fallback')
    args = parser.parse_args()
    if not args.replay_only:
        summary = run(args.output, history_id=args.history_id, inputs_from_evidence=args.inputs_from_evidence)
        print(summary, flush=True)
    if args.replay_only and args.inputs_from_evidence is not None:
        restored = load_compact_inputs(args.inputs_from_evidence, history_id=args.history_id)
        print(dict(input_mode=restored['input_mode'], history_id=restored['history_id'], compact_inputs_validated=True), flush=True)
    replay = independent_saved_replay(args.output)
    print(dict(replay_passed=replay['success'], tasks=replay['recorded_task_count'], saved_states=replay['actual_saved_state_replay_count']), flush=True)


if __name__ == '__main__': main()
