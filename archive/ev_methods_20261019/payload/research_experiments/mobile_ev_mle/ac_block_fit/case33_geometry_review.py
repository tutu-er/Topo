"""Replay retained case33 states and one predeclared rejected local step.

This module performs arithmetic/AC replay only.  It never optimizes a new
history or searches a step size.  The one local QP is the exact frozen
lambda=1e-4 first polish proposal, reconstructed for diagnostics.
"""
from pathlib import Path
import argparse
import numpy as np

from . import benchmark as b
from . import endpoint_polish_audit as e
from .blocks import problem_blocks, column_norms, matvec
from .linear import solve_block_step
from ..reference_benchmark.derivatives import fit_response_derivatives


def _norm(value):
    return float(np.linalg.norm(value))


def physical_blocks(problem, vector):
    problem.evaluate(vector)
    base = problem.base
    derivative = fit_response_derivatives(base.candidate, base.cache['physical'], include_lines=True)
    fields = []
    for key in ('v', 'master_p', 'master_q'):
        raw = derivative[key]
        sigma = np.asarray(base.sigmas[key])
        if key != 'v':
            raw, sigma = raw[:, None, :], sigma[:, None]
        fields.append(raw/sigma[:, :, None])
    full = np.concatenate(fields, axis=1)
    return full[:, :, :problem.shared], full[:, :, -2:]


def map_decomposition(problem, first, second):
    old, new = problem.expand(first), problem.expand(second)
    m = problem.m
    r, a = old[:m], old[m:2*m]
    rr, aa = new[:m], new[m:2*m]
    _, x, dxdr, width = problem.mapping.forward(r, a)
    _, xx, _, next_width = problem.mapping.forward(rr, aa)
    dr, da = rr-r, aa-a
    _, _, _, lp, hp = problem.mapping.section(r)
    linear_dx = dxdr*dr+width*da
    same_piece_cross = (hp-lp)*dr*da
    switch_low = (r >= problem.mapping.low_switch) != (rr >= problem.mapping.low_switch)
    switch_high = (r < problem.mapping.high_switch) != (rr < problem.mapping.high_switch)
    mapping_remainder = xx-x-linear_dx
    return dict(delta_r=dr, delta_alpha=da, delta_x=xx-x, linear_delta_x=linear_dx,
        mapping_remainder=mapping_remainder, bilinear_same_piece_remainder=same_piece_cross,
        switching_remainder=mapping_remainder-same_piece_cross,
        low_switch_crossed=switch_low, high_switch_crossed=switch_high,
        section_width=width, next_section_width=next_width,
        max_abs_delta_r=float(np.max(abs(dr), initial=0.)),
        max_abs_delta_alpha=float(np.max(abs(da), initial=0.)),
        map_remainder_l2=_norm(mapping_remainder),
        bilinear_same_piece_remainder_l2=_norm(same_piece_cross),
        switching_remainder_l2=_norm(mapping_remainder-same_piece_cross))


def state_geometry(problem, first, second):
    residual = problem.evaluate(first).copy()
    shared, local = physical_blocks(problem, first)
    r, x, s = problem.unpack(first)
    rr, xx, ss = problem.unpack(second)
    line_change = np.einsum('tkd,d->tk', shared, np.r_[rr-r, xx-x])
    source_change = np.einsum('tks,ts->tk', local, ss-s)
    actual = (problem.evaluate(second)-residual).reshape(local.shape[:2])
    profiled = line_change.copy()
    for t in range(problem.t):
        q, singular, _ = np.linalg.svd(local[t], full_matrices=False)
        rank = np.count_nonzero(singular > np.finfo(float).eps*max(local[t].shape)*singular[0])
        profiled[t] -= q[:, :rank]@(q[:, :rank].T@profiled[t])
    mapping = map_decomposition(problem, first, second)
    map_effect = np.einsum('tkd,d->tk', shared[:, :, problem.m:], mapping['mapping_remainder'])
    _, _, width, _, _ = problem.mapping.section(r)
    active = problem.mapping.constraint_record(r, x)
    return dict(physical_delta_RX_l2=_norm(np.r_[rr-r, xx-x]),
        physical_delta_source_l2=_norm(ss-s), line_effect_l2=_norm(line_change),
        source_effect_l2=_norm(source_change), linear_sum_l2=_norm(line_change+source_change),
        exact_residual_change_l2=_norm(actual),
        physical_AC_nonlinear_remainder_l2=_norm(actual-line_change-source_change),
        source_compensation_cosine=float(np.vdot(line_change, source_change)/max(1e-300,_norm(line_change)*_norm(source_change))),
        unconstrained_source_profiled_line_effect_l2=_norm(profiled),
        source_projection_is_finite_difference_identifiability_proof=False,
        source_projection_scope='local tangent source span, unconstrained; large finite displacement has nonlinear remainder',
        map=mapping, map_remainder_whitened_effect_l2=_norm(map_effect),
        section_width_min=float(width.min()), section_width_max=float(width.max()),
        collapsed_sections=int(np.count_nonzero(width <= 1e-12)),
        alpha_lower_active=int(np.count_nonzero(problem.expand(first)[problem.m:problem.shared] <= 1e-9)),
        alpha_upper_active=int(np.count_nonzero(problem.expand(first)[problem.m:problem.shared] >= 1-1e-9)),
        physical_active_counts={key:int(np.count_nonzero(value)) for key,value in active['active'].items()})


def rejected_step(problem, vector, saved_trace):
    residual = problem.evaluate(vector).copy()
    entry_loss = .5*float(residual@residual)
    physical_shared, physical_source = physical_blocks(problem, vector)
    line, source = problem_blocks(problem, vector)
    scale = 1./np.where(column_norms(line,source)>0, column_norms(line,source), 1.)
    d = line.shape[-1]
    proposal = solve_block_step(residual.reshape(source.shape[:2]), line*scale[:d][None,None,:],
        source*scale[d:].reshape(problem.t,1,2),
        (problem.lower-vector)/scale, (problem.upper-vector)/scale,
        damping=1e-4, kkt_tol=1e-10, max_iter=2000)
    target = np.clip(vector+scale*np.asarray(proposal['step']), problem.lower, problem.upper)
    mapped = map_decomposition(problem, vector, target)
    delta = matvec(line,source,target-vector).ravel()
    predicted = -float(residual@delta)-.5*float(delta@delta)
    trial = problem.evaluate(target).copy()
    trial_loss = .5*float(trial@trial)
    shared_delta = np.einsum('tkd,d->tk',physical_shared,
        np.r_[mapped['delta_r'],mapped['delta_x']])
    r, x, ss = problem.unpack(vector)
    rr, xx, tt = problem.unpack(target)
    source_delta = np.einsum('tks,ts->tk',physical_source,tt-ss)
    map_effect = np.einsum('tkd,d->tk',physical_shared[:,:,problem.m:],mapped['mapping_remainder'])
    actual = (trial-residual).reshape(source.shape[:2])
    physical_remainder = actual-shared_delta-source_delta
    result = dict(diagnostic='single predeclared candidate00/reference first rejected lambda1e-4 proposal',
        new_fit=False, new_step_search=False, entry_half_SSE=entry_loss, trial_half_SSE=trial_loss,
        actual_F_change=trial_loss-entry_loss,predicted_linear_decrease=predicted,
        saved_first_trace=saved_trace,
        saved_trial_F_reproduced=bool(abs(trial_loss-saved_trace['trial_F']) <= 1e-9),
        inner_KKT=float(proposal['kkt']),inner_success=bool(proposal['success']),
        map=mapped, optimizer_linear_delta_l2=_norm(delta),
        exact_residual_change_l2=_norm(actual),
        Ralpha_map_remainder_effect_l2=_norm(map_effect),
        physical_AC_curvature_effect_l2=_norm(physical_remainder),
        total_residual_nonlinearity_l2=_norm(actual-delta.reshape(source.shape[:2])),
        decomposition_identity_max_abs=float(np.max(abs(actual-delta.reshape(source.shape[:2])-map_effect-physical_remainder))),
        physical_trial_domain=bool(problem.mapping.contains(rr,xx)))
    # One fixed alternate physical point: keep same R/source trial but use
    # the first-order X displacement.  Replay only if the original cone/box
    # contains it.  This is a diagnostic attribution, never an accepted fit.
    linear_x = x+mapped['linear_delta_x']
    feasible = problem.mapping.contains(rr,linear_x)
    result['physical_first_order_X_point_in_domain']=bool(feasible)
    result['first_order_X_point_is_new_optimizer_result']=False
    if feasible:
        from .. import likelihood as lk
        base=problem.base
        physical=lk.ac_predict(base.candidate,rr,linear_x,base.p,base.q,tt[:,0],tt[:,1],base.observations.root_v)
        ar=np.column_stack(((physical['v']+base.bias['v']-base.observations.v)/base.sigmas['v'],
            (physical['master_p']-base.observations.master_p)/base.sigmas['master_p'],
            (physical['master_q']-base.observations.master_q)/base.sigmas['master_q'])).ravel()
        result.update(first_order_X_point_half_SSE=.5*float(ar@ar),
            first_order_X_point_F_change=.5*float(ar@ar)-entry_loss,
            first_order_X_point_AC_equation_residual=float(physical['ac_equation_residual']),
            first_order_X_point_voltage_range=[float(abs(physical['voltage']).min()),float(abs(physical['voltage']).max())])
    else:
        result['first_order_X_point_loss']='not evaluated: outside original physical domain'
    return result


def run(stage, output):
    stage,output=Path(stage),Path(output)
    if output.exists():
        raise FileExistsError('retain previous arithmetic review '+str(output))
    old_plan=b._read(stage/'outputs/candidate_h24_v1/plan.json')
    cross_dir=stage/'outputs/case33_crosswarm_v1'
    cross=b._read(cross_dir/'crosswarm_plan.json')
    whitelist=Path(cross['historical_whitelist'])
    selected=[task for task in old_plan['tasks'] if task['network']=='case33' and
              task['candidate_name'] in ('controlled_case33_00','controlled_case33_01')]
    plan=dict(purpose='saved endpoint geometry and one predeclared rejected-step replay',
        created_utc=b._utc(),historical_whitelist=str(whitelist),future_used=False,truth_used=False,
        numerical_fits_added=0,new_step_search=False,
        endpoint_candidates=['controlled_case33_00','controlled_case33_01'],
        single_replay_candidate='controlled_case33_00',single_replay_method='ref_jac',
        lambda_fixed=1e-4,source_sha256=b._digest(__file__),kernel_sha256=cross['kernel_sha256'])
    b._write(output/'plan.json',plan)
    records={}
    for task in selected:
        file=stage/'outputs/candidate_h24_v1/tasks'/task['task_id']/'worker_result.json'
        records[(task['candidate_name'],task['algorithm'])]=(task,b._read(file))
    geometries=[];replay=None
    for name in ('controlled_case33_00','controlled_case33_01'):
        task,ref=records[(name,'ref_jac')]
        _,other=records[(name,'block_jac_gap')]
        _,_,_,_,problem,vector,_,_,_=e._context(whitelist,task,ref)
        _,_,_,_,_,second,_,_,_=e._context(whitelist,task,other)
        geometries.append(dict(candidate=name,old_reference_vs_QR=state_geometry(problem,vector,second)))
        if name=='controlled_case33_00':
            pplan=b._read(stage/'outputs/candidate_h24_polish_v1/plan.json')
            ptask=next(t for t in pplan['tasks'] if t['original_task_id']==task['task_id'])
            prec=b._read(stage/'outputs/candidate_h24_polish_v1/tasks'/ptask['task_id']/'worker_result.json')
            replay=rejected_step(problem,vector,prec['fit']['diagnostics']['polish']['trace'][0])
    worker_records=[b._read(path) for path in (cross_dir/'tasks').glob('*/worker_result.json')]
    statuses=[]
    for record in worker_records:
        state=record['attained_state'];task=record['task']
        _,_,_,_,problem,vector,_,_,_=e._context(whitelist,task,record)
        r,x,_=problem.unpack(vector);a=problem.expand(vector)[problem.m:problem.shared]
        _,_,w,_,_=problem.mapping.section(r)
        trace=record['fit']['diagnostics'].get('trace',[])
        statuses.append(dict(candidate=task['candidate_name'],warm=task['warm_method'],method=task['algorithm'],
            status=record['fit']['status'],message=record['fit']['message'],nfev=record['fit']['nfev'],
            commonPG=record['audit']['reference_scaled_projected_gradient'],
            fit_seconds=record['timings']['fit_seconds'],
            alpha_lower_active=int(np.count_nonzero(a<=1e-9)),alpha_upper_active=int(np.count_nonzero(a>=1-1e-9)),
            section_width_min=float(w.min()),section_width_max=float(w.max()),collapsed_sections=int(np.count_nonzero(w<=1e-12)),
            last_trace=trace[-3:],old_endpoint_unchanged=record['old_endpoint_bytes_unchanged'],
            kernel_matches=all(record['source_hashes'].get(k)==v for k,v in cross['kernel_sha256'].items())))
    summary=b._read(cross_dir/'crosswarm_summary.json')
    result=dict(status='complete_arithmetic_review',plan=plan,crosswarm_fit_denominator=8,
        returned=len(worker_records),all_kernel_hashes_match=all(r['kernel_matches'] for r in statuses),
        all_old_endpoint_hashes_unchanged=all(r['old_endpoint_unchanged'] for r in statuses),
        optimizer_budget_respected=all(r['nfev']<=250 for r in statuses),
        all_same_start_and_scale=all(r['same_common_scale'] and r['same_effective_initial_state'] and
            r['same_requested_initial_state'] for r in summary['same_start_solver_comparisons']),
        common_stationarity_passes=summary['precision_qualified'],H192_expansion_allowed=False,
        crosswarm_statuses=statuses,retained_endpoint_geometry=geometries,single_rejected_step=replay,
        scientific_EV_efficacy_evaluated=False)
    b._write(output/'review.json',b._serial(result))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    result=run(args.stage,args.output)
    print('complete',result['returned'],'crosswarm records;',result['common_stationarity_passes'],'common-stationary')
