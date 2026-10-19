"""Same-witness AC audit of the moving-center mixed-derivative model.

Reads all 54 saved predicted-mean competitor fits. No candidate fitting or
new action selection is performed. Old frozen trial bytes are hashed before
and after. Both successful and failed audit rows are retained.
"""
from __future__ import annotations
import argparse
import gzip
import hashlib
import json
from pathlib import Path
from time import perf_counter
import numpy as np

from .likelihood import Observations, RadialCandidate
from .information_context import meter_from_record
from .symmetric_ac_local import whiten_observations, mean_and_jacobian
from .ev_profile_cross_term_audit import audit_row as old_audit_row, read_json, sha, serial
from .ev_ac_moving_cross import (build_from_saved, parameter_increment, physical_parameter,
    residual, channel_norms, state_vector, vector_state, future_geometry, original_bounds,
    add_control_curvature)

HERE=Path(__file__).resolve().parent
STATE=('p','q','source_p','source_q')


def _state(fit, start=0, stop=None):
    return {key:np.asarray(fit[key],float)[start:stop] for key in STATE}


def _norm(v): return float(np.linalg.norm(v))


def compare_errors(exact, predictions, history_rows, h, t, n, channels):
    result={}
    exact=np.asarray(exact,float); hrows=history_rows
    for name,value in predictions.items():
        value=np.asarray(value,float); error=value-exact
        future_ch=channel_norms(error[hrows:],channels,t,n)
        rms={key:item['norm']/np.sqrt(item['rows']) for key,item in future_ch.items()}
        result[name]=dict(residual_norm=_norm(value),sse=float(value@value),
            sse_error=float(value@value-exact@exact),residual_error_norm=_norm(error),
            history_error_norm=_norm(error[:hrows]),future_error_norm=_norm(error[hrows:]),
            history_channels=channel_norms(error[:hrows],channels,h,n),
            future_channels=future_ch,future_channel_RMS=rms,
            future_channel_RMS_gate_pass=all(v<=.25 for v in rms.values()),
            whole_white_RMS=_norm(error)/np.sqrt(len(error)))
    return result


def audit_saved_row(record, item, old_model, moved, curved, candidate, counter):
    original=old_audit_row(record,item,old_model,counter)
    h=len(record['history']['p']); t=len(old_model['reference_kw']); n=candidate.n
    channels=tuple(old_model['channels']); hrows=moved['history_rows']
    fits={fit['name']:fit for fit in item['predicted_joint_fits']}
    fit=fits[old_model['b']]; theta1=np.r_[fit['r'],fit['x']]
    theta0=np.asarray(old_model['full']['parameter_center'])[:len(theta1)]
    witness=np.r_[theta1,state_vector(_state(fit,0,h)),state_vector(_state(fit,h,None))]
    u=np.asarray(next(row['u_kw'] for row in record['rows'] if row['method']==item['method']))
    z=parameter_increment(moved,u,witness); du=u-moved['reference_kw']
    old_lin=np.asarray(original['linear_residual']); exact=np.asarray(original['nonlinear_residual'])
    linear_moved=np.asarray(moved['c'])+moved['B']@u+moved['C_at_reference']@z
    if not np.allclose(old_lin,linear_moved,atol=3e-10,rtol=0):
        raise ValueError('moving coordinate transformation changed first-order prediction')
    cross=residual(moved,u,z)
    cross_curve=residual(curved,u,z)
    source_error=np.asarray(original['source_forward_error'])
    corrected=cross.copy(); corrected[hrows:]+=source_error
    meter=meter_from_record(record['measurement_models']['future'])
    center=state_vector(moved['future_background'])+moved['domain']['shift'][-moved['future_state_columns']:]@u
    center_state=vector_state(center,n); future_state=_state(fit,h,None)
    dt=theta1-theta0; de=state_vector(future_state)-center
    root=np.asarray(moved['future_root_v'])
    def geometry(theta,state,with_jacobian=False):
        counter['ac_forward_attempts']+=1
        ans=mean_and_jacobian(candidate,theta[:candidate.m],theta[candidate.m:],
            state['p'],state['q'],state['source_p'],state['source_q'],root,meter,
            channels=channels,with_jacobian=with_jacobian)
        counter['ac_forward_completed']+=1
        return ans
    at_center=geometry(theta0,center_state,True)
    m10=geometry(theta1,center_state)['mean']
    m01=geometry(theta0,future_state)['mean']
    nobs=whiten_observations(Observations(**item['predicted_observations']),meter,channels)
    m11=nobs-exact[hrows:]
    future_first=at_center['mean']+at_center['line_jacobian']@dt+at_center['state_jacobian']@de
    exact_center=np.r_[linear_moved[:hrows],nobs-future_first]
    theta_remainder=m10-at_center['mean']-at_center['line_jacobian']@dt
    state_remainder=m01-at_center['mean']-at_center['state_jacobian']@de
    remaining_cross=m11-m10-m01+at_center['mean']
    closure=m11-future_first-theta_remainder-state_remainder-remaining_cross
    # Directional theta-times-background-increment Taylor term. This is a
    # diagnostic, not a convex-QP term: theta and background are both unknown.
    eta=1e-3
    gp=geometry(theta0,vector_state(center+eta*de,n),True)
    gm=geometry(theta0,vector_state(center-eta*de,n),True)
    theta_e_local=((gp['line_jacobian']-gm['line_jacobian'])/(2*eta))@dt
    with_theta_e=exact_center.copy(); with_theta_e[hrows:]-=theta_e_local
    control_term=np.einsum('t,tij->ij',du,moved['At'])
    lines=moved['line_columns']; hstates=moved['history_state_columns']
    u_theta=control_term[:, :lines]@z[:lines]
    u_e=control_term[:,lines+hstates:]@z[lines+hstates:]
    source_cols=lines+hstates+np.ravel(np.column_stack(
        (np.arange(t)*(2*n+2)+2*n,np.arange(t)*(2*n+2)+2*n+1)))
    u_source=control_term[:,source_cols]@z[source_cols]
    future_cross_channels=channel_norms(remaining_cross,channels,t,n)
    lo,hi=original_bounds(moved,u); outer=moved['domain']
    predictions=dict(old_linear=old_lin,moving_linear=linear_moved,
        moving_cross=cross,moving_cross_control_curvature=cross_curve,
        moving_cross_exact_reference=corrected,
        exact_moving_center_first_order=exact_center,
        exact_moving_center_plus_theta_e_directional=with_theta_e)
    metrics=compare_errors(exact,predictions,hrows,h,t,n,channels)
    return dict(status='complete',method=item['method'],a=item['a'],b=item['b'],u_kw=u,
        h=h,t=t,n=n,channels=channels,saved_fit_success=fit['success'],
        full_AC_cost=float(exact@exact),saved_AC_cost=float(fit['loss']),
        full_AC_cost_replay_abs_error=abs(float(exact@exact)-float(fit['loss'])),
        moving_coordinate_equivalence_max_abs=float(np.max(abs(old_lin-linear_moved))),
        domain=dict(original_violation=float(max(0.,np.max(lo-z),np.max(z-hi))),
            outer_violation=float(max(0.,np.max(np.asarray(outer['outer_lower'])-z),np.max(z-np.asarray(outer['outer_upper'])))),
            safe_intersection_violation=float(max(0.,np.max(np.asarray(outer['safe_lower'])-z),np.max(z-np.asarray(outer['safe_upper'])))),
            reconstruct_max_abs=float(np.max(abs(physical_parameter(moved,u,z)-witness)))),
        metrics=metrics,
        remaining_future=dict(theta_norm=_norm(theta_remainder),state_norm=_norm(state_remainder),
            theta_times_e_norm=_norm(remaining_cross),theta_times_e_local_norm=_norm(theta_e_local),
            theta_times_e_local_error=_norm(remaining_cross-theta_e_local),
            closure_max_abs=float(np.max(abs(closure))),channels=future_cross_channels,
            theta_increment_norm=_norm(dt),background_increment_norm=_norm(de),
            source_increment_norm=_norm(de.reshape(t,2*n+2)[:,-2:])),
        captured=dict(control_times_theta_norm=_norm(u_theta),control_times_e_norm=_norm(u_e),
            control_times_source_norm=_norm(u_source),
            source_control_squared_norm=_norm(.5*curved['control_curvature_reference']@(du*du)),
            competitor_control_squared_norm=_norm(.5*curved['control_curvature_competitor']@(du*du)),
            difference_control_squared_norm=_norm(cross_curve-cross),
            control_times_theta_future_channels=channel_norms(u_theta[hrows:],channels,t,n),
            control_times_e_future_channels=channel_norms(u_e[hrows:],channels,t,n)),
        source_direct_meter_channels_present=any(k in channels for k in ('source_p','source_q')),
        source_interface='same physical node 1; hidden source enters voltage and master channels',
        residual_vectors=dict(full_AC=exact,old_linear=old_lin,moving_cross=cross,
            moving_cross_control_curvature=cross_curve,
            exact_moving_center_first_order=exact_center,remaining_theta_e=remaining_cross),
        is_complete_AC_optimization=False)


def summarize(trials):
    rows=[r for trial in trials for r in trial['rows'] if r['status']=='complete']
    methods=sorted({r['method'] for r in rows}); proxies=list(rows[0]['metrics']) if rows else []
    def grouped(group):
        result=dict(rows=len(group),proxies={})
        for proxy in proxies:
            met=[r['metrics'][proxy] for r in group]
            result['proxies'][proxy]=dict(median_residual_error=float(np.median([m['residual_error_norm'] for m in met])),
                median_future_error=float(np.median([m['future_error_norm'] for m in met])),
                median_abs_sse_error=float(np.median([abs(m['sse_error']) for m in met])),
                max_abs_sse_error=max(abs(m['sse_error']) for m in met),
                max_future_channel_RMS=max(max(m['future_channel_RMS'].values()) for m in met),
                future_channel_RMS_gate_pass_rows=sum(m['future_channel_RMS_gate_pass'] for m in met),
                median_whole_white_RMS=float(np.median([m['whole_white_RMS'] for m in met])),
                future_channel_median_error={k:float(np.median([m['future_channels'][k]['norm'] for m in met]))
                    for k in group[0]['channels']})
        result['moving_cross_improves_residual_rows']=sum(r['metrics']['moving_cross']['residual_error_norm']<r['metrics']['old_linear']['residual_error_norm'] for r in group)
        result['moving_cross_improves_future_rows']=sum(r['metrics']['moving_cross']['future_error_norm']<r['metrics']['old_linear']['future_error_norm'] for r in group)
        result['moving_cross_improves_abs_cost_rows']=sum(abs(r['metrics']['moving_cross']['sse_error'])<abs(r['metrics']['old_linear']['sse_error']) for r in group)
        result['median_remaining_theta_e_norm']=float(np.median([r['remaining_future']['theta_times_e_norm'] for r in group]))
        result['median_captured_u_theta_norm']=float(np.median([r['captured']['control_times_theta_norm'] for r in group]))
        result['median_captured_u_e_norm']=float(np.median([r['captured']['control_times_e_norm'] for r in group]))
        result['median_source_control_squared_norm']=float(np.median([r['captured']['source_control_squared_norm'] for r in group]))
        result['median_competitor_control_squared_norm']=float(np.median([r['captured']['competitor_control_squared_norm'] for r in group]))
        result['median_difference_control_squared_norm']=float(np.median([r['captured']['difference_control_squared_norm'] for r in group]))
        return result
    ranking=[]
    for trial in trials:
        for a,b in sorted({(r['a'],r['b']) for r in trial['rows'] if r['status']=='complete'}):
            group=[r for r in trial['rows'] if r['status']=='complete' and (r['a'],r['b'])==(a,b)]
            if len(group)<2: continue
            exact_rank=[r['method'] for r in sorted(group,key=lambda r:r['full_AC_cost'],reverse=True)]
            ranks={proxy:[r['method'] for r in sorted(group,key=lambda r:r['metrics'][proxy]['sse'],reverse=True)] for proxy in proxies}
            ranking.append(dict(file=trial['file'],a=a,b=b,full_AC_witness_rank=exact_rank,proxy_witness_ranks=ranks))
    return dict(trials=len(trials),complete_rows=len(rows),failed_rows=sum(r['status']!='complete' for t in trials for r in t['rows']),
        all=grouped(rows) if rows else {},by_method={m:grouped([r for r in rows if r['method']==m]) for m in methods},
        max_coordinate_equivalence=max((r['moving_coordinate_equivalence_max_abs'] for r in rows),default=0.),
        max_original_domain_violation=max((r['domain']['original_violation'] for r in rows),default=0.),
        max_outer_domain_violation=max((r['domain']['outer_violation'] for r in rows),default=0.),
        safe_intersection_excluded_witness_rows=sum(r['domain']['safe_intersection_violation']>1e-9 for r in rows),
        max_cost_replay_error=max((r['full_AC_cost_replay_abs_error'] for r in rows),default=0.),
        max_mixed_step_check_rel=max((m['step_check_relative'] for t in trials for m in t['model_checks']),default=0.),
        new_candidate_fits=0,new_selectors=0,ac_forward_calls=sum(t['ac_forward_completed'] for t in trials),
        source_direct_meter_rows=sum(r['source_direct_meter_channels_present'] for r in rows),
        ranking_comparisons=len(ranking),rank_match_counts={p:sum(x['proxy_witness_ranks'][p]==x['full_AC_witness_rank'] for x in ranking) for p in proxies},
        rankings=ranking,
        mixed_proxy_gate='inspect same-witness error and unmodeled theta-times-background terms before new strategy evaluation')


def audit_trial(path):
    before=sha(path); record=read_json(path); models=record['proxy_models']['models']
    hfits={f['name']:f for f in record['historical_fits']}; counter=dict(ac_forward_attempts=0,ac_forward_completed=0)
    rows=[]; checks=[]; future_meter=meter_from_record(record['measurement_models']['future'])
    cache={}
    for old in models:
        c=RadialCandidate.from_supports(old['b'],len(record['history']['p'][0]),hfits[old['b']]['supports'])
        moved=build_from_saved(old,c,future_meter)
        twice=build_from_saved(old,c,future_meter,step_kw=.002)
        own=RadialCandidate.from_supports(old['a'],len(record['history']['p'][0]),hfits[old['a']]['supports'])
        curved=add_control_curvature(moved,old,own,future_meter)
        counter['ac_forward_attempts']+=6; counter['ac_forward_completed']+=6
        denom=max(np.linalg.norm(moved['At']),1e-15)
        checks.append(dict(a=old['a'],b=old['b'],step_kw=.001,compare_step_kw=.002,
            step_check_absolute=float(np.linalg.norm(moved['At']-twice['At'])),
            step_check_relative=float(np.linalg.norm(moved['At']-twice['At'])/denom)))
        cache[(old['a'],old['b'])]=(old,moved,curved,c)
    for item in record['audit']:
        if 'predicted_joint_fits' not in item: continue
        try:
            old,moved,curved,c=cache[(item['a'],item['b'])]
            rows.append(audit_saved_row(record,item,old,moved,curved,c,counter))
        except (KeyError,ValueError,RuntimeError,FloatingPointError,np.linalg.LinAlgError) as error:
            rows.append(dict(status='audit_failure',method=item['method'],a=item['a'],b=item['b'],error=type(error).__name__+': '+str(error)))
    if sha(path)!=before: raise ValueError('old frozen input changed during read-only audit')
    return dict(file=Path(path).name,file_sha256=before,seed=record['seed'],truth_r=record['truth_r'],fit_floor=record['fit_floor'],
        input_bytes_unchanged=True,rows=rows,model_checks=checks,**counter)


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--directory',type=Path,default=HERE/'evidence'/'ev_profile_milp'/'v1')
    parser.add_argument('--pattern',default='trial_986101_*.json.gz'); parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--limit',type=int); args=parser.parse_args(); began=perf_counter()
    paths=sorted(args.directory.glob(args.pattern)); paths=paths[:args.limit] if args.limit else paths
    if not paths: raise ValueError('no saved trials match')
    trials=[audit_trial(p) for p in paths]
    answer=dict(schema='moving_AC_cross_same_witness_audit_v1',
        scope='post-hoc 54 saved competitor witnesses; complete AC replays, no refitting or action choice',
        summary=summarize(trials),trials=trials,seconds=perf_counter()-began,
        source_sha256={name:sha(HERE/name) for name in ('ev_ac_moving_cross.py','ev_ac_moving_cross_audit.py','symmetric_ac_local.py','ev_profile_cross_term_audit.py')})
    args.output.parent.mkdir(parents=True,exist_ok=True)
    opener=gzip.open if args.output.suffix=='.gz' else open
    with opener(args.output,'wt',encoding='utf-8') as f: json.dump(serial(answer),f,indent=2,ensure_ascii=False,allow_nan=False)
    print(json.dumps(serial(answer['summary']),ensure_ascii=False))


if __name__=='__main__': main()
