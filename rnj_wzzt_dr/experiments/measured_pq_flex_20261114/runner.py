"""Paired DFS on frozen measured-P/Q histories; no identifier refits.

Four forecasts per each of 16 fixed physical days, eight conditional execution
paths per forecast. These are not 512 independent load days. Nominal exact
requested-action AC and actual response/clipping AC are recorded separately.
"""
from __future__ import annotations
import os
for _key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[_key]='1'

import argparse
import atexit
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime,timezone
from itertools import combinations
from pathlib import Path
import shutil
import sys
import traceback
from time import perf_counter

import numpy as np
from scipy.special import logsumexp

from ... import io
from ...observations import Observation, load
from ...history import freeze_history
from ...dr.selection_mse import choose_mse
from ..flexibility_sweep_20261113.fitting import pack_score, calibrated_means
from ..flexibility_sweep_20261113.execution import fixed_rotation, native_selection
from ..linear_ac_population_20261113.study import evaluate
from ..customer_flexibility_20261113.placement import CandidateEvidence
from ..absolute_identification_20261114.commonseeds import build_portfolio, solve_old_with_seeds
from ..absolute_identification_20261114.solver import plan as recovery_plan
from ..posterior_mse_20261112.scoring import topology_id
from ..measured_pq_20261114.physical import offline_truth, _network_hash
from ..measured_pq_20261114.validate import _environment, _git_receipt
from .domain import IncrementCase, catalog
from . import full_ac

ROOT=Path(__file__).resolve().parents[3]
MODES=('old_gamma2','symmetric_guard')
WORK={}


def utc():return datetime.now(timezone.utc).isoformat()


def checked(reference):
    path=Path(reference['path'])
    if io.ref(path)['sha256']!=reference['sha256']:
        raise ValueError('Sealed source changed: '+str(path))
    return path


def rng(seed,index,forecast,component,repeat=0):
    return np.random.default_rng(np.random.SeedSequence([int(seed),914,int(index),int(forecast),int(repeat),int(component)]))


def forecast_power(clean_power,seed,index,draw):
    return np.asarray(clean_power)*(1+.05*rng(seed,index,draw,991).normal(size=np.asarray(clean_power).shape))


def common_response(point,n):
    """The full-terminal common coefficient, with all cross terms kept later."""
    common=np.zeros((n,2*n))
    for C,r,x in zip(point['supports'],point['r'],point['x'],strict=True):
        if len(C)==n:
            common[:,:n]+=r;common[:,n:]+=x
    return common


def pair_components(H,common_H,history,sigma2,power,y=None,observed=None,history_without_common=None):
    """Every fixed unordered native pair; direct residual contrast is checked."""
    H=np.asarray(H);P=np.asarray(power);pairs=np.array(list(combinations(range(len(H)),2)),dtype=int).reshape(-1,2)
    if not len(pairs):
        return dict(pairs=pairs,h=np.empty(0),J=np.empty(0),J_without_common=np.empty(0),
                    J_common=np.empty(0),J_cross=np.empty(0))
    left,right=pairs.T;D=H[left]-H[right];Dc=common_H[left]-common_H[right]
    d=np.einsum('tc,knc->ktn',P,D);dc=np.einsum('tc,knc->ktn',P,Dc);dn=d-dc
    J=np.sum(d*d,axis=(1,2))/(2*sigma2)
    Jn=np.sum(dn*dn,axis=(1,2))/(2*sigma2);Jc=np.sum(dc*dc,axis=(1,2))/(2*sigma2)
    cross=np.sum(dn*dc,axis=(1,2))/sigma2
    if not np.allclose(J,Jn+Jc+cross,rtol=1e-10,atol=1e-9):raise ArithmeticError('Common cross-term identity failed')
    h=(np.asarray(history)[right]-np.asarray(history)[left])/(2*sigma2)
    answer=dict(pairs=pairs,h=h,J=J,J_without_common=Jn,J_common=Jc,J_cross=cross)
    if y is not None:
        residual=np.asarray(y)[None]-np.einsum('tc,knc->ktn',P,H[left])
        bias=np.sum(residual*d,axis=(1,2))/sigma2
        exact=np.array([(np.sum((y-P@H[j].T)**2)-np.sum((y-P@H[i].T)**2))/(2*sigma2)
                        for i,j in pairs])
        if not np.allclose(J+bias,exact,rtol=1e-9,atol=1e-8):raise ArithmeticError('Clean pair contrast identity failed')
        answer['bias']=bias;answer['clean_logodds']=h+exact
        if history_without_common is not None:
            reduced=H-common_H
            clean_sse_reduced=np.sum((np.asarray(y)[None]-np.einsum('tc,knc->ktn',P,reduced))**2,axis=(1,2))
            h0=np.asarray(history_without_common)
            answer['clean_logodds_without_common']=((h0[right]+clean_sse_reduced[right])-(h0[left]+clean_sse_reduced[left]))/(2*sigma2)
        if observed is not None:
            measured=np.c_[observed.p,observed.q]
            sse=np.sum((observed.y[None]-np.einsum('tc,knc->ktn',measured,H))**2,axis=(1,2))
            logodds=h+(sse[right]-sse[left])/(2*sigma2)
            answer['observed_logodds']=logodds
            answer['meter']=logodds-h-J-bias
            answer['native_new_sse']=sse
            reduced=H-common_H
            sse_reduced=np.sum((observed.y[None]-np.einsum('tc,knc->ktn',measured,reduced))**2,axis=(1,2))
            if history_without_common is not None:
                h0=np.asarray(history_without_common)
                answer['observed_logodds_without_common']=((h0[right]+sse_reduced[right])-(h0[left]+sse_reduced[left]))/(2*sigma2)
                answer['native_total_sse_without_common']=h0+sse_reduced
            answer['native_new_sse_without_common']=sse_reduced
    return answer


def pair_delta(before,after):
    if not np.array_equal(before['pairs'],after['pairs']):raise ValueError('Pair identities changed')
    out={key:after[key]-before[key] for key in ('J','J_without_common','J_common','J_cross','bias','meter','observed_logodds','observed_logodds_without_common') if key in before and key in after}
    if all(k in out for k in ('J','bias','meter','observed_logodds')):
        error=float(np.max(np.abs(out['observed_logodds']-out['J']-out['bias']-out['meter']),initial=0.))
        if error>1e-7:raise ArithmeticError('Paired log-odds change decomposition failed')
        out['maximum_identity_error']=error
    return out


def meter(p,q,v,root,noises,pq_sigma=.005,voltage_sigma=.0005):
    pm=p*(1+pq_sigma*noises[0]);qm=q*(1+pq_sigma*noises[1])
    vm=v*(1+voltage_sigma*noises[2]);rm=root*(1+voltage_sigma*noises[3])
    return pm,qm,rm[:,None]**2-vm**2,vm,rm


def prepare(index,ac_index,out,*,seed=20261114,forecasts=4,repeats=8,seconds=20.,workers=4,limit=None):
    out=Path(out).resolve()
    if out.exists():raise FileExistsError('Preserve previous DFS outputs')
    index,ac_index=Path(index).resolve(),Path(ac_index).resolve()
    source=io.read(index);cached=io.read(ac_index)
    resources=io.read(checked(source['resource_context']))
    resource_rows={(r['case'],r['start_date']):r for r in resources['rows']}
    cache_rows={(r['case'],r['start_date']):r for r in cached['rows']}
    selected=source['inputs'][:limit] if limit is not None else source['inputs']
    out.mkdir(parents=True);entries=[]
    for i,item in enumerate(selected):
        key=item['case'],item['start_date'];context=resource_rows[key];cache=cache_rows[key]
        for name in ('train.npz','validation.npz','new_day.npz','101_native.json'):
            checked(item['files'][name])
        if not context['PQ_bitwise_equal_to_executed_inputs']:
            raise ValueError('Resource context changed the frozen P/Q')
        checked(cache['cache']);checked(cache['original'])
        train=load(item['files']['train.npz']['path']);val=load(item['files']['validation.npz']['path'])
        native=io.read(item['files']['101_native.json']['path'])
        score=pack_score(native,train,val)
        identity=key[0]+'__'+key[1];d=out/'histories'/identity;d.mkdir(parents=True)
        io.write(d/'score.json',score);io.write(d/'resource_context.json',context)
        shutil.copyfile(item['files']['101_native.json']['path'],d/'101_native.json')
        entries.append(dict(id=identity,index=int(item.get('original_input_index',i)),case=key[0],start_date=key[1],files=item['files'],
            score=io.ref(d/'score.json'),native_copy=io.ref(d/'101_native.json'),
            resource_context=io.ref(d/'resource_context.json'),physical_cache=cache,
            historical_native_index=native['selected']['index'],historical_p_map=score['p_map'],
            map_matches_native_topology=score['map_matches_native_topology']))
    io.write(out/'inputs.json',entries)
    # Freeze the actually imported local dependency closure now. Unrelated
    # enrichment/report modules added later do not change this locked list.
    package=ROOT/'rnj_wzzt_dr'
    imported={Path(module.__file__).resolve() for module in list(sys.modules.values())
              if getattr(module,'__file__',None) and str(module.__file__).endswith('.py')
              and Path(module.__file__).resolve().is_relative_to(package)}
    imported.add(package/'experiments/expected_logodds_admm_20261113/evidence.py')
    source_paths=sorted(imported)+[Path(full_ac.backend.__file__)]
    source_paths.extend(full_ac.backend.ROOT/'cases'/case/name for case in sorted({e['case'] for e in entries})
                        for name in ('metadata.json','buses.csv','branches.csv'))
    sources=[]
    for path in source_paths:
        ref=io.ref(path);sources.append(ref)
        destination=out/'source_snapshot'/path.relative_to(ROOT)
        destination.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,destination)
    protocol=dict(created_utc=utc(),input_index=io.ref(index),AC_cache_index=io.ref(ac_index),
        source_resource_context=source['resource_context'],inputs=io.ref(out/'inputs.json'),sources=sources,
        environment=_environment(),**_git_receipt(source_paths),seed=seed,forecasts=forecasts,repeats=repeats,
        solver_seconds=seconds,workers=workers,modes=list(MODES),initial_fraction=.1,
        expansion=dict(fraction=.25,mode='old_gamma2',trigger='nonzero verified feasible quotes < 8 of the 64 primary forecast problems',
                       threshold=8,denominator=len(entries)*forecasts,automatic_only_for_full_16x4=len(entries)==16 and forecasts==4,
                       implementation_failure_guard='Any failed task or quote exception blocks automatic expansion; unresolved/infeasible numerical results retain the predeclared trigger'),
        interpretation=f'{len(entries)} fixed already-opened physical held-out days; {len(entries)*forecasts*repeats} conditional paired response/meter paths, not independent load days',
        phase='development DFS evaluation on existing measured-profile histories',
        forecast='True held-out net P/Q multiplied by 1+.05 independent standard Gaussian; no true V/branches into planner',
        meter=dict(additional_PQ_relative_sd=.005,additional_V_relative_sd=.0005),
        response='One independent Uniform(.9,1) multiplier per resource for whole day; synthetic PV contract clipped by min-hour actual net-export proxy, not a measured PV component',
        baseline='Reuse sealed no-action full-AC cache; never add any device baseline',
        scoring='All native frozen101 points; history validation96 + new96 equal residual-cell weights. Representative probabilities secondary.',
        refits=0,training_rows=192,validation_rows=96,new_rows=96,
        primary_structure='Proper internal clades exactly matched; all-active support exactness separately reported',
        common_term='Frozen H unchanged in planning/scoring. Same-action diagnostics recompute J including its cross terms and clean/observed residual contrasts after removing common; validation SSE is recomputed for this diagnostic only.',
        exact_execution='Requested action with eta=1 and no actual-PV clipping; ideal delivery diagnostic. Actual availability exceedance separately recorded.',
        failure='Infeasible/unresolved procurement uses explicit zero fallback; all executions stay in denominator; AC errors retained')
    io.write(out/'protocol.json',protocol)
    return protocol


def initialize(out):
    import gurobipy as gp
    out=Path(out);env=gp.Env(empty=True);env.setParam('OutputFlag',0);env.start();atexit.register(env.dispose)
    WORK.update(out=out,p=io.read(out/'protocol.json'),env=env,contexts={})


def context(entry):
    if entry['id'] in WORK['contexts']:return WORK['contexts'][entry['id']]
    score=io.read(checked(entry['score']));native=io.read(checked(entry['native_copy']))
    train=load(checked(entry['files']['train.npz']));val=load(checked(entry['files']['validation.npz']))
    frozen=freeze_history(native,val)
    if frozen.parameter_sha256!=score['parameter_sha256']:raise ValueError('Native bank differs from score')
    rc=io.read(checked(entry['resource_context']))
    cache=entry['physical_cache'];checked(cache['original'])
    with np.load(checked(cache['cache']),allow_pickle=False) as z:
        state={k:(z[k][288:384].copy() if z[k].ndim and z[k].shape[0]==384 else z[k].copy()) for k in z.files}
    with np.load(cache['original']['path'],allow_pickle=False) as z:
        p,q,root=[z[k][288:384].copy() for k in ('P','Q','root')]
        labels=tuple(map(int,z['labels']))
    if labels!=tuple(score['labels']):raise ValueError('Physical/score labels mismatch')
    network=full_ac.backend.load_case(entry['case'])
    if _network_hash(network)!=cache['network_hash']:raise ValueError('Network version changed')
    truth_supports=offline_truth(entry['case'],labels)
    clades=tuple(sorted(tuple(sorted(labels[i] for i in c)) for c in truth_supports if 1<len(c)<len(labels)))
    truth=dict(truth_id=topology_id(clades),truth_clades=clades)
    H=np.asarray([m['H'] for m in score['native_bank']]);common=np.asarray([common_response(m['point'],len(labels)) for m in native['frozen_candidates']])
    validation_power=np.c_[val.p,val.q]
    history_without_common=np.sum((val.y[None]-np.einsum('tc,knc->ktn',validation_power,H-common))**2,axis=(1,2))
    result=dict(score=score,native=native,train=train,val=val,resource=rc,means=calibrated_means(score,val),
        network=network,base=network['base_kw'],p=p,q=q,root=root,baseline_state=state,
        truth=truth,truth_supports=truth_supports,H=H,common_H=common,
        history=np.array([m['history_sse'] for m in score['native_bank']]),labels=labels,
        history_without_common=history_without_common,
        baseline_cache_ref=cache['cache'])
    target=int(score['native_index'])
    other=[i for i,m in enumerate(score['native_bank'])
           if m['topology_id']!=score['native_bank'][target]['topology_id']]
    result['history_runner_native_index']=min(other,key=lambda i:(score['native_bank'][i]['history_sse'],i)) if other else None
    WORK['contexts'][entry['id']]=result
    return result


def quote(mode,case,score,validation,means,forecast,base,seeds,seconds,env):
    """Only allowed observations, frozen fitted H, forecast, and contract enter."""
    target=score['map_index'];H=np.asarray([p['H'] for p in score['parameters']])
    others=[j for j in range(len(H)) if j!=target]
    if not others:
        return dict(feasible=True,status='single_candidate_zero',source='zero',cost_yuan=0.,lower_bound_yuan=0.,relative_gap=0.,global_optimality_verified=True,seconds=0.),np.zeros(case.shape)
    if mode=='old_gamma2':
        evidence=CandidateEvidence(case,H,target,others,forecast,score['sigma2'],base,means)
        return solve_old_with_seeds(evidence,2.,seed_points=seeds,seconds=seconds,env=env)
    return recovery_plan(case,score,validation,forecast,base,mode='symmetric_guard',seconds=seconds,env=env,seed_points=seeds)


def full_structure(point,truth,zero):
    active={tuple(c) for c,r,x in zip(point['supports'],point['r'],point['x'],strict=True) if max(r,x)>zero}
    truth=set(map(tuple,truth))
    return dict(native_full_support_exact=active==truth,native_full_support_missing=len(truth-active),native_full_support_extra=len(active-truth))


def score_observation(ctx,obs):
    s=ctx['score'];values=evaluate(s,obs,ctx['truth'],s['map_index']);ll=values.pop('log_probability')
    values.update(native_selection(s,obs,ctx['truth']))
    index=values['native_selected_index'];point=ctx['native']['frozen_candidates'][index]['point']
    values.update(full_structure(point,ctx['truth_supports'],ctx['native']['config']['zero_tol']))
    power=np.c_[obs.p,obs.q]
    new=np.sum((obs.y[None]-np.einsum('tc,knc->ktn',power,ctx['H']))**2,axis=(1,2))
    return values,ll,new


def physical_state(ctx,case,u,destination):
    dp,dq=case.aggregate(case.delivery(u));p,q=ctx['p']+dp/ctx['base'],ctx['q']+dq/ctx['base']
    if np.max(np.abs(u),initial=0.)==0:
        state=ctx['baseline_state'];checks=dict(status='reused_sealed_zero_action_cache',thermal_rating_check=False)
        reference=dict(**ctx['baseline_cache_ref'],time_slice=[288,384])
    else:
        state,checks=full_ac.solve(ctx['network'],ctx['labels'],p,q,ctx['root'])
        np.savez_compressed(destination,**state)
        reference=io.ref(destination)
    return p,q,state,checks,reference


def grid_value(case,root_kw):
    price=case.config['pricing'];buy=np.full(24,price['grid_buy_yuan_per_kwh']['day'])
    buy[(case.clock>=0)&(case.clock<7)]=price['grid_buy_yuan_per_kwh']['night']
    buy[(case.clock>=17)&(case.clock<22)]=price['grid_buy_yuan_per_kwh']['evening']
    buy=np.repeat(buy,4)
    return float(.25*np.sum(buy*np.maximum(root_kw,0)-.9*buy*np.maximum(-root_kw,0)))


def utilization(case,u):
    caps=np.array([r['contract_kw'] for r in case.resources]);total=float(caps.sum())
    envelope=float(np.maximum(abs(case.lower),abs(case.upper)).sum()*case.dt)
    return dict(contract_capacity_kw=total,maximum_simultaneous_abs_kw=float(np.abs(u).sum(1).max()),
        peak_fraction_of_total_contract=float(np.abs(u).sum(1).max()/total),
        hourly_absolute_envelope_kwh=envelope,
        absolute_energy_fraction_of_envelope=float(np.abs(u).sum()*case.dt/envelope) if envelope else 0.,
        per_resource_peak_fraction={r['id']:float(np.max(abs(u[:,j]))/caps[j]) for j,r in enumerate(case.resources)})


def evidence_summary(ctx,before,after,forecast_before,forecast_after,exact_before,exact_after):
    change=pair_delta(before,after);forecast=pair_delta(forecast_before,forecast_after)
    exact=pair_delta(exact_before,exact_after) if exact_before is not None and exact_after is not None else {}
    pairs=before['pairs'];t=ctx['score']['native_index'];records=[]
    for k,(i,j) in enumerate(pairs):
        if t not in (i,j):continue
        sign=1 if t==i else -1
        # Reverse orientation keeps J, but b_ji=-2J_ij-b_ij. Only L and meter negate.
        bias_change=float(change['bias'][k]) if sign==1 else -2*float(change['J'][k])-float(change['bias'][k])
        records.append(dict(target=t,competitor=int(j if t==i else i),
            forecast_J_gain=float(forecast['J'][k]),exact_physical_J_gain=float(exact['J'][k]) if 'J' in exact else None,
            actual_physical_J_gain=float(change['J'][k]),actual_bias_change=bias_change,
            actual_meter_change=sign*float(change['meter'][k]),
            actual_logodds_change=sign*float(change['observed_logodds'][k]),
            same_observations_logodds_change_without_common=sign*float(change['observed_logodds_without_common'][k]) if 'observed_logodds_without_common' in change else None,
            forecast_J_gain_without_common=float(forecast['J_without_common'][k]),
            forecast_common_square_gain=float(forecast['J_common'][k]),forecast_cross_gain=float(forecast['J_cross'][k])))
    for record in records:
        error=abs(record['actual_logodds_change']-record['actual_physical_J_gain']-record['actual_bias_change']-record['actual_meter_change'])
        if error>1e-7:raise ArithmeticError('Directed historical-target decomposition failed')
    runner=ctx.get('history_runner_native_index')
    nearest=next((r for r in records if r['competitor']==runner),None)
    return change,dict(fixed_native_history_target=t,fixed_history_runner=runner,
        nearest_competitor=nearest,target_pairs=records,exact_requested_diagnostic_available=bool(exact),
        maximum_identity_error=change.get('maximum_identity_error',0.))


def run_forecast(task):
    entry,draw,fraction=task;p=WORK['p'];out=WORK['out']
    directory=out/f'fraction{fraction:.2f}'/entry['id']/f'forecast{draw:02d}'
    if (directory/'complete.json').exists():return io.read(directory/'complete.json')
    if (directory/'started.json').exists():raise RuntimeError('Review interrupted task before resuming: '+str(directory))
    directory.mkdir(parents=True);io.write(directory/'started.json',dict(pid=os.getpid(),created_utc=utc()))
    began=perf_counter();ctx=context(entry);s=ctx['score']
    power=np.c_[ctx['p'],ctx['q']];prediction=forecast_power(power,p['seed'],entry['index'],draw)
    case=IncrementCase(ctx['resource'],ctx['labels'],prediction,ctx['base'],fraction)
    io.write(directory/'resource_case.json',case.config)
    seed_points,seed_receipt=build_portfolio(case,s,prediction,ctx['base'],env=WORK['env'],seconds=min(2.,p['solver_seconds']/4))
    np.savez_compressed(directory/'shared_initial_points.npz',**seed_points)
    io.write(directory/'shared_initial_receipt.json',seed_receipt)
    pattern,pattern_receipt=fixed_rotation(case);pattern_fee=case.fee(pattern)
    controls={'no_dfs':np.zeros(case.shape)};plans={}
    for mode in MODES:
        tick=perf_counter()
        try:
            receipt,u=quote(mode,case,s,ctx['val'],ctx['means'],prediction,ctx['base'],seed_points,
                max(.001,p['solver_seconds']-seed_receipt['seconds']),WORK['env'])
            if (u is not None)!=bool(receipt['feasible']):raise ArithmeticError('Policy feasibility flag mismatch')
            if u is not None and not case.audit(u)['passed']:raise ArithmeticError('Invalid returned resource action')
        except Exception as exc:
            receipt,u=dict(feasible=False,status='exception',error=repr(exc),traceback=traceback.format_exc()),None
        controls[mode]=np.zeros(case.shape) if u is None else u
        fee=case.fee(controls[mode]);amount=min(1.,fee/pattern_fee)
        controls[mode+'_rotation']=amount*pattern
        receipt.update(shared_seed_seconds=seed_receipt['seconds'],driver_quote_seconds=perf_counter()-tick,
            charged_total_seconds=seed_receipt['seconds']+perf_counter()-tick,zero_fallback=u is None,
            charged_total_budget_exceeded=seed_receipt['seconds']+perf_counter()-tick>p['solver_seconds']+.05,
            nonzero_verified_feasible=bool(u is not None and fee>1e-9),
            rotation_fee_yuan=case.fee(controls[mode+'_rotation']),rotation_same_fee=abs(case.fee(controls[mode+'_rotation'])-fee)<=1e-7,
            rotation_full_pattern_fee_yuan=pattern_fee)
        plans[mode]=receipt
    np.savez_compressed(directory/'planner.npz',forecast=prediction,**controls)
    io.write(directory/'plans.json',plans);io.write(directory/'rotation_receipt.json',pattern_receipt)
    io.write(directory/'seal.json',dict(created_utc=utc(),planner=io.ref(directory/'planner.npz'),plans=io.ref(directory/'plans.json'),
        protocol=io.ref(out/'protocol.json'),score=entry['score'],planner_actual_voltage_or_network_inputs=False))
    sigma2=s['sigma2'];hist=ctx['history'];H=ctx['H'];C=ctx['common_H']
    forecast_pairs={name:pair_components(H,C,hist,sigma2,prediction+case.shift_pq_pu(u,ctx['base'])) for name,u in controls.items()}
    exact={};exact_pairs={};rows=[]
    # This stage uses the requested action exactly, before random delivery/noise.
    for name,u in controls.items():
        d=directory/'exact_requested'/name;d.mkdir(parents=True)
        try:
            pp,qq,state,checks,ref=physical_state(ctx,case,u,d/'full_ac.npz')
            v=state['terminal_voltage_pu'];yy=ctx['root'][:,None]**2-v**2
            ideal=Observation(pp,qq,yy,ctx['labels'],v,ctx['root'])
            components=pair_components(H,C,hist,sigma2,np.c_[pp,qq],yy,ideal,ctx['history_without_common'])
            exact_pairs[name]=components;exact[name]=(pp,qq,state,checks,ref)
            _,availability=case.execute(u,ctx['p']*ctx['base'],np.ones(case.R))
            np.savez_compressed(d/'pair_evidence.npz',**components)
            np.savez_compressed(d/'forecast_pair_evidence.npz',**forecast_pairs[name])
            np.savez_compressed(d/'clean_observation.npz',P=pp,Q=qq,V=v,Y=yy,root=ctx['root'],labels=ctx['labels'])
            io.write(d/'receipt.json',dict(status='complete',physical_state=ref,checks=checks,
                requested_fee_yuan=case.fee(u),PV_net_export_proxy_excess_kwh=availability['PV_clipped_kwh'],
                interpretation='Exact requested action, no random response or clipping; ideal delivery diagnostic'))
        except Exception as exc:
            io.write(d/'receipt.json',dict(status='failed',error=repr(exc),traceback=traceback.format_exc()))
    for repeat in range(p['repeats']):
        rd=directory/f'execution{repeat:02d}';rd.mkdir()
        eta=rng(p['seed'],entry['index'],draw,773,repeat).uniform(.9,1.,case.R)
        noises=[rng(p['seed'],entry['index'],draw,k,repeat).normal(size=shape) for k,shape in
                ((2,ctx['p'].shape),(3,ctx['q'].shape),(4,ctx['p'].shape),(5,ctx['root'].shape))]
        np.savez_compressed(rd/'paired_randomness.npz',eta=eta,**{f'noise{i}':z for i,z in enumerate(noises)})
        baseline_pairs=None
        for name,u in controls.items():
            d=rd/name;d.mkdir()
            row=dict(id=entry['id'],case=entry['case'],start_date=entry['start_date'],forecast_index=draw,
                repeat=repeat,contract_fraction=fraction,strategy=name,planned_fee_yuan=case.fee(u),
                historical_p_map=s['p_map'],historical_native_index=s['native_index'],
                quote_zero_fallback=False if name=='no_dfs' else plans[name.removesuffix('_rotation')]['zero_fallback'])
            try:
                actual,delivery=case.execute(u,ctx['p']*ctx['base'],eta)
                pp,qq,state,checks,ref=physical_state(ctx,case,actual,d/'full_ac.npz')
                v=state['terminal_voltage_pu'];clean_y=ctx['root'][:,None]**2-v**2
                pm,qm,y,vm,rm=meter(pp,qq,v,ctx['root'],noises)
                obs=Observation(pm,qm,y,ctx['labels'],vm,rm)
                values,ll,new_sse=score_observation(ctx,obs)
                components=pair_components(H,C,hist,sigma2,np.c_[pp,qq],clean_y,obs,ctx['history_without_common'])
                if name=='no_dfs':baseline_pairs=components
                if baseline_pairs is None:
                    change,summary={},dict(status='unavailable',reason='Paired no-action observation failed; actual main score remains valid')
                else:
                    change,summary=evidence_summary(ctx,baseline_pairs,components,forecast_pairs['no_dfs'],forecast_pairs[name],exact_pairs.get('no_dfs'),exact_pairs.get(name))
                np.savez_compressed(d/'pair_evidence.npz',**components)
                np.savez_compressed(d/'pair_changes.npz',**change)
                np.savez_compressed(d/'observations.npz',P=pm,Q=qm,Y=y,V=vm,root=rm,labels=ctx['labels'],
                    true_P=pp,true_Q=qq,true_V=v,true_root=ctx['root'],requested_kw=u,delivered_kw=actual,
                    native_history_sse=hist,native_new_sse=new_sse,native_total_sse=hist+new_sse,
                    representative_log_probability=ll)
                row.update(status='complete',**values,delivery=delivery,evidence=summary,physical_state=ref,AC=checks,
                    settled_fee_yuan=case.fee(actual),requested_abs_kwh=float(np.abs(u).sum()),
                    delivered_abs_kwh=float(np.abs(actual).sum()),
                    requested_utilization=utilization(case,u),delivered_utilization=utilization(case,actual),
                    baseline_root_energy_value_yuan=grid_value(case,ctx['baseline_state']['root_supply_kw']),
                    root_energy_value_yuan=grid_value(case,state['root_supply_kw']),
                    root_energy_value_change_yuan=grid_value(case,state['root_supply_kw'])-grid_value(case,ctx['baseline_state']['root_supply_kw']),
                    terminal_voltage_min=float(v.min()),terminal_voltage_max=float(v.max()),
                    all_bus_voltage_min=float(state['voltage_magnitude_pu'].min()),all_bus_voltage_max=float(state['voltage_magnitude_pu'].max()),
                    voltage_outside_095_105_cells=int(((state['voltage_magnitude_pu']<.95)|(state['voltage_magnitude_pu']>1.05)).sum()))
            except Exception as exc:
                row.update(status='execution_failed',error=repr(exc),traceback=traceback.format_exc())
            rows.append(row);io.write(d/'receipt.json',row)
        io.write(directory/'results.json',rows)
    receipt=dict(status='complete',id=entry['id'],forecast_index=draw,contract_fraction=fraction,
        rows=len(rows),failed_rows=sum(r['status']!='complete' for r in rows),seconds=perf_counter()-began,
        plans=io.ref(directory/'plans.json'),results=io.ref(directory/'results.json'),
        quote_exceptions=sum(r.get('status')=='exception' for r in plans.values()),
        old_nonzero_feasible=plans['old_gamma2']['nonzero_verified_feasible'],
        symmetric_nonzero_feasible=plans['symmetric_guard']['nonzero_verified_feasible'])
    io.write(directory/'complete.json',receipt)
    return receipt


def inventory(index,ac_index,out,fraction=.1,seed=20261114):
    source=io.read(index);contexts=io.read(checked(source['resource_context']))
    by_key={(r['case'],r['start_date']):r for r in contexts['rows']};rows=[]
    caches={(r['case'],r['start_date']):r for r in io.read(ac_index)['rows']}
    for i,item in enumerate(source['inputs']):
        key=item['case'],item['start_date'];context=by_key[key]
        with np.load(checked(caches[key]['original']),allow_pickle=False) as z:
            P=np.c_[z['P'][288:384],z['Q'][288:384]];labels=tuple(map(int,z['labels']))
        base=full_ac.backend.load_case(item['case'])['base_kw']
        f=forecast_power(P,seed,i,0);case=IncrementCase(context,labels,f,base,fraction)
        zero=case.audit(np.zeros(case.shape));pattern,audit=fixed_rotation(case)
        rows.append(dict(case=item['case'],start_date=item['start_date'],fraction=fraction,
            resources=case.resources,contract_nameplate_kw=sum(r['contract_kw'] for r in case.resources),
            max_simultaneous_up_kw=float(case.upper.sum(1).max()),max_simultaneous_down_kw=float(-case.lower.sum(1).min()),
            PV_export_limited_max_kw=float(case.upper[:,[j for j,r in enumerate(case.resources) if r['kind']=='pv']].sum(1).max()),
            resource_count=case.R,distinct_nodes=len(set(case.node)),zero_audit=zero,
            fixed_rotation_fee_yuan=case.fee(pattern),fixed_rotation_audit=audit))
    result=dict(status='complete',inputs=len(rows),rows=rows,scope='Resource/linear-domain inventory only; no QCQP or new AC')
    io.write(out,result);return result


def dispatch(out,fraction):
    out=Path(out);p=io.read(out/'protocol.json');entries=io.read(checked(p['inputs']))
    tasks=[(e,d,fraction) for e in entries for d in range(p['forecasts'])]
    receipts=[];failures=[]
    with ProcessPoolExecutor(max_workers=p['workers'],initializer=initialize,initargs=(str(out),)) as pool:
        pending={pool.submit(run_forecast,t):t for t in tasks}
        for future in as_completed(pending):
            e,d,_=pending[future]
            try:receipts.append(future.result())
            except Exception as exc:failures.append(dict(id=e['id'],forecast_index=d,error=repr(exc),traceback=traceback.format_exc()))
            io.write(out/f'progress_fraction{fraction:.2f}.json',dict(completed=len(receipts),total=len(tasks),failed_tasks=failures,receipts=receipts))
            print(f"fraction={fraction:g} {len(receipts)+len(failures)}/{len(tasks)} failures={len(failures)}",flush=True)
    result=dict(fraction=fraction,total_forecasts=len(tasks),completed_forecasts=len(receipts),failed_tasks=failures,
        quote_exceptions=sum(r.get('quote_exceptions',0) for r in receipts),
        execution_failed_rows=sum(r['failed_rows'] for r in receipts),
        old_nonzero_feasible=sum(r['old_nonzero_feasible'] for r in receipts),
        symmetric_nonzero_feasible=sum(r['symmetric_nonzero_feasible'] for r in receipts),receipts=receipts)
    io.write(out/f'complete_fraction{fraction:.2f}.json',result);return result


def run(out):
    out=Path(out);p=io.read(out/'protocol.json')
    for ref in p['sources']:checked(ref)
    if (out/'lock.json').exists():raise RuntimeError('Review existing run lock')
    io.write(out/'lock.json',dict(pid=os.getpid(),created_utc=utc(),protocol=io.ref(out/'protocol.json')))
    try:
        main=dispatch(out,.1)
        threshold_hit=bool(p['expansion']['automatic_only_for_full_16x4'] and main['old_nonzero_feasible']<p['expansion']['threshold'])
        blocked=bool(main['failed_tasks'] or main['quote_exceptions'])
        expand=threshold_hit and not blocked
        decision=dict(triggered=expand,method='old_gamma2',nonzero_verified_feasible=main['old_nonzero_feasible'],
            planned_forecasts=main['total_forecasts'],threshold=p['expansion']['threshold'],
            failed_tasks=len(main['failed_tasks']),quote_exceptions=main['quote_exceptions'],
            numeric_trigger_met=threshold_hit,expansion_blocked_by_implementation_failure=blocked,
            interpretation='Low nonzero procurement count, not proof of physical infeasibility. Implementation exceptions require review before any automatic expansion.',created_utc=utc())
        io.write(out/'expansion_decision.json',decision)
        extra=dispatch(out,.25) if expand else None
        unchanged=all(io.ref(ref['path'])['sha256']==ref['sha256'] for ref in p['sources'])
        receipt=dict(status='complete',source_unchanged=unchanged,main=io.ref(out/'complete_fraction0.10.json'),
            expansion=None if extra is None else io.ref(out/'complete_fraction0.25.json'),decision=decision)
        io.write(out/'complete.json',receipt);return receipt
    finally:(out/'lock.json').unlink(missing_ok=True)


def main():
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument('--index',type=Path);a.add_argument('--ac-index',type=Path);a.add_argument('--out',type=Path,required=True)
    a.add_argument('--prepare-only',action='store_true');a.add_argument('--resume',action='store_true');a.add_argument('--inventory-only',action='store_true')
    a.add_argument('--forecasts',type=int,default=4);a.add_argument('--repeats',type=int,default=8)
    a.add_argument('--seconds',type=float,default=20.);a.add_argument('--workers',type=int,default=4);a.add_argument('--seed',type=int,default=20261114)
    a.add_argument('--limit',type=int);args=a.parse_args()
    if args.inventory_only:
        result=inventory(args.index,args.ac_index,args.out,seed=args.seed);print(dict(status=result['status'],inputs=result['inputs']));return
    if not args.resume:
        prepare(args.index,args.ac_index,args.out,seed=args.seed,forecasts=args.forecasts,repeats=args.repeats,
                seconds=args.seconds,workers=args.workers,limit=args.limit)
    if not args.prepare_only:print(run(args.out))


if __name__=='__main__':main()
