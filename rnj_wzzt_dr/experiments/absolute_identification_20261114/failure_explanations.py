"""Offline attribution of ACTUAL native-bank recovery gains and losses.

For each change, the exact winning true/wrong parameter pair is fixed before
and after the action. No simulator or optimizer is called by this module.
"""
from __future__ import annotations

import argparse
from collections import Counter,defaultdict
from pathlib import Path
import traceback

import numpy as np

from ... import io
from ...dr.selection_mse import choose_mse
from ..customer_flexibility_20261113.model import CustomerCase
from .data import load_entry


def components(Hi,Hj,history_i,history_j,sigma2,forecast_power,true_power,true_y,observed_power,observed_y):
    """Independent raw-SSE contrast and physical J/bias/meter decomposition."""
    Hi,Hj=np.asarray(Hi,float),np.asarray(Hj,float)
    P,Y,Pm,Ym=map(lambda a:np.asarray(a,float),(true_power,true_y,observed_power,observed_y))
    D=Hi-Hj
    d=P@D.T;residual=Y-P@Hi.T
    history=float((history_j-history_i)/(2*sigma2))
    J=float(np.sum(d*d)/(2*sigma2))
    bias=float(np.sum(residual*d)/sigma2)
    true_i=float(np.sum((Y-P@Hi.T)**2));true_j=float(np.sum((Y-P@Hj.T)**2))
    meter_i=float(np.sum((Ym-Pm@Hi.T)**2));meter_j=float(np.sum((Ym-Pm@Hj.T)**2))
    clean=history+(true_j-true_i)/(2*sigma2)
    observed=history+(meter_j-meter_i)/(2*sigma2)
    meter=((meter_j-meter_i)-(true_j-true_i))/(2*sigma2)
    return dict(history_logodds=history,forecast_J=float(np.sum((forecast_power@D.T)**2)/(2*sigma2)),
        actual_physical_J=J,physical_bias=bias,meter_difference=float(meter),
        clean_raw_SSE_contrast=float(clean),observed_raw_SSE_contrast=float(observed),
        expansion_error=float(clean-history-J-bias),
        observation_identity_error=float(observed-history-J-bias-meter))


def read_observation(path,true_root):
    with np.load(path) as z:
        observed_power=np.c_[z['P'],z['Q']]
        observed_y=z['root'][:,None]**2-z['V']**2
        if 'Y' in z and not np.allclose(observed_y,z['Y'],rtol=1e-12,atol=1e-13):
            raise ValueError('Stored observed Y differs from measured root²-V²')
        true_power=np.c_[z['true_P'],z['true_Q']]
        true_y=true_root[:,None]**2-z['true_V']**2
    return dict(observed_power=observed_power,observed_y=observed_y,true_power=true_power,true_y=true_y)


def native_winner(score,observation):
    bank=score['native_bank'];power,y=observation['observed_power'],observation['observed_y']
    loss=np.array([p['history_sse']+np.sum((y-power@np.asarray(p['H']).T)**2) for p in bank])
    chosen=choose_mse(loss/(score['cells']+y.size),[p['internal'] for p in bank],
        K=1/score['reference_mse'],epsilon_scaled=score['validation_tie'])['selected']
    return int(chosen),loss


def action_breakdown(case,u,eta):
    u=np.asarray(u,float);actual=u*np.asarray(eta)[None]
    kinds={}
    for kind in sorted({r['kind'] for r in case.resources}):
        mask=np.array([r['kind']==kind for r in case.resources])
        request,delivered=u[:,mask],actual[:,mask]
        def fee(x):return float(case.dt*np.sum(case.fee_plus[:,mask]*np.maximum(x,0)+case.fee_minus[:,mask]*np.maximum(-x,0)))
        kinds[kind]=dict(requested_abs_kwh=float(case.dt*np.abs(request).sum()),
            delivered_abs_kwh=float(case.dt*np.abs(delivered).sum()),
            delivered_positive_kwh=float(case.dt*np.maximum(delivered,0).sum()),
            delivered_negative_kwh=float(case.dt*np.maximum(-delivered,0).sum()),
            delivered_net_consumption_kwh=float(case.dt*delivered.sum()),
            planned_fee_yuan=fee(request),settled_fee_yuan=fee(delivered))
    return dict(by_kind=kinds,planned_fee_yuan=case.fee(u),settled_fee_yuan=case.fee(actual),
        delivered_abs_kwh=float(case.dt*np.abs(actual).sum()),
        sign_convention='Positive means increased net consumption; PV positive action is curtailment. Shifted energy counts both directions in absolute kWh.')


def summarize(events,comparison_counts):
    grouped=defaultdict(list)
    for r in events:grouped[(r['physics'],r['strategy'],r['event'])].append(r)
    groups={}
    for key,rows in sorted(grouped.items()):
        good=[r for r in rows if r['explanation_status']=='complete']
        fields=['forecast_J','actual_physical_J','physical_bias','meter_difference','observed_raw_SSE_contrast']
        means={f:None if not good else float(np.mean([r['delta'][f] for r in good])) for f in fields}
        groups['/'.join(key)]=dict(events=len(rows),decomposed=len(good),failed_or_unavailable=len(rows)-len(good),
            mean_delta=means,cases=dict(Counter(r['case'] for r in rows)),roles=dict(Counter(r['role'] for r in rows)),
            history_partitions=dict(Counter(r['history_partition'] for r in rows)),
            negative_component_counts={f:sum(r['delta'][f]<-1e-9 for r in good) for f in ('actual_physical_J','physical_bias','meter_difference')},
            largest_negative_component=dict(Counter(r['largest_negative_component'] for r in good)),
            identity_max_error=max([abs(r['delta_identity_error']) for r in good],default=None),
            mean_settled_fee=None if not good else float(np.mean([r['action']['settled_fee_yuan'] for r in good])),
            mean_delivered_abs_kwh=None if not good else float(np.mean([r['action']['delivered_abs_kwh'] for r in good])),
            action_by_kind={kind:{field:None if not good else float(np.mean([r['action']['by_kind'].get(kind,{}).get(field,0.) for r in good]))
                for field in ('delivered_abs_kwh','delivered_net_consumption_kwh','settled_fee_yuan')}
                for kind in ('pv','ev','thermal')})
    return dict(change_events=len(events),unique_future_paths_with_changes=len({(r['id'],r['day']) for r in events}),
        event_counts=dict(Counter(r['event'] for r in events)),explanation_statuses=dict(Counter(r['explanation_status'] for r in events)),
        comparisons=comparison_counts,groups=groups,
        interpretation='Algebraic attribution for an outcome-selected fixed native parameter pair; no causal independence of the components is asserted. Physical bias includes frozen-parameter and nonlinear-flow mismatch; meter term includes observed P/Q and squared-voltage effects.',
        pairing_scope='AC/linear and different strategies share future paths; change-event counts are not independent sample counts.')


def run(run_path,destination,*,physics=('AC','linear'),strategies=None):
    if not (run_path/'complete.json').exists():raise ValueError('Wait until the batch is complete, including failed tasks')
    if destination.exists():raise FileExistsError('Preserve completed attribution; use a new destination')
    destination.mkdir(parents=True)
    protocol=io.read(run_path/'protocol.json');entries=io.read(run_path/'inputs.json')
    modes=list(protocol['modes'])
    methods=[s for m in modes for s in (m,m+'_rotation')] if strategies is None else list(strategies)
    if any(s=='no_dfs' for s in methods):raise ValueError('The no-service baseline is the fixed reference')
    plan=dict(source=io.ref(__file__),run_protocol=io.ref(run_path/'protocol.json'),inputs=io.ref(run_path/'inputs.json'),
        completion=io.ref(run_path/'complete.json'),physics=list(physics),strategies=methods,new_AC_days=0,new_optimization_quotes=0,
        pair_rule='Loss: i=baseline actual native true winner, j=action actual native wrong winner. Gain: i=action actual native true winner, j=baseline actual native wrong winner. Use this identical pair in both conditions.',
        decomposition='observed contrast = historical contrast + physical J + physical residual projection + meter contribution; meter contribution is independently differenced raw-SSE contrasts.',
        failure_rule='Keep every native recovery change including execution-failure changes; missing physical or winner data gets an explicit unavailable explanation rather than being dropped.',
        future_paths_are_already_evaluated=True,private='Per-event parameter indices, margins, source paths, action details and observation refs remain private.')
    io.write(destination/'protocol.json',plan)
    events=[];counts=defaultdict(lambda:dict(planned_pairs=0,completed_pairs=0,gains=0,losses=0))
    context_cache={}
    for entry in entries:
        for day in range(protocol['days']):
            d=run_path/'days'/entry['id']/f'day{day:03d}'
            records=io.read(d/'results.json') if (d/'results.json').exists() else []
            table={(r['physics'],r['strategy']):r for r in records}
            needed=[]
            for phy in physics:
                before=table.get((phy,'no_dfs'),{})
                bcomplete=before.get('status')=='complete'
                bcorrect=bcomplete and bool(before.get('native_map_correct',False))
                for strategy in methods:
                    after=table.get((phy,strategy),{})
                    acomplete=after.get('status')=='complete'
                    acorrect=acomplete and bool(after.get('native_map_correct',False))
                    counter=counts[phy+'/'+strategy];counter['planned_pairs']+=1;counter['completed_pairs']+=int(bcomplete and acomplete)
                    if bcorrect==acorrect:continue
                    event='gain' if acorrect else 'loss';counter['gains' if acorrect else 'losses']+=1
                    row=dict(id=entry['id'],day=day,physics=phy,strategy=strategy,event=event,
                        case=entry['case'],role=entry['role'],history_partition=entry['history_partition'],
                        noise_voltage_sigma=entry['voltage_sigma'],profile=entry['physical_parameters']['profile'],
                        profile_level=entry['physical_parameters']['level'],historical_truth_covered=entry['truth_covered'],
                        before_native_correct=bool(bcorrect),after_native_correct=bool(acorrect),
                        before_status=before.get('status','missing'),after_status=after.get('status','missing'))
                    if not (bcomplete and acomplete):
                        row.update(explanation_status='unavailable_execution_failure');events.append(row)
                    else:needed.append((row,before,after))
            if not needed:continue
            try:
                seal=io.read(d/'seal.json')
                for name in ('planner','plans','protocol'):
                    ref=seal[name]
                    if io.ref(ref['path'])['sha256']!=ref['sha256']:raise ValueError('Sealed policy changed: '+name)
                if entry['id'] not in context_cache:
                    score,train,validation,truth=load_entry(entry)
                    case=CustomerCase(io.read(run_path/'resource_configs'/(entry['case']+'.json')),score['labels'])
                    context_cache[entry['id']]=(score,case,truth)
                score,case,truth=context_cache[entry['id']]
                with np.load(d/'background.npz') as bg:root=bg['root'].copy();eta=bg['response'].copy()
                with np.load(d/'planner.npz') as z:
                    forecast=z['forecast'].copy();controls={k:z[k].copy() for k in z.files if k!='forecast'}
                generation=io.read(d/'generation.json');base=float(generation['base_kw'])
                obs_cache={}
            except Exception as exc:
                for row,_,_ in needed:
                    row.update(explanation_status='input_audit_failed',error=repr(exc),traceback=traceback.format_exc());events.append(row)
                continue
            for row,before,after in needed:
                try:
                    pair_observations=[];winners=[];paths=[]
                    for strategy in ('no_dfs',row['strategy']):
                        key=(row['physics'],strategy);path=d/(strategy+'_'+row['physics']+'.npz');paths.append(path)
                        if key not in obs_cache:
                            observed=read_observation(path,root);chosen,losses=native_winner(score,observed)
                            obs_cache[key]=(observed,chosen,losses)
                        observed,chosen,losses=obs_cache[key];pair_observations.append(observed);winners.append(chosen)
                    if winners!=[int(before['native_selected_index']),int(after['native_selected_index'])]:
                        raise ArithmeticError('Recomputed raw-bank winner differs from saved native selection')
                    i,j=(winners[0],winners[1]) if row['event']=='loss' else (winners[1],winners[0])
                    bank=score['native_bank'];pi,pj=bank[i],bank[j]
                    if pi['topology_id']!=truth['truth_id'] or pj['topology_id']==truth['truth_id']:
                        raise ArithmeticError('The actual winning pair does not match recorded recovery change')
                    before_parts=components(pi['H'],pj['H'],pi['history_sse'],pj['history_sse'],score['sigma2'],
                        forecast+case.shift_pq_pu(controls['no_dfs'],base),**pair_observations[0])
                    after_parts=components(pi['H'],pj['H'],pi['history_sse'],pj['history_sse'],score['sigma2'],
                        forecast+case.shift_pq_pu(controls[row['strategy']],base),**pair_observations[1])
                    fields=('forecast_J','actual_physical_J','physical_bias','meter_difference','observed_raw_SSE_contrast')
                    delta={f:after_parts[f]-before_parts[f] for f in fields}
                    error=delta['observed_raw_SSE_contrast']-sum(delta[f] for f in ('actual_physical_J','physical_bias','meter_difference'))
                    if max(abs(error),abs(before_parts['observation_identity_error']),abs(after_parts['observation_identity_error']))>1e-6:
                        raise ArithmeticError('Fixed native-pair attribution identity failed')
                    action=action_breakdown(case,controls[row['strategy']],eta)
                    if abs(action['settled_fee_yuan']-after['settled_fee_yuan'])>1e-7:
                        raise ArithmeticError('Resource-type fee sum differs from saved settlement')
                    negatives={f:delta[f] for f in ('actual_physical_J','physical_bias','meter_difference') if delta[f]<-1e-9}
                    row.update(explanation_status='complete',true_native_index=i,wrong_native_index=j,
                        before_native_winner=winners[0],after_native_winner=winners[1],
                        true_topology_id=pi['topology_id'],wrong_topology_id=pj['topology_id'],
                        history_training_points=score['training_points'],history_validation_points=score['validation_points'],
                        before=before_parts,after=after_parts,delta=delta,delta_identity_error=float(error),action=action,
                        largest_negative_component=min(negatives,key=negatives.get) if negatives else 'none',
                        paired_observation_refs=[io.ref(p) for p in paths],background=io.ref(d/'background.npz'),seal=io.ref(d/'seal.json'))
                except Exception as exc:
                    row.update(explanation_status='decomposition_failed',error=repr(exc),traceback=traceback.format_exc())
                events.append(row)
    summary=summarize(events,dict(counts));summary.update(protocol=io.ref(destination/'protocol.json'))
    io.write(destination/'private_pairs.json',events);io.write(destination/'summary.json',summary)
    io.write(destination/'complete.json',dict(summary=io.ref(destination/'summary.json'),pairs=io.ref(destination/'private_pairs.json')))
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True);parser.add_argument('--destination',type=Path,required=True)
    parser.add_argument('--physics',default='AC,linear');parser.add_argument('--strategies')
    args=parser.parse_args()
    result=run(args.run.resolve(),args.destination.resolve(),physics=tuple(args.physics.split(',')),
        strategies=None if not args.strategies else args.strategies.split(','))
    print('native_changes',result['change_events'],'status',result['explanation_statuses'])
