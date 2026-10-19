"""Independent read-only AC/refit/cost audit; no optimizer or science imports.

Usage: python refit_cost_audit.py --root outputs/ev_ac_repair_development_v3
 --phases development --source research_experiments/mobile_ev_mle --output audit.json
"""
from __future__ import annotations
import argparse
import ast
from collections import Counter
import gzip
from hashlib import sha256
import json
import os
from pathlib import Path
from time import perf_counter

for _name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_name]='1'
import numpy as np


def read(path):
    p=Path(path)
    return json.loads(gzip.decompress(p.read_bytes()) if p.suffix=='.gz'
                      else p.read_text(encoding='utf-8'))


class Checks:
    def __init__(self):
        self.count=0;self.errors=[];self.maxima={};self.counts=Counter()
    def yes(self,condition,label,**evidence):
        self.count+=1
        if not bool(condition):self.errors.append(dict(label=label,**evidence))
    def close(self,a,b,label,atol=2e-7,rtol=2e-8):
        a,b=np.asarray(a,float),np.asarray(b,float)
        good=(a.shape==b.shape and np.isfinite(a).all() and np.isfinite(b).all())
        error=float(np.max(abs(a-b),initial=0)) if good else None
        if error is not None:self.maxima[label]=max(self.maxima.get(label,0.),error)
        self.yes(good and np.allclose(a,b,atol=atol,rtol=rtol),label,error=error)


def tree(record):
    n=int(record['n']);family={frozenset(range(n)),*(frozenset((j,)) for j in range(n))}
    family.update(frozenset(s) for s in record['supports'])
    order=sorted(family,key=lambda s:(-len(s),tuple(sorted(s))))
    parents=[-1 if len(s)==n else min((i for i,a in enumerate(order) if s<a),
                                   key=lambda i:len(order[i])) for s in order]
    terminals=[order.index(frozenset((j,))) for j in range(n)]
    return n,parents,terminals


def physical(desc,state,p,q,root):
    n,parents,terminals=tree(desc);r=np.asarray(state['r']);x=np.asarray(state['x'])
    m=len(parents);T=len(p)
    load=np.zeros((T,m),complex)
    load[:,0]=np.asarray(state['source_p'])+1j*np.asarray(state['source_q'])
    load[:,terminals]+=p+1j*q
    voltage=np.broadcast_to(root[:,None],load.shape).astype(complex).copy();z=r+1j*x
    for _ in range(1200):
        current=np.conj(load/voltage)
        for e in range(m-1,-1,-1):
            if parents[e]>=0:current[:,parents[e]]+=current[:,e]
        update=np.empty_like(voltage)
        for e,parent in enumerate(parents):
            update[:,e]=(root if parent<0 else update[:,parent])-z[e]*current[:,e]
        if np.max(abs(update-voltage))<1e-13:
            voltage=update;break
        voltage=update
    else:raise ValueError('independent sweep did not converge')
    current=np.conj(load/voltage)
    for e in range(m-1,-1,-1):
        if parents[e]>=0:current[:,parents[e]]+=current[:,e]
    master=root*np.conj(current[:,0])
    equation=max(float(np.max(abs(voltage[:,e]-(root if parent<0 else voltage[:,parent])
                      +z[e]*current[:,e]))) for e,parent in enumerate(parents))
    balance=float(np.max(abs(master-load.sum(1)-(abs(current)**2*z).sum(1))))
    return dict(v=abs(voltage[:,terminals]),master_p=master.real,master_q=master.imag,
                voltage=abs(voltage),equation=equation,balance=balance)


def combined(blocks,meters):
    obs={k:np.concatenate([np.asarray(b[k],float) for b in blocks])
         for k in ('p','q','v','root_v','master_p','master_q')}
    meter={}
    for k in ('p','q','v','master_p','master_q'):
        meter[k+'_sigma']=np.concatenate([np.broadcast_to(np.asarray(m[k+'_sigma'],float),
            np.shape(b[k])) for b,m in zip(blocks,meters)])
    for k in ('p','q','v'):
        meter['bias_'+k]=np.concatenate([np.broadcast_to(np.asarray(m['bias_'+k],float),
            np.shape(b[k])) for b,m in zip(blocks,meters)])
    return obs,meter


def replay(desc,state,obs,meter,options,check):
    n,parents,_=tree(desc);T=len(obs['p']);m=len(parents)
    r=np.asarray(state['r'],float);x=np.asarray(state['x'],float)
    check.yes(r.shape==x.shape==(m,),'line_state_dimension')
    check.yes(len(state['source_p'])==len(state['source_q'])==T,'source_PQ_time_dimension')
    for k,v in (('r',r),('x',x)):
        lo,hi=options[k+'_bounds']
        check.yes(np.all(v>=np.asarray(lo)-1e-10) and np.all(v<=np.asarray(hi)+1e-10),k+'_original_box')
    lo,hi=options['ratio_bounds']
    check.yes(np.all(x/r>=lo-1e-10) and np.all(x/r<=hi+1e-10),'ratio_original_cone')
    for k in ('source_p','source_q'):
        lo,hi=options[k+'_bounds'];s=np.asarray(state[k],float)
        check.yes(np.all(s>=lo-1e-10) and np.all(s<=hi+1e-10),k+'_original_box')
    phy=physical(desc,state,obs['p']-meter['bias_p'],obs['q']-meter['bias_q'],obs['root_v'])
    check.yes(phy['equation']<2e-10,'independent_AC_equation',error=phy['equation'])
    check.yes(phy['balance']<2e-10,'independent_power_balance',error=phy['balance'])
    check.maxima['independent_AC_equation']=max(check.maxima.get('independent_AC_equation',0),phy['equation'])
    check.maxima['independent_power_balance']=max(check.maxima.get('independent_power_balance',0),phy['balance'])
    lo,hi=options.get('voltage_bounds',(.85,1.15))
    check.yes(phy['voltage'].min()>=lo-1e-10 and phy['voltage'].max()<=hi+1e-10,'all_internal_voltage_domain')
    residual=np.c_[
        (phy['v']+meter['bias_v']-obs['v'])/meter['v_sigma'],
        (phy['master_p']-obs['master_p'])/meter['master_p_sigma'],
        (phy['master_q']-obs['master_q'])/meter['master_q_sigma']]
    return residual


def score_audit(desc,state,score,obs,meter,options,check,history_T):
    residual=replay(desc,state,obs,meter,options,check)
    loss=float(np.sum(residual**2));check.close(loss,score['loss'],'SSE_replay')
    normalizer=sum(float(np.log(2*np.pi*meter[k+'_sigma']**2).sum())
                   for k in ('v','master_p','master_q'))
    check.close(normalizer,score['normalizer'],'normalizer_replay')
    check.close(-.5*(loss+normalizer),score['loglik'],'conditional_loglik_replay')
    check.yes(score.get('usable'),'saved_score_usable')
    for label,slc in (('history',slice(0,history_T)),('future',slice(history_T,None))):
        if label not in score.get('blocks',{}):continue
        part=residual[slc];block=score['blocks'][label]
        check.close(float(np.sum(part**2)),block['loss'],'block_SSE_replay')
        check.close(part,block['residual'],'whitened_residual_replay')
        for k,cols in (('v',slice(0,-2)),('master_p',-2),('master_q',-1)):
            check.close(float(np.sum(part[:,cols]**2)),block['channel_loss'][k],
                        'channel_SSE_replay')
    check.counts['states_replayed']+=1


def optimizer_audit(fit,check):
    if not fit:return
    check.counts['optimizer_objects']+=1
    if fit.get('success'):
        check.yes(fit.get('status',0)>0,'success_requires_positive_solver_status')
        check.yes(fit.get('nfev',0)>0,'successful_fit_nfev_recorded')
    else:check.counts['unsuccessful_optimizer_objects']+=1
    diag=fit.get('diagnostics',{})
    check.yes(not diag.get('global_optimality_certified',False),'no_global_MLE_claim')
    for start in diag.get('restarts',[]):
        check.counts['local_solver_restarts']+=1
        if not start.get('success'):check.counts['failed_local_solver_restarts']+=1


def operation_audit(plan,cfg,check):
    d=plan['decoded'];u=np.asarray(plan['power']);ch=np.asarray(d['charge_kw']);dis=np.asarray(d['discharge_kw'])
    check.close(ch-dis,u,'net_EV_control_identity',atol=2e-7)
    check.yes(np.min(ch)>=-2e-7 and np.min(dis)>=-2e-7,'nonnegative_charge_discharge')
    check.yes(np.max(ch)<=cfg['p_charge_max_kw']+2e-7 and
              np.max(dis)<=cfg['p_discharge_max_kw']+2e-7,'charger_power_cap')
    check.yes(np.max(np.minimum(ch,dis))<2e-7,'charge_discharge_mutex')
    T,n=u.shape;connected=np.zeros((T,n),bool);drive=np.zeros(T);arcs=d['selected_arcs']
    time=0;site=cfg['initial_site'];is_connected=cfg['initial_connected'];moves=0
    for arc in arcs:
        a,b=arc['departure'],arc['arrival']
        check.yes(a==time and arc['origin']==site and arc['origin_connected']==is_connected,
                  'time_expanded_route_flow')
        check.yes(0<=a<b<=T,'route_time_order')
        if arc['kind']=='connected':connected[a:b,site]=True
        drive[b-1]+=arc['driving_energy_kwh']
        moves+=int(arc['origin']!=arc['destination'])
        time=b;site=arc['destination'];is_connected=arc['destination_connected']
    check.yes(time==T,'route_reaches_horizon')
    check.yes(np.max(abs(u[~connected]),initial=0)<2e-7,'only_connected_site_has_power')
    energy=np.r_[cfg['initial_soc']*cfg['battery_capacity_kwh'],
        cfg['initial_soc']*cfg['battery_capacity_kwh']+np.cumsum(cfg['dt_hours']*
        (cfg['eta_charge']*ch.sum(1)-dis.sum(1)/cfg['eta_discharge'])-drive)]
    check.close(energy,d['energy_kwh'],'battery_energy_replay',atol=2e-7)
    check.yes(abs(energy[-1]-cfg['terminal_energy_kwh'])<2e-7,'same_terminal_battery_task')
    check.yes(energy.min()>=cfg['soc_min']*cfg['battery_capacity_kwh']-2e-7 and
              energy.max()<=cfg['soc_max']*cfg['battery_capacity_kwh']+2e-7,'SOC_range')
    throughput=float(cfg['dt_hours']*(ch+dis).sum())
    check.yes(throughput<=cfg['max_throughput_kwh']+2e-7,'common_throughput_cap')
    check.yes(moves<=cfg['max_moves'],'mobility_optional_move_limit')
    check.yes(sum(a['driver_cost'] for a in arcs)<=cfg['driver_budget']+2e-7,'common_driver_budget')
    check.yes(plan['physicalcheck']['feasible'],'saved_operation_feasible')
    check.counts['operations_replayed']+=1
    return dict(moves=moves,discharge=bool(dis.max()>1e-6),throughput_kwh=throughput,
                grid_energy_kwh=float(cfg['dt_hours']*u.sum()),driver_cost=d['driver_cost'])


def audit_history(record,check):
    started=perf_counter();names=record.get('candidate_names',[]);candidate={r['name']:r for r in record.get('candidate_structures',[])}
    result=dict(spec=record['spec'],status=record.get('status'),failures=record.get('failures',{}),
                planned_methods=sorted(record.get('plans',{})),costs=record.get('costs',{}))
    if not names:
        check.counts['whole_history_failure']+=1
        check.yes(bool(record.get('failures')),'history_failure_explanation_preserved')
        return result
    check.yes(set(candidate)==set(names),'full_unfiltered_candidate_descriptions')
    check.yes(record['true_name'] in names,'candidate_bank_includes_truth_by_declared_simulation')
    if 'code_initial_sha' in record:
        changes=[n for n,h in record['code_initial_sha'].items() if record.get('code_end_sha',{}).get(n)!=h]
        result['source_changes_during_run']=changes
        check.yes(not [n for n in changes if n.endswith('.py')],'scientific_python_unchanged_during_run')
    blocks=[record['observed_training'],record['observed_evidence']]
    meters=[record['train_meter'],record['evidence_meter']]
    D,dm=combined(blocks,meters);H=len(D['p']);options=record['refit_options']
    history=record['refit_history'];result['history_refit_complete']=history['complete']
    for name,row in history['candidates'].items():
        optimizer_audit(row.get('optimizer'),check)
        if 'state' in row:score_audit(candidate[name],row['state'],row,D,dm,options,check,H)
    result['operations']={name:operation_audit(p,record['operation_config'],check)
                          for name,p in record.get('plans',{}).items()}
    sq=record.get('plans',{}).get('SQMD');bc=record.get('plans',{}).get('BCMD')
    if sq and bc:
        check.yes(sq['domain']==bc['domain']=='mobile_bidir','same_MILP_operation_domain')
        result['BC_same_action_as_square']=sq['action_key']==bc['action_key']
    for name,log in record.get('planning_logs',{}).get('square',{}).items():
        check.yes(log.get('inherited_Q_not_decreased'),'square_parent_plan_not_decreased')
    for name,log in record.get('planning_logs',{}).get('BC',{}).items():
        check.yes(log.get('inheritance_not_worse'),'Bayes_parent_score_not_decreased')
    result['MILP_events']=[]
    for domain,log in record.get('planning_logs',{}).get('BC',{}).items():
        for event in log.get('events',[]):
            mip=event['milp'];check.counts['BC_MILP_proposals']+=1
            check.counts['BC_MILP_accepted']+=int(event.get('accepted',False))
            check.counts['BC_MILP_reported_success']+=int(mip.get('success',False))
            check.yes(not mip.get('full_AC_optimality_claim',False),'no_full_AC_MILP_optimality_claim')
            inc,bound=mip.get('solver_incumbent'),mip.get('solver_best_bound')
            if inc is not None and bound is not None:
                check.yes(bound<=inc+1e-7*(1+abs(inc)),'min_surrogate_best_bound_orientation')
            if not mip.get('success'):check.counts['BC_MILP_failed_proposals']+=1
            result['MILP_events'].append(dict(domain=domain,accepted=event.get('accepted'),
                **{k:mip.get(k) for k in ('status','stop_reason','success','failure',
                    'solver_relative_gap','solver_incumbent','solver_best_bound','solver_seconds',
                    'build_seconds','proxy_certificate','proxy_uniform_error')}))
    refit_time=0.;prop_time=0.;risk_time=0.;prop_states=0;future_counter=Counter();slot_count=0
    for key,action in record.get('actions',{}).items():
        prop=action['propagation'];prop_time+=prop.get('seconds',0);prop_states+=prop.get('diagnostics',{}).get('AC_states',0)
        risk_time+=sum(action.get('declared_risk_seconds',{}).values())
        for regime,actual in action['regimes'].items():
            inputs=actual['truth_inputs_evaluation_only'];fm=record['future_meter']
            totalp=np.asarray(inputs['p'])+np.asarray(action['power'])/record['provenance']['base_kw']
            totalq=np.asarray(inputs['q'])
            truth_state=dict(record['true_lines_evaluation_only'],
                source_p=inputs['source_p'],source_q=inputs['source_q'])
            truephy=physical(candidate[record['true_name']],truth_state,totalp,totalq,
                             np.asarray(inputs['root_v']))
            truephy.update(p=totalp,q=totalq)
            white_mean=[]
            for channel in ('p','q','v','master_p','master_q'):
                value=truephy[channel]
                if channel in ('p','q','v'):value=value+np.asarray(fm['bias_'+channel])
                white_mean.append((value/np.asarray(fm[channel+'_sigma'])).ravel())
            check.close(np.concatenate(white_mean),actual['clean'],
                        'independent_actual_AC_EV_added_once',atol=2e-7)
            check.yes(truephy['equation']<2e-10 and truephy['balance']<2e-10,
                      'independent_actual_AC_equations')
            check.counts['actual_trajectories_replayed']+=1
            check.close(np.asarray(actual['values'])-np.asarray(actual['clean']),
                        record['actual_white_noise'],'paired_meter_noise_across_plans_regimes',atol=2e-10)
            for noise_idx,fit in enumerate(actual['refits']+[actual.get('clean_refit',{})]):
                if fit.get('status')=='not_run':continue
                slot_count+=1;refit_time+=fit.get('seconds',0);future_counter.update(fit.get('counters',{}))
                if not fit.get('complete'):
                    check.counts['failed_refit_tasks']+=1
                    check.yes(fit.get('keep_all_on_failure'),'failed_refit_keep_all')
                    check.yes(set(fit.get('retained_on_failure',[]))==set(names),'failure_preserves_candidate_bank')
                check.yes(set(fit.get('comparisons',{}))==set(names),'every_candidate_attempted')
                E=(actual['observations'][noise_idx] if noise_idx<len(actual['refits'])
                   else actual['clean_observations'])
                obs,met=combined(blocks+[E],meters+[record['future_meter']])
                for name,row in fit.get('comparisons',{}).items():
                    fixed=row.get('fixed');joint=row.get('joint')
                    if fixed and joint:
                        check.yes(joint['loss']<=fixed['loss']+1e-8*(1+abs(fixed['loss'])),
                                  'retained_joint_no_worse_than_fixed')
                        check.close(fixed['state']['r'],history['candidates'][name]['state']['r'],'fixed_history_R_identity')
                        check.close(fixed['state']['x'],history['candidates'][name]['state']['x'],'fixed_history_X_identity')
                        for route,score in (('fixed',fixed),('joint',joint)):
                            score_audit(candidate[name],score['state'],score,obs,met,options,check,H)
                            check.yes(not score.get('global_optimality_certified',False),'no_global_certificate')
                            check.counts[route+'_retained_states']+=1
                        if joint['success']:
                            good=[s['score']['loss'] for s in row['starts'] if s.get('success')]
                            check.yes(bool(good) and min(good)<=joint['loss']+1e-8*(1+abs(joint['loss'])),
                                      'joint_success_supported_by_converged_named_start')
                    for start in row.get('starts',[]):
                        optimizer_audit(start.get('optimizer'),check)
                        for f in start.get('source_fits',{}).values():
                            if isinstance(f,dict):optimizer_audit(f,check)
                        if start.get('witness') and start['witness'].get('usable'):
                            score_audit(candidate[name],start['witness_state'],start['witness'],obs,met,options,check,H)
                        if start.get('score') and start['score'].get('usable'):
                            score_audit(candidate[name],start['attained_state'],start['score'],obs,met,options,check,H)
    cost=record['costs'];check.close(refit_time,cost['refit_future_seconds'],'future_refit_cost_sum',atol=2e-6)
    check.close(prop_time,cost['propagation_seconds'],'propagation_cost_sum',atol=2e-6)
    check.close(risk_time,cost['declared_risk_seconds'],'declared_risk_cost_sum',atol=2e-6)
    check.yes(prop_states==cost['propagation_AC_states'],'propagation_AC_state_count')
    time_parts=dict(training=record['training_result']['seconds'],planning=cost['planning_seconds'],
        history=cost['conditional_history_seconds'],future_refit=refit_time,propagation=prop_time,
        declared_risk=risk_time,hash_begin=cost['source_hash_begin_seconds'],hash_end=cost['source_hash_end_seconds'],
        extra_center_after_planning=cost.get('final_center_objective',{}).get('extra_seconds_after_planning',0.))
    accounted=sum(time_parts.values());elapsed=record['seconds']
    check.yes(accounted<=elapsed+1e-3,'exclusive_cost_parts_do_not_double_count',accounted=accounted,elapsed=elapsed)
    result.update(refit_tasks=slot_count,future_fit_counters=dict(future_counter),
        history_fit_counters=history['counters'],exclusive_cost_parts=time_parts,
        unitemized_worker_seconds=elapsed-accounted,
        cold_BC_design_attempt_seconds=time_parts['training']+time_parts['planning'],
        BC_plan_available='BCMD' in record.get('plans',{}),
        square_design_seconds=cost['planning']['square'],
        history_cache_charged_once=True,audit_seconds=perf_counter()-started)
    return result


def source_audit(directory,check):
    source=Path(directory);text=(source/'ev_ac_repair_study.py').read_text(encoding='utf-8')
    module=ast.parse(text);funcs={n.name:n for n in module.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
    selector=funcs['_planning'];used={n.id for n in ast.walk(selector) if isinstance(n,ast.Name)}
    check.yes(not ({'case','truth','true_index','replay','white'}&used),'selector_no_truth_future_container_name')
    args=[a.arg for a in selector.args.args]
    check.yes(args==['observed','fits','posterior_options','seed','progress'],'selector_explicit_observed_only_API')
    case_tree=ast.parse((source/'ev_bayes_response_cases.py').read_text(encoding='utf-8'))
    entry=next(n for n in case_tree.body if isinstance(n,ast.FunctionDef) and n.name=='planning_inputs')
    ret=next(n for n in ast.walk(entry) if isinstance(n,ast.Return))
    allowed={k.arg for k in ret.value.keywords}
    check.yes(allowed=={'training','evidence','train_meter','evidence_meter','future_meter','candidates','horizon','base_kw'},
              'observed_sanitizer_fields')
    return dict(scope='AST data-boundary check, not a general noninterference proof',
        selector_arguments=args,sanitized_fields=sorted(allowed),
        source_sha256={p.name:sha256(p.read_bytes()).hexdigest() for p in
            (source/'ev_ac_repair_study.py',source/'ev_ac_repair_refit.py',
             source/'ev_ac_repair_analysis.py',source/'ev_ac_repair_control.py')})


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True)
    parser.add_argument('--phases',nargs='+',default=['development'])
    parser.add_argument('--source',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--allow-incomplete',action='store_true');args=parser.parse_args()
    began=perf_counter();script_sha=sha256(Path(__file__).read_bytes()).hexdigest()
    root=Path(args.root);check=Checks();manifest=read(root/'freeze.json') if (root/'freeze.json').exists() else None
    static=source_audit(args.source,check);records=[];inputs=[];phase_seeds={};ledgers={}
    for phase in args.phases:
        paths=sorted((root/phase).glob('history_*.json.gz'));seeds=[];ids=[];statuses=Counter()
        expected=(manifest or {}).get('specifications',{}).get(phase)
        if expected is not None:
            if not args.allow_incomplete:check.yes(len(paths)==len(expected),'all_expected_histories_present',phase=phase,present=len(paths),expected=len(expected))
            else:check.yes(len(paths)<=len(expected),'partial_phase_not_duplicated')
        for path in paths:
            value=read(path);spec=value['spec'];seeds.append(spec['seed']);ids.append(spec['index']);statuses[value.get('status')]+=1
            if expected is not None:check.yes(spec==expected[spec['index']],'frozen_spec_identity')
            if manifest and value.get('code_initial_sha'):
                check.yes(all(manifest['source_sha256'].get(k)==v for k,v in value['code_initial_sha'].items()),'raw_science_matches_frozen_source')
            try:
                records.append(audit_history(value,check))
            except Exception as exc:
                check.yes(False,'history_audit_exception',phase=phase,index=spec['index'],
                    error=type(exc).__name__+': '+str(exc))
                records.append(dict(spec=spec,status=value.get('status'),
                    failures=value.get('failures',{}),audit_failure=type(exc).__name__+': '+str(exc)))
            inputs.append(dict(path=str(path.resolve()),bytes=path.stat().st_size,sha256=sha256(path.read_bytes()).hexdigest()))
        check.yes(len(ids)==len(set(ids)),'unique_history_indices',phase=phase)
        check.yes(len(seeds)==len(set(seeds)),'unique_history_seeds',phase=phase)
        phase_seeds[phase]=set(seeds);ledgers[phase]=dict(total_records=len(paths),status_counts=dict(statuses),
            expected=len(expected) if expected is not None else None,
            missing_indices=sorted(set(s['index'] for s in expected)-set(ids)) if expected else [])
    if manifest:
        designed={p:{s['seed'] for s in rows} for p,rows in manifest.get('specifications',{}).items()}
        for a in designed:
            for b in designed:
                if a<b:check.yes(not designed[a]&designed[b],'independent_phase_seed_design',phases=[a,b])
    result=dict(success=not check.errors,checks=check.count,counts=dict(check.counts),maxima=check.maxima,
        errors=check.errors,input_files=inputs,phase_ledgers=ledgers,source_boundary=static,histories=records,
        audit_script_sha256=script_sha,
        audit_script_unchanged=script_sha==sha256(Path(__file__).read_bytes()).hexdigest(),
        audit_seconds=perf_counter()-began,audit_cost_separate_from_scientific_run=True,
        source_and_raw_records_modified=False,new_scientific_samples_generated=0,
        limitations=['conditional direct-PQ SSE differs from latent-PQ predictive density',
            'joint attained SSE is a local upper bound on profile minimum, not a global MLE',
            'SSE argmin does not correct unequal topology parameter dimensions',
            'failure-marked inference is unavailable in complete-case summaries; operational rule retains all candidates',
            'calibration and bootstrap describe the declared synthetic strata, not field reliability',
            'cold BC planning cost includes training, inference, square and inherited plans',
            'same feasible operation domain does not imply equal total computing time for BC and square planning',
            'Bayes MILP solver gaps certify only its frozen surrogate; retained parent score is the improvement check',
            'cached historical fits and posterior samples are charged once per history, not free or per future repeatedly'])
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(dict(success=result['success'],checks=check.count,counts=result['counts'],
        errors=len(check.errors),seconds=result['audit_seconds'],output=str(output)),ensure_ascii=False))
    if check.errors:raise SystemExit(1)


if __name__=='__main__':main()
