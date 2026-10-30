"""Independent post-identification evaluation; imports no project solver code.

Truth is read only after the bounded execution has ended. Reuses saved fitted
points and LP receipts, never optimizes a true family or reruns missing records.
"""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
from datetime import datetime,timezone
import hashlib,json,pickle
from pathlib import Path
import numpy as np

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def family(point):
    if point is None:return set()
    r=point.get('r',point.get('r_values'));x=point.get('x',point.get('x_values'))
    return {tuple(sorted(s)) for s,a,b in zip(point['supports'],r,x,strict=True) if a>0 or b>0}
def matrix(point,n):
    R=np.zeros((n,n));X=np.zeros_like(R)
    for s,r,x in zip(point['supports'],point['r'],point['x'],strict=True):
        R[np.ix_(s,s)]+=r;X[np.ix_(s,s)]+=x
    return R,X
def arrays(scenes):
    p=np.concatenate([s['P_terminal'].to_numpy() for s in scenes])
    q=np.concatenate([s['Q_terminal'].to_numpy() for s in scenes])
    y=np.concatenate([s['V_root_observed'].to_numpy()[:,None]**2-s['V_terminal'].to_numpy()**2 for s in scenes])
    return p,q,y
def supports(value):return [list(s) for s in sorted(value,key=lambda s:(len(s),s))]
def loss_flags(point,target):
    f=family(point)
    return dict(errors=len(f^target),full=f==target,missed=supports(target-f),extra=supports(f-target),effective_supports=supports(f))
def _raw(data):
    chosen=data.get('selected')
    if not chosen:return None
    if 'selected_uncontracted' in chosen:return chosen['selected_uncontracted']
    index=chosen.get('selection_index')
    return data.get('points',[])[index] if index is not None else None
def block_trace(block,data,target):
    raw=_raw(data);final=data.get('selected');rf=family(raw);ff=family(final)
    points=data.get('points',[]);slot=any(block in {tuple(s) for s in r.get('family',[])} for r in data.get('LP_solves',[]))
    pool=block in {tuple(s) for s in data.get('proposal_pool',[])}
    saved_positive=any(block in family(p) for p in points)
    saved_true=[(i,p) for i,p in enumerate(points) if family(p)==target]
    selection='no_saved_exact_true_family'
    if saved_true:
        best=min(p['validation_mae'] for p in points)
        eligible=[(i,p) for i,p in saved_true if p['validation_mae']<=1.03*best+1e-12]
        if family(raw)==target:selection='raw_selected_exact_true_family'
        elif not eligible:selection='saved_true_family_outside_3pct_validation_band'
        elif len(raw['supports'])<min(len(p['supports']) for _,p in eligible):selection='3pct_simplicity_prefers_smaller_raw_support'
        else:selection='saved_true_family_eligible_but_validation_or_index_selected_another'
    path_states=[];trials=[]
    for name,path in data.get('paths',{}).items():
        states=[block in family(p) for p in path.get('path',[])]
        path_states.append(dict(name=name,block_positive_on_path=states,reason=path.get('reason')))
        for iteration,decision in enumerate(path.get('decisions',[])):
            current=decision['current'];threshold=1e-10+.001*current
            for trial in decision.get('candidates',[]):
                if tuple(trial['candidate'])==block:
                    gain=current-trial['objective']
                    trials.append(dict(path=name,iteration=iteration,gain=gain,threshold=threshold,
                        passes_accept_threshold=gain>threshold,best_gain=decision['gain'],full_sweep=decision['full_sweep']))
    if block in rf and block not in ff:stage='contract_removed_positive_true_block'
    elif block in ff:stage='retained'
    elif saved_true:stage='selection_did_not_return_saved_exact_true_family'
    elif saved_positive:stage='positive_block_seen_in_saved_fit_but_not_raw_selected'
    elif trials and all(not t['passes_accept_threshold'] for t in trials):stage='assessed_below_accept_threshold_at_saved_states'
    elif trials:stage='assessed_beneficial_candidate_not_returned'
    elif not slot and not pool:stage='no_support_slot_in_declared_pool_or_fitted_families'
    elif slot:stage='support_fitted_but_no_saved_positive_block_nonuniqueness_unresolved'
    else:stage='pool_contains_block_but_no_saved_trial_or_positive_fit'
    return dict(block=list(block),stage=stage,in_declared_pool=pool,in_any_LP_family=slot,
        in_raw=block in rf,in_final=block in ff,positive_in_saved_points=saved_positive,
        saved_exact_true_family_points=[i for i,_ in saved_true],selection_detail=selection,
        paths=path_states,trials=trials,
        absence_is_not_infeasibility_or_global_exclusion=True)

def main(argv=None):
    parser=argparse.ArgumentParser();parser.add_argument('--generation-root',type=Path,required=True)
    parser.add_argument('--execution-root',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--allow-truth-read',action='store_true');args=parser.parse_args(argv)
    if not args.allow_truth_read:raise ValueError('explicit post-identifier truth-read gate required')
    gen,run=args.generation_root.resolve(),args.execution_root.resolve()
    if (run/'run.lock').exists():raise ValueError('execution is still running; no truth read')
    summary=read(run/'summary.json');cfg=read(run/'protocol.json');records=read(run/'records.json')
    if summary['processed_records']!=summary['planned_records']:raise ValueError('incomplete batch cannot be silently audited as complete')
    if args.out.exists():raise FileExistsError('existing audit output retained')
    checks=[];fail=[]
    def check(name,ok,**detail):
        item=dict(name=name,pass_=bool(ok),**detail);checks.append(item)
        if not ok:fail.append(item)
    for group in ('source_sha256','input_sha256'):
        for p,h in cfg[group].items():check(group,sha(p)==h,path=p)
    # Input arrays are reconstructed independently, with all old rows retained.
    histories={r['id']:pickle.loads(Path(r['path']).read_bytes()) for r in cfg['histories']}
    snapshots={r['id']:r for r in cfg['snapshots']};data_by_sid={};maxtrain=maxval=0.;method_status=Counter();selected_count=0
    all_method_names=cfg['methods'];raw_point_count=0;max_raw_train=max_raw_val=0.
    for rec in records:
        if rec.get('result_path') is None:
            data_by_sid[rec['id']]=dict(status=rec['status'],methods={});continue
        result=read(rec['result_path']);data_by_sid[rec['id']]=result
        if 'applied_calibration' not in result:continue
        snapshot=snapshots[rec['id']];future=pickle.loads(Path(snapshot['path']).read_bytes());old=histories[snapshot['history_id']]
        train=arrays(old['training']+future['training']);valid=arrays(old['validation']+future['validation'])
        check('all_old_rows_retained',len(train[0])==480 and len(valid[0])==480,snapshot=rec['id'])
        for packet in (old,future):
            check('observed_only_metadata',packet['metadata'].get('observed_only') is True,snapshot=rec['id'])
            check('metadata_no_truth',not({'truth','true_family','true_K','p0','q0','R_true','X_true'}&set(packet['metadata'])),snapshot=rec['id'])
        current=read(result['current_state_path']);cal=result['applied_calibration']
        check('same_current_calibration',cal==current['calibration'],snapshot=rec['id'])
        for method,data in result['methods'].items():
            method_status[data['status']]+=1
            if data['status']!='complete':continue
            selected_count+=1;point=data['selected'];R,X=matrix(point,len(result['terminal_labels']))
            lt=float(np.abs(train[0]@R.T+train[1]@X.T-train[2]).mean())
            lv=float(np.abs(valid[0]@R.T+valid[1]@X.T-valid[2]).mean())
            dt=abs(lt-point['train_mae']);dv=abs(lv-point['validation_mae']);maxtrain=max(maxtrain,dt);maxval=max(maxval,dv)
            check('selected_train_L1',dt<=1e-12,snapshot=rec['id'],method=method,difference=dt)
            check('selected_validation_L1',dv<=1e-12,snapshot=rec['id'],method=method,difference=dv)
            check('original_coefficient_box',all(-1e-12<=v<=2+1e-12 for v in point['r']+point['x']),snapshot=rec['id'],method=method)
            points=data['points'];minimum=min(p['validation_mae'] for p in points)
            for index,rawpoint in enumerate(points):
                rr,xx=matrix(rawpoint,len(result['terminal_labels']))
                measured_train=float(np.abs(train[0]@rr.T+train[1]@xx.T-train[2]).mean())
                measured_val=float(np.abs(valid[0]@rr.T+valid[1]@xx.T-valid[2]).mean())
                delta_train=abs(measured_train-rawpoint['objective']);delta_val=abs(measured_val-rawpoint['validation_mae'])
                max_raw_train=max(max_raw_train,delta_train);max_raw_val=max(max_raw_val,delta_val);raw_point_count+=1
                check('raw_point_L1',delta_train<=1e-12 and delta_val<=1e-12,snapshot=rec['id'],method=method,point=index,
                      train_difference=delta_train,validation_difference=delta_val)
            eligible=[i for i,p in enumerate(points) if p['validation_mae']<=1.03*minimum+1e-12]
            chosen=min(eligible,key=lambda i:(len(points[i]['supports']),points[i]['validation_mae'],i))
            check('3pct_selection',chosen==point['selection_index'],snapshot=rec['id'],method=method)
            p=points[chosen];keep=[i for i,s in enumerate(p['supports']) if len(s)==1 or cal['a']*p['r'][i]+cal['b']*p['x'][i]+1e-10>=.16*cal['median_root_depth']]
            check('shared_contract_support_and_weights',point['supports']==[p['supports'][i] for i in keep] and point['r']==[p['r'][i] for i in keep] and point['x']==[p['x'][i] for i in keep],snapshot=rec['id'],method=method)
            if method=='current_root_augmented_rnj':
                n=len(result['terminal_labels']);canonical=lambda f:tuple(sorted(set(tuple(sorted(s)) for s in f),key=lambda s:(len(s),s)))
                expected=[canonical([*tree['family'],tuple(range(n))]) for tree in current['native_bank']]
                actual=[canonical(f) for f in data['augmented_native_families']]
                check('formal_rnj_root_same15_native_sources',len(expected)==15 and actual==expected,snapshot=rec['id'])
                check('formal_rnj_root_optional_nonnegative',data.get('full_root_positive_weight_required') is False
                    and data.get('coefficient_lower_bound')==0. and data.get('coefficient_upper_bound')==2.
                    and data.get('shared_cross_method_LP_cache') is False,snapshot=rec['id'])
    # Only now, after every saved identifier output has been inspected, read truth.
    truth=read(gen/'evaluator_truth.json');cases={r['id']:r for r in truth['cases']};evaluations=[]
    for mapping in truth['evaluations']:
        case=cases[mapping['case_id']];target={tuple(sorted(s)) for s in mapping['target_family']}
        result=data_by_sid.get(mapping['snapshot_id'],{});noise=case.get('noise_id',mapping.get('history_id'))
        row=dict(case_id=mapping['case_id'],snapshot_id=mapping['snapshot_id'],noise_id=noise,policy=mapping['policy'],
            conditional_state=mapping.get('conditional_state',False),actual_payment=mapping.get('actual_payment'),
            target_family=supports(target),methods={},snapshot_status=result.get('status','not_started'))
        discovery=result.get('discovery',{});row['discovery']=dict(status=discovery.get('status'),
            candidate_supports=discovery.get('candidate_supports',[]),direct_family=discovery.get('direct_family'),
            reasons=[dict(pair=d.get('pair'),status=d.get('status'),unclosed=d.get('observation_model_unclosed',[])) for d in discovery.get('diagnostics',[])])
        row['expanded_equals_current_pool']=result.get('expanded_pool')==result.get('current_pool') if 'expanded_pool' in result else None
        for method in all_method_names:
            data=result.get('methods',{}).get(method,dict(status='not_started_snapshot_unresolved'))
            m=dict(status=data['status'],all_scans_complete=data.get('all_scans_complete'),
                   LP_solves=len(data.get('LP_solves',[])))
            if data['status']=='complete':
                raw=_raw(data);final=data['selected'];m.update(raw=loss_flags(raw,target),final=loss_flags(final,target),
                    true_removed_by_contract=supports((family(raw)-family(final))&target),
                    false_removed_by_contract=supports((family(raw)-family(final))-target),
                    trace=[block_trace(s,data,target) for s in sorted(target,key=lambda x:(len(x),x)) if s not in family(final)])
            row['methods'][method]=m
        evaluations.append(row)
    grouped=defaultdict(list)
    for row in evaluations:
        for method,m in row['methods'].items():grouped[(row['noise_id'],row['policy'],row['conditional_state'],method)].append(m)
    groups=[]
    for (noise,policy,conditional,method),rows in sorted(grouped.items()):
        complete=[r for r in rows if r['status']=='complete']
        groups.append(dict(noise_id=noise,policy=policy,conditional_state=conditional,method=method,planned=len(rows),
            complete=len(complete),statuses=dict(Counter(r['status'] for r in rows)),
            raw_errors=sum(r['raw']['errors'] for r in complete) if complete else None,
            final_errors=sum(r['final']['errors'] for r in complete) if complete else None,
            raw_full=sum(r['raw']['full'] for r in complete),final_full=sum(r['final']['full'] for r in complete),
            contract_true_blocks_removed=sum(len(r['true_removed_by_contract']) for r in complete),
            incomplete_search_records=sum(r.get('all_scans_complete') is False for r in complete)))
    comparisons=[]
    zero={r['case_id']:r for r in evaluations if r['policy'] in ('zero_sham','zero','no_reward')}
    for row in evaluations:
        if row['case_id'] not in zero or row['policy'] in ('zero_sham','zero','no_reward'):continue
        baseline=zero[row['case_id']]
        for method,m in row['methods'].items():
            b=baseline['methods'].get(method,{})
            comparable=m['status']=='complete' and b.get('status')=='complete'
            comparisons.append(dict(case_id=row['case_id'],noise_id=row['noise_id'],policy=row['policy'],
                conditional_state=row['conditional_state'],method=method,comparable=comparable,
                final_error_difference=m['final']['errors']-b['final']['errors'] if comparable else None,
                reason=None if comparable else 'missing or unresolved selected output; not assigned zero error'))
    same_input=[]
    reference_names=[m for m in all_method_names if 'rnj' in m or m in ('all_legal_family_reference','same_pool_tree_enumeration','direct_control_tree_refit')]
    for row in evaluations:
        for method in [m for m in all_method_names if 'wzzt' in m]:
            candidate=row['methods'][method]
            for name in reference_names:
                reference=row['methods'][name]
                comparable=candidate['status']=='complete' and reference['status']=='complete'
                same_input.append(dict(case_id=row['case_id'],snapshot_id=row['snapshot_id'],noise_id=row['noise_id'],policy=row['policy'],
                    method=method,reference=name,conditional_state=row['conditional_state'],comparable=comparable,
                    final_error_difference=candidate['final']['errors']-reference['final']['errors'] if comparable else None,
                    raw_error_difference=candidate['raw']['errors']-reference['raw']['errors'] if comparable else None))
    comparison_groups=[]
    comparison_bins=defaultdict(list)
    for row in comparisons:comparison_bins[(row['noise_id'],row['policy'],row['conditional_state'],row['method'])].append(row)
    for (noise,policy,conditional,method),rows in sorted(comparison_bins.items()):
        usable=[r['final_error_difference'] for r in rows if r['comparable']]
        comparison_groups.append(dict(noise_id=noise,policy=policy,conditional_state=conditional,method=method,
            planned_named_case_comparisons=len(rows),comparable=len(usable),unresolved=len(rows)-len(usable),
            better=sum(d<0 for d in usable),same=sum(d==0 for d in usable),worse=sum(d>0 for d in usable)))
    noise_cases=defaultdict(list)
    for case in cases.values():noise_cases[case.get('noise_id')].append(case)
    case_denominators=[]
    for noise,rows in sorted(noise_cases.items()):
        distinct={json.dumps([r['truth'].get('R'),r['truth'].get('X')],sort_keys=True) for r in rows}
        case_denominators.append(dict(noise_id=noise,named_truth_cells=len(rows),distinct_generating_R_X_models=len(distinct)))
    stages=defaultdict(Counter)
    for row in evaluations:
        for method,data in row['methods'].items():
            for item in data.get('trace',[]):stages[(row['noise_id'],row['policy'],method)][item['stage']]+=1
    stage_summary=[dict(noise_id=noise,policy=policy,method=method,missing_true_block_stages=dict(counts))
                   for (noise,policy,method),counts in sorted(stages.items())]
    report=dict(status='passed' if not fail else 'audit_failed',created_utc=datetime.now(timezone.utc).isoformat(),
        source_sha256=sha(__file__),generation_root=str(gen),execution_root=str(run),checks=len(checks),failed_checks=fail,
        unique_snapshots=len(records),selected_models=selected_count,evaluator_case_policy_records=len(evaluations),
        method_statuses=dict(method_status),max_selected_train_L1_difference=maxtrain,max_selected_validation_L1_difference=maxval,
        raw_points_replayed=raw_point_count,max_raw_train_L1_difference=max_raw_train,max_raw_validation_L1_difference=max_raw_val,
        groups=groups,case_denominators=case_denominators,DR_vs_same_case_zero=comparisons,
        DR_vs_zero_summary=comparison_groups,missing_true_block_stage_summary=stage_summary,
        same_input_method_comparisons=same_input,evaluations=evaluations,
        new_LP_calls=0,new_QP_calls=0,new_AC_calls=0,
        same_current_calibration_within_each_snapshot=True,
        numerical_nonunique_parameter_representatives_not_global_certificates=True,
        all_shared_random_streams_not_independent_trials=True,
        conditional_all_on_not_randomized_performance=True,
        all_checks=checks)
    args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:report[k] for k in ('status','checks','failed_checks','unique_snapshots','selected_models','evaluator_case_policy_records','max_selected_train_L1_difference','max_selected_validation_L1_difference')}))
    return report

if __name__=='__main__':main()
