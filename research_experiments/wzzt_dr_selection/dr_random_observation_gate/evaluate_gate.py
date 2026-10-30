"""Evaluator-only structure/L1 replay. No identifier, solver or simulator imports."""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
from datetime import datetime,timezone
import hashlib,json
from pathlib import Path
import pickle
import numpy as np

TOL=1e-9
METHODS=('historical_rnj_refit','current_rebuilt_rnj','old_pool_wzzt','current_pool_wzzt',
         'expanded_pool_wzzt','same_pool_tree_enumeration','all_legal_family_reference','direct_control_tree_refit')
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def fam(values):return frozenset(tuple(sorted(int(i) for i in s)) for s in values)


def point_parts(point,labels):
    """Keep input support order for identical floating matrix summation order."""
    pos={int(label):i for i,label in enumerate(labels)};n=len(labels)
    indices=point.get('support_indices',point.get('supports'))
    labelled=point.get('support_labels')
    if indices is None:
        if labelled is None:raise ValueError('point has no declared supports')
        indices=[[pos[int(label)] for label in s] for s in labelled]
    result=[tuple(int(i) for i in s) for s in indices]
    if any(not s or len(set(s))!=len(s) or min(s)<0 or max(s)>=n for s in result):raise ValueError('invalid point support')
    if labelled is not None and fam(result)!=fam([[pos[int(label)] for label in s] for s in labelled]):
        raise ValueError('label/index support mismatch')
    r=point.get('r_values',point.get('r'));x=point.get('x_values',point.get('x'))
    if r is None or x is None or len(result)!=len(r) or len(r)!=len(x):raise ValueError('point coefficient mismatch')
    r=np.asarray(r,float);x=np.asarray(x,float)
    if not np.isfinite(r).all() or not np.isfinite(x).all() or (r<0).any() or (x<0).any():raise ValueError('invalid point coefficients')
    return result,r,x


def effective(point,labels,tolerance=TOL):
    supports,r,x=point_parts(point,labels)
    return fam([s for s,rv,xv in zip(supports,r,x) if rv>tolerance or xv>tolerance])
def matrices(point,labels):
    supports,r,x=point_parts(point,labels);n=len(labels);R=np.zeros((n,n));X=np.zeros((n,n))
    for s,rv,xv in zip(supports,r,x):R[np.ix_(s,s)]+=rv;X[np.ix_(s,s)]+=xv
    return R,X
def raw_point(point):
    if 'selected_uncontracted' in point:return point['selected_uncontracted']
    if 'pre_contraction_support_indices' in point:
        return dict(supports=point['pre_contraction_support_indices'],r=point['pre_contraction_r_values'],x=point['pre_contraction_x_values'])
    raise ValueError('selected point lacks precontraction evidence')
def score(point,labels,target,tolerance=TOL):
    predicted=effective(point,labels,tolerance);target=fam(target)
    return dict(errors=len(predicted^target),full_recovery=predicted==target,
        missed=[list(s) for s in sorted(target-predicted)],extra=[list(s) for s in sorted(predicted-target)],
        predicted=[list(s) for s in sorted(predicted)])


def observed_arrays(scenes):
    values=[[],[],[]]
    for scene in scenes:
        p=scene['P_terminal'].to_numpy(dtype=float);q=scene['Q_terminal'].to_numpy(dtype=float)
        v=scene['V_terminal'].to_numpy(dtype=float);root=scene['V_root_observed'].to_numpy(dtype=float)
        y=root[:,None]**2-v**2
        if p.shape!=q.shape or p.shape!=y.shape:raise ValueError('observed channel shape mismatch')
        for store,a in zip(values,(p,q,y)):store.append(a)
    return tuple(np.concatenate(v) for v in values)
def loss(point,labels,data):
    p,q,y=data;R,X=matrices(point,labels)
    return float(np.mean(np.abs(p@R.T+q@X.T-y)))


def summarize(rows,keys):
    groups=defaultdict(list)
    for row in rows:groups[tuple(row[k] for k in keys)].append(row)
    result=[]
    for key,group in sorted(groups.items(),key=lambda item:str(item[0])):
        known=[r for r in group if r.get('final') is not None]
        complete=[r for r in known if r.get('all_scans_complete') is not False]
        result.append(dict(zip(keys,key),planned=len(group),known=len(known),unknown=len(group)-len(known),
            statuses=dict(Counter(r['status'] for r in group)),
            final_errors=sum(r['final']['errors'] for r in known) if known else None,
            final_full=sum(r['final']['full_recovery'] for r in known),
            raw_errors=sum(r['raw']['errors'] for r in known) if known else None,
            raw_full=sum(r['raw']['full_recovery'] for r in known),
            fully_scanned_known=len(complete),partial_search_known=len(known)-len(complete),
            unique_snapshots=len({r.get('snapshot_id') for r in group})))
    return result


def evaluate(generation_root,execution_root,out):
    generation_root,execution_root,out=map(lambda p:Path(p).resolve(),(generation_root,execution_root,out))
    if out.exists():raise FileExistsError('existing evaluation retained; no overwrite')
    truthfile=generation_root/'evaluator_truth.json';truth=read(truthfile);cfg=read(execution_root/'protocol.json')
    if truth.get('identifier_input') is not False:raise ValueError('evaluator truth must be separated from identifier')
    cases={c['id']:c for c in truth['cases']};snapshots={r['id']:r for r in cfg['snapshots']}
    histories={r['id']:r for r in cfg['histories']};states={k:read(execution_root/'history'/f'{k}.json') for k in histories}
    root_records=read(execution_root/'records.json');records={r['id']:r for r in root_records}
    results={};arrays={};audits=[];selected_count=0;max_l1=0.
    for sid,spec in snapshots.items():
        path=execution_root/'records'/f'{sid}.json'
        if not path.exists():continue
        result=read(path);results[sid]=result
        h=pickle.loads(Path(histories[spec['history_id']]['path']).read_bytes());f=pickle.loads(Path(spec['path']).read_bytes())
        train=observed_arrays(h['training']+f['training']);valid=observed_arrays(h['validation']+f['validation'])
        labels=result.get('terminal_labels',states[spec['history_id']].get('terminal_labels'))
        arrays[sid]=(train,valid,labels)
        for name,method in result.get('methods',{}).items():
            if method.get('status')!='complete':continue
            selected_count+=1;point=method['selected'];raw=raw_point(point)
            for part,selected in (('raw',raw),('final',point)):
                tr=loss(selected,labels,train);va=loss(selected,labels,valid)
                tr_key='train_mae' if part=='final' else 'objective'
                expected_tr=selected.get(tr_key);expected_va=selected.get('validation_mae')
                errors=[]
                if expected_tr is not None:errors.append(abs(tr-float(expected_tr)))
                if expected_va is not None:errors.append(abs(va-float(expected_va)))
                if part=='raw' and point.get('selection_validation_mae') is not None:errors.append(abs(va-point['selection_validation_mae']))
                difference=max(errors,default=0.);max_l1=max(max_l1,difference)
                audits.append(dict(snapshot_id=sid,method=name,part=part,train_mae=tr,validation_mae=va,
                                   max_saved_L1_difference=difference,pass_=difference<=1e-10))
    evaluated=[]
    for mapping in truth['evaluations']:
        case=cases[mapping['case_id']];target=case['truth'];sid=mapping['snapshot_id'];result=results.get(sid,{})
        expected=[(i,) for i,d in enumerate(target['d']) if d>0]
        if target['b']>0:expected.append((0,1,2))
        if target['c']>0:expected.append(tuple(target['pair']))
        if fam(expected)!=fam(mapping['target_family']) or fam(expected)!=fam(target['effective_supports']):raise ValueError('positive/zero truth family mismatch')
        for name in METHODS:
            method=result.get('methods',{}).get(name,{})
            row=dict(case_id=mapping['case_id'],snapshot_id=sid,history_id=mapping['history_id'],noise_id=case['noise_id'],
                c=target['c'],b=target['b'],true_pair=target['pair'],zero_root_stress=case['zero_root_stress'],
                zero_leaf_indices=[i for i,d in enumerate(target['d']) if d==0],policy=mapping['policy'],
                conditional_state=mapping['conditional_state'],actual_payment=mapping['actual_payment'],method=name,
                status=method.get('status',result.get('status','snapshot_not_returned')),
                all_scans_complete=method.get('all_scans_complete'),final=None,raw=None,strict_final=None,strict_raw=None)
            if row['status']=='complete':
                labels=result['terminal_labels'];point=method['selected'];raw=raw_point(point)
                row.update(final=score(point,labels,expected),raw=score(raw,labels,expected),
                    strict_final=score(point,labels,expected,0.),strict_raw=score(raw,labels,expected,0.),
                    selected_origin=point.get('selected_origin'),selection_index=point.get('selection_index'),
                    new_supports_vs_current=result.get('new_supports',[]))
            evaluated.append(row)
    paired=[]
    for row in evaluated:
        if row['policy']=='zero_sham':continue
        zero=next(r for r in evaluated if r['case_id']==row['case_id'] and r['policy']=='zero_sham' and r['method']==row['method'])
        known=row['final'] is not None and zero['final'] is not None
        paired.append(dict(case_id=row['case_id'],snapshot_id=row['snapshot_id'],method=row['method'],policy=row['policy'],
            conditional_state=row['conditional_state'],noise_id=row['noise_id'],c=row['c'],b=row['b'],known=known,
            delta_final_errors=row['final']['errors']-zero['final']['errors'] if known else None,
            delta_raw_errors=row['raw']['errors']-zero['raw']['errors'] if known else None,
            paid_actual_payment=row['actual_payment'],comparison='paid-minus-natural-addition; negative improves'))
    comparisons=[]
    for row in evaluated:
        if row['method']!='expanded_pool_wzzt':continue
        for baseline in ('current_pool_wzzt','current_rebuilt_rnj','same_pool_tree_enumeration','all_legal_family_reference','direct_control_tree_refit'):
            other=next(r for r in evaluated if r['case_id']==row['case_id'] and r['policy']==row['policy'] and r['method']==baseline)
            known=row['final'] is not None and other['final'] is not None
            comparisons.append(dict(case_id=row['case_id'],policy=row['policy'],conditional_state=row['conditional_state'],
                noise_id=row['noise_id'],c=row['c'],b=row['b'],baseline=baseline,known=known,
                delta_final_errors=row['final']['errors']-other['final']['errors'] if known else None,
                delta_raw_errors=row['raw']['errors']-other['raw']['errors'] if known else None))
    stopped=[]
    for cid,case in cases.items():
        mapping=next(r for r in truth['evaluations'] if r['case_id']==cid);state=states[mapping['history_id']]
        row=dict(case_id=cid,history_id=mapping['history_id'],noise_id=case['noise_id'],c=case['truth']['c'],b=case['truth']['b'],
            policy='historical_stop',method='historical_selected_RNJ',actual_payment=0.,status=state['status'],final=None,raw=None)
        if state['status']=='complete':
            row.update(final=score(state['selected'],state['terminal_labels'],mapping['target_family']),
                       raw=score(raw_point(state['selected']),state['terminal_labels'],mapping['target_family']))
        stopped.append(row)
    def pair_summary(rows,keys,delta='delta_final_errors'):
        grouped=defaultdict(list)
        for r in rows:grouped[tuple(r[k] for k in keys)].append(r)
        return [dict(zip(keys,key),planned=len(group),known=sum(r['known'] for r in group),
            better=sum(r['known'] and r[delta]<0 for r in group),same=sum(r['known'] and r[delta]==0 for r in group),
            worse=sum(r['known'] and r[delta]>0 for r in group),unknown=sum(not r['known'] for r in group))
            for key,group in sorted(grouped.items(),key=lambda item:str(item[0]))]
    all_lp=[];contexts=0
    for state in states.values():all_lp+=state.get('LP_solves',[]);contexts+=bool(state.get('LP_solves'))
    for result in results.values():
        current=Path(result.get('current_state_path',''))
        if current.is_file():state=read(current);all_lp+=state.get('LP_solves',[]);contexts+=bool(state.get('LP_solves'))
        for method in result.get('methods',{}).values():all_lp+=method.get('LP_solves',[]);contexts+=bool(method.get('LP_solves'))
    summary=dict(status='passed' if all(a['pass_'] for a in audits) else 'audit_failed',
        evaluator_cases=len(cases),evaluator_policy_mappings=len(truth['evaluations']),methods=len(METHODS),
        planned_structure_evaluations=len(truth['evaluations'])*len(METHODS),actual_structure_rows=len(evaluated),
        unique_observed_snapshots=len(snapshots),processed_snapshot_records=len(results),unique_complete_method_outputs=selected_count,
        method_output_statuses=dict(Counter(m.get('status','unknown') for r in results.values() for m in r.get('methods',{}).values())),
        numerical_effective_prune=TOL,true_positive_weak_weights_not_pruned=True,
        numeric_vs_strict_final_support_differences=sum(r['final'] is not None and r['final']['predicted']!=r['strict_final']['predicted'] for r in evaluated),
        overall_by_method_noise_policy=summarize(evaluated,['method','noise_id','policy','conditional_state']),
        grouped=summarize(evaluated,['noise_id','b','c','policy','method','conditional_state']),
        paid_vs_zero=pair_summary(paired,['method','noise_id','policy','conditional_state']),
        expanded_vs_baselines=pair_summary(comparisons,['baseline','noise_id','policy','conditional_state']),
        history_stop_records=len(stopped),historical_stop_is_only_the_returned_RNJ_selection=True,
        historical_WZZT_stop_not_computed_and_not_invented=True,
        fixed_LP_calls=len(all_lp),LP_statuses=dict(Counter(s['status'] for s in all_lp)),distinct_solver_contexts=contexts,
        independent_L1_checks=len(audits),max_selected_L1_difference=max_l1,
        identifier_truth_input=False,new_price_optimizer=False,
        randomization_scope='one shared natural/noise/acceptance/code stream; conditional all-on separate; mappings are not independent trials',
        costs='fees belong to evaluator policy trajectories, not summed across methods or repeated truth models',
        source_sha256=sha(__file__),truth_sha256=sha(truthfile),execution_protocol_sha256=sha(execution_root/'protocol.json'),
        new_LP=0,new_QP=0,new_AC=0)
    out.mkdir(parents=True)
    for name,value in [('evaluations.json',evaluated),('paid_vs_zero.json',paired),('expanded_vs_baselines.json',comparisons),
                       ('historical_stop.json',stopped),('selected_L1_audit.json',audits),('summary.json',summary)]:write(out/name,value)
    print(json.dumps({k:summary[k] for k in ('status','evaluator_policy_mappings','planned_structure_evaluations',
        'unique_observed_snapshots','unique_complete_method_outputs','fixed_LP_calls','independent_L1_checks','max_selected_L1_difference')}))
    for group in summary['paid_vs_zero']:
        if group['method']=='expanded_pool_wzzt':print(json.dumps(group))
    return summary


def main():
    p=argparse.ArgumentParser();p.add_argument('--generation-root',type=Path,required=True)
    p.add_argument('--execution-root',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--allow-truth-read',action='store_true');a=p.parse_args()
    if not a.allow_truth_read:p.error('evaluator truth read requires explicit root authorization after identifier returns')
    evaluate(a.generation_root,a.execution_root,a.out)

if __name__=='__main__':main()
