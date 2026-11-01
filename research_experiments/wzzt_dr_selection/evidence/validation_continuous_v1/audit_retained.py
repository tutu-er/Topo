"""Independent loss/selector audit of the completed retained-history diagnostic.

Reads only already-frozen observed packets and fitted results.  Truth supports
are opened only after independent original/retained/frozen selectors are fixed
for each record.  No optimizer, original selector or scoring helper is called.
No trajectories or fitted coefficients are exported.
"""
from __future__ import annotations
from collections import Counter
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import pickle
import sys

import numpy as np

ROOT=Path(r'C:\Users\23761\.codex\visualizations\2026\10\25\01a1726e-8ec7-7bc1-9f18-391f9cf79a1a\wzzt_validation_continuous_v1')
RUN=ROOT/'retained_history_v1'
counts=Counter();failed=[];artifacts={}


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def mark(path):
    path=Path(path);key=str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)
    artifacts[key]=dict(path=str(path),sha256=sha(path))
    return path
def read(path):return json.loads(mark(path).read_text(encoding='utf-8'))
def pk(path):
    with mark(path).open('rb') as stream:return pickle.load(stream)
def check(category,name,condition,**details):
    counts[category]+=1
    if not bool(condition):failed.append(dict(category=category,name=name,**details))
def close(a,b,tolerance=2.e-12):return abs(float(a)-float(b))<=tolerance
def ref(reference,name):
    path=Path(reference['path']);check('hash',name,sha(path)==reference['sha256']);return path
def family(point):return tuple(sorted(tuple(sorted(int(v) for v in s)) for s in point['supports']))
def requested(point):return tuple(sorted(tuple(sorted(int(v) for v in s)) for s in point.get('full_fit',{}).get('supports',point.get('native_family',point['supports']))))


def arrays(scenes,labels):
    pp=[];qq=[];yy=[];indices=[]
    for scene in scenes:
        p=scene['P_terminal'];q=scene['Q_terminal'];v=scene['V_terminal'];root=scene['V_root_observed']
        check('input','aligned_observed_frames',p.index.equals(q.index) and p.index.equals(v.index) and p.index.equals(root.index))
        check('input','terminal_columns',list(p.columns)==labels==list(q.columns)==list(v.columns))
        pv=p.to_numpy(dtype=float);qv=q.to_numpy(dtype=float);vv=v.to_numpy(dtype=float)
        rv=root.to_numpy(dtype=float).reshape(-1)
        pp.append(pv);qq.append(qv);yy.append(rv[:,None]**2-vv**2);indices.extend(p.index.tolist())
    check('input','unique_time_rows',len(indices)==len(set(indices)))
    return np.vstack(pp),np.vstack(qq),np.vstack(yy)


def predict(point,data):
    p,q,y=data;out=np.zeros_like(y)
    for block,r,x in zip(point['supports'],point['r'],point['x'],strict=True):
        block=list(block)
        value=float(r)*p[:,block].sum(axis=1)+float(x)*q[:,block].sum(axis=1)
        out[:,block]+=value[:,None]
    return out


def loss(point,data):return float(np.abs(predict(point,data)-data[2]).mean())


def choose(points,calibration,label):
    minimum=min(float(p['validation_mae']) for p in points)
    eligible=[i for i,p in enumerate(points) if p['validation_mae']<=1.03*minimum+1.e-12]
    selected=min(eligible,key=lambda i:(len(points[i]['supports']),points[i]['validation_mae'],i))
    raw=points[selected]
    threshold=.16*calibration['median_root_depth']
    keep=[i for i,s in enumerate(raw['supports']) if len(s)==1 or calibration['a']*raw['r'][i]+calibration['b']*raw['x'][i]+1.e-10>=threshold]
    final=dict(raw,supports=[raw['supports'][i] for i in keep],r=[raw['r'][i] for i in keep],x=[raw['x'][i] for i in keep])
    check('selection',label+'_raw_eligible',selected in eligible)
    return raw,final,selected,minimum,eligible


def selected_replay(raw,final,index,saved,label):
    check('selection',label+'_selected_index',index==saved['selection_index'])
    check('selection',label+'_raw_supports',family(raw)==family(saved['selected_uncontracted']))
    check('selection',label+'_final_supports',family(final)==family(saved))
    check('selection',label+'_raw_loss',close(raw['validation_mae'],saved['selection_validation_mae']))
    check('selection',label+'_retained_weights_match',np.allclose(final['r'],saved['r'],atol=1.e-12,rtol=0) and np.allclose(final['x'],saved['x'],atol=1.e-12,rtol=0))


def labels_family(point,labels):return {tuple(sorted(labels[i] for i in s)) for s in point['supports']}
def errors(point,labels,truth):
    fam=labels_family(point,labels);missing=truth-fam;extra=fam-truth
    return dict(full_errors=len(missing)+len(extra),missing=[list(s) for s in sorted(missing)],extra=[list(s) for s in sorted(extra)])


def main():
    if not (RUN/'ended.json').exists():raise RuntimeError('diagnostic is still running; no unfinished row inference permitted')
    ended=read(RUN/'ended.json');summary=read(RUN/'summary.json');protocol=read(RUN/'protocol.json')
    check('completion','eight_complete',summary['completed']==summary['denominator']==protocol['expected_records']==8)
    check('completion','136training_LP',summary['new_training_LP_calls']==protocol['maximum_new_training_LP']==136)
    check('completion','no_new_physics_or_search',summary['new_AC']==summary['new_RNJ_W_search']==summary['new_actual_observations']==0)
    check('completion','original_results_unchanged',summary['original_actual_results_unchanged'] and not summary['quotes_updated_after_actual'])
    check('rules','unchanged_validation_and_output',protocol['original_3pct']==.03 and protocol['original_contraction']==.16)
    frozen=read(ref(protocol['actual_fits_frozen'],'original_fits_frozen'))
    check('freeze','controller_truth_after_freeze',frozen['truth_has_not_yet_been_opened_by_controller'])
    price_freeze=read(ref(protocol['actual_price_freeze'],'original_price_freeze'))
    check('freeze','prices_frozen_before_actual',price_freeze['prices_frozen_before_actual'])
    check('freeze','prices_diagnostic_not90',price_freeze['quote_feasibility_status']=='diagnostic_phaseI_not90_feasible')
    scored=read(ref(protocol['actual_scored'],'actual_scored'))
    history=pk(ref(protocol['history'],'complete_history'))['history']
    historical=read(ref(protocol['historical_fit'],'historical_fit'))
    labels=list(history['labels']);target_labels=[201,202,203];target=tuple(labels.index(v) for v in target_labels)
    manifest=read(ROOT/'plan_v1/historical_family_manifest.json')
    specs=manifest['families']
    check('input','17full_historical_families',len(specs)==protocol['distinct_historical_families']==17)
    source_protocol=read(Path(read(ROOT/'plan_v1/protocol.json')['source_root'])/'main/protocol.json')
    coarse_source=source_protocol['competition']['selected_index'];fine_source=source_protocol['competition']['alternative_index']
    coarse_id=next(f['family_id'] for f in specs if coarse_source in f['source_indices'])
    fine_id=next(f['family_id'] for f in specs if fine_source in f['source_indices'])
    old_coarse=specs[coarse_id]['requested']
    coarse_key=tuple(sorted(tuple(sorted(s)) for s in old_coarse))
    summary_rows={(r['policy'],r['path']):r for r in summary['rows']}
    rows=[];totalLP=0;max_training_difference=0.;max_validation_difference=0.;actual_source_hashes={}
    for original in scored['rows']:
        name=original['record_id'];record=summary_rows[(original['policy'],original['path'])]
        check('completion',name+'_complete',record['status']=='complete')
        packet=pk(ref(original['packet'],name+'_observations'))
        actual=read(ref(original['result'],name+'_original_fit'))
        check('input',name+'_full_history_288',sum(len(d['P_terminal']) for d in history['training'])==288==sum(len(d['P_terminal']) for d in history['validation']))
        tr=arrays(history['training']+packet['training'],labels)
        va=arrays(history['validation']+packet['validation'],labels)
        check('input',name+'_480rows_preserved',tr[0].shape==va[0].shape==(480,16))
        private=pk(RUN/f'private_{name}.pkl')
        fit_by_id={p['historical_family_id']:p for p in private['refitted']}
        check('LP',name+'_17unique_refits',set(fit_by_id)==set(range(17)))
        check('LP',name+'_17receipt_calls',len(private['LP_receipts'])==record['new_training_LP_calls']==17)
        totalLP+=len(private['LP_receipts'])
        for j,receipt in enumerate(private['LP_receipts']):
            check('LP',name+f'_lp{j}_optimal',receipt['status']==2 and receipt['accepted'])
            check('LP',name+f'_lp{j}_objective',close(receipt['objective'],receipt['physical'],1.e-8))
            check('LP',name+f'_lp{j}_primal',receipt['max_constraint_violation']<=1.e-7 and receipt['max_bound_violation']<=1.e-7)
        refitted=[];frozen_points=[]
        for spec in specs:
            ident=spec['family_id'];p=fit_by_id[ident]
            check('family',name+f'_f{ident}_full_supports_retained',requested(p)==tuple(sorted(tuple(sorted(s)) for s in spec['requested'])))
            theta=np.r_[p['full_fit']['r'],p['full_fit']['x']]
            check('family',name+f'_f{ident}_zero_allowed_original_domain',theta.min()>=0 and theta.max()<=2)
            keep=[s for s,r,x in zip(p['full_fit']['supports'],p['full_fit']['r'],p['full_fit']['x'],strict=True) if len(s)==1 or max(r,x)>1.e-9]
            check('family',name+f'_f{ident}_only_original_numeric_prune',family(p)==tuple(sorted(tuple(sorted(s)) for s in keep)))
            jt=loss(p,tr);vv=loss(p,va)
            max_training_difference=max(max_training_difference,abs(jt-p['objective']))
            max_validation_difference=max(max_validation_difference,abs(vv-p['validation_mae']))
            check('loss',name+f'_f{ident}_train_replay',close(jt,p['objective'],1.e-8))
            check('loss',name+f'_f{ident}_validation_replay',close(vv,p['validation_mae']))
            refitted.append(dict(p,objective=jt,validation_mae=vv))
            old=historical['selection_candidates'][spec['source_indices'][0]]
            frozen_points.append(dict(old,objective=loss(old,tr),validation_mae=loss(old,va),historical_family_id=ident))
        for j,p in enumerate(actual['selection_candidates']):
            check('loss',name+f'_actual{j}_validation_replay',close(loss(p,va),p['validation_mae']))
        original_raw,original_final,original_index,_,_=choose(actual['selection_candidates'],actual['calibrations']['raw'],name+'_original')
        selected_replay(original_raw,original_final,original_index,actual['models']['multistart_original_rule'],name+'_original')
        enlarged=actual['selection_candidates']+refitted
        raw,final,index,best,eligible=choose(enlarged,actual['calibrations']['raw'],name+'_retained')
        selected_replay(raw,final,index,private['retained_selection'],name+'_retained')
        frozen_raw,frozen_final,frozen_index,frozen_best,_=choose(frozen_points,historical['calibrations']['raw'],name+'_frozen')
        selected_replay(frozen_raw,frozen_final,frozen_index,private['frozen_selection'],name+'_frozen')
        coarse=refitted[coarse_id];fine=refitted[fine_id]
        hcoarse=frozen_points[coarse_id];hfine=frozen_points[fine_id]
        margin=coarse['validation_mae']-1.03*best-1.e-12
        hmargin=hcoarse['validation_mae']-1.03*frozen_best-1.e-12
        pair=coarse['validation_mae']-1.03*fine['validation_mae']-1.e-12
        hpair=hcoarse['validation_mae']-1.03*hfine['validation_mae']-1.e-12
        pairchange=pair-hpair
        update_bound=float(np.abs(predict(coarse,va)-predict(hcoarse,va)).mean()+1.03*np.abs(predict(fine,va)-predict(hfine,va)).mean())
        check('margin',name+'_M0',close(margin,record['current_coarse_exit_margin']))
        check('margin',name+'_coarse_eligibility',bool(coarse['validation_mae']<=1.03*best+1.e-12)==record['current_coarse_eligible'])
        check('margin',name+'_frozen_M0',close(hmargin,record['frozen_predictor_coarse_margin']))
        check('margin',name+'_pair_joint',close(pair,record['pair_joint_refit_margin']))
        check('margin',name+'_pair_frozen',close(hpair,record['pair_frozen_margin']))
        check('margin',name+'_pair_change',close(pairchange,record['pair_margin_change']))
        check('margin',name+'_weight_update_bound',close(update_bound,record['validation_prediction_update_bound']))
        check('margin',name+'_triangle_bound',abs(pairchange)<=update_bound+2.e-12)
        check('target',name+'_raw_C_status',(target in family(raw))==record['raw_target'])
        check('target',name+'_final_C_status',(target in family(final))==record['final_target'])
        original_has_coarse=any(requested(p)==coarse_key or family(p)==coarse_key for p in actual['selection_candidates'])
        # Only now, after all independent selectors are fixed, open truth labels.
        truth_path=Path(original['physical_output'])/'truth_evaluation.json'
        check('hash',name+'_truth_frozen_hash',sha(truth_path)==original['truth_evaluation_sha256'])
        truth={tuple(sorted(s)) for s in read(truth_path)['true_supports']}
        e_original=errors(original_final,labels,truth);e_retained=errors(final,labels,truth);e_frozen=errors(frozen_final,labels,truth)
        check('score',name+'_original_errors',e_original['full_errors']==original['models']['multistart_original_rule']['full_errors']==record['original_full_errors'])
        check('score',name+'_retained_errors',e_retained['full_errors']==record['retained_history_full_errors'])
        check('score',name+'_retained_missing',e_retained['missing']==record['retained_history_missing'])
        check('score',name+'_retained_extra',e_retained['extra']==record['retained_history_extra'])
        check('score',name+'_frozen_errors',e_frozen['full_errors']==record['frozen_predictor_pool_full_errors'])
        check('score',name+'_same_fee',close(record['actual_fee'],original['actual_payment']))
        rows.append(dict(record_id=name,policy=original['policy'],path=original['path'],original_full_errors=e_original['full_errors'],
            retained_history_full_errors=e_retained['full_errors'],retained_missing=e_retained['missing'],retained_extra=e_retained['extra'],
            frozen_history_full_errors=e_frozen['full_errors'],original_pool_has_current_coarse=original_has_coarse,
            original_pool_candidates=len(actual['selection_candidates']),retained_candidates=len(enlarged),
            selected_index=index,retained_raw_blocks=len(raw['supports']),retained_final_blocks=len(final['supports']),
            raw_target_present=target in family(raw),final_target_present=target in family(final),
            current_coarse_exit_margin=margin,current_coarse_eligible=coarse['validation_mae']<=1.03*best+1.e-12,
            historical_frozen_coarse_margin=hmargin,actual_fee=original['actual_payment'],
            pair_margin_update=pairchange,prediction_update_triangle_bound=update_bound))
    check('completion','complete8replayed',len(rows)==8==len(scored['rows']))
    check('completion','136receipts_replayed',totalLP==136)
    check('freeze','all_inputs_unchanged_during_audit',all(sha(v['path'])==v['sha256'] for v in artifacts.values()))
    return dict(status='complete' if not failed else 'failed',audit_passed=not failed,created_utc=datetime.now(timezone.utc).isoformat(),
        check_count=sum(counts.values()),checks_by_category=dict(counts),failures=failed,rows=rows,
        denominator=8,refitted_families_per_observation=17,training_LP_receipts_checked=totalLP,
        max_training_loss_replay_difference=max_training_difference,max_validation_loss_replay_difference=max_validation_difference,
        original_zero_error_count=sum(r['original_full_errors']==0 for r in rows),
        retained_zero_error_count=sum(r['retained_history_full_errors']==0 for r in rows),
        frozen_predictor_zero_error_count=sum(r['frozen_history_full_errors']==0 for r in rows),
        current_coarse_eligible_count=sum(r['current_coarse_eligible'] for r in rows),
        current_coarse_present_in_original_pool_count=sum(r['original_pool_has_current_coarse'] for r in rows),
        raw_target_missing_count=sum(not r['raw_target_present'] for r in rows),
        computation=dict(new_solver_calls=0,new_LP=0,new_AC=0,new_RNJ_W_search=0,new_actual_observations=0),
        scope='Post-evaluation named portfolio sensitivity on same8 frozen observations; no newDReffect and no change to original outcomes.',
        interpretation='If reinserting historical F0 changes complete recovery to a missed block, original recovery depends on candidate generation/availability, not merely on validation exclusion or DR.',
        artifact_sha256={k:v['sha256'] for k,v in artifacts.items()})


if __name__=='__main__':
    destination=ROOT/'audit_retained_history_v1.json'
    if destination.exists():raise FileExistsError(destination)
    try:result=main()
    except Exception as error:
        import traceback
        result=dict(status='failed',audit_passed=False,error_type=type(error).__name__,message=str(error),traceback=traceback.format_exc(),checks_by_category=dict(counts),failures=failed)
    result['audit_source_sha256']=sha(__file__)
    destination.write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({k:result.get(k) for k in ['status','audit_passed','check_count','denominator','training_LP_receipts_checked',
        'original_zero_error_count','retained_zero_error_count','frozen_predictor_zero_error_count','current_coarse_eligible_count',
        'current_coarse_present_in_original_pool_count','raw_target_missing_count','error_type','message']},ensure_ascii=False))
    if result.get('failures'):print(json.dumps(result['failures'][:15],ensure_ascii=False))
    sys.exit(0 if result['audit_passed'] else 1)
