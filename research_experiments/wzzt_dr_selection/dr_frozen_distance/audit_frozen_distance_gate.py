"""External gate and post-run independent audit for a frozen two-model test.

The gate never opens a physical truth file or parses an observed packet.
The score audit first fixes ALL independently reconstructed predictions and
selections, then verifies receipt-bound truth hashes and opens truth labels.
No fitter, original selector, optimizer or physical generator is imported.
"""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import pickle
import sys
import zipfile

import numpy as np


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def near(a,b,tol=2.e-12):return abs(float(a)-float(b))<=tol


class Audit:
    def __init__(self,root,phase):
        self.root=Path(root).resolve();self.phase=phase
        self.count=Counter();self.failed=[];self.artifacts={}
    def check(self,category,name,condition,**details):
        self.count[category]+=1
        if not bool(condition):self.failed.append(dict(category=category,name=name,**details))
    def record(self,path):
        path=Path(path).resolve();self.artifacts[str(path)]=sha(path);return path
    def read(self,path):return json.loads(self.record(path).read_text(encoding='utf-8'))
    def bound(self,item,name):
        path=Path(item['path']).resolve()
        self.check('hash',name,sha(path)==item['sha256'])
        return path
    def read_ref(self,item,name):return self.read(self.bound(item,name))
    def unpickle(self,path):
        with self.record(path).open('rb') as stream:return pickle.load(stream)
    def finish(self,**payload):
        self.check('hash','all_inputs_unchanged_during_audit',all(sha(p)==h for p,h in self.artifacts.items()))
        return dict(status='complete' if not self.failed else 'failed',audit_passed=not self.failed,
            phase=self.phase,created_utc=datetime.now(timezone.utc).isoformat(),
            check_count=sum(self.count.values()),checks_by_category=dict(self.count),failures=self.failed,
            computation=dict(new_solver=0,new_LP=0,new_AC=0,new_search=0),
            artifact_sha256=self.artifacts,**payload)


def matrix(point,n):
    rr=np.zeros((n,n));xx=np.zeros((n,n))
    for block,r,x in zip(point['supports'],point['r'],point['x'],strict=True):
        rr[np.ix_(block,block)]+=r;xx[np.ix_(block,block)]+=x
    return rr,xx


def output(point,calibration):
    return [s for s,r,x in zip(point['supports'],point['r'],point['x'],strict=True)
        if len(s)==1 or calibration['a']*r+calibration['b']*x+1.e-10>=.16*calibration['median_root_depth']]


def family(blocks):return {tuple(sorted(s)) for s in blocks}
def label_family(blocks,labels):return {tuple(sorted(labels[i] for i in s)) for s in blocks}


def independent_choice(pool,losses):
    identities=pool['selection_order'];models={m['candidate_id']:m for m in pool['models']}
    best=min(losses.values())
    eligible=[i for i in identities if losses[i]<=1.03*best+1.e-12]
    selected=min(eligible,key=lambda i:(len(models[i]['point']['supports']),losses[i],identities.index(i)))
    return selected,eligible,output(models[selected]['point'],pool['calibration'])


def plan_cost(plan):
    mean=np.asarray(plan['amplitudes'],float);prices=np.asarray(plan['prices'],float)
    d=np.asarray(plan['response_factors'],float);lam=np.asarray(plan['elasticities'],float)
    hours=[]
    for module in plan['modules']:
        shape=np.asarray(next(iter(module['profiles'].values())),float)
        hours.append(.25*np.maximum(-shape,0.).sum())
    hours=np.asarray(hours)
    return mean,prices,d,lam,hours,float((hours*prices)@mean),float(np.sum(hours*lam*(1+np.abs(d).sum(axis=1))*prices**2))


def closure(a):
    root=a.root;c=a.read(root/'protocol.json');freeze=a.read(root/'price_freeze.json')
    a.check('freeze','protocol_frozen',c['status']==freeze['status']=='frozen')
    a.read_ref(freeze['protocol'],'price_freeze_protocol')
    pool=a.read_ref(freeze['frozen_pool'],'price_freeze_pool')
    a.check('freeze','same_pool_reference',freeze['frozen_pool']==c['frozen_pool'])
    a.bound(c['frozen_pool'],'protocol_pool');a.bound(c['history_packet'],'protocol_history')
    a.bound(freeze['problem'],'price_problem');a.check('freeze','same_problem_reference',freeze['problem']==c['problem'])
    preparation=root/'preparation_ended.json'
    if preparation.exists():
        ended=a.read(preparation)
        if 'price_freeze' in ended:a.bound(ended['price_freeze'],'preparation_pins_price_freeze')
    a.check('freeze','no_new_train_or_parameters',c['new_training_rows']==0 and c['original_history_rows']==[288,288])
    a.check('freeze','one96day_validation384',c['new_validation_rows']==96 and c['cumulative_validation_rows']==384)
    a.check('freeze','budget_same_declared',c['budget']==1.6)
    for path,digest in c['source_sha256'].items():a.check('source','live_'+Path(path).name,sha(path)==digest)
    zip_path=a.bound(freeze['source_snapshot'],'source_snapshot')
    with zipfile.ZipFile(zip_path) as z:
        zip_hashes=Counter(hashlib.sha256(z.read(name)).hexdigest() for name in z.namelist())
    a.check('source','snapshot_exact_source_hashes',zip_hashes==Counter(c['source_sha256'].values()))
    fingerprint=dict(pool);fingerprint.pop('pool_sha256',None)
    actual_fingerprint=hashlib.sha256(json.dumps(fingerprint,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
    a.check('parameters','pool_fingerprint',actual_fingerprint==pool['pool_sha256'])
    a.check('parameters','two_models',len(pool['models'])==2 and pool['selection_order']==['coarse','refined'])
    a.check('parameters','288historical_validation',pool['train_rows']==pool['validation_rows']==288 and pool['historical_validation_cells']==288*len(pool['labels']))
    baseline=a.read_ref(pool['historical_fit'],'original_historical_fit')
    a.check('parameters','historical_calibration_exact',pool['calibration']==baseline['calibrations']['raw'])
    for m in pool['models']:
        label=m['candidate_id'];old=baseline['selection_candidates'][m['historical_index']]
        a.check('parameters',label+'_entire_original_point',m['point']==old)
        r,x=matrix(old,len(pool['labels']))
        a.check('parameters',label+'_R_exact',np.array_equal(r,np.asarray(m['R'])))
        a.check('parameters',label+'_X_exact',np.array_equal(x,np.asarray(m['X'])))
        a.check('parameters',label+'_original_output_map',output(old,pool['calibration'])==m['frozen_output_supports'])
        a.check('parameters',label+'_output_labels',label_family(output(old,pool['calibration']),pool['labels'])==family(m['frozen_output_support_labels']))
        a.check('parameters',label+'_historical_loss_constant',near(old['validation_mae']*pool['historical_validation_cells'],m['historical_validation_abs_sum'],1.e-7))
    plans={}
    for policy,reference in freeze['plans'].items():
        plan=a.read_ref(reference,policy+'_plan');plans[policy]=plan
        mean,prices,d,lam,hours,expected,hard=plan_cost(plan)
        a.check('price',policy+'_price_link',np.max(np.abs(mean-lam*prices))<1.e-9)
        a.check('price',policy+'_nonnegative_prices',prices.min()>=0 and prices.max()<=2+1.e-7)
        a.check('price',policy+'_expected_fee',near(expected,plan['expected_payment'],1.e-7))
        a.check('price',policy+'_hard_fee',near(hard,plan['hard_payment_bound'],1.e-7) and hard<=1.6+1.e-7)
    return c,freeze,pool,plans


def ledger_gate(a,c,freeze,pool,plans):
    ledger=a.read(a.root/'actual/observations_frozen.json')
    a.check('ledger','observations_frozen',ledger['status']=='frozen')
    a.bound(ledger['price_freeze'],'observation_ledger_price_freeze')
    expected={(p,path) for p in c['policies'] for path in c['actual_paths']}
    actual={(r['policy'],r['path']) for r in ledger['rows']}
    a.check('ledger','all12requests',len(ledger['rows'])==ledger['planned']==c['expected_actual_records']==12 and actual==expected)
    costs=[];receipts={}
    for row in ledger['rows']:
        identity=row['record_id'];request=a.read_ref(row['request'],identity+'_request')
        a.check('ledger',identity+'_request_protocol',request['protocol']['sha256']==sha(a.root/'protocol.json'))
        a.check('ledger',identity+'_request_pool',request['frozen_pool']==c['frozen_pool'])
        a.check('ledger',identity+'_request_plan',request['plan']==freeze['plans'][row['policy']])
        if 'receipt' not in row:
            costs.append(dict(record_id=identity,status=row['status'],receipt_bound=False,actual_payment=row.get('actual_payment')))
            a.check('ledger',identity+'_observed_requires_bound_receipt',row['status']!='observed')
            continue
        receipt=a.read_ref(row['receipt'],identity+'_receipt');receipts[identity]=receipt
        a.check('ledger',identity+'_worker_request',receipt['request_sha256_expected']==row['request']['sha256'])
        if receipt.get('plan_sha256') is not None:a.check('ledger',identity+'_worker_plan',receipt['plan_sha256']==freeze['plans'][row['policy']]['sha256'])
        if receipt.get('frozen_pool_sha256') is not None:a.check('ledger',identity+'_worker_pool',receipt['frozen_pool_sha256']==c['frozen_pool']['sha256'])
        if row['status']=='observed':
            a.check('ledger',identity+'_complete_worker',receipt['status']=='complete' and receipt['observed_validation_rows']==96)
            a.bound(row['packet'],identity+'_observations')
            a.check('ledger',identity+'_worker_packet_hash',receipt['private_files']['packet.pkl']['sha256']==row['packet']['sha256'])
        costs.append(dict(record_id=identity,status=row['status'],receipt_bound=True,
            actual_payment=receipt.get('actual_payment'),known_actual_payment=receipt.get('known_actual_payment'),
            unknown_payment_events=receipt.get('unknown_payment_events'),AC_attempted=bool(receipt.get('AC_attempted')),
            physical_status=receipt['status']))
    return ledger,receipts,costs


def gate(a):
    a.check('boundary','before_controller_score',not (a.root/'actual/predictions_frozen.json').exists() and not (a.root/'actual/scored.json').exists())
    c,freeze,pool,plans=closure(a);ledger,receipts,costs=ledger_gate(a,c,freeze,pool,plans)
    return a.finish(expected_records=12,observed_records=sum(r['status']=='observed' for r in ledger['rows']),
        prices_and_pool_sha256=dict(price_freeze=sha(a.root/'price_freeze.json'),pool=c['frozen_pool']['sha256']),
        truth_files_opened=0,observed_packets_parsed=0,ledger_costs=costs,
        note='Opaque receipt metadata only; physical truth files and observed arrays have not been opened by this gate.')


def scene_arrays(scene,labels):
    for name in ['P_terminal','Q_terminal','V_terminal']:
        if list(scene[name].columns)!=labels:raise ValueError(name+' label order changed')
    p=scene['P_terminal'].to_numpy(float);q=scene['Q_terminal'].to_numpy(float)
    v=scene['V_terminal'].to_numpy(float);root=scene['V_root_observed'].to_numpy(float).reshape(-1)
    y=root[:,None]**2-v**2
    if 'drop_target' in scene and not np.allclose(y,scene['drop_target'].to_numpy(float),rtol=0,atol=1.e-12):raise ValueError('observed target mismatch')
    return p,q,y


def block_prediction(point,p,q):
    out=np.zeros_like(p)
    for block,r,x in zip(point['supports'],point['r'],point['x'],strict=True):
        value=r*p[:,block].sum(axis=1)+x*q[:,block].sum(axis=1)
        out[:,block]+=value[:,None]
    return out


def score_audit(a):
    previous=a.read(a.root/'external_gate_before_score.json')
    a.check('boundary','external_gate_passed',previous['audit_passed'] and previous['phase']=='gate' and previous['truth_files_opened']==0)
    c,freeze,pool,plans=closure(a);ledger,receipts,costs=ledger_gate(a,c,freeze,pool,plans)
    a.check('boundary','pool_unchanged_since_gate',previous['prices_and_pool_sha256']['pool']==c['frozen_pool']['sha256'])
    a.check('boundary','price_freeze_unchanged_since_gate',previous['prices_and_pool_sha256']['price_freeze']==sha(a.root/'price_freeze.json'))
    projections=a.read(a.root/'actual/predictions_frozen.json');scored=a.read(a.root/'actual/scored.json');summary=a.read(a.root/'summary.json')
    a.bound(scored['predictions_frozen'],'scored_prediction_freeze')
    a.bound(projections['observations_frozen'],'prediction_observation_freeze')
    a.check('boundary','predictions_before_truth_flag',projections['no_true_structure_read_by_controller'])
    a.check('denominator','all12raw_records',len(projections['rows'])==len(scored['rows'])==12==scored['planned'])
    indexed={(r['policy'],r['path']):r for r in scored['rows']}
    own={};score_files={};model_map={m['candidate_id']:m for m in pool['models']};n=len(pool['labels'])
    history=a.unpickle(a.bound(pool['history_packet'],'historical_packet'))['history']
    for model in pool['models']:
        total=0.;cells=0
        for scene in history['validation']:
            p,q,y=scene_arrays(scene,pool['labels']);total+=float(np.abs(block_prediction(model['point'],p,q)-y).sum());cells+=p.size
        a.check('loss',model['candidate_id']+'_historical_abs_sum_replay',near(total,model['historical_validation_abs_sum'],1.e-8))
        a.check('loss',model['candidate_id']+'_historical_cells',cells==pool['historical_validation_cells']==288*n)
    # First pass fixes all independent predictions and selectors, with no truth.
    for row in projections['rows']:
        identity=row['record_id']
        if row['status']!='predictions_frozen':continue
        packet=a.unpickle(a.bound(row['packet'],identity+'_packet'))
        a.check('observations',identity+'_single_new_validation_day',len(packet['validation'])==1 and not packet.get('training'))
        scene=packet['validation'][0];p,q,y=scene_arrays(scene,pool['labels'])
        a.check('observations',identity+'_96rows',p.shape==q.shape==y.shape==(96,n))
        saved=a.read_ref(row['scores'],identity+'_frozen_score');score_files[identity]=saved
        losses={};day_losses={};predictions={}
        for model in pool['models']:
            name=model['candidate_id'];prediction=block_prediction(model['point'],p,q);predictions[name]=prediction
            new_abs=float(np.abs(prediction-y).sum());day_losses[name]=new_abs/p.size
            losses[name]=(model['historical_validation_abs_sum']+new_abs)/(pool['historical_validation_cells']+p.size)
            record=next(r for r in saved['rows'] if r['candidate_id']==name)
            a.check('loss',identity+'_'+name+'_new_abs',near(new_abs,record['new_validation_abs_sum'],1.e-8))
            a.check('loss',identity+'_'+name+'_new_mae',near(day_losses[name],record['new_validation_mae']))
            a.check('loss',identity+'_'+name+'_cumulative_mae',near(losses[name],record['cumulative_validation_mae']))
        chosen,eligible,blocks=independent_choice(pool,losses)
        day_chosen,day_eligible,day_blocks=independent_choice(pool,day_losses)
        distance=float(np.abs(predictions['coarse']-predictions['refined']).mean())
        a.check('selection',identity+'_cumulative_selected',chosen==saved['selected_candidate_id'])
        a.check('selection',identity+'_day_selected',day_chosen==saved['day_only_selection']['selected_candidate_id'])
        a.check('selection',identity+'_eligibles',eligible==saved['eligible_candidate_ids'])
        a.check('selection',identity+'_original_contracted_output',blocks==saved['selected_output_supports'])
        a.check('selection',identity+'_day_contracted_output',day_blocks==saved['day_only_selection']['selected_output_supports'])
        a.check('selection',identity+'_384rows',saved['cumulative_validation_rows']==384 and saved['historical_validation_rows']==288 and saved['new_validation_rows']==96)
        a.check('selection',identity+'_no_parameter_update',not saved['parameters_changed'] and not saved['calibration_changed'] and saved['new_fit_calls']==saved['new_search_calls']==0)
        a.check('distance',identity+'_distance_replay',near(distance,saved['new_prediction_pair_distance_l1']))
        coarse_margin=losses['coarse']-1.03*min(losses.values())-1.e-12
        day_margin=day_losses['coarse']-1.03*min(day_losses.values())-1.e-12
        a.check('margin',identity+'_cumulative_M0',near(coarse_margin,saved['cumulative_current_coarse_exit_margin']))
        a.check('margin',identity+'_day_M0',near(day_margin,saved['day_only_current_coarse_exit_margin']))
        plan=plans[row['policy']];request=a.read_ref(row['request'],identity+'_fee_request')
        mean,prices,d,lam,hours,expected,hard=plan_cost(plan)
        actual=mean*(1+d@np.asarray(request['factors']))
        fee=float((hours*prices)@actual)
        a.check('fee',identity+'_response_fee',near(fee,row['actual_payment'],1.e-8))
        a.check('fee',identity+'_budget',fee<=hard+1.e-8 and hard<=1.6+1.e-8)
        own[identity]=dict(policy=row['policy'],path=row['path'],selected_candidate=chosen,day_candidate=day_chosen,
            output_labels=label_family(blocks,pool['labels']),day_output_labels=label_family(day_blocks,pool['labels']),
            distance=distance,coarse_margin=coarse_margin,day_margin=day_margin,fee=fee,losses=losses,day_losses=day_losses)
    # Only now inspect receipt-bound truth.  No result above is changed by it.
    truth_sets=[];rows=[]
    for identity,record in own.items():
        original=indexed[(record['policy'],record['path'])];receipt=receipts[identity]
        truth_path=Path(original['physical_output'])/'truth_evaluation.json'
        a.check('truth_hash',identity+'_truth_receipt_hash',sha(truth_path)==receipt['private_files']['truth_evaluation.json']['sha256'])
        truth_data=a.read(truth_path);truth=family(truth_data['true_supports']);truth_sets.append(truth)
        a.check('truth',identity+'_exact_positive_truth_count',len(truth)==truth_data['full_effective_truth_blocks'])
        a.check('truth',identity+'_zero_weak_convention',truth_data['zero_impedance_merged'] and truth_data['arbitrarily_weak_positive_lines_retained'])
        missing=truth-record['output_labels'];extra=record['output_labels']-truth;error=len(missing)+len(extra)
        day_error=len(truth^record['day_output_labels'])
        a.check('score',identity+'_complete_block_errors',error==original['full_errors'])
        a.check('score',identity+'_missing',family(original['missing'])==missing)
        a.check('score',identity+'_extra',family(original['extra'])==extra)
        a.check('score',identity+'_day_error',day_error==original['day_only_errors']['full_errors'])
        correct=[];raw_correct=[]
        for model in pool['models']:
            name=model['candidate_id'];output_labels=family(model['frozen_output_support_labels'])
            raw_labels=label_family(model['point']['supports'],pool['labels'])
            e=len(truth^output_labels)
            a.check('score',identity+'_'+name+'_candidate_error',e==original['candidate_structural_errors'][name]['full_errors'])
            if e==0:correct.append(name)
            if raw_labels==truth:raw_correct.append(name)
        a.check('coverage',identity+'_output_pool_coverage',bool(correct)==original['true_structure_in_frozen_pool'])
        a.check('coverage',identity+'_correct_ids',correct==original['correct_candidate_ids'])
        wrong=[name for name in model_map if name not in correct]
        exclusion={name:record['losses'][name]-1.03*min(record['losses'].values())-1.e-12 for name in wrong} if correct else {}
        a.check('margin',identity+'_wrong_excluded',bool(exclusion and all(v>0 for v in exclusion.values()))==original['wrong_model_excluded'])
        for name,value in exclusion.items():a.check('margin',identity+'_'+name+'_wrong_margin',near(value,original['cumulative_wrong_exit_margins'][name]))
        baseline=own.get('zero__path'+str(record['path']))
        gain=record['distance']-baseline['distance'] if baseline is not None else None
        if gain is not None:
            a.check('distance',identity+'_paired_gain',near(gain,original['distance_gain']))
            a.check('distance',identity+'_positive_gain',(gain>1.e-12)==original['distance_positive'])
            target=bool(gain>=freeze['gamma_native']-1.e-12) if freeze['positive_gamma'] else None
            a.check('distance',identity+'_target_gain',target==original['distance_target_met'])
        rows.append(dict(record_id=identity,policy=record['policy'],path=record['path'],full_errors=error,
            missing=[list(s) for s in sorted(missing)],extra=[list(s) for s in sorted(extra)],day_full_errors=day_error,
            selected_candidate=record['selected_candidate'],day_candidate=record['day_candidate'],
            raw_correct_candidate_ids=raw_correct,correct_output_candidate_ids=correct,
            coarse_exit_margin=record['coarse_margin'],day_coarse_exit_margin=record['day_margin'],
            distance=record['distance'],distance_gain=gain,actual_payment=record['fee']))
    if truth_sets:a.check('truth','same_static_true_family',all(t==truth_sets[0] for t in truth_sets))
    a.check('denominator','completed_count',len(rows)==summary['completed'])
    for table in summary['table']:
        policy=table['policy'];rr=[r for r in rows if r['policy']==policy]
        a.check('denominator',policy+'_keeps4',table['denominator']==4 and table['complete']==len(rr) and table['unresolved']==4-len(rr))
        a.check('denominator',policy+'_recovery_count',table['recoveries']==sum(r['full_errors']==0 for r in rr))
    return a.finish(rows=rows,requested_records=12,complete_records=len(rows),ledger_costs=costs,
        AC_attempted_day_calls=sum(r.get('AC_attempted',False) for r in costs),
        AC_requested_points_in_attempted_calls=96*sum(r.get('AC_attempted',False) for r in costs),
        completed_valid_AC_points=96*len(rows),truth_files_opened=len(truth_sets),
        probabilities_scope='Four physical backgrounds perpolicy retained; no population90% certificate.',
        audit_scope='Independent original block-sum predictor replay, fixed output mapping,3% selectors and full effective-clade errors.')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('phase',choices=['gate','score'])
    p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path)
    args=p.parse_args();a=Audit(args.root,args.phase)
    destination=args.output or a.root/('external_gate_before_score.json' if args.phase=='gate' else 'independent_score_audit.json')
    if destination.exists():raise FileExistsError(destination)
    try:result=gate(a) if args.phase=='gate' else score_audit(a)
    except Exception as error:
        import traceback
        result=dict(status='failed',audit_passed=False,phase=args.phase,error_type=type(error).__name__,message=str(error),
            traceback=traceback.format_exc(),check_count=sum(a.count.values()),checks_by_category=dict(a.count),failures=a.failed)
    result['audit_source_sha256']=sha(__file__)
    destination.write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({k:result.get(k) for k in ['status','audit_passed','phase','check_count','expected_records','observed_records','complete_records','truth_files_opened','error_type','message']},ensure_ascii=False))
    if result.get('failures'):print(json.dumps(result['failures'][:20],ensure_ascii=False))
    return 0 if result['audit_passed'] else 1


if __name__=='__main__':raise SystemExit(main())
