"""Independent post-freeze physics/settlement/CRN audit for one-day DR tests.

No optimizer, physical producer, identifier, refit or candidate generation is
called.  Actual packets and evaluator-only physical arrays are opened only
after the controller's observations and predictive decisions are frozen.
Static true supports are hash-checked but never parsed by this audit.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from datetime import datetime
from pathlib import Path
import pickle
import traceback

import numpy as np


REPO=Path(r'C:\Users\23761\.codex\worktrees\wzzt-dr-selection\Topo')
HELPER=REPO/'research_experiments/wzzt_dr_selection/dr_conic_pricing/effect_test/audit.py'
spec=importlib.util.spec_from_file_location('_independent_one_day_resource_helpers',HELPER)
old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def stamp(value):return datetime.fromisoformat(value.replace('Z','+00:00'))
def ref(path):return dict(path=str(Path(path).resolve()),sha256=sha(path))
def unpickle(path):
    with Path(path).open('rb') as stream:return pickle.load(stream)


def reference(value):
    p=Path(value['path'])
    if sha(p)!=value['sha256']:raise AssertionError('input hash changed: '+str(p))
    return p


def fingerprint(values):
    digest=hashlib.sha256()
    for value in values:
        a=np.ascontiguousarray(value,float)
        digest.update(str(a.shape).encode());digest.update(a.tobytes())
    return digest.hexdigest()


def audit(root):
    root=Path(root).resolve();folder=root/'actual'
    gates=[folder/'observations_frozen.json',folder/'predictions_frozen.json',folder/'scored.json']
    if any(not p.is_file() for p in gates):
        return dict(status='waiting_for_frozen_actual_predictions',missing=[p.name for p in gates if not p.exists()],
                    actual_arrays_opened=False,static_truth_opened=False,new_solver_calls=0,new_AC=0),None
    checks=[]
    def check(name,condition):
        checks.append(dict(name=name,passed=bool(condition)))
        if not condition:raise AssertionError(name)
    freeze=read(root/'price_freeze.json');c=read(reference(freeze['protocol']))
    observations=read(folder/'observations_frozen.json')
    projections=read(folder/'predictions_frozen.json')
    check('all_prices_and_models_frozen_before_actual',freeze['status']=='frozen' and freeze['models_and_prices_fixed_before_actual'])
    check('observation_freeze_precedes_truth',observations['status']=='frozen' and observations['controller_truth_read'] is False)
    check('prediction_selection_frozen_before_truth',projections['status']=='frozen' and projections['no_true_structure_read_by_controller'] is True)
    check('projection_observation_hash',reference(projections['observations_frozen'])==folder/'observations_frozen.json')
    scored=read(folder/'scored.json')
    check('scored_prediction_freeze_hash',reference(scored['predictions_frozen'])==folder/'predictions_frozen.json')
    for file,wanted in c['source_sha256'].items():check('source_'+Path(file).name,sha(file)==wanted)
    for name in ['frozen_pool','problem','source_snapshot','optimization','planning_summary']:
        reference(freeze[name]);check('freeze_'+name+'_binding',True)
    for name in ['frozen_pool','history_packet','source_metadata','problem']:
        reference(c[name]);check('protocol_'+name+'_binding',True)
    check('same_case_noise_power',c['case']=='flynn16' and c['base_kw']==200. and c['root_label']==1 and c['voltage_sigma']==.0005 and c['pq_sigma']==.005)
    check('new_declared_resource_budget',c['aggregate_units']==5 and c['budget']==1.6 and c['new_training_rows']==0 and c['new_validation_rows']==96 and c['cumulative_validation_rows']==384)
    check('all_history_preserved_in_frozen_reference',c['original_history_rows']==[288,288])
    check('no_new_fit_or_candidate',projections['new_fit_calls']==0 and projections['new_candidate_generation_calls']==0 and scored['new_parameter_fits']==0 and scored['new_pool_regenerations']==0)
    expected={(policy,path) for policy in c['policies'] for path in c['actual_paths']}
    check('twelve_predeclared_records',len(expected)==12 and len(observations['rows'])==len(projections['rows'])==len(scored['rows'])==12 and {(r['policy'],r['path']) for r in observations['rows']}==expected)
    plans={policy:read(reference(item)) for policy,item in freeze['plans'].items()}
    pool=read(reference(c['frozen_pool']))
    check('two_frozen_historical_predictors',len(pool['models'])==2 and pool['new_fit_calls']==pool['new_search_calls']==0 and pool['train_rows']==pool['validation_rows']==288)
    coarse,refined=pool['models'];dr=np.asarray(refined['R'])-np.asarray(coarse['R']);dx=np.asarray(refined['X'])-np.asarray(coarse['X'])
    def distance(p,q):return float(np.abs(np.asarray(p)@dr.T+np.asarray(q)@dx.T).mean())
    hist=unpickle(reference(c['history_packet']))['history']
    check('six_complete_historical_days',len(hist['training'])==len(hist['validation'])==3 and all(len(s['P_terminal'])==96 for s in hist['training']+hist['validation']))
    p=np.stack([s['P_terminal'].to_numpy() for s in hist['training']+hist['validation']]);q=np.stack([s['Q_terminal'].to_numpy() for s in hist['training']+hist['validation']]);labels=c['labels'];base=c['base_kw']
    pmin=p.min(axis=(0,1))*base;qmin=q.min(axis=(0,1))*base
    column=np.abs(dr+.35*dx).sum(axis=0)/(len(labels)*base)*1e6
    qualification=read(root/'resource_qualification.json')
    for r in qualification['nodes']:
        i=labels.index(r['node_label'])
        check(f'history_{labels[i]}_P_min',abs(r['past_P_min_kw']-pmin[i])<1e-12)
        check(f'history_{labels[i]}_Q_min',abs(r['past_Q_min_kvar']-qmin[i])<1e-12)
        check(f'history_{labels[i]}_eligible',r['eligible']==bool(pmin[i]>=1. and qmin[i]>=.35))
        check(f'history_{labels[i]}_column',abs(r['unit_response_column_micro']-column[i])<1e-10)
    focus=set(qualification['dispute_labels']);region=set(qualification['common_region'])
    eligible=[i for i in range(len(labels)) if pmin[i]>=1. and qmin[i]>=.35]
    chosen=[max(part,key=lambda i:(column[i],-labels[i])) for part in
            [[i for i in eligible if labels[i] in focus],[i for i in eligible if labels[i] in region-focus]]]
    check('historical_only_choice201_213',[labels[i] for i in chosen]==qualification['selected_labels']==[201,213])
    history_qualification=[dict(node=labels[i],past_P_min_kw=float(pmin[i]),past_Q_min_kvar=float(qmin[i]),
                                minimum_nonflex_P_reference_kw=float(pmin[i]-1.),minimum_nonflex_Q_reference_kvar=float(qmin[i]-.35)) for i in chosen]
    rows=[];natural={};noise={};path_fingerprints={};physical_valid=ac_attempted=0
    projection_by_id={r['record_id']:r for r in projections['rows']}
    for ledger in observations['rows']:
        rid,policy,path=ledger['record_id'],ledger['policy'],ledger['path']
        projection=projection_by_id[rid]
        receipt=read(reference(ledger['receipt']));request=read(reference(ledger['request']))
        plan=plans[policy];position=c['actual_paths'].index(path)
        check(rid+'_quote_before_actual',stamp(freeze['created_utc'])<=stamp(receipt['started_utc']))
        check(rid+'_receipt_request_hash',receipt['request_sha256_expected']==ledger['request']['sha256'])
        check(rid+'_request_protocol',reference(request['protocol'])==root/'protocol.json')
        check(rid+'_request_plan',request['plan']==freeze['plans'][policy])
        check(rid+'_request_pool',request['frozen_pool']==c['frozen_pool'])
        check(rid+'_frozen_day',request['profile_index']==c['actual_profiles'][position] and request['replicate']==c['actual_replicate_base']+10*path)
        z=np.random.default_rng(np.random.SeedSequence([c['actual_response_seed'],path])).uniform(-1,1,2)
        check(rid+'_prelocked_factors',np.array_equal(request['factors'],z))
        check(rid+'_no_identifier_or_pool_parse',receipt['identifier_run'] is False and receipt['parameter_fitting_run'] is False and receipt['old_or_new_pool_parameters_parsed'] is False and receipt.get('frozen_pool_parsed',False) is False)
        for name,item in receipt['private_files'].items():check(rid+'_private_hash_'+name,sha(Path(ledger['physical_output'])/name)==item['sha256'])
        row=dict(record_id=rid,policy=policy,path=path,status=ledger['status'],worker_status=receipt['status'],
            actual_payment=receipt.get('actual_payment'),known_actual_payment=receipt.get('known_actual_payment',0.),
            unknown_payment_events=receipt.get('unknown_payment_events',1),AC_attempted=receipt.get('AC_attempted',False),
            valid_observation_rows=receipt.get('observed_validation_rows',0),
            frozen_distance_physical=None,frozen_distance_observed=None)
        check(rid+'_payment_receipt_preserved',ledger.get('actual_payment')==receipt.get('actual_payment'))
        if 'prices' not in plan:raise AssertionError('frozen price plan unavailable')
        means=np.asarray(plan['amplitudes'],float);prices=np.asarray(plan['prices'],float)
        elastic=np.asarray(plan['elasticities'],float);d=np.asarray(plan['response_factors'],float)
        actual=means*(1+d@z);resources={r['resource_id']:r for r in plan['resources']}
        check(rid+'_current_inventory',len(means)==10 and set(r['node_label'] for r in resources.values())=={201,213} and np.all(elastic==.5))
        check(rid+'_mean_price',np.allclose(means,elastic*prices,atol=1e-12,rtol=0.) and np.all(prices>=0) and np.all(prices<=2+1e-10))
        dp={key:np.zeros(96) for key in resources};quotes={};hours=[]
        for j,module in enumerate(plan['modules']):
            key,shape=next(iter(module['profiles'].items()));shape=np.asarray(shape,float)
            check(rid+f'_module{j}_energy',len(module['profiles'])==1 and shape.shape==(96,) and abs(shape.sum())<1e-10)
            check(rid+f'_module{j}_cap',module['amplitude_upper_kw']==1. and actual[j]<=1.+1e-7 and actual[j]>=-1e-12)
            if means[j]>0:
                check(rid+f'_module{j}_exclusive',key not in quotes);quotes[key]=prices[j]
            dp[key]+=shape*actual[j];hours.append(.25*np.maximum(-shape,0.).sum())
        hours=np.asarray(hours);independent_fee=float((hours*prices)@actual);hard=float(np.sum(hours*elastic*(1+np.abs(d).sum(axis=1))*prices**2))
        if receipt.get('actual_payment') is not None:
            check(rid+'_independent_actual_fee',abs(receipt['actual_payment']-independent_fee)<1e-9)
            check(rid+'_independent_hard_fee',abs(receipt['worst_payment']-hard)<1e-9)
        check(rid+'_budget',hard<=1.6+1e-9 and independent_fee<=hard+1e-9)
        net_fee=0.
        for key,r in resources.items():
            check(rid+'_'+key+'_inventory',np.all(np.broadcast_to(r['baseline_kw'],(96,))==1.) and np.all(np.broadcast_to(r['power_upper_kw'],(96,))==2.) and np.all(np.broadcast_to(r['reactive_ratio'],(96,))==.35) and np.all(np.broadcast_to(r['prefix_delta_lower_kwh'],(96,))==-2.) and np.all(np.broadcast_to(r['prefix_delta_upper_kwh'],(96,))==2.))
            old.resource_checks(r,dp[key]);net_fee+=float(.25*np.maximum(-dp[key],0.).sum()*quotes.get(key,0.))
        check(rid+'_independent_net_fee',abs(net_fee-independent_fee)<1e-9)
        cornerfees=[]
        for corner in [(-1.,-1.),(-1.,1.),(1.,-1.),(1.,1.)]:
            amp=means*(1+d@np.array(corner));cp={key:np.zeros(96) for key in resources}
            for j,module in enumerate(plan['modules']):
                key,shape=next(iter(module['profiles'].items()));cp[key]+=np.asarray(shape)*amp[j]
                check(rid+f'_corner{corner}_module{j}_amp',0<=amp[j]<=module['amplitude_upper_kw']+1e-7)
            for key,r in resources.items():old.resource_checks(r,cp[key])
            cornerfees.append(sum(.25*np.maximum(-cp[key],0.).sum()*quotes.get(key,0.) for key in resources))
        check(rid+'_corners_budget',max(cornerfees)<=hard+1e-9)
        row.update(independent_contract_fee=independent_fee,independent_hard_cap=hard)
        physical_path=Path(ledger['physical_output'])/'physical_truth.pkl'
        if not physical_path.exists():rows.append(row);continue
        physical=unpickle(physical_path)
        if 'natural_P_pu' not in physical:rows.append(row);continue
        p0=np.asarray(physical['natural_P_pu']);q0=np.asarray(physical['natural_Q_pu']);root0=np.asarray(physical['root_pu'])
        minimum_p=minimum_q=float('inf')
        for key,r in resources.items():
            i=labels.index(r['node_label']);minimum_p=min(minimum_p,float((p0[:,i]*base-1.).min()));minimum_q=min(minimum_q,float((q0[:,i]*base-.35).min()))
        component=minimum_p>=-1e-8 and minimum_q>=-1e-8
        check(rid+'_component_report',receipt.get('contained_component_pass')==component)
        check(rid+'_component_P',abs(receipt['minimum_nonflexible_p_kw']-minimum_p)<1e-10)
        check(rid+'_component_Q',abs(receipt['minimum_nonflexible_q_kvar']-minimum_q)<1e-10)
        row.update(component_supported=component,minimum_nonflexible_P_kw=minimum_p,minimum_nonflexible_Q_kvar=minimum_q)
        natural_values=(p0,q0,root0,np.asarray(physical['factors']))
        if path in natural:check(rid+'_natural_and_response_CRN',all(np.array_equal(a,b) for a,b in zip(natural_values,natural[path],strict=True)))
        else:natural[path]=natural_values;path_fingerprints[path]=fingerprint(natural_values[:3])
        ac_attempted+=int(receipt.get('AC_attempted',False))
        if receipt['status']!='complete':rows.append(row);continue
        check(rid+'_component_available',component)
        packet=unpickle(reference(ledger['packet']))
        check(rid+'_validation_only96',len(packet['validation'])==1 and len(packet['validation'][0]['P_terminal'])==96 and 'training' not in packet)
        scene=packet['validation'][0]
        pp,qq=np.asarray(physical['P_pu']),np.asarray(physical['Q_pu'])
        pdelta=np.zeros_like(pp);qdelta=np.zeros_like(qq)
        for key,r in resources.items():
            i=labels.index(r['node_label']);pdelta[:,i]+=dp[key];qdelta[:,i]+=.35*dp[key]
        check(rid+'_once_active_increment',np.allclose((pp-p0)*base,pdelta,atol=1e-9,rtol=0.) and np.allclose(pdelta,physical['delta_kw'],atol=1e-12,rtol=0.))
        check(rid+'_once_reactive_increment',np.allclose((qq-q0)*base,qdelta,atol=1e-9,rtol=0.) and np.allclose(qdelta,physical['delta_kvar'],atol=1e-12,rtol=0.))
        check(rid+'_actual_amplitude',np.allclose(physical['actual_amplitudes'],actual,atol=1e-12,rtol=0.))
        check(rid+'_no_negative_target_load',min(pp[:,labels.index(r['node_label'])].min() for r in resources.values())>=-1e-10)
        checks_phys=receipt['checks']
        check(rid+'_AC_and_operation',receipt['AC_attempted'] and checks_phys['ac_all_96_converged'] and checks_phys['electrical_equations_pass'] and checks_phys['voltage_limits_pass'] and checks_phys['branch_ampacity_checked'] is False)
        physical_valid+=1
        measured=[scene['P_terminal'].to_numpy(),scene['Q_terminal'].to_numpy(),scene['V_terminal'].to_numpy(),scene['V_root_observed'].to_numpy()]
        true=[pp,qq,np.asarray(physical['V_pu']),root0]
        gain=[np.divide(a,b,out=np.zeros_like(a),where=np.abs(b)>1e-12) for a,b in zip(measured,true,strict=True)]
        mask=[np.abs(b)>1e-12 for b in true]
        if path in noise:
            for j,((prev,oldmask),current,newmask) in enumerate(zip(noise[path],gain,mask,strict=True)):
                use=oldmask&newmask
                check(rid+f'_measurement_channel{j}_CRN',np.max(np.abs(prev[use]-current[use]),initial=0.)<=1e-12)
        else:noise[path]=list(zip(gain,mask))
        qphys=distance(pp,qq);qobs=distance(measured[0],measured[1]);qnatural=distance(p0,q0)
        score=read(reference(projection['scores']))
        check(rid+'_observed_distance_independent',abs(qobs-score['new_prediction_pair_distance_l1'])<1e-12)
        row.update(valid_observation_rows=96,frozen_distance_physical=qphys,
            frozen_distance_observed=qobs,natural_physical_distance=qnatural)
        rows.append(row)
    check('no_duplicate_natural_backgrounds',len(set(path_fingerprints.values()))==len(path_fingerprints))
    gamma=freeze['gamma_native']
    for row in rows:
        zero=next(r for r in rows if r['policy']=='zero' and r['path']==row['path'])
        if row['frozen_distance_physical'] is not None and zero['frozen_distance_physical'] is not None:
            phys=row['frozen_distance_physical']-zero['frozen_distance_physical']
            obs=row['frozen_distance_observed']-zero['frozen_distance_observed']
            row.update(paired_physical_distance_gain=phys,paired_observed_distance_gain=obs,
                physical_positive_gain=phys>1e-12,observed_positive_gain=obs>1e-12,
                physical_target_met=phys>=gamma-1e-12 if freeze['positive_gamma'] else None,
                observed_target_met=obs>=gamma-1e-12 if freeze['positive_gamma'] else None)
    table=[]
    for policy in c['policies']:
        rr=[r for r in rows if r['policy']==policy];valid=[r for r in rr if r['frozen_distance_physical'] is not None]
        table.append(dict(policy=policy,denominator=4,valid=len(valid),failed_or_unresolved=4-len(valid),
            actual_fee_known_sum=sum(r['known_actual_payment'] for r in rr),
            unknown_payment_events=sum(r['unknown_payment_events'] for r in rr),
            mean_actual_fee=sum(r['actual_payment'] for r in rr)/4 if all(r['actual_payment'] is not None for r in rr) else None,
            actual_fee_by_path=[r['actual_payment'] for r in rr],
            physical_distance_gains=[r.get('paired_physical_distance_gain') for r in rr],
            observed_distance_gains=[r.get('paired_observed_distance_gain') for r in rr],
            physical_positive_gain_count=sum(r.get('physical_positive_gain',False) for r in rr),
            observed_positive_gain_count=sum(r.get('observed_positive_gain',False) for r in rr),
            physical_target_count=sum(r.get('physical_target_met') is True for r in rr),
            observed_target_count=sum(r.get('observed_target_met') is True for r in rr)))
    report=dict(status='complete',checks=checks,scalar_check_count=len(checks),passed_checks=sum(c['passed'] for c in checks),
        rows=rows,table=table,historical_component_qualification=history_qualification,
        requested_records=12,independent_predeclared_natural_days=4,distinct_completed_natural_backgrounds=len(path_fingerprints),
        valid_physical_observations=physical_valid,AC_attempted_days=ac_attempted,valid_AC_points=96*physical_valid,
        gamma_native=gamma,positive_target=freeze['positive_gamma'],
        new_AC=0,new_solver_calls=0,new_refits=0,static_truth_supports_parsed=False,
        price_freeze_sha256=sha(root/'price_freeze.json'),protocol_sha256=sha(root/'protocol.json'),
        observations_frozen_sha256=sha(folder/'observations_frozen.json'),predictions_frozen_sha256=sha(folder/'predictions_frozen.json'),
        source_sha256=sha(__file__),independent_helper_sha256=sha(HELPER),
        scope='physics, resource settlement, common random numbers and frozen-map distance; no topology-success probability',
        boundaries=['Four paths are predeclared profile-stratified simulated days, not12 independent days.',
                    'History qualification checks noisy archived loads, not calibrated participant inventories.',
                    'Physical-input distance uses evaluator-only noiseless simulated P/Q after quotes and predictions freeze.',
                    'No branch thermal rating is checked by the inherited physical checker.'])
    private=dict(protocol=ref(root/'protocol.json'),price_freeze=ref(root/'price_freeze.json'),
        observations_frozen=ref(folder/'observations_frozen.json'),predictions_frozen=ref(folder/'predictions_frozen.json'),
        scored=ref(folder/'scored.json'),frozen_pool=c['frozen_pool'],history=c['history_packet'])
    return report,private


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    if args.output.exists():raise FileExistsError('retain original audit output')
    try:report,private=audit(args.root)
    except Exception as error:
        report=dict(status='audit_failed',error_type=type(error).__name__,message=str(error),traceback=traceback.format_exc(),source_sha256=sha(__file__),new_AC=0,new_solver_calls=0)
        private=None
    if report['status'].startswith('waiting'):
        print(json.dumps(report));return 3
    args.output.parent.mkdir(parents=True,exist_ok=True)
    if private is not None:
        file=args.output.with_name(args.output.stem+'_private_manifest.json')
        with file.open('x',encoding='utf-8') as stream:json.dump(private,stream,indent=2,ensure_ascii=False,allow_nan=False)
        report['private_manifest_sha256']=sha(file)
    with args.output.open('x',encoding='utf-8') as stream:json.dump(report,stream,indent=2,ensure_ascii=False,allow_nan=False);stream.write('\n')
    print(json.dumps({k:report.get(k) for k in ['status','scalar_check_count','passed_checks','table','valid_physical_observations','AC_attempted_days','message']},ensure_ascii=False))
    return 0 if report['status']=='complete' else 2


if __name__=='__main__':raise SystemExit(main())
