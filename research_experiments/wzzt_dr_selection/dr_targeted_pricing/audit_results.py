"""Independent read-only audit of targeted finite-price DR actual study.

Terminal gates are checked before opening any trajectory, coefficient or truth
artifact. Imports no identifier, original score/select, optimizer or AC solver.
The pre-existing independent mathematical helper is pinned by SHA256; only its
frame, matrix-L1, family, and selection/acceptance audit functions are reused.
All new checks and exported records contain scalars, hashes and categorical IDs.
"""
from __future__ import annotations
import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import importlib.util
import itertools
import json
import os
from pathlib import Path
import pickle
import random
import traceback
for _k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_k]='1'
import numpy as np

BASE=Path(__file__).resolve().parents[1]
HELPER=BASE/'wzzt_day_module_actual_v1/audit_independent_v2.py'
HELPER_SHA='d4d344087413c5ed1e4df83ecff0fb1e08c7de9a7043beca42cf5ca65df38f87'
STATES=((0,0),(1,0),(0,1),(1,1))
PROB=(Fraction(0),Fraction(1,2),Fraction(3,4))
E=Fraction(2,5)
BUDGET=Fraction(8,5)
MODELS=('rnj_bank','baseline25','multistart_original_rule')

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def load(p):
    with Path(p).open('rb') as f:return pickle.load(f)
def stamp(s):return datetime.fromisoformat(s.replace('Z','+00:00'))
def fr(x):return Fraction(str(x))
def fdict(x):return dict(numerator=x.numerator,denominator=x.denominator,value=float(x))
def now():return datetime.now(timezone.utc).isoformat()

def helper(path=None):
    path=HELPER if path is None else Path(path).resolve()
    if sha(path)!=HELPER_SHA:raise RuntimeError('Independent mathematical helper hash mismatch')
    spec=importlib.util.spec_from_file_location('pinned_independent_math',path)
    h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h);return h

def pp(ci,cj):
    pi,pj=PROB[ci],PROB[cj]
    return ((1-pi)*(1-pj),pi*(1-pj),(1-pi)*pj,pi*pj)

def normalize_prices(value):
    if isinstance(value,list):return [normalize_prices(v) for v in value]
    if not isinstance(value,dict):return value
    out={k:normalize_prices(v) for k,v in value.items()}
    if 'prices' in out:out['prices']=[int(v) for v in out['prices']]
    return out

def static_truth(root_label,labels,buses_path,branches_path):
    buses=list(csv.DictReader(Path(buses_path).open(encoding='utf-8-sig')))
    terminals={int(b['bus_id']) for b in buses if b['bus_type']=='observed_terminal'}
    if terminals!=set(labels):raise RuntimeError('Static terminal labels differ from observed columns')
    adjacent={int(b['bus_id']):[] for b in buses};seen=set();family=set()
    for e in csv.DictReader(Path(branches_path).open(encoding='utf-8-sig')):
        if str(e.get('status','True')).lower() not in ('true','1') or str(e.get('is_true_closed','True')).lower() not in ('true','1'):continue
        i,j=int(e['from_bus']),int(e['to_bus']);adjacent[i].append((j,e));adjacent[j].append((i,e))
    def visit(i,parent):
        if i in seen:raise RuntimeError('Static topology contains a cycle')
        seen.add(i);desc={i}&terminals
        for j,e in adjacent[i]:
            if j==parent:continue
            s=visit(j,i);desc|=s
            if s and (float(e['r_ohm'])!=0. or float(e['x_ohm'])!=0.):family.add(tuple(sorted(s)))
        return desc
    if visit(root_label,None)!=terminals or seen!=set(adjacent):raise RuntimeError('Disconnected static truth')
    return family

def exact_columns(c,lo,hi,modules,tied=False):
    columns=[]
    for m in modules:
        k=c['modules'].index(m)
        vectors=([ci,cj,ci,cj] for ci,cj in itertools.product(range(3),repeat=2)) if tied else itertools.product(range(3),repeat=4)
        for flat in vectors:
            ci,cj,vi,vj=flat;train=pp(ci,cj);valid=pp(vi,vj)
            lower=sum((train[t]*valid[v]*fr(lo[k][t][v]) for t in range(4) for v in range(4)),Fraction())
            upper=sum((train[t]*valid[v]*fr(hi[k][t][v]) for t in range(4) for v in range(4)),Fraction())
            expected=E*sum((c0*PROB[c0] for c0 in flat),Fraction());worst=E*sum(flat)
            if worst<=BUDGET:
                columns.append(dict(module_id=m,prices=[[ci,cj],[vi,vj]],lower=lower,upper=upper,fee=expected,worst=worst,training_probabilities=train,validation_probabilities=valid))
    return columns

def check_optimizer(saved,c,lo,hi,modules,seed,A,where,tied=False):
    columns=exact_columns(c,lo,hi,modules,tied=tied)
    price_rows=lambda p:[[int(x) for x in p],[int(x) for x in p]] if tied else [[int(x) for x in row] for row in p]
    flat=lambda p:tuple(itertools.chain.from_iterable(p))
    savedcols={(v['module_id'],flat(price_rows(v['prices']))):v for v in saved['feasible_columns']}
    A.check(len(savedcols)==len(columns),'finite_feasible_quote_domain',where)
    A.check(len(columns)==len(modules)*(6 if tied else 50),'complete_restricted_or_four_price_quote_domain',where)
    for col in columns:
        k=(col['module_id'],flat(col['prices']));s=savedcols.get(k)
        A.check(s is not None,'finite_feasible_quote_present',where)
        if s is None:continue
        for key,ind in (('risk_lower',col['lower']),('risk_upper',col['upper']),('expected_fee',col['fee']),('worst_fee',col['worst'])):
            A.close(float(ind),s[key],'fraction_'+key,where+':'+str(k),atol=1e-12,rtol=0.)
        for axis in ('training','validation'):
            savedkey='pair_state_probabilities' if tied else axis+'_pair_probabilities'
            A.close([float(p) for p in col[axis+'_probabilities']],s[savedkey],'fraction_'+axis+'_pair_probabilities',where+':'+str(k),atol=0.,rtol=0.)
    minrisk=min(x['upper'] for x in columns)
    # The actual locked program has a declared 1e-9 risk/fee tie band.
    rt=Fraction(1,10**9);ft=rt
    riskset=[x for x in columns if x['upper']<=minrisk+rt]
    minfee=min(x['fee'] for x in riskset)
    ties=sorted((x for x in riskset if x['fee']<=minfee+ft),key=lambda x:(x['module_id'],*flat(x['prices'])))
    chosen=random.Random(seed).choice(ties)
    exact=all(x['lower']==x['upper'] for x in columns)
    A.check(saved['risk_table_exact']==exact,'exact_loss_domain_status',where)
    A.check(saved['status']=='optimal' if exact else saved['selected'] is None or saved['status']=='interval_dominance_certified','preserved_missing_risk_status',where)
    if exact:
        s=saved['selected'];A.check(s['module_id']==chosen['module_id'] and price_rows(s['prices'])==chosen['prices'],'fraction_exact_lex_and_locked_random_tie',where)
        savedties=[(x['module_id'],price_rows(x['prices'])) for x in saved['all_exact_lex_ties']]
        A.check(savedties==[(x['module_id'],x['prices']) for x in ties],'complete_exact_lex_tie_set',where)
        k=c['modules'].index(s['module_id']);train=chosen['training_probabilities'];valid=chosen['validation_probabilities']
        ml=[sum((valid[v]*fr(lo[k][t][v]) for v in range(4)),Fraction()) for t in range(4)]
        mh=[sum((valid[v]*fr(hi[k][t][v]) for v in range(4)),Fraction()) for t in range(4)]
        gamma_lo=ml[3]+ml[0]-mh[1]-mh[2];gamma_hi=mh[3]+mh[0]-ml[1]-ml[2]
        A.close([float(gamma_lo),float(gamma_hi)],[s['training_Gamma']['lower'],s['training_Gamma']['upper']],'Gamma_after_validation_integration',where,atol=1e-12,rtol=0.)
        vl=[sum((train[t]*fr(lo[k][t][v]) for t in range(4)),Fraction()) for v in range(4)];vh=[sum((train[t]*fr(hi[k][t][v]) for t in range(4)),Fraction()) for v in range(4)]
        vgamma_lo=vl[3]+vl[0]-vh[1]-vh[2];vgamma_hi=vh[3]+vh[0]-vl[1]-vl[2]
        if not tied:A.close([float(vgamma_lo),float(vgamma_hi)],[s['validation_Gamma']['lower'],s['validation_Gamma']['upper']],'Gamma_after_training_integration',where,atol=1e-12,rtol=0.)
    else:gamma_lo=gamma_hi=vgamma_lo=vgamma_hi=None
    for solver in saved['solver_receipts']:
        A.check(solver['solver_reported_optimal'] and solver['independent_primal_pass'],'saved_MILP_optimal_and_primal_receipt',where)
    return dict(complete_finite_risks=exact,feasible_quotes=len(columns),minimum_risk=fdict(minrisk),minimum_expected_fee=fdict(minfee),
        selected_module=chosen['module_id'],selected_prices=chosen['prices'],ties=len(ties),
        training_Gamma_lower=None if gamma_lo is None else fdict(gamma_lo),training_Gamma_upper=None if gamma_hi is None else fdict(gamma_hi),validation_Gamma_lower=None if vgamma_lo is None else fdict(vgamma_lo),validation_Gamma_upper=None if vgamma_hi is None else fdict(vgamma_hi))

def start_audit(result,training,validation,C,A,H,where):
    labels=result['terminal_labels'];options=result['start_preparation'];bank=result['bank'];cal=result['calibrations']['raw']
    A.check(len(options)==len(bank)==15,'all_fifteen_extra_start_preparations',where)
    canonical=lambda supports:tuple(sorted(tuple(sorted(s)) for s in supports))
    for i,(b,p) in enumerate(zip(bank,options)):
        cut=[s for s,r,x in zip(b['supports'],b['r'],b['x']) if len(s)==1 or cal['a']*r+cal['b']*x+1e-10>=.16*cal['median_root_depth']]
        A.check(canonical(cut)==canonical(p['full_fit']['supports']),'prepared_start_from_contract_bank_then_all_weight_refit',where+':'+str(i))
        A.close(H.point_loss(p,training),p['objective'],'prepared_start_joint_train_loss',where+':'+str(i))
        A.close(H.point_loss(p,validation),p['validation_mae'],'prepared_start_validation_loss',where+':'+str(i))
    primary=next(b for b in bank if b['config']=='raw_RX75_0.16');seen={canonical(primary['native_family'])};chosen=[]
    ranking=sorted(enumerate(options),key=lambda x:(x[1]['validation_mae'],len(x[1]['supports']),x[0]))
    for i,p in ranking:
        f=canonical(p['supports'])
        if f in seen:continue
        seen.add(f);chosen.append(i)
        if len(chosen)==2:break
    A.check(chosen==result['extra_start_sources'],'finite_two_extra_start_admission_order',where)
    for serial,i in enumerate(chosen):A.check(canonical(result['searches'][f'multistart_{serial}']['source_family'])==canonical(options[i]['supports']),'admitted_start_exact_family',where)
    has=lambda p:C in H.family(p,labels)
    return dict(prepared_target_count=sum(has(p) for p in options),admitted_extra_start_sources=chosen,admitted_target_starts=sum(has(options[i]) for i in chosen),
        prepared_target_best_rank=min((rank for rank,(i,p) in enumerate(ranking) if has(p)),default=None))

def physical(c,row,req,packet,private,receipt,plan,A,H,common):
    where=row['phase']+':'+row['record_id'];labels=c['labels'];n=len(labels);total=full=Fraction();completed=0
    scenes={s['actual_event_id']:s for split in ('training','validation') for s in packet[split]}
    A.check(len(private['events'])==4,'four_physical_events_retained',where)
    A.check(receipt['voltage_sigma']==c['voltage_sigma']==.0005 and receipt['pq_sigma']==c['pq_sigma']==.005,'locked_future_meter_noise',where)
    A.check(receipt['root_label']==c['root_label'] and receipt['terminal_labels']==labels and receipt['base_kw']==c['base_kw'],'physical_metadata_binding',where)
    for event,eventtruth in enumerate(private['events']):
        r=receipt['event_records'][event]
        if eventtruth.get('status')!='complete':continue
        completed+=1;scene=scenes[event];dp=np.zeros((96,n));dq=dp.copy();payment=Fraction();maxpay=Fraction()
        for resource in plan['resources_by_event'][event]:
            key=resource['resource_id'];planned=sum((float(a)*H.vec(m['profiles'][key]) for a,m in zip(plan['amplitudes_by_event'][event],plan['modules_by_event'][event]) if key in m['profiles']),np.zeros(96))
            actual=req['rho_by_event'][event][key]*planned;col=labels.index(resource['node_label']);dp[:,col]+=actual;dq[:,col]+=H.vec(resource['reactive_ratio'])*actual
            for slot in range(96):
                price=fr(float(H.vec(resource['incentive_per_kwh'])[slot]));payment+=Fraction(1,4)*price*max(-fr(float(actual[slot])),Fraction());maxpay+=Fraction(1,4)*price*max(-fr(float(planned[slot])),Fraction())
            baseline=H.vec(resource['baseline_kw']);lo=H.vec(resource.get('power_lower_kw',0.));hi=H.vec(resource['power_upper_kw']);elo=H.vec(resource['prefix_delta_lower_kwh']);ehi=H.vec(resource['prefix_delta_upper_kwh']);available=np.broadcast_to(np.asarray(resource['available'],bool),(96,))
            for tag,delta in (('planned',planned),('actual',actual)):
                energy=.25*np.cumsum(delta)
                A.check(np.all(baseline+delta>=lo-1e-9) and np.all(baseline+delta<=hi+1e-9),'resource_power_domain',where+':'+tag)
                A.check(np.all(energy>=elo-1e-9) and np.all(energy<=ehi+1e-9) and abs(energy[-1])<1e-9,'resource_energy_domain',where+':'+tag)
                A.check(np.max(np.abs(delta[~available]),initial=0.)<1e-10,'resource_availability',where+':'+tag)
            A.close(actual,eventtruth['resource_audit']['resources'][key]['executed_delta_kw'],'actual_whole_day_equals_rho_times_fixed_task',where)
            if event not in c['active_events']:A.close(planned,np.zeros(96),'inactive_event_has_no_task',where,atol=0.,rtol=0.)
        p=np.asarray(eventtruth['P_pu']);q=np.asarray(eventtruth['Q_pu']);v=np.asarray(eventtruth['V_pu']);root=np.asarray(eventtruth['root_pu'])
        A.close((p-eventtruth['natural_P_pu'])*c['base_kw'],dp,'actual_increment_added_once_P',where,atol=1e-9,rtol=0.)
        A.close((q-eventtruth['natural_Q_pu'])*c['base_kw'],dq,'actual_increment_added_once_Q',where,atol=1e-9,rtol=0.)
        A.close(dp,eventtruth['delta_kw'],'private_increment_P',where,atol=1e-12,rtol=0.);A.close(dq,eventtruth['delta_kvar'],'private_increment_Q',where,atol=1e-12,rtol=0.)
        A.close(float(payment),r['actual_payment'],'exact_decimal_event_payment',where);A.close(float(maxpay),r['full_delivery_incentive_payment'],'exact_decimal_event_commitment',where)
        total+=payment;full+=maxpay
        A.check(r['checks']['ac_all_96_converged'] and r['checks']['electrical_equations_pass'] and r['checks']['voltage_limits_pass'],'saved_AC_equations_and_voltage_receipt',where)
        A.check(bool(np.asarray(eventtruth['ac']['converged']).all()),'private_96_AC_converged',where)
        A.check(np.min(v)>=.9-1e-10 and np.max(v)<=1.1+1e-10,'independent_voltage_limits',where)
        A.check(r['replicate']==c['replicate_base']+10*row['path']+event and r['profile_index']==c['event_profiles'][event],'frozen_future_producer_indices',where)
        rng=np.random.default_rng(r['profile_seed']+2000)
        expected_p=p+rng.normal(0.,c['pq_sigma']*np.maximum(np.abs(p),1e-12),size=p.shape)
        expected_q=q+rng.normal(0.,c['pq_sigma']*np.maximum(np.abs(q),1e-12),size=q.shape)
        expected_v=v+rng.normal(0.,c['voltage_sigma']*np.maximum(np.abs(v),1e-12),size=v.shape)
        rootseed=70_000_019+100_003*r['replicate']+r['profile_index']
        expected_root=root+np.random.default_rng(rootseed).normal(0.,c['voltage_sigma']*np.maximum(np.abs(root),1e-12),size=len(root))
        for key,expected in zip(H.KEYS,(expected_p,expected_q,expected_v,expected_root)):
            A.close(scene[key].to_numpy(),expected,'independent_noise_replay_'+key,where,atol=5e-15,rtol=1e-13)
        sig=tuple(H.digest(eventtruth[k]) for k in ('natural_P_pu','natural_Q_pu','root_pu'))+(int(r['profile_seed']),int(rootseed))
        key=(row['phase'],row['path'],event)
        if key in common:A.check(sig==common[key],'common_background_and_standardized_errors',where)
        else:common[key]=sig
    if receipt['status']=='complete':
        A.check(completed==4,'complete_receipt_all_events_retained',where)
        A.close(float(total),receipt['actual_payment'],'four_event_actual_payment',where);A.close(float(full),receipt['worst_payment'],'four_event_commitment_payment',where)
        A.check(full<=BUDGET and total<=full,'same_hard_four_event_budget',where)
    return dict(physical_events_complete=completed,actual_fee=float(total) if completed==4 else None,commitment_fee=float(full))

def run(root,output,helper_path=None):
    root=Path(root).resolve();c=read(root/'protocol.json');phash=sha(root/'protocol.json')
    required=[root/p/n for p in ('planning','test') for n in ('observations.json','fits_frozen.json','scored.json')]
    required += [root/p/f'worker_{w}_ended.json' for p in ('planning','test') for w in range(c['workers'])]
    required += [root/n for n in ('summary.json','oracle_selected.json','event_price_optimization.json','decision_protocol.json','event_price_decision_lock.json','planning/loss_table.json','baseline_fits_frozen.json')]
    missing=[str(p.relative_to(root)) for p in required if not p.exists()]
    if missing:raise RuntimeError('No private data opened: missing terminal gates '+','.join(missing))
    H=helper(helper_path);A=H.Audit();phases={p:{n:read(root/p/(n+'.json')) for n in ('observations','fits_frozen','scored')} for p in ('planning','test')};jobs={}
    for phase,d in phases.items():
        f=d['fits_frozen'];s=d['scored'];o=d['observations']
        if f['status']!='frozen' or f['observation_ledger_sha256']!=sha(root/phase/'observations.json') or s['fits_frozen_sha256']!=sha(root/phase/'fits_frozen.json'):
            raise RuntimeError('No private data opened: invalid terminal binding '+phase)
        j={x['key']:x for x in f['jobs']};ended={x['key']:x for w in range(c['workers']) for x in read(root/phase/f'worker_{w}_ended.json')['rows']}
        if j!=ended or set(j)!={x['key'] for x in o['jobs']}:raise RuntimeError('No private data opened: incomplete worker freeze '+phase)
        jobs[phase]=j
    for p,h in c['source_sha256'].items():A.check(sha(p)==h,'locked_executed_source_hash',Path(p).name)
    bp=read(A.bound(c['baseline_protocol'],'baseline_protocol'))
    for name,v in bp['inputs'].items():A.bound(v,'baseline_source_'+name)
    selection=read(A.bound(bp['inputs']['case_selection_lock'],'source_selection'))
    A.check(selection['selected_job_id']==c['job_id']==bp['job_id'],'pre_DR_selected_source_binding')
    A.check(stamp(selection['created_utc'])<stamp(bp['created_utc'])<stamp(c['created_utc']),'case_selection_before_baseline_and_DR')
    A.check(selection['no_DR_or_future_test_results_generated'] and selection['new_source_baseline_not_yet_run'],'no_DR_outcome_source_selection')
    A.bound(selection['cohort_reference'],'source_selection_cohort')
    history=load(A.bound(c['history_packet'],'complete_history'))['history']
    A.check([sum(len(s['P_terminal']) for s in history[k]) for k in ('training','validation')]==[288,288],'complete_288_288_history')
    A.check(c['original_history_rows']==[288,288] and c['future_rows']==[480,480] and c['active_events']==[0,2] and c['pair_states']==[list(s) for s in STATES],'locked_state_and_row_domains')
    truth_static=static_truth(c['root_label'],c['labels'],bp['inputs']['static_buses']['path'],bp['inputs']['static_branches']['path'])
    be=read(root/'baseline_ended.json');baseline=read(A.bound(be['result'],'baseline_fit'))
    baseline_audit=H.selection_audit(baseline,H.frames(history['training'],A,'baseline:training'),H.frames(history['validation'],A,'baseline:validation'),truth_static,A,'baseline')
    saved_baseline=read(A.bound(c['baseline_evaluation'],'baseline_evaluation'))
    for name in MODELS:A.check(baseline_audit['models'][name]['final_errors']==saved_baseline['scores'][name]['full_errors'],'baseline_independent_full_errors',name)
    plans={k:load(A.bound(v,'plan:'+k)) for k,v in c['plans'].items()}
    decision=read(root/'decision_protocol.json');decision_lock=read(A.bound(decision['decision_lock'],'decision_lock'));A.bound(decision['base_protocol'],'base_protocol')
    A.check(decision['status']==decision_lock['status']=='frozen' and decision_lock['base_protocol_sha256']==phash,'decision_addendum_base_binding')
    A.check(stamp(decision_lock['created_utc'])<stamp(decision['created_utc'])<stamp(phases['planning']['fits_frozen']['created_utc']),'four_price_decision_lock_and_sources_before_score_freeze')
    A.check(decision_lock['planning_scored_has_not_been_created'] and decision_lock['test_not_created'] and decision['precomputation_new_W_calls']==0,'decision_locked_before_score_and_test_without_new_W')
    A.check(decision_lock['variable_count']==4 and not decision_lock['physical_task_or_noise_or_history_changes'],'four_price_only_extension')
    for p,h in decision['source_sha256'].items():A.check(sha(p)==h,'decision_executed_source_hash',Path(p).name)
    eventplans={k:load(A.bound(v,'event_plan:'+k)) for k,v in decision['plans'].items()}
    vectors=[list(x) for x in itertools.product(range(3),repeat=4) if sum(x)<=4]
    A.check(decision['price_vectors']==vectors and len(vectors)==50 and len(eventplans)==100,'fifty_four_price_vectors_and_hundred_bound_plans')
    for m in c['modules']:
        template=plans[c['plan_lookup'][m]['1,1']]
        for flat in vectors:
            key=decision['lookup'][m][','.join(map(str,flat))];dto=eventplans[key]
            for event in range(4):
                expected_prices=flat[:2] if event==0 else flat[2:] if event==2 else [0,0]
                A.check(dto['amplitudes_by_event'][event]==template['amplitudes_by_event'][event],'event_plan_amplitudes_unchanged',key)
                for i,(r0,r1) in enumerate(zip(template['resources_by_event'][event],dto['resources_by_event'][event])):
                    A.check({k:v for k,v in r0.items() if k!='incentive_per_kwh'}=={k:v for k,v in r1.items() if k!='incentive_per_kwh'} and r1['incentive_per_kwh']==expected_prices[i],'event_plan_only_prices_changed',key)
                for md0,md1 in zip(template['modules_by_event'][event],dto['modules_by_event'][event]):
                    A.check(set(md0['profiles'])==set(md1['profiles']),'event_plan_profile_keys_unchanged',key)
                    for rid in md0['profiles']:A.close(md0['profiles'][rid],md1['profiles'][rid],'event_plan_task_profile_unchanged',key,atol=0.,rtol=0.)
    oracle=read(root/'oracle_selected.json');opt=read(root/'event_price_optimization.json');loss=read(root/'planning/loss_table.json')
    A.check(oracle['protocol_sha256']==phash and oracle['planning_scored_sha256']==sha(root/'planning/scored.json'),'oracle_protocol_and_planning_hash')
    A.check(oracle['decision_protocol_sha256']==sha(root/'decision_protocol.json'),'oracle_decision_protocol_hash')
    A.check(oracle['loss_table_sha256']==sha(root/'planning/loss_table.json') and oracle['price_optimization_sha256']==sha(root/'event_price_optimization.json'),'oracle_loss_and_optimization_hash')
    A.check(stamp(oracle['created_utc'])>=stamp(phases['planning']['fits_frozen']['created_utc']),'price_after_planning_fit_freeze')
    addendum_time=stamp(decision['created_utc'])
    if (root/'quote_dtype_receipt.json').exists():
        receipt=read(root/'quote_dtype_receipt.json');old=read(root/'oracle_selected_before_dtype_canonicalization.json');add=read(root/'quote_dtype_addendum.json')
        A.check(sha(root/'oracle_selected_before_dtype_canonicalization.json')==receipt['original_sha256']==add['original_oracle_sha256'],'original_quote_preserved_hash')
        A.check(sha(root/'oracle_selected.json')==receipt['canonical_sha256'] and sha(root/'quote_dtype_addendum.json')==receipt['addendum_sha256'],'canonical_quote_and_addendum_hash')
        stripped={k:v for k,v in oracle.items() if k not in ('quote_dtype_addendum','quote_canonicalization_utc')}
        A.check(normalize_prices(old)==stripped and receipt['new_experiment_calls']==0,'quote_dtype_repair_no_mathematical_changes')
        A.check(add['test_directory_absent'] and add['solver_calls']==add['AC_calls']==add['identifier_calls']==0,'schema_repair_precedes_test_without_experiment')
        addendum_time=stamp(add['created_utc'])
    errors_by_record={};errors_by_model={k:{} for k in MODELS};actual={};common={};fit_cache={};phase_report={};physical_count=0;target_review=[]
    for phase,d in phases.items():
        o,s=d['observations'],d['scored'];scored={r['record_id']:r for r in s['rows']};records=[]
        A.check(len(scored)==len(o['rows'])==s['planned'],'all_success_failure_unresolved_denominators',phase)
        A.check(o['protocol_sha256']==phash,'phase_protocol_hash',phase)
        for row0 in o['rows']:
            row=dict(row0,phase=phase);rid=row['record_id'];where=phase+':'+rid;location=root/phase/'records'/rid;score=scored[rid];req=read(location/'request.json')
            if phase=='planning':
                planname=c['plan_lookup'][row['module']][','.join(map(str,row['prices']))];plan=plans[planname];planref=c['plans'][planname]
                expected=[{key:0 for key in c['resource_ids']} for _ in range(4)]
                for event,state in ((0,row['train_state']),(2,row['validation_state'])):expected[event]=dict(zip(c['resource_ids'],STATES[state]))
                A.check(row['prices']==[1,1] and row['payment_scope']=='nominal basis price1','planning_fee_is_basis_not_optimized_fee',where)
            else:
                flat=list(itertools.chain.from_iterable(row['prices']));planname=decision['lookup'][row['module']][','.join(map(str,flat))];plan=eventplans[planname];planref=decision['plans'][planname]
                policy=oracle['test_policies'][row['policy']];A.check(row['module']==policy['module_id'] and row['prices']==policy['prices'],'test_uses_frozen_policy',where)
                expected=[]
                for event in range(4):
                    u=np.random.default_rng(np.random.SeedSequence([c['participation_seed'],row['path'],event])).random(2)
                    expected.append({key:int(u[i]<float(PROB[policy['prices'][event//2][i]])) if event in c['active_events'] else 0 for i,key in enumerate(c['resource_ids'])})
                A.check(req['oracle']['sha256']==sha(root/'oracle_selected.json') and row['payment_scope']=='actual independent-day frozen quotes','test_actual_quote_not_basis_quote',where)
                A.check(req['decision_protocol']['sha256']==sha(root/'decision_protocol.json'),'test_bound_decision_addendum',where)
                A.check(all(isinstance(v,int) and not isinstance(v,bool) and v in (0,1,2) for v in flat),'test_integer_four_price_menu',where)
            A.check(req['plan_packet']==planref and req['protocol']['sha256']==phash,'request_prelocked_plan_and_protocol',where)
            A.check(req['rho_by_event']==row['rho_by_event']==expected,'independent_four_bit_or_exact_state_replay',where)
            rec=dict(record_id=rid,phase=phase,policy=row['policy'],path=row['path'],status=score['status'])
            if row['status']!='observed':records.append(rec);continue
            receipt=read(A.bound(row['physical_receipt'],where+':physical_receipt'))
            A.check(receipt['request_sha256_expected']==sha(location/'request.json'),'frozen_actual_request_hash',where)
            A.check(stamp(receipt['started_utc'])>=stamp(c['created_utc']),'physical_after_protocol_freeze',where)
            if phase=='test':
                A.check(stamp(receipt['started_utc'])>=stamp(oracle['created_utc']),'independent_test_after_price_freeze',where)
                if addendum_time:A.check(stamp(receipt['started_utc'])>addendum_time,'canonical_repair_before_first_test_observation',where)
            for name,v in receipt['private_files'].items():A.check(sha(location/'physical'/name)==v['sha256'],'private_artifact_hash',where+':'+name)
            packet=load(A.bound(row['packet'],where+':packet'));private=load(location/'physical/physical_truth.pkl')
            A.check(set(packet)=={'training','validation','metadata'} and packet['metadata']['observed_only'] is True,'observed_only_identifier_boundary',where)
            tr=H.frames(history['training']+packet['training'],A,where+':train');va=H.frames(history['validation']+packet['validation'],A,where+':val')
            A.check([len(tr[0]),len(va[0])]==[480,480] and not len(tr[0].index.intersection(va[0].index)),'complete480_480_disjoint_data',where)
            identity=dict(training=H.frame_hash(tr),validation=H.frame_hash(va),terminal_labels=list(tr[0].columns),root_label=int(c['root_label']),search_seconds=float(c['search_seconds']),lp_method=int(c['lp_method']),algorithm_source_sha256=c['algorithm_source_sha256'])
            A.check(identity==row['observation_identity'],'independent_full_observation_identity',where)
            key=hashlib.sha256(json.dumps(identity,sort_keys=True,separators=(',',':')).encode()).hexdigest();A.check(key==row['fit_key'],'independent_cache_key',where)
            rec.update(physical(c,row,req,packet,private,receipt,plan,A,H,common));physical_count+=rec['physical_events_complete']
            if score['status']=='complete':
                A.check(score['result']==jobs[phase][key]['result'],'scoring_only_frozen_fit',where);result=read(A.bound(score['result'],where+':result'))
                A.check(result['training_rows']==result['validation_rows']==480,'full_fit_row_count',where)
                A.check(result['input_sha256']=={'training':identity['training'],'validation':identity['validation']} and result['source_sha256']==c['algorithm_source_sha256'],'actual_fit_source_input_closure',where)
                A.check(result['settings']['lp_method']==result['settings']['actual_lp_method']==2,'explicit_Method2',where)
                A.check(all(x['status']==2 and x['accepted'] for x in result['LP_solves']),'fixed_family_LP_optimal_receipts',where)
                truth={tuple(sorted(s)) for s in read(location/'physical/truth_evaluation.json')['true_supports']}
                A.check(truth==truth_static,'physical_evaluation_truth_equals_independent_static_positive_supports',where)
                if key not in fit_cache:
                    fit_cache[key]=H.selection_audit(result,tr,va,truth,A,where)
                    fit_cache[key]['start_preparation']=start_audit(result,tr,va,tuple(c['target_labels']),A,H,where)
                rec.update(fit_cache[key])
                for name in MODELS:
                    model=result['models'][name];err=H.errors(model,truth,c['labels']);rawerr=H.errors(model['selected_uncontracted'],truth,c['labels'])
                    A.check(err==score['models'][name]['full_errors'] and rawerr==score['models'][name]['before_contraction_errors'],'independent_raw_final_full_errors',where+':'+name)
                    A.check((err==0)==score['models'][name]['full_recovery'],'independent_full_recovery',where+':'+name);errors_by_model[name][rid]=err
                native=next(x for x in result['bank'] if x['config']=='raw_RX75_0.16')['native_family'];family={tuple(sorted(c['labels'][i] for i in x)) for x in native};geoerr=len(family-truth)+len(truth-family)
                A.check(geoerr==score['models']['geometric_RNJ_RX75_0.16']['full_errors'],'independent_saved_native_RNJ_full_errors',where)
                errors_by_record[rid]=errors_by_model['multistart_original_rule'][rid]
                C=tuple(c['target_labels']);has=lambda point:C in H.family(point,c['labels']);points=result['selection_candidates'];best=min(p['validation_mae'] for p in points)
                missing=[p for p in points if not has(p)];present=[p for p in points if has(p)];model=result['models']['multistart_original_rule']
                target_review.append(dict(phase=phase,record_id=rid,final_error=errors_by_record[rid],raw_has_target=has(model['selected_uncontracted']),final_has_target=has(model),
                    all_candidates=len(points),target_candidates=len(present),coarse_candidates=len(missing),coarse_eligible=sum(p['validation_mae']<=1.03*best+1e-12 for p in missing),
                    target_eligible=sum(p['validation_mae']<=1.03*best+1e-12 for p in present),
                    coarse_exclusion_margin=None if not missing else min(p['validation_mae'] for p in missing)-1.03*best-1e-12,
                    accepted_path_has_target=any(has(p) for search in result['searches'].values() for p in search['path'][1:]),
                    selected_origin=model.get('selected_origin'),selected_raw_origin=model['selected_uncontracted'].get('origin')))
                target_review[-1].update(fit_cache[key]['start_preparation'])
            actual[(phase,rid)]=rec;records.append(rec)
        phase_report[phase]=dict(requests=len(records),statuses=dict(Counter(x['status'] for x in records)),unique_jobs=len(jobs[phase]),unique_completed=sum(x['status']=='complete' for x in jobs[phase].values()),unique_failed=sum(x['status']!='complete' for x in jobs[phase].values()),records=records)
    planning=phases['planning']['observations']['rows'];observed=set()
    A.check(len(planning)==2*4*4,'two_plans_complete16_state_denominator')
    for row in planning:
        k=(row['module'],row['train_state'],row['validation_state']);A.check(k not in observed,'no_duplicate_exact_state_cell');observed.add(k)
        m=c['modules'].index(row['module']);t=row['train_state'];v=row['validation_state'];e=errors_by_record.get(row['record_id'])
        if e is not None:A.close([e,e],[loss['lower'][m][t][v],loss['upper'][m][t][v]],'precompute_uses_actual_final_error',str(k),atol=0.,rtol=0.)
        else:A.check(loss['lower'][m][t][v]==0 and loss['upper'][m][t][v]==4*len(c['labels'])-2,'unresolved_cell_full_risk_bounds',str(k))
    global_audit=check_optimizer(opt['global_result'],c,loss['lower'],loss['upper'],c['modules'],decision['seed'],A,'global_four_price')
    rotation_audit=check_optimizer(opt['rotation_result'],c,loss['lower'],loss['upper'],['rotation'],decision['seed']+1,A,'rotation_only_four_price')
    tied_audit=check_optimizer(opt['tied_result'],c,loss['lower'],loss['upper'],c['modules'],decision['seed']+2,A,'restricted_tied_prices',tied=True)
    A.check(oracle['selected']==opt['global_result']['selected'] and oracle['rotation_selected']==opt['rotation_result']['selected'] and oracle['tied_selected']==opt['tied_result']['selected'],'oracle_selected_optimizer_result_identity')
    summary=read(root/'summary.json');A.check(summary['protocol_sha256']==phash and summary['oracle']==oracle,'summary_protocol_and_frozen_quote')
    A.check(summary['planning_requests']==len(planning) and summary['test_requests']==len(phases['test']['observations']['rows']),'summary_request_denominators')
    A.check(summary['completed_unique']==sum(x['unique_completed'] for x in phase_report.values()) and summary['failed_unique']==sum(x['unique_failed'] for x in phase_report.values()),'summary_all_unique_statuses')
    test_table=[]
    for policy,q in oracle['test_policies'].items():
        rows=[r for r in phases['test']['scored']['rows'] if r['policy']==policy];saved=next(x for x in summary['test'] if x['policy']==policy);values=[errors_by_record[r['record_id']] for r in rows if r['record_id'] in errors_by_record];fee=[actual.get(('test',r['record_id']),{}).get('actual_fee') for r in rows]
        A.check(len(rows)==4 and {r['path'] for r in rows}==set(c['test_paths']),'four_independent_complete_path_denominator',policy)
        A.check(saved['denominator']==len(rows) and saved['completed']==len(values) and saved['full_error_sum']==sum(values),'test_full_error_count_and_denominator',policy)
        A.check(saved['recoveries']==sum(v==0 for v in values),'test_complete_recoveries',policy)
        if len(values)==len(rows):A.close(sum(values)/len(rows),saved['mean_full_errors'],'test_final_mean',policy)
        if all(v is not None for v in fee):A.close(sum(fee),saved['actual_fee_sum'],'test_actual_fee_once',policy)
        test_table.append(dict(policy=policy,completed=len(values),denominator=len(rows),final_error_sum=sum(values),recoveries=sum(v==0 for v in values),actual_fee_sum=sum(fee) if all(v is not None for v in fee) else None))
    A.warnings.append('Geometric NJ is retained in source score but not independently reconstructed; saved native RNJ and all three fitted outputs are independently checked.')
    A.warnings.append('AC receipt equations and saved convergence are checked without solving AC again. Finite risk is conditional on one planning natural/measurement background and synthetic behavior law.')
    report=dict(status='pass' if not A.failures else 'fail',created_utc=now(),checks=A.checked,failed_checks=len(A.failures),failures=A.failures,warnings=A.warnings,
        audit_script_sha256=sha(__file__),mathematical_helper_sha256=sha(H.__file__),protocol_sha256=phash,oracle_sha256=sha(root/'oracle_selected.json'),phases=phase_report,
        baseline=baseline_audit,
        physical_events_checked=physical_count,independent_global_finite_price=global_audit,independent_rotation_price=rotation_audit,independent_restricted_tied_price=tied_audit,test=test_table,target_mechanisms=target_review,
        experiment_calls=dict(identifier=0,LP=0,MILP=0,AC=0),no_trajectories_or_parameters_in_output=True,
        scope='exact16 behavior integration and finite domain price optimum on actual saved final errors; independent test four complete paths; no empirical behavior calibration or population optimum')
    with Path(output).open('x',encoding='utf-8') as f:json.dump(report,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True);p.add_argument('--out',required=True);p.add_argument('--helper',default=None);args=p.parse_args()
    try:
        result=run(args.root,args.out,args.helper);print(json.dumps({k:result[k] for k in ('status','checks','failed_checks','audit_script_sha256')}));raise SystemExit(0 if result['status']=='pass' else 1)
    except Exception as error:
        print(json.dumps(dict(status='blocked_or_failed',type=type(error).__name__,message=str(error),traceback=traceback.format_exc())));raise SystemExit(2)
