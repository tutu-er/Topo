"""Independent read-only audit; no repository imports, solver or AC calls.

Run only after BOTH requested phases have fits_frozen.json and scored.json.
Private pickles are trusted local hash-bound artifacts. Output contains scalar
checks/checksums only, never trajectories, fitted parameters or raw supports.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
import os
from pathlib import Path
import pickle
import random
import sys
import traceback
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_key]='1'
import numpy as np
import pandas as pd

KEYS=('P_terminal','Q_terminal','V_terminal','V_root_observed')
MODELS=('rnj_bank','baseline25','multistart_original_rule')

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def load(path):
    with Path(path).open('rb') as f:return pickle.load(f)
def stamp(value):return datetime.fromisoformat(value.replace('Z','+00:00'))
def digest(a):return hashlib.sha256(np.ascontiguousarray(a,dtype=float).tobytes()).hexdigest()
def vec(a):return np.broadcast_to(np.asarray(a,float),(96,)).copy()

def exact_declared_expected_fee(plan):
    """Preserve the intended cost tie using declared decimal input values.

    Version 1 ranked binary floating reductions (.39999999999999997 versus
    .4) before the locked tie order. This independent decimal integration
    fixes that audit error without changing any experiment artifact.
    """
    full=Decimal(0)
    for event in range(4):
        for resource in plan['resources_by_event'][event]:
            key=resource['resource_id'];price=vec(resource['incentive_per_kwh'])
            for slot in range(96):
                delta=sum((Decimal(str(float(a)))*Decimal(str(float(vec(m['profiles'][key])[slot])))
                    for a,m in zip(plan['amplitudes_by_event'][event],plan['modules_by_event'][event])
                    if key in m['profiles']),Decimal(0))
                if delta<0:full+=Decimal('.25')*Decimal(str(float(price[slot])))*(-delta)
    return Decimal('.5')*full

class Audit:
    def __init__(self):self.checked=0;self.failures=[];self.warnings=[]
    def check(self,condition,code,where='',**scalars):
        self.checked+=1
        if not bool(condition):self.failures.append(dict(code=code,where=where,**scalars))
    def close(self,a,b,code,where='',atol=1e-10,rtol=1e-7):
        aa,bb=np.asarray(a,float),np.asarray(b,float)
        ok=aa.shape==bb.shape and np.isfinite(aa).all() and np.isfinite(bb).all() and np.allclose(aa,bb,atol=atol,rtol=rtol)
        self.check(ok,code,where,maximum_absolute_difference=float(np.max(np.abs(aa-bb),initial=0.)) if aa.shape==bb.shape else None)
    def bound(self,reference,where=''):
        p=Path(reference['path']);self.check(sha(p)==reference['sha256'],'reference_hash',where);return p

def frames(scenes,A,where):
    pieces=[[] for _ in KEYS]
    for scene in scenes:
        p=scene['P_terminal']
        for key,part in zip(KEYS,pieces):
            v=scene[key].astype(float)
            A.check(v.index.equals(p.index),'exact_frame_index',where)
            if isinstance(v,pd.DataFrame):A.check(v.columns.equals(p.columns),'exact_frame_columns',where)
            part.append(v)
        target=scene['V_root_observed'].to_numpy()[:,None]**2-scene['V_terminal'].to_numpy()**2
        A.close(target,scene['drop_target'].to_numpy(),'observed_root_target',where,atol=0.,rtol=0.)
    result=tuple(pd.concat(v,axis=0) for v in pieces)
    A.check(result[0].index.is_unique and result[0].index.is_monotonic_increasing,'chronological_unique_rows',where)
    return result

def frame_hash(obs):
    h=hashlib.sha256()
    for v in obs:h.update(pd.util.hash_pandas_object(v,index=True).values.tobytes())
    h.update(json.dumps(list(obs[0].columns),default=str).encode())
    h.update(json.dumps([str(v) for v in obs[0].index.normalize()]).encode())
    return h.hexdigest()

def family(point,labels):
    if 'support_labels' in point:return {tuple(sorted(map(int,s))) for s in point['support_labels']}
    supports=point.get('support_indices',point.get('supports',[]))
    return {tuple(sorted(int(labels[i]) for i in s)) for s in supports}

def errors(point,truth,labels):
    p=family(point,labels);return len(p-truth)+len(truth-p)

def point_loss(point,obs):
    n=len(obs[0].columns);R=np.zeros((n,n));X=R.copy()
    ss=point.get('support_indices',point.get('supports',[]))
    r=point.get('r',point.get('r_values'));x=point.get('x',point.get('x_values'))
    for s,rr,xx in zip(ss,r,x,strict=True):R[np.ix_(s,s)]+=rr;X[np.ix_(s,s)]+=xx
    p,q,v,root=(z.to_numpy() for z in obs)
    return float(np.mean(np.abs(p@R.T+q@X.T-(root[:,None]**2-v**2))))

def selection_audit(result,training,validation,truth,A,where):
    labels=result['terminal_labels'];cal=result['calibrations']['raw'];allpoints=result['selection_candidates']
    bank=result['bank'];primary_n=len(result['searches']['primary']['path'])
    A.check(len(bank)==15,'complete_15_RNJ_bank',where)
    expected_configs={f'raw_{mode}_{t:g}' for mode in ('R','RX75','X') for t in (0.,.04,.08,.16,.24)}
    A.check({p['config'] for p in bank}==expected_configs,'all_declared_RNJ_configs',where)
    path_order=['primary']+sorted(k for k in result['searches'] if k!='primary')
    expected_points=bank+[p for key in path_order for p in result['searches'][key]['path']]
    A.check(len(expected_points)==len(allpoints),'all_15_bank_and_all_path_candidates_retained',where)
    for i,(expected,actual) in enumerate(zip(expected_points,allpoints)):
        A.check(family(expected,labels)==family(actual,labels),'candidate_source_family',where+':'+str(i))
        A.close(expected['r'],actual['r'],'candidate_source_R_weights',where+':'+str(i))
        A.close(expected['x'],actual['x'],'candidate_source_X_weights',where+':'+str(i))
    candidate_sets={'rnj_bank':bank,'baseline25':allpoints[:len(bank)+primary_n],
                    'multistart_original_rule':allpoints}
    out={}
    for i,p in enumerate(allpoints):
        A.close(point_loss(p,training),p['objective'],'candidate_training_loss',where+':'+str(i))
        A.close(point_loss(p,validation),p['validation_mae'],'candidate_validation_loss',where+':'+str(i))
    for name,points in candidate_sets.items():
        best=min(p['validation_mae'] for p in points)
        eligible=[i for i,p in enumerate(points) if p['validation_mae']<=1.03*best+1e-12]
        chosen=min(eligible,key=lambda i:(len(points[i]['supports']),points[i]['validation_mae'],i))
        raw=points[chosen];actual=result['models'][name];saved=actual['selected_uncontracted']
        A.check(actual['selection_index']==chosen,'three_percent_selection_index',where+':'+name)
        A.check(family(raw,labels)==family(saved,labels),'selected_raw_family',where+':'+name)
        A.close(raw['r'],saved['r'],'selected_raw_r',where+':'+name)
        A.close(raw['x'],saved['x'],'selected_raw_x',where+':'+name)
        keep=[i for i,s in enumerate(raw['supports']) if len(s)==1 or cal['a']*raw['r'][i]+cal['b']*raw['x'][i]+1e-10>=.16*cal['median_root_depth']]
        contracted={tuple(sorted(labels[j] for j in raw['supports'][i])) for i in keep}
        A.check(contracted==family(actual,labels),'fixed_point16_contraction',where+':'+name)
        A.close(point_loss(actual,training),actual['train_mae'],'final_training_loss',where+':'+name)
        A.close(point_loss(actual,validation),actual['validation_mae'],'final_validation_loss',where+':'+name)
        out[name]=dict(raw_errors=errors(raw,truth,labels),final_errors=errors(actual,truth,labels),
            full_recovery=family(actual,labels)==truth,selection_index=chosen,
            selected_origin=actual.get('selected_origin'),eligible_count=len(eligible))
    accepted=0
    for name,search in result['searches'].items():
        path=search['path'];naccept=0
        for i,d in enumerate(search['decisions']):
            best=None
            for c in d['candidates']:
                if best is None or c['objective']<best['objective']-1e-12:best=c
            gain=0. if best is None else d['current']-best['objective']
            A.close(gain,d['gain'],'actual_acceptance_gain',where+':'+name+':'+str(i))
            take=best is not None and gain>1e-10+.001*d['current']
            if take:
                naccept+=1
                A.check(naccept<len(path),'accepted_point_exists',where+':'+name)
                if naccept<len(path):A.close(path[naccept]['objective'],best['objective'],'accepted_objective',where+':'+name)
        A.check(naccept==len(path)-1,'accepted_path_count',where+':'+name)
        accepted+=naccept
    return dict(models=out,accepted_updates=accepted,search_stops=result['search_stop_statuses'])

def desired_rho(c,path):
    return [{key:int(v<.5) for key,v in zip(c['resource_ids'],np.random.default_rng(np.random.SeedSequence([c['participation_seed'],path,e])).random(2))} for e in range(4)]

def physical_audit(c,row,request,packet,private,receipt,plan,A,common):
    where=row['record_id'];labels=c['labels'];n=len(labels);total=0.;full=0.
    A.check(len(private['events'])==4,'four_private_events',where)
    scenes={s['actual_event_id']:s for split in ('training','validation') for s in packet[split]}
    for event,eventtruth in enumerate(private['events']):
        if eventtruth.get('status')!='complete':continue
        scene=scenes[event];r=receipt['event_records'][event]
        dp=np.zeros((96,n));dq=dp.copy();payment=0.;maxpay=0.
        for resource in plan['resources_by_event'][event]:
            key=resource['resource_id'];planned=np.zeros(96)
            for amplitude,module in zip(plan['amplitudes_by_event'][event],plan['modules_by_event'][event],strict=True):
                if key in module['profiles']:planned+=float(amplitude)*vec(module['profiles'][key])
            actual=request['rho_by_event'][event][key]*planned
            col=labels.index(resource['node_label']);dp[:,col]+=actual;dq[:,col]+=vec(resource['reactive_ratio'])*actual
            price=vec(resource['incentive_per_kwh']);payment+=.25*float(np.dot(price,np.maximum(-actual,0.)))
            maxpay+=.25*float(np.dot(price,np.maximum(-planned,0.)))
            baseline=vec(resource['baseline_kw']);lo=vec(resource.get('power_lower_kw',0.));hi=vec(resource['power_upper_kw'])
            elo=vec(resource['prefix_delta_lower_kwh']);ehi=vec(resource['prefix_delta_upper_kwh'])
            available=np.broadcast_to(np.asarray(resource['available'],bool),(96,))
            for tag,delta in (('planned',planned),('actual',actual)):
                energy=.25*np.cumsum(delta)
                A.check(np.all(baseline+delta>=lo-1e-9) and np.all(baseline+delta<=hi+1e-9),'declared_power_domain',where+':'+tag)
                A.check(np.all(energy>=elo-1e-9) and np.all(energy<=ehi+1e-9) and abs(energy[-1])<1e-9,'declared_energy_domain',where+':'+tag)
                A.check(np.max(np.abs(delta[~available]),initial=0.)<1e-10,'availability',where+':'+tag)
            saved=eventtruth['resource_audit']['resources'][key]
            A.close(actual,saved['executed_delta_kw'],'resource_actual_equals_plan_times_rho',where)
        p=np.asarray(eventtruth['P_pu']);q=np.asarray(eventtruth['Q_pu']);v=np.asarray(eventtruth['V_pu']);root=np.asarray(eventtruth['root_pu'])
        A.close((p-eventtruth['natural_P_pu'])*c['base_kw'],dp,'independent_increment_bridge_P',where,atol=1e-9,rtol=0.)
        A.close((q-eventtruth['natural_Q_pu'])*c['base_kw'],dq,'independent_increment_bridge_Q',where,atol=1e-9,rtol=0.)
        A.close(dp,eventtruth['delta_kw'],'saved_increment_P',where,atol=1e-12,rtol=0.)
        A.close(dq,eventtruth['delta_kvar'],'saved_increment_Q',where,atol=1e-12,rtol=0.)
        A.close(payment,r['actual_payment'],'event_actual_payment',where)
        A.close(maxpay,r['full_delivery_incentive_payment'],'event_commitment_payment',where)
        total+=payment;full+=maxpay
        A.check(r['checks']['ac_all_96_converged'] and r['checks']['electrical_equations_pass'] and r['checks']['voltage_limits_pass'],'saved_AC_checks_pass',where)
        A.check(bool(np.asarray(eventtruth['ac']['converged']).all()),'private_AC_convergence',where)
        A.check(np.min(v)>=.9-1e-10 and np.max(v)<=1.1+1e-10,'independent_voltage_range',where)
        rng=np.random.default_rng(r['profile_seed']+2000)
        expected_p=p+rng.normal(0.,.005*np.maximum(np.abs(p),1e-12),size=p.shape)
        expected_q=q+rng.normal(0.,.005*np.maximum(np.abs(q),1e-12),size=q.shape)
        expected_v=v+rng.normal(0.,.0002*np.maximum(np.abs(v),1e-12),size=v.shape)
        rootseed=70_000_019+100_003*r['replicate']+r['profile_index']
        expected_root=root+np.random.default_rng(rootseed).normal(0.,.0002*np.maximum(np.abs(root),1e-12),size=len(root))
        for key,expected in zip(KEYS,(expected_p,expected_q,expected_v,expected_root)):
            A.close(scene[key].to_numpy(),expected,'independent_standardized_noise_replay_'+key,where,atol=5e-15,rtol=1e-13)
        commonkey=(row['path'],event)
        sig=tuple(digest(eventtruth[k]) for k in ('natural_P_pu','natural_Q_pu','root_pu'))+(int(r['profile_seed']),int(rootseed))
        if commonkey in common:A.check(sig==common[commonkey],'common_background_and_standard_noise_seeds',where)
        else:common[commonkey]=sig
    if receipt['status']=='complete':
        A.close(total,receipt['actual_payment'],'four_event_actual_payment',where)
        A.close(full,receipt['worst_payment'],'four_event_commitment_payment',where)
        A.check(full<=c['hard_four_event_budget']+1e-9 and total<=full+1e-9,'four_event_hard_budget',where)
    return dict(actual_payment=total if receipt['status']=='complete' else None,full_delivery_payment=full)

def bounds(rows,error_by_record,cap):
    coefficients={'00':1,'10':-1,'01':-1,'11':1};groups={}
    for row in rows:
        key=row.get('fit_key',row['record_id']);g=groups.setdefault(key,dict(coefficient=0,values=[]));g['coefficient']+=coefficients[row['variant']]
        if row['record_id'] in error_by_record:g['values'].append(error_by_record[row['record_id']])
    low=high=0.
    for g in groups.values():
        k=g['coefficient']
        if not k:continue
        known=g['values'];lo=min(known) if known else 0.;hi=max(known) if known else cap
        low+=k*(lo if k>0 else hi);high+=k*(hi if k>0 else lo)
    known=[error_by_record[r['record_id']] for r in rows if r['record_id'] in error_by_record]
    return dict(Gamma_lower=low,Gamma_upper=high,conditional_mean_lower=sum(known)/4.,
                conditional_mean_upper=(sum(known)+(4-len(known))*cap)/4.,unresolved_states=4-len(known))

def run(root,output):
    root=Path(root).resolve();A=Audit();c=read(root/'protocol.json');protocol_hash=sha(root/'protocol.json')
    # This preflight is deliberately before ANY observed/private pickle load.
    needed=[root/p/name for p in ('planning','test') for name in ('fits_frozen.json','scored.json','observations.json')]+[root/'oracle_selected.json',root/'summary.json']
    needed += [root/p/f'worker_{worker}_ended.json' for p in ('planning','test') for worker in range(c['workers'])]
    missing=[str(p.relative_to(root)) for p in needed if not p.exists()]
    if missing:raise RuntimeError('No private data opened: missing terminal gates '+','.join(missing))
    phasefiles={p:{name:read(root/p/(name+'.json')) for name in ('fits_frozen','scored','observations')} for p in ('planning','test')}
    frozen_jobs={}
    for p,data in phasefiles.items():
        if data['fits_frozen']['status']!='frozen' or data['fits_frozen']['observation_ledger_sha256']!=sha(root/p/'observations.json'):
            raise RuntimeError('No private data opened: invalid '+p+' fit freeze')
        if data['scored']['fits_frozen_sha256']!=sha(root/p/'fits_frozen.json'):raise RuntimeError('No private data opened: unbound phase score')
        jobs={j['key']:j for j in data['fits_frozen']['jobs']}
        ended=[]
        for worker in range(c['workers']):
            ended.extend(read(root/p/f'worker_{worker}_ended.json')['rows'])
        if {j['key']:j for j in ended}!=jobs:raise RuntimeError('No private data opened: fit freeze differs from ended worker ledger')
        if set(jobs)!={j['key'] for j in data['observations']['jobs']}:raise RuntimeError('No private data opened: frozen jobs omit an observation')
        frozen_jobs[p]=jobs
    for p,h in c['source_sha256'].items():A.check(sha(p)==h,'frozen_source_hash',Path(p).name)
    history=load(A.bound(c['history_packet'],'history'))['history']
    A.check([sum(len(s['P_terminal']) for s in history[k]) for k in ('training','validation')]==[288,288],'complete_original_history')
    plans={k:load(A.bound(v,'plan:'+k)) for k,v in c['plans'].items()}
    oracle=read(root/'oracle_selected.json');A.check(oracle['protocol_sha256']==protocol_hash,'oracle_protocol')
    A.check(oracle['planning_scored_sha256']==sha(root/'planning/scored.json'),'oracle_planning_score_hash')
    A.check(stamp(oracle['created_utc'])>=stamp(phasefiles['planning']['fits_frozen']['created_utc']),'oracle_after_planning_freeze')
    actual={};all_receipts={};common={};fit_cache={};phase_summaries={};error_by_record={};errors_by_model={k:{} for k in (*MODELS,'geometric_RNJ_RX75_0.16')}
    for phase,data in phasefiles.items():
        score={r['record_id']:r for r in data['scored']['rows']};records=[]
        A.check(len(score)==len(data['observations']['rows']),'all_record_denominators_retained',phase)
        A.check(data['observations']['protocol_sha256']==protocol_hash,'phase_protocol_hash',phase)
        for row in data['observations']['rows']:
            rid=row['record_id'];where=phase+':'+rid;location=root/phase/'records'/rid
            req=read(location/'request.json');A.check(sha(location/'request.json')==row['request_sha256'],'frozen_request',where)
            expected=desired_rho(c,row['path'])
            if phase=='test' and row['variant']!='natural':expected[0]={k:int(b) for k,b in zip(c['resource_ids'],row['variant'])}
            A.check(req['rho_by_event']==expected and row['rho_by_event']==expected,'independent_eight_bit_and_CRN_replay',where)
            A.check(req['protocol']['sha256']==protocol_hash,'request_protocol_hash',where)
            receipt=read(location/'physical/events_receipt.json');all_receipts[(phase,rid)]=receipt
            A.check(stamp(receipt['started_utc'])>=stamp(c['created_utc']),'physical_after_protocol_freeze',where)
            if phase=='test':
                A.check(req['oracle']['sha256']==sha(root/'oracle_selected.json'),'test_frozen_oracle',where)
                A.check(stamp(receipt['started_utc'])>=stamp(oracle['created_utc']),'test_started_after_reference_freeze',where)
            for filename,metadata in receipt.get('private_files',{}).items():A.check(sha(location/'physical'/filename)==metadata['sha256'],'private_physical_file_hash',where+':'+filename)
            rec=dict(record_id=rid,status=score[rid]['status'],path=row['path'],variant=row['variant'],policy=row['policy'])
            if row['status']!='observed':records.append(rec);continue
            packet=load(A.bound(row['packet'],where+':packet'));private=load(location/'physical/physical_truth.pkl')
            A.check(set(packet)=={'training','validation','metadata'} and packet['metadata'].get('observed_only') is True,'observed_only_identifier_packet_boundary',where)
            A.check(packet['metadata']['root_label']==c['root_label'] and packet['metadata']['terminal_labels']==c['labels'] and packet['metadata']['base_kw']==c['base_kw'],'packet_root_labels_power_base',where)
            training=frames(history['training']+packet['training'],A,where+':training');validation=frames(history['validation']+packet['validation'],A,where+':validation')
            A.check([len(training[0]),len(validation[0])]==[480,480],'480_480_complete_inputs',where)
            A.check(not len(training[0].index.intersection(validation[0].index)),'train_validation_disjoint',where)
            identity={'training':frame_hash(training),'validation':frame_hash(validation),'terminal_labels':list(training[0].columns),
                      'root_label':int(c['root_label']),'search_seconds':float(c['search_seconds']),'lp_method':int(c['lp_method']),
                      'algorithm_source_sha256':c['algorithm_source_sha256']}
            encoded=json.dumps(identity,sort_keys=True,separators=(',',':'))
            # runner.fingerprint uses compact canonical JSON; also compare the complete identity directly.
            A.check(identity==row['observation_identity'],'independent_complete_observation_identity',where)
            rec['input_identity_sha256']=hashlib.sha256(encoded.encode()).hexdigest()
            A.check(rec['input_identity_sha256']==row['fit_key'],'independent_observation_cache_key',where)
            rec.update(physical_audit(c,row,req,packet,private,receipt,plans[row['policy']],A,common))
            if score[rid]['status']=='complete':
                A.check(score[rid]['result']==frozen_jobs[phase][row['fit_key']]['result'],'scored_result_bound_to_frozen_job',where)
                result=read(A.bound(score[rid]['result'],where+':result'))
                A.check(result['training_rows']==480 and result['validation_rows']==480,'saved_fit_full_rows',where)
                A.check(result['input_sha256']=={'training':identity['training'],'validation':identity['validation']},'saved_fit_input_hashes',where)
                A.check(result['source_sha256']==c['algorithm_source_sha256'],'saved_fit_algorithm_source_hashes',where)
                A.check(result['settings']['lp_method']==2 and result['settings']['actual_lp_method']==2,'saved_explicit_LP_method2',where)
                A.check(all(s['status']==2 and s['accepted'] for s in result['LP_solves']),'saved_completed_fixed_family_LPs_optimal',where)
                truth_eval=read(location/'physical/truth_evaluation.json');truth={tuple(sorted(s)) for s in truth_eval['true_supports']}
                if row['fit_key'] not in fit_cache:fit_cache[row['fit_key']]=selection_audit(result,training,validation,truth,A,where)
                auditfit=fit_cache[row['fit_key']];rec.update(auditfit)
                # Re-score even reused predictions against each record's own evaluation target.
                for name in MODELS:
                    e=errors(result['models'][name],truth,c['labels']);A.check(e==score[rid]['models'][name]['full_errors'],'independent_final_full_error',where+':'+name)
                    A.check((e==0)==score[rid]['models'][name]['full_recovery'],'independent_full_recovery',where+':'+name)
                    raw_e=errors(result['models'][name]['selected_uncontracted'],truth,c['labels'])
                    A.check(raw_e==score[rid]['models'][name]['before_contraction_errors'],'independent_selected_raw_full_error',where+':'+name)
                    errors_by_model[name][rid]=e
                reference=[p for p in result['bank'] if p['config']=='raw_RX75_0.16']
                A.check(len(reference)==1,'fixed_geometric_RNJ_source_exists',where)
                native={tuple(sorted(c['labels'][j] for j in s)) for s in reference[0]['native_family']}
                native_e=len(native-truth)+len(truth-native)
                A.check(native_e==score[rid]['models']['geometric_RNJ_RX75_0.16']['full_errors'],'independent_geometric_RNJ_native_full_error',where)
                errors_by_model['geometric_RNJ_RX75_0.16'][rid]=native_e
                error_by_record[rid]=errors(result['models']['multistart_original_rule'],truth,c['labels'])
            actual[(phase,rid)]=rec;records.append(rec)
        phase_summaries[phase]=dict(requests=len(records),statuses=dict(Counter(r['status'] for r in records)),
            unique_observations=len({r.get('fit_key') for r in data['observations']['rows'] if r.get('fit_key')}),records=records)
    # Independent two-path MC oracle, preserving all failed/unresolved cells.
    table=[]
    for policy in c['policies']:
        rows=[r for r in phasefiles['planning']['scored']['rows'] if r['policy']==policy]
        ok=len(rows)==2 and all(r['record_id'] in error_by_record for r in rows)
        fullpay=0.
        for event in range(4):
            for resource in plans[policy]['resources_by_event'][event]:
                key=resource['resource_id'];delta=sum((float(a)*vec(m['profiles'][key]) for a,m in zip(plans[policy]['amplitudes_by_event'][event],plans[policy]['modules_by_event'][event]) if key in m['profiles']),np.zeros(96))
                fullpay+=.25*float(np.dot(vec(resource['incentive_per_kwh']),np.maximum(-delta,0.)))
        A.close(fullpay,c['policy_details'][policy]['full_delivery_payment'],'independent_full_delivery_fee',policy)
        A.close(.5*fullpay,c['policy_details'][policy]['expected_payment'],'independent_expected_fee',policy)
        exact_fee=exact_declared_expected_fee(plans[policy])
        A.check(exact_fee==Decimal(str(c['policy_details'][policy]['expected_payment'])),'exact_decimal_expected_fee_matches_locked_cost',policy)
        table.append(dict(policy=policy,complete=ok,mean_errors=sum(error_by_record[r['record_id']] for r in rows)/2. if ok else None,
            expected_payment=float(exact_fee),expected_payment_exact=str(exact_fee)))
    rank=lambda r:(r['mean_errors'],Decimal(r['expected_payment_exact']),c['tie_order'].index(r['policy']))
    chosen=min(table,key=rank)['policy'] if all(r['complete'] for r in table) else None
    paid=[r for r in table if r['policy']!='zero'];paidchosen=min(paid,key=rank)['policy'] if all(r['complete'] for r in paid) else None
    A.check(chosen==oracle['selected_policy'],'independent_oracle_two_MC_means_and_tie_cost')
    A.check(paidchosen==oracle['paid_only_mechanism_reference'],'independent_paid_mechanism_reference')
    A.check(chosen is None or chosen in oracle['test_policies'],'formal_selected_policy_in_test')
    A.check(oracle['mechanism_is_not_replacement_for_zero_selection'] is True,'zero_selection_not_replaced')
    saved_oracle_table={r['policy']:r for r in oracle['table']}
    for r in table:
        saved=saved_oracle_table[r['policy']]
        A.check((saved['status']=='complete')==r['complete'],'oracle_table_completion_status',r['policy'])
        if r['complete']:A.close(r['mean_errors'],saved['MC_mean_final_errors'],'oracle_each_two_path_mean',r['policy'])
        A.close(r['expected_payment'],saved['theoretical_expected_payment'],'oracle_expected_payment',r['policy'])
        fees=[s['actual_payment'] for s in phasefiles['planning']['scored']['rows'] if s['policy']==r['policy'] and s['status']=='complete']
        if r['complete']:A.close(sum(fees)/2.,saved['mean_actual_payment'],'oracle_observed_mean_payment',r['policy'])
    tie=c['policies'].copy();random.Random(1031032).shuffle(tie);A.check(tie==c['tie_order'],'prelocked_tie_order')
    cap=2**len(c['labels'])-1+2*len(c['labels'])-1;panels=[];unintervened=[]
    for policy in oracle['test_policies']:
        four=[r for r in phasefiles['test']['scored']['rows'] if r['policy']==policy and r['path']==100]
        A.check(len(four)==4 and {r['variant'] for r in four}=={'00','10','01','11'},'conditional_four_state_denominator',policy)
        A.check(all(r['rho_by_event'][1:]==four[0]['rho_by_event'][1:] for r in four),'other_six_participation_bits_fixed',policy)
        panel=dict(policy=policy,**bounds(four,error_by_record,cap))
        if all(r['record_id'] in error_by_record for r in four):
            e={r['variant']:error_by_record[r['record_id']] for r in four}
            panel['conditional_Frechet_risk_endpoints']=[.5*(e['10']+e['01']),.5*(e['00']+e['11'])]
        panels.append(panel)
        natural=[r for r in phasefiles['test']['scored']['rows'] if r['policy']==policy and r['path']==101]
        A.check(len(natural)==1 and natural[0]['variant']=='natural','single_untouched_path101',policy)
        if natural:
            r=natural[0];unintervened.append(dict(policy=policy,status=r['status'],final_errors=error_by_record.get(r['record_id']),actual_payment=r.get('actual_payment')))
    summary=read(root/'summary.json')
    A.check(summary['protocol_sha256']==protocol_hash,'summary_protocol_hash')
    A.check(summary['oracle']==oracle,'summary_frozen_oracle_copy')
    expected_requests=sum(len(v['observations']['rows']) for v in phasefiles.values())
    expected_complete=sum(r['record_id'] in error_by_record for v in phasefiles.values() for r in v['observations']['rows'])
    A.check(summary['requests']==expected_requests and summary['complete_requests']==expected_complete,'summary_complete_request_denominator')
    histogram=dict(Counter(str(v) for v in error_by_record.values()))
    A.check(summary['final_error_histogram']==histogram,'summary_final_error_histogram')
    saved_panels={p['policy']:p for p in summary['conditional_Gamma_panels']}
    for policy in oracle['test_policies']:
        four=[r for r in phasefiles['test']['scored']['rows'] if r['policy']==policy and r['path']==c['test_gamma_path']]
        panel=saved_panels[policy]
        A.check(panel['denominator']==4 and panel['complete']==sum(r['status']=='complete' for r in four),'summary_gamma_panel_denominators',policy)
        for model,values in errors_by_model.items():
            independent=bounds(four,values,cap);saved=panel['models'][model]
            states={r['variant']:values.get(r['record_id']) for r in four}
            A.check(saved['state_full_errors']==states,'summary_gamma_each_state_errors',policy+':'+model)
            A.close([independent['Gamma_lower'],independent['Gamma_upper']],
                    [saved['Gamma']['lower'],saved['Gamma']['upper']],'summary_independent_Gamma_interval',policy+':'+model)
            if all(v is not None for v in states.values()):
                expected_ends=[.5*(states['10']+states['01']),.5*(states['00']+states['11'])]
                A.close(expected_ends,saved['conditional_Frechet_risk_endpoints'],'summary_conditional_joint_participation_risk_endpoints',policy+':'+model)
                A.close(sum(states.values())/4.,saved['conditional_independent_mean_errors'],'summary_conditional_four_state_mean',policy+':'+model)
    A.warnings.append('Geometric NJ reconstruction is not independently rerun; three fitted methods and saved native geometric RNJ are audited.')
    A.warnings.append('Audit v1 retained one false paid-reference mismatch from binary floating payment ties; this v2 ranks independently integrated exact decimal declared costs.')
    report=dict(status='pass' if not A.failures else 'fail',created_utc=datetime.now(timezone.utc).isoformat(),
        audit_script_sha256=sha(__file__),protocol_sha256=protocol_hash,oracle_sha256=sha(root/'oracle_selected.json'),
        checks=A.checked,failed_checks=len(A.failures),failures=A.failures,warnings=A.warnings,
        phases=phase_summaries,independent_oracle=dict(selected_policy=chosen,paid_only_reference=paidchosen,table=table),
        conditional_event0_panels=panels,untouched_path101=unintervened,
        information_units=dict(planning_backgrounds=2,conditional_test_backgrounds=1,untouched_test_backgrounds=1),
        scope=['read-only standard-library numpy pandas audit; zero AC/LP/identifier calls',
               'saved AC receipts and convergence checked; AC equations not solved again',
               'conditional Gamma applies to event0 with six other participation bits fixed',
               'two-MC planning reference is approximate; no population or price optimum claim',
               'resource shape and p=.5 are explicit synthetic declarations; no empirical EV calibration'],
        no_trajectories_or_fitted_parameters_in_output=True)
    Path(output).write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',default=str(Path(__file__).parent/'run_v1'));p.add_argument('--out',default=str(Path(__file__).parent/'audit_independent_report_v2.json'))
    args=p.parse_args()
    try:
        result=run(args.root,args.out)
        print(json.dumps({k:result[k] for k in ('status','checks','failed_checks','protocol_sha256','audit_script_sha256')}))
        raise SystemExit(0 if result['status']=='pass' else 1)
    except Exception as error:
        print(json.dumps(dict(status='blocked_or_failed',error_type=type(error).__name__,message=str(error),traceback=traceback.format_exc())))
        raise SystemExit(2)
