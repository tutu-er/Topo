"""Independent follow-up arithmetic audit: NumPy/Pandas only; no project/solver imports. Adapted from the independently reconstructed multiblock audit."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
from itertools import combinations, product
import json
from pathlib import Path
import traceback
import numpy as np
import pandas as pd

LABELS=[101,102,103,104,105]
MODELS=('rnj_bank','baseline25','multistart_original_rule')
BASE=np.array([.9,1.,1.1,1.1,1.,.9]); SHAPE=np.array([0.,-1.,1.,0.,0.,0.])
RATE=np.array([0.,.5,.75]); HD=('2021-02-03','2021-02-04','2021-02-05')
ED=('2021-02-09','2021-02-10','2021-02-11','2021-02-12')
HM=(.95,1.,1.05); EM=(.98,1.02,1.03,.97)

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,allow_nan=False).encode()).hexdigest()
def close(a,b,tol=1e-11):return bool(np.allclose(a,b,atol=tol,rtol=0))
def same(a,b):return a['supports']==b['supports'] and np.array_equal(a['r'],b['r']) and np.array_equal(a['x'],b['x'])
def canon(f):return tuple(sorted({tuple(sorted(s)) for s in f},key=lambda s:(len(s),s)))
def laminar(f):return all(not(set(a)&set(b)) or set(a)<=set(b) or set(b)<=set(a) for a,b in combinations(f,2))

def patterns(plan,law):
    """Enumerate binary matrices; common-law probability from interval intersection."""
    rates=RATE[np.asarray(plan).reshape(2,5)]
    active=np.flatnonzero(rates.ravel())
    out=[]
    for bits in product((0,1),repeat=len(active)):
        z=np.zeros(10,dtype=int);z[active]=bits;z=z.reshape(2,5)
        if law=='local':
            probability=float(np.prod(np.where(z,rates,1-rates)))
        else:
            probability=1.
            for r,zz in zip(rates,z):
                lower=max([0.]+r[zz==0].tolist())
                upper=min([1.]+r[zz==1].tolist())
                probability*=max(0.,upper-lower)
        if probability>0:out.append((probability,z))
    return out

def frames(days):
    ix=pd.DatetimeIndex(np.concatenate([pd.date_range(d['date'],periods=6,freq='h').to_numpy() for d in days]))
    obs=tuple(pd.DataFrame(np.concatenate([np.asarray(d[k]) for d in days]),index=ix,columns=LABELS) for k in ('p_pu','q_pu','v_pu'))
    return (*obs,pd.Series(np.concatenate([d['root_pu'] for d in days]),index=ix))
def ih(obs):
    h=hashlib.sha256()
    for x in obs:h.update(pd.util.hash_pandas_object(x,index=True).values.tobytes())
    h.update(json.dumps(LABELS,default=str).encode())
    h.update(json.dumps([str(d) for d in obs[0].index.normalize()]).encode())
    return h.hexdigest()
def matrices(point):
    r=np.zeros((5,5));x=r.copy()
    for s,a,b in zip(point['supports'],point['r'],point['x'],strict=True):r[np.ix_(s,s)]+=a;x[np.ix_(s,s)]+=b
    return r,x
def loss(point,obs):
    p,q,v,u=[x.to_numpy() for x in obs]
    r,x=matrices(point)
    direct=np.zeros_like(p)
    for s,a,b in zip(point['supports'],point['r'],point['x'],strict=True):direct[:,s]+=(a*p[:,s].sum(axis=1)+b*q[:,s].sum(axis=1))[:,None]
    assert close(direct,p@r.T+q@x.T,1e-13)
    return float(np.mean(abs(direct-(u[:,None]**2-v*v))))
def metric(point,truth):
    active={frozenset(LABELS[i] for i in s) for s,r,x in zip(point['supports'],point['r'],point['x']) if max(r,x)>1e-9}
    stored={frozenset(LABELS[i] for i in s) for s in point['supports']};want={frozenset(s) for s in truth}
    internal=lambda f:{s for s in f if 1<len(s)<5}
    it,ip=internal(want),internal(active);roots={s for s in it if not any(s<t for t in it)}
    key=lambda s:(len(s),tuple(sorted(s)))
    targets={'A':frozenset((101,102)),'B':frozenset((103,104)),'S':frozenset((101,102,103,104))}
    return {'internal_true_count':len(it),'internal_matched':len(it&ip),'internal_false':len(ip-it),'internal_missing':len(it-ip),
      'internal_f1':2*len(it&ip)/(len(it)+len(ip)) if it or ip else 1.,'internal_exact':it==ip,
      'root_true_count':len(roots),'root_matched':len(roots&ip),'complete_effective_family':want==active,
      'stored_support_count':len(stored),'effective_support_count':len(active),
      'zero_weight_retained_supports':[sorted(s) for s in sorted(stored-active,key=key)],
      'extra_effective_supports':[sorted(s) for s in sorted(active-want,key=key)],
      'missing_effective_supports':[sorted(s) for s in sorted(want-active,key=key)],
      'all_block_errors':len(active^want),
      'target_presence':{k:s in want for k,s in targets.items()},'target_selected':{k:s in active for k,s in targets.items()}}

def audit_history(root, ck):
    """Check historical rank/witness and every saved unbounded family point."""
    observed=read(root/'observed_training_history.json');saved=read(root/'history_audit.json')
    ck(set(observed)=={'terminal_labels','history_days'} and observed['terminal_labels']==LABELS,'history observations-only schema')
    days=observed['history_days']
    ck(all(set(d)=={'date','p_pu','q_pu','v_pu','root_pu'} for d in days),'historical day schema')
    obs=frames(days);p,q,v,r0=[x.to_numpy() for x in obs]
    y=r0[:,None]**2-v*v;base=p[:,0]
    ck(saved['status']=='complete' and not saved['truth_input_allowed'] and not saved['future_input_allowed'],'history audit scope/status')
    ck(saved['identity']['history_sha256']==sha(root/'observed_training_history.json'),'history training file hash')
    ck(saved['normalized_input_sha256']==ih(obs),'normalized history hash')
    a=saved['history_input_analysis']
    rankp=int(np.linalg.matrix_rank(p));rankpq=int(np.linalg.matrix_rank(np.column_stack((p,q))))
    ck(a['input_rank_P']==rankp and a['input_rank_PQ']==rankpq,'history input ranks')
    k=float(np.sum(p*q)/np.sum(p*p));h=np.linalg.lstsq(base[:,None],y,rcond=None)[0][0]
    row_error=float(np.max(abs(base[:,None]*h-y)))
    ck(np.max(abs(p-base[:,None]))<1e-12 and np.max(abs(q-k*p))<1e-12 and k>=0,'history synchronous nonnegative P/Q relation')
    ck(a['analytic_singleton_witness_applicable'] and a['terminal_P_identical'] and a['common_Q_over_P_pass'],'history gate applies')
    ck(close(a['common_Q_over_P'],k) and close(a['inferred_combined_row_sums'],h),'historical row-sum reconstruction')
    witness=a['singleton_witness'];R=np.diag(h/(1+k));X=R.copy()
    ck(close(witness['R'],R) and close(witness['X'],X) and np.min(np.diag(R))>0,'explicit positive singleton witness')
    witness_error=float(np.mean(abs(p@R.T+q@X.T-y)))
    ck(witness_error<1e-12 and abs(witness_error-witness['objective'])<1e-12 and row_error<1e-12,'singleton witness fits history')
    ck(witness['all_nonleaf_coefficients']==0 and a['uniform_strict_acceptance_upper_bound']==0,'zero internal witness conclusion')
    allpast=read(root/'observed_history.json');past_witness=read(root/'all_past_witness.json')
    ck(set(allpast)=={'terminal_labels','history_days'} and allpast['terminal_labels']==LABELS and len(allpast['history_days'])==3 and len(days)==2 and allpast['history_days'][:2]==days,'training/all-past historical split')
    past_obs=frames(allpast['history_days']);pp,pq,pv,pr=[x.to_numpy() for x in past_obs]
    py=pr[:,None]**2-pv*pv;pb=pp[:,0];pk=float(np.sum(pp*pq)/np.sum(pp*pp));ph=np.linalg.lstsq(pb[:,None],py,rcond=None)[0][0]
    pR=np.diag(ph/(1+pk));pX=pR.copy();past_error=float(np.mean(abs(pp@pR.T+pq@pX.T-py)))
    ck(past_witness['analytic_singleton_witness_applicable'] and np.max(abs(pp-pb[:,None]))<1e-12 and np.max(abs(pq-pk*pp))<1e-12 and past_error<1e-12,'all-past singleton witness applicable')
    ck(close(past_witness['common_Q_over_P'],pk) and close(past_witness['inferred_combined_row_sums'],ph) and close(past_witness['singleton_witness']['R'],pR) and close(past_witness['singleton_witness']['X'],pX),'all-past witness independently reconstructed')
    settings=saved['settings']
    ck(settings['continuous_upper_bound']=='unbounded' and settings['nonnegative_weights'] and not settings['fixed_early_blocks'] and settings['no_price_objective'],'historical unrestricted reweighting domain')
    cc=saved['combinatorial_check']
    ck(cc['binary_family_counts']=={'1':1,'2':1,'3':3,'4':15,'5':105} and cc['four_terminal_covered']==cc['four_terminal_laminar_families'] and cc['all_checks_pass'],'historical combinatorial coverage receipts')
    families=saved['families'];seen=set();max_loss_error=0.;max_matrix_error=0.;positive_count=0;positive_max_error=0.;positive_min_weight=float('inf')
    for f in families:
        family=canon(f['native_family']);seen.add(family)
        ck(len(family)==9 and all((i,) in family for i in range(5)) and tuple(range(5)) in family and laminar(family),'historical binary family legal')
        ck(f['family_labels']==[[LABELS[i] for i in s] for s in family],'historical family labels')
        rr=np.asarray(f['native_r']);xx=np.asarray(f['native_x'])
        ck(len(rr)==9 and len(xx)==9 and np.isfinite(rr).all() and np.isfinite(xx).all() and min(rr)>=0 and min(xx)>=0,'historical unrestricted weights finite nonnegative')
        R=np.zeros((5,5));X=R.copy()
        for s,ri,xi in zip(family,rr,xx):R[np.ix_(s,s)]+=ri;X[np.ix_(s,s)]+=xi
        native_loss=float(np.mean(abs(p@R.T+q@X.T-y)))
        point=f['point'];pr,px=matrices(point)
        max_matrix_error=max(max_matrix_error,float(np.max(abs(R-pr))),float(np.max(abs(X-px))))
        max_loss_error=max(max_loss_error,abs(native_loss-point['objective']),abs(loss(point,obs)-point['objective']))
        pw=f['positive_completion_witness']
        ck(pw is not None,'strictly positive historical completion exists')
        if pw is not None:
            positive_count+=1
            ck(canon(pw['native_family'])==family,'positive witness native family')
            beta=np.asarray(pw['combined_weights']);rr=np.asarray(pw['r']);xx=np.asarray(pw['x'])
            ck(len(beta)==9 and min(beta)>0 and min(rr)>0 and min(xx)>0 and close(rr+k*xx,beta) and close(rr,xx),'positive completion coefficients')
            internal=[s for s in family if len(s)>1]
            load=np.asarray([sum(len(s) for s in internal if i in s) for i in range(5)])
            aa=float(np.min(h/(2*load)))
            expected_beta=np.asarray([h[s[0]]-aa*load[s[0]] if len(s)==1 else aa for s in family])
            ck(close(beta,expected_beta) and close(pw['internal_combined_weight'],aa),'positive witness formula')
            pR=np.zeros((5,5));pX=pR.copy()
            for s,ri,xi in zip(family,rr,xx):pR[np.ix_(s,s)]+=ri;pX[np.ix_(s,s)]+=xi
            error=float(np.max(abs(p@pR.T+q@pX.T-y)))
            positive_max_error=max(positive_max_error,error);positive_min_weight=min(positive_min_weight,float(min(beta)))
            ck(error<1e-12 and abs(error-pw['max_residual'])<1e-12,'positive completion raw P/Q/Y reconstruction')
    ck(len(families)==len(seen)==saved['completed_families']==saved['planned_families']==105 and not saved['failures'],'all historical binary refinements retained')
    ck(max_matrix_error<1e-7 and max_loss_error<1e-10 and saved['zero_loss_families']==105,'all historical family losses/matrices reconstructed')
    ck(positive_count==saved['strictly_positive_completion_families']==105,'all 105 strict positive completion witnesses retained')
    sols=saved['LP_solves']
    ck(all(s['status']==2 and s['accepted'] and abs(s['objective']-s['physical'])<1e-8 and s['max_constraint_violation']<=1e-7 and s['max_bound_violation']<=1e-7 for s in sols),'historical LP numeric receipts')
    scan=saved.get('actual_history_search');scan_count=0
    if scan is not None:
        ck(Path(scan['source_path']).exists() and sha(scan['source_path'])==scan['source_sha256'],'history source fit hash')
        history_fit=read(scan['source_path'])
        ck(history_fit['input_sha256']['training']==ih(obs) and history_fit['terminal_labels']==LABELS,'actual history fit uses exact two-day training observations')
        ck(not scan['refit_failures'] and scan['all_candidates_audited'],'historical search sweep completion')
        for decision in scan['scans']:
            family=canon(decision['current_family']);current=decision['unbounded_current_objective']
            for c in decision['candidates']:
                scan_count+=1;block=c['candidate'];removed=[list(s) for s in family if not laminar([s,block])]
                proposed=canon([s for s in family if list(s) not in removed]+[block])
                ck(removed==c['removed'] and canon(c['jointly_refitted_family'])==proposed and c['crossing_receipt_match'],'historical crossing deletion')
                gain=current-c['unbounded_objective'];threshold=1e-10+.001*current
                ck(close(gain,c['gain']) and close(threshold,c['actual_acceptance_threshold']) and c['strictly_acceptable']==(gain>threshold),'historical acceptance arithmetic')
    return dict(status=saved['status'],history_training_rows=len(p),all_past_rows=len(pp),rank_P=rankp,rank_PQ=rankpq,singleton_witness_loss=witness_error,all_past_singleton_loss=past_error,
                family_count=len(families),LP_records_checked=len(sols),native_point_matrix_max_error=max_matrix_error,
                family_loss_max_error=max_loss_error,history_scan_candidates_checked=scan_count,
                strictly_positive_witnesses_checked=positive_count,positive_witness_max_residual=positive_max_error,positive_witness_min_combined_weight=positive_min_weight,
                claim='History leaves a positive singleton explanation; controls remain exploration, not a new WZZT price method.')


def main(args):
    root=args.run_root.resolve();repo=args.repo_root.resolve()
    protocol=read(root/'protocol.json');planning=read(root/'planning.json');policies=read(root/'policies.json');draws=read(root/'test_draws.json');summary=read(root/'summary.json')
    out={'created_at':datetime.now(timezone.utc).isoformat(),'audit_source_sha256':sha(__file__),'protocol_sha256':sha(root/'protocol.json'),
      'mode':'independent NumPy/Pandas arithmetic only; no project, optimizer or AC import/call','global_issues':[],'planning':{},'cells':[],'fits':[]}
    def ck(test,message,item=None):
        if not bool(test):(out['global_issues'] if item is None else item['issues']).append(message)
    changed=[p for p,h in protocol['source_sha256'].items() if not Path(p).is_file() or sha(p)!=h]
    ck(not changed,'frozen source changed');out['changed_sources']=changed;out['frozen_source_count']=len(protocol['source_sha256'])
    for name,h in protocol['input_sha256'].items():ck(sha(root/name)==h,'frozen input changed '+name)
    helper=repo/'research_experiments/wzzt_dr_selection/dr_pool_v2/runner.py'
    ck(any(Path(p).resolve()==helper.resolve() and sha(helper)==h for p,h in protocol['source_sha256'].items()),'evaluation helper missing or changed')
    ck(protocol['truth_to_planner'] is False and protocol['truth_to_identifier'] is False,'truth permission flag')
    ck(protocol['main_cells']==192 and protocol['control_cells']==18 and protocol['total_cells']==210,'declared denominators')
    ck(protocol['paths']==8 and protocol['response_seed']==2026102919 and protocol['planning_seeds']==[2026102911,2026102912] and protocol['max_unique_fits']==160,'frozen random/count limits')
    ck(not protocol['unknown_meter_gains_added'] and protocol['price_status'].startswith('randomized exploration baselines'),'exploration control scope')
    constants=protocol['constants']
    for k,v in {'labels':LABELS,'base_kw':100.,'q_over_p':.35,'baseline_shape':BASE.tolist(),'response_shape':SHAPE.tolist(),'response_kw':.2,'price_probability':RATE.tolist(),'history_multipliers':list(HM),'actual_multipliers':list(EM),'history_dates':list(HD),'event_dates':list(ED),'ridge':1e-6,'tie_tolerance':1e-6,'total_budget_currency':1.6,'cases_B_r':{'normal':.06,'weak':.006,'zero':0.}}.items():ck(constants[k]==v,'constant '+k)
    rng=np.random.default_rng(2026102919);u=rng.random((8,4,5));ck(draws['seed']==2026102919 and np.array_equal(u,draws['uniforms']),'test RNG exact replay')
    plans=[p for p in product((0,1,2),repeat=10) if sum(p)<=4]
    def classes(p):
        pp=np.asarray(p).reshape(2,5)
        return {'single':np.count_nonzero(np.any(pp>0,axis=0))<=1,'sync':bool(np.array_equal(pp[0],pp[1])),'full':True}
    counts={k:sum(classes(p)[k] for p in plans) for k in ('single','sync','full')};ck(counts=={'single':41,'sync':21,'full':891},'independent class counts');ck(planning['counts']==counts,'recorded class counts')
    hp=np.concatenate([np.tile((m*BASE)[:,None],(1,5)) for m in HM]);h=np.column_stack((hp,.35*hp));scales=np.sqrt(np.mean(h*h,axis=0));score_cache={}
    def score(z):
        key=tuple(np.asarray(z).ravel())
        if key not in score_cache:
            f=np.concatenate([np.tile(BASE[:,None],(1,5))+.2*SHAPE[:,None]*a[None,:] for a in np.asarray(z)])
            d=np.vstack((h,np.column_stack((f,.35*f))))/scales
            sign,val=np.linalg.slogdet(d.T@d+1e-6*np.eye(10));assert sign>0
            score_cache[key]=float(val)
        return score_cache[key]
    score_errors=[];prob_errors=[];marg_errors=[];rows=[]
    ck(len(planning['plans'])==891 and [tuple(r['plan']) for r in planning['plans']]==plans,'exhaustive feasible-plan order')
    for p,record in zip(plans,planning['plans'],strict=True):
        q=RATE[np.asarray(p).reshape(2,5)]
        rr={'plan':list(p),'classes':classes(p),'budget_bound':.4*sum(p),'expected_payment':.4*float(np.sum(np.asarray(p)*RATE[list(p)])),'mean_score':score(q)}
        for law in ('local','common'):
            pat=patterns(p,law);rr[law+'_score']=sum(w*score(z) for w,z in pat)
            prob_errors.append(abs(sum(w for w,z in pat)-1));marg_errors.append(float(np.max(abs(sum(w*z for w,z in pat)-q))))
            ck(len(pat)==record[law+'_pattern_count'],'pattern count '+str(p))
        for key in ('mean_score','local_score','common_score'):score_errors.append(abs(rr[key]-record[key]))
        ck(rr['classes']==record['classes'] and close(rr['budget_bound'],record['budget_bound']) and close(rr['expected_payment'],record['expected_payment']),'plan constraints/payment '+str(p));rows.append(rr)
    ck(max(score_errors)<1e-6,'planning D-opt reconstruction');ck(max(prob_errors)<1e-12 and max(marg_errors)<1e-12,'participation probability/marginals')
    for law,scorekey in [('local','local_score'),('common','common_score'),('mean','mean_score')]:
        for cls in counts:
            feasible=[r for r in rows if r['classes'][cls]];maximum=max(r[scorekey] for r in feasible);ties=[r for r in feasible if maximum-r[scorekey]<=1e-6];winner=min(ties,key=lambda r:(r['budget_bound'],tuple(r['plan'])));saved=planning['selected'][law][cls]
            ck(winner['plan']==saved['plan'],'planning winner '+law+'/'+cls)
            ck(abs(maximum-saved['maximum_score'])<1e-6 and abs(winner[scorekey]-saved['score'])<1e-6,'planning selected score '+law+'/'+cls)
            ck(saved['class_count']==len(feasible) and saved['tied_plans']==len(ties),'planning tie/count '+law+'/'+cls)
            for ll in ('local','common'):
                actual={tuple(z.ravel()):w for w,z in patterns(winner['plan'],ll)};st={tuple(np.asarray(v['participation']).ravel()):v['probability'] for v in saved['participation_patterns'][ll]};ck(actual==st,'selected exact law '+law+'/'+cls+'/'+ll)
    out['planning']={'plans':891,'class_counts':counts,'max_score_error':max(score_errors),'max_probability_sum_error':max(prob_errors),'max_marginal_error':max(marg_errors),'score_method':'independent direct combined Gram matrix; E logdet, no model import','all_selection_ties_checked':True}
    expected_tasks={}
    policy_audit=[]
    for seed in (2026102911,2026102912):
        for law in ('local','common'):
            rng=np.random.default_rng(np.random.SeedSequence([seed,0 if law=='local' else 1]))
            expected=[dict(name='no_reward',kind='no_reward',plan=[0]*10,tie_count=1)]
            for name,cls,key in [('single','single',law+'_score'),('sync','sync',law+'_score'),('free','full',law+'_score'),('mean_free','full','mean_score')]:
                feasible=[r for r in planning['plans'] if r['classes'][cls]]
                maximum=max(r[key] for r in feasible)
                ties=[r for r in feasible if maximum-r[key]<=1e-6]
                reservation=min(r['budget_bound'] for r in ties)
                ties=[r for r in ties if r['budget_bound']==reservation]
                selected=int(rng.integers(len(ties)))
                expected.append(dict(name=name,kind=cls,plan=ties[selected]['plan'],tie_count=len(ties),tie_index=selected,score_key=key,maximum_score=maximum,chosen_score=ties[selected][key],tie_rule='uniform among score-tied minimum-reservation plans; independent planning RNG'))
                policy_audit.append(dict(seed=seed,law=law,policy=name,tie_count=len(ties),tie_index=selected,plan=ties[selected]['plan']))
            positions=rng.choice(5,size=2,replace=False)
            price=np.zeros((2,5),dtype=int);price[0,positions[0]]=2;price[1,positions[1]]=2
            expected.append(dict(name='rotation',kind='predeclared_rotation',plan=price.ravel().tolist(),tie_count=None,role='two distinct locations defines this exploration baseline only; not a constraint on optimized policies'))
            ck(expected==policies[str(seed)][law],'independent randomized policy replay '+str(seed)+'/'+law)
            if law=='local':
                for case in ('normal','weak','zero'):
                    for p in expected:
                        if p['name'] not in ('single','sync','free'):continue
                        key=f'control__{seed}__{case}__{p["name"]}'
                        expected_tasks[key]=dict(phase='control',planning_seed=seed,law='all_participate',case=case,policy=p['name'],plan=p['plan'],path=None)
            for p in expected:
                for path in range(8):
                    key=f'{seed}__{law}__{p["name"]}__path{path}'
                    expected_tasks[key]=dict(phase='main',planning_seed=seed,law=law,case='normal',policy=p['name'],plan=p['plan'],path=path)
    ck(len(expected_tasks)==210,'task grid enumeration')
    out['randomized_policy_replay']=policy_audit
    planned_patterns=set()
    for task in expected_tasks.values():
        price=np.asarray(task['plan']).reshape(2,5);price4=np.vstack((price,price))
        if task['phase']=='control':zz=(price4>0).astype(int)
        else:
            uniform=u[task['path']];uniform=uniform if task['law']=='local' else np.tile(uniform[:,:1],(1,5))
            zz=(uniform<RATE[price4]).astype(int)
        planned_patterns.add((task['case'],tuple(zz.ravel())))
    out['planned_distinct_case_participation_patterns']=len(planned_patterns)
    ck(len(planned_patterns)<=160,'unique-fit budget insufficient for frozen complete control')
    out['history']=audit_history(root,ck)
    records=[read(p) for p in sorted((root/'records').glob('*.json'))];seen=set();unique={};input_hashes={};physical_max=0.
    # Observation cache deliberately excludes offered prices. Its embedded
    # physical_audit belongs to the first plan producing those observations;
    # later plans can have the same Z/PQV but different payment reservations.
    owners={}
    for row in records:
        if not row['cache_reused']:
            ck(row['observation_key'] not in owners,'multiple first owners for one observation cache')
            owners[row['observation_key']]=row
    contract_fields={'payment','event_payments','payment_currency','budget_bound','upper_payment_currency'}
    contract_differences=Counter();contract_difference_cells=[]
    for row in records:
        co={'key':row['key'],'issues':[]};out['cells'].append(co)
        try:
            ck(row['key'] in expected_tasks and row['key'] not in seen,'task identity',co);seen.add(row['key']);task=expected_tasks[row['key']]
            ck(all(row[k]==v for k,v in task.items()),'task frozen fields',co)
            prices=np.asarray(row['plan']).reshape(2,5);prices4=np.vstack((prices,prices))
            if row['phase']=='control':z=(prices4>0).astype(int)
            else:
                uniform=u[row['path']];uniform=uniform if row['law']=='local' else np.tile(uniform[:,:1],(1,5))
                z=(uniform<RATE[prices4]).astype(int)
            ck(np.array_equal(z,row['participation']),'realized participation sampling',co)
            inp=root/'inputs'/(row['observation_key']+'.json');pay=read(inp);data=pay['data'];input_hashes[str(inp)]=sha(inp)
            supports=[[i] for i in range(5)]+[[0,1,2,3],[0,1],[2,3]];r=np.array([.08]*5+[.10,.06,{'normal':.06,'weak':.006,'zero':0}[row['case']]]);x=.5*r
            R=np.zeros((5,5));X=R.copy()
            for s,a,b in zip(supports,r,x):R[np.ix_(s,s)]+=a;X[np.ix_(s,s)]+=b
            effective=[[LABELS[i] for i in s] for s,a,b in zip(supports,r,x) if max(a,b)>1e-9]
            truth=data['truth'];ck(truth['supports']==effective and close(truth['R'],R) and close(truth['X'],X) and close(truth['r'],r) and close(truth['x'],x),'physical truth reconstruction',co)
            days=data['training_days']+data['validation_days'];ck(len(data['training_days'])==5 and len(data['validation_days'])==2,'train/validation day counts',co)
            allp=[];allv=[];energies=[];payments=[]
            for j,d in enumerate(days):
                if j<3:p=np.tile((HM[j]*BASE)[:,None],(1,5));date=HD[j]
                else:
                    k=j-3;delta=.2*SHAPE[:,None]*z[k][None,:];p=np.tile((EM[k]*BASE)[:,None],(1,5))+delta;date=ED[k];energies.append(delta.sum(axis=0));payments.append(.2*float(np.sum(prices4[k]*z[k])))
                pu=p/100;qu=.35*pu;y=pu@R.T+qu@X.T;voltage=np.sqrt(1-y)
                ck(d['date']==date,'date '+str(j),co)
                for name,val in [('p_kw',p),('q_kvar',.35*p),('p_pu',pu),('q_pu',qu),('drop_pu2',y),('v_pu',voltage),('root_pu',np.ones(6))]:ck(close(d[name],val,2e-14),'physical '+name+'/'+str(j),co)
                physical_max=max(physical_max,float(np.max(abs(1-np.asarray(d['v_pu'])**2-y))));allp.extend(p.ravel());allv.extend(voltage.ravel())
            ph=row['physical'];cached_ph=data['physical_audit']
            ck(row['observation_key'] in owners,'missing observation-cache first owner',co)
            if row['observation_key'] in owners:
                ck(cached_ph==owners[row['observation_key']]['physical'],'cached receipt differs from original owner',co)
            changed_fields={k for k in ph if ph[k]!=cached_ph.get(k)}
            ck(changed_fields<=contract_fields,'observation-invariant physical receipt differs',co)
            if changed_fields:
                contract_differences.update(changed_fields)
                contract_difference_cells.append(row['key'])
            # Every task's payment and budget are checked from its own price
            # vector below; they must not be copied from the cached owner.
            for key,val in [('payment',sum(payments)),('payment_currency',sum(payments)),('event_payments',payments),('budget_bound',.4*prices.sum()),('upper_payment_currency',.4*prices.sum()),('budget_limit',1.6),('per_event_net_energy_kwh',energies),('power_min_kw',min(allp)),('power_max_kw',max(allp)),('voltage_min_pu',min(allv)),('voltage_max_pu',max(allv))]:ck(close(ph[key],val),'physical receipt '+key,co)
            ck(ph['valid'] and ph['budget_pass'] and ph['energy_pass'] and ph['power_pass'] and ph['voltage_screen_pass'] and not ph['AC_run'],'physical check classifications',co)
            ck(sum(payments)<=1.6+1e-12 and np.max(np.abs(energies))<1e-12 and min(allp)>=0 and max(allp)<=1.5 and min(allv)>=.9 and max(allv)<=1.1,'independent physical checks',co)
            tr=frames(data['training_days']);va=frames(data['validation_days']);hashes={'training':ih(tr),'validation':ih(va)}
            identity={**hashes,'algorithm_sources':protocol['source_sha256'],'search_seconds':25.}
            ck(pay['identity']==identity and digest(identity)==row['observation_key'],'input/cache hash',co)
            if row['status']!='evaluated':
                ck(row['status']=='failed' and bool(row.get('error')),'failure accounting',co);continue
            fitpath=Path(row['fit_result']);fit=read(fitpath);ck(sha(fitpath)==row['fit_sha256'],'fit file hash',co)
            for name in MODELS:
                metrics=metric(fit['models'][name],effective)
                for k,v in metrics.items():ck(row['evaluation']['models'][name]['final'][k]==v,'metric '+name+'/'+k,co)
                before=metric(fit['models'][name]['selected_uncontracted'],effective)
                for k,v in before.items():
                    if k in row['evaluation']['models'][name]['before_final_contraction']:ck(row['evaluation']['models'][name]['before_final_contraction'][k]==v,'precontraction metric '+name+'/'+k,co)
            if row['observation_key'] not in unique:
                fo={'key':row['observation_key'],'first_cell':row['key'],'result_sha256':sha(fitpath),'issues':[],'models':{}};unique[row['observation_key']]=fo;out['fits'].append(fo)
                ck(fit['status']=='complete' and fit['truth_input_allowed'] is False,'fit status/truth',fo)
                ck(fit['input_sha256']==hashes and fit['terminal_labels']==LABELS and fit['root_label']==1 and fit['search_seconds']==25.,'fit identity',fo)
                ck(all(p in protocol['source_sha256'] and protocol['source_sha256'][p]==h for p,h in fit['source_sha256'].items()),'fit frozen sources',fo)
                fingerprint=digest({'input_sha256':hashes,'source_sha256':fit['source_sha256'],'root_label':1,'search_seconds':25.});ck(fingerprint==fit['fingerprint'],'fit fingerprint',fo)
                ck(fit['training_rows']==30 and fit['validation_rows']==12,'fit dimensions',fo)
                ck(dict(Counter(str(v) for v in tr[0].index.normalize()))==fit['training_day_counts'] and dict(Counter(str(v) for v in va[0].index.normalize()))==fit['validation_day_counts'],'physical day grouping',fo)
                ck(not len(tr[0].index.intersection(va[0].index)),'train validation overlap',fo)
                settings=fit['settings'];ck(settings['coefficient_bound']==2 and settings['per_path_seconds']==25 and settings['relative_gain_threshold']==.001 and settings['final_validation_fraction']==.03 and settings['resolution_factor']==.16 and not settings['final_refit'],'algorithm thresholds',fo)
                sols=fit['LP_solves'];fo['LP_count']=len(sols);fo['LP_status_counts']=dict(Counter(str(s['status']) for s in sols));ck(all(s['status']==2 and s['accepted'] for s in sols),'LP numerical status',fo)
                fo['LP_max_objective_discrepancy']=max(abs(s['objective']-s['physical']) for s in sols);ck(fo['LP_max_objective_discrepancy']<1e-8,'LP recorded objective discrepancy',fo)
                ck(all(s['max_constraint_violation']<=1e-7 and s['max_bound_violation']<=1e-7 for s in sols),'LP recorded primal feasibility',fo)
                allpoints=fit['selection_candidates']+fit['start_preparation'];errors=[]
                for pt in allpoints:
                    ck(laminar(pt['supports']) and all(s in fit['pool_support_indices'] for s in pt['supports']),'point legal family',fo)
                    ck(all(np.isfinite(w).all() and min(w)>=-1e-9 and max(w)<=2+1e-9 for w in (pt['r'],pt['x'])),'point coefficient bounds',fo)
                    errors.extend((abs(loss(pt,tr)-pt['objective']),abs(loss(pt,va)-pt['validation_mae'])))
                fo['stored_points_checked']=len(allpoints);fo['point_loss_max_error']=max(errors);ck(max(errors)<1e-12,'stored point loss reconstruction',fo)
                for name,pt in fit['models'].items():
                    candidates=fit['bank'] if name=='rnj_bank' else [p for p in fit['selection_candidates'] if name=='multistart_original_rule' or p['origin'] in ('tree_bank','tree_initialized_correction')]
                    best=min(p['validation_mae'] for p in candidates);eligible=[i for i,p in enumerate(candidates) if p['validation_mae']<=1.03*best+1e-12];winner=min(eligible,key=lambda i:(len(candidates[i]['supports']),candidates[i]['validation_mae'],i));before=pt['selected_uncontracted'];ck(same(before,candidates[winner]) and pt['selection_index']==winner,'3percent selection '+name,fo)
                    cal=fit['calibrations']['raw'];keep=[i for i,s in enumerate(before['supports']) if len(s)==1 or cal['a']*before['r'][i]+cal['b']*before['x'][i]+1e-10>=.16*cal['median_root_depth']];cut={k:[before[k][i] for i in keep] for k in ('supports','r','x')};ck(same(pt,cut),'final contraction '+name,fo)
                    trainloss=loss(pt,tr);valloss=loss(pt,va);mr,mx=matrices(pt)
                    ck(close(pt['R'],mr,1e-13) and close(pt['X'],mx,1e-13) and close(pt['r_matrix'],mr,1e-13) and close(pt['x_matrix'],mx,1e-13),'final matrices '+name,fo)
                    ck(close([pt['train_mae'],pt['objective'],pt['validation_mae'],pt['selection_validation_mae']],[trainloss,trainloss,valloss,loss(before,va)],1e-12),'final losses '+name,fo)
                    ck(pt['support_labels']==[[LABELS[i] for i in s] for s in pt['supports']],'final support labels '+name,fo)
                    fo['models'][name]={'selection_index':winner,'train_mae':trainloss,'validation_mae':valloss,'precontraction_validation_mae':loss(before,va),'metrics':metric(pt,effective)}
                for name,search in fit['searches'].items():
                    path=search['path'];cross=0
                    for j,dec in enumerate(search['decisions']):
                        current=path[j];ck(abs(current['objective']-dec['current'])<1e-12 and abs(loss(current,tr)-dec['current'])<1e-12,'search current '+name,fo)
                        best=None
                        for c in dec['candidates']:
                            dropped=[list(s) for s in canon(current['supports']) if not laminar([s,c['candidate']])];ck(dropped==c['removed'],'crossing deletion '+name,fo)
                            if best is None or c['objective']<best['objective']-1e-12:best=c
                        gain=0 if best is None else dec['current']-best['objective'];ck(abs(gain-dec['gain'])<1e-12,'search candidate ordering '+name,fo)
                        accepted=best is not None and gain>1e-10+.001*dec['current'];ck(accepted==(j+1<len(path)),'acceptance rule '+name,fo)
                        if accepted:
                            ck(abs(path[j+1]['objective']-best['objective'])<1e-12,'accepted objective '+name,fo)
                            if best['removed']:cross+=1
                    ev=row['evaluation']['paths'][name];ck(ev['accepted_steps']==len(path)-1 and ev['accepted_crossing_steps']==cross and close(ev['best_internal_f1'],max(metric(p,effective)['internal_f1'] for p in path)),'reported path diagnostics '+name,fo)
                ck(fit['search_stop_statuses']=={k:s['reason'] for k,s in fit['searches'].items()},'search stop status map',fo)
                fo['search_stop_statuses']=fit['search_stop_statuses'];fo['pass_audit']=not fo['issues']
        except Exception as exc:co['issues'].append(type(exc).__name__+': '+str(exc));co['traceback']=traceback.format_exc()
        co['pass_audit']=not co['issues']
    ck(summary['recorded']==len(records) and summary['main']==sum(r['phase']=='main' for r in records) and summary['controls']==sum(r['phase']=='control' for r in records),'summary recorded denominators')
    ck(summary['evaluated']==sum(r['status']=='evaluated' for r in records) and summary['failed']==sum(r['status']!='evaluated' for r in records),'summary failure accounting')
    ck(summary['status']==('complete' if len(seen)==210 else 'partial'),'completion status')
    ck(summary['planned']==210 and summary['wall_limited']==sum(r.get('evaluation',{}).get('search_wall_limited',False) for r in records),'summary planned/wall-limit accounting')
    expected_groups=[]
    for seed in (2026102911,2026102912):
        for law in ('local','common'):
            for policy in ('no_reward','single','sync','free','mean_free','rotation'):
                rr=[r for r in records if r['phase']=='main' and r['planning_seed']==seed and r['law']==law and r['policy']==policy]
                ok=[r for r in rr if r['status']=='evaluated'];models={}
                for name in MODELS:
                    vals=[r['evaluation']['models'][name] for r in ok]
                    if vals:
                        models[name]=dict(mean_errors=float(np.mean([v['final']['all_block_errors'] for v in vals])),full_count=sum(v['final']['complete_effective_family'] for v in vals),origins=dict(Counter(v['selected_origin'] for v in vals)))
                expected_groups.append(dict(seed=seed,law=law,policy=policy,recorded=len(rr),evaluated=len(ok),denominator=8,mean_payment=float(np.mean([r['physical']['payment_currency'] for r in ok])) if ok else None,models=models))
    ck(expected_groups==summary['groups'],'summary condition/policy means and denominators')
    out.update(planned_cells=210,recorded_cells=len(records),pending_cells=210-len(seen),main_cells=sum(r['phase']=='main' for r in records),control_cells=sum(r['phase']=='control' for r in records),unique_fits=len(unique),LP_records_checked=sum(f['LP_count'] for f in out['fits']),physical_max_reconstruction_error=physical_max,input_sha256=input_hashes,failed_cells=sum(r['status']!='evaluated' for r in records),wall_limited_cells=sum(r.get('evaluation',{}).get('search_wall_limited',False) for r in records),no_solves=True)
    out['cache_contract_metadata']={'first_owners_checked':len(owners),'different_contract_cells':len(contract_difference_cells),'different_contract_fields':dict(contract_differences),'scope':'Cache identity is observation-only. Stored input receipt belongs to first owner; each row payment and bound are independently reconstructed from its own offered prices.'}
    original=root/'audit_final.json'
    if original.exists() and args.receipt.resolve()!=original.resolve():
        out['corrects_previous_audit']={'path':str(original),'sha256':sha(original),'reason':'Removed invalid equality between cached first-owner contract receipt and a different-price task with identical observations; no experiment or data modification.'}
    out['all_passed']=not out['global_issues'] and all(not i['issues'] for i in out['cells']+out['fits'])
    out['ready_for_unchanged_bounded_resume']=out['all_passed'] and out['failed_cells']==0 and not (root/'run.lock').exists()
    out['scope']='Randomized exploration control only; no new WZZT price objective. Arithmetic/input/probability/selection consistency. LP statuses and residual receipts checked; no independent optimization or global topology certificate.'
    args.receipt.parent.mkdir(parents=True,exist_ok=True);args.receipt.write_text(json.dumps(out,indent=2,ensure_ascii=False,default=lambda x:x.item())+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in out.items() if k not in ('cells','fits','input_sha256')},indent=2,ensure_ascii=False,default=lambda x:x.item()));print('issues',[(r.get('key'),r['issues']) for r in out['cells']+out['fits'] if r['issues']])
    return 0 if out['all_passed'] else 2

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-root',type=Path,required=True);p.add_argument('--repo-root',type=Path,required=True);p.add_argument('--receipt',type=Path,required=True)
    raise SystemExit(main(p.parse_args()))
