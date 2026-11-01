"""Noise-preserving history-only scenario gate for operational excitation.

This is a bounded four-existing-plan pilot of the new criterion, not a global
continuous-price optimizer. The assumed reference law is explicit; it is not a
calibrated topology posterior or independent actual-effect validation.
"""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import pickle

import numpy as np

from research_experiments.wzzt_dr_selection.dr_pool_v2.engine import PoolLP,canonical
from research_experiments.wzzt_dr_selection.dr_pool_v2.observations import normalize,arrays,matrices
from .audit_existing import read,ref,bound,write,family
from .profile_margin import validation


def perturbation(plan, event, factors, labels, base_kw):
    delta_p=np.zeros((96,len(labels)));delta_q=np.zeros_like(delta_p)
    modules=plan['modules_by_event'][event]
    x=np.asarray(plan['expected_amplitudes_by_event'][event],float)
    prices=np.asarray(plan['prices_by_event'][event],float)
    d=np.asarray(plan['response_factors_by_event'][event],float).reshape(len(x),2)
    actual=x*(1+d@factors)
    resources={r['resource_id']:r for r in plan['resources_by_event'][event]}
    fee=0.
    for j,m in enumerate(modules):
        key,shape=next(iter(m['profiles'].items()));shape=np.asarray(shape,float)
        node=labels.index(resources[key]['node_label'])
        delta_p[:,node]+=shape*actual[j]/base_kw
        delta_q[:,node]+=resources[key]['reactive_ratio']*shape*actual[j]/base_kw
        fee+=.25*np.maximum(-shape,0.).sum()*prices[j]*actual[j]
    return delta_p,delta_q,float(fee)


def run(source,output,samples=10):
    source,output=Path(source),Path(output);output.mkdir(parents=True,exist_ok=False)
    c=read(source/'main/protocol.json')
    protocol=dict(status='frozen',created_utc=datetime.now(timezone.utc).isoformat(),
        history=c['history_packet'],historical_fit=c['historical_baseline_result'],plans=c['plans'],
        policies=c['policies'],samples=samples,seed=2026110121,budget=1.6,
        scenario_law='independent historical P/Q day and centered residual-day indices, equiprobable full-day residual sign, shared continuous response factors',
        prediction_error='full-day combined measurement/model residual relative to historical refined fit; not pure meter noise',
        scope='conditional reference-model planning gate, not calibrated field probability or global continuous price optimum',
        source_sha256={str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')},
        new_actual_future=0,new_AC=0,new_RNJ_W_search=0)
    write(output/'protocol.json',protocol)
    with bound(c['history_packet']).open('rb') as f:h=pickle.load(f)['history']
    old=read(bound(c['historical_baseline_result']));comp=c['competition']
    p0=old['selection_candidates'][comp['selected_index']]
    p1=old['selection_candidates'][comp['alternative_index']]
    coarse,fine=family(p0),family(p1);pool=canonical([*coarse,*fine])
    rh,xh=matrices(len(c['labels']),p1['supports'],p1['r'],p1['x'])
    days=[]
    for scene in h['training']+h['validation']:
        obs,_,_=normalize([scene],'history_day');days.append(arrays(obs))
    residual=np.stack([day[2]-day[0]@rh.T-day[1]@xh.T for day in days])
    bias=residual.mean(axis=0);centered=residual-bias
    obs,_,_=normalize(h['training'],'historical_train');past_train=arrays(obs)
    obs,_,_=normalize(h['validation'],'historical_valid');past_valid=arrays(obs)
    plans={k:read(bound(v)) for k,v in c['plans'].items()}
    rng=np.random.default_rng(protocol['seed']);rows=[];lp_count=0
    for scenario in range(samples):
        day_ids=rng.integers(0,6,4).tolist();error_ids=rng.integers(0,6,4).tolist()
        signs=rng.choice([-1.,1.],4);z=rng.uniform(-1,1,(4,2))
        for policy in c['policies']:
            row=dict(scenario=scenario,policy=policy,status='started')
            try:
                future=[];fee=0.
                for e in range(4):
                    dp,dq,cost=perturbation(plans[policy],e,z[e],c['labels'],c['base_kw']);fee+=cost
                    pp=days[day_ids[e]][0]+dp;qq=days[day_ids[e]][1]+dq
                    yy=pp@rh.T+qq@xh.T+bias+signs[e]*centered[error_ids[e]]
                    future.append((pp,qq,yy))
                if fee>1.6+1e-8:raise AssertionError('hard budget violated')
                train=tuple(np.vstack((past_train[k],future[0][k],future[1][k])) for k in range(3))
                valid=tuple(np.vstack((past_valid[k],future[2][k],future[3][k])) for k in range(3))
                lp=PoolLP(None,pool,data=train,lp_method=2)
                try:
                    f0=lp.fit(coarse);f1=lp.fit(fine)
                    v0,v1=validation(f0,valid),validation(f1,valid)
                    margin=v0-1.03*v1-1e-12
                    gain=f0['objective']-f1['objective']-1e-10-.001*f0['objective']
                    row.update(status='complete',validation_exit_margin=margin,
                        training_acceptance_margin=gain,operational_excitation_pass=bool(margin>0 and gain>0),
                        actual_model_fee=fee,LP_calls=len(lp.solves))
                    lp_count+=len(lp.solves)
                finally:lp.close()
            except Exception as e:
                import traceback
                row.update(status='failed',error_type=type(e).__name__,message=str(e),traceback=traceback.format_exc())
            rows.append(row)
        write(output/'progress.json',dict(status='running',rows=rows,planned=samples*4))
        print(json.dumps(dict(scenario=scenario,done=len(rows),planned=samples*4)),flush=True)
    table=[]
    for policy in c['policies']:
        rr=[r for r in rows if r['policy']==policy];ok=[r for r in rr if r['status']=='complete']
        vals=sorted(r['validation_exit_margin'] for r in ok)
        passed=sum(r['operational_excitation_pass'] for r in ok)
        table.append(dict(policy=policy,completed=len(ok),denominator=samples,
            operational_passes=passed,empirical_rate=passed/samples,
            sample_90pct_requirement_met=bool(len(ok)==samples and passed>=int(np.ceil(.9*samples))),
            minimum_margin=min(vals) if vals else None,maximum_margin=max(vals) if vals else None,
            mean_model_fee=sum(r['actual_model_fee'] for r in ok)/samples if len(ok)==samples else None))
    summary=dict(status='complete',rows=rows,table=table,samples=samples,scenario_policy_records=samples*4,
        training_LP_calls=lp_count,selected_plan=None,
        selection_status='no_existing_plan_meets_empirical90' if not any(t['sample_90pct_requirement_met'] for t in table) else 'candidate_needs_independent_check',
        probability_scope='empirical model-scenario excitation rate, not topology correctness or field confidence',
        new_actual_future=0,new_AC=0,new_RNJ_W_search=0,continuous_global_optimality_claim=False)
    write(output/'summary.json',summary)
    print(json.dumps(dict(status='complete',LP=lp_count,table=table)),flush=True)
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.source,a.output)


if __name__=='__main__':main()
