"""Source-bound decision layer: reuse complete impacts for independent day prices.

The base physical/identifier protocol is immutable. This addendum changes only
price degrees of freedom before the impact table is inspected, and binds this
new decision/execution source independently. No topology identifier is changed.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
from itertools import product
import json
from pathlib import Path
import pickle
import subprocess
import sys
import traceback
import zipfile
from datetime import datetime,timezone
REPO=Path(__file__).resolve().parents[3]
TASK=REPO/'research_experiments/wzzt_dr_selection'
sys.path.insert(0,str(REPO))
from research_experiments.wzzt_dr_selection.dr_targeted_pricing.run_precomputed import (read,write,sha,ref,bound,configuration,setup)

def config(root):
    root,c=configuration(root);d=read(root/'decision_protocol.json')
    bound(d['base_protocol']);bound(d['decision_lock'])
    for p,h in d['source_sha256'].items():
        if sha(p)!=h:raise RuntimeError('decision source drift '+p)
    for plan in d['plans'].values():bound(plan)
    return root,c,d

def prepare(root):
    root,c=configuration(root);decision=read(root/'event_price_decision_lock.json')
    if decision['base_protocol_sha256']!=sha(root/'protocol.json'):raise RuntimeError('decision lock mismatch')
    if not decision['planning_scored_has_not_been_created'] or (root/'planning/scored.json').exists():raise RuntimeError('extension source must freeze before impact scoring')
    if (root/'decision_protocol.json').exists():raise FileExistsError('decision extension already frozen')
    if (root/'test').exists():raise RuntimeError('test was already opened')
    quotes=[list(v) for v in product(range(3),repeat=4) if sum(v)<=4]
    assert len(quotes)==50
    plans={};lookup={}
    for m in c['modules']:
        with bound(c['plans'][c['plan_lookup'][m]['1,1']]).open('rb') as f:template=pickle.load(f)
        lookup[m]={}
        for flat in quotes:
            rows=[flat[:2],flat[2:]];dto=deepcopy(template)
            for event in range(4):
                prices=rows[0] if event==0 else rows[1] if event==2 else [0,0]
                for i,r in enumerate(dto['resources_by_event'][event]):r['incentive_per_kwh']=prices[i]
            key=m+'__p'+'_'.join(str(v) for v in flat)
            file=root/'event_plans'/f'{key}.pkl';file.parent.mkdir(exist_ok=True);file.write_bytes(pickle.dumps(dto,protocol=4))
            plans[key]=ref(file);lookup[m][','.join(str(v) for v in flat)]=key
    package=Path(__file__).parent
    files=[Path(__file__),package/'event_optimizer.py',package/'optimizer.py',package/'run_precomputed.py']
    sources={str(p.resolve()):sha(p) for p in files}
    d=dict(status='frozen',created_utc=datetime.now(timezone.utc).isoformat(),id='independent_event_price_extension_v1',
        base_protocol=ref(root/'protocol.json'),decision_lock=ref(root/'event_price_decision_lock.json'),source_sha256=sources,
        plans=plans,lookup=lookup,price_vectors=quotes,price_rows=['paid_training_event0','paid_validation_event2'],
        probabilities=[0.,.5,.75],hard_budget=1.6,energy_per_event=.4,seed=10311135,test_paths=c['test_paths'],
        same_tasks_unchanged_16_state_coverage=True,precomputation_new_W_calls=0)
    write(root/'decision_protocol.json',d)
    with zipfile.ZipFile(root/'decision_source_snapshot.zip','w',zipfile.ZIP_DEFLATED) as z:
        for i,p in enumerate(sorted(sources)):z.write(p,arcname=f'{i:02d}_'+Path(p).name)
    print(json.dumps(dict(status='decision_protocol_frozen',feasible_quotes_per_plan=50,plans=len(plans),no_impact_losses_read=True)),flush=True)

def optimize(root):
    root,c,d=config(root);import numpy as np
    from research_experiments.wzzt_dr_selection.dr_targeted_pricing.event_optimizer import optimize_event_prices
    from research_experiments.wzzt_dr_selection.dr_targeted_pricing.optimizer import optimize_price_plan,explicit_loss_intervals
    if (root/'oracle_selected.json').exists():raise FileExistsError('price policy already frozen')
    scored=read(root/'planning/scored.json');raw=np.full((len(c['modules']),4,4),np.nan)
    for row in scored['rows']:
        receipt=read(bound(row['physical_receipt']))
        if receipt.get('all_operation_checks_pass') is not True or receipt.get('budget_pass') is not True:
            raise RuntimeError('unknown/failed hard physical constraints are not a bounded topology loss')
        if row['status']=='complete':raw[c['modules'].index(row['module']),row['train_state'],row['validation_state']]=row['models']['multistart_original_rule']['full_errors']
    interval=explicit_loss_intervals(raw,missing_lower=0.,missing_upper=4*len(c['labels'])-2)
    lo,hi=interval['lower'],interval['upper']
    result=optimize_event_prices(lo,hi,module_ids=c['modules'],seed=d['seed'])
    ri=c['modules'].index('rotation');rotation=optimize_event_prices(lo[ri:ri+1],hi[ri:ri+1],module_ids=['rotation'],seed=d['seed']+1)
    tied=optimize_price_plan(lo,hi,module_ids=c['modules'],seed=d['seed']+2)
    write(root/'planning/loss_table.json',dict(modules=c['modules'],states=c['pair_states'],lower=lo.tolist(),upper=hi.tolist(),missing_cells=interval['missing_cells'],natural_backgrounds=1))
    write(root/'event_price_optimization.json',dict(global_result=result,rotation_result=rotation,tied_result=tied))
    if result['selected'] is None or rotation['selected'] is None or tied['selected'] is None:raise RuntimeError('price selection unverified: missing state intervals retained')
    def policy(s):
        p=np.asarray(s['prices'])
        if p.shape!=(2,2) or not np.isin(p,[0,1,2]).all() or not np.equal(p,np.floor(p)).all():raise ValueError('event quote schema')
        return dict(module_id=s['module_id'],prices=p.astype(int).tolist())
    tp=[int(v) for v in tied['selected']['prices']]
    policies=dict(zero=dict(module_id='targeted',prices=[[0,0],[0,0]]),
        rotation_equal=dict(module_id='rotation',prices=[[1,1],[1,1]]),rotation_optimized=policy(rotation['selected']),
        optimized=policy(result['selected']),tied_optimized=dict(module_id=tied['selected']['module_id'],prices=[tp,tp]))
    oracle=dict(status='frozen',created_utc=datetime.now(timezone.utc).isoformat(),protocol_sha256=sha(root/'protocol.json'),
        decision_protocol_sha256=sha(root/'decision_protocol.json'),planning_scored_sha256=sha(root/'planning/scored.json'),
        loss_table_sha256=sha(root/'planning/loss_table.json'),price_optimization_sha256=sha(root/'event_price_optimization.json'),
        test_policies=policies,selected=result['selected'],rotation_selected=rotation['selected'],tied_selected=tied['selected'],
        no_test_data_yet=not (root/'test').exists(),scope='known-model exact conditional16-state pricing; one planning natural background; no global stochastic optimality claim')
    write(root/'oracle_selected.json',oracle)
    print(json.dumps(dict(status='event_prices_frozen',selected=result['selected'],rotation_selected=rotation['selected'],tied_selected=tied['selected'],test_policies=policies)),flush=True)

def observe_test(root):
    root,c,d=config(root);import numpy as np
    from research_experiments.wzzt_dr_selection.dr_pool_v2.matrix_runner import observation_identity
    oracle=read(root/'oracle_selected.json')
    if oracle['protocol_sha256']!=sha(root/'protocol.json') or oracle['decision_protocol_sha256']!=sha(root/'decision_protocol.json'):raise RuntimeError('oracle protocol mismatch')
    if oracle['price_optimization_sha256']!=sha(root/'event_price_optimization.json'):raise RuntimeError('optimization drift')
    folder=root/'test';folder.mkdir(exist_ok=True)
    if (folder/'observations.json').exists():raise FileExistsError('test observations terminal; no duplicate generation')
    lock=folder/'observations.lock'
    with lock.open('x') as f:f.write(str(__import__('os').getpid()))
    with bound(c['history_packet']).open('rb') as f:history=pickle.load(f)['history']
    rows=[];jobs={};env=__import__('os').environ.copy();env['PYTHONPATH']=str(REPO)
    try:
        for path in d['test_paths']:
            for name,pol in oracle['test_policies'].items():
                m=pol['module_id'];prices=np.asarray(pol['prices'],int);rho=[]
                assert prices.shape==(2,2) and np.isin(prices,[0,1,2]).all()
                for event in range(4):
                    u=np.random.default_rng(np.random.SeedSequence([c['participation_seed'],path,event])).random(2)
                    if event in (0,2):p=np.asarray(d['probabilities'])[prices[event//2]];bits=(u<p).astype(int)
                    else:bits=np.zeros(2,int)
                    rho.append({key:int(v) for key,v in zip(c['resource_ids'],bits)})
                t=c['pair_states'].index([rho[0][key] for key in c['resource_ids']]);v=c['pair_states'].index([rho[2][key] for key in c['resource_ids']])
                rid=f'{name}__p{path}__t{t}v{v}';loc=folder/'records'/rid;loc.mkdir(parents=True,exist_ok=True)
                flat=prices.reshape(-1).tolist();plan=d['plans'][d['lookup'][m][','.join(str(q) for q in flat)]]
                request=dict(status='frozen',case=c['case'],repo_root=str(REPO),source_root=c['source_root'],source_metadata=c['source_metadata'],
                    labels=c['labels'],root_label=c['root_label'],base_kw=c['base_kw'],voltage_sigma=c['voltage_sigma'],phase='test',record_id=rid,path=path,
                    replicate_base=c['replicate_base'],event_profiles=c['event_profiles'],plan_packet=plan,rho_by_event=rho,response_mode='independent_resource_day_MC',
                    protocol=ref(root/'protocol.json'),source_sha256=c['source_sha256'],incentive_budget=1.6,oracle=ref(root/'oracle_selected.json'),decision_protocol=ref(root/'decision_protocol.json'))
                req=loc/'request.json';write(req,request)
                row=dict(record_id=rid,phase='test',policy=name,module=m,path=path,variant='natural',prices=prices.tolist(),train_state=t,validation_state=v,rho_by_event=rho,status='started',physical_output=str(loc/'physical'))
                try:
                    done=subprocess.run([sys.executable,'-B',str(TASK/'dr_day_modules/day_physical_worker.py'),'--request',str(req),'--expected-sha',sha(req),'--output',str(loc/'physical')],cwd=REPO,env=env,capture_output=True,text=True,timeout=600)
                    (loc/'physical_stdout.txt').write_text(done.stdout,encoding='utf-8');(loc/'physical_stderr.txt').write_text(done.stderr,encoding='utf-8')
                    receipt=read(loc/'physical/events_receipt.json');row.update(physical_status=receipt['status'],physical_receipt=ref(loc/'physical/events_receipt.json'))
                    if receipt['status']!='complete':raise RuntimeError('physical status '+receipt['status']+': '+str(receipt.get('message')))
                    with (loc/'physical/packet.pkl').open('rb') as f:packet=pickle.load(f)
                    train=history['training']+packet['training'];valid=history['validation']+packet['validation']
                    assert [sum(len(s['P_terminal']) for s in seq) for seq in (train,valid)]==[480,480]
                    key,identity=observation_identity(train,valid,root_label=c['root_label'],search_seconds=c['search_seconds'],sources=c['algorithm_source_sha256'],lp_method=c['lp_method'])
                    row.update(status='observed',fit_key=key,observation_identity=identity,packet=ref(loc/'physical/packet.pkl'),actual_payment=receipt['actual_payment'],worst_payment=receipt['worst_payment'],operation_checks=receipt['all_operation_checks_pass'],payment_scope='actual independent-day frozen quotes')
                    if key not in jobs:jobs[key]=dict(key=key,packet=row['packet'],record_ids=[],first_record_id=rid)
                    jobs[key]['record_ids'].append(rid)
                except Exception as error:row.update(status='physical_or_observation_failed',exception_type=type(error).__name__,message=str(error),traceback=traceback.format_exc())
                rows.append(row);write(folder/'observation_progress.json',dict(status='running',rows=rows,planned=len(d['test_paths'])*len(oracle['test_policies']),unique_jobs=len(jobs)))
                print(json.dumps(dict(stage='physical',phase='test',done=len(rows),id=rid,status=row['status'],unique=len(jobs))),flush=True)
        ordered=[jobs[k] for k in sorted(jobs)]
        for i,j in enumerate(ordered):j['worker']=i%c['workers']
        write(folder/'observations.json',dict(status='frozen',protocol_sha256=sha(root/'protocol.json'),decision_protocol_sha256=sha(root/'decision_protocol.json'),rows=rows,jobs=ordered,planned=len(rows),created_utc=datetime.now(timezone.utc).isoformat()))
    finally:lock.unlink(missing_ok=True)
    print(json.dumps(dict(status='test_observations_frozen',requests=len(rows),unique=len(jobs))),flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=('prepare','optimize','test'));p.add_argument('--root',required=True);a=p.parse_args()
    if a.command=='prepare':prepare(a.root)
    elif a.command=='optimize':optimize(a.root)
    else:observe_test(a.root)
if __name__=='__main__':main()
