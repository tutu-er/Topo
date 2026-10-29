"""Bounded stochastic multi-block mechanism test using the frozen real adapter.

The planner uses only historical/forecast P/Q and declared participation laws.
The simulator owns topology. Truth is used only after the identifier returns.
"""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import traceback
import numpy as np
import pandas as pd
from . import model
from research_experiments.wzzt_dr_selection.dr_pool_v2.adapter import fit_observed, source_hashes
from research_experiments.wzzt_dr_selection.dr_pool_v2.observations import normalize, input_hash
from research_experiments.wzzt_dr_selection.dr_pool_v2.runner import evaluate_fit
from research_experiments.wzzt_dr_selection.end_to_end.wzzt_adapter import make_observed_scenario

SEED=2026102907
PATHS=8
METHODS=('rnj_bank','baseline25','multistart_original_rule')

def plain(x):
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    if isinstance(x,Path):return str(x)
    if isinstance(x,dict):return {str(k):plain(v) for k,v in x.items()}
    if isinstance(x,(tuple,list)):return [plain(v) for v in x]
    return x

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def digest(x):return hashlib.sha256(json.dumps(plain(x),sort_keys=True,allow_nan=False).encode()).hexdigest()
def write(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    temp=p.with_suffix(p.suffix+'.partial')
    temp.write_text(json.dumps(plain(x),ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    temp.replace(p)
def now():return datetime.now(timezone.utc).isoformat()

def code_hashes():
    files=source_hashes()
    files.update({str(p.resolve()):sha(p) for p in (Path(__file__).parent/n for n in ('__init__.py','model.py','experiment.py'))})
    return files

def selected_plan(planning,law,kind):return tuple(planning['selected'][law][kind]['plan'])

def policy_list(planning,law):
    singles=[p for p in model.enumerate_plans() if model.classify(p)['single']]
    out=[{'name':f'single_{i:02d}','class':'single','plan':p} for i,p in enumerate(singles)]
    for name,key in (('stochastic_sync','sync'),('stochastic_free','full')):
        out.append({'name':name,'class':key,'plan':selected_plan(planning,law,key)})
    out.append({'name':'mean_free','class':'full','plan':selected_plan(planning,'mean','full')})
    return out

def prepare(root):
    root=Path(root)
    if (root/'protocol.json').exists():
        protocol=read(root/'protocol.json')
        if protocol['source_sha256']!=code_hashes():raise RuntimeError('Frozen source changed; refuse resume')
        for name,value in protocol['frozen_input_sha256'].items():
            if sha(root/name)!=value:raise RuntimeError('Frozen input changed: '+name)
        return protocol,read(root/'planning.json')
    root.mkdir(parents=True,exist_ok=True)
    planning=model.plan_all()
    write(root/'planning.json',planning)
    # Generated only after prices are chosen; separate event rows make validation independent.
    rng=np.random.default_rng(SEED)
    draws=rng.random((PATHS,4,5))
    write(root/'test_draws.json',{'seed':SEED,'uniforms':draws,'scope':'independent complete-event latent draws; shared between policies; not planner input'})
    policies={law:policy_list(planning,law) for law in ('local','common')}
    assert all(len(x)==44 for x in policies.values())
    write(root/'policies.json',policies)
    protocol=dict(version=1,created_at=now(),constants=model.constants(),source_sha256=code_hashes(),
        frozen_input_sha256={name:sha(root/name) for name in ('planning.json','test_draws.json','policies.json')},
        random_seed=SEED,test_paths=PATHS,main_cells=704,deterministic_control_cells=9,
        total_cells=713,max_unique_fits=160,
        algorithm='frozen finite-pool RNJ-WZZT multistart_original_rule; not A2 reference or free-support MILP',
        search_seconds_per_start=25.,
        planning='exact E[ordinary historical-RMS D-opt] over complete-event participation; no identifier in planning',
        single_comparison='41 feasible single-location price schedules; deployed score optimum plus posthoc best fixed schedule by 8-path mean; never per-path oracle',
        controls='normal, weak and zero B; each uses the three frozen local-law plans, all offered users participate; separate from random main cells',
        limitations=['Synthetic price participation law, not iFlex calibration','Linear noiseless voltage observations, not AC or field validation','Q=.35P retains R/X confounding','Eight independent DR paths conditional on fixed natural loads and zero measurement error; not independent whole data sets','Tie representative frozen before truth/results; not all multi-location optimal-score ties tested'],
        truth_to_planner=False,truth_to_identifier=False,
        evaluation=['all effective block false+missing count','complete effective family','individual A/B/S recovery','internal F1','actual payment','RNJ vs primary vs multistart and accepted updates'],
        environment=dict(python=sys.version,executable=sys.executable,platform=platform.platform(),numpy=np.__version__,pandas=pd.__version__))
    write(root/'protocol.json',protocol)
    return protocol,planning

def observations(data):
    def pack(day):
        idx=pd.date_range(day['date'],periods=len(day['p_pu']),freq='h')
        labels=[101,102,103,104,105]
        frames=[pd.DataFrame(day[k],index=idx,columns=labels) for k in ('p_pu','q_pu','v_pu')]
        root=pd.Series(day['root_pu'],index=idx)
        return make_observed_scenario(*frames,root,str(day['date']))
    return [pack(d) for d in data['training_days']],[pack(d) for d in data['validation_days']]

def keys(training,validation,protocol):
    tr,days,_=normalize(training,'training');va,vdays,_=normalize(validation,'validation')
    identity=dict(training=input_hash(tr,days),validation=input_hash(va,vdays),
                  algorithm_sources=protocol['source_sha256'],search_seconds=25.)
    return digest(identity),identity

def fit_metrics(fit,truth):
    ev=evaluate_fit(fit,truth['supports'],fit['terminal_labels'])
    for name,info in ev['models'].items():
        final=info['final']
        final['all_block_errors']=len(final['extra_effective_supports'])+len(final['missing_effective_supports'])
        point=fit['models'][name]
        active={tuple(sorted(s)) for s,r,x in zip(point['support_labels'],point['r'],point['x']) if max(r,x)>1e-9}
        final['target_presence']={label:list(s) in truth['supports'] for label,s in (('A',(101,102)),('B',(103,104)),('S',(101,102,103,104)))}
        final['target_selected']={label:s in active for label,s in (('A',(101,102)),('B',(103,104)),('S',(101,102,103,104)))}
        info['path_index']=point.get('path_index')
    ev['paths']={}
    for name,path in fit['searches'].items():
        crossing=0
        for j,decision in enumerate(path['decisions']):
            if j+1>=len(path['path']):continue
            winner=None
            for candidate in decision['candidates']:
                if winner is None or candidate['objective']<winner['objective']-1e-12:winner=candidate
            if winner and winner['removed']:crossing+=1
        ev['paths'][name]={'reason':path['reason'],'accepted_steps':len(path['path'])-1,
             'accepted_crossing_steps':crossing,
             'best_internal_f1':max(evaluate_fit_point(x,truth,fit['terminal_labels'])['internal_f1'] for x in path['path'])}
    return ev

def evaluate_fit_point(point,truth,labels):
    from research_experiments.wzzt_dr_selection.dr_pool_v2.runner import support_metrics
    return support_metrics(point,truth['supports'],labels)

def tasks(planning):
    # Controls are diagnostic and do not change planning, model constants or the main batch.
    for case in ('normal','weak','zero'):
        for kind in ('single','sync','full'):
            yield dict(key=f'control__{case}__{kind}',phase='control',law='all_participate',case=case,
                policy=kind,path=None,plan=selected_plan(planning,'local',kind))
    for law in ('local','common'):
        for policy in policy_list(planning,law):
            for path in range(PATHS):
                yield dict(key=f'{law}__{policy["name"]}__path{path}',phase='main',law=law,case='normal',
                    policy=policy['name'],policy_class=policy['class'],path=path,plan=policy['plan'])

def response(plan,uniform,law):
    prices=np.asarray(plan).reshape(2,5);rates=np.asarray([0.,.5,.75])[prices]
    rates=np.vstack((rates,rates))
    u=uniform if law=='local' else np.repeat(uniform[:,:1],5,axis=1)
    return (u<rates).astype(int)

def summary(root,protocol,planning):
    rows=[read(p) for p in sorted((root/'records').glob('*.json'))]
    groups=[]
    for law in ('local','common'):
        for pol in policy_list(planning,law):
            rr=[r for r in rows if r['phase']=='main' and r['law']==law and r['policy']==pol['name']]
            ok=[r for r in rr if r['status']=='evaluated']
            group=dict(law=law,policy=pol['name'],plan=pol['plan'],recorded=len(rr),evaluated=len(ok),planned=PATHS,
                       complete=len(ok)==PATHS,models={})
            if len(ok)==PATHS:
                group['mean_payment']=float(np.mean([r['physical']['payment_currency'] for r in ok]))
                for m in METHODS:
                    vals=[r['evaluation']['models'][m]['final'] for r in ok]
                    group['models'][m]=dict(mean_errors=float(np.mean([v['all_block_errors'] for v in vals])),
                        full_count=sum(v['complete_effective_family'] for v in vals),internal_exact_count=sum(v['internal_exact'] for v in vals),
                        mean_f1=float(np.mean([v['internal_f1'] for v in vals])),
                        target_counts={k:sum(v['target_selected'][k] for v in vals) for k in ('A','B','S')})
            groups.append(group)
    prior=[p for p in (root/'fits').glob('*/result.json')]
    result=dict(status='complete' if len(rows)==protocol['total_cells'] else 'partial',
        planned_cells=protocol['total_cells'],recorded_cells=len(rows),main_recorded=sum(r['phase']=='main' for r in rows),
        controls_recorded=sum(r['phase']=='control' for r in rows),evaluated=sum(r['status']=='evaluated' for r in rows),
        failures=sum(r['status']!='evaluated' for r in rows),
        wall_limited=sum(r.get('evaluation',{}).get('search_wall_limited',False) for r in rows),
        unique_fit_records=len(prior),groups=groups,updated_at=now())
    write(root/'summary.json',result);return result

def run(root,max_new_fits):
    root=Path(root);protocol,planning=prepare(root)
    if (root/'run.lock').exists():raise RuntimeError('Existing run lock; inspect before resuming')
    write(root/'run.lock',{'pid':os.getpid(),'started':now(),'source':sha(__file__)})
    for name in ('records','fits','inputs'):(root/name).mkdir(exist_ok=True)
    uniforms=np.asarray(read(root/'test_draws.json')['uniforms'])
    new=0
    try:
        for task in tasks(planning):
            target=root/'records'/(task['key']+'.json')
            if target.exists():continue
            plan=task['plan'];z=(np.vstack([np.asarray(plan).reshape(2,5)]*2)>0).astype(int) if task['phase']=='control' else response(plan,uniforms[task['path']],task['law'])
            data=model.make_data(plan,z,task['case'])
            training,validation=observations(data)
            cache_key,identity=keys(training,validation,protocol)
            fitdir=root/'fits'/cache_key;cached=(fitdir/'result.json').exists()
            if not cached and (new>=max_new_fits or len(list((root/'fits').glob('*/result.json')))>=protocol['max_unique_fits']):break
            row=dict(task,created_at=now(),participation=z,observation_key=cache_key,cache_reused=cached,
                     physical=data['physical_audit'],fit_result=str(fitdir/'result.json'))
            if not (root/'inputs'/(cache_key+'.json')).exists():
                write(root/'inputs'/(cache_key+'.json'),{'identity':identity,'data':data})
            try:
                if not data['physical_audit']['valid']:raise ValueError('Declared physical/contract check failed')
                if cached:
                    fit=read(fitdir/'result.json')
                    if fit.get('status')!='complete':raise RuntimeError('Cached failed fit retained; no implicit retry')
                else:
                    new+=1
                    fit=fit_observed(training,validation,fitdir,root_label=1,search_seconds=25.,resume=False)
                    print(json.dumps({'new':new,'key':task['key'],'seconds':fit['elapsed_seconds']},ensure_ascii=False),flush=True)
                row.update(status='evaluated',fit_sha256=sha(fitdir/'result.json'),evaluation=fit_metrics(fit,data['truth']),
                           true_supports=data['truth']['supports'],identifier_seconds=fit['elapsed_seconds'])
            except Exception as exc:
                row.update(status='failed',error=str(exc),traceback=traceback.format_exc())
                print(json.dumps({'failure':task['key'],'error':str(exc)}),flush=True)
            write(target,row)
            summary(root,protocol,planning)
        result=summary(root,protocol,planning)
        print(json.dumps({k:v for k,v in result.items() if k!='groups'},ensure_ascii=False),flush=True)
        return result
    finally:(root/'run.lock').unlink(missing_ok=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--prepare-only',action='store_true');p.add_argument('--max-new-fits',type=int,default=8)
    args=p.parse_args()
    if args.prepare_only:
        protocol,planning=prepare(args.output);print(json.dumps({'prepared':str(args.output),'main_cells':protocol['main_cells'],'controls':protocol['deterministic_control_cells']}))
    else:run(args.output,args.max_new_fits)
