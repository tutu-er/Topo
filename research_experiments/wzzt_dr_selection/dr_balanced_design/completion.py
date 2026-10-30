"""Separately bounded completion of the original fixed 22-menu job list.

Does not restart the first batch or optimize a new price. Completed records are
read-only cache entries. The one delivered case reuses its completed observed
preprocessing state and executes only previously unstarted methods.
"""
from __future__ import annotations
import argparse
from collections import Counter
from copy import deepcopy
from hashlib import sha256
import importlib.util
import json, os, pickle, sys, traceback, zipfile
from pathlib import Path
from time import perf_counter

REPO=Path(os.environ.get('WZZT_REPO_ROOT',r'C:\Users\23761\.codex\worktrees\wzzt-dr-selection\Topo'))
CACHE=Path(os.environ.get('WZZT_RESEARCH_CACHE',r'C:\Users\23761\.codex\visualizations\2026\10\25\01a1726e-8ec7-7bc1-9f18-391f9cf79a1a'))
METHODS=('current_rebuilt_rnj','current_pool_wzzt','current_root_augmented_rnj')

def sha(p):return sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def plain(v):
    if isinstance(v,dict):return {str(k):plain(x) for k,x in v.items()}
    if isinstance(v,(tuple,list)):return [plain(x) for x in v]
    if hasattr(v,'tolist'):return v.tolist()
    if hasattr(v,'item'):return v.item()
    return v
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    temp=p.with_suffix(p.suffix+'.partial')
    temp.write_text(json.dumps(plain(v),indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8');temp.replace(p)
def key(row):return tuple(row[k] for k in ('flow','case','menu_id'))
def complete(row):
    return all(row.get('evaluated',{}).get(m,{}).get('status')=='complete' and
               row['evaluated'][m].get('all_scans_complete',True) for m in METHODS)
def complete_fit(fit):
    return all(fit.get('methods',{}).get(m,{}).get('status')=='complete' and
               fit['methods'][m].get('all_scans_complete',True) for m in METHODS)
def pending(rows):
    ids=[key(r) for r in rows]
    if len(ids)!=len(set(ids)):raise ValueError('duplicate original job key')
    return [deepcopy(r) for r in rows if not complete(r)]
def merge_rows(old,updates):
    original={key(r):r for r in old};changes={key(r):r for r in updates}
    if len(changes)!=len(updates):raise ValueError('duplicate completion key')
    if not set(changes)<=set(original):raise ValueError('completion added an unplanned job')
    if any(complete(original[k]) for k in changes):raise ValueError('completed original record cannot change')
    return [deepcopy(changes.get(key(r),r)) for r in old]

def runtime():
    stage=REPO/'research_experiments/wzzt_dr_selection/dr_balanced_design'
    sys.path[:0]=[str(stage),str(REPO),str(REPO/'rnj_wzzt_core')]
    name='_wzzt_b_completion_helpers'
    if name in sys.modules:return sys.modules[name]
    spec=importlib.util.spec_from_file_location(name,stage/'study.py')
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    return module

def prepare(root,source):
    root=Path(root).resolve();source=Path(source).resolve()
    if root.exists():raise FileExistsError('new completion root required')
    if not (source/'run_ended.json').is_file():raise ValueError('original batch must have ended')
    original=read(source/'protocol.json');rows=read(source/'evaluations.json');jobs=pending(rows)
    if sha(source/'protocol.json')!=read(source/'protocol.sha256.json')['sha256']:
        raise ValueError('original protocol mismatch')
    if sha(source/'source_snapshot.zip')!=original['source_snapshot_sha256']:
        raise ValueError('original frozen source archive mismatch')
    actions_sha=sha(source/'actions_frozen.json')
    if actions_sha!=read(source/'actual_generation_started.json')['actions_sha256'] or any(r['actions_sha256']!=actions_sha for r in rows):
        raise ValueError('original frozen prices/actions mismatch')
    if [key(r) for r in rows]!=[key(r) for r in read(source/'actions_frozen.json')['ordered_jobs']]:
        raise ValueError('original job order/scope mismatch')
    for item in ('history','history_state','historical_W','math_gate'):
        if sha(original[item]['path'])!=original[item]['sha256']:
            raise ValueError('original frozen input changed: '+item)
    # Only the documented metadata correction in study.py may differ from the
    # first batch's source snapshot. All numerical/control modules stay exact.
    for p,h in original['source_sha256'].items():
        if sha(p)!=h and not (Path(p).name=='study.py' and Path(p).parent.name=='dr_balanced_design'):
            raise ValueError('original numerical/control source changed: '+p)
    if len(rows)!=308 or len(jobs)!=91:raise ValueError('this completion locks exactly the original308/91 scope')
    helper=runtime();sources=helper.base._fixed_source_hashes()
    for p in (Path(__file__).resolve(),Path(helper.__file__),Path(helper.execution.__file__),
              Path(helper.original.__file__),Path(helper.reference.__file__),
              REPO/'research_experiments/wzzt_dr_selection/dr_balanced_design/menus.py',
              REPO/'research_experiments/wzzt_dr_selection/dr_compensation_design/controls.py',
              REPO/'research_experiments/wzzt_dr_selection/dr_random_observation_gate/random_code.py'):
        sources[str(p.resolve())]=sha(p)
    # Bind all existing evidence, including completed observations. Nothing is
    # overwritten, and a final binding check detects any source-root mutation.
    inputs={str(p.resolve()):sha(p) for p in source.rglob('*') if p.is_file()}
    for k in ('history','history_state','historical_W','math_gate'):
        p=Path(original[k]['path']);inputs[str(p.resolve())]=sha(p)
    for p,h in inputs.items():
        if p==str((source/'protocol.json').resolve()) and h!=read(source/'protocol.sha256.json')['sha256']:
            raise ValueError('original protocol does not match its frozen hash')
    config=dict(status='frozen_before_completion_compute_or_generation',created_utc=helper.base.now(),
        source_root=str(source),original_protocol_sha256=sha(source/'protocol.json'),
        original_evaluations_sha256=sha(source/'evaluations.json'),original_actions_sha256=sha(source/'actions_frozen.json'),
        original_LP_calls=read(source/'run_ended.json')['LP'],
        source_sha256=sources,input_sha256=inputs,pending_keys=[key(j) for j in jobs],
        originally_delivered_pending=sum(j.get('execution_status')=='executed' for j in jobs),
        originally_unexecuted_pending=sum(j.get('execution_status')=='not_executed' for j in jobs),
        unchanged='all original menus/actions, cases, seeds, observations, prices, budget and rules; no independent-confirmation claim',
        cache='reuse every completed exact-packet fit; reuse completed preprocessing for delivered pending fit',
        completion_scope='only original91 incomplete mappings; no completed job may be rerun or replaced',
        computation=dict(max_LP_calls=1800,wall_seconds=600.,separate_from_original3000=True),
        stop='preserve every incomplete result; no implicit retry, extension, new seed, menu or noise change')
    root.mkdir(parents=True)
    with zipfile.ZipFile(root/'source_snapshot.zip','w',zipfile.ZIP_DEFLATED) as z:
        for i,p in enumerate(sources):z.writestr(f'{i:04d}_{Path(p).name}',Path(p).read_bytes())
    config['source_snapshot_sha256']=sha(root/'source_snapshot.zip')
    write(root/'protocol.json',config);write(root/'protocol.sha256.json',dict(sha256=sha(root/'protocol.json')))
    write(root/'pending_original_rows.json',jobs)
    return dict(status=config['status'],pending_maps=len(jobs),computation=config['computation'])

def verify(root):
    config=read(root/'protocol.json')
    if sha(root/'protocol.json')!=read(root/'protocol.sha256.json')['sha256']:raise RuntimeError('completion protocol drift')
    for kind in ('source_sha256','input_sha256'):
        for p,h in config[kind].items():
            if sha(p)!=h:raise RuntimeError(kind+' drift: '+p)
    return config

def resume_delivered(helper,history,future,oldfit,settings,budget,out):
    """No QP/shared-bank rebuilding: only unstarted per-method fit contexts."""
    execution=helper.execution;base=helper.base
    train,valid,days,hashes,counts=base._observations(history['training']+future['training'],history['validation']+future['validation'])
    current_path=Path(oldfit['current_state_path']);current=read(current_path)
    if current.get('status')!='complete' or hashes!=oldfit['input_sha256'] or hashes!=current['input_sha256']:
        raise ValueError('cannot reuse a different/incomplete observed preprocessing state')
    result=deepcopy(oldfit);result.update(status='completion_running',completion_reused_preprocessing=True,
        original_preprocessing_receipt_count=len(oldfit.get('shared_preprocessing_LP_solves',[])),
        shared_preprocessing_LP_solves=[],completion_current_state_sha256=sha(current_path),
        completion_source_sha256={str(Path(__file__).resolve()):sha(__file__)})
    write(out,result)
    for name in METHODS:
        if result.get('methods',{}).get(name,{}).get('status')=='complete':continue
        try:
            budget.check()
            if name=='current_root_augmented_rnj':
                method=helper.reference.run_augmented(current,train,valid,settings,budget,day_ids=days)
            else:method=base._run_method(name,current,train,valid,current['pool'],settings,budget,day_ids=days,current_state=current)
            result['methods'][name]=method
        except Exception as exc:
            result['methods'][name]=dict(status='completion_unresolved',error_type=type(exc).__name__,error=str(exc),LP_solves=[])
        write(out,result)
    result.update(status='complete_record',all_methods_complete=complete_fit(result),
        all_W_scans_complete=result.get('methods',{}).get('current_pool_wzzt',{}).get('all_scans_complete'))
    write(out,result);return result

def run(root):
    root=Path(root).resolve();config=verify(root)
    if (root/'run_started.json').exists():raise FileExistsError('completion already attempted; no implicit restart')
    helper=runtime();base=helper.base;source=Path(config['source_root'])
    original=read(source/'protocol.json');oldrows=read(source/'evaluations.json');jobs=pending(oldrows)
    if [list(key(j)) for j in jobs]!=config['pending_keys']:raise ValueError('pending list drift')
    write(root/'run_started.json',dict(pid=os.getpid(),created_utc=base.now(),protocol_sha256=sha(root/'protocol.json')))
    history=pickle.loads(Path(original['history']['path']).read_bytes());state=read(original['history_state']['path'])
    families=read(source/'planning/common_witness.json')['families'];design=read(source/'actions_frozen.json')
    budget=base.Budget(config['computation']['wall_seconds'],config['computation']['max_LP_calls'])
    cache={};fit_paths={}
    for row in oldrows:
        if not complete(row):continue
        receipt=read(row['execution_receipt']);h=receipt['packet_sha256'];p=source/'records'/(row['fit_id']+'.json')
        if h not in cache:cache[h]=read(p);fit_paths[h]=str(p)
    newrows=[];flows={f['id']:f for f in original['flows']};cases={c['id']:c for c in original['cases']};menus={m['id']:m for m in original['menus']}
    for j in jobs:
        row=deepcopy(j);row['completion_original_key']=key(j);row['completion_protocol_sha256']=sha(root/'protocol.json')
        try:
            budget.check();flow=flows[j['flow']];case=cases[j['case']];menu=menus[j['menu_id']]
            _,_,truth=helper.original.truth(.4,case['c'],case['pair'],original);target=set(map(tuple,truth['effective_supports']))
            if j.get('execution_status')=='executed':
                receipt=read(j['execution_receipt']);h=receipt['packet_sha256']
                packet=source/'private_observations/packets'/(h+'.pkl')
                if sha(packet)!=h:raise RuntimeError('already delivered packet drift')
                future=pickle.loads(packet.read_bytes());row['completion_reused_execution']=True
            else:
                R,X,_=helper.original.truth(.4,case['c'],case['pair'],original)
                row.update(execution_status='unresolved',actual_payment=None)
                future,receipt=helper.execution.generate_future(R,X,menu,flow['behavior'],flow['measurement'],original)
                h=receipt['packet_sha256'];receipt['actions_sha256']=config['original_actions_sha256']
                receipt['completion_protocol_sha256']=sha(root/'protocol.json')
                rp=root/'execution_receipts'/f"{j['flow']}_{j['case']}_{j['menu_id']}.json";write(rp,receipt)
                row.update(execution_status='executed',actual_payment=receipt['actual_payment'],execution_receipt=str(rp))
                helper.original.save_packet(root/'private_observations',future)
            fid='s_'+h[:20];row['fit_id']=fid
            if h in cache:
                result=cache[h];row['completion_reused_complete_fit']=True;row['completion_record_path']=fit_paths[h]
            else:
                dest=root/'records'/(fid+'.json')
                if j.get('execution_status')=='executed':
                    oldfit=read(source/'records'/(j['fit_id']+'.json'))
                    result=resume_delivered(helper,history,future,oldfit,original['settings'],budget,dest)
                else:result=helper.execution.identify_three(history,future,state,original['settings'],budget,dest)
                if complete_fit(result):
                    try:result['historical_family_diagnostic']=helper.diagnostic(history,future,families,original['settings'],budget,result['applied_calibration'])
                    except Exception as exc:result['historical_family_diagnostic']=dict(status='unresolved',error=str(exc))
                write(dest,result);cache[h]=result;fit_paths[h]=str(dest);row['completion_record_path']=str(dest)
            row.update(status=result['status'],evaluated={m:helper.evaluate_method(result.get('methods',{}).get(m,{}),target) for m in METHODS})
            hd=result.get('historical_family_diagnostic',{});row['historical_family_diagnostic']=None
            if hd.get('status')=='complete':
                s=hd['selected'];row['historical_family_diagnostic']=dict(best_raw_errors=min(helper.errors(p,target) for p in hd['points']),
                    selected_final_errors=helper.errors(s,target),selected_raw_errors=helper.errors(s['selected_uncontracted'],target))
        except Exception as exc:row.update(status='completion_unresolved',completion_error_type=type(exc).__name__,completion_error=str(exc))
        newrows.append(row);write(root/'completion_rows.json',newrows)
        if len(newrows)%10==0:print(json.dumps(dict(completion_maps=len(newrows),new_LP=budget.calls,seconds=perf_counter()-budget.started)),flush=True)
    merged=merge_rows(oldrows,newrows);write(root/'merged_evaluations.json',merged)
    result=helper.summarize(root,merged,original,design,budget)
    result.update(original_LP_calls=config['original_LP_calls'],completion_LP_calls=budget.calls,
        cumulative_LP_calls=config['original_LP_calls']+budget.calls,original_complete_maps=sum(complete(r) for r in oldrows),
        completion_attempted_maps=len(newrows),merged_complete_maps=sum(complete(r) for r in merged),
        completed_original_records_unchanged=True,separate_budget_scope=True,
        reused_completed_observation_fits=sum(r.get('completion_reused_complete_fit',False) for r in newrows),
        reused_delivered_execution_maps=sum(r.get('completion_reused_execution',False) for r in newrows),
        new_execution_maps=sum(r.get('execution_status')=='executed' and not r.get('completion_reused_execution',False) for r in newrows))
    # End check binds original evidence against any inadvertent overwrite.
    for p,h in config['input_sha256'].items():
        if sha(p)!=h:raise RuntimeError('original evidence changed during completion: '+p)
    write(root/'summary.json',result);write(root/'run_ended.json',dict(status=result['status'],created_utc=base.now(),
        original_LP_calls=config['original_LP_calls'],new_LP_calls=budget.calls,cumulative_LP_calls=config['original_LP_calls']+budget.calls))
    return {k:v for k,v in result.items() if k not in ('menu_results','policy_results','paired')}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','run']);p.add_argument('--root',required=True)
    p.add_argument('--source',default=str(CACHE/'wzzt_b_design_v1'));a=p.parse_args()
    print(json.dumps(prepare(a.root,a.source) if a.mode=='prepare' else run(a.root),ensure_ascii=False),flush=True)
