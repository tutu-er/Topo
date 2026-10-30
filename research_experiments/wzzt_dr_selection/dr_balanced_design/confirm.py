"""New behavior/measurement confirmation of a validation selector repair.

Prices are fixed simple exploration policies, never optimized against truth.
All identification remains unchanged. Alternative selectors see saved fitted
points and independently recomputed original validation losses only.
"""
from __future__ import annotations
import argparse
from collections import Counter
from copy import deepcopy
from hashlib import sha256
import importlib.util
import json, os, pickle, random, sys, zipfile
from pathlib import Path
from time import perf_counter

REPO=Path(os.environ.get('WZZT_REPO_ROOT',r'C:\Users\23761\.codex\worktrees\wzzt-dr-selection\Topo'))
CACHE=Path(os.environ.get('WZZT_RESEARCH_CACHE',r'C:\Users\23761\.codex\visualizations\2026\10\25\01a1726e-8ec7-7bc1-9f18-391f9cf79a1a'))
POLICIES=('zero','single2','rotation','pair1','pair_rotation')
RULES=('original_3pct_contract','strict_raw_or_contract')
METHODS=('current_rebuilt_rnj','current_pool_wzzt','current_root_augmented_rnj')

def sha(p):return sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def plain(v):
    if isinstance(v,dict):return {str(k):plain(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)):return [plain(x) for x in v]
    if hasattr(v,'tolist'):return v.tolist()
    if hasattr(v,'item'):return v.item()
    return v
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);temp=p.with_suffix(p.suffix+'.partial')
    temp.write_text(json.dumps(plain(v),indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8');temp.replace(p)
def load_module(path,name):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module;spec.loader.exec_module(module);return module
def runtime():
    stage=REPO/'research_experiments/wzzt_dr_selection/dr_balanced_design'
    sys.path[:0]=[str(stage),str(REPO),str(REPO/'rnj_wzzt_core')]
    return load_module(stage/'study.py','_wzzt_b_confirm_helper')
def key(r):return tuple(r[x] for x in ('flow','case','policy'))
def build_actions(flows,menu_module):
    actions=[]
    for flow in flows:
        picked={'zero':menu_module.resolve('zero')}
        for i,name in enumerate(POLICIES[1:]):picked[name]=menu_module.baseline_menu(name,flow['baseline_seed']+i)
        actions.append(dict(flow=deepcopy(flow),menus=picked))
    return actions
def make_jobs(actions,cases,seed):
    jobs=[dict(flow=a['flow']['id'],case=c['id'],policy=p) for a in actions for c in cases for p in POLICIES]
    random.Random(seed).shuffle(jobs)
    if len(jobs)!=len({key(j) for j in jobs}):raise ValueError('duplicate job key')
    return jobs
def effective(p):return {tuple(s) for s,r,x in zip(p['supports'],p['r'],p['x'],strict=True) if max(r,x)>1e-9}
def errors(p,target):return len(effective(p)^set(target))

def selector_replay(method,observations,cal,rules):
    """No truth argument exists at this boundary."""
    if method.get('status')!='complete':return dict(status=method.get('status','not_started'))
    scored=[]
    for p in method['points']:
        contracted=rules.contract(p,cal)
        scored.append(dict(raw=p,contract=contracted,raw_val=rules.score(p,observations),contract_val=rules.score(contracted,observations)))
    selected={name:rules.choose(scored,name) for name in RULES}
    actual=method['selected'];actual_point=dict(supports=actual['support_indices'],r=actual['r_values'],x=actual['x_values'])
    # Exact matrix equality is preferable to coefficient equality when feasible
    # R/X decompositions are nonunique. No fitted weights are changed here.
    actualR,actualX=rules.matrices(actual_point);replayR,replayX=rules.matrices(selected[RULES[0]]['point'])
    import numpy as np
    same=bool(np.array_equal(actualR,replayR) and np.array_equal(actualX,replayX))
    if not same:raise RuntimeError('recomputed original rule does not reproduce actual final output')
    return dict(status='complete',choices=selected,original_replay_matches=True,
        candidate_validation_losses=[dict(source_index=i,raw=v['raw_val'],contract=v['contract_val']) for i,v in enumerate(scored)],
        numerical_tie=1e-12,uses_truth=False)

def prepare(root,rules_path=None):
    root=Path(root).resolve()
    if root.exists():raise FileExistsError('new confirmation root required')
    helper=runtime();base=helper.base;oldroot=CACHE/'wzzt_b_design_v1';old=read(oldroot/'protocol.json')
    if sha(oldroot/'protocol.json')!=read(oldroot/'protocol.sha256.json')['sha256']:raise ValueError('original settings protocol drift')
    if rules_path is None:
        proposed=REPO/'research_experiments/wzzt_dr_selection/dr_balanced_design/selection_rules.py'
        rules_path=proposed if proposed.is_file() else CACHE/'wzzt_b_selection_stage/selection_rules.py'
    rules_path=Path(rules_path).resolve()
    flows=[dict(id='r0',behavior=934031,measurement=934131,baseline_seed=934331),
           dict(id='r1',behavior=934032,measurement=934132,baseline_seed=934332)]
    actions=build_actions(flows,helper.menus);jobs=make_jobs(actions,old['cases'],934431)
    sources=base._fixed_source_hashes()
    for p in [Path(__file__).resolve(),Path(helper.__file__),Path(helper.execution.__file__),Path(helper.original.__file__),Path(helper.reference.__file__),
              rules_path,REPO/'research_experiments/wzzt_dr_selection/dr_balanced_design/menus.py',
              REPO/'research_experiments/wzzt_dr_selection/dr_compensation_design/controls.py',
              REPO/'research_experiments/wzzt_dr_selection/dr_random_observation_gate/random_code.py']:
        sources[str(p.resolve())]=sha(p)
    cfg=dict(status='locked_before_new_behavior_measurement_generation_or_fit',created_utc=base.now(),
        old_settings_protocol=dict(path=str(oldroot/'protocol.json'),sha256=sha(oldroot/'protocol.json')),
        history=deepcopy(old['history']),history_state=deepcopy(old['history_state']),
        base_kw=200.,kappa=.35,L=2.6,b=.4,labels=[101,102,103],seeds=dict(natural=884001),
        regime=deepcopy(old['regime']),settings=deepcopy(old['settings']),cases=deepcopy(old['cases']),
        flows=flows,actions=actions,ordered_jobs=jobs,order_seed=934431,
        policies=list(POLICIES),methods=list(METHODS),selectors=list(RULES),planned_maps=70,
        rules_source=dict(path=str(rules_path),sha256=sha(rules_path)),source_sha256=sources,
        historical_rows=[288,288],actual_rows=[480,480],budget=1.6,
        contract='unchanged0/1/2 prices and0/.5/.75 participation; .2kW/.07kvar shift; four complete events',
        preservation='original whole-history identification, all continuous weights from0, full original accepted paths, gain, bound and search settings',
        changed='only compare original3%+unconditional contraction against strict validation minimum over saved raw/contract variants',
        information='all fixed exploration menus frozen before truth/future; selector receives observations and saved fitted points, no truth',
        scope='one new behavior/measurement confirmation of a development selector; shared original natural future; not a new price optimizer or70 independent samples',
        computation=dict(max_LP_calls=1200,wall_seconds=600.,single_batch=True),
        stop='no extra flow/seed/noise/budget/price/threshold tuning; every failure or unstarted result remains visible')
    for k in ('history','history_state'):
        if sha(cfg[k]['path'])!=cfg[k]['sha256']:raise ValueError('frozen historical input drift')
    root.mkdir(parents=True)
    with zipfile.ZipFile(root/'source_snapshot.zip','w',zipfile.ZIP_DEFLATED) as z:
        for i,p in enumerate(sources):z.writestr(f'{i:04d}_{Path(p).name}',Path(p).read_bytes())
    cfg['source_snapshot_sha256']=sha(root/'source_snapshot.zip')
    write(root/'protocol.json',cfg);write(root/'protocol.sha256.json',dict(sha256=sha(root/'protocol.json')))
    write(root/'actions_frozen.json',dict(actions=actions,ordered_jobs=jobs,truth_read=False,actual_future_generated=False))
    return dict(status=cfg['status'],planned_maps=70,policies=list(POLICIES),budget=cfg['computation'])

def verify(root):
    cfg=read(root/'protocol.json')
    if sha(root/'protocol.json')!=read(root/'protocol.sha256.json')['sha256']:raise RuntimeError('confirmation protocol drift')
    for p,h in cfg['source_sha256'].items():
        if sha(p)!=h:raise RuntimeError('confirmation source drift: '+p)
    for k in ('history','history_state','old_settings_protocol','rules_source'):
        if sha(cfg[k]['path'])!=cfg[k]['sha256']:raise RuntimeError('frozen input drift: '+k)
    actions=read(root/'actions_frozen.json')
    if actions['actions']!=cfg['actions'] or actions['ordered_jobs']!=cfg['ordered_jobs']:
        raise RuntimeError('frozen actions or job order drift')
    return cfg

def summarized(rows,cfg,budget):
    entries=[]
    for policy in POLICIES:
        for method in METHODS:
            for rule in RULES:
                group=[r for r in rows if r['policy']==policy];values=[r.get('evaluated',{}).get(method,{}).get(rule,{}) for r in group]
                done=[v for v in values if v.get('status')=='complete']
                entries.append(dict(policy=policy,method=method,rule=rule,planned=14,recorded=len(group),complete=len(done),
                    unresolved_or_unstarted=14-len(done),final_errors=sum(v['final_errors'] for v in done),
                    full=sum(v['final_errors']==0 for v in done),contract_selected=sum(v['variant']=='contract' for v in done),
                    partial_scans=sum(not v['all_scans_complete'] for v in done),completed_subset_only=True))
    paired=[]
    for method in METHODS:
        counts=Counter();err_old=err_new=0
        for r in rows:
            v=r.get('evaluated',{}).get(method,{});a=v.get(RULES[0],{});b=v.get(RULES[1],{})
            if a.get('status')!='complete' or b.get('status')!='complete':counts['unresolved']+=1;continue
            x,y=a['final_errors'],b['final_errors'];err_old+=x;err_new+=y;counts['better' if y<x else 'worse' if y>x else 'same']+=1
        paired.append(dict(method=method,strict_vs_original=dict(counts),original_errors=err_old,strict_errors=err_new))
    pay={}
    for r in rows:
        if r.get('execution_status')=='executed':
            k=r['flow']+'|'+r['policy']
            if k in pay and abs(pay[k]-r['actual_payment'])>1e-12:raise RuntimeError('counterfactual control costs differ within one deployment')
            pay[k]=r['actual_payment']
    complete=len(rows)==70 and all(r.get('evaluated',{}).get(m,{}).get(rule,{}).get('status')=='complete' and
        r['evaluated'][m][rule]['all_scans_complete'] for r in rows for m in METHODS for rule in RULES)
    return dict(status='all_complete' if complete else 'bounded_finished_with_unresolved',planned_maps=70,recorded_maps=len(rows),
        executed_maps=sum(r.get('execution_status')=='executed' for r in rows),
        different_observations=len({r['fit_id'] for r in rows if 'fit_id' in r}),LP_calls=budget.calls,seconds=perf_counter()-budget.started,
        results=entries,paired=paired,payments_per_four_event_deployment=pay,
        scope='new behavior and measurement, same natural input and seven counterfactual models; not70 independent natural paths')

def public_summary(summary,cfg,root):
    """Explicit aggregate whitelist: never a raw packet/receipt/point export."""
    return dict(status=summary['status'],planned_maps=70,recorded_maps=summary['recorded_maps'],executed_maps=summary['executed_maps'],
        different_observations=summary['different_observations'],LP_calls=summary['LP_calls'],seconds=summary['seconds'],
        results=summary['results'],paired=summary['paired'],payments_per_four_event_deployment=summary['payments_per_four_event_deployment'],
        behavior_seeds=[f['behavior'] for f in cfg['flows']],measurement_seeds=[f['measurement'] for f in cfg['flows']],
        natural_seed=884001,prices=[0,1,2],participation=[0,.5,.75],peak_kw=.2,peak_kvar=.07,worst_payment_limit=1.6,
        rules_source_sha256=cfg['rules_source']['sha256'],protocol_sha256=sha(root/'protocol.json'),actions_sha256=sha(root/'actions_frozen.json'),
        new_price_optimizer=False,new_independent_natural_load=False,private_arrays_exported=False)

def run(root):
    root=Path(root).resolve();cfg=verify(root)
    if (root/'run_started.json').exists():raise FileExistsError('confirmation already attempted; no implicit restart')
    helper=runtime();base=helper.base;rules=load_module(cfg['rules_source']['path'],'_wzzt_b_confirm_selectors')
    actions=read(root/'actions_frozen.json')
    if actions['truth_read'] or actions['actual_future_generated']:raise ValueError('invalid frozen information boundary')
    write(root/'run_started.json',dict(pid=os.getpid(),created_utc=base.now(),protocol_sha256=sha(root/'protocol.json'),actions_sha256=sha(root/'actions_frozen.json')))
    history=pickle.loads(Path(cfg['history']['path']).read_bytes());state=read(cfg['history_state']['path'])
    budget=base.Budget(cfg['computation']['wall_seconds'],cfg['computation']['max_LP_calls'])
    bycase={c['id']:c for c in cfg['cases']};byflow={f['id']:f for f in cfg['flows']};bymenu={a['flow']['id']:a['menus'] for a in cfg['actions']}
    cache={};rows=[]
    for job in cfg['ordered_jobs']:
        row=dict(**job,status='not_started',execution_status='not_executed',actual_payment=0.)
        try:
            budget.check();case=bycase[job['case']];flow=byflow[job['flow']];menu=bymenu[job['flow']][job['policy']]
            R,X,truth=helper.original.truth(.4,case['c'],case['pair'],cfg);target=set(map(tuple,truth['effective_supports']))
            row.update(execution_status='unresolved',actual_payment=None,menu_id=menu['id'])
            future,receipt=helper.execution.generate_future(R,X,menu,flow['behavior'],flow['measurement'],cfg)
            receipt['actions_sha256']=sha(root/'actions_frozen.json');rp=root/'execution_receipts'/f"{job['flow']}_{job['case']}_{job['policy']}.json";write(rp,receipt)
            h=receipt['packet_sha256'];fid='s_'+h[:20];row.update(execution_status='executed',actual_payment=receipt['actual_payment'],execution_receipt=str(rp),fit_id=fid)
            if h not in cache:
                helper.original.save_packet(root/'private_observations',future)
                record=helper.execution.identify_three(history,future,state,cfg['settings'],budget,root/'records'/(fid+'.json'))
                _,valid,_,_,_=base._observations(history['training']+future['training'],history['validation']+future['validation'])
                observations=helper.arrays(valid);comparison={}
                for name in METHODS:
                    try:comparison[name]=selector_replay(record.get('methods',{}).get(name,{}),observations,record['applied_calibration'],rules)
                    except Exception as exc:comparison[name]=dict(status='selector_unresolved',error_type=type(exc).__name__,error=str(exc))
                record['selection_comparison']=comparison;write(root/'records'/(fid+'.json'),record);cache[h]=record
            else:record=cache[h];row['reused_observed_fit']=True
            evaluated={}
            for name in METHODS:
                method=record.get('methods',{}).get(name,{});comp=record['selection_comparison'][name];out={}
                for rule in RULES:
                    if comp.get('status')!='complete':out[rule]=dict(status=comp.get('status','not_started'));continue
                    choice=comp['choices'][rule];out[rule]=dict(status='complete',final_errors=errors(choice['point'],target),
                        full=errors(choice['point'],target)==0,variant=choice['variant'],source_index=choice['source_index'],
                        validation_mae=choice['validation_mae'],supports=sorted(effective(choice['point'])),
                        target=sorted(target),all_scans_complete=method.get('all_scans_complete',True))
                evaluated[name]=out
            row.update(status=record['status'],evaluated=evaluated)
        except Exception as exc:row.update(status='unresolved',error_type=type(exc).__name__,error=str(exc))
        rows.append(row);write(root/'evaluations.json',rows)
        if len(rows)%10==0:print(json.dumps(dict(maps=len(rows),LP=budget.calls,seconds=perf_counter()-budget.started)),flush=True)
    summary=summarized(rows,cfg,budget);write(root/'summary.json',summary);write(root/'aggregate_summary.json',public_summary(summary,cfg,root))
    write(root/'run_ended.json',dict(status=summary['status'],created_utc=base.now(),LP_calls=budget.calls))
    return {k:v for k,v in summary.items() if k not in ('results','paired','payments_per_four_event_deployment')}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','run']);p.add_argument('--root',required=True);p.add_argument('--rules-path')
    a=p.parse_args();print(json.dumps(prepare(a.root,a.rules_path) if a.mode=='prepare' else run(a.root),ensure_ascii=False),flush=True)
