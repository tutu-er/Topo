"""B: full historical faces, zero-separation witness, all 22 fixed menus.

No new positive parameter floor, no solver or observations on import.  A
shared continuous feasible point proves degeneracy, rather than replacing the
faces by a few equally weighted parameter representatives.
"""
from __future__ import annotations
import argparse
from collections import Counter
from copy import deepcopy
from hashlib import sha256
import json, os, pickle, platform, random, sys, traceback, zipfile
from pathlib import Path
from time import perf_counter
import numpy as np

HERE=Path(__file__).resolve().parent
REPO=Path(os.environ.get('WZZT_REPO_ROOT',r'C:\Users\23761\.codex\worktrees\wzzt-dr-selection\Topo'))
CACHE=Path(os.environ.get('WZZT_RESEARCH_CACHE',r'C:\Users\23761\.codex\visualizations\2026\10\25\01a1726e-8ec7-7bc1-9f18-391f9cf79a1a'))
sys.path[:0]=[str(HERE),str(REPO),str(REPO/'rnj_wzzt_core')]
import menus
from research_experiments.wzzt_dr_selection.dr_compensation_design import execution
reference, original=execution._runtime()
base=reference.base
from research_experiments.wzzt_dr_selection.dr_pool_v2.observations import arrays,matrices
from research_experiments.wzzt_dr_selection.dr_pool_v2.engine import PoolLP
from research_experiments.wzzt_dr_selection.dr_pool_v2.selection import measure,select

def sha(p):return sha256(Path(p).read_bytes()).hexdigest()
def read(p):return base.read(p)
def write(p,v):return base.write(p,v)
def family(p):return base.canonical(p['supports'])
def effective(p):return {tuple(s) for s,r,x in zip(p['supports'],p['r'],p['x']) if max(r,x)>1e-9}
def errors(p,t):return len(effective(p)^set(t))
def features(p,q,fam):
    z=np.zeros((len(fam),p.shape[1]))
    for a,s in zip(z,fam):a[list(s)]=1.
    return np.column_stack([((p@a)[:,None]*a).ravel() for a in z]+[((q@a)[:,None]*a).ravel() for a in z])

def common_witness(packet,hist):
    """Existence witness over FULL continuous faces, not a coverage claim."""
    train,valid,_,hashes,_=base._observations(packet['training'],packet['validation'])
    p,q,y=arrays(train);pv,qv,yv=arrays(valid)
    point=hist['selected']['selected_uncontracted'];F=family(point)
    cap=point['objective']+1e-10+.001*point['objective']
    vcap=1.03*min(a['validation_mae'] for a in hist['points'])+1e-12
    families=tuple(dict.fromkeys(family(a) for a in hist['points']))
    rows=[];refR,refX=matrices(p.shape[1],point['supports'],point['r'],point['x'])
    for G in families:
        weights={tuple(s):(r,x) for s,r,x in zip(point['supports'],point['r'],point['x'])}
        if not set(F)<=set(G):raise RuntimeError('no common embedding: must solve general face problem')
        r=[weights.get(s,(0.,0.))[0] for s in G];x=[weights.get(s,(0.,0.))[1] for s in G]
        R,X=matrices(p.shape[1],G,r,x)
        tr=float(np.mean(abs(p@R.T+q@X.T-y)));va=float(np.mean(abs(pv@R.T+qv@X.T-yv)))
        checks=dict(train=tr<=cap+1e-12,validation=va<=vcap+1e-12,
                    bounds=min(r+x)>=0 and max(r+x)<=2,
                    common_RX=bool(np.array_equal(R,refR) and np.array_equal(X,refX)))
        if not all(checks.values()):raise RuntimeError('shared face witness failed: '+str(checks))
        rows.append(dict(family=G,r=r,x=x,train_loss=tr,validation_loss=va,checks=checks))
    return dict(status='shared_RX_witness_verified',historical_saved_points=len(hist['points']),
        family_count=len(families),families=families,parameter_domain='each r/x in[0,2], continuous including zero',
        train_cap=cap,validation_cap=vcap,witnesses=rows,input_sha256=hashes,
        response_identity='same R and X across all faces => every fixed input has a zero pairwise minimum distance',
        scope='historical algorithm-tolerance faces of all saved families; not physical uncertainty coverage',
        all_prices_uniform_separation=0.,new_LP_calls=0)

def prepare(root):
    root=Path(root).resolve()
    if root.exists():raise FileExistsError('new frozen root required')
    old=read(CACHE/'wzzt_dr_compensation_v1/protocol.json')
    history=old['history'];state=old['history_state']
    hw=CACHE/'wzzt_dr_compensation_v1/planning/historical_wzzt.json'
    mathroot=CACHE/'wzzt_b_math_stage/run_v1'
    mathresult=read(mathroot/'summary.json')
    if mathresult.get('status') not in ('passed','complete','all_checks_passed'):
        raise RuntimeError('mathematical gate not passed: '+str(mathresult.get('status')))
    sources=base._fixed_source_hashes()
    for p in [*HERE.glob('*.py'),Path(execution.__file__),Path(original.__file__),Path(reference.__file__),
              REPO/'research_experiments/wzzt_dr_selection/dr_compensation_design/controls.py',
              REPO/'research_experiments/wzzt_dr_selection/dr_random_observation_gate/random_code.py']:
        sources[str(p.resolve())]=sha(p)
    allmenus=menus.menus()
    cfg=dict(status='frozen_before_design_and_actual_generation',created_utc=base.now(),
        history=history,history_state=state,historical_W=dict(path=str(hw),sha256=sha(hw)),
        math_gate=dict(path=str(mathroot/'summary.json'),sha256=sha(mathroot/'summary.json')),
        source_sha256=sources,model='three-terminal linear mechanism; no AC or empirical response calibration',
        base_kw=200.,kappa=.35,L=2.6,b=.4,labels=[101,102,103],
        seeds=dict(natural=884001),regime=dict(id='combined_reference',pq_relative=.005,voltage_relative=.0002),
        settings=deepcopy(old['settings']),historical_rows=[288,288],actual_rows=[480,480],
        budget=1.6,menus=allmenus,menu_count=22,dependence='local',
        flows=[dict(id='r0',behavior=924031,measurement=924131,baseline_seed=924331),
               dict(id='r1',behavior=924032,measurement=924132,baseline_seed=924332)],
        cases=[dict(id='zero',c=0.,pair=[0,1])]+[dict(id=f'{name}_{a}{b}',c=c,pair=[a,b])
            for name,c in [('weak',.02),('strong',.6)] for a,b in [(0,1),(0,2),(1,2)]],
        order_seed=924431,expected_maps=308,methods=list(execution.METHODS),
        planner='max-min pairwise separation opportunity over full continuous historical faces; cost tie; common witness yields zero menu',
        counterfactual_sweep='all22 menus executed per case/stream; no test-best menu is called a deployable planner',
        baseline_policy_kinds=['single2','rotation','pair1','pair_rotation'],
        planned_information='observed history only; no simulated truth or actual future before frozen actions',
        complete_history='all saved joint families retained separately with all weights jointly free; never union crossing blocks into a tree',
        historical_diagnostic='refit each complete historical family on same merged observations, independent of actual current W bank; do not replace actual W output',
        rule='original gain>1e-10+.001J;3% validation;.16 contraction; coefficient upper2/lower0 unchanged',
        computation=dict(max_LP_calls=3000,wall_seconds=900.,one_bounded_batch=True),
        acceptance='final complete missing+extra blocks and full recovery against fixed simple exploration; path/raw/probe separation alone insufficient',
        stop='no bound/noise/budget/seed revision; preserve unstarted/failed/unresolved; no automatic larger network',
        environment=dict(python=sys.version,numpy=np.__version__,platform=platform.platform()))
    root.mkdir(parents=True)
    with zipfile.ZipFile(root/'source_snapshot.zip','w',zipfile.ZIP_DEFLATED) as z:
        for i,p in enumerate(sources):z.writestr(f'{i:04d}_{Path(p).name}',Path(p).read_bytes())
    cfg['source_snapshot_sha256']=sha(root/'source_snapshot.zip')
    write(root/'protocol.json',cfg);write(root/'protocol.sha256.json',dict(sha256=sha(root/'protocol.json')))
    return dict(status=cfg['status'],planned_maps=308,menu_count=22,budget=cfg['computation'])

def verify(root):
    cfg=read(root/'protocol.json')
    if sha(root/'protocol.json')!=read(root/'protocol.sha256.json')['sha256']:raise RuntimeError('protocol drift')
    for p,h in cfg['source_sha256'].items():
        if sha(p)!=h:raise RuntimeError('source drift: '+p)
    for key in ('history','history_state','historical_W','math_gate'):
        x=cfg[key]
        if sha(x['path'])!=x['sha256']:raise RuntimeError('input drift: '+key)
    return cfg

def diagnostic(history,future,families,settings,budget,cal):
    train,valid,_,_,_=base._observations(history['training']+future['training'],history['validation']+future['validation'])
    p,q,y=arrays(train);pool=base.canonical([s for f in families for s in f]);points=[]
    lp=PoolLP(None,pool,bound=2.,data=(p,q,y))
    try:
        view=base.SearchView(lp,pool,budget)
        for f in families:points.append(measure(view.fit(f),train,valid))
        return dict(status='complete',points=points,selected=select(points,cal),LP_solves=lp.solves,
            interpretation='historical-family counterfactual only; does not override actual W/RNJ outputs')
    finally:lp.close()

def evaluate_method(m,target):
    if m.get('status')!='complete':return dict(status=m.get('status','not_started'))
    s=m['selected'];raw=s['selected_uncontracted'];final=dict(supports=s['support_indices'],r=s['r_values'],x=s['x_values'])
    best=min(p['validation_mae'] for p in m['points'])
    correct=[p for p in m['points'] if errors(p,target)==0]
    eligible=[p for p in correct if p['validation_mae']<=1.03*best+1e-12]
    return dict(status='complete',final_errors=errors(final,target),raw_errors=errors(raw,target),
        best_saved_errors=min(errors(p,target) for p in m['points']),correct_saved_count=len(correct),
        correct_eligible_count=len(eligible),final=sorted(effective(final)),raw=sorted(effective(raw)),
        target=sorted(target),origin=s.get('selected_origin'),all_scans_complete=m.get('all_scans_complete',True),
        raw_to_final_missing=sorted(effective(raw)-effective(final)),
        path_count=len(m.get('paths',[])))

def summarize(root,rows,cfg,design,budget):
    groups=[]
    for menu in cfg['menus']:
        rr=[r for r in rows if r['menu_id']==menu['id']]
        for method in execution.METHODS:
            vv=[r['evaluated'][method] for r in rr if r.get('evaluated',{}).get(method,{}).get('status')=='complete']
            groups.append(dict(menu_id=menu['id'],kind=menu['kind'],method=method,planned=14,recorded=len(rr),complete=len(vv),
                unresolved_or_unstarted=14-len(vv),final_errors=sum(v['final_errors'] for v in vv),
                raw_errors=sum(v['raw_errors'] for v in vv),full=sum(v['final_errors']==0 for v in vv),
                correct_saved=sum(v['best_saved_errors']==0 for v in vv),
                correct_eligible=sum(v['correct_eligible_count']>0 for v in vv),
                partial_scans=sum(not v['all_scans_complete'] for v in vv),errors_on_completed_only=True))
    by={(r['flow'],r['case'],r['menu_id']):r for r in rows};policies=[]
    for a in design['actions']:
        for name,mid in a['policies'].items():
            for method in execution.METHODS:
                vv=[by.get((a['flow']['id'],case['id'],mid),{}).get('evaluated',{}).get(method,{}) for case in cfg['cases']]
                good=[v for v in vv if v.get('status')=='complete']
                policies.append(dict(flow=a['flow']['id'],policy=name,menu_id=mid,method=method,planned=7,complete=len(good),
                    final_errors=sum(v['final_errors'] for v in good),raw_errors=sum(v['raw_errors'] for v in good),
                    full=sum(v['final_errors']==0 for v in good)))
    paired=[]
    for method in execution.METHODS:
        for kind in cfg['baseline_policy_kinds']:
            counts=Counter();delta=0
            for a in design['actions']:
                for case in cfg['cases']:
                    x=by.get((a['flow']['id'],case['id'],'zero'),{}).get('evaluated',{}).get(method,{})
                    v=by.get((a['flow']['id'],case['id'],a['policies'][kind]),{}).get('evaluated',{}).get(method,{})
                    if x.get('status')!='complete' or v.get('status')!='complete':counts['unresolved']+=1;continue
                    d=v['final_errors']-x['final_errors'];delta+=d
                    counts['improved' if d<0 else 'worse' if d>0 else 'same']+=1
            paired.append(dict(method=method,baseline=kind,paid_vs_zero=dict(counts),paid_minus_zero_errors=delta))
    complete=len(rows)==308 and all(r.get('evaluated',{}).get(m,{}).get('status')=='complete' and r['evaluated'][m].get('all_scans_complete',True) for r in rows for m in execution.METHODS)
    result=dict(status='all_recorded_complete' if complete else 'bounded_finished_with_unresolved',planned_maps=308,recorded_maps=len(rows),
        unique_fits=len({r['fit_id'] for r in rows if 'fit_id' in r}),LP_calls=budget.calls,seconds=perf_counter()-budget.started,
        menu_count=22,price_design=design['status'],guided_menu='zero',menu_results=groups,policy_results=policies,paired=paired,
        interpretation='two behavior/measurement streams share natural input and counterfactual truths; not308 independent samples')
    write(root/'summary.json',result);return result

def run(root):
    root=Path(root).resolve();cfg=verify(root)
    if (root/'run_started.json').exists():raise FileExistsError('completed or interrupted root retained; no implicit restart')
    write(root/'run_started.json',dict(pid=os.getpid(),created_utc=base.now(),protocol_sha256=sha(root/'protocol.json')))
    history=pickle.loads(Path(cfg['history']['path']).read_bytes());hist=read(cfg['historical_W']['path']);state=read(cfg['history_state']['path'])
    witness=common_witness(history,hist);write(root/'planning/common_witness.json',witness)
    scores=[dict(menu_id=m['id'],uniform_separation=0.,minimum_pair_separation_probability=0.,
                 expected_payment=menus.expected_payment(m),worst_payment=m['worst_payment']) for m in cfg['menus']]
    actions=[]
    for flow in cfg['flows']:
        policies=dict(guided='zero',zero='zero')
        for i,kind in enumerate(cfg['baseline_policy_kinds']):policies[kind]=menus.baseline_menu(kind,flow['baseline_seed']+i)['id']
        actions.append(dict(flow=flow,policies=policies))
    jobs=[dict(flow=f['id'],case=c['id'],menu_id=m['id']) for f in cfg['flows'] for c in cfg['cases'] for m in cfg['menus']]
    random.Random(cfg['order_seed']).shuffle(jobs)
    design=dict(status='degenerate_exact_shared_point_all22',scores=scores,actions=actions,ordered_jobs=jobs,
                future_generated=False,truth_read_by_planner=False,full_historical_families=witness['families'])
    write(root/'actions_frozen.json',design);ah=sha(root/'actions_frozen.json')
    write(root/'actual_generation_started.json',dict(actions_sha256=ah,created_utc=base.now()))
    budget=base.Budget(cfg['computation']['wall_seconds'],cfg['computation']['max_LP_calls'])
    rows=[];fits={};byflow={f['id']:f for f in cfg['flows']};bycase={c['id']:c for c in cfg['cases']};bymenu={m['id']:m for m in cfg['menus']}
    for job in jobs:
        row=dict(**job,status='not_started',execution_status='not_executed',actual_payment=0.,actions_sha256=ah)
        try:
            budget.check();f=byflow[job['flow']];case=bycase[job['case']];m=bymenu[job['menu_id']]
            R,X,truth=original.truth(.4,case['c'],case['pair'],cfg);target=set(map(tuple,truth['effective_supports']))
            row.update(execution_status='unresolved',actual_payment=None)
            future,receipt=execution.generate_future(R,X,m,f['behavior'],f['measurement'],cfg)
            rp=root/'execution_receipts'/f"{f['id']}_{case['id']}_{m['id']}.json";receipt['actions_sha256']=ah;write(rp,receipt)
            h=receipt['packet_sha256'];fid='s_'+h[:20]
            row.update(execution_status='executed',actual_payment=receipt['actual_payment'],execution_receipt=str(rp),fit_id=fid)
            if h not in fits:
                original.save_packet(root/'private_observations',future)
                result=execution.identify_three(history,future,state,cfg['settings'],budget,root/'records'/(fid+'.json'))
                if result.get('all_methods_complete'):
                    try:result['historical_family_diagnostic']=diagnostic(history,future,witness['families'],cfg['settings'],budget,result['applied_calibration'])
                    except Exception as exc:result['historical_family_diagnostic']=dict(status='unresolved',error=str(exc))
                write(root/'records'/(fid+'.json'),result);fits[h]=result
            else:result=fits[h];row['reused_observed_fit']=True
            evaluated={name:evaluate_method(result.get('methods',{}).get(name,{}),target) for name in execution.METHODS}
            hd=result.get('historical_family_diagnostic',{})
            diag=None
            if hd.get('status')=='complete':
                s=hd['selected'];fp=dict(supports=s['supports'],r=s['r'],x=s['x'])
                diag=dict(best_raw_errors=min(errors(p,target) for p in hd['points']),selected_final_errors=errors(fp,target),
                          selected_raw_errors=errors(s['selected_uncontracted'],target))
            row.update(status=result['status'],evaluated=evaluated,historical_family_diagnostic=diag)
        except Exception as exc:row.update(status='unresolved',error_type=type(exc).__name__,error=str(exc))
        rows.append(row);write(root/'evaluations.json',rows)
        if len(rows)%22==0:print(json.dumps(dict(maps=len(rows),unique_fits=len(fits),LP=budget.calls,seconds=perf_counter()-budget.started)),flush=True)
    result=summarize(root,rows,cfg,design,budget)
    write(root/'run_ended.json',dict(status=result['status'],created_utc=base.now(),LP=budget.calls))
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','run']);p.add_argument('--root',required=True)
    a=p.parse_args();out=prepare(a.root) if a.mode=='prepare' else run(a.root)
    if a.mode=='run':out={k:v for k,v in out.items() if k not in ('menu_results','policy_results','paired')}
    print(json.dumps(out,ensure_ascii=False),flush=True)
