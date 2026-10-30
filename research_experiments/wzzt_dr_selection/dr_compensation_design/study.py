"""Locked bounded controller: history -> price -> independent DR -> final fit."""
from pathlib import Path
from hashlib import sha256
from datetime import datetime,timezone
from collections import Counter
import argparse,json,os,pickle,platform,sys,traceback,zipfile
import numpy as np

STAGE=Path(__file__).resolve().parent
REPO=Path(r'C:\Users\23761\.codex\worktrees\wzzt-dr-selection\Topo')
SCRATCH=Path(os.environ.get('WZZT_RESEARCH_CACHE',r'C:\Users\23761\.codex\visualizations\2026\10\25\01a1726e-8ec7-7bc1-9f18-391f9cf79a1a'))
sys.path[:0]=[str(STAGE),str(REPO),str(REPO/'rnj_wzzt_core')]
from controls import menus,baseline_menu
import planning
import execution
base=planning.base

def sha(path):return sha256(Path(path).read_bytes()).hexdigest()
def write(p,v):base.write(p,v)
def read(p):return base.read(p)

def prepare(root):
    root=Path(root).resolve()
    if root.exists():raise FileExistsError('one frozen root retained; no overwrite')
    old=read(SCRATCH/'wzzt_dr_reference_noise_generation_v1/config.json')
    history=next(h for h in old['histories'] if h['id']=='combined_reference')
    hp=SCRATCH/'wzzt_dr_reference_noise_execution_v1/history/combined_reference.json'
    if sha(history['path'])!=history['sha256']:raise RuntimeError('historical packet drift')
    sources={str(p.resolve()):sha(p) for p in STAGE.glob('*.py')}
    sources.update(base._fixed_source_hashes())
    extra=[REPO/'research_experiments/wzzt_dr_selection/dr_random_observation_gate/study.py',
           REPO/'research_experiments/wzzt_dr_selection/dr_random_observation_gate/random_code.py',
           REPO/'research_experiments/wzzt_dr_selection/dr_reference_noise/reference_noise_driver.py']
    sources.update({str(p):sha(p) for p in extra})
    cfg=dict(status='locked_before_planning_simulation_or_solver',created_utc=base.now(),
        history=history,history_state=dict(path=str(hp),sha256=sha(hp)),source_sha256=sources,
        information_boundary='planner receives old observed packet/state only; all actual menus and streams frozen before truth matrices/future generated',
        model='n3 linear matched-history mechanism case; not AC or field calibrated',
        base_kw=200.,kappa=.35,L=2.6,b=.4,labels=[101,102,103],
        seeds=dict(natural=884001,measurement=884002,acceptance_and_independent_code=884003),
        regime=dict(id='combined_reference',pq_relative=.005,voltage_relative=.0002),
        coefficient_domain='each r/x in[0,2]; lower0, all jointly reoptimized',
        historical_rows=[288,288],actual_rows=[480,480],budget=1.6,
        contract=dict(prices=[0,1,2],participation=[0.,.5,.75],peak_kw=.2,peak_kvar=.07,
                      reduction_kWh=.2,whole_events=4,energy_kWh=.4,power_upper_kw=.4),
        planning=dict(reference_F='uncontracted model selected after full historical W path',
            candidates='only actual historical pool; crossing delete then add one C',
            parameter_face='train loss <= J_F+1e-10+.001J_F and old validation <=1.03 minimum full historical candidate validation+1e-12',
            representatives='F and min/max(r_C+.35x_C) on each face; keep zero; K-deduplicate; uniform finite heuristic',
            forecast='three old training days by-quarter mean P; declared synthetic Q=.35P',
            prediction_noise='point mean with no new measurement noise; actual retains full reference noise',
            approximation='one-step closure with complete native candidates, historical raw output calibration; not full future W or future recalibration',
            objective='exact joint train/independent-validation response-state expectation of final complete block error against finite model supports',
            tie='1e-12 objective, then expected payment, then independent uniform among all ties',
            menu_domain='16 declared zero/single1or2/rotation2/simultaneous-opposed-pair1; not full price space',
            zero_weak_interpretation='unit/endpoints are not positive lower bounds or physical truth probabilities'),
        actual_flows=[dict(id='r0',behavior=914031,measurement=914131,menu_seed=914231,baseline_seed=914331),
                      dict(id='r1',behavior=914032,measurement=914132,menu_seed=914232,baseline_seed=914332)],
        actual_cases='b=.4; c0 once, c.02 all3 pairs, c.6 all3 pairs; all original leaf weights retained',
        policies=['guided','zero','single2','rotation','pair1'],expected_policy_case_maps=70,
        methods=list(execution.METHODS),dependence='local',
        settings=dict(coefficient_bound=2.,max_n=3,root_label=1,search_seconds=25.,calibration_snapshot='current_observed'),
        computation=dict(wall_seconds=900.,max_LP_calls=3000,one_bounded_batch=True),
        pass_rule='guided final complete errors lower than rotation and other paid baselines; raw/acceptance alone insufficient; inspect both flows and weak/zero harm; finite test is not general advantage',
        stop_rule='no parameter/menu/noise/budget revision after outcomes; preserve every failure/unresolved/not_started; no automatic expansion',
        environment=dict(python=sys.version,platform=platform.platform(),numpy=np.__version__))
    root.mkdir(parents=True)
    with zipfile.ZipFile(root/'source_snapshot.zip','w',zipfile.ZIP_DEFLATED) as z:
        for i,p in enumerate(sources):z.writestr(f'{i:04d}_{Path(p).name}',Path(p).read_bytes())
    cfg['source_snapshot_sha256']=sha(root/'source_snapshot.zip')
    write(root/'protocol.json',cfg);write(root/'protocol.sha256.json',dict(sha256=sha(root/'protocol.json')))
    return dict(status=cfg['status'],maps=70,inputs='old combined history only',planned_budget=cfg['computation'])

def verify(root):
    root=Path(root);cfg=read(root/'protocol.json')
    if sha(root/'protocol.json')!=read(root/'protocol.sha256.json')['sha256']:raise RuntimeError('protocol drift')
    for p,h in cfg['source_sha256'].items():
        if sha(p)!=h:raise RuntimeError('source drift: '+p)
    for item in (cfg['history'],cfg['history_state']):
        if sha(item['path'])!=item['sha256']:raise RuntimeError('history drift')
    return cfg

def summarize(rows,root,planning_result,budget):
    summary=[]
    for policy in ('guided','zero','single2','rotation','pair1'):
        for method in execution.METHODS:
            group=[r for r in rows if r['policy']==policy]
            complete=[r for r in group if r.get('evaluated',{}).get(method,{}).get('status')=='complete']
            values=[r['evaluated'][method] for r in complete]
            summary.append(dict(policy=policy,method=method,planned=len(group),complete=len(complete),
                unresolved=len(group)-len(complete),final_errors=sum(v['final_errors'] for v in values),
                raw_errors=sum(v['raw_errors'] for v in values),full=sum(v['final_errors']==0 for v in values),
                partial_search=sum(not v['all_scans_complete'] for v in values),errors_on_completed_only=True,
                correct_saved_path_points=sum(v['best_saved_errors']==0 for v in values),
                payment_sum_over_counterfactual_rows=sum(r['actual_payment'] for r in group if r.get('actual_payment') is not None),
                unknown_payment_rows=sum(r.get('actual_payment') is None for r in group),
                not_executed_rows=sum(r.get('execution_status')=='not_executed' for r in group)))
    pairs=[]
    index={(r['flow'],r['case'],r['policy']):r for r in rows}
    for baseline in ('zero','single2','rotation','pair1'):
        for method in execution.METHODS:
            out=Counter();delta=0
            for r in rows:
                if r['policy']!='guided':continue
                b=index.get((r['flow'],r['case'],baseline))
                g=r.get('evaluated',{}).get(method,{})
                bv={} if b is None else b.get('evaluated',{}).get(method,{})
                if g.get('status')!='complete' or bv.get('status')!='complete':out['unresolved']+=1;continue
                d=g['final_errors']-bv['final_errors'];delta+=d
                out['improved' if d<0 else 'worse' if d>0 else 'same']+=1
            pairs.append(dict(baseline=baseline,method=method,counts=dict(out),guided_minus_baseline_errors=delta))
    all_complete=len(rows)==70 and all(r.get('evaluated',{}).get(m,{}).get('status')=='complete' and
        r['evaluated'][m].get('all_scans_complete',True) for r in rows for m in execution.METHODS)
    result=dict(status='all_methods_complete' if all_complete else 'finished_recording_with_unresolved' if len(rows)==70 else 'incomplete',planned_maps=70,recorded_maps=len(rows),
        unique_fits=len({r['fit_id'] for r in rows if 'fit_id' in r}),summary=summary,paired=pairs,
        planned_price_ties=planning_result.get('tied_menu_ids',[]),LP_calls=budget.calls,
        seconds=base.perf_counter()-budget.started,
        interpretation='one historical observation and one natural future, two independent future measurement/behavior streams; conditional paired simulation, not70 independent samples')
    write(Path(root)/'summary.json',result);return result

def run(root):
    root=Path(root);cfg=verify(root)
    if (root/'run_started.json').exists():raise FileExistsError('prior run retained; no duplicate launch')
    write(root/'run_started.json',dict(started_utc=base.now(),protocol_sha256=sha(root/'protocol.json')))
    history=pickle.loads(Path(cfg['history']['path']).read_bytes());state=read(cfg['history_state']['path'])
    budget=base.Budget(cfg['computation']['wall_seconds'],cfg['computation']['max_LP_calls'])
    rows=[];design={}
    try:
        design=planning.run(history,state,cfg['settings'],budget,root/'planning')
        byid={m['id']:m for m in menus()};ties=design['tied_menu_ids']
        actions=[]
        for flow in cfg['actual_flows']:
            rng=np.random.default_rng(flow['menu_seed']);chosen=byid[ties[int(rng.integers(len(ties)))]]
            acts=dict(guided=chosen,zero=byid['zero'])
            for j,key in enumerate(('single2','rotation','pair1')):
                acts[key]=baseline_menu(key,flow['baseline_seed']+j)
            actions.append(dict(flow=flow,menus=acts))
        # Hard temporal boundary: freeze all policies before simulator truth/future.
        write(root/'actual_actions_frozen.json',dict(created_utc=base.now(),actions=actions,planning_source=sha(root/'planning/price_design.json'),
                                                   source_sha256=cfg['source_sha256'],actual_future_generated=False))
        actions_sha=sha(root/'actual_actions_frozen.json')
        write(root/'actual_generation_started.json',dict(created_utc=base.now(),actions_sha256=actions_sha,menus_already_frozen=True))
        from research_experiments.wzzt_dr_selection.dr_random_observation_gate import study as original
        cases=[('zero',0.,[0,1])]+[(f'{label}_{a}{b}',c,[a,b]) for label,c in [('weak',.02),('strong',.6)] for a,b in [(0,1),(0,2),(1,2)]]
        fits={}
        for action in actions:
            flow=action['flow']
            for cid,c,pair in cases:
                R,X,truth=original.truth(.4,c,pair,cfg);target={tuple(s) for s in truth['effective_supports']}
                for policy,menu in action['menus'].items():
                    row=dict(flow=flow['id'],case=cid,c=c,pair=pair,policy=policy,menu_id=menu['id'],
                             actions_sha256=actions_sha,execution_status='not_executed',actual_payment=0.)
                    try:
                        budget.check()
                        row.update(execution_status='unresolved',actual_payment=None)
                        future,receipt=execution.generate_future(R,X,menu,flow['behavior'],flow['measurement'],cfg)
                        receipt['actions_sha256']=actions_sha
                        receipt_path=root/'execution_receipts'/f"{flow['id']}_{cid}_{policy}.json"
                        write(receipt_path,receipt)
                        h=receipt['packet_sha256'];fid='s_'+h[:20]
                        row.update(execution_status='executed',actual_payment=receipt['actual_payment'],fit_id=fid,execution_receipt=str(receipt_path))
                        if h not in fits:
                            original.save_packet(root/'private_observations',future)
                            result=execution.identify_three(history,future,state,cfg['settings'],budget,root/'records'/(fid+'.json'))
                            fits[h]=result
                        else:result=fits[h];row['reused_observed_fit']=True
                        evaluated={}
                        for method in execution.METHODS:
                            m=result.get('methods',{}).get(method,{})
                            if m.get('status')!='complete':evaluated[method]=dict(status=m.get('status','not_started'));continue
                            selected=m['selected'];raw=selected['selected_uncontracted']
                            final=dict(supports=selected['support_indices'],r=selected['r_values'],x=selected['x_values'])
                            evaluated[method]=dict(status='complete',final_errors=planning.errors(final,target),
                                raw_errors=planning.errors(raw,target),best_saved_errors=min(planning.errors(p,target) for p in m['points']),
                                target=sorted(target),final=sorted(planning.effective(final)),raw=sorted(planning.effective(raw)),
                                origin=selected.get('selected_origin'),all_scans_complete=m.get('all_scans_complete',True))
                        row.update(status=result['status'],evaluated=evaluated)
                    except Exception as error:
                        row.update(status='unresolved',error_type=type(error).__name__,error=str(error))
                    rows.append(row);write(root/'evaluations.json',rows)
                    if len(rows)%5==0:print(json.dumps(dict(maps=len(rows),LP=budget.calls,seconds=base.perf_counter()-budget.started)),flush=True)
        result=summarize(rows,root,design,budget)
        write(root/'run_ended.json',dict(status=result['status'],ended_utc=base.now(),LP=budget.calls))
        return result
    except Exception as error:
        failed=dict(status='unresolved',error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc(),
                    LP_calls=budget.calls,completed_maps=len(rows),planned_maps=70,planning_completed=bool(design))
        write(root/'run_ended.json',failed);return failed

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','run']);p.add_argument('--root',required=True)
    a=p.parse_args();print(json.dumps(prepare(a.root) if a.mode=='prepare' else run(a.root),ensure_ascii=False),flush=True)
