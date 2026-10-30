"""Bounded coded-DR planning gate: prepare first, then root-authorized run.

Only old observed history and saved candidate fits are read. Three templates
share one900s/6000LP budget; partial tables never produce a zero-price fallback.
"""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parent))


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def timestamp():return datetime.now(timezone.utc).isoformat()
def plain(x):
    if isinstance(x,dict):return {str(k):plain(v) for k,v in x.items()}
    if isinstance(x,(tuple,list)):return [plain(v) for v in x]
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    return x
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(plain(value),ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def _family(point,cal):
    return frozenset(frozenset(s) for s,r,x in zip(point['supports'],point['r'],point['x'])
                     if len(s)==1 or cal['a']*r+cal['b']*x+1e-10>=.16*cal['median_root_depth'])


def _top(family,n):
    proper={s for s in family if len(s)<n}
    return {s for s in proper if not any(s<t for t in proper)}


def select_historical_pair(old,seed):
    """Current raw fit plus a historically eligible top-crossing explanation."""
    points=old['base_points'];cal=old['calibration'];n=len(old['labels'])
    best=min(p['validation_mae'] for p in points)
    eligible=[i for i,p in enumerate(points) if p['validation_mae']<=1.03*best+1e-12]
    current=min(eligible,key=lambda i:(len(points[i]['supports']),points[i]['validation_mae'],i))
    f=_family(points[current],cal);tf=_top(f,n)
    alternatives={}
    for i in eligible:
        g=_family(points[i],cal)
        if g==f:continue
        if g not in alternatives or (points[i]['validation_mae'],i)<(points[alternatives[g]]['validation_mae'],alternatives[g]):
            alternatives[g]=i
    options=[]
    for g,i in alternatives.items():
        tg=_top(g,n)
        for c in sorted(f,key=lambda s:(len(s),tuple(sorted(s)))):
            for d in sorted(g,key=lambda s:(len(s),tuple(sorted(s)))):
                if c&d and not c<=d and not d<=c:
                    token=json.dumps([seed,current,i,sorted(c),sorted(d)]).encode()
                    options.append((int(c in tf and d in tg),len(c|d),hashlib.sha256(token).hexdigest(),i,c,d))
    if not options:raise ValueError('no current-versus-alternative historical crossing; no substitute case chosen')
    high=max((a,b) for a,b,*_ in options)
    chosen=min((o for o in options if o[:2]==high),key=lambda o:o[2])
    _,_,tie,index,c,d=chosen
    rng=np.random.default_rng(seed)
    a=int(rng.choice(sorted(c-d)));b=int(rng.choice(sorted(d-c)))
    labels=[old['labels'][a],old['labels'][b]]
    return [current,index],labels,dict(rule='current historical fit versus3pct eligible crossing; prefer both candidate top groups then largest union; seeded ties/sites',
        seed=seed,current_raw_index=current,alternative_raw_index=index,C=sorted(c),D=sorted(d),
        C_labels=[old['labels'][j] for j in sorted(c)],D_labels=[old['labels'][j] for j in sorted(d)],
        both_candidate_top_groups=bool(high[0]),union_fraction=len(c|d)/n,
        tie_key=tie,selected_labels=labels,true_structure_used=False,actual_future_used=False)


def prepare(root,repo,qualification_root):
    root=Path(root);root.mkdir(parents=True,exist_ok=True);path=root/'protocol.json'
    if path.exists():return verify(root)
    repo=Path(repo).resolve();qualification_root=Path(qualification_root).resolve()
    index=repo/'research_experiments/wzzt_dr_selection/evidence/scenario_review_v1/source_inventory.json'
    inventory=read(index)
    panel=sorted([r for r in inventory['rows'] if r['case']=='paper15' and r['bundle'] in (8,9)],key=lambda r:r['job_id'])
    assert len(panel)==8 and all(r['N']==288 and r['condition']=='reference' for r in panel)
    pilot=next(r for r in panel if r['bundle']==8 and r['voltage_sigma']==.001)
    result_file=pilot['existing_result']['path']
    if sha(result_file)!=pilot['existing_result']['catalog_sha256']:raise RuntimeError('past result hash drift')
    old=read(result_file)
    indices,active,receipt=select_historical_pair(old,530013)
    qp=qualification_root/'protocol.json';qr=qualification_root/'qualification.json'
    qprotocol=read(qp);qualified=read(qr)
    qcase=next(r for r in qualified['records'] if r['job_id']==pilot['job_id'])
    bylabel={r['label']:r for r in qcase['nodes']}
    device_notes={str(label):bylabel[label] for label in active}
    conflicts=[label for label in active if bylabel[label]['observed_negative_remainder_cells']>0]
    sources={str(p):sha(p) for p in Path(__file__).parent.glob('*.py')}
    base=repo/'research_experiments/wzzt_dr_selection'
    for rel in ('dr_competition_bridge/physical_bridge.py','dr_competition_bridge/pricing_tables.py',
                'dr_competition_bridge/coded_shapes.py','dr_pool_v2/engine.py','dr_pool_v2/selection.py',
                'dr_pool_v2/observations.py'):
        p=base/rel;sources[str(p)]=sha(p)
    inputs={str(index):sha(index),str(qp):sha(qp),str(qr):sha(qr)}
    for row in panel:
        inputs[row['existing_result']['path']]=row['existing_result']['catalog_sha256']
        for role in ('train','validation'):inputs[row[role]['path']]=row[role]['catalog_sha256']
    cfg=dict(status='frozen_before_coded_LP_or_new_actual_future',created_utc=timestamp(),repo_root=str(repo),
        locked_panel=panel,pilot_job=pilot['job_id'],pilot_row=pilot,representative_indices=indices,
        active_labels=active,site_receipt=receipt,labels=old['labels'],
        source_sha256=sources,input_sha256=inputs,
        templates=['same','opposed','alternating_opposed'],
        policies=['zero','single','rotation','same','opposed','alternating','input_best','wzzt_best'],
        menu_scope='all50 feasible price menus on the same historical pair; no full15-node position optimum',
        planning_endpoint='full480 training/480 validation; early checkpoint is diagnostic only',
        budgets=dict(total_case_wall_seconds=900.,total_real_LP_calls=6000,path_seconds=15.),
        coefficient_domain='planner nonnegative unbounded; actual existing adapter[0,2] is a disclosed difference',
        proposal_scope='historical_representative_union',
        design_interpretation='two raw historical explanations with equal heuristic weights, not topology probabilities or parameter coverage',
        device_contract=qprotocol['device'],source_qualification=dict(protocol=dict(path=str(qp),sha256=sha(qp)),qualification=dict(path=str(qr),sha256=sha(qr))),
        active_device_notes=device_notes,historical_device_conflict_labels=conflicts,
        device_conflict_policy='no resampling: preserve as unsupported/incomplete contract qualification for root review',
        contract=dict(price=[0,1,2],participation=[0,.5,.75],power_kw=.2,reactive_kvar=.07,
                      reduced_energy_kWh=.2,rebound_energy_kWh=.2,window_hours=[19,21],interval_hours=.25,
                      worst_four_event_payment=1.6,capacity_kw=45.),
        no_old_test_or_truth_read=True,no_new_actual_future_read=True,
        panel_progress={r['job_id']:'pilot_pending' if r['job_id']==pilot['job_id'] else 'locked_not_started' for r in panel})
    write(path,cfg)
    return cfg


def verify(root):
    cfg=read(Path(root)/'protocol.json')
    for field in ('source_sha256','input_sha256'):
        for path,digest in cfg[field].items():
            if Path(path).suffix=='.npz':continue  # Full old historical bytes checked at the point of loading.
            if sha(path)!=digest:raise RuntimeError('frozen source/input changed: '+path)
    return cfg


def _choice(rows,seed,maximize=False):
    best=(max if maximize else min)(r['value'] for r in rows)
    eligible=[r for r in rows if abs(r['value']-best)<=1e-12]
    def tie(r):
        token=json.dumps([seed,r['template'],r['small_menu']],separators=(',',':')).encode()
        return r['expected_fee'],hashlib.sha256(token).hexdigest(),r['template'],r['menu_id']
    return min(eligible,key=tie)


def ordinary_tables(history,active,law):
    from coded_model import TEMPLATES
    from coded_replay import future_inputs
    from research_experiments.wzzt_dr_selection.dr_competition_bridge.physical_bridge import forecast_mean
    from research_experiments.wzzt_dr_selection.dr_competition_bridge.pricing_tables import enumerate_menus,enumerate_training_outcomes,payment
    positions=[history['labels'].index(label) for label in active];forecast=forecast_mean(history)
    past=np.vstack([np.column_stack((d['P_terminal'].to_numpy(),d['Q_terminal'].to_numpy())) for d in history['training']+history['validation']])
    scale=np.maximum(np.sqrt(np.mean(past*past,axis=0)),1e-12);scaled=past/scale
    gram_past=scaled.T@scaled+1e-6*np.eye(past.shape[1]);menus=enumerate_menus(2)
    rows=[]
    for kind in TEMPLATES:
        future=future_inputs(forecast,positions,kind);perday={}
        for key,(p,q) in future.items():
            x=np.column_stack((p,q))/scale;perday[key]=x.T@x
        states={}
        for a in range(4):
            for b in range(4):
                sign,value=np.linalg.slogdet(gram_past+perday[a]+perday[b])
                if sign<=0 or not np.isfinite(value):raise RuntimeError('ordinary input Gram unresolved')
                states[a,b]=float(value)
        for i,menu in enumerate(menus):
            value=sum(float(mass)*states[key] for key,mass in enumerate_training_outcomes(menu,law))
            rows.append(dict(template=kind,menu_id=i,small_menu=menu.tolist(),value=value,
                             expected_fee=float(payment(menu)['expected'])))
    return rows


def run(root,*,resume=False):
    from coded_model import TEMPLATES,POLICIES,directions,expand_prices,contract_spec
    from coded_replay import plan_case
    from research_experiments.wzzt_dr_selection.dr_competition_bridge.physical_bridge import load_history
    from research_experiments.wzzt_dr_selection.dr_competition_bridge.pricing_tables import enumerate_menus,payment
    root=Path(root);cfg=verify(root);done=root/'case_plans.json'
    if done.exists():
        old=read(done)
        if resume and old['status']=='complete' and old['protocol_sha256']==sha(root/'protocol.json'):return old
        raise FileExistsError('existing completed/unresolved plan retained; no implicit restart')
    if (root/'run.lock').exists() or (root/'run_checkpoint.json').exists():raise FileExistsError('partial coded run retained')
    write(root/'run.lock',dict(pid=__import__('os').getpid(),protocol_sha256=sha(root/'protocol.json')))
    started=perf_counter();used=0;results={}
    summary=dict(status='running',cases=[],policies=list(POLICIES),protocol_sha256=sha(root/'protocol.json'),
                 locked_panel=[dict(job_id=r['job_id'],status='planning_running' if r['job_id']==cfg['pilot_job'] else 'not_started') for r in cfg['locked_panel']],
                 planning_endpoint='full-day; early checkpoint not optimized',LP_calls=0)
    try:
        if cfg['historical_device_conflict_labels']:
            raise RuntimeError('history reserve conflict at fixed sites; no silent redraw: '+str(cfg['historical_device_conflict_labels']))
        history=load_history(cfg['pilot_row']);write(root/'history_receipt.json',history['receipt'])
        old=read(cfg['pilot_row']['existing_result']['path']);positions=[history['labels'].index(v) for v in cfg['active_labels']]
        for kind in TEMPLATES:
            remaining=cfg['budgets']['total_case_wall_seconds']-(perf_counter()-started)
            calls=cfg['budgets']['total_real_LP_calls']-used
            if remaining<=0 or calls<=0:raise RuntimeError('combined coded planning budget exhausted')
            print(json.dumps(dict(stage='template',template=kind,remaining_seconds=remaining,remaining_LPs=calls)),flush=True)
            result=plan_case(history,old,cfg['representative_indices'],cfg['active_labels'],root/'planning'/kind,
                wall_seconds=remaining,path_seconds=cfg['budgets']['path_seconds'],max_LP_calls=calls,
                tie_seed=530014,proposal_scope=cfg['proposal_scope'],template_kind=kind)
            used+=result['LP_calls'];results[kind]=result
            summary['LP_calls']=used;summary['templates_attempted']=list(results)
            summary['templates_completed']=[k for k,r in results.items() if r['status']=='complete']
            write(root/'run_checkpoint.json',summary)
            if result['status']!='complete':raise RuntimeError('template table unresolved: '+kind+'; '+result.get('exception',''))
        menus=enumerate_menus(2);case=dict(job_id=cfg['pilot_job'],bundle=8,voltage_sigma=.001,
            history_row=cfg['pilot_row'],active_labels=cfg['active_labels'],locks={},site_receipt=cfg['site_receipt'])
        for law in ('local','common'):
            ordinary=ordinary_tables(history,cfg['active_labels'],law)
            write(root/f'ordinary_{law}.json',ordinary)
            inp=_choice(ordinary,530014,True)
            single=_choice([r for r in ordinary if np.count_nonzero(np.any(np.asarray(r['small_menu'])>0,axis=0))<=1],530015,True)
            rotation=_choice([r for r in ordinary if r['small_menu']==[[2,0],[0,2]]],530016,True)
            wr=[]
            for kind in TEMPLATES:
                for row in results[kind]['scores'][law]['records']:
                    wr.append(dict(template=kind,menu_id=row['menu_id'],small_menu=menus[row['menu_id']].tolist(),
                                   value=row['expected_loss'],expected_fee=row['expected_fee']))
            w=_choice(wr,530014)
            choices=dict(zero=dict(template='same',small_menu=[[0,0],[0,0]],value=None),single=single,
                rotation=rotation,same=dict(template='same',small_menu=[[1,1],[1,1]],value=None),
                opposed=dict(template='opposed',small_menu=[[1,1],[1,1]],value=None),
                alternating=dict(template='alternating_opposed',small_menu=[[1,1],[1,1]],value=None),
                input_best=inp,wzzt_best=w)
            plans={name:expand_prices(c['small_menu'],positions,len(history['labels'])).tolist() for name,c in choices.items()}
            shapes={name:directions(c['template'],positions,len(history['labels'])).tolist() for name,c in choices.items()}
            template_map={name:c['template'] for name,c in choices.items()}
            lock=root/'locks'/f"{cfg['pilot_job']}_{law}.json"
            if lock.exists():raise FileExistsError('existing plan lock retained')
            write(lock,dict(status='frozen',created_utc=timestamp(),job_id=cfg['pilot_job'],dependence=law,
                protocol_sha256=sha(root/'protocol.json'),labels=history['labels'],active_labels=cfg['active_labels'],
                plans=plans,directions=shapes,templates=template_map,selection_receipts=choices,
                device_contract=cfg['device_contract'],source_qualification=cfg['source_qualification'],
                contract=cfg['contract'],template_specs={k:contract_spec(k) for k in TEMPLATES},
                scope='all policies have the same three coded contracts and two-site50 price menus; single not global15',
                future_read_before_freeze=False,early_price_optimization_claim=False))
            case['locks'][law]=dict(path=str(lock),sha256=sha(lock))
        summary.update(status='complete',cases=[case],LP_calls=used,elapsed_seconds=perf_counter()-started,
            templates_completed=list(results),template_results={k:str(root/'planning'/k/'plan_result.json') for k in results},
            panel_not_auto_started=True,actual_future_read=False)
        next(r for r in summary['locked_panel'] if r['job_id']==cfg['pilot_job'])['status']='planning_complete'
    except Exception as error:
        summary.update(status='planning_unresolved',exception_type=type(error).__name__,exception=str(error),
            LP_calls=used,elapsed_seconds=perf_counter()-started,templates_attempted=list(results),
            templates_completed=[k for k,r in results.items() if r['status']=='complete'],
            no_zero_fallback=True,actual_future_read=False)
        next(r for r in summary['locked_panel'] if r['job_id']==cfg['pilot_job'])['status']='planning_unresolved'
    finally:
        write(done,summary);write(root/'run_checkpoint.json',summary)
        (root/'run.lock').unlink(missing_ok=True)
    return summary


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','run']);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--repo-root',type=Path,default=Path.cwd());p.add_argument('--qualification-root',type=Path)
    p.add_argument('--resume',action='store_true');a=p.parse_args()
    sys.path.insert(0,str(a.repo_root.resolve()));sys.path.insert(0,str(a.repo_root.resolve()/'rnj_wzzt_core'))
    if a.mode=='prepare':
        if a.qualification_root is None:p.error('--qualification-root required for prepare')
        result=prepare(a.root,a.repo_root,a.qualification_root)
        print(json.dumps({k:result[k] for k in ('status','pilot_job','representative_indices','active_labels','budgets','historical_device_conflict_labels')}))
    else:
        result=run(a.root,resume=a.resume)
        print(json.dumps({k:result.get(k) for k in ('status','LP_calls','elapsed_seconds','templates_completed','exception')}))


if __name__=='__main__':main()
