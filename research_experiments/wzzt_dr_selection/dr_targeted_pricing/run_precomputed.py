"""Targeted failed-case DR: cache actual 16-state impacts, optimize, then test.

No topology search or power-flow implementation is duplicated. fit/finish use
existing run_day_trial wrappers and SharedFitter. All raw artifacts stay private.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import random
import subprocess
import sys
import traceback
import zipfile
from datetime import datetime,timezone
REPO=Path(__file__).resolve().parents[3]
TASK=REPO/'research_experiments/wzzt_dr_selection'

def now():return datetime.now(timezone.utc).isoformat()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ref(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def bound(v):
    if sha(v['path'])!=v['sha256']:raise RuntimeError('input drift '+v['path'])
    return Path(v['path'])
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.partial')
    tmp.write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8');tmp.replace(p)
def setup():
    for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
    sys.path.insert(0,str(REPO))
def configuration(root):
    setup()
    from research_experiments.wzzt_dr_selection.dr_day_modules.run_day_trial import config
    return config(root)

def prepare(root):
    setup();import numpy as np
    from research_experiments.wzzt_dr_selection.dr_pool_v2.adapter import source_hashes
    from research_experiments.wzzt_dr_selection.dr_day_modules.day_schedule import ResourceDay,DayModule,evaluate_schedule,shift_shape
    from research_experiments.wzzt_dr_selection.dr_day_modules.day_physical_worker import PRODUCER_MODULES,LOCAL_SOURCES
    from research_experiments.wzzt_dr_selection.dr_targeted_pricing.optimizer import PAIR_STATES
    root=Path(root).resolve();b=read(root/'baseline_protocol.json');q=read(root/'baseline_prepared.json');e=read(root/'baseline_evaluation.json')
    if (root/'protocol.json').exists():raise FileExistsError('actual trial protocol exists')
    if not e['current_has_any_full_errors'] or not e['current_target_missing']:raise RuntimeError('current baseline is not a target-missing failure')
    if not e['scores']['rnj_bank']['full_errors']:raise RuntimeError('free bank already correct: retain qualification, not a main DR benefit case')
    with bound(q['history_packet']).open('rb') as f:history=pickle.load(f)['history']
    fit=read(bound(read(root/'baseline_ended.json')['result']));labels=fit['terminal_labels'];C=set(b['target_support_labels'])
    def fam(point):return {frozenset(labels[i] for i in s) for s in point['supports']}
    current=fit['models']['multistart_original_rule']['selected_uncontracted'];alternatives=[v for v in fit['selection_candidates'] if frozenset(C) in fam(v)]
    if not alternatives:raise RuntimeError('target absent from historical competition; no witness action silently invented')
    best=min(alternatives,key=lambda v:v['validation_mae']);common=fam(current)&fam(best)
    parents=[s for s in common if C<set(s)];V=min(parents,key=lambda s:(len(s),sorted(s))) if parents else frozenset(labels)
    inside=min(C);outside=min(set(V)-C);pair=[inside,outside];witnesses=sorted(C-{inside});ids=['inside','outside']
    early=shift_shape(range(24,32),range(32,40));late=shift_shape(range(68,76),range(76,84))
    modules=['targeted','rotation'];prices=[(i,j) for i in range(3) for j in range(3) if i+j<=2]
    plans={};plan_lookup={};details={}
    for m in modules:
        plan_lookup[m]={}
        for ci,cj in prices:
            name=f'{m}__c{ci}_{cj}';dto=dict(resources_by_event=[],modules_by_event=[],amplitudes_by_event=[])
            total=0.
            for event in range(4):
                active=event in (0,2)
                resources=[dict(resource_id=key,node_label=int(label),baseline_kw=.2,power_upper_kw=.4,
                    prefix_delta_lower_kwh=-.4,prefix_delta_upper_kwh=.4,available=True,reactive_ratio=.35,
                    incentive_per_kwh=float(price),electricity_price_per_kwh=0.,
                    declaration_source='same explicit synthetic all-day shift resource; not empirical EV') for key,label,price in zip(ids,pair,(ci,cj))]
                profiles={ids[0]:late,ids[1]:-late} if m=='targeted' else {ids[0]:early,ids[1]:late}
                if not active:profiles={k:np.zeros(96) for k in ids}
                md=dict(module_id=m,profiles=profiles,amplitude_upper_kw=.2,description='target witness or simple time rotation')
                chk=evaluate_schedule([ResourceDay(**v) for v in resources],[DayModule(**md)],[.2])
                if not chk['resource_feasible']:raise RuntimeError('predeclared resource domain fails')
                if active:
                    assert all(abs(chk['resources'][key]['reduced_kwh']-.4)<1e-12 for key in ids)
                total+=chk['full_delivery_incentive_payment']
                dto['resources_by_event'].append(resources);dto['modules_by_event'].append([md]);dto['amplitudes_by_event'].append([.2])
            assert total<=1.6+1e-9
            p=root/'plans'/f'{name}.pkl';p.parent.mkdir(exist_ok=True);p.write_bytes(pickle.dumps(dto,protocol=4));plans[name]=ref(p)
            plan_lookup[m][f'{ci},{cj}']=name;details[name]=dict(module=m,prices=[ci,cj],worst_payment=total,energy_per_paid_event_per_resource=.4)
    sources=source_hashes();alg=dict(sources);source_root=Path(r'D:/0-github_workspace/Topo/outputs/wzzt_paper_20261029_full/source')
    files=[Path(__file__),TASK/'dr_targeted_pricing/optimizer.py',TASK/'dr_pool_v2/matrix_runner.py',TASK/'dr_day_modules/run_day_trial.py',TASK/'dr_balanced_design/budget_reference_diagnostic.py']+[REPO/x for x in LOCAL_SOURCES]
    files += [source_root.joinpath(*name.split('.')).with_suffix('.py') for name in PRODUCER_MODULES]
    for p in files:sources[str(p.resolve())]=sha(p)
    c=dict(id='targeted_precomputed_pricing_v1',status='frozen',created_utc=now(),case=b['case'],job_id=b['job_id'],repository=str(REPO),
        source_root=str(source_root),source_metadata=b['source_metadata'],base_kw=b['base_kw'],root_label=b['root_label'],labels=labels,
        voltage_sigma=b['voltage_sigma'],pq_sigma=.005,history_packet=q['history_packet'],baseline_protocol=ref(root/'baseline_protocol.json'),baseline_evaluation=ref(root/'baseline_evaluation.json'),
        target_labels=sorted(C),common_region=sorted(V),resource_pair=pair,passive_witnesses=witnesses,resource_ids=ids,
        targeting_rule='historical selected family versus target-containing validation competitor; lexicographic inside/outside in common parent; failed cohort known before DR',
        plans=plans,plan_lookup=plan_lookup,plan_details=details,modules=modules,price_pairs=[list(v) for v in prices],
        pair_states=[list(v) for v in PAIR_STATES],active_events=[0,2],planning_path=0,test_paths=[100,101,102,103],event_profiles=[0,2,0,2],
        replicate_base=248000,participation_seed=10311133,selection_seed=10311134,workers=2,maximum_unique_fits=80,fit_worker_wall_seconds=3600,
        original_history_rows=[288,288],future_rows=[480,480],hard_four_event_budget=1.6,energy_per_paid_event_kwh=.4,
        participation='independent whole-resource-day Bernoulli at p(0,1,2)=(0,.5,.75); only events0,2 offered; synthetic fixed-task law',
        precomputation='actual full final errors for 2 complete plans x4 training states x4 independent validation states; one planning natural/measurement background',
        pricing_scope='known-model conditional mechanism reference; exact finite Bernoulli integration, one MC natural background, no deployment posterior claim',
        price_independence='for fixed plan and delivered state, prices do not alter measured input; precompute nominal price1 receipt is basis payment only',
        algorithm_source_sha256=alg,source_sha256=sources,lp_method=2,search_seconds=25.,old_zero_controls='day_module_actual_v1 and budget_reference_diagnostic_v1 retained, not rerun')
    write(root/'protocol.json',c)
    with zipfile.ZipFile(root/'source_snapshot_targeted.zip','w',zipfile.ZIP_DEFLATED) as z:
        for i,p in enumerate(sorted(sources)):z.write(p,arcname=f'{i:03d}_'+Path(p).name)
    write(root/'target_prepared.json',dict(status='prepared',created_utc=now(),protocol=ref(root/'protocol.json'),target=sorted(C),pair=pair,witnesses=witnesses,
        history=q['history_packet'],new_AC=0,new_identifier=0,no_test_data_opened=True))
    print(json.dumps(dict(status='prepared',case=b['case'],target=sorted(C),pair=pair,planning_records=32)),flush=True)

def observe(root,phase):
    root,c=configuration(root);import numpy as np
    from research_experiments.wzzt_dr_selection.dr_pool_v2.matrix_runner import observation_identity
    folder=root/phase;folder.mkdir(exist_ok=True)
    if (folder/'observations.json').exists():raise FileExistsError('observations terminal; do not rerun')
    lock=folder/'observations.lock'
    with lock.open('x') as f:f.write(str(os.getpid()))
    rows=[];jobs={};spec=[]
    if phase=='planning':
        for m in c['modules']:
            for t in range(4):
                for v in range(4):spec.append(dict(policy=m,module=m,path=c['planning_path'],train_state=t,validation_state=v,prices=[1,1]))
    else:
        oracle=read(root/'oracle_selected.json')
        if oracle['status']!='frozen' or oracle['protocol_sha256']!=sha(root/'protocol.json'):raise RuntimeError('unfrozen price policy')
        for path in c['test_paths']:
            for name,q in oracle['test_policies'].items():
                rho=[]
                for event in range(4):
                    u=np.random.default_rng(np.random.SeedSequence([c['participation_seed'],path,event])).random(2)
                    pp=np.asarray([0.,.5,.75])[q['prices']]
                    rho.append({key:int(bit) if event in c['active_events'] else 0 for key,bit in zip(c['resource_ids'],u<pp)})
                t=c['pair_states'].index([rho[0][key] for key in c['resource_ids']]);v=c['pair_states'].index([rho[2][key] for key in c['resource_ids']])
                spec.append(dict(policy=name,module=q['module_id'],path=path,train_state=t,validation_state=v,prices=q['prices']))
    with bound(c['history_packet']).open('rb') as f:history=pickle.load(f)['history']
    env=os.environ.copy();env['PYTHONPATH']=str(REPO)
    try:
        for row in spec:
            row=dict(row);m=row['module'];ci,cj=row['prices'];path=row['path'];t,v=row['train_state'],row['validation_state']
            rid=f"{row['policy']}__p{path}__t{t}v{v}";location=folder/'records'/rid;location.mkdir(parents=True,exist_ok=True)
            rho=[{key:0 for key in c['resource_ids']} for _ in range(4)]
            for event,state in ((0,t),(2,v)):rho[event]={key:int(bit) for key,bit in zip(c['resource_ids'],c['pair_states'][state])}
            req=dict(status='frozen',case=c['case'],repo_root=str(REPO),source_root=c['source_root'],source_metadata=c['source_metadata'],labels=c['labels'],
                root_label=c['root_label'],base_kw=c['base_kw'],voltage_sigma=c['voltage_sigma'],phase=phase,record_id=rid,path=path,
                replicate_base=c['replicate_base'],event_profiles=c['event_profiles'],plan_packet=c['plans'][c['plan_lookup'][m][f'{ci},{cj}']],rho_by_event=rho,
                response_mode='exact_active_train_validation_states' if phase=='planning' else 'independent_resource_day_MC',
                protocol=ref(root/'protocol.json'),source_sha256=c['source_sha256'],incentive_budget=1.6)
            if phase=='test':req['oracle']=ref(root/'oracle_selected.json')
            request=location/'request.json';write(request,req)
            row.update(record_id=rid,phase=phase,variant='exact16' if phase=='planning' else 'natural',rho_by_event=rho,status='started',physical_output=str(location/'physical'))
            try:
                cmd=[sys.executable,'-B',str(TASK/'dr_day_modules/day_physical_worker.py'),'--request',str(request),'--expected-sha',sha(request),'--output',str(location/'physical')]
                done=subprocess.run(cmd,cwd=REPO,env=env,capture_output=True,text=True,timeout=600)
                (location/'physical_stdout.txt').write_text(done.stdout,encoding='utf-8');(location/'physical_stderr.txt').write_text(done.stderr,encoding='utf-8')
                receipt=read(location/'physical/events_receipt.json');row.update(physical_status=receipt['status'],physical_receipt=ref(location/'physical/events_receipt.json'))
                if receipt['status']!='complete':raise RuntimeError('physical status '+receipt['status']+': '+str(receipt.get('message')))
                with (location/'physical/packet.pkl').open('rb') as f:packet=pickle.load(f)
                train=history['training']+packet['training'];valid=history['validation']+packet['validation']
                assert [sum(len(d['P_terminal']) for d in seq) for seq in (train,valid)]==[480,480]
                key,identity=observation_identity(train,valid,root_label=c['root_label'],search_seconds=c['search_seconds'],sources=c['algorithm_source_sha256'],lp_method=c['lp_method'])
                row.update(status='observed',fit_key=key,observation_identity=identity,packet=ref(location/'physical/packet.pkl'),actual_payment=receipt['actual_payment'],worst_payment=receipt['worst_payment'],
                    operation_checks=receipt['all_operation_checks_pass'],payment_scope='nominal basis price1' if phase=='planning' else 'actual frozen offer')
                if key not in jobs:jobs[key]=dict(key=key,packet=row['packet'],record_ids=[],first_record_id=rid)
                jobs[key]['record_ids'].append(rid)
            except Exception as error:row.update(status='physical_or_observation_failed',exception_type=type(error).__name__,message=str(error),traceback=traceback.format_exc())
            rows.append(row);write(folder/'observation_progress.json',dict(status='running',rows=rows,planned=len(spec),unique_jobs=len(jobs)))
            print(json.dumps(dict(stage='physical',phase=phase,done=len(rows),planned=len(spec),id=rid,status=row['status'],unique=len(jobs))),flush=True)
        ordered=[jobs[k] for k in sorted(jobs)]
        for i,j in enumerate(ordered):j['worker']=i%c['workers']
        write(folder/'observations.json',dict(status='frozen',protocol_sha256=sha(root/'protocol.json'),rows=rows,jobs=ordered,planned=len(spec),created_utc=now()))
    finally:lock.unlink(missing_ok=True)
    print(json.dumps(dict(status='observations_frozen',phase=phase,records=len(rows),unique=len(jobs))),flush=True)

def fit_worker(root,phase,worker):
    setup()
    from research_experiments.wzzt_dr_selection.dr_day_modules.run_day_trial import fit_worker as existing
    return existing(root,phase,worker)

def finish(root,phase):
    setup()
    from research_experiments.wzzt_dr_selection.dr_day_modules.run_day_trial import finish as existing
    existing(root,phase)
    root,c=configuration(root);data=read(root/phase/'scored.json');C=tuple(c['target_labels']);extra=[]
    for row in data['rows']:
        if row['status']!='complete':extra.append(dict(record_id=row['record_id'],status=row['status']));continue
        result=read(bound(row['result']));labels=result['terminal_labels']
        def has(point):return C in {tuple(sorted(labels[i] for i in s)) for s in point['supports']}
        candidates=result['selection_candidates'];allbest=min(v['validation_mae'] for v in candidates);yes=[v['validation_mae'] for v in candidates if has(v)];no=[v['validation_mae'] for v in candidates if not has(v)]
        model=result['models']['multistart_original_rule'];raw=model['selected_uncontracted']
        extra.append(dict(record_id=row['record_id'],status='complete',target_in_pool=list(C) in result['pool_support_labels'],
            target_ever_in_accepted_path=any(has(p) for s in result['searches'].values() for p in s['path'][1:]),
            best_validation_with_target=min(yes) if yes else None,best_validation_without_target=min(no) if no else None,
            no_target_exclusion_margin=min(no)-1.03*allbest-1e-12 if no else None,
            selected_raw_target=has(raw),selected_final_target=has(model),selected_origin=model['selected_origin'],
            final_errors=row['models']['multistart_original_rule']['full_errors'],raw_errors=row['trace']['pre_contraction_errors']))
    write(root/phase/'target_mechanisms.json',dict(rows=extra,scope='actual candidate generation acceptance validation and final output; no truth used to alter decisions'))

def optimize(root):
    root,c=configuration(root);import numpy as np
    from research_experiments.wzzt_dr_selection.dr_targeted_pricing.optimizer import optimize_price_plan,explicit_loss_intervals
    if (root/'oracle_selected.json').exists():raise FileExistsError('price already frozen')
    scored=read(root/'planning/scored.json');raw=np.full((len(c['modules']),4,4),np.nan)
    if any(r.get('operation_checks') is False or r.get('physical_status')!='complete' for r in scored['rows']):
        write(root/'pricing_blocked.json',dict(status='physical_constraints_unverified_or_failed',no_quote_selected=True));raise RuntimeError('known physical constraints cannot be traded for a bounded topology loss')
    for row in scored['rows']:
        if row['status']=='complete':raw[c['modules'].index(row['module']),row['train_state'],row['validation_state']]=row['models']['multistart_original_rule']['full_errors']
    intervals=explicit_loss_intervals(raw,missing_lower=0.,missing_upper=4*len(c['labels'])-2)
    global_result=optimize_price_plan(intervals['lower'],intervals['upper'],module_ids=c['modules'],seed=c['selection_seed'])
    k=c['modules'].index('rotation');rotation=optimize_price_plan(intervals['lower'][k:k+1],intervals['upper'][k:k+1],module_ids=['rotation'],seed=c['selection_seed']+1)
    write(root/'planning/loss_table.json',dict(modules=c['modules'],states=c['pair_states'],lower=intervals['lower'].tolist(),upper=intervals['upper'].tolist(),missing_cells=intervals['missing_cells'],natural_backgrounds=1))
    write(root/'price_optimization.json',dict(global_result=global_result,rotation_result=rotation))
    if global_result['selected'] is None or rotation['selected'] is None:raise RuntimeError('finite-price optimum unresolved; no deployment quote is frozen')
    policies={'zero':dict(module_id='targeted',prices=[0,0]),'rotation_equal':dict(module_id='rotation',prices=[1,1]),
        'rotation_optimized':{k:rotation['selected'][k] for k in ('module_id','prices')},'optimized':{k:global_result['selected'][k] for k in ('module_id','prices')}}
    write(root/'oracle_selected.json',dict(status='frozen',created_utc=now(),protocol_sha256=sha(root/'protocol.json'),planning_scored_sha256=sha(root/'planning/scored.json'),
        loss_table_sha256=sha(root/'planning/loss_table.json'),price_optimization_sha256=sha(root/'price_optimization.json'),test_policies=policies,
        known_model_offline_reference=True,no_test_data_yet=True,selected=global_result['selected'],rotation_selected=rotation['selected']))
    print(json.dumps(dict(status='prices_frozen',selected=global_result['selected'],rotation_selected=rotation['selected'])),flush=True)

def aggregate(root):
    root,c=configuration(root);oracle=read(root/'oracle_selected.json');p=read(root/'planning/scored.json');t=read(root/'test/scored.json')
    table=[]
    for policy in oracle['test_policies']:
        rows=[r for r in t['rows'] if r['policy']==policy];ok=[r for r in rows if r['status']=='complete'];count=len(rows)
        table.append(dict(policy=policy,**oracle['test_policies'][policy],completed=len(ok),denominator=count,
            full_error_sum=sum(r['models']['multistart_original_rule']['full_errors'] for r in ok),
            mean_full_errors=sum(r['models']['multistart_original_rule']['full_errors'] for r in ok)/count if len(ok)==count else None,
            recoveries=sum(r['models']['multistart_original_rule']['full_recovery'] for r in ok),
            actual_fee_sum=sum(r['actual_payment'] for r in rows) if all(r.get('actual_payment') is not None for r in rows) else None,
            by_path=[dict(path=r['path'],status=r['status'],error=r['models']['multistart_original_rule']['full_errors'] if r['status']=='complete' else None,fee=r.get('actual_payment')) for r in rows]))
    summary=dict(status='completed',protocol_sha256=sha(root/'protocol.json'),baseline_current=ref(root/'baseline_evaluation.json'),
        planning_requests=p['planned'],planning_unique=p['unique_jobs'],test_requests=t['planned'],test_unique=t['unique_jobs'],
        completed_unique=p['unique_complete']+t['unique_complete'],failed_unique=p['unique_failed']+t['unique_failed'],
        LP_solves=p['actual_LP_calls']+t['actual_LP_calls'],oracle=oracle,test=table,
        source='actual frozen complete joint identifier, no hypothetical loss table',
        scope='failed-cohort known-model pricing mechanism; one planning background xexact16 behavior states; four independent test paths',
        no_population_or_full_continuous_space_optimality_claim=True)
    write(root/'summary.json',summary);print(json.dumps({k:summary[k] for k in ('status','planning_requests','planning_unique','test_requests','test_unique','completed_unique','failed_unique','LP_solves','test')}),flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=('prepare','observe','fit','finish','optimize','aggregate'));p.add_argument('--root',required=True);p.add_argument('--phase',choices=('planning','test'));p.add_argument('--worker',type=int)
    a=p.parse_args()
    if a.command=='prepare':prepare(a.root)
    elif a.command=='observe':observe(a.root,a.phase)
    elif a.command=='fit':fit_worker(a.root,a.phase,a.worker)
    elif a.command=='finish':finish(a.root,a.phase)
    elif a.command=='optimize':optimize(a.root)
    else:aggregate(a.root)
if __name__=='__main__':main()
