"""Bounded real AC/full RNJ+WZZT day-module trial; observed-only fits.

prepare -> observations planning -> fit workers -> finish planning -> select
-> observations test -> fit workers -> finish test -> aggregate.
No identifier, physics model, or support search is implemented here.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import pickle
import random
import subprocess
import sys
import time
import traceback
import zipfile

REPO=Path(__file__).resolve().parents[3]
TASK=REPO/'research_experiments/wzzt_dr_selection'
SCRATCH=Path(r'C:/Users/23761/.codex/visualizations/2026/10/25/01a1726e-8ec7-7bc1-9f18-391f9cf79a1a')
OLD=SCRATCH/'wzzt_budget_reference_diagnostic_v1/run_v1'

def now():return datetime.now(timezone.utc).isoformat()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def write(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.partial');tmp.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8');tmp.replace(p)
def ref(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def bound(r):
    if sha(r['path'])!=r['sha256']:raise RuntimeError('input hash drift '+r['path'])
    return Path(r['path'])
def setup():
    for n in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[n]='1'
    sys.path.insert(0,str(REPO))
def config(root):
    root=Path(root).resolve();c=read(root/'protocol.json')
    if c['status']!='frozen':raise RuntimeError('not frozen')
    for p,h in c['source_sha256'].items():
        if sha(p)!=h:raise RuntimeError('source drift '+p)
    for v in c['plans'].values():bound(v)
    bound(c['history_packet'])
    return root,c

def prepare(root):
    setup();import numpy as np
    from research_experiments.wzzt_dr_selection.dr_day_modules.day_schedule import ResourceDay,DayModule,evaluate_schedule,shift_shape
    from research_experiments.wzzt_dr_selection.dr_pool_v2.adapter import source_hashes
    from research_experiments.wzzt_dr_selection.dr_day_modules.day_physical_worker import PRODUCER_MODULES,LOCAL_SOURCES
    root=Path(root).resolve();root.mkdir(parents=True,exist_ok=True)
    if (root/'protocol.json').exists():raise FileExistsError('frozen trial exists')
    old=read(OLD/'protocol.json');case=sorted(old['cases'])[0];casecfg=old['cases'][case]
    with bound(casecfg['history_packet']).open('rb') as f:h=pickle.load(f)['history']
    labels=list(h['labels']);oldplans=read(bound(casecfg['plan_lock']))
    pair=list(oldplans['active_labels_by_row']['opposed'][0]);ids=['anchor','partner']
    assert len(pair)==2 and len(set(pair))==2
    for split in ('training','validation'):assert sum(len(s['P_terminal']) for s in h[split])==288
    early=shift_shape(range(24,28),range(28,32));late=shift_shape(range(76,80),range(80,84))
    resources=[dict(resource_id=key,node_label=int(label),baseline_kw=.2,power_upper_kw=.4,
        prefix_delta_lower_kwh=-.4,prefix_delta_upper_kwh=.4,available=True,
        reactive_ratio=.35,incentive_per_kwh=1.,electricity_price_per_kwh=0.,
        declaration_source='explicit synthetic all-day shift resource; not field EV calibration') for key,label in zip(ids,pair)]
    policies=['zero']+[f'{kind}_{a}' for kind in ('rotation','same','opposed','complement') for a in ('010','020')]
    plans={};details={}
    for policy in policies:
        a=0. if policy=='zero' else int(policy.rsplit('_',1)[1])/100.
        kind=policy if policy=='zero' else policy.rsplit('_',1)[0]
        dto=dict(resources_by_event=[],modules_by_event=[],amplitudes_by_event=[])
        fullpay=0.;red={key:0. for key in ids}
        for event in range(4):
            if kind=='rotation':profiles={key:(early+late if i==event%2 else np.zeros(96)) for i,key in enumerate(ids)}
            elif kind=='same':profiles={key:late for key in ids}
            elif kind=='opposed':profiles={ids[0]:late,ids[1]:-late}
            elif kind=='complement':profiles={ids[0]:early,ids[1]:late}
            else:profiles={key:np.zeros(96) for key in ids}
            module=dict(module_id=policy,profiles=profiles,amplitude_upper_kw=.2,description='predeclared day pattern')
            check=evaluate_schedule([ResourceDay(**v) for v in resources],[DayModule(**module)],[a])
            assert check['resource_feasible']
            fullpay+=check['full_delivery_incentive_payment']
            for key in ids:red[key]+=check['resources'][key]['reduced_kwh']
            dto['resources_by_event'].append(resources);dto['modules_by_event'].append([module]);dto['amplitudes_by_event'].append([a])
        assert fullpay<=1.6+1e-9
        if kind!='zero':assert max(abs(red[key]-4*a) for key in ids)<1e-12
        file=root/'plans'/f'{policy}.pkl';file.parent.mkdir(exist_ok=True);file.write_bytes(pickle.dumps(dto,protocol=4));plans[policy]=ref(file)
        details[policy]=dict(kind=kind,amplitude_kw=a,full_delivery_payment=fullpay,expected_payment=.5*fullpay,
             reduction_kwh_by_resource=red,all_declared_resource_checks_pass=True)
    alg=source_hashes();sources=dict(alg)
    extras=[Path(__file__),TASK/'dr_pool_v2/matrix_runner.py',TASK/'dr_balanced_design/budget_reference_diagnostic.py']+[REPO/p for p in LOCAL_SOURCES]
    source=Path(old['physical_source_root'])
    extras += [source.joinpath(*n.split('.')).with_suffix('.py') for n in PRODUCER_MODULES]
    for f in extras:sources[str(f.resolve())]=sha(f)
    tie=policies.copy();random.Random(1031032).shuffle(tie)
    c=dict(id='day_module_actual_trial_v1',status='frozen',created_utc=now(),repository=str(REPO),case=case,
        source_root=str(source),source_metadata=casecfg['source_metadata'],labels=labels,root_label=casecfg['root_label'],
        base_kw=200.,history_packet=casecfg['history_packet'],old_zero_gain_controls=old['cases'],
        case_selection='lexicographic first of the two already fixed controls; no truth screening or new network',
        resource_pair=pair,resource_ids=ids,plans=plans,policy_details=details,policies=policies,
        planning_paths=[0,1],test_gamma_path=100,test_unintervened_path=101,replicate_base=198000,
        participation_seed=1031031,event_profiles=[0,2,0,2],probability_each_resource_day=.5,
        participation_unit='one independent binary draw per resource per full day (8 draws per four-day path)',
        behavior_scope='synthetic p=.5 at fixed price1; amplitude/shape burden not calibrated; no price optimum claim',
        source_sha256=sources,algorithm_source_sha256=alg,lp_method=2,search_seconds=25.,workers=2,
        maximum_requested_records=48,maximum_unique_fits=48,fit_worker_wall_seconds=3600,
        reference_scope='minimum two-path MC mean final errors, then theoretical expected payment, then frozen tie order',
        tie_order=tie,mechanism_fallback_amplitude='020',truth_read_only_after_fits_frozen=True,
        Gamma_scope='event0 conditional on fixed other six resource-day responses and independent background100',
        independent_test_scope='path101 untouched full eight-bit stochastic response; one path only',
        future_rows=[480,480],original_history_rows=[288,288],hard_four_event_budget=1.6,
        unchanged=['network','natural load producer','voltage .0002','PQ .005','all existing history','RNJ W 3% .16 rules'],
        no_ACN_SOC_or_price_calibration_claim=True)
    write(root/'protocol.json',c)
    with zipfile.ZipFile(root/'source_snapshot.zip','w',zipfile.ZIP_DEFLATED) as z:
        for i,p in enumerate(sorted(sources)):z.write(p,arcname=f'{i:03d}_'+Path(p).name)
    write(root/'preparation_receipt.json',dict(status='prepared',protocol=ref(root/'protocol.json'),source_snapshot=ref(root/'source_snapshot.zip'),
       observations_read='existing full historical packet only',truth_loaded=False,new_AC_calls=0,new_identifier_calls=0,case=case,pair=pair,policies=details))
    print(json.dumps(dict(status='prepared',case=case,pair=pair,menus=len(policies))),flush=True)

def draw(c,path):
    import numpy as np
    return [{key:int(v<.5) for key,v in zip(c['resource_ids'],np.random.default_rng(np.random.SeedSequence([c['participation_seed'],path,e])).random(2))} for e in range(4)]

def observations(root,phase):
    setup();root,c=config(root)
    from research_experiments.wzzt_dr_selection.dr_pool_v2.matrix_runner import observation_identity
    folder=root/phase;folder.mkdir(exist_ok=True)
    if (folder/'observations.json').exists():raise FileExistsError('observations already terminal; do not repeat')
    lock=folder/'observations.lock';lock.write_text(str(os.getpid()))
    policies=c['policies'];spec=[]
    if phase=='planning':
        for path in c['planning_paths']:
            for policy in policies:spec.append((policy,path,'natural',None))
    else:
        oracle=read(root/'oracle_selected.json')
        if oracle['protocol_sha256']!=sha(root/'protocol.json'):raise RuntimeError('oracle protocol mismatch')
        policies=oracle['test_policies']
        for policy in policies:
            for state in ('00','10','01','11'):spec.append((policy,c['test_gamma_path'],state,state))
        for policy in policies:spec.append((policy,c['test_unintervened_path'],'natural',None))
    rows=[];jobs={}
    env=os.environ.copy();env['PYTHONPATH']=str(REPO)
    for policy,path,variant,forced in spec:
        rid=f'{policy}__p{path}__{variant}';location=folder/'records'/rid;location.mkdir(parents=True,exist_ok=True)
        rho=draw(c,path)
        if forced is not None:rho[0]={key:int(bit) for key,bit in zip(c['resource_ids'],forced)}
        req=dict(status='frozen',case=c['case'],repo_root=str(REPO),source_root=c['source_root'],source_metadata=c['source_metadata'],
            labels=c['labels'],root_label=c['root_label'],base_kw=c['base_kw'],phase=phase,record_id=rid,path=path,
            replicate_base=c['replicate_base'],event_profiles=c['event_profiles'],plan_packet=c['plans'][policy],
            rho_by_event=rho,response_mode='conditional_gamma_event0' if forced else 'independent_resource_day_MC',
            protocol=ref(root/'protocol.json'),source_sha256=c['source_sha256'],incentive_budget=1.6)
        if phase=='test':req['oracle']=ref(root/'oracle_selected.json')
        request=location/'request.json';write(request,req)
        row=dict(record_id=rid,phase=phase,policy=policy,path=path,variant=variant,rho_by_event=rho,
            request_sha256=sha(request),status='started',physical_output=str(location/'physical'))
        try:
            cmd=[sys.executable,'-B',str(TASK/'dr_day_modules/day_physical_worker.py'),'--request',str(request),
                 '--expected-sha',sha(request),'--output',str(location/'physical')]
            result=subprocess.run(cmd,cwd=REPO,env=env,capture_output=True,text=True,timeout=600)
            (location/'physical_stdout.txt').write_text(result.stdout,encoding='utf-8');(location/'physical_stderr.txt').write_text(result.stderr,encoding='utf-8')
            receipt=read(location/'physical/events_receipt.json');row.update(physical_status=receipt['status'],physical_receipt=ref(location/'physical/events_receipt.json'))
            if receipt['status']!='complete':row.update(status='physical_failed',error=receipt.get('message'));rows.append(row);continue
            with (location/'physical/packet.pkl').open('rb') as f:packet=pickle.load(f)
            with bound(c['history_packet']).open('rb') as f:history=pickle.load(f)['history']
            train=history['training']+packet['training'];valid=history['validation']+packet['validation']
            if [sum(len(v['P_terminal']) for v in seq) for seq in (train,valid)]!=[480,480]:raise AssertionError('full rows changed')
            key,identity=observation_identity(train,valid,root_label=c['root_label'],search_seconds=c['search_seconds'],sources=c['algorithm_source_sha256'],lp_method=c['lp_method'])
            row.update(status='observed',fit_key=key,observation_identity=identity,packet=ref(location/'physical/packet.pkl'),
                actual_payment=receipt['actual_payment'],worst_payment=receipt['worst_payment'],operation_checks=receipt.get('all_operation_checks_pass'))
            if key not in jobs:jobs[key]=dict(key=key,packet=row['packet'],record_ids=[],first_record_id=rid)
            jobs[key]['record_ids'].append(rid)
        except Exception as error:row.update(status='preparation_failed',exception_type=type(error).__name__,message=str(error),traceback=traceback.format_exc())
        rows.append(row)
        write(folder/'observation_progress.json',dict(status='running',planned=len(spec),rows=rows,unique_jobs=len(jobs)))
        print(json.dumps(dict(stage='physical',phase=phase,done=len(rows),planned=len(spec),id=rid,status=row['status'],unique=len(jobs))),flush=True)
    ordered=[jobs[k] for k in sorted(jobs)]
    for i,job in enumerate(ordered):job['worker']=i%c['workers']
    write(folder/'observations.json',dict(status='frozen',protocol_sha256=sha(root/'protocol.json'),rows=rows,jobs=ordered,planned=len(spec),created_utc=now()))
    lock.unlink(missing_ok=True)
    print(json.dumps(dict(status='observations_frozen',phase=phase,records=len(rows),unique=len(jobs))),flush=True)

def fit_worker(root,phase,worker):
    setup();root,c=config(root)
    from research_experiments.wzzt_dr_selection.dr_pool_v2.matrix_runner import SharedFitter
    from research_experiments.wzzt_dr_selection.dr_pool_v2.adapter import source_hashes
    if source_hashes()!=c['algorithm_source_sha256']:raise RuntimeError('algorithm drift')
    folder=root/phase;ledger=read(folder/'observations.json');jobs=[j for j in ledger['jobs'] if j['worker']==worker]
    ended=folder/f'worker_{worker}_ended.json'
    if ended.exists():raise FileExistsError('completed worker retained')
    lock=folder/f'worker_{worker}.lock'
    with lock.open('x') as f:f.write(str(os.getpid()))
    fitter=SharedFitter(root/'fit_cache',sources=c['algorithm_source_sha256'],max_new_fits=c['maximum_unique_fits'])
    with bound(c['history_packet']).open('rb') as f:history=pickle.load(f)['history']
    rows=[];start=time.perf_counter()
    try:
        for job in jobs:
            out=folder/'fits'/job['key'];row=dict(key=job['key'],record_ids=job['record_ids'],status='started')
            if time.perf_counter()-start>c['fit_worker_wall_seconds']:
                row['status']='unresolved_batch_boundary';rows.append(row);continue
            try:
                with bound(job['packet']).open('rb') as f:packet=pickle.load(f)
                fit=fitter(history['training']+packet['training'],history['validation']+packet['validation'],out,
                    root_label=c['root_label'],search_seconds=c['search_seconds'],lp_method=c['lp_method'])
                row.update(status='complete',result=ref(out/'result.json'),elapsed_seconds=fit['elapsed_seconds'],LP_solves=len(fit['LP_solves']),
                    source='existing fit_observed; no truth/participants supplied')
            except Exception as error:row.update(status='fit_failed',exception_type=type(error).__name__,message=str(error),traceback=traceback.format_exc())
            rows.append(row);write(folder/f'worker_{worker}_progress.json',dict(status='running',rows=rows,planned=len(jobs),elapsed=time.perf_counter()-start))
            print(json.dumps(dict(stage='fit',phase=phase,worker=worker,done=len(rows),planned=len(jobs),status=row['status'],LP=row.get('LP_solves'),seconds=row.get('elapsed_seconds'))),flush=True)
        write(ended,dict(status='ended',rows=rows,planned=len(jobs),elapsed_seconds=time.perf_counter()-start))
    finally:lock.unlink(missing_ok=True)

def trace(fit,truth):
    from research_experiments.wzzt_dr_selection.dr_balanced_design.budget_reference_diagnostic import _family,_error,search_audit
    labels=fit['terminal_labels'];target={tuple(sorted(s)) for s in truth}
    err=lambda p:_error(_family(p,labels),target)['full_errors']
    candidates=fit.get('selection_candidates',[]);best=min((p['validation_mae'] for p in candidates),default=0.)
    return dict(**search_audit(fit),path_points={name:[dict(step=i,full_errors=err(p),objective=p.get('objective')) for i,p in enumerate(s['path'])] for name,s in fit.get('searches',{}).items()},
        selection_candidates=[dict(origin=p.get('origin'),config=p.get('config'),path_index=p.get('path_index'),full_errors=err(p),objective=p['objective'],validation_mae=p['validation_mae'],eligible=p['validation_mae']<=1.03*best+1e-12) for p in candidates],
        final_source=fit['models']['multistart_original_rule'].get('selected_origin'),
        pre_contraction_errors=err(fit['models']['multistart_original_rule']['selected_uncontracted']),
        final_errors=err(fit['models']['multistart_original_rule']))

def finish(root,phase):
    setup();root,c=config(root);folder=root/phase
    if (folder/'scored.json').exists():raise FileExistsError('scored phase retained')
    led=read(folder/'observations.json');jobs={}
    for worker in range(c['workers']):
        for row in read(folder/f'worker_{worker}_ended.json')['rows']:jobs[row['key']]=row
    if set(jobs)!={j['key'] for j in led['jobs']}:raise RuntimeError('worker records incomplete')
    write(folder/'fits_frozen.json',dict(status='frozen',jobs=list(jobs.values()),observation_ledger_sha256=sha(folder/'observations.json'),created_utc=now(),truth_has_not_yet_been_opened_by_controller=True))
    from research_experiments.wzzt_dr_selection.dr_balanced_design.budget_reference_diagnostic import score_fit
    rows=[]
    for original in led['rows']:
        row=dict(original);job=jobs.get(row.get('fit_key'))
        if row['status']!='observed':rows.append(row);continue
        if job['status']!='complete':row.update(status=job['status'],fit_failure=job);rows.append(row);continue
        result=read(bound(job['result']));truth_path=Path(row['physical_output'])/'truth_evaluation.json';truth=read(truth_path)
        row.update(status='complete',result=job['result'],models=score_fit(result,truth['true_supports'],c['root_label']),
            trace=trace(result,truth['true_supports']),truth_evaluation_sha256=sha(truth_path))
        rows.append(row)
    write(folder/'scored.json',dict(status='complete',rows=rows,planned=led['planned'],unique_jobs=len(jobs),
          unique_complete=sum(j['status']=='complete' for j in jobs.values()),unique_failed=sum(j['status']!='complete' for j in jobs.values()),
          actual_LP_calls=sum(j.get('LP_solves',0) for j in jobs.values()),fits_frozen_sha256=sha(folder/'fits_frozen.json')))
    print(json.dumps(dict(status='scored',phase=phase,records=len(rows),complete=sum(r['status']=='complete' for r in rows),unique=len(jobs))),flush=True)

def select(root):
    root,c=config(root);planning=read(root/'planning/scored.json')
    if (root/'oracle_selected.json').exists():raise FileExistsError('oracle frozen already')
    table=[]
    for policy in c['policies']:
        rows=[r for r in planning['rows'] if r['policy']==policy]
        ok=len(rows)==len(c['planning_paths']) and all(r['status']=='complete' for r in rows)
        table.append(dict(policy=policy,status='complete' if ok else 'unresolved',
            MC_mean_final_errors=sum(r['models']['multistart_original_rule']['full_errors'] for r in rows)/len(rows) if ok else None,
            mean_actual_payment=sum(r['actual_payment'] for r in rows)/len(rows) if ok else None,
            theoretical_expected_payment=c['policy_details'][policy]['expected_payment']))
    rank=lambda r:(r['MC_mean_final_errors'],r['theoretical_expected_payment'],c['tie_order'].index(r['policy']))
    ready=all(r['status']=='complete' for r in table)
    chosen=min(table,key=rank)['policy'] if ready else None
    paid=[r for r in table if r['policy']!='zero'];paidchosen=min(paid,key=rank)['policy'] if all(r['status']=='complete' for r in paid) else None
    amp=paidchosen.rsplit('_',1)[1] if paidchosen else c['mechanism_fallback_amplitude']
    testpolicies=['zero','rotation_010','rotation_020']+[f'{kind}_{amp}' for kind in ('same','opposed','complement')]
    oracle=dict(status='frozen',created_utc=now(),protocol_sha256=sha(root/'protocol.json'),planning_scored_sha256=sha(root/'planning/scored.json'),
        formal_selection_status='selected' if ready else 'unresolved',selected_policy=chosen,
        paid_only_mechanism_reference=paidchosen,mechanism_amplitude=amp,mechanism_is_not_replacement_for_zero_selection=True,
        test_policies=testpolicies,table=table,test_future_has_not_yet_been_generated=True,
        scope='two-path MC approximate model-known reference; not full expectation/price-space optimum')
    write(root/'oracle_selected.json',oracle)
    print(json.dumps(dict(status='oracle_frozen',selected=chosen,paid_only=paidchosen,test_policies=testpolicies)),flush=True)

def gamma_interval(rows,model,cap):
    coefficients={'11':1,'10':-1,'01':-1,'00':1};groups={}
    for row in rows:
        key=row.get('fit_key',row['record_id']);entry=groups.setdefault(key,dict(coefficient=0,row=row));entry['coefficient']+=coefficients[row['variant']]
    low=high=0.
    for v in groups.values():
        a=v['coefficient'];row=v['row']
        if not a:continue
        if row['status']=='complete':low+=a*row['models'][model]['full_errors'];high+=a*row['models'][model]['full_errors']
        elif a>0:high+=a*cap
        else:low+=a*cap
    return dict(lower=low,upper=high,value=low if low==high else None,shared_observation_coefficients_cancelled=True)

def aggregate(root):
    root,c=config(root);plan=read(root/'planning/scored.json');test=read(root/'test/scored.json');oracle=read(root/'oracle_selected.json')
    modelnames=['multistart_original_rule','rnj_bank','baseline25','geometric_RNJ_RX75_0.16','geometric_NJ_RX75_0.16']
    panels=[];natural=[];cap=2**len(c['labels'])-1+2*len(c['labels'])-1
    for policy in oracle['test_policies']:
        four=[r for r in test['rows'] if r['policy']==policy and r['path']==c['test_gamma_path']]
        if {r['variant'] for r in four}!={'00','10','01','11'}:raise RuntimeError('missing gamma record')
        panel=dict(policy=policy,complete=sum(r['status']=='complete' for r in four),denominator=4,
            conditioning='one independent background; other six resource-day participation bits fixed',models={})
        for model in modelnames:
            values={r['variant']:r['models'][model]['full_errors'] if r['status']=='complete' else None for r in four}
            item=dict(state_full_errors=values,Gamma=gamma_interval(four,model,cap))
            if all(v is not None for v in values.values()):
                item.update(conditional_independent_mean_errors=sum(values.values())/4.,
                    conditional_Frechet_risk_endpoints=[.5*(values['10']+values['01']),.5*(values['00']+values['11'])])
            panel['models'][model]=item
        panels.append(panel)
        for row in test['rows']:
            if row['policy']==policy and row['path']==c['test_unintervened_path']:
                natural.append({k:row.get(k) for k in ('policy','status','models','actual_payment','worst_payment','trace')})
    allrows=plan['rows']+test['rows'];summary=dict(status='complete',protocol_sha256=sha(root/'protocol.json'),oracle=oracle,
        planning_requests=plan['planned'],test_requests=test['planned'],unique_planning=plan['unique_jobs'],unique_test=test['unique_jobs'],
        complete_requests=sum(r['status']=='complete' for r in allrows),requests=len(allrows),
        unique_completed=plan['unique_complete']+test['unique_complete'],unique_failed=plan['unique_failed']+test['unique_failed'],
        LP_calls_completed_fits=plan['actual_LP_calls']+test['actual_LP_calls'],
        independent_backgrounds=dict(planning=2,conditional_test=1,unintervened_test=1),
        conditional_Gamma_panels=panels,unintervened_test=natural,
        final_error_histogram={str(e):sum(r['status']=='complete' and r['models']['multistart_original_rule']['full_errors']==e for r in allrows) for e in sorted({r['models']['multistart_original_rule']['full_errors'] for r in allrows if r['status']=='complete'})},
        limitations=['n=2 approximate planning reference','one untouched test path','conditional Gamma only event0, not whole four-day correlation','synthetic fixed p=.5 and full-day DR capability, not calibrated EV pricing'])
    write(root/'summary.json',summary)
    print(json.dumps({k:summary[k] for k in ('status','requests','complete_requests','unique_planning','unique_test','unique_completed','unique_failed','LP_calls_completed_fits','final_error_histogram')}),flush=True)

def main():
    a=argparse.ArgumentParser();a.add_argument('command',choices=('prepare','observations','fit','finish','select','aggregate'));a.add_argument('--root',required=True);a.add_argument('--phase',choices=('planning','test'));a.add_argument('--worker',type=int)
    v=a.parse_args()
    if v.command=='prepare':prepare(v.root)
    elif v.command=='observations':observations(v.root,v.phase)
    elif v.command=='fit':fit_worker(v.root,v.phase,v.worker)
    elif v.command=='finish':finish(v.root,v.phase)
    elif v.command=='select':select(v.root)
    else:aggregate(v.root)
if __name__=='__main__':main()
