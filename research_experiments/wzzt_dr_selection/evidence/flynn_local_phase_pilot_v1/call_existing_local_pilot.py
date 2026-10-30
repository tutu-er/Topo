"""Frozen engineering pilot: existing physical worker + unchanged joint fitter.

Fixed prices, not a price optimizer. Truth is read only after all fit attempts.
No identifier, search or physical model is reimplemented here.
"""
from pathlib import Path
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
import pickle
import subprocess
import sys
import time
import traceback

REPO=Path(r'C:\Users\23761\.codex\worktrees\wzzt-dr-selection\Topo')
TASK=REPO/'research_experiments/wzzt_dr_selection'
OLD=Path(r'C:\Users\23761\.codex\visualizations\2026\10\25\01a1726e-8ec7-7bc1-9f18-391f9cf79a1a\wzzt_flynn_local_qualification_v1\run_v1')
PHYSICAL_SOURCE=Path(r'D:\0-github_workspace\Topo\outputs\wzzt_paper_20261029_full\source')
sys.path.insert(0,str(REPO))


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def write(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def lp_ledger(root):
    counts={};unknown=[]
    for folder in sorted((root/'fit_cache').glob('*')):
        if not folder.is_dir():continue
        path=folder/'result.json'
        if not path.exists():path=folder/'checkpoint.json'
        if path.exists() and 'LP_solves' in (data:=read(path)):
            counts[folder.name]=len(data['LP_solves'])
        else:unknown.append(folder.name)
    return dict(known_LP=sum(counts.values()),by_cache_key=counts,unknown_cache_keys=unknown)


def source_bindings():
    from research_experiments.wzzt_dr_selection.dr_pool_v2.adapter import source_hashes
    files=[Path(__file__),TASK/'dr_competition_bridge/physical_bridge.py',
        TASK/'dr_coded_selection/coded_physical.py',TASK/'dr_coded_selection/actual_worker.py',
        TASK/'dr_pool_v2/matrix_runner.py',TASK/'dr_pool_v2/runner.py']
    # Explicit archived code only, never datasets, network tables or diagnostics.
    files+=sorted((PHYSICAL_SOURCE/'rnj_wzzt').rglob('*.py'))
    return {**source_hashes(),**{str(p):sha(p) for p in files}}


def prepare(root):
    import numpy as np
    from research_experiments.wzzt_dr_selection.dr_coded_selection import coded_physical as coded
    from research_experiments.wzzt_dr_selection.dr_competition_bridge import physical_bridge as legacy
    root.mkdir(parents=True,exist_ok=True)
    if (root/'protocol.json').exists():raise FileExistsError('existing pilot protocol retained')
    q=read(OLD/'qualification_result.json');qp=read(OLD/'qualification_protocol.json')
    assert q['status']=='complete' and q['terminal_labels']==list(range(201,217))
    labels=q['terminal_labels'];n=len(labels)
    rng=np.random.default_rng(9562033)
    left=[201,int(rng.choice(q['local_roles']['left']['qualified_counterparts']))]
    right=[209,int(rng.choice(q['local_roles']['right']['qualified_counterparts']))]
    orientations=np.random.default_rng(9562032).choice([-1,1],4).astype(int).tolist()
    rowspec={'zero':[left,right], 'single_left':[left,left], 'single_right':[right,right],
        'rotation':[left,right], 'same_local':[left,right], 'opposed_local':[left,right],
        'repeat_left':[left,left], 'repeat_right':[right,right]}
    plans={};directions={};templates={}
    for name,pairs in rowspec.items():
        prices=np.zeros((2,n),int);d=np.zeros((2,n),int)
        for e,pair in enumerate(pairs):
            positions=[labels.index(v) for v in pair]
            if name=='zero':continue
            if name in ('single_left','single_right','rotation'):
                prices[e,positions[0]]=2;d[e,positions[0]]=1
            else:
                prices[e,positions]=1;d[e,positions]=[1,1] if name=='same_local' else [1,-1]
        plans[name]=prices.tolist();directions[name]=d.tolist()
        templates[name]='same' if name in ('zero','single_left','single_right','rotation','same_local') else 'opposed'
        legacy.validate_plan(prices,n)
    lock=dict(status='frozen',case='flynn16',base_kw=200.,root_label=1,
        source_metadata=qp['source_metadata'],labels=labels,plans=plans,directions=directions,
        templates=templates,active_labels_by_row=rowspec,orientations_by_event=orientations,
        device_contract=dict(baseline_kw=.2,baseline_kvar=.07,upper_kw=.4,start_hour=19,end_hour=21,
            initial_energy_kWh=0.,final_energy_kWh=.4,efficiency=1.,non_V2G=True),
        design_scope='randomized qualified partner tie; fixed-price engineering phase pilot, not optimized price',
        partner_seed=9562033,orientation_seed=9562032)
    write(root/'plans.json',lock)
    locksha=sha(root/'plans.json')
    for name in plans:
        coded._lock(root/'plans.json',locksha,name,legacy)
    legacy.frozen_source_metadata(lock,'flynn16')
    protocol=dict(id='flynn_local_phase_pilot_v1',created_utc=datetime.now(timezone.utc).isoformat(),
        source_sha256=source_bindings(),plan_sha256=locksha,history_packet_sha256=sha(OLD/'history_and_forecast.pkl'),
        qualification_sha256=sha(OLD/'qualification_result.json'),qualification_protocol_sha256=sha(OLD/'qualification_protocol.json'),
        old_baseline_sha256=sha(OLD/'baseline_request/result.json'),policies=list(plans),
        left_pair=left,right_pair=right,orientations_by_event=orientations,
        case='flynn16',root_label=1,voltage_sigma=.0002,pq_sigma=.005,dependence='local',
        path=0,replicate_base=52000,participation_seed=9562031,event_profiles=[0,2,0,2],
        training_rows=480,validation_rows=480,max_new_fits=8,per_path_seconds=25.,
        wall_seconds=900,wall_gate_scope='pre-policy gate; current original fit or worker may overrun; report actual elapsed',
        worker_timeout_seconds=120,LP_stop_gate_between_fits=4000,
        LP_gate_scope='checked between completed original fits; current fit can overrun gate',
        planned_policy_mappings=8,planned_physical_event_records=32,planned_AC_timepoints=3072,
        independent_natural_paths=1,price_optimization=False,
        truth_access='only after all eight observed-only fit attempts are frozen',
        zero_and_weak_truth='exact zero merged; every positive true block retained',
        actual_effectiveness='paired final full errors, payment, origins, raw vs final; no recovery probability estimate',
        stage='frozen_before_graph_or_actual_future')
    write(root/'protocol.json',protocol)
    print(json.dumps(dict(status='frozen',left=left,right=right,orientations=orientations,
        policies=protocol['policies'],protocol_sha256=sha(root/'protocol.json'))))


def full_errors(metrics):
    return len(metrics['missing_effective_supports'])+len(metrics['extra_effective_supports'])


def run(root):
    from research_experiments.wzzt_dr_selection.dr_pool_v2.adapter import source_hashes
    from research_experiments.wzzt_dr_selection.dr_pool_v2.matrix_runner import SharedFitter
    from research_experiments.wzzt_dr_selection.dr_pool_v2.runner import evaluate_fit
    p=read(root/'protocol.json')
    assert p['source_sha256']==source_bindings(),'source drift'
    assert p['plan_sha256']==sha(root/'plans.json')
    assert p['history_packet_sha256']==sha(OLD/'history_and_forecast.pkl')
    assert p['old_baseline_sha256']==sha(OLD/'baseline_request/result.json')
    assert p['qualification_sha256']==sha(OLD/'qualification_result.json')
    assert p['qualification_protocol_sha256']==sha(OLD/'qualification_protocol.json')
    if (root/'ended.json').exists() or (root/'pilot.lock').exists():raise FileExistsError('prior pilot retained')
    write(root/'pilot.lock',dict(pid=os.getpid(),protocol_sha256=sha(root/'protocol.json')))
    start=time.perf_counter();rows=[];lp_count=0;ledger=dict(known_LP=0,by_cache_key={},unknown_cache_keys=[])
    with (OLD/'history_and_forecast.pkl').open('rb') as stream:history=pickle.load(stream)['history']
    fitter=SharedFitter(root/'fit_cache',sources=source_hashes(),max_new_fits=8)
    try:
        for policy in p['policies']:
            row=dict(policy=policy,status='not_started',physical_status=None,fit_status=None,
                committed_payment=None,actual_payment=None,unknown_payment_events=4)
            rows.append(row)
            if time.perf_counter()-start>=p['wall_seconds'] or lp_count>=p['LP_stop_gate_between_fits'] or ledger['unknown_cache_keys']:
                row['status']='unresolved_batch_boundary';continue
            folder=root/policy;folder.mkdir()
            worker=folder/'physical'
            command=[sys.executable,str(TASK/'dr_coded_selection/actual_worker.py'),
                '--plan-lock',str(root/'plans.json'),'--expected-lock-sha',p['plan_sha256'],
                '--policy',policy,'--path','0','--dependence',p['dependence'],'--sigma',str(p['voltage_sigma']),
                '--case',p['case'],'--output',str(worker),'--repo-root',str(REPO),'--source-root',str(PHYSICAL_SOURCE),
                '--replicate-base',str(p['replicate_base']),'--participation-seed',str(p['participation_seed']),
                '--event-profiles',*[str(v) for v in p['event_profiles']]]
            try:
                result=subprocess.run(command,cwd=REPO,capture_output=True,text=True,
                    encoding='utf-8',errors='replace',timeout=p['worker_timeout_seconds'])
                (folder/'worker_stdout.txt').write_text(result.stdout,encoding='utf-8')
                (folder/'worker_stderr.txt').write_text(result.stderr,encoding='utf-8')
                row['worker_return_code']=result.returncode
                receipt=read(worker/'events_receipt.json')
                row.update(physical_status=receipt['status'],committed_payment=receipt.get('committed_payment'),
                    actual_payment=receipt.get('actual_payment'),physical_events_complete=receipt.get('physical_events_complete'),
                    unknown_payment_events=receipt.get('unknown_payment_events',4))
                if receipt['status']!='complete' or result.returncode!=0:
                    row['status']='physical_failed';continue
                with (worker/'packet.pkl').open('rb') as stream:packet=pickle.load(stream)
                assert packet['metadata']['observed_only'] is True
                fit=fitter(history['training']+packet['training'],history['validation']+packet['validation'],
                    folder/'fit',root_label=1,search_seconds=25.)
                assert fit['training_rows']==fit['validation_rows']==480
                new_lp=len(fit.get('LP_solves',[])) if not fit['matrix_cache']['reused'] else 0
                row.update(status='fit_complete',fit_status=fit['status'],new_LP=new_lp,
                    result_sha256=sha(folder/'fit/result.json'),result_path=str(folder/'fit/result.json'),
                    cache_reused=fit['matrix_cache']['reused'],search_stops=fit['search_stop_statuses'])
            except Exception as exc:
                row.update(status='failed',exception_type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc())
                receipt_path=worker/'events_receipt.json'
                if receipt_path.exists():
                    receipt=read(receipt_path)
                    row.update(physical_status=receipt.get('status'),committed_payment=receipt.get('committed_payment'),
                        actual_payment=receipt.get('actual_payment'),unknown_payment_events=receipt.get('unknown_payment_events',4))
            finally:
                ledger=lp_ledger(root);lp_count=ledger['known_LP']
                write(root/'progress.json',dict(rows=rows,LP_ledger=ledger,elapsed_seconds=time.perf_counter()-start))
        # No true topology/evaluator file is opened until this immutable fit list.
        write(root/'fits_frozen.json',dict(rows=rows,protocol_sha256=sha(root/'protocol.json'),
            frozen_utc=datetime.now(timezone.utc).isoformat()))
        truth=None;labels=history['labels']
        for row in rows:
            if row['status']!='fit_complete':continue
            truth_path=root/row['policy']/'physical/truth_evaluation.json'
            receipt=read(root/row['policy']/'physical/events_receipt.json')
            assert sha(truth_path)==receipt['private_files']['truth_evaluation.json']['sha256']
            assert sha(row['result_path'])==row['result_sha256']
            current=read(truth_path)['true_supports']
            if truth is None:truth=current
            assert {tuple(v) for v in current}=={tuple(v) for v in truth}
            fit=read(row['result_path'])
            evaluation=evaluate_fit(fit,truth,labels)
            write(root/row['policy']/'evaluation.json',evaluation)
            row['evaluation_sha256']=sha(root/row['policy']/'evaluation.json')
            row['models']={name:dict(full_errors=full_errors(m['final']),
                complete=m['final']['complete_effective_family'],origin=m['selected_origin'],
                before_contraction_errors=full_errors(m['before_final_contraction']) if m['before_final_contraction'] else None)
                for name,m in evaluation['models'].items()}
            row['status']='evaluated'
        historical=None
        if truth is not None:
            assert p['old_baseline_sha256']==sha(OLD/'baseline_request/result.json')
            old=evaluate_fit(read(OLD/'baseline_request/result.json'),truth,labels)
            historical={name:dict(full_errors=full_errors(m['final']),complete=m['final']['complete_effective_family'])
                for name,m in old['models'].items()}
        output=dict(status='bounded_complete',protocol_sha256=sha(root/'protocol.json'),
            planned=8,rows=rows,evaluated=sum(v['status']=='evaluated' for v in rows),
            distinct_new_fits=fitter.new_count,reused_fits=fitter.reused_count,new_LP=lp_count,
            LP_ledger=ledger,elapsed_over_wall_gate=max(0.,time.perf_counter()-start-p['wall_seconds']),
            historical_stop=historical,elapsed_seconds=time.perf_counter()-start,
            independent_natural_paths=1,price_optimization=False,unresolved_retained=True)
        write(root/'ended.json',output)
        print(json.dumps({k:output[k] for k in ('status','planned','evaluated','distinct_new_fits','reused_fits','new_LP','historical_stop','elapsed_seconds')}))
        for row in rows:print(json.dumps({k:row.get(k) for k in ('policy','status','actual_payment','models')}))
    finally:
        (root/'pilot.lock').unlink(missing_ok=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['prepare','run']);parser.add_argument('--root',type=Path,required=True)
    a=parser.parse_args();(prepare if a.mode=='prepare' else run)(a.root.resolve())
