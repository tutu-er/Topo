"""Supplementary warm-started free-support MILP baseline on 80 fixed inputs."""
from pathlib import Path
import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from research_experiments.wzzt_systematic.common import read,write,sha,matrices,linear_loss


def worker(file):
    from research_experiments.wzzt_paper.common import observed
    from research_experiments.wzzt_retune.engine import PoolLP,free_search
    from research_experiments.wzzt_refine.core import scored,finalize
    file=Path(file);p=read(file);parent=Path(p['parent_result'])
    if sha(Path(__file__))!=p['worker_script_sha256']:raise ValueError('supplement worker source changed')
    if sha(parent)!=p['parent_result_sha256']:raise ValueError('matched parent record changed')
    reference=read(parent);train=observed(p['train'],p['N']);validation=observed(p['validation'])
    lp=PoolLP(train,reference['pool'],'raw');start=time.perf_counter()
    try:
        trajectory=free_search(lp,file.parent/'steps',seconds=100,per_solve=10)
        points=[scored(q,validation,'free_MILP_path') for q in trajectory['path']]
        models={method:finalize(bank,train,validation,reference['calibration']) for method,bank in
            (('WZZT_free_path',points),('WZZT_free_bank',reference['rnj_bank']+points))}
        write(file.parent/'result.json',dict(job_id=p['job_id'],labels=reference['labels'],models=models,search=trajectory,LP=lp.solves,
            seconds=time.perf_counter()-start,truth_used_by_estimator=False,scope='warm-started unrestricted single-extension MILP; whole path remains greedy',
            reused_training_precomputation='original base pool and calibration; each extension enumerates finite warm starts inside the 100s path budget'))
    finally:lp.close()


def run(output):
    from research_experiments.wzzt_paper.lp_retry import run as retry_LP
    retry_LP(output)
    output=Path(output).resolve();folder=output/'free_domain';folder.mkdir(exist_ok=True);selected=[]
    for job in read(output/'protocol.json')['jobs']:
        p=read(output/'jobs'/job/'payload.json')
        if p['panel']=='main' and p['bundle']<5 and p['voltage_sigma'] in (.0002,.001) and p['N'] in (48,288):selected.append((job,p))
    worker_source=folder/'worker_source.py'
    if not worker_source.exists():worker_source.write_bytes(Path(__file__).read_bytes())
    protocol=dict(jobs=[j for j,p in selected],inputs=80,model_rows=160,independent_bundles=5,path_seconds=100,per_MIP_seconds=10,
        worker_script_sha256=sha(worker_source),
        MIPGap=.01,coefficient_bounds=[0,2],workers=4,solver_threads=1,worker_hard_seconds=360,
        initial_family='all singleton leaves',validation='same 3% simplicity; fixed0.16 final filter',
        purpose='supplementary free-domain comparison; not one of the three primary hypothesis tests',
        boundary='free single-extension domain, finite-pool feasible warm starts, certified or unresolved steps retained; not global whole-tree optimization')
    if (folder/'protocol.json').exists():
        if read(folder/'protocol.json')!=protocol:raise ValueError('supplement protocol changed')
    else:write(folder/'protocol.json',protocol)
    for job,p in selected:
        d=folder/job;d.mkdir(exist_ok=True)
        if not (d/'payload.json').exists():
            parent=output/'jobs'/job/'result.json'
            write(d/'payload.json',{**{k:p[k] for k in ('job_id','train','validation','N','case','bundle','voltage_sigma')},
                'parent_result':str(parent),'parent_result_sha256':sha(parent),'worker_script_sha256':sha(worker_source)})
    pending=[j for j,p in selected if not any((folder/j/n).exists() for n in ('result.json','error.json','timeout.json'))]
    env=os.environ.copy();env['PYTHONPATH']=str(output/'source')
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):env[key]='1'
    active={}
    while pending or active:
        while pending and len(active)<4:
            job=pending.pop(0);d=folder/job;log=(d/'worker.log').open('w',encoding='utf-8')
            proc=subprocess.Popen([sys.executable,'-B',str(worker_source),'--worker',str(d/'payload.json')],cwd=output/'source',env=env,
                stdout=log,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            active[job]=(proc,log,time.monotonic())
        for job,(proc,log,start) in list(active.items()):
            if proc.poll() is None and time.monotonic()-start>360:
                proc.kill();proc.wait();write(folder/job/'timeout.json',dict(reason='360s supplement worker hard limit'))
            if proc.poll() is not None:
                log.close()
                if proc.returncode and not (folder/job/'error.json').exists():write(folder/job/'error.json',dict(error=f'worker exit {proc.returncode}'))
                del active[job]
        done=len(selected)-len(pending)-len(active)
        write(folder/'progress.json',dict(planned=80,finished=done,active=list(active),pending=len(pending)))
        if active:time.sleep(1)
    write(folder/'completion.json',dict(planned=80,returned=sum((folder/j/'result.json').exists() for j,p in selected)))


def evaluate(output,destination):
    import numpy as np
    import pandas as pd
    from research_experiments.wzzt_paper.common import observed
    from research_experiments.rnj.audit_source_ac96_benchmark import physical_truth,clade_metrics
    from research_experiments.wzzt_paper.statistics import interval
    from research_experiments.wzzt_paper.lp_retry import evaluate as evaluate_retry
    evaluate_retry(output,destination)
    output=Path(output).resolve();folder=output/'free_domain';destination=Path(destination);records=[];events=[];truths={};checks=[]
    main=pd.read_csv(destination/'per_job.csv')
    for job in read(folder/'protocol.json')['jobs']:
        d=folder/job;p=read(output/'jobs'/job/'payload.json');file=d/'result.json';result=read(file) if file.exists() else {}
        labels=result.get('labels',[])
        if result:
            if p['truth_network'] not in truths:truths[p['truth_network']]=physical_truth(p['truth_network'],labels)['clades']
            train=observed(p['train'],p['N']);test=observed(p['test']);validation=observed(p['validation'])
            checks.append(result['truth_used_by_estimator'] is False)
            checks.extend(x['status']==2 and max(x['max_constraint_violation'],x['max_bound_violation'])<1e-7 for x in result['LP'])
            for i,a in enumerate(result['search']['attempts']):
                events.append(dict(job_id=job,step=i,raw_status=a['raw_status'],bound=a['bound'],incumbent=a['objective'],
                    selected_upper=a.get('selected_upper'),relative_gap=a['relative_gap'],primal_feasible=a['primal_audit']['feasible'],
                    selected_feasible=a.get('selected_primal_audit',{}).get('feasible'),decision=a.get('selected_bounded_decision',a['bounded_decision'])))
                checks.append(a.get('selected_primal_audit',{}).get('feasible',True))
                if a['bound'] is not None and a.get('selected_upper') is not None:checks.append(a['bound']<=a['selected_upper']+1e-7)
        for method in ('WZZT_free_path','WZZT_free_bank'):
            row={k:p[k] for k in ('job_id','case','bundle','N','voltage_sigma')};row.update(variant=method,model_available=bool(result),clade_f1=0.)
            if result:
                point=result['models'][method];predicted={frozenset(labels[i] for i in s) for s in point['supports'] if 2<=len(s)<len(labels)}
                R,X=matrices(len(labels),point['supports'],point['r'],point['x'])
                row.update(clade_metrics(predicted,truths[p['truth_network']],labels),train_mae=linear_loss(train,R,X)['mae'],
                    validation_mae=linear_loss(validation,R,X)['mae'],test_mae=linear_loss(test,R,X)['mae'],
                    search_stop=result['search']['reason'],search_seconds=result['search']['seconds'],MIP_attempts=len(result['search']['attempts']))
                checks.append(abs(row['train_mae']-point['objective'])<1e-8)
            records.append(row)
    frame=pd.DataFrame(records);comparisons=[]
    for method in ('WZZT_free_path','WZZT_free_bank'):
        d=frame[frame.variant==method]
        for reference in ('WZZT_leaf_path','WZZT_leaf_bank','multistart','ridge_union'):
            ref=main[main.variant==reference].set_index('job_id').clade_f1;delta=d.clade_f1-d.job_id.map(ref)
            comparisons.append(dict(variant=method,reference=reference,inputs=len(d),F1=d.clade_f1.mean(),
                **interval(delta.groupby(d.bundle).mean()),wins=int((delta>1e-10).sum()),ties=int((abs(delta)<=1e-10).sum()),losses=int((delta < -1e-10).sum())))
    frame.to_csv(destination/'free_domain_models.csv',index=False);pd.DataFrame(events).to_csv(destination/'free_domain_MIP_events.csv',index=False)
    pd.DataFrame(comparisons).to_csv(destination/'free_domain_comparisons.csv',index=False)
    write(destination/'free_domain_verification.json',dict(all_passed=all(checks),checks=len(checks),inputs=80,rows=len(frame),
        available_models=int(frame.model_available.sum()),MIP_attempts=len(events),inputs_without_MIP_attempt=sum(r.get('MIP_attempts')==0 for r in records[::2]),
        interpretation='native bounds apply to the bounded single-extension domain; warm start/domain search and greedy path are separated'))
    print('free-domain supplement evaluated',len(frame),'rows',len(events),'MIP attempts','audit',all(checks),flush=True)
    from research_experiments.wzzt_paper.joint_extension import run as run_joint,evaluate as evaluate_joint
    run_joint(output);evaluate_joint(output,destination)


if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='--worker':
        try:worker(sys.argv[2])
        except Exception as e:
            write(Path(sys.argv[2]).parent/'error.json',dict(error=str(e),traceback=traceback.format_exc()));raise
    else:
        p=argparse.ArgumentParser();p.add_argument('output');p.add_argument('--evaluate');a=p.parse_args()
        if a.evaluate:evaluate(a.output,a.evaluate)
        else:run(a.output)
