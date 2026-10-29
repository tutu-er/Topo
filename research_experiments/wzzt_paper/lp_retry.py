"""Retain original LP timeouts and separately retry identical inputs at 60s/LP."""
from pathlib import Path
import json
import os
import subprocess
import sys
import time
import traceback
from research_experiments.wzzt_systematic.common import read,write,sha


def worker(file):
    file=Path(file);p=read(file)
    if sha(Path(__file__))!=p['worker_script_sha256']:raise ValueError('retry worker source changed')
    from research_experiments.wzzt_paper.common import observed,MAIN
    from research_experiments.wzzt_retune import engine
    Base=engine.PoolLP
    class LongerLP(Base):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs);self.model.setParam('TimeLimit',60.)
    engine.PoolLP=LongerLP
    # Import the pipeline only after changing the isolated process's LP factory.
    from research_experiments.wzzt_paper.pipeline import infer
    r=infer(observed(p['train'],p['N']),observed(p['validation']),lambda_ablation=p['lambda_ablation'],
        budget_ablation=p['budget_ablation'],pair_ablation=p['pair_ablation'],
        checkpoint=lambda r:write(file.parent/'checkpoint.json',dict(r,job_id=p['job_id'],fixed_LP_seconds=60)))
    write(file.parent/'result.json',dict(r,job_id=p['job_id'],fixed_LP_seconds=60,
        original_failure_sha256=p['original_failure_sha256'],all_other_parameters_unchanged=True))


def run(output):
    output=Path(output).resolve();folder=output/'lp60_retry';folder.mkdir(exist_ok=True);selected=[]
    for job in read(output/'protocol.json')['jobs']:
        error=output/'jobs'/job/'error.json'
        if error.exists() and 'fixed-family LP status 9' in read(error).get('error',''):selected.append(job)
    script=folder/'worker_source.py'
    if not script.exists():script.write_bytes(Path(__file__).read_bytes())
    protocol=dict(jobs=selected,inputs=len(selected),original_fixed_LP_seconds=15,retry_fixed_LP_seconds=60,
        original_failures_retained=True,workers=4,worker_hard_seconds=900,source_sha256=sha(script),
        scope='only fixed-family LP cap changed; identical train/validation data and path budgets; separate supplement')
    if (folder/'protocol.json').exists():
        if read(folder/'protocol.json')!=protocol:raise ValueError('retry protocol changed')
    else:write(folder/'protocol.json',protocol)
    for job in selected:
        d=folder/job;d.mkdir(exist_ok=True)
        if not (d/'payload.json').exists():
            p=read(output/'jobs'/job/'payload.json');p.update(worker_script_sha256=sha(script),original_failure_sha256=sha(output/'jobs'/job/'error.json'))
            write(d/'payload.json',p)
    pending=[j for j in selected if not any((folder/j/n).exists() for n in ('result.json','error.json','timeout.json'))]
    active={};env=os.environ.copy();env['PYTHONPATH']=str(output/'source')
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):env[key]='1'
    while pending or active:
        while pending and len(active)<4:
            job=pending.pop(0);d=folder/job;log=(d/'worker.log').open('w',encoding='utf-8')
            proc=subprocess.Popen([sys.executable,'-B',str(script),'--worker',str(d/'payload.json')],cwd=output/'source',env=env,
                stdout=log,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            active[job]=(proc,log,time.monotonic())
        for job,(proc,log,start) in list(active.items()):
            if proc.poll() is None and time.monotonic()-start>900:
                proc.kill();proc.wait();write(folder/job/'timeout.json',dict(reason='900s retry worker hard limit'))
            if proc.poll() is not None:
                log.close()
                if proc.returncode and not (folder/job/'error.json').exists():write(folder/job/'error.json',dict(error=f'worker exit {proc.returncode}'))
                del active[job]
        write(folder/'progress.json',dict(planned=len(selected),finished=len(selected)-len(pending)-len(active),active=list(active),pending=len(pending)))
        if active:time.sleep(1)
    write(folder/'completion.json',dict(planned=len(selected),returned=sum((folder/j/'result.json').exists() for j in selected)))


def evaluate(output,destination):
    import pandas as pd
    from research_experiments.wzzt_paper.common import observed,MAIN
    from research_experiments.wzzt_paper.resolution import guard_filter
    from research_experiments.wzzt_systematic.common import matrices,linear_loss
    from research_experiments.rnj.audit_source_ac96_benchmark import physical_truth,clade_metrics
    output=Path(output).resolve();destination=Path(destination);folder=output/'lp60_retry';rows=[];checks=[];saved=[]
    for job in read(folder/'protocol.json')['jobs']:
        p=read(folder/job/'payload.json');file=folder/job/'result.json';r=read(file) if file.exists() else {}
        if not r:
            rows.extend(dict(job_id=job,variant=m,returned=False,clade_f1=0.,fixed_LP_seconds=60) for m in (*MAIN,'ridge_postfilter_guard'));continue
        train=observed(p['train'],p['N']);val=observed(p['validation']);test=observed(p['test']);labels=r['labels'];truth=physical_truth(p['truth_network'],labels)
        longest=max(x['seconds'] for group in r['LP'].values() for x in group)
        checks.extend(x['status']==2 and max(x['max_constraint_violation'],x['max_bound_violation'])<=1e-7 for group in r['LP'].values() for x in group)
        points=dict(r['models']);bank=r['base_points']+r['candidates']['ridge_union']
        points['ridge_postfilter_guard']=guard_filter(bank[r['models']['ridge_union']['selection_index']],r['calibration'],train,val)
        for method,point in points.items():
            pred={frozenset(labels[i] for i in s) for s in point['supports'] if 2<=len(s)<len(labels)}
            R,X=matrices(len(labels),point['supports'],point['r'],point['x'])
            rows.append(dict(job_id=job,variant=method,returned=True,**clade_metrics(pred,truth['clades'],labels),
                validation_mae=linear_loss(val,R,X)['mae'],test_mae=linear_loss(test,R,X)['mae'],
                longest_LP_seconds=longest,worker_seconds=r['seconds'],fixed_LP_seconds=60))
            saved.append(dict(job_id=job,variant=method,labels=labels,point=point,source_result=str(file),source_result_sha256=sha(file)))
            checks.append(abs(linear_loss(train,R,X)['mae']-point['objective'])<1e-8)
    pd.DataFrame(rows).to_csv(destination/'lp60_retry.csv',index=False)
    with (destination/'lp60_models.jsonl').open('w',encoding='utf-8') as stream:
        for row in saved:stream.write(json.dumps(row)+'\n')
    write(destination/'lp60_retry_verification.json',dict(available_evidence_passed=bool(checks) and all(checks),
        planned_inputs=read(folder/'protocol.json')['inputs'],returned_inputs=read(folder/'completion.json')['returned'],
        original_15s_results_unchanged=True))


if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='--worker':
        try:worker(sys.argv[2])
        except Exception as e:
            write(Path(sys.argv[2]).parent/'error.json',dict(error=str(e),traceback=traceback.format_exc()));raise
    else:run(sys.argv[1])
