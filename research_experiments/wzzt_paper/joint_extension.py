"""Secondary ten-bundle ablation of joint proposals, with a matched repair control."""
from pathlib import Path
import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from research_experiments.wzzt_systematic.common import read,write,sha,matrices,linear_loss


METHODS=('joint_control25','joint_width5','joint_width5_delete')


def worker(file):
    from research_experiments.wzzt_paper.common import observed
    from research_experiments.wzzt_paper.pipeline import guard
    from research_experiments.wzzt_retune.engine import PoolLP,search
    from research_experiments.wzzt_refine.core import scored,select_pre,finalize,two_step
    file=Path(file);p=read(file);parent=Path(p['parent_result'])
    if sha(Path(__file__))!=p['worker_script_sha256'] or sha(parent)!=p['parent_result_sha256']:
        raise ValueError('joint supplement source/parent bytes changed')
    r=read(parent);train=observed(p['train'],p['N']);validation=observed(p['validation'])
    bank=r['base_points']+r['candidates']['ridge_union'];chosen=select_pre(bank)
    models={};paths={};LP={};start=time.perf_counter()
    # Fresh LPs make each proposal policy start without the other policy's cache or basis.
    for policy in ('joint_control25','joint_width5'):
        lp=PoolLP(train,r['augmentation']['pool'],'raw')
        try:
            trajectory=search(lp,chosen['supports'],seconds=25,repair=True) if policy=='joint_control25' else two_step(lp,chosen['supports'],validation,seconds=25,width=5)
            points=[scored(q,validation,policy) for q in trajectory.get('path',trajectory.get('points',[]))]
            models[policy]=finalize(bank+points,train,validation,r['calibration']);paths[policy]=trajectory
            if policy=='joint_width5':
                models['joint_width5_delete'],deletion,accepted=guard(lp,bank+points,train,validation,r['calibration'])
                paths['joint_width5_delete']=deletion
            LP[policy]=lp.solves
        finally:lp.close()
    write(file.parent/'result.json',dict(job_id=p['job_id'],labels=r['labels'],models=models,paths=paths,LP=LP,
        seconds=time.perf_counter()-start,truth_used_by_estimator=False,
        parent_result_sha256=p['parent_result_sha256'],scope='secondary frozen-parent ablation; all ten existing data bundles; no new tuning'))


def run(output):
    output=Path(output).resolve();folder=output/'joint_extension';folder.mkdir(exist_ok=True);selected=[]
    for job in read(output/'protocol.json')['jobs']:
        p=read(output/'jobs'/job/'payload.json')
        if p['panel']=='main' and p['voltage_sigma'] in (.0005,.001) and p['N'] in (48,288):selected.append((job,p))
    script=folder/'worker_source.py'
    if not script.exists():script.write_bytes(Path(__file__).read_bytes())
    protocol=dict(jobs=[j for j,p in selected],inputs=160,model_rows=480,independent_bundles=10,
        source_sha256=sha(script),workers=4,worker_hard_seconds=360,path_seconds=25,width=5,deletion_seconds=15,
        fixed_LP_seconds=15,initial='same frozen ridge pre-filter bank and selected family',
        control='additional single-step repair on a fresh LP; same 25-second proposal budget',
        interpretation='secondary extension after main results, not a new primary hypothesis; unchanged algorithm; no truth tuning')
    if (folder/'protocol.json').exists():
        if read(folder/'protocol.json')!=protocol:raise ValueError('joint extension protocol changed')
    else:write(folder/'protocol.json',protocol)
    for job,p in selected:
        d=folder/job;d.mkdir(exist_ok=True)
        if not (d/'payload.json').exists():
            parent=output/'jobs'/job/'result.json'
            write(d/'payload.json',{**{k:p[k] for k in ('job_id','train','validation','N','case','bundle','voltage_sigma')},
                'parent_result':str(parent),'parent_result_sha256':sha(parent),'worker_script_sha256':sha(script)})
    pending=[j for j,p in selected if not any((folder/j/n).exists() for n in ('result.json','error.json','timeout.json'))]
    active={};env=os.environ.copy();env['PYTHONPATH']=str(output/'source')
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):env[key]='1'
    while pending or active:
        while pending and len(active)<4:
            job=pending.pop(0);d=folder/job;log=(d/'worker.log').open('w',encoding='utf-8')
            proc=subprocess.Popen([sys.executable,'-B',str(script),'--worker',str(d/'payload.json')],cwd=output/'source',env=env,
                stdout=log,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            active[job]=(proc,log,time.monotonic())
        for job,(proc,log,start) in list(active.items()):
            if proc.poll() is None and time.monotonic()-start>360:
                proc.kill();proc.wait();write(folder/job/'timeout.json',dict(reason='360s joint extension worker hard limit'))
            if proc.poll() is not None:
                log.close()
                if proc.returncode and not (folder/job/'error.json').exists():write(folder/job/'error.json',dict(error=f'worker exit {proc.returncode}'))
                del active[job]
        write(folder/'progress.json',dict(planned=160,finished=len(selected)-len(pending)-len(active),active=list(active),pending=len(pending)))
        if active:time.sleep(1)
    write(folder/'completion.json',dict(planned=160,returned=sum((folder/j/'result.json').exists() for j,p in selected)))


def evaluate(output,destination):
    import numpy as np
    import pandas as pd
    from research_experiments.wzzt_paper.common import observed
    from research_experiments.wzzt_paper.statistics import interval
    from research_experiments.rnj.audit_source_ac96_benchmark import physical_truth,clade_metrics
    output=Path(output).resolve();destination=Path(destination);folder=output/'joint_extension';rows=[];saved=[];checks=[];truths={}
    baseline=pd.read_csv(destination/'per_job.csv');baseline=baseline[baseline.variant=='ridge_union'].set_index('job_id')
    for job in read(folder/'protocol.json')['jobs']:
        p=read(output/'jobs'/job/'payload.json');file=folder/job/'result.json';r=read(file) if file.exists() else {}
        if r:
            labels=r['labels'];train=observed(p['train'],p['N']);val=observed(p['validation']);test=observed(p['test'])
            if p['truth_network'] not in truths:truths[p['truth_network']]=physical_truth(p['truth_network'],labels)['clades']
            checks.append(r['truth_used_by_estimator'] is False and r['parent_result_sha256']==sha(output/'jobs'/job/'result.json'))
            checks.extend(s['status']==2 and max(s['max_constraint_violation'],s['max_bound_violation'])<1e-7 for group in r['LP'].values() for s in group)
        for method in METHODS:
            row={k:p[k] for k in ('job_id','case','bundle','N','voltage_sigma')};row.update(variant=method,model_available=bool(r),clade_f1=0.)
            if r:
                point=r['models'][method];family=[tuple(s) for s in point['supports']]
                from research_experiments.wzzt_retune.engine import compatible
                checks.append(all(compatible(family[:i],s) for i,s in enumerate(family)))
                pred={frozenset(labels[i] for i in s) for s in family if 2<=len(s)<len(labels)}
                R,X=matrices(len(labels),point['supports'],point['r'],point['x'])
                row.update(clade_metrics(pred,truths[p['truth_network']],labels),validation_mae=linear_loss(val,R,X)['mae'],
                    test_mae=linear_loss(test,R,X)['mae'],worker_seconds=r['seconds'])
                checks.append(np.isfinite(R).all() and np.isfinite(X).all() and abs(linear_loss(train,R,X)['mae']-point['objective'])<1e-8)
                saved.append(dict(job_id=job,variant=method,labels=labels,point=point,source_result=str(file),source_result_sha256=sha(file)))
            rows.append(row)
    f=pd.DataFrame(rows);comparisons=[]
    refs={'ridge_union':baseline.clade_f1,**{m:f[f.variant==m].set_index('job_id').clade_f1 for m in METHODS}}
    for method,reference in [('joint_control25','ridge_union'),('joint_width5','ridge_union'),('joint_width5','joint_control25'),
            ('joint_width5_delete','ridge_union'),('joint_width5_delete','joint_width5')]:
        d=f[f.variant==method];delta=d.clade_f1-d.job_id.map(refs[reference])
        comparisons.append(dict(variant=method,reference=reference,inputs=len(d),F1=d.clade_f1.mean(),
            **interval(delta.groupby(d.bundle).mean()),wins=int((delta>1e-10).sum()),ties=int((abs(delta)<=1e-10).sum()),losses=int((delta<-1e-10).sum())))
    f.to_csv(destination/'joint_extension_models.csv',index=False);pd.DataFrame(comparisons).to_csv(destination/'joint_extension_comparisons.csv',index=False)
    with (destination/'joint_extension_models.jsonl').open('w',encoding='utf-8') as stream:
        for row in saved:stream.write(json.dumps(row)+'\n')
    write(destination/'joint_extension_verification.json',dict(available_evidence_passed=bool(checks) and all(checks),checks=len(checks),
        planned_inputs=160,returned_inputs=read(folder/'completion.json')['returned'],rows=len(f),available_models=int(f.model_available.sum()),
        independent_blocks=f.bundle.nunique(),scope='secondary extension; descriptive paired intervals'))
    print('joint extension',len(f),'rows','audit',all(checks),flush=True)


if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='--worker':
        try:worker(sys.argv[2])
        except Exception as e:
            write(Path(sys.argv[2]).parent/'error.json',dict(error=str(e),traceback=traceback.format_exc()));raise
    else:
        p=argparse.ArgumentParser();p.add_argument('output');p.add_argument('--evaluate');a=p.parse_args()
        if a.evaluate:evaluate(a.output,a.evaluate)
        else:run(a.output)
