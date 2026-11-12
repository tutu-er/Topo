"""Sequential, unshared estimator replays on a predeclared 16-input panel."""
from pathlib import Path
from time import perf_counter
import argparse
import json
import os
import subprocess
import sys
from research_experiments.wzzt_systematic.common import read,write,sha


def worker(payload,method,result_file):
    from research_experiments.wzzt_paper.common import observed
    from research_experiments.wzzt_paper.pipeline import infer
    p=read(payload);train=observed(p['train'],p['N']);validation=observed(p['validation']);start=perf_counter()
    if method in ('NJ','RNJ_native'):
        import numpy as np
        from rnj_wzzt.estimation.multiscenario import fit_projected_sensitivity
        from rnj_wzzt.graph.rooted_neighbor_joining import rooted_neighbor_joining
        from rnj_wzzt.graph.rooted_hierarchy import rooted_clades
        from research_experiments.rnj.rooted_study_baselines import infer_classical_nj
        from research_experiments.rnj.rx75_edge_floor import calibrate_rx75_floor
        from research_experiments.wzzt_retune.engine import arrays,canonical,PoolLP
        from research_experiments.wzzt_fine.pipeline import native_model
        P,Q,Y=arrays(train,'raw');labels=list(train[0].columns)
        import pandas as pd
        R,X,_,_=fit_projected_sensitivity([dict(P_terminal=pd.DataFrame(P,columns=labels),Q_terminal=pd.DataFrame(Q,columns=labels),drop_target=pd.DataFrame(Y,columns=labels))])
        cal=calibrate_rx75_floor(R,X);kernel=.75*R/cal['mu_r']+.25*X/cal['mu_x']
        tree=infer_classical_nj(kernel,kernel,labels,1,collapse_factor=.16) if method=='NJ' else rooted_neighbor_joining(kernel,np.diag(kernel),labels,1,group_tolerance=.24*max(float(np.median(np.diag(kernel))),1e-12))
        positions={v:i for i,v in enumerate(labels)}
        family=canonical([(i,) for i in range(len(labels))]+[tuple(positions[v] for v in s) for s in rooted_clades(tree.edges,1,labels)])
        lp=PoolLP(train,family)
        try:model=native_model(dict(lp.fit(family),native_family=family),train,validation)
        finally:lp.close()
    else:model=infer(train,validation,mode=method)['models'][method]
    elapsed=perf_counter()-start
    original=read(Path(payload).parent/'result.json')['models'][method]
    family=lambda p:{tuple(s) for s in p['supports'] if 2<=len(s)<len(train[0].columns)}
    write(result_file,dict(job_id=p['job_id'],variant=method,estimator_seconds=elapsed,structure_equal_to_shared=family(model)==family(original),
                          worker_code_sha256=sha(Path(__file__)),source='fresh subprocess; no shared estimated matrices, LP, candidate cache or paths',point=model))


def run(output):
    output=Path(output).resolve();folder=output/'timing';folder.mkdir(exist_ok=True);jobs=read(output/'protocol.json')['jobs'];selected=[]
    worker_source=folder/'worker_source.py'
    if not worker_source.exists():worker_source.write_bytes(Path(__file__).read_bytes())
    for job in jobs:
        p=read(output/'jobs'/job/'payload.json')
        if p['panel']=='main' and p['bundle']==0 and p['voltage_sigma'] in (.0002,.001) and p['N'] in (48,288):selected.append(job)
    records=[]
    methods=('NJ','RNJ_native','multistart','ridge_union')
    for index,job in enumerate(selected):
        offset=index%len(methods)
        for method in methods[offset:]+methods[:offset]:
            target=folder/(job+'__'+method+'.json')
            if not target.exists():
                start=perf_counter()
                env=os.environ.copy();env['PYTHONPATH']=str(output/'source')
                for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):env[key]='1'
                proc=subprocess.run([sys.executable,'-B',str(worker_source),'--worker',str(output/'jobs'/job/'payload.json'),method,str(target)],
                    cwd=output/'source',env=env,capture_output=True,text=True,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0),timeout=900)
                (folder/(target.stem+'.log')).write_text(proc.stdout+proc.stderr,encoding='utf-8')
                value=read(target) if target.exists() else dict(job_id=job,variant=method,error=f'exit {proc.returncode}')
                value['process_wall_seconds']=perf_counter()-start;write(target,value)
            records.append(read(target))
        print('TIMING',len(records),'/',len(selected)*4,flush=True)
    write(folder/'summary.json',records)


if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='--worker':worker(*sys.argv[2:])
    else:
        p=argparse.ArgumentParser();p.add_argument('output');a=p.parse_args();run(a.output)
