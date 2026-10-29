from pathlib import Path
import argparse
import os
import subprocess
import sys
import time
from research_experiments.wzzt_systematic.common import read,write,sha


def run(output):
    output=Path(output).resolve();p=read(output/'protocol.json');manifest=read(output/'manifest.json')
    if not all(sha(output/f)==h for f,h in manifest.items()):raise ValueError('frozen bytes changed')
    pending=[j for j in p['jobs'] if not any((output/'jobs'/j/n).exists() for n in ('result.json','error.json','timeout.json'))]
    active={};env=os.environ.copy();env['PYTHONPATH']=str(output/'source')
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):env[key]='1'
    while pending or active:
        while pending and len(active)<p['budgets']['workers']:
            job=pending.pop(0);d=output/'jobs'/job;log=(d/'worker.log').open('w',encoding='utf-8')
            proc=subprocess.Popen([sys.executable,'-B','-m','research_experiments.wzzt_paper.worker',str(d/'payload.json')],cwd=output/'source',env=env,
                                   stdout=log,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            active[job]=(proc,log,time.monotonic())
        for job,(proc,log,start) in list(active.items()):
            if proc.poll() is None and time.monotonic()-start>p['budgets']['worker_hard_seconds']:
                proc.kill();proc.wait();write(output/'jobs'/job/'timeout.json',dict(reason='frozen worker hard limit',seconds=time.monotonic()-start))
            if proc.poll() is not None:
                log.close()
                if proc.returncode and not (output/'jobs'/job/'error.json').exists():write(output/'jobs'/job/'error.json',dict(error=f'worker exit {proc.returncode}'))
                del active[job]
        done=len(p['jobs'])-len(pending)-len(active)
        write(output/'progress.json',dict(planned=len(p['jobs']),finished=done,active=list(active),pending=len(pending)))
        if done%20==0 and done!=locals().get('last',-1):print('FINISHED',done,'/',len(p['jobs']),flush=True);last=done
        if active:time.sleep(1)
    result=dict(planned=len(p['jobs']),returned=sum((output/'jobs'/j/'result.json').exists() for j in p['jobs']))
    write(output/'completion.json',result);print(result,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output');a=p.parse_args();run(a.output)
