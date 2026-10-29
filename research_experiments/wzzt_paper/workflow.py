"""Resume-safe analysis phases after the already running estimator batch."""
from pathlib import Path
from datetime import datetime,timezone
import argparse
import os
import subprocess
import sys
import time
from research_experiments.wzzt_systematic.common import read,write,sha

ROOT=Path(__file__).resolve().parents[2]


def run(output,destination,start=None):
    output=Path(output).resolve();destination=Path(destination).resolve();folder=output/'analysis_logs';folder.mkdir(exist_ok=True)
    phases=[('metrics_structural','evaluate',[str(output),str(destination),'--skip-ac']),
        ('statistics_initial','statistics',[str(destination)]),('audit','audit',[str(output),str(destination)]),
        ('resolution','resolution',[str(output),str(destination)]),('free_domain','free_domain',[str(output)]),
        ('metrics_AC','evaluate',[str(output),str(destination)]),('statistics_final','statistics',[str(destination)]),
        ('free_evaluation','free_domain',[str(output),'--evaluate',str(destination)]),
        ('timing','timing',[str(output)]),('report','report',[str(output),str(destination)]),('verify','verify',[str(output),str(destination)])]
    names=[x[0] for x in phases]
    if start is not None and start not in names:raise ValueError('unknown phase')
    state=read(output/'workflow_state.json') if (output/'workflow_state.json').exists() else dict(history=[],phase='waiting_for_primary')
    write(output/'workflow_state.json',state)
    while not (output/'completion.json').exists():time.sleep(10)
    env=os.environ.copy()
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):env[key]='1'
    for index,(name,module,args) in enumerate(phases):
        if start is not None and index<names.index(start):continue
        if start is None and any(h['phase']==name and h.get('exit_code')==0 for h in state['history']):continue
        attempt=1+sum(h['phase']==name for h in state['history']);log=folder/f'{name}_attempt{attempt}.log'
        record=dict(phase=name,attempt=attempt,started_utc=datetime.now(timezone.utc).isoformat(),
            code_sha256=sha(Path(__file__).parent/(module+'.py')),log=str(log))
        state.update(phase=name,status='running');state['history'].append(record);write(output/'workflow_state.json',state)
        print('PHASE START',name,'attempt',attempt,flush=True)
        with log.open('w',encoding='utf-8') as stream:
            process=subprocess.run([sys.executable,'-B','-m','research_experiments.wzzt_paper.'+module,*args],cwd=ROOT,env=env,
                stdout=stream,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        record.update(exit_code=process.returncode,ended_utc=datetime.now(timezone.utc).isoformat());state['status']='failed' if process.returncode else 'phase_complete'
        write(output/'workflow_state.json',state);print('PHASE END',name,process.returncode,flush=True)
        if process.returncode:raise RuntimeError('phase failed: '+name+'; see '+str(log))
    state.update(phase='complete',status='complete');write(output/'workflow_state.json',state)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output');p.add_argument('destination');p.add_argument('--from-phase');a=p.parse_args();run(a.output,a.destination,a.from_phase)
