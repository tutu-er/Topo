"""Sequential resumable orchestration, with no competing timed batches."""
from __future__ import annotations
import os,sys,subprocess
from pathlib import Path
from time import sleep
from .run import ROOT,read,write


def main(output):
    state=output/'pipeline_state.json'
    while not (output/'evaluation_completion.json').exists():
        write(state,dict(stage='waiting for immutable short search',running=True));sleep(5)
    assert read(output/'evaluation_completion.json')['returned']==168
    env=os.environ.copy();env['PYTHONPATH']=str(ROOT)+os.pathsep+str(ROOT/'rnj_wzzt_core')
    steps=[('short known-warm check',['research_experiments.rnj_small_seed.warm','run'],output/'warm_completion.json'),
        ('short independent verification',['research_experiments.rnj_small_seed.run','analyze'],output/'analysis/verification.json'),
        ('paired 300-second search',['research_experiments.rnj_small_seed.longrun','run'],output/'long_completion.json'),
        ('long independent verification',['research_experiments.rnj_small_seed.longanalyze'],output/'analysis/long_verification.json')]
    for name,args,receipt in steps:
        if receipt.exists():
            r=read(receipt)
            if ('returned' in r and r['returned']!=r['planned']) or ('passed' in r and not r['passed']):raise RuntimeError('incomplete receipt '+str(receipt))
            continue
        write(state,dict(stage=name,running=True));command=[sys.executable,'-B','-m',*args,str(output)]
        result=subprocess.run(command,cwd=ROOT,env=env,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        if result.returncode or not receipt.exists():
            write(state,dict(stage=name,running=False,failed=True,returncode=result.returncode));raise RuntimeError(name+' failed')
        r=read(receipt)
        if 'returned' in r and r['returned']!=r['planned']:
            write(state,dict(stage=name,running=False,failed=True,receipt=r));raise RuntimeError('incomplete search')
    write(state,dict(stage='complete',running=False,passed=True))


if __name__=='__main__':main(Path(sys.argv[1]).resolve())
