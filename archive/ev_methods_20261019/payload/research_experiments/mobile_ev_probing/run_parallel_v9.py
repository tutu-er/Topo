"""Run a frozen v9 protocol in distinct case processes; no protocol changes."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime,timezone
import hashlib

HERE=Path(__file__).resolve().parent


def main():
    parser=argparse.ArgumentParser();parser.add_argument('output');parser.add_argument('--workers',type=int,default=3)
    args=parser.parse_args(); folder=Path(args.output).resolve()
    protocol=json.loads((folder/'protocol.json').read_text(encoding='utf-8'))
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    execution=dict(started=datetime.now(timezone.utc).isoformat(),workers=args.workers,
                   protocol_sha256=hashlib.sha256((folder/'protocol.json').read_bytes()).hexdigest(),cases=[])
    def run(i):
        path=folder/f'worker_{i:02d}.txt'
        with path.open('w',encoding='utf-8') as stream:
            command=[sys.executable,'-u',str(HERE/'integrated_study_v9.py'),'--output',str(folder),'--case-index',str(i)]
            result=subprocess.run(command,env=env,stdout=stream,stderr=subprocess.STDOUT,
                                  creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        return dict(index=i,returncode=result.returncode,log=path.name)
    order=sorted(range(len(protocol['cases'])),key=lambda i:-protocol['cases'][i][0])
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures={pool.submit(run,i):i for i in order}
        for future in as_completed(futures):
            result=future.result();execution['cases'].append(result)
            (folder/'execution.json').write_text(json.dumps(execution,indent=2)+'\n',encoding='utf-8')
            print(json.dumps(result),flush=True)
    execution['completed']=datetime.now(timezone.utc).isoformat()
    execution['success']=all(r['returncode']==0 for r in execution['cases'])
    (folder/'execution.json').write_text(json.dumps(execution,indent=2)+'\n',encoding='utf-8')
    if not execution['success']: raise RuntimeError('case failure; inspect worker logs')
    cases=[]
    for path in sorted(folder.glob('*/result.json')):
        data=json.loads(path.read_text(encoding='utf-8'))
        cases.append(dict(case=data['case'],file=str(path.relative_to(folder)),
                          sha256=hashlib.sha256(path.read_bytes()).hexdigest(),seconds=data['seconds']))
    assert len(cases)==len(protocol['cases'])
    (folder/'results.json').write_text(json.dumps(dict(protocol=protocol,cases=cases),indent=2)+'\n',encoding='utf-8')


if __name__=='__main__':main()
