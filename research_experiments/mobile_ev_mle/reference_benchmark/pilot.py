"""Development-only parallel cost pilot; never appended to frozen science."""
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import argparse,os
from .study import run,specification
from .io import ROOT,read,write


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'outputs/pilot_v1')
    p.add_argument('--prepared',type=Path,default=ROOT/'outputs/inputs_v1/prepared.json.gz')
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--networks',nargs='+',default=['cigre_residential','paper11','paper15','case33'])
    p.add_argument('--source',default='none');a=p.parse_args()
    path=a.output/'pilot_controller.json'
    if path.exists():raise FileExistsError('pilot already dispatched: '+str(path))
    prepared=read(a.prepared);specs=[specification('development',0,n,a.source,prepared) for n in a.networks]
    record=dict(status='running',pid=os.getpid(),specs=specs,workers=a.workers,results=[],
        scope='development-unfrozen; interface and cost only; excluded from formal denominators')
    write(path,record)
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        futures={pool.submit(run,s,a.output,a.prepared,freeze_policy=False):s for s in specs}
        for future in as_completed(futures):
            try:row=future.result()
            except Exception as exc:row=dict(spec=futures[future],status='worker_exception',exception=repr(exc))
            record['results'].append(row);write(path,record)
    record['status']='complete';write(path,record)


if __name__=='__main__':main()
