"""Observed-only B replay on the six remaining saved development histories."""
from pathlib import Path
import sys,os,subprocess,json,time
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[name]='1'
WORK=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(WORK))
from research_experiments.mobile_ev_mle.ac_history_refresh import io,replay


def worker(index):
    source=WORK/'outputs/ev_ac_repair_development_v3/development'/f'history_{index:03d}.json.gz'
    destination=WORK/'outputs/ev_ac_history_refresh_development_v1/development'/source.name
    if destination.exists():raise FileExistsError(destination)
    r=replay.replay_file(source,destination,draws=256,replicates=4,
        progress=lambda stage,details:print(index,stage,details,flush=True))
    print(index,r['status'],r['seconds'],flush=True)


def main():
    pending=[1,2,3,4,5,6];active={};folder=WORK/'outputs/ev_ac_history_refresh_development_v1'
    started=time.monotonic()
    for i in pending:
        if (folder/'development'/f'history_{i:03d}.json.gz').exists():raise FileExistsError(i)
    while pending or active:
        while pending and len(active)<2:
            i=pending.pop(0);log=(folder/f'development_{i:03d}.log').open('xb',buffering=0)
            p=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'worker',str(i)],
                cwd=WORK,stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ),
                creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            active[i]=(p,log)
        for i,(p,log) in list(active.items()):
            if p.poll() is not None:
                log.close();print(i,'exit',p.returncode,flush=True);del active[i]
        status=dict(pid=os.getpid(),pending=pending,workers={str(i):p.pid for i,(p,_) in active.items()},
            seconds=time.monotonic()-started,new_case_action_noise=False)
        (folder/'development_controller.json').write_text(json.dumps(status),encoding='utf-8')
        if pending or active:time.sleep(2)


if __name__=='__main__':
    if len(sys.argv)>1:worker(int(sys.argv[2]))
    else:main()
