from pathlib import Path
import argparse
import os
import subprocess
import sys
import time
from research_experiments.wzzt_systematic.common import read,write,sha


def run(output,only=None):
    output=Path(output).resolve()
    manifest=read(output/"manifest.json")
    assert all(sha(output/p)==h for p,h in manifest["local"].items())
    assert all(sha(Path(p))==h for p,h in manifest["external"].items())
    protocol=read(output/"protocol.json")
    jobs=[j for j in protocol["jobs"] if only is None or any(s in j for s in only)]
    pending=[j for j in jobs if not any((output/"jobs"/j/n).exists() for n in ("result.json","error.json","timeout.json"))]
    env=os.environ.copy();env["PYTHONPATH"]=str(output/"source")
    active={}
    while pending or active:
        while pending and len(active)<4:
            job=pending.pop(0);folder=output/"jobs"/job
            log=(folder/"worker.log").open("w",encoding="utf-8")
            proc=subprocess.Popen([sys.executable,"-B","-m","research_experiments.wzzt_retune.worker",str(folder/"payload.json")],
                cwd=output/"source",env=env,stdout=log,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,"CREATE_NO_WINDOW",0))
            active[job]=(proc,log,time.monotonic());print("START",job,flush=True)
        for job,(proc,log,start) in list(active.items()):
            if proc.poll() is None and time.monotonic()-start>420:
                proc.kill();proc.wait()
                write(output/"jobs"/job/"timeout.json",dict(reason="declared worker wall limit",seconds=time.monotonic()-start))
            if proc.poll() is not None:
                log.close()
                if proc.returncode and not (output/"jobs"/job/"error.json").exists():
                    write(output/"jobs"/job/"error.json",dict(error=f"worker exit {proc.returncode}"))
                del active[job];print("END",job,proc.returncode,flush=True)
        write(output/"progress.json",dict(planned=len(jobs),finished=len(jobs)-len(pending)-len(active),active=list(active),pending=len(pending)))
        if active:time.sleep(1)
    write(output/"completion.json",dict(planned=len(jobs),finished=len(jobs),full_protocol=len(jobs)==80))


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("output");p.add_argument("--only",nargs="+");a=p.parse_args()
    run(a.output,a.only)
