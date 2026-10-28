from pathlib import Path
import argparse
import os
import subprocess
import sys
import time
from research_experiments.wzzt_systematic.common import read,write,sha


def run(output):
    output=Path(output).resolve();protocol=read(output/"protocol.json");m=read(output/"manifest.json")
    assert all(sha(output/p)==h for p,h in m["local"].items()) and all(sha(Path(p))==h for p,h in m["external"].items())
    pending=[j for j in protocol["jobs"] if not any((output/"jobs"/j/n).exists() for n in ("result.json","error.json","timeout.json"))]
    active={};env=os.environ.copy();env["PYTHONPATH"]=str(output/"source")
    while pending or active:
        while pending and len(active)<protocol["budgets"]["workers"]:
            job=pending.pop(0);d=output/"jobs"/job;log=(d/"worker.log").open("w",encoding="utf-8")
            process=subprocess.Popen([sys.executable,"-B","-m","research_experiments.wzzt_refine.worker",str(d/"payload.json")],
                cwd=output/"source",env=env,stdout=log,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,"CREATE_NO_WINDOW",0))
            active[job]=(process,log,time.monotonic());print("START",job,flush=True)
        for job,(process,log,start) in list(active.items()):
            if process.poll() is None and time.monotonic()-start>protocol["budgets"]["worker_hard_seconds"]:
                process.kill();process.wait();write(output/"jobs"/job/"timeout.json",dict(reason="declared 300s worker cap",seconds=time.monotonic()-start))
            if process.poll() is not None:
                log.close()
                if process.returncode and not (output/"jobs"/job/"error.json").exists():write(output/"jobs"/job/"error.json",dict(error=f"worker exit {process.returncode}"))
                del active[job];print("END",job,process.returncode,flush=True)
        write(output/"progress.json",dict(planned=len(protocol["jobs"]),finished=len(protocol["jobs"])-len(pending)-len(active),active=list(active),pending=len(pending)))
        if active:time.sleep(1)
    write(output/"completion.json",dict(planned=len(protocol["jobs"]),returned=sum((output/"jobs"/j/"result.json").exists() for j in protocol["jobs"])))


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("output");a=p.parse_args();run(a.output)
