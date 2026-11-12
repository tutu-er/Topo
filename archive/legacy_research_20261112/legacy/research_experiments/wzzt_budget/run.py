from pathlib import Path
import argparse
import os
import subprocess
import sys
import time
from research_experiments.wzzt_systematic.common import read,write,sha


def run(output):
    output=Path(output).resolve();manifest=read(output/"manifest.json");protocol=read(output/"protocol.json")
    assert all(sha(output/p)==h for p,h in manifest["local"].items())
    assert all(sha(Path(p))==h for p,h in manifest["external"].items())
    pending=[j for j in protocol["jobs"] if not any((output/"jobs"/j/n).exists() for n in ("result.json","error.json","timeout.json"))]
    active={};env=os.environ.copy();env["PYTHONPATH"]=str(output/"source")
    while pending or active:
        while pending and len(active)<protocol["budgets"]["workers"]:
            job=pending.pop(0);d=output/"jobs"/job;log=(d/"worker.log").open("w",encoding="utf-8")
            proc=subprocess.Popen([sys.executable,"-B","-m","research_experiments.wzzt_budget.worker",str(d/"payload.json")],
                cwd=output/"source",env=env,stdout=log,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,"CREATE_NO_WINDOW",0))
            active[job]=(proc,log,time.monotonic());print("START",job,flush=True)
        for job,(proc,log,t0) in list(active.items()):
            if proc.poll() is None and time.monotonic()-t0>protocol["budgets"]["worker_hard_seconds"]:
                proc.kill();proc.wait();write(output/"jobs"/job/"timeout.json",dict(reason="declared 300s worker limit",seconds=time.monotonic()-t0))
            if proc.poll() is not None:
                log.close()
                if proc.returncode and not (output/"jobs"/job/"error.json").exists():write(output/"jobs"/job/"error.json",dict(error=f"worker exit {proc.returncode}"))
                del active[job];print("END",job,proc.returncode,flush=True)
        finished=sum(any((output/"jobs"/j/n).exists() for n in ("result.json","error.json","timeout.json")) for j in protocol["jobs"])
        write(output/"progress.json",dict(planned=protocol["planned"],finished=finished,active=list(active),pending=len(pending)))
        if active:time.sleep(1)
    write(output/"completion.json",dict(planned=protocol["planned"],returned=sum((output/"jobs"/j/"result.json").exists() for j in protocol["jobs"])))


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("output");a=p.parse_args();run(a.output)
