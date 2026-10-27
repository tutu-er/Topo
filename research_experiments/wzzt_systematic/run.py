"""Resumable four-process controller; use only the run's frozen source copy."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

from .common import read, sha, write


def run(output, *, only=None, workers=None):
    output=Path(output).resolve()
    protocol=read(output/"protocol.json")
    manifest=read(output/"frozen_manifest.json")
    for relative, digest in manifest.items():
        if sha(output/relative) != digest:
            raise ValueError(f"frozen input changed: {relative}")
    jobs=[job for job in protocol["jobs"] if only is None or any(word in job for word in only)]
    pending=[job for job in jobs if not any((output/"jobs"/job/name).exists() for name in ("result.json","error.json","hard_timeout.json"))]
    active={}
    environment=os.environ.copy()
    environment["PYTHONPATH"]=str(output/"source")
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        environment[name]="1"
    while pending or active:
        while pending and len(active)<(workers or protocol["workers"]):
            job=pending.pop(0)
            directory=output/"jobs"/job
            log=(directory/"worker.log").open("w",encoding="utf-8")
            process=subprocess.Popen([sys.executable,"-m","research_experiments.wzzt_systematic.worker",str(directory/"payload.json")],
                cwd=output/"source",env=environment,stdout=log,stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess,"CREATE_NO_WINDOW",0))
            active[job]=(process,log,time.monotonic())
            print("START",job,flush=True)
        for job,(process,log,start) in list(active.items()):
            if process.poll() is None and time.monotonic()-start>protocol["solve"]["hard_worker_seconds"]:
                process.kill()
                process.wait()
                write(output/"jobs"/job/"hard_timeout.json", dict(seconds=time.monotonic()-start,
                      reason="predeclared worker safety limit; checkpoints retained"))
            if process.poll() is not None:
                log.close()
                if process.returncode and not any((output/"jobs"/job/name).exists() for name in ("error.json","hard_timeout.json")):
                    write(output/"jobs"/job/"error.json",dict(error=f"worker exit {process.returncode}; see worker.log"))
                print("END",job,"exit",process.returncode,flush=True)
                del active[job]
        finished=sum(any((output/"jobs"/job/name).exists() for name in ("result.json","error.json","hard_timeout.json")) for job in jobs)
        write(output/"progress.json",dict(planned=len(jobs),finished=finished,pending=len(pending),active=list(active)))
        if active:
            time.sleep(1)
    write(output/"completion.json",dict(planned=len(jobs),finished=len(jobs),full_protocol=len(jobs)==len(protocol["jobs"])))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output",type=Path)
    parser.add_argument("--only",nargs="+")
    parser.add_argument("--workers",type=int)
    args=parser.parse_args()
    run(args.output,only=args.only,workers=args.workers)
