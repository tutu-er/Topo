"""Rerun only program errors, retaining every original attempt unchanged."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from .common import read,sha,write


def repair(output):
    output=Path(output).resolve()
    if not read(output/"completion.json")["full_protocol"]:
        raise ValueError("finish all first attempts before repairing program errors")
    source=output/"repair_source"
    if not source.exists():
        shutil.copytree(output/"source",source,ignore=shutil.ignore_patterns("__pycache__","*.pyc"))
        for name in ("worker.py","common.py"):
            shutil.copy2(Path(__file__).with_name(name),source/"research_experiments/wzzt_systematic"/name)
        write(output/"repair_source_manifest.json",{p.relative_to(source).as_posix():sha(p) for p in source.rglob("*.py")})
    manifest=read(output/"repair_source_manifest.json")
    if any(sha(source/p)!=h for p,h in manifest.items()):
        raise ValueError("repair source changed after freezing")
    jobs=[p.parent.name for p in sorted((output/"jobs").glob("*/error.json"))]
    pending=[]
    for job in jobs:
        target=output/"repairs"/job
        target.mkdir(parents=True,exist_ok=True)
        if not (target/"payload.json").exists():
            shutil.copy2(output/"jobs"/job/"payload.json",target/"payload.json")
        if not any((target/n).exists() for n in ("result.json","error.json","hard_timeout.json")):
            pending.append(job)
    write(output/"repair_protocol.json",dict(jobs=jobs,scope="all program-error jobs only; no quality-based or incomplete-solver reruns",
        source_manifest="repair_source_manifest.json",changes=["separate solver lower and coefficient bounds in log",
        "accept valid decoded L1 below epigraph upper bound; use physical L1 as subsequent current loss",
        "verify zero pruning against physical candidate loss","canonicalize cycle detection"],
        parameters_and_training_data_changed=False))
    active={}
    environment=os.environ.copy()
    environment["PYTHONPATH"]=str(source)
    while pending or active:
        while pending and len(active)<4:
            job=pending.pop(0)
            directory=output/"repairs"/job
            log=(directory/"worker.log").open("w",encoding="utf-8")
            proc=subprocess.Popen([sys.executable,"-B","-m","research_experiments.wzzt_systematic.worker",str(directory/"payload.json")],
                cwd=source,env=environment,stdout=log,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,"CREATE_NO_WINDOW",0))
            active[job]=(proc,log,time.monotonic())
            print("REPAIR START",job,flush=True)
        for job,(proc,log,start) in list(active.items()):
            if proc.poll() is None and time.monotonic()-start>600:
                proc.kill()
                proc.wait()
                write(output/"repairs"/job/"hard_timeout.json",dict(seconds=time.monotonic()-start))
            if proc.poll() is not None:
                log.close()
                print("REPAIR END",job,proc.returncode,flush=True)
                del active[job]
        if active:
            time.sleep(1)
    write(output/"repair_completion.json",dict(planned=len(jobs),returned=sum((output/"repairs"/j/"result.json").exists() for j in jobs)))


if __name__=="__main__":
    repair(Path(sys.argv[1]))
