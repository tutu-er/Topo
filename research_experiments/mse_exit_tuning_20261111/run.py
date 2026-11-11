"""Freeze and replay saved final-E proposals; never launch structural MIQP."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from time import perf_counter, sleep

ROOT=Path(__file__).resolve().parents[2]
PARENT=ROOT/"outputs/miqp_loss_benchmark_20261111"
TAUS=(0.,.0001,.001,.003,.01)
def read(p): return json.loads(Path(p).read_text(encoding="utf-8"))
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_name(p.name+f".{os.getpid()}.tmp")
    for i in range(100):
        try:
            tmp.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8",newline="\n")
            tmp.replace(p);return
        except PermissionError:
            if i==99:raise
            sleep(.05)

def freeze(output,parent=PARENT):
    output,parent=Path(output).resolve(),Path(parent).resolve()
    if output.exists():raise FileExistsError("Use a new output; do not overwrite evidence")
    old=read(parent/"protocol.json");verified=read(parent/"analysis/verification.json")
    assert verified["passed"] and verified["new_native_verified"]==verified["reused_parent_verified"]==96
    catalog=read(parent/"catalog.json");assert len(catalog)==48
    for n,h in {**old["source_hashes"],**old["source_artifact_hashes"]}.items():
        assert sha(parent/"source"/n)==h,n
    required=("__init__.py","run.py","gate.py","replay_worker.py")
    assert all(Path(__file__).with_name(n).exists() for n in required)
    output.mkdir(parents=True);controllers=output/"controller_source";controllers.mkdir()
    for n in required:shutil.copy2(Path(__file__).with_name(n),controllers/n)
    hashes={n:sha(controllers/n) for n in required}
    write(output/"catalog.json",catalog)
    original={}
    for job in catalog:
        f=parent/"jobs"/job/"T003/result.json";assert f.exists()
        original[job]=sha(f)
        for s in ("train","validation"):assert sha(catalog[job][s])==old["input_hashes"][job][s]
    protocol=dict(created_utc=datetime.now(timezone.utc).isoformat(),parent_output=str(parent),
        parent_protocol_sha256=sha(parent/"protocol.json"),parent_verification_sha256=sha(parent/"analysis/verification.json"),
        parent_archive_sha256=read(ROOT/"docs"/parent.name/"archive_receipt.json")["sha256"],
        jobs=list(catalog),planned=48,source_method="T003",variants=["old_J",*[f"tau_{t:g}" for t in TAUS]],
        tau_grid=list(TAUS),tie_preference=[.001,.003,.0001,.01,0.],selection_tolerance=1e-12,
        source_hashes=old["source_hashes"],source_artifact_hashes=old["source_artifact_hashes"],
        input_hashes=old["input_hashes"],parent_result_sha256=original,
        controller_hashes=hashes,catalog_sha256=sha(output/"catalog.json"),
        alpha=.003,resolution=.24,epsilon_internal=1e-10,gate="raw physical training MSE relative improvement",
        gate_formula="M_old-M_raw > 1e-10/K + tau*M_old",
        candidate_generation_changed=False,forward_or_T_changed=False,MIQP_searches_reused=48,new_MIQP_calls=0,
        planned_views=48*6,new_QP="only missing fixed families and numerical zero closures",
        parameter_selection_metric="mapped_endpoint_validation_mse_ratio",
        endpoint_definition="single E endpoint after .24 contraction/QP, normalized by identically mapped pre-E endpoint",
        final_selection="unchanged strict validation MAE on own path plus singleton baseline",
        tuning="4-fold leave shared seed_index out; train 36 inputs, evaluate remaining 12; no F1 or test in selection",
        all_data_tau="validation-only historical calibration, not fresh generalization evidence",
        truth_test_opening="only after immutable selection.json exists",
        stale_time_scope="old native method cost is reused; replay wall is not MIQP speedup",
        source_protocol_limits=old["limits"],workers=8,positive_MSE_increase_allowance=False)
    write(output/"protocol.json",protocol)
    write(output/"state.json",dict(planned=48,returned=0,failed=0,pending=48,stage="frozen"))
    print("FROZEN 48 saved E proposals; 6 views/input; zero new MIQP calls")

def status(output,protocol):
    returned=failed=0
    for job in protocol["jobs"]:
        d=output/"jobs"/job
        returned+=(d/"result.json").exists()
        failed+=(d/"error.json").exists() and not (d/"result.json").exists()
    return dict(planned=48,returned=returned,failed=failed,pending=48-returned-failed)

def run(output,workers=8):
    output=Path(output).resolve();p=read(output/"protocol.json");parent=Path(p["parent_output"])
    assert 1<=workers<=8
    assert sha(parent/"protocol.json")==p["parent_protocol_sha256"]
    assert sha(parent/"analysis/verification.json")==p["parent_verification_sha256"]
    assert sha(output/"catalog.json")==p["catalog_sha256"]
    for n,h in p["controller_hashes"].items():assert sha(output/"controller_source"/n)==h,n
    for n,h in {**p["source_hashes"],**p["source_artifact_hashes"]}.items():assert sha(parent/"source"/n)==h,n
    queue=[];active={};started=perf_counter()
    for job in p["jobs"]:
        d=output/"jobs"/job
        if (d/"result.json").exists() or (d/"error.json").exists():continue
        if (d/"attempt.json").exists():raise RuntimeError("Audit an unresolved old attempt before restarting")
        queue.append(job)
    lock=output/"replay.lock"
    with lock.open("x") as f:f.write(str(os.getpid()))
    env=os.environ.copy()
    for n in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):env[n]="1"
    env["PYTHONPATH"]=str(parent/"source")+os.pathsep+str(parent/"source/rnj_wzzt_core")
    try:
        while queue or active:
            while queue and len(active)<workers:
                job=queue.pop(0);d=output/"jobs"/job;d.mkdir(parents=True)
                payload=dict(parent_output=str(parent),job=job,method="T003",result=str(d/"result.json"),
                    protocol_file=str(output/"protocol.json"),protocol_sha256=sha(output/"protocol.json"),
                    parent_protocol_sha256=p["parent_protocol_sha256"],source_hashes=p["source_hashes"],
                    gate_file=str(output/"controller_source/gate.py"),gate_sha256=p["controller_hashes"]["gate.py"],
                    worker_sha256=p["controller_hashes"]["replay_worker.py"],controller_hashes=p["controller_hashes"])
                write(d/"payload.json",payload);log=(d/"worker.log").open("w",encoding="utf-8")
                proc=subprocess.Popen([sys.executable,"-B",str(output/"controller_source/replay_worker.py"),str(d/"payload.json")],
                    cwd=parent/"source",env=env,stdout=log,stderr=subprocess.STDOUT,
                    creationflags=getattr(subprocess,"CREATE_NO_WINDOW",0))
                active[job]=(proc,log,d)
                write(d/"attempt.json",dict(state="running",pid=proc.pid,payload_sha256=sha(d/"payload.json"),
                    started_utc=datetime.now(timezone.utc).isoformat()))
            for job,(proc,log,d) in list(active.items()):
                if proc.poll() is None:continue
                log.close();del active[job]
                if not (d/"result.json").exists() and not (d/"error.json").exists():
                    write(d/"error.json",dict(stage="replay",exit_code=proc.returncode))
                a=read(d/"attempt.json")
                write(d/"attempt.json",dict(a,state="finished",exit_code=proc.returncode,finished_utc=datetime.now(timezone.utc).isoformat()))
                print("REPLAY",job,"OK" if (d/"result.json").exists() else "ERROR",flush=True)
            write(output/"state.json",dict(**status(output,p),queued=len(queue),active=list(active),wall_seconds=perf_counter()-started))
            if active:sleep(.5)
        for job,h in p["parent_result_sha256"].items():assert sha(parent/"jobs"/job/"T003/result.json")==h
        write(output/"completion.json",dict(**status(output,p),parent_results_unchanged=True,new_MIQP_calls=0,
            wall_seconds=perf_counter()-started,finished_utc=datetime.now(timezone.utc).isoformat()))
    finally:
        if not active:lock.unlink()

if __name__=="__main__":
    a=argparse.ArgumentParser(description=__doc__);a.add_argument("command",choices=("freeze","run","status"));a.add_argument("output",type=Path)
    a.add_argument("--parent",type=Path,default=PARENT);a.add_argument("--workers",type=int,default=8);x=a.parse_args()
    if x.command=="freeze":freeze(x.output,x.parent)
    elif x.command=="run":run(x.output,x.workers)
    else:print(json.dumps(status(x.output.resolve(),read(x.output/"protocol.json"))))
