"""Run only never-started panel forecasts and save every returned action.

The original protocol, 85-source closure, completed forecasts and interrupted
forecasts are retained. This stage does not write the main completion marker.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import ctypes
import json
import os
from pathlib import Path
import shutil
from time import perf_counter
import traceback

import numpy as np

from ... import io
from . import run as original

STEM='resume01_unstarted'
ORIGINAL_PLAN=original.solver.plan


def checkpoint_plan(*args, **kwargs):
    """Delegate unchanged arguments, then persist the returned point atomically."""
    directory=kwargs.get('record_directory')
    if directory is None:
        raise ValueError('Continuation requires an explicit diagnostics directory')
    directory=Path(directory)
    receipt,point=ORIGINAL_PLAN(*args,**kwargs)
    tick=perf_counter()
    directory.mkdir(parents=True,exist_ok=True)
    if point is not None:
        destination=directory/'action_checkpoint.npz'
        temporary=directory/'action_checkpoint.npz.tmp'
        if destination.exists():raise FileExistsError('Retain existing action checkpoint: '+str(destination))
        with temporary.open('xb') as stream:
            np.savez_compressed(stream,u=np.asarray(point))
            stream.flush();os.fsync(stream.fileno())
        temporary.replace(destination)
        receipt['action_checkpoint']=io.ref(destination)
        receipt['action_checkpoint_array_sha256']=io.contract_hash(np.asarray(point))
        receipt['action_checkpoint_scope']='Exact returned action; original caller still performs resource/evidence acceptance'
    else:
        destination=directory/'action_checkpoint_none.json'
        if destination.exists():raise FileExistsError('Retain existing no-point marker: '+str(destination))
        io.write(destination,dict(status='no_returned_point',mode=kwargs.get('mode'),
            solver_status=receipt.get('status'),feasible=receipt.get('feasible'),
            fallback_action_not_in_this_marker=True))
        receipt['action_checkpoint_none']=io.ref(destination)
    receipt['action_checkpoint_io_seconds']=perf_counter()-tick
    receipt['action_checkpoint_time_scope']='Extra persistence is included by the unchanged outer runner quote_wall_seconds'
    return receipt,point


def initialize(out):
    """Spawn-safe worker initializer; only persistence is added to plan()."""
    original.old.initialize(out)
    original.solver.plan=checkpoint_plan


def process_alive(pid):
    """Read process state without sending a signal, including on Windows."""
    if os.name=='nt':
        kernel=ctypes.WinDLL('kernel32',use_last_error=True)
        kernel.OpenProcess.argtypes=(ctypes.c_ulong,ctypes.c_int,ctypes.c_ulong)
        kernel.OpenProcess.restype=ctypes.c_void_p
        kernel.GetExitCodeProcess.argtypes=(ctypes.c_void_p,ctypes.POINTER(ctypes.c_ulong))
        kernel.CloseHandle.argtypes=(ctypes.c_void_p,)
        handle=kernel.OpenProcess(0x1000,False,int(pid))
        if not handle:
            error=ctypes.get_last_error()
            if error==87:return False
            raise OSError(error,'Unable to verify stopped original PID '+str(pid))
        try:
            code=ctypes.c_ulong()
            if not kernel.GetExitCodeProcess(handle,ctypes.byref(code)):
                raise ctypes.WinError(ctypes.get_last_error())
            return code.value==259
        finally:kernel.CloseHandle(handle)
    try:os.kill(int(pid),0)
    except ProcessLookupError:return False
    return True


def classify(out,entries,forecasts):
    complete,partial,pending=[],[],[]
    for entry in entries:
        for draw in range(forecasts):
            directory=out/'fraction0.10'/entry['id']/f'forecast{draw:02d}'
            task=dict(id=entry['id'],forecast_index=draw)
            if (directory/'complete.json').exists():
                done=io.read(directory/'complete.json')
                if done.get('status')!='complete':raise ValueError('Invalid existing completion '+str(directory))
                for key in ('plans','results'):original.old.checked(done[key])
                complete.append(dict(task,completion=io.ref(directory/'complete.json'),
                    seal=io.ref(directory/'seal.json'),plans=done['plans'],results=done['results']))
            elif directory.exists():partial.append(task)
            else:pending.append(task)
    return complete,partial,pending


def prepare(out):
    out=Path(out).resolve()
    if (out/'complete.json').exists():raise ValueError('Main panel already complete')
    if (out/(STEM+'_protocol.json')).exists():raise FileExistsError('Continuation stage already prepared')
    p=io.read(out/'protocol.json');paused=io.read(out/'paused.json')
    if paused.get('status')!='paused_by_user' or not paused.get('source_unchanged'):
        raise ValueError('Original pause receipt missing or invalid')
    stopped=paused.get('termination',{})
    if not stopped.get('recorded_pids_absent_on_GetProcess_check'):
        raise ValueError('Pause receipt does not establish stopped original processes')
    pids=stopped.get('recorded_main_and_worker_pids',[])
    if not pids or any(process_alive(pid) for pid in pids):
        raise RuntimeError('An original process is alive or stopped evidence is incomplete')
    original.old.checked(paused['protocol']);original.old.checked(paused['progress'])
    for row in paused['partial_tasks']:
        for ref in [row['started'],*row['saved_plans'],*row['native_logs']]:original.old.checked(ref)
    for ref in [*p['sources'],p['inputs'],p['qualification_ledger'],p['preparation_failures']]:original.old.checked(ref)
    entries=io.read(p['inputs']['path'])
    planning_refs=[]
    for entry in entries:
        refs=[entry[k] for k in ('score','resource_context','resource_spec','forecast_ref')]
        refs.append(entry['files']['validation.npz'])
        for ref in refs:original.old.checked(ref)
        planning_refs.extend(refs)
    complete,partial,pending=classify(out,entries,p['forecasts'])
    expected_partial={(r['id'],int(r['forecast'].removeprefix('forecast'))) for r in paused['partial_tasks']}
    if {(r['id'],r['forecast_index']) for r in partial}!=expected_partial:
        raise ValueError('Interrupted task inventory differs from sealed pause')
    if (len(complete),len(partial),len(pending))!=(paused['completed_forecasts'],paused['interrupted_forecasts'],paused['not_started_forecasts']):
        raise ValueError('Task counts differ from sealed pause')
    own=Path(__file__).resolve();snapshot=out/(STEM+'_source_snapshot')/own.name
    snapshot.parent.mkdir(parents=True,exist_ok=False);shutil.copyfile(own,snapshot)
    receipt=dict(status='prepared',created_utc=original.old.utc(),stage=STEM,workers=4,
        original_protocol=io.ref(out/'protocol.json'),pause=io.ref(out/'paused.json'),
        original_sources=p['sources'],original_input=p['inputs'],planning_input_refs=planning_refs,
        source=io.ref(own),source_snapshot=io.ref(snapshot),environment=io.environment(),
        solver_seconds=p['solver_seconds'],anchor_seconds=p['anchor_seconds'],shared_seed_seconds=p['shared_seed_seconds'],
        seed=p['seed'],modes=p['modes'],policies=p['policies'],
        original_pids_rechecked_absent=pids,completed_retained=complete,interrupted_retained=partial,tasks=pending,
        timing='Original budgets and solver arguments unchanged; atomic checkpoint I/O included by original outer quote timer',
        scope='Only absent forecast directories; no historical fitting, prepare, completed or interrupted forecast replay; no main completion marker',
        runtime_change='Worker wraps original distribution solver.plan only to save its returned action or no-point marker before returning')
    io.write(out/(STEM+'_protocol.json'),receipt)
    return receipt


def verify_sources(p):
    for ref in [*p['original_sources'],p['source'],p['source_snapshot'],p['original_protocol'],p['original_input'],*p['planning_input_refs']]:
        original.old.checked(ref)
    for row in p['completed_retained']:
        for key in ('completion','seal','plans','results'):original.old.checked(row[key])


def run_stage(out,p):
    out=Path(out).resolve();verify_sources(p)
    done=out/(STEM+'_complete.json')
    if done.exists():return io.read(done)
    entries={e['id']:e for e in io.read(p['original_input']['path'])}
    tasks=[(entries[t['id']],t['forecast_index']) for t in p['tasks']]
    for entry,draw in tasks:
        if (out/'fraction0.10'/entry['id']/f'forecast{draw:02d}').exists():
            raise RuntimeError('Never-started stage target appeared before execution')
    lock=out/(STEM+'_lock.json')
    with lock.open('x',encoding='utf-8') as stream:
        json.dump(dict(pid=os.getpid(),created_utc=original.old.utc(),tasks=len(tasks)),stream)
    receipts=[];failures=[];began=perf_counter()
    try:
        with ProcessPoolExecutor(max_workers=p['workers'],initializer=initialize,initargs=(str(out),)) as pool:
            jobs={pool.submit(original.run_forecast,t):t for t in tasks}
            for future in as_completed(jobs):
                entry,draw=jobs[future]
                try:receipts.append(future.result())
                except Exception as exc:
                    failures.append(dict(id=entry['id'],forecast_index=draw,status='task_failed',error=repr(exc),traceback=traceback.format_exc()))
                io.write(out/(STEM+'_progress.json'),dict(total=len(tasks),completed=len(receipts),failed_tasks=failures,receipts=receipts))
                print(f'{STEM}: {len(receipts)+len(failures)}/{len(tasks)}; {len(failures)} failed',flush=True)
        verify_sources(p)
        result=dict(status='complete',stage=STEM,total_forecasts=len(tasks),completed_forecasts=len(receipts),failed_tasks=failures,
            seconds=perf_counter()-began,source_unchanged=True,completed_original_forecasts_unchanged=True,
            stage_protocol=io.ref(out/(STEM+'_protocol.json')),receipts=receipts,
            main_completion_not_written=True,original_lock_and_pause_retained=True)
        io.write(done,result);return result
    finally:lock.unlink(missing_ok=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    args=parser.parse_args();p=prepare(args.run)
    print(run_stage(args.run,p))


if __name__=='__main__':main()
