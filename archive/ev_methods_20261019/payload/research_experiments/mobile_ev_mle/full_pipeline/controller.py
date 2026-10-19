"""Resumable spawn-safe controller for the declared full-pipeline cases.

Only the parent prepares/freezes policy. Workers write separate case files and
logs. Failed case records and controller-failure receipts remain in denominators
and are never silently rerun. This controller does not select scientific rules.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager, redirect_stdout, redirect_stderr
from copy import deepcopy
from datetime import datetime, timezone
import ctypes
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import sys
import traceback
import uuid

THREAD_VARIABLES=('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS')
PHASES=('development','calibration','evaluation')
_WORKER_OUTPUT=None
_WORKER_TOKEN=None


def _now():return datetime.now(timezone.utc).isoformat()


def _pin_execution_path():
    """Spawn inherits sys.path; keep the fingerprinted stage ahead of mainline."""
    root=str(Path(__file__).resolve().parents[3])
    normalized=os.path.normcase(os.path.abspath(root))
    sys.path[:]=[root]+[p for p in sys.path if os.path.normcase(os.path.abspath(p or os.curdir))!=normalized]


def _write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    blob=(json.dumps(value,ensure_ascii=False,sort_keys=True,allow_nan=False)+'\n').encode('utf8')
    temporary=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    temporary.write_bytes(blob);temporary.replace(path)


def _read(path):return json.loads(Path(path).read_text(encoding='utf8'))


def process_alive(pid):
    """Read-only existence check; Windows never calls os.kill(pid, 0)."""
    if isinstance(pid,bool) or not isinstance(pid,int) or pid<=0:return False
    if os.name!='nt':
        try:os.kill(pid,0)
        except ProcessLookupError:return False
        except PermissionError:return True
        return True
    from ctypes import wintypes
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.OpenProcess.argtypes=(wintypes.DWORD,wintypes.BOOL,wintypes.DWORD)
    kernel.OpenProcess.restype=wintypes.HANDLE
    kernel.GetExitCodeProcess.argtypes=(wintypes.HANDLE,ctypes.POINTER(wintypes.DWORD))
    kernel.GetExitCodeProcess.restype=wintypes.BOOL
    kernel.CloseHandle.argtypes=(wintypes.HANDLE,);kernel.CloseHandle.restype=wintypes.BOOL
    handle=kernel.OpenProcess(0x1000,False,pid)  # PROCESS_QUERY_LIMITED_INFORMATION
    if not handle:
        error=ctypes.get_last_error()
        if error==87:return False  # ERROR_INVALID_PARAMETER: no such PID
        # Access denied or an unexpected query failure is treated as alive.
        return True
    try:
        exit_code=wintypes.DWORD()
        if not kernel.GetExitCodeProcess(handle,ctypes.byref(exit_code)):return True
        return exit_code.value==259  # STILL_ACTIVE
    finally:kernel.CloseHandle(handle)


def live_workers(output):
    alive=[]
    for path in sorted((Path(output)/'controller_workers').glob('worker_*.json')):
        try:row=_read(path)
        except (OSError,ValueError):continue
        if process_alive(row.get('pid')):alive.append(row)
    return alive


@contextmanager
def controller_lock(output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    path=output/'controller.lock.json';token=uuid.uuid4().hex
    while True:
        try:fd=os.open(path,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
        except FileExistsError:
            try:previous=_read(path)
            except (OSError,ValueError) as exc:
                raise RuntimeError('Existing controller lock cannot be read; preserve it for inspection') from exc
            if process_alive(previous.get('pid')):
                raise RuntimeError('Controller already running, PID '+str(previous.get('pid')))
            active=live_workers(output)
            if active:raise RuntimeError('Previous controller ended but workers remain active: '+str([r['pid'] for r in active]))
            stale=path.with_name('controller.lock.stale.'+uuid.uuid4().hex+'.json')
            try:path.replace(stale)
            except FileNotFoundError:continue
        else:break
    owner=dict(pid=os.getpid(),token=token,started_utc=_now(),argv=sys.argv)
    with os.fdopen(fd,'w',encoding='utf8') as stream:json.dump(owner,stream,sort_keys=True)
    try:
        active=live_workers(output)
        if active:raise RuntimeError('Live workers already use this output: '+str([r['pid'] for r in active]))
        yield owner
    finally:
        # Never unlink a lock which another process has replaced.
        try:
            if _read(path).get('token')==token:path.unlink()
        except (FileNotFoundError,ValueError):pass


def _study():
    from . import study
    return study


def _worker_init(output,token):
    global _WORKER_OUTPUT,_WORKER_TOKEN
    for name in THREAD_VARIABLES:os.environ[name]='1'
    _WORKER_OUTPUT=str(output);_WORKER_TOKEN=token
    _write(Path(output)/'controller_workers'/('worker_%d.json'%os.getpid()),
        dict(pid=os.getpid(),controller_pid=os.getppid(),controller_token=token,started_utc=_now(),
            threads={name:os.environ[name] for name in THREAD_VARIABLES}))


def task_name(spec):
    return '%s_c%03d_n%d_%s'%(spec['phase'],spec['cluster_index'],spec['n'],spec['source_mode'])


def _worker(spec,output,prepared,freeze_policy):
    """Top-level picklable worker; no module-level numerical-library import."""
    for name in THREAD_VARIABLES:os.environ[name]='1'
    logs=Path(output)/'logs';logs.mkdir(parents=True,exist_ok=True)
    stem=task_name(spec)
    # Append on external interruption so incomplete earlier diagnostics survive.
    with (logs/(stem+'.stdout.log')).open('a',encoding='utf8',buffering=1) as stdout, \
         (logs/(stem+'.stderr.log')).open('a',encoding='utf8',buffering=1) as stderr, \
         redirect_stdout(stdout),redirect_stderr(stderr):
        print('WORKER_START',json.dumps(dict(pid=os.getpid(),spec=spec,utc=_now()),sort_keys=True),flush=True)
        try:
            result=_study().run(spec,output,prepared,freeze_policy=freeze_policy)
            print('WORKER_RETURN',json.dumps(result,sort_keys=True),flush=True)
            return dict(result,pid=os.getpid(),stdout=str(logs/(stem+'.stdout.log')),
                stderr=str(logs/(stem+'.stderr.log')))
        except BaseException:
            traceback.print_exc(file=stderr);raise


def _json_digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode('utf8')).hexdigest()


def _policy(output,prepared,development_unfrozen):
    study=_study();prepared=Path(prepared)
    data=study.read(prepared);config=deepcopy(study.CONFIG);config['forecast_config']=data['forecast_config']
    if development_unfrozen:
        value=dict(identity='development-unfrozen',config=config,prepared_sha256=study.digest(prepared),
            source_sha256=study.source_hashes())
    else:value=study.freeze(output,config,prepared)
    value=study.serial(value);value['controller_policy_sha256']=_json_digest(value)
    path=Path(output)/'controller_policy.json'
    if path.exists() and _read(path)!=value:
        raise RuntimeError('Saved controller policy differs; preserve old output and use a new output version')
    if not path.exists():_write(path,value)
    return data,value


def _failure_path(output,spec):return Path(output)/'controller_failures'/(task_name(spec)+'.json')


def existing_result(output,spec,policy):
    study=_study();path=study.path_for(output,spec)
    if path.exists():
        row=study.read(path)
        if row.get('freeze_identity')!=policy['identity'] or row.get('source_unchanged') is not True or row.get('spec')!=spec:
            raise RuntimeError('Existing case identity/spec/source does not match; retained without rerun: '+str(path))
        return dict(spec=spec,path=str(path),status=row.get('status','unknown'),
            seconds=row.get('seconds'),resumed=True,raw_saved=True)
    failure=_failure_path(output,spec)
    if failure.exists():
        row=_read(failure)
        if row.get('controller_policy_sha256')!=policy['controller_policy_sha256'] or row.get('spec')!=spec:
            raise RuntimeError('Existing controller failure policy differs: '+str(failure))
        return dict(spec=spec,path=str(failure),status='controller_worker_exception',
            resumed=True,raw_saved=False,failure=row['exception'])
    return None


def _counts(specs,finished):
    phases={}
    for phase in PHASES:
        expected=[s for s in specs if s['phase']==phase]
        if not expected:continue
        rows=[finished[task_name(s)] for s in expected if task_name(s) in finished]
        statuses={}
        for row in rows:statuses[row['status']]=statuses.get(row['status'],0)+1
        phases[phase]=dict(total=len(expected),computed=len(rows),remaining=len(expected)-len(rows),
            complete=sum(r['status']=='complete' for r in rows),
            computed_with_failure=sum(r['status']!='complete' for r in rows),statuses=statuses)
    return dict(total=len(specs),computed=len(finished),remaining=len(specs)-len(finished),phases=phases)


def run_controller(output,prepared=None,*,phases=('development','calibration','evaluation'),workers=12,
                   development_unfrozen=False,executor_factory=ProcessPoolExecutor):
    if isinstance(workers,bool) or int(workers)!=workers or workers<1:raise ValueError('workers must be a positive integer')
    phases=tuple(dict.fromkeys(phases))
    if not phases or any(p not in PHASES for p in phases):raise ValueError('registered nonempty phases required')
    phases=tuple(p for p in PHASES if p in phases)
    if development_unfrozen and phases!=('development',):raise ValueError('Unfrozen runs may contain only development')
    _pin_execution_path()
    output=Path(output).resolve();prepared=Path(prepared).resolve() if prepared else output/'prepared.json.gz'
    for name in THREAD_VARIABLES:os.environ[name]='1'
    with controller_lock(output) as owner:
        state=dict(schema='full_pipeline_controller_v1',pid=os.getpid(),controller_token=owner['token'],
            started_utc=_now(),updated_utc=_now(),output=str(output),prepared=str(prepared),
            requested_phases=list(phases),workers=int(workers),per_worker_threads=1,
            development_unfrozen=bool(development_unfrozen),status='preparing',finished={},exceptions=[])
        def save(**values):
            state.update(values);state['updated_utc']=_now();_write(output/'controller.json',state)
        save()
        try:
            study=_study()
            if not prepared.exists():
                default=output/'prepared.json.gz'
                if prepared!=default:raise FileNotFoundError('Explicit prepared file does not exist: '+str(prepared))
                study.prepare(output)
            data,policy=_policy(output,prepared,development_unfrozen)
            specs=[s for p in phases for s in study.specifications(data,p)]
            finished={}
            for spec in specs:
                previous=existing_result(output,spec,policy)
                if previous is not None:finished[task_name(spec)]=previous
            save(status='running',freeze_identity=policy['identity'],controller_policy_sha256=policy['controller_policy_sha256'],
                finished=finished,counts=_counts(specs,finished),active_phase=None)
            for phase in phases:
                queue=[s for s in specs if s['phase']==phase and task_name(s) not in finished]
                if not queue:continue
                save(active_phase=phase,pending=[task_name(s) for s in queue])
                executor=executor_factory(max_workers=min(int(workers),len(queue)),
                    mp_context=multiprocessing.get_context('spawn'),initializer=_worker_init,
                    initargs=(str(output),owner['token']))
                futures={}
                try:
                    futures={executor.submit(_worker,s,str(output),str(prepared),not development_unfrozen):s for s in queue}
                    for future in as_completed(futures):
                        spec=futures[future];name=task_name(spec)
                        try:
                            value=future.result()
                            # A returned result is accepted only after replaying
                            # the same raw-file identity/source checks as resume.
                            value=existing_result(output,spec,policy)
                            if value is None:raise RuntimeError('Worker returned without a durable case record')
                            value['resumed']=False
                        except Exception as exc:
                            failure=dict(spec=spec,controller_policy_sha256=policy['controller_policy_sha256'],
                                recorded_utc=_now(),exception=dict(type=type(exc).__name__,message=str(exc),
                                    traceback=traceback.format_exc()),raw_preserved=study.path_for(output,spec).exists())
                            _write(_failure_path(output,spec),failure)
                            value=dict(spec=spec,path=str(_failure_path(output,spec)),status='controller_worker_exception',
                                resumed=False,raw_saved=failure['raw_preserved'],failure=failure['exception'])
                            state['exceptions'].append(failure)
                        finished[name]=value
                        save(finished=finished,counts=_counts(specs,finished),
                            pending=[task_name(s) for s in queue if task_name(s) not in finished],
                            live_worker_pids=[r['pid'] for r in live_workers(output)])
                except BaseException:
                    for future in futures:future.cancel()
                    save(status='interrupted',finished=finished,counts=_counts(specs,finished),
                        controller_traceback=traceback.format_exc())
                    raise
                finally:
                    executor.shutdown(wait=True,cancel_futures=True)
            save(status='statistical_complete',finished=finished,counts=_counts(specs,finished),
                active_phase=None,pending=[],completed_utc=_now(),
                computation_complete=True,scientific_success_assumed=False)
            return state
        except BaseException:
            if state.get('status')!='interrupted':
                save(status='controller_failed',controller_traceback=traceback.format_exc())
            raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--prepared',type=Path)
    parser.add_argument('--phases',nargs='+',choices=PHASES,default=list(PHASES))
    parser.add_argument('--workers',type=int,default=12)
    parser.add_argument('--development-unfrozen',action='store_true')
    args=parser.parse_args()
    output=args.output
    if output is None:
        for name in THREAD_VARIABLES:os.environ[name]='1'
        output=_study().DEFAULT_OUTPUT
    result=run_controller(output,args.prepared,phases=args.phases,workers=args.workers,
        development_unfrozen=args.development_unfrozen)
    print(json.dumps(dict(status=result['status'],counts=result['counts'],output=str(output)),sort_keys=True),flush=True)


if __name__=='__main__':
    multiprocessing.freeze_support();main()
