"""Freeze and replay existing A records with independent B calibration.

No case/action/noise generator is used. A evaluation records are opened only
after a B source freeze and an immutable binding to all B calibration results.
"""
from __future__ import annotations
import argparse
import base64
import gzip
from hashlib import sha256
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
import numpy as np
import scipy

from . import io,replay,optimizer
from .. import ev_ac_repair_analysis as analysis
from ..ev_ac_repair_control import _pid_alive

HERE=Path(__file__).resolve().parent
MODULE_ROOT=HERE.parent
WORK=MODULE_ROOT.parents[1]
DEFAULT_SETTINGS=dict(draws=256,replicates=4,workers=2,timeout_seconds=1800.,
    base_kw=20.,future_root_v=1.,solver_options=dict(optimizer.DEFAULTS),
    truncation_options=dict(seed=0,power=10,max_power=18,replicates=8,min_ess_fraction=0.),
    propagation_mean_MCSE_limit=analysis.MEAN_MCSE_LIMIT,
    propagation_covariance_MCSE_relative_limit=analysis.COVARIANCE_MCSE_LIMIT,
    raw_variance_multiplier=1.,calibration_multipliers=[1,2,4,8,16,32])


def file_hash(path):return sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path,value,*,retry_seconds=10.):
    """Unique temporary name and bounded retries for Windows viewer locks."""
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.'+str(os.getpid())+'.tmp')
    payload=json.dumps(io._serial(value),ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+'\n'
    started=time.monotonic();attempt=0
    while True:
        try:
            temporary.write_text(payload,encoding='utf-8');temporary.replace(path);return
        except PermissionError:
            if time.monotonic()-started>=retry_seconds:raise
            time.sleep(min(.05*2**min(attempt,4),.8));attempt+=1


def source_manifest(input_A_root,*,module_root=MODULE_ROOT,b_root=HERE,work_root=WORK,expected_A_count=150):
    """Read A freeze/source only; no A history or summary is opened here."""
    input_A_root=Path(input_A_root).resolve();A=io.load_record(input_A_root/'freeze.json')
    if expected_A_count is not None and len(A['source_sha256'])!=expected_A_count:
        raise ValueError('A freeze must name exactly '+str(expected_A_count)+' source files')
    paths={};hashes={}
    for name,expected in A['source_sha256'].items():
        path=Path(work_root)/name if name.startswith('rnj_wzzt_core/') else Path(module_root)/name
        path=path.resolve();actual=file_hash(path)
        if actual!=expected:raise ValueError('A frozen source mismatch: '+name)
        paths['A/'+name]=str(path);hashes['A/'+name]=actual
    for path in sorted(Path(b_root).glob('*')):
        if path.is_file() and (path.suffix=='.py' or path.name in ('math.md','protocol.md')):
            name='B/'+path.name;paths[name]=str(path.resolve());hashes[name]=file_hash(path)
    return dict(input_A_root=str(input_A_root),input_A_policy_id=A['policy_id'],
        input_A_freeze_sha256=file_hash(input_A_root/'freeze.json'),
        specifications={phase:A['specifications'][phase] for phase in ('calibration','evaluation')},
        source_sha256=hashes,source_paths=paths,A_source_count=len(A['source_sha256']),
        B_source_count=sum(name.startswith('B/') for name in hashes))


def create_freeze(root,input_A_root,settings=None,**manifest_options):
    root=Path(root).resolve();path=root/'freeze.json'
    if path.exists():raise FileExistsError('B freeze is immutable; use existing run or a new root')
    manifest=source_manifest(input_A_root,**manifest_options)
    settings=io._serial({**DEFAULT_SETTINGS,**(settings or {})})
    for key in ('draws','replicates','workers'):
        if isinstance(settings[key],bool) or int(settings[key])!=settings[key] or settings[key]<(1 if key=='workers' else 2):
            raise ValueError('invalid positive integer setting: '+key)
    if settings['timeout_seconds']<=0 or not np.isfinite(settings['timeout_seconds']):raise ValueError('positive timeout required')
    if settings['raw_variance_multiplier']!=1. or settings['calibration_multipliers']!=[1,2,4,8,16,32]:
        raise ValueError('fixed raw variance and B calibration rule required')
    value=dict(version='ac_history_refresh_B_v1',settings=settings,**manifest,
        environment=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,executable=sys.executable),
        no_new_case_action_noise=True,old_A_calibration_used=False,
        freeze_before_A_evaluation_read=True)
    value['policy_id']=io.fingerprint(value)
    root.mkdir(parents=True,exist_ok=True)
    snapshot={name:base64.b64encode(Path(value['source_paths'][name]).read_bytes()).decode('ascii')
              for name in value['source_sha256']}
    snapshot_bytes=gzip.compress(json.dumps(snapshot,sort_keys=True).encode(),mtime=0)
    with (root/'source_snapshot.json.gz').open('xb') as handle:handle.write(snapshot_bytes)
    atomic_json(path,value);return value


def frozen(root):
    path=Path(root)/'freeze.json'
    if not path.exists():raise ValueError('B freeze required before reading any A record')
    value=io.load_record(path)
    identity={k:v for k,v in value.items() if k!='policy_id'}
    if io.fingerprint(identity)!=value['policy_id']:raise ValueError('B freeze identity mismatch')
    if file_hash(Path(value['input_A_root'])/'freeze.json')!=value['input_A_freeze_sha256']:
        raise ValueError('bound A source freeze changed')
    for name,expected in value['source_sha256'].items():
        if file_hash(value['source_paths'][name])!=expected:raise ValueError('B frozen source changed: '+name)
    return value


def specifications(version,phase):
    if phase not in ('calibration','evaluation'):raise ValueError('calibration/evaluation only')
    return version['specifications'][phase]


def result_path(root,phase,index):return Path(root)/phase/('history_%03d.json.gz'%index)
def meta_path(root,phase,index):return Path(root)/phase/('history_%03d.metadata.json'%index)


def saved_A_input(version,phase,index):
    root=Path(version['input_A_root']);path=root/phase/('history_%03d.json.gz'%index)
    claim=root/'launch_claims'/(phase+'_%03d.json'%index)
    if claim.exists() and io.load_record(claim).get('status')!='finished':
        raise RuntimeError('original A worker has not finished saving input: '+str(claim))
    if not path.exists():raise FileNotFoundError('preserve missing A input; no regeneration: '+str(path))
    return path


def validate_result(root,version,phase,spec,*,load=True):
    index=spec['index'];path=result_path(root,phase,index);meta=io.load_record(meta_path(root,phase,index))
    if meta['policy_id']!=version['policy_id'] or meta['result_sha256']!=file_hash(path):
        raise ValueError('B result provenance mismatch')
    if meta.get('spec')!=spec:raise ValueError('B result metadata specification mismatch')
    record=io.load_record(path) if load else None
    if load and record.get('spec')!=spec:raise ValueError('B result specification mismatch')
    if not meta.get('science_source_unchanged'):raise ValueError('B source changed during worker')
    source=Path(version['input_A_root'])/phase/path.name
    if file_hash(source)!=meta['source_A_sha256']:raise ValueError('original A raw changed')
    return record,meta


def calibrate_B(root):
    """Use the original NLL rule, retaining only B's G11 calibration output."""
    root=Path(root);version=frozen(root);records=[];hashes={};Ahashes={}
    for spec in specifications(version,'calibration'):
        record,meta=validate_result(root,version,'calibration',spec)
        records.append(analysis.compact_record(record));name=result_path(root,'calibration',spec['index']).name
        hashes[name]=meta['result_sha256'];Ahashes[name]=meta['source_A_sha256']
    calibrated=analysis.calibrate(records)
    keep=('histories','available_histories','G11_multiplier','objective','history_losses','objective_scope')
    value={k:calibrated[k] for k in keep}
    value.update(version='ac_history_refresh_B_calibration_v1',policy_id=version['policy_id'],
        input_A_policy_id=version['input_A_policy_id'],input_B_sha256=hashes,input_A_sha256=Ahashes,
        multipliers=[1,2,4,8,16,32],fixed_before_evaluation=True,old_A_calibration_used=False)
    path=root/'calibration.json'
    if path.exists() and io.load_record(path)!=io._serial(value):
        raise ValueError('preserve existing B calibration; inputs or rule changed')
    if not path.exists():atomic_json(path,value)
    return value


def bind_evaluation(root,version=None):
    root=Path(root);version=frozen(root) if version is None else version
    path=root/'calibration.json'
    if not path.exists():raise ValueError('independent B calibration required before A evaluation read')
    calibration=io.load_record(path)
    if calibration.get('policy_id')!=version['policy_id'] or calibration.get('histories')!=len(specifications(version,'calibration')):
        raise ValueError('B calibration policy/count mismatch')
    if calibration.get('G11_multiplier') not in (1,2,4,8,16,32):raise ValueError('B G11 calibration failed')
    actual={}
    for spec in specifications(version,'calibration'):
        _,meta=validate_result(root,version,'calibration',spec,load=False)
        actual[result_path(root,'calibration',spec['index']).name]=meta['result_sha256']
    if actual!=calibration['input_B_sha256']:raise ValueError('B calibration histories changed')
    binding=dict(policy_id=version['policy_id'],calibration_sha256=file_hash(path),
        G11_multiplier=calibration['G11_multiplier'],input_A_policy_id=version['input_A_policy_id'])
    destination=root/'evaluation_calibration_binding.json'
    if destination.exists() and io.load_record(destination)!=binding:
        raise ValueError('B calibration changed after evaluation binding')
    if not destination.exists():atomic_json(destination,binding)
    return binding


def _claim(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        old_claim=io.load_record(path)
        if old_claim.get('status')=='running' and _pid_alive(old_claim['pid']):
            raise RuntimeError('worker already active: '+str(old_claim['pid']))
        archived=path.with_name(path.name+'.previous.'+str(time.time_ns()))
        path.replace(archived)
    with path.open('x',encoding='utf-8') as handle:json.dump(value,handle)


def worker(root,phase,index,*,replay_function=None):
    """Binding is checked before source A bytes are opened."""
    root=Path(root).resolve();version=frozen(root)
    specs={s['index']:s for s in specifications(version,phase)}
    if index not in specs:raise ValueError('index outside frozen B specification')
    binding=bind_evaluation(root,version) if phase=='evaluation' else None
    destination=result_path(root,phase,index)
    if destination.exists():return validate_result(root,version,phase,specs[index])[0]
    source=saved_A_input(version,phase,index)
    claim=root/'worker_claims'/(phase+'_%03d.json'%index)
    _claim(claim,dict(status='running',pid=os.getpid(),phase=phase,index=index,policy_id=version['policy_id']))
    began=time.monotonic();source_hash=file_hash(source);settings=version['settings']
    def progress(stage,details):atomic_json(root/phase/('history_%03d.progress.json'%index),
        dict(stage=stage,details=details,pid=os.getpid(),policy_id=version['policy_id']))
    function=replay.replay_file if replay_function is None else replay_function
    try:
        record=function(source,destination,draws=settings['draws'],replicates=settings['replicates'],
            base_kw=settings['base_kw'],future_root_v=settings['future_root_v'],
            solver_options=settings['solver_options'],truncation_options=settings['truncation_options'],
            variance_multiplier=1.,progress=progress)
    except Exception as exc:
        if destination.exists():raise
        record=dict(schema='ac_history_refresh_fixed_replay_v1',spec=specs[index],status='process_failure',
            failures=dict(worker=type(exc).__name__+': '+str(exc)),n_independent_histories=1,
            keep_all_on_failure=True,no_new_case_action_noise=True,seconds=time.monotonic()-began)
        io.save_record(destination,record)
    unchanged=True
    try:frozen(root)
    except ValueError:unchanged=False
    if file_hash(source)!=source_hash:raise ValueError('A raw changed while worker read it')
    meta=dict(policy_id=version['policy_id'],input_A_policy_id=version['input_A_policy_id'],
        spec=specs[index],
        source_A_sha256=source_hash,source_A_path=str(source),result_sha256=file_hash(destination),
        science_source_unchanged=unchanged,calibration_binding=binding,pid=os.getpid(),
        seconds=time.monotonic()-began)
    atomic_json(meta_path(root,phase,index),meta)
    atomic_json(claim,dict(status='finished',pid=os.getpid(),policy_id=version['policy_id'],result=str(destination)))
    return record


def run_phase(root,phase):
    root=Path(root).resolve();version=frozen(root);settings=version['settings']
    if phase=='evaluation':bind_evaluation(root,version)
    lock=root/'controller_lock.json'
    if lock.exists():
        previous=io.load_record(lock)
        if _pid_alive(previous['pid']):raise RuntimeError('B controller already active')
        lock.replace(root/('stale_lock_'+str(time.time_ns())+'.json'))
    for p in (root/'launch_claims').glob('*.json'):
        claim=io.load_record(p)
        if claim['status']=='launching' or (claim['status']=='running' and _pid_alive(claim['pid'])):
            raise RuntimeError('unresolved/live launch claim: '+str(p))
    for p in (root/'worker_claims').glob('*.json'):
        claim=io.load_record(p)
        if claim['status']=='running' and _pid_alive(claim['pid']):raise RuntimeError('B worker remains active')
    specs=specifications(version,phase)
    for spec in specs:
        saved_A_input(version,phase,spec['index'])
    pending=[];active={};started=time.monotonic()
    for spec in specs:
        if result_path(root,phase,spec['index']).exists():validate_result(root,version,phase,spec)
        else:pending.append(spec)
    with lock.open('x',encoding='utf-8') as stream:json.dump(dict(pid=os.getpid(),phase=phase),stream)
    try:
        while pending or active:
            frozen(root)
            while pending and len(active)<settings['workers']:
                spec=pending.pop(0);index=spec['index'];folder=root/phase;folder.mkdir(exist_ok=True)
                log=(folder/('history_%03d.log'%index)).open('ab',buffering=0)
                command=[sys.executable,'-u','-m','research_experiments.mobile_ev_mle.ac_history_refresh.control',
                    'worker','--root',str(root),'--phase',phase,'--index',str(index)]
                claim=root/'launch_claims'/(phase+'_%03d.json'%index)
                atomic_json(claim,dict(status='launching',controller_pid=os.getpid(),command=command))
                process=subprocess.Popen(command,cwd=WORK,stdout=log,stderr=subprocess.STDOUT,
                    creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0),env=dict(os.environ))
                atomic_json(claim,dict(status='running',controller_pid=os.getpid(),pid=process.pid,command=command))
                active[index]=dict(process=process,spec=spec,started=time.monotonic(),log=log,claim=claim)
            for index,job in list(active.items()):
                process=job['process'];timeout=time.monotonic()-job['started']>settings['timeout_seconds']
                if process.poll() is None and not timeout:continue
                if timeout and process.poll() is None:
                    process.terminate()
                    try:process.wait(timeout=20)
                    except subprocess.TimeoutExpired:process.kill();process.wait(timeout=20)
                job['log'].close();path=result_path(root,phase,index)
                if not path.exists():
                    # Run the same frozen worker wrapper once with an injected
                    # immediate failure; never replace or regenerate an A input.
                    def failed(*args,**kwargs):raise RuntimeError('worker timeout' if timeout else 'worker exit '+str(process.returncode))
                    worker(root,phase,index,replay_function=failed)
                validate_result(root,version,phase,job['spec'])
                atomic_json(job['claim'],dict(status='finished',pid=process.pid,result=str(path)))
                del active[index]
            atomic_json(root/'controller.json',dict(status='running',phase=phase,pid=os.getpid(),
                policy_id=version['policy_id'],total=len(specs),pending=[s['index'] for s in pending],
                workers={str(i):dict(pid=j['process'].pid,seconds=time.monotonic()-j['started']) for i,j in active.items()},
                finished=[s['index'] for s in specs if result_path(root,phase,s['index']).exists()],seconds=time.monotonic()-started))
            if pending or active:time.sleep(2.)
        if phase=='calibration':summary=calibrate_B(root)
        else:
            calibration=io.load_record(root/'calibration.json')
            records=[analysis.compact_record(validate_result(root,version,phase,s)[0]) for s in specs]
            summary=analysis.summarize(records,calibration)
            atomic_json(root/'evaluation_summary.json',summary)
        atomic_json(root/'controller.json',dict(status='statistical_complete' if phase=='evaluation' else 'calibration_complete',
            phase=phase,pid=os.getpid(),policy_id=version['policy_id'],workers={},total=len(specs),
            seconds=time.monotonic()-started))
        return summary
    finally:
        if active:
            atomic_json(root/'orphan_workers.json',dict(workers={str(i):j['process'].pid for i,j in active.items()}))
        if lock.exists() and io.load_record(lock).get('pid')==os.getpid():lock.unlink()


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=('manifest','freeze','worker','run','calibrate','bind'))
    parser.add_argument('--root');parser.add_argument('--input-A-root');parser.add_argument('--output')
    parser.add_argument('--settings');parser.add_argument('--phase',choices=('calibration','evaluation'))
    parser.add_argument('--index',type=int);args=parser.parse_args()
    if args.command=='manifest':
        value=source_manifest(args.input_A_root)
        if args.output:atomic_json(args.output,value)
        print(json.dumps({k:v for k,v in value.items() if k not in ('source_sha256','source_paths','specifications')}));return
    if not args.root:parser.error('--root required')
    if args.command=='freeze':value=create_freeze(args.root,args.input_A_root,io.load_record(args.settings) if args.settings else None)
    elif args.command=='worker':value=worker(args.root,args.phase,args.index)
    elif args.command=='run':value=run_phase(args.root,args.phase)
    elif args.command=='calibrate':value=calibrate_B(args.root)
    else:value=bind_evaluation(args.root)
    print(json.dumps(dict(command=args.command,policy_id=value.get('policy_id'),status=value.get('status'),G11_multiplier=value.get('G11_multiplier'))))


if __name__=='__main__':main()
