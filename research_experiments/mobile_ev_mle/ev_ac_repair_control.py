"""Logged, resumable phase controller; evaluation requires frozen calibration."""
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

for variable in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
                 'NUMEXPR_NUM_THREADS'):
    os.environ[variable] = '1'

import numpy as np
import scipy

from . import ev_ac_repair_study as study
from . import ev_ac_repair_analysis as analysis
from . import ev_bayes_response_study as old


HERE = Path(__file__).resolve().parent
WORK = HERE.parents[1]


def atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(old.serial(value), ensure_ascii=False, sort_keys=True,
        indent=2, allow_nan=False)+'\n', encoding='utf-8', newline='\n')
    tmp.replace(path)


def hashes():
    result = old.source_hashes()
    for path in sorted(HERE.glob('*ev_ac_repair*')):
        if path.is_file() and (path.suffix == '.py' or path.name == 'ev_ac_repair_math.md'):
            result[path.name] = sha256(path.read_bytes()).hexdigest()
    return result


def freeze(root, settings):
    """New immutable version; originals and previous development stay intact."""
    root = Path(root); root.mkdir(parents=True, exist_ok=True)
    value = dict(settings=settings, study_config=study.CONFIG, source_sha256=hashes(),
        specifications={p: study.specifications(p) for p in
                        ('development', 'calibration', 'evaluation')},
        environment=dict(python=platform.python_version(), numpy=np.__version__,
            scipy=scipy.__version__, executable=sys.executable),
        pending='development validation then calibration then independent evaluation')
    value['policy_id'] = sha256(json.dumps(old.serial(value), sort_keys=True,
                                          ensure_ascii=False).encode()).hexdigest()
    path = root/'freeze.json'
    if path.exists():
        if analysis.load(path) != old.serial(value):
            raise ValueError('source/settings differ from immutable freeze; use a new version')
    else:
        atomic_json(path, value)
        snapshot = {}
        for name in value['source_sha256']:
            source = WORK/name if name.startswith('rnj_wzzt_core/') else HERE/name
            snapshot[name] = base64.b64encode(source.read_bytes()).decode()
        data = json.dumps(snapshot, sort_keys=True).encode()
        (root/'source_snapshot.json.gz').write_bytes(gzip.compress(data, mtime=0))
    return value


def _pid_alive(pid):
    if os.name == 'nt':
        import ctypes
        from ctypes import wintypes
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.GetExitCodeProcess.argtypes = (wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD))
        kernel.GetExitCodeProcess.restype = wintypes.BOOL
        kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
        handle = kernel.OpenProcess(0x1000, False, int(pid))
        if not handle:
            return ctypes.get_last_error() != 87  # invalid PID; access denied stays live
        try:
            code = wintypes.DWORD()
            if not kernel.GetExitCodeProcess(handle, ctypes.byref(code)):
                return True
            return code.value == 259  # STILL_ACTIVE
        finally:
            kernel.CloseHandle(handle)
    try:
        os.kill(int(pid), 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False


def run_phase(root, phase, *, workers=2, limit=None, draws=256, replicates=4,
              noise_repeats=1, clean_refit=True, timeout_seconds=1200.):
    root = Path(root).resolve()
    if limit is not None and phase != 'development':
        raise ValueError('calibration/evaluation require all 32 specifications')
    settings = dict(draws=draws, replicates=replicates, noise_repeats=noise_repeats,
                    clean_refit=clean_refit, timeout_seconds=timeout_seconds,
                    propagation_mean_MCSE_limit=analysis.MEAN_MCSE_LIMIT,
                    propagation_covariance_MCSE_relative_limit=analysis.COVARIANCE_MCSE_LIMIT,
                    workers=workers)
    version = freeze(root, settings)
    state_path = root/'controller.json'
    if state_path.exists():
        previous_workers = analysis.load(state_path).get('workers', {})
        alive = {i: j['pid'] for i, j in previous_workers.items()
                 if _pid_alive(j['pid'])}
        if alive:
            raise RuntimeError('workers in previous controller state remain active: '+str(alive))
    for claim in (root/'launch_claims').glob('*.json'):
        row = analysis.load(claim)
        if row.get('status') == 'launching':
            raise RuntimeError('unfinished launch claim; inspect before restart: '+str(claim))
        if row.get('status') == 'running' and _pid_alive(row['pid']):
            raise RuntimeError('registered worker remains active: '+str(row['pid']))
    orphan = root/'orphan_workers.json'
    if orphan.exists():
        previous = analysis.load(orphan)
        alive = {i: pid for i, pid in previous.get('workers', {}).items()
                 if _pid_alive(pid)}
        if alive:
            raise RuntimeError('previous workers remain active; inspect before restart: '+str(alive))
    if phase == 'evaluation':
        calibration = root/'calibration.json'
        if not calibration.exists():
            raise ValueError('evaluation requires saved calibration.json')
        calibration_record = analysis.load(calibration)
        if calibration_record.get('histories') != len(study.specifications('calibration')):
            raise ValueError('all 32 calibration histories, including failures, required')
        if calibration_record.get('G11_multiplier') is None:
            raise ValueError('calibration failed: no usable history for G11')
        if calibration_record.get('policy_id') != version['policy_id']:
            raise ValueError('calibration policy does not match source/settings freeze')
        expected_paths = {}
        for spec in study.specifications('calibration'):
            source = root/'calibration'/('history_%03d.json.gz'%spec['index'])
            if not source.exists() or analysis.load(source).get('spec') != spec:
                raise ValueError('missing/mismatched calibration specification')
            expected_paths[str(source.relative_to(root))] = sha256(source.read_bytes()).hexdigest()
        if expected_paths != calibration_record.get('input_sha256'):
            raise ValueError('calibration inputs differ from saved calibration')
        calibration_hash = sha256(calibration.read_bytes()).hexdigest()
        binding = root/'evaluation_calibration_binding.json'
        bound = dict(policy_id=version['policy_id'], calibration_sha256=calibration_hash)
        if binding.exists() and analysis.load(binding) != bound:
            raise ValueError('calibration changed after evaluation binding')
        atomic_json(binding, bound)
    lock = root/'controller_lock.json'
    if lock.exists():
        previous = analysis.load(lock)
        if _pid_alive(previous['pid']):
            raise RuntimeError('controller already active: '+str(previous))
        (root/('stale_lock_'+str(previous['pid'])+'.json')).write_bytes(lock.read_bytes())
        lock.unlink()
    with lock.open('x', encoding='utf-8') as handle:
        json.dump(dict(pid=os.getpid(), phase=phase, started=time.time()), handle)
    specs = study.specifications(phase)
    if limit is not None:
        specs = specs[:limit]
    active = {}; started = time.monotonic()
    def path(spec):
        return root/phase/('history_%03d.json.gz'%spec['index'])
    pending = [s for s in specs if not path(s).exists()]
    try:
        while pending or active:
            if hashes() != version['source_sha256']:
                raise RuntimeError('frozen source changed; pause without retuning')
            while pending and len(active) < workers:
                spec = pending.pop(0); index = spec['index']; path(spec).parent.mkdir(
                    parents=True, exist_ok=True)
                logfile = root/phase/('history_%03d.log'%index)
                log = logfile.open('ab', buffering=0)
                command = [sys.executable, '-u', '-m',
                    'research_experiments.mobile_ev_mle.ev_ac_repair_study',
                    '--output', str(root), '--phase', phase, '--index', str(index),
                    '--draws', str(draws), '--replicates', str(replicates),
                    '--noise-repeats', str(noise_repeats)]
                if not clean_refit:
                    command.append('--skip-clean-refit')
                claim = root/'launch_claims'/(phase+'_%03d.json'%index)
                atomic_json(claim, dict(status='launching', controller_pid=os.getpid(),
                    spec=spec, command=command, time=time.time()))
                process = subprocess.Popen(command, cwd=WORK, stdout=log,
                    stderr=subprocess.STDOUT, env=dict(os.environ),
                    creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
                atomic_json(claim, dict(status='running', controller_pid=os.getpid(),
                    pid=process.pid, spec=spec, command=command, time=time.time()))
                active[index] = dict(process=process, spec=spec, log=log,
                                     start=time.monotonic(), command=command, claim=claim)
            for index, job in list(active.items()):
                process = job['process']
                timeout = time.monotonic()-job['start'] > timeout_seconds
                if process.poll() is None and not timeout:
                    continue
                if timeout and process.poll() is None:
                    process.terminate(); process.wait(timeout=20)
                job['log'].close()
                if not path(job['spec']).exists():
                    failure = dict(spec=job['spec'], status='process_failure',
                        failures=dict(worker='timeout' if timeout else
                                      'exit '+str(process.returncode)),
                        worker_exit_code=process.returncode,
                        seconds=time.monotonic()-job['start'],
                        source_sha256=version['source_sha256'])
                    old.save_record(path(job['spec']), failure)
                atomic_json(job['claim'], dict(status='finished', pid=process.pid,
                    spec=job['spec'], worker_exit_code=process.returncode,
                    result=str(path(job['spec'])), time=time.time()))
                del active[index]
            finished = [s['index'] for s in specs if path(s).exists()]
            atomic_json(root/'controller.json', dict(status='running', phase=phase,
                pid=os.getpid(), policy_id=version['policy_id'],
                total=len(specs), finished=finished, pending=[s['index'] for s in pending],
                workers={str(i): dict(pid=j['process'].pid,
                    elapsed_seconds=time.monotonic()-j['start'])
                    for i, j in active.items()}, elapsed_seconds=time.monotonic()-started))
            if pending or active:
                time.sleep(2)
        records = [analysis.load(path(s)) for s in specs]
        if any(r.get('spec') != s for r, s in zip(records, specs)):
            raise ValueError('history file does not match frozen specification')
        calibration = analysis.load(root/'calibration.json') if (root/'calibration.json').exists() else None
        atomic_json(root/(phase+'_summary.json'), analysis.summarize(records, calibration))
        if phase == 'calibration':
            c = analysis.calibrate(records)
            c.update(policy_id=version['policy_id'],
                     input_sha256={str(path(s).relative_to(root)):
                                   sha256(path(s).read_bytes()).hexdigest() for s in specs})
            calibration_path = root/'calibration.json'
            if calibration_path.exists() and analysis.load(calibration_path) != old.serial(c):
                raise ValueError('existing calibration differs; preserve it')
            atomic_json(calibration_path, c)
        atomic_json(root/'controller.json', dict(status=phase+'_complete',
            phase=phase, pid=os.getpid(), policy_id=version['policy_id'],
            finished=[s['index'] for s in specs], total=len(specs),
            elapsed_seconds=time.monotonic()-started,
            next_step='review development' if phase == 'development' else
                      'independent evaluation' if phase == 'calibration' else
                      'physical audit, paired report, commit/push'))
    finally:
        # A controller exception does not kill running scientific children.
        children = {str(i): j['process'].pid for i, j in active.items()
                    if j['process'].poll() is None}
        if children:
            atomic_json(root/'orphan_workers.json', dict(workers=children,
                reason='controller stopped; inspect before any restart'))
        if lock.exists():
            lock.unlink()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', required=True)
    p.add_argument('--phase', required=True,
                   choices=('development', 'calibration', 'evaluation'))
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--limit', type=int)
    p.add_argument('--draws', type=int, default=256)
    p.add_argument('--replicates', type=int, default=4)
    p.add_argument('--noise-repeats', type=int, default=1)
    p.add_argument('--skip-clean-refit', action='store_true')
    args = p.parse_args()
    if args.workers < 1 or args.workers > 2:
        p.error('one or two workers allowed')
    run_phase(args.output, args.phase, workers=args.workers, limit=args.limit,
        draws=args.draws, replicates=args.replicates,
        noise_repeats=args.noise_repeats, clean_refit=not args.skip_clean_refit)


if __name__ == '__main__':
    main()
