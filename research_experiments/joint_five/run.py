"""Frozen 2x2 stage ablation on the recent four-network comparison matrix."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import shutil
import subprocess
import sys
from time import perf_counter, sleep
import traceback

for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'

from research_experiments.rebuild_benchmark.run import ROOT, read, write, sha

PARENT = ROOT/'outputs/paper_cases_lp_rebuild_20261031'
SETTINGS = dict(baseline=dict(pre_joint=False, tail_joint=False), rnj_pre=dict(pre_joint=True, tail_joint=False),
                wzzt_tail=dict(pre_joint=False, tail_joint=True), both=dict(pre_joint=True, tail_joint=True))


def freeze(output):
    import gurobipy as gp
    import numpy as np
    import scipy
    output.mkdir(parents=True, exist_ok=False); old = read(PARENT/'protocol.json')
    jobs = old['evaluation']; assert len(jobs) == 120
    snapshot = output/'source'; shutil.copytree(PARENT/'source', snapshot)
    for name, digest in old['source_hashes'].items():
        assert sha(snapshot/name) == digest
    for file in Path(__file__).parent.glob('*.py'):
        target = snapshot/'research_experiments/joint_five'/file.name
        target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(file.read_bytes())
    catalog = read(PARENT/'catalog.json'); reused = {}
    for job in jobs:
        p = catalog[job]; common = PARENT/'common'/job/'result.json'
        assert read(common)['source_input_hashes'] == dict(train=sha(p['train']), validation=sha(p['validation']))
        target = output/'common'/job/'result.json'; target.parent.mkdir(parents=True)
        shutil.copy2(common, target); reused[job] = sha(target)
    write(output/'catalog.json', catalog)
    write(output/'protocol.json', dict(created_utc=datetime.now(timezone.utc).isoformat(), evaluation=jobs, settings=SETTINGS,
        matrix='108 main +12 stress; same four-case observations and training frontend as recent comparison',
        RNJ_semantics='complete raw RX75 0.16 RNJ layer as initial support; not bootstrap-two-boundary production default',
        small_terminal_threshold=5, residual_block_limit=5,
        pre='release <=5-terminal RNJ merges only if parent frontier remains <=5; jointly optimize all eligible regions once',
        tail='original WZZT until maximum residual arity <=5, then open eligible small merges and solve one joint MILP, then stop',
        skipped_pre_regions='an existing >5 polytomy is retained for the original WZZT stage and explicitly reported',
        search_seconds=20., pre_seconds=5., per_MIP_seconds=2., LP_seconds=5., relative_gain=.001,
        workers=4, worker_hard_seconds=180., floor=0., sparsity=0., output_resolution=.16,
        selection='common raw RNJ bank + own path + singleton fallback; .16 contraction, train refit, strict validation MAE',
        error_audit='FP/FN by <=5/>5 terminal size, origin, first wrong addition, and structurally blocked true clades; truth offline only',
        parameters_tuned=False, truth_used_for_selection=False, frontend_reused=reused,
        source_hashes={p.relative_to(snapshot).as_posix(): sha(p) for p in snapshot.rglob('*.py')},
        parent_protocol_sha256=sha(PARENT/'protocol.json'),
        environment=dict(python=sys.version, executable=sys.executable, numpy=np.__version__, scipy=scipy.__version__, gurobi=gp.gurobi.version())))
    print('FROZEN', len(jobs), 'inputs', len(jobs)*len(SETTINGS), 'workers', flush=True)


def worker(payload):
    from research_experiments.wzzt_paper.common import observed
    from research_experiments.rebuild_benchmark.engine import prepare
    from research_experiments.lp_rebuild.engine import FloorBank, select_output
    from .search import search
    p = read(payload); common = read(p['common']); then = perf_counter()
    assert common['source_input_hashes'] == dict(train=sha(p['train']), validation=sha(p['validation']))
    train = prepare(observed(p['train'], p['N'])); validation = prepare(observed(p['validation']))
    bank = FloorBank(train, common['pool'], common['calibration'])
    try:
        trajectory = search(bank, common['initial'], **p['config'])
        selection_start = perf_counter(); star = bank.fit([(i,) for i in range(train.n)])
        selected = select_output(bank, [*common['bank'], *trajectory['path'], star], validation)
        selected_path = select_output(bank, [*trajectory['path'], star], validation)
        index = selected['index']; count = len(common['bank'])
        origin = 'complete_rnj' if index<count else 'path' if index<count+len(trajectory['path']) else 'singleton'
        write(p['result'], dict(job=p['job'], method=p['method'], config=p['config'], trajectory=trajectory,
            selected=selected, selected_path=selected_path, selected_origin=origin,
            LP=bank.lp.solves, LP_failures=bank.failures, selection_seconds=perf_counter()-selection_start,
            elapsed_seconds=perf_counter()-then, common_seconds=common['seconds'],
            truth_used=False, test_opened=False, common_sha256=sha(p['common']),
            protocol_sha256=p['protocol_sha256'], input_hashes=common['source_input_hashes']))
    finally:
        bank.close()


def run(output):
    protocol = read(output/'protocol.json'); catalog = read(output/'catalog.json'); source = output/'source'
    env = os.environ.copy(); env['PYTHONPATH'] = str(source)+os.pathsep+str(source/'rnj_wzzt_core')
    tasks = []
    for job in protocol['evaluation']:
        meta = catalog[job]
        for method, config in protocol['settings'].items():
            target = output/'evaluation'/job/method/'result.json'
            if target.exists() or target.with_name('error.json').exists():
                continue
            tasks.append(dict(job=job, method=method, config=config, N=meta['N'], train=meta['train'], validation=meta['validation'],
                              common=str(output/'common'/job/'result.json'), result=str(target), protocol_sha256=sha(output/'protocol.json')))
    pending = list(tasks); active = {}; done = 0; then = perf_counter()
    while pending or active:
        while pending and len(active)<protocol['workers']:
            p = pending.pop(0); target = Path(p['result']); payload = target.with_name('payload.json'); write(payload, p)
            log = target.with_name('worker.log').open('w', encoding='utf-8')
            proc = subprocess.Popen([sys.executable, '-B', '-m', 'research_experiments.joint_five.run', 'worker', str(payload)],
                cwd=source, env=env, stdout=log, stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            active[str(target)] = proc, log, perf_counter(), p
        for key, (proc, log, start, p) in list(active.items()):
            target = Path(p['result'])
            if proc.poll() is None and perf_counter()-start>protocol['worker_hard_seconds']:
                proc.kill(); proc.wait(); write(target.with_name('timeout.json'), dict(reason='worker hard timeout'))
            if proc.poll() is not None:
                log.close()
                if not target.exists() and not target.with_name('error.json').exists():
                    write(target.with_name('error.json'), dict(exit_code=proc.returncode))
                done += 1; del active[key]
                print('JOINT5', done, '/', len(tasks), p['job'], p['method'], target.exists(), flush=True)
        write(output/'progress.json', dict(planned=len(tasks), finished=done, active=len(active), pending=len(pending), seconds=perf_counter()-then))
        if active:
            sleep(.5)
    returned = sum((output/'evaluation'/j/m/'result.json').exists() for j in protocol['evaluation'] for m in protocol['settings'])
    write(output/'completion.json', dict(planned=len(protocol['evaluation'])*len(protocol['settings']), returned=returned, seconds=perf_counter()-then))


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('command', choices=('freeze', 'run', 'worker', 'analyze'))
    parser.add_argument('path'); args = parser.parse_args(); path = Path(args.path).resolve()
    if args.command == 'worker':
        try:
            worker(path)
        except Exception:
            write(Path(read(path)['result']).with_name('error.json'), dict(traceback=traceback.format_exc())); raise
    elif args.command == 'freeze':
        freeze(path)
    elif args.command == 'run':
        run(path)
    else:
        from .analyze import analyze
        analyze(path)


if __name__ == '__main__':
    main()
