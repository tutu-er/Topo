"""Freeze and run the predeclared 24-input deletion/restart comparison."""
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
from research_experiments.local_rebuild.model import canonical
from .model import select_block

PARENT = ROOT / 'outputs/rnj_four_methods_20261031'
LONG = ROOT / 'outputs/rnj_union_long_20261031'
METHODS = ('continue', 'drop_refit', 'drop_resume', 'mask_resume')
LIMITS = dict(total=14400., single_MIP=3600., LP=300., post_reserve=600., mip_gap=.0001)


def freeze(output):
    import numpy as np, scipy, gurobipy as gp
    if output.exists(): raise FileExistsError('Do not overwrite a frozen study.')
    old = read(PARENT / 'protocol.json'); catalog = read(PARENT / 'catalog.json')
    jobs = [j for j in old['jobs'] if catalog[j]['panel'] == 'main' and catalog[j]['N'] == 48 and catalog[j]['bundle'] in (2, 3)]
    assert len(jobs) == 24 and read(PARENT / 'analysis/verification.json')['passed']
    source_protocol = read(LONG / 'protocol.json')
    for name, digest in source_protocol['source_hashes'].items(): assert sha(LONG / 'source' / name) == digest
    output.mkdir(parents=True); shutil.copytree(LONG / 'source', output / 'source')
    for file in Path(__file__).parent.glob('*.py'):
        dest = output / 'source/research_experiments/flexible_exit' / file.name
        dest.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(file, dest)
    from research_experiments.iterative_exit.zero import snapshot_zero_support
    snapshot_zero_support(output/'source')
    write(output / 'catalog.json', {j: catalog[j] for j in jobs})
    references = {}; starts = {}; eligible = 0
    for job in jobs:
        common_file = PARENT / 'common' / job / 'result.json'
        result_file = PARENT / 'evaluation' / job / 'small3_original_exit/result.json'
        common, previous = read(common_file), read(result_file)
        point = previous['exit']['point']; target = select_block(point, len(common['labels'])); eligible += target is not None
        directory = output / 'parents' / job; directory.mkdir(parents=True)
        # Training-only start, separate from the historical validation finalists.
        file = directory / 'start.json'
        write(file, dict(point=point, block=target, pool=common['pool'], calibration=common['calibration'],
            reference=previous['trajectory']['reference_MAE'], labels=common['labels'],
            upstream_seconds=common['seconds']+previous['elapsed_seconds'], training_only=True))
        write(directory / 'baseline_choices.json', previous['choices']['strict_0.24'])
        starts[job] = sha(file)
        references[job] = {str(common_file): sha(common_file), str(result_file): sha(result_file)}
    manifests = {}
    for name in ('rnj_four_methods_20261031', 'rnj_nj_portfolio_20261031', 'rnj_union_long_20261031'):
        path = ROOT / 'docs' / name / 'manifest.json'; m = read(path)
        for file, record in m['files'].items(): assert sha(path.parent / file) == record['sha256']
        manifests[str(path)] = dict(sha256=sha(path), files=m['files'])
    write(output / 'protocol.json', dict(created_utc=datetime.now(timezone.utc).isoformat(), jobs=jobs,
        methods=METHODS, planned=96, eligible=eligible, workers=8, limits=LIMITS,
        scope='four networks x bundles2/3 x three voltage noise levels x N48; no truth-based filtering',
        stage='after saved small3_original_exit training endpoint; before final contraction/validation',
        block_rule='largest retained proper support with >5 terminals; include zero weights; lexicographic tie; at most one',
        candidates='continue/drop_resume/mask_resume use identical original unknown-support extension and five-block tail; no repeated global exit',
        temporary_uphill='deletion may raise L or J; proposals kept for validation comparison; only resumed additions must pass original J gate',
        selection='raw endpoint; own proposal path .24 contraction/train refit/strict validation; guarded adds old mainline final tree, which may contain the masked block',
        bounds=[0., 2.], coefficient=.003, resolution=.24, retuning=False,
        no_new_physical_floor=True, truth_used_in_search=False, test_opened_by_worker=False,
        timing='recorded common+old mainline cost charged once per method; measured new stages; component sum, not integrated replay',
        start_hashes=starts, reference_hashes=references, parent_manifests=manifests,
        input_hashes={j: {s: old['input_hashes'][j][s] for s in ('train', 'validation', 'test')} for j in jobs},
        baseline_choices_hashes={j: sha(output / 'parents' / j / 'baseline_choices.json') for j in jobs},
        source_hashes={f.relative_to(output / 'source').as_posix(): sha(f) for f in (output / 'source').rglob('*.py')},
        plan_sha256=sha(ROOT / 'docs/flexible_exit_20261101/PLAN.md'),
        environment=dict(python=sys.version, numpy=np.__version__, scipy=scipy.__version__, gurobi=gp.gurobi.version())))
    write(output / 'state.json', dict(stage='frozen', planned=96, eligible=eligible))
    print(f'Frozen 24 inputs x4 new branches; retained large block exists in {eligible}/24.', flush=True)


def worker(payload):
    import numpy as np
    from research_experiments.wzzt_paper.common import observed
    from research_experiments.rebuild_benchmark.engine import prepare, loss
    from research_experiments.joint_exit.selection import library, choose
    from research_experiments.rnj_union_long.model import criterion
    from .model import MaskBank, replay
    p = read(payload); then = perf_counter(); target = Path(p['result'])
    assert sha(p['start']) == p['start_sha256']
    saved = read(p['start']); block = saved['block']; mode = p['method']; limits = p['limits']
    assert {s: sha(p[s]) for s in ('train', 'validation')} == p['input_hashes']
    data = prepare(observed(p['train'], p['N'])); validation = prepare(observed(p['validation']))
    assert abs(loss(data, saved['point'])-saved['point']['objective']) < 1e-8
    assert select_block(saved['point'], data.n) == (tuple(block) if block else None)
    upstream = saved['upstream_seconds']; deadline = then+limits['total']-upstream
    structural_deadline = deadline-limits['post_reserve']; penalty = .003*saved['reference']
    forbidden = canonical([block]) if mode == 'mask_resume' and block else ()
    pool = canonical(C for C in [*saved['pool'], *saved['point']['supports']] if tuple(C) not in forbidden)
    bank = MaskBank(data, pool, saved['calibration'], forbidden=forbidden, deadline=deadline, seconds=limits['LP'])
    applied = mode == 'continue' or block is not None
    def heartbeat(event):
        write(target.with_name('progress.json'), dict(method=mode, seconds=perf_counter()-then, **event))
    try:
        family = canonical(C for C in saved['point']['supports'] if mode == 'continue' or not block or tuple(C) != tuple(block))
        initial = bank.fit(family, preserve=True)
        if mode != 'continue' and block:
            assert initial['objective'] >= saved['point']['objective']-2e-7
        if mode != 'drop_refit' and applied:
            trajectory = replay(bank, initial, penalty=penalty, deadline=structural_deadline, per_mip=limits['single_MIP'],
                heartbeat=heartbeat, checkpoint=lambda q: write(target.with_name('checkpoint.json'), q))
        else:
            trajectory = dict(point=initial, path=[initial], events=[], stop='LP_only' if applied else 'no_eligible_block', forbidden=forbidden)
        search_seconds = perf_counter()-then
        assert all(not set(canonical(point['supports'])) & set(forbidden) for point in trajectory['path'])
        # Open only historical validation choices after the training search ends.
        assert sha(p['baseline_choices']) == p['baseline_choices_sha256']
        baseline = read(p['baseline_choices'])['bank']
        rows = library(bank, trajectory['path'], validation, resolution=.24)
        proposal = choose(rows, 'strict'); proposal['source'] = 'proposal'
        guarded = choose([dict(baseline, index=0, source='historical'), dict(proposal, index=1)], 'strict')
        assert guarded['validation'] <= baseline['validation']+1e-12
        elapsed = perf_counter()-then; endpoint = trajectory['point']
        write(target, dict(job=p['job'], method=mode, block=block, applied=applied, initial=initial, trajectory=trajectory,
            proposal=proposal, guarded=guarded, historical_endpoint_choice=rows[0] if mode == 'continue' else None,
            reference=saved['reference'], penalty=penalty,
            before_L=saved['point']['objective'], before_J=criterion(saved['point'], data.n, penalty),
            drop_L=initial['objective'], drop_J=criterion(initial, data.n, penalty),
            after_L=endpoint['objective'], after_J=criterion(endpoint, data.n, penalty),
            block_retained=bool(block and tuple(block) in canonical(endpoint['supports'])),
            fresh_seconds=elapsed, upstream_seconds=upstream, total_seconds=upstream+elapsed,
            search_seconds=search_seconds, post_seconds=elapsed-search_seconds,
            total_cap_exceeded=upstream+elapsed > limits['total']+1.,
            LP=bank.lp.solves, LP_failures=bank.failures, truth_used=False, test_opened=False,
            protocol_sha256=p['protocol_sha256'], start_sha256=p['start_sha256'], input_hashes=p['input_hashes']))
    finally: bank.close()


def run(output):
    p = read(output / 'protocol.json'); catalog = read(output / 'catalog.json')
    for name, digest in p['source_hashes'].items(): assert sha(output / 'source' / name) == digest
    env = os.environ.copy(); env['PYTHONPATH'] = str(output / 'source')+os.pathsep+str(output / 'source/rnj_wzzt_core')
    tasks = []
    for job in p['jobs']:
        for method in METHODS:
            result = output / 'jobs' / job / method / 'result.json'
            if result.exists() or result.with_name('error.json').exists(): continue
            d = catalog[job]
            tasks.append(dict(job=job, method=method, train=d['train'], validation=d['validation'], N=d['N'],
                input_hashes={s: p['input_hashes'][job][s] for s in ('train', 'validation')}, limits=p['limits'],
                start=str(output / 'parents' / job / 'start.json'), start_sha256=p['start_hashes'][job],
                baseline_choices=str(output / 'parents' / job / 'baseline_choices.json'), baseline_choices_sha256=p['baseline_choices_hashes'][job],
                result=str(result), protocol_sha256=sha(output / 'protocol.json')))
    active = {}; done = 0
    while tasks or active:
        while tasks and len(active) < p['workers']:
            task = tasks.pop(0); result = Path(task['result']); payload = result.with_name('payload.json'); write(payload, task)
            log = result.with_name('worker.log').open('w', encoding='utf-8')
            proc = subprocess.Popen([sys.executable, '-B', '-m', 'research_experiments.flexible_exit.run', 'worker', str(payload)],
                cwd=output / 'source', env=env, stdout=log, stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            active[(task['job'], task['method'])] = proc, log, result
        for key, (proc, log, result) in list(active.items()):
            if proc.poll() is not None:
                log.close(); del active[key]; done += 1
                if not result.exists() and not result.with_name('error.json').exists(): write(result.with_name('error.json'), dict(exit_code=proc.returncode))
                print('finished', done, key, result.exists(), flush=True)
        write(output / 'progress.json', dict(finished_this_run=done, pending=len(tasks), active=[list(k) for k in active]))
        if active: sleep(.5)
    missing = [dict(job=j, method=m) for j in p['jobs'] for m in METHODS if not (output / 'jobs' / j / m / 'result.json').exists()]
    write(output / 'completion.json', dict(planned=96, returned=96-len(missing), failures=missing))
    write(output / 'state.json', dict(stage='solves_complete', returned=96-len(missing), failures=missing))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('command', choices=('freeze', 'run', 'worker')); parser.add_argument('path')
    args = parser.parse_args(); path = Path(args.path).resolve()
    try: {'freeze': freeze, 'run': run, 'worker': worker}[args.command](path)
    except Exception:
        if args.command == 'worker': write(Path(read(path)['result']).with_name('error.json'), dict(traceback=traceback.format_exc()))
        raise
