"""Complete the four requested methods without replacing earlier records."""
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

PARENT = ROOT / 'outputs/rnj_small_seed_20261031'
METHODS = ('ordinary', 'full_rnj_exit', 'small3_coarse_exit', 'small3_original_exit')
LABELS = dict(ordinary='普通RNJ＋WZZT', full_rnj_exit='完整RNJ＋联合修正＋退出',
              small3_coarse_exit='少量RNJ小块＋大块优先WZZT＋退出',
              small3_original_exit='少量RNJ小块＋原始WZZT步骤＋末端枚举＋退出')
LIMITS = dict(prefix=180., per_mip=30., rnj_pre=60., total=300., LP=30., hard_worker=900.)


def ordinary_search(bank, initial, limits):
    """The original unpenalized one-support extension, without either joint stage."""
    from research_experiments.joint_five.search import search
    return search(bank, initial, pre_joint=False, tail_joint=False,
                  seconds=limits['total'], per_mip=limits['per_mip'], relative=.001)


def reuse_source(job, method):
    if method in ('full_rnj_exit', 'small3_coarse_exit'):
        old_method = 'full_rnj' if method == 'full_rnj_exit' else 'small3'
        return PARENT / 'long' / job / old_method / 'result.json', 'previous_long'
    if method == 'small3_original_exit' and (PARENT / 'base' / job / 'result.json').exists():
        return PARENT / 'base' / job / 'result.json', 'previous_base'
    return None, 'fresh'


def freeze(output):
    import numpy as np
    import scipy
    import gurobipy as gp
    if output.exists():
        raise FileExistsError('New experiment directory required; do not overwrite a protocol.')
    old = read(PARENT / 'protocol.json')
    long = read(PARENT / 'long_protocol.json')
    base = read(PARENT / 'base_protocol.json')
    assert long['limits'] == base['limits'] == LIMITS
    for receipt in ('long_verification.json', 'base_verification.json'):
        audit = read(PARENT / 'analysis' / receipt)
        assert audit['passed'] and not audit['failures']
    # Verify both historical source snapshots before accepting their records.
    for protocol, folder in ((long, 'long_source'), (base, 'base_source')):
        for name, digest in protocol['source_hashes'].items():
            assert sha(PARENT / folder / name) == digest, (folder, name)
    output.mkdir(parents=True)
    shutil.copy2(PARENT / 'catalog.json', output / 'catalog.json')
    for name in ('long_protocol.json', 'base_protocol.json'):
        shutil.copy2(PARENT / name, output / ('previous_' + name))
    source = output / 'source'
    shutil.copytree(PARENT / 'base_source', source)
    for file in Path(__file__).parent.glob('*.py'):
        dest = source / 'research_experiments/rnj_four_methods' / file.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(file, dest)
    cat = read(output / 'catalog.json')
    common_hashes, input_hashes, reuse = {}, {}, []
    for job in old['evaluation']:
        file = output / 'common' / job / 'result.json'
        file.parent.mkdir(parents=True)
        shutil.copy2(PARENT / 'common' / job / 'result.json', file)
        common = read(file)
        common_hashes[job] = sha(file)
        input_hashes[job] = {k: sha(cat[job][k]) for k in ('train', 'validation', 'test')}
        assert common['source_input_hashes'] == {k: input_hashes[job][k] for k in ('train', 'validation')}
        for method in METHODS:
            previous, origin = reuse_source(job, method)
            if previous is None:
                continue
            record = read(previous)
            assert record['common_sha256'] == common_hashes[job]
            protocol_name = 'long_protocol.json' if origin == 'previous_long' else 'base_protocol.json'
            assert record['protocol_sha256'] == sha(PARENT / protocol_name)
            assert not record['truth_used'] and not record['test_opened']
            if 'limits' in record:
                assert record['limits'] == LIMITS
            dest = output / 'evaluation' / job / method / 'result.json'
            dest.parent.mkdir(parents=True)
            shutil.copy2(previous, dest)
            receipt = dict(job=job, method=method, origin=origin, source_path=str(previous),
                           result_sha256=sha(previous), original_protocol_sha256=sha(PARENT / protocol_name),
                           common_sha256=common_hashes[job])
            write(dest.with_name('provenance.json'), receipt)
            reuse.append(receipt)
    assert len(old['evaluation']) == 120 and len(reuse) == 264
    # Retain the original independent LP receipts as a verified fit cache.
    cached = output / 'previous_audits'
    cached.mkdir()
    for name in ('long_independent_LP.json', 'base_independent_LP.json',
                 'long_verification.json', 'base_verification.json'):
        shutil.copy2(PARENT / 'analysis' / name, cached / name)
    preservation = ROOT / 'docs/rnj_small_seed_20261031/delivery_manifest.json'
    write(output / 'protocol.json', dict(
        created_utc=datetime.now(timezone.utc).isoformat(), jobs=old['evaluation'], methods=METHODS, labels=LABELS,
        planned=480, reused=264, new_searches=216, workers=8, limits=LIMITS,
        ordinary=dict(initial='complete raw RNJ, matching recent comprehensive tests; not production bootstrap-two blocks',
                      pre_joint=False, tail_joint=False, exit=False, count_penalty=0., structural_seconds=300.),
        other_methods=dict(seed_mode='small3 where applicable', count_coefficient=.003,
                           prefix_seconds=180., tail_limit=5, hard_impedance_floor=None),
        selection='identical common RNJ bank + own path + singleton fallback; strict validation after contraction/refit',
        resolutions=[.16, .24], primary_resolution=.24, truth_used_for_tuning=False, retuning=False,
        comparisons='whole pipelines; ordinary retains historical unpenalized objective, extensions add count penalty and different candidate domains',
        reuse_policy='byte-identical completed 300-second records; same observations, frontend, limits and final selection; per-record provenance retained',
        common_hashes=common_hashes, input_hashes=input_hashes, reused_records=reuse,
        previous_protocol_hashes={n: sha(output / n) for n in ('previous_long_protocol.json', 'previous_base_protocol.json')},
        prior_audit_hashes={f.name: sha(f) for f in cached.iterdir()},
        previous_delivery_manifest=str(preservation), previous_delivery_manifest_sha256=sha(preservation),
        preserved_delivery_hashes=read(preservation)['files'],
        source_hashes={f.relative_to(source).as_posix(): sha(f) for f in source.rglob('*.py')},
        environment=dict(python=sys.version, executable=sys.executable, numpy=np.__version__, scipy=scipy.__version__,
                         gurobi=gp.gurobi.version(), logical_processors=os.cpu_count()))
    )
    write(output / 'state.json', dict(stage='frozen', running=False, completed_records=264, planned=480))
    print('FROZEN: 480 records = 264 verified reused + 216 new searches', flush=True)


def worker(payload):
    import numpy as np
    from research_experiments.wzzt_paper.common import observed
    from research_experiments.rebuild_benchmark.engine import prepare
    from research_experiments.lp_rebuild.engine import FloorBank
    from research_experiments.joint_exit.run import selections
    from research_experiments.rnj_small_seed.engine import prefix, exit_repair
    p = read(payload)
    common = read(p['common'])
    assert sha(p['common']) == p['common_sha256']
    assert common['source_input_hashes'] == {k: sha(p[k]) for k in ('train', 'validation')}
    limits = p['limits']
    then = perf_counter()
    data = prepare(observed(p['train'], p['N']))
    validation = prepare(observed(p['validation']))
    bank = FloorBank(data, common['pool'], common['calibration'], seconds=limits['LP'])
    bank.joint_exit_seed_pool = tuple(map(tuple, common['pool']))
    bank.seed_start_pool = bank.joint_exit_seed_pool
    try:
        if p['method'] == 'ordinary':
            t = ordinary_search(bank, common['initial'], limits)
            path = t['path']
            q, event = None, None
        elif p['method'] == 'small3_original_exit':
            t = prefix(bank, common['initial'], mode='small3', reference=p['reference'],
                       seconds=limits['prefix'], per_mip=limits['per_mip'], strategy='original')
            q, event = exit_repair(bank, common['initial'], t['point'], np.asarray(common['kernel']),
                                  reference=p['reference'], prefix_seconds=t['seconds'],
                                  warm_candidates=common['bank'], total_seconds=limits['total'])
            path = [*t['path'], q['point']] if q['accepted'] else t['path']
        else:
            raise ValueError('This method must use its verified previous result.')
        choices = selections(bank, common, path, validation, resolutions=(.16, .24))
        prefix_choices = choices if q is None else selections(bank, common, t['path'], validation, resolutions=(.16, .24))
        record = dict(job=p['job'], method=p['method'], trajectory=t, exit=q, event=event,
                      choices=choices, prefix_choices=prefix_choices, limits=limits,
                      protocol_sha256=p['protocol_sha256'], common_sha256=p['common_sha256'],
                      input_hashes=common['source_input_hashes'], truth_used=False, test_opened=False,
                      LP=bank.lp.solves, LP_failures=bank.failures, elapsed_seconds=perf_counter()-then,
                      postprocessing_excluded_from_structural_budget=True)
        write(p['result'], record)
        write(Path(p['result']).with_name('provenance.json'),
              dict(job=p['job'], method=p['method'], origin='fresh', result_sha256=sha(p['result']),
                   original_protocol_sha256=p['protocol_sha256'], common_sha256=p['common_sha256']))
    finally:
        bank.close()


def completion(output):
    p = read(output / 'protocol.json')
    rows = []
    for method in METHODS:
        good = [j for j in p['jobs'] if (output / 'evaluation' / j / method / 'result.json').exists()]
        failed = [dict(job=j, error=read(output / 'evaluation' / j / method / 'error.json')) for j in p['jobs']
                  if (output / 'evaluation' / j / method / 'error.json').exists() and j not in good]
        rows.append(dict(method=method, planned=120, returned=len(good), failures=failed))
    return dict(planned=480, returned=sum(r['returned'] for r in rows), methods=rows)


def run(output):
    p = read(output / 'protocol.json')
    cat = read(output / 'catalog.json')
    tasks = []
    for job in p['jobs']:
        common = output / 'common' / job / 'result.json'
        full = read(output / 'evaluation' / job / 'full_rnj_exit' / 'result.json')
        for method in METHODS:
            result = output / 'evaluation' / job / method / 'result.json'
            if result.exists() or result.with_name('error.json').exists():
                continue
            meta = cat[job]
            tasks.append(dict(job=job, method=method, N=meta['N'], train=meta['train'], validation=meta['validation'],
                              common=str(common), common_sha256=p['common_hashes'][job], reference=full['trajectory']['reference_MAE'],
                              limits=p['limits'], protocol_sha256=sha(output / 'protocol.json'), result=str(result)))
    source = output / 'source'
    env = os.environ.copy()
    env['PYTHONPATH'] = str(source) + os.pathsep + str(source / 'rnj_wzzt_core')
    active = {}
    done = 0
    then = perf_counter()
    while tasks or active:
        while tasks and len(active) < p['workers']:
            task = tasks.pop(0)
            result = Path(task['result'])
            payload = result.with_name('payload.json')
            write(payload, task)
            log = result.with_name('worker.log').open('w', encoding='utf-8')
            proc = subprocess.Popen([sys.executable, '-B', '-m', 'research_experiments.rnj_four_methods.run', 'worker', str(payload)],
                                    cwd=source, env=env, stdout=log, stderr=subprocess.STDOUT,
                                    creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            active[(task['job'], task['method'])] = proc, log, perf_counter(), task
        for key, (proc, log, start, task) in list(active.items()):
            result = Path(task['result'])
            if proc.poll() is None and perf_counter()-start > p['limits']['hard_worker']:
                proc.kill()
                proc.wait()
                write(result.with_name('error.json'), dict(reason='worker hard limit', seconds=perf_counter()-start))
            if proc.poll() is not None:
                log.close()
                del active[key]
                done += 1
                if not result.exists() and not result.with_name('error.json').exists():
                    write(result.with_name('error.json'), dict(exit_code=proc.returncode))
                print('four-methods', done, *key, result.exists(), flush=True)
        write(output / 'progress.json', dict(finished_new_this_run=done, active=len(active), pending=len(tasks),
                                            seconds=perf_counter()-then))
        if active:
            sleep(.5)
    receipt = completion(output)
    write(output / 'completion.json', receipt)
    write(output / 'state.json', dict(stage='search complete' if receipt['returned'] == 480 else 'search incomplete',
                                     running=False, completed_records=receipt['returned'], planned=480))


def pipeline(output):
    if not (output / 'protocol.json').exists():
        freeze(output)
    write(output / 'state.json', dict(stage='search', running=True, planned=480))
    run(output)
    write(output / 'state.json', dict(stage='independent verification', running=True, planned=480))
    from .analyze import analyze
    audit = analyze(output)
    write(output / 'state.json', dict(stage='complete', running=False, passed=audit['passed'],
                                     completed_records=read(output / 'completion.json')['returned'], planned=480))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('freeze', 'worker', 'run', 'pipeline'))
    parser.add_argument('path')
    args = parser.parse_args()
    target = Path(args.path).resolve()
    try:
        if args.command == 'freeze': freeze(target)
        elif args.command == 'worker': worker(target)
        elif args.command == 'run': run(target)
        else: pipeline(target)
    except Exception:
        if args.command == 'worker':
            write(Path(read(target)['result']).with_name('error.json'), dict(traceback=traceback.format_exc()))
        elif target.exists():
            write(target / 'state.json', dict(stage='needs attention', running=False, failed=True, traceback=traceback.format_exc()))
        raise
