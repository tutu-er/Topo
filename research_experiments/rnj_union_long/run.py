"""Frozen 120-input long-budget reconstruction, simplicity and pure-WZZT study."""
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
from research_experiments.local_rebuild.model import canonical
from .model import training_candidates

FOUR = ROOT / 'outputs/rnj_four_methods_20261031'
PORT = ROOT / 'outputs/rnj_nj_portfolio_20261031'
METHODS = ('union_count', 'union_plain', 'simple_count', 'pure_wzzt')
LIMITS = dict(total=14400., single_pure_MIP=3600., LP=300., post_reserve=600., mip_gap=.0001)


def freeze(output):
    import numpy as np, scipy, gurobipy as gp
    if output.exists(): raise FileExistsError('New immutable output directory required.')
    parent = read(FOUR / 'protocol.json'); portfolio = read(PORT / 'protocol.json')
    assert read(FOUR / 'analysis/verification.json')['passed'] and read(PORT / 'analysis/verification.json')['passed']
    output.mkdir(parents=True)
    shutil.copy2(FOUR / 'catalog.json', output / 'catalog.json')
    shutil.copytree(PORT / 'source', output / 'source')
    for file in Path(__file__).parent.glob('*.py'):
        target = output / 'source/research_experiments/rnj_union_long' / file.name
        target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(file, target)
    manifests = {}
    for folder in ('rnj_four_methods_20261031', 'rnj_nj_portfolio_20261031'):
        file = ROOT / 'docs' / folder / 'manifest.json'; manifests[str(file)] = dict(sha256=sha(file), files=read(file)['files'])
        for name, item in manifests[str(file)]['files'].items(): assert sha(file.parent / name) == item['sha256']
    pool_hashes, calibration, references = {}, {}, {}
    for job in parent['jobs']:
        common_file = FOUR / 'common' / job / 'result.json'; common = read(common_file)
        full_file = FOUR / 'evaluation' / job / 'full_rnj_exit/result.json'
        sparse_file = FOUR / 'evaluation' / job / 'small3_original_exit/result.json'
        nj_file = PORT / 'jobs' / job / 'result.json'
        full, sparse, nj = read(full_file), read(sparse_file), read(nj_file)
        pool, warm = training_candidates(common, full, sparse, nj['NJ_family'])
        # Validation choices are checked for representability only, never used to add candidates.
        for record in (full, sparse):
            for resolution in (.16, .24):
                assert set(canonical(record['choices'][f'strict_{resolution:g}']['bank']['point']['supports'])) <= set(pool)
        simple = canonical([*common['initial'], *nj['NJ_family'], *((i,) for i in range(len(common['labels']))), tuple(range(len(common['labels'])))])
        upstream = common['seconds']+full['elapsed_seconds']+sparse['elapsed_seconds']+nj['marginal_seconds']
        file = output / 'pools' / job / 'result.json'
        write(file, dict(pool=pool, warm=warm, calibration=common['calibration'], reference=full['trajectory']['reference_MAE'],
                         simple_reference=simple, upstream_recorded_seconds=upstream,
                         training_only_construction=True, choices_used_to_generate=False))
        pool_hashes[job] = sha(file); calibration[job] = common['calibration']
        references[job] = {str(f): sha(f) for f in (common_file, full_file, sparse_file, nj_file)}
    prev = output / 'previous'; prev.mkdir()
    for source, name in [(FOUR / 'analysis/independent_LP.json', 'four_profiles.json'),
                         (PORT / 'analysis/independent_LP.json', 'portfolio_profiles.json'),
                         (ROOT / 'docs/rnj_nj_portfolio_20261031/per_model.csv', 'per_model.csv')]:
        shutil.copy2(source, prev / name)
    write(output / 'protocol.json', dict(created_utc=datetime.now(timezone.utc).isoformat(), jobs=parent['jobs'], methods=METHODS,
        planned=480, workers=8, pure_parallel_while_other_tasks_pending=4, limits=LIMITS, coefficient=.003, resolutions=[.16, .24], primary_resolution=.24,
        union_pool='all raw training RNJ-bank supports + training path/end/exit supports of full and sparse methods + raw NJ(.16); no validation-selected supports used to construct pool',
        union_count='one joint optional-support MILP, training MAE+.003*f0*internal count; no RNJ fallback',
        union_plain='same pool and one joint MILP, no count penalty; fixed ablation, not tuning',
        simple_count='fresh single RX75(.16) RNJ and NJ trees, one union selection MILP; no WZZT generators, local scans, tail or extra exit',
        pure_wzzt='fresh leaf/full basis, one unrestricted support per step, training MAE, no NJ/RNJ support or candidate pool, no tail/exit/public-bank fallback',
        selection='fixed .24 primary/.16 sensitivity contraction and train refit; pure validates its own path, joint has only its final tree; raw endpoints also reported',
        budget='14400 seconds total per method/input; reused upstream measured seconds charged to union budget; 600 seconds reserved for closure/validation; pure single MIP3600, joint uses remaining structural deadline; each LP<=300 and remaining total',
        timing='fresh marginal wall measured; union total adds recorded common/full/sparse/NJ generation once; cumulative recorded cost is not a fresh integrated end-to-end timing',
        strategy='candidate generation -> one laminar sparse L1 fit -> fixed contraction/refit and independent validation; direct simple_count tests whether WZZT generators are needed',
        retuning=False, truth_used_in_search=False, test_used_in_search=False, hard_impedance_floor=None,
        pool_hashes=pool_hashes, calibration_references=calibration, reference_hashes=references,
        input_hashes=parent['input_hashes'], previous_manifests=manifests,
        previous_hashes={f.name: sha(f) for f in prev.iterdir()},
        source_hashes={f.relative_to(output / 'source').as_posix(): sha(f) for f in (output / 'source').rglob('*.py')},
        environment=dict(python=sys.version, numpy=np.__version__, scipy=scipy.__version__, gurobi=gp.gurobi.version())))
    write(output / 'state.json', dict(stage='frozen', running=False, planned=480))
    print('FROZEN long-budget study: 120 inputs x4 methods; total cap14400s each', flush=True)


def fresh_front(obs, *, make_trees=False):
    import numpy as np, pandas as pd
    from research_experiments.rebuild_benchmark.engine import prepare
    from rnj_wzzt.estimation.multiscenario import fit_projected_sensitivity
    from research_experiments.rnj.rx75_edge_floor import calibrate_rx75_floor
    then = perf_counter(); data = prepare(obs); labels = list(obs[0].columns)
    R, X, r2, condition = fit_projected_sensitivity([dict(P_terminal=obs[0], Q_terminal=obs[1],
        drop_target=pd.DataFrame(data.target[0], columns=labels))])
    cal = calibrate_rx75_floor(R, X)
    result = dict(calibration=cal, R2=r2, condition=condition, tree_generators=0)
    if make_trees:
        from rnj_wzzt.graph.rooted_neighbor_joining import rooted_neighbor_joining
        from rnj_wzzt.graph.rooted_hierarchy import rooted_clades
        from research_experiments.rnj.rooted_study_baselines import infer_classical_nj
        kernel = .75*R/cal['mu_r']+.25*X/cal['mu_x']; positions = {v: i for i, v in enumerate(labels)}
        rnj = rooted_neighbor_joining(kernel, np.diag(kernel), labels, 1,
            group_tolerance=.16*max(float(np.median(np.diag(kernel))), 1e-12))
        nj = infer_classical_nj(kernel, kernel, labels, 1, collapse_factor=.16)
        basis = [*((i,) for i in range(len(labels))), tuple(range(len(labels)))]
        families = [canonical([*basis, *(tuple(positions[v] for v in C) for C in rooted_clades(tree.edges, 1, labels))]) for tree in (rnj, nj)]
        result.update(families=families, tree_generators=2)
    result['seconds'] = perf_counter()-then
    return data, result


def worker(payload):
    import numpy as np
    from research_experiments.wzzt_paper.common import observed
    from research_experiments.rebuild_benchmark.engine import prepare
    from research_experiments.joint_exit.selection import library, choose
    from .model import DeadlineBank, pure_search, joint_rebuild
    p = read(payload); then = perf_counter(); target = Path(p['result']); method = p['method']
    assert {s: sha(p[s]) for s in ('train', 'validation')} == p['input_hashes']
    front, pooled = None, None
    if method.startswith('union_'):
        assert sha(p['pool_file']) == p['pool_sha256']
        pooled = read(p['pool_file']); upstream = pooled['upstream_recorded_seconds']
    else: upstream = 0.
    deadline = then+p['limits']['total']-upstream
    structural_deadline = deadline-p['limits']['post_reserve']
    if structural_deadline <= then: raise TimeoutError('Recorded upstream cost already exhausts total budget.')
    obs = observed(p['train'], p['N']); validation = prepare(observed(p['validation']))
    if method.startswith('union_'):
        data = prepare(obs); pool = pooled['pool']; cal = pooled['calibration']; reference = pooled['reference']
    else:
        data, front = fresh_front(obs, make_trees=method == 'simple_count'); cal = front['calibration']
        assert np.allclose([cal[k] for k in ('a', 'b', 'median_root_depth')],
                           [p['calibration_reference'][k] for k in ('a', 'b', 'median_root_depth')], atol=1e-7, rtol=1e-6)
        if method == 'pure_wzzt': pool = canonical([*((i,) for i in range(data.n)), tuple(range(data.n))])
        else: pool = canonical(C for family in front['families'] for C in family)
        reference = None
    bank = DeadlineBank(data, pool, cal, deadline=deadline, seconds=p['limits']['LP'])
    def heartbeat(event):
        write(target.with_name('progress.json'), dict(method=method, stage='MILP', elapsed=perf_counter()-then,
                                                     cumulative=upstream+perf_counter()-then, **event))
    try:
        if method == 'pure_wzzt':
            trajectory = pure_search(bank, structural_deadline=structural_deadline, single_mip=p['limits']['single_pure_MIP'],
                heartbeat=heartbeat, checkpoint=lambda t: write(target.with_name('checkpoint.json'), dict(trajectory=t, truth_used=False)))
            joint = None; path = trajectory['path']; endpoint = trajectory['point']; penalty = 0.
        else:
            if method == 'simple_count':
                warm = [bank.fit(family, preserve=True) for family in front['families']]
                reference = warm[0]['objective']
            else: warm = pooled['warm']
            penalty = 0. if method == 'union_plain' else .003*reference
            joint = joint_rebuild(bank, pool, warm, penalty=penalty, deadline=structural_deadline, heartbeat=heartbeat)
            trajectory = None; endpoint = joint['point']; path = [endpoint]
        solve_seconds = perf_counter()-then
        choices = {}
        star = bank.fit([(i,) for i in range(data.n)]) if method == 'pure_wzzt' else None
        for resolution in (.16, .24):
            candidates = [*path, star] if star is not None else path
            rows = library(bank, candidates, validation, resolution=resolution)
            chosen = choose(rows, 'strict')
            endpoint_choice = library(bank, [endpoint], validation, resolution=resolution)[0]
            choices[str(resolution)] = dict(selected=chosen, endpoint=endpoint_choice)
        elapsed = perf_counter()-then
        write(target, dict(job=p['job'], method=method, trajectory=trajectory, joint=joint, choices=choices,
            raw_endpoint=endpoint, front=front, calibration=cal, reference_MAE=reference, penalty=penalty,
            source_generation_recorded=upstream, fresh_seconds=elapsed, cumulative_seconds=upstream+elapsed,
            search_and_front_seconds=solve_seconds, post_seconds=elapsed-solve_seconds,
            declared_total_cap=p['limits']['total'], total_cap_exceeded=upstream+elapsed>p['limits']['total']+1.,
            LP=bank.lp.solves, LP_failures=bank.failures, truth_used=False, test_opened=False,
            protocol_sha256=p['protocol_sha256'], input_hashes=p['input_hashes'], pool_hash=p.get('pool_sha256')))
    finally: bank.close()


def run(output):
    protocol = read(output / 'protocol.json'); catalog = read(output / 'catalog.json')
    env = os.environ.copy(); env['PYTHONPATH'] = str(output / 'source')+os.pathsep+str(output / 'source/rnj_wzzt_core')
    tasks = []
    # Interleave methods so candidate-pool solves continue alongside the pure search.
    for job in protocol['jobs']:
        meta = catalog[job]
        for method in METHODS:
            result = output / 'jobs' / job / method / 'result.json'
            if result.exists() or result.with_name('error.json').exists(): continue
            p = dict(job=job, method=method, N=meta['N'], train=meta['train'], validation=meta['validation'],
                     input_hashes={s: protocol['input_hashes'][job][s] for s in ('train', 'validation')},
                     calibration_reference=protocol['calibration_references'][job], limits=protocol['limits'],
                     result=str(result), protocol_sha256=sha(output / 'protocol.json'))
            if method.startswith('union_'):
                p.update(pool_file=str(output / 'pools' / job / 'result.json'), pool_sha256=protocol['pool_hashes'][job])
            tasks.append(p)
    active, done = {}, 0
    write(output / 'state.json', dict(stage='long-budget structure searches', running=True, planned=480))
    while tasks or active:
        while tasks and len(active) < protocol['workers']:
            index = 0
            if sum(k[1] == 'pure_wzzt' for k in active) >= protocol['pure_parallel_while_other_tasks_pending']:
                index = next((i for i, t in enumerate(tasks) if t['method'] != 'pure_wzzt'), 0)
            task = tasks.pop(index); result = Path(task['result']); payload = result.with_name('payload.json'); write(payload, task)
            log = result.with_name('worker.log').open('w', encoding='utf-8')
            proc = subprocess.Popen([sys.executable, '-B', '-m', 'research_experiments.rnj_union_long.run', 'worker', str(payload)],
                cwd=output / 'source', env=env, stdout=log, stderr=subprocess.STDOUT, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            active[(task['job'], task['method'])] = proc, log, perf_counter(), result
        for key, (proc, log, started, result) in list(active.items()):
            if proc.poll() is None and perf_counter()-started > protocol['limits']['total']+60:
                proc.kill(); proc.wait(); write(result.with_name('error.json'), dict(reason='total method worker guard exceeded'))
            if proc.poll() is not None:
                log.close(); del active[key]; done += 1
                if not result.exists() and not result.with_name('error.json').exists(): write(result.with_name('error.json'), dict(exit_code=proc.returncode))
                print('union-long', done, *key, result.exists(), flush=True)
        write(output / 'progress.json', dict(finished_this_run=done, active=[dict(job=k[0], method=k[1], seconds=perf_counter()-v[2]) for k, v in active.items()], pending=len(tasks)))
        if active: sleep(1.)
    failed = [dict(job=j, method=m) for j in protocol['jobs'] for m in METHODS if not (output / 'jobs' / j / m / 'result.json').exists()]
    write(output / 'completion.json', dict(planned=480, returned=480-len(failed), failures=failed))
    write(output / 'state.json', dict(stage='search complete', running=False, returned=480-len(failed), planned=480))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('command', choices=('freeze', 'run', 'worker')); parser.add_argument('path'); args = parser.parse_args()
    target = Path(args.path).resolve()
    try:
        if args.command == 'freeze': freeze(target)
        elif args.command == 'worker': worker(target)
        else: run(target)
    except Exception:
        if args.command == 'worker': write(Path(read(target)['result']).with_name('error.json'), dict(traceback=traceback.format_exc()))
        else: write(target / 'state.json', dict(stage='needs attention', running=False, error=traceback.format_exc()))
        raise
