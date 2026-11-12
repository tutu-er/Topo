"""Add classical NJ and combine existing complete predictions; no new MILP."""
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

PARENT = ROOT / 'outputs/rnj_four_methods_20261031'
BASES = ('full_rnj_exit', 'small3_original_exit')
NEW_METHODS = ('NJ_native', 'NJ_matched', 'two_strict', 'two_paired', 'three_strict', 'three_paired')


def nj_family(kernel, labels):
    """Use the existing root-anchored Q-criterion NJ, fixed geometry cut .16."""
    import numpy as np
    from research_experiments.rnj.rooted_study_baselines import infer_classical_nj
    from rnj_wzzt.graph.rooted_hierarchy import rooted_clades
    from research_experiments.local_rebuild.model import canonical
    positions = {v: i for i, v in enumerate(labels)}
    kernel = np.asarray(kernel)
    tree = infer_classical_nj(kernel, kernel, labels, 1, collapse_factor=.16)
    family = canonical([*((i,) for i in range(len(labels))), tuple(range(len(labels))),
                        *(tuple(positions[v] for v in C) for C in rooted_clades(tree.edges, 1, labels))])
    return family, tree


def evaluate_candidate(point, validation, source):
    import numpy as np
    from rnj_wzzt.estimation import laminar_l1_milp as core
    R, X = core.build_matrices_from_atoms(validation.n, point['supports'], point['r'], point['x'])
    errors = np.concatenate([np.mean(abs(y-p@R.T-q@X.T), axis=1)
                             for p, q, y in zip(validation.p, validation.q, validation.target)])
    assert len(errors) % 12 == 0
    return dict(point=point, source=source, validation=float(errors.mean()), blocks=errors.reshape(-1, 12).mean(axis=1).tolist(),
                internal=sum(1 < len(C) < validation.n for C in point['supports']))


def select_whole_tree(candidates, rule):
    from research_experiments.joint_exit.selection import choose
    # Preserve a whole prediction, rather than merging crossing supports or averaging matrices.
    rows = [dict(row, index=i) for i, row in enumerate(candidates)]
    return choose(rows, rule)


def freeze(output):
    import numpy as np
    import scipy
    import gurobipy as gp
    if output.exists():
        raise FileExistsError('Do not overwrite a frozen experiment.')
    parent = read(PARENT / 'protocol.json')
    assert read(PARENT / 'analysis/verification.json')['passed']
    previous = ROOT / 'docs/rnj_four_methods_20261031'
    for name, item in read(previous / 'manifest.json')['files'].items():
        assert sha(previous / name) == item['sha256']
    output.mkdir(parents=True)
    shutil.copy2(PARENT / 'catalog.json', output / 'catalog.json')
    shutil.copytree(PARENT / 'source', output / 'source')
    for file in Path(__file__).parent.glob('*.py'):
        dest = output / 'source/research_experiments/rnj_nj_portfolio' / file.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(file, dest)
    refs = {}
    for job in parent['jobs']:
        common = PARENT / 'common' / job / 'result.json'
        refs[job] = dict(common=dict(path=str(common), sha256=sha(common)),
                        bases={m: dict(path=str(PARENT / 'evaluation' / job / m / 'result.json'),
                                       sha256=sha(PARENT / 'evaluation' / job / m / 'result.json')) for m in BASES})
        assert refs[job]['common']['sha256'] == parent['common_hashes'][job]
    previous_audits = output / 'previous'
    previous_audits.mkdir()
    for source, name in [(PARENT / 'analysis/independent_LP.json', 'independent_LP.json'),
                         (PARENT / 'analysis/verification.json', 'verification.json'),
                         (previous / 'per_model.csv', 'per_model.csv')]:
        shutil.copy2(source, previous_audits / name)
    write(output / 'protocol.json', dict(created_utc=datetime.now(timezone.utc).isoformat(), jobs=parent['jobs'], references=refs,
        new_methods=NEW_METHODS, primary_selectors=['two_strict', 'three_strict'], sensitivity_selectors=['two_paired', 'three_paired'],
        NJ='existing root-anchored classical Q-criterion NJ; same training RX75 kernel; fixed geometric collapse .16',
        NJ_native='retain geometric topology including zero-L1-weight supports; fit weights only to score original measurement loss',
        NJ_matched='same NJ tree, training L1 response fit, same .24/.16 response contraction and LP refit as four methods; no public RNJ fallback',
        combination='two pretrained final trees; optionally add NJ_matched; select a whole tree using validation only; no topology union or matrix average',
        strict='min validation MAE; at 1e-12 equality prefer fewer internal supports then fixed source order',
        paired='existing 12-time paired block one-SE rule with max(.001*best validation, paired SE), then fewer supports; sensitivity only',
        root=1, resolutions=[.16, .24], primary_resolution=.24, LP_seconds=30., workers=8, hard_worker_seconds=600.,
        no_new_structural_MILP=True, retuning=False, truth_used_by_optimizer=False, test_opened_by_optimizer=False,
        compute='postprocessing marginal time measured separately; deploying both existing pipelines costs up to 300+300 structural seconds; not a new 300-second method',
        oracle='post-hoc finite-portfolio best F1 and any-exact upper references, never used by selectors',
        input_hashes=parent['input_hashes'], parent_protocol_sha256=sha(PARENT / 'protocol.json'),
        parent_manifest=str(previous / 'manifest.json'), parent_manifest_sha256=sha(previous / 'manifest.json'),
        previous_hashes={f.name: sha(f) for f in previous_audits.iterdir()},
        source_hashes={f.relative_to(output / 'source').as_posix(): sha(f) for f in (output / 'source').rglob('*.py')},
        environment=dict(python=sys.version, numpy=np.__version__, scipy=scipy.__version__, gurobi=gp.gurobi.version())))
    write(output / 'state.json', dict(stage='frozen', running=False, planned_jobs=120))


def worker(payload):
    import numpy as np
    from research_experiments.wzzt_paper.common import observed
    from research_experiments.rebuild_benchmark.engine import prepare
    from research_experiments.lp_rebuild.engine import FloorBank
    from research_experiments.joint_exit.selection import library
    p = read(payload)
    for ref in [p['common'], *p['bases'].values()]:
        assert sha(ref['path']) == ref['sha256']
    common = read(p['common']['path'])
    assert common['source_input_hashes'] == {s: sha(p[s]) for s in ('train', 'validation')}
    then = perf_counter()
    data, validation = prepare(observed(p['train'], p['N'])), prepare(observed(p['validation']))
    family, tree = nj_family(common['kernel'], common['labels'])
    bank = FloorBank(data, family, common['calibration'], seconds=30.)
    try:
        raw = bank.fit(family, preserve=True)
        native = evaluate_candidate(raw, validation, 'NJ_native')
        bases = {m: read(ref['path']) for m, ref in p['bases'].items()}
        choices = {}
        for resolution in (.16, .24):
            key = f'strict_{resolution:g}'
            rows = []
            for method in BASES:
                previous = bases[method]['choices'][key]['bank']
                row = evaluate_candidate(previous['point'], validation, method)
                assert abs(row['validation']-previous['validation']) < 1e-10
                assert np.allclose(row['blocks'], previous['blocks'], atol=1e-10, rtol=0.)
                rows.append(row)
            fitted = library(bank, [raw], validation, resolution=resolution)[0]['point']
            matched = evaluate_candidate(fitted, validation, 'NJ_matched')
            choices[str(resolution)] = dict(candidates={r['source']: r for r in [*rows, matched]},
                methods=dict(NJ_native=native, NJ_matched=matched,
                             two_strict=select_whole_tree(rows, 'strict'), two_paired=select_whole_tree(rows, 'paired'),
                             three_strict=select_whole_tree([*rows, matched], 'strict'),
                             three_paired=select_whole_tree([*rows, matched], 'paired')))
        write(p['result'], dict(job=p['job'], choices=choices, NJ_family=family, NJ_edges=tree.edges,
                               marginal_seconds=perf_counter()-then,
                               recorded_two_pipeline_seconds=sum(r['elapsed_seconds'] for r in bases.values()),
                               truth_used=False, test_opened=False, new_structural_MILP=0,
                               protocol_sha256=p['protocol_sha256'], reference_hashes=p['reference_hashes'],
                               LP=bank.lp.solves, LP_failures=bank.failures))
    finally:
        bank.close()


def run(output):
    protocol = read(output / 'protocol.json')
    catalog = read(output / 'catalog.json')
    env = os.environ.copy()
    env['PYTHONPATH'] = str(output / 'source') + os.pathsep + str(output / 'source/rnj_wzzt_core')
    tasks = []
    for job in protocol['jobs']:
        result = output / 'jobs' / job / 'result.json'
        if result.exists() or result.with_name('error.json').exists():
            continue
        tasks.append(dict(job=job, **{k: catalog[job][k] for k in ('N', 'train', 'validation')},
                          **protocol['references'][job], result=str(result), protocol_sha256=sha(output / 'protocol.json'),
                          reference_hashes=protocol['references'][job]))
    active, done = {}, 0
    write(output / 'state.json', dict(stage='NJ fits and validation selection', running=True, planned_jobs=120))
    while tasks or active:
        while tasks and len(active) < protocol['workers']:
            task = tasks.pop(0)
            result = Path(task['result']); payload = result.with_name('payload.json'); write(payload, task)
            log = result.with_name('worker.log').open('w', encoding='utf-8')
            proc = subprocess.Popen([sys.executable, '-B', '-m', 'research_experiments.rnj_nj_portfolio.run', 'worker', str(payload)],
                                    cwd=output / 'source', env=env, stdout=log, stderr=subprocess.STDOUT,
                                    creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            active[task['job']] = proc, log, perf_counter(), result
        for job, (proc, log, start, result) in list(active.items()):
            if proc.poll() is None and perf_counter()-start > protocol['hard_worker_seconds']:
                proc.kill(); proc.wait(); write(result.with_name('error.json'), dict(reason='worker hard limit'))
            if proc.poll() is not None:
                log.close(); del active[job]; done += 1
                if not result.exists() and not result.with_name('error.json').exists():
                    write(result.with_name('error.json'), dict(exit_code=proc.returncode))
                print('NJ/portfolio', done, job, result.exists(), flush=True)
        write(output / 'progress.json', dict(finished_this_run=done, active=len(active), pending=len(tasks)))
        if active: sleep(.5)
    failed = [j for j in protocol['jobs'] if not (output / 'jobs' / j / 'result.json').exists()]
    write(output / 'completion.json', dict(planned=120, returned=120-len(failed), failures=failed))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('command', choices=('freeze', 'run', 'worker')); parser.add_argument('path')
    args = parser.parse_args(); target = Path(args.path).resolve()
    try:
        if args.command == 'freeze': freeze(target)
        elif args.command == 'worker': worker(target)
        else: run(target)
    except Exception:
        if args.command == 'worker': write(Path(read(target)['result']).with_name('error.json'), dict(traceback=traceback.format_exc()))
        else: write(target / 'state.json', dict(stage='needs attention', running=False, error=traceback.format_exc()))
        raise
