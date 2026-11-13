"""One bounded planning-only follow-up; never replace sealed AC outcomes."""
import os
for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_key] = '1'

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from ... import io
from ...observations import load
from ..customer_flexibility_20261113.model import CustomerCase
from ..customer_flexibility_20261113.placement import CandidateEvidence, solve
from .fitting import calibrated_means
from .study import utc, require


def one(task):
    import gurobipy as gp
    out, spec = Path(task['out']), task['entry']
    d = Path(task['daydir'])
    require(io.ref(d/'planner.npz')['sha256'] == task['planner']['sha256'], 'Sealed forecast changed')
    require(io.ref(spec['score']['path'])['sha256'] == spec['score']['sha256'], 'Frozen score changed')
    score = io.read(spec['score']['path'])
    case = CustomerCase(io.read(out/'resource_configs'/f"{spec['adoption']}.json"))
    H = np.asarray([p['H'] for p in score['parameters']])
    target = score['map_index']
    means = calibrated_means(score, load(spec['observed_refs']['validation']['path']))
    with np.load(d/'planner.npz') as z:
        prediction = np.array(z['forecast'])
    evidence = CandidateEvidence(case, H, target, [j for j in range(len(H)) if j != target], prediction,
        score['sigma2'], io.read(d/'generation.json')['base_kw'], means)
    with gp.Env(empty=True) as env:
        env.setParam('OutputFlag', 0)
        env.start()
        result, u = solve(evidence, task['gamma'], seconds=30., env=env)
    directory = out/'refinement'/task['id']
    directory.mkdir(parents=True, exist_ok=True)
    if u is not None:
        np.savez_compressed(directory/'private_point.npz', u=u)
    receipt = dict(id=task['id'], day=task['day'], gamma=task['gamma'], original_quote=task['old'],
        refined_quote=result, additional_budget_seconds=30.,
        cumulative_solve_seconds=task['old']['seconds']+result['seconds'],
        evaluated_on_new_AC=False, sealed_original_policies_changed=False)
    io.write(directory/'complete.json', receipt)
    return receipt


def run(out):
    directory = out/'refinement'
    require(not directory.exists(), 'One follow-up only; do not reset its budget')
    entries = {e['id']: e for e in io.read(out/'confirmation_selection.json')['entries']}
    tasks = []
    for index, gamma in enumerate(io.read(out/'protocol.json')['gamma']):
        candidates = []
        for path in sorted((out/'confirmation').glob('*/day*/plans.json')):
            old = io.read(path)[f'g{index}']
            if old['status'] == 'unresolved':
                d = path.parent
                candidates.append(dict(id=f"{d.parent.name}_{d.name}_g{index}", day=int(d.name[3:]),
                    out=str(out), entry=entries[d.parent.name], gamma=gamma,
                    daydir=str(d), planner=io.ref(d/'planner.npz'), old=old))
        tasks.extend(candidates[:2])
    require(tasks, 'There is no unresolved quote to inspect')
    directory.mkdir()
    protocol = dict(created_utc=utc(), tasks=tasks, workers=4, additional_seconds_per_task=30.,
        selection='First two unresolved quotes per intensity in directory order; no AC/posterior outcome used.',
        source={str(p): io.ref(p) for p in (Path(__file__), Path(__file__).with_name('fitting.py'),
            Path(__file__).parents[1]/'customer_flexibility_20261113/placement.py',
            Path(__file__).parents[1]/'customer_flexibility_20261113/model.py')},
        scope='Planning only on existing frozen forecasts. Original confirmation observations and all-day statistics unchanged.')
    io.write(directory/'protocol.json', protocol)
    receipts = []
    with ProcessPoolExecutor(max_workers=4) as pool:
        fs = {pool.submit(one, task): task for task in tasks}
        for f in as_completed(fs):
            receipt = f.result()
            receipts.append(receipt)
            print(receipt['id'], receipt['refined_quote']['status'], flush=True)
    io.write(directory/'complete.json', dict(receipts=receipts, protocol=io.ref(directory/'protocol.json')))
    return receipts


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    run(args.out.resolve())
