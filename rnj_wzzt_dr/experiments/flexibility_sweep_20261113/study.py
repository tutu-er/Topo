"""Lock -> observed-history fitting -> qualification -> paired AC -> confirmation."""
from __future__ import annotations

import os
for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_key] = '1'

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
import shutil
import subprocess
from time import perf_counter
import traceback

import numpy as np

from ... import io
from ..customer_flexibility_20261113.model import CustomerCase
from ..posterior_mse_20261112.scoring import topology_id
from .config import ROOT, STUDY_SEED, WORKERS, GAMMAS, FIT_BUDGET, grid, resource_configs
from .fitting import fit_one
from .simulation import runtime, background, observe, sample_history
from . import execution


def utc():
    return datetime.now(timezone.utc).isoformat()


def require(value, message):
    if not value:
        raise ValueError(message)


def verify_sources(out):
    p = io.read(out/'protocol.json')
    for rel, sha in p['sources'].items():
        require(io.ref(ROOT/rel)['sha256'] == sha, 'Source drift: '+rel)
    for name, ref in p['resource_config_refs'].items():
        require(io.ref(ref['path'])['sha256'] == ref['sha256'], 'Resource config drift: '+name)
    return p


def prepare(out):
    require(not out.exists(), 'Use a fresh output directory')
    out.mkdir(parents=True)
    source_requests = io.read(ROOT/'outputs/linear_ac_population_20261113_v2/requests.json')
    io.write(out/'requests.json', [next(r for r in source_requests if r['cohort'] == 'main')])
    configs = resource_configs()
    for name, config in configs.items():
        case = CustomerCase(config)
        require(case.audit(np.zeros(case.shape))['passed'], 'Zero action is not feasible')
        io.write(out/'resource_configs'/f'{name}.json', config)
    # The entire imported project implementation is snapshotted. Read-only
    # reporting/tests are excluded and record their versions separately.
    sources = {}
    for path in sorted((ROOT/'rnj_wzzt_dr').rglob('*.py')):
        if path.parent == Path(__file__).parent and (path.name.startswith('test_') or path.name in ('report.py', 'audit.py')):
            continue
        rel = path.relative_to(ROOT)
        destination = out/'source_snapshot'/rel
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
        sources[rel.as_posix()] = io.ref(path)['sha256']
    module = runtime(out)
    protocol = dict(created_utc=utc(), study_seed=STUDY_SEED, environment=io.environment(), sources=sources,
        commit_at_lock=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        resource_config_refs={k: io.ref(out/'resource_configs'/f'{k}.json') for k in configs},
        history_grid={'seed0': grid(0), 'optional_seed1': grid(1)},
        history_points_rule='48: full day uniform 30-min; 96: full day 15-min; 192: two full days 15-min; first 2/3 training, last 1/3 validation',
        fit_budget=FIT_BUDGET, workers=WORKERS, gamma=GAMMAS, solver_seconds=3.,
        target='history posterior MAP; truth is used only by offline qualification and final evaluation',
        qualification='truth candidate present AND original native winner correct AND posterior MAP correct; weak p<.95, secondary .95<=p<.99; easy>=.99; missing truth retained',
        extension_rule='Run the second prelocked 36-condition seed only if fewer than four weak qualified histories OR fewer than two development cases have positive paired primary-gamma posterior gain.',
        development_rule='At most 12 qualified p<.99, balanced deterministic strata (adoption,N,profile), smallest p in each stratum first; plus at most two easy and two missing-truth controls. Two independent new days per history.',
        confirmation_rule='At most four development positives with no-DFS mean p_true<.99, ranked by paired primary-gamma gain, plus at most two weakest/nonpositive qualified controls. Lock IDs and score hashes before 32 fresh days per history; no confirmation-dependent extension.',
        forecast='true net P/Q after device baselines, multiplied by independent 1+0.05Z; P/Q meter relative SD0.005; voltage meter SD follows grid',
        response='delta net P/Q is linear; independent whole-day per-resource multiplier Uniform(.9,1), common across strategies; nominal planning multiplier1; EV/thermal energy preserved by scaling feasible deviations toward zero',
        model='minimum piecewise-linear activation fee subject to original linear device domain, expected nominal logodds gain >=gamma for EVERY other candidate, plus existing two-sided history-residual guards >=0',
        solver='existing supporting LP warm start plus native NonConvex=2 QCQP; time includes both; direct original-domain checks, feasible quote and valid bound reported separately',
        rotation='fixed H-independent alternating hour/resource LP pattern, scaled toward zero to equal planned activation fee whenever the pattern budget suffices; actual settled fee reported separately',
        score='unchanged uniform prior over distinct topologies, one train-only representative each, sigma2=min validation MSE; frozen H/sigma/history SSE; cumulative validation+96-point new day; original full raw-bank strict-MSE result separately reported',
        physical='unchanged original paper15 branches/AC; device baselines present in BOTH history and new day; no line changes; all policies sealed before true execution voltage',
        outcomes='paired full-bank minimum logodds and fixed-runner logodds; mean p_true; p_true>=.8,.9,.95; posterior drops; complete recovery; planned/settled fees; execution and solver failures',
        inference='purposeful development-case search; confirmation concerns selected conditions, not an unbiased population estimate; independent day count is not multiplied by number of policies',
        failures='preserve failures/time stops/missing truth; planner failure executes explicit zero fallback; no repeat of completed fits or failed budgets; no post-confirmation tuning',
        private='raw P/Q/V, fitted coefficients, forecasts, action arrays and quote trajectories remain in ignored outputs')
    io.write(out/'protocol.json', protocol)
    io.write(out/'lock.json', dict(created_utc=utc(), protocol=io.ref(out/'protocol.json')))
    return protocol


def save_observed(path, o):
    np.savez_compressed(path, P=o.p, Q=o.q, V=o.voltage, root=o.root, labels=o.labels)


def prepare_histories(out, seed):
    verify_sources(out)
    target = out/f'history_jobs_s{seed}.json'
    require(not target.exists(), 'This history seed is already prepared')
    if seed:
        require((out/'extension_seal.json').exists(), 'Extension condition must be checked and sealed first')
    module = runtime(out)
    jobs = []
    for spec in grid(seed):
        case = CustomerCase(io.read(out/'resource_configs'/f"{spec['adoption']}.json"))
        directory = out/'histories'/spec['id']
        directory.mkdir(parents=True)
        days, audits = [], []
        for day in range(2 if spec['history_points'] == 192 else 1):
            b = background(module, case, spec, 1000+seed*20+day)
            observation, audit, true = observe(module, b)
            days.append(observation)
            audits.append(audit)
            np.savez_compressed(directory/f'private_day{day}.npz', P=true[0], Q=true[1], V=true[2], root=b['root'])
        all_observed, train, valid = sample_history(days, spec['history_points'])
        for name, o in [('observed', all_observed), ('train', train), ('validation', valid)]:
            save_observed(directory/(name+'.npz'), o)
        io.write(directory/'generation_audit.json', dict(spec=spec, audits=audits))
        jobs.append(dict(spec, directory=str(directory), observed_refs={k: io.ref(directory/(k+'.npz')) for k in ('train', 'validation')}))
    # Kept in the evaluator's directory; never supplied to a fitting task.
    labels = days[0].labels
    clades = tuple(sorted(tuple(sorted(labels[i] for i in C)) for C in module.case_truth('paper15', labels) if 1 < len(C) < len(labels)))
    truth = dict(truth_id=topology_id(clades), truth_clades=clades)
    if (out/'offline_truth.json').exists():
        require(io.read(out/'offline_truth.json')['truth_id'] == truth['truth_id'], 'Truth drift')
    else:
        io.write(out/'offline_truth.json', truth)
    io.write(target, jobs)
    return jobs


def jobs(out):
    return [row for path in sorted(out.glob('history_jobs_s*.json')) for row in io.read(path)]


def fit(out, limit=None):
    protocol = verify_sources(out)
    pending = [j for j in jobs(out) if not (Path(j['directory'])/'fit_complete.json').exists()]
    if limit:
        pending = pending[:limit]
    require(pending, 'No new history fits remain')
    require(not any((Path(j['directory'])/'fit_started.json').exists() for j in pending), 'Unfinished fit lock exists')
    receipts = []
    began = perf_counter()
    with ProcessPoolExecutor(max_workers=protocol['workers']) as pool:
        fs = {pool.submit(fit_one, j): j for j in pending}
        for f in as_completed(fs):
            try:
                r = f.result()
            except Exception as exc:
                r = dict(id=fs[f]['id'], status='worker_failed', error=repr(exc), traceback=traceback.format_exc())
            receipts.append(r)
            io.write(out/'history_checkpoint.json', dict(receipts=receipts, planned=len(pending), seconds=perf_counter()-began))
            print('history', len(receipts), '/', len(pending), r['id'], r['status'], round(r.get('seconds', 0), 1), flush=True)
    return receipts


def qualify(out):
    truth = io.read(out/'offline_truth.json')
    rows = []
    for j in jobs(out):
        p = Path(j['directory'])/'fit_complete.json'
        row = dict(j)
        if not p.exists():
            row.update(status='not_run')
        else:
            receipt = io.read(p)
            row.update(status=receipt['status'], fit_seconds=receipt['seconds'])
            if receipt['status'] == 'fitted':
                s = io.read(receipt['score']['path'])
                covered = any(m['topology_id'] == truth['truth_id'] for m in s['models'])
                correct = s['map_topology_id'] == truth['truth_id'] and s['native_topology_id'] == truth['truth_id']
                kind = 'missing_truth' if not covered else 'wrong_winner' if not correct else 'weak' if s['p_map'] < .95 else 'secondary' if s['p_map'] < .99 else 'easy'
                row.update(score=receipt['score'], truth_present=covered, qualification=kind, p_map=s['p_map'],
                    original_winner_correct=s['native_topology_id'] == truth['truth_id'], posterior_MAP_correct=s['map_topology_id'] == truth['truth_id'],
                    p_true=next((m['probability'] for m in s['models'] if m['topology_id'] == truth['truth_id']), None),
                    distinct_topologies=s['distinct_topologies'], native_stop_reason=receipt['native_stop_reason'],
                    search_budget_reached=receipt['search_budget_reached'], method_cap_exceeded=receipt['method_cap_exceeded'])
        rows.append(row)
    io.write(out/'qualification.json', rows)
    return rows


def balanced(rows, maximum):
    groups = {}
    for r in rows:
        groups.setdefault((r['adoption'], r['history_points'], r['profile']), []).append(r)
    for g in groups.values():
        g.sort(key=lambda r: (r['p_map'], r['index']))
    selected = []
    while groups and len(selected) < maximum:
        for key in sorted(list(groups)):
            if len(selected) == maximum:
                break
            selected.append(groups[key].pop(0))
            if not groups[key]:
                del groups[key]
    return selected


def select_development(out):
    rows = qualify(out)
    selected = balanced([r for r in rows if r.get('qualification') in ('weak', 'secondary')], 12)
    selected += balanced([r for r in rows if r.get('qualification') == 'easy'], 2)
    selected += balanced([r for r in rows if r.get('qualification') == 'missing_truth'], 2)
    existing = io.read(out/'development_selection.json') if (out/'development_selection.json').exists() else []
    # Preserve first-seed selection and its outcomes when a second seed opens.
    by_id = {r['id']: r for r in existing}
    by_id.update({r['id']: r for r in selected})
    io.write(out/'development_selection.json', list(by_id.values()))
    return list(by_id.values())


def collect(out, phase):
    return [r for p in sorted((out/phase).glob('*/results.json')) for r in io.read(p)]


def paired_summary(rows):
    lookup = {(r['id'], r['day'], r['strategy']): r for r in rows if r['status'] == 'complete'}
    results = []
    for identity in sorted({r['id'] for r in rows}):
        pairs = [(lookup[(identity, d, 'no_dfs')], lookup[(identity, d, 'optimized_g0')])
            for d in sorted({r['day'] for r in rows if r['id'] == identity})
            if (identity, d, 'no_dfs') in lookup and (identity, d, 'optimized_g0') in lookup]
        pairs = [(a, b) for a, b in pairs if a['p_true'] is not None]
        if pairs:
            results.append(dict(id=identity, qualification=pairs[0][0]['qualification'], paired_days=len(pairs),
                no_dfs_mean_p_true=float(np.mean([a['p_true'] for a, b in pairs])),
                mean_p_true_gain=float(np.mean([b['p_true']-a['p_true'] for a, b in pairs])),
                mean_min_logodds_gain=float(np.mean([b['target_minimum_logodds']-a['target_minimum_logodds'] for a, b in pairs
                                                   if a['target_minimum_logodds'] is not None])) if pairs[0][0]['target_minimum_logodds'] is not None else None))
    return results


def extension(out):
    rows = qualify(out)
    require(not (out/'history_jobs_s1.json').exists(), 'Second history seed already opened')
    dev = paired_summary(collect(out, 'development'))
    weak = sum(r.get('qualification') == 'weak' for r in rows)
    positive = sum(r['qualification'] in ('weak', 'secondary') and r['mean_p_true_gain'] > 0 and r['no_dfs_mean_p_true'] < .99 for r in dev)
    needed = weak < 4 or positive < 2
    io.write(out/'extension_decision.json', dict(weak_qualified=weak, development_positives=positive, second_seed_needed=needed))
    if needed:
        io.write(out/'extension_seal.json', dict(created_utc=utc(), decision=io.ref(out/'extension_decision.json'), grid=grid(1)))
        prepare_histories(out, 1)
    return needed


def select_confirmation(out):
    require(not (out/'confirmation_selection.json').exists(), 'Confirmation IDs already sealed')
    dev = paired_summary(collect(out, 'development'))
    eligible = [r for r in dev if r['qualification'] in ('weak', 'secondary')]
    positive = sorted([r for r in eligible if r['mean_p_true_gain'] > 0 and r['no_dfs_mean_p_true'] < .99],
                      key=lambda r: (-r['mean_p_true_gain'], r['id']))[:4]
    controls = sorted([r for r in eligible if r['id'] not in {x['id'] for x in positive}],
                      key=lambda r: (r['mean_p_true_gain'], r['id']))[:2]
    entries = {r['id']: r for r in io.read(out/'development_selection.json')}
    chosen = [dict(entries[r['id']], confirmation_role='positive' if r in positive else 'weak_or_negative_control', development_summary=r) for r in positive+controls]
    seal = dict(created_utc=utc(), entries=chosen, days=32, development_summary=dev,
                source_protocol=io.ref(out/'protocol.json'), no_confirmation_observations_opened=True)
    io.write(out/'confirmation_selection.json', seal)
    return seal


def run_phase(out, phase):
    protocol = verify_sources(out)
    selected = io.read(out/'development_selection.json') if phase == 'development' else io.read(out/'confirmation_selection.json')['entries']
    days = 2 if phase == 'development' else 32
    pending = [e for e in selected if not (out/phase/e['id']/'complete.json').exists()]
    require(pending, 'All selected phase jobs are already complete or no case qualifies')
    began = perf_counter()
    receipts = []
    with ProcessPoolExecutor(max_workers=protocol['workers'], initializer=execution.initialize, initargs=(str(out),)) as pool:
        futures = {pool.submit(execution.run_case, dict(entry=e, phase=phase, days=days)): e for e in pending}
        for f in as_completed(futures):
            try:
                receipt = f.result()
            except Exception as exc:
                receipt = dict(id=futures[f]['id'], status='worker_failed', error=repr(exc), traceback=traceback.format_exc())
            receipts.append(receipt)
            io.write(out/(phase+'_checkpoint.json'), dict(receipts=receipts, seconds=perf_counter()-began))
            print(phase, len(receipts), '/', len(pending), receipt['id'], receipt['status'], round(perf_counter()-began, 1), flush=True)
    return receipts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['prepare', 'fit', 'qualify', 'select-development', 'development', 'extension', 'select-confirmation', 'confirmation'])
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    out = args.out.resolve()
    if args.action == 'prepare':
        prepare(out)
        result = prepare_histories(out, 0)
    elif args.action == 'fit':
        result = fit(out, args.limit)
    elif args.action == 'qualify':
        result = qualify(out)
    elif args.action == 'select-development':
        result = select_development(out)
    elif args.action == 'extension':
        result = extension(out)
    elif args.action == 'select-confirmation':
        result = select_confirmation(out)
    else:
        result = run_phase(out, args.action)
    print('done', args.action, len(result) if isinstance(result, list) else '', flush=True)


if __name__ == '__main__':
    main()
