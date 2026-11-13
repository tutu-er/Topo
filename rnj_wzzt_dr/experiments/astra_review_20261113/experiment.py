"""Locked fresh-day comparison with frozen identifiers and original AC execution."""
from __future__ import annotations

import os
for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_key] = '1'

import argparse
import atexit
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
import shutil
import subprocess
from time import perf_counter
import traceback

import numpy as np
from scipy.special import logsumexp

from ... import io
from ...observations import load
from ..customer_flexibility_20261113.model import CustomerCase
from ..customer_flexibility_20261113.placement import CandidateEvidence, solve as old_solve
from ..flexibility_sweep_20261113.execution import fixed_rotation, margins, native_selection
from ..flexibility_sweep_20261113.fitting import calibrated_means
from ..flexibility_sweep_20261113.simulation import runtime, background, forecast, response, observe
from ..linear_ac_population_20261113.study import evaluate


ROOT = Path(__file__).resolve().parents[3]
REFERENCE = ROOT/'outputs/customer_flexibility_sweep_20261113_v1'
WORK = {}
SELECTION = [('s0_c013', 32, 'weak'), ('s0_c031', 8, 'easy'), ('s0_c002', 8, 'missing_truth')]
REJECTION_FACTORS = (1.25, 2., 4.)


def utc():
    return datetime.now(timezone.utc).isoformat()


def require(value, message):
    if not value:
        raise ValueError(message)


def rejection_scores(score, observation, truth, target, variance_reference):
    """Fixed external noise multiples; these are sensitivity rules, not calibration."""
    H = np.asarray([p['H'] for p in score['parameters']])
    power = np.column_stack((observation.p, observation.q))
    losses = np.array([p['history_sse'] for p in score['parameters']]) + np.sum(
        (observation.y[None]-np.einsum('tc,knc->ktn', power, H))**2, axis=(1, 2))
    ll = -losses/(2*score['sigma2'])
    ids = [m['topology_id'] for m in score['models']]
    ti = next((i for i, identity in enumerate(ids) if identity == truth['truth_id']), None)
    best = int(np.argmax(ll))
    rules = {}
    for factor in REJECTION_FACTORS:
        B = (score['cells']+observation.y.size)*factor*variance_reference
        lognull = -B/(2*score['sigma2'])
        normal = logsumexp(np.r_[ll, lognull])
        rejected = bool(lognull >= ll[best])
        rules[str(factor)] = dict(rejected=rejected, correct=bool(not rejected and best == ti),
            wrong=bool(not rejected and best != ti), p_target=float(np.exp(ll[target]-normal)),
            p_true=None if ti is None else float(np.exp(ll[ti]-normal)),
            rejection_weight=float(np.exp(lognull-normal)))
    return rules, losses


def prepare(out, *, seconds=12., workers=2):
    require(not out.exists(), 'Use a fresh output directory; preserve all prior runs')
    out.mkdir(parents=True)
    qualified = {r['id']: r for r in io.read(REFERENCE/'qualification.json')}
    entries = []
    for identity, days, kind in SELECTION:
        entry = dict(qualified[identity], new_days=days, role=kind)
        require(entry['qualification'] == kind, 'Qualification changed: '+identity)
        for ref in [entry['score'], *entry['observed_refs'].values()]:
            require(io.ref(ref['path'])['sha256'] == ref['sha256'], 'Historical input drift')
        entries.append(entry)
    io.write(out/'requests.json', io.read(REFERENCE/'requests.json'))
    io.write(out/'offline_truth.json', io.read(REFERENCE/'offline_truth.json'))
    for adoption in sorted({e['adoption'] for e in entries}):
        io.write(out/'resource_configs'/f'{adoption}.json', io.read(REFERENCE/'resource_configs'/f'{adoption}.json'))
    # Runtime materializes a byte-checked copy of the original generator/assets.
    runtime(out)
    sources = {}
    for p in sorted((ROOT/'rnj_wzzt_dr').rglob('*.py')):
        if p.parent == Path(__file__).parent and (p.name.startswith('test_') or p.name in ('diagnostics.py', 'report.py')):
            continue
        rel = p.relative_to(ROOT)
        dest = out/'source_snapshot'/rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, dest)
        sources[rel.as_posix()] = io.ref(p)['sha256']
    protocol = dict(created_utc=utc(), environment=io.environment(),
        commit_at_lock=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        sources=sources, reference_protocol=io.ref(REFERENCE/'protocol.json'), entries=entries,
        new_identifier_fits=0, new_frozen_parameter_fits=0, total_new_days=sum(e['new_days'] for e in entries),
        new_modes=['nominal', 'calibrated'], solver_seconds=seconds, workers=workers,
        replicate_base=900000, replicate_rule='900000 + 1000*historical_entry_index + day',
        selection='Fixed weak s0_c013(32), easy s0_c031(8), missing-truth s0_c002(8); no new-outcome filtering or extension.',
        planning='Historical MAP target; new solver constrains all frozen native parameter points of different target topology; old baseline retains its representative bank. New solver gets observed TRAIN and validation, forecast, known power base and resource case only.',
        old_baseline='Existing gamma2 solver with its existing validation-only mean guards; same 12 second total cap including LP initial generation.',
        new_policy='Direct Gaussian pairwise recovery conditions: total failure budget0.10, rejection0.02, competitors0.08; constrain every relevant native-bank point plus representatives. Exact parameters and formulas fixed by snapshotted solver.py.',
        calibration_selection='Nominal frozen-target model versus validation-only affine mean correction. The latter was selected using already opened old32 AC days as development diagnostics; these48 new days are unseen confirmation. Training observations are available only to historical diagnostics.',
        forecast='Original independent new physical load -> 5% multiplicative forecast; original P/Q meter SD0.5%; voltage SD remains each historical condition.',
        response='Original whole-day per-resource Uniform(.9,1) multipliers, paired across all strategies. Same original AC, resource bounds, baselines and fees.',
        strategies=['no_dfs', 'old_gamma2', 'old_rotation', 'nominal', 'nominal_rotation', 'calibrated', 'calibrated_rotation'],
        rotation='Original candidate-independent fixed LP pattern, scaled to equal optimized planned activation fee whenever its capacity suffices; equality flag always reported.',
        failure_rule='All unresolved/failed quotes execute explicit zero-action fallback; no successful-only denominators. AC failures count as nonrecovery.',
        primary='Original full raw-bank strict-MSE complete recovery; representative MAP recovery, mean target/true weight, gains and drops reported separately.',
        rejection=dict(main_factor=2., sensitivity_factors=REJECTION_FACTORS,
            B='(history_scalar_cells + new_scalar_cells) * factor * 8 * voltage_sigma^2',
            interpretation='Predeclared external noise multiples, not empirical calibration. Included as separate augmented-MAP score; rejection counts as failure to recover.'),
        cost='Original illustrative activation compensation only; planned and settled fees, delivered absolute energy reported. Reservation and full market procurement cost not added.',
        scope='Fresh paired days on three already selected frozen historical conditions; not an unbiased population recovery estimate.',
        private='Raw observations, H, forecasts, actions and quotes stay in ignored outputs')
    io.write(out/'protocol.json', protocol)
    io.write(out/'lock.json', dict(pid=os.getpid(), created_utc=utc(), protocol=io.ref(out/'protocol.json')))
    return protocol


def verify_sources(out):
    protocol = io.read(out/'protocol.json')
    require(io.ref(out/'protocol.json')['sha256'] == io.read(out/'lock.json')['protocol']['sha256'], 'Protocol drift')
    for relative, sha in protocol['sources'].items():
        require(io.ref(ROOT/relative)['sha256'] == sha, 'Source drift: '+relative)
    return protocol


def initialize(out):
    import gurobipy as gp
    out = Path(out)
    env = gp.Env(empty=True)
    env.setParam('OutputFlag', 0)
    env.start()
    atexit.register(env.dispose)
    WORK.update(out=out, protocol=io.read(out/'protocol.json'), env=env, module=runtime(out), cache={})


def context(entry):
    if entry['id'] not in WORK['cache']:
        for ref in [entry['score'], *entry['observed_refs'].values()]:
            require(io.ref(ref['path'])['sha256'] == ref['sha256'], 'Historical input drift')
        score = io.read(entry['score']['path'])
        training = load(entry['observed_refs']['train']['path'])
        validation = load(entry['observed_refs']['validation']['path'])
        case = CustomerCase(io.read(WORK['out']/'resource_configs'/f"{entry['adoption']}.json"))
        pattern, rotation_receipt = fixed_rotation(case)
        WORK['cache'][entry['id']] = dict(score=score, training=training, validation=validation, case=case,
            pattern=pattern, rotation_receipt=rotation_receipt, means=calibrated_means(score, validation))
    return WORK['cache'][entry['id']]


def run_day(task):
    from .solver import plan
    entry, day = task['entry'], task['day']
    out, protocol = WORK['out'], WORK['protocol']
    d = out/'confirmation'/entry['id']/f'day{day:03d}'
    if (d/'complete.json').exists():
        return io.read(d/'complete.json')
    require(not (d/'started.json').exists(), 'Unfinished day lock must be reviewed')
    d.mkdir(parents=True, exist_ok=True)
    io.write(d/'started.json', dict(pid=os.getpid(), created_utc=utc(), id=entry['id'], day=day))
    began = perf_counter()
    ctx = context(entry)
    s, case = ctx['score'], ctx['case']
    H = np.asarray([p['H'] for p in s['parameters']])
    target = s['map_index']
    competitors = [j for j in range(len(H)) if j != target]
    runner = max(competitors, key=lambda j: s['models'][j]['log_probability']) if competitors else None
    replicate = protocol['replicate_base']+entry['index']*1000+day
    common = dict(id=entry['id'], day=day, role=entry['role'], replicate=replicate,
        historical_p_map=s['p_map'], target=target, runner=runner, historical_truth_present=entry['truth_present'])
    controls = {'no_dfs': np.zeros(case.shape)}
    plans = {}
    b = background(WORK['module'], case, entry, replicate)
    prediction = forecast(b, replicate)
    for mode in ['old_gamma2', *protocol['new_modes']]:
        tick = perf_counter()
        try:
            if mode == 'old_gamma2':
                if competitors:
                    evidence = CandidateEvidence(case, H, target, competitors, prediction, s['sigma2'],
                        b['network']['base_kw'], ctx['means'])
                    receipt, point = old_solve(evidence, 2., seconds=protocol['solver_seconds'], env=WORK['env'])
                else:
                    receipt, point = dict(feasible=True, status='single_candidate_zero', source='zero',
                        cost_yuan=0., lower_bound_yuan=0., global_optimality_verified=True), np.zeros(case.shape)
            else:
                receipt, point = plan(case, s, ctx['validation'], prediction, b['network']['base_kw'],
                    mode=mode, seconds=protocol['solver_seconds'], env=WORK['env'], training=ctx['training'],
                    voltage_sigma=entry['voltage_sigma'])
            require((point is not None) == bool(receipt['feasible']), 'Policy and feasibility flag disagree')
            if point is not None:
                require(receipt['feasible'] and case.audit(point)['passed'], 'Invalid returned policy')
        except Exception as exc:
            receipt, point = dict(feasible=False, status='solver_exception_zero_fallback', error=repr(exc),
                traceback=traceback.format_exc(), cost_yuan=None, lower_bound_yuan=None,
                global_optimality_verified=False), None
        receipt['driver_planning_seconds'] = perf_counter()-tick
        receipt['explicit_zero_fallback'] = point is None
        controls[mode] = np.zeros(case.shape) if point is None else point
        fee = case.fee(controls[mode])
        pattern_fee = case.fee(ctx['pattern'])
        fraction = min(1., fee/pattern_fee)
        rotation_name = 'old_rotation' if mode == 'old_gamma2' else mode+'_rotation'
        controls[rotation_name] = fraction*ctx['pattern']
        receipt.update(rotation_planned_fee_yuan=case.fee(controls[rotation_name]),
            rotation_same_planned_fee=abs(case.fee(controls[rotation_name])-fee) <= 1e-7)
        plans[mode] = receipt
    # Neither response multipliers nor execution observations have been opened.
    np.savez_compressed(d/'planner.npz', forecast=prediction, **controls)
    io.write(d/'plans.json', plans)
    io.write(d/'policy_seal.json', dict(created_utc=utc(), planner=io.ref(d/'planner.npz'),
        plans=io.ref(d/'plans.json'), score=entry['score'], protocol=io.ref(out/'protocol.json'),
        planner_inputs=['frozen_score', 'observed_training', 'observed_validation', 'noisy_forecast', 'resource_contract', 'power_base'],
        competitors=competitors, actual_response_or_execution_voltage_opened=False))
    eta = response(case.R, replicate)
    np.savez_compressed(d/'physical_background.npz', P=b['p'], Q=b['q'], root=b['root'],
        meter_p=b['noise'][0], meter_q=b['noise'][1], meter_v=b['noise'][2], meter_root=b['noise'][3], response_multiplier=eta)
    io.write(d/'generation.json', b['metadata'])
    truth = io.read(out/'offline_truth.json')
    rows = []
    for strategy, u in controls.items():
        mode = 'old_gamma2' if strategy == 'old_rotation' else strategy.removesuffix('_rotation')
        receipt = plans.get(mode)
        info = dict(common, strategy=strategy, planned_fee_yuan=case.fee(u), settled_fee_yuan=case.fee(u*eta[None]),
            delivered_abs_kwh=float(case.dt*np.abs(u*eta[None]).sum()),
            planning_feasible=None if receipt is None else receipt['feasible'],
            zero_fallback=False if receipt is None else receipt['explicit_zero_fallback'],
            max_requested_kw=float(np.max(abs(u))), max_delivered_kw=float(np.max(abs(u*eta[None]))))
        try:
            observation, audit, physical = observe(WORK['module'], b, u, case, eta)
            values = evaluate(s, observation, truth, target)
            ll = values.pop('log_probability')
            values.update(margins(ll, target, runner))
            values.update(native_selection(s, observation, truth))
            reject, losses = rejection_scores(s, observation, truth, target, 8*entry['voltage_sigma']**2)
            info.update(status='complete', **values, **audit, rejection_rules=reject)
            np.savez_compressed(d/(strategy+'.npz'), P=observation.p, Q=observation.q, V=observation.voltage,
                root=observation.root, labels=observation.labels, log_probability=ll, cumulative_sse=losses,
                true_P=physical[0], true_Q=physical[1], true_V=physical[2])
        except Exception as exc:
            info.update(status='AC_or_scoring_failed', error=repr(exc), traceback=traceback.format_exc())
        rows.append(info)
    io.write(d/'results.json', rows)
    complete = dict(id=entry['id'], day=day, status='complete', rows=len(rows),
        failed_rows=sum(r['status'] != 'complete' for r in rows), seconds=perf_counter()-began,
        results=io.ref(d/'results.json'), plans=io.ref(d/'plans.json'))
    io.write(d/'complete.json', complete)
    return complete


def run(out):
    protocol = verify_sources(out)
    tasks = [dict(entry=e, day=d) for e in protocol['entries'] for d in range(e['new_days'])]
    pending = []
    for task in tasks:
        d = out/'confirmation'/task['entry']['id']/f"day{task['day']:03d}"
        if not (d/'complete.json').exists() and not (d/'worker_failure.json').exists():
            require(not (d/'started.json').exists(), 'Unfinished day lock: '+str(d))
            pending.append(task)
    if not pending:
        return summarize(out)
    began = perf_counter()
    receipts = []
    with ProcessPoolExecutor(max_workers=protocol['workers'], initializer=initialize, initargs=(str(out),)) as pool:
        futures = {pool.submit(run_day, task): task for task in pending}
        for f in as_completed(futures):
            task = futures[f]
            try:
                receipt = f.result()
            except Exception as exc:
                receipt = dict(id=task['entry']['id'], day=task['day'], status='worker_failed',
                    error=repr(exc), traceback=traceback.format_exc())
                io.write(out/'confirmation'/receipt['id']/f"day{receipt['day']:03d}"/'worker_failure.json', receipt)
            receipts.append(receipt)
            io.write(out/'checkpoint.json', dict(completed_tasks=len(receipts), launched_tasks=len(pending),
                seconds=perf_counter()-began, receipts=receipts))
            print('day', len(receipts), '/', len(pending), receipt['id'], receipt['day'], receipt['status'], flush=True)
    summary = summarize(out)
    io.write(out/'complete.json', dict(created_utc=utc(), summary=io.ref(out/'summary.json'),
        status='complete_with_failures' if summary['failed_jobs'] or summary['failed_strategy_rows'] else 'complete',
        protocol=io.ref(out/'protocol.json')))
    return summary


def summarize(out):
    protocol = io.read(out/'protocol.json')
    rows = [r for p in sorted((out/'confirmation').glob('*/day*/results.json')) for r in io.read(p)]
    failures = [io.read(p) for p in sorted((out/'confirmation').glob('*/day*/worker_failure.json'))]
    groups = {}
    for entry in protocol['entries']:
        by_strategy = {}
        for strategy in protocol['strategies']:
            available = [r for r in rows if r['id'] == entry['id'] and r['strategy'] == strategy and r['status'] == 'complete']
            valid_true = [r['p_true'] for r in available if r['p_true'] is not None]
            by_strategy[strategy] = dict(planned_days=entry['new_days'], completed_days=len(available),
                native_correct=sum(r['native_map_correct'] for r in available),
                representative_correct=sum(r['map_correct'] for r in available),
                mean_p_true=None if not valid_true else float(np.mean(valid_true)),
                mean_p_target=None if not available else float(np.mean([r['p_target'] for r in available])),
                mean_settled_fee_yuan=None if not available else float(np.mean([r['settled_fee_yuan'] for r in available])),
                mean_delivered_abs_kwh=None if not available else float(np.mean([r['delivered_abs_kwh'] for r in available])),
                zero_fallbacks=sum(r['zero_fallback'] for r in available),
                rejection_main=dict(correct=sum(r['rejection_rules']['2.0']['correct'] for r in available),
                    wrong=sum(r['rejection_rules']['2.0']['wrong'] for r in available),
                    rejected=sum(r['rejection_rules']['2.0']['rejected'] for r in available)))
        groups[entry['id']] = dict(role=entry['role'], strategies=by_strategy)
    summary = dict(protocol=io.ref(out/'protocol.json'), planned_days=protocol['total_new_days'],
        complete_jobs=len(list((out/'confirmation').glob('*/day*/complete.json'))), failed_jobs=len(failures),
        failed_strategy_rows=sum(r['status'] != 'complete' for r in rows), groups=groups, worker_failures=failures)
    io.write(out/'summary.json', summary)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'run', 'summarize'])
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--seconds', type=float, default=12.)
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    destination = args.out.resolve()
    result = prepare(destination, seconds=args.seconds, workers=args.workers) if args.action == 'prepare' else run(destination) if args.action == 'run' else summarize(destination)
    print('done', args.action, result.get('planned_days', result.get('total_new_days')), flush=True)
