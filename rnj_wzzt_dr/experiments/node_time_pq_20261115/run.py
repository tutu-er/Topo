"""Historical node/hour P/Q bands on the frozen single-target procurement."""
from __future__ import annotations

import os
for _name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_name] = '1'

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from pathlib import Path
import shutil
from time import perf_counter
import traceback

import numpy as np

from ... import io
from ..frozen_rx_guard_20261115 import run as paired
from ..single_target_20261115 import run as single
from ..single_target_20261115.evidence import SingleTargetEvidence
from ..distribution_gate_20261114 import solver as engine
from ..flexibility_stability_20261114 import solver as prices
from . import bandwidth, domain


ROOT = paired.old.ROOT
MODES = paired.MODES


def historical_band(entry):
    """Only observed training P/Q and its sealed timestamps cross this API."""
    train_ref = entry['files']['train.npz']
    sampling_ref = entry['files']['observations_sealed.json']
    train = paired.arrays(paired.old.checked(train_ref))
    sampling = io.read(paired.old.checked(sampling_ref))
    if sampling['observations']['train']['ref'] != train_ref:
        raise ValueError('Timestamp ledger and observed training reference differ')
    timestamps = sampling['sampling']['train']['timestamps']
    arrays, metadata = bandwidth.build_bandwidth(
        train['P'], train['Q'], train['labels'], timestamps, entry['base_kw'])
    metadata.update(training_source=train_ref, sampling_source=sampling_ref,
        sampling_indices=sampling['sampling']['train']['indices'],
        information_boundary='Observed training P/Q, labels, sealed training UTC timestamps and public normalization base only; no validation/forecast/truth')
    return arrays, metadata


def prepare(out, *, smoke=False, seconds=600., preflight=None,
            use_sdp=False, sdp_seconds=120., band_baseline_audit=None):
    out = Path(out).resolve()
    if out.exists():
        raise FileExistsError('Preserve existing outputs: '+str(out))
    if not np.isfinite([seconds, sdp_seconds]).all() or min(seconds, sdp_seconds) <= 0:
        raise ValueError('Positive finite solver budgets required')
    baseline = ROOT/'outputs/single_target_execution_20261115_v1'
    single_audit = ROOT/'outputs/single_target_audit_20261115_v1/report.json'
    sp = io.read(baseline/'protocol.json')
    sd = io.read(baseline/'complete.json')
    sa = io.read(single_audit)
    if sd['status'] != 'complete' or not sd['source_unchanged'] or sa['status'] != 'passed' or sa['audited_rows'] != 144:
        raise ValueError('Complete independently audited original single-target run required')
    paired.check_refs([sp['inputs'], sa['source_protocol'], sa['source_inputs']])
    original = io.read(sp['source_protocol']['path'])
    changes = single.baseline_sources(Path(sp['source_run']), original)
    if not smoke and (preflight is None or io.read(preflight)['status'] != 'passed'):
        raise ValueError('Passed independent smoke audit required')
    if use_sdp and not smoke and (band_baseline_audit is None or io.read(band_baseline_audit)['status'] != 'passed'):
        raise ValueError('Audited no-SDP node/hour run required before the SDP comparison')
    entries = deepcopy(io.read(sp['inputs']['path']))
    if tuple(e['id'] for e in entries) != tuple(paired.IDS):
        raise ValueError('Keep the three original paired input settings')
    # Require exactly the old physical resources, forecast, historical target and seed pool.
    for entry in entries:
        planning, forecast, physical, _ = paired.context_planning(entry)
        probe = SingleTargetEvidence(physical, planning['score'], planning['val'], forecast,
            planning['base'], mode='normal_fixed', meter_relative_sd=entry['meter']['additional_V_relative_sd'])
        if probe.native_indices[probe.targets[0]] != entry['planned_history_target_native']:
            raise ValueError('Historical target changed')
        entry['single_baseline_directory'] = str(baseline/'cases'/entry['id'])
    out.mkdir(parents=True)
    for entry in entries:
        arrays, metadata = historical_band(entry)
        folder = out/'bandwidths'/entry['id']
        folder.mkdir(parents=True)
        np.savez_compressed(folder/'bandwidth.npz', **arrays)
        metadata['arrays'] = io.ref(folder/'bandwidth.npz')
        io.write(folder/'bandwidth.json', metadata)
        entry['node_time_bandwidth'] = io.ref(folder/'bandwidth.json')
    io.write(out/'inputs.json', entries)
    if use_sdp:
        from . import sdp
    refs = paired.panel.snapshot(out, entries)
    extra = [Path(__file__), Path(bandwidth.__file__), Path(domain.__file__),
             Path(__file__).with_name('PROTOCOL.md')]
    if use_sdp:
        extra.append(Path(sdp.__file__))
    for path in extra:
        path = path.resolve()
        target = out/'source_snapshot'/path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        if not any(ref['path'] == str(path) for ref in refs):
            refs.append(io.ref(path))
    modes = ('mgf_fixed', 'mgf_forecast') if use_sdp else MODES
    band_run = None
    if band_baseline_audit is not None:
        band_run = str(Path(io.read(band_baseline_audit)['source_protocol']['path']).parent)
    p = dict(created_utc=paired.old.utc(), phase='node_hour_pq_development', oracle=False,
        variant='band_sdp' if use_sdp else 'band', smoke=smoke,
        inputs=io.ref(out/'inputs.json'), baseline_audit=sp['baseline_audit'],
        source_protocol=sp['source_protocol'], source_completion=sp['source_completion'],
        source_run=sp['source_run'], baseline_source_differences=changes,
        single_baseline_run=str(baseline), single_baseline_audit=io.ref(single_audit),
        band_baseline_audit=None if band_baseline_audit is None else io.ref(band_baseline_audit),
        band_baseline_run=band_run,
        preflight_audit=None if preflight is None else io.ref(preflight),
        modes=list(modes), policies=['single_'+m for m in modes],
        seed=sp['seed'], forecasts=1, repeats=1 if smoke else 16, workers=3,
        solver_seconds=2. if smoke else float(seconds),
        sdp_seconds=8. if smoke else float(sdp_seconds), use_sdp=bool(use_sdp),
        expected_quotes=len(entries)*len(modes), expected_rows=len(entries)*len(modes)*(1 if smoke else 16),
        expected_reused_baseline_rows=len(entries)*8*(1 if smoke else 16),
        target_rule=sp['target_rule'], competitor_rule=sp['competitor_rule'], guard_rule=sp['guard_rule'],
        bandwidth_rule='Training-only global centered OLS; node/clock-hour band is the range of affine residuals in that hour; no multiplier, implicit floor, validation, forecast or outcome tuning',
        bandwidth_timezone='Europe/Berlin', bandwidth_start_hour=8,
        objective=sp['objective'], reservation_prices=sp['reservation_prices'],
        forecast_relative_sd=sp['forecast_relative_sd'],
        shared_seed_source='Unchanged original LP/anchor pool; SDP variant adds only sealed historical single-target SDP-derived points before QCQP solving',
        sdp_scope='Optional hourly block Shor relaxation; original physical and P/Q-band linear domain; extracted actions independently checked; numeric relaxation objective is not a certified lower bound',
        sdp_budget_scope='Additional SDP construction/solve/extraction budget is reported separately from the same600s QCQP limit; not an equal-total-time speed comparison',
        zero_rule=sp['zero_rule'], fallback=sp['fallback'], pairing=sp['pairing'], independence=sp['independence'],
        sources=refs, environment=io.environment())
    p.update(paired.old._git_receipt([Path(r['path']) for r in refs]))
    io.write(out/'protocol.json', p)
    return p


def run_case(entry):
    p, out = paired.old.WORK['p'], paired.old.WORK['out']
    folder = out/'cases'/entry['id']
    if folder.exists():
        raise FileExistsError(folder)
    folder.mkdir(parents=True)
    began = perf_counter()
    io.write(folder/'started.json', dict(pid=os.getpid(), created_utc=paired.old.utc()))
    planning, forecast, physical, original_seeds = paired.context_planning(entry)
    band_meta = io.read(paired.old.checked(entry['node_time_bandwidth']))
    band_arrays = paired.arrays(paired.old.checked(band_meta['arrays']))
    case = domain.NodeTimePQCase(physical, band_arrays, band_meta)
    io.write(folder/'resource_case.json', physical.config)
    shutil.copyfile(band_meta['arrays']['path'], folder/'bandwidth.npz')
    io.write(folder/'bandwidth.json', band_meta)
    shutil.copyfile(entry['initial_points']['path'], folder/'original_initial_points.npz')
    models = {mode:SingleTargetEvidence(case, planning['score'], planning['val'], forecast,
        planning['base'], mode=mode, meter_relative_sd=entry['meter']['additional_V_relative_sd'],
        forecast_relative_sd=p['forecast_relative_sd']) for mode in p['modes']}
    seeds = {name:value.copy() for name,value in original_seeds.items()}
    sdp_receipts, seed_origins = {}, {}
    if p['use_sdp']:
        from . import sdp
        for number, mode in enumerate(p['modes']):
            tick = perf_counter()
            try:
                zero = np.zeros(case.shape)
                if case.audit(zero)['passed'] and models[mode].verify_expansion(zero)['feasible']:
                    receipt, candidates = dict(status='zero_sufficient_skip_sdp', optimizer_invoked=False), {}
                else:
                    receipt, candidates = sdp.solve(case, models[mode], seconds=p['sdp_seconds'],
                        seed=p['seed']+number, record_directory=folder/(mode+'_sdp'))
            except Exception as exc:
                receipt, candidates = dict(status='sdp_exception', error=repr(exc), traceback=traceback.format_exc()), {}
            accepted, duplicate = [], []
            for label, candidate in candidates.items():
                name = 'sdp_'+mode+'_'+label
                action = case.array(candidate)
                if not case.audit(action)['passed']:
                    raise ValueError('SDP extraction outside the original linear resource/band domain')
                if any(np.array_equal(action, value) for value in seeds.values()):
                    duplicate.append(label)
                    continue
                seeds[name] = action.copy()
                seed_origins[name] = dict(mode=mode, label=label, receipt=io.ref(folder/(mode+'_sdp')/'receipt.json')
                    if (folder/(mode+'_sdp')/'receipt.json').exists() else None,
                    direct_evidence=models[mode].verify_expansion(action), band=case.band_audit(action))
                accepted.append(name)
            receipt.update(wall_seconds=perf_counter()-tick, added_seed_names=accepted,
                duplicate_of_existing_pool_labels=duplicate,
                selection_scope='Resource/band-feasible warm point; each probability gate is rechecked by the original QCQP engine')
            io.write(folder/(mode+'_sdp_receipt.json'), receipt)
            sdp_receipts[mode] = io.ref(folder/(mode+'_sdp_receipt.json'))
            print(entry['id'], mode, 'SDP', receipt['status'], 'warm_points', len(accepted), flush=True)
    np.savez_compressed(folder/'shared_initial_points.npz', **seeds)
    io.write(folder/'sdp_seed_origins.json', seed_origins)
    seed_ref = io.ref(folder/'shared_initial_points.npz')
    controls, plans = {}, {}
    for mode in p['modes']:
        name = 'single_'+mode
        tick = perf_counter()
        io.write(folder/(name+'_started.json'), dict(pid=os.getpid(), allowed_seconds=p['solver_seconds']))
        try:
            if p['use_sdp'] and not seed_origins and p['band_baseline_run'] is not None:
                prior_case = Path(p['band_baseline_run'])/'cases'/entry['id']
                prior_protocol = io.read(Path(p['band_baseline_run'])/'protocol.json')
                prior_seal = io.read(prior_case/'seal.json')
                paired.check_refs([prior_seal[k] for k in ('planner', 'plans', 'score', 'train',
                    'validation', 'resource_case', 'bandwidth_npz', 'original_initial_points')])
                if (prior_seal['score'] != entry['score'] or prior_seal['train'] != entry['files']['train.npz']
                        or prior_seal['validation'] != entry['files']['validation.npz']
                        or io.read(prior_seal['resource_case']['path']) != physical.config
                        or prior_protocol['forecast_relative_sd'] != p['forecast_relative_sd']
                        or prior_protocol['seed'] != p['seed']):
                    raise ValueError('Cannot reuse a quote with different frozen score/history/resources/errors')
                prior_planner = paired.arrays(prior_seal['planner']['path'])
                prior_pool = paired.arrays(prior_seal['original_initial_points']['path'])
                if (not np.array_equal(prior_planner['forecast'], forecast)
                        or set(prior_pool) != set(original_seeds)
                        or any(not np.array_equal(prior_pool[k], original_seeds[k]) for k in prior_pool)):
                    raise ValueError('Cannot reuse a quote with different forecast or original seed pool')
                prior_band = paired.arrays(prior_case/'bandwidth.npz')
                if set(prior_band) != set(band_arrays) or any(not np.array_equal(prior_band[k], band_arrays[k]) for k in prior_band):
                    raise ValueError('Cannot reuse a quote with a different node/hour band')
                prior_quote = io.read(prior_case/(name+'_plan.json'))
                prior_action = paired.arrays(prior_case/(name+'_action.npz'))['action']
                if (prior_quote != io.read(prior_seal['plans']['path'])[name]
                        or not np.array_equal(prior_action, prior_planner[name])):
                    raise ValueError('Reused quote/action differ from sealed planning data')
                receipt = deepcopy(prior_quote)
                receipt.update(reused_quote_ref=io.ref(prior_case/(name+'_plan.json')),
                    reused_action_ref=io.ref(prior_case/(name+'_action.npz')),
                    reused_native_status=receipt.get('native_status'), native_status=None,
                    original_quote_seconds=receipt.get('seconds'), seconds=0., optimizer_invoked=False,
                    source='reused_audited_node_time_band', solver_result_reused=True,
                    reuse_reason='No SDP-derived resource/band-feasible warm point was produced; do not repeat the identical QCQP')
                # Reuse the action only. Never revive a raw bound withdrawn by an older audit.
                cost = receipt.get('cost_yuan')
                receipt.update(reused_raw_lower_bound_yuan=receipt.get('lower_bound_yuan'),
                    lower_bound_yuan=0., relative_gap=None if cost is None else (1. if cost > 1e-12 else 0.),
                    global_optimality_verified=bool(cost is not None and cost <= 1e-12),
                    lower_bound_source='nonnegative_cost_universal_bound',
                    lower_bound_withdrawn=True, lower_bound_withdrawal_reason='Reused action only; old raw native bound is not transferred')
                point = prior_action.copy() if receipt['feasible'] else None
            else:
                receipt, point = engine.plan(case, planning['score'], planning['val'], forecast,
                    planning['base'], mode=mode, seconds=p['solver_seconds'], env=paired.old.WORK['env'],
                    seed_points=seeds, record_directory=folder/(name+'_diagnostics'),
                    meter_relative_sd=entry['meter']['additional_V_relative_sd'],
                    forecast_relative_sd=p['forecast_relative_sd'], evidence_factory=SingleTargetEvidence)
                receipt['solver_result_reused'] = False
            if bool(receipt['feasible']) != (point is not None):
                raise ValueError('Quote and returned action disagree')
        except Exception as exc:
            receipt, point = dict(status='exception', feasible=False, error=repr(exc), traceback=traceback.format_exc()), None
        action = np.zeros(case.shape) if point is None else np.asarray(point).copy()
        check = models[mode].verify_expansion(action)
        band = case.band_audit(action)
        if point is not None and not (case.audit(action)['passed'] and check['feasible']):
            raise ValueError('Final direct original gate/resource/band acceptance failed')
        receipt.update(mode=mode, guard_kind='node_hour_pq', new_gate_feasible=point is not None,
            zero_fallback=point is None, quote_wall_seconds=perf_counter()-tick,
            requested_quote_budget_seconds=p['solver_seconds'], initial_points=seed_ref,
            original_initial_points=entry['initial_points'], sdp_receipts=sdp_receipts,
            executed_source='zero_fallback' if point is None else 'verified_single_target_node_hour_band',
            executed_probability_check=check, band_direct=band, planned_band_feasible=band['passed'],
            executed_cost=prices.cost_breakdown(physical, action),
            target_native=entry['planned_history_target_native'], band_encoding=case.band_encoding,
            raw_bound_scope='Activation cost for the encoded single-target probability QCQP with original resource contracts and node/hour P/Q bands reduced by1e-7 relative numerical buffer; not an unbuffered-band/AC/total-bill lower bound')
        controls[name], plans[name] = action, receipt
        np.savez_compressed(folder/(name+'_action.npz'), action=action)
        io.write(folder/(name+'_plan.json'), receipt)
        print(entry['id'], name, receipt['status'], 'fee', receipt.get('cost_yuan'),
            'fallback', receipt['zero_fallback'], flush=True)
    np.savez_compressed(folder/'planner.npz', forecast=forecast, **controls)
    io.write(folder/'plans.json', plans)
    io.write(folder/'prediction_audits.json', {m:{name:models[m].verify_expansion(u)
        for name,u in controls.items()} for m in p['modes']})
    io.write(folder/'seal.json', dict(created_utc=paired.old.utc(), planner=io.ref(folder/'planner.npz'),
        plans=io.ref(folder/'plans.json'), score=entry['score'], train=entry['files']['train.npz'],
        validation=entry['files']['validation.npz'], bandwidth_json=io.ref(folder/'bandwidth.json'),
        bandwidth_npz=io.ref(folder/'bandwidth.npz'), sampling_source=entry['files']['observations_sealed.json'],
        shared_initial_points=seed_ref, original_initial_points=io.ref(folder/'original_initial_points.npz'),
        shared_initial_points_source=entry['initial_points'], sdp_seed_origins=io.ref(folder/'sdp_seed_origins.json'),
        sdp_receipts=sdp_receipts, protocol=io.ref(out/'protocol.json'),
        resource_case=io.ref(folder/'resource_case.json'), prediction_audits=io.ref(folder/'prediction_audits.json'),
        planned_history_target_native=entry['planned_history_target_native'],
        planner_actual_voltage_or_network_inputs=False))
    # The planning-only band never clips or repairs actual delivery.
    rows = paired.execute(entry, p, folder, planning, physical, forecast, controls, plans, {}, dict(guards=[]))
    good = [r for r in rows if r['status'] == 'complete']
    result = dict(status='complete', id=entry['id'], rows=len(rows), failed_rows=len(rows)-len(good),
        quote_exceptions=sum(r['status']=='exception' for r in plans.values()),
        feasible_quotes=sum(r['feasible'] for r in plans.values()),
        zero_fallbacks=sum(r['zero_fallback'] for r in plans.values()),
        zero_sufficient_quotes=sum(r['status']=='zero_sufficient' for r in plans.values()),
        reused_ac_rows=sum('reused_action_source' in r for r in good),
        new_ac_rows=sum('reused_action_source' not in r for r in good), seconds=perf_counter()-began,
        plans=io.ref(folder/'plans.json'), results=io.ref(folder/'results.json'))
    io.write(folder/'complete.json', result)
    return result


def run(out):
    out = Path(out).resolve()
    p = io.read(out/'protocol.json')
    paired.check_refs([p['inputs'], *p['sources']])
    if (out/'lock.json').exists() or (out/'complete.json').exists():
        raise RuntimeError('Preserve existing lock and completed run')
    io.write(out/'lock.json', dict(pid=os.getpid(), created_utc=paired.old.utc()))
    receipts, failures = [], []
    try:
        with ProcessPoolExecutor(max_workers=p['workers'], initializer=paired.old.initialize,
                initargs=(str(out),)) as pool:
            tasks = {pool.submit(run_case, e):e for e in io.read(p['inputs']['path'])}
            for task in as_completed(tasks):
                try:
                    receipts.append(task.result())
                except Exception as exc:
                    failures.append(dict(id=tasks[task]['id'], error=repr(exc), traceback=traceback.format_exc()))
                io.write(out/'progress.json', dict(receipts=receipts, failed_tasks=failures))
        result = dict(status='complete' if not failures else 'incomplete', receipts=receipts, failed_tasks=failures,
            **{k:sum(r[k] for r in receipts) for k in ('rows','failed_rows','quote_exceptions','feasible_quotes',
                'zero_fallbacks','zero_sufficient_quotes','reused_ac_rows','new_ac_rows')},
            source_unchanged=all(io.ref(r['path']) == r for r in p['sources']))
        io.write(out/'complete.json', result)
        return result
    finally:
        (out/'lock.json').unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--seconds', type=float, default=600.)
    parser.add_argument('--preflight', type=Path)
    parser.add_argument('--sdp', action='store_true')
    parser.add_argument('--sdp-seconds', type=float, default=120.)
    parser.add_argument('--band-baseline-audit', type=Path)
    args = parser.parse_args()
    prepare(args.out, smoke=args.smoke, seconds=args.seconds, preflight=args.preflight,
        use_sdp=args.sdp, sdp_seconds=args.sdp_seconds, band_baseline_audit=args.band_baseline_audit)
    print(run(args.out))


if __name__ == '__main__':
    main()
