"""Read-only numerical replay of completed measured-P/Q DFS records.

No runner decomposition, native selector, domain audit, optimizer, or AC solver
is called. Predictions are rebuilt from branch supports; device energy states,
fees, Gaussian streams, directed evidence, and final selection are recomputed.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
from itertools import combinations
import json
from pathlib import Path
import shutil
import traceback

import numpy as np

from ... import io


ROOT = Path(__file__).resolve().parents[3]


def arrays(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key].copy() for key in data.files}


def random_stream(protocol, entry, draw, repeat, component):
    sequence = [int(protocol['seed']), 914, int(entry['index']), int(draw), int(repeat), int(component)]
    return np.random.default_rng(np.random.SeedSequence(sequence))


class Verify:
    def __init__(self):
        self.errors, self.failed_executions, self.failed_tasks = [], [], []
        self.maxima = defaultdict(float)
        self.counts = Counter()
        self.hashes = {}
        self.contexts = {}
        self.fee_mismatches = []
        self.where = ''

    def require(self, condition, message):
        if not condition:
            raise AssertionError(message)

    def same(self, name, actual, expected, atol=1e-9, rtol=1e-8):
        a, b = np.asarray(actual), np.asarray(expected)
        self.require(a.shape == b.shape, f'{name}: shape {a.shape} != {b.shape}')
        if a.size:
            self.require(np.isfinite(a).all() and np.isfinite(b).all(), name+': nonfinite')
            error = float(np.max(np.abs(a-b)))
            self.maxima[name] = max(self.maxima[name], error)
            self.require(bool(np.allclose(a, b, atol=atol, rtol=rtol)), f'{name}: difference {error}')
        self.counts['numeric_comparisons'] += 1

    def checked(self, ref):
        path = str(Path(ref['path']).resolve())
        if path not in self.hashes:
            self.hashes[path] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        self.require(self.hashes[path] == ref['sha256'], 'Sealed hash differs: '+path)
        return Path(path)

    def context(self, entry):
        if entry['id'] in self.contexts:
            return self.contexts[entry['id']]
        native = io.read(self.checked(entry['native_copy']))
        score = io.read(self.checked(entry['score']))
        validation = arrays(self.checked(entry['files']['validation.npz']))
        original = arrays(self.checked(entry['physical_cache']['original']))
        labels = tuple(map(int, validation['labels']))
        n = len(labels)
        H, C, active, internal = [], [], [], []
        for row in native['frozen_candidates']:
            R = np.zeros((n, n)); X = np.zeros_like(R)
            RC = np.zeros_like(R); XC = np.zeros_like(R)
            point = row['point']; supports = set()
            for group, r, x in zip(point['supports'], point['r'], point['x'], strict=True):
                block = np.ix_(group, group)
                R[block] += r; X[block] += x
                if len(group) == n:
                    RC[block] += r; XC[block] += x
                if max(r, x) > native['config']['zero_tol']:
                    supports.add(tuple(group))
            H.append(np.c_[R, X]); C.append(np.c_[RC, XC]); active.append(supports)
            internal.append(sum(1 < len(group) < n for group in point['supports']))
        H, C = np.asarray(H), np.asarray(C)
        self.same('frozen_full_H', [row['H'] for row in score['native_bank']], H, atol=1e-13, rtol=1e-13)
        P = np.c_[validation['P'], validation['Q']]
        y = validation['Y']
        history = np.array([np.sum((y-P@matrix.T)**2) for matrix in H])
        history0 = np.array([np.sum((y-P@matrix.T)**2) for matrix in H-C])
        self.same('historical_sse', [row['history_sse'] for row in score['native_bank']], history, atol=1e-15)
        case_root = ROOT/'topology_identification_standalone'/'cases'/entry['case']
        metadata = io.read(case_root/'metadata.json')
        children = defaultdict(list)
        with (case_root/'branches.csv').open(encoding='utf-8-sig', newline='') as stream:
            for branch in csv.DictReader(stream):
                if branch['status'].strip().lower() == 'true':
                    children[int(branch['from_bus'])].append(int(branch['to_bus']))
        positions = {label: i for i, label in enumerate(labels)}
        def descendants(node):
            result = {positions[node]} if node in positions else set()
            for child in children[node]:
                result.update(descendants(child))
            return result
        truth = {tuple(sorted(descendants(child))) for row in list(children.values()) for child in row}
        truth.discard(())
        ctx = dict(native=native, score=score, labels=labels, n=n, H=H, C=C, active=active,
                   internal=np.asarray(internal), history=history, history0=history0,
                   validation_cells=y.size, sigma2=float(score['sigma2']), truth=truth,
                   base=float(metadata['base_mva'])*1000.,
                   baseline_P=original['P'][288:384], baseline_Q=original['Q'][288:384],
                   baseline_root=original['root'][288:384], target=int(score['native_index']))
        self.contexts[entry['id']] = ctx
        self.counts['histories'] += 1
        return ctx

    def evaluate(self, ctx, power, y=None, observed_power=None, observed_y=None):
        """Direct residual contrasts first; fixed direction has no index reversal."""
        predictions = [power@matrix.T for matrix in ctx['H']]
        noncommon = [power@matrix.T for matrix in ctx['H']-ctx['C']]
        losses = None if y is None else np.array([np.sum((y-prediction)**2) for prediction in predictions])
        measured = None if observed_y is None else np.array([
            np.sum((observed_y-observed_power@matrix.T)**2) for matrix in ctx['H']])
        reduced = None if observed_y is None else np.array([
            np.sum((observed_y-observed_power@matrix.T)**2) for matrix in ctx['H']-ctx['C']])
        return dict(pred=predictions, noncommon=noncommon, y=y, clean_sse=losses,
                    observed_sse=measured, reduced_sse=reduced)

    def directed(self, ctx, values, i, j):
        difference = values['pred'][i]-values['pred'][j]
        without = values['noncommon'][i]-values['noncommon'][j]
        common = difference-without
        variance = ctx['sigma2']
        answer = dict(h=(ctx['history'][j]-ctx['history'][i])/(2*variance),
                      J=float(np.sum(difference*difference)/(2*variance)),
                      J_without_common=float(np.sum(without*without)/(2*variance)),
                      J_common=float(np.sum(common*common)/(2*variance)),
                      J_cross=float(np.sum(common*without)/variance))
        self.same('J_common_identity', answer['J'], answer['J_without_common']+
                  answer['J_common']+answer['J_cross'])
        if values['y'] is not None:
            clean = float((values['clean_sse'][j]-values['clean_sse'][i])/(2*variance))
            answer['bias'] = float(np.sum((values['y']-values['pred'][i])*difference)/variance)
            answer['clean_logodds'] = answer['h']+clean
            self.same('clean_direct_identity', clean, answer['J']+answer['bias'], atol=1e-8)
            if values['observed_sse'] is not None:
                measured = float((values['observed_sse'][j]-values['observed_sse'][i])/(2*variance))
                answer['observed_logodds'] = answer['h']+measured
                answer['meter'] = measured-clean
                total0 = ctx['history0']+values['reduced_sse']
                answer['observed_logodds_without_common'] = float((total0[j]-total0[i])/(2*variance))
        return answer

    def pair_file(self, path, ctx, values):
        saved = arrays(path)
        pairs = list(combinations(range(len(ctx['H'])), 2))
        self.same('pair_indices', saved['pairs'], np.asarray(pairs).reshape(-1, 2), atol=0, rtol=0)
        computed = [self.directed(ctx, values, i, j) for i, j in pairs]
        for key in computed[0] if computed else []:
            self.same('pair_'+key, saved[key], [row[key] for row in computed], atol=1e-8)
        if values['observed_sse'] is not None:
            self.same('pair_native_new_sse', saved['native_new_sse'], values['observed_sse'], atol=1e-14)
            self.same('pair_reduced_new_sse', saved['native_new_sse_without_common'], values['reduced_sse'], atol=1e-14)
            self.same('pair_reduced_total_sse', saved['native_total_sse_without_common'],
                      ctx['history0']+values['reduced_sse'], atol=1e-14)
        self.counts['pair_arrays_verified'] += 1
        return computed

    def native_selection(self, ctx, observation, row, values):
        total = ctx['history']+values['observed_sse']
        self.same('saved_history_sse', observation['native_history_sse'], ctx['history'], atol=1e-14)
        self.same('saved_new_sse', observation['native_new_sse'], values['observed_sse'], atol=1e-14)
        self.same('saved_total_sse', observation['native_total_sse'], total, atol=1e-14)
        scaled = total/(ctx['validation_cells']+observation['Y'].size)/ctx['score']['reference_mse']
        eligible = [i for i, value in enumerate(scaled)
                    if value <= min(scaled)+ctx['score']['validation_tie']]
        selected = min(eligible, key=lambda i: (ctx['internal'][i], scaled[i], i))
        self.require(selected == row['native_selected_index'], 'Native strict MSE/tie decision differs')
        active, truth = ctx['active'][selected], ctx['truth']
        proper = lambda supports: {group for group in supports if 1 < len(group) < ctx['n']}
        desired = dict(native_map_correct=proper(active) == proper(truth),
                       native_missing=len(proper(truth)-proper(active)),
                       native_extra=len(proper(active)-proper(truth)),
                       native_full_support_exact=active == truth,
                       native_full_support_missing=len(truth-active),
                       native_full_support_extra=len(active-truth))
        for key, expected in desired.items():
            self.require(row[key] == expected, key+' differs')
        representatives = [model['index'] for model in ctx['score']['models']]
        ll = -(total[representatives]-min(total[representatives]))/(2*ctx['sigma2'])
        ll -= np.log(np.exp(ll).sum())
        self.same('representative_log_probability', observation['representative_log_probability'], ll, atol=1e-8)
        self.same('p_target', row['p_target'], np.exp(ll[ctx['score']['map_index']]), atol=1e-10)
        self.same('p_max', row['p_max'], np.exp(max(ll)), atol=1e-10)
        self.counts['native_selections_verified'] += 1

    def device_values(self, config, labels, forecast, base, action):
        """Hourly loops independently replay device bounds, states, and fees."""
        self.require(action.shape == (24, len(config['resources'])), 'Action shape differs')
        nodes = {label: i for i, label in enumerate(labels)}
        dp = np.zeros((96, len(labels))); dq = np.zeros_like(dp)
        fee, violation, final_error = 0., 0., 0.
        for j, resource in enumerate(config['resources']):
            node = nodes[int(resource['node_label'])]
            cap, state = float(resource['contract_kw']), 0.
            for h in range(24):
                clock = (8+h+.5) % 24; value = float(action[h, j])
                if resource['kind'] == 'pv':
                    export = max(0., -float(np.max(forecast[4*h:4*h+4, node]))*base)
                    low, high = 0., min(cap, export) if 10 <= clock < 14 else 0.
                elif resource['kind'] == 'ev':
                    low, high = (-cap, cap) if clock >= 18 or clock < 7 else (0., 0.)
                    state += value
                    violation = max(violation, abs(state)-2*cap)
                else:
                    low, high = -cap, cap
                    state = .98*state+value
                    violation = max(violation, abs(state)-cap)
                violation = max(violation, low-value, value-high)
                factor = 1.25 if (7 <= clock < 10 or 17 <= clock < 22) else .8 if 0 <= clock < 7 else 1.
                fee += factor*(resource['fee_positive_yuan_per_kwh']*max(value, 0.)+
                               resource['fee_negative_yuan_per_kwh']*max(-value, 0.))
                dp[4*h:4*h+4, node] += value
                dq[4*h:4*h+4, node] += value*resource['q_over_p']
            if resource['kind'] != 'pv':
                final_error = max(final_error, abs(state))
        self.require(violation <= 1e-7 and final_error <= 1e-7,
                     f'Resource bounds/state failed: {violation}, {final_error}')
        self.maxima['device_inequality_violation'] = max(self.maxima['device_inequality_violation'], violation)
        self.maxima['device_terminal_state_error'] = max(self.maxima['device_terminal_state_error'], final_error)
        self.counts['device_actions_verified'] += 1
        return fee, dp/base, dq/base

    def forecast(self, directory, entry, protocol):
        ctx = self.context(entry)
        seal = io.read(directory/'seal.json')
        planner = arrays(self.checked(seal['planner']))
        plans = io.read(self.checked(seal['plans']))
        self.checked(seal['score']); self.checked(seal['protocol'])
        prediction = planner.pop('forecast')
        draw = int(directory.name.removeprefix('forecast'))
        P0 = np.c_[ctx['baseline_P'], ctx['baseline_Q']]
        expected_forecast = P0*(1+.05*random_stream(protocol, entry, draw, 0, 991).normal(size=P0.shape))
        self.same('forecast_randomness', prediction, expected_forecast, atol=0, rtol=0)
        config = io.read(directory/'resource_case.json')
        modes = protocol['modes']
        strategies = ['no_dfs']+[s for mode in modes for s in (mode, mode+'_rotation')]
        self.require(set(planner) == set(strategies), 'Missing/extra strategy in sealed planner')
        fees, forecast_values, exact_values = {}, {}, {}
        for strategy in strategies:
            u = planner[strategy]
            fee, dp, dq = self.device_values(config, ctx['labels'], prediction, ctx['base'], u)
            fees[strategy] = fee
            forecast_values[strategy] = self.evaluate(ctx, prediction+np.c_[dp, dq])
            path = directory/'exact_requested'/strategy
            receipt = io.read(path/'receipt.json')
            if receipt['status'] != 'complete':
                self.counts['exact_AC_failures_preserved'] += 1
                continue
            clean = arrays(path/'clean_observation.npz')
            self.same('exact_requested_P', clean['P'], ctx['baseline_P']+dp, atol=1e-13)
            self.same('exact_requested_Q', clean['Q'], ctx['baseline_Q']+dq, atol=1e-13)
            self.same('exact_requested_Y', clean['Y'], clean['root'][:, None]**2-clean['V']**2, atol=1e-14)
            exact_values[strategy] = self.evaluate(ctx, np.c_[clean['P'], clean['Q']], clean['Y'],
                                                  np.c_[clean['P'], clean['Q']], clean['Y'])
            self.pair_file(path/'pair_evidence.npz', ctx, exact_values[strategy])
            self.pair_file(path/'forecast_pair_evidence.npz', ctx, forecast_values[strategy])
            self.same('exact_requested_fee', receipt['requested_fee_yuan'], fee, atol=1e-8)
        for mode, plan in plans.items():
            self.counts['quotes'] += 1
            self.counts['quote_status/'+plan['status']] += 1
            if plan.get('zero_fallback'):
                self.same('zero_fallback', planner[mode], np.zeros_like(planner[mode]), atol=0, rtol=0)
            elif plan.get('cost_yuan') is not None:
                self.same('quote_cost', plan['cost_yuan'], fees[mode], atol=1e-7)
            self.same('rotation_fee', plan['rotation_fee_yuan'], fees[mode+'_rotation'], atol=1e-8)
            matched = abs(fees[mode]-fees[mode+'_rotation']) <= 1e-7
            self.require(plan['rotation_same_fee'] == matched, 'Rotation fee-match declaration differs')
            if not matched:
                self.fee_mismatches.append(dict(task=str(directory), mode=mode,
                                                planned=fees[mode], rotation=fees[mode+'_rotation']))
        results = io.read(directory/'results.json')
        self.require(len(results) == protocol['repeats']*len(strategies), 'Execution/strategy denominator differs')
        by_key = {(row['repeat'], row['strategy']): row for row in results}
        self.require(len(by_key) == len(results), 'Duplicate execution receipt')
        for repeat in range(protocol['repeats']):
            rd = directory/f'execution{repeat:02d}'
            random = arrays(rd/'paired_randomness.npz')
            self.same('eta_randomness', random['eta'], random_stream(protocol, entry, draw, repeat, 773).uniform(
                .9, 1., len(config['resources'])), atol=0, rtol=0)
            for k, shape in enumerate((ctx['baseline_P'].shape, ctx['baseline_Q'].shape,
                                       ctx['baseline_P'].shape, ctx['baseline_root'].shape)):
                self.same('meter_randomness', random[f'noise{k}'], random_stream(
                    protocol, entry, draw, repeat, k+2).normal(size=shape), atol=0, rtol=0)
            computed, computed_pairs = {}, {}
            for strategy in strategies:
                self.where = str(rd/strategy)
                row = by_key[(repeat, strategy)]
                self.require(row == io.read(rd/strategy/'receipt.json'), 'Combined and individual receipt differ')
                self.counts['execution_rows'] += 1
                if row['status'] != 'complete':
                    self.failed_executions.append(dict(path=self.where, error=row.get('error'), status=row['status']))
                    continue
                try:
                    obs = arrays(rd/strategy/'observations.npz')
                    u = planner[strategy]
                    self.same('requested_action', obs['requested_kw'], u, atol=0, rtol=0)
                    delivered = u*random['eta'][None]
                    for j, resource in enumerate(config['resources']):
                        if resource['kind'] == 'pv':
                            node = ctx['labels'].index(int(resource['node_label']))
                            for h in range(24):
                                available = max(0., -float(np.max(ctx['baseline_P'][4*h:4*h+4, node]))*ctx['base'])
                                delivered[h, j] = min(delivered[h, j], resource['contract_kw'], available)
                    self.same('delivered_action', obs['delivered_kw'], delivered, atol=1e-12)
                    fee, dp, dq = self.device_values(config, ctx['labels'], prediction, ctx['base'], delivered)
                    self.same('actual_P', obs['true_P'], ctx['baseline_P']+dp, atol=1e-13)
                    self.same('actual_Q', obs['true_Q'], ctx['baseline_Q']+dq, atol=1e-13)
                    self.same('physical_root_shared', obs['true_root'], ctx['baseline_root'], atol=0, rtol=0)
                    pq_sd = protocol['meter']['additional_PQ_relative_sd']
                    v_sd = protocol['meter']['additional_V_relative_sd']
                    self.same('meter_P', obs['P'], obs['true_P']*(1+pq_sd*random['noise0']), atol=1e-14)
                    self.same('meter_Q', obs['Q'], obs['true_Q']*(1+pq_sd*random['noise1']), atol=1e-14)
                    self.same('meter_V', obs['V'], obs['true_V']*(1+v_sd*random['noise2']), atol=1e-14)
                    self.same('meter_root', obs['root'], obs['true_root']*(1+v_sd*random['noise3']), atol=1e-14)
                    self.same('observed_Y', obs['Y'], obs['root'][:, None]**2-obs['V']**2, atol=1e-14)
                    self.same('planned_fee', row['planned_fee_yuan'], fees[strategy], atol=1e-8)
                    self.same('settled_fee', row['settled_fee_yuan'], fee, atol=1e-8)
                    self.same('requested_energy', row['requested_abs_kwh'], np.abs(u).sum(), atol=1e-8)
                    self.same('delivered_energy', row['delivered_abs_kwh'], np.abs(delivered).sum(), atol=1e-8)
                    y = obs['true_root'][:, None]**2-obs['true_V']**2
                    values = self.evaluate(ctx, np.c_[obs['true_P'], obs['true_Q']], y,
                                           np.c_[obs['P'], obs['Q']], obs['Y'])
                    computed[strategy] = values
                    self.native_selection(ctx, obs, row, values)
                    computed_pairs[strategy] = self.pair_file(rd/strategy/'pair_evidence.npz', ctx, values)
                    self.counts['successful_execution_rows_verified'] += 1
                except Exception as exc:
                    self.errors.append(dict(path=self.where, error=repr(exc), traceback=traceback.format_exc()))
            if 'no_dfs' not in computed:
                continue
            for strategy, values in computed.items():
                self.where = str(rd/strategy)
                try:
                    row = by_key[(repeat, strategy)]
                    saved_changes = arrays(rd/strategy/'pair_changes.npz')
                    before, after = computed_pairs['no_dfs'], computed_pairs[strategy]
                    for key in saved_changes:
                        if key == 'maximum_identity_error':
                            continue
                        self.same('change_'+key, saved_changes[key],
                                  [a[key]-b[key] for a, b in zip(after, before)], atol=1e-8)
                    records = row['evidence']['target_pairs']
                    self.require(len(records) == len(ctx['H'])-1, 'Directed target-pair count differs')
                    for record in records:
                        t, j = record['target'], record['competitor']
                        self.require(t == ctx['target'] and j != t, 'Historical native target changed')
                        baseline = self.directed(ctx, computed['no_dfs'], t, j)
                        actual = self.directed(ctx, values, t, j)
                        predicted0 = self.directed(ctx, forecast_values['no_dfs'], t, j)
                        predicted = self.directed(ctx, forecast_values[strategy], t, j)
                        expected = dict(actual_physical_J_gain=actual['J']-baseline['J'],
                            actual_bias_change=actual['bias']-baseline['bias'],
                            actual_meter_change=actual['meter']-baseline['meter'],
                            actual_logodds_change=actual['observed_logodds']-baseline['observed_logodds'],
                            same_observations_logodds_change_without_common=
                                actual['observed_logodds_without_common']-baseline['observed_logodds_without_common'],
                            forecast_J_gain=predicted['J']-predicted0['J'],
                            forecast_J_gain_without_common=predicted['J_without_common']-predicted0['J_without_common'],
                            forecast_common_square_gain=predicted['J_common']-predicted0['J_common'],
                            forecast_cross_gain=predicted['J_cross']-predicted0['J_cross'])
                        if strategy in exact_values and 'no_dfs' in exact_values:
                            exact = self.directed(ctx, exact_values[strategy], t, j)
                            exact0 = self.directed(ctx, exact_values['no_dfs'], t, j)
                            expected['exact_physical_J_gain'] = exact['J']-exact0['J']
                        for key, value in expected.items():
                            self.same('directed_'+key, record[key], value, atol=1e-8)
                        self.same('directed_identity', expected['actual_logodds_change'],
                                  expected['actual_physical_J_gain']+expected['actual_bias_change']+
                                  expected['actual_meter_change'], atol=1e-8)
                        self.counts['directed_target_pairs_verified'] += 1
                except Exception as exc:
                    self.errors.append(dict(path=self.where, error=repr(exc), traceback=traceback.format_exc()))
            self.counts['paired_execution_paths'] += 1
        self.counts['forecast_tasks_verified'] += 1


def run(run_path, destination, *, partial=False):
    run_path, destination = Path(run_path).resolve(), Path(destination).resolve()
    if destination.exists():
        raise FileExistsError(destination)
    if not partial and not (run_path/'complete.json').exists():
        raise RuntimeError('Require a completed batch, or explicitly request partial interface verification')
    verify = Verify()
    protocol = io.read(run_path/'protocol.json')
    entries = io.read(verify.checked(protocol['inputs']))
    by_id = {entry['id']: entry for entry in entries}
    fractions = []
    for fraction_dir in sorted(run_path.glob('fraction*')):
        if not fraction_dir.is_dir():
            continue
        tasks = sorted(fraction_dir.glob('*/forecast*/complete.json'))
        expected = len(entries)*protocol['forecasts']
        fractions.append(dict(fraction=fraction_dir.name, expected_forecast_problems=expected,
                              completed_forecast_problems=len(tasks),
                              expected_method_quotes=expected*len(protocol['modes']),
                              expected_paired_execution_paths=expected*protocol['repeats']))
        completion = run_path/f'complete_{fraction_dir.name}.json'
        if completion.exists():
            receipt = io.read(completion)
            verify.failed_tasks.extend(receipt.get('failed_tasks', []))
            if not partial:
                verify.require(len(tasks)+len(receipt.get('failed_tasks', [])) == expected,
                               'Forecast tasks do not match declared denominator')
        for path in tasks:
            verify.where = str(path.parent)
            try:
                verify.forecast(path.parent, by_id[path.parent.parent.name], protocol)
            except Exception as exc:
                verify.errors.append(dict(path=verify.where, error=repr(exc), traceback=traceback.format_exc()))
    report = dict(status='passed' if not verify.errors else 'failed', partial=partial,
        verified_utc=datetime.now(timezone.utc).isoformat(), run=io.ref(run_path/'protocol.json'),
        source=io.ref(__file__), fractions=fractions, counts=dict(verify.counts),
        maximum_absolute_errors=dict(verify.maxima), errors=verify.errors,
        original_failed_tasks=verify.failed_tasks, original_failed_executions=verify.failed_executions,
        unmatched_planned_rotation_fees=verify.fee_mismatches,
        unique_input_histories=len(entries), forecast_draws_per_history=protocol['forecasts'],
        conditional_execution_repeats_per_forecast=protocol['repeats'],
        scope='Independent stored-observation numerical replay; no new fitting, optimization, AC solve, or probability calibration',
        limitations='Physical AC equations and solver global bounds are not recomputed here; original failures and unmatched rotation fees remain visible')
    destination.mkdir(parents=True)
    shutil.copyfile(Path(__file__), destination/'verify_run.py')
    io.write(destination/'report.json', report)
    io.write(destination/'complete.json', dict(report=io.ref(destination/'report.json')))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--partial', action='store_true')
    args = parser.parse_args()
    report = run(args.run, args.out, partial=args.partial)
    print(json.dumps(dict(status=report['status'], counts=report['counts'], errors=len(report['errors'])), ensure_ascii=False))
    if report['errors']:
        raise SystemExit(1)
