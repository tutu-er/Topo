"""Independent stored-observation replay of the P/Q flexibility experiment.

No resource-domain class, runner scoring, optimizer, or AC solve is called.
Native models are reconstructed from supports by the prior independent
verifier; P/Q actions, units, costs, random streams and evidence are replayed.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import shutil
import traceback

import numpy as np

from ... import io
from ..measured_pq_flex_20261114 import verify_run as previous


class Verify(previous.Verify):
    def __init__(self):
        super().__init__()
        self.shared = {}

    def paired(self, key, name, value):
        key = (key, name)
        if key in self.shared:
            self.same('cross_placement_'+name, value, self.shared[key], atol=0, rtol=0)
            self.counts['cross_placement_arrays_verified'] += 1
        else:
            self.shared[key] = np.asarray(value).copy()

    def config_check(self, entry, config, labels):
        resources = config['resources']
        units = config['action_units']
        self.require(len(resources) == len(units), 'Resource/unit count differs')
        P_R = sum(unit == 'kW' for unit in units)
        self.require(units == ['kW']*P_R+['kvar']*(len(units)-P_R), 'P prefix/Q suffix ordering differs')
        self.require(len({r['id'] for r in resources}) == len(resources), 'Duplicate resource ID')
        self.require(all(r['node_label'] in labels for r in resources), 'Resource on an unmeasured terminal')
        context = io.read(self.checked(entry['resource_context']))
        registered = {int(r['feeder']):r for r in context['resource_metadata_at_history_start']}
        expected = []
        for kind, field, count, price in [('pv', 'PV_systems_kW', 3, .3),
                ('ev', 'EV_chargers_kW', 2, .1), ('thermal', 'hot_water_tanks_kW', 2, .15)]:
            available = []
            for node in context['node_mapping']:
                capacity = float(registered[int(node['feeder_id'])].get(field, 0.) or 0.)*float(node['scale'])
                if capacity > 0:
                    available.append((capacity, int(node['label'])))
            for capacity, label in sorted(available, key=lambda item: (-item[0], item[1]))[:count]:
                expected.append((kind, label, .1*capacity, price))
        self.require(len(expected) == P_R, 'Original P catalog size differs')
        for actual, (kind, label, capacity, price) in zip(resources[:P_R], expected, strict=True):
            self.require(actual['kind'] == kind and actual['node_label'] == label, 'Original P order/site differs')
            self.same('P_catalog_capacity', actual['contract_kw'], capacity, atol=1e-12)
            self.same('P_catalog_prices', [actual['fee_positive_yuan_per_kwh'], actual['fee_negative_yuan_per_kwh']],
                      [price, price], atol=0, rtol=0)
            self.same('P_catalog_qratio', actual['q_over_p'], 0., atol=0, rtol=0)
        self.require(len(entry['q_resources']) == len(resources)-P_R, 'Q catalog size differs')
        for actual, expected in zip(resources[P_R:], entry['q_resources'], strict=True):
            self.require(actual['id'] == expected['id'] and actual['node_label'] == expected['node_label'], 'Q identity differs')
            self.require(actual['kind'] == 'reactive', 'Q resource kind differs')
            for field in ('capacity_kvar', 'fee_positive_yuan_per_kvarh', 'fee_negative_yuan_per_kvarh'):
                self.same('Q_catalog_'+field, actual[field], expected[field], atol=0, rtol=0)
            self.same('Q_reservation_price', actual.get('reservation_fee_yuan_per_kvar_day', 0.), 0., atol=0, rtol=0)
            hours = expected.get('available_clock_hours')
            hours = [0, 24] if hours == list(range(24)) else hours
            self.require(actual.get('available_clock_hours') == hours, 'Q availability differs')
        self.counts['resource_catalogs_verified'] += 1
        return P_R

    def device_values(self, config, labels, forecast, base, action):
        """Direct hourly scalar checks, without the production map or audit."""
        resources, units = config['resources'], config['action_units']
        self.require(action.shape == (24, len(resources)), 'Action shape differs')
        self.require(np.isfinite(action).all(), 'Nonfinite action')
        nodes = {int(label): i for i, label in enumerate(labels)}
        dp, dq = np.zeros((96, len(labels))), np.zeros((96, len(labels)))
        amounts = dict(active_abs_kwh=0., reactive_abs_kvarh=0., active_yuan=0.,
                       reactive_yuan=0., reservation_yuan=0., total_yuan=0.)
        violation, final_error = 0., 0.
        for j, (resource, unit) in enumerate(zip(resources, units, strict=True)):
            node, state = nodes[int(resource['node_label'])], 0.
            for h in range(24):
                clock, value = (8+h+.5) % 24, float(action[h, j])
                if unit == 'kvar':
                    cap = float(resource['capacity_kvar']); hours = resource.get('available_clock_hours')
                    available = (hours is None or hours == list(range(24)) or
                        (hours[0] <= clock < hours[1] if hours[0] < hours[1] else clock >= hours[0] or clock < hours[1]))
                    low, high = (-cap, cap) if available else (0., 0.)
                    factor = float(resource.get('hourly_fee_multiplier', [1.]*24)[h])
                    fee = factor*(resource['fee_positive_yuan_per_kvarh']*max(value, 0.)+
                                  resource['fee_negative_yuan_per_kvarh']*max(-value, 0.))
                    amounts['reactive_abs_kvarh'] += abs(value)
                    amounts['reactive_yuan'] += fee
                    dq[4*h:4*h+4, node] += value
                    # Intentionally no active-power increment for this resource.
                else:
                    cap = float(resource['contract_kw'])
                    if resource['kind'] == 'pv':
                        export = max(0., -float(np.max(forecast[4*h:4*h+4, node]))*base)
                        low, high = 0., min(cap, export) if 10 <= clock < 14 else 0.
                    elif resource['kind'] == 'ev':
                        low, high = (-cap, cap) if clock >= 18 or clock < 7 else (0., 0.)
                        state += value; violation = max(violation, abs(state)-2*cap)
                    else:
                        self.require(resource['kind'] == 'thermal', 'Unexpected P device')
                        low, high = -cap, cap
                        state = .98*state+value; violation = max(violation, abs(state)-cap)
                    factor = 1.25 if 7 <= clock < 10 or 17 <= clock < 22 else .8 if clock < 7 else 1.
                    fee = factor*(resource['fee_positive_yuan_per_kwh']*max(value, 0.)+
                                  resource['fee_negative_yuan_per_kwh']*max(-value, 0.))
                    amounts['active_abs_kwh'] += abs(value); amounts['active_yuan'] += fee
                    dp[4*h:4*h+4, node] += value
                violation = max(violation, low-value, value-high)
            if unit == 'kW' and resource['kind'] != 'pv':
                final_error = max(final_error, abs(state))
        amounts['total_yuan'] = amounts['active_yuan']+amounts['reactive_yuan']
        self.require(violation <= 1e-7 and final_error <= 1e-7,
                     f'Resource bounds/state failed: {violation}, {final_error}')
        self.maxima['device_inequality_violation'] = max(self.maxima['device_inequality_violation'], violation)
        self.maxima['device_terminal_state_error'] = max(self.maxima['device_terminal_state_error'], final_error)
        self.counts['mixed_unit_device_actions_verified'] += 1
        return amounts, dp/base, dq/base

    def quantity_check(self, name, saved, calculated):
        for field, value in calculated.items():
            self.same(name+'_'+field, saved[field], value, atol=1e-8)

    def forecast(self, directory, entry, protocol):
        ctx = self.context(entry); seal = io.read(directory/'seal.json')
        planner = previous.arrays(self.checked(seal['planner']))
        plans = io.read(self.checked(seal['plans']))
        self.checked(seal['score']); self.checked(seal['protocol'])
        prediction = planner.pop('forecast'); draw = int(directory.name.removeprefix('forecast'))
        key = (entry['original_id'], draw)
        P0 = np.c_[ctx['baseline_P'], ctx['baseline_Q']]
        expected = P0*(1+.05*previous.random_stream(protocol, entry, draw, 0, 991).normal(size=P0.shape))
        self.same('forecast_randomness', prediction, expected, atol=0, rtol=0)
        self.paired(key, 'forecast', prediction)
        config = io.read(directory/'resource_case.json')
        P_R = self.config_check(entry, config, ctx['labels']); Q_R = len(config['resources'])-P_R
        strategies = ['no_dfs']+[name for mode in protocol['modes'] for name in (mode, mode+'_rotation')]
        self.require(set(planner) == set(strategies), 'Strategy set differs')
        quantities, forecast_values, exact_values = {}, {}, {}
        for strategy, u in planner.items():
            amounts, dp, dq = self.device_values(config, ctx['labels'], prediction, ctx['base'], u)
            quantities[strategy] = amounts
            forecast_values[strategy] = self.evaluate(ctx, prediction+np.c_[dp, dq])
            path = directory/'exact_requested'/strategy; receipt = io.read(path/'receipt.json')
            if receipt['status'] != 'complete':
                self.counts['exact_AC_failures_preserved'] += 1
                continue
            clean = previous.arrays(path/'clean_observation.npz')
            self.same('exact_requested_P', clean['P'], ctx['baseline_P']+dp, atol=1e-13)
            self.same('exact_requested_Q', clean['Q'], ctx['baseline_Q']+dq, atol=1e-13)
            self.same('exact_requested_Y', clean['Y'], clean['root'][:, None]**2-clean['V']**2, atol=1e-14)
            self.quantity_check('exact_requested', receipt['requested_quantities'], amounts)
            exact_values[strategy] = self.evaluate(ctx, np.c_[clean['P'], clean['Q']], clean['Y'],
                np.c_[clean['P'], clean['Q']], clean['Y'])
            self.pair_file(path/'pair_evidence.npz', ctx, exact_values[strategy])
            self.pair_file(path/'forecast_pair_evidence.npz', ctx, forecast_values[strategy])
        for mode, plan in plans.items():
            self.counts['quotes'] += 1; self.counts['quote_status/'+plan['status']] += 1
            if plan.get('zero_fallback'):
                self.same('zero_fallback', planner[mode], np.zeros_like(planner[mode]), atol=0, rtol=0)
            elif plan.get('cost_yuan') is not None:
                self.same('quote_fee', plan['cost_yuan'], quantities[mode]['total_yuan'], atol=1e-7)
            self.quantity_check('quote_quantities', plan['quantities'], quantities[mode])
            self.same('rotation_fee', plan['rotation_fee_yuan'], quantities[mode+'_rotation']['total_yuan'], atol=1e-8)
            matched = abs(quantities[mode]['total_yuan']-quantities[mode+'_rotation']['total_yuan']) <= 1e-7
            self.require(plan['rotation_same_fee'] == matched, 'Rotation fee-match declaration differs')
            if not matched:
                self.fee_mismatches.append(dict(task=str(directory), mode=mode,
                    planned=quantities[mode]['total_yuan'], rotation=quantities[mode+'_rotation']['total_yuan']))
        results = io.read(directory/'results.json')
        self.require(len(results) == protocol['repeats']*len(strategies), 'Execution denominator differs')
        by_key = {(r['repeat'], r['strategy']):r for r in results}
        self.require(len(by_key) == len(results), 'Duplicate execution row')
        for repeat in range(protocol['repeats']):
            rd = directory/f'execution{repeat:02d}'; random = previous.arrays(rd/'paired_randomness.npz')
            expected_eta = np.r_[previous.random_stream(protocol, entry, draw, repeat, 773).uniform(.9, 1., P_R),
                previous.random_stream(protocol, entry, draw, repeat, 774).uniform(.9, 1., Q_R)]
            self.same('eta_P_stream773', random['eta'][:P_R], expected_eta[:P_R], atol=0, rtol=0)
            self.same('eta_Q_stream774', random['eta'][P_R:], expected_eta[P_R:], atol=0, rtol=0)
            self.paired((key, repeat), 'eta_P', random['eta'][:P_R])
            self.paired((key, repeat), 'eta_Q', random['eta'][P_R:])
            for k, shape in enumerate((ctx['baseline_P'].shape, ctx['baseline_Q'].shape,
                    ctx['baseline_P'].shape, ctx['baseline_root'].shape)):
                self.same('meter_randomness', random[f'noise{k}'], previous.random_stream(
                    protocol, entry, draw, repeat, k+2).normal(size=shape), atol=0, rtol=0)
                self.paired((key, repeat), 'noise'+str(k), random[f'noise{k}'])
            computed, pairs = {}, {}
            for strategy in strategies:
                path = rd/strategy; self.where = str(path); row = by_key[(repeat, strategy)]
                self.counts['execution_rows'] += 1
                self.require(row == io.read(path/'receipt.json'), 'Individual/combined receipt differs')
                if row['status'] != 'complete':
                    self.failed_executions.append(dict(path=str(path), status=row['status'], error=row.get('error')))
                    continue
                try:
                    obs = previous.arrays(path/'observations.npz'); u = planner[strategy]
                    self.same('requested_action', obs['requested_action'], u, atol=0, rtol=0)
                    delivered = u*expected_eta[None]
                    for j, resource in enumerate(config['resources'][:P_R]):
                        if resource['kind'] == 'pv':
                            node = ctx['labels'].index(resource['node_label'])
                            for h in range(24):
                                available = max(0., -float(np.max(ctx['baseline_P'][4*h:4*h+4, node]))*ctx['base'])
                                delivered[h, j] = min(delivered[h, j], resource['contract_kw'], available)
                    self.same('delivered_action', obs['delivered_action'], delivered, atol=1e-12)
                    amounts, dp, dq = self.device_values(config, ctx['labels'], prediction, ctx['base'], delivered)
                    self.same('actual_P_no_Q_leakage', obs['true_P'], ctx['baseline_P']+dp, atol=1e-13)
                    self.same('actual_Q_no_P_leakage', obs['true_Q'], ctx['baseline_Q']+dq, atol=1e-13)
                    self.same('physical_root', obs['true_root'], ctx['baseline_root'], atol=0, rtol=0)
                    pq_sd = protocol['meter']['additional_PQ_relative_sd']; v_sd = protocol['meter']['additional_V_relative_sd']
                    self.same('meter_P', obs['P'], obs['true_P']*(1+pq_sd*random['noise0']), atol=1e-14)
                    self.same('meter_Q', obs['Q'], obs['true_Q']*(1+pq_sd*random['noise1']), atol=1e-14)
                    self.same('meter_V', obs['V'], obs['true_V']*(1+v_sd*random['noise2']), atol=1e-14)
                    self.same('meter_root', obs['root'], obs['true_root']*(1+v_sd*random['noise3']), atol=1e-14)
                    self.same('observed_Y', obs['Y'], obs['root'][:, None]**2-obs['V']**2, atol=1e-14)
                    self.same('planned_fee', row['planned_fee_yuan'], quantities[strategy]['total_yuan'], atol=1e-8)
                    self.same('settled_fee', row['settled_fee_yuan'], amounts['total_yuan'], atol=1e-8)
                    self.quantity_check('requested_quantities', row['requested_quantities'], quantities[strategy])
                    self.quantity_check('delivered_quantities', row['delivered_quantities'], amounts)
                    for suffix, field in [('kwh', 'active_abs_kwh'), ('kvarh', 'reactive_abs_kvarh')]:
                        self.same('delivered_'+suffix, row['delivered_abs_'+suffix], amounts[field], atol=1e-8)
                        self.same('delivery_receipt_'+suffix, row['delivery']['delivered_abs_'+suffix], amounts[field], atol=1e-8)
                    self.same('delivery_eta', row['delivery']['multipliers'], expected_eta, atol=0, rtol=0)
                    if strategy == 'no_dfs':
                        for field in ('P', 'Q', 'Y', 'true_P', 'true_Q', 'true_V'):
                            self.paired((key, repeat), 'zero_'+field, obs[field])
                    y = obs['true_root'][:, None]**2-obs['true_V']**2
                    values = self.evaluate(ctx, np.c_[obs['true_P'], obs['true_Q']], y, np.c_[obs['P'], obs['Q']], obs['Y'])
                    computed[strategy] = values
                    self.native_selection(ctx, obs, row, values)
                    pairs[strategy] = self.pair_file(path/'pair_evidence.npz', ctx, values)
                    self.counts['successful_execution_rows_verified'] += 1
                except Exception as exc:
                    self.errors.append(dict(path=self.where, error=repr(exc), traceback=traceback.format_exc()))
            if 'no_dfs' in computed:
                for strategy, values in computed.items():
                    self.where = str(rd/strategy)
                    try:
                        row = by_key[(repeat, strategy)]; saved = previous.arrays(rd/strategy/'pair_changes.npz')
                        for field in saved:
                            if field != 'maximum_identity_error':
                                self.same('change_'+field, saved[field],
                                    [a[field]-b[field] for a, b in zip(pairs[strategy], pairs['no_dfs'], strict=True)], atol=1e-8)
                        records = row['evidence']['target_pairs']
                        self.require(len(records) == len(ctx['H'])-1, 'Directed pair count differs')
                        for record in records:
                            t, j = record['target'], record['competitor']
                            self.require(t == ctx['target'] and j != t, 'Historical target changed')
                            before = self.directed(ctx, computed['no_dfs'], t, j); after = self.directed(ctx, values, t, j)
                            p0 = self.directed(ctx, forecast_values['no_dfs'], t, j)
                            p1 = self.directed(ctx, forecast_values[strategy], t, j)
                            expected = dict(actual_physical_J_gain=after['J']-before['J'],
                                actual_bias_change=after['bias']-before['bias'], actual_meter_change=after['meter']-before['meter'],
                                actual_logodds_change=after['observed_logodds']-before['observed_logodds'],
                                same_observations_logodds_change_without_common=after['observed_logodds_without_common']-before['observed_logodds_without_common'],
                                forecast_J_gain=p1['J']-p0['J'], forecast_J_gain_without_common=p1['J_without_common']-p0['J_without_common'],
                                forecast_common_square_gain=p1['J_common']-p0['J_common'], forecast_cross_gain=p1['J_cross']-p0['J_cross'])
                            if strategy in exact_values and 'no_dfs' in exact_values:
                                e1 = self.directed(ctx, exact_values[strategy], t, j)
                                e0 = self.directed(ctx, exact_values['no_dfs'], t, j)
                                expected['exact_physical_J_gain'] = e1['J']-e0['J']
                            for field, value in expected.items():
                                self.same('directed_'+field, record[field], value, atol=1e-8)
                            self.same('directed_gain_identity', expected['actual_logodds_change'],
                                expected['actual_physical_J_gain']+expected['actual_bias_change']+expected['actual_meter_change'], atol=1e-8)
                            self.counts['directed_target_pairs_verified'] += 1
                    except Exception as exc:
                        self.errors.append(dict(path=self.where, error=repr(exc), traceback=traceback.format_exc()))
            self.counts['paired_execution_paths'] += 1
        self.counts['forecast_tasks_verified'] += 1


def run(run_path, destination):
    run_path, destination = Path(run_path).resolve(), Path(destination).resolve()
    if destination.exists():
        raise FileExistsError(destination)
    if not (run_path/'complete.json').exists():
        raise RuntimeError('Verification requires completed saved results')
    verify = Verify(); protocol = io.read(run_path/'protocol.json')
    for ref in protocol['sources']:
        verify.checked(ref)
    verify.checked(protocol['placement']); verify.checked(protocol['source_protocol'])
    entries = io.read(verify.checked(protocol['inputs'])); expected = len(entries)*protocol['forecasts']
    completion = io.read(run_path/'complete_fraction0.10.json')
    verify.failed_tasks.extend(completion['failed_tasks'])
    tasks = sorted((run_path/'fraction0.10').glob('*/forecast*/complete.json'))
    verify.require(len(tasks)+len(verify.failed_tasks) == expected, 'Forecast denominator differs')
    by_id = {entry['id']:entry for entry in entries}
    for path in tasks:
        verify.where = str(path.parent)
        try:
            verify.checked(io.read(path)['plans']); verify.checked(io.read(path)['results'])
            verify.forecast(path.parent, by_id[path.parent.parent.name], protocol)
        except Exception as exc:
            verify.errors.append(dict(path=verify.where, error=repr(exc), traceback=traceback.format_exc()))
    report = dict(status='passed' if not verify.errors else 'failed',
        verified_utc=datetime.now(timezone.utc).isoformat(), run=io.ref(run_path/'protocol.json'),
        sources=[io.ref(__file__), io.ref(previous.__file__)],
        expected_forecast_problems=expected, completed_forecast_problems=len(tasks),
        original_histories=len({e['original_id'] for e in entries}), placement_history_combinations=len(entries),
        expected_paired_paths=expected*protocol['repeats'], counts=dict(verify.counts),
        maximum_absolute_errors=dict(verify.maxima), errors=verify.errors,
        original_failed_tasks=verify.failed_tasks, original_failed_executions=verify.failed_executions,
        unmatched_planned_rotation_fees=verify.fee_mismatches,
        scope='Independent stored-observation replay; no new fitting, optimization, domain audit, runner scoring, or AC solve',
        limitations='AC equations and optimizer bounds are not re-solved; all failures and fee mismatches retained')
    destination.mkdir(parents=True)
    shutil.copyfile(__file__, destination/'verify.py')
    shutil.copyfile(previous.__file__, destination/'verify_run_dependency.py')
    io.write(destination/'report.json', report)
    io.write(destination/'complete.json', dict(report=io.ref(destination/'report.json')))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args(); report = run(args.run, args.out)
    print(dict(status=report['status'], counts=report['counts'], errors=len(report['errors'])))
    if report['errors']:
        raise SystemExit(1)
