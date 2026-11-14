"""Independently replay sealed capacity-procurement records, without solving.

The imported auditors use scalar/NumPy arithmetic, not the production domain,
identifier, selector, optimization engine or AC solver. Truth is reconstructed
from the physical branch table exclusively for this post-execution evaluation.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
import traceback

import numpy as np

from ..measured_pq_flex_20261114 import verify_run as native
from ..flexibility_resource_validation_20261114 import operational_audit as operation
from ..flexibility_resource_validation_20261114.analyze import stats
from ..flexibility_extended_solve_20261114.analyze import direct_original


POLICIES = ('no_dfs', 'anchor', 'activation', 'stable_activation', 'capacity', 'capacity_rotation')
OBJECTIVES = {'activation': 'activation_compensation', 'capacity': 'activation_plus_daily_reserved_capacity'}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    def convert(item):
        if isinstance(item, np.ndarray):
            return item.tolist()
        if isinstance(item, np.generic):
            return item.item()
        raise TypeError(type(item).__name__)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
        allow_nan=False, default=convert)+'\n', encoding='utf-8')


def ref(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def procurement(config, request, delivered, kappa_p, kappa_q):
    """One service-day reservation from requests; delivered activation only.

    No device ordering assumption enters the price: each declared unit selects
    its own capacity price. The negative and positive ratings are not added.
    """
    request, delivered = np.asarray(request, float), np.asarray(delivered, float)
    if request.shape != delivered.shape or request.shape != (24, len(config['resources'])):
        raise ValueError('Expected matching 24-hour resource actions')
    if not np.isfinite(request).all() or not np.isfinite(delivered).all():
        raise ValueError('Nonfinite action')
    if min(kappa_p, kappa_q) < 0:
        raise ValueError('Negative reservation price')
    clock = (8+np.arange(24)+.5) % 24
    p_factor = np.where(clock < 7, .8, np.where(((clock >= 7) & (clock < 10)) |
        ((clock >= 17) & (clock < 22)), 1.25, 1.))
    peak = np.max(np.abs(request), axis=0)
    planned = dict(active_activation_yuan=0., reactive_activation_yuan=0., active_availability_yuan=0., reactive_availability_yuan=0.)
    settled = planned.copy()
    for col, (resource, unit) in enumerate(zip(config['resources'], config['action_units'], strict=True)):
        if unit not in ('kW', 'kvar'):
            raise ValueError('Unknown action unit '+str(unit))
        prefix, suffix, price = ('active', 'kwh', kappa_p) if unit == 'kW' else ('reactive', 'kvarh', kappa_q)
        factor = p_factor if unit == 'kW' else np.asarray(resource.get('hourly_fee_multiplier', np.ones(24)), float)
        for action, result in ((request, planned), (delivered, settled)):
            value = action[:, col]
            result[prefix+'_activation_yuan'] += float(np.sum(factor*(
                resource['fee_positive_yuan_per_'+suffix]*np.maximum(value, 0.)+
                resource['fee_negative_yuan_per_'+suffix]*np.maximum(-value, 0.))))
            result[prefix+'_availability_yuan'] += float(price*peak[col])
    for result in (planned, settled):
        result['activation_yuan'] = result['active_activation_yuan']+result['reactive_activation_yuan']
        result['availability_yuan'] = result['active_availability_yuan']+result['reactive_availability_yuan']
        result['total_yuan'] = result['activation_yuan']+result['availability_yuan']
        result['reserved_P_kw'] = float(sum(v for v, unit in zip(peak, config['action_units'], strict=True) if unit == 'kW'))
        result['reserved_Q_kvar'] = float(sum(v for v, unit in zip(peak, config['action_units'], strict=True) if unit == 'kvar'))
    return dict(planned=planned, settled=settled, s_plan=peak)


def stable_choice(anchor_feasible, activation_feasible, anchor_fee, activation_fee, threshold):
    """The fixed decision, based exclusively on two planned activation fees."""
    if anchor_feasible:
        return 'activation' if activation_feasible and anchor_fee-activation_fee >= threshold else 'anchor'
    return 'activation' if activation_feasible else 'no_dfs'


class Audit(native.Verify):
    def __init__(self):
        super().__init__()
        self.operation = operation.Audit()
        self.shared = {}
        self.records, self.quote_records, self.task_errors, self.exact_failures = [], [], [], []

    def file(self, path, expected=None):
        item = ref(path)
        if expected is not None:
            self.require(item['sha256'] == expected, 'Changed source: '+str(path))
        self.hashes[item['path']] = item['sha256']
        return Path(path)

    def paired(self, key, name, value):
        key = (key, name)
        if key in self.shared:
            self.same('paired_'+name, value, self.shared[key], atol=0., rtol=0.)
            self.counts['cross_resource_arrays'] += 1
        else:
            self.shared[key] = np.asarray(value).copy()

    def costs(self, saved, computed, prefix):
        # The canonical receipt explicitly itemizes all four charges.
        for key, value in computed.items():
            self.require(key in saved, prefix+' missing '+key)
            self.same(prefix+'_'+key, saved[key], value, atol=1e-8, rtol=1e-10)

    def catalogue(self, entry, config, ctx):
        resources, units = config['resources'], config['action_units']
        p_count = sum(unit == 'kW' for unit in units)
        self.require(units == ['kW']*p_count+['kvar']*(len(units)-p_count), 'Invalid mixed-unit ordering')
        self.require(len(set(r['id'] for r in resources)) == len(resources), 'Duplicate resource ID')
        self.require(len(entry['p_resources']) == p_count, 'P contract count changed')
        self.require(len(entry['q_resources']) == len(resources)-p_count, 'Q contract count changed')
        for actual, expected in zip(resources, entry['p_resources']+entry['q_resources'], strict=True):
            self.require(actual['id'] == expected['id'] and actual['node_label'] == expected['node_label'], 'Resource identity changed')
            self.require(int(actual['node_label']) in ctx['labels'], 'Unmeasured resource location')
            for key in ('contract_kw', 'capacity_kvar', 'fee_positive_yuan_per_kwh', 'fee_negative_yuan_per_kwh',
                        'fee_positive_yuan_per_kvarh', 'fee_negative_yuan_per_kvarh', 'q_over_p'):
                if key in expected:
                    self.same('catalogue_'+key, actual[key], expected[key], atol=0., rtol=0.)
        self.require(p_count in (7, 14), 'Unexpected P contract count')
        self.require(len(resources)-p_count == 15, 'Expected 15 Q contracts')
        self.same('Q_offered_capacity', sum(r['capacity_kvar'] for r in resources[p_count:]), 36., atol=1e-12)
        self.require(config.get('original_P_count') == 7, 'Original P prefix identity missing')
        return p_count

    def score(self, ctx, obs, row):
        clean_y = obs['true_root'][:, None]**2-obs['true_V']**2
        clean_power, power = np.c_[obs['true_P'], obs['true_Q']], np.c_[obs['P'], obs['Q']]
        values = self.evaluate(ctx, clean_power, clean_y, power, obs['Y'])
        self.native_selection(ctx, obs, row, values)
        proper = lambda groups: {group for group in groups if 1 < len(group) < ctx['n']}
        correct = [i for i, active in enumerate(ctx['active']) if proper(active) == proper(ctx['truth'])]
        candidates = [i for i in range(len(ctx['H'])) if i not in correct]
        target = min(correct, key=lambda i: (ctx['history'][i], i)) if correct else None
        other = ctx['target'] if ctx['target'] in candidates else min(candidates, key=lambda i: (ctx['history'][i], i)) if candidates else None
        representative_ids = [model['index'] for model in ctx['score']['models']]
        total = ctx['history']+values['observed_sse']
        ll = -(total[representative_ids]-min(total[representative_ids]))/(2*ctx['sigma2'])
        probabilities = np.exp(ll)/np.exp(ll).sum()
        correct_reps = [k for k, i in enumerate(representative_ids) if i in correct]
        p_true = float(probabilities[correct_reps].sum())
        best = int(np.argmax(probabilities))
        result = dict(native_correct=bool(row['native_map_correct']), native_full_support_exact=bool(row['native_full_support_exact']),
            p_true=p_true, true_high=p_true >= .9, wrong_high=bool(representative_ids[best] not in correct and probabilities[best] >= .9),
            native_selected_index=int(row['native_selected_index']), native_missing=int(row['native_missing']), native_extra=int(row['native_extra']))
        if row.get('p_true') is not None:
            self.same('true_probability', p_true, row['p_true'], atol=1e-10)
        if target is not None and other is not None:
            result['pair'] = [target, other]
            result['evidence'] = self.directed(ctx, values, target, other)
        return result, values

    def quote(self, folder, entry, ctx, config, planner, mode, plan, kappa_p, kappa_q, initial_points):
        u = planner[mode]
        objective = OBJECTIVES['capacity' if mode == 'capacity' else 'activation']
        billed = procurement(config, u, u, kappa_p, kappa_q)['planned']
        cost = billed['total_yuan'] if mode == 'capacity' else billed['activation_yuan']
        direct = direct_original(dict(entry=entry, score=ctx['score'], config=config,
            planner=planner, directory=folder), u, plan)
        feasible = bool(plan.get('feasible')) and not plan.get('zero_fallback', False)
        if feasible:
            self.require(direct['passed_original_acceptance'], mode+' failed original evidence/guard')
            self.same('quote_objective_'+mode, plan['cost_yuan'], cost, atol=1e-7)
        else:
            self.same('quote_failure_zero_'+mode, u, np.zeros_like(u), atol=0., rtol=0.)
        if mode in ('activation', 'capacity', 'anchor') and plan.get('status') != 'exception':
            self.require(plan.get('objective_cost_object') == objective, mode+' objective identity differs')
            self.require(plan.get('bound_cost_object') == objective, mode+' bound object differs')
        lower = plan.get('lower_bound_yuan')
        if lower is not None:
            self.require(np.isfinite(lower) and lower >= -1e-7, 'Invalid cost bound')
            if feasible:
                self.require(lower <= cost+1e-6, 'Lower bound exceeds feasible objective')
        initial = plan.get('initial') or {}
        if initial.get('feasible'):
            name = initial.get('name')
            self.require(name in initial_points, 'Initial point absent from sealed shared pool: '+str(name))
            ib = procurement(config, initial_points[name], initial_points[name], kappa_p, kappa_q)['planned']
            icost = ib['total_yuan'] if mode == 'capacity' else ib['activation_yuan']
            self.same('initial_objective_'+mode, initial['cost_yuan'], icost, atol=1e-7)
            initial_direct = direct_original(dict(entry=entry, score=ctx['score'], config=config,
                planner=planner, directory=folder), initial_points[name], plan)
            self.require(initial_direct['passed_original_acceptance'], 'Claimed feasible initial violates original evidence/guard')
            if feasible:
                self.require(cost <= icost+1e-6, 'Returned plan worse than retained feasible initial point')
        self.quote_records.append(dict(id=entry['id'], resource_variant=entry['resource_variant'], forecast=int(folder.name[8:]),
            policy=mode, status=plan.get('status'), feasible=feasible, zero_fallback=bool(plan.get('zero_fallback')),
            objective_cost_object=objective, objective_cost_yuan=cost, lower_bound_yuan=lower,
            raw_native_attributes=plan.get('native_attributes'), relative_gap=plan.get('relative_gap'),
            lower_bound_withdrawn=plan.get('lower_bound_withdrawn'), native_status=plan.get('native_status'),
            global_optimality_verified=plan.get('global_optimality_verified'),
            initial=initial, planned_cost=billed, source=plan.get('source'), seconds=plan.get('seconds'),
            charged_total_seconds=plan.get('charged_total_seconds'), shared_seed_seconds=plan.get('shared_seed_seconds'),
            shared_preparation_seconds=plan.get('shared_preparation_seconds'), quote_wall_seconds=plan.get('quote_wall_seconds'),
            requested_quote_budget_seconds=plan.get('requested_quote_budget_seconds'), quote_budget_exceeded=plan.get('quote_budget_exceeded'),
            direct_original=direct))
        return feasible

    def forecast(self, source, folder, entry, protocol, metadata, branches, bus_rows):
        self.where = str(folder)
        ctx = self.context(entry)
        draw = int(folder.name.removeprefix('forecast'))
        key = (entry.get('original_id', entry['id']), draw)
        task = read(self.file(folder/'complete.json'))
        self.require(task['id'] == entry['id'] and int(task['forecast_index']) == draw, 'Task identity changed')
        strategies = tuple(protocol['policies'])
        self.require(strategies == POLICIES, 'Unexpected policy set/order')
        expected_paths = {(repeat, strategy) for repeat in range(protocol['repeats']) for strategy in strategies}
        self.require(task['rows'] == len(expected_paths), 'Task row denominator differs')
        seal = read(self.file(folder/'seal.json'))
        for name in ('planner', 'plans', 'score', 'protocol'):
            self.checked(seal[name])
        planner = native.arrays(self.file(folder/'planner.npz'))
        self.require(set(planner) == {'forecast', *strategies}, 'Missing/extra sealed plan')
        config = read(self.file(folder/'resource_case.json'))
        p_count = self.catalogue(entry, config, ctx)
        q_count = len(config['resources'])-p_count
        kappa = protocol['reservation_prices']
        kp, kq = float(kappa['P']), float(kappa['Q'])
        prediction = planner['forecast']
        P0 = np.c_[ctx['baseline_P'], ctx['baseline_Q']]
        expected = P0*(1+.05*native.random_stream(protocol, entry, draw, 0, 991).normal(size=P0.shape))
        self.same('forecast_generated_from_new_PQ_only', prediction, expected, atol=0., rtol=0.)
        self.paired(key, 'forecast', prediction)
        reservation = native.arrays(self.file(folder/'reservation.npz'))
        self.require(set(reservation) == set(strategies), 'Reservation policy set differs')
        plans = read(self.file(folder/'plans.json'))
        self.require(set(plans) == {'activation', 'capacity', 'stable_activation'}, 'Optimization plan identity differs')
        anchor = read(self.file(folder/'anchor_plan.json'))
        points = native.arrays(self.file(folder/'shared_initial_points.npz'))
        generated = native.arrays(self.file(folder/'generated_initial_points.npz'))
        self.require(set(points) == set(generated) | ({'shared_150_second_anchor'} if anchor.get('feasible') else set()),
            'Shared seed pool does not equal generated pool plus available anchor')
        for name, value in generated.items():
            self.same('common_generated_seed', value, points[name], atol=0., rtol=0.)
        if anchor.get('feasible'):
            self.same('shared_anchor', points['shared_150_second_anchor'], planner['anchor'], atol=0., rtol=0.)
        planned_cost, requests, exact, direct = {}, {}, {}, {}
        fcontext = dict(entry=entry, score=ctx['score'], config=config, planner=planner, directory=folder)
        for policy in strategies:
            u = planner[policy]
            devices, violations, dp, dq = operation.resource_values(config, u, ctx['labels'], prediction[:, :ctx['n']], ctx['base'])
            requests[policy] = devices, violations
            billed = procurement(config, u, u, kp, kq)
            planned_cost[policy] = billed['planned']
            self.same('requested_peak_'+policy, reservation[policy], billed['s_plan'], atol=0., rtol=0.)
            forecast_values = self.evaluate(ctx, prediction+np.c_[dp, dq])
            exact_dir = folder/'exact_requested'/policy
            exact_receipt = read(self.file(exact_dir/'receipt.json'))
            if exact_receipt['status'] != 'complete':
                self.exact_failures.append(dict(path=str(exact_dir), status=exact_receipt['status'], error=exact_receipt.get('error')))
                continue
            clean = native.arrays(self.file(exact_dir/'clean_observation.npz'))
            self.same('exact_requested_P', clean['P'], ctx['baseline_P']+dp, atol=1e-13)
            self.same('exact_requested_Q', clean['Q'], ctx['baseline_Q']+dq, atol=1e-13)
            self.same('exact_requested_Y', clean['Y'], clean['root'][:, None]**2-clean['V']**2, atol=1e-14)
            values = self.evaluate(ctx, np.c_[clean['P'], clean['Q']], clean['Y'], np.c_[clean['P'], clean['Q']], clean['Y'])
            self.pair_file(self.file(exact_dir/'pair_evidence.npz'), ctx, values)
            self.pair_file(self.file(exact_dir/'forecast_pair_evidence.npz'), ctx, forecast_values)
            exact[policy] = values
            direct[policy] = direct_original(fcontext, u, plans.get(policy, anchor))
        feasible = {}
        for policy, plan in [('anchor', anchor), ('activation', plans['activation']), ('capacity', plans['capacity'])]:
            feasible[policy] = self.quote(folder, entry, ctx, config, planner, policy, plan, kp, kq,
                generated if policy == 'anchor' else points)
            if policy != 'anchor':
                self.checked(plan['initial_points'])
                self.same('timing_shared_plus_call', plan['charged_total_seconds'],
                    plan['shared_preparation_seconds']+plan['quote_wall_seconds'], atol=1e-12)
        selected = stable_choice(feasible['anchor'], feasible['activation'], planned_cost['anchor']['activation_yuan'],
            planned_cost['activation']['activation_yuan'], float(protocol['stable_fee_threshold_yuan']))
        self.same('stable_policy_action', planner['stable_activation'], planner[selected], atol=0., rtol=0.)
        self.require(bool(plans['stable_activation'].get('retained_anchor')) == (selected == 'anchor'), 'Stable retention flag differs')
        stable_direct = direct_original(fcontext, planner['stable_activation'], plans['stable_activation'])
        if feasible.get(selected, False):
            self.require(stable_direct['passed_original_acceptance'], 'Stable returned plan failed original constraints')
        if plans['stable_activation'].get('feasible'):
            self.same('stable_quote_activation_objective', plans['stable_activation']['cost_yuan'],
                planned_cost['stable_activation']['activation_yuan'], atol=1e-8)
        same_fee = abs(planned_cost['capacity_rotation']['total_yuan']-planned_cost['capacity']['total_yuan']) <= 1e-7
        rotation = read(self.file(folder/'rotation_receipt.json'))
        self.require(bool(rotation['same_planned_fee']) == same_fee, 'Rotation match flag differs')
        self.same('rotation_total_cost', rotation['rotation_total_yuan'], planned_cost['capacity_rotation']['total_yuan'], atol=1e-8)
        if not same_fee:
            self.fee_mismatches.append(dict(task=str(folder), capacity=planned_cost['capacity']['total_yuan'],
                rotation=planned_cost['capacity_rotation']['total_yuan']))
        combined = read(self.checked(task['results']))
        by_key = {(int(row['repeat']), row['strategy']): row for row in combined}
        self.require(len(by_key) == len(combined) and set(by_key) == expected_paths, 'Missing/duplicate execution records')
        self.require(task['failed_rows'] == sum(row['status'] != 'complete' for row in combined), 'Failed-row receipt count differs')
        for repeat in range(protocol['repeats']):
            rd = folder/f'execution{repeat:02d}'
            random = native.arrays(self.file(rd/'paired_randomness.npz'))
            eta = np.r_[native.random_stream(protocol, entry, draw, repeat, 773).uniform(.9, 1., p_count),
                native.random_stream(protocol, entry, draw, repeat, 774).uniform(.9, 1., q_count)]
            self.same('response_random_stream', random['eta'], eta, atol=0., rtol=0.)
            self.paired((key, repeat), 'original_seven_P_response', eta[:7])
            self.paired((key, repeat), 'all_fifteen_Q_response', eta[p_count:])
            for k, shape in enumerate((ctx['baseline_P'].shape, ctx['baseline_Q'].shape,
                    ctx['baseline_P'].shape, ctx['baseline_root'].shape)):
                noise = native.random_stream(protocol, entry, draw, repeat, k+2).normal(size=shape)
                self.same('meter_random_stream'+str(k), random['noise'+str(k)], noise, atol=0., rtol=0.)
                self.paired((key, repeat), 'noise'+str(k), noise)
            observations, evaluated = {}, {}
            for policy in strategies:
                path = rd/policy; row = by_key[(repeat, policy)]; self.where = str(path)
                record = dict(id=entry['id'], resource_variant=entry['resource_variant'], forecast=draw, repeat=repeat,
                    strategy=policy, source=str(path), original_status=row['status'])
                try:
                    self.require(row == read(self.file(path/'receipt.json')), 'Individual/combined receipt differs')
                    self.require(row['id'] == entry['id'] and row['forecast_index'] == draw, 'Row identity changed')
                    if row['status'] != 'complete':
                        self.failed_executions.append(dict(path=str(path), status=row['status'], error=row.get('error')))
                        raise ValueError('Original execution failed: '+row['status'])
                    obs = native.arrays(self.file(path/'observations.npz')); observations[policy] = obs
                    self.same('labels', obs['labels'], ctx['labels'], atol=0., rtol=0.)
                    request, delivered = obs['requested_action'], obs['delivered_action']
                    self.same('sealed_action', request, planner[policy], atol=0., rtol=0.)
                    self.same('receipt_response', row['delivery']['multipliers'], eta, atol=0., rtol=0.)
                    expected = request*eta[None]
                    for j, resource in enumerate(config['resources'][:p_count]):
                        if resource['kind'] == 'pv':
                            node = ctx['labels'].index(int(resource['node_label']))
                            cap = np.minimum(resource['contract_kw'], np.maximum(0., -ctx['baseline_P'][:, node]*ctx['base']).reshape(24, 4).min(1))
                            expected[:, j] = np.minimum(expected[:, j], cap)
                    self.same('response_PV_clip', delivered, expected, atol=1e-12, rtol=0.)
                    requested, rv = requests[policy]
                    actual, av, dp, dq = operation.resource_values(config, delivered, ctx['labels'], prediction[:, :ctx['n']], ctx['base'])
                    self.same('delivered_true_P', obs['true_P'], ctx['baseline_P']+dp, atol=1e-13)
                    self.same('delivered_true_Q', obs['true_Q'], ctx['baseline_Q']+dq, atol=1e-13)
                    self.same('measured_P', obs['P'], obs['true_P']*(1+.005*random['noise0']), atol=1e-13)
                    self.same('measured_Q', obs['Q'], obs['true_Q']*(1+.005*random['noise1']), atol=1e-13)
                    self.same('measured_V', obs['V'], obs['true_V']*(1+.0005*random['noise2']), atol=1e-13)
                    self.same('measured_root', obs['root'], obs['true_root']*(1+.0005*random['noise3']), atol=1e-13)
                    self.same('measured_Y', obs['Y'], obs['root'][:, None]**2-obs['V']**2, atol=1e-13)
                    billed = procurement(config, request, delivered, kp, kq)
                    self.costs(row['planned_cost'], billed['planned'], 'planned_cost')
                    self.costs(row['settled_cost'], billed['settled'], 'settled_cost')
                    self.same('planned_total_fee', row['planned_fee_yuan'], billed['planned']['total_yuan'], atol=1e-8)
                    self.same('settled_total_fee', row['settled_fee_yuan'], billed['settled']['total_yuan'], atol=1e-8)
                    self.same('paid_requested_peak', row['planned_cost']['resource_reserved_capacity'], billed['s_plan'], atol=1e-12)
                    self.same('paid_requested_peak_settled', row['settled_cost']['resource_reserved_capacity'], billed['s_plan'], atol=1e-12)
                    for unit, field in [('kW', 'active_abs_kwh'), ('kvar', 'reactive_abs_kvarh')]:
                        for name, devices in [('requested', requested), ('delivered', actual)]:
                            self.same(name+'_'+field, row[name+'_quantities'][field],
                                sum(d['absolute_energy'] for d in devices if d['unit'] == unit), atol=1e-8)
                    evaluation, values = self.score(ctx, obs, row)
                    evaluated[policy] = evaluation
                    self.pair_file(self.file(path/'pair_evidence.npz'), ctx, values)
                    if 'pair' in evaluation and policy in exact:
                        i, j = evaluation['pair']
                        evaluation['ideal_evidence'] = self.directed(ctx, exact[policy], i, j)
                    if policy == 'no_dfs':
                        for name in ('true_P', 'true_Q', 'true_V', 'true_root', 'P', 'Q', 'V', 'root', 'Y'):
                            self.paired((key, repeat), 'no_dfs_'+name, obs[name])
                    physical = row['physical_state']
                    ac = native.arrays(self.checked(physical))
                    if 'time_slice' in physical:
                        a, b = physical['time_slice']; count = len(ac['voltage_complex_pu'])
                        ac = {k: v[a:b] if v.ndim and v.shape[0] == count else v for k, v in ac.items()}
                    electrical, branch_values, buses = operation.electrical(self.operation, ac, obs, metadata, branches, bus_rows)
                    record.update(audit_status='passed', requested=requested, delivered=actual, requested_violations=rv,
                        delivered_violations=av, electrical=electrical, branches=branch_values, buses=buses,
                        planned_cost=billed['planned'], settled_cost=billed['settled'], evaluation=evaluation)
                except Exception as exc:
                    record.update(audit_status='failed', error=repr(exc), traceback=traceback.format_exc())
                self.records.append(record)
            if selected in observations and 'stable_activation' in observations:
                for name, value in observations[selected].items():
                    self.same('stable_selected_observations_'+name, observations['stable_activation'][name], value, atol=0., rtol=0.)
            zero = evaluated.get('no_dfs')
            if zero and 'evidence' in zero:
                for record in self.records[-len(strategies):]:
                    ev = record.get('evaluation', {})
                    if 'evidence' in ev:
                        ev['evidence_increment'] = {name: ev['evidence'][name]-zero['evidence'][name]
                            for name in ('J', 'bias', 'meter', 'clean_logodds', 'observed_logodds')}
                        self.same('increment_decomposition', ev['evidence_increment']['observed_logodds'],
                            sum(ev['evidence_increment'][k] for k in ('J', 'bias', 'meter')), atol=1e-8)
        self.counts['forecasts_completed'] += 1


def self_test():
    """Small arithmetic examples catch units, double-counting and settlement."""
    config = dict(action_units=['kW', 'kvar'], resources=[dict(id='p', fee_positive_yuan_per_kwh=.2,
        fee_negative_yuan_per_kwh=.3), dict(id='q', fee_positive_yuan_per_kvarh=.05, fee_negative_yuan_per_kvarh=.1)])
    u = np.zeros((24, 2)); u[4] = [2, 3]; u[5] = [-1, -2]
    result = procurement(config, u, .9*u, .5, .5)
    np.testing.assert_allclose(result['s_plan'], [2, 3], atol=0, rtol=0)
    np.testing.assert_allclose(result['planned']['activation_yuan'], 1.05)
    np.testing.assert_allclose(result['planned']['availability_yuan'], 2.5)
    np.testing.assert_allclose(result['settled']['total_yuan'], .9*1.05+2.5)
    reordered = dict(resources=config['resources'][::-1], action_units=config['action_units'][::-1])
    np.testing.assert_allclose(procurement(reordered, u[:, ::-1], .9*u[:, ::-1], .5, .5)['settled']['total_yuan'], result['settled']['total_yuan'])
    assert stable_choice(True, True, 2., 1.995, .01) == 'anchor'
    assert stable_choice(True, True, 2., 1.98, .01) == 'activation'
    assert stable_choice(True, False, 2., None, .01) == 'anchor'
    assert stable_choice(False, True, None, 3., .01) == 'activation'
    assert stable_choice(False, False, None, None, .01) == 'no_dfs'
    return dict(passed=True, scope='Fixed numeric examples only; no optimization, AC or random execution')


def summarize(records):
    good = [row for row in records if row['audit_status'] == 'passed']
    result = operation.summarize(records)
    result.update(native_correct=sum(row['evaluation']['native_correct'] for row in good),
        native_full_support_exact=sum(row['evaluation']['native_full_support_exact'] for row in good),
        true_high=sum(row['evaluation']['true_high'] for row in good),
        wrong_high=sum(row['evaluation']['wrong_high'] for row in good),
        mean_true_weight=stats([row['evaluation']['p_true'] for row in good]),
        nonzero_P_paths=sum(any(d['unit'] == 'kW' and d['materially_active'] for d in row['delivered']) for row in good),
        nonzero_Q_paths=sum(any(d['unit'] == 'kvar' and d['materially_active'] for d in row['delivered']) for row in good))
    for scope in ('planned', 'settled'):
        result[scope+'_cost'] = {key: stats([row[scope+'_cost'][key] for row in good])
            for key in ('active_activation_yuan', 'reactive_activation_yuan', 'activation_yuan',
                'active_availability_yuan', 'reactive_availability_yuan', 'availability_yuan', 'total_yuan',
                'reserved_P_kw', 'reserved_Q_kvar')}
    result['evidence_increment'] = {key: stats([row['evaluation']['evidence_increment'][key] for row in good
        if 'evidence_increment' in row['evaluation']]) for key in ('J', 'bias', 'meter', 'clean_logodds', 'observed_logodds')}
    result['negative_observed_evidence_increment'] = sum(row['evaluation'].get('evidence_increment', {}).get('observed_logodds', 0.) < -1e-8 for row in good)
    return result


def paired_outcomes(records):
    grouped = defaultdict(dict)
    for row in records:
        grouped[(row['resource_variant'], row['strategy'])][(row['forecast'], row['repeat'])] = row
    comparisons = {}
    for (variant, policy), a in grouped.items():
        for reference in ('no_dfs', 'anchor', 'activation', 'capacity_rotation'):
            if policy == reference or (variant, reference) not in grouped:
                continue
            b = grouped[(variant, reference)]
            gains, losses, weights, new_wrong, removed_wrong = [], [], [], [], []
            voltage_added = voltage_removed = 0
            complete = 0
            for key in sorted(set(a) | set(b)):
                x, y = a.get(key), b.get(key)
                if x is None or y is None or x['audit_status'] != 'passed' or y['audit_status'] != 'passed':
                    continue
                complete += 1; ex, ey = x['evaluation'], y['evaluation']
                if ex['native_correct'] and not ey['native_correct']:
                    gains.append(list(key))
                if ey['native_correct'] and not ex['native_correct']:
                    losses.append(list(key))
                if ex['wrong_high'] and not ey['wrong_high']:
                    new_wrong.append(list(key))
                if ey['wrong_high'] and not ex['wrong_high']:
                    removed_wrong.append(list(key))
                weights.append(ex['p_true']-ey['p_true'])
                xv = x['electrical']['cells_below_095']+x['electrical']['cells_above_105'] > 0
                yv = y['electrical']['cells_below_095']+y['electrical']['cells_above_105'] > 0
                voltage_added += xv and not yv; voltage_removed += yv and not xv
            comparisons[variant+'/'+policy+'_vs_'+reference] = dict(planned_pairs=len(set(a) | set(b)), complete_pairs=complete,
                added_recoveries=len(gains), lost_recoveries=len(losses), new_wrong_high=len(new_wrong), removed_wrong_high=len(removed_wrong),
                added_recovery_paths=gains, lost_recovery_paths=losses, new_wrong_high_paths=new_wrong,
                removed_wrong_high_paths=removed_wrong, true_weight_difference=stats(weights),
                newly_voltage_outside_095_105_paths=int(voltage_added), removed_voltage_outside_paths=int(voltage_removed))
    return comparisons


def planning_boundary(path):
    """Inspect sealed adapter keys/calls, without importing its code.

    This checks the documented adapter boundary, not an assertion of general
    program noninterference. Data provenance and physical-only truth use are
    separately exposed by this audit's source and input hashes.
    """
    tree = ast.parse(Path(path).read_text(encoding='utf-8-sig'))
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'safe_plan')
    accesses = sorted({node.slice.value for node in ast.walk(function) if isinstance(node, ast.Subscript)
        and isinstance(node.value, ast.Name) and node.value.id == 'ctx' and isinstance(node.slice, ast.Constant)})
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'safe_plan']
    context_arguments = [node.args[1].id if len(node.args) > 1 and isinstance(node.args[1], ast.Name) else '<expression>' for node in calls]
    narrow = [node.value for node in ast.walk(tree) if isinstance(node, ast.Assign) and
        any(isinstance(target, ast.Name) and target.id == 'planning' for target in node.targets)]
    keys = []
    for value in narrow:
        if isinstance(value, ast.Dict):
            keys.append([key.value if isinstance(key, ast.Constant) else '<expression>' for key in value.keys])
        elif isinstance(value, ast.Call) and isinstance(value.func, ast.Name) and value.func.id == 'dict' and not value.args:
            keys.append([key.arg for key in value.keywords])
        elif isinstance(value, ast.DictComp) and len(value.generators) == 1:
            gen = value.generators[0]
            if (not gen.ifs and isinstance(gen.target, ast.Name) and isinstance(gen.iter, (ast.Tuple, ast.List))
                    and isinstance(value.key, ast.Name) and value.key.id == gen.target.id
                    and isinstance(value.value, ast.Subscript) and isinstance(value.value.value, ast.Name)
                    and value.value.value.id == 'ctx' and isinstance(value.value.slice, ast.Name)
                    and value.value.slice.id == gen.target.id):
                keys.append([item.value if isinstance(item, ast.Constant) else '<expression>' for item in gen.iter.elts])
    valid = accesses == ['base', 'score', 'val'] and set(context_arguments) == {'planning'} and len(keys) == 1 and set(keys[0]) == {'score', 'val', 'base'}
    return dict(passed=valid, source=ref(path), safe_plan_context_keys=accesses, call_context_names=context_arguments,
        planning_dictionary_keys=keys, scope='Static sealed adapter check; numeric solver receives frozen score, validation, base, current noisy forecast and resource domain, with no truth or future voltage field.')


def run(source, destination):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if destination.exists():
        raise FileExistsError('Preserve the previous audit: '+str(destination))
    if not (source/'complete.json').exists():
        raise ValueError('Run must complete before independent auditing')
    self_test()
    destination.mkdir(parents=True)
    shutil.copyfile(__file__, destination/'audit.py')
    audit = Audit()
    protocol = read(audit.file(source/'protocol.json'))
    entries = read(audit.checked(protocol['inputs']))
    complete = read(audit.file(source/'complete.json'))
    completion = read(audit.checked(complete['main']))
    audit.require(complete.get('source_unchanged', False), 'Runner reported source drift')
    audit.require(tuple(protocol['policies']) == POLICIES, 'Policy protocol differs')
    audit.require(protocol['modes'] == ['activation', 'capacity'], 'Optimization modes differ')
    expected = {(entry['id'], draw) for entry in entries for draw in range(protocol['forecasts'])}
    audit.require(len({entry['id'] for entry in entries}) == len(entries), 'Duplicate inputs')
    receipts = {(row['id'], int(row['forecast_index'])) for row in completion['receipts']}
    failed = {(row['id'], int(row['forecast_index'])) for row in completion['failed_tasks']}
    audit.require(len(receipts) == len(completion['receipts']) and len(failed) == len(completion['failed_tasks']) and
        not (receipts & failed) and receipts | failed == expected, 'Completion tasks missing/duplicated')
    audit.require(completion['total_forecasts'] == len(expected) and completion['completed_forecasts'] == len(receipts), 'Completion task count differs')
    cases = {entry['case'] for entry in entries}
    audit.require(len(cases) == 1, 'Single physical network required by this locked audit')
    case = next(iter(cases))
    meta_dir = source/'source_snapshot/topology_identification_standalone/cases'/case
    metadata = read(audit.file(meta_dir/'metadata.json'))
    with audit.file(meta_dir/'branches.csv').open(encoding='utf-8-sig', newline='') as f:
        branches = list(csv.DictReader(f))
    with audit.file(meta_dir/'buses.csv').open(encoding='utf-8-sig', newline='') as f:
        buses = list(csv.DictReader(f))
    # Native support evaluator reads these same static physical tables. Check
    # equality with the sealed copy before reconstructing offline truth.
    current_case = native.ROOT/'topology_identification_standalone/cases'/case
    for filename in ('metadata.json', 'branches.csv', 'buses.csv'):
        audit.require(ref(current_case/filename)['sha256'] == ref(meta_dir/filename)['sha256'], 'Physical table changed since sealing')
        audit.file(current_case/filename)
    expected_rows = len(expected)*protocol['repeats']*len(POLICIES)
    if 'expected_rows' in protocol:
        audit.require(protocol['expected_rows'] == expected_rows, 'Protocol execution denominator differs')
    source_refs = []
    for item in protocol['sources']:
        relative = Path(item['path']).resolve().relative_to(native.ROOT)
        sealed = source/'source_snapshot'/relative
        audit.file(sealed, item['sha256'])
        source_refs.append(ref(sealed))
    boundary = planning_boundary(source/'source_snapshot/rnj_wzzt_dr/experiments/flexibility_stability_20261114/run.py')
    audit.require(boundary['passed'], 'Sealed planning adapter boundary differs from locked allowlist')
    own = dict(created_utc=datetime.now(timezone.utc).isoformat(), source_run=ref(source/'protocol.json'),
        source=ref(__file__), saved_source=ref(destination/'audit.py'), source_completion=ref(source/'complete.json'),
        helper_sources=[ref(native.__file__), ref(operation.__file__), ref(sys.modules[direct_original.__module__].__file__)],
        environment=dict(python=sys.version, numpy=np.__version__), phase=protocol['phase'],
        expected_tasks=len(expected), expected_rows=expected_rows, policies=list(POLICIES),
        planning_boundary=boundary, sealed_source_files=len(source_refs),
        scope='Post-execution stored-array recomputation only; no production selector, resource audit, fitting, optimization, or AC solve.',
        bill='All policies: requested max-absolute service-day capacity plus actual delivered activation. Activation-objective bounds and total-objective bounds remain separate.',
        truth='Only offline support recovery evaluation and post-execution evidence direction use the physical truth.',
        electrical_rating_scope='All bus voltages, branch currents, root P/Q/S and losses recomputed; base_mva is a normalization base, not a transformer nameplate. Thermal ratings absent.',
        independence_scope='Forecast/response/meter repetitions on one previously known physical day. Different resource variants share the same worlds.')
    write(destination/'protocol.json', own)
    for entry in entries:
        for draw in range(protocol['forecasts']):
            folder = source/'fraction0.10'/entry['id']/f'forecast{draw:02d}'
            try:
                if (entry['id'], draw) in failed:
                    raise ValueError('Original forecast task failed; no replacement is performed')
                audit.forecast(source, folder, entry, protocol, metadata, branches, buses)
            except Exception as exc:
                audit.task_errors.append(dict(path=str(folder), error=repr(exc), traceback=traceback.format_exc()))
            write(destination/'progress.json', dict(last_forecast=str(folder), examined_rows=len(audit.records),
                row_errors=sum(row['audit_status'] != 'passed' for row in audit.records), task_errors=len(audit.task_errors)))
    groups = defaultdict(list)
    for row in audit.records:
        groups[row['resource_variant']+'/'+row['strategy']].append(row)
    passed = len(audit.records) == expected_rows and not audit.task_errors and all(row['audit_status'] == 'passed' for row in audit.records)
    result = dict(status='passed' if passed else 'failed', protocol=ref(destination/'protocol.json'),
        expected_rows=expected_rows, audited_rows=len(audit.records),
        original_failed_tasks=completion['failed_tasks'], original_quote_exceptions=completion['quote_exceptions'],
        original_execution_failed_rows=completion['execution_failed_rows'], exact_diagnostic_failures=audit.exact_failures,
        task_errors=audit.task_errors, counts=dict(audit.counts), maximum_absolute_errors=dict(audit.maxima),
        electrical_maximum_errors=dict(audit.operation.maxima), groups={key:summarize(rows) for key, rows in groups.items()},
        paired=paired_outcomes(audit.records), quotes=audit.quote_records,
        rotation_unmatched_fees=audit.fee_mismatches, records=audit.records)
    write(destination/'inputs_manifest.json', [dict(path=path, sha256=digest) for path, digest in sorted(audit.hashes.items())])
    write(destination/'report.json', result)
    write(destination/'complete.json', dict(status=result['status'], report=ref(destination/'report.json'),
        inputs=ref(destination/'inputs_manifest.json'), source_unchanged=ref(__file__)['sha256'] == own['source']['sha256']))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(self_test()))
    else:
        if args.run is None or args.out is None:
            parser.error('--run and --out are required unless --self-test is set')
        result = run(args.run, args.out)
        print(json.dumps(dict(status=result['status'], rows=result['audited_rows'], expected=result['expected_rows'],
            row_errors=sum(row['audit_status'] != 'passed' for row in result['records']), task_errors=len(result['task_errors']))))
        raise SystemExit(0 if result['status'] == 'passed' else 1)
