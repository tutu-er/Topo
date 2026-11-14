"""Independent replay of a completed multi-network distribution-gate panel.

Only saved arrays are read. No production selector, evidence class, resource
domain, identifier, optimizer or AC solver is called. Recorded scientific
failures remain failures in the result ledger; missing or inconsistent evidence
fails this audit, rather than silently reducing a denominator.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import csv
from datetime import date, datetime, timedelta, timezone
import json
from pathlib import Path
import shutil
import sys
import traceback

import numpy as np

from ..distribution_gate_20261114 import probability as prob
from ..flexibility_stability_20261114 import audit as base

MODES = ('normal_fixed', 'mgf_fixed', 'mgf_forecast')
POLICIES = ('no_dfs', 'anchor', *MODES, *(m+'_rotation' for m in MODES))
ROOT = base.native.ROOT


def season_of(start_date):
    month = date.fromisoformat(start_date).month
    return ('winter' if month in (12, 1, 2) else 'spring' if month < 6
            else 'summer' if month < 9 else 'autumn')


def identity(entry):
    start = entry.get('start_date') or entry['id'].split('__')[-1]
    return dict(id=entry['id'], case=entry['case'], start_date=start,
        date_cluster=entry.get('date_cluster', start),
        season=entry.get('season', season_of(start)))


def json_line(stream, value):
    def convert(x):
        if isinstance(x, np.ndarray): return x.tolist()
        if isinstance(x, np.generic): return x.item()
        raise TypeError(type(x).__name__)
    stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False, default=convert)+'\n')


def planning_boundary(path):
    """A narrow static check of the recorded runner, not a general taint proof."""
    tree=ast.parse(Path(path).read_text(encoding='utf-8-sig'))
    functions=[node for node in tree.body if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)) and node.name=='run_forecast']
    if len(functions)!=1: raise ValueError('Expected one forecast runner')
    calls=[node for node in ast.walk(functions[0]) if isinstance(node,ast.Call)]
    contexts=[node for node in calls if ast.unparse(node.func)=='old.context']
    seals=[node for node in calls if ast.unparse(node.func)=='io.write' and node.args
        and isinstance(node.args[0],ast.BinOp) and isinstance(node.args[0].right,ast.Constant)
        and node.args[0].right.value=='seal.json']
    solves=[node for node in calls if ast.unparse(node.func)=='solver.plan']
    expected=['case',"planning['score']","planning['val']",'prediction',"planning['base']"]
    passed=len(contexts)==1 and len(seals)==1 and contexts[0].lineno>seals[0].lineno
    passed=passed and len(solves)==1 and [ast.unparse(arg) for arg in solves[0].args]==expected
    return dict(passed=passed,source=base.ref(path),seal_line=seals[0].lineno if len(seals)==1 else None,
        physical_context_line=contexts[0].lineno if len(contexts)==1 else None,
        solver_positional_inputs=[ast.unparse(arg) for arg in solves[0].args] if len(solves)==1 else [],
        scope='Recorded runner places actual context after seal and passes only contracts, historical score/validation, prediction and normalization base to the solver. This is a static adapter check, not a general data-flow proof.')


def resource_values(config, action, labels, forecast_p, power_base):
    """Independent scalar domain checks plus the declared P-to-Q coupling."""
    devices, violations, dp, dq = base.operation.resource_values(
        config, action, labels, forecast_p, power_base)
    positions = {int(v): k for k, v in enumerate(labels)}
    for col, (resource, unit) in enumerate(zip(config['resources'], config['action_units'], strict=True)):
        if unit == 'kW':
            dq[:, positions[int(resource['node_label'])]] += (
                np.repeat(action[:, col], 4)*float(resource.get('q_over_p', 0.))/power_base)
    return devices, violations, dp, dq


def exact_gates(model, forecast, power, mode):
    # A pool containing one topology has no foreign-family constraints. The
    # vacuous conditional certificate is kept separate from true-pool coverage.
    if len(set(model['families'])) == 1:
        return dict(rows=[], guards=[], target_failure_sums={str(t): 0. for t in model['targets']},
            worst_recovery_lower=1., feasible=True, minimum_margin=None, minimum_guard=None,
            tie=max(1e-7, (model['score']['cells']+power.shape[0]*model['n'])*
                model['score']['reference_mse']*model['score']['validation_tie']/(2*model['sigma2'])),
            probability_scope='Vacuous single-family conditional certificate; truth coverage is separate.')
    result = prob.exact_gates(model, forecast, power, mode)
    result['probability_scope'] = ('Conditional on each historical K mean, the stated fixed Gaussian '
        'law, and all frozen foreign-family native parameter points. Not an actual-AC guarantee.')
    return result


class Audit(base.Audit):
    def __init__(self):
        super().__init__()
        self.input_records = []

    def references(self, value):
        """Check every explicit path/hash reference in a sealed JSON object."""
        if isinstance(value, dict):
            if isinstance(value.get('path'), str) and isinstance(value.get('sha256'), str):
                self.checked(value)
            for child in value.values(): self.references(child)
        elif isinstance(value, list):
            for child in value: self.references(child)

    def catalogue(self, entry, config, ctx):
        resources, units = config['resources'], config['action_units']
        pc = sum(unit == 'kW' for unit in units)
        self.require(len(resources) == len(units) and units == ['kW']*pc+['kvar']*(len(units)-pc),
                     'Resource unit/order mismatch')
        self.require(len({r['id'] for r in resources}) == len(resources), 'Duplicate device ID')
        expected = entry['p_resources']+entry['q_resources']
        self.require(len(entry['p_resources']) == pc and len(resources) == len(expected), 'Contract count changed')
        for actual, declared in zip(resources, expected, strict=True):
            self.require(int(actual['node_label']) in ctx['labels'], 'Unmeasured resource location')
            for key in ('id', 'kind', 'node_label', 'contract_kw', 'capacity_kvar', 'q_over_p',
                        'fee_positive_yuan_per_kwh', 'fee_negative_yuan_per_kwh',
                        'fee_positive_yuan_per_kvarh', 'fee_negative_yuan_per_kvarh',
                        'available_clock_hours', 'hourly_fee_multiplier'):
                if key in declared:
                    self.require(key in actual and actual[key] == declared[key], 'Contract changed: '+key)
                    if key.startswith('fee_'):
                        self.require(np.isfinite(actual[key]) and actual[key]>=0,'Negative/nonfinite activation price invalidates zero-cost bound')
        return pc

    def coverage(self, entry, ctx, model):
        proper = lambda groups: {g for g in groups if 1 < len(g) < ctx['n']}
        correct = [i for i, active in enumerate(ctx['active']) if proper(active) == proper(ctx['truth'])]
        represented = model['representatives']
        targets = [represented[t] for t in model['targets']]
        return dict(**identity(entry), native_points=len(ctx['H']), representative_families=len(represented),
            true_native_indices=correct, truth_in_native_bank=bool(correct),
            truth_in_representatives=bool(set(correct)&set(represented)),
            truth_in_target_set=bool(set(correct)&set(targets)), selected_target_native_indices=targets,
            history_native_selected=ctx['target'], history_correct=ctx['target'] in correct,
            full_support_truth_in_native_bank=any(s == ctx['truth'] for s in ctx['active']),
            measured_terminals=ctx['n'],P_devices=len(entry['p_resources']),Q_devices=len(entry['q_resources']),
            P_total_contract_kw=sum(float(r['contract_kw']) for r in entry['p_resources']),
            Q_total_contract_kvar=sum(float(r['capacity_kvar']) for r in entry['q_resources']),
            P_resource_nodes=sorted({int(r['node_label']) for r in entry['p_resources']}),
            Q_resource_nodes=sorted({int(r['node_label']) for r in entry['q_resources']}))

    def gates(self, model, forecast, power, mode, saved=None):
        result = exact_gates(model, forecast, power, mode)
        if saved is not None and saved.get('status') not in ('evidence_construction_failed', 'evidence_check_exception'):
            for field, computed in (
                ('margins', [r['margin'] for r in result['rows']]),
                ('pair_failure', [r['pair_failure'] for r in result['rows']]),
                ('mu', [r['fixed_input_mean']-result['tie'] for r in result['rows']]),
                ('variance', [r['fixed_input_variance'] for r in result['rows']]),
                ('guard_gains', [g['gain'] for g in result['guards']])):
                self.same('gate_'+mode+'_'+field, saved[field], computed, atol=1e-8, rtol=1e-9)
            self.require(bool(saved['feasible']) == result['feasible'], 'Gate feasibility differs')
        return result

    def reported_evidence(self, ctx, summary, values, zero, forecast, forecast_zero, ideal=None, ideal_zero=None):
        """Check the saved historical-target summary from direct SSE contrasts."""
        target=ctx['target']
        self.require(summary['fixed_native_history_target']==target, 'Evidence target changed')
        pairs=summary['target_pairs']
        self.require(len(pairs)==len(ctx['H'])-1 and {r['competitor'] for r in pairs}==set(range(len(ctx['H'])))-{target},
            'Incomplete historical-target evidence pairs')
        for row in pairs:
            i,j=row['target'],row['competitor'];self.require(i==target,'Evidence orientation changed')
            a,b=self.directed(ctx,values,i,j),self.directed(ctx,zero,i,j)
            for field,name in (('actual_physical_J_gain','J'),('actual_bias_change','bias'),
                    ('actual_meter_change','meter'),('actual_logodds_change','observed_logodds'),
                    ('same_observations_logodds_change_without_common','observed_logodds_without_common')):
                self.same('reported_'+field,row[field],a[name]-b[name],atol=1e-8)
            a,b=self.directed(ctx,forecast,i,j),self.directed(ctx,forecast_zero,i,j)
            for field,name in (('forecast_J_gain','J'),('forecast_J_gain_without_common','J_without_common'),
                    ('forecast_common_square_gain','J_common'),('forecast_cross_gain','J_cross')):
                self.same('reported_'+field,row[field],a[name]-b[name],atol=1e-8)
            if ideal is not None and ideal_zero is not None:
                a,b=self.directed(ctx,ideal,i,j),self.directed(ctx,ideal_zero,i,j)
                self.same('reported_ideal_J_gain',row['exact_physical_J_gain'],a['J']-b['J'],atol=1e-8)

    def quote_check(self, entry, draw, mode, quote, action, direct, bill, seeds, gates_for_seed, anchor, planner):
        feasible = bool(quote.get('new_gate_feasible', quote.get('new_constraints_feasible', False)))
        self.require(feasible == bool(quote['feasible']) == bool(quote['new_constraints_feasible']), 'Quote feasibility flags differ')
        if feasible:
            self.require(direct['feasible'], 'Accepted quote violates its distribution gate')
            self.same('quote_activation_fee', quote['cost_yuan'], bill['activation_yuan'], atol=1e-7)
            self.require(not quote.get('anchor_fallback') and not quote.get('zero_fallback'), 'Feasible quote marked fallback')
        else:
            if quote.get('anchor_fallback'):
                self.require(bool(anchor.get('feasible')), 'Fallback anchor was not feasible')
                self.same('anchor_fallback', action, planner['anchor'], atol=0, rtol=0)
            elif quote.get('zero_fallback'):
                self.same('zero_fallback', action, np.zeros_like(action), atol=0, rtol=0)
            else: raise AssertionError('Failed quote lacks an explicit fallback')
        check_status=(quote.get('executed_gate_check') or {}).get('status')
        if check_status not in ('evidence_construction_failed','evidence_check_exception'):
            self.require(bool(quote['executed_gate_feasible']) == direct['feasible'], 'Executed-point flag differs')
        if quote.get('status') != 'exception':
            self.require(quote['original_nominal_gates_retained'] is False and quote['original_scalar_guards_retained'], 'Changed gate scope')
            self.require(quote['bound_cost_object'] == 'activation_yuan', 'Wrong cost-bound objective')
            self.require(len(quote['gate_rows']) == len(direct['rows']), 'Incomplete native comparisons')
            for saved, actual in zip(quote['gate_rows'], direct['rows'], strict=True):
                self.require((saved['target_rep'], saved['target_native'], saved['competitor_native']) ==
                    (actual['target'], actual['target_native'], actual['competitor_native']), 'Gate mapping mismatch')
                self.same('allocated_risk', saved['alpha'], actual['alpha'], atol=0, rtol=0)
                if mode != 'normal_fixed':
                    self.same('fixed_lambda', saved['lambda_selected'], actual['lambda_selected'], atol=0, rtol=0)
                    self.require(saved['lambda_halvings'] == actual['lambda_halvings'], 'Lambda rule changed')
        lower = quote.get('lower_bound_yuan')
        if lower is not None:
            self.require(np.isfinite(lower) and lower >= -1e-7, 'Invalid lower bound')
            if feasible: self.require(lower <= bill['activation_yuan']+1e-6, 'Bound exceeds verified incumbent')
        initial = quote.get('initial') or {}
        if initial.get('feasible'):
            self.require(initial['name'] in seeds, 'Initial point absent from sealed common pool')
            check, cost = gates_for_seed(seeds[initial['name']], mode)
            self.require(check['feasible'], 'Declared feasible initial point fails original probability expression')
            self.same('initial_cost', initial['cost_yuan'], cost, atol=1e-7)
        if all(k in quote for k in ('charged_total_seconds', 'shared_preparation_seconds', 'quote_wall_seconds')):
            self.same('charged_time', quote['charged_total_seconds'], quote['shared_preparation_seconds']+quote['quote_wall_seconds'], atol=1e-9)
        keys = ('status','native_status','source','initial','relative_gap','lower_bound_yuan','cost_yuan',
            'optimizer_invoked','native_attributes','charged_total_seconds','shared_preparation_seconds','quote_wall_seconds',
            'requested_quote_budget_seconds','quote_budget_exceeded','infeasibility_proved','lower_bound_withdrawn',
            'first_feasible_seconds','logging_errors')
        self.quote_records.append(dict(**identity(entry), forecast=draw, policy=mode,
            **{k: quote.get(k) for k in keys}, new_gate_feasible=feasible,
            anchor_fallback=bool(quote.get('anchor_fallback')), zero_fallback=bool(quote.get('zero_fallback')),
            executed_gate_feasible=direct['feasible'], direct_gate=direct, executed_cost=bill,
            minimum_buffered_margin=None if direct['minimum_margin'] is None else direct['minimum_margin']-1e-8,
            bound_scope='Activation-cost bound for this sufficient QCQP only; fallback is not its upper bound unless independently feasible.'))

    def forecast(self, folder, entry, protocol, metadata, branches, buses):
        draw = int(folder.name.removeprefix('forecast')); ctx = self.context(entry)
        seal = base.read(self.file(folder/'seal.json')); self.references(seal)
        planner = base.native.arrays(self.checked(seal['planner']))
        self.require(set(planner) == {'forecast', *POLICIES}, 'Missing or extra policy arrays')
        forecast = planner['forecast']; self.require(forecast.shape == (96, 2*ctx['n']), 'Forecast dimensions differ')
        true_power = np.c_[ctx['baseline_P'],ctx['baseline_Q']]
        expected_forecast = true_power*(1+.05*base.native.random_stream(protocol,entry,draw,0,991).normal(size=true_power.shape))
        self.same('forecast_error_stream',forecast,expected_forecast,atol=0,rtol=0)
        validation = base.native.arrays(self.checked(entry['files']['validation.npz']))
        model = prob.build_model(ctx['score'], validation, ctx['native'])
        if not any(r['id'] == entry['id'] for r in self.input_records): self.input_records.append(self.coverage(entry, ctx, model))
        config = base.read(self.file(folder/'resource_case.json')); pc = self.catalogue(entry, config, ctx)
        qc = len(config['resources'])-pc
        seeds = base.native.arrays(self.file(folder/'shared_initial_points.npz'))
        anchor = base.read(self.checked(seal['anchor'])); plans = base.read(self.checked(seal['plans']))
        self.references(anchor); self.references(plans)
        self.require(set(plans) == set(MODES), 'Quote identities differ')
        audits = base.read(self.checked(seal['prediction_audits']))
        reserves = base.native.arrays(self.checked(seal['reservation']))
        kp, kq = float(protocol['reservation_prices']['P']), float(protocol['reservation_prices']['Q'])
        controls, gates, forecasts, ideal = {}, {}, {}, {}
        self.same('no_purchase', planner['no_dfs'], np.zeros_like(planner['no_dfs']), atol=0, rtol=0)
        anchor_direct = base.direct_original(dict(entry=entry, score=ctx['score'], config=config,
            planner=planner, directory=folder), planner['anchor'], anchor)
        if anchor.get('feasible'): self.require(anchor_direct['passed_original_acceptance'], 'Anchor violates original constraints')
        else: self.same('failed_anchor_zero', planner['anchor'], np.zeros_like(planner['anchor']), atol=0, rtol=0)
        for policy in POLICIES:
            action = planner[policy]
            devices, violation, dp, dq = resource_values(config, action, ctx['labels'], forecast[:, :ctx['n']], ctx['base'])
            self.same('independent_injection_map', prob.shift(config, action, ctx['labels'], ctx['base']), np.c_[dp,dq], atol=1e-14)
            self.same('reserved_request_peak', reserves[policy], np.max(abs(action), axis=0), atol=0, rtol=0)
            bill = base.procurement(config, action, action, kp, kq)['planned']
            controls[policy] = dict(requested=devices, violations=violation, bill=bill)
            forecasts[policy] = self.evaluate(ctx, forecast+np.c_[dp,dq])
            for mode in MODES: gates[mode,policy] = self.gates(model, forecast, forecast+np.c_[dp,dq], mode, audits[mode][policy])
            ideal_path=folder/'exact_requested'/policy
            ideal_receipt=base.read(self.file(ideal_path/'receipt.json'));self.references(ideal_receipt)
            if ideal_receipt['status']=='complete':
                clean=base.native.arrays(self.file(ideal_path/'clean_observation.npz'))
                self.same('ideal_P',clean['P'],ctx['baseline_P']+dp,atol=1e-13)
                self.same('ideal_Q',clean['Q'],ctx['baseline_Q']+dq,atol=1e-13)
                self.same('ideal_Y',clean['Y'],clean['root'][:,None]**2-clean['V']**2,atol=1e-13)
                ideal[policy]=self.evaluate(ctx,np.c_[clean['P'],clean['Q']],clean['Y'],np.c_[clean['P'],clean['Q']],clean['Y'])
                self.pair_file(self.file(ideal_path/'pair_evidence.npz'),ctx,ideal[policy])
                self.pair_file(self.file(ideal_path/'forecast_pair_evidence.npz'),ctx,forecasts[policy])
            else:self.exact_failures.append(dict(id=entry['id'],forecast=draw,policy=policy,status=ideal_receipt['status'],error=ideal_receipt.get('error')))
        def seed_check(action, mode):
            _,_,dp,dq = resource_values(config, action, ctx['labels'], forecast[:, :ctx['n']], ctx['base'])
            return self.gates(model, forecast, forecast+np.c_[dp,dq], mode), base.procurement(config,action,action,kp,kq)['planned']['activation_yuan']
        for mode in MODES:
            self.quote_check(entry, draw, mode, plans[mode], planner[mode], gates[mode,mode],
                controls[mode]['bill'], seeds, seed_check, anchor, planner)
            self.quote_records[-1].update(zero_action_sufficient=gates[mode,'no_dfs']['feasible'],
                zero_action_direct_gate=gates[mode,'no_dfs'],
                nonzero_executed=bool(np.max(abs(planner[mode]),initial=0.)>1e-8))
        rotations = base.read(self.file(folder/'rotation_receipt.json'))
        self.require(set(rotations) == set(MODES), 'Incomplete rotation controls')
        for mode in MODES:
            matched = abs(controls[mode]['bill']['total_yuan']-controls[mode+'_rotation']['bill']['total_yuan']) <= 1e-7
            self.require(bool(rotations[mode]['same_planned_fee']) == matched, 'Rotation same-fee declaration differs')
            if not matched: self.fee_mismatches.append(dict(id=entry['id'], forecast=draw, policy=mode))
        task = base.read(self.file(folder/'complete.json')); self.references(task)
        rows = base.read(self.checked(task['results']))
        expected = {(r,p) for r in range(protocol['repeats']) for p in POLICIES}
        self.require(len(rows) == len(expected) and {(r['repeat'],r['strategy']) for r in rows} == expected, 'Execution identities missing/duplicated')
        current = {}; executed_values = {}
        for row in sorted(rows, key=lambda r: (r['repeat'], POLICIES.index(r['strategy']))):
            repeat, policy = row['repeat'], row['strategy']; rd = folder/f'execution{repeat:02d}'; path = rd/policy
            rec = dict(**identity(entry), forecast=draw, repeat=repeat, strategy=policy,
                source=str(path), original_status=row['status'])
            try:
                self.require(row == base.read(self.file(path/'receipt.json')), 'Execution receipt changed')
                if row['status'] != 'complete':
                    rec.update(audit_status='recorded_failure', error=row.get('error'), evaluation={'native_correct':False})
                    self.records.append(rec); continue
                random = base.native.arrays(self.file(rd/'paired_randomness.npz'))
                eta = np.r_[base.native.random_stream(protocol,entry,draw,repeat,773).uniform(.9,1,pc),
                    base.native.random_stream(protocol,entry,draw,repeat,774).uniform(.9,1,qc)]
                self.same('response_random_stream', random['eta'], eta, atol=0, rtol=0)
                shapes = (ctx['baseline_P'].shape,ctx['baseline_Q'].shape,ctx['baseline_P'].shape,ctx['baseline_root'].shape)
                for k, shape in enumerate(shapes):
                    self.same('meter_random_stream', random['noise'+str(k)], base.native.random_stream(protocol,entry,draw,repeat,k+2).normal(size=shape), atol=0, rtol=0)
                obs = base.native.arrays(self.file(path/'observations.npz')); action = planner[policy]
                self.same('sealed_request', obs['requested_action'], action, atol=0, rtol=0)
                expected_delivery = action*eta[None]
                for col, resource in enumerate(config['resources'][:pc]):
                    if resource['kind'] == 'pv':
                        node = ctx['labels'].index(resource['node_label'])
                        cap = np.minimum(resource['contract_kw'], np.maximum(0,-ctx['baseline_P'][:,node]*ctx['base']).reshape(24,4).min(1))
                        expected_delivery[:,col] = np.minimum(expected_delivery[:,col], cap)
                delivered = obs['delivered_action']; self.same('response_and_PV_clipping', delivered, expected_delivery, atol=1e-12, rtol=0)
                devices, violations, dp, dq = resource_values(config, delivered, ctx['labels'], forecast[:, :ctx['n']], ctx['base'])
                self.same('actual_P', obs['true_P'], ctx['baseline_P']+dp, atol=1e-13)
                self.same('actual_Q', obs['true_Q'], ctx['baseline_Q']+dq, atol=1e-13)
                for name,k,sigma in (('P',0,.005),('Q',1,.005),('V',2,.0005),('root',3,.0005)):
                    self.same('meter_'+name, obs[name], obs['true_'+name]*(1+sigma*random['noise'+str(k)]), atol=1e-13)
                self.same('observed_squared_voltage', obs['Y'], obs['root'][:,None]**2-obs['V']**2, atol=1e-13)
                bill = base.procurement(config, action, delivered, kp, kq)
                for field in ('planned','settled'):
                    self.costs(row[field+'_cost'], bill[field], field)
                    self.same('paid_request_capacity', row[field+'_cost']['resource_reserved_capacity'], np.max(abs(action),axis=0), atol=0, rtol=0)
                    self.same(field+'_fee', row[field+'_fee_yuan'], bill[field]['total_yuan'], atol=1e-8)
                for field, value in (('requested_quantities',action),('delivered_quantities',delivered)):
                    self.same(field+'_P',row[field]['active_abs_kwh'],abs(value[:,:pc]).sum(),atol=1e-9)
                    self.same(field+'_Q',row[field]['reactive_abs_kvarh'],abs(value[:,pc:]).sum(),atol=1e-9)
                evaluation, values = self.score(ctx, obs, row)
                self.pair_file(self.file(path/'pair_evidence.npz'), ctx, values)
                executed_values[repeat,policy]=values
                if (repeat,'no_dfs') in executed_values:
                    self.reported_evidence(ctx,row['evidence'],values,executed_values[repeat,'no_dfs'],
                        forecasts[policy],forecasts['no_dfs'],ideal.get(policy),ideal.get('no_dfs'))
                else:self.require(row['evidence'].get('status')=='unavailable','Zero execution failed but evidence summary claims availability')
                if 'pair' in evaluation:
                    i,j = evaluation['pair']; evaluation['forecast_evidence'] = self.directed(ctx,forecasts[policy],i,j)
                    if i in model['representatives']:
                        target = model['representatives'].index(i); K = model['means'][target]
                        Hi,Hj = model['H'][i],model['H'][j]; actual_power = np.c_[obs['true_P'],obs['true_Q']]
                        dpq = prob.shift(config,action,ctx['labels'],ctx['base']); nominal = forecast+dpq
                        def mean(x):
                            y=x@K.T
                            return float(evaluation['evidence']['h']+(np.square(y-x@Hj.T).sum()-np.square(y-x@Hi.T).sum())/(2*model['sigma2']))
                        nominal_mu,actual_mu = mean(nominal),mean(actual_power)
                        clean_y=obs['true_root'][:,None]**2-obs['true_V']**2
                        mean_error=clean_y-actual_power@K.T
                        frozen_error=clean_y-actual_power@Hi.T
                        evaluation['historical_mean_diagnostic'] = dict(native_target=i,native_competitor=j,
                            target_in_selected_set=target in model['targets'], K_nominal_forecast_mu=nominal_mu,
                            K_actual_input_mu=actual_mu, actual_power_minus_nominal_forecast=actual_mu-nominal_mu,
                            clean_minus_actual_K=evaluation['evidence']['clean_logodds']-actual_mu,
                            observed_minus_clean=evaluation['evidence']['meter'],
                            clean_minus_nominal_K=evaluation['evidence']['clean_logodds']-nominal_mu,
                            clean_Y_minus_K_mean=float(mean_error.mean()),clean_Y_minus_K_rmse=float(np.sqrt(np.square(mean_error).mean())),
                            clean_Y_minus_frozen_H_mean=float(frozen_error.mean()),clean_Y_minus_frozen_H_rmse=float(np.sqrt(np.square(frozen_error).mean())),
                            Y_units='squared-voltage pu; positive means actual root-squared minus terminal-squared voltage exceeds the fitted mean')
                    if (repeat,'no_dfs') in current and 'evidence' in current[repeat,'no_dfs']['evaluation']:
                        zero=current[repeat,'no_dfs']['evaluation']['evidence']
                        evaluation['evidence_increment']={k:evaluation['evidence'][k]-zero[k] for k in ('J','bias','meter','clean_logodds','observed_logodds')}
                    elif policy == 'no_dfs': evaluation['evidence_increment']={k:0. for k in ('J','bias','meter','clean_logodds','observed_logodds')}
                physical = row['physical_state']; ac = base.native.arrays(self.checked(physical))
                if 'time_slice' in physical:
                    a,b=physical['time_slice']; count=len(ac['voltage_complex_pu'])
                    ac={k:v[a:b] if v.ndim and v.shape[0]==count else v for k,v in ac.items()}
                electrical, branch_values, bus_values = base.operation.electrical(self.operation,ac,obs,metadata,branches,buses)
                voltage_pass = electrical['cells_below_095']+electrical['cells_above_105'] == 0
                rec.update(audit_status='passed',requested=controls[policy]['requested'],delivered=devices,
                    requested_violations=controls[policy]['violations'],delivered_violations=violations,
                    planned_cost=bill['planned'],settled_cost=bill['settled'],evaluation=evaluation,
                    electrical=electrical,branches=branch_values,buses=bus_values,voltage_screen_passed=voltage_pass,
                    recovered_and_voltage_screen_passed=bool(evaluation['native_correct'] and voltage_pass),
                    thermal_rating_status='unavailable_not_verified',
                    quote_new_gate_feasible=row.get('quote_new_constraints_feasible'),
                    quote_anchor_fallback=row.get('quote_anchor_fallback'))
                current[repeat,policy]=rec
            except Exception as exc:
                rec.update(audit_status='failed',error=repr(exc),traceback=traceback.format_exc())
            self.records.append(rec)


def entries_from(value):
    return value if isinstance(value,list) else value['inputs']


def run(source, out):
    source,out=Path(source).resolve(),Path(out).resolve()
    if out.exists(): raise FileExistsError(out)
    if not (source/'complete.json').exists(): raise ValueError('Completed run required')
    out.mkdir(parents=True); shutil.copyfile(__file__,out/'audit.py'); audit=Audit()
    protocol=base.read(audit.file(source/'protocol.json'))
    entries=entries_from(base.read(audit.checked(protocol['inputs'])))
    audit.require(tuple(protocol['policies'])==POLICIES, 'Unexpected panel policies')
    complete=base.read(audit.file(source/'complete.json')); audit.references(complete)
    summary=base.read(audit.checked(complete['main'])) if 'main' in complete else complete
    audit.require(complete.get('source_unchanged',True), 'Run reports changed sources')
    for item in protocol['sources']:
        relative=Path(item['path']).resolve().relative_to(ROOT)
        audit.file(source/'source_snapshot'/relative,item['sha256'])
    boundary=planning_boundary(source/'source_snapshot'/protocol['driver_relative_path'])
    audit.require(boundary['passed'],'Recorded planning boundary differs')
    expected={(e['id'],d) for e in entries for d in range(protocol['forecasts'])}
    successful=summary.get('receipts',[]); failed=summary.get('failed_tasks',[])
    task_key=lambda r:(r['id'],int(r.get('forecast_index',r.get('forecast'))))
    good_keys,failed_keys={task_key(r) for r in successful},{task_key(r) for r in failed}
    audit.require(not(good_keys&failed_keys) and good_keys|failed_keys==expected and len(good_keys)==len(successful)
        and len(failed_keys)==len(failed),'Incomplete or duplicated forecast task ledger')
    own=dict(created_utc=datetime.now(timezone.utc).isoformat(),source=base.ref(__file__),source_run=base.ref(source/'protocol.json'),
        policies=POLICIES,planning_boundary=boundary,expected_execution_rows=len(expected)*protocol['repeats']*len(POLICIES),
        scope='Independent frozen native score, Gaussian original expressions, request/delivery/fees and stored full-network AC. No fitting, optimization, AC solve or new random path.',
        environment=dict(python=sys.version,numpy=np.__version__))
    helpers=[prob,base,base.native,base.operation,sys.modules[base.direct_original.__module__]]
    own['helper_sources']=[base.ref(m.__file__) for m in helpers]
    for item in own['helper_sources']:
        path=Path(item['path']); dest=out/'helper_snapshot'/path.relative_to(ROOT)
        dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,dest)
    base.write(out/'protocol.json',own)
    failed_rows=[]
    for entry in entries:
        try:
            audit.references(entry)
            meta=source/'source_snapshot/topology_identification_standalone/cases'/entry['case']
            metadata=base.read(audit.file(meta/'metadata.json'))
            with audit.file(meta/'branches.csv').open(encoding='utf-8-sig',newline='') as stream: branches=list(csv.DictReader(stream))
            with audit.file(meta/'buses.csv').open(encoding='utf-8-sig',newline='') as stream: buses=list(csv.DictReader(stream))
            for name in ('metadata.json','branches.csv','buses.csv'):
                audit.require(base.ref(meta/name)['sha256']==base.ref(ROOT/'topology_identification_standalone/cases'/entry['case']/name)['sha256'], 'Physical table changed since snapshot')
            ctx=audit.context(entry)
            model=prob.build_model(ctx['score'],base.native.arrays(audit.checked(entry['files']['validation.npz'])),ctx['native'])
            audit.input_records.append(audit.coverage(entry,ctx,model))
        except Exception as exc:
            audit.task_errors.append(dict(id=entry['id'],error=repr(exc),traceback=traceback.format_exc()));continue
        for draw in range(protocol['forecasts']):
            folder=source/'fraction0.10'/entry['id']/f'forecast{draw:02d}'
            if (entry['id'],draw) in failed_keys:
                failed_rows.extend(dict(**identity(entry),forecast=draw,repeat=r,strategy=policy,
                    audit_status='recorded_failure',original_status='forecast_task_failed',evaluation={'native_correct':False})
                    for r in range(protocol['repeats']) for policy in POLICIES)
                continue
            try: audit.forecast(folder,entry,protocol,metadata,branches,buses)
            except Exception as exc: audit.task_errors.append(dict(path=str(folder),error=repr(exc),traceback=traceback.format_exc()))
            base.write(out/'progress.json',dict(rows=len(audit.records)+len(failed_rows),task_errors=len(audit.task_errors),
                row_errors=sum(r['audit_status']=='failed' for r in audit.records)))
    audit.records.extend(failed_rows)
    for path, sha in audit.hashes.items():
        if base.ref(path)['sha256'] != sha: audit.task_errors.append(dict(path=path,error='Input changed during audit'))
    passed=(len(audit.records)==own['expected_execution_rows'] and not audit.task_errors
        and all(r['audit_status'] in ('passed','recorded_failure') for r in audit.records))
    for name, rows in (('records',audit.records),('quotes',audit.quote_records)):
        with (out/(name+'.jsonl')).open('w',encoding='utf8') as stream:
            for row in rows: json_line(stream,row)
    result=dict(status='passed' if passed else 'failed',protocol=base.ref(out/'protocol.json'),
        source_protocol=base.ref(source/'protocol.json'),source_inputs=protocol['inputs'],
        expected_rows=own['expected_execution_rows'],audited_rows=len(audit.records),
        audit_status_counts=dict(Counter(r['audit_status'] for r in audit.records)),
        input_coverage=audit.input_records,original_failed_tasks=failed,
        original_quote_exceptions=summary.get('quote_exceptions',[]),
        original_execution_failed_rows=summary.get('execution_failed_rows',[]),
        ideal_requested_diagnostic_failures=audit.exact_failures,
        task_errors=audit.task_errors,records=base.ref(out/'records.jsonl'),quotes=base.ref(out/'quotes.jsonl'),
        rotation_unmatched_fees=audit.fee_mismatches,maximum_absolute_errors=dict(audit.maxima),
        electrical_maximum_errors=dict(audit.operation.maxima),counts=dict(audit.counts),
        missing_thermal_ratings='Currents and apparent powers are measured; allowable thermal loading is not verified.')
    base.write(out/'report.json',result)
    base.write(out/'inputs_manifest.json',[dict(path=k,sha256=v) for k,v in audit.hashes.items()])
    base.write(out/'complete.json',dict(status=result['status'],report=base.ref(out/'report.json'),
        source_unchanged=base.ref(__file__)['sha256']==own['source']['sha256']))
    return result


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();result=run(args.run,args.out)
    print({k:result[k] for k in ('status','expected_rows','audited_rows','audit_status_counts')})
    raise SystemExit(0 if result['status']=='passed' else 1)
