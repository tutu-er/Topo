"""Independent saved-array audit of the extra activation-budget policy.

This module imports only earlier independent audit/aggregation helpers. It
never calls the experiment runner, a resource-domain audit, an optimizer or AC
solver. Original comparison outcomes come from the completed original audit.
"""
from __future__ import annotations

import argparse
import ast
from collections import defaultdict
import csv
from datetime import datetime, timezone
import json
import hashlib
from pathlib import Path
import shutil
import sys
import traceback

import numpy as np

from ..flexibility_stability_20261114 import audit as base
from ..flexibility_stability_20261114.analyze import SealedFiles, paired


POLICY = 'budget_capacity'
ORIGINAL_VARIANT = 'original_p10_q36'


def activation_float(config, action):
    """Independent fee matrix with the recorded row-major summation order."""
    action = np.asarray(action, float)
    plus = np.zeros_like(action); minus = np.zeros_like(action)
    clock = (8+np.arange(24)+.5) % 24
    factor = np.where(clock < 7, .8, np.where(((clock >= 7) & (clock < 10)) |
        ((clock >= 17) & (clock < 22)), 1.25, 1.))
    for col, (resource, unit) in enumerate(zip(config['resources'], config['action_units'], strict=True)):
        suffix = 'kwh' if unit == 'kW' else 'kvarh'
        price = factor if unit == 'kW' else np.asarray(resource.get('hourly_fee_multiplier', np.ones(24)), float)
        plus[:, col] = price*resource['fee_positive_yuan_per_'+suffix]
        minus[:, col] = price*resource['fee_negative_yuan_per_'+suffix]
    return float(np.sum(plus*np.maximum(action, 0)+minus*np.maximum(-action, 0)))


def budget_terms(config, action, anchor, kappa_p=.5, kappa_q=.5, allowance=.01):
    """Evaluate the original, unrounded activation-fee cap at a saved point."""
    final = base.procurement(config, action, action, kappa_p, kappa_q)['planned']
    initial = base.procurement(config, anchor, anchor, kappa_p, kappa_q)['planned']
    anchor_float, final_float = activation_float(config, anchor), activation_float(config, action)
    limit = anchor_float+allowance
    margin = limit-final_float
    return dict(anchor_activation_yuan=anchor_float, allowance_yuan=allowance,
        activation_limit_yuan=limit, final_activation_yuan=final_float,
        final_total_yuan=final['total_yuan'], margin_yuan=margin, strict_passed=bool(margin >= 0),
        passed_to_1e_8_yuan=bool(margin >= -1e-8),
        alternative_column_sum_difference_yuan=final_float-final['activation_yuan'])


def compare_original(records, original_report):
    source = [row for row in original_report['records'] if row['resource_variant'] == ORIGINAL_VARIANT]
    return {policy: paired(records, [row for row in source if row['strategy'] == policy]) for policy in base.POLICIES}


def self_test():
    config = dict(action_units=['kvar'], resources=[dict(id='q', fee_positive_yuan_per_kvarh=.05,
        fee_negative_yuan_per_kvarh=.05)])
    anchor = np.zeros((24, 1)); anchor[0, 0] = 5.
    within = np.zeros_like(anchor); within[:2, 0] = 2.5
    outside = np.zeros_like(anchor); outside[:3, 0] = 2.
    one, two = budget_terms(config, within, anchor), budget_terms(config, outside, anchor)
    assert one['strict_passed'] and not two['passed_to_1e_8_yuan']
    assert one['activation_limit_yuan'] == .25+.01
    assert one['final_total_yuan'] == .25+1.25
    assert two['final_total_yuan'] < .25+2.5
    return dict(passed=True, scope='Fixed arithmetic: cheaper total cost alone does not satisfy the activation cap; no solving or AC.')


class Audit(base.Audit):
    def __init__(self, original):
        super().__init__()
        self.original = original

    def original_file(self, path):
        """Read only an original file already covered by its completed audit."""
        return self.original.audited(path)

    def math_quote(self, folder, entry, context, config, planner, receipt, anchor, kappa_p, kappa_q):
        point = planner[POLICY]
        terms = budget_terms(config, point, anchor, kappa_p, kappa_q)
        direct = base.direct_original(dict(entry=entry, score=context['score'], config=config,
            planner=planner, directory=folder), point, receipt)
        self.require(terms['strict_passed'], 'Returned point exceeds the strict unrounded activation budget')
        if receipt.get('feasible'):
            self.require(direct['passed_original_acceptance'], 'Returned point fails original evidence/guard')
            self.same('final_total_objective', receipt['cost_yuan'], terms['final_total_yuan'], atol=1e-7)
        lower = receipt.get('lower_bound_yuan')
        if lower is not None:
            self.require(np.isfinite(lower) and -1e-7 <= lower <= terms['final_total_yuan']+1e-6,
                'Budgeted-domain total-cost lower bound invalid at the returned point')
        return dict(budget=terms, original_constraints=direct,
            bound_scope='Total activation-plus-capacity objective on the original constraints with the additional activation budget; not an unconstrained-procurement bound.')

    def forecast(self, folder, entry, protocol, metadata, branches, buses):
        ctx = self.context(entry); draw = int(folder.name.removeprefix('forecast'))
        source = self.original.source_root/'fraction0.10'/entry['id']/f'forecast{draw:02d}'
        seal = base.read(self.file(folder/'seal.json'))
        for key in ('planner','plans','reservation','source_seal','source_anchor','source_shared_points',
                    'source_shared_receipt','score','protocol'):
            self.checked(seal[key])
        old_seal = self.original_file(source/'seal.json')
        old_plans = self.original_file(source/'plans.json')
        old_planner = self.original_file(source/'planner.npz')
        old_anchor = self.original_file(source/'anchor_plan.json')
        self.require(seal['source_anchor'] == old_seal['anchor'], 'Anchor reference changed')
        self.require(seal['source_shared_points'] == old_plans['activation']['initial_points'], 'Shared seed reference changed')
        seeds = self.original_file(seal['source_shared_points']['path'])
        planner = base.native.arrays(self.file(folder/'planner.npz'))
        self.require(set(planner) == {'forecast','anchor',POLICY}, 'Unexpected planner arrays')
        self.same('original_forecast',planner['forecast'],old_planner['forecast'],atol=0,rtol=0)
        self.same('original_anchor',planner['anchor'],old_planner['anchor'],atol=0,rtol=0)
        forecast=planner['forecast'];anchor=planner['anchor'];u=planner[POLICY]
        config=base.read(self.file(folder/'resource_case.json'))
        self.require(config == self.original_file(source/'resource_case.json'), 'Physical contract changed')
        p_count=self.catalogue(entry,config,ctx);q_count=len(config['resources'])-p_count
        prediction=np.c_[ctx['baseline_P'],ctx['baseline_Q']]*(1+.05*base.native.random_stream(protocol,entry,draw,0,991).normal(size=forecast.shape))
        self.same('forecast_random_law',forecast,prediction,atol=0,rtol=0)
        plans=base.read(self.file(folder/'plans.json'));self.require(set(plans)=={POLICY},'Extra policy plan')
        q=plans[POLICY];kp=float(protocol['reservation_prices']['P']);kq=float(protocol['reservation_prices']['Q'])
        terms=self.math_quote(folder,entry,ctx,config,planner,q,anchor,kp,kq)
        if old_anchor['feasible']:
            ad=base.direct_original(dict(entry=entry,score=ctx['score'],config=config,planner=planner,directory=folder),anchor,q)
            self.require(ad['passed_original_acceptance'],'Original fixed anchor is not feasible')
            self.same('executed_call_cap',q['activation_limit_yuan'],terms['budget']['activation_limit_yuan'],atol=0,rtol=0)
        self.same('executed_activation_fee',q['executed_activation_yuan'],activation_float(config,u),atol=0,rtol=0)
        self.same('charged_shared_time',q['shared_preparation_seconds'],old_plans['activation']['shared_preparation_seconds'],atol=0,rtol=0)
        self.same('charged_total_time',q['charged_total_seconds'],q['shared_preparation_seconds']+q['quote_wall_seconds'],atol=1e-12)
        fctx=dict(entry=entry,score=ctx['score'],config=config,planner=planner,directory=folder)
        if q.get('activation_cap_yuan') is not None:
            self.same('solver_call_cap',q['activation_cap_yuan'],terms['budget']['activation_limit_yuan'],atol=0,rtol=0)
            self.same('solver_anchor_fee',q['anchor_activation_yuan'],activation_float(config,anchor),atol=0,rtol=0)
            self.require(q['lower_bound_valid_for_uncapped_problem'] is False,'Restricted bound mislabeled')
            self.require(q['objective_cost_object']==q['bound_cost_object']=='activation_plus_daily_reserved_capacity','Wrong objective/bound object')
            self.require(q['anchor_action_sha256']==action_hash(anchor),'Wrong anchor action hash')
            if not q.get('anchor_fallback'):
                self.require(q['final_action_sha256']==action_hash(u),'Executed point differs from final wrapper point')
            call=q['call_budget'];self.same('saved_call_slack',call['slack_yuan'],terms['budget']['margin_yuan'],atol=0,rtol=0)
            self.require(call['passed'] and call['allowance_tolerance_yuan']==0,'Cap comparison not strict')
        if q.get('fallback_used') or q.get('anchor_fallback'):
            self.same('strict_anchor_fallback',u,anchor,atol=0,rtol=0)
        if q.get('zero_fallback'):
            self.same('unresolved_zero',u,np.zeros_like(u),atol=0,rtol=0)
        filtered={}
        for item in q.get('seed_filter',[]):
            name=item['name'];self.require(name in seeds,'Unexpected seed source')
            point=seeds[name]
            self.require(item['action_sha256']==action_hash(point),'Seed action hash differs')
            physical=True
            try:base.operation.resource_values(config,point,ctx['labels'],forecast[:,:ctx['n']],ctx['base'])
            except ArithmeticError:physical=False
            cap=activation_float(config,point)<=terms['budget']['activation_limit_yuan']
            self.require(item['resource_feasible']==physical and item['call_budget']['passed']==cap,'Seed filter differs')
            self.require(item['accepted_by_resource_and_call_cap']==(physical and cap),'Seed admission differs')
            if physical and cap:
                filtered[name]=point
                d=base.direct_original(fctx,point,q)
                self.require(item['evidence_feasible']==d['passed_original_acceptance'],'Seed original evidence check differs')
        filtered['__same_forecast_150s_anchor__']=anchor
        initial=q.get('initial') or {}
        if initial.get('feasible'):
            self.require(initial['name'] in filtered,'Selected initial is outside the filtered common pool')
            point=filtered[initial['name']]
            d=base.direct_original(fctx,point,q);self.require(d['passed_original_acceptance'],'Initial violates original gate')
            bill=base.procurement(config,point,point,kp,kq)['planned']
            self.same('initial_total_objective',initial['cost_yuan'],bill['total_yuan'],atol=1e-7)
        engine={}
        if q.get('engine_returned_action'):
            eu=base.native.arrays(self.checked(q['engine_returned_action']))['action']
            self.require(action_hash(eu)==q['engine_returned_action_sha256'],'Raw engine-point hash differs')
            engine=dict(call_budget=budget_terms(config,eu,anchor,kp,kq),same_as_executed=bool(np.array_equal(eu,u)))
            if not(q.get('fallback_used') or q.get('anchor_fallback')):
                self.same('engine_return_equals_execution',eu,u,atol=0,rtol=0)
        self.quote_records.append(dict(id=entry['id'],forecast=draw,policy=POLICY,status=q.get('status'),
            feasible=bool(q.get('feasible')),executed_source=q['executed_source'],
            anchor_fallback=bool(q.get('anchor_fallback')),wrapper_fallback=bool(q.get('fallback_used')),
            fallback_reason=q.get('fallback_reason'),zero_fallback=q['zero_fallback'],initial=initial,
            cost_yuan=q.get('cost_yuan'),executed_cost_yuan=q['executed_cost']['total_yuan'],
            lower_bound_yuan=q.get('lower_bound_yuan'),relative_gap=q.get('relative_gap'),native_attributes=q.get('native_attributes'),
            charged_total_seconds=q['charged_total_seconds'],quote_wall_seconds=q['quote_wall_seconds'],
            original_constraints=terms['original_constraints'],budget=terms['budget'],raw_engine_point=engine,
            bound_scope=terms['bound_scope']))
        requested,rv,dp,dq=base.operation.resource_values(config,u,ctx['labels'],forecast[:,:ctx['n']],ctx['base'])
        reserve=base.native.arrays(self.file(folder/'reservation.npz'));self.require(set(reserve)=={POLICY},'Extra reserve vector')
        self.same('purchased_peak_capacity',reserve[POLICY],np.max(abs(u),axis=0),atol=0,rtol=0)
        forecast_values=self.evaluate(ctx,forecast+np.c_[dp,dq])
        exact_dir=folder/'exact_requested'/POLICY;exact_receipt=base.read(self.file(exact_dir/'receipt.json'));exact=None
        if exact_receipt['status']=='complete':
            clean=base.native.arrays(self.file(exact_dir/'clean_observation.npz'))
            self.same('ideal_requested_P',clean['P'],ctx['baseline_P']+dp,atol=1e-13)
            self.same('ideal_requested_Q',clean['Q'],ctx['baseline_Q']+dq,atol=1e-13)
            self.same('ideal_requested_Y',clean['Y'],clean['root'][:,None]**2-clean['V']**2,atol=1e-13)
            exact=self.evaluate(ctx,np.c_[clean['P'],clean['Q']],clean['Y'],np.c_[clean['P'],clean['Q']],clean['Y'])
            self.pair_file(self.file(exact_dir/'pair_evidence.npz'),ctx,exact)
            self.pair_file(self.file(exact_dir/'forecast_pair_evidence.npz'),ctx,forecast_values)
        else:self.exact_failures.append(dict(path=str(exact_dir),status=exact_receipt['status']))
        task=base.read(self.file(folder/'complete.json'));rows=base.read(self.checked(task['results']))
        self.require(task['rows']==protocol['repeats'] and len(rows)==protocol['repeats'],'Execution denominator differs')
        self.require(task['failed_rows']==sum(r['status']!='complete' for r in rows),'Failed-row count differs')
        self.require({(r['repeat'],r['strategy']) for r in rows}=={(repeat,POLICY) for repeat in range(protocol['repeats'])},'Execution identity missing/duplicated')
        source_rows={(r['forecast'],r['repeat'],r['strategy']):r for r in self.original.report['records'] if r['resource_variant']==ORIGINAL_VARIANT}
        for row in rows:
            repeat=row['repeat'];rd=folder/f'execution{repeat:02d}';path=rd/POLICY
            record=dict(id=entry['id'],resource_variant=ORIGINAL_VARIANT,forecast=draw,repeat=repeat,strategy=POLICY,
                source=str(path),original_status=row['status'])
            try:
                self.require(row==base.read(self.file(path/'receipt.json')),'Individual/combined receipt differs')
                self.require(row['status']=='complete','Original execution failed')
                random=base.native.arrays(self.file(rd/'paired_randomness.npz'))
                old_random=self.original_file(source/f'execution{repeat:02d}'/'paired_randomness.npz')
                for key,value in random.items():self.same('source_random_'+key,value,old_random[key],atol=0,rtol=0)
                eta=np.r_[base.native.random_stream(protocol,entry,draw,repeat,773).uniform(.9,1.,p_count),
                    base.native.random_stream(protocol,entry,draw,repeat,774).uniform(.9,1.,q_count)]
                self.same('response_law',random['eta'],eta,atol=0,rtol=0)
                for k,shape in enumerate((ctx['baseline_P'].shape,ctx['baseline_Q'].shape,ctx['baseline_P'].shape,ctx['baseline_root'].shape)):
                    self.same('meter_law_'+str(k),random['noise'+str(k)],base.native.random_stream(protocol,entry,draw,repeat,k+2).normal(size=shape),atol=0,rtol=0)
                obs=base.native.arrays(self.file(path/'observations.npz'))
                self.same('sealed_request',obs['requested_action'],u,atol=0,rtol=0)
                expected=u*eta[None]
                for col,res in enumerate(config['resources'][:p_count]):
                    if res['kind']=='pv':
                        node=ctx['labels'].index(res['node_label'])
                        cap=np.minimum(res['contract_kw'],np.maximum(0.,-ctx['baseline_P'][:,node]*ctx['base']).reshape(24,4).min(1))
                        expected[:,col]=np.minimum(expected[:,col],cap)
                actual=obs['delivered_action'];self.same('delivered_response_clip',actual,expected,atol=1e-12,rtol=0)
                delivered,av,ap,aq=base.operation.resource_values(config,actual,ctx['labels'],forecast[:,:ctx['n']],ctx['base'])
                self.same('executed_P',obs['true_P'],ctx['baseline_P']+ap,atol=1e-13)
                self.same('executed_Q',obs['true_Q'],ctx['baseline_Q']+aq,atol=1e-13)
                for name,index,sigma in [('P',0,.005),('Q',1,.005),('V',2,.0005),('root',3,.0005)]:
                    self.same('measured_'+name,obs[name],obs['true_'+name]*(1+sigma*random['noise'+str(index)]),atol=1e-13)
                self.same('measured_Y',obs['Y'],obs['root'][:,None]**2-obs['V']**2,atol=1e-13)
                bill=base.procurement(config,u,actual,kp,kq)
                self.costs(row['planned_cost'],bill['planned'],'planned_cost');self.costs(row['settled_cost'],bill['settled'],'settled_cost')
                self.same('total_planned_bill',row['planned_fee_yuan'],bill['planned']['total_yuan'],atol=1e-8)
                self.same('total_settled_bill',row['settled_fee_yuan'],bill['settled']['total_yuan'],atol=1e-8)
                for key in ('planned_cost','settled_cost'):
                    self.same('paid_requested_peak',row[key]['resource_reserved_capacity'],np.max(abs(u),axis=0),atol=0,rtol=0)
                for unit,field in [('kW','active_abs_kwh'),('kvar','reactive_abs_kvarh')]:
                    for name,devices in [('requested',requested),('delivered',delivered)]:
                        self.same(name+'_'+field,row[name+'_quantities'][field],sum(d['absolute_energy'] for d in devices if d['unit']==unit),atol=1e-8)
                evaluation,values=self.score(ctx,obs,row);self.pair_file(self.file(path/'pair_evidence.npz'),ctx,values)
                if 'pair' in evaluation:
                    i,j=evaluation['pair'];evaluation['forecast_evidence']=self.directed(ctx,forecast_values,i,j)
                    if exact is not None:evaluation['ideal_evidence']=self.directed(ctx,exact,i,j)
                    zero=source_rows[(draw,repeat,'no_dfs')]['evaluation']
                    evaluation['evidence_increment']={key:evaluation['evidence'][key]-zero['evidence'][key]
                        for key in ('J','bias','meter','clean_logodds','observed_logodds')}
                    source0=self.original_file(source/f'execution{repeat:02d}'/'no_dfs'/'pair_evidence.npz')
                    changes=base.native.arrays(self.file(path/'pair_changes.npz'))
                    current=base.native.arrays(path/'pair_evidence.npz')
                    for name in ('J','bias','meter','clean_logodds','observed_logodds'):
                        if name in changes:self.same('saved_change_'+name,changes[name],current[name]-source0[name],atol=1e-8)
                    self.verify_reported_evidence(row['evidence'],current,source0,
                        base.native.arrays(exact_dir/'forecast_pair_evidence.npz') if exact is not None else None,
                        self.original_file(source/'exact_requested/no_dfs/forecast_pair_evidence.npz'),
                        base.native.arrays(exact_dir/'pair_evidence.npz') if exact is not None else None,
                        self.original_file(source/'exact_requested/no_dfs/pair_evidence.npz'))
                physical=row['physical_state'];ac=base.native.arrays(self.checked(physical))
                if 'time_slice' in physical:
                    start,stop=physical['time_slice'];count=len(ac['voltage_complex_pu'])
                    ac={k:v[start:stop] if v.ndim and v.shape[0]==count else v for k,v in ac.items()}
                electrical,branch_values,bus_values=base.operation.electrical(self.operation,ac,obs,metadata,branches,buses)
                self.same('reported_voltage_min',row['all_bus_voltage_min'],electrical['all_bus_voltage']['minimum'],atol=1e-13)
                self.same('reported_voltage_max',row['all_bus_voltage_max'],electrical['all_bus_voltage']['maximum'],atol=1e-13)
                self.require(row['voltage_outside_095_105_cells']==electrical['cells_below_095']+electrical['cells_above_105'],'Reported voltage violation count differs')
                record.update(audit_status='passed',requested=requested,delivered=delivered,requested_violations=rv,
                    delivered_violations=av,planned_cost=bill['planned'],settled_cost=bill['settled'],evaluation=evaluation,
                    electrical=electrical,branches=branch_values,buses=bus_values)
            except Exception as exc:record.update(audit_status='failed',error=repr(exc),traceback=traceback.format_exc())
            self.records.append(record)

    def verify_reported_evidence(self, summary, current, zero, forecast, forecast_zero, exact, exact_zero):
        def number(data, name, i, j):
            where=[k for k,pair in enumerate(data['pairs']) if set(map(int,pair))=={i,j}]
            self.require(len(where)==1,'Evidence pair missing/duplicated')
            k=where[0];value=float(data[name][k])
            if i<j or name.startswith('J'):return value
            if name=='bias':return -value-2*float(data['J'][k])
            return -value
        for pair in summary['target_pairs']:
            i,j=pair['target'],pair['competitor']
            for field,name in [('actual_physical_J_gain','J'),('actual_bias_change','bias'),
                    ('actual_meter_change','meter'),('actual_logodds_change','observed_logodds'),
                    ('same_observations_logodds_change_without_common','observed_logodds_without_common')]:
                self.same('reported_'+field,pair[field],number(current,name,i,j)-number(zero,name,i,j),atol=1e-8)
            if forecast is not None:
                self.same('reported_forecast_J_gain',pair['forecast_J_gain'],number(forecast,'J',i,j)-number(forecast_zero,'J',i,j),atol=1e-8)
            if exact is not None:
                self.same('reported_ideal_J_gain',pair['exact_physical_J_gain'],number(exact,'J',i,j)-number(exact_zero,'J',i,j),atol=1e-8)
        nearest=summary['nearest_competitor']
        self.require(nearest in summary['target_pairs'],'Nearest-competitor summary differs from target pair')


def action_hash(value):
    value=np.ascontiguousarray(value,dtype=np.float64)
    digest=hashlib.sha256(str(value.shape).encode());digest.update(value.tobytes());return digest.hexdigest()


def planning_boundary(path):
    tree=ast.parse(Path(path).read_text(encoding='utf-8-sig'))
    calls=[node for node in ast.walk(tree) if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute)
        and isinstance(node.func.value,ast.Name) and node.func.value.id=='solver' and node.func.attr=='plan']
    narrow=[node.value for node in ast.walk(tree) if isinstance(node,ast.Assign) and
        any(isinstance(target,ast.Name) and target.id=='planning' for target in node.targets)]
    fields=[]
    for node in narrow:
        if isinstance(node,ast.DictComp) and len(node.generators)==1:
            gen=node.generators[0]
            if (not gen.ifs and isinstance(gen.iter,(ast.Tuple,ast.List)) and isinstance(gen.target,ast.Name)
                and isinstance(node.key,ast.Name) and node.key.id==gen.target.id
                and isinstance(node.value,ast.Subscript) and isinstance(node.value.value,ast.Name)
                and node.value.value.id=='ctx' and isinstance(node.value.slice,ast.Name) and node.value.slice.id==gen.target.id):
                fields.append([item.value if isinstance(item,ast.Constant) else '<dynamic>' for item in gen.iter.elts])
    args=[ast.unparse(value) for value in calls[0].args] if len(calls)==1 else []
    expected=['case',"planning['score']","planning['val']",'prediction',"planning['base']"]
    passed=len(fields)==1 and set(fields[0])=={'score','val','base'} and args==expected
    return dict(passed=passed,source=base.ref(path),planning_keys=fields,solver_positional_inputs=args,
        scope='Static sealed adapter allowlist; quote values are historical score/validation, noisy P/Q forecast, contracts and sealed same-forecast seeds/anchor. Truth is used only in the post-seal evaluator.')


def run(source,original_audit,out):
    source,out=Path(source).resolve(),Path(out).resolve()
    if out.exists():raise FileExistsError('Preserve previous audit '+str(out))
    if not(source/'complete.json').exists():raise ValueError('Completed additional run required')
    original=SealedFiles(original_audit)
    if original.report['status']!='passed':raise ValueError('Original full comparison audit must pass first')
    out.mkdir(parents=True);shutil.copyfile(__file__,out/'audit.py');self_test()
    audit=Audit(original);p=base.read(audit.file(source/'protocol.json'));entries=base.read(audit.checked(p['inputs']))
    audit.require(Path(p['source_run']).resolve()==original.source_root,'Wrong comparison batch')
    audit.require(p['source_protocol']['sha256']==base.ref(original.source_root/'protocol.json')['sha256'],'Original protocol reference differs')
    audit.require(p['policies']==[POLICY] and len(entries)==1,'Unexpected appended policies or resources')
    audit.require(entries[0]==next(e for e in original.entries if e['resource_variant']==ORIGINAL_VARIANT),'Frozen entry changed')
    complete=base.read(audit.file(source/'complete.json'));completion=base.read(audit.checked(complete['main']))
    audit.require(complete['source_unchanged'],'Source drift during execution')
    expected={(entry['id'],draw) for entry in entries for draw in range(p['forecasts'])}
    passed_tasks={(r['id'],r['forecast_index']) for r in completion['receipts']};failed_tasks={(r['id'],r['forecast_index']) for r in completion['failed_tasks']}
    audit.require(not(passed_tasks&failed_tasks) and passed_tasks|failed_tasks==expected,'Forecast completion identities differ')
    audit.require(len(passed_tasks)==len(completion['receipts']) and len(failed_tasks)==len(completion['failed_tasks']),'Duplicate task receipts')
    for item in p['sources']:
        target=source/'source_snapshot'/Path(item['path']).resolve().relative_to(base.native.ROOT)
        audit.file(target,item['sha256'])
    meta=source/'source_snapshot/topology_identification_standalone/cases'/entries[0]['case']
    metadata=base.read(audit.file(meta/'metadata.json'))
    with audit.file(meta/'branches.csv').open(encoding='utf-8-sig',newline='') as f:branches=list(csv.DictReader(f))
    with audit.file(meta/'buses.csv').open(encoding='utf-8-sig',newline='') as f:buses=list(csv.DictReader(f))
    for name in ('metadata.json','branches.csv','buses.csv'):
        current=base.native.ROOT/'topology_identification_standalone/cases'/entries[0]['case']/name
        audit.require(base.ref(current)['sha256']==base.ref(meta/name)['sha256'],'Physical truth table changed since sealing')
    boundary=planning_boundary(source/'source_snapshot/rnj_wzzt_dr/experiments/flexibility_call_budget_20261114/run.py')
    audit.require(boundary['passed'],'New solver input boundary differs from the narrow historical/forecast interface')
    own=dict(created_utc=datetime.now(timezone.utc).isoformat(),source=base.ref(__file__),saved_source=base.ref(out/'audit.py'),
        original_audit=base.ref(original.audit_root/'report.json'),source_run=base.ref(source/'protocol.json'),
        scope='Stored-array independent math/fees/native selection/Ohm law/KCL only; no optimization, AC solve or refitting.',
        expected_rows=p['forecasts']*p['repeats'],bound_scope='Restricted activation-budget total-objective bounds only.',planning_boundary=boundary,
        environment=dict(python=sys.version,numpy=np.__version__),
        helper_sources=[base.ref(base.__file__),base.ref(base.native.__file__),base.ref(base.operation.__file__),
            base.ref(sys.modules[base.direct_original.__module__].__file__),base.ref(sys.modules[SealedFiles.__module__].__file__)])
    for item in own['helper_sources']:
        path=Path(item['path']);saved=out/'helper_snapshots'/path.relative_to(base.native.ROOT)
        saved.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,saved)
    base.write(out/'protocol.json',own)
    for entry in entries:
        for draw in range(p['forecasts']):
            folder=source/'fraction0.10'/entry['id']/f'forecast{draw:02d}'
            try:
                if (entry['id'],draw) in failed_tasks:raise ValueError('Original appended task failed')
                audit.forecast(folder,entry,p,metadata,branches,buses)
            except Exception as exc:audit.task_errors.append(dict(path=str(folder),error=repr(exc),traceback=traceback.format_exc()))
            base.write(out/'progress.json',dict(last_forecast=str(folder),rows=len(audit.records),task_errors=len(audit.task_errors),row_errors=sum(r['audit_status']!='passed' for r in audit.records)))
    status='passed' if len(audit.records)==own['expected_rows'] and not audit.task_errors and all(r['audit_status']=='passed' for r in audit.records) else 'failed'
    result=dict(status=status,protocol=base.ref(out/'protocol.json'),expected_rows=own['expected_rows'],audited_rows=len(audit.records),
        original_failed_tasks=completion['failed_tasks'],original_quote_exceptions=completion['quote_exceptions'],
        original_execution_failed_rows=completion['execution_failed_rows'],exact_diagnostic_failures=audit.exact_failures,
        task_errors=audit.task_errors,counts=dict(audit.counts),maximum_absolute_errors=dict(audit.maxima),electrical_maximum_errors=dict(audit.operation.maxima),
        summary=base.summarize(audit.records),comparisons=compare_original(audit.records,original.report),quotes=audit.quote_records,records=audit.records)
    base.write(out/'report.json',result)
    all_inputs={path:dict(path=path,sha256=digest) for path,digest in audit.hashes.items()}
    all_inputs.update(original.inputs)
    base.write(out/'inputs_manifest.json',list(all_inputs.values()))
    base.write(out/'complete.json',dict(status=status,report=base.ref(out/'report.json'),inputs=base.ref(out/'inputs_manifest.json'),
        source_unchanged=base.ref(__file__)['sha256']==own['source']['sha256']))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',type=Path);parser.add_argument('--source-audit',type=Path)
    parser.add_argument('--out',type=Path);parser.add_argument('--self-test',action='store_true');args=parser.parse_args()
    if args.self_test:print(json.dumps(self_test()))
    else:
        if any(value is None for value in (args.run,args.source_audit,args.out)):parser.error('--run, --source-audit and --out required')
        result=run(args.run,args.source_audit,args.out)
        print(json.dumps(dict(status=result['status'],rows=result['audited_rows'],expected=result['expected_rows'],task_errors=len(result['task_errors']),row_errors=sum(r['audit_status']!='passed' for r in result['records']))))
        raise SystemExit(0 if result['status']=='passed' else 1)
