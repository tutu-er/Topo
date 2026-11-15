"""Independent single-history-winner audit, using saved arrays only.

No optimizer, resource domain, production evidence/scorer or AC solver is
called. Previously audited multi-target rows are referenced, not rerun.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import csv
from datetime import datetime,timezone
from itertools import combinations
import json
from pathlib import Path
import shutil
import sys
import traceback

import numpy as np

from ..distribution_panel_20261115 import audit as panel
from ..frozen_rx_guard_20261115 import audit as previous

base,prob,ROOT=panel.base,panel.prob,panel.ROOT


def policy_names(protocol):
    modes=protocol['modes']
    if len(set(modes))!=len(modes) or not modes or not set(modes)<=set(panel.MODES):
        raise ValueError('Unknown/duplicated single-target modes')
    return ['single_'+mode for mode in modes]


def historical_target(model):
    """Use the exact frozen native winner, not truth or a same-family substitute."""
    native=int(model['score']['native_index'])
    score=model['score'];scaled=np.asarray(model['history'])/score['cells']/score['reference_mse']
    eligible=np.flatnonzero(scaled<=scaled.min()+score['validation_tie'])
    selected=min(eligible,key=lambda i:(model['internal'][i],scaled[i],i))
    if native!=selected:raise ValueError('Frozen target differs from independent historical MSE/tie winner')
    representatives=list(model['representatives'])
    if native not in representatives:
        raise ValueError('Historical native winner lacks an identical representative parameter point')
    target=representatives.index(native)
    selected=dict(model,targets=[target]);selected.pop('_integral_cache',None)
    return selected


def exact_gates(model,forecast,power,mode):
    """All foreign native likelihood constraints, forward scalar guards only."""
    if len(model['targets'])!=1 or model['representatives'][model['targets'][0]]!=model['score']['native_index']:
        raise ValueError('Expected exactly the strict historical winner')
    value=panel.exact_gates(model,forecast,power,mode)
    target=model['targets'][0]
    value['guards']=[g for g in value['guards'] if g['target']==target]
    value['minimum_guard']=min((g['gain'] for g in value['guards']),default=None)
    value['feasible']=(all(r['margin']>=-1e-7 and r['pair_failure']<=r['alpha']+1e-9 for r in value['rows'])
        and all(g['gain']>=-1e-7 for g in value['guards']))
    value['probability_scope']='Conditional on the unchanged historical K of the strict historical native winner; every foreign native parameter point remains a competitor. Forward guards only; no oracle/PQ/W action budget.'
    return value


class Audit(panel.Audit):
    def __init__(self):
        super().__init__();self.baseline_records=[];self.source_snapshots={};self.post_execution_source_changes=[]

    def checked(self,reference):
        original=str(Path(reference['path']).resolve())
        snapshot=self.source_snapshots.get((original,reference['sha256']))
        if snapshot is not None:
            return super().checked(dict(path=str(snapshot),sha256=reference['sha256']))
        return super().checked(reference)

    def gates(self,model,forecast,power,mode,saved=None):
        value=exact_gates(model,forecast,power,mode)
        if saved is not None and saved.get('status') not in ('evidence_construction_failed','evidence_check_exception'):
            for key,expected in dict(margins=[r['margin'] for r in value['rows']],
                pair_failure=[r['pair_failure'] for r in value['rows']],
                mu=[r['fixed_input_mean']-value['tie'] for r in value['rows']],
                variance=[r['fixed_input_variance'] for r in value['rows']],
                guard_gains=[g['gain'] for g in value['guards']]).items():
                self.same('single_gate_'+mode+'_'+key,saved[key],expected,atol=1e-8,rtol=1e-9)
            self.require(bool(saved['feasible'])==value['feasible'],'Single-target gate flag differs')
        return value

    def single_quote(self,entry,name,quote,action,direct,bill,seeds,seed_check):
        self.references(quote);mode=name.removeprefix('single_')
        self.require(quote['mode']==mode,'Single-target quote mode differs')
        self.require(quote['target_native']==entry['planned_history_target_native'],'Reported historical target differs')
        accepted=bool(quote['new_gate_feasible'])
        self.require(accepted==bool(quote['feasible']),'Accepted quote/engine feasibility differs')
        if accepted:
            self.require(direct['feasible'] and not quote['zero_fallback'],'Accepted quote violates single-target gate or is fallback')
            self.same('quote_current_activation_cost',quote['cost_yuan'],bill['activation_yuan'],atol=1e-7)
        else:
            self.require(quote['zero_fallback'],'Failed quote lacks zero fallback')
            self.same('failed_quote_executes_zero',action,np.zeros_like(action),atol=0,rtol=0)
        if quote.get('executed_probability_check') is not None:
            self.require(bool(quote['executed_probability_check']['feasible'])==direct['feasible'],'Executed zero/accepted probability flag differs')
        if quote['status']!='exception':
            self.require(len(quote['gate_rows'])==len(direct['rows']),'Missing/extra foreign native constraint')
            for saved,expected in zip(quote['gate_rows'],direct['rows'],strict=True):
                self.require((saved['target_rep'],saved['target_native'],saved['competitor_native'])==
                    (expected['target'],expected['target_native'],expected['competitor_native']),'Historical target/full native mapping differs')
                self.same('risk_allocation',saved['alpha'],expected['alpha'],atol=0,rtol=0)
                if mode!='normal_fixed':
                    self.same('unchanged_lambda',saved['lambda_selected'],expected['lambda_selected'],atol=0,rtol=0)
                    self.require(saved['lambda_halvings']==expected['lambda_halvings'],'Lambda selection rule differs')
        initial=quote.get('initial') or {}
        if initial.get('feasible'):
            self.require(initial['name'] in seeds,'Initial outside the original common portfolio')
            gate,fee=seed_check(initial['name'],mode)
            self.require(gate['feasible'],'Claimed feasible initial violates direct single-target equations')
            self.same('initial_cost',initial['cost_yuan'],fee,atol=1e-7)
            if accepted:self.require(bill['activation_yuan']<=fee+1e-6,'Returned quote costs more than original feasible initial')
        margin=min((r['margin']-1e-8 for r in direct['rows']),default=None)
        buffer_ok=all(r['margin']>=1e-8-1e-9 for r in direct['rows']) and all(g['gain']>=-1e-9 for g in direct['guards'])
        certificate=previous.cost_certificate(quote,bill['activation_yuan'],direct['feasible'],buffer_ok)
        folder=Path(self.where)/(name+'_diagnostics');log=folder/'native.log';trace=folder/'progress.jsonl'
        history=previous.native_bound_history(self.file(log) if log.exists() else None,
            self.file(trace) if trace.exists() else None,bill['activation_yuan'] if accepted else None,accepted and buffer_ok)
        certificate.update(native_bound_history=history,minimum_original_probability_buffer_margin=margin,
            planner_reported_lower_bound_yuan=quote.get('lower_bound_yuan'),planner_reported_relative_gap=quote.get('relative_gap'),
            audit_withdrawal_due_to_bound_trajectory=history['recorded_bound_conflict_with_verified_buffered_point'])
        if history['recorded_bound_conflict_with_verified_buffered_point']:
            certificate.update(lower_bound_yuan=0.,relative_gap=1. if bill['activation_yuan']>0 else 0.,
                lower_bound_withdrawn=True,audit_global_optimality_supported=False)
        else:certificate['audit_global_optimality_supported']=False
        keys=('status','native_status','source','optimizer_invoked','initial','native_attributes','seconds',
              'quote_wall_seconds','requested_quote_budget_seconds','shared_preparation_seconds','charged_total_seconds',
              'quote_budget_exceeded','infeasibility_proved','lower_bound_withdrawal_reason')
        self.quote_records.append(dict(**panel.identity(entry),forecast=0,policy=name,base_mode=mode,
            **{k:quote.get(k) for k in keys},new_gate_feasible=accepted,zero_fallback=quote['zero_fallback'],
            executed_probability_feasible=direct['feasible'],direct_gate=direct,certificate=certificate,planned_cost=bill,
            nonzero_executed=bool(np.max(abs(action),initial=0)>1e-8)))

    def case(self, folder, entry, protocol, metadata, branches, buses):
        self.where=str(folder);ctx=self.context(entry);self.references(entry)
        seal=base.read(self.file(folder/'seal.json'));self.references(seal)
        self.require(seal['score']==entry['score'] and seal['validation']==entry['files']['validation.npz'],
                     'Frozen score or validation reference changed')
        prior=Path(entry['prior_directory']);oldseal=base.read(self.checked(entry['prior_seal']))
        self.require(oldseal['score']==entry['score'],'Old and new score source differs')
        source_planner=base.native.arrays(self.checked(oldseal['planner']))
        planner=base.native.arrays(self.checked(seal['planner']));forecast=planner['forecast']
        self.same('old_forecast_preserved',forecast,source_planner['forecast'],atol=0,rtol=0)
        names=policies=policy_names(protocol)
        self.require(set(planner)==set(policies+['forecast']),'Missing or extra new action')
        config=base.read(self.checked(seal['resource_case']))
        self.require(config==base.read(self.checked(oldseal['resource_case'])),'Original physical offers changed')
        pc=self.catalogue(entry,config,ctx);qc=len(config['resources'])-pc
        self.same('explicit_base',entry['base_kw'],ctx['base'],atol=0,rtol=0)
        noise=panel.noise_parameters(entry,protocol)
        expected=np.c_[ctx['baseline_P'],ctx['baseline_Q']]
        expected=expected*(1+noise['forecast_relative_sd']*base.native.random_stream(protocol,entry,0,0,991).normal(size=expected.shape))
        self.same('paired_forecast_random_stream',forecast,expected,atol=0,rtol=0)
        model=prob.build_model(ctx['score'],base.native.arrays(self.checked(entry['files']['validation.npz'])),ctx['native'],
            meter_relative_sd=noise['meter_relative_sd'],forecast_relative_sd=noise['forecast_relative_sd'])
        model=historical_target(model)
        self.require(entry['planned_history_target_native']==ctx['score']['native_index'], 'Planning entry target differs from historical winner')
        self.input_records.append(self.coverage(entry,ctx,model))
        seeds=base.native.arrays(self.checked(entry['initial_points']))
        saved_seeds=base.native.arrays(self.file(folder/'shared_initial_points.npz'))
        self.require(set(seeds)==set(saved_seeds),'Common seed names changed')
        for key in seeds:self.same('common_seed_'+key,saved_seeds[key],seeds[key],atol=0,rtol=0)
        plans=base.read(self.checked(seal['plans']));self.require(set(plans)==set(names),'Incomplete quote set')
        saved_audits=base.read(self.checked(seal['prediction_audits']))
        kp,kq=float(protocol['reservation_prices']['P']),float(protocol['reservation_prices']['Q'])
        actions, forecasts, gates={}, {}, {}
        for name in policies:
            u=planner[name]
            requested,violations,dp,dq=panel.resource_values(config,u,ctx['labels'],forecast[:,:ctx['n']],ctx['base'])
            self.same('independent_nodal_map',prob.shift(config,u,ctx['labels'],ctx['base']),np.c_[dp,dq],atol=1e-14)
            bill=base.procurement(config,u,u,kp,kq)['planned']
            actions[name]=dict(requested=requested,violations=violations,bill=bill)
            forecasts[name]=self.evaluate(ctx,forecast+np.c_[dp,dq])
            for mode in protocol['modes']:
                gates[mode,name]=self.gates(model,forecast,forecast+np.c_[dp,dq],mode,saved_audits[mode][name])
        cache={}
        def seed_check(name,mode):
            key=name,mode
            if key not in cache:
                u=seeds[name]
                _,_,dp,dq=panel.resource_values(config,u,ctx['labels'],forecast[:,:ctx['n']],ctx['base'])
                cache[key]=(self.gates(model,forecast,forecast+np.c_[dp,dq],mode),
                    base.procurement(config,u,u,kp,kq)['planned']['activation_yuan'])
            return cache[key]
        for name in names:
            self.require(plans[name]['initial_points']==entry['initial_points'],'Mode-specific seed source changed')
            saved=base.native.arrays(self.file(folder/(name+'_action.npz')))
            self.same('per_quote_frozen_action',saved['action'],planner[name],atol=0,rtol=0)
            self.require(base.read(self.file(folder/(name+'_plan.json')))==plans[name],'Per-mode quote changed after planning')
            mode=name.removeprefix('single_')
            self.single_quote(entry,name,plans[name],planner[name],gates[mode,name],
                              actions[name]['bill'],seeds,seed_check)
        done=base.read(self.file(folder/'complete.json'));self.references(done)
        rows=base.read(self.checked(done['results']))
        expected={(repeat,name) for repeat in range(protocol['repeats']) for name in policies}
        self.require(len(rows)==len(expected) and {(r['repeat'],r['strategy']) for r in rows}==expected,
                     'Execution denominator or identities differ')
        zero_forecast=self.evaluate(ctx,forecast)
        zero_by_repeat={}
        for row in rows:
            repeat,name=row['repeat'],row['strategy'];path=folder/f'execution{repeat:02d}'/name
            rec=dict(**panel.identity(entry),forecast=0,repeat=repeat,strategy=name,source=str(path),original_status=row['status'])
            try:
                self.require(base.read(self.file(path/'receipt.json'))==row,'Execution receipt differs from result ledger')
                if row['status']!='complete':
                    rec.update(audit_status='recorded_failure',error=row.get('error'),evaluation={'native_correct':False})
                    self.records.append(rec);continue
                self.references(row)
                parent=name.removesuffix('_rotation');quote=plans[parent]
                self.require(row['parent_strategy']==parent and row['mode']==quote['mode'],
                             'Executed policy identity differs')
                self.require(row['quote_zero_fallback']==quote['zero_fallback'] and
                             row['quote_new_constraints_feasible']==quote['new_gate_feasible'],'Executed quote flags differ')
                random=base.native.arrays(self.file(folder/f'execution{repeat:02d}'/'paired_randomness.npz'))
                original_random=base.native.arrays(self.file(prior/f'execution{repeat:02d}'/'paired_randomness.npz'))
                self.require(set(random)==set(original_random),'Paired random variables changed')
                for key in random:self.same('original_paired_'+key,random[key],original_random[key],atol=0,rtol=0)
                eta=np.r_[base.native.random_stream(protocol,entry,0,repeat,773).uniform(.9,1,pc),
                    base.native.random_stream(protocol,entry,0,repeat,774).uniform(.9,1,qc)]
                self.same('response_stream',random['eta'],eta,atol=0,rtol=0)
                shapes=(ctx['baseline_P'].shape,ctx['baseline_Q'].shape,ctx['baseline_P'].shape,ctx['baseline_root'].shape)
                for k,shape in enumerate(shapes):
                    self.same('noise_stream',random['noise'+str(k)],base.native.random_stream(protocol,entry,0,repeat,k+2).normal(size=shape),atol=0,rtol=0)
                obs=base.native.arrays(self.file(path/'observations.npz'));u=planner[name]
                self.same('sealed_request',obs['requested_action'],u,atol=0,rtol=0)
                delivery=u*eta[None]
                for col,resource in enumerate(config['resources'][:pc]):
                    if resource['kind']=='pv':
                        node=ctx['labels'].index(int(resource['node_label']))
                        cap=np.minimum(resource['contract_kw'],np.maximum(0,-ctx['baseline_P'][:,node]*ctx['base']).reshape(24,4).min(1))
                        delivery[:,col]=np.minimum(delivery[:,col],cap)
                self.same('paired_delivery',obs['delivered_action'],delivery,atol=1e-12,rtol=0)
                delivered,violations,dp,dq=panel.resource_values(config,delivery,ctx['labels'],forecast[:,:ctx['n']],ctx['base'])
                self.same('true_P',obs['true_P'],ctx['baseline_P']+dp,atol=1e-13)
                self.same('true_Q',obs['true_Q'],ctx['baseline_Q']+dq,atol=1e-13)
                for label,k,sd in (('P',0,noise['pq_relative_sd']),('Q',1,noise['pq_relative_sd']),
                                   ('V',2,noise['meter_relative_sd']),('root',3,noise['meter_relative_sd'])):
                    self.same('meter_'+label,obs[label],obs['true_'+label]*(1+sd*random['noise'+str(k)]),atol=1e-13)
                self.same('observed_Y',obs['Y'],obs['root'][:,None]**2-obs['V']**2,atol=1e-13)
                if 'reused_action_source' in row:
                    cached=base.native.arrays(self.checked(row['reused_action_source']))
                    for key in obs:self.same('cached_action_'+key,obs[key],cached[key],atol=0,rtol=0)
                bill=base.procurement(config,u,delivery,kp,kq)
                for when in ('planned','settled'):
                    self.costs(row[when+'_cost'],bill[when],when)
                    self.same(when+'_fee',row[when+'_fee_yuan'],bill[when]['total_yuan'],atol=1e-8)
                for field,value in (('requested_quantities',u),('delivered_quantities',delivery)):
                    self.same(field+'_P',row[field]['active_abs_kwh'],abs(value[:,:pc]).sum(),atol=1e-9)
                    self.same(field+'_Q',row[field]['reactive_abs_kvarh'],abs(value[:,pc:]).sum(),atol=1e-9)
                evaluation,values=self.score(ctx,obs,row)
                evaluation['full_native_diagnostic']=previous.full_native_diagnostic(ctx,values)
                self.pair_file(self.file(path/'pair_evidence.npz'),ctx,values)
                if repeat not in zero_by_repeat:
                    oldfolder=prior/f'execution{repeat:02d}'/'no_dfs'
                    zero_obs=base.native.arrays(self.file(oldfolder/'observations.npz'))
                    zero_row=base.read(self.file(oldfolder/'receipt.json'))
                    zero_evaluation,zero_values=self.score(ctx,zero_obs,zero_row)
                    self.pair_file(self.file(oldfolder/'pair_evidence.npz'),ctx,zero_values)
                    zero_by_repeat[repeat]=(zero_obs,zero_evaluation,zero_values)
                zero_obs,zero_evaluation,zero_values=zero_by_repeat[repeat]
                self.reported_evidence(ctx,row['evidence'],values,zero_values,forecasts[name],zero_forecast)
                change=base.native.arrays(self.file(path/'pair_changes.npz'))
                pair_indices=list(combinations(range(len(ctx['H'])),2))
                for key in ('J','J_without_common','J_common','J_cross','bias','meter',
                            'observed_logodds','observed_logodds_without_common'):
                    differences=[self.directed(ctx,values,i,j)[key]-self.directed(ctx,zero_values,i,j)[key]
                                 for i,j in pair_indices]
                    self.same('canonical_pair_change_'+key,change[key],differences,atol=1e-8)
                if 'pair' in evaluation:
                    i,j=evaluation['pair'];evaluation['forecast_evidence']=self.directed(ctx,forecasts[name],i,j)
                    zero_pair=self.directed(ctx,zero_values,i,j)
                    evaluation['evidence_increment']={k:evaluation['evidence'][k]-zero_pair[k] for k in
                        ('J','bias','meter','clean_logodds','observed_logodds')}
                delta_power=np.c_[obs['true_P']-zero_obs['true_P'],obs['true_Q']-zero_obs['true_Q']]
                delta_y=(obs['true_root'][:,None]**2-obs['true_V']**2)-(zero_obs['true_root'][:,None]**2-zero_obs['true_V']**2)
                K_prediction=np.einsum('tc,knc->ktn',delta_power,model['means'])
                H_prediction=np.einsum('tc,knc->ktn',delta_power,model['H'][model['representatives']])
                errorK=np.square(delta_y[None]-K_prediction).mean(axis=(1,2))/model['sigma2']
                errorH=np.square(delta_y[None]-H_prediction).mean(axis=(1,2))/model['sigma2']
                self.same('actual_K_increment_error',row['actual_increment_response_error_K_over_sigma2_by_representative'],errorK,atol=1e-9)
                self.same('actual_H_increment_error',row['actual_increment_response_error_H_over_sigma2_by_representative'],errorH,atol=1e-9)
                physical=row['physical_state'];ac=base.native.arrays(self.checked(physical))
                if 'time_slice' in physical:
                    start,end=physical['time_slice'];length=len(ac['voltage_complex_pu'])
                    ac={k:v[start:end] if v.ndim and v.shape[0]==length else v for k,v in ac.items()}
                electrical,branch_values,bus_values=base.operation.electrical(self.operation,ac,obs,metadata,branches,buses)
                voltage_ok=electrical['cells_below_095']+electrical['cells_above_105']==0
                rec.update(audit_status='passed',requested=actions[name]['requested'],delivered=delivered,
                    requested_violations=actions[name]['violations'],delivered_violations=violations,
                    planned_cost=bill['planned'],settled_cost=bill['settled'],evaluation=evaluation,
                    electrical=electrical,branches=branch_values,buses=bus_values,voltage_screen_passed=voltage_ok,
                    recovered_and_voltage_screen_passed=bool(evaluation['native_correct'] and voltage_ok),
                    thermal_rating_status='unavailable_not_verified',quote_new_gate_feasible=quote['new_gate_feasible'],
                    quote_zero_fallback=quote['zero_fallback'],base_mode=quote['mode'],
                    is_rotation=False,
                    actual_H_increment_error_over_sigma2=errorH.tolist(),actual_K_increment_error_over_sigma2=errorK.tolist(),
                    
                    reused_action_source=row.get('reused_action_source'),baseline_source=row['baseline_source'])
            except Exception as exc:
                rec.update(audit_status='failed',error=repr(exc),traceback=traceback.format_exc())
            self.records.append(rec)


def planning_boundary(path):
    tree=ast.parse(Path(path).read_text(encoding='utf-8-sig'))
    runner=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run_case')
    calls=[n for n in ast.walk(runner) if isinstance(n,ast.Call)]
    solves=[n for n in calls if ast.unparse(n.func)=='solver.plan']
    seals=[n for n in calls if ast.unparse(n.func)=='io.write' and n.args and isinstance(n.args[0],ast.BinOp)
        and isinstance(n.args[0].right,ast.Constant) and n.args[0].right.value=='seal.json']
    executes=[n for n in calls if ast.unparse(n.func)=='paired.execute']
    forbidden=[n for n in calls if ast.unparse(n.func) in ('old.context','paired.old.context','physical_response')]
    expected=['case',"planning['score']","planning['val']",'forecast',"planning['base']"]
    passed=(len(solves)==len(seals)==len(executes)==1 and not forbidden
        and solves[0].lineno<seals[0].lineno<executes[0].lineno
        and [ast.unparse(a) for a in solves[0].args]==expected
        and [ast.unparse(a) for a in executes[0].args[-2:]]==['{}','dict(guards=[])'])
    return dict(passed=passed,source=base.ref(path),scope='Single-target quote receives only original contracts, history, forecast and base; original AC adapter is invoked after sealing with no added PQ/W/oracle matrices.')


def run(source,out):
    source,out=Path(source).resolve(),Path(out).resolve()
    if out.exists():raise FileExistsError(out)
    if not (source/'complete.json').exists():raise ValueError('Completed run required')
    out.mkdir(parents=True);shutil.copyfile(__file__,out/'audit.py')
    audit=Audit();p=base.read(audit.file(source/'protocol.json'))
    entries=base.read(audit.checked(p['inputs']));complete=base.read(audit.file(source/'complete.json'))
    audit.require(complete['source_unchanged'] and p['oracle'] is False,'Changed source or unexpected oracle scope')
    policies=policy_names(p);audit.require(p['policies']==policies,'Executed policy list differs')
    for item in p['sources']:
        relative=Path(item['path']).resolve().relative_to(ROOT)
        snapshot=source/'source_snapshot'/relative
        audit.file(snapshot,item['sha256'])
        audit.source_snapshots[(str(Path(item['path']).resolve()),item['sha256'])]=snapshot
        current=base.ref(item['path'])
        if current['sha256']!=item['sha256']:
            audit.post_execution_source_changes.append(dict(executed_source=item,snapshot=base.ref(snapshot),current_source=current,
                policy='Executed snapshot hash verified and used for all sealed source references; current code is not substituted into prior evidence.'))
    driver=source/'source_snapshot/rnj_wzzt_dr/experiments/single_target_20261115/run.py'
    boundary=planning_boundary(driver);audit.require(boundary['passed'],'Planning information boundary differs')
    original=base.read(audit.checked(p['source_protocol']))
    audit.require(p['seed']==original['seed'] and p['forecasts']==1,'Original paired random stream changed')
    source_audit=base.read(audit.checked(p['baseline_audit']))
    audit.require(source_audit['status']=='passed' and source_audit['audited_rows']==4608,'Original grid AC audit not complete')
    original_entries={e['id']:e for e in base.read(audit.checked(original['inputs']))}
    for entry in entries:
        old=original_entries[entry['id']]
        for key in ('score','native_copy','files','physical_cache','p_resources','q_resources','index','base_kw','meter'):
            audit.require(entry[key]==old[key],'Frozen input differs: '+key)
        audit.require(not any(k.startswith('oracle') for k in entry),'Oracle target information leaked into new entry')
    selected={e['id'] for e in entries}
    with audit.checked(source_audit['records']).open(encoding='utf8') as stream:
        for line in stream:
            row=json.loads(line)
            if row['id'] in selected and row['repeat']<p['repeats']:audit.baseline_records.append(row)
    baseline_expected=len(entries)*p['repeats']*len(panel.POLICIES)
    audit.require(len(audit.baseline_records)==baseline_expected==p['expected_reused_baseline_rows'],'Referenced baseline denominator differs')
    audit.require(all(r['audit_status']=='passed' for r in audit.baseline_records),'Referenced baseline is not independently audited')
    expected_rows=len(entries)*len(policies)*p['repeats']
    audit.require(p['expected_rows']==expected_rows and p['expected_quotes']==len(entries)*len(policies),'Protocol denominator differs')
    success={r['id'] for r in complete.get('receipts',[])};failures={r['id'] for r in complete.get('failed_tasks',[])}
    audit.require(not(success&failures) and success|failures==selected,'Missing or duplicate task identity')
    own=dict(created_utc=datetime.now(timezone.utc).isoformat(),source=base.ref(__file__),source_protocol=base.ref(source/'protocol.json'),
        planning_boundary=boundary,expected_new_execution_rows=expected_rows,referenced_baseline_rows=baseline_expected,
        scope='Saved-array independent probability formulas, forward-only history target, original paired AC/PQV observations, native scoring, resources, fees and electrical balance. No new optimization or AC.',
        environment=dict(python=sys.version,numpy=np.__version__),helpers=[base.ref(m.__file__) for m in (panel,previous,base,base.native,base.operation,prob)])
    base.write(out/'protocol.json',own)
    for entry in entries:
        if entry['id'] in failures:
            for repeat in range(p['repeats']):
                for name in policies:
                    audit.records.append(dict(**panel.identity(entry),forecast=0,repeat=repeat,strategy=name,
                        audit_status='recorded_failure',original_status='task_failed',evaluation={'native_correct':False}))
            continue
        try:
            casepath=source/'source_snapshot/topology_identification_standalone/cases'/entry['case']
            metadata=base.read(audit.file(casepath/'metadata.json'))
            with audit.file(casepath/'branches.csv').open(encoding='utf-8-sig',newline='') as stream:branches=list(csv.DictReader(stream))
            with audit.file(casepath/'buses.csv').open(encoding='utf-8-sig',newline='') as stream:buses=list(csv.DictReader(stream))
            for file in ('metadata.json','branches.csv','buses.csv'):
                audit.require(base.ref(casepath/file)['sha256']==base.ref(ROOT/'topology_identification_standalone/cases'/entry['case']/file)['sha256'],'Physical table changed')
            audit.case(source/'cases'/entry['id'],entry,p,metadata,branches,buses)
        except Exception as exc:audit.task_errors.append(dict(id=entry['id'],error=repr(exc),traceback=traceback.format_exc()))
        base.write(out/'progress.json',dict(rows=len(audit.records),expected=expected_rows,task_errors=len(audit.task_errors),row_errors=sum(r['audit_status']=='failed' for r in audit.records)))
    for path,sha in audit.hashes.items():
        if base.ref(path)['sha256']!=sha:audit.task_errors.append(dict(path=path,error='Input changed during audit'))
    passed=len(audit.records)==expected_rows and not audit.task_errors and all(r['audit_status'] in ('passed','recorded_failure') for r in audit.records)
    for name,rows in (('records',audit.records),('quotes',audit.quote_records),('baseline_records',audit.baseline_records)):
        with (out/(name+'.jsonl')).open('w',encoding='utf8') as stream:
            for row in rows:panel.json_line(stream,row)
    result=dict(status='passed' if passed else 'failed',protocol=base.ref(out/'protocol.json'),source_protocol=base.ref(source/'protocol.json'),source_inputs=p['inputs'],
        expected_rows=expected_rows,audited_rows=len(audit.records),baseline_rows=len(audit.baseline_records),
        audit_status_counts=dict(Counter(r['audit_status'] for r in audit.records)),input_coverage=audit.input_records,task_errors=audit.task_errors,
        original_failed_tasks=complete.get('failed_tasks',[]),original_quote_exceptions=complete.get('quote_exceptions',0),original_execution_failed_rows=complete.get('failed_rows',0),
        post_execution_source_changes=audit.post_execution_source_changes,
        records=base.ref(out/'records.jsonl'),quotes=base.ref(out/'quotes.jsonl'),baseline_records=base.ref(out/'baseline_records.jsonl'),
        maximum_numeric_errors=dict(audit.maxima),maximum_electrical_errors=dict(audit.operation.maxima),counts=dict(audit.counts),
        evidence_scope='History-selected single-target Gaussian sufficient gates and forward guards. AC recovery remains a separate paired observation; old 384 baseline rows are reused, not new independent evidence.',
        thermal_scope='Saved current/root power reconstructed; equipment thermal ratings unavailable.')
    base.write(out/'report.json',result);base.write(out/'complete.json',dict(status=result['status'],report=base.ref(out/'report.json')))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();result=run(args.run,args.out)
    print({k:result[k] for k in ('status','expected_rows','audited_rows','baseline_rows','task_errors')})
    if result['status']!='passed':raise SystemExit(1)


if __name__=='__main__':main()

