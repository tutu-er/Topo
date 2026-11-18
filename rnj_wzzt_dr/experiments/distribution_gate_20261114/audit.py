"""Independent Gaussian-gate, frozen-score, payment and stored-AC replay audit."""
from __future__ import annotations
import argparse
import ast
import csv
from datetime import datetime,timezone
from pathlib import Path
import shutil
import sys
import traceback
import numpy as np

from . import probability as prob
from ..flexibility_stability_20261114 import audit as base
from ..flexibility_stability_20261114.analyze import paired
from ..flexibility_call_budget_20261114.audit import Audit as BudgetAudit

POLICIES=prob.POLICIES;MODES=prob.MODES


class Audit(base.Audit):
    def gates(self,model,forecast,power,mode,saved=None):
        result=prob.exact_gates(model,forecast,power,mode)
        if saved is not None and saved.get('status') not in ('evidence_construction_failed','evidence_check_exception'):
            for field,values in (
                ('margins',[r['margin'] for r in result['rows']]),
                ('pair_failure',[r['pair_failure'] for r in result['rows']]),
                ('mu',[r['fixed_input_mean']-result['tie'] for r in result['rows']]),
                ('variance',[r['fixed_input_variance'] for r in result['rows']]),
                ('guard_gains',[g['gain'] for g in result['guards']])):
                self.same('gate_'+mode+'_'+field,saved[field],values,atol=1e-8,rtol=1e-9)
            self.require(saved['feasible']==result['feasible'],'New gate feasibility differs')
        return result

    def forecast(self,folder,entry,p,metadata,branches,buses):
        ctx=self.context(entry);draw=int(folder.name[8:]);seal=base.read(self.file(folder/'seal.json'))
        for key in ('planner','plans','score','protocol','validation','prediction_audits','reservation','anchor'):self.checked(seal[key])
        old=entry['replay_forecasts'][str(draw)]
        for key in ('seal','planner','initial_points','anchor','resource_case','anchor_receipt','shared_receipt','completion'):self.checked(old[key])
        previous=Path(old['directory']);prior_planner=base.native.arrays(self.checked(old['planner']))
        planner=base.native.arrays(self.checked(seal['planner']));forecast=planner['forecast']
        self.require(set(planner)=={'forecast',*POLICIES},'Policy identities differ')
        self.same('replayed_forecast',forecast,prior_planner['forecast'],atol=0,rtol=0)
        self.same('replayed_anchor',planner['anchor'],prior_planner['anchor'],atol=0,rtol=0)
        validation=base.native.arrays(self.checked(entry['files']['validation.npz']))
        model=prob.build_model(ctx['score'],validation,ctx['native'])
        config=base.read(self.file(folder/'resource_case.json'));self.require(config==base.read(self.checked(old['resource_case'])),'Physical domain changed')
        pc=self.catalogue(entry,config,ctx);qc=len(config['resources'])-pc
        seeds=base.native.arrays(self.file(folder/'shared_initial_points.npz'));prior_seeds=base.native.arrays(self.checked(old['initial_points']))
        self.require(set(seeds)==set(prior_seeds),'Common seed names changed')
        for key in seeds:self.same('common_seed_'+key,seeds[key],prior_seeds[key],atol=0,rtol=0)
        anchor=base.read(self.file(folder/'anchor_plan.json'));self.require(anchor==base.read(self.checked(old['anchor'])),'Old anchor receipt changed')
        kp,kq=float(p['reservation_prices']['P']),float(p['reservation_prices']['Q'])
        plans=base.read(self.checked(seal['plans']));self.require(set(plans)==set(MODES),'Quote identities differ')
        audits=base.read(self.checked(seal['prediction_audits']));reserves=base.native.arrays(self.checked(seal['reservation']))
        controls={};exact={};forecasts={};gates={}
        # Only anchor is tested against the old nominal absolute gate.
        anchor_direct=base.direct_original(dict(entry=entry,score=ctx['score'],config=config,planner=planner,directory=folder),planner['anchor'],anchor)
        self.require(anchor_direct['passed_original_acceptance'],'Replayed anchor failed its original constraints')
        for policy in POLICIES:
            u=planner[policy];devices,rv,dp,dq=base.operation.resource_values(config,u,ctx['labels'],forecast[:,:ctx['n']],ctx['base'])
            self.same('nominal_PQ_shift',prob.shift(config,u,ctx['labels'],ctx['base']),np.c_[dp,dq],atol=1e-14)
            self.same('reserved_peak',reserves[policy],np.max(abs(u),axis=0),atol=0,rtol=0)
            bill=base.procurement(config,u,u,kp,kq)['planned'];controls[policy]=dict(requested=devices,violations=rv,bill=bill)
            for mode in MODES:gates[mode,policy]=self.gates(model,forecast,forecast+np.c_[dp,dq],mode,audits[mode][policy])
            forecasts[policy]=self.evaluate(ctx,forecast+np.c_[dp,dq])
            ep=folder/'exact_requested'/policy;er=base.read(self.file(ep/'receipt.json'))
            if er['status']=='complete':
                clean=base.native.arrays(self.file(ep/'clean_observation.npz'))
                self.same('ideal_P',clean['P'],ctx['baseline_P']+dp,atol=1e-13);self.same('ideal_Q',clean['Q'],ctx['baseline_Q']+dq,atol=1e-13)
                self.same('ideal_Y',clean['Y'],clean['root'][:,None]**2-clean['V']**2,atol=1e-13)
                exact[policy]=self.evaluate(ctx,np.c_[clean['P'],clean['Q']],clean['Y'],np.c_[clean['P'],clean['Q']],clean['Y'])
                self.pair_file(self.file(ep/'pair_evidence.npz'),ctx,exact[policy]);self.pair_file(self.file(ep/'forecast_pair_evidence.npz'),ctx,forecasts[policy])
            else:self.exact_failures.append(dict(path=str(ep),status=er['status']))
        for mode,q in plans.items():
            direct=gates[mode,mode];feasible=bool(q['new_gate_feasible']);u=planner[mode]
            self.require(feasible==bool(q['feasible'])==bool(q['new_constraints_feasible']),'Quote feasibility flags differ')
            self.require(q['anchor_fallback']==(not feasible) and not q['zero_fallback'],'Fallback flags differ')
            if feasible:
                self.require(direct['feasible'],'Accepted new quote fails independent distribution gate')
                self.same('new_activation_objective',q['cost_yuan'],controls[mode]['bill']['activation_yuan'],atol=1e-7)
            else:self.same('explicit_old_anchor_fallback',u,planner['anchor'],atol=0,rtol=0)
            self.require(q['executed_gate_feasible']==direct['feasible'],'Executed-point gate flag differs')
            if q.get('status')!='exception':
                self.require(q['original_nominal_gates_retained'] is False and q['original_scalar_guards_retained'],'Gate scopes changed')
                self.require(q['bound_cost_object']=='activation_yuan','Wrong bound objective')
                self.require(len(q['gate_rows'])==len(direct['rows']),'Incomplete native competitors')
                for declared,independent in zip(q['gate_rows'],direct['rows'],strict=True):
                    self.require((declared['target_rep'],declared['target_native'],declared['competitor_native'])==
                        (independent['target'],independent['target_native'],independent['competitor_native']),'Gate target/native mapping differs')
                    self.same('allocated_risk',declared['alpha'],independent['alpha'],atol=0,rtol=0)
                    if mode!='normal_fixed':
                        self.same('lambda_rule',declared['lambda_selected'],independent['lambda_selected'],atol=0,rtol=0)
                        self.require(declared['lambda_halvings']==independent['lambda_halvings'],'Lambda halving count differs')
            lower=q.get('lower_bound_yuan')
            if feasible and lower is not None:self.require(lower<=q['cost_yuan']+1e-6,'New problem bound exceeds accepted new objective')
            initial=q.get('initial') or {}
            if initial.get('feasible'):
                self.require(initial['name'] in seeds,'Unknown initial point')
                value=seeds[initial['name']];_,_,dp,dq=base.operation.resource_values(config,value,ctx['labels'],forecast[:,:ctx['n']],ctx['base'])
                self.require(self.gates(model,forecast,forecast+np.c_[dp,dq],mode)['feasible'],'Declared new feasible seed fails new gate')
                self.same('initial_activation_fee',initial['cost_yuan'],base.procurement(config,value,value,kp,kq)['planned']['activation_yuan'],atol=1e-7)
            self.same('charged_compute',q['charged_total_seconds'],q['shared_preparation_seconds']+q['quote_wall_seconds'],atol=1e-9)
            self.same('including_sunk_compute',q['total_method_seconds_including_previous_shared_work'],q['charged_total_seconds']+q['previous_shared_preparation_seconds_sunk'],atol=1e-9)
            self.quote_records.append(dict(id=entry['id'],forecast=draw,policy=mode,status=q['status'],new_gate_feasible=feasible,
                anchor_fallback=q['anchor_fallback'],executed_gate_feasible=direct['feasible'],cost_yuan=q.get('cost_yuan'),lower_bound_yuan=lower,
                relative_gap=q.get('relative_gap'),native_status=q.get('native_status'),source=q.get('source'),initial=initial,
                optimizer_invoked=q.get('optimizer_invoked'),source_line=q.get('source_line'),search_scope=q.get('search_scope'),
                native_attributes=q.get('native_attributes'),new_quote_wall_seconds=q['quote_wall_seconds'],charged_total_seconds=q['charged_total_seconds'],
                previous_shared_preparation_seconds_sunk=q['previous_shared_preparation_seconds_sunk'],total_method_seconds=q['total_method_seconds_including_previous_shared_work'],
                resource_box_checks=q.get('resource_box_checks'),infeasibility_proved=q.get('infeasibility_proved',False),
                direct_gate=direct,executed_cost=controls[mode]['bill'],minimum_buffered_margin=direct['minimum_margin']-1e-8,
                bound_scope='Only this new distribution gate with unchanged scalar guards and physical resources. Old nominal absolute gate is absent, so old anchor lower bounds are not inherited. Old-anchor fallback is not a new-problem upper bound unless independently feasible.'))
        rotation=base.read(self.file(folder/'rotation_receipt.json'))
        matched=abs(controls['mgf_forecast']['bill']['total_yuan']-controls['mgf_forecast_rotation']['bill']['total_yuan'])<=1e-7
        self.require(rotation['same_planned_fee']==matched,'Rotation matching declaration differs')
        if not matched:self.fee_mismatches.append(dict(forecast=draw))
        task=base.read(self.file(folder/'complete.json'));rows=base.read(self.checked(task['results']))
        self.require(len(rows)==p['repeats']*len(POLICIES) and {(r['repeat'],r['strategy']) for r in rows}=={(r,s) for r in range(p['repeats']) for s in POLICIES},'Execution identities differ')
        current={};pairfiles={}
        for row in rows:
            repeat,policy=row['repeat'],row['strategy'];rd=folder/f'execution{repeat:02d}';path=rd/policy
            rec=dict(id=entry['id'],resource_variant=entry['resource_variant'],forecast=draw,repeat=repeat,strategy=policy,source=str(path),original_status=row['status'])
            try:
                self.require(row==base.read(self.file(path/'receipt.json')) and row['status']=='complete','Incomplete or changed execution')
                random=base.native.arrays(self.file(rd/'paired_randomness.npz'));oldrandom=base.native.arrays(self.file(previous/f'execution{repeat:02d}'/'paired_randomness.npz'))
                for k in random:self.same('replayed_random_'+k,random[k],oldrandom[k],atol=0,rtol=0)
                eta=np.r_[base.native.random_stream(p,entry,draw,repeat,773).uniform(.9,1.,pc),base.native.random_stream(p,entry,draw,repeat,774).uniform(.9,1.,qc)]
                self.same('response_law',random['eta'],eta,atol=0,rtol=0)
                for k,shape in enumerate((ctx['baseline_P'].shape,ctx['baseline_Q'].shape,ctx['baseline_P'].shape,ctx['baseline_root'].shape)):
                    self.same('meter_law',random['noise'+str(k)],base.native.random_stream(p,entry,draw,repeat,k+2).normal(size=shape),atol=0,rtol=0)
                obs=base.native.arrays(self.file(path/'observations.npz'));u=planner[policy]
                if policy in ('no_dfs','anchor'):
                    oldobs=base.native.arrays(self.file(previous/f'execution{repeat:02d}'/policy/'observations.npz'))
                    for k in ('P','Q','Y','V','root','requested_action','delivered_action'):self.same('replayed_observation_'+k,obs[k],oldobs[k],atol=0,rtol=0)
                self.same('sealed_request',obs['requested_action'],u,atol=0,rtol=0);expected=u*eta[None]
                for col,res in enumerate(config['resources'][:pc]):
                    if res['kind']=='pv':
                        node=ctx['labels'].index(res['node_label']);cap=np.minimum(res['contract_kw'],np.maximum(0.,-ctx['baseline_P'][:,node]*ctx['base']).reshape(24,4).min(1))
                        expected[:,col]=np.minimum(expected[:,col],cap)
                actual=obs['delivered_action'];self.same('delivered_response',actual,expected,atol=1e-12,rtol=0)
                delivered,av,dp,dq=base.operation.resource_values(config,actual,ctx['labels'],forecast[:,:ctx['n']],ctx['base'])
                self.same('actual_P',obs['true_P'],ctx['baseline_P']+dp,atol=1e-13);self.same('actual_Q',obs['true_Q'],ctx['baseline_Q']+dq,atol=1e-13)
                for name,k,sigma in [('P',0,.005),('Q',1,.005),('V',2,.0005),('root',3,.0005)]:self.same('measured_'+name,obs[name],obs['true_'+name]*(1+sigma*random['noise'+str(k)]),atol=1e-13)
                self.same('measured_Y',obs['Y'],obs['root'][:,None]**2-obs['V']**2,atol=1e-13)
                bill=base.procurement(config,u,actual,kp,kq)
                for field in ('planned','settled'):
                    self.costs(row[field+'_cost'],bill[field],field)
                    self.same('paid_request_peak',row[field+'_cost']['resource_reserved_capacity'],np.max(abs(u),axis=0),atol=0,rtol=0)
                    self.same(field+'_total',row[field+'_fee_yuan'],bill[field]['total_yuan'],atol=1e-8)
                for field,value in (('requested_quantities',u),('delivered_quantities',actual)):
                    self.same(field+'_P',row[field]['active_abs_kwh'],abs(value[:,:pc]).sum(),atol=1e-9)
                    self.same(field+'_Q',row[field]['reactive_abs_kvarh'],abs(value[:,pc:]).sum(),atol=1e-9)
                ev,values=self.score(ctx,obs,row);self.pair_file(self.file(path/'pair_evidence.npz'),ctx,values)
                pairfiles[repeat,policy]=base.native.arrays(path/'pair_evidence.npz')
                if 'pair' in ev:
                    i,j=ev['pair'];ev['forecast_evidence']=self.directed(ctx,forecasts[policy],i,j)
                    reps=model['representatives']
                    if i in reps:
                        target=reps.index(i);K=model['means'][target];Hi,Hj=model['H'][i],model['H'][j]
                        actual_power=np.c_[obs['true_P'],obs['true_Q']];expected_y=actual_power@K.T
                        actual_mu=float(ev['evidence']['h']+(np.square(expected_y-actual_power@Hj.T).sum()-np.square(expected_y-actual_power@Hi.T).sum())/(2*model['sigma2']))
                        actual_difference=actual_power@(Hi-Hj).T
                        actual_variance=float(np.einsum('ti,ij,tj->',actual_difference,model['covariance'],actual_difference,optimize=True)/model['sigma2']**2)
                        target_row=next(x for x in gates['mgf_forecast',policy]['rows'] if x['target_native']==i and x['competitor_native']==j)
                        fixed_mu=target_row['fixed_input_mean']
                        ev['historical_mean_diagnostic']=dict(native_target=i,native_competitor=j,
                            K_nominal_forecast_mu=fixed_mu,K_random_power_mean=target_row['mean'],K_actual_input_mu=actual_mu,
                            fixed_power_score_variance=target_row['fixed_input_variance'],random_power_score_variance=target_row['variance'],
                            actual_input_voltage_error_score_variance=actual_variance,
                            actual_power_minus_nominal_forecast=actual_mu-fixed_mu,
                            clean_minus_actual_K=ev['evidence']['clean_logodds']-actual_mu,
                            observed_minus_clean=ev['evidence']['meter'],clean_minus_nominal_K=ev['evidence']['clean_logodds']-fixed_mu,
                            scope='Offline true-family direction. Actual P/Q and physical voltage are used only after execution; planning K and all score H remain historical and frozen.')
                    if policy in exact:ev['ideal_evidence']=self.directed(ctx,exact[policy],i,j)
                    zero=ev if policy=='no_dfs' else current[repeat,'no_dfs']['evaluation']
                    ev['evidence_increment']={k:ev['evidence'][k]-zero['evidence'][k] for k in ('J','bias','meter','clean_logodds','observed_logodds')}
                    if policy in exact and 'no_dfs' in exact:
                        BudgetAudit.verify_reported_evidence(self,row['evidence'],pairfiles[repeat,policy],pairfiles[repeat,'no_dfs'],
                            base.native.arrays(folder/'exact_requested'/policy/'forecast_pair_evidence.npz'),base.native.arrays(folder/'exact_requested/no_dfs/forecast_pair_evidence.npz'),
                            base.native.arrays(folder/'exact_requested'/policy/'pair_evidence.npz'),base.native.arrays(folder/'exact_requested/no_dfs/pair_evidence.npz'))
                physical=row['physical_state'];ac=base.native.arrays(self.checked(physical))
                if 'time_slice' in physical:
                    a,b=physical['time_slice'];count=len(ac['voltage_complex_pu']);ac={k:v[a:b] if v.ndim and v.shape[0]==count else v for k,v in ac.items()}
                electrical,bv,bus=base.operation.electrical(self.operation,ac,obs,metadata,branches,buses)
                rec.update(audit_status='passed',requested=controls[policy]['requested'],delivered=delivered,requested_violations=controls[policy]['violations'],delivered_violations=av,
                    planned_cost=bill['planned'],settled_cost=bill['settled'],evaluation=ev,electrical=electrical,branches=bv,buses=bus,
                    quote_new_gate_feasible=row['quote_new_constraints_feasible'],quote_anchor_fallback=row['quote_anchor_fallback'])
                current[repeat,policy]=rec
            except Exception as exc:rec.update(audit_status='failed',error=repr(exc),traceback=traceback.format_exc())
            self.records.append(rec)


def boundary(path):
    tree=ast.parse(Path(path).read_text(encoding='utf8'))
    context=[x.lineno for x in ast.walk(tree) if isinstance(x,ast.Call) and ast.unparse(x.func)=='old.context']
    seals=[x.lineno for x in ast.walk(tree) if isinstance(x,ast.Call) and ast.unparse(x.func)=='io.write' and x.args and "'seal.json'" in ast.unparse(x.args[0])]
    plans=[x for x in ast.walk(tree) if isinstance(x,ast.Call) and ast.unparse(x.func)=='solver.plan']
    args=[ast.unparse(x) for x in plans[0].args] if len(plans)==1 else []
    expected=['case',"planning['score']","planning['val']",'prediction',"planning['base']"]
    line_driver=Path(path).name=='line_run.py'
    line_info={};planning_passed=args==expected
    if line_driver:
        calls={name:[x for x in ast.walk(tree) if isinstance(x,ast.Call) and ast.unparse(x.func)==name]
            for name in ('evidence.DistributionEvidence','line_seed.search_lines','line_polish.polish_resources')}
        line_info={name:[ast.unparse(a) for a in value[0].args] if len(value)==1 else [] for name,value in calls.items()}
        planning_passed=(not plans and line_info['evidence.DistributionEvidence']==expected
            and line_info['line_seed.search_lines']==['case','model','anchor','{k: v.copy() for k, v in seeds.items()}']
            and line_info['line_polish.polish_resources']==['case','model','point'])
    return dict(passed=len(context)==1 and len(seals)==1 and context[0]>seals[0] and planning_passed,
        context_lines=context,seal_lines=seals,solver_arguments=args,line_driver=line_driver,line_planning_arguments=line_info,
        scope='Physical context opened after planning seal; planning receives only contracts, frozen score, historical validation, sealed forecast and old planning-only anchor/seed arrays. Line driver calls no QCQP optimizer; resource-only rotation LP remains separately timed.')


def pair_moments(records):
    rows=[r['evaluation'] for r in records if r.get('audit_status')=='passed' and 'historical_mean_diagnostic' in r['evaluation']]
    if not rows:return dict(n=0)
    values={k:np.asarray([r['evidence'][k] for r in rows],float) for k in ('clean_logodds','observed_logodds','meter')}
    n=len(rows);variance=lambda x:float(np.var(x,ddof=1)) if n>1 else None
    answer=dict(n=n,native_pair=rows[0]['pair'],sample={k:dict(mean=float(v.mean()),variance=variance(v)) for k,v in values.items()},
        model={k:float(np.mean([r['historical_mean_diagnostic'][k] for r in rows])) for k in (
            'K_nominal_forecast_mu','K_random_power_mean','K_actual_input_mu','fixed_power_score_variance','random_power_score_variance','actual_input_voltage_error_score_variance')},
        scope='Within-forecast paired actual AC/response/meter paths. Conditional declared-model moments are compared descriptively; 32 paths do not establish calibration across physical days.')
    if n>1:
        cov=float(np.cov(values['clean_logodds'],values['meter'],ddof=1)[0,1])
        answer['clean_meter_covariance']=cov
        answer['variance_decomposition_error']=variance(values['observed_logodds'])-variance(values['clean_logodds'])-variance(values['meter'])-2*cov
        answer['observed_mean_standard_error']=float(np.std(values['observed_logodds'],ddof=1)/np.sqrt(n))
    return answer


def run(source,out):
    source,out=Path(source).resolve(),Path(out).resolve()
    if out.exists():raise FileExistsError(out)
    if not(source/'complete.json').exists():raise ValueError('Completed run required')
    out.mkdir(parents=True);shutil.copyfile(__file__,out/'audit.py');audit=Audit()
    p=base.read(audit.file(source/'protocol.json'));entries=base.read(audit.checked(p['inputs']));complete=base.read(audit.file(source/'complete.json'));completion=base.read(audit.checked(complete['main']))
    audit.require(complete['source_unchanged'] and tuple(p['policies'])==POLICIES and len(entries)==1,'Source/policy identity differs')
    expected={(e['id'],f) for e in entries for f in range(p['forecasts'])};ok={(r['id'],r['forecast_index']) for r in completion['receipts']};failed={(r['id'],r['forecast_index']) for r in completion['failed_tasks']}
    audit.require(not(ok&failed) and ok|failed==expected and len(ok)==len(completion['receipts']),'Task identities incomplete or duplicated')
    for item in p['sources']:audit.file(source/'source_snapshot'/Path(item['path']).resolve().relative_to(base.native.ROOT),item['sha256'])
    meta=source/'source_snapshot/topology_identification_standalone/cases'/entries[0]['case'];metadata=base.read(audit.file(meta/'metadata.json'))
    with audit.file(meta/'branches.csv').open(encoding='utf-8-sig',newline='') as f:branches=list(csv.DictReader(f))
    with audit.file(meta/'buses.csv').open(encoding='utf-8-sig',newline='') as f:buses=list(csv.DictReader(f))
    for name in ('metadata.json','branches.csv','buses.csv'):audit.require(base.ref(meta/name)['sha256']==base.ref(base.native.ROOT/'rnj_wzzt_dr/physical/cases'/entries[0]['case']/name)['sha256'],'Physical reference table changed')
    driver=Path(p.get('driver_relative_path','rnj_wzzt_dr/experiments/distribution_gate_20261114/run.py'))
    audit.require(not driver.is_absolute() and '..' not in driver.parts,'Driver snapshot path is not repository-relative')
    info=boundary(source/'source_snapshot'/driver);audit.require(info['passed'],'Planning boundary differs')
    own=dict(created_utc=datetime.now(timezone.utc).isoformat(),source=base.ref(__file__),source_run=base.ref(source/'protocol.json'),expected_rows=p['forecasts']*p['repeats']*len(POLICIES),
        planning_boundary=info,probability_source=base.ref(prob.__file__),scope='Independent all-native Gaussian integral, original scalar guards, physical resources, payments, native scores and stored full-network AC. No fitting, optimizer or AC solve.',
        environment=dict(python=sys.version,numpy=np.__version__))
    helpers={prob.__module__ if hasattr(prob,'__module__') else prob.__name__,base.__name__,base.native.__name__,base.operation.__name__,base.direct_original.__module__,BudgetAudit.__module__,paired.__module__}
    own['helper_sources']=[base.ref(sys.modules[name].__file__) for name in sorted(helpers)]
    for item in own['helper_sources']:
        path=Path(item['path']);dest=out/'helper_snapshots'/path.relative_to(base.native.ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,dest)
    base.write(out/'protocol.json',own)
    for entry in entries:
        for draw in range(p['forecasts']):
            folder=source/'fraction0.10'/entry['id']/f'forecast{draw:02d}'
            try:
                if (entry['id'],draw) in failed:raise ValueError('Original task failed')
                audit.forecast(folder,entry,p,metadata,branches,buses)
            except Exception as exc:audit.task_errors.append(dict(path=str(folder),error=repr(exc),traceback=traceback.format_exc()))
            base.write(out/'progress.json',dict(rows=len(audit.records),task_errors=len(audit.task_errors),row_errors=sum(r['audit_status']!='passed' for r in audit.records)))
    passed=len(audit.records)==own['expected_rows'] and not audit.task_errors and all(r['audit_status']=='passed' for r in audit.records)
    groups={s:[r for r in audit.records if r['strategy']==s] for s in POLICIES}
    result=dict(status='passed' if passed else 'failed',expected_rows=own['expected_rows'],audited_rows=len(audit.records),protocol=base.ref(out/'protocol.json'),
        original_failed_tasks=completion['failed_tasks'],original_quote_exceptions=completion['quote_exceptions'],original_execution_failed_rows=completion['execution_failed_rows'],
        task_errors=audit.task_errors,exact_diagnostic_failures=audit.exact_failures,quotes=audit.quote_records,records=audit.records,
        groups={s:base.summarize(rows) for s,rows in groups.items()},comparisons={s:{b:paired(groups[s],groups[b]) for b in ('no_dfs','anchor')} for s in MODES},
        per_forecast={str(draw):dict(groups={s:base.summarize([r for r in rows if r['forecast']==draw]) for s,rows in groups.items()},
            pair_moments={s:pair_moments([r for r in rows if r['forecast']==draw]) for s,rows in groups.items()},
            comparisons={s:{b:paired([r for r in groups[s] if r['forecast']==draw],[r for r in groups[b] if r['forecast']==draw]) for b in ('no_dfs','anchor')} for s in MODES}) for draw in range(p['forecasts'])},
        rotation_unmatched_fees=audit.fee_mismatches,maximum_absolute_errors=dict(audit.maxima),electrical_maximum_errors=dict(audit.operation.maxima),counts=dict(audit.counts))
    base.write(out/'report.json',result);base.write(out/'inputs_manifest.json',[dict(path=k,sha256=v) for k,v in audit.hashes.items()]);base.write(out/'complete.json',dict(status=result['status'],report=base.ref(out/'report.json'),source_unchanged=base.ref(__file__)['sha256']==own['source']['sha256']))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',type=Path,required=True);parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
    result=run(args.run,args.out);print(dict(status=result['status'],rows=result['audited_rows'],expected=result['expected_rows'],task_errors=len(result['task_errors']),row_errors=sum(r['audit_status']!='passed' for r in result['records'])))
    raise SystemExit(0 if result['status']=='passed' else 1)
