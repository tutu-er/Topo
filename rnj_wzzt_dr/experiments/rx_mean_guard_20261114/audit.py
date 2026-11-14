"""Independent audit of auxiliary-history-mean QCQP and frozen-score execution.

No production optimizer, mean evaluator, identifier or AC solver is imported.
Original numerical helpers reconstruct score supports and stored AC states.
"""
from __future__ import annotations

import argparse
import ast
from collections import defaultdict
import csv
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import shutil
import sys
from statistics import NormalDist
import traceback

import numpy as np

from ..flexibility_stability_20261114 import audit as base
from ..flexibility_stability_20261114.analyze import paired
from ..flexibility_call_budget_20261114.audit import Audit as BudgetAudit


POLICIES=('no_dfs','anchor','mean_guard','mean_guard_rotation')


def planning_boundary(path):
    """Record the sealed adapter's exact historical-data allowlist."""
    tree=ast.parse(Path(path).read_text(encoding='utf-8-sig'))
    calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)
        and isinstance(n.func.value,ast.Name) and n.func.value.id=='solver' and n.func.attr=='plan']
    fields=[];mean_assignments=[]
    for node in ast.walk(tree):
        if not isinstance(node,ast.Assign):continue
        if any(isinstance(t,ast.Name) and t.id=='planning' for t in node.targets):
            value=node.value
            if isinstance(value,ast.DictComp) and len(value.generators)==1:
                gen=value.generators[0]
                if (not gen.ifs and isinstance(gen.iter,(ast.Tuple,ast.List)) and isinstance(gen.target,ast.Name)
                    and isinstance(value.key,ast.Name) and value.key.id==gen.target.id
                    and isinstance(value.value,ast.Subscript) and isinstance(value.value.value,ast.Name)
                    and value.value.value.id=='ctx' and isinstance(value.value.slice,ast.Name) and value.value.slice.id==gen.target.id):
                    fields.append([x.value if isinstance(x,ast.Constant) else '<dynamic>' for x in gen.iter.elts])
        for target in node.targets:
            if isinstance(target,ast.Subscript) and isinstance(target.value,ast.Name) and target.value.id=='planning':
                mean_assignments.append((ast.unparse(target.slice),ast.unparse(node.value)))
    args=[ast.unparse(x) for x in calls[0].args] if len(calls)==1 else []
    keywords={x.arg:ast.unparse(x.value) for x in calls[0].keywords} if len(calls)==1 else {}
    passed=(fields==[['score','val','base']] and mean_assignments==[("'means'",'means')]
        and args==['case',"planning['score']","planning['val']",'prediction',"planning['base']"]
        and keywords.get('means')=="planning['means']")
    return dict(passed=passed,fields=fields,extra_assignments=mean_assignments,positional_arguments=args,
        auxiliary_mean_argument=keywords.get('means'),
        scope='Sealed solver call receives contracts, frozen score, history validation, noisy forecast, base and sealed historical K; actual voltage and physical truth are absent from this call.')


def full_mean_gates(score,means,power,nominal):
    """Compare a residual-SSE calculation with the explicit quadratic formula."""
    H=np.asarray([row['H'] for row in score['parameters']],float)
    history=np.asarray([row['history_sse'] for row in score['parameters']],float)
    sigma2=float(score['sigma2']);means=np.asarray(means,float)
    if H.shape!=means.shape or not np.isfinite(means).all():raise ValueError('Mean/scoring bank shape differs')
    rows=[];totals={str(i):0. for i in nominal['targets']};maximum=0.
    for index,pair in enumerate(nominal['pairs']):
        for i,j in (pair['pair'],pair['pair'][::-1]):
            if i not in nominal['targets']:continue
            y=power@means[i].T;pi=power@H[i].T;pj=power@H[j].T
            h=float((history[j]-history[i])/(2*sigma2));d=pi-pj
            J=float(np.square(d).sum()/(2*sigma2))
            bias=float(np.sum((y-pi)*d)/sigma2)
            mu=float(h+(np.square(y-pj).sum()-np.square(y-pi).sum())/(2*sigma2))
            maximum=max(maximum,abs(mu-h-J-bias))
            margin=float(mu-nominal['z']*np.sqrt(2*J)-nominal['tie'])
            failure=NormalDist().cdf((nominal['tie']-mu)/np.sqrt(2*J)) if J>0 else float(mu<=nominal['tie'])
            totals[str(i)]+=failure
            rows.append(dict(pair_index=index,target=i,competitor=j,h=h,J=J,mean_bias=bias,mu=mu,
                original_margin=margin,buffered_margin=margin-nominal['buffer'],failure=failure))
    if maximum>1e-8:raise ArithmeticError('Full-SSE and explicit mean formulas differ')
    mean_ok=all(row['original_margin']>=-1e-7 and row['failure']<=nominal['alpha']+1e-9 for row in rows)
    return dict(mean_feasible=mean_ok,original_feasible=nominal['passed_original_acceptance'],
        augmented_feasible=bool(mean_ok and nominal['passed_original_acceptance']),rows=rows,
        minimum_mean_margin=min((row['original_margin'] for row in rows),default=None),
        minimum_mean_buffered_margin=min((row['buffered_margin'] for row in rows),default=None),
        strict_buffered_mean_nonnegative=all(row['buffered_margin']>=0 for row in rows),
        mean_target_failure_sums=totals,mean_worst_recovery_lower=max(0.,1-max(totals.values(),default=0)),
        maximum_mean_formula_identity_error=maximum)


class Audit(base.Audit):
    def means_bank(self,protocol,entry,ctx):
        receipt=base.read(self.checked(protocol['means_source']));fitting=base.read(self.checked(receipt['protocol']))
        self.require(receipt['status']=='complete' and receipt['original_score_unchanged'],'Auxiliary fit incomplete or scoring changed')
        self.require(fitting['train']['sha256']==entry['files']['train.npz']['sha256'] and
            fitting['validation']['sha256']==entry['files']['validation.npz']['sha256'],'Means history lineage changed')
        self.require(base.read(self.checked(fitting['score']))==ctx['score'],'Frozen scoring H differs from mean-fit reference')
        original=base.native.arrays(self.checked(receipt['means']));data=base.native.arrays(self.checked(protocol['auxiliary_means']))
        for key in original:self.same('copied_mean_'+key,data[key],original[key],atol=0,rtol=0)
        indices=[row['index'] for row in ctx['score']['models']]
        self.same('mean_native_order',data['native_indices'],indices,atol=0,rtol=0)
        self.same('mean_label_order',data['labels'],ctx['labels'],atol=0,rtol=0)
        train=base.native.arrays(self.checked(fitting['train']));val=base.native.arrays(self.checked(fitting['validation']))
        self.require(len(train['P'])==192 and len(val['P'])==96,'Auxiliary fit row counts differ')
        for a in (train,val):self.same('fit_labels',a['labels'],ctx['labels'],atol=0,rtol=0)
        power=np.vstack([np.c_[a['P'],a['Q']] for a in (train,val)]);y=np.vstack([a['Y'] for a in (train,val)])
        checks=[]
        for k,item in enumerate(receipt['fits']):
            fit=base.read(self.checked(item));point=fit['point'];n=ctx['n'];K=np.zeros((n,2*n))
            self.require(fit['representative_index']==k and fit['native_index']==indices[k],'Mean-fit ordering changed')
            expected=ctx['native']['frozen_candidates'][indices[k]]['point']['supports']
            self.require(point['supports']==expected and fit['supports']==expected,'Auxiliary fixed structure changed')
            for support,r,x in zip(point['supports'],point['r'],point['x'],strict=True):
                self.require(0<=r<=2 and 0<=x<=2,'Auxiliary coefficient outside declared bound')
                block=np.ix_(support,support);K[:,:n][block]+=r;K[:,n:][block]+=x
            self.same('rebuilt_auxiliary_K',data['means'][k],K,atol=1e-14,rtol=1e-14)
            residual=y-power@K.T;mse=float(np.square(residual).mean())
            self.same('historical_mean_MSE',fit['history_fit_error']['mse'],mse,atol=1e-15)
            self.same('historical_mean_QP_objective',fit['QP']['objective'],mse/fitting['reference_mse'],atol=1e-8)
            checks.append(dict(representative_index=k,native_index=indices[k],rows=288,
                fixed_supports_verified=True,QP_status=fit['QP']['status'],objective=mse/fitting['reference_mse'],
                reported_lower_bound=fit['QP']['lower_bound'],reported_certificate_gap=fit['QP']['certificate_gap']))
        self.mean_lineage=dict(fitting=receipt['protocol'],fit_rows=288,train_rows=192,validation_rows=96,checks=checks,
            scope='K is fitted on historical train+validation; validation is not independent validation of K. Frozen H still scores every execution. No uncertainty-set coverage is asserted.')
        return data['means']

    def forecast(self,folder,entry,p,metadata,branches,buses):
        ctx=self.context(entry);means=self.means_bank(p,entry,ctx);draw=int(folder.name.removeprefix('forecast'))
        seal=base.read(self.file(folder/'seal.json'))
        for key in ('planner','plans','reservation','anchor','mean_prediction_audits','score','means_source','auxiliary_means','protocol'):self.checked(seal[key])
        planner=base.native.arrays(self.file(folder/'planner.npz'));self.require(set(planner)=={'forecast',*POLICIES},'Policy arrays differ')
        config=base.read(self.file(folder/'resource_case.json'));pcount=self.catalogue(entry,config,ctx);qcount=len(config['resources'])-pcount
        forecast=planner['forecast'];expected=np.c_[ctx['baseline_P'],ctx['baseline_Q']]*(1+.05*base.native.random_stream(p,entry,draw,0,991).normal(size=forecast.shape))
        self.same('forecast_random_law',forecast,expected,atol=0,rtol=0)
        kp=float(p['reservation_prices']['P']);kq=float(p['reservation_prices']['Q'])
        plans=base.read(self.file(folder/'plans.json'));self.require(set(plans)=={'mean_guard'},'Unexpected quotes')
        quote=plans['mean_guard'];anchor=base.read(self.file(folder/'anchor_plan.json'))
        generated=base.native.arrays(self.file(folder/'generated_initial_points.npz'));shared=base.native.arrays(self.file(folder/'shared_initial_points.npz'))
        self.require(set(shared)==set(generated)|({'shared_150_second_anchor'} if anchor['feasible'] else set()),'Shared seed identity differs')
        for name,value in generated.items():self.same('unchanged_common_seed',shared[name],value,atol=0,rtol=0)
        if anchor['feasible']:self.same('shared_anchor',shared['shared_150_second_anchor'],planner['anchor'],atol=0,rtol=0)
        self.quote(folder,entry,ctx,config,planner,'anchor',anchor,kp,kq,generated)
        reservations=base.native.arrays(self.file(folder/'reservation.npz'));audits=base.read(self.file(folder/'mean_prediction_audits.json'))
        fctx=dict(entry=entry,score=ctx['score'],config=config,planner=planner,directory=folder)
        controls={};exact={};forecast_values={}
        for policy in POLICIES:
            u=planner[policy];requested,rv,dp,dq=base.operation.resource_values(config,u,ctx['labels'],forecast[:,:ctx['n']],ctx['base'])
            direct=base.direct_original(fctx,u,quote);mg=full_mean_gates(ctx['score'],means,forecast+np.c_[dp,dq],direct)
            saved=audits[policy]['evidence']
            for key,name in [('mean_mu','mu'),('mean_pair_J','J'),('mean_margins','original_margin'),('mean_pair_failure','failure')]:
                self.same('saved_'+key,saved[key],[row[name] for row in mg['rows']],atol=1e-8)
            self.require(saved['original_feasible']==mg['original_feasible'] and saved['mean_feasible']==mg['mean_feasible'] and saved['feasible']==mg['augmented_feasible'],'Saved mean feasibility changed')
            self.same('reserved_peak_'+policy,reservations[policy],np.max(abs(u),axis=0),atol=0,rtol=0)
            bill=base.procurement(config,u,u,kp,kq)['planned'];controls[policy]=dict(requested=requested,violations=rv,direct=direct,mean=mg,bill=bill)
            forecast_values[policy]=self.evaluate(ctx,forecast+np.c_[dp,dq])
            ep=folder/'exact_requested'/policy;er=base.read(self.file(ep/'receipt.json'))
            if er['status']=='complete':
                clean=base.native.arrays(self.file(ep/'clean_observation.npz'))
                self.same('ideal_P',clean['P'],ctx['baseline_P']+dp,atol=1e-13);self.same('ideal_Q',clean['Q'],ctx['baseline_Q']+dq,atol=1e-13)
                self.same('ideal_Y',clean['Y'],clean['root'][:,None]**2-clean['V']**2,atol=1e-13)
                exact[policy]=self.evaluate(ctx,np.c_[clean['P'],clean['Q']],clean['Y'],np.c_[clean['P'],clean['Q']],clean['Y'])
                self.pair_file(self.file(ep/'pair_evidence.npz'),ctx,exact[policy]);self.pair_file(self.file(ep/'forecast_pair_evidence.npz'),ctx,forecast_values[policy])
            else:self.exact_failures.append(dict(path=str(ep),status=er['status']))
        final=controls['mean_guard'];new_ok=bool(quote['new_constraints_feasible'])
        if not hasattr(self,'forecast_records'):self.forecast_records=[]
        self.forecast_records.append(dict(id=entry['id'],forecast=draw,
            policies={name:dict(original=controls[name]['direct'],auxiliary_mean=controls[name]['mean'],planned_cost=controls[name]['bill']) for name in POLICIES}))
        if new_ok:
            self.require(quote['feasible'] and final['mean']['augmented_feasible'],'Augmented quote violates new or original constraints')
            self.same('augmented_activation_cost',quote['cost_yuan'],final['bill']['activation_yuan'],atol=1e-7)
        self.require(new_ok==bool(quote['feasible']),'Reported optimization and accepted-point feasibility differ')
        self.require(sum(bool(quote[k]) for k in ('new_constraints_feasible','anchor_fallback','zero_fallback'))==1,'Execution-source flags are not exclusive and exhaustive')
        if quote['anchor_fallback']:
            self.require(not new_ok and anchor['feasible'],'Invalid original-anchor fallback')
            self.same('old_anchor_fallback',planner['mean_guard'],planner['anchor'],atol=0,rtol=0)
        if quote['zero_fallback']:self.same('zero_fallback',planner['mean_guard'],np.zeros_like(planner['mean_guard']),atol=0,rtol=0)
        self.require(quote['executed_original_constraints_feasible']==final['mean']['original_feasible'] and
            quote['executed_mean_constraints_passed_direct']==final['mean']['mean_feasible'],'Executed-point feasibility flags differ')
        lower=quote.get('lower_bound_yuan')
        if lower is not None and new_ok:self.require(lower<=quote['cost_yuan']+1e-6,'Augmented bound exceeds augmented feasible objective')
        inherited=None
        if anchor.get('lower_bound_yuan') is not None and not anchor.get('lower_bound_withdrawn',False):
            self.require(anchor['objective_kind']=='activation' and quote['objective_kind']=='activation','Inherited bound objective differs')
            self.require(anchor['evidence_mode']=='symmetric_guard' and quote['original_gates_retained'] and quote['original_scalar_guards_retained'],'Original constraints were not retained')
            for key in ('target_mass','targets','individual_comparison_risk','numerical_gate_buffer','pair_count','guard_count'):
                self.require(anchor[key]==quote[key],'Inherited-bound gate metadata differs: '+key)
            self.require(anchor['pair_reduction']==quote['pair_reduction_for_audit'],'Original gate reductions differ')
            anchor_lower=max(0.,float(anchor['lower_bound_yuan']))
            if anchor.get('lower_bound_source')=='native_ObjBound':
                self.same('inherited_anchor_native_bound',anchor_lower,anchor['native_attributes']['ObjBound'],atol=1e-12)
            new_lower=0. if lower is None or quote.get('lower_bound_withdrawn',False) else max(0.,float(lower))
            combined=max(anchor_lower,new_lower)
            if new_ok:self.require(combined<=quote['cost_yuan']+1e-6,'Inherited bound exceeds accepted augmented objective')
            inherited=dict(anchor_receipt=base.ref(folder/'anchor_plan.json'),anchor_valid_lower_bound_yuan=anchor_lower,
                augmented_native_used_lower_bound_yuan=new_lower,combined_lower_bound_yuan=combined,
                augmented_accepted_upper_bound_yuan=quote.get('cost_yuan') if new_ok else None,
                augmented_gap_using_inherited_bound=(quote['cost_yuan']-combined)/max(quote['cost_yuan'],1e-12) if new_ok else None,
                derivation='Same forecast, contracts and activation objective. Augmented constraints retain every buffered nominal gate and historical scalar guard, adding K-mean gates; projecting feasible (u,s) therefore gives a subset of the original domain. Its optimum cannot be below the original valid lower bound.',
                scope='Post-hoc original-domain solver bound for this buffered representative-bank activation QCQP, subject to native numerical tolerances. Original receipts and native gaps unchanged; no AC, parameter-uncertainty, total-bill or global-identification claim.')
        self.same('charged_total_time',quote['charged_total_seconds'],quote['shared_preparation_seconds']+quote['quote_wall_seconds'],atol=1e-12)
        self.same('one_time_historical_fit_seconds',quote['auxiliary_mean_fit_seconds_once'],p['auxiliary_mean_fit_seconds_once'],atol=0,rtol=0)
        self.require(0<=quote['outer_evidence_seconds']<=quote['quote_wall_seconds'],'Outer evidence timing is not included in quote timing')
        initial=quote.get('initial') or {}
        if initial.get('feasible'):
            self.require(initial['name'] in shared,'Unknown augmented initial point')
            point=shared[initial['name']];_,_,dp,dq=base.operation.resource_values(config,point,ctx['labels'],forecast[:,:ctx['n']],ctx['base'])
            dd=base.direct_original(fctx,point,quote);mm=full_mean_gates(ctx['score'],means,forecast+np.c_[dp,dq],dd)
            self.require(mm['augmented_feasible'],'Claimed augmented feasible seed does not pass mean gate')
            self.same('augmented_initial_fee',initial['cost_yuan'],base.procurement(config,point,point,kp,kq)['planned']['activation_yuan'],atol=1e-7)
        self.quote_records.append(dict(id=entry['id'],forecast=draw,policy='mean_guard',status=quote['status'],
            new_constraints_feasible=new_ok,anchor_fallback=quote['anchor_fallback'],zero_fallback=quote['zero_fallback'],
            executed_source=quote['executed_source'],cost_yuan=quote.get('cost_yuan'),lower_bound_yuan=lower,
            relative_gap=quote.get('relative_gap'),native_status=quote.get('native_status'),native_attributes=quote.get('native_attributes'),
            charged_total_seconds=quote['charged_total_seconds'],quote_wall_seconds=quote['quote_wall_seconds'],initial=initial,
            outer_evidence_seconds=quote['outer_evidence_seconds'],shared_preparation_seconds=quote['shared_preparation_seconds'],
            auxiliary_mean_fit_seconds_once=quote['auxiliary_mean_fit_seconds_once'],
            requested_quote_budget_seconds=quote['requested_quote_budget_seconds'],quote_budget_exceeded=quote['quote_budget_exceeded'],
            posthoc_inherited_bound=inherited,
            direct_original=final['direct'],direct_mean=final['mean'],executed_bill=final['bill'],
            bound_scope='Augmented activation-objective bound. An old-anchor fallback outside the new domain is not an upper bound for this problem.'))
        rotation=base.read(self.file(folder/'rotation_receipt.json'));samefee=abs(final['bill']['total_yuan']-controls['mean_guard_rotation']['bill']['total_yuan'])<=1e-7
        self.require(rotation['same_planned_fee']==samefee,'Rotation total-fee flag differs')
        if not samefee:self.fee_mismatches.append(dict(forecast=draw,target=final['bill']['total_yuan'],rotation=controls['mean_guard_rotation']['bill']['total_yuan']))
        task=base.read(self.file(folder/'complete.json'));rows=base.read(self.checked(task['results']))
        self.require(task['rows']==len(rows)==p['repeats']*len(POLICIES),'Execution denominator differs')
        self.require({(r['repeat'],r['strategy']) for r in rows}=={(repeat,policy) for repeat in range(p['repeats']) for policy in POLICIES},'Missing/duplicate execution identity')
        current_pairs={};current_records={}
        for row in rows:
            repeat,policy=row['repeat'],row['strategy'];rd=folder/f'execution{repeat:02d}';path=rd/policy
            rec=dict(id=entry['id'],resource_variant=entry['resource_variant'],forecast=draw,repeat=repeat,strategy=policy,source=str(path),original_status=row['status'])
            try:
                self.require(row==base.read(self.file(path/'receipt.json')),'Receipt changed');self.require(row['status']=='complete','Original execution failed')
                random=base.native.arrays(self.file(rd/'paired_randomness.npz'))
                eta=np.r_[base.native.random_stream(p,entry,draw,repeat,773).uniform(.9,1.,pcount),base.native.random_stream(p,entry,draw,repeat,774).uniform(.9,1.,qcount)]
                self.same('response_law',random['eta'],eta,atol=0,rtol=0)
                for k,shape in enumerate((ctx['baseline_P'].shape,ctx['baseline_Q'].shape,ctx['baseline_P'].shape,ctx['baseline_root'].shape)):
                    self.same('meter_law_'+str(k),random['noise'+str(k)],base.native.random_stream(p,entry,draw,repeat,k+2).normal(size=shape),atol=0,rtol=0)
                obs=base.native.arrays(self.file(path/'observations.npz'));u=planner[policy]
                self.same('sealed_request',obs['requested_action'],u,atol=0,rtol=0);expected=u*eta[None]
                for col,res in enumerate(config['resources'][:pcount]):
                    if res['kind']=='pv':
                        node=ctx['labels'].index(res['node_label']);cap=np.minimum(res['contract_kw'],np.maximum(0.,-ctx['baseline_P'][:,node]*ctx['base']).reshape(24,4).min(1))
                        expected[:,col]=np.minimum(expected[:,col],cap)
                actual=obs['delivered_action'];self.same('response_delivery_clip',actual,expected,atol=1e-12,rtol=0)
                delivered,av,dp,dq=base.operation.resource_values(config,actual,ctx['labels'],forecast[:,:ctx['n']],ctx['base'])
                self.same('actual_P',obs['true_P'],ctx['baseline_P']+dp,atol=1e-13);self.same('actual_Q',obs['true_Q'],ctx['baseline_Q']+dq,atol=1e-13)
                for name,k,sigma in [('P',0,.005),('Q',1,.005),('V',2,.0005),('root',3,.0005)]:self.same('measured_'+name,obs[name],obs['true_'+name]*(1+sigma*random['noise'+str(k)]),atol=1e-13)
                self.same('measured_Y',obs['Y'],obs['root'][:,None]**2-obs['V']**2,atol=1e-13)
                bill=base.procurement(config,u,actual,kp,kq);self.costs(row['planned_cost'],bill['planned'],'planned_cost');self.costs(row['settled_cost'],bill['settled'],'settled_cost')
                for key in ('planned_cost','settled_cost'):
                    self.same('paid_requested_peak',row[key]['resource_reserved_capacity'],np.max(abs(u),axis=0),atol=0,rtol=0)
                for key,value in (('requested_quantities',u),('delivered_quantities',actual)):
                    self.same(key+'_P',row[key]['active_abs_kwh'],np.abs(value[:,:pcount]).sum(),atol=1e-9)
                    self.same(key+'_Q',row[key]['reactive_abs_kvarh'],np.abs(value[:,pcount:]).sum(),atol=1e-9)
                self.same('delivered_P_amount',row['delivered_abs_kwh'],np.abs(actual[:,:pcount]).sum(),atol=1e-9)
                self.same('delivered_Q_amount',row['delivered_abs_kvarh'],np.abs(actual[:,pcount:]).sum(),atol=1e-9)
                self.same('planned_total',row['planned_fee_yuan'],bill['planned']['total_yuan'],atol=1e-8);self.same('settled_total',row['settled_fee_yuan'],bill['settled']['total_yuan'],atol=1e-8)
                ev,values=self.score(ctx,obs,row);self.pair_file(self.file(path/'pair_evidence.npz'),ctx,values)
                cp=base.native.arrays(path/'pair_evidence.npz');current_pairs[(repeat,policy)]=cp
                if 'pair' in ev:
                    i,j=ev['pair'];ev['forecast_evidence']=self.directed(ctx,forecast_values[policy],i,j)
                    native_indices=[m['index'] for m in ctx['score']['models']]
                    if i in native_indices and j in native_indices:
                        ii,jj=native_indices.index(i),native_indices.index(j)
                        forecast_mean=next(x for x in controls[policy]['mean']['rows'] if x['target']==ii and x['competitor']==jj)
                        actual_power=np.c_[obs['true_P'],obs['true_Q']]
                        predicted_y=actual_power@means[ii].T
                        Hi=np.asarray(ctx['score']['parameters'][ii]['H']);Hj=np.asarray(ctx['score']['parameters'][jj]['H'])
                        actual_input_mu=float(ev['evidence']['h']+(np.square(predicted_y-actual_power@Hj.T).sum()-np.square(predicted_y-actual_power@Hi.T).sum())/(2*ctx['sigma2']))
                        ev['auxiliary_mean_diagnostic']=dict(native_target=i,native_competitor=j,
                            forecast_conditional_mu=forecast_mean['mu'],actual_input_conditional_mu=actual_input_mu,
                            clean_minus_forecast_mu=ev['evidence']['clean_logodds']-forecast_mean['mu'],
                            clean_minus_actual_input_mu=ev['evidence']['clean_logodds']-actual_input_mu,
                            changed_power_contribution=actual_input_mu-forecast_mean['mu'],
                            nominal_forecast_mu=ev['forecast_evidence']['h']+ev['forecast_evidence']['J'],
                            scope='Offline true-family direction for diagnosis. Actual inputs are used only after execution; K remains the same historical conditional mean, not known physical truth.')
                    if policy in exact:ev['ideal_evidence']=self.directed(ctx,exact[policy],i,j)
                    zero=ev if policy=='no_dfs' else current_records[(repeat,'no_dfs')]['evaluation']
                    ev['evidence_increment']={name:ev['evidence'][name]-zero['evidence'][name] for name in ('J','bias','meter','clean_logodds','observed_logodds')}
                    if policy in exact and 'no_dfs' in exact:
                        BudgetAudit.verify_reported_evidence(self,row['evidence'],cp,current_pairs[(repeat,'no_dfs')],
                            base.native.arrays(folder/'exact_requested'/policy/'forecast_pair_evidence.npz'),base.native.arrays(folder/'exact_requested/no_dfs/forecast_pair_evidence.npz'),
                            base.native.arrays(folder/'exact_requested'/policy/'pair_evidence.npz'),base.native.arrays(folder/'exact_requested/no_dfs/pair_evidence.npz'))
                physical=row['physical_state'];ac=base.native.arrays(self.checked(physical))
                if 'time_slice' in physical:
                    a,b=physical['time_slice'];count=len(ac['voltage_complex_pu']);ac={k:v[a:b] if v.ndim and v.shape[0]==count else v for k,v in ac.items()}
                electrical,bv,bus=base.operation.electrical(self.operation,ac,obs,metadata,branches,buses)
                rec.update(audit_status='passed',requested=controls[policy]['requested'],delivered=delivered,requested_violations=controls[policy]['violations'],
                    delivered_violations=av,planned_cost=bill['planned'],settled_cost=bill['settled'],evaluation=ev,electrical=electrical,branches=bv,buses=bus)
                current_records[(repeat,policy)]=rec
            except Exception as exc:rec.update(audit_status='failed',error=repr(exc),traceback=traceback.format_exc())
            self.records.append(rec)


def run(source,out):
    source,out=Path(source).resolve(),Path(out).resolve()
    if out.exists():raise FileExistsError(out)
    if not(source/'complete.json').exists():raise ValueError('Completed run required')
    out.mkdir(parents=True);shutil.copyfile(__file__,out/'audit.py');audit=Audit()
    p=base.read(audit.file(source/'protocol.json'));entries=base.read(audit.checked(p['inputs']))
    complete=base.read(audit.file(source/'complete.json'));completion=base.read(audit.checked(complete['main']))
    audit.require(complete['source_unchanged'] and tuple(p['policies'])==POLICIES and len(entries)==1,'Source or policy protocol differs')
    expected_tasks={(e['id'],draw) for e in entries for draw in range(p['forecasts'])}
    passed_tasks={(r['id'],r['forecast_index']) for r in completion['receipts']}
    failed_tasks={(r['id'],r['forecast_index']) for r in completion['failed_tasks']}
    audit.require(not(passed_tasks&failed_tasks) and passed_tasks|failed_tasks==expected_tasks,'Forecast completion identities differ')
    audit.require(len(passed_tasks)==len(completion['receipts']) and len(failed_tasks)==len(completion['failed_tasks']),'Duplicate forecast receipts')
    for item in p['sources']:audit.file(source/'source_snapshot'/Path(item['path']).resolve().relative_to(base.native.ROOT),item['sha256'])
    meta=source/'source_snapshot/topology_identification_standalone/cases'/entries[0]['case'];metadata=base.read(audit.file(meta/'metadata.json'))
    with audit.file(meta/'branches.csv').open(encoding='utf-8-sig',newline='') as f:branches=list(csv.DictReader(f))
    with audit.file(meta/'buses.csv').open(encoding='utf-8-sig',newline='') as f:buses=list(csv.DictReader(f))
    for name in ('metadata.json','branches.csv','buses.csv'):
        current=base.native.ROOT/'topology_identification_standalone/cases'/entries[0]['case']/name
        audit.require(base.ref(current)['sha256']==base.ref(meta/name)['sha256'],'Physical truth tables changed since sealing')
    boundary=planning_boundary(source/'source_snapshot/rnj_wzzt_dr/experiments/rx_mean_guard_20261114/run.py')
    audit.require(boundary['passed'],'Solver input boundary differs from declared historical-data interface')
    own=dict(created_utc=datetime.now(timezone.utc).isoformat(),source=base.ref(__file__),saved_source=base.ref(out/'audit.py'),source_run=base.ref(source/'protocol.json'),
        expected_rows=p['forecasts']*p['repeats']*len(POLICIES),
        scope='Original and new historical-mean gates, score supports, physical resources, settlement and stored AC independently recomputed. No fitting, optimization or AC solve.',
        mean_scope='One fixed historical K per candidate, with fixed residual law; no parameter-uncertainty coverage.',
        planning_boundary=boundary,environment=dict(python=sys.version,numpy=np.__version__),
        helper_sources=[base.ref(sys.modules[name].__file__) for name in sorted({base.__name__,base.native.__name__,base.operation.__name__,
            base.direct_original.__module__,BudgetAudit.__module__,paired.__module__})])
    for item in own['helper_sources']:
        path=Path(item['path']);saved=out/'helper_snapshots'/path.relative_to(base.native.ROOT)
        saved.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,saved)
    base.write(out/'protocol.json',own)
    failed={(r['id'],r['forecast_index']) for r in completion['failed_tasks']}
    for entry in entries:
        for draw in range(p['forecasts']):
            folder=source/'fraction0.10'/entry['id']/f'forecast{draw:02d}'
            try:
                if (entry['id'],draw) in failed:raise ValueError('Original forecast failed')
                audit.forecast(folder,entry,p,metadata,branches,buses)
            except Exception as exc:audit.task_errors.append(dict(path=str(folder),error=repr(exc),traceback=traceback.format_exc()))
            base.write(out/'progress.json',dict(last_forecast=str(folder),rows=len(audit.records),task_errors=len(audit.task_errors),row_errors=sum(r['audit_status']!='passed' for r in audit.records)))
    good=len(audit.records)==own['expected_rows'] and not audit.task_errors and all(r['audit_status']=='passed' for r in audit.records)
    groups={policy:[r for r in audit.records if r['strategy']==policy] for policy in POLICIES}
    result=dict(status='passed' if good else 'failed',protocol=base.ref(out/'protocol.json'),expected_rows=own['expected_rows'],audited_rows=len(audit.records),
        original_failed_tasks=completion['failed_tasks'],original_quote_exceptions=completion['quote_exceptions'],original_execution_failed_rows=completion['execution_failed_rows'],
        task_errors=audit.task_errors,exact_diagnostic_failures=audit.exact_failures,counts=dict(audit.counts),maximum_absolute_errors=dict(audit.maxima),
        electrical_maximum_errors=dict(audit.operation.maxima),mean_lineage=getattr(audit,'mean_lineage',None),quotes=audit.quote_records,
        forecasts=getattr(audit,'forecast_records',[]),
        groups={policy:base.summarize(rows) for policy,rows in groups.items()},comparisons={policy:paired(groups['mean_guard'],groups[policy]) for policy in ('no_dfs','anchor','mean_guard_rotation')},
        per_forecast={str(draw):dict(groups={policy:base.summarize([r for r in rows if r['forecast']==draw]) for policy,rows in groups.items()},
            comparisons={policy:paired([r for r in groups['mean_guard'] if r['forecast']==draw],[r for r in groups[policy] if r['forecast']==draw]) for policy in ('no_dfs','anchor','mean_guard_rotation')}) for draw in range(p['forecasts'])},
        rotation_unmatched_fees=audit.fee_mismatches,records=audit.records)
    base.write(out/'report.json',result);base.write(out/'inputs_manifest.json',[dict(path=path,sha256=digest) for path,digest in audit.hashes.items()])
    base.write(out/'complete.json',dict(status=result['status'],report=base.ref(out/'report.json'),inputs=base.ref(out/'inputs_manifest.json'),source_unchanged=base.ref(__file__)['sha256']==own['source']['sha256']))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();result=run(args.run,args.out)
    print(dict(status=result['status'],rows=result['audited_rows'],expected=result['expected_rows'],task_errors=len(result['task_errors']),row_errors=sum(r['audit_status']!='passed' for r in result['records'])))
    raise SystemExit(0 if result['status']=='passed' else 1)
