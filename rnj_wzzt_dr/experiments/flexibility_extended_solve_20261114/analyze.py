"""Compare sealed 300-second quotations with their additional-budget replays.

Read-only array arithmetic: no optimizer, identifier, fitting, or AC solver is
imported. The saved original formula is recomputed independently of both solver
implementations. Results are repeat evaluations of already examined paths.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
from statistics import NormalDist
import traceback

import numpy as np
from scipy.special import ndtr

from ..flexibility_resource_validation_20261114 import analyze as prior
from ..flexibility_resource_validation_20261114 import operational_audit as operation


BUFFER = 1e-8
MATERIAL_YUAN = 1e-6


def write(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False,
        default=lambda v: v.tolist() if isinstance(v, np.ndarray) else v.item())+'\n', encoding='utf-8')


def canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(',', ':'))


class Checks:
    def __init__(self):
        self.counts = Counter(); self.errors = []; self.maximum = {}; self.inputs = {}

    def require(self, condition, name, context=''):
        self.counts[name] += 1
        if not condition:
            self.errors.append(dict(check=name, context=str(context)))

    def near(self, a, b, name, context='', tolerance=1e-8):
        a, b = np.asarray(a), np.asarray(b)
        valid = a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all()
        error = float(np.max(np.abs(a-b), initial=0.)) if valid else None
        if error is not None:
            self.maximum[name] = max(self.maximum.get(name, 0.), error)
        self.require(valid and error <= tolerance, name, context)

    def checked(self, item):
        actual = prior.ref(item['path']); self.inputs[actual['path']] = actual
        self.require(actual['sha256'] == item['sha256'], 'file_hash', actual['path'])
        return Path(item['path'])

    def report(self):
        return dict(passed=not self.errors, counts=dict(self.counts), errors=self.errors,
            maximum_absolute_errors=self.maximum)


def target_indices(history, sigma2, mass):
    ll = -(history-history.min())/(2*sigma2)
    ll -= np.log(np.exp(ll).sum())
    order = np.argsort(-ll, kind='stable')
    if mass == 1:
        return order.tolist()
    cutoff = min(int(np.searchsorted(np.cumsum(np.exp(ll[order])), mass)), len(order)-1)
    return [int(i) for i in order if ll[i] >= ll[order[cutoff]]-1e-12]


def resource_box(config, forecast, labels, base):
    """Independent hourly box, used solely to check eliminated s upper bounds."""
    count = len(labels); positions = {int(n): i for i,n in enumerate(labels)}
    clock = (8+np.arange(24)+.5) % 24
    rows = config['resources']; B = np.zeros((2*count, len(rows)))
    low = np.zeros((24,len(rows))); high = low.copy()
    for j,(r,unit) in enumerate(zip(rows,config['action_units'],strict=True)):
        node=positions[int(r['node_label'])]
        B[node+(count if unit=='kvar' else 0),j]=1.
        if unit=='kvar':
            cap=float(r['capacity_kvar']); available=np.ones(24,bool)
            hours=r.get('available_clock_hours')
            if hours is not None and hours!=list(range(24)):
                a,b=hours; available=((clock>=a)&(clock<b)) if a<b else ((clock>=a)|(clock<b))
            low[available,j]=-cap; high[available,j]=cap
        elif r['kind']=='pv':
            available=(clock>=10)&(clock<14)
            export=np.maximum(0.,-forecast[:,node]*base).reshape(24,4).min(1)
            high[:,j]=np.where(available,np.minimum(r['contract_kw'],export),0.)
        else:
            available=((clock>=18)|(clock<7)) if r['kind']=='ev' else np.ones(24,bool)
            low[available,j]=-r['contract_kw']; high[available,j]=r['contract_kw']
    return B,low,high


def direct_original(ctx, action, quote):
    """Rebuild full-prediction J, all directed means, and original gate margins."""
    s=ctx['score']; config=ctx['config']; entry=ctx['entry']
    H=np.asarray([p['H'] for p in s['parameters']],float)
    history=np.asarray([p['history_sse'] for p in s['parameters']],float)
    sigma2=float(s['sigma2']); K,N,_=H.shape
    forecast=np.asarray(ctx['planner']['forecast'],float)
    run_root=ctx['directory'].parents[2]
    metadata=prior.read(run_root/'source_snapshot/topology_identification_standalone/cases'/entry['case']/'metadata.json')
    base=1000*float(metadata['base_mva'])
    devices,resource_residuals,dp,dq=operation.resource_values(config,action,s['labels'],forecast[:,:N],base)
    power=forecast+np.c_[dp,dq]
    validation=prior.values(entry['files']['validation.npz']['path'])
    historical_power=np.c_[validation['P'],validation['Q']]
    if 'Y' in validation:
        y=validation['Y']
    else:
        y=validation['root'][:,None]**2-validation['V']**2
    prediction=np.einsum('tc,knc->ktn',historical_power,H)
    factor=1+np.sum(prediction*(y[None]-prediction),axis=1)/(np.sum(prediction**2,axis=1)+len(historical_power)*sigma2)
    means=factor[:,:,None]*H
    mass=float(quote.get('target_mass',.99)); targets=target_indices(history,sigma2,mass)
    pairs=[(i,j) for i in range(K) for j in range(i+1,K) if i in targets or j in targets]
    alpha=.1/max(1,K-1); z=NormalDist().inv_cdf(1-alpha)
    tie=max(1e-7,(s['cells']+forecast.shape[0]*N)*s['reference_mse']*s['validation_tie']/(2*sigma2))
    B,lower,upper=resource_box(config,forecast,s['labels'],base)
    pair_records=[]; guards=[]
    for i,j in pairs:
        D=H[i]-H[j]; d=power@D.T
        J=float(np.sum(d*d)/(2*sigma2)); h=float((history[j]-history[i])/(2*sigma2))
        h=min(v for t,v in ((i,h),(j,-h)) if t in targets)
        margin=h+J-z*np.sqrt(2*J)-tie
        failure=float(ndtr((tie-h-J)/np.sqrt(2*J))) if J>0 else float(h<=tie)
        c=h-tie-BUFFER
        if c>=z*z/2:
            kind='all_J'; required=small=None
        else:
            root=np.sqrt(z*z-2*c); required=float((z+root)**2/2)
            kind='lower_bound' if c<0 else 'two_intervals'
            small=None if c<0 else float(2*c*c/(z+root)**2)
        effect=D@B/base; positive=np.maximum(effect,0);negative=np.minimum(effect,0)
        d0=forecast@D.T
        lo=d0+np.repeat(lower,4,axis=0)@positive.T+np.repeat(upper,4,axis=0)@negative.T
        hi=d0+np.repeat(upper,4,axis=0)@positive.T+np.repeat(lower,4,axis=0)@negative.T
        upper_J=float(np.maximum(lo*lo,hi*hi).sum()/(2*sigma2))
        reduced_residual=(J-required) if kind=='lower_bound' else None
        pair_records.append(dict(pair=[i,j],J=J,historical_offset=h,unbuffered_margin=float(margin),
            original_buffered_margin=float(margin-BUFFER),failure_probability=failure,
            reduction_kind=kind,required_J=required,upper_small_branch_J=small,
            J_minus_required=reduced_residual,relaxed_resource_box_upper_J=upper_J,
            original_auxiliary_s=float(np.sqrt(J)),original_auxiliary_s_upper=float(np.sqrt(upper_J)*(1+1e-9)),
            auxiliary_upper_satisfied=bool(J<=upper_J+1e-7*max(1.,J,upper_J))))
        for left,right in ((i,j),(j,i)):
            DD=H[left]-H[right];E=means[left]-H[left]
            def expected(P):
                delta=P@DD.T; bias=P@E.T
                return float(np.sum(delta*delta)/(2*sigma2)+np.sum(bias*delta)/sigma2)
            guards.append(dict(pair=[left,right],gain=expected(power)-expected(forecast)))
    pair_ok=all(p['unbuffered_margin']>=-1e-7 and p['failure_probability']<=alpha+1e-9 for p in pair_records)
    guard_ok=all(g['gain']>=-1e-7 for g in guards)
    return dict(passed_original_acceptance=bool(pair_ok and guard_ok),resource_passed=True,
        targets=targets,pair_count=len(pair_records),guard_count=len(guards),sigma2=sigma2,tie=tie,buffer=BUFFER,
        z=z,alpha=alpha,minimum_original_buffered_margin=min((p['original_buffered_margin'] for p in pair_records),default=None),
        strict_buffered_nonnegative=all(p['original_buffered_margin']>=0 for p in pair_records),
        minimum_guard=min((g['gain'] for g in guards),default=None),
        minimum_J_minus_required=min((p['J_minus_required'] for p in pair_records if p['J_minus_required'] is not None),default=None),
        cost_yuan=float(sum(r['activation_yuan'] for r in devices)),resource_residuals=resource_residuals,
        pairs=pair_records,guards=guards,
        scope='Independent full-prediction arithmetic; unchanged direct acceptance tolerance; strict buffered signs reported separately. No production evidence or reduction routine called.')


def trace_summary(quote,checks):
    files=quote.get('diagnostic_files',{}); reference=files.get('progress')
    if not reference:
        return dict(status='unavailable',reason='No saved callback trajectory in this phase')
    path=checks.checked(reference)
    events=[json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
    sampled=[e for e in events if e.get('event')=='mip']
    runtime=[e['native_runtime_seconds'] for e in sampled]
    checks.require(all(b>=a for a,b in zip(runtime,runtime[1:])), 'trace_runtime_order',path)
    bounds=[e['bound'] for e in sampled if e.get('bound') is not None]
    incumbents=[e['incumbent'] for e in sampled if e.get('incumbent') is not None]
    milestones={}
    for t in (0,60,180,300,600,900):
        available=[e for e in sampled if e['native_runtime_seconds']<=t]
        milestones[str(t)]=available[-1] if available else None
    return dict(status=quote.get('trace_status','available'),source=reference,events=len(events),samples=len(sampled),
        first=sampled[0] if sampled else None,last=sampled[-1] if sampled else None,
        sampled_bound=prior.stats(bounds),sampled_incumbent=prior.stats(incumbents),
        bound_increases=sum(b>a+1e-8 for a,b in zip(bounds,bounds[1:])),
        bound_decreases=sum(b<a-1e-8 for a,b in zip(bounds,bounds[1:])),
        incumbent_improvements=sum(b<a-MATERIAL_YUAN for a,b in zip(incumbents,incumbents[1:])),
        last_sample_at_or_before_native_seconds=milestones,
        milestone_scope='Actual sampled times only; no interpolation or claim of a sample exactly at the target second.',
        logging_errors=quote.get('logging_errors',[]),native_log_configured=quote.get('native_log_configured'),
        final_raw_attributes=quote.get('native_attributes'))


def quote_fields(q):
    raw=q.get('native_attributes') or {}
    return dict(status=q.get('status'),feasible=q.get('feasible'),native_status=q.get('native_status'),source=q.get('source'),
        initial_name=q.get('initial',{}).get('name'),initial_cost_yuan=q.get('initial',{}).get('cost_yuan'),
        cost_yuan=q.get('cost_yuan'),used_lower_bound_yuan=q.get('lower_bound_yuan'),
        raw_ObjBound=raw.get('ObjBound'),raw_ObjBoundC=raw.get('ObjBoundC'),raw_ObjVal=raw.get('ObjVal'),
        raw_native_attributes_available=q.get('native_attributes') is not None,
        universal_nonnegative_cost_lower_bound=0.,relative_gap=q.get('relative_gap'),
        lower_bound_withdrawn=q.get('lower_bound_withdrawn'),global_optimality_verified=q.get('global_optimality_verified'),
        charged_total_seconds=q.get('charged_total_seconds'),native_runtime_seconds=raw.get('Runtime'),
        charged_total_budget_exceeded=q.get('charged_total_budget_exceeded'),shared_seed_seconds=q.get('shared_seed_seconds'),
        solver_seconds=q.get('seconds'),zero_fallback=q.get('zero_fallback'),rotation_same_fee=q.get('rotation_same_fee'),
        native_attribute_unavailable=q.get('native_attribute_unavailable',[]))


def quote_comparisons(before,after,checks):
    answer=[]
    for key in sorted(set(before['forecasts'])|set(after['forecasts'])):
        a,b=before['forecasts'].get(key),after['forecasts'].get(key)
        if a is None or b is None:
            checks.require(False,'quote_present',key);answer.append(dict(key=key,status='missing'));continue
        try:
            checks.require(canonical(a['score'])==canonical(b['score']),'entire_score_identity',key)
            checks.require(canonical(a['config'])==canonical(b['config']),'entire_resource_identity',key)
            for field in ('train.npz','validation.npz','101_native.json'):
                ar,br=a['entry']['files'][field],b['entry']['files'][field]
                checks.checked(ar);checks.checked(br)
                checks.require(ar['sha256']==br['sha256'],'historical_artifact_identity/'+field,key)
            for c in (a,b):
                seal=prior.read(c['directory']/'seal.json')
                for field in ('planner','plans','score','protocol'):checks.checked(seal[field])
            qa=prior.read(a['directory']/'plans.json')['symmetric_guard']
            qb=prior.read(b['directory']/'plans.json')['symmetric_guard']
            ua=a['planner']['symmetric_guard'];ub=b['planner']['symmetric_guard']
            checks.near(a['planner']['forecast'],b['planner']['forecast'],'forecast_exact',key,0.)
            da,db=direct_original(a,ua,qa),direct_original(b,ub,qb)
            for label,q,d in (('before',qa,da),('after',qb,db)):
                if q.get('feasible'):
                    checks.require(d['passed_original_acceptance'],'original_gate_guard_'+label,key)
                    checks.near(q['cost_yuan'],d['cost_yuan'],'direct_fee_'+label,key)
                checks.require(all(p['auxiliary_upper_satisfied'] for p in d['pairs']),'original_auxiliary_upper_'+label,key)
                if q.get('final'):
                    checks.near(q['final']['J'],[p['J'] for p in d['pairs']],'original_final_J_'+label,key,1e-6)
                    checks.near(q['final']['guard_gains'],[p['gain'] for p in d['guards']],'original_final_guards_'+label,key,1e-6)
                    checks.near(q['final']['minimum_pair_margin'],d['minimum_original_buffered_margin']+BUFFER,'original_final_margin_'+label,key,1e-6)
                saved=q.get('buffered_direct_audit')
                if saved:
                    checks.near(saved['minimum_original_buffered_margin'],d['minimum_original_buffered_margin'],'saved_buffered_margin',key,1e-6)
                    checks.near([p['J'] for p in saved['pairs']],[p['J'] for p in d['pairs']],'saved_direct_J',key,1e-6)
                    checks.near([p['gain'] for p in saved['guards']],[p['gain'] for p in d['guards']],'saved_direct_guards',key,1e-6)
                if q.get('pair_reduction'):
                    checks.require([p['kind'] for p in q['pair_reduction']]==[p['reduction_kind'] for p in d['pairs']], 'reduction_branch_identity',key)
                    for r,p in zip(q['pair_reduction'],d['pairs'],strict=True):
                        if r['lower_J'] is not None:checks.near(r['lower_J'],p['required_J'],'reduced_threshold',key,1e-9)
            seed_check=dict(scope='Original 300-second run; no continuation seed assertion')
            if after['protocol'].get('continuation_source_run'):
                old=prior.values(a['directory']/'shared_initial_points.npz')
                new=prior.values(b['directory']/'shared_initial_points.npz')
                expected=set(old)|{'source300_final_symmetric_guard'}
                checks.require(set(new)==expected,'retained_seed_names',key)
                for name in old:
                    if name in new:checks.near(old[name],new[name],'retained_seed_array',str(key)+'/'+name,0.)
                if 'source300_final_symmetric_guard' in new:
                    checks.near(ua,new['source300_final_symmetric_guard'],'source300_incumbent_array',key,0.)
                checks.require(qb.get('initial',{}).get('feasible',False),'extended_initial_feasible',key)
                checks.require(qb.get('initial',{}).get('cost_yuan',float('inf'))<=da['cost_yuan']+1e-7,'initial_not_worse_than_source300',key)
                chosen=qb.get('initial',{}).get('name')
                checks.require(chosen in new,'chosen_initial_in_saved_pool',key)
                if chosen in new:
                    chosen_direct=direct_original(b,new[chosen],qb)
                    checks.require(chosen_direct['passed_original_acceptance'],'chosen_initial_original_feasibility',key)
                    checks.near(qb['initial']['cost_yuan'],chosen_direct['cost_yuan'],'chosen_initial_fee',key)
                seed_check=dict(source_shared_arrays=len(old),expected_extended_arrays=len(expected),
                    scope='Exact old seed names/arrays plus the exact source300 incumbent; no additional generated point')
            improvement=da['cost_yuan']-db['cost_yuan']
            if qa.get('feasible'):
                checks.require(qb.get('feasible') and improvement>=-1e-7,'incumbent_retained_or_improved',key)
            if qb.get('lower_bound_yuan') is not None and qb.get('feasible'):
                lower=float(qb['lower_bound_yuan']);cost=db['cost_yuan']
                checks.require(-1e-12<=lower<=cost+1e-7,'used_bound_inside_feasible_interval',key)
                checks.near(qb['relative_gap'],(cost-lower)/max(cost,1e-12),'recomputed_gap',key,1e-7)
                raw=(qb.get('native_attributes') or {}).get('ObjBound')
                if raw is not None and np.isfinite(raw):
                    conflict=raw>cost+1e-7*max(1.,cost)
                    checks.require(bool(qb.get('lower_bound_withdrawn'))==conflict,'raw_bound_conflict_flag',key)
                    expected_lower=0. if conflict else max(0.,min(cost,raw))
                    checks.near(lower,expected_lower,'raw_to_used_bound',key)
            answer.append(dict(original=key[0],variant=key[1],forecast=key[2],status='checked',
                before=quote_fields(qa),after=quote_fields(qb),before_original=da,after_original=db,
                cost_improvement_yuan=improvement,material_improvement=improvement>MATERIAL_YUAN,
                worsened_cost=improvement < -1e-7,action_exactly_equal=bool(np.array_equal(ua,ub)),
                maximum_action_change=float(np.max(np.abs(ua-ub),initial=0.)),seed_check=seed_check,
                before_trajectory=trace_summary(qa,checks),after_trajectory=trace_summary(qb,checks)))
        except Exception as exc:
            checks.require(False,'quote_analysis_exception',str(key)+': '+repr(exc))
            answer.append(dict(key=key,status='analysis_error',error=repr(exc),traceback=traceback.format_exc()))
    return answer


def execution_comparison(before,after,variant,strategy):
    aa=prior.selected(before,variant,strategy);bb=prior.selected(after,variant,strategy)
    result=prior.paired_compare(bb,aa)
    key=lambda r:(r['_original'],r['_draw'],r['_repeat'])
    old={key(r):r for r in aa};new={key(r):r for r in bb};changes=[]
    for k in sorted(set(old)|set(new)):
        a,b=old.get(k),new.get(k)
        row=dict(original=k[0],forecast=k[1],repeat=k[2],strategy=strategy,variant=variant,
            before_status=None if a is None else a.get('status'),after_status=None if b is None else b.get('status'))
        if a is not None and b is not None and '_replayed_correct' in a and '_replayed_correct' in b:
            row.update(before_correct=a['_replayed_correct'],after_correct=b['_replayed_correct'],
                before_true_weight=a['_true_weight'],after_true_weight=b['_true_weight'],
                true_weight_change=b['_true_weight']-a['_true_weight'],
                before_fee=a['settled_fee_yuan'],after_fee=b['settled_fee_yuan'],
                settled_fee_change=b['settled_fee_yuan']-a['settled_fee_yuan'],
                before_evidence=a.get('_evidence_delta'),after_evidence=b.get('_evidence_delta'),
                evidence_change={name:b['_evidence_delta'][name]-a['_evidence_delta'][name]
                    for name in set(a.get('_evidence_delta',{}))&set(b.get('_evidence_delta',{}))})
            row['recovery_change']='gain' if row['after_correct'] and not row['before_correct'] else 'loss' if row['before_correct'] and not row['after_correct'] else 'same'
        else:row['recovery_change']='incomplete'
        changes.append(row)
    result.update(changed_recovery_paths=[r for r in changes if r['recovery_change']!='same'],
        true_weight_decreased=sum(r.get('true_weight_change',0)<-1e-8 for r in changes),
        observed_evidence_decreased=sum(r.get('evidence_change',{}).get('observed_logodds',0)<-1e-8 for r in changes),
        observed_evidence_change=prior.stats([r.get('evidence_change',{}).get('observed_logodds') for r in changes]),
        settled_fee_change=prior.stats([r.get('settled_fee_change') for r in changes]),per_path=changes)
    return result


def operational_summary(path,run,checks):
    if path is None:return dict(status='not_supplied',scope='Separate full electrical audit may be merged later')
    path=Path(path);report=prior.read(path/'report.json');protocol=prior.read(path/'protocol.json')
    checks.require(protocol['source_run']['sha256']==prior.ref(run['path']/'protocol.json')['sha256'],'operational_run_identity',path)
    checks.require(report['status']=='passed','operational_audit_passed',path)
    return dict(status=report['status'],report=prior.ref(path/'report.json'),overall=report['overall'],groups=report['groups'],
        maximum_absolute_errors=report['maximum_absolute_errors'],task_errors=report['task_errors'])


def report(pairs,truth_run,out,operational=(),self_test=False):
    out=Path(out).resolve()
    if out.exists():raise FileExistsError('Preserve analysis output: '+str(out))
    names=[p[0] for p in pairs]
    if len(names)!=len(set(names)):raise ValueError('Duplicate pair labels')
    op={name:(before,after) for name,before,after in operational}
    reference=prior.load_run(truth_run,'previous_verified_truth',[prior.OLD])
    if not reference['complete']:raise ValueError('Truth-label reference must be complete')
    truths,truth_counts=prior.truth_identity(reference)
    checks=Checks();outputs={};source_states=[]
    for name,before_path,after_path in pairs:
        bp,ap=Path(before_path).resolve(),Path(after_path).resolve()
        if self_test and bp!=ap:raise ValueError('Self-test requires each before path equal its after path')
        contexts=[]
        for label,path in [('before',bp),('after',ap)]:
            protocol=prior.read(path/'protocol.json');variants=protocol['placement_variants']
            run=prior.load_run(path,name+'/'+label,variants)
            if not run['complete']:raise ValueError('Complete run required: '+str(path))
            checks.require(run['complete'].get('source_unchanged',False),'source_locked',path)
            checks.checked(protocol['inputs']);contexts.append(run)
            source_states.append(dict(label=name+'/'+label,protocol=prior.ref(path/'protocol.json'),
                completion=prior.ref(path/'complete.json'),task_completion=run['task_completion']))
        before,after=contexts
        for field in ('seed','forecasts','repeats','placement_variants','modes','training_rows','validation_rows','new_rows'):
            checks.require(before['protocol'].get(field)==after['protocol'].get(field),'protocol_identity/'+field,name)
        variants=sorted(set(before['protocol']['placement_variants'])|set(after['protocol']['placement_variants']))
        identity={v:prior.identical_paths(before,after,v,v) for v in variants}
        for v,item in identity.items():checks.require(item['passed'],'same_paths/'+v,name)
        arithmetic={label:prior.replay_rows(run,truths) for label,run in [('before',before),('after',after)]}
        for label,item in arithmetic.items():
            bad=sum(value for key,value in item['counts'].items() if 'mismatch' in key or 'missing' in key)
            checks.require(bad==0,'native_replay/'+label,name)
            for key,error in item['maximum_errors'].items():
                tolerance=1e-6 if 'logprob' in key else 1e-8
                checks.require(error<=tolerance,'arithmetic/'+key,name+'/'+label)
        summaries={label:prior.summarize(run) for label,run in [('before',before),('after',after)]}
        comparisons={v:{strategy:execution_comparison(before,after,v,strategy) for strategy in prior.POLICIES} for v in variants}
        quotes=quote_comparisons(before,after,checks)
        if self_test:
            checks.require(all(q.get('status')=='checked' and q['action_exactly_equal'] and q['cost_improvement_yuan']==0 for q in quotes),'self_test_quotes',name)
            for v,data in comparisons.items():
                for strategy,c in data.items():
                    checks.require(c['added_recoveries']==c['lost_recoveries']==c['incomplete_pairs']==0,'self_test_recovery',name+'/'+v+'/'+strategy)
                    checks.require(all(r.get('true_weight_change')==0 and r.get('settled_fee_change')==0 and all(value==0 for value in r.get('evidence_change',{}).values()) for r in c['per_path']),'self_test_scalar_changes',name+'/'+v+'/'+strategy)
        paths=op.get(name,(None,None))
        electrical={label:operational_summary(path,run,checks) for label,path,run in zip(('before','after'),paths,(before,after),strict=True)}
        outputs[name]=dict(before=prior.ref(bp/'protocol.json'),after=prior.ref(ap/'protocol.json'),
            groups=summaries,comparisons=comparisons,quotation_comparisons=quotes,
            stream_identity=identity,independent_arithmetic=arithmetic,operational=electrical,
            errors_and_failures={label:dict(failed_tasks=run['task_completion'].get('failed_tasks',[]),
                quote_exceptions=run['task_completion'].get('quote_exceptions'),execution_failed_rows=run['task_completion'].get('execution_failed_rows'))
                for label,run in [('before',before),('after',after)]})
    out.mkdir(parents=True)
    sources=[]
    for module in (__file__,prior.__file__,operation.__file__):
        path=Path(module);destination=out/'analysis_sources'/path.parent.name/path.name
        destination.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,destination)
        sources.append(dict(original=prior.ref(path),saved=prior.ref(destination)))
    answer=dict(created_utc=datetime.now(timezone.utc).isoformat(),status='passed' if not checks.errors else 'failed',
        self_test=self_test,sources=sources,source_runs=source_states,checks=checks.report(),pairs=outputs,
        truth_reference=prior.ref(Path(truth_run)/'protocol.json'),truth_label_evidence=truth_counts,
        scope='Additional optimization and replay of already evaluated paths; no new independent confirmation. Same physical day and error arrays. Native full-bank recovery and representative weights remain separate.',
        interpretation='A lower verified procurement cost or tighter bound does not imply improved identification. All adverse flips and incomplete paths remain recorded.',
        bound_scope='Used and raw bounds are separate; missing old callback/native fields remain unavailable. Feasibility is independently checked under original direct numerical tolerances; strict buffered residual signs are also retained.',
        execution_scope='Only saved observations, actions and full-state arrays were read; no new fitting, optimization or AC solve.')
    write(out/'inputs_manifest.json',list(checks.inputs.values()));write(out/'REPORT.json',answer)
    lines=['# Additional-budget comparison','',answer['scope'],'',f"Independent checks: {answer['status']}; self-test: {self_test}.",
        '', '| Pair | Resource | Policy | Before recovery | After recovery | Gains/losses | True ≥.9 before/after | Wrong ≥.9 before/after | Mean fee before/after |',
        '|---|---|---|---:|---:|---:|---:|---:|---:|']
    for name,p in outputs.items():
        for v,cs in p['comparisons'].items():
            for strategy,c in cs.items():
                a,b=p['groups']['before'][v][strategy],p['groups']['after'][v][strategy]
                lines.append(f"| {name} | {v} | {strategy} | {a['native_correct']}/{a['planned_paths']} | {b['native_correct']}/{b['planned_paths']} | {c['added_recoveries']}/{c['lost_recoveries']} | {a['true_weight_at_least_90']}/{b['true_weight_at_least_90']} | {a['wrong_weight_at_least_90']}/{b['wrong_weight_at_least_90']} | {a['settled_fee_yuan']['mean']}/{b['settled_fee_yuan']['mean']} |")
    lines+=['','## Quotes','', '| Pair | Resource / forecast | Before cost | After cost | Used lower bound | Gap | Source | Seconds | Buffered margin | Minimum guard |',
        '|---|---|---:|---:|---:|---:|---|---:|---:|---:|']
    for name,p in outputs.items():
        for q in p['quotation_comparisons']:
            if q['status']!='checked':lines.append(f"| {name} | {q.get('key')} | analysis error | | | | | | | |");continue
            a,b=q['before'],q['after'];d=q['after_original']
            lines.append(f"| {name} | {q['variant']} / {q['forecast']} | {a['cost_yuan']} | {b['cost_yuan']} | {b['used_lower_bound_yuan']} | {b['relative_gap']} | {b['source']} | {b['charged_total_seconds']} | {d['minimum_original_buffered_margin']} | {d['minimum_guard']} |")
    lines+=['',answer['interpretation'],'',answer['bound_scope'],'',f"Failures in independent analysis: {len(checks.errors)}. All status counts, changed paths, original constraints, trajectories, and optional electrical summaries are in REPORT.json.",'']
    (out/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    write(out/'complete.json',dict(status=answer['status'],report=prior.ref(out/'REPORT.json'),source_unchanged=prior.ref(__file__)['sha256']==sources[0]['original']['sha256']))
    return answer


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pair',nargs=3,action='append',required=True,metavar=('LABEL','BEFORE','AFTER'))
    parser.add_argument('--truth-run',type=Path,default=Path('outputs/flexibility_reaudit_long180_20261114_v1'))
    parser.add_argument('--operational',nargs=3,action='append',default=[],metavar=('LABEL','BEFORE_AUDIT','AFTER_AUDIT'))
    parser.add_argument('--out',type=Path,required=True);parser.add_argument('--self-test',action='store_true')
    args=parser.parse_args();result=report(args.pair,args.truth_run,args.out,args.operational,args.self_test)
    print(json.dumps(dict(status=result['status'],pairs=list(result['pairs']),errors=len(result['checks']['errors']),output=str(args.out.resolve())),ensure_ascii=False))
    raise SystemExit(0 if result['status']=='passed' else 1)
