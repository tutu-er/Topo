"""Complete paired readout of independently audited frozen-R/X guard runs.

Only saved JSON/NPZ data are read. No simulator, optimizer or identifier imports.
Additional response diagnostics are post-processing, not another experiment.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np


class Reader:
    def __init__(self): self.refs={}
    def checked(self, item):
        path=Path(item['path'] if isinstance(item,dict) else item).resolve()
        if str(path) not in self.refs:
            self.refs[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
        if isinstance(item,dict) and item.get('sha256') != self.refs[str(path)]:
            raise ValueError('Changed input: '+str(path))
        return path
    def json(self,item): return json.loads(self.checked(item).read_text(encoding='utf-8'))
    def lines(self,item):
        with self.checked(item).open(encoding='utf-8') as stream:
            return [json.loads(line) for line in stream if line.strip()]
    def arrays(self,item):
        with np.load(self.checked(item)) as archive:
            return {name:archive[name].copy() for name in archive.files}
    def unchanged(self):
        return all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in self.refs.items())


def scalar(value):
    if value is None: return None
    value=float(value)
    return value if np.isfinite(value) else None


def stats(values):
    values=[float(x) for x in values if x is not None and np.isfinite(x)]
    return dict(n=len(values),mean=float(np.mean(values)) if values else None,
        minimum=min(values) if values else None,maximum=max(values) if values else None)


def key(row): return row['id'],int(row['forecast']),int(row['repeat']),row['strategy']


def calibrated_means(score, validation):
    """Replay the frozen historical row-scale formula, without parameter fitting."""
    H=np.asarray([p['H'] for p in score['parameters']],float)
    power=np.c_[validation['P'],validation['Q']]
    prediction=np.einsum('tc,knc->ktn',power,H)
    factor=1+np.sum(prediction*(validation['Y'][None]-prediction),axis=1)/(
        np.sum(prediction*prediction,axis=1)+len(power)*score['sigma2'])
    return factor[:,:,None]*H


class Postprocess:
    def __init__(self,reader,entries,coverage,run_root,baseline_only=False):
        self.reader,self.entries,self.coverage=reader,entries,coverage
        self.run_root,self.baseline_only=run_root,baseline_only
        self.contexts={};self.zeros={};self.maximum_error=0.
    def context(self,row):
        tag=row['id']
        if tag in self.contexts:return self.contexts[tag]
        e=self.entries[tag];s=self.reader.json(e['score']);labels=s['labels'];n=len(labels)
        H=np.asarray([p['H'] for p in s['native_bank']],float)
        hist=np.asarray([p['history_sse'] for p in s['native_bank']],float)
        rep_ids=[p['index'] for p in s['parameters']]
        val=self.reader.arrays(e['files']['validation.npz'])
        K=calibrated_means(s,val)
        prior=Path(e['prior_directory']) if 'prior_directory' in e else Path(row['source']).parents[1]
        oldseal=self.reader.json(e['prior_seal'] if 'prior_seal' in e else prior/'seal.json')
        planner=self.reader.arrays(oldseal['planner']);forecast=planner['forecast']
        config=self.reader.json(oldseal['resource_case'])
        targets=np.zeros((2*n,len(config['resources'])))
        for j,(r,unit) in enumerate(zip(config['resources'],config['action_units'],strict=True)):
            node=labels.index(r['node_label'])
            if unit=='kW':targets[node,j]=1/e['base_kw'];targets[n+node,j]=r.get('q_over_p',0.)/e['base_kw']
            elif unit=='kvar':targets[n+node,j]=1/e['base_kw']
            else:raise ValueError('Unknown resource unit')
        cov=self.coverage[tag]
        true=list(map(int,cov['true_native_indices']))
        historical=int(cov.get('history_native_selected',s['native_index']))
        family=s['native_bank'][historical]['topology_id']
        others=[j for j,p in enumerate(s['native_bank']) if p['topology_id']!=family]
        rival=min(others,key=lambda j:(hist[j],j)) if others else None
        true_target=min(true,key=lambda j:(hist[j],j)) if true else None
        wrong=[j for j in range(len(H)) if j not in true]
        geometry=None
        if not self.baseline_only:
            seal=self.reader.json(self.run_root/'cases'/tag/'seal.json')
            metadata=self.reader.json(seal['geometry'])
            geometry=(metadata,self.reader.arrays(metadata['arrays']))
        c=dict(entry=e,score=s,H=H,hist=hist,sigma2=float(s['sigma2']),n=n,labels=labels,
            representatives=rep_ids,K=K,prior=prior,forecast=forecast,map=targets,
            true=true,wrong=wrong,historical=historical,rival=rival,true_target=true_target,
            other_families=others,geometry=geometry)
        self.contexts[tag]=c
        return c
    def zero(self,c,repeat):
        k=c['entry']['id'],repeat
        if k not in self.zeros:
            ob=self.reader.arrays(c['prior']/f'execution{repeat:02d}'/'no_dfs'/'observations.npz')
            self.zeros[k]=(ob,self.values(c,ob))
        return self.zeros[k]
    def values(self,c,ob):
        X=np.c_[ob['true_P'],ob['true_Q']]
        Y=ob['true_root'][:,None]**2-ob['true_V']**2
        Xm=np.c_[ob['P'],ob['Q']]
        clean=c['hist']+np.square(Y[None]-np.einsum('tc,knc->ktn',X,c['H'])).sum((1,2))
        observed=c['hist']+np.square(ob['Y'][None]-np.einsum('tc,knc->ktn',Xm,c['H'])).sum((1,2))
        error=float(np.max(abs(observed-ob['native_total_sse'])))
        self.maximum_error=max(self.maximum_error,error)
        if error>1e-10:raise ValueError('Saved full-native SSE differs from direct report replay')
        return dict(X=X,Y=Y,clean=clean,observed=observed)
    @staticmethod
    def pair(c,values,i,j,forecast):
        D=c['H'][i]-c['H'][j];scale=c['sigma2']
        h=float((c['hist'][j]-c['hist'][i])/(2*scale))
        answer=dict(target=i,rival=j,target_is_true=i in c['true'],
            historical_logodds=h,
            forecast_J=float(np.square(forecast@D.T).sum()/(2*scale)),
            actual_J=float(np.square(values['X']@D.T).sum()/(2*scale)),
            clean_logodds=float((values['clean'][j]-values['clean'][i])/(2*scale)),
            observed_logodds=float((values['observed'][j]-values['observed'][i])/(2*scale)))
        if i in c['representatives']:
            K=c['K'][c['representatives'].index(i)]
            def mean_odds(X):
                return h+float((np.square(X@(K-c['H'][j]).T).sum()-
                    np.square(X@(K-c['H'][i]).T).sum())/(2*scale))
            answer.update(forecast_K_logodds=mean_odds(forecast),
                actual_input_K_logodds=mean_odds(values['X']),K_available=True,K_unavailable_reason=None)
        else:
            answer.update(forecast_K_logodds=None,actual_input_K_logodds=None,K_available=False,
                K_unavailable_reason='Fixed native target has no representative historical K; no new K is fitted or substituted.')
        return answer
    def budgets(self,c,u):
        if c['geometry'] is None:return None
        metadata,arrays=c['geometry'];x=u.ravel();answer=[]
        for g in metadata['guards']:
            value=float(x@arrays[g['matrix_key']]@x);limit=float(g['limit'])
            answer.append(dict(id=g['id'],kind=g['kind'],value=value,limit=limit,
                passed_with_display_tolerance=bool(value<=limit+1e-9*max(1.,limit)),
                scope='Report replay of audited geometry; optimizer acceptance uses its separately recorded roundoff check'))
        return answer
    def enrich(self,row,origin):
        answer=dict(id=row['id'],forecast=int(row['forecast']),repeat=int(row['repeat']),
            strategy=row['strategy'],data_origin=origin,audit_status=row['audit_status'],
            original_status=row['original_status'],source=row.get('source'),
            quote_zero_fallback=bool(row.get('quote_zero_fallback',False)),
            quote_new_gate_feasible=row.get('quote_new_gate_feasible'),
            is_rotation=bool(row.get('is_rotation',row['strategy'].endswith('_rotation'))),
            evaluation=dict(row.get('evaluation',{})),planned_cost=row.get('planned_cost'),settled_cost=row.get('settled_cost'),
            voltage_screen_passed=row.get('voltage_screen_passed'))
        if row['audit_status']!='passed':
            answer['error']=row.get('error');return answer
        answer['quote_fields_scope']='Parent procurement quote; rotation gate/budget success is not asserted' if answer['is_rotation'] else 'This procurement quote'
        c=self.context(row);ob=self.reader.arrays(Path(row['source'])/'observations.npz')
        original=self.reader.json(Path(row['source'])/'receipt.json')
        for field in ('native_full_support_missing','native_full_support_extra'):
            answer['evaluation'][field]=original.get(field)
        zero,zvalues=self.zero(c,int(row['repeat']));v=self.values(c,ob)
        u=ob['requested_action'];d=ob['delivered_action'];repeat=len(c['forecast'])//len(u)
        forecast=c['forecast']+np.repeat(u@c['map'].T,repeat,axis=0)
        def fixed_pair(i,j):
            if i is None or j is None:return None
            now=self.pair(c,v,i,j,forecast);before=self.pair(c,zvalues,i,j,c['forecast'])
            fields=('forecast_J','actual_J','forecast_K_logodds','actual_input_K_logodds','clean_logodds','observed_logodds')
            now['baseline']={k:before[k] for k in fields}
            now['increment']={k:now[k]-before[k] if now[k] is not None and before[k] is not None else None for k in fields}
            inc=now['increment']
            if now['K_available']:
                gaps=dict(input_and_delivery=inc['actual_input_K_logodds']-inc['forecast_K_logodds'],
                    response_model=inc['clean_logodds']-inc['actual_input_K_logodds'],
                    meter=inc['observed_logodds']-inc['clean_logodds'])
                total=inc['observed_logodds']-inc['forecast_K_logodds']
                residual=sum(gaps.values())-total
                if abs(residual)>1e-9*max(1.,abs(total),sum(abs(x) for x in gaps.values())):
                    raise ValueError('Forecast/actual logodds gap decomposition failed')
                now['forecast_actual_gap']=dict(components=gaps,total_observed_minus_forecast=total,
                    additivity_residual=residual,
                    scope='Fixed historical K and fixed candidate pair; input/delivery includes forecast error and delivered response. No future-fitted parameters.')
            else:now['forecast_actual_gap']=None
            return now
        historic=fixed_pair(c['historical'],c['rival'])
        offline_pair=row.get('evaluation',{}).get('pair')
        offline=fixed_pair(*offline_pair) if offline_pair else None
        whole={}
        for name,target,rivals in (('historical_target',c['historical'],c['other_families']),
                                   ('fixed_true_target',c['true_target'],c['wrong'])):
            if target is None or not rivals:whole[name]=None;continue
            item=dict(target=target,target_is_true=target in c['true'])
            def minimum_contrast(law_now,law_before):
                rival=min(rivals,key=lambda j:(law_now[j],j))
                old=min(rivals,key=lambda j:(law_before[j],j))
                margin=float((law_now[rival]-law_now[target])/(2*c['sigma2']))
                before=float((law_before[old]-law_before[target])/(2*c['sigma2']))
                return dict(minimum_logodds=margin,strongest_competitor=rival,
                    baseline_strongest_competitor=old,baseline_minimum_logodds=before,
                    minimum_logodds_increment=margin-before,competitor_changed=bool(rival!=old))
            for law in ('clean','observed'):
                item[law]=minimum_contrast(v[law],zvalues[law])
            if target in c['representatives']:
                K=c['K'][c['representatives'].index(target)]
                def expected_score(X):
                    residual=np.einsum('tc,knc->ktn',X,K[None]-c['H'])
                    return c['hist']+np.square(residual).sum((1,2))
                for law,X,X0 in (('forecast_K',forecast,c['forecast']),
                                  ('actual_input_K',v['X'],zvalues['X'])):
                    item[law]=minimum_contrast(expected_score(X),expected_score(X0))
                item['K_unavailable_reason']=None
            else:
                item.update(forecast_K=None,actual_input_K=None,
                    K_unavailable_reason='Fixed native target has no representative historical K; no substitute mean is fitted.')
            item['K_minimum_scope']='Minimum over competitors of expected pair logodds for one fixed historical K; not the expectation of a random minimum.'
            whole[name]=item
        whole['all_native_clean_sse_minimizer']=int(np.argmin(v['clean']))
        whole['audited_observed_winner']=int(row['evaluation']['native_selected_index'])
        baseline_winner=int(np.argmin(zvalues['observed']))
        whole['baseline_observed_sse_minimizer']=baseline_winner
        whole['observed_winner_changed_from_no_dfs']=bool(whole['audited_observed_winner']!=baseline_winner)
        if c['true'] and c['wrong']:
            whole['best_true_family_vs_best_wrong_observed']=float((min(v['observed'][c['wrong']])-min(v['observed'][c['true']]))/(2*c['sigma2']))
        audited_full=row['evaluation'].get('full_native_diagnostic')
        if audited_full and whole['fixed_true_target']:
            if audited_full['true_history_fixed_target']!=whole['fixed_true_target']['target']:
                raise ValueError('Audited full-bank fixed true target differs')
            for law in ('clean','observed'):
                if not np.isclose(audited_full[law+'_worst_true_logodds'],whole['fixed_true_target'][law]['minimum_logodds'],atol=1e-9,rtol=1e-10):
                    raise ValueError('Audited full-bank minimum margin differs')
        deltaX=v['X']-zvalues['X'];deltaY=v['Y']-zvalues['Y']
        errorH=np.square(deltaY[None]-np.einsum('tc,knc->ktn',deltaX,c['H'][c['representatives']])).mean((1,2))/c['sigma2']
        errorK=np.square(deltaY[None]-np.einsum('tc,knc->ktn',deltaX,c['K'])).mean((1,2))/c['sigma2']
        for field,value in (('actual_H_increment_error_over_sigma2',errorH),('actual_K_increment_error_over_sigma2',errorK)):
            if field in row and not np.allclose(row[field],value,atol=1e-9,rtol=1e-9):
                raise ValueError('Audited response-error field differs from report replay')
        response=[dict(representative=i,native_index=n,is_true=n in c['true'],
            H_mean_square_over_sigma2=float(errorH[i]),K_mean_square_over_sigma2=float(errorK[i])) for i,n in enumerate(c['representatives'])]
        unit_p=[i for i,r in enumerate(c['map'].T) if np.any(r[:c['n']])]
        unit_q=[i for i in range(u.shape[1]) if i not in unit_p]
        quantities=dict(requested_P_kwh=float(abs(u[:,unit_p]).sum()),requested_Q_kvarh=float(abs(u[:,unit_q]).sum()),
            delivered_P_kwh=float(abs(d[:,unit_p]).sum()),delivered_Q_kvarh=float(abs(d[:,unit_q]).sum()))
        for label,action in (('requested',u),('delivered',d)):
            net=np.sum(action[:,unit_q],axis=1);total=float(abs(action[:,unit_q]).sum())
            quantities[label+'_net_Q_rms']=float(np.sqrt(np.mean(net*net)))
            quantities[label+'_abs_net_Q_kvarh']=float(abs(net).sum())
            quantities[label+'_absnet_to_total_Q_ratio']=float(abs(net).sum()/total) if total>0 else None
        answer.update(historical_fixed_pair=historic,offline_true_fixed_pair=offline,whole_native_bank=whole,
            actual_response_error_by_representative=response,
            request_budget=self.budgets(c,u),delivery_budget=self.budgets(c,d),
            quantities=quantities,net_Q_scope='Signed sum of pure-Q resource columns across nodes; hourly actions. RMS unit kvar; zero-total-Q absnet ratio is null. Diagnostic only.',
            voltage_range=[row['electrical']['all_bus_voltage']['minimum'],row['electrical']['all_bus_voltage']['maximum']],
            postprocessing_scope='Historical fixed pairs and full-native contrasts recomputed from sealed arrays; true labels only identify offline evaluation references. H/K scores and parameters unchanged.')
        return answer


def recovered(row):return row['audit_status']=='passed' and bool(row['evaluation'].get('native_correct'))


def paired(rows,policy,reference):
    a={(r['id'],r['forecast'],r['repeat']):r for r in rows if r['strategy']==policy}
    b={(r['id'],r['forecast'],r['repeat']):r for r in rows if r['strategy']==reference}
    if set(a)!=set(b):raise ValueError('Different paired denominators: '+policy+' / '+reference)
    return dict(n=len(a),reference=reference,new_recoveries=sum(recovered(a[k]) and not recovered(b[k]) for k in a),
        lost_recoveries=sum(recovered(b[k]) and not recovered(a[k]) for k in a),
        changed_winner=sum(a[k]['evaluation'].get('native_selected_index')!=b[k]['evaluation'].get('native_selected_index') for k in a))


def summarize(rows):
    good=[r for r in rows if r['audit_status']=='passed'];n=len(rows)
    result=dict(n=n,completed=len(good),recorded_failures=n-len(good),recovered=sum(recovered(r) for r in rows),
        full_support_recovered=sum(bool(r['evaluation'].get('native_full_support_exact')) for r in good),
        full_support_exact_count=sum(bool(r['evaluation'].get('native_full_support_exact')) for r in good),
        internal_missing=stats([r['evaluation'].get('native_missing') for r in good]),
        internal_extra=stats([r['evaluation'].get('native_extra') for r in good]),
        full_support_missing=stats([r['evaluation'].get('native_full_support_missing') for r in good]),
        full_support_extra=stats([r['evaluation'].get('native_full_support_extra') for r in good]),
        wrong_high=sum(bool(r['evaluation'].get('wrong_high')) for r in good),
        true_high=sum(bool(r['evaluation'].get('true_high')) for r in good),
        p_true=stats([r['evaluation'].get('p_true') for r in good]),
        zero_fallback_rows=sum(r['quote_zero_fallback'] for r in rows),
        audited_winner_counts=dict(Counter(str(r['evaluation'].get('native_selected_index')) for r in good)),
        data_origin_counts=dict(Counter(r['data_origin'] for r in rows)),
        planned_total_yuan=stats([r['planned_cost']['total_yuan'] for r in good]),
        settled_total_yuan=stats([r['settled_cost']['total_yuan'] for r in good]),
        settled_activation_yuan=stats([r['settled_cost']['activation_yuan'] for r in good]),
        settled_availability_yuan=stats([r['settled_cost']['availability_yuan'] for r in good]),
        quantities={k:stats([r['quantities'][k] for r in good]) for k in ('requested_P_kwh','requested_Q_kvarh','delivered_P_kwh','delivered_Q_kvarh',
            'requested_net_Q_rms','requested_abs_net_Q_kvarh','requested_absnet_to_total_Q_ratio',
            'delivered_net_Q_rms','delivered_abs_net_Q_kvarh','delivered_absnet_to_total_Q_ratio')},
        voltage_range=None if not good else [min(r['voltage_range'][0] for r in good),max(r['voltage_range'][1] for r in good)],
        voltage_failed_rows=sum(not r['voltage_screen_passed'] for r in good),
        failure_cost_imputed=False)
    for pairname in ('historical_fixed_pair','offline_true_fixed_pair'):
        points=[r[pairname] for r in good if r.get(pairname)]
        identities=sorted({(p['target'],p['rival']) for p in points})
        result[pairname]=dict(pairs=[list(p) for p in identities],n=len(points),
            K_available_count=sum(p['K_available'] for p in points),
            K_unavailable_reasons=dict(Counter(p['K_unavailable_reason'] for p in points if not p['K_available'])),
            increments={k:stats([p['increment'][k] for p in points]) for k in ('forecast_J','actual_J','forecast_K_logodds','actual_input_K_logodds','clean_logodds','observed_logodds')},
            negative_observed_increment_count=sum(p['increment']['observed_logodds'] < -1e-8 for p in points))
        gaps=[p['forecast_actual_gap'] for p in points if p.get('forecast_actual_gap')]
        result[pairname]['forecast_actual_gap']=dict(n=len(gaps),
            components={k:stats([p['components'][k] for p in gaps]) for k in ('input_and_delivery','response_model','meter')},
            total_observed_minus_forecast=stats([p['total_observed_minus_forecast'] for p in gaps]),
            maximum_absolute_additivity_residual=max((abs(p['additivity_residual']) for p in gaps),default=None))
    for target in ('historical_target','fixed_true_target'):
        data=[r['whole_native_bank'][target] for r in good if r.get('whole_native_bank',{}).get(target)]
        laws={}
        for law in ('forecast_K','actual_input_K','clean','observed'):
            available=[p[law] for p in data if p.get(law) is not None]
            laws[law]=dict(n=len(available),minimum_logodds=stats([p['minimum_logodds'] for p in available]),
                baseline_minimum_logodds=stats([p['baseline_minimum_logodds'] for p in available]),
                minimum_logodds_increment=stats([p['minimum_logodds_increment'] for p in available]),
                strongest_competitor_counts=dict(Counter(str(p['strongest_competitor']) for p in available)),
                changed_competitor_count=sum(p['competitor_changed'] for p in available)) if available or law in ('clean','observed') else None
        laws['K_unavailable_reasons']=dict(Counter(p['K_unavailable_reason'] for p in data if p.get('K_unavailable_reason')))
        result['whole_bank_'+target]=laws
    errors=defaultdict(list)
    for r in good:
        for item in r['actual_response_error_by_representative']:errors[item['native_index']].append(item)
    result['actual_response_errors']={str(k):dict(is_true=group[0]['is_true'],
        H_mean_square_over_sigma2=stats([g['H_mean_square_over_sigma2'] for g in group]),
        K_mean_square_over_sigma2=stats([g['K_mean_square_over_sigma2'] for g in group])) for k,group in errors.items()}
    for which in ('request_budget','delivery_budget'):
        grouped=defaultdict(list)
        for r in good:
            for value in r.get(which) or []:grouped[value['id']].append(value)
        result[which]={k:dict(value=stats([r['value'] for r in g]),limit=g[0]['limit'],
            passed_display_count=sum(r['passed_with_display_tolerance'] for r in g)) for k,g in grouped.items()}
    return result


def quote_summary(quotes):
    return dict(n=len(quotes),status_counts=dict(Counter(str(q.get('status')) for q in quotes)),
        native_status_counts=dict(Counter(str(q.get('native_status')) for q in quotes)),
        source_counts=dict(Counter(str(q.get('source')) for q in quotes)),
        feasible=sum(bool(q.get('new_gate_feasible',q.get('feasible'))) for q in quotes),
        zero_fallback=sum(bool(q.get('zero_fallback')) for q in quotes),
        lower_bound_withdrawn=sum(bool(q.get('lower_bound_withdrawn',q.get('certificate',{}).get('lower_bound_withdrawn'))) for q in quotes),
        bound_trajectory_conflict_count=sum(bool((q.get('certificate',{}).get('native_bound_history') or {}).get('recorded_bound_conflict_with_verified_buffered_point')) for q in quotes),
        audit_global_optimality_supported_count=sum(q.get('certificate',{}).get('audit_global_optimality_supported') is True for q in quotes),
        feasible_gap=stats([q.get('relative_gap',q.get('certificate',{}).get('relative_gap')) for q in quotes if q.get('new_gate_feasible')]),
        actual_quote_seconds=stats([q.get('quote_wall_seconds',q.get('seconds')) for q in quotes]),
        rows=quotes)


def quote_details(reader,quotes,entries,post,run_root,original_audit,baseline_only):
    """Preserve audited bounds and add source-labelled wall-clock accounting."""
    current=[];prior=[];timing=[]
    if 'quotes' in original_audit:
        prior=[dict(q,data_origin='reused_baseline_quote') for q in reader.lines(original_audit['quotes'])
            if q['id'] in entries]
    bycase=defaultdict(list)
    for q in quotes:bycase[q['id']].append(q)
    for tag in entries:
        c=post.contexts[tag];oldpath=c['prior']
        anchor=reader.json(oldpath/'anchor_plan.json')
        anchor_lineage=reader.json(oldpath/'anchor.json')
        shared=reader.json(oldpath/'shared_initial_receipt.json')
        prior.append(dict(anchor,id=tag,forecast=0,policy='anchor',data_origin='reused_baseline_quote',
            source_receipt=str(oldpath/'anchor_plan.json')))
        t=dict(id=tag,old_shared_pool_generation_seconds=shared.get('seconds'),
            old_anchor_quote_wall_seconds=anchor.get('quote_wall_seconds',anchor.get('seconds')),
            old_shared_pool_and_anchor_seconds=shared.get('shared_preparation_seconds',anchor_lineage.get('shared_seconds')),
            old_shared_work_reexecuted=False,
            meaning='Old pool/anchor costs are sunk. Shared geometry and prediction-model construction occur once per case; do not sum them across quotes.')
        if not baseline_only:
            seal=reader.json(run_root/'cases'/tag/'seal.json')
            plans=reader.json(seal['plans']);complete=reader.json(run_root/'cases'/tag/'complete.json')
            t.update(geometry_seconds=complete.get('geometry_seconds'),
                shared_prediction_models_seconds=complete.get('shared_prediction_models_seconds'),
                case_wall_seconds_including_execution=complete.get('seconds'))
            walls=[]
            for q in bycase[tag]:
                source=plans[q['policy']];a=dict(q,data_origin='new_guard_quote')
                a['shared_prediction_models_seconds']=source.get('shared_prediction_models_seconds')
                a['initial_points_source']=source.get('initial_points')
                a['executed_source']=source.get('executed_source')
                cert=q.get('certificate') or {};native=cert.get('raw_native_attributes') or source.get('native_attributes') or {}
                # The independent audit recomputes this activation fee from the final action.
                direct=cert.get('executed_activation_yuan') if cert.get('accepted') else None
                a['direct_fee_for_bound_comparison_yuan']=direct
                a['raw_native_bound_minus_direct_fee_yuan']=scalar(native.get('ObjBound')-direct) if native.get('ObjBound') is not None and direct is not None else None
                a['raw_native_objective_minus_direct_fee_yuan']=scalar(native.get('ObjVal')-direct) if native.get('ObjVal') is not None and direct is not None else None
                a['bound_comparison_scope']='Audited direct activation fee of an accepted guarded action; zero fallback is never used as a feasible upper bound. Official certificate remains unchanged.'
                current.append(a);walls.append(a.get('quote_wall_seconds',a.get('seconds',0.)))
            t['new_quote_wall_seconds_sum']=float(sum(x for x in walls if x is not None))
            t['new_geometry_models_and_quotes_seconds']=float(t['new_quote_wall_seconds_sum']+
                (t['geometry_seconds'] or 0.)+(t['shared_prediction_models_seconds'] or 0.))
            t['time_accounting_scope']='Preparation plus quote times, excluding execution and other driver I/O; case_wall_seconds includes the latter. Geometry is charged conservatively against each quote budget but generated only once.'
        timing.append(t)
    return current,prior,timing


def number(value,digits=3): return '—' if value is None else f'{value:.{digits}f}'


def markdown(report):
    lines=['# 冻结R/X：新增调度限制的配对结果','',
        f"独立审计通过。复用基线 {report['baseline_rows']} 行，新策略执行 {report['new_rows']} 行；均保留失败和零回退。",
        '这些是已知物理日的开发配对重放，重复误差路径不作为独立负荷日期。',
        '新增预算未改变冻结评分、原概率约束或物理设备；轮换未被要求满足新增预算，其预算数值仅作诊断。轮换行的报价可行/零回退字段指对应采购方案；报价可行数只统计quotes中的采购报价。',
        '“内部正确”对应原proper-clade指标；“完整支持”还检查单节点及公共支路。真权重p_true同样按内部拓扑族计算，不能称全部线路的正确概率。','']
    for case in report['cases']:
        lines += ['## '+case['id'],'',
            '|策略|来源|内部正确|完整支持正确|错误高权重|内部真族权重|结算元|零回退|相对无动作 新增/丢失|',
            '|---|---|---:|---:|---:|---:|---:|---:|---:|']
        for name,s in case['strategies'].items():
            p=case['paired_vs_no_dfs'][name]
            lines.append(f"|{name}|{'复用' if s['data_origin_counts'].get('reused_baseline') else '新执行'}|{s['recovered']}/{s['n']}|{s['full_support_recovered']}/{s['n']}|{s['wrong_high']}|{number(s['p_true']['mean'],5)}|{number(s['settled_total_yuan']['mean'])}|{s['zero_fallback_rows']}|{p['new_recoveries']}/{p['lost_recoveries']}|")
        lines += ['', '历史赢家固定对手的增量与全库检查：','',
            '|策略|预测ΔL_K|真实输入下ΔL_K|干净ΔL|量测ΔL|预测ΔJ|实际ΔJ|全库最弱真/错L均值|最强错误竞争者|',
            '|---|---:|---:|---:|---:|---:|---:|---:|---|']
        for name,s in case['strategies'].items():
            inc=s['historical_fixed_pair']['increments'];full=s['whole_bank_fixed_true_target']['observed']
            lines.append('|'+name+'|'+'|'.join(number(inc[k]['mean']) for k in ('forecast_K_logodds','actual_input_K_logodds','clean_logodds','observed_logodds','forecast_J','actual_J'))+'|'+number(full['minimum_logodds']['mean'])+'|'+str(full['strongest_competitor_counts'])+'|')
        lines += ['', '预测与实测增益的差额分解：','',
            '|策略|输入/交付差额|响应模型差额|量测差额|实测增量−预测增量|K可用路径|',
            '|---|---:|---:|---:|---:|---:|']
        for name,s in case['strategies'].items():
            g=s['historical_fixed_pair']['forecast_actual_gap']
            lines.append('|'+name+'|'+'|'.join(number(g['components'][k]['mean']) for k in ('input_and_delivery','response_model','meter'))+'|'+number(g['total_observed_minus_forecast']['mean'])+f"|{g['n']}/{s['n']}|")
        lines += ['', 'ΔL_K用同一历史K及冻结评分H计算期望对数赔率；ΔJ仅为两候选预测距离的变化。差额三项之和逐路径核对等于实测ΔL减预测ΔL_K；目标没有代表K时记空值，不替换目标或另拟合。',
            '固定对手不会随动作改变；全库列独立检查其他候选反超，两个口径不得混用。完整机器表保留固定对手身份、逐路径赢家、全部预算与H/K响应误差。','']
        targets=case['comparison_targets']
        shown=[('fixed_true_target','历史固定真族目标')]
        if targets['historical_target']!=targets['fixed_true_target']:
            shown.append(('historical_target','历史赢家目标'))
        for target,label in shown:
            lines += [f"全库最弱优势（{label} native {targets[target]}）：",'',
                '|策略|预测K 最弱L / 增量|真实输入K 最弱L / 增量|干净 最弱L / 增量|量测 最弱L / 增量|预测/量测最强竞争者|',
                '|---|---:|---:|---:|---:|---|']
            for name,s in case['strategies'].items():
                bank=s['whole_bank_'+target];columns=[]
                for law in ('forecast_K','actual_input_K','clean','observed'):
                    z=bank[law]
                    columns.append('—' if z is None else number(z['minimum_logodds']['mean'])+' / '+number(z['minimum_logodds_increment']['mean']))
                pred=bank['forecast_K'];obs=bank['observed']
                opponents=str(pred['strongest_competitor_counts']) if pred is not None else 'K不可用'
                lines.append('|'+name+'|'+'|'.join(columns)+'|'+opponents+' / '+str(obs['strongest_competitor_counts'])+'|')
        lines += ['', '预测列取“各竞争者期望对数赔率的最小值”，未计算随机最小值的期望。各行保留同一目标，最强竞争者可以改变；历史赢家与固定真族目标相同时只展示一次。','']
    lines += ['## 新报价状态与费用界','',
        '|输入|方案|状态/原生码|可行|零回退|激活费UB/LB|原始LB−直接费|gap|秒|初值/返回来源|',
        '|---|---|---|---:|---:|---:|---:|---:|---:|---|']
    for q in report['quotes']['rows']:
        cert=q.get('certificate',{});initial=q.get('initial') or {}
        ub=cert.get('feasible_upper_bound_yuan',q.get('cost_yuan'))
        lb=cert.get('lower_bound_yuan',q.get('lower_bound_yuan'))
        delta=q.get('raw_native_bound_minus_direct_fee_yuan')
        lines.append(f"|{q['id']}|{q['policy']}|{q.get('status')}/{q.get('native_status')}|{bool(q.get('new_gate_feasible'))}|{bool(q.get('zero_fallback'))}|{number(ub)}/{number(lb)}|{'—' if delta is None else format(delta,'.9g')}|{number(cert.get('relative_gap',q.get('relative_gap')))}|{number(q.get('quote_wall_seconds',q.get('seconds')),1)}|{initial.get('name','—')} / {q.get('source','—')}|")
    conflicts=[q for q in report['quotes']['rows'] if (q.get('certificate',{}).get('native_bound_history') or {}).get('recorded_bound_conflict_with_verified_buffered_point')]
    if conflicts:
        lines += ['', '**求解轨迹中的费用界冲突：**','',
            '|输入/方案|原生最终码|先前最大下界|已直接核验可行费|冲突callback条数|审计保留LB/gap|',
            '|---|---:|---:|---:|---:|---:|']
        for q in conflicts:
            c=q['certificate'];h=c['native_bound_history'];b=h.get('maximum_recorded_callback_lower_bound') or {}
            lines.append(f"|{q['id']} / {q['policy']}|{q.get('native_status')}|{number(b.get('bound'),9)}|{number(h.get('verified_activation_cost'),9)}|{h.get('callback_prior_bounds_above_verified_point')}|{number(c.get('lower_bound_yuan'))}/{number(c.get('relative_gap'))}|")
        lines += ['', '先前下界超过同一带数值缓冲约束下的已核验可行费，属于费用证书冲突。最终原生状态2或gap=0不能恢复这个最优性证书；报告采用独立审计保留的界。']
    lines += ['', '费用界只属于对应新增预算下的调用费优化；表中结算总费另外包含预留费。超时、初值保留、不可行与界冲突原样保留。',
        '撤回下界不自动解释为严重求解冲突；表中另外保留原始下界与直接调用费之差，区分微小数值差和实质冲突。',
        'H/K响应误差为实际AC动作增量减冻结H/历史K预测的均方，再除评分方差；设计方差预算与实际误差是两个量。新增预算可行本身不提供90%的实际电网恢复保证。','',
        '## 计算时间','',
        '|输入|旧初值池＋参考报价秒（复用）|新几何秒|新共享预测模型秒|新报价秒合计|新案例总墙钟秒含执行|',
        '|---|---:|---:|---:|---:|---:|']
    for t in report['case_timing']:
        lines.append('|'+t['id']+'|'+'|'.join(number(t.get(k),2) for k in (
            'old_shared_pool_and_anchor_seconds','geometry_seconds','shared_prediction_models_seconds',
            'new_quote_wall_seconds_sum','case_wall_seconds_including_execution'))+'|')
    lines += ['', '旧工作不在本轮重新执行；几何和共享模型只计一次。逐报价字段保留各自实际耗时，旧基线报价详表保存在baseline_quotes。',
        '只计算封存观测的算术复核，没有新增AC、辨识拟合或报价。输入哈希结束时复核未变。','']
    return '\n'.join(lines)


def report(audit_path,out,*,baseline_only=False,ids=None):
    reader=Reader();audit=reader.json(audit_path)
    if audit.get('status')!='passed':raise ValueError('Passed independent audit required')
    protocol=reader.json(audit['source_protocol']);entries=reader.json(audit['source_inputs'])
    if isinstance(entries,dict):entries=entries['inputs']
    selected=set(ids or [e['id'] for e in entries]);entrymap={e['id']:e for e in entries if e['id'] in selected}
    if set(entrymap)!=selected:raise ValueError('Selected input absent from audited protocol')
    original_audit=reader.json(protocol['source_audit']) if 'source_audit' in protocol else audit
    coverage={x['id']:x for x in audit.get('input_coverage',[]) if x['id'] in selected}
    if len(coverage)!=len(entrymap):
        old=original_audit
        if old:coverage.update({x['id']:x for x in old['input_coverage'] if x['id'] in selected})
    if set(coverage)!=selected:raise ValueError('Missing audited true-family coverage')
    original=reader.lines(audit['records'])
    if baseline_only:baseline=[r for r in original if r['id'] in selected];new=[];quotes=[]
    else:
        new=[r for r in original if r['id'] in selected]
        baseline=[r for r in reader.lines(audit['baseline_records']) if r['id'] in selected]
        quotes=[q for q in reader.lines(audit['quotes']) if q['id'] in selected]
    raw=new+baseline
    if len({key(r) for r in raw})!=len(raw):raise ValueError('Duplicate policy path across new and reused records')
    if any(r['audit_status'] not in ('passed','recorded_failure') for r in raw):raise ValueError('Unverified execution row')
    post=Postprocess(reader,entrymap,coverage,Path(audit['source_protocol']['path']).parent,baseline_only)
    enriched=[post.enrich(r,'reused_baseline') for r in baseline]+[post.enrich(r,'new_guard_execution') for r in new]
    quotes,oldquotes,timing=quote_details(reader,quotes,entrymap,post,post.run_root,original_audit,baseline_only)
    cases=[]
    for tag in entrymap:
        rows=[r for r in enriched if r['id']==tag]
        names=list(dict.fromkeys(r['strategy'] for r in rows))
        if 'no_dfs' not in names:raise ValueError('No paired zero baseline')
        if any(sum(r['strategy']==name for r in rows)!=protocol['repeats'] for name in names):
            raise ValueError('Policy denominator differs from locked repeats: '+tag)
        if not baseline_only:
            expected={mode+'__'+kind+suffix for mode in protocol['modes'] for kind in protocol['kinds']
                for suffix in ('','_rotation')}
            actual={r['strategy'] for r in rows if r['data_origin']=='new_guard_execution'}
            if expected!=actual:raise ValueError('Missing or unexpected new policy: '+tag)
        group={name:summarize([r for r in rows if r['strategy']==name]) for name in names}
        comparisons={name:paired(rows,name,'no_dfs') for name in names}
        before={name:paired(rows,name,name.split('__')[0]) for name in names if '__' in name and not name.endswith('_rotation')}
        rotation={name:paired(rows,name,name+'_rotation') for name in names if not name.endswith('_rotation') and name+'_rotation' in names}
        dimensions={k:entrymap[tag].get(k) for k in ('case','original_id','start_date','stride','meter_multiplier','training_rows','validation_rows','base_kw')}
        comparison_targets=dict(historical_target=post.contexts[tag]['historical'],fixed_true_target=post.contexts[tag]['true_target'])
        cases.append(dict(id=tag,dimensions=dimensions,comparison_targets=comparison_targets,coverage=coverage[tag],strategies=group,paired_vs_no_dfs=comparisons,
            paired_vs_original_mode=before,paired_vs_same_fee_rotation=rotation,
            planned_repeats=protocol['repeats']))
    reader.checked(__file__)
    unchanged=reader.unchanged()
    if not unchanged:raise ValueError('Input changed during report')
    result=dict(status='complete',created_utc=datetime.now(timezone.utc).isoformat(),
        audit=reader.refs[str(Path(audit_path).resolve())],source_audit=str(Path(audit_path).resolve()),
        source_protocol=audit['source_protocol'],source_inputs=audit['source_inputs'],
        phase=protocol.get('phase'),parent_run=protocol.get('parent_run'),smoke=protocol.get('smoke',False),
        modes=protocol.get('modes'),kinds=protocol.get('kinds'),pq_rho=protocol.get('pq_rho'),response_rms=protocol.get('response_rms'),
        baseline_only=baseline_only,baseline_rows=len(baseline),new_rows=len(new),cases=cases,
        quotes=quote_summary(quotes),baseline_quotes=quote_summary(oldquotes),case_timing=timing,path_records=enriched,
        rotation_unmatched_fees=audit.get('rotation_unmatched_fees',[]),
        original_failed_tasks=audit.get('original_failed_tasks',[]),
        task_errors=audit.get('task_errors',[]),original_execution_failed_rows=audit.get('original_execution_failed_rows',[]),
        recorded_failures=[r for r in enriched if r['audit_status']!='passed'],
        maximum_native_SSE_replay_error=post.maximum_error,sources_unchanged=unchanged,
        statistical_scope='Known-world development replay: one fixed forecast per selected history, paired repeated execution/meter streams. No independent-date or probability-coverage claim.',
        diagnostic_scope='Read-only postprocessing. Historical-target fixed rival and full-native worst wrong competitor are separate; physical truth only labels offline evaluation. No outcome-chosen planning or changes to frozen H/K.',
        input_refs=[dict(path=p,sha256=h) for p,h in reader.refs.items()])
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    for name,data in [('report.json',result),('complete.json',dict(status='complete',sources_unchanged=unchanged,baseline_rows=len(baseline),new_rows=len(new)))]:
        (out/name).write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    (out/'REPORT.md').write_text(markdown(result),encoding='utf-8')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--baseline-only',action='store_true')
    parser.add_argument('--ids',nargs='+')
    args=parser.parse_args()
    result=report(args.audit,args.out,baseline_only=args.baseline_only,ids=args.ids)
    print(json.dumps({k:result[k] for k in ('status','baseline_rows','new_rows','maximum_native_SSE_replay_error','sources_unchanged')},ensure_ascii=False))


if __name__=='__main__':main()
