"""Outer saved-state audit for compact EV policies; never refit or select.

Reuses the independent branch-current AC arithmetic and route replay of the
original reviewer. Saved prediction states are all checked, including failed
fits. Proxy matrices are not stored by this driver: QP box/norm checks below
are explicitly weaker than a complete matrix/certificate replay.
"""
from __future__ import annotations
import argparse
import base64
from copy import deepcopy
import hashlib
from pathlib import Path
import numpy as np
from . import ev_original_review as old
from . import ev_policy_calibration as cal
from .information_replay import _obs, _meter
from .likelihood import RadialCandidate, concat_observations

HERE=Path(__file__).resolve().parent
METHODS=('none','constant','quality_fixed','quality_mobile','random_mobile',
         'charge_fixed','v2g_fixed','charge_mobile','v2g_mobile','capped_v2g_mobile')
DESIGNS=('charge_fixed','v2g_fixed','charge_mobile','v2g_mobile','refined_v2g_mobile','capped_v2g_mobile')
INHERITS={'charge_fixed':(),'v2g_fixed':('charge_fixed',),
    'charge_mobile':('charge_fixed',),'v2g_mobile':('charge_fixed','v2g_fixed','charge_mobile'),
    'refined_v2g_mobile':('charge_fixed','v2g_fixed','charge_mobile','v2g_mobile'),
    'capped_v2g_mobile':('charge_fixed','v2g_fixed','charge_mobile','v2g_mobile')}


def key(power):
    value=np.round(old.arr(power),10);value[value==0]=0.
    return value.astype('<f8').tobytes()


def physical_config(trial):
    return trial['config']['physical']


def service_config(config,method):
    return old.service_config(config,'v2g_mobile' if method in ('capped_v2g_mobile','refined_v2g_mobile') else method)


def operation_config(trial):
    hardware=physical_config(trial)
    if 'operations_config' not in trial:return hardware
    expected=deepcopy(hardware);command=trial['config']['command_power_kw']
    expected['device']['power_kw']=command
    old.require(trial['operations_config']==expected,'operation configuration differs beyond declared command limit')
    maximum_gain=1+trial['config']['ensemble']['tracking_fraction']
    old.require(command>0 and command*maximum_gain<=hardware['device']['power_kw']+1e-12,
        'command plus declared tracking exceeds hardware power')
    return trial['operations_config']


def method_names(trial):
    names=tuple(trial['config'].get('methods',METHODS))
    old.require(len(set(names))==len(names) and set(names) in
        (set(METHODS),set(METHODS)|{'refined_v2g_mobile'}),'declared method set')
    return names


def score(saved,cap=None):
    if not saved.get('success'):return None
    pairs=saved['pairs'];old.require(bool(pairs),'empty successful prediction')
    values=old.arr([p['added'] for p in pairs]);weights=old.arr([p.get('weight',1/len(pairs)) for p in pairs])
    old.require(np.isfinite(values).all() and np.isfinite(weights).all()
        and np.all(weights>=0) and abs(weights.sum()-1)<1e-10,'invalid prediction task weights')
    if cap is not None:
        old.require(np.isfinite(cap) and cap>=0,'invalid cap')
        values=np.minimum(values,cap)
    return float(weights@values)


def _prediction_contexts(selection,history,hm,hfits,config):
    """Read saved H explanations and verify their costs without optimizing."""
    n=history.n;T=config['horizon'];names=list(hfits)
    center=dict(p=np.tile(np.mean(old.arr(history.p)-old.arr(hm.bias_p),axis=0),(T,1)),
        q=np.tile(np.mean(old.arr(history.q)-old.arr(hm.bias_q),axis=0),(T,1)),
        root_v=np.full(T,np.mean(history.root_v)))
    if selection.get('prediction')=='center':
        return {None:dict(weight=1.,fits=hfits,control_gain=np.ones((T,n)),
            forecast_background={a:dict(**center,source_p=np.full(T,np.mean(hfits[a]['source_p'])),
                source_q=np.full(T,np.mean(hfits[a]['source_q']))) for a in names})},0
    diagnostic=selection['context'];banks=diagnostic['history_explanations']
    contexts={};extra=0;explanations={}
    for a,bank in banks.items():
        old.require(a in hfits and bank,'required explanation candidate')
        explanations[a]={}
        for entry in bank:
            old.require(entry['id'] not in explanations[a],'duplicate explanation id')
            fit=entry['fit'];explanations[a][entry['id']]=fit
            old.require(fit['name']==a,'explanation fit name')
            extra+=1
    scenes=diagnostic['forecast_scenarios'];declared=scenes['diagnostics']
    old.require(not declared['uses_truth'] and not declared['uses_future_noise']
        and declared['finite_stress_scenarios'] and not declared['predictive_distribution_calibrated'],
        'scenario information/probability scope')
    by_scene={s['id']:s for s in scenes['scenarios']}
    old.require(set(by_scene)=={'center','minus','plus'},'three declared forecast scenarios')
    for scene in by_scene.values():
        for k,v in center.items():old.close(scene['background'][k],v,'scenario H background:'+k,atol=1e-12)
        old.require(scene['source_interface']==1 and scene['EV_in_total_PQ_once'],'scenario source/EV coordinate')
        for a in names:
            mode=old.arr(declared['mode_records'][a]['declared_mode_pu']);shape=old.arr(declared['shock_shape'])
            expected=dict(p=shape[:,None]*mode[:n],q=shape[:,None]*mode[n:2*n],
                source_p=shape*mode[2*n],source_q=shape*mode[2*n+1])
            for k,v in expected.items():old.close(scene['offsets'][a][k],v,'declared forecast offset:'+k,atol=1e-12)
        old.close(scene['control_gain'],np.full((T,n),1+scene['sign']*declared['tracking_fraction']),
            'declared tracking gain',atol=1e-12)
    for context in diagnostic['context_records']:
        ident=context['id'];old.require(ident not in contexts,'duplicate saved context')
        scene=by_scene[context['scenario_id']];chosen={}
        for a in names:
            index=context['axis'] if context['axis']<len(banks[a]) else 0
            old.require(context['explanation_ids'][a]==banks[a][index]['id'],'context axis explanation choice')
            chosen[a]=explanations[a][context['explanation_ids'][a]]
            fit=chosen[a];offset=scene['offsets'][a];sign=scene['sign']
            expected=dict(p=center['p']+sign*old.arr(offset['p']),q=center['q']+sign*old.arr(offset['q']),
                source_p=np.full(T,np.mean(fit['source_p']))+sign*old.arr(offset['source_p']),
                source_q=np.full(T,np.mean(fit['source_q']))+sign*old.arr(offset['source_q']),root_v=center['root_v'])
            for k,v in expected.items():old.close(context['forecast_background'][a][k],v,'context H forecast:'+k,atol=1e-12)
        old.close(context['control_gain'],scene['control_gain'],'context control gain',atol=1e-12)
        old.close(context['weight'],sum(a['weight'] for a in context['alias_records']),'context alias weight',atol=1e-12)
        contexts[ident]=dict(context,fits=chosen)
    old.require(abs(sum(v['weight'] for v in contexts.values())-1)<1e-12,'context weights')
    return contexts,extra


def audit_prediction(saved,history,hm,fm,jm,candidates,hfits,config,domain,contexts):
    old.require(not saved['uses_truth'] and not saved['uses_future_noise'],'prediction information boundary')
    old.require(saved.get('source_interface',1)==1 and saved.get('source_interface_shared',True),
        'prediction source interface')
    old.require(not saved.get('alternative_futures_stacked',False),'stacked alternative futures')
    power=old.arr(saved['power_kw']);names=list(candidates);pair_order=[(a,b) for a in names for b in names if a!=b]
    old.require(power.shape==(config['horizon'],history.n),'prediction command shape')
    fit_lookup={};seen=set();replayed=0;failures=0
    for rec in saved['records']:
        cid=rec.get('ensemble_id');a=rec['a'];cell=(cid,a)
        old.require(cell not in seen and cid in contexts and a in candidates,'prediction context/reference coverage')
        seen.add(cell);context=contexts[cid];reference=context['fits'][a]
        forecast=rec['forecast_state'];background=context['forecast_background'][a]
        for k in ('q','source_p','source_q','root_v'):old.close(forecast[k],background[k],'prediction forecast:'+k,atol=1e-12)
        old.close(forecast['p'],old.arr(background['p'])+old.arr(context['control_gain'])*power/config['base_kw'],
            'prediction total P and one EV',atol=1e-12)
        if 'control_gain' in rec:old.close(rec['control_gain'],context['control_gain'],'prediction gain',atol=1e-12)
        if rec.get('predicted_observations') is None:
            old.require(not rec['success'],'successful forecast without observations');failures+=1;continue
        physical=old.forward_physics(candidates[a],reference['r'],reference['x'],forecast['p'],forecast['q'],
            forecast['source_p'],forecast['source_q'],forecast['root_v'])
        obs=_obs(rec['predicted_observations'])
        for k in old.CHANNELS:
            if getattr(obs,k) is not None:
                old.close(getattr(obs,k),old.arr(physical[k])+(getattr(fm,'bias_'+k) if k in ('p','q','v') else 0),
                    'predicted noiseless AC:'+k,atol=2e-10)
        audit=old.replay_fits(rec['fits'],candidates,concat_observations(history,obs),jm,rec['failures'],
            complete=rec['success'],config=config,domain=domain)
        replayed+=audit['replayed_states'];fit_lookup[cell]={f['name']:f for f in rec['fits']}
        if 'joint_observation_count' in rec:
            old.require(rec['joint_observation_count']==history.t+len(power)
                and rec['history_observation_count']==history.t and rec['future_observation_count']==len(power),
                'joint fit includes exactly one future')
    old.require(seen=={(cid,a) for cid in contexts for a in names},'missing prediction reference cell')
    pair_seen=set();values={k:[] for k in ('added','joint','self_relative')};weights=[]
    for pair in saved['pairs']:
        cid=pair.get('ensemble_id');a,b=pair['a'],pair['b'];cell=(cid,a,b)
        old.require(cell not in pair_seen and cid in contexts and (a,b) in pair_order,'prediction pair coverage')
        pair_seen.add(cell)
        weight=contexts[cid]['weight']/len(pair_order)
        if 'weight' in pair:old.close(pair['weight'],weight,'pair planning weight',atol=1e-12)
        if not pair['success']:continue
        own=fit_lookup[cid,a][a];other=fit_lookup[cid,a][b]
        old.require(own['success'] and other['success'],'successful pair uses unsuccessful fit')
        computed=dict(added=.5*(other['loss']-hfits[b]['loss']),
            joint=.5*(other['loss']-hfits[a]['loss']),self_relative=.5*(other['loss']-own['loss']))
        for k,v in computed.items():old.close(pair[k],v,'predicted pair score:'+k);values[k].append(v)
        weights.append(weight)
    old.require(pair_seen=={(cid,a,b) for cid in contexts for a,b in pair_order},'missing directed prediction task')
    if saved['success']:
        old.require(len(weights)==len(pair_seen) and failures==0,'incomplete successful prediction')
        for k,v in values.items():
            if 'mean_'+k+'_score' in saved:old.close(saved['mean_'+k+'_score'],old.arr(weights)@old.arr(v),'mean prediction '+k)
            if 'worst_'+k+'_score' in saved:old.close(saved['worst_'+k+'_score'],min(v),'worst prediction '+k)
        old.close(saved['score'],old.arr(weights)@old.arr(values['added']),'selected prediction score')
    return dict(replayed_fit_states=replayed,reference_cells=len(seen),directed_tasks=len(pair_seen),
        success=bool(saved['success']),fit_calls=saved.get('fit_calls',0),ac_calls=saved.get('ac_calls',0))


def audit_qp_boxes_and_same_witness(raw,check_round,candidates,history,hm,fm,hfits,config,domain,contexts):
    """Recompute exact AC at saved xi; no QP optimizer or derivative rebuild."""
    if raw is None or raw.get('u') is None:return dict(box_norm_states=0,same_witness_AC_states=0,full_matrix_QP_states=0)
    states=0;names=raw['required_pairs'];n=history.n;H=history.t;T=config['horizon'];width=2*n+2
    for evaluation in raw['evaluations']:
        old.require(len(evaluation['qp_results'])==len(names),'QP pair denominator')
        for name,qp in zip(names,evaluation['qp_results']):
            a,tail=name.split('__to__');b,_,cid=tail.partition('__');cid=cid or None
            old.require(a in candidates and b in candidates and cid in contexts,'QP task name/context')
            xi=old.arr(qp['physical_parameter']);m=candidates[b].m;states+=1
            old.require(xi.shape==(2*m+width*(H+T),),'QP absolute state dimensions')
            lohi=config['asset_bounds'] if domain=='asset' else {'r_bounds':(1e-6,.1),'x_bounds':(1e-6,.1)}
            lower=np.r_[np.full(m,lohi['r_bounds'][0]),np.full(m,lohi['x_bounds'][0]),np.tile([-1.]*(2*n)+[-.5]*2,H+T)]
            upper=np.r_[np.full(m,lohi['r_bounds'][1]),np.full(m,lohi['x_bounds'][1]),np.tile([1.]*(2*n)+[.5]*2,H+T)]
            old.require(np.all(xi>=lower-1e-8) and np.all(xi<=upper+1e-8),'QP original physical box')
            residual=old.arr(qp['residual']);old.close(qp['primal_upper'],residual@residual,'QP saved norm')
            old.require(qp['tangent_lower_numeric']>=0 and qp['tangent_lower_numeric']<=qp['primal_upper']+1e-8,'QP saved interval order')
    if check_round is None:return dict(box_norm_states=states,same_witness_AC_states=0,full_matrix_QP_states=0)
    best=raw['evaluations'][raw['best_evaluation_index']];old.require(len(check_round)==len(names),'same-witness denominator')
    for name,qp,row in zip(names,best['qp_results'],check_round):
        old.require(row['name']==name,'same-witness task order')
        a,tail=name.split('__to__');b,_,cid=tail.partition('__');cid=cid or None
        context=contexts[cid];xi=old.arr(qp['physical_parameter']);m=candidates[b].m
        wh=old.unpack_state(xi[2*m:2*m+width*H],n);wn=old.unpack_state(xi[2*m+width*H:],n)
        background=context['forecast_background'][a];reference=dict(background)
        reference['p']=old.arr(background['p'])+old.arr(context['control_gain'])*old.arr(raw['u']).reshape(T,n)/config['base_kw']
        fit=context['fits'][a]
        ph=old.forward_physics(candidates[b],xi[:m],xi[m:2*m],wh['p'],wh['q'],wh['source_p'],wh['source_q'],history.root_v)
        pn=old.forward_physics(candidates[b],xi[:m],xi[m:2*m],wn['p'],wn['q'],wn['source_p'],wn['source_q'],reference['root_v'])
        pa=old.forward_physics(candidates[a],fit['r'],fit['x'],reference['p'],reference['q'],reference['source_p'],reference['source_q'],reference['root_v'])
        channels=tuple(k for k in old.CHANNELS if getattr(history,k) is not None)
        obsH={k:getattr(history,k) for k in channels}
        exact=np.r_[old.whiten(ph,hm,channels)-old.whiten(obsH,hm,channels,observation=True),
            old.whiten(pa,fm,channels)-old.whiten(pn,fm,channels)]
        saved=row['check'];old.close(saved['complete_AC_residual'],exact,'same-witness full AC residual')
        old.require(saved.get('source_interface',1)==1,'same-witness source interface')
        old.require(saved.get('physical_box_feasible',True),'legal saved witness labelled outside physical box')
        old.close(saved['proxy_residual'],qp['residual'],'same-witness vs selected QP residual')
        error=old.arr(saved['proxy_residual'])-exact
        old.close(saved['proxy_minus_AC'],error,'same-witness AC error')
        old.close(saved['complete_AC_cost'],exact@exact,'same-witness AC cost')
        old.close(saved['proxy_cost'],old.arr(qp['residual'])@old.arr(qp['residual']),'same-witness proxy cost')
        offset=0;rms=[]
        for count in (H,T):
            side=[]
            for channel in channels:
                length=count*(n if channel in ('p','q','v') else 1)
                side.append(float(np.linalg.norm(error[offset:offset+length])/np.sqrt(length)));offset+=length
            rms.extend(side)
        old.close(saved['max_channel_RMS'],max(rms),'same-witness channel RMS')
        ranges={k:[v['voltage_min'],v['voltage_max']] for k,v in
            [('competitor_history',ph),('competitor_future',pn),('reference_future',pa)]}
        for k,v in ranges.items():old.close(saved['voltage_ranges'][k],v,'same-witness voltage range')
        old.require(saved['complete_AC_voltage_feasible']==all(v[0]>=.9 and v[1]<=1.1 for v in ranges.values()),'same-witness voltage gate')
    return dict(box_norm_states=states,same_witness_AC_states=3*len(names),full_matrix_QP_states=0)


def audit_trial(trial):
    config=physical_config(trial);history=_obs(trial['history']);meter=trial['measurement_models']
    opconfig=operation_config(trial)
    hm,fm,jm=(_meter(meter[k]) for k in ('hm','fm','jm'));selection=trial['selection']
    old.require(not selection['uses_truth'] and not selection['uses_future_noise']
        and not selection['random_used_by_primary'],'selector information boundary')
    candidates={r['name']:RadialCandidate.from_supports(r['name'],history.n,r['supports']) for r in trial['proposal']['bank']}
    old.require(len(candidates)==4 and not trial['proposal']['uses_truth'],'four candidate repair bank')
    hfits={r['name']:r for r in trial['historical_fits']}
    hist=old.replay_fits(trial['historical_fits'],candidates,history,hm,trial['historical_failures'],
        complete=not trial['historical_failures'],config=config,domain=trial['domain'])
    view=dict(trial,config=config);assets=old.asset_unit_audit(view)
    truth=trial['truth'];H=history.t;n=history.n
    tc=RadialCandidate.from_supports('independent_truth',n,truth['family'])
    trueH=old.forward_physics(tc,truth['r'],truth['x'],old.arr(truth['p'])[:H],old.arr(truth['q'])[:H],
        old.arr(truth['source_p'])[:H],old.arr(truth['source_q'])[:H],old.arr(truth['root_v'])[:H])
    for channel in old.CHANNELS:
        if getattr(history,channel) is not None:old.close(getattr(history,channel),trueH[channel]+
            np.asarray(getattr(hm,channel+'_sigma'))*old.arr(truth['white'][channel])[:H]
            +(getattr(hm,'bias_'+channel) if channel in ('p','q','v') else 0),'true historical measurement:'+channel,atol=2e-10)
    repeats=old.arr(config['public_voltage_repeat_counts'])[:,None]
    old.close(fm.v_sigma,np.broadcast_to(old.arr(hm.v_sigma)/np.sqrt(repeats),(config['horizon'],n)),'public future V precision')
    for channel in ('p','q','master_p','master_q'):
        target=old.arr(getattr(fm,channel+'_sigma'))
        old.close(target,np.broadcast_to(old.arr(getattr(hm,channel+'_sigma'))/np.sqrt(config['other_future_repeat_count']),target.shape),'public future precision:'+channel)
    for channel in old.CHANNELS:
        if getattr(history,channel) is None:continue
        shape=old.arr(getattr(history,channel)).shape;future_shape=(config['horizon'],)+shape[1:]
        old.close(getattr(jm,channel+'_sigma'),np.concatenate((np.broadcast_to(getattr(hm,channel+'_sigma'),shape),
            np.broadcast_to(getattr(fm,channel+'_sigma'),future_shape))),'joint meter same H plus one future:'+channel)
    rows={r['method']:r for r in trial['rows']};old.require(len(rows)==len(trial['rows']),'duplicate method row')
    expected=set(method_names(trial))
    if not selection.get('failures',{}):old.require(set(rows)==expected,'complete method denominator')
    else:old.require({'none','constant'}<=set(rows)<=expected,'failed selection row coverage')
    saved_states=hist['replayed_states'];prediction_audits=[];cache={};contexts={};explanation_states=0
    if selection.get('all_prediction_records'):
        contexts,_=_prediction_contexts(selection,history,hm,hfits,config)
        if selection['prediction']=='ensemble':
            diagnostics=selection['context']['forecast_scenarios']['diagnostics']
            old.close(diagnostics['shock_shape'],np.ones(config['horizon']),'declared constant forecast shape')
            p=old.arr(history.p)-old.arr(hm.bias_p);q=old.arr(history.q)-old.arr(hm.bias_q)
            options=trial['config']['ensemble']
            for a in hfits:
                values=np.column_stack((p,q,hfits[a]['source_p'],hfits[a]['source_q']))
                _,singular,right=np.linalg.svd(values-values.mean(axis=0),full_matrices=False)
                raw=(singular[0]/np.sqrt(max(H-1,1)))*right[0]
                if raw[:n].sum()<0:raw=-raw
                floors=np.r_[np.full(2*n,options['load_floor_pu']),np.full(2,options['source_floor_pu'])]
                mode=np.where(raw<0,-1.,1.)*np.maximum(abs(raw),floors)
                saved=diagnostics['mode_records'][a]
                old.close(saved['raw_history_mode_pu'],raw,'H-only forecast principal mode',atol=1e-12)
                old.close(saved['declared_mode_pu'],mode,'declared forecast floors',atol=1e-12)
                old.close(saved['component_floor_pu'],floors,'forecast component floors',atol=1e-12)
            for a,bank in selection['context']['history_explanations'].items():
                for entry in bank:
                    audit=old.replay_fits([entry['fit']],{a:candidates[a]},history,hm,{},complete=True,config=config,domain=trial['domain'])
                    explanation_states+=audit['replayed_states']
                    old.require(entry['fit']['loss']<=hfits[a]['loss']+2*trial['config']['ensemble']['history_gate_half_sse']+1e-7,'historical explanation loss gate')
        for rec in selection['all_prediction_records']:
            k=key(rec['power_kw']);old.require(k not in cache,'duplicate predicted-cache record')
            cache[k]=rec;audit=audit_prediction(rec,history,hm,fm,jm,candidates,hfits,config,trial['domain'],contexts)
            prediction_audits.append(audit);saved_states+=audit['replayed_fit_states']
        old.require(selection['unique_predicted_actions']==len(cache),'prediction cache denominator')
        old.require(selection['predicted_fit_calls']==sum(r.get('fit_calls',0) for r in cache.values()),'predicted fit calls')
        old.require(selection['predicted_ac_calls']==sum(r.get('ac_calls',0) for r in cache.values()),'predicted AC calls')
        for rec in selection['predicted_AC'].values():
            old.require(key(rec['power_kw']) in cache and old.identity(rec)==old.identity(cache[key(rec['power_kw'])]),'named prediction cache mismatch')
    operations={};inheritance=[];qp_audits=[]
    if selection.get('constant_operation',{}).get('feasible'):
        operation=selection['constant_operation'];operations['constant']=old.replay_operation_record(operation,service_config(opconfig,'charge_fixed'))
        if rows.get('constant',{}).get('power_kw') is not None:old.close(rows['constant']['power_kw'],operation['u_kw'],'constant executed power')
    for name,record in selection['methods'].items():
        cfg=service_config(opconfig,name)
        operation=record.get('operation')
        if operation and operation.get('feasible'):
            operations[name]=old.replay_operation_record(operation,cfg)
            if rows.get(name,{}).get('power_kw') is not None:old.close(rows[name]['power_kw'],operation['u_kw'],'selected operation curve')
        if name not in DESIGNS:continue
        old.require(not record['random_used'],'random helped primary selector')
        cap=record.get('cap');checked=[];initial_keys=set()
        for initial in record['initial_candidates']:
            old.require('random' not in initial['origin'],'random initial plan')
            if not initial['operation'].get('feasible'):
                old.require(initial['AC_score'] is None,'infeasible initial has score');continue
            old.replay_operation_record(initial['operation'],cfg)
            k=key(initial['operation']['u_kw']);initial_keys.add(k)
            old.require(k in cache,'initial curve has no saved prediction')
            value=score(cache[k],cap)
            if value is None:old.require(initial['AC_score'] is None,'failed prediction has initial score')
            else:old.close(initial['AC_score'],value,'initial AC score');checked.append((initial['origin'],k,value))
        if checked:old.close(record['initial_best_score'],max(r[2] for r in checked),'best initial score')
        rounds=[]
        if record.get('before_recenter') is not None:rounds.append(record['before_recenter'])
        if record.get('raw') is not None:rounds.append(record['raw'])
        for index,raw in enumerate(rounds):
            check=record.get('proxy_AC_checks',[])
            qp_audits.append(audit_qp_boxes_and_same_witness(raw,check[index] if index<len(check) else None,
                candidates,history,hm,fm,hfits,config,trial['domain'],contexts))
        if 'raw_proxy_gate' in record:
            last=record['proxy_AC_checks'][-1]
            gate=all(r['check']['physical_box_feasible'] and r['check']['complete_AC_voltage_feasible']
                and r['check']['max_channel_RMS']<=trial['config']['proxy_channel_RMS_gate'] for r in last)
            old.require(record['raw_proxy_gate']==gate,'saved same-witness gate differs from recomputed checks')
        if record.get('raw_operation',{}).get('feasible'):
            old.replay_operation_record(record['raw_operation'],cfg)
            rawkey=key(record['raw_operation']['u_kw']);value=score(record['raw_prediction'],cap)
            if record.get('raw_proxy_gate') and value is not None:checked.append(('MILP',rawkey,value))
        if operation and operation.get('feasible') and record.get('selected_score') is not None:
            finalkey=key(operation['u_kw']);old.require(finalkey in cache,'selected curve absent from cache')
            final=score(cache[finalkey],cap);old.close(record['selected_score'],final,'selected prediction score')
            old.require(checked and final>=max(r[2] for r in checked)-1e-8,'selection loses checked plan')
            old.require(any(origin==record['selected_origin'] and k==finalkey for origin,k,_ in checked),'selected origin/curve mismatch')
            old.require(record['guard']['accepted']==(record['selected_origin']=='MILP'),'selected-origin guard')
            for parent in INHERITS[name]:
                pop=selection['methods'].get(parent,{}).get('operation',{})
                if pop.get('feasible'):
                    pk=key(pop['u_kw']);old.require(pk in initial_keys,'feasible parent omitted from inheritance')
                    parent_score=score(cache[pk],cap)
                    if parent_score is not None:
                        old.require(final>=parent_score-1e-8,'larger domain loses inherited AC score')
                        inheritance.append(dict(method=name,parent=parent,final=final,parent_score=parent_score))
    if 'refined_v2g_mobile' in selection['methods'] and 'capped_v2g_mobile' in selection['methods']:
        siblings=[selection['methods'][m] for m in ('refined_v2g_mobile','capped_v2g_mobile')]
        initial_sets=[{key(r['operation']['u_kw']) for r in s['initial_candidates'] if r['operation'].get('feasible')}
                      for s in siblings]
        old.require(initial_sets[0]==initial_sets[1],'mean/cap siblings received different initial plans')
    physical_replays=0;eval_cache={};mean_cache={};eval_states=0;mean_states=0;row_results=[]
    family=tuple(map(tuple,trial['truth']['family']))
    true_names=[a for a,c in candidates.items() if tuple(s for s in c.supports if 1<len(s)<history.n)==family]
    ordered=[(a,b) for a in candidates for b in candidates if a!=b]
    for name,row in rows.items():
        old.require(row['candidate_miss']==(len(true_names)!=1),'candidate-miss label')
        if row.get('physical') is not None:old.future_audit(view,row,fm);physical_replays+=1
        if row.get('observations') is not None and 'fits' in row:
            obs=_obs(row['observations']);audit=old.replay_fits(row['fits'],candidates,concat_observations(history,obs),jm,row['failures'],
                complete=row['status']=='complete',config=config,domain=trial['domain']);eval_states+=audit['replayed_states']
            k=old.control_key(row['power_kw'])
            if row['cache_hit']:old.require(k in eval_cache and row['fits']==eval_cache[k]['fits'],'actual future cache')
            else:old.require(k not in eval_cache,'duplicate actual action refitted');eval_cache[k]=row
            if row['status']=='complete':
                losses={f['name']:f['loss'] for f in row['fits']};old.require(losses==row['losses'],'actual losses')
                ranking=sorted(losses,key=lambda a:(losses[a],a))
                old.require(row['winner']==ranking[0] and row['candidate_ranking']==ranking,'actual candidate ranking')
                for a,b in ordered:old.close(row['pair_signed_loss_differences'][a+'__to__'+b],.5*(losses[b]-losses[a]),'actual pair likelihood gap')
                if true_names:
                    true=true_names[0];margin=min(v for a,v in losses.items() if a!=true)-losses[true]
                    old.close(row['margin'],margin,'actual true SSE margin');old.close(row['loglik_margin'],margin/2,'actual LL margin')
        mean=row.get('noiseless_AC_refit')
        if mean:
            obs=_obs(mean['future_mean_observations'])
            for channel in old.CHANNELS:
                if getattr(obs,channel) is not None:old.close(getattr(obs,channel),old.arr(row['physical'][channel])+
                    (getattr(fm,'bias_'+channel) if channel in ('p','q','v') else 0),'noiseless mean:'+channel,atol=2e-10)
            audit=old.replay_fits(mean['fits'],candidates,concat_observations(history,obs),jm,mean['failures'],
                complete=mean['all_candidates'],config=config,domain=trial['domain']);mean_states+=audit['replayed_states']
            for detail in mean['candidate_distances']:old.noiseless_distance_audit(detail,mean['fits'],hfits,obs,fm,history.t)
            old.require([d['candidate'] for d in mean['candidate_distances']]==list(candidates),'noiseless failure denominator')
            k=old.control_key(row['power_kw'])
            if k in mean_cache:old.require(mean['fits']==mean_cache[k]['fits'],'noiseless cache')
            else:mean_cache[k]=mean
        row_results.append(dict(method=name,status=row['status'],candidate_miss=row['candidate_miss']))
    saved_states+=explanation_states+eval_states+mean_states
    actual_fits=trial['historical_cost']['fit_calls']+sum(v.get('fit_calls',0) for v in cache.values())
    actual_fits+=sum(v['cost']['fit_calls'] for v in eval_cache.values())+sum(v['cost']['fit_calls'] for v in mean_cache.values())
    explanation_diagnostic=(selection.get('context') or {}).get('explanation_diagnostics',{})
    profile_fits=explanation_diagnostic.get('fit_calls',0)
    counted_ac=dict(history_fits=trial['historical_cost']['ac_calls'],
        predictions=sum(v.get('ac_calls',0) for v in cache.values()),
        historical_profiles=explanation_diagnostic.get('ac_calls',0),
        context=(selection.get('context') or {}).get('ac_calls',0),
        actual_future_fits=sum(v['cost']['ac_calls'] for v in eval_cache.values()),
        noiseless_future_fits=sum(v['cost']['ac_calls'] for v in mean_cache.values()),
        same_witness=sum(v['same_witness_AC_states'] for v in qp_audits),recenter=0)
    for record in selection['methods'].values():
        if record.get('recentered') and record.get('proxy_AC_checks'):
            counted_ac['recenter']+=(3+2*n)*sum(r['check']['max_channel_RMS']>trial['config']['proxy_channel_RMS_gate']
                for r in record['proxy_AC_checks'][0])
    return dict(seed=trial['seed'],success=True,methods=row_results,historical=hist,asset_unit_audit=assets,
        saved_fit_states_replayed=saved_states,prediction_audits=prediction_audits,
        explanation_states_replayed=explanation_states,physical_future_states_replayed=physical_replays,
        operations=operations,inheritance_checks=inheritance,QP_audits=qp_audits,
        actual_candidate_fit_calls=actual_fits,independent_histories=1,
        historical_fixed_line_profile_fit_calls=profile_fits,total_recorded_fitting_calls=actual_fits+profile_fits,
        actual_counted_AC_calls=sum(counted_ac.values()),actual_AC_breakdown=counted_ac,
        simulator_forward_calls_with_saved_outputs=1+physical_replays,
        coverage=dict(complete_AC_saved_fit_states=True,final_and_initial_operations=True,
            inheritance_prediction_scores=True,same_witness_complete_AC=True,
            original_QP_boxes_and_saved_residual_norms=True,
            full_QP_matrix_residual_and_tangent_certificate=False,
            global_MILP_witness_upper_rebuilt=False),
        scope='saved-state arithmetic replay; no new fit, selector or statistical calibration')


def review_directory(directory,output=None):
    directory=Path(directory).resolve();freeze=old.read_json(directory/'freeze.json')
    identity=cal.freeze_identity(freeze);sources=old.source_audit(directory,freeze)
    old.require(set(freeze['config']['development_seeds']).isdisjoint(freeze['config']['evaluation_seeds'])
        and set(freeze['config']['calibration_seeds']).isdisjoint(freeze['config']['evaluation_seeds'])
        and set(freeze['config']['calibration_seeds']).isdisjoint(freeze['config']['development_seeds']),
        'development/calibration/evaluation seed overlap')
    audits=[];errors=[];hashes=[];seen=set()
    for path in sorted(directory.glob('trial_*.json.gz')):
        before=hashlib.sha256(path.read_bytes()).hexdigest();trial=old.read_json(path)
        try:
            old.require(trial['seed'] in freeze['seeds'] and trial['seed'] not in seen,'unexpected/duplicate history')
            seen.add(trial['seed'])
            old.require(cal.freeze_identity(trial)==identity,'trial/freeze policy identity')
            old.require(trial['phase']==freeze['phase'] and trial['domain']==freeze['domain'],'trial phase/domain')
            audits.append(audit_trial(trial))
        except (ValueError,KeyError,TypeError,IndexError,RuntimeError,FloatingPointError) as error:
            errors.append(dict(seed=trial.get('seed'),error=type(error).__name__+':'+str(error)))
        old.require(hashlib.sha256(path.read_bytes()).hexdigest()==before,'raw trial changed during review')
        hashes.append(dict(name=path.name,sha256=before))
    out=dict(success=not errors,phase=freeze['phase'],policy_id=identity['policy_id'],
        source_audit=sources,planned_independent_histories=len(freeze['seeds']),
        available_independent_histories=len(seen),audited_histories=len(audits),
        seeds=sorted(seen),trial_sha256={r['name']:r['sha256'] for r in hashes},
        missing_histories=[s for s in freeze['seeds'] if s not in seen],review_errors=errors,
        trial_hashes=hashes,trials=audits,
        totals={k:sum(t[k] for t in audits) for k in ('saved_fit_states_replayed',
            'physical_future_states_replayed','actual_candidate_fit_calls','explanation_states_replayed',
            'historical_fixed_line_profile_fit_calls','total_recorded_fitting_calls','actual_counted_AC_calls')},
        formal_calibration_histories=0,formal_rejection_evaluation_histories=0,
        reviewer=dict(name=Path(__file__).name,sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            bytes_base64=base64.b64encode(Path(__file__).read_bytes()).decode('ascii'),
            independent_AC_route_dependency=dict(name=Path(old.__file__).name,
                sha256=hashlib.sha256(Path(old.__file__).read_bytes()).hexdigest(),
                bytes_base64=base64.b64encode(Path(old.__file__).read_bytes()).decode('ascii')),
            science_source=False,no_new_fit=True,no_selector=True),
        complete_QP_MILP_certificate_replay=False)
    if output is not None:old.write_json(output,out)
    return out


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--directory',type=Path,required=True)
    parser.add_argument('--output',type=Path);args=parser.parse_args()
    result=review_directory(args.directory,args.output)
    import json
    print(json.dumps({k:result[k] for k in ('success','available_independent_histories','audited_histories','review_errors','totals')}))


if __name__=='__main__':main()
