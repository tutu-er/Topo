"""Rebuild saved compact proxies and replay their original-box QP certificates.

Historical fits/explanations are restored from saved states. This reviewer
never calls a fitter, profile builder, selector, QP solver or MILP. AC means and
derivatives are recomputed with the matched frozen numerical source; the QP
residual, gradient and convex tangent bound are separate audit arithmetic.
"""
from __future__ import annotations
import argparse
import base64
from collections import Counter
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from .information_fit_recovery import read_json,write_json,exact_candidate,independent_replay
from .information_replay import _obs,_meter
from .ev_original_review import replay_qp,close,require
from .ev_original_ac_check import build_multisite_context,make_ac_context,evaluate_ac_witness,recenter_at_absolute

HERE=Path(__file__).resolve().parent
NUMERICAL_FILES=('ev_original_ac_check.py','ev_ac_moving_cross.py','symmetric_ac_local.py',
                 'likelihood.py','candidates.py','ev_prediction_ensemble.py')


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def restore_fit(row):
    values=dict(row)
    for k in ('r','x','p','q','source_p','source_q','bias_p','bias_q','bias_v'):
        values[k]=np.asarray(row[k],float)
    return SimpleNamespace(**values)


def audit_sources(directory,freeze):
    snapshot=read_json(directory/'source_snapshot.json.gz');hashes=freeze['source_sha256']
    require(set(snapshot)==set(hashes),'snapshot source coverage')
    for name,expected in hashes.items():
        raw=base64.b64decode(snapshot[name]['bytes_base64'],validate=True)
        require(snapshot[name]['sha256']==expected and hashlib.sha256(raw).hexdigest()==expected,'snapshot bytes:'+name)
    numerical=[]
    for name in NUMERICAL_FILES:
        require(name in hashes,'numerical source not frozen:'+name)
        actual=digest(HERE/name);require(actual==hashes[name],'numerical source changed:'+name)
        numerical.append(dict(name=name,sha256=actual))
    return dict(snapshot_sources=len(hashes),matched_numerical_sources=numerical,
        scope='AC derivatives rebuilt with matched frozen numerical code, not an independent derivative proof')


def rebuild_models(trial):
    history=_obs(trial['history']);hm=_meter(trial['measurement_models']['hm']);fm=_meter(trial['measurement_models']['fm'])
    candidates=[exact_candidate(row,history.n) for row in trial['historical_fits']]
    hfits={row['name']:restore_fit(row) for row in trial['historical_fits']}
    data=trial['selection']['context'];reference=np.asarray(trial['selection']['constant_operation']['u_kw'],float)
    cfg=trial['config']['physical'];base_kw=cfg['base_kw'];models=[];derivative_calls=0
    if trial['prediction']=='center':
        contexts=[dict(id=None,diagnostics=data)]
    else:contexts=data['context_records']
    historical_explanations=data.get('history_explanations',{})
    byid={r['id']:(name,r) for name,rows in historical_explanations.items() for r in rows}
    explanation_audits=[];best_H={}
    for name,rows in historical_explanations.items():
        losses=[]
        for row in rows:
            fitrow=row['fit'];candidate=next(c for c in candidates if c.name==name)
            audit=independent_replay(candidate,fitrow,history,hm)
            require(audit['success'],'historical profile AC/SSE/loglik:'+row['id'])
            losses.append(float(fitrow['loss']))
            close(row['history_gap_half_sse'],.5*(fitrow['loss']-hfits[name].loss),'historical support gate basis')
            require(row['history_gap_half_sse']<=trial['config']['ensemble']['history_gate_half_sse']+1e-8,
                    'retained historical explanation outside saved gate')
            explanation_audits.append(dict(id=row['id'],candidate=name,**audit))
        best_H[name]=dict(original_H_SSE=float(hfits[name].loss),best_retained_H_SSE=min(losses),
            best_delta_full_SSE=min(losses)-hfits[name].loss,retained=len(rows))
    for context in contexts:
        diagnostics=context['diagnostics']
        bounds=diagnostics.get('line_bounds') or (cfg['asset_bounds'] if trial['domain']=='asset' else dict(r_bounds=(1e-6,.1),x_bounds=(1e-6,.1)))
        states=diagnostics.get('state_bounds')
        overrides=None;fits=hfits
        if context['id'] is not None:
            fits={name:restore_fit(byid[identifier][1]['fit']) for name,identifier in context['explanation_ids'].items()}
            gain=np.asarray(context['control_gain'],float);overrides={}
            for name,background in context['forecast_background'].items():
                bg={k:np.asarray(v,float) for k,v in background.items()}
                overrides[name]=dict(bg,p=bg['p']+gain*reference/base_kw,control_gain=gain)
        # Both frozen v1/v2 model-bank interfaces use this complete planning
        # box; a narrower operational command box does not shrink the proxy.
        built=build_multisite_context(history,candidates,fits,hm,fm,reference,-6.,6.,base_kw,
            line_bounds=bounds,state_bounds=states,step_kw=.001,forecast_overrides=overrides)
        require(built['success'],'saved-state proxy reconstruction:'+str(built['diagnostics']))
        derivative_calls+=built['diagnostics']['ac_calls']
        for model in built['models']:
            if context['id'] is not None:
                model['name']+='__'+context['id']
                model.update(ensemble_id=context['id'],task_weight=context['weight']/(len(candidates)*(len(candidates)-1)),
                    expansion_reference_history_cost=float(fits[model['a']].loss),
                    expansion_competitor_history_cost=float(fits[model['b']].loss),
                    reference_history_cost=float(hfits[model['a']].loss),competitor_history_cost=float(hfits[model['b']].loss))
            else:model['task_weight']=1/(len(candidates)*(len(candidates)-1))
            models.append(model)
    require(len({m['name'] for m in models})==len(models),'rebuilt model identity collision')
    require(len(models)==trial['selection']['model_task_count'],'rebuilt model task denominator')
    return dict(models=models,history=history,hm=hm,fm=fm,candidates=candidates,hfits=hfits,
        derivative_rebuild_AC_calls=derivative_calls,explanation_audits=explanation_audits,best_H=best_H)


def replay_exchange(result,models):
    require(result['required_pairs']==[m['name'] for m in models],'saved QP/model order')
    require(result['aggregation']=='mean','compact driver aggregation differs from saved mean task value')
    weights=result['weights']
    for model in models:
        close(weights[model['name']],model['task_weight'],'saved task weight')
        close(result['offsets'][model['name']],model['competitor_history_cost'],'common historical baseline')
    records=[];count=Counter()
    for index,evaluation in enumerate(result['evaluations']):
        require(len(evaluation['qp_results'])==len(models),'QP pair denominator')
        rows=[];lower=[];upper=[]
        for model,saved in zip(models,evaluation['qp_results']):
            audit=replay_qp(model,np.asarray(evaluation['u'],float),saved)
            rows.append(dict(name=model['name'],**audit));count['QP_states_replayed']+=1
            count['QP_successful']+=int(audit['success'])
            count['QP_exception_fallback']+=int(audit.get('declared_exception_fallback',False))
            weight=weights[model['name']];offset=model['competitor_history_cost'];scale=result['score_scale']
            lower.append(weight*scale*(audit['lower']-offset))
            upper.append(weight*scale*(audit['upper']-offset))
        # Capped exchange clips each task before averaging; recompute both
        # score endpoints from QP interval endpoints with the saved cap.
        caps=result.get('score_caps')
        if caps is not None:
            lower=[];upper=[]
            for i,(model,row) in enumerate(zip(models,rows)):
                weight=weights[model['name']];offset=model['competitor_history_cost'];scale=result['score_scale']
                cap=caps[model['name']] if isinstance(caps,dict) else caps[i]
                lower.append(weight*min(scale*(row['lower']-offset),cap))
                upper.append(weight*min(scale*(row['upper']-offset),cap))
        lo,hi=float(sum(lower)),float(sum(upper))
        close(evaluation['score_lower'],lo,'saved action QP lower aggregate')
        close(evaluation['score_upper'],hi,'saved action QP upper aggregate')
        records.append(dict(index=index,origin=evaluation.get('origin'),score_lower=lo,score_upper=hi,pairs=rows))
    return dict(status=result['status'],counts=dict(count),evaluations=records)


def audit_trial(trial):
    built=rebuild_models(trial);base=built['models'];history=built['history'];hm=built['hm'];fm=built['fm']
    candidates=built['candidates'];hfits=built['hfits'];gate=trial['config']['proxy_channel_RMS_gate']
    methods={};total=Counter();same_witness=0;recenter_calls=0
    for name,record in trial['selection']['methods'].items():
        if record.get('raw') is None:continue
        raw=record['raw'];rounds=[]
        if record.get('before_recenter') is not None:
            previous=record['before_recenter'];first=replay_exchange(previous,base);rounds.append(first);total.update(first['counts'])
            chosen=previous['evaluations'][previous['best_evaluation_index']];local=[]
            saved_audits=record['proxy_AC_checks'][0]
            require(len(saved_audits)==len(base),'same witness recenter denominator')
            for model,qp,saved in zip(base,chosen['qp_results'],saved_audits):
                require(saved['name']==model['name'],'same witness task identity')
                context=make_ac_context(model,history,candidates,hfits,hm,fm,voltage_limits=(.9,1.1))
                actual=evaluate_ac_witness(model,previous['u'],qp['physical_parameter'],context);same_witness+=1
                close(saved['check']['max_channel_RMS'],actual['max_channel_RMS'],'recenter RMS decision')
                if not actual['max_channel_RMS']<=gate:
                    changed=recenter_at_absolute(model,previous['u'],qp['physical_parameter'],
                        make_ac_context(model,history,candidates,hfits,hm,fm))
                    local.append(changed);recenter_calls+=changed['expansion_ac_calls']
                else:local.append(model)
        else:local=base
        final=replay_exchange(raw,local);rounds.append(final);total.update(final['counts'])
        methods[name]=dict(recentered=bool(record.get('recentered')),rounds=rounds)
    return dict(seed=trial['seed'],success=True,methods=methods,counts=dict(total),
        source_interface=1,best_retained_H=built['best_H'],historical_explanations_replayed=len(built['explanation_audits']),
        historical_explanation_audits=built['explanation_audits'],
        derivative_rebuild_AC_calls=built['derivative_rebuild_AC_calls'],same_witness_replays=same_witness,
        recenter_derivative_rebuild_AC_calls=recenter_calls,
        numerical_calls_are_review_only=True,no_new_fit=True,no_new_QP_or_MILP_solve=True)


def review_directory(path,output=None,limit=None):
    directory=Path(path);freeze=read_json(directory/'freeze.json');sources=audit_sources(directory,freeze)
    files=sorted(directory.glob('trial_*.json.gz'))
    if limit is not None:files=files[:limit]
    histories=[];errors=[];hashes=[];seeds=[]
    for file in files:
        trial=read_json(file);require(trial['source_sha256']==freeze['source_sha256'],'trial/source freeze identity')
        require(trial['seed'] in freeze['seeds'] and trial['seed'] not in seeds,'planned unique independent history')
        require(trial['policy_id']==freeze['policy_id'] and trial['config']==freeze['config']
            and trial['prediction']==freeze['prediction'] and trial['domain']==freeze['domain']
            and trial['master_form']==freeze['master_form'],'trial/policy freeze identity')
        seeds.append(trial['seed'])
        try:histories.append(audit_trial(trial))
        except (ValueError,KeyError,RuntimeError,TypeError,IndexError) as error:
            errors.append(dict(seed=trial['seed'],error=type(error).__name__+':'+str(error)))
        hashes.append(dict(name=file.name,sha256=digest(file)))
    totals=Counter()
    for history in histories:totals.update(history['counts'])
    out=dict(success=not errors,planned_independent_histories=len(freeze['seeds']),available_independent_histories=len(files),
        audited_histories=len(histories),review_errors=errors,totals=dict(totals),source_audit=sources,
        policy_id=freeze['policy_id'],seeds=seeds,planned_seeds=freeze['seeds'],
        trial_sha256={r['name']:r['sha256'] for r in hashes},freeze_sha256=digest(directory/'freeze.json'),
        trial_hashes=hashes,histories=histories,
        reviewer=dict(file=Path(__file__).name,sha256=digest(__file__),science_source=False),
        scope='saved-state matrix reconstruction, original-box residual and convex tangent-bound replay; no optimizer',
        derivative_scope='matched frozen AC derivative code; not an independent analytic-derivative proof',
        formal_calibration_histories=0,formal_rejection_evaluation_histories=0)
    if output:write_json(output,out)
    return out


def main():
    p=argparse.ArgumentParser();p.add_argument('--directory',required=True,type=Path)
    p.add_argument('--output',type=Path);p.add_argument('--limit',type=int);a=p.parse_args()
    r=review_directory(a.directory,a.output,a.limit)
    print(json.dumps({k:r[k] for k in ('success','available_independent_histories','audited_histories','review_errors','totals')}))


if __name__=='__main__':main()
