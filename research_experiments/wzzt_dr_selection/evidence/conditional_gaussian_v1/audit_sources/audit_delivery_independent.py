"""Independent delivery-diagnostic saved-observation audit. Never imports the new runner/problem.

Run only after predictions_frozen.json AND summary.json exist. Replays saved
Gaussian draws, the original scorer, frozen contraction, and label accounting.
No optimization, refitting, new experiment, new draws, or raw-trajectory export.
"""
from __future__ import annotations

import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import pickle
import sys
import traceback

import numpy as np
import pandas as pd
from scipy.special import betaincinv, ndtr
from scipy.optimize import brentq

REPO = Path(r'C:/Users/23761/.codex/worktrees/wzzt-dr-selection/Topo')
STAGE = Path(__file__).resolve().parent
PRIVATE = STAGE.parent
sys.path.insert(0, str(REPO))


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Audit:
    def __init__(self):
        self.checks = []
        self.opened = []
        self.truth_opened = False

    def check(self, name, condition, **detail):
        passed = bool(condition)
        self.checks.append(dict(name=name, passed=passed, **detail))
        return passed

    def same(self, name, actual, expected, atol=1e-8, rtol=1e-10):
        x, y = np.asarray(actual), np.asarray(expected)
        if x.shape != y.shape:
            return self.check(name, False, actual_shape=list(x.shape), expected_shape=list(y.shape))
        if x.dtype.kind in 'USOb' or y.dtype.kind in 'USOb':
            return self.check(name, np.array_equal(x, y))
        finite = np.isfinite(x).all() and np.isfinite(y).all()
        err = float(np.max(np.abs(x-y), initial=0)) if finite else None
        return self.check(name, finite and np.allclose(x, y, atol=atol, rtol=rtol), max_error=err)

    def read(self, path):
        path = Path(path).resolve()
        self.opened.append(str(path))
        return json.loads(path.read_text(encoding='utf-8'))

    def checked(self, reference, name):
        path = Path(reference['path'])
        if not self.check(name, sha(path) == reference['sha256']):
            raise ValueError('required artifact hash mismatch: '+str(path))
        return path

    def load(self, path):
        path = Path(path).resolve()
        self.opened.append(str(path))
        with path.open('rb') as handle:
            return pickle.load(handle)


def power_target(scene, labels):
    for key in ('P_terminal', 'Q_terminal'):
        value = scene[key]
        if hasattr(value, 'columns') and list(value.columns) != labels:
            raise ValueError('original historical label order changed')
    p, q = (np.asarray(scene[key], float) for key in ('P_terminal', 'Q_terminal'))
    v, root = np.asarray(scene['V_terminal'], float), np.asarray(scene['V_root_observed'], float)
    y = root[:, None]**2-v**2
    if not np.allclose(y, np.asarray(scene['drop_target']), atol=1e-12, rtol=0):
        raise ValueError('original voltage target mismatch')
    if p.shape != q.shape or p.shape != y.shape or not all(np.isfinite(v).all() for v in (p,q,y)):
        raise ValueError('invalid original observations')
    return p, q, y


def observation_hash(scenes):
    frames = [pd.concat([s[k] for s in scenes], axis=0) for k in
              ('P_terminal', 'Q_terminal', 'V_terminal', 'V_root_observed')]
    days = [str(v) for v in frames[0].index.normalize()]
    digest = hashlib.sha256()
    for frame in frames:
        digest.update(pd.util.hash_pandas_object(frame, index=True).values.tobytes())
    digest.update(json.dumps(list(frames[0].columns), default=str).encode())
    digest.update(json.dumps(days).encode())
    return digest.hexdigest()


def cp_lower(k, n, delta=.05):
    return float(betaincinv(k, n-k+1, delta)) if k else 0.0


def canonical(blocks):
    return {tuple(sorted(map(int, b))) for b in blocks}


def audit_run(root, audit):
    root = Path(root).resolve()
    required = ['protocol.json','problem_frozen.json','problem.pkl','price_freeze.json',
                'planning_results.json','evaluation_draws.npz','evaluation_scalars.npz',
                'predictions_frozen.json','summary.json']
    missing = [name for name in required if not (root/name).exists()]
    if missing:
        raise RuntimeError('not ready for one complete audit: '+','.join(missing))
    protocol = audit.read(root/'protocol.json')
    pred = audit.read(root/'predictions_frozen.json')
    summary = audit.read(root/'summary.json')
    freeze = audit.read(root/'price_freeze.json')
    planning = audit.read(root/'planning_results.json')
    problem_ref = audit.read(root/'problem_frozen.json')
    audit.check('prediction_truth_gate', pred['status']=='frozen' and pred['truth_read'] is False)
    audit.checked(pred['price_freeze'], 'prediction_binds_price_freeze')
    audit.checked(summary['predictions'], 'summary_binds_frozen_predictions')
    parent_protocol=audit.read(audit.checked(protocol['parent_protocol'],'parent_protocol_hash'))
    parent_price=audit.read(audit.checked(protocol['parent_price'],'parent_price_hash'))
    audit.checked(protocol['parent_ended'],'parent_completed_before_diagnostic')
    parent_problem=audit.checked(problem_ref['parent_problem'],'original_problem_hash')
    audit.check('problem_byte_identical_to_main',sha(root/'problem.pkl')==sha(parent_problem))
    main_audit=audit.read(STAGE/'audit_v1.json')
    audit.check('original_audit_preserved',main_audit['audit_passed'] and
        sha(STAGE/'audit_independent.py')==main_audit['audit_source_sha256'])
    for old_source,expected in parent_protocol['source_sha256'].items():
        old_path=Path(old_source)
        archived=STAGE/'executed_v1'/old_path.name if old_path.parent==STAGE/'src' else old_path
        audit.check('archived_main_source:'+old_path.name,sha(archived)==expected)
    audit.check('new_evaluation_seed',protocol['evaluation_seed']==2026110204 and
        protocol['evaluation_seed']!=parent_protocol['evaluation_seed'])
    audit.check('no_new_excitation_optimizer',planning['new_optimization_calls']==0 and
        planning['excitation_target_not_certified'] is True and freeze['numerically_qualified'] is False and
        protocol['main_excitation_requirement_unchanged']==.95)
    audit.checked(freeze['planning_results'], 'price_binds_planning_results')
    audit.check('recorded_freeze_before_evaluation',
        datetime.fromisoformat(freeze['created_utc']) < datetime.fromisoformat(pred['created_utc'])
        and freeze['independent_evaluation_stream_not_opened'] is True)
    for source, expected in protocol['source_sha256'].items():
        audit.check('frozen_source:'+Path(source).name, sha(source)==expected)
    refs = protocol['inputs']
    pool = audit.read(audit.checked(refs['pool'], 'original_pool_sha'))
    packet = audit.load(audit.checked(refs['history'], 'original_history_sha'))
    history = packet['history']  # Do not access the outer forecast field.
    del packet
    resource_file = audit.read(audit.checked(refs['resources'], 'original_resources_sha'))
    resources = resource_file['resources']
    fit = audit.read(audit.checked(refs['historical_fit'], 'historical_fit_sha'))
    saved = audit.load(audit.checked(problem_ref['input'], 'frozen_problem_sha'))
    audit.check('same_history_provenance', pool['history_packet']==refs['history'])
    audit.check('same_A0_provenance', pool['historical_fit']==refs['historical_fit'])
    labels = list(pool['labels']); n=len(labels); t=96; m=n*t
    audit.check('same_label_order', labels==fit['terminal_labels']==history['labels']==saved['labels'])
    audit.check('unchanged_power_base', pool['base_kw']==history['base_kw']==200.)
    audit.check('full_history_rows', len(history['training'])==len(history['validation'])==3
        and [sum(len(s['P_terminal']) for s in history[k]) for k in ('training','validation')]==[288,288]
        and fit['training_rows']==fit['validation_rows']==288)
    for split in ('training','validation'):
        audit.check('fit_input_hash:'+split, observation_hash(history[split])==fit['input_sha256'][split])
    pool_value={k:v for k,v in pool.items() if k!='pool_sha256'}
    pool_hash=hashlib.sha256(json.dumps(pool_value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
    audit.check('pool_internal_fingerprint', pool_hash==pool['pool_sha256'])
    audit.check('saved_pool_unchanged', saved['pool']==pool)
    audit.check('resources_unchanged', saved['resources']==resources)
    r0=np.asarray(fit['fronts']['raw']['R']); x0=np.asarray(fit['fronts']['raw']['X'])
    audit.same('A0_R_reused',saved['r0'],r0,atol=0,rtol=0)
    audit.same('A0_X_reused',saved['x0'],x0,atol=0,rtol=0)
    audit.check('historical_A0_solver_success',fit['fronts']['raw']['diagnostics']['success'])
    models={a['candidate_id']:a for a in pool['models']}
    order=pool['selection_order']; audit.check('locked_pair',order==['coarse','refined'])
    matrices={}
    for key,model in models.items():
        point=model['point']; rr=np.zeros((n,n)); xx=np.zeros((n,n))
        for block,r,x in zip(point['supports'],point['r'],point['x'],strict=True):
            rr[np.ix_(block,block)]+=r; xx[np.ix_(block,block)]+=x
        matrices[key]=(rr,xx)
        audit.same(key+':R_from_fixed_weights',rr,model['R'],atol=1e-14)
        audit.same(key+':X_from_fixed_weights',xx,model['X'],atol=1e-14)
        cal=pool['calibration']; threshold=.16*cal['median_root_depth']
        frozen=[b for b,r,x in zip(point['supports'],point['r'],point['x'],strict=True)
                if len(b)==1 or cal['a']*r+cal['b']*x+1e-10>=threshold]
        audit.check(key+':frozen_contraction',frozen==model['frozen_output_supports'])
    training=[]; H={}; trainloss={}
    for scene in history['training']:
        pp,qq,yy=power_target(scene,labels)
        audit.check('complete_training_day:'+scene['name'],pp.shape==(96,n))
        training.append(np.stack([pp,qq,yy-pp@r0.T-qq@x0.T]))
    for key,(rr,xx) in matrices.items():
        split_losses={}
        for split in ('training','validation'):
            total=0.;count=0
            for scene in history[split]:
                pp,qq,yy=power_target(scene,labels);total+=np.abs(yy-pp@rr.T-qq@xx.T).sum();count+=yy.size
            split_losses[split]=total/count
            audit.same(key+':historical_'+split+'_MAE',total/count,models[key]['historical_'+split+'_mae'],atol=1e-12)
        H[key]=split_losses['validation']*1e6;trainloss[key]=split_losses['training']
    a=np.stack(training);mu=a.mean(0);noise=(a-mu).reshape(3,-1).T/np.sqrt(2.)
    audit.same('joint_history_mu',saved['mu'],mu,atol=1e-13)
    audit.same('joint_history_factor',saved['noise'],noise,atol=1e-13)
    audit.check('rank_two_three_day_law',np.linalg.matrix_rank(noise)==2 and noise.shape==(4320,3))
    audit.check('no_extra_refit',planning['new_fit_calls']==0)
    residual={};effects={}
    for short,key in [('C','coarse'),('F','refined')]:
        rr,xx=matrices[key]
        allr=(a[:,2]+np.einsum('btn,jn->btj',a[:,0],r0-rr)+np.einsum('btn,jn->btj',a[:,1],x0-xx)).reshape(3,-1)*1e6
        mean=allr.mean(0);fac=(allr-mean).T/np.sqrt(2.)
        effect=np.stack([(r0-rr)[:,labels.index(v['node_label'])]+float(v['reactive_ratio'])*(x0-xx)[:,labels.index(v['node_label'])]
            for v in resources],axis=1)*1e6/pool['base_kw']
        residual[short]=(mean,fac);effects[short]=effect
        audit.same('residual_mean_'+short,saved['mu_'+short],mean)
        audit.same('residual_factor_'+short,saved['L_'+short],fac)
        audit.same('resource_effect_'+short,saved['effect_'+short],effect)
    lower=np.stack([np.maximum(float(v.get('power_lower_kw',0))-np.broadcast_to(v['baseline_kw'],96),-1.) for v in resources])
    upper=np.stack([np.minimum(float(v['power_upper_kw'])-np.broadcast_to(v['baseline_kw'],96),1.) for v in resources])
    cv=np.full((2,96),np.sqrt(.025/3)); elasticity=np.full((2,96),.5)
    audit.same('original_mean_lower',saved['lower'],lower,atol=0,rtol=0)
    audit.same('original_mean_upper',saved['upper'],upper,atol=0,rtol=0)
    audit.same('response_CV',saved['cv'],cv,atol=0,rtol=0)
    audit.same('price_elasticity',saved['elasticities'],elasticity,atol=0,rtol=0)
    audit.check('main_independent_response_factor',saved['response_factor'] is None)
    threshold=1.5*(1.03*H['refined']-H['coarse'])+2.5e-6
    audit.same('full_history_threshold_micro',saved['threshold'],threshold,atol=1e-9)
    audit.same('full_history_H',list(saved['H'].values()),list(H.values()),atol=1e-8)
    audit.check('original_rule_3pct_w2',protocol['rule']==dict(historical_fraction=.03,new_fraction=.03,new_weight=2,epsilon=1e-12,history_gate=True))
    audit.check('kappa_1_03',saved['kappa']==1.03)
    oldroot=PRIVATE/'wzzt_gaussian_margin_stage/run_v1'
    oldref=audit.read(oldroot/'problem_frozen.json')
    old=audit.load(audit.checked(oldref['input'],'old_working_law_hash'))
    audit.check('old_same_frozen_pool',old['pool']['pool_sha256']==pool['pool_sha256'])
    oldmu=np.asarray(old['mu']);oldnoise=np.asarray(old['noise']).reshape(3,t,n,3)
    nf=noise.reshape(3,t,n,3);rf,xf=matrices['refined']
    yymean=mu[2]+mu[0]@r0.T+mu[1]@x0.T
    oldyymean=oldmu[2]+oldmu[0]@rf.T+oldmu[1]@xf.T
    yyfactor=nf[2]+np.einsum('tnf,jn->tjf',nf[0],r0)+np.einsum('tnf,jn->tjf',nf[1],x0)
    oldyyfactor=oldnoise[2]+np.einsum('tnf,jn->tjf',oldnoise[0],rf)+np.einsum('tnf,jn->tjf',oldnoise[1],xf)
    audit.same('zero_DR_PQ_mean_same',mu[:2],oldmu[:2],atol=1e-14)
    audit.same('zero_DR_PQ_factors_same',nf[:2],oldnoise[:2],atol=1e-14)
    audit.same('zero_DR_Y_mean_same',yymean,oldyymean,atol=1e-13)
    audit.same('zero_DR_Y_factors_same',yyfactor,oldyyfactor,atol=1e-13)
    N=int(protocol['evaluation_paths']);audit.check('independent_denominator_2000',N==2000 and pred['independent_denominator']==N)
    audit.check('separate_random_seeds',len({protocol[k] for k in ['integration_seed','qualification_seed','evaluation_seed']})==3)
    audit.opened.extend([str(root/'evaluation_draws.npz'),str(root/'evaluation_scalars.npz')])
    draws=np.load(root/'evaluation_draws.npz');scalars=np.load(root/'evaluation_scalars.npz')
    zb=draws['zbase'];zd=draws['zdr'];rng=np.random.default_rng(protocol['evaluation_seed'])
    audit.same('saved_base_draws_exact_seed',zb,rng.normal(size=(N,3)),atol=0,rtol=0)
    audit.same('saved_response_draws_exact_seed',zd,rng.normal(size=(N,2,96)),atol=0,rtol=0)
    base=(mu.ravel()[None]+zb@noise.T).reshape(N,3,t,n)
    rc0=residual['C'][0][None]+zb@residual['C'][1].T
    rf0=residual['F'][0][None]+zb@residual['F'][1].T
    D0=np.abs(rc0).mean(1)-1.03*np.abs(rf0).mean(1)
    from research_experiments.wzzt_dr_selection.dr_newday_weighted.weighted_selection import score
    expected_rows={}; selections={}; max_shared=0.; max_direct=0.; scorer_calls=0
    policies=freeze['policies'];predrows={r['policy']:r for r in pred['rows']}
    audit.check('all_policies_reported',set(policies)==set(predrows))
    quote=np.asarray(policies['continuous']['mean_u']);quote_cost=.25*np.sum(quote**2/elasticity)
    parent_u=np.asarray(parent_price['policies']['continuous']['mean_u'])
    def log_delivery(a):
        a=np.asarray(a);sd=cv*np.abs(a);nz=sd>0;failure=np.zeros_like(a)
        failure[nz]=ndtr((lower[nz]-a[nz])/sd[nz])+ndtr((a[nz]-upper[nz])/sd[nz])
        failure[~nz]=((a[~nz]<lower[~nz])|(a[~nz]>upper[~nz])).astype(float)
        return float(np.log1p(-failure).sum())
    independent_alpha=brentq(lambda z:log_delivery(z*parent_u)-np.log(.95),0.,1.,xtol=5e-15,rtol=1e-14)
    audit.same('maximal_derating_alpha',planning['alpha'],independent_alpha,atol=2e-13,rtol=1e-12)
    audit.same('protocol_derating_alpha',protocol['optimizer']['alpha'],independent_alpha,atol=2e-13,rtol=1e-12)
    audit.same('selected_quote_is_radial_derating',quote,planning['alpha']*parent_u,atol=1e-13)
    audit.check('all_four_prelocked_policies',set(policies)=={'zero','continuous','rotation','single'})
    audit.check('no_added_source_or_case',protocol['inputs']==parent_protocol['inputs'])
    delivery_probabilities={};violations_by_policy={} 
    for name,policy in policies.items():
        u=np.asarray(policy['mean_u']);cost=float(.25*np.sum(u**2/elasticity))
        audit.check(name+':mean_domain',u.shape==(2,96) and np.all(u>=lower-1e-9) and np.all(u<=upper+1e-9))
        audit.same(name+':prepaid_fee',cost,policy['planned_cost'],atol=1e-10)
        if name not in ('zero','continuous'):audit.same(name+':matched_fee',cost,quote_cost,atol=1e-9)
        if name=='zero':audit.same('zero_quote_exactly_zero',u,np.zeros_like(u),atol=0,rtol=0)
        analytic_delivery=float(np.exp(log_delivery(u)));delivery_probabilities[name]=analytic_delivery
        audit.same(name+':analytic_delivery_probability',analytic_delivery,policy['analytic_component_bounds_probability'],atol=2e-12)
        audit.check(name+':delivery_flag',bool(analytic_delivery>=.95-1e-12)==policy['delivery95_feasible'])
        if name=='single':audit.check('single_mean_feasible_but_not_delivery95',np.all(u>=lower) and np.all(u<=upper) and analytic_delivery<.95)
        if name=='continuous':audit.same('continuous_delivery_target',analytic_delivery,.95,atol=2e-12)
        U=u[None]*(1+cv[None]*zd)  # Shared physical execution multipliers, no clipping.
        rc=rc0+np.einsum('nr,srt->stn',effects['C'],U).reshape(N,m)
        rf=rf0+np.einsum('nr,srt->stn',effects['F'],U).reshape(N,m)
        ec=np.abs(rc).mean(1);ef=np.abs(rf).mean(1);D=ec-1.03*ef;event=D>threshold
        chosen=[];exits=[];raw=[];final=[];minp=[]
        for j in range(N):
            pp=base[j,0].copy();qq=base[j,1].copy()
            for aidx,resource in enumerate(resources):
                pos=labels.index(resource['node_label']);pp[:,pos]+=U[j,aidx]/pool['base_kw']
                qq[:,pos]+=float(resource['reactive_ratio'])*U[j,aidx]/pool['base_kw']
            yy=pp@r0.T+qq@x0.T+base[j,2]
            direct={k:(yy-pp@matrices[k][0].T-qq@matrices[k][1].T)*1e6 for k in order}
            max_direct=max(max_direct,float(np.max(np.abs(direct['coarse'].ravel()-rc[j]))),float(np.max(np.abs(direct['refined'].ravel()-rf[j]))))
            shared=pp@(matrices['refined'][0]-matrices['coarse'][0]).T+qq@(matrices['refined'][1]-matrices['coarse'][1]).T
            max_shared=max(max_shared,float(np.max(np.abs(direct['coarse']-direct['refined']-shared*1e6))))
            losses={k:(.6*H[k]+.4*np.abs(direct[k]).mean())/1e6 for k in order}
            best=min(losses.values());eligible=[k for k in order if losses[k]<=1.03*best+1e-12]
            independent_choice=min(eligible,key=lambda k:(len(models[k]['point']['supports']),losses[k],order.index(k)))
            result=score(pool,dict(P_terminal=pp,Q_terminal=qq,drop_target=yy,labels=labels),history_gate=True)
            rule=next(v for v in result['rules'] if v['new_day_weight']==2 and v['new_day_fraction']==.03)
            if rule['selected_candidate_id']!=independent_choice or rule['eligible_candidate_ids']!=eligible:
                raise AssertionError('original scorer disagrees with independent arithmetic')
            if rule['selected_output_supports']!=models[independent_choice]['frozen_output_supports']:
                raise AssertionError('final output mapping changed')
            chosen.append(independent_choice);exits.append('coarse' not in eligible)
            raw.append(len(models[independent_choice]['point']['supports']))
            final.append(len(models[independent_choice]['frozen_output_supports']));minp.append(float(pp.min()));scorer_calls+=1
        violation=np.any((U<lower[None]-1e-10)|(U>upper[None]+1e-10),axis=(1,2))
        raw=np.asarray(raw);final=np.asarray(final);minp=np.asarray(minp);chosen=np.asarray(chosen)
        replay=dict(D=D,D0=D0,event=event,selected=chosen,resource_violation=violation,
                    raw_support_count=raw,final_support_count=final,minimum_post_power=minp)
        for key,value in replay.items():audit.same(name+':saved_'+key,value,scalars[name+'__'+key],atol=1e-7)
        audit.check(name+':pair_event_equals_full_selector',np.array_equal(event,exits))
        count=int(event.sum());selections[name]=chosen;violations_by_policy[name]=violation
        row=dict(paths=N,completed=N,failed=0,timeout=0,event_count=count,event_rate=count/N,
            cp_lower95=cp_lower(count,N),cp_lower_bonferroni95=cp_lower(count,N,.05/len(policies)),
            target_certified_95=cp_lower(count,N)>=.95,coarse_excluded=int(sum(exits)),refined_selected=int(np.sum(chosen=='refined')),
            planned_cost=cost,mean_D=float(D.mean()),mean_increment=float(np.mean(D-D0)),threshold=threshold,
            q05_margin=float(np.quantile(D-threshold,.05)),mean_margin=float(np.mean(D-threshold)),
            new_successes_vs_zero=int(np.sum(event&(D0<=threshold))),lost_successes_vs_zero=int(np.sum(~event&(D0>threshold))),
            resource_violation_paths=int(violation.sum()),event_and_resource_valid=int(np.sum(event&~violation)),
            negative_background_paths=int(np.any(base[:,0]<0,axis=(1,2)).sum()),negative_post_power_paths=int(np.sum(minp<0)))
        for key,value in row.items():audit.same(name+':summary_'+key,value,predrows[name][key],atol=1e-7)
        expected_rows[name]=dict(policy=name,**row)
        print('audited observations',name,count,'/',N,flush=True)
    audit.check('all_saved_paths_preserved',len(scalars.files)==len(policies)*8 and all(len(scalars[name+'__event'])==N for name in policies))
    audit.check('shared_residual_identity',max_shared<1e-7,max_error_micro=max_shared)
    audit.check('direct_observation_residuals',max_direct<1e-7,max_error_micro=max_direct)
    audit.check('full_scorer_call_count',scorer_calls==pred['selection_calls']==N*len(policies))
    audit.check('no_new_identifier_or_AC_calls',pred['new_fit_calls']==pred['new_search_calls']==pred['new_AC_calls']==0)

    # Truth is opened only after every observation, event, and choice was replayed.
    audit.check('no_truth_prior_to_frozen_choices',not audit.truth_opened)
    truthpath=audit.checked(summary['truth_source'],'same_case_truth_sha')
    if 'paper15_day00_zero' not in str(truthpath):raise ValueError('unexpected historical truth scope')
    receiptpath=truthpath.parent/'events_receipt.json'
    receipt=audit.read(receiptpath)
    audit.check('truth_receipt_hash',receipt['private_files']['truth_evaluation.json']['sha256']==sha(truthpath))
    truthdata=audit.read(truthpath);audit.truth_opened=True
    audit.check('truth_labels',truthdata['terminal_labels']==labels)
    truth=canonical(truthdata['true_supports'])
    audit.check('zero_weak_truth_convention',truthdata['zero_impedance_merged'] is True and truthdata['arbitrarily_weak_positive_lines_retained'] is True)
    audit.check('truth_denominator',len(truth)==truthdata['full_effective_truth_blocks'])
    def errors(blocks):
        s=canonical(blocks)
        return dict(missing=len(truth-s),extra=len(s-truth),errors=len(truth^s),complete=bool(s==truth))
    model_errors={}
    for key,mod in models.items():
        raw=[[labels[i] for i in b] for b in mod['point']['supports']]
        model_errors[key]=dict(raw=errors(raw),final=errors(mod['frozen_output_support_labels']))
        audit.check(key+':raw_final_truth_accounting',model_errors[key]==summary['models'][key])
    zero_errors=np.array([model_errors[k]['final']['errors'] for k in selections['zero']])
    output_rows={r['policy']:r for r in summary['rows']}
    for name,choices in selections.items():
        finalerr=np.array([model_errors[k]['final']['errors'] for k in choices])
        rawerr=np.array([model_errors[k]['raw']['errors'] for k in choices])
        extra=dict(final_errors_sum=int(finalerr.sum()),raw_errors_sum=int(rawerr.sum()),full_recoveries=int(np.sum(finalerr==0)),
            improved_vs_zero=int(np.sum(finalerr<zero_errors)),worse_vs_zero=int(np.sum(finalerr>zero_errors)),equal_vs_zero=int(np.sum(finalerr==zero_errors)))
        for key,value in extra.items():audit.same(name+':label_summary_'+key,value,output_rows[name][key],atol=0,rtol=0)
        expected_rows[name].update(extra)
        expected_rows[name]['analytic_all_day_delivery_probability']=delivery_probabilities[name]
    continuous_errors=np.asarray([model_errors[k]['final']['errors'] for k in selections['continuous']])
    rotation_errors=np.asarray([model_errors[k]['final']['errors'] for k in selections['rotation']])
    def comparison(mask):
        return dict(denominator=int(mask.sum()),continuous_better=int(np.sum((continuous_errors<rotation_errors)&mask)),
            continuous_worse=int(np.sum((continuous_errors>rotation_errors)&mask)),same=int(np.sum((continuous_errors==rotation_errors)&mask)),
            continuous_full_recoveries=int(np.sum((continuous_errors==0)&mask)),rotation_full_recoveries=int(np.sum((rotation_errors==0)&mask)),
            continuous_error_sum=int(continuous_errors[mask].sum()),rotation_error_sum=int(rotation_errors[mask].sum()))
    paired=dict(all_paths=comparison(np.ones(N,dtype=bool)),
        both_deliveries_within_bounds=comparison(~violations_by_policy['continuous']&~violations_by_policy['rotation']),
        subset_scope='Post-execution paired subset is descriptive only, not an unconditional recovery probability',
        violations_overlap=int(np.sum(violations_by_policy['continuous']&violations_by_policy['rotation'])))
    audit.check('paired_count_complete',sum(paired['all_paths'][k] for k in ['continuous_better','continuous_worse','same'])==N)
    audit.check('no_policy_certified_excitation95',all(not r['target_certified_95'] for r in expected_rows.values()))
    return dict(paired_continuous_vs_rotation=paired,alpha=independent_alpha,analytic_delivery_probabilities=delivery_probabilities,rows=list(expected_rows.values()),models=model_errors,truth_source=summary['truth_source'],
        original_history_rows=[288,288],independent_full_paths=N,policies=len(policies),scorer_calls=scorer_calls,
        source_and_output_hashes={name:sha(root/name) for name in required},
        probability_scope='fixed quoted policy under stated A0/Gaussian working law; no population calibration or AC recovery certificate',
        identity_scope='all original records and final frozen output maps; no fit, solver, new experiment, redraw, or clipping')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,default=STAGE/'delivery_v1')
    ap.add_argument('--output',type=Path,default=STAGE/'audit_delivery_v1.json');args=ap.parse_args()
    audit=Audit();result={}
    try:
        result=audit_run(args.run,audit)
        success=all(v['passed'] for v in audit.checks)
        result.update(status='passed' if success else 'failed',audit_passed=success)
    except Exception as exc:
        result.update(status='failed',audit_passed=False,error_type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc())
    result.update(created_utc=now(),audit_source_sha256=sha(__file__),check_count=len(audit.checks),
        failed_checks=[v for v in audit.checks if not v['passed']],checks=audit.checks,
        truth_opened=audit.truth_opened,opened_artifacts=audit.opened)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({k:result.get(k) for k in ('status','audit_passed','check_count','error_type','message')},ensure_ascii=False),flush=True)
    raise SystemExit(0 if result['audit_passed'] else 1)


if __name__=='__main__':main()
