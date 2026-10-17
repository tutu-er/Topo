"""Independent read-only audit of saved fixed-plan AC repair histories.

No optimizer, planner, posterior update or new scientific case is called.
Existing failed histories remain in the audited denominator. Each AC call
replays a saved parameter state or the saved evaluation-only truth trajectory.
"""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
from time import perf_counter

for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name]='1'
sys.path.insert(0,str(Path(__file__).resolve().parents[5]))
import numpy as np
from scipy.special import logsumexp

from research_experiments.mobile_ev_mle import likelihood as lk
from research_experiments.mobile_ev_mle.candidates import RadialCandidate

CHANNELS=('p','q','v','master_p','master_q')


def digest(path):
    h=sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def maximum_error(a,b):
    a,b=np.asarray(a,float),np.asarray(b,float)
    if a.shape!=b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('matching finite arrays required')
    return float(np.max(abs(a-b),initial=0.))


class Audit:
    def __init__(self):
        self.issues=[];self.ac_calls=0;self.ac_states=0;self.ac_seconds=0.

    def issue(self,category,location,**details):
        self.issues.append(dict(category=category,location=location,**details))

    def close(self,a,b,tolerance,location):
        error=maximum_error(a,b)
        if error>tolerance:self.issue('numeric_mismatch',location,error=error,tolerance=tolerance)
        return error

    def predict(self,c,r,x,p,q,sp,sq,root):
        began=perf_counter();self.ac_calls+=1;self.ac_states+=len(p)
        try:return lk.ac_predict(c,r,x,p,q,sp,sq,root)
        finally:self.ac_seconds+=perf_counter()-began


def whiten(phy,meter):
    result=[]
    for key in CHANNELS:
        value=np.asarray(phy[key],float);shape=value.shape
        if key in ('p','q','v'):value=value+np.broadcast_to(meter['bias_'+key],shape)
        result.append((value/np.broadcast_to(meter[key+'_sigma'],shape)).ravel())
    return np.concatenate(result)


def whiten_observed(obs,meter):
    """Observed meters already include their fixed bias; do not add it twice."""
    return np.concatenate([(np.asarray(obs[key])/np.broadcast_to(meter[key+'_sigma'],
        np.asarray(obs[key]).shape)).ravel() for key in CHANNELS])


def bounds_ok(value,bounds):
    value=np.asarray(value,float)
    low,high=(np.broadcast_to(np.asarray(b,float),value.shape) for b in bounds)
    return bool(np.isfinite(value).all() and np.all(value>=low-1e-12) and np.all(value<=high+1e-12))


def line_domain(state,options):
    r,x=np.asarray(state['r']),np.asarray(state['x'])
    return bool(bounds_ok(r,options['r_bounds']) and bounds_ok(x,options['x_bounds'])
        and np.all(r>0) and bounds_ok(x/r,options['ratio_bounds']))


def replay_refit_state(audit,c,state,segments,options,location):
    loss=0.;max_residual=0.;offset=0;voltage_legal=True
    for obs,meter in segments:
        p,q=np.asarray(obs['p']),np.asarray(obs['q']);T,n=p.shape
        p=p-np.broadcast_to(meter['bias_p'],p.shape)
        q=q-np.broadcast_to(meter['bias_q'],q.shape)
        sp=np.asarray(state['source_p'])[offset:offset+T]
        sq=np.asarray(state['source_q'])[offset:offset+T]
        phy=audit.predict(c,state['r'],state['x'],p,q,sp,sq,obs['root_v'])
        v=(phy['v']+np.broadcast_to(meter['bias_v'],p.shape)-obs['v'])/np.broadcast_to(meter['v_sigma'],p.shape)
        mp=(phy['master_p']-obs['master_p'])/np.broadcast_to(meter['master_p_sigma'],(T,))
        mq=(phy['master_q']-obs['master_q'])/np.broadcast_to(meter['master_q_sigma'],(T,))
        loss+=float(np.sum(v*v)+mp@mp+mq@mq)
        max_residual=max(max_residual,phy['ac_equation_residual'],phy['power_balance_residual'])
        voltage_legal &= bounds_ok(abs(phy['voltage']),options['voltage_bounds'])
        offset+=T
    if offset!=len(state['source_p']) or offset!=len(state['source_q']):
        audit.issue('source_length',location)
    legal=dict(line=line_domain(state,options),
        sourceP=bounds_ok(state['source_p'],options['source_p_bounds']),
        sourceQ=bounds_ok(state['source_q'],options['source_q_bounds']),voltage=bool(voltage_legal))
    return loss,max_residual,legal


def gaussian_probabilities(values,means,covs,log_prior):
    values=np.atleast_2d(values);scores=[]
    for mean,cov,lp in zip(means,covs,log_prior):
        L=np.linalg.cholesky(cov);z=np.linalg.solve(L,(values-mean).T)
        scores.append(lp-.5*(np.sum(z*z,axis=0)+2*np.log(np.diag(L)).sum()+len(mean)*np.log(2*np.pi)))
    scores=np.asarray(scores).T
    return np.exp(scores-logsumexp(scores,axis=1)[:,None])


def audit_banks(audit,action,K,dimension,location):
    propagation=action['propagation'];banks=propagation.get('banks',{});result={}
    if propagation['status']=='ok' and set(banks)!={'G00','G10','G01','G11'}:
        audit.issue('missing_success_banks',location,banks=list(banks))
    for row in propagation.get('rows',[]):
        if row.get('discarded_draws',0)!=0:audit.issue('posterior_draws_discarded',location,candidate=row['candidate'])
        invalid=any(row.get('violation_fractions',{}).get(key,0)>0 for key in
            ('source_p_violations','source_q_violations','terminal_voltage_violations','ac_failure'))
        if propagation['status']=='ok' and invalid:
            audit.issue('propagation_success_despite_physical_violation',location,candidate=row['candidate'])
    for label,bank in banks.items():
        where=location+'/'+label;m=np.asarray(bank['means']);C=np.asarray(bank['covariances']);lp=np.asarray(bank['log_prior'])
        if m.shape!=(1,K,dimension) or C.shape!=(1,K,dimension,dimension) or lp.shape!=(K,):
            audit.issue('bank_shape',where);continue
        if not np.isfinite(m).all() or not np.isfinite(C).all() or not np.isfinite(lp).all():
            audit.issue('nonfinite_bank',where);continue
        normalized=abs(float(logsumexp(lp)))
        if normalized>1e-9:audit.issue('prior_normalization',where,error=normalized)
        prior_error=audit.close(np.exp(lp),bank['prior'],1e-12,where+'/prior')
        asymmetry=float(np.max(abs(C-C.swapaxes(-1,-2))))
        if asymmetry>1e-9:audit.issue('covariance_asymmetry',where,error=asymmetry)
        positive=True
        for candidate in range(K):
            try:np.linalg.cholesky(C[0,candidate])
            except np.linalg.LinAlgError:positive=False;audit.issue('covariance_not_PD',where,candidate=candidate)
        result[label]=dict(prior_error=prior_error,covariance_asymmetry=asymmetry,positive_definite=positive)
    if 'G00' in banks:
        base=banks['G00']
        for label,bank in banks.items():audit.close(bank['log_prior'],base['log_prior'],1e-12,location+'/'+label+'/same_prior')
        if all(label in banks for label in ('G10','G01','G11')):
            for a,field,b in (('G10','covariances','G00'),('G01','means','G00'),('G11','means','G10'),('G11','covariances','G01')):
                audit.close(banks[a][field],banks[b][field],1e-12,location+'/'+a+'/'+field+'_wiring')
            for j,row in enumerate(propagation.get('rows',[])):
                if 'mean' not in row or 'covariance' not in row:continue
                audit.close(banks['G11']['means'][0][j],row['mean'],1e-9,location+'/row_mean')
                audit.close(banks['G11']['covariances'][0][j],row['covariance'],1e-8,location+'/row_covariance')
    for label,risk in action.get('declared_risk',{}).items():
        where=location+'/'+label+'/declared_risk'
        for key in ('risk','map_error','accuracy','abstention_rate','abstention_risk'):
            if key in risk and not -1e-12<=risk[key]<=1+1e-12:audit.issue('risk_outside_unit_interval',where,field=key)
        audit.close(risk['risk']+risk['accuracy'],1.,1e-12,where+'/risk_accuracy')
        if risk.get('strata'):
            for metric in ('risk','map_error','brier_risk','abstention_risk'):
                weighted=sum(s['weight']*s['means'][metric] for s in risk['strata'])
                audit.close(risk[metric],weighted,2e-12,where+'/'+metric+'_strata')
    return result


def audit_history(path,audit=None):
    began=perf_counter();audit=Audit() if audit is None else audit;initial_hash=digest(path)
    with gzip.open(path,'rt',encoding='utf-8') as stream:r=json.load(stream)
    out=dict(path=str(path.resolve()),raw_sha256=initial_hash,spec=r.get('spec'),status=r.get('status'),
        saved_failures=r.get('failures',{}),source_unchanged=r.get('source_unchanged_during_run'),
        actions=[],preserved_in_denominator=True)
    if r.get('status')=='runner_failure':
        out.update(issues=[],audit_cost=dict(AC_calls=0,AC_states=0,seconds=perf_counter()-began));return out
    if r.get('source_unchanged_during_run') is False:audit.issue('science_source_changed_during_run','history')
    cs={x['name']:RadialCandidate.from_supports(x['name'],x['n'],[s for s in x['supports'] if 1<len(s)<x['n']]) for x in r['candidate_structures']}
    n=len(r['observed_training']['p'][0]);T=r['config']['horizon'];K=len(cs);dimension=T*(3*n+2)
    options=dict(r['refit_options']);options.setdefault('voltage_bounds',(.85,1.15))
    names=r['candidate_names'];white=np.asarray(r['actual_white_noise']);truth_name=r['true_name'];truth_index=r['true_index']
    if names[truth_index]!=truth_name:audit.issue('true_index_name','history')
    history_segments=[(r['observed_training'],r['train_meter']),(r['observed_evidence'],r['evidence_meter'])]
    if white.shape!=(r['config']['noise_repeats'],dimension):audit.issue('actual_white_shape','history')
    expected_white=np.random.default_rng(np.random.SeedSequence([r['spec']['seed'],4000,20261017])).standard_normal(white.shape)
    audit.close(white,expected_white,0.,'history/actual_noise_seed_replay')
    integration=np.random.default_rng(np.random.SeedSequence([r['spec']['seed'],3000,20261017])).standard_normal(
        (1,K,r['config']['independent_samples_per_candidate'],dimension))
    expected_integration_sha=sha256(integration.tobytes()).hexdigest()
    sse_states=0;sse_legal=0;sse_error=0.;physical_error=0.;probability_rows=0;noise_hashes=set()
    for key,a in r['actions'].items():
        location='action/'+key;power=np.asarray(a['power'],float)
        canonical=power.astype('<f8',copy=True);canonical[canonical==0.]=0.
        action_hash=sha256(np.asarray(power.shape,dtype='<i8').tobytes()+canonical.tobytes()).hexdigest()
        if action_hash!=key:audit.issue('action_hash',location)
        decoded=[]
        for method in a['methods']:
            if method not in r['plans'] or r['plans'][method]['action_key']!=key:audit.issue('method_action_link',location,method=method)
            else:audit.close(r['plans'][method]['power'],power,0.,location+'/'+method+'/same_power')
            decoded.append(method)
        ar=dict(key=key,methods=decoded,propagation_status=a['propagation']['status'],failures=a.get('failures',{}),
            banks=audit_banks(audit,a,K,dimension,location),regimes={})
        for risk in a.get('declared_risk',{}).values():noise_hashes.add(risk['noise_sha256'])
        for regime,g in a['regimes'].items():
            where=location+'/'+regime;clean=np.asarray(g['clean']);values=np.asarray(g['values'])
            white_error=audit.close(values-clean[None],white,2e-9,where+'/same_white_noise')
            truth=g['truth_inputs_evaluation_only'];phy=audit.predict(cs[truth_name],
                r['true_lines_evaluation_only']['r'],r['true_lines_evaluation_only']['x'],
                np.asarray(truth['p'])+power/r['provenance']['base_kw'],truth['q'],
                truth['source_p'],truth['source_q'],truth['root_v'])
            EV_once_error=audit.close(whiten(phy,r['future_meter']),clean,2e-8,where+'/EV_once_clean_AC')
            audit.close(whiten_observed(g['clean_observations'],r['future_meter']),clean,2e-8,where+'/clean_observation')
            for j,observation in enumerate(g['observations']):
                audit.close(whiten_observed(observation,r['future_meter']),values[j],2e-8,where+'/observed_whitening/'+str(j))
            ar['regimes'][regime]=dict(noise_max_error=white_error,EV_once_clean_max_error=EV_once_error,
                predictions={},saved_refit_complete=[z.get('complete') for z in g['refits']])
            for label,bank in a['propagation'].get('banks',{}).items():
                try:
                    posterior=gaussian_probabilities(np.vstack((clean,values)),np.asarray(bank['means'])[0],np.asarray(bank['covariances'])[0],bank['log_prior'])
                    probability_rows+=len(posterior)
                    audit.close(posterior.sum(1),np.ones(len(posterior)),2e-12,where+'/'+label+'/posterior_sum')
                    ar['regimes'][regime]['predictions'][label]=dict(clean_choice=int(posterior[0].argmax()),
                        noisy_choices=posterior[1:].argmax(1).tolist(),clean_true_probability=float(posterior[0,truth_index]))
                except Exception as exc:audit.issue('probability_replay',where+'/'+label,error=str(exc))
            pairs=list(zip(g['observations'],g['refits']))
            if 'comparisons' in g.get('clean_refit',{}):pairs.append((g['clean_observations'],g['clean_refit']))
            for j,(E,z) in enumerate(pairs):
                rz=where+'/refit/'+str(j)
                if not z.get('complete') and (not z.get('keep_all_on_failure') or set(z.get('retained_on_failure',[]))!=set(names)):
                    audit.issue('failure_candidates_not_retained',rz)
                for name,row in z.get('comparisons',{}).items():
                    if 'fixed' in row and 'joint' in row and row['joint']['loss']>row['fixed']['loss']+1e-8*(1+abs(row['fixed']['loss'])):
                        audit.issue('joint_worse_than_fixed',rz+'/'+name)
                    for route in ('fixed','joint'):
                        if route not in row or 'state' not in row[route]:continue
                        state=row[route]['state']
                        loss,residual,legal=replay_refit_state(audit,cs[name],state,history_segments+[(E,r['future_meter'])],options,rz+'/'+name+'/'+route)
                        sse_states+=1;sse_legal+=int(all(legal.values()));sse_error=max(sse_error,abs(loss-row[route]['loss']));physical_error=max(physical_error,residual)
                        if abs(loss-row[route]['loss'])>1e-7*(1+abs(loss)):audit.issue('SSE_replay',rz+'/'+name+'/'+route,error=abs(loss-row[route]['loss']))
                        if row[route].get('usable') and not all(legal.values()):audit.issue('usable_state_outside_domain',rz+'/'+name+'/'+route,legal=legal)
        out['actions'].append(ar)
    if len(noise_hashes)>1:audit.issue('different_integration_noise','history',hashes=sorted(noise_hashes))
    if noise_hashes and noise_hashes!={expected_integration_sha}:
        audit.issue('integration_noise_seed_replay','history',expected=expected_integration_sha,actual=sorted(noise_hashes))
    final_hash=digest(path)
    if final_hash!=initial_hash:audit.issue('raw_changed_during_audit','history')
    out.update(raw_unchanged=initial_hash==final_hash,issues=audit.issues,
        replay_summary=dict(SSE_states=sse_states,SSE_legal_states=sse_legal,SSE_max_abs_error=sse_error,
            AC_equation_balance_max=physical_error,probability_rows=probability_rows,
            unique_actions=len(r['actions']),saved_methods=len(r['plans'])),
        audit_cost=dict(AC_calls=audit.ac_calls,AC_states=audit.ac_states,AC_seconds=audit.ac_seconds,
            seconds=perf_counter()-began))
    return out


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--phase',choices=('development','calibration','evaluation'))
    parser.add_argument('--expected-histories',type=int)
    parser.add_argument('--limit',type=int);parser.add_argument('--overwrite',action='store_true')
    args=parser.parse_args();root=Path(args.root).resolve();output=Path(args.output).resolve()
    if output.exists() and not args.overwrite:raise FileExistsError(str(output))
    paths=sorted((root/args.phase if args.phase else root).glob('**/history_*.json.gz'))
    if args.limit is not None:paths=paths[:args.limit]
    if not paths:raise FileNotFoundError('no saved histories in '+str(root))
    began=perf_counter();rows=[]
    for path in paths:
        counter=Audit();entry_started=perf_counter()
        try:row=audit_history(path,counter)
        except Exception as exc:row=dict(path=str(path),raw_sha256=digest(path),
            audit_exception=type(exc).__name__+': '+str(exc),
            issues=counter.issues+[dict(category='audit_exception',error=str(exc))],
            preserved_in_denominator=True,audit_cost=dict(AC_calls=counter.ac_calls,
                AC_states=counter.ac_states,AC_seconds=counter.ac_seconds,
                seconds=perf_counter()-entry_started))
        rows.append(row);print('AUDITED',path.name,'issues',len(row['issues']),flush=True)
    result=dict(root=str(root),script_sha256=digest(__file__),scope='saved state replay only; no new scientific sample, fitting or planning',
        histories=rows,summary=dict(histories=len(rows),status_counts=dict(Counter(r.get('status','audit_exception') for r in rows)),
            histories_with_issues=sum(bool(r['issues']) for r in rows),issues=sum(len(r['issues']) for r in rows),
            audit_AC_calls=sum(r['audit_cost']['AC_calls'] for r in rows),audit_AC_states=sum(r['audit_cost']['AC_states'] for r in rows),
            audit_seconds=perf_counter()-began),failures_preserved=True)
    result['summary']['expected_histories']=args.expected_histories
    result['summary']['missing_raw_histories']=(None if args.expected_histories is None else
        max(0,args.expected_histories-len(rows)))
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print('SAVED',output,json.dumps(result['summary']),flush=True)


if __name__=='__main__':main()
