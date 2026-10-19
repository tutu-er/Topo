"""Full Gaussian directional derivatives, for diagnosis rather than planning.

The current MILP freezes covariance. These helpers expose the covariance term
and independently check the stated predictive-risk gradient with common noise.
No AC fit, future truth, topology selection, or policy update is performed here.
"""
from __future__ import annotations

import numpy as np
from . import ev_bayes_small_probability as probability


def _directions(bank, mean_direction, covariance_direction):
    b=probability.validate_bank(bank)
    dm=np.asarray(mean_direction,float)
    dc=np.zeros_like(b['covariances']) if covariance_direction is None else np.asarray(covariance_direction,float)
    if dm.shape!=b['means'].shape or dc.shape!=b['covariances'].shape:
        raise ValueError('direction must match bank mean/covariance shapes')
    if not np.isfinite(dm).all() or not np.isfinite(dc).all(): raise ValueError('finite direction required')
    if not np.allclose(dc,dc.swapaxes(-1,-2),atol=1e-12,rtol=1e-10):
        raise ValueError('symmetric covariance direction required')
    return b,dm,.5*(dc+dc.swapaxes(-1,-2))


def bhattacharyya_directional(bank,mean_direction,covariance_direction=None):
    """Derivative per unit scalar path, split into mean and covariance terms."""
    b,dm,dc=_directions(bank,mean_direction,covariance_direction)
    distances=probability.bhattacharyya_bank(bank);rows=[]
    mean_total=cov_total=0.
    for pair in distances['pairs']:
        l,i,j=pair['l'],pair['i'],pair['j']
        S=.5*(b['covariances'][l,i]+b['covariances'][l,j]);dS=.5*(dc[l,i]+dc[l,j])
        inv=np.linalg.inv(S);delta=b['means'][l,i]-b['means'][l,j];v=inv@delta
        mean=float(.25*(dm[l,i]-dm[l,j])@v)
        covariance=float(-.125*v@dS@v+.5*np.trace(inv@dS)-
            .25*(np.trace(b['_models'][l]['precision'][i]@dc[l,i])+
                 np.trace(b['_models'][l]['precision'][j]@dc[l,j])))
        dC_mean=-pair['weighted_contribution']*mean
        dC_cov=-pair['weighted_contribution']*covariance
        mean_total+=dC_mean;cov_total+=dC_cov
        rows.append(dict(l=l,i=i,j=j,B=pair['B'],dB_mean=mean,dB_covariance=covariance,
            dB_full=mean+covariance,dC_mean=dC_mean,dC_covariance=dC_cov,dC_full=dC_mean+dC_cov))
    return dict(C=distances['bound'],dC_mean=float(mean_total),dC_covariance=float(cov_total),
        dC_full=float(mean_total+cov_total),pairs=rows,
        scope='exact directional derivative of the supplied differentiable Gaussian bank; historical weights fixed')


def cholesky_directional(cholesky,covariance_direction):
    """dL=L*lower(L^-1 dSigma L^-T), with half diagonal."""
    L=np.asarray(cholesky,float);dS=np.asarray(covariance_direction,float)
    E=np.linalg.solve(L,np.linalg.solve(L,dS).T).T
    lower=np.tril(E);idx=np.diag_indices_from(lower);lower[idx]*=.5
    return L@lower


def _summary(samples,weights):
    R=samples.shape[-1]
    return dict(value=float(np.sum(weights*samples.mean(-1))),
        standard_error=float(np.sqrt(np.sum(weights**2*samples.var(-1,ddof=1)/R))))


def risk_directional(bank,noise,mean_direction,covariance_direction=None,*,return_samples=False):
    """Risk derivative by posterior paths and the independent envelope score.

    R=1-integral max_k pi_k f_k.  Its score representation is
    -E[1{MAP(Y)=K} dlog f_K(Y)], where the log-density derivative holds Y fixed.
    The path representation also differentiates Y=mu_K+chol(Sigma_K)*noise.
    They estimate the same integral and have different Monte Carlo variance.
    """
    b,dm,dc=_directions(bank,mean_direction,covariance_direction)
    noise=np.asarray(noise,float);L,K,d=b['means'].shape
    if noise.ndim!=4 or noise.shape[:2]!=(L,K) or noise.shape[-1]!=d or noise.shape[2]<2 or not np.isfinite(noise).all():
        raise ValueError('noise must have shape [L,K,R,d], R>=2')
    R=noise.shape[2];weights=b['weights'][:,None]*b['prior'][None,:]
    samples={key:np.zeros((L,K,R)) for key in ('score_mean','score_covariance','score_full','path_mean','path_covariance','path_full')}
    ties=0
    for l in range(L):
        model=b['_models'][l]
        for k in range(K):
            eps=noise[l,k];Y=b['means'][l,k]+eps@b['cholesky'][l,k].T
            posterior=probability.gaussian_posterior(Y,b['means'][l],b['covariances'][l],log_prior=b['log_prior'])['posterior']
            winner=posterior.argmax(-1);correct=(winner==k)
            if K>1:
                ordered=np.sort(posterior,axis=-1);ties+=int(np.sum(abs(ordered[:,-1]-ordered[:,-2])<1e-12))
            residual=Y-b['means'][l,k];v=residual@model['precision'][k]
            score_mean=v@dm[l,k]
            score_cov=.5*(np.einsum('ri,ij,rj->r',v,dc[l,k],v)-np.trace(model['precision'][k]@dc[l,k]))
            correct_float=correct.astype(float)
            samples['score_mean'][l,k]=-correct_float*score_mean
            samples['score_covariance'][l,k]=-correct_float*score_cov
            samples['score_full'][l,k]=-correct_float*(score_mean+score_cov)
            dL=cholesky_directional(b['cholesky'][l,k],dc[l,k])
            dy_mean=np.broadcast_to(dm[l,k],(R,d));dy_cov=eps@dL.T
            for key,dy,covariance_included in (('path_mean',dy_mean,False),('path_covariance',dy_cov,True),('path_full',dy_mean+dy_cov,True)):
                dlog=np.zeros((R,K))
                for j in range(K):
                    r=Y-b['means'][l,j];vj=r@model['precision'][j]
                    moving_mean=dm[l,j] if key!='path_covariance' else np.zeros(d)
                    dlog[:,j]=np.einsum('ri,ri->r',vj,moving_mean-dy)
                    if covariance_included:
                        dlog[:,j]+=.5*(np.einsum('ri,ij,rj->r',vj,dc[l,j],vj)-np.trace(model['precision'][j]@dc[l,j]))
                dpost=posterior*(dlog-np.sum(posterior*dlog,axis=-1,keepdims=True))
                samples[key][l,k]=-dpost[np.arange(R),winner]
    result={key:_summary(value,weights) for key,value in samples.items()}
    result.update(tie_count=ties,path_valid_away_from_ties=ties==0,noise_per_stratum=R,
        scope='declared Gaussian model, fixed historical probabilities; MC SE is conditional integration error')
    if return_samples:result['samples']=samples
    return result


def risk_crn_finite_difference(bank_plus,bank_minus,noise,step):
    if not np.isfinite(step) or step<=0: raise ValueError('positive finite step required')
    compared=probability.compare_banks(bank_plus,bank_minus,noise)
    return dict(value=float(compared['risk']/(2*step)),
        standard_error=float(compared['risk_se']/(2*step)),step=float(step),
        scope='central finite difference, same standard Normal draws and historical weights')


def ac_path_diagnostic(context,power,direction,noise,*,step=1e-3):
    """Observed-only AC direction audit. Does not alter a chosen plan or model."""
    from . import ev_bayes_response_ac as ac
    power=np.asarray(power,float);direction=np.asarray(direction,float)
    if direction.shape!=power.shape or not np.isfinite(direction).all():raise ValueError('finite matching power direction required')
    center=ac.forecast_bank(context,power,with_jacobian=True)
    plus=ac.forecast_bank(context,power+step*direction,with_jacobian=True)
    minus=ac.forecast_bank(context,power-step*direction,with_jacobian=True)
    dm=np.einsum('lkdp,p->lkd',center['jacobian'],direction.ravel())
    dm_fd=(plus['bank']['means']-minus['bank']['means'])/(2*step)
    dc=(plus['bank']['covariances']-minus['bank']['covariances'])/(2*step)
    bc=bhattacharyya_directional(center['bank'],dm,dc)
    bc_fd=(probability.bhattacharyya_bank(plus['bank'])['bound']-probability.bhattacharyya_bank(minus['bank'])['bound'])/(2*step)
    risk=risk_directional(center['bank'],noise,dm,dc)
    finite=risk_crn_finite_difference(plus['bank'],minus['bank'],noise,step)
    return dict(power_kw=power,direction_kw=direction,step=float(step),BC=bc,
        BC_full_finite_difference=float(bc_fd),risk=risk,risk_CRN_finite_difference=finite,
        mean_AC_direction_max_absolute_error=float(np.max(abs(dm-dm_fd))),
        covariance_direction_frobenius_norm=float(np.linalg.norm(dc)),
        AC_states=sum(x['AC_states'] for x in (center,plus,minus)),
        covariance_derivative_source='central complete-AC propagated covariance; historical posterior fixed',
        used_future_truth=False,planning_policy_modified=False)


def context_from_saved(record):
    """Rebuild future-only context from saved posterior; no fitting/generator."""
    from . import ev_bayes_response_ac as ac
    from .information_context import meter_from_record
    from .likelihood import RadialCandidate
    prediction=record['prediction']['joint'];n=np.shape(record['observed_training']['p'])[1]
    T=int(prediction['horizon']);fits={r.get('candidate',r.get('name')):r for r in record['training_result']['fits']}
    rows=[]
    for name in record['candidate_names']:
        candidate=RadialCandidate.from_supports(name,n,[tuple(s) for s in fits[name]['supports'] if 1<len(s)<n])
        update=prediction['candidate_updates'][name];d=len(update['line_prior']['mean'])
        mean=np.asarray(update['TV_posterior_future']['mean'],float)
        covariance=np.asarray(update['TV_posterior_future']['cov'],float)
        r,x=ac._rx(mean[:d]);latent=mean[d:].reshape(T,2*n+2)
        rows.append(dict(candidate=candidate,r=r,x=x,line_dimension=d,joint_mean=mean,
            joint_covariance=covariance,latent_mean=latent,latent_cov=covariance[d:,d:],line_sha256=ac._line_id(r,x)))
    return dict(rows=rows,prior=np.exp(prediction['log_prior']),log_prior=np.asarray(prediction['log_prior']),
        log_prior_all_history=np.asarray(prediction['log_prior']),
        log_prior_evidence_only=np.asarray(prediction['log_prior_evidence_only']),
        future_meter=meter_from_record(record['future_meter']),horizon=T,n=n,
        base_kw=float(prediction['base_kw']),root_v=np.ones(T),options=prediction['options'])


def run_saved(path,output,*,samples=2048,step=1e-3):
    """All joint_bc proposal directions of predeclared development histories.

    This helper reads no true_lines/true_future_inputs/actual fields. Missing
    saved anchors are retained as failures; a rejected proposal can recover its
    anchor from the unchanged final plan when max_proposals=1.
    """
    from pathlib import Path
    from hashlib import sha256
    from time import perf_counter
    from . import gradient_price_study as io
    began=perf_counter();path=Path(path);out=Path(output)
    if out.exists():raise ValueError('preserve existing gradient evidence')
    record=io.read_json(path)
    if record['spec']['phase']!='development' or record['spec']['index'] not in (0,1):
        raise ValueError('predeclared diagnostics cover development histories 0 and 1 only')
    ctx=context_from_saved(record);seed=[int(record['spec']['seed']),7100,20261017]
    noise=np.random.default_rng(seed).standard_normal((1,len(ctx['rows']),int(samples),(3*ctx['n']+2)*ctx['horizon']))
    rows=[]
    for domain,log in record['planning_logs'].get('joint_bc',{}).items():
        for event in log['events']:
            proposal=event['milp'].get('power_kw')
            row=dict(domain=domain,iteration=event['iteration'],accepted=event['accepted'])
            if proposal is None:
                row.update(status='no_executable_proposal',failure=event['milp'].get('failure'));rows.append(row);continue
            anchor=event.get('anchor_power_kw');source='saved_anchor'
            if anchor is None and not event['accepted'] and len(log['events'])==1:
                anchor=record['methods']['joint_bc__'+domain]['power_kw'];source='unchanged_final_plan'
            if anchor is None:
                row.update(status='missing_saved_accepted_anchor');rows.append(row);continue
            anchor=np.asarray(anchor,float);direction=np.asarray(proposal,float)-anchor
            row.update(anchor_source=source,anchor_power_kw=anchor,proposal_power_kw=np.asarray(proposal),
                maximum_power_direction_kw=float(abs(direction).max()),direction_parameter='alpha, power=anchor+alpha*(proposal-anchor)')
            if np.max(abs(direction))<=1e-10:
                row.update(status='zero_proposal_direction');rows.append(row);continue
            try:row.update(status='complete',diagnostic=ac_path_diagnostic(ctx,anchor,direction,noise,step=step))
            except Exception as exc:row.update(status='diagnostic_failure',failure=type(exc).__name__+': '+str(exc))
            rows.append(row)
    result=dict(status='complete',input=str(path.resolve()),input_sha256=sha256(path.read_bytes()).hexdigest(),
        history_spec=record['spec'],noise_seed=seed,noise_per_candidate=int(samples),step=float(step),rows=rows,
        candidate_refits=0,case_generator_calls=0,optimizer_calls=0,used_future_truth=False,
        policy_changed=False,seconds=float(perf_counter()-began),
        source_sha256={name:sha256((Path(__file__).parent/name).read_bytes()).hexdigest() for name in
            ('ev_bayes_response_gradient.py','ev_bayes_response_ac.py','ev_bayes_small_probability.py')})
    io.write_json(out,result)
    return result


def main():
    import argparse
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--input',required=True)
    parser.add_argument('--output',required=True);parser.add_argument('--samples',type=int,default=2048)
    parser.add_argument('--step',type=float,default=1e-3);args=parser.parse_args()
    result=run_saved(args.input,args.output,samples=args.samples,step=args.step)
    print({key:value for key,value in result.items() if key not in ('rows','source_sha256')},flush=True)


if __name__=='__main__':main()
