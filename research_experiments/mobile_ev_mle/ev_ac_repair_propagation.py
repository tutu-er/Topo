"""Complete-AC moments of the existing approximate historical posterior.

The historical likelihood is unchanged. Gaussian shape after propagation is
still an approximation. Invalid draws are counted and retained, never removed
and renormalized. This module does not fit or select an EV plan.
"""
from __future__ import annotations

from time import perf_counter
from hashlib import sha256
import json
import numpy as np
from scipy.special import ndtri
from scipy.stats import qmc

from . import ev_bayes_response_ac as old
from . import ev_bayes_response_truncation as trunc
from .symmetric_ac_local import CHANNELS


def _factor(cov):
    cov, diagnostic = old.symmetric_psd(cov)
    values, vectors = np.linalg.eigh(cov)
    return vectors * np.sqrt(np.maximum(values, 0.)), diagnostic


def posterior_fingerprint(update):
    """Identify the exact Gaussian, physical domain and GHK settings."""
    h=sha256()
    for key in ('mean','cov'):
        value=np.asarray(update['TV_untruncated_future'][key],dtype='<f8')
        h.update(key.encode());h.update(str(value.shape).encode());h.update(value.tobytes())
    serial=dict(bounds=update['line_prior']['bounds'],settings=update['TV_integration_options'])
    h.update(json.dumps(serial,sort_keys=True,separators=(',',':'),
        default=lambda a:a.tolist() if isinstance(a,np.ndarray) else a.item()).encode('utf-8'))
    return h.hexdigest()


def batched_ac(candidate, r, x, latent, root_v, meter, *, power=None,
               base_kw=20., options=None, tol=2e-13, max_iter=400):
    """Vectorize old radial AC over draws and time, retaining every draw.

    ``latent`` is [draw,time,2*n+2], in per unit, before EV. ``power`` is
    [time,n] kW. R/X are [draw,m], shared over time for each draw. The returned
    whitened mean uses the original channel-block ordering p,q,v,rootP,rootQ.
    """
    began = perf_counter(); opts = old._options(options)
    latent = np.asarray(latent, float).copy(); r = np.asarray(r, float); x = np.asarray(x, float)
    if latent.ndim != 3 or latent.shape[2] != 2*candidate.n+2:
        raise ValueError('latent must be draw,time,2*n+2')
    N, T, _ = latent.shape; n, m = candidate.n, candidate.m
    if r.shape != (N,m) or x.shape != (N,m) or not np.isfinite(r).all() or not np.isfinite(x).all():
        raise ValueError('finite R/X must be draw,m')
    if np.any(r < 0) or np.any(x < 0) or not np.isfinite(latent).all() or base_kw <= 0:
        raise ValueError('nonnegative R/X and finite inputs required')
    root = np.broadcast_to(np.asarray(root_v, float), (T,))
    if not np.isfinite(root).all() or np.any(root <= 0): raise ValueError('invalid root')
    if power is not None:
        power = np.asarray(power, float)
        if power.shape != (T,n) or not np.isfinite(power).all(): raise ValueError('invalid power')
        latent[:,:,:n] += power[None]/base_kw
    loads = np.zeros((N,T,m), complex)
    loads[:,:,0] = latent[:,:,-2] + 1j*latent[:,:,-1]
    loads[:,:,list(candidate.terminal_edges)] += latent[:,:,:n] + 1j*latent[:,:,n:2*n]
    z = r+1j*x
    zbus = np.einsum('ie,de,je->dij', candidate.B, z, candidate.B)
    voltage = np.broadcast_to(root[None,:,None], (N,T,m)).astype(complex).copy()
    active = np.ones(N, bool); converged = np.zeros(N, bool); failed = np.zeros(N, bool)
    iteration_count = np.zeros(N, int)
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        for iteration in range(1, max_iter+1):
            if not active.any(): break
            bad = (~np.isfinite(voltage).all((1,2))) | (np.min(abs(voltage), axis=(1,2)) < .1)
            failed |= active & bad; active &= ~bad
            current = np.zeros_like(loads)
            current[active] = np.conj(loads[active]/voltage[active])
            new = root[None,:,None] - np.einsum('dtj,dij->dti', current, zbus)
            bad = ~np.isfinite(new).all((1,2)); failed |= active & bad; active &= ~bad
            change = np.max(abs(new-voltage), axis=(1,2))
            done = active & (change <= tol)
            voltage[active] = new[active]; iteration_count[active] = iteration
            converged |= done; active &= ~done
        failed |= active
        currents = np.conj(loads/voltage)
        branch = np.einsum('dti,ie->dte', currents, candidate.B)
        master = root[None,:]*np.conj(currents.sum(2))
        balance = master-loads.sum(2)-(abs(branch)**2*z[:,None,:]).sum(2)
        equation = voltage-root[None,:,None]+np.einsum('dtj,dij->dti',currents,zbus)
    eq = np.max(abs(equation),axis=(1,2)); pb = np.max(abs(balance),axis=1)
    failed |= ~np.isfinite(eq) | ~np.isfinite(pb) | (eq > 2e-10) | (pb > 2e-10)
    physical = dict(p=latent[:,:,:n], q=latent[:,:,n:2*n],
        v=abs(voltage[:,:,list(candidate.terminal_edges)]), master_p=master.real, master_q=master.imag)
    blocks = []
    for name in CHANNELS:
        value = physical[name]
        shape = (T,n) if name in ('p','q','v') else (T,)
        sigma = np.broadcast_to(np.asarray(getattr(meter,name+'_sigma'),float),shape)
        if not np.isfinite(sigma).all() or np.any(sigma <= 0): raise ValueError('invalid meter sigma')
        if name in ('p','q','v'):
            value = value+np.broadcast_to(np.asarray(getattr(meter,'bias_'+name),float),shape)[None]
        blocks.append((value/sigma[None]).reshape(N,-1))
    means = np.concatenate(blocks, axis=1)
    means[failed] = np.nan
    source_p_bad = np.any((latent[:,:,-2]<opts['source_p_bounds'][0]) |
                         (latent[:,:,-2]>opts['source_p_bounds'][1]),axis=1)
    source_q_bad = np.any((latent[:,:,-1]<opts['source_q_bounds'][0]) |
                         (latent[:,:,-1]>opts['source_q_bounds'][1]),axis=1)
    voltage_mask=(abs(voltage)<opts['voltage_bounds'][0]) | (abs(voltage)>opts['voltage_bounds'][1])
    terminal_bad=np.any(voltage_mask[:,:,list(candidate.terminal_edges)],axis=(1,2))
    internal=[j for j in range(m) if j not in candidate.terminal_edges]
    internal_bad=np.any(voltage_mask[:,:,internal],axis=(1,2)) if internal else np.zeros(N,bool)
    return dict(mean=means, physical=physical, status='ac_failure' if failed.any() else 'ok',
        diagnostics=dict(draws=N,time_count=T,AC_states=N*T,EV_added_once=True,
            source_interface=1,source_p_violations=source_p_bad,source_q_violations=source_q_bad,
            voltage_violations=terminal_bad,terminal_voltage_violations=terminal_bad,
            internal_voltage_violations=internal_bad,all_node_voltage_violations=terminal_bad|internal_bad,
            ac_failure=failed,iterations=iteration_count,
            equation_residual=eq,power_balance_residual=pb,discarded_draws=0),
        seconds=perf_counter()-began)


def sample_posterior(update, *, draws=256, replicates=4, seed=0):
    """GHK line pool plus conditional-state Gaussian, with systematic resampling.

    Replicate error is conditional on the reconstructed GHK pool. Its original
    mass/moment integration diagnostics are reported separately. No truncated
    moment Gaussian is substituted for the bounded posterior distribution.
    """
    if int(draws)!=draws or draws<2 or int(replicates)!=replicates or replicates<2:
        raise ValueError('draws and replicates must be integers >=2')
    began=perf_counter(); untr=update['TV_untruncated_future']
    m=np.asarray(untr['mean'],float); C=np.asarray(untr['cov'],float)
    prior=update['line_prior']; d=len(prior['mean'])
    settings=dict(update['TV_integration_options']); settings['return_samples']=True
    pool=trunc.bounded_line_moments(m[:d],C[:d,:d],prior['bounds'],**settings)
    result=dict(status=pool['status'],line_dimension=d,posterior_fingerprint=posterior_fingerprint(update),diagnostics=dict(
        integration=pool['diagnostics'],line_pool_points=len(pool['samples']) if pool['samples'] is not None else 0,
        conditional_replicates=True,seed=int(seed),draws=int(draws),replicates=int(replicates)),
        samples=None,seconds=perf_counter()-began)
    if pool['status']!='ok': return result
    gain=C[d:,:d]@old._inverse_psd(C[:d,:d])
    conditional=C[d:,d:]-gain@C[:d,d:]
    L,factor=_factor(conditional)
    cumulative=np.cumsum(pool['weights']); cumulative[-1]=1.
    reps=[]
    for rep in range(int(replicates)):
        rng=np.random.default_rng(np.random.SeedSequence([int(seed),rep,173]))
        positions=(np.arange(int(draws))+rng.random())/draws
        ids=np.searchsorted(cumulative,positions,side='right')
        # Random permutation avoids artificial state/line QMC ordering.
        eta=pool['samples'][ids][rng.permutation(int(draws))]
        sobol=qmc.Sobol(d=len(m)-d,scramble=True,seed=int(rng.integers(0,2**32-1)))
        # random_base2 uses exactly all points when draws is a power of two.
        uniforms=sobol.random_base2(int(np.ceil(np.log2(draws))))[:int(draws)]
        normals=ndtri(np.clip(uniforms,np.finfo(float).eps,1-np.finfo(float).eps))
        state=m[d:]+(eta-m[:d])@gain.T+normals@L.T
        reps.append(np.c_[eta,state])
    result.update(samples=np.asarray(reps),status='ok',seconds=perf_counter()-began)
    result['diagnostics']['conditional_covariance_factor']=factor
    return result


def control_variate_moments(samples, outputs, center, mean0, jacobian, covariance):
    """Numerical control-variate moment estimator; affine f is exact to roundoff.

    The draws use randomized Sobol rules, so no iid-unbiased claim is made.
    """
    samples, outputs=np.asarray(samples),np.asarray(outputs)
    linear=mean0+(samples-center)@jacobian.T
    remainder=outputs-linear
    g=linear-linear.mean(0); rr=remainder-remainder.mean(0)
    denominator=len(samples)-1
    residual_cov=rr.T@rr/denominator
    cross=g.T@rr/denominator
    baseline=jacobian@covariance@jacobian.T
    cov=baseline+residual_cov+cross+cross.T
    return mean0+remainder.mean(0), .5*(cov+cov.T),dict(
        remainder_RMS=float(np.sqrt(np.mean(remainder**2))),
        mean_remainder=remainder.mean(0),control_covariance=baseline)


def forecast_moments(context,context_record,power,*,seed=0,draws=256,replicates=4,
                     posterior_cache=None,reference=None):
    """Return G00/G10/G01/G11 banks, same historical weights and fixed plan.

    All diagnostics include failures. A failed candidate prevents corrected
    banks from being returned; no candidate or posterior draw is dropped.
    """
    began=perf_counter(); cache={} if posterior_cache is None else posterior_cache
    reference=old.forecast_bank(context,power) if reference is None else reference
    base=reference['bank']; rows=[]; mus=[]; covs=[]; all_ok=True; total_states=0
    for index,row in enumerate(context['rows']):
        name=row['candidate'].name; update=context_record['candidate_updates'][name]
        if name in cache and cache[name].get('posterior_fingerprint')!=posterior_fingerprint(update):
            raise ValueError('posterior cache fingerprint mismatch for '+name)
        if name not in cache or cache[name]['diagnostics'].get('draws')!=draws or cache[name]['diagnostics'].get('replicates')!=replicates:
            cache[name]=sample_posterior(update,draws=draws,replicates=replicates,
                seed=int(np.random.SeedSequence([int(seed),index]).generate_state(1)[0]))
        sampled=cache[name]; diag=dict(candidate=name,sampling=sampled['diagnostics'],status=sampled['status'])
        rows.append(diag)
        if sampled['status']!='ok': all_ok=False; continue
        W=np.asarray(sampled['samples']); R,N,D=W.shape; d=row['line_dimension']
        F=np.asarray(reference['joint_jacobian'][index]); mu0=base['means'][0,index]
        center=np.asarray(row['joint_mean']); C=np.asarray(row['joint_covariance'])
        eta=W[:,:,:d].reshape(-1,d); r,x=trunc.eta_to_rx(eta)
        latent=W[:,:,d:].reshape(R*N,context['horizon'],2*context['n']+2)
        physical=batched_ac(row['candidate'],r,x,latent,context['root_v'],context['future_meter'],
            power=power,base_kw=context['base_kw'],options=context['options'])
        total_states+=physical['diagnostics']['AC_states']; pd=physical['diagnostics']
        diag['physical']={k:v for k,v in pd.items() if k not in ('iterations',)}
        diag['physical']['max_iterations']=int(np.max(pd['iterations']))
        diag['violation_fractions']={k:float(np.mean(pd[k])) for k in (
            'source_p_violations','source_q_violations','voltage_violations',
            'terminal_voltage_violations','internal_voltage_violations','all_node_voltage_violations','ac_failure')}
        diag['discarded_draws']=0
        if physical['status']!='ok': diag['status']='ac_failure'; all_ok=False; continue
        outputs=physical['mean'].reshape(R,N,-1); means=[]; cc=[]; rem=[]
        for rep in range(R):
            mu,cov,diagnostic=control_variate_moments(W[rep],outputs[rep],center,mu0,F,C)
            means.append(mu);cc.append(cov);rem.append(diagnostic['remainder_RMS'])
        means=np.asarray(means); cc=np.asarray(cc); mu=means.mean(0)
        cv_cov=.5*(cc.mean(0)+cc.mean(0).T)
        eigen,vectors=np.linalg.eigh(cv_cov);scale=max(float(abs(eigen).max()),1e-30)
        fallback=bool(eigen.min() < -1e-10*scale)
        repair=dict(CV_minimum_eigenvalue=float(eigen.min()),relative_roundoff_tolerance=1e-10,
            ordinary_AC_sample_covariance_fallback=fallback)
        if fallback:
            # A finite-sample CV correction need not be PSD. The ordinary
            # covariance uses ALL AC outputs, preserves their cross channels,
            # and its independent-replicate error is reported below.
            cc=np.asarray([np.cov(y,rowvar=False,ddof=1) for y in outputs])
            cv_cov=.5*(cc.mean(0)+cc.mean(0).T)
            eigen,vectors=np.linalg.eigh(cv_cov);scale=max(float(abs(eigen).max()),1e-30)
        if eigen.min() < -1e-10*scale:
            diag.update(status='covariance_failure',covariance_roundoff=repair);all_ok=False;continue
        repair['repaired_negative_trace']=float(-np.minimum(eigen,0.).sum())
        if np.any(eigen<0):cv_cov=(vectors*np.maximum(eigen,0.))@vectors.T
        covariance=cv_cov+np.eye(len(mu))
        # Domain violations remain visible and invalidate the corrected result.
        domain_bad=any(np.any(pd[k]) for k in ('source_p_violations','source_q_violations','voltage_violations'))
        if domain_bad: diag['status']='physical_domain_failure';all_ok=False
        mean_se=means.std(0,ddof=1)/np.sqrt(R)
        cov_se=cc.std(0,ddof=1)/np.sqrt(R)
        diag.update(mean=mu,covariance=covariance,mean_correction=mu-mu0,
            covariance_correction=covariance-base['covariances'][0,index],
            mean_MCSE=mean_se,covariance_MCSE_fro=float(np.linalg.norm(cov_se)),
            mean_MCSE_whitened_max=float(np.max(mean_se)),
            covariance_MCSE_relative_fro=float(np.linalg.norm(cov_se)/max(np.linalg.norm(covariance-np.eye(len(mu))),1e-30)),
            remainder_RMS_by_replicate=rem,covariance_roundoff=repair)
        mus.append(mu);covs.append(covariance)
    banks={'G00':base}
    if len(mus)==len(context['rows']) and all_ok:
        means=np.asarray(mus)[None]; covariance=np.asarray(covs)[None]
        for name,mm,cc in (('G10',means,base['covariances']),('G01',base['means'],covariance),('G11',means,covariance)):
            banks[name]={**base,'means':mm.copy(),'covariances':cc.copy()}
    return dict(banks=banks,status='ok' if all_ok else 'propagation_failure',rows=rows,
        diagnostics=dict(draws=int(draws),replicates=int(replicates),AC_states=total_states,
            historical_posterior_unchanged=True,full_AC_future_moments=True,
            output_density_Gaussian_shape_approximation=True,discarded_draws=0,
            source_interface=1,EV_added_once=True,conditional_MC_error_excludes_GHK_pool_error=True),
        seconds=perf_counter()-began)
