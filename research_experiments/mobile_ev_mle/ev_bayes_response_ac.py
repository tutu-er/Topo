"""Joint local line/PQ/source uncertainty, with one historical AC center.

All raw training meter channels enter once.  Fitted sources are AC expansion
coordinates only, never pseudo-observations or process-prior means.  Process
hyperparameters are empirical training-only estimates; the line prior is public
and proper.  Linearized likelihoods, bounded line integration, and a future
Gaussian moment approximation are separate approximations with saved evidence.
"""
from __future__ import annotations

from hashlib import sha256
from time import perf_counter
from copy import deepcopy
import numpy as np
from scipy.linalg import cho_factor, cho_solve, block_diag
from scipy.special import logsumexp

from .symmetric_ac_local import mean_and_jacobian, whiten_observations, CHANNELS
from .ev_mobile_v4_cases import harmonic_features
from .ev_rx_fit import DEFAULT_RATIO_BOUNDS
from . import ev_bayes_response_truncation as truncation


DEFAULT_OPTIONS = dict(r_bounds=(1e-6, .1), x_bounds=(1e-6, .1),
    ratio_bounds=DEFAULT_RATIO_BOUNDS, voltage_bounds=(.85, 1.15),
    source_p_bounds=(-.5, .5), source_q_bounds=(-.5, .5),
    coefficient_sd=.5, ar_limit=.9, innovation_floor=.01,
    truncation=dict(seed=0, power=10, max_power=16, replicates=8))


class BoundedIntegrationError(RuntimeError):
    """Inference failure with structured, reproducible numerical evidence."""
    def __init__(self, result, mean, covariance, bounds, options):
        diagnostics=result.get('diagnostics',{})
        super().__init__('bounded line integration failed: status='+str(result.get('status'))+
            ', points='+str(diagnostics.get('points'))+', relative_SE='+str(diagnostics.get('relative_se'))+
            ', ESS='+str(diagnostics.get('ess')))
        self.evidence=dict(integration=result, line_mean=np.asarray(mean).copy(),
            line_covariance=np.asarray(covariance).copy(), bounds=deepcopy(bounds), options=deepcopy(options))


def _finite(value, name):
    a = np.asarray(value, float)
    if not np.isfinite(a).all(): raise ValueError(name + ' must be finite')
    return a


def _options(options):
    result = {**DEFAULT_OPTIONS, **(options or {})}
    result['truncation'] = {**DEFAULT_OPTIONS['truncation'], **(options or {}).get('truncation', {})}
    if 'seed' in (options or {}) and 'seed' not in (options or {}).get('truncation', {}):
        result['truncation']['seed'] = int(options['seed'])
    if not 0 < float(result['ar_limit']) < 1: raise ValueError('ar_limit must lie in (0,1)')
    if float(result['innovation_floor']) <= 0 or float(result['coefficient_sd']) <= 0:
        raise ValueError('proper positive process floors and coefficient_sd required')
    return result


def symmetric_psd(value, *, name='covariance'):
    """Repair negative roundoff only; keep deterministic/tiny positive directions."""
    matrix = _finite(value, name)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]: raise ValueError(name + ' must be square')
    matrix = .5 * (matrix + matrix.T)
    eigen, vectors = np.linalg.eigh(matrix)
    scale = max(float(np.max(abs(eigen), initial=0)), 1e-30)
    if eigen.min(initial=0) < -1e-8 * scale:
        raise ValueError(name + ' is materially indefinite')
    negative = eigen < 0
    if negative.any(): matrix = (vectors * np.maximum(eigen, 0)) @ vectors.T
    return .5 * (matrix + matrix.T), dict(minimum_eigenvalue=float(eigen.min(initial=0)),
        negative_roundoff_count=int(negative.sum()),
        repaired_trace=float(-eigen[negative].sum()), positive_variances_not_floored=True)


def _inverse_psd(value):
    matrix = .5 * (np.asarray(value) + np.asarray(value).T)
    try: return cho_solve(cho_factor(matrix, lower=True), np.eye(len(matrix)))
    except np.linalg.LinAlgError:
        eigen, vectors = np.linalg.eigh(matrix)
        threshold = 1e-12 * max(float(eigen.max(initial=0)), 1e-30)
        return (vectors * np.divide(1., eigen, out=np.zeros_like(eigen), where=eigen > threshold)) @ vectors.T


def condition_cross_gaussian(target_mean, target_covariance, observation_mean,
                             observation_covariance, cross_covariance, observed):
    """Condition a joint Gaussian; cross=Cov(target,observation).

    This helper is used before physical truncation.  Moment-matched truncated
    moments are never substituted as the prior of the next untruncated update.
    """
    m, C = _finite(target_mean, 'target_mean'), _finite(target_covariance, 'target_covariance')
    mu, S, cross, y = map(np.asarray, (observation_mean, observation_covariance, cross_covariance, observed))
    if m.ndim != 1 or C.shape != (len(m), len(m)) or mu.shape != y.shape or y.ndim != 1:
        raise ValueError('Gaussian target/observation dimensions disagree')
    if S.shape != (len(y), len(y)) or cross.shape != (len(m), len(y)):
        raise ValueError('Gaussian covariance dimensions disagree')
    if not all(np.isfinite(x).all() for x in (mu, S, cross, y)): raise ValueError('finite Gaussian data required')
    factor = cho_factor(.5 * (S + S.T), lower=True)
    residual = y - mu
    alpha = cho_solve(factor, residual)
    posterior_mean = m + cross @ alpha
    covariance, correction = symmetric_psd(C - cross @ cho_solve(factor, cross.T), name='posterior covariance')
    logdet = float(2 * np.log(np.diag(factor[0])).sum())
    log_density = float(-.5 * (residual @ alpha + logdet + len(y) * np.log(2*np.pi)))
    return dict(mean=posterior_mean, cov=covariance, log_density=log_density,
        observation_mean=mu.copy(), observation_covariance=S.copy(), residual=residual,
        logdet=logdet, covariance_roundoff=correction)


def gaussian_linear_update(mean, covariance, observed, offset, matrix):
    """Exact local Gaussian update y=offset+matrix*w+N(0,I)."""
    mean, covariance, matrix = np.asarray(mean), np.asarray(covariance), np.asarray(matrix)
    cross = covariance @ matrix.T
    return condition_cross_gaussian(mean, covariance, np.asarray(offset) + matrix @ mean,
        np.eye(matrix.shape[0]) + matrix @ cross, cross, observed)


def restore_joint_moments(mean, covariance, line_dimension, line_mean, line_covariance):
    """Exact moments of a joint Gaussian truncated on line coordinates only."""
    mean, C = np.asarray(mean, float), np.asarray(covariance, float)
    d = int(line_dimension); lm, LC = np.asarray(line_mean), np.asarray(line_covariance)
    if not 0 < d <= len(mean) or lm.shape != (d,) or LC.shape != (d,d):
        raise ValueError('line moment dimensions disagree')
    A = C[d:, :d] @ _inverse_psd(C[:d, :d])
    conditional = C[d:, d:] - A @ C[:d, d:]
    result_mean = np.r_[lm, mean[d:] + A @ (lm - mean[:d])]
    result_covariance = np.block([[LC, LC @ A.T], [A @ LC, conditional + A @ LC @ A.T]])
    result_covariance, correction = symmetric_psd(result_covariance, name='bounded joint covariance')
    return dict(mean=result_mean, cov=result_covariance, state_compensation=A,
        conditional_state_covariance=.5*(conditional+conditional.T), covariance_roundoff=correction)


def _truncated(mean, covariance, d, prior, options, seed_offset):
    opts = dict(options['truncation']); opts['seed'] = int(opts['seed']) + int(seed_offset)
    line = truncation.bounded_line_moments(mean[:d], covariance[:d,:d], prior['bounds'], **opts)
    if line.get('status') != 'ok' or not np.isfinite(line['log_mass']):
        raise BoundedIntegrationError(line,mean[:d],covariance[:d,:d],prior['bounds'],opts)
    joint = restore_joint_moments(mean, covariance, d, line['mean'], line['cov'])
    # Raw samples are reproducible from the Gaussian, domain, settings and seed.
    summary = {key:value for key,value in line.items() if key not in ('samples','weights','log_weights')}
    return dict(**joint, log_mass=float(line['log_mass']), line_integration=summary, integration_options=opts)


def _sigma(meter, name, shape):
    value = np.broadcast_to(_finite(getattr(meter, name+'_sigma'), name+'_sigma'), shape)
    if np.any(value <= 0): raise ValueError(name+'_sigma must be positive')
    return value


def proxy_training_process(training, meter):
    """Raw-data proxy and its *joint* meter-noise covariance, for hyperparameters.

    Source proxy=master-sum(terminals) includes line losses.  It is not source
    telemetry and is never inserted as an extra likelihood observation.
    """
    H, n = training.t, training.n; w = 2*n+2
    p = training.p - np.broadcast_to(np.asarray(meter.bias_p), training.p.shape)
    q = training.q - np.broadcast_to(np.asarray(meter.bias_q), training.q.shape)
    if training.master_p is None or training.master_q is None: raise ValueError('master PQ required')
    raw = np.c_[p, q, training.master_p, training.master_q]
    transform = np.eye(w); transform[-2,:n] = -1; transform[-1,n:2*n] = -1
    values = raw @ transform.T
    standard = np.c_[_sigma(meter,'p',(H,n)), _sigma(meter,'q',(H,n)),
                     _sigma(meter,'master_p',(H,)), _sigma(meter,'master_q',(H,))]
    noise_covariance = np.einsum('ij,tj,kj->tik', transform, standard**2, transform)
    return values, noise_covariance, dict(noise_transform=transform,
        raw_channel_order='P_all,Q_all,masterP,masterQ',
        proxy_channel_order='P_all,Q_all,masterP-sumP,masterQ-sumQ',
        proxy_used_as_measurement=False, fitted_source_used_as_measurement=False,
        proxy_source_contains_losses=True, negative_PQ_source_meter_crosscovariance_retained=True)


def estimate_process_hyperparameters(training, meter, *, start_time=0, options=None):
    options = _options(options)
    values, noise, proxy_info = proxy_training_process(training,meter)
    H, width = values.shape
    if H < 12: raise ValueError('at least 12 training times required')
    features = harmonic_features(np.arange(start_time,start_time+H))
    beta, _, rank, _ = np.linalg.lstsq(features, values, rcond=None)
    if rank != features.shape[1]: raise ValueError('rank-deficient process features')
    projection = np.eye(H) - features @ np.linalg.pinv(features)
    residual = projection @ values
    # Projected meter errors are correlated in time, even if raw errors are
    # independent. Correct the plug-in lag estimate for that known correlation.
    meter_lag0 = np.einsum('it,tjk,it->jk', projection[:-1], noise, projection[:-1])/(H-1)
    meter_lag1 = np.einsum('it,tjk,it->jk', projection[1:], noise, projection[:-1])/(H-1)
    lag_variance = np.trace(residual[:-1].T @ residual[:-1]/(H-1) - meter_lag0)
    denominator = max(float(lag_variance), .01*float(np.trace(noise.mean(0))), 1e-30)
    phi_raw = float(np.trace(residual[1:].T @ residual[:-1]/(H-1)-meter_lag1)/denominator)
    phi = float(np.clip(phi_raw, -options['ar_limit'], options['ar_limit']))
    innovation = residual[1:] - phi*residual[:-1]
    D = np.eye(H)[1:] - phi*np.eye(H)[:-1]
    W = D @ projection
    # For fixed phi and a proper stationary AR process, E[innovation'innovation]
    # after removing fitted harmonics is c_process*Q+c_meter*R. Time-varying
    # meter covariance is subtracted exactly instead of using (1+phi**2)*R.
    subtracted = np.einsum('it,tjk,it->jk', W, noise, W)/(H-1)
    times = np.arange(H)
    unit_AR = phi**abs(times[:,None]-times[None,:])/(1-phi**2)
    c_process = float(np.einsum('ij,ij->', W@unit_AR, W)/(H-1))
    c_meter = float(np.sum(W**2)/(H-1))
    if not np.isfinite(c_process) or c_process <= 0:
        raise ValueError('positive process projection factor required')
    raw_Q = (innovation.T@innovation/(H-1) - subtracted)/c_process
    average_R = noise.mean(0); L = np.linalg.cholesky(average_R)
    white = np.linalg.solve(L, np.linalg.solve(L, raw_Q).T).T
    white = .5*(white+white.T); eigen,vectors=np.linalg.eigh(white)
    adjusted = np.maximum(eigen, options['innovation_floor'])
    Q = L @ ((vectors*adjusted)@vectors.T) @ L.T
    return dict(phi=phi, raw_phi=phi_raw, innovation_covariance=.5*(Q+Q.T),
        coefficient_sd=float(options['coefficient_sd']), training_count=H, start_time=int(start_time),
        channel_count=width, proxy_meter_covariance=average_R, subtracted_innovation_meter_covariance=subtracted,
        harmonic_projection_rank=int(rank), innovation_process_projection_factor=c_process,
        innovation_meter_projection_factor=c_meter, projected_meter_lag0_covariance=meter_lag0,
        projected_meter_lag1_covariance=meter_lag1,
        raw_innovation_covariance=raw_Q, whitened_innovation_eigenvalues=eigen,
        innovation_floor=float(options['innovation_floor']),
        innovation_eigenvalues_floored=int(np.sum(eigen<options['innovation_floor'])),
        innovation_covariance_adjustment_norm=float(np.linalg.norm(Q-raw_Q)),
        empirical_hyperparameters_training_only=True, hyperparameter_uncertainty_integrated=False,
        coefficient_prior_mean=0., proxy_info=proxy_info)


def process_kernel(hyperparameters, count, *, start_time=0):
    """Unconditioned proper joint process prior; raw T is conditioned later once."""
    Q=np.asarray(hyperparameters['innovation_covariance'],float); w=len(Q)
    phi=float(hyperparameters['phi']); times=np.arange(start_time,start_time+int(count))
    features=harmonic_features(times)
    stationary=phi**abs(times[:,None]-times[None,:])/(1-phi**2)
    covariance=np.kron(stationary,Q)+float(hyperparameters['coefficient_sd'])**2*np.kron(features@features.T,np.eye(w))
    return np.zeros(int(count)*w), .5*(covariance+covariance.T)


def _eta(r,x):
    r,x=np.asarray(r,float),np.asarray(x,float)
    if r.shape!=x.shape or np.any(r<=0) or np.any(x<=0): raise ValueError('positive paired R/X required')
    return np.c_[np.log(r),np.log(x/r)].ravel()


def _rx(eta):
    eta=np.asarray(eta,float).reshape(-1,2)
    return np.exp(eta[:,0]),np.exp(eta.sum(1))


def _eta_jacobian(r,x):
    """Physical derivative columns are R_all,X_all; eta columns are interleaved."""
    r,x=np.asarray(r),np.asarray(x); m=len(r); result=np.zeros((2*m,2*m))
    for e in range(m): result[e,2*e]=r[e]; result[m+e,2*e]=result[m+e,2*e+1]=x[e]
    return result


def _line_id(r,x): return sha256(np.r_[r,x].astype('<f8').tobytes()).hexdigest()


def _physical(candidate,r,x,latent,root,meter,options):
    n=candidate.n
    response=mean_and_jacobian(candidate,r,x,latent[:,:n],latent[:,n:2*n],latent[:,-2],latent[:,-1],root,meter,
        channels=CHANNELS,with_jacobian=True)
    voltage=response['physical']['v']; lo,hi=options['voltage_bounds']
    if not lo<=np.min(voltage)<=np.max(voltage)<=hi: raise ValueError('AC expansion mean violates voltage domain')
    for column,bounds in ((-2,options['source_p_bounds']),(-1,options['source_q_bounds'])):
        if np.any(latent[:,column]<bounds[0]) or np.any(latent[:,column]>bounds[1]):
            raise ValueError('AC expansion source mean violates the fixed interface domain')
    response['line_jacobian_eta']=response['line_jacobian']@_eta_jacobian(r,x)
    return response


def _reduced(mean,cov,d,width,V,F):
    indices=np.r_[np.arange(d),d+np.arange(V*width,(V+F)*width)]
    return dict(mean=mean[indices].copy(),cov=cov[np.ix_(indices,indices)].copy())


def build_context(training,evidence,train_meter,evidence_meter,candidates,fits,future_meter,
                  horizon=6,base_kw=20.,start_time=0,options=None):
    """Training raw meters once, T then V Gaussian updates, line truncation once.

    Only observed inputs enter; no case, generating parameter, truth or future
    noise argument exists.  Every future action shares its historical center.
    """
    began=perf_counter(); options=_options(options); candidates=tuple(candidates)
    H,V,F,n=training.t,evidence.t,int(horizon),training.n; width=2*n+2
    if evidence.n!=n or F<1 or not np.isfinite(base_kw) or base_kw<=0: raise ValueError('invalid dimensions/base')
    if set(fits)!={c.name for c in candidates} or not candidates: raise ValueError('complete named fit bank required')
    hyper=estimate_process_hyperparameters(training,train_meter,start_time=start_time,options=options)
    _,K=process_kernel(hyper,H+V+F,start_time=start_time)
    Ktt=K[:H*width,:H*width]; Kpt=K[H*width:,:H*width]; Kpp=K[H*width:,H*width:]
    observedT=whiten_observations(training,train_meter); observedV=whiten_observations(evidence,evidence_meter)
    p=training.p-np.broadcast_to(train_meter.bias_p,training.p.shape)
    q=training.q-np.broadcast_to(train_meter.bias_q,training.q.shape)
    rows=[]; records={}; log_all=[]; log_v=[]
    for index,candidate in enumerate(candidates):
        fit=fits[candidate.name]; r,x=np.asarray(fit.r,float),np.asarray(fit.x,float)
        if not fit.success or len(r)!=candidate.m: raise ValueError('complete physical training center required')
        prior=truncation.line_prior(candidate.m,options['r_bounds'],options['x_bounds'],options['ratio_bounds'])
        eta0=_eta(r,x); d=len(eta0)
        latentT=np.c_[p,q,np.asarray(fit.source_p),np.asarray(fit.source_q)]
        if latentT.shape!=(H,width): raise ValueError('training source center dimensions disagree')
        train=_physical(candidate,r,x,latentT,training.root_v,train_meter,options)
        JL,Jz=train['line_jacobian_eta'],train['state_jacobian']
        mean0=np.r_[prior['mean'],np.zeros((V+F)*width)]; covariance0=block_diag(prior['cov'],Kpp)
        prior_observation_mean=train['mean']+JL@(prior['mean']-eta0)-Jz@latentT.ravel()
        S=np.eye(len(observedT))+JL@prior['cov']@JL.T+Jz@Ktt@Jz.T
        cross=np.vstack((prior['cov']@JL.T,Kpt@Jz.T))
        T=condition_cross_gaussian(mean0,covariance0,prior_observation_mean,S,cross,observedT)
        boundedT=_truncated(T['mean'],T['cov'],d,prior,options,1000+2*index)
        logZ0=prior['log_mass']
        if not np.isfinite(logZ0): raise ValueError('proper line prior has invalid normalization')
        lineT=boundedT['mean'][:d]; rt,xt=_rx(lineT)
        latentV=boundedT['mean'][d:d+V*width].reshape(V,width)
        ev=_physical(candidate,rt,xt,latentV,evidence.root_v,evidence_meter,options)
        JV=np.zeros((len(observedV),d+(V+F)*width)); JV[:,:d]=ev['line_jacobian_eta']; JV[:,d:d+V*width]=ev['state_jacobian']
        offsetV=ev['mean']-JV@boundedT['mean']
        TV=gaussian_linear_update(T['mean'],T['cov'],observedV,offsetV,JV)
        boundedTV=_truncated(TV['mean'],TV['cov'],d,prior,options,1001+2*index)
        logH=float(T['log_density']+boundedT['log_mass']-logZ0)
        logE=float(TV['log_density']+boundedTV['log_mass']-boundedT['log_mass'])
        final=_reduced(boundedTV['mean'],boundedTV['cov'],d,width,V,F)
        stageT=_reduced(boundedT['mean'],boundedT['cov'],d,width,V,F)
        untrT=_reduced(T['mean'],T['cov'],d,width,V,F)
        untrTV=_reduced(TV['mean'],TV['cov'],d,width,V,F)
        rp,xp=_rx(final['mean'][:d]); latentF=final['mean'][d:].reshape(F,width)
        row=dict(candidate=candidate,r=rp,x=xp,line_sha256=_line_id(rp,xp),line_dimension=d,
            joint_mean=final['mean'],joint_covariance=final['cov'],latent_mean=latentF,
            latent_cov=final['cov'][d:,d:],T_joint_mean=stageT['mean'],T_joint_covariance=stageT['cov'],
            trained_center_eta=eta0.copy(),trained_center_r=r.copy(),trained_center_x=x.copy())
        rows.append(row); log_all.append(logH+logE); log_v.append(logE)
        records[candidate.name]=dict(line_prior=prior,training_center=dict(eta=eta0,r=r,x=x,latent=latentT),
            training_Gaussian_log_predictive=T['log_density'],training_log_predictive=logH,
            evidence_Gaussian_log_predictive=TV['log_density'],evidence_log_predictive=logE,
            logZprior=float(logZ0),logZT=boundedT['log_mass'],logZTV=boundedTV['log_mass'],
            T_line_integration=boundedT['line_integration'],TV_line_integration=boundedTV['line_integration'],
            T_integration_options=boundedT['integration_options'],TV_integration_options=boundedTV['integration_options'],
            T_posterior_future=stageT,TV_posterior_future=final,
            T_untruncated_future=untrT,TV_untruncated_future=untrTV,
            future_latent_mean=latentF,future_latent_covariance=final['cov'][d:,d:],
            evidence_linearization_mean=ev['mean'],evidence_linearization_joint_mean=boundedT['mean'],
            evidence_observed_whitened=observedV,evidence_predicted_mean=TV['observation_mean'],
            evidence_covariance=TV['observation_covariance'],evidence_affine_jacobian=JV,evidence_affine_offset=offsetV,
            T_untruncated_EF_mean=T['mean'],T_untruncated_EF_covariance=T['cov'],
            training_covariance_roundoff=T['covariance_roundoff'],evidence_covariance_roundoff=TV['covariance_roundoff'],
            training_JL_sha256=sha256(JL.astype('<f8').tobytes()).hexdigest(),
            training_Jz_sha256=sha256(Jz.astype('<f8').tobytes()).hexdigest(),
            training_observation_mean=prior_observation_mean,training_logdet=T['logdet'],
            training_latent_sources_are_expansion_coordinates_only=True)
    log_prior=np.asarray(log_all)-logsumexp(log_all); log_evidence=np.asarray(log_v)-logsumexp(log_v)
    context=dict(rows=rows,prior=np.exp(log_prior),log_prior=log_prior,log_prior_all_history=log_prior.copy(),
        log_prior_evidence_only=log_evidence,future_meter=future_meter,horizon=F,n=n,base_kw=float(base_kw),
        root_v=np.ones(F),source_interface=1,training_count=H,evidence_count=V,options=options,
        model_scope='bounded joint line/PQ/source posterior; local Gaussian future moment approximation')
    record=dict(training_count=H,evidence_count=V,horizon=F,base_kw=float(base_kw),start_time=int(start_time),
        candidate_names=[c.name for c in candidates],prior=np.exp(log_prior),log_prior=log_prior,
        log_prior_evidence_only=log_evidence,prior_display_underflow_count=int(np.sum(np.exp(log_prior)==0)),
        process_hyperparameters=hyper,candidate_updates=records,options=options,
        source_interface=1,shared_lines_all_times=True,one_historical_center_per_candidate=True,
        training_raw_likelihood_count=1,fitted_source_pseudo_likelihood_count=0,
        latent_future_PQ_optimized=False,P_Q_measurement_error_in_joint_density=True,
        posterior_probability_scope=context['model_scope'],
        empirical_hyperparameters_training_only=True,nonlinear_AC_likelihood_linearized=True,
        future_latent_distribution_after_truncation_nonGaussian=True,future_Gaussian_is_moment_approximation=True,
        seconds=float(perf_counter()-began))
    return context,record


def _mode_covariance(row,mode):
    C=np.asarray(row['joint_covariance']).copy(); d=row['line_dimension']
    if mode=='full': return C
    if mode=='no_cross': C[:d,d:]=0; C[d:,:d]=0; return C
    if mode=='line_known':
        state=C[d:,d:]-C[d:,:d]@_inverse_psd(C[:d,:d])@C[:d,d:]
        C[:]=0; C[d:,d:]=state
        return symmetric_psd(C,name='line-known conditional covariance')[0]
    raise ValueError('covariance_mode must be full, no_cross or line_known')


def forecast_bank(context,power,*,with_jacobian=False,covariance_mode='full',weights_mode='all_history'):
    began=perf_counter(); T,n=context['horizon'],context['n']; power=_finite(power,'power')
    if power.shape!=(T,n): raise ValueError('power must have shape [T,n]')
    if weights_mode not in ('all_history','evidence_only'): raise ValueError('unknown weights_mode')
    means=[]; covariances=[]; jacobians=[]; joint_jacobians=[]; physical=[]
    for row in context['rows']:
        if _line_id(row['r'],row['x'])!=row['line_sha256']: raise ValueError('historical center changed')
        latent=row['latent_mean'].copy(); latent[:,:n]+=power/context['base_kw']
        response=_physical(row['candidate'],row['r'],row['x'],latent,context['root_v'],context['future_meter'],context['options'])
        JL,Jz=response['line_jacobian_eta'],response['state_jacobian']; F=np.c_[JL,Jz]
        C=_mode_covariance(row,covariance_mode); d=row['line_dimension']; dimension=len(response['mean'])
        line=JL@C[:d,:d]@JL.T; state=Jz@C[d:,d:]@Jz.T
        cross=JL@C[:d,d:]@Jz.T; cross=cross+cross.T
        covariance=np.eye(dimension)+line+state+cross; covariance=.5*(covariance+covariance.T)
        # Meter I makes this positive definite if the joint covariance is PSD.
        np.linalg.cholesky(covariance)
        columns=np.array([t*(2*n+2)+j for t in range(T) for j in range(n)])
        means.append(response['mean']); covariances.append(covariance)
        jacobians.append(Jz[:,columns]/context['base_kw']); joint_jacobians.append(F)
        physical.append(dict(candidate=row['candidate'].name,v_min=float(response['physical']['v'].min()),
            v_max=float(response['physical']['v'].max()),ac_equation_residual=float(response['physical']['ac_equation_residual']),
            power_balance_residual=float(response['physical']['power_balance_residual']),
            latent_inputs_with_EV=latent,predicted_whitened_mean=response['mean'],
            covariance_parts=dict(line=line,state=state,cross=cross,meter_identity=True),
            covariance_mode=covariance_mode,line_center_sha256=row['line_sha256'],source_interface=1,EV_added_once=True))
    lp=context['log_prior_all_history' if weights_mode=='all_history' else 'log_prior_evidence_only']
    bank=dict(means=np.asarray(means)[None],covariances=np.asarray(covariances)[None],prior=np.exp(lp),
        log_prior=np.asarray(lp).copy(),weights=np.ones(1))
    return dict(bank=bank,jacobian=np.asarray(jacobians)[None] if with_jacobian else None,
        joint_jacobian=joint_jacobians,physical=physical,AC_batches=len(physical),AC_states=len(physical)*T,
        covariance_mode=covariance_mode,weights_mode=weights_mode,
        seconds=float(perf_counter()-began),
        scope='full AC historical-center mean; bounded posterior joint moment propagation')


def response_covariance(context,power,reference_power=None,*,covariance_mode='full'):
    """Same latent/line realization under two actions; no meter noise is added."""
    if reference_power is None: reference_power=np.zeros_like(power)
    a=forecast_bank(context,power,covariance_mode=covariance_mode)
    b=forecast_bank(context,reference_power,covariance_mode=covariance_mode)
    rows=[]
    for row,F,G in zip(context['rows'],a['joint_jacobian'],b['joint_jacobian']):
        D=F-G; C=_mode_covariance(row,covariance_mode); d=row['line_dimension']
        line=D[:,:d]@C[:d,:d]@D[:,:d].T; state=D[:,d:]@C[d:,d:]@D[:,d:].T
        cross=D[:,:d]@C[:d,d:]@D[:,d:].T; cross=cross+cross.T
        rows.append(dict(candidate=row['candidate'].name,covariance=line+state+cross,
            line=line,state=state,cross=cross,joint_jacobian_difference=D))
    return dict(rows=rows,physical_response_meter_noise_included=False,shared_random_line_and_state=True)


def ablate_forecast(context,forecast,*,covariance_mode='full',weights_mode='all_history'):
    """Derive same-action ablations from saved AC/Jacobians, with zero new AC.

    The original forecast is not modified. Mean, historical center and full AC
    derivatives stay fixed; only the explicitly declared covariance/weight
    component changes. This is a reporting ablation, not a new physical solve.
    """
    began=perf_counter()
    if weights_mode not in ('all_history','evidence_only'): raise ValueError('unknown weights_mode')
    result=deepcopy(forecast);covariances=[]
    if len(forecast['joint_jacobian'])!=len(context['rows']): raise ValueError('candidate bank differs')
    for i,(row,F) in enumerate(zip(context['rows'],forecast['joint_jacobian'])):
        if forecast['physical'][i]['line_center_sha256']!=row['line_sha256']:
            raise ValueError('forecast historical center differs')
        F=np.asarray(F);d=row['line_dimension'];C=_mode_covariance(row,covariance_mode)
        JL,Jz=F[:,:d],F[:,d:]
        line=JL@C[:d,:d]@JL.T;state=Jz@C[d:,d:]@Jz.T
        cross=JL@C[:d,d:]@Jz.T;cross=cross+cross.T
        S=np.eye(len(F))+line+state+cross;S=.5*(S+S.T);np.linalg.cholesky(S)
        covariances.append(S)
        result['physical'][i]['covariance_parts']=dict(line=line,state=state,cross=cross,meter_identity=True)
        result['physical'][i]['covariance_mode']=covariance_mode
    lp=context['log_prior_all_history' if weights_mode=='all_history' else 'log_prior_evidence_only']
    result['bank']['covariances']=np.asarray(covariances)[None]
    result['bank']['log_prior']=np.asarray(lp).copy();result['bank']['prior']=np.exp(lp)
    result.update(covariance_mode=covariance_mode,weights_mode=weights_mode,
        AC_batches=0,AC_states=0,reused_physical_AC=True,seconds=float(perf_counter()-began))
    return result


def observations_vector(observations,meter): return whiten_observations(observations,meter)
