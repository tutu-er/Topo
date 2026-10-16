"""Fixed trained lines, proper local Gaussian forecasts and full AC means.

Only observed records, fitted references and public meter quantities enter the
selector. Latent P/Q and a single-interface source are integrated analytically;
they are not additional fitted future variables. All AC coordinates are pu.
"""
from __future__ import annotations

from hashlib import sha256
from time import perf_counter
import numpy as np
from scipy.linalg import cho_factor, cho_solve, block_diag
from scipy.special import logsumexp

from . import likelihood as lk
from .symmetric_ac_local import mean_and_jacobian, whiten_observations, CHANNELS
from .ev_mobile_v4_cases import harmonic_features


def _spd(value):
    value = (np.asarray(value, float) + np.asarray(value, float).T) / 2
    eigen, vectors = np.linalg.eigh(value)
    scale = max(float(np.max(abs(eigen), initial=0)), 1e-20)
    if eigen.min(initial=0) < -1e-8 * scale:
        raise ValueError('materially indefinite covariance')
    return (vectors * np.maximum(eigen, scale * 1e-12)) @ vectors.T


def process_forecast(values, count, noise_floor):
    """Proper harmonic Gaussian prior + stationary AR kernel, conditioned on T.

    Plug-in hyperparameters use training only. This specifies a conditional
    Gaussian predictive model, not an exact posterior for unknown dynamics.
    Independent channel processes retain their within-channel time covariance.
    """
    values = np.asarray(values, float)
    if values.ndim != 2 or len(values) < 20 or count < 1:
        raise ValueError('training [time,channel] and positive horizon required')
    H, width = values.shape
    X = harmonic_features(np.arange(H))
    Z = harmonic_features(np.arange(H, H + count))
    beta, _, rank, _ = np.linalg.lstsq(X, values, rcond=None)
    if rank != X.shape[1]:
        raise ValueError('rank deficient training forecast')
    residual = values - X @ beta
    floor = np.broadcast_to(np.asarray(noise_floor, float), (width,))
    means = np.zeros((count,width))
    covariance = np.zeros((count * width, count * width))
    info = []
    for j in range(width):
        r = residual[:, j]
        meter_variance = float(floor[j]) ** 2
        corrected_lag_variance = max(float(np.mean(r[:-1] ** 2)) - meter_variance,
                                     .01 * meter_variance, 1e-30)
        phi_raw = float(np.mean(r[:-1] * r[1:]) / corrected_lag_variance)
        phi = float(np.clip(phi_raw, -.9, .9))
        innovations = r[1:] - phi * r[:-1]
        raw_innovation_variance = float(np.mean(innovations ** 2))
        variance = max(raw_innovation_variance - (1+phi**2)*meter_variance,
                       .01*meter_variance)
        # Public proper coefficient prior N(0, .5^2 I) is declared before data.
        # OLS above estimates AR hyperparameters only, never a data-dependent prior mean.
        coefficient_variance = .5**2
        full_features=np.vstack((X,Z));times=np.arange(H+count)
        stationary_variance=variance/(1-phi**2)
        kernel=(coefficient_variance*full_features@full_features.T
                +stationary_variance*phi**abs(times[:,None]-times[None,:]))
        train_factor=cho_factor(kernel[:H,:H]+meter_variance*np.eye(H),lower=True)
        cross=kernel[H:,:H]
        means[:,j]=cross@cho_solve(train_factor,values[:,j])
        cov=kernel[H:,H:]-cross@cho_solve(train_factor,cross.T)
        indices = np.arange(count) * width + j
        covariance[np.ix_(indices, indices)] = cov
        info.append(dict(phi=phi, unclipped_phi=phi_raw, innovation_variance=variance,
                         coefficient_variance=coefficient_variance,
                         coefficient_prior_mean=0.,coefficient_prior_sd=.5,
                         conditional_AR_state_variance_retained=True,
                         training_meter_variance=meter_variance,
                         raw_innovation_variance=raw_innovation_variance,
                         subtracted_innovation_meter_variance=(1+phi**2)*meter_variance,
                         process_variance_floor=.01*meter_variance,
                         floor_standard_deviation=float(floor[j])))
    return means, _spd(covariance), dict(rank=int(rank), channels=info, training_count=H,
        hyperparameters_frozen=True, cross_channel_process_covariance='independent before conditioning',
        kernel='proper harmonic coefficient prior plus stationary AR; exact Gaussian conditioning',
        source_or_PQ_true_values_used=False)


def _meter_sigma(model, name, shape):
    return np.broadcast_to(np.asarray(getattr(model, name + '_sigma'), float), shape)


def _line_id(r, x):
    return sha256(np.r_[r, x].astype('<f8').tobytes()).hexdigest()


def _physical_model(candidate, r, x, latent, meter, *, with_jacobian=True):
    n = candidate.n
    result = mean_and_jacobian(candidate, r, x, latent[:, :n], latent[:, n:2*n],
        latent[:, -2], latent[:, -1], np.ones(len(latent)), meter,
        channels=CHANNELS, with_jacobian=with_jacobian)
    if not .85 < np.min(result['physical']['v']) <= np.max(result['physical']['v']) < 1.15:
        raise ValueError('AC mean outside original voltage domain')
    if np.any(abs(latent[:, -2:]) > .5):
        raise ValueError('AC mean source outside original physical interface domain')
    return result


def build_context(training, evidence, train_meter, evidence_meter, candidates, fits,
                  future_meter, *, horizon=6, base_kw=20.):
    """No case, truth, generating coefficients, future data or noise argument."""
    began = perf_counter()
    n, H, V = training.n, training.t, evidence.t
    if evidence.n != n or any(not fits[c.name].success for c in candidates):
        raise ValueError('complete single trained reference per topology required')
    p = training.p - np.broadcast_to(train_meter.bias_p, training.p.shape)
    q = training.q - np.broadcast_to(train_meter.bias_q, training.q.shape)
    pq = np.c_[p, q]
    pq_floor = np.r_[_meter_sigma(train_meter, 'p', p.shape).mean(axis=0),
                     _meter_sigma(train_meter, 'q', q.shape).mean(axis=0)]
    common_mean, common_cov, common_info = process_forecast(pq, V + horizon, pq_floor)
    observed = whiten_observations(evidence, evidence_meter)
    raw_log_weights, rows, source_updates = [], [], {}
    latent_width = 2*n + 2
    for candidate in candidates:
        fit = fits[candidate.name]
        source_values = np.c_[fit.source_p, fit.source_q]
        source_floor = np.sqrt(np.array([
            np.mean(np.asarray(train_meter.master_p_sigma))**2 + np.sum(pq_floor[:n]**2),
            np.mean(np.asarray(train_meter.master_q_sigma))**2 + np.sum(pq_floor[n:]**2)]))
        source_mean, source_cov, source_info = process_forecast(source_values, V + horizon, source_floor)
        mean = np.c_[common_mean, source_mean]
        covariance = np.zeros(((V+horizon)*latent_width, (V+horizon)*latent_width))
        pq_indices = np.array([t*latent_width+j for t in range(V+horizon) for j in range(2*n)])
        source_indices = np.array([t*latent_width+j for t in range(V+horizon) for j in (2*n, 2*n+1)])
        covariance[np.ix_(pq_indices, pq_indices)] = common_cov
        covariance[np.ix_(source_indices, source_indices)] = source_cov
        predicted = _physical_model(candidate, fit.r, fit.x, mean[:V], evidence_meter)
        J = predicted['state_jacobian']
        cross = covariance[:, :V*latent_width] @ J.T
        output_cov = np.eye(len(observed)) + J @ covariance[:V*latent_width, :V*latent_width] @ J.T
        factor = cho_factor(output_cov, lower=True)
        residual = observed - predicted['mean']
        log_likelihood = -.5 * (residual @ cho_solve(factor, residual)
            + 2*np.log(np.diag(factor[0])).sum() + len(observed)*np.log(2*np.pi))
        posterior_mean = mean.ravel() + cross @ cho_solve(factor, residual)
        posterior_cov = _spd(covariance - cross @ cho_solve(factor, cross.T))
        future_mean = posterior_mean.reshape(V+horizon, latent_width)[V:]
        future_cov = posterior_cov[V*latent_width:, V*latent_width:]
        raw_log_weights.append(log_likelihood - np.log(len(candidates)))
        source_updates[candidate.name] = dict(training_process=source_info,
            source_training_meter_noise_proxy='slack-minus-terminal noise; unit loss slopes; excludes fitted-line uncertainty',
            evidence_log_predictive=float(log_likelihood),
            prior_latent_mean=mean, prior_latent_covariance=covariance,
            evidence_predicted_mean=predicted['mean'], evidence_covariance=output_cov,
            evidence_observed_whitened=observed,
            posterior_latent_mean=posterior_mean.reshape(V+horizon, latent_width),
            posterior_latent_covariance=posterior_cov,
            future_latent_mean=future_mean.copy(),future_latent_covariance=future_cov.copy(),
            evidence_ac_equation=float(predicted['physical']['ac_equation_residual']))
        rows.append(dict(candidate=candidate, r=fit.r.copy(), x=fit.x.copy(),
            line_sha256=_line_id(fit.r, fit.x), latent_mean=future_mean, latent_cov=future_cov))
    log_prior = np.asarray(raw_log_weights) - logsumexp(raw_log_weights)
    prior = np.exp(log_prior)
    context = dict(rows=rows, prior=prior, log_prior=log_prior, future_meter=future_meter,
        horizon=horizon, n=n, base_kw=base_kw, source_interface=1,
        training_count=H,evidence_count=V,
        model_scope='fixed training R/X; full joint PQ/V/master local Gaussian prediction',
        source_process='same fixed physical interface; temporal and conditioning correlations retained')
    record = dict(training_count=H,evidence_count=V,horizon=horizon,base_kw=base_kw,
        candidate_names=[c.name for c in candidates], prior=prior,log_prior=log_prior,
        prior_display_underflow_count=int(np.count_nonzero(prior == 0)),
        evidence_log_likelihoods=np.asarray(raw_log_weights)+np.log(len(candidates)),
        common_PQ_process=common_info,candidate_updates=source_updates,
        source_interface=1,fixed_R_X=True,training_used_for_evidence=False,
        posterior_probability_scope=context['model_scope'],seconds=perf_counter()-began,
        latent_future_PQ_optimized=False, P_Q_measurement_error_in_joint_density=True)
    return context, record


def forecast_bank(context, power, *, with_jacobian=False):
    """Full AC means; first-order propagation of conditioned latent covariance."""
    began = perf_counter()
    T, n = context['horizon'], context['n']
    power = np.asarray(power, float)
    if power.shape != (T, n) or not np.isfinite(power).all():
        raise ValueError('finite power [T,n] required')
    means, covariances, jacobians, physical = [], [], [], []
    for row in context['rows']:
        if _line_id(row['r'], row['x']) != row['line_sha256']:
            raise AssertionError('trained lines changed')
        latent = row['latent_mean'].copy()
        latent[:, :n] += power/context['base_kw']
        response = _physical_model(row['candidate'], row['r'], row['x'], latent, context['future_meter'])
        J = response['state_jacobian']
        means.append(response['mean'])
        covariances.append(_spd(np.eye(len(response['mean'])) + J @ row['latent_cov'] @ J.T))
        columns = np.array([t*(2*n+2)+j for t in range(T) for j in range(n)])
        jacobians.append(J[:, columns]/context['base_kw'])
        physical.append(dict(candidate=row['candidate'].name,
            v_min=float(np.min(response['physical']['v'])),v_max=float(np.max(response['physical']['v'])),
            ac_equation_residual=float(response['physical']['ac_equation_residual']),
            power_balance_residual=float(response['physical']['power_balance_residual']),
            latent_inputs_with_EV=latent,
            predicted_whitened_mean=response['mean'],source_interface=1,EV_added_once=True))
    bank = dict(means=np.asarray(means)[None],covariances=np.asarray(covariances)[None],
        prior=context['prior'].copy(),log_prior=context['log_prior'].copy(),weights=np.ones(1))
    return dict(bank=bank,jacobian=np.asarray(jacobians)[None] if with_jacobian else None,
        physical=physical,seconds=perf_counter()-began,AC_batches=len(physical),AC_states=len(physical)*T)


def observations_vector(observations, meter):
    return whiten_observations(observations,meter)


def recover_physical(observed, context, *, candidate_index):
    """Condition the declared Gaussian on measured PQ; no load optimization."""
    row=context['rows'][candidate_index]
    T,n=context['horizon'],context['n']
    b=forecast_bank(context,np.zeros((T,n)))['bank']
    d=2*T*n
    y=observations_vector(observed,context['future_meter'])
    mu=b['means'][0,candidate_index];S=b['covariances'][0,candidate_index]
    factor=cho_factor(S[:d,:d],lower=True)
    cond_mean=mu[d:]+S[d:,:d]@cho_solve(factor,y[:d]-mu[:d])
    cond_cov=S[d:,d:]-S[d:,:d]@cho_solve(factor,S[:d,d:])
    return cond_mean,_spd(cond_cov)
