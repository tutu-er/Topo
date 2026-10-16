"""Read-only replay of saved Bayesian EV records, outside the science freeze.

No selector, optimizer, data generator or candidate fitting is called.  AC means
use an independent backward-current/forward-voltage sweep.  Its differentiated
nodal equations provide a separate covariance replay.  Synthetic truth enters
only the post-hoc physical and mean-error diagnostics.
"""
from __future__ import annotations

import argparse
import base64
from collections import Counter
from hashlib import sha256
from itertools import permutations, product
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.linalg import cho_factor, cho_solve, solve_triangular
from scipy.special import logsumexp

from . import gradient_price_study as io
from . import ev_mobile_v4_domain as nested
from . import ev_mobile_v2_review as old_review
from . import ev_mobile_cases_v2 as declared
from .ev_mobile_service_domain import MobileServiceConfig
from .likelihood import RadialCandidate


HERE = Path(__file__).resolve().parent
CHANNELS = ('p', 'q', 'v', 'master_p', 'master_q')
EXCEPTIONS = (ValueError, KeyError, TypeError, IndexError, AssertionError,
              FloatingPointError, RuntimeError, np.linalg.LinAlgError)
FACTORS = ('lines', 'PQ', 'source')


def _array(value):
    result = np.asarray(value, float)
    if not np.isfinite(result).all():
        raise ValueError('nonfinite saved numeric evidence')
    return result


def _require(condition, label):
    if not condition:
        raise ValueError(label)


def _close(actual, expected, label, *, atol=2e-7, rtol=2e-8):
    a, b = _array(actual), _array(expected)
    _require(a.shape == b.shape and np.allclose(a, b, atol=atol, rtol=rtol), label)
    return float(np.max(abs(a-b), initial=0.))


def _physics(candidate, r, x, latent, root=None):
    """Independent radial sweep, retaining complex states for differentiation."""
    r, x, latent = map(_array, (r, x, latent))
    T, n, m = len(latent), candidate.n, candidate.m
    _require(latent.shape == (T, 2*n+2), 'latent state dimensions')
    root = np.ones(T) if root is None else _array(root)
    load = np.zeros((T, m), complex)
    load[:, 0] = latent[:, -2] + 1j*latent[:, -1]
    load[:, candidate.terminal_edges] += latent[:, :n] + 1j*latent[:, n:2*n]
    z = r+1j*x
    voltage = np.broadcast_to(root[:, None], load.shape).astype(complex).copy()
    for iteration in range(1000):
        branch = np.conj(load/voltage)
        for e in range(m-1, -1, -1):
            parent = candidate.parents[e]
            if parent >= 0:
                branch[:, parent] += branch[:, e]
        updated = np.empty_like(voltage)
        for e, parent in enumerate(candidate.parents):
            updated[:, e] = (root if parent < 0 else updated[:, parent])-z[e]*branch[:, e]
        if np.max(abs(updated-voltage)) < 2e-13:
            voltage = updated
            break
        voltage = updated
    else:
        raise ValueError('independent AC sweep did not converge')
    branch = np.conj(load/voltage)
    for e in range(m-1, -1, -1):
        if candidate.parents[e] >= 0:
            branch[:, candidate.parents[e]] += branch[:, e]
    master = root*np.conj(branch[:, 0])
    equation = max(float(np.max(abs(voltage[:, e]-(root if p < 0 else voltage[:, p])+z[e]*branch[:, e])))
                   for e, p in enumerate(candidate.parents))
    balance = float(np.max(abs(master-load.sum(axis=1)-(abs(branch)**2*z).sum(axis=1))))
    _require(equation < 2e-10 and balance < 2e-10, 'independent AC equations and power balance')
    return dict(p=latent[:, :n], q=latent[:, n:2*n], v=abs(voltage[:, candidate.terminal_edges]),
        master_p=master.real, master_q=master.imag, source_p=latent[:, -2], source_q=latent[:, -1],
        root_v=root, voltage=voltage, load=load, equation=equation, balance=balance)


def _whiten(physical, meter):
    rows = []
    for channel in CHANNELS:
        value = _array(physical[channel])
        if channel in ('p', 'q', 'v'):
            value = value+_array(meter['bias_'+channel])
        sigma = np.broadcast_to(_array(meter[channel+'_sigma']), value.shape)
        _require(np.all(sigma > 0), 'positive meter standard deviation')
        rows.append((value/sigma).ravel())
    return np.concatenate(rows)


def _observed_vector(observations, meter):
    return np.concatenate([(_array(observations[c])/
        np.broadcast_to(_array(meter[c+'_sigma']), np.shape(observations[c]))).ravel() for c in CHANNELS])


def _jacobian(candidate, r, x, physical, meter):
    """Differentiate independent AC states; columns are timewise P,Q,source."""
    voltage, load = physical['voltage'], physical['load']
    T, m = voltage.shape
    n, width = candidate.n, 2*candidate.n+2
    zbus = (candidate.B*(_array(r)+1j*_array(x)))@candidate.B.T
    coefficient = np.conj(load)/np.conj(voltage)**2
    K = zbus[None]*coefficient[:, None, :]
    identity = np.broadcast_to(np.eye(m), K.shape)
    matrix = np.concatenate((np.concatenate((identity-K.real, -K.imag), axis=2),
                             np.concatenate((-K.imag, identity+K.real), axis=2)), axis=1)
    ds = np.zeros((m, width), complex)
    ds[candidate.terminal_edges, np.arange(n)] = 1.
    ds[candidate.terminal_edges, n+np.arange(n)] = 1j
    ds[0, -2:] = (1., 1j)
    direct = np.conj(ds)[None]/np.conj(voltage)[:, :, None]
    rhs = -np.einsum('ij,tjk->tik', zbus, direct)
    solved = np.linalg.solve(matrix, np.concatenate((rhs.real, rhs.imag), axis=1))
    dvoltage = solved[:, :m]+1j*solved[:, m:]
    dcurrent = direct-coefficient[:, :, None]*np.conj(dvoltage)
    dmaster = physical['root_v'][:, None]*np.conj(dcurrent.sum(axis=1))
    ids = candidate.terminal_edges
    dvmag = (np.conj(voltage[:, ids, None])*dvoltage[:, ids]).real/abs(voltage[:, ids, None])
    blocks = []
    for channel in CHANNELS:
        count = n if channel in ('p', 'q', 'v') else 1
        local = np.zeros((T, count, width))
        if channel == 'p':
            local[:, np.arange(n), np.arange(n)] = 1.
        elif channel == 'q':
            local[:, np.arange(n), n+np.arange(n)] = 1.
        elif channel == 'v':
            local = dvmag
        else:
            local = (dmaster.real if channel == 'master_p' else dmaster.imag)[:, None, :]
        sigma = np.broadcast_to(_array(meter[channel+'_sigma']), (T, n) if count == n else (T,))
        local = local/sigma.reshape(T, count, 1)
        block = np.zeros((T*count, T*width))
        for t in range(T):
            block[t*count:(t+1)*count, t*width:(t+1)*width] = local[t]
        blocks.append(block)
    return np.vstack(blocks)


def _features(times):
    t = _array(times)
    return np.column_stack([np.ones(len(t)), *[v for period in (96., 32., 24., 12.)
        for v in (np.sin(2*np.pi*t/period), np.cos(2*np.pi*t/period))]])


def _process_replay(values, count, metadata):
    """Replay the declared proper Gaussian kernel; training only, no true data."""
    values = _array(values)
    H, width = values.shape
    features = _features(np.arange(H+count))
    result, covariance = np.zeros((count, width)), np.zeros((count*width, count*width))
    coefficients, _, rank, _ = np.linalg.lstsq(features[:H], values, rcond=None)
    residual = values-features[:H]@coefficients
    _require(int(metadata['training_count']) == H and int(metadata['rank']) == rank, 'process training rows/rank')
    for j, row in enumerate(metadata['channels']):
        R = float(row['training_meter_variance'])
        phi = float(np.clip(np.mean(residual[:-1, j]*residual[1:, j])/
            max(float(np.mean(residual[:-1, j]**2))-R, .01*R, 1e-30), -.9, .9))
        raw = float(np.mean((residual[1:, j]-phi*residual[:-1, j])**2))
        Q = max(raw-(1+phi**2)*R, .01*R)
        _close(row['phi'], phi, 'training-only AR hyperparameter', atol=1e-12)
        _close(row['innovation_variance'], Q, 'deconvolved process innovation', atol=1e-15)
        _close(row['coefficient_variance'], .25, 'proper fixed coefficient prior', atol=0., rtol=0.)
        t = np.arange(H+count)
        kernel = .25*features@features.T+Q/(1-phi**2)*phi**abs(t[:, None]-t[None, :])
        factor = cho_factor(kernel[:H, :H]+R*np.eye(H), lower=True)
        cross = kernel[H:, :H]
        result[:, j] = cross@cho_solve(factor, values[:, j])
        block = kernel[H:, H:]-cross@cho_solve(factor, cross.T)
        ids = np.arange(count)*width+j
        covariance[np.ix_(ids, ids)] = block
    return result, (covariance+covariance.T)/2


def _logdensity(values, means, covariances, log_prior):
    means, covariances, log_prior = map(_array, (means, covariances, log_prior))
    y = _array(values)
    out = []
    for mu, covariance in zip(means, covariances):
        factor = np.linalg.cholesky(covariance)
        residual = (y-mu).reshape(-1, len(mu))
        white = solve_triangular(factor, residual.T, lower=True).T
        out.append(-.5*(len(mu)*np.log(2*np.pi)+2*np.log(np.diag(factor)).sum()+(white**2).sum(axis=1)))
    likelihood = np.column_stack(out)
    joint = likelihood+log_prior
    logpost = joint-logsumexp(joint, axis=1)[:, None]
    return np.exp(logpost), logpost


def _integration(bank, noise, cost, credible_mass=.9):
    means, covariance = _array(bank['means'])[0], _array(bank['covariances'])[0]
    lp = _array(bank['log_prior']); prior = np.exp(lp)
    K, d = means.shape
    R = noise.shape[2]
    samples = {k: np.zeros((K, R)) for k in ('risk', 'eig', 'brier', 'brier_risk', 'abstention_risk', 'map_error')}
    calibration = {k: np.zeros((K, R)) for k in ('true_rejected', 'confidence', 'correct', 'confidence_minus_correct')}
    bins = [dict(lower=j/10, upper=(j+1)/10, mass=0., confidence_mass=0., correct_mass=0.) for j in range(10)]
    for k in range(K):
        y = means[k]+noise[0, k]@np.linalg.cholesky(covariance[k]).T
        posterior, logpost = _logdensity(y, means, covariance, lp)
        samples['risk'][k] = 1-posterior.max(axis=1)
        samples['eig'][k] = (posterior*(logpost-lp)).sum(axis=1)
        samples['brier'][k] = (posterior**2).sum(axis=1)
        samples['brier_risk'][k] = 1-samples['brier'][k]
        samples['abstention_risk'][k] = np.minimum(cost, samples['risk'][k])
        samples['map_error'][k] = np.argmax(posterior, axis=1) != k
        order = np.argsort(-posterior, axis=1, kind='stable')
        cumulative = np.cumsum(np.take_along_axis(posterior, order, axis=1), axis=1)
        sizes = np.minimum(K, np.sum(cumulative < credible_mass, axis=1)+1)
        true_position = np.argmax(order == k, axis=1)
        confidence = posterior.max(axis=1); correct = np.argmax(posterior, axis=1) == k
        calibration['true_rejected'][k] = true_position >= sizes
        calibration['confidence'][k] = confidence; calibration['correct'][k] = correct
        calibration['confidence_minus_correct'][k] = confidence-correct
        assignment = np.minimum(9, (confidence*10).astype(int))
        for j, row in enumerate(bins):
            selected = assignment == j; weight = prior[k]/R
            row['mass'] += float(weight*selected.sum())
            row['confidence_mass'] += float(weight*confidence[selected].sum())
            row['correct_mass'] += float(weight*correct[selected].sum())
    result = {}
    for label, values in samples.items():
        result[label] = float(prior@values.mean(axis=1))
        result[label+'_se'] = float(np.sqrt(prior**2@values.var(axis=1, ddof=1)/R))
    result.update(sample_count_per_candidate=R, candidate_strata=K, output_dimension=d)
    calibration_summary = {}
    for label, values in calibration.items():
        calibration_summary[label] = float(prior@values.mean(axis=1))
        calibration_summary[label+'_se'] = float(np.sqrt(prior**2@values.var(axis=1, ddof=1)/R))
    for row in bins:
        mass = row['mass']
        row['mean_confidence'] = row.pop('confidence_mass')/mass if mass else None
        row['mean_correct'] = row.pop('correct_mass')/mass if mass else None
        row['confidence_minus_correct'] = row['mean_confidence']-row['mean_correct'] if mass else None
    calibration_summary.update(credible_mass=float(credible_mass), confidence_bins=bins,
        MAP_error=result['map_error'], MAP_error_se=result['map_error_se'],
        scope='prior-weighted internal check of the declared joint Gaussian bank, independent integration noise',
        independent_physical_histories=0, sample_count_per_candidate=R,
        interpretation='tests Gaussian implementation and integration; physical predictive calibration assessed separately')
    result['declared_model_internal_calibration'] = calibration_summary
    return result, samples


def _paired(left, right, prior):
    result = {}
    for label in left:
        values = left[label]-right[label]
        result[label] = float(prior@values.mean(axis=1))
        result[label+'_se'] = float(np.sqrt(prior**2@values.var(axis=1, ddof=1)/values.shape[1]))
    return result


def _bc(bank):
    means, covariance = _array(bank['means'])[0], _array(bank['covariances'])[0]
    lp = _array(bank['log_prior'])
    rows = []
    for i in range(len(means)):
        for j in range(i+1, len(means)):
            average = (covariance[i]+covariance[j])/2
            factor = cho_factor(average, lower=True)
            delta = means[i]-means[j]
            mean_term = float(delta@cho_solve(factor, delta)/8)
            covariance_term = float(.5*(np.linalg.slogdet(average)[1]-.5*(
                np.linalg.slogdet(covariance[i])[1]+np.linalg.slogdet(covariance[j])[1])))
            distance = mean_term+covariance_term
            _require(distance >= -1e-8, 'nonnegative Gaussian Bhattacharyya distance')
            contribution = float(np.exp(.5*(lp[i]+lp[j])-max(0., distance)))
            rows.append(dict(i=i, j=j, B=max(0., distance), contribution=contribution,
                             mean_term=mean_term, covariance_term=covariance_term))
    return dict(bound=float(sum(v['contribution'] for v in rows)), pairs=rows,
                scope='saved complete joint Gaussian model, not physical data error guarantee')


def _shapley(errors):
    values = np.zeros(3)
    for order in permutations(range(3)):
        before = '000'; bits = [0, 0, 0]
        for j in order:
            bits[j] = 1; after = ''.join(map(str, bits))
            values[j] += (errors[before]-errors[after])/6
            before = after
    return dict(contributions=dict(zip(FACTORS, map(float, values))),
        total_reduction=float(errors['000']-errors['111']),
        identity_error=float(abs(values.sum()-(errors['000']-errors['111']))),
        negative_contribution='correction exposes a cancelling prediction error')


def _attribution(candidate, trained, true_lines, latent_mean, truth, power, base_kw):
    predicted, errors = {}, {c: {} for c in ('v', 'master_p', 'master_q')}
    for bits in product('01', repeat=3):
        mask = ''.join(bits)
        r, x = (true_lines['r'], true_lines['x']) if bits[0] == '1' else (trained['r'], trained['x'])
        latent = _array(latent_mean).copy()
        if bits[1] == '1':
            latent[:, :candidate.n] = _array(truth['p']); latent[:, candidate.n:2*candidate.n] = _array(truth['q'])
        if bits[2] == '1':
            latent[:, -2] = _array(truth['source_p']); latent[:, -1] = _array(truth['source_q'])
        latent[:, :candidate.n] += power/base_kw
        predicted[mask] = _physics(candidate, r, x, latent, truth['root_v'])
    for c in errors:
        for mask, physical in predicted.items():
            errors[c][mask] = float(np.mean((physical[c]-predicted['111'][c])**2))
    return dict(success=True, mean_MSE_by_mask=errors,
        allocation={c: _shapley(values) for c, values in errors.items()},
        units='voltage pu squared; master power pu squared', independent_AC_batches=8,
        target='complete AC clean mean at the same fixed action; absolute response, not increment',
        is_decision_or_error_rejection_attribution=False, synthetic_truth_used_posthoc_only=True)


def audit_record(record, *, include_attribution=True):
    began = perf_counter(); errors, missing = [], []
    counts = Counter(); maxima = Counter(); action_results = {}; actual_results = {}; first_observations = {}

    def attempt(label, function):
        try:
            return function()
        except EXCEPTIONS as exc:
            errors.append(dict(check=label, reason=type(exc).__name__+': '+str(exc)))
            return None

    def compare(a, b, label, key, **kwargs):
        error = _close(a, b, label, **kwargs); maxima[key] = max(maxima[key], error)

    required = ('prediction', 'training_result', 'future_meter', 'observed', 'observed_meter', 'methods', 'actions')
    for field in required:
        if field not in record:
            missing.append(dict(field=field, reason=record.get('failure', record.get('status'))))
    if missing:
        return dict(success=False, spec=record.get('spec'), status=record.get('status'),
                    counts=dict(counts), maxima=dict(maxima), errors=errors, missing=missing)
    cfg, prediction = record['config'], record['prediction']
    H, V, T = (int(cfg[k]) for k in ('train', 'evidence', 'horizon'))
    n = len(record['observed']['p'][0]); width = 2*n+2; d = (3*n+2)*T
    source_cfg = declared.case_config('remote8', training_points=96)
    fits = {v.get('candidate', v.get('name')): v for v in record['training_result']['fits']}
    candidates = {name: RadialCandidate.from_supports(name, n,
        [tuple(s) for s in fits[name]['supports'] if 1 < len(s) < n]) for name in record['candidate_names']}
    names = list(candidates); K = len(names); seed = int(record['spec']['seed'])
    fm = record['future_meter']; dm = record['observed_meter']; base_kw = float(prediction['base_kw'])
    domains = nested.build_nested_domains(MobileServiceConfig(**record['operation_config']))
    counts.update(independent_histories=1, candidate_slots=K, method_slots=len(record['methods']),
                  unique_action_slots=len(record['actions']))

    def line_domain(name, row):
        state = {k: _array(row[k]) for k in ('r', 'x', 'source_p', 'source_q')}
        line, source = old_review._domain(state, source_cfg)
        _require(line and source, 'training original line/source domain '+name)
        counts['trained_line_and_source_domains'] += 1
    for name in names:
        attempt('training_domain/'+name, lambda name=name: line_domain(name, fits[name]))

    observed = {c: _array(record['observed'][c]) for c in CHANNELS}
    observed['root_v'] = _array(record['observed']['root_v'])
    training_meter = {k: (_array(v)[:H] if isinstance(v, list) and np.asarray(v).ndim and len(v) == H+V else v)
                      for k, v in dm.items()}
    evidence_meter = {k: (_array(v)[H:] if isinstance(v, list) and np.asarray(v).ndim and len(v) == H+V else v)
                      for k, v in dm.items()}
    training_PQ = np.c_[observed['p'][:H]-_array(training_meter['bias_p']),
                          observed['q'][:H]-_array(training_meter['bias_q'])]
    training_observed = {c: value[:H] for c, value in observed.items()}

    def saved_state_check(candidate, row, state, blocks, label):
        actual = old_review.replay_state(candidate, state, blocks, source_cfg)
        compare(row['loss'], actual['loss'], label+' output SSE', 'saved_fit_SSE_error')
        if 'loglik' in row:
            compare(row['loglik'], actual['loglik'], label+' output log likelihood', 'saved_fit_loglik_error')
        if 'normalizer' in row:
            compare(row['normalizer'], actual['normalizer'], label+' output normalizer', 'saved_fit_normalizer_error')
        for block, saved in row.get('blocks', {}).items():
            compare(saved['loss'], actual['blocks'][block]['loss'], label+' block SSE', 'saved_fit_SSE_error')
            compare(saved['residual'], actual['blocks'][block]['residual'], label+' whitened residual', 'saved_fit_residual_error')
        for flag in ('line_domain_feasible', 'source_box_feasible', 'physical_voltage_feasible'):
            if flag in row:
                _require(bool(row[flag]) == bool(actual[flag]), label+' domain flag '+flag)
            if row.get('success'):
                _require(actual[flag], label+' successful fit domain '+flag)
        maxima['AC_equation'] = max(maxima['AC_equation'], actual['max_AC_equation'])
        maxima['power_balance'] = max(maxima['power_balance'], actual['max_power_balance'])
        counts['saved_fit_states'] += 1; counts['saved_fit_AC_blocks'] += len(blocks)
        return actual

    for name, candidate in candidates.items():
        attempt('training_SSE/'+name, lambda name=name, candidate=candidate: saved_state_check(
            candidate, fits[name], fits[name], (('training', training_observed, training_meter),), 'training/'+name))
    common_process = attempt('training_only_PQ_process', lambda: _process_replay(
        training_PQ, V+T, prediction['common_PQ_process']))
    evidence_logs = []
    for name, candidate in candidates.items():
        def evidence_check(name=name, candidate=candidate):
            update = prediction['candidate_updates'][name]; trained = fits[name]
            latent, covariance = _array(update['prior_latent_mean']), _array(update['prior_latent_covariance'])
            sp = np.c_[trained['source_p'], trained['source_q']]
            source_mean, source_cov = _process_replay(sp, V+T, update['training_process'])
            _require(common_process is not None, 'common process replay required')
            m0 = np.c_[common_process[0], source_mean]
            C0 = np.zeros_like(covariance)
            pq_ids = [t*width+j for t in range(V+T) for j in range(2*n)]
            source_ids = [t*width+j for t in range(V+T) for j in (2*n, 2*n+1)]
            C0[np.ix_(pq_ids, pq_ids)] = common_process[1]
            C0[np.ix_(source_ids, source_ids)] = source_cov
            compare(latent, m0, 'training-only process mean', 'process_mean_error', atol=1e-8)
            compare(covariance, C0, 'training-only process covariance', 'process_covariance_error', atol=1e-10)
            physical = _physics(candidate, trained['r'], trained['x'], latent[:V])
            mu = _whiten(physical, evidence_meter); J = _jacobian(candidate, trained['r'], trained['x'], physical, evidence_meter)
            y = _observed_vector({c: observed[c][H:] for c in CHANNELS}, evidence_meter)
            S = np.eye(len(y))+J@covariance[:V*width, :V*width]@J.T
            cross = covariance[:, :V*width]@J.T; factor = cho_factor(S, lower=True)
            residual = y-mu
            loglike = -.5*(residual@cho_solve(factor, residual)+2*np.log(np.diag(factor[0])).sum()+len(y)*np.log(2*np.pi))
            posterior_mean = latent.ravel()+cross@cho_solve(factor, residual)
            posterior_cov = covariance-cross@cho_solve(factor, cross.T)
            for saved, actual, label, key in (
                (update['evidence_predicted_mean'], mu, 'evidence AC mean', 'white_mean_error'),
                (update['evidence_covariance'], S, 'evidence covariance', 'white_covariance_error'),
                (update['evidence_observed_whitened'], y, 'evidence observed vector', 'white_observation_error'),
                (update['posterior_latent_mean'], posterior_mean.reshape(V+T, width), 'evidence posterior mean', 'posterior_mean_error'),
                (update['posterior_latent_covariance'], posterior_cov, 'evidence posterior covariance', 'posterior_covariance_error')):
                compare(saved, actual, label, key)
            compare(update['evidence_log_predictive'], loglike, 'prior predictive evidence before conditioning', 'evidence_log_density_error')
            compare(update['future_latent_mean'], posterior_mean.reshape(V+T, width)[V:], 'future conditional mean', 'posterior_mean_error')
            compare(update['future_latent_covariance'], posterior_cov[V*width:, V*width:], 'future conditional covariance', 'posterior_covariance_error')
            maxima['AC_equation'] = max(maxima['AC_equation'], physical['equation'])
            maxima['power_balance'] = max(maxima['power_balance'], physical['balance'])
            evidence_logs.append((name, float(loglike))); counts['evidence_candidates_replayed'] += 1
        attempt('evidence/'+name, evidence_check)
    if len(evidence_logs) == K:
        loglike = np.array([dict(evidence_logs)[name] for name in names])
        lp = loglike-logsumexp(loglike)
        attempt('evidence weights', lambda: compare(prediction['log_prior'], lp, 'normalized evidence log weights', 'log_prior_error'))

    noise = np.random.default_rng(seed+3000).standard_normal((1, K, int(cfg['independent_samples_per_candidate']), d))
    independent_samples = {}
    for key, action in record['actions'].items():
        def action_check(key=key, action=action):
            power = _array(action['power_kw']); saved = action['forecast']; bank = saved['bank']
            _require(nested.action_key(power) == key, 'action key')
            for i, (name, candidate) in enumerate(candidates.items()):
                trained = fits[name]; update = prediction['candidate_updates'][name]
                latent = _array(update['future_latent_mean']).copy(); latent[:, :n] += power/base_kw
                physical = _physics(candidate, trained['r'], trained['x'], latent)
                mu = _whiten(physical, fm); J = _jacobian(candidate, trained['r'], trained['x'], physical, fm)
                covariance = np.eye(d)+J@_array(update['future_latent_covariance'])@J.T
                compare(_array(bank['means'])[0, i], mu, 'future full AC mean', 'white_mean_error')
                compare(_array(bank['covariances'])[0, i], covariance, 'future joint covariance once', 'white_covariance_error')
                saved_physical = saved['physical'][i]
                compare(saved_physical['latent_inputs_with_EV'], latent, 'EV injection once at terminal', 'latent_EV_error', atol=1e-12)
                compare(saved_physical['v_min'], physical['v'].min(), 'predicted terminal minimum voltage', 'physical_voltage_error', atol=1e-11)
                compare(saved_physical['v_max'], physical['v'].max(), 'predicted terminal maximum voltage', 'physical_voltage_error', atol=1e-11)
                _require(np.all(abs(latent[:, -2:]) <= .5), 'predicted mean source physical interface domain')
                _require(.85 < physical['v'].min() and physical['v'].max() < 1.15, 'predicted mean voltage domain')
                if saved.get('jacobian') is not None:
                    columns = [t*width+j for t in range(T) for j in range(n)]
                    compare(_array(saved['jacobian'])[0, i], J[:, columns]/base_kw, 'EV mean Jacobian', 'EV_jacobian_error')
                maxima['AC_equation'] = max(maxima['AC_equation'], physical['equation'])
                maxima['power_balance'] = max(maxima['power_balance'], physical['balance'])
                counts['predictive_candidate_action_states'] += 1
            compare(bank['log_prior'], prediction['log_prior'], 'fixed posterior across actions', 'log_prior_error', atol=0., rtol=0.)
            _close(bank['weights'], [1.], 'single joint Gaussian bank', atol=0., rtol=0.)
            metrics, samples = _integration(bank, noise, float(cfg['abstention_cost']), float(cfg['credible_mass']))
            independent_samples[key] = samples
            for label in samples:
                for field in (label, label+'_se'):
                    compare(action['independent_score'][field], metrics[field], 'independent integral '+field, 'integration_error', atol=2e-10)
            bound = _bc(bank)
            _require(metrics['risk'] <= bound['bound']+8*metrics['risk_se']+1e-10,
                     'risk integral incompatible with analytical Gaussian BC upper bound')
            action_results[key] = dict(success=True, independent=metrics, BC=bound,
                risk_to_BC_margin=bound['bound']-metrics['risk'],
                compatibility_test='MC risk <= analytical BC + 8 integration SE; not a statistical coverage certificate')
            counts['independent_integrals'] += 1; counts['BC_pair_checks'] += len(bound['pairs'])
        attempt('action/'+key, action_check)

    for method_name, method in record['methods'].items():
        def method_check(method_name=method_name, method=method):
            key = method['action_key']; action = record['actions'][key]
            compare(method['power_kw'], action['power_kw'], 'method/action power equality', 'operation_power_error', atol=1e-12)
            if method['domain'] is None:
                _require(method_name == 'none' and np.max(abs(_array(method['power_kw']))) == 0., 'only noEV has no service domain')
                return
            domain = domains[method['domain']]; x = _array(method['operation_x'])
            check = domain.check(x)
            _require(check['feasible'], 'complete EV path/SOC/task physical domain')
            compare(domain.decode(x)['u_kw'], method['power_kw'], 'decoded path control', 'operation_power_error', atol=1e-10)
            counts['complete_operation_vectors'] += 1
        attempt('method/'+method_name, method_check)
    for domain, parents in nested.PARENTS.items():
        def inheritance_check(domain=domain, parents=parents):
            method = record['methods']['bayes__'+domain]; risk = float(method['planning_score']['risk'])
            log = record['bayes_logs'][domain]
            compare(log['risk'], risk, 'selected common risk', 'inheritance_score_error', atol=1e-12)
            _require(risk <= float(log['initial_risk'])+1e-12, 'accepted risk not above initial anchor')
            counts['recorded_MILP_failures'] += sum(not bool(e['milp'].get('success')) for e in log['events'])
            counts['recorded_MILP_acceptances'] += sum(bool(e['accepted']) for e in log['events'])
            for event in log['events']:
                if event['accepted']:
                    _require(float(event['proposed_risk']) < float(event['parent_risk'])-1e-12, 'accepted event risk rule')
            for parent in parents:
                p = record['methods']['bayes__'+parent]
                lifted = nested.lift_operation(domains[parent], domains[domain], _array(p['operation_x']))
                _require(domains[domain].check(lifted)['feasible'], 'parent full path legal lift')
                _require(risk <= float(p['planning_score']['risk'])+1e-12, 'common risk nonincrease over direct parent')
                counts['parent_path_lifts'] += 1
        attempt('inheritance/'+domain, inheritance_check)
    for label, saved in record.get('independent_paired', {}).items():
        def pair_check(label=label, saved=saved):
            if label.startswith('bayes_minus_square__'):
                domain = label.split('__', 1)[1]; left, right = 'bayes__'+domain, 'square__'+domain
            else:
                left, parent = label.split('_minus_', 1); right = 'bayes__'+parent
            a, b = (record['methods'][m]['action_key'] for m in (left, right))
            pair = _paired(independent_samples[a], independent_samples[b], np.exp(_array(prediction['log_prior'])))
            for field, value in pair.items():
                compare(saved[field], value, 'paired integral '+field, 'paired_integration_error', atol=2e-10)
            counts['paired_integral_comparisons'] += 1
        attempt('paired/'+label, pair_check)

    true_name = record.get('true_name'); truths = record.get('true_future_inputs_evaluation_only', {})
    if true_name not in candidates or not truths or 'true_lines' not in record:
        missing.append(dict(field='true_future_inputs/true_lines/true_name', reason='offline physical replay evidence unavailable'))
    else:
        true_candidate = candidates[true_name]; true_index = names.index(true_name)
        for regime, truth in truths.items():
            actual_results[regime] = {}
            for key, action in record['actions'].items():
                def actual_check(key=key, action=action, truth=truth, regime=regime):
                    power = _array(action['power_kw']); latent = np.c_[truth['p'], truth['q'], truth['source_p'], truth['source_q']]
                    latent[:, :n] += power/base_kw
                    physical = _physics(true_candidate, record['true_lines']['r'], record['true_lines']['x'], latent, truth['root_v'])
                    saved = action['actual'][regime]; R = int(cfg['actual_noise_repeats'])
                    _require(len(saved['samples']) == R and saved['n_noise'] == R and saved['n_independent_histories'] == 1,
                             'actual noise repeats distinct from independent history count')
                    compare(saved['actual_v_min'], physical['v'].min(), 'actual clean voltage minimum', 'actual_voltage_error', atol=1e-11)
                    compare(saved['actual_v_max'], physical['v'].max(), 'actual clean voltage maximum', 'actual_voltage_error', atol=1e-11)
                    feasible = bool(physical['v'].min() >= .85 and physical['v'].max() <= 1.15)
                    _require(bool(saved['physical_voltage_feasible']) == feasible, 'actual physical feasibility flag')
                    counts['recorded_actual_voltage_failures'] += not feasible
                    rng = np.random.default_rng(saved['noise_seed'])
                    white = {c: rng.standard_normal((R, T, n) if c in ('p', 'q', 'v') else (R, T)) for c in CHANNELS}
                    first = {c: physical[c]+np.broadcast_to(_array(fm[c+'_sigma']), physical[c].shape)*white[c][0]
                             +(np.broadcast_to(_array(fm['bias_'+c]), physical[c].shape) if c in ('p', 'q', 'v') else 0.)
                             for c in CHANNELS}
                    first['root_v'] = physical['root_v']
                    first_observations[(regime, key)] = first
                    y = np.tile(_whiten(physical, fm), (R, 1))+np.concatenate([white[c].reshape(R, -1) for c in CHANNELS], axis=1)
                    bank = action['forecast']['bank']; posterior, lp = _logdensity(y, _array(bank['means'])[0], _array(bank['covariances'])[0], bank['log_prior'])
                    for i, row in enumerate(saved['samples']):
                        compare(row['posterior'], posterior[i], 'actual joint posterior', 'actual_posterior_error', atol=2e-10)
                        order = np.argsort(-posterior[i], kind='stable')
                        count = min(K, int(np.searchsorted(np.cumsum(posterior[i, order]), float(cfg['credible_mass'])))+1)
                        credible = order[:count]; chosen = int(order[0])
                        target = np.eye(K)[true_index]
                        expected = dict(correct=int(chosen == true_index), confidence=float(posterior[i, chosen]),
                            true_probability=float(posterior[i, true_index]), brier=float(np.sum((posterior[i]-target)**2)),
                            negative_log_true_probability=float(-lp[i, true_index]), unique_correct=int(count == 1 and chosen == true_index),
                            wrong_unique=int(count == 1 and chosen != true_index), unresolved=int(count > 1),
                            true_rejected=int(true_index not in credible), credible_size=count)
                        for field, value in expected.items():
                            compare(row[field], value, 'actual classification '+field, 'actual_summary_error', atol=2e-8)
                    for field in saved['samples'][0]:
                        if field != 'posterior':
                            compare(saved[field], np.mean([v[field] for v in saved['samples']]), 'actual repeat summary '+field, 'actual_summary_error')
                    maxima['AC_equation'] = max(maxima['AC_equation'], physical['equation'])
                    maxima['power_balance'] = max(maxima['power_balance'], physical['balance'])
                    counts['actual_action_regime_clean_states'] += 1; counts['actual_noise_observations'] += R
                    actual_results[regime][key] = dict(success=True, noise_repeats=R, true_index=true_index,
                        AC_equation=physical['equation'], power_balance=physical['balance'])
                attempt('actual/'+regime+'/'+key, actual_check)

    legacy = record.get('legacy_joint_audit', {})
    counts['legacy_action_regime_slots'] = len(legacy)
    for identifier, comparison in legacy.items():
        regime, _, key = identifier.partition('__')
        for name, candidate in candidates.items():
            counts['legacy_candidate_slots'] += 1
            if name not in comparison.get('comparisons', {}):
                missing.append(dict(field='legacy/'+identifier+'/'+name,
                    reason=str(comparison.get('failures', {}).get(name, 'candidate saved state absent'))))
                continue
            rows = comparison['comparisons'][name]
            for mode in ('fixed', 'joint'):
                counts['legacy_fit_slots'] += 1
                row = rows.get(mode, {})
                if not row.get('success', mode == 'fixed'):
                    counts['recorded_legacy_fit_failures'] += 1
                if not row.get('state'):
                    missing.append(dict(field='legacy/'+identifier+'/'+name+'/'+mode,
                                        reason='complete fitted state unavailable'))
                    continue
                def legacy_check(candidate=candidate, row=row, identifier=identifier, regime=regime, key=key, mode=mode):
                    _require((regime, key) in first_observations, 'actual first observation replay required')
                    blocks = (('observed', observed, dm), ('future', first_observations[(regime, key)], fm))
                    saved_state_check(candidate, row, row['state'], blocks, 'legacy/'+identifier+'/'+candidate.name+'/'+mode)
                attempt('legacy/'+identifier+'/'+name+'/'+mode, legacy_check)

    attribution = {}
    if include_attribution and true_name in candidates and truths:
        for method in ('constant', 'square__mobile_bidir', 'bayes__mobile_bidir'):
            for regime, truth in truths.items():
                key = record['methods'][method]['action_key']; identifier = regime+'__'+key
                if identifier not in attribution:
                    value = attempt('attribution/'+identifier, lambda truth=truth, key=key: _attribution(
                        candidates[true_name], fits[true_name], record['true_lines'],
                        prediction['candidate_updates'][true_name]['future_latent_mean'], truth,
                        _array(record['actions'][key]['power_kw']), base_kw))
                    if value is not None:
                        attribution[identifier] = dict(value, methods=[])
                        counts['attribution_AC_batches'] += 8
                if identifier in attribution:
                    attribution[identifier]['methods'].append(method)
    return dict(success=not errors and not missing, spec=record.get('spec'), status=record.get('status'),
        counts=dict(counts), maxima=dict(maxima), errors=errors, missing=missing, actions=action_results,
        actual=actual_results, mean_error_attribution=attribution, seconds=perf_counter()-began,
        candidate_refits=0, optimizer_calls=0, selector_calls=0, case_generation_calls=0,
        scope='independent saved-state AC, conditioning, joint Gaussian integrals and complete EV path replay',
        probability_scope='declared fixed-trained-line local Gaussian; empirical coverage is a separate question')


def run(directory, output, *, phases=('development', 'evaluation'), include_attribution=True):
    directory, output = Path(directory), Path(output)
    if output.exists():
        raise FileExistsError(output)
    began = perf_counter(); frozen = io.read_json(directory/'freeze.json')
    snapshot = io.read_json(directory/'source_snapshot.json.gz')
    source_errors = []
    for name, expected in frozen['source_sha256'].items():
        try:
            payload = base64.b64decode(snapshot['source_base64'][name], validate=True)
            _require(sha256(payload).hexdigest() == expected, 'snapshot hash')
            path = HERE.parents[1]/name if name.startswith('rnj_wzzt_core/') else HERE/name
            _require(sha256(path.read_bytes()).hexdigest() == expected, 'current frozen science hash')
        except (KeyError, ValueError, OSError) as exc:
            source_errors.append(dict(source=name, error=str(exc)))
    rows, coverage = [], {}
    for phase in phases:
        paths = sorted((directory/phase).glob('history_*.json.gz'))
        expected = frozen['specifications'].get(phase, [])
        present = []
        for path in paths:
            record = io.read_json(path); present.append(record.get('spec'))
            try:
                _require(record.get('policy_id') == frozen['policy_id'], 'record/freeze policy identity')
                result = audit_record(record, include_attribution=include_attribution)
            except EXCEPTIONS as exc:
                result = dict(success=False, spec=record.get('spec'), status=record.get('status'), counts={}, maxima={},
                    errors=[dict(check='record', reason=type(exc).__name__+': '+str(exc))], missing=[])
            result.update(input_file=str(path.resolve()), input_sha256=sha256(path.read_bytes()).hexdigest())
            rows.append(result)
            print('BAYES_AUDIT', phase, path.name, result['success'], flush=True)
        present_ids = [io.identity(v, {}) for v in present]
        expected_ids = [io.identity(v, {}) for v in expected]
        coverage[phase] = dict(expected_histories=len(expected), present_histories=len(present),
            exact_coverage=len(present_ids) == len(set(present_ids)) and set(present_ids) == set(expected_ids),
            missing_specifications=[v for v in expected if io.identity(v, {}) not in present_ids])
    counts, maxima = Counter(), Counter()
    for row in rows:
        counts.update(row['counts'])
        for key, value in row['maxima'].items():
            maxima[key] = max(maxima[key], value)
    result = dict(schema='ev_bayes_small_independent_audit_v1', input_root=str(directory.resolve()),
        policy_id=frozen['policy_id'], tool_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
        science_source_count=len(frozen['source_sha256']), source_errors=source_errors,
        success=not source_errors and all(v['exact_coverage'] for v in coverage.values()) and all(r['success'] for r in rows),
        coverage=coverage, counts=dict(counts), maxima=dict(maxima), records=rows, seconds=perf_counter()-began,
        history_status_counts=dict(Counter(r['status'] for r in rows)),
        missing_count=sum(len(r['missing']) for r in rows), error_count=sum(len(r['errors']) for r in rows)+len(source_errors),
        no_new_fits_or_actions=True, synthetic_truth_used_posthoc_only=True)
    io.write_json(output, result, compressed=output.suffix == '.gz')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True); parser.add_argument('--output', required=True)
    parser.add_argument('--phase', nargs='+', choices=('development', 'evaluation'), default=['development', 'evaluation'])
    parser.add_argument('--skip-attribution', action='store_true')
    args = parser.parse_args()
    result = run(args.input, args.output, phases=tuple(args.phase), include_attribution=not args.skip_attribution)
    print('AUDIT_SAVED', result['success'], result['error_count'], result['missing_count'], flush=True)


if __name__ == '__main__':
    main()
