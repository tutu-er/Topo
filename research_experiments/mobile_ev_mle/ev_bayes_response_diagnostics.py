"""Offline response and ranking diagnostics; truth is used only after planning.

Saved actions are never changed.  The optional AC calls below replay fixed
parameters and inputs; they do not fit, generate observations or select plans.
Three predeclared plans receive eight-corner mean-error diagnostics by default.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from itertools import combinations, permutations, product
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.stats import norm

from .ev_bayes_response_analysis import read, serial, coverage


CHANNELS = ('p', 'q', 'v', 'master_p', 'master_q')
FACTORS = ('lines', 'PQ', 'source')
ATTR_METHODS = ('constant', 'square__mobile_bidir', 'joint_bc__mobile_bidir')
MASKS = tuple(product((0, 1), repeat=3))
REGIMES = ('regular', 'shift')
Z90 = float(norm.ppf(.95))


def _finite(value):
    value = np.asarray(value, float)
    if not np.isfinite(value).all(): raise ValueError('nonfinite saved numbers')
    return value


def _require(condition, message):
    if not condition: raise ValueError(message)


def channel_slices(T, n):
    result, offset = {}, 0
    for channel in CHANNELS:
        size = T*(n if channel in ('p', 'q', 'v') else 1)
        result[channel] = slice(offset, offset+size); offset += size
    return result


def _channel_shape(channel, T, n):
    return (T, n) if channel in ('p', 'q', 'v') else (T,)


def _scale(channel, base_kw):
    return (400., 'V_LL') if channel == 'v' else (base_kw, 'kvar' if channel in ('q', 'master_q') else 'kW')


def error_summary(predicted, actual, *, scale=1., units='pu'):
    p, a = _finite(predicted)*scale, _finite(actual)*scale
    _require(p.shape == a.shape, 'prediction/truth shape')
    error, amplitude = p-a, float(np.sqrt(np.mean(a*a)))
    return dict(RMSE=float(np.sqrt(np.mean(error*error))), bias=float(np.mean(error)),
        max_absolute_error=float(np.max(abs(error), initial=0.)), truth_RMS=amplitude,
        relative_RMSE=None if amplitude <= 1e-14 else float(np.sqrt(np.mean(error*error)))/amplitude,
        units=units)


def decompose_mse(errors):
    """All six orders average the reduction E(000)-E(111); signed effects remain.

    A bit equal to one replaces that one input by its physical truth.  Pair
    and triple terms are anchored at 000 and explicitly retain interactions.
    These are errors of one deterministic response, not decision-risk causes.
    """
    e = {tuple(map(int, k)): float(v) for k, v in errors.items()}
    _require(set(e) == set(MASKS) and np.isfinite(list(e.values())).all(), 'all eight finite MSE corners required')
    gain = {mask: e[(0, 0, 0)]-value for mask, value in e.items()}
    averaged = {f: 0. for f in FACTORS}; orders = []
    for order in permutations(range(3)):
        mask = [0, 0, 0]; previous = e[tuple(mask)]; steps = {}
        for i in order:
            mask[i] = 1; following = e[tuple(mask)]
            steps[FACTORS[i]] = previous-following
            averaged[FACTORS[i]] += (previous-following)/6
            previous = following
        orders.append(dict(order=[FACTORS[i] for i in order], reductions=steps))
    singles = {FACTORS[i]: gain[tuple(int(j == i) for j in range(3))] for i in range(3)}
    interactions = {}
    for i, j in combinations(range(3), 2):
        interactions[FACTORS[i]+'*'+FACTORS[j]] = gain[tuple(int(k in (i, j)) for k in range(3))]-singles[FACTORS[i]]-singles[FACTORS[j]]
    triple = gain[(1, 1, 1)]-sum(singles.values())-sum(interactions.values())
    return dict(corner_MSE={''.join(map(str, k)): e[k] for k in MASKS},
        nominal_MSE=e[(0, 0, 0)], all_truth_MSE=e[(1, 1, 1)],
        six_order_average_reduction=averaged, anchored_single_reduction=singles,
        anchored_pair_interaction=interactions, anchored_triple_interaction=triple,
        sequential_orders=orders,
        sum_identity_error=abs(sum(averaged.values())-(e[(0, 0, 0)]-e[(1, 1, 1)])),
        scope='fixed-response mean MSE reduction; signed interactions retained; not causal decision-risk attribution')


def factorial_errors(curves, *, base_kw=20.):
    """Physical outputs at all corners, each containing full and increment."""
    target = curves[(1, 1, 1)]; rows = []
    for response in ('full', 'increment'):
        for channel in CHANNELS:
            scale, units = _scale(channel, base_kw)
            errors = {mask: float(np.mean(((np.asarray(row[response][channel])-np.asarray(target[response][channel]))*scale)**2))
                      for mask, row in curves.items()}
            rows.append(dict(response=response, channel=channel, units=units+'^2',
                **decompose_mse(errors), amplitude=error_summary(curves[(0, 0, 0)][response][channel],
                    target[response][channel], scale=scale, units=units)))
    return rows


def _ranks(values, tolerance):
    """Transitive adjacent tie groups, with explicit finite resolution."""
    values = _finite(values); order = np.argsort(values, kind='stable')
    groups, last = [], None
    for i in order:
        if last is None or values[i]-last > tolerance: groups.append([])
        groups[-1].append(int(i)); last = values[i]
    ranks = np.empty(len(values))
    offset = 0
    for group in groups:
        ranks[group] = offset+(len(group)+1)/2; offset += len(group)
    return ranks


def rank_agreement(predicted, actual, *, predicted_tolerance=1e-10, actual_tolerance=1e-12):
    p, a = _finite(predicted), _finite(actual)
    _require(p.shape == a.shape and p.ndim == 1, 'same one-dimensional ranking vectors required')
    rp, ra = _ranks(p, predicted_tolerance), _ranks(a, actual_tolerance)
    rho = None if len(p) < 2 or np.ptp(rp) == 0 or np.ptp(ra) == 0 else float(np.corrcoef(rp, ra)[0, 1])
    counts = Counter()
    for i, j in combinations(range(len(p)), 2):
        dp, da = p[i]-p[j], a[i]-a[j]
        if abs(dp) <= predicted_tolerance: counts['predicted_tie'] += 1
        elif abs(da) <= actual_tolerance: counts['actual_tie'] += 1
        else: counts['concordant' if dp*da > 0 else 'discordant'] += 1
    return dict(Spearman=rho, actions=len(p), pairs=len(p)*(len(p)-1)//2,
        pair_counts=dict(counts), predicted_tolerance=predicted_tolerance,
        actual_tolerance=actual_tolerance, inference='descriptive ranking of saved actions only')


def selection_regret(predicted_risk, actual_loss, *, tolerance=1e-10):
    """Tie set is retained; no truth-favorable tie breaker is introduced."""
    p, a = _finite(predicted_risk), _finite(actual_loss)
    _require(p.ndim == a.ndim == 1 and len(p) == len(a) and len(p), 'nonempty aligned risks required')
    selected = np.flatnonzero(p <= np.min(p)+tolerance)
    regrets = a[selected]-np.min(a)
    return dict(predicted_best_indices=selected, predicted_best_tie_count=len(selected),
        empirical_best_loss=float(np.min(a)), tie_regret_range=[float(np.min(regrets)), float(np.max(regrets))],
        inference='saved-action empirical regret; actual minimum is estimated and selection-biased')


def paired_error_differences(correct, *, z=1.96):
    """Within-history paired noise uncertainty, separate from history intervals.

    Zero discordant draws provide no evidence for an exact equality, so these
    pairs are explicitly unresolved even when their sample SE is zero.
    """
    c = _finite(correct)
    _require(c.ndim == 2 and c.shape[1] > 1 and np.isin(c, [0., 1.]).all(), 'binary correct[action,noise] required')
    rows = []
    for i, j in combinations(range(len(c)), 2):
        d = c[j]-c[i]  # error_i - error_j
        difference, se = float(d.mean()), float(d.std(ddof=1)/np.sqrt(c.shape[1]))
        interval = [max(-1., difference-z*se), min(1., difference+z*se)]
        discordant = int(np.count_nonzero(d)); unresolved = not discordant or interval[0] <= 0 <= interval[1]
        rows.append(dict(left=i, right=j, mean_error_left_minus_right=difference,
            paired_noise_SE=se, approximate_interval95=interval, discordant_noise_draws=discordant,
            classification='noise_indeterminate' if unresolved else 'left_lower_error' if interval[1] < 0 else 'right_lower_error',
            conditional_draws=c.shape[1], independent_histories=1,
            interval_scope='descriptive paired normal approximation over meter noise, conditional on one saved history'))
    return rows


def clean_geometry(bank, clean, true_index):
    """Geometry on true physical noiseless meters; density gaps include history.

    Positive signed gap favors the true candidate.  The minimum over false
    candidates includes their log determinants, unlike Euclidean distances.
    """
    means = _finite(bank['means']); covariance = _finite(bank['covariances'])
    _require(means.shape[0] == covariance.shape[0] == 1, 'one joint Gaussian stratum required')
    means, covariance = means[0], covariance[0]; clean = _finite(clean)
    _require(means.shape[1:] == clean.shape and covariance.shape == (len(means), len(clean), len(clean)), 'Gaussian dimensions')
    prior = _finite(bank['log_prior']) if 'log_prior' in bank else np.log(_finite(bank['prior']))
    squared = np.sum((means-clean)**2, axis=1); quadratic, logpdf = [], []
    for mean, cov in zip(means, covariance):
        factor = cho_factor(cov, lower=True, check_finite=False); residual = clean-mean
        q = float(residual@cho_solve(factor, residual, check_finite=False)); quadratic.append(q)
        logpdf.append(-.5*(q+2*np.log(np.diag(factor[0])).sum()+len(clean)*np.log(2*np.pi)))
    logpdf = np.asarray(logpdf); score = logpdf+prior
    others = [i for i in range(len(means)) if i != true_index]
    _require(others, 'at least two candidates for geometry')
    return dict(true_index=int(true_index), pair_rows=[dict(competitor=i,
        whitened_square_gap=float(squared[i]-squared[true_index]),
        Mahalanobis_square_gap=float(quadratic[i]-quadratic[true_index]),
        log_density_gap=float(logpdf[true_index]-logpdf[i]),
        log_posterior_score_gap=float(score[true_index]-score[i])) for i in others],
        minimum_whitened_square_gap=float(np.min(squared[others]-squared[true_index])),
        minimum_log_density_gap=float(np.min(logpdf[true_index]-logpdf[others])),
        minimum_log_posterior_score_gap=float(np.min(score[true_index]-score[others])),
        scope='posthoc geometry at one clean physical future, not an expected topology-error probability')


def increment_diagnostics(saved, actual, meter, T, n, base_kw):
    mean, variance, target = _finite(saved['mean']), _finite(saved['variance']), _finite(actual)
    _require(mean.shape == variance.shape == target.shape == ((3*n+2)*T,), 'increment vector shape')
    _require(np.min(variance) >= -1e-10, 'nonnegative increment variance')
    rows = []
    for channel, slc in channel_slices(T, n).items():
        shape = _channel_shape(channel, T, n)
        sigma = np.broadcast_to(_finite(meter[channel+'_sigma']), shape).ravel()
        scale, units = _scale(channel, base_kw)
        sd = np.sqrt(np.maximum(variance[slc], 0.)); error = abs(mean[slc]-target[slc])
        zero = sd <= 1e-14; covered = error <= Z90*sd+1e-8
        rows.append(dict(channel=channel, **error_summary(mean[slc]*sigma, target[slc]*sigma, scale=scale, units=units),
            interval90_point_coverage=float(np.mean(covered)), coordinates=len(error),
            zero_variance_coordinates=int(zero.sum()),
            zero_variance_miss_count=int(np.count_nonzero(zero & ~covered)),
            average_interval90_halfwidth=float(np.mean(Z90*sd*sigma*scale)),
            variance_identity_max_error=float(np.max(abs(variance[slc]-sum(_finite(saved[k])[slc]
                for k in ('line_variance', 'state_variance', 'cross_variance'))), initial=0.)),
            scope='clean shared-background EV response interval; meter noise excluded; coordinates are not independent trials'))
    return rows


def level_diagnostics(bank, actual, meter, T, n, base_kw, true_index):
    """Clean mean error using saved full predictive variance, including meter I."""
    mean = _finite(bank['means'])[0, true_index]
    variance = np.diag(_finite(bank['covariances'])[0, true_index])
    actual = _finite(actual)
    _require(mean.shape == variance.shape == actual.shape == ((3*n+2)*T,), 'level vector shape')
    _require(np.min(variance) > 0, 'positive full predictive variance')
    rows = []
    for channel, slc in channel_slices(T, n).items():
        sigma = np.broadcast_to(_finite(meter[channel+'_sigma']), _channel_shape(channel, T, n)).ravel()
        scale, units = _scale(channel, base_kw)
        sd = np.sqrt(variance[slc])
        rows.append(dict(channel=channel, **error_summary(mean[slc]*sigma, actual[slc]*sigma, scale=scale, units=units),
            clean_point_interval90_coverage=float(np.mean(abs(mean[slc]-actual[slc]) <= Z90*sd+1e-8)),
            coordinates=len(sd), average_interval90_halfwidth=float(np.mean(Z90*sd*sigma*scale)),
            scope='true-candidate full-level density interval including meter noise; coverage of one clean future, not repeated-sample calibration'))
    return rows


def _predict(candidate, r, x, latent, root):
    # Independent radial sweep already used by the frozen physical audit.
    from .ev_bayes_small_audit import _physics
    return _physics(candidate, r, x, latent, root)


def _candidate(fit, n):
    from .likelihood import RadialCandidate
    return RadialCandidate.from_supports(fit.get('candidate', fit.get('name')), n,
        [tuple(s) for s in fit['supports'] if 1 < len(s) < n])


def _line_center(update, m):
    eta = _finite(update['TV_posterior_future']['mean'])[:2*m].reshape(m, 2)
    return np.exp(eta[:, 0]), np.exp(eta.sum(axis=1))


def _truth_latent(value):
    return np.column_stack([_finite(value[k]) for k in ('p', 'q', 'source_p', 'source_q')])


def _forward_factorial(candidate, r, x, lines_true, predicted, actual, root, power, base_kw, predictor, cache):
    curves, failures = {}, []
    for mask in MASKS:
        try:
            rr, xx = lines_true if mask[0] else (r, x)
            latent = predicted.copy()
            if mask[1]: latent[:, :2*candidate.n] = actual[:, :2*candidate.n]
            if mask[2]: latent[:, -2:] = actual[:, -2:]
            def run(injection):
                z = latent.copy(); z[:, :candidate.n] += injection/base_kw
                identity = (np.r_[rr, xx].astype('<f8').tobytes(), z.astype('<f8').tobytes(), np.asarray(root).astype('<f8').tobytes())
                if identity not in cache:
                    cache[identity] = predictor(candidate, rr, xx, z, root)
                return cache[identity]
            full, zero = run(power), run(np.zeros_like(power))
            curves[mask] = dict(full={k: full[k] for k in CHANNELS},
                increment={k: np.asarray(full[k])-np.asarray(zero[k]) for k in CHANNELS})
        except Exception as exc:
            failures.append(dict(corner=''.join(map(str, mask)), failure=type(exc).__name__+': '+str(exc)))
    return curves, failures


def _ranking(record, regime, true_index):
    keys, risks, actual, geometries, errors = [], [], [], [], []
    expected = len(record.get('actions', {})); noise_seed, R = None, None
    for key, row in record.get('actions', {}).items():
        try:
            predicted = row['predictions']['joint']; value = row['actual'][regime]
            correct = _finite(value['models']['joint']['samples']['correct'])
            _require(correct.ndim == 1 and len(correct) >= 2, 'at least two paired noise draws required')
            _require(np.isin(correct, [0, 1]).all(), 'binary correctness')
            if R is None: R, noise_seed = len(correct), value['noise_seed']
            _require(R == len(correct) and noise_seed == value['noise_seed'], 'common meter-noise stream identity')
            risk = float(predicted['independent_score']['risk']); _require(np.isfinite(risk), 'finite predicted risk')
            geometry = clean_geometry(predicted['forecast']['bank'], value['clean_whitened'], true_index)
            keys.append(key); risks.append(risk); actual.append(correct); geometries.append(geometry)
        except Exception as exc:
            errors.append(dict(action_key=key, failure=type(exc).__name__+': '+str(exc)))
    result = dict(regime=regime, expected_actions=expected, available_actions=len(keys),
        action_keys=keys, missing_or_failed_actions=errors, conditional_noise_draws=R, independent_histories=1)
    if not keys: return result
    loss = 1-np.asarray(actual).mean(axis=1)
    result.update(actions=[dict(action_key=k, predicted_risk=p, empirical_error=float(a), clean_geometry=g)
        for k, p, a, g in zip(keys, risks, loss, geometries)],
        actual_error_ranking=rank_agreement(risks, loss),
        selection_regret=selection_regret(risks, loss),
        paired_actual_error=paired_error_differences(actual))
    empirical_best = np.flatnonzero(loss <= np.min(loss)+1e-12)
    regret = result['selection_regret']
    regret['empirical_best_indices'] = empirical_best
    regret['paired_comparisons'] = []
    for i in regret['predicted_best_indices']:
        for j in empirical_best:
            if i == j:
                comparison = dict(mean_error_left_minus_right=0., paired_noise_SE=0.,
                    classification='same_electrical_action', conditional_draws=R, independent_histories=1)
            else:
                comparison = paired_error_differences([actual[i], actual[j]])[0]
            regret['paired_comparisons'].append(dict(predicted_selected_action=keys[i],
                empirical_best_action=keys[j], **comparison))
    for field in ('minimum_whitened_square_gap', 'minimum_log_density_gap', 'minimum_log_posterior_score_gap'):
        geometry_loss = -np.asarray([g[field] for g in geometries])
        result[field] = dict(agreement=rank_agreement(risks, geometry_loss),
            selection_regret=selection_regret(risks, geometry_loss),
            positive_gap_favors_truth=True, selection_scope='posthoc among saved unique actions only')
    result['ranking_scope'] = 'Bayes risk averages candidate prior; actual error is conditional on one physical truth. Differences can reflect different objectives, mean/covariance approximation and finite MC.'
    return result


def diagnose_record(record, *, all_factorial=False, predictor=None):
    began = perf_counter(); predictor = _predict if predictor is None else predictor
    result = dict(spec=record.get('spec'), status=record.get('status'), posthoc_truth_only=True,
        failures=[], expected_predeclared_methods=list(ATTR_METHODS), increment_rows=[], level_rows=[],
        factorial=[], forecast_errors=[], line_errors=[], rankings=[], methods=[],
        new_fits=0, new_plans=0, new_observations=0, source_interface=1)
    try:
        name = record['true_name']; fits = record['training_result']['fits']
        fit = next(v for v in fits if v.get('candidate', v.get('name')) == name)
        joint = record['prediction']['joint']; update = joint['candidate_updates'][name]
        true_index = list(record['candidate_names']).index(name)
        T = len(record['methods']['none']['power_kw']); n = len(record['methods']['none']['power_kw'][0])
        base_kw = float(joint['base_kw']); candidate = _candidate(fit, n)
        meter = record['future_meter']; predicted = _finite(update['future_latent_mean'])
        _require(predicted.shape == (T, 2*n+2), 'future background shape')
        true_lines = tuple(_finite(record['true_lines'][k]) for k in ('r', 'x'))
        centers = {'train_ML': tuple(_finite(fit[k]) for k in ('r', 'x')),
                   'bounded_posterior': _line_center(update, candidate.m)}
        for label, (r, x) in centers.items():
            result['line_errors'].append(dict(center=label, **{k: error_summary(a, b, units='pu') | dict(relative_L2=float(np.linalg.norm(a-b)/np.linalg.norm(b)))
                for k, a, b in zip(('R', 'X'), (r, x), true_lines)}))
        action_aliases = defaultdict(list)
        for method, row in record.get('methods', {}).items(): action_aliases[row.get('action_key')].append(method)
        selected = set(record['actions']) if all_factorial else {
            record.get('methods', {}).get(m, {}).get('action_key') for m in ATTR_METHODS}
        selected.discard(None)
        result['factorial_scope'] = 'all saved unique actions' if all_factorial else 'constant/SQ_MD/J_MD; duplicate electrical actions replayed once'
        for method in ATTR_METHODS:
            row = record.get('methods', {}).get(method)
            result['methods'].append(dict(method=method, action_key=None if row is None else row.get('action_key'), available=row is not None))
            if row is None: result['failures'].append(dict(method=method, stage='predeclared_method', failure='missing method'))
        result['expected_action_count'] = len(record['actions']); result['expected_factorial_action_count'] = len(selected)
        cache = {}
        for regime in record.get('config', {}).get('regimes', REGIMES):
            truth = record['true_future_inputs_evaluation_only'][regime]; actual = _truth_latent(truth)
            root = _finite(truth['root_v'])
            ranking = _ranking(record, regime, true_index)
            result['rankings'].append(ranking)
            result['failures'].extend(dict(stage='action_ranking', regime=regime, **f) for f in ranking['missing_or_failed_actions'])
            for model in ('fixed', 'joint'):
                try:
                    background = _finite(record['prediction'][model]['candidate_updates'][name]['future_latent_mean'])
                    result['forecast_errors'].append(dict(regime=regime, model=model,
                        **{k: error_summary(background[:, slc], actual[:, slc], scale=base_kw, units=unit)
                           for k, slc, unit in (('P', slice(0, n), 'kW'), ('Q', slice(n, 2*n), 'kvar'),
                               ('source_P', slice(-2, -1), 'kW'), ('source_Q', slice(-1, None), 'kvar'))}))
                except Exception as exc:
                    result['failures'].append(dict(stage='forecast_error', regime=regime, model=model, failure=type(exc).__name__+': '+str(exc)))
            for key, action in record['actions'].items():
                try:
                    bank = action['predictions']['joint']['forecast']['bank']
                    result['level_rows'].extend(dict(action_key=key, methods=action_aliases[key], regime=regime, **r)
                        for r in level_diagnostics(bank, action['actual'][regime]['clean_whitened'], meter, T, n, base_kw, true_index))
                except Exception as exc:
                    result['failures'].append(dict(stage='saved_level', action_key=key, regime=regime, failure=type(exc).__name__+': '+str(exc)))
                try:
                    saved = action['response_increment']
                    _require(not saved['meter_noise_included'] and saved['shared_line_and_background'], 'shared physical increment covariance declaration')
                    _require(saved['true_index'] == true_index, 'increment true index identity')
                    row = saved['predictions'][true_index]
                    _require(row['candidate'] == name, 'increment candidate identity')
                    increment_rows = increment_diagnostics(row, saved['actual'][regime], meter, T, n, base_kw)
                    result['increment_rows'].extend(dict(action_key=key, methods=action_aliases[key], regime=regime, **r) for r in increment_rows)
                except Exception as exc:
                    result['failures'].append(dict(stage='saved_increment', action_key=key, regime=regime, failure=type(exc).__name__+': '+str(exc)))
                if key not in selected: continue
                power = _finite(action['power_kw'])
                for center, (r, x) in centers.items():
                    curves, failures = _forward_factorial(candidate, r, x, true_lines, predicted, actual, root, power, base_kw, predictor, cache)
                    entry = dict(action_key=key, methods=action_aliases[key], regime=regime, line_center=center,
                        corners_expected=8, corners_available=len(curves), failures=failures,
                        common_background='joint posterior future mean held common between line centers',
                        root='declared fixed actual root (known independently)', source_interface=1)
                    if not failures:
                        entry['errors'] = factorial_errors(curves, base_kw=base_kw)
                        # Independently verify all-truth replay against saved physical clean meters.
                        from .ev_bayes_small_audit import _whiten
                        replay = _whiten(curves[(1, 1, 1)]['full'], meter)
                        entry['saved_clean_replay_max_error'] = float(np.max(abs(replay-_finite(action['actual'][regime]['clean_whitened']))))
                    result['factorial'].append(entry)
                    result['failures'].extend(dict(stage='factorial', action_key=key, line_center=center, regime=regime, **f) for f in failures)
        result['new_AC_batches'] = len(cache)
        result['new_AC_states'] = len(cache)*T
        result['denominators'] = dict(independent_histories=1, unique_saved_actions=len(record['actions']),
            factorial_slots_expected=len(selected)*len(centers)*len(record.get('config', {}).get('regimes', REGIMES)),
            factorial_slots_available=len(result['factorial']),
            factorial_complete_slots=sum(v['corners_available'] == 8 for v in result['factorial']),
            increment_channel_slots_expected=len(record['actions'])*5*len(record.get('config', {}).get('regimes', REGIMES)),
            increment_channel_slots_available=len(result['increment_rows']),
            level_channel_slots_expected=len(record['actions'])*5*len(record.get('config', {}).get('regimes', REGIMES)),
            level_channel_slots_available=len(result['level_rows']))
    except Exception as exc:
        result['failures'].append(dict(stage='history_inputs', failure=type(exc).__name__+': '+str(exc)))
    result['success'] = not result['failures']; result['seconds'] = perf_counter()-began
    return result


def markdown(result):
    records = result['records']; lines = ['# 保存动作的响应与排序诊断', '',
        f"阶段：{result['phase']}；记录{len(records)}/{result['coverage']['expected_histories']}；诊断失败项{sum(len(r['failures']) for r in records)}。", '',
        '真值仅用于离线诊断。动作和模型参数均不重新拟合；完整AC复算只替换指定输入。',
        'full是全部未来读数的水平误差；increment是相同线路和背景下，有EV减去无EV的响应误差。P/Q已含EV一次。',
        '八角中的三位依次表示线路、负荷P/Q、同接口源，1表示换成真值。六种替换次序的平均贡献之和等于000到111的MSE下降；单项可能为负，交互另存。该分解针对均值误差，不能解释为辨识错误率的因果分解。',
        '两组线路中心共用joint预测负荷和源；fixed/joint预测误差单列。电压单位为400 V线电压基准，功率使用记录中的kW基准。', '',
        '|历史|情景|动作数|预测风险与实际错误Spearman|有差异但噪声未分清的动作对|预测最佳并列数|经验后悔范围|',
        '|---:|---|---:|---:|---:|---:|---|']
    for row in records:
        for rank in row['rankings']:
            pairs = rank.get('paired_actual_error', []); rho = rank.get('actual_error_ranking', {}).get('Spearman')
            regret = rank.get('selection_regret', {})
            lines.append('|'+str(row.get('spec', {}).get('index'))+'|'+rank['regime']+'|'+str(rank['available_actions'])+'|'+
                ('—' if rho is None else f'{rho:.3f}')+'|'+str(sum(p['classification'] == 'noise_indeterminate' for p in pairs))+'|'+
                str(regret.get('predicted_best_tie_count', '—'))+'|'+str(regret.get('tie_regret_range', '—'))+'|')
    lines += ['', '排序只在本次已保存的不同电气动作之间计算。32次计量噪声来自同一历史，不能增加独立历史数。配对区间描述该历史条件下的噪声差异；全零差异保持未分清。预测风险是候选先验平均，实际错误率条件于单一真候选，两者的目标本身也有差异。',
        'clean几何是在真实无噪量测上，真候选相对于最强竞争者的有符号差。另存白化平方距离、含协方差行列式的密度差、加历史先验的分数差；正值支持真候选。',
        '所有缺失方法、失败动作和失败AC角点均保留。JSON包含完整分母、各通道MSE、交互、覆盖、幅值及逐对结果。', '']
    return '\n'.join(lines)


def run(directory, phase, output, *, indices=None, all_factorial=False):
    directory, output = Path(directory), Path(output); began = perf_counter()
    if (output/'diagnostics.json').exists(): raise FileExistsError(output/'diagnostics.json')
    freeze = read(directory/'freeze.json'); expected = freeze['specifications'][phase]
    if indices is not None:
        _require(indices <= {int(s['index']) for s in expected}, 'requested indices outside the frozen specification')
        expected = [s for s in expected if int(s['index']) in indices]
    records, present, inputs = [], [], []
    for spec in expected:
        path = directory/phase/('history_'+str(spec['index']).zfill(3)+'.json.gz')
        if not path.exists(): continue
        raw = read(path)
        _require(raw.get('policy_id') == freeze['policy_id'], 'record/freeze policy identity')
        records.append(diagnose_record(raw, all_factorial=all_factorial)); present.append(raw['spec'])
        inputs.append(dict(file=str(path.resolve()), sha256=sha256(path.read_bytes()).hexdigest()))
        del raw
    result = dict(schema='ev_bayes_response_offline_diagnostics_v1', phase=phase,
        policy_id=freeze['policy_id'], coverage=coverage(expected, present), records=records,
        selected_indices=None if indices is None else sorted(indices),
        input_files=inputs, tool_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
        source_freeze_sha256=sha256((directory/'freeze.json').read_bytes()).hexdigest(),
        posthoc_truth_only=True, new_fits=0, new_plans=0, new_observations=0,
        new_AC_batches=sum(r.get('new_AC_batches', 0) for r in records), seconds=perf_counter()-began)
    output.mkdir(parents=True, exist_ok=True)
    (output/'diagnostics.json').write_text(json.dumps(serial(result), ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)+'\n', encoding='utf-8')
    (output/'diagnostics.md').write_text(markdown(result), encoding='utf-8')
    print('RESPONSE_DIAGNOSTICS_SAVED', phase, len(records), result['coverage'], flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True); parser.add_argument('--output', required=True)
    parser.add_argument('--phase', choices=('development', 'calibration', 'evaluation'), default='development')
    parser.add_argument('--indices', help='Comma-separated predeclared history indices; default all.')
    parser.add_argument('--all-factorial', action='store_true', help='Replay all saved actions instead of the three predeclared plans.')
    args = parser.parse_args(); indices = None if args.indices is None else set(map(int, args.indices.split(',')))
    if indices is not None and (not indices or min(indices) < 0): parser.error('nonnegative indices required')
    run(args.input, args.phase, args.output, indices=indices, all_factorial=args.all_factorial)


if __name__ == '__main__': main()
