"""Fixed-plan, history-weighted calibration and paired inference summaries."""
from __future__ import annotations

import argparse
from copy import deepcopy
import gzip
from hashlib import sha256
import json
from pathlib import Path
from collections import defaultdict

import numpy as np

from .ev_bayes_small_probability import gaussian_posterior


MULTIPLIERS = (1., 2., 4., 8., 16., 32.)
MEAN_MCSE_LIMIT = .2
COVARIANCE_MCSE_LIMIT = .05
BOUNDED_ERROR_METRICS = frozenset(('error', 'clean_error', 'false_rejection',
                                 'wrong_decisive'))


def usable_banks(action):
    propagation = action.get('propagation', {})
    banks = dict(propagation.get('banks', {}))
    rows = propagation.get('rows', [])
    # Raw banks remain in the evidence; imprecise correction is not scored.
    precise = all(r.get('status') == 'ok'
        and r.get('mean_MCSE_whitened_max', np.inf) <= MEAN_MCSE_LIMIT
        and r.get('covariance_MCSE_relative_fro', np.inf) <= COVARIANCE_MCSE_LIMIT
        for r in rows)
    if not precise:
        banks = {k: v for k, v in banks.items() if k == 'G00'}
    return banks


def load(path):
    path = Path(path)
    return json.loads(gzip.decompress(path.read_bytes()) if path.suffix == '.gz'
                      else path.read_text(encoding='utf-8'))


def _status_ledger(value, prefix=''):
    """Keep nested status/failure fields without retaining numerical tensors."""
    result = {}
    if isinstance(value, dict):
        for key, child in value.items():
            path = prefix+'.'+str(key) if prefix else str(key)
            if (key in ('status', 'success', 'complete', 'error', 'usable',
                        'available', 'precision_met', 'all_complete')
                    or 'fail' in str(key).lower()):
                result[path] = deepcopy(child)
            if isinstance(child, dict):
                result.update(_status_ledger(child, path))
            elif isinstance(child, (list, tuple)) and child and isinstance(child[0], dict):
                for index, item in enumerate(child):
                    result.update(_status_ledger(item, path+'.'+str(index)))
    return result


def compact_record(record):
    """Lossless for this module's statistics, not a replacement for raw audits.

    Raw files stay unchanged. Large historical posterior tensors, optimizer
    states and AC samples are unnecessary for measurements/calibrate/summarize.
    Predictive bank numbers are retained exactly as float64, with identical
    arrays shared read-only across the factorial ablations. All nested status
    and failure fields are retained separately for inspection.
    """
    fields = ('spec', 'config', 'candidate_names', 'true_name', 'true_index',
              'status', 'failures', 'n_independent_histories', 'costs',
              'source_unchanged_during_run', 'code_initial_sha', 'code_end_sha')
    compact = {key: deepcopy(record[key]) for key in fields if key in record}
    compact['plans'] = {label: {'action_key': plan.get('action_key')}
                       for label, plan in record.get('plans', {}).items()}
    compact['status_audit'] = _status_ledger(record)
    compact['compact_statistics_version'] = 'ac_repair_statistics_v1'
    pool = {}
    def array(value):
        numeric = np.array(value, dtype=np.float64, copy=True)
        key = (numeric.shape, sha256(numeric.tobytes()).digest())
        if key in pool and np.array_equal(pool[key], numeric, equal_nan=True):
            return pool[key]
        numeric.flags.writeable = False
        pool[key] = numeric
        return numeric
    def fit(value):
        result = {key: deepcopy(value[key]) for key in
                  ('complete', 'failures', 'keep_all_on_failure', 'retained_on_failure')
                  if key in value}
        result['losses'] = deepcopy(value.get('losses', {}))
        result['comparisons'] = {name: {route: {'success': row.get('success')}
            for route, row in comparison.items() if route in ('fixed', 'joint')}
            for name, comparison in value.get('comparisons', {}).items()}
        return result
    actions = {}
    for key, action in record.get('actions', {}).items():
        propagated = action.get('propagation', {})
        banks = {name: {field: array(bank[field]) for field in
                       ('means', 'covariances', 'log_prior')}
                 for name, bank in propagated.get('banks', {}).items()}
        row_fields = ('candidate', 'status', 'mean_MCSE_whitened_max',
                      'covariance_MCSE_relative_fro', 'violation_fractions',
                      'discarded_draws')
        rows = [{field: deepcopy(row[field]) for field in row_fields if field in row}
                for row in propagated.get('rows', [])]
        result = dict(methods=deepcopy(action.get('methods', [])),
            failures=deepcopy(action.get('failures', {})),
            propagation=dict(status=propagated.get('status'), banks=banks, rows=rows),
            regimes={})
        for regime, actual in action.get('regimes', {}).items():
            actual_result = {field: array(actual[field]) for field in ('values', 'clean')
                             if field in actual}
            actual_result['refits'] = [fit(value) for value in actual.get('refits', [])]
            if 'clean_refit' in actual:
                actual_result['clean_refit'] = fit(actual['clean_refit'])
            result['regimes'][regime] = actual_result
        actions[key] = result
    compact['actions'] = actions
    return compact


def load_records(paths):
    """Release each expanded raw history before loading the next one."""
    result = []
    for path in paths:
        result.append(compact_record(load(path)))
    return result


def write(path, record):
    Path(path).write_text(json.dumps(record, ensure_ascii=False, indent=2,
        allow_nan=False)+'\n', encoding='utf-8', newline='\n')


def scaled_bank(bank, a=1.):
    """Scale latent predictive variance; retain independent meter variance I."""
    mean = np.asarray(bank['means'], float)
    cov = np.asarray(bank['covariances'], float)
    if mean.ndim != 3 or mean.shape[0] != 1:
        raise ValueError('one fixed-plan scenario required')
    if cov.shape != (*mean.shape[:2], mean.shape[-1], mean.shape[-1]):
        raise ValueError('mean/covariance shape mismatch')
    if a < 1 or not np.isfinite(a):
        raise ValueError('declared calibration multiplier must be finite >= 1')
    eye = np.eye(mean.shape[-1])
    return mean[0], eye+a*(cov[0]-eye), np.asarray(bank['log_prior'], float)


def scores(bank, values, true_index, *, a=1., mass=.9):
    """Normalized model probabilities plus empirical labelled losses."""
    means, covs, prior = scaled_bank(bank, a)
    y = np.atleast_2d(np.asarray(values, float))
    result = gaussian_posterior(y, means, covs, log_prior=prior)
    posterior = result['posterior']
    chosen = np.argmax(posterior, axis=-1)
    order = np.argsort(-posterior, axis=-1, kind='stable')
    kept = np.zeros_like(posterior, dtype=bool)
    for r, ids in enumerate(order):
        count = min(len(ids), int(np.searchsorted(np.cumsum(posterior[r, ids]),
                                                 mass, side='left'))+1)
        kept[r, ids[:count]] = True
    size = kept.sum(-1)
    onehot = np.eye(len(means))[true_index]
    total_score = result['log_likelihood']+prior
    others = np.arange(len(means)) != true_index
    margin = total_score[:, true_index]-total_score[:, others].max(-1)
    diag = np.diagonal(covs[true_index])
    covered = abs(y-means[true_index]) <= 1.6448536269514722*np.sqrt(diag)
    return dict(error=float(np.mean(chosen != true_index)),
        false_rejection=float(np.mean(~kept[:, true_index])),
        unresolved=float(np.mean(size != 1)),
        correct_decisive=float(np.mean((size == 1) & (chosen == true_index))),
        wrong_decisive=float(np.mean((size == 1) & (chosen != true_index))),
        mean_set_size=float(size.mean()),
        true_probability=float(posterior[:, true_index].mean()),
        true_predictive_NLL=float(-result['log_likelihood'][:, true_index].mean()),
        classification_NLL=float(-result['log_posterior'][:, true_index].mean()),
        brier=float(np.mean(np.sum((posterior-onehot)**2, axis=-1))),
        coverage90=float(covered.mean()), clean_or_noisy_margin=margin.tolist(),
        predicted=chosen.tolist())


def _actions(record):
    for label, plan in record.get('plans', {}).items():
        key = plan.get('action_key')
        action = record.get('actions', {}).get(key)
        if action is not None:
            yield label, action


def calibrate(records):
    """One multiplier, equal independent-history and prescribed-task weights."""
    rows = []
    for record in records:
        names = record.get('candidate_names', [])
        true = record.get('true_name')
        if true not in names:
            continue
        losses = {str(a): [] for a in MULTIPLIERS}
        complete = len(record.get('plans', {})) == 3
        for _, action in _actions(record):
            bank = usable_banks(action).get('G11')
            if bank is None:
                complete = False
                continue
            for regime in ('regular', 'shift'):
                actual = action.get('regimes', {}).get(regime, {})
                values = actual.get('values')
                if values is None:
                    complete = False
                    continue
                for a in MULTIPLIERS:
                    losses[str(a)].append(scores(bank, values, names.index(true),
                                                  a=a)['true_predictive_NLL'])
        # Six tasks = three prescribed methods times two future regimes.
        complete &= all(len(v) == 6 for v in losses.values())
        rows.append(dict(index=record['spec']['index'], complete=bool(complete),
            losses={a: float(np.mean(v)) if v else None for a, v in losses.items()}))
    valid = [r for r in rows if r['complete']]
    objective = {str(a): float(np.mean([r['losses'][str(a)] for r in valid]))
                 if valid else None for a in MULTIPLIERS}
    chosen = min(MULTIPLIERS, key=lambda a: objective[str(a)]) if valid else None
    thresholds = {}
    for route in ('fixed', 'joint'):
        maxima = []
        for record in records:
            values = []
            expected_methods = {'constant', 'SQMD', 'BCMD'}
            complete = set(record.get('plans', {})) == expected_methods
            names, true = record.get('candidate_names', []), record.get('true_name')
            task_count = 0
            expected_repeats = int(record.get('config', {}).get('noise_repeats', 1))
            for _, action in _actions(record):
                for regime in ('regular', 'shift'):
                    refits = action.get('regimes', {}).get(regime, {}).get('refits', [])
                    task_count += 1
                    if len(refits) != expected_repeats:
                        complete = False
                    for fit in refits:
                        loss = fit.get('losses', {}).get(route, {})
                        comparison = fit.get('comparisons', {})
                        usable = (set(loss) == set(names) and true in loss
                            and all(comparison.get(k, {}).get(route, {}).get('success')
                                    for k in names))
                        if usable:
                            values.append(float(loss[true]-min(loss.values())))
                        else:
                            complete = False
            if complete and task_count == 6 and len(values) == 6*expected_repeats:
                maxima.append(max(values))
        # Labelled simulation calibration; no field coverage guarantee.
        rank = int(np.ceil((len(maxima)+1)*.9))
        thresholds[route] = dict(histories=len(maxima),
            threshold=float(sorted(maxima)[rank-1]) if 1 <= rank <= len(maxima)
                      else None, maxima=maxima, rank=rank)
    return dict(version='ac_repair_calibration_v1', histories=len(records),
        available_histories=len(valid), G11_multiplier=chosen,
        objective=objective, history_losses=rows, SSE_thresholds=thresholds,
        objective_scope='true-candidate predictive NLL, equal history weights',
        fixed_before_evaluation=True)


def measurements(record, calibration=None):
    """Every planned task has a row, including unavailable inference."""
    names = record.get('candidate_names', [])
    true = record.get('true_name')
    if true not in names:
        return []
    index = names.index(true)
    rows = []
    for label, plan in record.get('plans', {}).items():
        action = record.get('actions', {}).get(plan.get('action_key'), {})
        banks = usable_banks(action)
        for regime in ('regular', 'shift'):
            actual = action.get('regimes', {}).get(regime, {})
            versions = [(m, 1.) for m in ('G00', 'G10', 'G01', 'G11')]
            a = (calibration or {}).get('G11_multiplier')
            if a is not None:
                versions.append(('G11_cal', a))
            for model, multiplier in versions:
                row = dict(index=record['spec']['index'], seed=record['spec']['seed'],
                    plan=label, regime=regime, model=model, available=False)
                bank = banks.get('G11' if model == 'G11_cal' else model)
                if bank is not None and actual.get('values') is not None:
                    try:
                        row.update(scores(bank, actual['values'], index, a=multiplier),
                            available=True)
                        clean = scores(bank, actual['clean'], index, a=multiplier)
                        row['clean_error'] = clean['error']
                        row['clean_margin'] = clean['clean_or_noisy_margin'][0]
                    except Exception as exc:
                        row['failure'] = type(exc).__name__+': '+str(exc)
                rows.append(row)
            for route in ('fixed', 'joint'):
                row = dict(index=record['spec']['index'], seed=record['spec']['seed'],
                    plan=label, regime=regime, model='AC_'+route, available=False)
                fits = actual.get('refits', [])
                metrics = []
                threshold = (calibration or {}).get('SSE_thresholds', {}).get(
                    route, {}).get('threshold')
                for fit in fits:
                    losses = fit.get('losses', {}).get(route, {})
                    comparison = fit.get('comparisons', {})
                    usable = (set(losses) == set(names)
                        and all(comparison.get(k, {}).get(route, {}).get('success')
                                for k in names))
                    if not usable:
                        continue
                    best = min(losses, key=losses.get)
                    part = dict(error=float(best != true),
                        SSE_margin=float(min(v for k, v in losses.items() if k != true)
                                         -losses[true]))
                    if threshold is not None:
                        kept = [k for k in names if losses[k]-min(losses.values()) <= threshold]
                        part.update(false_rejection=float(true not in kept),
                            unresolved=float(len(kept) != 1),
                            mean_set_size=float(len(kept)),
                            wrong_decisive=float(len(kept) == 1 and best != true),
                            correct_decisive=float(len(kept) == 1 and best == true))
                    metrics.append(part)
                if metrics and len(metrics) == len(fits):
                    row.update({k: float(np.mean([r[k] for r in metrics]))
                                for k in metrics[0]}, available=True)
                    cleanfit = actual.get('clean_refit', {})
                    loss = cleanfit.get('losses', {}).get(route, {})
                    cmp = cleanfit.get('comparisons', {})
                    if set(loss) == set(names) and all(
                            cmp.get(k, {}).get(route, {}).get('success') for k in names):
                        row['clean_error'] = float(min(loss, key=loss.get) != true)
                rows.append(row)
    return rows


def interval(values, seed=173, repetitions=5000):
    v = np.asarray(values, float)
    if len(v) < 2:
        return None
    rng = np.random.default_rng(seed)
    means = v[rng.integers(len(v), size=(repetitions, len(v)))].mean(1)
    return [float(t) for t in np.quantile(means, [.025, .975])]


def zero_observed_upper(values, alpha=.05):
    """One-sided bound for the mean expectation of independent [0,1] losses.

    When every observed loss is zero, P(all zero) <= product(1-E[X_i])
    <= (1-mean(E[X_i]))**n. No identical-distribution assumption is used.
    Independence and a fixed policy are required; missing histories stay
    outside this complete-case bound and retain their separate error range.
    """
    values = np.asarray(values, float)
    if values.ndim != 1 or np.any(~np.isfinite(values)) or np.any((values < 0) | (values > 1)):
        raise ValueError('independent history losses must be finite in [0,1]')
    if not 0 < alpha < 1:
        raise ValueError('alpha must lie strictly between zero and one')
    if not len(values) or np.any(values != 0):
        return None
    return float(-np.expm1(np.log(alpha)/len(values)))


def summarize(records, calibration=None):
    rows = [row for record in records for row in measurements(record, calibration)]
    groups = defaultdict(list)
    models = ['G00', 'G10', 'G01', 'G11', 'AC_fixed', 'AC_joint']
    if (calibration or {}).get('G11_multiplier') is not None:
        models.append('G11_cal')
    for plan in ('constant', 'SQMD', 'BCMD'):
        for model in models:
            groups[(plan, model)] = []
    for row in rows:
        groups[(row['plan'], row['model'])].append(row)
    totals = {}
    for (plan, model), tasks in groups.items():
        hist = defaultdict(list)
        for row in tasks:
            if row['available']:
                hist[row['index']].append(row)
        # Require both future regimes for a history-level complete-case mean.
        good = {i: r for i, r in hist.items() if len(r) == 2}
        metrics = {}
        for name in ('error', 'clean_error', 'false_rejection', 'unresolved',
                     'wrong_decisive', 'correct_decisive', 'true_probability',
                     'coverage90', 'classification_NLL', 'true_predictive_NLL', 'brier'):
            v = [float(np.mean([r[name] for r in rr])) for rr in good.values()
                 if all(name in r for r in rr)]
            if v:
                metrics[name] = dict(mean=float(np.mean(v)), histories=len(v),
                    bootstrap95=interval(v),
                    bootstrap_degenerate=bool(np.all(np.asarray(v) == v[0])))
                if name in BOUNDED_ERROR_METRICS:
                    upper = zero_observed_upper(v)
                    if upper is not None:
                        metrics[name]['zero_observed_one_sided95_upper'] = upper
                        metrics[name]['zero_observed_bound_scope'] = (
                            'average expected loss of independent [0,1] history values; '
                            'fixed policy, conditional on complete-case availability')
        available = len(good)
        error = metrics.get('error', {}).get('mean')
        count = len(records)
        metrics['availability'] = dict(histories=available, total_histories=count,
            failed_histories=count-available,
            failed_history_indices=[r['spec']['index'] for r in records
                                    if r['spec']['index'] not in good],
            error_all_histories_range=[0., 1.] if error is None else
                [error*available/count, (error*available+count-available)/count])
        totals[plan+'__'+model] = metrics
    paired = {}
    by = {(r['index'], r['plan'], r['model'], r['regime']): r for r in rows
          if r['available']}
    comparisons = []
    for plan in ('constant', 'SQMD', 'BCMD'):
        comparisons.extend((plan, 'G00', plan, m) for m in
                           ('G10', 'G01', 'G11', 'G11_cal', 'AC_fixed', 'AC_joint'))
    comparisons.extend(('SQMD', m, 'BCMD', m)
                       for m in ('G00', 'G11', 'G11_cal', 'AC_fixed', 'AC_joint'))
    for p0, m0, p1, m1 in comparisons:
        parts = []
        for record in records:
            i = record['spec']['index']; values = []
            for regime in ('regular', 'shift'):
                a = by.get((i, p0, m0, regime)); b = by.get((i, p1, m1, regime))
                if a is not None and b is not None:
                    values.append(b['error']-a['error'])
            if len(values) == 2:
                parts.append(float(np.mean(values)))
        if parts:
            paired[p0+'_'+m0+' -> '+p1+'_'+m1] = dict(histories=len(parts),
                error_difference=float(np.mean(parts)), bootstrap95=interval(parts))
    return dict(histories=len(records), summaries=totals, paired=paired, rows=rows,
        history_statuses=[dict(index=r['spec']['index'], status=r.get('status'),
                               failures=r.get('failures', {})) for r in records],
        interval_unit='independent historical dataset; future repeats averaged',
        uncertainty_scope='bootstrap descriptive interval for stated simulated strata',
        calibration=calibration)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--calibrate', action='store_true')
    parser.add_argument('--calibration')
    args = parser.parse_args()
    files = sorted(Path(args.input).glob('history_*.json.gz'))
    records = load_records(files)
    calibration = load(args.calibration) if args.calibration else None
    result = calibrate(records) if args.calibrate else summarize(records, calibration)
    write(args.output, result)


if __name__ == '__main__':
    main()
