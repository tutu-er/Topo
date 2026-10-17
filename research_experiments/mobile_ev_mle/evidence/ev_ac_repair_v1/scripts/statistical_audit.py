"""Independent stored-data probability audit; no AC, fitting or new samples.

Development mode uses a=1 and never reads/selects calibration. Completed mode
requires evaluation_complete, checks its calibration binding, and recomputes
the declared calibration rule using calibration histories only. This script
does not import the scientific study or its analysis/probability functions.
"""
from __future__ import annotations

import argparse
import base64
from collections import defaultdict
from datetime import datetime, timezone
import gzip
from hashlib import sha256
import json
import math
import os
from pathlib import Path

for _name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
              'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '1'

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.special import logsumexp


PLANS = ('constant', 'SQMD', 'BCMD')
REGIMES = ('regular', 'shift')
GAUSSIANS = ('G00', 'G10', 'G01', 'G11')
GRID = (1., 2., 4., 8., 16., 32.)
ERROR_METRICS = ('error', 'clean_error', 'false_rejection', 'wrong_decisive')
SUMMARY_METRICS = (*ERROR_METRICS, 'unresolved', 'correct_decisive',
                   'true_probability', 'coverage90', 'classification_NLL',
                   'true_predictive_NLL', 'brier')
COVERAGE_METRICS = ('coverage90_terminal_P', 'coverage90_terminal_Q',
                    'coverage90_voltage', 'coverage90_root_P', 'coverage90_root_Q')


def read(path):
    data = Path(path).read_bytes()
    return json.loads(gzip.decompress(data) if str(path).endswith('.gz') else data)


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def close(left, right):
    return bool(np.isclose(float(left), float(right), atol=1e-8, rtol=1e-11))


def gaussian_score(bank, values, true_index, multiplier=1., voltage_slice=None,
                   channel_slices=None):
    """Independent Cholesky density, MAP, 90% mass set and marginal coverage."""
    means = np.asarray(bank['means'], dtype=np.float64)
    covariance = np.asarray(bank['covariances'], dtype=np.float64)
    if means.ndim != 3 or means.shape[0] != 1:
        raise ValueError('one fixed-plan scenario required')
    means = means[0]; covariance = covariance[0]
    if covariance.shape != (len(means), means.shape[1], means.shape[1]):
        raise ValueError('Gaussian covariance dimensions disagree')
    y = np.atleast_2d(np.asarray(values, dtype=np.float64))
    prior = np.asarray(bank['log_prior'], dtype=np.float64)
    if (y.shape[1] != means.shape[1] or prior.shape != (len(means),)
            or not 0 <= true_index < len(means)):
        raise ValueError('data, candidate index or prior dimensions disagree')
    if multiplier < 1 or not np.isfinite(multiplier):
        raise ValueError('variance multiplier must be finite >=1')
    covariances = np.eye(y.shape[1])+multiplier*(covariance-np.eye(y.shape[1]))
    likelihoods = []
    for mean, sigma in zip(means, covariances):
        factor = cho_factor(sigma, lower=True)
        residual = y-mean
        square = np.einsum('ij,ji->i', residual, cho_solve(factor, residual.T))
        logdet = 2*np.log(np.diag(factor[0])).sum()
        likelihoods.append(-.5*(square+logdet+y.shape[1]*math.log(2*math.pi)))
    likelihoods = np.asarray(likelihoods).T
    scores = likelihoods+prior
    logpost = scores-logsumexp(scores, axis=1, keepdims=True)
    posterior = np.exp(logpost)
    selected = posterior.argmax(axis=1)
    sets = []
    for probability in posterior:
        order = np.argsort(-probability, kind='stable')
        count = min(len(order), int(np.searchsorted(np.cumsum(probability[order]),
                                                    .9, side='left'))+1)
        sets.append(order[:count].tolist())
    sizes = np.asarray([len(kept) for kept in sets])
    margin = scores[:, true_index]-scores[:, np.arange(len(means)) != true_index].max(axis=1)
    covered = abs(y-means[true_index]) <= 1.6448536269514722*np.sqrt(
        np.diag(covariances[true_index]))
    target = np.zeros(len(means)); target[true_index] = 1.
    result = dict(error=float(np.mean(selected != true_index)),
        false_rejection=float(np.mean([true_index not in kept for kept in sets])),
        unresolved=float(np.mean(sizes != 1)),
        correct_decisive=float(np.mean((sizes == 1) & (selected == true_index))),
        wrong_decisive=float(np.mean((sizes == 1) & (selected != true_index))),
        true_probability=float(posterior[:, true_index].mean()),
        coverage90=float(covered.mean()),
        true_predictive_NLL=float(-likelihoods[:, true_index].mean()),
        classification_NLL=float(-logpost[:, true_index].mean()),
        brier=float(np.mean(np.sum((posterior-target)**2, axis=1))),
        mean_set_size=float(sizes.mean()), selected_indices=selected.tolist(),
        posterior=posterior.tolist(), credible_sets90=sets,
        truth_score_margin=margin.tolist(),
        historical_log_weight_normalizer=float(logsumexp(prior)))
    if voltage_slice is not None:
        result['coverage90_voltage'] = float(covered[:, voltage_slice].mean())
    for name, section in (channel_slices or {}).items():
        result['coverage90_'+name] = float(covered[:, section].mean())
    return result


def eligible_banks(action, settings):
    propagation = action.get('propagation', {})
    banks = propagation.get('banks', {})
    precise = all(row.get('status') == 'ok'
        and row.get('mean_MCSE_whitened_max', np.inf) <= settings['propagation_mean_MCSE_limit']
        and row.get('covariance_MCSE_relative_fro', np.inf) <= settings['propagation_covariance_MCSE_relative_limit']
        for row in propagation.get('rows', []))
    return banks if precise else {name: value for name, value in banks.items() if name == 'G00'}


def refit_score(fit, names, true_name, route, threshold=None):
    losses = fit.get('losses', {}).get(route, {})
    comparisons = fit.get('comparisons', {})
    usable = (set(losses) == set(names) and true_name in losses and
        all(comparisons.get(name, {}).get(route, {}).get('success') for name in names))
    result = dict(available=bool(usable), complete_refit=fit.get('complete'),
        failures=fit.get('failures', {}))
    if set(losses) == set(names) and true_name in losses:
        best = min(losses, key=losses.get)
        result.update(selected=best, ranked_candidates=sorted(losses, key=losses.get),
            losses=losses, truth_score_margin=float(min(loss for name, loss in losses.items()
                if name != true_name)-losses[true_name]))
    if usable:
        result['error'] = float(best != true_name)
        if threshold is not None:
            kept = [name for name in names if losses[name]-min(losses.values()) <= threshold]
            result.update(false_rejection=float(true_name not in kept),
                unresolved=float(len(kept) != 1), mean_set_size=float(len(kept)),
                correct_decisive=float(len(kept) == 1 and best == true_name),
                wrong_decisive=float(len(kept) == 1 and best != true_name),
                retained_candidates=kept)
    return result


def audit_history(record, settings, calibration=None, calibration_mode=False):
    """Scalar rows only; release the large raw history before loading another."""
    names = record.get('candidate_names', [])
    truth = record.get('true_name')
    true_index = names.index(truth) if truth in names else None
    metadata = dict(index=record['spec']['index'], seed=record['spec']['seed'],
        spec=record['spec'], status=record.get('status'), failures=record.get('failures', {}),
        true_name=truth, candidate_names=names, true_index_by_name=true_index,
        saved_true_index=record.get('true_index'),
        true_index_agrees=true_index is None or record.get('true_index', true_index) == true_index,
        plan_action_keys={name: plan.get('action_key') for name, plan in record.get('plans', {}).items()})
    rows = []; nll = {str(a): [] for a in GRID}
    calibration_complete = len(record.get('plans', {})) == 3 and true_index is not None
    threshold_rows = {route: dict(complete=set(record.get('plans', {})) == set(PLANS),
        values=[], tasks=0) for route in ('fixed', 'joint')}
    versions = [(name, 1.) for name in GAUSSIANS]
    if calibration and calibration.get('G11_multiplier') is not None:
        versions.append(('G11_cal', calibration['G11_multiplier']))
    repeats = int(record.get('config', {}).get('noise_repeats', settings['noise_repeats']))
    n = int(str(record['spec'].get('bank', 'n0'))[1:])
    T = int(record.get('config', {}).get('horizon', 8))
    voltage = slice(2*T*n, 3*T*n) if n else None
    sections = dict(terminal_P=slice(0, T*n), terminal_Q=slice(T*n, 2*T*n),
        voltage=voltage, root_P=slice(3*T*n, 3*T*n+T),
        root_Q=slice(3*T*n+T, 3*T*n+2*T)) if n else {}
    metadata['channel_coordinate_counts'] = {name: section.stop-section.start
                                             for name, section in sections.items()}
    noise_consistent = []
    for plan in PLANS:
        action = record.get('actions', {}).get(record.get('plans', {}).get(plan, {}).get('action_key'), {})
        banks = eligible_banks(action, settings)
        for regime in REGIMES:
            actual = action.get('regimes', {}).get(regime, {})
            values = actual.get('values')
            if values is not None:
                noise_consistent.append(len(np.atleast_2d(values)) == repeats)
            for model, multiplier in versions:
                row = dict(index=metadata['index'], seed=metadata['seed'], plan=plan,
                    regime=regime, model=model, available=False)
                bank = banks.get('G11' if model == 'G11_cal' else model)
                if bank is not None and values is not None and true_index is not None:
                    try:
                        result = gaussian_score(bank, values, true_index, multiplier, voltage, sections)
                        row.update(result, available=True)
                        if actual.get('clean') is not None:
                            clean = gaussian_score(bank, actual['clean'], true_index, multiplier, voltage, sections)
                            row.update(clean_error=clean['error'], clean=clean)
                    except Exception as exc:
                        row['audit_scoring_failure'] = type(exc).__name__+': '+str(exc)
                rows.append(row)
            if calibration_mode and plan in record.get('plans', {}):
                if 'G11' not in banks or values is None or true_index is None:
                    calibration_complete = False
                else:
                    for a in GRID:
                        nll[str(a)].append(gaussian_score(banks['G11'], values,
                            true_index, a)['true_predictive_NLL'])
            fits = actual.get('refits', [])
            for route in ('fixed', 'joint'):
                threshold = (calibration or {}).get('SSE_thresholds', {}).get(route, {}).get('threshold')
                noisy = [refit_score(fit, names, truth, route, threshold) for fit in fits]
                clean = refit_score(actual.get('clean_refit', {}), names, truth, route, threshold)
                row = dict(index=metadata['index'], seed=metadata['seed'], plan=plan,
                    regime=regime, model='AC_'+route, available=False,
                    noisy_refit_samples=noisy, clean_refit=clean,
                    declared_noise_repeats=repeats)
                if noisy and all(item['available'] for item in noisy):
                    for metric in ('error', 'false_rejection', 'unresolved', 'mean_set_size',
                                   'wrong_decisive', 'correct_decisive'):
                        if all(metric in item for item in noisy):
                            row[metric] = float(np.mean([item[metric] for item in noisy]))
                    row['available'] = True
                    if clean['available']:
                        row['clean_error'] = clean['error']
                rows.append(row)
                if calibration_mode and plan in record.get('plans', {}):
                    condition = threshold_rows[route]
                    condition['tasks'] += 1
                    if len(fits) != repeats or not all(item['available'] for item in noisy):
                        condition['complete'] = False
                    condition['values'].extend(float(max(0., -item['truth_score_margin']))
                        for item in noisy if item['available'])
    metadata['noise_repeat_counts_consistent'] = all(noise_consistent)
    for route, item in threshold_rows.items():
        item['complete'] &= item['tasks'] == 6 and len(item['values']) == 6*repeats
        item['maximum'] = max(item['values']) if item['complete'] else None
    calibration_complete &= all(len(values) == 6 for values in nll.values())
    # The declared calibration rows omit process failures with unknown truth.
    cal_row = None if not calibration_mode or true_index is None else dict(index=metadata['index'],
        complete=bool(calibration_complete), losses={key:float(np.mean(value))
            if value else None for key, value in nll.items()})
    return dict(metadata=metadata, rows=rows, calibration_row=cal_row,
                threshold_rows=threshold_rows)


def calibration_from_audits(audits):
    rows = [item['calibration_row'] for item in audits if item['calibration_row'] is not None]
    valid = [row for row in rows if row['complete']]
    objective = {str(a): float(np.mean([r['losses'][str(a)] for r in valid]))
                 if valid else None for a in GRID}
    chosen = min(GRID, key=lambda a: objective[str(a)]) if valid else None
    thresholds = {}
    for route in ('fixed', 'joint'):
        maxima = [item['threshold_rows'][route]['maximum'] for item in audits
                  if item['threshold_rows'][route]['complete']]
        rank = math.ceil((len(maxima)+1)*.9)
        thresholds[route] = dict(histories=len(maxima), maxima=maxima, rank=rank,
            threshold=sorted(maxima)[rank-1] if 1 <= rank <= len(maxima) else None)
    return dict(histories=len(audits), available_histories=len(valid),
        G11_multiplier=chosen, objective=objective, history_losses=rows,
        SSE_thresholds=thresholds)


def bootstrap(values):
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return None
    rng = np.random.default_rng(173)
    means = values[rng.integers(len(values), size=(5000, len(values)))].mean(axis=1)
    return np.quantile(means, [.025, .975]).tolist()


def summary(audits, calibrated=False):
    models = (*GAUSSIANS, 'AC_fixed', 'AC_joint')+ (('G11_cal',) if calibrated else ())
    indices = [item['metadata']['index'] for item in audits]
    groups = {}; good_rows = {}
    for plan in PLANS:
        for model in models:
            histories = {}
            for audit in audits:
                rows = [row for row in audit['rows'] if row['plan'] == plan
                        and row['model'] == model and row['available']]
                if len(rows) == 2:
                    histories[audit['metadata']['index']] = rows
                    for row in rows:
                        good_rows[(row['index'], plan, model, row['regime'])] = row
            metrics = {}
            for name in (*SUMMARY_METRICS, *COVERAGE_METRICS):
                values = [float(np.mean([r[name] for r in rows])) for rows in histories.values()
                          if all(name in row for row in rows)]
                if values:
                    metric = dict(mean=float(np.mean(values)), histories=len(values),
                        bootstrap95=bootstrap(values), bootstrap_degenerate=len(set(values)) == 1)
                    if name in ERROR_METRICS and all(value == 0 for value in values):
                        metric['zero_observed_one_sided95_upper'] = float(
                            -np.expm1(np.log(.05)/len(values)))
                    metrics[name] = metric
            count = len(audits); available = len(histories)
            error = metrics.get('error', {}).get('mean')
            metrics['availability'] = dict(histories=available, total_histories=count,
                failed_histories=count-available,
                failed_history_indices=[index for index in indices if index not in histories],
                error_all_histories_range=[0., 1.] if error is None else [
                    error*available/count, (error*available+count-available)/count])
            operational = defaultdict(list); fallback_histories = []; fallback_tasks = 0
            for audit in audits:
                values = defaultdict(list); fallback = False
                for regime in REGIMES:
                    found = next((row for row in audit['rows'] if row['plan'] == plan
                        and row['model'] == model and row['regime'] == regime), None)
                    fields = ('false_rejection', 'unresolved', 'correct_decisive', 'wrong_decisive')
                    valid_set = found is not None and found['available'] and all(
                        field in found for field in fields)
                    if not valid_set:
                        fallback = True; fallback_tasks += 1
                    for field in fields:
                        values[field].append(found[field] if valid_set else float(field == 'unresolved'))
                for field, value in values.items():
                    operational[field].append(float(np.mean(value)))
                if fallback:fallback_histories.append(audit['metadata']['index'])
            metrics['operational_all_histories'] = dict(
                metrics={name:dict(mean=float(np.mean(value)), histories=count)
                         for name, value in operational.items()},
                fallback_history_indices=fallback_histories, fallback_regime_tasks=fallback_tasks,
                rule='missing/failed inference or missing calibrated set rule retains all candidates',
                candidate_coverage_assumption='the declared synthetic candidate banks contain the true structure',
                MAP_error_scope='point MAP remains complete-case with separate missing-data range')
            groups[plan+'__'+model] = metrics
    pairs = []
    for plan in PLANS:
        pairs.extend((plan, 'G00', plan, model) for model in
                     ('G10', 'G01', 'G11', 'G11_cal', 'AC_fixed', 'AC_joint'))
        pairs.append((plan, 'AC_fixed', plan, 'AC_joint'))
    pairs.extend(('SQMD', model, 'BCMD', model) for model in models)
    pairs.extend(('constant', model, 'SQMD', model) for model in models)
    paired = {}
    for p0, m0, p1, m1 in pairs:
        result = {}
        for metric in ('error', 'clean_error'):
            parts = []; missing = []
            for index in indices:
                values = []
                for regime in REGIMES:
                    first = good_rows.get((index, p0, m0, regime))
                    second = good_rows.get((index, p1, m1, regime))
                    if first is not None and second is not None and metric in first and metric in second:
                        values.append(second[metric]-first[metric])
                if len(values) == 2:
                    parts.append(float(np.mean(values)))
                else:
                    missing.append(index)
            if parts:
                total = len(indices)
                result[metric] = dict(histories=len(parts), total_histories=total,
                    unavailable_history_indices=missing, difference_new_minus_old=float(np.mean(parts)),
                    bootstrap95=bootstrap(parts),
                    all_history_difference_range=[max(-1., (sum(parts)-len(missing))/total),
                        min(1., (sum(parts)+len(missing))/total)])
        if result:
            paired[p0+'_'+m0+' -> '+p1+'_'+m1] = result
    return dict(histories=len(audits), summaries=groups, paired=paired,
        history_statuses=[item['metadata'] for item in audits],
        denominator='independent histories; two regimes and meter repeats averaged within history',
        interval_scope='bootstrap is descriptive; zero-event bound requires independent histories and fixed policy')


def verify_specs(frozen):
    seeds = {phase: [row['seed'] for row in rows]
             for phase, rows in frozen['specifications'].items()}
    unique = {phase: len(values) == len(set(values)) for phase, values in seeds.items()}
    overlaps = {a+'__'+b: sorted(set(seeds[a]) & set(seeds[b]))
                for a in seeds for b in seeds if a < b}
    return dict(seed_counts={key: len(value) for key, value in seeds.items()},
        within_phase_unique=unique, cross_phase_overlaps=overlaps,
        passed=all(unique.values()) and not any(overlaps.values()))


def walk_compare(saved, independent, prefix='', errors=None):
    """Compare saved fields that this independent implementation recomputes."""
    errors = [] if errors is None else errors
    if isinstance(saved, dict) and isinstance(independent, dict):
        for key in set(saved) & set(independent):
            walk_compare(saved[key], independent[key], prefix+'.'+str(key), errors)
    elif isinstance(saved, list) and isinstance(independent, list):
        if len(saved) != len(independent):
            errors.append(prefix+': list length differs')
        else:
            for index, (a, b) in enumerate(zip(saved, independent)):
                walk_compare(a, b, prefix+'.'+str(index), errors)
    elif saved is None or independent is None:
        if saved != independent: errors.append(prefix+': None differs')
    elif isinstance(saved, (int, float)) and isinstance(independent, (int, float)):
        if not close(saved, independent):errors.append(prefix+': numeric value differs')
    elif saved != independent:
        errors.append(prefix+': value differs')
    return errors


def run(root, mode='development', expected_policy=None):
    root = Path(root).resolve(); frozen = read(root/'freeze.json')
    controller = read(root/'controller.json')
    phase = 'development' if mode == 'development' else 'evaluation'
    if controller.get('status') != phase+'_complete':
        raise ValueError('audit waits for '+phase+'_complete; no running scientific batch is read')
    if expected_policy is not None and frozen['policy_id'] != expected_policy:
        raise ValueError('unexpected policy ID')
    policy = dict(frozen); saved_policy = policy.pop('policy_id')
    calculated_policy = sha256(json.dumps(policy, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    issues = []
    if saved_policy != calculated_policy:issues.append('freeze policy digest mismatch')
    if controller.get('policy_id') != saved_policy:issues.append('controller policy digest mismatch')
    spec_check = verify_specs(frozen)
    if not spec_check['passed']:issues.append('seed overlap or duplicate')
    snapshot = read(root/'source_snapshot.json.gz')
    snapshot_ok = set(snapshot) == set(frozen['source_sha256']) and all(
        sha256(base64.b64decode(snapshot[name])).hexdigest() == value
        for name, value in frozen['source_sha256'].items())
    if not snapshot_ok:issues.append('frozen source snapshot hash mismatch')
    report = dict(audit_version='independent_stored_probability_v1', mode=mode,
        created_UTC=datetime.now(timezone.utc).isoformat(), policy_id=saved_policy,
        audit_script_sha256=digest(__file__), freeze_sha256=digest(root/'freeze.json'),
        source_snapshot_verified=snapshot_ok, seed_registry=spec_check,
        Gaussian_scoring='independent Cholesky; full mean/covariance stored banks; meter variance I retained',
        non_Gaussian_predictive_density_implemented=False,
        nonlinear_AC_history_posterior_implemented=False,
        conditional_AC_scope='measured terminal P/Q fixed; differs from latent-input Gaussian likelihood',
        no_new_AC_or_physical_samples=True, bootstrap_uses_existing_histories_only=True,
        evaluation_used_to_select_multiplier=False,
        stages={}, raw_inputs={}, calibration_audit=None)
    calibration = None
    if mode != 'development':
        if any(len(frozen['specifications'][name]) != 32 for name in ('calibration', 'evaluation')):
            raise ValueError('completed main audit requires all 32 calibration and 32 evaluation specifications')
        calibration = read(root/'calibration.json')
        binding = read(root/'evaluation_calibration_binding.json')
        expected_binding = dict(policy_id=saved_policy, calibration_sha256=digest(root/'calibration.json'))
        if binding != expected_binding:issues.append('evaluation calibration hash binding mismatch')
        if calibration.get('policy_id') != saved_policy or calibration.get('histories') != 32:
            issues.append('saved calibration policy/count mismatch')
        if calibration.get('G11_multiplier') not in GRID:
            issues.append('completed evaluation has no declared calibrated multiplier')
        actual_hashes = {('calibration/history_%03d.json.gz'%row['index']):
            digest(root/'calibration'/('history_%03d.json.gz'%row['index']))
            for row in frozen['specifications']['calibration']}
        saved_hashes = {key.replace('\\', '/'): value for key, value in calibration.get('input_sha256', {}).items()}
        if saved_hashes != actual_hashes:issues.append('calibration input hashes mismatch')
        report['calibration_hash_binding'] = dict(saved=binding, expected=expected_binding,
            input_hashes_equal=saved_hashes == actual_hashes)
    stages = ('development',) if mode == 'development' else ('calibration', 'evaluation')
    for stage in stages:
        audits = []
        for spec in frozen['specifications'][stage]:
            relative = stage+'/history_%03d.json.gz'%spec['index']; path = root/relative
            record = read(path); report['raw_inputs'][relative] = digest(path)
            if record.get('spec') != spec:issues.append(relative+': frozen specification mismatch')
            initial = record.get('code_initial_sha', {})
            if record.get('status') == 'process_failure':
                if record.get('source_sha256') != frozen['source_sha256']:
                    issues.append(relative+': failed-worker frozen source hash mismatch')
            elif not initial:
                issues.append(relative+': worker source hashes missing')
            if initial and (record.get('code_end_sha') != initial or not record.get('source_unchanged_during_run')):
                issues.append(relative+': source changed while worker ran')
            if any(frozen['source_sha256'].get(key) != value for key, value in initial.items()):
                issues.append(relative+': worker code hash differs from snapshot')
            if record.get('config'):
                for setting in ('draws', 'replicates', 'noise_repeats', 'clean_refit'):
                    if record['config'].get(setting) != frozen['settings'][setting]:
                        issues.append(relative+': frozen runtime setting differs: '+setting)
            audited = audit_history(record, frozen['settings'], calibration if stage == 'evaluation' else None,
                                    calibration_mode=stage == 'calibration')
            if not audited['metadata']['true_index_agrees']:issues.append(relative+': true index/name mismatch')
            if not audited['metadata']['noise_repeat_counts_consistent']:
                issues.append(relative+': declared noise repeat count differs')
            audits.append(audited)
            del record
        values = summary(audits, calibrated=stage == 'evaluation')
        stored_path = root/(stage+'_summary.json')
        comparison = []
        if stored_path.exists():
            stored = read(stored_path)
            if set(stored.get('summaries', {})) != set(values['summaries']):
                comparison.append('saved summary plan/model group keys differ')
            for key, own in values['summaries'].items():
                saved = stored.get('summaries', {}).get(key, {})
                for field in ('availability', *SUMMARY_METRICS):
                    if (field in own) != (field in saved):
                        comparison.append(key+': missing/extra saved metric '+field)
            comparison.extend(walk_compare(stored.get('summaries', {}), values['summaries']))
            for key, saved in stored.get('paired', {}).items():
                own = values['paired'].get(key, {}).get('error')
                if own is None:
                    comparison.append(key+': paired row missing')
                else:
                    comparison.extend(walk_compare(saved, dict(histories=own['histories'],
                        error_difference=own['difference_new_minus_old'], bootstrap95=own['bootstrap95']), key))
        else:
            comparison.append('stored stage summary missing')
        if comparison:issues.extend(stage+': '+item for item in comparison)
        report['stages'][stage] = dict(summary=values, histories=audits,
            saved_summary_comparison_errors=comparison)
        if stage == 'calibration':
            recalculated = calibration_from_audits(audits)
            errors = walk_compare(calibration, recalculated)
            report['calibration_audit'] = dict(recalculated=recalculated,
                saved_multiplier=calibration.get('G11_multiplier'), comparison_errors=errors,
                selection_data='calibration histories only; never evaluation')
            if errors:issues.extend('calibration rule: '+item for item in errors)
    report['audit_issues'] = issues
    report['passed'] = not issues
    report['limits'] = [
        'No new planning, posterior estimation, AC solve, posterior density integration or physical audit.',
        'Development a=1 results are exploratory; no covariance multiplier is selected in development mode.',
        'Gaussian scores and credible mass sets are model quantities; calibration is empirical on the declared synthetic cases.',
        'Conditional AC scores are attained local fits; no global minimum or calibrated topology probability is claimed.',
        'Zero observed errors do not imply zero risk; all failed histories retain their original denominator.',
        'Pairs average two regimes within an independent history, not as additional independent histories.']
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--mode', choices=('development', 'completed'), default='development')
    parser.add_argument('--expected-policy')
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():parser.error('output already exists; choose a new audit filename')
    result = run(args.root, args.mode, args.expected_policy)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8', newline='\n') as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')
    print(json.dumps(dict(output=str(output), passed=result['passed'],
        audit_issues=result['audit_issues']), ensure_ascii=False))
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
