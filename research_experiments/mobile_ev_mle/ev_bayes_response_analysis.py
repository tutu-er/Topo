"""Offline history-level statistics for frozen response-uncertainty records.

Only saved files are read. No AC, optimizer, predictor or generator is imported.
Calibration and evaluation reports remain separate. Conditional meter draws
are averaged within each independent history before any interval or contrast.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from functools import lru_cache
import gzip
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter

import numpy as np


DOMAINS = ('fixed_charge', 'fixed_bidir', 'mobile_charge', 'mobile_bidir')
PARENTS = dict(fixed_charge=(), fixed_bidir=('fixed_charge',),
    mobile_charge=('fixed_charge',), mobile_bidir=('fixed_bidir', 'mobile_charge'))
FAMILIES = ('square', 'square_budget', 'ranked', 'fixed_bc', 'joint_oldbc', 'joint_bc')
METHODS = ('none', 'constant', 'highlow')+tuple(f+'__'+d for f in FAMILIES for d in DOMAINS)
MODELS = ('joint', 'fixed', 'no_cross', 'line_known', 'evidence_only')
REGIMES = ('regular', 'shift')
INFERENCE_METHODS = ('constant',)+tuple('square__'+d for d in DOMAINS)+(
    'joint_bc__mobile_bidir', 'ranked__mobile_bidir', 'fixed_bc__mobile_bidir')
ACTUAL = ('error', 'brier', 'negative_log_true_probability', 'true_rejected', 'unresolved',
    'unique_correct', 'wrong_unique', 'confidence', 'confidence_minus_correct',
    'true_probability', 'credible_size', 'abstention_loss',
    'meter_interval90_coverage', 'voltage_interval90_coverage')
PREDICTED = ('risk', 'map_error', 'eig', 'brier_risk', 'abstention_risk')
BINS = (0., .25, .5, .7, .8, .9, .95, .99, 1.)
BOUNDS = {k: (0., 1.) for k in ACTUAL if k not in ('negative_log_true_probability', 'brier', 'confidence_minus_correct', 'credible_size')}
BOUNDS.update(brier=(0., 2.), confidence_minus_correct=(-1., 1.))


def read(path):
    path = Path(path); data = path.read_bytes()
    return json.loads(gzip.decompress(data) if path.suffix == '.gz' else data)


def serial(value):
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, dict): return {str(k): serial(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)): return [serial(v) for v in value]
    return value


def _identity(value):
    return json.dumps(serial(value), sort_keys=True, separators=(',', ':'), allow_nan=False)


def _number(value):
    try:
        value = float(value)
        return value if np.isfinite(value) else None
    except (ValueError, TypeError):
        return None


def lightweight_record(record):
    """Discard dense predictors immediately after reading each raw history.

    This is an in-memory analysis view, not a replacement evidence record.
    Actual score columns, failure slots and every requested comparison remain.
    """
    keep = ('spec', 'status', 'true_name', 'failures', 'failure', 'seconds', 'config',
            'operation_config', 'planning_seconds', 'independent_paired', 'true_lines')
    result = {k: record[k] for k in keep if k in record}
    result['training_result'] = dict(fits=[dict(candidate=r.get('candidate'),
        diagnostics=dict(ratio_constraints=r.get('diagnostics', {}).get('ratio_constraints', {})))
        for r in record.get('training_result', {}).get('fits', [])])
    joint = record.get('prediction', {}).get('joint', {})
    result['prediction'] = dict(joint={k: joint[k] for k in ('candidate_names', 'log_prior') if k in joint})
    name = record.get('true_name'); update = joint.get('candidate_updates', {}).get(name)
    if update is not None:
        small = {k: update[k] for k in ('logZprior', 'logZT', 'logZTV') if k in update}
        for k in ('T_line_integration', 'TV_line_integration'):
            small[k] = {a: b for a, b in update.get(k, {}).items() if isinstance(b, (int, float, bool, str))}
        moment = update.get('TV_posterior_future', {}); d = 2*len(record.get('true_lines', {}).get('r', []))
        if d and 'mean' in moment and 'cov' in moment:
            small['TV_posterior_future'] = dict(mean=moment['mean'][:d], cov=[row[:d] for row in moment['cov'][:d]])
        result['prediction']['joint']['candidate_updates'] = {name: small}
    result['methods'] = {}
    for name, saved in record.get('methods', {}).items():
        row = {k: saved[k] for k in ('domain', 'action_key') if k in saved}
        if saved.get('operation'):
            row['operation'] = {k: saved['operation'][k] for k in ('feasible', 'moves', 'throughput_kwh',
                'driver_cost', 'travel_energy_kwh', 'energy_kwh') if k in saved['operation']}
        if saved.get('decoded'):
            row['decoded'] = {k: saved['decoded'][k] for k in ('charge_kw', 'discharge_kw') if k in saved['decoded']}
        result['methods'][name] = row
    result['actions'] = {}
    for key, action in record.get('actions', {}).items():
        result['actions'][key] = dict(prediction_failures=action.get('prediction_failures', {}),
            predictions={m: dict(independent_score={k: row.get('independent_score', {}).get(k)
                for k in PREDICTED+('risk_se',)}) for m, row in action.get('predictions', {}).items()},
            actual={r: dict(models=row.get('models', {})) for r, row in action.get('actual', {}).items()})
    result['planning_logs'] = {}
    for family, domains in record.get('planning_logs', {}).items():
        result['planning_logs'][family] = {}
        for domain, log in domains.items():
            events = []
            for e in log.get('events', []):
                event = {k: e[k] for k in ('iteration', 'accepted', 'parent_risk', 'proposed_risk', 'full_AC_failure') if k in e}
                event['milp'] = {k: e.get('milp', {}).get(k) for k in ('success', 'failure', 'solver_incumbent',
                    'solver_best_bound', 'solver_relative_gap', 'proxy_value', 'proxy_uniform_error', 'proxy_absolute_gap', 'stop_reason', 'seconds')}
                events.append(event)
            result['planning_logs'][family][domain] = dict(events=events)
    return result


def coverage(expected, present):
    ids = [_identity(v) for v in present]; targets = [_identity(v) for v in expected]
    count = Counter(ids)
    return dict(exact=len(ids) == len(set(ids)) and set(ids) == set(targets),
        expected_histories=len(targets), present_records=len(ids),
        missing=[v for v in expected if _identity(v) not in count],
        unexpected=[v for v in present if _identity(v) not in targets],
        duplicates=[json.loads(v) for v, n in count.items() if n > 1])


@lru_cache(maxsize=16)
def _bootstrap_weights(n, repetitions, seed):
    if n < 1: return np.zeros((repetitions, 0))
    return np.random.default_rng(seed).multinomial(n, np.full(n, 1/n), size=repetitions).astype(float)


def history_estimate(values, *, expected=None, bootstrap=1000, seed=2026119001, bounds=None, strata=None):
    """One value per history; missing slots remain in the declared denominator.

    The pooled bootstrap describes the experimental condition mixture. A
    separate stratified SE uses independent replicates within each fixed cell
    when every cell has at least two available histories.
    """
    numbers = [_number(v) for v in values]; N = len(numbers) if expected is None else int(expected)
    if N != len(numbers): raise ValueError('supply one value or None for every expected history')
    valid = np.array([v is not None for v in numbers]); x = np.array([0. if v is None else v for v in numbers])
    n = int(valid.sum()); mean = float(x.sum()/n) if n else None
    result = dict(mean=mean, available_histories=n, expected_histories=N, missing_histories=N-n,
        se_between_histories=None, pooled_history_bootstrap95=None,
        interval_scope='descriptive independent-history bootstrap over the declared condition mixture',
        stratified_SE=None, stratified_normal95=None)
    if n > 1:
        result['se_between_histories'] = float(x[valid].std(ddof=1)/np.sqrt(n))
        weights = _bootstrap_weights(N, int(bootstrap), int(seed))
        count = weights@valid; usable = count > 0
        means = (weights@x)[usable]/count[usable]
        result['pooled_history_bootstrap95'] = np.quantile(means, [.025, .975]).tolist()
    if bounds is not None and N:
        lo, hi = bounds
        result['missing_outcome_bounds'] = [(float(x.sum())+(N-n)*lo)/N, (float(x.sum())+(N-n)*hi)/N]
    if strata is not None and len(strata) == N and n == N and N:
        groups = defaultdict(list)
        for group, value in zip(strata, x): groups[str(group)].append(float(value))
        if all(len(v) >= 2 for v in groups.values()):
            variance = sum((len(v)/N)**2*np.var(v, ddof=1)/len(v) for v in groups.values())
            se = float(np.sqrt(variance)); result['stratified_SE'] = se
            result['stratified_normal95'] = [mean-1.959963984540054*se, mean+1.959963984540054*se]
            result['stratified_scope'] = 'fixed-cell allocation, within-cell replicate variance; normal approximation'
        else:
            result['stratified_scope'] = 'within-cell variance unavailable: fewer than two independent histories per cell'
    return result


def calibration_moments(confidence, correct, edges=BINS):
    c, y = np.asarray(confidence, float), np.asarray(correct, float)
    e = np.asarray(edges, float)
    if c.ndim != 1 or y.shape != c.shape or not len(c) or not np.isfinite(c).all() or not np.isfinite(y).all():
        raise ValueError('finite paired confidence/correct conditional draws required')
    if np.any((c < 0) | (c > 1)) or np.any((y < 0) | (y > 1)) or e[0] != 0 or e[-1] != 1 or np.any(np.diff(e) <= 0):
        raise ValueError('proper confidence bins and probabilities required')
    index = np.clip(np.searchsorted(e, c, side='right')-1, 0, len(e)-2)
    return [dict(lower=float(a), upper=float(b), mass=float(np.mean(index == j)),
        confidence_moment=float(np.sum(c[index == j])/len(c)),
        correct_moment=float(np.sum(y[index == j])/len(c)), conditional_draws=int(np.sum(index == j)))
        for j, (a, b) in enumerate(zip(e[:-1], e[1:]))]


def aggregate_calibration(histories, expected):
    valid = [v for v in histories if v is not None]; rows = []
    if not valid: return dict(available_histories=0, expected_histories=expected, ECE=None, bins=[])
    for j in range(len(BINS)-1):
        selected = [h[j] for h in valid]
        mass = float(np.mean([v['mass'] for v in selected])); cm = float(np.mean([v['confidence_moment'] for v in selected])); ym = float(np.mean([v['correct_moment'] for v in selected]))
        rows.append(dict(lower=BINS[j], upper=BINS[j+1], mass=mass,
            mean_confidence=cm/mass if mass else None, mean_correct=ym/mass if mass else None,
            confidence_minus_correct=(cm-ym)/mass if mass else None,
            histories_with_bin=sum(v['mass'] > 0 for v in selected),
            conditional_draws=sum(v['conditional_draws'] for v in selected),
            weighted_absolute_gap=abs(cm-ym)))
    return dict(available_histories=len(valid), expected_histories=expected,
        ECE=float(sum(v['weighted_absolute_gap'] for v in rows)), bins=rows,
        weighting='each available independent history has equal total weight',
        scope='physical simulator confidence calibration; ECE is a descriptive binned statistic')


def actual_metrics(evaluation, expected_noise, cost=.1):
    samples = evaluation.get('samples', {})
    if not isinstance(samples, dict) or not samples: raise ValueError('saved column-array samples unavailable')
    if int(evaluation.get('n_independent_histories', -1)) != 1: raise ValueError('conditional draws must belong to one history')
    R = int(evaluation.get('n_noise', -1))
    if R != int(expected_noise): raise ValueError('conditional-noise denominator differs from frozen count')
    vectors = {k: np.asarray(v, float) for k, v in samples.items() if k != 'posterior'}
    if any(v.shape != (R,) or not np.isfinite(v).all() for v in vectors.values()):
        raise ValueError('all conditional metric columns must be finite [R] vectors')
    required = {'correct', 'confidence', 'true_rejected', 'unresolved', 'wrong_unique', 'brier', 'negative_log_true_probability'}
    if not required <= set(vectors): raise ValueError('required actual metric columns absent')
    vectors['error'] = 1-vectors['correct']; vectors['confidence_minus_correct'] = vectors['confidence']-vectors['correct']
    vectors['abstention_loss'] = cost*vectors['unresolved']+vectors['wrong_unique']
    post = np.asarray(samples.get('posterior'), float)
    if post.ndim != 2 or post.shape[0] != R or not np.isfinite(post).all() or np.any(post < 0) or not np.allclose(post.sum(axis=1), 1., atol=1e-9):
        raise ValueError('proper saved posterior[R,K] required')
    for field, value in evaluation.get('summary', {}).items():
        if field in vectors and not np.isclose(float(value), vectors[field].mean(), atol=1e-10, rtol=1e-9):
            raise ValueError('saved summary differs from conditional columns: '+field)
    return {k: float(vectors[k].mean()) if k in vectors else None for k in ACTUAL}, calibration_moments(vectors['confidence'], vectors['correct'])


def _history_metadata(record, spec):
    result = dict(spec=spec, index=spec['index'], seed=spec['seed'], status=record.get('status', 'missing_history'),
        true_name=record.get('true_name'), failures=record.get('failures', {}), failure=record.get('failure'),
        seconds=_number(record.get('seconds')), active_training_line_boundary=None)
    trained = next((r for r in record.get('training_result', {}).get('fits', []) if r.get('candidate') == record.get('true_name')), None)
    if trained is not None:
        masks = trained.get('diagnostics', {}).get('ratio_constraints', {}).get('active', {})
        result['active_training_line_boundary'] = bool(any(np.any(v) for v in masks.values())) if masks else None
    joint = record.get('prediction', {}).get('joint', {})
    if result['true_name'] in joint.get('candidate_updates', {}):
        update = joint['candidate_updates'][result['true_name']]
        moment = update.get('TV_posterior_future', {}); true = record.get('true_lines', {})
        if 'mean' in moment and 'cov' in moment and 'r' in true:
            m = len(true['r']); eta = np.asarray(moment['mean'], float)[:2*m]
            C = np.asarray(moment['cov'], float)[:2*m, :2*m]
            truth = np.column_stack((np.log(true['r']), np.log(np.asarray(true['x'])/true['r']))).ravel()
            r = np.exp(eta[0::2]); x = np.exp(eta[0::2]+eta[1::2])
            result['posterior_center_R_relative_error'] = float(np.linalg.norm(r-true['r'])/np.linalg.norm(true['r']))
            result['posterior_center_X_relative_error'] = float(np.linalg.norm(x-true['x'])/np.linalg.norm(true['x']))
            result['log_line_moment_Gaussian90_marginal_coverage'] = float(np.mean(abs(eta-truth) <= 1.6448536269514722*np.sqrt(np.maximum(np.diag(C), 0))))
            result['line_interval_scope'] = 'marginal Gaussian moment intervals in logR/log(X/R); bounded posterior is nonGaussian'
        result['true_candidate_integration'] = {key: update.get(key) for key in ('logZprior', 'logZT', 'logZTV')}
        for key in ('T_line_integration', 'TV_line_integration'):
            result['true_candidate_integration'][key] = {k: v for k, v in update.get(key, {}).items() if isinstance(v, (int, float, bool, str))}
        if result['true_name'] in joint.get('candidate_names', []):
            i = joint['candidate_names'].index(result['true_name'])
            result['joint_truth_log_mass'] = float(joint['log_prior'][i])
    return result


def collect_records(records, expected, *, methods=METHODS, models=MODELS):
    """Return explicit slots for failures and missing histories, never filter success."""
    by_id = {}; issues = []
    for record in records: by_id.setdefault(_identity(record.get('spec')), record)
    table, histories, operations, proposals, costs, integration_pairs = [], [], [], [], [], []
    for spec in expected:
        record = by_id.get(_identity(spec), dict(spec=spec, status='missing_history'))
        metadata = _history_metadata(record, spec); histories.append(metadata)
        for name, pair in record.get('independent_paired', {}).items():
            if '_minus_' not in name or '__' not in name: continue
            comparison, model = name.rsplit('__', 1)
            left, right = comparison.split('_minus_', 1)
            integration_pairs.append(dict(index=spec['index'], method=left, baseline=right, model=model,
                risk_delta=_number(pair.get('risk')), paired_risk_MC_SE=_number(pair.get('risk_se')),
                scope=pair.get('se_scope'), noise_pairing='saved common integration noise, not independent histories'))
        for family, seconds in record.get('planning_seconds', {}).items():
            dependencies = ('square',) if family == 'ranked' or family == 'fixed_bc' else ('square', 'ranked') if family in ('joint_oldbc', 'joint_bc') else ()
            observed_sum = float(seconds)+sum(float(record['planning_seconds'].get(d, 0)) for d in dependencies)
            costs.append(dict(index=spec['index'], family=family, observed_family_seconds=float(seconds),
                observed_dependency_sum_seconds=observed_sum, dependencies=list(dependencies),
                shared_objective_cache=True, cold_standalone_cost_measured=False))
        for family, domains in record.get('planning_logs', {}).items():
            for domain, log in domains.items():
                for event in log.get('events', []):
                    milp = event.get('milp', {})
                    proposals.append(dict(index=spec['index'], family=family, domain=domain,
                        iteration=event.get('iteration'), solver_success=bool(milp.get('success')),
                        accepted=bool(event.get('accepted')), failure=milp.get('failure'),
                        full_AC_failure=event.get('full_AC_failure'), parent_risk=event.get('parent_risk'),
                        proposed_risk=event.get('proposed_risk'),
                        **{k: milp.get(k) for k in ('solver_incumbent', 'solver_best_bound', 'solver_relative_gap',
                            'proxy_value', 'proxy_uniform_error', 'proxy_absolute_gap', 'stop_reason', 'seconds')}))
        R = int(record.get('config', {}).get('actual_noise_repeats', 32)); cost = float(record.get('config', {}).get('abstention_cost', .1))
        for method in methods:
            saved = record.get('methods', {}).get(method); action = record.get('actions', {}).get(saved.get('action_key'), {}) if saved else {}
            if saved and saved.get('operation'):
                operation = saved['operation']; decoded = saved.get('decoded', {})
                charge, discharge = np.asarray(decoded.get('charge_kw', []), float), np.asarray(decoded.get('discharge_kw', []), float)
                dt = float(record.get('operation_config', {}).get('dt_hours', .25))
                operations.append(dict(index=spec['index'], method=method, feasible=bool(operation.get('feasible')),
                    moves=_number(operation.get('moves')), charge_kwh=float(charge.sum()*dt), discharge_kwh=float(discharge.sum()*dt),
                    throughput_kwh=_number(operation.get('throughput_kwh')), driver_cost=_number(operation.get('driver_cost')),
                    driving_kwh=_number(operation.get('travel_energy_kwh')),
                    terminal_energy_kwh=_number(operation.get('energy_kwh', [None])[-1])))
            wanted = models if method in INFERENCE_METHODS else ('joint',)
            for model in wanted:
                predicted = action.get('predictions', {}).get(model, {}).get('independent_score', {})
                for regime in REGIMES:
                    evaluation = action.get('actual', {}).get(regime, {}).get('models', {}).get(model)
                    row = dict(index=spec['index'], seed=spec['seed'], stratum_id=spec.get('stratum_id'),
                        bank=spec.get('bank'), H=spec.get('H'), excitation=spec.get('excitation'), innovation=spec.get('innovation'),
                        active_boundary=metadata['active_training_line_boundary'], status=metadata['status'],
                        method=method, model=model, regime=regime, true_name=record.get('true_name'),
                        action_key=saved.get('action_key') if saved else None, available=False,
                        n_conditional_noise=0, n_independent_histories=1,
                        native_planning_model='fixed' if method.startswith('fixed_bc__') else 'precision' if method.startswith('square') else 'joint')
                    row.update({k: None for k in ACTUAL}); row.update({'predicted_'+k: _number(predicted.get(k)) for k in PREDICTED})
                    row['predicted_risk_se'] = _number(predicted.get('risk_se'))
                    if evaluation is not None:
                        try:
                            values, bins = actual_metrics(evaluation, R, cost)
                            row.update(values, available=True, n_conditional_noise=R, _calibration=bins)
                        except (ValueError, TypeError, KeyError, IndexError) as exc:
                            issues.append(dict(index=spec['index'], method=method, model=model, regime=regime, reason=str(exc)))
                    if not row['available']:
                        row['missing_reason'] = ('method_missing' if not saved else
                            action.get('prediction_failures', {}).get(model, record.get('failure', 'actual model/regime absent')))
                    table.append(row)
    return dict(coverage=coverage(expected, [r.get('spec') for r in records]), histories=histories,
        rows=table, operations=operations, proposals=proposals, costs=costs, issues=issues,
        independent_integration_pairs=integration_pairs)


def _groups(histories):
    result = {'all': [h['index'] for h in histories]}
    for key in ('bank', 'H', 'excitation', 'innovation'):
        values = sorted({h['spec'].get(key) for h in histories}, key=str)
        for value in values: result[key+'='+str(value)] = [h['index'] for h in histories if h['spec'].get(key) == value]
    for value in (False, True):
        ids = [h['index'] for h in histories if h.get('active_training_line_boundary') is value]
        if ids: result['training_boundary='+str(value)] = ids
    return result


def summarize(collected, *, bootstrap=1000):
    table, histories = collected['rows'], collected['histories']; summary = []
    by_key = {(r['index'], r['method'], r['model'], r['regime']): r for r in table}
    configurations = sorted({(r['method'], r['model']) for r in table})
    strata = {h['index']: h['spec'].get('stratum_id', h['index']) for h in histories}
    for group, indices in _groups(histories).items():
        for method, model in configurations:
            for regime in REGIMES+('paired_regime_mean',):
                chosen = []
                for i in indices:
                    parts = [by_key[(i, method, model, r)] for r in REGIMES] if regime == 'paired_regime_mean' else [by_key[(i, method, model, regime)]]
                    values = {m: float(np.mean([r[m] for r in parts])) if all(r[m] is not None for r in parts) else None for m in ACTUAL}
                    values.update({'predicted_'+m: parts[0]['predicted_'+m] for m in PREDICTED})
                    values['_calibration'] = (None if len(parts) != 1 else parts[0].get('_calibration'))
                    chosen.append(values)
                metrics = {m: history_estimate([r[m] for r in chosen], bootstrap=bootstrap,
                    bounds=BOUNDS.get(m), strata=[strata[i] for i in indices]) for m in ACTUAL}
                prediction = {m: history_estimate([r['predicted_'+m] for r in chosen], bootstrap=bootstrap) for m in PREDICTED}
                bins = aggregate_calibration([r['_calibration'] for r in chosen], len(indices)) if regime in REGIMES else None
                summary.append(dict(group=group, method=method, model=model, regime=regime,
                    metrics=metrics, declared_Gaussian_integral=prediction, confidence_calibration=bins))
    return summary


def planned_contrasts():
    result = []
    for domain in DOMAINS:
        for other in ('square', 'square_budget', 'ranked', 'joint_oldbc', 'fixed_bc'):
            result.append(('joint_bc__'+domain, 'joint', other+'__'+domain, 'joint', 'policy'))
        result += [('ranked__'+domain, 'joint', 'square__'+domain, 'joint', 'initial_plan_ranking'),
            ('fixed_bc__'+domain, 'fixed', 'square__'+domain, 'fixed', 'fixed_native_policy')]
        for parent in PARENTS[domain]: result.append(('joint_bc__'+domain, 'joint', 'joint_bc__'+parent, 'joint', 'nested_domain'))
    for method in INFERENCE_METHODS:
        for other in MODELS[1:]: result.append((method, 'joint', method, other, 'same_action_inference'))
    return result


def paired_statistics(collected, *, bootstrap=1000):
    table = {(r['index'], r['method'], r['model'], r['regime']): r for r in collected['rows']}
    contrasts, details = [], []
    indices = [h['index'] for h in collected['histories']]
    strata = [h['spec'].get('stratum_id', h['index']) for h in collected['histories']]
    integration = {(r['index'], r['method'], r['baseline'], r['model']): r
        for r in collected.get('independent_integration_pairs', [])}
    for left, lm, right, rm, kind in planned_contrasts():
        if not any(k[1:3] == (left, lm) for k in table) or not any(k[1:3] == (right, rm) for k in table): continue
        for regime in REGIMES:
            rows = []
            for i in indices:
                a, b = table.get((i, left, lm, regime)), table.get((i, right, rm, regime))
                if a is None or b is None: continue
                row = dict(index=i, method=left, model=lm, baseline=right, baseline_model=rm, regime=regime,
                    kind=kind, same_action=a['action_key'] is not None and a['action_key'] == b['action_key'])
                row.update({m: None if a[m] is None or b[m] is None else a[m]-b[m] for m in ACTUAL})
                row['predicted_risk_delta'] = (None if lm != rm or a['predicted_risk'] is None or b['predicted_risk'] is None
                                              else a['predicted_risk']-b['predicted_risk'])
                pair = integration.get((i, left, right, lm), {}) if lm == rm else {}
                row['saved_paired_risk_MC_SE'] = pair.get('paired_risk_MC_SE')
                rows.append(row); details.append(row)
            if len(rows) != len(indices): continue
            contrasts.append(dict(method=left, model=lm, baseline=right, baseline_model=rm, regime=regime, kind=kind,
                expected_histories=len(indices), same_action_histories=sum(r['same_action'] for r in rows),
                differences={m: history_estimate([r[m] for r in rows], bootstrap=bootstrap, strata=strata) for m in ACTUAL},
                predicted_risk_delta=history_estimate([r['predicted_risk_delta'] for r in rows], bootstrap=bootstrap, strata=strata),
                direction='method minus baseline; lower error/Brier/logscore/exclusion is favorable'))
    return contrasts, details


def _fmt(value): return '—' if value is None else f'{value:.4g}'
def _pct(value): return '—' if value is None else f'{100*value:.1f}%'


def markdown(result):
    count = result['coverage']; rows = result['summary']
    lines = ['# 线路响应不确定性：冻结流程统计', '',
        f"阶段：{result['phase']}；记录{count['present_records']}/{count['expected_histories']}；状态分母：{json.dumps(result['status_counts'], ensure_ascii=False)}。",
        '每项指标先在同一历史内平均计量噪声，再跨历史等权汇总。regular/shift分别报告；两者的平均仍以同一历史为单位。', '',
        '## 全部计划：联合线路模型评价', '',
        '|方法|情景|可用历史|MAP正确|Brier|真标签负log概率|90%集合排真|未决|平均最大后验|电压90%区间覆盖|',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        if row['group'] != 'all' or row['model'] != 'joint' or row['regime'] not in REGIMES: continue
        m = row['metrics']; error = m['error']['mean']
        lines.append('|'+row['method']+'|'+row['regime']+'|'+str(m['error']['available_histories'])+'|'+
            _pct(None if error is None else 1-error)+'|'+_fmt(m['brier']['mean'])+'|'+_fmt(m['negative_log_true_probability']['mean'])+'|'+
            '|'.join(_pct(m[k]['mean']) for k in ('true_rejected', 'unresolved', 'confidence', 'voltage_interval90_coverage'))+'|')
    lines += ['', '## 相同动作的推断对照', '',
        'fixed为旧固定线路模型；joint保留线路及其与源/负荷的共同不确定性。其余模型使用同一保存动作作概率消融。', '',
        '|方法|推断模型|情景|MAP正确|Brier|负log概率|排真|电压区间覆盖|', '|---|---|---|---:|---:|---:|---:|---:|']
    for row in rows:
        if row['group'] != 'all' or row['method'] not in ('constant', 'square__mobile_bidir', 'joint_bc__mobile_bidir') or row['regime'] != 'regular': continue
        m = row['metrics']; e = m['error']['mean']
        lines.append('|'+row['method']+'|'+row['model']+'|regular|'+_pct(None if e is None else 1-e)+'|'+
            _fmt(m['brier']['mean'])+'|'+_fmt(m['negative_log_true_probability']['mean'])+'|'+_pct(m['true_rejected']['mean'])+'|'+_pct(m['voltage_interval90_coverage']['mean'])+'|')
    lines += ['', '区间覆盖采用真候选的条件预测均值和方差，另于拓扑后验集合覆盖。固定线路策略fixed_bc原本按fixed风险规划，其joint重评与fixed原模型结果分别保存。', '',
        '## 统计与费用口径', '',
        'summary.json保留按银行、训练长度、激励、创新与训练边界分组的指标、历史bootstrap区间、可估计时的格内分层标准误，以及缺失结果的范围。校准每格一条时，格内方差无法估计。bootstrap区间描述本实验条件混合分布；它不构成所有配电网的覆盖保证。',
        'paired.csv先按同一历史计算策略或推断差，再进行历史层汇总。旧/新预测模型的风险值分属不同分布，不直接解释为同目标改进。',
        'costs.csv保留各family的观测耗时及依赖费用之和；不同family共用缓存，因此这些数值不称冷启动独立成本。proposals.csv保留MILP失败、接受及求解上下界。',
        '声明Gaussian内部积分直接读取冻结independent_score，涉及随机候选混合分布；物理校准读取固定真结构与轨迹下的新计量噪声。未保存的内部90%集合/分箱保持缺失，后续独立审计可补齐。', '',
        f"数据问题{len(result['issues'])}项；未启动新拟合、规划、潮流或物理仿真。", '']
    return '\n'.join(lines)


def _csv(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row if not k.startswith('_')))
    with Path(path).open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(serial(v), ensure_ascii=False) if isinstance(v, (dict, list, tuple)) else v
                             for k, v in row.items() if k in fields})


def run(directory, phase, output, *, bootstrap=1000):
    began = perf_counter(); directory, output = Path(directory), Path(output)
    if (output/'summary.json').exists(): raise FileExistsError(output/'summary.json')
    frozen = read(directory/'freeze.json'); expected = frozen['specifications'][phase]
    records, files = [], []
    for path in sorted((directory/phase).glob('history_*.json.gz')):
        row = read(path)
        if row.get('policy_id') != frozen['policy_id']: raise ValueError('record/freeze policy mismatch')
        records.append(lightweight_record(row)); files.append(dict(file=str(path.resolve()), sha256=sha256(path.read_bytes()).hexdigest()))
        del row
    collected = collect_records(records, expected); summary = summarize(collected, bootstrap=bootstrap)
    contrasts, pair_rows = paired_statistics(collected, bootstrap=bootstrap)
    result = dict(schema='ev_bayes_response_history_statistics_v1', phase=phase, policy_id=frozen['policy_id'],
        coverage=collected['coverage'], status_counts=dict(Counter(h['status'] for h in collected['histories'])),
        histories=collected['histories'], summary=summary, paired=contrasts, issues=collected['issues'],
        expected_methods=list(METHODS), expected_models=list(MODELS), inference_methods=list(INFERENCE_METHODS),
        proposal_counts=dict(proposals=len(collected['proposals']), failed=sum(not r['solver_success'] for r in collected['proposals']),
            full_AC_failed=sum(bool(r['full_AC_failure']) for r in collected['proposals']),
            accepted=sum(r['accepted'] for r in collected['proposals'])),
        input_files=files, source_freeze_sha256=sha256((directory/'freeze.json').read_bytes()).hexdigest(),
        tool_sha256=sha256(Path(__file__).read_bytes()).hexdigest(), bootstrap_repetitions=bootstrap,
        bootstrap_seed=2026119001, new_fits=0, new_plans=0, new_AC_calls=0, new_simulations=0,
        inference_scope='moment-Gaussian future density after bounded local-Gaussian history model',
        interval_scope='history clusters; conditional noise repetitions are not independent histories', seconds=perf_counter()-began)
    output.mkdir(parents=True, exist_ok=True)
    (output/'summary.json').write_text(json.dumps(serial(result), ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    (output/'results.md').write_text(markdown(result), encoding='utf-8')
    for name, rows in (('history_metrics.csv', collected['rows']), ('paired.csv', pair_rows),
        ('operations.csv', collected['operations']), ('proposals.csv', collected['proposals']), ('costs.csv', collected['costs']),
        ('independent_integration_pairs.csv', collected['independent_integration_pairs'])):
        _csv(output/name, rows)
    print('RESPONSE_ANALYSIS_SAVED', phase, result['coverage'], result['status_counts'], flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True); parser.add_argument('--output', required=True)
    parser.add_argument('--phase', choices=('development', 'calibration', 'evaluation'), default='evaluation')
    parser.add_argument('--bootstrap', type=int, default=1000); args = parser.parse_args()
    if args.bootstrap < 100: parser.error('--bootstrap must be >=100')
    run(args.input, args.phase, args.output, bootstrap=args.bootstrap)


if __name__ == '__main__': main()
