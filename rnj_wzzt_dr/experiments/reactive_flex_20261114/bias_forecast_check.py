"""History-only bias forecasts tested against already sealed P/Q actions.

The primary forecast is the exact existing validation multiplicative mean.
Separate P/Q scales plus an intercept are a fixed side diagnostic, not a new
identifier or selected planning policy. All parameters use history only;
future clean/observed voltages are opened only to assess frozen predictions.
No optimization, AC solve, model-library mutation, or action rescaling run.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from ... import io
from ...observations import Observation, load
from ..flexibility_sweep_20261113.fitting import calibrated_means


def arrays(path):
    with np.load(path, allow_pickle=False) as source:
        return {key: source[key].copy() for key in source.files}


def checked(ref):
    if io.ref(ref['path'])['sha256'] != ref['sha256']:
        raise ValueError('Changed sealed input: '+ref['path'])
    return Path(ref['path'])


def power(obs):
    return np.c_[obs.p, obs.q]


def section(obs, start, stop):
    return Observation(obs.p[start:stop], obs.q[start:stop], obs.y[start:stop], obs.labels)


def frozen_matrices(native, n):
    result = []
    for row in native['frozen_candidates']:
        H = np.zeros((n, 2*n))
        for support, r, x in zip(row['point']['supports'], row['point']['r'], row['point']['x'], strict=True):
            H[np.ix_(support, support)] += r
            H[np.ix_(support, np.asarray(support)+n)] += x
        result.append(H)
    return np.asarray(result)


def existing_model(score, obs, native_ids):
    means = calibrated_means(score, obs)
    indices = {int(model['index']):i for i, model in enumerate(score['models'])}
    K = np.asarray([means[indices[i]] for i in native_ids])
    H = np.asarray([score['parameters'][indices[i]]['H'] for i in native_ids])
    factor = np.divide(np.sum(K*H, axis=2), np.sum(H*H, axis=2),
        out=np.ones(H.shape[:2]), where=np.sum(H*H, axis=2) > 0)
    return dict(K=K, c=np.zeros(K.shape[:2]), factors=factor,
        method='Existing one-factor-per-output mean: same factor scales all P and Q columns')


def separate_model(H, obs):
    """Per-output 3-parameter OLS, using standardized P/Q predictions."""
    n = len(obs.labels); K = np.empty_like(H); intercept = np.zeros(H.shape[:2]); diagnostics = []
    for model, h in enumerate(H):
        part = np.stack((obs.p@h[:, :n].T, obs.q@h[:, n:].T), axis=2)
        rows = []
        for node in range(n):
            x = part[:, node]; mean = x.mean(0); std = x.std(0)
            safe = np.where(std > 1e-14, std, 1.)
            design = np.c_[np.ones(len(x)), (x-mean)/safe]
            coef, _, rank, singular = np.linalg.lstsq(design, obs.y[:, node], rcond=None)
            slopes = coef[1:]/safe; constant = coef[0]-slopes@mean
            K[model, node, :n] = slopes[0]*h[node, :n]
            K[model, node, n:] = slopes[1]*h[node, n:]
            intercept[model, node] = constant
            rows.append(dict(label=int(obs.labels[node]), a_P=slopes[0], a_Q=slopes[1], intercept=constant,
                rank=int(rank), standardized_condition=np.linalg.cond(design),
                raw_condition=np.linalg.cond(np.c_[np.ones(len(x)), x]),
                P_prediction_std=std[0], Q_prediction_std=std[1], singular_values=singular))
        diagnostics.append(rows)
    return dict(K=K, c=intercept, per_model_nodes=diagnostics,
        method='Fixed history-only per-output OLS: a_P*H_P*P+a_Q*H_Q*Q+c; no ridge/model selection')


def bias(H, index, X, Y, sigma2):
    D = H[index]-H[1-index]
    return float(np.sum((Y-X@H[index].T)*(X@D.T))/sigma2)


def predicted_bias(H, model, index, X, sigma2):
    mean = X@model['K'][index].T+model['c'][index]
    return bias(H, index, X, mean, sigma2)


def prediction_delta(H, model, X0, X1, sigma2):
    return [predicted_bias(H, model, i, X1, sigma2)-predicted_bias(H, model, i, X0, sigma2)
            for i in range(2)]


def actual_values(H, X, Y, sigma2):
    J = float(np.sum((X@(H[0]-H[1]).T)**2)/(2*sigma2))
    b = [bias(H, i, X, Y, sigma2) for i in range(2)]
    sse = [float(np.sum((Y-X@h.T)**2)) for h in H]
    L = (sse[1]-sse[0])/(2*sigma2)
    if not np.allclose([J+b[0], J+b[1]], [L, -L], atol=1e-8, rtol=1e-8):
        raise ArithmeticError('Fixed-direction evidence identity failed')
    return dict(J=J, b=b, L=L)


def history_diagnostic(H, train, validation, sigma2):
    result = {}
    for name, obs in [('train', train), ('validation', validation)]:
        X = power(obs); centered = X-X.mean(0); singular = np.linalg.svd(centered, compute_uv=False)
        residual = np.asarray([obs.y-X@h.T for h in H])
        result[name] = dict(rows=len(X), residual_mean_by_output=residual.mean(1),
            residual_mean_sign_by_output=np.sign(residual.mean(1)), residual_mse=np.mean(residual**2, axis=(1, 2)),
            raw_centered_PQ_condition=singular[0]/singular[-1], raw_centered_PQ_singular_values=singular,
            Q_std=obs.q.std(0), P_std=obs.p.std(0),
            actual_observed_bias=[bias(H, i, X, obs.y, sigma2) for i in range(2)])
    return result


def history_holdout(H, model, obs, sigma2):
    X = power(obs); actual = actual_values(H, X, obs.y, sigma2)
    estimated = [predicted_bias(H, model, i, X, sigma2) for i in range(2)]
    m = np.asarray([X@model['K'][i].T+model['c'][i] for i in range(2)])
    original = X@(H[0]-H[1]).T
    return dict(rows=len(X), actual_observed_b=actual['b'], predicted_b=estimated,
        bias_error=np.asarray(estimated)-actual['b'], actual_observed_L=actual['L'],
        predicted_conditional_L=[actual['J']+estimated[i] for i in range(2)],
        corrected_prediction_mse=np.mean((m-obs.y)**2, axis=(1, 2)),
        corrected_mean_separation_over_frozen=float(np.linalg.norm(m[0]-m[1])/max(np.linalg.norm(original), 1e-30)),
        calibration_diagnostics=model.get('per_model_nodes', model.get('factors')),
        note='Held-out historical measured Y includes meter error; neither future Y nor truth enters fit')


def coverage(train, shift, threshold=.05):
    X = power(train); centered = X-X.mean(0)
    _, singular, vt = np.linalg.svd(centered, full_matrices=False)
    weak = singular < threshold*singular[0]
    n = len(train.labels); Qshift = shift[:, n:]
    std = train.q.std(0); request = np.sqrt(np.mean(Qshift**2, axis=0))
    norm = float(np.sum(shift*shift))
    q_only = np.c_[np.zeros_like(Qshift), Qshift]; qnorm = float(np.sum(q_only*q_only))
    standardized = np.divide(request, std, out=np.zeros_like(request), where=std > 0)
    return dict(threshold_relative_singular_value=threshold, weak_dimensions=int(weak.sum()),
        total_dimensions=len(singular), raw_centered_condition=float(singular[0]/singular[-1]),
        action_weak_direction_energy_fraction=None if norm == 0 else float(np.sum((shift@vt[weak].T)**2)/norm),
        Q_action_weak_direction_energy_fraction=None if qnorm == 0 else float(np.sum((q_only@vt[weak].T)**2)/qnorm),
        Q_action_rms_over_historical_std=standardized,
        max_Q_action_rms_over_historical_std=float(standardized.max(initial=0.)),
        Q_action_rms_pu=request, historical_Q_std_pu=std,
        note='Geometric history coverage diagnostic; not a confidence or feasibility probability')


def map_action(config, labels, u, base):
    shift = np.zeros((96, 2*len(labels))); n = len(labels)
    for j, (resource, unit) in enumerate(zip(config['resources'], config['action_units'], strict=True)):
        node = labels.index(resource['node_label'])+(n if unit == 'kvar' else 0)
        shift[:, node] += np.repeat(u[:, j], 4)/base
    return shift


def sign(value):
    return int(np.sign(value)) if abs(value) > 1e-9 else 0


def safe(value):
    if isinstance(value, dict): return {str(k): safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [safe(v) for v in value]
    if isinstance(value, np.ndarray): return safe(value.tolist())
    if isinstance(value, np.generic): return safe(value.item())
    if isinstance(value, float) and not np.isfinite(value): return None
    return value


def run(run_paths, output):
    output = Path(output)
    if output.exists(): raise FileExistsError(output)
    histories, rows, paths, sources = {}, [], [], []
    for run_path in map(Path, run_paths):
        protocol = io.read(run_path/'protocol.json'); complete = io.read(run_path/'complete.json')
        if complete['status'] != 'complete': raise ValueError('Require completed experiment')
        sources.append(dict(protocol=io.ref(run_path/'protocol.json'), complete=io.ref(run_path/'complete.json')))
        for entry in io.read(checked(protocol['inputs'])):
            history_key = entry['score']['sha256']
            if history_key not in histories:
                score = io.read(checked(entry['score'])); native = io.read(checked(entry['native_copy']))
                train, valid = load(checked(entry['files']['train.npz'])), load(checked(entry['files']['validation.npz']))
                H = frozen_matrices(native, len(train.labels))[[4, 2]]
                sigma2 = float(score['sigma2']); models = dict(
                    existing_validation=existing_model(score, valid, (4, 2)),
                    existing_train=existing_model(score, train, (4, 2)),
                    separate_PQ_validation=separate_model(H, valid), separate_PQ_train=separate_model(H, train))
                for model in models.values():
                    columns = []
                    for k, native_id in enumerate((4, 2)):
                        for j, label in enumerate(train.labels):
                            old, corrected = H[k, :, len(train.labels)+j], model['K'][k, :, len(train.labels)+j]
                            norm = np.linalg.norm(old)*np.linalg.norm(corrected)
                            columns.append(dict(native_index=native_id, injection_node_label=int(label),
                                response_projection_multiplier=float(old@corrected/max(old@old, 1e-30)),
                                response_direction_degrees=None if norm == 0 else float(np.degrees(np.arccos(
                                    np.clip(old@corrected/norm, -1., 1.)))),
                                interpretation='All-output pure-Q derivative column of history-only mean versus frozen H; no intercept or actual future voltage used'))
                    model['Q_derivative_columns'] = columns
                front, back = section(valid, 0, 48), section(valid, 48, 96)
                tests = dict(train_to_validation_existing=history_holdout(H, models['existing_train'], valid, sigma2),
                    train_to_validation_separate_PQ=history_holdout(H, models['separate_PQ_train'], valid, sigma2),
                    first48_to_last48_existing=history_holdout(H, existing_model(score, front, (4, 2)), back, sigma2),
                    last48_to_first48_existing=history_holdout(H, existing_model(score, back, (4, 2)), front, sigma2),
                    first48_to_last48_separate_PQ=history_holdout(H, separate_model(H, front), back, sigma2),
                    last48_to_first48_separate_PQ=history_holdout(H, separate_model(H, back), front, sigma2))
                histories[history_key] = dict(input_id=entry['original_id'], sigma2=sigma2,
                    native_order=[4, 2], labels=list(train.labels), source_refs={key: entry['files'][key] for key in ('train.npz', 'validation.npz')},
                    score=entry['score'], native=entry['native_copy'], diagnostic=history_diagnostic(H, train, valid, sigma2),
                    historical_holdout=tests, models=models, H=H, train=train)
            data = histories[history_key]; H, sigma2, models = data['H'], data['sigma2'], data['models']
            labels = data['labels']; train = data['train']
            for folder in sorted((run_path/'fraction0.10'/entry['id']).glob('forecast*')):
                seal = io.read(folder/'seal.json'); planner = arrays(checked(seal['planner']))
                checked(seal['plans']); checked(seal['score']); checked(seal['protocol'])
                config = io.read(folder/'resource_case.json'); X0 = planner.pop('forecast')
                exact = {}
                for strategy in planner:
                    f = folder/'exact_requested'/strategy/'clean_observation.npz'
                    if f.exists():
                        a = arrays(f); exact[strategy] = (np.c_[a['P'], a['Q']], a['Y'])
                for strategy, u in planner.items():
                    shift = map_action(config, labels, u, 200.)
                    row = dict(run=run_path.name, seed=protocol['seed'], variant=entry['resource_variant'],
                        forecast=folder.name, strategy=strategy, nonzero=bool(np.any(np.abs(u) > 1e-9)),
                        planner_ref=seal['planner'], coverage=coverage(train, shift),
                        quarter_action_coverage=coverage(train, .25*shift),
                        history_predictions={name:dict(
                            requested_delta_b=prediction_delta(H, model, X0, X0+shift, sigma2),
                            quarter_requested_delta_b=prediction_delta(H, model, X0, X0+.25*shift, sigma2))
                            for name, model in models.items()})
                    if strategy in exact and 'no_dfs' in exact:
                        x0, y0 = exact['no_dfs']; x1, y1 = exact[strategy]
                        v0, v1 = actual_values(H, x0, y0, sigma2), actual_values(H, x1, y1, sigma2)
                        row['exact_actual_delta_b'] = np.asarray(v1['b'])-v0['b']
                        row['exact_actual_delta_J'] = v1['J']-v0['J']
                        for name, model in models.items():
                            row['history_predictions'][name]['exact_input_delta_b'] = prediction_delta(H, model, x0, x1, sigma2)
                    observations = []
                    for execution in sorted(folder.glob('execution*')):
                        f0 = execution/'no_dfs'/'observations.npz'; f1 = execution/strategy/'observations.npz'
                        if not f0.exists() or not f1.exists(): continue
                        a0, a1 = arrays(f0), arrays(f1)
                        x0, x1 = np.c_[a0['true_P'], a0['true_Q']], np.c_[a1['true_P'], a1['true_Q']]
                        y0, y1 = a0['true_root'][:, None]**2-a0['true_V']**2, a1['true_root'][:, None]**2-a1['true_V']**2
                        v0, v1 = actual_values(H, x0, y0, sigma2), actual_values(H, x1, y1, sigma2)
                        measured0 = actual_values(H, np.c_[a0['P'], a0['Q']], a0['Y'], sigma2)
                        measured1 = actual_values(H, np.c_[a1['P'], a1['Q']], a1['Y'], sigma2)
                        path = dict(run=run_path.name, variant=entry['resource_variant'], forecast=folder.name,
                            strategy=strategy, execution=execution.name, nonzero=row['nonzero'],
                            actual_delta_b=np.asarray(v1['b'])-v0['b'], actual_delta_J=v1['J']-v0['J'],
                            actual_clean_delta_L=v1['L']-v0['L'], actual_measured_delta_L=measured1['L']-measured0['L'],
                            predicted_requested_delta_b={name:row['history_predictions'][name]['requested_delta_b'] for name in models},
                            predicted_actual_input_delta_b={name:prediction_delta(H, model, x0, x1, sigma2) for name, model in models.items()})
                        observations.append(path); paths.append(path)
                    row['actual_paths'] = len(observations)
                    row['mean_actual_delta_b'] = np.mean([p['actual_delta_b'] for p in observations], axis=0)
                    row['mean_actual_delta_J'] = float(np.mean([p['actual_delta_J'] for p in observations]))
                    row['mean_actual_clean_delta_L'] = float(np.mean([p['actual_clean_delta_L'] for p in observations]))
                    row['mean_actual_measured_delta_L'] = float(np.mean([p['actual_measured_delta_L'] for p in observations]))
                    rows.append(row)
    summaries = []
    keys = sorted({(r['run'], r['variant'], r['strategy']) for r in rows})
    for run_name, variant, strategy in keys:
        plans = [r for r in rows if (r['run'], r['variant'], r['strategy']) == (run_name, variant, strategy) and r['nonzero']]
        executions = [p for p in paths if (p['run'], p['variant'], p['strategy']) == (run_name, variant, strategy) and p['nonzero']]
        summary = dict(run=run_name, variant=variant, strategy=strategy, nonzero_forecasts=len(plans),
            nonzero_paths=len(executions), mean_actual_delta_b=None if not plans else np.mean([r['mean_actual_delta_b'] for r in plans], axis=0), estimators={})
        for name in next(iter(histories.values()))['models']:
            item = dict(mean_predicted_delta_b=None if not plans else np.mean([
                r['history_predictions'][name]['requested_delta_b'] for r in plans], axis=0),
                mean_quarter_predicted_delta_b=None if not plans else np.mean([
                r['history_predictions'][name]['quarter_requested_delta_b'] for r in plans], axis=0))
            for i, label in enumerate(('H4_against_H2', 'H2_against_H4')):
                valid_plans = [r for r in plans if sign(r['mean_actual_delta_b'][i])]
                valid_paths = [p for p in executions if sign(p['actual_delta_b'][i])]
                item[label] = dict(forecast_sign_denominator=len(valid_plans),
                    forecast_sign_agree=sum(sign(r['history_predictions'][name]['requested_delta_b'][i]) == sign(r['mean_actual_delta_b'][i]) for r in valid_plans),
                    path_sign_denominator=len(valid_paths), path_sign_agree=sum(
                    sign(p['predicted_requested_delta_b'][name][i]) == sign(p['actual_delta_b'][i]) for p in valid_paths),
                    actual_input_path_sign_agree=sum(sign(p['predicted_actual_input_delta_b'][name][i]) == sign(p['actual_delta_b'][i]) for p in valid_paths))
            summary['estimators'][name] = item
        summaries.append(summary)
    for data in histories.values():
        del data['train']
    report = dict(created_utc=datetime.now(timezone.utc).isoformat(), sources=sources,
        scripts=[io.ref(__file__), io.ref(Path(__file__).parents[1]/'flexibility_sweep_20261113'/'fitting.py')],
        conditions=dict(native_direction_order=[4, 2], model_parameters='History P/Q/Y only; frozen H unchanged',
            existing_model_limitation='One output factor scales both P and Q coefficients identically; it cannot independently repair their compensating errors',
            split='train192 -> validation96; validation first48/local08-20 <-> last48/local20-08',
            diagnostic_selection='Existing means primary; separate P/Q plus intercept fixed side diagnostic; no future-based model selection',
            coverage_weak_singular_ratio=.05, sign_zero_tolerance=1e-9, base_kw=200.,
            quarter='Only evaluate the frozen mean formula at 0.25 of each saved action; no quarter AC result fabricated',
            actual='Saved clean AC and measured values are used only after predictions to assess direction',
            no_new_AC=True, no_optimization=True, no_identifier_refit=True),
        histories=histories, plans=rows, executions=paths, summary=summaries)
    io.write(output, safe(report))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); result = run(args.runs, args.output)
    print(dict(plans=len(result['plans']), execution_rows=len(result['executions']), output=str(args.output)))
