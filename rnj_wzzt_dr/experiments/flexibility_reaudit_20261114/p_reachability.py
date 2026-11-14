"""Independent pure-P power-box bounds on the completed long-run forecasts.

No optimization, AC, fitting, or future observation is performed. Q actions are
fixed to zero, while background Q remains part of the sealed forecast. The 25%
P domain is constructed afresh, including its forecast-export PV restriction.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
from pathlib import Path
import platform
from statistics import NormalDist

import numpy as np

from ..measured_pq_flex_20261114 import domain as p_domain
from ..reactive_flex_20261114 import domain as pq_domain


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def reference(path, expected=None):
    path = Path(path).resolve()
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if expected is not None and digest != expected:
        raise ValueError(f'Sealed source changed: {path}')
    return dict(path=str(path), sha256=digest)


def checked(ref):
    return reference(ref['path'], ref['sha256'])


def box_upper(d0, effect, lower, upper, repeat, sigma2):
    """A separate output coordinate may use its own favourable box endpoint."""
    lo = np.repeat(np.asarray(lower, float), repeat, axis=0)
    hi = np.repeat(np.asarray(upper, float), repeat, axis=0)
    if np.any(lo > hi) or sigma2 <= 0:
        raise ValueError('Expected a valid resource box and positive variance')
    middle = d0 + ((lo + hi) / 2) @ effect.T
    radius = ((hi - lo) / 2) @ np.abs(effect).T
    squared_max = np.square(np.abs(middle) + radius)
    positive, negative = np.maximum(effect, 0), np.minimum(effect, 0)
    left = d0 + lo @ positive.T + hi @ negative.T
    right = d0 + hi @ positive.T + lo @ negative.T
    endpoint = np.maximum(left * left, right * right)
    error = float(np.max(np.abs(squared_max - endpoint), initial=0.))
    if not np.allclose(squared_max, endpoint, rtol=1e-12, atol=1e-15):
        raise ArithmeticError('Centre-radius and independent endpoint formulas disagree')
    return float(squared_max.sum() / (2 * sigma2)), error


def check_small_box():
    """An explicit four-corner example checks the per-output relaxation."""
    d0 = np.array([[.3, -.2]])
    effect = np.array([[1., -2.], [-.5, 1.5]])
    lower, upper = np.array([[-1., 0.]]), np.array([[2., .5]])
    corners = np.asarray(list(itertools.product(*zip(lower[0], upper[0]))))
    squared = np.square(d0 + corners @ effect.T)
    bound, error = box_upper(d0, effect, lower, upper, 1, 1.)
    exact_separate = float(squared.max(axis=0).sum() / 2)
    best_common_corner = float(squared.sum(axis=1).max() / 2)
    zero, _ = box_upper(d0, effect, np.zeros_like(lower), np.zeros_like(upper), 1, 1.)
    if abs(bound - exact_separate) > 1e-12 or bound < best_common_corner - 1e-12:
        raise ArithmeticError('Explicit corner enumeration does not support the bound')
    if abs(zero - float(np.square(d0).sum() / 2)) > 1e-12:
        raise ArithmeticError('Zero-width box changed the zero-action distance')
    return dict(passed=True, corner_count=len(corners), separate_output_upper=bound,
                largest_common_corner=best_common_corner, endpoint_identity_error=error)


def sealed_forecast(folder, score_ref):
    seal_ref = reference(Path(folder) / 'seal.json')
    seal = read(seal_ref['path'])
    planner_ref = checked(seal['planner'])
    actual_score_ref = checked(seal['score'])
    if actual_score_ref != score_ref:
        raise ValueError('Entry and sealed planner use different score sources')
    with np.load(planner_ref['path'], allow_pickle=False) as z:
        forecast = z['forecast'].copy()  # Deliberately do not read saved controls.
    return forecast, dict(seal=seal_ref, planner=planner_ref,
                          score=actual_score_ref, protocol=checked(seal['protocol']))


def run(directory, prior_result, base_kw=200.):
    directory = Path(directory).resolve()
    protocol_ref = reference(directory / 'protocol.json')
    protocol = read(protocol_ref['path'])
    complete_ref = reference(directory / 'complete.json')
    complete = read(complete_ref['path'])
    if complete.get('status') != 'complete' or not complete.get('source_unchanged'):
        raise ValueError('Expected a completed, source-unchanged original run')
    finished_ref = checked(complete['main'])
    finished = read(finished_ref['path'])
    if finished['failed_tasks'] or finished['completed_forecasts'] != finished['total_forecasts']:
        raise ValueError('The full original forecast batch is not complete')
    inputs_ref = checked(protocol['inputs'])
    entries = [e for e in read(inputs_ref['path']) if e['resource_variant'] == 'registered_plus1']
    if len(entries) != 1 or protocol['forecasts'] != 4:
        raise ValueError('This diagnostic is locked to the one-case, four-forecast plus1 run')
    entry = entries[0]
    score_ref, context_ref = checked(entry['score']), checked(entry['resource_context'])
    score, context = read(score_ref['path']), read(context_ref['path'])
    H = np.asarray([p['H'] for p in score['parameters']], float)
    history = np.asarray([p['history_sse'] for p in score['parameters']], float)
    sigma2 = float(score['sigma2'])
    target = int(score['map_index'])
    nearest = min((j for j in range(len(H)) if j != target), key=lambda j: (history[j], j))
    weights = np.exp(-(history - history.min()) / (2 * sigma2))
    weights /= weights.sum()
    order = np.argsort(-weights, kind='stable')
    cutoff = min(int(np.searchsorted(np.cumsum(weights[order]), .99)), len(order) - 1)
    targets = [int(i) for i in order if weights[i] >= weights[order[cutoff]] - 1e-14]
    if target not in targets or nearest not in targets:
        raise ValueError('This case requires both members of the near pair in the target set')
    h = -abs(float((history[nearest] - history[target]) / (2 * sigma2)))
    alpha = .1 / (len(H) - 1)
    z_quantile = NormalDist().inv_cdf(1 - alpha)
    N = len(score['labels'])
    tie = max(1e-7, (score['cells'] + 96 * N) * score['reference_mse']
              * score['validation_tie'] / (2 * sigma2))
    if h >= tie:
        raise ValueError('The single high-root threshold is not general outside h < tie')
    threshold = lambda margin: .5 * (z_quantile + math.sqrt(z_quantile**2 + 2 * (margin - h)))**2
    required, native_required = threshold(tie), threshold(tie + 1e-8)
    D = H[target] - H[nearest]

    old_result_ref = reference(prior_result)
    old_result = read(old_result_ref['path'])
    old_protocol_ref = checked(old_result['protocol'])
    old_protocol = read(old_protocol_ref['path'])
    old_inputs_ref = checked(old_result['inputs'])
    old_entries = [e for e in read(old_inputs_ref['path']) if e['id'] == entry['original_id']]
    if len(old_entries) != 1:
        raise ValueError('Expected one matching original pure-P input')
    old_entry = old_entries[0]
    old_score_ref, old_context_ref = checked(old_entry['score']), checked(old_entry['resource_context'])
    if old_score_ref['sha256'] != score_ref['sha256'] or old_context_ref['sha256'] != context_ref['sha256']:
        raise ValueError('The cited old range has different frozen models or registration')
    rows, forecast_comparisons = [], []
    max_identity_error = 0.
    for draw in range(protocol['forecasts']):
        folder = directory / 'fraction0.10' / entry['id'] / f'forecast{draw:02d}'
        forecast, sealed = sealed_forecast(folder, score_ref)
        old_folder = Path(old_result['run_directory']) / 'fraction0.10' / old_entry['id'] / f'forecast{draw:02d}'
        old_forecast, old_sealed = sealed_forecast(old_folder, old_score_ref)
        forecast_comparisons.append(dict(forecast=draw, current=sealed, previous=old_sealed,
            bitwise_equal=bool(np.array_equal(forecast, old_forecast)),
            maximum_absolute_difference_pu=float(np.max(np.abs(forecast - old_forecast)))))
        original = pq_domain.PQIncrementCase(context, score['labels'], forecast, base_kw,
                                             q_resources=entry['q_resources'], fraction=.1)
        d0 = forecast @ D.T
        J0 = float(np.square(d0).sum() / (2 * sigma2))
        baseline_p = None
        for fraction in (.1, .25):
            case = p_domain.IncrementCase(context, score['labels'], forecast, base_kw, fraction=fraction)
            if not np.all(case.qratio == 0):
                raise ValueError('Pure-P diagnostic cannot include reactive response ratios')
            effect = D[:, case.node] / base_kw
            if fraction == .1:
                baseline_p = case
                if (case.resources != original.p_case.resources
                    or not np.array_equal(case.lower, original.lower[:, :original.P_R])
                    or not np.array_equal(case.upper, original.upper[:, :original.P_R])
                    or not np.array_equal(effect, D @ original.B[:, :original.P_R] / base_kw)):
                    raise ArithmeticError('Reconstructed pure-P domain differs from original P columns')
            bound, identity_error = box_upper(d0, effect, case.lower, case.upper, case.repeat, sigma2)
            max_identity_error = max(max_identity_error, identity_error)
            if bound < J0 - 1e-12:
                raise ArithmeticError('Zero is in the resource box but exceeds the bound')
            naive_upper = baseline_p.upper * (fraction / .1)
            pv = [j for j, r in enumerate(case.resources) if r['kind'] == 'pv']
            pv_difference = np.abs(case.upper[:, pv] - naive_upper[:, pv])
            rows.append(dict(forecast=draw, contract_fraction=fraction, sealed_sources=sealed,
                J_at_zero=J0, J_box_upper=bound, gain_box_upper=bound-J0,
                J_required_nominal=required, J_required_native_constraint=native_required,
                upper_over_nominal_required=bound/required,
                absolute_requirement_infeasible_by_bound=bool(bound < required - 1e-8),
                old_gain2_requirement_infeasible_by_bound=bool(bound-J0 < 2 - 1e-8),
                Q_action_fixed_zero=True, background_Q_retained=True,
                original_P_domain_exact_match=fraction == .1,
                domain_constructed_afresh=True,
                resource_ids=[r['id'] for r in case.resources],
                resource_node_labels=[r['node_label'] for r in case.resources],
                resource_contract_kw=[r['contract_kw'] for r in case.resources],
                power_lower_kw=case.lower.tolist(), power_upper_kw=case.upper.tolist(),
                PV_cells_differing_from_naive_scale=int(np.sum(pv_difference > 1e-12)),
                PV_max_difference_from_naive_scale_kw=float(pv_difference.max(initial=0.)),
                centre_radius_endpoint_identity_error=identity_error))
    summary = []
    for fraction in (.1, .25):
        selected = [row for row in rows if row['contract_fraction'] == fraction]
        summary.append(dict(contract_fraction=fraction, forecast_problems=len(selected),
            J_at_zero_range=[min(r['J_at_zero'] for r in selected), max(r['J_at_zero'] for r in selected)],
            J_box_upper_range=[min(r['J_box_upper'] for r in selected), max(r['J_box_upper'] for r in selected)],
            gain_box_upper_range=[min(r['gain_box_upper'] for r in selected), max(r['gain_box_upper'] for r in selected)],
            absolute_infeasible_by_bound=sum(r['absolute_requirement_infeasible_by_bound'] for r in selected),
            old_gain2_infeasible_by_bound=sum(r['old_gain2_requirement_infeasible_by_bound'] for r in selected),
            PV_cells_differing_from_naive_scale=sum(r['PV_cells_differing_from_naive_scale'] for r in selected)))
    source_paths = [Path(__file__), Path(p_domain.__file__), Path(pq_domain.__file__),
                    Path(__file__).parents[1]/'customer_flexibility_20261113'/'model.py',
                    Path(__file__).parents[1]/'absolute_identification_20261114'/'solver.py']
    return dict(created_utc=datetime.now(timezone.utc).isoformat(), source=reference(__file__),
        source_dependencies=[reference(path) for path in source_paths[1:]],
        environment=dict(python=platform.python_version(), numpy=np.__version__),
        run=str(directory), protocol=protocol_ref, completion=complete_ref,
        completed_fraction=finished_ref, inputs=inputs_ref, score=score_ref, resource_context=context_ref,
        case_id=entry['original_id'], variant=entry['resource_variant'], base_kw=base_kw,
        pair_representative=[target, nearest], pair_native=[score['models'][i]['index'] for i in (target, nearest)],
        targets=targets, historical_offset=h, alpha=alpha, normal_quantile=z_quantile,
        tie=tie, native_additional_margin=1e-8, J_required_nominal=required,
        J_required_native_constraint=native_required,
        conditions=dict(no_optimization=True, no_AC=True, no_refit=True, no_truth=True,
            no_actual_future_observations=True, Q='Action fixed to zero, forecast background Q retained',
            bound='Per-output centre-radius power-box bound; retains individual clock windows and forecast PV export caps, relaxes energy/terminal coupling and all evidence/mean guards',
            formula='J <= sum((abs(d0 + center @ F.T) + radius @ abs(F).T)**2)/(2*sigma2)',
            fractions='Fresh IncrementCase(fraction=0.1 or 0.25), not multiplied original PV bounds',
            interpretation='Below-threshold upper bound proves infeasibility of this declared frozen-forecast condition, not impossibility of physical recovery',
            no_purchase_attribution='P alone being insufficient does not prove P has no marginal value together with Q, or that zero P in a mixed optimum is uniquely explained by this bound'),
        prior_citation=dict(result=old_result_ref, protocol=old_protocol_ref, inputs=old_inputs_ref,
            seed=old_protocol['seed'], current_seed=protocol['seed'], score=old_score_ref,
            resource_context=old_context_ref, same_frozen_score=True, same_registration=True,
            old_summary=old_result['summary'], forecast_comparisons=forecast_comparisons,
            exact_old_range_not_reused_for_current_forecasts=True),
        arithmetic_check=check_small_box(), maximum_endpoint_identity_error=max_identity_error,
        summary=summary, rows=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=Path('outputs/flexibility_reaudit_long180_20261114_v1'))
    parser.add_argument('--prior-result', type=Path, default=Path('outputs/measured_pq_dfs_enriched_reachability_20261114_v1.json'))
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = run(args.run, args.prior_result)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(output=reference(args.out), summary=result['summary'],
        nominal_J_required=result['J_required_nominal'], native_J_required=result['J_required_native_constraint'],
        previous_seed=result['prior_citation']['seed'], current_seed=result['prior_citation']['current_seed'],
        arithmetic_check=result['arithmetic_check'])))


if __name__ == '__main__':
    main()
