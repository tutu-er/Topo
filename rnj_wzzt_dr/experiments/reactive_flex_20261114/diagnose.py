"""Offline response-bias audit of completed reactive-flexibility executions.

No optimization, AC solve, parameter fitting, or resource change is performed.
Physical line parameters are used only here, after plans and executions finish.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from .placement import checked, read, ref
from ..measured_pq_flex_20261114.analyze import physical_response
from topology_identification_standalone import data as backend


def mean_stats(values):
    a = np.asarray([v for v in values if v is not None], float)
    return dict(n=len(a), mean=float(a.mean()) if len(a) else None,
                minimum=float(a.min()) if len(a) else None,
                maximum=float(a.max()) if len(a) else None)


def observation(path):
    with np.load(path, allow_pickle=False) as z:
        X = np.c_[z['true_P'], z['true_Q']]
        Y = z['true_root'][:, None]**2-z['true_V']**2
        observed_X = np.c_[z['P'], z['Q']]
        observed_Y = z['Y'].copy()
    return X, Y, observed_X, observed_Y


def evidence(X, Y, Ht, Hj, sigma2):
    # Direct squared-error subtraction, separate from the analytical split.
    return float((np.square(Y-X@Hj.T).sum()-np.square(Y-X@Ht.T).sum())/(2*sigma2))


def split(X, Y, Ht, Hj, physical, sigma2):
    d = X@(Ht-Hj).T
    return dict(J=float(np.square(d).sum()/(2*sigma2)),
                parameter=float(np.sum((X@(physical-Ht).T)*d)/sigma2),
                AC=float(np.sum((Y-X@physical.T)*d)/sigma2))


def physical_coefficients(case, labels):
    network = backend.load_case(case)
    order, parent, incoming = backend._radial_order(network)
    descendants = {bus: set() for bus in order}
    for k, bus in enumerate(labels):
        descendants[int(bus)].add(k)
    values = defaultdict(lambda: np.zeros(2))
    base = network['base_kv']**2/network['base_mva']
    for bus in reversed(order[1:]):
        group = tuple(sorted(descendants[bus]))
        if group:
            edge = incoming[bus]
            values[group] += np.array([2*edge['r_ohm']/base, 2*edge['x_ohm']/base])
        descendants[parent[bus]].update(descendants[bus])
    return values


def matrices(native, n):
    H, C, coefficients = [], [], []
    for item in native['frozen_candidates']:
        h, c = np.zeros((n, 2*n)), np.zeros((n, 2*n))
        values = defaultdict(lambda: np.zeros(2))
        point = item['point']
        for group, r, x in zip(point['supports'], point['r'], point['x'], strict=True):
            idx = np.ix_(group, group)
            h[:, :n][idx] += r
            h[:, n:][idx] += x
            values[tuple(group)] += [r, x]
            if len(group) == n:
                c[:, :n] += r
                c[:, n:] += x
        H.append(h); C.append(c); coefficients.append(values)
    return np.asarray(H), np.asarray(C), coefficients


def response_error(prediction, actual):
    squared_actual = float(np.square(actual).sum())
    error = prediction-actual
    return dict(actual_norm=float(np.sqrt(squared_actual)),
                predicted_norm=float(np.linalg.norm(prediction)),
                error_norm=float(np.linalg.norm(error)),
                error_RMS=float(np.sqrt(np.mean(error*error))),
                relative_error=float(np.linalg.norm(error)/np.sqrt(squared_actual)) if squared_actual > 1e-28 else None,
                projection_scale=float(np.sum(prediction*actual)/squared_actual) if squared_actual > 1e-28 else None)


def context(entry, target, competitor):
    score_ref, native_ref = checked(entry['score']), checked(entry['native_copy'])
    score, native = read(score_ref['path']), read(native_ref['path'])
    labels = list(map(int, score['labels'])); n = len(labels)
    H, C, coeff = matrices(native, n)
    frozen = np.asarray([p['H'] for p in score['native_bank']])
    if not np.allclose(H, frozen, atol=1e-14, rtol=1e-13):
        raise ArithmeticError('Independent support reconstruction differs from frozen bank')
    physical = physical_response(entry['case'], labels, entry['physical_cache']['network_hash'])
    true_coeff = physical_coefficients(entry['case'], labels)
    D, DC = H[target]-H[competitor], C[target]-C[competitor]
    groups = sorted(set(true_coeff)|set(coeff[target])|set(coeff[competitor]), key=lambda x:(len(x), x))
    coefficient_rows = [dict(node_labels=[labels[i] for i in g],
        physical_response=true_coeff.get(g, np.zeros(2)).tolist(),
        target_response=coeff[target].get(g, np.zeros(2)).tolist(),
        competitor_response=coeff[competitor].get(g, np.zeros(2)).tolist()) for g in groups]
    node_rows = []
    for k, label in enumerate(labels):
        fields = {}
        for channel, col in [('P', k), ('Q', k+n)]:
            fields[channel] = dict(physical_norm=float(np.linalg.norm(physical[:, col])),
                target=response_error(H[target, :, col], physical[:, col]),
                competitor=response_error(H[competitor, :, col], physical[:, col]),
                target_without_common=response_error(H[target, :, col]-C[target, :, col], physical[:, col]),
                competitor_without_common=response_error(H[competitor, :, col]-C[competitor, :, col], physical[:, col]),
                pair_difference_norm=float(np.linalg.norm(D[:, col])),
                common_difference_norm=float(np.linalg.norm(DC[:, col])))
        node_rows.append(dict(node_label=label, **fields))
    audit = dict(id=entry['id'], original_id=entry.get('original_id', entry['id']),
        resource_variant=entry.get('resource_variant'), score=score_ref, native=native_ref,
        physical_network_hash=entry['physical_cache']['network_hash'],
        target_native=target, competitor_native=competitor,
        labels=labels, sigma2=score['sigma2'],
        response_units='Squared-voltage response coefficients, not raw line ohms',
        target_scope='Native index 4 has correct internal topology in this Pengwah supplement; its numerical parameters and common term are not the physical truth',
        node_response_errors=node_rows, branch_response_coefficients=coefficient_rows,
        pair_Q_difference_norm=float(np.linalg.norm(D[:, n:])),
        pair_Q_common_difference_norm=float(np.linalg.norm(DC[:, n:])))
    return dict(H=H, C=C, physical=physical, sigma2=float(score['sigma2']), audit=audit)


def analyze(run_directory, target=4, competitor=2):
    run_directory = Path(run_directory).resolve()
    completion = read(run_directory/'complete.json')
    if completion['status'] != 'complete':
        raise ValueError('Only completed batches may be analyzed')
    inputs_ref = ref(run_directory/'inputs.json')
    entries = read(inputs_ref['path'])
    contexts = {e['id']: context(e, target, competitor) for e in entries}
    rows, errors = [], defaultdict(float)
    for results_file in sorted(run_directory.glob('fraction*/*/forecast*/results.json')):
        for result in read(results_file):
            if result['status'] != 'complete':
                rows.append(dict(id=result['id'], strategy=result['strategy'], status=result['status']))
                continue
            ctx = contexts[result['id']]
            H, C, physical, sigma2 = (ctx[k] for k in ('H','C','physical','sigma2'))
            repeat_dir = results_file.parent/f"execution{result['repeat']:02d}"
            zero_path = repeat_dir/'no_dfs'/'observations.npz'
            action_path = repeat_dir/result['strategy']/'observations.npz'
            X0, Y0, OX0, OY0 = observation(zero_path)
            X1, Y1, OX1, OY1 = observation(action_path)
            dX, dY = X1-X0, Y1-Y0
            before = split(X0,Y0,H[target],H[competitor],physical,sigma2)
            after = split(X1,Y1,H[target],H[competitor],physical,sigma2)
            change = {key:after[key]-before[key] for key in before}
            clean_gain = evidence(X1,Y1,H[target],H[competitor],sigma2)-evidence(X0,Y0,H[target],H[competitor],sigma2)
            observed_gain = evidence(OX1,OY1,H[target],H[competitor],sigma2)-evidence(OX0,OY0,H[target],H[competitor],sigma2)
            change['clean_L'] = clean_gain
            change['meter'] = observed_gain-clean_gain
            change['observed_L'] = observed_gain
            change['clean_L_without_common'] = evidence(X1,Y1,H[target]-C[target],H[competitor]-C[competitor],sigma2)-evidence(X0,Y0,H[target]-C[target],H[competitor]-C[competitor],sigma2)
            change['observed_L_without_common'] = evidence(OX1,OY1,H[target]-C[target],H[competitor]-C[competitor],sigma2)-evidence(OX0,OY0,H[target]-C[target],H[competitor]-C[competitor],sigma2)
            errors['direct_clean_vs_J_parameter_AC'] = max(errors['direct_clean_vs_J_parameter_AC'], abs(clean_gain-change['J']-change['parameter']-change['AC']))
            D = H[target]-H[competitor]
            d0, dd = X0@D.T, dX@D.T
            residual0, delta_residual = Y0-X0@H[target].T, dY-dX@H[target].T
            bias_terms = dict(baseline_residual_on_action_difference=float(np.sum(residual0*dd)/sigma2),
                action_prediction_error_on_baseline_difference=float(np.sum(delta_residual*d0)/sigma2),
                action_prediction_error_on_action_difference=float(np.sum(delta_residual*dd)/sigma2))
            errors['bias_three_terms'] = max(errors['bias_three_terms'], abs(sum(bias_terms.values())-change['parameter']-change['AC']))
            # Reverse-target identity catches the easy sign mistake: J is
            # unchanged, so reverse bias is -bias-2J, not simply -bias.
            reversed_gain = evidence(OX1,OY1,H[competitor],H[target],sigma2)-evidence(OX0,OY0,H[competitor],H[target],sigma2)
            errors['opposite_orientation'] = max(errors['opposite_orientation'], abs(observed_gain+reversed_gain))
            records = {f'native_{i}':response_error(dX@h.T,dY) for i,h in enumerate(H)}
            records['physical_linear'] = response_error(dX@physical.T,dY)
            records['target_without_common'] = response_error(dX@(H[target]-C[target]).T,dY)
            records['competitor_without_common'] = response_error(dX@(H[competitor]-C[competitor]).T,dY)
            row = dict(id=result['id'], resource_variant=result['resource_variant'],
                forecast=result['forecast_index'], repeat=result['repeat'], strategy=result['strategy'],
                status='complete', source=ref(action_path), baseline_source=ref(zero_path),
                target_native=target, competitor_native=competitor,
                gain_toward_target=change, bias_change_terms=bias_terms,
                action_response_errors=records,
                actual_deltaY_norm=float(np.linalg.norm(dY)),
                actual_deltaP_norm=float(np.linalg.norm(dX[:,:H.shape[1]])),
                actual_deltaQ_norm=float(np.linalg.norm(dX[:,H.shape[1]:])),
                native_selected_index=result['native_selected_index'],
                internal_topology_recovered=result['native_map_correct'],
                reported_true_topology_weight=result['p_true'],
                reported_wrong_historical_target_weight=result['p_target'])
            nearest = result.get('evidence',{}).get('nearest_competitor')
            if nearest is not None:
                row['reported_forecast_J_gain'] = nearest['forecast_J_gain']
                if nearest['target']==competitor and nearest['competitor']==target:
                    errors['observed_L_vs_saved_reverse'] = max(errors['observed_L_vs_saved_reverse'], abs(observed_gain+nearest['actual_logodds_change']))
            rows.append(row)
    if max(errors.values(), default=0.) > 1e-7:
        raise ArithmeticError(f'Independent decomposition failed: {dict(errors)}')
    groups=defaultdict(list)
    for row in rows:
        groups[row['id'],row['strategy']].append(row)
    summaries=[]
    for (identifier,strategy), group in sorted(groups.items()):
        complete=[r for r in group if r['status']=='complete']
        summary=dict(id=identifier,strategy=strategy,rows=len(group),complete=len(complete),
            recovered=sum(r['internal_topology_recovered'] for r in complete))
        if complete:
            summary['gain_toward_target']={k:mean_stats([r['gain_toward_target'][k] for r in complete]) for k in complete[0]['gain_toward_target']}
            summary['bias_change_terms']={k:mean_stats([r['bias_change_terms'][k] for r in complete]) for k in complete[0]['bias_change_terms']}
            summary['action_response_relative_errors']={k:mean_stats([r['action_response_errors'][k]['relative_error'] for r in complete]) for k in complete[0]['action_response_errors']}
            summary['actual_deltaY_norm']=mean_stats([r['actual_deltaY_norm'] for r in complete])
            gains=np.array([r['gain_toward_target']['observed_L'] for r in complete])
            summary['positive_gain_paths']=int((gains>1e-8).sum())
            summary['negative_gain_paths']=int((gains < -1e-8).sum())
            summary['true_topology_weight']=mean_stats([r['reported_true_topology_weight'] for r in complete])
            summary['wrong_historical_target_weight']=mean_stats([r['reported_wrong_historical_target_weight'] for r in complete])
            summary['true_topology_weight_at_least_90pct']=sum(r['reported_true_topology_weight'] >= .9 for r in complete)
            summary['wrong_historical_target_weight_at_least_90pct']=sum(r['reported_wrong_historical_target_weight'] >= .9 for r in complete)
        summaries.append(summary)
    return dict(created_utc=datetime.now(timezone.utc).isoformat(), run=str(run_directory),
        inputs=inputs_ref, completion=ref(run_directory/'complete.json'), source=ref(__file__),
        physical_response_source=ref(Path(__file__).parents[1]/'measured_pq_flex_20261114'/'analyze.py'),
        conditions=dict(no_new_optimization=True,no_new_AC=True,no_parameter_fitting=True,
            no_new_resource_selection=True,physical_lines_used_only_after_execution=True,
            target_orientation='Positive L means native 4 is supported over native 2; J has no target orientation',
            no_probability_claim='Distance realization and actual evidence realization are reported separately'),
        contexts=[c['audit'] for c in contexts.values()],maximum_identity_errors=dict(errors),
        summary=summaries,rows=rows)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    result=analyze(args.run)
    args.out.mkdir(parents=True,exist_ok=False)
    for name,data in [('diagnostic.json',result),('SUMMARY.json',{k:v for k,v in result.items() if k!='rows'})]:
        with (args.out/name).open('x',encoding='utf-8') as stream:
            json.dump(data,stream,indent=2,ensure_ascii=False,allow_nan=False)
            stream.write('\n')
    print(json.dumps(dict(output=ref(args.out/'diagnostic.json'), rows=len(result['rows']), maximum_identity_errors=result['maximum_identity_errors'])))


if __name__=='__main__':
    main()
