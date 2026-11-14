"""Historical-only placement audit for new, explicitly contracted pure-Q devices.

Read frozen candidate responses, registered device locations, and already sealed
power forecasts.  Do not load true lines, new-day observations, or AC caches.
Existing PV/battery kW registrations establish eligible locations only; they are
not interpreted as inverter kVA ratings or available kvar.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np

from ..measured_pq_flex_20261114.domain import catalog


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def ref(path, expected=None):
    path = Path(path).resolve()
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if expected is not None and digest != expected:
        raise ValueError(f'Changed sealed input: {path}')
    return dict(path=str(path), sha256=digest)


def checked(reference):
    return ref(reference['path'], reference['sha256'])


def historical_targets(score, mass=.99):
    probabilities = np.array([m['probability'] for m in score['models']], float)
    order = sorted(range(len(probabilities)), key=lambda i: (-probabilities[i], i))
    chosen = []
    total = 0.
    for i in order:
        chosen.append(i)
        total += probabilities[i]
        if total >= mass:
            break
    cutoff = probabilities[chosen[-1]]
    chosen = [i for i in order if probabilities[i] >= cutoff-1e-15]
    return chosen


def candidate_pairs(score):
    """Every different-topology raw parameter rival of a historical 99% target."""
    bank = score['native_bank']
    target_representatives = historical_targets(score)
    target_raw = [int(score['models'][i]['index']) for i in target_representatives]
    pairs = sorted({tuple(sorted((i, j))) for i in target_raw
                    for j in range(len(bank))
                    if bank[i]['topology_id'] != bank[j]['topology_id']})
    if not pairs:
        raise ValueError('No different-topology rival')
    H = np.asarray([p['H'] for p in bank], float)
    differences = np.asarray([H[i]-H[j] for i, j in pairs])
    return target_representatives, target_raw, pairs, differences


def choose_nodes(energy_per_kvar2, eligible, labels, count=2, capacity_weights=None):
    """Exhaustive small set choice: maximize the weakest retained pair sensitivity.

    Independent zero-mean hourly +/-Q signs give expected distance increment
    96*sum_n(Q_n**2*||D_Q[:,n]||**2)/(2*sigma2*base_kw**2).
    Only this analytical placement score is used; no sign schedules are sampled.
    The sum across pairs and then terminal labels break exact ties.
    """
    eligible = sorted(set(map(int, eligible)), key=lambda n: labels[n])
    count = min(count, len(eligible))
    if count == 0:
        return dict(node_indices=[], node_labels=[], minimum_sensitivity=0.,
                    pair_sensitivities=[], combinations_evaluated=0)
    weights = np.ones(len(labels)) if capacity_weights is None else np.asarray(capacity_weights, float)
    best = None
    evaluated = 0
    for nodes in itertools.combinations(eligible, count):
        values = (energy_per_kvar2[:, nodes]*weights[list(nodes)]**2).sum(axis=1)
        key = (float(values.min()), float(values.sum()), tuple(-labels[n] for n in nodes))
        evaluated += 1
        if best is None or key > best[0]:
            best = (key, nodes, values)
    _, nodes, values = best
    return dict(node_indices=list(nodes), node_labels=[labels[n] for n in nodes],
                minimum_sensitivity=float(values.min()), pair_sensitivities=values.tolist(),
                worst_pair_index=int(values.argmin()), combinations_evaluated=evaluated)


def q_resources(node_labels, capacity_kvar=5.):
    return [dict(id=f'svg_{int(label)}', node_label=int(label),
                 capacity_kvar=float(capacity_kvar),
                 fee_positive_yuan_per_kvarh=.05,
                 fee_negative_yuan_per_kvarh=.05,
                 available_clock_hours=list(range(24)),
                 reservation_fee_yuan_per_kvar_day=0.,
                 kind='new_independent_bidirectional_Q_contract',
                 capacity_origin='Explicit synthetic contracted SVG capacity; not inferred from registered kW',
                 price_origin='Illustrative, not a measured market quote') for label in node_labels]


def exact_pair_q_box(D, forecast, nodes, capacity_kvar, sigma2, base_kw=200.):
    """Exact maximum of one pair's J over independent hourly Q box, P unchanged.

    The quadratic is convex and separable by hour, so its maximum is attained
    at one of the 2**len(nodes) vertices in each hour. Different pairs may need
    different signs, so their separate maxima do not certify joint feasibility.
    """
    forecast = np.asarray(forecast, float)
    n = D.shape[0]
    if forecast.shape != (96, 2*n) or D.shape != (n, 2*n):
        raise ValueError('Expected 96 input rows and n by 2n frozen response')
    origin = (forecast@D.T).reshape(24, 4, n)
    J0 = float(np.square(origin).sum()/(2*sigma2))
    values = []
    for signs in itertools.product((-1., 1.), repeat=len(nodes)):
        delta = D[:, np.asarray(nodes, int)+n]@(np.asarray(signs)*capacity_kvar/base_kw)
        values.append(np.square(origin+delta).sum(axis=(1, 2))/(2*sigma2))
    Jmax = float(np.max(values, axis=0).sum())
    return dict(J0=J0, J_max=Jmax, gain_max=Jmax-J0)


def inspect_entry(entry, run_directory, cohort, base_kw=200.):
    score_ref, context_ref = checked(entry['score']), checked(entry['resource_context'])
    score, context = read(score_ref['path']), read(context_ref['path'])
    labels = list(map(int, score['labels']))
    if labels != [int(r['label']) for r in context['node_mapping']]:
        raise ValueError('Node ordering mismatch')
    n, sigma2 = len(labels), float(score['sigma2'])
    representatives, raw_targets, pairs, differences = candidate_pairs(score)
    qnorm = np.linalg.norm(differences[:, :, n:], axis=1)
    pnorm = np.linalg.norm(differences[:, :, :n], axis=1)
    qenergy = 96*qnorm*qnorm/(2*sigma2*base_kw*base_kw)
    penergy = 96*pnorm*pnorm/(2*sigma2*base_kw*base_kw)
    target = int(score['models'][int(score['map_index'])]['index'])
    bank = score['native_bank']
    nearest = min((j for j in range(len(bank))
                   if bank[j]['topology_id'] != bank[target]['topology_id']),
                  key=lambda j: (bank[j]['history_sse'], j))
    near_pair = tuple(sorted((target, nearest)))
    near_index = pairs.index(near_pair)
    registrations = {int(r['feeder']): r for r in context['resource_metadata_at_history_start']}
    old_resources = catalog(context, .1)
    old_labels = sorted({int(r['node_label']) for r in old_resources})
    rows, eligible, inv_weights = [], [], []
    fields = ['PV_systems_kW', 'batteries_kW', 'EV_chargers_kW',
              'hot_water_tanks_kW', 'heat_pumps_kW', 'chp_kW']
    for k, mapping in enumerate(context['node_mapping']):
        registration = registrations[int(mapping['feeder_id'])]
        scale = float(mapping['scale'])
        raw_capacity = {field: float(registration.get(field, 0.) or 0.) for field in fields}
        scaled = {field: raw_capacity[field]*scale for field in fields}
        inv = scaled['PV_systems_kW']+scaled['batteries_kW']
        inv_weights.append(inv)
        if inv >= 5.:
            eligible.append(k)
        rows.append(dict(label=labels[k], node_index=k, feeder_id=int(mapping['feeder_id']),
                         registration_date=registration['date'], source_to_network_scale=scale,
                         registered_capacity_kw=raw_capacity, scaled_registered_capacity_kw=scaled,
                         on_original_P_contract=labels[k] in old_labels,
                         PV_or_battery_registered=inv > 0,
                         meets_5kw_location_eligibility=inv >= 5.,
                         near_pair_P_column_norm=float(pnorm[near_index, k]),
                         near_pair_Q_column_norm=float(qnorm[near_index, k]),
                         all_pair_P_column_norms=pnorm[:, k].tolist(),
                         all_pair_Q_column_norms=qnorm[:, k].tolist(),
                         minimum_pair_P_sensitivity_per_kw2=float(penergy[:, k].min()),
                         minimum_pair_Q_sensitivity_per_kvar2=float(qenergy[:, k].min()),
                         capacity_weighted_minimum_Q_score=float(qenergy[:, k].min()*inv*inv),
                         capacity_weighted_score_scope='Registration kW used only as a ranking weight, not usable kvar'))
    schemes = {}
    for name, nodes in [('original_P_nodes', [labels.index(x) for x in old_labels]),
                        ('registered_inverter_nodes', eligible)]:
        chosen = choose_nodes(qenergy, nodes, labels)
        chosen.update(eligibility_node_labels=[labels[i] for i in nodes],
                      q_resources=q_resources(chosen['node_labels'], 5.),
                      backup_10kvar_resources=q_resources(chosen['node_labels'], 10.),
                      total_contracted_kvar=5.*len(chosen['node_labels']),
                      independent_sign_expected_gain_5kvar_per_pair=(np.asarray(chosen['pair_sensitivities'])*25).tolist(),
                      scope='Placement proxy over all frozen raw rivals of 99% history targets; no joint chance/mean-guard feasibility asserted')
        schemes[name] = chosen
    if len(schemes['original_P_nodes']['node_labels']) != len(schemes['registered_inverter_nodes']['node_labels']):
        raise ValueError('Paired configurations do not have the same total Q capacity')
    extra = dict(unrestricted_SVG=choose_nodes(qenergy, range(n), labels),
                 registration_capacity_weighted=choose_nodes(qenergy, eligible, labels, capacity_weights=inv_weights),
                 use='Diagnostic alternatives only; not the proposed equal-contract comparison')
    forecasts = []
    for folder in sorted((Path(run_directory)/'fraction0.10'/entry['id']).glob('forecast*')):
        seal = read(folder/'seal.json')
        planner_ref = checked(seal['planner'])
        with np.load(planner_ref['path'], allow_pickle=False) as arrays:
            forecast = arrays['forecast'].copy()  # no other array is opened
        result = dict(name=folder.name, source=planner_ref, schemes={})
        for name, scheme in schemes.items():
            result['schemes'][name] = {}
            for cap in (5., 10.):
                bounds = [exact_pair_q_box(D, forecast, scheme['node_indices'], cap, sigma2, base_kw)
                          for D in differences]
                result['schemes'][name][str(int(cap))] = dict(per_pair=bounds,
                    minimum_individual_gain_max=min(b['gain_max'] for b in bounds),
                    near_pair=bounds[near_index],
                    joint_feasibility_proven=False)
        if cohort == 'enriched':
            result['all_nodes_near_pair'] = [dict(label=labels[k],
                capacity_5=exact_pair_q_box(differences[near_index], forecast, [k], 5., sigma2, base_kw),
                capacity_10=exact_pair_q_box(differences[near_index], forecast, [k], 10., sigma2, base_kw)) for k in range(n)]
        forecasts.append(result)
    return dict(id=entry['id'], cohort=cohort, score=score_ref, resource_context=context_ref,
                sigma2=sigma2, base_kw=base_kw, targets_representative=representatives,
                targets_native=raw_targets, raw_parameter_count=len(bank),
                pairs_native=[list(p) for p in pairs], nearest_pair_native=list(near_pair),
                near_pair_index=near_index, original_P_contracts=old_resources,
                original_P_node_labels=old_labels, node_audit=rows,
                node_rank_by_near_Q=[r['label'] for r in sorted(rows, key=lambda r:(-r['near_pair_Q_column_norm'], r['label']))],
                node_rank_by_all_pair_Q=[r['label'] for r in sorted(rows, key=lambda r:(-r['minimum_pair_Q_sensitivity_per_kvar2'], r['label']))],
                schemes=schemes, alternative_rankings=extra, forecast_box_checks=forecasts)


def run(main_directory, enriched_directory):
    rows, sources = [], []
    for cohort, directory in [('main', main_directory), ('enriched', enriched_directory)]:
        directory = Path(directory).resolve()
        inputs_ref = ref(directory/'inputs.json')
        sources.append(dict(cohort=cohort, inputs=inputs_ref))
        for entry in read(inputs_ref['path']):
            rows.append(inspect_entry(entry, directory, cohort))
    return dict(created_utc=datetime.now(timezone.utc).isoformat(), source=ref(__file__), inputs=sources,
        conditions=dict(no_refit=True, no_AC=True, no_new_day_observations_read=True,
            true_network_parameters_read=False,
            forecasts='Only already sealed P/Q planner forecasts; forecasts do not choose locations',
            placement='Exhaustive best set of two by weakest all-native-pair squared Q-column sensitivity; sum then smaller labels break ties',
            pair_scope='All different-topology raw rivals of representatives covering cumulative 99% historical probability, including cutoff ties',
            registration='Expanded locations require historical scaled PV plus battery registration >=5 kW; original-location arm uses original P contract locations. This is only location eligibility and provides no certified Q headroom',
            proposed_Q='Two explicit new independent 5-kvar bidirectional compensation contracts, 24h; 10-kvar only an upper-bound diagnostic',
            expected_sensitivity='Algebraic expectation under independent zero-mean +/-Q signs, not a claimed posterior or AC success probability',
            exact_pair_max='Each hour enumerates all 2^m Q-box vertices; exact for each pair separately; not a joint all-pair feasibility certificate'),
        rows=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--main-run', type=Path, default=Path('outputs/measured_pq_dfs_main_20261114_v1'))
    parser.add_argument('--enriched-run', type=Path, default=Path('outputs/measured_pq_dfs_enriched_20261114_v1'))
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    report = run(args.main_run, args.enriched_run)
    args.out.mkdir(parents=True, exist_ok=False)
    for name, data in [('placement_audit.json', report), ('configurations.json', dict(
            audit_source=ref(__file__), entries=[dict(id=r['id'], cohort=r['cohort'],
            score=r['score'], resource_context=r['resource_context'],
            schemes={name: dict(q_resources=scheme['q_resources'],
                 backup_10kvar_resources=scheme['backup_10kvar_resources'],
                 node_labels=scheme['node_labels'], total_contracted_kvar=scheme['total_contracted_kvar'])
                 for name, scheme in r['schemes'].items()}) for r in report['rows']]))]:
        with (args.out/name).open('x', encoding='utf-8') as stream:
            json.dump(data, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write('\n')
    for cohort in ('main', 'enriched'):
        configurations = {}
        for row in report['rows']:
            if row['cohort'] != cohort:
                continue
            configurations[row['id']] = {
                'original_sites': row['schemes']['original_P_nodes']['q_resources'],
                'eligible_sites': row['schemes']['registered_inverter_nodes']['q_resources']}
        data = dict(cohort=cohort, source=ref(__file__),
                    audit=ref(args.out/'placement_audit.json'),
                    configurations_by_input=configurations,
                    registration_eligibility_kw=5., devices_per_input=2,
                    capacity_per_device_kvar=5.,
                    capacity_origin='Explicit new independent Q contracts; no registered Q capability asserted')
        with (args.out/f'{cohort}_configurations.json').open('x', encoding='utf-8') as stream:
            json.dump(data, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write('\n')
    print(json.dumps(dict(report=ref(args.out/'placement_audit.json'), configurations=ref(args.out/'configurations.json'),
        inputs=len(report['rows']), enriched_schemes=report['rows'][-1]['schemes']), ensure_ascii=False))


if __name__ == '__main__':
    main()
