"""Read-only result aggregation and independent physical-bias decomposition."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

from ... import io
from topology_identification_standalone import data as backend
from ..measured_pq_20261114.physical import _network_hash


def physical_response(case, labels, expected_hash):
    network = backend.load_case(case)
    if _network_hash(network) != expected_hash:
        raise ValueError('Physical reference network differs from executed version')
    order, parent, incoming = backend._radial_order(network)
    descendants = {bus: set() for bus in order}
    for i, bus in enumerate(labels):
        descendants[int(bus)].add(i)
    R = np.zeros((len(labels), len(labels))); X = R.copy()
    zbase = network['base_kv']**2/network['base_mva']
    for bus in reversed(order[1:]):
        mask = np.array([i in descendants[bus] for i in range(len(labels))], float)
        edge = incoming[bus]
        R += 2*edge['r_ohm']/zbase*np.outer(mask, mask)
        X += 2*edge['x_ohm']/zbase*np.outer(mask, mask)
        descendants[parent[bus]].update(descendants[bus])
    return np.c_[R, X]


def bias_parts(path, Ht, Hj, physical, sigma2):
    with np.load(path, allow_pickle=False) as z:
        power = np.c_[z['true_P'], z['true_Q']]
        y = z['true_root'][:, None]**2-z['true_V']**2
    d = power@(Ht-Hj).T
    linear_y = power@physical.T
    return dict(parameter=float(np.sum((linear_y-power@Ht.T)*d)/sigma2),
                AC=float(np.sum((y-linear_y)*d)/sigma2))


def stats(values):
    values = np.array([v for v in values if v is not None and np.isfinite(v)], float)
    if not len(values):
        return dict(n=0, mean=None, minimum=None, maximum=None, median=None)
    return dict(n=len(values), mean=float(values.mean()), minimum=float(values.min()),
                maximum=float(values.max()), median=float(np.median(values)))


def summarize(rows, planned_rows=None):
    successful = [r for r in rows if r['status'] == 'complete']
    evidence = [r['evidence']['nearest_competitor'] for r in successful
                if r.get('evidence', {}).get('nearest_competitor') is not None]
    active = [r for r in successful if r['planned_fee_yuan'] > 1e-9]
    positive_forecast = [r for r in active if r['evidence'].get('nearest_competitor') and
                         r['evidence']['nearest_competitor']['forecast_J_gain'] > 1e-6]
    denominator = sum(r['evidence']['nearest_competitor']['forecast_J_gain'] for r in positive_forecast)
    ratio = {}
    for key in ('exact_physical_J_gain', 'actual_physical_J_gain', 'actual_logodds_change'):
        available = [r for r in positive_forecast if r['evidence']['nearest_competitor'].get(key) is not None]
        d = sum(r['evidence']['nearest_competitor']['forecast_J_gain'] for r in available)
        ratio[key] = dict(n=len(available), forecast_gain_sum=d,
                         realized_gain_sum=sum(r['evidence']['nearest_competitor'][key] for r in available),
                         ratio=sum(r['evidence']['nearest_competitor'][key] for r in available)/d if d else None)
    signed_ratio = {}
    for key in ('exact_physical_J_gain', 'actual_physical_J_gain', 'actual_logodds_change'):
        available = [r for r in active if r['evidence'].get('nearest_competitor') and
                     r['evidence']['nearest_competitor'].get(key) is not None]
        d = sum(r['evidence']['nearest_competitor']['forecast_J_gain'] for r in available)
        n = sum(r['evidence']['nearest_competitor'][key] for r in available)
        signed_ratio[key] = dict(n=len(available), forecast_gain_sum=d, realized_gain_sum=n,
                                 ratio=n/d if abs(d)>1e-6 else None)
    fields = ('forecast_J_gain', 'exact_physical_J_gain', 'actual_physical_J_gain', 'actual_bias_change',
              'actual_parameter_bias_change', 'actual_AC_nonlinearity_change', 'actual_meter_change',
              'actual_logodds_change', 'same_observations_logodds_change_without_common')
    nearest = {key: stats([r.get(key) for r in evidence]) for key in fields}
    planned = len(rows) if planned_rows is None else planned_rows
    return dict(planned_rows=planned, rows=len(rows), missing_rows=planned-len(rows),
        complete=len(successful), failed_recorded=len(rows)-len(successful), failed_or_missing=planned-len(successful),
        internal_recovered=sum(r.get('native_map_correct', False) for r in successful),
        full_support_exact=sum(r.get('native_full_support_exact', False) for r in successful),
        true_probability=stats([r.get('p_true') for r in successful]),
        target_probability=stats([r.get('p_target') for r in successful]),
        nonzero_paths=len(active), fallback_paths=sum(r.get('quote_zero_fallback', False) for r in rows),
        settled_fee_yuan=stats([r.get('settled_fee_yuan') for r in successful]),
        planned_fee_yuan=stats([r.get('planned_fee_yuan') for r in successful]),
        delivered_abs_kwh=stats([r.get('delivered_abs_kwh') for r in successful]),
        PV_clipped_kwh=stats([r.get('delivery', {}).get('PV_clipped_kwh') for r in successful]),
        root_energy_value_change_yuan=stats([r.get('root_energy_value_change_yuan') for r in successful]),
        baseline_root_energy_value_yuan=stats([r.get('baseline_root_energy_value_yuan') for r in successful]),
        positive_actual_gain=sum(r['evidence']['nearest_competitor']['actual_logodds_change'] > 1e-8 for r in active if r['evidence'].get('nearest_competitor')),
        negative_actual_gain=sum(r['evidence']['nearest_competitor']['actual_logodds_change'] < -1e-8 for r in active if r['evidence'].get('nearest_competitor')),
        nearest_evidence=nearest, realization=ratio, signed_realization_all_nonzero=signed_ratio,
        realization_cohort='Complete nonzero actions with fixed-runner predicted J gain >1e-6; same rows in numerator/denominator',
        nonzero_with_nonpositive_or_small_forecast=len(active)-len(positive_forecast),
        voltage_excursion_paths=sum(r.get('voltage_outside_095_105_cells', 0) > 0 for r in successful))


def analyze(run, out):
    run, out = Path(run).resolve(), Path(out).resolve()
    if out.exists():
        raise FileExistsError('Preserve completed analysis')
    if not (run/'complete.json').exists():
        raise ValueError('Analyze a completed run only')
    protocol = io.read(run/'protocol.json'); entries = io.read(run/'inputs.json')
    entry_map = {e['id']:e for e in entries}; contexts = {}
    rows, quotes, exact_failures = [], [], []
    for e in entries:
        score = io.read(e['score']['path'])
        H = np.array([p['H'] for p in score['native_bank']])
        contexts[e['id']] = score, H, physical_response(e['case'], score['labels'], e['physical_cache']['network_hash'])
    maximum_bias_error = 0.
    for file in sorted(run.glob('fraction*/*/forecast*/results.json')):
        folder = file.parent; records = io.read(file)
        for mode, quote in io.read(folder/'plans.json').items():
            quotes.append({**quote, 'id':folder.parent.name, 'forecast':folder.name,
                'fraction':float(folder.parent.parent.name.removeprefix('fraction')), 'mode':mode})
        for path in folder.glob('exact_requested/*/receipt.json'):
            exact = io.read(path)
            if exact['status'] != 'complete':
                exact_failures.append(dict(path=str(path), **exact))
        before_cache = {}
        for row in records:
            if row['status'] != 'complete':
                rows.append(row); continue
            ev = row.get('evidence', {}).get('nearest_competitor')
            if ev is not None:
                score, H, physical = contexts[row['id']]
                target, other = ev['target'], ev['competitor']
                repeat_dir = folder/f"execution{row['repeat']:02d}"
                key = row['repeat'], target, other
                if key not in before_cache:
                    before_cache[key] = bias_parts(repeat_dir/'no_dfs/observations.npz', H[target], H[other], physical, score['sigma2'])
                before = before_cache[key]
                after = bias_parts(repeat_dir/row['strategy']/'observations.npz', H[target], H[other], physical, score['sigma2'])
                ev['actual_parameter_bias_change'] = after['parameter']-before['parameter']
                ev['actual_AC_nonlinearity_change'] = after['AC']-before['AC']
                error = abs(ev['actual_parameter_bias_change']+ev['actual_AC_nonlinearity_change']-ev['actual_bias_change'])
                maximum_bias_error = max(maximum_bias_error, error)
                if error > 1e-7:
                    raise ArithmeticError('Physical linear/AC residual decomposition does not close')
            rows.append(row)
    by_key = {(r['contract_fraction'], r['id'], r['forecast_index'], r['repeat'], r['strategy']):r for r in rows}
    groups = defaultdict(list); cases = defaultdict(list); target_groups = defaultdict(list)
    for row in rows:
        groups[row['contract_fraction'], row['strategy']].append(row)
        cases[row['contract_fraction'], row['id'], row['strategy']].append(row)
        target_groups[row['contract_fraction'], row.get('historical_target_correct'), row['strategy']].append(row)
    paths_per_history = protocol['forecasts']*protocol['repeats']
    def with_comparison(group, planned=None):
        s = summarize(group, planned); rescuing = losing = count = 0
        weight = []
        for r in group:
            zero = by_key.get((r['contract_fraction'],r['id'],r['forecast_index'],r['repeat'],'no_dfs'))
            if r['status'] != 'complete' or zero is None or zero['status'] != 'complete':
                continue
            count += 1
            rescuing += r['native_map_correct'] and not zero['native_map_correct']
            losing += zero['native_map_correct'] and not r['native_map_correct']
            if r.get('p_true') is not None and zero.get('p_true') is not None:
                weight.append(r['p_true']-zero['p_true'])
        s.update(paired_no_dfs_paths=count, recovered_additions=int(rescuing), recovery_losses=int(losing),
                 true_probability_change=stats(weight))
        return s
    quote_groups = {}
    for fraction, mode in sorted({(q['fraction'],q['mode']) for q in quotes}):
        selected = [q for q in quotes if q['fraction']==fraction and q['mode']==mode]
        quote_groups[f'{fraction:.2f}/{mode}'] = dict(n=len(selected),
            planned_quotes=len(entries)*protocol['forecasts'],
            missing_quotes=len(entries)*protocol['forecasts']-len(selected),
            feasible=sum(q['feasible'] for q in selected),nonzero=sum(q['nonzero_verified_feasible'] for q in selected),
            zero_fallback=sum(q['zero_fallback'] for q in selected),
            proved_optimal=sum(q.get('global_optimality_verified',False) for q in selected),
            initial_source=Counter(q.get('source','unavailable') for q in selected),
            native_status=Counter(str(q.get('native_status','unavailable')) for q in selected),
            status=Counter(q.get('status','unavailable') for q in selected),
            seconds=stats([q['charged_total_seconds'] for q in selected]),
            budget_exceeded=sum(q['charged_total_budget_exceeded'] for q in selected),
            relative_gap=stats([q.get('relative_gap') for q in selected]),
            initial_fee_improvement=stats([q.get('improvement_from_verified_initial_yuan') for q in selected]),
            rotation_fee_mismatch=sum(not q['rotation_same_fee'] for q in selected))
    completion_by_fraction = {f.name:io.read(f) for f in sorted(run.glob('complete_fraction*.json'))}
    for finished in completion_by_fraction.values():
        for strategy in ('no_dfs','old_gamma2','old_gamma2_rotation','symmetric_guard','symmetric_guard_rotation'):
            groups.setdefault((finished['fraction'],strategy), [])
            for entry in entries:
                cases.setdefault((finished['fraction'],entry['id'],strategy), [])
    result = dict(run=io.ref(run/'complete.json'),protocol=io.ref(run/'protocol.json'),source=io.ref(__file__),
        fixed_background_inputs=len(entries),forecasts_per_input=protocol['forecasts'],repeats_per_forecast=protocol['repeats'],
        conditional_paths_per_fraction=len(entries)*protocol['forecasts']*protocol['repeats'],
        rows=len(rows),maximum_physical_bias_split_error=maximum_bias_error,exact_diagnostic_failures=exact_failures,
        planned_strategy_rows=len(completion_by_fraction)*5*len(entries)*paths_per_history,
        fraction_completion=completion_by_fraction,
        source_unchanged=io.read(run/'complete.json')['source_unchanged'],
        overall={f'{f:.2f}/{s}':with_comparison(g,len(entries)*paths_per_history) for (f,s),g in groups.items()},
        per_history={f'{f:.2f}/{i}/{s}':with_comparison(g,paths_per_history) for (f,i,s),g in cases.items()},
        by_target_correctness={f'{f:.2f}/{c}/{s}':with_comparison(g) for (f,c,s),g in target_groups.items()},
        quotes=quote_groups)
    out.mkdir(parents=True)
    io.write(out/'SUMMARY.json',result);io.write(out/'rows_with_bias_split.json',rows)
    io.write(out/'complete.json',dict(summary=io.ref(out/'SUMMARY.json')))
    return result


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();r=analyze(a.run,a.out)
    print(dict(rows=r['rows'],fixed_background_inputs=r['fixed_background_inputs'],maximum_bias_split_error=r['maximum_physical_bias_split_error']))
