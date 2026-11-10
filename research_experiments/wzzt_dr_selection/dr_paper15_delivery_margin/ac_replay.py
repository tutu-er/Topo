"""Summarize frozen paper15 AC observations, independent of the delivery-margin fit."""
import argparse
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np


CASE = 'paper15_b2_v0005'
POLICIES = ('zero', 'rotation', 'candidate')
LAWS = ('main', 'stress')


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def checked_pickle(reference):
    data = Path(reference['path']).read_bytes()
    if hashlib.sha256(data).hexdigest() != reference['sha256']:
        raise ValueError('Archived packet does not match its frozen hash')
    return pickle.loads(data)


def verify(condition, label):
    if not condition:
        raise ValueError(label)


def score_scene(scene, pool, true_support):
    p, q, voltage = (np.asarray(scene[key]) for key in ('P_terminal', 'Q_terminal', 'V_terminal'))
    root_voltage = np.asarray(scene['V_root_observed']).reshape(-1, 1)
    y = root_voltage**2 - voltage**2
    verify(p.shape == (96, len(pool['labels'])), 'Wrong observation shape')
    models = {m['candidate_id']: m for m in pool['models']}
    loss = {k: float(np.abs(y - p @ np.asarray(m['R']).T - q @ np.asarray(m['X']).T).mean())
            for k, m in models.items()}
    hist = {k: m['historical_validation_mae'] for k, m in models.items()}
    combined = {k: (288 * hist[k] + 192 * loss[k]) / 480 for k in models}
    order = pool['selection_order']
    hist_eligible = [k for k in order if hist[k] <= 1.03 * min(hist.values()) + 1e-12]
    limit = 1.03 * min(combined[k] for k in hist_eligible) + 1e-12
    eligible = [k for k in hist_eligible if combined[k] <= limit]
    selected = min(eligible, key=lambda k: (len(models[k]['point']['supports']), combined[k], order.index(k)))
    support = {tuple(sorted(block)) for block in models[selected]['frozen_output_support_labels']}
    return (selected, 'coarse' not in eligible, len(support ^ true_support),
            1e6 * (loss['coarse'] - 1.03 * loss['refined']))


def analyze(root):
    freeze = read_json(root / 'price_freeze.json')
    plan_ref = freeze['cases'][CASE]['plans']
    plans = checked_pickle(plan_ref)
    pool = read_json(root / CASE / 'pool.json')
    verify((pool['validation_rows'], pool['train_rows']) == (288, 288), 'Full history required')
    truth = {tuple(sorted(b)) for b in read_json(root / CASE / 'truth_private.json')['missed_blocks']}
    scores = {r['record_id']: r for r in read_json(root / 'scored_private.json')['rows']}
    groups = {(law, policy): {} for law in LAWS for policy in POLICIES}
    observation_hashes = []
    for row in read_json(root / 'observations_frozen.json')['rows']:
        if row['condition'] != CASE or row['policy'] not in POLICIES:
            continue
        identity = row['record_id']
        verify(row['status'] == 'complete', 'Unresolved observation ' + identity)
        observation_hashes.append((identity, row['packet']['sha256']))
        packet = checked_pickle(row['packet'])
        selected, coarse_excluded, errors, gap = score_scene(packet['scene'], pool, truth)
        old = scores[identity]
        verify((selected, coarse_excluded, errors) == (old['selected'], old['coarse_excluded'], old['errors']), identity)
        verify(abs(gap - old['D_micro']) < 1e-8, 'Loss mismatch ' + identity)
        verify(packet['summary']['converged'] == 96, 'AC convergence ' + identity)
        nominal = plans[row['policy']]['mean_u']
        delivered = np.asarray(packet['private']['delivered'])
        verify(np.array_equal(nominal, packet['private']['nominal']), 'Frozen contract mismatch ' + identity)
        contract_abs = np.abs(nominal).sum()
        delivered_abs = np.abs(delivered).sum()
        energy = float(0.25 * delivered_abs)
        verify(abs(energy - packet['summary']['resource_audit']['absolute_DR_kwh']) < 1e-9, identity)
        norm_sq = np.sum(nominal**2)
        scale = np.sum(nominal * delivered) / norm_sq if norm_sq else 1.0
        group = groups[row['law'], row['policy']]
        verify(row['day'] not in group, 'Duplicate realization ' + identity)
        group[row['day']] = dict(errors=errors, gap=float(gap), energy_kwh=energy,
            shortfall_kw=float((contract_abs - delivered_abs) / 96),
            tracking_kw=float(np.abs(delivered - nominal).sum() / 96),
            peak_kw=float(np.abs(delivered).sum(axis=0).max()),
            best_scale_max_error_kw=float(np.max(np.abs(delivered - scale * nominal))))

    results = []
    for (law, policy), daily in groups.items():
        verify(set(daily) == set(range(12)), 'Incomplete day set ' + law + ' ' + policy)
        records = list(daily.values())
        def values(key):
            return np.asarray([r[key] for r in records])
        nominal = plans[policy]['mean_u']
        results.append(dict(law=law, policy=policy, days=len(records),
            unresolved=0, errors=int(values('errors').sum()), recoveries=int(sum(values('errors') == 0)),
            contract_fee=float(plans[policy]['cost']), nominal_kwh=float(.25 * np.abs(nominal).sum()),
            nominal_avg_gross_kw=float(np.abs(nominal).sum() / 96),
            nominal_peak_gross_kw=float(np.abs(nominal).sum(axis=0).max()),
            delivered_mean_kwh=float(values('energy_kwh').mean()),
            shortfall_mean_kw=float(values('shortfall_kw').mean()),
            shortfall_range_kw=[float(values('shortfall_kw').min()), float(values('shortfall_kw').max())],
            mean_resource_tracking_error_kw=float(values('tracking_kw').mean()),
            mean_daily_peak_gross_kw=float(values('peak_kw').mean()), mean_D_micro=float(values('gap').mean()),
            days_not_on_best_scalar_ray=int(np.sum(values('best_scale_max_error_kw') > 1e-8)),
            smallest_daily_best_scale_max_error_kw=float(values('best_scale_max_error_kw').min())))
    paired = []
    for law in LAWS:
        candidate = groups[law, 'candidate']
        for other in ('zero', 'rotation'):
            baseline = groups[law, other]
            change = np.array([candidate[day]['errors'] - baseline[day]['errors'] for day in sorted(candidate)])
            paired.append(dict(law=law, baseline=other, better=int(sum(change < 0)), same=int(sum(change == 0)), worse=int(sum(change > 0))))

    candidate = groups['main', 'candidate']
    failures = [day for day, rec in candidate.items() if rec['errors'] > 0]
    successes = [day for day, rec in candidate.items() if rec['errors'] == 0]
    failed_day = min(failures, key=lambda day: candidate[day]['shortfall_kw'])
    recovered_day = max(successes, key=lambda day: candidate[day]['shortfall_kw'])
    verify(candidate[failed_day]['shortfall_kw'] < candidate[recovered_day]['shortfall_kw'], 'Counterexample missing')
    examples = [dict(day=day, shortfall_kw=candidate[day]['shortfall_kw'], errors=candidate[day]['errors'],
                     D_micro=candidate[day]['gap']) for day in (failed_day, recovered_day)]
    return dict(case=CASE, parent_protocol='p90_cost_v1', frozen_plan_sha256=plan_ref['sha256'],
                packet_hash_manifest_sha256=hashlib.sha256(json.dumps(sorted(observation_hashes)).encode()).hexdigest(),
                records=len(observation_hashes), independent_backgrounds=12, results=results, paired=paired,
                main_shortfall_counterexample=examples,
                scope='Archived random delivered trajectories, matched days, frozen R/X, measured P/Q/V. No imposed common delay or ray-specific counterfactual AC observations.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive', type=Path)
    parser.add_argument('--out', type=Path, default=Path(__file__).with_name('ac_results.json'))
    args = parser.parse_args()
    result = analyze(args.archive)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(f"Checked {result['records']} complete AC observations in {len(result['results'])} groups")
