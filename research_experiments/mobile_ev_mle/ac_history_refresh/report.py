"""Paired, history-level comparison of saved A and B inference.

This module does not fit, calibrate, select actions, or generate observations.
Run it only after B has frozen and all requested evaluation records exist.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
from pathlib import Path

import numpy as np

from . import io
from .. import ev_ac_repair_analysis as old


PLANS = ('constant', 'SQMD', 'BCMD')
METRICS = ('error', 'clean_error', 'false_rejection', 'unresolved',
           'correct_decisive', 'wrong_decisive', 'coverage90',
           'true_predictive_NLL', 'classification_NLL', 'brier')


def history_values(rows, metric):
    """Both future regimes must be available; one value per history."""
    grouped = defaultdict(dict)
    for row in rows:
        if row.get('available') and metric in row:
            key = (int(row['index']), row['plan'], row['model'])
            grouped[key][row['regime']] = float(row[metric])
    return {key: float(np.mean(list(values.values())))
            for key, values in grouped.items()
            if set(values) == {'regular', 'shift'}}


def paired_rows(rows_a, rows_b, count):
    """B minus A on the exact same histories; missing pairs remain visible."""
    output = {}
    for plan in PLANS:
        for model_a, model_b in (('G00', 'G00'), ('G11', 'G11'),
                                  ('G11_cal', 'G11_cal'),
                                  ('AC_fixed', 'G11_cal'),
                                  ('AC_joint', 'G11_cal')):
            key = f'{plan}:A_{model_a}->B_{model_b}'
            entry = {}
            for metric in METRICS:
                aa = history_values(rows_a, metric)
                bb = history_values(rows_b, metric)
                a = {i: v for (i, p, m), v in aa.items()
                     if p == plan and m == model_a}
                b = {i: v for (i, p, m), v in bb.items()
                     if p == plan and m == model_b}
                ids = sorted(set(a) & set(b))
                if not ids:
                    entry[metric] = dict(histories=0,total_histories=count,missing_pairs=count,
                        paired_indices=[],A_mean=None,B_mean=None,B_minus_A=None,bootstrap95=None,
                        lower_histories=0,tied_histories=0,higher_histories=0)
                    continue
                differences = [b[i]-a[i] for i in ids]
                entry[metric] = dict(histories=len(ids), total_histories=count,
                    missing_pairs=count-len(ids), paired_indices=ids,
                    A_mean=float(np.mean([a[i] for i in ids])),
                    B_mean=float(np.mean([b[i] for i in ids])),
                    B_minus_A=float(np.mean(differences)),
                    bootstrap95=old.interval(differences),
                    lower_histories=sum(d < 0 for d in differences),
                    tied_histories=sum(d == 0 for d in differences),
                    higher_histories=sum(d > 0 for d in differences))
            output[key] = entry
    return output


def channel_coverage(record, calibration=None):
    """Marginal 90% coverage, grouped by actual meter channel, per history."""
    result = []
    n = 3 if record.get('spec', {}).get('bank') == 'n3' else 4
    horizon = 8
    width = 3*n+2
    # symmetric_ac_local.whiten_observations concatenates whole channels.
    tn = horizon*n
    masks = dict(P=slice(0,tn), Q=slice(tn,2*tn), V=slice(2*tn,3*tn),
                 master_P=slice(3*tn,3*tn+horizon),
                 master_Q=slice(3*tn+horizon,3*tn+2*horizon))
    names = record.get('candidate_names', [])
    if record.get('true_name') not in names:
        return result
    true = names.index(record['true_name'])
    for label, plan in record.get('plans', {}).items():
        action = record.get('actions', {}).get(plan.get('action_key'), {})
        usable = old.usable_banks(action)
        for model in ('G00', 'G11', 'G11_cal'):
            multiplier = (calibration or {}).get('G11_multiplier') if model == 'G11_cal' else 1.
            bank = usable.get('G11' if model == 'G11_cal' else model)
            if bank is None or multiplier is None:
                continue
            means, covariances, _ = old.scaled_bank(bank, multiplier)
            if means.shape[1] != horizon*width:
                raise ValueError('declared meter dimension does not match bank')
            for regime in ('regular', 'shift'):
                actual = action.get('regimes', {}).get(regime, {})
                if 'values' not in actual:
                    continue
                y = np.atleast_2d(actual['values'])
                covered = abs(y-means[true]) <= 1.6448536269514722*np.sqrt(
                    np.diag(covariances[true]))
                result.append(dict(index=record['spec']['index'], plan=label,
                    regime=regime, model=model,
                    coverage={channel: float(covered[:, ids].mean())
                              for channel, ids in masks.items()}))
    return result


def summarize_channels(rows):
    grouped = defaultdict(lambda: defaultdict(list))
    for row in rows:
        for channel, value in row['coverage'].items():
            grouped[(row['plan'], row['model'], channel)][row['index']].append(value)
    output = {}
    for key, hist in grouped.items():
        values = [float(np.mean(v)) for v in hist.values() if len(v) == 2]
        output['__'.join(key)] = dict(histories=len(values),
            mean=None if not values else float(np.mean(values)),
            bootstrap95=old.interval(values))
    return output


def operational_summary(records, calibration, candidate_counts=None):
    """Keep every prescribed task; unavailable inference retains all candidates."""
    rows = [row for record in records for row in old.measurements(record, calibration)]
    by = {(r['index'],r['plan'],r['model'],r['regime']): r for r in rows if r['available']}
    models = ['G00','G10','G01','G11']
    if (calibration or {}).get('G11_multiplier') is not None:models.append('G11_cal')
    output = {}
    for plan in PLANS:
        for model in models:
            hist = [];available_tasks=0
            for record in records:
                index=record['spec']['index'];parts=[]
                for regime in ('regular','shift'):
                    row=by.get((index,plan,model,regime))
                    if row is None:
                        parts.append(dict(false_rejection=0.,unresolved=1.,correct_decisive=0.,
                                          wrong_decisive=0.))
                    else:
                        available_tasks+=1
                        parts.append({k:row[k] for k in ('false_rejection','unresolved',
                            'correct_decisive','wrong_decisive')})
                hist.append({k:float(np.mean([p[k] for p in parts])) for k in parts[0]})
            output[f'{plan}__{model}']=dict(histories=len(records),tasks=2*len(records),
                available_tasks=available_tasks,failed_tasks=2*len(records)-available_tasks,
                **{k:dict(mean=float(np.mean([h[k] for h in hist])),
                           bootstrap95=old.interval([h[k] for h in hist])) for k in hist[0]})
    return output


def compare(root_a, root_b, phase='evaluation', expected=32):
    root_a, root_b = Path(root_a), Path(root_b)
    freeze_a, freeze_b = io.load_record(root_a/'freeze.json'), io.load_record(root_b/'freeze.json')
    cal_a, cal_b = io.load_record(root_a/'calibration.json'), io.load_record(root_b/'calibration.json')
    binding=None
    if phase=='evaluation':
        binding=io.load_record(root_b/'evaluation_calibration_binding.json')
        if (binding.get('policy_id')!=freeze_b['policy_id'] or
            binding.get('calibration_sha256')!=sha256((root_b/'calibration.json').read_bytes()).hexdigest()):
            raise ValueError('B evaluation calibration binding changed')
    rows_a, rows_b, channels_a, channels_b = [], [], [], []
    statuses, failures, inputs = [], Counter(), []
    missing = Counter();candidate_counts={}
    summaries = {}
    counts = dict(MAP_solves=0, MAP_successes=0, MAP_seconds=0.,
                  objective_evaluations=0, AC_states=0, qp_solves=0,
                  replay_seconds=0.)
    for index in range(expected):
        pa = root_a/phase/f'history_{index:03d}.json.gz'
        pb = root_b/phase/f'history_{index:03d}.json.gz'
        a, b = io.load_record(pa), io.load_record(pb)
        if a['spec'] != b['spec']:
            raise ValueError('paired specification changed')
        source_hash = sha256(pa.read_bytes()).hexdigest()
        metadata_path=pb.with_name(f'history_{index:03d}.metadata.json')
        metadata=io.load_record(metadata_path)
        if (metadata.get('policy_id')!=freeze_b['policy_id'] or
            metadata.get('result_sha256')!=sha256(pb.read_bytes()).hexdigest() or
            metadata.get('spec')!=b['spec'] or not metadata.get('science_source_unchanged')):
            raise ValueError('B result metadata changed')
        if phase=='evaluation' and metadata.get('calibration_binding')!=binding:
            raise ValueError('B worker calibration binding changed')
        bound_hash=b.get('source_A',{}).get('sha256') or metadata.get('source_A_sha256')
        if bound_hash is not None and bound_hash != source_hash:
            raise ValueError('source A raw hash changed')
        if metadata.get('source_A_sha256')!=source_hash:
            raise ValueError('B metadata source A raw hash changed')
        if bound_hash is None:missing['source_hash_bindings']+=1
        candidate_counts[index]=len(a.get('candidate_names',[]))
        missing['B_plans']+=len(set(PLANS)-set(b.get('plans',{})))
        for label, plan in b.get('plans', {}).items():
            if a.get('plans', {}).get(label, {}).get('action_key') != plan['action_key']:
                raise ValueError('saved action changed')
            key = plan['action_key']
            if not np.array_equal(a['actions'][key]['power'], b['actions'][key]['power']):
                raise ValueError('saved power changed')
            for candidate_record in (a,b):
                declared=candidate_record['plans'][label].get('power')
                if declared is not None and not np.array_equal(declared,candidate_record['actions'][key]['power']):
                    raise ValueError('plan power differs from its action power')
            for regime, actual in b['actions'][key]['regimes'].items():
                for field in ('values', 'clean'):
                    if field in actual and not np.array_equal(actual[field], a['actions'][key]['regimes'][regime][field]):
                        raise ValueError('saved observation changed')
                    if field not in actual:missing['B_measurement_fields']+=1
        inputs.append(dict(index=index, A_sha256=source_hash,
                           B_sha256=sha256(pb.read_bytes()).hexdigest()))
        rows_a.extend(old.measurements(a, cal_a));rows_b.extend(old.measurements(b, cal_b))
        channels_a.extend(channel_coverage(a, cal_a));channels_b.extend(channel_coverage(b, cal_b))
        hist = b.get('history_refresh', {})
        candidates = hist.get('candidates', {})
        statuses.append(dict(index=index, A_status=a.get('status'), B_status=b.get('status'),
            history_status=hist.get('status'), candidate_statuses={name: r.get('status') for name, r in candidates.items()},
            B_failures=b.get('failures', {})))
        counts['replay_seconds'] += float(b.get('seconds', 0))
        counts['MAP_solves'] += int(hist.get('MAP_solves', 0))
        for name, row in candidates.items():
            fitted = row.get('MAP', {})
            failures[row.get('status', 'missing')] += 1
            counts['MAP_successes'] += int(bool(fitted.get('success')))
            counts['MAP_seconds'] += float(fitted.get('seconds', 0))
            for key in ('objective_evaluations', 'AC_states', 'qp_solves'):
                counts[key] += int(fitted.get('counters', {}).get(key, 0))
        # Compact statistics are safe to retain; all raw records remain on disk.
        summaries.setdefault('A', []).append(old.compact_record(a))
        summaries.setdefault('B', []).append(old.compact_record(b))
    summary_a = old.summarize(summaries['A'], cal_a)
    summary_b = old.summarize(summaries['B'], cal_b)
    summary_b['summaries'] = {k: v for k, v in summary_b['summaries'].items() if '__AC_' not in k}
    return dict(phase=phase, histories=expected,
        A_policy=freeze_a['policy_id'], B_policy=freeze_b['policy_id'],
        existing_A_fixed_plans_and_measurements_unchanged=True,
        missing_integrity_fields=dict(missing),
        B_result_metadata_verified=True,B_evaluation_calibration_binding=binding,
        generated_new_plans=False, generated_new_histories=False,
        A=summary_a, B=summary_b, paired=paired_rows(rows_a, rows_b, expected),
        A_channel_coverage90=summarize_channels(channels_a),
        B_channel_coverage90=summarize_channels(channels_b),
        B_candidate_status_counts=dict(failures), B_inference_cost=counts,
        B_operational_retain_all=operational_summary(summaries['B'],cal_b,candidate_counts),
        history_statuses=statuses, input_hashes=inputs,
        interpretation='prediction/inference repair on fixed actions; no new MILP planning gain established',
        interval_unit='independent history, averaging the two saved future regimes')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--A-root', required=True);parser.add_argument('--B-root', required=True)
    parser.add_argument('--phase', choices=('development','calibration','evaluation'), default='evaluation')
    parser.add_argument('--expected', type=int, default=32);parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if Path(args.output).exists():raise FileExistsError('preserve existing report')
    result = compare(args.A_root, args.B_root, args.phase, args.expected)
    io.save_record(args.output, result)


if __name__ == '__main__':main()
