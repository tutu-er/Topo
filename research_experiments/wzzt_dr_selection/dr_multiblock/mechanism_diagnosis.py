"""Read-only diagnosis of completed stochastic DR fits; no solver imports."""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import numpy as np

METHODS = ('rnj_bank', 'baseline25', 'multistart_original_rule')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def effective(point, labels):
    return {tuple(labels[i] for i in s) for s, r, x in zip(point['supports'], point['r'], point['x'])
            if max(r, x) > 1e-9}


def score(point, labels, truth):
    active = effective(point, labels)
    return {'all_block_errors': len(active ^ truth), 'complete': active == truth,
            'effective_supports': [list(s) for s in sorted(active)],
            'missing': [list(s) for s in sorted(truth - active)],
            'extra': [list(s) for s in sorted(active - truth)]}


def pack_row(row):
    z = np.asarray(row['participation'], dtype=int)
    final = row['evaluation']['models']['multistart_original_rule']
    return {'key': row['key'], 'observation_key': row['observation_key'],
            'law': row['law'], 'policy': row['policy'], 'path': row.get('path'),
            'plan': row['plan'], 'participation': row['participation'],
            'training_count': int(z[:2].sum()), 'validation_count': int(z[2:].sum()),
            'training_rank': int(np.linalg.matrix_rank(z[:2])),
            'total_payment': row['physical']['payment_currency'],
            'event_payments': row['physical']['event_payments'],
            'training_delta_p_squared_kw2': float(2 * 0.2 ** 2 * z[:2].sum()),
            'validation_delta_p_squared_kw2': float(2 * 0.2 ** 2 * z[2:].sum()),
            'all_block_errors': final['final']['all_block_errors'],
            'complete': final['final']['complete_effective_family'],
            'missing': final['final']['missing_effective_supports'],
            'extra': final['final']['extra_effective_supports'],
            'selected_origin': final['selected_origin'], 'selected_path_index': final.get('path_index')}


def run(root):
    root = Path(root)
    paths = sorted((root / 'records').glob('*.json'))
    rows = [read(p) for p in paths]
    if any(r['status'] != 'evaluated' for r in rows):
        raise ValueError('All records must have completed for this diagnosis')
    summary, planning, analysis = (read(root / n) for n in ('summary.json', 'planning.json', 'analysis.json'))
    unique = {}
    for row in rows:
        unique.setdefault(row['observation_key'], row)
    fits = {}; discrepancies = []; all_steps = []; fit_details = []; unique_origins = Counter()
    for key, row in unique.items():
        file = Path(row['fit_result'])
        if sha(file) != row['fit_sha256']:
            raise AssertionError('Fit hash changed: ' + key)
        fit = read(file); fits[key] = fit
        labels = fit['terminal_labels']; truth = {tuple(s) for s in row['true_supports']}
        calculated = {m: score(fit['models'][m], labels, truth) for m in METHODS}
        for m in METHODS:
            if calculated[m]['all_block_errors'] != row['evaluation']['models'][m]['final']['all_block_errors']:
                raise AssertionError('Structural score differs: ' + key)
        families = [effective(fit['models'][m], labels) for m in METHODS]
        params_equal = all(fit['models'][m]['supports'] == fit['models']['rnj_bank']['supports']
                           and np.allclose(fit['models'][m]['r'], fit['models']['rnj_bank']['r'], atol=1e-12, rtol=0)
                           and np.allclose(fit['models'][m]['x'], fit['models']['rnj_bank']['x'], atol=1e-12, rtol=0)
                           for m in METHODS)
        if any(f != families[0] for f in families[1:]) or not params_equal:
            discrepancies.append({'key': key, 'representative': row['key'],
                                  'families_equal': all(f == families[0] for f in families),
                                  'weights_equal_1e12': params_equal, 'scores': calculated})
        unique_origins.update(fit['models'][m]['origin'] for m in METHODS)
        step_records = []
        for name, trajectory in fit['searches'].items():
            for i, (before, after) in enumerate(zip(trajectory['path'][:-1], trajectory['path'][1:])):
                decision = trajectory['decisions'][i]
                best = None
                for candidate in decision['candidates']:
                    if best is None or candidate['objective'] < best['objective'] - 1e-12:
                        best = candidate
                if best is None or abs(best['objective'] - after['objective']) > 1e-10:
                    raise AssertionError('Could not reconstruct accepted best candidate')
                gain = before['objective'] - after['objective']
                if gain <= 1e-10 + .001 * before['objective']:
                    raise AssertionError('Stored accepted step did not cross threshold')
                removed = [[labels[j] for j in s] for s in best['removed']]
                step = {'fit_key': key, 'representative': row['key'], 'search': name, 'step': i + 1,
                        'candidate': [labels[j] for j in best['candidate']],
                        'crossing_removed': removed, 'crossing': bool(removed),
                        'objective_before': before['objective'], 'objective_after': after['objective'],
                        'gain': gain, 'acceptance_threshold': 1e-10 + .001 * before['objective'],
                        'before': score(before, labels, truth), 'after': score(after, labels, truth)}
                step_records.append(step); all_steps.append(step)
        fit_details.append({'fit_key': key, 'representative': row['key'], 'fit_sha256': sha(file),
                            'models': {m: {'origin': fit['models'][m]['origin'],
                                           'path_index': fit['models'][m].get('path_index'),
                                           'selection_index': fit['models'][m]['selection_index'],
                                           **calculated[m]} for m in METHODS},
                            'model_weights_equal_1e12': params_equal,
                            'accepted_steps': len(step_records),
                            'accepted_crossing_steps': sum(s['crossing'] for s in step_records)})

    # Same independent execution path across the two policy classes, then stricter
    # post-hoc matching by realized counts and total payment across all stored paths.
    local = [r for r in rows if r['phase'] == 'main' and r['law'] == 'local']
    sync = [pack_row(r) for r in local if r['policy'] == 'stochastic_sync']
    free = [pack_row(r) for r in local if r['policy'] == 'stochastic_free']
    pairs = []
    for a in sync:
        for b in free:
            if (a['training_count'], a['validation_count']) == (b['training_count'], b['validation_count']) and abs(a['total_payment'] - b['total_payment']) <= 1e-12:
                pairs.append({'same_execution_path': a['path'] == b['path'],
                              'same_event_payment_vector': bool(np.allclose(a['event_payments'], b['event_payments'], atol=1e-12, rtol=0)),
                              'sync': a, 'free': b,
                              'free_minus_sync_errors': b['all_block_errors'] - a['all_block_errors']})
    pairs.sort(key=lambda p: (not p['same_execution_path'], p['sync']['key'], p['free']['key']))
    local_groups = []
    for policy in ('stochastic_sync', 'stochastic_free', 'mean_free'):
        rr = [r for r in local if r['policy'] == policy]
        if not rr:
            continue
        plan = np.asarray(rr[0]['plan']).reshape(2, 5)
        probs = np.asarray([0, .5, .75])[plan]
        local_groups.append({'policy': policy, 'plan': rr[0]['plan'], 'denominator': len(rr),
                             'expected_successful_training_offers': float(probs.sum()),
                             'actual_mean_successful_training_offers': float(np.mean([np.asarray(r['participation'])[:2].sum() for r in rr])),
                             'actual_mean_successful_validation_offers': float(np.mean([np.asarray(r['participation'])[2:].sum() for r in rr])),
                             'mean_payment': float(np.mean([r['physical']['payment_currency'] for r in rr])),
                             'mean_errors': float(np.mean([r['evaluation']['models']['multistart_original_rule']['final']['all_block_errors'] for r in rr])),
                             'full_recoveries': sum(r['evaluation']['models']['multistart_original_rule']['final']['complete_effective_family'] for r in rr)})
    complete = []
    for row in rows:
        if row['evaluation']['models']['multistart_original_rule']['final']['complete_effective_family']:
            fit = fits[row['observation_key']]
            complete.append({**pack_row(row), 'phase': row['phase'], 'case': row['case'],
                             'rnj_already_complete': row['evaluation']['models']['rnj_bank']['final']['complete_effective_family'],
                             'origins': {m: fit['models'][m]['origin'] for m in METHODS},
                             'path_steps': {name: len(p['path']) - 1 for name, p in fit['searches'].items()}})
    controls = []
    for row in rows:
        if row['phase'] == 'control':
            controls.append({**pack_row(row), 'case': row['case'],
                             'truth_effective_supports': row['true_supports'],
                             'pool_coverage': row['evaluation']['pool_coverage'],
                             'before_final_contraction': row['evaluation']['models']['multistart_original_rule']['before_final_contraction'],
                             'models': {m: row['evaluation']['models'][m]['final'] for m in METHODS}})
    result = {'created_at': datetime.now(timezone.utc).isoformat(),
              'script_sha256': sha(__file__), 'no_new_optimization': True, 'no_new_simulation': True,
              'record_count': len(rows), 'main_record_count': sum(r['phase'] == 'main' for r in rows),
              'control_record_count': len(controls), 'unique_fits': len(unique),
              'model_discrepancy_count': len(discrepancies), 'model_discrepancies': discrepancies,
              'final_structure_discrepancy_count': sum(not d['families_equal'] for d in discrepancies),
              'weight_only_discrepancy_count': sum(d['families_equal'] and not d['weights_equal_1e12'] for d in discrepancies),
              'unique_model_origin_counts': dict(unique_origins),
              'accepted_step_count': len(all_steps),
              'accepted_crossing_step_count': sum(s['crossing'] for s in all_steps),
              'accepted_added_block_counts': dict(Counter(','.join(map(str,s['candidate'])) for s in all_steps)),
              'accepted_structure_improvement_count': sum(s['after']['all_block_errors'] < s['before']['all_block_errors'] for s in all_steps),
              'selected_after_greedy_step_count': sum(d['models'][m]['path_index'] is not None and d['models'][m]['path_index'] > 0 for d in fit_details for m in METHODS),
              'accepted_steps': all_steps, 'fit_details': fit_details,
              'full_recovery_records': complete, 'local_policy_comparisons': local_groups,
              'matched_response_count_payment_pairs': pairs,
              'controls': controls,
              'best_single_references': [c for c in analysis['comparisons']
                                         if c['method'] == 'multistart_original_rule' and c['comparison'] == 'posthoc_best_fixed_single'],
              'planning_ties': {law: {c: {k: v for k, v in d.items() if k != 'participation_patterns'}
                                     for c, d in planning['selected'][law].items()} for law in planning['selected']},
              'limitations': ['Matched examples are post-hoc explanations, not additional independent experiments.',
                              'The same-path count/payment match changes one event location from outside S (105) to inside A (102); it separates response amount from location but does not isolate diversity from node usefulness.',
                              'Final model equality can coexist with accepted exploratory paths; equality rules out an extra final benefit from those paths in these records.',
                              'Input diversity and support recovery are different endpoints; the primary mean-error endpoint does not beat the post-hoc best fixed single schedule.',
                              'Declared local/common response laws are synthetic dependence controls, not calibrated user probabilities.',
                              'Symmetric planner ties select labels deterministically; locations were not learned from topology.'],
              'source_hashes': {n: sha(root / n) for n in ('protocol.json', 'planning.json', 'summary.json', 'analysis.json')},
              'record_manifest_sha256': hashlib.sha256(json.dumps({p.name: sha(p) for p in paths}, sort_keys=True).encode()).hexdigest()}
    destination = root / 'mechanism_diagnosis.json'
    destination.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args(); result = run(args.root)
    print(json.dumps({k: result[k] for k in ('record_count', 'unique_fits', 'model_discrepancy_count', 'unique_model_origin_counts', 'accepted_step_count', 'accepted_crossing_step_count')}, indent=2))
    print(json.dumps(result['local_policy_comparisons'], indent=2))
    print(json.dumps({'full_recovery_records': [r['key'] for r in result['full_recovery_records']],
                      'matched_pairs': len(result['matched_response_count_payment_pairs']),
                      'same_path_pairs': [p for p in result['matched_response_count_payment_pairs'] if p['same_execution_path']]}, indent=2))
