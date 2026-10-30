"""Audit saved historical candidates; no solver, truth, NPZ or test losses."""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def family(point):
    return tuple(sorted(tuple(sorted(s)) for s in point['supports']))


def contracted(point, cal):
    return tuple(sorted(tuple(sorted(s)) for s, r, x in zip(point['supports'], point['r'], point['x'])
                        if len(s) == 1 or cal['a']*r+cal['b']*x+1e-10 >= .16*cal['median_root_depth']))


def audit(row):
    record = dict(job_id=row['job_id'], case=row['case'], bundle=row['bundle'],
                  voltage_sigma=row['voltage_sigma'], N=row['N'])
    try:
        entry = row['existing_result']
        if not entry['exists']:
            return dict(record, status='missing_result')
        path = Path(entry['path'])
        digest = sha(path)
        if digest != entry['catalog_sha256']:
            return dict(record, status='result_hash_mismatch', actual_sha256=digest)
        saved = json.loads(path.read_text(encoding='utf-8'))
        points = saved.get('base_points', [])
        models = saved.get('models', {})
        if not points or any(m not in models for m in ('multistart', 'RNJ_L1_bank')):
            return dict(record, status='existing_failure_or_incomplete', original_status=saved.get('status'))
        if saved.get('truth_used_by_estimator') is not False:
            return dict(record, status='estimator_truth_flag_not_false')
        losses = [p['validation_mae'] for p in points]
        if not all(math.isfinite(v) and v >= 0 for v in losses):
            return dict(record, status='invalid_saved_validation_loss')
        best = min(losses)
        indices = [i for i, v in enumerate(losses) if v <= 1.03*best+1e-12]
        chosen = min(indices, key=lambda i:(len(points[i]['supports']), losses[i], i))
        outputs = {contracted(points[i], saved['calibration']) for i in indices}
        raw = {family(points[i]) for i in indices}
        reproduced = contracted(points[chosen], saved['calibration'])
        selected = family(models['multistart'])
        if reproduced != selected:
            return dict(record, status='saved_multistart_selection_mismatch')
        n_family = family(models['RNJ_L1_bank'])
        return dict(record, status='audited', result_sha256=digest,
                    raw_candidate_points=len(points), eligible_point_indices=indices,
                    eligible_raw_families=len(raw), eligible_contracted_families=len(outputs),
                    contracted_families=[[list(s) for s in f] for f in sorted(outputs)],
                    selected_index=chosen, selected_origin=points[chosen].get('origin'),
                    RNJ_multistart_same_final_support=(n_family == selected),
                    RNJ_final_support=[list(s) for s in n_family],
                    multistart_final_support=[list(s) for s in selected],
                    historical_candidate_diversity=(len(outputs) > 1),
                    full_parameter_coverage=False, topology_correctness_evaluated=False)
    except Exception as error:
        return dict(record, status='audit_error', error_type=type(error).__name__, message=str(error))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--protocol', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise RuntimeError('Existing bounded audit retained; do not overwrite')
    cfg = json.loads(a.protocol.read_text(encoding='utf-8'))
    for source, digest in cfg['source_sha256'].items():
        if sha(source) != digest:
            raise RuntimeError('Frozen source changed: '+source)
    inventory = json.loads(Path(cfg['inventory']).read_text(encoding='utf-8'))
    rows = [audit(r) for r in inventory['rows']]
    good = [r for r in rows if r['status'] == 'audited']
    group = {}
    for sigma in sorted({r['voltage_sigma'] for r in rows}):
        g = [r for r in good if r['voltage_sigma'] == sigma]
        group[str(sigma)] = dict(planned=sum(r['voltage_sigma'] == sigma for r in rows), audited=len(g),
            multiple_historical_outputs=sum(r['historical_candidate_diversity'] for r in g),
            RNJ_multistart_different=sum(not r['RNJ_multistart_same_final_support'] for r in g))
    result = dict(created_utc=datetime.now(timezone.utc).isoformat(), protocol_sha256=sha(a.protocol),
        status='complete', planned=len(rows), statuses=dict(Counter(r['status'] for r in rows)),
        audited=len(good), multiple_historical_outputs=sum(r['historical_candidate_diversity'] for r in good),
        RNJ_multistart_different=sum(not r['RNJ_multistart_same_final_support'] for r in good),
        by_voltage_sigma=group, records=rows, solver_calls=0, AC_calls=0, NPZ_files_loaded=0,
        new_future_trajectories_read=False, topology_truth_evaluated=False,
        scope='Saved finite historical candidate diversity only; 3 percent is not a confidence region, RNJ/WZZT disagreement is not a recovery gain')
    a.output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({k:result[k] for k in ('status','planned','statuses','multiple_historical_outputs','RNJ_multistart_different','by_voltage_sigma')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
