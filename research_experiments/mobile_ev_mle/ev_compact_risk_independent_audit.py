"""Independent arithmetic audit of saved compact-policy risk records.

No scientific module is imported.  Compressed trial bytes are hashed, never
decoded; all decisions below use the saved small standard_records only.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from fractions import Fraction
import gzip
import hashlib
import json
import math
from pathlib import Path
import time


PRIMARY = ('refined_v2g_mobile', 'capped_v2g_mobile')
IDENTITY = ('policy_id', 'config_sha256', 'source_manifest_sha256', 'reference_id')


def canonical_hash(value):
    data = json.dumps(value, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=False, allow_nan=False).encode('utf-8')
    return hashlib.sha256(data).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    p = Path(path)
    raw = p.read_bytes()
    return json.loads(gzip.decompress(raw) if p.suffix == '.gz' else raw)


def valid_method(record, candidates):
    losses = record.get('losses')
    return (record.get('status') == 'complete' and isinstance(losses, dict)
            and set(losses) == set(candidates)
            and all(isinstance(v, (int, float)) and not isinstance(v, bool)
                    and math.isfinite(v) and v >= 0 for v in losses.values()))


def projected_records(records, methods):
    out = []
    for record in records:
        local = dict(record, methods={m: record['methods'][m] for m in methods})
        local['candidate_miss'] = bool(record['base_candidate_miss'] or any(
            record['method_candidate_miss'][m] for m in methods))
        out.append(local)
    return out


def coverage_miss(record, spec):
    return (record['true_name'] not in spec['candidates']
            or bool(record.get('candidate_miss'))
            or record['true_name'] not in record['candidate_names'])


def independent_quantile(values, alpha):
    """Exact noninterpolated upper rank, with strict exceedance rejection."""
    values = sorted(values)
    if any(math.isnan(v) or v < 0 for v in values):
        raise ValueError('invalid nonnegative calibration score')
    a = Fraction(str(alpha))
    target = (len(values) + 1) * (1 - a)
    rank = (target.numerator + target.denominator - 1) // target.denominator
    bound = values[rank - 1] if rank <= len(values) else math.inf
    return dict(n=len(values), rank=rank,
                threshold=None if math.isinf(bound) else bound,
                infinite=math.isinf(bound), alpha=float(a),
                rank_marginal_error_bound=(float(Fraction(len(values) + 1 - rank,
                                                          len(values) + 1))
                                           if rank <= len(values) else 0.),
                strict_rejection=True,
                reason=('rank_exceeds_n' if rank > len(values) else
                        'infinite_observed_score' if math.isinf(bound) else
                        'finite_order_statistic'))


def reconstruct_calibration(records, spec, planned_ids):
    """Independently rebuild every score, threshold, field and content hash."""
    methods, candidates = spec['methods'], spec['candidates']
    groups = ['__family__'] if spec['familywise'] else methods
    cells = {s: {a: {m: [] for m in groups} for a in candidates}
             for s in spec['strata']}
    details = []
    misses = outside_count = 0
    for record in records:
        miss = coverage_miss(record, spec)
        outside = record['stratum'] not in spec['strata']
        misses += miss
        outside_count += outside
        scores = {}
        for m in methods:
            row = record['methods'][m]
            if miss or outside:
                scores[m] = math.inf
            elif (set(record['candidate_names']) != set(candidates)
                  or not valid_method(row, candidates)):
                scores[m] = 0.
            else:
                scores[m] = row['losses'][record['true_name']] - min(row['losses'].values())
        details.append(dict(history_id=str(record['history_id']),
            stratum=record['stratum'], true_name=record['true_name'],
            candidate_miss=miss, out_of_reference=outside,
            scores={m: None if math.isinf(s) else float(s) for m, s in scores.items()},
            failed_methods=[m for m in methods if not valid_method(record['methods'][m], candidates)]))
        if not outside and record['true_name'] in candidates:
            target = cells[record['stratum']][record['true_name']]
            for m in groups:
                target[m].append(max(scores.values()) if spec['familywise'] else scores[m])
    thresholds = {s: {a: {m: independent_quantile(v, spec['alpha'])
                         for m, v in d.items()} for a, d in aa.items()}
                  for s, aa in cells.items()}
    blocked = bool(misses or outside_count)
    if blocked:
        for aa in thresholds.values():
            for mm in aa.values():
                for value in mm.values():
                    value.update(threshold=None, infinite=True,
                                 reason='reference_or_coverage_failure_keep_all')
    ids = sorted(str(r['history_id']) for r in records)
    planned = sorted(str(i) for i in planned_ids)
    out = dict(schema='ev_frozen_policy_calibration_v1', spec=deepcopy(spec),
        thresholds=thresholds, history_ids=ids,
        independent_unit_ids=sorted(str(r.get('independent_unit_id', r['history_id'])) for r in records),
        histories_total=len(records), histories_planned=len(planned),
        histories_pending=len(set(planned) - set(ids)), coverage_failures=misses,
        out_of_reference_histories=outside_count, rejection_enabled=not blocked,
        records=details,
        scope='conditional exchangeability in the declared simulated reference distribution only',
        guarantee_kind='marginal over calibration and one new independent history',
        exchangeability_assumed_not_verified=True, conditional_guarantee_available=not blocked,
        fixed_threshold_risk_confidence_claim=False, topology_probability=False)
    out['calibration_id'] = canonical_hash(out)
    return out


def reconstruct_evaluation(records, calibration):
    """Set inversion directly from SSE gaps; no production evaluation import."""
    spec = calibration['spec']
    candidates, methods = spec['candidates'], spec['methods']
    totals = dict(histories_total=len(records), candidate_miss_histories=0,
        out_of_reference_histories=0, any_method_false_rejections=0,
        known_covered_histories=0, globally_unresolved_histories=0)
    method_totals = {m: dict(total=len(records), computational_failures=0,
        operation_failures=0, fit_failures=0, complete_decisions=0,
        false_rejections=0, unique_correct=0, unresolved=0,
        wrong_candidates_excluded=0, known_covered_denominator=0) for m in methods}
    details = []
    for record in records:
        miss = coverage_miss(record, spec)
        outside = record['stratum'] not in spec['strata']
        global_unresolved = miss or outside or not calibration['rejection_enabled']
        totals['candidate_miss_histories'] += miss
        totals['out_of_reference_histories'] += outside
        totals['known_covered_histories'] += not miss
        totals['globally_unresolved_histories'] += global_unresolved
        decisions = {}
        for m in methods:
            row = record['methods'][m]
            valid = valid_method(row, candidates) and set(record['candidate_names']) == set(candidates)
            group = '__family__' if spec['familywise'] else m
            strata = spec['strata'] if spec['stratum_mode'] == 'worst' else [record['stratum']]
            thresholds = {}
            for a in candidates:
                qs = [calibration['thresholds'][s][a][group] for s in strata
                      if s in calibration['thresholds']]
                thresholds[a] = max((math.inf if q['infinite'] else q['threshold']
                                     for q in qs), default=math.inf)
            gaps = {} if global_unresolved or not valid else {
                a: float(row['losses'][a] - min(row['losses'].values())) for a in candidates}
            retained = (list(candidates) if global_unresolved or not valid else
                        [a for a in candidates if gaps[a] <= thresholds[a]])
            false = not miss and record['true_name'] not in retained
            unique = not miss and retained == [record['true_name']]
            unresolved = global_unresolved or not valid or len(retained) != 1
            stats = method_totals[m]
            stats['computational_failures'] += not valid
            status = row['status'].lower()
            stats['operation_failures'] += not valid and ('operation' in status or 'design' in status)
            stats['fit_failures'] += not valid and ('fit' in status or 'historical' in status)
            stats['complete_decisions'] += valid and not global_unresolved
            stats['known_covered_denominator'] += not miss
            stats['false_rejections'] += false
            stats['unique_correct'] += unique
            stats['unresolved'] += unresolved
            stats['wrong_candidates_excluded'] += (sum(a != record['true_name'] and a not in retained
                                                     for a in candidates) if not miss else 0)
            decisions[m] = dict(retained=retained, rejected=[a for a in candidates if a not in retained],
                gaps=gaps, thresholds={a: None if math.isinf(t) else t for a, t in thresholds.items()},
                infinite_thresholds=[a for a, t in thresholds.items() if math.isinf(t)],
                status=row['status'], compute_failure=not valid, global_unresolved=global_unresolved,
                false_rejection=false if not miss else None, unique_correct=unique, unresolved=unresolved)
        totals['any_method_false_rejections'] += any(d['false_rejection'] for d in decisions.values())
        details.append(dict(history_id=str(record['history_id']), true_name=record['true_name'],
            stratum=record['stratum'], candidate_miss=miss, out_of_reference=outside, methods=decisions))
    return dict(schema='ev_frozen_policy_calibration_v1', calibration_id=calibration['calibration_id'],
        spec=deepcopy(spec), totals=totals, methods=method_totals, records=details,
        empirical_counts_only=True, topology_probability=False)


def audit(before_path, final_path, *, verify_current_raw=True):
    before, final = read_json(before_path), read_json(final_path)
    errors, checks = [], []
    def check(name, passed, detail=None):
        checks.append(dict(name=name, passed=bool(passed), detail=detail))
        if not passed:
            errors.append(name)
    check('36 calibration and 12 evaluation',
          before['actual_calibration'] == final['actual_calibration'] == 36
          and before['actual_evaluation'] == 0 and final['actual_evaluation'] == 12)
    bundles = [x['standard_records'] for x in (before, final)]
    for label, bundle in zip(('before', 'final'), bundles):
        check(label + ' bundle content hash', canonical_hash({k: v for k, v in bundle.items()
              if k != 'bundle_id'}) == bundle.get('bundle_id'))
        check(label + ' root/raw bundle manifest equality',
              (before if label == 'before' else final)['raw_manifest'] == bundle['raw_manifest'])
    b, f = bundles
    check('standard calibration records unchanged', b['calibration'] == f['calibration'])
    check('frozen standard spec unchanged', b['spec'] == f['spec'])
    check('planned IDs unchanged', all(b[k] == f[k] for k in
                                      ('planned_calibration_ids', 'planned_evaluation_ids')))
    check('calibration raw manifests unchanged', b['raw_manifest']['calibration'] == f['raw_manifest']['calibration'])
    methods, candidates = f['spec']['methods'], f['spec']['candidates']
    check('eleven methods/four candidates', len(methods) == 11 and len(set(methods)) == 11
          and len(candidates) == 4 and len(set(candidates)) == 4)
    phase_ids, phase_units = {}, {}
    for phase, count in (('calibration', 36), ('evaluation', 12)):
        rows = f[phase]
        ids = [str(r['history_id']) for r in rows]
        units = [str(r['independent_unit_id']) for r in rows]
        phase_ids[phase], phase_units[phase] = set(ids), set(units)
        check(phase + ' records complete unique planned', len(rows) == count and len(set(ids)) == count
              and len(set(units)) == count and set(ids) == set(f['planned_' + phase + '_ids']))
        check(phase + ' every record frozen identities/methods', all(
            all(r[k] == f['spec'][k] for k in IDENTITY) and set(r['methods']) == set(methods)
            and set(r['candidate_names']) == set(candidates) for r in rows))
        check(phase + ' four family fixed quotas',
              Counter(r['true_name'] for r in rows) == {a: count // 4 for a in candidates})
    check('disjoint calibration/evaluation history and independent IDs',
          not phase_ids['calibration'] & phase_ids['evaluation']
          and not phase_units['calibration'] & phase_units['evaluation'])
    raw_checks = []
    for phase, count in (('calibration', 36), ('evaluation', 12)):
        directory = Path(f['raw_directories'][phase])
        reviewed = read_json(directory / 'review_pair_completion.json')
        review_sha = reviewed['actual_raw_sha256']
        manifest = f['raw_manifest'][phase]
        check(phase + ' existing full/qp audit passed', reviewed['success'] is True and not reviewed['errors']
              and reviewed['source_count'] == 72 and all(j['success'] and j['audited'] == count
                  and j['available'] == count for j in reviewed['jobs']))
        check(phase + ' manifest exact reviewed/raw coverage', len(manifest) == count
              and {r['name']: r['sha256'] for r in manifest} == review_sha
              and {str(r['seed']) for r in manifest} == phase_ids[phase])
        for entry in manifest:
            p = directory / entry['name']
            stat = p.stat()
            observed = file_hash(p) if verify_current_raw else review_sha[entry['name']]
            passed = (observed == entry['sha256'] and stat.st_size == entry['compressed_bytes']
                      and stat.st_mtime_ns == entry['mtime_ns'])
            check(phase + ' compressed raw ' + p.name, passed)
            raw_checks.append(dict(phase=phase, name=p.name, sha256=observed,
                                  compressed_bytes=stat.st_size, unchanged=passed))
        check(phase + ' prior audit freeze bytes unchanged',
              file_hash(directory / 'freeze.json') == reviewed['freeze_sha256'])
    cell_rows, replay_summary = [], {}
    for label, use_methods, familywise in (
            ('individual_policies', methods, False), ('primary_family', list(PRIMARY), True)):
        old_cal, saved_cal = before[label]['calibration'], final[label]['calibration']
        check(label + ' complete frozen calibration unchanged', old_cal == saved_cal)
        check(label + ' calibration ID verified', saved_cal['calibration_id'] == canonical_hash(
            {k: v for k, v in saved_cal.items() if k != 'calibration_id'}))
        spec = dict(f['spec'], methods=use_methods, familywise=familywise)
        cr = f['calibration'] if not familywise else projected_records(f['calibration'], use_methods)
        er = f['evaluation'] if not familywise else projected_records(f['evaluation'], use_methods)
        rebuilt_cal = reconstruct_calibration(cr, spec, f['planned_calibration_ids'])
        check(label + ' independent whole calibration reconstruction', rebuilt_cal == saved_cal)
        evaluation = reconstruct_evaluation(er, rebuilt_cal)
        check(label + ' independent whole decision reconstruction', evaluation == final[label]['evaluation'])
        for s, aa in saved_cal['thresholds'].items():
            for a, mm in aa.items():
                for m, q in mm.items():
                    cell_rows.append(dict(group=label, stratum=s, candidate=a, method=m, **q))
        replay_summary[label] = dict(calibration_id=saved_cal['calibration_id'],
            evaluation_totals=evaluation['totals'], methods=evaluation['methods'],
            decisions_sha256=canonical_hash(evaluation),
            false_rejection_history_ids=[r['history_id'] for r in evaluation['records']
                if any(x['false_rejection'] for x in r['methods'].values())])
    check('44 individual and 4 family cells n9 rank9',
          Counter(r['group'] for r in cell_rows) == {'individual_policies': 44, 'primary_family': 4}
          and all(r['n'] == 9 and r['rank'] == 9 and r['strict_rejection']
                  for r in cell_rows))
    return dict(schema='ev_compact_risk_independent_audit_v1', success=not errors,
        errors=errors, checks=checks,
        input_sha256={str(p): file_hash(p) for p in (Path(before_path), Path(final_path))},
        input_paths=dict(before=str(Path(before_path).resolve()), final=str(Path(final_path).resolve())),
        historical_units=dict(calibration=36, evaluation=12, disjoint=True),
        policy_id=f['spec']['policy_id'], standard_bundle_id=f['bundle_id'],
        threshold_cells=cell_rows, summary=replay_summary, compressed_raw=raw_checks,
        raw_bytes_rehashed=verify_current_raw, large_raw_json_decoded=False,
        no_new_fit_or_selection=True, production_calibration_code_imported=False,
        interpretation='descriptive held-out counts; rank guarantee assumes declared conditional exchangeability',
        fixed_threshold_risk_confidence_claim=False, topology_probability=False,
        auditor_source_sha256=file_hash(__file__))


def wait_stable(paths, seconds=30):
    previous = None
    while True:
        signature = tuple((str(p), p.stat().st_size, p.stat().st_mtime_ns)
                          for p in paths if p.exists())
        if len(signature) == len(paths) and signature == previous:
            return
        previous = signature
        time.sleep(seconds)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--before', type=Path, required=True)
    p.add_argument('--final', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--wait-stable', action='store_true')
    args = p.parse_args()
    if args.wait_stable:
        wait_stable((args.before, args.final))
    result = audit(args.before, args.final)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, sort_keys=True, ensure_ascii=False, indent=2,
                                      allow_nan=False), encoding='utf-8')
    print(json.dumps(dict(success=result['success'], errors=result['errors'],
                         summary=result['summary']), ensure_ascii=False), flush=True)
    raise SystemExit(0 if result['success'] else 1)


if __name__ == '__main__':
    main()
