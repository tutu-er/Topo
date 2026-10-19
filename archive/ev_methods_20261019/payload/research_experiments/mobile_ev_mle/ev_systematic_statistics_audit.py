"""Independent, standard-library-only calibration/decision arithmetic audit.

Imports no production analyzer, selector, fit, adequacy or stratification code.
Evaluation records are opened only when --evaluation is explicitly supplied.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from fractions import Fraction
import gzip
from hashlib import sha256
import json
import math
from pathlib import Path
from time import perf_counter

METHODS = ('none', 'constant', 'highlow', 'square_fixed', 'square_mobile',
           'nearest_fixed', 'gradac_fixed', 'scalar_fixed', 'scalar_hist_fixed', 'scalar_mobile')
MODES = ('fixed_total', 'fixed_holdout', 'joint_total')
GROUPS = ('8', '32', '96')


def read(path):
    path = Path(path)
    with (gzip.open(path, 'rt', encoding='utf8') if path.suffix == '.gz' else path.open(encoding='utf8')) as stream:
        return json.load(stream)


def finite(value):
    return isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value)


def cutoff(scores, alpha):
    """Exact integer rank ceil((n+1)(1-alpha)); no numerical quantile API."""
    a = Fraction(str(alpha)); n = len(scores)
    numerator = (n + 1) * (a.denominator - a.numerator)
    rank = (numerator + a.denominator - 1) // a.denominator
    return dict(n=n, alpha=float(a), order_rank=rank, infinite=rank > n,
                threshold=None if rank > n else sorted(scores)[rank - 1])


def rows_from(directory):
    paths = sorted(Path(directory).glob('case_*.json.gz'))
    paths += sorted(Path(directory).glob('case_*.json'))
    rows, inputs = [], []
    for path in paths:
        record = read(path)
        # Raw inputs are also accepted without keeping large fit/table arrays.
        fields = ('case_id', 'seed', 'phase', 'true_name', 'candidate_miss', 'candidates',
                  'candidate_names', 'config', 'methods', 'policy_id', 'source_sha256')
        rows.append({k: record[k] for k in fields if k in record})
        inputs.append(dict(path=str(path.resolve()), sha256=sha256(path.read_bytes()).hexdigest(),
            raw_sha256=record.get('delivery_provenance', {}).get('raw_sha256', sha256(path.read_bytes()).hexdigest())))
    return rows, inputs


def values(record, name, mode):
    """Compute scores directly from losses, preserving failed/missing semantics."""
    methods = {row['name']: row for row in record['methods']}
    method = methods[name]; result = method['inference'][mode]
    declared = record.get('candidate_names', record.get('candidates', []))
    bank = sorted({row if isinstance(row, str) else row.get('name', row.get('candidate')) for row in declared})
    if not bank:
        bank = sorted({candidate for row in record['methods'] for v in row['inference'].values() for candidate in v['losses']})
    losses = result['losses']; true = record['true_name']
    complete = bool(result['complete'] and method.get('operation', {}).get('success', True)
        and bank and set(losses) == set(bank) and all(finite(v) and v >= 0 for v in losses.values()))
    miss = bool(record['candidate_miss'] or true is None or true not in bank)
    config = record.get('config', {}); H, V, T = (config.get(k) for k in ('training_points', 'validation_points', 'horizon'))
    power = method.get('power_kw'); N = None
    metadata_error = None
    if (not all(isinstance(v, int) and not isinstance(v, bool) for v in (H, V, T))
            or H <= 0 or V < 0 or T <= 0):
        metadata_error = 'invalid time counts'
    elif (not isinstance(power, list) or len(power) != T or not power
            or not isinstance(power[0], list) or not power[0]
            or any(not isinstance(row, list) or len(row) != len(power[0]) or not all(finite(p) for p in row) for row in power)):
        metadata_error = 'invalid power shape'
    else:
        N = ((V + T) if mode == 'fixed_holdout' else H + V + T) * (len(power[0]) + 2)
    relative = {k: max(0., v - min(losses.values())) for k, v in losses.items()} if complete else {}
    absolute = {k: v / N for k, v in losses.items()} if complete and N is not None else {}
    return dict(bank=bank, true=true, complete=complete, candidate_miss=miss, relative=relative,
        absolute=absolute, adequate_complete=complete and N is not None, Nobs=N, H_group=str(H) if str(H) in GROUPS else None,
        R_true=relative[true] if complete and not miss else 0.,
        A_true=absolute[true] if absolute and not miss else 0., metadata_failure=metadata_error)


class Audit:
    def __init__(self):
        self.checks = 0; self.errors = []

    def equal(self, actual, expected, label):
        self.checks += 1
        if actual != expected:
            self.errors.append(dict(check=label, actual=sorted(actual) if isinstance(actual, set) else actual,
                expected=sorted(expected) if isinstance(expected, set) else expected))

    def yes(self, condition, label):
        self.equal(bool(condition), True, label)


def audit_calibration(rows, inputs, primary, dual, stratified, audit):
    audit.equal(len(rows), 108, 'calibration original denominator')
    audit.equal(len({r['seed'] for r in rows}), len(rows), 'calibration unique histories')
    audit.yes(all(r['phase'] in ('cal', 'calibration') for r in rows), 'calibration phase')
    groups = {h: [r for r in rows if str(r['config']['training_points']) == h] for h in GROUPS}
    audit.equal({h: len(v) for h, v in groups.items()}, {'8': 16, '32': 18, '96': 74}, 'predeclared H calibration counts')
    tables = []; raw_hashes = sorted(x['raw_sha256'] for x in inputs)
    policy_ids = {r['policy_id'] for r in rows}
    source_digests = {sha256((json.dumps(r['source_sha256'], ensure_ascii=False, sort_keys=True,
        indent=2, allow_nan=False) + '\n').encode('utf8')).hexdigest() for r in rows}
    for label, threshold in (('primary', primary), ('dual', dual), ('stratified', stratified)):
        audit.equal(sorted(x['sha256'] for x in threshold['calibration_files']), raw_hashes, label + ': raw input cohort')
        audit.equal(threshold['calibration_count'], len(rows), label + ': saved denominator')
        audit.equal(policy_ids, {threshold['identity']['policy_id']}, label + ': policy identity')
        audit.equal(source_digests, {threshold['identity']['source_sha256_digest']}, label + ': source identity')
        audit.equal(set(threshold['methods']), set(METHODS), label + ': method set')
        audit.yes(threshold['thresholds_saved_before_evaluation_read'], label + ': declared blind threshold save')
    audit.equal(stratified['group_counts'], {h: len(v) for h, v in groups.items()}, 'saved H counts')
    for name in METHODS:
        for mode in MODES:
            score = [values(r, name, mode) for r in rows]
            audit.yes(not any(v['candidate_miss'] for v in score), name + '/' + mode + ': true candidate coverage')
            rs = [v['R_true'] for v in score]; ass = [v['A_true'] for v in score]
            expected = cutoff(rs, '.1'); saved = primary['methods'][name][mode]
            for key, value in expected.items(): audit.equal(saved[key], value, 'primary/' + name + '/' + mode + '/' + key)
            audit.equal(saved['scores'], rs, 'primary/' + name + '/' + mode + '/all108scores')
            audit.equal(saved['incomplete_calibration'], sum(not v['complete'] for v in score), 'primary failures')
            tables.append(dict(rule='primary', method=name, mode=mode, **expected))
            aux = dual['methods'][name][mode]
            for gate, sample in (('R', rs), ('A', ass)):
                expected = cutoff(sample, '.05')
                for key, value in expected.items(): audit.equal(aux[gate][key], value, 'dual/' + gate + '/' + name + '/' + mode + '/' + key)
                audit.equal(aux[gate + '_scores'], sample, 'dual/' + gate + '/all108scores/' + name + '/' + mode)
                tables.append(dict(rule='dual_' + gate, method=name, mode=mode, **expected))
            audit.equal(aux['incomplete_calibration'], sum(not v['adequate_complete'] for v in score), 'dual failures')
            audit.equal(aux['metadata_failures'], sum(v['metadata_failure'] is not None for v in score), 'dual metadata failures')
            audit.equal(aux['Nobs_counts'], dict(Counter(str(v['Nobs']) for v in score)), 'dual actual output counts')
            reference = stratified['global_reference'][name][mode]
            for key, value in cutoff(rs, '.1').items(): audit.equal(reference[key], value, 'H/global reference/' + name + '/' + mode + '/' + key)
            tables.append(dict(rule='H_global_reference', method=name, mode=mode, **cutoff(rs, '.1')))
            for h, group in groups.items():
                local = [values(r, name, mode) for r in group]; sample = [v['R_true'] for v in local]
                expected = cutoff(sample, '.1'); saved = stratified['methods'][name][mode][h]
                for key, value in expected.items(): audit.equal(saved[key], value, 'H/' + h + '/' + name + '/' + mode + '/' + key)
                audit.equal(saved['scores'], sample, 'H/all group scores/' + h + '/' + name + '/' + mode)
                audit.equal(saved['incomplete_calibration'], sum(not v['complete'] for v in local), 'H failures')
                tables.append(dict(rule='H_' + h, method=name, mode=mode, **expected))
    return dict(histories=len(rows), unique_histories=len({r['seed'] for r in rows}),
        group_counts={h: len(v) for h, v in groups.items()}, threshold_blocks=len(tables), thresholds=tables)


def limit(threshold):
    return math.inf if threshold['infinite'] else threshold['threshold']


def decide(value, rule, threshold):
    """Direct set comparison from losses and already-saved numeric cutoffs."""
    bank = value['bank']; true = value['true']; miss = value['candidate_miss']
    complete = value['adequate_complete'] if rule == 'dual' else value['complete']
    if rule != 'dual' and miss:
        return dict(category='candidate_miss', retained=bank, false_reject=None)
    if not complete:
        return dict(category='candidate_miss' if miss else 'failed_keep_all', retained=bank,
                    false_reject=None if miss else False)
    if rule == 'dual':
        retained = sorted(k for k in bank if value['relative'][k] <= limit(threshold['R']) and value['absolute'][k] <= limit(threshold['A']))
        all_bad = all(v > limit(threshold['A']) for v in value['absolute'].values())
    else:
        retained = sorted(k for k in bank if value['relative'][k] <= limit(threshold))
        all_bad = False
    category = ('candidate_miss' if miss else 'empty_all_models_fail_adequacy' if not retained else
                'unique_correct' if retained == [true] else 'unique_wrong' if len(retained) == 1 else 'unresolved')
    return dict(category=category, retained=retained, false_reject=None if miss else true not in retained,
                all_models_fail_adequacy=all_bad)


def audit_evaluation(rows, inputs, calibration, primary, dual, stratified, references, audit):
    audit.equal(len(rows), 72, 'evaluation original denominator')
    audit.equal(len({r['seed'] for r in rows}), len(rows), 'evaluation unique histories')
    audit.yes(not ({r['seed'] for r in rows} & {r['seed'] for r in calibration}), 'independent calibration/evaluation histories')
    audit.yes(all(r['phase'] in ('eval', 'evaluation') for r in rows), 'evaluation phase')
    decisions = {}; aggregates = defaultdict(lambda: defaultdict(dict))
    for rule in ('primary', 'dual', 'H'):
        per_history = []
        for r in rows:
            methods = {}
            for name in METHODS:
                methods[name] = {}
                for mode in MODES:
                    value = values(r, name, mode)
                    threshold = (primary['methods'][name][mode] if rule == 'primary' else
                                 dual['methods'][name][mode] if rule == 'dual' else
                                 stratified['methods'][name][mode].get(value['H_group']))
                    result = decide(value, 'dual' if rule == 'dual' else 'R', threshold or cutoff([], '.1'))
                    if rule == 'H' and (threshold is None or threshold['n'] == 0) and value['complete'] and not value['candidate_miss']:
                        result.update(category='missing_group_keep_all', retained=value['bank'], false_reject=False)
                    methods[name][mode] = result
            per_history.append(dict(case_id=r['case_id'], seed=r['seed'], methods=methods))
        decisions[rule] = per_history
        reference = references.get(rule)
        if reference is not None:
            indexed = {(r['case_id'], r['seed']): r for r in reference['by_history']}
            audit.equal(set(indexed), {(r['case_id'], r['seed']) for r in rows}, rule + ': reported history denominator')
        for name in METHODS:
            for mode in MODES:
                observed = [r['methods'][name][mode] for r in per_history]
                counts = Counter(v['category'] for v in observed)
                categories = ('unique_correct', 'unique_wrong', 'unresolved', 'failed_keep_all', 'candidate_miss',
                              'empty_all_models_fail_adequacy', 'missing_group_keep_all')
                aggregate = dict(denominator=len(rows), **{k: counts[k] for k in categories},
                    false_reject=sum(v['false_reject'] is True for v in observed))
                aggregates[rule][name][mode] = aggregate
                audit.equal(sum(counts.values()), len(rows), rule + ': all categories partition denominator')
                if reference is None: continue
                for r in per_history:
                    actual = indexed[(r['case_id'], r['seed'])]['methods'][name][mode]
                    if rule == 'H': actual = actual['stratified']
                    expected = r['methods'][name][mode]
                    for key in ('category', 'retained', 'false_reject'):
                        audit.equal(actual[key], expected[key], rule + '/' + str(r['seed']) + '/' + name + '/' + mode + '/' + key)
                reported = reference['overall'][name][mode]
                for key, value in aggregate.items():
                    if key in reported: audit.equal(reported[key], value, rule + '/overall/' + name + '/' + mode + '/' + key)
                audit.equal(reported['false_reject_rate'], aggregate['false_reject'] / len(rows), rule + ': original-denominator false-reject rate')
                if 'unresolved_including_failed' in reported:
                    audit.equal(reported['unresolved_including_failed'], counts['unresolved'] + counts['failed_keep_all'], rule + ': unresolved including failed')
                if 'all_models_fail_adequacy' in reported:
                    audit.equal(reported['all_models_fail_adequacy'], sum(v.get('all_models_fail_adequacy', False) for v in observed), 'dual empty/all-bad identity')
    return dict(histories=len(rows), unique_histories=len({r['seed'] for r in rows}),
                overall=dict(aggregates), by_history=decisions, input_files=inputs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--calibration', type=Path, required=True)
    parser.add_argument('--primary-thresholds', type=Path, required=True)
    parser.add_argument('--dual-thresholds', type=Path, required=True)
    parser.add_argument('--stratified-thresholds', type=Path, required=True)
    parser.add_argument('--evaluation', type=Path)
    parser.add_argument('--primary-results', type=Path)
    parser.add_argument('--dual-results', type=Path)
    parser.add_argument('--stratified-results', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); began = perf_counter(); audit = Audit()
    primary, dual, stratified = (read(p) for p in (args.primary_thresholds, args.dual_thresholds, args.stratified_thresholds))
    calibration, files = rows_from(args.calibration)
    result = dict(schema='independent_systematic_statistics_audit_v1', independent_stdlib_only=True,
        calibration=audit_calibration(calibration, files, primary, dual, stratified, audit), calibration_files=files)
    if args.evaluation is not None:
        # No evaluation path is probed or opened before this explicit branch.
        evaluation, evaluation_files = rows_from(args.evaluation)
        reference = {rule: read(path) for rule, path in (('primary', args.primary_results), ('dual', args.dual_results),
                     ('H', args.stratified_results)) if path is not None}
        result['evaluation'] = audit_evaluation(evaluation, evaluation_files, calibration,
            primary, dual, stratified, reference, audit)
    result.update(success=not audit.errors, checks=audit.checks, errors=audit.errors, seconds=perf_counter() - began,
        independent_tool_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
        threshold_file_sha256={str(path.resolve()): sha256(path.read_bytes()).hexdigest() for path in
            (args.primary_thresholds, args.dual_thresholds, args.stratified_thresholds)},
        scope='Independent arithmetic from saved candidate losses; preserves failure/missing-candidate denominators. '
              'No rerun, quantile or decision imports; no change to original thresholds.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode('utf8')
    args.output.write_bytes(gzip.compress(payload, mtime=0) if args.output.suffix == '.gz' else payload)
    print(json.dumps(dict(success=result['success'], checks=result['checks'], errors=len(audit.errors),
        calibration_histories=len(calibration), threshold_blocks=result['calibration']['threshold_blocks'],
        evaluation_histories=result.get('evaluation', {}).get('histories'), seconds=result['seconds'])))


if __name__ == '__main__':
    main()
