"""Frozen-policy calibration from saved independent-history records only.

No optimizer, simulator, or AC module is imported. Losses are attained full
H+N whitened SSE, not probabilities. Invalid computations retain all declared
candidates. Calibration and evaluation history units must be disjoint.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass
from fractions import Fraction
import hashlib
import math
from typing import Mapping, Sequence

SCHEMA = 'ev_frozen_policy_calibration_v1'


def canonical_bytes(value):
    import json
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=False, allow_nan=False).encode('utf-8')


def object_hash(value):
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def freeze_identity(freeze):
    """Verify the complete selector document, including explicit top-level modes.

    Full policy_config may contain config/prediction/master_form/domain and
    source_sha256. Legacy records use config+source_sha256. Both explicitly
    supported JSON encodings are verified, rather than trusting a supplied ID.
    """
    import json
    config = freeze.get('policy_config', freeze.get('config'))
    sources = freeze.get('source_sha256')
    if not isinstance(config, Mapping) or not isinstance(sources, Mapping) or not sources:
        raise ValueError('freeze requires config/policy_config and nonempty source_sha256')
    if any(not isinstance(k, str) or not isinstance(v, str) or len(v) != 64
           or any(c not in '0123456789abcdef' for c in v) for k, v in sources.items()):
        raise ValueError('source hashes must be lowercase SHA-256 strings')
    if 'source_sha256' in config:
        payload = dict(config)
        if payload['source_sha256'] != sources:
            raise ValueError('inner and outer source manifests differ')
        for key in ('config', 'prediction', 'master_form', 'domain'):
            if key in freeze and key in payload and freeze[key] != payload[key]:
                raise ValueError('inner and outer selector configuration differ: '+key)
        config = {k: v for k, v in payload.items() if k != 'source_sha256'}
    elif 'policy_config' not in freeze and all(k in freeze for k in ('prediction', 'master_form', 'domain')):
        config = {k: freeze[k] for k in ('config', 'prediction', 'master_form', 'domain')}
        payload = dict(config, source_sha256=sources)
    else:
        payload = dict(config=config, source_sha256=sources)
    canonical_id = object_hash(payload)
    driver_id = hashlib.sha256(json.dumps(payload, sort_keys=True, allow_nan=False).encode('utf-8')).hexdigest()
    declared_id = freeze.get('policy_id', canonical_id)
    if declared_id not in (canonical_id, driver_id):
        raise ValueError('declared freeze policy_id does not match config and source hashes')
    out = dict(config_sha256=object_hash(config), source_manifest_sha256=object_hash(sources),
               policy_id=declared_id)
    return out


def _names(values, label):
    if isinstance(values, (str, bytes)):
        raise ValueError(label+' must be a sequence of names')
    values = tuple(values)
    if not values or any(not isinstance(v, str) or not v for v in values) or len(set(values)) != len(values):
        raise ValueError(label+' must contain distinct nonempty names')
    return values


@dataclass(frozen=True)
class CalibrationSpec:
    policy_id: str
    config_sha256: str
    source_manifest_sha256: str
    reference_id: str
    candidates: tuple[str, ...]
    methods: tuple[str, ...]
    strata: tuple[str, ...] = ('declared_asset_reference',)
    alpha: float = .1
    familywise: bool = False
    stratum_mode: str = 'worst'
    stratum_is_public: bool = False

    def __post_init__(self):
        for field in ('policy_id', 'config_sha256', 'source_manifest_sha256', 'reference_id'):
            if not isinstance(getattr(self, field), str) or not getattr(self, field):
                raise ValueError(field+' is required')
        for field in ('candidates', 'methods', 'strata'):
            object.__setattr__(self, field, _names(getattr(self, field), field))
        if not 0 < Fraction(str(self.alpha)) < 1:
            raise ValueError('alpha must lie strictly between zero and one')
        object.__setattr__(self, 'alpha', float(self.alpha))
        if not isinstance(self.familywise, bool) or not isinstance(self.stratum_is_public, bool):
            raise ValueError('familywise and stratum_is_public must be booleans')
        if self.stratum_mode not in ('worst', 'public'):
            raise ValueError('stratum_mode must be worst or public')
        if self.stratum_mode == 'public' and not self.stratum_is_public:
            raise ValueError('public stratum mode requires a declared observable stratum')

    def summary(self):
        return asdict(self)


def spec_from_freeze(freeze, *, reference_id, candidates, methods, **options):
    return CalibrationSpec(**freeze_identity(freeze), reference_id=reference_id,
                           candidates=tuple(candidates), methods=tuple(methods), **options)


def upper_order_quantile(scores, alpha=.1):
    """Exact rank ceil((n+1)(1-alpha)); ties retained, no interpolation."""
    scores = sorted(float(s) for s in scores)
    if any(math.isnan(s) or s < 0 for s in scores):
        raise ValueError('scores must be nonnegative and not NaN')
    a = Fraction(str(alpha))
    if not 0 < a < 1:
        raise ValueError('invalid alpha')
    n = len(scores); target = (n+1)*(1-a)
    rank = (target.numerator+target.denominator-1)//target.denominator
    threshold = scores[rank-1] if rank <= n else math.inf
    return dict(n=n, rank=rank, threshold=None if math.isinf(threshold) else threshold,
        infinite=math.isinf(threshold), alpha=float(a),
        rank_marginal_error_bound=float(Fraction(n+1-rank, n+1)) if rank <= n else 0.,
        strict_rejection=True, reason='rank_exceeds_n' if rank > n else
        'infinite_observed_score' if math.isinf(threshold) else 'finite_order_statistic')


def _threshold(value):
    return math.inf if value['infinite'] else float(value['threshold'])


def _method_valid(method, candidates):
    losses = method.get('losses', {})
    return (method.get('status') == 'complete' and isinstance(losses, Mapping) and set(losses) == set(candidates)
            and all(isinstance(v, (float, int)) and not isinstance(v, bool)
                    and math.isfinite(v) and v >= 0 for v in losses.values()))


def _records(records, spec, expected_history_ids=None):
    records = list(records); ids = set(); units = set()
    for record in records:
        for key in ('policy_id', 'config_sha256', 'source_manifest_sha256', 'reference_id'):
            if record.get(key) != getattr(spec, key):
                raise ValueError('record differs from frozen '+key)
        raw_h = record.get('history_id'); raw_u = record.get('independent_unit_id', raw_h)
        if raw_h is None or raw_u is None or isinstance(raw_h, bool) or isinstance(raw_u, bool):
            raise ValueError('missing independent history unit')
        h = str(raw_h); u = str(raw_u)
        if not h or not u or h in ids or u in units:
            raise ValueError('duplicate or missing independent history unit')
        ids.add(h); units.add(u)
        if not isinstance(record.get('stratum'), str) or not record['stratum']:
            raise ValueError('missing declared stratum')
        if not isinstance(record.get('true_name'), str) or not record['true_name']:
            raise ValueError('offline true_name is required')
        _names(record['candidate_names'], 'candidate_names')
        if set(record.get('methods', {})) != set(spec.methods):
            raise ValueError('missing or extra frozen method')
        for method in record['methods'].values():
            if not isinstance(method.get('status'), str) or not method['status']:
                raise ValueError('explicit method status required')
            if method['status'] == 'complete' and not _method_valid(method, spec.candidates):
                raise ValueError('complete method has malformed/partial/nonfinite SSE')
    if expected_history_ids is not None:
        expected = [str(h) for h in expected_history_ids]
        if len(set(expected)) != len(expected) or ids != set(expected):
            raise ValueError('saved histories differ from expected completed histories')
    return records


def _coverage_failure(record, spec):
    return (record['true_name'] not in spec.candidates or bool(record.get('candidate_miss'))
            or record['true_name'] not in record['candidate_names'])


def _scores(record, spec):
    if _coverage_failure(record, spec) or record['stratum'] not in spec.strata:
        return {m: math.inf for m in spec.methods}
    out = {}
    for name, method in record['methods'].items():
        if set(record['candidate_names']) != set(spec.candidates) or not _method_valid(method, spec.candidates):
            out[name] = 0.  # Exactly the keep-all decision score.
        else:
            losses = method['losses']
            out[name] = float(losses[record['true_name']]-min(losses.values()))
    return out


def calibrate(records, spec, *, expected_history_ids=None, planned_history_ids=None):
    """One score per independent H, per candidate-null and declared stratum."""
    records = _records(records, spec, expected_history_ids)
    ids = sorted(str(r['history_id']) for r in records)
    planned = sorted(str(h) for h in (planned_history_ids if planned_history_ids is not None else ids))
    if len(set(planned)) != len(planned) or not set(ids) <= set(planned):
        raise ValueError('invalid planned history list')
    groups = ('__family__',) if spec.familywise else spec.methods
    cells = {s: {a: {m: [] for m in groups} for a in spec.candidates} for s in spec.strata}
    details = []; coverage_failures = 0; out_of_reference = 0
    for record in records:
        coverage = _coverage_failure(record, spec)
        unknown = record['stratum'] not in spec.strata
        coverage_failures += coverage; out_of_reference += unknown
        scores = _scores(record, spec)
        details.append(dict(history_id=str(record['history_id']), stratum=record['stratum'],
            true_name=record['true_name'], candidate_miss=coverage,
            out_of_reference=unknown,
            scores={m: None if math.isinf(v) else v for m, v in scores.items()},
            failed_methods=[m for m in spec.methods if not _method_valid(record['methods'][m], spec.candidates)]))
        if not unknown and record['true_name'] in spec.candidates:
            target = cells[record['stratum']][record['true_name']]
            for m in groups:
                target[m].append(max(scores.values()) if spec.familywise else scores[m])
    blocked = bool(coverage_failures or out_of_reference)
    thresholds = {s: {a: {m: upper_order_quantile(values, spec.alpha) for m, values in methods.items()}
                          for a, methods in candidates.items()} for s, candidates in cells.items()}
    if blocked:
        for candidates in thresholds.values():
            for methods in candidates.values():
                for q in methods.values():
                    q.update(threshold=None, infinite=True, reason='reference_or_coverage_failure_keep_all')
    out = dict(schema=SCHEMA, spec=spec.summary(), thresholds=thresholds,
        history_ids=ids, independent_unit_ids=sorted(str(r.get('independent_unit_id', r['history_id'])) for r in records),
        histories_total=len(records), histories_planned=len(planned), histories_pending=len(set(planned)-set(ids)),
        coverage_failures=coverage_failures, out_of_reference_histories=out_of_reference,
        rejection_enabled=not blocked, records=details,
        scope='conditional exchangeability in the declared simulated reference distribution only',
        guarantee_kind='marginal over calibration and one new independent history',
        exchangeability_assumed_not_verified=True, conditional_guarantee_available=not blocked,
        fixed_threshold_risk_confidence_claim=False, topology_probability=False)
    out['calibration_id'] = object_hash(out)
    return out


def evaluate(records, calibration, *, expected_history_ids=None):
    """Apply frozen thresholds. Failures and infinite thresholds remain unresolved."""
    if calibration.get('schema') != SCHEMA:
        raise ValueError('unknown calibration schema')
    if object_hash({k: v for k, v in calibration.items() if k != 'calibration_id'}) != calibration['calibration_id']:
        raise ValueError('calibration content/hash mismatch')
    spec = CalibrationSpec(**calibration['spec'])
    records = _records(records, spec, expected_history_ids)
    if set(calibration['independent_unit_ids']) & {str(r.get('independent_unit_id', r['history_id'])) for r in records}:
        raise ValueError('calibration and evaluation share an independent history')
    if set(calibration['history_ids']) & {str(r['history_id']) for r in records}:
        raise ValueError('calibration and evaluation share history identifiers')
    totals = dict(histories_total=len(records), candidate_miss_histories=0,
        out_of_reference_histories=0, any_method_false_rejections=0,
        known_covered_histories=0, globally_unresolved_histories=0)
    method_totals = {m: dict(total=len(records), computational_failures=0,
        operation_failures=0, fit_failures=0, complete_decisions=0,
        false_rejections=0, unique_correct=0, unresolved=0,
        wrong_candidates_excluded=0, known_covered_denominator=0) for m in spec.methods}
    details = []
    for record in records:
        miss = _coverage_failure(record, spec); outside = record['stratum'] not in spec.strata
        global_unresolved = miss or outside or not calibration['rejection_enabled']
        totals['candidate_miss_histories'] += miss; totals['out_of_reference_histories'] += outside
        totals['known_covered_histories'] += not miss
        totals['globally_unresolved_histories'] += global_unresolved
        decisions = {}; any_false = False
        for m in spec.methods:
            method = record['methods'][m]; stats = method_totals[m]
            valid = _method_valid(method, spec.candidates) and set(record['candidate_names']) == set(spec.candidates)
            thresholds = {}; gaps = {}
            for a in spec.candidates:
                group = '__family__' if spec.familywise else m
                strata = spec.strata if spec.stratum_mode == 'worst' else (record['stratum'],)
                thresholds[a] = max((_threshold(calibration['thresholds'][s][a][group])
                                     for s in strata if s in calibration['thresholds']), default=math.inf)
            if global_unresolved or not valid:
                retained = list(spec.candidates)
            else:
                minimum = min(method['losses'].values())
                gaps = {a: float(method['losses'][a]-minimum) for a in spec.candidates}
                retained = [a for a in spec.candidates if gaps[a] <= thresholds[a]]
            false = not miss and record['true_name'] not in retained
            unique = not miss and retained == [record['true_name']]
            unresolved = global_unresolved or not valid or len(retained) != 1
            stats['computational_failures'] += not valid
            status = method['status'].lower()
            stats['operation_failures'] += not valid and ('operation' in status or 'design' in status)
            stats['fit_failures'] += not valid and ('fit' in status or 'historical' in status)
            stats['complete_decisions'] += valid and not global_unresolved
            stats['known_covered_denominator'] += not miss
            stats['false_rejections'] += false; stats['unique_correct'] += unique
            stats['unresolved'] += unresolved
            stats['wrong_candidates_excluded'] += sum(a != record['true_name'] and a not in retained for a in spec.candidates) if not miss else 0
            any_false |= false
            decisions[m] = dict(retained=retained, rejected=[a for a in spec.candidates if a not in retained],
                gaps=gaps, thresholds={a: None if math.isinf(t) else t for a, t in thresholds.items()},
                infinite_thresholds=[a for a, t in thresholds.items() if math.isinf(t)],
                status=method['status'], compute_failure=not valid, global_unresolved=global_unresolved,
                false_rejection=false if not miss else None, unique_correct=unique, unresolved=unresolved)
        totals['any_method_false_rejections'] += any_false
        details.append(dict(history_id=str(record['history_id']), true_name=record['true_name'],
            stratum=record['stratum'], candidate_miss=miss, out_of_reference=outside, methods=decisions))
    return dict(schema=SCHEMA, calibration_id=calibration['calibration_id'], spec=spec.summary(),
        totals=totals, methods=method_totals, records=details,
        empirical_counts_only=True, topology_probability=False)


def _fit_rows_valid(rows, names):
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        return False
    by_name = {r.get('name'): r for r in rows}
    return (len(by_name) == len(rows) and set(by_name) == set(names)
            and all(r.get('success') is True and isinstance(r.get('loss'), (float, int))
                    and math.isfinite(r['loss']) and r['loss'] >= 0 for r in rows))


def adapt_saved_trial(trial, freeze, spec, *, stratum='declared_asset_reference'):
    """Adapt original/compact saved trials offline; truth is used only for scoring.

    Historical and joint fit failures keep all candidates. A valid operational
    fallback with complete fits is scored normally. Config/source mismatch is a
    provenance error and raises, rather than becoming a cheap zero score.
    """
    if freeze.get('frozen_before_trials') is not True:
        raise ValueError('a before-trials frozen policy record is required')
    identity = freeze_identity(freeze)
    if any(getattr(spec, k) != value for k, value in identity.items()):
        raise ValueError('spec does not match freeze identity')
    trial_identity = freeze_identity(trial)
    if identity != trial_identity or trial.get('source_changed_during_trial'):
        raise ValueError('trial config/source differs from frozen policy')
    bank = trial.get('proposal', {}).get('bank', [])
    names = list(trial.get('candidate_names', [r['name'] for r in bank]))
    if not names:
        names = list(spec.candidates)  # Prespecified fallback set, without fitted claims.
    true = trial.get('true_name')
    if true is None:
        family = trial.get('truth', {}).get('family')
        n = len(trial.get('history', {}).get('p', [[0, 0, 0]])[0])
        def family_key(groups):
            return tuple(sorted(tuple(sorted(s)) for s in groups if 1 < len(s) < n))
        if family is None:
            raise ValueError('saved trial has no offline truth label or family')
        matches = [r['name'] for r in bank if family_key(r['supports']) == family_key(family)]
        true = matches[0] if len(matches) == 1 else '__uncovered_truth__'
    historical_ok = (not trial.get('historical_failures')
                     and _fit_rows_valid(trial.get('historical_fits'), spec.candidates))
    rows = trial.get('rows', []); indexed = {r.get('method'): r for r in rows}
    if len(indexed) != len(rows):
        raise ValueError('duplicate saved method row')
    if any(r.get('true_name') is not None and r['true_name'] != true for r in rows):
        raise ValueError('saved method truth labels disagree with offline truth')
    methods = {}
    for m in spec.methods:
        row = indexed.get(m, {}); status = row.get('status', 'missing_method_failure')
        if not historical_ok:
            status = 'historical_fit_failure'
        elif status == 'complete' and not _fit_rows_valid(row.get('fits'), spec.candidates):
            status = 'required_joint_fit_failure'
        elif status == 'complete' and not _method_valid(row, spec.candidates):
            status = 'invalid_joint_loss_failure'
        elif status == 'complete' and any(row['losses'][f['name']] != f['loss'] for f in row['fits']):
            status = 'saved_fit_loss_mismatch_failure'
        methods[m] = dict(status=status, raw_status=row.get('status'), losses=row.get('losses', {}))
    return dict(**identity, reference_id=spec.reference_id,
        history_id=str(trial.get('history_id', trial.get('seed', ''))),
        independent_unit_id=str(trial.get('independent_unit_id', trial.get('seed', ''))),
        stratum=stratum, true_name=true, candidate_names=names, methods=methods,
        candidate_miss=bool(trial.get('candidate_miss', False) or true not in names
                            or any(indexed.get(m, {}).get('candidate_miss', False) for m in spec.methods)),
        adapter='saved_original_or_compact_trial_no_refitting')
