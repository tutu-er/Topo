"""Independent read-only replay of source-profile gradient policy evidence.

Saved AC values are reconstructed by the separate backward/forward sweep.
No new fit, selector, optimizer, or production AC predictor is called.  I2
records receive payload/source-box checks, not a claim of exact AC agreement.
The existing dynamic reviewer independently checks training and heldout fits.
"""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

from . import ev_dynamic_review as dynamic
from .ev_original_review import forward_physics


PROFILE_KEYS = frozenset(('selected_profile', 'selected_prediction', 'raw_profile',
    'profile', 'prediction', 'gradient_profile', 'candidate_profile'))
PAIR_PAYLOAD = ('a', 'b', 'success', 'source', 'half_SSE', 'residual',
                'gradient_per_kw', 'reference_values', 'competitor_values')


def _require(value, message):
    if not value:
        raise ValueError(message)


def _close(a, b, message, *, atol=2e-7, rtol=5e-8):
    return dynamic._close(a, b, message, atol=atol, rtol=rtol)


def _digest(value):
    return dynamic._identity(value)


def _line_id(r, x):
    digest = hashlib.sha256()
    for value in (r, x):
        arr = np.ascontiguousarray(value, dtype='<f8')
        digest.update(repr(arr.shape).encode())
        digest.update(arr.tobytes())
    return digest.hexdigest()


def _find_profiles(value, path=''):
    """Recursively find only explicit new-format profile records."""
    if isinstance(value, dict):
        if value.get('method') in ('ac', 'i2') and isinstance(value.get('pairs'), list):
            yield path, value
            return
        for key, child in value.items():
            yield from _find_profiles(child, path+'/'+str(key))
    elif isinstance(value, list):
        for i, child in enumerate(value):
            yield from _find_profiles(child, path+'/'+str(i))


def _profile_key(profile):
    # Repeated serialized copies with identical sources/powers are audited once.
    witness = [(p.get('a'), p.get('b'), p.get('source'), p.get('success'))
               for p in profile['pairs']]
    return _digest(dict(method=profile['method'], power=profile.get('power_kw'), pairs=witness))


def _payload(profile):
    return [dict((k, row.get(k)) for k in PAIR_PAYLOAD) for row in profile['pairs']]


def _aggregate(profile, aggregation):
    _require(profile.get('success') is True, 'aggregation requires complete successful profile')
    rows = profile['pairs']
    if aggregation in ('mean', 'binarymean'):
        return float(np.mean([r['half_SSE'] for r in rows]))
    _require(aggregation == 'nearest', 'unknown saved aggregation')
    groups = sorted({r['a'] for r in rows})
    return float(np.mean([min(r['half_SSE'] for r in rows if r['a'] == a) for a in groups]))


def _selected_profile(row):
    return row.get('selected_profile', row.get('selected_prediction'))


def review_profile_trial(trial):
    """Check a parsed trial's new-format planning profiles without refitting."""
    checks, audits, errors = [], [], []
    counts = Counter(unique_profiles=0, profile_occurrences=0, duplicate_profiles=0,
        AC_pairs_verified=0, I2_payload_pairs_verified=0, failed_pairs_retained=0,
        price_pairs_verified=0, price_coordinates_verified=0,
        independent_forward_batches=0, accepted_steps_verified=0)

    def check(name, function):
        try:
            out = function()
            checks.append(dict(name=name, status='passed', result=out))
            return out
        except (ValueError, KeyError, IndexError, TypeError, AssertionError,
                FloatingPointError, np.linalg.LinAlgError) as error:
            message = type(error).__name__+': '+str(error)
            errors.append(dict(name=name, message=message))
            checks.append(dict(name=name, status='failed', message=message))
            return None

    config = trial['config']
    forecast = trial['forecast']
    p, q = (np.asarray(forecast[k], float) for k in ('p', 'q'))
    T, n = p.shape
    bkw = float(trial.get('selection', {}).get('configuration', {}).get('base_kw', 20.))
    _require(p.shape == q.shape and np.isfinite([p, q]).all(), 'forecast P/Q shape')
    root = np.asarray(forecast.get('root', forecast.get('root_v')), float)
    bank = {r['name']:dynamic._candidate(r['name'], r['supports'], n) for r in trial['candidates']}
    trained = {r.get('name', r.get('candidate')):r for r in trial['training_result']['fits']}
    meter = dynamic._meter(trial['future_meter'])
    sigma = np.column_stack((np.broadcast_to(meter.v_sigma, (T, n)),
        np.broadcast_to(meter.master_p_sigma, (T,)), np.broadcast_to(meter.master_q_sigma, (T,))))
    shortlist = trial.get('selection', {}).get('training_top2_names', [])
    expected_shortlist = sorted(bank, key=lambda name:(float(trained[name]['loss']), name))[:2]

    def planning_schema():
        _require(len(bank) == 4 and len(trained) == 4, 'this experiment must retain the four-model bank')
        _require(shortlist == expected_shortlist, 'shortlist is not stable training-loss top two')
        saved = trial.get('training_top2', {})
        _require(saved.get('names') == shortlist, 'shortlist metadata differs')
        _require(saved.get('miss_true') == (trial.get('true_name') not in shortlist), 'offline shortlist-miss flag')
        for key, names in (('all', list(bank)), ('pair', shortlist)):
            lines = trial.get('selection', {}).get('contexts', {}).get(key, {}).get('training_line_sha256')
            if lines is not None:
                _require(set(lines) == set(names), 'context line-hash coverage '+key)
                for name in names:
                    _require(lines[name] == _line_id(trained[name]['r'], trained[name]['x']),
                             'context line snapshot differs from training '+name)
        return dict(final_candidate_count=4, shortlist=shortlist,
            shortlist_uses_training_loss_only=True, shortlist_miss_offline_only=True)

    check('planning_schema', planning_schema)
    occurrences = {}
    for method, record in trial.get('selection', {}).get('methods', {}).items():
        for path, profile in _find_profiles(record, '/'+method):
            counts['profile_occurrences'] += 1
            key = _profile_key(profile)
            if key not in occurrences:
                occurrences[key] = dict(profile=profile, locations=[], methods=set(), check_price=False)
            item = occurrences[key]
            # A duplicate witness cannot silently replace its saved residual/price.
            check('duplicate_payload'+path,
                lambda item=item, profile=profile: _require(
                    _digest(_payload(item['profile'])) == _digest(_payload(profile)),
                    'identical control/source witness has inconsistent payload'))
            item['locations'].append(path)
            item['methods'].add(method)
            if ('selected_profile' in path or 'selected_prediction' in path
                    or ('/iterations/' in path and '/gradient_profile' in path)):
                item['check_price'] = True
    counts['unique_profiles'] = len(occurrences)
    counts['duplicate_profiles'] = counts['profile_occurrences']-len(occurrences)

    def source_box(source, name):
        s = np.asarray(source, float)
        _require(s.shape == (T, 2) and np.isfinite(s).all(), 'source witness shape '+name)
        for column, channel in enumerate(('source_p', 'source_q')):
            lo, hi = config['source'][channel+'_bounds']
            _require(np.all(s[:, column] >= lo-1e-10) and np.all(s[:, column] <= hi+1e-10),
                     'source witness outside physical box '+name)
        return s

    def forward(name, control, source):
        values = forward_physics(bank[name], trained[name]['r'], trained[name]['x'],
            p+control/bkw, q, source[:, 0], source[:, 1], root)
        counts['independent_forward_batches'] += 1
        return np.column_stack((values['v'], values['master_p'], values['master_q']))

    for key, item in occurrences.items():
        profile = item['profile']
        label = item['locations'][0]
        method = profile['method']
        control = np.asarray(profile.get('power_kw'), float)
        expected_names = shortlist if all(m.startswith('pairac_') for m in item['methods']) else list(bank)
        expected_pairs = {(a, b) for a in expected_names for b in expected_names if a != b}

        def coverage():
            _require(control.shape == (T, n) and np.isfinite(control).all(), 'profile control shape')
            rows = profile['pairs']
            present = [(r.get('a'), r.get('b')) for r in rows]
            _require(len(present) == len(set(present)) and set(present) == expected_pairs,
                     'planning directed-pair denominator')
            _require(profile.get('expected_pair_count') == len(expected_pairs), 'saved pair count')
            if profile.get('success'):
                _require(all(r.get('success') for r in rows), 'successful profile contains failed pair')
                _require(profile.get('line_parameters_unchanged') is True, 'profile lines changed')
            return dict(method=method, pairs=len(rows), declared_success=profile.get('success'),
                        locations=item['locations'])

        check('coverage'+label, coverage)
        for row in profile['pairs']:
            pairlabel = label+'/'+str(row.get('name', '?'))
            if not row.get('success'):
                counts['failed_pairs_retained'] += 1
                audits.append(dict(label=pairlabel, success=False, failure=row.get('failure'),
                    attempted=row.get('attempted'), failed_record_preserved=True))
                continue

            def one(row=row):
                a, b = row['a'], row['b']
                _require(a in bank and b in bank and a != b, 'unknown planning pair')
                _require(row['name'] == a+'__to__'+b, 'directed pair name')
                _require(row.get('source_interface') == 1 and row.get('line_parameters_fixed') is True,
                         'source or fixed-line flags')
                source = source_box(row['source'], row['name'])
                rs = forecast['source_by_candidate'][a]
                reference_source = source_box(np.column_stack((rs['source_p'], rs['source_q'])), a)
                ref = np.asarray(row['reference_values'], float)
                comp = np.asarray(row['competitor_values'], float)
                saved_r = np.asarray(row['residual'], float)
                gradient = np.asarray(row['gradient_per_kw'], float)
                _require(ref.shape == comp.shape == saved_r.shape == (T, n+2), 'response payload shapes')
                _require(gradient.shape == (T, n) and np.isfinite(gradient).all(), 'price payload shape')
                _close(saved_r, (comp-ref)/sigma, 'saved whitening', atol=3e-7)
                _close(row['half_SSE'], 0.5*float(np.sum(saved_r**2)), 'saved half SSE', atol=1e-7)
                result = dict(a=a, b=b, method=method, half_SSE=float(row['half_SSE']),
                    source_box_checked=True, training_snapshot_metadata_checked=True,
                    scaled_source_kkt_violation=row.get('scaled_source_kkt_violation'))
                if method == 'i2':
                    counts['I2_payload_pairs_verified'] += 1
                    result.update(AC_agreement_checked=False,
                        scope='payload/source-box only; I2 equations and gradients have separate unit audits')
                    return result
                rref = forward(a, control, reference_source)
                rcomp = forward(b, control, source)
                error = max(_close(ref, rref, 'reference AC response', atol=3e-10, rtol=0.),
                            _close(comp, rcomp, 'competitor AC response', atol=3e-10, rtol=0.))
                residual = (rcomp-rref)/sigma
                _close(saved_r, residual, 'independent AC whitening', atol=3e-6)
                _close(row['half_SSE'], 0.5*float(np.sum(residual**2)), 'independent AC half SSE', atol=2e-6)
                counts['AC_pairs_verified'] += 1
                result.update(max_response_error=error, AC_agreement_checked=True,
                              training_R_X_reconstructed=True)
                if item['check_price']:
                    coordinates = list(dict.fromkeys(((0, 0), (T-1, n-1))))
                    step = 1e-3
                    fd, saved = [], []
                    for t, j in coordinates:
                        plus, minus = control.copy(), control.copy()
                        plus[t, j] += step
                        minus[t, j] -= step
                        rp = (forward(b, plus, source)-forward(a, plus, reference_source))/sigma
                        rm = (forward(b, minus, source)-forward(a, minus, reference_source))/sigma
                        fd.append(float(np.sum((rp-rm)/(2*step)*residual)))
                        saved.append(float(gradient[t, j]))
                    graderr = _close(saved, fd, 'independent fixed-witness AC price', atol=3e-5, rtol=2e-5)
                    counts['price_pairs_verified'] += 1
                    counts['price_coordinates_verified'] += len(coordinates)
                    result.update(price_coordinates=coordinates, price_step_kw=step,
                        fixed_witness_finite_difference_per_kw=fd, max_abs_price_error=graderr,
                        price_scope='partial derivative at saved source witness; no source refit or global certificate')
                return result

            result = check('pair'+pairlabel, one)
            if result is not None:
                audits.append(dict(label=pairlabel, **result))

    for method, record in trial.get('selection', {}).get('methods', {}).items():
        aggregation = record.get('aggregation', 'binarymean' if method.startswith('pairac_') else 'nearest')
        selected = _selected_profile(record)
        if selected is not None and selected.get('success'):
            check('selected_score/'+method, lambda selected=selected, record=record:
                _close(record['selected_score'], _aggregate(selected, aggregation), 'selected profile score'))
        previous = {}
        for i, row in enumerate(record.get('iterations', [])):
            def step(row=row):
                start = row.get('start_index', 0)
                before = row['before_score']
                if start in previous:
                    _close(before, previous[start], 'iteration chain before/after')
                after = row.get('after_score', before)
                if row.get('accepted'):
                    candidate = row.get('candidate_profile')
                    _require(candidate is not None and candidate.get('success') is True
                             and candidate.get('method') == 'ac', 'accepted step lacks complete AC profile')
                    score = _aggregate(candidate, aggregation)
                    _close(row['candidate_score'], score, 'accepted candidate score')
                    _close(after, score, 'accepted after score')
                    _require(after > before+row.get('acceptance_tolerance', 0.),
                             'accepted step did not strictly improve full AC score')
                    counts['accepted_steps_verified'] += 1
                else:
                    _close(after, before, 'rejected step changed full AC score')
                previous[start] = after
                return dict(accepted=row.get('accepted', False), before_score=before, after_score=after)
            check('iteration/'+method+'/'+str(i), step)

    return dynamic._serial(dict(schema='ev_gradient_flow_profile_review_v1', seed=trial.get('seed'),
        success=bool(occurrences) and not errors, counts=dict(counts), checks=checks,
        pair_audits=audits, errors=errors, no_new_fit=True, no_selector=True,
        no_production_AC_predictor=True, source_global_optimality_certified=False,
        I2_AC_agreement_claimed=False, independent_unit='one saved history'))


def review_directory(directory, output=None):
    directory = Path(directory)
    base = dynamic.review_directory(directory)
    reports = []
    for path in sorted(directory.glob('trial_*.json.gz')):
        try:
            report = review_profile_trial(dynamic._read(path))
        except (ValueError, KeyError, TypeError, IndexError, OSError, np.linalg.LinAlgError) as error:
            report = dict(success=False, review_exception=type(error).__name__+': '+str(error))
        report.update(input_path=str(path.resolve()), input_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        reports.append(report)
    counts = Counter()
    for row in reports:
        counts.update(row.get('counts', {}))
    result = dict(schema='ev_gradient_flow_readonly_review_v1', directory=str(directory.resolve()),
        success=bool(reports) and base['success'] and all(r['success'] for r in reports),
        trial_count=len(reports), verified_profile_trials=sum(r['success'] for r in reports),
        profile_counts=dict(counts), base_review=base, profile_reviews=reports,
        no_new_fit=True, no_selector=True, no_production_AC_predictor=True,
        calibrated_rejection_statistics_computed=False,
        reviewer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    if output is not None:
        output = Path(output)
        protected = {p.resolve() for p in directory.glob('trial_*.json.gz')}
        protected |= {(directory/'freeze.json').resolve(), (directory/'source_snapshot.json.gz').resolve()}
        _require(output.resolve() not in protected, 'review must not overwrite evidence')
        output.parent.mkdir(parents=True, exist_ok=True)
        data = (json.dumps(dynamic._serial(result), ensure_ascii=False, sort_keys=True,
                           allow_nan=False)+'\n').encode('utf-8')
        output.write_bytes(gzip.compress(data, mtime=0) if output.suffix == '.gz' else data)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = review_directory(args.directory, args.output)
    print(json.dumps({k:result[k] for k in ('success', 'trial_count', 'verified_profile_trials', 'profile_counts')}))
    raise SystemExit(0 if result['success'] else 1)


if __name__ == '__main__':
    main()
