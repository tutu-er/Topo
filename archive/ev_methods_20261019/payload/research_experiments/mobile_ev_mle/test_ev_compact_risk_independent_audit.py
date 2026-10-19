from copy import deepcopy
import math

from .ev_compact_risk_independent_audit import (
    canonical_hash, independent_quantile, reconstruct_calibration,
    reconstruct_evaluation, projected_records,
)


def spec(methods=('m',), familywise=False, strata=('s',)):
    return dict(policy_id='p', config_sha256='c', source_manifest_sha256='h',
        reference_id='r', candidates=['a', 'b'], methods=list(methods),
        strata=list(strata), alpha=.1, familywise=familywise,
        stratum_mode='worst', stratum_is_public=False)


def record(i, *, true='a', losses=None, status='complete', methods=('m',), stratum='s'):
    losses = {'a': 1., 'b': 0.} if losses is None else losses
    return dict(history_id=str(i), independent_unit_id=str(i), true_name=true,
        candidate_names=['a', 'b'], candidate_miss=False, base_candidate_miss=False,
        method_candidate_miss={m: False for m in methods}, stratum=stratum,
        methods={m: dict(status=status, losses=deepcopy(losses)) for m in methods})


def calibrated(methods=('m',), familywise=False):
    s = spec(methods, familywise)
    rows = [record(i, methods=methods) for i in range(9)]
    rows += [record(i + 9, true='b', losses={'a': 0., 'b': 2.}, methods=methods) for i in range(9)]
    return reconstruct_calibration(rows, s, range(18))


def test_exact_rank_nine():
    q = independent_quantile(list(range(9)), .1)
    assert q['rank'] == q['n'] == 9
    assert q['threshold'] == 8
    assert q['rank_marginal_error_bound'] == .1


def test_insufficient_rank_is_infinite():
    q = independent_quantile(list(range(8)), .1)
    assert q['rank'] == 9 and q['infinite'] and q['threshold'] is None


def test_calibration_hash_and_thresholds():
    c = calibrated()
    assert c['thresholds']['s']['a']['m']['threshold'] == 1.
    assert c['thresholds']['s']['b']['m']['threshold'] == 2.
    assert c['calibration_id'] == canonical_hash({k: v for k, v in c.items() if k != 'calibration_id'})


def test_equal_threshold_is_retained_but_strict_exceedance_rejected():
    c = calibrated()
    e = reconstruct_evaluation([record(100), record(101, losses={'a': 1.01, 'b': 0.})], c)
    assert e['records'][0]['methods']['m']['retained'] == ['a', 'b']
    assert e['records'][1]['methods']['m']['retained'] == ['b']
    assert e['methods']['m']['false_rejections'] == 1
    assert e['methods']['m']['unresolved'] == 1


def test_failed_execution_keeps_all_and_full_denominator():
    c = calibrated()
    e = reconstruct_evaluation([record(100, losses={}, status='operation_or_design_failure')], c)
    assert e['records'][0]['methods']['m']['retained'] == ['a', 'b']
    assert e['methods']['m']['total'] == 1
    assert e['methods']['m']['operation_failures'] == 1
    assert e['methods']['m']['false_rejections'] == 0
    assert e['methods']['m']['unresolved'] == 1


def test_family_calibration_takes_same_history_maximum():
    methods = ('m', 'n')
    rows = [record(i, methods=methods) for i in range(9)]
    for i, row in enumerate(rows):
        row['methods']['n']['losses'] = {'a': float(i), 'b': 0.}
    c = reconstruct_calibration(rows, spec(methods, True), range(9))
    assert c['thresholds']['s']['a']['__family__']['threshold'] == 8.
    e = reconstruct_evaluation([record(100, methods=methods, losses={'a': 9., 'b': 0.})], c)
    assert e['totals']['any_method_false_rejections'] == 1
    assert all(e['methods'][m]['false_rejections'] == 1 for m in methods)


def test_failure_zero_score_and_candidate_miss_distinguished():
    rows = [record(i, losses={}, status='required_joint_fit_failure') for i in range(9)]
    c = reconstruct_calibration(rows, spec(), range(9))
    assert c['thresholds']['s']['a']['m']['threshold'] == 0.
    assert c['rejection_enabled']
    rows[0]['candidate_miss'] = True
    c = reconstruct_calibration(rows, spec(), range(9))
    assert not c['rejection_enabled']
    assert all(q['infinite'] for aa in c['thresholds'].values() for mm in aa.values() for q in mm.values())


def test_primary_projection_preserves_relevant_candidate_miss_only():
    r = record(0, methods=('m', 'n'))
    r['method_candidate_miss']['n'] = True
    r['candidate_miss'] = True
    assert not projected_records([r], ['m'])[0]['candidate_miss']
    assert projected_records([r], ['n'])[0]['candidate_miss']


def test_unknown_stratum_is_global_unresolved():
    e = reconstruct_evaluation([record(100, stratum='outside')], calibrated())
    assert e['totals']['out_of_reference_histories'] == 1
    assert e['methods']['m']['unresolved'] == 1
    assert e['methods']['m']['complete_decisions'] == 0
    assert e['methods']['m']['false_rejections'] == 0


def test_worst_declared_stratum_uses_larger_threshold():
    c = calibrated()
    c['spec']['strata'] = ['s', 's2']
    c['thresholds']['s2'] = deepcopy(c['thresholds']['s'])
    c['thresholds']['s2']['a']['m']['threshold'] = 10.
    e = reconstruct_evaluation([record(100, losses={'a': 5., 'b': 0.})], c)
    assert e['records'][0]['methods']['m']['retained'] == ['a', 'b']
