"""Aggregation and calibration arithmetic; no physical experiments."""
from copy import deepcopy
import gzip
import json

import numpy as np
import pytest

from . import ev_bayes_response_analysis as analysis
from . import ev_bayes_response_cases as cases


def fake_evaluation(correct=(1, 0, 1, 0), confidence=(.8, .9, .7, 1.)):
    y, c = np.asarray(correct), np.asarray(confidence); R = len(y)
    posterior = np.column_stack((c, 1-c))
    samples = dict(correct=y.tolist(), confidence=c.tolist(), true_probability=(1-c).tolist(),
        posterior=posterior.tolist(), true_rejected=(1-y).tolist(), unresolved=[0]*R,
        unique_correct=y.tolist(), wrong_unique=(1-y).tolist(), brier=[.25]*R,
        negative_log_true_probability=[1.]*R, credible_size=[1]*R,
        meter_interval90_coverage=[.8]*R, voltage_interval90_coverage=[.75]*R)
    return dict(samples=samples, summary={k: float(np.mean(v)) for k, v in samples.items() if k != 'posterior'},
        n_noise=R, n_independent_histories=1)


def fake_record(spec, evaluation=None, status='complete'):
    evaluation = fake_evaluation() if evaluation is None else evaluation
    return dict(spec=spec, status=status, true_name='star', config=dict(actual_noise_repeats=4),
        methods={'constant': dict(action_key='same', domain='fixed_charge', power_kw=[[1.]])},
        actions={'same': dict(predictions={m: dict(independent_score=dict(risk=.2, map_error=.2, eig=.3, brier_risk=.4, abstention_risk=.1)) for m in analysis.MODELS},
            actual={r: dict(models={m: deepcopy(evaluation) for m in analysis.MODELS}) for r in analysis.REGIMES})})


def test_history_unit_64_and_128_not_noise_replicates():
    for phase, N in (('calibration', 64), ('evaluation', 128)):
        expected = cases.specifications(phase)
        records = [fake_record(spec) for spec in expected]
        data = analysis.collect_records(records, expected, methods=('constant',), models=('joint',))
        assert data['coverage']['exact']
        values = [r['error'] for r in data['rows'] if r['regime'] == 'regular']
        result = analysis.history_estimate(values, bootstrap=100)
        assert result['available_histories'] == N and result['expected_histories'] == N
        assert result['mean'] == .5
        assert result['available_histories'] != 4*N


def test_missing_and_recorded_failure_stay_in_expected_population():
    expected = cases.specifications('calibration')[:3]
    records = [fake_record(expected[0]), dict(spec=expected[1], status='training_failure', failure='retained')]
    data = analysis.collect_records(records, expected, methods=('constant',), models=('joint',))
    assert not data['coverage']['exact'] and len(data['coverage']['missing']) == 1
    assert [h['status'] for h in data['histories']] == ['complete', 'training_failure', 'missing_history']
    rows = [r for r in data['rows'] if r['regime'] == 'regular']
    result = analysis.history_estimate([r['error'] for r in rows], bounds=(0, 1), bootstrap=100)
    assert result['available_histories'] == 1 and result['expected_histories'] == 3
    np.testing.assert_allclose(result['missing_outcome_bounds'], [1/6, 5/6])


def test_exact_coverage_rejects_duplicate_and_extra_records():
    expected = cases.specifications('calibration')[:2]
    assert analysis.coverage(expected, list(reversed(expected)))['exact']
    assert not analysis.coverage(expected, [expected[0], expected[0]])['exact']
    assert len(analysis.coverage(expected, [expected[0], expected[0]])['duplicates']) == 1
    assert not analysis.coverage(expected, expected+[dict(index=999)])['exact']


def test_column_arrays_proper_scores_and_summary_checks():
    metrics, bins = analysis.actual_metrics(fake_evaluation(), 4)
    assert metrics['error'] == .5 and metrics['brier'] == .25
    assert metrics['negative_log_true_probability'] == 1.
    assert metrics['confidence_minus_correct'] == pytest.approx(.35)
    assert metrics['abstention_loss'] == .5
    assert sum(v['conditional_draws'] for v in bins) == 4
    bad = fake_evaluation(); bad['summary']['brier'] = 99.
    with pytest.raises(ValueError, match='summary'): analysis.actual_metrics(bad, 4)
    bad = fake_evaluation(); bad['n_independent_histories'] = 4
    with pytest.raises(ValueError, match='one history'): analysis.actual_metrics(bad, 4)
    with pytest.raises(ValueError, match='denominator'): analysis.actual_metrics(fake_evaluation(), 32)


def test_calibration_bins_include_confidence_one_and_equal_history_weight():
    a = analysis.calibration_moments([1.], [0.])
    b = analysis.calibration_moments([.5]*32, [1.]*32)
    result = analysis.aggregate_calibration([a, b], 2)
    assert result['available_histories'] == 2
    assert result['bins'][-1]['mass'] == .5 and result['bins'][-1]['mean_correct'] == 0.
    assert result['ECE'] == pytest.approx(.75)  # History weighting, not 1-vs-32 pooling.
    assert sum(r['conditional_draws'] for r in result['bins']) == 33


def test_fixed_cell_variance_uses_two_history_replicates():
    result = analysis.history_estimate([0., 1., 0., 1.], strata=[0, 0, 1, 1], bootstrap=100)
    assert result['stratified_SE'] == pytest.approx(np.sqrt(.125))
    result = analysis.history_estimate([0., 1.], strata=[0, 1], bootstrap=100)
    assert result['stratified_SE'] is None


def test_same_action_inference_contrast_and_no_regime_pseudoreplication():
    expected = cases.specifications('calibration')[:2]
    records = [fake_record(spec) for spec in expected]
    records[0]['actions']['same']['actual']['regular']['models']['fixed'] = fake_evaluation((0, 0, 0, 0))
    data = analysis.collect_records(records, expected, methods=('constant',), models=('joint', 'fixed'))
    contrasts, rows = analysis.paired_statistics(data, bootstrap=100)
    value = next(r for r in contrasts if r['kind'] == 'same_action_inference' and r['baseline_model'] == 'fixed' and r['regime'] == 'regular')
    assert value['same_action_histories'] == 2
    assert value['differences']['error']['mean'] == -.25
    assert value['differences']['error']['available_histories'] == 2
    summary = analysis.summarize(data, bootstrap=100)
    pooled = next(r for r in summary if r['group'] == 'all' and r['model'] == 'joint' and r['regime'] == 'paired_regime_mean')
    assert pooled['metrics']['error']['available_histories'] == 2
    assert pooled['metrics']['error']['mean'] == .5


def test_failed_prediction_keeps_slots_without_fabricated_metrics():
    expected = cases.specifications('calibration')[:1]; record = fake_record(expected[0])
    del record['actions']['same']['actual']['regular']['models']['joint']
    record['actions']['same']['prediction_failures'] = {'joint': 'saved failure'}
    data = analysis.collect_records([record], expected, methods=('constant',), models=('joint',))
    row = next(v for v in data['rows'] if v['regime'] == 'regular')
    assert row['available'] is False and row['error'] is None
    assert row['missing_reason'] == 'saved failure'


def test_saved_paired_integral_error_is_separate_from_history_uncertainty():
    expected = cases.specifications('calibration')[:1]; record = fake_record(expected[0])
    record['methods']['joint_bc__fixed_charge'] = deepcopy(record['methods']['constant'])
    record['methods']['square__fixed_charge'] = deepcopy(record['methods']['constant'])
    record['independent_paired'] = {'joint_bc__fixed_charge_minus_square__fixed_charge__joint':
        dict(risk=0., risk_se=.004, se_scope='conditional integration')}
    data = analysis.collect_records([record], expected,
        methods=('joint_bc__fixed_charge', 'square__fixed_charge'), models=('joint',))
    _, rows = analysis.paired_statistics(data, bootstrap=100)
    value = next(r for r in rows if r['method'] == 'joint_bc__fixed_charge' and r['baseline'] == 'square__fixed_charge')
    assert value['saved_paired_risk_MC_SE'] == .004
    assert data['independent_integration_pairs'][0]['risk_delta'] == 0.


def test_end_to_end_offline_report_preserves_missing_method_slots(tmp_path):
    expected = cases.specifications('evaluation')[:2]
    source = tmp_path/'records'; target = tmp_path/'report'
    (source/'evaluation').mkdir(parents=True)
    (source/'freeze.json').write_text(json.dumps(dict(policy_id='unit', specifications=dict(evaluation=expected))), encoding='utf-8')
    for spec in expected:
        record = fake_record(spec); record['policy_id'] = 'unit'
        (source/'evaluation'/('history_%03d.json.gz' % spec['index'])).write_bytes(gzip.compress(json.dumps(record).encode(), mtime=0))
    result = analysis.run(source, 'evaluation', target, bootstrap=100)
    assert result['coverage']['exact'] and result['coverage']['expected_histories'] == 2
    assert (target/'results.md').is_file() and (target/'history_metrics.csv').is_file()
    summary = json.loads((target/'summary.json').read_text(encoding='utf-8'))
    missing = next(r for r in summary['summary'] if r['group'] == 'all' and r['method'] == 'joint_bc__mobile_bidir' and r['regime'] == 'regular')
    assert missing['metrics']['error']['available_histories'] == 0
    assert missing['metrics']['error']['expected_histories'] == 2
    assert missing['metrics']['error']['missing_outcome_bounds'] == [0., 1.]
    assert summary['new_fits'] == summary['new_plans'] == summary['new_AC_calls'] == 0


def test_lightweight_view_omits_dense_banks_without_losing_statistics():
    expected = cases.specifications('calibration')[:1]; record = fake_record(expected[0])
    record['actions']['same']['predictions']['joint']['forecast'] = {'bank': {'large_covariance': object()}}
    record['training_result'] = {'fits': [{'candidate': 'star', 'diagnostics': {'giant_J': object()}}]}
    lite = analysis.lightweight_record(record)
    assert 'forecast' not in lite['actions']['same']['predictions']['joint']
    assert 'giant_J' not in lite['training_result']['fits'][0]['diagnostics']
    a = analysis.collect_records([record], expected, methods=('constant',), models=('joint',))
    b = analysis.collect_records([lite], expected, methods=('constant',), models=('joint',))
    assert a['rows'] == b['rows'] and a['coverage'] == b['coverage']
