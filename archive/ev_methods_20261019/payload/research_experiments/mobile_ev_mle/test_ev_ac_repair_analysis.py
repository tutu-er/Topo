from copy import deepcopy
import gzip
import json
import weakref

import numpy as np
import pytest
from . import ev_ac_repair_analysis as tool


def bank(means, variance):
    means = np.asarray(means, float).reshape(1, 2, 1)
    return dict(means=means, covariances=np.array(variance, float).reshape(1, 2, 1, 1),
                log_prior=np.log([.5, .5]))


def test_noiseless_wrong_from_mean_bias_and_mean_repair():
    assert tool.scores(bank([1., .1], [1., 1.]), [0.], 0)['error'] == 1.
    assert tool.scores(bank([0., .1], [1., 1.]), [0.], 0)['error'] == 0.


def test_covariance_can_prefer_farther_mean_at_clean_observation():
    assert tool.scores(bank([.1, 1.], [100., 1.]), [0.], 0)['error'] == 1.
    assert tool.scores(bank([.1, 1.], [1., 1.]), [0.], 0)['error'] == 0.


def test_calibration_keeps_meter_variance():
    _, cov, _ = tool.scaled_bank(bank([0., 1.], [3., 1.]), 4.)
    np.testing.assert_allclose(cov[:, 0, 0], [9., 1.])


def test_numerical_failure_retains_raw_corrections_and_G00():
    b = bank([0., 1.], [1., 1.])
    action = dict(propagation=dict(banks=dict(G00=b, G11=b), rows=[
        dict(status='ok', mean_MCSE_whitened_max=.21,
             covariance_MCSE_relative_fro=.01)]))
    assert set(tool.usable_banks(action)) == {'G00'}
    assert 'G11' in action['propagation']['banks']


def test_failures_preserved_and_regimes_not_independent_histories():
    b = bank([0., 4.], [1., 1.])
    r = dict(spec=dict(index=0, seed=1), candidate_names=['a', 'b'], true_name='a',
        plans={k: dict(action_key='same') for k in ('constant', 'SQMD', 'BCMD')},
        actions=dict(same=dict(propagation=dict(banks=dict(G00=b, G11=b)),
            regimes={r: dict(clean=[0.], values=[[0.], [.1]])
                     for r in ('regular', 'shift')})))
    failed = dict(spec=dict(index=1, seed=2), candidate_names=['a', 'b'], true_name='a',
        plans={k: dict(action_key='missing') for k in ('constant', 'SQMD', 'BCMD')})
    result = tool.summarize([r, failed])
    summary = result['summaries']['BCMD__G00']
    assert summary['error']['histories'] == 1
    assert summary['availability']['total_histories'] == 2
    assert summary['availability']['error_all_histories_range'] == [0., .5]
    c = tool.calibrate([r, failed])
    assert c['available_histories'] == 1
    assert c['G11_multiplier'] == 1.


def test_all_failed_group_keeps_full_denominator():
    failed = dict(spec=dict(index=0, seed=1), status='process_failure')
    result = tool.summarize([failed])
    for summary in result['summaries'].values():
        assert summary['availability']['histories'] == 0
        assert summary['availability']['failed_history_indices'] == [0]
        assert summary['availability']['error_all_histories_range'] == [0., 1.]


def complete_record(index):
    first = bank([0., 4.], [1.5, 1.7])
    second = bank([.1, 3.9], [2.2, 2.3])
    banks = dict(G00=first, G11=second,
        G10=dict(means=second['means'], covariances=first['covariances'],
                 log_prior=first['log_prior']),
        G01=dict(means=first['means'], covariances=second['covariances'],
                 log_prior=first['log_prior']))
    names = ['a', 'b']; true_index = index % 2; true = names[true_index]
    other = names[1-true_index]
    fixed = {true: 1.+index*.1, other: .8+index*.15}
    joint = {true: .7+index*.05, other: .5+index*.1}
    fitted = dict(complete=True, failures={}, losses=dict(fixed=fixed, joint=joint),
        comparisons={k: {route: dict(success=True, usable=True,
            optimization_failure=False, state=dict(unused_source=list(range(20))))
            for route in ('fixed', 'joint')} for k in names},
        starts=[dict(status='recorded_start', success=True)])
    actual = dict(clean=[.2 if true_index == 0 else 3.8],
        values=[[.3 if true_index == 0 else 3.9]],
        refits=[fitted], clean_refit=fitted)
    return dict(spec=dict(index=index, seed=100+index), config=dict(noise_repeats=1),
        candidate_names=names, true_name=true, true_index=true_index,
        status='complete', failures={},
        unused_historical_tensor=np.zeros((50, 50)),
        plans={k: dict(action_key='shared', unused_operation=list(range(10)))
               for k in ('constant', 'SQMD', 'BCMD')},
        actions=dict(shared=dict(methods=['constant', 'SQMD', 'BCMD'], failures={},
            propagation=dict(status='ok', banks=banks, rows=[dict(candidate=k,
                status='ok', mean_MCSE_whitened_max=.001,
                covariance_MCSE_relative_fro=.001) for k in names]),
            regimes={r: deepcopy(actual) for r in ('regular', 'shift')})))


def serialized(value):
    return json.dumps(value, sort_keys=True, allow_nan=False,
                      default=lambda x: np.asarray(x).tolist())


def test_compact_measurements_calibration_and_summary_are_exactly_equal():
    records = [complete_record(i) for i in range(10)]
    partial = complete_record(10)
    partial['status'] = 'partial_failure'
    partial['failures'] = dict(action='missing constant task')
    partial['plans']['constant']['action_key'] = 'missing'
    records.extend([partial, dict(spec=dict(index=11, seed=111),
        status='process_failure', failures=dict(worker='exit 9'))])
    before = serialized(records)
    compact = [tool.compact_record(r) for r in records]
    assert serialized(records) == before
    for full, small in zip(records, compact):
        assert tool.measurements(full) == tool.measurements(small)
    calibration = tool.calibrate(records)
    assert calibration['available_histories'] == 10
    assert calibration['SSE_thresholds']['fixed']['histories'] == 10
    assert calibration['SSE_thresholds']['fixed']['threshold'] is not None
    assert tool.calibrate(compact) == calibration
    assert tool.summarize(records, calibration) == tool.summarize(compact, calibration)
    assert 'unused_historical_tensor' not in compact[0]
    banks = compact[0]['actions']['shared']['propagation']['banks']
    assert banks['G00']['covariances'] is banks['G10']['covariances']
    assert banks['G10']['means'] is banks['G11']['means']
    assert not banks['G11']['covariances'].flags.writeable
    np.testing.assert_array_equal(banks['G11']['means'],
        records[0]['actions']['shared']['propagation']['banks']['G11']['means'])
    assert compact[0]['status_audit'][
        'actions.shared.regimes.regular.refits.0.starts.0.status'] == 'recorded_start'
    assert compact[-1]['status_audit']['failures'] == dict(worker='exit 9')


def test_load_records_reads_gzip_and_json_without_rewriting_raw(tmp_path):
    records = [complete_record(0), complete_record(1)]
    paths = [tmp_path/'history_0.json.gz', tmp_path/'history_1.json']
    paths[0].write_bytes(gzip.compress(serialized(records[0]).encode(), mtime=0))
    paths[1].write_text(serialized(records[1]), encoding='utf-8')
    before = [p.read_bytes() for p in paths]
    result = tool.load_records(paths)
    assert [r['spec']['index'] for r in result] == [0, 1]
    assert [p.read_bytes() for p in paths] == before
    # JSON sort_keys changes display-row order; compare the identical loaded
    # raw representation, rather than the fixture's insertion order.
    assert tool.summarize(result) == tool.summarize([tool.load(p) for p in paths])


def test_load_records_releases_each_raw_record_before_next_load(monkeypatch):
    class Raw(dict):
        pass
    references = []
    def load(index):
        assert all(ref() is None for ref in references)
        record = Raw(spec=dict(index=index, seed=100+index), status='process_failure')
        references.append(weakref.ref(record))
        return record
    monkeypatch.setattr(tool, 'load', load)
    result = tool.load_records(range(3))
    assert len(result) == 3
    assert all(ref() is None for ref in references)


def test_zero_observed_bound_keeps_nonzero_uncertainty():
    assert tool.zero_observed_upper([0.]) == pytest.approx(.95)
    assert tool.zero_observed_upper(np.zeros(32)) == pytest.approx(1-.05**(1/32))
    assert tool.zero_observed_upper([]) is None
    assert tool.zero_observed_upper([0., .5]) is None
    with pytest.raises(ValueError, match=r'\[0,1\]'):
        tool.zero_observed_upper([1.1])
    result = tool.summarize([complete_record(i) for i in range(3)])
    metrics = result['summaries']['constant__G00']
    assert metrics['error']['bootstrap_degenerate']
    assert metrics['error']['bootstrap95'] == [0., 0.]
    assert metrics['error']['zero_observed_one_sided95_upper'] > 0
    assert 'zero_observed_one_sided95_upper' not in metrics['classification_NLL']
    assert 'zero_observed_one_sided95_upper' not in metrics['brier']
