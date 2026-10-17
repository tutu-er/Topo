import numpy as np
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
