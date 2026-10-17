"""Analytic checks of the independent auditor; no scientific simulation."""
from importlib.util import module_from_spec, spec_from_file_location
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pytest


spec = spec_from_file_location('independent_statistical_audit',
                               Path(__file__).with_name('statistical_audit.py'))
audit = module_from_spec(spec); spec.loader.exec_module(audit)

SETTINGS = dict(propagation_mean_MCSE_limit=.2,
                propagation_covariance_MCSE_relative_limit=.05, noise_repeats=1)


def test_cholesky_score_matches_scalar_formula_and_retains_meter_variance():
    bank = dict(means=[[[0.], [4.]]], covariances=[[[[3.]], [[1.]]]],
                log_prior=np.log([.5, .5]))
    result = audit.gaussian_score(bank, [[0.]], 0, multiplier=4.)
    assert result['true_predictive_NLL'] == pytest.approx(.5*np.log(2*np.pi*9.))
    assert result['selected_indices'] == [0]
    assert result['credible_sets90'] == [[0]]
    assert result['coverage90'] == 1.
    assert result['error'] == 0.


def toy_record(index, values=None):
    values = [[0.]*5] if values is None else values
    bank = dict(means=[[np.zeros(5).tolist(), np.full(5, 4.).tolist()]],
        covariances=[[(3*np.eye(5)).tolist(), np.eye(5).tolist()]],
        log_prior=np.log([.5, .5]).tolist())
    fitted = dict(complete=True, failures={},
        losses=dict(fixed=dict(a=2., b=1.), joint=dict(a=.5, b=2.)),
        comparisons={name:{route:dict(success=True)
            for route in ('fixed', 'joint')} for name in ('a', 'b')})
    actual = dict(values=values, clean=[0.]*5,
        refits=[fitted for _ in values], clean_refit=fitted)
    return dict(spec=dict(index=index, seed=100+index, bank='n1'),
        config=dict(noise_repeats=len(values), horizon=1), status='complete', failures={},
        true_name='a', true_index=0, candidate_names=['a', 'b'],
        plans={plan:dict(action_key='shared') for plan in audit.PLANS},
        actions=dict(shared=dict(propagation=dict(status='ok', banks=dict(G00=bank, G11=bank),
            rows=[dict(status='ok', mean_MCSE_whitened_max=.001,
                       covariance_MCSE_relative_fro=.001)]),
            regimes={reg:actual for reg in audit.REGIMES})))


def test_calibration_equal_history_weights_and_zero_truth_SSE_gap():
    first = audit.audit_history(toy_record(0), SETTINGS, calibration_mode=True)
    second = audit.audit_history(toy_record(1, [[10.]*5]*9),
                                 SETTINGS, calibration_mode=True)
    result = audit.calibration_from_audits([first, second])
    for a in audit.GRID:
        left = first['calibration_row']['losses'][str(a)]
        right = second['calibration_row']['losses'][str(a)]
        assert result['objective'][str(a)] == pytest.approx((left+right)/2)
        assert result['objective'][str(a)] != pytest.approx((left+9*right)/10)
    assert result['SSE_thresholds']['fixed']['maxima'] == [1., 1.]
    assert result['SSE_thresholds']['joint']['maxima'] == [0., 0.]
    assert result['SSE_thresholds']['joint']['threshold'] is None


def test_failure_denominator_and_fixed_joint_pairs_use_histories():
    first = audit.audit_history(toy_record(0), SETTINGS)
    failed = audit.audit_history(dict(spec=dict(index=1, seed=101, bank='n1'),
        status='process_failure', failures=dict(worker='timeout')), SETTINGS)
    result = audit.summary([first, failed])
    counts = result['summaries']['SQMD__G00']['availability']
    assert counts['histories'] == 1 and counts['total_histories'] == 2
    assert counts['failed_history_indices'] == [1]
    operational = result['summaries']['SQMD__G00']['operational_all_histories']
    assert operational['metrics']['unresolved']['mean'] == .5
    assert operational['metrics']['false_rejection']['mean'] == 0.
    assert operational['metrics']['correct_decisive']['mean'] == .5
    assert operational['fallback_history_indices'] == [1]
    assert result['summaries']['SQMD__AC_joint']['operational_all_histories'][
        'metrics']['unresolved']['mean'] == 1.  # no calibrated SSE set yet
    pair = result['paired']['SQMD_AC_fixed -> SQMD_AC_joint']
    assert pair['error']['histories'] == 1
    assert pair['error']['total_histories'] == 2
    assert pair['error']['difference_new_minus_old'] == -1.
    assert pair['clean_error']['difference_new_minus_old'] == -1.
    assert result['summaries']['SQMD__AC_joint']['error'][
        'zero_observed_one_sided95_upper'] == pytest.approx(.95)


def test_seed_registry_rejects_overlap_and_duplicate():
    registry = dict(specifications=dict(development=[dict(seed=1)],
        calibration=[dict(seed=2), dict(seed=3)], evaluation=[dict(seed=4)]))
    assert audit.verify_specs(registry)['passed']
    registry['specifications']['evaluation'].append(dict(seed=2))
    assert not audit.verify_specs(registry)['passed']
    registry['specifications']['calibration'].append(dict(seed=2))
    assert not audit.verify_specs(registry)['within_phase_unique']['calibration']


def test_completed_mode_does_not_read_live_batch(tmp_path):
    (tmp_path/'freeze.json').write_text('{}', encoding='utf-8')
    (tmp_path/'controller.json').write_text('{"status":"running"}', encoding='utf-8')
    with pytest.raises(ValueError, match='waits for evaluation_complete'):
        audit.run(tmp_path, mode='completed')


def test_channel_coverage_distinguishes_voltage_from_all_channels():
    bank = dict(means=[[np.zeros(11).tolist(), np.full(11, 10.).tolist()]],
        covariances=[[np.eye(11).tolist(), np.eye(11).tolist()]],
        log_prior=np.log([.5, .5]).tolist())
    values = np.zeros((1, 11)); values[:, 6:9] = 10.
    result = audit.gaussian_score(bank, values, 0, channel_slices=dict(
        terminal_P=slice(0, 3), terminal_Q=slice(3, 6), voltage=slice(6, 9),
        root_P=slice(9, 10), root_Q=slice(10, 11)))
    assert result['coverage90'] == pytest.approx(8/11)
    assert result['coverage90_voltage'] == 0.
    for channel in ('terminal_P', 'terminal_Q', 'root_P', 'root_Q'):
        assert result['coverage90_'+channel] == 1.


def test_completed_mode_independently_rejects_empty_calibration(monkeypatch, tmp_path):
    frozen = dict(settings=dict(SETTINGS, draws=1, replicates=2, clean_refit=True),
        source_sha256={}, specifications={phase:[dict(index=i, seed=base+i,
            bank='n1', phase=phase) for i in range(count)] for phase, base, count in
            [('development', 0, 8), ('calibration', 100, 32), ('evaluation', 200, 32)]})
    frozen['policy_id'] = sha256(json.dumps(frozen, sort_keys=True,
                                           ensure_ascii=False).encode()).hexdigest()
    calibration = dict(histories=32, available_histories=0, G11_multiplier=None,
        policy_id=frozen['policy_id'], input_sha256={
            'calibration/history_%03d.json.gz'%i:'raw-hash' for i in range(32)})
    def read(path):
        path = Path(path)
        if path.name == 'freeze.json': return frozen
        if path.name == 'controller.json': return dict(status='evaluation_complete',
            policy_id=frozen['policy_id'])
        if path.name == 'source_snapshot.json.gz': return {}
        if path.name == 'calibration.json': return calibration
        if path.name == 'evaluation_calibration_binding.json': return dict(
            policy_id=frozen['policy_id'], calibration_sha256='raw-hash')
        index = int(path.name.split('_')[1].split('.')[0])
        return dict(spec=frozen['specifications'][path.parent.name][index],
                    status='process_failure', source_sha256={}, failures=dict(worker='mock'))
    monkeypatch.setattr(audit, 'read', read)
    monkeypatch.setattr(audit, 'digest', lambda _path:'raw-hash')
    result = audit.run(tmp_path, mode='completed')
    assert not result['passed']
    assert result['calibration_hash_binding']['input_hashes_equal']
    assert 'completed evaluation has no declared calibrated multiplier' in result['audit_issues']
