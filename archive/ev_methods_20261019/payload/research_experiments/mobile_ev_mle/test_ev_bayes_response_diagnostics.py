"""Offline response diagnostics: analytic interactions and saved-record schema."""
from copy import deepcopy
from itertools import product
import gzip
import json

import numpy as np
import pytest

from . import ev_bayes_response_diagnostics as tool


def test_eight_corner_average_and_explicit_interactions():
    # A signed interaction makes isolated fixes misleading; all orders matter.
    e = {bits: 20.-(2*bits[0]+3*bits[1]+4*bits[2]-5*bits[0]*bits[1]+16*np.prod(bits))
         for bits in product((0, 1), repeat=3)}
    row = tool.decompose_mse(e)
    assert row['anchored_single_reduction'] == dict(lines=2., PQ=3., source=4.)
    assert row['anchored_pair_interaction']['lines*PQ'] == -5
    assert row['anchored_triple_interaction'] == 16
    assert sum(row['six_order_average_reduction'].values()) == pytest.approx(20.)
    assert row['sum_identity_error'] < 1e-12
    assert len(row['sequential_orders']) == 6
    with pytest.raises(ValueError, match='eight'): tool.decompose_mse({(0, 0, 0): 1.})


def test_noise_ties_are_not_claimed_exact_rank():
    correct = [[1, 1, 0, 0], [1, 1, 0, 0], [0, 1, 1, 0]]
    pairs = tool.paired_error_differences(correct)
    assert pairs[0]['discordant_noise_draws'] == 0
    assert all(p['classification'] == 'noise_indeterminate' for p in pairs)
    ranking = tool.rank_agreement([.1, .2, .3], [.5, .5, .5])
    assert ranking['Spearman'] is None and ranking['pair_counts']['actual_tie'] == 3
    regret = tool.selection_regret([.1, .1, .2], [.8, .3, .1])
    assert regret['predicted_best_tie_count'] == 2
    assert regret['tie_regret_range'] == pytest.approx([.2, .7])
    with pytest.raises(ValueError, match='binary'): tool.paired_error_differences([[.5, 1], [1, 0]])


def test_clean_geometry_retains_covariance_and_prior_terms():
    bank = dict(means=np.array([[[0.], [1.]]]), covariances=np.array([[[[100.]], [[.1]]]]), log_prior=np.log([.01, .99]))
    row = tool.clean_geometry(bank, [0.], 0)
    assert row['minimum_whitened_square_gap'] == 1.
    pair = row['pair_rows'][0]
    assert pair['log_density_gap'] != pytest.approx(.5*pair['whitened_square_gap'])
    assert pair['log_posterior_score_gap'] == pytest.approx(pair['log_density_gap']+np.log(.01/.99))
    bad = deepcopy(bank); bad['covariances'][0, 0, 0, 0] = -1.
    with pytest.raises(np.linalg.LinAlgError): tool.clean_geometry(bad, [0.], 0)


def test_increment_clean_response_units_and_zero_variance():
    T, n = 2, 2; d = (3*n+2)*T
    saved = dict(mean=np.zeros(d), variance=np.ones(d)*.25,
        line_variance=np.ones(d)*.1, state_variance=np.ones(d)*.2, cross_variance=-np.ones(d)*.05)
    actual = np.zeros(d); actual[8] = 3.
    meter = {c+'_sigma': .01 for c in tool.CHANNELS}
    rows = tool.increment_diagnostics(saved, actual, meter, T, n, 20.)
    voltage = next(r for r in rows if r['channel'] == 'v')
    assert voltage['units'] == 'V_LL' and voltage['RMSE'] == 6.
    assert voltage['interval90_point_coverage'] == .75
    assert max(r['variance_identity_max_error'] for r in rows) < 1e-14
    saved['variance'] = np.zeros(d)
    saved['line_variance'] = saved['state_variance'] = saved['cross_variance'] = np.zeros(d)
    voltage = next(r for r in tool.increment_diagnostics(saved, actual, meter, T, n, 20.) if r['channel'] == 'v')
    assert voltage['zero_variance_miss_count'] == 1


def analytic_physics(candidate, r, x, latent, root):
    n = candidate.n; p, q, sp, sq = latent[:, :n], latent[:, n:2*n], latent[:, -2], latent[:, -1]
    rr, xx = float(np.sum(r)), float(np.sum(x))
    # Includes ordinary line-load-source interactions and a nonlinear load term.
    voltage = np.asarray(root)[:, None]-rr*(p+.2*sp[:, None])-xx*(q+.3*sq[:, None])-.1*rr*p*p
    return dict(p=p, q=q, v=voltage, master_p=p.sum(1)+sp+rr*p.sum(1)**2,
        master_q=q.sum(1)+sq+xx*q.sum(1)**2)


def fake_record():
    T, n, m = 2, 2, 3; d = (3*n+2)*T; base = 20.
    fit = dict(name='star', supports=[[0, 1], [0], [1]], r=[.01]*m, x=[.008]*m)
    truthlines = dict(r=[.015]*m, x=[.005]*m)
    posterior_r, posterior_x = np.ones(m)*.014, np.ones(m)*.006
    eta = np.c_[np.log(posterior_r), np.log(posterior_x/posterior_r)].ravel()
    predicted = np.array([[.1, .12, .03, .04, .1, .03], [.11, .1, .05, .03, .11, .04]])
    actual = predicted+np.array([.01, -.01, .005, -.002, .005, -.003])
    meter = {c+'_sigma': .01 for c in tool.CHANNELS}
    meter.update({c: 0. for c in ('bias_p', 'bias_q', 'bias_v')})
    update = dict(future_latent_mean=predicted, TV_posterior_future=dict(mean=np.r_[eta, predicted.ravel()], cov=np.eye(2*m+len(predicted.ravel()))))
    candidate = tool._candidate(fit, n); powers = dict(zero=np.zeros((T, n)), charge=np.array([[2., 0.], [0., 2.]]), bi=np.array([[4., 0.], [0., -1.]]))
    from .ev_bayes_small_audit import _whiten
    zero_pred = analytic_physics(candidate, posterior_r, posterior_x, predicted, np.ones(T))
    zero_true = analytic_physics(candidate, truthlines['r'], truthlines['x'], actual, np.ones(T))
    actions = {}
    for i, (key, power) in enumerate(powers.items()):
        pp, aa = predicted.copy(), actual.copy(); pp[:, :n] += power/base; aa[:, :n] += power/base
        pred = _whiten(analytic_physics(candidate, posterior_r, posterior_x, pp, np.ones(T)), meter)
        clean = _whiten(analytic_physics(candidate, truthlines['r'], truthlines['x'], aa, np.ones(T)), meter)
        mean = np.array([pred, pred+.03]); cov = np.array([np.eye(d), np.eye(d)])
        delta = pred-_whiten(zero_pred, meter); actual_delta = clean-_whiten(zero_true, meter)
        increment = [dict(candidate=name, mean=delta, variance=np.ones(d)*.2, line_variance=np.ones(d)*.05,
            state_variance=np.ones(d)*.2, cross_variance=-np.ones(d)*.05) for name in ('star', 'other')]
        actions[key] = dict(power_kw=power, predictions=dict(joint=dict(forecast=dict(bank=dict(means=mean[None], covariances=cov[None], log_prior=np.log([.5, .5]))),
            independent_score=dict(risk=.2-.05*i))), response_increment=dict(predictions=increment,
                meter_noise_included=False, shared_line_and_background=True, true_index=0,
                actual={regime: actual_delta for regime in tool.REGIMES}),
            actual={regime: dict(clean_whitened=clean, noise_seed=[123, 4000, 20261017],
                models=dict(joint=dict(samples=dict(correct=[1, 0, 1, int(i != 1)])))) for regime in tool.REGIMES})
    methods = dict(none=dict(action_key='zero', power_kw=powers['zero']), constant=dict(action_key='charge', power_kw=powers['charge']),
        square__mobile_bidir=dict(action_key='bi', power_kw=powers['bi']), joint_bc__mobile_bidir=dict(action_key='bi', power_kw=powers['bi']))
    truth = dict(p=actual[:, :n], q=actual[:, n:2*n], source_p=actual[:, -2], source_q=actual[:, -1], root_v=np.ones(T))
    return dict(spec=dict(index=0, phase='development'), status='complete', candidate_names=['star', 'other'], true_name='star', true_lines=truthlines,
        config=dict(regimes=list(tool.REGIMES)), training_result=dict(fits=[fit]), methods=methods, actions=actions, future_meter=meter,
        prediction=dict(joint=dict(base_kw=base, candidate_updates=dict(star=update)), fixed=dict(candidate_updates=dict(star=dict(future_latent_mean=predicted)))),
        true_future_inputs_evaluation_only={regime: truth for regime in tool.REGIMES})


def test_record_deduplicates_electrical_actions_and_retains_all_slots():
    raw = fake_record(); snapshot = json.dumps(tool.serial(raw), sort_keys=True)
    result = tool.diagnose_record(raw, predictor=analytic_physics)
    assert result['success'], result['failures']
    assert result['denominators']['factorial_slots_expected'] == 8  # 2 unique predeclared x2 centers x2 regimes.
    assert len(result['factorial']) == 8 and len(result['increment_rows']) == 30
    assert len(result['level_rows']) == 30
    assert result['denominators']['factorial_complete_slots'] == 8
    assert all(v['saved_clean_replay_max_error'] < 1e-12 for v in result['factorial'])
    assert all(v['corners_available'] == 8 for v in result['factorial'])
    assert all(v['actual_error_ranking']['actions'] == 3 for v in result['rankings'])
    assert result['new_fits'] == result['new_plans'] == 0
    assert json.dumps(tool.serial(raw), sort_keys=True) == snapshot
    full = tool.diagnose_record(raw, predictor=analytic_physics, all_factorial=True)
    assert full['denominators']['factorial_slots_expected'] == 12
    assert full['new_AC_batches'] == result['new_AC_batches']  # zero baseline already cached.


def test_truth_only_replacement_line_centers_keep_common_forecast():
    raw = fake_record(); result = tool.diagnose_record(raw, predictor=analytic_physics)
    rows = [r for r in result['factorial'] if r['action_key'] == 'bi' and r['regime'] == 'regular']
    line_truth_corner = [r['errors'][2]['corner_MSE']['100'] for r in rows]
    assert line_truth_corner[0] == pytest.approx(line_truth_corner[1])
    assert result['line_errors'][0]['R']['relative_L2'] > result['line_errors'][1]['R']['relative_L2']
    assert rows[0]['errors'][2]['nominal_MSE'] != rows[0]['errors'][7]['nominal_MSE']  # full versus increment.


def test_failures_and_missing_actions_are_not_filtered_from_denominator():
    raw = fake_record(); del raw['actions']['bi']['response_increment']; del raw['methods']['constant']
    result = tool.diagnose_record(raw, predictor=analytic_physics)
    assert not result['success']
    assert result['denominators']['increment_channel_slots_expected'] == 30
    assert result['denominators']['increment_channel_slots_available'] == 20
    assert any(f['stage'] == 'predeclared_method' for f in result['failures'])
    def failing(candidate, r, x, latent, root):
        if np.allclose(r, .015): raise ValueError('kept AC failure')
        return analytic_physics(candidate, r, x, latent, root)
    result = tool.diagnose_record(fake_record(), predictor=failing)
    assert result['denominators']['factorial_slots_expected'] == 8
    assert result['denominators']['factorial_complete_slots'] == 0
    assert all(v['failures'] for v in result['factorial'])
    assert all('errors' not in v for v in result['factorial'])


def test_paired_ranking_requires_common_noise_stream():
    raw = fake_record(); raw['actions']['bi']['actual']['regular']['noise_seed'] = [2, 3, 4]
    row = tool.diagnose_record(raw, predictor=analytic_physics)['rankings'][0]
    assert row['expected_actions'] == 3 and row['available_actions'] == 2
    assert len(row['missing_or_failed_actions']) == 1


def test_cli_io_preserves_missing_histories_and_refuses_overwrite(tmp_path, monkeypatch):
    raw = fake_record(); raw['policy_id'] = 'frozen'; second = dict(index=1, phase='development')
    root = tmp_path/'raw'; (root/'development').mkdir(parents=True)
    (root/'freeze.json').write_text(json.dumps(dict(policy_id='frozen', specifications=dict(development=[raw['spec'], second]))))
    (root/'development'/'history_000.json.gz').write_bytes(gzip.compress(json.dumps(tool.serial(raw)).encode()))
    monkeypatch.setattr(tool, '_predict', analytic_physics)
    result = tool.run(root, 'development', tmp_path/'report')
    assert result['coverage']['expected_histories'] == 2 and len(result['coverage']['missing']) == 1
    assert (tmp_path/'report'/'diagnostics.md').exists()
    with pytest.raises(FileExistsError): tool.run(root, 'development', tmp_path/'report')
    with pytest.raises(ValueError, match='outside'): tool.run(root, 'development', tmp_path/'bad', indices={9})
