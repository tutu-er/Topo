import copy
import json
from pathlib import Path

import numpy as np
import pytest

try:
    from .empirical_design_analysis import (wrong_margin, interval_gain, proxy_diagnostics,
        extract_history, measurement_summary, paired_gain_summary, correlation_summary,
        summarize, load_batch, load_collector, run_analysis)
except ImportError:
    from empirical_design_analysis import (wrong_margin, interval_gain, proxy_diagnostics,
        extract_history, measurement_summary, paired_gain_summary, correlation_summary,
        summarize, load_batch, load_collector, run_analysis)


METHODS = ['none', 'constant_fixed', 'fixed_pulse', 'random_action',
           'legacy_library_proxy', 'fixed_mle_milp', 'refit_milp']


def _action(name, kw):
    return dict(name=name, p=(np.asarray(kw) / 20).tolist())


def _record(method, kw, losses, name=None):
    return dict(method=method, action=_action(name or method, kw), losses=losses,
                success=[True] * len(losses), operational_voltage_ok=True,
                status='complete', selection_status='complete', ac_calls=7,
                returned_fit_ac_calls=5, seconds=1.)


def _row(history='h1'):
    zero, const, best, fixed = [[0., 0.]], [[.5, 0.]], [[1., 0.]], [[0., 1.]]
    rollback = [dict(pair_name='star_centre_0__clade_01_centre_0__mean__local_refit',
        proxy_q_lower=30., proxy_q_upper=31., actual_soft_cost=27.,
        actual_history_costs=[10., 12.], actual_future_distance_squared=5.,
        joint_taylor_error_norm=2., physically_feasible=True)]
    refit = dict(proxy_lower=30., proxy_upper=999., selected_u_kw=best,
        records=[dict(u_kw=zero, q_lower=10., q_upper=11.),
                 dict(u_kw=const, q_lower=20., q_upper=21.),
                 dict(u_kw=best, q_lower=30., q_upper=31.)],
        actual_rollback=rollback, selection_ac_predict_calls=50,
        total_selection_seconds=3., seconds=2.5, qp_count=5,
        solver_records=[dict(kind='direction_lower', status=0, variables=30),
                        dict(kind='secant_upper', status=1, variables=80, bound=None)],
        proxy_gap_met=False, proxy_stop_reason='budget_or_iteration_limit')
    fixed_search = dict(proxy_upper=1000., records=[dict(u_kw=fixed, q_lower=80., q_upper=81.)],
                        selected_u_kw=fixed)
    library = dict(criterion='same frozen soft-history proxy as refit MILP', selected='random_site_1',
        actions=[dict(action='random_site_1', success=True, q_lower=22., q_upper=23.),
                 dict(action='pulse_site_0', success=True, q_lower=25., q_upper=26.)])
    records = [_record('none', zero, [9., 8.]), _record('constant_fixed', const, [7., 9.]),
        _record('fixed_pulse', best, [8., 12.], 'pulse_site_0'),
        _record('random_action', [[0., .7]], [8., 10.], 'random_site_1'),
        _record('legacy_library_proxy', [[0., .7]], [8., 10.], 'random_site_1'),
        _record('fixed_mle_milp', fixed, [9., 13.]), _record('refit_milp', best, [6., 10.])]
    return dict(history_id=history, candidate_names=['star', 'clade_01'], true_index=0,
        truth_topology=0, case='weak', independent_history_unit=True,
        selection_uses_truth=False, selection_uses_future_noise=False, seconds=20.,
        selection=dict(history=dict(config=dict(base_kva=20.)),
            selector_ac_predict_calls=100, methods=dict(refit_milp=refit,
            fixed_mle_milp=fixed_search, legacy_library_proxy=library)), records=records)


def _batch(path, rows, count=None):
    path.mkdir()
    config = dict(count=len(rows) if count is None else count, methods=METHODS,
                  policy_id='frozen', population_id='iid', source_sha256={'driver': 'abc'},
                  phase='pilot', policy=dict(gates_influence_shadow_policy=False))
    (path / 'config.json').write_text(json.dumps(config), encoding='utf-8')
    for index, row in enumerate(rows):
        (path / f'trial_{index + 1}.json').write_text(json.dumps(row), encoding='utf-8')
    return path


def test_signed_margin_preserves_positive_evidence_and_negative_order():
    names = ['true', 'bad1', 'bad2']
    a = wrong_margin(dict(losses=[4., 9., 7.], success=[True] * 3), 0, names)
    b = wrong_margin(dict(losses=[10., 9., 7.], success=[True] * 3), 0, names)
    assert a['wrong_margin'] == 3 and a['true_gap_reconstructed'] == 0
    assert b['wrong_margin'] == -3 and b['true_gap_reconstructed'] == 3
    assert a['signed_half_loglik_margin'] == 1.5
    assert a['closest_wrong_candidate'] == 'bad2'


def test_failed_competitor_cannot_create_a_positive_margin():
    out = wrong_margin(dict(losses=[4., 9.], success=[True, False]), 0, ['true', 'bad'])
    assert out['wrong_margin'] is None and out['margin_missing_reason'] == 'necessary_fit_failed'


def test_null_nonfinite_and_missing_truth_are_preserved():
    assert wrong_margin(dict(losses=[4., None], success=[True, True]), 0, ['a', 'b'])['wrong_margin'] is None
    assert wrong_margin(dict(losses=[4., 9.], success=[True, True]), None, ['a', 'b'])['margin_missing_reason'] == 'truth_omitted'


def test_invalid_truth_axis_raises():
    with pytest.raises(ValueError, match='true_index'):
        wrong_margin(dict(losses=[1., 2.], success=[True, True]), 2, ['a', 'b'])


def test_gain_interval_uses_cross_endpoints():
    gain = interval_gain(dict(lower=20., upper=23., midpoint=21.5),
                         dict(lower=10., upper=12., midpoint=11.))
    assert gain == dict(lower=8., upper=13., midpoint=10.5)


def test_selected_value_is_not_global_design_upper():
    row = _row(); record = next(r for r in row['records'] if r['method'] == 'refit_milp')
    out = proxy_diagnostics(row, record)
    assert out['refit_proxy']['upper'] == 31.
    assert out['global_design_upper'] == 999.
    assert not out['global_design_upper_is_selected_value']
    assert out['active_actual_future_share_of_witness'] == pytest.approx(5 / 27)
    assert out['proxy_active_pair_contains_truth']
    assert out['actual_rollback_cost_is_nonlinear_infimum_upper_only']


def test_global_numeric_crossing_is_not_clipped_or_called_strict_order():
    row = _row(); search = row['selection']['methods']['refit_milp']
    search.update(proxy_lower=30., proxy_upper=30. - 3e-6)
    out = proxy_diagnostics(row, row['records'][-1])
    assert out['global_design_raw_numeric_gap'] < 0
    assert out['global_design_direction_conflict']
    assert out['global_design_conflict_within_frozen_tolerance']
    assert not out['global_design_bounds_strictly_ordered']


def test_selected_qp_numeric_crossing_keeps_raw_and_removes_gain_bound_claim():
    row = _row(); search = row['selection']['methods']['refit_milp']
    search['records'][0].update(q_lower=10. + 1e-7, q_upper=10.)
    entry = extract_history(row, METHODS)[-1]
    gain = entry['refit_proxy_gain_vs_none']
    assert gain['midpoint'] is not None
    assert gain['lower'] is None and gain['upper'] is None
    assert gain['numerical_direction_conflict']


def test_fixed_mle_score_not_silently_called_refit_score():
    row = _row(); record = next(r for r in row['records'] if r['method'] == 'fixed_mle_milp')
    out = proxy_diagnostics(row, record)
    assert out['own_proxy']['lower'] == 80.
    assert out['refit_proxy'] is None


def test_library_matching_produces_same_frozen_proxy_only():
    row = _row(); record = next(r for r in row['records'] if r['method'] == 'random_action')
    assert proxy_diagnostics(row, record)['refit_proxy']['lower'] == 22.
    row['selection']['methods']['legacy_library_proxy']['criterion'] = 'other'
    assert proxy_diagnostics(row, record)['refit_proxy'] is None


def test_wrong_action_rollback_is_not_reused():
    row = _row(); record = next(r for r in row['records'] if r['method'] == 'refit_milp')
    record['action']['p'] = [[0., .06]]
    assert proxy_diagnostics(row, record)['refit_proxy'] is None


def test_partial_rollback_cannot_delete_missing_competitor_to_raise_score():
    row = _row(); record = row['records'][-1]
    search = row['selection']['methods']['refit_milp']
    search['records'] = []
    search['pair_hashes'] = ['present_pair', 'missing_pair']
    out = proxy_diagnostics(row, record)
    assert out['refit_proxy'] is None and out['own_proxy'] is None
    assert not out['rollback_proxy_bounds_complete']
    assert out['proxy_active_pair_may_be_partial_rollback']


def test_missing_taylor_diagnostic_is_not_zero_error():
    row = _row(); del row['selection']['methods']['refit_milp']['actual_rollback'][0]['joint_taylor_error_norm']
    assert proxy_diagnostics(row, row['records'][-1])['worst_joint_taylor_error_norm'] is None


def test_baseline_matching_records_uses_power_scale_once():
    row = _row(); record = next(r for r in row['records'] if r['method'] == 'constant_fixed')
    out = proxy_diagnostics(row, record)
    assert out['refit_proxy']['lower'] == 20.


def test_tiny_solver_roundoff_matches_but_real_action_change_does_not():
    row = _row(); record = next(r for r in row['records'] if r['method'] == 'constant_fixed')
    record['action']['p'][0][0] += 1e-9
    assert proxy_diagnostics(row, record)['refit_proxy'] is not None
    record['action']['p'][0][0] += .001
    assert proxy_diagnostics(row, record)['refit_proxy'] is None


def test_rolling_full_plan_score_is_not_prefix_score():
    row = _row(); record = copy.deepcopy(row['records'][-1]); record['round'] = 1
    out = proxy_diagnostics(row, record)
    assert out['refit_proxy'] is None
    assert 'executed-prefix' in out['proxy_missing_reason']


def test_all_history_failure_counts_survive_fallback_numeric_results():
    row = _row(); row['records'][-1]['selection_status'] = 'selection_failure_fallback'
    entries = extract_history(row, METHODS)
    refit = next(r for r in entries if r['method'] == 'refit_milp')
    assert refit['wrong_margin'] == 4.
    assert not refit['algorithm_margin_usable'] and not refit['strict_truth_winner_all_history_event']
    assert refit['margin_gain_vs_none'] == 5.
    assert refit['refit_proxy_gain_vs_none']['lower'] == 19.


def test_no_competitor_deletion_for_fit_failure():
    row = _row(); row['records'][-1]['success'][1] = False
    entries = extract_history(row, METHODS)
    refit = next(r for r in entries if r['method'] == 'refit_milp')
    assert refit['wrong_margin'] is None and refit['fit_failure']
    assert refit['margin_gain_vs_none'] is None


def test_complexity_uses_instrumented_count_instead_of_returned_fits():
    refit = next(r for r in extract_history(_row(), METHODS) if r['method'] == 'refit_milp')
    assert refit['joint_fit_ac_calls'] == 7
    assert refit['selection_ac_calls'] == 50
    assert refit['milp_time_limit_count'] == 1
    assert refit['upper_milp_missing_bound_count'] == 1
    assert refit['maximum_milp_variables'] == 80


def test_missing_complexity_is_not_zero_calls():
    row = _row(); row['records'][-1]['ac_calls'] = None
    entry = extract_history(row, METHODS)[-1]
    assert entry['joint_fit_ac_calls'] is None


def test_signed_gain_wins_losses_ties_and_missing_partition_all_histories():
    out = paired_gain_summary([2., -1., 0., None], draws=100, seed=3)
    assert (out['win_count'], out['loss_count'], out['tie_count'], out['missing_count']) == (1, 1, 1, 1)
    assert out['win_fraction_full_history'] == .25
    assert out['mean'] == pytest.approx(1 / 3)
    assert out['median'] == 0


def test_bootstrap_reproducible_and_keeps_missing_denominator():
    first = measurement_summary([1., None, 2., 7.], draws=100, seed=7)
    second = measurement_summary([1., None, 2., 7.], draws=100, seed=7)
    assert first == second
    assert first['history_count'] == 4 and first['available_count'] == 3
    assert first['bootstrap_unit'] == 'independent history'


def test_spearman_uses_tied_ranks_and_constant_is_undefined():
    out = correlation_summary([1., 1., 3., None], [3., 3., 1., 2.], draws=10, seed=9)
    assert out['spearman'] == pytest.approx(-1.) and out['available_pair_count'] == 3
    assert correlation_summary([1., 1., 1.], [1., 2., 3.])['spearman'] is None


def test_within_history_method_rows_are_not_independent_sample_count():
    entries = extract_history(_row('h1'), METHODS) + extract_history(_row('h2'), METHODS)
    out = summarize(entries, METHODS, draws=30, seed=2)
    assert out['history_count'] == 2
    assert out['within_history_rank_associations']['none']['history_count'] == 2
    assert out['methods']['refit_milp']['wrong_margin']['history_count'] == 2


def test_duplicate_or_missing_history_method_axis_is_rejected():
    entries = extract_history(_row(), METHODS)
    with pytest.raises(ValueError, match='incomplete or duplicated'):
        summarize(entries[:-1], METHODS, draws=0)
    with pytest.raises(ValueError, match='incomplete or duplicated'):
        summarize(entries + entries[:1], METHODS, draws=0)


def test_running_batch_is_not_implicitly_analyzed(tmp_path):
    folder = _batch(tmp_path / 'running', [_row()], count=128)
    with pytest.raises(ValueError, match='running batch'):
        load_batch(folder)


def test_selector_leakage_and_duplicate_histories_rejected(tmp_path):
    row = _row(); row['selection_uses_future_noise'] = True
    folder = _batch(tmp_path / 'leak', [row])
    with pytest.raises(ValueError, match='separation'):
        load_batch(folder)
    folder = _batch(tmp_path / 'dup', [_row(), _row()])
    with pytest.raises(ValueError, match='duplicate'):
        load_batch(folder)


def test_collector_binding_rejects_stale_manifest(tmp_path):
    folder = _batch(tmp_path / 'batch', [_row()]); batch = load_batch(folder)
    audit = tmp_path / 'audit'; audit.mkdir()
    summary = dict(evaluation_config=copy.deepcopy(batch['config']), evaluation_manifest=copy.deepcopy(batch['manifest']), calibration={'external': True})
    (audit / 'audit_summary.json').write_text(json.dumps(summary), encoding='utf-8')
    (audit / 'history_gates.json').write_text(json.dumps([dict(history_id='h1', methods={})]), encoding='utf-8')
    assert load_collector(audit, batch)[1]['thresholds_fitted_by_this_module'] is False
    summary['evaluation_manifest']['config_sha256'] = 'changed'
    (audit / 'audit_summary.json').write_text(json.dumps(summary), encoding='utf-8')
    with pytest.raises(ValueError, match='different saved trials'):
        load_collector(audit, batch)


def test_complete_pilot_artifacts_and_truth_strata(tmp_path):
    a, b = _row('h1'), _row('h2'); b.update(truth_topology=2, case='strong')
    folder = _batch(tmp_path / 'pilot', [a, b])
    out = tmp_path / 'output'
    summary = run_analysis(folder, out, draws=20, seed=1)
    assert summary['history_count'] == 2 and set(summary['strata']) == {'star', 'strong_cherry'}
    assert summary['complexity_accounting']['instrumented_calls']['mean'] == 149
    assert (out / 'history_method_metrics.csv').exists()
    assert not summary['optimization_or_gate_tuning_performed']


def test_failure_denominators_are_not_listwise_deleted_across_methods():
    a, b = _row('h1'), _row('h2'); b['records'][-1]['success'][1] = False
    entries = extract_history(a, METHODS) + extract_history(b, METHODS)
    out = summarize(entries, METHODS, draws=20)
    assert out['methods']['none']['wrong_margin']['available_count'] == 2
    assert out['methods']['refit_milp']['wrong_margin']['available_count'] == 1
    assert out['methods']['refit_milp']['events']['fit_failure']['history_count'] == 2
    assert out['paired_against']['none']['refit_milp']['signed_margin_gain']['missing_count'] == 1


def test_optional_calibrated_elimination_association_preserves_history_unit():
    entries = []
    for index in range(3):
        gates = {method: dict(unique_correct_retention=bool(index), algorithm_unique_correct_retention=bool(index),
            false_rejection=False, any_wrong_candidate_eliminated=bool(index), unresolved=not bool(index),
            wrong_elimination_fraction=index / 2 if method == 'refit_milp' else 0.) for method in METHODS}
        entries.extend(extract_history(_row(f'h{index}'), METHODS, gates))
    summary = summarize(entries, METHODS, draws=10)
    assert summary['methods']['refit_milp']['refit_proxy_vs_calibrated_elimination_fraction']['history_count'] == 3
    assert summary['paired_against']['none']['refit_milp']['refit_proxy_gain_vs_calibrated_elimination_gain']['available_pair_count'] == 3
