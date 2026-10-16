"""Pure saved-record tests; no model fits, optimization, or power flow."""
from copy import deepcopy
import gzip
import json

import pytest

from . import ev_mobile_v4_analysis as a


def _spec(index=0, regimes=('regular', 'shift')):
    return dict(index=index, seed=2026102000+index, phase='evaluation', bank='n3', H=96, T=12,
                family_index=index%4, regimes=list(regimes))


def _evaluation(halfM=1., complete=True):
    losses = dict(a=2., b=2.+2*halfM, c=100., d=110.)
    return dict(complete=complete, M=2*halfM, losses=losses,
                comparisons={k:dict(joint=dict(success=True, loss=v)) for k, v in losses.items()}, failures={})


def _record(spec=None):
    spec = spec or _spec()
    r = dict(spec=spec, status='complete', true_name='a', methods={}, actual={}, families={},
        config=dict(training=dict(r_bounds=[.5, 1.5], x_bounds=[.1, .5]), ratio_bounds=[.1, .5],
                    source=dict(source_p_bounds=[-.5, .5], source_q_bounds=[-.5, .5])),
        training_result=dict(fits=[dict(candidate=k, success=True, r=[1.], x=[.2], source_p=[.1], source_q=[.1])
                                  for k in ['a', 'b', 'c', 'd']]),
        forecast=dict(source_by_candidate={k:dict(source_p=[.05], source_q=[.03]) for k in ['a', 'b', 'c', 'd']}))
    for family in a.FAMILIES:
        r['families'][family] = dict(logs={})
        for di, domain in enumerate(a.DOMAINS):
            name = family+'__'+domain
            key = name
            r['methods'][name] = dict(action_key=key, planning_success=True, operation=dict(feasible=True),
                nominal=dict(success=True, score=1., nearest=[dict(a=k, half_gap=1.) for k in ['a', 'b', 'c', 'd']]))
            r['families'][family]['logs'][domain] = dict(Q=float(di), seconds=1., steps=[], initials=[])
    for regime in spec['regimes']:
        r['actual'][regime] = dict(actions={name:dict(evaluation=_evaluation()) for name in a.METHODS})
    return r


def _nominal(record, name, values):
    record['methods'][name]['nominal'] = dict(success=True, score=sum(values.values())/len(values),
        nearest=[dict(a=k, half_gap=v) for k, v in values.items()])


def _halfM(record, name, value, regime='regular'):
    key = record['methods'][name]['action_key']
    record['actual'][regime]['actions'][key]['evaluation'] = _evaluation(value)


def test_missing_histories_keep_all_regime_method_and_candidate_slots():
    specs = [_spec(0), _spec(1)]
    summary = a.summarize_records(specs, {})
    assert summary['history_denominator'] == 2
    assert summary['overall']['independent_history_count'] == 2
    assert summary['overall']['history_regime_denominator'] == 4
    assert summary['overall']['method_slot_denominator'] == 48
    assert summary['overall']['candidate_slot_denominator'] == 192
    assert summary['planning']['expected_plan_slots'] == 24
    assert summary['monotonicity']['expected_edges'] == 24
    assert summary['monotonicity']['known'] == 0
    for method in summary['overall']['methods'].values():
        assert method['denominator'] == 4 and method['usable'] == 0
        assert method['candidate_slots'] == dict(expected=16, present=0, successful=0, failed=0, missing=16)
        assert method['M']['mean'] is None


def test_partial_candidates_do_not_create_M_or_precision_denominator():
    record = _record(); name = 'aligned__fixed_charge'
    evaluation = record['actual']['regular']['actions'][name]['evaluation']
    del evaluation['comparisons']['d']; del evaluation['losses']['d']
    projected = a.project_method(record, 'regular', name)
    assert projected['M'] is None and projected['raw_unique_correct'] is None
    assert projected['candidate_slots']['missing'] == 1
    assert projected['fit_status'] == 'failed'
    assert projected['issues'] == ['declared_complete_without_all_valid_candidate_fits']
    summary = a.summarize_records([record['spec']], {0:record})
    method = summary['overall']['methods'][name]
    assert method['denominator'] == 2 and method['usable'] == 1
    assert method['decisions']['raw']['complete_subset'] == 1


def test_physical_and_planning_failure_are_separate_from_valid_fallback_fit():
    record = _record(); name = 'guarded__mobile_bidir'
    record['methods'][name]['planning_success'] = True
    record['methods'][name]['planning_failure'] = 'QP alignment stopped, executable parent retained'
    result = a.project_method(record, 'regular', name)
    assert result['planning_failure'] and result['usable']
    record['methods'][name]['operation']['feasible'] = False
    result = a.project_method(record, 'regular', name)
    assert result['M'] is None and result['diagnostic_M'] == 2.
    assert result['physical_status'] == 'failed' and result['fit_status'] == 'success'


def test_explicit_candidate_exceptions_count_as_failed_slots_not_unattempted_missing():
    record = _record(); name = a.METHODS[0]
    record['actual']['regular']['actions'][name]['evaluation'] = dict(complete=False, M=None,
        losses={}, comparisons={}, failures={k:"KeyError: 'voltage_bounds'" for k in ['a','b','c','d']})
    method = a.project_method(record, 'regular', name)
    assert method['candidate_slots'] == dict(expected=4, present=4, successful=0, failed=4, missing=0)
    assert method['M'] is None


def test_MD_other_parent_reference_drop_is_not_guard_violation():
    record = _record()
    _nominal(record, 'guarded__fixed_charge', dict(a=0., b=0., c=0., d=0.))
    _nominal(record, 'guarded__fixed_bidir', dict(a=2., b=6., c=6., d=6.))
    _nominal(record, 'guarded__mobile_charge', dict(a=10., b=2., c=2., d=2.))
    _nominal(record, 'guarded__mobile_bidir', dict(a=3., b=7., c=7., d=7.))
    methods = {name:a.project_method(record, 'regular', name) for name in a.METHODS}
    rows = [r for r in a.parent_comparisons(methods) if r['family'] == 'guarded' and r['child'] == 'mobile_bidir']
    assert rows[0]['parent'] == 'fixed_bidir' and rows[0]['is_selected_parent']
    assert rows[0]['guard_violation'] is False
    assert rows[1]['parent'] == 'mobile_charge' and rows[1]['role'] == 'other_parent'
    assert rows[1]['delta_nominal_true_ref_m'] == -7.
    assert rows[1]['guard_violation'] is None


def test_metric_and_prediction_discrepancies_add_to_actual_change():
    record = _record()
    _nominal(record, 'aligned__fixed_charge', dict(a=1., b=1., c=1., d=1.))
    _nominal(record, 'aligned__fixed_bidir', dict(a=0., b=3., c=3., d=2.))
    _halfM(record, 'aligned__fixed_charge', 1.)
    _halfM(record, 'aligned__fixed_bidir', .5)
    methods = {name:a.project_method(record, 'regular', name) for name in a.METHODS}
    row = next(r for r in a.parent_comparisons(methods) if r['family'] == 'aligned' and r['child'] == 'fixed_bidir')
    assert row['delta_nominal_Phi'] == 1.
    assert row['delta_nominal_true_ref_m'] == -1.
    assert row['delta_actual_halfM'] == -.5
    assert row['metric_aggregation_discrepancy'] == -2.
    assert row['prediction_actual_discrepancy'] == .5
    assert row['decomposition_error'] == 0.
    assert row['average_metric_sacrifices_true_reference']
    assert not row['predicted_true_reference_gain_actual_loss']


def test_prediction_reversal_is_reported_when_true_reference_also_increases():
    record = _record()
    _nominal(record, 'aligned__fixed_bidir', dict(a=2., b=2., c=2., d=2.))
    _halfM(record, 'aligned__fixed_bidir', -.1)
    methods = {name:a.project_method(record, 'regular', name) for name in a.METHODS}
    row = next(r for r in a.parent_comparisons(methods) if r['family'] == 'aligned' and r['child'] == 'fixed_bidir')
    assert row['predicted_true_reference_gain_actual_loss']
    assert row['predicted_mean_gain_actual_loss']


def test_exact_duplicate_actions_do_not_create_rank_pairs():
    record = _record()
    for name in a.METHODS: record['methods'][name]['action_key'] = 'shared'
    record['actual']['regular']['actions']['shared'] = dict(evaluation=_evaluation())
    methods = {name:a.project_method(record, 'regular', name) for name in a.METHODS}
    ranking = a.rank_pairs(methods)
    assert ranking['method_slots'] == 12
    assert ranking['unique_actions'] == 1 and ranking['duplicate_method_slots'] == 11
    assert ranking['pair_denominator'] == 0


def test_rank_ties_missing_and_strict_reversals_keep_separate_denominators():
    record = _record()
    for i, name in enumerate(a.METHODS): record['methods'][name]['action_key'] = 'x'+str(i%3)
    for i in range(3): record['actual']['regular']['actions']['x'+str(i)] = dict(evaluation=_evaluation(3.-i))
    for i, name in enumerate(a.METHODS):
        value = 1. if i%3 < 2 else 2.
        _nominal(record, name, dict(a=value, b=value, c=value, d=value))
    methods = {name:a.project_method(record, 'regular', name) for name in a.METHODS}
    ranking = a.rank_pairs(methods)
    assert ranking['unique_actions'] == 3 and ranking['pair_denominator'] == 3
    assert sum(r['nominal_Phi']['predicted_tie'] for r in ranking['rows']) == 1
    assert sum(r['nominal_Phi']['reversed'] for r in ranking['rows']) == 2
    assert sum(r['nominal_Phi']['strict'] for r in ranking['rows']) == 2


def test_duplicate_action_conflicting_saved_nominal_is_flagged_not_averaged():
    record = _record(); one, two = a.METHODS[:2]
    record['methods'][two]['action_key'] = one
    _nominal(record, two, dict(a=2., b=2., c=2., d=2.))
    methods = {name:a.project_method(record, 'regular', name) for name in a.METHODS}
    ranking = a.rank_pairs(methods)
    assert len(ranking['issues']) == 2
    assert all(p['nominal_Phi']['complete'] is False for p in ranking['rows'] if p['action_a'] == one or p['action_b'] == one)


def test_square_requires_Q_monotonicity_not_reporting_Phi_monotonicity():
    record = _record()
    _nominal(record, 'square__fixed_bidir', dict(a=-5., b=-5., c=-5., d=-5.))
    summary = a.summarize_records([record['spec']], {0:record})
    row = next(r for r in summary['monotonicity']['rows'] if r['family'] == 'square' and r['child'] == 'fixed_bidir')
    assert row['objective'] == 'precision_Q' and row['delta'] == 1.
    assert row['violation'] is False
    diagnostic = next(r for r in summary['rows'][0]['parent_comparisons'] if r['family'] == 'square' and r['child'] == 'fixed_bidir')
    assert diagnostic['delta_nominal_Phi'] == -6.
    assert diagnostic['Phi_monotonicity_violation'] is None


def test_parameter_audit_checks_line_ratio_and_source_channels_separately():
    record = _record()
    legal = a.parameter_audit(record, 'b', [0., 0., 0., 0., 0., 0.])
    assert legal['known'] and legal['feasible']
    source = a.parameter_audit(record, 'b', [0., 0., .5, 0., 0., 0.])
    assert source['source_failure'] and not source['line_failure']
    line = a.parameter_audit(record, 'b', [0., .5, 0., 0., 0., 0.])
    assert line['line_failure'] and not line['source_failure']
    unknown = a.parameter_audit(record, 'b', [0.])
    assert not unknown['known'] and unknown['feasible'] is None


def test_guard_accepted_floor_is_recomputed_not_trusted_from_success_flag():
    record = _record()
    floor = dict(success=True, complete=True, floor=dict(a=1., b=1., c=1., d=1.),
                 slacks=dict(a=-1., b=2., c=2., d=2.), expected_references=['a','b','c','d'], tolerance=1e-7)
    record['families']['guarded']['logs']['mobile_bidir']['steps'] = [dict(
        accepted=True, mean_improves=True, AC_score=2., previous_AC_score=1., reference_floor=floor)]
    logs = a.planning_logs(record)
    assert logs['guarded_accepted_floor_checked'] == 1
    assert logs['guarded_accepted_floor_violations'] == 1
    assert logs['accepted_events'][0]['floor_audit']['agrees_with_saved_success'] is False


def test_plan_solver_counts_once_even_with_two_regimes():
    record = _record()
    qp = dict(a='a', b='b', z=[0.]*6, primal_upper=1., dual_lower=.9, inner_gap=.1,
              completed=False, explanation_audit=dict(feasible=True), solver=dict(success=True))
    master = dict(objective_lower=.8, objective_upper=1.2, gap=.4, completed=False,
        evaluations=[dict(solutions=[qp])], iterations=[dict(feasible_lower=.8, upper=1.2,
        n_segment_binary=8, generic_segment_binary_estimate=32, segment_binary_saving=24,
        shared_control_square=dict(used=True, zero_product_certificate=dict(valid=True)))])
    record['families']['aligned']['logs']['fixed_charge']['steps'] = [dict(
        accepted=False, master=master, alignment=dict(pairs=[dict(b='b', parent_QP=qp)]))]
    summary = a.summarize_records([record['spec']], {0:record})
    assert summary['planning']['QP_attempts'] == 2
    assert summary['planning']['master_attempts'] == 1
    assert summary['planning']['master_iterations'] == 1
    assert summary['planning']['compression_used'] == 1
    assert summary['planning']['QP_original_domain_known'] == 2
    assert summary['overall']['history_regime_denominator'] == 2


def test_complete_candidate_tied_winner_is_not_unique_correct():
    record = _record(); name = a.METHODS[0]
    _halfM(record, name, 0.)
    method = a.project_method(record, 'regular', name)
    assert method['raw_argmin'] == ['a', 'b']
    assert method['raw_unique_correct'] is False and method['raw_contains_truth'] is True
    assert method['AIC_argmin'] == ['a', 'b']


def test_file_reader_preserves_freeze_identity_and_refuses_overwrite(tmp_path):
    root = tmp_path/'run'; (root/'evaluation').mkdir(parents=True)
    spec = _spec(); record = _record(spec); record['policy_id'] = 'policy'
    frozen = dict(policy_id='policy', source_sha256={'science.py':'abc'},
                  specifications=dict(development=[], evaluation=[spec]))
    (root/'freeze.json').write_text(json.dumps(frozen), encoding='utf8')
    (root/'evaluation'/'000.json.gz').write_bytes(gzip.compress(json.dumps(record).encode()))
    output = tmp_path/'analysis.json'
    result = a.run(root, output, phases=('evaluation',), require_complete=True)
    assert result['phases']['evaluation']['records_present'] == 1
    assert output.with_suffix('.md').exists()
    with pytest.raises(FileExistsError): a.run(root, output)
    record['policy_id'] = 'different'
    (root/'evaluation'/'000.json.gz').write_bytes(gzip.compress(json.dumps(record).encode()))
    with pytest.raises(ValueError, match='policy differs'):
        a.run(root, tmp_path/'other.json', phases=('evaluation',))
