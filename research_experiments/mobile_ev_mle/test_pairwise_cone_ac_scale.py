"""Gate-2 mathematical-object, physical-domain and H-only isolation checks."""
from copy import deepcopy
from pathlib import Path
import numpy as np
import pytest
from . import pairwise_cone_ac_scale as audit
from .information_context import meter_from_record
from .single_site_milp import SingleSiteConstraints
from .information_fit_recovery import independent_replay, write_json

COMPACT_ROOT = Path(__file__).resolve().parent/'evidence/structured_information/pairwise_cone_ac_scale'
COMPACT_H1 = COMPACT_ROOT/'H930001/inputs.json.gz'
COMPACT_H2 = COMPACT_ROOT/'H930002/inputs.json.gz'


@pytest.fixture(scope='module')
def context():
    inputs = audit.load_compact_inputs(COMPACT_H1)
    constraints = SingleSiteConstraints(**inputs['device_constraints'])
    actions = audit.fixed_actions(constraints)
    states = audit.total_states(inputs['centres'][0], actions['constant0']['plan'])
    meter = meter_from_record(inputs['models']['joint'])
    geometry = audit.geometry(inputs, states, meter, audit.likelihood.ac_predict)
    return inputs, actions, states, meter, geometry


def test_filename_first_H_and_common_anchor_are_fixed_without_truth(context):
    inputs, *_ = context
    assert Path(inputs['original_saved_input_file']).name == 'trial_930001_spatial.json.gz'
    assert inputs['input_mode'] == 'archived_compact_evidence' and not inputs['original_raw_file_required']
    assert inputs['history_id'] == '930001' and inputs['anchor_id'] == 'history_star_0'
    assert not inputs['future_truth_accessed'] and not inputs['future_realized_measurements_accessed']
    assert 'truth' not in inputs and 'true_name' not in inputs


def test_all_nine_directed_pairs_and_108_tasks_are_retained(context):
    inputs, *_ = context
    tasks = audit.planned_tasks(sorted(inputs['candidates']))
    assert len(tasks) == 108 and len({row['id'] for row in tasks}) == 108
    assert len({(row['source'], row['competitor']) for row in tasks}) == 9
    assert all(row['source'] != row['competitor'] for row in tasks)
    assert {row['epsilon'] for row in tasks} == {1., .5, .25}
    assert {row['action'] for row in tasks} == {'constant0', 'alternating0'}


def test_two_positive_engineering_directions_respect_original_floor_at_all_scales():
    for direction in audit.DIRECTIONS.values():
        for epsilon in audit.SCALES:
            assert np.all(epsilon*direction >= 1e-6)
            assert np.all(epsilon*direction <= .1)


def test_public_actions_are_same_energy_operationally_feasible(context):
    _, actions, *_ = context
    for row in actions.values():
        assert row['audit']['feasible']
        assert np.isclose(.25*row['plan'].p.sum(), .15)
        assert np.all(row['plan'].q == 0)
        assert np.all(row['plan'].p[:, 1:] == 0)
    np.testing.assert_allclose(actions['constant0']['plan'].p[:, 0], [.15]*4)
    np.testing.assert_allclose(actions['alternating0']['plan'].p[:, 0], [.3, 0., .3, 0.])


def test_EV_enters_source_means_once_and_source_stays_at_same_junction(context):
    inputs, actions, states, *_ = context
    centre = inputs['centres'][0]
    np.testing.assert_allclose(states['p'][12:], np.asarray(centre['future']['p'])+actions['constant0']['plan'].p)
    np.testing.assert_allclose(states['source_p'][12:], centre['future']['source_p'])
    np.testing.assert_allclose(states['p'][:12], centre['history_p'])


def test_all_zero_edge_embeddings_share_star_B_and_physical_means(context):
    *_, g = context
    assert all(row['same_common_nuisance'] for row in g['zero_edge_mapping'])
    assert g['B'].shape == (176, 136)
    assert g['B_rank'] == 136 and g['projection_error'] < 1e-7
    assert g['free_projection_identity'] == 'relaxation_of_original_one_sided_competitor_tangent'
    assert any(row['index'] == 6 and row['side'] == 'lower' for row in g['common_active_boundaries'])


def test_same_calibrated_bias_and_actual_time_array_sigma_enter_physical_means(context):
    inputs, _, states, meter, _ = context
    star = inputs['star']
    physical = audit.likelihood.ac_predict(star, inputs['star_r'], inputs['star_x'],
        states['p'], states['q'], states['source_p'], states['source_q'], states['root_v'])
    obs = audit.means_observations(physical, meter)
    np.testing.assert_allclose(obs.p-physical['p'], np.broadcast_to(meter.bias_p, (16, 3)), atol=1e-14)
    assert np.asarray(meter.v_sigma).shape == (16, 3)
    assert not np.array_equal(np.asarray(meter.v_sigma)[:12], np.asarray(meter.v_sigma)[12:])
    candidate, point = audit.competitor_initial(inputs, states, 'star')
    row = audit.point_record(candidate, point, meter)
    row['loss'] = 0.
    assert independent_replay(candidate, row, obs, meter)['success']


@pytest.mark.parametrize('competitor', ['star', 'clade_02', 'clade_12'])
def test_projected_two_column_QP_has_KKT_nnls_and_homogeneous_scale_identity(context, competitor):
    *_, g = context
    unit = audit.closure_qp(g, 'clade_01', competitor, audit.DIRECTIONS['R'])
    half = audit.closure_qp(g, 'clade_01', competitor, .5*audit.DIRECTIONS['R'])
    assert unit['numeric_KKT_passed'] and half['numeric_KKT_passed']
    assert unit['distance_squared'] <= unit['current_free_projection_distance_squared']+1e-8
    assert half['distance_squared'] == pytest.approx(.25*unit['distance_squared'], rel=1e-9, abs=1e-12)
    assert unit['dual']['candidate_feasible_exact']
    assert unit['full_original_matrix_unconditional_lower_bound'] == 0.
    assert unit['full_original_matrix_positive_rigorous_lower_bound'] is None


@pytest.mark.parametrize('competitor', ['star', 'clade_02', 'clade_12'])
def test_finite_box_and_original_one_sided_tangent_obey_competitor_lower_bound_sign(context, competitor):
    inputs, _, states, _, g = context
    closure = audit.closure_qp(g, 'clade_01', competitor, audit.DIRECTIONS['R'])
    tangent = audit.bounded_local_qp(g, 'clade_01', competitor, audit.DIRECTIONS['R'], tangent=True)
    finite = audit.bounded_local_qp(g, 'clade_01', competitor, audit.DIRECTIONS['R'])
    assert tangent['success'] and finite['success']
    assert tangent['numeric_KKT_passed'] and finite['numeric_KKT_passed']
    assert tangent['common_competitor_step'][6] >= -1e-12
    assert finite['cost']+1e-7 >= tangent['cost']
    assert tangent['cost']+1e-7 >= closure['distance_squared']
    candidate, point = audit.competitor_initial(inputs, states, competitor, finite)
    assert np.all(point['r'] >= 1e-6-1e-12) and np.all(point['x'] >= 1e-6-1e-12)
    # The solver uses model-source = Ar*eta+B*competitor_step-As*zeta.
    residual = g['A'][competitor]@finite['eta']+g['B']@finite['common_competitor_step']-g['A']['clade_01']@audit.DIRECTIONS['R']
    assert float(residual@residual) == pytest.approx(finite['cost'], rel=1e-9, abs=1e-10)


def test_mismatched_future_or_common_zero_edge_point_is_not_silently_replaced(monkeypatch):
    original = audit.read_json
    def changed(path):
        saved = deepcopy(original(path))
        saved['centres'][0]['future']['p'][0][0] += .01
        return saved
    monkeypatch.setattr(audit, 'read_json', changed)
    with pytest.raises(ValueError, match='forecast differs'):
        audit.load_compact_inputs(COMPACT_H1)


def test_nonfinite_failed_state_is_retained_without_interrupting_saved_replay(context):
    inputs, _, states, meter, _ = context
    candidate, point = audit.competitor_initial(inputs, states, 'star')
    physical = audit.likelihood.ac_predict(candidate, **point, root_v=states['root_v'])
    obs = audit.means_observations(physical, meter)
    point['r'][0] = np.nan
    row = audit.point_record(candidate, point, meter)
    result = audit.checked_replay(candidate, row, obs, meter)
    assert not result['success'] and result['failure_state_retained']


def test_safe_instrumentation_restores_replay_binding_on_exception(monkeypatch, context):
    inputs, _, states, meter, _ = context
    candidate, point = audit.competitor_initial(inputs, states, 'star')
    original = audit.recovery_module.independent_replay
    def failure(*args, **kwargs):
        assert audit.recovery_module.independent_replay is audit.checked_replay
        raise ValueError('fixed initial failure')
    monkeypatch.setattr(audit, 'run_fit', failure)
    with pytest.raises(ValueError, match='fixed initial failure'):
        audit.safe_fit(candidate, None, meter, initial=point, mode='test', seed=0)
    assert audit.recovery_module.independent_replay is original


def test_time_major_residual_uses_same_order_as_full_joint_J(context):
    inputs, _, states, meter, _ = context
    candidate, point = audit.competitor_initial(inputs, states, 'star')
    physical = audit.likelihood.ac_predict(candidate, **point, root_v=states['root_v'])
    obs = audit.means_observations(physical, meter)
    np.testing.assert_allclose(audit.time_major_residual(physical, obs, meter), np.zeros(176))
    moved = deepcopy(physical)
    moved['p'] = moved['p'].copy(); moved['p'][0, 1] += np.asarray(meter.p_sigma)[0, 1]
    moved['master_q'] = moved['master_q'].copy(); moved['master_q'][1] += np.asarray(meter.master_q_sigma)[1]
    residual = audit.time_major_residual(moved, obs, meter)
    assert residual[1] == pytest.approx(1.) and residual[21] == pytest.approx(1.)
    assert np.count_nonzero(abs(residual) > 1e-10) == 2


def test_explicit_second_unique_H_is_930002_and_first_quality_filename():
    inputs = audit.load_compact_inputs(COMPACT_H2, history_id='930002')
    assert Path(inputs['original_saved_input_file']).name == 'trial_930002_spatial.json.gz'
    assert inputs['history_id'] == '930002' and inputs['unique_H_order_position'] == 2
    assert inputs['unique_saved_H_ids'][:2] == ['930001', '930002']
    assert not inputs['future_truth_accessed'] and not inputs['future_realized_measurements_accessed']


@pytest.mark.parametrize('history_id', ['930000', 'not_a_history'])
def test_requested_absent_or_invalid_H_is_not_replaced(history_id, tmp_path):
    with pytest.raises(ValueError):
        audit.select_saved_input(pilot=tmp_path, history_id=history_id)


def test_normal_raw_filename_rule_remains_explicit_and_unchanged(tmp_path):
    for compact_path in (COMPACT_H1, COMPACT_H2):
        compact = audit.read_json(compact_path)
        raw = dict(history_id=compact['history_id'], quality=compact['quality'],
            selection=dict(centres=compact['centres']), history_observations=compact['history_observations'],
            measurement_models=compact['models'], device_constraints=compact['device_constraints'],
            seeds=dict(policy=compact['policy_seed']), truth='irrelevant sentinel')
        for suffix in ('spatial', 'uniform'):
            write_json(tmp_path/f"trial_{compact['history_id']}_{suffix}.json.gz", raw)
    first = audit.select_saved_input(pilot=tmp_path)
    second = audit.select_saved_input(pilot=tmp_path, history_id='930002')
    assert first['input_mode'] == second['input_mode'] == 'saved_pilot_filename_order'
    assert Path(first['path']).name == 'trial_930001_spatial.json.gz'
    assert Path(second['path']).name == 'trial_930002_spatial.json.gz'
    assert 'truth' not in first


def test_compact_requested_H_mismatch_is_rejected_without_raw_fallback():
    with pytest.raises(ValueError, match='compact evidence identity'):
        audit.load_compact_inputs(COMPACT_H1, history_id='930002')


@pytest.mark.parametrize('corruption,error', [
    ('same_but_non_H_only_future', 'not the H-only mean'),
    ('joint_sigma', 'joint sigma does not join H/N'),
    ('bias_across_H_N', 'calibrated bias differs'),
    ('constraints', 'action/constraints mismatch'),
    ('EV_twice', 'states mismatch or EV counted twice'),
    ('zero_edge', 'target must be exactly zero'),
])
def test_compact_corruptions_are_rejected_without_repair(tmp_path, corruption, error):
    compact = deepcopy(audit.read_json(COMPACT_H1))
    if corruption == 'same_but_non_H_only_future':
        for centre in compact['centres']: centre['future']['p'][0][0] += .01
    elif corruption == 'joint_sigma':
        compact['models']['joint']['v_sigma'][0][0] *= 2
        compact['actual_joint_meter']['v_sigma'][0][0] *= 2
    elif corruption == 'bias_across_H_N':
        compact['models']['future']['bias_v'][0] += .001
    elif corruption == 'constraints':
        compact['device_constraints']['energy'] += .01
    elif corruption == 'EV_twice':
        compact['total_H_N_states']['constant0']['p'][12][0] += .15
    else:
        compact['centres'][0]['r'][1] = 1e-6
    path = tmp_path/'bad_compact.json.gz'
    write_json(path, compact)
    with pytest.raises(ValueError, match=error):
        audit.load_compact_inputs(path)
