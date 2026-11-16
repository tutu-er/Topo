import numpy as np
import pytest
import json

from rnj_wzzt_dr.experiments.wzzt_long_grid_20261117 import evaluate as e


def point(supports, r, x=None):
    return dict(supports=supports, r=r, x=[0.]*len(r) if x is None else x)


def observation(p, y):
    p, y = np.asarray(p, float), np.asarray(y, float)
    return dict(P=p, Q=np.zeros_like(p), Y=y, labels=list(range(p.shape[1])))


def test_rank_one_response_and_unequal_split_weights():
    p = point([[0], [0, 1]], [2., 3.], [1., 4.])
    R, X = e.matrices(p, 2)
    np.testing.assert_array_equal(R, [[5, 3], [3, 3]])
    np.testing.assert_array_equal(X, [[5, 4], [4, 4]])
    zero = point([], [])
    score = e.loss_totals(zero, [observation([[0., 0.]], [[2., 2.]]),
                               observation(np.zeros((3, 2)), np.ones((3, 2)))])
    assert score['sse'] == 14 and score['cells'] == 8 and score['mse'] == 1.75
    assert score['mse'] != (4+1)/2


def test_strict_tie_uses_common_scale_and_stored_not_active_count():
    rows = [dict(index=9, stored_internal=2, mse=1e-6),
            dict(index=2, stored_internal=0, mse=1e-6+5e-13)]
    assert e.strict_select(rows, reference=1., tie=1e-12)['index'] == 2
    assert e.strict_select(rows, reference=1e-6, tie=1e-12)['index'] == 9
    with pytest.raises(ValueError, match='Duplicate'):
        e.strict_select([rows[0], rows[0]])


def geometry():
    # Shared stem at depth1, proper {0,1} at depth2, terminals below.
    edges = [dict(from_bus=9, to_bus=8, r_ohm=.1, x_ohm=.2),
             dict(from_bus=8, to_bus=7, r_ohm=.2, x_ohm=.3),
             dict(from_bus=7, to_bus=0, r_ohm=.3, x_ohm=.4),
             dict(from_bus=7, to_bus=1, r_ohm=0., x_ohm=0.),
             dict(from_bus=8, to_bus=2, r_ohm=.4, x_ohm=.5)]
    return e.physical_geometry(edges, [0, 1, 2], 9, .8)


def test_physical_depth_zero_line_truth_and_common_separation():
    g = geometry()
    assert (1,) in g['groups']  # zero impedance does not remove physical truth
    assert g['groups'][(0, 1)]['min_root_depth'] == 2
    p = point([[0], [2], [0, 1], [0, 1, 2]], [1., 1., 1., 1.])
    t = e.stratified_topology(p, g)
    assert t['internal']['exact'] and not t['full']['exact']
    assert t['full']['missing'] == [[1]]
    assert t['strata']['internal_root_depth_le2']['true_supports'] == 1
    assert t['strata']['internal_downstream_fraction_ge_half']['true_supports'] == 1
    assert t['strata']['common']['true_supports'] == 1


def test_false_clade_has_no_true_depth_and_crossing_point_rejected():
    g = geometry(); p = point([[0, 2], [0, 1, 2]], [1., 1.])
    t = e.stratified_topology(p, g)
    assert t['false_supports'][0]['true_root_depth'] is None
    assert t['false_supports'][0]['predicted_support_tree_depth'] == 2
    with pytest.raises(ValueError, match='Nonlaminar'):
        e.matrices(point([[0, 1], [1, 2]], [1., 1.]), 3)


def test_full_response_reconstruction_and_matched_units():
    g = geometry()
    p = dict(supports=[list(c) for c in g['groups']],
             r=[2*r['r_ohm']/.8 for r in g['groups'].values()],
             x=[2*r['x_ohm']/.8 for r in g['groups'].values()])
    out = e.response_errors(p, g)
    assert out['matrix']['R']['absolute_frobenius'] == 0
    assert out['matrix']['X']['absolute_frobenius'] == 0
    assert all(row['R']['absolute_error_ohm'] < 1e-15 for row in out['lines'])
    assert next(row for row in out['lines'] if row['support'] == [1])['R']['relative_error'] is None


def test_bank_cumulative_excludes_training_and_preserves_duplicate_topology_points():
    g = e.physical_geometry([dict(from_bus=9, to_bus=0, r_ohm=.4, x_ohm=0.)], [0], 9, .8)
    a, b = point([[0]], [1.]), point([[0]], [2.])
    native = dict(frozen_candidates=[dict(index=0, point=a), dict(index=1, point=b)],
                  config=dict(validation_tie=1e-12, zero_tol=1e-9), reference_mse=1., selected=dict(index=0))
    result = e.evaluate_bank(native, observation([[1.]]*100, [[2.]]*100),
                             observation([[1.]], [[1.]]), observation([[1.]], [[2.]]), g)
    assert result['native_candidate_count'] == 2
    assert result['selections']['validation']['mse']['index'] == 0
    assert result['selections']['cumulative']['mse']['index'] == 0
    assert result['models'][0]['scores']['cumulative']['cells'] == 2
    assert result['selections']['validation']['normalized_mae'] is None


def test_missing_results_not_silent_zero_and_paired_losses_retained():
    left = [dict(id='a', status='complete', internal_correct=True), dict(id='b', status='failed')]
    right = [dict(id='a', status='complete', internal_correct=False), dict(id='b', status='pending')]
    summary = e.status_summary(left)
    assert summary['accuracy_all_planned'] == .5 and summary['completed'] == 1
    assert e.status_summary(right)['accuracy_all_planned'] is None
    pair = e.paired_summary(left, right)
    assert pair['planned_union'] == 2 and pair['completed_pairs'] == 1 and pair['losses'] == 1


def test_solver_bound_conflict_not_promoted_by_status_two():
    obs = observation([[1., 1., 1.]], [[0., 0., 0.]])
    p = point([], [])
    native = dict(config=dict(alpha=.003, gain_tol=1e-10), reference_mse=1.,
                  events=[dict(stage='W', initial_point=p, raw_point=p, closed_point=p,
                               solver=dict(status=2, solve_started=True, objective=0., bound=.2,
                                           gap=0., requested_gap=.0001, seconds=2.))])
    s = e.solver_evidence(native, obs)
    assert s['bound_conflicts'] == 1 and s['events'][0]['valid_lower_bound'] is None
    assert not s['events'][0]['native_subproblem_certificate']
    assert s['events'][0]['first_incumbent_seconds'] is None


def test_actual_start_count_preserved_and_missing_initial_is_unknown():
    obs = observation([[0., 0., 0.]], [[0., 0., 0.]])
    initial = point([[0, 1]], [0.])
    empty = point([], [])
    native = dict(config=dict(alpha=.003, gain_tol=1e-10), reference_mse=1.,
                  events=[dict(stage='W', initial_point=initial, initial_score=.003,
                               raw_point=initial, native_returned_score=.003, closed_point=empty,
                               closed_score=0., solver=dict(objective=.003)),
                          dict(stage='E', solver=dict(status=9, solve_started=True))])
    result = e.solver_evidence(native, obs)
    assert result['events'][0]['native_gain'] == 0
    assert result['events'][0]['closure_gain'] == .003
    assert result['events'][1]['native_improved_initial'] is None


def _mock_run(tmp_path):
    root = tmp_path/'run';root.mkdir()
    def save(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding='utf8')
        return e.file_ref(path)
    files={}
    for split in ('train','validation','new_day'):
        path=root/(split+'.npz')
        np.savez(path, **observation([[1.]], [[1.]]))
        files[split+'.npz']=e.file_ref(path)
    meta=save(root/'metadata.json', dict(root_bus=9,base_kv=.4,base_mva=.2))
    (root/'branches.csv').write_text('from_bus,to_bus,r_ohm,x_ohm,status\n9,0,.4,0,True\n',encoding='utf8')
    ev=save(root/'evaluation.json',dict(evaluation_case_refs={'metadata.json':meta,'branches.csv':e.file_ref(root/'branches.csv')}))
    entry=dict(id='a',case='toy',N=1,pq_sd=.005,repeat=0,labels=[0],zbase_ohm=.8,files=files,evaluation_ref=ev)
    manifest=save(root/'inputs.json',[entry])
    tasks=[dict(task_id='a_'+loss,panel='grid',input_id='a',method='101',loss=loss,result_dir='tasks/a_'+loss)
           for loss in ('mse','mae')]
    tasks += [dict(task_id='a_rnj',panel='grid',input_id='a',method='RNJ1',loss='mse',result_dir='tasks/a_rnj')]
    save(root/'tasks.json',tasks)
    protocol=save(root/'protocol.json',dict(inputs_ref=manifest))
    native=dict(frozen_candidates=[dict(index=0,point=point([[0]],[1.]))],
        config=dict(loss='mse',alpha=.003,gain_tol=1e-10,validation_tie=1e-12,zero_tol=1e-9),
        reference_mse=1.,reference_loss=1.,selected=dict(index=0),events=[])
    nr=save(root/'tasks/a_mse/native.json',native)
    save(root/'tasks/a_mse/result.json',dict(task_id='a_mse',status='complete',native=nr,source_protocol=protocol,seconds=1))
    save(root/'tasks/a_rnj/result.json',dict(task_id='a_rnj',status='failed',source_protocol=protocol,error='budget example',seconds=2))
    return root,save


def test_end_to_end_adapter_keeps_failed_pending_and_terminal_denominators(tmp_path):
    root,save=_mock_run(tmp_path)
    report=e.evaluate_run(root,tmp_path/'report1')
    assert report['status']=='incomplete' and report['completed']==1 and report['planned']==3
    assert report['accuracy_all_planned'] is None and report['errors']==[]
    assert report['status_counts']==dict(complete=1,pending=1,failed=1)
    assert all(p['unpaired_or_incomplete']==1 for p in report['paired'])
    save(root/'complete.json',dict(status='complete_with_failures'))
    report=e.evaluate_run(root,tmp_path/'report2')
    assert report['status']=='passed' and report['all_tasks_terminal']
    assert not report['all_tasks_successful'] and report['accuracy_all_planned']==1/3
    assert report['status_counts']['missing_final_result']==1
    with pytest.raises(ValueError,match='Output exists'):
        e.evaluate_run(root,tmp_path/'report2')


def test_hash_change_is_audit_failure_without_relabeling_success(tmp_path):
    root,_=_mock_run(tmp_path)
    with (root/'train.npz').open('ab') as f:
        f.write(b'changed')
    report=e.evaluate_run(root,tmp_path/'report')
    assert report['status']=='failed' and report['status_counts']['audit_failed']==1
    assert 'Changed source' in report['errors'][0]['error']


def test_saved_matrix_and_earlier_start_bound_counterexample():
    obs=observation([[0.,0.,0.]],[[0.,0.,0.]])
    native=dict(config=dict(alpha=.003,gain_tol=1e-10),reference_mse=1.,events=[dict(
        initial_point=point([],[]),raw_point=point([[0,1]],[0.]),
        solver=dict(status=2,solve_started=True,objective=.003,bound=.003,gap=0.,
                    progress=[dict(callback_event='MIPSOL',seconds=.5,objective=.003)]))])
    result=e.solver_evidence(native,obs)
    assert result['bound_conflicts']==1
    assert result['events'][0]['first_incumbent_seconds']==.5


def test_controller_final_rejects_pending_and_uses_new_output(tmp_path, monkeypatch):
    root,_=_mock_run(tmp_path)
    monkeypatch.setattr(e,'plot_results',lambda out: [])
    receipt=e.study(root,allow_partial=True)
    assert receipt['status']=='incomplete' and receipt['completed']==1
    with pytest.raises(RuntimeError,match='pending tasks'):
        e.study(root,allow_partial=False)
    assert len(list((root/'evaluations').glob('*/report.json')))==2


def test_native_bound_history_cannot_be_erased_by_final_zero_gap():
    obs=observation([[0.,0.,0.]],[[0.,0.,0.]])
    p=point([],[])
    native=dict(config=dict(alpha=.003,gain_tol=1e-10),reference_mse=1.,events=[dict(
        initial_point=p,raw_point=p,solver=dict(status=2,solve_started=True,objective=0.,bound=0.,gap=0.,
        progress=[dict(callback_event='MIP',seconds=.2,objective=None,bound=5.),
                  dict(callback_event='FINAL',seconds=.3,objective=0.,bound=0.)]))])
    result=e.solver_evidence(native,obs)
    assert result['bound_history_conflicts']==1
    assert result['events'][0]['valid_lower_bound'] is None
    assert result['events'][0]['first_incumbent_seconds'] is None
    assert not result['events'][0]['native_subproblem_certificate']
