import json
from .summarize import summarize


def save(root,algorithm,*,loss=10.,gradient=.01,initial='same',status='returned'):
    path=root/'tasks'/algorithm/'result.json';path.parent.mkdir(parents=True)
    task=dict(network='small',candidate_name='a',prefix=12,start='prefix_linear',
        algorithm=algorithm,repeat=0,inner_tol=1e-10,history_content_sha256='history',task_id=algorithm)
    record=dict(task=task,status=status,process_wall_seconds=2.)
    if status=='returned':
        record['worker']=dict(solver_success=True,initial_state_sha256=initial,
            timings=dict(fit_seconds=1. if algorithm=='ref_sparse' else .1,initialization_seconds=.8),
            fit=dict(loss=loss),audit=dict(available=True,loss=loss,
                reference_scaled_projected_gradient=gradient,
                physical_voltage_feasible=True,line_domain_feasible=True,
                source_and_optimizer_box_feasible=True,ac_equation_residual=1e-15,
                power_balance_residual=1e-15,reported_loss_absolute_difference=0.))
    path.write_text(json.dumps(record))


def test_faster_wrong_loss_or_initial_is_not_qualified(tmp_path):
    save(tmp_path,'ref_sparse');save(tmp_path,'block_reference',loss=11.)
    save(tmp_path,'block_jac',initial='different');save(tmp_path,'block_gpu',status='timeout')
    report=summarize(tmp_path)
    assert report['attempted_saved']==4 and report['timeout_count']==1
    rows={row['algorithm']:row for row in report['rows']}
    assert not rows['block_reference']['qualified_speed_comparison']
    assert not rows['block_jac']['qualified_speed_comparison']


def test_good_pair_keeps_cold_process_time_separate(tmp_path):
    save(tmp_path,'ref_sparse');save(tmp_path,'block_jac',loss=9.,gradient=.005)
    report=summarize(tmp_path)
    row=next(r for r in report['rows'] if r['algorithm']=='block_jac')
    assert row['qualified_speed_comparison'] and row['fit_speed_ratio']==10.
    assert row['process_seconds']==2. and row['initialization_seconds']==.8
