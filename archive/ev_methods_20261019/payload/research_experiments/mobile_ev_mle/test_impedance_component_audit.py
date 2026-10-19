import copy
import json

import numpy as np
import pytest

try:
    from .impedance_component_audit import analyze_record, summarize, run_audit
except ImportError:
    from impedance_component_audit import analyze_record, summarize, run_audit


def fixture(history='h1',*,star=False):
    supports=[[0,1,2],[0,1],[0],[1],[2]]
    if star:supports=[[0,1,2],[0],[1],[2]]
    r=[.005,3e-5,.02,.03,.04] if not star else [.005,.02,.03,.04]
    x=[.003,2e-5,.01,.02,.03] if not star else [.003,.01,.02,.03]
    truth=dict(evaluation_only=True,supports=[] if star else [[0,1]],r_pu=r,x_pu=x)
    fit=dict(name='star' if star else 'clade_01',supports=supports,r_pu=list(r),x_pu=list(x),success=True,
             diagnostics=dict(physical_voltage_feasible=True,ac_equation_residual=1e-14,power_balance_residual=1e-14))
    row=dict(history_id=history,case='weak',truth_topology=0 if star else 1,
             candidate_names=[fit['name'],'other'],true_index=0,truth=truth,
             independent_history_unit=True,selection_uses_truth=False,selection_uses_future_noise=False)
    record=dict(method='refit_milp',round=0,success=[True,True],fits=[fit],true_index=0,
                selection_status='complete',operational_voltage_ok=True)
    row['records']=[record]
    return row,record


def test_support_mapping_handles_estimate_order_and_truth_explicit_order():
    row,record=fixture();fit=record['fits'][0]
    fit['r_pu'][1]=.00033;fit['x_pu'][1]=.00022
    permutation=[4,1,3,0,2]
    for key in ('supports','r_pu','x_pu'):fit[key]=[fit[key][i] for i in permutation]
    edge,component=analyze_record(row,record)
    inner=next(e for e in edge if e['edge_class']=='cherry_internal')
    assert inner['absolute_R_error_pu']==pytest.approx(.0003)
    assert inner['absolute_relative_R_error']==pytest.approx(10.)
    assert next(e for e in edge if e['edge_class']=='root_stem')['absolute_R_error_pu']==0
    truth=row['truth'];canonical=[[0,1,2],[0,1],[0],[1],[2]]
    truth['edge_supports']=[canonical[i] for i in permutation]
    for key in ('r_pu','x_pu'):truth[key]=[truth[key][i] for i in permutation]
    edge2,_=analyze_record(row,record)
    assert next(e for e in edge2 if e['edge_class']=='cherry_internal')['absolute_relative_R_error']==pytest.approx(10.)


def test_weak_internal_large_relative_error_does_not_get_averaged_with_leaves():
    row,record=fixture();record['fits'][0]['r_pu'][1]=.00303
    _,classes=analyze_record(row,record);out=summarize(classes)
    inner=out['refit_milp__round_0__cherry_internal']
    leaves=out['refit_milp__round_0__terminal_leaf']
    assert inner['converged_summary']['R_relative_rmse']['mean']==pytest.approx(100.)
    assert leaves['history_count']==1 and leaves['converged_summary']['R_relative_rmse']['available_history_count']==1
    assert leaves['converged_summary']['R_relative_rmse']['mean']==0


def test_star_label_weak_has_no_weak_internal_edge_and_not_fake_zero():
    row,record=fixture(star=True);_,classes=analyze_record(row,record)
    assert all(c['stratum']=='star' for c in classes)
    inner=summarize(classes)['refit_milp__round_0__cherry_internal']['converged_summary']['R_relative_rmse']
    assert inner['history_count']==1 and inner['applicable_history_count']==0
    assert inner['not_applicable_history_count']==1 and inner['mean'] is None


def test_truth_omission_and_missing_true_fit_preserve_total_history_denominator():
    a,ra=fixture('h1');b,rb=fixture('h2');c,rc=fixture('h3')
    rb['true_index']=None;rc['fits']=[]
    classes=[]
    for row,record in ((a,ra),(b,rb),(c,rc)):
        edges,items=analyze_record(row,record);classes.extend(items)
        if row['history_id']!='h1':assert all(e['estimated_R_pu'] is None for e in edges)
    metric=summarize(classes)['refit_milp__round_0__root_stem']['converged_summary']['R_absolute_rmse_pu']
    assert metric['history_count']==3 and metric['available_history_count']==1
    assert metric['missing_applicable_history_count']==2


def test_saved_feasible_unconverged_is_separate_from_converged_summary():
    row,record=fixture();fit=record['fits'][0];fit['success']=False;record['success'][0]=False
    fit['r_pu'][0]=.015
    edges,classes=analyze_record(row,record)
    assert edges[0]['estimate_status']=='saved_feasible_unconverged'
    out=summarize(classes)['refit_milp__round_0__root_stem']
    assert out['converged_summary']['R_absolute_rmse_pu']['mean'] is None
    assert out['saved_feasible_unconverged_summary']['R_absolute_rmse_pu']['mean']==pytest.approx(.01)
    fit['diagnostics']['physical_voltage_feasible']=False
    _,classes=analyze_record(row,record)
    assert summarize(classes)['refit_milp__round_0__root_stem']['saved_feasible_unconverged_summary']['R_absolute_rmse_pu']['mean'] is None


def test_evaluator_truth_marker_required_and_invalid_support_export_not_silently_matched():
    row,record=fixture();row['truth']['evaluation_only']=False
    with pytest.raises(ValueError,match='evaluation_only'):
        analyze_record(row,record)
    row['truth']['evaluation_only']=True;record['fits'][0]['supports'][2]=[1]
    edges,classes=analyze_record(row,record)
    assert all(e['estimate_status']=='missing' and e['estimated_R_pu'] is None for e in edges)
    assert 'invalid_true_fit_export' in classes[0]['missing_reason']


def test_three_leaf_errors_are_one_history_RMS_and_zero_truth_relative_is_undefined():
    row,record=fixture(star=True);record['fits'][0]['r_pu'][1:]=[.021,.032,.043]
    _,classes=analyze_record(row,record)
    leaves=next(c for c in classes if c['edge_class']=='terminal_leaf')
    assert leaves['edge_count']==3
    assert leaves['R_absolute_rmse_pu']==pytest.approx(np.sqrt((.001**2+.002**2+.003**2)/3))
    row['truth']['r_pu'][0]=0.
    _,classes=analyze_record(row,record)
    root=next(c for c in classes if c['edge_class']=='root_stem')
    assert root['R_absolute_rmse_pu']==.005 and root['R_relative_rmse'] is None


def test_complete_folder_rounds_are_not_pooled_and_running_batch_rejected(tmp_path):
    folder=tmp_path/'batch';folder.mkdir();row,r1=fixture();r1['round']=1
    r2=copy.deepcopy(r1);r2['round']=2;row['records']=[r1,r2]
    config=dict(count=1,methods=['refit_milp'],policy_id='fixed',source_sha256={'x':'a'})
    (folder/'config.json').write_text(json.dumps(config),encoding='utf-8')
    (folder/'trial_1.json').write_text(json.dumps(row),encoding='utf-8')
    out=run_audit(folder,tmp_path/'out')
    assert out['history_count']==1
    assert out['component_summary']['refit_milp__round_1__root_stem']['history_count']==1
    assert out['component_summary']['refit_milp__round_2__root_stem']['history_count']==1
    config['count']=256;(folder/'config.json').write_text(json.dumps(config),encoding='utf-8')
    with pytest.raises(ValueError,match='incomplete'):
        run_audit(folder,tmp_path/'unfinished')
