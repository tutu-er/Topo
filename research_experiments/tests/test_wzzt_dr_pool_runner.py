"""Frozen-data boundaries and restart behavior of the DR pool replay driver."""
import json
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from research_experiments.wzzt_dr_selection.dr_pool_v2 import runner


def scenario(day, name):
    index=pd.date_range(day,periods=24,freq="h")
    p=pd.DataFrame(np.ones((24,3))*.02,index=index,columns=[101,102,103])
    return runner.observed.make_observed_scenario(p,p*.3,p*0+1.,pd.Series(1.01,index=index),name)


def simulation(day, name):
    s=scenario(day,name)
    return {"observed":{"name":name,"P_terminal":s["P_terminal"],"Q_terminal":s["Q_terminal"],
        "V_terminal":s["V_terminal"],"root_voltage":s["V_root_observed"]},
        "truth":{"truth_supports":[[101],[102],[103],[101,102]]},
        "checks":{"usable_ac_solution":True,"voltage_limits_pass":True}}


def write_pickle(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("wb") as stream:pickle.dump(value,stream)


def source(tmp_path):
    root=tmp_path/"source";root.mkdir()
    days=["2021-02-03","2021-02-04","2021-02-05"]
    runner.write(root/"summary.json",{"status":"complete","wzzt_training_dates":days})
    runner.write(root/"plan.json",{"objective":"d_opt","selected_assignment":{"menus":["P_15"]*3}})
    runner.write(root/"response_draws.json",{"draws":{"unchanged":True}})
    runner.write(root/"run_state.json",{"provenance":{"config":{"seeds":{"action":42,"validation":43}}}})
    for day in days:write_pickle(root/"history"/(day+".pkl"),simulation(day,day))
    for strategy in runner.STRATEGIES:
        folder=root/"strategies"/strategy
        runner.write(folder/"result.json",{"assignment":{"name":strategy,"menus":["no_reward"]*3},"planning_audit":{"feasible":True}})
        for phase,day in [("action","2021-02-09"),("validation","2021-02-10")]:
            write_pickle(folder/(phase+".pkl"),{"simulation":simulation(day,strategy+phase),
                "response":{"total_payment_nok":7.25,"audit":{"valid":True}}})
    return root


def fake_fit(calls):
    def fit(training, validation, output_dir, **kwargs):
        calls.append(sum(len(s['P_terminal']) for s in training))
        assert len(training)==4 and len(validation)==1
        assert all('truth' not in s and 'truth_supports' not in s for s in training)
        assert [s['P_terminal'].index[0].date().isoformat() for s in training]==['2021-02-03','2021-02-04','2021-02-05','2021-02-09']
        point={'supports':[[0],[1],[2],[0,1]],'r':[1,1,1,1],'x':[0,0,0,0],
            'train_mae':.01,'validation_mae':.02,'selection_validation_mae':.015,'origin':'tree_bank'}
        result={'status':'complete','models':{key:point for key in runner.MODEL_NAMES},
            'pool_support_indices':point['supports'],'searches':{'primary':{'reason':'pool_local_optimum'}},'seconds':1.}
        runner.write(Path(output_dir)/'result.json',result)
        return result
    return fit


@pytest.mark.parametrize('n,per_day,step',[(24,6,4),(48,12,2),(96,24,1)])
def test_sampling_preserves_physical_days_channels_and_original_data(n,per_day,step):
    scenes=[scenario(day,day) for day in ['2021-02-03','2021-02-04','2021-02-05','2021-02-09']]
    out=runner.subsample_training(scenes,n)
    assert sum(len(s['P_terminal']) for s in out)==n
    for old,new in zip(scenes,out):
        assert len(old['P_terminal'])==24
        assert len(new['P_terminal'])==per_day
        assert new['P_terminal'].index.hour.tolist()==list(range(0,24,step))
        pd.testing.assert_frame_equal(new['P_terminal'],old['P_terminal'].iloc[::step])
        assert new['V_root_observed'].index.equals(new['P_terminal'].index)


def test_incomplete_day_is_rejected():
    scenes=[scenario(day,day) for day in ['2021-02-03','2021-02-04','2021-02-05','2021-02-09']]
    scenes[0]['P_terminal']=scenes[0]['P_terminal'].iloc[:-1]
    with pytest.raises(ValueError,match='24'):runner.subsample_training(scenes,24)


def test_effective_family_distinguishes_zero_leaf_and_internal_f1():
    point={'supports':[[0],[1],[2],[0,1]],'r':[1,1,0,1],'x':[0,0,0,0]}
    result=runner.support_metrics(point,[[101],[102],[103],[101,102]],[101,102,103])
    assert result['internal_f1']==1 and result['internal_exact']
    assert not result['complete_effective_family']
    assert result['zero_weight_retained_supports']==[[103]]
    assert result['root_matched']==1


def test_replay_keeps_missing_additional_tasks_and_does_not_refit_on_resume(tmp_path):
    root=source(tmp_path);out=tmp_path/'out';calls=[]
    extra=tmp_path/'missing_additional'
    summary=runner.run(root,out,additional_strategy=extra,fitter=fake_fit(calls))
    assert summary['task_denominator']==summary['recorded_tasks']==8
    assert summary['evaluated_tasks']==6 and summary['failed_or_unavailable_tasks']==2
    assert calls==[24,24,24,96,96,96]
    assert summary['source_inputs_unchanged']
    assert all(r['periods']['action']['payment_nok']==7.25 for r in summary['records'] if r['status']=='evaluated')
    again=runner.run(root,out,additional_strategy=extra,resume=True,fitter=lambda *a,**k:pytest.fail('completed task rerun'))
    assert again['evaluated_tasks']==6
    assert not (out/'runner.lock').exists()
    with pytest.raises(FileExistsError):runner.run(root,out,additional_strategy=extra,fitter=fake_fit([]))


def test_resume_rejects_changed_planning_input(tmp_path):
    root=source(tmp_path);out=tmp_path/'out'
    runner.run(root,out,fitter=fake_fit([]))
    plan=runner.read(root/'plan.json');plan['objective']='changed';runner.write(root/'plan.json',plan)
    with pytest.raises(ValueError,match='fingerprint'):runner.run(root,out,resume=True,fitter=fake_fit([]))


def test_unfinished_identifier_checkpoint_is_retained_without_rerun(tmp_path):
    root=source(tmp_path);out=tmp_path/'out';calls=[]
    def fail_first(training,validation,output_dir,**kwargs):
        runner.write(Path(output_dir)/'result.json',{'status':'running'})
        raise RuntimeError('solver stopped')
    summary=runner.run(root,out,sample_sizes=[48],fitter=fail_first)
    assert summary['task_denominator']==3 and summary['failed_or_unavailable_tasks']==3
    again=runner.run(root,out,sample_sizes=[48],resume=True,fitter=lambda *a,**k:pytest.fail('failed task rerun'))
    assert again['failed_or_unavailable_tasks']==3


def test_matched_menu_control_has_separate_cost_and_recovery_comparison(tmp_path):
    root=source(tmp_path);out=tmp_path/'out';extra=tmp_path/'extra'
    import shutil
    shutil.copytree(root/'strategies'/'fixed_P2_first_node',extra)
    summary=runner.run(root,out,additional_strategy=extra,sample_sizes=[24],fitter=fake_fit([]))
    assert summary['task_denominator']==summary['evaluated_tasks']==4
    assert len(summary['paired_vs_fixed_same_menu'])==3
    assert all(p['internal_f1_delta']==0 and p['planned_action_payment_nok']==7.25 for p in summary['paired_vs_fixed_same_menu'])
