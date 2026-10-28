"""Content-key fit reuse and bounded-batch persistence of the DR matrix."""
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
import pytest
from research_experiments.wzzt_dr_selection.dr_pool_v2 import matrix_runner as matrix
from research_experiments.wzzt_dr_selection.dr_pool_v2 import runner


def simulation(day, shift, name):
    idx=pd.date_range(day,periods=24,freq='h');cols=[101,102,103]
    p=pd.DataFrame(.02+np.arange(72).reshape(24,3)*1e-5,index=idx,columns=cols)
    v=pd.DataFrame(1.-shift,index=idx,columns=cols)
    return {'observed':{'name':name,'P_terminal':p,'Q_terminal':p*.3,'V_terminal':v,
        'root_voltage':pd.Series(1.01,index=idx)},
        'truth':{'truth_supports':[[101],[102],[103],[101,102]]},
        'checks':{'usable_ac_solution':True,'voltage_limits_pass':True}}


def dump(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('wb') as stream:pickle.dump(value,stream)


def physical_sources(tmp):
    scenarios=[];days=['2021-02-03','2021-02-04','2021-02-05']
    for noise,k in [('reference',0),('high_voltage_noise',1)]:
        for mode in (0,1):
            name=f'{noise}__response{mode}';folder=tmp/'physical'/name;folder.mkdir(parents=True)
            runner.write(folder/'summary.json',{'status':'complete','wzzt_training_dates':days})
            runner.write(folder/'plan.json',{'objective':'d_opt','selected_assignment':{'name':'frozen'}})
            runner.write(folder/'response_draws.json',{'mode':mode})
            runner.write(folder/'run_state.json',{'provenance':{'config':{'seeds':{'action':42,'validation':43}}}})
            for day in days:dump(folder/'history'/(day+'.pkl'),simulation(day,k*.0003,'history_'+day))
            for i,strategy in enumerate((*runner.STRATEGIES,runner.EXTRA_STRATEGY)):
                dest=folder/'strategies'/strategy
                runner.write(dest/'result.json',{'assignment':{'name':strategy},'planning_audit':{'feasible':True}})
                for phase,day in [('action','2021-02-09'),('validation','2021-02-10')]:
                    shift=k*.0003+i*.0002+(mode*.00007 if i and phase=='action' else 0.)
                    sim=simulation(day,shift,strategy+'_'+phase)
                    dump(dest/(phase+'.pkl'),{'simulation':sim,'response':{'total_payment_nok':float(i),'audit':{'valid':True}}})
            scenarios.append({'id':name,'source_run':str(folder),'additional_strategy':str(folder/'strategies'/runner.EXTRA_STRATEGY),
                'noise':noise,'response_mode':mode,'response_weight':.5})
    manifest=tmp/'manifest.json';runner.write(manifest,{'status':'complete','scenarios':scenarios});return manifest


def fit_factory(calls,fail_first=False):
    def fit(train,valid,out,**kwargs):
        calls.append(sum(len(s['P_terminal']) for s in train))
        if fail_first and len(calls)==1:raise RuntimeError('frozen terminal solver failure')
        point={'supports':[[0],[1],[2],[0,1]],'r':[1.,1.,1.,1.],'x':[0.,0.,0.,0.],
            'train_mae':0.,'validation_mae':0.,'origin':'tree_bank'}
        result={'status':'complete','models':{name:point for name in runner.MODEL_NAMES},
            'pool_support_indices':point['supports'],'searches':{'main':{'reason':'pool_local_optimum'}}}
        runner.write(Path(out)/'result.json',result)
        return result
    return fit


def test_full48_protocol_batch_limit_and_resume_use42_unique_fits(tmp_path):
    manifest=physical_sources(tmp_path);out=tmp_path/'runs';calls=[]
    first=matrix.run_matrix(manifest,out,max_new_fits=8,fit_function=fit_factory(calls))
    assert first['planned_cells']==48 and first['planned_unique_observation_fits']==42
    assert first['evaluated_cells']==8 and first['pending_cells']==40
    assert first['failed_or_unavailable_cells']==0 and first['stopped_at_invocation_limit']
    assert len(calls)==8 and first['attempted_unique_fits']==8
    assert not (out/'runner.lock').exists()
    complete=matrix.run_matrix(manifest,out,resume=True,fit_function=fit_factory(calls))
    assert complete['status']=='complete' and complete['evaluated_cells']==48
    assert complete['attempted_unique_fits']==complete['completed_unique_fits']==42
    assert complete['reused_evaluated_cells']==6 and len(calls)==42
    assert complete['failed_or_unavailable_cells']==0 and complete['source_inputs_unchanged']
    assert all(r['weighted_internal_f1']==1. and r['complete_empirical_support'] for r in complete['weighted_results'])
    again=matrix.run_matrix(manifest,out,resume=True,fit_function=lambda *a,**k:pytest.fail('completed fit repeated'))
    assert again['evaluated_cells']==48 and again['batch_new_unique_fits']==0
    assert len(calls)==42
    with pytest.raises(ValueError,match='fingerprint'):
        matrix.run_matrix(manifest,out,resume=True,sample_sizes=[24],fit_function=fit_factory([]))


def test_cached_failure_is_reused_and_empirical_weights_are_not_renormalized(tmp_path):
    manifest=physical_sources(tmp_path);out=tmp_path/'runs';calls=[]
    result=matrix.run_matrix(manifest,out,sample_sizes=[24],fit_function=fit_factory(calls,fail_first=True))
    assert result['planned_cells']==16 and result['planned_unique_observation_fits']==14
    assert result['failed_unique_fits']==1 and result['failed_or_unavailable_cells']==2
    assert len(calls)==14 and result['evaluated_cells']==14
    failed=[r for r in result['records'] if r['status']=='failed']
    assert len({r['observation_key'] for r in failed})==1
    values=[r for r in result['weighted_results'] if r['noise']=='reference' and r['strategy']=='no_reward']
    assert all(r['weighted_internal_f1'] is None and r['missing_weight']==1 for r in values)
    again=matrix.run_matrix(manifest,out,resume=True,sample_sizes=[24],fit_function=lambda *a,**k:pytest.fail('failed fit repeated'))
    assert again['failed_or_unavailable_cells']==2


def test_interrupted_shared_attempt_is_not_restarted(tmp_path):
    scenes=[runner.observed_only(simulation(day,0.,day)) for day in ['2021-02-03','2021-02-04','2021-02-05','2021-02-09']]
    valid=[runner.observed_only(simulation('2021-02-10',0.,'valid'))]
    def interrupted(*a,**k):raise KeyboardInterrupt()
    first=matrix.SharedFitter(tmp_path/'cache',sources={'test':'fixed'},fit_function=interrupted)
    with pytest.raises(KeyboardInterrupt):first(scenes,valid,tmp_path/'one')
    second=matrix.SharedFitter(tmp_path/'cache',sources={'test':'fixed'},fit_function=lambda *a,**k:pytest.fail('interrupted solve repeated'))
    with pytest.raises(matrix.CachedFitFailure,match='interrupted'):second(scenes,valid,tmp_path/'two')
    assert second.new_count==0 and second.reused_count==1


def test_identity_ignores_scenario_names_but_retains_dates_and_algorithm(tmp_path):
    scenes=[runner.observed_only(simulation(day,0.,day)) for day in ['2021-02-03','2021-02-04','2021-02-05','2021-02-09']]
    valid=[runner.observed_only(simulation('2021-02-10',0.,'valid'))]
    first,_=matrix.observation_identity(scenes,valid,sources={'test':'fixed'})
    renamed=[dict(s,name='other_'+s['name']) for s in scenes]
    same,_=matrix.observation_identity(renamed,valid,sources={'test':'fixed'})
    changed,_=matrix.observation_identity(renamed,valid,sources={'test':'changed'})
    assert same==first and changed!=first
