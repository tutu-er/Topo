"""Predeclared two-candidate experiment and planning information boundary."""
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from . import ev_response_metric_study as study
from . import likelihood as lk


def test_exact_pair_and_nine_methods_declared_without_data():
    bank=study.declared_bank()
    assert [c.name for c in bank]==['star','clade_01']
    assert bank[0].supports!=bank[1].supports
    assert study.METHODS==('none','constant','highlow','square','nearest','gradac','contrast_joint','contrast_box','contrast_load')
    assert study.CONFIG['domains']==['v2g_fixed']
    assert study.CONFIG['wzzt_used'] is False
    assert [study.family_for_seed(seed) for seed in study.CONFIG['calibration_seeds']].count(0)==9
    assert [study.family_for_seed(seed) for seed in study.CONFIG['evaluation_seeds']].count(1)==4
    with pytest.raises(ValueError):study.family_for_seed(-1)


def test_config_rejects_cross_phase_overlap_extra_candidates_and_mobile_domain():
    study.validate_config(study.CONFIG)
    for key,value in [('domains',['v2g_mobile']),('declared_candidates',['star','clade_01','clade_02'])]:
        cfg=deepcopy(study.CONFIG);cfg[key]=value
        with pytest.raises(ValueError):study.validate_config(cfg)
    cfg=deepcopy(study.CONFIG);cfg['evaluation_seeds'][0]=cfg['development_seeds'][0]
    with pytest.raises(ValueError,match='disjoint'):study.validate_config(cfg)


def test_public_square_has_correct_fixed_site_task_and_analytic_value():
    curves=study.previous.public_curves(6,3)
    for name in ('constant','highlow','analytic_square'):
        op=study.dynamic.operation_record(curves[name],mobile=False,discharge=name=='analytic_square')
        assert op['success'] and op['moves']==0 and op['driving_kwh']==0
        assert op['energy_kwh'][-1]==pytest.approx(26.85,abs=1e-7)
        assert curves[name][:,1:]==pytest.approx(np.zeros((6,2)))
    assert np.sum(curves['analytic_square']**2)==pytest.approx(157.2383160982642)


def test_truth_and_validation_changes_do_not_reach_planner(monkeypatch):
    cfg=deepcopy(study.CONFIG);cfg['training_points']=8
    first=study.dynamic.make_case(994901,1,cfg);changed=deepcopy(first)
    start=changed['future_start'];changed['p'][start:]+=.3;changed['q'][start:]+=.1
    changed['white']['v'][start:]*=-3
    obs=changed['validation']
    changed['validation']=lk.Observations(p=obs.p+.1,q=obs.q,v=obs.v+.001,
        root_v=obs.root_v,master_p=obs.master_p,master_q=obs.master_q)
    bank=study.declared_bank()
    fitted={c.name:SimpleNamespace(r=np.full(c.m,.004),x=np.full(c.m,.001),
        source_p=np.full(8,.1),source_q=np.full(8,.03),success=True,loss=float(i))
        for i,c in enumerate(bank)}
    rows=[dict(candidate=c.name,name=c.name,success=True,loss=1.,
        r=fitted[c.name].r.tolist(),x=fitted[c.name].x.tolist()) for c in bank]
    block=dict(all_complete=True,fits=rows,failures={},seconds=.01,cost={})
    monkeypatch.setattr(study,'source_hashes',lambda:{'fixture':'constant'})
    monkeypatch.setattr(study.rx,'train_bank',lambda *a,**kw:(fitted,deepcopy(block)))
    monkeypatch.setattr(study.old,'fit_bank',lambda *a,**kw:(fitted,deepcopy(block)))
    calls=[];current=[first]
    def planner(training,candidates,trained,tm,fm,forecast,config,*,seed,counter):
        assert [c.name for c in candidates]==list(study.FAMILIES) and trained is fitted
        assert training is current[0]['train']
        calls.append(study.io.serial(dict(p=training.p,forecast=forecast)))
        u=study.previous.public_curves(6,3)['constant']
        return {name:u.copy() for name in study.METHODS},dict(methods={},timing={name:dict(
            planning_actual_seconds=0.,planning_standalone_seconds=0.,dependency_seconds=0.,
            soft_budget_overrun_seconds=0.,actual_calls={}) for name in study.METHODS})
    monkeypatch.setattr(study,'select_plans',planner)
    monkeypatch.setattr(study,'planning_diagnostics',lambda *a,**kw:dict(success=True))
    monkeypatch.setattr(study.dynamic,'operation_record',lambda *a,**kw:dict(success=True))
    monkeypatch.setattr(study.dynamic,'actual_curve',lambda *a,**kw:dict(status='complete',M=0.,
        noisy=dict(seconds=.01,cost={}),noiseless=dict(seconds=.01)))
    monkeypatch.setattr(study.dynamic,'make_case',lambda *a,**kw:current[0])
    a=study.run_trial(994901,1,cfg);current[0]=changed;b=study.run_trial(994901,1,cfg)
    assert a['status']==b['status']=='complete'
    assert calls[0]==calls[1]
    assert len(a['methods'])==len(b['methods'])==9
    assert [c['name'] for c in a['candidates']]==['star','clade_01']
    assert not a['proposal']['wzzt_used'] and not a['proposal']['candidate_selection_uses_truth']


def test_driver_failure_keeps_declared_pair_and_original_seed(monkeypatch):
    monkeypatch.setattr(study,'source_hashes',lambda:{'fixture':'constant'})
    monkeypatch.setattr(study,'_run_trial',lambda *a,**kw:(_ for _ in ()).throw(ValueError('fixture')))
    record=study.run_trial(994101,0)
    assert record['status']=='driver_failure' and record['seed']==994101
    assert [c['name'] for c in record['candidates']]==['star','clade_01']
    assert record['true_name']=='star' and record['methods']==[]
    with pytest.raises(ValueError,match='two-candidate'):study.run_trial(994101,2)


def test_square_is_actually_solved_and_shared_cost_is_charged(monkeypatch):
    from . import ev_response_metric_design as contrast
    cfg=deepcopy(study.CONFIG);bank=study.declared_bank();T=6
    fitted={c.name:SimpleNamespace(r=np.full(c.m,.004),x=np.full(c.m,.001)) for c in bank}
    training=SimpleNamespace(n=3);called=[]
    def squared(domain,meter,initial_powers,**kw):
        called.append(dict(initial=initial_powers,options=kw))
        return dict(success=True,power_kw=initial_powers[0],Q=float(np.sum(initial_powers[0]**2)))
    monkeypatch.setattr(study,'solve_squared_precision',squared)
    monkeypatch.setattr(study,'load_pressure_scenarios',lambda *a,**kw:([dict(name='nominal',p=[],q=[])],{}))
    monkeypatch.setattr(study.fixed,'build_fixed_contexts',lambda *a,**kw:dict(success=True,diagnostics={}))
    curves=study.previous.public_curves(T,3)
    def select(*args,**kwargs):return curves['analytic_square'].copy(),dict(success=True,selected_origin='square')
    monkeypatch.setattr(study.previous,'fixed_surrogate_policy',select)
    monkeypatch.setattr(study.gradient,'select_gradient_controls',select)
    monkeypatch.setattr(contrast,'select_contrast_controls',select)
    plans,record=study.select_plans(training,bank,fitted,None,None,{},cfg,seed=1,counter={})
    assert len(called)==1 and len(called[0]['initial'])==3
    assert plans['square']==pytest.approx(curves['analytic_square'])
    for name in ('nearest','gradac','contrast_joint','contrast_box','contrast_load'):
        timing=record['timing'][name]
        assert timing['dependencies']==['square'] and timing['charged_ancestors']==['square']
        assert timing['planning_standalone_seconds']==pytest.approx(
            timing['planning_actual_seconds']+timing['dependency_seconds'])
    assert record['line_parameters_unchanged']


def test_pressure_scenarios_are_training_only_deterministic_PQ_and_physical():
    cfg=deepcopy(study.CONFIG);cfg['training_points']=24
    case=study.dynamic.make_case(994991,0,cfg);tr,tm=case['train'],case['tm']
    fits={c.name:SimpleNamespace(source_p=np.full(len(tr.p),.1),source_q=np.full(len(tr.p),.03),success=True)
        for c in study.declared_bank()}
    forecast=study.fit_forecast(tr,tm,fits,horizon=6,gap=4)
    a,meta=study.load_pressure_scenarios(tr,tm,forecast)
    b,again=study.load_pressure_scenarios(tr,tm,forecast)
    assert study.io.serial(a)==study.io.serial(b)
    assert study.io.serial(meta)==study.io.serial(again)
    assert len(a)==meta['scenario_count']==5
    assert a[0]['p']==pytest.approx(forecast['p']) and a[0]['q']==pytest.approx(forecast['q'])
    assert meta['pressure_scale']==pytest.approx(np.sqrt(np.arange(5,11)))
    assert np.linalg.eigvalsh(meta['covariance']).min()>-1e-14
    assert not meta['statistical_coverage_claimed']
    for row in a:
        assert row['metadata']['training_only'] and not row['metadata']['uses_validation']
        pf=row['p']/np.hypot(row['p'],row['q'])
        assert pf.min()>=.94-1e-12 and pf.max()<=.99+1e-12
        assert row['p'].min()>=.05/20.-1e-12 and row['p'].max()<=12./20.+1e-12
        assert row['q'].min()>=0.


def test_pressure_scenarios_preserve_five_rows_when_training_has_no_residual_signal():
    p=np.full((16,3),.12);q=p*np.tan(np.arccos(.97))
    tr=lk.Observations(p=p,q=q,v=np.ones_like(p),root_v=np.ones(len(p)),
        master_p=np.full(len(p),.5),master_q=np.full(len(p),.1))
    tm=SimpleNamespace(bias_p=0.,bias_q=0.)
    fits={c.name:SimpleNamespace(source_p=np.full(16,.1),source_q=np.full(16,.03),success=True)
        for c in study.declared_bank()}
    forecast=study.fit_forecast(tr,tm,fits,horizon=6,gap=4)
    rows,meta=study.load_pressure_scenarios(tr,tm,forecast)
    assert len(rows)==5
    assert max(meta['eigenvalues'])<1e-20
    for row in rows:assert row['p']==pytest.approx(forecast['p'],abs=1e-12)


def test_real_planner_interfaces_include_exactly_five_load_scenarios():
    bank=study.declared_bank();tm=study.dynamic.base.model(3);H=12;T=6
    p=.12+.01*np.cos(np.arange(H)[:,None]+np.arange(3)[None,:])
    q=p*np.tan(np.arccos(.97));sp=np.full(H,.1);sq=np.full(H,.03)
    fits={c.name:SimpleNamespace(r=np.full(c.m,.004),x=np.full(c.m,.001),
        source_p=sp.copy(),source_q=sq.copy(),success=True,loss=1.,p=p.copy(),q=q.copy()) for c in bank}
    phy=lk.ac_predict(bank[0],fits['star'].r,fits['star'].x,p,q,sp,sq,np.ones(H))
    tr=lk.Observations(p=p+tm.bias_p,q=q+tm.bias_q,v=phy['v']+tm.bias_v,
        root_v=np.ones(H),master_p=phy['master_p'],master_q=phy['master_q'])
    fm=replace(tm,v_sigma=np.full((T,3),.00012),master_p_sigma=np.full(T,.001),master_q_sigma=np.full(T,.001))
    fore=study.fit_forecast(tr,tm,fits,horizon=T,gap=4)
    cfg=deepcopy(study.CONFIG);cfg['planning_seconds']=8.
    cfg['exchange'].update(backend='scipy',max_iterations=1,solver_seconds=.3,total_seconds=2.,max_segments=4)
    cfg['gradient'].update(backend='scipy',max_starts=1,max_iterations=1)
    cfg['contrast'].update(backend='scipy',solver_seconds=.3)
    cfg['profile'].update(multistart=1,max_nfev=100)
    plans,record=study.select_plans(tr,bank,fits,tm,fm,fore,cfg,seed=991,counter={})
    assert set(plans)==set(study.METHODS)
    assert all(power is not None for power in plans.values())
    load=record['methods']['contrast_load']
    assert load['success'],load
    assert load['scenario_names']==['nominal','pc1_plus','pc1_minus','pc2_plus','pc2_minus']
    assert record['load_pressure']['scenario_count']==5
    assert record['line_parameters_unchanged']
    diagnostics=study.planning_diagnostics(tr,bank,fits,tm,fm,fore,plans,cfg,seed=991,counter={})
    assert set(diagnostics['methods'])==set(study.METHODS)
    assert len(diagnostics['load_models'])==5
    assert diagnostics['methods']['constant']['contrast_load']['success']
    assert diagnostics['methods']['constant']['future_profile_distance'] is not None
    assert not diagnostics['uses_truth'] and not diagnostics['uses_validation']
    assert study.io.serial(diagnostics)['scope'].startswith('training-model diagnostics')
