"""Electrical-location ablation plumbing and identities; no new AC fit."""
from copy import deepcopy
from dataclasses import fields
import numpy as np
import pytest
from . import ev_original_location_ablation as a
from .likelihood import MeasurementModel,Observations
from .information_context import meter_to_record


def fixture():
    H,T=1,2;meter=MeasurementModel(p_sigma=1.,q_sigma=1.,v_sigma=1.,master_p_sigma=1.,master_q_sigma=1.)
    def obs(times):
        return Observations(p=np.zeros((times,3)),q=np.zeros((times,3)),v=np.ones((times,3)),
            master_p=np.zeros(times),master_q=np.zeros(times),root_v=np.ones(times))
    history=obs(H);future=obs(T);record=lambda o:{f.name:getattr(o,f.name) for f in fields(Observations)}
    config=dict(history=H,horizon=T,base_kw=20.,dt_hours=.25,fit={'multistart':2,'max_nfev':250},
        asset_bounds=dict(r_bounds=[.001,.1],x_bounds=[.001,.1]),
        device=dict(initial_energy_kwh=24.,terminal_energy_kwh=24.8,eta_charge=1.,eta_discharge=1.))
    names=['star','c01','c02','c12'];families=[[],[[0,1]],[[0,2]],[[1,2]]]
    full=obs(H+T)
    fits=[dict(name=n,loss=1. if n=='star' else 3.,success=True,
        r=[.001]* (4 if n=='star' else 5),x=[.001]* (4 if n=='star' else 5),
        p=full.p,q=full.q,source_p=np.zeros(H+T),source_q=np.zeros(H+T),predicted=record(full)) for n in names]
    power1=np.array([[0.,2.,0.],[0.,2.,0.]]);power2=np.array([[0.,0.,2.],[0.,0.,2.]])
    rows=[];methods={}
    for name,power in zip(a.METHODS,(power1,power2)):
        rows.append(dict(method=name,status='complete',power_kw=power,fits=deepcopy(fits),failures={},
            physical={key:getattr(future,key) for key in ('p','q','v','master_p','master_q')},
            observations=record(future),noiseless_AC_refit=dict(fits=deepcopy(fits),failures={},future_mean_observations=record(future))))
        methods[name]=dict(guard={'accepted':True},operation=dict(feasible=True,u_kw=power,
            travel_energy_kwh=.2,driver_cost=30.,energy_kwh=[24.,24.8]))
    truth=dict(family=[],r=[.001]*4,x=[.001]*4,p=np.full((H+T,3),.12),q=np.full((H+T,3),.04),
        source_p=np.full(H+T,.1),source_q=np.full(H+T,.03),root_v=np.ones(H+T),
        white={k:np.zeros((H+T,3) if k in ('p','q','v') else H+T) for k in ('p','q','v','master_p','master_q')})
    return dict(seed=1,phase='evaluation',domain='asset',config=config,seeds={'policy':100},truth=truth,
        history=record(history),historical_fits=fits,historical_failures={},
        proposal={'bank':[dict(name=n,supports=f) for n,f in zip(names,families)]},rows=rows,
        measurement_models={k:meter_to_record(meter) for k in ('hm','fm','jm')},
        selection=dict(methods=methods,failures={}))


def test_signed_waveform_and_grid_energy_unchanged_by_remap():
    original=np.array([[0.,6.,0.],[0.,0.,-3.],[1.,0.,0.],[0.,0.,0.]])
    remap=a.remap_to_site0(original)
    assert remap[:,0]==pytest.approx([6.,-3.,1.,0.])
    assert remap.sum(axis=1)==pytest.approx(original.sum(axis=1))
    assert abs(remap).sum()==pytest.approx(abs(original).sum())
    assert .25*remap.sum()==pytest.approx(.25*original.sum())


def test_no_driving_remap_has_extra_terminal_energy_not_equal_service():
    trial=fixture();scope=a.resource_scope(trial,a.METHODS[0],trial['rows'][0]['power_kw'])
    assert scope['remap_hypothetical_terminal_without_driving_kwh']==pytest.approx(25.)
    assert scope['remap_terminal_excess_over_original_target_kwh']==pytest.approx(.2)
    assert scope['remap_has_no_operation_feasibility_or_equal_service_claim']
    assert scope['original_executed_operation']['driver_cost']==30.


def test_observation_uses_same_saved_white_errors_and_bias_for_both_locations():
    meter=MeasurementModel(1.,2.,3.,4.,5.,bias_p=.1,bias_q=.2,bias_v=.3)
    physical=dict(p=np.zeros((2,3)),q=np.zeros((2,3)),v=np.ones((2,3)),master_p=np.zeros(2),master_q=np.zeros(2),root_v=np.ones(2))
    white={k:np.full_like(v,.25) for k,v in physical.items() if k!='root_v'}
    original=a.observation(physical,meter,white);shifted={**physical,'v':physical['v']+.01}
    remap=a.observation(shifted,meter,white)
    assert remap.v-original.v==pytest.approx(np.full((2,3),.01))
    assert original.p==pytest.approx(np.full((2,3),.35))
    assert original.source_p is None


def test_absent_candidates_keep_four_rows_and_margin_unavailable():
    trial=fixture();bank=a.candidate_bank(trial);names=[c.name for c in bank]
    fits=trial['historical_fits'][:3];result=a.fit_summary(fits,{'c12':'failed'},names,'star')
    assert not result['success'] and result['M'] is None and result['required_candidates']==4
    costs=a.channel_costs(fits,names,None,None,1,{})
    assert len(costs)==4 and costs[-1]['candidate']=='c12' and costs[-1]['future_channel_SSE'] is None


def test_channel_costs_partition_joint_cost_and_historical_penalty():
    trial=fixture();meter=MeasurementModel(1.,1.,1.,1.,1.);future=a.observations_from_record(trial['rows'][0]['observations'])
    fit=deepcopy(trial['historical_fits'][0]);fit['loss']=100.
    for key in ('p','q','v','master_p','master_q'):fit['predicted'][key]=np.asarray(fit['predicted'][key]).copy();fit['predicted'][key][1:]+=1.
    costs=a.channel_costs([fit],['star'],future,meter,1,{'star':{'loss':40.}})[0]
    assert sum(costs['future_channel_SSE'].values())==pytest.approx(22.)
    assert costs['historical_SSE']==pytest.approx(78.) and costs['historical_penalty']==pytest.approx(38.)


def test_stationary_saved_curves_are_reused_without_new_AC_or_fits(monkeypatch):
    trial=fixture()
    for row in trial['rows']:row['power_kw']=a.remap_to_site0(row['power_kw'])
    def forbidden(*args,**kwargs):raise AssertionError('unnecessary AC or fit')
    monkeypatch.setattr(a.lk,'ac_predict',forbidden);monkeypatch.setattr(a.io,'fit_all_candidates',forbidden)
    result=a.process_history(trial)
    assert len(result['conditions'])==2 and all(c['status']=='complete' for c in result['conditions'])
    assert result['counts']==dict(forward_AC_calls=0,candidate_fit_calls=0,fit_AC_calls=0)
    assert all(c['delta_M']==0 for c in result['conditions'])


def test_same_remapped_waveform_across_two_methods_is_computed_once_and_refit_H_N(monkeypatch):
    trial=fixture();captures=[];calls=[]
    def forward(candidate,r,x,p,q,sp,sq,root):
        captures.append((np.asarray(p).copy(),np.asarray(q).copy(),np.asarray(sp).copy(),np.asarray(sq).copy(),np.asarray(root).copy()))
        return dict(p=p,q=q,v=np.ones((2,3)),master_p=np.asarray(p).sum(axis=1)+sp,
            master_q=np.asarray(q).sum(axis=1)+sq,root_v=root,ac_equation_residual=0.,power_balance_residual=0.)
    def fitted(bank,joint,meter,**kwargs):
        calls.append((joint,kwargs));records={r['name']:deepcopy(r) for r in trial['historical_fits']}
        for row in records.values():row['predicted']={k:np.asarray(getattr(joint,k)) for k in ('p','q','v','master_p','master_q')}
        return records,{},dict(fit_calls=4,ac_calls=7,seconds=.001)
    monkeypatch.setattr(a.lk,'ac_predict',forward);monkeypatch.setattr(a.io,'fit_all_candidates',fitted)
    monkeypatch.setattr(a.io,'fit_record',lambda candidate,fit:fit)
    result=a.process_history(trial)
    assert result['counts']==dict(forward_AC_calls=1,candidate_fit_calls=8,fit_AC_calls=14)
    assert len(calls)==2 and all(call[0].t==3 for call in calls)
    assert [call[1]['seed'] for call in calls]==[902,1302]
    assert all(set(call[1]['initial'])=={'star','c01','c02','c12'} for call in calls)
    expected=np.asarray(trial['truth']['p'])[1:].copy();expected[:,0]+=.1
    assert captures[0][0]==pytest.approx(expected)
    assert captures[0][2]==pytest.approx(np.asarray(trial['truth']['source_p'])[1:])
    assert result['conditions'][1]['remap']['noisy']['cache_hit']
    assert all(c['waveform_max_abs_difference_kw']==0 for c in result['conditions'])
    assert all(c['grid_throughput_difference_kwh']==0 and c['meter_noise_audit']['passed'] for c in result['conditions'])


def test_missing_original_action_stays_in_two_condition_denominator(monkeypatch):
    trial=fixture();trial['rows'][0]['power_kw']=None
    trial['rows'][1]['power_kw']=a.remap_to_site0(trial['rows'][1]['power_kw'])
    result=a.process_history(trial)
    assert len(result['conditions'])==2 and result['conditions'][0]['status']=='missing_original_action'
    rows,costs=a.tabulate([result]);assert len(rows)==2 and rows[0]['delta_M'] is None


def test_nonfinite_failed_fit_is_saved_as_missing_cost_without_losing_candidate():
    trial=fixture();fit=deepcopy(trial['historical_fits'][0]);fit.update(loss=None,success=False)
    future=a.observations_from_record(trial['rows'][0]['observations'])
    result=a.channel_costs([fit],['star','c01'],future,None,1,{})
    assert len(result)==2 and result[0]['reason']=='nonfinite_joint_cost'
    assert result[0]['joint_SSE'] is None and not result[0]['success']


def test_nested_nonfinite_arrays_are_preserved_as_null_in_failure_evidence():
    value=a.serial(dict(values=np.array([np.inf,np.nan,1.]),scalar=np.float64(-np.inf)))
    assert value==dict(values=[None,None,1.],scalar=None)
